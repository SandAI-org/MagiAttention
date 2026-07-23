#!/usr/bin/env python3

# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark: UseMaskDispatch vs original mask loop for causal BWD.

Usage:
    # Pre-compile both variants first (one-time):
    MAGI_ATTENTION_FFA_USE_MASK_DISPATCH=true  CUDA_HOME=/usr/local/cuda-13.0 python -c "
import torch; from magi_attention.functional import flex_flash_attn_func
q=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda',requires_grad=True)
k=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
v=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
qr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
kr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
am=torch.tensor([1],dtype=torch.int32,device='cuda')
o,_=flex_flash_attn_func(q=q,k=k,v=v,q_ranges=qr,k_ranges=kr,attn_type_map=am)
o.backward(torch.randn_like(o))
print('UseMaskDispatch=true compiled OK')
"
    MAGI_ATTENTION_FFA_USE_MASK_DISPATCH=false CUDA_HOME=/usr/local/cuda-13.0 python -c "..."

    # Then run this benchmark:
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/ablation/bench_causal_partition.py
"""

import os

import torch

from magi_attention.env import ffa as ffa_env
from magi_attention.functional import flex_flash_attn_func


def bench_one(
    seqlen: int,
    nhq: int,
    nhk: int,
    head_dim: int,
    attn_type: int,
    use_mask_dispatch: bool,
    warmup: int = 25,
    iters: int = 100,
    q_ranges_t: "torch.Tensor | None" = None,
    k_ranges_t: "torch.Tensor | None" = None,
    attn_type_map_t: "torch.Tensor | None" = None,
) -> dict:
    """Run FWD+BWD and return timing stats (median, following do_bench convention)."""
    device = "cuda"
    dtype = torch.bfloat16

    env_val = "true" if use_mask_dispatch else "false"
    os.environ[ffa_env.USE_MASK_DISPATCH] = env_val

    from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

    if hasattr(get_ffa_jit_mod, "cache_clear"):
        get_ffa_jit_mod.cache_clear()

    q = torch.randn(
        seqlen, nhq, head_dim, dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        seqlen, nhk, head_dim, dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        seqlen, nhk, head_dim, dtype=dtype, device=device, requires_grad=True
    )
    do = torch.randn(seqlen, nhq, head_dim, dtype=dtype, device=device)

    if q_ranges_t is None:
        q_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
        attn_type_map = torch.tensor([attn_type], dtype=torch.int32, device=device)
    else:
        assert q_ranges_t is not None
        assert k_ranges_t is not None
        assert attn_type_map_t is not None
        q_ranges = q_ranges_t.to(device)
        k_ranges = k_ranges_t.to(device)
        attn_type_map = attn_type_map_t.to(device)

    # L2 flush buffer (256MB), same as magi_attention.benchmarking.do_bench
    l2_cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)

    def run():
        q.grad = None
        k.grad = None
        v.grad = None
        o, _ = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
        )
        o.backward(do)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Timed iterations with L2 flush between each
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        l2_cache.zero_()
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Use median (P50) as primary metric, following do_bench convention
    times_sorted = sorted(times_ms)
    median_ms = times_sorted[len(times_sorted) // 2]
    min_ms = times_sorted[0]

    # Compute FLOPs (FWD + BWD = 4 * seqlen^2 * nhq * head_dim for MHA)
    flops_fwd = 2 * seqlen * seqlen * nhq * head_dim * 2  # Q@K + P@V
    flops_bwd = flops_fwd * 2.5  # BWD ~2.5x FWD
    total_flops = flops_fwd + flops_bwd

    tflops_median = total_flops / median_ms * 1e-9
    tflops_peak = total_flops / min_ms * 1e-9

    return {
        "median_ms": median_ms,
        "min_ms": min_ms,
        "tflops_median": tflops_median,
        "tflops_peak": tflops_peak,
    }


def main():
    configs = [
        # -- GQA baselines (expected: ~no diff) --
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 8k GQA48/8",
        },
        # -- MHA configs --
        {
            "seqlen": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 8k MHA8",
        },
        {
            "seqlen": 8192,
            "nhq": 16,
            "nhk": 16,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 8k MHA16",
        },
        # -- BiCausal MHA --
        {
            "seqlen": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 3,
            "label": "bicausal 8k MHA8",
        },
    ]

    n_repeats = int(os.environ.get("BENCH_REPEATS", "1"))

    print(
        f"{'Config':<35} {'UMD=true P50':<14} {'UMD=false P50':<14} {'speedup':>8} {'TFLOPS(on)':>11}"
    )
    print("-" * 90)

    for cfg in configs:
        label = cfg.pop("label")
        try:
            speedups = []
            r_on_last = None
            r_off_last = None
            for _ in range(n_repeats):
                r_on = bench_one(**cfg, use_mask_dispatch=True)
                r_off = bench_one(**cfg, use_mask_dispatch=False)
                speedups.append(r_off["median_ms"] / r_on["median_ms"])
                r_on_last = r_on
                r_off_last = r_off
            avg_speedup = sum(speedups) / len(speedups)
            tflops = r_on_last["tflops_median"]
            if n_repeats > 1:
                sp_str = " ".join(f"{s:.3f}" for s in speedups)
                print(
                    f"{label:<35} {r_on_last['median_ms']:>8.2f} ms"
                    f"    {r_off_last['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x  {tflops:>8.1f}  [{sp_str}]"
                )
            else:
                print(
                    f"{label:<35} {r_on_last['median_ms']:>8.2f} ms"
                    f"    {r_off_last['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x  {tflops:>8.1f}"
                )
        except Exception as e:
            print(f"{label:<35} ERROR: {e}")
        cfg["label"] = label

    # -- Varlen multi-segment mixed mask --
    print("\n--- Varlen Multi-Segment Mixed Mask ---")
    print(f"{'Config':<35} {'UMD=true P50':<14} {'UMD=false P50':<14} {'speedup':>8}")
    print("-" * 75)

    import random

    random.seed(42)

    varlen_configs = [
        {
            "n_segs": 10,
            "seg_len": 1024,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "label": "10x1k mixed MHA8",
        },
        {
            "n_segs": 16,
            "seg_len": 512,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "label": "16x512 mixed MHA8",
        },
        {
            "n_segs": 8,
            "seg_len": 2048,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "label": "8x2k mixed MHA8",
        },
        {
            "n_segs": 10,
            "seg_len": 1024,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "label": "10x1k mixed GQA48/8",
        },
    ]

    for vcfg in varlen_configs:
        label = vcfg["label"]
        n_segs = vcfg["n_segs"]
        seg_len = vcfg["seg_len"]
        total_seqlen = n_segs * seg_len

        ranges_list = []
        attn_types = []
        offset = 0
        for _ in range(n_segs):
            ranges_list.append([offset, offset + seg_len])
            attn_types.append(random.choice([0, 1, 2]))  # Full/Causal/InvCausal
            offset += seg_len

        q_ranges_t = torch.tensor(ranges_list, dtype=torch.int32)
        k_ranges_t = torch.tensor(ranges_list, dtype=torch.int32)
        attn_type_map_t = torch.tensor(attn_types, dtype=torch.int32)

        try:
            speedups = []
            for _ in range(n_repeats):
                r_on = bench_one(
                    seqlen=total_seqlen,
                    nhq=vcfg["nhq"],
                    nhk=vcfg["nhk"],
                    head_dim=vcfg["head_dim"],
                    attn_type=0,
                    use_mask_dispatch=True,
                    q_ranges_t=q_ranges_t,
                    k_ranges_t=k_ranges_t,
                    attn_type_map_t=attn_type_map_t,
                )
                r_off = bench_one(
                    seqlen=total_seqlen,
                    nhq=vcfg["nhq"],
                    nhk=vcfg["nhk"],
                    head_dim=vcfg["head_dim"],
                    attn_type=0,
                    use_mask_dispatch=False,
                    q_ranges_t=q_ranges_t,
                    k_ranges_t=k_ranges_t,
                    attn_type_map_t=attn_type_map_t,
                )
                speedups.append(r_off["median_ms"] / r_on["median_ms"])
            avg_speedup = sum(speedups) / len(speedups)
            if n_repeats > 1:
                sp_str = " ".join(f"{s:.3f}" for s in speedups)
                print(
                    f"{label:<35} {r_on['median_ms']:>8.2f} ms"
                    f"    {r_off['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x  [{sp_str}]"
                )
            else:
                print(
                    f"{label:<35} {r_on['median_ms']:>8.2f} ms"
                    f"    {r_off['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x"
                )
        except Exception as e:
            print(f"{label:<35} ERROR: {e}")

    # -- seqlen_q != seqlen_k --
    print("\n--- seqlen_q != seqlen_k (Cross-Attention Style) ---")
    print(f"{'Config':<35} {'UMD=true P50':<14} {'UMD=false P50':<14} {'speedup':>8}")
    print("-" * 75)

    cross_configs = [
        {
            "q_len": 4096,
            "k_len": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal q4k/k8k MHA8",
        },
        {
            "q_len": 2048,
            "k_len": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal q2k/k8k MHA8",
        },
    ]

    for xcfg in cross_configs:
        label = xcfg["label"]
        q_len = xcfg["q_len"]
        k_len = xcfg["k_len"]
        total_seqlen = max(q_len, k_len)

        q_ranges_t = torch.tensor([[0, q_len]], dtype=torch.int32)
        k_ranges_t = torch.tensor([[0, k_len]], dtype=torch.int32)
        attn_type_map_t = torch.tensor([xcfg["attn_type"]], dtype=torch.int32)

        try:
            speedups = []
            for _ in range(n_repeats):
                r_on = bench_one(
                    seqlen=total_seqlen,
                    nhq=xcfg["nhq"],
                    nhk=xcfg["nhk"],
                    head_dim=xcfg["head_dim"],
                    attn_type=xcfg["attn_type"],
                    use_mask_dispatch=True,
                    q_ranges_t=q_ranges_t,
                    k_ranges_t=k_ranges_t,
                    attn_type_map_t=attn_type_map_t,
                )
                r_off = bench_one(
                    seqlen=total_seqlen,
                    nhq=xcfg["nhq"],
                    nhk=xcfg["nhk"],
                    head_dim=xcfg["head_dim"],
                    attn_type=xcfg["attn_type"],
                    use_mask_dispatch=False,
                    q_ranges_t=q_ranges_t,
                    k_ranges_t=k_ranges_t,
                    attn_type_map_t=attn_type_map_t,
                )
                speedups.append(r_off["median_ms"] / r_on["median_ms"])
            avg_speedup = sum(speedups) / len(speedups)
            if n_repeats > 1:
                sp_str = " ".join(f"{s:.3f}" for s in speedups)
                print(
                    f"{label:<35} {r_on['median_ms']:>8.2f} ms"
                    f"    {r_off['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x  [{sp_str}]"
                )
            else:
                print(
                    f"{label:<35} {r_on['median_ms']:>8.2f} ms"
                    f"    {r_off['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x"
                )
        except Exception as e:
            print(f"{label:<35} ERROR: {e}")

    if "MAGI_ATTENTION_FFA_USE_MASK_DISPATCH" in os.environ:
        del os.environ[ffa_env.USE_MASK_DISPATCH]


if __name__ == "__main__":
    main()
