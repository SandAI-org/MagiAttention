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

"""Benchmark: inner-loop direction MaxToMin vs MinToMax for FWD and BWD.

Toggles MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN between true/false and compares performance.

Usage:
    # Pre-compile both variants first (one-time):
    MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN=true  CUDA_HOME=/usr/local/cuda-13.0 python -c "
import torch; from magi_attention.functional import flex_flash_attn_func
q=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda',requires_grad=True)
k=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
v=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
qr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
kr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
am=torch.tensor([1],dtype=torch.int32,device='cuda')
o,_=flex_flash_attn_func(q=q,k=k,v=v,q_ranges=qr,k_ranges=kr,attn_type_map=am)
o.backward(torch.randn_like(o))
print('MaxToMin compiled OK')
"
    MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN=false CUDA_HOME=/usr/local/cuda-13.0 python -c "..."

    # Then run this benchmark:
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/bench_inner_dir.py
"""

import os

import torch

from magi_attention.functional import flex_flash_attn_func


def bench_one(
    seqlen: int,
    nhq: int,
    nhk: int,
    head_dim: int,
    attn_type: int,
    inner_dir_max_to_min: bool,
    fwd_only: bool = False,
    warmup: int = 25,
    iters: int = 100,
) -> dict:
    """Run FWD (+ optional BWD) and return timing stats."""
    device = "cuda"
    dtype = torch.bfloat16

    env_val = "true" if inner_dir_max_to_min else "false"
    os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"] = env_val

    from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_mod

    if hasattr(get_ffa_jit_mod, "cache_clear"):
        get_ffa_jit_mod.cache_clear()

    q = torch.randn(
        seqlen, nhq, head_dim, dtype=dtype, device=device, requires_grad=not fwd_only
    )
    k = torch.randn(
        seqlen, nhk, head_dim, dtype=dtype, device=device, requires_grad=not fwd_only
    )
    v = torch.randn(
        seqlen, nhk, head_dim, dtype=dtype, device=device, requires_grad=not fwd_only
    )
    do = (
        torch.randn(seqlen, nhq, head_dim, dtype=dtype, device=device)
        if not fwd_only
        else None
    )

    q_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([attn_type], dtype=torch.int32, device=device)

    l2_cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)

    def run():
        if not fwd_only:
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
        if not fwd_only:
            o.backward(do)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        l2_cache.zero_()
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    times_sorted = sorted(times_ms)
    median_ms = times_sorted[len(times_sorted) // 2]
    min_ms = times_sorted[0]

    flops_fwd = 2 * seqlen * seqlen * nhq * head_dim * 2
    total_flops = flops_fwd if fwd_only else flops_fwd * 3.5

    tflops_median = total_flops / median_ms * 1e-9
    tflops_peak = total_flops / min_ms * 1e-9

    return {
        "median_ms": median_ms,
        "min_ms": min_ms,
        "tflops_median": tflops_median,
        "tflops_peak": tflops_peak,
    }


def main():
    n_repeats = int(os.environ.get("BENCH_REPEATS", "1"))

    configs = [
        # FWD-only
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": True,
            "label": "causal 8k GQA48/8 FWD",
        },
        {
            "seqlen": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": True,
            "label": "causal 8k MHA8 FWD",
        },
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 0,
            "fwd_only": True,
            "label": "full 8k GQA48/8 FWD",
        },
        {
            "seqlen": 16384,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": True,
            "label": "causal 16k GQA48/8 FWD",
        },
        # FWD+BWD
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 8k GQA48/8 FWD+BWD",
        },
        {
            "seqlen": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 8k MHA8 FWD+BWD",
        },
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 0,
            "fwd_only": False,
            "label": "full 8k GQA48/8 FWD+BWD",
        },
        {
            "seqlen": 16384,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 16k GQA48/8 FWD+BWD",
        },
        # head_dim=64
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 64,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 8k GQA48/8 hd64 FWD+BWD",
        },
        # head_dim=256
        {
            "seqlen": 4096,
            "nhq": 16,
            "nhk": 4,
            "head_dim": 256,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 4k GQA16/4 hd256 FWD+BWD",
        },
    ]

    print(f"{'Config':<40} {'Max2Min P50':<14} {'Min2Max P50':<14} {'speedup':>8}")
    print("-" * 80)

    for cfg in configs:
        label = cfg.pop("label")
        try:
            speedups = []
            r_m2m_last = None
            r_m2M_last = None
            for _ in range(n_repeats):
                r_m2m = bench_one(**cfg, inner_dir_max_to_min=True)
                r_m2M = bench_one(**cfg, inner_dir_max_to_min=False)
                speedups.append(r_m2M["median_ms"] / r_m2m["median_ms"])
                r_m2m_last = r_m2m
                r_m2M_last = r_m2M
            avg_speedup = sum(speedups) / len(speedups)
            if n_repeats > 1:
                sp_str = " ".join(f"{s:.3f}" for s in speedups)
                print(
                    f"{label:<40} {r_m2m_last['median_ms']:>8.2f} ms"
                    f"    {r_m2M_last['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x  [{sp_str}]"
                )
            else:
                print(
                    f"{label:<40} {r_m2m_last['median_ms']:>8.2f} ms"
                    f"    {r_m2M_last['median_ms']:>8.2f} ms"
                    f"    {avg_speedup:>7.3f}x"
                )
        except Exception as e:
            print(f"{label:<40} ERROR: {e}")
        cfg["label"] = label

    if "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN" in os.environ:
        del os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"]


if __name__ == "__main__":
    main()
