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

"""Benchmark: IntraWGOverlap true vs false for FWD dense attention.

Since SparseLoad/IndexAttn require IntraWGOverlap=true (static_assert),
this benchmark only tests dense causal/full attention.

Usage:
    # Pre-compile both variants first (one-time):
    FFA_INTRA_WG_OVERLAP=true  CUDA_HOME=/usr/local/cuda-13.0 python -c "
import torch; from magi_attention.functional import flex_flash_attn_func
q=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda',requires_grad=True)
k=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
v=torch.randn(256,8,128,dtype=torch.bfloat16,device='cuda')
qr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
kr=torch.tensor([[0,256]],dtype=torch.int32,device='cuda')
am=torch.tensor([1],dtype=torch.int32,device='cuda')
o,_=flex_flash_attn_func(q=q,k=k,v=v,q_ranges=qr,k_ranges=kr,attn_type_map=am)
o.backward(torch.randn_like(o))
print('IntraWGOverlap=true compiled OK')
"
    FFA_INTRA_WG_OVERLAP=false CUDA_HOME=/usr/local/cuda-13.0 python -c "..."

    # Then run this benchmark:
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/bench_intra_wg_overlap.py
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
    intra_wg_overlap: bool,
    fwd_only: bool = False,
    warmup: int = 10,
    iters: int = 25,
) -> dict:
    """Run FWD (+ optional BWD) and return timing stats."""
    device = "cuda"
    dtype = torch.bfloat16

    env_val = "true" if intra_wg_overlap else "false"
    os.environ["FFA_INTRA_WG_OVERLAP"] = env_val

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
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    flops_fwd = 2 * seqlen * seqlen * nhq * head_dim * 2
    total_flops = flops_fwd if fwd_only else flops_fwd * 3.5

    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    tflops_avg = total_flops / avg_ms * 1e-9
    tflops_peak = total_flops / min_ms * 1e-9

    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "tflops_avg": tflops_avg,
        "tflops_peak": tflops_peak,
    }


def main():
    configs = [
        {
            "seqlen": 2048,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": False,
            "label": "causal 2k GQA48/8 FWD+BWD",
        },
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
            "seqlen": 2048,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "fwd_only": True,
            "label": "causal 2k GQA48/8 FWD",
        },
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
    ]

    print(f"{'Config':<35} {'IWG=true avg':<14} {'IWG=false avg':<14} {'speedup':>8}")
    print("-" * 75)

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r_on = bench_one(**cfg, intra_wg_overlap=True)
            r_off = bench_one(**cfg, intra_wg_overlap=False)
            speedup = r_off["avg_ms"] / r_on["avg_ms"]
            print(
                f"{label:<35} {r_on['avg_ms']:>8.2f} ms    {r_off['avg_ms']:>8.2f} ms    {speedup:>7.3f}x"
            )
        except Exception as e:
            print(f"{label:<35} ERROR: {e}")
        cfg["label"] = label

    if "FFA_INTRA_WG_OVERLAP" in os.environ:
        del os.environ["FFA_INTRA_WG_OVERLAP"]


if __name__ == "__main__":
    main()
