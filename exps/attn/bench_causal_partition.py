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
    FFA_USE_MASK_DISPATCH=true  CUDA_HOME=/usr/local/cuda-13.0 python -c "
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
    FFA_USE_MASK_DISPATCH=false CUDA_HOME=/usr/local/cuda-13.0 python -c "..."

    # Then run this benchmark:
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/bench_causal_partition.py
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
    use_mask_dispatch: bool,
    warmup: int = 10,
    iters: int = 25,
) -> dict:
    """Run FWD+BWD and return timing stats."""
    device = "cuda"
    dtype = torch.bfloat16

    env_val = "true" if use_mask_dispatch else "false"
    os.environ["FFA_USE_MASK_DISPATCH"] = env_val

    # Clear JIT mod cache to pick up env var change
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

    q_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, seqlen]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([attn_type], dtype=torch.int32, device=device)

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

    # Timed
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        run()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    # Compute FLOPs (FWD + BWD = 4 * seqlen^2 * nhq * head_dim for MHA)
    flops_fwd = 2 * seqlen * seqlen * nhq * head_dim * 2  # Q@K + P@V
    flops_bwd = flops_fwd * 2.5  # BWD ~2.5x FWD
    total_flops = flops_fwd + flops_bwd

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
            "label": "causal 2k GQA48/8",
        },
        {
            "seqlen": 8192,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 8k GQA48/8",
        },
        {
            "seqlen": 32768,
            "nhq": 48,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 32k GQA48/8",
        },
        {
            "seqlen": 8192,
            "nhq": 8,
            "nhk": 8,
            "head_dim": 128,
            "attn_type": 1,
            "label": "causal 8k MHA8",
        },
    ]

    print(f"{'Config':<30} {'UMD=true avg':<14} {'UMD=false avg':<14} {'speedup':>8}")
    print("-" * 70)

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r_on = bench_one(**cfg, use_mask_dispatch=True)
            r_off = bench_one(**cfg, use_mask_dispatch=False)
            speedup = r_off["avg_ms"] / r_on["avg_ms"]
            print(
                f"{label:<30} {r_on['avg_ms']:>8.2f} ms    {r_off['avg_ms']:>8.2f} ms    {speedup:>7.3f}x"
            )
        except Exception as e:
            print(f"{label:<30} ERROR: {e}")
        cfg["label"] = label

    # Cleanup env
    if "FFA_USE_MASK_DISPATCH" in os.environ:
        del os.environ["FFA_USE_MASK_DISPATCH"]


if __name__ == "__main__":
    main()
