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

"""NCU profiling script: FFA Dense BWD vs FFA IndexAttn BWD.

Usage:
    # Full profile (kernel-level metrics)
    ncu --set full -o /tmp/bwd_profile python exps/attn/ncu_bwd_profile.py --mode dense_loopq
    ncu --set full -o /tmp/bwd_idx_profile python exps/attn/ncu_bwd_profile.py --mode index_attn

    # Quick summary (SM throughput, memory, occupancy)
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    python exps/attn/ncu_bwd_profile.py --mode dense_loopq
"""

import argparse
import sys

import torch

sys.path.insert(0, "exps/attn")
from baselines.attn_impl import ffa_func  # noqa: E402


def profile_dense_bwd(S, nhq, nhk, hd, swap_loop=False):
    """Dense BWD with optional loop swap."""
    device, dtype = "cuda", torch.bfloat16
    q = torch.randn(S, nhq, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(S, nhk, hd, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(S, nhk, hd, device=device, dtype=dtype, requires_grad=True)
    q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    o, *_ = ffa_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        swap_bwd_qk_loop=swap_loop,
    )
    do = torch.randn_like(o)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(2):
        o.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()

    # Profiled iteration
    torch.cuda.cudart().cudaProfilerStart()
    o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    label = "LoopK (swapped)" if swap_loop else "LoopQ (default)"
    print(f"Dense BWD {label} profiled at S={S}")


def profile_index_attn_bwd(S, nhq, nhk, hd, topk):
    """IndexAttn BWD."""
    device, dtype = "cuda", torch.bfloat16
    total_q = S * nhk
    q = torch.randn(total_q, nhq, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_q, 1, hd, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_q, 1, hd, device=device, dtype=dtype, requires_grad=True)
    idx = torch.randint(
        0, total_q, (total_q, nhk, topk), device=device, dtype=torch.int32
    )
    idx = idx.sort(dim=-1).values

    o, *_ = ffa_func(
        q,
        k,
        v,
        index_attn_indices=idx,
        q_block_size=1,
        k_block_size=1,
        pack_gqa=True,
    )
    do = torch.randn_like(o)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(2):
        o.backward(do, retain_graph=True)
        q.grad = None
        k.grad = None
        v.grad = None
    torch.cuda.synchronize()

    # Profiled iteration
    torch.cuda.cudart().cudaProfilerStart()
    o.backward(do, retain_graph=True)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"IndexAttn BWD profiled at S={S}, topk={topk}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["dense_loopq", "dense_loopk", "index_attn"], required=True
    )
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=2048)
    args = parser.parse_args()

    nhq, nhk, hd = 128, 1, 128

    if args.mode == "dense_loopq":
        profile_dense_bwd(args.seqlen, nhq, nhk, hd, swap_loop=False)
    elif args.mode == "dense_loopk":
        profile_dense_bwd(args.seqlen, nhq, nhk, hd, swap_loop=True)
    elif args.mode == "index_attn":
        profile_index_attn_bwd(args.seqlen, nhq, nhk, hd, args.topk)
