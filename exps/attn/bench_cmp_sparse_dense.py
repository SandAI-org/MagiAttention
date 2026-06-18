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

"""
Framework overhead isolation: Dense vs SparseLoad vs IndexAttn
with IDENTICAL effective compute.

Config: nhq=128, nhk=1 (MQA), hd=128, PackGQA, bf16
        seqlen=32K, topk=seqlen=32K, k_block_size=128
        → every Q attends to ALL KV → same FLOPs as Dense

FWD:  Dense, SparseLoad, IndexAttn
BWD:  Dense-LoopQ, Dense-LoopK, SparseLoad-LoopQ, SparseLoad-LoopK,
      IndexAttn-LoopK, IndexAttn-LoopQ (kbs=128)

Usage:
  cd /home/niubility2/cenzhiyao/MagiAttention-build
  CUDA_VISIBLE_DEVICES=1 python exps/attn/bench_cmp_sparse_dense.py
  CUDA_VISIBLE_DEVICES=1 python exps/attn/bench_cmp_sparse_dense.py --bwd
  CUDA_VISIBLE_DEVICES=1 python exps/attn/bench_cmp_sparse_dense.py --bwd-only
"""

import argparse

import torch
from baselines.attn_impl import ffa_func
from baselines.utils import seed_everything
from einops import rearrange
from triton.testing import do_bench

from magi_attention.utils.sparse_utils import generate_ranges_from_block_mask_triton

nhq = 128
nhk = 1
hd = 128
q_block_size = 1
k_block_size = 128
dtype = torch.bfloat16

seed_everything()


def _build_ia_tensors(S, device, requires_grad=False):
    """Build IndexAttn tensors with topk=S (arange indices)."""
    q = torch.randn(1, S, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(1, S, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(1, S, nhk, hd, device=device, dtype=dtype)
    local_pos = torch.arange(S, device=device).unsqueeze(0).expand(S, -1)
    h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
    index_attn_indices = (local_pos.unsqueeze(1) * nhk + h_offsets).int()
    q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
    k_t = rearrange(k, "b s h d -> (b s h) 1 d")
    v_t = rearrange(v, "b s h d -> (b s h) 1 d")
    if requires_grad:
        q_t.requires_grad_(True)
        k_t.requires_grad_(True)
        v_t.requires_grad_(True)
    return q_t, k_t, v_t, index_attn_indices


def bench_dense(S, direction="fwd", swap_bwd_qk_loop=False, warmup=25, rep=100):
    """Dense full-attention baseline with MQA 128/1."""
    device = torch.cuda.current_device()
    fwd_flops = 4 * S * S * nhq * hd

    q = torch.randn(S, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(S, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(S, nhk, hd, device=device, dtype=dtype)
    q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([0], dtype=torch.int32, device=device)

    if direction == "fwd":

        def fn():
            return ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
            )

        target_flops = fwd_flops
    else:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out, _ = ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )
        do = torch.randn_like(out)

        def fn():
            out.backward(do, retain_graph=True)

        target_flops = fwd_flops * 2.5

    ms = do_bench(fn, warmup=warmup, rep=rep)
    return target_flops / ms * 1e-9


def bench_sparse_load(S, direction="fwd", swap_bwd_qk_loop=False, warmup=25, rep=100):
    """SparseLoad with effective_kv=S (all-True block mask)."""
    device = torch.cuda.current_device()
    n_k_blocks = S // k_block_size
    fwd_flops = 4 * S * S * nhq * hd

    block_mask = torch.ones(
        1, nhk, S // q_block_size, n_k_blocks, dtype=torch.bool, device=device
    )
    q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
        block_mask, q_block_size, k_block_size
    )
    attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

    q = torch.randn(S, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(S, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(S, nhk, hd, device=device, dtype=dtype)

    if direction == "fwd":

        def fn():
            return ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                sparse_load=True,
                auto_range_merge=True,
                pack_gqa=True,
            )

        target_flops = fwd_flops
    else:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out, _ = ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sparse_load=True,
            auto_range_merge=True,
            pack_gqa=True,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )
        do = torch.randn_like(out)

        def fn():
            out.backward(do, retain_graph=True)

        target_flops = fwd_flops * 2.5

    ms = do_bench(fn, warmup=warmup, rep=rep)
    return target_flops / ms * 1e-9


def bench_index_attn(
    S, direction="fwd", swap_bwd_qk_loop=True, warmup=25, rep=100, use_kbs128=False
):
    """IndexAttn with topk=S (arange indices).

    use_kbs128: if True, use k_block_size=128 and LoopQ BWD (requires env vars).
    """
    import os

    device = torch.cuda.current_device()
    topk = S
    fwd_flops = 4 * S * topk * nhq * hd
    kbs = 128 if use_kbs128 else 1

    if use_kbs128 and direction == "bwd":
        os.environ["MAGI_ATTENTION_INDEX_ATTN_BWD_LOOP_Q"] = "1"
        os.environ["MAGI_ATTENTION_INDEX_ATTN_BWD_K_BLOCK_SIZE"] = "128"

    q_t, k_t, v_t, index_attn_indices = _build_ia_tensors(
        S, device, requires_grad=(direction == "bwd")
    )

    if use_kbs128:
        num_kv_blocks = S // k_block_size
        ia_indices_kbs128 = torch.arange(
            num_kv_blocks, device=device, dtype=torch.int32
        )
        ia_indices_kbs128 = ia_indices_kbs128.unsqueeze(0).expand(S, -1).contiguous()
        ia_indices_kbs128 = (
            ia_indices_kbs128.unsqueeze(1) * nhk
            + torch.arange(nhk, device=device).view(1, -1, 1)
        ).int()
        indices_to_use = ia_indices_kbs128
    else:
        indices_to_use = index_attn_indices

    if direction == "fwd":

        def fn():
            return ffa_func(
                q_t,
                k_t,
                v_t,
                index_attn_indices=indices_to_use,
                q_block_size=1,
                k_block_size=kbs,
                pack_gqa=True,
            )

        target_flops = fwd_flops
    else:
        loop = False if use_kbs128 else swap_bwd_qk_loop
        out, _ = ffa_func(
            q_t,
            k_t,
            v_t,
            index_attn_indices=indices_to_use,
            q_block_size=1,
            k_block_size=kbs,
            pack_gqa=True,
            swap_bwd_qk_loop=loop,
        )
        do = torch.randn_like(out)

        def fn():
            out.backward(do, retain_graph=True)

        target_flops = fwd_flops * 2.5

    ms = do_bench(fn, warmup=warmup, rep=rep)

    if use_kbs128 and direction == "bwd":
        os.environ.pop("MAGI_ATTENTION_INDEX_ATTN_BWD_LOOP_Q", None)
        os.environ.pop("MAGI_ATTENTION_INDEX_ATTN_BWD_K_BLOCK_SIZE", None)

    return target_flops / ms * 1e-9


def run_one(label, func, *args, **kwargs):
    try:
        t = func(*args, **kwargs)
        print(f"{label:<25} {t:>10.1f}")
        return t
    except Exception as e:
        print(f"{label:<25} {'ERR':>10}  ({e})")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Dense vs Sparse framework overhead bench"
    )
    parser.add_argument("--seqlen", type=int, default=32768)
    parser.add_argument("--bwd", action="store_true", help="Also run BWD benchmark")
    parser.add_argument("--bwd-only", action="store_true", help="Only BWD")
    parser.add_argument("--fwd-only", action="store_true", help="Only FWD")
    args = parser.parse_args()

    S = args.seqlen
    run_fwd = not args.bwd_only
    run_bwd = args.bwd or args.bwd_only

    print("=== Dense vs SparseLoad vs IndexAttn — Framework Overhead Isolation ===")
    print(f"Config: nhq={nhq}, nhk={nhk}, hd={hd}, seqlen={S}, topk=seqlen={S}")
    print(f"        k_block_size={k_block_size}, PackGQA=True, dtype=bf16")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    if run_fwd:
        print("--- FWD ---")
        print(f"{'Method':<25} {'TFLOPS':>10}")
        print("-" * 37)
        run_one("Dense", bench_dense, S, "fwd")
        run_one("SparseLoad", bench_sparse_load, S, "fwd")
        run_one("IndexAttn", bench_index_attn, S, "fwd")
        print()

    if run_bwd:
        print("--- BWD ---")
        print(f"{'Method':<25} {'TFLOPS':>10}")
        print("-" * 37)
        run_one("Dense LoopQ", bench_dense, S, "bwd", swap_bwd_qk_loop=False)
        run_one("Dense LoopK", bench_dense, S, "bwd", swap_bwd_qk_loop=True)
        run_one("SparseLoad LoopQ", bench_sparse_load, S, "bwd", swap_bwd_qk_loop=False)
        run_one("SparseLoad LoopK", bench_sparse_load, S, "bwd", swap_bwd_qk_loop=True)
        run_one("IndexAttn LoopK", bench_index_attn, S, "bwd", swap_bwd_qk_loop=True)
        run_one(
            "IndexAttn LoopQ kbs128",
            bench_index_attn,
            S,
            "bwd",
            swap_bwd_qk_loop=False,
            use_kbs128=True,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
