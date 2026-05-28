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

"""Minimal repro for SparseLoad BWD and IndexAttn BWD.

Usage:
    CUDA_VISIBLE_DEVICES=7 CUDA_HOME=/usr/local/cuda-13.0 MAGI_ATTENTION_FORCE_JIT_BUILD=1 \
        python repro_bwd_sparse.py 2>&1 | tee /tmp/repro_bwd_sparse.log
"""

import datetime
import sys
import traceback

import torch

DEVICE = "cuda"


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def test_sparse_load_bwd():
    """SparseLoad FWD+BWD: small block-sparse with sparse_load=True."""
    from einops import rearrange as einops_rearrange

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_block_mask_triton
    from tests.test_attn.test_block_sparse_attn import TestBlockSparseAttn

    torch.manual_seed(42)

    seqlen = 2048
    dtype = torch.bfloat16
    NHQ, NHK, D = 16, 4, 128
    q_block_size, k_block_size = 64, 64
    ref_block_size = (64, 128)
    block_size = (q_block_size, k_block_size)
    max_seqlen_q = q_block_size

    helper = TestBlockSparseAttn.__new__(TestBlockSparseAttn)
    (
        block_mask,
        block_sizes,
        block_row_sz,
        block_col_sz,
    ) = helper._generate_sparse_pattern(
        test_type="uniform",
        num_heads_q=NHQ,
        num_heads_kv=NHK,
        seqlen=seqlen,
        sparsity_ratio=0.5,
        sparsity_granularity="per_kv_head",
        sparse_format="block_mask",
        block_size=block_size,
    )

    q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
        block_mask, q_block_size, k_block_size
    )
    attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=DEVICE)

    q_raw = torch.randn(
        1, seqlen, NHQ, D, dtype=dtype, device=DEVICE, requires_grad=True
    )
    k_raw = torch.randn(
        1, seqlen, NHK, D, dtype=dtype, device=DEVICE, requires_grad=True
    )
    v_raw = torch.randn(
        1, seqlen, NHK, D, dtype=dtype, device=DEVICE, requires_grad=True
    )

    q = (
        einops_rearrange(q_raw, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k = (
        einops_rearrange(k_raw, "b s h d -> (b h s) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    v = (
        einops_rearrange(v_raw, "b s h d -> (b h s) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    do = torch.randn_like(q)

    print(
        f"  q_ranges.shape={q_ranges.shape}, k_ranges.shape={k_ranges.shape}",
        flush=True,
    )
    print(f"  [FWD] start {now()}", flush=True)
    o, meta = flex_flash_attn_func(
        q=q,
        k=k,
        v=v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        pack_gqa=False,
        swap_ab=False,
        sparse_load=True,
        ref_block_size=ref_block_size,
        swap_bwd_qk_loop=True,
        auto_range_merge=True,
        max_seqlen_q=max_seqlen_q,
    )
    print(f"  [FWD] finish {now()}, o.shape={o.shape}", flush=True)

    print(f"  [BWD] start {now()}", flush=True)
    o.backward(do)
    print(f"  [BWD] finish {now()}", flush=True)
    print(
        f"  dQ.shape={q.grad.shape}, dK.shape={k.grad.shape}, dV.shape={v.grad.shape}"
    )
    print(f"  dQ abs max={q.grad.abs().max().item():.4f}")


def test_index_attn_bwd():
    """IndexAttn FWD+BWD: small index-based attention."""
    from einops import rearrange as einops_rearrange

    from magi_attention.functional import flex_flash_attn_func

    torch.manual_seed(42)

    B, S, NHQ, NHK, D, topk = 1, 256, 4, 4, 64, 128

    indices = torch.full((B * S, NHK, topk), -1, dtype=torch.int32, device=DEVICE)
    for b in range(B):
        for qi in range(S):
            row = b * S + qi
            for h in range(NHK):
                perm = torch.randperm(S, device=DEVICE)[:topk].sort().values
                global_ids = (b * S + perm) * NHK + h
                indices[row, h, :topk] = global_ids.int()

    q_raw = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=DEVICE)
    k_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=DEVICE)
    v_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=DEVICE)

    q_ffa = (
        einops_rearrange(q_raw, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k_ffa = (
        einops_rearrange(k_raw, "b s h d -> (b s h) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    v_ffa = (
        einops_rearrange(v_raw, "b s h d -> (b s h) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )

    print(f"  [FWD] start {now()}", flush=True)
    o, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_attn_indices=indices,
        q_block_size=1,
        k_block_size=1,
    )
    print(f"  [FWD] finish {now()}, o.shape={o.shape}", flush=True)

    do = torch.randn_like(o)
    print(f"  [BWD] start {now()}", flush=True)
    o.backward(do)
    print(f"  [BWD] finish {now()}", flush=True)
    print(f"  dQ.shape={q_ffa.grad.shape}")
    print(f"  dQ abs max={q_ffa.grad.abs().max().item():.4f}")


def test_index_attn_bwd_tma_contiguous():
    """IndexAttn BWD with index_attn_tma_contiguous=True."""
    from einops import rearrange as einops_rearrange

    from magi_attention.functional import flex_flash_attn_func

    torch.manual_seed(42)

    B, S, NHQ, NHK, D, topk = 2, 256, 4, 4, 128, 128

    indices = torch.full((B * S, NHK, topk), -1, dtype=torch.int32, device=DEVICE)
    for b in range(B):
        for qi in range(S):
            row = b * S + qi
            for h in range(NHK):
                start = b * S * NHK + 0 * NHK + h
                global_ids = torch.arange(topk, device=DEVICE) * NHK + start
                indices[row, h, :topk] = global_ids.int()

    q_raw = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=DEVICE)
    k_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=DEVICE)
    v_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=DEVICE)

    q_ffa = (
        einops_rearrange(q_raw, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k_ffa = (
        einops_rearrange(k_raw, "b s h d -> (b s h) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )
    v_ffa = (
        einops_rearrange(v_raw, "b s h d -> (b s h) 1 d")
        .detach()
        .clone()
        .requires_grad_(True)
    )

    print(f"  [FWD] start {now()}", flush=True)
    o, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_attn_indices=indices,
        q_block_size=1,
        k_block_size=1,
        index_attn_tma_contiguous=True,
    )
    print(f"  [FWD] finish {now()}, o.shape={o.shape}", flush=True)

    do = torch.randn_like(o)
    print(f"  [BWD] start {now()}", flush=True)
    o.backward(do)
    print(f"  [BWD] finish {now()}", flush=True)
    print(f"  dQ.shape={q_ffa.grad.shape}")
    print(f"  dQ abs max={q_ffa.grad.abs().max().item():.4f}")


if __name__ == "__main__":
    cases = [
        ("SparseLoad BWD", test_sparse_load_bwd),
        ("IndexAttn BWD", test_index_attn_bwd),
    ]

    print(f"=== repro_bwd_sparse.py start {now()} ===", flush=True)
    results = {}
    for name, fn in cases:
        print(f"\n[case] {name} start {now()}", flush=True)
        try:
            fn()
            results[name] = "PASSED"
            print(f"[case] {name} finish {now()} -> PASSED", flush=True)
        except Exception as e:
            results[name] = f"FAILED: {e}"
            traceback.print_exc()
            print(f"[case] {name} finish {now()} -> FAILED", flush=True)

    print(f"\n=== Summary {now()} ===")
    for name, result in results.items():
        print(f"  {name}: {result}")

    if any("FAILED" in r for r in results.values()):
        sys.exit(1)
