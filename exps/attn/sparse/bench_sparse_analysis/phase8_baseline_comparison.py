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

"""Phase 8: Baseline Comparison — FFA IndexSparse/BlockSparse vs external kernels.

Two sub-benchmarks organized by K block size:

  8a) kbs=1 (token-sparse):
      FFA IndexSparse vs approximate block-sparse FlexAttention vs Triton
      Passes: FWD, BWD_LoopK, BWD_LoopQ

  8b) kbs=128 (block-sparse):
      FFA BlockSparse vs FFA IndexSparse(kbs=128) vs FlexAttention vs Triton Token-Sparse
      Passes: FWD, BWD_LoopK, BWD_LoopQ

Config: nhq=128, nhk=1, hd=128, video-production scenario
        (qseqlen=kvseqlen/64, topk=kvseqlen/8), sweep kvseqlen=[32K..512K].
"""

import gc
import os
import sys

from bench_sparse_analysis._common import (
    COLOR_BLOCK_SPARSE,
    COLOR_FLEXATTN,
    COLOR_INDEX_SPARSE,
    COLOR_TRITON,
    HD,
    NHK,
    NHQ,
    PLOT_DPI_SAVE,
    PLOT_SUBPLOT_HEIGHT,
    PLOT_SUBPLOT_WIDTH,
    VIDEO_SCENARIOS,
    _bench_kernel,
    _load_results,
    _out_dir,
    _save_results,
    _set_gpu,
    _ts,
    plot_grouped_bars,
)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

PHASE = "8-baseline-comparison"
KBS_BLOCK = 128

# Use shared video-production scenarios from _common
SCENARIOS = VIDEO_SCENARIOS

SPARSITY_SEED = 42

# Pass-specific matrices avoid invalid and duplicate method/pass products.
KBS1_PASS_METHODS = {
    "fwd": ["ffa_is", "flexattn", "triton"],
    "bwd_loopq": ["flexattn", "triton"],
    "bwd_loopk": ["ffa_is", "flexattn", "triton"],
}
KBS1_LABELS = {
    "ffa_is": "FFA IndexSparse (kbs=1)",
    "flexattn": "FlexAttention",
    "triton": "Triton",
}
KBS128_PASS_METHODS = {
    "fwd": ["ffa_bs", "ffa_is128", "flexattn", "triton"],
    "bwd_loopq": ["ffa_bs", "ffa_is128", "flexattn", "triton"],
    "bwd_loopk": ["ffa_bs", "ffa_is128", "flexattn", "triton"],
}
KBS128_LABELS = {
    "ffa_bs": "FFA BlockSparse",
    "ffa_is128": "FFA IndexSparse (kbs=128)",
    "flexattn": "FlexAttention",
    "triton": "Triton",
}

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _calc_sparse_flops(qseqlen, topk, is_bwd):
    fwd = 4 * qseqlen * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_unique_indices(num_rows, population, count, device, seed=SPARSITY_SEED):
    """Return sorted unique indices without a rows-by-population random matrix.

    Phase8 populations are powers of two.  Each row takes a prefix of an
    odd-stride permutation modulo ``population``, so all selected items are
    unique.  Only the required rows-by-count index payload is materialized.
    """
    import torch

    assert population > 1 and population & (population - 1) == 0
    assert 0 < count <= population
    mask = population - 1
    half_mask = population // 2 - 1
    rows = torch.arange(num_rows, dtype=torch.int32, device=device)
    offsets = (rows * 65537 + seed * 17) & mask
    strides = (((rows * 8191 + seed * 31) & half_mask) * 2 + 1).to(torch.int32)
    indices = (
        torch.arange(count, dtype=torch.int32, device=device)
        .expand(num_rows, -1)
        .clone()
    )
    indices.mul_(strides[:, None])
    indices.add_(offsets[:, None])
    indices.bitwise_and_(mask)
    return indices.sort(dim=1).values


def _build_sparse_indices(
    qseqlen, kvseqlen, topk, sparse_k_block_size, device, seed=SPARSITY_SEED
):
    """Build deterministic token indices, preserving full blocks for kbs=128."""
    import torch

    if sparse_k_block_size == 1:
        return _build_unique_indices(qseqlen, kvseqlen, topk, device, seed)
    assert sparse_k_block_size == KBS_BLOCK
    block_indices = _build_unique_indices(
        qseqlen,
        kvseqlen // sparse_k_block_size,
        topk // sparse_k_block_size,
        device,
        seed,
    )
    offsets = torch.arange(sparse_k_block_size, dtype=torch.int32, device=device)
    return (block_indices.unsqueeze(-1) * sparse_k_block_size + offsets).reshape(
        qseqlen, topk
    )


def _build_flex_block_mask(qseqlen, kvseqlen, topk, device):
    """Build Q-BLOCK-level sparse mask for FlexAttention (kbs=128).

    FlexAttention operates at (BLOCK_M=128, BLOCK_N=128) granularity.
    Each Q-block of 128 rows shares the same set of selected KV-blocks,
    matching the reference implementation in run_index_sparse_comparison_benchmark.py.
    """
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    num_q_blocks = (qseqlen + KBS_BLOCK - 1) // KBS_BLOCK
    num_kv_blocks = kvseqlen // KBS_BLOCK
    kv_blocks_needed = min(topk // KBS_BLOCK, num_kv_blocks)

    selected = _build_unique_indices(
        num_q_blocks, num_kv_blocks, kv_blocks_needed, device
    ).long()
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        return mask_dense[q_idx // KBS_BLOCK, kv_idx // KBS_BLOCK]

    return create_block_mask(
        sparse_mask_mod,
        B=None,
        H=None,
        Q_LEN=qseqlen,
        KV_LEN=kvseqlen,
        device=device,
    )


def _has_entry(results, key, kvseqlen):
    d = results.get(key, {})
    if not d or "kvseqlen" not in d:
        return False
    try:
        idx = d["kvseqlen"].index(kvseqlen)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, kvseqlen, tflops, ms):
    if key not in results:
        results[key] = {"kvseqlen": [], "tflops": [], "ms": []}
    d = results[key]
    if kvseqlen in d["kvseqlen"]:
        idx = d["kvseqlen"].index(kvseqlen)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["kvseqlen"].append(kvseqlen)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


# ═══════════════════════════════════════════════════════════════
#  8a: kbs=1 Runners
# ═══════════════════════════════════════════════════════════════


def _run_kbs1_ffa_is(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA IndexSparse kbs=1 with pass-selected BWD traversal."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = (
        _build_sparse_indices(qseqlen, kvseqlen, topk, 1, device)
        .unsqueeze(1)
        .expand(-1, NHK, -1)
        .contiguous()
    )
    is_bwd = pass_type != "fwd"
    kw = dict(
        index_sparse_indices=indices,
        q_block_size=1,
        sparse_k_block_size=1,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention with Q-BLOCK-level block-sparse mask for the same sparsity ratio."""
    import torch
    import torch._functorch.config
    from torch.nn.attention.flex_attention import flex_attention

    torch._functorch.config.donated_buffer = False
    q = torch.randn(1, NHQ, qseqlen, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    block_mask = _build_flex_block_mask(qseqlen, kvseqlen, topk, device)
    flex_fn = torch.compile(flex_attention)
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    else:

        def run_fn():
            flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_triton(kvseqlen, qseqlen, topk, pass_type, device, sparse_k_block_size):
    """Time Triton through the same autograd boundary as FFA and FlexAttention."""
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = _build_sparse_indices(
        qseqlen, kvseqlen, topk, sparse_k_block_size, device
    )
    if pass_type == "fwd":

        def run_fn():
            token_sparse_fwd(q, k, v, indices)

        is_bwd = False
    elif pass_type in ("bwd_loopk", "bwd_loopq"):
        bwd_mode = "loopk" if pass_type == "bwd_loopk" else "loopq"
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = token_sparse_attn(
            q,
            k,
            v,
            indices,
            bwd_mode=bwd_mode,
            sparse_k_block_size=sparse_k_block_size,
        )
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

        is_bwd = True
    else:
        raise ValueError(f"unsupported Triton pass_type={pass_type!r}")
    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_triton(kvseqlen, qseqlen, topk, pass_type, device):
    return _run_triton(kvseqlen, qseqlen, topk, pass_type, device, 1)


# ═══════════════════════════════════════════════════════════════
#  8b: kbs=128 Runners
# ═══════════════════════════════════════════════════════════════


def _run_kbs128_ffa_bs(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA BlockSparse (q_ranges/k_ranges + block_sparse + range_merge)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)

    n_kv_blocks = kvseqlen // KBS_BLOCK
    indices = (
        _build_sparse_indices(qseqlen, kvseqlen, topk, KBS_BLOCK, device)[
            :, ::KBS_BLOCK
        ]
        .div(KBS_BLOCK, rounding_mode="floor")
        .unsqueeze(1)
        .expand(-1, NHK, -1)
        .contiguous()
    )
    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_BLOCK, num_k_blocks=n_kv_blocks
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)

    is_bwd = pass_type != "fwd"
    swap_qk = pass_type == "bwd_loopk"
    kw = dict(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=atm,
        pack_gqa=True,
        block_sparse=True,
        range_merge=True,
        disable_fwd_atomic_reduction=True,
        sparse_k_block_size=KBS_BLOCK,
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = swap_qk
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_ffa_is(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA IndexSparse with kbs=128 (index_sparse_indices with block indices)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)

    indices = (
        _build_sparse_indices(qseqlen, kvseqlen, topk, KBS_BLOCK, device)[
            :, ::KBS_BLOCK
        ]
        .div(KBS_BLOCK, rounding_mode="floor")
        .unsqueeze(1)
        .expand(-1, NHK, -1)
        .contiguous()
    )

    is_bwd = pass_type != "fwd"
    swap_qk = pass_type == "bwd_loopk"
    kw = dict(
        index_sparse_indices=indices,
        q_block_size=1,
        sparse_k_block_size=KBS_BLOCK,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = swap_qk
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention over the exact per-query kbs=128 sparse pattern."""
    import torch
    import torch._functorch.config
    from torch.nn.attention.flex_attention import flex_attention

    torch._functorch.config.donated_buffer = False
    q = torch.randn(1, NHQ, qseqlen, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    block_mask = _build_flex_block_mask(qseqlen, kvseqlen, topk, device)
    flex_fn = torch.compile(flex_attention)
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(o)

        def run_fn():
            torch.autograd.grad(o, (q, k, v), do, retain_graph=True)

    else:

        def run_fn():
            flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_triton(kvseqlen, qseqlen, topk, pass_type, device):
    return _run_triton(kvseqlen, qseqlen, topk, pass_type, device, KBS_BLOCK)


# ═══════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════

_KBS1_RUNNERS = {
    "ffa_is": _run_kbs1_ffa_is,
    "flexattn": _run_kbs1_flexattn,
    "triton": _run_kbs1_triton,
}
_KBS128_RUNNERS = {
    "ffa_bs": _run_kbs128_ffa_bs,
    "ffa_is128": _run_kbs128_ffa_is,
    "flexattn": _run_kbs128_flexattn,
    "triton": _run_kbs128_triton,
}

# ═══════════════════════════════════════════════════════════════
#  Correctness Verification
# ═══════════════════════════════════════════════════════════════


def _ref_token_sparse_attn(q, k, v, indices, sm_scale=None):
    """Naive reference: per-query gather topk KV → softmax → output (first 64 rows)."""
    import torch

    S, _Hq, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D**0.5)
    q_f = q.float()
    k_f = k.squeeze(1).float()
    v_f = v.squeeze(1).float()
    o = torch.zeros(min(S, 64), _Hq, D, dtype=torch.float32, device=q.device)
    for i in range(min(S, 64)):
        idx = indices[i].long()
        ki = k_f[idx]
        vi = v_f[idx]
        qi = q_f[i]
        scores = (qi @ ki.T) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        o[i] = weights @ vi
    return o.to(q.dtype)


def _sanity_check(device):
    """Retain FWD checks and compare direct Triton LoopK/LoopQ gradients."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    kvseqlen_ck, qseqlen_ck, topk_ck = 2048, 256, 256
    torch.manual_seed(SPARSITY_SEED)
    q = torch.randn(qseqlen_ck, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)
    indices_2d = _build_sparse_indices(qseqlen_ck, kvseqlen_ck, topk_ck, 1, device)
    assert torch.all(indices_2d[:, 1:] > indices_2d[:, :-1])
    ref = _ref_token_sparse_attn(q, k, v, indices_2d)
    fwd_results = {}

    is_indices = indices_2d.unsqueeze(1).expand(-1, NHK, -1).contiguous()
    ffa_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=is_indices,
        q_block_size=1,
        sparse_k_block_size=1,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    fwd_results["ffa_is_kbs1"] = (ffa_out[:64].float() - ref.float()).abs().max().item()

    token_idx_128 = _build_sparse_indices(
        qseqlen_ck, kvseqlen_ck, topk_ck, KBS_BLOCK, device
    )
    ref128 = _ref_token_sparse_attn(q, k, v, token_idx_128)
    bs_idx = (
        token_idx_128[:, ::KBS_BLOCK]
        .div(KBS_BLOCK, rounding_mode="floor")
        .unsqueeze(1)
        .expand(-1, NHK, -1)
        .contiguous()
    )
    n_kv_blocks = kvseqlen_ck // KBS_BLOCK
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        bs_idx.permute(1, 0, 2).contiguous(),
        block_m=1,
        block_n=KBS_BLOCK,
        num_k_blocks=n_kv_blocks,
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)
    bs_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=atm,
        pack_gqa=True,
        block_sparse=True,
        range_merge=True,
        disable_fwd_atomic_reduction=True,
        sparse_k_block_size=KBS_BLOCK,
    )
    fwd_results["ffa_bs_kbs128"] = (
        (bs_out[:64].float() - ref128.float()).abs().max().item()
    )
    is128_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=bs_idx,
        q_block_size=1,
        sparse_k_block_size=KBS_BLOCK,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    fwd_results["ffa_is_kbs128"] = (
        (is128_out[:64].float() - ref128.float()).abs().max().item()
    )

    from torch.nn.attention.flex_attention import flex_attention

    q_bhsd = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_bhsd = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_bhsd = v.unsqueeze(0).permute(0, 2, 1, 3)
    block_mask = _build_flex_block_mask(qseqlen_ck, kvseqlen_ck, topk_ck, device)
    flex_out = torch.compile(flex_attention)(
        q_bhsd, k_bhsd, v_bhsd, block_mask=block_mask, enable_gqa=True
    )
    fwd_results["flexattn_kbs128"] = (
        0.0 if torch.isfinite(flex_out).all() else float("inf")
    )

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_bwd, token_sparse_fwd

    tri_out = token_sparse_fwd(q, k, v, indices_2d)
    fwd_results["triton"] = (tri_out[:64].float() - ref.float()).abs().max().item()

    bwd_results = {}
    for kbs, indices in ((1, indices_2d), (KBS_BLOCK, token_idx_128)):
        o, lse = token_sparse_fwd(q, k, v, indices, return_lse=True)
        torch.manual_seed(SPARSITY_SEED + kbs)
        do = torch.randn_like(o)
        loopk = token_sparse_bwd(
            q,
            k,
            v,
            indices,
            o,
            do,
            lse,
            bwd_mode="loopk",
            sparse_k_block_size=kbs,
        )
        loopq = token_sparse_bwd(
            q,
            k,
            v,
            indices,
            o,
            do,
            lse,
            bwd_mode="loopq",
            sparse_k_block_size=kbs,
        )
        for name, lhs, rhs in zip(("dQ", "dK", "dV"), loopk, loopq):
            finite = torch.isfinite(lhs).all() and torch.isfinite(rhs).all()
            close = torch.allclose(lhs, rhs, atol=0.125, rtol=0.125)
            max_abs = (lhs.float() - rhs.float()).abs().max().item()
            bwd_results[f"triton_kbs{kbs}_{name}"] = (bool(finite and close), max_abs)

    print("  Correctness check (kvseqlen=2048, qseqlen=256, topk=256):", flush=True)
    all_pass = True
    for method, err in fwd_results.items():
        passed = err < 0.05
        all_pass &= passed
        print(
            f"    {method:22s}: {'PASS' if passed else f'FAIL (err={err:.4f})'}",
            flush=True,
        )
    for method, (passed, max_abs) in bwd_results.items():
        all_pass &= passed
        print(
            f"    {method:22s}: {'PASS' if passed else 'FAIL'} (max_abs={max_abs:.4f})",
            flush=True,
        )
    if not all_pass:
        print("  [ERROR] Correctness failed! Aborting.", flush=True)
    return all_pass


# ═══════════════════════════════════════════════════════════════
#  --exp
# ═══════════════════════════════════════════════════════════════


def _phase8_bench(force=False, rerun_filter=None):
    import torch

    valid_keys = {
        f"{prefix}/{pass_type}/{method}"
        for prefix, pass_methods in (
            ("kbs1", KBS1_PASS_METHODS),
            ("kbs128", KBS128_PASS_METHODS),
        )
        for pass_type, methods in pass_methods.items()
        for method in methods
    }
    # Keep cached values only for runners still present in the current schema.
    results = (
        {}
        if force
        else {
            key: value
            for key, value in _load_results(PHASE).items()
            if key in valid_keys
        }
    )
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 8: Baseline Comparison (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, "
        f"scenarios: kvseqlen=[{','.join(f'{s[0] // 1024}k' for s in SCENARIOS)}], "
        f"qseqlen=kvseqlen/64, topk=kvseqlen/8",
        flush=True,
    )
    print(
        "  All BWD methods use torch.autograd.grad with no leaf .grad accumulation.",
        flush=True,
    )
    if not _sanity_check(device):
        return

    def _run_group(prefix, pass_methods, runners, label):
        print("\n  " + "─" * 55, flush=True)
        print(f"  {label}", flush=True)
        print("  " + "─" * 55, flush=True)
        for kvseqlen, qseqlen, topk in SCENARIOS:
            print(
                f"  ── kvseqlen={kvseqlen // 1024}k, qseqlen={qseqlen}, "
                f"topk={topk // 1024}k ──",
                flush=True,
            )
            for pass_type, methods in pass_methods.items():
                for method in methods:
                    key = f"{prefix}/{pass_type}/{method}"
                    if not force and _has_entry(results, key, kvseqlen):
                        d = results[key]
                        tf = d["tflops"][d["kvseqlen"].index(kvseqlen)]
                        print(
                            f"    {pass_type:10s} {method:10s}: {tf:>7.1f} T (cached)",
                            flush=True,
                        )
                        continue
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        tf, ms = runners[method](
                            kvseqlen, qseqlen, topk, pass_type, device
                        )
                        _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                        _save_results(PHASE, results)
                        print(
                            f"    {pass_type:10s} {method:10s}: {tf:>7.1f} T  ({ms:.3f} ms)",
                            flush=True,
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"    {pass_type:10s} {method:10s}: OOM", flush=True)
                        _set_entry(results, key, kvseqlen, None, None)
                        _save_results(PHASE, results)
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(
                            f"    {pass_type:10s} {method:10s}: ERR — {error}",
                            flush=True,
                        )
                        _set_entry(results, key, kvseqlen, None, None)
                        _save_results(PHASE, results)
                        torch.cuda.empty_cache()

    _run_group(
        "kbs1",
        KBS1_PASS_METHODS,
        _KBS1_RUNNERS,
        "8a) kbs=1: pass-specific token-sparse baselines",
    )
    _run_group(
        "kbs128",
        KBS128_PASS_METHODS,
        _KBS128_RUNNERS,
        f"8b) kbs={KBS_BLOCK}: pass-specific full-block baselines",
    )
    print(f"\n[{_ts()}] Phase 8 done.", flush=True)


# ═══════════════════════════════════════════════════════════════
#  --plot
# ═══════════════════════════════════════════════════════════════


def _phase8_plot():
    """Plot only methods present in each pass-specific runner matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results(PHASE)
    if not results:
        print("  [WARN] No results found. Run --exp first.")
        return
    out = _out_dir(PHASE)
    os.makedirs(out, exist_ok=True)
    kvseqlens = [scenario[0] for scenario in SCENARIOS]
    x = np.arange(len(kvseqlens))
    x_labels = [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 8192}K)" for kv in kvseqlens]
    xlabel = "kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)"
    colors = {
        "ffa_bs": COLOR_BLOCK_SPARSE,
        "ffa_is": COLOR_INDEX_SPARSE,
        "ffa_is128": COLOR_INDEX_SPARSE,
        "flexattn": COLOR_FLEXATTN,
        "triton": COLOR_TRITON,
    }

    def _vals(prefix, pass_type, method):
        data = results.get(f"{prefix}/{pass_type}/{method}", {})
        values = []
        for kv in kvseqlens:
            if kv in data.get("kvseqlen", []):
                value = data["tflops"][data["kvseqlen"].index(kv)]
                values.append(value or 0)
            else:
                values.append(0)
        return values

    def _plot_group(prefix, pass_methods, labels, title, filename):
        names = {
            "fwd": "FWD",
            "bwd_loopq": "BWD InnerLoopQ",
            "bwd_loopk": "BWD InnerLoopK",
        }
        ncols = len(pass_methods)
        fig, axes = plt.subplots(
            1, ncols, figsize=(PLOT_SUBPLOT_WIDTH * ncols, PLOT_SUBPLOT_HEIGHT), dpi=150
        )
        if ncols == 1:
            axes = [axes]
        for axis, (pass_type, methods) in zip(axes, pass_methods.items()):
            plot_methods = [
                (method, labels[method], colors[method]) for method in methods
            ]
            plot_grouped_bars(
                axis,
                x,
                plot_methods,
                lambda method, selected_pass=pass_type: _vals(
                    prefix, selected_pass, method
                ),
                title=names[pass_type],
                xlabel=xlabel,
                x_labels=x_labels,
            )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
        plt.tight_layout()
        output_path = os.path.join(out, filename)
        fig.savefig(output_path, dpi=PLOT_DPI_SAVE, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_path}", flush=True)

    _plot_group(
        "kbs1",
        KBS1_PASS_METHODS,
        KBS1_LABELS,
        f"Phase 8a: Token-Sparse (kbs=1), nhq={NHQ}, nhk={NHK}, hd={HD}",
        "phase8a_kbs1.png",
    )
    _plot_group(
        "kbs128",
        KBS128_PASS_METHODS,
        KBS128_LABELS,
        f"Phase 8b: Full-Block Pattern (kbs={KBS_BLOCK}), nhq={NHQ}, nhk={NHK}, hd={HD}",
        "phase8b_kbs128.png",
    )

    print("\n  " + "─" * 60)
    print("  Summary Tables (TFLOPS; unified autograd.grad timing)")
    print("  " + "─" * 60)
    for prefix, pass_methods, labels in (
        ("kbs1", KBS1_PASS_METHODS, KBS1_LABELS),
        ("kbs128", KBS128_PASS_METHODS, KBS128_LABELS),
    ):
        for pass_type, methods in pass_methods.items():
            print(f"\n  {pass_type.upper()} ({prefix}):")
            print(
                "  "
                + f"{'kvseqlen':>8s}"
                + "".join(f"  {labels[method]:>36s}" for method in methods)
            )
            for kv in kvseqlens:
                row = f"  {kv // 1024:>7d}k"
                for method in methods:
                    data = results.get(f"{prefix}/{pass_type}/{method}", {})
                    if kv in data.get("kvseqlen", []):
                        tf = data["tflops"][data["kvseqlen"].index(kv)]
                        row += f"  {tf:>36.1f}" if tf else "  " + "—".rjust(36)
                    else:
                        row += "  " + "—".rjust(36)
                print(row)
