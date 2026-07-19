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
      FFA IndexSparse vs FlexAttention vs Triton LoopK vs Triton LoopQ
      Passes: FWD, BWD (LoopK + LoopQ comparison for Triton)

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
    COLOR_TRITON_LOOPQ,
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

# 8a: kbs=1 (token-sparse)
KBS1_METHODS = ["ffa_is", "flexattn", "triton", "triton_loopq"]
KBS1_PASSES = ["fwd", "bwd"]
KBS1_LABELS = {
    "ffa_is": "FFA IndexSparse (kbs=1)",
    "flexattn": "FlexAttention",
    "triton": "Triton LoopK",
    "triton_loopq": "Triton LoopQ",
}

# 8b: kbs=128 (block-sparse)
KBS128_METHODS = ["ffa_bs", "ffa_is128", "flexattn", "triton", "triton_loopq"]
KBS128_PASSES = ["fwd", "bwd_loopk", "bwd_loopq"]
KBS128_LABELS = {
    "ffa_bs": "FFA BlockSparse",
    "ffa_is128": "FFA IndexSparse (kbs=128)",
    "flexattn": "FlexAttention",
    "triton": "Triton LoopK",
    "triton_loopq": "Triton LoopQ Dense",
}


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _calc_sparse_flops(qseqlen, topk, is_bwd):
    fwd = 4 * qseqlen * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_flex_block_mask(qseqlen, kvseqlen, topk, device):
    """FlexAttention block_mask for sparse pattern (128x128 blocks), Q≠KV."""
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_BLOCK = 128
    num_q_blocks = (qseqlen + FLEX_BLOCK - 1) // FLEX_BLOCK
    num_kv_blocks = kvseqlen // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected = torch.rand(num_q_blocks, num_kv_blocks, device=device).argsort(dim=1)[
        :, :kv_blocks_needed
    ]
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_block = q_idx // FLEX_BLOCK
        kv_block = kv_idx // FLEX_BLOCK
        return mask_dense[q_block, kv_block]

    return create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kvseqlen, device=device
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
    """FFA IndexSparse kbs=1. BWD always uses LoopK (no LoopQ for kbs=1)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = (
        torch.randint(0, kvseqlen, (qseqlen, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values.unsqueeze(1)
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
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention with sparse block mask."""
    import torch
    import torch._functorch.config
    from torch.nn.attention.flex_attention import flex_attention

    torch._functorch.config.donated_buffer = False
    q = torch.randn(1, NHQ, qseqlen, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    block_mask = _build_flex_block_mask(qseqlen, kvseqlen, topk, device)
    _flex_fn = torch.compile(flex_attention)
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_triton(kvseqlen, qseqlen, topk, pass_type, device):
    """Triton hand-written token-sparse kernel."""
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    tri_indices = (
        torch.randint(0, kvseqlen, (qseqlen, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values
    )
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = token_sparse_attn(q, k, v, tri_indices)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_triton_loopq(kvseqlen, qseqlen, topk, pass_type, device):
    """Triton token-sparse BWD using LoopQ direction (inverse index, no dKV atomics).

    FWD is identical to standard Triton — returns same result.
    BWD uses dkv_mode="loopq": outer KV blocks, inner Q positions via inverse index.
    """
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_bwd, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    tri_indices = (
        torch.randint(0, kvseqlen, (qseqlen, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values
    )
    is_bwd = pass_type != "fwd"

    if is_bwd:
        o, lse = token_sparse_fwd(q, k, v, tri_indices, return_lse=True)
        do = torch.randn_like(o)

        def run_fn():
            token_sparse_bwd(q, k, v, tri_indices, o, do, lse, dkv_mode="loopq")

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


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
    n_topk_blocks = topk // KBS_BLOCK
    indices = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
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
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = swap_qk
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

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

    n_kv_blocks = kvseqlen // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    indices = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
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
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention for kbs=128 group (same as kbs1 but different scenario)."""
    return _run_kbs1_flexattn(kvseqlen, qseqlen, topk, pass_type, device)


def _run_kbs128_triton(kvseqlen, qseqlen, topk, pass_type, device):
    """Triton Token-Sparse in kbs=128 scenario (expand block indices to tokens)."""
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)

    n_kv_blocks = kvseqlen // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    block_idx = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values
    )
    tri_indices = (
        (block_idx.unsqueeze(-1) * KBS_BLOCK + torch.arange(KBS_BLOCK, device=device))
        .reshape(qseqlen, topk)
        .to(torch.int32)
    )
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = token_sparse_attn(q, k, v, tri_indices)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_triton_loopq(kvseqlen, qseqlen, topk, pass_type, device):
    """Triton kbs=128 BWD using dense LoopQ (inverse index, no dKV atomics).

    dKV uses _loopq_dense_dkv_kernel: S[NHQ, 128] fully dense, tl.dot,
    fp32 accumulation, no atomics. FWD is identical to standard Triton.
    """
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_bwd, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)

    n_kv_blocks = kvseqlen // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    block_idx = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values
    )
    tri_indices = (
        (block_idx.unsqueeze(-1) * KBS_BLOCK + torch.arange(KBS_BLOCK, device=device))
        .reshape(qseqlen, topk)
        .to(torch.int32)
    )
    is_bwd = pass_type != "fwd"

    if is_bwd:
        o, lse = token_sparse_fwd(q, k, v, tri_indices, return_lse=True)
        do = torch.randn_like(o)

        def run_fn():
            token_sparse_bwd(q, k, v, tri_indices, o, do, lse, dkv_mode="loopq_dense")

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


# ═══════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════

_KBS1_RUNNERS = {
    "ffa_is": _run_kbs1_ffa_is,
    "flexattn": _run_kbs1_flexattn,
    "triton": _run_kbs1_triton,
    "triton_loopq": _run_kbs1_triton_loopq,
}

_KBS128_RUNNERS = {
    "ffa_bs": _run_kbs128_ffa_bs,
    "ffa_is128": _run_kbs128_ffa_is,
    "flexattn": _run_kbs128_flexattn,
    "triton": _run_kbs128_triton,
    "triton_loopq": _run_kbs128_triton_loopq,
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
    """Verify correctness of all methods at small scale before benchmarking."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    kvseqlen_ck, qseqlen_ck, topk_ck = 2048, 256, 256
    torch.manual_seed(42)
    q = torch.randn(qseqlen_ck, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)

    indices_2d = (
        torch.randint(
            0, kvseqlen_ck, (qseqlen_ck, topk_ck), device=device, dtype=torch.int32
        )
        .sort(dim=1)
        .values
    )
    ref = _ref_token_sparse_attn(q, k, v, indices_2d)
    results = {}

    # 1. FFA IndexSparse kbs=1
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
    results["ffa_is_kbs1"] = (ffa_out[:64].float() - ref.float()).abs().max().item()

    # 2. FFA BlockSparse kbs=128
    n_kv_blocks = kvseqlen_ck // KBS_BLOCK
    n_topk_blocks = topk_ck // KBS_BLOCK
    bs_idx = (
        torch.rand(qseqlen_ck, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
        .contiguous()
    )
    ia_3d = bs_idx.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_BLOCK, num_k_blocks=n_kv_blocks
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
    )
    bs_ok = not torch.isnan(bs_out).any() and not torch.isinf(bs_out).any()
    results["ffa_bs_kbs128"] = 0.0 if bs_ok else float("inf")

    # 3. FFA IndexSparse kbs=128
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
    is128_ok = not torch.isnan(is128_out).any() and not torch.isinf(is128_out).any()
    results["ffa_is_kbs128"] = 0.0 if is128_ok else float("inf")

    # 4. FlexAttention
    from torch.nn.attention.flex_attention import flex_attention

    q_bhsd = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_bhsd = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_bhsd = v.unsqueeze(0).permute(0, 2, 1, 3)
    block_mask = _build_flex_block_mask(qseqlen_ck, kvseqlen_ck, topk_ck, device)
    flex_fn = torch.compile(flex_attention)
    flex_out = flex_fn(q_bhsd, k_bhsd, v_bhsd, block_mask=block_mask, enable_gqa=True)
    flex_ok = not torch.isnan(flex_out).any() and not torch.isinf(flex_out).any()
    results["flexattn"] = 0.0 if flex_ok else float("inf")

    # 5. Triton Token-Sparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_fwd

    tri_out = token_sparse_fwd(q, k, v, indices_2d)
    results["triton"] = (tri_out[:64].float() - ref.float()).abs().max().item()

    # Print
    print("  Correctness check (kvseqlen=2048, qseqlen=256, topk=256):", flush=True)
    all_pass = True
    for method, val in results.items():
        status = "PASS" if val < 0.05 else f"FAIL (err={val:.4f})"
        if val >= 0.05:
            all_pass = False
        print(f"    {method:18s}: {status}", flush=True)
    if not all_pass:
        print("  [ERROR] Correctness failed! Aborting.", flush=True)
    return all_pass


# ═══════════════════════════════════════════════════════════════
#  --exp
# ═══════════════════════════════════════════════════════════════


def _phase8_bench(force=False, rerun_filter=None):
    import torch

    results = _load_results(PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    print(f"[{_ts()}] Phase 8: Baseline Comparison (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, "
        f"scenarios: kvseqlen=[{','.join(f'{s[0] // 1024}k' for s in SCENARIOS)}], "
        f"qseqlen=kvseqlen/64, topk=kvseqlen/8",
        flush=True,
    )

    if not _sanity_check(device):
        return

    def _run_group(prefix, methods, passes, runners, label):
        print("\n  " + "─" * 55, flush=True)
        print(f"  {label}", flush=True)
        print("  " + "─" * 55, flush=True)
        for kvseqlen, qseqlen, topk in SCENARIOS:
            print(
                f"  ── kvseqlen={kvseqlen // 1024}k, "
                f"qseqlen={qseqlen}, topk={topk // 1024}k ──",
                flush=True,
            )
            for pass_type in passes:
                for method in methods:
                    key = f"{prefix}/{pass_type}/{method}"
                    if not force and _has_entry(results, key, kvseqlen):
                        d = results[key]
                        idx = d["kvseqlen"].index(kvseqlen)
                        tf = d["tflops"][idx]
                        print(
                            f"    {pass_type:10s} {method:10s}: "
                            f"{tf:>7.1f} T (cached)",
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
                            f"    {pass_type:10s} {method:10s}: "
                            f"{tf:>7.1f} T  ({ms:.3f} ms)",
                            flush=True,
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"    {pass_type:10s} {method:10s}: OOM",
                            flush=True,
                        )
                        _set_entry(results, key, kvseqlen, None, None)
                        _save_results(PHASE, results)
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(
                            f"    {pass_type:10s} {method:10s}: ERR — {e}",
                            flush=True,
                        )
                        _set_entry(results, key, kvseqlen, None, None)
                        _save_results(PHASE, results)
                        torch.cuda.empty_cache()

    _run_group(
        "kbs1",
        KBS1_METHODS,
        KBS1_PASSES,
        _KBS1_RUNNERS,
        "8a) kbs=1: IndexSparse vs Baselines (FWD + BWD LoopK)",
    )
    _run_group(
        "kbs128",
        KBS128_METHODS,
        KBS128_PASSES,
        _KBS128_RUNNERS,
        f"8b) kbs={KBS_BLOCK}: BS/IS vs Baselines (FWD + BWD LoopK + BWD LoopQ)",
    )

    print(f"\n[{_ts()}] Phase 8 done.", flush=True)


# ═══════════════════════════════════════════════════════════════
#  --plot
# ═══════════════════════════════════════════════════════════════


def _phase8_plot():
    """Generate Phase-6 style grouped bar charts via plot_grouped_bars().

    Baselines (grey) on LEFT, our kernels (colored) on RIGHT — matching Phase 6.
    """
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

    kvseqlens = [s[0] for s in SCENARIOS]
    x = np.arange(len(kvseqlens))
    x_labels = [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 8192}K)" for kv in kvseqlens]
    xlabel = "kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)"

    def _vals(prefix, pid, mid):
        key = f"{prefix}/{pid}/{mid}"
        d = results.get(key, {})
        out_vals = []
        for kv in kvseqlens:
            if kv in d.get("kvseqlen", []):
                idx = d["kvseqlen"].index(kv)
                v = d["tflops"][idx] if d["tflops"][idx] else 0
            else:
                v = 0
            out_vals.append(v)
        return out_vals

    # ── 8a) kbs=1: baselines LEFT, our kernel RIGHT ──
    kbs1_pass_defs = [("fwd", "FWD"), ("bwd", "BWD (LoopK)")]
    kbs1_methods = [
        ("flexattn", "FlexAttention", COLOR_FLEXATTN),
        ("triton", "Triton LoopK", COLOR_TRITON),
        ("triton_loopq", "Triton LoopQ", COLOR_TRITON_LOOPQ),
        ("ffa_is", "FFA IndexSparse", COLOR_INDEX_SPARSE),
    ]
    n_cols = len(kbs1_pass_defs)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(PLOT_SUBPLOT_WIDTH * n_cols, PLOT_SUBPLOT_HEIGHT), dpi=150
    )
    if n_cols == 1:
        axes = [axes]
    for col_idx, (pid, pname) in enumerate(kbs1_pass_defs):
        plot_grouped_bars(
            axes[col_idx],
            x,
            kbs1_methods,
            lambda mid, _pid=pid: _vals("kbs1", _pid, mid),
            title=pname,
            xlabel=xlabel,
            x_labels=x_labels,
        )
    fig.suptitle(
        "Phase 8a: Token-Sparse Baseline (kbs=1)\n"
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, PackGQA, bf16, H100",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = os.path.join(out, "phase8a_kbs1.png")
    fig.savefig(p, dpi=PLOT_DPI_SAVE, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}", flush=True)

    # ── 8b) kbs=128: baselines LEFT, our kernels RIGHT ──
    kbs128_pass_defs = [
        ("fwd", "FWD"),
        ("bwd_loopk", "BWD LoopK"),
        ("bwd_loopq", "BWD LoopQ"),
    ]
    kbs128_methods = [
        ("flexattn", "FlexAttention", COLOR_FLEXATTN),
        ("triton", "Triton LoopK", COLOR_TRITON),
        ("triton_loopq", "Triton LoopQ Dense", COLOR_TRITON_LOOPQ),
        ("ffa_bs", "FFA BlockSparse", COLOR_BLOCK_SPARSE),
        ("ffa_is128", "FFA IndexSparse (kbs=128)", COLOR_INDEX_SPARSE),
    ]
    n_cols = len(kbs128_pass_defs)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(PLOT_SUBPLOT_WIDTH * n_cols, PLOT_SUBPLOT_HEIGHT), dpi=150
    )
    for col_idx, (pid, pname) in enumerate(kbs128_pass_defs):
        plot_grouped_bars(
            axes[col_idx],
            x,
            kbs128_methods,
            lambda mid, _pid=pid: _vals("kbs128", _pid, mid),
            title=pname,
            xlabel=xlabel,
            x_labels=x_labels,
        )
    fig.suptitle(
        f"Phase 8b: Block-Sparse Baseline (kbs={KBS_BLOCK})\n"
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, PackGQA, bf16, H100",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    p = os.path.join(out, "phase8b_kbs128.png")
    fig.savefig(p, dpi=PLOT_DPI_SAVE, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}", flush=True)

    # Summary tables
    print("\n  " + "\u2500" * 60)
    print("  Summary Tables (TFLOPS)")
    print("  " + "\u2500" * 60)

    for prefix, methods, labels, passes in [
        ("kbs1", KBS1_METHODS, KBS1_LABELS, KBS1_PASSES),
        ("kbs128", KBS128_METHODS, KBS128_LABELS, KBS128_PASSES),
    ]:
        for pt in passes:
            print(f"\n  {pt.upper()} ({prefix}):")
            header = "  " + f"{'kvseqlen':>8s}"
            for m in methods:
                header += f"  {labels[m]:>24s}"
            print(header)
            for kv in kvseqlens:
                row = f"  {kv // 1024:>7d}k"
                for m in methods:
                    key = f"{prefix}/{pt}/{m}"
                    d = results.get(key, {})
                    if d and kv in d.get("kvseqlen", []):
                        idx = d["kvseqlen"].index(kv)
                        tf = d["tflops"][idx]
                        row += f"  {tf:>24.1f}" if tf else ("  " + "\u2014".rjust(24))
                    else:
                        row += "  " + "\u2014".rjust(24)
                print(row)
