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

"""IndexSparse (token-level scatter) sweep tests for the cutedsl SM100 kernel.

Mirrors the structure of ``tests/test_attn/test_index_sparse.py`` on main, but
drives the kernel-level entry points directly (``_flex_flash_attn_fwd`` /
``_flex_flash_attn_bwd`` + ``prepare_index_sparse_tiles``) since this fork does
not expose the ``index_sparse_indices`` autograd wrapper yet.

Mirrors the tier structure and parameter ranges of SM90's
``tests/test_attn/test_index_sparse.py`` (all with *random* per-Q-tile token indices):
  * Tier 1  (test_simple_index_sparse):  PackGQA GQA ratios 128/64/32/16, S=256
  * Tier 2a (test_sparse_cross_batch):   B = 2 / 3 / 8, uniform topk
  * Tier 2b (test_sparse_qkv_lengths):   short/unaligned Q vs long KV (64/1024, 8/512, 100/512)
  * Tier 3a (test_sparse_head_dim):      D = 64 / 128
  * Tier 3b (test_sparse_long_seq):      S = 8192, topk = 1024
  * Tier 3c (test_sparse_gqa):           NHK > 1 (GQA), PackGQA on/off
  * Tier 3d (test_sparse_mha):           NHQ == NHK (MHA)
  * Tier P  (test_partial_topk_multi_niter): SM100-only partial topk (topk % 128 != 0)

Differences from SM90 (this fork drives the kernel entry points directly and does
not expose the ``index_sparse_indices`` autograd wrapper yet):
  * swap_ab / k_block_size > 1 are not plumbed through the direct-kernel helper
    (the small-ratio / MHA configs run fine without an explicit swap_ab knob).
  * Per-batch *variable* topk is unsupported (prepare_index_sparse_tiles takes a
    scalar topk), so the cross-batch tier uses a *uniform* topk.
  * The S=65536 int32-overflow regression is omitted (the dense fp32 SDPA ref
    would OOM); Tier P instead covers partial topk, which SM90 forbids.

Scatter store modes (bulk cp.async vs per-element atomicAdd) are selected via
MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC at kernel-compile time, which is
NOT part of the JIT cache key — so each mode must run in its own process:

    pytest tests/test_kernel/cutedsl/test_ffa_index_sparse.py -v
    MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC=1 \
        pytest tests/test_kernel/cutedsl/test_ffa_index_sparse.py -v

Known constraints (asserted in the kernel):
  * BWD IndexSparse requires swap_bwd_qk_loop=True (LoopK) and head_dim <= 128
  * All queries inside one FWD sparse Q-tile must share the same token set
    (prepare_index_sparse_tiles takes the tile's first query as representative)
"""

import os

import pytest
import torch

from magi_attention.kernel.cutedsl.ffa_utils import TorchFlexAttnArgs
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.sparse_utils import prepare_index_sparse_tiles
from magi_attention.testing.precision import assert_close

SEED = 42

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="IndexSparse cutedsl path requires SM100+",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fwd_sparse_m_block(seqlen_q: int, qhpk: int, pack_gqa: bool) -> int:
    """FWD sparse Q-tile rows: q_stage=2 doubles the 128-row tile when the
    packed Q extent exceeds one tile (mirrors _flex_flash_attn_fwd)."""
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


def _build_block_random_indices(
    B: int,
    NHQ: int,
    NHK: int,
    SQ: int,
    SK: int,
    topk: int,
    tokens_per_block: int,
    device: str,
) -> torch.Tensor:
    """Random sorted token indices, shared by all queries within one index
    block and by all Q heads within one KV group.

    Returns (B, NHQ, SQ, topk) int32.
    """
    qhpk = NHQ // NHK
    num_blocks = (SQ + tokens_per_block - 1) // tokens_per_block
    picks = torch.stack(
        [torch.randperm(SK, device=device)[:topk] for _ in range(B * NHK * num_blocks)]
    ).view(B, NHK, num_blocks, topk)
    picks = picks.sort(dim=-1).values.int()
    # expand KV-head blocks to Q heads and per-token rows
    q_to_block = torch.arange(SQ, device=device) // tokens_per_block
    per_token = picks[:, :, q_to_block, :]  # (B, NHK, SQ, topk)
    indices = (
        per_token.unsqueeze(2)
        .expand(B, NHK, qhpk, SQ, topk)
        .reshape(B, NHQ, SQ, topk)
        .contiguous()
    )
    return indices


def _sdpa_ref_fwd_bwd(q, k, v, dO, indices, softmax_scale):
    """Vectorized fp32 SDPA reference with token-level index mask.

    indices: (B, NHQ, SQ, topk) local KV token ids (all valid, >= 0).
    Returns (O, dQ, dK, dV) in bf16.
    """
    B, SQ, NHQ, HD = q.shape
    _, SK, NHK, _ = k.shape
    HDV = v.shape[-1]
    qhpk = NHQ // NHK

    q_f = q.float().detach().requires_grad_(True)
    k_f = k.float().detach().requires_grad_(True)
    v_f = v.float().detach().requires_grad_(True)

    k_exp = k_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HD).reshape(B, SK, NHQ, HD)
    v_exp = v_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HDV).reshape(B, SK, NHQ, HDV)

    scores = torch.einsum("bsnh,btnh->bnst", q_f, k_exp) * softmax_scale

    mask = torch.zeros(B, NHQ, SQ, SK, dtype=torch.bool, device=q.device)
    mask.scatter_(3, indices.long(), True)

    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bnst,btnh->bsnh", probs, v_exp)
    out.backward(dO.float())

    return (
        out.bfloat16(),
        q_f.grad.bfloat16(),
        k_f.grad.bfloat16(),
        v_f.grad.bfloat16(),
    )


# Tolerances mirror remote main tests/test_attn/sparse_test_utils.py
# (DEFAULT_*_ATOL/RTOL + DEFAULT_MISMATCH_THRES). Validation reuses the same
# magi_attention.testing.precision.assert_close helper as main: element-wise
# |a-b| <= atol + rtol*|b| with up to MISMATCH_THRES fraction of outliers.
_FWD_ATOL, _FWD_RTOL = 0.01, 0.05
_BWD_DQ_ATOL, _BWD_DQ_RTOL = 0.02, 0.3
_BWD_DKV_ATOL, _BWD_DK_RTOL, _BWD_DV_RTOL = 0.02, 0.15, 0.05
_MISMATCH_THRES = 0.01


def _run_index_sparse_case(
    *,
    B: int = 1,
    NHQ: int,
    NHK: int,
    D: int = 128,
    SQ: int,
    SK: int,
    topk: int,
    pack_gqa: bool,
    device: str = "cuda",
):
    torch.manual_seed(SEED)
    qhpk = NHQ // NHK
    softmax_scale = D**-0.5

    q = torch.randn(B, SQ, NHQ, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, SK, NHK, D, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dO = torch.randn_like(q)

    fwd_m_block = _fwd_sparse_m_block(SQ, qhpk, pack_gqa)
    # Indices must be uniform at the coarsest tile granularity (FWD tile);
    # the 128-row BWD LoopK tiles are then automatically uniform too.
    tokens_per_block = max(fwd_m_block // (qhpk if pack_gqa else 1), 1)
    indices = _build_block_random_indices(
        B, NHQ, NHK, SQ, SK, topk, tokens_per_block, device
    )

    o_ref, dq_ref, dk_ref, dv_ref = _sdpa_ref_fwd_bwd(
        q, k, v, dO, indices, softmax_scale
    )

    fwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=B,
        seqlen_q=SQ,
        seqlen_k=SK,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=fwd_m_block,
        n_block_size=128,
        pack_gqa=pack_gqa,
    )
    out, lse = _flex_flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        flex_attn_args=TorchFlexAttnArgs(index_sparse_tiles=fwd_tiles),
        pack_gqa=pack_gqa,
    )

    bwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=B,
        seqlen_q=SQ,
        seqlen_k=SK,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=128,  # LoopK BWD native packed-M tile
        n_block_size=128,
        pack_gqa=pack_gqa,
    )
    dq, dk, dv = _flex_flash_attn_bwd(
        q,
        k,
        v,
        out,
        lse,
        dO,
        softmax_scale=softmax_scale,
        flex_attn_args=TorchFlexAttnArgs(index_sparse_tiles=bwd_tiles),
        swap_bwd_qk_loop=True,
        pack_gqa=pack_gqa,
    )

    tc = (
        f"B={B},NHQ={NHQ},NHK={NHK},D={D},SQ={SQ},SK={SK},topk={topk},"
        f"pack_gqa={pack_gqa},atomic="
        f"{os.environ.get('MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC', '0')}"
    )
    assert_close(
        out,
        o_ref,
        atol=_FWD_ATOL,
        rtol=_FWD_RTOL,
        mismatch_threshold=_MISMATCH_THRES,
        test_case=f"[{tc}] => fwd_out",
    )
    assert_close(
        dq,
        dq_ref,
        atol=_BWD_DQ_ATOL,
        rtol=_BWD_DQ_RTOL,
        mismatch_threshold=_MISMATCH_THRES,
        test_case=f"[{tc}] => dq",
    )
    assert_close(
        dk,
        dk_ref,
        atol=_BWD_DKV_ATOL,
        rtol=_BWD_DK_RTOL,
        mismatch_threshold=_MISMATCH_THRES,
        test_case=f"[{tc}] => dk",
    )
    assert_close(
        dv,
        dv_ref,
        atol=_BWD_DKV_ATOL,
        rtol=_BWD_DV_RTOL,
        mismatch_threshold=_MISMATCH_THRES,
        test_case=f"[{tc}] => dv",
    )


# =============================================================================
# SM90-mirrored sweep (see tests/test_attn/test_index_sparse.py)
# =============================================================================


def _run_config(cfg: dict):
    """Map an SM90-style config dict onto the kernel-level runner."""
    S = cfg.get("S", None)
    S_kv = cfg.get("S_kv", S)
    S_q = cfg.get("S_q", min(S_kv, 256))
    _run_index_sparse_case(
        B=cfg.get("B", 1),
        NHQ=cfg["NHQ"],
        NHK=cfg["NHK"],
        D=cfg.get("D", 128),
        SQ=S_q,
        SK=S_kv,
        topk=cfg["topk"],
        pack_gqa=cfg.get("pack_gqa", True),
    )


def _cfgs(configs):
    return pytest.mark.parametrize("config", configs, ids=[c["name"] for c in configs])


# ─── Tier 1: CI quick (PackGQA GQA ratios) ───────────────────────────────────
@requires_sm100
@_cfgs(
    [
        # ratio=128, kBlockM=128 — canonical DiT
        {"name": "mqa128_packgqa", "S": 256, "NHQ": 128, "NHK": 1, "topk": 128},
        # ratio=64, kBlockM=64 full fill
        {"name": "mqa64_packgqa", "S": 256, "NHQ": 64, "NHK": 1, "topk": 128},
        # ratio=32, kBlockM=64 half fill
        {"name": "mqa32_packgqa", "S": 256, "NHQ": 32, "NHK": 1, "topk": 128},
        # ratio=16, small Q tile
        {"name": "mqa16_packgqa", "S": 256, "NHQ": 16, "NHK": 1, "topk": 128},
    ]
)
def test_simple_index_sparse(config):
    _run_config(config)


# ─── Tier 2a: Cross-batch (uniform topk; SM90 uses variable) ─────────────────
@requires_sm100
@_cfgs(
    [
        {"name": "mqa128_B2", "B": 2, "S": 256, "NHQ": 128, "NHK": 1, "topk": 128},
        {"name": "mqa128_B3", "B": 3, "S": 256, "NHQ": 128, "NHK": 1, "topk": 128},
        {"name": "mqa128_B8", "B": 8, "S": 256, "NHQ": 128, "NHK": 1, "topk": 128},
    ]
)
def test_sparse_cross_batch(config):
    _run_config(config)


# ─── Tier 2b: Q/KV different lengths ─────────────────────────────────────────
@requires_sm100
@_cfgs(
    [
        {
            "name": "short_q_long_kv",
            "S_q": 64,
            "S_kv": 1024,
            "NHQ": 128,
            "NHK": 1,
            "topk": 128,
        },
        {"name": "tiny_q", "S_q": 8, "S_kv": 512, "NHQ": 128, "NHK": 1, "topk": 128},
        {
            "name": "unaligned_q",
            "S_q": 100,
            "S_kv": 512,
            "NHQ": 128,
            "NHK": 1,
            "topk": 128,
        },
    ]
)
def test_sparse_qkv_lengths(config):
    _run_config(config)


# ─── Tier 3a: Head dim variants ──────────────────────────────────────────────
@requires_sm100
@_cfgs(
    [
        {"name": "D64", "S": 256, "NHQ": 128, "NHK": 1, "D": 64, "topk": 128},
        {"name": "D128", "S": 256, "NHQ": 128, "NHK": 1, "D": 128, "topk": 128},
    ]
)
def test_sparse_head_dim(config):
    _run_config(config)


# ─── Tier 3b: Long sequence ──────────────────────────────────────────────────
@requires_sm100
@_cfgs(
    [
        {"name": "mqa128_long_seq", "S": 8192, "NHQ": 128, "NHK": 1, "topk": 1024},
        {"name": "mqa16_long_seq", "S": 8192, "NHQ": 16, "NHK": 1, "topk": 1024},
    ]
)
def test_sparse_long_seq(config):
    _run_config(config)


# ─── Tier 3c: GQA (NHK>1, NHQ>NHK) ───────────────────────────────────────────
@requires_sm100
@_cfgs(
    [
        {
            "name": "gqa64x2_packgqa",
            "S": 256,
            "NHQ": 128,
            "NHK": 2,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa64x2_no_packgqa",
            "S": 256,
            "NHQ": 128,
            "NHK": 2,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa4x4_packgqa",
            "S": 256,
            "NHQ": 16,
            "NHK": 4,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa8x2_packgqa",
            "S": 256,
            "NHQ": 16,
            "NHK": 2,
            "topk": 128,
            "pack_gqa": True,
        },
    ]
)
def test_sparse_gqa(config):
    _run_config(config)


# ─── Tier 3d: MHA (NHQ==NHK, multi-KV-head) ──────────────────────────────────
@requires_sm100
@_cfgs(
    [
        {"name": "mha4", "S": 256, "NHQ": 4, "NHK": 4, "topk": 128, "pack_gqa": False},
        {
            "name": "mha16",
            "S": 256,
            "NHQ": 16,
            "NHK": 16,
            "topk": 128,
            "pack_gqa": False,
        },
    ]
)
def test_sparse_mha(config):
    _run_config(config)


# ─── Tier P: SM100-only partial topk (topk % 128 != 0) ───────────────────────
@requires_sm100
@_cfgs(
    [
        {
            "name": "partial_256_512_192",
            "S_q": 256,
            "S_kv": 512,
            "NHQ": 8,
            "NHK": 1,
            "topk": 192,
        },
        {
            "name": "partial_256_512_448",
            "S_q": 256,
            "S_kv": 512,
            "NHQ": 8,
            "NHK": 1,
            "topk": 448,
        },
        {
            "name": "partial_128_1024_320",
            "S_q": 128,
            "S_kv": 1024,
            "NHQ": 8,
            "NHK": 1,
            "topk": 320,
        },
        {
            "name": "partial_512_2048_576",
            "S_q": 512,
            "S_kv": 2048,
            "NHQ": 8,
            "NHK": 1,
            "topk": 576,
        },
        {
            "name": "partial_1024_4096_960",
            "S_q": 1024,
            "S_kv": 4096,
            "NHQ": 8,
            "NHK": 1,
            "topk": 960,
        },
    ]
)
def test_partial_topk_multi_niter(config):
    """SM100-only: partial last block (topk % 128 != 0, mask_block_cnt > 1).

    Regression for the block-sparse softmax mask-ordering fix: softmax consumes
    S-tiles in ascending block order (the reverse of the load order), so the
    per-step mask label must also be ascending and mask_seqlen must be applied on
    every mask step, otherwise the seqlen_k / padding mask lands on the wrong
    physical block and the partial tail is left unmasked. SM90 forbids this case
    (max_topk must be a multiple of the tile size).
    """
    _run_config(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
