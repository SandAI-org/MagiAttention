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

"""CuTeDSL SM100 IndexSparse (token-level scatter) sweep tests.

Drives the CuTeDSL kernel-level entry points directly
(``_flex_flash_attn_fwd`` / ``_flex_flash_attn_bwd`` +
``prepare_index_sparse_tiles``) and validates against a dense fp32 SDPA
reference.

Sweep structure (mirrors ``tests/test_attn/test_index_sparse.py``):

  * ``test_index_sparse_classic_sweep``
        Q_SEQLENS × KV_SEQLENS × TOPKS cross-product.
        Fixed: MQA128, D=128, PackGQA=True, kbs=1.

  * ``test_index_sparse_comprehensive_sweep``
        head_config × head_dim × sparse_k_block_size.
        8 head configs (MQA / GQA / MHA), D ∈ {64, 128}, kbs ∈ {1, 128}.
        Skips invalid combos (kbs=128 requires NHK=1 and D=128).

  * ``test_partial_topk_sweep``
        SM100-only partial topk (topk % 128 ≠ 0), scatter mode only (kbs=1).

CuTeDSL differences from SM90:
  * Uses ``_flex_flash_attn_fwd`` / ``_flex_flash_attn_bwd`` directly
  * Uses ``prepare_index_sparse_tiles()`` for tile preparation
  * No ``inner_dir`` / ``inner_load_mode`` / ``inner_store_mode`` env vars
  * BWD requires ``swap_bwd_qk_loop=True`` and ``m_block_size=128``

Index generation uses ``build_index_sparse_indices`` from the shared
``magi_attention.utils.sparse_utils`` module (SM90 format: global logical
positions, per-query) and adapts to CuTeDSL format
(``(B, NHQ, SQ, topk)`` local IDs with per-tile sharing) via
``_adapt_indices``.
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
from magi_attention.utils.sparse_utils import build_index_sparse_indices

SEED = 42

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="IndexSparse cutedsl path requires SM100+",
)

_FWD_ATOL, _FWD_RTOL = 0.01, 0.05
_BWD_DQ_ATOL, _BWD_DQ_RTOL = 0.02, 0.3
_BWD_DKV_ATOL, _BWD_DK_RTOL, _BWD_DV_RTOL = 0.02, 0.15, 0.05
_MISMATCH_THRES = 0.01


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _fwd_sparse_m_block(seqlen_q: int, qhpk: int, pack_gqa: bool) -> int:
    """FWD sparse Q-tile rows: q_stage=2 doubles the 128-row tile when the
    packed Q extent exceeds one tile (mirrors _flex_flash_attn_fwd)."""
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


def _adapt_indices(
    sm90_indices: torch.Tensor,
    *,
    B: int,
    SQ: int,
    SK: int,
    NHQ: int,
    NHK: int,
    tokens_per_block: int,
    sparse_k_block_size: int = 1,
) -> torch.Tensor:
    """Convert ``build_index_sparse_indices`` output to CuTeDSL format.

    SM90 format:  ``(B*SQ, NHK, topk)`` int32 with global logical positions
                  (``b * num_kv_blocks + local_id``).
    CuTeDSL format: ``(B, NHQ, SQ, topk)`` int32 with local IDs and per-tile
                    index sharing (all queries in one FWD Q-tile share the same
                    K index set).
    """
    topk = sm90_indices.shape[-1]
    device = sm90_indices.device
    num_kv_blocks = SK // sparse_k_block_size
    qhpk = NHQ // NHK

    # (B*SQ, NHK, topk) → (B, SQ, NHK, topk), then global → local
    local = sm90_indices.reshape(B, SQ, NHK, topk).clone()
    batch_offsets = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1, 1, 1)
        * num_kv_blocks
    )
    valid = local >= 0
    local = torch.where(valid, local - batch_offsets, torch.full_like(local, -1))

    # Per-tile sharing: map each query to the first query in its FWD tile
    first_in_tile = (
        torch.arange(SQ, device=device) // tokens_per_block * tokens_per_block
    )
    local = local[:, first_in_tile, :, :]

    # Expand NHK → NHQ: replicate for each Q head in the KV group
    expanded = (
        local.unsqueeze(3).expand(B, SQ, NHK, qhpk, topk).reshape(B, SQ, NHQ, topk)
    )
    return expanded.permute(0, 2, 1, 3).contiguous()


def _sdpa_ref_fwd_bwd(q, k, v, dO, indices, softmax_scale, sparse_k_block_size=1):
    """Vectorized fp32 SDPA reference with token-level index mask.

    Args:
        indices: ``(B, NHQ, SQ, topk)`` — local token IDs when
            ``sparse_k_block_size == 1``, or local block IDs otherwise.

    Returns:
        ``(O, dQ, dK, dV)`` in bf16.
    """
    B, SQ, NHQ, HD = q.shape
    _, SK, NHK, _ = k.shape
    HDV = v.shape[-1]
    qhpk = NHQ // NHK
    device = q.device

    q_f = q.float().detach().requires_grad_(True)
    k_f = k.float().detach().requires_grad_(True)
    v_f = v.float().detach().requires_grad_(True)

    k_exp = k_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HD).reshape(B, SK, NHQ, HD)
    v_exp = v_f.unsqueeze(3).expand(B, SK, NHK, qhpk, HDV).reshape(B, SK, NHQ, HDV)

    scores = torch.einsum("bsnh,btnh->bnst", q_f, k_exp) * softmax_scale

    if sparse_k_block_size == 1:
        mask = torch.zeros(B, NHQ, SQ, SK, dtype=torch.bool, device=device)
        mask.scatter_(3, indices.long(), True)
    else:
        num_kv_blocks = SK // sparse_k_block_size
        blk_int = torch.zeros(
            B, NHQ, SQ, num_kv_blocks, dtype=torch.int32, device=device
        )
        safe_idx = indices.clamp(min=0).long()
        blk_int.scatter_add_(3, safe_idx, (indices >= 0).int())
        blk_mask = blk_int > 0
        mask = (
            blk_mask.unsqueeze(-1)
            .expand(B, NHQ, SQ, num_kv_blocks, sparse_k_block_size)
            .reshape(B, NHQ, SQ, num_kv_blocks * sparse_k_block_size)
        )
        if mask.shape[-1] > SK:
            mask = mask[:, :, :, :SK]

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


def _run_config(device: str, cfg: dict):
    """Run one FWD + BWD IndexSparse config and assert against SDPA reference.

    Config dict keys:
        B, SQ, SK, NHQ, NHK, D, topk, pack_gqa, sparse_k_block_size
    ``topk`` is always token-level; when ``sparse_k_block_size > 1`` the block
    count passed to the index builder is ``topk // sparse_k_block_size``.
    """
    torch.manual_seed(SEED)

    B = cfg.get("B", 1)
    SQ = cfg["SQ"]
    SK = cfg["SK"]
    NHQ = cfg["NHQ"]
    NHK = cfg["NHK"]
    D = cfg.get("D", 128)
    topk = cfg["topk"]
    pack_gqa = cfg.get("pack_gqa", True)
    kbs = cfg.get("sparse_k_block_size", 1)

    qhpk = NHQ // NHK
    softmax_scale = D**-0.5

    q = torch.randn(B, SQ, NHQ, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, SK, NHK, D, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    dO = torch.randn_like(q)

    # Determine tile granularity for per-tile index sharing
    fwd_m_block = _fwd_sparse_m_block(SQ, qhpk, pack_gqa)
    tokens_per_block = max(fwd_m_block // (qhpk if pack_gqa else 1), 1)

    if kbs > 1:
        # build_index_sparse_indices generates block IDs; expand to token IDs
        # so that prepare_index_sparse_tiles (which stores token-level IDs)
        # and the SDPA reference see the same token-level mask.
        num_selected_blocks = topk // kbs
        sm90_block_ids = build_index_sparse_indices(
            B,
            NHK,
            SQ,
            SK,
            num_selected_blocks,
            num_selected_blocks,
            device,
            sparse_k_block_size=kbs,
        )
        block_indices = _adapt_indices(
            sm90_block_ids,
            B=B,
            SQ=SQ,
            SK=SK,
            NHQ=NHQ,
            NHK=NHK,
            tokens_per_block=tokens_per_block,
            sparse_k_block_size=kbs,
        )
        # Expand block IDs → contiguous token IDs within each block
        offsets = torch.arange(kbs, device=device, dtype=torch.int32)
        valid = block_indices >= 0
        expanded = block_indices.unsqueeze(-1) * kbs + offsets
        expanded = torch.where(
            valid.unsqueeze(-1),
            expanded,
            torch.full_like(expanded, -1),
        )
        indices = expanded.reshape(B, NHQ, SQ, topk)
    else:
        sm90_indices = build_index_sparse_indices(
            B,
            NHK,
            SQ,
            SK,
            topk,
            topk,
            device,
            sparse_k_block_size=kbs,
        )
        indices = _adapt_indices(
            sm90_indices,
            B=B,
            SQ=SQ,
            SK=SK,
            NHQ=NHQ,
            NHK=NHK,
            tokens_per_block=tokens_per_block,
            sparse_k_block_size=kbs,
        )

    # Reference FWD + BWD (always token-level indices after expansion)
    o_ref, dq_ref, dk_ref, dv_ref = _sdpa_ref_fwd_bwd(
        q,
        k,
        v,
        dO,
        indices,
        softmax_scale,
    )

    # ── FWD ──
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
        sparse_k_block_size=kbs,
    )
    out, lse = _flex_flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        flex_attn_args=TorchFlexAttnArgs(index_sparse_tiles=fwd_tiles),
        pack_gqa=pack_gqa,
    )

    # ── BWD (LoopK, m_block_size=128) ──
    bwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=B,
        seqlen_q=SQ,
        seqlen_k=SK,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=128,
        n_block_size=128,
        pack_gqa=pack_gqa,
        sparse_k_block_size=kbs,
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

    # ── Check ──
    tc = (
        f"B={B},NHQ={NHQ},NHK={NHK},D={D},SQ={SQ},SK={SK},topk={topk},"
        f"pack_gqa={pack_gqa},kbs={kbs},"
        f"atomic={os.environ.get('MAGI_ATTENTION_FFA_CUTEDSL_IS_SCATTER_ATOMIC', '0')}"
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


# ═════════════════════════════════════════════════════════════════════
# Sweep tests
# ═════════════════════════════════════════════════════════════════════


@requires_sm100
def test_index_sparse_classic_sweep():
    """Classic sweep: Q_SEQLENS × KV_SEQLENS × TOPKS, MQA128 D=128 PackGQA kbs=1."""
    Q_SEQLENS = [128, 512, 1024]
    KV_SEQLENS = [512, 1024, 4096]
    TOPKS = [128, 256]
    device = "cuda"

    configs = []
    for sq in Q_SEQLENS:
        for sk in KV_SEQLENS:
            for topk in TOPKS:
                if topk > sk:
                    continue
                configs.append(
                    dict(
                        B=1,
                        SQ=sq,
                        SK=sk,
                        NHQ=128,
                        NHK=1,
                        D=128,
                        topk=topk,
                        pack_gqa=True,
                        sparse_k_block_size=1,
                    )
                )

    for i, cfg in enumerate(configs, 1):
        print(
            f"Testing config {i}/{len(configs)}: "
            f"SQ={cfg['SQ']}, SK={cfg['SK']}, topk={cfg['topk']}"
        )
        _run_config(device, cfg)


@requires_sm100
def test_index_sparse_comprehensive_sweep():
    """Comprehensive sweep: head_config × D × sparse_k_block_size."""
    HEAD_CONFIGS = [
        # (NHQ, NHK, pack_gqa)
        (128, 1, True),  # MQA128
        (64, 1, True),  # MQA64
        (32, 1, True),  # MQA32
        (16, 1, True),  # MQA16
        (4, 4, True),  # GQA 4:4 (effectively MHA4)
        (8, 2, True),  # GQA 8:2
        (4, 1, True),  # MQA4
        (16, 16, True),  # MHA16
    ]
    DIMS = [64, 128]
    KBS_LIST = [1, 128]
    device = "cuda"

    configs = []
    for nhq, nhk, pack_gqa in HEAD_CONFIGS:
        for D in DIMS:
            for kbs in KBS_LIST:
                if kbs > 1 and (nhk > 1 or D != 128):
                    continue
                if kbs == 1:
                    sq, sk, topk = 256, 512, 128
                else:
                    sq, sk, topk = 256, 4096, 256
                if topk % kbs != 0:
                    continue
                configs.append(
                    dict(
                        B=1,
                        SQ=sq,
                        SK=sk,
                        NHQ=nhq,
                        NHK=nhk,
                        D=D,
                        topk=topk,
                        pack_gqa=pack_gqa,
                        sparse_k_block_size=kbs,
                    )
                )

    for i, cfg in enumerate(configs, 1):
        print(
            f"Testing config {i}/{len(configs)}: "
            f"NHQ={cfg['NHQ']}, NHK={cfg['NHK']}, D={cfg['D']}, "
            f"kbs={cfg['sparse_k_block_size']}"
        )
        _run_config(device, cfg)


@requires_sm100
def test_partial_topk_sweep():
    """SM100-only: topk not a multiple of 128 (scatter mode, kbs=1 only)."""
    PARTIAL_CONFIGS = [
        dict(SQ=256, SK=512, topk=192),
        dict(SQ=256, SK=512, topk=448),
        dict(SQ=128, SK=1024, topk=320),
        dict(SQ=512, SK=2048, topk=576),
        dict(SQ=1024, SK=4096, topk=960),
    ]
    device = "cuda"

    configs = []
    for pc in PARTIAL_CONFIGS:
        configs.append(
            dict(
                B=1,
                NHQ=8,
                NHK=1,
                D=128,
                pack_gqa=True,
                sparse_k_block_size=1,
                **pc,
            )
        )

    for i, cfg in enumerate(configs, 1):
        print(
            f"Testing config {i}/{len(configs)}: "
            f"SQ={cfg['SQ']}, SK={cfg['SK']}, topk={cfg['topk']}"
        )
        _run_config(device, cfg)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
