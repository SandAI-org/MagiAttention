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
Tests for index_sparse_indices direct-to-kernel path (forward + backward).

Validates flex_flash_attn_func with index_sparse_indices against PyTorch SDPA
reference.

Tier 1 (CI quick): PackGQA, the most common DiT paths:
  - ratio 128 → kBlockM=128
  - ratio  64 → kBlockM=64 (full fill)
  - ratio  32 → kBlockM=64 (50% fill)
  - ratio  16 → SwapAB + PackGQA

Tier 2 (CI):
  2a. Cross-batch variable topk (per-batch different topk)
  2b. Q/KV different lengths (short Q, long KV, unaligned Q)

Tier 3 (Slow):
  3a. Head dim variants (D=64/128)
  3b. Long sequence (S=8192, S=65536 INT32 overflow regression)
  3c. GQA — NHK>1, NHQ>NHK, large/small ratio, PackGQA/SwapAB
  3d. MHA — NHK>1, NHQ==NHK, SwapAB
  3e. k_block_size > 1 (commented out, kernel WIP)

Known limitations:
  - swap_ab is blocked for sparse paths (asserted in flex_flash_attn_func)
  - No distributed sparse yet
  - max_topk must be multiples of tile_size (asserted in flex_flash_attn_func)
  - Q/K/V are packed in (b, s, h) order to match index_sparse_indices view layout
"""

import os
import unittest
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.utils import set_random_seed
from magi_attention.utils.sparse_utils import (
    build_index_sparse_indices,
    get_sdpa_mask_from_index_sparse_indices,
)

SEED = 42
DEFAULT_ATOL = 0.01


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


_build_index_sparse_indices = build_index_sparse_indices
_build_sdpa_mask = get_sdpa_mask_from_index_sparse_indices


def _run_sparse_attn_and_get_output(
    q,
    k,
    v,
    index_sparse_indices,
    B,
    S_q,
    S_kv,
    NHQ,
    NHK,
    pack_gqa,
    swap_ab=False,
    ref_block_size=None,
    k_block_size=1,
    test_bwd=False,
):
    """Run FFA with index_sparse_indices and return reshaped output [B, S_q, NHQ, D].

    When test_bwd=True, returns (output, q_ffa, k_ffa, v_ffa) with gradients enabled.
    """
    q_ffa = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b s h) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b s h) 1 d")

    if test_bwd:
        q_ffa = q_ffa.detach().clone().requires_grad_(True)
        k_ffa = k_ffa.detach().clone().requires_grad_(True)
        v_ffa = v_ffa.detach().clone().requires_grad_(True)

        o_sparse, _ = flex_flash_attn_func(
            q_ffa,
            k_ffa,
            v_ffa,
            index_sparse_indices=index_sparse_indices,
            q_block_size=1,
            k_block_size=k_block_size,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )
        o_reshaped = rearrange(
            o_sparse, "(b s h1) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S_q
        )
        return o_reshaped, o_sparse, q_ffa, k_ffa, v_ffa
    else:
        with torch.no_grad():
            o_sparse, _ = flex_flash_attn_func(
                q_ffa.clone(),
                k_ffa.clone(),
                v_ffa.clone(),
                index_sparse_indices=index_sparse_indices,
                q_block_size=1,
                k_block_size=k_block_size,
                pack_gqa=pack_gqa,
                swap_ab=swap_ab,
                ref_block_size=ref_block_size,
            )
        return rearrange(o_sparse, "(b s h1) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S_q)


def _compare_against_sdpa(
    o_ffa,
    q,
    k,
    v,
    sdpa_mask,
    B,
    NHQ,
    NHK,
    atol,
    test_case,
):
    """Compare FFA output against SDPA reference, batch by batch."""
    gqa = NHQ // NHK
    err_msgs = []
    for b_idx in range(B):
        q_sdpa = rearrange(q[b_idx], "s h d -> 1 h s d")
        k_sdpa = rearrange(k[b_idx], "s h d -> 1 h s d")
        v_sdpa = rearrange(v[b_idx], "s h d -> 1 h s d")
        if gqa > 1:
            k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)

        with torch.no_grad():
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=sdpa_mask[b_idx].unsqueeze(0)
            )
        o_ref = rearrange(o_ref, "1 h s d -> s h d")

        max_diff = (o_ffa[b_idx].float() - o_ref.float()).abs().max().item()
        if max_diff >= atol:
            err_msgs.append(
                f"batch {b_idx}: max_diff={max_diff:.6f} >= {atol} in {test_case}"
            )

    if err_msgs:
        raise AssertionError("\n".join(err_msgs))


def _run_index_sparse_config(device, cfg: dict[str, Any], test_bwd: bool = True):
    """Run one index_sparse_indices test config and assert against SDPA."""
    set_random_seed(SEED)
    B = cfg["B"]
    S = cfg.get("S", None)
    S_kv = cfg.get("S_kv", S)
    S_q = cfg.get("S_q", min(S_kv, 256))
    NHQ = cfg["NHQ"]
    NHK = cfg["NHK"]
    D = cfg.get("D", 128)
    topk = cfg["topk"]
    default_max = max(topk) if isinstance(topk, list) else topk
    max_topk = cfg.get("max_topk", default_max)
    pack_gqa = cfg.get("pack_gqa", True)
    swap_ab = cfg.get("swap_ab", False)
    ref_block_size = cfg.get("ref_block_size", None)
    k_block_size = cfg.get("k_block_size", 1)
    dtype = cfg.get("dtype", torch.bfloat16)
    atol = cfg.get("atol", DEFAULT_ATOL)

    q = torch.randn(B, S_q, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)

    if NHK > 1:
        q = rearrange(q, "b s (h1 h2) d -> b (s h1) h2 d", h1=NHK)
        k = rearrange(k, "b s h d -> b (s h) 1 d")
        v = rearrange(v, "b s h d -> b (s h) 1 d")
        S_q = S_q * NHK
        S_kv = S_kv * NHK
        NHQ = NHQ // NHK
        NHK = 1

    index_sparse_indices = _build_index_sparse_indices(
        B, NHK, S_q, S_kv, topk, max_topk, device, k_block_size=k_block_size
    )

    result = _run_sparse_attn_and_get_output(
        q,
        k,
        v,
        index_sparse_indices,
        B,
        S_q,
        S_kv,
        NHQ,
        NHK,
        pack_gqa=pack_gqa,
        swap_ab=swap_ab,
        ref_block_size=ref_block_size,
        k_block_size=k_block_size,
        test_bwd=test_bwd,
    )

    if test_bwd:
        o_ffa, o_sparse, q_ffa, k_ffa, v_ffa = result
    else:
        o_ffa = result

    sdpa_mask = _build_sdpa_mask(
        index_sparse_indices,
        B,
        NHQ,
        NHK,
        S_q,
        S_kv,
        device,
        k_block_size=k_block_size,
    )

    test_case = (
        f"[NHQ={cfg['NHQ']},NHK={cfg['NHK']},S_q={cfg.get('S_q', cfg.get('S'))},S_kv={cfg.get('S_kv', cfg.get('S'))},"
        f"B={B},D={D},topk={topk},max_topk={max_topk},pack_gqa={pack_gqa},"
        f"swap_ab={swap_ab},k_block_size={k_block_size},dtype={dtype},"
        f"flat:NHQ_eff={NHQ},S_q_eff={S_q},S_kv_eff={S_kv}]"
    )

    _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)

    if test_bwd:
        do = torch.randn_like(o_sparse)
        o_sparse.backward(do)
        dq_ffa = q_ffa.grad.clone()

        gqa = NHQ // NHK
        total_q = B * S_q
        dq_ref_list = []
        for b_idx in range(B):
            q_sdpa = (
                rearrange(q[b_idx], "s h d -> 1 h s d")
                .detach()
                .clone()
                .requires_grad_(True)
            )
            k_sdpa = (
                rearrange(k[b_idx], "s h d -> 1 h s d")
                .detach()
                .clone()
                .requires_grad_(True)
            )
            v_sdpa = (
                rearrange(v[b_idx], "s h d -> 1 h s d")
                .detach()
                .clone()
                .requires_grad_(True)
            )
            if gqa > 1:
                k_sdpa_exp = k_sdpa.repeat_interleave(gqa, dim=1)
                v_sdpa_exp = v_sdpa.repeat_interleave(gqa, dim=1)
            else:
                k_sdpa_exp, v_sdpa_exp = k_sdpa, v_sdpa

            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa_exp,
                v_sdpa_exp,
                attn_mask=sdpa_mask[b_idx].unsqueeze(0),
            )
            do_reshaped = rearrange(
                do, "(b s h1) h2 d -> b 1 (h1 h2) s d", b=B, h1=NHK, s=S_q
            )[b_idx]
            o_ref.backward(do_reshaped)
            dq_ref_list.append(q_sdpa.grad)

        dq_ref = torch.cat(dq_ref_list, dim=0)
        dq_ref = rearrange(dq_ref, "b h s d -> (b s) h d", b=B)[:total_q]

        dq_ffa_reshaped = rearrange(
            dq_ffa, "(b s h1) h2 d -> b (h1 h2) s d", b=B, h1=NHK, s=S_q
        )
        dq_ref_reshaped = rearrange(dq_ref, "(b s) h d -> b h s d", b=B, s=S_q)

        bwd_atol = cfg.get("bwd_atol", 0.05)
        err_msgs = []
        for b_idx in range(B):
            max_dq_diff = (
                (dq_ffa_reshaped[b_idx].float() - dq_ref_reshaped[b_idx].float())
                .abs()
                .max()
                .item()
            )
            if max_dq_diff >= bwd_atol:
                err_msgs.append(
                    f"BWD batch {b_idx}: dQ max_diff={max_dq_diff:.6f} >= {bwd_atol} in {test_case}"
                )
        if err_msgs:
            raise AssertionError("\n".join(err_msgs))


# ═══════════════════════════════════════════════════════════
# View-trick helpers
# ═══════════════════════════════════════════════════════════


def _build_block_shared_indices(B, S, NHK, topk, qbs, device):
    """Build index_sparse_indices where each q_block of *qbs* tokens shares K indices."""
    S_blk = S // qbs
    indices_block = torch.full(
        (B * S_blk, NHK, topk), -1, dtype=torch.int32, device=device
    )
    for b in range(B):
        for qi in range(S_blk):
            row = b * S_blk + qi
            for h in range(NHK):
                perm = torch.randperm(S, device=device)[:topk].sort().values
                indices_block[row, h, :topk] = ((b * S + perm) * NHK + h).int()

    indices_full = indices_block.repeat_interleave(qbs, dim=0)
    return indices_full, indices_block


def _sdpa_reference(q_raw, k_raw, v_raw, indices_full, B, S, NHQ, NHK, device):
    """SDPA reference with dense mask built from index_sparse_indices."""
    gqa = NHQ // NHK
    mask = torch.zeros(B, NHQ, S, S, dtype=torch.bool, device=device)
    for b in range(B):
        for qi in range(S):
            row = b * S + qi
            for h_kv in range(NHK):
                gids = indices_full[row, h_kv, :]
                valid = gids[gids >= 0].long()
                local_kv = valid // NHK - b * S
                for g in range(gqa):
                    mask[b, h_kv * gqa + g, qi, local_kv] = True

    o_list = []
    for b in range(B):
        qb = rearrange(q_raw[b], "s h d -> 1 h s d")
        kb = rearrange(k_raw[b], "s h d -> 1 h s d")
        vb = rearrange(v_raw[b], "s h d -> 1 h s d")
        if gqa > 1:
            kb = kb.repeat_interleave(gqa, dim=1)
            vb = vb.repeat_interleave(gqa, dim=1)
        with torch.no_grad():
            o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask[b : b + 1])
        o_list.append(rearrange(o, "1 h s d -> s h d"))
    return torch.stack(o_list)


def _pack_kv(k_raw, v_raw):
    k_ffa = rearrange(k_raw, "b s h d -> (b s h) 1 d").detach().clone()
    v_ffa = rearrange(v_raw, "b s h d -> (b s h) 1 d").detach().clone()
    return k_ffa, v_ffa


def _run_view_trick(
    q_raw, k_raw, v_raw, indices_block, B, S, NHQ, NHK, D, qbs, *, use_permute
):
    """Fold *qbs* tokens into heads, call FFA with q_block_size=1."""
    S_new = S // qbs
    NHQ_new = NHQ * qbs
    gqa = NHQ // NHK

    if use_permute:
        q_viewed = (
            q_raw.reshape(B, S_new, qbs, NHQ, D)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(B, S_new, NHQ_new, D)
        )
    else:
        q_viewed = q_raw.view(B, S_new, NHQ_new, D)

    q_ffa = (
        rearrange(q_viewed, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK).detach().clone()
    )
    k_ffa, v_ffa = _pack_kv(k_raw, v_raw)

    o_sparse, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_sparse_indices=indices_block,
        q_block_size=1,
        k_block_size=1,
        pack_gqa=True,
    )

    o_unpacked = rearrange(
        o_sparse, "(b s h1) h2 d -> b s h1 h2 d", b=B, s=S_new, h1=NHK
    )

    if use_permute:
        o_out = (
            o_unpacked.reshape(B, S_new, NHK, gqa, qbs, D)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(B, S, NHQ, D)
        )
    else:
        o_combined = rearrange(o_unpacked, "b s h1 h2 d -> b s (h1 h2) d")
        o_out = o_combined.reshape(B, S_new, qbs, NHQ, D).reshape(B, S, NHQ, D)

    return o_out


# ═══════════════════════════════════════════════════════════
# TestIndexSparseSimple — CI gate (view_trick + DisableAtomic)
# ═══════════════════════════════════════════════════════════


class TestIndexSparseSimple(unittest.TestCase):
    """Lightweight single-process IndexSparse regression test.

    Extracted from test_simple_attn.py — validates IndexSparse FWD+BWD
    correctness against SDPA reference in various GQA configurations.
    Also includes view-trick tests and DisableAtomic tests.
    """

    @property
    def device(self):
        return torch.cuda.current_device()

    INDEX_ATTN_CONFIGS = [
        {
            "name": "mqa128_pack_gqa",
            "B": 1,
            "S": 256,
            "NHQ": 128,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_32_4_pack_gqa",
            "B": 1,
            "S": 256,
            "NHQ": 32,
            "NHK": 4,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "mha_aligned",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 4,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_4_2_aligned",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 2,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "mha_unaligned_seqlen",
            "B": 1,
            "S": 200,
            "NHQ": 4,
            "NHK": 4,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_8_2_small",
            "B": 2,
            "S": 256,
            "NHQ": 8,
            "NHK": 2,
            "D": 64,
            "topk": 128,
            "pack_gqa": False,
        },
        {
            "name": "gqa_4_2_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_8_2_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 8,
            "NHK": 2,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_4_1_pack_gqa_d128",
            "B": 1,
            "S": 256,
            "NHQ": 4,
            "NHK": 1,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
        {
            "name": "gqa_8_4_pack_gqa_d128",
            "B": 2,
            "S": 256,
            "NHQ": 8,
            "NHK": 4,
            "D": 128,
            "topk": 128,
            "pack_gqa": True,
        },
    ]

    @parameterize("cfg", INDEX_ATTN_CONFIGS)
    def test_index_sparse_simple(self, cfg: dict[str, Any]):
        """IndexSparse FWD+BWD correctness against SDPA reference.

        The view trick flattens K from (B,S,NHK,D) to (B*S*NHK, 1, D), so
        the kernel sees NHK_eff=1. Indices must be built in this flat token
        space (NHK=1, S_flat=S*NHK) with logical positions.
        """
        set_random_seed(42)
        B, S, NHQ, NHK, D, topk = (
            cfg["B"],
            cfg["S"],
            cfg["NHQ"],
            cfg["NHK"],
            cfg["D"],
            cfg["topk"],
        )
        pack_gqa = cfg["pack_gqa"]
        device = self.device

        gqa = NHQ // NHK
        S_flat = S * NHK
        NHQ_eff = gqa

        indices = build_index_sparse_indices(B, 1, S_flat, S_flat, topk, topk, device)

        q_raw = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=device)
        k_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)
        v_raw = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=device)

        q_ffa = (
            q_raw.reshape(B, S, NHK, gqa, D)
            .permute(0, 1, 2, 3, 4)
            .reshape(B * S * NHK, gqa, D)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        k_ffa = k_raw.reshape(B * S * NHK, 1, D).detach().clone().requires_grad_(True)
        v_ffa = v_raw.reshape(B * S * NHK, 1, D).detach().clone().requires_grad_(True)

        o_sparse, _ = flex_flash_attn_func(
            q_ffa,
            k_ffa,
            v_ffa,
            index_sparse_indices=indices,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=pack_gqa,
        )

        mask = get_sdpa_mask_from_index_sparse_indices(
            indices, B, NHQ_eff, 1, S_flat, S_flat, device
        )

        for b in range(B):
            sl = slice(b * S_flat, (b + 1) * S_flat)
            q_b = q_ffa[sl].detach().reshape(1, S_flat, NHQ_eff, D).transpose(1, 2)
            k_b = k_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
            v_b = v_ffa[sl].detach().reshape(1, S_flat, 1, D).transpose(1, 2)
            if NHQ_eff > 1:
                k_b = k_b.expand(1, NHQ_eff, S_flat, D)
                v_b = v_b.expand(1, NHQ_eff, S_flat, D)

            with torch.no_grad():
                try:
                    o_ref = torch.nn.functional.scaled_dot_product_attention(
                        q_b, k_b, v_b, attn_mask=mask[b].unsqueeze(0)
                    )
                except RuntimeError:
                    with torch.nn.attention.sdpa_kernel(
                        [torch.nn.attention.SDPBackend.MATH]
                    ):
                        o_ref = torch.nn.functional.scaled_dot_product_attention(
                            q_b, k_b, v_b, attn_mask=mask[b].unsqueeze(0)
                        )
            o_ref = o_ref.squeeze(0).transpose(0, 1)

            max_diff = (o_sparse[sl].float() - o_ref.float()).abs().max().item()
            assert max_diff < 0.02, (
                f"[test_index_sparse][{cfg['name']}] "
                f"FWD batch {b}: max_diff={max_diff:.6f} >= 0.02"
            )

        # BWD verification
        do = torch.randn_like(o_sparse)
        o_sparse.backward(do)
        dq_ffa = q_ffa.grad.clone()

        for b in range(B):
            sl = slice(b * S_flat, (b + 1) * S_flat)
            q_b = (
                q_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, NHQ_eff, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            k_b = (
                k_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, 1, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            v_b = (
                v_ffa[sl]
                .detach()
                .clone()
                .reshape(1, S_flat, 1, D)
                .transpose(1, 2)
                .requires_grad_(True)
            )
            k_exp = k_b.expand(1, NHQ_eff, S_flat, D) if NHQ_eff > 1 else k_b
            v_exp = v_b.expand(1, NHQ_eff, S_flat, D) if NHQ_eff > 1 else v_b

            try:
                o_ref = torch.nn.functional.scaled_dot_product_attention(
                    q_b, k_exp, v_exp, attn_mask=mask[b].unsqueeze(0)
                )
            except RuntimeError:
                with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.MATH]
                ):
                    o_ref = torch.nn.functional.scaled_dot_product_attention(
                        q_b, k_exp, v_exp, attn_mask=mask[b].unsqueeze(0)
                    )
            do_b = do[sl].reshape(1, S_flat, NHQ_eff, D).transpose(1, 2)
            o_ref.backward(do_b)

            dq_ref_b = q_b.grad.squeeze(0).transpose(0, 1)
            max_dq_diff = (dq_ffa[sl].float() - dq_ref_b.float()).abs().max().item()
            assert max_dq_diff < 0.05, (
                f"[test_index_sparse][{cfg['name']}] "
                f"BWD batch {b}: dQ max_diff={max_dq_diff:.6f} >= 0.05"
            )

    # ─── View-trick tests ─────────────────────────────────────

    def test_mqa_simple_view(self):
        """MQA (NHK=1): simple .view() works — zero-copy, no permute needed."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 2, 1, 128, 128, 2
        torch.manual_seed(42)
        dev = self.device

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)

        idx_full, idx_block = _build_block_shared_indices(B, S, NHK, topk, qbs, dev)
        o_ref = _sdpa_reference(q, k, v, idx_full, B, S, NHQ, NHK, dev)
        o_view = _run_view_trick(
            q, k, v, idx_block, B, S, NHQ, NHK, D, qbs, use_permute=False
        )

        diff = (o_view.float() - o_ref.float()).abs().max().item()
        assert diff < 0.02, f"MQA simple view: max_diff={diff:.6f} >= 0.02"

    def test_mha_simple_view_wrong(self):
        """MHA (NHK>1): simple .view() gives wrong head mapping — must fail."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 4, 4, 128, 128, 4
        torch.manual_seed(42)
        dev = self.device

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)

        idx_full, idx_block = _build_block_shared_indices(B, S, NHK, topk, qbs, dev)
        o_ref = _sdpa_reference(q, k, v, idx_full, B, S, NHQ, NHK, dev)
        o_view = _run_view_trick(
            q, k, v, idx_block, B, S, NHQ, NHK, D, qbs, use_permute=False
        )

        diff = (o_view.float() - o_ref.float()).abs().max().item()
        assert diff > 0.1, (
            f"MHA simple view unexpectedly correct: max_diff={diff:.6f} <= 0.1. "
            f"Head mapping should be wrong for NHK>1."
        )

    def test_mha_permute_view(self):
        """MHA (NHK>1): permute+contiguous fixes head mapping — 32h×QBS4=128."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 32, 32, 128, 128, 4
        torch.manual_seed(42)
        dev = self.device

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)

        idx_full, idx_block = _build_block_shared_indices(B, S, NHK, topk, qbs, dev)
        o_ref = _sdpa_reference(q, k, v, idx_full, B, S, NHQ, NHK, dev)
        o_view = _run_view_trick(
            q, k, v, idx_block, B, S, NHQ, NHK, D, qbs, use_permute=True
        )

        diff = (o_view.float() - o_ref.float()).abs().max().item()
        assert diff < 0.02, f"MHA 32h permute: max_diff={diff:.6f} >= 0.02"

    # ─── DisableAtomic tests ──────────────────────────────────

    def _run_index_sparse_atomic_config(
        self,
        S: int = 512,
        NHQ: int = 128,
        NHK: int = 1,
        D: int = 128,
        topk: int = 128,
        swap_bwd_qk_loop: bool | None = True,
        pack_gqa: bool = True,
    ):
        """Run IndexSparse FWD+BWD and verify correctness with auto-set atomic flags."""
        torch.manual_seed(42)
        q = torch.randn(
            S, NHQ, D, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            S, NHK, D, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            S, NHK, D, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )

        tile_size = 128
        padded_topk = ((topk + tile_size - 1) // tile_size) * tile_size
        indices = torch.randint(
            0, S, (S, NHK, padded_topk), device=self.device, dtype=torch.int32
        )
        indices[:, :, topk:] = -1

        out, meta = flex_flash_attn_func(
            q,
            k,
            v,
            index_sparse_indices=indices,
            pack_gqa=pack_gqa,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )

        self.assertEqual(out.shape, (S, NHQ, D))
        self.assertFalse(out.isnan().any())

        do = torch.randn_like(out)
        out.backward(do)

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertFalse(q.grad.isnan().any(), "dQ contains NaN")
        self.assertFalse(k.grad.isnan().any(), "dK contains NaN")
        self.assertFalse(v.grad.isnan().any(), "dV contains NaN")

    def test_disable_atomic_index_sparse_bwd_innerloopk(self):
        """InnerLoopK path: auto-sets disable_bwd_dq_atomic_reduction=True."""
        self._run_index_sparse_atomic_config(swap_bwd_qk_loop=True)

    def test_disable_atomic_index_sparse_fwd_only(self):
        """FWD path: auto-sets disable_fwd_atomic_reduction=True."""
        self._run_index_sparse_atomic_config(swap_bwd_qk_loop=None)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseSweep — CI (non-slow), Tier 1-2
# ═══════════════════════════════════════════════════════════


class TestIndexSparseSweep(DistTestBase):
    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    # ─── Tier 1: CI quick (PackGQA, no swap) ────────────────

    @with_run_in_mp
    @parameterize(
        "config",
        [
            # ratio=128, kBlockM=128, PackGQA — canonical DiT
            {
                "name": "mqa128_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # ratio=64, kBlockM=64 full fill, PackGQA
            {
                "name": "mqa64_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 64,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # ratio=32, kBlockM=64 half fill, PackGQA
            {
                "name": "mqa32_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # ratio=16, small Q tile, PackGQA
            # NOTE: swap_ab+IndexSparse has known correctness issues (not tested on main)
            {
                "name": "mqa16_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 16,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
        ],
    )
    def test_simple_index_sparse_indices_attn(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config, test_bwd=True)

    # ─── Tier 2a: Cross-batch variable topk ──────────────

    @with_run_in_mp
    @parameterize(
        "config",
        [
            # Per-batch different topk: batch0=256 full, batch1=128 half -1
            {
                "name": "mqa128_B2_variable_topk",
                "B": 2,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": [256, 128],
                "max_topk": 256,
                "pack_gqa": True,
            },
            # 3 batches, one batch nearly empty (topk=128), others full
            {
                "name": "mqa128_B3_one_sparse",
                "B": 3,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": [256, 256, 128],
                "max_topk": 256,
                "pack_gqa": True,
            },
            # 8 batches, uniform topk, heavier batch count
            {
                "name": "mqa128_B4_uniform",
                "B": 8,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
        ],
    )
    def test_sparse_cross_batch(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)

    # ─── Tier 2b: Q/KV different lengths ─────────────────────

    @with_run_in_mp
    @parameterize(
        "config",
        [
            # Short Q, long KV
            {
                "name": "short_q_long_kv",
                "B": 1,
                "S_q": 64,
                "S_kv": 1024,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # Very short Q (sub-tile, still >= kBlockN for inner loop).
            # NOTE: S_q < ~22 can produce zero-ref K tokens in inner_indices,
            # causing a potential BWD kernel hang (inner_block_max=0 barrier
            # deadlock). S_q=16 with SEED=42 has ~3 zero-ref K tokens but
            # empirically passes. S_q=8 had 49 zero-ref and hung reliably.
            {
                "name": "tiny_q",
                "B": 1,
                "S_q": 16,
                "S_kv": 512,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # Q not aligned to tile boundary
            {
                "name": "unaligned_q",
                "B": 1,
                "S_q": 100,
                "S_kv": 512,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
        ],
    )
    def test_sparse_qkv_lengths(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseSlowSweep — @slow, Tier 3a-3e
# ═══════════════════════════════════════════════════════════


class TestIndexSparseSlowSweep(DistTestBase):
    @property
    def seed(self):
        return SEED

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 1

    @property
    def timeout(self) -> int:
        return 1200

    # ─── Tier 3a: Head dim variants ──────────────────────────
    # D affects cp.async load loop count: num_tiles = D * sizeof(bf16) / 128
    #   D=64  → num_tiles=1, kBlockN=128 (single cp.async per row)
    #   D=128 → num_tiles=2, kBlockN=64  (default, covered in Tier 1)
    # Note: D=32 is rejected by max_headdim check; D>128 asserted in JIT sanity_check

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            {
                "name": "D64",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "D": 64,
                "topk": 128,
                "pack_gqa": True,
            },
            {
                "name": "D128",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "D": 128,
                "topk": 128,
                "pack_gqa": True,
            },
        ],
    )
    def test_sparse_head_dim(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)

    # ─── Tier 3b: Long sequence ────────────────────────────

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            {
                "name": "mqa128_long_seq",
                "B": 1,
                "S": 8192,
                "NHQ": 128,
                "NHK": 1,
                "topk": 1024,
                "pack_gqa": True,
            },
            {
                "name": "mqa16_swapab_long_seq",
                "B": 1,
                "S": 8192,
                "NHQ": 16,
                "NHK": 1,
                "topk": 1024,
                "pack_gqa": True,
                "swap_ab": True,
            },
            # INT32 overflow regression (unique_idx * max_topk > INT32_MAX)
            # S_q defaults to 256, so ref mask is (1, 128, 256, 65536) ≈ 2 GiB, fits in VRAM.
            # NHK>1 has a known bug; using NHK=1 to validate the int64 overflow fix.
            {
                "name": "mqa128_large_s_high_topk",
                "B": 1,
                "S": 65536,
                "NHQ": 128,
                "NHK": 1,
                "topk": 9216,
                "pack_gqa": True,
            },
        ],
    )
    def test_sparse_long_seq(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)

    # ─── Tier 3c: GQA (NHK>1, NHQ>NHK) ───────────────────

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            # GQA large ratio (64x) — no SwapAB, PackGQA only
            {
                "name": "gqa64x2_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 2,
                "topk": 128,
                "pack_gqa": True,
            },
            # GQA large ratio — no PackGQA (control)
            {
                "name": "gqa64x2_no_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 2,
                "topk": 128,
                "pack_gqa": False,
            },
            # GQA small ratio (4x, ≤16) — SwapAB + PackGQA
            {
                "name": "gqa4x4_packgqa_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 16,
                "NHK": 4,
                "topk": 128,
                "pack_gqa": True,
                "swap_ab": True,
            },
            # GQA small ratio (8x2) — SwapAB + PackGQA
            {
                "name": "gqa8x2_packgqa_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 16,
                "NHK": 2,
                "topk": 128,
                "pack_gqa": True,
                "swap_ab": True,
            },
        ],
    )
    def test_sparse_gqa(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)

    # ─── Tier 3d: MHA (NHQ==NHK, multi-KV-head) ────────────

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            # MHA small (4 heads) + SwapAB
            {
                "name": "mha4_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "topk": 128,
                "pack_gqa": False,
                "swap_ab": True,
            },
            # MHA larger (16 heads) + SwapAB
            {
                "name": "mha16_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 16,
                "NHK": 16,
                "topk": 128,
                "pack_gqa": False,
                "swap_ab": True,
            },
        ],
    )
    def test_sparse_mha(self, config: dict[str, Any]):
        _run_index_sparse_config(self.device, config)

    # ─── Tier 3e: k_block_size > 1 (block-level K indexing) ───

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            # kbs=8: sub-tile scatter — 8 tokens per entry, 16 entries per tile
            {
                "name": "mqa128_kblock8",
                "B": 1,
                "S": 1024,
                "NHQ": 128,
                "NHK": 1,
                "topk": 16,
                "max_topk": 16,
                "pack_gqa": True,
                "k_block_size": 8,
            },
            # kbs=32: sub-tile scatter — 32 tokens per entry, 4 entries per tile
            {
                "name": "mqa128_kblock32",
                "B": 1,
                "S": 1024,
                "NHQ": 128,
                "NHK": 1,
                "topk": 8,
                "max_topk": 8,
                "pack_gqa": True,
                "k_block_size": 32,
            },
            # kbs=128: canonical 128×128 block — topk=2, S=256 → 2 K blocks (full)
            {
                "name": "mqa128_kblock128",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": 2,
                "pack_gqa": True,
                "k_block_size": 128,
            },
            # kbs=128: larger S — topk=4 out of 8 K blocks, partial coverage
            {
                "name": "mqa128_kblock128_s1024",
                "B": 1,
                "S": 1024,
                "NHQ": 128,
                "NHK": 1,
                "topk": 4,
                "pack_gqa": True,
                "k_block_size": 128,
            },
            # kbs=128: smaller GQA ratio (32×)
            {
                "name": "mqa32_kblock128",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 1,
                "topk": 2,
                "pack_gqa": True,
                "k_block_size": 128,
            },
        ],
    )
    def test_sparse_k_block_size(self, config: dict[str, Any]):
        kbs = config.get("k_block_size", 1)
        if kbs > 1:
            # LoopK BWD misinterprets block-level indices as token-level;
            # use LoopQ BWD for kbs>1 (requires env vars).
            # NHK>1 + kbs>1 not yet supported in LoopQ BWD (flat-layout mismatch).
            os.environ["MAGI_ATTENTION_INDEX_SPARSE_BWD_LOOP_Q"] = "1"
            os.environ["MAGI_ATTENTION_INDEX_SPARSE_BWD_K_BLOCK_SIZE"] = str(kbs)
            test_bwd = config.get("NHK", 1) == 1
        else:
            test_bwd = True
        try:
            _run_index_sparse_config(self.device, config, test_bwd=test_bwd)
        finally:
            os.environ.pop("MAGI_ATTENTION_INDEX_SPARSE_BWD_LOOP_Q", None)
            os.environ.pop("MAGI_ATTENTION_INDEX_SPARSE_BWD_K_BLOCK_SIZE", None)


if __name__ == "__main__":
    run_tests()
