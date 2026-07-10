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

Structure:
  TestIndexSparseViewTrick — view-trick correctness (MQA view, MHA wrong, MHA permute)
  TestIndexSparseSweep     — Classic CI sweep: q_seqlen × kv_seqlen × topk
  TestIndexSparseComprehensiveSweep — CI: orthogonal parameterization:
      head_config × head_dim × kbs × inner_dir × inner_load_mode × inner_store_mode

Classic sweep (CI gate):
  - Fixed: MQA128, D=128, PackGQA=True, kbs=1
  - Parameterizes: q_seqlen(512/1000/16384) × kv_seqlen(512/1000/16384) × topk(128/256)

Comprehensive sweep (CI):
  - head_config: 3 MQA + 3 GQA + 3 MHA = 9
  - head_dim: 64, 128
  - kbs: 1, 8, 32, 128, 256 (kbs>1 only valid for NHK=1, PackGQA=True, D=128)
  - inner_dir: None (default), "true", "false"
  - inner_load_mode: None (default), "tma", "cpasync"
  - inner_store_mode: None (default), "atomicadd", "tma1d"

Known limitations:
  - swap_ab is prohibited for IndexSparse (asserted in flex_flash_attn_func)
  - max_topk must be multiples of tile_size (128, or 64 if swap_ab)
  - Q/K/V are packed in (b, s, h) order to match index_sparse_indices view layout
  - BWD with kbs>=256 exceeds SM90 smem limit (BwdTileN=256 → >228KB)
"""

import os
import unittest
from typing import Any

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
    sparse_k_block_size=1,
    test_bwd=False,
    swap_bwd_qk_loop=None,
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
            sparse_k_block_size=sparse_k_block_size,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
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
                sparse_k_block_size=sparse_k_block_size,
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
    """Compare FFA output against SDPA reference, batch by batch, head by head.

    Loops per Q-head to avoid materializing the full (NHQ, S_q, S_kv)
    attention matrix which is catastrophic for large MQA configs.
    sdpa_mask: compact (B, NHK, S_q, S_kv) — broadcasts across GQA heads.
    """
    gqa = NHQ // NHK
    err_msgs = []
    for b_idx in range(B):
        o_ref_heads = []
        with torch.no_grad():
            for h_q in range(NHQ):
                h_kv = h_q // gqa
                q_h = q[b_idx, :, h_q : h_q + 1, :].unsqueeze(0).transpose(1, 2)
                k_h = k[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                v_h = v[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                mask_h = sdpa_mask[b_idx, h_kv : h_kv + 1].unsqueeze(0)
                o_h = torch.nn.functional.scaled_dot_product_attention(
                    q_h, k_h, v_h, attn_mask=mask_h
                )
                o_ref_heads.append(o_h)
        o_ref = torch.cat(o_ref_heads, dim=1)
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
    sparse_k_block_size = cfg.get("sparse_k_block_size", 1)
    dtype = cfg.get("dtype", torch.bfloat16)
    atol = cfg.get("atol", DEFAULT_ATOL)
    swap_bwd_qk_loop = cfg.get("swap_bwd_qk_loop", None)

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
        B,
        NHK,
        S_q,
        S_kv,
        topk,
        max_topk,
        device,
        sparse_k_block_size=sparse_k_block_size,
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
        sparse_k_block_size=sparse_k_block_size,
        test_bwd=test_bwd,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
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
        sparse_k_block_size=sparse_k_block_size,
    )

    test_case = (
        f"[NHQ={cfg['NHQ']},NHK={cfg['NHK']},S_q={cfg.get('S_q', cfg.get('S'))},S_kv={cfg.get('S_kv', cfg.get('S'))},"
        f"B={B},D={D},topk={topk},max_topk={max_topk},pack_gqa={pack_gqa},"
        f"swap_ab={swap_ab},sparse_k_block_size={sparse_k_block_size},dtype={dtype},"
        f"flat:NHQ_eff={NHQ},S_q_eff={S_q},S_kv_eff={S_kv}]"
    )

    _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)

    if test_bwd:
        do = torch.randn_like(o_sparse)
        o_sparse.backward(do)
        dq_ffa = q_ffa.grad.clone()

        gqa = NHQ // NHK
        total_q = B * S_q
        do_reshaped_all = rearrange(
            do, "(b s h1) h2 d -> b (h1 h2) s d", b=B, h1=NHK, s=S_q
        )
        dq_ref_list = []
        for b_idx in range(B):
            dq_heads = []
            for h_q in range(NHQ):
                h_kv = h_q // gqa
                q_h = (
                    q[b_idx, :, h_q : h_q + 1, :]
                    .unsqueeze(0)
                    .transpose(1, 2)
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
                k_h = (
                    k[b_idx, :, h_kv : h_kv + 1, :]
                    .unsqueeze(0)
                    .transpose(1, 2)
                    .detach()
                    .clone()
                )
                v_h = (
                    v[b_idx, :, h_kv : h_kv + 1, :]
                    .unsqueeze(0)
                    .transpose(1, 2)
                    .detach()
                    .clone()
                )
                mask_h = sdpa_mask[b_idx, h_kv : h_kv + 1].unsqueeze(0)
                o_h = torch.nn.functional.scaled_dot_product_attention(
                    q_h, k_h, v_h, attn_mask=mask_h
                )
                do_h = do_reshaped_all[b_idx, h_q : h_q + 1].unsqueeze(0)
                o_h.backward(do_h)
                dq_heads.append(q_h.grad)
            dq_ref_list.append(torch.cat(dq_heads, dim=1))

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

        if cfg.get("check_deterministic", True) and swap_bwd_qk_loop is not True:
            det_errs = _check_deterministic_index_sparse(
                q_ffa=q_ffa,
                k_ffa=k_ffa,
                v_ffa=v_ffa,
                index_sparse_indices=index_sparse_indices,
                pack_gqa=pack_gqa,
                sparse_k_block_size=sparse_k_block_size,
                ref_block_size=ref_block_size,
                swap_ab=swap_ab,
                swap_bwd_qk_loop=swap_bwd_qk_loop,
                o_ref=o_sparse.detach(),
                dq_ref=dq_ffa,
                test_case=test_case,
            )
            if det_errs:
                raise AssertionError("\n".join(det_errs))


def _check_deterministic_index_sparse(
    q_ffa,
    k_ffa,
    v_ffa,
    index_sparse_indices,
    *,
    pack_gqa,
    sparse_k_block_size,
    ref_block_size,
    swap_ab,
    swap_bwd_qk_loop=None,
    o_ref,
    dq_ref,
    test_case,
):
    """Verify bit-exact reproducibility by running twice in deterministic mode.

    For LoopK: deterministic=True is unsupported by the kernel, skip check.
    For LoopQ: run twice with deterministic=True and compare the two runs.
    """
    if swap_bwd_qk_loop is True:
        return []

    err_msgs: list[str] = []
    do = torch.randn_like(q_ffa)

    def _run_det():
        q2 = q_ffa.clone().detach().requires_grad_(True)
        k2 = k_ffa.clone().detach().requires_grad_(True)
        v2 = v_ffa.clone().detach().requires_grad_(True)
        o2, _ = flex_flash_attn_func(
            q2,
            k2,
            v2,
            index_sparse_indices=index_sparse_indices,
            q_block_size=1,
            sparse_k_block_size=sparse_k_block_size,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
            deterministic=True,
        )
        o2.backward(do)
        return o2.detach(), q2.grad.detach()

    o_det1, dq_det1 = _run_det()
    o_det2, dq_det2 = _run_det()

    if not torch.equal(o_det1, o_det2):
        err_msgs.append(f"For {test_case=}: forward output not deterministic")
    if not torch.equal(dq_det1, dq_det2):
        err_msgs.append(f"For {test_case=}: backward dQ not deterministic")
    return err_msgs


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

    # indices_block is contiguous (from torch.full), so view is safe here.
    # If indices_block were non-contiguous (e.g. from slicing), reshape would be needed.
    # View trick flattens K to (B*S*NHK, 1, D), so kernel sees 1 KV head.
    # The flat order matches Q's rearrange("b s (h1 h2) d -> (b s h1) h2 d", h1=NHK).
    indices_for_kernel = indices_block.view(-1, 1, indices_block.shape[-1])

    o_sparse, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_sparse_indices=indices_for_kernel,
        q_block_size=1,
        sparse_k_block_size=1,
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
# TestIndexSparseViewTrick — view-trick correctness tests
# ═══════════════════════════════════════════════════════════


class TestIndexSparseViewTrick(unittest.TestCase):
    """Standalone IndexSparse view-trick tests.

    Tests specific behaviors:
    - MQA: simple .view() works (zero-copy)
    - MHA: simple .view() fails (wrong head mapping)
    - MHA: permute+contiguous fixes head mapping
    """

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py."""
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}

        # view-trick tests: all FWD only, index_sparse kbs=1
        # After view-trick rearrange, kernel sees NHK=1 → pack_gqa_factor=gqa
        for nhq, nhk in [(2, 1), (4, 4), (32, 32)]:
            pack_f = nhq // nhk if nhk == 1 else 1
            pgqa = nhk == 1
            add_ffa_spec(
                specs,
                direction="fwd",
                disable_atomic=True,
                pack_gqa=pgqa,
                pack_gqa_factor=pack_f,
                index_sparse=True,
                sparse_k_block_size=1,
            )

        return specs

    @property
    def device(self):
        return torch.cuda.current_device()

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


# ═══════════════════════════════════════════════════════════
# TestIndexSparseSweep — unified CI sweep
# ═══════════════════════════════════════════════════════════


class TestIndexSparseSweep(DistTestBase):
    """IndexSparse Classic sweep — CI gate.

    Cross-product of q_seqlen × kv_seqlen × topk.
    Fixed compile params: MQA128, D=128, PackGQA=True, kbs=1.
    """

    Q_SEQLENS = [512, 1000, 16384]
    KV_SEQLENS = [512, 1000, 16384]
    TOPKS = [128, 256]

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        All classic combos share the same compile params: MQA128 kbs=1.
        FWD + BWD InnerLoopK (swap_bwd_qk_loop=True).
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}
        # FWD: disable_fwd_atomic, ref_block_size=(128,128)
        add_ffa_spec(
            specs,
            direction="fwd",
            disable_atomic=True,
            pack_gqa=True,
            pack_gqa_factor=128,
            index_sparse=True,
            sparse_k_block_size=1,
        )
        # BWD InnerLoopK: disable_dq_atomic
        add_ffa_spec(
            specs,
            direction="bwd",
            disable_dq_atomic=True,
            pack_gqa=True,
            pack_gqa_factor=128,
            index_sparse=True,
            bwd_inner_loop_k=True,
            sparse_k_block_size=1,
            bwd_dq_bf16=True,
        )
        return specs

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
        return 600

    @with_run_in_mp
    @parameterize("q_seqlen", Q_SEQLENS)
    @parameterize("kv_seqlen", KV_SEQLENS)
    @parameterize("topk", TOPKS)
    def test_index_sparse_classic(self, q_seqlen, kv_seqlen, topk):
        if topk > kv_seqlen:
            return
        config = {
            "B": 1,
            "S_q": q_seqlen,
            "S_kv": kv_seqlen,
            "NHQ": 128,
            "NHK": 1,
            "D": 128,
            "topk": topk,
            "pack_gqa": True,
            "swap_bwd_qk_loop": True,
        }
        _run_index_sparse_config(self.device, config, test_bwd=True)


# ═══════════════════════════════════════════════════════════
# TestIndexSparseComprehensiveSweep — comprehensive coverage (CI)
# ═══════════════════════════════════════════════════════════


class TestIndexSparseComprehensiveSweep(DistTestBase):
    """IndexSparse Comprehensive sweep — CI.

    Orthogonal parameterization:
      head_config × head_dim × kbs × inner_dir × inner_load_mode × inner_store_mode
    Skips invalid combos (kbs>1 requires NHK=1+PackGQA+D=128; kbs>=256 no BWD).
    """

    # ─── Axis 1: Head configuration (NHQ, NHK, pack_gqa) ───
    # 3 MQA + 3 GQA + 3 MHA = 9 configs
    HEAD_CONFIGS = [
        (128, 1, True),  # MQA128
        (64, 1, True),  # MQA64
        (4, 1, True),  # MQA4
        (128, 2, True),  # GQA 128:2
        (32, 4, True),  # GQA 32:4
        (4, 2, True),  # GQA 4:2
        (8, 8, True),  # MHA8
        (4, 4, True),  # MHA4
        (32, 32, True),  # MHA32
    ]

    # ─── Axis 2: Head dimension ───
    HEAD_DIMS = [64, 128]

    # ─── Axis 3: Sparse K block size ───
    KBS_VALUES = [1, 8, 32, 128, 256]

    # ─── Axis 4: Inner loop direction ───
    INNER_DIRS = [None, "true", "false"]

    # ─── Axis 5: Inner load mode ───
    INNER_LOAD_MODES = [None, "tma", "cpasync"]

    # ─── Axis 6: Inner store mode ───
    INNER_STORE_MODES = [None, "atomicadd", "tma1d"]

    @classmethod
    def _iter_valid_configs(cls):
        """Iterate all valid (non-skipped) parameter combinations for precompile."""
        for nhq, nhk, pack_gqa in cls.HEAD_CONFIGS:
            pack_f = nhq // nhk
            _is_mha = nhq == nhk
            effective_pack_gqa = pack_gqa
            if not _is_mha and not pack_gqa:
                effective_pack_gqa = True
            _gqa_safe = _is_mha or effective_pack_gqa

            for hd in cls.HEAD_DIMS:
                for kbs in cls.KBS_VALUES:
                    if kbs > 1 and (nhk > 1 or not effective_pack_gqa or hd != 128):
                        continue
                    yield nhq, nhk, pack_gqa, hd, kbs, pack_f, _is_mha, effective_pack_gqa, _gqa_safe

    @classmethod
    def precompile_kernel_specs(cls):
        """Standard precompile interface — see magi_attention/testing/precompile.py.

        Covers both LoopQ and LoopK directions, and all inner mode env variants.
        """
        from magi_attention.testing.precompile import add_ffa_spec

        specs: dict = {}

        # Build env combos (cross-product of dir × load × store, deduplicated)
        env_combos: list[dict[str, str]] = [{}]
        for d in cls.INNER_DIRS:
            for lm in cls.INNER_LOAD_MODES:
                for sm in cls.INNER_STORE_MODES:
                    env: dict[str, str] = {}
                    if d is not None:
                        env["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"] = d
                    if lm is not None:
                        env["MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD_MODE"] = lm
                    if sm is not None:
                        env["MAGI_ATTENTION_FFA_SPARSE_INNER_STORE_MODE"] = sm
                    if env and env not in env_combos:
                        env_combos.append(env)

        def _add(
            direction,
            env,
            *,
            disable_atomic=False,
            disable_dq_atomic=False,
            bwd_inner_loop_k=False,
            bwd_dq_bf16=False,
            **kw,
        ):
            add_ffa_spec(
                specs,
                direction=direction,
                env=env,
                disable_atomic=disable_atomic,
                disable_dq_atomic=disable_dq_atomic,
                bwd_inner_loop_k=bwd_inner_loop_k,
                bwd_dq_bf16=bwd_dq_bf16,
                index_sparse=True,
                **kw,
            )

        for (
            nhq,
            nhk,
            pack_gqa,
            hd,
            kbs,
            pack_f,
            _is_mha,
            effective_pack_gqa,
            _gqa_safe,
        ) in cls._iter_valid_configs():
            common = dict(
                head_dim=hd,
                pack_gqa=effective_pack_gqa,
                pack_gqa_factor=pack_f,
                sparse_k_block_size=kbs,
            )
            test_bwd = kbs < 256
            for env in env_combos:
                _add("fwd", env, disable_atomic=True, **common)
                if test_bwd:
                    _add("bwd", env, disable_atomic=_gqa_safe, **common)
                    if pack_f >= 128:
                        _add(
                            "bwd",
                            env,
                            disable_dq_atomic=True,
                            bwd_inner_loop_k=True,
                            bwd_dq_bf16=True,
                            head_dim=hd,
                            pack_gqa=pack_gqa,
                            pack_gqa_factor=pack_f,
                            sparse_k_block_size=kbs,
                        )

        return specs

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

    @with_run_in_mp
    @parameterize("head_config", HEAD_CONFIGS)
    @parameterize("hd", HEAD_DIMS)
    @parameterize("kbs", KBS_VALUES)
    @parameterize("inner_dir", INNER_DIRS)
    @parameterize("inner_load_mode", INNER_LOAD_MODES)
    @parameterize("inner_store_mode", INNER_STORE_MODES)
    def test_index_sparse_comprehensive(
        self, head_config, hd, kbs, inner_dir, inner_load_mode, inner_store_mode
    ):
        """LoopQ (default) direction — full orthogonal sweep."""
        nhq, nhk, pack_gqa = head_config
        if kbs > 1 and (nhk > 1 or not pack_gqa or hd != 128):
            return

        if kbs <= 1:
            S, topk = 256, 128
        else:
            S = 1024
            topk = max(2, 128 // kbs)

        test_bwd = kbs < 256
        config: dict[str, Any] = {
            "B": 1,
            "S": S,
            "NHQ": nhq,
            "NHK": nhk,
            "D": hd,
            "topk": topk,
            "pack_gqa": pack_gqa,
            "sparse_k_block_size": kbs,
            "check_deterministic": False,
        }
        if kbs > 1:
            config["max_topk"] = topk

        env_keys: list[str] = []
        if inner_dir is not None:
            os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"] = inner_dir
            env_keys.append("MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN")
        if inner_load_mode is not None:
            os.environ["MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD_MODE"] = inner_load_mode
            env_keys.append("MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD_MODE")
        if inner_store_mode is not None:
            os.environ["MAGI_ATTENTION_FFA_SPARSE_INNER_STORE_MODE"] = inner_store_mode
            env_keys.append("MAGI_ATTENTION_FFA_SPARSE_INNER_STORE_MODE")
        try:
            _run_index_sparse_config(self.device, config, test_bwd=test_bwd)
        finally:
            for key in env_keys:
                os.environ.pop(key, None)

    @with_run_in_mp
    @parameterize("head_config", HEAD_CONFIGS)
    @parameterize("hd", HEAD_DIMS)
    @parameterize("kbs", KBS_VALUES)
    @parameterize("inner_dir", INNER_DIRS)
    @parameterize("inner_load_mode", INNER_LOAD_MODES)
    @parameterize("inner_store_mode", INNER_STORE_MODES)
    def test_index_sparse_comprehensive_loopk(
        self, head_config, hd, kbs, inner_dir, inner_load_mode, inner_store_mode
    ):
        """LoopK direction — tests dQ non-atomic path.

        When pack_gqa_factor < 128, runtime auto-falls back to LoopQ.
        """
        nhq, nhk, pack_gqa = head_config
        if kbs > 1 and (nhk > 1 or not pack_gqa or hd != 128):
            return
        if kbs >= 256:
            return

        if kbs <= 1:
            S, topk = 256, 128
        else:
            S = 1024
            topk = max(2, 128 // kbs)

        config: dict[str, Any] = {
            "B": 1,
            "S": S,
            "NHQ": nhq,
            "NHK": nhk,
            "D": hd,
            "topk": topk,
            "pack_gqa": pack_gqa,
            "sparse_k_block_size": kbs,
            "swap_bwd_qk_loop": True,
        }
        if kbs > 1:
            config["max_topk"] = topk

        env_keys: list[str] = []
        if inner_dir is not None:
            os.environ["MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"] = inner_dir
            env_keys.append("MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN")
        if inner_load_mode is not None:
            os.environ["MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD_MODE"] = inner_load_mode
            env_keys.append("MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD_MODE")
        if inner_store_mode is not None:
            os.environ["MAGI_ATTENTION_FFA_SPARSE_INNER_STORE_MODE"] = inner_store_mode
            env_keys.append("MAGI_ATTENTION_FFA_SPARSE_INNER_STORE_MODE")
        try:
            _run_index_sparse_config(self.device, config, test_bwd=True)
        finally:
            for key in env_keys:
                os.environ.pop(key, None)


if __name__ == "__main__":
    run_tests()
