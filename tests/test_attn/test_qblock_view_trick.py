# Copyright (c) 2025 SandAI. All Rights Reserved.
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

"""Test: simulate q_block_size>1 via external Q view/reshape for IndexAttn.

Idea
----
Instead of modifying the kernel to support q_block_size>1, fold QBS consecutive
Q tokens into the *heads* dimension externally, then call index_attn with
q_block_size=1.

    Q: (B, S, NHQ, D) → (B, S//QBS, NHQ*QBS, D)

NHQ_new = NHQ * QBS, NHK unchanged, gqa_new = NHQ_new / NHK.

Results
-------
- NHK=1 (MQA): simple ``.view()`` works (zero-copy).
  All virtual heads share the single K head, so ordering doesn't matter.

- NHK>1 (MHA/GQA): simple ``.view()`` gives **wrong** GQA head mapping.
  Need ``permute + contiguous`` to group token offsets within each original head::

      (B, S//QBS, QBS, NHQ, D)  →  permute(0,1,3,2,4)  →  contiguous
      →  (B, S//QBS, NHQ*QBS, D)

  This ensures virtual heads {QBS*h, ..., QBS*h+QBS-1} all map to original
  head h (and thus the correct K head).  Requires a data copy (not zero-copy).

Constraints
-----------
- gqa_new = (NHQ * QBS) / NHK must satisfy: kBlockM % gqa_new == 0 (kBlockM=128).
- The kernel's packgqa path currently supports gqa_new up to ~4.
  Larger gqa_new (e.g., MQA with NHQ=4, QBS=4 → gqa_new=16) fails to compile.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_block_shared_indices(B, S, NHK, topk, qbs, device):
    """Build index_attn_indices where each q_block of *qbs* tokens shares K indices.

    Returns
    -------
    indices_full : (B*S, NHK, topk)       — per-token (baseline)
    indices_block : (B*S//qbs, NHK, topk)  — per-block (view trick)
    """
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
    """SDPA reference with dense mask built from index_attn_indices."""
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


def _run_baseline(q_raw, k_raw, v_raw, indices_full, B, S, NHQ, NHK):
    """FFA index_attn baseline: q_block_size=1 with per-token indices."""
    q_ffa = (
        rearrange(q_raw, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
        .detach()
        .clone()
    )
    k_ffa, v_ffa = _pack_kv(k_raw, v_raw)
    o, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_attn_indices=indices_full,
        q_block_size=1,
        k_block_size=1,
        pack_gqa=True,
    )
    return rearrange(o, "(b s h1) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S)


def _run_view_trick(
    q_raw, k_raw, v_raw, indices_block, B, S, NHQ, NHK, D, qbs, *, use_permute
):
    """Fold *qbs* tokens into heads, call FFA with q_block_size=1.

    Parameters
    ----------
    use_permute : bool
        True  → permute(0,1,3,2,4)+contiguous to fix head mapping (data copy).
        False → simple .view() (zero-copy, correct only when NHK=1).
    """
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
        rearrange(q_viewed, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
        .detach()
        .clone()
    )
    k_ffa, v_ffa = _pack_kv(k_raw, v_raw)

    o_sparse, _ = flex_flash_attn_func(
        q_ffa,
        k_ffa,
        v_ffa,
        index_attn_indices=indices_block,
        q_block_size=1,
        k_block_size=1,
        pack_gqa=True,
    )

    # Reshape output: (B*S_new*NHK, gqa_new, D) → (B, S, NHQ, D)
    o_unpacked = rearrange(
        o_sparse, "(b s h1) h2 d -> b s h1 h2 d", b=B, s=S_new, h1=NHK
    )

    if use_permute:
        # gqa_new = gqa * qbs; sub-head j → gqa_sub = j//qbs, tok_off = j%qbs
        o_out = (
            o_unpacked.reshape(B, S_new, NHK, gqa, qbs, D)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(B, S, NHQ, D)
        )
    else:
        # h_v = tok_off * NHQ + orig_head → reshape (qbs, NHQ) unfolds correctly
        o_combined = rearrange(o_unpacked, "b s h1 h2 d -> b s (h1 h2) d")
        o_out = o_combined.reshape(B, S_new, qbs, NHQ, D).reshape(B, S, NHQ, D)

    return o_out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# All tests use gqa_new=4 (packgqa4) to avoid compilation issues with gqa>4.
# - MHA (NHQ=NHK=4, QBS=4): gqa=1 → gqa_new=4  ✓
# - Single-head MQA (NHQ=NHK=1, QBS=4): gqa=1 → gqa_new=4  ✓


class TestQBlockViewTrick:
    device = "cuda"

    def test_baseline_matches_sdpa(self):
        """Sanity: FFA baseline (q_block_size=1, per-token indices) matches SDPA."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 4, 1, 128, 128, 4
        torch.manual_seed(42)
        dev = self.device

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)

        idx_full, _ = _build_block_shared_indices(B, S, NHK, topk, qbs, dev)
        o_ref = _sdpa_reference(q, k, v, idx_full, B, S, NHQ, NHK, dev)
        o_ffa = _run_baseline(q, k, v, idx_full, B, S, NHQ, NHK)

        diff = (o_ffa.float() - o_ref.float()).abs().max().item()
        print(f"[baseline vs SDPA] max_diff = {diff:.6f}")
        assert diff < 0.02, f"Baseline max_diff={diff:.6f} >= 0.02"

    def test_single_head_simple_view(self):
        """NHQ=NHK=1 (single head): simple .view() should work (zero-copy)."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 1, 1, 128, 128, 4
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
        print(f"[single-head simple view] max_diff = {diff:.6f}")
        assert diff < 0.02, f"single-head simple view: max_diff={diff:.6f} >= 0.02"

    def test_mha_simple_view_wrong(self):
        """MHA (NHQ=NHK=4, QBS=4): simple .view() gives wrong head mapping."""
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
        print(f"[MHA simple view] max_diff = {diff:.6f} (expected LARGE)")
        assert diff > 0.1, (
            f"MHA simple view unexpectedly correct: max_diff={diff:.6f} <= 0.1. "
            f"Head mapping should be wrong for NHK>1."
        )

    def test_mha_permute_view(self):
        """MHA (NHQ=NHK=4, QBS=4): permute+contiguous fixes head mapping."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 4, 4, 128, 128, 4
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
        print(f"[MHA permute view] max_diff = {diff:.6f}")
        assert diff < 0.02, f"MHA permute view: max_diff={diff:.6f} >= 0.02"

    def test_mha_permute_view_matches_baseline(self):
        """MHA: permuted view trick output matches FFA baseline (not just SDPA)."""
        B, S, NHQ, NHK, D, topk, qbs = 1, 256, 4, 4, 128, 128, 4
        torch.manual_seed(42)
        dev = self.device

        q = torch.randn(B, S, NHQ, D, dtype=torch.bfloat16, device=dev)
        k = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)
        v = torch.randn(B, S, NHK, D, dtype=torch.bfloat16, device=dev)

        idx_full, idx_block = _build_block_shared_indices(B, S, NHK, topk, qbs, dev)
        o_base = _run_baseline(q, k, v, idx_full, B, S, NHQ, NHK)
        o_view = _run_view_trick(
            q, k, v, idx_block, B, S, NHQ, NHK, D, qbs, use_permute=True
        )

        diff = (o_view.float() - o_base.float()).abs().max().item()
        print(f"[MHA permute view vs baseline] max_diff = {diff:.6f}")
        assert diff < 0.02, f"MHA view vs baseline: max_diff={diff:.6f} >= 0.02"
