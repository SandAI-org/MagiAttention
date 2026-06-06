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

"""Test: simulate q_block_size>1 via external Q view/reshape for IndexAttn.

Fold QBS consecutive Q tokens into the heads dimension, then call index_attn
with q_block_size=1.  No kernel modification needed.

    Q: (B, S, NHQ, D) → (B, S//QBS, NHQ*QBS, D)

Three conclusions verified:

1. MQA (NHK=1): simple ``.view()`` works (zero-copy).
   Only 1 K head — head mapping trivially correct.

2. GQA/MHA (NHK>1): simple ``.view()`` gives **wrong** results.

3. GQA/MHA (NHK>1): ``permute + contiguous`` fixes it (data copy required).

Why view alone fails for NHK>1 — example (S=8, H=4, QBS=4, D omitted):

    Memory: [t0h0 t0h1 t0h2 t0h3 | t1h0 t1h1 t1h2 t1h3 | t2... | t3...]

    .view(2, 16, D) — zero-copy, same memory order:

        GQA group 0 → K head 0      GQA group 1 → K head 1
        [t0h0  t0h1  t0h2  t0h3]    [t1h0  t1h1  t1h2  t1h3]    ...
                ^^^                         ^^^
         t0h1 should use K head 1!   t1h1 should use K head 1!
         ⇒ WRONG: same token's different heads land in one GQA group.

    .reshape(2,4,4,D).permute(0,2,1,3).contiguous().reshape(2,16,D):

        GQA group 0 → K head 0      GQA group 1 → K head 1
        [t0h0  t1h0  t2h0  t3h0]    [t0h1  t1h1  t2h1  t3h1]    ...
         ⇒ CORRECT: same head across 4 token offsets share one K head.

    The permute swaps the (QBS, NHQ) axes so token offsets become the inner
    (contiguous) dimension within each GQA group.  ``.contiguous()`` is needed
    because the permuted strides are non-contiguous.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_block_shared_indices(B, S, NHK, topk, qbs, device):
    """Build index_attn_indices where each q_block of *qbs* tokens shares K indices."""
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
        index_attn_indices=indices_block,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQBlockViewTrick:
    device = "cuda"

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
        print(f"[MQA simple view] max_diff = {diff:.6f}")
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
        print(f"[MHA simple view] max_diff = {diff:.6f} (expected LARGE)")
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
        print(f"[MHA 32h permute view] max_diff = {diff:.6f}")
        assert diff < 0.02, f"MHA 32h permute: max_diff={diff:.6f} >= 0.02"
