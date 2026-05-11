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
Tests for sparse_kv_indices direct path (forward only).

Validates flex_flash_attn_func with sparse_kv_indices input against
PyTorch SDPA reference, covering:
  - MQA(128) + pack_gqa  (highest priority)
  - GQA + pack_gqa
  - MHA (no pack_gqa)
  - Multi-batch with uniform / variable seqlen
  - Padding (actual_topk < max_topk)
"""

import pytest
import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils import set_random_seed

# All tests marked skip until kernel implementation is complete.
# Remove this marker once the sparse_kv_indices direct kernel path lands.
SKIP_REASON = "sparse_kv_indices direct kernel path not yet implemented"


def _build_sparse_kv_indices(B, NHK, S, actual_topk, max_topk, device):
    """Build sparse_kv_indices [B, NHK, S, max_topk] with random KV selection."""
    indices = torch.full((B, NHK, S, max_topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        k = actual_topk[b]
        for h in range(NHK):
            for qi in range(S):
                perm = torch.randperm(S, device=device)[:k].sort().values
                indices[b, h, qi, :k] = perm.int()
    return indices


def _build_sdpa_mask(sparse_kv_indices, actual_topk, B, NHQ, NHK, S, device):
    """Build SDPA boolean mask from sparse_kv_indices for reference computation."""
    mask = torch.zeros(B, NHQ, S, S, dtype=torch.bool, device=device)
    gqa = NHQ // NHK
    for b in range(B):
        vk = actual_topk[b]
        for h_kv in range(NHK):
            for qi in range(S):
                valid_kv = sparse_kv_indices[b, h_kv, qi, :vk]
                valid_kv = valid_kv[valid_kv >= 0]
                for h_q_offset in range(gqa):
                    h_q = h_kv * gqa + h_q_offset
                    mask[b, h_q, qi, valid_kv.long()] = True
    return mask


def _run_sparse_attn_vs_sdpa(
    B: int,
    S: int,
    NHQ: int,
    NHK: int,
    D: int,
    actual_topk: list[int],
    pack_gqa: bool,
    dtype=torch.bfloat16,
    atol: float = 0.01,
):
    """Run sparse attention and compare against SDPA reference."""
    device = "cuda:0"
    set_random_seed(42)
    max_topk = max(actual_topk)

    sparse_kv_indices = _build_sparse_kv_indices(B, NHK, S, actual_topk, max_topk, device)

    q = torch.randn(B, S, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S, NHK, D, dtype=dtype, device=device)

    # FFA layout: Q is (B*NHK*S, NHQ//NHK, D), K/V is (B*NHK*S, 1, D)
    gqa = NHQ // NHK
    q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")

    with torch.no_grad():
        o_sparse, _ = flex_flash_attn_func(
            q_ffa.clone(), k_ffa.clone(), v_ffa.clone(),
            sparse_kv_indices=sparse_kv_indices,
            actual_topk=actual_topk,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=pack_gqa,
        )

    o_sparse_reshaped = rearrange(
        o_sparse, "(b h1 s) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S
    )

    sdpa_mask = _build_sdpa_mask(sparse_kv_indices, actual_topk, B, NHQ, NHK, S, device)

    for b in range(B):
        q_sdpa = rearrange(q[b], "s h d -> 1 h s d")
        k_sdpa = rearrange(k[b], "s h d -> 1 h s d")
        v_sdpa = rearrange(v[b], "s h d -> 1 h s d")
        if gqa > 1:
            k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)

        with torch.no_grad():
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=sdpa_mask[b].unsqueeze(0)
            )
        o_ref = rearrange(o_ref, "1 h s d -> s h d")

        max_diff = (o_sparse_reshaped[b].float() - o_ref.float()).abs().max().item()
        assert max_diff < atol, (
            f"batch {b}: max_diff={max_diff:.6f} >= {atol} "
            f"(topk={actual_topk[b]}, B={B}, NHQ={NHQ}, NHK={NHK}, S={S}, pack_gqa={pack_gqa})"
        )


# ═══════════════════════════════════════════════════════════
# Priority 1: MQA(128) + pack_gqa=True
# ═══════════════════════════════════════════════════════════

@pytest.mark.skipif(True, reason=SKIP_REASON)
class TestSparseKvMQA128PackGQA:
    """MQA with 128 Q heads per KV head, pack_gqa=True."""

    def test_single_batch_basic(self):
        _run_sparse_attn_vs_sdpa(
            B=1, S=256, NHQ=128, NHK=1, D=128,
            actual_topk=[64], pack_gqa=True,
        )

    def test_single_batch_no_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=1, S=256, NHQ=128, NHK=1, D=128,
            actual_topk=[128], pack_gqa=True,
        )

    def test_single_batch_heavy_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=1, S=256, NHQ=128, NHK=1, D=128,
            actual_topk=[16], pack_gqa=True,
        )

    def test_multi_batch_uniform_seqlen(self):
        _run_sparse_attn_vs_sdpa(
            B=3, S=256, NHQ=128, NHK=1, D=128,
            actual_topk=[64, 128, 48], pack_gqa=True,
        )

    def test_multi_batch_all_same_topk(self):
        _run_sparse_attn_vs_sdpa(
            B=2, S=256, NHQ=128, NHK=1, D=128,
            actual_topk=[64, 64], pack_gqa=True,
        )


# ═══════════════════════════════════════════════════════════
# Priority 2: GQA + pack_gqa
# ═══════════════════════════════════════════════════════════

@pytest.mark.skipif(True, reason=SKIP_REASON)
class TestSparseKvGQAPackGQA:
    """GQA with NHQ > NHK, pack_gqa=True."""

    def test_gqa_8_2(self):
        _run_sparse_attn_vs_sdpa(
            B=2, S=256, NHQ=8, NHK=2, D=128,
            actual_topk=[64, 48], pack_gqa=True,
        )

    def test_gqa_16_4(self):
        _run_sparse_attn_vs_sdpa(
            B=2, S=256, NHQ=16, NHK=4, D=128,
            actual_topk=[64, 128], pack_gqa=True,
        )


# ═══════════════════════════════════════════════════════════
# Priority 3: MHA (NHQ=NHK, no pack_gqa)
# ═══════════════════════════════════════════════════════════

@pytest.mark.skipif(True, reason=SKIP_REASON)
class TestSparseKvMHA:
    """MHA with NHQ=NHK, pack_gqa=False."""

    def test_mha_basic(self):
        _run_sparse_attn_vs_sdpa(
            B=2, S=256, NHQ=4, NHK=4, D=128,
            actual_topk=[64, 48], pack_gqa=False,
        )

    def test_mha_no_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=1, S=256, NHQ=4, NHK=4, D=128,
            actual_topk=[128], pack_gqa=False,
        )

    def test_mha_heavy_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=2, S=256, NHQ=4, NHK=4, D=128,
            actual_topk=[16, 8], pack_gqa=False,
        )
