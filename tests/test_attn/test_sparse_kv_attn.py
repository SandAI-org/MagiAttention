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
Tests for sparse_kv_indices direct-to-kernel path (forward only).

Validates flex_flash_attn_func with sparse_kv_indices against PyTorch SDPA
reference. Organized by priority tiers:

  P0: MQA(128) + pack_gqa=True  (most important DiT scenario)
  P1: SwapAB (with compatible head configs -- GQA ratio must fit kBlockM)
  P2: GQA + pack_gqa
  P3: MHA (NHQ=NHK)
  P4: Edge cases (non-aligned topk, extreme sparsity, head_dim=64)

Known limitations:
  - Forward only (no backward)
  - k_block_size = 1 only (future: 32/64)
  - No distributed sparse yet
  - No sink tokens in sparse mode
  - SwapAB + MQA(128) + pack_gqa not supported: kBlockM(<=64) < gqa_ratio(128)
"""

import pytest
import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils import set_random_seed

DEVICE = "cuda:0"
SEED = 42
DEFAULT_ATOL = 0.01
HEAVY_PAD_ATOL = 0.02


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _build_sparse_kv_indices(B, NHK, S, actual_topk, max_topk, device=DEVICE):
    """Build sparse_kv_indices [B, NHK, S, max_topk] with random KV selection."""
    indices = torch.full((B, NHK, S, max_topk), -1, dtype=torch.int32, device=device)
    for b in range(B):
        k = actual_topk[b]
        for h in range(NHK):
            for qi in range(S):
                perm = torch.randperm(S, device=device)[:k].sort().values
                indices[b, h, qi, :k] = perm.int()
    return indices


def _build_sdpa_mask(sparse_kv_indices, actual_topk, B, NHQ, NHK, S, device=DEVICE):
    """Build dense boolean mask [B, NHQ, S, S] from sparse_kv_indices for SDPA ref."""
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
    swap_ab: bool = False,
    ref_block_size: tuple[int, int] | None = None,
    dtype=torch.bfloat16,
    atol: float = DEFAULT_ATOL,
):
    """Run sparse attention via flex_flash_attn_func and compare against SDPA."""
    set_random_seed(SEED)
    max_topk = max(actual_topk)

    sparse_kv_indices = _build_sparse_kv_indices(B, NHK, S, actual_topk, max_topk)

    q = torch.randn(B, S, NHQ, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, S, NHK, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, S, NHK, D, dtype=dtype, device=DEVICE)

    gqa = NHQ // NHK
    q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")

    with torch.no_grad():
        o_sparse, _ = flex_flash_attn_func(
            q_ffa.clone(),
            k_ffa.clone(),
            v_ffa.clone(),
            sparse_kv_indices=sparse_kv_indices,
            actual_topk=actual_topk,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )

    o_sparse_reshaped = rearrange(
        o_sparse, "(b h1 s) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S
    )

    sdpa_mask = _build_sdpa_mask(sparse_kv_indices, actual_topk, B, NHQ, NHK, S)

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
            f"(topk={actual_topk[b]}, B={B}, NHQ={NHQ}, NHK={NHK}, S={S}, D={D}, "
            f"pack_gqa={pack_gqa}, swap_ab={swap_ab}, ref_block_size={ref_block_size})"
        )


# ═══════════════════════════════════════════════════════════
# P0: MQA(128) + pack_gqa=True  -- highest priority
# ═══════════════════════════════════════════════════════════


class TestSparseKvP0_MQA128:
    """MQA(128,1) + pack_gqa=True: the canonical DiT sparse scenario."""

    def test_basic_topk_eq_S(self):
        """topk = S = 256, no padding, single batch."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[256],
            pack_gqa=True,
        )

    def test_topk_lt_S(self):
        """topk < S, moderate sparsity."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=512,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[128],
            pack_gqa=True,
        )

    def test_heavy_padding(self):
        """topk << S, heavy padding."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=512,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[16],
            pack_gqa=True,
            atol=HEAVY_PAD_ATOL,
        )

    def test_multi_batch_uniform_topk(self):
        """Multiple batches, all with same topk."""
        _run_sparse_attn_vs_sdpa(
            B=3,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[64, 64, 64],
            pack_gqa=True,
        )

    def test_multi_batch_variable_topk(self):
        """Multiple batches, each with different topk."""
        _run_sparse_attn_vs_sdpa(
            B=3,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[64, 128, 48],
            pack_gqa=True,
        )

    def test_large_S(self):
        """S=2048 with topk=256, realistic sparsity ratio."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=2048,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[256],
            pack_gqa=True,
        )


# ═══════════════════════════════════════════════════════════
# P1: SwapAB (GQA configs that fit kBlockM <= 64)
#
# NOTE: SwapAB + MQA(128) + pack_gqa is NOT supported because
#       kBlockM (max 64) < gqa_ratio (128).
#       SwapAB tests use GQA(32,4) ratio=8 which fits all M sizes.
# ═══════════════════════════════════════════════════════════


class TestSparseKvP1_SwapAB:
    """SwapAB variants. tile_size=64 when swap_ab=True.

    Uses GQA(32,4) (ratio=8, fits kBlockM=8..64) and MHA(4,4) (ratio=1).
    """

    def test_swap_ab_gqa_32_4(self):
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[64],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_swap_ab_gqa_32_4_multi_batch(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[64, 48],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_swap_ab_gqa_32_4_no_padding(self):
        """topk = S, no invalid tokens."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[256],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_swap_ab_mha_no_pack(self):
        """MHA(4,4) + SwapAB, pack_gqa=False."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=4,
            NHK=4,
            D=128,
            actual_topk=[64],
            pack_gqa=False,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    @pytest.mark.parametrize(
        "ref_block_size",
        [(8, 64), (16, 64), (32, 64), (64, 64)],
        ids=["M8", "M16", "M32", "M64"],
    )
    def test_swap_ab_all_block_sizes(self, ref_block_size):
        """Sweep all valid SwapAB ref_block_size values with GQA(32,4)."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[64],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=ref_block_size,
        )


# ═══════════════════════════════════════════════════════════
# P2: GQA + pack_gqa
# ═══════════════════════════════════════════════════════════


class TestSparseKvP2_GQA:
    """GQA with NHQ > NHK, pack_gqa=True."""

    def test_gqa_8_2(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=8,
            NHK=2,
            D=128,
            actual_topk=[64, 48],
            pack_gqa=True,
        )

    def test_gqa_32_4(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[64, 128],
            pack_gqa=True,
        )

    def test_gqa_8_2_swap_ab(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=8,
            NHK=2,
            D=128,
            actual_topk=[64, 48],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_gqa_32_4_heavy_pad(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[16, 8],
            pack_gqa=True,
            atol=HEAVY_PAD_ATOL,
        )


# ═══════════════════════════════════════════════════════════
# P3: MHA (NHQ = NHK, no pack_gqa)
# ═══════════════════════════════════════════════════════════


class TestSparseKvP3_MHA:
    """MHA where NHQ = NHK, pack_gqa=False."""

    def test_mha_4_4(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=4,
            NHK=4,
            D=128,
            actual_topk=[64, 48],
            pack_gqa=False,
        )

    def test_mha_1_1(self):
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=1,
            NHK=1,
            D=128,
            actual_topk=[128],
            pack_gqa=False,
        )

    def test_mha_4_4_swap_ab(self):
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=4,
            NHK=4,
            D=128,
            actual_topk=[64],
            pack_gqa=False,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_mha_4_4_no_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=4,
            NHK=4,
            D=128,
            actual_topk=[256],
            pack_gqa=False,
        )

    def test_mha_4_4_heavy_padding(self):
        _run_sparse_attn_vs_sdpa(
            B=2,
            S=256,
            NHQ=4,
            NHK=4,
            D=128,
            actual_topk=[16, 8],
            pack_gqa=False,
            atol=HEAVY_PAD_ATOL,
        )


# ═══════════════════════════════════════════════════════════
# P4: Edge cases / alignment
# ═══════════════════════════════════════════════════════════


class TestSparseKvP4_Edge:
    """Edge cases: non-aligned topk, extreme sparsity, head_dim=64."""

    def test_topk_not_tile_aligned(self):
        """topk=100 not a multiple of tile_size=128, requires internal padding."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[100],
            pack_gqa=True,
        )

    def test_topk_1_extreme_sparsity(self):
        """Each Q token only attends to 1 KV token."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[1],
            pack_gqa=True,
            atol=0.05,
        )

    def test_S_non_power_of_2(self):
        """S=300 is not a power of 2."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=300,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[64],
            pack_gqa=True,
        )

    def test_head_dim_64_mha(self):
        """head_dim=64 with MHA."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=4,
            NHK=4,
            D=64,
            actual_topk=[64],
            pack_gqa=False,
        )

    def test_head_dim_64_gqa_pack(self):
        """head_dim=64 with GQA + pack_gqa."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=8,
            NHK=2,
            D=64,
            actual_topk=[64],
            pack_gqa=True,
        )

    def test_topk_not_tile_aligned_swap_ab(self):
        """topk=50, not a multiple of tile_size=64 (SwapAB), GQA(32,4)."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=32,
            NHK=4,
            D=128,
            actual_topk=[50],
            pack_gqa=True,
            swap_ab=True,
            ref_block_size=(64, 64),
        )

    def test_fp16_dtype(self):
        """float16 instead of default bfloat16."""
        _run_sparse_attn_vs_sdpa(
            B=1,
            S=256,
            NHQ=128,
            NHK=1,
            D=128,
            actual_topk=[64],
            pack_gqa=True,
            dtype=torch.float16,
        )


# ═══════════════════════════════════════════════════════════
# Phase 2 skeleton: distributed (placeholder, expected to fail)
# ═══════════════════════════════════════════════════════════


@pytest.mark.skip(
    reason="distributed sparse not yet supported (dist_attn.py needs sparse_kv_indices passthrough)"
)
class TestSparseKvDistributed:
    """Placeholder for distributed sparse tests. Requires dist_attn.py changes."""

    def test_dist_mqa128_single_batch(self):
        """8-GPU, B=1, MQA(128), uniform topk."""
        pass

    def test_dist_mqa128_multi_batch(self):
        """8-GPU, B=4, MQA(128), variable topk per batch."""
        pass

    def test_dist_gqa_32_4(self):
        """8-GPU, B=2, GQA(32,4), pack_gqa=True."""
        pass
