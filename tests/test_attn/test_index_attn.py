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
Tests for index_attn_indices direct-to-kernel path (forward only).

Validates flex_flash_attn_func with index_attn_indices against PyTorch SDPA
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
  - Forward only (no backward)
  - k_block_size > 1 tests commented out, kernel support is WIP (future: 32/64/128)
  - No distributed sparse yet
  - max_topk must be multiples of tile_size (asserted in flex_flash_attn_func)
  - Q/K/V are packed in (b, s, h) order to match index_attn_indices view layout
"""

from typing import Any

import pytest
import torch
from einops import rearrange
from torch.testing._internal.common_utils import run_tests

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.utils import set_random_seed

SEED = 42
DEFAULT_ATOL = 0.01


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _build_index_attn_indices(
    B, NHK, S_q, S_kv, topk, max_topk, device, k_block_size=1
):
    """Build index_attn_indices (total_q, NHK, max_topk) with global KV row ids.

    total_q = B * S_q. Values are global row indices into the concatenated KV
    tensor packed in (b, s, h) order: shape (B * S_kv * NHK, 1, D).
    For batch b, token t, head h the global row is (b * S_kv + t) * NHK + h.

    topk: int or list[int]. If list, per-batch topk (length must be B).
    """
    num_kv_blocks = S_kv // k_block_size
    total_q = B * S_q
    if isinstance(topk, int):
        topk_per_batch = [topk] * B
    else:
        assert len(topk) == B
        topk_per_batch = topk
    indices = torch.full((total_q, NHK, max_topk), -1, dtype=torch.int32, device=device)
    for b_idx in range(B):
        tk = topk_per_batch[b_idx]
        for qi in range(S_q):
            row = b_idx * S_q + qi
            for h in range(NHK):
                perm = torch.randperm(num_kv_blocks, device=device)[:tk].sort().values
                global_ids = (b_idx * S_kv + perm) * NHK + h
                indices[row, h, :tk] = global_ids.int()
    return indices


def _build_sdpa_mask(
    index_attn_indices, B, NHQ, NHK, S_q, S_kv, device, k_block_size=1
):
    """Build dense boolean mask [B, NHQ, S_q, S_kv] from index_attn_indices for SDPA ref.

    index_attn_indices: (total_q, NHK, max_topk) with global KV row ids.
    Global row id = (b * S_kv + local_token) * NHK + h  (b,s,h order).
    """
    mask = torch.zeros(B, NHQ, S_q, S_kv, dtype=torch.bool, device=device)
    gqa = NHQ // NHK
    for b_idx in range(B):
        for qi in range(S_q):
            row = b_idx * S_q + qi
            for h_kv in range(NHK):
                global_ids = index_attn_indices[row, h_kv, :]
                valid = global_ids[global_ids >= 0].long()
                local_ids = valid // NHK - b_idx * S_kv
                if k_block_size == 1:
                    kv_tokens = local_ids
                else:
                    kv_tokens = torch.cat(
                        [
                            torch.arange(
                                bi * k_block_size,
                                bi * k_block_size + k_block_size,
                                device=device,
                            )
                            for bi in local_ids
                        ]
                    )
                for h_q_offset in range(gqa):
                    h_q = h_kv * gqa + h_q_offset
                    mask[b_idx, h_q, qi, kv_tokens] = True
    return mask


def _run_sparse_attn_and_get_output(
    q,
    k,
    v,
    index_attn_indices,
    B,
    S_q,
    S_kv,
    NHQ,
    NHK,
    pack_gqa,
    swap_ab=False,
    ref_block_size=None,
    k_block_size=1,
):
    """Run FFA with index_attn_indices and return reshaped output [B, S_q, NHQ, D]."""
    q_ffa = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b s h) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b s h) 1 d")

    with torch.no_grad():
        o_sparse, _ = flex_flash_attn_func(
            q_ffa.clone(),
            k_ffa.clone(),
            v_ffa.clone(),
            index_attn_indices=index_attn_indices,
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


# ═══════════════════════════════════════════════════════════
# Test class
# ═══════════════════════════════════════════════════════════


class TestIndexAttnIndicesAttn(DistTestBase):
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

    def _run_config(self, cfg: dict[str, Any]):
        """Run one index_attn_indices test config and assert against SDPA."""
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

        device = self.device

        index_attn_indices = _build_index_attn_indices(
            B, NHK, S_q, S_kv, topk, max_topk, device, k_block_size=k_block_size
        )

        q = torch.randn(B, S_q, NHQ, D, dtype=dtype, device=device)
        k = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)
        v = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)

        o_ffa = _run_sparse_attn_and_get_output(
            q,
            k,
            v,
            index_attn_indices,
            B,
            S_q,
            S_kv,
            NHQ,
            NHK,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            k_block_size=k_block_size,
        )

        sdpa_mask = _build_sdpa_mask(
            index_attn_indices,
            B,
            NHQ,
            NHK,
            S_q,
            S_kv,
            device,
            k_block_size=k_block_size,
        )

        test_case = (
            f"[NHQ={NHQ},NHK={NHK},S_q={S_q},S_kv={S_kv},B={B},D={D},"
            f"topk={topk},max_topk={max_topk},pack_gqa={pack_gqa},"
            f"swap_ab={swap_ab},k_block_size={k_block_size},dtype={dtype}]"
        )

        _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)

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
            # ratio=16, small Q tile → SwapAB + PackGQA
            {
                "name": "mqa16_packgqa_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 16,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
                "swap_ab": True,
            },
        ],
    )
    def test_simple_index_attn_indices_attn(self, config: dict[str, Any]):
        self._run_config(config)

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
        self._run_config(config)

    # ─── Tier 2b: Q/KV different lengths ─────────────────────
    # Real scenario: Q is short (e.g. new tokens), KV pool is large.

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
            # Very short Q (single tile)
            {
                "name": "tiny_q",
                "B": 1,
                "S_q": 8,
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
        self._run_config(config)

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
        self._run_config(config)

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
        self._run_config(config)

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
        self._run_config(config)

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
        self._run_config(config)

    # ─── Tier 3e: k_block_size > 1 (WIP, commented out) ───

    # @pytest.mark.slow
    # @with_run_in_mp
    # @parameterize(
    #     "config",
    #     [
    #         {
    #             "name": "mqa128_kblock32",
    #             "B": 1,
    #             "S": 256,
    #             "NHQ": 128,
    #             "NHK": 1,
    #             "topk": 4,
    #             "pack_gqa": True,
    #             "k_block_size": 32,
    #         },
    #         {
    #             "name": "mqa128_kblock128",
    #             "B": 1,
    #             "S": 256,
    #             "NHQ": 128,
    #             "NHK": 1,
    #             "topk": 2,
    #             "pack_gqa": True,
    #             "k_block_size": 128,
    #         },
    #     ],
    # )
    # def test_sparse_k_block_size(self, config: dict[str, Any]):
    #     self._run_config(config)


if __name__ == "__main__":
    run_tests()
