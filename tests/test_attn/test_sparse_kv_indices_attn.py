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
reference.

Tier 1 (CI quick): PackGQA without swap, the most common DiT paths:
  - ratio 128 → kBlockM=128
  - ratio  64 → kBlockM=64 (full fill)
  - ratio  32 → kBlockM=64 (50% fill)

Tier 2 (Slow): extended coverage:
  - SwapAB paths (ratio 16, MHA, multi-KV-head GQA)
  - Cross-batch variable topk (pad to max)
  - Long sequence (S=8192, topk=1024)
  - k_block_size > 1 (32/128)

Known limitations:
  - Forward only (no backward)
  - k_block_size > 1 tests exist but kernel support is WIP (future: 32/64/128)
  - No distributed sparse yet
  - topk must be multiples of 128
  - SwapAB + MQA(128) not supported: kBlockM(<=64) < ratio(128)
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


def _build_sparse_kv_indices(B, NHK, S, topk, max_topk, device, k_block_size=1):
    """Build sparse_kv_indices (total_q, NHK, max_topk) with global KV row ids.

    total_q = B * S. Values are global row indices into the concatenated KV
    tensor of shape (B * NHK * S, 1, D), i.e. for batch b, head h, token t
    the global row is b * NHK * S + h * S + t.

    When k_block_size=1, values are token indices in [0, S).
    When k_block_size>1, values are block indices in [0, S // k_block_size).

    topk is a single int (same across all batches).
    """
    num_kv_blocks = S // k_block_size
    total_q = B * S
    indices = torch.full((total_q, NHK, max_topk), -1, dtype=torch.int32, device=device)
    for b_idx in range(B):
        for qi in range(S):
            row = b_idx * S + qi
            for h in range(NHK):
                perm = torch.randperm(num_kv_blocks, device=device)[:topk].sort().values
                global_ids = b_idx * NHK * S + h * S + perm
                indices[row, h, :topk] = global_ids.int()
    return indices


def _build_sdpa_mask(sparse_kv_indices, B, NHQ, NHK, S, device, k_block_size=1):
    """Build dense boolean mask [B, NHQ, S, S] from sparse_kv_indices for SDPA ref.

    sparse_kv_indices: (total_q, NHK, max_topk) with global KV row ids.
    Global row id = b * NHK * S + h * S + local_token.
    """
    mask = torch.zeros(B, NHQ, S, S, dtype=torch.bool, device=device)
    gqa = NHQ // NHK
    for b_idx in range(B):
        for qi in range(S):
            row = b_idx * S + qi
            for h_kv in range(NHK):
                global_ids = sparse_kv_indices[row, h_kv, :]
                valid = global_ids[global_ids >= 0].long()
                base = b_idx * NHK * S + h_kv * S
                local_ids = valid - base
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
    sparse_kv_indices,
    B,
    S,
    NHQ,
    NHK,
    pack_gqa,
    swap_ab=False,
    ref_block_size=None,
    k_block_size=1,
):
    """Run FFA with sparse_kv_indices and return reshaped output [B, S, NHQ, D]."""
    q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")

    with torch.no_grad():
        o_sparse, _ = flex_flash_attn_func(
            q_ffa.clone(),
            k_ffa.clone(),
            v_ffa.clone(),
            sparse_kv_indices=sparse_kv_indices,
            q_block_size=1,
            k_block_size=k_block_size,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )

    return rearrange(o_sparse, "(b h1 s) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S)


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


class TestSparseKvIndicesAttn(DistTestBase):
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
        """Run one sparse_kv_indices test config and assert against SDPA."""
        set_random_seed(SEED)
        B = cfg["B"]
        S = cfg["S"]
        NHQ = cfg["NHQ"]
        NHK = cfg["NHK"]
        D = cfg.get("D", 128)
        topk = cfg["topk"]
        max_topk = cfg.get("max_topk", topk)
        pack_gqa = cfg.get("pack_gqa", True)
        swap_ab = cfg.get("swap_ab", False)
        ref_block_size = cfg.get("ref_block_size", None)
        k_block_size = cfg.get("k_block_size", 1)
        dtype = cfg.get("dtype", torch.bfloat16)
        atol = cfg.get("atol", DEFAULT_ATOL)

        device = self.device

        sparse_kv_indices = _build_sparse_kv_indices(
            B, NHK, S, topk, max_topk, device, k_block_size=k_block_size
        )

        q = torch.randn(B, S, NHQ, D, dtype=dtype, device=device)
        k = torch.randn(B, S, NHK, D, dtype=dtype, device=device)
        v = torch.randn(B, S, NHK, D, dtype=dtype, device=device)

        o_ffa = _run_sparse_attn_and_get_output(
            q,
            k,
            v,
            sparse_kv_indices,
            B,
            S,
            NHQ,
            NHK,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
            k_block_size=k_block_size,
        )

        sdpa_mask = _build_sdpa_mask(
            sparse_kv_indices,
            B,
            NHQ,
            NHK,
            S,
            device,
            k_block_size=k_block_size,
        )

        test_case = (
            f"[NHQ={NHQ},NHK={NHK},S={S},B={B},D={D},"
            f"topk={topk},max_topk={max_topk},pack_gqa={pack_gqa},"
            f"swap_ab={swap_ab},k_block_size={k_block_size},dtype={dtype}]"
        )

        _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)

    # ─── Tier 1: CI quick (PackGQA, no swap) ────────────────

    @with_run_in_mp
    @parameterize(
        "config",
        [
            # ratio=128, kBlockM=128, PackGQA, no swap — canonical DiT
            {
                "name": "mqa128_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # ratio=64, kBlockM=64 full fill, PackGQA, no swap
            {
                "name": "mqa64_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 64,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
            # ratio=32, kBlockM=64 half fill, PackGQA, no swap
            {
                "name": "mqa32_packgqa",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 1,
                "topk": 128,
                "pack_gqa": True,
            },
        ],
    )
    def test_simple_sparse_kv_indices_attn(self, config: dict[str, Any]):
        self._run_config(config)

    # ─── Tier 2: Slow ─────────────────────────────────────

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            # SwapAB + PackGQA (ratio=16, boundary)
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
            # MHA ratio=1, no swap, no pack
            {
                "name": "mha_no_swap",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "topk": 128,
                "pack_gqa": False,
            },
            # MHA ratio=1 + SwapAB
            {
                "name": "mha_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "topk": 128,
                "pack_gqa": False,
                "swap_ab": True,
            },
            # Multi-KV-head GQA + SwapAB (ratio=4)
            {
                "name": "gqa8x2_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 8,
                "NHK": 2,
                "topk": 128,
                "pack_gqa": True,
                "swap_ab": True,
            },
            # Multi-batch with padding (topk < max_topk, padded with -1)
            {
                "name": "mqa128_with_padding",
                "B": 2,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "topk": 128,
                "max_topk": 256,
                "pack_gqa": True,
            },
            # Long sequence — main path
            {
                "name": "mqa128_long_seq",
                "B": 1,
                "S": 8192,
                "NHQ": 128,
                "NHK": 1,
                "topk": 1024,
                "pack_gqa": True,
            },
            # Long sequence — SwapAB path
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
            # Large S + high topk — regression test for INT32 overflow in
            # sparse_kv_indices row pointer (unique_idx * max_topk > INT32_MAX)
            {
                "name": "gqa16x4_large_s_high_topk",
                "B": 1,
                "S": 65536,
                "NHQ": 16,
                "NHK": 4,
                "topk": 9216,
                "pack_gqa": True,
            },
            # TODO: k_block_size > 1 — uncomment when kernel support is implemented
            # {
            #     "name": "mqa128_kblock32",
            #     "B": 1,
            #     "S": 256,
            #     "NHQ": 128,
            #     "NHK": 1,
            #     "topk": 4,
            #     "pack_gqa": True,
            #     "k_block_size": 32,
            # },
            # {
            #     "name": "mqa128_kblock128",
            #     "B": 1,
            #     "S": 256,
            #     "NHQ": 128,
            #     "NHK": 1,
            #     "topk": 2,
            #     "pack_gqa": True,
            #     "k_block_size": 128,
            # },
            # {
            #     "name": "mqa16_swapab_kblock32",
            #     "B": 1,
            #     "S": 256,
            #     "NHQ": 16,
            #     "NHK": 1,
            #     "topk": 4,
            #     "pack_gqa": True,
            #     "swap_ab": True,
            #     "k_block_size": 32,
            # },
            # {
            #     "name": "mqa16_swapab_kblock128",
            #     "B": 1,
            #     "S": 256,
            #     "NHQ": 16,
            #     "NHK": 1,
            #     "topk": 2,
            #     "pack_gqa": True,
            #     "swap_ab": True,
            #     "k_block_size": 128,
            # },
        ],
    )
    def test_sparse_kv_indices_attn(self, config: dict[str, Any]):
        self._run_config(config)


if __name__ == "__main__":
    run_tests()
