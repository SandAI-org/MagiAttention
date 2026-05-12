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
reference. Tests cover:

  - MQA(128,1) + pack_gqa=True  (most important DiT scenario)
  - GQA(32,4) + pack_gqa=True
  - GQA(8,2) + swap_ab (with compatible head configs)
  - MHA(4,4)
  - Edge cases (non-aligned topk, extreme sparsity, head_dim=64, fp16)
  - Multi-batch with variable topk

Known limitations:
  - Forward only (no backward)
  - k_block_size = 1 only (future: 32/64)
  - No distributed sparse yet
  - SwapAB + MQA(128) + pack_gqa not supported: kBlockM(<=64) < gqa_ratio(128)
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
HEAVY_PAD_ATOL = 0.02


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _build_sparse_kv_indices(B, NHK, S, actual_topk, max_topk, device):
    """Build sparse_kv_indices [B, NHK, S, max_topk] with random KV selection."""
    indices = torch.full((B, NHK, S, max_topk), -1, dtype=torch.int32, device=device)
    for b_idx in range(B):
        k = actual_topk[b_idx]
        for h in range(NHK):
            for qi in range(S):
                perm = torch.randperm(S, device=device)[:k].sort().values
                indices[b_idx, h, qi, :k] = perm.int()
    return indices


def _build_sdpa_mask(sparse_kv_indices, actual_topk, B, NHQ, NHK, S, device):
    """Build dense boolean mask [B, NHQ, S, S] from sparse_kv_indices for SDPA ref."""
    mask = torch.zeros(B, NHQ, S, S, dtype=torch.bool, device=device)
    gqa = NHQ // NHK
    for b_idx in range(B):
        vk = actual_topk[b_idx]
        for h_kv in range(NHK):
            for qi in range(S):
                valid_kv = sparse_kv_indices[b_idx, h_kv, qi, :vk]
                valid_kv = valid_kv[valid_kv >= 0]
                for h_q_offset in range(gqa):
                    h_q = h_kv * gqa + h_q_offset
                    mask[b_idx, h_q, qi, valid_kv.long()] = True
    return mask


def _run_sparse_attn_and_get_output(
    q,
    k,
    v,
    sparse_kv_indices,
    actual_topk,
    B,
    S,
    NHQ,
    NHK,
    pack_gqa,
    swap_ab=False,
    ref_block_size=None,
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
            actual_topk=actual_topk,
            q_block_size=1,
            k_block_size=1,
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
        actual_topk = cfg["actual_topk"]
        pack_gqa = cfg.get("pack_gqa", True)
        swap_ab = cfg.get("swap_ab", False)
        ref_block_size = cfg.get("ref_block_size", None)
        dtype = cfg.get("dtype", torch.bfloat16)
        atol = cfg.get("atol", DEFAULT_ATOL)

        max_topk = max(actual_topk)
        device = self.device

        sparse_kv_indices = _build_sparse_kv_indices(
            B, NHK, S, actual_topk, max_topk, device
        )

        q = torch.randn(B, S, NHQ, D, dtype=dtype, device=device)
        k = torch.randn(B, S, NHK, D, dtype=dtype, device=device)
        v = torch.randn(B, S, NHK, D, dtype=dtype, device=device)

        o_ffa = _run_sparse_attn_and_get_output(
            q,
            k,
            v,
            sparse_kv_indices,
            actual_topk,
            B,
            S,
            NHQ,
            NHK,
            pack_gqa=pack_gqa,
            swap_ab=swap_ab,
            ref_block_size=ref_block_size,
        )

        sdpa_mask = _build_sdpa_mask(
            sparse_kv_indices, actual_topk, B, NHQ, NHK, S, device
        )

        test_case = (
            f"[NHQ={NHQ},NHK={NHK},S={S},B={B},D={D},"
            f"topk={actual_topk},pack_gqa={pack_gqa},"
            f"swap_ab={swap_ab},dtype={dtype}]"
        )

        _compare_against_sdpa(o_ffa, q, k, v, sdpa_mask, B, NHQ, NHK, atol, test_case)

    # ─── CI quick tests ──────────────────────────────────

    @with_run_in_mp
    @parameterize(
        "config",
        [
            # P0: MQA(128,1) + pack_gqa — canonical DiT scenario
            {
                "name": "mqa128_topk_eq_S",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [256],
                "pack_gqa": True,
            },
            {
                "name": "mqa128_topk_lt_S",
                "B": 1,
                "S": 512,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [128],
                "pack_gqa": True,
            },
            {
                "name": "mqa128_heavy_pad",
                "B": 1,
                "S": 512,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [16],
                "pack_gqa": True,
                "atol": HEAVY_PAD_ATOL,
            },
            {
                "name": "mqa128_multi_batch",
                "B": 3,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [64, 128, 48],
                "pack_gqa": True,
            },
            # P2: GQA(32,4) + pack_gqa
            {
                "name": "gqa_32_4_basic",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 4,
                "actual_topk": [128],
                "pack_gqa": True,
            },
            # P3: MHA(4,4)
            {
                "name": "mha_4_4",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "actual_topk": [64],
                "pack_gqa": False,
            },
        ],
    )
    def test_simple_sparse_kv_indices_attn(self, config: dict[str, Any]):
        self._run_config(config)

    # ─── Full coverage (slow) ────────────────────────────

    @pytest.mark.slow
    @with_run_in_mp
    @parameterize(
        "config",
        [
            # P0: MQA extended
            {
                "name": "mqa128_topk_eq_S",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [256],
                "pack_gqa": True,
            },
            {
                "name": "mqa128_topk_lt_S",
                "B": 1,
                "S": 512,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [128],
                "pack_gqa": True,
            },
            {
                "name": "mqa128_heavy_pad",
                "B": 1,
                "S": 512,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [16],
                "pack_gqa": True,
                "atol": HEAVY_PAD_ATOL,
            },
            {
                "name": "mqa128_multi_batch_uniform",
                "B": 3,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [64, 64, 64],
                "pack_gqa": True,
            },
            {
                "name": "mqa128_multi_batch_variable",
                "B": 3,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [64, 128, 48],
                "pack_gqa": True,
            },
            # P1: SwapAB (compatible head configs only, GQA ratio must fit kBlockM)
            {
                "name": "gqa_8_2_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 8,
                "NHK": 2,
                "actual_topk": [64],
                "pack_gqa": True,
                "swap_ab": True,
            },
            {
                "name": "mha_4_4_swapab",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "actual_topk": [64],
                "pack_gqa": False,
                "swap_ab": True,
            },
            # P2: GQA(32,4) + pack_gqa
            {
                "name": "gqa_32_4_basic",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 4,
                "actual_topk": [128],
                "pack_gqa": True,
            },
            {
                "name": "gqa_32_4_heavy_pad",
                "B": 1,
                "S": 512,
                "NHQ": 32,
                "NHK": 4,
                "actual_topk": [16],
                "pack_gqa": True,
                "atol": HEAVY_PAD_ATOL,
            },
            {
                "name": "gqa_32_4_multi_batch",
                "B": 2,
                "S": 256,
                "NHQ": 32,
                "NHK": 4,
                "actual_topk": [64, 128],
                "pack_gqa": True,
            },
            # P3: MHA
            {
                "name": "mha_4_4",
                "B": 1,
                "S": 256,
                "NHQ": 4,
                "NHK": 4,
                "actual_topk": [64],
                "pack_gqa": False,
            },
            {
                "name": "mha_8_8",
                "B": 1,
                "S": 256,
                "NHQ": 8,
                "NHK": 8,
                "actual_topk": [128],
                "pack_gqa": False,
            },
            # P4: Edge cases
            {
                "name": "non_aligned_topk",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [100],
                "pack_gqa": True,
            },
            {
                "name": "topk_1_extreme",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [1],
                "pack_gqa": True,
                "atol": 0.05,
            },
            {
                "name": "hd64",
                "B": 1,
                "S": 256,
                "NHQ": 32,
                "NHK": 4,
                "D": 64,
                "actual_topk": [64],
                "pack_gqa": True,
            },
            {
                "name": "fp16",
                "B": 1,
                "S": 256,
                "NHQ": 128,
                "NHK": 1,
                "actual_topk": [128],
                "pack_gqa": True,
                "dtype": torch.float16,
            },
        ],
    )
    def test_sparse_kv_indices_attn(self, config: dict[str, Any]):
        self._run_config(config)


if __name__ == "__main__":
    run_tests()
