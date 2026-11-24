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

import math
import unittest
from dataclasses import dataclass
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import range_write_index
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    is_sm100_family,
    to_ranges_tensor,
)
from magi_attention.testing.precision import assert_equal


@dataclass(frozen=True)
class _BwdCkptCase:
    range_merge: bool = False
    mask_type: int | None = None


def _ref_m_bounds_per_range(
    seqlen_q: int, seqlen_k: int, n_block: int, tile_m: int, tile_n: int, attn: int
) -> tuple[int, int]:
    if seqlen_q <= 0 or seqlen_k <= 0:
        return 0, 0
    if n_block < 0 or n_block >= math.ceil(seqlen_k / tile_n):
        return 0, 0
    m_min = 0
    m_max = math.ceil(seqlen_q / tile_m)
    if attn in (MT_MAP.inv_causal, MT_MAP.bi_causal):
        m_max = min(m_max, math.ceil(min(seqlen_k, (n_block + 1) * tile_n) / tile_m))
    if attn in (MT_MAP.causal, MT_MAP.bi_causal):
        m_idx_right = n_block * tile_n + seqlen_q - seqlen_k
        if m_idx_right > 0:
            m_min = min(m_max, m_idx_right // tile_m)
    return m_min, m_max


def _ref_dq_pair_k_tile_end_on_block(
    seqlen_q: int,
    seqlen_k: int,
    q_start: int,
    block: int,
    tile_m: int,
    tile_n: int,
    attn: int,
) -> int:
    if seqlen_q <= 0 or seqlen_k <= 0:
        return 0
    n_upper = math.ceil(seqlen_k / tile_n)
    b0 = q_start // tile_m
    mt_lo = block - b0 - 1 if (q_start % tile_m) != 0 and block > b0 else block - b0
    k_tile_end = 0
    for mt in range(mt_lo, block - b0 + 1):
        if mt * tile_m >= seqlen_q:
            continue
        for n in range(n_upper):
            m_lo, m_hi = _ref_m_bounds_per_range(
                seqlen_q, seqlen_k, n, tile_m, tile_n, attn
            )
            if m_lo <= mt < m_hi:
                k_tile_end = max(k_tile_end, n + 1)
    return k_tile_end


def _ref_fwd_conflict_ckpt(
    ranges: list[list[int]],
    num_blocks: int,
    num_sm: int,
    block_size: int,
    rows_per_token: int = 1,
) -> tuple[torch.Tensor, int]:
    num_ckpts, ckpt_stride = range_write_index.conflict_ckpt_shape(len(ranges), num_sm)
    want = torch.zeros(num_ckpts, num_blocks, dtype=torch.int32)
    running = torch.zeros(num_blocks, dtype=torch.int32)
    for c in range(num_ckpts):
        want[c] = running
        for r in range(c * ckpt_stride, min(len(ranges), (c + 1) * ckpt_stride)):
            s, e = ranges[r]
            if e > s:
                lo = s * rows_per_token // block_size
                hi = (e * rows_per_token - 1) // block_size + 1
                running[lo:hi] = r + 1
    return want, ckpt_stride


def _ref_bwd_conflict_ckpts(
    q_list: list[list[int]],
    k_list: list[list[int]],
    masks: list[int],
    group_of: list[int],
    num_q_blocks: int,
    num_k_blocks: int,
    num_sm: int,
    tile_m: int,
    cluster_tile_k: int,
    dq_stride: int,
    dkv_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_groups = len(k_list)
    num_ckpts, ckpt_stride = range_write_index.conflict_ckpt_shape(num_groups, num_sm)
    want_dq = torch.zeros(num_ckpts, num_q_blocks, dtype=torch.int32)
    want_dkv = torch.zeros(num_ckpts, num_k_blocks, dtype=torch.int32)
    run_dq = torch.zeros(num_q_blocks, dtype=torch.int32)
    run_dkv = torch.zeros(num_k_blocks, dtype=torch.int32)
    want_all_k = torch.zeros(num_groups, dtype=torch.uint8)

    for c in range(num_ckpts):
        want_dq[c] = run_dq
        want_dkv[c] = run_dkv
        g_lo = c * ckpt_stride
        g_hi = min(num_groups, g_lo + ckpt_stride)
        for row, g in enumerate(group_of):
            qs, qe = q_list[row]
            ks, ke = k_list[g]
            sq, sk = qe - qs, ke - ks
            if sq <= 0 or sk <= 0:
                continue
            attn = masks[row]
            if attn != MT_MAP.bi_causal or sq <= sk:
                want_all_k[g] = 1
            if not (g_lo <= g < g_hi):
                continue
            for b in range(qs // tile_m, (qe - 1) // tile_m + 1):
                k_tile_end = _ref_dq_pair_k_tile_end_on_block(
                    sq, sk, qs, b, tile_m, cluster_tile_k, attn
                )
                if k_tile_end > 0:
                    run_dq[b] = max(int(run_dq[b]), g * dq_stride + k_tile_end)
            kb0 = ks // cluster_tile_k
            for kb in range(kb0, (ke - 1) // cluster_tile_k + 1):
                active = False
                if (ks % cluster_tile_k) != 0 and kb > kb0:
                    lo, hi = _ref_m_bounds_per_range(
                        sq, sk, kb - kb0 - 1, tile_m, cluster_tile_k, attn
                    )
                    active = active or (lo < hi)
                n_b = kb - kb0
                if n_b * cluster_tile_k < sk:
                    lo, hi = _ref_m_bounds_per_range(
                        sq, sk, n_b, tile_m, cluster_tile_k, attn
                    )
                    active = active or (lo < hi)
                if active:
                    run_dkv[kb] = max(int(run_dkv[kb]), (g + 1) * dkv_stride)

    return want_dq, want_dkv, want_all_k, ckpt_stride


_BWD_PLAIN_FIXTURE = (
    [[0, 300], [64, 512], [10, 10], [200, 600], [0, 130], [350, 700]],
    [[0, 512], [100, 300], [0, 256], [256, 768], [128, 384], [50, 150]],
    [
        MT_MAP.full,
        MT_MAP.bi_causal,
        MT_MAP.causal,
        MT_MAP.causal,
        MT_MAP.inv_causal,
        MT_MAP.bi_causal,
    ],
    [0, 1, 2, 3, 4, 5],
)

_BWD_MERGE_FIXTURE = (
    [[0, 300], [200, 600], [0, 512], [10, 10], [64, 400], [0, 520], [128, 640]],
    [[0, 512], [100, 300], [256, 768], [50, 150], [400, 500]],
    [
        MT_MAP.causal,
        MT_MAP.full,
        MT_MAP.bi_causal,
        MT_MAP.full,
        MT_MAP.inv_causal,
        MT_MAP.bi_causal,
        MT_MAP.bi_causal,
    ],
    [0, 0, 1, 2, 2, 3, 4],
)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeWriteIndex(TestCase):
    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    @parameterize("num_sm", [1, 2, 16])
    def test_fwd_conflict_ckpt_matches_reference(self, num_sm: int):
        block_size = 128
        ranges = [
            [0, 256],
            [128, 256],
            [5, 300],
            [130, 140],
            [0, 129],
            [0, 40 * 128],
            [33 * 128 + 7, 35 * 128],
            [200, 200],
            [129, 384],
        ]
        num_blocks = max(e for _, e in ranges) // block_size + 2
        q_ranges = to_ranges_tensor(ranges, self.device)
        ckpt, ckpt_stride = range_write_index.build_fwd(
            q_ranges, num_blocks, num_sm, block_size, rows_per_token=1
        )
        want, want_stride = _ref_fwd_conflict_ckpt(
            ranges, num_blocks, num_sm, block_size
        )
        self.assertEqual(ckpt_stride, want_stride)
        assert_equal(ckpt.cpu(), want, print_no_mismatch=False)

    @parameterize("rows_per_token", [2, 4])
    def test_fwd_conflict_ckpt_scales_packed_row_blocks(self, rows_per_token: int):
        block_size = 128
        ranges = [[61, 100], [32, 64], [0, 33], [70, 70], [90, 130], [10, 80]]
        num_blocks = 130 * rows_per_token // block_size + 2
        q_ranges = to_ranges_tensor(ranges, self.device)
        for num_sm in (2, 16):
            ckpt, ckpt_stride = range_write_index.build_fwd(
                q_ranges, num_blocks, num_sm, block_size, rows_per_token=rows_per_token
            )
            want, want_stride = _ref_fwd_conflict_ckpt(
                ranges, num_blocks, num_sm, block_size, rows_per_token=rows_per_token
            )
            self.assertEqual(ckpt_stride, want_stride)
            assert_equal(ckpt.cpu(), want, print_no_mismatch=False)

    @parameterize(
        "case",
        [
            _BwdCkptCase(range_merge=False, mask_type=None),
            _BwdCkptCase(range_merge=False, mask_type=MT_MAP.full),
            _BwdCkptCase(range_merge=False, mask_type=MT_MAP.causal),
            _BwdCkptCase(range_merge=True, mask_type=None),
        ],
    )
    @parameterize("num_sm", [2, 16])
    def test_bwd_dq_and_dkv_conflict_ckpt_match_reference(
        self, case: _BwdCkptCase, num_sm: int
    ):
        tile_m, cluster_tile_k = 128, 256
        q_list, k_list, base_masks, group_of = (
            _BWD_MERGE_FIXTURE if case.range_merge else _BWD_PLAIN_FIXTURE
        )
        masks = (
            list(base_masks)
            if case.mask_type is None
            else [case.mask_type] * len(base_masks)
        )
        row_group = (
            torch.tensor(group_of, dtype=torch.int32, device=self.device)
            if case.range_merge
            else None
        )
        mask_t = (
            torch.tensor(masks, dtype=torch.int32, device=self.device)
            if case.mask_type is None
            else None
        )
        q_ranges = to_ranges_tensor(q_list, self.device)
        k_ranges = to_ranges_tensor(k_list, self.device)
        num_q_blocks = max(e for _, e in q_list) // tile_m + 2
        num_k_blocks = max(e for _, e in k_list) // cluster_tile_k + 2
        dq_stride, dkv_stride = 8, 4

        spec = range_write_index.BwdCkptSpec(
            tile_m=tile_m,
            cluster_tile_k=cluster_tile_k,
            mask_type=case.mask_type,
            range_merge=case.range_merge,
        )
        dq_ckpt, dq_ckpt_stride = range_write_index.build_bwd_dq(
            spec,
            q_ranges,
            k_ranges,
            mask_t,
            row_group,
            num_q_blocks,
            num_sm,
            dq_stride,
        )
        (
            dkv_ckpt,
            dkv_ckpt_stride,
            all_k_active,
        ) = range_write_index.build_bwd_dkv(
            spec,
            q_ranges,
            k_ranges,
            mask_t,
            row_group,
            num_k_blocks,
            num_sm,
            dkv_stride,
        )
        want_dq, want_dkv, want_all_k, want_stride = _ref_bwd_conflict_ckpts(
            q_list,
            k_list,
            masks,
            group_of,
            num_q_blocks,
            num_k_blocks,
            num_sm,
            tile_m,
            cluster_tile_k,
            dq_stride,
            dkv_stride,
        )

        self.assertEqual(dq_ckpt_stride, want_stride)
        self.assertEqual(dkv_ckpt_stride, want_stride)
        assert_equal(dq_ckpt.cpu(), want_dq, print_no_mismatch=False)
        assert_equal(dkv_ckpt.cpu(), want_dkv, print_no_mismatch=False)
        if case.range_merge:
            self.assertIsNotNone(all_k_active)
            assert all_k_active is not None
            assert_equal(all_k_active.cpu(), want_all_k, print_no_mismatch=False)
            # Groups 1, 3, 4 have only bi_causal pairs with seqlen_q > seqlen_k.
            self.assertEqual(all_k_active.cpu().tolist(), [1, 0, 1, 0, 0])
        else:
            self.assertIsNone(all_k_active)


if __name__ == "__main__":
    run_tests()
