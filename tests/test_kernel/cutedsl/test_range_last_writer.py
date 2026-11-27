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
from typing import Optional
from unittest import TestCase

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from quack.compile_utils import make_fake_tensor as fake_tensor
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import range_last_writer, range_write_order
from magi_attention.kernel.cutedsl.block_info import BlockInfo
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    is_sm100_family,
    to_ranges_tensor,
)
from magi_attention.testing.precision import assert_equal


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


def _ref_dq_pair_prev_k_tile_on_block(
    seqlen_q: int,
    seqlen_k: int,
    q_start: int,
    block: int,
    tile_m: int,
    tile_n: int,
    attn: int,
) -> int:
    if seqlen_q <= 0 or seqlen_k <= 0:
        return -1
    n_upper = math.ceil(seqlen_k / tile_n)
    b0 = q_start // tile_m
    mt_lo = block - b0 - 1 if (q_start % tile_m) != 0 and block > b0 else block - b0
    prev_n = -1
    for mt in range(mt_lo, block - b0 + 1):
        if mt * tile_m >= seqlen_q:
            continue
        for n in range(n_upper):
            m_lo, m_hi = _ref_m_bounds_per_range(
                seqlen_q, seqlen_k, n, tile_m, tile_n, attn
            )
            if m_lo <= mt < m_hi:
                prev_n = max(prev_n, n)
    return prev_n


def _ref_bwd_last(
    q_list: list[list[int]],
    k_list: list[list[int]],
    masks: list[int],
    group_of: list[int],
    num_q_blocks: int,
    num_k_blocks: int,
    tile_m: int,
    cluster_tile_k: int,
    dq_stride: int,
    dkv_stride: int,
    range_merge: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    run_dq = torch.zeros(num_q_blocks, dtype=torch.int32)
    run_dkv = None if range_merge else torch.zeros(num_k_blocks, dtype=torch.int32)
    for row, g in enumerate(group_of):
        qs, qe = q_list[row]
        ks, ke = k_list[g]
        sq, sk = qe - qs, ke - ks
        if sq <= 0 or sk <= 0:
            continue
        attn = masks[row]
        for b in range(qs // tile_m, (qe - 1) // tile_m + 1):
            prev_n = _ref_dq_pair_prev_k_tile_on_block(
                sq, sk, qs, b, tile_m, cluster_tile_k, attn
            )
            if prev_n >= 0:
                run_dq[b] = max(int(run_dq[b]), g * dq_stride + prev_n + 1)
        if run_dkv is not None:
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
    return run_dq, run_dkv


class _FoldBwdHarness:
    def __init__(
        self,
        tile_m: int,
        cluster_tile_k: int,
        mask_type: int | None,
        range_merge: bool,
    ):
        self.tile_m = tile_m
        self.cluster_tile_k = cluster_tile_k
        self.mask_type = mask_type
        self.range_merge = range_merge
        self.use_per_range_mask = mask_type is None

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mCuBatches: Optional[cute.Tensor],
        mDQLast: cute.Tensor,
        mDKVLast: Optional[cute.Tensor],
        range_from: Int32,
        range_to: Int32,
        dq_stride: Int32,
        dkv_stride: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mDQLast,
            mDKVLast,
            range_from,
            range_to,
            dq_stride,
            dkv_stride,
        ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mCuBatches: Optional[cute.Tensor],
        mDQLast: cute.Tensor,
        mDKVLast: Optional[cute.Tensor],
        range_from: Int32,
        range_to: Int32,
        dq_stride: Int32,
        dkv_stride: Int32,
    ):
        block_info = BlockInfo(
            self.tile_m,
            self.cluster_tile_k,
            self.mask_type is None or self.mask_type == MT_MAP.causal,
        )
        tidx = cute.arch.thread_idx()[0]
        range_last_writer.fold_dq_last(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mDQLast,
            Int32(0),
            range_from,
            range_to,
            dq_stride,
            self.tile_m,
            self.range_merge,
            self.use_per_range_mask,
            tidx,
            128,
        )
        if cutlass.const_expr(not self.range_merge):
            assert mDKVLast is not None
            if tidx < cute.arch.WARP_SIZE:
                range_last_writer.fold_dkv_last(
                    block_info,
                    mQRanges,
                    mKRanges,
                    mMaskTypes,
                    mDKVLast,
                    Int32(0),
                    range_from,
                    range_to,
                    dkv_stride,
                    self.cluster_tile_k,
                    self.use_per_range_mask,
                )


def _ref_dq_pair_needs_last(
    seqlen_q: int, seqlen_k: int, k_tile: int, tile_m: int, tile_n: int, attn: int
) -> bool:
    """Whether some M tile of the pair has ``k_tile`` as its first K tile."""
    if seqlen_q <= 0 or seqlen_k <= 0:
        return False
    for mt in range(math.ceil(seqlen_q / tile_m)):
        for n in range(math.ceil(seqlen_k / tile_n)):
            m_lo, m_hi = _ref_m_bounds_per_range(
                seqlen_q, seqlen_k, n, tile_m, tile_n, attn
            )
            if m_lo <= mt < m_hi:
                if n == k_tile:
                    return True
                break
    return False


class _DqNeedsLastHarness:
    """Stores ``dq_pair_needs_last`` (one pair per row) or ``dq_group_needs_last``
    (one RangeMerge group per row) for every K tile of the row into ``mNeeds``."""

    def __init__(self, tile_m: int, cluster_tile_k: int, range_merge: bool):
        self.tile_m = tile_m
        self.cluster_tile_k = cluster_tile_k
        self.range_merge = range_merge

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: cute.Tensor,
        mCuBatches: Optional[cute.Tensor],
        mNeeds: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        self.kernel(mQRanges, mKRanges, mMaskTypes, mCuBatches, mNeeds).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: cute.Tensor,
        mCuBatches: Optional[cute.Tensor],
        mNeeds: cute.Tensor,
    ):
        block_info = BlockInfo(self.tile_m, self.cluster_tile_k, True)
        if cute.arch.thread_idx()[0] == 0:
            row = Int32(0)
            while row < mNeeds.shape[0]:
                seqlen_k = mKRanges[row, 1] - mKRanges[row, 0]
                k_tile = Int32(0)
                while k_tile * self.cluster_tile_k < seqlen_k:
                    needs = cutlass.Boolean(False)
                    if cutlass.const_expr(self.range_merge):
                        assert mCuBatches is not None
                        needs = range_write_order.dq_group_needs_last(
                            block_info,
                            mQRanges,
                            mMaskTypes,
                            mCuBatches[row],
                            mCuBatches[row + 1],
                            seqlen_k,
                            k_tile,
                        )
                    else:
                        seqlen_q = mQRanges[row, 1] - mQRanges[row, 0]
                        attn = Int32(mMaskTypes[row])
                        m_lo, m_hi = range_last_writer.pair_m_bounds(
                            block_info, seqlen_q, seqlen_k, k_tile, attn, True
                        )
                        needs = range_write_order.dq_pair_needs_last(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            k_tile,
                            m_lo,
                            m_hi,
                            attn,
                            True,
                        )
                    if needs:
                        mNeeds[row, k_tile] = Int32(1)
                    k_tile += 1
                row += 1


def _fake_i32(*shape):
    return fake_tensor(Int32, shape, divisibility=1)


_BWD_PLAIN_FIXTURE = (
    [[0, 300], [64, 512], [10, 10], [200, 600], [0, 130], [350, 700]] * 7,
    [[0, 512], [100, 300], [0, 256], [256, 768], [128, 384], [50, 150]] * 7,
    [
        MT_MAP.full,
        MT_MAP.bi_causal,
        MT_MAP.causal,
        MT_MAP.causal,
        MT_MAP.inv_causal,
        MT_MAP.bi_causal,
    ]
    * 7,
    list(range(42)),
)

_BWD_MERGE_FIXTURE = (
    [[0, 300], [200, 600], [0, 512], [10, 10], [64, 400], [0, 520], [128, 640]] * 8,
    [[0, 512], [100, 300], [256, 768], [50, 150], [400, 500]] * 8,
    [
        MT_MAP.causal,
        MT_MAP.full,
        MT_MAP.bi_causal,
        MT_MAP.full,
        MT_MAP.inv_causal,
        MT_MAP.bi_causal,
        MT_MAP.bi_causal,
    ]
    * 8,
    [g + 5 * rep for rep in range(8) for g in (0, 0, 1, 2, 2, 3, 4)],
)

# K groups with the (Q range, mask) rows that share each of them. Every mask
# type appears with seqlen_q below, equal to and above seqlen_k, so M tiles
# start on K tile 0 as well as on later K tiles.
_DQ_NEEDS_LAST_GROUPS: list[tuple[list[int], list[tuple[list[int], int]]]] = [
    (
        [0, 1500],
        [
            ([0, 1200], MT_MAP.full),
            ([64, 1100], MT_MAP.causal),
            ([200, 1400], MT_MAP.inv_causal),
            ([0, 1500], MT_MAP.bi_causal),
        ],
    ),
    (
        [100, 1900],
        [
            ([0, 700], MT_MAP.inv_causal),
            ([64, 600], MT_MAP.bi_causal),
            ([10, 1046], MT_MAP.causal),
        ],
    ),
    (
        [0, 700],
        [
            ([0, 1500], MT_MAP.causal),
            ([350, 1700], MT_MAP.inv_causal),
            ([10, 10], MT_MAP.causal),
            ([0, 1600], MT_MAP.bi_causal),
        ],
    ),
    ([256, 1156], [([10, 900], MT_MAP.causal)]),
    ([300, 2100], [([0, 520], MT_MAP.bi_causal), ([0, 1300], MT_MAP.full)]),
    ([512, 512], [([0, 300], MT_MAP.full)]),
    ([0, 1300], [([128, 1428], MT_MAP.causal), ([0, 1300], MT_MAP.inv_causal)]),
]

_TILE_M = 128
_CLUSTER_TILE_K = 256
_DQ_STRIDE = 8
_DKV_STRIDE = 4
# Split at 7 so [0, 7) exercises delta < 32 (block-parallel) and
# [7, num_groups) exercises delta >= 32 (range-parallel).
_FOLD_SPLIT = 7


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeWriteFold(TestCase):
    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    def _check_plain_fold(
        self,
        compile_mask: int | None,
        masks: list[int],
        mask_t: torch.Tensor | None,
        mask_fake,
        test_case: str,
    ) -> None:
        """fold_dq_last/fold_dkv_last of the plain fixture against the oracle.

        ``compile_mask=None`` runs the per-range-mask kernel reading
        ``mMaskTypes``; an ``MT_MAP`` value compiles that mask in statically.
        """
        q_list, k_list, _, group_of = _BWD_PLAIN_FIXTURE
        num_groups = len(k_list)
        q_ranges = to_ranges_tensor(q_list, self.device)
        k_ranges = to_ranges_tensor(k_list, self.device)
        num_q_blocks = max(e for _, e in q_list) // _TILE_M + 2
        num_k_blocks = max(e for _, e in k_list) // _CLUSTER_TILE_K + 2

        sym = cute.sym_int
        fn = cute.compile(
            _FoldBwdHarness(_TILE_M, _CLUSTER_TILE_K, compile_mask, False),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), 2),
            mask_fake,
            None,
            _fake_i32(sym(), sym()),
            _fake_i32(sym(), sym()),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        dq_last = torch.zeros(1, num_q_blocks, dtype=torch.int32, device=self.device)
        dkv_last = torch.zeros(1, num_k_blocks, dtype=torch.int32, device=self.device)
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            None,
            dq_last,
            dkv_last,
            0,
            _FOLD_SPLIT,
            _DQ_STRIDE,
            _DKV_STRIDE,
        )
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            None,
            dq_last,
            dkv_last,
            _FOLD_SPLIT,
            num_groups,
            _DQ_STRIDE,
            _DKV_STRIDE,
        )
        want_dq, want_dkv = _ref_bwd_last(
            q_list,
            k_list,
            masks,
            group_of,
            num_q_blocks,
            num_k_blocks,
            _TILE_M,
            _CLUSTER_TILE_K,
            _DQ_STRIDE,
            _DKV_STRIDE,
            range_merge=False,
        )
        assert_equal(
            dq_last[0].cpu(), want_dq, test_case=test_case, print_no_mismatch=False
        )
        assert want_dkv is not None
        assert_equal(
            dkv_last[0].cpu(), want_dkv, test_case=test_case, print_no_mismatch=False
        )

    def test_bwd_fold_per_range_masks_match_reference(self):
        masks = _BWD_PLAIN_FIXTURE[2]
        self._check_plain_fold(
            None,
            masks,
            torch.tensor(masks, dtype=torch.int32, device=self.device),
            _fake_i32(cute.sym_int()),
            "[fold][per_range]",
        )

    def test_bwd_fold_compile_time_full_matches_reference(self):
        self._check_plain_fold(
            MT_MAP.full,
            [MT_MAP.full] * len(_BWD_PLAIN_FIXTURE[2]),
            None,
            None,
            "[fold][full]",
        )

    def test_bwd_fold_compile_time_causal_matches_reference(self):
        self._check_plain_fold(
            MT_MAP.causal,
            [MT_MAP.causal] * len(_BWD_PLAIN_FIXTURE[2]),
            None,
            None,
            "[fold][causal]",
        )

    def test_bwd_dq_fold_range_merge_matches_reference(self):
        test_case = "[fold][range_merge]"
        q_list, k_list, masks, group_of = _BWD_MERGE_FIXTURE
        num_groups = len(k_list)
        counts = [0] * num_groups
        for g in group_of:
            counts[g] += 1
        cu = [0]
        for c in counts:
            cu.append(cu[-1] + c)
        cu_batches = torch.tensor(cu, dtype=torch.int32, device=self.device)
        mask_t = torch.tensor(masks, dtype=torch.int32, device=self.device)
        q_ranges = to_ranges_tensor(q_list, self.device)
        k_ranges = to_ranges_tensor(k_list, self.device)
        num_q_blocks = max(e for _, e in q_list) // _TILE_M + 2
        num_k_blocks = max(e for _, e in k_list) // _CLUSTER_TILE_K + 2

        sym = cute.sym_int
        fn = cute.compile(
            _FoldBwdHarness(_TILE_M, _CLUSTER_TILE_K, None, True),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), 2),
            _fake_i32(sym()),
            _fake_i32(sym()),
            _fake_i32(sym(), sym()),
            None,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        dq_last = torch.zeros(1, num_q_blocks, dtype=torch.int32, device=self.device)
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            cu_batches,
            dq_last,
            None,
            0,
            _FOLD_SPLIT,
            _DQ_STRIDE,
            _DKV_STRIDE,
        )
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            cu_batches,
            dq_last,
            None,
            _FOLD_SPLIT,
            num_groups,
            _DQ_STRIDE,
            _DKV_STRIDE,
        )
        want_dq, _ = _ref_bwd_last(
            q_list,
            k_list,
            masks,
            group_of,
            num_q_blocks,
            num_k_blocks,
            _TILE_M,
            _CLUSTER_TILE_K,
            _DQ_STRIDE,
            _DKV_STRIDE,
            range_merge=True,
        )
        assert_equal(
            dq_last[0].cpu(), want_dq, test_case=test_case, print_no_mismatch=False
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestDqNeedsLast(TestCase):
    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    def _run(
        self,
        q_list: list[list[int]],
        k_list: list[list[int]],
        masks: list[int],
        cu: list[int] | None,
        num_k_tiles: int,
    ) -> torch.Tensor:
        sym = cute.sym_int
        fn = cute.compile(
            _DqNeedsLastHarness(_TILE_M, _CLUSTER_TILE_K, cu is not None),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), 2),
            _fake_i32(sym()),
            None if cu is None else _fake_i32(sym()),
            _fake_i32(sym(), sym()),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        needs = torch.zeros(
            len(k_list), num_k_tiles, dtype=torch.int32, device=self.device
        )
        fn(
            to_ranges_tensor(q_list, self.device),
            to_ranges_tensor(k_list, self.device),
            torch.tensor(masks, dtype=torch.int32, device=self.device),
            None
            if cu is None
            else torch.tensor(cu, dtype=torch.int32, device=self.device),
            needs,
        )
        return needs.cpu()

    def test_dq_pair_needs_last_matches_reference(self):
        test_case = "[dq_needs_last][pair]"
        q_list = [q for _, rows in _DQ_NEEDS_LAST_GROUPS for q, _ in rows]
        k_list = [k for k, rows in _DQ_NEEDS_LAST_GROUPS for _ in rows]
        masks = [mask for _, rows in _DQ_NEEDS_LAST_GROUPS for _, mask in rows]
        num_k_tiles = max(math.ceil((e - s) / _CLUSTER_TILE_K) for s, e in k_list)
        want = torch.zeros(len(k_list), num_k_tiles, dtype=torch.int32)
        for row, ((qs, qe), (ks, ke), mask) in enumerate(zip(q_list, k_list, masks)):
            for k_tile in range(math.ceil((ke - ks) / _CLUSTER_TILE_K)):
                want[row, k_tile] = _ref_dq_pair_needs_last(
                    qe - qs, ke - ks, k_tile, _TILE_M, _CLUSTER_TILE_K, mask
                )
        self.assertTrue(bool(want[:, 1:].any()))
        assert_equal(
            self._run(q_list, k_list, masks, None, num_k_tiles),
            want,
            test_case=test_case,
            print_no_mismatch=False,
        )

    def test_dq_group_needs_last_matches_reference(self):
        test_case = "[dq_needs_last][group]"
        q_list = [q for _, rows in _DQ_NEEDS_LAST_GROUPS for q, _ in rows]
        k_list = [k for k, _ in _DQ_NEEDS_LAST_GROUPS]
        masks = [mask for _, rows in _DQ_NEEDS_LAST_GROUPS for _, mask in rows]
        cu = [0]
        for _, rows in _DQ_NEEDS_LAST_GROUPS:
            cu.append(cu[-1] + len(rows))
        num_k_tiles = max(math.ceil((e - s) / _CLUSTER_TILE_K) for s, e in k_list)
        want = torch.zeros(len(k_list), num_k_tiles, dtype=torch.int32)
        for g, ((ks, ke), rows) in enumerate(_DQ_NEEDS_LAST_GROUPS):
            for k_tile in range(math.ceil((ke - ks) / _CLUSTER_TILE_K)):
                want[g, k_tile] = any(
                    _ref_dq_pair_needs_last(
                        qe - qs, ke - ks, k_tile, _TILE_M, _CLUSTER_TILE_K, mask
                    )
                    for (qs, qe), mask in rows
                )
        self.assertTrue(bool(want[:, 1:].any()))
        assert_equal(
            self._run(q_list, k_list, masks, cu, num_k_tiles),
            want,
            test_case=test_case,
            print_no_mismatch=False,
        )


if __name__ == "__main__":
    run_tests()
