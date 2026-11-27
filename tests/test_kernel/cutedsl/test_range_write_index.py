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
from typing import Optional
from unittest import TestCase

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from quack.compile_utils import make_fake_tensor as fake_tensor
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import range_write_index
from magi_attention.kernel.cutedsl.block_info import BlockInfo
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    is_sm100_family,
    to_ranges_tensor,
)
from magi_attention.testing.precision import assert_equal


@dataclass(frozen=True)
class _BwdFoldCase:
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


def _ref_fwd_last(
    ranges: list[list[int]],
    num_blocks: int,
    block_size: int,
    rows_per_token: int = 1,
) -> torch.Tensor:
    running = torch.zeros(num_blocks, dtype=torch.int32)
    for r, (s, e) in enumerate(ranges):
        if e > s:
            lo = s * rows_per_token // block_size
            hi = (e * rows_per_token - 1) // block_size + 1
            running[lo:hi] = r + 1
    return running


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


class _FoldFwdHarness:
    def __init__(self, block_size: int, rows_per_token: int):
        self.block_size = block_size
        self.rows_per_token = rows_per_token

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mLast: cute.Tensor,
        range_from: Int32,
        range_to: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(mQRanges, mLast, range_from, range_to).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mLast: cute.Tensor,
        range_from: Int32,
        range_to: Int32,
    ):
        range_write_index.fold_fwd_last(
            mQRanges,
            mLast,
            Int32(0),
            range_from,
            range_to,
            self.block_size,
            self.rows_per_token,
            cute.arch.thread_idx()[0],
            128,
        )


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
        range_write_index.fold_dq_last(
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
                range_write_index.fold_dkv_last(
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


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeWriteFold(TestCase):
    def setUp(self) -> None:
        self.device = torch.cuda.current_device()

    def _compile_fwd(self, block_size: int, rows_per_token: int):
        sym = cute.sym_int
        return cute.compile(
            _FoldFwdHarness(block_size, rows_per_token),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), sym()),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    @parameterize("rows_per_token", [1, 2, 4])
    def test_fwd_fold_matches_reference(self, rows_per_token: int):
        block_size = 128
        ranges = [
            [0, 256],
            [128, 256],
            [5, 300],
            [130, 140],
            [0, 129],
            [33 * 4 + 7, 35 * 4],
            [200, 200],
            [129, 384],
        ] * 6  # 48 ranges -> exercises >32-lane multi-round fold
        num_blocks = max(e for _, e in ranges) * rows_per_token // block_size + 2
        q_ranges = to_ranges_tensor(ranges, self.device)
        m_last = torch.zeros(1, num_blocks, dtype=torch.int32, device=self.device)
        fn = self._compile_fwd(block_size, rows_per_token)
        # Incremental fold: [0, 12) exercises delta < 32 (warp block-parallel),
        # [12, len) exercises delta >= 32 (thread range-parallel).
        fn(q_ranges, m_last, 0, 12)
        fn(q_ranges, m_last, 12, len(ranges))
        want = _ref_fwd_last(ranges, num_blocks, block_size, rows_per_token)
        assert_equal(m_last[0].cpu(), want, print_no_mismatch=False)

    @parameterize(
        "case",
        [
            _BwdFoldCase(range_merge=False, mask_type=None),
            _BwdFoldCase(range_merge=False, mask_type=MT_MAP.full),
            _BwdFoldCase(range_merge=False, mask_type=MT_MAP.causal),
            _BwdFoldCase(range_merge=True, mask_type=None),
        ],
    )
    def test_bwd_dq_and_dkv_fold_match_reference(self, case: _BwdFoldCase):
        tile_m, cluster_tile_k = 128, 256
        q_list, k_list, base_masks, group_of = (
            _BWD_MERGE_FIXTURE if case.range_merge else _BWD_PLAIN_FIXTURE
        )
        masks = (
            list(base_masks)
            if case.mask_type is None
            else [case.mask_type] * len(base_masks)
        )
        num_groups = len(k_list)
        cu_batches = None
        if case.range_merge:
            counts = [0] * num_groups
            for g in group_of:
                counts[g] += 1
            cu = [0]
            for c in counts:
                cu.append(cu[-1] + c)
            cu_batches = torch.tensor(cu, dtype=torch.int32, device=self.device)
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

        sym = cute.sym_int
        fn = cute.compile(
            _FoldBwdHarness(tile_m, cluster_tile_k, case.mask_type, case.range_merge),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), 2),
            _fake_i32(sym()) if case.mask_type is None else None,
            _fake_i32(sym()) if case.range_merge else None,
            _fake_i32(sym(), sym()),
            None if case.range_merge else _fake_i32(sym(), sym()),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        dq_last = torch.zeros(1, num_q_blocks, dtype=torch.int32, device=self.device)
        dkv_last = (
            None
            if case.range_merge
            else torch.zeros(1, num_k_blocks, dtype=torch.int32, device=self.device)
        )
        # Split at 7 so [0, 7) exercises delta < 32 (block-parallel) and
        # [7, num_groups) exercises delta >= 32 (range-parallel).
        mid = 7
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            cu_batches,
            dq_last,
            dkv_last,
            0,
            mid,
            dq_stride,
            dkv_stride,
        )
        fn(
            q_ranges,
            k_ranges,
            mask_t,
            cu_batches,
            dq_last,
            dkv_last,
            mid,
            num_groups,
            dq_stride,
            dkv_stride,
        )
        want_dq, want_dkv = _ref_bwd_last(
            q_list,
            k_list,
            masks,
            group_of,
            num_q_blocks,
            num_k_blocks,
            tile_m,
            cluster_tile_k,
            dq_stride,
            dkv_stride,
            range_merge=case.range_merge,
        )
        assert_equal(dq_last[0].cpu(), want_dq, print_no_mismatch=False)
        if not case.range_merge:
            assert dkv_last is not None and want_dkv is not None
            assert_equal(dkv_last[0].cpu(), want_dkv, print_no_mismatch=False)


if __name__ == "__main__":
    run_tests()
