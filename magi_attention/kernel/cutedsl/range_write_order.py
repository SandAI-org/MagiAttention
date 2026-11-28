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

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

# isort: split
from quack.cute_dsl_utils import ParamsBase

from .block_info import BlockInfo
from .cutedsl_utils import (
    elem_pointer,
    ld_acquire,
    ld_acquire_cta,
    red_max_relaxed_cta,
    st_global_release_gpu_i32,
)
from .ffa_utils import MT_MAP
from .seqlen_info import SeqlenInfoQK

# ---------------------------------------------------------------------------
# Core write-order primitives
# ---------------------------------------------------------------------------


@dataclass
class WriteOrderTensors(ParamsBase):
    """Write order of one output and per-CTA last-writer state.

    ``mWriteOrder[b, h] = [published, count]``. Writer ``j`` of range ``r`` is
    ``writer_number(wo, r, j)`` (0: none), so ``writer_stride`` is the count of
    writer numbers per range. ``mLast[cta, b]`` holds the highest writer on
    block ``b`` folded by CTA ``cta`` so far (0: none).
    """

    mWriteOrder: cute.Tensor
    mLast: cute.Tensor
    writer_stride: Int32


@cute.jit
def writer_number(
    wo: WriteOrderTensors, range_idx: Int32, rank: Int32 = Int32(0)
) -> Int32:
    """1-based writer number for ``rank`` of ``range_idx``; 0 means none."""
    return range_idx * wo.writer_stride + rank + 1


@cute.jit
def last_writer(wo: WriteOrderTensors, slot: Int32, block: Int32) -> Int32:
    """Highest writer that CTA ``slot`` folded on ``block`` so far (0: none)."""
    return ld_acquire_cta(elem_pointer(wo.mLast, (slot, block)))


@cute.jit
def _wait_block(
    wo: WriteOrderTensors, block: Int32, head: Int32, writer: Int32
) -> None:
    ptr = elem_pointer(wo.mWriteOrder, (block, head, 0))
    published = ld_acquire(ptr)
    while published != writer:
        published = ld_acquire(ptr)
    cute.arch.fence_proxy("async.global")


@cute.jit
def wait_blocks(
    wo: WriteOrderTensors,
    left: Int32,
    right: Int32,
    head: Int32,
    pred_left: Int32,
    pred_right: Int32,
) -> None:
    """Wait for the predecessor on the left block of a tile and, when the
    tile spans two blocks, on the right block; negative skips its block."""
    if pred_left >= 0:
        _wait_block(wo, left, head, pred_left)
    if right != left and pred_right >= 0:
        _wait_block(wo, right, head, pred_right)


@cute.jit
def _arrive_block(
    wo: WriteOrderTensors,
    block: Int32,
    head: Int32,
    writer: Int32,
    weight: Int32,
    cluster_size: cutlass.Constexpr[int],
) -> None:
    pub_ptr = elem_pointer(wo.mWriteOrder, (block, head, 0))
    if weight == 2 * cluster_size:
        st_global_release_gpu_i32(pub_ptr, writer)
    else:
        count_ptr = elem_pointer(wo.mWriteOrder, (block, head, 1))
        prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
        if prev + weight == 2 * cluster_size:
            wo.mWriteOrder[block, head, 1] = Int32(0)
            st_global_release_gpu_i32(pub_ptr, writer)


@cute.jit
def arrive_blocks(
    wo: WriteOrderTensors,
    left: Int32,
    right: Int32,
    head: Int32,
    writer: Int32,
    left_weight: Int32,
    right_weight: Int32,
    cluster_size: cutlass.Constexpr[int],
) -> None:
    """Arrive on the left block of a tile and, when the tile spans two
    blocks, on the right block; zero weight skips its block."""
    if left_weight > 0:
        _arrive_block(wo, left, head, writer, left_weight, cluster_size)
    if right != left and right_weight > 0:
        _arrive_block(wo, right, head, writer, right_weight, cluster_size)


@cute.jit
def tile_blocks(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    row = range_start + tile_idx * block_size
    row_end = cutlass.min(row + block_size, range_end)
    return row // block_size, (row_end - 1) // block_size


@cute.jit
def block_tiles(range_start: Int32, block: Int32, block_size: cutlass.Constexpr[int]):
    """Range-local tiles ``[first, last]`` that overlap physical ``block``;
    the caller bounds them to the tiles of the range."""
    first_block = range_start // block_size
    last = block - first_block
    first = last
    if range_start % block_size != 0 and block > first_block:
        first = last - 1
    return first, last


@cute.jit
def tile_arrivals(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    row = range_start + tile_idx * block_size
    left, right = tile_blocks(range_start, range_end, tile_idx, block_size)
    unaligned = range_start % block_size != 0
    left_weight = Int32(2)
    if unaligned and tile_idx > 0:
        left_weight = Int32(1)
    right_weight = Int32(2)
    if unaligned and row + block_size < range_end:
        right_weight = Int32(1)
    return left, right, left_weight, right_weight


@cute.jit
def _fold_schedule(
    range_from: Int32,
    range_to: Int32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    """Choose block-parallel (per-warp range, coalesced 32-lane blocks) when
    fewer than ``WARP_SIZE`` ranges are folded, else thread-parallel ranges."""
    num_warps = const_expr(num_threads // cute.arch.WARP_SIZE)
    r = range_from + tidx
    r_step = Int32(num_threads)
    b_offset = Int32(0)
    b_stride = Int32(1)
    if range_to - range_from < cute.arch.WARP_SIZE:
        r = range_from + tidx // cute.arch.WARP_SIZE
        r_step = Int32(num_warps)
        b_offset = tidx % cute.arch.WARP_SIZE
        b_stride = Int32(cute.arch.WARP_SIZE)
    return r, r_step, b_offset, b_stride


# ---------------------------------------------------------------------------
# Forward O / LSE
# ---------------------------------------------------------------------------


@cute.jit
def fold_fwd_last(
    mQRanges: cute.Tensor,
    mLast: cute.Tensor,
    slot: Int32,
    range_from: Int32,
    range_to: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
) -> None:
    """Fold forward writers of ``[range_from, range_to)`` into ``mLast[slot, :]``.

    Caller synchronizes the participating threads after return.
    """
    r, r_step, b_offset, b_stride = _fold_schedule(
        range_from, range_to, tidx, num_threads
    )
    while r < range_to:
        start = mQRanges[r, 0] * rows_per_token
        end = mQRanges[r, 1] * rows_per_token
        if end > start:
            writer = r + 1
            b = start // block_size + b_offset
            last_b = (end - 1) // block_size
            while b <= last_b:
                red_max_relaxed_cta(elem_pointer(mLast, (slot, b)), writer)
                b += b_stride
        r += r_step
    cute.arch.fence_acq_rel_cta()


# ---------------------------------------------------------------------------
# Backward dK / dV
# ---------------------------------------------------------------------------


@cute.jit
def _dkv_k_tile_active(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_tile: Int32,
    attn_type: Optional[Int32],
) -> cutlass.Boolean:
    active = cutlass.Boolean(False)
    if seqlen_q > 0 and k_tile >= 0 and k_tile * block_info.tile_n < seqlen_k:
        m_lo, m_hi = block_info.get_m_block_min_max_seqlens(
            seqlen_q, seqlen_k, k_tile, attn_type
        )
        active = m_lo < m_hi
    return active


@cute.jit
def fold_dkv_last(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mLast: cute.Tensor,
    slot: Int32,
    range_from: Int32,
    range_to: Int32,
    writer_stride: Int32,
) -> None:
    """Fold backward dK/dV writers of ``[range_from, range_to)`` into
    ``mLast[slot, :]`` within one warp (32 lanes processing 32 ranges)."""
    block_rows = block_info.tile_n
    r = range_from + cute.arch.lane_idx()
    while r < range_to:
        k_s = mKRanges[r, 0]
        k_e = mKRanges[r, 1]
        seqlen_k = k_e - k_s
        seqlen_q = mQRanges[r, 1] - mQRanges[r, 0]
        if seqlen_k > 0 and seqlen_q > 0:
            writer = (r + 1) * writer_stride
            kb = k_s // block_rows
            last_kb = (k_e - 1) // block_rows
            attn_r: Optional[Int32] = None
            if const_expr(mMaskTypes is not None):
                assert mMaskTypes is not None
                attn_r = Int32(mMaskTypes[r])
            if const_expr(not block_info.is_local) and (
                const_expr(mMaskTypes is None)
                or attn_r != MT_MAP.bi_causal
                or seqlen_q <= seqlen_k
            ):
                while kb <= last_kb:
                    red_max_relaxed_cta(elem_pointer(mLast, (slot, kb)), writer)
                    kb += 1
            else:
                while kb <= last_kb:
                    first, last = block_tiles(k_s, kb, block_rows)
                    active = _dkv_k_tile_active(
                        block_info, seqlen_q, seqlen_k, last, attn_r
                    )
                    if not active and first != last:
                        active = _dkv_k_tile_active(
                            block_info, seqlen_q, seqlen_k, first, attn_r
                        )
                    if active:
                        red_max_relaxed_cta(elem_pointer(mLast, (slot, kb)), writer)
                    kb += 1
        r += cute.arch.WARP_SIZE
    cute.arch.fence_acq_rel_cta()
    cute.arch.sync_warp()


@cute.jit
def dkv_wait(
    dkv_write_order: WriteOrderTensors,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    seqlen_info: SeqlenInfoQK,
    attn_type: Optional[Int32],
    n_cta: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    dkv_folded: Int32,
    is_fold_warp: cutlass.Boolean,
    is_waiter: cutlass.Boolean,
    covers_all_qheads: cutlass.Constexpr[bool],
    kv_group: cutlass.Constexpr[int],
):
    k_start = seqlen_info.offset_k
    left, right, left_weight, right_weight = tile_arrivals(
        k_start,
        k_start + seqlen_info.seqlen_k,
        n_cta,
        block_info.tile_n,
    )
    if left_weight == 1:
        if not _dkv_k_tile_active(
            block_info,
            seqlen_info.seqlen_q,
            seqlen_info.seqlen_k,
            n_cta - 1,
            attn_type,
        ):
            left_weight = Int32(2)
    if right != left and right_weight == 1:
        if not _dkv_k_tile_active(
            block_info,
            seqlen_info.seqlen_q,
            seqlen_info.seqlen_k,
            n_cta + 1,
            attn_type,
        ):
            right_weight = Int32(2)
    if const_expr(covers_all_qheads):
        head_kv = head_idx
        group_rank = Int32(0)
    else:
        head_kv = head_idx // kv_group
        group_rank = head_idx % kv_group
    writer = writer_number(dkv_write_order, batch_idx, group_rank)
    slot = cute.arch.block_idx()[0]
    if group_rank == 0 and dkv_folded < batch_idx:
        if is_fold_warp:
            fold_dkv_last(
                block_info,
                mQRanges,
                mKRanges,
                mMaskTypes,
                dkv_write_order.mLast,
                slot,
                dkv_folded,
                batch_idx,
                dkv_write_order.writer_stride,
            )
        dkv_folded = batch_idx
    if is_waiter:
        pred_left = writer - 1
        pred_right = writer - 1
        if group_rank == 0:
            pred_left = last_writer(dkv_write_order, slot, left)
            pred_right = pred_left
            if right != left:
                pred_right = last_writer(dkv_write_order, slot, right)
        wait_blocks(dkv_write_order, left, right, head_kv, pred_left, pred_right)
    return left, right, left_weight, right_weight, writer, head_kv, dkv_folded


# ---------------------------------------------------------------------------
# Backward dQ
# ---------------------------------------------------------------------------


@cute.jit
def _dq_prev_k_tile(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    q_start: Int32,
    attn_type: Optional[Int32],
    block: Int32,
    n_limit: Int32,
) -> Int32:
    """Largest K tile ``< n_limit`` for which a pair writes dQ on physical Q
    ``block``; -1 if none or if ``block`` does not overlap the Q range."""
    prev_n = Int32(-1)
    tile_m = block_info.tile_m
    if (
        seqlen_q > 0
        and seqlen_k > 0
        and n_limit > 0
        and q_start // tile_m <= block
        and block <= (q_start + seqlen_q - 1) // tile_m
    ):
        mt, mt_last = block_tiles(q_start, block, tile_m)
        while mt <= mt_last:
            if mt * tile_m < seqlen_q:
                l_m, r_m = block_info.get_n_block_min_max_seqlens(
                    seqlen_q, seqlen_k, mt, attn_type
                )
                if r_m > l_m and l_m < n_limit:
                    prev_n = cutlass.max(prev_n, cutlass.min(n_limit, r_m) - 1)
            mt += 1
    return prev_n


@cute.jit
def fold_dq_last(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mLast: cute.Tensor,
    slot: Int32,
    range_from: Int32,
    range_to: Int32,
    writer_stride: Int32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
) -> None:
    """Fold backward dQ writers of ``[range_from, range_to)`` into ``mLast[slot, :]``.

    Caller synchronizes the participating threads after return.
    """
    tile_m = block_info.tile_m
    r, r_step, b_offset, b_stride = _fold_schedule(
        range_from, range_to, tidx, num_threads
    )
    while r < range_to:
        seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
        if seqlen_k > 0:
            k_tiles = cute.ceil_div(seqlen_k, block_info.tile_n)
            if const_expr(mCuBatches is not None):
                assert mCuBatches is not None
                row = Int32(mCuBatches[r])
                row_end = Int32(mCuBatches[r + 1])
            else:
                row = r
                row_end = r + 1
            while row < row_end:
                q_s = mQRanges[row, 0]
                q_e = mQRanges[row, 1]
                seqlen_q = q_e - q_s
                if seqlen_q > 0:
                    attn_r: Optional[Int32] = None
                    if const_expr(mMaskTypes is not None):
                        assert mMaskTypes is not None
                        attn_r = Int32(mMaskTypes[row])
                    b = q_s // tile_m + b_offset
                    last_b = (q_e - 1) // tile_m
                    while b <= last_b:
                        prev_n = _dq_prev_k_tile(
                            block_info, seqlen_q, seqlen_k, q_s, attn_r, b, k_tiles
                        )
                        if prev_n >= 0:
                            writer = r * writer_stride + prev_n + 1
                            red_max_relaxed_cta(elem_pointer(mLast, (slot, b)), writer)
                        b += b_stride
                row += 1
        r += r_step
    cute.arch.fence_acq_rel_cta()


@cute.jit
def dq_group_has_block_overlap(
    mQRanges: cute.Tensor,
    pair_beg: Int32,
    pair_cnt: Int32,
    tile_m: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    all_inc = cutlass.Boolean(True)
    all_dec = cutlass.Boolean(True)
    seen_nonempty = cutlass.Boolean(False)
    prev_b0 = Int32(0)
    prev_b1 = Int32(0)
    pj = Int32(0)
    while pj < pair_cnt and (all_inc or all_dec):
        row = pair_beg + pj
        qs = mQRanges[row, 0]
        qe = mQRanges[row, 1]
        if qe > qs:
            b0 = qs // tile_m
            b1 = (qe - 1) // tile_m
            if seen_nonempty:
                if b0 <= prev_b1:
                    all_inc = cutlass.Boolean(False)
                if b1 >= prev_b0:
                    all_dec = cutlass.Boolean(False)
            else:
                seen_nonempty = cutlass.Boolean(True)
            prev_b0 = b0
            prev_b1 = b1
        pj += 1
    return not (all_inc or all_dec)


@cute.jit
def dq_plan(
    dq_write_order: WriteOrderTensors,
    block_info: BlockInfo,
    batch_idx: Int32,
    n_block_cta_group: Int32,
    seqlen_info: SeqlenInfoQK,
    attn_type: Optional[Int32],
    m_block: Int32,
    m_block_active_start: Int32,
    m_block_active_end: Int32,
    mQRanges: cute.Tensor,
    is_waiter: cutlass.Boolean,
    mMaskTypes: Optional[cute.Tensor] = None,
    group_pairs: Optional[tuple[Int32, Int32, Int32]] = None,
    seqlen_k_group: Int32 = Int32(0),
    group_has_block_overlap: cutlass.Boolean = cutlass.Boolean(False),
):
    """Plan the waits and arrivals of one M tile on its left/right physical dQ blocks."""
    left = Int32(0)
    right = Int32(0)
    wait_pred_left = Int32(-1)
    wait_pred_right = Int32(-1)
    arrive_weight_left = Int32(0)
    arrive_weight_right = Int32(0)
    if is_waiter:
        tile_m = block_info.tile_m
        q_start = seqlen_info.offset_q
        left, right = tile_blocks(
            q_start, q_start + seqlen_info.seqlen_q, m_block, tile_m
        )
        wait_left = cutlass.Boolean(True)
        if m_block > m_block_active_start and (q_start % tile_m) != 0:
            wait_left = cutlass.Boolean(False)
        wait_right = right != left

        arrive_weight_left = Int32(2)
        if right != left and m_block == m_block_active_end - 1:
            arrive_weight_right = Int32(2)

        need_prev_n = cutlass.Boolean(False)
        if n_block_cta_group > 0:
            n_min, _ = block_info.get_n_block_min_max_seqlens(
                seqlen_info.seqlen_q, seqlen_info.seqlen_k, m_block, attn_type
            )
            need_prev_n = n_min == n_block_cta_group

        prev_n_left = n_block_cta_group - 1
        prev_n_right = n_block_cta_group - 1
        if n_block_cta_group == 0:
            prev_n_left = Int32(-1)
            prev_n_right = Int32(-1)
        elif need_prev_n:
            prev_n_left = Int32(-1)
            prev_n_right = Int32(-1)
            if not (const_expr(group_pairs is not None) and group_has_block_overlap):
                if wait_left and (q_start % tile_m) != 0 and m_block > 0:
                    prev_n_left = _dq_prev_k_tile(
                        block_info,
                        seqlen_info.seqlen_q,
                        seqlen_info.seqlen_k,
                        q_start,
                        attn_type,
                        left,
                        n_block_cta_group,
                    )

        if const_expr(group_pairs is not None):
            if group_has_block_overlap:
                assert group_pairs is not None
                assert mMaskTypes is not None
                pair_beg, pair_cur, pair_end = group_pairs
                row = pair_beg
                while row < pair_end:
                    q_s = mQRanges[row, 0]
                    q_e = mQRanges[row, 1]
                    seqlen_q = q_e - q_s
                    if (
                        seqlen_q > 0
                        and q_s // tile_m <= right
                        and (q_e - 1) // tile_m >= left
                    ):
                        attn_r = Int32(mMaskTypes[row])
                        if row != pair_cur:
                            act_s = q_s
                            act_e = q_e
                            if attn_r != MT_MAP.full:
                                lo_r, hi_r = block_info.get_m_block_min_max_seqlens(
                                    seqlen_q, seqlen_k_group, n_block_cta_group, attn_r
                                )
                                act_s = q_s + lo_r * tile_m
                                act_e = cutlass.min(q_s + hi_r * tile_m, q_e)
                            if act_e > act_s:
                                ab0 = act_s // tile_m
                                ab1 = (act_e - 1) // tile_m
                                if ab0 <= left and left <= ab1:
                                    if row < pair_cur:
                                        wait_left = cutlass.Boolean(False)
                                    else:
                                        arrive_weight_left = Int32(0)
                                if right != left and ab0 <= right and right <= ab1:
                                    if row < pair_cur:
                                        wait_right = cutlass.Boolean(False)
                                    else:
                                        arrive_weight_right = Int32(0)
                        if need_prev_n:
                            prev_n_left = cutlass.max(
                                prev_n_left,
                                _dq_prev_k_tile(
                                    block_info,
                                    seqlen_q,
                                    seqlen_k_group,
                                    q_s,
                                    attn_r,
                                    left,
                                    n_block_cta_group,
                                ),
                            )
                            if right != left:
                                prev_n_right = cutlass.max(
                                    prev_n_right,
                                    _dq_prev_k_tile(
                                        block_info,
                                        seqlen_q,
                                        seqlen_k_group,
                                        q_s,
                                        attn_r,
                                        right,
                                        n_block_cta_group,
                                    ),
                                )
                    row += 1

        slot = cute.arch.block_idx()[0]
        if wait_left:
            if prev_n_left >= 0:
                wait_pred_left = writer_number(dq_write_order, batch_idx, prev_n_left)
            else:
                wait_pred_left = last_writer(dq_write_order, slot, left)
        if wait_right:
            if prev_n_right >= 0:
                wait_pred_right = writer_number(dq_write_order, batch_idx, prev_n_right)
            else:
                wait_pred_right = last_writer(dq_write_order, slot, right)

    return (
        left,
        right,
        wait_pred_left,
        wait_pred_right,
        arrive_weight_left,
        arrive_weight_right,
    )
