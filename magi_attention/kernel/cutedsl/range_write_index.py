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

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

# isort: split
from .block_info import BlockInfo
from .cutedsl_utils import elem_pointer, red_max_relaxed_cta
from .ffa_utils import MT_MAP


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
def pair_n_bounds(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    m_tile: Int32,
    attn_type: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
):
    if const_expr(use_per_range_mask):
        return block_info.get_n_block_min_max_per_range_seqlens(
            seqlen_q, seqlen_k, m_tile, attn_type
        )
    return block_info.get_n_block_min_max_seqlens(seqlen_q, seqlen_k, m_tile)


@cute.jit
def pair_m_bounds(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_tile: Int32,
    attn_type: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
):
    if const_expr(use_per_range_mask):
        return block_info.get_m_block_min_max_per_range_seqlens(
            seqlen_q, seqlen_k, k_tile, attn_type
        )
    return block_info.get_m_block_min_max_seqlens(seqlen_q, seqlen_k, k_tile)


@cute.jit
def dq_pair_prev_k_tile_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    q_start: Int32,
    attn_type: Int32,
    block: Int32,
    n_limit: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    """Largest K tile ``< n_limit`` for which a pair writes dQ on physical Q
    ``block``; -1 if none. ``block`` must overlap the Q range of the pair."""
    prev_n = Int32(-1)
    if seqlen_q > 0 and seqlen_k > 0 and n_limit > 0:
        if const_expr(
            not use_per_range_mask
            and not block_info.is_causal
            and not block_info.is_local
        ):
            prev_n = (
                cutlass.min(n_limit, cute.ceil_div(seqlen_k, block_info.tile_n)) - 1
            )
        elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
            prev_n = (
                cutlass.min(n_limit, cute.ceil_div(seqlen_k, block_info.tile_n)) - 1
            )
        else:
            mt, mt_last = block_tiles(q_start, block, block_info.tile_m)
            while mt <= mt_last:
                if mt * block_info.tile_m < seqlen_q:
                    l_m, r_m = pair_n_bounds(
                        block_info,
                        seqlen_q,
                        seqlen_k,
                        mt,
                        attn_type,
                        use_per_range_mask,
                    )
                    if r_m > l_m and l_m < n_limit:
                        prev_n = cutlass.max(prev_n, cutlass.min(n_limit, r_m) - 1)
                mt += 1
    return prev_n


@cute.jit
def dkv_pair_k_tile_is_active(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_tile: Int32,
    attn_type: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    """Whether a pair has an unmasked Q tile on its K tile ``k_tile``; a tile
    outside the K range is inactive."""
    active = cutlass.Boolean(False)
    if seqlen_q > 0 and k_tile >= 0 and k_tile * block_info.tile_n < seqlen_k:
        if const_expr(
            not use_per_range_mask
            and not block_info.is_causal
            and not block_info.is_local
        ):
            active = cutlass.Boolean(True)
        elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
            active = cutlass.Boolean(True)
        else:
            m_lo, m_hi = pair_m_bounds(
                block_info, seqlen_q, seqlen_k, k_tile, attn_type, use_per_range_mask
            )
            if m_lo < m_hi:
                active = cutlass.Boolean(True)
    return active


@cute.jit
def _dkv_block_is_active(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_start: Int32,
    attn_type: Int32,
    kb: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    first, last = block_tiles(k_start, kb, block_info.tile_n)
    active = dkv_pair_k_tile_is_active(
        block_info, seqlen_q, seqlen_k, last, attn_type, use_per_range_mask
    )
    if not active and first != last:
        active = dkv_pair_k_tile_is_active(
            block_info, seqlen_q, seqlen_k, first, attn_type, use_per_range_mask
        )
    return active


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
    tile_m: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
) -> None:
    """Fold backward dQ writers of ``[range_from, range_to)`` into ``mLast[slot, :]``.

    Caller synchronizes the participating threads after return.
    """
    r, r_step, b_offset, b_stride = _fold_schedule(
        range_from, range_to, tidx, num_threads
    )
    while r < range_to:
        seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
        if seqlen_k > 0:
            k_tiles = cute.ceil_div(seqlen_k, block_info.tile_n)
            if const_expr(range_merge):
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
                    attn_r = Int32(MT_MAP.full)
                    if const_expr(use_per_range_mask):
                        assert mMaskTypes is not None
                        attn_r = Int32(mMaskTypes[row])
                    b = q_s // tile_m + b_offset
                    last_b = (q_e - 1) // tile_m
                    while b <= last_b:
                        prev_n = dq_pair_prev_k_tile_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            q_s,
                            attn_r,
                            b,
                            k_tiles,
                            use_per_range_mask,
                        )
                        if prev_n >= 0:
                            writer = r * writer_stride + prev_n + 1
                            red_max_relaxed_cta(elem_pointer(mLast, (slot, b)), writer)
                        b += b_stride
                row += 1
        r += r_step
    cute.arch.fence_acq_rel_cta()


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
    block_rows: cutlass.Constexpr[int],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> None:
    """Fold backward dK/dV writers of ``[range_from, range_to)`` into
    ``mLast[slot, :]`` within one warp (32 lanes processing 32 ranges)."""
    lane = cute.arch.lane_idx()
    r = range_from + lane
    while r < range_to:
        k_s = mKRanges[r, 0]
        k_e = mKRanges[r, 1]
        seqlen_k = k_e - k_s
        seqlen_q = mQRanges[r, 1] - mQRanges[r, 0]
        if seqlen_k > 0 and seqlen_q > 0:
            writer = (r + 1) * writer_stride
            kb = k_s // block_rows
            last_kb = (k_e - 1) // block_rows
            attn_r = Int32(MT_MAP.full)
            if const_expr(use_per_range_mask):
                assert mMaskTypes is not None
                attn_r = Int32(mMaskTypes[r])
            if const_expr(
                not use_per_range_mask
                and not block_info.is_causal
                and not block_info.is_local
            ) or (
                const_expr(use_per_range_mask)
                and (attn_r != MT_MAP.bi_causal or seqlen_q <= seqlen_k)
            ):
                while kb <= last_kb:
                    red_max_relaxed_cta(elem_pointer(mLast, (slot, kb)), writer)
                    kb += 1
            else:
                while kb <= last_kb:
                    if _dkv_block_is_active(
                        block_info,
                        seqlen_q,
                        seqlen_k,
                        k_s,
                        attn_r,
                        kb,
                        use_per_range_mask,
                    ):
                        red_max_relaxed_cta(elem_pointer(mLast, (slot, kb)), writer)
                    kb += 1
        r += cute.arch.WARP_SIZE
    cute.arch.fence_acq_rel_cta()
    cute.arch.sync_warp()
