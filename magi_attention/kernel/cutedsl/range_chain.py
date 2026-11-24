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
    mbarrier_arrive_peer_release_cluster,
    mbarrier_wait_acquire_cluster,
    st_global_release_gpu_i32,
)
from .ffa_utils import MT_MAP
from .seqlen_info import SeqlenInfoQK


@dataclass
class ChainTensors(ParamsBase):
    """Publish chain of one output and the checkpoints of its writers.

    ``mChain[b, h] = [published, count]``. Writer ``j`` of range ``r`` is
    ``r * writer_stride + j + 1`` (0: none), so ``writer_stride`` is the count
    of writer numbers per range. ``mConflictCkpt[c, b]`` is the highest writer
    on block ``b`` among ranges ``[0, c * ckpt_stride)`` (0: none), so
    ``ckpt_stride`` is the count of ranges per checkpoint row; the kernel only
    reads it.
    """

    mChain: cute.Tensor
    mConflictCkpt: cute.Tensor
    writer_stride: Int32
    ckpt_stride: Int32


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
def tile_events(
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
def dq_pair_n_bounds(
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
def dq_pair_k_tile_end_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    q_start: Int32,
    attn_type: Int32,
    block: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    """1 + the largest K tile for which a pair writes dQ on physical Q
    ``block``; 0 if none. ``block`` must overlap the Q range of the pair."""
    k_tile_end = Int32(0)
    if seqlen_q > 0 and seqlen_k > 0:
        if const_expr(
            not use_per_range_mask
            and not block_info.is_causal
            and not block_info.is_local
        ):
            k_tile_end = cute.ceil_div(seqlen_k, block_info.tile_n)
        elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
            k_tile_end = cute.ceil_div(seqlen_k, block_info.tile_n)
        else:
            mt, mt_last = block_tiles(q_start, block, block_info.tile_m)
            while mt <= mt_last:
                if mt * block_info.tile_m < seqlen_q:
                    n_lo, n_hi = dq_pair_n_bounds(
                        block_info,
                        seqlen_q,
                        seqlen_k,
                        mt,
                        attn_type,
                        use_per_range_mask,
                    )
                    if n_hi > n_lo:
                        k_tile_end = cutlass.max(k_tile_end, n_hi)
                mt += 1
    return k_tile_end


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
    if const_expr(
        not use_per_range_mask and not block_info.is_causal and not block_info.is_local
    ):
        prev_n = n_limit - 1
    elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
        prev_n = n_limit - 1
    else:
        mt, mt_last = block_tiles(q_start, block, block_info.tile_m)
        while mt <= mt_last:
            if mt * block_info.tile_m < seqlen_q:
                l_m, r_m = dq_pair_n_bounds(
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
        if const_expr(use_per_range_mask):
            if attn_type == MT_MAP.full:
                active = cutlass.Boolean(True)
            else:
                m_lo, m_hi = block_info.get_m_block_min_max_per_range_seqlens(
                    seqlen_q, seqlen_k, k_tile, attn_type
                )
                if m_lo < m_hi:
                    active = cutlass.Boolean(True)
        elif const_expr(not block_info.is_causal and not block_info.is_local):
            active = cutlass.Boolean(True)
        else:
            m_lo, m_hi = block_info.get_m_block_min_max_seqlens(
                seqlen_q, seqlen_k, k_tile
            )
            if m_lo < m_hi:
                active = cutlass.Boolean(True)
    return active


@cute.jit
def dkv_pair_is_active_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_start: Int32,
    attn_type: Int32,
    block: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    """Whether a pair has an active K tile that overlaps physical K ``block``."""
    first, last = block_tiles(k_start, block, block_info.tile_n)
    active = dkv_pair_k_tile_is_active(
        block_info, seqlen_q, seqlen_k, last, attn_type, use_per_range_mask
    )
    if not active and first != last:
        active = dkv_pair_k_tile_is_active(
            block_info, seqlen_q, seqlen_k, first, attn_type, use_per_range_mask
        )
    return active


@cute.jit
def lookup_fwd_conflict(
    mConflictCkpt: cute.Tensor,
    mQRanges: cute.Tensor,
    batch_idx: Int32,
    block: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    ckpt_stride: Int32,
) -> Int32:
    """Highest earlier fwd writer (r + 1) touching physical ``block``; 0 if none."""
    c = batch_idx // ckpt_stride
    pred = Int32(mConflictCkpt[c, block])
    if ckpt_stride > 1:
        r_lo = c * ckpt_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            s = mQRanges[r, 0] * rows_per_token
            e = mQRanges[r, 1] * rows_per_token
            if e > s and s // block_size <= block and (e - 1) // block_size >= block:
                pred = r + 1
                found = cutlass.Boolean(True)
            r -= 1
    return pred


@cute.jit
def wait_published(
    mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32
) -> None:
    """The fence orders the generic acquire before cross-CTA async-proxy ops."""
    ptr = elem_pointer(mChain, (block, head, 0))
    published = ld_acquire(ptr)
    while published != writer:
        published = ld_acquire(ptr)
    cute.arch.fence_proxy("async.global")


@cute.jit
def arrive(
    mChain: cute.Tensor,
    block: Int32,
    head: Int32,
    writer: Int32,
    weight: Int32,
    cluster_size: cutlass.Constexpr[int] = 1,
) -> None:
    """Arrive after CTA writes complete; count resets before release publish while successor waits."""
    if weight == 2 * cluster_size:
        publish(mChain, block, head, writer)
    else:
        count_ptr = elem_pointer(mChain, (block, head, 1))
        prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
        if prev + weight == 2 * cluster_size:
            mChain[block, head, 1] = Int32(0)
            publish(mChain, block, head, writer)


@cute.jit
def publish(mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32) -> None:
    st_global_release_gpu_i32(elem_pointer(mChain, (block, head, 0)), writer)


@cute.jit
def pair_handoff_arrive(mbar: cute.Pointer, handoff: Int32, peer: Int32) -> None:
    """Arrive after bulk reductions complete; alternating mbarriers bound the phase lead to one."""
    mbarrier_arrive_peer_release_cluster(mbar + (handoff & 1), peer)


@cute.jit
def pair_handoff_wait(mbar: cute.Pointer, handoff: Int32) -> None:
    """Both CTAs must hand off at the same pairs."""
    mbarrier_wait_acquire_cluster(mbar + (handoff & 1), (handoff >> 1) & 1)
    cute.arch.fence_proxy("async.global")


@cute.jit
def dkv_k_tile_is_active(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    r: Int32,
    n_cta: Int32,
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
    active = cutlass.Boolean(False)
    if const_expr(range_merge):
        assert (
            mMaskTypes is not None
            and mCuBatches is not None
            and mGroupAllKActive is not None
        )
        if n_cta >= 0 and n_cta * block_info.tile_n < seqlen_k:
            if mGroupAllKActive[r] != 0:
                active = cutlass.Boolean(True)
            else:
                row = Int32(mCuBatches[r])
                row_end = Int32(mCuBatches[r + 1])
                while row < row_end and not active:
                    active = dkv_pair_k_tile_is_active(
                        block_info,
                        mQRanges[row, 1] - mQRanges[row, 0],
                        seqlen_k,
                        n_cta,
                        Int32(mMaskTypes[row]),
                        True,
                    )
                    row += 1
    else:
        attn_r = Int32(MT_MAP.full)
        if const_expr(use_per_range_mask):
            assert mMaskTypes is not None
            attn_r = Int32(mMaskTypes[r])
        active = dkv_pair_k_tile_is_active(
            block_info,
            mQRanges[r, 1] - mQRanges[r, 0],
            seqlen_k,
            n_cta,
            attn_r,
            use_per_range_mask,
        )
    return active


@cute.jit
def _dkv_block_writer(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    writer_stride: Int32,
    r: Int32,
    kb: Int32,
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    first, last = block_tiles(mKRanges[r, 0], kb, block_info.tile_n)
    active = dkv_k_tile_is_active(
        block_info,
        mQRanges,
        mKRanges,
        mMaskTypes,
        mCuBatches,
        mGroupAllKActive,
        r,
        last,
        range_merge,
        use_per_range_mask,
    )
    if not active and first != last:
        active = dkv_k_tile_is_active(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mGroupAllKActive,
            r,
            first,
            range_merge,
            use_per_range_mask,
        )
    writer = Int32(0)
    if active:
        writer = (r + 1) * writer_stride
    return writer


@cute.jit
def lookup_dkv_conflict(
    dkv_chain: ChainTensors,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    batch_idx: Int32,
    kb: Int32,
    block_rows: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    c = batch_idx // dkv_chain.ckpt_stride
    pred = Int32(dkv_chain.mConflictCkpt[c, kb])
    if dkv_chain.ckpt_stride > 1:
        r_lo = c * dkv_chain.ckpt_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            k_s = mKRanges[r, 0]
            k_e = mKRanges[r, 1]
            if k_e > k_s and k_s // block_rows <= kb and (k_e - 1) // block_rows >= kb:
                w = _dkv_block_writer(
                    block_info,
                    mQRanges,
                    mKRanges,
                    mMaskTypes,
                    mCuBatches,
                    mGroupAllKActive,
                    dkv_chain.writer_stride,
                    r,
                    kb,
                    range_merge,
                    use_per_range_mask,
                )
                if w > 0:
                    pred = w
                    found = cutlass.Boolean(True)
            r -= 1
    return pred


@cute.jit
def _dkv_neighbor_active(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    seqlen_info: SeqlenInfoQK,
    attn_type: Int32,
    batch_idx: Int32,
    n_cta_neighbor: Int32,
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    if const_expr(range_merge):
        return dkv_k_tile_is_active(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mGroupAllKActive,
            batch_idx,
            n_cta_neighbor,
            True,
            use_per_range_mask,
        )
    if const_expr(use_per_range_mask):
        m_lo, m_hi = block_info.get_m_block_min_max_per_range(
            seqlen_info, n_cta_neighbor, attn_type
        )
        return m_lo < m_hi
    m_lo, m_hi = block_info.get_m_block_min_max(seqlen_info, n_cta_neighbor)
    return m_lo < m_hi


@cute.jit
def dkv_chain_wait(
    dkv_chain: ChainTensors,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    seqlen_info: SeqlenInfoQK,
    attn_type: Int32,
    n_cta: Int32,
    head_idx: Int32,
    batch_idx: Int32,
    is_waiter: cutlass.Boolean,
    block_rows: cutlass.Constexpr[int],
    covers_all_qheads: cutlass.Constexpr[bool],
    kv_group: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
):
    k_start = seqlen_info.offset_k
    left, right, left_weight, right_weight = tile_events(
        k_start,
        k_start + seqlen_info.seqlen_k,
        n_cta,
        block_rows,
    )
    if left_weight == 1:
        if not _dkv_neighbor_active(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mGroupAllKActive,
            seqlen_info,
            attn_type,
            batch_idx,
            n_cta - 1,
            range_merge,
            use_per_range_mask,
        ):
            left_weight = Int32(2)
    if right != left and right_weight == 1:
        if not _dkv_neighbor_active(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mGroupAllKActive,
            seqlen_info,
            attn_type,
            batch_idx,
            n_cta + 1,
            range_merge,
            use_per_range_mask,
        ):
            right_weight = Int32(2)
    if const_expr(covers_all_qheads):
        head_kv = head_idx
        group_rank = Int32(0)
    else:
        head_kv = head_idx // kv_group
        group_rank = head_idx % kv_group
    writer = batch_idx * dkv_chain.writer_stride + group_rank + 1
    if is_waiter:
        pred_left = writer - 1
        pred_right = writer - 1
        if group_rank == 0:
            pred_left = lookup_dkv_conflict(
                dkv_chain,
                block_info,
                mQRanges,
                mKRanges,
                mMaskTypes,
                mCuBatches,
                mGroupAllKActive,
                batch_idx,
                left,
                block_rows,
                range_merge,
                use_per_range_mask,
            )
            if right != left:
                pred_right = lookup_dkv_conflict(
                    dkv_chain,
                    block_info,
                    mQRanges,
                    mKRanges,
                    mMaskTypes,
                    mCuBatches,
                    mGroupAllKActive,
                    batch_idx,
                    right,
                    block_rows,
                    range_merge,
                    use_per_range_mask,
                )
        wait_published(dkv_chain.mChain, left, head_kv, pred_left)
        if right != left:
            wait_published(dkv_chain.mChain, right, head_kv, pred_right)
    return left, right, left_weight, right_weight, writer, head_kv


@cute.jit
def lookup_dq_conflict(
    dq_chain: ChainTensors,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    batch_idx: Int32,
    block: Int32,
    tile_m: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    c = batch_idx // dq_chain.ckpt_stride
    pred = Int32(dq_chain.mConflictCkpt[c, block])
    if dq_chain.ckpt_stride > 1:
        r_lo = c * dq_chain.ckpt_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
            if seqlen_k > 0:
                k_tile_end = Int32(0)
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
                    if (
                        q_e > q_s
                        and q_s // tile_m <= block
                        and (q_e - 1) // tile_m >= block
                    ):
                        attn_r = Int32(MT_MAP.full)
                        if const_expr(use_per_range_mask):
                            assert mMaskTypes is not None
                            attn_r = Int32(mMaskTypes[row])
                        k_tile_end = cutlass.max(
                            k_tile_end,
                            dq_pair_k_tile_end_on_block(
                                block_info,
                                q_e - q_s,
                                seqlen_k,
                                q_s,
                                attn_r,
                                block,
                                use_per_range_mask,
                            ),
                        )
                    row += 1
                if k_tile_end > 0:
                    pred = r * dq_chain.writer_stride + k_tile_end
                    found = cutlass.Boolean(True)
            r -= 1
    return pred


@cute.jit
def _dq_merge_overlap_scan(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    pair_beg: Int32,
    pair_cur: Int32,
    pair_end: Int32,
    seqlen_k_group: Int32,
    n_block_cta_group: Int32,
    left: Int32,
    right: Int32,
    need_prev_n: cutlass.Boolean,
    tile_m: cutlass.Constexpr[int],
    use_per_range_mask: cutlass.Constexpr[bool],
):
    earlier_left = cutlass.Boolean(False)
    earlier_right = cutlass.Boolean(False)
    later_left = cutlass.Boolean(False)
    later_right = cutlass.Boolean(False)
    prev_n_left = Int32(-1)
    prev_n_right = Int32(-1)
    row = pair_beg
    while row < pair_end:
        q_s = mQRanges[row, 0]
        q_e = mQRanges[row, 1]
        seqlen_q = q_e - q_s
        if seqlen_q > 0:
            b0 = q_s // tile_m
            b1 = (q_e - 1) // tile_m
            if b0 <= right and b1 >= left:
                attn_r = Int32(MT_MAP.full)
                if const_expr(use_per_range_mask):
                    assert mMaskTypes is not None
                    attn_r = Int32(mMaskTypes[row])
                if row != pair_cur:
                    act_s = q_s
                    act_e = q_e
                    if const_expr(use_per_range_mask):
                        if attn_r != MT_MAP.full:
                            (
                                lo_r,
                                hi_r,
                            ) = block_info.get_m_block_min_max_per_range_seqlens(
                                seqlen_q,
                                seqlen_k_group,
                                n_block_cta_group,
                                attn_r,
                            )
                            act_s = q_s + lo_r * tile_m
                            act_e = cutlass.min(q_s + hi_r * tile_m, q_e)
                    elif const_expr(block_info.is_causal or block_info.is_local):
                        lo_r, hi_r = block_info.get_m_block_min_max_seqlens(
                            seqlen_q, seqlen_k_group, n_block_cta_group
                        )
                        act_s = q_s + lo_r * tile_m
                        act_e = cutlass.min(q_s + hi_r * tile_m, q_e)
                    if act_e > act_s:
                        ab0 = act_s // tile_m
                        ab1 = (act_e - 1) // tile_m
                        if ab0 <= left and left <= ab1:
                            if row < pair_cur:
                                earlier_left = cutlass.Boolean(True)
                            else:
                                later_left = cutlass.Boolean(True)
                        if right != left and ab0 <= right and right <= ab1:
                            if row < pair_cur:
                                earlier_right = cutlass.Boolean(True)
                            else:
                                later_right = cutlass.Boolean(True)
                if need_prev_n:
                    blk = cutlass.max(left, b0)
                    blk_hi = cutlass.min(right, b1)
                    while blk <= blk_hi:
                        cur_prev = dq_pair_prev_k_tile_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k_group,
                            q_s,
                            attn_r,
                            blk,
                            n_block_cta_group,
                            use_per_range_mask,
                        )
                        if blk == left:
                            prev_n_left = cutlass.max(prev_n_left, cur_prev)
                        if blk == right:
                            prev_n_right = cutlass.max(prev_n_right, cur_prev)
                        blk += 1
        row += 1
    return (
        earlier_left,
        earlier_right,
        later_left,
        later_right,
        prev_n_left,
        prev_n_right,
    )


@cute.jit
def dq_chain_plan(
    dq_chain: ChainTensors,
    block_info: BlockInfo,
    batch_idx: Int32,
    n_block_cta_group: Int32,
    seqlen_info: SeqlenInfoQK,
    attn_type: Int32,
    m_block: Int32,
    m_block_active_start: Int32,
    m_block_active_end: Int32,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    tile_m: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
    mMaskTypes: Optional[cute.Tensor] = None,
    mCuBatches: Optional[cute.Tensor] = None,
    group_pairs: Optional[tuple[Int32, Int32, Int32]] = None,
    seqlen_k_group: Int32 = Int32(0),
    group_has_block_overlap: cutlass.Boolean = cutlass.Boolean(False),
):
    q_start = seqlen_info.offset_q
    left, right = tile_blocks(q_start, q_start + seqlen_info.seqlen_q, m_block, tile_m)
    wait_left = cutlass.Boolean(True)
    if m_block > m_block_active_start and (q_start % tile_m) != 0:
        wait_left = cutlass.Boolean(False)
    wait_right = right != left

    arrive_left = cutlass.Boolean(True)
    arrive_right = cutlass.Boolean(False)
    if right != left and m_block == m_block_active_end - 1:
        arrive_right = cutlass.Boolean(True)

    n_min_m = Int32(0)
    if n_block_cta_group > 0:
        n_min_m, _ = dq_pair_n_bounds(
            block_info,
            seqlen_info.seqlen_q,
            seqlen_info.seqlen_k,
            m_block,
            attn_type,
            use_per_range_mask,
        )
    need_prev_n = n_block_cta_group > 0 and n_min_m == n_block_cta_group

    prev_n_left = n_block_cta_group - 1
    prev_n_right = n_block_cta_group - 1
    if n_block_cta_group == 0:
        prev_n_left = Int32(-1)
        prev_n_right = Int32(-1)
    elif need_prev_n:
        prev_n_left = Int32(-1)
        prev_n_right = Int32(-1)
        if not (const_expr(range_merge) and group_has_block_overlap):
            if wait_left and (q_start % tile_m) != 0 and m_block > 0:
                l_prev, r_prev = dq_pair_n_bounds(
                    block_info,
                    seqlen_info.seqlen_q,
                    seqlen_info.seqlen_k,
                    m_block - 1,
                    attn_type,
                    use_per_range_mask,
                )
                if r_prev > l_prev and l_prev < n_block_cta_group:
                    prev_n_left = cutlass.min(n_block_cta_group, r_prev) - 1

    if const_expr(range_merge):
        if group_has_block_overlap:
            assert group_pairs is not None
            (
                earlier_left,
                earlier_right,
                later_left,
                later_right,
                ov_prev_left,
                ov_prev_right,
            ) = _dq_merge_overlap_scan(
                block_info,
                mQRanges,
                mMaskTypes,
                group_pairs[0],
                group_pairs[1],
                group_pairs[2],
                seqlen_k_group,
                n_block_cta_group,
                left,
                right,
                need_prev_n,
                tile_m,
                use_per_range_mask,
            )
            if earlier_left:
                wait_left = cutlass.Boolean(False)
            if earlier_right:
                wait_right = cutlass.Boolean(False)
            if later_left:
                arrive_left = cutlass.Boolean(False)
            if later_right:
                arrive_right = cutlass.Boolean(False)
            if need_prev_n:
                prev_n_left = ov_prev_left
                prev_n_right = ov_prev_right

    wait_pred_left = Int32(-1)
    if wait_left:
        if prev_n_left >= 0:
            wait_pred_left = batch_idx * dq_chain.writer_stride + prev_n_left + 1
        else:
            wait_pred_left = lookup_dq_conflict(
                dq_chain,
                block_info,
                mQRanges,
                mKRanges,
                mMaskTypes,
                mCuBatches,
                batch_idx,
                left,
                tile_m,
                range_merge,
                use_per_range_mask,
            )

    wait_pred_right = Int32(-1)
    if wait_right:
        if prev_n_right >= 0:
            wait_pred_right = batch_idx * dq_chain.writer_stride + prev_n_right + 1
        else:
            wait_pred_right = lookup_dq_conflict(
                dq_chain,
                block_info,
                mQRanges,
                mKRanges,
                mMaskTypes,
                mCuBatches,
                batch_idx,
                right,
                tile_m,
                range_merge,
                use_per_range_mask,
            )

    return (
        left,
        right,
        wait_pred_left,
        wait_pred_right,
        arrive_left,
        arrive_right,
    )


@cute.jit
def dq_chain_wait_planned(
    dq_chain: ChainTensors,
    left: Int32,
    right: Int32,
    wait_pred_left: Int32,
    wait_pred_right: Int32,
    q_head: Int32,
) -> None:
    if wait_pred_left >= 0:
        wait_published(dq_chain.mChain, left, q_head, wait_pred_left)
    if wait_pred_right >= 0:
        wait_published(dq_chain.mChain, right, q_head, wait_pred_right)


@cute.jit
def dq_chain_arrive_planned(
    dq_chain: ChainTensors,
    dq_writer: Int32,
    left: Int32,
    right: Int32,
    arrive_left: cutlass.Boolean,
    arrive_right: cutlass.Boolean,
    q_head: Int32,
    cluster_size: cutlass.Constexpr[int],
) -> None:
    if arrive_left or arrive_right:
        cute.arch.fence_acq_rel_gpu()
        if arrive_left:
            arrive(
                dq_chain.mChain,
                left,
                q_head,
                dq_writer,
                Int32(2),
                cluster_size,
            )
        if arrive_right:
            arrive(
                dq_chain.mChain,
                right,
                q_head,
                dq_writer,
                Int32(2),
                cluster_size,
            )
