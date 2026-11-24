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

"""Order writes to shared output and gradient blocks in deterministic range attention."""

from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

# isort: split
from quack.cute_dsl_utils import ParamsBase

from . import range_write_index
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
class WriteOrderTensors(ParamsBase):
    """Write order of one output and the write index of its writers.

    ``mWriteOrder[b, h] = [published, count]``. Writer ``j`` of range ``r`` is
    ``r * writer_stride + j + 1`` (0: none), so ``writer_stride`` is the count
    of writer numbers per range. ``mWriteIndex[c, b]`` is the highest writer on
    block ``b`` among ranges ``[0, c * index_stride)`` (0: none), so
    ``index_stride`` is the count of ranges per index row; the kernel only
    reads it.
    """

    mWriteOrder: cute.Tensor
    mWriteIndex: cute.Tensor
    writer_stride: Int32
    index_stride: Int32


@cute.jit
def tile_arrivals(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    row = range_start + tile_idx * block_size
    left, right = range_write_index.tile_blocks(
        range_start, range_end, tile_idx, block_size
    )
    unaligned = range_start % block_size != 0
    left_weight = Int32(2)
    if unaligned and tile_idx > 0:
        left_weight = Int32(1)
    right_weight = Int32(2)
    if unaligned and row + block_size < range_end:
        right_weight = Int32(1)
    return left, right, left_weight, right_weight


@cute.jit
def wait_published(
    mWriteOrder: cute.Tensor, block: Int32, head: Int32, writer: Int32
) -> None:
    """The fence orders the generic acquire before cross-CTA async-proxy ops."""
    ptr = elem_pointer(mWriteOrder, (block, head, 0))
    published = ld_acquire(ptr)
    while published != writer:
        published = ld_acquire(ptr)
    cute.arch.fence_proxy("async.global")


@cute.jit
def arrive(
    mWriteOrder: cute.Tensor,
    block: Int32,
    head: Int32,
    writer: Int32,
    weight: Int32,
    cluster_size: cutlass.Constexpr[int] = 1,
) -> None:
    """Arrive after CTA writes complete; count resets before release publish while successor waits."""
    if weight == 2 * cluster_size:
        publish(mWriteOrder, block, head, writer)
    else:
        count_ptr = elem_pointer(mWriteOrder, (block, head, 1))
        prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
        if prev + weight == 2 * cluster_size:
            mWriteOrder[block, head, 1] = Int32(0)
            publish(mWriteOrder, block, head, writer)


@cute.jit
def publish(mWriteOrder: cute.Tensor, block: Int32, head: Int32, writer: Int32) -> None:
    st_global_release_gpu_i32(elem_pointer(mWriteOrder, (block, head, 0)), writer)


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
        return range_write_index.dkv_k_tile_is_active(
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
def dkv_wait(
    dkv_write_order: WriteOrderTensors,
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
    left, right, left_weight, right_weight = tile_arrivals(
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
    writer = batch_idx * dkv_write_order.writer_stride + group_rank + 1
    if is_waiter:
        pred_left = writer - 1
        pred_right = writer - 1
        if group_rank == 0:
            pred_left = range_write_index.lookup_dkv_predecessor(
                dkv_write_order.mWriteIndex,
                dkv_write_order.index_stride,
                dkv_write_order.writer_stride,
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
                pred_right = range_write_index.lookup_dkv_predecessor(
                    dkv_write_order.mWriteIndex,
                    dkv_write_order.index_stride,
                    dkv_write_order.writer_stride,
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
        wait_published(dkv_write_order.mWriteOrder, left, head_kv, pred_left)
        if right != left:
            wait_published(dkv_write_order.mWriteOrder, right, head_kv, pred_right)
    return left, right, left_weight, right_weight, writer, head_kv


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
                        cur_prev = range_write_index.dq_pair_prev_k_tile_on_block(
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
def dq_plan(
    dq_write_order: WriteOrderTensors,
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
    left, right = range_write_index.tile_blocks(
        q_start, q_start + seqlen_info.seqlen_q, m_block, tile_m
    )
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
        n_min_m, _ = range_write_index.dq_pair_n_bounds(
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
                l_prev, r_prev = range_write_index.dq_pair_n_bounds(
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
            wait_pred_left = batch_idx * dq_write_order.writer_stride + prev_n_left + 1
        else:
            wait_pred_left = range_write_index.lookup_dq_predecessor(
                dq_write_order.mWriteIndex,
                dq_write_order.index_stride,
                dq_write_order.writer_stride,
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
            wait_pred_right = (
                batch_idx * dq_write_order.writer_stride + prev_n_right + 1
            )
        else:
            wait_pred_right = range_write_index.lookup_dq_predecessor(
                dq_write_order.mWriteIndex,
                dq_write_order.index_stride,
                dq_write_order.writer_stride,
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
def dq_wait_planned(
    dq_write_order: WriteOrderTensors,
    left: Int32,
    right: Int32,
    wait_pred_left: Int32,
    wait_pred_right: Int32,
    q_head: Int32,
) -> None:
    if wait_pred_left >= 0:
        wait_published(dq_write_order.mWriteOrder, left, q_head, wait_pred_left)
    if wait_pred_right >= 0:
        wait_published(dq_write_order.mWriteOrder, right, q_head, wait_pred_right)


@cute.jit
def dq_arrive_planned(
    dq_write_order: WriteOrderTensors,
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
                dq_write_order.mWriteOrder,
                left,
                q_head,
                dq_writer,
                Int32(2),
                cluster_size,
            )
        if arrive_right:
            arrive(
                dq_write_order.mWriteOrder,
                right,
                q_head,
                dq_writer,
                Int32(2),
                cluster_size,
            )
