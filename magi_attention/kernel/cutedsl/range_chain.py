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
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

# isort: split
from quack.cute_dsl_utils import ParamsBase

from .cutedsl_utils import (
    elem_pointer,
    ld_acquire,
    mbarrier_arrive_peer_release_cluster,
    mbarrier_wait_acquire_cluster,
    st_global_release_gpu_i32,
)


@dataclass
class ChainTensors(ParamsBase):
    """Zero-init: mChain[b, h] = [published, count], mConflict[slot, b] = last_writer.
    Writer j of range r is r * writer_stride + j + 1 (0: none)."""

    mChain: cute.Tensor
    mConflict: cute.Tensor
    writer_stride: Int32


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
def scan_conflicts(
    mRanges: cute.Tensor,
    mConflict: cute.Tensor,
    slot: Int32,
    range_from: Int32,
    range_to: Int32,
    block_size: cutlass.Constexpr[int],
    last_writer: Optional[Callable[[Int32], Int32]] = None,
    rows_per_token: cutlass.Constexpr[int] = 1,
    mCuBatches: Optional[cute.Tensor] = None,
    block_writer: Optional[Callable[[Int32, Int32, Int32], Int32]] = None,
) -> None:
    """Warp-collective scan; sync_warp per row orders lanes writing shared blocks."""
    lane = cute.arch.lane_idx()
    r = range_from
    while r < range_to:
        writer = Int32(0)
        if const_expr(block_writer is None):
            assert last_writer is not None
            writer = last_writer(r)
        if const_expr(mCuBatches is None):
            _record_row_blocks(
                mRanges,
                mConflict,
                slot,
                r,
                r,
                writer,
                block_size,
                rows_per_token,
                lane,
                block_writer=block_writer,
                monotonic_group_merge=False,
            )
            cute.arch.sync_warp()
        else:
            assert mCuBatches is not None
            row_start = Int32(mCuBatches[r])
            row = row_start
            row_end = Int32(mCuBatches[r + 1])
            while row < row_end:
                _record_row_blocks(
                    mRanges,
                    mConflict,
                    slot,
                    r,
                    row,
                    writer,
                    block_size,
                    rows_per_token,
                    lane,
                    block_writer=block_writer,
                    monotonic_group_merge=const_expr(block_writer is not None),
                    is_first_group_row=(row == row_start),
                )
                cute.arch.sync_warp()
                row += 1
        r += 1


@cute.jit
def _record_row_blocks(
    mRanges: cute.Tensor,
    mConflict: cute.Tensor,
    slot: Int32,
    group_idx: Int32,
    row: Int32,
    writer: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    lane: Int32,
    block_writer: Optional[Callable[[Int32, Int32, Int32], Int32]] = None,
    monotonic_group_merge: cutlass.Constexpr[bool] = False,
    is_first_group_row: cutlass.Boolean = cutlass.Boolean(True),
) -> None:
    start = mRanges[row, 0] * rows_per_token
    end = mRanges[row, 1] * rows_per_token
    if const_expr(block_writer is None):
        if end > start and writer != 0:
            block = start // block_size + lane
            last_block = (end - 1) // block_size
            while block <= last_block:
                record_conflict(mConflict, slot, block, writer)
                block += cute.arch.WARP_SIZE
    else:
        assert block_writer is not None
        if end > start:
            block = start // block_size + lane
            last_block = (end - 1) // block_size
            while block <= last_block:
                w = block_writer(group_idx, row, block)
                if w != 0:
                    if const_expr(monotonic_group_merge):
                        if is_first_group_row or w > Int32(mConflict[slot, block]):
                            record_conflict(mConflict, slot, block, w)
                    else:
                        record_conflict(mConflict, slot, block, w)
                block += cute.arch.WARP_SIZE


@cute.jit
def rows_have_block(
    mRanges: cute.Tensor,
    row_from: Int32,
    row_to: Int32,
    block: Int32,
    block_size: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    found = cutlass.Boolean(False)
    row = row_from
    while row < row_to:
        start = mRanges[row, 0]
        end = mRanges[row, 1]
        if end > start and start // block_size <= block:
            if (end - 1) // block_size >= block:
                found = cutlass.Boolean(True)
        row += 1
    return found


@cute.jit
def record_conflict(
    mConflict: cute.Tensor, slot: Int32, block: Int32, writer: Int32
) -> None:
    mConflict[slot, block] = writer


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
