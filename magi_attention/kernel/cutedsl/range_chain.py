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

"""Deterministic range-lock chain of the SM100 range kernels.

The writers of overlapping row ranges merge into a shared buffer in writer
number order instead of lock-acquisition order, so the merged result does
not depend on the grid or on timing. The CPU model of this protocol is
``tests/test_kernel/cutedsl/test_ffa_range_deterministic_protocol.py``.

Rows are grouped into physical blocks of ``block_size`` rows. A logical tile
whose valid rows are ``[a, e)`` has one event on each block it has a valid
row in, ``a // block_size`` to ``(e - 1) // block_size``; a tile of at most
``block_size`` rows has events on at most two blocks. Protection rows past
the range end, and stage tiles without a valid row, have no event. An event
of writer ``w`` on block ``b``:

1. waits until ``b`` publishes the predecessor: the closest lower writer
   with an event on ``b``, 0 if none;
2. merges its rows, makes the stores gpu-visible;
3. arrives with weight 2 if it is the writer's only tile with rows in ``b``
   and 1 otherwise; the arrival completing 2 publishes ``w`` on ``b``.

Every event arrives exactly once, a tile whose rows got no contribution
included, so a successor never waits on an arrival that never comes.

Chain state, int32 ``(num_blocks, num_heads, 2)``: per (block, head), the
last published writer and the arrival count of the writer in flight.

Conflict state, int32 ``(num_slots, num_blocks)``: per execution slot, the
writers whose blocks the slot's incremental scan already recorded, each
block holding the highest such writer with an event there. A slot walks its
tiles in non-decreasing writer order, so before a tile of writer ``w`` it
records the ranges in ``[w_last, w)`` and reads its predecessors back.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .cutedsl_utils import elem_pointer, ld_acquire


@cute.jit
def tile_events(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    """Events of tile ``tile_idx`` of a range cut into ``block_size``-row tiles.

    Returns ``(left, right, left_weight, right_weight)``; ``left == right``
    means a single event, whose weight is ``left_weight``. The tile must have
    a valid row.
    """
    row = range_start + tile_idx * block_size
    row_end = cutlass.min(row + block_size, range_end)
    left = row // block_size
    right = (row_end - 1) // block_size
    # Tiles start off a block boundary iff the range does; then tile k - 1
    # also has rows in this tile's left block, and tile k + 1, if any, in its
    # right block.
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
) -> None:
    """Record the blocks of the ranges in ``[range_from, range_to)`` for ``slot``.

    Block ``b`` of range ``r`` gets ``r + 1``. The whole warp calls; its lanes
    write in parallel and the warp is synchronized on return, so any lane may
    read ``mConflict[slot, ...]`` afterwards. An empty range has no tile, so
    it records nothing.
    """
    lane = cute.arch.lane_idx()
    r = range_from
    while r < range_to:
        start = mRanges[r, 0]
        end = mRanges[r, 1]
        if end > start:
            block = start // block_size + lane
            last_block = (end - 1) // block_size
            while block <= last_block:
                mConflict[slot, block] = r + 1
                block += cute.arch.WARP_SIZE
        r += 1
    cute.arch.sync_warp()


@cute.jit
def wait_published(
    mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32
) -> None:
    """Spin until ``block`` of ``head`` publishes ``writer`` (acquire)."""
    ptr = elem_pointer(mChain, (block, head, 0))
    published = ld_acquire(ptr)
    while published != writer:
        published = ld_acquire(ptr)


@cute.jit
def arrive(
    mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32, weight: Int32
) -> None:
    """Arrive with ``weight`` on ``block`` of ``head``; completing 2 publishes
    ``writer``. The caller's stores must already be gpu-visible."""
    count_ptr = elem_pointer(mChain, (block, head, 1))
    prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
    if prev + weight == 2:
        # The next writer of this block starts only after the publish below,
        # so resetting the count before it cannot race with its arrivals.
        cute.arch.atomic_exch(count_ptr, Int32(0), sem="relaxed", scope="gpu")
        cute.arch.atomic_exch(
            elem_pointer(mChain, (block, head, 0)),
            writer,
            sem="release",
            scope="gpu",
        )
