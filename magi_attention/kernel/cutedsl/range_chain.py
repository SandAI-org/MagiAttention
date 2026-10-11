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
3. arrives; the arrivals of ``w`` on ``b`` sum to 2 and the one completing
   that sum publishes ``w`` on ``b``.

A writer whose tiles sit in several work tiles (fwd O/LSE) arrives with
weight 2 from a tile that is its only one with rows in ``b`` and 1 from
each of two tiles sharing ``b`` (``tile_events``). A writer whose tiles all
sit in one work tile (bwd dQ: one relation and cluster K tile walking the
relation's Q tiles in order) arrives once per block, with weight 2, after
the last of its tiles with rows in ``b`` (``tile_blocks``). Every event
arrives exactly once, a tile whose rows got no contribution included, so a
successor never waits on an arrival that never comes.

Under 2-CTA both CTAs of a cluster write parts of one logical tile. Each
arrives the logical tile's weight on the logical tile's blocks after its own
part is complete, and a block publishes once the sum reaches 2 times the
cluster size, so the publish covers the parts of every CTA.

The chain orders writers, not the two CTAs inside one. Under RangeMerge a
writer's pairs may overlap in Q, and a row in the half of CTA 1 for one
pair can be in the half of CTA 0 for the next, so the bwd dQ of a 2-CTA
merged writer also hands off between its CTAs at every pair boundary
(``pair_handoff_arrive`` / ``pair_handoff_wait``): pair ``j`` adds only
after both CTAs' additions of the pairs before it landed.

Chain state, int32 ``(num_blocks, num_heads, 2)``: per (block, head), the
last published writer and the arrival count of the writer in flight.

Conflict state, int32 ``(num_slots, num_blocks)``: per execution slot, the
writers whose blocks the slot's incremental scan already recorded, each
block holding the highest such writer with an event there. A slot walks its
tiles in non-decreasing range order, so before a tile of range ``r`` it
records the ranges in ``[r_last, r)`` and reads its predecessors back.

Rows are buffer rows, ``rows_per_token`` per token of the ranges. A
PackGQA buffer (fwd O/LSE with ``G`` q heads per kv head) has ``G`` rows
per token, row ``G * token + g`` for q head ``g`` of a kv head, so token
range ``[s, e)`` is the row range ``[s * G, e * G)``, the head axis of the
chain state is the kv head, and ``num_blocks`` counts blocks of packed rows.
Its tiles are cut from ``s * G``; ``G`` divides ``block_size``, so a tile
starts on a token boundary and still has events on at most two blocks.

A range is the scheduling unit the writer numbers count: a relation, or
under RangeMerge a merged group, whose rows are the row ranges of its pairs,
rows ``cu_batches[r]`` to ``cu_batches[r + 1]`` of the pair table. Pairs of
one group may overlap, so a writer walking them can have rows in a block
from several pairs; it arrives on the block after the last of them
(``rows_have_block`` over the later pairs).
"""

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
)


@dataclass
class ChainTensors(ParamsBase):
    """Chain and conflict state of one reduced buffer, zero before the launch.

    Writer ``j`` of range ``r`` is number ``r * writer_stride + j + 1``, so
    writer numbers grow with (range, j) and stay unique while ``j`` is below
    ``writer_stride``.
    """

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
    """``(left, right)``: the first and last block tile ``tile_idx`` of a range
    cut into ``block_size``-row tiles has a valid row in. The tile must have
    a valid row; ``left == right`` means one block."""
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
    """Events of tile ``tile_idx`` of a range cut into ``block_size``-row tiles.

    Returns ``(left, right, left_weight, right_weight)`` for a writer whose
    tiles sit in several work tiles; ``left == right`` means a single event,
    whose weight is ``left_weight``. The tile must have a valid row.
    """
    row = range_start + tile_idx * block_size
    left, right = tile_blocks(range_start, range_end, tile_idx, block_size)
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
    last_writer: Callable[[Int32], Int32],
    rows_per_token: cutlass.Constexpr[int] = 1,
    mCuBatches: Optional[cute.Tensor] = None,
) -> None:
    """Record the blocks of the ranges in ``[range_from, range_to)`` for ``slot``.

    Block ``b`` of range ``r`` gets ``last_writer(r)``, the highest writer of
    ``r`` with events on its blocks; 0 means ``r`` has no work tile, which
    records nothing, as does an empty range. ``mRanges`` holds token ranges;
    range ``[s, e)`` covers the rows ``[s, e) * rows_per_token``, the blocks
    ``tile_events`` gives for its tiles. Without ``mCuBatches`` range ``r``
    is row ``r`` of ``mRanges``; with it, a merged group whose rows are
    ``mCuBatches[r]`` to ``mCuBatches[r + 1]``, all recorded with the group's
    writer. The whole warp calls; its lanes write the blocks of one row in
    parallel, and any lane may read ``mConflict[slot, ...]`` after the
    return.

    A block shared by consecutive rows is stored by different lanes (its
    lane follows the row's first block), so the warp synchronizes after
    every row, empty ones included: without it a lane's store of range
    ``r`` may land after another lane's store of range ``r + 1`` and leave
    a stale predecessor.
    """
    lane = cute.arch.lane_idx()
    r = range_from
    while r < range_to:
        writer = last_writer(r)
        if const_expr(mCuBatches is None):
            _record_row_blocks(
                mRanges, mConflict, slot, r, writer, block_size, rows_per_token, lane
            )
            cute.arch.sync_warp()
        else:
            assert mCuBatches is not None
            row = Int32(mCuBatches[r])
            row_end = Int32(mCuBatches[r + 1])
            while row < row_end:
                _record_row_blocks(
                    mRanges,
                    mConflict,
                    slot,
                    row,
                    writer,
                    block_size,
                    rows_per_token,
                    lane,
                )
                cute.arch.sync_warp()
                row += 1
        r += 1


@cute.jit
def _record_row_blocks(
    mRanges: cute.Tensor,
    mConflict: cute.Tensor,
    slot: Int32,
    row: Int32,
    writer: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    lane: Int32,
) -> None:
    """This lane's share of recording ``writer`` on the blocks of row ``row``
    of ``mRanges``, a token range of ``rows_per_token`` rows per token;
    nothing for an empty row or writer 0."""
    start = mRanges[row, 0] * rows_per_token
    end = mRanges[row, 1] * rows_per_token
    if end > start and writer != 0:
        block = start // block_size + lane
        last_block = (end - 1) // block_size
        while block <= last_block:
            record_conflict(mConflict, slot, block, writer)
            block += cute.arch.WARP_SIZE


@cute.jit
def rows_have_block(
    mRanges: cute.Tensor,
    row_from: Int32,
    row_to: Int32,
    block: Int32,
    block_size: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Whether a row in ``[row_from, row_to)`` of ``mRanges`` has a valid row
    in ``block``; empty rows have none."""
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
    """Record ``writer`` as the last writer of ``block`` in ``slot``'s row."""
    mConflict[slot, block] = writer


@cute.jit
def wait_published(
    mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32
) -> None:
    """Spin until ``block`` of ``head`` publishes ``writer`` (acquire), then
    order the acquired view before this thread's later async-proxy accesses.

    The predecessor merged its rows with bulk reductions or stores (async
    proxy) from another thread block, and the waiter's merge may be bulk
    too. The acquire is in the generic proxy, and the PTX memory model
    (8.9.5, proxy-preserved base causality) orders two async-proxy accesses
    to one address only within one thread block; cross-proxy ordering needs
    a proxy fence (9.7.9.26.2), so ``fence.proxy.async.global`` follows the
    acquire. Without it a bulk reduction may read a row before the
    predecessor's merge. The release side needs none: a completed bulk
    operation is followed by an implicit async-to-generic proxy fence."""
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
    """Arrive with ``weight`` on ``block`` of ``head``; completing
    ``2 * cluster_size`` publishes ``writer``.

    One thread arrives per event. Before it, every thread of the CTA that
    wrote rows of the event must have completed them at the destination
    (bulk reductions: ``cp_async_bulk_wait_group(0, read=False)`` by the
    issuing thread) and reached a CTA barrier with the arriving thread; the
    arriving thread then fences (``fence_acq_rel_gpu``) so the acq_rel
    arrival carries those writes."""
    count_ptr = elem_pointer(mChain, (block, head, 1))
    prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
    # Exact match: weights are those of the logical tile, so the arrival that
    # completes the writer's sum is unique, and a sum stepping past the
    # threshold never publishes (the successor then waits forever, which the
    # tests detect, instead of starting early).
    if prev + weight == 2 * cluster_size:
        # The next writer of this block starts only after the publish below,
        # so resetting the count before it cannot race with its arrivals; the
        # release of the publish orders the reset before it.
        cute.arch.atomic_exch(count_ptr, Int32(0), sem="relaxed", scope="gpu")
        publish(mChain, block, head, writer)


@cute.jit
def publish(mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32) -> None:
    """Publish ``writer`` on ``block`` of ``head`` (release, gpu scope).

    Called once per (writer, block), by the arrival completing the writer's
    weight, after it reset the count. The acq_rel arrivals before it carry
    every participant's destination writes, so a successor that reads
    ``writer`` with ``wait_published`` (acquire) sees them.
    """
    cute.arch.atomic_exch(
        elem_pointer(mChain, (block, head, 0)),
        writer,
        sem="release",
        scope="gpu",
    )


@cute.jit
def pair_handoff_arrive(mbar: cute.Pointer, handoff: Int32, peer: Int32) -> None:
    """Arrive on handoff ``handoff`` of a 2-CTA RangeMerge dQ writer, on the
    barrier ``mbar[handoff & 1]`` of the peer CTA of rank ``peer``.

    One thread per CTA arrives, at the start of every pair after the first,
    after every bulk reduction of the earlier pairs completed
    (``cp_async_bulk_wait_group(0, read=False)`` by the issuing thread) and
    the issuing and arriving threads met at a CTA barrier. A completed bulk
    reduction is followed by an implicit async-to-generic proxy fence (PTX
    ISA 9.7.9.26.2), and the cluster-scope release carries the reduced rows
    to the peer's wait. Handoffs alternate between two barriers of arrival
    count 1: a CTA arrives on handoff ``k + 2`` only after its wait of
    ``k + 1``, which needs the peer's arrival of ``k + 1``, issued after the
    peer's wait of ``k``, so a barrier is at most one phase ahead of its
    waiter; with one barrier the peer's arrivals of ``k`` and ``k + 1``
    could both land before the wait of ``k``, and a parity wait reads two
    completions as none."""
    mbarrier_arrive_peer_release_cluster(mbar + (handoff & 1), peer)


@cute.jit
def pair_handoff_wait(mbar: cute.Pointer, handoff: Int32) -> None:
    """Wait for the peer's arrival of handoff ``handoff`` on this CTA's
    ``mbar[handoff & 1]`` (parity ``(handoff >> 1) & 1``), then order the
    acquired view before this CTA's bulk reductions.

    The thread that arrived waits, before the CTA barrier that precedes the
    pair's first bulk reduction. Both CTAs must hand off at the same pairs,
    or one waits forever. The acquire is in the generic proxy, the bulk
    reductions that follow are in the async proxy, and the peer's reductions
    of the same rows were made by another thread block: the PTX memory
    model (8.9.5, proxy-preserved base causality) orders two async-proxy
    accesses to one address only within one thread block, and cross-proxy
    ordering needs a proxy fence (9.7.9.26.2), so ``fence.proxy.async.global``
    follows the acquire; without it the reduction may read the row before
    the peer's addition."""
    mbarrier_wait_acquire_cluster(mbar + (handoff & 1), (handoff >> 1) & 1)
    cute.arch.fence_proxy("async.global")
