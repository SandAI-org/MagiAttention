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

"""Load-balanced dispatching of q_ranges with various chunk sizes.

Each input q_range carries its own attn workload, atomic chunk size and sample id.
The ranges are cut into atomic slices, packed into big chunks (one sample each),
and then dispatched to ``num_groups`` groups under a token capacity constraint
so that the attn workload of each group is as balanced as possible.
"""

import heapq
import math
from dataclasses import dataclass
from typing import Callable

from magi_attention.utils import ceil_div


@dataclass(frozen=True)
class VariousChunkRange:
    """One input q_range to be dispatched

    Args:
        q_range (tuple[int, int]): the token range [start, end)
        weight (float): the attn workload of the whole range,
            assumed to be uniformly distributed over its tokens
        chunk_size (int): the atomic slice size, e.g. the q_cluster size of DSA,
            the last slice of the range might be shorter
        sample_id (int): the id of the sample this range belongs to
    """

    q_range: tuple[int, int]
    weight: float
    chunk_size: int
    sample_id: int

    @property
    def start(self) -> int:
        return self.q_range[0]

    @property
    def end(self) -> int:
        return self.q_range[1]

    @property
    def seqlen(self) -> int:
        return self.q_range[1] - self.q_range[0]


@dataclass(frozen=True)
class VariousChunkSlice:
    """An atomic slice cut from one input range, which can not be split any more"""

    range_idx: int
    start: int
    end: int
    sample_id: int
    workload: float

    @property
    def seqlen(self) -> int:
        return self.end - self.start


@dataclass
class BigChunk:
    """A pack of adjacent slices from the same sample"""

    chunk_id: int
    sample_id: int
    slices: list[VariousChunkSlice]
    oversized: bool = False

    @property
    def seqlen(self) -> int:
        return sum(s.seqlen for s in self.slices)

    @property
    def workload(self) -> float:
        return sum(s.workload for s in self.slices)

    @property
    def density(self) -> float:
        return self.workload / self.seqlen


@dataclass
class VariousChunkSolution:
    """The dispatch solution, where each list has length ``num_groups``

    Args:
        group_q_ranges (list[list[tuple[int, int]]]): the q_ranges of each group,
            sorted by start, with contiguous slices of the same sample merged
        group_sample_ids (list[list[int]]): the sample id of each range in ``group_q_ranges``
        group_slices (list[list[VariousChunkSlice]]): the atomic slices of each group, sorted by start
        group_seqlens (list[int]): the number of valid tokens of each group
        group_pad_sizes (list[int]): the number of padding tokens of each group
        padded_seqlen (int): the seqlen of every group after padding
        group_workloads (list[float]): the attn workload of each group
        capacity (int): the token capacity of each group
        num_overflow_slices (int): the number of slices placed beyond the capacity by fallback
    """

    group_q_ranges: list[list[tuple[int, int]]]
    group_sample_ids: list[list[int]]
    group_slices: list[list[VariousChunkSlice]]
    group_seqlens: list[int]
    group_pad_sizes: list[int]
    padded_seqlen: int
    group_workloads: list[float]
    capacity: int
    num_overflow_slices: int

    @property
    def max_workload(self) -> float:
        return max(self.group_workloads)

    @property
    def imbalance_ratio(self) -> float:
        """max / mean of the group workloads, 1.0 means perfectly balanced"""
        mean = sum(self.group_workloads) / len(self.group_workloads)
        return self.max_workload / mean if mean > 0 else 1.0


def _round_up(x: int, align: int) -> int:
    return ceil_div(x, align) * align


def _validate_ranges(ranges: list[VariousChunkRange]) -> None:
    if len(ranges) == 0:
        raise ValueError("ranges should not be empty")

    prev_end = None
    for i, r in enumerate(ranges):
        if r.seqlen <= 0:
            raise ValueError(f"ranges[{i}] should be non-empty, got {r.q_range=}")
        if r.chunk_size <= 0:
            raise ValueError(f"ranges[{i}] should have positive {r.chunk_size=}")
        if prev_end is not None and r.start < prev_end:
            raise ValueError(
                f"ranges should be sorted and non-overlapping, but {ranges[i].q_range=} "
                f"starts before the previous end {prev_end}"
            )
        prev_end = r.end


def make_slices(
    ranges: list[VariousChunkRange],
    workloads: list[float],
) -> list[VariousChunkSlice]:
    """Cut each range into atomic slices by its own chunk_size,
    and distribute its workload to the slices proportionally to their seqlens
    """
    slices: list[VariousChunkSlice] = []
    for range_idx, (r, workload) in enumerate(zip(ranges, workloads)):
        for start in range(r.start, r.end, r.chunk_size):
            end = min(start + r.chunk_size, r.end)
            slices.append(
                VariousChunkSlice(
                    range_idx=range_idx,
                    start=start,
                    end=end,
                    sample_id=r.sample_id,
                    workload=workload * (end - start) / r.seqlen,
                )
            )
    return slices


def pack_big_chunks(
    slices: list[VariousChunkSlice],
    big_chunk_size: int,
) -> list[BigChunk]:
    """Pack adjacent slices in order into big chunks, where a new big chunk is opened
    when the sample changes or the next slice does not fit into ``big_chunk_size``.
    A slice longer than ``big_chunk_size`` becomes an oversized big chunk on its own.
    """
    big_chunks: list[BigChunk] = []
    cur: BigChunk | None = None

    def _flush() -> None:
        nonlocal cur
        if cur is not None:
            big_chunks.append(cur)
            cur = None

    for s in slices:
        if s.seqlen > big_chunk_size:
            _flush()
            big_chunks.append(
                BigChunk(
                    chunk_id=len(big_chunks),
                    sample_id=s.sample_id,
                    slices=[s],
                    oversized=True,
                )
            )
            continue

        if (
            cur is None
            or cur.sample_id != s.sample_id
            or cur.seqlen + s.seqlen > big_chunk_size
        ):
            _flush()
            cur = BigChunk(chunk_id=len(big_chunks), sample_id=s.sample_id, slices=[])
        cur.slices.append(s)

    _flush()

    return big_chunks


class _GroupState:
    """The mutable state of all groups with a min-heap keyed by (workload, group_idx)"""

    def __init__(self, num_groups: int, capacity: int):
        self.capacity = capacity
        self.seqlens = [0] * num_groups
        self.workloads = [0.0] * num_groups
        self.slices: list[list[VariousChunkSlice]] = [[] for _ in range(num_groups)]
        self.heap = [(0.0, g) for g in range(num_groups)]
        heapq.heapify(self.heap)
        self.num_overflow_slices = 0

    def remaining(self, group_idx: int) -> int:
        return self.capacity - self.seqlens[group_idx]

    def place(self, group_idx: int, s: VariousChunkSlice) -> None:
        self.slices[group_idx].append(s)
        self.seqlens[group_idx] += s.seqlen
        self.workloads[group_idx] += s.workload

    def pop_min_fit(self, seqlen: int) -> tuple[int | None, list[tuple[float, int]]]:
        """Pop the group with the minimum workload whose remaining capacity fits ``seqlen``,
        returning it with the popped but unfit heap entries, which should be pushed back later
        """
        skipped: list[tuple[float, int]] = []
        while self.heap:
            entry = heapq.heappop(self.heap)
            if self.remaining(entry[1]) >= seqlen:
                return entry[1], skipped
            skipped.append(entry)
        return None, skipped

    def push(self, group_idx: int) -> None:
        heapq.heappush(self.heap, (self.workloads[group_idx], group_idx))

    def push_back(self, entries: list[tuple[float, int]]) -> None:
        for entry in entries:
            heapq.heappush(self.heap, entry)

    def place_overflow(self, s: VariousChunkSlice) -> None:
        """Place the slice to the group with the minimum seqlen beyond the capacity"""
        num_groups = len(self.seqlens)
        group_idx = min(
            range(num_groups), key=lambda g: (self.seqlens[g], self.workloads[g], g)
        )
        self.heap.remove((self.workloads[group_idx], group_idx))
        heapq.heapify(self.heap)
        self.place(group_idx, s)
        self.push(group_idx)
        self.num_overflow_slices += 1


def _assign_whole(state: _GroupState, big_chunk: BigChunk) -> None:
    """Assign the big chunk as a whole to the min-workload group that fits it"""
    group_idx, skipped = state.pop_min_fit(big_chunk.seqlen)
    state.push_back(skipped)
    if group_idx is None:
        for s in big_chunk.slices:
            state.place_overflow(s)
        return

    for s in big_chunk.slices:
        state.place(group_idx, s)
    state.push(group_idx)


def _assign_by_slices(state: _GroupState, big_chunk: BigChunk) -> None:
    """Assign the slices of the big chunk in order to the min-workload group,
    until the group is full or the next slice does not fit,
    then continue with the remaining slices to the next min-workload group
    """
    slices = big_chunk.slices
    i = 0
    while i < len(slices):
        group_idx, skipped = state.pop_min_fit(slices[i].seqlen)
        state.push_back(skipped)
        if group_idx is None:
            state.place_overflow(slices[i])
            i += 1
            continue

        while i < len(slices) and state.remaining(group_idx) >= slices[i].seqlen:
            state.place(group_idx, slices[i])
            i += 1
        state.push(group_idx)


def _merge_group_slices(
    slices: list[VariousChunkSlice],
) -> tuple[list[tuple[int, int]], list[int]]:
    q_ranges: list[tuple[int, int]] = []
    sample_ids: list[int] = []
    for s in slices:
        if q_ranges and sample_ids[-1] == s.sample_id and q_ranges[-1][1] == s.start:
            q_ranges[-1] = (q_ranges[-1][0], s.end)
        else:
            q_ranges.append((s.start, s.end))
            sample_ids.append(s.sample_id)
    return q_ranges, sample_ids


def solve_various_chunk_dispatch(
    ranges: list[VariousChunkRange],
    num_groups: int,
    big_chunk_size: int = 512,
    workload_fn: Callable[[VariousChunkRange], float] | None = None,
    align: int | None = None,
    slack: float = 0.0,
    pad_align: int = 1,
) -> VariousChunkSolution:
    """Dispatch the q_ranges with various chunk sizes to ``num_groups`` groups,
    to balance the attn workload of each group under a token capacity constraint.

    Steps:
        1. cut each range into atomic slices by its chunk_size.
        2. pack adjacent slices of the same sample into big chunks of at most ``big_chunk_size``,
            where a slice longer than ``big_chunk_size`` becomes an oversized big chunk on its own.
        3. the token capacity of each group is
            ``round_up(ceil(total_seqlen * (1 + slack) / num_groups), align)``,
            and at least the seqlen of the longest oversized big chunk.
        4. the 1st wave assigns the oversized big chunks in descending order of seqlen,
            each as a whole to the min-workload group that fits it.
        5. the 2nd wave assigns the other big chunks in descending order of density (workload / seqlen),
            slice by slice to the min-workload group until it is full or the next slice does not fit,
            and the remaining slices go on to the next min-workload group.
        6. if no group fits a slice, it is placed to the group with the minimum seqlen beyond the capacity.

    Args:
        ranges (list[VariousChunkRange]): the sorted and non-overlapping q_ranges to dispatch
        num_groups (int): the number of groups, e.g. the inter cp size
        big_chunk_size (int): the maximum seqlen of a non-oversized big chunk
        workload_fn (Callable[[VariousChunkRange], float] | None): the attn cost function
            to estimate the workload of each range, if None, use ``range.weight`` instead
        align (int | None): the alignment of the capacity,
            if None, use the maximum chunk_size capped by ``big_chunk_size``
        slack (float): the ratio of extra capacity of each group
        pad_align (int): the alignment of the padded seqlen of all groups

    Returns:
        VariousChunkSolution: the dispatch solution
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups should be positive, got {num_groups=}")
    if big_chunk_size <= 0:
        raise ValueError(f"big_chunk_size should be positive, got {big_chunk_size=}")
    if align is not None and align <= 0:
        raise ValueError(f"align should be positive, got {align=}")
    if slack < 0:
        raise ValueError(f"slack should be non-negative, got {slack=}")
    if pad_align <= 0:
        raise ValueError(f"pad_align should be positive, got {pad_align=}")
    _validate_ranges(ranges)

    workloads = [
        float(workload_fn(r)) if workload_fn is not None else float(r.weight)
        for r in ranges
    ]
    for i, workload in enumerate(workloads):
        if workload < 0 or math.isnan(workload):
            raise ValueError(
                f"the workload of ranges[{i}] should be >= 0, got {workload}"
            )

    # --------------      make big chunks      -------------- #

    slices = make_slices(ranges, workloads)
    big_chunks = pack_big_chunks(slices, big_chunk_size)
    oversized_chunks = [c for c in big_chunks if c.oversized]
    normal_chunks = [c for c in big_chunks if not c.oversized]

    # --------------      compute capacity      -------------- #

    if align is None:
        align = min(max(r.chunk_size for r in ranges), big_chunk_size)
    total_seqlen = sum(r.seqlen for r in ranges)
    capacity = _round_up(math.ceil(total_seqlen * (1 + slack) / num_groups), align)
    if oversized_chunks:
        capacity = max(capacity, max(c.seqlen for c in oversized_chunks))

    # --------------      assign in two waves      -------------- #

    state = _GroupState(num_groups, capacity)

    oversized_chunks.sort(key=lambda c: (-c.seqlen, c.chunk_id))
    for c in oversized_chunks:
        _assign_whole(state, c)

    normal_chunks.sort(key=lambda c: (-c.density, c.chunk_id))
    for c in normal_chunks:
        _assign_by_slices(state, c)

    # --------------      make solution      -------------- #

    group_slices = [sorted(s, key=lambda x: x.start) for s in state.slices]
    group_q_ranges, group_sample_ids = [], []
    for s in group_slices:
        q_ranges, sample_ids = _merge_group_slices(s)
        group_q_ranges.append(q_ranges)
        group_sample_ids.append(sample_ids)

    padded_seqlen = _round_up(max(state.seqlens), pad_align)

    return VariousChunkSolution(
        group_q_ranges=group_q_ranges,
        group_sample_ids=group_sample_ids,
        group_slices=group_slices,
        group_seqlens=list(state.seqlens),
        group_pad_sizes=[padded_seqlen - seqlen for seqlen in state.seqlens],
        padded_seqlen=padded_seqlen,
        group_workloads=list(state.workloads),
        capacity=capacity,
        num_overflow_slices=state.num_overflow_slices,
    )
