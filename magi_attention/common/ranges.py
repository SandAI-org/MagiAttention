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

from functools import lru_cache
from typing import Any, Iterator, Sequence, TypeAlias, Union

import torch

from magi_attention.utils import nvtx

from . import is_cpp_backend_enable
from .range import AttnRange, NaiveRange, RangeError

NaiveRanges: TypeAlias = Sequence[NaiveRange]

__all__ = [
    "is_valid_cu_seqlens",
    "check_valid_cu_seqlens",
    "AttnRanges",
]


def is_valid_cu_seqlens(cu_seqlens: list[int], seq_len: int) -> bool:
    if len(cu_seqlens) == 0:
        return True

    if not cu_seqlens[0] == 0:
        return False

    if not all(cu_seqlens[i - 1] < cu_seqlens[i] for i in range(1, len(cu_seqlens))):
        return False

    if not cu_seqlens[-1] == seq_len:
        return False

    return True


def check_valid_cu_seqlens(cu_seqlens: list[int], seq_len: int) -> None:
    if not is_valid_cu_seqlens(cu_seqlens, seq_len):
        raise ValueError(
            f"The cu_seqlens {cu_seqlens} is invalid against the rule: 'cu_seqlens[0] == 0', \
            and 'cu_seqlens[i-1] < cu_seqlens[i], for any i in [1, len(cu_seqlens))'"  # noqa
        )


def _calc_prefix_offset(merged_ranges: "AttnRanges") -> list[int]:
    """NOTE: the ranges passed in should be merged already,
    no check is applyed here
    """
    prefix_offset = [0]
    for item in merged_ranges:
        prefix_offset.append(prefix_offset[-1] + item.seqlen)
    prefix_offset.pop()

    return prefix_offset


def _binary_search(
    attn_ranges: Union[list[AttnRange], "AttnRanges"], target_range: AttnRange
) -> int:
    # Elements to the left of 'left' are less than or equal to target, elements to the right of 'right' are greater than target
    left, right = 0, len(attn_ranges) - 1
    while left <= right:
        mid = (left + right) // 2
        if attn_ranges[mid].start > target_range.start:
            right = mid - 1
        else:
            left = mid + 1
    return right


class AttnRanges:
    """
    A dataclass to manage a list of 'AttnRange' objects for attention computation

    Advanced methods, including make_range_local, make_ranges_local, to_local_ranges,
    find_hole_ranges, find_overlap_ranges, etc, are designed to facilitate various range mappings

    NOTE: What are the local_ranges of AttnRanges?
        Since each attn_range in AttnRanges describes a continuous storage in actual memory,
        AttnRanges can choose to map to a continuous storage in actual memory that satisfies:
            1. It can store all attn_ranges in AttnRanges
            2. All attn_ranges are stored in order (by range.start and range.end)
        So the local_ranges of AttnRanges are the actual positions of each attn_range in this memory

    Example:
        The local_ranges of [[5, 10), [15, 20), [25, 30)] are [[0, 5), [5, 10), [10, 15)]
        The local_ranges of [[5, 10), [25, 30), [15, 20)] are [[0, 5), [5, 10), [10, 15)]
        Their mapping relationships are as follows:
        .....[5, 10).....[15, 20).....[25, 30).....
                |           |             |
             [0, 5)      [5, 10)      [10, 15)

    NOTE: Explanation of sorted and merged for AttnRanges:
        sorted requires that each attn_range in AttnRanges is sorted by start in ascending order
        merged requires that each attn_range in AttnRanges is sorted by start in ascending order,
        and adjacent attn_ranges have no overlap

    NOTE:
        Generally, the naming convention for attn_ranges in this repo follows these rules:
        attn ranges naming qualifier convention:
            [DIST]_[SCOPE]_[PERM]_[ORDER]_[NAME]_ranges_[SUFFIX...]
                - DIST: host/remote/total - indicates whether ranges are on host or remote device
                    - host: ranges are on host
                    - remote: ranges are on remote device
                    - total: ranges are not on host or remote device, e.g. original reference ranges
                - SCOPE: global/local - indicates whether ranges use global or local indices
                    - global: ranges use global indices
                    - local: ranges use local indices
                - PERM: unperm/perm - indicates whether ranges have been permuted
                    - unperm: ranges have not been permuted
                    - perm: ranges have been permuted
                - ORDER: unordered/sorted/merged - indicates whether ranges are sorted or merged,
                    - unordered: ranges are not sorted
                    - sorted: ranges are sorted by start
                    - merged: ranges are merged
                - NAME: the name of the ranges
                - SUFFIX: per_rank/per_stage/etc - additional qualifiers for list[AttnRanges]

        Example:
            host_global_sorted_unperm_ranges_per_rank
            remote_local_merged_perm_ranges_per_stage
    """

    def __init__(self) -> None:
        self._ranges: list[AttnRange] = []
        self.local_range_cache: dict[int, tuple[int, AttnRange]] = dict()

    def is_valid(
        self,
    ) -> bool:
        if self.is_empty():  # empty ranges are always valid
            return True

        if not all(attn_range.is_valid_close() for attn_range in self._ranges):
            return False

        return True

    def check_valid(
        self,
    ) -> None:
        if not self.is_valid():
            raise ValueError(
                f"Some of the {self._ranges=} is invalid against the rule: 'start <= end'"
            )

    # NOTE: Inplace Operation (append, insert, extend, pop)
    def append(self, attn_range: AttnRange, check: bool = False) -> None:
        """Add the attn_range to the end"""
        if check:
            attn_range.check_valid()

        self.local_range_cache.clear()
        self._ranges.append(attn_range)

    def insert(self, idx: int, attn_range: AttnRange, check: bool = False) -> None:
        """Insert the attn_range to the 'idx'-th position,
        NOTE: if idx >= len(self._ranges), then use 'append' instead
        """
        if check:
            attn_range.check_valid()

        self.local_range_cache.clear()
        self._ranges.insert(idx, attn_range)

    def extend(self, attn_ranges: "AttnRanges", check: bool = False) -> None:
        if check:
            attn_ranges.check_valid()

        self.local_range_cache.clear()
        self._ranges.extend(attn_ranges._ranges)

    def pop(self, idx: int = -1) -> AttnRange:
        """Remove and return item at index (default last).

        Args:
            idx: The index of the element to remove. Default is -1 (last element).

        Returns:
            The removed AttnRange object.

        Raises:
            IndexError: If the list is empty or idx is out of range.
        """
        if self.is_empty():
            raise IndexError("pop from empty AttnRanges")

        self.local_range_cache.clear()
        return self._ranges.pop(idx)

    def clear_empty(self) -> "AttnRanges":
        non_empty_ranges = AttnRanges()
        for attn_range in self._ranges:
            if not attn_range.is_empty():
                non_empty_ranges.append(attn_range)

        return non_empty_ranges

    @nvtx.instrument_nvtx
    def sort(self) -> "AttnRanges":
        """
        Sort the attn_ranges by 'attn_range.start' in ascending order

        NOTE: Python's sort is stable, so when start values are the same,
        the original order will be preserved
        """

        return AttnRanges.from_ranges(
            sorted(self._ranges, key=lambda attn_range: attn_range.start)
        )

    @nvtx.instrument_nvtx
    def merge(self) -> "AttnRanges":
        """Merge the attn_ranges for the overlapped / tangent parts
        in ascending order by 'attn_range.start'
        """

        _ranges = self.sort()._ranges  # required to be sorted first

        _merged_ranges = AttnRanges()

        start, end = None, None
        for attn_range in _ranges:
            if start is None:
                start = attn_range.start
                end = attn_range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif attn_range.start > end:  # type: ignore[operator]
                start = attn_range.start
                end = attn_range.end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif attn_range.end > end:  # type: ignore[operator]
                end = attn_range.end
                _merged_ranges[-1].end = end

        return _merged_ranges

    @nvtx.instrument_nvtx
    def merge_with_split_alignment(self, split_alignment: int = 1) -> "AttnRanges":
        """Merge the attn_ranges for the overlapped / tangent parts with split alignment
        in ascending order by 'attn_range.start'

        Args:
            split_alignment (int): The alignment of the split, default is 1

        Returns:
            AttnRanges: The merged attn_ranges with split alignment, all start and end are aligned to the split_alignment
        """

        _ranges = self.sort()._ranges  # required to be sorted first

        _merged_ranges = AttnRanges()

        start, end = None, None
        for attn_range in _ranges:
            attn_range_start = attn_range.start // split_alignment * split_alignment
            attn_range_end = (
                (attn_range.end + split_alignment - 1)
                // split_alignment
                * split_alignment
            )
            if start is None:
                start, end = attn_range_start, attn_range_end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif attn_range_start > end:  # type: ignore[operator]
                start, end = attn_range_start, attn_range_end
                _merged_ranges.append(AttnRange(start=start, end=end))
            elif attn_range_end > end:  # type: ignore[operator]
                end = attn_range_end
                _merged_ranges[-1].end = end

        return _merged_ranges

    @nvtx.instrument_nvtx
    def chunk(self, chunk_size: int, check: bool = True) -> list["AttnRanges"]:
        if check:  # required to be non-overlap
            assert (
                self.is_non_overlap()
            ), "the ranges should be non-overlap if needed to be chunked"

        chunked_ranges_list = []
        chunked_ranges = AttnRanges()
        cnt = 0
        for attn_range in self._ranges:
            seqlen, start = attn_range.seqlen, attn_range.start
            new_cnt = cnt + seqlen
            while new_cnt >= chunk_size:
                seqlen_truc = chunk_size - cnt
                end = start + seqlen_truc
                chunked_ranges.append(AttnRange(start=start, end=end))
                chunked_ranges_list.append(chunked_ranges)

                chunked_ranges = AttnRanges()
                new_cnt -= chunk_size
                start = end
                cnt = 0
            cnt = new_cnt

            if cnt > 0:
                chunked_ranges.append(AttnRange(start=start, end=attn_range.end))

        if len(chunked_ranges) > 0:
            chunked_ranges_list.append(chunked_ranges)

        return chunked_ranges_list

    def truncate(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> "AttnRanges":
        trunc_ranges = AttnRanges()
        for attn_range in self._ranges:
            trunc_range = attn_range.truncate(start, end)
            if trunc_range.is_empty():
                # NOTE: skip the empty range, i.e. those beyond the truncate range
                continue
            trunc_ranges.append(trunc_range)

        return trunc_ranges

    def is_sorted(self) -> bool:
        """Whether the ranges are sorted by 'attn_range.start' in ascending order"""

        if not all(
            self._ranges[i - 1].start <= self._ranges[i].start
            for i in range(1, len(self._ranges))
        ):
            return False
        return True

    def is_merged(self) -> bool:
        """Whether the ranges are merged,
        which means:
            1. if the ranges are sorted by 'attn_range.start' in ascending order
            2. if any pair of the ranges have neither overlapped nor tangent parts
        """

        if self.is_sorted():
            if not all(
                self._ranges[i - 1].end < self._ranges[i].start
                for i in range(1, len(self._ranges))
            ):
                return False
            else:
                return True
        else:
            return False

    def is_non_overlap(self) -> bool:
        """Whether any pair of the ranges have overlapped parts"""

        return self.total_seqlen == self.merge().total_seqlen

    def is_cu_seqlens(self, seqlen: int) -> bool:
        if self.is_empty():
            return seqlen == 0

        if not self._ranges[0].start == 0:
            return False
        if not all(
            self._ranges[i - 1].end == self._ranges[i].start
            for i in range(1, len(self._ranges))
        ):
            return False
        if not self._ranges[-1].end == seqlen:
            return False

        return True

    def to_cu_seqlens(self, seq_len: int) -> list[int]:
        assert self.is_cu_seqlens(
            seq_len
        ), "The ranges can not be converted to cu_seqlens"
        if self.is_empty():
            return [0]
        return [0] + [attn_range.end for attn_range in self._ranges]

    @nvtx.instrument_nvtx
    def make_range_local(
        self,
        other_attn_range: AttnRange,
        is_self_merged: bool = False,
        prefix_offset: list[int] | None = None,
    ) -> tuple[AttnRange, AttnRange]:
        """
        Map the other_attn_range to the corresponding local_ranges within self_ranges,
        and return the position of other_attn_range in local_ranges
        (truncation is allowed via an optional positional parameter).

        Args:
            other_attn_range (AttnRange): The other_attn_range to be converted.
            is_self_merged (bool): Indicates whether self has been merged.
            prefix_offset (list[int] | None): If prefix_offset is None, it will be computed.

        Returns:
            tuple[AttnRange, AttnRange]:
                - local_range (AttnRange): The position of other_attn_range within self's local_ranges.
                    (May return an empty range if truncation occurs.)
                - target_range (AttnRange): The position of other_attn_range within global self._ranges.
                    It should cover the other_attn_range.
        """

        merged_ranges = self if is_self_merged else self.merge()

        if prefix_offset is None:
            prefix_offset = _calc_prefix_offset(merged_ranges)
        else:
            assert len(prefix_offset) == len(merged_ranges)

        le_idx = _binary_search(merged_ranges, other_attn_range)
        target_range: AttnRange = merged_ranges[le_idx]

        if other_attn_range.is_subrange_of(target_range):
            start = prefix_offset[le_idx] + other_attn_range.start - target_range.start
            local_range = AttnRange(start=start, end=start + other_attn_range.seqlen)
            return local_range, target_range
        else:
            raise ValueError(
                f"The attn_range {other_attn_range} is not in the (even merged) attn_ranges {merged_ranges}"
            )

    @nvtx.instrument_nvtx
    @lru_cache(maxsize=65536)
    def make_ranges_local(
        self,
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
    ) -> "AttnRanges":
        """
        Maps each attn_range in other_attn_ranges to the local ranges of self,
        and returns the position of each attn_range in self's local ranges.

        Args:
            ranges(AttnRanges): The ranges to be converted, must be merged ranges
            is_self_merged(bool): Whether self is already merged

        Returns:
            local_ranges(AttnRanges): The position of each attn_range in the reference local ranges
                                     (may contain empty ranges if truncation occurs)

        Complexity:
            assume len(self) = m, len(other_attn_ranges) = n
            then the complexity is O(m + n * log(m))
        """
        local_ranges = AttnRanges()

        merged_ranges = self if is_self_merged else self.merge()

        if len(self.local_range_cache) == 0 and len(merged_ranges) > 0:
            # make the class frozen to prohibit modifications to it.
            local_index = 0
            # init local_cache
            for range in merged_ranges:
                self.local_range_cache[range.start] = (local_index, range)
                local_index += range.seqlen
                self.local_range_cache[range.end] = (local_index, range)

        prefix_offset = _calc_prefix_offset(merged_ranges)

        for attn_range in other_attn_ranges:
            if self.local_range_cache.get(attn_range.start, None) is not None:
                # hit the left endpoint and calculate the value of the right endpoint.
                start, target_range = self.local_range_cache.get(attn_range.start)  # type: ignore[misc]
                assert attn_range.is_subrange_of(target_range), (
                    f"in make_ranges_local function, target range should cover attn_range, "
                    f"but got {target_range=} and {attn_range=}"
                )
                local_range = AttnRange(
                    start=start,
                    end=start + attn_range.seqlen,
                )
                self.local_range_cache[attn_range.end] = (
                    start + attn_range.seqlen,
                    target_range,
                )
            elif self.local_range_cache.get(attn_range.end, None) is not None:
                # hit the right endpoint and calculate the value of the left endpoint.
                end, target_range = self.local_range_cache.get(attn_range.end)  # type: ignore[misc]
                assert attn_range.is_subrange_of(target_range), (
                    f"in make_ranges_local function, target range should cover attn_range, "
                    f"but got {target_range=} and {attn_range=}"
                )
                local_range = AttnRange(
                    start=end - attn_range.seqlen,
                    end=end,
                )
                self.local_range_cache[attn_range.start] = (
                    end - attn_range.seqlen,
                    target_range,
                )
            else:
                # neither endpoint is hit, call make_range_local to calculate.
                local_range, target_range = merged_ranges.make_range_local(
                    attn_range,
                    is_self_merged=True,
                    prefix_offset=prefix_offset,
                )
                self.local_range_cache[attn_range.start] = (
                    local_range.start,
                    target_range,
                )
                self.local_range_cache[attn_range.end] = (local_range.end, target_range)
            local_ranges.append(local_range)

        return local_ranges

    @nvtx.instrument_nvtx
    def find_hole_ranges(
        self,
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> "AttnRanges":
        """
        Returns the result of self - other_attn_ranges
        NOTE: The '-' here is the set difference, so the returned hole_ranges contains
              ranges that are in self but not in other_attn_ranges

        Args:
            other_attn_ranges(AttnRanges): The ranges to be subtracted

        Returns:
            NOTE: hole_ranges is merged
            hole_ranges(AttnRanges): The result of self - other_attn_ranges

        Example::
            self = [[0, 10), [15, 20), [20, 30)]
            other_attn_ranges = [[5, 10), [25, 30)]
            return [[0, 5), [15, 25)]
        """

        ranges1 = self if is_self_merged else self.merge()
        ranges2 = other_attn_ranges if is_other_merged else other_attn_ranges.merge()

        p1 = 0
        p2 = 0

        hole_ranges = AttnRanges()

        def get_hole_range(r1: AttnRange, r2: AttnRange) -> AttnRange:
            return AttnRange(start=r1.start, end=min(r1.end, r2.start))

        while p1 < len(ranges1) and p2 < len(ranges2):
            r1: AttnRange = ranges1[p1]
            r2: AttnRange = ranges2[p2]

            if r1.end > r2.end:
                p2 += 1
            else:
                p1 += 1

            if r1.start < r2.start:
                hole_ranges.append(get_hole_range(r1, r2))

            if r1.start < r2.end:
                try:
                    r1.start = r2.end
                except RangeError:
                    pass

        hole_ranges.extend(ranges1[p1:])

        return hole_ranges

    @nvtx.instrument_nvtx
    @lru_cache(maxsize=65536)
    def find_overlap_ranges(
        self: "AttnRanges",
        other_attn_ranges: "AttnRanges",
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> "AttnRanges":
        """
        Returns the intersection of self and other_attn_ranges

        Args:
            other_attn_ranges(AttnRanges): The ranges to find intersection with

        Returns:
            NOTE: overlap_ranges is guaranteed to be merged
            overlap_ranges(AttnRanges): The intersection of self and other_attn_ranges

        Example::
            self = [[0, 10), [15, 20), [25, 30)]
            other_attn_ranges = [[5, 10), [18, 30)]
            return [[5, 10), [18, 20), [25, 30)]
        """

        ranges1 = self if is_self_merged else self.merge()
        ranges2 = other_attn_ranges if is_other_merged else other_attn_ranges.merge()

        p1 = 0
        p2 = 0

        overlap_ranges = AttnRanges()

        while p1 < len(ranges1) and p2 < len(ranges2):
            r1: AttnRange = ranges1[p1]
            r2: AttnRange = ranges2[p2]

            if r1.end > r2.end:
                p2 += 1
            else:
                p1 += 1

            if r1.is_overlap_with(r2):
                overlap_ranges.append(r1.intersect(r2))

        return overlap_ranges

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        if self.is_empty():
            return torch.empty([0, 2], dtype=torch.int32, device=device)
        else:
            return torch.tensor(
                self.to_naive_ranges(), dtype=torch.int32, device=device
            )

    @staticmethod
    def from_cu_seqlens(
        cu_seqlens: list[int],
        seq_len: int,
    ) -> "AttnRanges":
        check_valid_cu_seqlens(cu_seqlens, seq_len)

        ranges = AttnRanges()

        for i in range(1, len(cu_seqlens)):
            ranges.append(AttnRange(cu_seqlens[i - 1], cu_seqlens[i]))

        return ranges

    def clone(self) -> "AttnRanges":
        """Clone the current AttnRanges efficiently"""
        new_ranges = AttnRanges()
        new_ranges._ranges = [r.clone() for r in self._ranges]
        return new_ranges

    @staticmethod
    def from_ranges(
        ranges: Union[NaiveRanges, list[AttnRange], "AttnRanges"],
        check: bool = False,
    ) -> "AttnRanges":
        if isinstance(ranges, AttnRanges):
            attn_ranges = ranges.clone()
        else:
            attn_ranges = AttnRanges()
            _ranges = [AttnRange.from_range(attn_range) for attn_range in ranges]
            attn_ranges._ranges = _ranges

        if check:
            attn_ranges.check_valid()

        return attn_ranges

    def to_naive_ranges(self) -> NaiveRanges:
        return [attn_range.to_naive_range() for attn_range in self._ranges]

    def intersect_size(self) -> int:
        """Calculate the total size of overlapping parts between all attn_ranges

        Uses a sweep line algorithm to calculate the total overlap size between all interval pairs.
        Time complexity is O(n log n), where n is the number of intervals.

        Returns:
            int: Total size of overlapping parts between all interval pairs
        """
        if self.is_empty() or len(self) == 1:
            return 0

        # Use sweep line algorithm to calculate overlapping regions
        # 1. Collect all start and end points of intervals
        events = []
        for r in self._ranges:
            events.append((r.start, 1))  # start point
            events.append((r.end, -1))  # end point

        # 2. Sort by position
        events.sort()

        # 3. Scan and calculate overlapping regions
        count = 0  # number of intervals covering current position
        last_pos = 0  # previous position
        overlap_size = 0  # total size of overlapping regions

        for pos, event_type in events:
            # If more than 1 interval, calculate overlap
            if count > 1:
                # Calculate the region [last_pos, pos) covered by multiple intervals
                # count-1 represents the number of overlaps
                overlap_size += (count - 1) * (pos - last_pos)

            # Update count and position
            count += event_type
            last_pos = pos

        return overlap_size

    def intersect_size_with(self, other: "AttnRanges") -> int:
        intersec_ranges = AttnRanges()
        total_ranges = AttnRanges()
        # HACK: directly modify _ranges attr to improve performance
        total_ranges._ranges = self._ranges + other._ranges

        non_overlap_ranges = total_ranges.merge().find_hole_ranges(
            self.find_overlap_ranges(other)
        )
        for r in total_ranges:
            ranges_worker = AttnRanges()
            ranges_worker.append(r)
            intersec_ranges.extend(ranges_worker.find_hole_ranges(non_overlap_ranges))

        return intersec_ranges.intersect_size()

    def union_size(self) -> int:
        return self.total_seqlen

    def union_size_with(self, other: "AttnRanges") -> int:
        return self.total_seqlen + other.total_seqlen

    @property
    def total_seqlen(self) -> int:
        """The total seqlen this ranges represent
        which is equal to the sum of the size of each range
        NOTE: distinguish with the property 'self.max_seqlen'
        """
        return sum(attn_range.seqlen for attn_range in self._ranges)

    @property
    def max_seqlen(self) -> int:
        """The maximum seqlen this ranges represent
        which is equal to the maximum size of each range
        NOTE: distinguish with the property 'self.total_seqlen'
        """
        if self.is_empty():
            return 0
        return max(attn_range.seqlen for attn_range in self._ranges)

    @property
    def start(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no start")
        return min(attn_range.start for attn_range in self._ranges)

    @property
    def end(self) -> int:
        if self.is_empty():
            raise ValueError("The ranges is empty, there is no end")
        return max(attn_range.end for attn_range in self._ranges)

    @property
    def size(self) -> int:
        return len(self._ranges)

    @property
    def points(self) -> list[int]:
        """The axis points covered by this ranges
        in ascending order and without duplicates
        """

        _points = set()
        for r in self._ranges:
            _points.add(r.start)
            _points.add(r.end)

        return sorted(list(_points))

    def is_empty(self) -> bool:
        return len(self._ranges) == 0

    def __len__(self) -> int:
        return len(self._ranges)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub_attn_ranges = AttnRanges()
            for attn_range in self._ranges[idx]:
                sub_attn_ranges.append(attn_range)
            return sub_attn_ranges

        return self._ranges[idx]

    def __setitem__(self, idx: int | slice, value: Union[AttnRange, "AttnRanges"]):
        if isinstance(idx, slice):
            assert isinstance(value, AttnRanges) and idx.stop - idx.start == len(value)
            self._ranges[idx] = value._ranges
        else:
            assert isinstance(value, AttnRange)
            self._ranges[idx] = value

    def __iter__(self) -> Iterator[AttnRange]:
        return iter(self._ranges)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRanges):
            return self._ranges == other._ranges
        return False

    def __hash__(self) -> int:
        return hash(tuple(self._ranges))

    def __repr__(self) -> str:  # pragma: no cover
        if self.is_empty():  # to prevent repr as "[]" to mix up with empty list
            return "[[,)]"
        return f"{self._ranges}"


if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRanges as _AttnRanges

        AttnRanges = _AttnRanges  # type: ignore[misc, assignment] # noqa: F811
    except ImportError:
        pass


RangesType: TypeAlias = AttnRanges | NaiveRanges
