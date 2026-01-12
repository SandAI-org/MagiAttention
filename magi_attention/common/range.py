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

from typing import Any, TypeAlias, Union

NaiveRange: TypeAlias = tuple[int, int] | list[int]


class RangeError(Exception):
    pass


class AttnRange:
    """A dataclass to manage any indices range for attention computation"""

    def __init__(self, start: int, end: int) -> None:
        self.check_valid(start=start, end=end)

        self._start = start
        self._end = end

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, value) -> None:
        self._start = value

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, value) -> None:
        self._end = value

    @property
    def seqlen(self) -> int:
        """The length of this range in the axis"""

        return self._end - self._start

    def to_naive_range(self) -> NaiveRange:
        return (self._start, self._end)

    @staticmethod
    def from_range(
        attn_range: Union[NaiveRange, list[int], "AttnRange"],
        check: bool = False,
    ) -> "AttnRange":
        if isinstance(attn_range, AttnRange):
            res = AttnRange(attn_range.start, attn_range.end)
        else:
            res = AttnRange(start=attn_range[0], end=attn_range[1])

        if check:
            res.check_valid()

        return res

    def clone(self) -> "AttnRange":
        return AttnRange(self._start, self._end)

    def offset(self, offset: int) -> "AttnRange":
        return AttnRange(start=self._start + offset, end=self._end + offset)

    def truncate(self, start: int | None = None, end: int | None = None) -> "AttnRange":
        start = self._start if start is None else max(self._start, start)
        end = self._end if end is None else min(self._end, end)

        # NOTE: if start > end, then return empty range: [start, start)
        return AttnRange(start=start, end=max(start, end))

    def intersect(self, other: "AttnRange") -> "AttnRange":
        start = max(self._start, other._start)
        end = min(self._end, other._end)

        return AttnRange(start=min(start, end), end=end)

    def intersect_size(self, other: "AttnRange") -> int:
        return self.intersect(other).seqlen

    def union(self, other: "AttnRange") -> list["AttnRange"]:
        # REVIEW: Is this interface correct?
        if self.is_empty() or other.is_empty():
            return [self, other]

        if self.is_subrange_of(other):
            return [other]

        if other.is_subrange_of(self):
            return [self]

        if self.is_overlap_with(other):
            union_range = AttnRange(
                start=min(self._start, other._start), end=max(self._end, other._end)
            )
            return [union_range]

        return [self, other]

    def union_size(self, other: "AttnRange") -> int:
        return sum(r.seqlen for r in self.union(other))

    def diff_by(self, other: "AttnRange") -> list["AttnRange"]:
        """other - self"""
        diff_ranges = []

        inter_range = self.intersect(other)

        if inter_range == self:  # self is a subrange of other
            diff_ranges.append(AttnRange(other.start, self.start))
            diff_ranges.append(AttnRange(self.end, other.end))
        elif inter_range == other:  # k_range is a subrange of q_range
            diff_ranges.append(AttnRange(other.start, other.start))
        elif inter_range.is_empty():  # q_range and k_range are disjoint
            diff_ranges.append(AttnRange.from_range(other))
        else:  # q_range and k_range are overlapping, but neither of them cover the other
            if other.start < self.start:
                diff_ranges.append(AttnRange(other.start, self.start))
            else:
                diff_ranges.append(AttnRange(self.end, other.end))

        diff_ranges = [
            diff_range for diff_range in diff_ranges if not diff_range.is_empty()
        ]

        return diff_ranges

    def is_subrange_of(self, other: "AttnRange") -> bool:
        return self._start >= other._start and self._end <= other._end

    def is_overlap_with(self, other: "AttnRange") -> bool:
        return not (self._start >= other._end or self._end <= other._start)

    def is_empty(self) -> bool:
        return self._start == self._end

    # range is [start, end] closed interval
    def is_valid_close(self, start: int | None = None, end: int | None = None) -> bool:
        start = self._start if start is None else start
        end = self._end if end is None else end

        return start <= end

    # range is [start, end) Left-closed and right-open intervals
    def is_valid_open(self, start: int | None = None, end: int | None = None) -> bool:
        start = self._start if start is None else start
        end = self._end if end is None else end

        return start < end

    def check_valid(self, start: int | None = None, end: int | None = None) -> None:
        if not self.is_valid_close(start, end):
            raise RangeError(
                f"The attn_range {(start, end)} is invalid against the rule: 'start <= end'"
            )

    def __len__(self) -> int:
        return self.seqlen

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRange):
            return self._start == other._start and self._end == other._end
        return False

    def __hash__(self) -> int:
        return hash((self._start, self._end))

    def __repr__(self) -> str:
        return f"[{self._start}, {self._end})"


from magi_attention import is_cpp_backend_enable  # noqa: E402

if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRange as _AttnRange

        AttnRange = _AttnRange  # type: ignore[misc, assignment] # noqa: F811
    except ImportError:
        pass


RangeType: TypeAlias = AttnRange | NaiveRange
