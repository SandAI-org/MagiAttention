# Copyright (c) 2025 SandAI. All Rights Reserved.
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

from typing import Any, Iterator, Union

from .range import AttnRange
from .ranges import AttnRanges, NaiveRanges
from .rect_range import AttnRectRange


class AttnRectRanges:
    """
    A dataclass to manage a list of 'AttnRectRange' objects for attention computation
    """

    def __init__(self) -> None:
        self._ranges: list[AttnRectRange] = []

    def is_valid(
        self,
    ) -> bool:
        if self.is_empty():  # empty ranges are always valid
            return True

        if not all(attn_range.is_valid() for attn_range in self._ranges):
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
    def append(self, attn_range: AttnRectRange, check: bool = False) -> None:
        """Add the attn_rect_range to the end"""
        if check:
            attn_range.check_valid()

        self._ranges.append(attn_range)

    @staticmethod
    def from_ranges(
        ranges: Union[
            NaiveRanges,
            list[AttnRectRange],
            list[AttnRange],
            AttnRanges,
            "AttnRectRanges",
        ],
        check: bool = False,
    ) -> "AttnRectRanges":
        if isinstance(ranges, AttnRectRanges):  # just copy
            attn_ranges = ranges
        else:
            attn_ranges = AttnRectRanges()
            _ranges = [AttnRectRange.from_range(attn_range) for attn_range in ranges]
            attn_ranges._ranges = _ranges

        if check:
            attn_ranges.check_valid()

        return attn_ranges

    def to_naive_ranges(self) -> NaiveRanges:
        return [attn_range.to_naive_range() for attn_range in self._ranges]

    def is_empty(self) -> bool:
        return len(self._ranges) == 0

    def __len__(self) -> int:
        return len(self._ranges)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub_attn_ranges = AttnRectRanges()
            for attn_range in self._ranges[idx]:
                sub_attn_ranges.append(attn_range)
            return sub_attn_ranges

        return self._ranges[idx]

    def __setitem__(
        self, idx: int | slice, value: Union[AttnRectRange, "AttnRectRanges"]
    ):
        if isinstance(idx, slice):
            assert isinstance(value, AttnRectRanges) and idx.stop - idx.start == len(
                value
            )
            self._ranges[idx] = value._ranges
        else:
            assert isinstance(value, AttnRectRange)
            self._ranges[idx] = value

    def __iter__(self) -> Iterator[AttnRectRange]:
        return iter(self._ranges)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRectRanges):
            return self._ranges == other._ranges
        return False

    def __hash__(self) -> int:
        return hash(tuple(self._ranges))

    def __repr__(self) -> str:
        if self.is_empty():  # to prevent repr as "[]" to mix up with empty list
            return "[[,)]"
        return f"{self._ranges}"
