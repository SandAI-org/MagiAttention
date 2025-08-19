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

from typing import Any, Iterator, Sequence, TypeAlias, Union

from .enum import AttnMaskType
from .range import AttnRange, NaiveRange
from .ranges import AttnRanges
from .rectangle import AttnRectangle

NaiveRanges: TypeAlias = Sequence[NaiveRange]

__all__ = [
    "AttnRectangles",
]


class AttnRectangles:
    """
    A dataclass to manage a list of 'AttnRectangle' objects for attention computation
    """

    def __init__(self) -> None:
        self._rects: list[AttnRectangle] = []

    def is_valid(
        self,
    ) -> bool:
        if self.is_empty():  # empty rects are always valid
            return True

        if not all(rect.is_valid() for rect in self._rects):
            return False

        return True

    def check_valid(
        self,
    ) -> None:
        if not self.is_valid():
            raise ValueError(
                f"Some of the {self._rects=} is invalid against the rule: '0 <= start <= end'"
            )

    # NOTE: Inplace Operation (append, insert, extend, pop)
    def append(self, attn_rect: AttnRectangle, check: bool = False) -> None:
        """Add the attn_rect to the end"""
        if check:
            attn_rect.check_valid()

        self._rects.append(attn_rect)

    def extend(self, attn_rects: "AttnRectangles", check: bool = False) -> None:
        if check:
            attn_rects.check_valid()

        self._rects.extend(attn_rects._rects)

    # def clear_empty(self) -> "AttnRectangles":
    #     pass

    # @nvtx.instrument_nvtx
    # def sort(self) -> "AttnRectangles":
    #     pass

    # @nvtx.instrument_nvtx
    # def merge(self) -> "AttnRectangles":
    #     """Merge the attn_ranges for the overlapped / tangent parts
    #     in ascending order by 'attn_range.start'
    #     """

    #     _ranges = self.sort()._ranges  # required to be sorted first

    #     _merged_ranges = AttnRanges()

    #     start, end = None, None
    #     for attn_range in _ranges:
    #         if start is None:
    #             start = attn_range.start
    #             end = attn_range.end
    #             _merged_ranges.append(AttnRange(start=start, end=end))
    #         elif attn_range.start > end:  # type: ignore[operator]
    #             start = attn_range.start
    #             end = attn_range.end
    #             _merged_ranges.append(AttnRange(start=start, end=end))
    #         elif attn_range.end > end:  # type: ignore[operator]
    #             end = attn_range.end
    #             _merged_ranges[-1].end = end

    #     return _merged_ranges

    # @nvtx.instrument_nvtx
    # def chunk(self, chunk_size: int, check: bool = True) -> list["AttnRanges"]:
    #     if check:  # required to be non-overlap
    #         assert (
    #             self.is_non_overlap()
    #         ), "the ranges should be non-overlap if needed to be chunked"

    #     chunked_ranges_list = []
    #     chunked_ranges = AttnRanges()
    #     cnt = 0
    #     for attn_range in self._ranges:
    #         seqlen, start = attn_range.seqlen, attn_range.start
    #         new_cnt = cnt + seqlen
    #         while new_cnt >= chunk_size:
    #             seqlen_truc = chunk_size - cnt
    #             end = start + seqlen_truc
    #             chunked_ranges.append(AttnRange(start=start, end=end))
    #             chunked_ranges_list.append(chunked_ranges)

    #             chunked_ranges = AttnRanges()
    #             new_cnt -= chunk_size
    #             start = end
    #             cnt = 0
    #         cnt = new_cnt

    #         if cnt > 0:
    #             chunked_ranges.append(AttnRange(start=start, end=attn_range.end))

    #     if len(chunked_ranges) > 0:
    #         chunked_ranges_list.append(chunked_ranges)

    #     return chunked_ranges_list

    # def is_sorted(self) -> bool:
    #     """Whether the ranges are sorted by 'attn_range.start' in ascending order"""

    #     if not all(
    #         self._ranges[i - 1].start <= self._ranges[i].start
    #         for i in range(1, len(self._ranges))
    #     ):
    #         return False
    #     return True

    # def is_merged(self) -> bool:
    #     """Whether the ranges are merged,
    #     which means:
    #         1. if the ranges are sorted by 'attn_range.start' in ascending order
    #         2. if any pair of the ranges have neither overlapped nor tangent parts
    #     """

    #     if self.is_sorted():
    #         if not all(
    #             self._ranges[i - 1].end < self._ranges[i].start
    #             for i in range(1, len(self._ranges))
    #         ):
    #             return False
    #         else:
    #             return True
    #     else:
    #         return False

    @staticmethod
    def from_ranges(
        q_ranges: Union[NaiveRanges, list[AttnRange], AttnRanges],
        k_ranges: Union[NaiveRanges, list[AttnRange], AttnRanges],
        mask_types: Union[list[int | AttnMaskType]],
        check: bool = False,
    ) -> "AttnRectangles":
        attn_q_ranges = AttnRanges.from_ranges(q_ranges, check)
        attn_k_ranges = AttnRanges.from_ranges(k_ranges, check)
        attn_mask_type = [
            {
                0: AttnMaskType.FULL,
                1: AttnMaskType.CAUSAL,
                2: AttnMaskType.INVCAUSAL,
                3: AttnMaskType.BICAUSAL,
            }[i]
            if isinstance(i, int)
            else i
            for i in mask_types
        ]

        if len(attn_q_ranges) != len(attn_k_ranges) or len(attn_q_ranges) != len(
            attn_mask_type
        ):
            raise ValueError("q_ranges, k_ranges, mask_types length should be equal")

        rects_len = len(attn_mask_type)
        attn_rects = AttnRectangles()
        _rects = [
            AttnRectangle(
                q_range=attn_q_ranges[i],
                k_range=attn_k_ranges[i],
                mask_type=attn_mask_type[i],
            )
            for i in range(rects_len)
        ]
        attn_rects._rects = _rects

        if check:
            attn_rects.check_valid()

        return attn_rects

    @property
    def size(self) -> int:
        return len(self._rects)

    def is_empty(self) -> bool:
        return len(self._rects) == 0

    def __len__(self) -> int:
        return len(self._rects)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            sub_attn_ranges = AttnRectangles()
            for attn_range in self._rects[idx]:
                sub_attn_ranges.append(attn_range)
            return sub_attn_ranges

        return self._rects[idx]

    def __setitem__(
        self, idx: int | slice, value: Union[AttnRectangle, "AttnRectangles"]
    ):
        if isinstance(idx, slice):
            assert isinstance(value, AttnRectangles) and idx.stop - idx.start == len(
                value
            )
            self._rects[idx] = value._rects
        else:
            assert isinstance(value, AttnRectangle)
            self._rects[idx] = value

    def __iter__(self) -> Iterator[AttnRectangle]:
        return iter(self._rects)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRectangles):
            return self._rects == other._rects
        return False

    def __hash__(self) -> int:
        return hash(tuple(self._rects))

    def __repr__(self) -> str:
        if self.is_empty():
            return "[-1, -1) x [-1, -1): None"
        return f"{self._rects}"
