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
            raise ValueError(f"Some of the {self._rects=} is invalid")

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

    @staticmethod
    def from_ranges(
        q_ranges: Union[
            NaiveRanges,
            list[AttnRange],
            AttnRanges,
        ],
        k_ranges: Union[
            NaiveRanges,
            list[AttnRange],
            AttnRanges,
        ],
        mask_types: Union[list[int], list[AttnMaskType]],
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

        assert (
            len(attn_q_ranges) == len(attn_k_ranges) == len(attn_mask_type)
        ), "q_ranges, k_ranges, mask_types length should be equal"

        attn_rects = AttnRectangles()
        _rects = []
        for q_range, k_range, mask_type in zip(
            attn_q_ranges, attn_k_ranges, attn_mask_type
        ):
            # remove empty ranges
            if q_range.is_empty() or k_range.is_empty():
                continue
            # remove bi_causal invalid mask area
            if mask_type is AttnMaskType.BICAUSAL and q_range.seqlen > k_range.seqlen:
                continue

            _rects.append(
                AttnRectangle(
                    q_range=q_range,
                    k_range=k_range,
                    mask_type=mask_type,
                )
            )

        attn_rects._rects = _rects

        if check:
            attn_rects.check_valid()

        return attn_rects

    def get_qo_ranges_union(self) -> AttnRanges:
        qo_ranges = AttnRanges()
        for rect in self._rects:
            qo_ranges.append(rect.q_range)
        return qo_ranges.merge()

    def get_kv_ranges_union(self) -> AttnRanges:
        kv_ranges = AttnRanges()
        for rect in self._rects:
            kv_ranges.append(rect.k_range)
        return kv_ranges.merge()

    def total_seqlen_qo(self) -> int:
        return self.get_qo_ranges_union().total_seqlen

    def total_seqlen_kv(self) -> int:
        return self.get_kv_ranges_union().total_seqlen

    def cut_q(self, cut_pos: int) -> tuple["AttnRectangles", "AttnRectangles"]:
        rects_left = AttnRectangles()
        rects_right = AttnRectangles()
        for rect in self._rects:
            rect_left, rect_right = rect.cut_q(cut_pos=cut_pos)
            if rect_left is not None:
                rects_left.append(rect_left)
            if rect_right is not None:
                rects_right.append(rect_right)

        return rects_left, rects_right

    def cut_k(self, cut_pos: int) -> tuple["AttnRectangles", "AttnRectangles"]:
        rects_left = AttnRectangles()
        rects_right = AttnRectangles()
        for rect in self._rects:
            rect_left, rect_right = rect.cut_k(cut_pos=cut_pos)
            if rect_left is not None:
                rects_left.append(rect_left)
            if rect_right is not None:
                rects_right.append(rect_right)

        return rects_left, rects_right

    def get_rects_within_q_segment(
        self,
        q_start: int,
        q_end: int,
    ) -> "AttnRectangles":
        rects_in_seg = AttnRectangles()
        for rect in self._rects:
            rect_in_seg = rect.get_rect_within_q_segment(q_start, q_end)
            if rect_in_seg is not None:
                rects_in_seg.append(rect_in_seg)

        return rects_in_seg

    def get_rects_within_k_segment(
        self,
        k_start: int,
        k_end: int,
    ) -> "AttnRectangles":
        rects_in_seg = AttnRectangles()
        for rect in self._rects:
            rect_in_seg = rect.get_rect_within_k_segment(k_start, k_end)
            if rect_in_seg is not None:
                rects_in_seg.append(rect_in_seg)

        return rects_in_seg

    def area(self) -> int:
        total_area = 0
        for rect in self._rects:
            total_area += rect.area()
        return total_area

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
