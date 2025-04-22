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

from dataclasses import dataclass

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges


@dataclass(repr=False)
class AttnSlice:
    slice_id: int | None = None

    mask_type: AttnMaskType | None = None

    q_range: AttnRange | None = None
    k_range: AttnRange | None = None

    _area: int | None = None

    # HACK: The logic for calculating the area will be encapsulated in AttnSlice later, and the area will be read-only.
    # This feature of directly setting the area is retained here for now.

    @property
    def area(self) -> int:
        if self._area is None:
            if self.mask_type is AttnMaskType.FULL:
                # just the area of a full rectangle mask
                self._area = self.q_range.seqlen * self.k_range.seqlen  # type: ignore
            elif self.mask_type is AttnMaskType.CAUSAL:
                if self.k_range.seqlen > self.q_range.seqlen:  # type: ignore
                    # the area of a trapezoid
                    self._area = (
                        (2 * self.k_range.seqlen - self.q_range.seqlen + 1)  # type: ignore
                        * self.q_range.seqlen  # type: ignore
                        // 2
                    )
                else:  # the area of a triangle
                    self._area = (1 + self.k_range.seqlen) * self.k_range.seqlen // 2  # type: ignore
            else:
                raise ValueError(
                    f"Only support full or causal mask, but got {self.mask_type}."
                )

        return self._area

    @area.setter
    def area(self, area: int):
        self._area = area

    def iou_with(self, other: "AttnSlice") -> float:
        return self.k_range.intersect_size(other.k_range) / self.k_range.union_size(  # type: ignore
            other.k_range  # type: ignore
        )

    def __post_init__(self):
        pass

    def __repr__(self) -> str:
        return (
            f"AttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_range={self.k_range}, "
            f"mask_type={self.mask_type}, area={self.area})"
        )


@dataclass(repr=False)
class MultiKAttnSlice:
    q_range: AttnRange
    k_ranges: AttnRanges
    mask_types: list[AttnMaskType]

    slice_id: int | None = None
    _area: int | None = None

    @property
    def area(self) -> int:
        if self._area is None:
            self._area = 0
            for k_range, mask_type in zip(self.k_ranges._ranges, self.mask_types):
                if mask_type is AttnMaskType.FULL:
                    # just the area of a full rectangle mask
                    self._area += self.q_range.seqlen * k_range.seqlen
                elif mask_type is AttnMaskType.CAUSAL:
                    if k_range.seqlen > self.q_range.seqlen:  # the area of a trapezoid
                        self._area += (
                            (2 * k_range.seqlen - self.q_range.seqlen + 1)
                            * self.q_range.seqlen
                            // 2
                        )
                    else:  # the area of a triangle
                        self._area += (1 + k_range.seqlen) * k_range.seqlen // 2
                else:
                    raise ValueError(
                        f"Only support full or causal mask, but got {mask_type} in mask_types."
                    )

        return self._area

    @area.setter
    def area(self, area: int):
        self._area = area

    def __post_init__(self):
        assert len(self.mask_types) == len(self.k_ranges), (
            f"The length of mask_types and k_ranges should be the same, "
            f"but got {len(self.mask_types)} and {len(self.k_ranges)}"
        )

    def __repr__(self) -> str:
        return (
            f"MultiKAttnSlice(slice_id={self.slice_id}, "
            f"q_range={self.q_range}, k_ranges={self.k_ranges}, "
            f"mask_types={self.mask_types}, "
            f"area={self.area})"
        )
