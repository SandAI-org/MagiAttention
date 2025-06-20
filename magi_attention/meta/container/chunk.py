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

from dataclasses import dataclass, field

from magi_attention.common.ranges import AttnRanges

from .slice import AttnSlice


@dataclass(repr=False)
class AttnChunk:
    chunk_id: int | None = None

    q_slices: list[AttnSlice] = field(default_factory=list)

    sample_ids: list[int] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges()
        for slice in self.q_slices:
            q_ranges.append(slice.q_range, check=False)  # type: ignore
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges()
        for slice in self.q_slices:
            k_ranges.append(slice.k_range, check=False)  # type: ignore
        return k_ranges

    @property
    def attn_slices(self) -> list[AttnSlice]:
        return self.q_slices

    @property
    def area(self) -> int:
        return sum(slice.area for slice in self.q_slices)

    @property
    def iou(self) -> float:
        intersect_size = self.k_ranges.intersect_size()
        union_size = self.k_ranges.union_size()
        if union_size == 0:
            return 0.0
        return intersect_size / union_size

    def iou_with(self, other: "AttnChunk") -> float:
        return self.k_ranges.intersect_size_with(
            other.k_ranges
        ) / self.k_ranges.union_size_with(other.k_ranges)

    def __repr__(self, indent: str = "") -> str:
        repr_str = f"{indent}AttnChunk(chunk_id={self.chunk_id}, area={self.area}, q_slices=[\n"
        for i, slice in enumerate(self.q_slices):
            repr_str += (
                f"{indent}    └── {slice}\n"
                if i == len(self.q_slices) - 1
                else f"{indent}    ├── {slice}\n"
            )
        repr_str += f"{indent}])"
        return repr_str
