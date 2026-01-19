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

from dataclasses import dataclass, field

from magi_attention.common.ranges import AttnRanges

from .chunk import AttnChunk
from .slice import AttnSlice


@dataclass(repr=False)
class AttnBucket:
    cp_rank: int | None = None

    q_chunks: list[AttnChunk] = field(default_factory=list)

    @property
    def q_ranges(self) -> AttnRanges:
        q_ranges = AttnRanges()
        for chunk in self.q_chunks:
            q_ranges.extend(chunk.q_ranges, check=False)
        return q_ranges

    @property
    def k_ranges(self) -> AttnRanges:
        k_ranges = AttnRanges()
        for chunk in self.q_chunks:
            k_ranges.extend(chunk.k_ranges, check=False)
        return k_ranges

    @property
    def attn_slices(self) -> list[AttnSlice]:
        _attn_slices = []
        for chunk in self.q_chunks:
            _attn_slices.extend(chunk.attn_slices)
        return _attn_slices

    @property
    def area(self) -> int:
        return sum(chunk.area for chunk in self.q_chunks)

    @property
    def areas(self) -> list[int]:
        return [chunk.area for chunk in self.q_chunks]

    @property
    def iou(self) -> float:
        intersect_size = self.k_ranges.intersect_size()
        union_size = self.k_ranges.union_size()
        if union_size == 0:
            return 0.0
        return intersect_size / union_size

    def iou_with(self, other: "AttnBucket") -> float:
        return self.k_ranges.intersect_size_with(
            other.k_ranges
        ) / self.k_ranges.union_size_with(other.k_ranges)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AttnBucket):
            return False
        return self.cp_rank == other.cp_rank and self.q_chunks == other.q_chunks

    def __repr__(self, indent: str = "") -> str:
        repr_str = (
            f"{indent}AttnBucket(cp_rank={self.cp_rank}, area={self.area}, q_chunks=[\n"
        )
        for i, chunk in enumerate(self.q_chunks):
            repr_str += (
                f"{indent}    └── {chunk.__repr__(indent + '    ')}\n"
                if i == len(self.q_chunks) - 1
                else f"{indent}    ├── {chunk.__repr__(indent + '    ')}\n"
            )
        repr_str += f"{indent}])"
        return repr_str
