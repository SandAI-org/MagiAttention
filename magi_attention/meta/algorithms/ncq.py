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

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class NCQDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The non-communication-qo dynamic dispatch algorithm implementation"""

    def __init__(self, debug_print: bool = False):
        self.debug_print = debug_print

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.NON_COMMUNICATION_QO

    def solve(
        self,
        rects: AttnRectangles,
        host_ranges_q: list[AttnRanges],
        host_ranges_k: list[AttnRanges],
        num_heads_q: int,
        num_heads_kv: int,
        num_heads_group: int,
        bucket_per_rank: list[AttnRectangles],
    ) -> None:
        """
        # Execute the non-communication QO algorithm logic

        Args:
            rect: Attention rectangles
            host_ranges_q: Q ranges for each rank
            host_ranges_k: K ranges for each rank
            num_heads_q: Number of Q heads
            num_heads_kv: Number of KV heads
            bucket_per_rank: Buckets for each rank
        """
        indexed_host_ranges_q = []
        for idx, intervals in enumerate(host_ranges_q):
            indexed_host_ranges_q.extend([(interval, idx) for interval in intervals])

        indexed_host_ranges_q.sort(key=lambda x: x[0].start)

        rest_rects = rects
        cut_pos = 0
        for item in indexed_host_ranges_q:
            interval: AttnRange = item[0]
            host_rank: int = item[1]
            if cut_pos != interval.start:
                cut_pos = interval.start
                _, rest_rects = rest_rects.cut_q(cut_pos=cut_pos)
            cut_pos = interval.end
            cut_rects, rest_rects = rest_rects.cut_q(cut_pos=cut_pos)

            bucket_per_rank[host_rank].extend(cut_rects)
