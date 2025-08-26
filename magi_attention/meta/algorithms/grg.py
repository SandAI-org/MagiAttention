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

from magi_attention.common import AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class GRGDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The greedy-random-grid dynamic dispatch algorithm implementation"""

    def __init__(self):
        """
        The init method of the greedy-random-grid dynamic dispatch algorithm
        """
        pass

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.GREEDY_RANDOM_GRID

    def solve(
        self,
        rects: AttnRectangles,
        host_ranges_q: list[AttnRanges],
        bucket_per_rank: list[AttnRectangles],
    ) -> None:
        """
        The solve method of the greedy-random-grid dynamic dispatch algorithm

        Args:
            rects: The attention rectangles
            host_ranges_q: The Q ranges of each rank
            bucket_per_rank: The buckets of each rank
        """
        # TODO: Implement the greedy-random-grid algorithm logic
        # Currently, it is a placeholder implementation
        pass
