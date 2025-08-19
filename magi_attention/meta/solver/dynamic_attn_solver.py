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

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from magi_attention.common.enum import AttnMaskType, DynamicAttnAlgType
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.utils import nvtx


@dataclass(frozen=True)
class DynamicAttnAlg(ABC):
    """The abstract config/meta info dataclass for specific dynamic dispatch algorithm"""

    @property
    @abstractmethod
    def type(self) -> DynamicAttnAlgType:
        """The type enum of the dynamic dispatch algorithm"""


@dataclass(frozen=True)
class NCQDynamicAttnAlg(DynamicAttnAlg):
    """The config/meta info dataclass for the non-comm-qo dynamic dispatch algorithm"""

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.NON_COMMUNICATION_QO


@dataclass(frozen=True)
class GRGDynamicAttnAlg(DynamicAttnAlg):
    """The config/meta info dataclass for the greedy-random-grid dynamic dispatch algorithm"""

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.GREEDY_RANDOM_GRID


class DynamicAttnSolver:
    """The dynamic-attn solver class to process dispatch meta for calc/comm meta"""

    @nvtx.instrument_nvtx
    def __init__(
        self,
        alg: DynamicAttnAlg,
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        num_heads_q: int,
        num_heads_kv: int,
        cp_group: dist.ProcessGroup,
        cp_mesh: DeviceMesh | None = None,
    ):
        self.alg = alg

        self.cp_rank = dist.get_rank(cp_group)
        self.cp_size = dist.get_world_size(cp_group)
        self.cp_group = cp_group
        self.cp_mesh = cp_mesh

        self.total_seqlen_q: int = dispatch_meta_q.total_seqlen
        self.total_seqlen_k: int = dispatch_meta_k.total_seqlen
        self.host_ranges_q: list[AttnRanges] = dispatch_meta_q.host_ranges_per_rank
        self.host_ranges_k: list[AttnRanges] = dispatch_meta_k.host_ranges_per_rank

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.solve_func = {
            DynamicAttnAlgType.NON_COMMUNICATION_QO: self._solve_with_non_comm_qo,
            DynamicAttnAlgType.GREEDY_RANDOM_GRID: self._solve_with_greedy_random_grid,
        }[self.alg.type]

    def solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
    ):
        self.q_ranges = q_ranges
        self.k_ranges = k_ranges
        self.attn_mask_type = attn_mask_type
        # DynamicAttnSolver currently only supports full mask
        for mask_type in attn_mask_type:
            if mask_type != AttnMaskType.FULL:
                raise ValueError(
                    "Dynamic attn solver currently only supports full mask"
                )
        self.solve_func(**asdict(self.alg))

    def _solve_with_non_comm_qo(
        self,
        **kwargs,
    ):
        pass

    def _solve_with_greedy_random_grid(
        self,
        **kwargs,
    ):
        pass
