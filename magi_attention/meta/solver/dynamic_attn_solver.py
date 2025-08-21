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
from typing import Union

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import AttnMaskType, DynamicAttnAlgType
from magi_attention.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from magi_attention.utils import nvtx

# from magi_attention.meta.collection.dispatch_meta import DispatchMeta

# import torch.distributed as dist
# from torch.distributed.device_mesh import DeviceMesh


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

    def __init__(
        self,
        alg: DynamicAttnAlg,
        # dispatch_meta_q: DispatchMeta,
        # dispatch_meta_k: DispatchMeta,
        total_seqlen_q: int,
        total_seqlen_k: int,
        host_ranges_q: list[AttnRanges],
        host_ranges_k: list[AttnRanges],
        num_heads_q: int,
        num_heads_kv: int,
        # cp_group: dist.ProcessGroup,
        # cp_mesh: DeviceMesh | None = None,
        cp_rank: int,
        cp_size: int,
    ):
        self.alg = alg

        # for dispatch solver
        # self.cp_rank = dist.get_rank(cp_group)
        # self.cp_size = dist.get_world_size(cp_group)
        # self.cp_group = cp_group
        # self.cp_mesh = cp_mesh

        # self.total_seqlen_q: int = dispatch_meta_q.total_seqlen
        # self.total_seqlen_k: int = dispatch_meta_k.total_seqlen
        # self.host_ranges_q: list[AttnRanges] = dispatch_meta_q.host_ranges_per_rank
        # self.host_ranges_k: list[AttnRanges] = dispatch_meta_k.host_ranges_per_rank

        # for test
        self.cp_rank = cp_rank
        self.cp_size = cp_size

        self.total_seqlen_q: int = total_seqlen_q
        self.total_seqlen_k: int = total_seqlen_k
        self.host_ranges_q: list[AttnRanges] = [
            host_range.merge() for host_range in host_ranges_q
        ]
        self.host_ranges_k: list[AttnRanges] = [
            host_range.merge() for host_range in host_ranges_k
        ]

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.solve_func = {
            DynamicAttnAlgType.NON_COMMUNICATION_QO: self._solve_with_non_comm_qo,
            DynamicAttnAlgType.GREEDY_RANDOM_GRID: self._solve_with_greedy_random_grid,
        }[self.alg.type]

        self.buckets = [AttnRectangles() for _ in range(self.cp_size)]

    def solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        mask_types: Union[list[int], list[AttnMaskType]],
    ):
        print("=============== solve begin ===================")
        # save solve original message
        # self.q_ranges = q_ranges
        # self.k_ranges = k_ranges
        # attn_mask_types = [
        #     {
        #         0: AttnMaskType.FULL,
        #         1: AttnMaskType.CAUSAL,
        #         2: AttnMaskType.INVCAUSAL,
        #         3: AttnMaskType.BICAUSAL,
        #     }[i]
        #     if isinstance(i, int)
        #     else i
        #     for i in mask_types
        # ]
        # self.attn_mask_types = attn_mask_types
        self.rect = AttnRectangles.from_ranges(
            q_ranges=q_ranges, k_ranges=k_ranges, mask_types=mask_types
        )
        self.solve_func(**asdict(self.alg))

    def _solve_with_non_comm_qo(
        self,
        **kwargs,
    ):
        indexed_intervals = []
        for idx, intervals in enumerate(self.host_ranges_q):
            indexed_intervals.extend([(interval, idx) for interval in intervals])

        # sort with range start
        indexed_intervals.sort(key=lambda x: x[0].start)

        # cut rects with host_ranges endpoint
        rest_rects = self.rect
        cut_pos = 0
        for item in indexed_intervals:
            interval: AttnRange = item[0]
            host_rank: int = item[1]
            if cut_pos != interval.start:
                cut_pos = interval.start
                _, rest_rects = rest_rects.cut_q(cut_pos=cut_pos)
            cut_pos = interval.end
            cut_rects, rest_rects = rest_rects.cut_q(cut_pos=cut_pos)
            # give cut_rects to host_rank
            self.buckets[host_rank].extend(cut_rects)

        for idx, bucket in enumerate(self.buckets):
            print(f"bucket{idx}:")
            print(bucket)

    def _solve_with_greedy_random_grid(
        self,
        **kwargs,
    ):
        pass

    @nvtx.instrument_nvtx
    def calc_attn_calc_meta(self) -> AttnCalcMeta:
        """Calculate flex-flash-attention calculation meta"""
        local_attn_arg = AttnArg(
            q_ranges=AttnRanges(),
            k_ranges=AttnRanges(),
            attn_type_map=[],
            shard_seqlen_q=0,
            total_area=0,
        )
        remote_attn_args_list: list[AttnArg] = []
        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=local_attn_arg,
            remote_attn_args_list=remote_attn_args_list,
        )

        return attn_calc_meta
