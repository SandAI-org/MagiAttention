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

from typing import Union

# import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.algorithms import DynamicAttnAlgorithm
from magi_attention.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.utils import nvtx

# from magi_attention.meta.collection.dispatch_meta import DispatchMeta


class DynamicAttnSolver:
    """The dynamic-attn solver class to process dispatch meta for calc/comm meta"""

    def __init__(
        self,
        algorithm: DynamicAttnAlgorithm,
        # dispatch_meta_q: DispatchMeta,
        # dispatch_meta_k: DispatchMeta,
        total_seqlen_q: int,
        total_seqlen_k: int,
        host_ranges_q: list[AttnRanges],
        host_ranges_k: list[AttnRanges],
        num_heads_q: int,
        num_heads_kv: int,
        cp_rank: int,
        cp_size: int,
        # cp_group: dist.ProcessGroup,
        cp_mesh: DeviceMesh | None = None,
        deterministic: bool = False,
    ):
        self.algorithm = algorithm

        # for dispatch solver
        # self.cp_rank = dist.get_rank(cp_group)
        # self.cp_size = dist.get_world_size(cp_group)
        # self.cp_group = cp_group
        self.cp_mesh = cp_mesh

        # self.total_seqlen_q: int = dispatch_meta_q.total_seqlen
        # self.total_seqlen_k: int = dispatch_meta_k.total_seqlen
        # self.host_ranges_q: list[AttnRanges] = dispatch_meta_q.host_ranges_per_rank
        # self.host_ranges_k: list[AttnRanges] = dispatch_meta_k.host_ranges_per_rank
        self.deterministic = deterministic

        # for test
        self.cp_rank = cp_rank
        self.cp_size = cp_size

        self.total_seqlen_q: int = total_seqlen_q
        self.total_seqlen_k: int = total_seqlen_k
        self.host_ranges_q: list[AttnRanges] = [
            host_ranges.merge() for host_ranges in host_ranges_q
        ]
        self.host_ranges_k: list[AttnRanges] = [
            host_ranges.merge() for host_ranges in host_ranges_k
        ]

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.bucket_per_rank = [AttnRectangles() for _ in range(self.cp_size)]

    def solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        mask_types: Union[list[int], list[AttnMaskType]],
    ):
        print("=============== solve begin ===================")
        self.rect = AttnRectangles.from_ranges(
            q_ranges=q_ranges, k_ranges=k_ranges, mask_types=mask_types
        )

        self.algorithm.solve(
            rects=self.rect,
            host_ranges_q=self.host_ranges_q,
            bucket_per_rank=self.bucket_per_rank,
        )
        # for idx, bucket in enumerate(self.bucket_per_rank):
        #     print(f"rank{idx}'s bucket:")
        #     print(bucket)
        self.calc_host_and_remote_bucket_this_rank()
        print(f"host bucket this rank {self.cp_rank}:")
        print(self.host_bucket_this_rank)
        print(f"remote bucket this rank {self.cp_rank}:")
        print(self.remote_bucket_this_rank)

    @nvtx.instrument_nvtx
    def _calc_intersection_with_index(
        self,
        rangesA: AttnRanges,
        rangesB: list[tuple[AttnRange, int]],
    ) -> list[list[int]]:
        # Calculate the intersection of intervals using the two-pointer method
        i = j = 0
        intersections: list[list[int]] = []
        while i < len(rangesA) and j < len(rangesB):
            rangeA = rangesA[i]
            rangeB, index = rangesB[j]
            start = max(rangeA.start, rangeB.start)
            end = min(rangeA.end, rangeB.end)
            if start < end:
                if len(intersections) != 0:
                    if intersections[-1][1] == start and intersections[-1][2] == index:
                        # merge with previous interval
                        intersections[-1][1] = end
                    else:
                        intersections.append([start, end, index])
                else:
                    intersections.append([start, end, index])
            if rangeA.end < rangeB.end:
                i += 1
            else:
                j += 1
        return intersections

    @nvtx.instrument_nvtx
    def _calc_intersection(
        self,
        rangesA: AttnRanges,
        rangesB: AttnRanges,
    ) -> list[list[int]]:
        # Calculate the intersection of intervals using the two-pointer method
        i = j = 0
        intersections: list[list[int]] = []
        while i < len(rangesA) and j < len(rangesB):
            rangeA = rangesA[i]
            rangeB = rangesB[j]
            start = max(rangeA.start, rangeB.start)
            end = min(rangeA.end, rangeB.end)
            if start < end:
                if len(intersections) != 0:
                    if intersections[-1][1] == start:
                        # merge with previous interval
                        intersections[-1][1] = end
                    else:
                        intersections.append([start, end])
                else:
                    intersections.append([start, end])

            if rangeA.end < rangeB.end:
                i += 1
            else:
                j += 1
        return intersections

    @nvtx.instrument_nvtx
    def _calc_group_collective_arg(
        self,
        calc_kv: bool = True,
    ) -> GroupCollectiveArg:
        host_ranges = self.host_ranges_k if calc_kv else self.host_ranges_q
        # =========== process local-calc-remote-hold message ===========
        indexed_remote_hold_ranges = []
        for idx, intervals in enumerate(host_ranges):
            if idx != self.cp_rank:
                indexed_remote_hold_ranges.extend(
                    [(interval, idx) for interval in intervals]
                )
        # sort with range start
        indexed_remote_hold_ranges.sort(key=lambda x: x[0].start)

        local_calc_ranges: AttnRanges = (
            self.remote_bucket_this_rank.get_kv_ranges_union()
            if calc_kv
            else self.remote_bucket_this_rank.get_qo_ranges_union()
        )
        # local_calc_ranges is sorted and merged
        intersections = self._calc_intersection_with_index(
            local_calc_ranges, indexed_remote_hold_ranges
        )

        # splict_size = end - start
        output_split_size_list = [x[1] - x[0] for x in intersections]
        src_index_list = [x[2] for x in intersections]

        # print("local calc remote hold message")
        # print(output_split_size_list)
        # print(src_index_list)

        # =========== process local-hold-remote-calc message ===========
        host_ranges_this_rank: AttnRanges = host_ranges[self.cp_rank]
        # host_ranges is sorted and merged

        # Obtain the sending ranges and ranks using the scan line method
        scanning_line_event = []
        for remote_rank in range(self.cp_size):
            if remote_rank == self.cp_rank:
                continue
            remote_calc_ranges = (
                self.bucket_per_rank[remote_rank].get_kv_ranges_union()
                if calc_kv
                else self.bucket_per_rank[remote_rank].get_qo_ranges_union()
            )
            intersections = self._calc_intersection(
                host_ranges_this_rank, remote_calc_ranges
            )
            for interval in intersections:
                # add remote rank at start and delete at end
                # event msg = +- (remote_rank + 1) to deal with remote_rank = 0
                scanning_line_event.append((interval[0], remote_rank + 1))
                scanning_line_event.append((interval[1], -remote_rank - 1))

        scanning_line_event.sort(key=lambda x: x[0])

        input_split_size_list = []
        dst_indices_list = []
        dst_indices = set()
        i = 0
        for host_range in host_ranges_this_rank:
            cur_start = host_range.start
            while cur_start < host_range.end:
                while (
                    i < len(scanning_line_event)
                    and scanning_line_event[i][0] <= cur_start
                ):
                    event_msg = scanning_line_event[i][1]
                    if event_msg > 0:
                        add_rank = event_msg - 1
                        dst_indices.add(add_rank)
                    else:
                        del_rank = -event_msg - 1
                        dst_indices.remove(del_rank)
                    i += 1
                if i < len(scanning_line_event):
                    cur_end = min(host_range.end, scanning_line_event[i][0])
                else:
                    cur_end = host_range.end
                input_split_size_list.append(cur_end - cur_start)
                dst_indices_list.append(list(dst_indices))
                cur_start = cur_end

        # print("local hold remote calc message")
        # print(input_split_size_list)
        # print(dst_indices_list)

        # build group collective arg
        group_collective_arg = GroupCollectiveArg(
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            rank=self.cp_rank,
            world_size=self.cp_size,
            device_mesh=self.cp_mesh,
            deterministic=self.deterministic,
        )
        return group_collective_arg

    @nvtx.instrument_nvtx
    def calc_comm_meta(self) -> CommMeta:
        """Calculate communication meta for kv and qo group collective"""

        num_remote_kv_tokens_per_stage: list[int] = []
        kv_group_collective_args_list: list[GroupCollectiveArg] = []

        kv_group_collective_arg: GroupCollectiveArg = self._calc_group_collective_arg(
            True
        )
        kv_group_collective_args_list.append(kv_group_collective_arg)
        num_remote_kv_tokens_per_stage.append(
            sum(kv_group_collective_arg.output_split_size_list)
        )

        num_remote_qo_tokens_per_stage: list[int] = []
        qo_group_collective_args_list: list[GroupCollectiveArg] = []

        qo_group_collective_arg: GroupCollectiveArg = self._calc_group_collective_arg(
            False
        )
        qo_group_collective_args_list.append(qo_group_collective_arg)
        num_remote_qo_tokens_per_stage.append(
            sum(qo_group_collective_arg.output_split_size_list)
        )

        # build comm meta
        comm_meta = CommMeta(
            num_remote_kv_tokens_per_stage=num_remote_kv_tokens_per_stage,
            kv_group_collective_args_list=kv_group_collective_args_list,
            num_remote_qo_tokens_per_stage=num_remote_qo_tokens_per_stage,
            qo_group_collective_args_list=qo_group_collective_args_list,
        )

        return comm_meta

    @nvtx.instrument_nvtx
    def calc_host_and_remote_bucket_this_rank(self) -> None:
        bucket_this_rank: AttnRectangles = self.bucket_per_rank[self.cp_rank]
        host_ranges_q_this_rank: AttnRanges = self.host_ranges_q[self.cp_rank]
        host_ranges_k_this_rank: AttnRanges = self.host_ranges_q[self.cp_rank]
        # host_ranges is sorted and merged
        self.host_bucket_this_rank = AttnRectangles()
        self.remote_bucket_this_rank = AttnRectangles()

        # cut rects with host_ranges_q endpoint and host_ranges_k endpoint
        cut_pos_q = 0
        rest_rects_q = bucket_this_rank
        for host_range_q in host_ranges_q_this_rank:
            if cut_pos_q != host_range_q.start:
                # q_range cut_pos_q ~ host_range_q.start is remote job
                cut_pos_q = host_range_q.start
                cut_rects, rest_rects_q = rest_rects_q.cut_q(cut_pos=cut_pos_q)
                self.remote_bucket_this_rank.extend(cut_rects)
            # q_range cut_pos_q ~ host_range_q.start is host job
            cut_pos_q = host_range_q.end
            rest_rects_k, rest_rects_q = rest_rects_q.cut_q(cut_pos=cut_pos_q)

            # cut host job tile (rest_rects_k) with host range k
            for host_range_k in host_ranges_k_this_rank:
                cut_pos_k = 0
                for host_range_k in host_ranges_k_this_rank:
                    if cut_pos_k != host_range_k.start:
                        # k_range cut_pos_k ~ host_range_k.start is remote job
                        cut_pos_k = host_range_k.start
                        cut_rects, rest_rects_k = rest_rects_k.cut_k(cut_pos=cut_pos_k)
                        self.remote_bucket_this_rank.extend(cut_rects)
                    # k_range cut_pos_k ~ host_range_k.end is host job
                    cut_pos_k = host_range_k.end
                    cut_rects, rest_rects_k = rest_rects_k.cut_k(cut_pos=cut_pos_k)
                    self.host_bucket_this_rank.extend(cut_rects)

            # leftover rest_rects_k is remote job
            self.remote_bucket_this_rank.extend(rest_rects_k)

        # leftover rest_rects_q is remote job
        self.remote_bucket_this_rank.extend(rest_rects_q)

    @nvtx.instrument_nvtx
    def calc_attn_calc_meta(self) -> AttnCalcMeta:
        """Calculate flex-flash-attention calculation meta for this rank"""
        local_attn_arg = AttnArg(
            q_ranges=AttnRanges(),
            k_ranges=AttnRanges(),
            attn_type_map=[],
            shard_seqlen_q=self.total_seqlen_q,
            total_area=0,
        )
        for rect in self.host_bucket_this_rank:
            qk_range_mask_type_list = rect.to_qk_range_mask_type()
            for q_range, k_range, mask_type in qk_range_mask_type_list:
                local_attn_arg.q_ranges.append(q_range)
                local_attn_arg.k_ranges.append(k_range)
                local_attn_arg.attn_type_map.append(mask_type)
            local_attn_arg.total_area += rect.area()

        remote_attn_arg = AttnArg(
            q_ranges=AttnRanges(),
            k_ranges=AttnRanges(),
            attn_type_map=[],
            shard_seqlen_q=self.total_seqlen_q,
            total_area=0,
        )
        for rect in self.remote_bucket_this_rank:
            qk_range_mask_type_list = rect.to_qk_range_mask_type()
            for q_range, k_range, mask_type in qk_range_mask_type_list:
                remote_attn_arg.q_ranges.append(q_range)
                remote_attn_arg.k_ranges.append(k_range)
                remote_attn_arg.attn_type_map.append(mask_type)
            remote_attn_arg.total_area += rect.area()

        remote_attn_args_list: list[AttnArg] = []
        remote_attn_args_list.append(remote_attn_arg)

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=local_attn_arg,
            remote_attn_args_list=remote_attn_args_list,
        )

        return attn_calc_meta
