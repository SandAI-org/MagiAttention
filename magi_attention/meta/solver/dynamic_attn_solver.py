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

from typing import Union

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.common import (
    AttnRange,
    AttnRanges,
    AttnRectangles,
    is_cpp_backend_enable,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.algorithms import DynamicAttnAlgorithm
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.utils import nvtx

from .dist_attn_solver import BaseDistAttnSolver

USE_CPP_EXT = False
if is_cpp_backend_enable():
    try:
        from magi_attention import magi_attn_ext

        USE_CPP_EXT = True
    except ImportError:
        pass


class DynamicAttnSolver(BaseDistAttnSolver):
    """The dynamic-attn solver class to process dispatch meta for calc/comm meta"""

    @nvtx.instrument_nvtx
    def __init__(
        self,
        algorithm: DynamicAttnAlgorithm,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        cp_group: dist.ProcessGroup,
        dispatch_meta_q: DispatchMeta | None = None,
        dispatch_meta_k: DispatchMeta | None = None,
        total_seqlen_q: int | None = None,
        total_seqlen_k: int | None = None,
        host_ranges_q: list[AttnRanges] | None = None,
        host_ranges_k: list[AttnRanges] | None = None,
        cp_rank: int | None = None,
        cp_size: int | None = None,
        cp_mesh: DeviceMesh | None = None,
        calc_local_range: bool = True,
    ):
        self.algorithm = algorithm
        self.cp_group = cp_group
        self.cp_mesh = cp_mesh
        self.deterministic = magi_attention.is_deterministic_mode_enable()
        self.calc_local_range = calc_local_range

        # Prefer values from dispatch meta if provided
        if dispatch_meta_q is not None and dispatch_meta_k is not None:
            self.cp_rank = dispatch_meta_q.cp_rank
            self.cp_size = dispatch_meta_q.cp_size
            self.total_seqlen_q = dispatch_meta_q.total_seqlen
            self.total_seqlen_k = dispatch_meta_k.total_seqlen
            self.host_ranges_q = [
                hr.merge() for hr in dispatch_meta_q.host_ranges_per_rank
            ]
            self.host_ranges_k = [
                hr.merge() for hr in dispatch_meta_k.host_ranges_per_rank
            ]
            self.dispatch_chunk_size = dispatch_meta_q.chunk_size
        else:
            assert total_seqlen_q is not None and total_seqlen_k is not None
            assert host_ranges_q is not None and host_ranges_k is not None
            assert cp_rank is not None and cp_size is not None
            # for test/manual path
            self.cp_rank = cp_rank
            self.cp_size = cp_size
            self.total_seqlen_q = total_seqlen_q
            self.total_seqlen_k = total_seqlen_k
            self.host_ranges_q = [host_ranges.merge() for host_ranges in host_ranges_q]
            self.host_ranges_k = [host_ranges.merge() for host_ranges in host_ranges_k]
            self.dispatch_chunk_size = total_seqlen_q

        assert (
            num_heads_q % num_heads_kv == 0
        ), f"num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"

        self.org_num_heads_q = num_heads_q
        self.org_num_heads_kv = num_heads_kv
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.num_heads_group = 1
        self.head_dim = head_dim

        # set some attributes that might be fetched from outside
        self.host_q_ranges_global = self.host_ranges_q[self.cp_rank]
        self.host_k_ranges_global = self.host_ranges_k[self.cp_rank]

        self.bucket_per_rank = [AttnRectangles() for _ in range(self.cp_size)]

        self._is_solved = False

    def _expand_attn_ranges(self, ranges: AttnRanges, stride: int) -> AttnRanges:
        if USE_CPP_EXT:
            return magi_attn_ext.expand_attn_ranges(
                ranges, stride, self.num_heads_group  # type: ignore[arg-type, return-value]
            )

        new_ranges = AttnRanges()
        for i in range(self.num_heads_group):
            offset = i * stride
            for r in ranges:
                new_ranges.append(AttnRange(r.start + offset, r.end + offset))
        return new_ranges

    @nvtx.instrument_nvtx
    def solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: Union[list[int], list[AttnMaskType], AttnMaskType],
    ):
        flatten_head_groups = magi_attention.is_flatten_head_groups_enable()
        if flatten_head_groups:
            self.num_heads_group = self.num_heads_kv
            self.num_heads_q = self.num_heads_q // self.num_heads_group
            self.num_heads_kv = 1

            self.host_ranges_q = [
                self._expand_attn_ranges(ranges, self.total_seqlen_q)
                for ranges in self.host_ranges_q
            ]
            self.host_ranges_k = [
                self._expand_attn_ranges(ranges, self.total_seqlen_k)
                for ranges in self.host_ranges_k
            ]
            q_ranges = self._expand_attn_ranges(q_ranges, self.total_seqlen_q)
            k_ranges = self._expand_attn_ranges(k_ranges, self.total_seqlen_k)

            self.total_seqlen_q *= self.num_heads_group
            self.total_seqlen_k *= self.num_heads_group

            # reset some attributes that might be fetched from outside after flattening head groups
            self.host_q_ranges_global = self.host_ranges_q[self.cp_rank]
            self.host_k_ranges_global = self.host_ranges_k[self.cp_rank]

        # normalize attn_mask_type to list[AttnMaskType]
        if isinstance(attn_mask_type, AttnMaskType):
            attn_mask_type_list: list[int] | list[AttnMaskType] = [
                attn_mask_type
            ] * len(q_ranges)
        if isinstance(attn_mask_type, list):
            if flatten_head_groups:
                attn_mask_type_list = attn_mask_type * self.num_heads_group
            else:
                attn_mask_type_list = attn_mask_type
        else:
            raise TypeError(f"Unsupported attn_mask_type type: {type(attn_mask_type)}")

        self.rect = AttnRectangles.from_ranges(
            q_ranges=q_ranges, k_ranges=k_ranges, mask_types=attn_mask_type_list
        )

        self.algorithm.solve(
            rects=self.rect,
            host_ranges_q=self.host_ranges_q,
            host_ranges_k=self.host_ranges_k,
            num_heads_q=self.num_heads_q,
            num_heads_kv=self.num_heads_kv,
            num_heads_group=self.num_heads_group,
            bucket_per_rank=self.bucket_per_rank,
        )

        self.calc_host_and_remote_bucket_this_rank()

        self._is_solved = True

        # Calculate kv split alignment for native grpcoll
        if self.cp_size == 1:  # cp1 shortcut
            self.split_alignment_kv = 1
        else:
            self.split_alignment_kv = self.calc_split_alignment(
                chunk_size=self.dispatch_chunk_size,
                num_heads=self.num_heads_kv,
                head_dim=self.head_dim,
            )

        # Calculate qo split alignment for native grpcoll
        if self.cp_size == 1:  # cp1 shortcut
            self.split_alignment_qo = 1
        else:
            self.split_alignment_qo = self.calc_split_alignment(
                chunk_size=self.dispatch_chunk_size,
                num_heads=self.num_heads_q,
                head_dim=self.head_dim,
            )

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    def output_solve_result(
        self,
        visualize: bool = False,
        save_path: str | None = None,
        before_dispatch: bool = False,
    ) -> None:
        if not visualize:
            return

        # only for debug: local import, not depend on matplotlib in production environment
        try:
            from .dynamic_solver_vis import visualize_buckets

            vis_buckets = []
            if before_dispatch:
                # Re-map q/k ranges for visualization
                # Map global ranges to rank-ordered sequential ranges
                # Rule: rank as primary key, host_range as secondary key

                def get_vis_mapping(host_ranges_per_rank: list[AttnRanges]):
                    mapping = []  # list of (global_start, global_end, vis_start)
                    current_vis_start = 0
                    for rank_ranges in host_ranges_per_rank:
                        for r in rank_ranges:
                            mapping.append((r.start, r.end, current_vis_start))
                            current_vis_start += r.end - r.start
                    return mapping

                def apply_mapping(rects: AttnRectangles, q_mapping, k_mapping):
                    new_rects = AttnRectangles()
                    for rect in rects:
                        # Handle both Python and C++ AttnRectangle
                        # In C++ version, q_range and k_range are AttnRange objects, not AttnRanges (list-like)
                        qr = rect.q_range
                        kr = rect.k_range

                        new_q_range = None
                        for start, end, vis_start in q_mapping:
                            if qr.start >= start and qr.end <= end:
                                new_q_range = AttnRange(
                                    qr.start - start + vis_start,
                                    qr.end - start + vis_start,
                                )
                                break

                        new_k_range = None
                        for start, end, vis_start in k_mapping:
                            if kr.start >= start and kr.end <= end:
                                new_k_range = AttnRange(
                                    kr.start - start + vis_start,
                                    kr.end - start + vis_start,
                                )
                                break

                        if new_q_range is not None and new_k_range is not None:
                            # Use positional arguments to avoid keyword argument issues with C++ extension
                            # C++ AttnRectangle doesn't have mask_type attribute, but we can get it from to_qk_range_mask_type
                            mask_type = 0  # Default to FULL
                            qk_mask_list = rect.to_qk_range_mask_type()
                            if len(qk_mask_list) > 0:
                                mask_type = qk_mask_list[0][2]

                            new_rects.append(
                                magi_attention.common.AttnRectangle(
                                    q_range=new_q_range,
                                    k_range=new_k_range,
                                    mask_type=mask_type,
                                )
                            )
                    return new_rects

                q_mapping = get_vis_mapping(self.host_ranges_q)
                k_mapping = get_vis_mapping(self.host_ranges_k)

                for i in range(self.cp_size):
                    vis_buckets.append(
                        apply_mapping(self.bucket_per_rank[i], q_mapping, k_mapping)
                    )
            else:
                vis_buckets = self.bucket_per_rank

            visualize_buckets(
                bucket_per_rank=vis_buckets,
                title="DynamicAttnSolver buckets",
                save_path=save_path,
            )
        except Exception as e:  # pragma: no cover - debug only
            raise e

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

    def make_split_alignment(self, ranges: AttnRanges, calc_kv: bool) -> AttnRanges:
        if calc_kv and self.split_alignment_kv > 1:
            return ranges.merge_with_split_alignment(
                split_alignment=self.split_alignment_kv
            )
        elif not calc_kv and self.split_alignment_qo > 1:
            return ranges.merge_with_split_alignment(
                split_alignment=self.split_alignment_qo
            )
        return ranges

    @nvtx.instrument_nvtx
    def _calc_group_collective_arg(
        self,
        calc_kv: bool = True,
        calc_local_range: bool = True,
    ) -> GroupCollectiveArg:
        host_ranges = self.host_ranges_k if calc_kv else self.host_ranges_q
        # =========== process local-calc-remote-hold message ===========
        indexed_remote_hold_ranges = []
        for idx, intervals in enumerate(host_ranges):
            # if calc_local_range == True, host range needs send to itself
            if (idx != self.cp_rank) or calc_local_range:
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

        # make split_alignment for group collective optimization
        local_calc_ranges = self.make_split_alignment(local_calc_ranges, calc_kv)

        # local_calc_ranges is sorted and merged
        intersections = self._calc_intersection_with_index(
            local_calc_ranges, indexed_remote_hold_ranges
        )

        # splict_size = end - start
        output_split_size_list = [x[1] - x[0] for x in intersections]
        src_index_list = [x[2] for x in intersections]

        # =========== process local-hold-remote-calc message ===========
        host_ranges_this_rank: AttnRanges = host_ranges[self.cp_rank]
        # host_ranges is sorted and merged

        # Obtain the sending ranges and ranks using the scan line method
        scanning_line_event = []
        for remote_rank in range(self.cp_size):
            # calc global range do not need host range communicate to itself
            if (not calc_local_range) and remote_rank == self.cp_rank:
                continue
            # calc local range need host range communicate to itself when have calc job
            if calc_local_range and remote_rank == self.cp_rank:
                remote_calc_ranges = (
                    self.remote_bucket_this_rank.get_kv_ranges_union()
                    if calc_kv
                    else self.remote_bucket_this_rank.get_qo_ranges_union()
                )
            else:
                remote_calc_ranges = (
                    self.bucket_per_rank[remote_rank].get_kv_ranges_union()
                    if calc_kv
                    else self.bucket_per_rank[remote_rank].get_qo_ranges_union()
                )

            # make split_alignment for group collective optimization
            remote_calc_ranges = self.make_split_alignment(remote_calc_ranges, calc_kv)

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

        # build group collective arg
        group_collective_arg = GroupCollectiveArg(
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            rank=self.cp_rank,
            world_size=self.cp_size,
            group=self.cp_group,
            device_mesh=self.cp_mesh,
            deterministic=self.deterministic,
            split_alignment=self.split_alignment_kv
            if calc_kv
            else self.split_alignment_qo,
        )
        return group_collective_arg

    @nvtx.instrument_nvtx
    def make_comm_meta(self) -> CommMeta:
        """Calculate communication meta for kv and qo group collective"""

        num_remote_kv_tokens_per_stage: list[int] = []
        kv_group_collective_args_list: list[GroupCollectiveArg] = []
        num_remote_qo_tokens_per_stage: list[int] = []
        qo_group_collective_args_list: list[GroupCollectiveArg] = []

        if self.cp_size > 1:  # cp1 shortcut
            kv_group_collective_arg: GroupCollectiveArg = (
                self._calc_group_collective_arg(
                    calc_kv=True,
                    calc_local_range=self.calc_local_range,
                )
            )
            kv_group_collective_args_list.append(kv_group_collective_arg)
            num_remote_kv_tokens_per_stage.append(
                sum(kv_group_collective_arg.output_split_size_list)
            )

            qo_group_collective_arg: GroupCollectiveArg = (
                self._calc_group_collective_arg(
                    calc_kv=False,
                    calc_local_range=self.calc_local_range,
                )
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
            num_heads_q=self.org_num_heads_q,
            num_heads_kv=self.org_num_heads_kv,
            head_dim=self.head_dim,
        )

        return comm_meta

    def calc_area(self) -> int:
        """Calculate the area of the buckets for testing correctness"""
        area = 0
        for rects in self.bucket_per_rank:
            area += rects.area()
        return area

    @nvtx.instrument_nvtx
    def calc_host_and_remote_bucket_this_rank(self) -> None:
        bucket_this_rank: AttnRectangles = self.bucket_per_rank[self.cp_rank]
        host_ranges_q_this_rank: AttnRanges = self.host_ranges_q[self.cp_rank]
        host_ranges_k_this_rank: AttnRanges = self.host_ranges_k[self.cp_rank]
        # host_ranges is sorted and merged

        if USE_CPP_EXT:
            (
                self.host_bucket_this_rank,
                self.remote_bucket_this_rank,
            ) = magi_attn_ext.cut_host_remote_buckets(
                bucket_this_rank,  # type: ignore[arg-type]
                host_ranges_q_this_rank,  # type: ignore[arg-type]
                host_ranges_k_this_rank,  # type: ignore[arg-type]
            )
            return

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
            # q_range host_range_q.start ~ host_range_q.end is host job
            cut_pos_q = host_range_q.end
            rest_rects_k, rest_rects_q = rest_rects_q.cut_q(cut_pos=cut_pos_q)

            # cut host job tile (rest_rects_k) with host range k
            cut_pos_k = 0
            for host_range_k in host_ranges_k_this_rank:
                if cut_pos_k != host_range_k.start:
                    # k_range cut_pos_k ~ host_range_k.start is remote job
                    cut_pos_k = host_range_k.start
                    cut_rects, rest_rects_k = rest_rects_k.cut_k(cut_pos=cut_pos_k)
                    self.remote_bucket_this_rank.extend(cut_rects)
                # k_range host_range_k.start ~ host_range_k.end is host job
                cut_pos_k = host_range_k.end
                cut_rects, rest_rects_k = rest_rects_k.cut_k(cut_pos=cut_pos_k)
                self.host_bucket_this_rank.extend(cut_rects)

            # leftover rest_rects_k is remote job
            self.remote_bucket_this_rank.extend(rest_rects_k)

        # leftover rest_rects_q is remote job
        self.remote_bucket_this_rank.extend(rest_rects_q)

    @nvtx.instrument_nvtx
    def make_calc_meta(
        self,
    ) -> CalcMeta:
        """Calculate flex-flash-attention calculation meta for this rank"""
        local_attn_arg_q_ranges = AttnRanges()
        local_attn_arg_k_ranges = AttnRanges()
        local_attn_arg_attn_type_map = []
        local_attn_arg_total_area = 0
        for rect in self.host_bucket_this_rank:
            qk_range_mask_type_list = rect.to_qk_range_mask_type()
            for q_range, k_range, mask_type in qk_range_mask_type_list:
                local_attn_arg_q_ranges.append(q_range)
                local_attn_arg_k_ranges.append(k_range)
                local_attn_arg_attn_type_map.append(mask_type)
            local_attn_arg_total_area += rect.area()

        remote_attn_arg_q_ranges = AttnRanges()
        remote_attn_arg_k_ranges = AttnRanges()
        remote_attn_arg_attn_type_map = []
        remote_attn_arg_total_area = 0

        for rect in self.remote_bucket_this_rank:
            qk_range_mask_type_list = rect.to_qk_range_mask_type()
            for q_range, k_range, mask_type in qk_range_mask_type_list:
                remote_attn_arg_q_ranges.append(q_range)
                remote_attn_arg_k_ranges.append(k_range)
                remote_attn_arg_attn_type_map.append(mask_type)
            remote_attn_arg_total_area += rect.area()

        if self.calc_local_range:
            local_attn_arg_q_ranges = self.host_q_ranges_global.make_ranges_local(
                local_attn_arg_q_ranges
            )
            local_attn_arg_k_ranges = self.host_k_ranges_global.make_ranges_local(
                local_attn_arg_k_ranges
            )
            # make split_alignment for remote q ranges
            remote_q_ranges_global = self.make_split_alignment(
                remote_attn_arg_q_ranges, calc_kv=False
            )
            remote_attn_arg_q_ranges = remote_q_ranges_global.make_ranges_local(
                remote_attn_arg_q_ranges
            )
            # make split_alignment for remote k ranges
            remote_k_ranges_global = self.make_split_alignment(
                remote_attn_arg_k_ranges, calc_kv=True
            )
            remote_attn_arg_k_ranges = remote_k_ranges_global.make_ranges_local(
                remote_attn_arg_k_ranges
            )

        local_attn_arg = AttnArg(
            q_ranges=local_attn_arg_q_ranges,
            k_ranges=local_attn_arg_k_ranges,
            attn_type_map=local_attn_arg_attn_type_map,
            total_area=local_attn_arg_total_area,
        )

        remote_attn_args_list: list[AttnArg] = []
        if self.cp_size > 1:  # cp1 shortcut
            remote_attn_arg = AttnArg(
                q_ranges=remote_attn_arg_q_ranges,
                k_ranges=remote_attn_arg_k_ranges,
                attn_type_map=remote_attn_arg_attn_type_map,
                total_area=remote_attn_arg_total_area,
            )
            remote_attn_args_list.append(remote_attn_arg)

        # build calc meta
        calc_meta = CalcMeta(
            local_attn_arg=local_attn_arg,
            remote_attn_args_list=remote_attn_args_list,
        )

        return calc_meta
