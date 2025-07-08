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

from collections import defaultdict
from functools import partial
from itertools import accumulate, chain, pairwise
from logging import getLogger
from typing import Callable, Optional, TypeAlias

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.common.range import NaiveRange
from magi_attention.common.range_op import range_gather, range_reduce
from magi_attention.common.ranges import NaiveRanges
from magi_attention.utils import nvtx

logger = getLogger("magi_attention")

__all__ = [
    # for group-cast
    "sanity_check_for_group_cast_meta_args_per_rank",
    "_calc_group_cast_a2a_args",
    "_calc_group_cast_a2a_input_meta_args",
    "_calc_group_cast_a2a_output_meta_args",
    "_init_hier_group_cast_meta_solver",
    "_hier_group_cast_impl_with_a2av",
    # for group-reduce
    "sanity_check_for_group_reduce_meta_args_per_rank",
    "_calc_group_reduce_a2a_args",
    "_calc_group_reduce_a2a_input_meta_args",
    "_calc_group_reduce_a2a_output_meta_args",
    # for others
    "get_pg_backend",
]


def get_pg_backend(pg: dist.ProcessGroup, device: str = "cuda") -> dist.Backend:
    return pg._get_backend(torch.device(device))


def _sanity_check_nccl_send_recv(
    num_send_list: list[int], num_recv_list: list[int], world_size: int
):
    if num_send_list != num_recv_list:
        num_diff_idxs: list[int] = torch.nonzero(
            torch.tensor(num_send_list) - torch.tensor(num_recv_list), as_tuple=True
        )[0].tolist()

        msg = [
            (
                "For each pair of src_rank and dst_rank, "
                "The number of nccl send calls launched by src_rank for dst_rank "
                "should be identical to the number of nccl recv calls launched by dst_rank for src_rank, "
                "but got: "
            )
        ]
        for idx in num_diff_idxs:
            src_rank, dst_rank = divmod(idx, world_size)
            msg.append(
                f"The number of send calls launched by rank{src_rank} for rank{dst_rank} is {num_send_list[idx]} "
                f"while the number of recv calls launched by rank{dst_rank} for rank{src_rank} is {num_recv_list[idx]}."
            )

        raise AssertionError("\n".join(msg))


def _seqlens2curanges(
    seqlens: list[int],
) -> NaiveRanges:
    """Make seqlens to cumulative ranges

    Args:
        seqlens (list[int]): the seqlen list, e.g. [4, 2, 7]

    Returns:
        NaiveRanges: the cumulative ranges, e.g. [(0, 4), (4, 6), (6, 13)]
    """
    return list(pairwise(accumulate([0] + seqlens)))


RangesWithRank: TypeAlias = list[tuple[NaiveRange, int]]


def _calc_range_gather_kwargs_from_ranges_with_rank(
    a2a_input_size_ranges_with_rank: RangesWithRank,
    device: torch.device,
) -> dict:
    # get range_gather's ranges from a2a_input_size_ranges_with_rank
    ranges = [(start, end) for (start, end), _ in a2a_input_size_ranges_with_rank]

    # calculate range sizes
    range_sizes = [end - start for start, end in ranges]

    # calculate the output size
    total_size = sum(range_sizes)

    # calculate row_map from row idx to range idx
    range_sizes = torch.tensor([0] + range_sizes, dtype=torch.int32, device=device)
    row_map = torch.repeat_interleave(
        torch.arange(0, len(ranges), device=device),
        range_sizes[1:],
        dim=0,
        output_size=total_size,
    )

    # calculate cu_range_sizes
    cu_range_sizes = torch.cumsum(range_sizes, dim=0)

    range_gather_kwargs = {
        "ranges": torch.tensor(ranges, device=device),
        "cu_range_sizes": cu_range_sizes,
        "row_map": row_map,
        "total_size": total_size,
    }

    return range_gather_kwargs


def _calc_unperm_range_gather_kwargs_from_split_size_list(
    split_size_list: list[int],
    unpermute_index_list: list[int],
    device: torch.device,
) -> dict:
    # calculate the output size
    total_size = sum(split_size_list)

    # calculate each range's start and end
    ranges = _seqlens2curanges(split_size_list)

    # re-order ranges to be in the order of the output tensor
    ranges = [ranges[i] for i in unpermute_index_list]

    # calculate range sizes
    range_sizes = [end - start for start, end in ranges]
    range_sizes = torch.tensor(
        [0] + range_sizes,
        dtype=torch.int32,
        device=device,
    )

    # calculate cu_range_sizes
    cu_range_sizes = torch.cumsum(range_sizes, dim=0)

    # calculate row_map from row idx to range idx
    row_map = torch.repeat_interleave(
        torch.arange(0, len(ranges), device=device),
        range_sizes[1:],
        dim=0,
        output_size=total_size,
    )

    ranges = torch.tensor(ranges, device=device)

    unperm_range_gather_kwargs = {
        "ranges": ranges,
        "cu_range_sizes": cu_range_sizes,
        "row_map": row_map,
        "total_size": total_size,
    }

    return unperm_range_gather_kwargs


def _calc_range_reduce_kwargs_from_ranges(
    cu_ranges: NaiveRanges,
    reduce_ranges_list: list[NaiveRanges],
    device: torch.device,
) -> dict:
    input_ranges = []
    output_ranges = []
    range_sizes = []
    total_size = 0
    for (out_start, out_end), reduce_ranges in zip(cu_ranges, reduce_ranges_list):
        for reduce_start, reduce_end in reduce_ranges:
            input_ranges.append([reduce_start, reduce_end])
            output_ranges.append([out_start, out_end])
            range_sizes.append(reduce_end - reduce_start)
            total_size += reduce_end - reduce_start

    input_ranges = torch.tensor(input_ranges, dtype=torch.int32, device=device)
    output_ranges = torch.tensor(output_ranges, dtype=torch.int32, device=device)
    range_sizes = torch.tensor([0] + range_sizes, dtype=torch.int32, device=device)
    cu_range_sizes = torch.cumsum(range_sizes, dim=0)
    row_map = torch.repeat_interleave(
        torch.arange(0, input_ranges.shape[0], device=device),
        range_sizes[1:],
        dim=0,
        output_size=total_size,
    )

    range_reduce_kwargs = {
        "input_ranges": input_ranges,
        "output_ranges": output_ranges,
        "cu_range_sizes": cu_range_sizes,
        "row_map": row_map,
        "total_size": total_size,
    }

    return range_reduce_kwargs


# ------------------        utils for group cast collective       ------------------ #


# TODO: add ut for this meta solver
class HierarchicalGroupCastMetaSolver:
    def __init__(
        self,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        src_index_list: list[int],
        intra_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
        use_a2av_impl: bool = True,
    ):
        self.use_a2av_impl = use_a2av_impl

        # --------   prepare env info  -------- #

        self._prepare_env_info(intra_group, inter_group)

        # ----   build group-cast meta for pre-intra  ---- #

        self._build_group_cast_meta_args_pre_intra(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            return_local_rank=True,
        )

        # ----   build a2a meta for pre-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_pre_intra()

        # ----   build group-cast meta for inter + half post-intra  ---- #

        self._build_group_cast_meta_args_inter_plus_half_post_intra(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            return_local_rank=True,
        )

        # ----   build a2a meta for inter  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_inter()

        # ----   build group-cast meta for post-intra  ---- #

        self._build_group_cast_meta_args_rest_half_post_intra(
            return_local_rank=True,
        )

        # ----   build a2a meta for post-intra  ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_meta_args_post_intra()

        # ----   build a2a post-process fn ---- #

        if self.use_a2av_impl:
            self._build_group_cast_a2a_post_process_fn(
                output_split_size_list=output_split_size_list,
                src_index_list=src_index_list,
            )

    def _prepare_env_info(
        self,
        intra_group: dist.ProcessGroup,
        inter_group: dist.ProcessGroup,
    ):
        self.intra_group = intra_group
        self.inter_group = inter_group

        self.world_size_intra_node = dist.get_world_size(intra_group)
        self.world_size_inter_node = dist.get_world_size(inter_group)

        self.rank = dist.get_rank()
        self.device = torch.cuda.current_device()
        self.world_size = self.world_size_intra_node * self.world_size_inter_node

        self.local_rank_intra_node = self._to_local_rank_intra_node(self.rank)
        self.local_rank_inter_node = self._to_local_rank_inter_node(self.rank)

        self.first_rank_this_intra_node = (
            self.local_rank_inter_node * self.world_size_intra_node
        )
        self.last_rank_this_intra_node = (
            self.first_rank_this_intra_node + self.world_size_intra_node - 1
        )

    def _to_local_rank_intra_node(self, r):
        return r % self.world_size_intra_node

    def _to_local_rank_inter_node(self, r):
        return r // self.world_size_intra_node

    def _is_rank_within_this_intra_node(self, r):
        return self.first_rank_this_intra_node <= r <= self.last_rank_this_intra_node

    def _to_rank_with_same_local_rank_intra_node(self, r):
        return (
            self._to_local_rank_inter_node(r) * self.world_size_intra_node
            + self.local_rank_intra_node
        )

    def _build_group_cast_meta_args_pre_intra(
        self,
        input_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        output_split_size_list: list[int],
        src_index_list: list[int],
        return_local_rank: bool = False,
    ):
        # input split size list does not need to change
        input_split_size_list_pre_intra = input_split_size_list

        # filter dst indices list within this intra node
        dst_indices_list_pre_intra = [
            list(
                map(
                    self._to_local_rank_intra_node
                    if return_local_rank
                    else lambda r: r,
                    filter(
                        self._is_rank_within_this_intra_node,
                        dst_indices,
                    ),
                )
            )
            for dst_indices in dst_indices_list
        ]

        # filter src index list along with output split size list
        # within this intra node
        src_index_list_pre_intra = []
        output_split_size_list_pre_intra = []
        for src_index, output_split_size in zip(src_index_list, output_split_size_list):
            if self._is_rank_within_this_intra_node(src_index):
                src_index_list_pre_intra.append(
                    self._to_local_rank_intra_node(src_index)
                    if return_local_rank
                    else src_index
                )
                output_split_size_list_pre_intra.append(output_split_size)

        # set pre-intra group-cast meta
        self.output_seqlen_pre_intra = sum(output_split_size_list_pre_intra)
        self.input_split_size_list_pre_intra = input_split_size_list_pre_intra
        self.dst_indices_list_pre_intra = dst_indices_list_pre_intra
        self.output_split_size_list_pre_intra = output_split_size_list_pre_intra
        self.src_index_list_pre_intra = src_index_list_pre_intra

    def _build_group_cast_a2a_meta_args_pre_intra(self):
        (
            a2a_input_split_size_pre_intra,
            perm_range_gather_kwargs_pre_intra,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_pre_intra,
            dst_indices_list=self.dst_indices_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )
        (
            a2a_output_split_size_pre_intra,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_pre_intra,
            src_index_list=self.src_index_list_pre_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set pre-intra a2a meta
        self.a2a_input_split_size_pre_intra = a2a_input_split_size_pre_intra
        self.a2a_output_split_size_pre_intra = a2a_output_split_size_pre_intra
        self.perm_range_gather_kwargs_pre_intra = perm_range_gather_kwargs_pre_intra

    def _build_group_cast_meta_args_inter_plus_half_post_intra(
        self,
        input_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        return_local_rank: bool = False,
    ):
        # input split size list does not need to change
        input_split_size_list_inter = input_split_size_list

        # build dst indices list within this inter node
        dst_indices_list_inter = [
            list(
                set(
                    map(
                        self._to_local_rank_inter_node
                        if return_local_rank
                        else lambda r: r,
                        map(
                            self._to_rank_with_same_local_rank_intra_node,
                            filter(
                                lambda r: not self._is_rank_within_this_intra_node(r),
                                dst_indices,
                            ),
                        ),
                    )
                )
            )
            for dst_indices in dst_indices_list
        ]

        # all gather send info per rank within this inter node
        send_info_this_rank_inter = (
            input_split_size_list_inter,
            dst_indices_list_inter,
            dst_indices_list,
        )
        send_info_per_rank_inter: list[
            tuple[list[int], list[list[int]], list[list[int]]]
        ] = [
            None  # type: ignore
        ] * self.world_size_inter_node
        dist.all_gather_object(
            send_info_per_rank_inter,
            send_info_this_rank_inter,
            group=self.inter_group,
        )

        # build naive src index list and output split size list
        # within this inter node
        # as well as preparing group cast input meta args for the third step
        src_index_list_inter = []
        output_split_size_list_inter = []
        dst_indices_list_post_intra = []
        is_this_rank_in_dst_indices_inter = (  # noqa: E731
            lambda dst_indices: (self.local_rank_inter_node in dst_indices)
            if return_local_rank
            else (self.rank in dst_indices)
        )
        for i, (
            input_split_size_list_inter_ith_rank,
            dst_indices_list_inter_ith_rank,
            dst_indices_list_ith_rank,
        ) in enumerate(send_info_per_rank_inter):
            if i == self.local_rank_inter_node:
                continue

            src_index_inter = (
                i
                if return_local_rank
                else (i * self.world_size_intra_node + self.local_rank_intra_node)
            )
            for input_split_size_inter, dst_indices_inter, dst_indices in zip(
                input_split_size_list_inter_ith_rank,
                dst_indices_list_inter_ith_rank,
                dst_indices_list_ith_rank,
            ):
                if is_this_rank_in_dst_indices_inter(dst_indices_inter):
                    # append src index with corr. input split size for the second step
                    src_index_list_inter.append(src_index_inter)
                    output_split_size_list_inter.append(input_split_size_inter)

                    # append filtered dst indices for the third step
                    dst_indices_list_post_intra.append(
                        list(
                            map(
                                self._to_local_rank_intra_node
                                if return_local_rank
                                else lambda r: r,
                                filter(
                                    self._is_rank_within_this_intra_node,
                                    dst_indices,
                                ),
                            )
                        )
                    )

        input_split_size_list_post_intra = output_split_size_list_inter

        # set inter group-cast meta
        self.output_seqlen_inter = sum(output_split_size_list_inter)
        self.input_split_size_list_inter = input_split_size_list_inter
        self.dst_indices_list_inter = dst_indices_list_inter
        self.output_split_size_list_inter = output_split_size_list_inter
        self.src_index_list_inter = src_index_list_inter

        # set half post-intra group-cast meta
        self.input_split_size_list_post_intra = input_split_size_list_post_intra
        self.dst_indices_list_post_intra = dst_indices_list_post_intra

    def _build_group_cast_a2a_meta_args_inter(self):
        (
            a2a_input_split_size_inter,
            perm_range_gather_kwargs_inter,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_inter,
            dst_indices_list=self.dst_indices_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
        )
        (
            a2a_output_split_size_inter,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_inter,
            src_index_list=self.src_index_list_inter,
            world_size=self.world_size_inter_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set inter a2a meta
        self.a2a_input_split_size_inter = a2a_input_split_size_inter
        self.a2a_output_split_size_inter = a2a_output_split_size_inter
        self.perm_range_gather_kwargs_inter = perm_range_gather_kwargs_inter

    def _build_group_cast_meta_args_rest_half_post_intra(
        self,
        return_local_rank: bool = False,
    ):
        # all gather send info per rank within this intra node
        send_info_this_rank_post_intra = (
            self.input_split_size_list_post_intra,
            self.dst_indices_list_post_intra,
        )
        send_info_per_rank_post_intra: list[tuple[list[int], list[list[int]]]] = [
            None  # type: ignore
        ] * self.world_size_intra_node
        dist.all_gather_object(
            send_info_per_rank_post_intra,
            send_info_this_rank_post_intra,
            group=self.intra_group,
        )

        # build naive src index list and output split size list
        # within this intra node
        src_index_list_post_intra = []
        output_split_size_list_post_intra = []
        is_this_rank_in_dst_indices_post_intra = (  # noqa: E731
            lambda dst_indices: (self.local_rank_intra_node in dst_indices)
            if return_local_rank
            else (self.rank in dst_indices)
        )
        for i, (
            input_split_size_list_post_intra_ith_rank,
            dst_indices_list_post_intra_ith_rank,
        ) in enumerate(send_info_per_rank_post_intra):
            src_index_post_intra = (
                i if return_local_rank else (i + self.first_rank_this_intra_node)
            )
            for input_split_size_post_intra, dst_indices_post_intra in zip(
                input_split_size_list_post_intra_ith_rank,
                dst_indices_list_post_intra_ith_rank,
            ):
                if is_this_rank_in_dst_indices_post_intra(dst_indices_post_intra):
                    # append src index with corr. input split size for the second step
                    src_index_list_post_intra.append(src_index_post_intra)
                    output_split_size_list_post_intra.append(
                        input_split_size_post_intra
                    )

        # set post-intra group-cast meta
        self.output_seqlen_post_intra = sum(output_split_size_list_post_intra)
        self.output_split_size_list_post_intra = output_split_size_list_post_intra
        self.src_index_list_post_intra = src_index_list_post_intra

    def _build_group_cast_a2a_meta_args_post_intra(self):
        (
            a2a_input_split_size_post_intra,
            perm_range_gather_kwargs_post_intra,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=self.input_split_size_list_post_intra,
            dst_indices_list=self.dst_indices_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
        )
        (
            a2a_output_split_size_post_intra,
            _,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=self.output_split_size_list_post_intra,
            src_index_list=self.src_index_list_post_intra,
            world_size=self.world_size_intra_node,
            device=self.device,
            calc_unperm_after_a2a_kwargs=False,
        )

        # set post-intra a2a meta
        self.a2a_input_seqlen_post_intra = sum(a2a_input_split_size_post_intra)
        self.a2a_input_split_size_post_intra = a2a_input_split_size_post_intra
        self.a2a_output_split_size_post_intra = a2a_output_split_size_post_intra
        self.perm_range_gather_kwargs_post_intra = perm_range_gather_kwargs_post_intra

    def _build_group_cast_a2a_post_process_fn(
        self,
        output_split_size_list: list[int],
        src_index_list: list[int],
    ):
        rank_list_intra = list(
            range(self.first_rank_this_intra_node, self.last_rank_this_intra_node + 1)
        )
        rank_list_per_inter_wo_this_intra_node = [
            [
                r
                for inter_offset in range(self.world_size_inter_node)
                if not self._is_rank_within_this_intra_node(
                    r := inter_offset * self.world_size_intra_node + intra_offset
                )
            ]
            for intra_offset in range(self.world_size_intra_node)
        ]
        reorder_list = rank_list_intra + list(
            chain(*rank_list_per_inter_wo_this_intra_node)
        )

        _, unperm_after_a2a_kwargs_hier = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            world_size=self.world_size,
            device=self.device,
            reorder_list=reorder_list,
            calc_unperm_after_a2a_kwargs=True,
        )

        # set post-process fn to unperm after a2a
        self.post_process_fn = partial(
            _unpermute_tensor,
            unperm_after_a2a_kwargs=unperm_after_a2a_kwargs_hier,
        )


def _init_hier_group_cast_meta_solver(
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    intra_group: dist.ProcessGroup,
    inter_group: dist.ProcessGroup,
    use_a2av_impl: bool = True,
    **kwargs,
) -> HierarchicalGroupCastMetaSolver:
    if "hier_group_cast_meta_solver" in kwargs:
        # if pre-calculated, directly return
        return kwargs["hier_group_cast_meta_solver"]

    return HierarchicalGroupCastMetaSolver(
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=use_a2av_impl,
    )


@nvtx.instrument_nvtx
def _hier_group_cast_impl_with_a2av(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    assert (
        async_op
    ), "async_op must be True for hierarchical group-cast collective by now"

    intra_group = kwargs.pop("intra_group", None)
    inter_group = kwargs.pop("inter_group", None)
    assert intra_group is not None and inter_group is not None

    side_stream: torch.cuda.Stream = kwargs.pop("side_stream", None)
    assert side_stream is not None

    # --------      get hier-comm meta solver       -------- #

    meta_solver: HierarchicalGroupCastMetaSolver = _init_hier_group_cast_meta_solver(
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        intra_group=intra_group,
        inter_group=inter_group,
        use_a2av_impl=True,
        **kwargs,
    )

    # --------      prepare a2a output buffer for 2nd step       -------- #

    output_other_shape = output_tensor.shape[1:]
    a2a_output_inter = torch.empty(
        size=[meta_solver.output_seqlen_inter, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # --------      prepare a2a input buffer for 2nd step       -------- #

    a2a_input_inter = range_gather(
        input=input_tensor,
        **meta_solver.perm_range_gather_kwargs_inter,
    )

    # --------      apply a2a for 2nd step       -------- #

    with nvtx.add_nvtx_event(
        (
            f"{a2a_output_inter.shape=} | "
            f"{a2a_input_inter.shape=} | "
            f"{meta_solver.a2a_output_split_size_inter=} | "
            f"{meta_solver.a2a_input_split_size_inter=}"
        )
    ):
        work_inter = dist.all_to_all_single(
            output=a2a_output_inter,
            input=a2a_input_inter,
            output_split_sizes=meta_solver.a2a_output_split_size_inter,
            input_split_sizes=meta_solver.a2a_input_split_size_inter,
            group=inter_group,
            async_op=async_op,
        )

    # --------      prepare a2a output buffer for 1st/3rd step     -------- #

    a2a_output_pre_intra, a2a_output_post_intra = torch.split(
        output_tensor,
        [meta_solver.output_seqlen_pre_intra, meta_solver.output_seqlen_post_intra],
        dim=0,
    )

    # --------      prepare a2a input buffer for 1st step     -------- #

    a2a_input_pre_intra = range_gather(
        input=input_tensor,
        **meta_solver.perm_range_gather_kwargs_pre_intra,
    )

    # --------      apply a2a for 1st step       -------- #

    with nvtx.add_nvtx_event(
        (
            f"{a2a_output_pre_intra.shape=} | "
            f"{a2a_input_pre_intra.shape=} | "
            f"{meta_solver.a2a_output_split_size_pre_intra=} | "
            f"{meta_solver.a2a_input_split_size_pre_intra=}"
        )
    ):
        work_pre_intra = dist.all_to_all_single(
            output=a2a_output_pre_intra,
            input=a2a_input_pre_intra,
            output_split_sizes=meta_solver.a2a_output_split_size_pre_intra,
            input_split_sizes=meta_solver.a2a_input_split_size_pre_intra,
            group=intra_group,
            async_op=async_op,
        )

    # --------      prepare a2a input buffer for 3rd step     -------- #

    a2a_input_post_intra = torch.empty(
        size=[meta_solver.a2a_input_seqlen_post_intra, *output_other_shape],
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    side_stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(side_stream):
        # --------      prepare a2a input buffer for 3rd step       -------- #

        work_inter.wait()
        a2a_input_post_intra_ = range_gather(
            input=a2a_output_inter,
            output=a2a_input_post_intra,
            **meta_solver.perm_range_gather_kwargs_post_intra,
        )

        # --------      apply a2a for 3rd step       -------- #

        with nvtx.add_nvtx_event(
            (
                f"{a2a_output_post_intra.shape=} | "
                f"{a2a_input_post_intra.shape=} | "
                f"{meta_solver.a2a_output_split_size_post_intra=} | "
                f"{meta_solver.a2a_input_split_size_post_intra=}"
            )
        ):
            work_post_intra = dist.all_to_all_single(
                output=a2a_output_post_intra,
                input=a2a_input_post_intra_,
                output_split_sizes=meta_solver.a2a_output_split_size_post_intra,
                input_split_sizes=meta_solver.a2a_input_split_size_post_intra,
                group=intra_group,
                async_op=async_op,
            )
        work_post_intra.wait()

    a2a_output_inter.record_stream(side_stream)
    a2a_input_post_intra.record_stream(side_stream)
    a2a_output_post_intra.record_stream(side_stream)

    # ---------    prepare work with post-process fn    --------- #

    work_with_post_process_fn = WorkWithPostProcessFn(
        work=[work_pre_intra, side_stream],
        post_process_fn=meta_solver.post_process_fn,
        sync=not async_op,
    )

    return work_with_post_process_fn


def _group_cast_impl_with_batch_p2p(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.Backend = None,
    async_op: bool = False,
    **kwargs,
):
    input_list = input.split(input_split_size_list, dim=0)
    output_list = output.split(output_split_size_list, dim=0)

    p2p_op_list = []

    # send
    for input_split_idx in range(len(input_split_size_list)):
        for dst_rank in dst_indices_list[input_split_idx]:
            p2p_op_list.append(
                dist.P2POp(
                    op=dist.isend,
                    tensor=input_list[input_split_idx],
                    peer=dst_rank,
                    group=group,
                )
            )
    # recv
    for output_split_idx in range(len(output_split_size_list)):
        src_rank = src_index_list[output_split_idx]
        p2p_op_list.append(
            dist.P2POp(
                op=dist.irecv,
                tensor=output_list[output_split_idx],
                peer=src_rank,
                group=group,
            )
        )

    work_list = dist.batch_isend_irecv(p2p_op_list)

    return WorkWithPostProcessFn(
        work=work_list,
        post_process_fn=lambda x: x,
        sync=not async_op,
    )


def sanity_check_for_group_cast_meta_args_per_rank(
    input_split_size_list_per_rank: list[list[int]],
    output_split_size_list_per_rank: list[list[int]],
    dst_indices_list_per_rank: list[list[list[int]]],
    src_index_list_per_rank: list[list[int]],
    world_size: int,
    check_nccl_send_recv: bool = False,
) -> None:
    for rank in range(world_size):
        # sanity check for shape
        input_split_size_list = input_split_size_list_per_rank[rank]
        output_split_size_list = output_split_size_list_per_rank[rank]
        dst_indices_list = dst_indices_list_per_rank[rank]
        src_index_list = src_index_list_per_rank[rank]
        assert len(input_split_size_list) == len(dst_indices_list), (
            f"input_split_size_list and dst_indices_list should have the same length, "
            f"but got {len(input_split_size_list)=} and {len(dst_indices_list)=}"
        )
        assert len(output_split_size_list) == len(src_index_list), (
            f"output_split_size_list and src_index_list should have the same length, "
            f"but got {len(output_split_size_list)=} and {len(src_index_list)=}"
        )

        # sanity check for rank value
        assert all(
            0 <= dst_rank < world_size for dst_rank in chain(*dst_indices_list)
        ), (
            f"dst_indices_list should contain ranks in [0, {world_size - 1}], "
            f"but got {dst_indices_list=}"
        )
        assert all(0 <= src_rank < world_size for src_rank in src_index_list), (
            f"src_index_list should contain ranks in [0, {world_size - 1}], "
            f"but got {src_index_list=}"
        )

    # sanity check for dst rank order and unique
    for dst_indices in dst_indices_list:
        assert dst_indices == sorted(
            list(set(dst_indices))
        ), f"dst_indices should be sorted and unique, but got {dst_indices_list=}"

    # sanity check for nccl send/recv consistent number of calls
    if check_nccl_send_recv:
        # num_send[src_rank*world_size + dst_rank]: the number of nccl send calls launched by src_rank for dst_rank
        num_send_list: list[int] = [0] * world_size**2
        # num_recv[src_rank*world_size + dst_rank]: the number of nccl recv calls launched by dst_rank for src_rank
        num_recv_list: list[int] = [0] * world_size**2

        for rank in range(world_size):
            src_rank = rank
            dst_indices_list = dst_indices_list_per_rank[src_rank]
            for dst_rank in chain(*dst_indices_list):
                num_send_list[src_rank * world_size + dst_rank] += 1

            dst_rank = rank
            src_index_list = src_index_list_per_rank[dst_rank]
            for src_rank in src_index_list:
                num_recv_list[src_rank * world_size + dst_rank] += 1

        _sanity_check_nccl_send_recv(
            num_send_list=num_send_list,
            num_recv_list=num_recv_list,
            world_size=world_size,
        )


@nvtx.instrument_nvtx
def _unpermute_tensor(
    tensor: torch.Tensor,
    unperm_after_a2a_kwargs: dict,
) -> torch.Tensor:
    """unpermute a2a output to output
    as a post-processing func for group_cast_collective
    """

    return range_gather(
        input=tensor,
        **unperm_after_a2a_kwargs,
    )


@nvtx.instrument_nvtx
def _calc_group_cast_a2a_input_meta_args(
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
    device: torch.device,
) -> tuple[list[int], dict]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)

    a2a_input_size_ranges_with_rank: RangesWithRank = sorted(
        list(
            chain(
                *[
                    [(input_size_ranges[i], dst_rank) for dst_rank in dst_indices]
                    for i, dst_indices in enumerate(dst_indices_list)
                ]
            )
        ),
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in a2a_input_size_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    perm_range_gather_kwargs = _calc_range_gather_kwargs_from_ranges_with_rank(
        a2a_input_size_ranges_with_rank=a2a_input_size_ranges_with_rank,
        device=device,
    )

    return (
        a2a_input_split_size,
        perm_range_gather_kwargs,
    )


def _calc_group_cast_a2a_input_args(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int]]:
    # -----     group_cast_a2a_input meta args     ----- #

    # check if pre-calculated
    a2a_input_split_size = kwargs.get("a2a_input_split_size", None)
    perm_before_a2a_kwargs = kwargs.get("perm_before_a2a_kwargs", None)

    if a2a_input_split_size is None or perm_before_a2a_kwargs is None:
        (
            a2a_input_split_size,
            perm_before_a2a_kwargs,
        ) = _calc_group_cast_a2a_input_meta_args(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            world_size=world_size,
            device=input.device,
        )

    # -----     group_cast_a2a_input tensor args     ----- #

    a2a_input = range_gather(
        input=input,
        **perm_before_a2a_kwargs,
    )

    return a2a_input, a2a_input_split_size


@nvtx.instrument_nvtx
def _calc_group_cast_a2a_output_meta_args(
    output_split_size_list: list[int],
    src_index_list: list[int],
    world_size: int,
    device: torch.device,
    reorder_list: list[int] | None = None,
    calc_unperm_after_a2a_kwargs: bool = True,
) -> tuple[list[int], dict]:
    a2a_output_split_size_per_rank: list[list[int]] = [[] for _ in range(world_size)]
    a2a_output_permute_index_list_per_rank: list[list[int]] = [
        [] for _ in range(world_size)
    ]
    for i, src_index in enumerate(src_index_list):
        a2a_output_split_size_per_rank[src_index].append(output_split_size_list[i])
        a2a_output_permute_index_list_per_rank[src_index].append(i)

    if reorder_list is not None:
        if magi_attention.is_sanity_check_enable():
            assert sorted(reorder_list) == list(range(world_size)), (
                "The reorder list must be a permutation of [0, 1, ..., world_size-1] if not None, "
                f"but got {reorder_list=} when {world_size=}"
            )
        a2a_output_split_size_per_rank = [
            a2a_output_split_size_per_rank[i] for i in reorder_list
        ]
        a2a_output_permute_index_list_per_rank = [
            a2a_output_permute_index_list_per_rank[i] for i in reorder_list
        ]

    a2a_output_split_size = [sum(x) for x in a2a_output_split_size_per_rank]
    a2a_output_tensor_size_list = list(chain(*a2a_output_split_size_per_rank))
    a2a_output_permute_index_list = list(chain(*a2a_output_permute_index_list_per_rank))
    a2a_output_unpermute_index_list = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )

    # ---------    calc unperm after a2a kwargs     --------- #
    if calc_unperm_after_a2a_kwargs:
        unperm_range_gather_kwargs = (
            _calc_unperm_range_gather_kwargs_from_split_size_list(
                split_size_list=a2a_output_tensor_size_list,
                unpermute_index_list=a2a_output_unpermute_index_list,
                device=device,
            )
        )
    else:
        unperm_range_gather_kwargs = {}

    return (
        a2a_output_split_size,
        unperm_range_gather_kwargs,
    )


def _calc_group_cast_a2a_output_args(
    output: torch.Tensor,
    output_split_size_list: list[int],
    src_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int], dict]:
    # -----     group_cast_a2a_output meta args     ----- #

    # check if pre-calculated
    a2a_output_split_size = kwargs.get("a2a_output_split_size", None)
    unperm_after_a2a_kwargs = kwargs.get("unperm_after_a2a_kwargs", None)
    if a2a_output_split_size is None or unperm_after_a2a_kwargs is None:
        (
            a2a_output_split_size,
            unperm_after_a2a_kwargs,
        ) = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            world_size=world_size,
            device=output.device,
        )

    # -----     group_cast_a2a_output tensor args     ----- #

    a2a_output = output

    return (
        a2a_output,
        a2a_output_split_size,
        unperm_after_a2a_kwargs,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def _calc_group_cast_a2a_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    Callable[[torch.Tensor], torch.Tensor],
]:
    # ---------    calc a2a_input_split_size and a2a input     --------- #

    a2a_input, a2a_input_split_size = _calc_group_cast_a2a_input_args(
        input=input,
        input_split_size_list=input_split_size_list,
        dst_indices_list=dst_indices_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    calc a2a_output_split_size and a2a output     --------- #

    (
        a2a_output,
        a2a_output_split_size,
        unperm_after_a2a_kwargs,
    ) = _calc_group_cast_a2a_output_args(
        output=output,
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    prepare post-process fn    --------- #

    post_process_fn = partial(
        _unpermute_tensor,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs,
    )

    # DE-BUG
    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_cast_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_indices_list=}, {src_index_list=}, "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}, "
    )

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )


# ------------------        utils for group reduce collective       ------------------ #


def sanity_check_for_group_reduce_meta_args_per_rank(
    input_split_size_list_per_rank: list[list[int]],
    output_split_size_list_per_rank: list[list[int]],
    dst_index_list_per_rank: list[list[int]],
    src_indices_list_per_rank: list[list[list[int]]],
    world_size: int,
    check_nccl_send_recv: bool = False,
) -> None:
    for rank in range(world_size):
        # sanity check for shape
        input_split_size_list = input_split_size_list_per_rank[rank]
        output_split_size_list = output_split_size_list_per_rank[rank]
        dst_index_list = dst_index_list_per_rank[rank]
        src_indices_list = src_indices_list_per_rank[rank]
        assert len(input_split_size_list) == len(dst_index_list), (
            f"input_split_size_list and dst_index_list should have the same length, "
            f"but got {len(input_split_size_list)=} and {len(dst_index_list)=}"
        )
        assert len(output_split_size_list) == len(src_indices_list), (
            f"output_split_size_list and src_indices_list should have the same length, "
            f"but got {len(output_split_size_list)=} and {len(src_indices_list)=}"
        )

        # sanity check for rank value
        assert all(0 <= dst_rank < world_size for dst_rank in dst_index_list), (
            f"dst_index_list should contain ranks in [0, {world_size - 1}], "
            f"but got {dst_index_list=}"
        )
        assert all(
            0 <= src_rank < world_size for src_rank in chain(*src_indices_list)
        ), (
            f"src_indices_list should contain ranks in [0, {world_size - 1}], "
            f"but got {src_indices_list=}"
        )

    # sanity check for src rank order and unique
    for src_indices in src_indices_list:
        assert src_indices == sorted(
            list(set(src_indices))
        ), f"src_indices should be sorted and unique, but got {src_indices_list=}"

    # sanity check for nccl send/recv consistent number of calls
    if check_nccl_send_recv:
        # num_send[src_rank*world_size + dst_rank]: the number of nccl send calls launched by src_rank for dst_rank
        num_send_list: list[int] = [0] * world_size**2
        # num_recv[src_rank*world_size + dst_rank]: the number of nccl recv calls launched by dst_rank for src_rank
        num_recv_list: list[int] = [0] * world_size**2

        for rank in range(world_size):
            src_rank = rank
            dst_index_list = dst_index_list_per_rank[src_rank]
            for dst_rank in dst_index_list:
                num_send_list[src_rank * world_size + dst_rank] += 1

            dst_rank = rank
            src_indices_list = src_indices_list_per_rank[dst_rank]
            for src_rank in chain(*src_indices_list):
                num_recv_list[src_rank * world_size + dst_rank] += 1

        _sanity_check_nccl_send_recv(
            num_send_list=num_send_list,
            num_recv_list=num_recv_list,
            world_size=world_size,
        )


@nvtx.instrument_nvtx
def _reduce_to_tensor(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    range_reduce_kwargs: dict,
) -> torch.Tensor:
    """sum-reduce a2a output to output
    as a post-processing func for group_reduce_collective
    """

    output = range_reduce(
        input=a2a_output,
        output=output,
        **range_reduce_kwargs,
    )

    return output


@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_input_meta_args(
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
    device: torch.device,
) -> tuple[list[int], dict]:
    input_size_ranges = _seqlens2curanges(input_split_size_list)
    a2a_input_size_ranges_with_rank: RangesWithRank = sorted(
        [(input_size_ranges[i], dst_rank) for i, dst_rank in enumerate(dst_index_list)],
        key=lambda x: x[1],
    )

    a2a_input_split_size_dict: dict[int, int] = defaultdict(int)
    for (start, end), rank in a2a_input_size_ranges_with_rank:
        a2a_input_split_size_dict[rank] += end - start
    a2a_input_split_size = [
        a2a_input_split_size_dict[rank] for rank in range(world_size)
    ]

    perm_range_gather_kwargs = _calc_range_gather_kwargs_from_ranges_with_rank(
        a2a_input_size_ranges_with_rank=a2a_input_size_ranges_with_rank,
        device=device,
    )

    return a2a_input_split_size, perm_range_gather_kwargs


def _calc_group_reduce_a2a_input_args(
    input: torch.Tensor,
    input_split_size_list: list[int],
    dst_index_list: list[int],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int]]:
    # -----     group_reduce_a2a_input meta args     ----- #

    # check if pre-calculated
    a2a_input_split_size = kwargs.get("a2a_input_split_size", None)
    perm_before_a2a_kwargs = kwargs.get("perm_before_a2a_kwargs", None)

    if perm_before_a2a_kwargs is None or a2a_input_split_size is None:
        (
            a2a_input_split_size,
            perm_before_a2a_kwargs,
        ) = _calc_group_reduce_a2a_input_meta_args(
            input_split_size_list=input_split_size_list,
            dst_index_list=dst_index_list,
            world_size=world_size,
            device=input.device,
        )

    a2a_input = range_gather(
        input=input,
        **perm_before_a2a_kwargs,
    )

    return a2a_input, a2a_input_split_size


@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_output_meta_args(
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    device: torch.device,
) -> tuple[list[int], dict]:
    # phase1 meta
    a2a_output_split_size = [0 for _ in range(world_size)]
    size_src_index_i_list = []
    idx = 0
    for output_split_size, src_indices in zip(output_split_size_list, src_indices_list):
        for src_index in src_indices:
            a2a_output_split_size[src_index] += output_split_size
            size_src_index_i_list.append((output_split_size, src_index, idx))
            idx += 1
    size_src_index_i_list.sort(key=lambda x: x[1])
    a2a_output_permute_index_list = [x[2] for x in size_src_index_i_list]
    a2a_output_unpermute_index_list = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )
    a2a_output_tensor_size_list = [x[0] for x in size_src_index_i_list]
    num_src_list = [len(src_indices) for src_indices in src_indices_list]

    # phase2 meta
    a2a_output_size_ranges = _seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = _seqlens2curanges(output_split_size_list)
    cum_src_ranges = _seqlens2curanges(num_src_list)
    a2a_output_reduce_ranges_list: list[NaiveRanges] = []
    for start, end in cum_src_ranges:
        a2a_output_reduce_ranges_list.append(
            [
                a2a_output_size_ranges[index]
                for index in a2a_output_unpermute_index_list[start:end]
            ]
        )

    # calc range_reduce kwargs
    range_reduce_kwargs = _calc_range_reduce_kwargs_from_ranges(
        cu_ranges=output_size_ranges,
        reduce_ranges_list=a2a_output_reduce_ranges_list,
        device=device,
    )

    return (
        a2a_output_split_size,
        range_reduce_kwargs,
    )


def _calc_group_reduce_a2a_output_args(
    output: torch.Tensor,
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[torch.Tensor, list[int], dict]:
    # -----     group_reduce_a2a_output meta args     ----- #

    # check if pre-calculated
    a2a_output_split_size = kwargs.get("a2a_output_split_size", None)
    range_reduce_kwargs = kwargs.get("range_reduce_kwargs", None)

    if a2a_output_split_size is None or range_reduce_kwargs is None:
        (
            a2a_output_split_size,
            range_reduce_kwargs,
        ) = _calc_group_reduce_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_indices_list=src_indices_list,
            world_size=world_size,
            device=output.device,
        )

    # -----     group_reduce_a2a_output tensor args     ----- #

    a2a_output = torch.empty(
        [sum(a2a_output_split_size), *output.shape[1:]],
        device=output.device,
        dtype=output.dtype,
    )

    return (
        a2a_output,
        a2a_output_split_size,
        range_reduce_kwargs,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    **kwargs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[int],
    Callable[[torch.Tensor], torch.Tensor],
]:
    # ---------    calc a2a_input_split_size and a2a input     --------- #

    a2a_input, a2a_input_split_size = _calc_group_reduce_a2a_input_args(
        input=input,
        input_split_size_list=input_split_size_list,
        dst_index_list=dst_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    calc a2a_output_split_size and a2a output     --------- #

    (
        a2a_output,
        a2a_output_split_size,
        range_reduce_kwargs,
    ) = _calc_group_reduce_a2a_output_args(
        output=output,
        output_split_size_list=output_split_size_list,
        src_indices_list=src_indices_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    prepare post process fn     --------- #

    post_process_fn = partial(
        _reduce_to_tensor,
        a2a_output=a2a_output,
        range_reduce_kwargs=range_reduce_kwargs,
    )

    # DE-BUG
    logger.debug(
        f"RANK {dist.get_rank()}:: args for group_reduce_collective: {input.shape=}, {output.shape=}, "
        f"{input_split_size_list=}, {output_split_size_list=}, {dst_index_list=}, {src_indices_list=}. "
        f"args: {a2a_input.shape=}, {a2a_output.shape=}, {a2a_output_split_size=}, {a2a_input_split_size=}, "
    )

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )
