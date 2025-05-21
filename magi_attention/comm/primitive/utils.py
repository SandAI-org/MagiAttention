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
    "_calc_group_cast_a2a_args",
    "_calc_group_reduce_a2a_args",
    "_calc_group_cast_a2a_input_meta_args",
    "_calc_group_cast_a2a_output_meta_args",
    "_calc_group_reduce_a2a_input_meta_args",
    "_calc_group_reduce_a2a_output_meta_args",
    "_trans_with_dim0",
    "_get_dims_as_trans_with_dim0",
]


def _seqlens2curanges(
    seqlens: list[int],
) -> NaiveRanges:
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

    # calculate cu_range_sizes
    cu_range_sizes = torch.cumsum(
        torch.tensor(
            [0] + range_sizes,
            dtype=torch.int32,
        ),
        dim=0,
    )

    # convert to tensor
    ranges = torch.tensor(
        ranges,
    )

    perm_range_gather_kwargs = {
        "ranges": ranges.to(device),
        "cu_range_sizes": cu_range_sizes.to(device),
        "total_size": total_size,
    }

    return perm_range_gather_kwargs


# ------------------        utils for group cast collective       ------------------ #


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
        # calculate the output size from a2a_output_split_size_list
        total_size = sum(a2a_output_tensor_size_list)

        # calculate each range's start and end
        ranges = torch.tensor(
            [0] + a2a_output_tensor_size_list,
            dtype=torch.int32,
        )
        ranges = torch.cumsum(ranges, dim=0)
        ranges = torch.cat(
            [ranges[0:-1, None], ranges[1:, None]],
            dim=1,
        )

        # re-order ranges by a2a_output_unpermute_index_list
        # this means the ranges are in the order of the output tensor
        # convert to list for easy indexing
        ranges = ranges.tolist()
        ranges = [ranges[i] for i in a2a_output_unpermute_index_list]

        # calculate cu_range_sizes
        cu_range_sizes = torch.cumsum(
            torch.tensor(
                [0] + [end - start for start, end in ranges],
                dtype=torch.int32,
            ),
            dim=0,
        )

        # convert to tensor again
        ranges = torch.tensor(
            ranges,
        )

        unperm_range_gather_kwargs = {
            "ranges": ranges.to(device),
            "cu_range_sizes": cu_range_sizes.to(device),
            "total_size": total_size,
        }
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


# TODO: fuse this kernel in the future
# FIXME: if using torch.compile, it's fused incompletely w/o performance gain
# what's worse, the re-compilation in online exps would hang the comm
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
    input_ranges = []
    output_ranges = []
    cu_range_sizes = [0]
    total_size = 0
    for (out_start, out_end), reduce_ranges in zip(
        output_size_ranges, a2a_output_reduce_ranges_list
    ):
        for reduce_start, reduce_end in reduce_ranges:
            input_ranges.append([reduce_start, reduce_end])
            output_ranges.append([out_start, out_end])
            cu_range_sizes.append(reduce_end - reduce_start)
            total_size += reduce_end - reduce_start

    input_ranges = torch.tensor(input_ranges, dtype=torch.int32)
    output_ranges = torch.tensor(output_ranges, dtype=torch.int32)
    cu_range_sizes = torch.tensor(cu_range_sizes, dtype=torch.int32)
    cu_range_sizes = torch.cumsum(cu_range_sizes, dim=0)

    range_reduce_kwargs = {
        "input_ranges": input_ranges.to(device),
        "output_ranges": output_ranges.to(device),
        "cu_range_sizes": cu_range_sizes.to(device),
        "total_size": total_size,
    }

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


# ------------------        utils for all-gather-v       ------------------ #


def _trans_with_dim0(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    is_first_dim = dim == 0 or (dim == -1 and len(x.shape) == 1)

    if not is_first_dim:
        x = x.transpose(0, dim)
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _get_dims_as_trans_with_dim0(
    x_shape: list[int],
    dim: int = 0,
) -> tuple[int, list[int]]:
    shape_len = len(x_shape)
    assert dim == -1 or 0 <= dim < len(
        x_shape
    ), f"dim should be in [0, {shape_len - 1}) or -1"

    this_dim = x_shape[dim]

    other_dims = x_shape.copy()
    other_dims[0] = this_dim
    other_dims[dim] = x_shape[0]
    other_dims = other_dims[1:]

    return this_dim, other_dims


# ------------------        utils for scatter-v       ------------------ #


# ------------------        utils for hier-comm       ------------------ #


def _prepare_meta_for_group_cast_collective_hier(
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    intra_group: dist.ProcessGroup,
    inter_group: dist.ProcessGroup,
    **kwargs,
):
    # check if pre-calculated
    if kwargs.get("hier_comm_meta_kwargs", None) is not None:
        return kwargs["hier_comm_meta_kwargs"]

    # --------------      prepare env info       -------------- #

    world_size_intra_node = dist.get_world_size(intra_group)
    world_size_inter_node = dist.get_world_size(inter_group)

    rank = dist.get_rank()
    device = torch.cuda.current_device()
    world_size = world_size_intra_node * world_size_inter_node

    to_local_rank_intra_node = lambda r: r % world_size_intra_node  # noqa: E731
    local_rank_intra_node = to_local_rank_intra_node(rank)

    to_local_rank_inter_node = lambda r: r // world_size_intra_node  # noqa: E731
    local_rank_inter_node = to_local_rank_inter_node(rank)

    first_rank_this_intra_node = local_rank_inter_node * world_size_intra_node
    last_rank_this_intra_node = first_rank_this_intra_node + world_size_intra_node - 1
    is_rank_within_this_intra_node = (  # noqa: E731
        lambda r: first_rank_this_intra_node <= r <= last_rank_this_intra_node
    )

    to_rank_with_same_local_rank_intra_node = (  # noqa: E731
        lambda r: to_local_rank_inter_node(r) * world_size_intra_node
        + local_rank_intra_node
    )

    # --------------      build group-cast meta for 1st step       -------------- #

    def build_group_cast_meta_args_pre_intra(
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
                    to_local_rank_intra_node if return_local_rank else lambda r: r,
                    filter(
                        is_rank_within_this_intra_node,
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
            if is_rank_within_this_intra_node(src_index):
                src_index_list_pre_intra.append(
                    to_local_rank_intra_node(src_index)
                    if return_local_rank
                    else src_index
                )
                output_split_size_list_pre_intra.append(output_split_size)

        return (
            input_split_size_list_pre_intra,
            dst_indices_list_pre_intra,
            output_split_size_list_pre_intra,
            src_index_list_pre_intra,
        )

    (
        input_split_size_list_pre_intra,
        dst_indices_list_pre_intra,
        output_split_size_list_pre_intra,
        src_index_list_pre_intra,
    ) = build_group_cast_meta_args_pre_intra(
        input_split_size_list,
        dst_indices_list,
        output_split_size_list,
        src_index_list,
        return_local_rank=True,
    )

    # --------------      get a2a meta for 1st step       -------------- #

    (
        a2a_input_split_size_pre_intra,
        perm_range_gather_kwargs_pre_intra,
    ) = _calc_group_cast_a2a_input_meta_args(
        input_split_size_list=input_split_size_list_pre_intra,
        dst_indices_list=dst_indices_list_pre_intra,
        world_size=world_size_intra_node,
        device=device,
    )
    (
        a2a_output_split_size_pre_intra,
        unperm_after_a2a_kwargs_pre_intra,
    ) = _calc_group_cast_a2a_output_meta_args(
        output_split_size_list=output_split_size_list_pre_intra,
        src_index_list=src_index_list_pre_intra,
        world_size=world_size_intra_node,
        device=device,
        calc_unperm_after_a2a_kwargs=False,
    )

    # --------------      build group-cast meta for 2nd step       -------------- #

    def build_group_cast_meta_args_inter_plus_half_post_intra(
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
                        to_local_rank_inter_node if return_local_rank else lambda r: r,
                        map(
                            to_rank_with_same_local_rank_intra_node,
                            filter(
                                lambda r: not is_rank_within_this_intra_node(r),
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
        ] * world_size_inter_node
        dist.all_gather_object(
            send_info_per_rank_inter,
            send_info_this_rank_inter,
            group=inter_group,
        )

        # build naive src index list and output split size list
        # within this inter node
        # as well as preparing group cast input meta args for the third step
        src_index_list_inter = []
        output_split_size_list_inter = []
        dst_indices_list_post_intra = []
        is_this_rank_in_dst_indices_inter = (  # noqa: E731
            lambda dst_indices: (local_rank_inter_node in dst_indices)
            if return_local_rank
            else (rank in dst_indices)
        )
        for i, (
            input_split_size_list_inter_ith_rank,
            dst_indices_list_inter_ith_rank,
            dst_indices_list_ith_rank,
        ) in enumerate(send_info_per_rank_inter):
            if i == local_rank_inter_node:
                continue

            src_index_inter = (
                i
                if return_local_rank
                else (i * world_size_intra_node + local_rank_intra_node)
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
                                to_local_rank_intra_node
                                if return_local_rank
                                else lambda r: r,
                                filter(
                                    is_rank_within_this_intra_node,
                                    dst_indices,
                                ),
                            )
                        )
                    )

        input_split_size_list_post_intra = output_split_size_list_inter

        return (
            input_split_size_list_inter,
            dst_indices_list_inter,
            output_split_size_list_inter,
            src_index_list_inter,
            input_split_size_list_post_intra,
            dst_indices_list_post_intra,
        )

    (
        input_split_size_list_inter,
        dst_indices_list_inter,
        output_split_size_list_inter,
        src_index_list_inter,
        input_split_size_list_post_intra,
        dst_indices_list_post_intra,
    ) = build_group_cast_meta_args_inter_plus_half_post_intra(
        input_split_size_list,
        dst_indices_list,
        return_local_rank=True,
    )

    # --------------      get a2a meta for 2nd step       -------------- #

    (
        a2a_input_split_size_inter,
        perm_range_gather_kwargs_inter,
    ) = _calc_group_cast_a2a_input_meta_args(
        input_split_size_list=input_split_size_list_inter,
        dst_indices_list=dst_indices_list_inter,
        world_size=world_size_inter_node,
        device=device,
    )
    (
        a2a_output_split_size_inter,
        unperm_after_a2a_kwargs_inter,
    ) = _calc_group_cast_a2a_output_meta_args(
        output_split_size_list=output_split_size_list_inter,
        src_index_list=src_index_list_inter,
        world_size=world_size_inter_node,
        device=device,
        calc_unperm_after_a2a_kwargs=False,
    )

    # --------------      build group-cast meta for 3rd step       -------------- #

    def build_group_cast_meta_args_rest_half_post_intra(
        input_split_size_list_post_intra: list[list[int]],
        dst_indices_list_post_intra: list[list[int]],
        return_local_rank: bool = False,
    ):
        # all gather send info per rank within this intra node
        send_info_this_rank_post_intra = (
            input_split_size_list_post_intra,
            dst_indices_list_post_intra,
        )
        send_info_per_rank_post_intra: list[tuple[list[int], list[list[int]]]] = [
            None  # type: ignore
        ] * world_size_intra_node
        dist.all_gather_object(
            send_info_per_rank_post_intra,
            send_info_this_rank_post_intra,
            group=intra_group,
        )

        # build naive src index list and output split size list
        # within this intra node
        src_index_list_post_intra = []
        output_split_size_list_post_intra = []
        is_this_rank_in_dst_indices_post_intra = (  # noqa: E731
            lambda dst_indices: (local_rank_intra_node in dst_indices)
            if return_local_rank
            else (rank in dst_indices)
        )
        for i, (
            input_split_size_list_post_intra_ith_rank,
            dst_indices_list_post_intra_ith_rank,
        ) in enumerate(send_info_per_rank_post_intra):
            src_index_post_intra = (
                i if return_local_rank else (i + first_rank_this_intra_node)
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

        return (
            output_split_size_list_post_intra,
            src_index_list_post_intra,
        )

    (
        output_split_size_list_post_intra,
        src_index_list_post_intra,
    ) = build_group_cast_meta_args_rest_half_post_intra(
        input_split_size_list_post_intra,
        dst_indices_list_post_intra,
        return_local_rank=True,
    )

    # --------------      get a2a meta for 3rd step       -------------- #

    (
        a2a_input_split_size_post_intra,
        perm_range_gather_kwargs_post_intra,
    ) = _calc_group_cast_a2a_input_meta_args(
        input_split_size_list=input_split_size_list_post_intra,
        dst_indices_list=dst_indices_list_post_intra,
        world_size=world_size_intra_node,
        device=device,
    )
    (
        a2a_output_split_size_post_intra,
        unperm_after_a2a_kwargs_post_intra,
    ) = _calc_group_cast_a2a_output_meta_args(
        output_split_size_list=output_split_size_list_post_intra,
        src_index_list=src_index_list_post_intra,
        world_size=world_size_intra_node,
        device=device,
        calc_unperm_after_a2a_kwargs=False,
    )

    # --------------      get post-process fn for 4th step       -------------- #

    def get_unperm_after_a2a_kwargs_hier(
        output_split_size_list: list[int],
        src_index_list: list[int],
        world_size: int,
        device: torch.device,
    ):
        rank_list_intra = list(
            range(first_rank_this_intra_node, last_rank_this_intra_node + 1)
        )
        rank_list_per_inter_wo_this_intra_node = [
            [
                r
                for inter_offset in range(world_size_inter_node)
                if not is_rank_within_this_intra_node(
                    r := inter_offset * world_size_intra_node + intra_offset
                )
            ]
            for intra_offset in range(world_size_intra_node)
        ]
        reorder_list = rank_list_intra + list(
            chain(*rank_list_per_inter_wo_this_intra_node)
        )

        _, unperm_after_a2a_kwargs_hier = _calc_group_cast_a2a_output_meta_args(
            output_split_size_list=output_split_size_list,
            src_index_list=src_index_list,
            world_size=world_size,
            device=device,
            reorder_list=reorder_list,
            calc_unperm_after_a2a_kwargs=True,
        )

        return unperm_after_a2a_kwargs_hier

    unperm_after_a2a_kwargs_hier = get_unperm_after_a2a_kwargs_hier(
        output_split_size_list=output_split_size_list,
        src_index_list=src_index_list,
        world_size=world_size,
        device=device,
    )

    post_process_fn = partial(
        _unpermute_tensor,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs_hier,
    )

    return (
        # for pre intra
        input_split_size_list_pre_intra,
        output_split_size_list_pre_intra,
        dst_indices_list_pre_intra,
        src_index_list_pre_intra,
        a2a_input_split_size_pre_intra,
        a2a_output_split_size_pre_intra,
        perm_range_gather_kwargs_pre_intra,
        unperm_after_a2a_kwargs_pre_intra,
        # for inter
        input_split_size_list_inter,
        output_split_size_list_inter,
        dst_indices_list_inter,
        src_index_list_inter,
        a2a_input_split_size_inter,
        a2a_output_split_size_inter,
        perm_range_gather_kwargs_inter,
        unperm_after_a2a_kwargs_inter,
        # for post intra
        input_split_size_list_post_intra,
        output_split_size_list_post_intra,
        dst_indices_list_post_intra,
        src_index_list_post_intra,
        a2a_input_split_size_post_intra,
        a2a_output_split_size_post_intra,
        perm_range_gather_kwargs_post_intra,
        unperm_after_a2a_kwargs_post_intra,
        # for post process
        post_process_fn,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def _group_cast_collective_hier(
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
    intra_group = kwargs.pop("intra_group", None)
    inter_group = kwargs.pop("inter_group", None)
    assert intra_group is not None and inter_group is not None

    world_size_intra_node = dist.get_world_size(intra_group)
    world_size_inter_node = dist.get_world_size(inter_group)

    # --------------------------------------------------------------------
    # hier comm meta
    # --------------------------------------------------------------------

    (
        # for pre intra
        input_split_size_list_pre_intra,
        output_split_size_list_pre_intra,
        dst_indices_list_pre_intra,
        src_index_list_pre_intra,
        a2a_input_split_size_pre_intra,
        a2a_output_split_size_pre_intra,
        perm_range_gather_kwargs_pre_intra,
        unperm_after_a2a_kwargs_pre_intra,
        # for inter
        input_split_size_list_inter,
        output_split_size_list_inter,
        dst_indices_list_inter,
        src_index_list_inter,
        a2a_input_split_size_inter,
        a2a_output_split_size_inter,
        perm_range_gather_kwargs_inter,
        unperm_after_a2a_kwargs_inter,
        # for post intra
        input_split_size_list_post_intra,
        output_split_size_list_post_intra,
        dst_indices_list_post_intra,
        src_index_list_post_intra,
        a2a_input_split_size_post_intra,
        a2a_output_split_size_post_intra,
        perm_range_gather_kwargs_post_intra,
        unperm_after_a2a_kwargs_post_intra,
        # for post process
        post_process_fn,
    ) = _prepare_meta_for_group_cast_collective_hier(
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        intra_group=intra_group,
        inter_group=inter_group,
        **kwargs,
    )

    # --------------------------------------------------------------------
    # hier comm tensor for 1st and 2nd steps
    # --------------------------------------------------------------------

    # --------------      pre-init buffer for all steps       -------------- #
    output_tensor_pre_intra, output_tensor_post_intra = torch.split(
        output_tensor,
        [sum(output_split_size_list_pre_intra), sum(output_split_size_list_post_intra)],
        dim=0,
    )
    output_tensor_inter = torch.empty(
        size=[sum(output_split_size_list_inter)] + list(output_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

    # --------------      get a2a buffer for 1st step       -------------- #

    input_tensor_pre_intra = input_tensor
    (
        a2a_input_pre_intra,
        a2a_input_split_size_pre_intra,
    ) = _calc_group_cast_a2a_input_args(
        input=input_tensor_pre_intra,
        input_split_size_list=input_split_size_list_pre_intra,
        dst_indices_list=dst_indices_list_pre_intra,
        world_size=world_size_intra_node,
        a2a_input_split_size=a2a_input_split_size_pre_intra,
        perm_before_a2a_kwargs=perm_range_gather_kwargs_pre_intra,
    )

    (
        a2a_output_pre_intra,
        a2a_output_split_size_pre_intra,
        unperm_after_a2a_kwargs_pre_intra,
    ) = _calc_group_cast_a2a_output_args(
        output=output_tensor_pre_intra,
        output_split_size_list=output_split_size_list_pre_intra,
        src_index_list=src_index_list_pre_intra,
        world_size=world_size_intra_node,
        a2a_output_split_size=a2a_output_split_size_pre_intra,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs_pre_intra,
    )

    # --------------      get a2a buffer for 2nd step       -------------- #

    input_tensor_inter = input_tensor
    (
        a2a_input_inter,
        a2a_input_split_size_inter,
    ) = _calc_group_cast_a2a_input_args(
        input=input_tensor_inter,
        input_split_size_list=input_split_size_list_inter,
        dst_indices_list=dst_indices_list_inter,
        world_size=world_size_inter_node,
        a2a_input_split_size=a2a_input_split_size_inter,
        perm_before_a2a_kwargs=perm_range_gather_kwargs_inter,
    )
    (
        a2a_output_inter,
        a2a_output_split_size_inter,
        unperm_after_a2a_kwargs_inter,
    ) = _calc_group_cast_a2a_output_args(
        output=output_tensor_inter,
        output_split_size_list=output_split_size_list_inter,
        src_index_list=src_index_list_inter,
        world_size=world_size_inter_node,
        a2a_output_split_size=a2a_output_split_size_inter,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs_inter,
    )

    # --------------------------------------------------------------------
    # hier comm operation for 1st and 2nd steps (async)
    # --------------------------------------------------------------------

    # --------------      apply a2a for 1st step       -------------- #

    with nvtx.add_nvtx_event(
        (
            f"{a2a_output_pre_intra.shape=} | "
            f"{a2a_input_pre_intra.shape=} | "
            f"{a2a_output_split_size_pre_intra=} | "
            f"{a2a_input_split_size_pre_intra=}"
        )
    ):
        work_pre_intra = dist.all_to_all_single(
            output=a2a_output_pre_intra,
            input=a2a_input_pre_intra,
            output_split_sizes=a2a_output_split_size_pre_intra,
            input_split_sizes=a2a_input_split_size_pre_intra,
            group=intra_group,
            async_op=async_op,
        )

    # --------------      apply a2a for 2nd step       -------------- #

    with nvtx.add_nvtx_event(
        (
            f"{a2a_output_inter.shape=} | "
            f"{a2a_input_inter.shape=} | "
            f"{a2a_output_split_size_inter=} | "
            f"{a2a_input_split_size_inter=}"
        )
    ):
        work_inter = dist.all_to_all_single(
            output=a2a_output_inter,
            input=a2a_input_inter,
            output_split_sizes=a2a_output_split_size_inter,
            input_split_sizes=a2a_input_split_size_inter,
            group=inter_group,
            async_op=async_op,
        )

    # --------------------------------------------------------------------
    # hier comm tensor for 3rd step
    # --------------------------------------------------------------------

    # --------------      get a2a buffer for 3rd step       -------------- #

    if work_inter is not None:
        work_inter.wait()
    input_tensor_post_intra = a2a_output_inter
    (
        a2a_input_post_intra,
        a2a_input_split_size_post_intra,
    ) = _calc_group_cast_a2a_input_args(
        input=input_tensor_post_intra,
        input_split_size_list=input_split_size_list_post_intra,
        dst_indices_list=dst_indices_list_post_intra,
        world_size=world_size_intra_node,
        a2a_input_split_size=a2a_input_split_size_post_intra,
        perm_before_a2a_kwargs=perm_range_gather_kwargs_post_intra,
    )
    (
        a2a_output_post_intra,
        a2a_output_split_size_post_intra,
        unperm_after_a2a_kwargs_post_intra,
    ) = _calc_group_cast_a2a_output_args(
        output=output_tensor_post_intra,
        output_split_size_list=output_split_size_list_post_intra,
        src_index_list=src_index_list_post_intra,
        world_size=world_size_intra_node,
        a2a_output_split_size=a2a_output_split_size_post_intra,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs_post_intra,
    )

    # --------------------------------------------------------------------
    # hier comm operation for 3rd step
    # --------------------------------------------------------------------

    # --------------      apply a2a for 3rd step       -------------- #

    with nvtx.add_nvtx_event(
        (
            f"{a2a_output_post_intra.shape=} | "
            f"{a2a_input_post_intra.shape=} | "
            f"{a2a_output_split_size_post_intra=} | "
            f"{a2a_input_split_size_post_intra=}"
        )
    ):
        work_post_intra = dist.all_to_all_single(
            output=a2a_output_post_intra,
            input=a2a_input_post_intra,
            output_split_sizes=a2a_output_split_size_post_intra,
            input_split_sizes=a2a_input_split_size_post_intra,
            group=intra_group,
            async_op=async_op,
        )

    # ---------    prepare work with post-process fn    --------- #

    work_with_post_process_fn = WorkWithPostProcessFn(
        work=[work_pre_intra, work_post_intra],
        post_process_fn=post_process_fn,
        sync=not async_op,
    )

    return work_with_post_process_fn
