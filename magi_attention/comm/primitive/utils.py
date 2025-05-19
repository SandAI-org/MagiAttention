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
from typing import Callable, TypeAlias

import torch
import torch.distributed as dist

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
) -> tuple[list[int], dict]:
    a2a_output_split_size_per_rank: list[list[int]] = [[] for _ in range(world_size)]
    a2a_output_permute_index_list_per_rank: list[list[int]] = [
        [] for _ in range(world_size)
    ]
    for i, src_index in enumerate(src_index_list):
        a2a_output_split_size_per_rank[src_index].append(output_split_size_list[i])
        a2a_output_permute_index_list_per_rank[src_index].append(i)

    a2a_output_split_size = [sum(x) for x in a2a_output_split_size_per_rank]
    a2a_output_tensor_size_list = list(chain(*a2a_output_split_size_per_rank))
    a2a_output_permute_index_list = list(chain(*a2a_output_permute_index_list_per_rank))
    a2a_output_unpermute_index_list = sorted(
        range(len(a2a_output_permute_index_list)),
        key=lambda x: a2a_output_permute_index_list[x],
    )

    # ---------    calc unperm before a2a kwargs     --------- #
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
