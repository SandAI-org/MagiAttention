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

import os
import socket
import warnings
from collections import defaultdict
from functools import partial
from itertools import accumulate, chain, pairwise
from typing import Any, Callable, TypeAlias, overload

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import magi_attention
from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.common.enum import GroupReduceOp, OutMaybeWithLSE
from magi_attention.common.range import NaiveRange
from magi_attention.common.range_op import range_gather, range_reduce
from magi_attention.common.range_op.utils import (
    _calc_cu_range_sizes,
    _calc_out2inp_range_map,
    _calc_ranges_row_map,
)
from magi_attention.common.ranges import NaiveRanges
from magi_attention.utils import is_list_type_all, nvtx, perm_idxs2unperm_idxs

# ------------------        common helpers       ------------------ #

RangesWithRank: TypeAlias = list[tuple[NaiveRange, int]]


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


def seqlens2curanges(
    seqlens: list[int],
) -> NaiveRanges:
    """Make seqlens to cumulative ranges

    Args:
        seqlens (list[int]): the seqlen list, e.g. [4, 2, 7]

    Returns:
        NaiveRanges: the cumulative ranges, e.g. [(0, 4), (4, 6), (6, 13)]
    """
    return list(pairwise(accumulate([0] + seqlens)))


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
    ranges = seqlens2curanges(split_size_list)

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
    deterministic: bool = False,
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

    range_reduce_kwargs: dict[str, Any] = {"deterministic": deterministic}
    input_ranges = torch.tensor(input_ranges, dtype=torch.int32, device=device)
    range_reduce_kwargs["input_ranges"] = input_ranges

    if deterministic:
        (out2inp_range_map, unique_ordered_out_ranges, _) = _calc_out2inp_range_map(
            output_ranges,
            # first put to cpu to avoid d2h in `_calc_cu_range_sizes` below
            device=torch.device("cpu"),
        )

        cu_range_sizes, total_size = _calc_cu_range_sizes(
            unique_ordered_out_ranges,
            device=device,
        )

        # put back to device before `_calc_ranges_row_map`
        # to make row_map a device tensor
        out2inp_range_map = out2inp_range_map.to(device)
        unique_ordered_out_ranges = unique_ordered_out_ranges.to(device)

        row_map = _calc_ranges_row_map(
            unique_ordered_out_ranges,
            total_size,
        )

        range_reduce_kwargs["out2inp_range_map"] = out2inp_range_map
        range_reduce_kwargs["unique_ordered_out_ranges"] = unique_ordered_out_ranges
    else:
        range_sizes = torch.tensor([0] + range_sizes, dtype=torch.int32, device=device)
        cu_range_sizes = torch.cumsum(range_sizes, dim=0)
        row_map = torch.repeat_interleave(
            torch.arange(0, input_ranges.shape[0], device=device),
            range_sizes[1:],
            dim=0,
            output_size=total_size,
        )

    range_reduce_kwargs["cu_range_sizes"] = cu_range_sizes
    range_reduce_kwargs["total_size"] = total_size
    range_reduce_kwargs["row_map"] = row_map

    output_ranges = torch.tensor(output_ranges, dtype=torch.int32, device=device)
    range_reduce_kwargs["output_ranges"] = output_ranges

    return range_reduce_kwargs


# ------------------        utils for group cast       ------------------ #


def group_cast_impl_with_batch_p2p(
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
        work=GeneralWork(work=work_list),
        post_process_fn=lambda x: x,
        async_op=async_op,
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
def unpermute_tensor(
    tensor: torch.Tensor,
    unperm_after_a2a_kwargs: dict,
) -> torch.Tensor:
    """unpermute a2a output to output
    as a post-processing func for group_cast
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
    input_size_ranges = seqlens2curanges(input_split_size_list)

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
    return_verbose: bool = False,
):
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

    if return_verbose:
        return (
            a2a_output_split_size,
            unperm_range_gather_kwargs,
            # verbose
            a2a_output_tensor_size_list,
            a2a_output_permute_index_list,
            a2a_output_unpermute_index_list,
        )

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
def calc_group_cast_a2a_args(
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
        unpermute_tensor,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs,
    )

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )


# ------------------        utils for group reduce       ------------------ #


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
def sum_reduce_to_tensor(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    range_reduce_kwargs: dict,
) -> torch.Tensor:
    """sum-reduce a2a output to output
    as a post-processing func for group_reduce
    """

    output = range_reduce(
        input=a2a_output,
        output=output,
        reduce_op="sum",
        **range_reduce_kwargs,
    )

    return output


@nvtx.instrument_nvtx
def avg_reduce_to_tensor(
    output: torch.Tensor,
    a2a_output: torch.Tensor,
    range_reduce_kwargs: dict,
) -> torch.Tensor:
    """avg-reduce a2a output to output
    as a post-processing func for group_reduce
    """

    output = range_reduce(
        input=a2a_output,
        output=output,
        reduce_op="avg",
        **range_reduce_kwargs,
    )

    return output


@nvtx.instrument_nvtx
def lse_reduce_to_tensor(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    a2a_output: torch.Tensor,
    a2a_output_lse: torch.Tensor,
    range_reduce_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """lse-reduce a2a output and a2a_output_lse to output and output_lse
    as a post-processing func for group_reduce with reduce_op="lse"
    """

    output = range_reduce(
        input=a2a_output,
        output=output,
        reduce_op="lse",
        input_lse=a2a_output_lse,
        output_lse=output_lse,
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
    input_size_ranges = seqlens2curanges(input_split_size_list)
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
    reduce_op: GroupReduceOp = "sum",
    input_lse: torch.Tensor | None = None,
    **kwargs,
) -> tuple[OutMaybeWithLSE, list[int]]:
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
    if reduce_op == "lse":
        a2a_input_lse = range_gather(
            input=input_lse,
            **perm_before_a2a_kwargs,
        )
        a2a_input = (a2a_input, a2a_input_lse)

    return a2a_input, a2a_input_split_size


@nvtx.instrument_nvtx
def _calc_group_reduce_a2a_output_meta_args(
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    device: torch.device,
    deterministic: bool = False,
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
    a2a_output_size_ranges = seqlens2curanges(a2a_output_tensor_size_list)
    output_size_ranges = seqlens2curanges(output_split_size_list)
    cum_src_ranges = seqlens2curanges(num_src_list)
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
        deterministic=deterministic,
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
    reduce_op: GroupReduceOp = "sum",
    output_lse: torch.Tensor | None = None,
    deterministic: bool = False,
    **kwargs,
) -> tuple[OutMaybeWithLSE, list[int], dict]:
    # only sum-reduce has non-deterministic kernel by now
    deterministic |= reduce_op != "sum"

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
            deterministic=deterministic,
        )

    # -----     group_reduce_a2a_output tensor args     ----- #

    a2a_output = torch.empty(
        [sum(a2a_output_split_size), *output.shape[1:]],
        device=output.device,
        dtype=output.dtype,
    )
    if reduce_op == "lse":
        a2a_output_lse = torch.empty(
            [sum(a2a_output_split_size), *output_lse.shape[1:]],  # type: ignore[union-attr]
            device=output.device,
            dtype=output_lse.dtype,  # type: ignore[union-attr]
        )
        a2a_output = (a2a_output, a2a_output_lse)

    return (
        a2a_output,
        a2a_output_split_size,
        range_reduce_kwargs,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def calc_group_reduce_a2a_args(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    world_size: int,
    reduce_op: GroupReduceOp = "sum",
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> tuple[
    OutMaybeWithLSE,
    OutMaybeWithLSE,
    list[int],
    list[int],
    Callable[[OutMaybeWithLSE], OutMaybeWithLSE],
]:
    # ---------    calc a2a_input_split_size and a2a input     --------- #

    (
        a2a_input,
        a2a_input_split_size,
    ) = _calc_group_reduce_a2a_input_args(
        input=input,
        input_split_size_list=input_split_size_list,
        dst_index_list=dst_index_list,
        world_size=world_size,
        reduce_op=reduce_op,
        input_lse=input_lse,
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
        reduce_op=reduce_op,
        output_lse=output_lse,
        **kwargs,
    )

    # ---------    prepare post process fn     --------- #

    match reduce_op:
        case "lse":
            post_process_fn = partial(
                lse_reduce_to_tensor,
                a2a_output=a2a_output[0],  # a2a_output
                a2a_output_lse=a2a_output[1],  # a2a_output_lse
                range_reduce_kwargs=range_reduce_kwargs,
            )
        case "sum":
            post_process_fn = partial(
                sum_reduce_to_tensor,
                a2a_output=a2a_output,
                range_reduce_kwargs=range_reduce_kwargs,
            )
        case "avg":
            post_process_fn = partial(
                avg_reduce_to_tensor,
                a2a_output=a2a_output,
                range_reduce_kwargs=range_reduce_kwargs,
            )
        case _:
            raise RuntimeError(f"reduce_op={reduce_op} not supported")

    return (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    )


# ------------------        utils for native grpcoll       ------------------ #


def check_nvlink_connections(group: dist.ProcessGroup):
    """
    Check NVLink connection between every pair of GPUs.

    Arguments:
        group: the communication group.
    """
    # Check NVLink connection
    # NOTES: some A100 PCIE GPUs only have pairwise NVLink connection, so that we can only use EP2
    # TODO: check all cases, all local-node GPUs in the group should be connected via NVLink
    if "PCIE" in torch.cuda.get_device_name():
        assert group.size() <= 2, "PCIe GPUs only have pairwise NVLink connections"

        import pynvml

        pynvml.nvmlInit()

        devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            .strip(",")
            .split(",")
        )
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [
            0,
        ] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        # Check whether they are all connected via NVLink, Reference:
        # https://github.com/vllm-project/vllm/blob/b8e809a057765c574726a6077fd124db5077ce1f/vllm/platforms/cuda.py#L438
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                assert (
                    status == pynvml.NVML_P2P_STATUS_OK
                ), f"GPU {physical_device_indices[i]} and GPU {physical_device_indices[j]} are not connected via NVLink"

        # Close NVML
        pynvml.nvmlShutdown()


def are_all_ranks_on_same_host(group: dist.ProcessGroup = None) -> bool:
    """
    Checks if all ranks within a given ProcessGroup are running on the same host node.

    This function works by having each rank gather its hostname and then comparing
    these hostnames across the entire group.

    Args:
        group (dist.ProcessGroup, optional): The process group to check.
                                             If None, the default process group is used.

    Returns:
        bool: True if all ranks are on the same host; False otherwise.

    Raises:
        RuntimeError: If torch.distributed has not been initialized.
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed has not been initialized. Call dist.init_process_group first."
        )

    # Get the hostname of the current process
    current_hostname = socket.gethostname()

    # Create a list to store hostnames from all ranks in the group.
    # all_gather_object requires a list as an output buffer,
    # and its size must match the world size of the group.
    world_size = dist.get_world_size(group)
    gathered_hostnames = [None] * world_size

    # Use all_gather_object to collect the current hostname from all ranks in the group.
    # Note: all_gather_object is a blocking operation; all ranks must call it.
    dist.all_gather_object(gathered_hostnames, current_hostname, group)

    # Check if all gathered hostnames are identical.
    # If the set of hostnames has only one unique element, it means all ranks
    # are on the same host.
    return len(set(gathered_hostnames)) == 1


def maybe_lazy_init_buffer(group: dist.ProcessGroup) -> None:
    from ._mgr import grpcoll_mgr

    if not grpcoll_mgr.is_registered(group):
        grpcoll_mgr.register_buffer(group=group)
        warnings.warn(
            f"Since the GrpCollBuffer is not registered for {group.group_name}, "
            "we lazily register it here with the default config, "
            "which might not be the best choice and cause performance/memory issue."
        )


def get_group_reduce_handle_from_sym_group_cast(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_index: list[int] | torch.Tensor,
    src_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    output_seqlen: int | None = None,
) -> tuple:
    from ._buffer import GrpCollBuffer
    from ._native_grpcoll_impl import native_group_cast_impl

    # prepare dummpy input/output for symmetric group-cast
    if output is not None:
        sym_gc_input_seqlen = output.size(0)
    elif output_seqlen is not None:
        sym_gc_input_seqlen = output_seqlen
    elif isinstance(output_split_sizes, list):
        sym_gc_input_seqlen = sum(output_split_sizes)
    else:
        # NOTE: had better pass output_seqlen to avoid gpu-cpu sync
        sym_gc_input_seqlen = output_split_sizes.sum().item()
    sym_gc_output_seqlen = input.size(0)
    sym_gc_dummy_input = torch.empty(
        (sym_gc_input_seqlen, GrpCollBuffer.hidden_size_alignment),
        dtype=torch.bfloat16,
        device="cuda",
    )
    sym_gc_dummy_output = (
        torch.empty(
            (sym_gc_output_seqlen, GrpCollBuffer.hidden_size_alignment),
            dtype=torch.bfloat16,
            device="cuda",
        )
        if output is not None
        else None
    )

    # prepare native group-cast handle dict
    # to receive the handle from symmetric group-cast
    native_grpcoll_handle_dict: dict[str, Any] = {"group_cast": None}

    sym_work_gc = native_group_cast_impl(
        input=sym_gc_dummy_input,
        output=sym_gc_dummy_output,
        input_split_sizes=output_split_sizes,
        output_split_sizes=input_split_sizes,
        dst_indices=src_indices,
        src_index=dst_index,
        group=group,
        async_op=async_op,
        native_grpcoll_handle_dict=native_grpcoll_handle_dict,
    )
    sym_work_gc.wait_post_process()

    handle: tuple = native_grpcoll_handle_dict["group_reduce"]

    return handle


@triton.jit
def _varlen_for_each_copy_kernel(
    output_ptr,
    input_flat_ptr,
    num_dst_ptr,
    offsets_ptr,
    stride_output_row,
    num_ranks,
    BLOCK_SIZE_N: tl.constexpr,
):
    split_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    col_start = col_block_idx * BLOCK_SIZE_N

    num_dst = tl.load(num_dst_ptr + split_idx)

    offset_in_flat_input = tl.load(offsets_ptr + split_idx)

    output_row_base_ptr = output_ptr + split_idx * stride_output_row
    input_flat_base_ptr = input_flat_ptr + offset_in_flat_input

    cols_in_block = col_start + tl.arange(0, BLOCK_SIZE_N)

    write_mask = (cols_in_block < num_ranks) & (cols_in_block < num_dst)

    values_to_write = tl.load(
        input_flat_base_ptr + cols_in_block, mask=write_mask, other=-1
    )

    tl.store(output_row_base_ptr + cols_in_block, values_to_write, mask=write_mask)


def _varlen_for_each_copy_triton(
    dst: torch.Tensor,
    src: torch.Tensor,
    num_dst: torch.Tensor,
    cu_num_dst: torch.Tensor,
) -> torch.Tensor:
    num_splits, num_ranks = dst.shape

    if num_splits == 0:
        return dst

    BLOCK_SIZE_N = triton.next_power_of_2(num_ranks)
    grid = (num_splits, triton.cdiv(num_ranks, BLOCK_SIZE_N))

    _varlen_for_each_copy_kernel[grid](
        dst,
        src,
        num_dst,
        cu_num_dst,
        dst.stride(0),
        num_ranks,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return dst


@overload
def transfer_splits_and_dst_idxs_to_topk_idx(
    input_split_sizes: list[int],
    dst_indices: list[list[int]],
    num_ranks: int,
    input_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    ...


@overload
def transfer_splits_and_dst_idxs_to_topk_idx(
    input_split_sizes: torch.Tensor,
    dst_indices: torch.Tensor,
    num_ranks: int,
    input_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    ...


@nvtx.instrument_nvtx
def transfer_splits_and_dst_idxs_to_topk_idx(
    input_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    num_ranks: int,
    input_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    if is_list_type_all([input_split_sizes, dst_indices], list):
        if input_seqlen is None:
            input_seqlen = sum(input_split_sizes)
        num_splits = len(input_split_sizes)

        num_dst_list: list[int] = [len(dst_indices) for dst_indices in dst_indices]
        cu_num_dst_list: list[int] = list(accumulate([0] + num_dst_list))
        flatten_dst_indices_list: list[int] = list(chain(*dst_indices))
        all_meta_list: list[list[int]] = [
            num_dst_list,
            cu_num_dst_list,
            input_split_sizes,
            flatten_dst_indices_list,
        ]
        all_meta_size_list: list[int] = [len(meta) for meta in all_meta_list]
        flatten_all_meta_list: list[int] = list(chain(*all_meta_list))

        (
            num_dst_tensor,
            cu_num_dst_tensor,
            split_size_tensor,
            dst_indices_tensor,
        ) = torch.tensor(
            flatten_all_meta_list,
            dtype=dtype,
            device=device,
        ).split(
            all_meta_size_list
        )

        topk_idx = torch.full(
            (num_splits, num_ranks),  # assuming num_local_experts == 1
            fill_value=-1,
            dtype=dtype,
            device=device,
        )

        _varlen_for_each_copy_triton(
            dst=topk_idx,
            src=dst_indices_tensor,
            num_dst=num_dst_tensor,
            cu_num_dst=cu_num_dst_tensor,
        )
    elif is_list_type_all([input_split_sizes, dst_indices], torch.Tensor):
        assert dst_indices.size(1) == num_ranks  # type: ignore[union-attr]
        # NOTE: had better pass input_seqlen to avoid gpu-cpu sync
        if input_seqlen is None:
            input_seqlen = input_split_sizes.sum().item()  # type: ignore[union-attr]
        topk_idx = dst_indices.to(dtype=dtype, device=device)  # type: ignore[union-attr]
        split_size_tensor = input_split_sizes.to(device=device)  # type: ignore[union-attr]
    else:
        raise ValueError(
            "input_split_sizes and dst_indices must be "
            "either all list or all torch.Tensor"
        )

    # shape: (num_splits, num_ranks) -> (input_seqlen, num_ranks)
    topk_idx = topk_idx.repeat_interleave(
        split_size_tensor,
        dim=0,
        output_size=input_seqlen,
    )

    return topk_idx


@overload
def get_dispatch_layout_from_group_cast_meta(
    input_split_sizes: list[int],
    dst_indices: list[list[int]],
    group: dist.ProcessGroup,
    topk_idx: torch.Tensor | None = None,
    input_seqlen: int | None = None,
    num_nodes: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    ...


@overload
def get_dispatch_layout_from_group_cast_meta(
    input_split_sizes: torch.Tensor,
    dst_indices: torch.Tensor,
    group: dist.ProcessGroup,
    topk_idx: torch.Tensor | None = None,
    input_seqlen: int | None = None,
    num_nodes: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    ...


@nvtx.instrument_nvtx
def get_dispatch_layout_from_group_cast_meta(
    input_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    topk_idx: torch.Tensor | None = None,
    input_seqlen: int | None = None,
    num_nodes: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    from ._buffer import GrpCollBuffer
    from ._mgr import grpcoll_mgr

    num_ranks = group.size()
    num_experts = num_ranks

    # TODO: support directly pass input_split_sizes and dst_indices
    # and no need to transfer to topk_idx first
    if topk_idx is None:
        topk_idx = transfer_splits_and_dst_idxs_to_topk_idx(
            input_split_sizes=input_split_sizes,
            dst_indices=dst_indices,
            num_ranks=num_ranks,
            input_seqlen=input_seqlen,
            device=device,
            dtype=dtype,
        )

    buffer_registered = grpcoll_mgr.is_registered(group)
    if buffer_registered:
        buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,  # event_overlap
        ) = buffer.get_dispatch_layout(
            topk_idx=topk_idx,
            num_experts=num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
    else:
        assert num_nodes is not None, (
            "``num_nodes`` must be provided "
            "since the grpcoll buffer is not registered for the given group"
        )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,  # event_overlap,
        ) = GrpCollBuffer.get_dispatch_meta_from_topk_idx(
            topk_idx=topk_idx,
            num_ranks=num_ranks,
            num_nodes=num_nodes,
            num_experts=num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_meta_stream=False,
        )

    return (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
    )


@overload
def get_a2av_perm_idxs_from_group_cast_meta(
    output_split_sizes: list[int],
    src_index: list[int],
    num_ranks: int,
    output_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    ...


@overload
def get_a2av_perm_idxs_from_group_cast_meta(
    output_split_sizes: torch.Tensor,
    src_index: torch.Tensor,
    num_ranks: int,
    output_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    ...


@nvtx.instrument_nvtx
def get_a2av_perm_idxs_from_group_cast_meta(
    output_split_sizes: list[int] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    num_ranks: int,
    output_seqlen: int | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    from ._buffer import GrpCollBuffer

    if is_list_type_all([output_split_sizes, src_index], list):
        if output_seqlen is None:
            output_seqlen = sum(output_split_sizes)
        output_split_sizes = torch.tensor(
            output_split_sizes,
            dtype=dtype,
            device=device,
        )
        src_index = torch.tensor(
            src_index,
            dtype=dtype,
            device=device,
        )
    elif is_list_type_all([output_split_sizes, src_index], torch.Tensor):
        # NOTE: had better pass output_seqlen to avoid gpu-cpu sync
        if output_seqlen is None:
            output_seqlen = output_split_sizes.sum().item()  # type: ignore[union-attr]
        output_split_sizes = output_split_sizes.to(  # type: ignore[union-attr]
            dtype=dtype, device=device
        )
        src_index = src_index.to(dtype=dtype, device=device)  # type: ignore[union-attr]
    else:
        raise ValueError(
            "output_split_sizes and src_index must be "
            "either all list or all torch.Tensor"
        )

    (
        perm_to_a2av_idx,
        _,  # event_overlap
    ) = GrpCollBuffer.get_a2av_perm_idx_from_src_idx(
        output_split_sizes=output_split_sizes,
        src_idx=src_index,
        num_ranks=num_ranks,
        output_seqlen=output_seqlen,
        previous_event=None,
        async_finish=False,
        allocate_on_meta_stream=False,
    )

    return perm_to_a2av_idx


def _get_a2av_perm_idxs_from_group_cast_meta_ref(
    output_split_size_list: list[int],
    src_index_list: list[int],
    num_ranks: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    # count the total split size of each rank
    rank_split_sizes = [0] * num_ranks
    for i in range(len(output_split_size_list)):
        rank_split_sizes[src_index_list[i]] += output_split_size_list[i]

    # count the global rank offset
    a2av_rank_offsets = list(accumulate([0] + rank_split_sizes))[:-1]

    # a2a_output[unperm_from_a2av_idx] => output
    unperm_from_a2av_idx: list[int] = []
    current_offset_within_rank = [0] * num_ranks
    for i in range(len(output_split_size_list)):
        target_size = output_split_size_list[i]
        target_rank = src_index_list[i]

        # get the start offset of the target buffer sent from the target rank
        global_start_offset_in_output = (
            a2av_rank_offsets[target_rank] + current_offset_within_rank[target_rank]
        )
        unperm_from_a2av_idx.extend(
            range(
                global_start_offset_in_output,
                global_start_offset_in_output + target_size,
            )
        )
        current_offset_within_rank[target_rank] += target_size

    # output[perm_to_a2av_idx] => a2a_output
    perm_to_a2av_idx = perm_idxs2unperm_idxs(unperm_from_a2av_idx)

    # convert to tensor
    (
        unperm_from_a2av_idx,
        perm_to_a2av_idx,
    ) = torch.tensor(
        unperm_from_a2av_idx + perm_to_a2av_idx,
        dtype=dtype,
        device=device,
    ).chunk(2)

    return unperm_from_a2av_idx, perm_to_a2av_idx
