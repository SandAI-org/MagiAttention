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
from typing import Any, Callable, Literal, TypeAlias

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.common.range import NaiveRange
from magi_attention.common.range_op import range_gather, range_reduce
from magi_attention.common.range_op.utils import (
    _calc_cu_range_sizes,
    _calc_out2inp_range_map,
    _calc_ranges_row_map,
)
from magi_attention.common.ranges import NaiveRanges
from magi_attention.utils import nvtx


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

OutMaybeWithLSE: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


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
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
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
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
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
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
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
            a2a_output, a2a_output_lse = a2a_output
            post_process_fn = partial(
                lse_reduce_to_tensor,
                a2a_output=a2a_output,
                a2a_output_lse=a2a_output_lse,
                range_reduce_kwargs=range_reduce_kwargs,
            )
        case "sum":
            post_process_fn = partial(
                sum_reduce_to_tensor,
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
