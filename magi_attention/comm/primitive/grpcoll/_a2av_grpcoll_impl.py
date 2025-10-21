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

from typing import overload

import torch
import torch.distributed as dist

from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import is_list_type_all, nvtx

from .._all2all_v import all2all_v
from .utils import calc_group_cast_a2a_args, calc_group_reduce_a2a_args

# ------------------        a2av group cast       ------------------ #


# host meta interface
@overload
def a2av_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_indices: list[list[int]],
    src_index: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def a2av_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_index: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def a2av_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group-cast implementation based on all2all_v"""

    # ---------    check     --------- #

    # check functionalities
    assert output is not None, "A2A-based group-cast only supports output to be given"
    assert is_list_type_all(
        [input_split_sizes, output_split_sizes, dst_indices, src_index], list
    ), (
        "A2A-based group-cast only supports host meta interface, "
        "thus the input_split_sizes, output_split_sizes, dst_indices, src_index should all be list type"
    )
    if cast_lse:
        assert input_lse is not None and output_lse is not None, (
            "A2A-based group-cast only supports input_lse and output_lse to be given"
            "if cast_lse is True"
        )

    # check shapes
    assert len(input_split_sizes) == len(dst_indices), (
        f"The length of input_split_sizes and dst_indices should be the same, "
        f"but got {len(input_split_sizes)=} and {len(dst_indices)=}"
    )
    assert len(output_split_sizes) == len(src_index), (
        f"The length of output_split_sizes and src_index should be the same, "
        f"but got {len(output_split_sizes)=} and {len(src_index)=}"
    )
    assert input.shape[0] == sum(input_split_sizes), (
        f"The sum of input_split_sizes should be equal to input_seqlen, "
        f"but got {sum(input_split_sizes)=} and {input.shape[0]=}"
    )
    assert output.shape[0] == sum(output_split_sizes), (
        f"The sum of output_split_sizes should be equal to output_seqlen, "
        f"but got {sum(output_split_sizes)=} and {output.shape[0]=}"
    )

    # ---------    calc group cast a2a args     --------- #

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size_list,
        a2a_input_split_size_list,
        post_process_fn,
    ) = calc_group_cast_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_sizes,
        output_split_size_list=output_split_sizes,
        dst_indices_list=dst_indices,
        src_index_list=src_index,
        world_size=dist.get_world_size(group),
        cast_lse=cast_lse,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    if cast_lse:
        # NOTE: we can not fuse lse comm with out comm based on nccl APIs
        # due to different shape and dtype
        a2a_input, a2a_input_lse = a2a_input
        a2a_output, a2a_output_lse = a2a_output
        work_out = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )
        work_lse = all2all_v(
            input=a2a_input_lse,
            output=a2a_output_lse,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )
        work = [work_out, work_lse]
    else:
        work = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )

    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=post_process_fn,
        async_op=async_op,
    )


# ------------------        a2av group reduce       ------------------ #


# host meta interface
@overload
def a2av_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_index: list[int],
    src_indices: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def a2av_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_index: torch.Tensor,
    src_indices: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def a2av_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_index: list[int] | torch.Tensor,
    src_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group-reduce implementation based on all2all_v"""

    # ---------    check     --------- #

    # check functionalities
    assert (
        acc_reduce and output is not None
    ), "A2A-based group-reduce only supports acc_reduce=True and output is given"
    assert is_list_type_all(
        [input_split_sizes, output_split_sizes, dst_index, src_indices], list
    ), (
        "A2A-based group-reduce only supports host meta interface, "
        "thus the input_split_sizes, output_split_sizes, dst_index, src_indice should all be list type"
    )

    # check shapes
    assert len(input_split_sizes) == len(dst_index), (
        f"input_split_sizes and dst_index should have the same length, "
        f"but got {len(input_split_sizes)=} and {len(dst_index)=}"
    )
    assert len(output_split_sizes) == len(src_indices), (
        f"output_split_sizes and src_indices should have the same length, "
        f"but got {len(output_split_sizes)=} and {len(src_indices)=}"
    )
    assert input.shape[0] == sum(input_split_sizes), (
        f"The sum of input_split_sizes should be equal to input_seqlen, "
        f"but got {sum(input_split_sizes)=} and {input.shape[0]=}"
    )
    assert output.shape[0] == sum(output_split_sizes), (
        f"The sum of output_split_sizes should be equal to output_seqlen, "
        f"but got {sum(output_split_sizes)=} and {output.shape[0]=}"
    )

    # ---------    calc group reduce a2a args     --------- #

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size_list,
        a2a_input_split_size_list,
        post_process_fn,
    ) = calc_group_reduce_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_sizes,
        output_split_size_list=output_split_sizes,
        dst_index_list=dst_index,
        src_indices_list=src_indices,
        world_size=dist.get_world_size(group),
        reduce_op=reduce_op,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    if reduce_op == "lse":
        # NOTE: we can not fuse lse comm with out comm based on nccl APIs
        # due to different shape and dtype
        a2a_input, a2a_input_lse = a2a_input
        a2a_output, a2a_output_lse = a2a_output
        work_out = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )
        work_lse = all2all_v(
            input=a2a_input_lse,
            output=a2a_output_lse,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )
        work = [work_out, work_lse]
    else:
        work = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )

    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=post_process_fn,
        async_op=async_op,
    )
