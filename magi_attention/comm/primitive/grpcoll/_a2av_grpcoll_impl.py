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


import torch
import torch.distributed as dist

from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.common.enum import ReduceOp
from magi_attention.utils import nvtx

from .._all2all_v import all2all_v
from .utils import calc_group_cast_a2a_args, calc_group_reduce_a2a_args


@nvtx.instrument_nvtx
def a2av_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group-cast implementation based on all2all_v"""
    # ---------    check     --------- #

    # check functionalities
    assert output is not None, "A2A-based group-cast only supports output is given"

    # check shapes
    assert len(input_split_size_list) == len(dst_indices_list), (
        f"The length of input_split_size_list and dst_indices_list should be the same, "
        f"but got {len(input_split_size_list)=} and {len(dst_indices_list)=}"
    )
    assert len(output_split_size_list) == len(src_index_list), (
        f"The length of output_split_size_list and src_index_list should be the same, "
        f"but got {len(output_split_size_list)=} and {len(src_index_list)=}"
    )
    assert input.shape[0] == sum(input_split_size_list), (
        f"The sum of input_split_size_list should be equal to input_seqlen, "
        f"but got {sum(input_split_size_list)=} and {input.shape[0]=}"
    )
    assert output.shape[0] == sum(output_split_size_list), (
        f"The sum of output_split_size_list should be equal to output_seqlen, "
        f"but got {sum(output_split_size_list)=} and {output.shape[0]=}"
    )

    # ---------    calc group cast a2a args     --------- #

    world_size = dist.get_world_size(group)

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = calc_group_cast_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        world_size=world_size,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    work = all2all_v(
        input=a2a_input,
        output=a2a_output,
        input_split_size_list=a2a_input_split_size,
        output_split_size_list=a2a_output_split_size,
        group=group,
        async_op=async_op,
    )

    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=post_process_fn,
        async_op=async_op,
    )


@nvtx.instrument_nvtx
def a2av_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: ReduceOp = "sum",
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

    # check shapes
    assert len(input_split_size_list) == len(dst_index_list), (
        f"input_split_size_list and dst_index_list should have the same length, "
        f"but got {len(input_split_size_list)=} and {len(dst_index_list)=}"
    )
    assert len(output_split_size_list) == len(src_indices_list), (
        f"output_split_size_list and src_indices_list should have the same length, "
        f"but got {len(output_split_size_list)=} and {len(src_indices_list)=}"
    )
    assert input.shape[0] == sum(input_split_size_list), (
        f"The sum of input_split_size_list should be equal to input_seqlen, "
        f"but got {sum(input_split_size_list)=} and {input.shape[0]=}"
    )
    assert output.shape[0] == sum(output_split_size_list), (
        f"The sum of output_split_size_list should be equal to output_seqlen, "
        f"but got {sum(output_split_size_list)=} and {output.shape[0]=}"
    )

    # ---------    calc group reduce a2a args     --------- #

    world_size = dist.get_world_size(group)

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = calc_group_reduce_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_index_list=dst_index_list,
        src_indices_list=src_indices_list,
        world_size=world_size,
        reduce_op=reduce_op,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    if reduce_op == "lse":
        # FIXME: for now, we can not fuse lse comm with out comm
        # due to different shape and dtype, which should be considered and fixed in the future
        a2a_input, a2a_input_lse = a2a_input
        a2a_output, a2a_output_lse = a2a_output
        work_out = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size,
            output_split_size_list=a2a_output_split_size,
            group=group,
            async_op=async_op,
        )
        work_lse = all2all_v(
            input=a2a_input_lse,
            output=a2a_output_lse,
            input_split_size_list=a2a_input_split_size,
            output_split_size_list=a2a_output_split_size,
            group=group,
            async_op=async_op,
        )
        work = [work_out, work_lse]
    else:
        work = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size,
            output_split_size_list=a2a_output_split_size,
            group=group,
            async_op=async_op,
        )

    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=post_process_fn,
        async_op=async_op,
    )
