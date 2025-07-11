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

from typing import Optional

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.utils import nvtx

from ._all2all_v import all2all_v
from ._group_collective_hier import (
    hier_group_cast_impl_with_a2av,
    hier_group_reduce_impl_with_a2av,
)
from .utils import _calc_group_cast_a2a_args, _calc_group_reduce_a2a_args

__all__ = [
    "group_cast_collective",
    "group_reduce_collective",
]


@torch.no_grad()
@nvtx.instrument_nvtx
def group_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """
    Args:
        input(torch.Tensor): input tensor with shape [input_seqlen,...]
        output(torch.Tensor): output tensor with shape [output_seqlen,...]
        input_split_size_list(list[int]): the size list to split the input tensor,
            where sum(input_split_size_list) == input_seqlen
        output_split_size_list(list[int]): the size list to split the output tensor,
            where sum(output_split_size_list) == output_seqlen
        dst_indices_list(list[list[int]]): the destination indices list for each input split to broadcast to,
            where len(dst_indices_list) == len(input_split_size_list)
        src_index_list(list[int]): the source index list for each output split to receive from,
            where len(src_index_list) == len(output_split_size_list)
            NOTE: the order of the output splits are "stable", which means the ones from the same source
            will be in the same order as the input splits

        HACK:
        **kwargs: additional keyword arguments,
        this kernel is for now based on all2all-v,
        thus introducing pre-/post-processing overhead
        on both tensor and meta info to be compatible with all2all-v input/output.
        Therefore, we add `kwargs` since the processing of meta info
        can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work_with_post_process_fn(WorkWithPostProcessFn): async work with the post-process function
        to transfer from a2a-v output tensor to group-cast output tensor

    TODO: add examples

    NOTE(xiaowu):
        * The input can be split into a list of [splited_input] using input_split_size_list,
            where each splited_input can be sent to 0 or multiple ranks,
            and the destination ranks are determined by dst_indices_list.
        * The output can be split into a list of [splited_output] using output_split_size_list,
            where each splited_output must be received from exactly 1 src_rank,
            and the source ranks are determined by src_index_list.

    REVIEW(xiaowu):
        * Must each splited_output be received from exactly 1 src_rank? Could it be 0?
    """

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

    if magi_attention.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group-cast
        return hier_group_cast_impl_with_a2av(
            input_tensor=input,
            output_tensor=output,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=group,
            async_op=async_op,
            **kwargs,
        )

    # ---------    calc group cast a2a args     --------- #

    world_size = dist.get_world_size(group)

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = _calc_group_cast_a2a_args(
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
        work=work,
        post_process_fn=post_process_fn,
        sync=not async_op,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """

    Args:
        input(torch.Tensor): input tensor with shape [input_seqlen,...]
        output(torch.Tensor): output tensor with shape [output_seqlen,...]
        input_split_size_list(list[int]): the size list to split the input tensor,
            where sum(input_split_size_list) == input_seqlen
        output_split_size_list(list[int]): the size list to split the output tensor,
            where sum(output_split_size_list) == output_seqlen
        dst_index_list(list[int]): the destination index list for each input split to return to,
            where len(dst_index_list) == len(input_split_size_list)
        src_indices_list(list[list[int]]): the source indices list for each output split to reduce from,
            where len(src_indices_list) == len(output_split_size_list)
            NOTE: since any reduce operation satisfies the commutative property, the order of the input splits to reduce
            to the same output split does not matter

        HACK:
        **kwargs: additional keyword arguments,
        this kernel is for now based on all2all-v,
        thus introducing pre-/post-processing overhead
        on both tensor and meta info to be compatible with all2all-v input/output.
        Therefore, we add `kwargs` since the processing of meta info
        can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work_with_post_process_fn(WorkWithPostProcessFn): async work with the post-process function
        to transfer from a2a-v output tensor to group-reduce output tensor

    NOTE(xiaowu):
        * The input can be split into a list of [splited_input] using input_split_size_list,
            where each splited_input must be sent to one rank,
            and the destination rank is determined by dst_index_list.
        * The output can be split into a list of [splited_output] using output_split_size_list,
            where each splited_output can be reduced from 0 or multiple src_ranks,
            and the source ranks for reduction are determined by src_indices_list.
    """
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

    if magi_attention.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group-reduce
        return hier_group_reduce_impl_with_a2av(
            input_tensor=input,
            output_tensor=output,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_index_list=dst_index_list,
            src_indices_list=src_indices_list,
            group=group,
            async_op=async_op,
            **kwargs,
        )

    # ---------    calc group reduce a2a args     --------- #

    world_size = dist.get_world_size(group)

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size,
        a2a_input_split_size,
        post_process_fn,
    ) = _calc_group_reduce_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_index_list=dst_index_list,
        src_indices_list=src_indices_list,
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
        work=work,
        post_process_fn=post_process_fn,
        sync=not async_op,
    )
