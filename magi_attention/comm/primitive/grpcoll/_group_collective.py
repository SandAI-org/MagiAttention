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

from typing import Literal

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.utils import nvtx

from .._all2all_v import all2all_v
from ._group_collective_hier import (
    hier_group_cast_impl_with_a2av,
    hier_group_reduce_impl_with_a2av,
)
from .utils import calc_group_cast_a2a_args, calc_group_reduce_a2a_args

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
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """
    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...]
        output (torch.Tensor): output tensor with shape [output_seqlen, ...]
        input_split_size_list (list[int]): the size list to split the input tensor,
            where sum(input_split_size_list) == input_seqlen
        output_split_size_list (list[int]): the size list to split the output tensor,
            where sum(output_split_size_list) == output_seqlen
        dst_indices_list (list[list[int]]): the destination indices list for each input split to broadcast to,
            where len(dst_indices_list) == len(input_split_size_list)
        src_index_list (list[int]): the source index list for each output split to receive from,
            where len(src_index_list) == len(output_split_size_list)

            NOTE: the order of the output splits are "stable", which means the ones from the same source
            will be in the same order as the input splits
        group (dist.ProcessGroup | None): the process group to use, if None, use the default group
        async_op (bool): whether to use async op. Defaults to False
        kwargs: additional keyword arguments,
            this kernel is for now based on all2all-v,
            thus introducing pre-/post-processing overhead
            on both tensor and meta info to be compatible with all2all-v input/output.
            Therefore, we add `kwargs` since the processing of meta info
            can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work_with_post_process_fn (WorkWithPostProcessFn): async work with the post-process function
        to transfer from a2a-v output tensor to group-cast output tensor
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
        assert (
            not magi_attention.comm.is_native_grpcoll_enable()
        ), "Hierarchical group-cast is not compatible with native grpcoll implementation"
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

    if magi_attention.comm.is_native_grpcoll_enable():
        # XXX: a workaround to implement native group-cast
        # by original deep-ep dispatch kernels
        # with pre-meta-args processing and post-output processing
        pass

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


@torch.no_grad()
@nvtx.instrument_nvtx
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group reduce collective

    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...] to reduce from
        output (torch.Tensor): output tensor with shape [output_seqlen, ...] to reduce to
        input_split_size_list (list[int]): the size list to split the input tensor,
            where sum(input_split_size_list) == input_seqlen
        output_split_size_list (list[int]): the size list to split the output tensor,
            where sum(output_split_size_list) == output_seqlen
        dst_index_list (list[int]): the destination index list for each input split to return to,
            where len(dst_index_list) == len(input_split_size_list)
        src_indices_list (list[list[int]]): the source indices list for each output split to reduce from,
            where len(src_indices_list) == len(output_split_size_list)

            NOTE: since any reduce operation satisfies the commutative property, the order of the input splits to reduce
            to the same output split does not matter
        group (dist.ProcessGroup | None): the process group to use, if None, use the default group
        async_op (bool): whether to use async op. Defaults to False
        reduce_op (Literal["sum", "avg", "weight", "lse"]): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction

            NOTE:
                1. if reduce_op is "avg", we will sum-reduce to the output tensor and apply average division afterwards,
                    so the user should guarantee that the output tensor is initialized to zero
                    otherwise the semantics will be incorrect unless the user intentionally does this
                2. if reduce_op is "lse", the user is required to pass "input_lse" and "output_lse",
                    and we only support input/output has shape [seqlen, num_heads, head_dim]
                    while input_lse/output_lse has shape [seqlen, num_heads] for now
        input_lse (torch.Tensor | None): the log-sum-exp tensor for the input tensor,
            only required and used if reduce_op is "lse"
        output_lse (torch.Tensor | None): the log-sum-exp tensor for the output tensor,
            only required and used if reduce_op is "lse"
        kwargs: additional keyword arguments,
            this kernel is for now based on all2all-v,
            thus introducing pre-/post-processing overhead
            on both tensor and meta info to be compatible with all2all-v input/output.
            Therefore, we add `kwargs` since the processing of meta info
            can be processed in advance, and just passed in through `kwargs` to reduce runtime overhead

    Returns:
        work_with_post_process_fn (WorkWithPostProcessFn): async work with the post-process function
        to transfer from a2a-v output tensor to group-reduce output tensor
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
        assert (
            not magi_attention.comm.is_native_grpcoll_enable()
        ), "Hierarchical group-reduce is not compatible with native grpcoll implementation"

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
            reduce_op=reduce_op,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    if magi_attention.comm.is_native_grpcoll_enable():
        # XXX: a workaround to implement native group-reduce
        # by original deep-ep combine kernels
        # with pre-meta-args processing and pre-input processing
        pass

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
