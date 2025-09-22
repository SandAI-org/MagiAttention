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
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.utils import nvtx

from ._a2av_grpcoll_impl import a2av_group_cast_impl, a2av_group_reduce_impl
from ._group_collective_hier import (
    hier_group_cast_impl_with_a2av,
    hier_group_reduce_impl_with_a2av,
)
from ._native_grpcoll_impl import native_group_cast_impl, native_group_reduce_impl

__all__ = [
    "group_cast_collective",
    "group_reduce_collective",
]


@torch.no_grad()
@nvtx.instrument_nvtx
def group_cast_collective(
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
        group (dist.ProcessGroup): the process group to comm
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

    if magi_attention.comm.is_native_grpcoll_enable():
        return native_group_cast_impl(
            input=input,
            output=output,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=group,
            async_op=async_op,
            **kwargs,
        )

    return a2av_group_cast_impl(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_indices_list=dst_indices_list,
        src_index_list=src_index_list,
        group=group,
        async_op=async_op,
        **kwargs,
    )


@torch.no_grad()
@nvtx.instrument_nvtx
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group reduce collective

    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...] to reduce from
        output (torch.Tensor | None): output tensor with shape [output_seqlen, ...] to reduce to

            NOTE: if output is None, the output tensor will be allocated in this function
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
        group (dist.ProcessGroup): the process group to comm
        async_op (bool): whether to use async op. Defaults to False
        reduce_op (Literal["sum", "avg", "weight", "lse"]): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction

            NOTE:
                If reduce_op is "lse", the user is required to pass "input_lse" and "output_lse",
                and we only support input/output has shape [seqlen, num_heads, head_dim]
                while input_lse/output_lse has shape [seqlen, num_heads] for now
        acc_reduce (bool): whether to accumulate the reduction to the given output buffer. Defaults to True.

            NOTE:
                If False, the output will be overwritten and the initial value will be ignored.
                Otherwise, the output buffer must be given and the initial value will be accumulated
                w.r.t. the reduction operation according to the ``reduce_op``.
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
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    if magi_attention.comm.is_native_grpcoll_enable():
        return native_group_reduce_impl(
            input=input,
            output=output,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_index_list=dst_index_list,
            src_indices_list=src_indices_list,
            group=group,
            async_op=async_op,
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    return a2av_group_reduce_impl(
        input=input,
        output=output,
        input_split_size_list=input_split_size_list,
        output_split_size_list=output_split_size_list,
        dst_index_list=dst_index_list,
        src_indices_list=src_indices_list,
        group=group,
        async_op=async_op,
        reduce_op=reduce_op,
        acc_reduce=acc_reduce,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )
