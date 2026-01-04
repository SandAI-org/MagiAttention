# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

import magi_attention
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import nvtx

from ._a2av_grpcoll_impl import a2av_group_cast_impl, a2av_group_reduce_impl
from ._group_collective_hier import (
    hier_group_cast_impl_with_a2av,
    hier_group_reduce_impl_with_a2av,
)
from ._native_grpcoll_impl import native_group_cast_impl, native_group_reduce_impl

__all__ = [
    "group_cast",
    "group_reduce",
]


# ------------------        group cast       ------------------ #


# host meta interface
@overload
def group_cast(
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
def group_cast(
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


@torch.no_grad()
@nvtx.instrument_nvtx
def group_cast(
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
    """Group cast interface

    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...]
        output (torch.Tensor | None): output tensor buffer with shape [output_seqlen, ...]
            or None to let the function allocate the output buffer itself

        input_split_sizes (list[int] | torch.Tensor):
            the 1D size list / tensor to split the input tensor,
            where sum(input_split_sizes) == input_seqlen

        output_split_sizes (list[int] | torch.Tensor):
            the 1D size list / tensor to split the output tensor,
            where sum(output_split_sizes) == output_seqlen, except that:
            if the seqlen of output buffer is not accurately known but has a maximum value,
            we actually allow sum(output_split_sizes) < output_seqlen,
            where the right-padded entries in the output buffer are ignored,
            which is helpful in some cases to avoid GPU-CPU sync

        dst_indices (list[list[int]] | torch.Tensor):
            the 2D destination rank indices list / tensor for each input split to send to,

            NOTE:
                1. if dst_indices is a 2D list, then len(dst_indices) == len(input_split_sizes),
                and dst_indices[i] is a list of distinct, valid destination ranks for the i-th input split to send to
                2. if dst_indices is a 2D tensor, then dst_indices.shape == [len(input_split_sizes), world_size],
                and dst_indices[i, :] indicates the distinct destination ranks for the i-th input split to send to
                where the right-padded entries should be filled with ``-1``

        src_index (list[int] | torch.Tensor):
            the 1D source rank index list / tensor for each output split to receive from,
            where len(src_index) == len(output_split_sizes)

            NOTE:
                1. the order of the output splits are "stable",
                i.e. the ones from the same source will be in the same order as the input splits
                2. if the length of src_index is not accurately known but has a maximum value,
                we actually allow src_index to be right-padded with invalid ``world_size``,
                where the corr. entries in the output_split_sizes are also supposed to be right-padded with ``0``,
                which is helpful in some cases to avoid GPU-CPU sync

        group (dist.ProcessGroup): the process group to comm
        async_op (bool): whether to use async op. Defaults to ``False``
        cast_lse (bool): whether to cast the log-sum-exp tensor along with the input tensor. Defaults to ``False``.

        input_lse (torch.Tensor | None): the log-sum-exp tensor w.r.t. the input tensor,
            only required and used when ``cast_lse`` is ``True``,
        output_lse (torch.Tensor | None): the log-sum-exp tensor w.r.t. the output tensor,
            only required and used when ``cast_lse`` is ``True``

            NOTE:
                1. the input/output lse tensors can have different dtypes from the ones of input/output tensors,
                and for now we only support ``float32`` dtype for lse tensors.
                2. the shapes of input/output lse tensors must be 3D shape as ``[seqlen, num_heads, head_dim]``,
                which are also supposed to be aligned with the ones of input/output tensors semantically

        kwargs: additional keyword arguments, varying from different implementations

    Returns:
        work_with_post_process_fn (WorkWithPostProcessFn): async work with the post-process function
    """

    if magi_attention.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group-cast
        return hier_group_cast_impl_with_a2av(
            input_tensor=input,
            output_tensor=output,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            dst_indices=dst_indices,
            src_index=src_index,
            group=group,
            async_op=async_op,
            cast_lse=cast_lse,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    if magi_attention.comm.is_native_grpcoll_enable():
        # NOTE: a feature under early development
        return native_group_cast_impl(
            input=input,
            output=output,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            dst_indices=dst_indices,
            src_index=src_index,
            group=group,
            async_op=async_op,
            cast_lse=cast_lse,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    # fall back to the a2a-v implementation
    return a2av_group_cast_impl(
        input=input,
        output=output,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        dst_indices=dst_indices,
        src_index=src_index,
        group=group,
        async_op=async_op,
        cast_lse=cast_lse,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )


# ------------------        group reduce       ------------------ #


# host meta interface
@overload
def group_reduce(
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
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def group_reduce(
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
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@torch.no_grad()
@nvtx.instrument_nvtx
def group_reduce(
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
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group reduce interface

    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...]
        output (torch.Tensor | None): output tensor buffer with shape [output_seqlen, ...]
            or None to let the function allocate the output buffer itself

        input_split_sizes (list[int] | torch.Tensor):
            the 1D size list / tensor to split the input tensor,
            where sum(input_split_sizes) == input_seqlen, except that:
            if the seqlen of input buffer is not accurately known but has a maximum value,
            we actually allow sum(input_split_sizes) < input_seqlen,
            where the right-padded entries in the input buffer are ignored,
            which is helpful in some cases to avoid GPU-CPU sync

        output_split_sizes (list[int] | torch.Tensor):
            the 1D size list / tensor to split the output tensor,
            where sum(output_split_sizes) == output_seqlen

        dst_index (list[int] | torch.Tensor):
            the 1D destination rank index list / tensor for each input split to send to,
            where len(dst_index) == len(input_split_sizes)

            NOTE:
                1. the order of the input splits are "stable",
                i.e. the ones to the same destination will be in the same order as the input splits
                2. if the length of dst_index is not accurately known but has a maximum value,
                we actually allow dst_index to be right-padded with invalid ``world_size``,
                where the corr. entries in the input_split_sizes are also supposed to be right-padded with ``0``,
                which is helpful in some cases to avoid GPU-CPU sync

        src_indices (list[list[int]] | torch.Tensor):
            the 2D source rank indices list / tensor for each output split to reduce from,

            NOTE:
                1. if src_indices is a 2D list, then len(src_indices) == len(output_split_sizes),
                and src_indices[i] is a list of distinct, valid source ranks for the i-th output split to reduce from
                2. if src_indices is a 2D tensor, then src_indices.shape == [len(output_split_sizes), world_size],
                and src_indices[i, :] indicates the distinct source ranks for the i-th output split to reduce from
                where the right-padded entries should be filled with ``-1``
                3. since any reduce operation satisfies the commutative property,
                the order to reduce to the same output split does not matter, except for numerical errors

        group (dist.ProcessGroup): the process group to comm
        async_op (bool): whether to use async op. Defaults to ``False``
        reduce_op (GroupReduceOp): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction

            NOTE:
                if reduce_op is "lse", the user is required to pass "input_lse" and "output_lse",
                and we only support input/output with shape [seqlen, num_heads, head_dim]
                while input_lse/output_lse with shape [seqlen, num_heads] for now

        acc_reduce (bool): whether to accumulate the reduction to the given output buffer. Defaults to ``True``.

            NOTE:
                if False, the output will be overwritten and the initial value will be ignored.
                Otherwise, the output buffer must be given and the initial value will be accumulated
                w.r.t. the reduction operation according to the ``reduce_op``.

        comm_dtype (torch.dtype | None): the communication dtype. Defaults to `input.dtype` if not given.

        input_lse (torch.Tensor | None): the log-sum-exp tensor for the input tensor,
            only required and used if reduce_op is "lse"
        output_lse (torch.Tensor | None): the log-sum-exp tensor for the output tensor,
            only required and used if reduce_op is "lse"

            NOTE:
                1. the input/output lse tensors can have different dtypes from the ones of input/output tensors,
                and for now we only support ``float32`` dtype for lse tensors.
                2. the shapes of input/output lse tensors must be 3D shape as ``[seqlen, num_heads, head_dim]``,
                which are also supposed to be aligned with the ones of input/output tensors semantically

        kwargs: additional keyword arguments, varying from different implementations

    Returns:
        work_with_post_process_fn (WorkWithPostProcessFn): async work with the post-process function
    """

    if magi_attention.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group collective
        # which might be deprecated when the native hierarchical group collective is ready
        return hier_group_reduce_impl_with_a2av(
            input_tensor=input,
            output_tensor=output,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            dst_index=dst_index,
            src_indices=src_indices,
            group=group,
            async_op=async_op,
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            comm_dtype=comm_dtype,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    if magi_attention.comm.is_native_grpcoll_enable():
        # NOTE: the new feature under development
        # which might be the default implementation in the future
        return native_group_reduce_impl(
            input=input,
            output=output,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            dst_index=dst_index,
            src_indices=src_indices,
            group=group,
            async_op=async_op,
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            comm_dtype=comm_dtype,
            input_lse=input_lse,
            output_lse=output_lse,
            **kwargs,
        )

    # fall back to the original a2a-v implementation
    return a2av_group_reduce_impl(
        input=input,
        output=output,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        dst_index=dst_index,
        src_indices=src_indices,
        group=group,
        async_op=async_op,
        reduce_op=reduce_op,
        acc_reduce=acc_reduce,
        comm_dtype=comm_dtype,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )
