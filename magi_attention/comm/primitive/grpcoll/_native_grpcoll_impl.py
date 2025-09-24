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

from typing import Any, overload

import torch
import torch.distributed as dist

from magi_attention.common.enum import ReduceOp
from magi_attention.utils import nvtx

from ...work import GeneralWork, WorkWithPostProcessFn
from ._buffer import GrpCollBuffer
from ._config import GrpCollConfig
from ._mgr import grpcoll_mgr
from .utils import (
    get_a2av_perm_idxs_from_group_cast_meta,
    get_dispatch_layout_from_group_cast_meta,
    maybe_lazy_init_buffer,
)

__all__ = [
    "native_group_cast_impl",
    "native_group_reduce_impl",
]


# ------------------        native group cast       ------------------ #


# host meta interface
@overload
def native_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_indices: list[list[int]],
    src_index: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def native_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_indices: torch.Tensor,
    src_index: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def native_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Native group-cast implementation"""
    # maybe lazy init buffer
    maybe_lazy_init_buffer(group)

    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    meta_dict: dict[str, Any] = kwargs.pop("native_group_cast_meta_dict", {})
    handle_dict: dict[str, tuple] = kwargs.pop("native_grpcoll_handle_dict", {})
    # NOTE: here we had better use "get"
    # since the group_cast handle can be reused later
    handle: tuple | None = handle_dict.get("group_cast", None)
    assert config is not None and buffer is not None

    # transfer group-cast meta args to dispatch meta args
    if meta_dict:
        num_tokens_per_rank = meta_dict["num_tokens_per_rank"]
        num_tokens_per_rdma_rank = meta_dict["num_tokens_per_rdma_rank"]
        is_token_in_rank = meta_dict["is_token_in_rank"]
        num_tokens_per_expert = meta_dict["num_tokens_per_expert"]
        post_perm_idx = meta_dict["post_perm_idx"]
    else:
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        ) = get_dispatch_layout_from_group_cast_meta(
            input_split_size_list=input_split_sizes,
            dst_indices_list=dst_indices,
            group=group,
            # HACK: leave a slot for topk_idx
            # since for now, we transfer the group_cast meta to it inside anyway
            # which is helpful in the ep/nsa communication scenario
            topk_idx=kwargs.pop("topk_idx", None),
        )

        # for group-cast, perm_to_a2av_idx is the post_perm_idx
        _, post_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_size_list=output_split_sizes,
            src_index_list=src_index,
            world_size=group.size(),
        )

    # launch dispatch kernel
    (
        recv_x,
        _,  # recv_topk_idx
        _,  # recv_topk_weights
        _,  # recv_num_tokens_per_expert_list
        handle,
        event,
    ) = buffer.dispatch(
        x=input,
        recv_x=output,
        handle=handle,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        post_perm_idx=post_perm_idx,
        config=config,
        previous_event=None,
        async_finish=async_op,
        allocate_on_comm_stream=False,
    )

    # HACK: prepare handle for symmetric group-reduce
    # by inserting an item into the handle dict
    handle_dict["group_reduce"] = handle

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        post_process_fn=lambda *args, **kwargs: recv_x,
        async_op=async_op,
    )

    return work_with_post_process_fn


# ------------------        native group reduce       ------------------ #


# host meta interface
@overload
def native_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int],
    output_split_sizes: list[int],
    dst_index: list[int],
    src_indices: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: ReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


# device meta interface
@overload
def native_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: torch.Tensor,
    output_split_sizes: torch.Tensor,
    dst_index: torch.Tensor,
    src_indices: torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: ReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    ...


@nvtx.instrument_nvtx
def native_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_index: list[int] | torch.Tensor,
    src_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: ReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Native group-reduce implementation"""
    # maybe lazy init buffer
    maybe_lazy_init_buffer(group)

    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    meta_dict: dict[str, Any] = kwargs.pop("native_group_reduce_meta_dict", {})
    handle_dict: dict[str, tuple] = kwargs.pop("native_grpcoll_handle_dict", {})
    # NOTE: here we had better use "pop"
    # since the group_reduce handle is always jit-prepared
    # by the symmetric group-cast
    handle: tuple | None = handle_dict.pop("group_reduce", None)
    assert config is not None and buffer is not None
    # FIXME: deal with the missing handle when only group-reduce is called
    assert handle is not None, (
        "No group-reduce handle given! "
        "Please run symmetric group-cast to prepare handle first."
    )

    # transfer symmetric group-cast meta args to dispatch meta args
    if meta_dict:
        pre_perm_idx = meta_dict["pre_perm_idx"]
    else:
        # for group-reduce, perm_to_a2av_idx is the pre_perm_idx
        _, pre_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_size_list=input_split_sizes,
            src_index_list=dst_index,
            world_size=group.size(),
        )

    # launch combine kernel
    combined_x, _, event = buffer.combine(
        x=input,
        handle=handle,
        combined_x=output,
        reduce_op=reduce_op,
        acc_reduce=acc_reduce,
        pre_perm_idx=pre_perm_idx,
        config=config,
        previous_event=None,
        async_finish=async_op,
        allocate_on_comm_stream=False,
        allow_empty_init_out_buf=kwargs.pop("allow_empty_init_out_buf", False),
        input_lse=input_lse,
        output_lse=output_lse,
    )

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        post_process_fn=lambda *args, **kwargs: combined_x,
        async_op=async_op,
    )

    return work_with_post_process_fn
