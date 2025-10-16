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

from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import nvtx

from ...work import GeneralWork, WorkWithPostProcessFn
from ._buffer import GrpCollBuffer
from ._config import GrpCollConfig
from ._handle import GrpCollHandle
from ._mgr import grpcoll_mgr
from .utils import (
    get_a2av_perm_idxs_from_group_cast_meta,
    get_dispatch_layout_from_group_cast_meta,
    get_group_reduce_handle_from_sym_group_cast,
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

    # get grpcoll config and buffer
    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    assert config is not None and buffer is not None

    # get meta dict and handle
    input_seqlen: int = input.size(0)
    output_seqlen: int | None = (
        output.size(0) if output is not None else kwargs.pop("output_seqlen", None)
    )
    meta_dict: dict[str, Any] = kwargs.pop("native_group_cast_meta_dict", {})
    handle_dict: dict[str, GrpCollHandle] = kwargs.pop("native_grpcoll_handle_dict", {})
    handle: GrpCollHandle | None = handle_dict.get("group_cast", None)

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
            input_split_sizes=input_split_sizes,
            dst_indices=dst_indices,
            group=group,
            input_seqlen=input_seqlen,
            # HACK: leave a slot for topk_idx
            # since for now, we transfer the group_cast meta to it inside anyway
            # which is helpful in the ep/nsa communication scenarios
            topk_idx=kwargs.pop("topk_idx", None),
        )

        # for group-cast, perm_to_a2av_idx is the post_perm_idx
        post_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_sizes=output_split_sizes,
            src_index=src_index,
            num_ranks=group.size(),
            output_seqlen=output_seqlen,
        )

    # launch dispatch kernel
    (
        recv_x,
        _,  # recv_lse
        handle,
        event,
    ) = buffer.group_cast(
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

    # HACK: prepare handle for symmetric group-reduce or cached group-cast
    handle_dict["group_cast"] = handle
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
    reduce_op: GroupReduceOp = "sum",
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
    reduce_op: GroupReduceOp = "sum",
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
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Native group-reduce implementation"""
    # maybe lazy init buffer
    maybe_lazy_init_buffer(group)

    # get grpcoll config and buffer
    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    assert config is not None and buffer is not None

    # get meta dict and handle
    input_seqlen: int = input.size(0)
    output_seqlen: int | None = (
        output.size(0) if output is not None else kwargs.pop("output_seqlen", None)
    )
    meta_dict: dict[str, Any] = kwargs.pop("native_group_reduce_meta_dict", {})
    handle_dict: dict[str, GrpCollHandle] = kwargs.pop("native_grpcoll_handle_dict", {})
    handle: GrpCollHandle | None = handle_dict.get("group_reduce", None)
    if handle is None:
        # FIXME: for now, we don't support individual group-reduce
        # since the necessary handle is not known until the symmetric group-cast returns
        handle = get_group_reduce_handle_from_sym_group_cast(
            input=input,
            output=output,
            input_split_sizes=input_split_sizes,
            output_split_sizes=output_split_sizes,
            dst_index=dst_index,
            src_indices=src_indices,
            group=group,
            async_op=async_op,
            output_seqlen=output_seqlen,
            topk_idx=kwargs.pop("topk_idx", None),
        )

    # transfer symmetric group-cast meta args to dispatch meta args
    if meta_dict:
        pre_perm_idx = meta_dict["pre_perm_idx"]
    else:
        # for group-reduce, perm_to_a2av_idx is the pre_perm_idx
        # the same as the post_perm_idx for symmetric group-cast
        pre_perm_idx = get_a2av_perm_idxs_from_group_cast_meta(
            output_split_sizes=input_split_sizes,
            src_index=dst_index,
            num_ranks=group.size(),
            output_seqlen=input_seqlen,
        )

    # launch combine kernel
    combined_x, event = buffer.group_reduce(
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

    # HACK: prepare handle for symmetric group-cast or cached group-reduce
    handle_dict["group_cast"] = handle
    handle_dict["group_reduce"] = handle

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        post_process_fn=lambda *args, **kwargs: combined_x,
        async_op=async_op,
    )

    return work_with_post_process_fn
