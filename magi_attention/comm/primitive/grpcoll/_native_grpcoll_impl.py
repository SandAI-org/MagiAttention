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

import math
import warnings
from functools import partial
from typing import Any, Literal

import torch
import torch.distributed as dist

from magi_attention.utils import nvtx

from ...work import GeneralWork, WorkWithPostProcessFn
from ._buffer import GrpCollBuffer
from ._config import GrpCollConfig
from ._mgr import grpcoll_mgr
from .utils import transfer_group_cast_meta_to_dispatch_meta, unpermute_tensor

__all__ = [
    "native_group_cast_impl",
    "native_group_reduce_impl",
]


def _maybe_lazy_init_buffer(group: dist.ProcessGroup) -> None:
    if not grpcoll_mgr.is_registered(group):
        grpcoll_mgr.register_buffer(group=group)
        warnings.warn(
            f"Since the GrpCollBuffer is not registered for {group.group_name}, "
            "we lazily register it here with the default config, "
            "which might not be the best choice and cause performance/memory issue."
        )


def _native_group_cast_post_process(
    *args,
    recv_x: torch.Tensor,
    unperm_after_a2a_kwargs: dict,
    output_shape: torch.Size,
    **kwargs,
) -> torch.Tensor:
    return unpermute_tensor(
        tensor=recv_x,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs,
    ).view(output_shape)


def _native_group_reduce_post_process(
    *args,
    output: torch.Tensor,
    combined_x: torch.Tensor,
    output_shape: torch.Size,
    **kwargs,
) -> torch.Tensor:
    return output.add_(
        # TODO: move this view inside the buffer API
        combined_x.view(output_shape)
    )


@nvtx.instrument_nvtx
def native_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.ProcessGroup,
    async_op: bool = False,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Native group-cast implementation"""
    # TODO: move these checks inside of the buffer API
    # check
    # TODO: support other dtypes
    assert (
        input.dtype == output.dtype == torch.bfloat16
    ), "Only support bfloat16 for now"
    input_shape, output_shape = input.shape, output.shape
    hidden_size = math.prod(input_shape[1:])
    # FIXME: figure out the alignment requirement
    assert hidden_size % 256 == 0, (
        "The hidden size should be a multiple of 256, " f"but got {hidden_size=}."
    )

    # maybe lazy init buffer
    _maybe_lazy_init_buffer(group)

    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    meta_dict: dict[str, Any] = kwargs.pop("native_group_cast_meta_dict", {})
    handle_dict: dict[str, tuple] = kwargs.pop("native_grpcoll_handle_dict", {})
    # NOTE: here we had better use "get"
    # since the group_cast handle can be reused later
    handle: tuple | None = handle_dict.get("group_cast", None)
    assert config is not None and buffer is not None

    # XXX: a workaround to implement native group-cast
    # by original deep-ep dispatch kernels
    # with pre-meta-args processing and post-output processing

    # transfer group-cast meta args to dispatch meta args
    if meta_dict:
        num_tokens_per_rank = meta_dict["num_tokens_per_rank"]
        num_tokens_per_rdma_rank = meta_dict["num_tokens_per_rdma_rank"]
        is_token_in_rank = meta_dict["is_token_in_rank"]
        num_tokens_per_expert = meta_dict["num_tokens_per_expert"]
        range_gather_post_dispatch_kwargs = meta_dict[
            "range_gather_post_dispatch_kwargs"
        ]
    else:
        (
            _,  # rank_idx
            _,  # rdma_rank_idx
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            _,  # topk_idx
            _,  # topk_weights
            num_tokens_per_expert,
            range_gather_post_dispatch_kwargs,
            _,  # range_gather_pre_combine_kwargs
        ) = transfer_group_cast_meta_to_dispatch_meta(
            rank=group.rank(),
            num_ranks=group.size(),
            num_nodes=kwargs.pop("num_nodes", 1),
            num_local_experts=1,
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            use_topk=False,
            use_a2a_order_output=False,
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
        # TODO: move these view inside the buffer API
        x=input.view(input_shape[0], hidden_size),
        recv_x=output.view(output_shape[0], hidden_size),
        handle=handle,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
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
        # XXX: remove this post-process
        post_process_fn=partial(
            _native_group_cast_post_process,
            recv_x=recv_x,
            unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
            output_shape=output_shape,
        ),
    )

    return work_with_post_process_fn


@nvtx.instrument_nvtx
def native_group_reduce_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: Literal["sum", "avg", "lse"] = "sum",
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Native group-reduce implementation"""
    # TODO: move these checks inside of the buffer API
    # check
    assert reduce_op == "sum", "Only support sum-reduce for now"
    # TODO: support other dtypes
    assert (
        input.dtype == output.dtype == torch.bfloat16
    ), "Only support bfloat16 for now"
    input_shape, output_shape = input.shape, output.shape
    hidden_size = math.prod(input_shape[1:])
    # FIXME: figure out the alignment requirement
    assert hidden_size % 256 == 0, (
        "The hidden size should be a multiple of 256, " f"but got {hidden_size=}."
    )

    # maybe lazy init buffer
    _maybe_lazy_init_buffer(group)

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

    # XXX: a workaround to implement native group-reduce
    # by original deep-ep combine kernels
    # with pre-meta-args processing and pre-input processing

    # transfer symmetric group-cast meta args to dispatch meta args
    if meta_dict:
        range_gather_pre_combine_kwargs = meta_dict["range_gather_pre_combine_kwargs"]
    else:
        (
            _,  # rank_idx
            _,  # rdma_rank_idx
            _,  # num_tokens_per_rank
            _,  # num_tokens_per_rdma_rank
            _,  # is_token_in_rank
            _,  # topk_idx
            _,  # topk_weights
            _,  # num_tokens_per_expert
            _,  # range_gather_post_dispatch_kwargs
            range_gather_pre_combine_kwargs,
        ) = transfer_group_cast_meta_to_dispatch_meta(
            rank=group.rank(),
            num_ranks=group.size(),
            num_nodes=kwargs.pop("num_nodes", 1),
            num_local_experts=1,
            input_split_size_list=output_split_size_list,
            output_split_size_list=input_split_size_list,
            dst_indices_list=src_indices_list,
            src_index_list=dst_index_list,
            use_topk=False,
            use_a2a_order_output=False,
        )

    # permute input
    # XXX: remove this pre-process
    input = unpermute_tensor(
        tensor=input,
        unperm_after_a2a_kwargs=range_gather_pre_combine_kwargs,
    )

    # launch combine kernel
    combined_x, _, event = buffer.combine(  # combined_topk_weights
        # TODO: move this view inside the buffer API
        x=input.view(input_shape[0], hidden_size),
        handle=handle,
        # FIXME: since the combine kernel directly write to the buffer,
        # we have to init a new buffer and reduce to the given output buffer later
        combined_x=None,
        config=config,
        previous_event=None,
        async_finish=async_op,
        allocate_on_comm_stream=False,
        allow_empty_init_out_buf=kwargs.pop("allow_empty_init_out_buf", False),
    )

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        # XXX: remove this post-process
        post_process_fn=partial(
            _native_group_reduce_post_process,
            output=output,
            combined_x=combined_x,
            output_shape=output_shape,
        ),
    )

    return work_with_post_process_fn
