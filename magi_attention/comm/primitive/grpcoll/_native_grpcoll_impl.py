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
from .utils import (
    get_combine_pre_process_args_from_group_reduce_meta,
    get_dispatch_layout_from_group_cast_meta,
    get_dispatch_post_process_args_from_group_cast_meta,
    unpermute_tensor,
)

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
    **kwargs,
) -> torch.Tensor:
    return unpermute_tensor(
        tensor=recv_x,
        unperm_after_a2a_kwargs=unperm_after_a2a_kwargs,
    )


@nvtx.instrument_nvtx
def native_group_cast_impl(
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
    """Native group-cast implementation"""
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
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
        ) = get_dispatch_layout_from_group_cast_meta(
            input_split_size_list=input_split_size_list,
            dst_indices_list=dst_indices_list,
            group=group,
            # HACK: leave a slot for topk_idx
            # since for now, we transfer the group_cast meta to it inside anyway
            # which is helpful in the ep/nsa communication scenario
            topk_idx=kwargs.pop("topk_idx", None),
        )

        range_gather_post_dispatch_kwargs = (
            get_dispatch_post_process_args_from_group_cast_meta(
                output_split_size_list=output_split_size_list,
                src_index_list=src_index_list,
                group=group,
            )
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
        ),
        async_op=async_op,
    )

    return work_with_post_process_fn


@nvtx.instrument_nvtx
def native_group_reduce_impl(
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
    """Native group-reduce implementation"""
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

    # transfer symmetric group-cast meta args to dispatch meta args
    if meta_dict:
        range_gather_pre_combine_kwargs = meta_dict["range_gather_pre_combine_kwargs"]
    else:
        range_gather_pre_combine_kwargs = (
            get_combine_pre_process_args_from_group_reduce_meta(
                input_split_size_list=input_split_size_list,
                dst_index_list=dst_index_list,
                group=group,
            )
        )

    # permute input
    # XXX: remove this pre-process
    input = unpermute_tensor(
        tensor=input,
        unperm_after_a2a_kwargs=range_gather_pre_combine_kwargs,
    )

    # launch combine kernel
    combined_x, _, event = buffer.combine(  # combined_topk_weights
        x=input,
        handle=handle,
        combined_x=output,
        reduce_op=reduce_op,
        acc_reduce=acc_reduce,
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
