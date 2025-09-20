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
from typing import Literal

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

    _maybe_lazy_init_buffer(group)

    config: GrpCollConfig = grpcoll_mgr.get_config(group)
    buffer: GrpCollBuffer = grpcoll_mgr.get_buffer(group)
    handle_dict: dict[str, tuple] = kwargs.pop("native_grpcoll_handle_dict", {})
    handle: tuple | None = handle_dict.pop("group_cast", None)
    assert config is not None and buffer is not None

    # XXX: a workaround to implement native group-cast
    # by original deep-ep dispatch kernels
    # with pre-meta-args processing and post-output processing

    # transfer group-cast meta args to dispatch meta args
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
        x=input,
        recv_x=output,
        handle=handle,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        config=config,
        async_finish=async_op,
        allocate_on_comm_stream=False,
    )

    # prepare handle for symmetric group-reduce
    handle_dict["group_reduce"] = handle

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        # post_process_fn=lambda: recv_x,
        # XXX: remove this post-process
        post_process_fn=partial(
            unpermute_tensor,
            unperm_after_a2a_kwargs=range_gather_post_dispatch_kwargs,
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

    assert reduce_op == "sum", "Only support sum-reduce for now"

    _maybe_lazy_init_buffer(group)

    config: GrpCollConfig = kwargs.pop("native_grpcoll_config", None)
    buffer: GrpCollBuffer = kwargs.pop("native_grpcoll_buffer", None)
    handle_dict: dict[str, tuple] = kwargs.pop("native_grpcoll_handle_dict", {})
    handle: tuple | None = handle_dict.pop("group_reduce", None)
    assert config is not None and buffer is not None and handle is not None

    # XXX: a workaround to implement native group-reduce
    # by original deep-ep combine kernels
    # with pre-meta-args processing and pre-input processing

    # transfer symmetric group-cast meta args to dispatch meta args
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
    (combined_x, _, event) = buffer.combine(  # combined_topk_weights
        x=input,
        handle=handle,
        combined_x=output,
        config=config,
        async_finish=async_op,
        allocate_on_comm_stream=False,
        allow_empty_init_out_buf=kwargs.get("allow_empty_init_out_buf", False),
    )

    # prepare work with post-process
    work_with_post_process_fn = WorkWithPostProcessFn(
        work=GeneralWork(event),
        post_process_fn=lambda: combined_x,
    )

    return work_with_post_process_fn
