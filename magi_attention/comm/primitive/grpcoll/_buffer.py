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

# Copyright (c) 2025 DeepSeek. All Rights Reserved.
#
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
from typing import Callable, Union

import torch
import torch.distributed as dist

from magi_attention.common.enum import GroupReduceOp
from magi_attention.magi_attn_comm import grpcoll

from ._config import GrpCollConfig
from ._event import EventHandle, EventOverlap
from ._handle import GrpCollHandle, GrpCollInterHandle, GrpCollIntraHandle
from .utils import check_nvlink_connections

__all__ = ["GrpCollBuffer"]


reduce_op_str2int_map = {
    "sum": 0,
    "avg": 1,
    "lse": 2,
}


class GrpCollBuffer:
    """
    The core group collective buffer class with several group-collective comm kernel implementations:
        - high-throughput intranode group-collective (group-cast and group-reduce, using NVLink)
        - high-throughput internode group-collective (group-cast and group-reduce, using RDMA and NVLink)
        - low-latency group-collective (group-cast and group-reduce, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20
    hidden_size_alignment: int = 2 * 128
    hidden_size_ub: int = 64 * 128  # upper bound

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
        """

        check_nvlink_connections(group)

        # Initialize the CPP runtime
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy

        # TODO: make the runtime API torch compilable
        self.runtime = grpcoll.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
        )

        # Synchronize device IDs
        device_ids = [
            None,
        ] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        # Synchronize IPC handles
        ipc_handles = [
            None,
        ] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        # Synchronize NVSHMEM unique IDs
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # Enable IBGDA
            assert num_qps_per_rank > 0
            os.environ["NVSHMEM_DISABLE_P2P"] = (
                "0" if allow_nvlink_for_low_latency_mode else "1"
            )
            os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
            os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = f"{num_qps_per_rank}"
            # Make sure QP depth is always larger than the number of on-flight WRs, so that we can skip WQ slot check
            os.environ["NVSHMEM_QP_DEPTH"] = os.environ.get("NVSHMEM_QP_DEPTH", "1024")

            # Reduce gpu memory usage
            # 6 default teams + 1 extra team
            os.environ["NVSHMEM_MAX_TEAMS"] = "7"
            # Disable NVLink SHArP
            os.environ["NVSHMEM_DISABLE_NVLS"] = "1"
            # NOTES: NVSHMEM initialization requires at least 256 MiB
            os.environ["NVSHMEM_CUMEM_GRANULARITY"] = f"{2 ** 29}"

            if not allow_mnnvl:
                # Disable multi-node NVLink detection
                os.environ["NVSHMEM_DISABLE_MNNVL"] = "1"

            # Synchronize using the root ID
            nvshmem_unique_ids = [
                None,
            ] * self.group_size
            if (low_latency_mode and self.rank == 0) or (
                not low_latency_mode and self.runtime.get_rdma_rank() == 0
            ):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
            root_unique_id = nvshmem_unique_ids[
                0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)
            ]

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return grpcoll.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, "The SM count must be even"
        GrpCollBuffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(
            stream_id=ts.stream_id,
            device_index=ts.device_index,
            device_type=ts.device_type,
        )

    def get_local_buffer_tensor(
        self,
        dtype: torch.dtype,
        size: torch.Size | None = None,
        offset: int = 0,
        use_rdma_buffer: bool = False,
    ) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[: size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, torch.Tensor | torch.Tensor]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> GrpCollConfig:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: GrpCollConfig(GrpCollBuffer.num_sms, 24, 256, 6, 128),
            4: GrpCollConfig(GrpCollBuffer.num_sms, 6, 256, 6, 128),
            8: GrpCollConfig(GrpCollBuffer.num_sms, 6, 256, 6, 128),
            16: GrpCollConfig(GrpCollBuffer.num_sms, 36, 288, 20, 128),
            24: GrpCollConfig(GrpCollBuffer.num_sms, 8, 288, 32, 128),
            32: GrpCollConfig(GrpCollBuffer.num_sms, 32, 288, 32, 128),
            64: GrpCollConfig(GrpCollBuffer.num_sms, 20, 288, 28, 128),
            128: GrpCollConfig(GrpCollBuffer.num_sms, 20, 560, 32, 128),
            144: GrpCollConfig(GrpCollBuffer.num_sms, 32, 720, 12, 128),
            160: GrpCollConfig(GrpCollBuffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> GrpCollConfig:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: GrpCollConfig(GrpCollBuffer.num_sms, 10, 256, 6, 128),
            4: GrpCollConfig(GrpCollBuffer.num_sms, 9, 256, 6, 128),
            8: GrpCollConfig(GrpCollBuffer.num_sms, 4, 256, 6, 128),
            16: GrpCollConfig(GrpCollBuffer.num_sms, 4, 288, 12, 128),
            24: GrpCollConfig(GrpCollBuffer.num_sms, 1, 288, 8, 128),
            32: GrpCollConfig(GrpCollBuffer.num_sms, 1, 288, 8, 128),
            64: GrpCollConfig(GrpCollBuffer.num_sms, 1, 288, 20, 128),
            128: GrpCollConfig(GrpCollBuffer.num_sms, 1, 560, 12, 128),
            144: GrpCollConfig(GrpCollBuffer.num_sms, 2, 720, 8, 128),
            160: GrpCollConfig(GrpCollBuffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f"Unsupported number of EP ranks: {num_ranks}"
        return config_map[num_ranks]

    @classmethod
    def get_dispatch_meta_from_topk_idx(
        cls,
        topk_idx: torch.Tensor,
        num_ranks: int,
        num_nodes: int,
        num_experts: int,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_meta_stream: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """
        Calculate the dispatch meta from the topk indices required for later communication.

        NOTE:
            1. this is for now a static replacement API for ``buffer.get_dispatch_layout``
            when the buffer runtime is not available.

            2. this API is excuted not on the buffer comm stream but on a hidden ``meta_stream``
            since the buffer runtime is not available.
        """

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = grpcoll.Meta.get_dispatch_meta_from_topk_idx(
            topk_idx,
            num_ranks,
            num_nodes,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_meta_stream,
            None,  # auto set hidden meta_stream
        )

        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    @classmethod
    def get_a2av_perm_idx_from_src_idx(
        cls,
        output_split_sizes: torch.Tensor,
        src_idx: torch.Tensor,
        output_seqlen: int,
        num_ranks: int,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_meta_stream: bool = False,
    ) -> tuple[torch.Tensor, EventOverlap]:
        """
        Calculate the permutation indices to transfer the output buffer to all2all-v rank order

        Args:
            output_split_sizes (torch.Tensor): output split sizes tensor
            src_idx (torch.Tensor): src index tensor
            output_seqlen (int): the number of tokens of the output buffer
            num_ranks (int): the number of ranks
            previous_event (EventOverlap | None, optional):
                the event to wait before actually executing the kernel. Defaults to None.
            async_finish (bool, optional):
                the current stream will not wait for the meta kernels to be finished if set. Defaults to False.
            allocate_on_meta_stream (bool, optional):
                control whether all the allocated tensors' ownership to be on the hidden meta stream. Defaults to False.

        Returns:
            perm_to_a2av_idxs (torch.Tensor): the permutation indices to transfer the output buffer to all2all-v rank order,
                i.e. output[perm_to_a2av_idxs] => a2av_output
            event (EventOverlap): the event after executing the kernel (valid only if `async_finish` is set).
        """

        (
            perm_to_a2av_idx,
            event,
        ) = grpcoll.Meta.get_a2av_perm_idx_from_src_idx(
            output_split_sizes,
            src_idx,
            output_seqlen,  # num_tokens
            num_ranks,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_meta_stream,
            None,  # auto set hidden meta_stream
        )

        return (
            perm_to_a2av_idx,
            EventOverlap(event),
        )

    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, EventOverlap
    ]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = self.runtime.get_dispatch_layout(
            topk_idx,
            num_experts,
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
        )

        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            EventOverlap(event),
        )

    def dispatch(
        self,
        x: torch.Tensor,
        recv_x: torch.Tensor | None = None,
        handle: GrpCollHandle | None = None,
        num_tokens_per_rank: torch.Tensor | None = None,
        num_tokens_per_rdma_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        num_tokens_per_expert: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        config: GrpCollConfig | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
        GrpCollHandle,
        EventOverlap,
    ]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            recv_x: received tokens buffer to return, if given, or `None` to allocate a new buffer to return.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `torch.int64`, the expert indices selected by each token,
                `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            post_perm_idx: `[num_recv_tokens]` with `torch.int64`, the post-permutation indices of each token,
                i.e. recv_x[post_perm_idx] can recover to the original recv_x in rank order.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        is_out_buf_given = recv_x is not None

        # TODO: support other dtypes
        assert (
            x.dtype == torch.bfloat16  # type: ignore[union-attr]
        ), f"Only support bfloat16 input for now, but got {x.dtype=}."  # type: ignore[union-attr]
        assert (
            not is_out_buf_given or recv_x.dtype == torch.bfloat16  # type: ignore[union-attr]
        ), f"Only support bfloat16 output buffer for now, but got {recv_x.dtype=}."  # type: ignore[union-attr]

        # FIXME: figure out the alignment requirement
        hidden_shape = x.shape[1:]  # type: ignore[union-attr]
        hidden_size = math.prod(hidden_shape)
        assert hidden_size % GrpCollBuffer.hidden_size_alignment == 0, (
            f"The hidden size should be a multiple of {GrpCollBuffer.hidden_size_alignment}, "
            f"but got {hidden_size=}."
        )
        assert hidden_size < GrpCollBuffer.hidden_size_ub, (
            f"The hidden size should be less than {GrpCollBuffer.hidden_size_ub}, "
            f"but got {hidden_size=}."
        )
        if is_out_buf_given:
            assert recv_x.shape[1:] == hidden_shape, (  # type: ignore[union-attr]
                "The hidden shape (except dim0) of input and output buffer should be the same, "
                f"but got {x.shape=}, {recv_x.shape=}."  # type: ignore[union-attr]
            )

        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # View input/output to 2D shape
        x = x.view(-1, hidden_size)  # type: ignore[union-attr]
        if is_out_buf_given:
            recv_x = recv_x.view(-1, hidden_size)  # type: ignore[union-attr]

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_dispatch(
                x=x,
                recv_x=recv_x,
                config=config,
                handle=handle,
                hidden_shape=hidden_shape,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                post_perm_idx=post_perm_idx,
                expert_alignment=expert_alignment,
                num_worst_tokens=num_worst_tokens,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )

        # Intranode
        return self._intranode_dispatch(
            x=x,
            recv_x=recv_x,
            config=config,
            handle=handle,
            hidden_shape=hidden_shape,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            post_perm_idx=post_perm_idx,
            expert_alignment=expert_alignment,
            num_worst_tokens=num_worst_tokens,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

    def combine(
        self,
        x: torch.Tensor,
        handle: GrpCollHandle,
        combined_x: torch.Tensor | None = None,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        topk_weights: torch.Tensor | None = None,
        bias: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = None,
        pre_perm_idx: torch.Tensor | None = None,
        config: GrpCollConfig | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        allow_empty_init_out_buf: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            combined_x: received tokens buffer to return, if given, or `None` to allocate a new buffer to return.
            reduce_op (GroupReduceOp): the reduce operation to use. Defaults to "sum"
                - "sum": sum reduction
                - "avg": average reduction
                - "lse": log-sum-exp weighted average reduction, with lse correction
            acc_reduce (bool): whether to accumulate the reduction to the given combined_x buffer. Defaults to False.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`,
                the tokens' top-k weights for reducing to its original ranks.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            allow_empty_init_out_buf: whether to allow empty-initialize the output buffer if not given,
                this is useful when the user knows that no token is missed to be reduced in the output buffer
                such as during the ep communication scenario,
                but it is unsafe to set it to True in the general group-reduce case
            kwargs: additional arguments for the kernel.

        Returns:
            combined_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        is_out_buf_given = combined_x is not None

        # TODO: support other dtypes
        assert (
            x.dtype == torch.bfloat16
        ), f"Only support bfloat16 input for now, but got {x.dtype=}."
        assert (
            not is_out_buf_given or combined_x.dtype == torch.bfloat16  # type: ignore[union-attr]
        ), f"Only support bfloat16 output buffer for now, but got {combined_x.dtype=}."  # type: ignore[union-attr]
        # TODO: support other reduce ops
        assert reduce_op == "sum", "Only support sum-reduce for now"

        # FIXME: figure out the alignment requirement
        hidden_shape = x.shape[1:]
        hidden_size = math.prod(hidden_shape)
        assert hidden_size % GrpCollBuffer.hidden_size_alignment == 0, (
            f"The hidden size should be a multiple of {GrpCollBuffer.hidden_size_alignment}, "
            f"but got {hidden_size=}."
        )
        assert hidden_size < GrpCollBuffer.hidden_size_ub, (
            f"The hidden size should be less than {GrpCollBuffer.hidden_size_ub}, "
            f"but got {hidden_size=}."
        )
        if is_out_buf_given:
            assert combined_x.shape[1:] == hidden_shape, (  # type: ignore[union-attr]
                "The hidden shape (except dim0) of input and output buffer should be the same, "
                f"but got {x.shape=}, {combined_x.shape=}."  # type: ignore[union-attr]
            )

        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # View input/output to 2D shape
        x = x.view(-1, hidden_size)
        if is_out_buf_given:
            combined_x = combined_x.view(-1, hidden_size)  # type: ignore[union-attr]

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_combine(
                x=x,
                combined_x=combined_x,
                config=config,
                handle=handle,
                hidden_shape=hidden_shape,
                reduce_op=reduce_op,
                acc_reduce=acc_reduce,
                topk_weights=topk_weights,
                bias=bias,
                pre_perm_idx=pre_perm_idx,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
                allow_empty_init_out_buf=allow_empty_init_out_buf,
            )

        # Intranode
        return self._intranode_combine(
            x=x,
            combined_x=combined_x,
            config=config,
            handle=handle,
            hidden_shape=hidden_shape,
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            topk_weights=topk_weights,
            bias=bias,
            pre_perm_idx=pre_perm_idx,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
            allow_empty_init_out_buf=allow_empty_init_out_buf,
        )

    def _intranode_dispatch(
        self,
        x: torch.Tensor,
        recv_x: torch.Tensor | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        num_tokens_per_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        num_tokens_per_expert: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
        GrpCollIntraHandle,
        EventOverlap,
    ]:
        # Launch the kernel with cached or non-cached mode
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            assert isinstance(handle, GrpCollIntraHandle)

            (
                recv_x,
                _,  # recv_x_scales
                _,  # recv_topk_idx
                _,  # recv_topk_weights
                _,  # num_recv_tokens_per_expert_list
                _,  # rank_prefix_matrix
                _,  # channel_prefix_matrix
                _,  # recv_channel_prefix_matrix
                _,  # recv_src_idx
                _,  # send_head
                event,
            ) = self.runtime.intranode_dispatch(
                x,
                recv_x,
                None,  # x_scales
                None,  # topk_idx
                None,  # topk_weights
                None,  # num_tokens_per_rank
                handle.is_token_in_rank,
                None,  # num_tokens_per_expert
                handle.num_recv_tokens,
                handle.rank_prefix_matrix,
                handle.channel_prefix_matrix,
                post_perm_idx,
                expert_alignment,
                num_worst_tokens,
                config.to_kernel_config(),
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )

            # View output to hidden shape
            recv_x = recv_x.view(-1, *hidden_shape)

            return (  # type: ignore[return-value]
                recv_x,
                None,  # recv_topk_idx
                None,  # recv_topk_weights
                None,  # num_recv_tokens_per_expert_list
                handle,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )

            (
                recv_x,
                _,  # recv_x_scales
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                send_head,
                event,
            ) = self.runtime.intranode_dispatch(
                x,
                recv_x,
                None,  # x_scales
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,  # num_recv_tokens
                None,  # rank_prefix_matrix
                None,  # channel_prefix_matrix
                post_perm_idx,
                expert_alignment,
                num_worst_tokens,
                config.to_kernel_config(),
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )

            handle = GrpCollIntraHandle(
                rank_prefix_matrix=rank_prefix_matrix,
                channel_prefix_matrix=channel_prefix_matrix,
                recv_channel_prefix_matrix=recv_channel_prefix_matrix,
                recv_src_idx=recv_src_idx,
                is_token_in_rank=is_token_in_rank,
                send_head=send_head,
            )

            # View output to hidden shape
            recv_x = recv_x.view(-1, *hidden_shape)

            return (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    def _intranode_combine(
        self,
        x: torch.Tensor,
        combined_x: torch.Tensor | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        topk_weights: torch.Tensor | None = None,
        bias: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = None,
        pre_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        allow_empty_init_out_buf: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, EventOverlap]:
        assert isinstance(handle, GrpCollIntraHandle)

        bias_0, bias_1 = GrpCollBuffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, recv_topk_weights, event = self.runtime.intranode_combine(
            x,
            combined_x,
            topk_weights,
            bias_0,
            bias_1,
            pre_perm_idx,
            handle.recv_src_idx,  # src_idx
            handle.rank_prefix_matrix,  # rank_prefix_matrix
            handle.recv_channel_prefix_matrix,  # channel_prefix_matrix
            handle.send_head,  # send_head
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            reduce_op_str2int_map[reduce_op],
            acc_reduce,
            allow_empty_init_out_buf,
        )

        # View output to hidden shape
        combined_x = combined_x.view(-1, *hidden_shape)

        return combined_x, recv_topk_weights, EventOverlap(event)

    def _internode_dispatch(
        self,
        x: torch.Tensor,
        recv_x: torch.Tensor | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        num_tokens_per_rank: torch.Tensor | None = None,
        num_tokens_per_rdma_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        num_tokens_per_expert: torch.Tensor | None = None,
        topk_idx: torch.Tensor | None = None,
        topk_weights: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        expert_alignment: int = 1,
        num_worst_tokens: int = 0,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
        GrpCollInterHandle,
        EventOverlap,
    ]:
        """
        Internode dispatch implementation, for more details, please refer to the `dispatch` docs.
        Normally, you should not directly call this function.
        """
        assert post_perm_idx is None  # TODO: support post-perm for internode dispatch
        assert (
            num_worst_tokens == 0
        ), "Internode dispatch does not support `num_worst_tokens > 0`"

        # Launch the kernel with cached or non-cached mode
        if handle is not None:
            assert isinstance(handle, GrpCollInterHandle)
            assert topk_idx is None and topk_weights is None

            (
                recv_x,
                _,  # recv_x_scales
                _,  # recv_topk_idx
                _,  # recv_topk_weights
                _,  # num_recv_tokens_per_expert_list
                _,  # rdma_channel_prefix_matrix
                _,  # gbl_channel_prefix_matrix
                _,  # recv_rdma_channel_prefix_matrix
                _,  # recv_rdma_rank_prefix_sum
                _,  # recv_gbl_channel_prefix_matrix
                _,  # recv_gbl_rank_prefix_sum
                _,  # recv_src_meta
                _,  # send_rdma_head
                _,  # send_nvl_head
                event,
            ) = self.runtime.internode_dispatch(
                x,
                recv_x,
                None,  # x_scales
                None,  # topk_idx
                None,  # topk_weights
                None,  # num_tokens_per_rank
                None,  # num_tokens_per_rdma_rank
                handle.is_token_in_rank,
                None,  # num_tokens_per_expert
                handle.num_recv_tokens,
                handle.num_rdma_recv_tokens,
                handle.rdma_channel_prefix_matrix,
                handle.recv_rdma_rank_prefix_sum,
                handle.gbl_channel_prefix_matrix,
                handle.recv_gbl_rank_prefix_sum,
                expert_alignment,
                config.to_kernel_config(),
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )

            # View output to hidden shape
            recv_x = recv_x.view(-1, *hidden_shape)

            return (  # type: ignore[return-value]
                recv_x,
                None,  # recv_topk_idx
                None,  # recv_topk_weights
                None,  # num_recv_tokens_per_expert_list
                handle,
                EventOverlap(event),
            )
        else:
            assert (
                num_tokens_per_rank is not None
                and is_token_in_rank is not None
                and num_tokens_per_expert is not None
            )

            (
                recv_x,
                _,  # recv_x_scales
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum,
                recv_src_meta,
                send_rdma_head,
                send_nvl_head,
                event,
            ) = self.runtime.internode_dispatch(
                x,
                recv_x,
                None,  # x_scales
                topk_idx,
                topk_weights,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                is_token_in_rank,
                num_tokens_per_expert,
                0,
                0,
                None,
                None,
                None,
                None,
                expert_alignment,
                config.to_kernel_config(),
                getattr(previous_event, "event", None),
                async_finish,
                allocate_on_comm_stream,
            )

            handle = GrpCollInterHandle(
                is_token_in_rank=is_token_in_rank,
                rdma_channel_prefix_matrix=rdma_channel_prefix_matrix,
                gbl_channel_prefix_matrix=gbl_channel_prefix_matrix,
                recv_rdma_channel_prefix_matrix=recv_rdma_channel_prefix_matrix,
                recv_rdma_rank_prefix_sum=recv_rdma_rank_prefix_sum,
                recv_gbl_channel_prefix_matrix=recv_gbl_channel_prefix_matrix,
                recv_gbl_rank_prefix_sum=recv_gbl_rank_prefix_sum,
                recv_src_meta=recv_src_meta,
                send_rdma_head=send_rdma_head,
                send_nvl_head=send_nvl_head,
            )

            # View output to hidden shape
            recv_x = recv_x.view(-1, *hidden_shape)

            return (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                num_recv_tokens_per_expert_list,
                handle,
                EventOverlap(event),
            )

    def _internode_combine(
        self,
        x: torch.Tensor,
        combined_x: torch.Tensor | None,
        config: GrpCollConfig,
        handle: GrpCollHandle,
        hidden_shape: torch.Size,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        topk_weights: torch.Tensor | None = None,
        bias: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = None,
        pre_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
        allow_empty_init_out_buf: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, EventOverlap]:
        """
        Internode combine implementation, for more details, please refer to the `combine` docs.
        Normally, you should not directly call this function.
        """
        assert pre_perm_idx is None  # TODO: Support pre_perm_idx for internode combine
        assert isinstance(handle, GrpCollInterHandle)

        # Unpack handle and bias
        bias_0, bias_1 = GrpCollBuffer._unpack_bias(bias)

        # Launch the kernel
        combined_x, combined_topk_weights, event = self.runtime.internode_combine(
            x,
            combined_x,
            topk_weights,
            bias_0,
            bias_1,
            handle.recv_src_meta,  # src_meta
            handle.is_token_in_rank,  # is_combined_token_in_rank
            handle.recv_rdma_channel_prefix_matrix,  # rdma_channel_prefix_matrix
            handle.recv_rdma_rank_prefix_sum,  # rdma_rank_prefix_sum
            handle.recv_gbl_channel_prefix_matrix,  # gbl_channel_prefix_matrix
            handle.send_rdma_head,  # send_rdma_head
            handle.send_nvl_head,  # send_nvl_head
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_finish,
            allocate_on_comm_stream,
            reduce_op_str2int_map[reduce_op],
            acc_reduce,
            allow_empty_init_out_buf,
        )

        # View output to hidden shape
        combined_x = combined_x.view(-1, *hidden_shape)

        return combined_x, combined_topk_weights, EventOverlap(event)

    # TODO: deal with low-latency mode
    def clean_low_latency_buffer(
        self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int
    ) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        self.runtime.clean_low_latency_buffer(
            num_max_dispatch_tokens_per_rank, hidden, num_experts
        )

    def low_latency_dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        cumulative_local_expert_recv_stats: torch.Tensor | None = None,
        use_fp8: bool = True,
        round_scale: bool = False,
        use_ue8m0: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor], torch.Tensor, tuple, EventOverlap, Callable
    ]:
        """
        A low-latency implementation for dispatching with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden]`, only several hidden shapes are
                supported. The number of tokens to be dispatched must be less than `num_max_dispatch_tokens_per_rank`.
            topk_idx: `torch.Tensor` with `torch.int64`, shaped as `[num_tokens, num_topk]`, only several top-k shapes
                are supported. `-1` indices (not selecting any expert) are supported.
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            num_experts: the number of all experts.
            cumulative_local_expert_recv_stats: a cumulative expert count tensor for statistics, which should have shape
                `[num_local_experts]` and be typed as `torch.int`.
                This is useful for online service EP load balance monitoring.
            use_fp8: whether to enable FP8 casting, with this,
                the received data will be a tuple of FP8 tensor and scaling factors.
            round_scale: whether round the scaling factors into power of 2.
            use_ue8m0: whether use UE8M0 as scaling factor format (available only with `round_scale=True`).
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            event,
            hook,
        ) = self.runtime.low_latency_dispatch(
            x,
            topk_idx,
            cumulative_local_expert_recv_stats,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8,
            round_scale,
            use_ue8m0,
            async_finish,
            return_recv_hook,
        )
        handle = (
            packed_recv_src_info,
            packed_recv_layout_range,
            num_max_dispatch_tokens_per_rank,
            x.size(1),
            num_experts,
        )
        tensors_to_record = (
            x,
            topk_idx,
            packed_recv_x,
            packed_recv_x_scales,
            packed_recv_count,
            packed_recv_src_info,
            packed_recv_layout_range,
            cumulative_local_expert_recv_stats,
        )
        return (
            (packed_recv_x, packed_recv_x_scales) if use_fp8 else packed_recv_x,
            packed_recv_count,
            handle,
            EventOverlap(event, tensors_to_record if async_finish else None),  # type: ignore[arg-type]
            hook,
        )

    def low_latency_combine(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        handle: tuple,
        use_logfmt: bool = False,
        zero_copy: bool = False,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, EventOverlap, Callable]:
        """
        A low-latency implementation for combining tokens (reduce **with weights**) with IBGDA.
        This kernel requires all the ranks (no matter intranode or internode) should be visible via RDMA
            (specifically, IBGDA must be enabled).
        Warning: as there are only two buffers, and the returned tensors reuse the buffer, you cannot hold more than 2
            low-latency kernels' result tensors at a single moment.

        Arguments:
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]` with `torch.bfloat16`,
                the local calculated tokens to be sent to this original rank and reduced.
            topk_idx: `[num_combined_tokens, num_topk]` with `torch.int64`, the expert indices selected by the dispatched
                tokens. `-1` indices (not selecting any expert) are supported. Note that, `num_combined_tokens` equals
                to the number of dispatched tokens.
            topk_weights: `[num_combined_tokens, num_topk]` with `torch.float`, the expert weights selected by the dispatched
                tokens. The received tokens will be reduced with the weights in this tensor.
            handle: the communication handle given by the `dispatch` function.
            use_logfmt: whether to use an internal "LogFMT with dynamic per-64-channel cast" format (10 bits).
            zero_copy: whether the tensor is already copied into the RDMA buffer, should be cooperative
                with `get_next_low_latency_combine_buffer`.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle
        combined_x, event, hook = self.runtime.low_latency_combine(
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            num_experts,
            use_logfmt,
            zero_copy,
            async_finish,
            return_recv_hook,
            out,
        )
        tensors_to_record = (
            x,
            topk_idx,
            topk_weights,
            src_info,
            layout_range,
            combined_x,
        )
        return (
            combined_x,
            EventOverlap(event, tensors_to_record if async_finish else None),  # type: ignore[arg-type]
            hook,
        )

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine,
        so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        (
            src_info,
            layout_range,
            num_max_dispatch_tokens_per_rank,
            hidden,
            num_experts,
        ) = handle  # type: ignore[misc]
        return self.runtime.get_next_low_latency_combine_buffer(
            num_max_dispatch_tokens_per_rank,  # type: ignore[has-type]
            hidden,  # type: ignore[has-type]
            num_experts,  # type: ignore[has-type]
        )

    @staticmethod
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_ranks: int,
        num_experts: int,
    ) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return grpcoll.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts
        )
