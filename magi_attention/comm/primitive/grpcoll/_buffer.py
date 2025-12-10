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

# mypy: disable-error-code="union-attr"
import math
import os
from typing import Callable

import torch
import torch.distributed as dist

from magi_attention.common.enum import GroupReduceOp
from magi_attention.utils import wrap_to_list

from ._config import GrpCollConfig
from ._event import EventHandle, EventOverlap
from ._handle import GrpCollHandle, GrpCollInterHandle, GrpCollIntraHandle
from .utils import check_nvlink_connections

is_magi_attn_comm_installed = False
try:
    from magi_attention.magi_attn_comm import grpcoll

    is_magi_attn_comm_installed = True
except ImportError:
    pass

__all__ = ["GrpCollBuffer"]


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

    # TODO: make it enum in cpp backend
    reduce_op_str2int_map = {
        "sum": 0,
        "avg": 1,
        "lse": 2,
    }

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        explicitly_destroy: bool = False,
    ) -> None:
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
        """

        # Checks
        assert (
            is_magi_attn_comm_installed
        ), "The `magi_attn_comm` extension module is not installed."
        check_nvlink_connections(group)

        # Set attributes
        self.rank = group.rank()
        self.group_size = group.size()
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy

        # Initialize the CPP backend of grpcoll buffer as runtime
        # TODO: make the runtime API compatible with `torch.compile`
        self.runtime = grpcoll.Buffer(
            self.rank,
            self.group_size,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            explicitly_destroy,
        )

        # All gather device IDs
        device_ids = [None] * self.group_size
        local_device_id = self.runtime.get_local_device_id()
        dist.all_gather_object(device_ids, local_device_id, group)

        # All gather NVLink IPC handles
        ipc_handles = [None] * self.group_size
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        dist.all_gather_object(ipc_handles, local_ipc_handle, group)

        # Get NVSHMEM unique IDs from the root rank
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

            # Disable multi-node NVLink detection
            os.environ["NVSHMEM_DISABLE_MNNVL"] = "1"

            # Broadcast NVSHMEM unique IDs from the root rank
            nvshmem_unique_ids = [None] * self.group_size
            if (low_latency_mode and self.rank == 0) or (
                not low_latency_mode and self.runtime.get_rdma_rank() == 0
            ):
                root_unique_id = self.runtime.get_local_nvshmem_unique_id()
            dist.all_gather_object(nvshmem_unique_ids, root_unique_id, group)
            root_unique_id = nvshmem_unique_ids[
                0 if low_latency_mode else self.runtime.get_root_rdma_rank(True)
            ]

        # Synchronize device IDs, NVLink IPC handles and NVSHMEM unique IDs
        # and make the runtime available
        assert (
            not self.runtime.is_available()
        ), "Runtime is already available before initialization"
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert (
            self.runtime.is_available()
        ), "Runtime is still not available after initialization"

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, "`explicitly_destroy` flag must be set"

        self.runtime.destroy()
        self.runtime = None

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

    @classmethod
    def get_group_cast_meta_from_t2r_idx(
        cls,
        t2r_idx: torch.Tensor,
        num_ranks: int,
        num_nodes: int,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_meta_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, EventOverlap]:
        """
        Calculate the dispatch meta from the topk indices required for later communication.

        NOTE:
            1. this is for now a static replacement API for `buffer.get_group_cast_meta`
            when the buffer runtime is not available.

            2. this API is excuted not on the buffer comm stream but on a hidden `meta_stream`
            since the buffer runtime is not available.
        """

        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            event,
        ) = grpcoll.Meta.get_group_cast_meta_from_t2r_idx(
            t2r_idx,
            num_ranks,
            num_nodes,
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_meta_stream,
            None,  # meta_stream, auto set
        )

        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
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
        async_op: bool = False,
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
            async_op (bool, optional):
                the current stream will not wait for the meta kernels to be finished if set. Defaults to False.
            allocate_on_meta_stream (bool, optional):
                control whether all the allocated tensors' ownership to be on the hidden meta stream. Defaults to False.

        Returns:
            perm_to_a2av_idxs (torch.Tensor): the permutation indices to transfer the output buffer to all2all-v rank order,
                i.e. output[perm_to_a2av_idxs] => a2av_output
            event (EventOverlap): the event after executing the kernel (valid only if `async_op` is set).
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
            async_op,
            allocate_on_meta_stream,
            None,  # meta_stream, auto set
        )

        return (
            perm_to_a2av_idx,
            EventOverlap(event),
        )

    def get_group_cast_meta(
        self,
        t2r_idx: torch.Tensor,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            t2r_idx: `[num_tokens, num_ranks]`, dtype must be `torch.int64`,
                the rank indices selected by each token, `-1` means no selections.
            previous_event: the event to wait before actually executing the kernel.
            async_op: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_op` is set).
        """
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            event,
        ) = self.runtime.get_group_cast_meta(
            t2r_idx,
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_comm_stream,
        )

        return (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            EventOverlap(event),
        )

    def group_cast(
        self,
        x: torch.Tensor | list[torch.Tensor],
        recv_x: torch.Tensor | list[torch.Tensor] | None = None,
        handle: GrpCollHandle | None = None,
        num_tokens_per_rank: torch.Tensor | None = None,
        num_tokens_per_rdma_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        config: GrpCollConfig | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        cast_lse: bool = False,
        lse: torch.Tensor | None = None,
        recv_lse: torch.Tensor | None = None,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor | None, GrpCollIntraHandle, EventOverlap
    ]:
        """
        Group cast tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: groups of tokens in a list of tensors to be sent simultaneously.
            recv_x: received tokens buffer to return, if given,
                or `None` to allocate a new buffer to return for each group of `x`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            post_perm_idx: `[num_recv_tokens]` with `torch.int64`, the post-permutation indices of each token,
                i.e. recv_x[post_perm_idx] can recover to the original recv_x in rank order.
            config: the performance tuning config if given.
            previous_event: the event to wait before actually executing the kernel if given.
            async_op: the current stream will not wait for the communication kernels to be finished if set.
                Defaults to `False`.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
                Defaults to `False`.

            cast_lse: whether to cast the given `lse` along with the x and write it to `recv_lse`.
                Defaults to `False`.
            lse: the logsumexp of each token in `x` for each attention head,
                with shape `[num_tokens, num_heads]`, to be sent along with `x`,
                only required and used when `cast_lse` is `True`.
            recv_lse: the logsumexp of each token in `recv_x` for each attention head,
                with shape `[num_recv_tokens, num_heads]`, to be received along with `recv_x`,
                when `cast_lse` is `True`.

        Returns:
            recv_x: received tokens for each group,
                with the same type and number of groups as the input group `x`,
                while the number of tokens equals to the received token count.
            recv_lse: the logsumexp of each token in `reduced_x` for each attention head,
                with shape `[num_recv_tokens, num_heads]`,
                valid if `cast_lse` is `True`, otherwise `None`.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_op` is set).
        """
        is_out_buf_given = recv_x is not None

        x = wrap_to_list(x)
        num_groups = len(x)
        if is_out_buf_given:
            assert recv_x is not None  # mypy
            recv_x = wrap_to_list(recv_x)
            assert len(recv_x) == len(x), (
                "The number of groups of input and output buffer should be the same, "
                f"but got {len(x)=}, {len(recv_x)=}."
            )

        hidden_shape = x[0].shape[1:]
        hidden_size = math.prod(hidden_shape)
        if is_out_buf_given:
            assert recv_x is not None  # mypy
            for i in range(num_groups):
                assert recv_x[i].shape[1:] == hidden_shape, (
                    "The hidden shape (except dim0) of input and output buffer should be the same, "
                    f"but got {x[i].shape=}, {recv_x[i].shape=}."
                )

        # Default config
        config = (
            GrpCollConfig.get_default_group_cast_config(self.group_size)
            if config is None
            else config
        )

        # View input/output to 2D shape
        for i in range(num_groups):
            x[i] = x[i].view(-1, hidden_size)
        if is_out_buf_given:
            assert recv_x is not None  # mypy
            for i in range(num_groups):
                recv_x[i] = recv_x[i].view(-1, hidden_size)

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_group_cast(
                x=x,
                recv_x=recv_x,
                config=config,
                handle=handle,
                hidden_shape=hidden_shape,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                post_perm_idx=post_perm_idx,
                previous_event=previous_event,
                async_op=async_op,
                allocate_on_comm_stream=allocate_on_comm_stream,
                cast_lse=cast_lse,
                lse=lse,
                recv_lse=recv_lse,
            )

        # Intranode
        return self._intranode_group_cast(
            x=x,
            recv_x=recv_x,
            config=config,
            handle=handle,
            hidden_shape=hidden_shape,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            post_perm_idx=post_perm_idx,
            previous_event=previous_event,
            async_op=async_op,
            allocate_on_comm_stream=allocate_on_comm_stream,
            cast_lse=cast_lse,
            lse=lse,
            recv_lse=recv_lse,
        )

    def group_reduce(
        self,
        x: torch.Tensor | list[torch.Tensor],
        handle: GrpCollHandle,
        reduced_x: torch.Tensor | list[torch.Tensor] | None = None,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        pre_perm_idx: torch.Tensor | None = None,
        config: GrpCollConfig | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        comm_dtype: torch.dtype | None = None,
        lse: torch.Tensor | None = None,
        reduced_lse: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, EventOverlap]:
        """
        Group reduce tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: groups of tokens in a list of tensors to be reduced simultaneously.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            reduced_x: received reduced tokens buffer to return, if given,
                or `None` to allocate a new buffer to return for each group of `x`.
            reduce_op (GroupReduceOp): the reduce operation to use. Defaults to "sum"
                - "sum": sum reduction
                - "avg": average reduction
                - "lse": log-sum-exp weighted average reduction, with lse correction
            acc_reduce (bool): whether to accumulate the reduction to the given reduced_x buffer. Defaults to False.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_op: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.
            comm_dtype: the communication dtype. Defaults to `x.dtype` if not given.

            lse: the logsumexp of each token in `x` for each attention head,
                with shape `[num_tokens, num_heads]`, to be sent along with `x`,
                only required and used when `reduce_op` is "lse".
            reduced_lse: the logsumexp of each token in `reduced_x` for each attention head,
                with shape `[num_recv_tokens, num_heads]`, to be received and reduced along with `reduced_x`,
                when `reduce_op` is "lse".

        Returns:
            reduced_x: reduced tokens for each group,
                with the same type and number of groups as the input group `x`,
                while the number of tokens equals to the received token count.
            reduced_lse: the reduced logsumexp of each token in `reduced_x` for each attention head,
                with shape `[num_recv_tokens, num_heads]`,
                valid if `reduce_op` is "lse", otherwise `None`.
            event: the event after executing the kernel (valid only if `async_op` is set).
        """
        is_out_buf_given = reduced_x is not None

        x = wrap_to_list(x)
        num_groups = len(x)
        if is_out_buf_given:
            assert reduced_x is not None  # mypy
            reduced_x = wrap_to_list(reduced_x)
            assert len(reduced_x) == len(x), (
                "The number of groups of input and output buffer should be the same, "
                f"but got {len(x)=}, {len(reduced_x)=}."
            )

        hidden_shape = x[0].shape[1:]
        hidden_size = math.prod(hidden_shape)
        if is_out_buf_given:
            assert reduced_x is not None  # mypy
            for i in range(num_groups):
                assert reduced_x[i].shape[1:] == hidden_shape, (
                    "The hidden shape (except dim0) of input and output buffer should be the same, "
                    f"but got {x[i].shape=}, {reduced_x[i].shape=}."
                )

        # Default config
        config = (
            GrpCollConfig.get_default_group_reduce_config(self.group_size)
            if config is None
            else config
        )

        # View input/output to 2D shape
        for i in range(num_groups):
            x[i] = x[i].view(-1, hidden_size)
        if is_out_buf_given:
            assert reduced_x is not None  # mypy
            for i in range(num_groups):
                reduced_x[i] = reduced_x[i].view(-1, hidden_size)

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self._internode_group_reduce(
                x=x,
                reduced_x=reduced_x,
                config=config,
                handle=handle,
                hidden_shape=hidden_shape,
                reduce_op=reduce_op,
                acc_reduce=acc_reduce,
                pre_perm_idx=pre_perm_idx,
                previous_event=previous_event,
                async_op=async_op,
                allocate_on_comm_stream=allocate_on_comm_stream,
                comm_dtype=comm_dtype,
                lse=lse,
                reduced_lse=reduced_lse,
            )

        # Intranode
        return self._intranode_group_reduce(
            x=x,
            reduced_x=reduced_x,
            config=config,
            handle=handle,
            hidden_shape=hidden_shape,
            reduce_op=reduce_op,
            acc_reduce=acc_reduce,
            pre_perm_idx=pre_perm_idx,
            previous_event=previous_event,
            async_op=async_op,
            allocate_on_comm_stream=allocate_on_comm_stream,
            comm_dtype=comm_dtype,
            lse=lse,
            reduced_lse=reduced_lse,
        )

    def _intranode_group_cast(
        self,
        x: list[torch.Tensor],
        recv_x: list[torch.Tensor] | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        num_tokens_per_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        cast_lse: bool = False,
        lse: torch.Tensor | None = None,
        recv_lse: torch.Tensor | None = None,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor | None, GrpCollIntraHandle, EventOverlap
    ]:
        """Intranode group cast implementation"""

        # Unpack handle if given
        is_handle_given = handle is not None
        if is_handle_given:  # cached mode
            assert isinstance(handle, GrpCollIntraHandle)
            num_tokens_per_rank = None
            is_token_in_rank = handle.is_token_in_rank
            num_recv_tokens = handle.num_recv_tokens
            rank_prefix_matrix = handle.rank_prefix_matrix
            channel_prefix_matrix = handle.channel_prefix_matrix
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None
            num_recv_tokens = 0
            rank_prefix_matrix = None
            channel_prefix_matrix = None

        # Prepare lse and recv_lse
        if cast_lse:
            assert lse is not None, "lse should not be None when `cast_lse` is set"
        else:  # no need to cast lse, even passed in
            lse = None
            recv_lse = None

        # Unpack (x,recv_x) groups
        num_groups = len(x)
        assert 1 <= num_groups <= 3, "num_groups only supports {1,2,3}"
        x_1st = x[0]
        recv_x_1st = recv_x[0] if recv_x is not None else None
        if num_groups > 1:
            x_2nd = x[1]
            recv_x_2nd = recv_x[1] if recv_x is not None else None
        else:
            x_2nd = None
            recv_x_2nd = None
        if num_groups > 2:
            x_3rd = x[2]
            recv_x_3rd = recv_x[2] if recv_x is not None else None
        else:
            x_3rd = None
            recv_x_3rd = None

        # Launch the intranode group cast kernel
        (
            recv_x_1st,
            recv_lse,
            recv_x_2nd,
            recv_x_3rd,
            # handle
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            # event
            event,
        ) = self.runtime.intranode_group_cast(
            x_1st,
            recv_x_1st,
            lse,
            recv_lse,
            x_2nd,
            recv_x_2nd,
            x_3rd,
            recv_x_3rd,
            num_tokens_per_rank,
            is_token_in_rank,
            num_recv_tokens,
            rank_prefix_matrix,
            channel_prefix_matrix,
            post_perm_idx,
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_comm_stream,
        )

        # Prepare the intranode handle for non-cached mode
        if not is_handle_given:
            handle = GrpCollIntraHandle(
                rank_prefix_matrix=rank_prefix_matrix,
                channel_prefix_matrix=channel_prefix_matrix,
                recv_channel_prefix_matrix=recv_channel_prefix_matrix,
                recv_src_idx=recv_src_idx,
                is_token_in_rank=is_token_in_rank,
                send_head=send_head,
            )

        # Pack recv_x groups
        recv_x = [
            recv_x_1st,
        ]
        if num_groups > 1:
            recv_x.append(recv_x_2nd)
        if num_groups > 2:
            recv_x.append(recv_x_3rd)

        # View output to hidden shape
        for i in range(num_groups):
            recv_x[i] = recv_x[i].view(-1, *hidden_shape)

        return (
            recv_x,
            recv_lse,
            handle,  # type: ignore[return-value]
            EventOverlap(event),
        )

    def _intranode_group_reduce(
        self,
        x: list[torch.Tensor],
        reduced_x: list[torch.Tensor] | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        pre_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        comm_dtype: torch.dtype | None = None,
        lse: torch.Tensor | None = None,
        reduced_lse: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, EventOverlap]:
        """Intranode group reduce implementation"""

        assert isinstance(handle, GrpCollIntraHandle)

        # Prepare lse and reduced_lse
        if reduce_op == "lse":
            assert lse is not None, "lse should not be None when `reduce_op == lse`"
        else:  # no need to reduce lse, even passed in
            lse = None
            reduced_lse = None

        # Unpack (x,reduced_x) groups
        num_groups = len(x)
        assert 1 <= num_groups <= 2, "num_groups only supports {1,2}"
        x_1st = x[0]
        reduced_x_1st = reduced_x[0] if reduced_x is not None else None
        if num_groups > 1:
            x_2nd = x[1]
            reduced_x_2nd = reduced_x[1] if reduced_x is not None else None
        else:
            x_2nd = None
            reduced_x_2nd = None

        # Launch the intranode group reduce kernel
        (
            reduced_x_1st,
            reduced_lse,
            reduced_x_2nd,
            # event
            event,
        ) = self.runtime.intranode_group_reduce(
            x_1st,
            reduced_x_1st,
            lse,
            reduced_lse,
            x_2nd,
            reduced_x_2nd,
            pre_perm_idx,
            handle.recv_src_idx,  # src_idx
            handle.rank_prefix_matrix,  # rank_prefix_matrix
            handle.recv_channel_prefix_matrix,  # channel_prefix_matrix
            handle.send_head,  # send_head
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_comm_stream,
            reduce_op,
            acc_reduce,
            comm_dtype,
        )

        # Pack reduced_x groups
        reduced_x = [
            reduced_x_1st,
        ]
        if num_groups > 1:
            reduced_x.append(reduced_x_2nd)

        # View output to hidden shape
        for i in range(num_groups):
            reduced_x[i] = reduced_x[i].view(-1, *hidden_shape)

        return (reduced_x, reduced_lse, EventOverlap(event))

    def _internode_group_cast(
        self,
        x: list[torch.Tensor],
        recv_x: list[torch.Tensor] | None,
        config: GrpCollConfig,
        handle: GrpCollHandle | None,
        hidden_shape: torch.Size,
        num_tokens_per_rank: torch.Tensor | None = None,
        num_tokens_per_rdma_rank: torch.Tensor | None = None,
        is_token_in_rank: torch.Tensor | None = None,
        post_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        cast_lse: bool = False,
        lse: torch.Tensor | None = None,
        recv_lse: torch.Tensor | None = None,
    ) -> tuple[
        list[torch.Tensor], torch.Tensor | None, GrpCollIntraHandle, EventOverlap
    ]:
        """Internode group cast implementation"""

        # Unpack handle if given
        is_handle_given = handle is not None
        if is_handle_given:  # cached mode
            assert isinstance(handle, GrpCollInterHandle)
            num_tokens_per_rank = None
            num_tokens_per_rdma_rank = None
            is_token_in_rank = handle.is_token_in_rank
            num_recv_tokens = handle.num_recv_tokens
            num_rdma_recv_tokens = handle.num_rdma_recv_tokens
            rdma_channel_prefix_matrix = handle.rdma_channel_prefix_matrix
            recv_rdma_rank_prefix_sum = handle.recv_rdma_rank_prefix_sum
            gbl_channel_prefix_matrix = handle.gbl_channel_prefix_matrix
            recv_gbl_rank_prefix_sum = handle.recv_gbl_rank_prefix_sum
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None
            num_recv_tokens = 0
            num_rdma_recv_tokens = 0
            rdma_channel_prefix_matrix = None
            recv_rdma_rank_prefix_sum = None
            gbl_channel_prefix_matrix = None
            recv_gbl_rank_prefix_sum = None

        # Prepare lse and recv_lse
        if cast_lse:
            assert lse is not None, "lse should not be None when `cast_lse` is set"
        else:  # no need to cast lse, even passed in
            lse = None
            # recv_lse = None

        # Unpack (x,recv_x) groups
        num_groups = len(x)
        assert num_groups == 1, "num_groups only supports {1,}"
        x_1st = x[0]
        recv_x_1st = recv_x[0] if recv_x is not None else None

        # Launch the internode group cast kernel
        (
            recv_x_1st,
            recv_lse,
            # handle
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            # event
            event,
        ) = self.runtime.internode_group_cast(
            x_1st,
            recv_x_1st,
            lse,
            recv_lse,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_recv_tokens,
            num_rdma_recv_tokens,
            rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            post_perm_idx,
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_comm_stream,
        )

        # Prepare the internode handle for non-cached mode
        if not is_handle_given:
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

        # Pack recv_x groups
        recv_x = [
            recv_x_1st,
        ]

        # View output to hidden shape
        for i in range(num_groups):
            recv_x[i] = recv_x[i].view(-1, *hidden_shape)

        return (
            recv_x,
            recv_lse,
            handle,  # type: ignore[return-value]
            EventOverlap(event),
        )

    def _internode_group_reduce(
        self,
        x: list[torch.Tensor],
        reduced_x: list[torch.Tensor] | None,
        config: GrpCollConfig,
        handle: GrpCollHandle,
        hidden_shape: torch.Size,
        reduce_op: GroupReduceOp = "sum",
        acc_reduce: bool = False,
        pre_perm_idx: torch.Tensor | None = None,
        previous_event: EventOverlap | None = None,
        async_op: bool = False,
        allocate_on_comm_stream: bool = False,
        comm_dtype: torch.dtype | None = None,
        lse: torch.Tensor | None = None,
        reduced_lse: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, EventOverlap]:
        """Internode group reduce implementation"""

        assert isinstance(handle, GrpCollInterHandle)

        # Prepare lse and reduced_lse
        if reduce_op == "lse":
            assert lse is not None, "lse should not be None when `reduce_op == lse`"
        else:  # no need to reduce lse, even passed in
            lse = None
            # reduced_lse = None

        # Unpack (x,reduced_x) groups
        num_groups = len(x)
        assert num_groups == 1, "num_groups only supports {1,}"
        x_1st = x[0]
        reduced_x_1st = reduced_x[0] if reduced_x is not None else None

        # Launch the internode group reduce kernel
        (
            reduced_x_1st,
            reduced_lse,
            event,
        ) = self.runtime.internode_group_reduce(
            x_1st,
            reduced_x_1st,
            lse,
            reduced_lse,
            handle.recv_src_meta,  # src_meta
            handle.is_token_in_rank,  # is_reduced_token_in_rank
            handle.recv_rdma_channel_prefix_matrix,  # rdma_channel_prefix_matrix
            handle.recv_rdma_rank_prefix_sum,  # rdma_rank_prefix_sum
            handle.recv_gbl_channel_prefix_matrix,  # gbl_channel_prefix_matrix
            handle.send_rdma_head,  # send_rdma_head
            handle.send_nvl_head,  # send_nvl_head
            pre_perm_idx,
            config.to_kernel_config(),
            getattr(previous_event, "event", None),
            async_op,
            allocate_on_comm_stream,
            reduce_op,
            acc_reduce,
            comm_dtype,
        )

        # Pack reduced_x groups
        reduced_x = [
            reduced_x_1st,
        ]

        # View output to hidden shape
        for i in range(num_groups):
            reduced_x[i] = reduced_x[i].view(-1, *hidden_shape)

        return (reduced_x, None, EventOverlap(event))  # reduced_lse

    # NOTE: remain original low-latency interface here for future potential usage,
    # which won't be exposed to users for now, but guaranteed its compatibility internally
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
        async_op: bool = False,
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
            x: `torch.Tensor` with `torch.bfloat16`, shaped as `[num_tokens, hidden_size]`, only several hidden shapes are
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
            async_op: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.

        Returns:
            recv_x: a tensor or tuple with received tokens for each expert.
                With `use_fp8=True`: the first element is a `torch.Tensor` shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden_size]` with `torch.float8_e4m3fn`.
                The second tensor is the corresponding scales for the first element with shape
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 128]` with `torch.float`,
                if `use_ue8m0=False`. With `use_ue8m0=True`, the second one is packed and shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden // 512]` with type `torch.int`.
                Notice that, the last-two-dimension of the scaling tensors are in column-major for TMA compatibility.
                With `use_fp8=False`, the result would be a tensor shaped as
                `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden_size]` with `torch.bfloat16`.
                Moreover, not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are,
                as we do not synchronize CPU received count with GPU (also not incompatible with CUDA graph if synced).
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receives. As mentioned before, not all tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_op` is set).
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
            async_op,
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
            EventOverlap(event, tensors_to_record if async_op else None),  # type: ignore[arg-type]
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
        async_op: bool = False,
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
            x: `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden_size]` with `torch.bfloat16`,
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
            async_op: the current stream will not wait for the communication kernels to be finished if set.
            return_recv_hook: return a receiving hook if set. If set, the kernel will just do the RDMA request issues,
                but **without actually receiving the data**. You must call the received hook to make sure the data's arrival.
                If you do not set this flag, the kernel will ensure the data's arrival.
            out: the in-place output tensor, if set, the kernel will write the result to this tensor and return it directly.

        Returns:
            combined_x: the reduced token tensor, with shape `[num_combined_tokens, hidden_size]` and type `torch.bfloat16`.
            event: the event after executing the kernel (valid only if `async_op` is set).
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
            async_op,
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
            EventOverlap(event, tensors_to_record if async_op else None),  # type: ignore[arg-type]
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
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden_size]`, you should fill this buffer
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

    @staticmethod
    def is_sm90_compiled() -> bool:
        return grpcoll.is_sm90_compiled()

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_hidden_size_alignment(dtype: torch.dtype) -> int:
        # At least for intranode group_reduce kernel,
        # it requires the `hidden_size` in int4 to be divisible by WARP_SIZE
        # thus for bf16/fp16, the hidden size alignment is:
        #   WARP_SIZE * sizeof(int4) / sizeof(dtype) = 32 * 16 / 2 = 256
        return 32 * 16 // dtype.itemsize
