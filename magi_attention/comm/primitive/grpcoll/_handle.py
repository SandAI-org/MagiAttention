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

from dataclasses import dataclass

import torch

__all__ = ["GrpCollHandle", "GrpCollIntraHandle", "GrpCollInterHandle"]


@dataclass
class GrpCollHandle:
    """This is a base dataclass for some meta tensors for symmetric group collective
    that will be passed to the group-reduce from the symmetric group-cast
    or directly passed to the cached group-cast to avoid notifying
    """


@dataclass
class GrpCollIntraHandle(GrpCollHandle):
    """Some meta tensors for intranode group collective
    that will be passed to the group-reduce from the symmetric group-cast
    or directly passed to the cached group-cast to avoid notifying

    rank_prefix_matrix: shape=[num_ranks, num_ranks]:
        rank_prefix_matrix[:, r]: the prefix sum of number of tokens (i.e. end idxs)
        sent by each rank to rank r calculated in notify stage

    channel_prefix_matrix: shape=[num_ranks, num_channels]:
        channel_prefix_matrix[r, :]: the prefix sum of send token end idxs
        sent by each send-channel to rank r calculated in notify stage

    recv_channel_prefix_matrix: shape=[num_ranks, num_channels]:
        recv_channel_prefix_matrix[r, :]: the prefix sum of recv token start idxs
        recv by each recv-channel from rank r calculated in group cast stage

    recv_src_idx: shape=[num_recv_tokens,]:
        the original token idx in the sender's buffer of each recv token
        so this is used in group reduce stage to indicate the original token position
        that each recv token should be reduced to calculated in group cast stage

    is_token_in_rank: shape=[num_tokens, num_ranks]
        is_token_in_rank[i][r]: whether ith token is sent to rank r (bool values)

    send_head: shape=[num_tokens, num_ranks]:
        send_head[i, r]: the offset in the corr. channel of send token i
        if it needs to be sent to rank r calculated in group cast stage
        NOTE: since the cached_channel_tail_idx starts at 0,
        when `token_idx == token_start_idx` for the corr. channel,
        the send_head[:, r] will be several cu_seqlens that look like:
            [0, 1, ... channel0_size, 0, 1, ... channel1_size, ...]
        and if is_token_in_rank[i, r] == -1, then send_head[i, r] == -1
        (and should be ignored in the cu_seqlens above)
    """

    rank_prefix_matrix: torch.Tensor
    channel_prefix_matrix: torch.Tensor
    recv_channel_prefix_matrix: torch.Tensor
    recv_src_idx: torch.Tensor
    is_token_in_rank: torch.Tensor
    send_head: torch.Tensor

    @property
    def num_recv_tokens(self) -> int:
        return self.recv_src_idx.size(0)


@dataclass
class GrpCollInterHandle(GrpCollHandle):
    """Some meta tensors for internode group collective
    that will be passed to the group-reduce from the symmetric group-cast
    or directly passed to the cached group-cast to avoid notifying

    is_token_in_rank: shape=[num_tokens, num_ranks]
        is_token_in_rank[i][r]: whether ith token is sent to rank r (bool values)

    rdma_channel_prefix_matrix: shape=[num_rdma_ranks, num_channels]:
        rdma_channel_prefix_matrix[r, :]: the prefix sum of send token end idxs sent by
        each send-channel to rdma rank r calculated in notify stage

    gbl_channel_prefix_matrix: shape=[num_ranks, num_channels]:
        gbl_channel_prefix_matrix[r, :]: the prefix sum of send token end idxs sent by
        each send-channel to rank r calculated in notify stage

    recv_rdma_channel_prefix_matrix: shape=[num_rdma_ranks, num_channels]:
        recv_rdma_channel_prefix_matrix[r, :]: the prefix sum of recv token end idxs recv by
        each recv-channel from rdma rank r calculated in group cast stage

    recv_rdma_rank_prefix_sum: shape=[num_rdma_ranks,]:
        the prefix sum of the number of tokens to recv from each rdma rank calculated in notify stage

    recv_gbl_channel_prefix_matrix: shape=[num_ranks, num_channels]:
        recv_gbl_channel_prefix_matrix[r, :]: the prefix sum of recv token start idxs
        recv by each recv-channel from global rank r calculated in group cast stage
        NOTE: the start idx is a global idx with rank prefix offsets,
        i.e. recv_gbl_channel_prefix_matrix[r, 0] does not start from 0 except for r == 0

    recv_gbl_rank_prefix_sum: shape=[num_ranks,]:
        the prefix sum of the number of tokens to recv from each global rank,
        thus recv_gbl_rank_prefix_sum[-1] == num_recv_tokens calculated in notify stage

    recv_src_meta: shape=[num_recv_tokens, sizeof(internode::SourceMeta)=8]:
        the source meta for each recv token calculated in group cast stage,
        NOTE: a `SourceMeta` struct object stores the src_rdma_rank
        and the is_token_in_nvl_rank_bits map of this recv token
        where the j-bit of is_token_in_nvl_rank_bits indicates
        whether this recv token needs to be sent to the j-th local rank of this node

    send_rdma_head: shape=[num_tokens, num_rdma_ranks]:
        send_rdma_head[i, r]: the offset in the corr. channel of send token i
        if it needs to be sent to rdma rank r calculated in group cast stage
        NOTE: since the rdma_tail_idx starts at 0 when token_idx == token_start_idx for the corr. channel
        thus the send_rdma_head[:, r] will be several cu_seqlens like:
            [0, 1, ... channel0_size, 0, 1, ... channel1_size, ...]
        and if all is_token_in_rank[i, r*8:(r+1)*8] == -1, then send_rdma_head[i, r] == -1 as well
        (and should be ignored in the cu_seqlens above)

    send_nvl_head: shape=[num_rdma_recv_tokens, num_local_ranks]:
        send_nvl_head[i, r]: the token offset of the ith recv token
        in the nvl forward "list" for local rank r calculated in group cast stage
        and if this recv token won't be sent to local rank r, then send_nvl_head[i, r] == -1 as well
    """

    is_token_in_rank: torch.Tensor
    rdma_channel_prefix_matrix: torch.Tensor
    gbl_channel_prefix_matrix: torch.Tensor
    recv_rdma_channel_prefix_matrix: torch.Tensor
    recv_rdma_rank_prefix_sum: torch.Tensor
    recv_gbl_channel_prefix_matrix: torch.Tensor
    recv_gbl_rank_prefix_sum: torch.Tensor
    recv_src_meta: torch.Tensor
    send_rdma_head: torch.Tensor
    send_nvl_head: torch.Tensor

    @property
    def num_recv_tokens(self) -> int:
        return self.recv_src_meta.size(0)

    @property
    def num_rdma_recv_tokens(self) -> int:
        return self.send_nvl_head.size(0)
