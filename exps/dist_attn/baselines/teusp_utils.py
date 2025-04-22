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

from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat


def get_distributed_world_size(group: dist.ProcessGroup) -> int:
    """Return world size for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_world_size(group=group)


def get_distributed_rank(group: dist.ProcessGroup) -> int:
    """Return my rank for the distributed group."""
    assert torch.distributed.is_initialized(), "torch.distributed is not initialized."
    return torch.distributed.get_rank(group=group)


# sbhd / bshd -> thd data
def _get_unpad_data(attention_mask_thd: torch.Tensor, cu_seqlens: torch.Tensor):
    indices = torch.nonzero(attention_mask_thd, as_tuple=False).flatten()
    seqlens_in_batch = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    batch = len(cu_seqlens) - 1
    cu_seqlens_padded = torch.zeros(batch + 1, dtype=torch.int32)
    for i in range(batch):
        valid_len = seqlens_in_batch[i]
        end_idx = cu_seqlens[i + 1] - 1
        if end_idx + 1 < cu_seqlens[-1]:
            pad_len = indices[end_idx + 1] - indices[end_idx] - 1
        else:
            pad_len = len(attention_mask_thd) - indices[end_idx] - 1
        cu_seqlens_padded[i + 1] = cu_seqlens_padded[i] + valid_len + pad_len
    seqlens_in_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen_in_padded = seqlens_in_padded.max().item()
    return (indices, cu_seqlens_padded, max_seqlen_in_batch, max_seqlen_in_padded)
    # unpad_max_sequence = attention_mask.size(seq_dim)
    # seqlens_in_batch = attention_mask.sum(dim=seq_dim, dtype=torch.int32)
    # indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # max_seqlen_in_batch = seqlens_in_batch.max().item()
    # cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # cu_pad_needed = _get_te_pad_needed(seqlens_in_batch, padding_factor)
    # cu_padlens = cu_pad_needed + seqlens_in_batch
    # max_seqlen_in_padded = cu_padlens.max().item()
    # cu_seqlens_padded = F.pad(torch.cumsum(cu_padlens, dim=0, dtype=torch.int32), (1, 0))
    # indices_in_thd_padded = _get_indices_in_thd_padded(indices, cu_seqlens_padded, unpad_max_sequence)
    # return (
    #     indices,
    #     indices_in_thd_padded,
    #     cu_seqlens,
    #     cu_seqlens_padded,
    #     max_seqlen_in_batch,
    #     max_seqlen_in_padded
    # )


# thd -> thd data
def _get_varlen_unpad_data(
    attention_mask_thd: torch.Tensor, cu_seqlens: torch.Tensor, padding_factor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    indices = torch.nonzero(attention_mask_thd, as_tuple=False).flatten()
    seqlens_in_batch = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_pad_needed = _get_te_pad_needed(seqlens_in_batch, padding_factor)
    cu_padlens = cu_pad_needed + seqlens_in_batch
    max_seqlen_in_padded = cu_padlens.max().item()
    cu_seqlens_padded = F.pad(
        torch.cumsum(cu_padlens, dim=0, dtype=torch.int32), (1, 0)
    )
    indices_in_thd_padded = _get_indices_in_thd_padded2(
        seqlens_in_batch, cu_seqlens_padded
    )
    return (
        indices,
        indices_in_thd_padded,
        cu_seqlens_padded,
        max_seqlen_in_batch,
        max_seqlen_in_padded,
    )


# load balance 2d repad
def _get_zigzag2d_varlen_unpad_data(
    attention_mask_thd: torch.Tensor,
    cu_seqlens: torch.Tensor,
    padding_factor_p2p,
    padding_factor_a2a,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    indices = torch.nonzero(attention_mask_thd, as_tuple=False).flatten()
    seqlens_in_batch = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_pad_needed = _get_te_pad_needed(seqlens_in_batch, padding_factor_p2p)
    cu_padlens = cu_pad_needed + seqlens_in_batch
    # a2a 2d pad
    total_seqlen_p2p = cu_padlens.sum().item()
    last_pad_needed = (
        total_seqlen_p2p + padding_factor_a2a - 1
    ) // padding_factor_a2a * padding_factor_a2a - total_seqlen_p2p
    cu_padlens[-1] += last_pad_needed
    max_seqlen_in_padded = cu_padlens.max().item()
    cu_seqlens_padded = F.pad(
        torch.cumsum(cu_padlens, dim=0, dtype=torch.int32), (1, 0)
    )
    indices_in_thd_padded = _get_indices_in_thd_padded2(
        seqlens_in_batch, cu_seqlens_padded
    )
    return (
        indices,
        indices_in_thd_padded,
        cu_seqlens_padded,
        max_seqlen_in_batch,
        max_seqlen_in_padded,
    )


# pad to padding_factor‘s integer multiple
def _get_te_pad_needed(
    seqlens_in_batch: torch.Tensor, padding_factor: int
) -> torch.Tensor:
    pad_needed = (seqlens_in_batch + padding_factor - 1) // (padding_factor) * (
        padding_factor
    ) - seqlens_in_batch
    return pad_needed


# token‘s thd index to cu_seqlens_padded
def _get_indices_in_thd_padded(
    indices: torch.Tensor, cu_seqlens_padded: torch.Tensor, unpad_max_sequence: int
) -> torch.Tensor:
    batch_id = indices // unpad_max_sequence
    indice_offset_in_batch = indices % unpad_max_sequence
    indices_in_thd_padded = indice_offset_in_batch + cu_seqlens_padded[batch_id]
    return indices_in_thd_padded


# token's thd index to cu_seqlens_padded
def _get_indices_in_thd_padded2(
    seqlens_in_batch: torch.Tensor, cu_seqlens_padded: torch.Tensor
) -> torch.Tensor:
    batch = len(seqlens_in_batch)
    indices: list[int] = []
    for i in range(batch):
        start_idx_padded = cu_seqlens_padded[i]
        valid_length = seqlens_in_batch[i]
        indices.extend(range(start_idx_padded, start_idx_padded + valid_length))

    return torch.tensor(indices, device=cu_seqlens_padded.device, dtype=torch.int32)


# load balance a2a+p2p chunk indices
def _get_chunk_indices_on_cp_rank(
    cu_seqlens_padded: torch.Tensor,
    cp_size_a2a: int,
    rank_a2a: int,
    cp_size_p2p: int,
    rank_p2p: int,
):
    device = cu_seqlens_padded.device
    chunks_p2p = []
    # interate
    for i in range(len(cu_seqlens_padded) - 1):
        start_idx = cu_seqlens_padded[i]
        end_idx = cu_seqlens_padded[i + 1]
        seqlen = end_idx - start_idx
        assert seqlen % (2 * cp_size_p2p * cp_size_a2a) == 0
        # ring
        chunk_size_p2p = seqlen // (2 * cp_size_p2p)
        chunk_start_1 = start_idx + rank_p2p * chunk_size_p2p
        chunk_end_1 = chunk_start_1 + chunk_size_p2p
        chunk_start_2 = start_idx + (2 * cp_size_p2p - rank_p2p - 1) * chunk_size_p2p
        chunk_end_2 = chunk_start_2 + chunk_size_p2p
        chunk_p2p = torch.cat(
            [
                torch.arange(chunk_start_1, chunk_end_1, device=device),
                torch.arange(chunk_start_2, chunk_end_2, device=device),
            ]
        )
        chunks_p2p.append(chunk_p2p)
    # ulysess
    chunks_a2a = torch.cat(chunks_p2p).to(device=device)
    total_seqlen = cu_seqlens_padded[-1]
    chunk_size_a2a = total_seqlen // cp_size_p2p // cp_size_a2a
    chunk_start_a2a = rank_a2a * chunk_size_a2a
    # print(chunk_start_a2a,chunk_start_a2a+chunk_size_a2a)
    return chunks_a2a[chunk_start_a2a : chunk_start_a2a + chunk_size_a2a]


def all_to_all_3D(
    input: torch.tensor,
    scatter_idx: int = 1,
    gather_idx: int = 0,
    group=None,
    use_sync: bool = False,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group
        use_sync (bool): whether to synchronize after all-to-all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 3
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 1 and gather_idx == 0:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 1)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, shard_hc, hs).contiguous()

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        # output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 0 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)
        # -> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(1, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 1).contiguous()

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll3D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync
        # input_ = input.unsqueeze(0)
        ret = all_to_all_3D(
            input, scatter_idx, gather_idx, group=group, use_sync=use_sync
        )
        # ret = ret.squeeze(0)
        return ret

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None, None]:
        # print('backward',len(grad_output),type(grad_output))
        # grad_output_ = grad_output.unsqueeze(0)
        grad_ret = all_to_all_3D(
            grad_output,
            ctx.gather_idx,
            ctx.scatter_idx,
            group=ctx.group,
            use_sync=ctx.use_sync,
        )
        # grad_ret = grad_ret.squeeze(0)
        return (None, grad_ret, None, None, None)


def fa_varlen_thd_unpad(
    input: torch.Tensor,
    cu_seqlens_per_step: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    packed=False,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_per_step = cu_seqlens_per_step[1:] - cu_seqlens_per_step[:-1]
    max_seqlen_per_step = seqlens_per_step.max().item()
    batch = len(seqlens_per_step)
    indices: list[int] = []
    for i in range(batch):
        start_idx_padded = cu_seqlens_padded[i]
        valid_length = seqlens_per_step[i]
        indices.extend(range(start_idx_padded, start_idx_padded + valid_length))
    unpad_indices = torch.tensor(indices, device=input.device, dtype=torch.int64)

    if packed:
        other_shape = input[0].shape[1:]
    else:
        other_shape = input.shape[1:]
    second_dim = other_shape.numel()

    if packed:
        unpad_0 = (
            torch.gather(
                rearrange(input[0], "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
        unpad_1 = (
            torch.gather(
                rearrange(input[1], "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
        unpad_input = torch.stack([unpad_0, unpad_1], dim=0)
    else:
        unpad_input = (
            torch.gather(
                rearrange(input, "b ... -> b (...)"),
                0,
                repeat(unpad_indices, "z -> z d", d=second_dim),
            )
            .reshape(-1, *other_shape)
            .contiguous()
        )
    return (unpad_input, unpad_indices, max_seqlen_per_step)


def fa_varlen_thd_pad(
    input: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpad_indices: torch.Tensor,
    packed=False,
) -> torch.Tensor:
    total_padded_len = cu_seqlens_padded[-1]
    if packed:
        other_shape = input[0].shape[1:]
    else:
        other_shape = input.shape[1:]
    second_dim = other_shape.numel()
    if packed:
        pad_input_0 = torch.zeros(
            total_padded_len, second_dim, device=input[0].device, dtype=input[0].dtype
        )
        pad_input_1 = torch.zeros(
            total_padded_len, second_dim, device=input[0].device, dtype=input[0].dtype
        )
        input_0 = rearrange(input[0], "b ... -> b (...)")
        input_1 = rearrange(input[1], "b ... -> b (...)")
        pad_input_0.scatter_(
            0, repeat(unpad_indices, "z -> z d", d=second_dim), input_0
        )
        pad_input_1.scatter_(
            0, repeat(unpad_indices, "z -> z d", d=second_dim), input_1
        )
        pad_input_0 = pad_input_0.reshape(-1, *other_shape).contiguous()
        pad_input_1 = pad_input_1.reshape(-1, *other_shape).contiguous()
        pad_input = torch.stack([pad_input_0, pad_input_1], dim=0)
    else:
        pad_input = torch.zeros(
            total_padded_len, second_dim, device=input.device, dtype=input.dtype
        )
        input = rearrange(input, "b ... -> b (...)")
        pad_input.scatter_(0, repeat(unpad_indices, "z -> z d", d=second_dim), input)
        pad_input = pad_input.reshape(-1, *other_shape).contiguous()
    return pad_input


def fa_thd_unpad(input: torch.Tensor, indices: torch.Tensor, qkv_format: str):
    if qkv_format == "bshd":
        _input = input.view(-1, *input.shape[2:]).contiguous()
    elif qkv_format == "sbhd":
        _input = (
            input.permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, *input.shape[2:])
            .contiguous()
        )
    else:
        _input = input
    other_shape = _input.shape[1:]
    second_dim = other_shape.numel()
    unpad_input = (
        torch.gather(
            rearrange(_input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        )
        .reshape(-1, *other_shape)
        .contiguous()
    )
    return unpad_input


def fa_thd_pad(
    input: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    qkv_format: str,
):
    total_padded_len = cu_seqlens_padded[-1]
    batch = len(cu_seqlens_padded) - 1
    other_shape = input.shape[1:]
    second_dim = other_shape.numel()
    pad_input = torch.zeros(
        total_padded_len, second_dim, device=input.device, dtype=input.dtype
    )
    input = rearrange(input, "b ... -> b (...)")
    pad_input.scatter_(0, repeat(indices, "z -> z d", d=second_dim), input)
    pad_input = pad_input.reshape(-1, *other_shape).contiguous()
    if qkv_format == "bshd":
        pad_input = pad_input.view(batch, -1, *other_shape).contiguous()
    elif qkv_format == "sbhd":
        pad_input = (
            pad_input.view(batch, -1, *other_shape).permute(1, 0, 2, 3).contiguous()
        )
    return pad_input


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


# def fa_varlen_lse_repad(lse: torch.Tensor, cu_seqlens, indices: torch.Tensor, max_seqlen_pad: int):
#     flatten_lse = flatten_varlen_lse(lse, cu_seqlens)
#     batch = lse.shape[0]
#     pad_lse = torch.zeros((flatten_lse.shape[0], batch*max_seqlen_pad), device=lse.device, dtype=lse.dtype)
#     pad_lse.scatter_(1, repeat(indices, "z -> d z", d=flatten_lse.shape[0]), flatten_lse)
#     return pad_lse.view(lse.shape[1], batch, max_seqlen_pad).permute(1, 0, 2).contiguous()


def fa_varlen_lse_repad(lse: torch.Tensor, max_seqlen_pad: int):
    max_seqlen_pad_prime = lse.shape[2]
    repad_lse = torch.zeros(
        (lse.shape[0], lse.shape[1], max_seqlen_pad), device=lse.device, dtype=lse.dtype
    )
    lse_seq_len = min(max_seqlen_pad_prime, max_seqlen_pad)
    repad_lse[:, :, :lse_seq_len] = lse[:, :, :lse_seq_len]
    return repad_lse


def fa_varlen_lse_unpad(
    input: torch.Tensor,
    unpad_indices: torch.Tensor,
):
    head_dim = input.shape[0]
    unpad_input = (
        torch.gather(
            input,
            1,
            repeat(unpad_indices, "z -> d z", d=head_dim),
        )
        .reshape(head_dim, -1)
        .contiguous()
    )
    return unpad_input


def fa_varlen_lse_pad(
    input: torch.Tensor, total_padded_len: int, unpad_indices: torch.Tensor
):
    head_dim = input.shape[0]
    pad_input = torch.zeros(
        head_dim, total_padded_len, device=input.device, dtype=input.dtype
    )
    pad_input.scatter_(1, repeat(unpad_indices, "z -> d z", d=head_dim), input)
    pad_input = pad_input.reshape(head_dim, -1).contiguous()
    return pad_input


def fa_varlen_lse_pad2(
    input: torch.Tensor,
    cu_seqlens_per_step: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
):
    seqlens_per_step = cu_seqlens_per_step[1:] - cu_seqlens_per_step[:-1]
    # max_seqlen_per_step = seqlens_per_step.max().item()
    batch = len(seqlens_per_step)
    indices: list[int] = []
    for i in range(batch):
        start_idx_padded = cu_seqlens_padded[i]
        valid_length = seqlens_per_step[i]
        indices.extend(range(start_idx_padded, start_idx_padded + valid_length))
    unpad_indices = torch.tensor(indices, device=input.device, dtype=torch.int64)

    total_padded_len = cu_seqlens_padded[-1]
    head_dim = input.shape[0]
    pad_input = torch.zeros(
        head_dim, total_padded_len, device=input.device, dtype=input.dtype
    )
    pad_input.scatter_(1, repeat(unpad_indices, "z -> d z", d=head_dim), input)
    pad_input = pad_input.reshape(head_dim, -1).contiguous()
    return pad_input, unpad_indices


def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log(1 + torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


#########################################
# adpated from InternEvo LoongTrain
#########################################


class _SeqAllToAll(torch.autograd.Function):
    "sequence alltoall function"

    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        # scatter_idx: Optional[Union[List[int], int]],
        # gather_idx: Optional[Union[List[int], int]],
        scatter_idx: int,
        gather_idx: int,
        *input_: torch.Tensor,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        seq_world_size = dist.get_world_size(group)

        if dist.get_world_size(group) <= 1:
            if len(input_) == 1:
                return input_[0]
            return input_

        if len(input_) == 1:
            input_list = [
                t.contiguous()
                for t in torch.tensor_split(input_[0], seq_world_size, scatter_idx)
            ]
            output_list = [
                torch.empty_like(input_list[0]) for _ in range(seq_world_size)
            ]
            # TODO: use all_to_all_single instead
            dist.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()

        # outputs = []

        # assert len(scatter_idx) == len(gather_idx)
        # assert len(gather_idx) == len(input_)

        # for i in range(len(input_)):

        #     if i == 0:
        #         input_list = [t.contiguous() for t in torch.tensor_split(input_[i], seq_world_size, scatter_idx[i])]
        #         output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        #         handle_last = dist.all_to_all(output_list, input_list, group=group, async_op=True)

        #     # conduct the next all2all
        #     if i + 1 < len(input_):
        #         input_list_next = [
        #             t.contiguous() for t in torch.tensor_split(input_[i + 1], seq_world_size, scatter_idx[i + 1])
        #         ]
        #         output_list_next = [torch.empty_like(input_list_next[0]) for _ in range(seq_world_size)]
        #         handle_next = dist.all_to_all(output_list_next, input_list_next, group=group, async_op=True)

        #     handle_last.wait()

        #     outputs.append(torch.cat(output_list, dim=gather_idx[i]).contiguous())

        #     if i + 1 < len(input_):
        #         handle_last = handle_next
        #         input_list = input_list_next
        #         output_list = output_list_next

        # return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_output: torch.Tensor):
        if dist.get_world_size(ctx.group) <= 1:
            return (None, None, None, *grad_output)
        res = _SeqAllToAll.apply(
            ctx.group, ctx.gather_idx, ctx.scatter_idx, *grad_output
        )
        if len(grad_output) == 1:
            return (None, None, None, res)

        return (None, None, None, *res)


#########################################
# adpated from InternEvo LoongTrain
#########################################
class RingComm:
    """
    P2P communicator for double ring zigzag flash attn.
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops: List[Any] = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
            # print(f'rank:{self.rank},send_rank:{self.send_rank},recv_rank:{self.recv_rank}')

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
