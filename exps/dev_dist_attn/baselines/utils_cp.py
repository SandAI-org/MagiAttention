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

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import transformer_engine as te  # noqa
import transformer_engine_torch as tex
from einops import rearrange, repeat

jit_fuser = torch.jit.script

############################################################
# attention
############################################################


def get_p2p_send_recv_rank(rank, world_size, process_group, reverse=False):
    if not reverse:
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size
    else:
        send_rank = (rank - 1) % world_size
        recv_rank = (rank + 1) % world_size

    next_send_rank = dist.get_global_rank(process_group, send_rank)
    next_recv_rank = dist.get_global_rank(process_group, recv_rank)
    return next_send_rank, next_recv_rank


def prepare_q_fwd(
    q,
    chunk_idx,
    qkv_format,
    cu_seqlens_q,
    cu_seqlens_q_padded,
    pad_between_seqs_q,
    cp_size,
    cp_rank,
    host_meta=None,
):
    is_half = False
    if chunk_idx >= 0:
        is_half = True
    if is_half:  # q0 | q1
        first_index, second_index, chunk_num = (
            (chunk_idx == 0),
            (chunk_idx == 1),
            2 * cp_size,
        )
    else:  # q
        first_index, second_index, chunk_num = True, True, cp_size

    if pad_between_seqs_q or qkv_format == "thd":
        cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
            cu_seqlens_q,
            cu_seqlens_q_padded,
            cp_size,
            cp_rank,
            first_index,
            second_index,
        )
    else:
        cu_seqlens_q_per_step = cu_seqlens_q // chunk_num

    host_meta_per_step = None
    if host_meta is not None:
        host_meta_per_step = [None, host_meta[1]]
        if pad_between_seqs_q or qkv_format == "thd":
            host_meta_per_step[0] = get_cu_seqlens_on_cp_rank_host(
                host_meta[0], host_meta[1], cp_size, cp_rank, first_index, second_index
            )
        else:
            host_meta_per_step[0] = divide_lst(host_meta[0], chunk_num)
        if is_half:
            host_meta_per_step[1] = divide_lst(host_meta_per_step[1], 2)

    q_ret = None
    if qkv_format == "bshd":
        if is_half:
            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
            q_ret = q[:, chunk_idx, ...]
        else:
            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
            q_ret = q.view(q.shape[0], -1, *q.shape[-2:])
    elif qkv_format == "sbhd":
        if is_half:
            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
            q_ret = q[chunk_idx]
        else:
            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
            q_ret = q.view(-1, *q.shape[-3:])
    elif qkv_format == "thd":
        if is_half:
            # [t, np, hn] -> [t/2, np, hn]
            q_ret = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, chunk_idx)
        else:
            q_ret = q

    return cu_seqlens_q_per_step, q_ret, host_meta_per_step


def prepare_kv_fwd(
    k,
    kv,
    is_half,
    qkv_format,
    cu_seqlens_kv,
    cu_seqlens_kv_padded,
    pad_between_seqs_kv,
    cp_size,
    cp_rank,
    host_meta=None,
):
    if is_half:  # k0, v0
        first_index, second_index, chunk_num = True, False, 2 * cp_size
    else:
        first_index, second_index, chunk_num = True, True, cp_size

    if pad_between_seqs_kv or qkv_format == "thd":
        cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
            cu_seqlens_kv,
            cu_seqlens_kv_padded,
            cp_size,
            cp_rank,
            first_index,
            second_index,
        )
    else:
        cu_seqlens_kv_per_step = cu_seqlens_kv // chunk_num

    host_meta_per_step = None
    if host_meta is not None:
        host_meta_per_step = [None, host_meta[1]]
        if pad_between_seqs_kv or qkv_format == "thd":
            host_meta_per_step[0] = get_cu_seqlens_on_cp_rank_host(
                host_meta[0], host_meta[1], cp_size, cp_rank, first_index, second_index
            )
        else:
            host_meta_per_step[0] = divide_lst(host_meta[0], chunk_num)
        if is_half:
            host_meta_per_step[1] = divide_lst(host_meta_per_step[1], 2)

    kv_ret = None
    if qkv_format == "bshd":
        if is_half:
            # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
            kv_ret = kv[:, 0, ...]
        else:
            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
            kv_ret = kv.view(k.shape[0], -1, 2, *k.shape[-2:])
    elif qkv_format == "sbhd":
        if is_half:
            # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
            kv_ret = kv[0]
        else:
            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
            kv_ret = kv.view(-1, *kv.shape[-4:])
            # kv_ret = kv.view(-1, k.shape[2], 2, *k.shape[-2:])
    elif qkv_format == "thd":
        if is_half:
            # [2, t, np, hn] -> [2, t/2, np, hn]
            kv_ret = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
        else:
            kv_ret = kv

    return cu_seqlens_kv_per_step, kv_ret, host_meta_per_step


def prepare_q_bwd(
    inputs,
    chunk_idx,
    cu_seqlens_q_padded,
    qkv_format,
):
    is_half = False
    if chunk_idx >= 0:
        is_half = True
    if qkv_format == "bshd":
        if is_half:
            # [b, 2, sq//2, np, hn] -> [b, sq//2, np, hn]
            outputs = [x[:, chunk_idx] for x in inputs]
            # q_, out_, dout_ = q[:, 1], out[:, 1], dout[:, 1]
        else:
            # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
            outputs = [x.view(x.shape[0], -1, *x.shape[-2:]) for x in inputs]
            # q_, out_, dout_ = [x.view(x.shape[0], -1, *x.shape[-2:]) for x in [q, out, dout]]
    elif qkv_format == "sbhd":
        if is_half:
            # [2, sq//2, b, np, hn] -> [sq//2, b, np, hn]
            outputs = [x[chunk_idx] for x in inputs]
            # q_, out_, dout_ = q[1], out[1], dout[1]
        else:
            # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
            outputs = [x.view(-1, *x.shape[-3:]) for x in inputs]
            # q_, out_, dout_ = [x.view(-1, *x.shape[-3:]) for x in [q, out, dout]]
    elif qkv_format == "thd":
        if is_half:
            # [t, np, hn] -> [t/2, np, hn]
            outputs = [
                tex.thd_read_half_tensor(x, cu_seqlens_q_padded, chunk_idx)
                for x in inputs
            ]
            # q_, out_, dout_ = [
            #     tex.thd_read_half_tensor(x, cu_seqlens_q_padded, 1)
            #     for x in [q, out, dout]
            # ]
        else:
            outputs = inputs
            # q_, out_, dout_ = q, out, dout
    return outputs
    # return q_, out_, dout_


def prepare_kv_bwd(kv, is_half, cu_seqlens_kv_padded, qkv_format):
    if qkv_format == "bshd":
        if is_half:
            # [b, 2, sk//2, 2, np, hn] -> [b, sk//2, 2, np, hn]
            kv_ = kv[:, 0]
        else:
            # [b, 2, sk//2, 2, np, hn] -> [b, sk, 2, np, hn]
            kv_ = kv.view(kv.shape[0], -1, *kv.shape[-3:])
    elif qkv_format == "sbhd":
        if is_half:
            # [2, sk//2, b, 2, np, hn] -> [sk//2, b, 2, np, hn]
            kv_ = kv[0]
        else:
            # [2, sk//2, b, 2, np, hn] -> [sk, b, 2, np, hn]
            kv_ = kv.view(-1, *kv.shape[-4:])
    elif qkv_format == "thd":
        if is_half:
            kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
        else:
            kv_ = kv

    return kv_


@jit_fuser
def flash_attn_fwd_softmax_lse_correction(
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
):
    """Merge softmax stats of each step in Attention with context parallelism"""
    max_scale = torch.max(softmax_lse, softmax_lse_per_step)
    min_scale = torch.min(softmax_lse, softmax_lse_per_step)
    new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
    softmax_lse.copy_(new_scale)


# @jit_fuser
def get_cu_seqlens_on_cp_rank(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded_on_cp_rank: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    first_half: bool,
    second_half: bool,
):
    """Compute cu_seqlens of a context parallelism rank"""
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (
        cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]
    ) // 2
    zeros = torch.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = torch.zeros_like(cu_seqlens)
    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = seqlens_1.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_1)
    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = seqlens_2.clamp(zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:].add_(seqlens_2)

    cu_seqlens_on_cp_rank.cumsum_(dim=0)
    return cu_seqlens_on_cp_rank


def get_cu_seqlens_on_cp_rank_host(
    cu_seqlens,  # list[int]
    cu_seqlens_padded_on_cp_rank,  # list[int]
    cp_size: int,
    cp_rank: int,
    first_half: bool,
    second_half: bool,
):
    cu_seqlens = np.array(cu_seqlens)
    cu_seqlens_padded_on_cp_rank = np.array(cu_seqlens_padded_on_cp_rank)

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = (
        cu_seqlens_padded_on_cp_rank[1:] - cu_seqlens_padded_on_cp_rank[:-1]
    ) // 2

    zeros = np.zeros_like(seqlens)
    cu_seqlens_on_cp_rank = np.zeros_like(cu_seqlens)

    if first_half:
        seqlens_1 = seqlens - cp_rank * seqlens_padded
        seqlens_1 = np.clip(seqlens_1, zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:] += seqlens_1

    if second_half:
        seqlens_2 = seqlens - (2 * cp_size - cp_rank - 1) * seqlens_padded
        seqlens_2 = np.clip(seqlens_2, zeros, seqlens_padded)
        cu_seqlens_on_cp_rank[1:] += seqlens_2

    cu_seqlens_on_cp_rank = np.cumsum(cu_seqlens_on_cp_rank)

    return cu_seqlens_on_cp_rank.tolist()


# compute cu_seqlens for kv all-gather causal
@jit_fuser
def generate_cu_seqlens_kv_ag_causal(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    rank: int,
    cp_size: int,
):
    cu_seqlens_padded = cu_seqlens_padded // (2 * cp_size)
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    seqlens_unpad = cu_seqlens[1:] - cu_seqlens[:-1]
    causal_seqlens = seqlens_padded * (rank + 1)
    seqlens_unpad = torch.min(seqlens_unpad, causal_seqlens)
    cu_seqlens_causal = torch.zeros_like(cu_seqlens)
    cu_seqlens_causal[1:].add_(seqlens_unpad)
    cu_seqlens_causal.cumsum_(dim=0)

    return cu_seqlens_causal


def generate_cu_seqlens_kv_ag_causal_host(
    cu_seqlens: List[int],
    cu_seqlens_padded: List[int],
    rank: int,
    cp_size: int,
):
    cu_seqlens = np.array(cu_seqlens)
    cu_seqlens_padded = np.array(cu_seqlens_padded)

    cu_seqlens_padded = cu_seqlens_padded // (2 * cp_size)
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    seqlens_unpad = cu_seqlens[1:] - cu_seqlens[:-1]
    causal_seqlens = seqlens_padded * (rank + 1)
    seqlens_unpad = np.minimum(seqlens_unpad, causal_seqlens)
    cu_seqlens_causal = np.zeros_like(cu_seqlens)
    cu_seqlens_causal[1:] = seqlens_unpad
    cu_seqlens_causal = np.cumsum(cu_seqlens_causal)

    return cu_seqlens_causal.tolist()


@jit_fuser
def flash_attn_fwd_out_correction(
    out: torch.Tensor,
    out_per_step: torch.Tensor,
    softmax_lse: torch.Tensor,
    softmax_lse_per_step: torch.Tensor,
    movedim_src: int,
    movedim_dst: int,
):
    """Merge partial outputs of each step in Attention with context parallelism"""
    softmax_lse_corrected_exp = torch.exp(softmax_lse_per_step - softmax_lse).movedim(
        movedim_src, movedim_dst
    )
    softmax_lse_corrected_exp = softmax_lse_corrected_exp.unsqueeze(-1)
    out_corrected = out_per_step * softmax_lse_corrected_exp
    out.add_(out_corrected)


def fwd_out_update(
    out,
    out_per_step,
    softmax_lse,
    softmax_lse_per_step,
    cu_seqlens_q_padded,
    is_half,
    qkv_format,
    softmax_lse_in_packed_format,
):
    if qkv_format in ["bshd", "sbhd"]:
        seq_dim = qkv_format.index("s")
        movedim_src = 0 if softmax_lse_in_packed_format else 2
        movedim_dst = 2 if softmax_lse_in_packed_format else seq_dim
        if is_half:
            out_ = out.select(seq_dim, 1)
            softmax_lse_ = softmax_lse.view(
                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
            )[..., 1, :]
        else:
            out_ = out.view(out_per_step.shape)
            softmax_lse_ = softmax_lse
        flash_attn_fwd_out_correction(
            out_,
            out_per_step,
            softmax_lse_,
            softmax_lse_per_step,
            movedim_src,
            movedim_dst,
        )
    elif qkv_format == "thd":
        tex.thd_out_correction(
            out,
            out_per_step,
            softmax_lse,
            softmax_lse_per_step,
            cu_seqlens_q_padded,
            is_half,
            softmax_lse_in_packed_format,
        )
    return out


def bwd_dq_update(
    dq,
    dq_,
    cu_seqlens_q_padded,
    qkv_format,
    first_op,
    second_op,
):
    if first_op == "copy" and second_op == "copy":
        dq.copy_(dq_)
    elif first_op == "add" and second_op == "add":
        dq.add_(dq_)
    else:
        if qkv_format == "thd":
            tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, first_op, second_op)
        else:
            if first_op == "copy" and second_op == "add":
                if qkv_format == "bshd":
                    dq[:, 0, ...].copy_(dq_[:, 0, ...])
                    dq[:, 1, ...].add_(dq_[:, 1, ...])
                elif qkv_format == "sbhd":
                    dq[0].copy_(dq_[0])
                    dq[1].add_(dq_[1])
            elif first_op == "none":
                if second_op == "copy":
                    if qkv_format == "bshd":
                        dq[:, 1, ...].copy_(dq_)
                    elif qkv_format == "sbhd":
                        dq[1].copy_(dq_)
                elif second_op == "add":
                    if qkv_format == "bshd":
                        dq[:, 1, ...].add_(dq_)
                    elif qkv_format == "sbhd":
                        dq[1].add_(dq_)
    return dq


def bwd_dkv_update(dkv, dkv_, cu_seqlens_kv_padded, qkv_format, first_op, second_op):
    if first_op == "copy" and second_op == "copy":
        dkv.copy_(dkv_)
    elif first_op == "add" and second_op == "add":
        dkv.add_(dkv_)
    else:
        if qkv_format == "thd":
            tex.thd_grad_correction(
                dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op
            )
        else:
            if first_op == "add" and second_op == "copy":
                if qkv_format == "bshd":
                    dkv[:, :, 0, ...].add_(dkv_[:, :, 0, ...])
                    dkv[:, :, 1, ...].copy_(dkv_[:, :, 1, ...])
                elif qkv_format == "sbhd":
                    dkv[:, 0, ...].add_(dkv_[:, 0, ...])
                    dkv[:, 1, ...].copy_(dkv_[:, 1, ...])
            elif second_op == "none":
                if first_op == "copy":
                    if qkv_format == "bshd":
                        dkv[:, :, 0, ...].copy_(dkv_)
                    elif qkv_format == "sbhd":
                        dkv[:, 0, ...].copy_(dkv_)
                elif first_op == "add":
                    if qkv_format == "bshd":
                        dkv[:, :, 0, ...].add_(dkv_)
                    elif qkv_format == "sbhd":
                        dkv[:, 0, ...].add_(dkv_)
    return dkv


def prepare_for_saving(
    *tensors,
) -> Tuple[list[Optional[Union[torch.Tensor, torch.nn.Parameter]]], Optional[Any]]:
    """Prepare tensors for saving. Needed because save_for_backward accepts only
    torch.Tensor/torch.nn.Parameter types, while we want to be able to save
    the internal TensorBase types too."""
    # pylint: disable=unidiomatic-typecheck  # Using type instead of isinstance to check exact type
    tensor_list, tensor_objects_list = [], []  # type: ignore
    for tensor in tensors:
        if tensor is None:
            tensor_list.append(None)
            tensor_objects_list.append(None)
        elif type(tensor) in (torch.Tensor, torch.nn.Parameter):
            tensor_list.append(tensor)
            tensor_objects_list.append(None)
        else:
            t, t_obj = tensor.prepare_for_saving()
            tensor_list.extend(t)
            tensor_objects_list.append(t_obj)
    return tensor_list, tensor_objects_list


def restore_from_saved(
    tensors: list[Optional[Any]],
    saved_tensors: list[Optional[Union[torch.Tensor, torch.nn.Parameter]]],
) -> list[Optional[Any]]:
    """Recombine the tensor data and metadata during backward pass."""
    tensor_objects = []
    for tensor in tensors:
        if tensor is None:
            tensor_objects.append(saved_tensors[0])
            saved_tensors = saved_tensors[1:]
        else:
            saved_tensors = tensor.restore_from_saved(saved_tensors)
            tensor_objects.append(tensor)
    return tensor_objects


############################################################
# comm
############################################################


# p2p comm
def attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""
    send_recv_ops = []

    if batch_p2p_comm:
        if rank % 2 == 0:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, recv_tensor, recv_src, cp_group
            )
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, send_tensor, send_dst, cp_group
            )
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
    else:
        if rank % 2 == 0:
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_recv_ops.append(send_op)
            send_recv_ops.append(recv_op)
        else:
            recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
            send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
            send_recv_ops.append(recv_op)
            send_recv_ops.append(send_op)
        send_recv_reqs = send_recv_ops

    return send_recv_reqs


# all2all comm
# TODO: to Optimize
class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group, input, scatter_idx, gather_idx, batch_dim_idx):
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.batch_dim_idx = batch_dim_idx
        res = single_all_to_all(
            input, scatter_idx, gather_idx, batch_dim_idx, group, False
        )
        return res

    @staticmethod
    def backward(ctx, *dout):
        res = single_all_to_all(
            *dout, ctx.gather_idx, ctx.scatter_idx, ctx.batch_dim_idx, ctx.group, False
        )
        return (
            None,
            res,
            None,
            None,
            None,
        )


def single_all_to_all(
    input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False
):
    seq_world_size = dist.get_world_size(group)
    if batch_dim_idx == 0:
        # b, s, n, h
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [
                    bs,
                    seq_world_size,
                    global_seq_len // seq_world_size,
                    num_local_head,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [
                    bs,
                    local_seq_len,
                    seq_world_size,
                    num_total_head // seq_world_size,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        # s, b, n, h
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            input_t = input.reshape(
                [
                    seq_world_size,
                    global_seq_len // seq_world_size,
                    bs,
                    num_local_head,
                    head_dim,
                ]
            ).contiguous()
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert (
                num_total_head % seq_world_size == 0
            ), f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape(
                [
                    local_seq_len,
                    bs,
                    seq_world_size,
                    num_total_head // seq_world_size,
                    head_dim,
                ]
            ).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    if scatter_idx < 2:
        post_all2all_fun = post_all2all(
            scatter_idx,
            batch_dim_idx,
            seq_world_size,
            bs,
            global_seq_len,
            num_local_head,
            head_dim,
        )
    else:
        post_all2all_fun = post_all2all(
            scatter_idx,
            batch_dim_idx,
            seq_world_size,
            bs,
            local_seq_len,
            num_total_head,
            head_dim,
        )

    output = torch.empty_like(input_t)
    # work =
    dist.all_to_all_single(output, input_t, group=group, async_op=async_op)
    dist.barrier(group=group)

    res = post_all2all_fun(output)
    return res


def post_all2all(
    scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim
):
    def post_func(input):
        if batch_dim_idx == 0:
            # b, s, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(
                    bs, seq_len // seq_world_size, seq_world_size * num_head, head_dim
                ).contiguous()
            else:
                output = input.permute(1, 0, 2, 3, 4).contiguous()
                output = output.reshape(
                    bs, seq_world_size * seq_len, num_head // seq_world_size, head_dim
                ).contiguous()
        else:
            # s, b, n, h
            if scatter_idx < 2:
                output = input.permute(1, 2, 0, 3, 4).contiguous()
                output = output.reshape(
                    seq_len // seq_world_size, bs, seq_world_size * num_head, head_dim
                ).contiguous()
            else:
                output = input.reshape(
                    seq_len * seq_world_size, bs, num_head // seq_world_size, head_dim
                ).contiguous()
        return output

    return post_func


############################################################
# padding
############################################################


def get_cu_seqlens_indices(
    host_cu_seqlens_per_step: List[int],
    host_cu_seqlens_padded: List[int],
    device,
    ref_cu_seqlens_per_step=None,
):
    batch_size = len(host_cu_seqlens_per_step) - 1
    host_cu_seqlens_np = np.array(host_cu_seqlens_per_step)
    host_cu_seqlens_padded_np = np.array(host_cu_seqlens_padded)
    host_cu_seqlens = host_cu_seqlens_np[1:] - host_cu_seqlens_np[:-1]  # [batch]
    indices_varlen = []
    for i in range(batch_size):
        if ref_cu_seqlens_per_step is not None:
            if ref_cu_seqlens_per_step[i + 1] - ref_cu_seqlens_per_step[i] > 0:
                st = host_cu_seqlens_padded_np[i]
                ed = st + host_cu_seqlens[i]
                indices = torch.arange(start=st, end=ed, device=device)
                indices_varlen.append(indices)
    indices = torch.cat(indices_varlen, dim=0)
    return indices


# thd format, unpad input for fa func
# q,k,v,out,dout
# seq_dim = 0
def fa_varlen_thd_unpad(input: torch.Tensor, indices: torch.Tensor):
    unpad_input = torch.gather(
        input,
        dim=0,
        index=indices[:, None, None].expand(-1, input.shape[1], input.shape[2]),
    )
    # other_shape = input.shape[1:]
    # second_dim = other_shape.numel()
    # unpad_input = (
    #         torch.gather(
    #             rearrange(input, "t ... -> t (...)"),
    #             0,
    #             repeat(indices, "t -> t d", d=second_dim),
    #         )
    #         .reshape(-1, *other_shape)
    #         .contiguous()
    #     )
    return unpad_input


# softmax_lse [h,t]
# seq_dim = -1
def fa_varlen_lse_unpad(input: torch.Tensor, indices: torch.Tensor):
    unpad_input = torch.gather(
        input, dim=1, index=indices[None, :].expand(input.shape[0], -1)
    )
    # other_shape = input.shape[:-1]
    # second_dim = other_shape.numel()
    # unpad_input = (
    #         torch.gather(
    #             rearrange(input, "... t -> (...) t"),
    #             -1,
    #             repeat(indices, "t -> d t", d=second_dim),
    #         )
    #         .reshape(*other_shape, -1)
    #         .contiguous()
    #     )
    return unpad_input


# thd format, pad input for fa func
# q,k,v,out,lse
# seq_dim = 0
def fa_varlen_thd_pad(input: torch.Tensor, indices: torch.Tensor, shape):
    other_shape = shape[1:]
    second_dim = other_shape.numel()
    pad_input = torch.zeros(
        (shape[0], second_dim), device=input.device, dtype=input.dtype
    )
    input = rearrange(input, "t ... -> t (...)")
    pad_input.scatter_(0, repeat(indices, "t -> t d", d=second_dim), input)
    pad_input = pad_input.reshape(-1, *other_shape).contiguous()
    return pad_input


# softmax_lse
# seq_dim = -1
def fa_varlen_lse_pad(input: torch.Tensor, indices: torch.Tensor, shape):
    other_shape = input.shape[:-1]
    second_dim = other_shape.numel()

    pad_input = torch.zeros(
        (second_dim, shape[-1]), device=input.device, dtype=input.dtype
    )
    input = rearrange(input, "... t -> (...) t")
    pad_input.scatter_(-1, repeat(indices, "t -> d t", d=second_dim), input)
    pad_input = pad_input.reshape(*other_shape, -1).contiguous()
    return pad_input


def divide_lst(lst, k):
    assert k > 0
    return [x // k for x in lst]


# [b,s,h,d] or [s,b,h,d] -> [t,h,d] indices
def generate_flatten_indices(cu_seqlens_host, cu_seqlens_padded_host, device):
    batch_size = len(cu_seqlens_host) - 1
    indices_lst = []
    for i in range(batch_size):
        st = cu_seqlens_padded_host[i]
        ed = st + (cu_seqlens_host[i + 1] - cu_seqlens_host[i])
        indices_lst.append(torch.arange(start=st, end=ed, device=device))
    indices = torch.cat(indices_lst, dim=0)
    return indices


# [b,s,h,d] or [s,b,h,d] -> varlen [t,h,d]
def flatten_data_to_varlen(input: torch.Tensor, cu_seqlens_host: List[int], qkv_format):
    other_shape = input.shape[2:]
    restore_shape = input.shape
    if qkv_format == "bshd":
        batch_size, seqlen = input.shape[0], input.shape[1]
        flatten_input = input.view(-1, *other_shape)
    elif qkv_format == "shdb":
        batch_size, seqlen = input.shape[1], input.shape[0]
        flatten_input = input.transpose(0, 1).contiguous().view(-1, *other_shape)

    cu_seqlens_padded_host = [0]
    for i in range(batch_size):
        cu_seqlens_padded_host.append(cu_seqlens_padded_host[-1] + seqlen)

    indices = generate_flatten_indices(
        cu_seqlens_host, cu_seqlens_padded_host, input.device
    )
    output = torch.gather(
        flatten_input, dim=0, index=indices[:, None, None].expand(-1, *other_shape)
    )
    return output, restore_shape


# [t,h,d] -> [b,s,h,d] or [s,b,h,d]
def unflatten_data_from_varlen(
    input: torch.Tensor, cu_seqlens_host: List[int], restore_shape, qkv_format
):
    other_shape = restore_shape.shape[2:]
    if qkv_format == "bshd":
        batch_size, seqlen = restore_shape[0], restore_shape[1]
    elif qkv_format == "shdb":
        batch_size, seqlen = restore_shape[1], restore_shape[0]

    cu_seqlens_padded_host = [0]
    for i in range(batch_size):
        cu_seqlens_padded_host.append(cu_seqlens_padded_host[-1] + seqlen)

    indices = generate_flatten_indices(
        cu_seqlens_host, cu_seqlens_padded_host, input.device
    )
    output = torch.zeros(
        (batch_size * seqlen, *other_shape), device=input.device, dtype=input.dtype
    )
    output.scatter_(0, indices[:, None, None].expand(-1, *other_shape), input)
    output = output.view(batch_size, seqlen, *other_shape)
    if qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()

    return output
