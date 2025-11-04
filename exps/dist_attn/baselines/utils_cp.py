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

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformer_engine as te  # noqa
import transformer_engine_torch as tex
from einops import rearrange

# fa3
try:
    from flash_attn_interface import _flash_attn_backward, _flash_attn_forward
except:
    pass

# from flashattn_hopper.flash_attn_interface import (
#     _flash_attn_backward,
#     _flash_attn_forward,
# )
from torch.distributed._functional_collectives import all_to_all_single_autograd

from magi_attention.common.ranges import AttnRanges

jit_fuser = torch.jit.script


class AttnBackend(Enum):
    TE = ("te",)
    FA3 = "fa3"


@dataclass
class RunTimeMeta:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    unpad_indices_q: torch.Tensor
    unpad_indices_kv: torch.Tensor
    max_seqlen_q: int
    max_seqlen_kv: int


# -----    padding    ---- #


# generate varlen gather or scatter valid token indices
def generate_unpad_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    total_seqlen: int,  # with pad
    device,
):
    batch_size = cu_seqlens.shape[0] - 1
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    offsets = cu_seqlens_padded[:-1]
    total_indices = torch.arange(total_seqlen, device=device)
    batch_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device), seqlens_padded
    )  # cuda synchronize
    relative_pos = total_indices - offsets[batch_ids]
    mask = relative_pos < seqlens[batch_ids]
    valid_indices = total_indices[mask]  # cuda synchronize

    return valid_indices


# [b,s,h,d] or [s,b,h,d] -> varlen [t,h,d]
def flatten_data_to_varlen(input: torch.Tensor, cu_seqlens: torch.Tensor, qkv_format):
    device = input.device
    other_shape = input.shape[2:]
    restore_shape = input.shape
    if qkv_format == "thd":
        return input, restore_shape
    if qkv_format == "bshd":
        batch_size, seqlen = input.shape[0], input.shape[1]
        flatten_input = input.view(-1, *other_shape)
    elif qkv_format == "sbhd":
        batch_size, seqlen = input.shape[1], input.shape[0]
        flatten_input = input.transpose(0, 1).contiguous().view(-1, *other_shape)

    seqlens_padded = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)
    cu_seqlens_padded = F.pad(
        torch.cumsum(seqlens_padded, dim=0), (1, 0), mode="constant", value=0
    )
    total_seqlen = flatten_input.shape[0]

    indices = generate_unpad_indices(
        cu_seqlens, cu_seqlens_padded, total_seqlen, device
    )
    output = torch.gather(
        flatten_input, dim=0, index=indices[:, None, None].expand(-1, *other_shape)
    )
    return output, restore_shape


# [t,h,d] -> [b,s,h,d] or [s,b,h,d]
def unflatten_data_from_varlen(
    input: torch.Tensor, cu_seqlens: torch.Tensor, restore_shape, qkv_format
):
    device = input.device
    other_shape = restore_shape[2:]
    if qkv_format == "thd":
        return input
    if qkv_format == "bshd":
        batch_size, seqlen = restore_shape[0], restore_shape[1]
    elif qkv_format == "sbhd":
        batch_size, seqlen = restore_shape[1], restore_shape[0]

    seqlens_padded = torch.full((batch_size,), seqlen, dtype=torch.int32, device=device)
    cu_seqlens_padded = F.pad(
        torch.cumsum(seqlens_padded, dim=0), (1, 0), mode="constant", value=0
    )
    total_seqlen = batch_size * seqlen

    indices = generate_unpad_indices(
        cu_seqlens, cu_seqlens_padded, total_seqlen, device
    )
    output = torch.zeros(
        (batch_size * seqlen, *other_shape), device=input.device, dtype=input.dtype
    )
    output.scatter_(0, indices[:, None, None].expand(-1, *other_shape), input)
    output = output.view(batch_size, seqlen, *other_shape)
    if qkv_format == "sbhd":
        output = output.transpose(0, 1).contiguous()

    return output


# thd format, unpad input for fa func
# q, k, v, out, dout, seq_dim = 0
def fa_varlen_thd_unpad(input: torch.Tensor, indices: torch.Tensor):
    unpad_input = torch.gather(
        input,
        dim=0,
        index=indices[:, None, None].expand(-1, input.shape[1], input.shape[2]),
    )
    return unpad_input


# softmax_lse [h,t]
# seq_dim = -1
def fa_varlen_lse_unpad(input: torch.Tensor, indices: torch.Tensor):
    unpad_input = torch.gather(
        input, dim=1, index=indices[None, :].expand(input.shape[0], -1)
    )
    return unpad_input


# thd format, pad input for fa func
# q, k, v, out, lse, seq_dim = 0
def fa_varlen_thd_pad(input: torch.Tensor, indices: torch.Tensor, shape):
    pad_input = torch.zeros(*shape, device=input.device, dtype=input.dtype)
    pad_input.scatter_(0, indices[:, None, None].expand(-1, *input.shape[1:]), input)
    return pad_input


# softmax_lse
# seq_dim = -1
def fa_varlen_lse_pad(input: torch.Tensor, indices: torch.Tensor, shape):
    # pad_input = torch.zeros(*shape, device=input.device, dtype=input.dtype)
    pad_input = torch.full(shape, float("-inf"), device=input.device, dtype=input.dtype)
    pad_input.scatter_(-1, indices[None, :].expand(input.shape[0], -1), input)
    return pad_input


# -----    wrap runtime meta per step for fa    ---- #


# pre-compute cu_seqlens_per_step, max_seqlen_per_step, unpad_indices_per_step
def generate_runtime_meta_per_step(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    cu_seqlens_padded_q: torch.Tensor,
    cu_seqlens_padded_kv: torch.Tensor,
    host_cu_seqlens_q: List[int],
    host_cu_seqlens_kv: List[int],
    total_seqlen_q: int,
    total_seqlen_kv: int,
    device,
):
    # compute q unpad indices meta
    valid_cu_seqlens_q, valid_cu_seqlens_kv = generate_valid_cu_seqlens(
        cu_seqlens_q, cu_seqlens_kv
    )
    (
        valid_max_len_q,
        valid_total_len_q,
        valid_max_len_kv,
        valid_total_len_kv,
    ) = get_valid_max_seqlen(host_cu_seqlens_q, host_cu_seqlens_kv)
    unpad_indices_q = generate_unpad_indices(
        valid_cu_seqlens_q, cu_seqlens_padded_q, total_seqlen_q, device
    )
    unpad_indices_kv = generate_unpad_indices(
        valid_cu_seqlens_kv, cu_seqlens_padded_kv, total_seqlen_kv, device
    )

    runtime_meta = RunTimeMeta(
        cu_seqlens_q=valid_cu_seqlens_q,
        cu_seqlens_kv=valid_cu_seqlens_kv,
        unpad_indices_q=unpad_indices_q,
        unpad_indices_kv=unpad_indices_kv,
        max_seqlen_q=valid_max_len_q,
        max_seqlen_kv=valid_max_len_kv,
    )
    return runtime_meta


# to skip per_seqlen_q == 0 or per_seqlen_kv == 0
def generate_valid_cu_seqlens(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
):
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    mask_q = seqlens_q > 0
    mask_kv = seqlens_kv > 0
    valid_mask = mask_q & mask_kv
    valid_seqlens_q = seqlens_q * valid_mask
    valid_seqlens_kv = seqlens_kv * valid_mask
    valid_cu_seqlens_q = F.pad(
        torch.cumsum(valid_seqlens_q, dim=0, dtype=torch.int32), (1, 0)
    )
    valid_cu_seqlens_kv = F.pad(
        torch.cumsum(valid_seqlens_kv, dim=0, dtype=torch.int32), (1, 0)
    )
    return valid_cu_seqlens_q, valid_cu_seqlens_kv


# get max_seqlen for fa
def get_valid_max_seqlen(list1: List[int], list2: List[int]):
    max_diff1 = 0
    max_diff2 = 0
    total_len1 = 0
    total_len2 = 0
    for a1, a2, b1, b2 in zip(list1, list1[1:], list2, list2[1:]):
        diff1 = a2 - a1
        diff2 = b2 - b1
        if diff1 > 0 and diff2 > 0:
            max_diff1 = max(max_diff1, diff1)
            max_diff2 = max(max_diff2, diff2)
            total_len1 += diff1
            total_len2 += diff2
    return max_diff1, total_len1, max_diff2, total_len2


# -----    attention    ---- #


def _pre_process(
    input: torch.Tensor,
    ranges: AttnRanges,
    valid_total_seqlen: int,
    qkv_format: str,
    device,
):
    host_cu_seqlens = ranges.to_cu_seqlens(valid_total_seqlen)
    cu_seqlens = torch.tensor(host_cu_seqlens, device=device, dtype=torch.int32)
    # [b,s,h,d] or [s,b,h,d] -> [t,h,d]
    output, origin_restore_shape = flatten_data_to_varlen(input, cu_seqlens, qkv_format)

    return output, origin_restore_shape, cu_seqlens, host_cu_seqlens


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


@jit_fuser
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


# te ctx saving tensor func
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


# te ctx restore tensor func
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


# fa3 varlen forward
def _fa3_varlen_forward(
    q,
    k,
    v,
    softmax_scale,
    is_causal,
    rumtime_meta_per_step,
    fa_forward_kwargs,
):
    restore_shape = q.shape

    fa_forward_args_thd = [
        None,
        None,
        None,
        None,  # k_new, v_new, qv, out
        rumtime_meta_per_step.cu_seqlens_q,
        rumtime_meta_per_step.cu_seqlens_kv,
        None,
        None,
        None,  # cu_seqlens_k_new, seqused_q, seqused_k
        rumtime_meta_per_step.max_seqlen_q,
        rumtime_meta_per_step.max_seqlen_kv,
        None,
        None,
        None,  # page_table, kv_batch_idx, leftpad_k,
        None,
        None,
        None,  # rotary_cos/sin, seqlens_rotary
        None,
        None,
        None,  # q_descale, k_descale, v_descale
        softmax_scale,
        is_causal,
    ]
    unpad_indices_q = rumtime_meta_per_step.unpad_indices_q
    unpad_indices_kv = rumtime_meta_per_step.unpad_indices_kv
    # unpad
    q_part = fa_varlen_thd_unpad(q, unpad_indices_q)
    k_part, v_part = [fa_varlen_thd_unpad(x, unpad_indices_kv) for x in [k, v]]

    if is_causal:
        fa_forward_kwargs["window_size"] = (-1, 0)
    else:
        fa_forward_kwargs["window_size"] = (-1, -1)

    out, lse, *rest = _flash_attn_forward(
        q_part,
        k_part,
        v_part,
        *fa_forward_args_thd,
        **fa_forward_kwargs,
    )

    # pad
    out_per_step = fa_varlen_thd_pad(out, unpad_indices_q, restore_shape)  # thd
    softmax_lse_per_step = fa_varlen_lse_pad(
        lse, unpad_indices_q, (restore_shape[1], restore_shape[0])
    )  # h,t

    return out_per_step, softmax_lse_per_step  # lse: h,t


def _fa3_varlen_backward(
    q,
    k,
    v,
    out,
    dout,
    softmax_lse,
    softmax_scale,
    is_causal,
    window_size,
    deterministic,
    rumtime_meta_per_step,
):
    restore_shape_q = q.shape
    restore_shape_kv = k.shape

    unpad_indices_q = rumtime_meta_per_step.unpad_indices_q
    unpad_indices_kv = rumtime_meta_per_step.unpad_indices_kv
    # unpad
    q_part, out_part, dout_part = [
        fa_varlen_thd_unpad(x, unpad_indices_q) for x in [q, out, dout]
    ]
    lse_part = fa_varlen_lse_unpad(softmax_lse, unpad_indices_q)  # h,t
    k_part, v_part = [fa_varlen_thd_unpad(x, unpad_indices_kv) for x in [k, v]]

    dq_ = torch.empty_like(q_part)
    dk_ = torch.empty_like(k_part)
    dv_ = torch.empty_like(v_part)

    fa_backward_args_thd = [
        rumtime_meta_per_step.cu_seqlens_q,
        rumtime_meta_per_step.cu_seqlens_kv,
        None,
        None,  # seqused_q, seqused_k
        rumtime_meta_per_step.max_seqlen_q,
        rumtime_meta_per_step.max_seqlen_kv,
        dq_,
        dk_,
        dv_,
        softmax_scale,
        is_causal,
        window_size,
        0.0,
        deterministic,
    ]

    _flash_attn_backward(
        dout_part, q_part, k_part, v_part, out_part, lse_part, *fa_backward_args_thd
    )

    # pad
    dq = fa_varlen_thd_pad(dq_, unpad_indices_q, restore_shape_q)
    dk, dv = [
        fa_varlen_thd_pad(x, unpad_indices_kv, restore_shape_kv) for x in [dk_, dv_]
    ]

    return dq, dk, dv


# NOTE: for pad token, lse is set to -INF, however, this func meets issues when dealing with -INF in log1p


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


# def flash_attn_fwd_softmax_lse_correction(
#     softmax_lse: torch.Tensor,
#     softmax_lse_per_step: torch.Tensor,
# ):
#     max_scale = torch.max(softmax_lse, softmax_lse_per_step)
#     min_scale = torch.min(softmax_lse, softmax_lse_per_step)
#     both_inf = torch.isneginf(max_scale) & torch.isneginf(min_scale)
#     new_scale = max_scale + torch.log1p(torch.exp(min_scale - max_scale))
#     new_scale = torch.where(both_inf, torch.full_like(new_scale, float('-inf')), new_scale)
#     softmax_lse.copy_(new_scale)


def bwd_dq_update(
    dq,
    dq_,
    cu_seqlens_q_padded,
    first_op,
    second_op,
):
    if first_op == "copy" and second_op == "copy":
        dq.copy_(dq_)
    elif first_op == "add" and second_op == "add":
        dq.add_(dq_)
    else:
        tex.thd_grad_correction(dq, dq_, cu_seqlens_q_padded, first_op, second_op)
    return dq


def bwd_dkv_update(dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op):
    if first_op == "copy" and second_op == "copy":
        dkv.copy_(dkv_)
    elif first_op == "add" and second_op == "add":
        dkv.add_(dkv_)
    else:
        tex.thd_grad_correction(dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op)
    return dkv


# -----    comm    ---- #


# p2p comm
# def attn_p2p_communicate(
#     rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
# ):
#     """Point-to-point communications of KV and dKV in Attention with context parallelism"""
#     send_recv_ops = []

#     send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
#     recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
#     send_recv_ops.append(send_op)
#     send_recv_ops.append(recv_op)
#     send_recv_reqs = send_recv_ops

#     return send_recv_reqs


# if batch_p2p_comm:
#     if rank % 2 == 0:
#         send_op = torch.distributed.P2POp(
#             torch.distributed.isend, send_tensor, send_dst, cp_group
#         )
#         recv_op = torch.distributed.P2POp(
#             torch.distributed.irecv, recv_tensor, recv_src, cp_group
#         )
#         send_recv_ops.append(send_op)
#         send_recv_ops.append(recv_op)
#     else:
#         recv_op = torch.distributed.P2POp(
#             torch.distributed.irecv, recv_tensor, recv_src, cp_group
#         )
#         send_op = torch.distributed.P2POp(
#             torch.distributed.isend, send_tensor, send_dst, cp_group
#         )
#         send_recv_ops.append(recv_op)
#         send_recv_ops.append(send_op)
#     send_recv_reqs = torch.distributed.batch_isend_irecv(send_recv_ops)
# else:
#     if rank % 2 == 0:
#         send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
#         recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
#         send_recv_ops.append(send_op)
#         send_recv_ops.append(recv_op)
#     else:
#         recv_op = torch.distributed.irecv(recv_tensor, recv_src, cp_group)
#         send_op = torch.distributed.isend(send_tensor, send_dst, cp_group)
#         send_recv_ops.append(recv_op)
#         send_recv_ops.append(send_op)
#     send_recv_reqs = send_recv_ops

# return send_recv_reqs


def attn_p2p_communicate(
    rank, send_tensor, send_dst, recv_tensor, recv_src, cp_group, batch_p2p_comm
):
    """Point-to-point communications of KV and dKV in Attention with context parallelism"""

    send_recv_ops = []
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
def _varlen_all2all_before_attn(input_: torch.Tensor, cp_group):
    cp_size = dist.get_world_size(cp_group)
    x = rearrange(
        input_,
        "t h d -> h t d",
    ).contiguous()
    x = all_to_all_single_autograd(
        x, output_split_sizes=None, input_split_sizes=None, group=cp_group
    )
    x = rearrange(x, "(cp_size h) t d -> (cp_size t) h d", cp_size=cp_size).contiguous()
    return x


def _varlen_all2all_after_attn(input_: torch.Tensor, cp_group):
    cp_size = dist.get_world_size(cp_group)
    x = all_to_all_single_autograd(
        input_, output_split_sizes=None, input_split_sizes=None, group=cp_group
    )
    x = rearrange(x, "(cp_size t) h d -> t (cp_size h) d", cp_size=cp_size).contiguous()
    return x


############################################################
# attention
############################################################


def divide_lst(lst, k):
    assert k > 0
    return [x // k for x in lst]
