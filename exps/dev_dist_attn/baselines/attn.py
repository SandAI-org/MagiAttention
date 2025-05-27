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

from enum import Enum
from typing import List

import torch
import torch.nn.functional as F
import transformer_engine as te  # noqa
import transformer_engine_torch as tex

# fa3
from flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

# te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.utils import get_cudnn_version

from .utils_cp import (
    attn_p2p_communicate,
    bwd_dkv_update,
    bwd_dq_update,
    divide_lst,
    fa_varlen_lse_pad,
    fa_varlen_lse_unpad,
    fa_varlen_thd_pad,
    fa_varlen_thd_unpad,
    flash_attn_fwd_softmax_lse_correction,
    fwd_out_update,
    get_cu_seqlens_indices,
    get_p2p_send_recv_rank,
    prepare_for_saving,
    prepare_kv_bwd,
    prepare_kv_fwd,
    prepare_q_bwd,
    prepare_q_fwd,
    restore_from_saved,
)


class AttnBackend(Enum):
    TE = ("te",)
    FA3 = "fa3"


def _combine_tensors(
    tensors: List[torch.Tensor],
    dim: int,
) -> torch.Tensor:
    """Combine tensors along a particular dimension"""

    num_tensors = len(tensors)
    new_shape = list(tensors[0].shape)
    new_shape.insert(dim, num_tensors)

    new_stride = list(tensors[0].stride())
    new_stride.insert(dim, int(new_stride[dim - 1] / num_tensors))
    combined_tensor = torch.Tensor().to(
        device=tensors[0].device, dtype=tensors[0].dtype
    )
    combined_tensor.set_(
        tensors[0].untyped_storage(), tensors[0].storage_offset(), new_shape, new_stride
    )

    return combined_tensor


def _fused_attn_forward(
    q_inputs,
    kv_inputs,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    is_half_q,
    is_half_kv,
    qkv_format,
    fused_attn_meta_args,
    fused_attn_meta_kwargs,
):
    q_part = q_inputs
    k_part = kv_inputs[..., 0, :, :] if qkv_format in ["bshd", "sbhd"] else kv_inputs[0]
    v_part = kv_inputs[..., 1, :, :] if qkv_format in ["bshd", "sbhd"] else kv_inputs[1]
    fp8_meta_kwargs = {}

    _cu_seqlens_q_padded = (
        cu_seqlens_q_padded // 2 if is_half_q else cu_seqlens_q_padded
    )
    _cu_seqlens_kv_padded = (
        cu_seqlens_kv_padded // 2 if is_half_kv else cu_seqlens_kv_padded
    )

    out_per_step, aux_ctx_tensors = fused_attn_fwd(
        True,  # is_training
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        q_part,
        k_part,
        v_part,
        *fused_attn_meta_args,
        **fused_attn_meta_kwargs,
        cu_seqlens_q_padded=_cu_seqlens_q_padded,
        cu_seqlens_kv_padded=_cu_seqlens_kv_padded,
        **fp8_meta_kwargs,
    )

    softmax_lse_per_step, rng_states, *rest = aux_ctx_tensors
    return out_per_step, softmax_lse_per_step, rng_states


def _fused_attn_backward(
    q_,
    kv_,
    out_,
    dout_,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q_padded,
    cu_seqlens_kv_padded,
    is_half_q,
    is_half_kv,
    qkv_format,
    fused_attn_meta_args,
    fused_attn_meta_kwargs,
):
    q_part = q_
    k_part = kv_[..., 0, :, :] if qkv_format in ["bshd", "sbhd"] else kv_[0]
    v_part = kv_[..., 1, :, :] if qkv_format in ["bshd", "sbhd"] else kv_[1]
    out_part = out_
    dout_part = dout_

    fp8_meta_kwargs = {}

    _cu_seqlens_q_padded = (
        cu_seqlens_q_padded // 2 if is_half_q else cu_seqlens_q_padded
    )
    _cu_seqlens_kv_padded = (
        cu_seqlens_kv_padded // 2 if is_half_kv else cu_seqlens_kv_padded
    )

    dq_, dk_, dv_, _ = fused_attn_bwd(
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_per_step,
        cu_seqlens_kv_per_step,
        q_part,
        k_part,
        v_part,
        out_part,
        dout_part,
        *fused_attn_meta_args,
        cu_seqlens_q_padded=_cu_seqlens_q_padded,
        cu_seqlens_kv_padded=_cu_seqlens_kv_padded,
        **fused_attn_meta_kwargs,
        **fp8_meta_kwargs,
    )

    return dq_, dk_, dv_


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
def get_valid_max_seqlen(list1, list2):
    max_diff1 = 0
    max_diff2 = 0
    for i in range(len(list1) - 1):
        diff1 = list1[i + 1] - list1[i]
        diff2 = list2[i + 1] - list2[i]
        if diff1 > 0 and diff2 > 0:
            max_diff1 = max(max_diff1, diff1)
            max_diff2 = max(max_diff2, diff2)
    return max_diff1, max_diff2


def _fa3_attn_forward(
    q_part,
    k_part,
    v_part,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    # max_seqlen_q,
    # max_seqlen_kv,
    # cu_seqlens_q_padded,
    # cu_seqlens_kv_padded,
    is_causal,
    qkv_format,
    pad_between_seqs,
    fa_forward_kwargs,
    host_meta=(None, None, None, None),
):
    (
        host_cu_seqlens_q,
        host_cu_padded_q,
        host_cu_seqlens_kv,
        host_cu_padded_kv,
    ) = host_meta
    # sbhd -> bshd
    if qkv_format == "sbhd":
        q_part, k_part, v_part = [
            x.transpose(1, 0).contiguous() for x in [q_part, k_part, v_part]
        ]
    batch_size = q_part.shape[0]

    fa_forward_args_thd = []
    flash_attn_fwd = _flash_attn_varlen_forward
    if qkv_format != "thd" and not pad_between_seqs:  # attn_forward
        flash_attn_fwd = _flash_attn_forward
    else:  # attn_varlen_forward
        # q_part, k_part, v_part
        # bshd -> thd
        if qkv_format != "thd":
            # q_part, k_part, v_part = [x.contiguous() for x in [q_part, k_part, v_part]]
            q_part, k_part, v_part = [
                x.reshape(-1, *x.shape[-2:]) for x in [q_part, k_part, v_part]
            ]
        shape_meta = q_part.shape
        # unpad
        valid_cu_seqlens_q, valid_cu_seqlens_kv = generate_valid_cu_seqlens(
            cu_seqlens_q_per_step, cu_seqlens_kv_per_step
        )
        valid_max_len_q, valid_max_len_kv = get_valid_max_seqlen(
            host_cu_seqlens_q, host_cu_seqlens_kv
        )
        fa_forward_args_thd = [
            valid_cu_seqlens_q,
            valid_cu_seqlens_kv,
            valid_max_len_q,
            valid_max_len_kv,
        ]
        q_indices = get_cu_seqlens_indices(
            host_cu_seqlens_q, host_cu_padded_q, q_part.device, host_cu_seqlens_kv
        )
        q_part = fa_varlen_thd_unpad(q_part, q_indices)

        kv_indices = get_cu_seqlens_indices(
            host_cu_seqlens_kv, host_cu_padded_kv, k_part.device, host_cu_padded_q
        )
        k_part, v_part = [fa_varlen_thd_unpad(x, kv_indices) for x in [k_part, v_part]]

    if is_causal:
        fa_forward_kwargs["window_size"] = (-1, 0)
    else:
        fa_forward_kwargs["window_size"] = (-1, -1)

    fa_outputs = flash_attn_fwd(
        q_part,
        k_part,
        v_part,
        *fa_forward_args_thd,
        causal=is_causal,
        **fa_forward_kwargs,
    )
    out, lse = fa_outputs[0], fa_outputs[5]

    if qkv_format == "thd" or pad_between_seqs:
        # pad
        out_per_step = fa_varlen_thd_pad(out, q_indices, shape_meta)  # thd
        softmax_lse_per_step = fa_varlen_lse_pad(
            lse, q_indices, (shape_meta[1], shape_meta[0])
        )  # h,t
        # thd -> bshd
        if qkv_format != "thd":
            out_per_step = out_per_step.view(
                batch_size, -1, *out_per_step.shape[-2:]
            )  # thd -> bshd
            # lse h,t -> b,h,t
            softmax_lse_per_step = softmax_lse_per_step.view(
                softmax_lse_per_step.shape[0], batch_size, -1
            )
            softmax_lse_per_step = softmax_lse_per_step.transpose(0, 1).contiguous()
    else:
        out_per_step = out
        softmax_lse_per_step = lse
    # bshd -> sbhd
    if qkv_format == "sbhd":
        out_per_step = out_per_step.transpose(0, 1).contiguous()

    return out_per_step, softmax_lse_per_step  # lse: b,h,t or h,t


def _fa3_attn_backward(
    q_,
    k_part,
    v_part,
    out_,
    dout_,
    cu_seqlens_q_per_step,
    cu_seqlens_kv_per_step,
    # max_seqlen_q,
    # max_seqlen_kv,
    # cu_seqlens_q_padded,
    # cu_seqlens_kv_padded,
    softmax_lse,
    is_causal,
    qkv_format,
    pad_between_seqs,
    fa_backward_kwargs,
    host_meta=(None, None, None, None),
):
    (
        host_cu_seqlens_q,
        host_cu_padded_q,
        host_cu_seqlens_kv,
        host_cu_padded_kv,
    ) = host_meta
    q_part = q_
    out_part = out_
    dout_part = dout_

    lse_part = softmax_lse

    # sbhd -> bshd
    if qkv_format == "sbhd":
        q_part, k_part, v_part, out_part, dout_part = [
            x.transpose(1, 0).contiguous()
            for x in [q_part, k_part, v_part, out_part, dout_part]
        ]
    batch_size = q_part.shape[0]

    fa_backward_args_thd = []
    flash_attn_bwd = _flash_attn_varlen_backward
    if qkv_format != "thd" and not pad_between_seqs:  # attn_backward
        flash_attn_bwd = _flash_attn_backward
    else:  # attn_varlen_backward
        # q_, k_part, v_part, out_, dout_, softmax_lse
        # bshd -> thd
        if qkv_format != "thd":
            q_part, k_part, v_part, out_part, dout_part = [
                x.reshape(-1, *x.shape[-2:])
                for x in [q_part, k_part, v_part, out_part, dout_part]
            ]
            # lse: b,h,t -> h,t
            lse_part = lse_part.transpose(0, 1).contiguous()  # h,b,t
            lse_part = lse_part.view(lse_part.shape[0], -1)  # h,t
        shape_meta_q = q_part.shape
        shape_meta_kv = k_part.shape
        # unpad
        valid_cu_seqlens_q, valid_cu_seqlens_kv = generate_valid_cu_seqlens(
            cu_seqlens_q_per_step, cu_seqlens_kv_per_step
        )
        valid_max_len_q, valid_max_len_kv = get_valid_max_seqlen(
            host_cu_seqlens_q, host_cu_seqlens_kv
        )
        fa_backward_args_thd = [
            valid_cu_seqlens_q,
            valid_cu_seqlens_kv,
            valid_max_len_q,
            valid_max_len_kv,
        ]
        q_indices = get_cu_seqlens_indices(
            host_cu_seqlens_q, host_cu_padded_q, q_.device, host_cu_seqlens_kv
        )
        q_part, out_part, dout_part = [
            fa_varlen_thd_unpad(x, q_indices) for x in [q_part, out_part, dout_part]
        ]
        lse_part = fa_varlen_lse_unpad(lse_part, q_indices)  # h,t
        kv_indices = get_cu_seqlens_indices(
            host_cu_seqlens_kv, host_cu_padded_kv, k_part.device, host_cu_seqlens_q
        )
        k_part, v_part = [fa_varlen_thd_unpad(x, kv_indices) for x in [k_part, v_part]]

    dq_ = torch.empty_like(q_part)
    dk_ = torch.empty_like(k_part)
    dv_ = torch.empty_like(v_part)

    if is_causal:
        fa_backward_kwargs["window_size"] = (-1, 0)
    else:
        fa_backward_kwargs["window_size"] = (-1, -1)

    flash_attn_bwd(
        dout_part,
        q_part,
        k_part,
        v_part,
        out_part,
        lse_part,
        dq_,
        dk_,
        dv_,
        *fa_backward_args_thd,
        causal=is_causal,
        **fa_backward_kwargs,
    )

    if qkv_format == "thd" or pad_between_seqs:
        # pad
        dq = fa_varlen_thd_pad(dq_, q_indices, shape_meta_q)
        dk, dv = [fa_varlen_thd_pad(x, kv_indices, shape_meta_kv) for x in [dk_, dv_]]
        # thd -> bshd
        if qkv_format != "thd":
            dq, dk, dv = [
                x.view(batch_size, -1, *x.shape[-2:]) for x in [dq, dk, dv]
            ]  # thd -> bshd
    else:
        dq, dk, dv = dq_, dk_, dv_
    if qkv_format == "sbhd":
        dq, dk, dv = [x.transpose(1, 0).contiguous() for x in [dq, dk, dv]]

    return dq, dk, dv


class FA3RingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        causal,
        dropout_p,
        softmax_scale,
        qkv_format,
        cp_group,
        cp_stream,
        deterministic,
        pad_between_seqs,
        batch_p2p_comm=True,
        host_meta=[None, None, None, None],
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(cp_rank, cp_size, cp_group)

        # print(f"{pad_between_seqs=}")
        softmax_lse_in_packed_format = qkv_format == "thd"

        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]
        host_cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        host_cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        qkv_dtype = q.dtype
        q_f16 = q

        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [
                    x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
                    for x in [q, k, v]
                ]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        fa_forward_kwargs = {"softmax_scale": softmax_scale}
        fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()

        p2p_comm_buffers = [None for _ in range(cp_size)]
        if qkv_format in ["bshd", "sbhd"]:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(-3), v.unsqueeze(-3)), dim=-3)
        else:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]  # type: ignore

        softmax_lse_ = None
        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = attn_p2p_communicate(
                            cp_rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )
                    kv_inputs[i % 2] = p2p_comm_buffers[i]

                    is_half_q, is_half_kv, is_causal = False, False, False
                    # _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
                    #     cu_seqlens_q_padded,
                    #     cu_seqlens_kv_padded,
                    # )
                    if causal:
                        if i == 0:  # q, k, v
                            is_causal = True
                        elif i <= cp_rank:  # q, k0, v0
                            is_half_kv = True
                            # if cu_seqlens_kv_padded is not None:
                            #     _cu_seqlens_kv_padded = cu_seqlens_kv_padded // 2
                        else:  # q1, k, v
                            is_half_q = True
                            # if cu_seqlens_q_padded is not None:
                            #     _cu_seqlens_q_padded = cu_seqlens_q_padded // 2
                    else:
                        pass

                    chunk_idx_q = 1 if is_half_q else -1
                    (
                        cu_seqlens_q_per_step[i],
                        q_inputs[i % 2],
                        host_meta_q,
                    ) = prepare_q_fwd(
                        q,
                        chunk_idx_q,
                        qkv_format,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        pad_between_seqs,
                        cp_size,
                        cp_rank,
                        host_meta[:2],
                    )
                    (
                        cu_seqlens_kv_per_step[i],
                        kv_inputs[i % 2],
                        host_meta_kv,
                    ) = prepare_kv_fwd(
                        k,
                        kv_inputs[i % 2],
                        is_half_kv,
                        qkv_format,
                        cu_seqlens_kv,
                        cu_seqlens_kv_padded,
                        pad_between_seqs,
                        cp_size,
                        (cp_rank - i) % cp_size,
                        host_meta[2:],
                    )

                    host_cu_seqlens_q_per_step[i] = host_meta_q[0]
                    host_cu_seqlens_kv_per_step[i] = host_meta_kv[0]
                    q_part = q_inputs[i % 2]
                    k_part = (
                        kv_inputs[i % 2][..., 0, :, :]  # type: ignore[assignment, index]
                        if qkv_format in ["bshd", "sbhd"]
                        else kv_inputs[i % 2][0]  # type: ignore[assignment, index]
                    )
                    v_part = (
                        kv_inputs[i % 2][..., 1, :, :]  # type: ignore[assignment, index]
                        if qkv_format in ["bshd", "sbhd"]
                        else kv_inputs[i % 2][1]  # type: ignore[assignment, index]
                    )
                    # q_part, k_part, v_part = q_part.contiguous(), k_part.contiguous(), v_part.contiguous()
                    out_per_step[i], softmax_lse_per_step[i] = _fa3_attn_forward(
                        q_part,
                        k_part,
                        v_part,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        # get_max_seqlen(host_cu_seqlens_q_per_step[i]),
                        # get_max_seqlen(host_cu_seqlens_kv_per_step[i]),
                        # _cu_seqlens_q_padded,
                        # _cu_seqlens_kv_padded,
                        is_causal,
                        qkv_format,
                        pad_between_seqs,
                        fa_forward_kwargs,
                        host_meta_q + host_meta_kv,
                    )

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if i == 1:
                        out = torch.zeros_like(q)
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(
                            torch.double
                        )
                        if causal and qkv_format != "thd":
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                            )
                    elif (i - 1) <= cp_rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                        else:
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse_[..., 1, :], softmax_lse_per_step[i - 1]  # type: ignore[assignment, index]
                            )

                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(
                        fwd_results_correction_done
                    )

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and cp_rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]  # type: ignore[attr-defined]

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            is_half = not (i <= cp_rank or not causal)
            out = fwd_out_update(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                qkv_format,
                softmax_lse_in_packed_format,
            )

        kv = p2p_comm_buffers[-1]
        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])  # type: ignore[union-attr]
            ctx.batch_size = out.shape[0]
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])  # type: ignore[union-attr]
            ctx.batch_size = out.shape[1]

        out_f16 = out.to(qkv_dtype)  # type: ignore[union-attr]
        out_ret = out_f16
        q_f16 = q_f16.view(q.shape)
        q_save, kv_save, out_save = q_f16, kv, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.batch_p2p_comm = batch_p2p_comm
        ctx.deterministic = deterministic
        ctx.pad_between_seqs = pad_between_seqs
        ctx.host_cu_seqlens_q_per_step = host_cu_seqlens_q_per_step
        ctx.host_cu_seqlens_kv_per_step = host_cu_seqlens_kv_per_step
        ctx.host_meta = host_meta

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        cp_size = torch.distributed.get_world_size(group=ctx.cp_group)
        cp_rank = torch.distributed.get_rank(group=ctx.cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(
            cp_rank, cp_size, ctx.cp_group, reverse=True
        )
        batch_p2p_comm = ctx.batch_p2p_comm

        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        cu_seqlens_q_per_step = other_tensors[:cp_size]
        cu_seqlens_kv_per_step = other_tensors[cp_size : cp_size * 2]
        # max_seqlens_q = other_tensors[cp_size * 2 : cp_size * 3]
        # max_seqlens_kv = other_tensors[cp_size * 3 : cp_size * 4]
        causal = ctx.causal
        host_cu_seqlens_q_per_step = ctx.host_cu_seqlens_q_per_step
        host_cu_seqlens_kv_per_step = ctx.host_cu_seqlens_kv_per_step

        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse,
                    cu_seqlens_q_padded,
                    ctx.softmax_lse_in_packed_format,
                    ctx.second_half_lse_seqlen,
                )
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = softmax_lse.view(
                    *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                )
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()

        dq = torch.empty_like(q)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]
        p2p_comm_buffers[0][0].copy_(kv)
        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

        fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
        fa_backward_kwargs["deterministic"] = ctx.deterministic

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]

            if i == 0:
                send_tensor = send_tensor[0]
                recv_tensor = recv_tensor[0]
            if i == (cp_size - 1):
                send_tensor = send_tensor[1]
                recv_tensor = recv_tensor[1]
            send_recv_reqs = attn_p2p_communicate(
                cp_rank,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                ctx.cp_group,
                batch_p2p_comm,
            )

            kv = p2p_comm_buffers[i % 2][0]
            q_, kv_, out_, dout_ = None, None, None, None
            dq_, dk_, dv_ = None, None, None
            # In reversed order of fwd
            is_half_q, is_half_kv, is_causal = False, False, False
            # _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
            #     cu_seqlens_q_padded,
            #     cu_seqlens_kv_padded,
            # )
            lse = softmax_lse
            if causal:
                if i == (cp_size - 1):  # q, k, v
                    is_causal = True
                elif i >= (cp_size - cp_rank - 1):  # q, k0, v0
                    is_half_kv = True
                    # _cu_seqlens_kv_padded = cu_seqlens_kv_padded // 2
                else:  # q1, k, v
                    is_half_q = True
                    lse = softmax_lse_
                    # _cu_seqlens_q_padded = cu_seqlens_q_padded // 2
            else:
                pass

            chunk_idx_q = 1 if is_half_q else -1
            q_, out_, dout_ = prepare_q_bwd(
                [q, out, dout], chunk_idx_q, cu_seqlens_q_padded, ctx.qkv_format
            )
            kv_ = prepare_kv_bwd(kv, is_half_kv, cu_seqlens_kv_padded, ctx.qkv_format)
            k_part = kv_[..., 0, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[0]
            v_part = kv_[..., 1, :, :] if ctx.qkv_format in ["bshd", "sbhd"] else kv_[1]
            host_meta_per_step = [
                host_cu_seqlens_q_per_step[cp_size - i - 1],
                ctx.host_meta[1],
                host_cu_seqlens_kv_per_step[cp_size - i - 1],
                ctx.host_meta[3],
            ]
            if is_half_q:
                host_meta_per_step[1] = divide_lst(host_meta_per_step[1], 2)
            if is_half_kv:
                host_meta_per_step[3] = divide_lst(host_meta_per_step[3], 2)
            # if ctx.pad_between_seqs and ctx.qkv_format != "thd":
            #     q_, k_part, v_part = q_.contiguous(), k_part.contiguous(), v_part.contiguous()
            dq_, dk_, dv_ = _fa3_attn_backward(
                q_,
                k_part,
                v_part,
                out_,
                dout_,
                cu_seqlens_q_per_step[cp_size - i - 1],
                cu_seqlens_kv_per_step[cp_size - i - 1],
                # get_max_seqlen(host_cu_seqlens_q_per_step[cp_size - i - 1]),
                # get_max_seqlen(host_cu_seqlens_kv_per_step[cp_size - i - 1]),
                # _cu_seqlens_q_padded,
                # _cu_seqlens_kv_padded,
                lse,
                is_causal,
                ctx.qkv_format,
                ctx.pad_between_seqs,
                fa_backward_kwargs,
                host_meta_per_step,
            )

            if (
                causal
                and ctx.qkv_format in ["bshd", "sbhd"]
                and i >= (cp_size - cp_rank - 1)
            ):
                # [b, sq, np, hn] -> [b, 2, sq//2, np, hn] or
                # [sq, b, np, hn] -> [2, sq//2, b, np, hn]
                dq_ = dq_.view(*dq.shape)

            # update dq
            first_op, second_op = "none", "none"
            if causal:
                if i > (cp_size - cp_rank - 1):  # q add
                    first_op = second_op = "add"
                elif i == (cp_size - cp_rank - 1):
                    if cp_rank == (cp_size - 1):  # q 0 iter copy
                        first_op = second_op = "copy"
                    else:  # q1 -> q copy & add
                        first_op = "copy"
                        second_op = "add"
                elif i > 0:  # q1, k, v add
                    second_op = "add"
                else:  # q1, k, v copy
                    second_op = "copy"
            else:
                if i == 0:
                    first_op = second_op = "copy"
                else:
                    first_op = second_op = "add"
            dq = bwd_dq_update(
                dq, dq_, cu_seqlens_q_padded, ctx.qkv_format, first_op, second_op
            )

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()
            dkv = p2p_comm_buffers[(i + 1) % 2][1]

            if ctx.qkv_format in ["bshd", "sbhd"]:
                dkv_ = torch.cat((dk_.unsqueeze(-3), dv_.unsqueeze(-3)), dim=-3)
            elif ctx.qkv_format == "thd":
                dkv_ = torch.cat(
                    (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                )  # pylint: disable=used-before-assignment
            if ctx.qkv_format in ["bshd", "sbhd"]:
                # [b, 2, sk//2, 2, np, hn] -> [2, b, 2, sk//2, np, hn] or
                # [2, sk//2, b, 2, np, hn] -> [2, 2, sk//2, b, np, hn]
                dkv = dkv.view(2, *dkv.shape[0:-3], *dkv.shape[-2:])
                dkv_ = dkv_.movedim(-3, 0)
                if causal and (i < (cp_size - cp_rank - 1) or i == (cp_size - 1)):
                    # [2, b, sk, np, hn] -> [2, b, 2, sk//2, np, hn] or
                    # [2, sk, b, np, hn] -> [2, 2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(*dkv.shape)

            # update dkv
            first_op, second_op = "none", "none"
            if causal:
                if i == (cp_size - 1):  # k, v
                    if cp_rank == 0:  # copy
                        first_op = "add"
                        second_op = "copy"
                    else:  # k, v add
                        first_op = second_op = "add"
                elif i >= (cp_size - cp_rank - 1):  # k0, v0
                    if i == 0 and cp_rank == (cp_size - 1):  # copy 0 iter
                        first_op = "copy"
                    else:  # add k0, v0
                        first_op = "add"
                elif i > 0:  # k, v add
                    first_op = second_op = "add"
                else:  # k, v, copy
                    first_op = second_op = "copy"
            else:
                if i == 0:
                    first_op = second_op = "copy"
                else:
                    first_op = second_op = "add"
            dkv = bwd_dkv_update(
                dkv, dkv_, cu_seqlens_kv_padded, ctx.qkv_format, first_op, second_op
            )

        if causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                dkv = dkv.view(*dkv.shape[0:2], -1, *dkv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *dq.shape[-3:])
                # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                dkv = dkv.view(dkv.shape[0], -1, *dkv.shape[-3:])
        if ctx.qkv_format == "thd":
            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            dkv[:, cu_seqlens_kv_padded[-1] :].fill_(0)
        dk, dv = dkv[0], dkv[1]

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class TERingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,  # int max(cu_seqlens_padded)
        max_seqlen_kv,  # int max(cu_seqlens_padded)
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        dropout_p,
        softmax_scale,
        qkv_format,
        cp_group,
        attn_mask_type,
        cp_stream,
        deterministic,
        pad_between_seqs,
        batch_p2p_comm=True,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(cp_rank, cp_size, cp_group)

        if qkv_format in ["bshd", "sbhd"]:
            qkv_layout = qkv_format + "_" + qkv_format[:-2] + "2" + qkv_format[-2:]
        else:
            qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        qkv_dtype = q.dtype
        q_f16 = q
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        if causal:
            if qkv_format == "bshd":
                # [b, s, np, hn] -> [b, 2, s//2, np, hn]
                q, k, v = [
                    x.view(x.shape[0], 2, x.shape[1] // 2, *x.shape[2:])
                    for x in [q, k, v]
                ]
            elif qkv_format == "sbhd":
                # [s, b, np, hn] -> [2, s//2, b, np, hn]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = False
        if qkv_format == "thd":
            softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)

        # Flash Attn inputs
        q_inputs = [None, None]
        kv_inputs = [None, None]
        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]

        p2p_comm_buffers = [None for _ in range(cp_size)]
        if qkv_format in ["bshd", "sbhd"]:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(-3), v.unsqueeze(-3)), dim=-3)
        else:
            p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        send_recv_reqs = [[], []]  # type: ignore

        softmax_lse_ = None
        out = None

        fused_attn_meta_args = (qkv_dtype, fused_attn_backend)
        fused_attn_meta_kwargs = {
            "attn_scale": softmax_scale,
            "dropout": dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": attn_mask_type,
            "attn_bias_type": "no_bias",
            "attn_bias": None,
        }
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
                        send_recv_reqs[i % 2] = attn_p2p_communicate(
                            cp_rank,
                            p2p_comm_buffers[i],
                            send_dst,
                            p2p_comm_buffers[i + 1],
                            recv_src,
                            cp_group,
                            batch_p2p_comm,
                        )
                    # contiguous tensor
                    kv_inputs[i % 2] = p2p_comm_buffers[i]

                    is_half_q, is_half_kv = False, False
                    _max_seqlen_q, _max_seqlen_kv = max_seqlen_q, max_seqlen_kv
                    if causal:
                        if i == 0:  # q, k, v
                            pass
                        elif i <= cp_rank:  # q, k0, v0
                            fused_attn_meta_kwargs["attn_mask_type"] = (
                                "padding" if padding else "no_mask"
                            )
                            is_half_kv = True
                            _max_seqlen_kv = max_seqlen_kv // 2
                        else:  # q1, k, v
                            fused_attn_meta_kwargs["attn_mask_type"] = (
                                "padding" if padding else "no_mask"
                            )
                            is_half_q = True
                            _max_seqlen_q = max_seqlen_q // 2
                    else:  # full
                        pass

                    chunk_idx_q = 1 if is_half_q else -1
                    cu_seqlens_q_per_step[i], q_inputs[i % 2], _ = prepare_q_fwd(
                        q,
                        chunk_idx_q,
                        qkv_format,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        pad_between_seqs,
                        cp_size,
                        cp_rank,
                    )
                    cu_seqlens_kv_per_step[i], kv_inputs[i % 2], _ = prepare_kv_fwd(
                        k,
                        kv_inputs[i % 2],
                        is_half_kv,
                        qkv_format,
                        cu_seqlens_kv,
                        cu_seqlens_kv_padded,
                        pad_between_seqs,
                        cp_size,
                        (cp_rank - i) % cp_size,
                    )
                    if is_half_q:
                        q_inputs[i % 2] = q_inputs[i % 2].contiguous()  # type: ignore[attr-defined]
                    if is_half_kv:
                        kv_inputs[i % 2] = kv_inputs[i % 2].contiguous()  # type: ignore[attr-defined]
                    (
                        out_per_step[i],
                        softmax_lse_per_step[i],
                        rng_states[i],
                    ) = _fused_attn_forward(
                        q_inputs[i % 2],
                        kv_inputs[i % 2],
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        _max_seqlen_q,
                        _max_seqlen_kv,
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                        is_half_q,
                        is_half_kv,
                        qkv_format,
                        fused_attn_meta_args,
                        fused_attn_meta_kwargs,
                    )

            if i > 0:
                # wait until fwd restuls correction of last step is done
                if i > 1:
                    flash_attn_streams[(i - 1) % 2].wait_event(
                        fwd_results_correction_done
                    )
                # [b, np, sq, 1] -> [b, np, sq]
                # or [t, np, 1] -> [t, np]
                softmax_lse_per_step[i - 1].squeeze_(-1)  # type: ignore[attr-defined]
                if softmax_lse_in_packed_format:
                    softmax_lse_per_step[i - 1] = (
                        softmax_lse_per_step[i - 1].transpose(0, 1).contiguous()  # type: ignore[attr-defined]
                    )

                with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                    if i == 1:
                        out = torch.zeros_like(q)
                        softmax_lse = torch.clone(softmax_lse_per_step[0]).to(
                            torch.double
                        )
                        if causal and qkv_format != "thd":
                            # [b, np, sq] -> [b, np, 2, sq//2]
                            softmax_lse_ = softmax_lse.view(
                                *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                            )
                    elif (i - 1) <= cp_rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        if qkv_format == "thd":
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                        else:
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse_[..., 1, :], softmax_lse_per_step[i - 1]  # type: ignore[index]
                            )
                if i < cp_size:
                    flash_attn_streams[(i - 1) % 2].record_event(
                        fwd_results_correction_done
                    )

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and cp_rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]  # type: ignore[attr-defined]

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            is_half = not (i <= cp_rank or not causal)
            out = fwd_out_update(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                qkv_format,
                softmax_lse_in_packed_format,
            )

        kv = p2p_comm_buffers[-1]
        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])  # type: ignore[union-attr]
            ctx.batch_size = out.shape[0]
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])  # type: ignore[union-attr]
            ctx.batch_size = out.shape[1]

        out_f16 = out.to(qkv_dtype)  # type: ignore[union-attr]
        out_ret = out_f16
        q_f16 = q_f16.view(q.shape)
        q_save, kv_save, out_save = q_f16, kv, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            kv_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *rng_states,
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.batch_p2p_comm = batch_p2p_comm
        ctx.deterministic = deterministic

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        dout = dout.contiguous()

        cp_size = torch.distributed.get_world_size(group=ctx.cp_group)
        cp_rank = torch.distributed.get_rank(group=ctx.cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(
            cp_rank, cp_size, ctx.cp_group, reverse=True
        )
        batch_p2p_comm = ctx.batch_p2p_comm

        (
            q,
            kv,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        cu_seqlens_q_per_step = other_tensors[:cp_size]
        cu_seqlens_kv_per_step = other_tensors[cp_size : cp_size * 2]
        rng_states = other_tensors[cp_size * 2 : cp_size * 3]

        causal = "causal" in ctx.attn_mask_type
        padding = "padding" in ctx.attn_mask_type

        if ctx.qkv_format in ["bshd", "sbhd"]:
            qkv_layout = (
                ctx.qkv_format + "_" + ctx.qkv_format[:-2] + "2" + ctx.qkv_format[-2:]
            )
        else:
            qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            if ctx.qkv_format == "thd":
                softmax_lse_ = tex.thd_read_second_half_lse(
                    softmax_lse,
                    cu_seqlens_q_padded,
                    ctx.softmax_lse_in_packed_format,
                    ctx.second_half_lse_seqlen,
                )
            else:
                # [b, np, sq] -> [b, np, 2, sq//2]
                softmax_lse_ = softmax_lse.view(
                    *softmax_lse.shape[:-1], 2, softmax_lse.shape[-1] // 2
                )
                softmax_lse_ = softmax_lse_[..., 1, :].contiguous()

            if ctx.softmax_lse_in_packed_format:
                softmax_lse_ = softmax_lse_.transpose(0, 1).contiguous()
            # [b, np, sq//2] -> [b, np, sq//2, 1] or
            # [t//2, np] -> [t//2, np, 1]
            softmax_lse_.unsqueeze_(-1)
        if ctx.softmax_lse_in_packed_format:
            softmax_lse = softmax_lse.transpose(0, 1).contiguous()
        # [b, np, sq] -> [b, np, sq, 1] or [t, np] -> [t, np, 1]
        softmax_lse.unsqueeze_(-1)

        dout_dtype = dout.dtype
        dq = torch.empty_like(q)
        # dq = torch.empty_like(q)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]
        p2p_comm_buffers[0][0].copy_(kv)
        fused_attn_dqkv_dtype = TE_DType[dout_dtype]
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

        fused_attn_meta_args = [
            ctx.qkv_dtype,
            fused_attn_dqkv_dtype,
            None,
            fused_attn_backend,
        ]
        fused_attn_meta_kwargs = {
            "attn_scale": ctx.softmax_scale,
            "dropout": ctx.dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": ctx.attn_mask_type,
            "attn_bias_type": "no_bias",
            "deterministic": ctx.deterministic,
        }

        for i in range(cp_size):
            # wait until KV is received
            for req in send_recv_reqs:
                req.wait()

            send_tensor = p2p_comm_buffers[i % 2]
            recv_tensor = p2p_comm_buffers[(i + 1) % 2]

            if i == 0:
                send_tensor = send_tensor[0]
                recv_tensor = recv_tensor[0]
            if i == (cp_size - 1):
                send_tensor = send_tensor[1]
                recv_tensor = recv_tensor[1]
            send_recv_reqs = attn_p2p_communicate(
                cp_rank,
                send_tensor,
                send_dst,
                recv_tensor,
                recv_src,
                ctx.cp_group,
                batch_p2p_comm,
            )

            kv = p2p_comm_buffers[i % 2][0]
            dq_, dk_, dv_ = None, None, None
            # In reversed order of fwd

            is_half_q, is_half_kv = False, False
            _max_seqlen_q, _max_seqlen_kv = ctx.max_seqlen_q, ctx.max_seqlen_kv
            if causal:
                if i == (cp_size - 1):  # q, k, v
                    fused_attn_meta_kwargs["attn_mask_type"] = ctx.attn_mask_type
                    aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    fused_attn_meta_args[2] = aux_ctx_tensors
                elif i >= (cp_size - cp_rank - 1):  # q, k0, v0
                    aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                    fused_attn_meta_args[2] = aux_ctx_tensors
                    fused_attn_meta_kwargs["attn_mask_type"] = (
                        "padding" if padding else "no_mask"
                    )
                    is_half_kv = True
                    _max_seqlen_kv = ctx.max_seqlen_kv // 2
                else:  # q1, k, v
                    assert softmax_lse_ is not None
                    aux_ctx_tensors = [softmax_lse_, rng_states[cp_size - i - 1]]
                    fused_attn_meta_args[2] = aux_ctx_tensors
                    fused_attn_meta_kwargs["attn_mask_type"] = (
                        "padding" if padding else "no_mask"
                    )
                    is_half_q = True
                    _max_seqlen_q = ctx.max_seqlen_q // 2
            else:
                aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                fused_attn_meta_args[2] = aux_ctx_tensors

            chunk_idx_q = 1 if is_half_q else -1
            q_, out_, dout_ = prepare_q_bwd(
                [q, out, dout], chunk_idx_q, cu_seqlens_q_padded, ctx.qkv_format
            )
            kv_ = prepare_kv_bwd(kv, is_half_kv, cu_seqlens_kv_padded, ctx.qkv_format)
            if is_half_q:
                q_, out_, dout_ = [x.contiguous() for x in [q_, out_, dout_]]
            if is_half_kv:
                kv_ = kv_.contiguous()

            dq_, dk_, dv_ = _fused_attn_backward(
                q_,
                kv_,
                out_,
                dout_,
                cu_seqlens_q_per_step[cp_size - i - 1],
                cu_seqlens_kv_per_step[cp_size - i - 1],
                _max_seqlen_q,
                _max_seqlen_kv,
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
                is_half_q,
                is_half_kv,
                ctx.qkv_format,
                fused_attn_meta_args,
                fused_attn_meta_kwargs,
            )

            if (
                causal
                and ctx.qkv_format in ["bshd", "sbhd"]
                and i >= (cp_size - cp_rank - 1)
            ):
                # [b, sq, np, hn] -> [b, 2, sq//2, np, hn] or
                # [sq, b, np, hn] -> [2, sq//2, b, np, hn]
                dq_ = dq_.view(*dq.shape)
            # update dq
            first_op, second_op = "none", "none"
            if causal:
                if i > (cp_size - cp_rank - 1):  # q add
                    first_op = second_op = "add"
                elif i == (cp_size - cp_rank - 1):
                    if cp_rank == (cp_size - 1):  # q 0 iter copy
                        first_op = second_op = "copy"
                    else:  # q1 -> q copy & add
                        first_op = "copy"
                        second_op = "add"
                elif i > 0:  # q1, k, v add
                    second_op = "add"
                else:  # q1, k, v copy
                    second_op = "copy"
            else:
                if i == 0:
                    first_op = second_op = "copy"
                else:
                    first_op = second_op = "add"

            dq = bwd_dq_update(
                dq, dq_, cu_seqlens_q_padded, ctx.qkv_format, first_op, second_op
            )

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            dkv = p2p_comm_buffers[(i + 1) % 2][1]
            if ctx.qkv_format in ["bshd", "sbhd"]:
                dkv_ = _combine_tensors([dk_, dv_], -2)
            elif ctx.qkv_format == "thd":
                dkv_ = torch.cat(
                    (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                )  # pylint: disable=used-before-assignment
            if ctx.qkv_format in ["bshd", "sbhd"]:
                # [b, 2, sk//2, 2, np, hn] -> [2, b, 2, sk//2, np, hn] or
                # [2, sk//2, b, 2, np, hn] -> [2, 2, sk//2, b, np, hn]
                dkv = dkv.view(2, *dkv.shape[0:-3], *dkv.shape[-2:])
                dkv_ = dkv_.movedim(-3, 0)
                if causal and (i < (cp_size - cp_rank - 1) or i == (cp_size - 1)):
                    # [2, b, sk, np, hn] -> [2, b, 2, sk//2, np, hn] or
                    # [2, sk, b, np, hn] -> [2, 2, sk//2, b, np, hn]
                    dkv_ = dkv_.view(*dkv.shape)

            # update dkv
            first_op, second_op = "none", "none"
            if causal:
                if i == (cp_size - 1):  # k, v
                    if cp_rank == 0:  # copy
                        first_op = "add"
                        second_op = "copy"
                    else:  # k, v add
                        first_op = second_op = "add"
                elif i >= (cp_size - cp_rank - 1):  # k0, v0
                    if i == 0 and cp_rank == (cp_size - 1):  # copy 0 iter
                        first_op = "copy"
                    else:  # add k0, v0
                        first_op = "add"
                elif i > 0:  # k, v add
                    first_op = second_op = "add"
                else:  # k, v, copy
                    first_op = second_op = "copy"
            else:
                if i == 0:
                    first_op = second_op = "copy"
                else:
                    first_op = second_op = "add"
            dkv = bwd_dkv_update(
                dkv, dkv_, cu_seqlens_kv_padded, ctx.qkv_format, first_op, second_op
            )

        if causal:
            if ctx.qkv_format == "bshd":
                # [b, 2, sq//2, np, hn] -> [b, sq, np, hn]
                dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
                # [2, b, 2, sk//2, np, hn] -> [2, b, sk, np, hn]
                dkv = dkv.view(*dkv.shape[0:2], -1, *dkv.shape[-2:])
            elif ctx.qkv_format == "sbhd":
                # [2, sq//2, b, np, hn] -> [sq, b, np, hn]
                dq = dq.view(-1, *dq.shape[-3:])
                # [2, 2, sk//2, b, np, hn] -> [2, sk, b, np, hn]
                dkv = dkv.view(dkv.shape[0], -1, *dkv.shape[-3:])

        dk, dv = dkv[0], dkv[1]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def get_max_seqlen(lst):
    batch_size = len(lst) - 1
    return max(lst[i + 1] - lst[i] for i in range(batch_size))


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
