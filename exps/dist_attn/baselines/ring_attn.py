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

from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist

# te
import transformer_engine as te  # noqa
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import reduce_scatter_along_first_dim
from transformer_engine.pytorch.utils import get_cudnn_version

from exps.dist_attn.baselines.interface import AttnBaselineInterface
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    ShardMeta,
    generate_zigzag_dispatch_indices,
    generate_zigzag_undispatch_indices,
    get_cu_seqlens_padded,
    get_max_seqlen,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from exps.dist_attn.baselines.utils_cp import (
    prepare_for_saving,  # type: ignore[attr-defined]
)
from exps.dist_attn.baselines.utils_cp import (
    restore_from_saved,  # type: ignore[attr-defined]
)
from exps.dist_attn.baselines.utils_cp import (
    AttnBackend,
    _fa3_varlen_backward,
    _fa3_varlen_forward,
    _pre_process,
    attn_p2p_communicate,
    bwd_dkv_update,
    bwd_dq_update,
    flash_attn_fwd_softmax_lse_correction,
    generate_runtime_meta_per_step,
    get_cu_seqlens_on_cp_rank,
    get_p2p_send_recv_rank,
    unflatten_data_from_varlen,
)
from magi_attention.comm.functional import all_gather_fwd_scatter_bwd
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

jit_fuser = torch.jit.script


# all-gather with zigzag to contiguous
def gather_with_reorder_before_attn(
    input: torch.Tensor,
    total_indices: torch.Tensor,
    cp_group,
):
    other_shape = input.shape[1:]
    out = all_gather_fwd_scatter_bwd(input, cp_group, dim=0).contiguous()
    output = torch.gather(
        out, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
    )

    return output


# contiguous to zigzag
def reorder_before_reduce_scatter(
    input: torch.Tensor,
    total_indices: torch.Tensor,
):
    other_shape = input.shape[1:]
    output = torch.gather(
        input, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
    )

    return output


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


# generate total zigzag indices
def generate_scattar_reorder_indices(
    cu_seqlens_padded_host: List[int],
    cp_size,
):
    batch_size = len(cu_seqlens_padded_host) - 1
    zigzag_indices_lst = []
    for cp_rank in range(cp_size):
        zigzag_indices, _ = generate_zigzag_dispatch_indices(
            cu_seqlens_padded_host,
            cu_seqlens_padded_host,
            cu_seqlens_padded_host[:batch_size],
            None,  # type: ignore[arg-type]
            cp_size,
            cp_rank,
        )
        zigzag_indices_lst.append(zigzag_indices)
    zigzag_total_indices_np = np.concatenate(zigzag_indices_lst)
    return zigzag_total_indices_np


# use te tex.thd_grad_correction for varlen result collection
def _collect_result_varlen(
    out: torch.Tensor, out_: torch.Tensor, cu_seqlens_padded: torch.Tensor, chunk_idx
):
    if chunk_idx == 0:
        first_op, second_op = "copy", "none"
    elif chunk_idx == 1:
        first_op, second_op = "none", "copy"
    tex.thd_grad_correction(out, out_, cu_seqlens_padded, first_op, second_op)


def prepare_input_fwd(
    input,
    chunk_idx,
    cu_seqlens,
    cu_seqlens_padded,
    cp_size,
    cp_rank,
):
    is_half = False
    if chunk_idx >= 0:
        is_half = True
    if is_half:  # chunk0 | chunk1
        first_index, second_index = ((chunk_idx == 0), (chunk_idx == 1))
    else:  # q
        first_index, second_index = True, True
    cu_seqlens_per_step = get_cu_seqlens_on_cp_rank(
        cu_seqlens,
        cu_seqlens_padded,
        cp_size,
        cp_rank,
        first_index,
        second_index,
    )
    if is_half:
        output = tex.thd_read_half_tensor(input, cu_seqlens_padded, chunk_idx)
        output = output.contiguous()
    else:
        output = input
    return output, cu_seqlens_per_step


def prepare_input_bwd(
    inputs,
    chunk_idx,
    cu_seqlens_padded,
):
    is_half = False
    if chunk_idx >= 0:
        is_half = True
    if is_half:
        output = [
            tex.thd_read_half_tensor(x, cu_seqlens_padded, chunk_idx) for x in inputs
        ]
        output = [x.contiguous() for x in output]
    else:
        output = inputs
    return output


class TERingAGAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,  # int
        max_seqlen_kv,  # int
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        total_gather_indices,
        total_scatter_indices,
        dropout_p,
        softmax_scale,
        qkv_format,
        cp_group,
        attn_mask_type,
        cp_stream,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)

        causal = "causal" in attn_mask_type
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        k_ag = gather_with_reorder_before_attn(k, total_gather_indices, cp_group)
        v_ag = gather_with_reorder_before_attn(v, total_gather_indices, cp_group)
        cp_stream.wait_stream(torch.cuda.current_stream())

        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        local_seq_chunk_idx = [cp_rank, 2 * cp_size - cp_rank - 1]
        local_seq_num = 2

        cu_seqlens_q_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]

        # thd lse [t,h,16]
        softmax_lse = torch.empty(
            *(q.shape[0], q.shape[1], 16), dtype=torch.float, device=q.device
        )
        out = torch.empty_like(q)

        qkv_dtype = q.dtype
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]
        fused_attn_meta_args = (qkv_dtype, fused_attn_backend)
        fused_attn_meta_kwargs = {
            "attn_scale": softmax_scale,
            "dropout": dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": attn_mask_type,
            "attn_bias_type": "no_bias",
            "attn_bias": None,
        }

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    q_part, cu_seqlens_q_per_step[i] = prepare_input_fwd(
                        q,
                        i,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        cp_size,
                        cp_rank,
                    )

                    if causal:
                        cu_seqlens_kv_per_step[i] = generate_cu_seqlens_kv_ag_causal(
                            cu_seqlens_kv,
                            cu_seqlens_kv_padded,
                            local_seq_chunk_idx[i],
                            cp_size,
                        )
                    else:
                        cu_seqlens_kv_per_step[i] = cu_seqlens_kv

                    out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                        True,
                        max_seqlen_q // 2,
                        max_seqlen_kv,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        q_part,
                        k_ag,
                        v_ag,
                        *fused_attn_meta_args,
                        **fused_attn_meta_kwargs,
                        cu_seqlens_q_padded=cu_seqlens_q_padded // 2,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        **{},
                    )
                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    _collect_result_varlen(
                        out, out_per_step[i - 1], cu_seqlens_q_padded, i - 1
                    )
                    # TODO: te softmax_lse_per_step[i - 1] shape
                    # lse_per_step = softmax_lse_per_step[i - 1]

                    # _collect_result_varlen(
                    #     softmax_lse,
                    #     lse_per_step.expand(*lse_per_step.shape[:-1], 16),
                    #     cu_seqlens_q_padded,
                    #     i - 1,
                    # )

        # [t,h,16] -> [h,t]
        softmax_lse = softmax_lse[:, :, 0].transpose(0, 1).contiguous()
        torch.cuda.current_stream().wait_stream(cp_stream)

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *cu_seqlens_q_per_step,
            *cu_seqlens_kv_per_step,
            *out_per_step,
            *softmax_lse_per_step,
            *rng_states,
        )

        ctx.total_gather_indices = total_gather_indices
        ctx.total_scatter_indices = total_scatter_indices
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.causal = causal

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        dout = dout.contiguous()
        cp_size = torch.distributed.get_world_size(group=ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q_padded, cu_seqlens_kv_padded) = saved_tensors[:5]
        cu_seqlens_q_per_step = saved_tensors[5:7]
        cu_seqlens_kv_per_step = saved_tensors[7:9]
        out_per_step = saved_tensors[9:11]
        softmax_lse_per_step = saved_tensors[11:13]
        rng_states = saved_tensors[13:15]

        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        dk = torch.zeros(
            (k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device
        )
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        k_ag = gather_with_reorder_before_attn(
            k, ctx.total_gather_indices, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, ctx.total_gather_indices, ctx.cp_group
        )
        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        local_seq_num = 2
        fused_attn_meta_args = [
            ctx.qkv_dtype,
            TE_DType[dout.dtype],
            None,
            FusedAttnBackend["F16_arbitrary_seqlen"],
        ]
        fused_attn_meta_kwargs = {
            "attn_scale": ctx.softmax_scale,
            "dropout": ctx.dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": ctx.attn_mask_type,
            "attn_bias_type": "no_bias",
            "deterministic": ctx.deterministic,
        }

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    out_part = out_per_step[i]
                    q_part = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, i)
                    dout_part = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, i)
                    # q_part, dout_part = q_part.contiguous(), dout_part.contiguous()
                    aux_ctx_tensors = [softmax_lse_per_step[i], rng_states[i]]
                    fused_attn_meta_args[2] = aux_ctx_tensors
                    dq_per_step[i], dk_per_step[i], dv_per_step[i], _ = fused_attn_bwd(
                        ctx.max_seqlen_q // 2,
                        ctx.max_seqlen_kv,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        q_part,
                        k_ag,
                        v_ag,
                        out_part,
                        dout_part,
                        *fused_attn_meta_args,
                        cu_seqlens_q_padded=cu_seqlens_q_padded // 2,
                        cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                        **fused_attn_meta_kwargs,
                    )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    _collect_result_varlen(
                        dq, dq_per_step[i - 1], cu_seqlens_q_padded, i - 1
                    )
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)

                    dk.add_(dk_per_step[i - 1])
                    dv.add_(dv_per_step[i - 1])
                    if i < local_seq_num:
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        dk = reorder_before_reduce_scatter(dk, ctx.total_scatter_indices)
        dv = reorder_before_reduce_scatter(dv, ctx.total_scatter_indices)

        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk_part,
            dv_part,
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


class FA3RingAGAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q_padded,
        total_gather_indices,
        total_scatter_indices,
        runtime_meta,
        causal,
        dropout_p,
        softmax_scale,
        cp_group,
        cp_stream,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        qkv_dtype = q.dtype
        fa_forward_kwargs = {"window_size": (-1, -1)}

        k_ag = gather_with_reorder_before_attn(k, total_gather_indices, cp_group)
        v_ag = gather_with_reorder_before_attn(v, total_gather_indices, cp_group)
        cp_stream.wait_stream(torch.cuda.current_stream())

        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        local_seq_num = 2
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]

        # thd lse [t,h,16]
        softmax_lse = torch.empty(
            *(q.shape[0], q.shape[1], 16), dtype=torch.float, device=q.device
        )
        out = torch.empty_like(q)

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    rumtime_meta_per_step = runtime_meta[i]
                    q_part = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, i)
                    out_per_step[i], softmax_lse_per_step[i] = _fa3_varlen_forward(
                        q_part,
                        k_ag,
                        v_ag,
                        softmax_scale,
                        causal,
                        rumtime_meta_per_step,
                        fa_forward_kwargs,
                    )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    _collect_result_varlen(
                        out, out_per_step[i - 1], cu_seqlens_q_padded, i - 1
                    )
                    # [h,t] -> [t,h]
                    # lse_per_step = (
                    #     softmax_lse_per_step[i - 1].transpose(0, 1).contiguous()
                    # )
                    # lse_per_step = lse_per_step[:, :, None].expand(*lse_per_step.shape, 16)
                    # _collect_result_varlen(
                    #     softmax_lse,
                    #     lse_per_step,
                    #     cu_seqlens_q_padded,
                    #     i - 1,
                    # )

        # softmax_lse
        # [t,h,16] -> [h,t]
        softmax_lse = softmax_lse[:, :, 0]
        # .transpose(0, 1).contiguous()

        torch.cuda.current_stream().wait_stream(cp_stream)

        ctx.save_for_backward(
            q,
            k,
            v,
            cu_seqlens_q_padded,
            *out_per_step,
            *softmax_lse_per_step,
        )
        ctx.total_gather_indices = total_gather_indices
        ctx.total_scatter_indices = total_scatter_indices
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.causal = causal
        ctx.runtime_meta = runtime_meta

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        cp_size = torch.distributed.get_world_size(group=ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q_padded) = saved_tensors[:4]
        out_per_step = saved_tensors[4:6]
        softmax_lse_per_step = saved_tensors[6:8]

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        dk = torch.zeros(
            (k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device
        )
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        k_ag = gather_with_reorder_before_attn(
            k, ctx.total_gather_indices, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, ctx.total_gather_indices, ctx.cp_group
        )

        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        local_seq_num = 2

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    rumtime_meta_per_step = ctx.runtime_meta[i]
                    out_part = out_per_step[i]
                    q_part = tex.thd_read_half_tensor(q, cu_seqlens_q_padded, i)
                    dout_part = tex.thd_read_half_tensor(dout, cu_seqlens_q_padded, i)
                    # q_part, dout_part = q_part.contiguous(), dout_part.contiguous()

                    window_size = (-1, 0) if ctx.causal else (-1, -1)
                    (
                        dq_per_step[i],
                        dk_per_step[i],
                        dv_per_step[i],
                    ) = _fa3_varlen_backward(
                        q_part,
                        k_ag,
                        v_ag,
                        out_part,
                        dout_part,
                        softmax_lse_per_step[i],
                        ctx.softmax_scale,
                        ctx.causal,
                        window_size,
                        ctx.deterministic,
                        rumtime_meta_per_step,
                    )

            if i > 0:
                _collect_result_varlen(
                    dq, dq_per_step[i - 1], cu_seqlens_q_padded, i - 1
                )
                if i > 1:
                    flash_attn_streams[i - 1].wait_event(dkv_update_done)
                dk.add_(dk_per_step[i - 1])
                dv.add_(dv_per_step[i - 1])
                if i < local_seq_num:
                    flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.cuda.current_stream().wait_stream(ctx.cp_stream)

        dk = reorder_before_reduce_scatter(dk, ctx.total_scatter_indices)
        dv = reorder_before_reduce_scatter(dv, ctx.total_scatter_indices)

        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)

        return (
            dq,
            dk_part,
            dv_part,
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
        batch_p2p_comm=False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(cp_rank, cp_size, cp_group)

        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format

        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        qkv_dtype = q.dtype
        q_f16 = q
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)

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
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        for i in range(1, cp_size):
            p2p_comm_buffers[i] = torch.empty_like(p2p_comm_buffers[i - 1])
        send_recv_reqs = [[], []]  # type: ignore
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
                        # p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
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
                    _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
                        cu_seqlens_q_padded,
                        cu_seqlens_kv_padded,
                    )
                    if causal:
                        if i == 0:  # q, k, v
                            pass
                        elif i <= cp_rank:  # q, k0, v0
                            fused_attn_meta_kwargs["attn_mask_type"] = (
                                "padding" if padding else "no_mask"
                            )
                            is_half_kv = True
                            _max_seqlen_kv = max_seqlen_kv // 2
                            _cu_seqlens_kv_padded = _cu_seqlens_kv_padded // 2
                        else:  # q1, k, v
                            fused_attn_meta_kwargs["attn_mask_type"] = (
                                "padding" if padding else "no_mask"
                            )
                            is_half_q = True
                            _max_seqlen_q = max_seqlen_q // 2
                            _cu_seqlens_q_padded = _cu_seqlens_q_padded // 2
                    else:  # full
                        pass

                    chunk_idx_q = 1 if is_half_q else -1
                    q_inputs[i % 2], cu_seqlens_q_per_step[i] = prepare_input_fwd(
                        q,
                        chunk_idx_q,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        cp_size,
                        cp_rank,
                    )
                    chunk_idx_kv = 0 if is_half_kv else -1
                    kv_inputs[i % 2], cu_seqlens_kv_per_step[i] = prepare_input_fwd(
                        kv_inputs[i % 2],
                        chunk_idx_kv,
                        cu_seqlens_kv,
                        cu_seqlens_kv_padded,
                        cp_size,
                        (cp_rank - i) % cp_size,
                    )

                    out_per_step[i], aux_ctx_tensors = fused_attn_fwd(
                        True,
                        _max_seqlen_q,
                        _max_seqlen_kv,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        q_inputs[i % 2],
                        kv_inputs[i % 2][0],  # type: ignore[index]
                        kv_inputs[i % 2][1],  # type: ignore[index]
                        *fused_attn_meta_args,
                        **fused_attn_meta_kwargs,
                        cu_seqlens_q_padded=_cu_seqlens_q_padded,
                        cu_seqlens_kv_padded=_cu_seqlens_kv_padded,
                        **{},
                    )
                    softmax_lse_per_step[i], rng_states[i], *rest = aux_ctx_tensors
                    # softmax_lse_per_step[i] = softmax_lse_per_step[i].narrow(0, 0, q_inputs[i % 2].shape[0])
                    # [b, np, sq, 1] -> [b, np, sq]
                    # or [t, np, 1] -> [t, np]
                    softmax_lse_per_step[i].squeeze_(-1)  # type: ignore[attr-defined]
                    if softmax_lse_in_packed_format:
                        softmax_lse_per_step[i] = (
                            softmax_lse_per_step[i].transpose(0, 1).contiguous()  # type: ignore[attr-defined]
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
                    elif (i - 1) <= cp_rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        tex.thd_second_half_lse_correction(
                            softmax_lse,
                            softmax_lse_per_step[i - 1],
                            cu_seqlens_q_padded,
                            softmax_lse_in_packed_format,
                        )

                # if i < cp_size:
                flash_attn_streams[(i - 1) % 2].record_event(
                    fwd_results_correction_done
                )

        flash_attn_streams[cp_size % 2].wait_event(fwd_results_correction_done)
        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        second_half_lse_seqlen = None
        if causal and cp_rank < (cp_size - 1):
            second_half_lse_seqlen = softmax_lse_per_step[-1].shape[-1]  # type: ignore[attr-defined]

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            is_half = not (i <= cp_rank or not causal)
            tex.thd_out_correction(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                softmax_lse_in_packed_format,
            )

        kv = p2p_comm_buffers[-1]
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
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            softmax_lse_ = tex.thd_read_second_half_lse(
                softmax_lse,
                cu_seqlens_q_padded,
                ctx.softmax_lse_in_packed_format,
                ctx.second_half_lse_seqlen,
            )

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

            if cp_size > 1:
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
            _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
                cu_seqlens_q_padded,
                cu_seqlens_kv_padded,
            )
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
                    _cu_seqlens_kv_padded = _cu_seqlens_kv_padded // 2
                else:  # q1, k, v
                    assert softmax_lse_ is not None
                    aux_ctx_tensors = [softmax_lse_, rng_states[cp_size - i - 1]]
                    fused_attn_meta_args[2] = aux_ctx_tensors
                    fused_attn_meta_kwargs["attn_mask_type"] = (
                        "padding" if padding else "no_mask"
                    )
                    is_half_q = True
                    _max_seqlen_q = ctx.max_seqlen_q // 2
                    _cu_seqlens_q_padded = _cu_seqlens_q_padded // 2
            else:
                aux_ctx_tensors = [softmax_lse, rng_states[cp_size - i - 1]]
                fused_attn_meta_args[2] = aux_ctx_tensors

            chunk_idx_q = 1 if is_half_q else -1
            q_part, out_part, dout_part = prepare_input_bwd(
                [q, out, dout], chunk_idx_q, cu_seqlens_q_padded
            )
            if is_half_kv:
                kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
            else:
                kv_ = kv
            k_part, v_part = kv_[0], kv_[1]

            dq_, dk_, dv_, _ = fused_attn_bwd(
                _max_seqlen_q,
                _max_seqlen_kv,
                cu_seqlens_q_per_step[cp_size - i - 1],
                cu_seqlens_kv_per_step[cp_size - i - 1],
                q_part,
                k_part,
                v_part,
                out_part,
                dout_part,
                *fused_attn_meta_args,
                cu_seqlens_q_padded=_cu_seqlens_q_padded,
                cu_seqlens_kv_padded=_cu_seqlens_kv_padded,
                **fused_attn_meta_kwargs,
                **{},
            )

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

            dq = bwd_dq_update(dq, dq_, cu_seqlens_q_padded, first_op, second_op)

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()

            dkv = p2p_comm_buffers[(i + 1) % 2][1]
            dkv_ = torch.cat(
                (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
            )  # pylint: disable=used-before-assignment

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
            dkv = bwd_dkv_update(dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op)

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


class FA3RingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        runtime_meta,
        causal,
        dropout_p,
        softmax_scale,
        cp_group,
        cp_stream,
        deterministic,
        batch_p2p_comm=False,
    ) -> torch.Tensor:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        send_dst, recv_src = get_p2p_send_recv_rank(cp_rank, cp_size, cp_group)

        softmax_lse_in_packed_format = True
        qkv_dtype = q.dtype
        q_f16 = q

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        fa_forward_kwargs = {"window_size": (-1, -1)}

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
        p2p_comm_buffers[0] = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        for i in range(1, cp_size):
            p2p_comm_buffers[i] = torch.empty_like(p2p_comm_buffers[i - 1])
        send_recv_reqs = [[], []]  # type: ignore

        out = None
        for i in range(cp_size + 1):
            if i < cp_size:
                with torch.cuda.stream(flash_attn_streams[i % 2]):
                    # wait until KV is received
                    for req in send_recv_reqs[(i + 1) % 2]:
                        req.wait()

                    if i < (cp_size - 1):
                        # p2p_comm_buffers[i + 1] = torch.empty_like(p2p_comm_buffers[i])
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
                    if causal:
                        if i == 0:  # q, k, v
                            is_causal = True
                        elif i <= cp_rank:  # q, k0, v0
                            is_half_kv = True
                        else:  # q1, k, v
                            is_half_q = True
                    else:
                        pass

                    rumtime_meta_per_step = runtime_meta[i]
                    if is_half_q:
                        q_inputs[i % 2] = tex.thd_read_half_tensor(
                            q, cu_seqlens_q_padded, 1
                        )
                    else:
                        q_inputs[i % 2] = q
                    if is_half_kv:
                        kv_inputs[i % 2] = tex.thd_read_half_tensor(
                            kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                        )

                    out_per_step[i], softmax_lse_per_step[i] = _fa3_varlen_forward(
                        q_inputs[i % 2],
                        kv_inputs[i % 2][0],  # type: ignore[index]
                        kv_inputs[i % 2][1],  # type: ignore[index]
                        softmax_scale,
                        is_causal,
                        rumtime_meta_per_step,
                        fa_forward_kwargs,
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
                    elif (i - 1) <= cp_rank or not causal:
                        flash_attn_fwd_softmax_lse_correction(
                            softmax_lse, softmax_lse_per_step[i - 1]
                        )
                    else:
                        tex.thd_second_half_lse_correction(
                            softmax_lse,
                            softmax_lse_per_step[i - 1],
                            cu_seqlens_q_padded,
                            softmax_lse_in_packed_format,
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
            tex.thd_out_correction(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                softmax_lse_in_packed_format,
            )

        kv = p2p_comm_buffers[-1]
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
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.softmax_lse_in_packed_format = softmax_lse_in_packed_format
        ctx.second_half_lse_seqlen = second_half_lse_seqlen
        ctx.batch_p2p_comm = batch_p2p_comm
        ctx.deterministic = deterministic
        ctx.runtime_meta = runtime_meta

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

        causal = ctx.causal
        softmax_lse_ = None
        if causal and ctx.second_half_lse_seqlen is not None:
            softmax_lse_ = tex.thd_read_second_half_lse(
                softmax_lse,
                cu_seqlens_q_padded,
                ctx.softmax_lse_in_packed_format,
                ctx.second_half_lse_seqlen,
            )

        dq = torch.empty_like(q)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]
        p2p_comm_buffers[0][0].copy_(kv)
        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        send_recv_reqs = []

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

            if cp_size > 1:
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
            lse = softmax_lse
            if causal:
                if i == (cp_size - 1):  # q, k, v
                    is_causal = True
                elif i >= (cp_size - cp_rank - 1):  # q, k0, v0
                    is_half_kv = True
                else:  # q1, k, v
                    is_half_q = True
                    lse = softmax_lse_
            else:
                pass

            chunk_idx_q = 1 if is_half_q else -1
            q_, out_, dout_ = prepare_input_bwd(
                [q, out, dout], chunk_idx_q, cu_seqlens_q_padded
            )
            if is_half_kv:
                kv_ = tex.thd_read_half_tensor(kv, cu_seqlens_kv_padded, 0)
            else:
                kv_ = kv
            k_part, v_part = kv_[0], kv_[1]

            # if ctx.pad_between_seqs and ctx.qkv_format != "thd":
            #     q_, k_part, v_part = q_.contiguous(), k_part.contiguous(), v_part.contiguous()
            rumtime_meta_per_step = ctx.runtime_meta[cp_size - i - 1]
            window_size = (-1, 0) if is_causal else (-1, -1)
            dq_, dk_, dv_ = _fa3_varlen_backward(
                q_,
                k_part,
                v_part,
                out_,
                dout_,
                lse,
                ctx.softmax_scale,
                is_causal,
                window_size,
                ctx.deterministic,
                rumtime_meta_per_step,
            )

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
            dq = bwd_dq_update(dq, dq_, cu_seqlens_q_padded, first_op, second_op)

            # wait until dKV is received
            for req in send_recv_reqs:
                req.wait()
            dkv = p2p_comm_buffers[(i + 1) % 2][1]

            dkv_ = torch.cat(
                (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
            )  # pylint: disable=used-before-assignment

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
            dkv = bwd_dkv_update(dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op)

        # if ctx.qkv_format == "thd":
        #     dq[cu_seqlens_q_padded[-1] :].fill_(0)
        #     dkv[:, cu_seqlens_kv_padded[-1] :].fill_(0)
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
        )


class RingAttnAllGather(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        self.pg_p2p = cp_process_group[ParallelMode.RING]
        # pad factor for ulysess & ring
        self.pad_factor_p2p, self.pad_factor_a2a = get_pad_factor(
            cp_group_p2p=self.pg_p2p, cp_group_a2a=None
        )
        # NOTE: te padding_causal_bottom_right need max_seqlen_q % 64 == 0 and max_seqlen_kv % 64 == 0
        if backend == AttnBackend.TE:
            self.pad_factor_p2p *= 64
        self.backend = backend
        self.qkv_format = qkv_format
        self.shard_meta = {}  # type: ignore
        self.runtime_meta_per_step = []  # type: ignore
        self.gather_indices = None
        self.scatter_indices = None

    # to call after q,k,v dispatch
    def pre_compute_attn_runtime_meta(self, attn_mask_type: AttnMaskType, device):
        causal = attn_mask_type == AttnMaskType.CAUSAL
        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]

        cp_rank = dist.get_rank(group=self.pg_p2p)
        cp_size = dist.get_world_size(group=self.pg_p2p)
        if self.backend == AttnBackend.FA3:
            local_seq_chunk_idx = [cp_rank, 2 * cp_size - cp_rank - 1]
            for i in range(len(local_seq_chunk_idx)):
                cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                    shard_q_meta.cu_seqlens,
                    shard_q_meta.cu_seqlens_padded // cp_size,
                    cp_size,
                    cp_rank,
                    i == 0,
                    i == 1,
                )
                if causal:
                    cu_seqlens_kv_per_step = generate_cu_seqlens_kv_ag_causal(
                        shard_kv_meta.cu_seqlens,
                        shard_kv_meta.cu_seqlens_padded,
                        local_seq_chunk_idx[i],
                        cp_size,
                    )
                else:
                    cu_seqlens_kv_per_step = shard_kv_meta.cu_seqlens
                # TODO:
                host_cu_seqlens_q_per_step = cu_seqlens_q_per_step.tolist()
                host_cu_seqlens_kv_per_step = cu_seqlens_kv_per_step.tolist()
                rumtime_meta = generate_runtime_meta_per_step(
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    shard_q_meta.cu_seqlens_padded // (2 * cp_size),
                    shard_kv_meta.cu_seqlens_padded,
                    host_cu_seqlens_q_per_step,
                    host_cu_seqlens_kv_per_step,
                    shard_q_meta.host_cu_seqlens_padded[-1] // (2 * cp_size),
                    shard_kv_meta.host_cu_seqlens_padded[-1],
                    device,
                )
                self.runtime_meta_per_step.append(rumtime_meta)

        # generate total_indices for kv gather and scatter
        gather_indices_np = generate_zigzag_undispatch_indices(
            shard_kv_meta.host_cu_seqlens_padded,
            cp_size,
        )
        self.gather_indices = torch.from_numpy(gather_indices_np).to(
            device=device, dtype=torch.int64
        )
        self.scatter_indices_np = generate_scattar_reorder_indices(
            shard_kv_meta.host_cu_seqlens_padded, cp_size
        )
        self.scatter_indices = torch.from_numpy(self.scatter_indices_np).to(
            device=device, dtype=torch.int64
        )

    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str,  # key name for shard_meta
    ):
        # pre-process data
        x_global_varlen, origin_shape, cu_seqlens, host_cu_seqlens = _pre_process(
            x_global, ranges, valid_total_seqlen, self.qkv_format, x_global.device
        )
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            "thd",
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global_varlen,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )

        max_seqlen_padded = get_max_seqlen(host_cu_seqlens_padded)
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            origin_shape=origin_shape,
            max_seqlen_padded=max_seqlen_padded,
        )
        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global_varlen = zigzag_undispatch(
            x_local,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )
        x_global = unflatten_data_from_varlen(
            x_global_varlen, smeta.cu_seqlens, smeta.origin_shape, self.qkv_format
        )

        return x_global

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size = dist.get_world_size(group=self.pg_p2p)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]

        if self.backend == AttnBackend.TE:
            if attn_mask_type == AttnMaskType.CAUSAL:
                attn_mask = "padding_causal_bottom_right"
            elif attn_mask_type == AttnMaskType.FULL:
                attn_mask = "padding"

            out_layer, lse = TERingAGAttnFunc.apply(
                q,
                k,
                v,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen_padded // cp_size,
                shard_kv_meta.max_seqlen_padded,
                shard_q_meta.cu_seqlens_padded // cp_size,
                shard_kv_meta.cu_seqlens_padded,
                self.gather_indices,
                self.scatter_indices,
                dropout_p,
                softmax_scale,
                "thd",
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
            )
        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out_layer, lse = FA3RingAGAttnFunc.apply(
                q,
                k,
                v,
                shard_q_meta.cu_seqlens_padded // cp_size,
                self.gather_indices,
                self.scatter_indices,
                self.runtime_meta_per_step,
                is_causal,
                dropout_p,
                softmax_scale,
                self.pg_p2p,
                cp_stream,
                deterministic,
            )

        return out_layer, lse


class RingAttnP2P(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        self.pg_p2p = cp_process_group[ParallelMode.RING]
        # pad factor for ulysess & ring
        self.pad_factor_p2p, self.pad_factor_a2a = get_pad_factor(
            cp_group_p2p=self.pg_p2p, cp_group_a2a=None
        )
        self.backend = backend
        self.qkv_format = qkv_format
        self.shard_meta = {}  # type: ignore
        self.runtime_meta_per_step = []  # type: ignore

    # to call after q,k,v dispatch
    def pre_compute_attn_runtime_meta(self, attn_mask_type: AttnMaskType, device):
        if self.backend == AttnBackend.FA3:
            causal = attn_mask_type == AttnMaskType.CAUSAL
            shard_q_meta = self.shard_meta["q"]
            shard_kv_meta = self.shard_meta["k"]

            cp_rank = dist.get_rank(group=self.pg_p2p)
            cp_size = dist.get_world_size(group=self.pg_p2p)
            self.runtime_meta_per_step = [None for i in range(cp_size)]

            for i in range(cp_size):
                first_idx_q, second_idx_q, first_idx_kv, second_idx_kv = (
                    True,
                    True,
                    True,
                    True,
                )
                factor_q, factor_kv = cp_size, cp_size
                if causal:
                    if i == 0:  # q, k, v
                        pass
                    elif i <= cp_rank:  # q, k0, v0
                        second_idx_kv = False
                        factor_kv *= 2
                    else:  # q1, k, v
                        first_idx_q = False
                        factor_q *= 2
                else:  # full
                    pass

                cu_seqlens_q_per_step = get_cu_seqlens_on_cp_rank(
                    shard_q_meta.cu_seqlens,
                    shard_q_meta.cu_seqlens_padded // cp_size,
                    cp_size,
                    cp_rank,
                    first_idx_q,
                    second_idx_q,
                )
                cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                    shard_kv_meta.cu_seqlens,
                    shard_kv_meta.cu_seqlens_padded // cp_size,
                    cp_size,
                    (cp_rank - i) % cp_size,
                    first_idx_kv,
                    second_idx_kv,
                )
                host_cu_seqlens_q_per_step = cu_seqlens_q_per_step.tolist()
                host_cu_seqlens_kv_per_step = cu_seqlens_kv_per_step.tolist()
                rumtime_meta = generate_runtime_meta_per_step(
                    cu_seqlens_q_per_step,
                    cu_seqlens_kv_per_step,
                    shard_q_meta.cu_seqlens_padded // factor_q,
                    shard_kv_meta.cu_seqlens_padded // factor_kv,
                    host_cu_seqlens_q_per_step,
                    host_cu_seqlens_kv_per_step,
                    shard_q_meta.host_cu_seqlens_padded[-1] // factor_q,
                    shard_kv_meta.host_cu_seqlens_padded[-1] // factor_kv,
                    device,
                )
                self.runtime_meta_per_step[i] = rumtime_meta

    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str,  # key name for shard_meta
    ):
        # pre-process data
        x_global_varlen, origin_shape, cu_seqlens, host_cu_seqlens = _pre_process(
            x_global, ranges, valid_total_seqlen, self.qkv_format, x_global.device
        )
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            "thd",
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global_varlen,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )

        max_seqlen_padded = get_max_seqlen(host_cu_seqlens_padded)
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            origin_shape=origin_shape,
            max_seqlen_padded=max_seqlen_padded,
        )
        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global_varlen = zigzag_undispatch(
            x_local,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )
        x_global = unflatten_data_from_varlen(
            x_global_varlen, smeta.cu_seqlens, smeta.origin_shape, self.qkv_format
        )

        return x_global

    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size = dist.get_world_size(group=self.pg_p2p)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]

        if self.backend == AttnBackend.TE:
            if attn_mask_type == AttnMaskType.CAUSAL:
                attn_mask = "padding_causal"
            elif attn_mask_type == AttnMaskType.FULL:
                attn_mask = "padding"

            out_layer, lse = TERingAttnFunc.apply(
                q,
                k,
                v,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen_padded // cp_size,
                shard_kv_meta.max_seqlen_padded // cp_size,
                shard_q_meta.cu_seqlens_padded // cp_size,
                shard_kv_meta.cu_seqlens_padded // cp_size,
                dropout_p,
                softmax_scale,
                "thd",
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
            )

        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out_layer, lse = FA3RingAttnFunc.apply(
                q,
                k,
                v,
                shard_q_meta.cu_seqlens_padded // cp_size,
                shard_kv_meta.cu_seqlens_padded // cp_size,
                self.runtime_meta_per_step,
                is_causal,
                dropout_p,
                softmax_scale,
                self.pg_p2p,
                cp_stream,
                deterministic,
            )

        return out_layer, lse
