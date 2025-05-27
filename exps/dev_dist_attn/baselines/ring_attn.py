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

import torch
import torch.distributed as dist
import transformer_engine as te  # noqa
import transformer_engine_torch as tex

# te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import reduce_scatter_along_first_dim

from magi_attention.comm.functional import all_gather_fwd_scatter_bwd
from magi_attention.common.enum import AttnMaskType

from .attn import (
    AttnBackend,
    FA3RingAttnFunc,
    TERingAttnFunc,
    _fa3_attn_backward,
    _fa3_attn_forward,
)
from .interface import AttnBaselineInterface
from .shard import (
    ParallelMode,
    ShardMeta,
    _zigzag_undispatch_non_varlen,
    generate_reorder_chunk_ids_zigzag,
    generate_zigzag_dispatch_indices,
    generate_zigzag_undispatch_indices,
    get_cu_seqlens_padded,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from .utils_cp import (
    divide_lst,
    generate_cu_seqlens_kv_ag_causal,
    generate_cu_seqlens_kv_ag_causal_host,
    prepare_q_bwd,
    prepare_q_fwd,
)


def gather_with_reorder_before_attn(
    input: torch.Tensor,
    cu_seqlens_host: List[int],
    cu_seqlens_padded_host: List[int],
    qkv_format,
    cp_size,
    cp_group,
):
    device = input.device
    if qkv_format == "thd":
        other_shape = input.shape[1:]
        out = all_gather_fwd_scatter_bwd(input, cp_group, dim=0).contiguous()
        total_indices = generate_zigzag_undispatch_indices(
            cu_seqlens_padded_host, cp_size, device, cu_seqlens_host
        )
        output = torch.gather(
            out, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
        )

    else:
        output = _zigzag_undispatch_non_varlen(input, qkv_format, cp_size, cp_group)
    return output


def reorder_before_reduce_scatter(
    input,
    cu_seqlens_padded_host: List[int],
    cu_seqlens_padded: torch.Tensor,
    qkv_format,
    cp_size,
):
    device = input.device
    if qkv_format == "thd":
        batch_size = len(cu_seqlens_padded) - 1
        other_shape = input.shape[1:]
        indices_lst = []
        for cp_rank in range(cp_size):
            zigzag_indices, _ = generate_zigzag_dispatch_indices(
                cu_seqlens_padded_host,
                cu_seqlens_padded_host,
                cu_seqlens_padded[:batch_size],
                cu_seqlens_padded[:batch_size],
                cp_size,
                cp_rank,
                device,
            )
            indices_lst.append(zigzag_indices)
        total_indices = torch.cat(indices_lst, dim=0)
        output = torch.gather(
            input, dim=0, index=total_indices[:, None, None].expand(-1, *other_shape)
        )
    else:
        reorder_index = generate_reorder_chunk_ids_zigzag(cp_size, device)
        seq_dim = qkv_format.index("s")
        other_shape = input.shape[2:]
        out = input.view(
            *input.shape[:seq_dim], 2 * cp_size, -1, *input.shape[seq_dim + 1 :]
        )
        output = torch.index_select(out, seq_dim, reorder_index)
        if qkv_format == "bshd":
            output = output.view(*input.shape[:seq_dim], -1, *other_shape)
        elif qkv_format == "sbhd":
            # output = output.view(input.shape[0], -1, *other_shape)
            output = output.view(-1, *output.shape[-3:])
    return output


# use te tex.thd_grad_correction for varlen result collection
def _collect_result_varlen(
    out: torch.Tensor, out_: torch.Tensor, cu_seqlens_padded: torch.Tensor, chunk_idx
):
    if chunk_idx == 0:
        first_op, second_op = "copy", "none"
    elif chunk_idx == 1:
        first_op, second_op = "none", "copy"
    tex.thd_grad_correction(out, out_, cu_seqlens_padded, first_op, second_op)


# [h,t] same as fa3
def _collect_lse_result_varlen(
    softmax_lse: torch.Tensor,
    lse_per_step: torch.Tensor,
    cu_seqlens_padded_q_host,
    chunk_idx,
):
    device = softmax_lse.device
    batch_size = len(cu_seqlens_padded_q_host) - 1
    indices_lst = []
    for i in range(batch_size):
        seqlen = cu_seqlens_padded_q_host[i + 1] - cu_seqlens_padded_q_host[i]
        st = cu_seqlens_padded_q_host[i] + chunk_idx * (seqlen // 2)
        ed = st + seqlen // 2
        indices = torch.arange(start=st, end=ed, dtype=torch.int64, device=device)
        indices_lst.append(indices)
    total_indices = torch.cat(indices_lst, dim=0)
    softmax_lse.scatter_(
        1, total_indices[None, :].expand(softmax_lse.shape[0], -1), lse_per_step
    )


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
        dropout_p,
        softmax_scale,
        qkv_format,
        cp_group,
        attn_mask_type,
        cp_stream,
        deterministic,
        pad_between_seqs,
        host_meta=[None, None, None, None],
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)

        causal = "causal" in attn_mask_type
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
        # List[int] total cu_seqlens_padded
        cu_seqlens_kv_padded_host = host_meta[3]

        k_ag = gather_with_reorder_before_attn(
            k, None, cu_seqlens_kv_padded_host, qkv_format, cp_size, cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, None, cu_seqlens_kv_padded_host, qkv_format, cp_size, cp_group
        )
        cp_stream.wait_stream(torch.cuda.current_stream())

        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        local_seq_chunk_idx = [cp_rank, 2 * cp_size - cp_rank - 1]
        local_seq_num = 2
        # window_size_per_step = [None, None]
        cu_seqlens_q_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        # max_seqlen_q_per_step = [None, None]
        # max_seqlen_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        rng_states = [None, None]

        softmax_lse = None
        seq_dim = 0 if qkv_format != "bshd" else 1
        # b,s,h,d -> b,2,s/2,h,d or
        # s,b,h,d -> 2,s/2,b,h,d
        if qkv_format != "thd":
            q = q.view(
                *q.shape[:seq_dim], 2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :]
            )
        else:  # thd lse [h,t]
            softmax_lse = torch.empty(
                *(q.shape[1], q.shape[0]), dtype=torch.float, device=q.device
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
                    cu_seqlens_q_per_step[i], q_part, _ = prepare_q_fwd(
                        q,
                        i,
                        qkv_format,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        pad_between_seqs,
                        cp_size,
                        cp_rank,
                    )
                    q_part = q_part.contiguous()
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
                    if qkv_format == "bshd":
                        out[:, i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "sbhd":
                        out[i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "thd":
                        _collect_result_varlen(
                            out, out_per_step[i - 1], cu_seqlens_q_padded, i - 1
                        )
                        lse_per_step = softmax_lse_per_step[i - 1].squeeze(-1)
                        _collect_lse_result_varlen(
                            softmax_lse,
                            lse_per_step.transpose(0, 1).contiguous(),
                            host_meta[1],
                            i - 1,
                        )

        # softmax_lse
        if qkv_format != "thd":
            softmax_lse = torch.cat(
                [
                    softmax_lse_per_step[0].squeeze(-1),
                    softmax_lse_per_step[1].squeeze(-1),
                ],
                dim=-1,
            )
        # else:   # [t,h] -> [h,t]
        #     softmax_lse = softmax_lse.transpose(0, 1).contiguous()

        torch.cuda.current_stream().wait_stream(cp_stream)

        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])

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
        ctx.cu_seqlens_kv_padded_host = cu_seqlens_kv_padded_host

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
        if ctx.qkv_format != "bshd":
            dk = torch.zeros(
                (k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device
            )
        else:
            dk = torch.zeros(
                (k.shape[0], k.shape[1] * cp_size, *k.shape[2:]),
                dtype=k.dtype,
                device=k.device,
            )
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        cu_seqlens_kv_padded_host = ctx.cu_seqlens_kv_padded_host
        k_ag = gather_with_reorder_before_attn(
            k, None, cu_seqlens_kv_padded_host, ctx.qkv_format, cp_size, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, None, cu_seqlens_kv_padded_host, ctx.qkv_format, cp_size, ctx.cp_group
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

        if ctx.qkv_format != "thd":
            dout = dout.view(*q.shape)

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    out_part = out_per_step[i]
                    q_part, dout_part = prepare_q_bwd(
                        [q, dout], i, cu_seqlens_q_padded, ctx.qkv_format
                    )
                    q_part, dout_part = q_part.contiguous(), dout_part.contiguous()
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
                    if ctx.qkv_format == "bshd":
                        dq[:, i - 1].copy_(dq_per_step[i - 1])
                    elif ctx.qkv_format == "sbhd":
                        dq[i - 1].copy_(dq_per_step[i - 1])
                    else:
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

        dk = reorder_before_reduce_scatter(
            dk, cu_seqlens_kv_padded_host, cu_seqlens_kv_padded, ctx.qkv_format, cp_size
        )
        dv = reorder_before_reduce_scatter(
            dv, cu_seqlens_kv_padded_host, cu_seqlens_kv_padded, ctx.qkv_format, cp_size
        )

        if ctx.qkv_format == "bshd":
            dk = dk.transpose(0, 1).contiguous()
            dv = dv.transpose(0, 1).contiguous()
        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)
        if ctx.qkv_format == "bshd":
            dk_part = dk_part.transpose(0, 1).contiguous()
            dv_part = dv_part.transpose(0, 1).contiguous()

        if ctx.qkv_format == "bshd":
            dq = dq.view(dq.shape[0], -1, *dq.shape[3:])
        elif ctx.qkv_format == "sbhd":
            dq = dq.view(-1, *dq.shape[2:])

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


# NOTE: this may cause deadlock in fa bwd
def assert_no_consecutive_duplicates(cu_seqlens):
    same_as_next = cu_seqlens[:-1] == cu_seqlens[1:]
    assert (
        not same_as_next.any()
    ), "Temporarily does not support sequence of length 0 in fa3 ring all-gather"


class FA3RingAGAttnFunc(torch.autograd.Function):
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
        host_meta=[None, None, None, None],
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        cp_size = torch.distributed.get_world_size(group=cp_group)
        cp_rank = torch.distributed.get_rank(group=cp_group)

        qkv_dtype = q.dtype
        fa_forward_kwargs = {"softmax_scale": softmax_scale}
        fa_forward_kwargs["window_size"] = (-1, 0) if causal else (-1, -1)

        # List[int] total cu_seqlens_padded
        k_ag = gather_with_reorder_before_attn(
            k, None, host_meta[3], qkv_format, cp_size, cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, None, host_meta[3], qkv_format, cp_size, cp_group
        )
        cp_stream.wait_stream(torch.cuda.current_stream())
        dist.barrier()
        torch.cuda.synchronize()

        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        local_seq_chunk_idx = [cp_rank, 2 * cp_size - cp_rank - 1]
        local_seq_num = 2
        # window_size_per_step = [None, None]
        cu_seqlens_q_per_step = [None, None]
        cu_seqlens_kv_per_step = [None, None]
        # max_seqlen_q_per_step = [None, None]
        # max_seqlen_kv_per_step = [None, None]
        out_per_step = [None, None]
        softmax_lse_per_step = [None, None]
        host_meta_per_step = [None, None]

        softmax_lse = None
        seq_dim = 0 if qkv_format != "bshd" else 1
        # b,s,h,d -> b,2,s/2,h,d or
        # s,b,h,d -> 2,s/2,b,h,d
        if qkv_format != "thd":
            q = q.view(
                *q.shape[:seq_dim], 2, q.shape[seq_dim] // 2, *q.shape[(seq_dim + 1) :]
            )
        else:  # thd lse [h,t]
            softmax_lse = torch.empty(
                *(q.shape[1], q.shape[0]), dtype=torch.float, device=q.device
            )
        out = torch.empty_like(q)

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    cu_seqlens_q_per_step[i], q_part, host_meta_q = prepare_q_fwd(
                        q,
                        i,
                        qkv_format,
                        cu_seqlens_q,
                        cu_seqlens_q_padded,
                        pad_between_seqs,
                        cp_size,
                        cp_rank,
                        host_meta[:2],
                    )
                    # assert_no_consecutive_duplicates(cu_seqlens_q_per_step[i])

                    host_meta_fwd = [
                        host_meta_q[0],
                        host_meta_q[1],
                        host_meta[2],
                        host_meta[3],
                    ]
                    q_part = q_part.contiguous()
                    if causal:
                        cu_seqlens_kv_per_step[i] = generate_cu_seqlens_kv_ag_causal(
                            cu_seqlens_kv,
                            cu_seqlens_kv_padded,
                            local_seq_chunk_idx[i],
                            cp_size,
                        )
                        host_meta_fwd[2] = generate_cu_seqlens_kv_ag_causal_host(
                            host_meta[2], host_meta[3], local_seq_chunk_idx[i], cp_size
                        )
                    else:
                        cu_seqlens_kv_per_step[i] = cu_seqlens_kv
                    host_meta_per_step[i] = host_meta_fwd

                    out_per_step[i], softmax_lse_per_step[i] = _fa3_attn_forward(
                        q_part,
                        k_ag,
                        v_ag,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        # get_max_seqlen(host_meta_fwd[0]),
                        # get_max_seqlen(host_meta_fwd[2]),
                        # cu_seqlens_q_padded // 2,
                        # cu_seqlens_kv_padded,
                        causal,
                        qkv_format,
                        pad_between_seqs,
                        fa_forward_kwargs,
                        host_meta_fwd,
                    )

            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if qkv_format == "bshd":
                        out[:, i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "sbhd":
                        out[i - 1].copy_(out_per_step[i - 1])
                    elif qkv_format == "thd":
                        _collect_result_varlen(
                            out, out_per_step[i - 1], cu_seqlens_q_padded, i - 1
                        )
                        _collect_lse_result_varlen(
                            softmax_lse,
                            softmax_lse_per_step[i - 1],
                            host_meta[1],
                            i - 1,
                        )

        # softmax_lse
        if qkv_format != "thd":
            softmax_lse = torch.cat(softmax_lse_per_step, dim=-1)

        torch.cuda.current_stream().wait_stream(cp_stream)

        dist.barrier()
        torch.cuda.synchronize()

        if qkv_format == "bshd":
            out = out.view(out.shape[0], -1, *out.shape[-2:])
        elif qkv_format == "sbhd":
            out = out.view(-1, *out.shape[-3:])

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
        )
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.deterministic = deterministic
        ctx.causal = causal
        ctx.pad_between_seqs = pad_between_seqs
        ctx.host_meta_per_step = host_meta_per_step

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        cp_size = torch.distributed.get_world_size(group=ctx.cp_group)
        dist.barrier()
        torch.cuda.synchronize()

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, cu_seqlens_q_padded, cu_seqlens_kv_padded) = saved_tensors[:5]
        cu_seqlens_q_per_step = saved_tensors[5:7]
        cu_seqlens_kv_per_step = saved_tensors[7:9]
        out_per_step = saved_tensors[9:11]
        softmax_lse_per_step = saved_tensors[11:13]

        dout = dout.view(q.shape)
        dq = torch.empty_like(q)
        if ctx.qkv_format != "bshd":
            dk = torch.zeros(
                (k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device
            )
        else:
            dk = torch.zeros(
                (k.shape[0], k.shape[1] * cp_size, *k.shape[2:]),
                dtype=k.dtype,
                device=k.device,
            )
        dv = torch.zeros_like(dk)
        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]

        host_meta_per_step = ctx.host_meta_per_step
        host_cu_padded_kv = host_meta_per_step[0][3]

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.cuda.Event()

        k_ag = gather_with_reorder_before_attn(
            k, None, host_cu_padded_kv, ctx.qkv_format, cp_size, ctx.cp_group
        )
        v_ag = gather_with_reorder_before_attn(
            v, None, host_cu_padded_kv, ctx.qkv_format, cp_size, ctx.cp_group
        )

        ctx.cp_stream.wait_stream(torch.cuda.current_stream())

        fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
        fa_backward_kwargs["deterministic"] = ctx.deterministic
        local_seq_num = 2

        if ctx.qkv_format != "thd":
            dout = dout.view(*q.shape)

        for i in range(local_seq_num + 1):
            if i < local_seq_num:
                with torch.cuda.stream(flash_attn_streams[i]):
                    out_part = out_per_step[i]
                    q_part, dout_part = prepare_q_bwd(
                        [q, dout], i, cu_seqlens_q_padded, ctx.qkv_format
                    )
                    q_part, dout_part = q_part.contiguous(), dout_part.contiguous()

                    # max_lenq = get_max_seqlen(host_meta_per_step[i][0])
                    # max_lenk = get_max_seqlen(host_meta_per_step[i][2])
                    dq_per_step[i], dk_per_step[i], dv_per_step[i] = _fa3_attn_backward(
                        q_part,
                        k_ag,
                        v_ag,
                        out_part,
                        dout_part,
                        cu_seqlens_q_per_step[i],
                        cu_seqlens_kv_per_step[i],
                        # max_lenq,
                        # max_lenk,
                        # cu_seqlens_q_padded // 2,
                        # cu_seqlens_kv_padded,
                        softmax_lse_per_step[i],
                        ctx.causal,
                        ctx.qkv_format,
                        ctx.pad_between_seqs,
                        fa_backward_kwargs,
                        host_meta_per_step[i],
                    )
            if i > 0:
                with torch.cuda.stream(flash_attn_streams[i - 1]):
                    if ctx.qkv_format == "bshd":
                        dq[:, i - 1].copy_(dq_per_step[i - 1])
                    elif ctx.qkv_format == "sbhd":
                        dq[i - 1].copy_(dq_per_step[i - 1])
                    else:
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

        dk = reorder_before_reduce_scatter(
            dk, host_cu_padded_kv, cu_seqlens_kv_padded, ctx.qkv_format, cp_size
        )
        dv = reorder_before_reduce_scatter(
            dv, host_cu_padded_kv, cu_seqlens_kv_padded, ctx.qkv_format, cp_size
        )

        if ctx.qkv_format == "bshd":
            dk = dk.transpose(0, 1).contiguous()
            dv = dv.transpose(0, 1).contiguous()
        dk_part, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
        dv_part, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)
        if ctx.qkv_format == "bshd":
            dk_part = dk_part.transpose(0, 1).contiguous()
            dv_part = dv_part.transpose(0, 1).contiguous()

        if ctx.qkv_format == "bshd":
            dq = dq.view(dq.shape[0], -1, *dq.shape[3:])
        elif ctx.qkv_format == "sbhd":
            dq = dq.view(-1, *dq.shape[2:])

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
        )


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

    def dispatch(
        self,
        x_global: torch.Tensor,
        cu_seqlens: torch.Tensor,
        host_cu_seqlens: List[int],
        name: str,  # key name for shard_meta
        **kwargs,
    ):
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            self.qkv_format,
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global,
            cu_seqlens,
            cu_seqlens_padded,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            self.qkv_format,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )
        max_seqlen = max(
            [
                (host_cu_seqlens[i + 1] - host_cu_seqlens[i])
                for i in range(len(host_cu_seqlens) - 1)
            ]
        )
        max_seqlen_padded = max(
            [
                (host_cu_seqlens_padded[i + 1] - host_cu_seqlens_padded[i])
                for i in range(len(host_cu_seqlens_padded) - 1)
            ]
        )
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            restore_shape=restore_shape,
            max_seqlen=max_seqlen,
            max_seqlen_padded=max_seqlen_padded,
        )
        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
        **kwargs,
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global = zigzag_undispatch(
            x_local,
            smeta.cu_seqlens,
            smeta.cu_seqlens_padded,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            self.qkv_format,
            smeta.restore_shape,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size = dist.get_world_size(group=self.pg_p2p)
        batch_p2p_comm = kwargs.get("batch_p2p_comm", True)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]
        pad_between_seqs = not (
            shard_q_meta.host_cu_seqlens[-1] == shard_q_meta.host_cu_seqlens_padded[-1]
        )

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
                self.qkv_format,
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
                pad_between_seqs,
                batch_p2p_comm,
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
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.cu_seqlens_padded // cp_size,
                shard_kv_meta.cu_seqlens_padded // cp_size,
                is_causal,
                dropout_p,
                softmax_scale,
                self.qkv_format,
                self.pg_p2p,
                cp_stream,
                deterministic,
                pad_between_seqs,
                batch_p2p_comm,
                [
                    shard_q_meta.host_cu_seqlens,
                    divide_lst(shard_q_meta.host_cu_seqlens_padded, cp_size),
                    shard_kv_meta.host_cu_seqlens,
                    divide_lst(shard_kv_meta.host_cu_seqlens_padded, cp_size),
                ],
            )

        return out_layer, lse


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

    def dispatch(
        self,
        x_global: torch.Tensor,
        cu_seqlens: torch.Tensor,
        host_cu_seqlens: List[int],
        name: str,  # key name for shard_meta
        **kwargs,
    ):
        # compute cu_seqlens_padded and host_cu_seqlens_padded
        cu_seqlens_padded, host_cu_seqlens_padded = get_cu_seqlens_padded(
            cu_seqlens,
            host_cu_seqlens,
            self.qkv_format,
            pad_factor_p2p=self.pad_factor_p2p,
            pad_factor_a2a=self.pad_factor_a2a,
        )

        x_local, restore_shape = zigzag_dispatch(
            x_global,
            cu_seqlens,
            cu_seqlens_padded,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            self.qkv_format,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
        )
        max_seqlen = max(
            [
                (host_cu_seqlens[i + 1] - host_cu_seqlens[i])
                for i in range(len(host_cu_seqlens) - 1)
            ]
        )
        max_seqlen_padded = max(
            [
                (host_cu_seqlens_padded[i + 1] - host_cu_seqlens_padded[i])
                for i in range(len(host_cu_seqlens_padded) - 1)
            ]
        )
        self.shard_meta[name] = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            restore_shape=restore_shape,
            max_seqlen=max_seqlen,
            max_seqlen_padded=max_seqlen_padded,
        )
        return x_local

    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
        **kwargs,
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global = zigzag_undispatch(
            x_local,
            smeta.cu_seqlens,
            smeta.cu_seqlens_padded,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            self.qkv_format,
            smeta.restore_shape,
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=None,
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size = dist.get_world_size(group=self.pg_p2p)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]
        pad_between_seqs = not (
            shard_q_meta.host_cu_seqlens[-1] == shard_q_meta.host_cu_seqlens_padded[-1]
        )

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
                dropout_p,
                softmax_scale,
                self.qkv_format,
                self.pg_p2p,
                attn_mask,
                cp_stream,
                deterministic,
                pad_between_seqs,
                [
                    None,
                    divide_lst(shard_q_meta.host_cu_seqlens_padded, cp_size),
                    None,
                    shard_kv_meta.host_cu_seqlens_padded,
                ],
            )
        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
                pad_between_seqs = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out_layer, lse = FA3RingAGAttnFunc.apply(
                q,
                k,
                v,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.cu_seqlens_padded // cp_size,
                shard_kv_meta.cu_seqlens_padded,
                is_causal,
                dropout_p,
                softmax_scale,
                self.qkv_format,
                self.pg_p2p,
                cp_stream,
                deterministic,
                pad_between_seqs,
                [
                    shard_q_meta.host_cu_seqlens,
                    divide_lst(shard_q_meta.host_cu_seqlens_padded, cp_size),
                    shard_kv_meta.host_cu_seqlens,
                    shard_kv_meta.host_cu_seqlens_padded,
                ],
            )

        return out_layer, lse
