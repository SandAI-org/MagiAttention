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

from typing import Dict

import torch
import torch.distributed as dist

# te
import transformer_engine as te  # noqa
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.utils import get_cudnn_version

import transformer_engine_torch as tex
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

from .interface import AttnBaselineInterface
from .ring_attn import prepare_input_bwd, prepare_input_fwd
from .shard import (
    ParallelMode,
    ShardMeta,
    get_cu_seqlens_padded,
    get_max_seqlen,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from .utils_cp import (
    AttnBackend,
    _fa3_varlen_backward,
    _fa3_varlen_forward,
    _pre_process,
    _SeqAllToAll,
    attn_p2p_communicate,
    bwd_dkv_update,
    bwd_dq_update,
    flash_attn_fwd_softmax_lse_correction,
    generate_runtime_meta_per_step,
    get_cu_seqlens_on_cp_rank,
    get_p2p_send_recv_rank,
    prepare_for_saving,
    restore_from_saved,
    unflatten_data_from_varlen,
)


class FA3DoubleRingAttnFunc(torch.autograd.Function):
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
        cp_groups,
        cp_stream,
        deterministic,
        batch_p2p_comm=True,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        inter_p2p_pg = cp_groups[ParallelMode.INTER_WINDOW]
        intra_p2p_pg = cp_groups[ParallelMode.INTRA_WINDOW]
        inter_cp_size = torch.distributed.get_world_size(group=inter_p2p_pg)
        intra_cp_size = torch.distributed.get_world_size(group=intra_p2p_pg)
        inter_cp_rank = torch.distributed.get_rank(group=inter_p2p_pg)
        intra_cp_rank = torch.distributed.get_rank(group=intra_p2p_pg)

        inter_send_dst, inter_recv_src = get_p2p_send_recv_rank(
            inter_cp_rank, inter_cp_size, inter_p2p_pg
        )
        intra_send_dst, intra_recv_src = get_p2p_send_recv_rank(
            intra_cp_rank, intra_cp_size, intra_p2p_pg
        )

        cp_size = torch.distributed.get_world_size(group=cp_groups[ParallelMode.RING])
        window_num = cp_size // intra_cp_size

        qkv_dtype = q.dtype
        q_f16, k_f16, v_f16 = q, k, v
        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = True
        q_inputs = [None, None]
        intra_kv_inputs = [None, None]
        inter_kv_inputs = [None, None]
        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        is_half_q_per_step = [None for _ in range(cp_size)]

        local_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        intra_p2p_comm_buffers = [None for _ in range(intra_cp_size)]
        inter_kv_inputs[0] = local_kv
        inter_kv_inputs[1] = torch.empty_like(local_kv)
        intra_send_recv_reqs = [[], []]  # type: ignore
        inter_send_recv_reqs = []  # type: ignore

        second_half_lse_seqlen = None
        out = None
        fa_forward_kwargs = {"window_size": (-1, -1)}

        # outer ring loop
        for window_idx in range(window_num):
            if window_idx > 0:
                for req in inter_send_recv_reqs:
                    req.wait()

            if window_idx + 1 < window_num:
                inter_send_recv_reqs = attn_p2p_communicate(
                    inter_cp_rank,
                    inter_kv_inputs[window_idx % 2],
                    inter_send_dst,
                    inter_kv_inputs[(window_idx + 1) % 2],
                    inter_recv_src,
                    inter_p2p_pg,
                    batch_p2p_comm,
                )
            local_kv = inter_kv_inputs[window_idx % 2]
            window_offset = ((inter_cp_rank - window_idx) % window_num) * intra_cp_size

            # inner ring loop
            intra_p2p_comm_buffers[0] = local_kv
            for i in range(intra_cp_size + 1):
                if i < intra_cp_size:
                    with torch.cuda.stream(flash_attn_streams[i % 2]):
                        # wait until KV is received
                        for req in intra_send_recv_reqs[(i + 1) % 2]:
                            req.wait()

                        if i < (intra_cp_size - 1):
                            intra_p2p_comm_buffers[i + 1] = torch.empty_like(
                                intra_p2p_comm_buffers[i]
                            )
                            intra_send_recv_reqs[i % 2] = attn_p2p_communicate(
                                intra_cp_rank,
                                intra_p2p_comm_buffers[i],
                                intra_send_dst,
                                intra_p2p_comm_buffers[i + 1],
                                intra_recv_src,
                                intra_p2p_pg,
                                batch_p2p_comm,
                            )
                        # contiguous tensor
                        intra_kv_inputs[i % 2] = intra_p2p_comm_buffers[i]
                        is_half_q, is_half_kv, is_causal = False, False, False
                        if causal:
                            if i == 0 and window_idx == 0:  # q, k, v
                                is_causal = True
                            elif (window_idx == 0 and i <= intra_cp_rank) or (
                                0 < window_idx <= inter_cp_rank
                            ):  # q, k0, v0
                                is_half_kv = True
                            else:
                                is_half_q = True
                        else:
                            pass

                        rumtime_meta_per_step = runtime_meta[window_offset + i]
                        if is_half_q:
                            q_inputs[i % 2] = tex.thd_read_half_tensor(
                                q, cu_seqlens_q_padded, 1
                            )
                        else:
                            q_inputs[i % 2] = q
                        if is_half_kv:
                            intra_kv_inputs[i % 2] = tex.thd_read_half_tensor(
                                intra_kv_inputs[i % 2], cu_seqlens_kv_padded, 0
                            )

                        is_half_q_per_step[window_offset + i] = is_half_q

                        (
                            out_per_step[window_offset + i],
                            softmax_lse_per_step[window_offset + i],
                        ) = _fa3_varlen_forward(
                            q_inputs[i % 2],
                            intra_kv_inputs[i % 2][0],
                            intra_kv_inputs[i % 2][1],
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

                    if is_half_q:
                        second_half_lse_seqlen = softmax_lse_per_step[
                            window_offset + i - 1
                        ].shape[-1]
                    with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                        if window_idx == 0 and i == 1:
                            out = torch.zeros_like(q)
                            softmax_lse = torch.clone(
                                softmax_lse_per_step[window_offset]
                            ).to(torch.double)
                        elif (
                            (window_idx == 0 and (i - 1) <= intra_cp_rank)
                            or (0 < window_idx <= inter_cp_rank)
                            or not causal
                        ):  # q, k0, v0
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse, softmax_lse_per_step[window_offset + i - 1]
                            )
                        else:
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[window_offset + i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                    if i < intra_cp_size:
                        flash_attn_streams[(i - 1) % 2].record_event(
                            fwd_results_correction_done
                        )

            intra_p2p_comm_buffers[:] = [None] * len(intra_p2p_comm_buffers)
            intra_send_recv_reqs = [[], []]
            torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            is_half = is_half_q_per_step[i]
            tex.thd_out_correction(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                softmax_lse_in_packed_format,
            )

        out_f16 = out.to(qkv_dtype)  # type: ignore[union-attr]
        out_ret = out_f16
        q_save, k_save, v_save, out_save = q_f16, k_f16, v_f16, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            k_save,
            v_save,
            out_save,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.qkv_dtype = qkv_dtype
        ctx.cp_groups = cp_groups
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
        inter_p2p_pg = ctx.cp_groups[ParallelMode.INTER_WINDOW]
        intra_p2p_pg = ctx.cp_groups[ParallelMode.INTRA_WINDOW]
        inter_dkv_p2p_pg = ctx.cp_groups[ParallelMode.DKV_INTER_WINDOW]
        intra_dkv_p2p_pg = ctx.cp_groups[ParallelMode.DKV_INTRA_WINDOW]
        inter_cp_size = torch.distributed.get_world_size(group=inter_p2p_pg)
        intra_cp_size = torch.distributed.get_world_size(group=intra_p2p_pg)
        inter_cp_rank = torch.distributed.get_rank(group=inter_p2p_pg)
        intra_cp_rank = torch.distributed.get_rank(group=intra_p2p_pg)

        inter_send_dst, inter_recv_src = get_p2p_send_recv_rank(
            inter_cp_rank, inter_cp_size, inter_p2p_pg
        )
        intra_send_dst, intra_recv_src = get_p2p_send_recv_rank(
            intra_cp_rank, intra_cp_size, intra_p2p_pg
        )
        cp_size = torch.distributed.get_world_size(
            group=ctx.cp_groups[ParallelMode.RING]
        )

        window_num = cp_size // intra_cp_size
        batch_p2p_comm = ctx.batch_p2p_comm

        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            *other_tensors,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        causal = ctx.causal
        softmax_lse_ = None
        if ctx.second_half_lse_seqlen is not None:
            softmax_lse_ = tex.thd_read_second_half_lse(
                softmax_lse,
                cu_seqlens_q_padded,
                ctx.softmax_lse_in_packed_format,
                ctx.second_half_lse_seqlen,
            )

        dq = torch.empty_like(q)
        # dq = torch.empty_like(q)
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        intra_send_recv_reqs = []
        intra_dkv_send_recv_reqs = []
        inter_kv_send_recv_reqs = []
        inter_dkv_send_recv_reqs = []
        inter_kv_inputs = [None, None]
        inter_dkv_inputs = [
            torch.empty(*kv.shape, dtype=kv.dtype, device=kv.device),
            torch.empty(*kv.shape, dtype=kv.dtype, device=kv.device),
        ]

        inter_kv_inputs[0] = kv
        inter_kv_inputs[1] = torch.empty_like(kv)
        # outer ring
        for window_idx in range(window_num):
            if window_idx > 0:
                for req in inter_kv_send_recv_reqs:
                    req.wait()
            # kv
            if window_idx + 1 < window_num:
                inter_kv_send_recv_reqs = attn_p2p_communicate(
                    inter_cp_rank,
                    inter_kv_inputs[window_idx % 2],
                    inter_send_dst,
                    inter_kv_inputs[(window_idx + 1) % 2],
                    inter_recv_src,
                    inter_p2p_pg,
                    batch_p2p_comm,
                )
            local_kv = inter_kv_inputs[window_idx % 2]
            window_offset = ((inter_cp_rank - window_idx) % window_num) * intra_cp_size
            # inner ring
            p2p_comm_buffers[0][0].copy_(local_kv)
            for i in range(intra_cp_size):
                # wait until KV is received
                for req in intra_send_recv_reqs:
                    req.wait()

                send_tensor = p2p_comm_buffers[i % 2][0]
                recv_tensor = p2p_comm_buffers[(i + 1) % 2][0]
                dkv_send_tensor = p2p_comm_buffers[i % 2][1]
                dkv_recv_tensor = p2p_comm_buffers[(i + 1) % 2][1]

                if i < (intra_cp_size - 1):
                    intra_send_recv_reqs = attn_p2p_communicate(
                        intra_cp_rank,
                        send_tensor,
                        intra_send_dst,
                        recv_tensor,
                        intra_recv_src,
                        intra_p2p_pg,
                        batch_p2p_comm,
                    )

                kv = send_tensor
                q_, kv_, out_, dout_ = None, None, None, None
                dq_, dk_, dv_ = None, None, None
                is_half_q, is_half_kv, is_causal = False, False, False
                lse = softmax_lse

                if causal:
                    if i == 0 and window_idx == 0:  # q, k, v
                        is_causal = True
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        is_half_kv = True
                    else:
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

                rumtime_meta_per_step = ctx.runtime_meta[window_offset + i]
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
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        first_op = second_op = "add"
                    else:  # q1, k, v
                        second_op = "add"
                else:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    else:
                        first_op = second_op = "add"

                dq = bwd_dq_update(dq, dq_, cu_seqlens_q_padded, first_op, second_op)
                # wait until dKV is received
                if window_idx == 0:
                    for req in intra_dkv_send_recv_reqs:
                        req.wait()
                    dkv = dkv_send_tensor
                else:
                    if i == 0:  # other window
                        for req in inter_dkv_send_recv_reqs:
                            req.wait()
                        dkv = inter_dkv_inputs[window_idx % 2]
                        dkv_send_tensor = dkv
                    else:
                        for req in intra_dkv_send_recv_reqs:
                            req.wait()
                        dkv = dkv_send_tensor

                dkv_ = torch.cat(
                    (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                )  # pylint: disable=used-before-assignment
                # update dkv
                first_op, second_op = "none", "none"
                if causal:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        first_op = "add"
                    else:  # q1, k, v
                        first_op = second_op = "add"
                else:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    else:
                        first_op = second_op = "add"
                dkv = bwd_dkv_update(
                    dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op
                )
                # assert torch.equal(dkv,dkv_send_tensor)
                # intra dkv
                intra_dkv_send_recv_reqs = attn_p2p_communicate(
                    intra_cp_rank,
                    dkv_send_tensor,
                    intra_send_dst,
                    dkv_recv_tensor,
                    intra_recv_src,
                    intra_dkv_p2p_pg,
                    batch_p2p_comm,
                )

            for req in intra_dkv_send_recv_reqs:
                req.wait()
            # inter dkv
            inter_dkv_inputs[window_idx % 2].copy_(
                p2p_comm_buffers[intra_cp_size % 2][1]
            )
            inter_dkv_send_recv_reqs = attn_p2p_communicate(
                inter_cp_rank,
                inter_dkv_inputs[window_idx % 2],
                inter_send_dst,
                inter_dkv_inputs[(window_idx + 1) % 2],
                inter_recv_src,
                inter_dkv_p2p_pg,
                batch_p2p_comm,
            )

        for req in inter_dkv_send_recv_reqs:
            req.wait()

        dkv = inter_dkv_inputs[window_num % 2]
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


class TEDoubleRingAttnFunc(torch.autograd.Function):
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
        cp_groups,
        attn_mask_type,
        cp_stream,
        deterministic,
        batch_p2p_comm=True,
    ):
        assert qkv_format == "thd"

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        causal = "causal" in attn_mask_type
        padding = "padding" in attn_mask_type

        inter_p2p_pg = cp_groups[ParallelMode.INTER_WINDOW]
        intra_p2p_pg = cp_groups[ParallelMode.INTRA_WINDOW]
        inter_cp_size = torch.distributed.get_world_size(group=inter_p2p_pg)
        intra_cp_size = torch.distributed.get_world_size(group=intra_p2p_pg)
        inter_cp_rank = torch.distributed.get_rank(group=inter_p2p_pg)
        intra_cp_rank = torch.distributed.get_rank(group=intra_p2p_pg)

        inter_send_dst, inter_recv_src = get_p2p_send_recv_rank(
            inter_cp_rank, inter_cp_size, inter_p2p_pg
        )
        intra_send_dst, intra_recv_src = get_p2p_send_recv_rank(
            intra_cp_rank, intra_cp_size, intra_p2p_pg
        )

        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
        cp_size = torch.distributed.get_world_size(group=cp_groups[ParallelMode.RING])
        window_num = cp_size // intra_cp_size

        qkv_dtype = q.dtype
        q_f16, k_f16, v_f16 = q, k, v
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        assert (
            q.shape[-1] % 8 == 0
        ), "hidden size per attention head should be multiple of 8"

        softmax_lse_in_packed_format = get_cudnn_version() >= (9, 6, 0)

        cu_seqlens_q_per_step = [None for _ in range(cp_size)]
        cu_seqlens_kv_per_step = [None for _ in range(cp_size)]

        q_inputs = [None, None]
        intra_kv_inputs = [None, None]
        inter_kv_inputs = [None, None]
        # create two streams to resolve wave quantization issue of Flash Attn in each step
        flash_attn_streams = [torch.cuda.current_stream(), cp_stream]
        # synchronize fwd results correction across steps
        fwd_results_correction_done = torch.cuda.Event()
        # Flash Attn outputs
        out_per_step = [None for _ in range(cp_size)]
        softmax_lse_per_step = [None for _ in range(cp_size)]
        rng_states = [None for _ in range(cp_size)]
        is_half_q_per_step = [None for _ in range(cp_size)]

        local_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        intra_p2p_comm_buffers = [None for _ in range(intra_cp_size)]
        inter_kv_inputs[0] = local_kv
        inter_kv_inputs[1] = torch.empty_like(local_kv)
        intra_send_recv_reqs = [[], []]  # type: ignore
        inter_send_recv_reqs = []  # type: ignore

        second_half_lse_seqlen = None
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

        # outer ring loop
        for window_idx in range(window_num):
            if window_idx > 0:
                for req in inter_send_recv_reqs:
                    req.wait()

            if window_idx + 1 < window_num:
                inter_send_recv_reqs = attn_p2p_communicate(
                    inter_cp_rank,
                    inter_kv_inputs[window_idx % 2],
                    inter_send_dst,
                    inter_kv_inputs[(window_idx + 1) % 2],
                    inter_recv_src,
                    inter_p2p_pg,
                    batch_p2p_comm,
                )
            local_kv = inter_kv_inputs[window_idx % 2]
            window_offset = ((inter_cp_rank - window_idx) % window_num) * intra_cp_size

            # inner ring loop
            intra_p2p_comm_buffers[0] = local_kv
            for i in range(intra_cp_size + 1):
                if i < intra_cp_size:
                    with torch.cuda.stream(flash_attn_streams[i % 2]):
                        # wait until KV is received
                        for req in intra_send_recv_reqs[(i + 1) % 2]:
                            req.wait()

                        if i < (intra_cp_size - 1):
                            intra_p2p_comm_buffers[i + 1] = torch.empty_like(
                                intra_p2p_comm_buffers[i]
                            )
                            intra_send_recv_reqs[i % 2] = attn_p2p_communicate(
                                intra_cp_rank,
                                intra_p2p_comm_buffers[i],
                                intra_send_dst,
                                intra_p2p_comm_buffers[i + 1],
                                intra_recv_src,
                                intra_p2p_pg,
                                batch_p2p_comm,
                            )
                        # contiguous tensor
                        intra_kv_inputs[i % 2] = intra_p2p_comm_buffers[i]

                        is_half_q, is_half_kv = False, False
                        _max_seqlen_q, _max_seqlen_kv = max_seqlen_q, max_seqlen_kv
                        _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
                            cu_seqlens_q_padded,
                            cu_seqlens_kv_padded,
                        )
                        if causal:
                            if i == 0 and window_idx == 0:  # q, k, v
                                fused_attn_meta_kwargs[
                                    "attn_mask_type"
                                ] = attn_mask_type
                            elif (window_idx == 0 and i <= intra_cp_rank) or (
                                0 < window_idx <= inter_cp_rank
                            ):  # q, k0, v0
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

                        is_half_q_per_step[window_offset + i] = is_half_q
                        chunk_idx_q = 1 if is_half_q else -1
                        (
                            q_inputs[i % 2],
                            cu_seqlens_q_per_step[window_offset + i],
                        ) = prepare_input_fwd(
                            q,
                            chunk_idx_q,
                            cu_seqlens_q,
                            cu_seqlens_q_padded,
                            intra_cp_size,
                            window_offset + intra_cp_rank,
                        )

                        chunk_idx_kv = 0 if is_half_kv else -1
                        (
                            intra_kv_inputs[i % 2],
                            cu_seqlens_kv_per_step[window_offset + i],
                        ) = prepare_input_fwd(
                            intra_kv_inputs[i % 2],
                            chunk_idx_kv,
                            cu_seqlens_kv,
                            cu_seqlens_kv_padded,
                            intra_cp_size,
                            window_offset + (intra_cp_rank - i) % intra_cp_size,
                        )
                        (
                            out_per_step[window_offset + i],
                            aux_ctx_tensors,
                        ) = fused_attn_fwd(
                            True,
                            _max_seqlen_q,
                            _max_seqlen_kv,
                            cu_seqlens_q_per_step[window_offset + i],
                            cu_seqlens_kv_per_step[window_offset + i],
                            q_inputs[i % 2],
                            intra_kv_inputs[i % 2][0],
                            intra_kv_inputs[i % 2][1],
                            *fused_attn_meta_args,
                            **fused_attn_meta_kwargs,
                            cu_seqlens_q_padded=_cu_seqlens_q_padded,
                            cu_seqlens_kv_padded=_cu_seqlens_kv_padded,
                            **{},
                        )
                        (
                            softmax_lse_per_step[window_offset + i],
                            rng_states[window_offset + i],
                            *rest,
                        ) = aux_ctx_tensors

                if i > 0:
                    # wait until fwd restuls correction of last step is done
                    if i > 1:
                        flash_attn_streams[(i - 1) % 2].wait_event(
                            fwd_results_correction_done
                        )
                    # [b, np, sq, 1] -> [b, np, sq]
                    # or [t, np, 1] -> [t, np]
                    softmax_lse_per_step[window_offset + i - 1].squeeze_(-1)  # type: ignore[attr-defined]
                    if softmax_lse_in_packed_format:
                        softmax_lse_per_step[window_offset + i - 1] = (
                            softmax_lse_per_step[window_offset + i - 1]
                            .transpose(0, 1)
                            .contiguous()
                        )
                    if is_half_q:
                        second_half_lse_seqlen = softmax_lse_per_step[
                            window_offset + i - 1
                        ].shape[-1]
                    with torch.cuda.stream(flash_attn_streams[(i - 1) % 2]):
                        if window_idx == 0 and i == 1:
                            out = torch.zeros_like(q)
                            softmax_lse = torch.clone(
                                softmax_lse_per_step[window_offset]
                            ).to(torch.double)
                        elif (
                            (window_idx == 0 and (i - 1) <= intra_cp_rank)
                            or (0 < window_idx <= inter_cp_rank)
                            or not causal
                        ):  # q, k0, v0
                            flash_attn_fwd_softmax_lse_correction(
                                softmax_lse, softmax_lse_per_step[window_offset + i - 1]
                            )
                        else:
                            tex.thd_second_half_lse_correction(
                                softmax_lse,
                                softmax_lse_per_step[window_offset + i - 1],
                                cu_seqlens_q_padded,
                                softmax_lse_in_packed_format,
                            )
                    if i < intra_cp_size:
                        flash_attn_streams[(i - 1) % 2].record_event(
                            fwd_results_correction_done
                        )

            intra_p2p_comm_buffers[:] = [None] * len(intra_p2p_comm_buffers)
            intra_send_recv_reqs = [[], []]
            torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        torch.cuda.current_stream().wait_stream(flash_attn_streams[1])

        softmax_lse = softmax_lse.to(torch.float)
        for i in range(cp_size):
            is_half = is_half_q_per_step[i]
            tex.thd_out_correction(
                out,
                out_per_step[i],
                softmax_lse,
                softmax_lse_per_step[i],
                cu_seqlens_q_padded,
                is_half,
                softmax_lse_in_packed_format,
            )

        out_f16 = out.to(qkv_dtype)  # type: ignore[union-attr]
        out_ret = out_f16
        q_save, k_save, v_save, out_save = q_f16, k_f16, v_f16, out_f16

        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            k_save,
            v_save,
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
        ctx.cp_groups = cp_groups
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

        inter_p2p_pg = ctx.cp_groups[ParallelMode.INTER_WINDOW]
        intra_p2p_pg = ctx.cp_groups[ParallelMode.INTRA_WINDOW]
        inter_dkv_p2p_pg = ctx.cp_groups[ParallelMode.DKV_INTER_WINDOW]
        intra_dkv_p2p_pg = ctx.cp_groups[ParallelMode.DKV_INTRA_WINDOW]
        inter_cp_size = torch.distributed.get_world_size(group=inter_p2p_pg)
        intra_cp_size = torch.distributed.get_world_size(group=intra_p2p_pg)
        inter_cp_rank = torch.distributed.get_rank(group=inter_p2p_pg)
        intra_cp_rank = torch.distributed.get_rank(group=intra_p2p_pg)

        inter_send_dst, inter_recv_src = get_p2p_send_recv_rank(
            inter_cp_rank, inter_cp_size, inter_p2p_pg
        )
        intra_send_dst, intra_recv_src = get_p2p_send_recv_rank(
            intra_cp_rank, intra_cp_size, intra_p2p_pg
        )
        cp_size = torch.distributed.get_world_size(
            group=ctx.cp_groups[ParallelMode.RING]
        )
        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format

        window_num = cp_size // intra_cp_size
        batch_p2p_comm = ctx.batch_p2p_comm

        (
            q,
            k,
            v,
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

        softmax_lse_ = None
        if ctx.second_half_lse_seqlen is not None:
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
        kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0)
        p2p_comm_buffers = [
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
            torch.empty((2, *kv.shape), dtype=kv.dtype, device=kv.device),
        ]
        fused_attn_dqkv_dtype = TE_DType[dout_dtype]
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        out = out.view(*q.shape)
        dout = dout.view(*q.shape)
        intra_send_recv_reqs = []
        intra_dkv_send_recv_reqs = []
        inter_kv_send_recv_reqs = []
        inter_dkv_send_recv_reqs = []
        inter_kv_inputs = [None, None]
        inter_dkv_inputs = [
            torch.empty(*kv.shape, dtype=kv.dtype, device=kv.device),
            torch.empty(*kv.shape, dtype=kv.dtype, device=kv.device),
        ]

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

        inter_kv_inputs[0] = kv
        inter_kv_inputs[1] = torch.empty_like(kv)
        # outer ring
        for window_idx in range(window_num):
            if window_idx > 0:
                for req in inter_kv_send_recv_reqs:
                    req.wait()
            # kv
            if window_idx + 1 < window_num:
                inter_kv_send_recv_reqs = attn_p2p_communicate(
                    inter_cp_rank,
                    inter_kv_inputs[window_idx % 2],
                    inter_send_dst,
                    inter_kv_inputs[(window_idx + 1) % 2],
                    inter_recv_src,
                    inter_p2p_pg,
                    batch_p2p_comm,
                )
            local_kv = inter_kv_inputs[window_idx % 2]
            window_offset = ((inter_cp_rank - window_idx) % window_num) * intra_cp_size
            # inner ring
            p2p_comm_buffers[0][0].copy_(local_kv)
            for i in range(intra_cp_size):
                # wait until KV is received
                for req in intra_send_recv_reqs:
                    req.wait()

                send_tensor = p2p_comm_buffers[i % 2][0]
                recv_tensor = p2p_comm_buffers[(i + 1) % 2][0]
                dkv_send_tensor = p2p_comm_buffers[i % 2][1]
                dkv_recv_tensor = p2p_comm_buffers[(i + 1) % 2][1]

                if i < (intra_cp_size - 1):
                    intra_send_recv_reqs = attn_p2p_communicate(
                        intra_cp_rank,
                        send_tensor,
                        intra_send_dst,
                        recv_tensor,
                        intra_recv_src,
                        intra_p2p_pg,
                        batch_p2p_comm,
                    )

                kv = send_tensor
                dq_, dk_, dv_ = None, None, None

                is_half_q, is_half_kv = False, False
                _max_seqlen_q, _max_seqlen_kv = ctx.max_seqlen_q, ctx.max_seqlen_kv
                _cu_seqlens_q_padded, _cu_seqlens_kv_padded = (
                    cu_seqlens_q_padded,
                    cu_seqlens_kv_padded,
                )
                if causal:
                    if i == 0 and window_idx == 0:  # q, k, v
                        fused_attn_meta_kwargs["attn_mask_type"] = ctx.attn_mask_type
                        aux_ctx_tensors = [softmax_lse, rng_states[window_offset + i]]
                        fused_attn_meta_args[2] = aux_ctx_tensors
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        aux_ctx_tensors = [softmax_lse, rng_states[window_offset + i]]
                        fused_attn_meta_args[2] = aux_ctx_tensors
                        fused_attn_meta_kwargs["attn_mask_type"] = (
                            "padding" if padding else "no_mask"
                        )
                        is_half_kv = True
                        _max_seqlen_kv = ctx.max_seqlen_kv // 2
                        _cu_seqlens_kv_padded = _cu_seqlens_kv_padded // 2
                    else:  # q1, k, v
                        assert softmax_lse_ is not None
                        aux_ctx_tensors = [softmax_lse_, rng_states[window_offset + i]]
                        fused_attn_meta_args[2] = aux_ctx_tensors
                        fused_attn_meta_kwargs["attn_mask_type"] = (
                            "padding" if padding else "no_mask"
                        )
                        is_half_q = True
                        _max_seqlen_q = ctx.max_seqlen_q // 2
                        _cu_seqlens_q_padded = _cu_seqlens_q_padded // 2
                else:  # full
                    aux_ctx_tensors = [softmax_lse, rng_states[window_offset + i]]
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
                    cu_seqlens_q_per_step[window_offset + i],
                    cu_seqlens_kv_per_step[window_offset + i],
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
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        first_op = second_op = "add"
                    else:  # q1, k, v
                        second_op = "add"
                else:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    else:
                        first_op = second_op = "add"

                dq = bwd_dq_update(dq, dq_, cu_seqlens_q_padded, first_op, second_op)
                # wait until dKV is received
                if window_idx == 0:
                    for req in intra_dkv_send_recv_reqs:
                        req.wait()
                    dkv = dkv_send_tensor
                else:
                    if i == 0:  # other window
                        for req in inter_dkv_send_recv_reqs:
                            req.wait()
                        dkv = inter_dkv_inputs[window_idx % 2]
                        dkv_send_tensor = dkv
                    else:
                        for req in intra_dkv_send_recv_reqs:
                            req.wait()
                        dkv = dkv_send_tensor

                dkv_ = torch.cat(
                    (dk_.unsqueeze(0), dv_.unsqueeze(0)), dim=0
                )  # pylint: disable=used-before-assignment
                # update dkv
                first_op, second_op = "none", "none"
                if causal:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    elif (window_idx == 0 and i <= intra_cp_rank) or (
                        0 < window_idx <= inter_cp_rank
                    ):  # q, k0, v0
                        first_op = "add"
                    else:  # q1, k, v
                        first_op = second_op = "add"
                else:
                    if i == 0 and window_idx == 0:
                        first_op = second_op = "copy"
                    else:
                        first_op = second_op = "add"
                dkv = bwd_dkv_update(
                    dkv, dkv_, cu_seqlens_kv_padded, first_op, second_op
                )
                # assert torch.equal(dkv,dkv_send_tensor)
                # intra dkv
                intra_dkv_send_recv_reqs = attn_p2p_communicate(
                    intra_cp_rank,
                    dkv_send_tensor,
                    intra_send_dst,
                    dkv_recv_tensor,
                    intra_recv_src,
                    intra_dkv_p2p_pg,
                    batch_p2p_comm,
                )

            for req in intra_dkv_send_recv_reqs:
                req.wait()
            # inter dkv
            inter_dkv_inputs[window_idx % 2].copy_(
                p2p_comm_buffers[intra_cp_size % 2][1]
            )
            inter_dkv_send_recv_reqs = attn_p2p_communicate(
                inter_cp_rank,
                inter_dkv_inputs[window_idx % 2],
                inter_send_dst,
                inter_dkv_inputs[(window_idx + 1) % 2],
                inter_recv_src,
                inter_dkv_p2p_pg,
                batch_p2p_comm,
            )

        for req in inter_dkv_send_recv_reqs:
            req.wait()

        dkv = inter_dkv_inputs[window_num % 2]
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


class LoongTrain(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        self.pg_p2p = cp_process_group[ParallelMode.RING]
        self.pg_a2a = cp_process_group[ParallelMode.ULYSESS]
        self.cp_process_group = cp_process_group
        # pad factor for ulysess & ring
        self.pad_factor_p2p, self.pad_factor_a2a = get_pad_factor(
            cp_group_p2p=self.pg_p2p, cp_group_a2a=self.pg_a2a
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

            cp_groups = self.cp_process_group
            inter_p2p_pg = cp_groups[ParallelMode.INTER_WINDOW]
            intra_p2p_pg = cp_groups[ParallelMode.INTRA_WINDOW]
            intra_cp_size = torch.distributed.get_world_size(group=intra_p2p_pg)
            inter_cp_rank = torch.distributed.get_rank(group=inter_p2p_pg)
            intra_cp_rank = torch.distributed.get_rank(group=intra_p2p_pg)
            cp_size = torch.distributed.get_world_size(
                group=cp_groups[ParallelMode.RING]
            )
            self.runtime_meta_per_step = [None for i in range(cp_size)]
            window_num = cp_size // intra_cp_size

            for window_idx in range(window_num):
                window_offset = (
                    (inter_cp_rank - window_idx) % window_num
                ) * intra_cp_size
                for i in range(intra_cp_size):
                    first_idx_q, second_idx_q, first_idx_kv, second_idx_kv = (
                        True,
                        True,
                        True,
                        True,
                    )
                    factor_q, factor_kv = cp_size, cp_size
                    if causal:
                        if i == 0 and window_idx == 0:  # q, k, v
                            pass
                        elif (window_idx == 0 and i <= intra_cp_rank) or (
                            0 < window_idx <= inter_cp_rank
                        ):  # q, k0, v0
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
                        intra_cp_size,
                        window_offset + intra_cp_rank,
                        first_idx_q,
                        second_idx_q,
                    )
                    cu_seqlens_kv_per_step = get_cu_seqlens_on_cp_rank(
                        shard_kv_meta.cu_seqlens,
                        shard_kv_meta.cu_seqlens_padded // cp_size,
                        intra_cp_size,
                        window_offset + (intra_cp_rank - i) % intra_cp_size,
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
                    self.runtime_meta_per_step[window_offset + i] = rumtime_meta

    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str,  # key name for shard_meta
        **kwargs,
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
            cp_group_a2a=self.pg_a2a,
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
        **kwargs,
    ) -> torch.Tensor:
        smeta = self.shard_meta[name]
        x_global_varlen = zigzag_undispatch(
            x_local,
            smeta.host_cu_seqlens,
            smeta.host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=self.pg_p2p,
            cp_group_a2a=self.pg_a2a,
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
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_size_p2p = dist.get_world_size(group=self.pg_p2p)
        # ulysess all2all
        batch_dim, seq_dim = 0, 1

        # thd -> 1,thd
        q, k, v = [x.unsqueeze(0) for x in [q, k, v]]
        q_layer, k_layer, v_layer = [
            _SeqAllToAll.apply(self.pg_a2a, x, 2, seq_dim, batch_dim) for x in [q, k, v]
        ]
        q_layer, k_layer, v_layer = [x.squeeze(0) for x in [q_layer, k_layer, v_layer]]

        batch_p2p_comm = kwargs.get("batch_p2p_comm", True)
        with torch.cuda.device(q.device):
            cp_stream = torch.cuda.Stream()

        # ring attention p2p
        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]

        if self.backend == AttnBackend.TE:
            if attn_mask_type == AttnMaskType.CAUSAL:
                attn_mask = "padding_causal"
            elif attn_mask_type == AttnMaskType.FULL:
                attn_mask = "padding"

            out, lse = TEDoubleRingAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen_padded // cp_size_p2p,
                shard_kv_meta.max_seqlen_padded // cp_size_p2p,
                shard_q_meta.cu_seqlens_padded // cp_size_p2p,
                shard_kv_meta.cu_seqlens_padded // cp_size_p2p,
                dropout_p,
                softmax_scale,
                "thd",
                self.cp_process_group,
                attn_mask,
                cp_stream,
                deterministic,
                batch_p2p_comm,
            )

        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out, lse = FA3DoubleRingAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens_padded // cp_size_p2p,
                shard_kv_meta.cu_seqlens_padded // cp_size_p2p,
                self.runtime_meta_per_step,
                is_causal,
                dropout_p,
                softmax_scale,
                self.cp_process_group,
                cp_stream,
                deterministic,
                batch_p2p_comm,
            )

        # ulysess all2all
        out = out.unsqueeze(0)
        out_layer = _SeqAllToAll.apply(self.pg_a2a, out, seq_dim, 2, batch_dim)
        out_layer = out_layer.squeeze(0)

        return out_layer, lse
