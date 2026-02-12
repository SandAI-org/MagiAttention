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

# mypy: disable-error-code="union-attr,list-item"
import logging
import warnings
from logging import getLogger
from typing import Any, TypeAlias

import torch
import torch.distributed as dist
from einops import rearrange
from torch.distributed import ReduceOp

import magi_attention
from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.work import GeneralWork, WorkWithPostProcessFn
from magi_attention.common import AttnForwardMeta
from magi_attention.common.enum import GrpCollBufferName
from magi_attention.magi_attn_ext import KernelBarrier
from magi_attention.meta.collection import CalcMeta, CommMeta
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.utils import is_same_process_group, max_fp_dtype, nvtx

from .fa4 import fa4_bwd, fa4_fwd
from .flex_flash_attn import _flex_flash_attn_backward, _flex_flash_attn_forward
from .sdpa import sdpa_bwd, sdpa_fwd
from .utils import calc_lse_sink_compiled, correct_attn_out_lse, sink_bwd_compiled

logger = getLogger(__name__)

FusedOrTupleTensor: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
WorkWithBuffer: TypeAlias = (
    tuple[WorkWithPostProcessFn, torch.Tensor]
    | tuple[WorkWithPostProcessFn, FusedOrTupleTensor]
)


class DistAttnRuntime:
    """
    Runtime class for Distributed Flash Attention.

    Args:
        comm_meta (CommMeta): the communication metadata
        calc_meta (CalcMeta): the calculation metadata
        cp_group_gc (dist.ProcessGroup): the cp process group for group-cast
        cp_group_gr (dist.ProcessGroup): the cp process group for group-reduce
    """

    remote_q_work_with_buffer_per_stage: list[WorkWithBuffer]
    remote_kv_work_with_buffer_per_stage: list[WorkWithBuffer]
    partial_out_lse_reduce_work_per_stage: list[WorkWithPostProcessFn]
    remote_qo_do_lse_work_with_buffer_per_stage: list[WorkWithBuffer]
    partial_dq_reduce_work_per_stage: list[WorkWithPostProcessFn]
    partial_dkv_reduce_work_per_stage: list[WorkWithPostProcessFn]
    partial_dsink_reduce_work: WorkWithPostProcessFn

    def __init__(
        self,
        comm_meta: CommMeta,
        calc_meta: CalcMeta,
        cp_group_gc: dist.ProcessGroup,
        cp_group_gr: dist.ProcessGroup,
    ):
        self.comm_meta = comm_meta
        self.calc_meta = calc_meta
        self.cp_group_gc = cp_group_gc
        self.cp_group_gr = cp_group_gr
        self.overlap_degree = comm_meta.overlap_degree

        # ----------    other control flags for fwd   --------- #

        # cp1 shortcut: skip all communication
        self.skip_comm = cp_group_gc.size() == 1

        # NOTE: concat kv together for comm only when not using native grpcoll and world_size > 1
        self.concat_kv = (not self.skip_comm) and (not self.use_native_grpcoll)

        # NOTE: when disabling qo comm
        # if not using sdpa backend and not using fa4 backend,
        # we will use accumulative buffer for partial out and lse
        # to avoid the storage of partial results
        # and an additional explicit `correct_attn_out_lse`
        self.fwd_out_lse_use_acc = (
            not self.enable_qo_comm
            and not self.use_sdpa_backend
            and not self.use_fa4_backend
        )

        # NOTE: When enabling qo comm without native group collectives
        # or high-precision reduction, we must initialize the partial local output
        # in low precision.
        #
        # Reasoning:
        # 1. The Flex Flash Attention forward kernel always outputs in FP32 (high precision).
        # 2. When qo comm is enabled but high-precision reduce is disabled, communication
        #    occurs in low precision (16-bit).
        # 3. Unless using native grpcoll (which handles FP32 buffers with internal
        #    low-precision communication), we must initialize the local output in
        #    low precision to avoid explicit downcasting during communication.
        self.fwd_local_out_lp_init = (
            self.enable_qo_comm
            and not self.use_native_grpcoll
            and not self.fwd_hp_reduce
        )

        # ----------    other control flags for bwd   --------- #

        # NOTE: concat dkv together for comm only when not using native grpcoll and world_size > 1
        self.concat_dkv = (not self.skip_comm) and (not self.use_native_grpcoll)

        # NOTE: concat q,o,do together for comm only when enabling qo comm and not using native grpcoll and world_size > 1
        self.concat_qo_do = (
            self.enable_qo_comm
            and (not self.skip_comm)
            and (not self.use_native_grpcoll)
        )

        # NOTE: when disabling qo comm
        # if not using sdpa backend and not using fa4 backend,
        # we will use accumulative buffer for partial dq
        # to avoid an additional explicit `add_`
        self.bwd_dq_use_acc = (
            not self.enable_qo_comm
            and not self.use_sdpa_backend
            and not self.use_fa4_backend
        )

        # NOTE: when neither using native grpcoll nor enabling bwd high precision reduce
        # we're supposed to initialize the partial local dk,dv in low precision
        # and the same applies to dq when enabling qo comm
        self.bwd_local_dkv_lp_init = (
            not self.use_native_grpcoll and not self.bwd_hp_reduce
        )
        self.bwd_local_dq_lp_init = (
            self.enable_qo_comm
            and not self.use_native_grpcoll
            and not self.bwd_hp_reduce
        )

        # ----------    internal temporary work list   --------- #

        self._reset_work_list()

    # ----------    API for fwd   --------- #

    @nvtx.instrument_nvtx
    def apply_fwd_partial_attn(
        self,
        q: torch.Tensor,
        kv: FusedOrTupleTensor,
        out_acc: torch.Tensor | None = None,
        lse_acc: torch.Tensor | None = None,
        max_logits_acc: torch.Tensor | None = None,
        overlap_stage: int | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        sink: torch.Tensor | None = None,
        return_max_logits: bool = False,
    ) -> tuple[torch.Tensor | None, AttnForwardMeta | None]:
        """
        Apply forward partial attention with given q,kv for the given overlap stage

        Args:
            q (torch.Tensor): current q tensor
            kv (FusedOrTupleTensor): current kv fused or tupled tensor
            out_acc (torch.Tensor, optional): accumulative buffer for out
            lse_acc (torch.Tensor, optional): accumulative buffer for lse
            overlap_stage (int, optional): given overlap stage. Defaults to None.

            softmax_scale (float, optional): softmax scale.
                Defaults to ``None`` to use default value: ``1/sqrt(head_dim)``
            softcap (float, optional): softcap. Defaults to ``0``.

            sink (torch.Tensor, optional): sink tensor.
                Defaults to ``None`` to not apply attention sink.

            return_max_logits (bool, optional): whether to return max logits per head.
                Defaults to ``False``.

        Returns:
            out (torch.Tensor | None): partial out, or ``None`` if skipped
            meta (AttnForwardMeta | None): partial attention meta, or ``None`` if skipped

        Shape:
            q: [num_tokens_q, num_heads_q, head_dim]
            kv: [num_tokens_kv*2, num_heads_kv, head_dim]
            sink: [num_tokens_sink, num_heads_q]
            out: [num_tokens_q, num_heads_q, head_dim]
            lse: [num_tokens_q, num_heads_q]
            out_acc: [num_tokens_q, num_heads_q, head_dim]
            lse_acc: [num_tokens_q, num_heads_q]
        """
        is_host_stage = self.is_host_stage(overlap_stage)

        # FIXME
        if self.flatten_head_groups:
            assert sink is None, "Flattening head groups is incompatible with attn sink"
            assert (
                return_max_logits is False
            ), "Flattening head groups is incompatible with return_max_logits"

        # fetch attn arg
        if is_host_stage:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            attn_arg = self.calc_meta.remote_attn_args_list[curr_remote_stage]

        # skipped case
        if attn_arg.can_skip(is_bwd=False):
            if is_host_stage:
                partial_out, partial_lse = self._init_out_lse_skipped_host_stage(
                    q=q,
                    sink=sink,
                )
                partial_max_logits = self._init_max_logits_skipped_host_stage(
                    q=q,
                    return_max_logits=return_max_logits,
                )
                return partial_out, AttnForwardMeta(
                    lse=partial_lse, max_logits=partial_max_logits
                )
            return None, None

        # attention forward pass
        k, v = self._maybe_chunk(kv, num_chunks=2)
        _softmax_scale: float = (
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )
        partial_out, meta = self._launch_attn_fwd_kernel(
            q=q,
            k=k,
            v=v,
            sink=sink,
            out_acc=out_acc,
            lse_acc=lse_acc,
            max_logits_acc=max_logits_acc,
            attn_arg=attn_arg,
            softmax_scale=_softmax_scale,
            softcap=softcap,
            is_host_stage=is_host_stage,
            return_max_logits=return_max_logits,
        )

        # maybe downcast out to q dtype for the host stage
        if is_host_stage and self.fwd_local_out_lp_init:
            partial_out = partial_out.to(q.dtype)

        return partial_out, meta

    @nvtx.instrument_nvtx
    def get_curr_q_kv_and_fetch_next(
        self,
        local_q: torch.Tensor,
        local_kv: FusedOrTupleTensor,
        overlap_stage: int | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> tuple[torch.Tensor, FusedOrTupleTensor]:
        """
        Get current q,kv and fetch next q,kv for the given overlap stage

        Args:
            local_q (torch.Tensor): local q
            local_kv (FusedOrTupleTensor): local kv fused or tupled tensor
            overlap_stage (int, optional): given overlap stage. Defaults to None.

        Returns:
            curr_q (torch.Tensor): current q
            curr_kv (FusedOrTupleTensor): current kv fused or tupled tensor
        """
        next_stage = self.get_next_stage(overlap_stage)
        is_host_stage = self.is_host_stage(overlap_stage)
        is_last_remote_stage = self.is_last_remote_stage(overlap_stage)

        # wait for host/remote qkv prepared for current stage
        if is_host_stage:
            local_q, local_kv = self._maybe_flatten_local_qkv_head_groups(
                local_q=local_q,
                local_kv=local_kv,
            )
            local_kv = self._maybe_concat(*local_kv, need_concat=self.concat_kv)
            curr_q, curr_kv = local_q, local_kv
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            (
                remote_q_work,
                remote_q_buffer,
            ) = self.remote_q_work_with_buffer_per_stage[curr_remote_stage]
            curr_q = remote_q_work.wait_post_process(remote_q_buffer)

            (
                remote_kv_work,
                remote_kv_buffer,
            ) = self.remote_kv_work_with_buffer_per_stage[curr_remote_stage]
            curr_kv = remote_kv_work.wait_post_process(remote_kv_buffer)

        # pre-fetch remote qkv for next stage(s)
        if self.prefetch_stage_by_stage and not is_last_remote_stage:
            # if using stage-by-stage prefetch, we only pre-fetch the next stage
            # to avoid blocking the current ffa fwd
            (
                self.remote_q_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_q(
                local_q=local_q,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastQO,
                kernel_barrier=kernel_barrier,
            )
            (
                self.remote_kv_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_kv(
                local_kv=local_kv,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastDefault,
                kernel_barrier=kernel_barrier,
            )
        elif is_host_stage:
            # when not using stage-by-stage prefetch,
            # we issue all fetch-remote comms in advance of ffa fwd
            # and ffa fwd can still overlap with these comms
            # with the support of non-zero `sm_margin`, thanks to persistent kernel design
            self.remote_q_work_with_buffer_per_stage = [
                self._fetch_remote_q(
                    local_q=local_q,
                    overlap_stage=ith_stage,
                    buffer_name=GrpCollBufferName.GroupCastQO,
                    kernel_barrier=kernel_barrier,
                )
                for ith_stage in range(self.overlap_degree)
            ]
            self.remote_kv_work_with_buffer_per_stage = [
                self._fetch_remote_kv(
                    local_kv=local_kv,
                    overlap_stage=ith_stage,
                    buffer_name=GrpCollBufferName.GroupCastDefault,
                    kernel_barrier=kernel_barrier,
                )
                for ith_stage in range(self.overlap_degree)
            ]

        return curr_q, curr_kv

    @nvtx.instrument_nvtx
    def reduce_partial_out_lse(
        self,
        partial_remote_out: torch.Tensor | None,
        partial_remote_lse: torch.Tensor | None,
        partial_local_out: torch.Tensor,
        partial_local_lse: torch.Tensor,
        ref_remote_out: torch.Tensor,
        overlap_stage: int,
        kernel_barrier: KernelBarrier | None = None,
    ) -> None:
        """
        Reduce remote out and lse to local out and lse for the given remote overlap stage
        and push the returned partial_out_lse_reduce_work to self.partial_out_lse_reduce_work_per_stage

        Args:
            partial_remote_out (torch.Tensor, optional):
                partial remote out in float32 dtype, or ``None`` if skipped
            partial_remote_lse (torch.Tensor, optional):
                partial remote lse in float32 dtype, or ``None`` if skipped
            partial_local_out (torch.Tensor): partial local out to be reduced
            partial_local_lse (torch.Tensor): partial local lse to be reduced
            ref_remote_out (torch.Tensor):
                reference remote out, to provide meta info like dtype and shape
            overlap_stage (int): given remote overlap stage
        """

        partial_out_lse_reduce_work = self._reduce_partial_out_lse(
            partial_remote_out=partial_remote_out,
            partial_remote_lse=partial_remote_lse,
            partial_local_out=partial_local_out,
            partial_local_lse=partial_local_lse,
            ref_remote_out=ref_remote_out,
            overlap_stage=overlap_stage,
            buffer_name=GrpCollBufferName.GroupReduceQO,
            kernel_barrier=kernel_barrier,
        )
        self.partial_out_lse_reduce_work_per_stage[
            overlap_stage
        ] = partial_out_lse_reduce_work

    @nvtx.instrument_nvtx
    def prepare_reduced_local_out_lse(
        self,
        partial_local_out: torch.Tensor,
        partial_local_lse: torch.Tensor,
        ref_local_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the final reduced local out and lse
        before returning from forward
        with clearing the temporary work list

        Args:
            partial_local_out (torch.Tensor): partial local out to be reduced
            partial_local_lse (torch.Tensor): partial local lse to be reduced
            ref_local_out (torch.Tensor):
                reference local out, to provide meta info like dtype and shape

        Returns:
            local_out (torch.Tensor): reduced local out tensor
            local_lse (torch.Tensor): reduced local lse tensor
        """
        # wait for all partial out reduced
        for partial_out_lse_reduce_work in self.partial_out_lse_reduce_work_per_stage:
            (
                partial_local_out,
                partial_local_lse,
            ) = partial_out_lse_reduce_work.wait_post_process(
                partial_local_out, partial_local_lse
            )

        # cast final local out to ref dtype
        local_out = partial_local_out.to(ref_local_out.dtype)
        local_lse = partial_local_lse  # lse always in high-precision

        # maybe unflatten head groups of local out/lse
        # NOTE: if we flatten head groups in forward, there are two strategies for backward:
        # 1. return the unflattened out/lse, while save flattened ones for backward,
        #   which trades off double activations of out for avoiding re-flattening of out/lse in the backward pass
        # 2. return the unflattened out/lse, and save them for backward as well,
        #   which remains the activation still as before, but causes re-flattening of out/lse in the backward pass
        # for now, we use strategy 1 due to the fact that activation is usually more expensive than re-flattening
        local_out, local_lse = self._maybe_unflatten_local_out_lse_head_groups(
            local_out=local_out, local_lse=local_lse
        )

        # reset the temporary work list
        self._reset_work_list()

        return local_out, local_lse

    def reduce_max_logits(
        self,
        partial_local_max_logits: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        All-reduce max_logits across all ranks (element-wise MAX),
        and optionally unflatten when flatten_head_groups is enabled.

        Call this at the end of distributed attention forward when return_max_logits is True.

        Args:
            partial_local_max_logits (torch.Tensor | None): per-rank partial max_logits
                per head, shape [num_heads_q].

        Returns:
            local_max_logits (torch.Tensor): all-reduced max_logits [num_heads_q]
        """
        if not self.skip_comm:
            dist.all_reduce(
                partial_local_max_logits,
                op=ReduceOp.MAX,
                group=self.cp_group_gr,
            )
        return partial_local_max_logits

    def save_tensors_for_bwd(
        self,
        ctx,
        local_q: torch.Tensor,
        local_kv: FusedOrTupleTensor,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
        last_stage_q: torch.Tensor | None,
        last_stage_kv: FusedOrTupleTensor | None,
        global_sink: torch.Tensor | None,
    ) -> None:
        if last_stage_kv is None:
            self.save_last_stage_for_backward = False
            if self.concat_kv:  # local_kv is a fused tensor
                ctx.save_for_backward(
                    local_q, local_kv, local_out, local_lse, global_sink
                )
            else:  # local_kv are tupled tensors
                ctx.save_for_backward(
                    local_q, *local_kv, local_out, local_lse, global_sink
                )
        else:
            self.save_last_stage_for_backward = True
            if self.concat_kv:  # local_kv is a fused tensor
                ctx.save_for_backward(
                    local_q,
                    local_kv,
                    local_out,
                    local_lse,
                    last_stage_q,
                    last_stage_kv,
                    global_sink,
                )
            else:  # local_kv are tupled tensors
                ctx.save_for_backward(
                    local_q,
                    *local_kv,
                    local_out,
                    local_lse,
                    last_stage_q,
                    *last_stage_kv,
                    global_sink,
                )

    # ----------    API for bwd   --------- #

    @nvtx.instrument_nvtx
    def apply_bwd_partial_attn(
        self,
        qo_do: FusedOrTupleTensor,
        kv: FusedOrTupleTensor,
        lse: torch.Tensor,
        dq_acc: torch.Tensor | None = None,
        overlap_stage: int | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        sink: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, FusedOrTupleTensor | None, torch.Tensor | None]:
        """
        Apply backward partial attention with given qo_do,kv,lse for the given overlap stage

        Args:
            qo_do (FusedOrTupleTensor): current q, o, do fused or tupled tensor
            kv (FusedOrTupleTensor): current kv
            lse (torch.Tensor): current lse
            dq_acc (torch.Tensor, optional): accumulative buffer for dq
            overlap_stage (int, optional): given overlap stage. Defaults to None.

            softmax_scale (float, optional): softmax scale.
                Defaults to ``None`` to use default value: ``1/sqrt(head_dim)``
            softcap (float, optional): softcap. Defaults to ``0``.

            sink (torch.Tensor, optional): sink tensor.
                Defaults to ``None`` to not apply backward of sink to compute partial dsink.

        Returns:
            partial_dq (torch.Tensor | None): partial dq, or ``None`` if skipped
            partial_dkv (FusedOrTupleTensor | None): partial dkv, or ``None`` if skipped
            partial_dsink (torch.Tensor | None): partial dsink, or ``None`` if skipped

        Shape:
            qo_do: [num_tokens_q*3, num_heads_q, head_dim]
            kv: [num_tokens_kv*2, num_heads_kv, head_dim]
            sink: [num_tokens_sink, num_heads_q]
            lse: [num_tokens_q, num_heads_q]
            partial_dq: [num_tokens_q, num_heads_q, head_dim]
            partial_dkv: [num_tokens_kv*2, num_heads_kv, head_dim]
            partial_dsink: [num_tokens_sink, num_heads_q]
        """

        is_host_stage = self.is_host_stage(overlap_stage)

        # FIXME
        if self.flatten_head_groups:
            assert sink is None, "Flattening head groups is incompatible with attn sink"

        # fetch attn arg
        if is_host_stage:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            attn_arg = self.calc_meta.remote_attn_args_list[curr_remote_stage]

        # skipped case
        if attn_arg.can_skip(is_bwd=True):
            if is_host_stage:
                return self._init_dq_dkv_dsink_skipped_host_stage(
                    qo_do=qo_do,
                    kv=kv,
                    lse=lse,
                    sink=sink,
                )
            return None, None, None

        # prepare tensors and other meta
        q, o, do = self._maybe_chunk(qo_do, num_chunks=3)
        k, v = self._maybe_chunk(kv, num_chunks=2)
        _softmax_scale: float = (
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )
        if self.concat_kv:  # kv is a fused tensor
            dkv_shape = kv.shape
        else:  # kv are tupled tensors
            dkv_shape = (k.shape[0] * 2, *k.shape[1:])

        # attention backward pass
        (
            partial_dq,
            partial_dkv,
            partial_dsink,
        ) = self._launch_attn_bwd_kernel(
            do=do,
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            sink=sink,
            dq_acc=dq_acc,
            attn_arg=attn_arg,
            softmax_scale=_softmax_scale,
            softcap=softcap,
            is_host_stage=is_host_stage,
            dkv_shape=dkv_shape,
        )

        # maybe downcast dq,dkv to q,kv dtype for the host stage
        if is_host_stage:
            if self.bwd_local_dq_lp_init:
                partial_dq = partial_dq.to(q.dtype)
            if self.bwd_local_dkv_lp_init:
                if self.concat_dkv:  # partial_dkv is a fused tensor
                    partial_dkv = partial_dkv.to(kv.dtype)
                else:  # partial_dkv are tupled tensors
                    partial_dkv = tuple(pdkv.to(kv[0].dtype) for pdkv in partial_dkv)

        return partial_dq, partial_dkv, partial_dsink

    @nvtx.instrument_nvtx
    def get_curr_qo_do_kv_lse_and_fetch_next(
        self,
        local_qo_do: FusedOrTupleTensor,
        local_kv: FusedOrTupleTensor,
        local_lse: torch.Tensor,
        overlap_stage: int | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> tuple[FusedOrTupleTensor, FusedOrTupleTensor, torch.Tensor]:
        """
        Get current qo_do,kv,lse and fetch next qo_do,kv,lse for the given overlap stage

        Args:
            local_qo_do (FusedOrTupleTensor): local qo_do fused or tupled tensor
            local_kv (torch.Tensor): local kv fused or tupled tensor
            local_lse (torch.Tensor): local lse tensor
            overlap_stage (int, optional): given overlap stage. Defaults to None.

        Returns:
            curr_qo_do (FusedOrTupleTensor): current qo_do fused or tupled tensor
            curr_kv (torch.Tensor): current kv fused or tupled tensor
            curr_lse (torch.Tensor): current lse tensor
        """
        next_stage = self.get_next_stage(overlap_stage)
        is_host_stage = self.is_host_stage(overlap_stage)
        is_last_remote_stage = self.is_last_remote_stage(overlap_stage)

        # wait for host/remote qo_do,kv,lse prepared for current stage
        if is_host_stage:
            local_qo_do, local_lse = self._maybe_flatten_local_qo_do_lse_head_groups(
                local_qo_do=local_qo_do,
                local_lse=local_lse,
            )
            local_qo_do = self._maybe_concat(
                *local_qo_do, need_concat=self.concat_qo_do
            )
            curr_qo_do, curr_kv, curr_lse = local_qo_do, local_kv, local_lse
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            (
                curr_remote_kv_work,
                curr_remote_kv_buffer,
            ) = self.remote_kv_work_with_buffer_per_stage[curr_remote_stage]
            curr_kv = curr_remote_kv_work.wait_post_process(curr_remote_kv_buffer)
            (
                curr_remote_qo_do_lse_work,
                curr_remote_qo_do_lse_buffer,
            ) = self.remote_qo_do_lse_work_with_buffer_per_stage[curr_remote_stage]
            curr_qo_do, curr_lse = curr_remote_qo_do_lse_work.wait_post_process(
                curr_remote_qo_do_lse_buffer
            )

        # pre-fetch remote qo_do,kv,lse for next stage(s)
        if self.prefetch_stage_by_stage and not is_last_remote_stage:
            if (
                magi_attention.dist_attn_backward_hide_tail_reduce()
                and self.is_penultimate_stage(overlap_stage)
            ):
                return curr_qo_do, curr_kv, curr_lse

            (
                self.remote_kv_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_kv(
                local_kv=local_kv,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastDefault,
                kernel_barrier=kernel_barrier,
            )
            (
                self.remote_qo_do_lse_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_qo_do_lse(
                local_qo_do=local_qo_do,
                local_lse=local_lse,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastQO,
                kernel_barrier=kernel_barrier,
            )
        elif is_host_stage:
            # NOTE: if not using stage-by-stage prefetch,
            # we issue all fetch-remote comms in advance of ffa bwd
            # and ffa bwd can still overlap with these comms
            # with the support of `sm_margin`, thanks to persistent kernel design
            if (
                magi_attention.dist_attn_backward_hide_tail_reduce()
                and self.overlap_degree > 0
            ):
                degree = self.overlap_degree - 1
            else:
                degree = self.overlap_degree

            self.remote_kv_work_with_buffer_per_stage = [
                self._fetch_remote_kv(local_kv=local_kv, overlap_stage=ith_stage)
                for ith_stage in range(degree)
            ]
            self.remote_qo_do_lse_work_with_buffer_per_stage = [
                self._fetch_remote_qo_do_lse(
                    local_qo_do=local_qo_do,
                    local_lse=local_lse,
                    overlap_stage=ith_stage,
                )
                for ith_stage in range(degree)
            ]

        return curr_qo_do, curr_kv, curr_lse

    @nvtx.instrument_nvtx
    def reduce_partial_dq_dkv(
        self,
        partial_remote_dq: torch.Tensor | None,
        partial_local_dq: torch.Tensor,
        ref_remote_qo_do: FusedOrTupleTensor,
        partial_remote_dkv: FusedOrTupleTensor | None,
        partial_local_dkv: FusedOrTupleTensor,
        ref_remote_kv: FusedOrTupleTensor,
        overlap_stage: int,
        kernel_barrier: KernelBarrier | None = None,
    ) -> None:
        """
        Reduce remote dq,dkv to local dq,dkv for the given remote overlap stage
        and push the returned partial_dq_reduce_work,partial_dkv_reduce_work
        to self.partial_dq_reduce_work_per_stage, self.partial_dkv_reduce_work_per_stage
        respectively

        Args:
            partial_remote_dq (torch.Tensor, optional):
                partial remote dq in float32 dtype, or ``None`` if skipped
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            ref_remote_qo_do (FusedOrTupleTensor):
                reference remote qo_do fused or tupled tensor,
                to provide meta info like dtype and shape
            partial_remote_dkv (FusedOrTupleTensor, optional):
                partial remote dkv in float32 dtype, or ``None`` if skipped
            partial_local_dkv (FusedOrTupleTensor): partial local dkv to be reduced
            ref_remote_kv (FusedOrTupleTensor):
                reference remote kv fused or tupled tensor, to provide meta info like dtype and shape
            overlap_stage (int): given remote overlap stage
        """

        # reduce ith partial dkv
        (
            self.partial_dkv_reduce_work_per_stage[overlap_stage]
        ) = self._reduce_partial_dkv(
            partial_remote_dkv=partial_remote_dkv,
            partial_local_dkv=partial_local_dkv,
            ref_remote_dkv=ref_remote_kv,
            overlap_stage=overlap_stage,
            buffer_name=GrpCollBufferName.GroupReduceDefault,
            kernel_barrier=kernel_barrier,
        )

        # reduce ith partial dq
        ref_remote_q, _, _ = self._maybe_chunk(ref_remote_qo_do, num_chunks=3)
        (
            self.partial_dq_reduce_work_per_stage[overlap_stage]
        ) = self._reduce_partial_dq(
            partial_remote_dq=partial_remote_dq,
            partial_local_dq=partial_local_dq,
            ref_remote_dq=ref_remote_q,
            overlap_stage=overlap_stage,
            buffer_name=GrpCollBufferName.GroupReduceQO,
            kernel_barrier=kernel_barrier,
        )

    @nvtx.instrument_nvtx
    def reduce_partial_dsink(
        self,
        partial_global_dsink: torch.Tensor | None,
    ) -> None:
        """
        Reduce partial global dsink to replicated global dsink
        and assign the returned work to self.partial_dsink_reduce_work

        Args:
            partial_global_dsink (torch.Tensor, optional):
                partial global dsink to be reduced if given
        """
        self.partial_dsink_reduce_work = self._reduce_partial_dsink(
            partial_global_dsink=partial_global_dsink,
        )

    @nvtx.instrument_nvtx
    def prepare_reduced_local_dqkv_global_dsink(
        self,
        partial_local_dq: torch.Tensor,
        partial_local_dkv: FusedOrTupleTensor,
        partial_global_dsink: torch.Tensor | None,
        ref_local_dq: torch.Tensor,
        ref_local_dkv: FusedOrTupleTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Prepare the final reduced local dq,dk,dv before returning from backward
        with clearing the temporary work list

        Args:
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            partial_local_dkv (FusedOrTupleTensor):
                partial local dkv fused or tupled tensor to be reduced
            partial_global_dsink (torch.Tensor, optional):
                partial global dsink to be reduced if given

            ref_local_dq (torch.Tensor):
                reference local dq, to provide meta info like dtype and shape
            ref_local_dkv (FusedOrTupleTensor):
                reference local dkv fused or tupled tensor, to provide meta info like dtype and shape

        Returns:
            local_dq (torch.Tensor): reduced local dq tensor
            local_dk (torch.Tensor): reduced local dk tensor
            local_dv (torch.Tensor): reduced local dv tensor
            global_dsink (torch.Tensor, optional):
                reduced global dsink tensor (replicated among cp ranks) if required
        """
        # wait for all partial dq reduced
        for partial_dq_reduce_work in self.partial_dq_reduce_work_per_stage:
            partial_local_dq = partial_dq_reduce_work.wait_post_process(
                partial_local_dq
            )

        # cast final local dq to ref dtype
        local_dq = partial_local_dq.to(ref_local_dq.dtype)

        # wait for all partial dkv reduced
        for partial_dkv_reduce_work in self.partial_dkv_reduce_work_per_stage:
            partial_local_dkv = partial_dkv_reduce_work.wait_post_process(
                partial_local_dkv
            )

        # cast final local dkv to kv dtype and chunk to dk and dv
        if self.concat_kv:  # ref_local_dkv is a fused tensor
            kv_dtype = ref_local_dkv.dtype
        else:  # ref_local_dkv are tupled tensors
            kv_dtype = ref_local_dkv[0].dtype

        if self.concat_dkv:
            local_dkv = partial_local_dkv.to(kv_dtype)
            local_dk, local_dv = self._maybe_chunk(local_dkv, num_chunks=2)
        else:
            local_dk, local_dv = self._maybe_chunk(partial_local_dkv, num_chunks=2)
            local_dk = local_dk.to(kv_dtype)
            local_dv = local_dv.to(kv_dtype)

        # wait for partial global dsink reduced
        global_dsink = self.partial_dsink_reduce_work.wait_post_process(
            partial_global_dsink
        )

        # maybe unflatten head groups of local dq/dkv
        local_dq, local_dk, local_dv = self._maybe_unflatten_local_dqkv_head_groups(
            local_dq=local_dq,
            local_dk=local_dk,
            local_dv=local_dv,
        )

        # reset the temporary work list
        self._reset_work_list()

        return local_dq, local_dk, local_dv, global_dsink

    def load_tensors_from_fwd(
        self, ctx
    ) -> tuple[
        torch.Tensor,
        FusedOrTupleTensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        FusedOrTupleTensor | None,
        torch.Tensor | None,
    ]:
        if self.save_last_stage_for_backward:
            if self.concat_kv:  # local kv is a fused tensor
                (
                    local_q,
                    local_kv,
                    local_out,
                    local_lse,
                    last_q,
                    last_kv,
                    global_sink,
                ) = ctx.saved_tensors
            else:  # local kv are tupled tensors
                (
                    local_q,
                    local_k,
                    local_v,
                    local_out,
                    local_lse,
                    last_q,
                    last_k,
                    last_v,
                    global_sink,
                ) = ctx.saved_tensors
                local_kv = (local_k, local_v)
                last_kv = (last_k, last_v)
        else:
            last_q, last_kv = None, None
            if self.concat_kv:  # local kv is a fused tensor
                local_q, local_kv, local_out, local_lse, global_sink = ctx.saved_tensors
            else:  # local kv are tupled tensors
                (
                    local_q,
                    local_k,
                    local_v,
                    local_out,
                    local_lse,
                    global_sink,
                ) = ctx.saved_tensors
                local_kv = (local_k, local_v)

        return local_q, local_kv, local_out, local_lse, last_q, last_kv, global_sink

    # ----------    common API   --------- #

    @property
    def deterministic(self) -> bool:
        return magi_attention.is_deterministic_mode_enable()

    @property
    def prefetch_stage_by_stage(self) -> bool:
        """
        NOTE:
        1. When CUDA_DEVICE_MAX_CONNECTIONS == 1, prefetch must be done stage-by-stage to avoid blocking
           the FFA forward/backward computation; otherwise only the last stage's prefetch can overlap with
           computation.
        2. When native grpcoll is enabled, prefetch must also be done stage-by-stage to avoid blocking
           the FFA forward/backward computation; otherwise only the last stage's prefetch can overlap with
           computation (unless allocating many grpcoll buffers, which is very memory intensive because each
           grpcoll_buffer's memory is managed separately).
        """
        return (
            magi_attention.is_cuda_device_max_connections_one()
            or magi_attention.comm.is_native_grpcoll_enable()
        )

    @property
    def use_sdpa_backend(self) -> bool:
        return magi_attention.is_sdpa_backend_enable()

    @property
    def use_fa4_backend(self) -> bool:
        return magi_attention.is_fa4_backend_enable()

    @property
    def use_native_grpcoll(self) -> bool:
        return magi_attention.comm.is_native_grpcoll_enable()

    @property
    def enable_qo_comm(self) -> bool:
        return magi_attention.comm.is_qo_comm_enable()

    @property
    def flatten_head_groups(self) -> bool:
        return magi_attention.is_flatten_head_groups_enable()

    @property
    def hp_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def fwd_sm_margin(self) -> int:
        """
        Get the forward sm_margin reserved for communication.

        1. When native grpcoll is enabled, a kernel barrier guarantees the correct ordering
           between communication and compute kernels, so no additional sm_margin is required;
           return 0.
        2. Otherwise, return the saved sm_margin for communication to allow communication to
           properly overlap with computation.
        """
        if magi_attention.comm.is_native_grpcoll_enable():
            return 0
        else:
            return magi_attention.comm.ffa_fwd_sm_margin_save_for_comm()

    @property
    def bwd_sm_margin(self) -> int:
        """
        Get the backward sm_margin reserved for communication.

        1. When native grpcoll is enabled, a kernel barrier guarantees the correct ordering
           between communication and compute kernels, so no additional sm_margin is required;
           return 0.
        2. Otherwise, return the saved sm_margin for communication to allow communication to
           properly overlap with computation.
        """

        if magi_attention.comm.is_native_grpcoll_enable():
            return 0
        else:
            return magi_attention.comm.ffa_bwd_sm_margin_save_for_comm()

    @property
    def fwd_hp_reduce(self) -> bool:
        return magi_attention.comm.is_fwd_high_precision_reduce_enable()

    @property
    def bwd_hp_reduce(self) -> bool:
        return magi_attention.comm.is_bwd_high_precision_reduce_enable()

    @property
    def dsink_reduce_op(self) -> ReduceOp | None:
        return {
            "none": None,
            "sum": ReduceOp.SUM,
            "avg": ReduceOp.AVG,
        }[magi_attention.comm.dsink_all_reduce_op()]

    @property
    def fwd_kernel_barrier_fetch_target(self) -> int:
        """The target number the kernel barrier should wait for during forward fetch"""
        if self.cp_group_gc.size() == 1 or not self.use_native_grpcoll:
            return 0

        if self.enable_qo_comm:
            return 2

        return 1

    @property
    def fwd_kernel_barrier_reduce_target(self) -> int:
        """The target number the kernel barrier should wait for during forward reduce"""
        if self.cp_group_gc.size() == 1 or not self.use_native_grpcoll:
            return 0

        if self.enable_qo_comm:
            return 1

        return 0

    @property
    def bwd_kernel_barrier_fetch_target(self) -> int:
        """The target number the kernel barrier should wait for during backward fetch"""
        if self.cp_group_gc.size() == 1 or not self.use_native_grpcoll:
            return 0

        if self.enable_qo_comm:
            return 2

        return 1

    @property
    def bwd_kernel_barrier_reduce_target(self) -> int:
        """The target number the kernel barrier should wait for during backward reduce"""
        if self.cp_group_gc.size() == 1 or not self.use_native_grpcoll:
            return 0

        if self.enable_qo_comm:
            return 2

        return 1

    def is_host_stage(self, overlap_stage: int | None) -> bool:
        """
        Check if the given overlap stage is the host stage
        """
        return overlap_stage is None

    def is_last_remote_stage(self, overlap_stage: int | None) -> bool:
        """
        Check if the given overlap stage is the last remote stage
        """
        return self.get_next_stage(overlap_stage) == self.overlap_degree

    def is_first_remote_stage(self, overlap_stage: int) -> bool:
        """
        Check if the given overlap stage is the first remote stage
        """
        return overlap_stage == 0

    def is_penultimate_stage(self, overlap_stage: int | None) -> bool:
        """
        Check if the given overlap stage is the penultimate stage
        """
        return self.get_next_stage(overlap_stage) == self.overlap_degree - 1

    def get_next_stage(self, overlap_stage: int | None) -> int:
        """
        Get the next overlap stage
        """
        return 0 if overlap_stage is None else overlap_stage + 1

    def get_curr_remote_stage(self, overlap_stage: int | None) -> int:
        """
        Get the current remote overlap stage
        mostly to avoid mypy typing error
        """
        assert not self.is_host_stage(
            overlap_stage
        ), "The overlap_stage is None, thus the current stage is the host stage."
        return self.get_next_stage(overlap_stage) - 1

    # ----------    internal API   --------- #

    def _launch_attn_fwd_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None,
        out_acc: torch.Tensor | None,
        lse_acc: torch.Tensor | None,
        max_logits_acc: torch.Tensor | None,
        attn_arg: AttnArg,
        softmax_scale: float,
        softcap: float,
        is_host_stage: bool,
        return_max_logits: bool = False,
    ) -> tuple[torch.Tensor, AttnForwardMeta]:
        if return_max_logits:
            assert (
                not self.use_fa4_backend
            ), "FA4 backend does not support return max logits"
        with nvtx.add_nvtx_event(
            f"attn-fwd: "
            f"{attn_arg.total_area=} | "
            f"{attn_arg.q_ranges=} | "
            f"{attn_arg.k_ranges=}"
        ):
            if self.use_sdpa_backend:
                partial_out, meta = sdpa_fwd(
                    q=q,
                    k=k,
                    v=v,
                    # NOTE: sink token needs to be applied only once
                    # thus we only apply it at the host stage if not skipped
                    sink=sink if is_host_stage else None,
                    attn_arg=attn_arg,
                    softmax_scale=softmax_scale,
                    softcap=softcap,
                    sink_layout="sh",
                    return_max_logits=return_max_logits,
                )
                if return_max_logits and max_logits_acc is not None:
                    assert meta.max_logits is not None
                    torch.maximum(max_logits_acc, meta.max_logits, out=max_logits_acc)
                    meta.max_logits = max_logits_acc
            elif self.use_fa4_backend:
                partial_out, partial_lse = fa4_fwd(
                    q=q,
                    k=k,
                    v=v,
                    # NOTE: sink token needs to be applied only once
                    # thus we only apply it at the host stage if not skipped
                    sink=sink if is_host_stage else None,
                    attn_arg=attn_arg,
                    softmax_scale=softmax_scale,
                    softcap=softcap,
                    sink_layout="sh",
                )
                meta = AttnForwardMeta(lse=partial_lse, max_logits=None)
            else:
                partial_out, meta = _flex_flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    # NOTE: sink token needs to be applied only once
                    # thus we only apply it at the host stage if not skipped
                    sink=sink if is_host_stage else None,
                    sink_layout="sh",
                    out=out_acc,  # directly reduce to out_acc
                    lse=lse_acc,  # directly reduce to lse_acc
                    **attn_arg.to_ffa_args(is_bwd=False),
                    softmax_scale=softmax_scale,
                    softcap=softcap,
                    # NOTE: always use high-precision for the partial out,
                    # to reduce the error caused by the out/lse correction
                    out_type=self.hp_dtype,
                    # NOTE: when using accumulative buffer, we need to always enable atomic reduction
                    # unless it is the first call when accumulative buffer is still None
                    disable_fwd_atomic_reduction=(
                        attn_arg.disable_fwd_atomic_reduction and out_acc is None
                    ),
                    deterministic=self.deterministic,
                    sm_margin=self.fwd_sm_margin,
                    # optional args below mainly for sparse attn
                    ref_block_size=None,
                    max_seqlen_q=None,
                    auto_range_merge=magi_attention.is_auto_range_merge_enable(),
                    swap_ab=False,
                    pack_gqa=False,
                    sparse_load=False,
                    sparse_load_loop_count=None,
                    sparse_load_invalid_count=None,
                    equal_k_range_size=None,
                    return_max_logits=return_max_logits,
                    max_logits=max_logits_acc,  # directly reduce to max_logits_acc
                )

        return partial_out, meta

    def _launch_attn_bwd_kernel(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
        dq_acc: torch.Tensor | None,
        attn_arg: AttnArg,
        softmax_scale: float,
        softcap: float,
        is_host_stage: bool,
        dkv_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, FusedOrTupleTensor, torch.Tensor | None]:
        if self.use_sdpa_backend:
            partial_dq, partial_dk, partial_dv, partial_dsink = sdpa_bwd(
                do=do,
                q=q,
                k=k,
                v=v,
                # NOTE: dsink should be computed only once
                # thus we only compute it at the host stage if not skipped
                sink=sink if is_host_stage else None,
                o=o,
                lse=lse,
                attn_arg=attn_arg,
                softmax_scale=softmax_scale,
                softcap=softcap,
                sink_layout="sh",
            )
            partial_dkv = self._maybe_concat(
                partial_dk, partial_dv, need_concat=self.concat_dkv
            )
        elif self.use_fa4_backend:
            partial_dq, partial_dk, partial_dv, partial_dsink = fa4_bwd(
                do=do,
                q=q,
                k=k,
                v=v,
                # NOTE: dsink should be computed only once
                # thus we only compute it at the host stage if not skipped
                sink=sink if is_host_stage else None,
                o=o,
                lse=lse,
                attn_arg=attn_arg,
                softmax_scale=softmax_scale,
                softcap=softcap,
                sink_layout="sh",
                deterministic=self.deterministic,
            )
            partial_dkv = self._maybe_concat(
                partial_dk, partial_dv, need_concat=self.concat_dkv
            )
        else:
            # init partial_dkv buffer
            # NOTE: we initial partial dkv and chunk to dk, dv to avoid concat them back before return
            # and we need to zero-initialize partial_dkv since it needs to be reduced
            partial_dkv = torch.zeros(
                dkv_shape,
                dtype=self.hp_dtype,
                device=k.device,
            )
            partial_dk, partial_dv = self._maybe_chunk(partial_dkv, num_chunks=2)

            (
                partial_dq,
                partial_dk,
                partial_dv,
                partial_dsink,
            ) = _flex_flash_attn_backward(
                dout=do,
                q=q,
                k=k,
                v=v,
                # NOTE: dsink should be computed only once
                # thus we only compute it at the host stage if not skipped
                sink=sink if is_host_stage else None,
                sink_layout="sh",
                out=o,
                lse=lse,
                dq=dq_acc,  # directly reduce to dq_acc
                dk=partial_dk,
                dv=partial_dv,
                dsink=None,  # let kernel initialize dsink if required
                **attn_arg.to_ffa_args(is_bwd=True),
                softmax_scale=softmax_scale,
                softcap=softcap,
                # NOTE: always use high precision for the partial dq, dkv
                # to reduce the error caused by the atomic reduction inside the kernel
                # FIXME: use real dq dk dv cast type instead of always using hp_dtype
                dq_type=self.hp_dtype,
                dk_type=self.hp_dtype,
                dv_type=self.hp_dtype,
                # NOTE: we enable disable_bwd_dkv_atomic_reduction only with MHA or CatGQA enabled
                # and attn_arg.disable_bwd_dkv_atomic_reduction only considers
                # whether k_ranges is non-overlapped and sorted
                # TODO: add CatGQA flag here
                disable_bwd_dkv_atomic_reduction=(
                    attn_arg.disable_bwd_dkv_atomic_reduction
                    and self.comm_meta.num_heads_per_group == 1
                ),
                deterministic=self.deterministic,
                sm_margin=self.bwd_sm_margin,
                # optional args below mainly for sparse attn
                auto_range_merge=magi_attention.is_auto_range_merge_enable(),
                swap_bwd_qk_loop=False,
            )

            if not self.concat_dkv:  # make partial_dkv tupled tensors
                partial_dkv = (partial_dk, partial_dv)

        return partial_dq, partial_dkv, partial_dsink

    @nvtx.instrument_nvtx
    def _fetch_remote_kv(
        self,
        local_kv: FusedOrTupleTensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithBuffer:
        """
        Fetch remote kv buffer from other ranks to local for the given overlap stage,
        and return the corresponding work and buffer

        Args:
            local_kv (FusedOrTupleTensor): the local kv fused or tupled tensor
            overlap_stage (int): current overlap stage

        Returns:
            WorkWithBuffer:
                remote_kv_work (WorkWithPostProcessFn):
                    communication handle, used to wait for communication completion
                remote_kv_buffer (FusedOrTupleTensor,): remote kv buffer

        Shape:
            local_kv: [num_tokens_kv_local*2, num_heads_kv, head_dim]
            remote_kv_buffer: [num_tokens_kv_remote_i*2, num_heads_kv, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        # prepare the meta info
        if self.concat_kv:
            _, num_heads, head_dim = local_kv.shape
            dtype = local_kv.dtype
            device = local_kv.device
        else:
            _, num_heads, head_dim = local_kv[0].shape
            dtype = local_kv[0].dtype
            device = local_kv[0].device

        # get the group-cast args for kv
        group_cast_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_kv_seqlen = self.comm_meta.num_remote_kv_tokens_per_stage[overlap_stage]
        if not self.concat_kv:
            remote_kv_seqlen *= 2  # still x2 to allocate once

        # init remote kv buffer
        remote_kv_buffer = torch.empty(
            remote_kv_seqlen,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        if not self.concat_kv:  # chunk to k,v individual buffers
            remote_kv_buffer = self._maybe_chunk(remote_kv_buffer, num_chunks=2)

        # launch group cast kernel
        remote_kv_work = group_cast(
            input=local_kv,
            output=remote_kv_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
            buffer_name=buffer_name,
            kernel_barrier=kernel_barrier,
        )

        return remote_kv_work, remote_kv_buffer

    @nvtx.instrument_nvtx
    def _fetch_remote_q(
        self,
        local_q: torch.Tensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithBuffer:
        """
        Fetch remote q buffer from other ranks to local for the given overlap stage,
        and return the corresponding work and buffer

        Args:
            local_q (torch.Tensor): the local q tensor
            overlap_stage (int): current overlap stage

        Returns:
            WorkWithBuffer:
                remote_q_work (WorkWithPostProcessFn):
                    communication handle, used to wait for communication completion
                remote_q_buffer (torch.Tensor): remote q buffer

        Shape:
            local_q: [num_tokens_q_local, num_heads_q, head_dim]
            remote_q_buffer: [num_tokens_q_remote_i, num_heads_q, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        if not self.enable_qo_comm:
            remote_q_buffer = local_q
            remote_q_work = WorkWithPostProcessFn(post_process_fn=lambda x: x)
            return remote_q_work, remote_q_buffer

        _, num_heads, head_dim = local_q.shape

        # get the group-cast args for q
        group_cast_args = self.comm_meta.qo_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_q_seqlen = self.comm_meta.num_remote_qo_tokens_per_stage[overlap_stage]

        # init remote q buffer
        remote_q_buffer = torch.empty(
            remote_q_seqlen,
            num_heads,
            head_dim,
            dtype=local_q.dtype,
            device=local_q.device,
        )

        # launch group cast kernel
        remote_q_work = group_cast(
            input=local_q,
            output=remote_q_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
            buffer_name=buffer_name,
            kernel_barrier=kernel_barrier,
        )

        return remote_q_work, remote_q_buffer

    @nvtx.instrument_nvtx
    def _fetch_remote_qo_do_lse(
        self,
        local_qo_do: FusedOrTupleTensor,
        local_lse: torch.Tensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithBuffer:
        """
        Fetch remote q, o, do, lse buffer from other ranks to local for the given overlap stage,
        and return the corresponding work and buffer

        Args:
            local_qo_do (FusedOrTupleTensor): the local q, o, do (fused or tupled) tensor
            local_lse (torch.Tensor): the local lse tensor
            overlap_stage (int): current overlap stage

        Returns:
            WorkWithBuffer:
                remote_qo_do_lse_work (WorkWithPostProcessFn):
                    communication handle, used to wait for communication completion
                remote_qo_do_lse_buffer:
                    - remote_qo_do_buffer (FusedOrTupleTensor): remote q, o, do buffer
                    - remote_lse_buffer (torch.Tensor): remote lse buffer

        Shape:
            local_qo_do: [num_tokens_qo_do_local*3, num_heads_q, head_dim]
            local_lse: [num_tokens_q_local, num_heads_q]
            remote_qo_do_lse_buffer:
                - remote_qo_do_buffer: [num_tokens_q_remote_i*3, num_heads_q, head_dim],
                    for i = 0, 1, ..., overlap_degree - 1
                - remote_lse_buffer: [num_tokens_q_remote_i, num_heads_q],
                    for i = 0, 1, ..., overlap_degree - 1
        """

        if not self.enable_qo_comm:
            remote_qo_do_lse_buffer = (local_qo_do, local_lse)
            remote_qo_do_lse_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take q,o,do,lse and return q,o,do,lse
            )
            return remote_qo_do_lse_work, remote_qo_do_lse_buffer

        # prepare the meta info
        if self.concat_qo_do:  # local_qo_do is a fused tensor
            _, num_heads, head_dim = local_qo_do.shape
            dtype = local_qo_do.dtype
            device = local_qo_do.device
        else:  # local_qo_do are tupled tensors
            _, num_heads, head_dim = local_qo_do[0].shape
            dtype = local_qo_do[0].dtype
            device = local_qo_do[0].device

        if self.use_native_grpcoll:
            assert (
                not self.concat_qo_do
            ), "Only support native grpcoll without concating q,o,do"

            # get the group-cast args
            group_cast_args = self.comm_meta.qo_group_collective_args_list[
                overlap_stage
            ].to_group_cast_args()
            remote_lse_seqlen = self.comm_meta.num_remote_qo_tokens_per_stage[
                overlap_stage
            ]
            remote_qo_do_seqlen = remote_lse_seqlen * 3

            # init remote lse buffer
            remote_lse_buffer = torch.empty(
                (remote_lse_seqlen, num_heads),
                dtype=self._maybe_hp_dtype(
                    dtype, need_hp_dtype=True
                ),  # lse always in high-precision
                device=device,
            )

            # init remote qo_do buffers
            remote_qo_do_buffer = torch.empty(
                (remote_qo_do_seqlen, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            remote_qo_do_buffer = self._maybe_chunk(remote_qo_do_buffer, num_chunks=3)

            # launch group cast kernel
            remote_qo_do_lse_work = group_cast(
                input=local_qo_do,
                output=remote_qo_do_buffer,
                **group_cast_args,
                group=self.cp_group_gc,
                async_op=True,
                cast_lse=True,
                input_lse=local_lse,
                output_lse=remote_lse_buffer,
                buffer_name=buffer_name,
                kernel_barrier=kernel_barrier,
            )

            # pack the buffers for qo_do and lse together
            remote_qo_do_lse_buffer = (remote_qo_do_buffer, remote_lse_buffer)
        else:
            # HACK: since lse usually has different shape and dtype from q,o,do
            # and we concat the q,o,do along the seqlen dim for non-native grpcoll,
            # resulting in different seqlen between q,o,do and lse as well,
            # we can not trivially fuse lse comm with q,o,do comm,
            # thus here we use specific comm to fetch lse

            # -------   for lse   ------- #

            # get the group-cast args for lse
            group_cast_args_lse = self.comm_meta.qo_group_collective_args_list[
                overlap_stage
            ].to_group_cast_args()
            remote_lse_seqlen = self.comm_meta.num_remote_qo_tokens_per_stage[
                overlap_stage
            ]

            # init remote lse buffer
            remote_lse_buffer = torch.empty(
                (remote_lse_seqlen, num_heads),
                dtype=self._maybe_hp_dtype(
                    dtype, need_hp_dtype=True
                ),  # lse always in high-precision
                device=device,
            )

            # launch group cast kernel for lse
            remote_lse_work = group_cast(
                input=local_lse,
                output=remote_lse_buffer,
                **group_cast_args_lse,
                group=self.cp_group_gc,
                async_op=True,
                buffer_name=buffer_name,
            )

            # -------   for q,o,do   ------- #

            # get the group-cast args for q,o,do
            group_cast_args_qo_do = self.comm_meta.qo_do_group_collective_args_list[
                overlap_stage
            ].to_group_cast_args()
            remote_qo_do_seqlen = self.comm_meta.num_remote_qo_do_tokens_per_stage[
                overlap_stage
            ]

            # init remote q,o,do output buffer
            remote_qo_do_buffer = torch.empty(
                (remote_qo_do_seqlen, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )

            # launch group cast kernel for qo_do
            assert isinstance(local_qo_do, torch.Tensor)  # mypy
            remote_qo_do_work = group_cast(
                input=local_qo_do,
                output=remote_qo_do_buffer,
                **group_cast_args_qo_do,
                group=self.cp_group_gc,
                async_op=True,
                buffer_name=buffer_name,
            )

            # pack the works for qo_do and lse together
            remote_qo_do_lse_work = WorkWithPostProcessFn(
                work=GeneralWork([remote_qo_do_work.work, remote_lse_work.work]),
                post_process_fn=lambda x: (  # take qo_do, lse and return qo_do, lse
                    remote_qo_do_work.post_process_fn(x[0]),
                    remote_lse_work.post_process_fn(x[1]),
                ),
                async_op=True,
            )

            # pack the buffers for qo_do and lse together
            remote_qo_do_lse_buffer = (remote_qo_do_buffer, remote_lse_buffer)

        return remote_qo_do_lse_work, remote_qo_do_lse_buffer

    @nvtx.instrument_nvtx
    def _reduce_partial_out_lse(
        self,
        partial_remote_out: torch.Tensor | None,
        partial_remote_lse: torch.Tensor | None,
        partial_local_out: torch.Tensor,
        partial_local_lse: torch.Tensor,
        ref_remote_out: torch.Tensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote out and lse to lse-reduce to local out and lse for the given overlap stage,
        and return the corresponding work

        Args:
            partial_remote_out (torch.Tensor, optional):
                partial remote out in float32 dtype, or ``None`` if skipped
            partial_remote_lse (torch.Tensor, optional):
                partial remote lse in float32 dtype, or ``None`` if skipped
            partial_local_out (torch.Tensor): partial local out to be reduced
            partial_local_lse (torch.Tensor): partial local lse to be reduced
            ref_remote_out (torch.Tensor):
                reference remote out, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_out_lse_reduce_work (WorkWithPostProcessFn): partial out and lse group-reduce work
        """
        if self.enable_qo_comm:
            # get the group-reduce args for out and lse
            if self.use_native_grpcoll:  # just the same as original qo args
                group_collective_args_list = (
                    self.comm_meta.qo_group_collective_args_list
                )
            else:  # the specific args for out and lse
                group_collective_args_list = (
                    self.comm_meta.out_lse_group_collective_args_list  # type: ignore[assignment]
                )

            group_reduce_args = group_collective_args_list[
                overlap_stage
            ].to_group_reduce_args()

            # init remote out/lse buffer
            if partial_remote_out is None:
                # skipped for this rank, but still reduced from other ranks
                partial_remote_out = torch.empty_like(
                    ref_remote_out,
                    dtype=self._maybe_hp_dtype(
                        ref_remote_out.dtype,
                        # out always in high-precision if using native grpcoll
                        need_hp_dtype=(self.use_native_grpcoll or self.fwd_hp_reduce),
                    ),
                    device=ref_remote_out.device,
                )
                partial_remote_lse = torch.empty(
                    (ref_remote_out.size(0), ref_remote_out.size(1)),
                    dtype=self._maybe_hp_dtype(
                        ref_remote_out.dtype,
                        need_hp_dtype=True,  # lse always in high-precision
                    ),
                    device=ref_remote_out.device,
                )
            elif not self.use_native_grpcoll and not self.fwd_hp_reduce:
                # downcast to the same dtype as out
                # if using non-native grpcoll and not reduce in high-precision
                partial_remote_out = partial_remote_out.to(ref_remote_out.dtype)

            # init some additional kwargs for native grpcoll
            partial_out_lse_reduce_kwargs: dict[str, Any] = {}
            if self.use_native_grpcoll:
                partial_out_lse_reduce_kwargs.update(
                    acc_reduce=True,
                    reduce_op="lse",
                    comm_dtype=self._maybe_hp_dtype(
                        ref_remote_out.dtype, self.fwd_hp_reduce
                    ),
                )

            # launch group-reduce kernel
            partial_out_lse_reduce_work = group_reduce(
                input=partial_remote_out,
                input_lse=partial_remote_lse,
                output=partial_local_out,
                output_lse=partial_local_lse,
                **group_reduce_args,
                group=self.cp_group_gr,
                async_op=True,
                **partial_out_lse_reduce_kwargs,
                buffer_name=buffer_name,
                kernel_barrier=kernel_barrier,
            )
        else:
            if not self.fwd_out_lse_use_acc and partial_remote_out is not None:
                # NOTE: the partial remote out and lse have NOT been reduced to
                # partial local out and lse by neither ffa fwd kernel nor group-reduce
                # thus we need to manually reduce here
                correct_attn_out_lse(
                    out1=partial_local_out,
                    lse1=partial_local_lse,
                    out2=partial_remote_out,
                    lse2=partial_remote_lse,
                    inplace=True,  # inplace reduce to the partial local out and lse
                )

            partial_out_lse_reduce_work = WorkWithPostProcessFn(
                post_process_fn=lambda *x: x  # take out, lse and return out, lse
            )

        return partial_out_lse_reduce_work

    @nvtx.instrument_nvtx
    def _reduce_partial_dkv(
        self,
        partial_remote_dkv: FusedOrTupleTensor | None,
        partial_local_dkv: FusedOrTupleTensor,
        ref_remote_dkv: FusedOrTupleTensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote dkv to local dkv for the given overlap stage

        Args:
            partial_remote_dkv (FusedOrTupleTensor, optional):
                partial remote dkv fused or tupled tensor, or ``None`` if skipped
            partial_local_dkv (FusedOrTupleTensor):
                partial local dkv fused or tupled tensor to be reduced
            ref_remote_dkv (FusedOrTupleTensor):
                reference remote dkv fused or tupled tensor, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_dkv_reduce_work (WorkWithPostProcessFn): partial dkv group-reduce work
        """

        # prepare the meta info
        if self.concat_kv:  # ref_remote_dkv is a fused tensor
            dtype = ref_remote_dkv.dtype
            device = ref_remote_dkv.device
            remote_dkv_shape = ref_remote_dkv.shape
        else:  # ref_remote_dkv are tupled tensors
            dtype = ref_remote_dkv[0].dtype
            device = ref_remote_dkv[0].device
            remote_dkv_shape = (
                ref_remote_dkv[0].shape[0] * 2,
                *ref_remote_dkv[0].shape[1:],
            )

        # get the group-reduce args for dkv
        group_reduce_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_reduce_args()

        # init remote dkv buffer(s)
        if partial_remote_dkv is None:
            # skipped for this rank, but still reduced from other ranks
            partial_remote_dkv = torch.empty(
                remote_dkv_shape,
                dtype=self._maybe_hp_dtype(
                    dtype,
                    # dkv always in high-precision if using native grpcoll
                    # unless using fa4 backend which only supports fp16/bf16 for now
                    need_hp_dtype=(not self.use_fa4_backend)
                    and (self.use_native_grpcoll or self.bwd_hp_reduce),
                ),
                device=device,
            )
            if not self.concat_dkv:
                partial_remote_dkv = self._maybe_chunk(partial_remote_dkv, num_chunks=2)
        elif not self.use_native_grpcoll and not self.bwd_hp_reduce:
            assert self.concat_dkv
            # downcast to the same dtype as dkv
            # if using non-native grpcoll and not reduce in high-precision
            partial_remote_dkv = partial_remote_dkv.to(dtype)

        # init some additional kwargs for native grpcoll
        partial_dkv_reduce_kwargs: dict[str, Any] = {}
        if self.use_native_grpcoll:
            partial_dkv_reduce_kwargs.update(
                acc_reduce=True,
                reduce_op="sum",
                comm_dtype=self._maybe_hp_dtype(dtype, self.bwd_hp_reduce),
            )

        # launch group-reduce kernel
        partial_dkv_reduce_work = group_reduce(
            input=partial_remote_dkv,
            output=partial_local_dkv,
            **group_reduce_args,
            group=self.cp_group_gr,
            async_op=True,
            **partial_dkv_reduce_kwargs,
            buffer_name=buffer_name,
            kernel_barrier=kernel_barrier,
        )

        return partial_dkv_reduce_work

    @nvtx.instrument_nvtx
    def _reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor | None,
        partial_local_dq: torch.Tensor,
        ref_remote_dq: torch.Tensor,
        overlap_stage: int,
        buffer_name: GrpCollBufferName | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote dq to local dq for the given overlap stage

        Args:
            partial_remote_dq (torch.Tensor, optional):
                partial remote dq in float32 dtype, or ``None`` if skipped
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            ref_remote_dq (torch.Tensor):
                reference remote dq, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_dq_reduce_work (WorkWithPostProcessFn): partial dq group-reduce work
        """
        if self.enable_qo_comm:
            # get the group-reduce args for dq
            group_reduce_args = self.comm_meta.qo_group_collective_args_list[
                overlap_stage
            ].to_group_reduce_args()

            # init remote dq buffer
            if partial_remote_dq is None:
                # skipped for this rank, but still reduced from other ranks
                partial_remote_dq = torch.empty_like(
                    ref_remote_dq,
                    dtype=self._maybe_hp_dtype(
                        ref_remote_dq.dtype,
                        # dq always in high-precision if using native grpcoll
                        # unless using fa4 backend which only supports fp16/bf16 for now
                        need_hp_dtype=(not self.use_fa4_backend)
                        and (self.use_native_grpcoll or self.bwd_hp_reduce),
                    ),
                )
            elif not self.use_native_grpcoll and not self.bwd_hp_reduce:
                # downcast to the same dtype as dq
                # if using non-native grpcoll and not reduce in high-precision
                partial_remote_dq = partial_remote_dq.to(ref_remote_dq.dtype)

            # init some additional kwargs for native grpcoll
            partial_dq_reduce_kwargs: dict[str, Any] = {}
            if self.use_native_grpcoll:
                partial_dq_reduce_kwargs.update(
                    acc_reduce=True,
                    reduce_op="sum",
                    comm_dtype=self._maybe_hp_dtype(
                        ref_remote_dq.dtype, self.bwd_hp_reduce
                    ),
                )

            # launch group-reduce kernel
            partial_dq_reduce_work = group_reduce(
                input=partial_remote_dq,
                output=partial_local_dq,
                **group_reduce_args,
                group=self.cp_group_gr,
                async_op=True,
                **partial_dq_reduce_kwargs,
                buffer_name=buffer_name,
                kernel_barrier=kernel_barrier,
            )
        else:
            if not self.bwd_dq_use_acc and partial_remote_dq is not None:
                # NOTE: the partial remote dq has NOT been reduced to partial local dq
                # by neither ffa bwd kernel nor group-reduce
                # thus we need to manually reduce here
                partial_local_dq.add_(partial_remote_dq)

            partial_dq_reduce_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take dq and return dq
            )

        return partial_dq_reduce_work

    @nvtx.instrument_nvtx
    def _reduce_partial_dsink(
        self,
        partial_global_dsink: torch.Tensor | None,
    ) -> WorkWithPostProcessFn:
        """
        Reduce partial global dsink to replicated global dsink

        Args:
            partial_global_dsink (torch.Tensor, optional):
                partial global dsink to be reduced if given

        Returns:
            partial_dsink_reduce_work (WorkWithPostProcessFn):
                partial global dsink all-reduce work if required
        """
        if partial_global_dsink is not None:
            if (op := self.dsink_reduce_op) is not None:  # required to reduce
                work = dist.all_reduce(
                    partial_global_dsink,
                    op=op,
                    group=self.cp_group_gc,
                    async_op=True,
                )
                partial_dsink_reduce_work = WorkWithPostProcessFn(
                    work=GeneralWork(work),
                    post_process_fn=lambda x: x,  # take partial dsink and return in-place reduced dsink
                    async_op=True,
                )
            else:  # let the caller handle the reduction
                warnings.warn(
                    "The dsink reduction is skipped by default "
                    "since usually the training framework will handle it automatically. "
                    "However, under the scenarios w/o any framework mechanism to reduce parameters across cp ranks, "
                    "you can set the env var `MAGI_ATTENTION_DSINK_ALL_REDUCE_OP` to `sum` "
                    "to let `magi_attention` apply reduction. Otherwise, you might need to reduce the dsink manually."
                    ""
                )
                partial_dsink_reduce_work = WorkWithPostProcessFn(
                    work=None,
                    post_process_fn=lambda x: x,  # take partial dsink and return partial dsink
                )
        else:
            partial_dsink_reduce_work = WorkWithPostProcessFn(
                work=None, post_process_fn=lambda *args, **kwargs: None  # return None
            )

        return partial_dsink_reduce_work

    def _init_out_lse_skipped_host_stage(
        self,
        q: torch.Tensor,
        sink: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: we can NOT use empty initialization here,
        # since we have nowhere to zero-fill it
        # when all attn computations are skipped for certain q range
        out = torch.zeros_like(
            q,
            dtype=self._maybe_hp_dtype(q.dtype, not self.fwd_local_out_lp_init),
            device=q.device,
        )

        if sink is not None:
            # in skipped host stage if sink is given,
            # we directly use lse_sink to initialize lse
            lse = calc_lse_sink_compiled(
                sink=sink,
                seqlen_q=q.size(0),
                sink_layout="sh",
            )
        else:
            lse = torch.full(
                (q.size(0), q.size(1)),  # [sq, nhq]
                fill_value=float("-inf"),
                dtype=self._maybe_hp_dtype(
                    q.dtype, need_hp_dtype=True
                ),  # lse always in high-precision
                device=q.device,
            )

        return out, lse

    def _init_max_logits_skipped_host_stage(
        self,
        q: torch.Tensor,
        return_max_logits: bool,
    ) -> torch.Tensor | None:
        if return_max_logits:
            return torch.full(
                (q.size(1),),  # [nhq]
                fill_value=float("-inf"),
                dtype=q.dtype,
                device=q.device,
            )
        return None

    def _init_dq_dkv_dsink_skipped_host_stage(
        self,
        qo_do: FusedOrTupleTensor,
        kv: FusedOrTupleTensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
    ) -> tuple[torch.Tensor, FusedOrTupleTensor, torch.Tensor | None]:
        q, o, do = self._maybe_chunk(qo_do, num_chunks=3)
        k, _ = self._maybe_chunk(kv, num_chunks=2)
        if self.concat_kv:  # kv is a fused tensor
            dkv_shape = kv.shape
        else:  # kv are tupled tensors
            dkv_shape = (k.shape[0] * 2, *k.shape[1:])

        # NOTE: if local_dq and local_dkv calculation are skipped,
        # we need to zero-initialize them since they might be reduced later
        dq = torch.zeros_like(
            q,
            dtype=self._maybe_hp_dtype(q.dtype, not self.bwd_local_dq_lp_init),
        )
        dkv = torch.zeros(
            dkv_shape,
            dtype=self._maybe_hp_dtype(k.dtype, not self.bwd_local_dkv_lp_init),
            device=k.device,
        )
        if not self.concat_dkv:  # make partial_dkv tupled tensors
            dkv = self._maybe_chunk(dkv, num_chunks=2)

        # in skipped host stage,
        # we directly calculate dsink here
        if sink is not None:
            dsink = sink_bwd_compiled(
                sink=sink,
                lse=lse,
                o=o,
                do=do,
                sink_layout="sh",
            )
        else:
            dsink = None

        return dq, dkv, dsink

    def _init_dq_skipped_host_stage(
        self,
        qo_do: FusedOrTupleTensor,
    ) -> torch.Tensor:
        q, _, _ = self._maybe_chunk(qo_do, num_chunks=3)

        # NOTE: if local_dq and local_dkv calculation are skipped,
        # we need to zero-initialize them since they might be reduced later
        dq = torch.zeros_like(
            q,
            dtype=self._maybe_hp_dtype(q.dtype, not self.bwd_local_dq_lp_init),
        )

        return dq

    def _init_dkv_skipped_host_stage(
        self,
        kv: FusedOrTupleTensor,
    ) -> FusedOrTupleTensor:
        k, _ = self._maybe_chunk(kv, num_chunks=2)
        if self.concat_kv:  # kv is a fused tensor
            dkv_shape = kv.shape
        else:  # kv are tupled tensors
            dkv_shape = (k.shape[0] * 2, *k.shape[1:])

        dkv = torch.zeros(
            dkv_shape,
            dtype=self._maybe_hp_dtype(k.dtype, not self.bwd_local_dkv_lp_init),
            device=k.device,
        )
        if not self.concat_dkv:  # make partial_dkv tupled tensors
            dkv = self._maybe_chunk(dkv, num_chunks=2)

        return dkv

    # TODO: unify this specific scheduling with the original one
    def _hide_tail_stage_reduce_backward(
        self, ctx, grad_output: torch.Tensor, *args
    ):  # pragma: no cover
        (
            local_q,
            local_kv,
            local_out,
            local_lse,
            last_stage_q,
            last_stage_kv,
            global_sink,
        ) = self.load_tensors_from_fwd(ctx)
        softmax_scale: float | None = ctx.softmax_scale
        softcap: float = ctx.softcap
        save_last_stage = magi_attention.dist_attn_backward_hide_tail_reduce()
        assert (
            not save_last_stage or not self.enable_qo_comm
        ), "save_last_stage and enable_qo_comm can not be both True"
        assert self.overlap_degree > 0, (
            f"when self.overlap_degree == 0, this branch should not be entered, "
            f"but got {self.overlap_degree=}"
        )

        kernel_barrier_fetch = KernelBarrier(self.bwd_kernel_barrier_fetch_target)
        kernel_barrier_reduce = KernelBarrier(self.bwd_kernel_barrier_reduce_target)

        # get local qo_do,kv,lse and pre-fetch qo_do,kv,lse for remote stage(s)
        (
            local_qo_do,
            local_kv,
            local_lse,
        ) = self.get_curr_qo_do_kv_lse_and_fetch_next(
            local_qo_do=(local_q, local_out, grad_output),
            local_kv=local_kv,
            local_lse=local_lse,
            overlap_stage=None,
            kernel_barrier=kernel_barrier_fetch,
        )

        if not self.is_penultimate_stage(None):
            kernel_barrier_fetch.synchronize()
        kernel_barrier_reduce.reset()

        # apply bwd partial attn with ith remote qo_do,kv,lse
        # overlapped with (i+1)th pre-fetch
        (
            partial_local_dq,
            partial_remote_dkv,
            _,  # partial_global_dsink
        ) = self.apply_bwd_partial_attn(
            qo_do=local_qo_do,
            kv=last_stage_kv,
            lse=local_lse,
            dq_acc=None,
            overlap_stage=self.overlap_degree - 1,
            softmax_scale=softmax_scale,
            softcap=softcap,
            sink=global_sink,
        )
        if partial_local_dq is None:
            partial_local_dq = self._init_dq_skipped_host_stage(local_qo_do)
        partial_local_dkv = self._init_dkv_skipped_host_stage(local_kv)

        # reduce ith partial dq,dkv
        # overlapped with (i+1)th bwd partial attn and maybe (i+2)th pre-fetch
        self.reduce_partial_dq_dkv(
            partial_remote_dq=None,
            partial_local_dq=partial_local_dq,
            ref_remote_qo_do=local_qo_do,
            partial_remote_dkv=partial_remote_dkv,
            partial_local_dkv=partial_local_dkv,
            ref_remote_kv=last_stage_kv,
            overlap_stage=self.overlap_degree - 1,
            kernel_barrier=kernel_barrier_reduce,
        )
        num_of_degree = self.overlap_degree - 1

        # loop into remote stages
        for ith_overlap_stage in range(num_of_degree):
            kernel_barrier_fetch.reset()

            # wait for ith remote qo_do,kv,lse prepared
            # and pre-fetch (i+1)th remote qo_do,kv,lse
            (
                curr_remote_qo_do,
                curr_remote_kv,
                curr_remote_lse,
            ) = self.get_curr_qo_do_kv_lse_and_fetch_next(
                local_qo_do=local_qo_do,
                local_kv=local_kv,
                local_lse=local_lse,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_fetch,
            )
            if not self.is_penultimate_stage(ith_overlap_stage):
                kernel_barrier_fetch.synchronize()

            kernel_barrier_reduce.synchronize()
            kernel_barrier_reduce.reset()

            # apply bwd partial attn with ith remote qo_do,kv,lse
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_dq,
                partial_remote_dkv,
                _,  # partial_global_dsink
            ) = self.apply_bwd_partial_attn(
                qo_do=curr_remote_qo_do,
                kv=curr_remote_kv,
                lse=curr_remote_lse,
                dq_acc=partial_local_dq if self.bwd_dq_use_acc else None,
                overlap_stage=ith_overlap_stage,
                softmax_scale=softmax_scale,
                softcap=softcap,
                sink=global_sink,
            )

            # reduce ith partial dq,dkv
            # overlapped with (i+1)th bwd partial attn and maybe (i+2)th pre-fetch
            self.reduce_partial_dq_dkv(
                partial_remote_dq=partial_remote_dq,
                partial_local_dq=partial_local_dq,
                ref_remote_qo_do=curr_remote_qo_do,
                partial_remote_dkv=partial_remote_dkv,
                partial_local_dkv=partial_local_dkv,
                ref_remote_kv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_reduce,
            )

        kernel_barrier_reduce.synchronize()
        (
            partial_host_dq,
            partial_host_dkv,
            partial_global_dsink,
        ) = self.apply_bwd_partial_attn(
            qo_do=local_qo_do,
            kv=local_kv,
            lse=local_lse,
            dq_acc=partial_local_dq if self.bwd_dq_use_acc else None,
            overlap_stage=None,
            softmax_scale=softmax_scale,
            softcap=softcap,
            sink=global_sink,
        )
        assert global_sink is None or partial_global_dsink is not None

        # reduce partial global dsink if required
        self.reduce_partial_dsink(
            partial_global_dsink=partial_global_dsink,
        )

        # if only one remote stage, num_of_degree = 0, get last remote stage work
        # else, get self.overlap_degree - 1 remote stage
        self.partial_dkv_reduce_work_per_stage[num_of_degree - 1]._wait_work()
        if not self.bwd_dq_use_acc and partial_host_dq is not None:
            partial_local_dq.add_(partial_host_dq)
        if partial_host_dkv is not None:
            if self.concat_dkv:
                partial_local_dkv.add_(partial_host_dkv)
            else:
                for local_dkv, host_dkv in zip(partial_local_dkv, partial_host_dkv):
                    local_dkv.add_(host_dkv)

        # prepare reduced local dq,dk,dv and maybe global dsink
        # before returning from backward
        (
            local_dq,
            local_dk,
            local_dv,
            global_dsink,
        ) = self.prepare_reduced_local_dqkv_global_dsink(
            partial_local_dq=partial_local_dq,
            partial_local_dkv=partial_local_dkv,
            partial_global_dsink=partial_global_dsink,
            ref_local_dq=local_q,
            ref_local_dkv=local_kv,
        )

        return (
            local_dq,
            local_dk,
            local_dv,
            global_dsink,
            None,  # dist_attn_runtime
            None,  # softmax_scale
            None,  # softcap
            None,  # return_max_logits
        )

    def _maybe_concat(
        self,
        *x: torch.Tensor,
        need_concat: bool = False,
    ) -> FusedOrTupleTensor:
        """
        Maybe concatenate given tensors
        into a fused tensor along first dim
        if `need_concat` is ``True``
        otherwise return the tupled tensors
        """
        return torch.cat(x, dim=0) if need_concat else x

    def _maybe_chunk(
        self,
        x: FusedOrTupleTensor,
        num_chunks: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        Maybe chunk the fused tensor
        into the tupled tensor views along the seqlen dim
        if it is concatenated before
        """
        return torch.chunk(x, num_chunks, dim=0) if isinstance(x, torch.Tensor) else x

    def _maybe_hp_dtype(
        self, dtype: torch.dtype, need_hp_dtype: bool = True
    ) -> torch.dtype:
        """Maybe return higher precision dtype at least as `self.hp_dtype`
        for the given `dtype`, if `need_hp_dtype` is ``True``

        NOTE: for the `dtype` that is already higher than `self.hp_dtype`,
        this function will always return the same `dtype`, no matter the `need_hp_dtype`
        """
        return max_fp_dtype(dtype, self.hp_dtype) if need_hp_dtype else dtype

    def _maybe_flatten_local_qkv_head_groups(
        self,
        local_q: torch.Tensor,
        local_kv: FusedOrTupleTensor,
    ) -> tuple[torch.Tensor, FusedOrTupleTensor]:
        """Maybe flatten the head groups of local q,kv
        for better dynamic solver performance

        Args:
            local_q (torch.Tensor): local query tensor.
            local_kv (FusedOrTupleTensor): local key/value fused/tupled tensor.

        Returns:
            tuple[torch.Tensor, FusedOrTupleTensor]: maybe flattened local q and local kv.

        Shape (before flatten):
            local_q: [num_tokens_q_local, num_heads_q, head_dim]
            local_k: [num_tokens_kv_local, num_heads_kv, head_dim]
            local_v: [num_tokens_kv_local, num_heads_kv, head_dim]

        Shape (after flatten):
            local_q: [num_heads_kv * num_tokens_q_local, heads_per_group, head_dim]
            local_k: [num_heads_kv * num_tokens_kv_local, 1, head_dim]
            local_v: [num_heads_kv * num_tokens_kv_local, 1, head_dim]
        """
        assert isinstance(
            local_kv, tuple
        ), "local_kv should be tupled tensors for this API"
        assert (
            local_q.size(1) == self.comm_meta.num_heads_q
        ), f"local_q.num_heads ({local_q.size(1)}) != comm_meta.num_heads_q ({self.comm_meta.num_heads_q})"
        assert (
            local_kv[0].size(1) == self.comm_meta.num_heads_kv
        ), f"local_k.num_heads ({local_kv[0].size(1)}) != comm_meta.num_heads_kv ({self.comm_meta.num_heads_kv})"

        if not self.flatten_head_groups:
            return local_q, local_kv

        # Transpose local_q: flatten groups into sequence dimension
        # [num_tokens, num_heads_q, head_dim] -> [num_heads_kv * num_tokens, num_heads_per_group, head_dim]
        # Order: Group 0 (all tokens), Group 1 (all tokens), ...
        local_q = rearrange(
            local_q,
            "n (g h) d -> (g n) h d",
            g=self.comm_meta.num_heads_kv,
            h=self.comm_meta.num_heads_per_group,
        ).contiguous()

        # Transpose local_k and local_v: flatten groups (heads) into sequence dimension
        # [num_tokens_kv_local, num_heads_kv, head_dim] -> [num_heads_kv * num_tokens_kv_local, 1, head_dim]
        local_k, local_v = local_kv
        local_k, local_v = [
            rearrange(x, "n h d -> (h n) 1 d").contiguous() for x in (local_k, local_v)
        ]
        local_kv = (local_k, local_v)

        return local_q, local_kv

    def _maybe_unflatten_local_out_lse_head_groups(
        self,
        local_out: torch.Tensor,
        local_lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Maybe unflatten the head groups of local out/lse
        since we flatten local q/kv for better dynamic solver performance

        Args:
            local_out (torch.Tensor): maybe flattened local out tensor.
            local_lse (torch.Tensor): maybe flattened local lse tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: maybe unflattened local out and local lse.

        Shape (before unflatten):
            local_out: [num_heads_kv * num_tokens_q_local, num_heads_per_group, head_dim]
            local_lse: [num_heads_kv * num_tokens_q_local, num_heads_per_group]

        Shape (after unflatten):
            local_out: [num_tokens_q_local, num_heads_q, head_dim]
            local_lse: [num_tokens_q_local, num_heads_q]
        """
        if not self.flatten_head_groups:
            return local_out, local_lse

        # local_out: [(g * n_q), h_per_group, d] -> [n_q, num_heads_q, d]
        local_out = rearrange(
            local_out,
            "(g n) h d -> n (g h) d",
            g=self.comm_meta.num_heads_kv,
            h=self.comm_meta.num_heads_per_group,
        ).contiguous()

        # local_lse: [(g * n_q), h_per_group] -> [n_q, num_heads_q]
        local_lse = rearrange(
            local_lse,
            "(g n) h -> n (g h)",
            g=self.comm_meta.num_heads_kv,
            h=self.comm_meta.num_heads_per_group,
        ).contiguous()

        return local_out, local_lse

    def _maybe_flatten_local_qo_do_lse_head_groups(
        self,
        local_qo_do: FusedOrTupleTensor,
        local_lse: torch.Tensor,
    ) -> tuple[FusedOrTupleTensor, torch.Tensor]:
        """Maybe flatten the head groups of local qo_do/kv/lse
        for better dynamic solver performance

        Args:
            local_qo_do (FusedOrTupleTensor): local qo_do tensor.
            local_lse (torch.Tensor): local lse tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: maybe flattened local qo_do/kv/lse.

        Shape (before flatten):
            local_q: [num_tokens_q_local, num_heads_q, head_dim]
            local_out: [num_tokens_q_local, num_heads_q, head_dim]
            local_do: [num_tokens_q_local, num_heads_q, head_dim]
            local_lse:  [num_tokens_q_local, num_heads_q]

        Shape (after flatten):
            local_q: [num_heads_kv * num_tokens_q_local, num_heads_per_group, head_dim]
            local_out: [num_heads_kv * num_tokens_q_local, num_heads_per_group, head_dim]
            local_do: [num_heads_kv * num_tokens_q_local, num_heads_per_group, head_dim]
            local_lse: [num_heads_kv * num_tokens_q_local, num_heads_per_group]
        """
        if not self.flatten_head_groups:
            return local_qo_do, local_lse

        assert isinstance(
            local_qo_do, tuple
        ), "local_qo_do should be tupled tensors for this API"

        # out/do: [n_q, num_heads_q, d] -> [(g * n_q), h_per_group, d]
        local_q, local_out, local_do = local_qo_do
        local_out, local_do = [
            rearrange(
                x,
                "n (g h) d -> (g n) h d",
                g=self.comm_meta.num_heads_kv,
                h=self.comm_meta.num_heads_per_group,
            ).contiguous()
            for x in [local_out, local_do]
        ]
        local_qo_do = (local_q, local_out, local_do)

        # lse: [n_q, num_heads_q] -> [(g * n_q), h_per_group]
        local_lse = rearrange(
            local_lse,
            "n (g h) -> (g n) h",
            g=self.comm_meta.num_heads_kv,
            h=self.comm_meta.num_heads_per_group,
        ).contiguous()

        return local_qo_do, local_lse

    def _maybe_unflatten_local_dqkv_head_groups(
        self,
        local_dq: torch.Tensor,
        local_dk: torch.Tensor,
        local_dv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maybe unflatten the head groups of local dq/kv/dsink
        since we flatten local q/kv for better dynamic solver performance

        Args:
            local_dq (torch.Tensor): maybe flattened local dq tensor.
            local_dkv (torch.Tensor): maybe flattened local dkv tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: maybe unflattened local dq/kv/dsink.

        Shape (before unflatten):
            local_dq: [num_heads_kv * num_tokens_q_local, num_heads_per_group, head_dim]
            local_dk: [num_heads_kv * num_tokens_kv_local, 1, head_dim]
            local_dv: [num_heads_kv * num_tokens_kv_local, 1, head_dim]

        Shape (after unflatten):
            local_dq: [num_tokens_q_local, num_heads_q, head_dim]
            local_dk: [num_tokens_kv_local, num_heads_kv, head_dim]
            local_dv: [num_tokens_kv_local, num_heads_kv, head_dim]
        """

        if not self.flatten_head_groups:
            return local_dq, local_dk, local_dv

        # local_dq: [(g * n_q), h_per_group, d] -> [n_q, num_heads_q, d]
        local_dq = rearrange(
            local_dq,
            "(g n) h d -> n (g h) d",
            g=self.comm_meta.num_heads_kv,
            h=self.comm_meta.num_heads_per_group,
        )

        # local_dk/local_dv: [(num_heads_kv * n_kv), 1, d] -> [n_kv, num_heads_kv, d]
        local_dk, local_dv = [
            rearrange(
                x,
                "(h n) 1 d -> n h d",
                h=self.comm_meta.num_heads_kv,
            )
            for x in [local_dk, local_dv]
        ]

        return local_dq, local_dk, local_dv

    def _reset_work_list(self):
        self.remote_q_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None
        ] * self.overlap_degree  # fwd
        self.remote_kv_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None
        ] * self.overlap_degree  # fwd + bwd
        self.partial_out_lse_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None
        ] * self.overlap_degree  # fwd
        self.remote_qo_do_lse_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None
        ] * self.overlap_degree  # bwd
        self.partial_dq_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None
        ] * self.overlap_degree  # bwd
        self.partial_dkv_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None
        ] * self.overlap_degree  # bwd
        self.partial_dsink_reduce_work: WorkWithPostProcessFn = None  # bwd

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DistAttnRuntime):
            return False
        return (
            is_same_process_group(self.cp_group_gc, other.cp_group_gc)
            and is_same_process_group(self.cp_group_gr, other.cp_group_gr)
            and (self.comm_meta, self.calc_meta, self.deterministic)
            == (other.comm_meta, other.calc_meta, other.deterministic)
        )


class DistAttnFunc(torch.autograd.Function):
    """Distributed Flash Attention Function"""

    @staticmethod
    def forward(
        ctx,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        global_sink: torch.Tensor | None,
        dist_attn_runtime: DistAttnRuntime,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        return_max_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Distributed Attention forward function

        Args:
            local_q (torch.Tensor): local q tensor
            local_k (torch.Tensor): local k tensor
            local_v (torch.Tensor): local v tensor
            global_sink (torch.Tensor | None): optional global sink tensor
                to apply attention sink if given

            dist_attn_runtime(DistAttnRuntime): dist attn runtime

            softmax_scale (float, optional): softmax scale.
                Defaults to None to use default value: 1/sqrt(head_dim)
            softcap (float, optional): softcap. Defaults to 0.
            return_max_logits (bool, optional): whether to compute and return max_logits
                (per head, all-reduced MAX across ranks). Defaults to False.

        Returns:
            local_out (torch.Tensor): local out tensor
            local_lse (torch.Tensor): local lse tensor
            local_max_logits (torch.Tensor | None): all-reduced max_logits [num_heads_q], or None

        Shape:
            local_q: [num_tokens_q_local, num_heads_q, head_dim]
            local_k: [num_tokens_kv_local, num_heads_kv, head_dim]
            local_v: [num_tokens_kv_local, num_heads_kv, head_dim]
            global_sink: [num_tokens_sink_global, num_heads_q]
            local_out: [num_tokens_q_local, num_heads_q, head_dim]
            local_lse: [num_tokens_q_local, num_heads_q]
            local_max_logits: [num_heads_q] when return_max_logits is True
        """
        # init kernel barrier for native grpcoll to ensure comm kernel is always preceded by compute kernel
        kernel_barrier_fetch = KernelBarrier(
            dist_attn_runtime.fwd_kernel_barrier_fetch_target
        )
        kernel_barrier_reduce = KernelBarrier(
            dist_attn_runtime.fwd_kernel_barrier_reduce_target
        )

        # get local qkv and pre-fetch qkv for remote stage(s)
        local_q, local_kv = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
            local_q=local_q,
            local_kv=(local_k, local_v),
            overlap_stage=None,
            kernel_barrier=kernel_barrier_fetch,
        )

        kernel_barrier_fetch.synchronize()
        (
            partial_local_out,
            partial_local_meta,
        ) = dist_attn_runtime.apply_fwd_partial_attn(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            softmax_scale=softmax_scale,
            softcap=softcap,
            sink=global_sink,
            return_max_logits=return_max_logits,
        )
        assert partial_local_out is not None and partial_local_meta is not None
        partial_local_lse = partial_local_meta.lse
        partial_local_max_logits = partial_local_meta.max_logits

        # loop into remote stages
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(  # NOTE: this debug info will introduce GPU-CPU sync
                    f"DistAttnFunc.forward: {dist_attn_runtime.overlap_degree=} {ith_overlap_stage=} "
                    f"{kernel_barrier_fetch.get_value()=} {kernel_barrier_reduce.get_value()=}"
                )

            # reset kernel barrier for next stage
            kernel_barrier_fetch.reset()

            # wait for ith remote qkv prepared and pre-fetch (i+1)th remote qkv
            (
                curr_remote_q,
                curr_remote_kv,
            ) = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
                local_q=local_q,
                local_kv=local_kv,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_fetch,
            )

            if not dist_attn_runtime.is_last_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_fetch.synchronize()

            if not dist_attn_runtime.is_first_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_reduce.synchronize()

            # apply fwd partial attn with ith remote qkv
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_out,
                partial_remote_meta,
            ) = dist_attn_runtime.apply_fwd_partial_attn(
                q=curr_remote_q,
                kv=curr_remote_kv,
                out_acc=partial_local_out
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                lse_acc=partial_local_lse
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                max_logits_acc=partial_local_max_logits,
                overlap_stage=ith_overlap_stage,
                softmax_scale=softmax_scale,
                softcap=softcap,
                sink=global_sink,
                return_max_logits=return_max_logits,
            )
            partial_remote_lse = (
                partial_remote_meta.lse if partial_remote_meta is not None else None
            )
            if return_max_logits and partial_remote_meta is not None:
                partial_local_max_logits = partial_remote_meta.max_logits

            # reset kernel barrier for next stage
            kernel_barrier_reduce.reset()

            # reduce ith partial out with partial lse
            # overlapped with (i+1)th fwd partial attn and maybe (i+2)th pre-fetch
            dist_attn_runtime.reduce_partial_out_lse(
                partial_remote_out=partial_remote_out,
                partial_remote_lse=partial_remote_lse,
                partial_local_out=partial_local_out,
                partial_local_lse=partial_local_lse,
                ref_remote_out=curr_remote_q,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_reduce,
            )

        # prepare reduced local out and lse
        # before returning from forward and saving for backward
        local_out, local_lse = dist_attn_runtime.prepare_reduced_local_out_lse(
            partial_local_out=partial_local_out,
            partial_local_lse=partial_local_lse,
            ref_local_out=local_q,
        )
        # reduce max_logits across all ranks at the end of distributed attention forward
        if return_max_logits:
            local_max_logits = dist_attn_runtime.reduce_max_logits(
                partial_local_max_logits=partial_local_max_logits,
            )
        else:
            local_max_logits = None

        if (
            magi_attention.dist_attn_backward_hide_tail_reduce()
            and dist_attn_runtime.overlap_degree > 0
        ):
            last_stage_q, last_stage_kv = curr_remote_q, curr_remote_kv
        else:
            last_stage_q, last_stage_kv = None, None

        dist_attn_runtime.save_tensors_for_bwd(
            ctx,
            local_q=local_q,
            local_kv=local_kv,
            local_out=local_out,
            local_lse=local_lse,
            last_stage_q=last_stage_q,
            last_stage_kv=last_stage_kv,
            global_sink=global_sink,
        )
        ctx.dist_attn_runtime = dist_attn_runtime
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap

        return local_out, local_lse, local_max_logits

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        dist_attn_runtime: DistAttnRuntime = ctx.dist_attn_runtime
        if (
            magi_attention.dist_attn_backward_hide_tail_reduce()
            and dist_attn_runtime.overlap_degree > 0
        ):
            return dist_attn_runtime._hide_tail_stage_reduce_backward(
                ctx, grad_output, *args
            )

        (
            local_q,
            local_kv,
            local_out,
            local_lse,
            _,
            _,
            global_sink,
        ) = dist_attn_runtime.load_tensors_from_fwd(ctx)
        softmax_scale: float | None = ctx.softmax_scale
        softcap: float = ctx.softcap

        kernel_barrier_fetch = KernelBarrier(
            dist_attn_runtime.bwd_kernel_barrier_fetch_target
        )
        kernel_barrier_reduce = KernelBarrier(
            dist_attn_runtime.bwd_kernel_barrier_reduce_target
        )

        # get local qo_do,kv,lse and pre-fetch qo_do,kv,lse for remote stage(s)
        (
            local_qo_do,
            local_kv,
            local_lse,
        ) = dist_attn_runtime.get_curr_qo_do_kv_lse_and_fetch_next(
            local_qo_do=(local_q, local_out, grad_output),
            local_kv=local_kv,
            local_lse=local_lse,
            overlap_stage=None,
            kernel_barrier=kernel_barrier_fetch,
        )

        kernel_barrier_fetch.synchronize()

        # apply bwd partial attn with local qo_do,kv,lse
        # overlapped with 0th pre-fetch
        (
            partial_local_dq,
            partial_local_dkv,
            partial_global_dsink,
        ) = dist_attn_runtime.apply_bwd_partial_attn(
            qo_do=local_qo_do,
            kv=local_kv,
            lse=local_lse,
            overlap_stage=None,
            softmax_scale=softmax_scale,
            softcap=softcap,
            sink=global_sink,
        )
        assert partial_local_dq is not None and partial_local_dkv is not None
        assert global_sink is None or partial_global_dsink is not None

        # reduce partial global dsink if required
        dist_attn_runtime.reduce_partial_dsink(
            partial_global_dsink=partial_global_dsink,
        )

        # loop into remote stages
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            kernel_barrier_fetch.reset()

            # wait for ith remote qo_do,kv,lse prepared
            # and pre-fetch (i+1)th remote qo_do,kv,lse
            (
                curr_remote_qo_do,
                curr_remote_kv,
                curr_remote_lse,
            ) = dist_attn_runtime.get_curr_qo_do_kv_lse_and_fetch_next(
                local_qo_do=local_qo_do,
                local_kv=local_kv,
                local_lse=local_lse,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_fetch,
            )

            if not dist_attn_runtime.is_last_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_fetch.synchronize()

            if not dist_attn_runtime.is_first_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_reduce.synchronize()

            kernel_barrier_reduce.reset()

            # apply bwd partial attn with ith remote qo_do,kv,lse
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_dq,
                partial_remote_dkv,
                _,  # partial_global_dsink
            ) = dist_attn_runtime.apply_bwd_partial_attn(
                qo_do=curr_remote_qo_do,
                kv=curr_remote_kv,
                lse=curr_remote_lse,
                dq_acc=partial_local_dq if dist_attn_runtime.bwd_dq_use_acc else None,
                overlap_stage=ith_overlap_stage,
                softmax_scale=softmax_scale,
                softcap=softcap,
                sink=global_sink,
            )

            # reduce ith partial dq,dkv
            # overlapped with (i+1)th bwd partial attn and maybe (i+2)th pre-fetch
            dist_attn_runtime.reduce_partial_dq_dkv(
                partial_remote_dq=partial_remote_dq,
                partial_local_dq=partial_local_dq,
                ref_remote_qo_do=curr_remote_qo_do,
                partial_remote_dkv=partial_remote_dkv,
                partial_local_dkv=partial_local_dkv,
                ref_remote_kv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_reduce,
            )

        # prepare reduced local dq,dk,dv and maybe global dsink
        # before returning from backward
        (
            local_dq,
            local_dk,
            local_dv,
            global_dsink,
        ) = dist_attn_runtime.prepare_reduced_local_dqkv_global_dsink(
            partial_local_dq=partial_local_dq,
            partial_local_dkv=partial_local_dkv,
            partial_global_dsink=partial_global_dsink,
            ref_local_dq=local_q,
            ref_local_dkv=local_kv,
        )

        return (
            local_dq,
            local_dk,
            local_dv,
            global_dsink,
            None,  # dist_attn_runtime
            None,  # softmax_scale
            None,  # softcap
            None,  # return_max_logits
        )


def dist_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dist_attn_runtime: DistAttnRuntime,
    sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    return_max_logits: bool = False,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """Distributed attention autograd function

    Args:
        q (torch.Tensor): local q tensor
        k (torch.Tensor): local k tensor
        v (torch.Tensor): local v tensor

        dist_attn_runtime (DistAttnRuntime): distributed attention runtime

        sink (torch.Tensor, optional): global sink tensor (replicated among cp ranks).
            Defaults to ``None`` to not apply attention sink.

        NOTE: for now, we only support sink tensor with dtype=torch.float32

        softmax_scale (float, optional): softmax scale.
            Defaults to ``None`` to use default value: ``1/sqrt(head_dim)``
        softcap (float, optional): softcap. Defaults to ``0.0``.
        return_max_logits (bool, optional): whether to compute and return max_logits
            (per head, shape [num_heads_q], all-reduced MAX across all ranks).
            Defaults to ``False``.

    Returns:
        out (torch.Tensor): local out tensor
        meta (AttnForwardMeta): attention forward meta (lse, and max_logits when return_max_logits is True)

    Shapes:
        q: [num_tokens_q_local, num_heads_q, head_dim]
        k: [num_tokens_kv_local, num_heads_kv, head_dim]
        v: [num_tokens_kv_local, num_heads_kv, head_dim]
        sink: [num_tokens_sink_global, num_heads_q]
        out: [num_tokens_q_local, num_heads_q, head_dim]
        lse: [num_tokens_q_local, num_heads_q]
        meta.max_logits: [num_heads_q] when return_max_logits is True
    """
    out, lse, max_logits = DistAttnFunc.apply(
        q,
        k,
        v,
        sink,
        dist_attn_runtime,
        softmax_scale,
        softcap,
        return_max_logits,
    )
    return out, AttnForwardMeta(lse=lse, max_logits=max_logits)
