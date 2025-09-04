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

from typing import Any, TypeAlias

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.comm.primitive.grpcoll import (
    group_cast_collective,
    group_reduce_collective,
)
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.meta.collection import CalcMeta, CommMeta
from magi_attention.utils import is_same_process_group, max_fp_dtype, nvtx

from .flex_flash_attn import _flex_flash_attn_backward, _flex_flash_attn_forward
from .sdpa import sdpa_bwd, sdpa_fwd
from .utils import correct_attn_fwd_result

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
    remote_lse_work_with_buffer_per_stage: list[WorkWithBuffer]
    remote_qo_do_work_with_buffer_per_stage: list[WorkWithBuffer]
    partial_dq_reduce_work_per_stage: list[WorkWithPostProcessFn]
    partial_dkv_reduce_work_per_stage: list[WorkWithPostProcessFn]

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
        self.deterministic = magi_attention.is_deterministic_mode_enable()
        self.overlap_degree = comm_meta.overlap_degree
        self.prefetch_stage_by_stage = (
            magi_attention.is_cuda_device_max_connections_one()
        )

        # ----------    flags for fwd   --------- #

        self.fwd_sm_margin = magi_attention.comm.ffa_fwd_sm_margin_save_for_comm()

        # NOTE: we now always concat kv and dkv together for comm
        self.concat_kv = True

        # NOTE: when enabling FFA fwd high precision reduce,
        # we won't downcast partial out to q dtype before group-reduce comm,
        # to trade-off double comm overhead for increased precision and less dtype-cast overhead
        # and this works for out only when `is_qo_comm_enable` is ``True``
        self.fwd_hp_reduce = (
            magi_attention.comm.is_ffa_fwd_high_precision_reduce_enable()
        )

        # NOTE: when neither using sdpa backend nor qo comm,
        # we will use accumulative buffer for partial out and lse
        # to avoid the storage of partial results
        # and an additional explicit `correct_attn_fwd_result`
        self.fwd_out_lse_use_acc = (
            not magi_attention.comm.is_qo_comm_enable()
            and not magi_attention.is_sdpa_backend_enable()
        )

        # NOTE: when enabling qo comm and disabling FFA fwd high precision reduce
        # we're supposed to initialize the partial local out in low precision
        self.fwd_local_out_lp_init = (
            magi_attention.comm.is_qo_comm_enable() and not self.fwd_hp_reduce
        )

        # ----------    flags for bwd   --------- #

        self.bwd_sm_margin = magi_attention.comm.ffa_bwd_sm_margin_save_for_comm()

        # NOTE: we concat q,o,do in dist-attn bwd when qo comm is enabled
        self.concat_qo_do = magi_attention.comm.is_qo_comm_enable()

        # NOTE: when enabling FFA bwd high precision reduce,
        # we won't downcast downcast partial dq,dk,dv to q,k,v dtype before group-reduce comm,
        # to trade-off double comm overhead for increased precision and less dtype-cast overhead
        # and this works for dq only when `is_qo_comm_enable` is ``True``
        self.bwd_hp_reduce = (
            magi_attention.comm.is_ffa_bwd_high_precision_reduce_enable()
        )

        # NOTE: when neither using sdpa backend nor qo comm
        # we will use accumulative buffer for partial dq
        # to avoid an additional explicit `add_`
        self.bwd_dq_use_acc = (
            not magi_attention.comm.is_qo_comm_enable()
            and not magi_attention.is_sdpa_backend_enable()
        )

        # NOTE: when disabling FFA bwd high precision reduce
        # we're supposed to initialize the partial local dk,dv in low precision
        # and the same applies to dq if enabling qo comm
        self.bwd_local_dkv_lp_init = not self.bwd_hp_reduce
        self.bwd_local_dq_lp_init = (
            magi_attention.comm.is_qo_comm_enable() and not self.bwd_hp_reduce
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
        overlap_stage: int | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Apply forward partial attention with given q,kv for the given overlap stage

        Args:
            q (torch.Tensor): current q
            kv (FusedOrTupleTensor): current kv fused or tupled tensor
            out_acc (torch.Tensor, optional): accumulative buffer for out
            lse_acc (torch.Tensor, optional): accumulative buffer for lse
            overlap_stage (int, optional): given overlap stage. Defaults to None.

            softmax_scale (float, optional): softmax scale.
                Defaults to None to use default value: 1/sqrt(head_dim)
            softcap (float, optional): softcap. Defaults to 0.

        Returns:
            out (torch.Tensor | None): partial out, or None if skipped
            lse (torch.Tensor | None): partial log-sum-exp, or None if skipped

        Shape:
            q: [num_tokens_q, num_heads_q, head_dim]
            kv: [num_tokens_kv*2, num_heads_kv, head_dim]
            out: [num_tokens_q, num_heads_q, head_dim]
            lse: [num_tokens_q, num_heads_q]
            out_acc: [num_tokens_q, num_heads_q, head_dim]
            lse_acc: [num_tokens_q, num_heads_q]
        """
        is_host_stage = self.is_host_stage(overlap_stage)

        # fetch attn arg
        if is_host_stage:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            attn_arg = self.calc_meta.remote_attn_args_list[curr_remote_stage]

        # skipped case
        if attn_arg.can_skip(is_bwd=False):
            partial_out, partial_lse = None, None
            if is_host_stage:
                hp_init_dtype = max_fp_dtype(q.dtype, torch.float32)
                # NOTE: we can NOT use empty_like here,
                # since when all attn computations are skipped for certain q range
                # we have nowhere to zero-fill it
                partial_out = torch.zeros_like(
                    q,
                    dtype=q.dtype if self.fwd_local_out_lp_init else hp_init_dtype,
                    device=q.device,
                )
                partial_lse = torch.full(
                    (q.size(0), q.size(1)),
                    fill_value=float("-inf"),
                    dtype=hp_init_dtype,  # lse always in high-precision
                    device=q.device,
                )
            return partial_out, partial_lse

        # attention forward pass
        k, v = self._maybe_chunk_kv(kv)
        _softmax_scale: float = (
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )
        with nvtx.add_nvtx_event(
            f"attn-fwd: "
            f"{attn_arg.total_area=} | "
            f"{attn_arg.q_ranges=} | "
            f"{attn_arg.k_ranges=}"
        ):
            if magi_attention.is_sdpa_backend_enable():
                partial_out, partial_lse = sdpa_fwd(
                    q=q,
                    k=k,
                    v=v,
                    attn_arg=attn_arg,
                    softmax_scale=_softmax_scale,
                    softcap=softcap,
                )
            else:
                partial_out, partial_lse = _flex_flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    out=out_acc,  # directly reduce to out_acc
                    lse=lse_acc,  # directly reduce to lse_acc
                    **attn_arg.to_ffa_args(is_bwd=False),
                    merge_q_ranges=None,
                    qk_map=None,
                    fwd_unique_count=None,
                    ref_block_size=None,
                    softmax_scale=_softmax_scale,
                    deterministic=self.deterministic,
                    softcap=softcap,
                    sm_margin=self.fwd_sm_margin,
                    # NOTE: increase the partial out precision temporarily,
                    # to reduce the error caused by the out correction
                    out_type=torch.float32,
                    # NOTE: when using accumulative buffer, we need to always enable atomic reduction
                    # unless it is the first call when accumulative buffer is still None
                    disable_fwd_atomic_reduction=(
                        attn_arg.disable_fwd_atomic_reduction and out_acc is None
                    ),
                )

        # maybe downcast out to q dtype for the host stage
        if is_host_stage and self.fwd_local_out_lp_init:
            partial_out = partial_out.to(q.dtype)

        return partial_out, partial_lse

    @nvtx.instrument_nvtx
    def get_curr_q_kv_and_fetch_next(
        self,
        local_q: torch.Tensor,
        local_kv: FusedOrTupleTensor,
        overlap_stage: int | None = None,
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
            local_kv = self._maybe_concat_kv(*local_kv)
            curr_q = local_q
            curr_kv = local_kv
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
            (
                self.remote_q_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_q(local_q=local_q, overlap_stage=next_stage)
            (
                self.remote_kv_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_kv(local_kv=local_kv, overlap_stage=next_stage)
        elif is_host_stage:
            # when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote comms in advance of ffa fwd
            # and ffa fwd can still overlap with these comms
            # with the support of non-zero `sm_margin`, thx to persistent kernel design
            self.remote_q_work_with_buffer_per_stage = [
                self._fetch_remote_q(local_q=local_q, overlap_stage=ith_stage)
                for ith_stage in range(self.overlap_degree)
            ]
            self.remote_kv_work_with_buffer_per_stage = [
                self._fetch_remote_kv(local_kv=local_kv, overlap_stage=ith_stage)
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
    ) -> None:
        """
        Reduce remote out and lse to local out and lse for the given remote overlap stage
        and push the returned partial_out_lse_reduce_work to self.partial_out_lse_reduce_work_per_stage

        Args:
            partial_remote_out (torch.Tensor, optional):
                partial remote out in float32 dtype, or None if skipped
            partial_remote_lse (torch.Tensor, optional):
                partial remote lse in float32 dtype, or None if skipped
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

        # reset the temporary work list
        self._reset_work_list()

        return local_out, local_lse

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
    ) -> tuple[torch.Tensor | None, FusedOrTupleTensor | None]:
        """
        Apply backward partial attention with given qo_do,kv,lse for the given overlap stage

        Args:
            qo_do (FusedOrTupleTensor): current q, o, do fused or tupled tensor
            kv (FusedOrTupleTensor): current kv
            lse (torch.Tensor): current lse
            dq_acc (torch.Tensor, optional): accumulative buffer for dq
            overlap_stage (int, optional): given overlap stage. Defaults to None.

            softmax_scale (float, optional): softmax scale.
                Defaults to None to use default value: 1/sqrt(head_dim)
            softcap (float, optional): softcap. Defaults to 0.

        Returns:
            partial_dq (torch.Tensor | None): partial dq, or None if skipped
            partial_dkv (FusedOrTupleTensor | None): partial dkv, or None if skipped

        Shape:
            qo_do: [num_tokens_q*3, num_heads_q, head_dim]
            kv: [num_tokens_kv*2, num_heads_kv, head_dim]
            lse: [num_tokens_q, num_heads_q]
            partial_dq: [num_tokens_q, num_heads_q, head_dim]
            partial_dkv: [num_tokens_kv*2, num_heads_kv, head_dim]
        """
        assert self.concat_kv, "only support concat_kv"

        is_host_stage = self.is_host_stage(overlap_stage)

        # fetch attn arg
        if is_host_stage:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            attn_arg = self.calc_meta.remote_attn_args_list[curr_remote_stage]

        # skipped case
        if attn_arg.can_skip(is_bwd=True):
            partial_dq, partial_dkv = None, None
            if is_host_stage:
                q, _, _ = self._maybe_chunk_qo_do(qo_do)
                # NOTE: if local_dq and local_dkv calculation are skipped,
                # we need to zeros initialize them since they might be reduced later
                partial_dq = torch.zeros_like(
                    q,
                    dtype=q.dtype
                    if self.bwd_local_dq_lp_init
                    else max_fp_dtype(q.dtype, torch.float32),
                )
                partial_dkv = torch.zeros_like(
                    kv,
                    dtype=kv.dtype  # type: ignore[union-attr]
                    if self.bwd_local_dkv_lp_init
                    else max_fp_dtype(kv.dtype, torch.float32),  # type: ignore[union-attr]
                )
            return partial_dq, partial_dkv

        # attention backward pass
        q, o, do = self._maybe_chunk_qo_do(qo_do)
        k, v = self._maybe_chunk_kv(kv)
        _softmax_scale: float = (
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )
        if magi_attention.is_sdpa_backend_enable():
            partial_dq, partial_dk, partial_dv = sdpa_bwd(
                do=do,
                q=q,
                k=k,
                v=v,
                o=o,
                lse=lse,
                attn_arg=attn_arg,
                softmax_scale=_softmax_scale,
                softcap=softcap,
            )
            partial_dkv = self._maybe_concat_kv(partial_dk, partial_dv)
        else:
            # NOTE: we initial partial dkv and chunk to dk, dv to avoid concat them back before return
            # and we need to zero-initialize partial_dkv since it needs to be reduced
            partial_dkv = torch.zeros_like(kv, dtype=torch.float32)
            partial_dk, partial_dv = self._maybe_chunk_kv(partial_dkv)
            partial_dq, partial_dk, partial_dv = _flex_flash_attn_backward(
                dout=do,
                q=q,
                k=k,
                v=v,
                out=o,
                lse=lse,
                dq=dq_acc,  # directly reduce to dq_acc
                dk=partial_dk,
                dv=partial_dv,
                # NOTE: increase the partial dq, dkv precision temporarily,
                # to reduce the error caused by the atomic reduction inside the kernel
                dq_type=torch.float32,
                dk_type=torch.float32,
                dv_type=torch.float32,
                **attn_arg.to_ffa_args(is_bwd=True),
                merge_k_ranges=None,
                bwd_kq_map=None,
                bwd_unique_count=None,
                softmax_scale=_softmax_scale,
                deterministic=self.deterministic,
                softcap=softcap,
                disable_bwd_dkv_atomic_reduction=attn_arg.disable_bwd_dkv_atomic_reduction,
                sm_margin=self.bwd_sm_margin,
            )

        # maybe downcast dq,dkv to q,kv dtype for the host stage
        if is_host_stage:
            if self.bwd_local_dq_lp_init:
                partial_dq = partial_dq.to(q.dtype)

            if self.bwd_local_dkv_lp_init:
                partial_dkv = partial_dkv.to(kv.dtype)  # type: ignore[union-attr]

        return partial_dq, partial_dkv

    @nvtx.instrument_nvtx
    def get_curr_qo_do_kv_lse_and_fetch_next(
        self,
        local_qo_do: FusedOrTupleTensor,
        local_kv: torch.Tensor,
        local_lse: torch.Tensor,
        overlap_stage: int | None = None,
    ) -> tuple[FusedOrTupleTensor, torch.Tensor, torch.Tensor]:
        """
        Get current qo_do,kv,lse and fetch next qo_do,kv,lse for the given overlap stage

        Args:
            local_qo_do (FusedOrTupleTensor): local qo_do fused or tupled tensor
            local_kv (torch.Tensor): local kv
            local_lse (torch.Tensor): local lse
            overlap_stage (int, optional): given overlap stage. Defaults to None.

        Returns:
            curr_qo_do (FusedOrTupleTensor): current qo_do fused or tupled tensor
            curr_kv (torch.Tensor): current kv
            curr_lse (torch.Tensor): current lse
        """
        next_stage = self.get_next_stage(overlap_stage)
        is_host_stage = self.is_host_stage(overlap_stage)
        is_last_remote_stage = self.is_last_remote_stage(overlap_stage)

        # wait for host/remote qo_do,kv,lse prepared for current stage
        if is_host_stage:
            local_qo_do = self._maybe_concat_qo_do(*local_qo_do)
            curr_qo_do = local_qo_do
            curr_kv = local_kv
            curr_lse = local_lse
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            (
                curr_remote_kv_work,
                curr_remote_kv_buffer,
            ) = self.remote_kv_work_with_buffer_per_stage[curr_remote_stage]
            curr_kv = curr_remote_kv_work.wait_post_process(curr_remote_kv_buffer)
            (
                curr_remote_qo_do_work,
                curr_remote_qo_do_buffer,
            ) = self.remote_qo_do_work_with_buffer_per_stage[curr_remote_stage]
            curr_qo_do = curr_remote_qo_do_work.wait_post_process(
                curr_remote_qo_do_buffer
            )
            (
                curr_remote_lse_work,
                curr_remote_lse_buffer,
            ) = self.remote_lse_work_with_buffer_per_stage[curr_remote_stage]
            curr_lse = curr_remote_lse_work.wait_post_process(curr_remote_lse_buffer)

        # pre-fetch remote qo_do,kv,lse for next stage(s)
        if self.prefetch_stage_by_stage and not is_last_remote_stage:
            (
                self.remote_lse_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_lse(
                local_lse=local_lse,
                overlap_stage=next_stage,
            )
            (
                self.remote_kv_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_kv(
                local_kv=local_kv,
                overlap_stage=next_stage,
            )
            (
                self.remote_qo_do_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_qo_do(
                local_qo_do=local_qo_do,
                overlap_stage=next_stage,
            )
        elif is_host_stage:
            # NOTE: when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote comms in advance of ffa bwd
            # and ffa bwd can still overlap with these comms
            # with the support of `sm_margin`, thx to persistent kernel design
            self.remote_kv_work_with_buffer_per_stage = [
                self._fetch_remote_kv(local_kv=local_kv, overlap_stage=ith_stage)
                for ith_stage in range(self.overlap_degree)
            ]
            self.remote_qo_do_work_with_buffer_per_stage = [
                self._fetch_remote_qo_do(
                    local_qo_do=local_qo_do,
                    overlap_stage=ith_stage,
                )
                for ith_stage in range(self.overlap_degree)
            ]
            self.remote_lse_work_with_buffer_per_stage = [
                self._fetch_remote_lse(
                    local_lse=local_lse,
                    overlap_stage=ith_stage,
                )
                for ith_stage in range(self.overlap_degree)
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
    ) -> None:
        """
        Reduce remote dq,dkv to local dq,dkv for the given remote overlap stage
        and push the returned partial_dq_reduce_work,partial_dkv_reduce_work
        to self.partial_dq_reduce_work_per_stage, self.partial_dkv_reduce_work_per_stage
        respectively

        Args:
            partial_remote_dq (torch.Tensor, optional):
                partial remote dq in float32 dtype, or None if skipped
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            ref_remote_qo_do (FusedOrTupleTensor):
                reference remote qo_do fused or tupled tensor,
                to provide meta info like dtype and shape
            partial_remote_dkv (FusedOrTupleTensor, optional):
                partial remote dkv in float32 dtype, or None if skipped
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
        )

        # reduce ith partial dq
        ref_remote_q, _, _ = self._maybe_chunk_qo_do(ref_remote_qo_do)
        (
            self.partial_dq_reduce_work_per_stage[overlap_stage]
        ) = self._reduce_partial_dq(
            partial_remote_dq=partial_remote_dq,
            partial_local_dq=partial_local_dq,
            ref_remote_dq=ref_remote_q,
            overlap_stage=overlap_stage,
        )

    @nvtx.instrument_nvtx
    def prepare_reduced_local_dq_dk_dv(
        self,
        partial_local_dq: torch.Tensor,
        partial_local_dkv: torch.Tensor,
        ref_local_dq: torch.Tensor,
        ref_local_dkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the final reduced local dq,dk,dv
        before returning from backward
        with clearing the temporary work list

        Args:
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            partial_local_dkv (torch.Tensor): partial local dkv to be reduced
            ref_local_dq (torch.Tensor):
                reference local dq, to provide meta info like dtype and shape
            ref_local_dkv (torch.Tensor):
                reference local dkv, to provide meta info like dtype and shape

        Returns:
            local_dq (torch.Tensor): reduced local dq tensor
            local_dk (torch.Tensor): reduced local dk tensor
            local_dv (torch.Tensor): reduced local dv tensor
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

        # cast final local dkv to kv dtype
        local_dkv = partial_local_dkv.to(ref_local_dkv.dtype)

        # chunk final local dkv into dk and dv
        local_dk, local_dv = self._maybe_chunk_kv(local_dkv)

        # reset the temporary work list
        self._reset_work_list()

        return local_dq, local_dk, local_dv

    # ----------    common API   --------- #

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

    @nvtx.instrument_nvtx
    def _fetch_remote_kv(
        self,
        local_kv: FusedOrTupleTensor,
        overlap_stage: int,
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
        assert self.concat_kv, "only support concat_kv"
        _, num_heads, head_dim = local_kv.shape  # type: ignore[union-attr]

        # get the group-cast args for kv
        group_cast_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_kv_seqlen = self.comm_meta.num_remote_kv_tokens_per_stage[overlap_stage]

        # init remote kv buffer
        remote_kv_buffer = torch.empty(
            remote_kv_seqlen,
            num_heads,
            head_dim,
            dtype=local_kv.dtype,  # type: ignore[union-attr]
            device=local_kv.device,  # type: ignore[union-attr]
        )

        # launch group cast kernel
        remote_kv_work = group_cast_collective(
            input=local_kv,
            output=remote_kv_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
        )

        return remote_kv_work, remote_kv_buffer

    @nvtx.instrument_nvtx
    def _fetch_remote_q(
        self,
        local_q: torch.Tensor,
        overlap_stage: int,
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

        if not magi_attention.comm.is_qo_comm_enable():
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
        remote_q_work = group_cast_collective(
            input=local_q,
            output=remote_q_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
        )

        return remote_q_work, remote_q_buffer

    @nvtx.instrument_nvtx
    def _fetch_remote_qo_do(
        self,
        local_qo_do: FusedOrTupleTensor,
        overlap_stage: int,
    ) -> WorkWithBuffer:
        """
        Fetch remote q, o, do buffer from other ranks to local for the given overlap stage,
        and return the corresponding work and buffer

        Args:
            local_qo_do (FusedOrTupleTensor): the local q, o, do (fused or tupled) tensor
            overlap_stage (int): current overlap stage

        Returns:
            WorkWithBuffer:
                remote_qo_do_work (WorkWithPostProcessFn):
                    communication handle, used to wait for communication completion
                remote_qo_do_buffer (FusedOrTupleTensor): remote q, o, do buffer

        Shape:
            local_qo_do: [num_tokens_qo_do_local*3, num_heads_q, head_dim]
            remote_qo_do_buffer: [num_tokens_qo_do_remote_i*3, num_heads_q, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        if not magi_attention.comm.is_qo_comm_enable():
            remote_qo_do_buffer = local_qo_do
            remote_qo_do_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take q,o,do and return q,o,do
            )
            return remote_qo_do_work, remote_qo_do_buffer

        assert self.concat_qo_do, "only support concat_qo_do"
        _, num_heads, head_dim = local_qo_do.shape  # type: ignore[union-attr]

        # get the group-cast args for q,o,do
        group_cast_args = self.comm_meta.qo_do_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_qo_do_seqlen = self.comm_meta.num_remote_qo_do_tokens_per_stage[
            overlap_stage
        ]

        # init remote q,o,do output buffer
        remote_qo_do_buffer = torch.empty(
            remote_qo_do_seqlen,
            num_heads,
            head_dim,
            dtype=local_qo_do.dtype,  # type: ignore[union-attr]
            device=local_qo_do.device,  # type: ignore[union-attr]
        )

        # launch group cast kernel
        remote_qo_do_work = group_cast_collective(
            input=local_qo_do,
            output=remote_qo_do_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
        )

        return remote_qo_do_work, remote_qo_do_buffer

    @nvtx.instrument_nvtx
    def _fetch_remote_lse(
        self,
        local_lse: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithBuffer:
        """
        Fetch remote lse buffer from other ranks to local for the given overlap stage,
        and return the corresponding work and buffer

        Args:
            local_lse (torch.Tensor): the local lse tensor
            overlap_stage (int): current overlap stage

        Returns:
            WorkWithBuffer:
                remote_lse_work (WorkWithPostProcessFn):
                    communication handle, used to wait for communication completion
                remote_lse_buffer (torch.Tensor): remote lse buffer

        Shape:
            local_lse: [num_tokens_q_local, num_heads_q]
            remote_lse_buffer: [num_tokens_q_remote_i, num_heads_q],
                for i = 0, 1, ..., overlap_degree - 1
        """
        # HACK: since lse usually has different shape and dtype from q,o,do
        # we can not trivially fuse lse comm with qo comm, thus here we use specific comm to fetch lse
        # try to find a better way to handle it in the future

        if not magi_attention.comm.is_qo_comm_enable():
            remote_lse_buffer = local_lse
            remote_lse_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take lse and return lse
            )
            return remote_lse_work, remote_lse_buffer

        _, num_heads = local_lse.shape

        # get the group-cast args for lse
        group_cast_args = self.comm_meta.qo_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_lse_seqlen = self.comm_meta.num_remote_qo_tokens_per_stage[overlap_stage]

        # init remote lse buffer with the shape: [sq, nhq]
        remote_lse_buffer = torch.empty(
            remote_lse_seqlen,
            num_heads,
            dtype=local_lse.dtype,
            device=local_lse.device,
        )

        # launch group cast kernel
        remote_lse_work = group_cast_collective(
            input=local_lse,
            output=remote_lse_buffer,
            **group_cast_args,
            group=self.cp_group_gc,
            async_op=True,
        )

        return remote_lse_work, remote_lse_buffer

    @nvtx.instrument_nvtx
    def _reduce_partial_out_lse(
        self,
        partial_remote_out: torch.Tensor | None,
        partial_remote_lse: torch.Tensor | None,
        partial_local_out: torch.Tensor,
        partial_local_lse: torch.Tensor,
        ref_remote_out: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote out and lse to lse-reduce to local out and lse for the given overlap stage,
        and return the corresponding work

        Args:
            partial_remote_out (torch.Tensor, optional):
                partial remote out in float32 dtype, or None if skipped
            partial_remote_lse (torch.Tensor, optional):
                partial remote lse in float32 dtype, or None if skipped
            partial_local_out (torch.Tensor): partial local out to be reduced
            partial_local_lse (torch.Tensor): partial local lse to be reduced
            ref_remote_out (torch.Tensor):
                reference remote out, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_out_lse_reduce_work (WorkWithPostProcessFn): partial out and lse group-reduce work
        """
        if magi_attention.comm.is_qo_comm_enable():
            # get the group-reduce args for out and lse
            group_reduce_args = self.comm_meta.out_lse_group_collective_args_list[
                overlap_stage
            ].to_group_reduce_args()

            # init remote dkv buffer
            if partial_remote_out is None:  # skipped
                hp_init_dtype = max_fp_dtype(ref_remote_out.dtype, torch.float32)
                partial_remote_out = torch.empty_like(
                    ref_remote_out,
                    dtype=hp_init_dtype if self.fwd_hp_reduce else ref_remote_out.dtype,
                    device=ref_remote_out.device,
                )
                partial_remote_lse = torch.empty(
                    (ref_remote_out.size(0), ref_remote_out.size(1)),  # [sq, nhq]
                    dtype=hp_init_dtype,  # lse always in high-precision
                    device=ref_remote_out.device,
                )
            elif not self.fwd_hp_reduce:
                partial_remote_out = partial_remote_out.to(ref_remote_out.dtype)

            # launch group-reduce kernel
            partial_out_lse_reduce_work = group_reduce_collective(
                input=partial_remote_out,
                input_lse=partial_remote_lse,
                output=partial_local_out,
                output_lse=partial_local_lse,
                **group_reduce_args,
                group=self.cp_group_gr,
                async_op=True,
            )
        else:
            if not self.fwd_out_lse_use_acc and partial_remote_out is not None:
                # the partial remote out and lse have NOT been reduced to
                # partial local out and lse by neither ffa fwd kernel nor group-reduce
                # thus we need to manually reduce here

                correct_attn_fwd_result(
                    out_list=[partial_local_out, partial_remote_out],
                    lse_list=[partial_local_lse, partial_remote_lse],
                    inplace=True,  # inplace reduce to the partial local (first) out and lse
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
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote dkv to local dkv for the given overlap stage

        Args:
            partial_remote_dkv (FusedOrTupleTensor, optional):
                partial remote dkv fused or tupled tensor, or None if skipped
            partial_local_dkv (FusedOrTupleTensor):
                partial local dkv fused or tupled tensor to be reduced
            ref_remote_dkv (FusedOrTupleTensor):
                reference remote dkv fused or tupled tensor, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_dkv_reduce_work (WorkWithPostProcessFn): partial dkv group-reduce work
        """
        assert self.concat_kv, "only support concat_kv"

        # get the group-reduce args for dkv
        group_reduce_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_reduce_args()

        # init remote dkv buffer
        if partial_remote_dkv is None:  # skipped
            partial_remote_dkv = torch.empty_like(
                ref_remote_dkv,
                dtype=max_fp_dtype(ref_remote_dkv.dtype, torch.float32)  # type: ignore[union-attr]
                if self.bwd_hp_reduce
                else ref_remote_dkv.dtype,  # type: ignore[union-attr]
            )
        elif not self.bwd_hp_reduce:
            partial_remote_dkv = partial_remote_dkv.to(ref_remote_dkv.dtype)  # type: ignore[union-attr]

        # launch group-reduce kernel
        partial_dkv_reduce_work = group_reduce_collective(
            input=partial_remote_dkv,
            output=partial_local_dkv,
            **group_reduce_args,
            group=self.cp_group_gr,
            async_op=True,
        )

        return partial_dkv_reduce_work

    @nvtx.instrument_nvtx
    def _reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor | None,
        partial_local_dq: torch.Tensor,
        ref_remote_dq: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """
        Reduce remote dq to local dq for the given overlap stage

        Args:
            partial_remote_dq (torch.Tensor, optional):
                partial remote dq in float32 dtype, or None if skipped
            partial_local_dq (torch.Tensor): partial local dq to be reduced
            ref_remote_dq (torch.Tensor):
                reference remote dq, to provide meta info like dtype and shape
            overlap_stage (int): current overlap stage

        Returns:
            partial_dq_reduce_work (WorkWithPostProcessFn): partial dq group-reduce work
        """
        if magi_attention.comm.is_qo_comm_enable():
            # get the group-reduce args for dq
            group_reduce_args = self.comm_meta.qo_group_collective_args_list[
                overlap_stage
            ].to_group_reduce_args()

            # init remote dq buffer
            if partial_remote_dq is None:  # skipped
                partial_remote_dq = torch.empty_like(
                    ref_remote_dq,
                    dtype=max_fp_dtype(ref_remote_dq.dtype, torch.float32)
                    if self.bwd_hp_reduce
                    else ref_remote_dq.dtype,
                )
            elif not self.bwd_hp_reduce:
                partial_remote_dq = partial_remote_dq.to(ref_remote_dq.dtype)

            # launch group-reduce kernel
            partial_dq_reduce_work = group_reduce_collective(
                input=partial_remote_dq,
                output=partial_local_dq,
                **group_reduce_args,
                group=self.cp_group_gr,
                async_op=True,
            )
        else:
            if not self.bwd_dq_use_acc and partial_remote_dq is not None:
                # the partial remote dq has NOT been reduced to partial local dq
                # by neither ffa bwd kernel nor group-reduce
                # thus we need to manually reduce here
                partial_local_dq.add_(partial_remote_dq)

            partial_dq_reduce_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take dq and return dq
            )

        return partial_dq_reduce_work

    def _maybe_concat_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> FusedOrTupleTensor:
        """
        Maybe concatenate k, v tensors
        into a fused or tupled kv along the seqlen dim
        """
        # TODO: whether can we pack kv togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([k, v], dim=0) if self.concat_kv else (k, v)

    def _maybe_chunk_kv(
        self,
        kv: FusedOrTupleTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maybe chunk the kv fused or tupled tensor
        into k, v tensor views along the seqlen dim
        """
        return torch.chunk(kv, 2, dim=0) if self.concat_kv else kv

    def _maybe_concat_qo_do(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
    ) -> FusedOrTupleTensor:
        """
        Maybe concatenate q, o, do tensors
        into a fused or tupled qo_do along the seqlen dim
        """
        # TODO: whether can we pack q, o, do togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([q, o, do], dim=0) if self.concat_qo_do else (q, o, do)

    def _maybe_chunk_qo_do(
        self,
        qo_do: FusedOrTupleTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maybe chunk the qo_do fused or tupled tensor
        into q, o, do tensor views along the seqlen dim
        """
        return torch.chunk(qo_do, 3, dim=0) if self.concat_qo_do else qo_do

    def _reset_work_list(self):
        self.remote_q_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # fwd
        self.remote_kv_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # fwd + bwd
        self.partial_out_lse_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # fwd
        self.remote_lse_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # bwd
        self.remote_qo_do_work_with_buffer_per_stage: list[WorkWithBuffer] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # bwd
        self.partial_dq_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # bwd
        self.partial_dkv_reduce_work_per_stage: list[WorkWithPostProcessFn] = [
            None  # type: ignore[list-item]
        ] * self.overlap_degree  # bwd

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
        dist_attn_runtime: DistAttnRuntime,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
    ):
        """
        Distributed Attention forward function

        Args:
            local_q (torch.Tensor): local q tensor
            local_k (torch.Tensor): local k tensor
            local_v (torch.Tensor): local v tensor
            dist_attn_runtime(DistAttnRuntime): dist attn runtime

            softmax_scale (float, optional): softmax scale.
                Defaults to None to use default value: 1/sqrt(head_dim)
            softcap (float, optional): softcap. Defaults to 0.

        Returns:
            local_out (torch.Tensor): local out tensor

            local_lse (torch.Tensor): local lse tensor

        Shape:
            local_q: [num_tokens_q_local, num_heads_q, head_dim]
            local_k: [num_tokens_kv_local, num_heads_kv, head_dim]
            local_v: [num_tokens_kv_local, num_heads_kv, head_dim]
            local_out: [num_tokens_q_local, num_heads_q, head_dim]
            local_lse: [num_tokens_q_local, num_heads_q]
        """
        # get local qkv and pre-fetch qkv for remote stage(s)
        local_q, local_kv = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
            local_q=local_q,
            local_kv=(local_k, local_v),
            overlap_stage=None,
        )

        # apply fwd partial attn with local qkv
        # overlapped with 0th pre-fetch
        partial_local_out, partial_local_lse = dist_attn_runtime.apply_fwd_partial_attn(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )
        assert partial_local_out is not None and partial_local_lse is not None

        # loop into remote stages
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote qkv prepared and pre-fetch (i+1)th remote qkv
            (
                curr_remote_q,
                curr_remote_kv,
            ) = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
                local_q=local_q,
                local_kv=local_kv,
                overlap_stage=ith_overlap_stage,
            )

            # apply fwd partial attn with ith remote qkv
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_out,
                partial_remote_lse,
            ) = dist_attn_runtime.apply_fwd_partial_attn(
                q=curr_remote_q,
                kv=curr_remote_kv,
                out_acc=partial_local_out
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                lse_acc=partial_local_lse
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                overlap_stage=ith_overlap_stage,
                softmax_scale=softmax_scale,
                softcap=softcap,
            )

            # reduce ith partial out with partial lse
            # overlapped with (i+1)th fwd partial attn and maybe (i+2)th pre-fetch
            dist_attn_runtime.reduce_partial_out_lse(
                partial_remote_out=partial_remote_out,
                partial_remote_lse=partial_remote_lse,
                partial_local_out=partial_local_out,
                partial_local_lse=partial_local_lse,
                ref_remote_out=curr_remote_q,
                overlap_stage=ith_overlap_stage,
            )

        # prepare reduced local out and lse
        # before returning from forward and saving for backward
        local_out, local_lse = dist_attn_runtime.prepare_reduced_local_out_lse(
            partial_local_out=partial_local_out,
            partial_local_lse=partial_local_lse,
            ref_local_out=local_q,
        )

        ctx.save_for_backward(local_q, local_kv, local_out, local_lse)
        ctx.dist_attn_runtime = dist_attn_runtime
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap

        return local_out, local_lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        local_q, local_kv, local_out, local_lse = ctx.saved_tensors
        dist_attn_runtime: DistAttnRuntime = ctx.dist_attn_runtime
        softmax_scale: float | None = ctx.softmax_scale
        softcap: float = ctx.softcap

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
        )

        # apply bwd partial attn with local qo_do,kv,lse
        # overlapped with 0th pre-fetch
        (
            partial_local_dq,
            partial_local_dkv,
        ) = dist_attn_runtime.apply_bwd_partial_attn(
            qo_do=local_qo_do,
            kv=local_kv,
            lse=local_lse,
            overlap_stage=None,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )
        assert partial_local_dq is not None and partial_local_dkv is not None

        # loop into remote stages
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
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
            )

            # apply bwd partial attn with ith remote qo_do,kv,lse
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_dq,
                partial_remote_dkv,
            ) = dist_attn_runtime.apply_bwd_partial_attn(
                qo_do=curr_remote_qo_do,
                kv=curr_remote_kv,
                lse=curr_remote_lse,
                dq_acc=partial_local_dq if dist_attn_runtime.bwd_dq_use_acc else None,
                overlap_stage=ith_overlap_stage,
                softmax_scale=softmax_scale,
                softcap=softcap,
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
            )

        # prepare reduced local dq,dk,dv
        # before returning from backward
        local_dq, local_dk, local_dv = dist_attn_runtime.prepare_reduced_local_dq_dk_dv(
            partial_local_dq=partial_local_dq,
            partial_local_dkv=partial_local_dkv,
            ref_local_dq=local_q,
            ref_local_dkv=local_kv,
        )

        return (
            local_dq,
            local_dk,
            local_dv,
            None,  # dist_attn_runtime
            None,  # softmax_scale
            None,  # softcap
        )


def dist_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dist_attn_runtime: DistAttnRuntime,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Distributed attention autograd function

    Args:
        q (torch.Tensor): local q
        k (torch.Tensor): local k
        v (torch.Tensor): local v
        dist_attn_runtime (DistAttnRuntime): distributed attention runtime

        softmax_scale (float, optional): softmax scale.
            Defaults to None to use default value: 1/sqrt(head_dim)
        softcap (float, optional): softcap. Defaults to 0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: local out and local lse

    Shapes:
        q: [num_tokens_q_local, num_heads_q, head_dim]
        k: [num_tokens_kv_local, num_heads_kv, head_dim]
        v: [num_tokens_kv_local, num_heads_kv, head_dim]
        out: [num_tokens_q_local, num_heads_q, head_dim]
        lse: [num_tokens_q_local, num_heads_q]
    """
    return DistAttnFunc.apply(
        q,
        k,
        v,
        dist_attn_runtime,
        softmax_scale,
        softcap,
    )
