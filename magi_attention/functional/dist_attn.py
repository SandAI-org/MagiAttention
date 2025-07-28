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

from logging import getLogger
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

import magi_attention
from magi_attention.comm.primitive import group_cast_collective, group_reduce_collective
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.meta.collection import AttnCalcMeta, CommMeta
from magi_attention.utils import (
    is_same_process_group,
    max_fp_dtype,
    nvtx,
    to_higher_fp_dtype,
)

from .flex_flash_attn import _flex_flash_attn_backward, _flex_flash_attn_forward
from .sdpa import sdpa_bwd, sdpa_fwd
from .utils import safe_subtract

logger = getLogger("magi_attention")


@nvtx.instrument_nvtx
@torch.compile
def correct_attn_lse(
    lse1: torch.Tensor,
    lse2: torch.Tensor,
) -> torch.Tensor:
    """
    Corrects the log sum exp tensor for online attention.

    Args:
        lse1(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
        lse2(torch.Tensor): log sum exp tensor, with shape: [batch_size, num_heads, seq_len]

    Returns:
        lse(torch.Tensor): corrected log sum exp tensor, with shape: [batch_size, num_heads, seq_len]
    """

    min_lse = to_higher_fp_dtype(torch.min(lse1, lse2), torch.float32)
    max_lse = to_higher_fp_dtype(torch.max(lse1, lse2), torch.float32)

    # formula derivation:
    # lse = log(exp(lse1) + exp(lse2))
    #     = lse1 + log(1 + exp(lse2 - lse1))
    #     = max_lse + log(1 + exp(min_lse - max_lse))
    #     = max_lse + log1p(exp(min_lse - max_lse))
    #     = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))

    return lse.to(lse1.dtype)


@nvtx.instrument_nvtx
@torch.compile
def correct_attn_output(
    o1: torch.Tensor,
    lse1: torch.Tensor,
    o2: torch.Tensor,
    lse2: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """
    Corrects the output tensor for online attention.

    Args:
        o1(torch.Tensor): local output tensor o1, with shape: [batch_size, seq_len, num_heads, head_dim]
        lse1(torch.Tensor): local lse for o1, with shape: [batch_size, num_heads, seq_len]
        o2(torch.Tensor): local output tensor o2, with shape: [batch_size, seq_len, num_heads, head_dim]
        lse2(torch.Tensor): local lse for o2, with shape: [batch_size, num_heads, seq_len]
        lse(torch.Tensor): global lse, with shape: [batch_size, num_heads, seq_len]

    Returns:
        o(torch.Tensor): corrected global output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
    """
    # formula: lsei_ = exp(lsei - lse)
    # shape: [b, h, s] -> [b, s, h] -> [b, s, h, 1]
    lse1_, lse2_ = [
        to_higher_fp_dtype(
            safe_subtract(lsei, lse).exp().transpose(-1, -2).unsqueeze(-1),
            torch.float32,
        )
        for lsei in [lse1, lse2]
    ]

    o = lse1_ * o1 + lse2_ * o2

    return o.to(o1.dtype)


@nvtx.instrument_nvtx
def result_correction(
    out_list: list[torch.Tensor],
    lse_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the attention result.

    Args:
        out_list(list[torch.Tensor]):
        lse_list(list[torch.Tensor]):

    Returns:
        out(torch.Tensor):
        lse(torch.Tensor):

    Shape:
        - out: [num_tokens_q, num_heads, head_dim]
        - lse: [num_heads, num_tokens_q]
    """
    if len(lse_list) == 1:
        # NOTE: if there is only one out and lse,
        # we just return them directly, no need to correct
        return out_list[0], lse_list[0]

    curr_lse = None
    curr_out = None

    for i in range(len(lse_list) - 1):
        if i == 0:
            curr_lse = correct_attn_lse(lse_list[0], lse_list[1])
            curr_out = correct_attn_output(
                out_list[0], lse_list[0], out_list[1], lse_list[1], curr_lse
            )
        else:
            original_lse = curr_lse
            original_out = curr_out
            curr_lse = correct_attn_lse(original_lse, lse_list[i + 1])
            curr_out = correct_attn_output(
                original_out,
                original_lse,
                out_list[i + 1],
                lse_list[i + 1],
                curr_lse,
            )

    return curr_out, curr_lse


class DistFlashAttnRuntime:
    """
    Runtime class for Distributed Flash Attention.

    Args:
        config (DistFlashAttnConfig): Static configuration for distributed Flash Attention
        runtime_meta (DistFlashAttnRuntimeMeta): Runtime metadata for distributed Flash Attention


    NOTE:
        A new DistFlashAttnRuntime should be instantiated for each forward pass
        This runtime instance provides schedulable primitives for each layer's forward pass:
            - fetch_remote_kv: Fetch remote kv buffer from other ranks to local
            - do_attn_partially: Compute part of the attention result
    """

    def __init__(
        self,
        comm_meta: CommMeta,
        calc_meta: AttnCalcMeta,
        cp_group_kv: dist.ProcessGroup,
        cp_group_dkv: dist.ProcessGroup,
    ):
        self.comm_meta = comm_meta
        self.calc_meta = calc_meta
        self.cp_group_kv = cp_group_kv
        self.cp_group_dkv = cp_group_dkv
        self.deterministic = magi_attention.is_deterministic_mode_enable()

        # NOTE: get the real overlap degree from comm meta
        # instead of the initial one from overlap config
        self.overlap_degree = comm_meta.overlap_degree

        assert self.overlap_degree >= 1, f"{self.overlap_degree=} must be >= 1"

        # when enabling FFA fwd inplace correct and not using sdpa backend
        # we will use accumulative buffer for forward out and lse
        # to avoid the storage of partial results and the memory-bound `result_correction`
        self.fwd_use_acc = (
            magi_attention.functional.is_ffa_fwd_inplace_correct_enable()
            and not magi_attention.is_sdpa_backend_enable()
        )

    @nvtx.instrument_nvtx
    def fetch_remote_kv(
        self,
        local_kv: torch.Tensor,
        overlap_stage: int,
    ) -> tuple[WorkWithPostProcessFn, torch.Tensor]:
        """
        Fetch remote kv buffer from other ranks to local, and return the corresponding Work and buffer

        Args:
            local_kv(torch.Tensor): the concatenated local kv tensor
            overlap_stage(int): current overlap stage

        Returns:
            remote_kv_work(WorkWithPostProcessFn): communication handle, used to wait for communication completion
            remote_kv_buffer(torch.Tensor): remote kv buffer

        Shape:
            - local_kv: [num_tokens_kv_local, num_heads, head_dim]
            - remote_kv_buffer: [num_tokens_kv_remote_i, num_heads, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        _, num_heads, head_dim = local_kv.shape
        dtype = local_kv.dtype
        device = local_kv.device

        group_collective_args = self.comm_meta.group_collective_args_list[overlap_stage]

        remote_kv_buffer = torch.empty(
            [
                self.comm_meta.num_remote_tokens_per_stage[overlap_stage] * 2,
                num_heads,
                head_dim,
            ],
            dtype=dtype,
            device=device,
        )

        remote_kv_work = group_cast_collective(
            input=local_kv,
            output=remote_kv_buffer,
            **group_collective_args.to_group_cast_args(),
            group=self.cp_group_kv,
            async_op=True,
        )

        return remote_kv_work, remote_kv_buffer

    @nvtx.instrument_nvtx
    def attn_fwd_partial(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        overlap_stage: int | None = None,
        deterministic: bool = False,
        out_acc: torch.Tensor | None = None,
        lse_acc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute a part of the attention result

        Args:
            q(torch.Tensor): local q
            kv(torch.Tensor): current kv
            overlap_stage(int, optional): Current overlap stage,
                if None, it means local attention, otherwise it means remote attention
            deterministic(bool): Whether to use deterministic algorithm
            out_acc (torch.Tensor, optional): accumulative buffer for out
            lse_acc (torch.Tensor, optional): accumulative buffer for lse

        Returns:
            out(torch.Tensor | None): partial out, or None if skipped
            lse(torch.Tensor | None): partial log-sum-exp, or None if skipped
        Shape:
            - q: [num_tokens_q, num_heads, head_dim]
            - kv: [num_tokens_kv, num_heads, head_dim]
            - out: [num_tokens_q, num_heads, head_dim]
            - lse: [num_heads, num_tokens_q]
        """
        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        # Calculate attention
        if attn_arg.can_skip(is_bwd=False):
            out, lse = (out_acc, lse_acc) if self.fwd_use_acc else (None, None)
        else:
            k, v = self.chunk_kv(kv)
            if magi_attention.is_sdpa_backend_enable():
                out, lse = sdpa_fwd(
                    q,
                    k,
                    v,
                    attn_arg=attn_arg,
                )
            else:
                with nvtx.add_nvtx_event(
                    f"attn-fwd: area={attn_arg.total_area} | "
                    f"qr={attn_arg.q_ranges} | kr={attn_arg.k_ranges}"
                ):
                    out, lse, *rest = _flex_flash_attn_forward(
                        q=q,
                        k=k,
                        v=v,
                        out=out_acc,
                        lse=lse_acc,
                        **attn_arg.to_ffa_args(is_bwd=False),
                        merge_q_ranges=None,
                        qk_map=None,
                        fwd_unique_count=None,
                        softmax_scale=q.shape[-1] ** -0.5,
                        deterministic=deterministic,
                        softcap=0.0,
                        sm_margin=magi_attention.comm.ffa_fwd_sm_margin_save_for_comm(),
                        # NOTE: increase the partial out precision temporarily,
                        # to reduce the error caused by the out correction
                        out_type=max_fp_dtype(q.dtype, torch.float32),
                        # NOTE: when using accumulative buffer, we need to always enable atomic reduction
                        # unless it is the first call when accumulative buffer is still None
                        disable_fwd_atomic_reduction=(
                            attn_arg.disable_fwd_atomic_reduction and out_acc is None
                        ),
                    )

        return out, lse

    @nvtx.instrument_nvtx
    def attn_bwd_partial(
        self,
        do: torch.Tensor,
        q: torch.Tensor,
        kv: torch.Tensor,
        o: torch.Tensor,
        lse: torch.Tensor,
        overlap_stage: int | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Apply ffa bwd kernel to get partial dqkv.
        Returns:
            partial_dq(torch.Tensor): partial dq
            partial_dkv(torch.Tensor): partial dkv
            skip_attn(bool): Whether to skip attention computation,
                NOTE: if True, the partial_dq and partial_dkv will be random initialized
        """

        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        skip_attn = attn_arg.can_skip(is_bwd=True)

        if skip_attn:
            partial_dq, partial_dkv = torch.empty_like(q), torch.empty_like(kv)
        else:
            k, v = self.chunk_kv(kv)
            if magi_attention.is_sdpa_backend_enable():
                partial_dq, partial_dk, partial_dv = sdpa_bwd(
                    do=do,
                    q=q,
                    k=k,
                    v=v,
                    o=o,
                    lse=lse,
                    attn_arg=attn_arg,
                )
            else:
                # TODO: pre-allocate the dkdv buffer to avoid dkv concat
                partial_dq, partial_dk, partial_dv, *rest = _flex_flash_attn_backward(
                    dout=do,
                    q=q,
                    k=k,
                    v=v,
                    out=o,
                    softmax_lse=lse,
                    dq=None,
                    dk=None,
                    dv=None,
                    dq_type=torch.float32,
                    dk_type=torch.float32,
                    dv_type=torch.float32,
                    **attn_arg.to_ffa_args(is_bwd=True),
                    merge_k_ranges=None,
                    bwd_kq_map=None,
                    bwd_unique_count=None,
                    softmax_scale=q.shape[-1] ** -0.5,
                    deterministic=deterministic,
                    softcap=0.0,
                    disable_bwd_dkv_atomic_reduction=False,  # FIXME: set dynamically
                    sm_margin=magi_attention.comm.ffa_bwd_sm_margin_save_for_comm(),
                )
            partial_dkv = torch.cat([partial_dk, partial_dv], dim=0).to(kv.dtype)

        return partial_dq, partial_dkv, skip_attn

    @nvtx.instrument_nvtx
    def reduce_partial_dkv(
        self,
        partial_remote_dkv: torch.Tensor,
        partial_local_dkv: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """reduce remote dkv to add to local dkv for the given overlap stage."""
        group_collective_args = self.comm_meta.group_collective_args_list[overlap_stage]

        partial_local_dkv_work = group_reduce_collective(
            input=partial_remote_dkv,
            output=partial_local_dkv,
            **group_collective_args.to_group_reduce_args(),
            group=self.cp_group_dkv,
            async_op=True,
        )

        return partial_local_dkv_work

    @nvtx.instrument_nvtx
    def reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor,
        partial_local_dq: torch.Tensor,
    ) -> torch.Tensor:
        return partial_local_dq.add_(partial_remote_dq)

    @staticmethod
    def concat_kv(
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """concatenate k, v tensors into a single coalesced kv"""
        # TODO: whether can we pack kv togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([k, v], dim=0)

    @staticmethod
    def chunk_kv(
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """chunk the kv tensor into k, v tensor views"""
        return torch.chunk(kv, 2, dim=0)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DistFlashAttnRuntime):
            return False
        return (
            is_same_process_group(self.cp_group_kv, other.cp_group_kv)
            and is_same_process_group(self.cp_group_dkv, other.cp_group_dkv)
            and (self.comm_meta, self.calc_meta, self.deterministic)
            == (other.comm_meta, other.calc_meta, other.deterministic)
        )


class DistFlashAttnFunc(torch.autograd.Function):
    """Distributed Flash Attention Function"""

    @staticmethod
    def forward(
        ctx,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        dist_attn_runtime: DistFlashAttnRuntime,
    ):
        """
        Distributed Flash Attention forward function

        Args:
            local_q(torch.Tensor):
            local_k(torch.Tensor):
            local_v(torch.Tensor):
            dist_attn_runtime(DistFlashAttnRuntime):

        Returns:
            out(torch.Tensor):

        Shape:
            - local_q: [num_tokens_q_local, num_heads, head_dim]
            - local_k: [num_tokens_k_local, num_heads, head_dim]
            - local_v: [num_tokens_v_local, num_heads, head_dim]
        """

        if not dist_attn_runtime.fwd_use_acc:
            out_list = []
            lse_list = []

        # cat local k, v into a single coalesced kv
        local_kv = dist_attn_runtime.concat_kv(local_k, local_v)

        if magi_attention.is_cuda_device_max_connections_one():
            # pre-fetch 0th remote kv
            (
                remote_kv_work,
                remote_kv_buffer,
            ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)
        else:
            # when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote-kv comms in advance of ffa fwd
            # and ffa fwd can still overlap with these comms
            # with the support of `sm_margin`, thx to persistent kernel design
            remote_kv_works_with_buffers = [
                dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]

        # do attn fwd with local kv
        # overlapped with 0th remote kv comm
        out, lse = dist_attn_runtime.attn_fwd_partial(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )

        if not dist_attn_runtime.fwd_use_acc and out is not None:
            out_list.append(out)
            lse_list.append(lse)

        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote kv prepared
            if magi_attention.is_cuda_device_max_connections_one():
                curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)
            else:
                curr_remote_work, curr_remote_buffer = remote_kv_works_with_buffers[
                    ith_overlap_stage
                ]
                curr_remote_kv = curr_remote_work.wait_post_process(curr_remote_buffer)

            # pre-fetch (i+1)th remote kv
            if magi_attention.is_cuda_device_max_connections_one():
                if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                    (
                        remote_kv_work,
                        remote_kv_buffer,
                    ) = dist_attn_runtime.fetch_remote_kv(
                        local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                    )

            # do attn fwd with ith remote kv
            # overlapped with (i+1)th remote kv comm
            out, lse = dist_attn_runtime.attn_fwd_partial(
                q=local_q,
                kv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
                out_acc=out if dist_attn_runtime.fwd_use_acc else None,
                lse_acc=lse if dist_attn_runtime.fwd_use_acc else None,
            )

            if not dist_attn_runtime.fwd_use_acc and out is not None:
                out_list.append(out)
                lse_list.append(lse)

        # do result correction to get final out and lse
        if not dist_attn_runtime.fwd_use_acc:
            out, lse = result_correction(
                out_list=out_list,
                lse_list=lse_list,
            )

        if out is None:  # attn computation are all skipped
            # NOTE: We cannot use torch.empty_like here, because empty_like may contain nan values,
            #       and once gradients between different tokens need to be reduced, the nan values
            #       from pad tokens would interfere with the gradients of other tokens
            out = torch.zeros_like(local_q)
        else:
            # NOTE: since we've increased the precision of partial out for correction
            # here we need to downcast to q dtype to both return and save for backward
            out = out.to(local_q.dtype)

        ctx.save_for_backward(local_q, local_kv, out, lse)
        ctx.dist_attn_runtime = dist_attn_runtime

        return out, lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        local_q, local_kv, out, final_lse = ctx.saved_tensors
        dist_attn_runtime: DistFlashAttnRuntime = ctx.dist_attn_runtime

        if magi_attention.is_cuda_device_max_connections_one():
            # pre-fetch 0th remote kv
            (
                remote_kv_work,
                remote_kv_buffer,
            ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)
        else:
            # when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote-kv comms in advance of ffa bwd
            # and ffa bwd can still overlap with these comms
            # with the support of `sm_margin`, thx to persistent kernel design
            remote_kv_works_with_buffers = [
                dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]
        # do attn bwd with local kv
        # overlapped with 0th remote kv comm
        (
            partial_local_dq,
            partial_local_dkv,
            skip_attn,
        ) = dist_attn_runtime.attn_bwd_partial(
            do=grad_output,
            q=local_q,
            kv=local_kv,
            o=out,
            lse=final_lse,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )

        if skip_attn:
            # NOTE: if local_dq and local_dkv calculation are skipped, we need to zeros initialize them.
            partial_local_dq = torch.zeros_like(local_q)
            partial_local_dkv = torch.zeros_like(local_kv)

        partial_local_dkv_work = WorkWithPostProcessFn()
        partial_local_dkv_works = []
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote kv prepared
            if magi_attention.is_cuda_device_max_connections_one():
                curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)
            else:
                curr_remote_work, curr_remote_buffer = remote_kv_works_with_buffers[
                    ith_overlap_stage
                ]
                curr_remote_kv = curr_remote_work.wait_post_process(curr_remote_buffer)

            # pre-fetch (i+1)th remote kv
            if magi_attention.is_cuda_device_max_connections_one():
                if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                    (
                        remote_kv_work,
                        remote_kv_buffer,
                    ) = dist_attn_runtime.fetch_remote_kv(
                        local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                    )

            # do attn bwd with ith remote kv
            # overlapped with (i+1)th remote kv comm
            (
                partial_remote_dq,
                partial_remote_dkv,
                skip_attn,
            ) = dist_attn_runtime.attn_bwd_partial(
                do=grad_output,
                q=local_q,
                kv=curr_remote_kv,
                o=out,
                lse=final_lse,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
            )

            # reduce ith partial dkv
            # NOTE: even if skip_attn is True, we still need to launch the group_reduce_collective,
            # because not all ranks are skipped.
            partial_local_dkv_work = dist_attn_runtime.reduce_partial_dkv(
                partial_remote_dkv=partial_remote_dkv,
                partial_local_dkv=partial_local_dkv,
                overlap_stage=ith_overlap_stage,
            )

            partial_local_dkv_works.append(partial_local_dkv_work)

            # NOTE: since dq reduce is doing on local rank, if skip_attn is True,
            # we just skip the reduce operation.
            if not skip_attn:
                # reduce ith partial dq, overlapped with ith remote dkv comm
                partial_local_dq = dist_attn_runtime.reduce_partial_dq(
                    partial_remote_dq=partial_remote_dq,
                    partial_local_dq=partial_local_dq,
                )

        # wait for last partial dkv reduced
        for partial_local_dkv_work in partial_local_dkv_works:
            partial_local_dkv = partial_local_dkv_work.wait_post_process(
                partial_local_dkv
            )

        # chunk final dkv into dk and dv
        partial_local_dk, partial_local_dv = dist_attn_runtime.chunk_kv(
            partial_local_dkv
        )

        return partial_local_dq, partial_local_dk, partial_local_dv, None, None


def dist_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dist_attn_runtime: DistFlashAttnRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    return DistFlashAttnFunc.apply(q, k, v, dist_attn_runtime)
