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
from logging import getLogger

import torch

from magi_attention import env
from magi_attention.common import AttnForwardMeta

from .dist_attn_runtime import DistAttnRuntime
from .dist_attn_utils import (
    _prepare_attn_inputs,
    _restore_attn_output,
    create_kernel_barrier,
)

logger = getLogger(__name__)


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
        dist_attn_runtime._check_forward_args(global_sink, return_max_logits)

        if dist_attn_runtime.no_overlap and not dist_attn_runtime.skip_comm:
            return DistAttnFunc._no_overlap_forward(
                ctx,
                local_q=local_q,
                local_k=local_k,
                local_v=local_v,
                global_sink=global_sink,
                dist_attn_runtime=dist_attn_runtime,
                softmax_scale=softmax_scale,
                softcap=softcap,
                return_max_logits=return_max_logits,
            )

        # init kernel barrier for native grpcoll to ensure comm kernel is always preceded by compute kernel
        kernel_barrier_fetch = create_kernel_barrier(
            dist_attn_runtime.fwd_kernel_barrier_fetch_target
        )
        kernel_barrier_reduce = create_kernel_barrier(
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

        if dist_attn_runtime.save_tail_stage:
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
    def _no_overlap_forward(
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
        """No-overlap forward: blocking group_cast, concat KV, single attention call.
        Avoids LSE reduce entirely by computing attention over the full KV in one pass.
        """
        # -- Step 1: prepare local qkv (flatten head groups, maybe concat kv) --
        local_q, local_kv = dist_attn_runtime._maybe_flatten_local_qkv_head_groups(
            local_q=local_q,
            local_kv=(local_k, local_v),
        )
        local_kv = dist_attn_runtime._maybe_concat(
            *local_kv, need_concat=dist_attn_runtime.concat_kv
        )

        # -- Step 2: blocking group_cast to fetch all remote KV --
        remote_kv_work, remote_kv_buffer = dist_attn_runtime._fetch_remote_kv(
            local_kv=local_kv,
            overlap_stage=0,
        )
        remote_kv_buffer = remote_kv_work.wait_post_process(remote_kv_buffer)

        # -- Step 3: split local/remote into K and V, then concat --
        local_k_t, local_v_t = dist_attn_runtime._maybe_chunk(local_kv, num_chunks=2)
        remote_k_t, remote_v_t = dist_attn_runtime._maybe_chunk(
            remote_kv_buffer, num_chunks=2
        )
        full_k = torch.cat([local_k_t, remote_k_t], dim=0)
        full_v = torch.cat([local_v_t, remote_v_t], dim=0)

        # -- Step 4: get merged attn_arg (pre-built in CalcMeta.__post_init__) --
        merged_attn_arg = dist_attn_runtime.calc_meta.merged_attn_arg
        assert merged_attn_arg is not None
        _softmax_scale: float = (
            local_q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )
        if merged_attn_arg.can_skip(is_bwd=False):
            # An empty merged arg has no range to launch: return zero O and the
            # sink-only LSE.
            local_out, local_lse = dist_attn_runtime._init_out_lse_skipped_host_stage(
                q=local_q,
                head_dim_v=full_v.shape[-1],
                sink=global_sink,
            )
            meta = AttnForwardMeta(
                lse=local_lse,
                max_logits=dist_attn_runtime._init_max_logits_skipped_host_stage(
                    q=local_q,
                    return_max_logits=return_max_logits,
                ),
            )
        else:
            local_out, meta = dist_attn_runtime._launch_attn_fwd_kernel(
                q=local_q,
                k=full_k,
                v=full_v,
                sink=global_sink,
                out_acc=None,
                lse_acc=None,
                max_logits_acc=None,
                attn_arg=merged_attn_arg,
                softmax_scale=_softmax_scale,
                softcap=softcap,
                is_host_stage=True,
                return_max_logits=return_max_logits,
            )

        # -- Step 6: finalize output --
        local_out = local_out.to(local_q.dtype)
        local_lse = meta.lse

        (
            local_out,
            local_lse,
        ) = dist_attn_runtime._maybe_unflatten_local_out_lse_head_groups(
            local_out=local_out, local_lse=local_lse
        )

        if return_max_logits:
            local_max_logits = dist_attn_runtime.reduce_max_logits(
                partial_local_max_logits=meta.max_logits,
            )
        else:
            local_max_logits = None

        # -- Step 7: save for backward --
        dist_attn_runtime.save_tensors_for_bwd(
            ctx,
            local_q=local_q,
            local_kv=local_kv,
            local_out=local_out,
            local_lse=local_lse,
            last_stage_q=None,
            last_stage_kv=None,
            global_sink=global_sink,
        )
        ctx.dist_attn_runtime = dist_attn_runtime
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap

        return local_out, local_lse, local_max_logits

    @staticmethod
    def _no_overlap_backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        """No-overlap backward: blocking group_cast, concat KV, single backward call.
        The dkv for the remote portion is sent back via blocking group_reduce.
        """
        dist_attn_runtime: DistAttnRuntime = ctx.dist_attn_runtime
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

        # -- Step 1: flatten head groups for backward --
        (
            local_qo_do,
            local_lse,
        ) = dist_attn_runtime._maybe_flatten_local_qo_do_lse_head_groups(
            local_qo_do=(local_q, local_out, grad_output),
            local_lse=local_lse,
        )

        # -- Step 2: blocking group_cast to fetch all remote KV --
        remote_kv_work, remote_kv_buffer = dist_attn_runtime._fetch_remote_kv(
            local_kv=local_kv,
            overlap_stage=0,
        )
        remote_kv_buffer = remote_kv_work.wait_post_process(remote_kv_buffer)

        # -- Step 3: split local/remote into K and V, then concat --
        local_k_t, local_v_t = dist_attn_runtime._maybe_chunk(local_kv, num_chunks=2)
        remote_k_t, remote_v_t = dist_attn_runtime._maybe_chunk(
            remote_kv_buffer, num_chunks=2
        )
        full_k = torch.cat([local_k_t, remote_k_t], dim=0)
        full_v = torch.cat([local_v_t, remote_v_t], dim=0)
        local_k_seqlen = local_k_t.shape[0]

        # -- Step 4: get merged attn_arg (lazy-built in forward, reused here) --
        merged_attn_arg = dist_attn_runtime.calc_meta.merged_attn_arg
        assert (
            merged_attn_arg is not None
        ), "merged_attn_arg should have been built during forward"

        # -- Step 5: prepare tensors for backward --
        q, o, do = dist_attn_runtime._maybe_chunk(local_qo_do, num_chunks=3)
        _softmax_scale: float = (
            q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        )

        # -- Step 6: single backward call with merged arg --
        if merged_attn_arg.can_skip(is_bwd=True):
            # Empty merged arg: zero dq/dkv over local+remote KV and compute
            # dsink from the saved out/lse.
            (
                partial_dq,
                partial_dkv,
                partial_dsink,
            ) = dist_attn_runtime._init_dq_dkv_dsink_skipped_host_stage(
                qo_do=local_qo_do,
                kv=(full_k, full_v),
                lse=local_lse,
                sink=global_sink,
            )
        else:
            (
                partial_dq,
                partial_dkv,
                partial_dsink,
            ) = dist_attn_runtime._launch_attn_bwd_kernel(
                do=do,
                q=q,
                k=full_k,
                v=full_v,
                o=o,
                lse=local_lse,
                sink=global_sink,
                dq_acc=None,
                attn_arg=merged_attn_arg,
                softmax_scale=_softmax_scale,
                softcap=softcap,
                is_host_stage=True,
            )

        # -- Step 7: split dkv into local and remote parts --
        partial_dk, partial_dv = dist_attn_runtime._maybe_chunk(
            partial_dkv, num_chunks=2
        )
        local_dk = partial_dk[:local_k_seqlen]
        local_dv = partial_dv[:local_k_seqlen]
        remote_dk = partial_dk[local_k_seqlen:]
        remote_dv = partial_dv[local_k_seqlen:]

        if dist_attn_runtime.concat_dkv:
            local_dkv = torch.cat([local_dk, local_dv], dim=0)
            remote_dkv = torch.cat([remote_dk, remote_dv], dim=0)
        else:
            local_dkv = (local_dk, local_dv)
            remote_dkv = (remote_dk, remote_dv)

        # -- Step 8: blocking group_reduce to send remote dkv back --
        partial_dkv_reduce_work = dist_attn_runtime._reduce_partial_dkv(
            partial_remote_dkv=remote_dkv,
            partial_local_dkv=local_dkv,
            ref_remote_dkv=remote_kv_buffer,
            overlap_stage=0,
        )
        local_dkv = partial_dkv_reduce_work.wait_post_process(local_dkv)

        # -- Step 9: reduce dsink if required --
        dist_attn_runtime.reduce_partial_dsink(
            partial_global_dsink=partial_dsink,
        )
        global_dsink = dist_attn_runtime.partial_dsink_reduce_work.wait_post_process(
            partial_dsink
        )

        # -- Step 10: finalize gradients --
        local_dq = partial_dq.to(local_q.dtype)

        if dist_attn_runtime.concat_kv:
            kv_dtype = local_kv.dtype
        else:
            kv_dtype = local_kv[0].dtype

        if dist_attn_runtime.concat_dkv:
            local_dkv_final = (
                local_dkv.to(kv_dtype)
                if isinstance(local_dkv, torch.Tensor)
                else local_dkv
            )
            local_dk, local_dv = dist_attn_runtime._maybe_chunk(
                local_dkv_final, num_chunks=2
            )
        else:
            local_dk, local_dv = dist_attn_runtime._maybe_chunk(local_dkv, num_chunks=2)
            local_dk = local_dk.to(kv_dtype)
            local_dv = local_dv.to(kv_dtype)

        (
            local_dq,
            local_dk,
            local_dv,
        ) = dist_attn_runtime._maybe_unflatten_local_dqkv_head_groups(
            local_dq=local_dq,
            local_dk=local_dk,
            local_dv=local_dv,
        )

        dist_attn_runtime._reset_work_list()

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

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        dist_attn_runtime: DistAttnRuntime = ctx.dist_attn_runtime
        if dist_attn_runtime.no_overlap and not dist_attn_runtime.skip_comm:
            return DistAttnFunc._no_overlap_backward(ctx, grad_output, *args)
        if dist_attn_runtime.save_tail_stage:
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

        kernel_barrier_fetch = create_kernel_barrier(
            dist_attn_runtime.bwd_kernel_barrier_fetch_target
        )
        kernel_barrier_reduce = create_kernel_barrier(
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

        sink (torch.Tensor, optional): global sink tensor, fp32 (fp64 with
            fp64 inputs), replicated among cp ranks. Defaults to ``None`` to
            not apply attention sink.

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
    q, k, v, output_dtype = _prepare_attn_inputs(
        q, k, v, dist_attn_runtime.kernel_backend, env.general.precision()
    )

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

    out = _restore_attn_output(out, output_dtype)

    return out, AttnForwardMeta(lse=lse, max_logits=max_logits)
