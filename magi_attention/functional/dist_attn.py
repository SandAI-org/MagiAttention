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
from magi_attention.comm.primitive import group_cast_collective, group_reduce_collective
from magi_attention.comm.work import WorkWithPostProcessFn
from magi_attention.meta.collection import AttnCalcMeta, CommMeta
from magi_attention.utils import is_same_process_group, max_fp_dtype, nvtx

from .flex_flash_attn import _flex_flash_attn_backward, _flex_flash_attn_forward
from .sdpa import sdpa_bwd, sdpa_fwd
from .utils import correct_attn_fwd_result

FusedOrTupleTensor: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


class DistAttnRuntime:
    """
    Runtime class for Distributed Flash Attention.

    Args:
        comm_meta (CommMeta): the communication metadata
        calc_meta (AttnCalcMeta): the calculation metadata
        cp_group_gc (dist.ProcessGroup): the cp group for group-cast
        cp_group_gr (dist.ProcessGroup): the cp group for group-reduce
    """

    def __init__(
        self,
        comm_meta: CommMeta,
        calc_meta: AttnCalcMeta,
        cp_group_gc: dist.ProcessGroup,
        cp_group_gr: dist.ProcessGroup,
    ):
        self.comm_meta = comm_meta
        self.calc_meta = calc_meta
        self.cp_group_gc = cp_group_gc
        self.cp_group_gr = cp_group_gr
        self.deterministic = magi_attention.is_deterministic_mode_enable()
        self.overlap_degree = comm_meta.overlap_degree

        # NOTE: when enabling FFA fwd inplace correct w/o using sdpa backend nor qo comm
        # we will use accumulative buffer for forward out and lse
        # to avoid the storage of partial results and the memory-bound `result_correction`
        self.fwd_use_acc = (
            magi_attention.functional.is_ffa_fwd_inplace_correct_enable()
            and not magi_attention.is_sdpa_backend_enable()
            and not magi_attention.comm.is_qo_comm_enable()
        )

        # NOTE: when not using sdpa backend nor qo comm
        # we will use accumulative buffer for bwd dq
        # to avoid the outside sum-reduce
        self.bwd_use_acc = (
            not magi_attention.is_sdpa_backend_enable()
            and not magi_attention.comm.is_qo_comm_enable()
        )

        # NOTE: when enabling FFA bwd high precision reduce, we will no longer downcast partial dkv to kv dtype
        # before reducing among ranks, increasing the precision at the cost of double comm overhead
        self.bwd_hp_reduce = (
            magi_attention.functional.is_ffa_bwd_high_precision_reduce_enable()
            and not magi_attention.is_sdpa_backend_enable()
        )

        # NOTE: for now, we concat q, o, do when and only when qo comm is enabled
        self.concat_qo_do = magi_attention.comm.is_qo_comm_enable()

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

        # get the group-cast args for kv
        # NOTE: we concat kv along seqlen dim,
        # and the group_collective args already handle this behind
        # thus we only need to handle the num_remote_kv_tokens_per_stage by multiplying 2
        group_cast_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_kv_seqlen = (
            self.comm_meta.num_remote_kv_tokens_per_stage[overlap_stage] * 2
        )

        # init remote kv buffer
        remote_kv_buffer = torch.empty(
            remote_kv_seqlen,
            num_heads,
            head_dim,
            dtype=local_kv.dtype,
            device=local_kv.device,
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
    def fetch_remote_q(
        self,
        local_q: torch.Tensor,
        overlap_stage: int,
    ) -> tuple[WorkWithPostProcessFn, torch.Tensor]:
        """
        Fetch remote q buffer from other ranks to local, and return the corresponding Work and buffer

        Args:
            local_q(torch.Tensor): the local q tensor
            overlap_stage(int): current overlap stage

        Returns:
            remote_q_work(WorkWithPostProcessFn): communication handle, used to wait for communication completion
            remote_q_buffer(torch.Tensor): remote q buffer

        Shape:
            - local_q: [num_tokens_q_local, num_heads, head_dim]
            - remote_q_buffer: [num_tokens_q_remote_i, num_heads, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        if not magi_attention.comm.is_qo_comm_enable():
            remote_q_buffer = local_q
            remote_q_work = WorkWithPostProcessFn(post_process_fn=lambda x: x)
            return remote_q_work, remote_q_buffer

        _, num_heads, head_dim = local_q.shape

        # get the group-cast args for q
        # HACK: we concat q,o,do along seqlen dim,
        # and the group_collective args for qo already handle this behind
        # thus we have to borrow the args for lse, which is the same for q only
        group_cast_args = self.comm_meta.lse_group_collective_args_list[
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
    def fetch_remote_qo_do(
        self,
        local_qo_do: FusedOrTupleTensor,
        overlap_stage: int,
    ) -> tuple[WorkWithPostProcessFn, FusedOrTupleTensor]:
        """
        Fetch remote q, o, do buffer from other ranks to local, and return the corresponding Work and buffer

        Args:
            local_qo_do(FusedOrTupleTensor): the local q, o, do (fused or tupled) tensor
            overlap_stage(int): current overlap stage

        Returns:
            remote_qo_do_work(WorkWithPostProcessFn):
                communication handle, used to wait for communication completion
            remote_qo_do_buffer(FusedOrTupleTensor): remote q, o, do buffer

        Shape:
            - local_qo_do: [num_tokens_qo_do_local, num_heads, head_dim]
            - remote_qo_do_buffer: [num_tokens_qo_do_remote_i, num_heads, head_dim],
                for i = 0, 1, ..., overlap_degree - 1
        """

        if not magi_attention.comm.is_qo_comm_enable():
            assert isinstance(local_qo_do, tuple)
            remote_qo_do_buffer = local_qo_do
            remote_qo_do_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take q,o,do and return q,o,do
            )
            return remote_qo_do_work, remote_qo_do_buffer

        assert isinstance(local_qo_do, torch.Tensor)
        _, num_heads, head_dim = local_qo_do.shape

        # get the group-cast args for q
        # NOTE: we concat q,o,do along seqlen dim,
        # and the group_collective args already handle this behind
        # thus we only need to handle the num_remote_qo_tokens_per_stage by multiplying 3
        group_cast_args = self.comm_meta.qo_group_collective_args_list[
            overlap_stage
        ].to_group_cast_args()
        remote_q_seqlen = self.comm_meta.num_remote_qo_tokens_per_stage[overlap_stage]
        remote_qo_do_seqlen = remote_q_seqlen * 3

        # init remote q,o,do output buffer
        remote_qo_do_buffer = torch.empty(
            remote_qo_do_seqlen,
            num_heads,
            head_dim,
            dtype=local_qo_do.dtype,
            device=local_qo_do.device,
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
    def fetch_remote_lse(
        self,
        local_lse: torch.Tensor,
        overlap_stage: int,
    ) -> tuple[WorkWithPostProcessFn, torch.Tensor]:
        """
        Fetch remote lse buffer from other ranks to local, and return the corresponding Work and buffer

        Args:
            local_lse(torch.Tensor): the local lse tensor
            overlap_stage(int): current overlap stage

        Returns:
            remote_lse_work(WorkWithPostProcessFn):
                communication handle, used to wait for communication completion
            remote_lse_buffer(torch.Tensor): remote lse buffer

        Shape:
            - local_lse: [num_heads, num_tokens_q_local]
            - remote_lse_buffer: [num_heads, num_tokens_q_remote_i],
                for i = 0, 1, ..., overlap_degree - 1
        """

        if not magi_attention.comm.is_qo_comm_enable():
            remote_lse_buffer = local_lse
            remote_lse_work = WorkWithPostProcessFn(
                post_process_fn=lambda x: x  # take lse and return lse
            )
            return remote_lse_work, remote_lse_buffer

        # FIXME: for now, ffa uses lse with the shape: [nhq, sq]
        # for both fwd and bwd, which is not suitable for comm
        local_lse = local_lse.transpose(0, 1).contiguous()
        _, num_heads = local_lse.shape

        # get the group-cast args for lse
        group_cast_args = self.comm_meta.lse_group_collective_args_list[
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

        # append post_process_fn to transpose lse from [sq, nhq] to [nhq, sq]
        orig_post_process_fn = remote_lse_work.post_process_fn
        remote_lse_work.post_process_fn = (
            lambda x: orig_post_process_fn(x).transpose(0, 1).contiguous()
        )

        return remote_lse_work, remote_lse_buffer

    @nvtx.instrument_nvtx
    def attn_fwd_partial(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out_acc: torch.Tensor | None = None,
        lse_acc: torch.Tensor | None = None,
        overlap_stage: int | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute a part of the attention result

        Args:
            q(torch.Tensor): local q
            kv(torch.Tensor): current kv
            out_acc (torch.Tensor, optional): accumulative buffer for out
            lse_acc (torch.Tensor, optional): accumulative buffer for lse
            overlap_stage(int, optional): Current overlap stage,
                if None, it means local attention, otherwise it means remote attention
            deterministic(bool): Whether to use deterministic algorithm

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
                    out, lse = _flex_flash_attn_forward(
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
                        softmax_scale=q.shape[-1] ** -0.5,
                        deterministic=deterministic,
                        softcap=0.0,
                        sm_margin=magi_attention.comm.ffa_fwd_sm_margin_save_for_comm(),
                        # NOTE: increase the partial out precision temporarily,
                        # to reduce the error caused by the out correction
                        out_type=torch.float32,
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
        qo_do: FusedOrTupleTensor,
        kv: torch.Tensor,
        lse: torch.Tensor,
        dq_acc: torch.Tensor | None = None,
        overlap_stage: int | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Apply ffa bwd kernel to get partial dq, dkv

        Args:
            qo_do (FusedOrTupleTensor): q, o, do fused or tupled tensor
            kv (torch.Tensor): current kv
            lse (torch.Tensor): partial lse
            dq_acc (torch.Tensor, optional): accumulative buffer for dq
            overlap_stage(int, optional): Current overlap stage,
                if None, it means local attention, otherwise it means remote attention
            deterministic(bool): Whether to use deterministic algorithm

        Returns:
            partial_dq(torch.Tensor): partial dq, or None if skipped
            partial_dkv(torch.Tensor): partial dkv, or None if skipped
        """

        if overlap_stage is None:
            attn_arg = self.calc_meta.local_attn_arg
        else:
            attn_arg = self.calc_meta.remote_attn_args_list[overlap_stage]

        if attn_arg.can_skip(is_bwd=True):
            partial_dq = dq_acc if self.bwd_use_acc else None
            partial_dkv = None
        else:
            q, o, do = self.maybe_chunk_qo_do(qo_do)
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
                partial_dkv = self.concat_kv(partial_dk, partial_dv)
            else:
                # NOTE: we need to zero-initialize partial_dkv since it needs to be reduced
                # and also increase the partial dkv precision temporarily,
                # to reduce the error caused by the out correction
                partial_dkv = torch.zeros_like(kv, dtype=torch.float32)
                partial_dk, partial_dv = self.chunk_kv(partial_dkv)
                partial_dq, partial_dk, partial_dv, *rest = _flex_flash_attn_backward(
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
                    softmax_scale=q.shape[-1] ** -0.5,
                    deterministic=deterministic,
                    softcap=0.0,
                    disable_bwd_dkv_atomic_reduction=attn_arg.disable_bwd_dkv_atomic_reduction,
                    sm_margin=magi_attention.comm.ffa_bwd_sm_margin_save_for_comm(),
                )

        return partial_dq, partial_dkv

    @nvtx.instrument_nvtx
    def reduce_partial_out(
        self,
        partial_remote_out: torch.Tensor | None,
        partial_remote_lse: torch.Tensor | None,
        partial_local_out: torch.Tensor,
        partial_local_lse: torch.Tensor,
        ref_remote_out: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """reduce remote dkv to add to local dkv for the given overlap stage.

        Args:
            partial_remote_out(torch.Tensor, optional):
                partial remote out in float32 dtype, or None if skipped
            partial_remote_lse(torch.Tensor, optional):
                partial remote lse in float32 dtype, or None if skipped
            partial_local_out(torch.Tensor): partial local out to be reduced
            partial_local_lse(torch.Tensor): partial local lse to be reduced
            ref_remote_out(torch.Tensor):
                reference remote dkv, to provide meta info like dtype and shape
            overlap_stage(int): current overlap stage

        Returns:
            partial_out_reduce_work(WorkWithPostProcessFn): partial out group-reduce work

        """

        if not magi_attention.comm.is_qo_comm_enable():
            partial_out_reduce_work = WorkWithPostProcessFn(
                post_process_fn=lambda *x: x  # take out, lse and return out, lse
            )
            return partial_out_reduce_work

        raise NotImplementedError(
            "TODO: implement the group-reduce with attn-partial reduce"
        )

    @nvtx.instrument_nvtx
    def reduce_partial_dkv(
        self,
        partial_remote_dkv: torch.Tensor | None,
        partial_local_dkv: torch.Tensor,
        ref_remote_dkv: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        """reduce remote dkv to add to local dkv for the given overlap stage.

        Args:
            partial_remote_dkv(torch.Tensor, optional):
                partial remote dkv in float32 dtype, or None if skipped
            partial_local_dkv(torch.Tensor): partial local dkv to be reduced
            ref_remote_dkv(torch.Tensor):
                reference remote dkv, to provide meta info like dtype and shape
            overlap_stage(int): current overlap stage

        Returns:
            partial_dkv_reduce_work(WorkWithPostProcessFn): partial dkv group-reduce work

        """
        # get the group-reduce args for dkv
        # NOTE: we concat kv along seqlen dim,
        # and the group_collective args already handle this behind
        group_reduce_args = self.comm_meta.kv_group_collective_args_list[
            overlap_stage
        ].to_group_reduce_args()

        # init remote dkv buffer
        if partial_remote_dkv is None:  # skipped
            partial_remote_dkv = torch.empty_like(
                ref_remote_dkv,
                dtype=torch.float32 if self.bwd_hp_reduce else ref_remote_dkv.dtype,
            )
        elif not self.bwd_hp_reduce:
            partial_remote_dkv = partial_remote_dkv.to(ref_remote_dkv.dtype)

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
    def reduce_partial_dq(
        self,
        partial_remote_dq: torch.Tensor | None,
        partial_local_dq: torch.Tensor,
        ref_remote_dq: torch.Tensor,
        overlap_stage: int,
    ) -> WorkWithPostProcessFn:
        # NOTE: no need to reduce partial_remote_dq for ffa backend
        # since it is already reduced to partial_local_dq in the ffa bwd kernel
        if self.bwd_use_acc:
            # the local dq has already been reduced to partial_local_dq by ffa bwd
            partial_dq_reduce_work = WorkWithPostProcessFn(post_process_fn=lambda x: x)
        elif magi_attention.comm.is_qo_comm_enable():
            # get the group-reduce args for dq
            # HACK: we concat q,o,do along seqlen dim,
            # and the group_collective args for qo already handle this behind
            # thus we have to borrow the args for lse, which is the same for dq only
            group_reduce_args = self.comm_meta.lse_group_collective_args_list[
                overlap_stage
            ].to_group_reduce_args()

            # init remote dq buffer
            if partial_remote_dq is None:  # skipped
                partial_remote_dq = torch.empty_like(
                    ref_remote_dq,
                    dtype=torch.float32 if self.bwd_hp_reduce else ref_remote_dq.dtype,
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
            if partial_remote_dq is not None:
                # the local dq is reduced by neither ffa bwd nor group-reduce
                # thus we need to reduce manually from current partial_remote_dq
                partial_local_dq.add_(partial_remote_dq)
            partial_dq_reduce_work = WorkWithPostProcessFn(post_process_fn=lambda x: x)

        return partial_dq_reduce_work

    def concat_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """concatenate k, v tensors into a single coalesced kv"""
        # TODO: whether can we pack kv togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([k, v], dim=0)

    def chunk_kv(
        self,
        kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """chunk the kv tensor into k, v tensor views"""
        return torch.chunk(kv, 2, dim=0)

    def maybe_concat_qo_do(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
    ) -> FusedOrTupleTensor:
        """maybe concatenate q, o, do tensors into a single fused or tupled qo_do"""
        # TODO: whether can we pack q, o, do togather along certain dim
        # to enhance the performance of ffa kernel
        return torch.cat([q, o, do], dim=0) if self.concat_qo_do else (q, o, do)

    def maybe_chunk_qo_do(
        self,
        qo_do: FusedOrTupleTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """maybe chunk the qo_do fused tensor into q, o, do tensor views"""
        return torch.chunk(qo_do, 3, dim=0) if self.concat_qo_do else qo_do

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
    ):
        """
        Distributed Flash Attention forward function

        Args:
            local_q(torch.Tensor):
            local_k(torch.Tensor):
            local_v(torch.Tensor):
            dist_attn_runtime(DistAttnRuntime):

        Returns:
            out(torch.Tensor):

        Shape:
            - local_q: [num_tokens_q_local, num_heads, head_dim]
            - local_k: [num_tokens_k_local, num_heads, head_dim]
            - local_v: [num_tokens_v_local, num_heads, head_dim]
        """

        if not dist_attn_runtime.fwd_use_acc:
            partial_out_list = []
            partial_lse_list = []

        # cat local k, v into a single coalesced kv
        local_kv = dist_attn_runtime.concat_kv(local_k, local_v)

        if magi_attention.is_cuda_device_max_connections_one():
            # pre-fetch 0th remote kv
            (
                remote_kv_work,
                remote_kv_buffer,
            ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)
            # pre-fetch 0th remote q
            (
                remote_q_work,
                remote_q_buffer,
            ) = dist_attn_runtime.fetch_remote_q(local_q=local_q, overlap_stage=0)
        else:
            # when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote comms in advance of ffa fwd
            # and ffa fwd can still overlap with these comms
            # with the support of non-zero `sm_margin`, thx to persistent kernel design
            remote_kv_works_with_buffers = [
                dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]
            remote_q_works_with_buffers = [
                dist_attn_runtime.fetch_remote_q(
                    local_q=local_q, overlap_stage=ith_overlap_stage
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]

        # do attn fwd with local data
        # overlapped with 0th remote comm
        partial_local_out, partial_local_lse = dist_attn_runtime.attn_fwd_partial(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )
        if not dist_attn_runtime.fwd_use_acc and partial_local_out is not None:
            partial_out_list.append(partial_local_out)
            partial_lse_list.append(partial_local_lse)

        partial_remote_out, partial_remote_lse = (
            partial_local_out,
            partial_local_lse,
        )  # init acc buffer if used
        partial_out_reduce_works = []
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote data prepared
            if magi_attention.is_cuda_device_max_connections_one():
                curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)
                curr_remote_q = remote_q_work.wait_post_process(remote_q_buffer)
                # pre-fetch (i+1)th remote data
                if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                    (
                        remote_kv_work,
                        remote_kv_buffer,
                    ) = dist_attn_runtime.fetch_remote_kv(
                        local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                    )
                    (
                        remote_q_work,
                        remote_q_buffer,
                    ) = dist_attn_runtime.fetch_remote_q(
                        local_q=local_q, overlap_stage=ith_overlap_stage + 1
                    )
            else:
                (
                    curr_remote_kv_work,
                    curr_remote_kv_buffer,
                ) = remote_kv_works_with_buffers[ith_overlap_stage]
                curr_remote_kv = curr_remote_kv_work.wait_post_process(
                    curr_remote_kv_buffer
                )
                (
                    curr_remote_q_work,
                    curr_remote_q_buffer,
                ) = remote_q_works_with_buffers[ith_overlap_stage]
                curr_remote_q = curr_remote_q_work.wait_post_process(
                    curr_remote_q_buffer
                )

            # do attn fwd with ith remote data
            # overlapped with (i+1)th remote comm
            partial_remote_out, partial_remote_lse = dist_attn_runtime.attn_fwd_partial(
                q=curr_remote_q,
                kv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
                out_acc=partial_remote_out if dist_attn_runtime.fwd_use_acc else None,
                lse_acc=partial_remote_lse if dist_attn_runtime.fwd_use_acc else None,
            )

            # reduce ith partial out with partial lse
            partial_out_reduce_work = dist_attn_runtime.reduce_partial_out(
                partial_remote_out=partial_remote_out,
                partial_remote_lse=partial_remote_lse,
                partial_local_out=partial_local_out,
                partial_local_lse=partial_local_lse,
                ref_remote_out=curr_remote_q,
                overlap_stage=ith_overlap_stage,
            )
            partial_out_reduce_works.append(partial_out_reduce_work)

            if not dist_attn_runtime.fwd_use_acc and partial_remote_out is not None:
                partial_out_list.append(partial_remote_out)
                partial_lse_list.append(partial_remote_lse)

        # wait for all partial out reduced
        for partial_out_reduce_work in partial_out_reduce_works:
            (
                partial_local_out,
                partial_local_lse,
            ) = partial_out_reduce_work.wait_post_process(
                partial_local_out, partial_local_lse
            )

        # do result correction to get final local out and lse
        if dist_attn_runtime.fwd_use_acc:
            # the final local out, lse has already been reduced into acc buffer by ffa fwd
            local_out = partial_remote_out
            local_lse = partial_remote_lse
        elif magi_attention.comm.is_qo_comm_enable():
            # the final local out, lse has already been reduced into local buffer by group reduce
            local_out = partial_local_out
            local_lse = partial_local_lse
        else:  # the final local out, lse need to be reduced manually from all partial out, lse
            local_out, local_lse = correct_attn_fwd_result(
                out_list=partial_out_list,
                lse_list=partial_lse_list,
            )

        if local_out is None:  # attn computation are all skipped
            # NOTE: We cannot use torch.empty_like here, because empty_like may contain nan values,
            #       and once gradients between different tokens need to be reduced, the nan values
            #       from pad tokens would interfere with the gradients of other tokens
            local_out = torch.zeros_like(local_q)
        else:
            # NOTE: since we've increased the precision of partial out for correction
            # here we need to downcast to q dtype to both return and save for backward
            local_out = local_out.to(local_q.dtype)

        ctx.save_for_backward(local_q, local_kv, local_out, local_lse)
        ctx.dist_attn_runtime = dist_attn_runtime

        return local_out, local_lse

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):  # pragma: no cover
        local_q, local_kv, local_out, local_lse = ctx.saved_tensors
        dist_attn_runtime: DistAttnRuntime = ctx.dist_attn_runtime
        local_qo_do = dist_attn_runtime.maybe_concat_qo_do(
            q=local_q, o=local_out, do=grad_output
        )

        if magi_attention.is_cuda_device_max_connections_one():
            # pre-fetch 0th remote kv
            (
                remote_kv_work,
                remote_kv_buffer,
            ) = dist_attn_runtime.fetch_remote_kv(local_kv=local_kv, overlap_stage=0)
            # pre-fetch 0th remote q,o,do
            (
                remote_qo_do_work,
                remote_qo_do_buffer,
            ) = dist_attn_runtime.fetch_remote_qo_do(
                local_qo_do=local_qo_do,
                overlap_stage=0,
            )
            # pre-fetch 0th remote lse
            (
                remote_lse_work,
                remote_lse_buffer,
            ) = dist_attn_runtime.fetch_remote_lse(
                local_lse=local_lse,
                overlap_stage=0,
            )
        else:
            # NOTE: when `CUDA_DEVICE_MAX_CONNECTIONS` > 1,
            # we issue all fetch-remote comms in advance of ffa bwd
            # and ffa bwd can still overlap with these comms
            # with the support of `sm_margin`, thx to persistent kernel design

            # pre-fetch all remote kv
            remote_kv_works_with_buffers = [
                dist_attn_runtime.fetch_remote_kv(
                    local_kv=local_kv, overlap_stage=ith_overlap_stage
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]
            # pre-fetch all remote q,o,do
            remote_qo_do_works_with_buffers = [
                dist_attn_runtime.fetch_remote_qo_do(
                    local_qo_do=local_qo_do,
                    overlap_stage=ith_overlap_stage,
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]
            # pre-fetch all remote lse
            remote_lse_works_with_buffers = [
                dist_attn_runtime.fetch_remote_lse(
                    local_lse=local_lse,
                    overlap_stage=ith_overlap_stage,
                )
                for ith_overlap_stage in range(dist_attn_runtime.overlap_degree)
            ]

        # do attn bwd with local kv
        # overlapped with 0th remote kv comm
        (
            partial_local_dq,
            partial_local_dkv,
        ) = dist_attn_runtime.attn_bwd_partial(
            qo_do=local_qo_do,
            kv=local_kv,
            lse=local_lse,
            overlap_stage=None,
            deterministic=dist_attn_runtime.deterministic,
        )

        # NOTE: if local_dq and local_dkv calculation are skipped,
        # we need to zeros initialize them since they might be reduced later
        if partial_local_dq is None or partial_local_dkv is None:
            partial_local_dq = torch.zeros_like(
                local_q,
                dtype=max_fp_dtype(local_q.dtype, torch.float32),
            )
            partial_local_dkv = torch.zeros_like(
                local_kv,
                dtype=torch.float32
                if dist_attn_runtime.bwd_hp_reduce
                else local_kv.dtype,
            )
        elif not dist_attn_runtime.bwd_hp_reduce:
            partial_local_dkv = partial_local_dkv.to(local_kv.dtype)

        partial_dq_reduce_works = []
        partial_dkv_reduce_works = []
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # wait for ith remote data prepared
            if magi_attention.is_cuda_device_max_connections_one():
                curr_remote_kv = remote_kv_work.wait_post_process(remote_kv_buffer)
                curr_remote_qo_do = remote_qo_do_work.wait_post_process(
                    remote_qo_do_buffer
                )
                curr_remote_lse = remote_lse_work.wait_post_process(remote_lse_buffer)

                # pre-fetch (i+1)th remote data
                if ith_overlap_stage < dist_attn_runtime.overlap_degree - 1:
                    (
                        remote_kv_work,
                        remote_kv_buffer,
                    ) = dist_attn_runtime.fetch_remote_kv(
                        local_kv=local_kv, overlap_stage=ith_overlap_stage + 1
                    )
                    (
                        remote_qo_do_work,
                        remote_qo_do_buffer,
                    ) = dist_attn_runtime.fetch_remote_qo_do(
                        local_qo_do=local_qo_do,
                        overlap_stage=ith_overlap_stage + 1,
                    )
                    (
                        remote_lse_work,
                        remote_lse_buffer,
                    ) = dist_attn_runtime.fetch_remote_lse(
                        local_lse=local_lse,
                        overlap_stage=ith_overlap_stage + 1,
                    )
            else:
                (
                    curr_remote_kv_work,
                    curr_remote_kv_buffer,
                ) = remote_kv_works_with_buffers[ith_overlap_stage]
                curr_remote_kv = curr_remote_kv_work.wait_post_process(
                    curr_remote_kv_buffer
                )
                (
                    curr_remote_qo_do_work,
                    curr_remote_qo_do_buffer,
                ) = remote_qo_do_works_with_buffers[ith_overlap_stage]
                curr_remote_qo_do = curr_remote_qo_do_work.wait_post_process(
                    curr_remote_qo_do_buffer
                )
                (
                    curr_remote_lse_work,
                    curr_remote_lse_buffer,
                ) = remote_lse_works_with_buffers[ith_overlap_stage]
                curr_remote_lse = curr_remote_lse_work.wait_post_process(
                    curr_remote_lse_buffer
                )

            # do attn bwd with ith remote data
            # overlapped with (i+1)th remote comm
            (
                partial_remote_dq,
                partial_remote_dkv,
            ) = dist_attn_runtime.attn_bwd_partial(
                qo_do=curr_remote_qo_do,
                kv=curr_remote_kv,
                lse=curr_remote_lse,
                dq_acc=partial_local_dq,
                overlap_stage=ith_overlap_stage,
                deterministic=dist_attn_runtime.deterministic,
            )

            # reduce ith partial dkv
            partial_dkv_reduce_work = dist_attn_runtime.reduce_partial_dkv(
                partial_remote_dkv=partial_remote_dkv,
                partial_local_dkv=partial_local_dkv,
                ref_remote_dkv=curr_remote_kv,
                overlap_stage=ith_overlap_stage,
            )
            partial_dkv_reduce_works.append(partial_dkv_reduce_work)

            # reduce ith partial dq
            curr_remote_q, _, _ = dist_attn_runtime.maybe_chunk_qo_do(curr_remote_qo_do)
            partial_dq_reduce_work = dist_attn_runtime.reduce_partial_dq(
                partial_remote_dq=partial_remote_dq,
                partial_local_dq=partial_local_dq,
                ref_remote_dq=curr_remote_q,
                overlap_stage=ith_overlap_stage,
            )
            partial_dq_reduce_works.append(partial_dq_reduce_work)

        # wait for all partial dq reduced
        for partial_dq_reduce_work in partial_dq_reduce_works:
            partial_local_dq = partial_dq_reduce_work.wait_post_process(
                partial_local_dq
            )

        # downcast final local dq to q dtype
        local_dq = partial_local_dq.to(local_q.dtype)

        # wait for all partial dkv reduced
        for partial_dkv_reduce_work in partial_dkv_reduce_works:
            partial_local_dkv = partial_dkv_reduce_work.wait_post_process(
                partial_local_dkv
            )

        # downcast final local dkv to kv dtype
        local_dkv = partial_local_dkv.to(local_kv.dtype)

        # chunk final local dkv into dk and dv
        local_dk, local_dv = dist_attn_runtime.chunk_kv(local_dkv)

        return local_dq, local_dk, local_dv, None, None


def dist_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dist_attn_runtime: DistAttnRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    return DistAttnFunc.apply(q, k, v, dist_attn_runtime)
