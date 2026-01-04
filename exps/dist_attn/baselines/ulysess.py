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

import copy
from typing import Dict, List

import torch

# te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)

from exps.dist_attn.baselines.interface import AttnBaselineInterface
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    ShardMeta,
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
    _varlen_all2all_after_attn,
    _varlen_all2all_before_attn,
    generate_runtime_meta_per_step,
    unflatten_data_from_varlen,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.utils import nvtx


class FA3UlysessAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        runtime_meta,
        causal,
        dropout_p,
        softmax_scale,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert (
            q.shape[-1] % 8 == 0
        ), "Hidden size per attention head should be multiple of 8!"

        fa_forward_kwargs = {"window_size": (-1, -1)}
        rumtime_meta_per_step = runtime_meta[0]
        out, softmax_lse = _fa3_varlen_forward(
            q, k, v, softmax_scale, causal, rumtime_meta_per_step, fa_forward_kwargs
        )

        out_ret = out
        q_save, k_save, v_save, out_save = q, k, v, out
        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            k_save,
            v_save,
            out_save,
            softmax_lse,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.rumtime_meta_per_step = rumtime_meta_per_step

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        rumtime_meta_per_step = ctx.rumtime_meta_per_step
        dout = dout.view(*out.shape)

        window_size = (-1, 0) if ctx.causal else (-1, -1)
        dq, dk, dv = _fa3_varlen_backward(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            ctx.softmax_scale,
            ctx.causal,
            window_size,
            ctx.deterministic,
            rumtime_meta_per_step,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
        )


class TEUlysessAttnFunc(torch.autograd.Function):
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
        attn_mask_type,
        deterministic,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        qkv_dtype = q.dtype
        causal = "causal" in attn_mask_type
        assert (
            q.shape[-1] % 8 == 0
        ), "Hidden size per attention head should be multiple of 8!"

        # contiguous q_k_v
        qkv_layout = qkv_format + "_" + qkv_format + "_" + qkv_format
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        q_part, k_part, v_part = q, k, v
        fused_attn_meta_args = (qkv_dtype, fused_attn_backend)
        fused_attn_meta_kwargs = {
            "attn_scale": softmax_scale,
            "dropout": dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": attn_mask_type,
            "attn_bias_type": "no_bias",
            "attn_bias": None,
        }
        fp8_meta_kwargs = {}
        window_size = (-1, 0) if causal else (-1, -1)

        with nvtx.add_nvtx_event("fused_attn_fwd"):
            out, aux_ctx_tensors = fused_attn_fwd(
                True,  # is_training
                max_seqlen_q,
                max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_part,
                k_part,
                v_part,
                *fused_attn_meta_args,
                **fused_attn_meta_kwargs,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                window_size=window_size,
                **fp8_meta_kwargs,
            )
        softmax_lse, rng_states, *rest = aux_ctx_tensors

        out_ret = out
        q_save, k_save, v_save, out_save = q, k, v, out
        tensors_to_save, tensor_objects = prepare_for_saving(
            q_save,
            k_save,
            v_save,
            out_save,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            rng_states,
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.qkv_dtype = qkv_dtype
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.attn_mask_type = attn_mask_type
        ctx.deterministic = deterministic
        ctx.window_size = window_size

        return out_ret, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_kv,
            cu_seqlens_q_padded,
            cu_seqlens_kv_padded,
            rng_states,
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        qkv_layout = ctx.qkv_format + "_" + ctx.qkv_format + "_" + ctx.qkv_format
        dout_dtype = dout.dtype
        fused_attn_dqkv_dtype = TE_DType[dout_dtype]
        fused_attn_backend = FusedAttnBackend["F16_arbitrary_seqlen"]

        dout = dout.view(*out.shape)
        fused_attn_meta_args = (
            ctx.qkv_dtype,
            fused_attn_dqkv_dtype,
            [softmax_lse, rng_states],
            fused_attn_backend,
        )
        fused_attn_meta_kwargs = {
            "attn_scale": ctx.softmax_scale,
            "dropout": ctx.dropout_p,
            "qkv_layout": qkv_layout,
            "attn_mask_type": ctx.attn_mask_type,
            "attn_bias_type": "no_bias",
            "window_size": ctx.window_size,
            "deterministic": ctx.deterministic,
        }
        fp8_meta_kwargs = {}
        with nvtx.add_nvtx_event("fused_attn_bwd"):
            dq, dk, dv, _, _ = fused_attn_bwd(
                ctx.max_seqlen_q,
                ctx.max_seqlen_kv,
                cu_seqlens_q,
                cu_seqlens_kv,
                q,
                k,
                v,
                out,
                dout,
                *fused_attn_meta_args,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                **fused_attn_meta_kwargs,
                **fp8_meta_kwargs,
            )

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
        )


class Ulysess(AttnBaselineInterface):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,  # "thd" or "bshd" or "sbhd"
        backend: AttnBackend,
    ):
        self.pg_a2a = cp_process_group[ParallelMode.ULYSESS]
        # pad factor for ulysess & ring
        self.pad_factor_p2p, self.pad_factor_a2a = get_pad_factor(
            cp_group_p2p=None, cp_group_a2a=self.pg_a2a
        )
        self.backend = backend
        self.qkv_format = qkv_format
        self.shard_meta = {}  # type: ignore
        self.runtime_meta_per_step = []  # type: ignore

    # to call after q,k,v dispatch
    def pre_compute_attn_runtime_meta(self, device):
        self.runtime_meta_per_step.clear()
        if self.backend == AttnBackend.FA3:
            shard_q_meta = self.shard_meta["q"]
            shard_kv_meta = self.shard_meta["k"]
            rumtime_meta = generate_runtime_meta_per_step(
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.cu_seqlens_padded,
                shard_kv_meta.cu_seqlens_padded,
                shard_q_meta.host_cu_seqlens,
                shard_kv_meta.host_cu_seqlens,
                shard_q_meta.host_cu_seqlens_padded[-1],
                shard_kv_meta.host_cu_seqlens_padded[-1],
                device,
            )
            self.runtime_meta_per_step.append(rumtime_meta)
        pass

    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str | List[str],  # key names for shard_meta
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

        x_local, _ = zigzag_dispatch(
            x_global_varlen,
            host_cu_seqlens,
            host_cu_seqlens_padded,
            "thd",
            cp_group_p2p=None,
            cp_group_a2a=self.pg_a2a,
        )

        max_seqlen_padded = get_max_seqlen(host_cu_seqlens_padded)
        dispatch_keys = name
        if isinstance(name, str):
            dispatch_keys = [name]
        shard_meta = ShardMeta(
            cu_seqlens=cu_seqlens,
            cu_seqlens_padded=cu_seqlens_padded,
            host_cu_seqlens=host_cu_seqlens,
            host_cu_seqlens_padded=host_cu_seqlens_padded,
            origin_shape=origin_shape,
            max_seqlen_padded=max_seqlen_padded,
        )
        for key in dispatch_keys:
            self.shard_meta[key] = copy.deepcopy(shard_meta)
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
            cp_group_p2p=None,
            cp_group_a2a=self.pg_a2a,
        )
        x_global = unflatten_data_from_varlen(
            x_global_varlen, smeta.cu_seqlens, smeta.origin_shape, self.qkv_format
        )

        return x_global

    @nvtx.instrument_nvtx
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
        # all2all comm
        q_layer = _varlen_all2all_before_attn(q, self.pg_a2a)
        k_layer = _varlen_all2all_before_attn(k, self.pg_a2a)
        v_layer = _varlen_all2all_before_attn(v, self.pg_a2a)

        shard_q_meta = self.shard_meta["q"]
        shard_kv_meta = self.shard_meta["k"]
        if self.backend == AttnBackend.TE:
            if attn_mask_type == AttnMaskType.CAUSAL:
                attn_mask = "padding_causal"
            elif attn_mask_type == AttnMaskType.FULL:
                attn_mask = "padding"

            out, lse = TEUlysessAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen_padded,
                shard_kv_meta.max_seqlen_padded,
                shard_q_meta.cu_seqlens_padded,
                shard_kv_meta.cu_seqlens_padded,
                dropout_p,
                softmax_scale,
                "thd",
                attn_mask,
                deterministic,
            )
        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False

            out, lse = FA3UlysessAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                self.runtime_meta_per_step,
                is_causal,
                dropout_p,
                softmax_scale,
                deterministic,
            )

        # all2all comm
        out_layer = _varlen_all2all_after_attn(out, self.pg_a2a)

        return out_layer, lse
