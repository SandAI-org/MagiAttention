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

# te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)

from magi_attention.common.enum import AttnMaskType

from .attn import prepare_for_saving  # type: ignore[attr-defined]
from .attn import restore_from_saved  # type: ignore[attr-defined]
from .attn import AttnBackend, _fa3_attn_backward, _fa3_attn_forward
from .interface import AttnBaselineInterface
from .shard import (
    ParallelMode,
    ShardMeta,
    get_cu_seqlens_padded,
    get_pad_factor,
    zigzag_dispatch,
    zigzag_undispatch,
)
from .utils_cp import _SeqAllToAll


class FA3UlysessAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,  # python int
        max_seqlen_kv,  # python int
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        causal,
        dropout_p,
        softmax_scale,
        qkv_format,
        deterministic,
        pad_between_seqs,
        host_meta=[None, None, None, None],
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        qkv_dtype = q.dtype
        assert (
            q.shape[-1] % 8 == 0
        ), "Hidden size per attention head should be multiple of 8!"

        fa_forward_kwargs = {"softmax_scale": softmax_scale}
        out, softmax_lse = _fa3_attn_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            # max_seqlen_q,
            # max_seqlen_kv,
            # cu_seqlens_q_padded,
            # cu_seqlens_kv_padded,
            causal,
            qkv_format,
            pad_between_seqs,
            fa_forward_kwargs,
            host_meta,
        )

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
        )
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects

        ctx.qkv_dtype = qkv_dtype
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.softmax_scale = softmax_scale
        ctx.qkv_format = qkv_format
        ctx.deterministic = deterministic
        ctx.pad_between_seqs = pad_between_seqs
        ctx.host_meta = host_meta

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
        ) = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)

        dout = dout.view(*out.shape)

        fa_backward_kwargs = {"softmax_scale": ctx.softmax_scale}
        fa_backward_kwargs["deterministic"] = ctx.deterministic

        dq, dk, dv = _fa3_attn_backward(
            q,
            k,
            v,
            out,
            dout,
            cu_seqlens_q,
            cu_seqlens_kv,
            # ctx.max_seqlen_q,
            # ctx.max_seqlen_kv,
            # cu_seqlens_q_padded,
            # cu_seqlens_kv_padded,
            softmax_lse,
            ctx.causal,
            ctx.qkv_format,
            ctx.pad_between_seqs,
            fa_backward_kwargs,
            ctx.host_meta,
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
        dq, dk, dv, _ = fused_attn_bwd(
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
        qkv_format: str,
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
            cp_group_p2p=None,
            cp_group_a2a=self.pg_a2a,
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
            cp_group_p2p=None,
            cp_group_a2a=self.pg_a2a,
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
        if self.qkv_format != "sbhd":
            batch_dim, seq_dim = 0, 1
        else:
            batch_dim, seq_dim = 1, 0

        if self.qkv_format == "thd":  # thd -> 1,thd
            q, k, v = [x.unsqueeze(0) for x in [q, k, v]]
        q_layer, k_layer, v_layer = [
            _SeqAllToAll.apply(self.pg_a2a, x, 2, seq_dim, batch_dim) for x in [q, k, v]
        ]
        if self.qkv_format == "thd":
            q_layer, k_layer, v_layer = [
                x.squeeze(0) for x in [q_layer, k_layer, v_layer]
            ]

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
                self.qkv_format,
                attn_mask,
                deterministic,
            )
        elif self.backend == AttnBackend.FA3:
            if attn_mask_type == AttnMaskType.CAUSAL:
                is_causal = True
            elif attn_mask_type == AttnMaskType.FULL:
                is_causal = False
            pad_between_seqs = not (
                shard_q_meta.host_cu_seqlens[-1]
                == shard_q_meta.host_cu_seqlens_padded[-1]
            )
            out, lse = FA3UlysessAttnFunc.apply(
                q_layer,
                k_layer,
                v_layer,
                shard_q_meta.cu_seqlens,
                shard_kv_meta.cu_seqlens,
                shard_q_meta.max_seqlen,
                shard_kv_meta.max_seqlen,
                shard_q_meta.cu_seqlens_padded,
                shard_kv_meta.cu_seqlens_padded,
                is_causal,
                dropout_p,
                softmax_scale,
                self.qkv_format,
                deterministic,
                pad_between_seqs,
                [
                    shard_q_meta.host_cu_seqlens,
                    shard_q_meta.host_cu_seqlens_padded,
                    shard_kv_meta.host_cu_seqlens,
                    shard_kv_meta.host_cu_seqlens_padded,
                ],
            )

        if self.qkv_format == "thd":  # thd -> 1,thd
            out = out.unsqueeze(0)
        out_layer = _SeqAllToAll.apply(self.pg_a2a, out, seq_dim, 2, batch_dim)
        if self.qkv_format == "thd":
            out_layer = out_layer.squeeze(0)

        return out_layer, lse
