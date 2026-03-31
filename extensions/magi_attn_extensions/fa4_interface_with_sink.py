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

# Copyright (c) 2023, Tri Dao.

import torch
from einops import rearrange

from magi_attention.api.functools import infer_attn_mask_from_cu_seqlens
from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import (
    correct_attn_out_lse_with_sink_compiled,
    sink_bwd_compiled,
)
from magi_attention.meta.collection.calc_meta import FA4AttnArg

is_fa4_installed = False
try:
    from flash_attn_cute.interface import _flash_attn_bwd, _flash_attn_fwd

    is_fa4_installed = True
except ImportError:
    pass

if is_fa4_installed:
    from magi_attention.functional.fa4_utils import load_precompiled_ffa_fa4

    load_precompiled_ffa_fa4()


class FA4QKVPackedFuncWithSink(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)

        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)

        # FA4 forward call
        out, softmax_lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            learnable_sink=None,
            pack_gqa=False,
            return_lse=True,
        )

        out, softmax_lse = FA4FuncWithSink.correct_out_lse_with_sink(
            out=out,
            lse=softmax_lse,
            sink=sink,
            sink_layout=sink_layout,
        )

        ctx.save_for_backward(q, k, v, sink, out, softmax_lse)
        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, sink, out, softmax_lse = ctx.saved_tensors

        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)

        dsink = FA4FuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        dq_res, dk_res, dv_res = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
        )

        dq.copy_(dq_res)
        dk.copy_(dk_res)
        dv.copy_(dv_res)

        return (
            dqkv,
            dsink,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FA4FuncWithSink(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        softcap=0.0,
        num_splits=1,
        deterministic=False,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, softmax_lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            num_splits=num_splits,
            learnable_sink=None,
            pack_gqa=False,  # pack_gqa is not supported in FA4
            return_lse=True,
        )

        out, softmax_lse = FA4FuncWithSink.correct_out_lse_with_sink(
            out=out,
            lse=softmax_lse,
            sink=sink,
            sink_layout=sink_layout,
        )

        ctx.save_for_backward(q, k, v, sink, out, softmax_lse)
        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap
        ctx.deterministic = deterministic

        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, sink, out, softmax_lse = ctx.saved_tensors

        dsink = FA4FuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
        )

        return (
            dq,
            dk,
            dv,
            dsink,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def correct_out_lse_with_sink(
        out: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sink is not None:
            FA4FuncWithSink._check_sink_layout(sink, sink_layout)
            b = out.shape[0]

            # rearrange out, lse
            out = rearrange(out, "b s h d -> (b s) h d")
            lse = rearrange(lse, "b h s -> (b s) h")
            sink = (
                rearrange(sink, "b sq s h -> (b sq) s h")
                if sink_layout == "ssh"
                else sink
            ).detach()

            # correct out, lse with sink
            out, lse = correct_attn_out_lse_with_sink_compiled(
                out=out,
                lse=lse,
                sink=sink,
                sink_layout=sink_layout,
                inplace=True,
            )

            # rearrange out, lse back
            out = rearrange(out, "(b s) h d -> b s h d", b=b).contiguous()
            lse = rearrange(lse, "(b s) h -> b h s", b=b).contiguous()

        return out, lse

    @staticmethod
    def compute_dsink(
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
    ) -> torch.Tensor | None:
        if sink is not None:
            b = out.shape[0]
            # rearrange out, do, lse
            out = rearrange(out, "b s h d -> (b s) h d")
            dout = rearrange(dout, "b s h d -> (b s) h d")
            lse = rearrange(lse, "b h s -> (b s) h")
            sink = (
                rearrange(sink, "b sq s h -> (b sq) s h")
                if sink_layout == "ssh"
                else sink
            ).detach()

            # compute dsink
            dsink = sink_bwd_compiled(
                sink=sink,
                lse=lse,
                o=out,
                do=dout,
                sink_layout=sink_layout,
            )

            # rearrange dsink back
            dsink = (
                rearrange(dsink, "(b sq) s h -> b sq s h", b=b)
                if sink_layout == "ssh"
                else dsink
            )
        else:
            dsink = None

        return dsink

    @staticmethod
    def _check_sink_layout(
        sink: torch.Tensor,
        sink_layout: AttnSinkLayout,
    ) -> None:
        match sink.ndim:
            case 2:
                assert sink_layout == "sh"
            case 3:
                assert sink_layout == "shd"
            case 4:
                assert sink_layout == "ssh"
            case _:
                raise ValueError(f"Invalid sink shape {sink.shape}")


class FA4VarlenFuncWithSink(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sink,
        sink_layout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        softcap=0.0,
        num_splits=1,
        deterministic=False,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # Create AttnRanges from cu_seqlens
        (
            q_ranges,
            k_ranges,
            attn_mask_type,
            _total_seqlen_q,
            _total_seqlen_k,
        ) = infer_attn_mask_from_cu_seqlens(
            cu_seqlens_q,
            cu_seqlens_k,
            causal=causal,
        )
        attn_type_map = [m.to_int_type() for m in attn_mask_type]

        # Use FA4AttnArg to generate sparse masks
        fa4_arg = FA4AttnArg(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            seqlen_q=q.shape[0],
            seqlen_k=k.shape[0],
            headdim=q.shape[-1],
            tile_m=128,
            tile_n=128,
        )

        fa4_args_dict = fa4_arg.to_fa4_args(is_bwd=False)

        out, softmax_lse = _flash_attn_fwd(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            softmax_scale=softmax_scale,
            causal=False,
            arbitrary=True,
            softcap=softcap,
            num_splits=num_splits,
            learnable_sink=None,
            pack_gqa=False,
            return_lse=True,
            block_sparse_tensors=fa4_args_dict["linear_k_block_sparse_mask"],
            aux_tensors=fa4_args_dict["aux_tensors"],
        )

        # Rearrange out: (1, s, h, d) -> (s, h, d)
        out = out.squeeze(0)
        # Rearrange lse: (1, h, s) -> (h, s)
        softmax_lse = softmax_lse.squeeze(0)

        out, softmax_lse = FA4VarlenFuncWithSink.correct_out_lse_with_sink(
            out=out,
            lse=softmax_lse,
            sink=sink,
            sink_layout=sink_layout,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            sink,
            out,
            softmax_lse,
        )
        ctx.fa4_arg = fa4_arg
        ctx.sink_layout = sink_layout
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap
        ctx.deterministic = deterministic

        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            sink,
            out,
            softmax_lse,
        ) = ctx.saved_tensors

        dsink = FA4VarlenFuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        fa4_arg = ctx.fa4_arg
        fa4_args_dict = fa4_arg.to_fa4_args(is_bwd=True)

        # Rearrange q,k,v,o,do: (s, h, d) -> (1, s, h, d)
        q, k, v, out, dout = [x.unsqueeze(0) for x in (q, k, v, out, dout)]

        # Rearrange lse: (h, s) -> (1, h, s)
        softmax_lse = softmax_lse.unsqueeze(0)

        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            causal=False,
            arbitrary=True,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
            block_sparse_tensors=fa4_args_dict["linear_q_block_sparse_mask"],
            aux_tensors=fa4_args_dict["aux_tensors"],
        )

        dq = dq.squeeze(0)
        dk = dk.squeeze(0)
        dv = dv.squeeze(0)

        return (
            dq,
            dk,
            dv,
            dsink,
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

    @staticmethod
    def correct_out_lse_with_sink(
        out: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sink is not None:
            FA4VarlenFuncWithSink._check_sink_layout(lse, sink, sink_layout)

            # rearrange lse
            lse = rearrange(lse, "h s -> s h")

            # correct out, lse with sink
            out, lse = correct_attn_out_lse_with_sink_compiled(
                out=out,
                lse=lse,
                sink=sink.detach(),
                sink_layout=sink_layout,
                inplace=True,
            )

            # rearrange lse back
            lse = rearrange(lse, "s h -> h s").contiguous()

        return out, lse

    @staticmethod
    def compute_dsink(
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
        sink_layout: AttnSinkLayout,
    ) -> torch.Tensor | None:
        if sink is not None:
            # rearrange lse
            lse = rearrange(lse, "h s -> s h")

            # compute dsink
            dsink = sink_bwd_compiled(
                sink=sink.detach(),
                lse=lse,
                o=out,
                do=dout,
                sink_layout=sink_layout,
            )
        else:
            dsink = None

        return dsink

    @staticmethod
    def _check_sink_layout(
        lse: torch.Tensor,
        sink: torch.Tensor,
        sink_layout: AttnSinkLayout,
    ) -> None:
        match sink.ndim:
            case 2:
                assert sink_layout == "sh"
            case 3:
                if sink.size(0) == lse.size(1):  # lse.shape = [nhq, sq]
                    assert sink_layout == "ssh"
                else:
                    assert sink_layout == "shd"
            case _:
                raise ValueError(f"Invalid sink shape {sink.shape}")


def fa4_qkvpacked_func_with_sink(
    qkv,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    return_attn_probs=False,
):
    return FA4QKVPackedFuncWithSink.apply(
        qkv,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        softcap,
        deterministic,
        num_heads_q,
        return_attn_probs,
    )


def fa4_func_with_sink(
    q,
    k,
    v,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    num_splits=1,
    deterministic=False,
    return_attn_probs=False,
):
    return FA4FuncWithSink.apply(
        q,
        k,
        v,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        softcap,
        num_splits,
        deterministic,
        return_attn_probs,
    )


def fa4_varlen_func_with_sink(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale=None,
    causal=False,
    softcap=0.0,
    num_splits=1,
    deterministic=False,
    return_attn_probs=False,
):
    return FA4VarlenFuncWithSink.apply(
        q,
        k,
        v,
        sink,
        sink_layout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        softcap,
        num_splits,
        deterministic,
        return_attn_probs,
    )
