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

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import (
    correct_attn_out_lse_with_sink_compiled,
    sink_bwd_compiled,
)

is_fa4_installed = False
try:
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    is_fa4_installed = True
except ImportError:
    pass


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
        pack_gqa=None,
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
            pack_gqa=pack_gqa,
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
        ctx.pack_gqa = pack_gqa
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
            pack_gqa=ctx.pack_gqa,
            deterministic=ctx.deterministic,
        )

        return (
            dq,
            dk,
            dv,
            dsink,
            None,  # sink_layout
            None,  # softmax_scale
            None,  # causal
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # return_softmax
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
        pack_gqa=None,
        deterministic=False,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # q/k/v: (total, nheads, headdim) — 3D varlen format
        out, softmax_lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            num_splits=num_splits,
            learnable_sink=None,
            pack_gqa=pack_gqa,
            return_lse=True,
        )
        # out: (total_q, nheads, headdim)
        # softmax_lse: (nheads, total_q)

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
            cu_seqlens_q,
            cu_seqlens_k,
        )
        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.softcap = softcap
        ctx.pack_gqa = pack_gqa
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
            cu_seqlens_q,
            cu_seqlens_k,
        ) = ctx.saved_tensors

        dsink = FA4VarlenFuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        # q/k/v: (total, nheads, headdim) — 3D varlen format
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
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            pack_gqa=ctx.pack_gqa,
            deterministic=ctx.deterministic,
        )

        return (
            dq,
            dk,
            dv,
            dsink,
            None,  # sink_layout
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # softmax_scale
            None,  # causal
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # return_softmax
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
    pack_gqa=None,
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
        pack_gqa,
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
    pack_gqa=None,
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
        pack_gqa,
        deterministic,
        return_attn_probs,
    )
