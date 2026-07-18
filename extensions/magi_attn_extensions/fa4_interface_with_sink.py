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

# mypy: disable-error-code="arg-type"

import torch
from einops import rearrange
from packaging import version

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


# Copied from https://github.com/Dao-AILab/flash-attention/blob/v2.8.2/flash_attn/flash_attn_interface.py#L56-73
is_torch_compile_supported = version.parse(torch.__version__) >= version.parse("2.4.0")
if is_torch_compile_supported:
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(
        name, fn=None, /, *, mutates_args, device_types=None, schema=None
    ):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------


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

        out, softmax_lse = _fa4_fwd_op(
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

        dq, dk, dv, dsink = _fa4_bwd_op(
            q,
            k,
            v,
            out,
            softmax_lse,
            dout,
            sink,
            ctx.sink_layout,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            ctx.pack_gqa,
            ctx.deterministic,
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
        # max_seqlen_q/max_seqlen_k are accepted for API compatibility but unused
        # by the underlying fa4 varlen kernel (it derives lengths from cu_seqlens).
        out, softmax_lse = _fa4_varlen_fwd_op(
            q,
            k,
            v,
            sink,
            sink_layout,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
            causal,
            softcap,
            num_splits,
            pack_gqa,
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

        # q/k/v: (total, nheads, headdim) — 3D varlen format
        dq, dk, dv, dsink = _fa4_varlen_bwd_op(
            q,
            k,
            v,
            out,
            softmax_lse,
            dout,
            sink,
            ctx.sink_layout,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            ctx.pack_gqa,
            ctx.deterministic,
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


# ---------------------------------------------------------------------------
# Dense custom ops
# ---------------------------------------------------------------------------


@_torch_custom_op_wrapper("magi_attn_ext::fa4_fwd", mutates_args=())
def _fa4_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    num_splits: int,
    pack_gqa: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.ops.magi_attn_ext.fa4_fwd"""
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
    return out, softmax_lse


@_torch_register_fake_wrapper("magi_attn_ext::fa4_fwd")
def _fa4_fwd_fake(
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
):
    # dense: q is (b, s, h, d); lse is (b, h, s)
    b, s, h, _ = q.shape
    out = torch.empty_like(q)
    softmax_lse = q.new_empty((b, h, s), dtype=torch.float32)
    return out, softmax_lse


@_torch_custom_op_wrapper("magi_attn_ext::fa4_bwd", mutates_args=())
def _fa4_bwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dout: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    pack_gqa: bool | None,
    deterministic: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # dsink: actually torch.Tensor | None,
    # NOTE: but torch custom_op schema inference rejects
    # Optional in the return tuple, so we hint Tensor and return
    # None at runtime when sink is None.
]:
    """torch.ops.magi_attn_ext.fa4_bwd"""
    dsink = FA4FuncWithSink.compute_dsink(
        out=out,
        dout=dout,
        lse=softmax_lse,
        sink=sink,
        sink_layout=sink_layout,
    )
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
    )
    return dq, dk, dv, dsink


@_torch_register_fake_wrapper("magi_attn_ext::fa4_bwd")
def _fa4_bwd_fake(
    q,
    k,
    v,
    out,
    softmax_lse,
    dout,
    sink,
    sink_layout,
    softmax_scale,
    causal,
    softcap,
    pack_gqa,
    deterministic,
):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dsink = torch.empty_like(sink) if sink is not None else None
    return dq, dk, dv, dsink


# ---------------------------------------------------------------------------
# Varlen custom ops
# ---------------------------------------------------------------------------


@_torch_custom_op_wrapper("magi_attn_ext::fa4_varlen_fwd", mutates_args=())
def _fa4_varlen_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    num_splits: int,
    pack_gqa: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """torch.ops.magi_attn_ext.fa4_varlen_fwd"""
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
    # out: (total_q, nheads, headdim); softmax_lse: (nheads, total_q)
    out, softmax_lse = FA4VarlenFuncWithSink.correct_out_lse_with_sink(
        out=out,
        lse=softmax_lse,
        sink=sink,
        sink_layout=sink_layout,
    )
    return out, softmax_lse


@_torch_register_fake_wrapper("magi_attn_ext::fa4_varlen_fwd")
def _fa4_varlen_fwd_fake(
    q,
    k,
    v,
    sink,
    sink_layout,
    cu_seqlens_q,
    cu_seqlens_k,
    softmax_scale,
    causal,
    softcap,
    num_splits,
    pack_gqa,
):
    # varlen: q is (total_q, nheads, headdim); lse is (nheads, total_q)
    total_q, nheads, _ = q.shape
    out = torch.empty_like(q)
    softmax_lse = q.new_empty((nheads, total_q), dtype=torch.float32)
    return out, softmax_lse


@_torch_custom_op_wrapper("magi_attn_ext::fa4_varlen_bwd", mutates_args=())
def _fa4_varlen_bwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dout: torch.Tensor,
    sink: torch.Tensor | None,
    sink_layout: str,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    softcap: float,
    pack_gqa: bool | None,
    deterministic: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,  # dsink: actually torch.Tensor | None, but torch custom_op schema
    # inference rejects Optional in the return tuple, so we hint Tensor and return
    # None at runtime when sink is None (same trick as magi_moe flash_grouped_gemm_bwd_cutedsl).
]:
    """torch.ops.magi_attn_ext.fa4_varlen_bwd"""
    dsink = FA4VarlenFuncWithSink.compute_dsink(
        out=out,
        dout=dout,
        lse=softmax_lse,
        sink=sink,
        sink_layout=sink_layout,
    )
    # q/k/v: (total, nheads, headdim) — 3D varlen format
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pack_gqa=pack_gqa,
        deterministic=deterministic,
    )
    return dq, dk, dv, dsink


@_torch_register_fake_wrapper("magi_attn_ext::fa4_varlen_bwd")
def _fa4_varlen_bwd_fake(
    q,
    k,
    v,
    out,
    softmax_lse,
    dout,
    sink,
    sink_layout,
    cu_seqlens_q,
    cu_seqlens_k,
    softmax_scale,
    causal,
    softcap,
    pack_gqa,
    deterministic,
):
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dsink = torch.empty_like(sink) if sink is not None else None
    return dq, dk, dv, dsink


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


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
