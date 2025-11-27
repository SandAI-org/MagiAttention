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

# Copyright (c) 2023, Tri Dao.

import torch
from einops import rearrange
from flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import (
    correct_attn_out_lse_with_sink_compiled,
    sink_bwd_compiled,
)


class FA3QKVPackedFuncWithSink(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
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

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            None,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )

        out, softmax_lse = FA3FuncWithSink.correct_out_lse_with_sink(
            out=out,
            lse=softmax_lse,
            sink=sink,
            sink_layout=sink_layout,
        )

        ctx.save_for_backward(q, k, v, sink, out, softmax_lse)
        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, sink, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
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

        dsink = FA3FuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )

        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension

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
            None,
            None,
            None,
            None,
            None,
        )


class FA3FuncWithSink(torch.autograd.Function):
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
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
                -0.5
            )

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

        out, softmax_lse = FA3FuncWithSink.correct_out_lse_with_sink(
            out=out,
            lse=softmax_lse,
            sink=sink,
            sink_layout=sink_layout,
        )

        ctx.save_for_backward(q, k, v, sink, out, softmax_lse)
        ctx.sink_layout = sink_layout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin

        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, sink, out, softmax_lse = ctx.saved_tensors

        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        dsink = FA3FuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        dq, dk, dv, softmax_d = _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]

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
            FA3FuncWithSink._check_sink_layout(sink, sink_layout)
            b = out.shape[0]

            # rearrange out, lse
            out = rearrange(out, "b s h d -> (b s) h d")
            lse = rearrange(lse, "b h s -> (b s) h")
            sink = (
                rearrange(sink, "b sq s h -> (b sq) s h")
                if sink_layout == "ssh"
                else sink
            )

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
            )

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


class FA3VarlenFuncWithSink(torch.autograd.Function):
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
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
                -0.5
            )

        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )

        out, softmax_lse = FA3VarlenFuncWithSink.correct_out_lse_with_sink(
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
            seqused_q,
            seqused_k,
        )
        ctx.sink_layout = sink_layout
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin

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
            seqused_q,
            seqused_k,
        ) = ctx.saved_tensors

        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        dsink = FA3VarlenFuncWithSink.compute_dsink(
            out=out,
            dout=dout,
            lse=softmax_lse,
            sink=sink,
            sink_layout=ctx.sink_layout,
        )

        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]

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
            FA3VarlenFuncWithSink._check_sink_layout(lse, sink, sink_layout)

            # rearrange lse
            lse = rearrange(lse, "h s -> s h")

            # correct out, lse with sink
            out, lse = correct_attn_out_lse_with_sink_compiled(
                out=out,
                lse=lse,
                sink=sink,
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
                sink=sink,
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


def fa3_qkvpacked_func_with_sink(
    qkv,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale=None,
    causal=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        sink:
            if sink_layout == "sh": (seqlen_sink, nheads)
            if sink_layout == "ssh": (batch_size, seqlen, seqlen_sink, nheads).
            Default to None to not apply attention sink.
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of (-alibi_slope * |i - j|) is added to
            the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FA3QKVPackedFuncWithSink.apply(
        qkv,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
        return_attn_probs,
    )


def fa3_func_with_sink(
    q,
    k,
    v,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        sink:
            if sink_layout == "sh": (seqlen_sink, nheads)
            if sink_layout == "ssh": (batch_size, seqlen, seqlen_sink, nheads).
            Default to None to not apply attention sink.
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return FA3FuncWithSink.apply(
        q,
        k,
        v,
        sink,
        sink_layout,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def fa3_varlen_func_with_sink(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    sink=None,
    sink_layout: AttnSinkLayout = "sh",
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        sink:
            if sink_layout == "sh": (seqlen_sink, nheads)
            if sink_layout == "ssh": (total_q, seqlen_sink, nheads).
            Default to None to not apply attention sink.
        sink_layout (AttnSinkLayout, optional): the layout of the sink tokens.
            Defaults to "sh".
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    return FA3VarlenFuncWithSink.apply(
        q,
        k,
        v,
        sink,
        sink_layout,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )
