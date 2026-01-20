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


import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from packaging import version
from torch.nn.attention import SDPBackend, sdpa_kernel

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import (
    correct_attn_out_lse,
    correct_attn_lse_with_sink,
    safe_lse,
    safe_softmax,
    sink_bwd,
)
from magi_attention.utils import max_fp_dtype, to_higher_fp_dtype

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


class RefAttnTorchImplMainProcessOnline(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kt: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None,
        bias: torch.Tensor,
        softmax_scale: float,
        block_q: int = 1024,
        block_k: int = 1024,
    ):
        # fetch meta info
        # q.shape = [nhq, sq, d]
        # kt.shape = [nhq, d, sk]
        # v.shape = [nhq, sk, d]
        sq, sk, nhq = q.size(-2), kt.size(-1), q.size(0)
        lse_dtype = max_fp_dtype(q.dtype, torch.float32)

        # init out buffer
        # out.shape = [nhq, sq, d]
        out = torch.zeros_like(q)

        # init lse buffer with sink if sink is provided
        # sink.shape = [nhq, sq, s_sink]
        # lse.shape = [nhq, sq]
        lse = RefAttnTorchImplMainProcessOnline.init_lse_with_sink(
            sq=sq, nhq=nhq, sink=sink, lse_dtype=lse_dtype, device=q.device
        )

        # outer loop for q,o
        for q_start in range(0, sq, block_q):
            q_end = min(q_start + block_q, sq)
            # bq/bout.shape: [nhq, block_q, hd]
            # blse.shape: [nhq, block_q]
            bq, bout, blse = (
                q[:, q_start:q_end],
                out[:, q_start:q_end],
                lse[:, q_start:q_end],
            )

            # inner loop for k,v
            for k_start in range(0, sk, block_k):
                k_end = min(k_start + block_k, sk)
                # bkt.shape: [nhq, hd, block_k]
                # bv.shape: [nhq, block_k, hd]
                # bbias.shape: [nhq, block_q, block_k]
                bkt, bv, bbias = (
                    kt[..., k_start:k_end],
                    v[:, k_start:k_end],
                    bias[:, q_start:q_end, k_start:k_end],
                )

                # apply `S = Q x K.T * scale + bias`
                # where bs.shape = [nhq, block_q, block_k]
                bs = to_higher_fp_dtype(
                    bq @ bkt * softmax_scale,
                    lowest_precision=torch.float32,
                )
                bs += bbias

                # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
                # where blse.shape = [nhq, block_q, 1]
                blse_ = bs.logsumexp(dim=-1, keepdim=True)

                # apply row-wise softmax `P = softmax(S, dim=-1)`
                # where bp.shape = [nhq, block_q, block_k]
                # NOTE: pytorch softmax has many limitations and bugs
                # thus we use our own safe_softmax with lse involved
                bp = safe_softmax(bs, blse_).to(q.dtype)

                # apply `O = P x V`
                # where O.shape = [nhq, block_q, hd]
                bout_ = bp @ bv

                # correct blse
                blse_ = blse_.squeeze_(-1)
                correct_attn_out_lse(
                    out1=bout,
                    lse1=blse,
                    out2=bout_,
                    lse2=blse_,
                    inplace=True,
                )

        ctx.save_for_backward(q, kt, v, bias, sink, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.block_q = block_q
        ctx.block_k = block_k
        ctx.sq, ctx.sk = sq, sk

        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):  # pragma: no cover
        # fetch saved tensors
        q, kt, v, bias, sink, out, lse = ctx.saved_tensors

        # fetch saved meta info
        softmax_scale = ctx.softmax_scale
        block_q = ctx.block_q
        block_k = ctx.block_k
        sq, sk = ctx.sq, ctx.sk

        # init dq, dkt, dv buffer
        # dq.shape = [nhq, sq, d]
        # dkt.shape = [nhq, d, sk]
        # dv.shape = [nhq, sk, d]
        dq = torch.zeros_like(q)
        dkt = torch.zeros_like(kt)
        dv = torch.zeros_like(v)

        # compute dpsum and dsink if sink is provided
        # dpsum.shape = [nhq, sq, 1]
        # dsink.shape = [nhq, sq, s_sink]
        dpsum, dsink = RefAttnTorchImplMainProcessOnline.compute_dpsum_dsink(
            out=out,
            dout=dout,
            lse=lse,
            sink=sink,
        )

        # outer loop for k,dk,v,dv
        for k_start in range(0, sk, block_k):
            k_end = min(k_start + block_k, sk)
            # bkt.shape: [nhq, hd, block_k]
            # bdkt.shape: [nhq, hd, block_k]
            # bv.shape: [nhq, block_k, hd]
            # bdv.shape: [nhq, block_k, hd]
            bkt, bdkt, bv, bdv = (
                kt[..., k_start:k_end],
                dkt[..., k_start:k_end],
                v[:, k_start:k_end],
                dv[:, k_start:k_end],
            )

            # inner loop for q,dq,do,lse
            for q_start in range(0, sq, block_q):
                q_end = min(q_start + block_q, sq)
                # bq/bdq/bdout.shape: [nhq, block_q, hd]
                # blse.shape: [nhq, block_q]
                # bbias.shape: [nhq, block_q, block_k]
                # bdpsum.shape: [nhq, block_q, 1]
                bq, bdq, bdout, blse, bbias, bdpsum = (
                    q[:, q_start:q_end],
                    dq[:, q_start:q_end],
                    dout[:, q_start:q_end],
                    lse[:, q_start:q_end],
                    bias[:, q_start:q_end, k_start:k_end],
                    dpsum[:, q_start:q_end],
                )

                # recompute bp
                # bp.shape: [nhq, block_q, block_k]
                bp = RefAttnTorchImplMainProcessOnline.bwd_recompute_bp(
                    bq=bq,
                    bkt=bkt,
                    blse=blse,
                    bbias=bbias,
                    softmax_scale=softmax_scale,
                )

                # apply `dV = P.T x dO` and reduce
                # bdv.shape: [nhq, block_k, hd]
                bdv.add_(bp.transpose(-1, -2) @ bdout)

                # apply `dP = dO x V.T`
                # bdp.shape: [nhq, block_q, block_k]
                bdp = bdout @ bv.transpose(-1, -2)

                # apply `dS_ = P * (dP - dPsum)`
                # bds.shape: [nhq, block_q, block_k]
                bds = bp * (bdp - bdpsum)

                # apply `dS = dS_ * scale`
                # bds.shape: [nhq, block_q, block_k]
                bds = (bds * softmax_scale).to(bdq.dtype)

                # apply `dQ = dS x K`
                bdq.add_(bds @ bkt.transpose(-1, -2))

                # apply `dK.T = Q.T x dS`
                bdkt.add_(bq.transpose(-1, -2) @ bds)

        return (
            dq,
            dkt,
            dv,
            dsink,
            None,  # bias
            None,  # softmax_scale
            None,  # block_q
            None,  # block_k
        )

    @staticmethod
    def init_lse_with_sink(
        sq: int,
        nhq: int,
        sink: torch.Tensor | None,
        lse_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        lse = torch.full((nhq, sq), -float("inf"), dtype=lse_dtype, device=device)
        if sink is not None:
            # directly initialize lse with sink
            # if sink is provided
            lse = rearrange(
                correct_attn_lse_with_sink(
                    lse=rearrange(lse, "nhq sq -> sq nhq"),
                    sink=rearrange(sink, "nhq sq s_sink -> sq s_sink nhq"),
                    sink_layout="ssh",
                    inplace=False,
                ),
                "sq nhq -> nhq sq",
            ).contiguous()

        return lse

    @staticmethod
    def compute_dpsum_dsink(
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        sink: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # compute dpsum (i.e. delta)
        # dpsum.shape = [nhq, sq, 1]
        dpsum = reduce((out * dout).to(lse.dtype), "nhq sq d -> nhq sq 1", "sum")

        # compute dsink if sink is provided
        # dsink.shape = [nhq, sq, s_sink]
        if sink is not None:
            dsink = rearrange(
                sink_bwd(
                    sink=rearrange(sink, "nhq sq s_sink -> sq s_sink nhq"),
                    lse=rearrange(lse, "nhq sq -> sq nhq"),
                    o=rearrange(out, "nhq sq d -> sq nhq d"),
                    do=rearrange(dout, "nhq sq d -> sq nhq d"),
                    sink_layout="ssh",
                ),
                "sq s_sink nhq -> nhq sq s_sink",
            ).contiguous()
        else:
            dsink = None

        return dpsum, dsink

    @staticmethod
    def bwd_recompute_bp(
        bq: torch.Tensor,
        bkt: torch.Tensor,
        blse: torch.Tensor,
        bbias: torch.Tensor,
        softmax_scale: float,
    ):
        # apply `S = Q x K.T * scale + bias`
        # where S.shape = [nhq, block_q, block_k]
        bs = to_higher_fp_dtype(
            bq @ bkt * softmax_scale,
            lowest_precision=torch.float32,
        )
        bs += bbias

        # apply row-wise softmax `P = softmax(S, dim=-1)`
        # where P.shape = [nhq, block_q, block_k]
        # NOTE: pytorch softmax has many limitations and bugs
        # thus we use our own safe_softmax with lse involved
        bp = safe_softmax(bs, blse.unsqueeze(-1)).to(bq.dtype)

        return bp


@torch.no_grad
def _calc_attn_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
):
    (q, kt, _, _, bias, softmax_scale) = _ref_attn_torch_impl_preprocess(
        q=q,
        k=k,
        v=None,
        sink=None,
        mask=mask,
        softmax_scale=softmax_scale,
    )

    # calculate lse
    lse = (
        # apply `S = Q x K.T * scale + bias`
        # where S.shape = [nhq, sq, sk]
        # when mask.shape = [h, sq, sk], s.shape is [1, h, sq, sk] broadcast by bias.
        to_higher_fp_dtype(
            q @ kt * softmax_scale + bias,
            lowest_precision=torch.float32,
        )
        # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
        # where LSE.shape = [nhq, sq]
        .logsumexp(dim=-1)
        # transpose and make contiguous
        # where LSE.shape = [sq, nhq]
        # when mask.shape = [h, sq, sk], lse.shape is [1, h, sq]
        .transpose(-1, -2).contiguous()
    )

    return lse


def _ref_attn_sdpa_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if return_lse:
        lse = _calc_attn_lse(
            q,
            k,
            mask,
            softmax_scale,
        )
    else:
        lse = None

    q = rearrange(q, "t h d -> 1 h t d")
    k = rearrange(k, "t h d -> 1 h t d")
    v = rearrange(v, "t h d -> 1 h t d")

    with sdpa_kernel(backends=[SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            enable_gqa=True,
            scale=softmax_scale,
        )

    out = rearrange(out, "1 h t d -> t h d")

    return out, lse


def _ref_attn_torch_impl_preprocess(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor | None,
    sink: torch.Tensor | None,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
    sink_layout: AttnSinkLayout = "sh",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    float,
]:
    # prepare softmax scale
    softmax_scale = q.size(-1) ** (-0.5) if softmax_scale is None else softmax_scale
    assert softmax_scale is not None  # mypy
    # prepare bias
    # where bias.shape = [1, sq, sk]
    # when mask.shape = [h, sq, sk], bias.shape is [1, h, sq, sk]
    bias = torch.zeros_like(
        mask, dtype=max_fp_dtype(q.dtype, torch.float32), device=q.device
    )
    bias.masked_fill_(mask.logical_not(), float("-inf")).unsqueeze_(0)

    # prepare sink
    # where sink.shape = [nhq, sq, s_sink]
    if sink is not None:
        match sink_layout:
            case "sh":
                sink = repeat(sink, "s hq -> hq sq s", sq=q.size(0))
            case "ssh":
                sink = rearrange(sink, "sq s hq -> hq sq s")
            case "shd":
                raise NotImplementedError(
                    f"sink_layout {sink_layout} is not supported yet"
                )
            case _:
                raise ValueError(f"Invalid sink_layout {sink_layout}")
        sink = to_higher_fp_dtype(
            sink, lowest_precision=max_fp_dtype(q.dtype, torch.float32)
        )

    # prepare q,k,v
    # where:
    #   q.shape = [nhq, sq, d]
    #   k.shape = [nhq, d, sk]
    #   v.shape = [nhq, sk, d]
    nhq, nhk = q.size(-2), k.size(-2)
    assert nhq % nhk == 0
    rep_nhk = nhq // nhk
    q = rearrange(q, "s hq d -> hq s d")  # Q
    k = repeat(k, "s hk d -> (hk rep) d s", rep=rep_nhk)  # K.T
    if v is not None:
        v = repeat(v, "s hk d -> (hk rep) s d", rep=rep_nhk)  # V

    return q, k, v, sink, bias, softmax_scale


def _ref_attn_torch_impl_postprocess(
    out: torch.Tensor,
    lse: torch.Tensor | None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # rearrange and make contiguous
    # where O.shape = [sq, nhq, d]
    out = rearrange(out, "nhq sq d -> sq nhq d")

    # prepare lse if required to return
    # where LSE.shape = [sq, nhq]
    if return_lse:
        assert lse is not None
        if lse.ndim == 3:
            # [nhq, sq, 1] -> [sq, nhq]
            lse = lse.squeeze(-1)
        # [nhq, sq] -> [sq, nhq]
        lse = lse.t().contiguous()
    else:
        lse = None

    return out, lse


def _ref_attn_torch_impl_mainprocess_offline(
    q: torch.Tensor,
    kt: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    bias: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # apply `S = Q x K.T * scale + bias`
    # where S.shape = [nhq, sq, sk]
    s = to_higher_fp_dtype(
        q @ kt * softmax_scale,
        lowest_precision=torch.float32,
    )
    s += bias
    if sink is not None:
        # apply `S = S.concat(sink, dim=-1)`
        # where S.shape = [nhq, sq, sk + s_sink]
        s = torch.concat([s, sink], dim=-1)

    # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
    # where LSE.shape = [nhq, sq, 1]
    lse = safe_lse(s, dim=-1, keepdim=True)

    # apply row-wise softmax `P = softmax(S, dim=-1)`
    # where P.shape = [nhq, sq, sk + s_sink]
    # NOTE: pytorch softmax has many limitations and bugs
    # thus we use our own safe_softmax with lse involved
    p = safe_softmax(s, lse).to(q.dtype)
    if sink is not None:
        # apply `P = P.drop(sink, dim=-1)`
        # where P.shape = [nhq, sq, sk]
        p = p[..., : -sink.size(dim=-1)]

    # apply `O = P x V`
    # where O.shape = [nhq, sq, d]
    out = p @ v

    return out, lse


def _ref_attn_torch_impl_mainprocess_online(
    q: torch.Tensor,
    kt: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    bias: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = RefAttnTorchImplMainProcessOnline.apply(
        q, kt, v, sink, bias, softmax_scale
    )

    return out, lse


def _ref_attn_torch_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    mask: torch.Tensor,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    sink_layout: AttnSinkLayout = "sh",
    online_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    (q, kt, v, sink, bias, softmax_scale) = _ref_attn_torch_impl_preprocess(
        q=q,
        k=k,
        v=v,
        sink=sink,
        mask=mask,
        softmax_scale=softmax_scale,
        sink_layout=sink_layout,
    )

    mainprocess_func = (
        _ref_attn_torch_impl_mainprocess_online
        if online_softmax
        else _ref_attn_torch_impl_mainprocess_offline
    )

    out, lse = mainprocess_func(
        q=q,
        kt=kt,
        v=v,
        sink=sink,
        bias=bias,
        softmax_scale=softmax_scale,
    )

    out, lse = _ref_attn_torch_impl_postprocess(
        out=out,
        lse=lse,
        return_lse=return_lse,
    )

    return out, lse


def ref_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    layout: str = "thd",
    sink_layout: AttnSinkLayout = "sh",
    backend: str = "sdpa",
    high_precision: bool = False,
    return_lse: bool = False,
    online_softmax: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference Implementation of Attention Autograd Function

    Args:
        q (torch.Tensor): the query tensor
        k (torch.Tensor): the key tensor
        v (torch.Tensor): the value tensor
        mask (torch.Tensor): the boolean mask tensor,
            where the entries with ``False`` indicate that the corresponding positions are masked
        sink (torch.Tensor | None, optional): the sink tensor.
            Defaults to ``None`` to not apply attention sink.
        softmax_scale (float | None, optional): the softmax scale factor.
            Defaults to ``None`` to use the default value of ``1/sqrt(head_dim)``.
        softcap (float, optional): the softcap value.
            Defaults to ``0.0``.
        layout (str, optional): the shape layout of q,k,v,o tensors.
            Defaults to "thd" like ``[total_seqlen, num_heads, head_dim]``.
        sink_layout (AttnSinkLayout, optional): the shape layout of the sink tensor.
            Defaults to "sh" like ``[seqlen_sink, num_heads]``.
        backend (str, optional): the implementation backend.
            Defaults to "sdpa".
        high_precision (bool, optional): whether to use high precision (fp64) for computation.
            Defaults to ``False``.
        return_lse (bool, optional): whether to return log-sum-exp tensor.
            Defaults to ``False`` to return ``None``.
        online_softmax (bool, optional): whether to use online softmax to reduce memory overhead.
            Defaults to ``False``.

            NOTE: ``online_softmax`` flag takes no effect on sdpa backend,
                since it always uses online softmax.

    Raises:
        NotImplementedError: the specified backend is not supported

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            the output tensor and the optional log-sum-exp tensor
            if ``return_lse`` is ``True``, otherwise ``None``
    """
    assert layout in ("thd",), f"Unsupported layout: {layout}"
    assert softcap == 0.0, "non-zero softcap is not supported by now"

    # maybe cast input to high precision
    org_dtype = q.dtype
    lse_dtype = max_fp_dtype(org_dtype, torch.float32)
    if high_precision:  # use fp64 as ground-truth
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        v = v.to(torch.float64)

    # apply reference attention with specified backend
    match backend:
        case "sdpa":
            assert sink is None, "sink is not supported for sdpa backend by now"
            out, lse = _ref_attn_sdpa_impl(
                q=q,
                k=k,
                v=v,
                mask=mask,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
            )
        case "torch":
            out, lse = _ref_attn_torch_impl(
                q=q,
                k=k,
                v=v,
                sink=sink,
                mask=mask,
                softmax_scale=softmax_scale,
                return_lse=return_lse,
                sink_layout=sink_layout,
                online_softmax=online_softmax,
            )
        case _:
            raise NotImplementedError(f"Unsupported backend: {backend}")

    # maybe cast output back to original dtype
    out = out.to(org_dtype)
    if return_lse:
        assert lse is not None  # mypy
        lse = lse.to(lse_dtype)

    return out, lse
