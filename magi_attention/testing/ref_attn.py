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


import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from packaging import version
from torch.nn.attention import SDPBackend, sdpa_kernel

from magi_attention.common.enum import AttnSinkLayout
from magi_attention.functional.utils import (
    correct_attn_lse,
    correct_attn_out,
    safe_softmax,
)
from magi_attention.utils import max_fp_dtype, to_higher_fp_dtype

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in testing, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


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
        to_higher_fp_dtype(
            q @ kt * softmax_scale + bias,
            lowest_precision=torch.float32,
        )
        # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
        # where LSE.shape = [nhq, sq]
        .logsumexp(dim=-1)
        # transpose and make contiguous
        # where LSE.shape = [sq, nhq]
        .t().contiguous()
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
        lse = rearrange(lse, "nhq sq 1 -> sq nhq")
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
    lse = s.logsumexp(dim=-1, keepdim=True)

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
    block_q: int = 1024,
    block_k: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    # fetch meta info
    # q.shape = [nhq, sq, d]
    # kt.shape = [nhq, d, sk]
    # v.shape = [nhq, sk, d]
    sq, sk, nhq = q.size(-2), kt.size(-1), q.size(0)
    lse_dtype = max_fp_dtype(q.dtype, torch.float32)

    # init out, lse buffer
    # out.shape = [nhq, sq, d]
    # lse.shape = [nhq, sq]
    out = torch.zeros_like(q)
    lse = torch.full((nhq, sq), -float("inf"), dtype=lse_dtype, device=q.device)
    if sink is not None:
        # directly initialize lse with lse_sink
        # if sink is provided
        lse_sink = (
            # shape: [nhq, sq, s_sink] -> [nhq, sq]
            torch.logsumexp(sink, dim=-1, keepdim=False)
        )
        correct_attn_lse(
            lse1=lse,
            lse2=lse_sink,
            inplace=True,
        )

    # outer loop of Q,O
    for q_start in range(0, sq, block_q):
        q_end = min(q_start + block_q, sq)
        # bq/bout.shape: [nhq, block_q, hd]
        # blse.shape: [nhq, block_q]
        bq, bout, blse = (
            q[:, q_start:q_end],
            out[:, q_start:q_end],
            lse[:, q_start:q_end],
        )

        # inner loop of K,V
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
            # where S.shape = [nhq, block_q, block_k]
            bs = to_higher_fp_dtype(
                bq @ bkt * softmax_scale,
                lowest_precision=torch.float32,
            )
            bs += bbias

            # apply row-wise lse `LSE = logsumexp(S, dim=-1)`
            # where LSE.shape = [nhq, block_q, 1]
            blse_ = bs.logsumexp(dim=-1, keepdim=True)

            # apply row-wise softmax `P = softmax(S, dim=-1)`
            # where P.shape = [nhq, block_q, block_k]
            # NOTE: pytorch softmax has many limitations and bugs
            # thus we use our own safe_softmax with lse involved
            bp = safe_softmax(bs, blse_).to(q.dtype)

            # apply `O = P x V`
            # where O.shape = [nhq, block_q, hd]
            bout_ = bp @ bv

            # correct blse
            blse_ = blse_.squeeze_(-1)
            blse_old = blse.clone()
            correct_attn_lse(
                lse1=blse,
                lse2=blse_,
                inplace=True,
            )

            # correct bout
            correct_attn_out(
                out1=bout,
                lse1=blse_old,
                out2=bout_,
                lse2=blse_,
                lse=blse,
                inplace=True,
            )

    return out, lse.unsqueeze_(-1)


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
