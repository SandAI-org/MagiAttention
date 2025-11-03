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
from einops import rearrange, reduce, repeat

from magi_attention.utils import to_higher_fp_dtype


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors,
    where the subtraction results of two `-inf` will be set to `-inf`, instead of `nan`.
    """

    eq = (a == b) & (a == float("-inf"))
    sub = a - b
    sub = torch.where(eq, torch.fill(sub, float("-inf")), sub)

    return sub


def safe_softmax(
    a: torch.Tensor,
    dim: int = -1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Safely applies row-wise softmax to the input tensor,
    where all-`-inf` / all-nan rows will be set to all-zero rows during forward/backward resp.
    """
    all_neg_inf_mask = (a == float("-inf")).all(dim=dim, keepdim=True)

    def safe_softmax_bwd_hook(g):
        all_nan_mask = (g != g).all(dim=dim, keepdim=True)
        g.masked_fill_(all_nan_mask, 0.0)
        return g

    a.register_hook(safe_softmax_bwd_hook)

    sm = F.softmax(a, dim=dim, dtype=dtype)

    sm = sm.masked_fill(all_neg_inf_mask, 0.0)

    return sm


def softmax_bwd(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Standard backward func for `out = softmax(inp)`"""

    diag_out = torch.diag_embed(out)
    outer_out = torch.einsum("...ij, ...ik -> ...ijk", out, out)
    dinp = torch.einsum("...ij, ...ijk -> ...ik", dout, diag_out - outer_out)

    return dinp


def sink_bwd(
    sink: torch.Tensor,
    lse: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
) -> torch.Tensor:
    # prepare sink, lse
    sink_dtype = sink.dtype
    sink = repeat(sink, "s_sink hq -> hq sq s_sink", sq=lse.size(0)).to(lse.dtype)
    lse = rearrange(lse, "sq hq -> hq sq 1")

    # calculate delta = (o * do).sum(dim=-1)
    # where o.shape = [sq, nhq, d]
    #       do.shape = [sq, nhq, d]
    #       delta.shape = [nhq, sq, 1]
    delta = reduce(o.to(lse.dtype) * do.to(lse.dtype), "sq hq d -> hq sq 1", "sum")

    # calculat p_sink = exp(sink - lse)
    # where sink.shape = [nhq, sq, s_sink]
    #       lse.shape = [nhq, sq, 1]
    #       p_sink.shape = [nhq, sq, s_sink]
    p_sink = torch.exp(sink - lse)

    # calculate dsink = p_sink.T x -delta
    # where p_sink.shape = [nhq, sq, s_sink]
    #       delta.shape = [[nhq, sq, 1]
    #       dsink.shape = [s_sink, nhq]
    dsink = (
        # shape: [nhq, s_sink, sq] x [nhq, sq, 1] -> [nhq, s_sink, 1]
        (p_sink.transpose(-1, -2) @ -delta)
        # shape: [nhq, s_sink, 1] -> [nhq, s_sink]
        .squeeze(-1)
        # shape: [nhq, s_sink] -> [s_sink, nhq]
        .t()
    ).to(sink_dtype)

    return dsink


def calc_lse_rescale_weight(
    lse_to_rescale: torch.Tensor,
    rescaled_lse: torch.Tensor,
) -> torch.Tensor:
    """Calculate the rescale weight to correct attention output
    given the lse to rescale (i.e. the old original lse)
    and the rescaled lse (i.e. the new lse)

    Args:
        lse_to_rescale (torch.Tensor): the old log-sum-exp tensor to rescale
            with shape: [seqlen_q, num_heads_q]
        rescaled_lse (torch.Tensor): the new rescaled log-sum-exp tensor
            with shape: [seqlen_q, num_heads_q]

    Returns:
        torch.Tensor: the rescale weight with shape: [seqlen_q, num_heads_q, 1]
    """
    # calculate the rescale weight with shape: [sq, nhq, 1]
    # formula: w = exp(lse_old - lse_new)
    return safe_subtract(lse_to_rescale, rescaled_lse).exp().unsqueeze(-1)


def calc_lse_sink(
    sink: torch.Tensor,
    ref_lse: torch.Tensor,
) -> torch.Tensor:
    """Calculate the log-sum-exp of the sink tokens
    and broadcast it to the shape of the reference lse

    Args:
        sink (torch.Tensor): the sink tokens with shape: [seqlen_sink, num_heads_q]
        ref_lse (torch.Tensor): the reference lse tensor with shape: [seqlen_q, num_heads_q]
            to provide the meta info like shape

    Returns:
        torch.Tensor: the log-sum-exp of the sink tokens
    """
    # calculate lse_sink and broadcast it to ref lse's shape
    lse_sink = (
        # shape: [s_sink, nhq] -> [nhq,]
        torch.logsumexp(sink, dim=0)
        # shape: [nhq,] -> [1, nhq]
        .unsqueeze(0)
        # shape: [1, nhq] -> [sq, nhq]
        .expand_as(ref_lse)
    )

    return lse_sink


def correct_attn_lse(
    lse1: torch.Tensor,
    lse2: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the log-sum-exp tensor for online attention.

    Args:
        lse1 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        lse2 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        inplace (bool, optional): whether to correct ``lse1`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
    """

    assert lse1.dtype == lse2.dtype

    min_lse = to_higher_fp_dtype(torch.min(lse1, lse2), torch.float32)
    max_lse = to_higher_fp_dtype(torch.max(lse1, lse2), torch.float32)

    # formula derivation:
    # lse = log(exp(lse1) + exp(lse2))
    #     = lse1 + log(1 + exp(lse2 - lse1))
    #     = max_lse + log(1 + exp(min_lse - max_lse))
    #     = max_lse + log1p(exp(min_lse - max_lse))
    #     = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))

    lse = lse1.copy_(lse) if inplace else lse.to(lse1.dtype)

    return lse


correct_attn_lse_compiled = torch.compile(correct_attn_lse)


def correct_attn_out(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
    lse: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the output tensor for online attention.

    Args:
        out1 (torch.Tensor): local output tensor1, with shape: [seqlen_q, num_heads_q, head_dim]
        lse1 (torch.Tensor): local lse for out1, with shape: [seqlen_q, num_heads_q]
        out2 (torch.Tensor): local output tensor2, with shape: [seqlen_q, num_heads_q, head_dim]
        lse2 (torch.Tensor): local lse for out2, with shape: [seqlen_q, num_heads_q]
        lse (torch.Tensor): global lse, with shape: [seqlen_q, num_heads_q]
        inplace (bool, optional): whether to correct ``out1`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected global output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
    """
    assert out1.dtype == out2.dtype and lse1.dtype == lse2.dtype == lse.dtype

    # calculate the rescale weight for out1 and out2 resp.
    w1, w2 = [calc_lse_rescale_weight(lsei, lse) for lsei in (lse1, lse2)]

    # correct out
    # with formula: out = w1 * out1 + w2 * out2
    if inplace:
        out1 *= w1
        out = out1.add_(w2 * out2)
    else:
        out = w1 * out1 + w2 * out2
        out = out.to(out1.dtype)

    return out


correct_attn_out_compiled = torch.compile(correct_attn_out)


def correct_attn_fwd_result(
    out_list: list[torch.Tensor], lse_list: list[torch.Tensor], inplace: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the attention result given all of the partial out and lse

    Args:
        out_list (list[torch.Tensor]): the list of partial out tensors
        lse_list (list[torch.Tensor]): the list of partial lse tensors
        inplace (bool, optional):
            whether to reduce the corrected results to the first ``out`` and ``lse``
            in the list inplace. Defaults to ``False``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the corrected out and lse

    Shape:
        out: [seqlen_q, num_heads_q, head_dim]
        lse: [seqlen_q, num_heads_q]
    """
    assert len(out_list) == len(lse_list) and len(out_list) >= 1

    corrected_out, corrected_lse = out_list[0], lse_list[0]
    for i in range(1, len(out_list)):
        last_lse = corrected_lse.clone() if inplace else corrected_lse
        corrected_lse = correct_attn_lse_compiled(
            lse1=corrected_lse,
            lse2=lse_list[i],
            inplace=inplace,
        )
        corrected_out = correct_attn_out_compiled(
            out1=corrected_out,
            lse1=last_lse,
            out2=out_list[i],
            lse2=lse_list[i],
            lse=corrected_lse,
            inplace=inplace,
        )

    return corrected_out, corrected_lse


def correct_attn_lse_with_sink(
    lse: torch.Tensor,
    sink: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the log-sum-exp tensor with sink tokens

    Args:
        lse (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): log-sum-exp tensor, with shape: [seqlen_sink, num_heads_q]
        inplace (bool, optional): whether to correct ``lse`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected log-sum-exp tensor, with shape: [seqlen_q, num_heads_q]
    """
    lse_sink = calc_lse_sink(sink, lse)

    return correct_attn_lse(lse, lse_sink, inplace=inplace)


def correct_attn_out_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Corrects the output tensor with sink tokens

    Args:
        out (torch.Tensor): output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        lse (torch.Tensor): lse tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): local lse for out2, with shape: [seqlen_sink, num_heads_q]
        inplace (bool, optional): whether to correct ``out`` inplace. Defaults to ``False``.

    Returns:
        torch.Tensor: corrected output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
    """
    lse_sink = calc_lse_sink(sink, lse)

    # calculate the rescale weight with shape: [sq, nhq, 1]
    # formula derivation:
    # w = exp(lse_old - lse_new)
    #   = exp(lse - log(exp(lse) + exp(lse_sink)))
    #   = exp(lse) / (exp(lse) + exp(lse_sink))
    #   = 1 / (1 + exp(lse_sink - lse))
    w = torch.reciprocal(1 + calc_lse_rescale_weight(lse_sink, lse))

    return out.mul_(w) if inplace else (out * w).to(out.dtype)


def correct_attn_out_lse_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the output tensor and log-sum-exp tensor with sink tokens

    Args:
        out (torch.Tensor): output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
        lse (torch.Tensor): lse tensor, with shape: [seqlen_q, num_heads_q]
        sink (torch.Tensor): local lse for out2, with shape: [seqlen_sink, num_heads_q]
        inplace (bool, optional): whether to correct ``out`` inplace. Defaults to ``False``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: corrected output and lse
            - out: corrected output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
            - lse: corrected lse tensor, with shape: [seqlen_q, num_heads_q]
    """
    lse_sink = calc_lse_sink(sink, lse)
    lse_new = correct_attn_lse(lse, lse_sink, inplace=False)
    w = calc_lse_rescale_weight(lse, lse_new)

    out = out.mul_(w) if inplace else (out * w).to(out.dtype)
    lse = lse.copy_(lse_new) if inplace else lse_new

    return out, lse
