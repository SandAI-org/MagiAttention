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
        inplace (bool, optional): whether to reduce to lse1 inplace. Defaults to False.

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
        lse1 (torch.Tensor): local lse for out1, with shape: [seqlen_q, num_heads]
        out2 (torch.Tensor): local output tensor2, with shape: [seqlen_q, num_heads_q, head_dim]
        lse2 (torch.Tensor): local lse for out2, with shape: [seqlen_q, num_heads]
        lse (torch.Tensor): global lse, with shape: [seqlen_q, num_heads]
        inplace (bool, optional): whether to reduce to out1 inplace. Defaults to False.

    Returns:
        torch.Tensor: corrected global output tensor, with shape: [seqlen_q, num_heads_q, head_dim]
    """
    assert out1.dtype == out2.dtype and lse1.dtype == lse2.dtype == lse.dtype

    w1, w2 = [
        # formula: wi = exp(lsei - lse)
        to_higher_fp_dtype(
            # shape: [s, h] -> [s, h, 1]
            safe_subtract(lsei, lse).exp().unsqueeze(-1),
            torch.float32,
        )
        for lsei in [lse1, lse2]
    ]

    # formula: out = w1 * out1 + w2 * out2
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
        inplace (bool, optional): whether to reduce to the first out and lse in the list inplace. Defaults to False.

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
