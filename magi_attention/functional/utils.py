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

from magi_attention.utils import nvtx, to_higher_fp_dtype


def safe_subtract(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Safely subtracts two tensors,
    where the subtraction results of two -inf will be set to -inf.
    """

    eq = (a == b) & (a == float("-inf"))
    sub = a - b
    sub = torch.where(eq, torch.fill(sub, float("-inf")), sub)

    return sub


def softmax_bwd(dout: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Standard backward func for `out = softmax(inp)`"""

    diag_out = torch.diag_embed(out)
    outer_out = torch.einsum("...ij, ...ik -> ...ijk", out, out)

    dinp = torch.einsum("...ij, ...ijk -> ...ik", dout, diag_out - outer_out)

    return dinp


def correct_attn_lse(
    lse1: torch.Tensor,
    lse2: torch.Tensor,
) -> torch.Tensor:
    """
    Corrects the log-sum-exp tensor for online attention.

    Args:
        lse1 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen, num_heads]
        lse2 (torch.Tensor): log-sum-exp tensor, with shape: [seqlen, num_heads]

    Returns:
        torch.Tensor: corrected log-sum-exp tensor, with shape: [seqlen, num_heads]
    """

    min_lse = to_higher_fp_dtype(torch.min(lse1, lse2), torch.float32)
    max_lse = to_higher_fp_dtype(torch.max(lse1, lse2), torch.float32)

    # formula derivation:
    # lse = log(exp(lse1) + exp(lse2))
    #     = lse1 + log(1 + exp(lse2 - lse1))
    #     = max_lse + log(1 + exp(min_lse - max_lse))
    #     = max_lse + log1p(exp(min_lse - max_lse))
    #     = max_lse + softplus(min_lse - max_lse)
    lse = max_lse + F.softplus(safe_subtract(min_lse, max_lse))

    return lse.to(lse1.dtype)


correct_attn_lse_compiled = torch.compile(correct_attn_lse)


def correct_attn_out(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """
    Corrects the output tensor for online attention.

    Args:
        out1 (torch.Tensor): local output tensor1, with shape: [seqlen, num_heads, head_dim]
        lse1 (torch.Tensor): local lse for out1, with shape: [seqlen, num_heads]
        out2 (torch.Tensor): local output tensor2, with shape: [seqlen, num_heads, head_dim]
        lse2 (torch.Tensor): local lse for out2, with shape: [seqlen, num_heads]
        lse (torch.Tensor): global lse, with shape: [seqlen, num_heads]

    Returns:
        torch.Tensor: corrected global output tensor, with shape: [seqlen, num_heads, head_dim]
    """
    # formula: lsei_ = exp(lsei - lse)
    # shape: [s, h] -> [s, h, 1]
    lse1_, lse2_ = [
        to_higher_fp_dtype(
            safe_subtract(lsei, lse).exp().unsqueeze(-1),
            torch.float32,
        )
        for lsei in [lse1, lse2]
    ]

    out = lse1_ * out1 + lse2_ * out2

    return out.to(out1.dtype)


correct_attn_out_compiled = torch.compile(correct_attn_out)


@nvtx.instrument_nvtx
def correct_attn_fwd_result(
    out_list: list[torch.Tensor],
    lse_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Corrects the attention result.

    Args:
        out_list (list[torch.Tensor]): the list of partial out tensors
        lse_list (list[torch.Tensor]): the list of partial lse tensors

    Returns:
        tuple[torch.Tensor, torch.Tensor]: the corrected out and lse

    Shape:
        out: [seqlen, num_heads, head_dim]
        lse: [seqlen, num_heads]
    """
    if len(lse_list) == 1:
        # NOTE: if there is only one out and lse,
        # we just return them directly, no need to correct
        return out_list[0], lse_list[0]

    curr_lse = None
    curr_out = None

    for i in range(len(lse_list) - 1):
        if i == 0:
            curr_lse = correct_attn_lse_compiled(lse_list[0], lse_list[1])
            curr_out = correct_attn_out_compiled(
                out_list[0], lse_list[0], out_list[1], lse_list[1], curr_lse
            )
        else:
            original_lse = curr_lse
            original_out = curr_out
            curr_lse = correct_attn_lse_compiled(original_lse, lse_list[i + 1])
            curr_out = correct_attn_out_compiled(
                original_out,
                original_lse,
                out_list[i + 1],
                lse_list[i + 1],
                curr_lse,
            )

    return curr_out, curr_lse
