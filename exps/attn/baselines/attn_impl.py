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

import math
from typing import Any, Optional

import torch
from flash_attn import flash_attn_func as fa2_func
from flash_attn import flash_attn_varlen_func as fa2_varlen_func
from flash_attn.cute.interface import flash_attn_func as fa4_func
from flash_attn.cute.interface import flash_attn_varlen_func as fa4_varlen_func
from flash_attn_interface import flash_attn_func as fa3_func
from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func
from packaging import version
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention as sdpa_func
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    FusedAttnFunc,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend

from magi_attention.functional import flex_flash_attn_func as ffa_func

if version.parse(torch.__version__) > version.parse("2.4"):
    # NOTE: in benchmarking, we should explicitly allow bf16/fp16 reduction for sdpa
    # by setting `torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)`
    # due to the new feature since torch2.5:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-in-scaled-dot-product-attention-sdpa
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)


# NOTE: we need to disable donated buffer to resolve the flex-attn error:
# RuntimeError: This backward function was compiled with non-empty donated buffers
# which requires create_graph=False and retain_graph=False.
# Please keep backward(create_graph=False, retain_graph=False) across all backward() function calls,
# or set torch._functorch.config.donated_buffer=False to disable donated buffer.
torch._functorch.config.donated_buffer = False
flex_attn_func = torch.compile(flex_attention)


def torch_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: Optional[float] = 0.0,
    is_causal: Optional[bool] = False,
    scale: Optional[float] = None,
    return_attn_probs: Optional[bool] = False,
) -> torch.Tensor:
    """naive pytorch implementation of scaled dot product attention (sdpa)

    Args:
        q: [b, h, sq, d]
        k: [b, h, sk, d]
        v: [b, h, sv, d]

    Returns:
        o: [b, h, sq, d]
    """
    # init
    sq, sk = q.size(-2), k.size(-2)
    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(sq, sk, dtype=q.dtype).to(q.device)

    # get attn_bias / attn_mask
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(sq, sk, dtype=torch.bool).tril(diagonal=0).to(q.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    # scaled dot for q,k
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    if return_attn_probs:
        lse = torch.logsumexp(attn_weight, dim=-1)

    # softmax
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # dropout
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    # weighted sum with v
    attn_out = attn_weight @ v

    if return_attn_probs:
        return attn_out, lse.to(q.dtype)
    return attn_out


def cudnn_fused_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    dropout_p: Optional[float] = 0.0,
    is_causal: Optional[bool] = False,
    softmax_scale: Optional[float] = None,
    window_size: Optional[tuple[int, int]] = None,
    is_training: bool = False,
    deterministic: bool = False,
) -> torch.Tensor:
    # prepare args
    attn_mask_type = "padding_causal" if is_causal else "padding"
    qkv_layout = "thd_thd_thd"
    softmax_type = "vanilla"

    fast_zero_fill = True
    softmax_scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    fused_attention_backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    use_FAv2_bwd = False
    fp8 = False
    fp8_meta: dict[str, Any] = {}
    fp8_output = False
    quantizers = None
    window_size = window_size if window_size is not None else (-1, -1)
    page_table_k = None
    page_table_v = None

    # for thd input, cudnn does not support bias
    core_attention_bias_type = "no_bias"
    core_attention_bias = None
    softmax_offset = None
    layer_number = 1

    output = FusedAttnFunc.apply(
        is_training,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        page_table_k,
        page_table_v,
        q,
        k,
        v,
        core_attention_bias,
        softmax_scale,
        dropout_p if is_training else 0.0,
        fast_zero_fill,
        qkv_layout,
        core_attention_bias_type,
        attn_mask_type,
        softmax_type,
        window_size,
        None,  # rng_gen
        fused_attention_backend,
        use_FAv2_bwd,
        fp8,
        fp8_meta,
        quantizers,
        deterministic,
        softmax_offset,
        fp8_output,
        layer_number,
    )

    return output


__all__ = [
    "torch_attn_func",
    "fa2_func",
    "fa3_func",
    "fa4_func",
    "ffa_func",
    "sdpa_func",
    "cudnn_fused_attn_func",
    "flex_attn_func",
    "fa2_varlen_func",
    "fa3_varlen_func",
    "fa4_varlen_func",
]
