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

from typing import Optional

import torch

from magi_attention.utils import nvtx

# isort: off
# We need to import the CUDA kernels after importing torch
import flexible_flash_attention_cuda
from magi_attention.functional.merge_range import unique_consecutive_pairs

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def merge_ranges(
    outer_ranges: torch.Tensor, inner_ranges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sorts and deduplicates range tensors that represent Q-K attention block pairs.

    Args:
        outer_ranges (torch.Tensor): A tensor of shape `[num_ranges, 2]`,
            representing the outer ranges (e.g., Q-block ranges).
        inner_ranges (torch.Tensor): A tensor of shape `[num_ranges, 2]`,
            where each row is paired with the corresponding row in
            `outer_ranges`, representing the inner ranges (e.g., K-block ranges).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A tuple containing five tensors:
        - merge_outer_ranges (torch.Tensor): The consecutive and unique ranges
          extracted from `outer_ranges`. Shape: `[unique_ranges, 2]`.
        - sorted_outer_ranges (torch.Tensor): The original `outer_ranges` tensor
          after being sorted. Shape: `[num_ranges, 2]`.
        - sorted_inner_ranges (torch.Tensor): The original `inner_ranges` tensor,
          reordered to match the sorting of `outer_ranges`. Shape: `[num_ranges, 2]`.
        - range_map (torch.Tensor): The inverse index map. A tensor of shape
          `[num_ranges]`, where the value `range_map[i]` is the index of
          `sorted_outer_ranges[i]` in the `merge_outer_ranges` tensor.
        - unique_count (torch.Tensor): A scalar tensor containing a single
          integer, representing the number of unique ranges in `merge_outer_ranges`.
    """
    sorted_idx = torch.argsort(outer_ranges[:, 0], dim=0, stable=True)
    sorted_outer_ranges = outer_ranges[sorted_idx]
    sorted_inner_ranges = inner_ranges[sorted_idx]
    merge_outer_ranges, range_map, unique_count = unique_consecutive_pairs(
        sorted_outer_ranges
    )

    return (
        merge_outer_ranges,
        sorted_outer_ranges,
        sorted_inner_ranges,
        range_map,
        unique_count,
    )


@nvtx.instrument_nvtx
def _flex_flash_attn_forward(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    max_seqlen_q,
    max_seqlen_k,
    attn_type_map,
    merge_q_ranges,
    qk_map,
    unique_count,
    softmax_scale,
    softcap,
    deterministic,
    sm_margin,
    return_dtype,
    disable_fwd_atomic_reduction,
):
    q, k, v, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (q, k, v, q_ranges, k_ranges)
    ]

    if q_ranges.shape[0] == 0:
        # FIXME: This logic should be written in the cuda kernel, this is a temporary workaround
        ttk, nh, hd = q.shape
        out = torch.zeros_like(q)
        softmax_lse = torch.empty(nh, ttk, dtype=torch.float32)
        softmax_lse.fill_(-float("inf"))
    else:
        out, softmax_lse = flexible_flash_attention_cuda.fwd(
            q,
            k,
            v,
            q_ranges,
            k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            merge_q_ranges,
            qk_map,
            unique_count,
            softmax_scale,
            softcap,
            sm_margin,
            disable_fwd_atomic_reduction,
            return_dtype,
        )

    return out, softmax_lse


@nvtx.instrument_nvtx
def _flex_flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    q_ranges,
    k_ranges,
    max_seqlen_q,
    max_seqlen_k,
    attn_type_map,
    merge_k_ranges,
    bwd_kq_map,
    bwd_unique_count,
    softmax_scale,
    softcap,
    deterministic,
    sm_margin,
):
    dout, q, k, v, out, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (dout, q, k, v, out, q_ranges, k_ranges)
    ]

    if q_ranges.shape[0] == 0:
        # FIXME: This logic should be written in the cuda kernel, this is a temporary workaround
        ttk, nh, hd = q.shape
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        softmax_d = torch.zeros(nh, ttk, dtype=torch.float32)
    else:
        (
            dq,
            dk,
            dv,
            softmax_d,
            _,
        ) = flexible_flash_attention_cuda.bwd(
            dout,
            q,
            k,
            v,
            out,
            None,
            None,
            None,
            softmax_lse,
            q_ranges,
            k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale,
            softcap,
            torch.float32,
            deterministic,
            sm_margin,
        )

    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype), softmax_d


class FlexFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map,
        softmax_scale,
        softcap=0.0,
        deterministic=False,
        sm_margin=0,
        return_dtype=None,
        disable_fwd_atomic_reduction=False,
        auto_range_merge=False,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert isinstance(
            max_seqlen_q, int
        ), "max_seqlen_q must be an int, otherwise would lead to performance degradation"
        assert isinstance(
            max_seqlen_k, int
        ), "max_seqlen_k must be an int, otherwise would lead to performance degradation"

        if auto_range_merge:
            (
                merge_q_ranges,
                fwd_q_ranges,
                fwd_k_ranges,
                fwd_qk_map,
                fwd_unique_count,
            ) = merge_ranges(q_ranges, k_ranges)
            (
                merge_k_ranges,
                bwd_k_ranges,
                bwd_q_ranges,
                bwd_kq_map,
                bwd_unique_count,
            ) = merge_ranges(k_ranges, q_ranges)
        else:
            fwd_q_ranges = q_ranges
            fwd_k_ranges = k_ranges
            bwd_q_ranges = q_ranges
            bwd_k_ranges = k_ranges
            merge_q_ranges = None
            merge_k_ranges = None
            fwd_qk_map = None
            bwd_kq_map = None
            fwd_unique_count = None
            bwd_unique_count = None

        out, softmax_lse = _flex_flash_attn_forward(
            q,
            k,
            v,
            fwd_q_ranges,
            fwd_k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            attn_type_map,
            merge_q_ranges,
            fwd_qk_map,
            fwd_unique_count,
            softmax_scale,
            softcap,
            deterministic,
            sm_margin,
            return_dtype,
            disable_fwd_atomic_reduction,
        )

        if auto_range_merge:
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                softmax_lse,
                bwd_q_ranges,
                bwd_k_ranges,
                attn_type_map,
                merge_k_ranges,
                bwd_kq_map,
                bwd_unique_count,
            )
        else:
            ctx.save_for_backward(
                q, k, v, out, softmax_lse, bwd_q_ranges, bwd_k_ranges, attn_type_map
            )

        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.auto_range_merge = auto_range_merge

        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.auto_range_merge:
            (
                q,
                k,
                v,
                out,
                softmax_lse,
                bwd_q_ranges,
                bwd_k_ranges,
                attn_type_map,
                merge_k_ranges,
                bwd_kq_map,
                bwd_unique_count,
            ) = ctx.saved_tensors
        else:
            (
                q,
                k,
                v,
                out,
                softmax_lse,
                bwd_q_ranges,
                bwd_k_ranges,
                attn_type_map,
            ) = ctx.saved_tensors
            merge_k_ranges = None
            bwd_kq_map = None
            bwd_unique_count = None

        dq, dk, dv, _ = _flex_flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            bwd_q_ranges,
            bwd_k_ranges,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
            deterministic=ctx.deterministic,
            sm_margin=ctx.sm_margin,
        )

        return (
            dq,
            dk,
            dv,
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


def flex_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    attn_type_map: Optional[torch.Tensor] = None,
    softmax_scale=None,
    softcap=0.0,
    deterministic=False,
    sm_margin=0,
    return_dtype=None,
    disable_fwd_atomic_reduction=False,
    auto_range_merge=False,
):
    """
    An interface similar to flash attention that doesn't require distributed environment, dispatch or undispatch.
    Directly call magi_attn_kernel to get attention output and lse. This is faster when you don't need context parallel.
    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        q_ranges (torch.Tensor): query ranges in the ref attn mask.
        k_ranges (torch.Tensor): key ranges in the ref attn mask.
        max_seqlen_q (int): Maximum sequence length of q_ranges.
        max_seqlen_k (int): Maximum sequence length of k_ranges.
        attn_type_map (torch.Tensor): Attention type map with dtype=torch.int32.
            0: full attention
            1: causal attention
            2: inverse causal attention
            3: bidirectional causal attention
        softmax_scale (float): Softmax scale.
        softcap (float): Softcap.
        deterministic (bool): Whether to use deterministic attention.
        sm_margin (int): the amount of SMs(streaming multiprocessors) reserved for communication.
        return_dtype (torch.dtype): Return dtype.
        disable_fwd_atomic_reduction (bool): Whether to disable forward atomic reduction.
            If you can ensure q_ranges has no overlap, you can set this to True for better performance.
            Overlap in q_ranges is defined as: if any two q_ranges have non-empty intersection, then there is overlap.
            For example, q_ranges = [[0, 15], [10, 20], [20, 30]] has overlap because [0, 15] and [10, 20] intersect.
            While q_ranges = [[0, 15], [15, 20], [20, 30]] has no overlap.
    Returns:
        out (torch.Tensor): Attention output tensor
        lse (torch.Tensor): Log-sum-exp values with dtype=torch.float32.
    Shape:
        q: (num_tokens_q, num_heads, head_dim)
        k: (num_tokens_kv, num_heads, head_dim)
        v: (num_tokens_kv, num_heads, head_dim)
        q_ranges: (num_ranges, 2)
        k_ranges: (num_ranges, 2)
        attn_type_map: (num_ranges, )
        out: (num_tokens_q, num_heads, head_dim)
        lse: (num_heads, num_tokens_q)
    NOTE: attn_type_map explanation:
        (In addition to the textual explanations provided below, feel free to check out our blog for a visual interpretation:
        https://sandai-org.github.io/MagiAttention/#flex-flash-attn)
        1. full attention
            If seqlen_q = 5 and seqlen_k = 2, the full mask is:
                1 1
                1 1
                1 1
                1 1
                1 1
            If seqlen_q = 2 and seqlen_k = 5, the full mask is:
                1 1 1 1 1
                1 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the full mask is:
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
        2: causal attention (bottom-right aligned)
            If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
                0 0
                0 0
                0 0
                1 0
                1 1
            if seqlen_q = 2 and seqlen_k = 5, the causal mask is:
                1 1 1 1 0
                1 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the causal mask is:
                1 0 0 0 0
                1 1 0 0 0
                1 1 1 0 0
                1 1 1 1 0
                1 1 1 1 1
        3: inverse causal attention (top-left aligned)
            if seqlen_q = 5 and seqlen_k = 2, the inverse causal mask is:
                1 1
                0 1
                0 0
                0 0
                0 0
            if seqlen_q = 2 and seqlen_k = 5, the inverse causal mask is:
                1 1 1 1 1
                0 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the inverse causal mask is:
                1 1 1 1 1
                0 1 1 1 1
                0 0 1 1 1
                0 0 0 1 1
                0 0 0 0 1
        4. bidirectional causal attention (top-left & bottom-right intersect-aligned)
            bidirectional causal attention mask is an 'and mask' of causal and inverse causal attention.
            if seqlen_q = 5 and seqlen_k = 2, the bidirectional causal mask is:
                0 0
                0 0
                0 0
                0 0
                0 0
            if seqlen_q = 2 and seqlen_k = 5, the bidirectional causal mask is:
                1 1 1 1 0
                0 1 1 1 1
            if seqlen_q = 5 and seqlen_k = 5, the bidirectional causal mask is:
                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0
                0 0 0 0 1
    """
    assert not deterministic, "deterministic is not supported yet."

    return FlexFlashAttnFunc.apply(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map,
        softmax_scale,
        softcap,
        deterministic,
        sm_margin,
        return_dtype,
        disable_fwd_atomic_reduction,
        auto_range_merge,
    )
