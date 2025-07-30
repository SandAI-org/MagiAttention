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

import warnings

import torch

from magi_attention.utils import nvtx

# isort: off
# We need to import the CUDA kernels after importing torch
is_ffa_installed = False
try:
    import flexible_flash_attention_cuda

    is_ffa_installed = True
except ImportError:
    warnings.warn("FFA is not installed.")

# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def merge_ranges(
    outer_ranges: torch.Tensor, inner_ranges: torch.Tensor, attn_type_map: torch.Tensor
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
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
          extracted from `outer_ranges`. Shape: `[num_ranges, 2]`.
        - sorted_outer_ranges (torch.Tensor): The original `outer_ranges` tensor
          after being sorted. Shape: `[num_ranges, 2]`.
        - sorted_inner_ranges (torch.Tensor): The original `inner_ranges` tensor,
          reordered to match the sorting of `outer_ranges`. Shape: `[num_ranges, 2]`.
        - range_map (torch.Tensor): The inverse index map. A tensor of shape
          `[num_ranges]`, where the value `range_map[i]` is the index of
          `sorted_outer_ranges[i]` in the `merge_outer_ranges` tensor.
        - unique_count (torch.Tensor): A scalar tensor containing a single
          integer, representing the number of unique ranges in `merge_outer_ranges`.

    Example:
        >>> outer_ranges = torch.tensor([[20, 30], [10, 20], [10, 20], [20, 30]], device='cuda', type=torch.int32)
        >>> inner_ranges = torch.tensor([[100, 110], [120, 130], [140, 150], [160, 170]], device='cuda', type=torch.int32)
        >>> attn_type_map = torch.tensor([0, 1, 0, 0], device='cuda', type=torch.int32)
        >>> (
        ...     merge_outer_ranges,
        ...     sorted_outer_ranges,
        ...     sorted_inner_ranges,
        ...     sorted_attn_type_map,
        ...     range_map,
        ...     unique_count,
        ... ) = merge_ranges(outer_ranges, inner_ranges)
        >>> print("Unique Merged Outer Ranges:", merge_outer_ranges)
        Unique Merged Outer Ranges:
         tensor([[10, 20],
                [20, 30],
                [0, 0],
                [0, 0]])
        >>> print("Sorted Outer Ranges:", sorted_outer_ranges)
        Sorted Outer Ranges:
         tensor([[10, 20],
                [10, 20],
                [20, 30],
                [20, 30]])
        >>> print("Sorted Inner Ranges (paired with sorted outer):", sorted_inner_ranges)
        Sorted Inner Ranges (paired with sorted outer):
         tensor([[120, 130],
                [140, 150],
                [100, 110],
                [160, 170]])
        >>> print("Sorted Attention Type Map:", sorted_attn_type_map)
        Sorted Attention Type Map:
         tensor([1, 0, 0, 0])
        >>> print("Range Map (inverse indices):", range_map)
        Range Map (inverse indices):
         tensor([0, 2, 0, 0])
        >>> print("Unique Count:", unique_count)
        Unique Count:
         tensor(2, dtype=torch.int32)
    """
    assert is_ffa_installed, "FFA is not installed."

    sorted_idx = torch.argsort(outer_ranges[:, 0], dim=0, stable=True)
    sorted_outer_ranges = outer_ranges[sorted_idx]
    sorted_inner_ranges = inner_ranges[sorted_idx]
    sorted_attn_type_map = attn_type_map[sorted_idx]
    (
        merge_outer_ranges,
        range_map,
        unique_count,
    ) = flexible_flash_attention_cuda.unique_consecutive_pairs(sorted_outer_ranges)

    return (
        merge_outer_ranges,
        sorted_outer_ranges,
        sorted_inner_ranges,
        sorted_attn_type_map,
        range_map,
        unique_count,
    )


@nvtx.instrument_nvtx
def _flex_flash_attn_forward(
    q,
    k,
    v,
    out,
    lse,
    q_ranges,
    k_ranges,
    max_seqlen_q,
    max_seqlen_k,
    attn_type_map,
    merge_q_ranges,
    qk_map,
    fwd_unique_count,
    softmax_scale,
    softcap,
    disable_fwd_atomic_reduction,
    out_type,
    deterministic,
    sm_margin,
):
    assert is_ffa_installed, "FFA is not installed."

    q, k, v, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (q, k, v, q_ranges, k_ranges)
    ]

    out, softmax_lse = flexible_flash_attention_cuda.fwd(
        q,
        k,
        v,
        out,
        lse,
        q_ranges,
        k_ranges,
        max_seqlen_q,
        max_seqlen_k,
        attn_type_map,
        merge_q_ranges,
        qk_map,
        fwd_unique_count,
        softmax_scale,
        softcap,
        disable_fwd_atomic_reduction,
        out_type,
        deterministic,
        sm_margin,
    )

    return out, softmax_lse


@nvtx.instrument_nvtx
def _flex_flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    dq,
    dk,
    dv,
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
    disable_bwd_dkv_atomic_reduction,
    dq_type,
    dk_type,
    dv_type,
    deterministic,
    sm_margin,
):
    assert is_ffa_installed, "FFA is not installed."

    dout, q, k, v, out, q_ranges, k_ranges = [
        maybe_contiguous(x) for x in (dout, q, k, v, out, q_ranges, k_ranges)
    ]

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
        dq,
        dk,
        dv,
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
        disable_bwd_dkv_atomic_reduction,
        dq_type,
        dk_type,
        dv_type,
        deterministic,
        sm_margin,
    )

    return dq, dk, dv, softmax_d


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
                fwd_attn_type_map,
                fwd_qk_map,
                fwd_unique_count,
            ) = merge_ranges(q_ranges, k_ranges, attn_type_map=attn_type_map)
            (
                merge_k_ranges,
                bwd_k_ranges,
                bwd_q_ranges,
                bwd_attn_type_map,
                bwd_kq_map,
                bwd_unique_count,
            ) = merge_ranges(k_ranges, q_ranges, attn_type_map=attn_type_map)
        else:
            fwd_q_ranges = q_ranges
            fwd_k_ranges = k_ranges
            fwd_attn_type_map = attn_type_map
            bwd_q_ranges = q_ranges
            bwd_k_ranges = k_ranges
            bwd_attn_type_map = attn_type_map
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
            None,  # out
            None,  # lse
            fwd_q_ranges,
            fwd_k_ranges,
            max_seqlen_q,
            max_seqlen_k,
            fwd_attn_type_map,
            merge_q_ranges,
            fwd_qk_map,
            fwd_unique_count,
            softmax_scale,
            softcap,
            disable_fwd_atomic_reduction,
            torch.float32,  # out_type
            deterministic,
            sm_margin,
        )

        # Cast output to the same dtype as q
        out = out.to(q.dtype)

        if auto_range_merge:
            ctx.save_for_backward(
                q,
                k,
                v,
                out,
                softmax_lse,
                bwd_q_ranges,
                bwd_k_ranges,
                bwd_attn_type_map,
                merge_k_ranges,
                bwd_kq_map,
                bwd_unique_count,
            )
        else:
            ctx.save_for_backward(
                q, k, v, out, softmax_lse, bwd_q_ranges, bwd_k_ranges, bwd_attn_type_map
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
                bwd_attn_type_map,
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
                bwd_attn_type_map,
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
            None,  # dq
            None,  # dk
            None,  # dv
            softmax_lse,
            bwd_q_ranges,
            bwd_k_ranges,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            bwd_attn_type_map,
            merge_k_ranges,
            bwd_kq_map,
            bwd_unique_count,
            softmax_scale=ctx.softmax_scale,
            softcap=ctx.softcap,
            disable_bwd_dkv_atomic_reduction=False,
            dq_type=torch.float32,
            dk_type=torch.float32,
            dv_type=torch.float32,
            deterministic=ctx.deterministic,
            sm_margin=ctx.sm_margin,
        )

        dq = dq.to(q.dtype)
        dk = dk.to(k.dtype)
        dv = dv.to(v.dtype)

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
    attn_type_map: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
    disable_fwd_atomic_reduction: bool = False,
    auto_range_merge: bool = False,
):
    """
    An interface similar to flash attention that doesn't require distributed environment, dispatch or undispatch.
    Directly call magi_attn_kernel to get attention output and lse. This is faster when you don't need context parallel.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        q_ranges (torch.Tensor): query ranges tensor to represent the attn mask.
        k_ranges (torch.Tensor): key ranges tensor to represent the attn mask.
        max_seqlen_q (int): Maximum sequence length of q_ranges.
        max_seqlen_k (int): Maximum sequence length of k_ranges.
        attn_type_map (torch.Tensor): Attention type map tenspr with dtype=torch.int32.
            The values specify the attention type for each token:

                - 0: full attention
                - 1: causal attention
                - 2: inverse causal attention
                - 3: bidirectional causal attention

            More information about the attention type map can be found in the ``Note`` below.

        softmax_scale (float, optional): Softmax scale, defaults to 1/sqrt(head_dim).
        softcap (float, optional): Softcap value, defaults to 0.
        deterministic (bool, optional): Whether to use deterministic attention, defaults to False.
        sm_margin (int, optional): the amount of SMs(streaming multiprocessors) reserved for communication.
        disable_fwd_atomic_reduction (bool):
            Whether to disable forward atomic reduction:

                If you can ensure q_ranges has no overlap, you can set this to True for better performance.
                Overlap in q_ranges is defined as: if any two q_ranges have non-empty intersection, then there is overlap.
                For example, q_ranges = ``[[0, 15], [10, 20], [20, 30]]`` has overlap because
                ``[0, 15]`` and ``[10, 20]`` intersect. While q_ranges = ``[[0, 15], [15, 20], [20, 30]]`` has no overlap.

        auto_range_merge (bool, optional): Whether to automatically merge k_ranges for the same q_range, defaults to False.

            **Note:** This flag is usually used in sparse attention cases but still under development.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Attention output tensor
            - lse (torch.Tensor): Log-sum-exp values with dtype=torch.float32.

    Shape:
        - q: (num_tokens_q, num_heads, head_dim)
        - k: (num_tokens_kv, num_heads, head_dim)
        - v: (num_tokens_kv, num_heads, head_dim)
        - q_ranges: (num_ranges, 2)
        - k_ranges: (num_ranges, 2)
        - attn_type_map: (num_ranges, )
        - out: (num_tokens_q, num_heads, head_dim)
        - lse: (num_heads, num_tokens_q)

    Note:
        The `attn_type_map` explains the semantics of different attention mask types.
        In addition to the descriptions below, see our blog for a visual explanation:
        https://sandai-org.github.io/MagiAttention/blog/#flex-flash-attn

        1. Full attention:
            If seqlen_q = 5 and seqlen_k = 2::

                1 1
                1 1
                1 1
                1 1
                1 1

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 1
                1 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1
                1 1 1 1 1

        2. Causal attention (bottom-right aligned):
            If seqlen_q = 5 and seqlen_k = 2::

                0 0
                0 0
                0 0
                1 0
                1 1

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 0
                1 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 0 0 0 0
                1 1 0 0 0
                1 1 1 0 0
                1 1 1 1 0
                1 1 1 1 1

        3. Inverse causal attention (top-left aligned):
            If seqlen_q = 5 and seqlen_k = 2::

                1 1
                0 1
                0 0
                0 0
                0 0

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 1
                0 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 1 1 1 1
                0 1 1 1 1
                0 0 1 1 1
                0 0 0 1 1
                0 0 0 0 1

        4. Bidirectional causal attention (intersection of causal and inverse causal):
            This is the element-wise AND of causal and inverse causal masks.

            If seqlen_q = 5 and seqlen_k = 2::

                0 0
                0 0
                0 0
                0 0
                0 0

            If seqlen_q = 2 and seqlen_k = 5::

                1 1 1 1 0
                0 1 1 1 1

            If seqlen_q = 5 and seqlen_k = 5::

                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                0 0 0 1 0
                0 0 0 0 1
    """

    assert not (auto_range_merge and deterministic), (
        "auto_range_merge and deterministic can't be True at the same time, "
        "due to some unresolved bug to be fixed as soon as possible."
    )

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
        disable_fwd_atomic_reduction,
        auto_range_merge,
    )
