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

"""Host-side RangeMerge."""

from dataclasses import dataclass

import torch

from .ffa_utils import MT_MAP, materialize_mask_types

try:
    from magi_attention import magi_attn_ext  # type: ignore[attr-defined]
except ImportError:
    is_magi_attn_ext_installed = False
else:
    is_magi_attn_ext_installed = True

__all__ = [
    "merge_qk_ranges",
    "RangeMergePlan",
    "plan_range_merge",
    "plan_range_merge_bwd",
    "bwd_range_merge_arg",
]


@dataclass(frozen=True)
class RangeMergePlan:
    """Precomputed merge tables for RangeMerge.

    ``R`` relations fold into ``G`` groups of identical outer interval. The
    kernels schedule one batch per group, so the grid is sized from ``G``.
    """

    merged_outer_ranges: torch.Tensor  # [G, 2], sorted by start
    sorted_inner_ranges: torch.Tensor  # [R, 2], group-contiguous
    sorted_mask_types: torch.Tensor  # [R]
    cu_batches: torch.Tensor  # [G + 1] CSR into the R sorted rows
    bwd: "RangeMergePlan | None" = None  # K-merge; None => rebuild in bwd


def plan_range_merge(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | int | None = None,
) -> RangeMergePlan:
    """Q-merge plus paired K-merge on ``.bwd``."""
    mask_types = materialize_mask_types(
        MT_MAP.full if mask_types is None else mask_types,
        q_ranges.shape[0],
        q_ranges.device,
    )
    merged, sk, sm, cu = merge_qk_ranges(q_ranges, k_ranges, mask_types)
    return RangeMergePlan(
        merged, sk, sm, cu, bwd=plan_range_merge_bwd(q_ranges, k_ranges, mask_types)
    )


def plan_range_merge_bwd(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | int | None = None,
) -> RangeMergePlan:
    """K-merge (outer = K)."""
    mask_types = materialize_mask_types(
        MT_MAP.full if mask_types is None else mask_types,
        q_ranges.shape[0],
        q_ranges.device,
    )
    merged_k, sq, sm, cu = merge_qk_ranges(k_ranges, q_ranges, mask_types)
    return RangeMergePlan(merged_k, sq, sm, cu)


def bwd_range_merge_arg(
    range_merge: "bool | RangeMergePlan",
) -> "bool | RangeMergePlan":
    """Pick the K-merge plan, or ``True`` to rebuild."""
    if isinstance(range_merge, RangeMergePlan):
        if range_merge.bwd is not None:
            return range_merge.bwd
        return True
    return bool(range_merge)


def merge_qk_ranges(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Group relations by identical outer interval.

    Returns ``(merged_outer, sorted_inner, sorted_mask, cu_batches)``:
    ``[G, 2]``, ``[R, 2]``, ``[R]`` and ``[G + 1]`` for ``R`` relations in
    ``G`` groups, where group ``g`` owns sorted rows ``cu_batches[g]`` to
    ``cu_batches[g + 1]`` and ``cu_batches[G] == R``.

    Reading ``G`` synchronizes the device once per call. A precomputed
    :class:`RangeMergePlan` pays it once at construction; ``range_merge=True``
    pays it in every forward and backward call, and cannot be captured in a
    CUDA graph.
    """
    assert is_magi_attn_ext_installed, "magi_attn_ext must be installed for RangeMerge."
    assert q_ranges.shape == k_ranges.shape and q_ranges.shape[1] == 2

    sorted_idx, is_sorted = magi_attn_ext.argsort_ranges(q_ranges)
    sorted_q, sorted_k, sorted_mask = magi_attn_ext.reorder_ranges_and_attn_type_maps(
        q_ranges, k_ranges, mask_types, sorted_idx, is_sorted
    )
    # unique_consecutive_pairs returns R rows of unique pairs and R + 1 row
    # offsets, of which only the first G and G + 1 are written; the kernels
    # take num_batch from the row count, so the tail would launch CTAs that
    # read unwritten rows.
    merged_q, cu_batches, unique_count = magi_attn_ext.unique_consecutive_pairs(
        sorted_q
    )
    num_groups = int(unique_count.item())
    return merged_q[:num_groups], sorted_k, sorted_mask, cu_batches[: num_groups + 1]
