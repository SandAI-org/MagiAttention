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

from dataclasses import dataclass, replace

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

    ``R`` relations fold into ``G`` groups of identical outer interval, one
    kernel batch per group, in the launch order of :func:`merge_qk_ranges`.
    ``merged_outer_ranges_by_start`` holds the same groups by outer start for
    the dK/dV hole sweep, which binary-searches range starts.
    """

    merged_outer_ranges: torch.Tensor  # [G, 2], launch order
    sorted_inner_ranges: torch.Tensor  # [R, 2], group-contiguous
    sorted_mask_types: torch.Tensor  # [R]
    cu_batches: torch.Tensor  # [G + 1] CSR into the R sorted rows
    merged_outer_ranges_by_start: torch.Tensor  # [G, 2], start ascending
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
    return replace(
        merge_qk_ranges(q_ranges, k_ranges, mask_types),
        bwd=plan_range_merge_bwd(q_ranges, k_ranges, mask_types),
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
    return merge_qk_ranges(k_ranges, q_ranges, mask_types)


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
    mask_types: torch.Tensor,
) -> RangeMergePlan:
    """Group relations by identical outer interval, heaviest group first.

    Returns the plan without ``bwd``: ``merged_outer_ranges`` ``[G, 2]``,
    ``sorted_inner_ranges`` ``[R, 2]``, ``sorted_mask_types`` ``[R]`` and
    ``cu_batches`` ``[G + 1]`` for ``R`` relations in exactly ``G`` groups;
    group ``g`` owns rows ``cu_batches[g]`` to ``cu_batches[g + 1]``, in
    input order. Groups are in :func:`_group_launch_order`: the kernels
    launch group ``g`` before group ``g + 1``, and the deterministic chains
    number writers by ``g``. Reading ``G`` synchronizes the device once per
    call, so ``range_merge=True`` cannot be captured in a CUDA graph; a
    precomputed :class:`RangeMergePlan` can.
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
    num_groups, num_rows = int(unique_count.item()), sorted_k.shape[0]
    # Before the launch-order gather the groups are in outer-start order.
    merged_q, cu_batches = merged_q[:num_groups], cu_batches[: num_groups + 1]
    if num_rows == 0:
        return RangeMergePlan(merged_q, sorted_k, sorted_mask, cu_batches, merged_q)

    cu = cu_batches.to(torch.int64)
    row_count = cu[1:] - cu[:-1]
    order = _group_launch_order(sorted_k, row_count)
    new_row_count = row_count[order]
    new_cu_batches = torch.cat([cu.new_zeros(1), new_row_count.cumsum(0)])
    # Every row belongs to exactly one group, so the counts sum to num_rows
    # and output_size keeps repeat_interleave free of a host sync.
    group = torch.arange(num_groups, dtype=torch.int64, device=cu.device)
    row_group = torch.repeat_interleave(group, new_row_count, output_size=num_rows)
    row = torch.arange(num_rows, dtype=torch.int64, device=cu.device)
    src_row = cu[:-1][order][row_group] + (row - new_cu_batches[row_group])
    return RangeMergePlan(
        merged_outer_ranges=merged_q[order],
        sorted_inner_ranges=sorted_k[src_row],
        sorted_mask_types=sorted_mask[src_row],
        cu_batches=new_cu_batches.to(cu_batches.dtype),
        merged_outer_ranges_by_start=merged_q,
    )


# Ordering estimate of the inner-axis tile (fwd K, bwd Q); the kernel tile
# may differ by config.
_COST_TILE = 128


def _group_launch_order(
    sorted_inner: torch.Tensor, row_count: torch.Tensor
) -> torch.Tensor:
    """Permutation of the merged groups by the bit length of their work-tile
    length (the sum over pairs of ``ceil(inner width / _COST_TILE)``),
    descending. Groups of one class keep outer-start order, so neighboring
    groups still share inner ranges in L2. Masks are not counted. No host
    sync.
    """
    pair_tiles = torch.div(
        sorted_inner[:, 1] - sorted_inner[:, 0] + (_COST_TILE - 1),
        _COST_TILE,
        rounding_mode="floor",
    ).to(torch.int64)
    group_of_row = torch.repeat_interleave(
        torch.arange(row_count.shape[0], device=row_count.device),
        row_count,
        output_size=sorted_inner.shape[0],
    )
    tile_length = torch.zeros_like(row_count).index_add_(0, group_of_row, pair_tiles)
    # frexp's exponent of a non-negative integer is its bit length, exact in
    # float64 for every tile count an int32 range table can produce.
    _, length_class = torch.frexp(tile_length.to(torch.float64))
    return torch.argsort(length_class, descending=True, stable=True)
