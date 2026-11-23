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
    """Precomputed merge tables for RangeMerge."""

    merged_outer_ranges: torch.Tensor  # [R, 2], pad [0, 0]
    sorted_inner_ranges: torch.Tensor  # [R, 2], group-contiguous
    sorted_mask_types: torch.Tensor  # [R]
    cu_batches: torch.Tensor  # [R + 1] CSR
    merged_outer_ranges_by_start: torch.Tensor  # [R, 2], start ascending + INT32_MAX tail
    unique_count: torch.Tensor | None = None  # [1] int32 active group count G
    row_group: torch.Tensor | None = None  # [R] int32 launch-order group per row
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
    assert is_magi_attn_ext_installed, "magi_attn_ext must be installed for RangeMerge."
    assert q_ranges.shape == k_ranges.shape and q_ranges.shape[1] == 2

    num_rows = q_ranges.shape[0]
    if num_rows == 0:
        return RangeMergePlan(
            merged_outer_ranges=q_ranges,
            sorted_inner_ranges=k_ranges,
            sorted_mask_types=mask_types,
            cu_batches=q_ranges.new_zeros(1),
            merged_outer_ranges_by_start=q_ranges,
            unique_count=q_ranges.new_zeros(1),
            row_group=q_ranges.new_empty(0, dtype=torch.int32),
        )

    pair_key = (q_ranges[:, 0].to(torch.int64) << 32) + (
        q_ranges[:, 1].to(torch.int64) - torch.iinfo(torch.int32).min
    )
    sorted_idx = torch.argsort(pair_key, stable=True)
    sorted_q = q_ranges[sorted_idx]
    sorted_k = k_ranges[sorted_idx]
    sorted_mask = mask_types[sorted_idx]
    merged_q, cu_batches, unique_count = magi_attn_ext.unique_consecutive_pairs(
        sorted_q
    )
    unique_count = unique_count.reshape(1)

    cu = cu_batches.to(torch.int64)
    row_count = cu[1:] - cu[:-1]
    order = _group_launch_order(sorted_k, row_count, unique_count)
    new_row_count = row_count[order]
    new_cu_batches = torch.cat([cu.new_zeros(1), new_row_count.cumsum(0)])
    group = torch.arange(num_rows, dtype=torch.int64, device=cu.device)
    row_group = torch.repeat_interleave(group, new_row_count, output_size=num_rows)
    row = torch.arange(num_rows, dtype=torch.int64, device=cu.device)
    src_row = cu[:-1][order][row_group] + (row - new_cu_batches[row_group])

    # Inactive and empty groups sort last as INT32_MAX for the hole cleanup's binary search.
    inactive_or_empty = (group >= unique_count.to(torch.int64)) | (
        merged_q[:, 0] == merged_q[:, 1]
    )
    by_start = torch.where(
        inactive_or_empty.unsqueeze(1),
        torch.iinfo(torch.int32).max,
        merged_q,
    )
    by_start = by_start[torch.argsort(by_start[:, 0], stable=True)]

    return RangeMergePlan(
        merged_outer_ranges=merged_q[order],
        sorted_inner_ranges=sorted_k[src_row],
        sorted_mask_types=sorted_mask[src_row],
        cu_batches=new_cu_batches.to(cu_batches.dtype),
        merged_outer_ranges_by_start=by_start,
        unique_count=unique_count,
        row_group=row_group.to(torch.int32),
    )


_COST_TILE = 128


def _group_launch_order(
    sorted_inner: torch.Tensor,
    row_count: torch.Tensor,
    unique_count: torch.Tensor,
) -> torch.Tensor:
    pair_tiles = torch.div(
        sorted_inner[:, 1] - sorted_inner[:, 0] + (_COST_TILE - 1),
        _COST_TILE,
        rounding_mode="floor",
    ).to(torch.int64)
    group_idx = torch.arange(row_count.shape[0], device=row_count.device)
    group_of_row = torch.repeat_interleave(
        group_idx,
        row_count,
        output_size=sorted_inner.shape[0],
    )
    tile_length = torch.zeros_like(row_count).index_add_(0, group_of_row, pair_tiles)
    # frexp's exponent is the bit length of the tile length.
    _, length_class = torch.frexp(tile_length.to(torch.float64))
    length_class = torch.where(
        group_idx < unique_count.to(torch.int64),
        length_class,
        -1,
    )
    return torch.argsort(length_class, descending=True, stable=True)
