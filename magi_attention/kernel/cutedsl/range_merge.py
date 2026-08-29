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

# Copyright (c) 2026 MagiAttention Authors.
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
    sorted_outer_ranges: torch.Tensor  # [R, 2], group-contiguous
    sorted_inner_ranges: torch.Tensor  # [R, 2], group-contiguous
    sorted_mask_types: torch.Tensor  # [R]
    cu_batches: torch.Tensor  # [R + 1] CSR
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
    merged, sq, sk, sm, cu = merge_qk_ranges(q_ranges, k_ranges, mask_types)
    return RangeMergePlan(
        merged, sq, sk, sm, cu, bwd=plan_range_merge_bwd(q_ranges, k_ranges, mask_types)
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
    merged_k, sk, sq, sm, cu = merge_qk_ranges(k_ranges, q_ranges, mask_types)
    return RangeMergePlan(merged_k, sk, sq, sm, cu)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Group relations by identical outer interval.

    Returns ``(merged_outer, sorted_outer, sorted_inner, sorted_mask, cu_batches)``.
    """
    assert q_ranges.shape == k_ranges.shape and q_ranges.shape[1] == 2
    num_ranges = q_ranges.shape[0]
    device = q_ranges.device

    key = (q_ranges[:, 0].to(torch.int64) << 32) | q_ranges[:, 1].to(torch.int64)
    order = torch.argsort(key, stable=True)
    sorted_q = q_ranges[order].contiguous()
    sorted_k = k_ranges[order].contiguous()
    sorted_mask = mask_types[order].contiguous() if mask_types is not None else None

    group_head = torch.ones(num_ranges, dtype=torch.bool, device=device)
    if num_ranges > 1:
        group_head[1:] = (sorted_q[1:] != sorted_q[:-1]).any(dim=1)
    group_idx = torch.cumsum(group_head, dim=0) - 1

    # Pad-to-R via index_copy (no host sync).
    merged_q = torch.zeros_like(q_ranges)
    merged_q.index_copy_(0, group_idx, sorted_q)

    counts = torch.bincount(group_idx, minlength=num_ranges)
    cu_batches = torch.zeros(num_ranges + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=cu_batches[1:])

    return (
        merged_q,
        sorted_q,
        sorted_k,
        sorted_mask,
        cu_batches.to(torch.int32),
    )
