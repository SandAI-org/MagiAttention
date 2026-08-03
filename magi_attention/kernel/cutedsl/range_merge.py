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

"""Host-side RangeMerge preprocessing for the CuTeDSL SM100 forward.

Same contract as the C++ ``merge_ranges``: relations whose Q intervals are
byte-identical collapse into one merged group; the kernel walks each
group's K ranges inside a single work tile, so the softmax runs once and
the output needs no atomic reduction.

Sync-free by design: outputs stay on device and keep the original row
count R. Groups are padded (``merged_q_ranges`` carries ``[0, 0]`` rows
past the unique count), so the launch grid stays at R and padded groups
decode into zero blocks, exactly like C++ pads with empty ranges.
"""

from dataclasses import dataclass

import torch

__all__ = [
    "merge_qk_ranges",
    "RangeMergePlan",
    "plan_range_merge",
    "plan_range_merge_bwd",
]


@dataclass(frozen=True)
class RangeMergePlan:
    """Precomputed RangeMerge metadata for reuse across calls.

    Build once with :func:`plan_range_merge` when the ranges are stable
    (typical training loops) and pass as ``range_merge=plan`` — the per-call
    sort/dedup work and its CPU submit cost disappear from the hot path.
    """

    merged_outer_ranges: torch.Tensor  # [R, 2], rows past unique_count are [0, 0]
    sorted_outer_ranges: torch.Tensor  # [R, 2] pair list, group-contiguous
    sorted_inner_ranges: torch.Tensor  # [R, 2] pair list, group-contiguous
    sorted_mask_types: torch.Tensor  # [R]
    cu_batches: torch.Tensor  # [R + 1] CSR over the sorted pair list


def plan_range_merge(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | int | None = None,
) -> RangeMergePlan:
    """One-shot RangeMerge preprocessing (same caller contract as inline
    ``range_merge=True``: the merged Q intervals must be pairwise disjoint)."""
    num_ranges = q_ranges.shape[0]
    if mask_types is None:
        mask_types = torch.zeros(
            num_ranges, dtype=torch.int32, device=q_ranges.device
        )
    elif isinstance(mask_types, int):
        mask_types = torch.full(
            (num_ranges,), mask_types, dtype=torch.int32, device=q_ranges.device
        )
    merged, sq, sk, sm, cu = merge_qk_ranges(q_ranges, k_ranges, mask_types)
    return RangeMergePlan(merged, sq, sk, sm, cu)


def plan_range_merge_bwd(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | int | None = None,
) -> RangeMergePlan:
    """Backward K-merge preprocessing: outer = K interval.

    Groups relations that share one K interval so a single backward CTA per
    K tile walks all their Q ranges, accumulating dK/dV in TMEM across pairs
    before one direct store (mirrors C++ ``merge_ranges(k_ranges, q_ranges)``).
    Caller contract: the merged K intervals are pairwise disjoint, which is
    what re-legalizes ``disable_bwd_dkv_atomic_reduction`` under shared K.
    """
    num_ranges = q_ranges.shape[0]
    if mask_types is None:
        mask_types = torch.zeros(
            num_ranges, dtype=torch.int32, device=q_ranges.device
        )
    elif isinstance(mask_types, int):
        mask_types = torch.full(
            (num_ranges,), mask_types, dtype=torch.int32, device=q_ranges.device
        )
    merged_k, sk, sq, sm, cu = merge_qk_ranges(k_ranges, q_ranges, mask_types)
    return RangeMergePlan(merged_k, sk, sq, sm, cu)


def merge_qk_ranges(
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor
]:
    """Group relations by identical Q interval.

    Args:
        q_ranges/k_ranges: ``[R, 2]`` int32 cuda tensors (paired rows).
        mask_types: optional ``[R]`` int32 cuda tensor (per-pair mask).

    Returns:
        merged_q_ranges: ``[R, 2]`` — row g is group g's Q interval for
            ``g < unique_count``, ``[0, 0]`` past it.
        sorted_q_ranges / sorted_k_ranges: the pair lists reordered so each
            group's pairs are contiguous (Q-major stable sort).
        sorted_mask_types: reordered mask types, or ``None`` if not given.
        cu_batches: ``[R + 1]`` int32 — group g owns sorted pairs
            ``[cu_batches[g], cu_batches[g + 1])``; padded groups are empty.

    The caller certifies (never read back — validating costs a sync) that
    the merged Q intervals are pairwise disjoint whenever the result feeds
    the non-atomic forward path.
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
    group_idx = torch.cumsum(group_head, dim=0) - 1  # [R], 0-based group id

    # Every pair writes its group slot with the (identical) group Q interval;
    # duplicate-index writes are idempotent here.  No boolean-mask indexing:
    # that would read the nonzero count back and sync the device.
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
