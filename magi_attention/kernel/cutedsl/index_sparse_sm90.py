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

"""SM90-specific IndexSparse: per-token K indices via cp.async scatter.

Ports C++ IndexSparseBlockMeta (block_meta.h L446-717) to CuTe-DSL.

FWD (LoopK): scheduler assigns one tile per (Q_token, kv_head).
Each tile's producer reads index_sparse_indices[unique_idx, :actual_topk]
and scatter-loads K/V rows via cp.async (SparseLoadCopyEngine).

BWD (LoopQ): scheduler assigns one tile per K block.
Producer reads inner_indices[k_block, :actual_inner_topk] for Q positions.

Reference: block_meta.h IndexSparseBlockMeta, mainloop_fwd_sm90 L697-764.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32


# ---------------------------------------------------------------------------
# IndexSparseProducerState: register state for FWD LoopK token-level path
# ---------------------------------------------------------------------------


@dataclass
class IndexSparseProducerState:
    """Producer state for IndexSparse FWD (LoopK, kbs=1).

    Maps to C++ IndexSparseBlockMeta fields for the token-level producer.
    group_token_ptr_offset: offset into mIndices row for this thread group.
    """

    inner_block_cur: Int32
    inner_block_max: Int32
    num_invalid_token: Int32
    actual_topk: Int32
    # Offset into mIndices row (unique_idx * max_topk + group_offset)
    row_base_offset: Int32
    group_offset: Int32
    head_local: Int32
    nhk: Int32
    unique_idx: Int32


@cute.jit
def compute_actual_topk(
    mIndices: cute.Tensor,
    row_start: Int32,
    max_topk: int,
) -> Int32:
    """Count non-negative entries in mIndices[row_start : row_start + max_topk].

    Scans from the end backwards (C++ block_meta.h L514-516).
    """
    actual = Int32(max_topk)
    i = Int32(max_topk - 1)
    # CuTe-DSL: no break, use while with compound condition
    while (i >= 0) & (mIndices[row_start + i] < 0):
        actual = actual - 1
        i = i - 1
    return actual


@cute.jit
def create_index_sparse_state(
    mIndices: cute.Tensor,
    unique_idx: Int32,
    max_topk: int,
    kBlockN: cutlass.Constexpr[int],
    nhk: Int32,
    thread_idx: Int32,
    num_producer_threads: cutlass.Constexpr[int],
    group_size: cutlass.Constexpr[int],
    num_rows_per_group: cutlass.Constexpr[int],
) -> IndexSparseProducerState:
    """Initialize IndexSparse producer state for FWD LoopK (kbs=1).

    Mirrors C++ IndexSparseBlockMeta constructor L505-568:
    - unique_idx = scheduler batch_idx = bidb * nhk + kv_head
    - head_local = unique_idx % nhk
    - row_ptr = mIndices + unique_idx * max_topk
    - group_token_ptr = row_ptr + group_offset (MinToMax)
    """
    row_base_offset = unique_idx * max_topk
    actual_topk = compute_actual_topk(mIndices, row_base_offset, max_topk)

    inner_block_max = (actual_topk + kBlockN - 1) // kBlockN
    num_invalid_token = inner_block_max * kBlockN - actual_topk

    head_local = unique_idx % nhk

    # Producer group staggering (C++ L556-563, MinToMax direction)
    idx_in_warpgroup = thread_idx % num_producer_threads
    group_idx = idx_in_warpgroup // group_size
    group_offset = group_idx * num_rows_per_group

    return IndexSparseProducerState(
        inner_block_cur=Int32(0),
        inner_block_max=inner_block_max,
        num_invalid_token=num_invalid_token,
        actual_topk=actual_topk,
        row_base_offset=row_base_offset,
        group_offset=group_offset,
        head_local=head_local,
        nhk=nhk,
        unique_idx=unique_idx,
    )


@cute.jit
def index_sparse_fill_token_indices(
    state: IndexSparseProducerState,
    mIndices: cute.Tensor,
    token_indices: cute.Tensor,
    kBlockN: cutlass.Constexpr[int],
    num_rows_per_group: cutlass.Constexpr[int],
):
    """Fill smem token_indices for current inner block (C++ fill_token_indices L579-585).

    LoopK kbs=1: reads sequential entries from mIndices row, converts to
    physical K row: physical = logical_id * nhk + head_local.

    token_indices: (num_rows_per_group,) rmem tensor, output.
    The caller (SparseLoadCopyEngine.load_scatter) uses these as scatter addresses.
    """
    ptr_offset = state.row_base_offset + state.group_offset + state.inner_block_cur * kBlockN
    for j in cutlass.range(num_rows_per_group, unroll=1):
        idx = mIndices[ptr_offset + Int32(j)]
        # C++ L584: group_rows[j] = (id >= 0) ? id * nhk + head_local : 0
        if idx >= 0:
            token_indices[Int32(j)] = idx * state.nhk + state.head_local
        else:
            token_indices[Int32(j)] = Int32(-1)


@cute.jit
def index_sparse_prefetch(state: IndexSparseProducerState):
    """Advance to next inner block (C++ prefetch L675-686, MinToMax).

    Token-level LoopK: just increment inner_block_cur.
    group_token_ptr advance is handled implicitly via inner_block_cur * kBlockN.
    """
    state.inner_block_cur = state.inner_block_cur + 1
