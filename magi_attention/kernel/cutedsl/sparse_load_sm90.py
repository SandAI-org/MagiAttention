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

"""SM90-specific SparseLoad: token-walking scatter loads via cp.async.

Ports Cutlass C++ SparseLoadBlockMeta (block_meta.h) to CuTe-DSL.

Outer loop = Q tiles (m_block); inner loop walks K tokens across merged
k_ranges in kBlockN-wide windows using cp.async per-row scatter.

Reference: paged_kv.py for CuTe-DSL cp.async scatter pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Type

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from . import cutedsl_utils

# ---------------------------------------------------------------------------
# SparseLoadProducerState: per-thread-group register state for token walk
# ---------------------------------------------------------------------------


@dataclass
class SparseLoadProducerState:
    """Producer-side state for sparse load token walk (MinToMax direction).

    Each producer thread group (GroupSize=8 threads) maintains NumRowsPerGroup
    register-based cursors. The cursors walk through concatenated k_ranges
    [bidb, end_batches) in token order.
    """

    cur_k_range_indices: cute.Tensor
    cur_k_range_inner_indices: cute.Tensor
    token_indices: cute.Tensor
    prev_token_indices: cute.Tensor

    inner_block_cur: Int32
    inner_block_max: Int32
    num_invalid_token: Int32

    bidb: Int32
    end_batches: Int32
    is_equal_k_range_size: bool
    k_range_size: Int32


@cute.jit
def _compute_total_k_tokens(
    mKRanges: cute.Tensor,
    bidb: Int32,
    end_batches: Int32,
    is_equal_k_range_size: bool,
) -> Int32:
    """Compute total K tokens in the merged [bidb, end_batches) slice."""
    if const_expr(is_equal_k_range_size):
        k_start = mKRanges[bidb, 0]
        k_end = mKRanges[bidb, 1]
        total = (end_batches - bidb) * (k_end - k_start)
    else:
        total = Int32(0)
        i = bidb
        while i < end_batches:
            total += mKRanges[i, 1] - mKRanges[i, 0]
            i += 1
    return total


@cute.jit
def _clamp_to_boundary_min_to_max(
    cur_k_range_indices: cute.Tensor,
    cur_k_range_inner_indices: cute.Tensor,
    mKRanges: cute.Tensor,
    idx: Int32,
    end_batches: Int32,
):
    """Clamp cursor at idx if it overflowed past end_batches (MinToMax)."""
    if cur_k_range_indices[idx] >= end_batches:
        cur_k_range_indices[idx] = end_batches - 1
        last_start = mKRanges[end_batches - 1, 0]
        last_end = mKRanges[end_batches - 1, 1]
        cur_k_range_inner_indices[idx] = last_end - last_start - 1


@cute.jit
def _step_one_token_min_to_max(
    cur_k_range_indices: cute.Tensor,
    cur_k_range_inner_indices: cute.Tensor,
    mKRanges: cute.Tensor,
    dst: Int32,
    src: Int32,
    end_batches: Int32,
):
    """Step one token forward (MinToMax): advance src→dst."""
    r_start = mKRanges[cur_k_range_indices[src], 0]
    r_end = mKRanges[cur_k_range_indices[src], 1]
    r_len = r_end - r_start
    if cur_k_range_inner_indices[src] + 1 < r_len:
        cur_k_range_indices[dst] = cur_k_range_indices[src]
        cur_k_range_inner_indices[dst] = cur_k_range_inner_indices[src] + 1
    else:
        cur_k_range_indices[dst] = cur_k_range_indices[src] + 1
        cur_k_range_inner_indices[dst] = 0
    _clamp_to_boundary_min_to_max(
        cur_k_range_indices, cur_k_range_inner_indices, mKRanges, dst, end_batches
    )


@cute.jit
def _advance_anchor_equal(
    cur_k_range_indices: cute.Tensor,
    cur_k_range_inner_indices: cute.Tensor,
    anchor: Int32,
    num_steps: Int32,
    k_range_size: Int32,
):
    """Advance anchor cursor by num_steps tokens (equal range O(1) path, MinToMax)."""
    n_k_ranges = num_steps // k_range_size
    n_k_range_inner = num_steps % k_range_size

    remaining = k_range_size - 1 - cur_k_range_inner_indices[anchor]
    if remaining >= n_k_range_inner:
        cur_k_range_indices[anchor] = cur_k_range_indices[anchor] + n_k_ranges
        cur_k_range_inner_indices[anchor] = (
            cur_k_range_inner_indices[anchor] + n_k_range_inner
        )
    else:
        cur_k_range_indices[anchor] = cur_k_range_indices[anchor] + n_k_ranges + 1
        cur_k_range_inner_indices[anchor] = n_k_range_inner - remaining - 1


@cute.jit
def _advance_anchor_unequal(
    cur_k_range_indices: cute.Tensor,
    cur_k_range_inner_indices: cute.Tensor,
    mKRanges: cute.Tensor,
    anchor: Int32,
    num_steps: Int32,
    end_batches: Int32,
):
    """Advance anchor cursor by num_steps tokens (unequal range slow path, MinToMax)."""
    cnt = Int32(0)
    while cnt < num_steps:
        if cur_k_range_indices[anchor] >= end_batches:
            break
        rest = num_steps - cnt
        r_start = mKRanges[cur_k_range_indices[anchor], 0]
        r_end = mKRanges[cur_k_range_indices[anchor], 1]
        remaining = r_end - r_start - 1 - cur_k_range_inner_indices[anchor]
        if remaining >= rest:
            cur_k_range_inner_indices[anchor] = cur_k_range_inner_indices[anchor] + rest
            break
        cnt = cnt + remaining + 1
        cur_k_range_indices[anchor] = cur_k_range_indices[anchor] + 1
        cur_k_range_inner_indices[anchor] = 0


@cute.jit
def advance_and_fill(
    cur_k_range_indices: cute.Tensor,
    cur_k_range_inner_indices: cute.Tensor,
    token_indices: cute.Tensor,
    mKRanges: cute.Tensor,
    num_steps: Int32,
    bidb: Int32,
    end_batches: Int32,
    is_equal_k_range_size: bool,
    k_range_size: Int32,
    num_rows_per_group: cutlass.Constexpr[int],
):
    """Advance anchor by num_steps, then fill all row positions (MinToMax).

    Anchor is index 0 (MinToMax); fill rows 1..NumRowsPerGroup-1 via step_one_token.
    """
    anchor = Int32(0)

    if const_expr(is_equal_k_range_size):
        _advance_anchor_equal(
            cur_k_range_indices,
            cur_k_range_inner_indices,
            anchor,
            num_steps,
            k_range_size,
        )
    else:
        _advance_anchor_unequal(
            cur_k_range_indices,
            cur_k_range_inner_indices,
            mKRanges,
            anchor,
            num_steps,
            end_batches,
        )

    _clamp_to_boundary_min_to_max(
        cur_k_range_indices,
        cur_k_range_inner_indices,
        mKRanges,
        anchor,
        end_batches,
    )

    r_start_a = mKRanges[cur_k_range_indices[anchor], 0]
    token_indices[anchor] = r_start_a + cur_k_range_inner_indices[anchor]

    for j in cutlass.range(1, num_rows_per_group, unroll=True):
        _step_one_token_min_to_max(
            cur_k_range_indices,
            cur_k_range_inner_indices,
            mKRanges,
            Int32(j),
            Int32(j - 1),
            end_batches,
        )
        r_start_j = mKRanges[cur_k_range_indices[Int32(j)], 0]
        token_indices[Int32(j)] = r_start_j + cur_k_range_inner_indices[Int32(j)]


@cute.jit
def create_sparse_load_producer_state(
    thread_idx: Int32,
    mKRanges: cute.Tensor,
    bidb: Int32,
    end_batches: Int32,
    is_equal_k_range_size: bool,
    kBlockN: cutlass.Constexpr[int],
    GroupSize: cutlass.Constexpr[int],
    NumRowsPerGroup: cutlass.Constexpr[int],
) -> SparseLoadProducerState:
    """Initialize producer state for sparse load (MinToMax direction)."""
    total_k_tokens = _compute_total_k_tokens(
        mKRanges, bidb, end_batches, is_equal_k_range_size
    )
    inner_block_max = (total_k_tokens + kBlockN - 1) // kBlockN
    num_invalid_token = inner_block_max * kBlockN - total_k_tokens
    inner_block_cur = Int32(0)

    cur_k_range_indices = cute.make_rmem_tensor((NumRowsPerGroup,), Int32)
    cur_k_range_inner_indices = cute.make_rmem_tensor((NumRowsPerGroup,), Int32)
    token_indices = cute.make_rmem_tensor((NumRowsPerGroup,), Int32)
    prev_token_indices = cute.make_rmem_tensor((NumRowsPerGroup,), Int32)
    prev_token_indices[NumRowsPerGroup - 1] = -1

    k_range_size = Int32(0)
    if const_expr(is_equal_k_range_size):
        k_range_size = mKRanges[bidb, 1] - mKRanges[bidb, 0]

    cur_k_range_indices[0] = bidb
    cur_k_range_inner_indices[0] = 0

    idx_in_warpgroup = thread_idx % 128
    group_idx = idx_in_warpgroup // GroupSize
    num_steps = group_idx * NumRowsPerGroup

    if inner_block_max > 0:
        advance_and_fill(
            cur_k_range_indices,
            cur_k_range_inner_indices,
            token_indices,
            mKRanges,
            num_steps,
            bidb,
            end_batches,
            is_equal_k_range_size,
            k_range_size,
            NumRowsPerGroup,
        )

    return SparseLoadProducerState(
        cur_k_range_indices=cur_k_range_indices,
        cur_k_range_inner_indices=cur_k_range_inner_indices,
        token_indices=token_indices,
        prev_token_indices=prev_token_indices,
        inner_block_cur=inner_block_cur,
        inner_block_max=inner_block_max,
        num_invalid_token=num_invalid_token,
        bidb=bidb,
        end_batches=end_batches,
        is_equal_k_range_size=is_equal_k_range_size,
        k_range_size=k_range_size,
    )


@cute.jit
def prefetch_sparse_load(
    state: SparseLoadProducerState,
    mKRanges: cute.Tensor,
    kBlockN: cutlass.Constexpr[int],
    NumRowsPerGroup: cutlass.Constexpr[int],
):
    """Save prev_token_indices, advance inner_block_cur, advance_and_fill(kBlockN)."""
    for i in cutlass.range(NumRowsPerGroup, unroll=True):
        state.prev_token_indices[Int32(i)] = state.token_indices[Int32(i)]

    state.inner_block_cur = state.inner_block_cur + 1

    if state.inner_block_cur < state.inner_block_max:
        advance_and_fill(
            state.cur_k_range_indices,
            state.cur_k_range_inner_indices,
            state.token_indices,
            mKRanges,
            Int32(kBlockN),
            state.bidb,
            state.end_batches,
            state.is_equal_k_range_size,
            state.k_range_size,
            NumRowsPerGroup,
        )


# ---------------------------------------------------------------------------
# cp.async scatter load for K/V
# ---------------------------------------------------------------------------


@dataclass
class SparseLoadCopyEngine:
    """Manages cp.async scatter load geometry, mirrors paged_kv.py pattern."""

    gmem_tiled_copy_KV: cute.TiledCopy
    gmem_thr_copy_KV: cute.TiledCopy
    async_copy_elems: int
    gmem_threads_per_row: int
    num_rows_per_group: int

    @staticmethod
    def create(
        thread_idx: Int32,
        n_block_size: cutlass.Constexpr[int],
        head_dim: cutlass.Constexpr[int],
        num_producer_threads: cutlass.Constexpr[int],
        dtype: Type[cutlass.Numeric],
    ) -> SparseLoadCopyEngine:
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // dtype.width
        dtype_bytes = dtype.width // 8
        gmem_k_block_size = math.gcd(head_dim, 128 // dtype_bytes)
        gmem_threads_per_row = gmem_k_block_size // async_copy_elems
        group_size = gmem_threads_per_row
        num_groups = num_producer_threads // group_size
        num_rows_per_group = n_block_size // num_groups

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        thr_layout = cute.make_ordered_layout(
            (num_producer_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        val_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(
            atom_async_copy, thr_layout, val_layout
        )
        gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(thread_idx)

        return SparseLoadCopyEngine(
            gmem_tiled_copy_KV=gmem_tiled_copy_KV,
            gmem_thr_copy_KV=gmem_thr_copy_KV,
            async_copy_elems=async_copy_elems,
            gmem_threads_per_row=gmem_threads_per_row,
            num_rows_per_group=num_rows_per_group,
        )

    @cute.jit
    def load_scatter(
        self,
        token_indices: cute.Tensor,
        mX: cute.Tensor,
        sX: cute.Tensor,
        head_dim: cutlass.Constexpr[int],
        seqlen_k_limit: Int32,
    ):
        """Load kBlockN rows from scattered token positions into dense smem tile.

        Each thread group loads NumRowsPerGroup rows. Within each row,
        threads cooperatively load head_dim elements via 128-bit cp.async.

        Args:
            token_indices: rmem tensor [NumRowsPerGroup] with absolute K row indices
            mX: global K or V tensor with shape (seqlen, headdim)
            sX: smem tile for this pipeline stage, shape (kBlockN, headdim)
            head_dim: head dimension (compile-time)
            seqlen_k_limit: valid row limit for boundary check
        """
        sX_pi = cute.group_modes(sX, 0, 1)
        cX = cute.make_identity_tensor(
            (self.num_rows_per_group * (128 // self.gmem_threads_per_row), head_dim)
        )
        tXsX = self.gmem_thr_copy_KV.partition_D(sX_pi)
        tXcX = self.gmem_thr_copy_KV.partition_S(cX)
        tXc0X = self.gmem_thr_copy_KV.get_slice(0).partition_S(cX)

        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row_token = token_indices[m // self.gmem_threads_per_row]
            row_valid = (row_token >= 0) & (row_token < seqlen_k_limit)
            should_load = cute.make_fragment_like(tXsX[(0, None), m, 0], cute.Boolean)
            should_load.fill(row_valid)

            x_ptr_i64 = cutedsl_utils.shuffle_sync(
                cutedsl_utils.elem_pointer(mX, (row_token, 0)).toint(),
                m % self.gmem_threads_per_row,
                width=self.gmem_threads_per_row,
            )
            x_gmem_ptr = cute.make_ptr(
                mX.element_type,
                x_ptr_i64,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            mX_row = cute.make_tensor(x_gmem_ptr, cute.make_layout((head_dim,)))
            mX_row_tiled = cute.tiled_divide(mX_row, (self.async_copy_elems,))

            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, k][1] // self.async_copy_elems
                mX_row_ki = mX_row_tiled[None, ki]
                tXsX_k = tXsX[None, m, k]
                mX_row_ki_copy = cute.make_tensor(mX_row_ki.iterator, tXsX_k.layout)
                cute.copy(
                    self.gmem_tiled_copy_KV,
                    mX_row_ki_copy,
                    tXsX_k,
                    pred=should_load,
                )
