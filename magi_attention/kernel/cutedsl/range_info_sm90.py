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

"""SM90-specific range info and runtime mask-type block skipping.

- ``create_seqlen_info_from_ranges``: reads q_ranges/k_ranges [N,2] ranges
  and returns a standard SeqlenInfoQK.
- ``get_n_block_min_max_runtime`` / ``get_m_block_min_max_runtime``:
  per-range block-skipping using a runtime ``mask_type`` (full=0, causal=1,
  inv_causal=2, bi_causal=3).
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from .seqlen_info import SeqlenInfoQK


@cute.jit
def read_attn_type_map(mAttnTypeMap: cute.Tensor, batch_idx: Int32) -> Int32:
    """Read mask type for a given batch index from the attn_type_map tensor.

    This exists as a standalone function so that it can be captured via
    ``functools.partial`` before a while-loop boundary.  CuTe-DSL's SCF
    framework does not carry ``cute.Tensor`` MLIR values across while-loop
    iterations, but it does carry Python objects (like ``partial`` closures).
    """
    return mAttnTypeMap[batch_idx]


def create_seqlen_info_from_ranges(
    batch_idx: Int32,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    tile_m: cutlass.Constexpr[int] = 128,
    tile_n: cutlass.Constexpr[int] = 128,
    mCuTotalMBlocks: Optional[cute.Tensor] = None,
    mCuBlockIdxOffsets: Optional[cute.Tensor] = None,
) -> SeqlenInfoQK:
    """Create a SeqlenInfoQK from [N,2] q/k range tensors.

    The returned object is a standard SeqlenInfoQK with has_ranges=True,
    so all downstream consumers (BlockInfo, AttentionMask, offset_batch_Q/K,
    TMA ragged tensors) behave identically to the ranges path.

    Args:
        batch_idx: Current batch (range) index.
        mQRanges: [N, 2] int32 tensor of [start, end) Q ranges.
        mKRanges: [N, 2] int32 tensor of [start, end) K ranges.
        tile_m: Q tile size (constexpr).
        tile_n: K tile size (constexpr).
        mCuTotalMBlocks: Optional block-sparse cumulative m-block counts.
        mCuBlockIdxOffsets: Optional block-sparse block index offsets.
    """
    offset_q = mQRanges[batch_idx, 0]
    offset_k = mKRanges[batch_idx, 0]

    seqlen_q = mQRanges[batch_idx, 1] - offset_q
    seqlen_k = mKRanges[batch_idx, 1] - offset_k

    padded_offset_q = cute.assume(
        (offset_q + batch_idx * tile_m) // tile_m * tile_m, divby=tile_m
    )
    padded_offset_k = cute.assume(
        (offset_k + batch_idx * tile_n) // tile_n * tile_n, divby=tile_n
    )

    m_block_offset = (
        0 if const_expr(mCuTotalMBlocks is None) else mCuTotalMBlocks[batch_idx]  # type: ignore[index]
    )
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n
    block_idx_offset = (
        mCuBlockIdxOffsets[batch_idx]  # type: ignore[index]
        if const_expr(mCuBlockIdxOffsets is not None)
        else m_block_offset * num_n_blocks
    )

    return SeqlenInfoQK(
        offset_q,
        offset_k,
        padded_offset_q,
        padded_offset_k,
        seqlen_q,
        seqlen_k,
        m_block_offset,
        block_idx_offset,
        num_n_blocks,
        has_ranges_q=True,
        has_ranges_k=True,
        has_seqused_q=False,
        has_seqused_k=False,
    )


# ---------------------------------------------------------------------------
# Runtime mask-type block skipping (FWD: iterate N blocks for fixed M block)
# ---------------------------------------------------------------------------


@cute.jit
def get_n_block_min_max_runtime(
    seqlen_info: SeqlenInfoQK,
    m_block: Int32,
    mask_type: Int32,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1,
) -> Tuple[Int32, Int32]:
    """Compute n_block range with runtime mask_type dispatch.

    mask_type: 0=full, 1=causal, 2=inv_causal, 3=bi_causal.
    """
    n_block_max = cute.ceil_div(seqlen_info.seqlen_k, tile_n)
    n_block_min = Int32(0)

    has_causal = (mask_type == 1) | (mask_type == 3)
    if has_causal:
        m_idx_max = (m_block + 1) * tile_m
        if const_expr(qhead_per_kvhead_packgqa > 1):
            m_idx_max = cute.ceil_div(m_idx_max, qhead_per_kvhead_packgqa)
        n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_block_max = cutlass.min(n_block_max, cute.ceil_div(n_idx, tile_n))

    has_inv = (mask_type == 2) | (mask_type == 3)
    if has_inv:
        m_idx_min = m_block * tile_m
        if const_expr(qhead_per_kvhead_packgqa > 1):
            m_idx_min = m_idx_min // qhead_per_kvhead_packgqa
        n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
        n_block_min = cutlass.max(n_idx // tile_n, Int32(0))

    return n_block_min, n_block_max


# ---------------------------------------------------------------------------
# Runtime mask-type block skipping (BWD: iterate M blocks for fixed N block)
# ---------------------------------------------------------------------------


@cute.jit
def get_m_block_min_max_runtime(
    seqlen_info: SeqlenInfoQK,
    n_block: Int32,
    mask_type: Int32,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
) -> Tuple[Int32, Int32]:
    """Compute m_block range with runtime mask_type dispatch (BWD)."""
    m_block_max = cute.ceil_div(seqlen_info.seqlen_q, tile_m)
    m_block_min = Int32(0)

    has_causal = (mask_type == 1) | (mask_type == 3)
    if has_causal:
        n_idx_min = n_block * tile_n
        m_idx = n_idx_min + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_block_min = cutlass.max(m_block_min, m_idx // tile_m)

    has_inv = (mask_type == 2) | (mask_type == 3)
    if has_inv:
        n_idx_max = (n_block + 1) * tile_n
        m_idx = n_idx_max + seqlen_info.seqlen_q - seqlen_info.seqlen_k
        m_block_max = cutlass.min(m_block_max, cute.ceil_div(m_idx, tile_m))

    return m_block_min, m_block_max
