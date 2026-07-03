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

"""SM90-specific range info: reads q_ranges/k_ranges [N,2] directly instead of cu_seqlens.

Drop-in replacement for SeqlenInfoQK with identical field names and methods,
so downstream code (BlockInfo, AttentionMask, offset_batch_Q/K, score_mod)
requires zero changes.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from .seqlen_info import SeqlenInfoQK


def create_seqlen_info_from_ranges(
    batch_idx: Int32,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    tile_m: cutlass.Constexpr[int] = 128,
    tile_n: cutlass.Constexpr[int] = 128,
    mCuTotalMBlocks: Optional[cute.Tensor] = None,
    mCuBlockIdxOffsets: Optional[cute.Tensor] = None,
) -> SeqlenInfoQK:
    """Create a SeqlenInfoQK from [N,2] q/k range tensors instead of cu_seqlens.

    The returned object is a standard SeqlenInfoQK with has_cu_seqlens=True,
    so all downstream consumers (BlockInfo, AttentionMask, offset_batch_Q/K,
    TMA ragged tensors) behave identically to the cu_seqlens path.

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
        0 if const_expr(mCuTotalMBlocks is None) else mCuTotalMBlocks[batch_idx]
    )
    num_n_blocks = (seqlen_k + tile_n - 1) // tile_n
    block_idx_offset = (
        mCuBlockIdxOffsets[batch_idx]
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
        has_cu_seqlens_q=True,
        has_cu_seqlens_k=True,
        has_seqused_q=False,
        has_seqused_k=False,
    )
