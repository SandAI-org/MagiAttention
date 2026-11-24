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

"""Index the writers of shared output and gradient blocks in deterministic range attention."""

from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint8, const_expr

# isort: split
from quack.compile_utils import make_fake_tensor as fake_tensor

from .block_info import BlockInfo
from .cache_utils import get_jit_cache
from .cutedsl_utils import elem_pointer
from .ffa_utils import MT_MAP

_INDEX_THREADS = 128
_INDEX_WARPS = _INDEX_THREADS // 32


@dataclass(frozen=True)
class BwdWriteIndexSpec:
    """Compile-time specialization of the backward write index kernels; it
    must equal the tiling and mask configuration of the attention kernel.

    ``mask_type`` is the static mask of the attention kernel, None for a
    per-range mask.
    """

    tile_m: int
    cluster_tile_k: int
    mask_type: Optional[int]
    range_merge: bool

    @property
    def use_per_range_mask(self) -> bool:
        return self.mask_type is None

    @property
    def block_info(self) -> BlockInfo:
        return BlockInfo(
            self.tile_m,
            self.cluster_tile_k,
            self.mask_type is None or self.mask_type == MT_MAP.causal,
        )


def write_index_shape(num_ranges: int, max_index_rows: int) -> tuple[int, int]:
    """Return ``(num_index_rows, index_stride)`` with
    ``num_index_rows <= max_index_rows``.

    Index row ``c`` holds, per physical block, the highest writer among ranges
    ``[0, c * index_stride)``, 0 if none. A chunk-max kernel reduces the
    writers of chunk ``c`` (``index_stride`` ranges) into row ``c``; an
    exclusive prefix max along the rows then yields the index. The
    ``lookup_*_predecessor`` functions read row ``r // index_stride`` and scan
    the earlier ranges of that chunk themselves, so both sides must number
    writers identically.
    """
    assert max_index_rows > 0
    index_stride = max(1, (num_ranges + max_index_rows - 1) // max_index_rows)
    num_index_rows = max(1, (num_ranges + index_stride - 1) // index_stride)
    return num_index_rows, index_stride


@cute.jit
def tile_blocks(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    row = range_start + tile_idx * block_size
    row_end = cutlass.min(row + block_size, range_end)
    return row // block_size, (row_end - 1) // block_size


@cute.jit
def block_tiles(range_start: Int32, block: Int32, block_size: cutlass.Constexpr[int]):
    """Range-local tiles ``[first, last]`` that overlap physical ``block``;
    the caller bounds them to the tiles of the range."""
    first_block = range_start // block_size
    last = block - first_block
    first = last
    if range_start % block_size != 0 and block > first_block:
        first = last - 1
    return first, last


@cute.jit
def dq_pair_n_bounds(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    m_tile: Int32,
    attn_type: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
):
    if const_expr(use_per_range_mask):
        return block_info.get_n_block_min_max_per_range_seqlens(
            seqlen_q, seqlen_k, m_tile, attn_type
        )
    return block_info.get_n_block_min_max_seqlens(seqlen_q, seqlen_k, m_tile)


@cute.jit
def dq_pair_k_tile_end_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    q_start: Int32,
    attn_type: Int32,
    block: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    """1 + the largest K tile for which a pair writes dQ on physical Q
    ``block``; 0 if none. ``block`` must overlap the Q range of the pair."""
    k_tile_end = Int32(0)
    if seqlen_q > 0 and seqlen_k > 0:
        if const_expr(
            not use_per_range_mask
            and not block_info.is_causal
            and not block_info.is_local
        ):
            k_tile_end = cute.ceil_div(seqlen_k, block_info.tile_n)
        elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
            k_tile_end = cute.ceil_div(seqlen_k, block_info.tile_n)
        else:
            mt, mt_last = block_tiles(q_start, block, block_info.tile_m)
            while mt <= mt_last:
                if mt * block_info.tile_m < seqlen_q:
                    n_lo, n_hi = dq_pair_n_bounds(
                        block_info,
                        seqlen_q,
                        seqlen_k,
                        mt,
                        attn_type,
                        use_per_range_mask,
                    )
                    if n_hi > n_lo:
                        k_tile_end = cutlass.max(k_tile_end, n_hi)
                mt += 1
    return k_tile_end


@cute.jit
def dq_pair_prev_k_tile_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    q_start: Int32,
    attn_type: Int32,
    block: Int32,
    n_limit: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    """Largest K tile ``< n_limit`` for which a pair writes dQ on physical Q
    ``block``; -1 if none. ``block`` must overlap the Q range of the pair."""
    prev_n = Int32(-1)
    if const_expr(
        not use_per_range_mask and not block_info.is_causal and not block_info.is_local
    ):
        prev_n = n_limit - 1
    elif const_expr(use_per_range_mask) and attn_type == MT_MAP.full:
        prev_n = n_limit - 1
    else:
        mt, mt_last = block_tiles(q_start, block, block_info.tile_m)
        while mt <= mt_last:
            if mt * block_info.tile_m < seqlen_q:
                l_m, r_m = dq_pair_n_bounds(
                    block_info,
                    seqlen_q,
                    seqlen_k,
                    mt,
                    attn_type,
                    use_per_range_mask,
                )
                if r_m > l_m and l_m < n_limit:
                    prev_n = cutlass.max(prev_n, cutlass.min(n_limit, r_m) - 1)
            mt += 1
    return prev_n


@cute.jit
def dkv_pair_k_tile_is_active(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_tile: Int32,
    attn_type: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    """Whether a pair has an unmasked Q tile on its K tile ``k_tile``; a tile
    outside the K range is inactive."""
    active = cutlass.Boolean(False)
    if seqlen_q > 0 and k_tile >= 0 and k_tile * block_info.tile_n < seqlen_k:
        if const_expr(use_per_range_mask):
            if attn_type == MT_MAP.full:
                active = cutlass.Boolean(True)
            else:
                m_lo, m_hi = block_info.get_m_block_min_max_per_range_seqlens(
                    seqlen_q, seqlen_k, k_tile, attn_type
                )
                if m_lo < m_hi:
                    active = cutlass.Boolean(True)
        elif const_expr(not block_info.is_causal and not block_info.is_local):
            active = cutlass.Boolean(True)
        else:
            m_lo, m_hi = block_info.get_m_block_min_max_seqlens(
                seqlen_q, seqlen_k, k_tile
            )
            if m_lo < m_hi:
                active = cutlass.Boolean(True)
    return active


@cute.jit
def dkv_pair_is_active_on_block(
    block_info: BlockInfo,
    seqlen_q: Int32,
    seqlen_k: Int32,
    k_start: Int32,
    attn_type: Int32,
    block: Int32,
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    """Whether a pair has an active K tile that overlaps physical K ``block``."""
    first, last = block_tiles(k_start, block, block_info.tile_n)
    active = dkv_pair_k_tile_is_active(
        block_info, seqlen_q, seqlen_k, last, attn_type, use_per_range_mask
    )
    if not active and first != last:
        active = dkv_pair_k_tile_is_active(
            block_info, seqlen_q, seqlen_k, first, attn_type, use_per_range_mask
        )
    return active


@cute.jit
def dkv_k_tile_is_active(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    r: Int32,
    n_cta: Int32,
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> cutlass.Boolean:
    seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
    active = cutlass.Boolean(False)
    if const_expr(range_merge):
        assert (
            mMaskTypes is not None
            and mCuBatches is not None
            and mGroupAllKActive is not None
        )
        if n_cta >= 0 and n_cta * block_info.tile_n < seqlen_k:
            if mGroupAllKActive[r] != 0:
                active = cutlass.Boolean(True)
            else:
                row = Int32(mCuBatches[r])
                row_end = Int32(mCuBatches[r + 1])
                while row < row_end and not active:
                    active = dkv_pair_k_tile_is_active(
                        block_info,
                        mQRanges[row, 1] - mQRanges[row, 0],
                        seqlen_k,
                        n_cta,
                        Int32(mMaskTypes[row]),
                        True,
                    )
                    row += 1
    else:
        attn_r = Int32(MT_MAP.full)
        if const_expr(use_per_range_mask):
            assert mMaskTypes is not None
            attn_r = Int32(mMaskTypes[r])
        active = dkv_pair_k_tile_is_active(
            block_info,
            mQRanges[r, 1] - mQRanges[r, 0],
            seqlen_k,
            n_cta,
            attn_r,
            use_per_range_mask,
        )
    return active


@cute.jit
def _dkv_block_writer(
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    writer_stride: Int32,
    r: Int32,
    kb: Int32,
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    first, last = block_tiles(mKRanges[r, 0], kb, block_info.tile_n)
    active = dkv_k_tile_is_active(
        block_info,
        mQRanges,
        mKRanges,
        mMaskTypes,
        mCuBatches,
        mGroupAllKActive,
        r,
        last,
        range_merge,
        use_per_range_mask,
    )
    if not active and first != last:
        active = dkv_k_tile_is_active(
            block_info,
            mQRanges,
            mKRanges,
            mMaskTypes,
            mCuBatches,
            mGroupAllKActive,
            r,
            first,
            range_merge,
            use_per_range_mask,
        )
    writer = Int32(0)
    if active:
        writer = (r + 1) * writer_stride
    return writer


@cute.jit
def lookup_fwd_predecessor(
    mWriteIndex: cute.Tensor,
    mQRanges: cute.Tensor,
    batch_idx: Int32,
    block: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    index_stride: Int32,
) -> Int32:
    """Highest earlier fwd writer (r + 1) touching physical ``block``; 0 if none."""
    c = batch_idx // index_stride
    pred = Int32(mWriteIndex[c, block])
    if index_stride > 1:
        r_lo = c * index_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            s = mQRanges[r, 0] * rows_per_token
            e = mQRanges[r, 1] * rows_per_token
            if e > s and s // block_size <= block and (e - 1) // block_size >= block:
                pred = r + 1
                found = cutlass.Boolean(True)
            r -= 1
    return pred


@cute.jit
def lookup_dkv_predecessor(
    mWriteIndex: cute.Tensor,
    index_stride: Int32,
    writer_stride: Int32,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    mGroupAllKActive: Optional[cute.Tensor],
    batch_idx: Int32,
    kb: Int32,
    block_rows: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    c = batch_idx // index_stride
    pred = Int32(mWriteIndex[c, kb])
    if index_stride > 1:
        r_lo = c * index_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            k_s = mKRanges[r, 0]
            k_e = mKRanges[r, 1]
            if k_e > k_s and k_s // block_rows <= kb and (k_e - 1) // block_rows >= kb:
                w = _dkv_block_writer(
                    block_info,
                    mQRanges,
                    mKRanges,
                    mMaskTypes,
                    mCuBatches,
                    mGroupAllKActive,
                    writer_stride,
                    r,
                    kb,
                    range_merge,
                    use_per_range_mask,
                )
                if w > 0:
                    pred = w
                    found = cutlass.Boolean(True)
            r -= 1
    return pred


@cute.jit
def lookup_dq_predecessor(
    mWriteIndex: cute.Tensor,
    index_stride: Int32,
    writer_stride: Int32,
    block_info: BlockInfo,
    mQRanges: cute.Tensor,
    mKRanges: cute.Tensor,
    mMaskTypes: Optional[cute.Tensor],
    mCuBatches: Optional[cute.Tensor],
    batch_idx: Int32,
    block: Int32,
    tile_m: cutlass.Constexpr[int],
    range_merge: cutlass.Constexpr[bool],
    use_per_range_mask: cutlass.Constexpr[bool],
) -> Int32:
    c = batch_idx // index_stride
    pred = Int32(mWriteIndex[c, block])
    if index_stride > 1:
        r_lo = c * index_stride
        r = batch_idx - 1
        found = cutlass.Boolean(False)
        while r >= r_lo and not found:
            seqlen_k = mKRanges[r, 1] - mKRanges[r, 0]
            if seqlen_k > 0:
                k_tile_end = Int32(0)
                if const_expr(range_merge):
                    assert mCuBatches is not None
                    row = Int32(mCuBatches[r])
                    row_end = Int32(mCuBatches[r + 1])
                else:
                    row = r
                    row_end = r + 1
                while row < row_end:
                    q_s = mQRanges[row, 0]
                    q_e = mQRanges[row, 1]
                    if (
                        q_e > q_s
                        and q_s // tile_m <= block
                        and (q_e - 1) // tile_m >= block
                    ):
                        attn_r = Int32(MT_MAP.full)
                        if const_expr(use_per_range_mask):
                            assert mMaskTypes is not None
                            attn_r = Int32(mMaskTypes[row])
                        k_tile_end = cutlass.max(
                            k_tile_end,
                            dq_pair_k_tile_end_on_block(
                                block_info,
                                q_e - q_s,
                                seqlen_k,
                                q_s,
                                attn_r,
                                block,
                                use_per_range_mask,
                            ),
                        )
                    row += 1
                if k_tile_end > 0:
                    pred = r * writer_stride + k_tile_end
                    found = cutlass.Boolean(True)
            r -= 1
    return pred


class _WriteIndexPrefixMax:
    """In-place exclusive prefix max along axis 0 of
    ``(num_index_rows, num_blocks)``."""

    @cute.jit
    def __call__(
        self,
        mWriteIndex: cute.Tensor,
        num_index_rows: Int32,
        num_blocks: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(mWriteIndex, num_index_rows, num_blocks).launch(
            grid=(cute.ceil_div(num_blocks, _INDEX_THREADS), 1, 1),
            block=(_INDEX_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, mWriteIndex: cute.Tensor, num_index_rows: Int32, num_blocks: Int32
    ):
        b = Int32(cute.arch.block_idx()[0]) * _INDEX_THREADS + cute.arch.thread_idx()[0]
        if b < num_blocks:
            running = Int32(0)
            c = Int32(0)
            while c < num_index_rows:
                cur = Int32(mWriteIndex[c, b])
                mWriteIndex[c, b] = running
                running = cutlass.max(running, cur)
                c += 1


class _FwdWriteIndexChunkMax:
    """Reduce the forward writer ``r + 1`` of each range into
    ``mWriteIndex[r // index_stride, b]``."""

    def __init__(self, block_size: int, rows_per_token: int):
        self.block_size = block_size
        self.rows_per_token = rows_per_token

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mWriteIndex: cute.Tensor,
        num_ranges: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges, mWriteIndex, num_ranges, index_stride, num_index_rows
        ).launch(
            grid=(cute.ceil_div(num_ranges, _INDEX_WARPS), 1, 1),
            block=(_INDEX_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mWriteIndex: cute.Tensor,
        num_ranges: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        r = Int32(cute.arch.block_idx()[0]) * _INDEX_WARPS + warp_in_cta
        if r < num_ranges:
            c = r // index_stride
            # The exclusive prefix max never reads the last chunk.
            if c < num_index_rows - 1:
                start = mQRanges[r, 0] * self.rows_per_token
                end = mQRanges[r, 1] * self.rows_per_token
                if end > start:
                    writer = r + 1
                    b = start // self.block_size + lane
                    last_b = (end - 1) // self.block_size
                    while b <= last_b:
                        if index_stride == 1:
                            mWriteIndex[c, b] = writer
                        else:
                            cute.arch.atomic_max(
                                elem_pointer(mWriteIndex, (c, b)),
                                writer,
                                sem="relaxed",
                                scope="gpu",
                            )
                        b += cute.arch.WARP_SIZE


class _BwdDQWriteIndexChunkMax:
    """Reduce the highest dQ writer of each pair into
    ``mWriteIndex[g // index_stride, b]``, with ``g`` the range or RangeMerge
    group."""

    def __init__(self, spec: BwdWriteIndexSpec):
        self.spec = spec

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mWriteIndex: cute.Tensor,
        writer_stride: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
        num_pairs: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges,
            mKRanges,
            mMaskTypes,
            mRowGroup,
            mWriteIndex,
            writer_stride,
            index_stride,
            num_index_rows,
            num_pairs,
        ).launch(
            grid=(cute.ceil_div(num_pairs, _INDEX_WARPS), 1, 1),
            block=(_INDEX_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mWriteIndex: cute.Tensor,
        writer_stride: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
        num_pairs: Int32,
    ):
        spec = self.spec
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row = Int32(cute.arch.block_idx()[0]) * _INDEX_WARPS + warp_in_cta
        if row < num_pairs:
            group_idx = row
            if const_expr(spec.range_merge):
                assert mRowGroup is not None
                group_idx = Int32(mRowGroup[row])
            c = group_idx // index_stride
            if c < num_index_rows - 1:
                q_s = mQRanges[row, 0]
                q_e = mQRanges[row, 1]
                seqlen_q = q_e - q_s
                seqlen_k = mKRanges[group_idx, 1] - mKRanges[group_idx, 0]
                if seqlen_q > 0 and seqlen_k > 0:
                    attn_r = Int32(MT_MAP.full)
                    if const_expr(spec.use_per_range_mask):
                        assert mMaskTypes is not None
                        attn_r = Int32(mMaskTypes[row])
                    block_info = spec.block_info
                    b = q_s // spec.tile_m + lane
                    last_b = (q_e - 1) // spec.tile_m
                    while b <= last_b:
                        k_tile_end = dq_pair_k_tile_end_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            q_s,
                            attn_r,
                            b,
                            spec.use_per_range_mask,
                        )
                        if k_tile_end > 0:
                            writer = group_idx * writer_stride + k_tile_end
                            if const_expr(not spec.range_merge) and index_stride == 1:
                                mWriteIndex[c, b] = writer
                            else:
                                cute.arch.atomic_max(
                                    elem_pointer(mWriteIndex, (c, b)),
                                    writer,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                        b += cute.arch.WARP_SIZE


class _BwdDKVWriteIndexChunkMax:
    """Reduce the last dK/dV writer of each pair into
    ``mWriteIndex[g // index_stride, kb]`` and, under RangeMerge, set
    ``mGroupAllKActive[g]`` when a pair of group ``g`` makes every K tile of
    the group active."""

    def __init__(self, spec: BwdWriteIndexSpec):
        self.spec = spec

    @cute.jit
    def __call__(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mWriteIndex: cute.Tensor,
        mGroupAllKActive: Optional[cute.Tensor],
        writer_stride: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
        num_pairs: Int32,
        stream: cuda.CUstream = None,
    ):
        self.kernel(
            mQRanges,
            mKRanges,
            mMaskTypes,
            mRowGroup,
            mWriteIndex,
            mGroupAllKActive,
            writer_stride,
            index_stride,
            num_index_rows,
            num_pairs,
        ).launch(
            grid=(cute.ceil_div(num_pairs, _INDEX_WARPS), 1, 1),
            block=(_INDEX_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRanges: cute.Tensor,
        mKRanges: cute.Tensor,
        mMaskTypes: Optional[cute.Tensor],
        mRowGroup: Optional[cute.Tensor],
        mWriteIndex: cute.Tensor,
        mGroupAllKActive: Optional[cute.Tensor],
        writer_stride: Int32,
        index_stride: Int32,
        num_index_rows: Int32,
        num_pairs: Int32,
    ):
        spec = self.spec
        tidx = cute.arch.thread_idx()[0]
        warp_in_cta = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row = Int32(cute.arch.block_idx()[0]) * _INDEX_WARPS + warp_in_cta
        if row < num_pairs:
            group_idx = row
            if const_expr(spec.range_merge):
                assert mRowGroup is not None
                group_idx = Int32(mRowGroup[row])
            k_s = mKRanges[group_idx, 0]
            k_e = mKRanges[group_idx, 1]
            seqlen_q = mQRanges[row, 1] - mQRanges[row, 0]
            seqlen_k = k_e - k_s
            if seqlen_q > 0 and seqlen_k > 0:
                attn_r = Int32(MT_MAP.full)
                if const_expr(spec.use_per_range_mask):
                    assert mMaskTypes is not None
                    attn_r = Int32(mMaskTypes[row])
                if const_expr(spec.range_merge):
                    assert mGroupAllKActive is not None
                    # Only bi_causal with seqlen_q > seqlen_k leaves a K tile
                    # of the pair without an unmasked Q tile.
                    if lane == 0 and (
                        attn_r != MT_MAP.bi_causal or seqlen_q <= seqlen_k
                    ):
                        mGroupAllKActive[group_idx] = Uint8(1)
                c = group_idx // index_stride
                if c < num_index_rows - 1:
                    writer = (group_idx + 1) * writer_stride
                    block_info = spec.block_info
                    kb = k_s // spec.cluster_tile_k + lane
                    last_kb = (k_e - 1) // spec.cluster_tile_k
                    while kb <= last_kb:
                        if dkv_pair_is_active_on_block(
                            block_info,
                            seqlen_q,
                            seqlen_k,
                            k_s,
                            attn_r,
                            kb,
                            spec.use_per_range_mask,
                        ):
                            if const_expr(not spec.range_merge) and index_stride == 1:
                                mWriteIndex[c, kb] = writer
                            else:
                                cute.arch.atomic_max(
                                    elem_pointer(mWriteIndex, (c, kb)),
                                    writer,
                                    sem="relaxed",
                                    scope="gpu",
                                )
                        kb += cute.arch.WARP_SIZE


_PREFIX_MAX_CACHE = get_jit_cache("range_write_index_prefix_max")
_FWD_INDEX_CACHE = get_jit_cache("range_write_index_fwd")
_BWD_DQ_INDEX_CACHE = get_jit_cache("range_write_index_bwd_dq")
_BWD_DKV_INDEX_CACHE = get_jit_cache("range_write_index_bwd_dkv")


def _fake_i32(*shape):
    return fake_tensor(Int32, shape, divisibility=1)


def _compile(launcher, *fake_args):
    return cute.compile(
        launcher,
        *fake_args,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _fake_bwd_range_args(spec: BwdWriteIndexSpec) -> tuple:
    """Fake ``(mQRanges, mKRanges, mMaskTypes, mRowGroup, mWriteIndex)``."""
    sym = cute.sym_int
    return (
        _fake_i32(sym(), 2),
        _fake_i32(sym(), 2),
        _fake_i32(sym()) if spec.use_per_range_mask else None,
        _fake_i32(sym()) if spec.range_merge else None,
        _fake_i32(sym(), sym()),
    )


def _run_write_index_prefix_max(write_index: torch.Tensor) -> None:
    num_index_rows, num_blocks = write_index.shape
    key = ("prefix_max",)
    if key not in _PREFIX_MAX_CACHE:
        sym = cute.sym_int
        _PREFIX_MAX_CACHE[key] = _compile(
            _WriteIndexPrefixMax(), _fake_i32(sym(), sym()), Int32(0), Int32(0)
        )
    _PREFIX_MAX_CACHE[key](write_index, num_index_rows, num_blocks)


def build_fwd(
    q_ranges: torch.Tensor,
    num_blocks: int,
    max_index_rows: int,
    block_size: int,
    rows_per_token: int,
) -> tuple[torch.Tensor, int]:
    """Build the forward write index ``(num_index_rows, num_blocks)`` on the
    device of ``q_ranges`` and return it with ``index_stride``.

    A block holds ``block_size`` rows and a token of ``q_ranges`` spans
    ``rows_per_token`` rows.
    """
    num_ranges = q_ranges.shape[0]
    num_index_rows, index_stride = write_index_shape(num_ranges, max_index_rows)
    write_index = torch.zeros(
        num_index_rows, num_blocks, dtype=torch.int32, device=q_ranges.device
    )
    if num_index_rows == 1:
        return write_index, index_stride
    key = (block_size, rows_per_token)
    if key not in _FWD_INDEX_CACHE:
        sym = cute.sym_int
        _FWD_INDEX_CACHE[key] = _compile(
            _FwdWriteIndexChunkMax(block_size, rows_per_token),
            _fake_i32(sym(), 2),
            _fake_i32(sym(), sym()),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _FWD_INDEX_CACHE[key](
        q_ranges, write_index, num_ranges, index_stride, num_index_rows
    )
    _run_write_index_prefix_max(write_index)
    return write_index, index_stride


def build_bwd_dq(
    spec: BwdWriteIndexSpec,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: Optional[torch.Tensor],
    row_group: Optional[torch.Tensor],
    num_blocks: int,
    max_index_rows: int,
    writer_stride: int,
) -> tuple[torch.Tensor, int]:
    """Build the backward dQ write index ``(num_index_rows, num_blocks)`` and
    return it with ``index_stride``.

    ``k_ranges`` has one row per ordered range. Without RangeMerge
    ``q_ranges`` pairs with it row by row; with RangeMerge ``row_group[i]`` is
    the ``k_ranges`` row of pair ``i``.
    """
    num_pairs = q_ranges.shape[0]
    num_index_rows, index_stride = write_index_shape(k_ranges.shape[0], max_index_rows)
    write_index = torch.zeros(
        num_index_rows, num_blocks, dtype=torch.int32, device=q_ranges.device
    )
    if num_index_rows == 1 or num_pairs == 0:
        return write_index, index_stride
    key = (spec,)
    if key not in _BWD_DQ_INDEX_CACHE:
        _BWD_DQ_INDEX_CACHE[key] = _compile(
            _BwdDQWriteIndexChunkMax(spec),
            *_fake_bwd_range_args(spec),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _BWD_DQ_INDEX_CACHE[key](
        q_ranges,
        k_ranges,
        mask_types if spec.use_per_range_mask else None,
        row_group if spec.range_merge else None,
        write_index,
        writer_stride,
        index_stride,
        num_index_rows,
        num_pairs,
    )
    _run_write_index_prefix_max(write_index)
    return write_index, index_stride


def build_bwd_dkv(
    spec: BwdWriteIndexSpec,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    mask_types: Optional[torch.Tensor],
    row_group: Optional[torch.Tensor],
    num_blocks: int,
    max_index_rows: int,
    writer_stride: int,
) -> tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """Build the backward dK/dV write index ``(num_index_rows, num_blocks)``
    and return it with ``index_stride`` and ``group_all_k_active``.

    The ranges follow :func:`build_bwd_dq`. Under RangeMerge
    ``group_all_k_active`` is uint8 ``[num_groups]``: 1 if a single pair of the
    group makes every K tile of the group active, so the attention kernel may
    skip its walk over the pairs of the group. It is ``None`` otherwise.
    """
    num_pairs = q_ranges.shape[0]
    num_groups = k_ranges.shape[0]
    num_index_rows, index_stride = write_index_shape(num_groups, max_index_rows)
    write_index = torch.zeros(
        num_index_rows, num_blocks, dtype=torch.int32, device=q_ranges.device
    )
    group_all_k_active = (
        torch.zeros(num_groups, dtype=torch.uint8, device=q_ranges.device)
        if spec.range_merge
        else None
    )
    if num_pairs == 0 or (num_index_rows == 1 and not spec.range_merge):
        return write_index, index_stride, group_all_k_active
    key = (spec,)
    if key not in _BWD_DKV_INDEX_CACHE:
        sym = cute.sym_int
        _BWD_DKV_INDEX_CACHE[key] = _compile(
            _BwdDKVWriteIndexChunkMax(spec),
            *_fake_bwd_range_args(spec),
            fake_tensor(Uint8, (sym(),), divisibility=1) if spec.range_merge else None,
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
        )
    _BWD_DKV_INDEX_CACHE[key](
        q_ranges,
        k_ranges,
        mask_types if spec.use_per_range_mask else None,
        row_group if spec.range_merge else None,
        write_index,
        group_all_k_active,
        writer_stride,
        index_stride,
        num_index_rows,
        num_pairs,
    )
    if num_index_rows > 1:
        _run_write_index_prefix_max(write_index)
    return write_index, index_stride, group_all_k_active
