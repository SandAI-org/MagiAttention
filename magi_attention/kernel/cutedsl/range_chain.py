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

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

# isort: split
from quack.cute_dsl_utils import ParamsBase

from .block_info import BlockInfo
from .cutedsl_utils import (
    elem_pointer,
    ld_acquire,
    mbarrier_arrive_peer_release_cluster,
    mbarrier_wait_acquire_cluster,
    st_global_release_gpu_i32,
)
from .ffa_utils import MT_MAP


@dataclass
class ChainTensors(ParamsBase):
    """Publish chain of one output and the checkpoints of its writers.

    ``mChain[b, h] = [published, count]``. Writer ``j`` of range ``r`` is
    ``r * writer_stride + j + 1`` (0: none), so ``writer_stride`` is the count
    of writer numbers per range. ``mConflictCkpt[c, b]`` is the highest writer
    on block ``b`` among ranges ``[0, c * ckpt_stride)`` (0: none), so
    ``ckpt_stride`` is the count of ranges per checkpoint row; the kernel only
    reads it.
    """

    mChain: cute.Tensor
    mConflictCkpt: cute.Tensor
    writer_stride: Int32
    ckpt_stride: Int32


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
def tile_events(
    range_start: Int32,
    range_end: Int32,
    tile_idx: Int32,
    block_size: cutlass.Constexpr[int],
):
    row = range_start + tile_idx * block_size
    left, right = tile_blocks(range_start, range_end, tile_idx, block_size)
    unaligned = range_start % block_size != 0
    left_weight = Int32(2)
    if unaligned and tile_idx > 0:
        left_weight = Int32(1)
    right_weight = Int32(2)
    if unaligned and row + block_size < range_end:
        right_weight = Int32(1)
    return left, right, left_weight, right_weight


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
                    if const_expr(use_per_range_mask):
                        (
                            n_lo,
                            n_hi,
                        ) = block_info.get_n_block_min_max_for_m_block_per_range_seqlens(
                            seqlen_q, seqlen_k, mt, attn_type
                        )
                    else:
                        (
                            n_lo,
                            n_hi,
                        ) = block_info.get_n_block_min_max_for_m_block_seqlens(
                            seqlen_q, seqlen_k, mt
                        )
                    if n_hi > n_lo:
                        k_tile_end = cutlass.max(k_tile_end, n_hi)
                mt += 1
    return k_tile_end


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
def lookup_fwd_conflict(
    mConflictCkpt: cute.Tensor,
    mQRanges: cute.Tensor,
    batch_idx: Int32,
    block: Int32,
    block_size: cutlass.Constexpr[int],
    rows_per_token: cutlass.Constexpr[int],
    ckpt_stride: Int32,
) -> Int32:
    """Highest earlier fwd writer (r + 1) touching physical ``block``; 0 if none."""
    c = batch_idx // ckpt_stride
    pred = Int32(mConflictCkpt[c, block])
    if ckpt_stride > 1:
        r_lo = c * ckpt_stride
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
def wait_published(
    mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32
) -> None:
    """The fence orders the generic acquire before cross-CTA async-proxy ops."""
    ptr = elem_pointer(mChain, (block, head, 0))
    published = ld_acquire(ptr)
    while published != writer:
        published = ld_acquire(ptr)
    cute.arch.fence_proxy("async.global")


@cute.jit
def arrive(
    mChain: cute.Tensor,
    block: Int32,
    head: Int32,
    writer: Int32,
    weight: Int32,
    cluster_size: cutlass.Constexpr[int] = 1,
) -> None:
    """Arrive after CTA writes complete; count resets before release publish while successor waits."""
    if weight == 2 * cluster_size:
        publish(mChain, block, head, writer)
    else:
        count_ptr = elem_pointer(mChain, (block, head, 1))
        prev = cute.arch.atomic_add(count_ptr, weight, sem="acq_rel", scope="gpu")
        if prev + weight == 2 * cluster_size:
            mChain[block, head, 1] = Int32(0)
            publish(mChain, block, head, writer)


@cute.jit
def publish(mChain: cute.Tensor, block: Int32, head: Int32, writer: Int32) -> None:
    st_global_release_gpu_i32(elem_pointer(mChain, (block, head, 0)), writer)


@cute.jit
def pair_handoff_arrive(mbar: cute.Pointer, handoff: Int32, peer: Int32) -> None:
    """Arrive after bulk reductions complete; alternating mbarriers bound the phase lead to one."""
    mbarrier_arrive_peer_release_cluster(mbar + (handoff & 1), peer)


@cute.jit
def pair_handoff_wait(mbar: cute.Pointer, handoff: Int32) -> None:
    """Both CTAs must hand off at the same pairs."""
    mbarrier_wait_acquire_cluster(mbar + (handoff & 1), (handoff >> 1) & 1)
    cute.arch.fence_proxy("async.global")
