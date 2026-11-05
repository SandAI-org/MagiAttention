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

"""CuTe DSL Radix Select Top-K kernel.

Unified implementation controlled by compile-time constants:
  IS_BF16:      True  -> BF16 input, Int16 ordered, 2-round radix
                False -> FP32 input, Int32 ordered, 4-round radix
  IS_MULTI_CTA: True  -> Multi-CTA with two-level histogram reduction
                False -> Single-CTA, data loaded once to SMEM

Multi-CTA histogram reduction uses two levels:
  Level 1 (intra-pair):  Cluster DSMEM (store_shared_remote + mbarrier)
                          between 2 CTAs in one cluster pair (~50ns)
  Level 2 (inter-pair):  GMEM atomicAdd across cluster pairs
                          + GMEM barrier + read-back (~400ns)

Cluster size is fixed at 2 (compile-time). ctas_per_group is runtime.
Grid layout: [num_groups, ctas_per_group, 1], cluster=[1, 2, 1].

Write-out is ordered: each thread owns a contiguous segment, a block-wide
prefix scan (plus a cross-CTA prefix in the multi-CTA path) gives every
selected element its slot.  Each row's indices come out ascending, and ties at
the k-th value keep the lowest indices, so the result is deterministic.

Supports: float32, float16, bfloat16 inputs on CUDA.

Public API
----------
- ``cutedsl_topk(x, k, sorted=False, out_dtype=torch.int64)`` -> ``indices``
- ``cutedsl_topk_available()`` -> ``bool``
- ``cutedsl_topk_error()`` -> ``str | None``
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int16, Int32, Int64, const_expr
from quack.compile_utils import make_fake_tensor
from quack.utils import elem_pointer, store_shared_remote

from magi_attention.utils import nvtx

from .cutedsl_utils import (
    VEC_BF16,
    VEC_FP32,
    bf16_to_ordered16,
    float_to_ordered,
    gmem_atomic_add_relaxed,
)
from .cutedsl_utils import gmem_ld_acquire as ld_acquire
from .cutedsl_utils import gmem_red_release as red_release
from .cutedsl_utils import (
    gmem_store_cs_i32,
    int16_zext_to_int32,
    st_release,
    tensor_atomic_add,
    u32_eq,
    u32_gt,
    vec_load_bf16_to_ordered16,
    vec_load_fp32_to_ordered32,
)

# ====================================================================
# Constants
# ====================================================================

RADIX = 256
BLOCK_THREADS = 1024
CLUSTER_SIZE = 2

MAX_SMEM_BYTES = 232448

SMEM_OVERHEAD_BYTES = (
    RADIX * 4 + RADIX * 4 + 4 * 4  # s_hist  # s_sfx  # s_prefix_t, s_remk_t, s_fb, s_fr
)

# GMEM state per group (all int32):
#   [0..255]     histogram buffer 0
#   [256..511]   histogram buffer 1
#   [512..767]   histogram buffer 2
#   [768]        arrival_counter
#   [769..]      per-CTA (gt_count, eq_count) pairs for the ordered write-out
MAX_CTAS_PER_GROUP = 256
GSTATE_HIST0 = 0
GSTATE_HIST1 = 256
GSTATE_HIST2 = 512
GSTATE_ARRIVAL = 768
GSTATE_CTA_COUNTS = 769
GSTATE_SIZE = GSTATE_CTA_COUNTS + 2 * MAX_CTAS_PER_GROUP

# s_sfx scratch layout reused by the ordered write-out after radix select:
#   [0, 2 * SCAN_WARPS)     per-warp partial sums of the two scanned counts
#   [SCAN_TOTAL, +2)        block totals
#   [SCAN_BASE, +2)         cross-CTA (gt, eq) bases
SCAN_WARPS = BLOCK_THREADS // 32
SCAN_TOTAL = 2 * SCAN_WARPS
SCAN_BASE = SCAN_TOTAL + 2


# ====================================================================
# Common radix routines
# ====================================================================


@cute.jit
def _suffix_sum_from_reg(s_sfx: cute.Tensor, val: Int32, tidx: Int32):
    """Warp-shuffle suffix sum. Caller must supply val=s_hist[tidx] (already barrier-protected)."""
    NUM_WARPS = Int32(const_expr(RADIX // 32))  # 8
    lane = tidx & Int32(31)
    warp_id = tidx >> Int32(5)

    for i in cutlass.range_constexpr(5):
        offset = 1 << i
        partial = cute.arch.shuffle_sync_down(val, Int32(const_expr(offset)))
        if lane + Int32(const_expr(offset)) < Int32(32):
            val = val + partial

    if tidx < Int32(RADIX):
        if lane == Int32(0):
            s_sfx[warp_id] = val
    cute.arch.barrier()

    warp_sum = Int32(0)
    if warp_id == Int32(0):
        if lane < NUM_WARPS:
            warp_sum = s_sfx[lane]
        for i in cutlass.range_constexpr(3):
            offset = 1 << i
            partial = cute.arch.shuffle_sync_down(warp_sum, Int32(const_expr(offset)))
            if lane + Int32(const_expr(offset)) < NUM_WARPS:
                warp_sum = warp_sum + partial
        if lane < NUM_WARPS:
            s_sfx[lane] = warp_sum
    cute.arch.barrier()

    correction = Int32(0)
    if tidx < Int32(RADIX):
        if warp_id + Int32(1) < NUM_WARPS:
            correction = s_sfx[warp_id + Int32(1)]
        val = val + correction
        s_sfx[tidx] = val
    cute.arch.barrier()


@cute.jit
def _find_bucket(
    s_sfx: cute.Tensor,
    s_fb: cute.Tensor,
    s_fr: cute.Tensor,
    s_prefix_t: cute.Tensor,
    s_remk_t: cute.Tensor,
    s_hist: cute.Tensor,
    tidx: Int32,
    pv: Int32,
    rk: Int32,
    sh: Int32,
):
    if tidx == 0:
        s_fb[0] = Int32(0)
        s_fr[0] = rk
    cute.arch.barrier()
    if tidx < Int32(RADIX):
        cge = s_sfx[tidx]
        cgt = Int32(0)
        if tidx < Int32(const_expr(RADIX - 1)):
            cgt = s_sfx[tidx + Int32(1)]
        if (cge >= rk) & (cgt < rk):
            s_fb[0] = tidx
            s_fr[0] = rk - cgt
    cute.arch.barrier()
    if tidx == 0:
        f = s_fb[0]
        s_prefix_t[0] = pv | (f << sh)
        s_remk_t[0] = s_fr[0]
    # Clear s_hist for next round (piggyback on final barrier)
    if tidx < Int32(RADIX):
        s_hist[tidx] = Int32(0)
    cute.arch.barrier()


@cute.jit
def _block_exclusive_scan2(s_scan: cute.Tensor, a: Int32, b: Int32, tidx: Int32):
    """Block-wide exclusive prefix sums of two per-thread Int32 values.

    Must be called by all BLOCK_THREADS threads.  Returns
    (excl_a, excl_b, total_a, total_b).
    """
    lane = tidx & Int32(31)
    warp_id = tidx >> Int32(5)

    inc_a = a
    inc_b = b
    for i in cutlass.range_constexpr(5):
        offset = 1 << i
        pa = cute.arch.shuffle_sync_up(inc_a, offset, mask_and_clamp=0)
        pb = cute.arch.shuffle_sync_up(inc_b, offset, mask_and_clamp=0)
        if lane >= Int32(const_expr(offset)):
            inc_a = inc_a + pa
            inc_b = inc_b + pb
    if lane == Int32(31):
        s_scan[warp_id] = inc_a
        s_scan[Int32(SCAN_WARPS) + warp_id] = inc_b
    cute.arch.barrier()

    if warp_id == Int32(0):
        wa = s_scan[lane]
        wb = s_scan[Int32(SCAN_WARPS) + lane]
        sa = wa
        sb = wb
        for i in cutlass.range_constexpr(5):
            offset = 1 << i
            pa = cute.arch.shuffle_sync_up(sa, offset, mask_and_clamp=0)
            pb = cute.arch.shuffle_sync_up(sb, offset, mask_and_clamp=0)
            if lane >= Int32(const_expr(offset)):
                sa = sa + pa
                sb = sb + pb
        s_scan[lane] = sa - wa
        s_scan[Int32(SCAN_WARPS) + lane] = sb - wb
        if lane == Int32(31):
            s_scan[Int32(SCAN_TOTAL)] = sa
            s_scan[Int32(SCAN_TOTAL + 1)] = sb
    cute.arch.barrier()

    excl_a = s_scan[warp_id] + inc_a - a
    excl_b = s_scan[Int32(SCAN_WARPS) + warp_id] + inc_b - b
    return excl_a, excl_b, s_scan[Int32(SCAN_TOTAL)], s_scan[Int32(SCAN_TOTAL + 1)]


@cute.jit
def _inter_cta_barrier(
    mState: cute.Tensor,
    arrival_off: Int32,
    barrier_phase: Int32,
    ctas_per_group_val: Int32,
    tidx: Int32,
):
    """Lightweight GMEM inter-CTA barrier across all CTAs in a group."""
    if tidx == 0:
        red_release(mState, arrival_off, Int32(1))
    cute.arch.barrier()
    target = (barrier_phase + Int32(1)) * ctas_per_group_val
    if tidx == 0:
        while ld_acquire(mState, arrival_off) < target:
            pass
    cute.arch.barrier()


def _compute_smem_cap(is_bf16: bool) -> int:
    """Compute max ordered elements that fit in SMEM with cluster=2 overhead."""
    elem_bytes = 2 if is_bf16 else 4
    recv_hist_bytes = CLUSTER_SIZE * RADIX * 4
    num_rounds = 2 if is_bf16 else 4
    mbar_bytes = num_rounds * 8
    overhead = SMEM_OVERHEAD_BYTES + recv_hist_bytes + mbar_bytes + 64
    avail = MAX_SMEM_BYTES - overhead
    return avail // elem_bytes


# ====================================================================
# Unified RadixTopK kernel
# ====================================================================


class RadixTopK:
    """Unified Radix Top-K kernel with two-level histogram reduction.

    IS_BF16:      BF16 path (Int16 ordered, 2 rounds) vs FP32 (Int32, 4 rounds)
    IS_MULTI_CTA: Multi-CTA (cluster=2 DSMEM + GMEM) vs single-CTA
    """

    def __init__(self, is_bf16: bool, is_multi_cta: bool):
        self.IS_BF16 = is_bf16
        self.IS_MULTI_CTA = is_multi_cta
        self.NUM_ROUNDS = 2 if is_bf16 else 4
        self.TOTAL_BITS = 16 if is_bf16 else 32
        self.NUM_MBAR = self.NUM_ROUNDS
        if is_multi_cta:
            self.SMEM_CAP = _compute_smem_cap(is_bf16)
        else:
            elem_bytes = 2 if is_bf16 else 4
            overhead = SMEM_OVERHEAD_BYTES + 64
            self.SMEM_CAP = (MAX_SMEM_BYTES - overhead) // elem_bytes

    @cute.jit
    def _load_ordered(self, s_ordered: cute.Tensor, i: Int32) -> Int32:
        if const_expr(self.IS_BF16):
            return int16_zext_to_int32(s_ordered[i])
        else:
            return s_ordered[i]

    @cute.jit
    def _segment_counts(
        self,
        s_ordered: cute.Tensor,
        seg_start: Int32,
        seg_end: Int32,
        pivot: Int32,
    ):
        """Count elements > pivot and == pivot in [seg_start, seg_end)."""
        cnt_gt = Int32(0)
        cnt_eq = Int32(0)
        i = seg_start
        while i < seg_end:
            o = self._load_ordered(s_ordered, i)
            if u32_gt(o, pivot):
                cnt_gt = cnt_gt + Int32(1)
            if u32_eq(o, pivot):
                cnt_eq = cnt_eq + Int32(1)
            i = i + Int32(1)
        return cnt_gt, cnt_eq

    @cute.jit
    def _segment_write(
        self,
        s_ordered: cute.Tensor,
        mIndices: cute.Tensor,
        row: Int32,
        index_offset: Int32,
        seg_start: Int32,
        seg_end: Int32,
        pivot: Int32,
        remk: Int32,
        gt_before: Int32,
        eq_before: Int32,
        k_val: Int32,
    ):
        """Write selected indices of [seg_start, seg_end) in ascending order.

        gt_before / eq_before count the elements > / == pivot that precede the
        segment in the row.  An element's output slot is the number of selected
        elements before it: gt_before + min(eq_before, remk).
        """
        i = seg_start
        while i < seg_end:
            o = self._load_ordered(s_ordered, i)
            if u32_gt(o, pivot):
                eq_taken = eq_before
                if eq_taken > remk:
                    eq_taken = remk
                pos = gt_before + eq_taken
                if pos < k_val:
                    gmem_store_cs_i32(mIndices, (row, pos), index_offset + i)
                gt_before = gt_before + Int32(1)
            if u32_eq(o, pivot):
                if eq_before < remk:
                    pos = gt_before + eq_before
                    if pos < k_val:
                        gmem_store_cs_i32(mIndices, (row, pos), index_offset + i)
                eq_before = eq_before + Int32(1)
            i = i + Int32(1)

    @cute.jit
    def __call__(
        self,
        mInput: cute.Tensor,
        mIndices: cute.Tensor,
        mState: cute.Tensor,
        k_val: Int32,
        chunk_size_val: Int32,
        ctas_per_group_val: Int32,
        num_groups_val: Int32,
        num_rows_val: Int32,
        stream,
    ):
        if const_expr(self.IS_MULTI_CTA):
            _total_ctas = num_groups_val * ctas_per_group_val  # noqa: F841
            self.kernel(
                mInput,
                mIndices,
                mState,
                k_val,
                chunk_size_val,
                ctas_per_group_val,
                num_groups_val,
                num_rows_val,
            ).launch(
                grid=[num_groups_val, ctas_per_group_val, 1],
                block=[BLOCK_THREADS, 1, 1],
                cluster=[1, const_expr(CLUSTER_SIZE), 1],
                stream=stream,
                min_blocks_per_mp=1,
            )
        else:
            self.kernel(
                mInput,
                mIndices,
                mState,
                k_val,
                chunk_size_val,
                ctas_per_group_val,
                num_groups_val,
                num_rows_val,
            ).launch(
                grid=[mInput.shape[0], 1, 1],
                block=[BLOCK_THREADS, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mIndices: cute.Tensor,
        mState: cute.Tensor,
        k_val: Int32,
        chunk_size_val: Int32,
        ctas_per_group_val: Int32,
        num_groups_val: Int32,
        num_rows_val: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        n = mInput.shape[1]

        smem = cutlass.utils.SmemAllocator()
        s_hist = smem.allocate_tensor(Int32, cute.make_layout(RADIX), byte_alignment=16)
        s_sfx = smem.allocate_tensor(Int32, cute.make_layout(RADIX), byte_alignment=16)
        s_prefix_t = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
        s_remk_t = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
        s_fb = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
        s_fr = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)

        if const_expr(self.IS_MULTI_CTA):
            s_recv_hist = smem.allocate_tensor(
                Int32,
                cute.make_layout((const_expr(CLUSTER_SIZE), RADIX)),
                byte_alignment=16,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=const_expr(self.NUM_MBAR))

        if const_expr(self.IS_BF16):
            s_ordered_16 = smem.allocate_tensor(
                Int16, cute.make_layout(const_expr(self.SMEM_CAP)), byte_alignment=16
            )
            s_ord = s_ordered_16
        else:
            s_ordered_32 = smem.allocate_tensor(
                Int32, cute.make_layout(const_expr(self.SMEM_CAP)), byte_alignment=16
            )
            s_ord = s_ordered_32

        # ===============================================================
        # Multi-CTA path: two-level histogram reduction
        # ===============================================================
        if const_expr(self.IS_MULTI_CTA):
            group_id = cute.arch.block_idx()[0]
            cta_in_group = cute.arch.block_idx()[1]
            cta_in_pair = cute.arch.block_idx_in_cluster()
            _pair_id = cta_in_group >> Int32(1)  # noqa: F841
            num_pairs = (ctas_per_group_val + Int32(1)) >> Int32(1)
            is_multi_pair = num_pairs > Int32(1)

            if tidx < Int32(const_expr(self.NUM_MBAR)):
                cute.arch.mbarrier_init(mbar_ptr + tidx, Int32(1))
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

            state_off = group_id * Int32(GSTATE_SIZE)
            arrival_off = state_off + Int32(GSTATE_ARRIVAL)
            hist_base0 = state_off + Int32(GSTATE_HIST0)
            hist_base1 = state_off + Int32(GSTATE_HIST1)
            hist_base2 = state_off + Int32(GSTATE_HIST2)

            barrier_phase = Int32(0)
            mbar_phase = Int32(0)
            total_iters = (num_rows_val + num_groups_val - Int32(1)) // num_groups_val

            iter_idx = Int32(0)
            while iter_idx < total_iters:
                row = group_id + iter_idx * num_groups_val
                if row < num_rows_val:
                    chunk_start = cta_in_group * chunk_size_val
                    chunk_end = chunk_start + chunk_size_val
                    if chunk_end > n:
                        chunk_end = n
                    actual = chunk_end - chunk_start
                    if actual < Int32(0):
                        actual = Int32(0)

                    if const_expr(self.IS_BF16):
                        vec_sz = Int32(const_expr(VEC_BF16))
                        aligned_actual = (actual // vec_sz) * vec_sz
                        i = tidx * vec_sz
                        while i < aligned_actual:
                            vec_load_bf16_to_ordered16(
                                mInput[row, None], chunk_start + i, s_ordered_16, i
                            )
                            i = i + Int32(BLOCK_THREADS) * vec_sz
                        i = aligned_actual + tidx
                        while i < actual:
                            s_ordered_16[i] = bf16_to_ordered16(
                                BFloat16(mInput[row, chunk_start + i])
                            )
                            i = i + Int32(BLOCK_THREADS)
                    else:
                        vec_sz = Int32(const_expr(VEC_FP32))
                        aligned_actual = (actual // vec_sz) * vec_sz
                        i = tidx * vec_sz
                        while i < aligned_actual:
                            vec_load_fp32_to_ordered32(
                                mInput[row, None], chunk_start + i, s_ordered_32, i
                            )
                            i = i + Int32(BLOCK_THREADS) * vec_sz
                        i = aligned_actual + tidx
                        while i < actual:
                            s_ordered_32[i] = float_to_ordered(
                                Float32(mInput[row, chunk_start + i])
                            )
                            i = i + Int32(BLOCK_THREADS)
                    cute.arch.barrier()

                    if tidx == 0:
                        s_prefix_t[0] = Int32(0)
                        s_remk_t[0] = k_val

                    _inter_cta_barrier(
                        mState, arrival_off, barrier_phase, ctas_per_group_val, tidx
                    )
                    barrier_phase = barrier_phase + Int32(1)

                    for ri in cutlass.range_constexpr(const_expr(self.NUM_ROUNDS)):
                        sh = Int32(const_expr((self.TOTAL_BITS // 8 - 1 - ri) * 8))
                        pv = s_prefix_t[0]
                        rk = s_remk_t[0]

                        mbar_idx = Int32(const_expr(ri))

                        global_round = iter_idx * Int32(
                            const_expr(self.NUM_ROUNDS)
                        ) + Int32(const_expr(ri))
                        buf_idx = global_round - (global_round // Int32(3)) * Int32(3)
                        next_buf_idx = (global_round + Int32(1)) - (
                            (global_round + Int32(1)) // Int32(3)
                        ) * Int32(3)

                        hist_off = hist_base0
                        if buf_idx == Int32(1):
                            hist_off = hist_base1
                        if buf_idx == Int32(2):
                            hist_off = hist_base2

                        next_hist_off = hist_base0
                        if next_buf_idx == Int32(1):
                            next_hist_off = hist_base1
                        if next_buf_idx == Int32(2):
                            next_hist_off = hist_base2

                        if const_expr(ri == 0):
                            c = tidx
                            while c < Int32(RADIX):
                                s_hist[c] = Int32(0)
                                c = c + Int32(BLOCK_THREADS)
                            cute.arch.barrier()

                        i = tidx
                        while i < actual:
                            if const_expr(self.IS_BF16):
                                o = int16_zext_to_int32(s_ordered_16[i])
                            else:
                                o = s_ordered_32[i]
                            m = Boolean(True)
                            if const_expr(ri > 0):
                                rb = Int32(const_expr(ri * 8))
                                msk = ~Int32(0) << (
                                    Int32(const_expr(self.TOTAL_BITS)) - rb
                                )
                                m = u32_eq(o & msk, pv)
                            if m:
                                tensor_atomic_add(
                                    s_hist, (o >> sh) & Int32(0xFF), Int32(1)
                                )
                            i = i + Int32(BLOCK_THREADS)
                        cute.arch.barrier()

                        warp_id = tidx >> Int32(5)
                        if warp_id == Int32(0):
                            with cute.arch.elect_one():
                                total_tx = Int32(const_expr(CLUSTER_SIZE * RADIX * 4))
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    mbar_ptr + mbar_idx, total_tx
                                )

                        c = tidx
                        while c < Int32(RADIX):
                            val = s_hist[c]
                            for peer in cutlass.range_constexpr(
                                const_expr(CLUSTER_SIZE)
                            ):
                                store_shared_remote(
                                    val,
                                    elem_pointer(s_recv_hist, (cta_in_pair, c)),
                                    mbar_ptr + mbar_idx,
                                    peer_cta_rank_in_cluster=Int32(const_expr(peer)),
                                )
                            c = c + Int32(BLOCK_THREADS)

                        cute.arch.mbarrier_wait(mbar_ptr + mbar_idx, phase=mbar_phase)

                        sfx_val = Int32(0)
                        c = tidx
                        while c < Int32(RADIX):
                            v = s_recv_hist[Int32(0), c] + s_recv_hist[Int32(1), c]
                            s_hist[c] = v
                            c = c + Int32(BLOCK_THREADS)
                        cute.arch.barrier()

                        if is_multi_pair:
                            if cta_in_pair == Int32(0):
                                c = tidx
                                while c < Int32(RADIX):
                                    lv = s_hist[c]
                                    if lv > Int32(0):
                                        gmem_atomic_add_relaxed(
                                            mState, hist_off + c, lv
                                        )
                                    c = c + Int32(BLOCK_THREADS)

                            if cta_in_group == Int32(0):
                                c = tidx
                                while c < Int32(RADIX):
                                    mState[next_hist_off + c] = Int32(0)
                                    c = c + Int32(BLOCK_THREADS)

                            _inter_cta_barrier(
                                mState,
                                arrival_off,
                                barrier_phase,
                                ctas_per_group_val,
                                tidx,
                            )
                            barrier_phase = barrier_phase + Int32(1)

                            if tidx < Int32(RADIX):
                                sfx_val = mState[hist_off + tidx]
                                s_hist[tidx] = sfx_val
                            cute.arch.barrier()
                        else:
                            if tidx < Int32(RADIX):
                                sfx_val = s_hist[tidx]

                        _suffix_sum_from_reg(s_sfx, sfx_val, tidx)
                        _find_bucket(
                            s_sfx,
                            s_fb,
                            s_fr,
                            s_prefix_t,
                            s_remk_t,
                            s_hist,
                            tidx,
                            pv,
                            rk,
                            sh,
                        )

                    pivot = s_prefix_t[0]
                    remk = s_remk_t[0]

                    seg_len = (actual + Int32(BLOCK_THREADS - 1)) // Int32(
                        BLOCK_THREADS
                    ) | Int32(1)
                    seg_start = tidx * seg_len
                    if seg_start > actual:
                        seg_start = actual
                    seg_end = seg_start + seg_len
                    if seg_end > actual:
                        seg_end = actual
                    cnt_gt, cnt_eq = self._segment_counts(
                        s_ord, seg_start, seg_end, pivot
                    )
                    ex_gt, ex_eq, tot_gt, tot_eq = _block_exclusive_scan2(
                        s_sfx, cnt_gt, cnt_eq, tidx
                    )

                    counts_off = state_off + Int32(GSTATE_CTA_COUNTS)
                    if tidx == 0:
                        mState[counts_off + cta_in_group * Int32(2)] = tot_gt
                        mState[counts_off + cta_in_group * Int32(2) + Int32(1)] = tot_eq
                    _inter_cta_barrier(
                        mState, arrival_off, barrier_phase, ctas_per_group_val, tidx
                    )
                    barrier_phase = barrier_phase + Int32(1)

                    if tidx == 0:
                        gt_base = Int32(0)
                        eq_base = Int32(0)
                        j = Int32(0)
                        while j < cta_in_group:
                            gt_base = gt_base + mState[counts_off + j * Int32(2)]
                            eq_base = (
                                eq_base + mState[counts_off + j * Int32(2) + Int32(1)]
                            )
                            j = j + Int32(1)
                        s_sfx[Int32(SCAN_BASE)] = gt_base
                        s_sfx[Int32(SCAN_BASE + 1)] = eq_base
                    cute.arch.barrier()

                    self._segment_write(
                        s_ord,
                        mIndices,
                        row,
                        chunk_start,
                        seg_start,
                        seg_end,
                        pivot,
                        remk,
                        s_sfx[Int32(SCAN_BASE)] + ex_gt,
                        s_sfx[Int32(SCAN_BASE + 1)] + ex_eq,
                        k_val,
                    )

                    # The next row reloads s_ordered; wait for every reader.
                    cute.arch.barrier()

                mbar_phase = mbar_phase + Int32(1)
                iter_idx = iter_idx + Int32(1)

            if cta_in_group == Int32(0):
                c = tidx
                while c < Int32(RADIX):
                    mState[hist_base0 + c] = Int32(0)
                    mState[hist_base1 + c] = Int32(0)
                    mState[hist_base2 + c] = Int32(0)
                    c = c + Int32(BLOCK_THREADS)
                if tidx == 0:
                    st_release(mState, arrival_off, Int32(0))

        else:
            # ===============================================================
            # Single-CTA path
            # ===============================================================
            row = cute.arch.block_idx()[0]

            actual = n
            cap = Int32(const_expr(self.SMEM_CAP))
            if actual > cap:
                actual = cap

            if const_expr(self.IS_BF16):
                vec_sz = Int32(const_expr(VEC_BF16))
                aligned_actual = (actual // vec_sz) * vec_sz
                i = tidx * vec_sz
                while i < aligned_actual:
                    vec_load_bf16_to_ordered16(mInput[row, None], i, s_ordered_16, i)
                    i = i + Int32(BLOCK_THREADS) * vec_sz
                i = aligned_actual + tidx
                while i < actual:
                    s_ordered_16[i] = bf16_to_ordered16(BFloat16(mInput[row, i]))
                    i = i + Int32(BLOCK_THREADS)
            else:
                vec_sz = Int32(const_expr(VEC_FP32))
                aligned_actual = (actual // vec_sz) * vec_sz
                i = tidx * vec_sz
                while i < aligned_actual:
                    vec_load_fp32_to_ordered32(mInput[row, None], i, s_ordered_32, i)
                    i = i + Int32(BLOCK_THREADS) * vec_sz
                i = aligned_actual + tidx
                while i < actual:
                    s_ordered_32[i] = float_to_ordered(Float32(mInput[row, i]))
                    i = i + Int32(BLOCK_THREADS)
            cute.arch.barrier()

            if tidx == 0:
                s_prefix_t[0] = Int32(0)
                s_remk_t[0] = k_val
            cute.arch.barrier()

            for ri in cutlass.range_constexpr(const_expr(self.NUM_ROUNDS)):
                sh = Int32(const_expr((self.TOTAL_BITS // 8 - 1 - ri) * 8))
                pv = s_prefix_t[0]
                rk = s_remk_t[0]

                if const_expr(ri == 0):
                    c = tidx
                    while c < Int32(RADIX):
                        s_hist[c] = Int32(0)
                        c = c + Int32(BLOCK_THREADS)
                    cute.arch.barrier()

                i = tidx
                while i < actual:
                    if const_expr(self.IS_BF16):
                        o = int16_zext_to_int32(s_ordered_16[i])
                    else:
                        o = s_ordered_32[i]
                    m = Boolean(True)
                    if const_expr(ri > 0):
                        rb = Int32(const_expr(ri * 8))
                        msk = ~Int32(0) << (Int32(const_expr(self.TOTAL_BITS)) - rb)
                        m = u32_eq(o & msk, pv)
                    if m:
                        tensor_atomic_add(s_hist, (o >> sh) & Int32(0xFF), Int32(1))
                    i = i + Int32(BLOCK_THREADS)
                cute.arch.barrier()

                sfx_val = Int32(0)
                if tidx < Int32(RADIX):
                    sfx_val = s_hist[tidx]
                _suffix_sum_from_reg(s_sfx, sfx_val, tidx)
                _find_bucket(
                    s_sfx, s_fb, s_fr, s_prefix_t, s_remk_t, s_hist, tidx, pv, rk, sh
                )

            pivot = s_prefix_t[0]
            remk = s_remk_t[0]
            seg_len = (actual + Int32(BLOCK_THREADS - 1)) // Int32(
                BLOCK_THREADS
            ) | Int32(1)
            seg_start = tidx * seg_len
            if seg_start > actual:
                seg_start = actual
            seg_end = seg_start + seg_len
            if seg_end > actual:
                seg_end = actual
            cnt_gt, cnt_eq = self._segment_counts(s_ord, seg_start, seg_end, pivot)
            ex_gt, ex_eq, _tot_gt, _tot_eq = _block_exclusive_scan2(
                s_sfx, cnt_gt, cnt_eq, tidx
            )
            self._segment_write(
                s_ord,
                mIndices,
                row,
                Int32(0),
                seg_start,
                seg_end,
                pivot,
                remk,
                ex_gt,
                ex_eq,
                k_val,
            )


# ====================================================================
# Compilation
# ====================================================================

_compiled_cache: dict[tuple[bool, bool], object] = {}


def _compile(is_bf16: bool, is_multi_cta: bool):
    key = (is_bf16, is_multi_cta)
    if key in _compiled_cache:
        return _compiled_cache[key]

    bs, ns, ks = cute.sym_int(), cute.sym_int(), cute.sym_int()
    data_dtype = cute.BFloat16 if is_bf16 else cute.Float32

    op = RadixTopK(is_bf16, is_multi_cta)

    if is_multi_cta:
        gs = cute.sym_int()
        compiled = cute.compile(
            op,
            make_fake_tensor(data_dtype, shape=(bs, ns), divisibility=1),
            make_fake_tensor(cute.Int32, shape=(bs, ks), divisibility=1),
            make_fake_tensor(cute.Int32, shape=(gs,), divisibility=1),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    else:
        compiled = cute.compile(
            op,
            make_fake_tensor(data_dtype, shape=(bs, ns), divisibility=1),
            make_fake_tensor(cute.Int32, shape=(bs, ks), divisibility=1),
            make_fake_tensor(cute.Int32, shape=(1,), divisibility=1),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _compiled_cache[key] = compiled
    return compiled


# ====================================================================
# Python entry: cute_topk (internal)
# ====================================================================


def _get_num_sms() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


_gstate_cache: dict[tuple, torch.Tensor] = {}


def _get_gstate(num_groups: int, device: torch.device) -> torch.Tensor:
    key = (num_groups, device)
    cached = _gstate_cache.get(key)
    if cached is not None and cached.shape[0] >= num_groups * GSTATE_SIZE:
        return cached
    t = torch.zeros(num_groups * GSTATE_SIZE, dtype=torch.int32, device=device)
    _gstate_cache[key] = t
    return t


_dummy_state_cache: dict[torch.device, torch.Tensor] = {}


def _get_dummy_state(device: torch.device) -> torch.Tensor:
    cached = _dummy_state_cache.get(device)
    if cached is not None:
        return cached
    t = torch.zeros(1, dtype=torch.int32, device=device)
    _dummy_state_cache[device] = t
    return t


def _cute_topk_indices(
    x: torch.Tensor,
    k: int,
    sorted: bool = False,
    out_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Low-level entry returning only indices (``out_dtype``, int64 by default).

    Indices are ascending per row unless ``sorted`` reorders them by value.
    """
    if out_dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Unsupported out_dtype: {out_dtype}")
    if not x.is_cuda:
        raise ValueError("cute_topk requires a CUDA tensor")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported dtype: {x.dtype}")

    orig_shape = x.shape
    orig_dtype = x.dtype
    x2d = x.reshape(-1, orig_shape[-1])
    batch_size, n = x2d.shape
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")

    out_indices = torch.empty(batch_size, k, dtype=torch.int32, device=x.device)
    is_bf16 = orig_dtype == torch.bfloat16

    if is_bf16:
        x_input = x2d.contiguous()
        align = 2
    else:
        x_input = x2d.float().contiguous()
        align = 4

    # The row loads are 16-byte vectors, so every row must start 16-byte
    # aligned.  -inf pads never get selected: k <= n and ties keep the lowest
    # indices, so any real element wins over a pad.
    vec_elems = VEC_BF16 if is_bf16 else VEC_FP32
    pad = (-n) % vec_elems
    if pad:
        x_input = torch.nn.functional.pad(x_input, (0, pad), value=float("-inf"))
        n += pad
    elif x_input.data_ptr() % 16:
        x_input = x_input.clone()

    smem_cap = _compute_smem_cap(is_bf16)
    is_multi_cta = n > smem_cap

    if is_multi_cta:
        ctas_per_group = (n + smem_cap - 1) // smem_cap
        if ctas_per_group & 1:
            ctas_per_group += 1
        ctas_per_group = max(ctas_per_group, 2)
        if ctas_per_group > MAX_CTAS_PER_GROUP:
            raise ValueError(
                f"n ({n}) needs {ctas_per_group} CTAs per row, more than "
                f"MAX_CTAS_PER_GROUP={MAX_CTAS_PER_GROUP}"
            )

        chunk_size = (n + ctas_per_group - 1) // ctas_per_group
        chunk_size = ((chunk_size + vec_elems - 1) // vec_elems) * vec_elems
        chunk_size = min(chunk_size, smem_cap)

        num_sms = _get_num_sms()
        num_groups = min(num_sms // ctas_per_group, batch_size)
        num_groups = max(num_groups, 1)

        gstate = _get_gstate(num_groups, x.device)

        compiled = _compile(is_bf16, True)
        compiled(
            cute.runtime.from_dlpack(x_input, assumed_align=align, enable_tvm_ffi=True),
            cute.runtime.from_dlpack(out_indices, assumed_align=4, enable_tvm_ffi=True),
            cute.runtime.from_dlpack(gstate, assumed_align=4, enable_tvm_ffi=True),
            k,
            chunk_size,
            ctas_per_group,
            num_groups,
            batch_size,
        )
    else:
        dummy_state = _get_dummy_state(x.device)
        compiled = _compile(is_bf16, False)
        compiled(
            cute.runtime.from_dlpack(x_input, assumed_align=align, enable_tvm_ffi=True),
            cute.runtime.from_dlpack(out_indices, assumed_align=4, enable_tvm_ffi=True),
            cute.runtime.from_dlpack(dummy_state, assumed_align=4, enable_tvm_ffi=True),
            k,
            0,
            0,
            0,
            0,
        )

    if sorted:
        gathered_vals = torch.gather(
            x.reshape(-1, orig_shape[-1]).float(), -1, out_indices.long()
        )
        _, sort_perm = gathered_vals.sort(dim=-1, descending=True)
        out_indices = torch.gather(out_indices, -1, sort_perm)

    out_shape = (*orig_shape[:-1], k)
    return out_indices.to(out_dtype).reshape(out_shape)


# ====================================================================
# Public API
# ====================================================================


@lru_cache(maxsize=1)
def _load_cutedsl_topk() -> tuple[Callable | None, str | None]:
    try:
        _compile(False, False)
        return _cute_topk_indices, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def cutedsl_topk_available() -> bool:
    fn, _ = _load_cutedsl_topk()
    return fn is not None


def cutedsl_topk_error() -> str | None:
    _, error = _load_cutedsl_topk()
    return error


def cutedsl_topk(
    x: torch.Tensor,
    k: int,
    *,
    sorted: bool = False,
    out_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Return top-k **indices** along the last dimension.

    Each row's indices are ascending; ties at the k-th value keep the lowest
    indices, so the result is deterministic.  Value-sorted output is not
    supported; pass ``sorted=False`` (the default).  ``out_dtype`` is
    torch.int64 (default) or torch.int32.
    """
    if sorted:
        raise ValueError("cutedsl_topk does not support sorted=True")

    fn, error = _load_cutedsl_topk()
    if fn is None:
        raise RuntimeError(f"cutedsl topk is not available: {error}")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    with nvtx.add_nvtx_event(
        f"DSA/cutedsl_topk rows={x2d.shape[0]} cols={x2d.shape[1]} k={k}"
    ):
        indices = fn(x2d, k, sorted=False, out_dtype=out_dtype)

    out_shape = (*orig_shape[:-1], k)
    return indices.reshape(out_shape)
