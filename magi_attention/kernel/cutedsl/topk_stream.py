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

"""CuTe DSL streaming top-k for 16-bit rows (bf16 / fp16), one CTA per row.

Each row is read from global memory once. A sampled threshold filters it into a shared
buffer of candidates — one 32-bit word each, the order-preserving key of the value in
the high half and the low 16 bits of its index in the low half, so candidates stay in
index order — and a two-byte radix select finds the k-th key.

When a round of hits would overflow the buffer, the buffer is first cut to its top k
and the threshold becomes their minimum, compared with ``>`` from then on: an element
equal to it has a larger index than every kept tie, so it can never be selected. When
the sampled threshold turns out too high (fewer than k elements reach it), the row is
streamed again with ``t = -inf``.

Limits: ``k <= MAX_K`` leaves room for one full round of hits after a cut, and rows
have at most ``MAX_N = 2**21`` elements, the 32 segments that ``seg_start`` covers.
"""

from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from magi_attention.utils import nvtx

from .cutedsl_utils import (
    _asm,
    cp_async_16_zfill,
    cp_async_bulk_g2s,
    cp_async_bulk_prefetch_l2,
    cp_async_wait_all,
    ld_global_b32_pred,
    ld_shared_b32,
    ld_shared_u16,
    ld_shared_v4,
    mbar_arrive,
    mbar_arrive_expect_tx,
    mbar_init,
    mbar_wait,
    red_shared_add_u32,
    red_shared_min_u32,
    st_global_b32,
    st_global_b64,
    st_shared_b32,
)
from .dsl_compat import SmemAllocator

# ====================================================================
# Streaming layout and supported bounds
# ====================================================================

# Each ring stage holds one 8 KiB tile of 16-bit elements.
TILE_ELEMS = 4096
RING_STAGES = 4
# Tiles past the ring are prefetched into L2 this far ahead, so a stage refill reads L2
# instead of waiting on DRAM; the ring alone keeps only two rounds in flight.
L2_PREFETCH_TILES = 8

# A round uses 256 consumer threads: 128 per tile, 32 contiguous elements per thread.
CONSUMER_WARPS = 8
TILES_PER_ROUND = 2
CONSUMER_THREADS = 32 * CONSUMER_WARPS
THREADS_PER_TILE = CONSUMER_THREADS // TILES_PER_ROUND
ROUND_ELEMS = TILES_PER_ROUND * TILE_ELEMS

# Each candidate packs a 16-bit key and the low 16 index bits into one 32-bit word.
CAND_CAPACITY = 18432
INDEX_SEGMENT_ELEMS = 1 << 16
INDEX_SEGMENTS = 32
# After a cut to k entries, a full round of hits must still fit.
MAX_K = CAND_CAPACITY - ROUND_ELEMS
MAX_N = INDEX_SEGMENTS * INDEX_SEGMENT_ELEMS

# ====================================================================
# PTX helpers
# ====================================================================


def _cmp_nibble(op: str, ty: str):
    @dsl_user_op
    def cmp_nibble(
        w_lo: Int32, w_hi: Int32, thr_x2: Int32, *, loc=None, ip=None
    ) -> Int32:
        """Bit e (e < 4) is set when the e-th 16-bit float of (w_lo, w_hi) compares ``op``
        to the threshold replicated in both halves of ``thr_x2``. NaN compares false."""
        return Int32(
            _asm(
                T.i32(),
                [
                    Int32(w_lo).ir_value(loc=loc, ip=ip),
                    Int32(w_hi).ir_value(loc=loc, ip=ip),
                    Int32(thr_x2).ir_value(loc=loc, ip=ip),
                ],
                "{\n\t.reg .b32 r0, r1, ind;\n\t"
                # 0xFFFF per 16-bit half where the compare holds.
                f"set.{op}.s32.{ty}x2 r0, $1, $3;\n\t"
                f"set.{op}.s32.{ty}x2 r1, $2, $3;\n\t"
                # One signed byte (-1 or 0) per element, then the dot product with
                # (-1, -2, -4, -8) packs the four bytes into the 4-bit mask.
                "prmt.b32 ind, r0, r1, 0x7531;\n\t"
                "dp4a.s32.s32 $0, ind, 0xF8FCFEFF, 0;\n\t}",
                "=r,r,r,r",
                False,
                loc,
                ip,
            )
        )

    return cmp_nibble


# cmp_nibble per (compare, 16-bit float type): "ge" before a cut, "gt" after it.
_CMP = {(op, ty): _cmp_nibble(op, ty) for op in ("ge", "gt") for ty in ("bf16", "f16")}


@dsl_user_op
def to_key_x2(w: Int32, *, loc=None, ip=None) -> Int32:
    """``StreamTopK.to_key`` of each of the two 16-bit floats packed in ``w``."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(w).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 s, m;\n\t"
            "shr.u32 s, $1, 15;\n\t"  # sign bits to bits 0 and 16
            "and.b32 s, s, 0x00010001;\n\t"  # one 1 per negative half
            "mul.lo.u32 m, s, 0x7FFF;\n\t"  # 0x7FFF in each negative half
            "xor.b32 m, m, 0x80008000;\n\t"  # 0xFFFF if negative, else 0x8000
            "xor.b32 m, $1, m;\n\t"  # b | 0x8000 if positive, else ~b
            # ~b + 1 cannot carry into the other half: ~b is below 0x8000.
            "add.u32 $0, m, s;\n\t}",  # negative: ~b + 1 = 0x10000 - b
            "=r,r",
            False,
            loc,
            ip,
        )
    )


# ====================================================================
# Kernel
# ====================================================================

# Shared-storage layouts: 256 key-byte bins and an unused sink, padded to 16 B.
HIST_BINS = 256
HIST_SINK = HIST_BINS
HIST_SLOTS = HIST_BINS + 4
# The extra entry stays at SEG_NONE so a forward segment walk always stops.
SEG_TABLE_LEN = INDEX_SEGMENTS + 1
SEG_NONE = 0x7FFFFFFF

# Named barriers; barrier 0 is the CTA-wide one.
CONSUMER_BAR = 1  # the consumer threads
RESCAN_BAR = 2  # the consumers and producer warp, once per row with end > k


class SmemAddrs(NamedTuple):
    """Shared-memory addresses of the ``SharedStorage`` fields of ``StreamTopK.__call__``."""

    full_bar: Int32
    empty_bar: Int32
    warp_cnt: Int32
    scan2: Int32
    hist_hi: Int32
    hist_lo: Int32
    bucket: Int32
    count_gt: Int32
    rescan: Int32
    ring: Int32
    cand: Int32
    # seg_start of the candidate buffer, and the table a cut builds while seg_cur is read.
    seg_cur: Int32
    seg_new: Int32


class StreamTopK:
    """Streaming top-k kernel, specialized on the dtype tag, the output index width and
    whether ``mEnd`` holds a per-row length.

    A CTA has ``CONSUMER_THREADS + 32`` threads: warps ``0 .. CONSUMER_WARPS - 1`` sample
    the threshold, filter the ring tiles into the candidate buffer and select; warp
    ``CONSUMER_WARPS`` is the producer, whose lane 0 issues the bulk copies. A row with
    ``end <= k`` is written directly by the consumers.
    """

    def __init__(self, dtype_tag: str, out_i64: bool, has_end: bool):
        self.ty = dtype_tag
        neg_inf_bits = {"bf16": 0xFF80, "f16": 0xFC00}
        self.neg_inf = neg_inf_bits[dtype_tag]
        # Magnitude bits of +-inf: larger magnitudes are NaN.
        self.inf_abs = self.neg_inf & 0x7FFF
        self.out_i64 = out_i64
        self.has_end = has_end

    @cute.jit
    def __call__(
        self, mX: cute.Tensor, mEnd: cute.Tensor, mOut: cute.Tensor, k: Int32, stream
    ):
        @cute.struct
        class SharedStorage:
            # full_bar[s]: the bulk copy into ring stage s completed (one producer arrival
            # plus the copied bytes). empty_bar[s]: all consumer warps that read stage s
            # released it.
            full_bar: cute.struct.MemRange[Int64, RING_STAGES]
            empty_bar: cute.struct.MemRange[Int64, RING_STAGES]
            # Per-warp slots of block_sum and find_bucket (warp_cnt) and of block_sum2
            # (scan2); the block sums alternate between two slot sets.
            warp_cnt: cute.struct.Align[
                cute.struct.MemRange[Int32, 2 * CONSUMER_WARPS], 16
            ]
            scan2: cute.struct.Align[
                cute.struct.MemRange[Int32, 4 * CONSUMER_WARPS], 16
            ]
            # Histograms of the high and of the low key byte.
            hist_hi: cute.struct.Align[cute.struct.MemRange[Int32, HIST_SLOTS], 16]
            hist_lo: cute.struct.Align[cute.struct.MemRange[Int32, HIST_SLOTS], 16]
            # seg_start of the buffer, then the table a cut rebuilds.
            seg: cute.struct.Align[cute.struct.MemRange[Int32, 2 * SEG_TABLE_LEN], 16]
            # Results published by find_bucket and the first filter pass.
            bucket: cute.struct.Align[cute.struct.MemRange[Int32, 1], 16]
            count_gt: cute.struct.MemRange[Int32, 1]
            rescan: cute.struct.MemRange[Int32, 1]
            # The ring; the final take_top also stages output indices here.
            ring: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int16, RING_STAGES * TILE_ELEMS], 128
            ]
            # Candidate words; the threshold sample is staged here before the filter.
            cand: cute.struct.Align[cute.struct.MemRange[Int32, CAND_CAPACITY], 128]

        self.shared_storage = SharedStorage
        self.kernel(mX, mEnd, mOut, k).launch(
            grid=[cute.size(mX.shape[0]), 1, 1],
            block=[CONSUMER_THREADS + 32, 1, 1],
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            # Two CTAs per SM (the register cap and the shared-memory carveout are sized
            # for two): the bulk copies of one CTA overlap the select step of the other.
            min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mEnd: cute.Tensor, mOut: cute.Tensor, k: Int32):
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        row = cute.arch.block_idx()[0]
        n = Int32(mX.shape[1])

        allocator = SmemAllocator()
        storage = allocator.allocate(self.shared_storage)
        seg_cur = Int32(storage.seg.data_ptr().toint())
        smem = SmemAddrs(
            full_bar=Int32(storage.full_bar.data_ptr().toint()),
            empty_bar=Int32(storage.empty_bar.data_ptr().toint()),
            warp_cnt=Int32(storage.warp_cnt.data_ptr().toint()),
            scan2=Int32(storage.scan2.data_ptr().toint()),
            hist_hi=Int32(storage.hist_hi.data_ptr().toint()),
            hist_lo=Int32(storage.hist_lo.data_ptr().toint()),
            bucket=Int32(storage.bucket.data_ptr().toint()),
            count_gt=Int32(storage.count_gt.data_ptr().toint()),
            rescan=Int32(storage.rescan.data_ptr().toint()),
            ring=Int32(storage.ring.data_ptr().toint()),
            cand=Int32(storage.cand.data_ptr().toint()),
            seg_cur=seg_cur,
            seg_new=seg_cur + SEG_TABLE_LEN * 4,
        )

        end = n
        if const_expr(self.has_end):
            end = cutlass.max(cutlass.min(mEnd[row], n), Int32(0))
        num_tiles = (end + (TILE_ELEMS - 1)) // TILE_ELEMS
        row_ptr = Int64(mX.iterator.toint()) + Int64(row) * Int64(n) * Int64(2)
        out_row = Int64(mOut.iterator.toint()) + Int64(row) * Int64(
            mOut.stride[0]
        ) * Int64(8 if const_expr(self.out_i64) else 4)

        if tidx == 0:
            for s in cutlass.range_constexpr(RING_STAGES):
                # full_bar: the producer's arrive_expect_tx; empty_bar: one arrival per
                # consumer warp that reads the stage.
                mbar_init(smem.full_bar + 8 * s, 1)
                mbar_init(smem.empty_bar + 8 * s, THREADS_PER_TILE // 32)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if end > k:
            if warp == CONSUMER_WARPS:
                if lane == 0:
                    self.produce(smem, row_ptr, end, num_tiles, Int32(0))
                # At the end of pass 0 the consumers publish whether the row is
                # streamed again.
                cute.arch.barrier(
                    barrier_id=RESCAN_BAR, number_of_threads=CONSUMER_THREADS + 32
                )
                if lane == 0:
                    if ld_shared_b32(smem.rescan) != 0:
                        self.produce(smem, row_ptr, end, num_tiles, num_tiles)
            else:
                self.consume(
                    out_row,
                    row_ptr,
                    end,
                    num_tiles,
                    k,
                    smem,
                    tidx,
                    lane,
                    warp,
                )
        else:
            # Every selectable element of a short row is in its top k.
            if warp < CONSUMER_WARPS:
                self.emit_short_row(
                    out_row, row_ptr, end, k, smem.warp_cnt, tidx, lane, warp
                )

    @cute.jit
    def produce(
        self,
        smem: SmemAddrs,
        row_ptr: Int64,
        end: Int32,
        num_tiles: Int32,
        tile_seq_base: Int32,
    ):
        """Stream tiles ``0 .. num_tiles - 1`` of the row through the ring; called by lane 0
        of the producer warp.

        Tile t is the ``tile_seq_base + t``-th tile through the ring. Its stage is that
        sequence number modulo ``RING_STAGES``, and lap ``L = tile_seq // RING_STAGES`` of
        a stage completes phase ``L & 1`` of its ``full_bar`` and ``empty_bar``. A second
        pass starts at ``tile_seq_base = num_tiles``.
        """
        # Prefetch the tiles after the ring's first fill, then stay L2_PREFETCH_TILES
        # ahead of the one being copied.
        prefetch_tile = Int32(RING_STAGES)
        while prefetch_tile < cutlass.min(num_tiles, Int32(L2_PREFETCH_TILES)):
            cp_async_bulk_prefetch_l2(
                row_ptr + Int64(prefetch_tile) * Int64(TILE_ELEMS * 2),
                self.tile_bytes(end, prefetch_tile),
            )
            prefetch_tile = prefetch_tile + 1
        t = Int32(0)
        while t < num_tiles:
            tile_seq = tile_seq_base + t
            stage = tile_seq % RING_STAGES
            prefetch_tile = t + L2_PREFETCH_TILES
            if prefetch_tile < num_tiles:
                cp_async_bulk_prefetch_l2(
                    row_ptr + Int64(prefetch_tile) * Int64(TILE_ELEMS * 2),
                    self.tile_bytes(end, prefetch_tile),
                )
            if tile_seq >= RING_STAGES:
                # The readers of the stage's previous tile must release it first.
                mbar_wait(
                    smem.empty_bar + 8 * stage,
                    (tile_seq // RING_STAGES - 1) & 1,
                )
            nbytes = self.tile_bytes(end, t)
            mbar_arrive_expect_tx(smem.full_bar + 8 * stage, nbytes)
            cp_async_bulk_g2s(
                smem.ring + stage * (TILE_ELEMS * 2),
                row_ptr + Int64(t) * Int64(TILE_ELEMS * 2),
                nbytes,
                smem.full_bar + 8 * stage,
            )
            t = t + 1

    @cute.jit
    def consume(
        self,
        out_row: Int64,
        row_ptr: Int64,
        end: Int32,
        num_tiles: Int32,
        k: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ):
        """Threshold, filter and select for a row with ``end > k``; called by all consumer
        threads."""
        thr_bits = self.sample_threshold(row_ptr, end, k, smem, tidx, lane, warp)
        num_cand = self.filter_row(
            out_row, end, num_tiles, k, thr_bits, smem, tidx, lane, warp
        )
        self.select_and_store(out_row, num_cand, k, smem, tidx, lane, warp)

    @cute.jit
    def sample_threshold(
        self,
        row_ptr: Int64,
        end: Int32,
        k: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ) -> Int32:
        """Bits of the filter threshold, or of -inf when the sample holds fewer values
        than the rank it needs; called by all consumer threads."""
        sample_block_elems = const_expr(64)
        min_sample_stride = const_expr(32 * sample_block_elems)
        max_sample_blocks = const_expr(512)
        # Fixed-point c <= 1.5, targeting at most 717/1024 of candidate capacity.
        max_oversample_x1024 = const_expr(1536)
        cand_target_fraction_x1024 = const_expr(717)
        assert const_expr(
            max_sample_blocks * sample_block_elems * 2 <= CAND_CAPACITY * 4
        ), "the threshold sample must fit in the candidate buffer"

        # Use complete sample blocks, spaced by a block-aligned stride.
        stride = cutlass.max(
            Int32(min_sample_stride),
            (
                (end + (max_sample_blocks - 1)) // max_sample_blocks
                + (sample_block_elems - 1)
            )
            // sample_block_elems
            * sample_block_elems,
        )
        num_sample_blocks = Int32(0)
        if end >= sample_block_elems:
            num_sample_blocks = (end - sample_block_elems) // stride + 1
        thr_bits = Int32(self.neg_inf)
        num_samples = num_sample_blocks * sample_block_elems
        if num_sample_blocks > 0:
            # Each step of a thread copies 16 B (8 elements) of a sample block.
            c = tidx
            while c < num_sample_blocks * (sample_block_elems // 8):
                blk = c // (sample_block_elems // 8)
                cp_async_16_zfill(
                    smem.cand + c * 16,
                    row_ptr
                    + (
                        Int64(blk) * Int64(stride)
                        + Int64((c % (sample_block_elems // 8)) * 8)
                    )
                    * Int64(2),
                    Int32(0),
                )
                c = c + CONSUMER_THREADS
            oversample_x1024 = cutlass.min(
                Int32(max_oversample_x1024),
                Int32(CAND_CAPACITY * cand_target_fraction_x1024) // k,
            )
            # sample_rank = ceil(oversample_x1024 / 1024 * k * num_samples / end)
            sample_rank = Int32(
                (
                    Int64(num_samples) * Int64(oversample_x1024) * Int64(k)
                    + Int64(end) * 1024
                    - 1
                )
                // (Int64(end) * 1024)
            )
            # The filter writes candidates over the sample, so its copies must land even
            # when the sample goes unused.
            cp_async_wait_all()
            cute.arch.barrier(
                barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS
            )
            if sample_rank <= num_samples:
                pivot_key, _ = self.select_pivot(
                    True,
                    False,
                    smem.cand,
                    num_samples,
                    sample_rank,
                    smem,
                    tidx,
                    lane,
                    warp,
                )
                thr_bits = self.from_key(pivot_key)
        return thr_bits

    @cute.jit
    def filter_row(
        self,
        out_row: Int64,
        end: Int32,
        num_tiles: Int32,
        k: Int32,
        thr_bits: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ) -> Int32:
        """Append every element of the row that passes the threshold to the candidate
        buffer; returns the candidate count. Called by all consumer threads.

        In round r, thread ``tidx`` reads 32 elements of tile ``2 * r + tidx // 128``,
        starting at ``32 * (tidx % 128)``. A warp releases the tile's stage after it
        appended its hits. Pass 0 ends with the rescan decision, which the producer warp
        waits for.
        """
        elems_per_thread = const_expr(ROUND_ELEMS // CONSUMER_THREADS)
        assert const_expr(
            elems_per_thread == 32
        ), "a thread's hit mask is one 32-bit word"
        assert const_expr(
            INDEX_SEGMENT_ELEMS % ROUND_ELEMS == 0
        ), "a round must lie inside one index segment"
        self.reset_filter_state(smem.hist_hi, smem.seg_cur, tidx)
        thr_x2 = thr_bits | (thr_bits << 16)
        strict_gt = Int32(0)
        num_cand = Int32(0)
        was_cut = Int32(0)
        parity = Int32(0)
        tile_half = tidx // THREADS_PER_TILE
        tidx_in_tile = tidx % THREADS_PER_TILE
        num_rounds = (end + (ROUND_ELEMS - 1)) // ROUND_ELEMS
        # Byte offsets and hit-mask shifts of the thread's four 16 B chunks, in a
        # lane-rotated order so the lanes of a warp read distinct banks.
        chunk_offs = tuple(
            tidx_in_tile * (elems_per_thread * 2) + ((lane + c) & 3) * 16
            for c in range(4)
        )
        chunk_shifts = tuple(((lane + c) & 3) * 8 for c in range(4))

        elem0 = tidx_in_tile * elems_per_thread
        # Elements of this thread that lie inside the row in the last round.
        tail_valid = (
            end - (TILES_PER_ROUND * (num_rounds - 1) + tile_half) * TILE_ELEMS - elem0
        )
        tail_mask = Int32(-1)
        if tail_valid < elems_per_thread:
            tail_mask = Int32(0)
            if tail_valid > 0:
                tail_mask = (Int32(1) << tail_valid) - 1

        pass_idx = Int32(0)
        num_passes = Int32(1)
        while pass_idx < num_passes:
            tile_seq_base = pass_idx * num_tiles
            round_idx = Int32(0)
            while round_idx < num_rounds:
                # The inner loop stops at a round whose hits would overflow the buffer;
                # the buffer is then cut to its top k and that round is filtered again.
                stop_round = num_rounds
                while round_idx < stop_round:
                    tile = TILES_PER_ROUND * round_idx + tile_half
                    tile_seq = tile_seq_base + tile
                    stage = tile_seq % RING_STAGES
                    tile_buf = smem.ring + stage * (TILE_ELEMS * 2)
                    # The second tile of the last round can lie past the row.
                    live = tile < num_tiles
                    if tidx == 0:
                        if (round_idx * ROUND_ELEMS) % INDEX_SEGMENT_ELEMS == 0:
                            st_shared_b32(
                                smem.seg_cur
                                + (round_idx * ROUND_ELEMS // INDEX_SEGMENT_ELEMS) * 4,
                                num_cand,
                            )
                    if live:
                        mbar_wait(
                            smem.full_bar + 8 * stage,
                            (tile_seq // RING_STAGES) & 1,
                        )
                    valid_mask = Int32(-1)
                    if round_idx == num_rounds - 1:
                        valid_mask = tail_mask
                    mask = Int32(0)
                    if live:
                        mask = (
                            self.hit_mask(
                                tile_buf, chunk_offs, chunk_shifts, thr_x2, strict_gt
                            )
                            & valid_mask
                        )
                    hit_off, num_hits = self.block_sum(
                        cute.arch.popc(mask), lane, warp, smem.warp_cnt, parity
                    )
                    parity = parity ^ 1
                    if num_cand + num_hits > CAND_CAPACITY:
                        # The stage stays held: the round is filtered again after the cut.
                        stop_round = round_idx
                    else:
                        slot = num_cand + hit_off
                        m = mask
                        # A round lies inside one segment, so only the low index bits vary.
                        idx_lo0 = (tile * TILE_ELEMS + elem0) % INDEX_SEGMENT_ELEMS
                        elem_buf = tile_buf + elem0 * 2
                        while m != 0:
                            e = cute.arch.popc((m & (-m)) - 1)
                            m = m & (m - 1)
                            key = self.to_key(ld_shared_u16(elem_buf + e * 2))
                            st_shared_b32(
                                smem.cand + slot * 4, (key << 16) | (idx_lo0 + e)
                            )
                            red_shared_add_u32(smem.hist_hi + (key >> 8) * 4, Int32(1))
                            slot = slot + 1
                        # Every lane has read the stage before lane 0 releases it.
                        cute.arch.sync_warp()
                        if lane == 0 and live:
                            mbar_arrive(smem.empty_bar + 8 * stage)
                        num_cand = num_cand + num_hits
                        round_idx = round_idx + 1
                if stop_round < num_rounds:
                    pivot_key, num_ties = self.select_pivot(
                        False, True, smem.cand, num_cand, k, smem, tidx, lane, warp
                    )
                    num_cand = self.take_top(
                        False,
                        out_row,
                        num_cand,
                        pivot_key,
                        num_ties,
                        round_idx * ROUND_ELEMS // INDEX_SEGMENT_ELEMS,
                        smem,
                        tidx,
                        lane,
                        warp,
                    )
                    thr_bits = self.from_key(pivot_key)
                    thr_x2 = thr_bits | (thr_bits << 16)
                    strict_gt = Int32(1)
                    was_cut = Int32(1)
            if pass_idx == 0:
                # The sampled threshold was too high: fewer than k elements reached it,
                # no cut happened and it lies above -inf, so stream the row again with
                # t = -inf. All consumers evaluate the same inputs.
                rescan = Int32(0)
                if num_cand < k and was_cut == 0 and thr_bits != self.neg_inf:
                    rescan = Int32(1)
                if tidx == 0:
                    st_shared_b32(smem.rescan, rescan)
                cute.arch.barrier(
                    barrier_id=RESCAN_BAR, number_of_threads=CONSUMER_THREADS + 32
                )
                if rescan != 0:
                    num_passes = Int32(2)
                    thr_bits = Int32(self.neg_inf)
                    thr_x2 = thr_bits | (thr_bits << 16)
                    strict_gt = Int32(0)
                    num_cand = Int32(0)
                    self.reset_filter_state(smem.hist_hi, smem.seg_cur, tidx)
            pass_idx = pass_idx + 1
        return num_cand

    @cute.jit
    def select_and_store(
        self,
        out_row: Int64,
        num_cand: Int32,
        k: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ):
        """Store the indices of the top k candidates ascending, and pad the row with -1
        up to k; called by all consumer threads."""
        # The candidate words and histogram counts of the last round must be visible to
        # every consumer.
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        # With at most k candidates the pivot -1 keeps them all: every key is >= 0.
        pivot_key = Int32(-1)
        num_ties = Int32(0)
        if num_cand > k:
            pivot_key, num_ties = self.select_pivot(
                False, True, smem.cand, num_cand, k, smem, tidx, lane, warp
            )
        kept = self.take_top(
            True,
            out_row,
            num_cand,
            pivot_key,
            num_ties,
            Int32(0),
            smem,
            tidx,
            lane,
            warp,
        )
        i = kept + tidx
        while i < k:
            self.store_out(out_row, i, Int32(-1))
            i = i + CONSUMER_THREADS

    @cute.jit
    def emit_short_row(
        self,
        out_row: Int64,
        row_ptr: Int64,
        end: Int32,
        k: Int32,
        warp_cnt: Int32,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ):
        """Store every non-NaN index of ``[0, end)`` ascending and pad the tail with -1,
        for a row with ``end <= k``; called by all consumer threads."""
        written = Int32(0)
        parity = Int32(0)
        base = Int32(0)
        while base < end:
            i = base + tidx
            live = i < end
            # The 32-bit word holding element i; rows start 16 B aligned.
            w = ld_global_b32_pred(row_ptr + Int64(i >> 1) * 4, live)
            hit = Int32(0)
            if live:
                if ((w >> ((i & 1) * 16)) & 0x7FFF) <= self.inf_abs:
                    hit = Int32(1)
            hit_off, num_hits = self.block_sum(hit, lane, warp, warp_cnt, parity)
            parity = parity ^ 1
            if hit != 0:
                self.store_out(out_row, written + hit_off, i)
            written = written + num_hits
            base = base + CONSUMER_THREADS
        i = written + tidx
        while i < k:
            self.store_out(out_row, i, Int32(-1))
            i = i + CONSUMER_THREADS

    @cute.jit
    def tile_bytes(self, end: Int32, t: Int32) -> Int32:
        """Bytes of tile ``t`` up to ``end``, rounded up to 16; rows are padded to 8
        elements, so the rounding stays inside the row."""
        return (
            (cutlass.min(end - t * TILE_ELEMS, Int32(TILE_ELEMS)) * 2 + 15) // 16 * 16
        )

    @cute.jit
    def to_key(self, b: Int32) -> Int32:
        """Order-preserving unsigned key of the 16-bit float bits ``b``. Negative values map
        to the two's complement of their bits, so -0 and +0 share the key 0x8000 and the
        radix order agrees with the float compares."""
        key = b | 0x8000
        if (b & 0x8000) != 0:
            key = (0x10000 - b) & 0xFFFF
        return key

    @cute.jit
    def from_key(self, key: Int32) -> Int32:
        """The 16-bit float bits of ``key``; +0 for the key 0x8000."""
        b = (0x10000 - key) & 0xFFFF
        if (key & 0x8000) != 0:
            b = key & 0x7FFF
        return b

    @cute.jit
    def store_out(self, out_row: Int64, i: Int32, v: Int32):
        if const_expr(self.out_i64):
            st_global_b64(out_row + Int64(i) * 8, Int64(v))
        else:
            st_global_b32(out_row + Int64(i) * 4, v)

    @cute.jit
    def reset_filter_state(self, hist_hi: Int32, seg_cur: Int32, tidx: Int32):
        """Empty the high-byte histogram and the segment table of the candidate buffer.
        Called by all consumer threads before a filter pass."""
        st_shared_b32(hist_hi + tidx * 4, Int32(0))
        if tidx == 0:
            st_shared_b32(hist_hi + HIST_SINK * 4, Int32(0))
        if tidx < SEG_TABLE_LEN:
            st_shared_b32(seg_cur + tidx * 4, Int32(SEG_NONE))

    @cute.jit
    def seg_of(self, seg: Int32, slot: Int32) -> Int32:
        """Segment of candidate ``slot``: the largest h with ``seg[h] <= slot``.
        ``seg[0] == 0`` whenever the buffer is non-empty."""
        assert const_expr(
            INDEX_SEGMENTS == 32
        ), "segment search uses five halving steps"
        h = Int32(0)
        for i in cutlass.range_constexpr(5):
            if ld_shared_b32(seg + (h + (16 >> i)) * 4) <= slot:
                h = h + (16 >> i)
        return h

    @cute.jit
    def hit_mask(
        self, buf: Int32, chunk_offs, chunk_shifts, thr_x2: Int32, strict_gt: Int32
    ) -> Int32:
        """Bit e is set when element e of this thread's 32 in the tile at ``buf`` passes
        the threshold: ``>`` when ``strict_gt``, else ``>=``."""
        mask = Int32(0)
        if strict_gt != 0:
            f = _CMP[("gt", self.ty)]
            for c in cutlass.range_constexpr(4):
                w = ld_shared_v4(buf + chunk_offs[c])
                mask = mask | (
                    (f(w[0], w[1], thr_x2) | (f(w[2], w[3], thr_x2) << 4))
                    << chunk_shifts[c]
                )
        else:
            f = _CMP[("ge", self.ty)]
            for c in cutlass.range_constexpr(4):
                w = ld_shared_v4(buf + chunk_offs[c])
                mask = mask | (
                    (f(w[0], w[1], thr_x2) | (f(w[2], w[3], thr_x2) << 4))
                    << chunk_shifts[c]
                )
        return mask

    @cute.jit
    def block_sum(self, v: Int32, lane: Int32, warp: Int32, buf: Int32, parity: Int32):
        """Exclusive prefix sum and total of ``v`` over the consumer threads in thread
        order. Called by all consumer threads; one ``CONSUMER_BAR`` barrier.

        ``parity`` selects one of two slot sets of ``buf``. Consecutive calls alternate
        it, so a call never writes the slots that a slower thread still reads from the
        previous call, and no second barrier is needed.
        """
        assert const_expr(CONSUMER_WARPS == 8), "warp totals use three shuffle steps"
        incl = v
        for s in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(incl, 1 << s, mask_and_clamp=0)
            if lane >= (1 << s):
                incl = incl + up
        if lane == 31:
            st_shared_b32(buf + (parity * CONSUMER_WARPS + warp) * 4, incl)
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        # Lane l of every warp gets the total of warps 0 .. l % CONSUMER_WARPS.
        sub = lane & (CONSUMER_WARPS - 1)
        wv = ld_shared_b32(buf + (parity * CONSUMER_WARPS + sub) * 4)
        for s in cutlass.range_constexpr(3):
            up = cute.arch.shuffle_sync_up(wv, 1 << s, mask_and_clamp=0)
            if sub >= (1 << s):
                wv = wv + up
        tot = cute.arch.shuffle_sync(wv, CONSUMER_WARPS - 1)
        prev = cute.arch.shuffle_sync(
            wv, (warp + CONSUMER_WARPS - 1) & (CONSUMER_WARPS - 1)
        )
        woff = Int32(0)
        if warp > 0:
            woff = prev
        return woff + incl - v, tot

    @cute.jit
    def block_sum2(
        self, a: Int32, b: Int32, lane: Int32, warp: Int32, buf: Int32, parity: Int32
    ):
        """``block_sum`` of two counts at once: prefix of ``a``, prefix of ``b``, total of
        ``a``, total of ``b``."""
        assert const_expr(CONSUMER_WARPS == 8), "warp totals use three shuffle steps"
        ia = a
        ib = b
        for s in cutlass.range_constexpr(5):
            pa = cute.arch.shuffle_sync_up(ia, 1 << s, mask_and_clamp=0)
            pb = cute.arch.shuffle_sync_up(ib, 1 << s, mask_and_clamp=0)
            if lane >= (1 << s):
                ia = ia + pa
                ib = ib + pb
        base = buf + parity * (2 * CONSUMER_WARPS) * 4
        if lane == 31:
            st_shared_b32(base + warp * 4, ia)
            st_shared_b32(base + (CONSUMER_WARPS + warp) * 4, ib)
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        sub = lane & (CONSUMER_WARPS - 1)
        wa = ld_shared_b32(base + sub * 4)
        wb = ld_shared_b32(base + (CONSUMER_WARPS + sub) * 4)
        for s in cutlass.range_constexpr(3):
            ua = cute.arch.shuffle_sync_up(wa, 1 << s, mask_and_clamp=0)
            ub = cute.arch.shuffle_sync_up(wb, 1 << s, mask_and_clamp=0)
            if sub >= (1 << s):
                wa = wa + ua
                wb = wb + ub
        ta = cute.arch.shuffle_sync(wa, CONSUMER_WARPS - 1)
        tb = cute.arch.shuffle_sync(wb, CONSUMER_WARPS - 1)
        src = (warp + CONSUMER_WARPS - 1) & (CONSUMER_WARPS - 1)
        pa = cute.arch.shuffle_sync(wa, src)
        pb = cute.arch.shuffle_sync(wb, src)
        oa = Int32(0)
        ob = Int32(0)
        if warp > 0:
            oa = pa
            ob = pb
        return oa + ia - a, ob + ib - b, ta, tb

    @cute.jit
    def find_bucket(
        self,
        hist: Int32,
        bucket: Int32,
        count_gt: Int32,
        wbuf: Int32,
        rank: Int32,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ):
        """The bin b of a ``HIST_BINS``-bin histogram with ``count(> b) < rank <=
        count(>= b)``, and ``count(> b)``; requires ``1 <= rank <= count(>= 0)``. Called by
        all consumer threads, thread t reading bin t; the result is broadcast through
        ``bucket`` and ``count_gt``. Two ``CONSUMER_BAR`` barriers."""
        assert const_expr(
            CONSUMER_THREADS == HIST_BINS
        ), "consumer thread t owns histogram bin t"
        bin_cnt = ld_shared_b32(hist + tidx * 4)
        # cnt_ge = count(>= this bin): suffix sum within the warp, then over later warps.
        cnt_ge = bin_cnt
        for i in cutlass.range_constexpr(5):
            d = cute.arch.shuffle_sync_down(cnt_ge, 1 << i, mask_and_clamp=31)
            if lane + (1 << i) < 32:
                cnt_ge = cnt_ge + d
        if lane == 0:
            st_shared_b32(wbuf + warp * 4, cnt_ge)
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        for w_i in cutlass.range_constexpr(CONSUMER_WARPS):
            if w_i > warp:
                cnt_ge = cnt_ge + ld_shared_b32(wbuf + w_i * 4)
        cnt_gt = cnt_ge - bin_cnt
        if cnt_ge >= rank and cnt_gt < rank:
            st_shared_b32(bucket, tidx)
            st_shared_b32(count_gt, cnt_gt)
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        return (
            ld_shared_b32(bucket),
            ld_shared_b32(count_gt),
        )

    @cute.jit
    def hist_pass(
        self,
        low_byte: cutlass.Constexpr[bool],
        raw16: cutlass.Constexpr[bool],
        vals: Int32,
        cnt: Int32,
        prefix: Int32,
        hist: Int32,
        tidx: Int32,
    ):
        """Histogram of the high key byte, or (``low_byte``) of the low key byte among the
        keys whose high byte is ``prefix``, over the ``cnt`` values at ``vals``: 16-bit
        floats (``raw16``) or candidate words. Values past ``cnt`` and keys outside
        ``prefix`` go to the sink bin ``HIST_SINK``, which no reader uses. Called by all
        consumer threads; begins and ends with a ``CONSUMER_BAR`` barrier."""
        st_shared_b32(hist + tidx * 4, Int32(0))
        if tidx == 0:
            st_shared_b32(hist + HIST_SINK * 4, Int32(0))
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)
        per = 8 if const_expr(raw16) else 4
        off = tidx * per
        while off < cnt:
            if const_expr(raw16):
                w = ld_shared_v4(vals + off * 2)
            else:
                w = ld_shared_v4(vals + off * 4)
            for j in cutlass.range_constexpr(4):
                halves = 2 if const_expr(raw16) else 1
                if const_expr(raw16):
                    key2 = to_key_x2(w[j])
                else:
                    key2 = w[j]
                for h in cutlass.range_constexpr(halves):
                    if const_expr(raw16):
                        key = (key2 >> (16 * h)) & 0xFFFF
                        slot = off + 2 * j + h
                    else:
                        key = (key2 >> 16) & 0xFFFF
                        slot = off + j
                    bin_ = Int32(HIST_SINK)
                    if slot < cnt:
                        if const_expr(low_byte):
                            if (key >> 8) == prefix:
                                bin_ = key & 0xFF
                        else:
                            bin_ = key >> 8
                    red_shared_add_u32(hist + bin_ * 4, Int32(1))
            off = off + CONSUMER_THREADS * per
        cute.arch.barrier(barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS)

    @cute.jit
    def select_pivot(
        self,
        raw16: cutlass.Constexpr[bool],
        have_hi: cutlass.Constexpr[bool],
        vals: Int32,
        cnt: Int32,
        rank: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ):
        """Key of the ``rank``-th largest of the ``cnt`` values at ``vals``, and how many
        values with that key the top ``rank`` takes; requires ``1 <= rank <= cnt``.
        ``have_hi``: ``smem.hist_hi`` already holds the high-byte histogram of the values,
        which the filter keeps up to date for the candidate buffer. Called by all consumer
        threads."""
        if const_expr(not have_hi):
            self.hist_pass(False, raw16, vals, cnt, Int32(0), smem.hist_hi, tidx)
        key_hi, cnt_gt_hi = self.find_bucket(
            smem.hist_hi,
            smem.bucket,
            smem.count_gt,
            smem.warp_cnt,
            rank,
            tidx,
            lane,
            warp,
        )
        rank_in_hi = rank - cnt_gt_hi
        self.hist_pass(True, raw16, vals, cnt, key_hi, smem.hist_lo, tidx)
        key_lo, cnt_gt_lo = self.find_bucket(
            smem.hist_lo,
            smem.bucket,
            smem.count_gt,
            smem.warp_cnt,
            rank_in_hi,
            tidx,
            lane,
            warp,
        )
        return (key_hi << 8) | key_lo, rank_in_hi - cnt_gt_lo

    @cute.jit
    def take_top(
        self,
        to_gmem: cutlass.Constexpr[bool],
        out_row: Int64,
        num_cand: Int32,
        pivot_key: Int32,
        num_ties: Int32,
        seg_now: Int32,
        smem: SmemAddrs,
        tidx: Int32,
        lane: Int32,
        warp: Int32,
    ) -> Int32:
        """Keep the candidates with key > ``pivot_key`` and the first ``num_ties`` with key
        == ``pivot_key``, in buffer order; returns how many were kept. Called by all
        consumer threads.

        ``to_gmem``: the kept indices go to ``out_row``, staged in the ring (free in the
        select step) so the global stores coalesce. Otherwise the buffer is compacted in
        place and its high-byte histogram and segment table are rebuilt; ``seg_now`` is
        the segment of the round being filtered, whose hits follow the kept entries. Each
        chunk is read before the barrier in ``block_sum2`` and written after it, and an
        entry never moves past its own slot, so the in-place compaction is safe.
        """
        keep_per_thread = const_expr(16)
        keep_chunk = const_expr(CONSUMER_THREADS * keep_per_thread)
        assert const_expr(
            keep_chunk * 4 <= RING_STAGES * TILE_ELEMS * 2
        ), "a chunk's output indices must fit in the ring"
        if const_expr(not to_gmem):
            st_shared_b32(smem.hist_hi + tidx * 4, Int32(0))
            if tidx < SEG_TABLE_LEN:
                st_shared_b32(smem.seg_new + tidx * 4, Int32(SEG_NONE))
        gt_before = Int32(0)
        eq_before = Int32(0)
        parity = Int32(0)
        chunk_start = Int32(0)
        while chunk_start < num_cand:
            slot0 = chunk_start + tidx * keep_per_thread
            words = cute.make_rmem_tensor((keep_per_thread,), Int32)
            segs = cute.make_rmem_tensor((keep_per_thread,), Int32)
            num_gt = Int32(0)
            num_eq = Int32(0)
            if slot0 < num_cand:
                for q in cutlass.range_constexpr(keep_per_thread // 4):
                    wv = ld_shared_v4(smem.cand + (slot0 + 4 * q) * 4)
                    for j in cutlass.range_constexpr(4):
                        words[4 * q + j] = wv[j]
                # Segment of each word: search the table for the first, then walk on.
                seg = self.seg_of(smem.seg_cur, slot0)
                next_start = ld_shared_b32(smem.seg_cur + (seg + 1) * 4)
                for j in cutlass.range_constexpr(keep_per_thread):
                    while next_start <= slot0 + j:
                        seg = seg + 1
                        next_start = ld_shared_b32(smem.seg_cur + (seg + 1) * 4)
                    segs[j] = seg
            for j in cutlass.range_constexpr(keep_per_thread):
                key = Int32(-1)
                if slot0 + j < num_cand:
                    key = (words[j] >> 16) & 0xFFFF
                if key > pivot_key:
                    num_gt = num_gt + 1
                if key == pivot_key:
                    num_eq = num_eq + 1
            gt_off, eq_off, gt_total, eq_total = self.block_sum2(
                num_gt, num_eq, lane, warp, smem.scan2, parity
            )
            gt_rank = gt_before + gt_off
            eq_rank = eq_before + eq_off
            # Output position of the chunk's first kept entry.
            out0 = gt_before + cutlass.min(eq_before, num_ties)
            for j in cutlass.range_constexpr(keep_per_thread):
                key = Int32(-1)
                if slot0 + j < num_cand:
                    key = (words[j] >> 16) & 0xFFFF
                pos = Int32(-1)
                if key > pivot_key:
                    pos = gt_rank + cutlass.min(eq_rank, num_ties)
                    gt_rank = gt_rank + 1
                if key == pivot_key:
                    if eq_rank < num_ties:
                        pos = gt_rank + eq_rank
                    eq_rank = eq_rank + 1
                if pos >= 0:
                    if const_expr(to_gmem):
                        st_shared_b32(
                            smem.ring + (pos - out0) * 4,
                            (segs[j] * INDEX_SEGMENT_ELEMS) | (words[j] & 0xFFFF),
                        )
                    else:
                        st_shared_b32(smem.cand + pos * 4, words[j])
                        red_shared_min_u32(smem.seg_new + segs[j] * 4, pos)
                        red_shared_add_u32(
                            smem.hist_hi + ((words[j] >> 24) & 0xFF) * 4, Int32(1)
                        )
            if const_expr(to_gmem):
                # The next chunk writes the staging area only after its block_sum2
                # barrier, which every thread reaches after these stores.
                cute.arch.barrier(
                    barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS
                )
                kept = (
                    gt_total
                    + cutlass.min(eq_before + eq_total, num_ties)
                    - cutlass.min(eq_before, num_ties)
                )
                i = tidx
                while i < kept:
                    self.store_out(out_row, out0 + i, ld_shared_b32(smem.ring + i * 4))
                    i = i + CONSUMER_THREADS
            gt_before = gt_before + gt_total
            eq_before = eq_before + eq_total
            parity = parity ^ 1
            chunk_start = chunk_start + keep_chunk
        total = gt_before + cutlass.min(eq_before, num_ties)
        if const_expr(not to_gmem):
            cute.arch.barrier(
                barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS
            )
            assert const_expr(
                INDEX_SEGMENTS == 32
            ), "one warp rebuilds the segment table"
            if warp == 0:
                # seg_start[h] = first kept slot in segment >= h; the segment being
                # filtered starts no later than the end of the kept entries.
                start = ld_shared_b32(smem.seg_new + lane * 4)
                if lane == seg_now:
                    start = cutlass.min(start, total)
                for s in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_down(
                        start, 1 << s, mask_and_clamp=31
                    )
                    if lane + (1 << s) < 32:
                        start = cutlass.min(start, other)
                st_shared_b32(smem.seg_cur + lane * 4, start)
            cute.arch.barrier(
                barrier_id=CONSUMER_BAR, number_of_threads=CONSUMER_THREADS
            )
        return total


# ====================================================================
# Host entry
# ====================================================================

_compiled_cache: dict[tuple, Any] = {}


def stream_topk_indices(
    x: torch.Tensor,
    k: int,
    end: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Indices ``[rows, k]`` of each row's top k.

    ``x`` is a contiguous bf16 / fp16 ``[rows, n]`` tensor with ``n % 8 == 0``,
    ``n <= MAX_N`` and a 16 B aligned base; ``end`` (int32 ``[rows]``) limits row r to
    ``x[r, :end[r]]``; ``out_dtype`` is torch.int32 or torch.int64.
    """
    rows, n = x.shape
    assert x.dtype in (torch.bfloat16, torch.float16) and x.is_contiguous()
    assert n % 8 == 0 and x.data_ptr() % 16 == 0, "rows must start 16 B aligned"
    assert n <= MAX_N, f"rows longer than {MAX_N} are not supported"
    assert 1 <= k <= MAX_K, f"k must be in [1, {MAX_K}]"
    assert out_dtype in (torch.int32, torch.int64)
    out = torch.empty(rows, k, dtype=out_dtype, device=x.device)
    if end is None:
        end_t = torch.empty(1, dtype=torch.int32, device=x.device)
    else:
        assert end.dtype == torch.int32 and end.shape == (rows,) and end.is_contiguous()
        end_t = end
    key = (x.dtype, out_dtype, end is not None)
    if key not in _compiled_cache:
        rows_sym, n_sym = cute.sym_int(), cute.sym_int()
        _compiled_cache[key] = cute.compile(
            StreamTopK(
                "bf16" if x.dtype == torch.bfloat16 else "f16",
                out_dtype == torch.int64,
                end is not None,
            ),
            make_fake_tensor(
                cute.BFloat16 if x.dtype == torch.bfloat16 else cute.Float16,
                shape=(rows_sym, n_sym),
                divisibility=8,
            ),
            make_fake_tensor(cute.Int32, shape=(cute.sym_int(),), divisibility=1),
            make_fake_tensor(
                cute.Int64 if out_dtype == torch.int64 else cute.Int32,
                shape=(rows_sym, cute.sym_int()),
                divisibility=1,
            ),
            Int32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compiled_cache[key](x, end_t, out, k)
    return out


# ====================================================================
# Public API
# ====================================================================


def cutedsl_topk_stream(
    x: torch.Tensor,
    k: int,
    *,
    end: torch.Tensor | None = None,
    sorted: bool = False,
    out_dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Return top-k **indices** along the last dimension.

    ``x`` is bf16 or fp16 with at most ``MAX_N`` elements per row, and
    ``1 <= k <= min(MAX_K, n)``.  Each row's indices are ascending; ties at the
    k-th value keep the lowest indices, so the result is deterministic.  Values
    compare as IEEE floats: -0 equals +0, and NaN is never selected.  ``end``
    (int32, shape ``x.shape[:-1]``) limits each row to its first ``end``
    elements; a row with fewer than ``k`` selectable elements pads its tail with
    -1.  Value-sorted output is not supported; pass ``sorted=False`` (the
    default).  ``out_dtype`` is torch.int64 (default) or torch.int32.
    """
    if sorted:
        raise ValueError("cutedsl_topk_stream does not support sorted=True")

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    rows, n = x2d.shape
    if x2d.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"cutedsl_topk_stream needs a bf16 / fp16 input, got {x2d.dtype}"
        )
    if n > MAX_N:
        raise ValueError(f"rows longer than {MAX_N} are not supported, got n={n}")
    if not 1 <= k <= min(MAX_K, n):
        raise ValueError(f"k must be in [1, min({MAX_K}, n={n})], got {k}")
    if end is not None:
        end = end.reshape(-1).to(torch.int32).contiguous()
    with nvtx.add_nvtx_event(f"DSA/cutedsl_topk_stream rows={rows} cols={n} k={k}"):
        pad = (-n) % 8
        if pad or x2d.data_ptr() % 16:
            # Rows must start 16 B aligned; the padding is excluded through ``end``.
            x2d = torch.nn.functional.pad(x2d, (0, pad))
            if end is None:
                end = torch.full((rows,), n, dtype=torch.int32, device=x2d.device)
            else:
                end = end.clamp(max=n)
        indices = stream_topk_indices(x2d, k, end, out_dtype)

    return indices.reshape(*orig_shape[:-1], k)
