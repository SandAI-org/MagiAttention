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

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr
from quack.compile_utils import make_fake_tensor

from .cutedsl_utils import (
    lanemask_gt,
    lanemask_lt,
    ld_global_b32_pred,
    ld_shared_b32,
    match_any_sync,
    red_shared_add_u32,
    st_global_b32_pred,
    st_shared_b32,
    warp_prefix_sum,
)
from .dsl_compat import FastDivmodDivisor, SmemAllocator

# A CTA takes one index row, split into ROW_WARPS contiguous warp segments.
ROW_WARPS = 8
# Lane l loads ids t + 32 * e + l for e in range(IDS_PER_LANE): each load is one
# coalesced 128 B warp access, and the whole step is issued before any use.
IDS_PER_LANE = 8
STEP = 32 * IDS_PER_LANE
# Lane c publishes chunk c, so a row has at most one chunk per lane.
MAX_CHUNKS = 32


class IndexSparseRowChunks:
    """Stable counting sort of every index row by K chunk ``id // chunk_size``.

    An id's chunk is ``min(max(id, 0) // chunk_size, num_chunks)``: -1 padding
    lands in the tail bucket. Valid ids come first in a row, so the tail's own
    columns lie past every slot the valid ids fill. Without ``partition`` only
    ``mLens`` is written.
    """

    def __init__(self, num_chunks: int, partition: bool):
        assert 1 <= num_chunks <= MAX_CHUNKS
        self.num_chunks = num_chunks
        self.partition = partition

    @cute.jit
    def __call__(
        self,
        mIdx: cute.Tensor,
        mOut: cute.Tensor | None,
        mStarts: cute.Tensor | None,
        mLens: cute.Tensor,
        chunk_size: Int32,
        stream,
    ):
        @cute.struct
        class SharedStorage:
            # Per warp and chunk: ids counted, then the next free slot. Chunk
            # num_chunks is the tail bucket, whose slot is never read.
            counts: cute.struct.Align[
                cute.struct.MemRange[Int32, ROW_WARPS * (MAX_CHUNKS + 1)], 16
            ]
            slots: cute.struct.Align[
                cute.struct.MemRange[Int32, ROW_WARPS * (MAX_CHUNKS + 1)], 16
            ]

        self.shared_storage = SharedStorage
        self.kernel(mIdx, mOut, mStarts, mLens, FastDivmodDivisor(chunk_size)).launch(
            grid=[mIdx.shape[0], 1, 1],
            block=[ROW_WARPS * 32, 1, 1],
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIdx: cute.Tensor,
        mOut: cute.Tensor | None,
        mStarts: cute.Tensor | None,
        mLens: cute.Tensor,
        chunk_div: FastDivmodDivisor,
    ):
        num_chunks = self.num_chunks
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        row = Int32(cute.arch.block_idx()[0])
        topk = Int32(mIdx.shape[1])

        storage = SmemAllocator().allocate(self.shared_storage)
        counts = Int32(storage.counts.data_ptr().toint())
        slots = Int32(storage.slots.data_ptr().toint())
        chunk_stride = (MAX_CHUNKS + 1) * 4
        warp_counts = counts + warp * chunk_stride
        warp_slots = slots + warp * chunk_stride
        ids = cute.make_rmem_tensor((IDS_PER_LANE,), Int32)
        chunks = cute.make_rmem_tensor((IDS_PER_LANE,), Int32)

        seg_len = (topk + ROW_WARPS * 32 - 1) // (ROW_WARPS * 32) * 32
        seg_begin = cutlass.min(warp * seg_len, topk)
        seg_end = cutlass.min(seg_begin + seg_len, topk)
        row_in = Int64(mIdx.iterator.toint()) + Int64(row) * Int64(
            mIdx.stride[0]
        ) * Int64(4)

        if lane <= num_chunks:
            st_shared_b32(warp_counts + lane * 4, Int32(0))
        cute.arch.sync_warp()
        if const_expr(num_chunks == 1):
            valid = Int32(0)
            for t in cutlass.range(seg_begin, seg_end, STEP, unroll=1):
                self._load_step(row_in, t, seg_end, lane, chunk_div, ids, chunks)
                for e in cutlass.range_constexpr(IDS_PER_LANE):
                    valid += Int32(chunks[e] == 0)
            red_shared_add_u32(warp_counts, valid)
        else:
            gt_mask = lanemask_gt()
            for t in cutlass.range(seg_begin, seg_end, STEP, unroll=1):
                self._load_step(row_in, t, seg_end, lane, chunk_div, ids, chunks)
                for e in cutlass.range_constexpr(IDS_PER_LANE):
                    peers = match_any_sync(chunks[e])
                    if (peers & gt_mask) == 0:
                        red_shared_add_u32(
                            warp_counts + chunks[e] * 4, cute.arch.popc(peers)
                        )
        cute.arch.barrier()
        # Lane c sums chunk c over the warps: ``chunk_len`` is its row-wide
        # length, ``len_before`` the part held by the earlier warps.
        chunk_len = Int32(0)
        len_before = Int32(0)
        if lane < num_chunks:
            for w in cutlass.range_constexpr(ROW_WARPS):
                c = ld_shared_b32(counts + w * chunk_stride + lane * 4)
                chunk_len += c
                if w < warp:
                    len_before += c
        chunk_start = warp_prefix_sum(chunk_len, lane) - chunk_len
        if warp == 0 and lane < num_chunks:
            mLens[lane, row] = chunk_len
            if const_expr(self.partition):
                assert mStarts is not None
                mStarts[lane, row] = chunk_start

        if const_expr(self.partition):
            assert mOut is not None
            row_out = Int64(mOut.iterator.toint()) + Int64(row) * Int64(
                mOut.stride[0]
            ) * Int64(4)
            if const_expr(num_chunks == 1):
                # valid ids first, then -1: the row is already in chunk order
                for t in cutlass.range(seg_begin, seg_end, STEP, unroll=1):
                    self._load_step(row_in, t, seg_end, lane, chunk_div, ids, chunks)
                    for e in cutlass.range_constexpr(IDS_PER_LANE):
                        col = t + 32 * e + lane
                        st_global_b32_pred(
                            row_out + Int64(col) * Int64(4), ids[e], col < seg_end
                        )
            else:
                lt_mask = lanemask_lt()
                gt_mask = lanemask_gt()
                if lane < num_chunks:
                    st_shared_b32(warp_slots + lane * 4, chunk_start + len_before)
                cute.arch.sync_warp()
                for t in cutlass.range(seg_begin, seg_end, STEP, unroll=1):
                    self._load_step(row_in, t, seg_end, lane, chunk_div, ids, chunks)
                    for e in cutlass.range_constexpr(IDS_PER_LANE):
                        col = t + 32 * e + lane
                        peers = match_any_sync(chunks[e])
                        slot = ld_shared_b32(warp_slots + chunks[e] * 4)
                        # every lane reads the slot before the group's leader
                        # advances it past the group
                        cute.arch.sync_warp()
                        if (peers & gt_mask) == 0:
                            st_shared_b32(
                                warp_slots + chunks[e] * 4,
                                slot + cute.arch.popc(peers),
                            )
                        cute.arch.sync_warp()
                        pos = col
                        if chunks[e] < num_chunks:
                            pos = slot + cute.arch.popc(peers & lt_mask)
                        st_global_b32_pred(
                            row_out + Int64(pos) * Int64(4), ids[e], col < seg_end
                        )

    @cute.jit
    def _load_step(
        self,
        row_in: Int64,
        t: Int32,
        end: Int32,
        lane: Int32,
        chunk_div: FastDivmodDivisor,
        ids: cute.Tensor,
        chunks: cute.Tensor,
    ):
        for e in cutlass.range_constexpr(IDS_PER_LANE):
            col = t + 32 * e + lane
            ids[e] = ld_global_b32_pred(
                row_in + Int64(col) * Int64(4), col < end, fill=-1
            )
        for e in cutlass.range_constexpr(IDS_PER_LANE):
            if const_expr(self.num_chunks == 1):
                # single chunk: the id is either valid (0) or tail (1)
                chunks[e] = Int32(ids[e] < 0)
            else:
                chunk = Int32(self.num_chunks)
                if ids[e] >= 0:
                    chunk = cutlass.min(Int32(ids[e] // chunk_div), chunk)
                chunks[e] = chunk


_compiled_cache: dict[tuple, Any] = {}


def _launch_row_chunks(
    indices: torch.Tensor,
    out: torch.Tensor | None,
    starts: torch.Tensor | None,
    lens: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> None:
    partition = out is not None
    key = (num_chunks, partition)
    if key not in _compiled_cache:
        rows_sym, topk_sym = cute.sym_int(), cute.sym_int()
        _compiled_cache[key] = cute.compile(
            IndexSparseRowChunks(num_chunks, partition),
            make_fake_tensor(Int32, (rows_sym, topk_sym)),
            make_fake_tensor(Int32, (rows_sym, topk_sym)) if partition else None,
            make_fake_tensor(Int32, (num_chunks, rows_sym)) if partition else None,
            make_fake_tensor(Int32, (num_chunks, rows_sym)),
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _compiled_cache[key](indices, out, starts, lens, chunk_size)


def partition_index_sparse_rows_by_k_chunk(
    index_sparse_indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group every index row by K chunk ``id // chunk_size``, keeping row order.

    Args:
        index_sparse_indices: (num_q_blocks, nhk, topk) int32. Valid ids come
            first and lie in ``[0, num_chunks * chunk_size)``, the tail is -1.
        chunk_size: K tokens per chunk.
        num_chunks: number of chunks, at most ``MAX_CHUNKS``.

    Returns:
        partitioned: (num_q_blocks, nhk, topk) int32, each row holding chunk 0,
            chunk 1, ..., then the -1 tail.
        starts, lens: (num_chunks, num_q_blocks, nhk) int32; chunk ``c`` of row
            ``r`` is ``partitioned[r][starts[c][r] : starts[c][r] + lens[c][r]]``.
    """
    assert index_sparse_indices.dim() == 3
    assert index_sparse_indices.dtype == torch.int32
    assert 1 <= chunk_size < 2**31 and 1 <= num_chunks <= MAX_CHUNKS
    indices = index_sparse_indices.contiguous()
    num_q_blocks, nhk, topk = indices.shape
    num_rows = num_q_blocks * nhk
    partitioned = torch.empty_like(indices)
    starts = torch.empty(
        num_chunks, num_q_blocks, nhk, dtype=torch.int32, device=indices.device
    )
    lens = torch.empty_like(starts)
    if num_rows == 0 or topk == 0:
        starts.zero_()
        lens.zero_()
        return partitioned, starts, lens
    _launch_row_chunks(
        indices.view(num_rows, topk),
        partitioned.view(num_rows, topk),
        starts.view(num_chunks, num_rows),
        lens.view(num_chunks, num_rows),
        chunk_size,
        num_chunks,
    )
    return partitioned, starts, lens


def count_index_sparse_valid_ids(index_sparse_indices: torch.Tensor) -> torch.Tensor:
    """Number of valid (>= 0) ids of every index row.

    Args:
        index_sparse_indices: (num_q_blocks, nhk, topk) int32.

    Returns:
        (num_q_blocks, nhk) int32.
    """
    assert index_sparse_indices.dim() == 3
    assert index_sparse_indices.dtype == torch.int32
    indices = index_sparse_indices.contiguous()
    num_q_blocks, nhk, topk = indices.shape
    num_rows = num_q_blocks * nhk
    lens = torch.empty(num_q_blocks, nhk, dtype=torch.int32, device=indices.device)
    if num_rows == 0 or topk == 0:
        lens.zero_()
        return lens
    # One chunk covering every non-negative int32: its length is the valid count
    _launch_row_chunks(
        indices.view(num_rows, topk), None, None, lens.view(1, num_rows), 2**31 - 1, 1
    )
    return lens
