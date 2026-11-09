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

"""Index-sparse (DSA) attention helpers used by the kernels.

``partition_index_sparse_rows_by_k_chunk`` groups the ids of every index row by
K chunk for the K-chunked passes of ``ffa_bwd_sm100_index``.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partition_index_sparse_rows_by_k_chunk_kernel(
    indices_ptr,  # input: (num_rows, topk) int32, valid ids first, -1 padded
    out_ptr,  # output: (num_rows, topk) int32
    starts_ptr,  # output: (num_chunks, num_rows) int32
    lens_ptr,  # output: (num_chunks, num_rows) int32
    num_rows,
    topk,
    chunk_size,
    NUM_CHUNKS: tl.constexpr,
    NUM_CHUNKS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    in_row = offs < topk
    chunk_ids = tl.arange(0, NUM_CHUNKS_POW2)
    # persistent over rows: one row is too little work for a CTA of its own
    for row in tl.range(tl.program_id(0), num_rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        ids = tl.load(indices_ptr + row64 * topk + offs, mask=in_row, other=-1)
        # -1 padding goes past the last chunk, i.e. stays at the tail
        chunk = tl.where(ids >= 0, ids // chunk_size, NUM_CHUNKS)

        # Counting sort by chunk: a chunk's ids keep their order within the row.
        # One int64 scan ranks 4 chunks at once, one 16-bit count per chunk
        # (the caller bounds topk below 2**16).
        base = 0
        # per-chunk start / length, stored once per row (a scalar store would
        # be issued by every thread)
        starts = tl.zeros([NUM_CHUNKS_POW2], dtype=tl.int32)
        lens = tl.zeros([NUM_CHUNKS_POW2], dtype=tl.int32)
        # the -1 tail keeps its slot: everything before it is a valid id
        pos = offs
        for g in tl.static_range((NUM_CHUNKS + 3) // 4):
            in_group = (chunk >= 4 * g) & (chunk < min(4 * g + 4, NUM_CHUNKS)) & in_row
            shift = ((chunk - 4 * g) * 16).to(tl.int64)
            one_hot = tl.where(in_group, tl.full([BLOCK], 1, tl.int64) << shift, 0)
            rank = (tl.cumsum(one_hot, axis=0) >> shift) & 0xFFFF
            totals = tl.sum(one_hot, axis=0)
            start = tl.zeros([BLOCK], dtype=tl.int32)
            for j in tl.static_range(4):
                if 4 * g + j < NUM_CHUNKS:
                    count = ((totals >> (16 * j)) & 0xFFFF).to(tl.int32)
                    starts = tl.where(chunk_ids == 4 * g + j, base, starts)
                    lens = tl.where(chunk_ids == 4 * g + j, count, lens)
                    start = tl.where(chunk == 4 * g + j, base, start)
                    base += count
            pos = tl.where(in_group, start + rank.to(tl.int32) - 1, pos)
        tl.store(out_ptr + row64 * topk + pos, ids, mask=in_row)
        in_chunks = chunk_ids < NUM_CHUNKS
        tl.store(starts_ptr + chunk_ids * num_rows + row64, starts, mask=in_chunks)
        tl.store(lens_ptr + chunk_ids * num_rows + row64, lens, mask=in_chunks)


@triton.jit
def _partition_index_sparse_rows_by_k_chunk_tiled_kernel(
    indices_ptr,  # input: (num_rows, topk) int32, valid ids first, -1 padded
    out_ptr,  # output: (num_rows, topk) int32
    starts_ptr,  # output: (num_chunks, num_rows) int32
    lens_ptr,  # output: (num_chunks, num_rows) int32
    num_rows,
    topk,
    chunk_size,
    NUM_CHUNKS: tl.constexpr,
    NUM_CHUNKS_POW2: tl.constexpr,
    TILE: tl.constexpr,  # < 2**16
):
    """Rows too long for one block: count every chunk in a first sweep over the
    row, then scatter in a second, each sweep TILE ids at a time."""
    offs = tl.arange(0, TILE)
    chunk_ids = tl.arange(0, NUM_CHUNKS_POW2)
    in_chunks = chunk_ids < NUM_CHUNKS
    for row in tl.range(tl.program_id(0), num_rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        row_in = indices_ptr + row64 * topk
        row_out = out_ptr + row64 * topk

        # sweep 1: per-chunk lengths. A tile's per-chunk count fits the 16-bit
        # fields, the row total accumulates in int32.
        lens = tl.zeros([NUM_CHUNKS_POW2], dtype=tl.int32)
        for t0 in range(0, topk, TILE):
            in_row = t0 + offs < topk
            ids = tl.load(row_in + t0 + offs, mask=in_row, other=-1)
            chunk = tl.where(ids >= 0, ids // chunk_size, NUM_CHUNKS)
            for g in tl.static_range((NUM_CHUNKS + 3) // 4):
                in_group = (
                    (chunk >= 4 * g) & (chunk < min(4 * g + 4, NUM_CHUNKS)) & in_row
                )
                shift = ((chunk - 4 * g) * 16).to(tl.int64)
                one_hot = tl.where(in_group, tl.full([TILE], 1, tl.int64) << shift, 0)
                totals = tl.sum(one_hot, axis=0)
                for j in tl.static_range(4):
                    if 4 * g + j < NUM_CHUNKS:
                        count = ((totals >> (16 * j)) & 0xFFFF).to(tl.int32)
                        lens = tl.where(chunk_ids == 4 * g + j, lens + count, lens)
        starts = tl.cumsum(lens, axis=0) - lens
        tl.store(starts_ptr + chunk_ids * num_rows + row64, starts, mask=in_chunks)
        tl.store(lens_ptr + chunk_ids * num_rows + row64, lens, mask=in_chunks)

        # sweep 2: a chunk's next free slot advances by its count in every tile,
        # so its ids keep their order across tiles
        next_slot = starts
        for t0 in range(0, topk, TILE):
            in_row = t0 + offs < topk
            ids = tl.load(row_in + t0 + offs, mask=in_row, other=-1)
            chunk = tl.where(ids >= 0, ids // chunk_size, NUM_CHUNKS)
            # the -1 tail keeps its slot: everything before it is a valid id
            pos = t0 + offs
            for g in tl.static_range((NUM_CHUNKS + 3) // 4):
                in_group = (
                    (chunk >= 4 * g) & (chunk < min(4 * g + 4, NUM_CHUNKS)) & in_row
                )
                shift = ((chunk - 4 * g) * 16).to(tl.int64)
                one_hot = tl.where(in_group, tl.full([TILE], 1, tl.int64) << shift, 0)
                rank = (tl.cumsum(one_hot, axis=0) >> shift) & 0xFFFF
                totals = tl.sum(one_hot, axis=0)
                start = tl.zeros([TILE], dtype=tl.int32)
                for j in tl.static_range(4):
                    if 4 * g + j < NUM_CHUNKS:
                        is_c = chunk_ids == 4 * g + j
                        base = tl.sum(tl.where(is_c, next_slot, 0), axis=0)
                        count = ((totals >> (16 * j)) & 0xFFFF).to(tl.int32)
                        start = tl.where(chunk == 4 * g + j, base, start)
                        next_slot = tl.where(is_c, next_slot + count, next_slot)
                pos = tl.where(in_group, start + rank.to(tl.int32) - 1, pos)
            tl.store(row_out + pos, ids, mask=in_row)


@triton.jit
def _lower_bound(row_ptr, topk, value, mask, SEARCH_STEPS: tl.constexpr):
    """First slot of each row holding an id >= ``value``, -1 counting as +inf."""
    lo = tl.zeros_like(value)
    hi = tl.zeros_like(value) + topk
    for _ in tl.static_range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        ids = tl.load(row_ptr + mid, mask=mask & active, other=-1)
        go_right = (ids >= 0) & (ids < value)
        lo = tl.where(active & go_right, mid + 1, lo)
        hi = tl.where(active & ~go_right, mid, hi)
    return lo


@triton.jit
def _sorted_rows_chunk_bounds_kernel(
    indices_ptr,  # input: (num_rows, topk) int32, ascending valid ids, -1 padded
    starts_ptr,  # output: (num_chunks, num_rows) int32
    lens_ptr,  # output: (num_chunks, num_rows) int32
    num_rows,
    topk,
    chunk_size,
    NUM_CHUNKS: tl.constexpr,
    NUM_CHUNKS_POW2: tl.constexpr,
    ROWS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,  # >= bit_length(topk)
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    chunk_ids = tl.arange(0, NUM_CHUNKS_POW2)[None, :]
    mask = (rows < num_rows) & (chunk_ids < NUM_CHUNKS)
    row_ptr = indices_ptr + rows.to(tl.int64) * topk
    starts = _lower_bound(row_ptr, topk, chunk_ids * chunk_size, mask, SEARCH_STEPS)
    ends = _lower_bound(row_ptr, topk, (chunk_ids + 1) * chunk_size, mask, SEARCH_STEPS)
    tl.store(starts_ptr + chunk_ids * num_rows + rows, starts, mask=mask)
    tl.store(lens_ptr + chunk_ids * num_rows + rows, ends - starts, mask=mask)


# Rows up to this many ids sort in one register block, longer ones sweep TILE ids
# at a time: a whole row in registers spills past it (and at 32K+ ids takes
# minutes to compile).
_MAX_SINGLE_BLOCK = 4096
_TILE = 4096


def _partition_index_sparse_rows_by_k_chunk_launch(
    indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
    tile: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``tile=None``: one block per row; otherwise the two-sweep kernel."""
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
    num_sms = torch.cuda.get_device_properties(indices.device).multi_processor_count
    grid = (min(num_rows, 8 * num_sms),)
    if tile is None:
        # the kernel counts each chunk's ids in a 16-bit field
        assert topk < 2**16
        _partition_index_sparse_rows_by_k_chunk_kernel[grid](
            indices,
            partitioned,
            starts,
            lens,
            num_rows,
            topk,
            chunk_size,
            NUM_CHUNKS=num_chunks,
            NUM_CHUNKS_POW2=triton.next_power_of_2(num_chunks),
            BLOCK=triton.next_power_of_2(topk),
            num_warps=4,
        )
    else:
        assert tile < 2**16 and tile == triton.next_power_of_2(tile)
        # the scatter positions and per-chunk lengths are int32
        assert topk < 2**31
        _partition_index_sparse_rows_by_k_chunk_tiled_kernel[grid](
            indices,
            partitioned,
            starts,
            lens,
            num_rows,
            topk,
            chunk_size,
            NUM_CHUNKS=num_chunks,
            NUM_CHUNKS_POW2=triton.next_power_of_2(num_chunks),
            TILE=tile,
            num_warps=16,
        )
    return partitioned, starts, lens


def _sorted_rows_chunk_bounds(
    indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_q_blocks, nhk, topk = indices.shape
    num_rows = num_q_blocks * nhk
    starts = torch.empty(
        num_chunks, num_q_blocks, nhk, dtype=torch.int32, device=indices.device
    )
    lens = torch.empty_like(starts)
    if num_rows == 0 or topk == 0:
        starts.zero_()
        lens.zero_()
        return indices, starts, lens
    # (chunk_ids + 1) * chunk_size and the search slots are int32
    assert num_chunks * chunk_size < 2**31 and topk < 2**31
    rows_per_cta = 32
    _sorted_rows_chunk_bounds_kernel[(triton.cdiv(num_rows, rows_per_cta),)](
        indices,
        starts,
        lens,
        num_rows,
        topk,
        chunk_size,
        NUM_CHUNKS=num_chunks,
        NUM_CHUNKS_POW2=triton.next_power_of_2(num_chunks),
        ROWS=rows_per_cta,
        SEARCH_STEPS=topk.bit_length(),
        num_warps=4,
    )
    return indices, starts, lens


def partition_index_sparse_rows_by_k_chunk(
    index_sparse_indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
    *,
    rows_sorted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the ids of every index row by K chunk ``id // chunk_size``.

    Args:
        index_sparse_indices: (num_q_blocks, nhk, topk) int32. Valid ids come
            first and lie in ``[0, num_chunks * chunk_size)``, the tail is -1.
        chunk_size: K tokens per chunk.
        num_chunks: number of chunks.
        rows_sorted: the caller guarantees every row's valid ids ascend. Such
            a row is already grouped by chunk, so it is returned as is and only
            the chunk bounds are binary searched. Unsorted rows passed with
            ``rows_sorted=True`` give wrong starts / lens.

    Returns:
        partitioned: same shape as the input. Row ``r`` holds the ids of chunk 0,
            then chunk 1, ..., each chunk in its original order, then the -1 tail.
        starts / lens: (num_chunks, num_q_blocks, nhk) int32. Chunk ``c`` of row
            ``r`` is ``partitioned[r][starts[c][r] : starts[c][r] + lens[c][r]]``.
    """
    assert index_sparse_indices.dim() == 3
    assert index_sparse_indices.dtype == torch.int32
    assert chunk_size >= 1 and num_chunks >= 1
    indices = index_sparse_indices.contiguous()
    if rows_sorted:
        return _sorted_rows_chunk_bounds(indices, chunk_size, num_chunks)
    topk = indices.shape[-1]
    tile = None if topk <= _MAX_SINGLE_BLOCK else _TILE
    return _partition_index_sparse_rows_by_k_chunk_launch(
        indices, chunk_size, num_chunks, tile
    )
