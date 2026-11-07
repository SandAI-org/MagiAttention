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


def partition_index_sparse_rows_by_k_chunk(
    index_sparse_indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the ids of every index row by K chunk ``id // chunk_size``.

    Args:
        index_sparse_indices: (num_q_blocks, nhk, topk) int32. Valid ids come
            first and lie in ``[0, num_chunks * chunk_size)``, the tail is -1.
        chunk_size: K tokens per chunk.
        num_chunks: number of chunks.

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
    num_q_blocks, nhk, topk = indices.shape
    # the kernel counts each chunk's ids in a 16-bit field
    assert topk < 2**16
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
    block = triton.next_power_of_2(topk)
    num_sms = torch.cuda.get_device_properties(indices.device).multi_processor_count
    _partition_index_sparse_rows_by_k_chunk_kernel[(min(num_rows, 8 * num_sms),)](
        indices,
        partitioned,
        starts,
        lens,
        num_rows,
        topk,
        chunk_size,
        NUM_CHUNKS=num_chunks,
        NUM_CHUNKS_POW2=triton.next_power_of_2(num_chunks),
        BLOCK=block,
        num_warps=4,
    )
    return partitioned, starts, lens
