# Copyright (c) 2025 SandAI. All Rights Reserved.
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .triton_barrier import blockwise_barrier
from .triton_utils import sync_threads


@triton.jit
def _exchange_row_offsets(
    split_sizes_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
):
    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK

    # split_sizes_ptr for all ranks
    # All these vector stacks into split_sizes_matrix
    split_sizes_ptrs = split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # split_sizes_matrix[remote_rank, :]
    input_split_sizes_ptr = tl.load(split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    offsets_ = tl.arange(0, world_size)
    input_split_sizes = tl.load(
        input_split_sizes_ptr + offsets_, mask=offsets_ <= rank, other=0
    )

    num_rows = tl.load(input_split_sizes_ptr + rank)
    input_row_offset = tl.sum(input_split_sizes) - num_rows

    # split_sizes_matrix[:, rank]
    output_split_sizes_ptrs = (
        tl.load(split_sizes_ptrs + offsets_).to(tl.pointer_type(tl.int64)) + rank
    )
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=offsets_ <= remote_rank, other=0
    )
    output_row_offset = tl.sum(output_split_sizes) - num_rows

    return input_row_offset, output_row_offset, num_rows


@triton.jit
def on_device_a2av_kernel(
    output_ptr,
    output_splits_ptr,
    input_ptrs,
    input_splits_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,  # Separate dim for easier vectorization
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    input_row_offset, output_row_offset, num_rows = _exchange_row_offsets(
        input_splits_ptr, rank, world_size, BLOCKS_PER_REMOTE_RANK
    )

    output_splits_ptr = output_splits_ptr.to(tl.pointer_type(tl.uint64))
    if block_offset == 0:
        # Update output_splits
        tl.store(output_splits_ptr + remote_rank, num_rows)

    input_ptr = (
        tl.load(input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank).to(
            tl.pointer_type(tl.bfloat16)
        )
        + input_row_offset * dim
    )
    output_ptr = output_ptr + output_row_offset * dim

    outer_loop_step = BLOCK_SIZE * UNROLL_FACTOR
    outer_loop_iters_per_rank = tl.cdiv(
        tl.cdiv(num_rows * dim, outer_loop_step), BLOCKS_PER_REMOTE_RANK
    )
    numel_per_rank = outer_loop_step * outer_loop_iters_per_rank
    offset = numel_per_rank * block_offset
    end = tl.minimum(numel_per_rank * (block_offset + 1), num_rows * dim)

    unroll_region_size = (end - offset) // outer_loop_step * outer_loop_step
    for i in tl.range(offset, offset + unroll_region_size, outer_loop_step):
        for j in tl.range(
            i,
            i + outer_loop_step,
            BLOCK_SIZE,
            loop_unroll_factor=UNROLL_FACTOR,
        ):
            offsets = j + tl.arange(0, BLOCK_SIZE)
            data = tl.load(input_ptr + offsets)
            tl.store(output_ptr + offsets, data)

    offset += unroll_region_size
    while offset < end:
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_rows * dim
        data = tl.load(input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)
        offset += BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


def on_device_a2av_triton_impl(
    output: torch.Tensor,
    output_splits: torch.Tensor,
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK=8,
    UNROLL_FACTOR: int = 8,
    BLOCK_SIZE: int = 16384,
):
    assert (
        input.is_contiguous()
        and output.is_contiguous()
        and input_splits.is_contiguous()
        and output_splits.is_contiguous()
    )
    assert input.dim() == output.dim() == 2 and output.shape[1] == input.shape[1]
    assert input_splits.shape[0] == output_splits.shape[0] == group.size()
    assert input_splits.dtype == output_splits.dtype == torch.int64

    dim = output.shape[1]
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK
    on_device_a2av_kernel[(num_blocks, 1, 1)](
        output,
        output_splits,
        input_hdl.buffer_ptrs_dev,
        input_splits_hdl.buffer_ptrs_dev,
        input_hdl.signal_pad_ptrs_dev,
        dim=dim,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        UNROLL_FACTOR=UNROLL_FACTOR,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output
