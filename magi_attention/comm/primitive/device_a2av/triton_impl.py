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

from .triton_barrier import blockwise_barrier, sync_threads


@triton.jit
def _exchange_split_offsets(
    input_split_sizes_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
):
    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    split_sizes_offsets = tl.arange(0, world_size)

    # split_sizes_ptrs for all ranks, with a shape of [world_size, world_size]
    # where split_sizes_ptrs[i, j] is the split_sizes ptr sending to rank j from rank i
    input_split_sizes_ptrs = input_split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # remote_input_split_sizes_ptr := split_sizes_ptrs[remote_rank, :]
    remote_input_split_sizes_ptr = tl.load(input_split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    # remote_input_split_size_for_this_rank := split_sizes_ptrs[remote_rank, rank]
    # which is the number of tokens that remote_rank sends to this rank
    # i.e. the output_split_sizes[remote_rank] for this rank
    remote_input_split_size_for_this_rank = tl.load(remote_input_split_sizes_ptr + rank)

    # load remote_input_split_sizes, with a shape of [world_size,]
    # but since we only want to retrieve the prefix-sum input offsets for this rank sent by remote rank
    # we need to mask its split sizes over this rank
    remote_input_split_sizes = tl.load(
        remote_input_split_sizes_ptr + split_sizes_offsets,
        mask=split_sizes_offsets <= rank,
        other=0,
    )

    # calc the row offset in the input buffer of the remote rank to send to this rank
    remote_input_row_offset = (
        tl.sum(remote_input_split_sizes) - remote_input_split_size_for_this_rank
    )

    # output_split_sizes_ptrs := split_sizes_matrix[:, rank]
    output_split_sizes_ptrs = (
        tl.load(input_split_sizes_ptrs + split_sizes_offsets).to(
            tl.pointer_type(tl.int64)
        )
        + rank
    )

    # load remote_output_split_sizes, with a shape of [world_size,]
    # but since we only want to retrieve the prefix-sum output offsets for this rank recv from remote rank
    # we need to mask its split sizes over the remote rank
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=split_sizes_offsets <= remote_rank, other=0
    )

    # calc the row offset in the output buffer of this rank to recv from remote rank
    output_row_offset_for_remote_rank = (
        tl.sum(output_split_sizes) - remote_input_split_size_for_this_rank
    )

    return (
        remote_input_row_offset,
        output_row_offset_for_remote_rank,
        remote_input_split_size_for_this_rank,
    )


@triton.jit
def on_device_a2av_kernel(
    output_ptr,
    output_splits_ptr,
    input_ptrs,
    input_splits_ptr,
    signal_pad_ptrs,
    stride0: tl.constexpr,  # Separate dim for easier vectorization
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,  # the number of SMs to process each peer
    UNROLL_FACTOR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # All matching blocks from all devices barrier here to ensure
    # all writes to symm_mem buffers from previous kernels across all devices
    # are visible to the current kernel
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    # 2D Blocks layout: [world_size, BLOCKS_PER_REMOTE_RANK]
    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_idx_for_remote_rank = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK
    block_offsets = tl.arange(0, BLOCK_SIZE)

    (
        remote_input_row_offset,
        output_row_offset,
        num_rows_from_remote_rank,
    ) = _exchange_split_offsets(
        input_splits_ptr, rank, world_size, BLOCKS_PER_REMOTE_RANK
    )
    numel_from_remote_rank = num_rows_from_remote_rank * stride0

    output_splits_ptr = output_splits_ptr.to(tl.pointer_type(tl.uint64))

    # let the first block processing the remote rank to update its output split size
    if block_idx_for_remote_rank == 0:
        tl.store(output_splits_ptr + remote_rank, num_rows_from_remote_rank)

    # prepare the input ptr of the remote rank, thx to symmetric memory mechanism
    remote_input_ptr = (
        tl.load(input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank).to(
            tl.pointer_type(tl.bfloat16)
        )
        + remote_input_row_offset * stride0
    )

    # prepare the output ptr of this rank to recv from the remote rank
    output_ptr = output_ptr + output_row_offset * stride0

    # move the data from remote rank's remote_input_ptr to this rank's output_ptr
    # where the data granularity is BLOCK_SIZE, and the unroll factor is UNROLL_FACTOR
    # so each move step is BLOCK_SIZE * UNROLL_FACTOR, i.e. the outer loop step
    outer_loop_step = BLOCK_SIZE * UNROLL_FACTOR
    outer_loop_iters_all_blocks = tl.cdiv(numel_from_remote_rank, outer_loop_step)
    outer_loop_iters_per_block = tl.cdiv(
        outer_loop_iters_all_blocks, BLOCKS_PER_REMOTE_RANK
    )
    numel_per_block = outer_loop_step * outer_loop_iters_per_block
    start_offset_this_block = numel_per_block * block_idx_for_remote_rank
    end_offset_this_block = tl.minimum(
        numel_per_block * (block_idx_for_remote_rank + 1), numel_from_remote_rank
    )
    unroll_region_size = (
        (end_offset_this_block - start_offset_this_block)
        // outer_loop_step
        * outer_loop_step
    )
    end_offset_this_block_in_unroll_region = (
        start_offset_this_block + unroll_region_size
    )

    # move the data in the divisible unroll region
    for start_offset_this_outer_step in tl.range(
        start_offset_this_block, end_offset_this_block_in_unroll_region, outer_loop_step
    ):
        end_offset_this_outer_step = start_offset_this_outer_step + outer_loop_step
        for start_offset_this_inner_step in tl.range(
            start_offset_this_outer_step,
            end_offset_this_outer_step,
            BLOCK_SIZE,
            loop_unroll_factor=UNROLL_FACTOR,
        ):
            offsets = start_offset_this_inner_step + block_offsets
            data = tl.load(remote_input_ptr + offsets)
            tl.store(output_ptr + offsets, data)

    # move the rest of the data out of the divisible unroll region
    for start_offset_this_step in tl.range(
        end_offset_this_block_in_unroll_region,
        end_offset_this_block,
        BLOCK_SIZE,
    ):
        offsets = start_offset_this_step + block_offsets
        mask = offsets < numel_from_remote_rank
        data = tl.load(remote_input_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, data, mask=mask)

    # All matching blocks from all devices barrier here to ensure
    # symm_mem buffers read by the current kernel are safe
    # from been writing by subsequent kernels across all devices.
    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")


def on_device_a2av_triton_impl(
    output: torch.Tensor,
    output_splits: torch.Tensor,
    input: torch.Tensor,
    input_splits: torch.Tensor,
    group: dist.ProcessGroup,
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
    assert input.dtype == output.dtype == torch.bfloat16
    assert input_splits.dtype == output_splits.dtype == torch.int64

    stride0 = output.shape[1]

    # redendezvous: exchange handle with each other device in the given group
    # and return the `_SymmetricMemory` handle
    # so that they can directly access the remote data in any other device
    # whose data ptrs are all stored in the handle's `buffer_ptrs_dev`, which is a vector of data ptrs
    # where buffer_ptrs_dev[i] is the data ptr of the i-th device
    # and the signal ptrs are stored in the handle's `signal_pad_ptrs_dev`, which is a vector of signal ptrs
    # where signal_pad_ptrs_dev[i] is the signal ptr of the i-th device
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

    # determine the number of blocks
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    on_device_a2av_kernel[(num_blocks, 1, 1)](
        output,
        output_splits,
        input_hdl.buffer_ptrs_dev,
        input_splits_hdl.buffer_ptrs_dev,
        input_hdl.signal_pad_ptrs_dev,
        stride0=stride0,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        UNROLL_FACTOR=UNROLL_FACTOR,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return output
