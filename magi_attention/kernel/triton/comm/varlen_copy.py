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

import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_for_each_copy_kernel(
    output_ptr,
    input_flat_ptr,
    num_dst_ptr,
    offsets_ptr,
    stride_output_row,
    num_ranks,
    BLOCK_SIZE_N: tl.constexpr,
):
    split_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    col_start = col_block_idx * BLOCK_SIZE_N

    num_dst = tl.load(num_dst_ptr + split_idx)

    offset_in_flat_input = tl.load(offsets_ptr + split_idx)

    output_row_base_ptr = output_ptr + split_idx * stride_output_row
    input_flat_base_ptr = input_flat_ptr + offset_in_flat_input

    cols_in_block = col_start + tl.arange(0, BLOCK_SIZE_N)

    write_mask = (cols_in_block < num_ranks) & (cols_in_block < num_dst)

    values_to_write = tl.load(
        input_flat_base_ptr + cols_in_block, mask=write_mask, other=-1
    )

    tl.store(output_row_base_ptr + cols_in_block, values_to_write, mask=write_mask)


def varlen_for_each_copy_triton(
    dst: torch.Tensor,
    src: torch.Tensor,
    num_dst: torch.Tensor,
    cu_num_dst: torch.Tensor,
) -> torch.Tensor:
    num_splits, num_ranks = dst.shape

    if num_splits == 0:
        return dst

    BLOCK_SIZE_N = triton.next_power_of_2(num_ranks)
    grid = (num_splits, triton.cdiv(num_ranks, BLOCK_SIZE_N))

    _varlen_for_each_copy_kernel[grid](
        dst,
        src,
        num_dst,
        cu_num_dst,
        dst.stride(0),
        num_ranks,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return dst
