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

from typing import Optional

import torch
import triton
import triton.language as tl

__all__ = ["range_fill_"]


@triton.jit
def range_fill_kernel(
    input_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    val,
    row_map_ptr,
    n_ranges,
    input_stride,
    M,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ELEM_PER_BLOCK: tl.constexpr,
):
    # Current thread processes this range index
    row_idx = tl.program_id(0)
    block_idx_in_row = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + row_idx)
    cu_range_size = tl.load(cu_range_sizes_ptr + range_idx)
    row_idx_in_range = row_idx - cu_range_size

    range_start = tl.load(ranges_ptr + range_idx * 2)
    range_end = tl.load(ranges_ptr + range_idx * 2 + 1)
    range_size = range_end - range_start  # noqa

    inp_idx = (
        range_start + row_idx_in_range
    ) * input_stride + block_idx_in_row * ELEM_PER_BLOCK
    curr_inp_ptr = input_ptr + inp_idx

    is_last_block = block_idx_in_row == N_BLOCK - 1

    if not is_last_block:
        cols = tl.arange(0, ELEM_PER_BLOCK)
        tl.store(curr_inp_ptr + cols, val)
    else:
        elem_in_last_block = N - block_idx_in_row * ELEM_PER_BLOCK
        cols = tl.arange(0, ELEM_PER_BLOCK)
        tl.store(curr_inp_ptr + cols, val, mask=cols < elem_in_last_block)


def range_fill_(
    input: torch.Tensor,
    ranges: torch.Tensor,
    cu_range_sizes: torch.Tensor,
    total_size: int,
    val: float,
    dim: int = 0,
    row_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fill specified ranges in the input tensor with a given value.

    Args:
        input: Tensor to be filled in-place
        ranges: Tensor of [start, end] ranges to fill
        cu_range_sizes: Cumulative sizes of ranges
        total_size: Total number of rows to process
        val: Value to fill the ranges with
        dim: Dimension along which to perform the fill operation
        row_map: Optional mapping from row indices to range indices

    Returns:
        The modified input tensor
    """

    # Check that input has no gradient
    assert not input.requires_grad, "input must not require grad"

    # Get the number of ranges
    n_ranges = ranges.shape[0]

    # Return directly if empty tensor
    if n_ranges == 0 or input.numel() == 0:
        return input

    # Handle the case when dim is not 0
    if dim != 0:
        kernel_input = input.transpose(0, dim).contiguous()
    else:
        kernel_input = input.contiguous()

    ranges = ranges.contiguous()
    cu_range_sizes = cu_range_sizes.contiguous()

    # Calculate stride (considering memory step size of elements)
    input_stride = kernel_input.stride(0)

    if row_map is None:
        row_map = torch.arange(0, ranges.shape[0], device=ranges.device)
        range_sizes = ranges[:, 1] - ranges[:, 0]
        row_map = torch.repeat_interleave(
            row_map, range_sizes, dim=0, output_size=total_size
        )

    M = total_size
    N = kernel_input.numel() // kernel_input.shape[0]

    ELEM_PER_BLOCK = 2048 // kernel_input.element_size()
    N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

    # Calculate grid size
    grid = (M, N_BLOCK)

    # Launch kernel
    range_fill_kernel[grid](
        kernel_input,
        ranges,
        cu_range_sizes,
        val,
        row_map,
        n_ranges,
        input_stride,
        M,
        N,
        N_BLOCK,
        ELEM_PER_BLOCK,
    )

    # If transposed earlier, transpose back
    if dim != 0:
        kernel_input = kernel_input.transpose(0, dim)

    # Copy the data back to the input tensor
    input.data = kernel_input.data

    return input
