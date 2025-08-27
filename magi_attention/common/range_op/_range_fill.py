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


import torch
import triton
import triton.language as tl

from magi_attention.utils import nvtx

from .utils import _calc_cu_range_sizes, _calc_ranges_row_map

__all__ = ["range_fill_"]


@triton.jit
def range_fill_kernel(
    input_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    val,
    row_map_ptr,
    input_stride,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ELEM_PER_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx_in_row = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + row_idx)
    cu_range_size = tl.load(cu_range_sizes_ptr + range_idx)
    row_idx_in_range = row_idx - cu_range_size

    range_start = tl.load(ranges_ptr + range_idx * 2)

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


@nvtx.instrument_nvtx
def range_fill_(
    input: torch.Tensor,
    ranges: torch.Tensor,
    val: float,
    dim: int = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Fill specified ranges in the input tensor with a given value.

    Args:
        input (torch.Tensor): Tensor to be filled in-place
        ranges (torch.Tensor): Tensor of [start, end] ranges to fill
        val: Value to fill the ranges with
        dim: Dimension along which to perform the fill operation

        kwargs:
            - cu_range_sizes (torch.Tensor): Cumulative sizes of ranges
            - total_size (int): Total number of rows to process
            - row_map (torch.Tensor): mapping from row indices to range indices

    Returns:
        The in-place filled input tensor
    """

    # Check that input has no gradient
    assert not input.requires_grad, "input must not require grad"

    # ---   calculate meta   --- #

    # Return directly if empty tensor
    if ranges.shape[0] == 0 or input.numel() == 0:
        return input

    # Make ranges contiguous
    ranges = ranges.contiguous()

    # Calculate cu_range_sizes and total_size if not provided
    cu_range_sizes = kwargs.pop("cu_range_sizes", None)
    total_size = kwargs.pop("total_size", None)
    if cu_range_sizes is None or total_size is None:
        cu_range_sizes, total_size = _calc_cu_range_sizes(
            ranges,
            device=input.device,
        )
    else:
        cu_range_sizes = cu_range_sizes.contiguous()

    # Calculate row_map if not provided
    row_map = kwargs.pop("row_map", None)
    if row_map is None:
        row_map = _calc_ranges_row_map(ranges, total_size)
    else:
        row_map = row_map.contiguous()

    # ---   pre-process input/output   --- #

    # Handle the case when dim is not 0
    if dim != 0:
        kernel_input = input.transpose(0, dim).contiguous()
    else:
        kernel_input = input.contiguous()

    # Calculate stride (considering memory step size of elements)
    input_stride = kernel_input.stride(0)

    # ---   calculate grid size   --- #

    M = total_size
    N = kernel_input.numel() // kernel_input.shape[0]

    ELEM_PER_BLOCK = 2048 // kernel_input.element_size()
    N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

    # Calculate grid size
    grid = (M, N_BLOCK)

    # ---   launch kernel   --- #

    range_fill_kernel[grid](
        kernel_input,
        ranges,
        cu_range_sizes,
        val,
        row_map,
        input_stride,
        N,
        N_BLOCK,
        ELEM_PER_BLOCK,
    )

    # ---   post-process output   --- #

    # If transposed earlier, transpose back
    if dim != 0:
        kernel_input = kernel_input.transpose(0, dim)

    # Copy the data back to the input tensor
    input.data = kernel_input.data

    return input
