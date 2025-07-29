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

__all__ = ["range_gather"]


@triton.jit
def range_gather_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    input_stride,
    output_stride,
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
    out_idx = (
        cu_range_size + row_idx_in_range
    ) * output_stride + block_idx_in_row * ELEM_PER_BLOCK
    curr_inp_ptr = input_ptr + inp_idx
    curr_out_ptr = output_ptr + out_idx

    is_last_block = block_idx_in_row == N_BLOCK - 1

    if not is_last_block:
        cols = tl.arange(0, ELEM_PER_BLOCK)
        inp = tl.load(curr_inp_ptr + cols)
        tl.store(curr_out_ptr + cols, inp)
    else:
        elem_in_last_block = N - block_idx_in_row * ELEM_PER_BLOCK
        cols = tl.arange(0, ELEM_PER_BLOCK)
        inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)
        tl.store(curr_out_ptr + cols, inp, mask=cols < elem_in_last_block)


@nvtx.instrument_nvtx
def range_gather(
    input: torch.Tensor,
    ranges: torch.Tensor,
    dim: int = 0,
    output: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gather values from input tensor based on specified ranges into a new output tensor.

    Args:
        input (torch.Tensor): Source tensor to gather from
        ranges (torch.Tensor): Tensor of [start, end] ranges in the input
        dim: Dimension along which to perform the gather operation
        output: Optional output tensor buffer to store the result

        kwargs:
            - cu_range_sizes: Cumulative sizes of ranges
            - total_size: Total number of rows in the output tensor
            - row_map: mapping from row indices to range indices

    Returns:
        A new tensor containing the gathered values, put into output if provided.
    """

    # ---   calculate meta   --- #

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

    if output is None:
        output_shape = list(input.shape)
        output_shape[dim] = total_size
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
    else:
        assert dim == 0, "dim must be 0 when output is provided"
        assert output.is_contiguous(), "output must be contiguous when provided"

    # Return directly if empty tensor
    if ranges.shape[0] == 0 or input.numel() == 0:
        return output

    # Handle the case when dim is not 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
        output = output.transpose(0, dim).contiguous()
    else:
        input = input.contiguous()
        output = output.contiguous()

    # Calculate stride (considering memory step size of elements)
    input_stride = input.stride(0)
    output_stride = output.stride(0)

    # ---   calculate grid size   --- #

    M = total_size
    N = input.numel() // input.shape[0]

    ELEM_PER_BLOCK = 2048 // input.element_size()
    N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

    grid = (M, N_BLOCK)

    # ---   launch kernel   --- #

    range_gather_kernel[grid](
        input,
        output,
        ranges,
        cu_range_sizes,
        row_map,
        input_stride,
        output_stride,
        N,
        N_BLOCK,
        ELEM_PER_BLOCK,
    )

    # ---   post-process output   --- #

    # If transposed earlier, transpose back
    if dim != 0:
        output = output.transpose(0, dim)

    return output
