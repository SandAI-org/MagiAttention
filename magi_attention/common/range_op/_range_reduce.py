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

__all__ = ["range_reduce"]


def _calc_range_reduce_row_map(
    input_ranges: torch.Tensor, total_size: int
) -> torch.Tensor:
    row_map = torch.arange(0, input_ranges.shape[0], device=input_ranges.device)
    range_sizes = input_ranges[:, 1] - input_ranges[:, 0]
    row_map = torch.repeat_interleave(
        row_map, range_sizes, dim=0, output_size=total_size
    )

    return row_map


@triton.jit
def range_reduce_kernel(
    input_ptr,
    output_ptr,
    input_ranges_ptr,
    output_ranges_ptr,
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

    input_range_start = tl.load(input_ranges_ptr + range_idx * 2)
    output_range_start = tl.load(output_ranges_ptr + range_idx * 2)

    inp_idx = (
        input_range_start + row_idx_in_range
    ) * input_stride + block_idx_in_row * ELEM_PER_BLOCK
    out_idx = (
        output_range_start + row_idx_in_range
    ) * output_stride + block_idx_in_row * ELEM_PER_BLOCK
    curr_inp_ptr = input_ptr + inp_idx
    curr_out_ptr = output_ptr + out_idx

    is_last_block = block_idx_in_row == N_BLOCK - 1
    if not is_last_block:
        cols = tl.arange(0, ELEM_PER_BLOCK)
        inp = tl.load(curr_inp_ptr + cols)
        tl.atomic_add(curr_out_ptr + cols, inp)
    else:
        elem_in_last_block = N - block_idx_in_row * ELEM_PER_BLOCK
        cols = tl.arange(0, ELEM_PER_BLOCK)
        inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)
        tl.atomic_add(curr_out_ptr + cols, inp, mask=cols < elem_in_last_block)


@triton.jit
def range_reduce_kernel_deterministic(
    input_ptr,
    output_ptr,
    input_ranges_ptr,
    output_ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    n_ranges,
    input_stride,
    output_stride,
    M,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ELEM_PER_BLOCK: tl.constexpr,
):
    # TODO: finish deterministic range reduction kernel
    pass


@nvtx.instrument_nvtx
def range_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    cu_range_sizes: torch.Tensor,
    total_size: int,
    dim: int = 0,
    row_map: torch.Tensor | None = None,
    deterministic: bool = False,
    range_split_sizes: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reduce values from input tensor to output tensor based on specified ranges.

    Args:
        input: Source tensor to reduce from
        output: Destination tensor to reduce into
        input_ranges: Tensor of [start, end] ranges in the input
        output_ranges: Tensor of [start, end] ranges in the output
        cu_range_sizes: Cumulative sizes of ranges
        total_size: Total number of rows of the input to process
        dim: Dimension along which to perform the reduction
        row_map(Optional):  mapping from row indices to range indices
        # TODO(littsk): finish deterministic reduction docstring

    Returns:
        The output tensor after reduction
    """
    assert (
        input_ranges.shape == output_ranges.shape
    ), f"{input_ranges=} and {output_ranges=} must have the same shape"

    if deterministic:
        assert range_split_sizes is not None
        raise NotImplementedError("Deterministic range reduction is not implemented")

    # Return directly if empty tensor
    if input_ranges.shape[0] == 0 or input.numel() == 0:
        return output

    output_ = output
    need_to_copy = False

    # Handle the case when dim is not 0
    if dim != 0:
        input = input.transpose(0, dim).contiguous()
        output_ = output_.transpose(0, dim).contiguous()
        need_to_copy = True
    else:
        need_to_copy |= not output.is_contiguous()
        input = input.contiguous()
        output_ = output_.contiguous()

    if output.dtype == torch.bfloat16:
        # NOTE: triton atomic op does not support bfloat16, see issue:
        # https://github.com/pytorch/pytorch/issues/97016
        output_ = output_.to(torch.float32)
        need_to_copy = True

    input_ranges = input_ranges.contiguous()
    output_ranges = output_ranges.contiguous()
    cu_range_sizes = cu_range_sizes.contiguous()

    # Calculate row_map if not provided
    if row_map is None:
        row_map = _calc_range_reduce_row_map(input_ranges, total_size)
    else:
        row_map = row_map.contiguous()

    # Calculate stride (considering memory step size of elements)
    input_stride = input.stride(0)
    output_stride = output.stride(0)

    # Calculate grid size
    M = total_size
    N = input.numel() // input.shape[0]

    ELEM_PER_BLOCK = 2048 // input.element_size()
    N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

    grid = (M, N_BLOCK)

    # Launch kernel
    range_reduce_kernel[grid](
        input,
        output_,
        input_ranges,
        output_ranges,
        cu_range_sizes,
        row_map,
        input_stride,
        output_stride,
        N,
        N_BLOCK,
        ELEM_PER_BLOCK,
    )

    # If transposed earlier, transpose back
    if dim != 0:
        output_ = output_.transpose(0, dim)

    if need_to_copy:
        output.data.copy_(output_)

    return output
