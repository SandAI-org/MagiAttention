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

from .utils import _calc_cu_range_sizes, _calc_out2inp_range_map, _calc_ranges_row_map

__all__ = ["range_reduce"]


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
def range_reduce_deter_kernel(
    input_ptr,
    output_ptr,
    input_ranges_ptr,
    output_ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    out2inp_range_map_ptr,
    input_stride,
    output_stride,
    out2inp_range_map_stride,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ELEM_PER_BLOCK: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx_in_row = tl.program_id(1)

    range_idx = tl.load(row_map_ptr + row_idx)
    cu_range_size = tl.load(cu_range_sizes_ptr + range_idx)
    row_idx_in_range = row_idx - cu_range_size
    is_last_block = block_idx_in_row == N_BLOCK - 1
    elem_in_last_block = N - block_idx_in_row * ELEM_PER_BLOCK
    cols = tl.arange(0, ELEM_PER_BLOCK)

    output_range_start = tl.load(output_ranges_ptr + range_idx * 2)
    out_idx = (
        output_range_start + row_idx_in_range
    ) * output_stride + block_idx_in_row * ELEM_PER_BLOCK
    curr_out_ptr = output_ptr + out_idx

    if not is_last_block:
        out = tl.load(curr_out_ptr + cols)
    else:
        out = tl.load(curr_out_ptr + cols, mask=cols < elem_in_last_block)

    out2inp_range_map_start = (
        out2inp_range_map_ptr + range_idx * out2inp_range_map_stride
    )
    for idx in tl.range(0, out2inp_range_map_stride):
        inp_range_idx = tl.load(out2inp_range_map_start + idx)
        if inp_range_idx == -1:
            pass
        else:
            input_range_start = tl.load(input_ranges_ptr + inp_range_idx * 2)
            inp_idx = (
                input_range_start + row_idx_in_range
            ) * input_stride + block_idx_in_row * ELEM_PER_BLOCK
            curr_inp_ptr = input_ptr + inp_idx

            if not is_last_block:
                inp = tl.load(curr_inp_ptr + cols)
            else:
                inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)

            out += inp

    if not is_last_block:
        tl.store(curr_out_ptr + cols, out)
    else:
        tl.store(curr_out_ptr + cols, out, mask=cols < elem_in_last_block)


@nvtx.instrument_nvtx
def range_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    dim: int = 0,
    deterministic: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Reduce values from input tensor to output tensor based on specified ranges.

    Args:
        input (torch.Tensor): Source tensor to reduce from
        output (torch.Tensor): Destination tensor to reduce into
        input_ranges (torch.Tensor): Tensor of [start, end] ranges in the input
        output_ranges (torch.Tensor): Tensor of [start, end] ranges in the output
        dim (int, optional): Dimension along which to perform the reduction. Default is 0.
        deterministic(bool, optional): Whether to enable deterministic mode

        kwargs:
            - cu_range_sizes (torch.Tensor) : Cumulative sizes of input ranges,
                or cumulative sizes of output ranges in deterministic mode
            - total_size (int): Total number of rows of the input to process,
                or total number of rows of the output to be reduced in deterministic mode

            - row_map (torch.Tensor): mapping from row indices to input range indices,
                or mapping from row indices to output range indices in deterministic mode
            - out2inp_range_map (torch.Tensor): mapping from each output range index to the list of input range indices
                that need to be reduced, e.g. [(2, -1), (1, 3)] means that:
                    1. input_range[2] will reduce to output_range[0] (-1 is just the placeholder to be equal shape)
                    2. input_ranges[1] and input_ranges[3] will reduce to output_range[1]
                **NOTE**: this is only used in deterministic mode

    Returns:
        The output tensor after reduction
    """
    assert (
        input_ranges.shape == output_ranges.shape
    ), f"{input_ranges=} and {output_ranges=} must have the same shape"

    # Return directly if empty tensor
    if input_ranges.shape[0] == 0 or input.numel() == 0:
        return output

    # ---   calculate meta   --- #

    # Make input_ranges and output_ranges contiguous
    input_ranges = input_ranges.contiguous()
    output_ranges = output_ranges.contiguous()

    if deterministic:
        # Calculate out2inp_range_map and unique_ordered_out_ranges
        # if not provided for deterministic mode
        out2inp_range_map = kwargs.pop("out2inp_range_map", None)
        unique_ordered_out_ranges = kwargs.pop("unique_ordered_out_ranges", None)

        if out2inp_range_map is None or unique_ordered_out_ranges is None:
            (
                out2inp_range_map,
                unique_ordered_out_ranges,
                out2inp_range_map_stride,
            ) = _calc_out2inp_range_map(
                output_ranges,
                device=input.device,
            )
        else:
            out2inp_range_map = out2inp_range_map.contiguous()
            out2inp_range_map_stride = out2inp_range_map.shape[1]

    # Calculate cu_range_sizes and total_size if not provided
    cu_range_sizes = kwargs.pop("cu_range_sizes", None)
    total_size = kwargs.pop("total_size", None)
    if cu_range_sizes is None or total_size is None:
        cu_range_sizes, total_size = _calc_cu_range_sizes(
            unique_ordered_out_ranges if deterministic else input_ranges,
            device=input.device,
        )
    else:
        cu_range_sizes = cu_range_sizes.contiguous()

    # Calculate row_map if not provided
    row_map = kwargs.pop("row_map", None)
    if row_map is None:
        row_map = _calc_ranges_row_map(
            unique_ordered_out_ranges if deterministic else input_ranges,
            total_size,
        )
    else:
        row_map = row_map.contiguous()

    # ---   pre-process input/output   --- #

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

    if not deterministic and output.dtype == torch.bfloat16:
        # NOTE: in non-deterministic mode, we will use triton atomic op
        # which does not support bfloat16, w.r.t. the issue:
        # https://github.com/pytorch/pytorch/issues/97016
        output_ = output_.to(torch.float32)
        need_to_copy = True

    # Calculate stride (considering memory step size of elements)
    input_stride = input.stride(0)
    output_stride = output_.stride(0)

    # ---   calculate grid size   --- #

    # Calculate grid size
    M = total_size
    N = input.numel() // input.shape[0]

    ELEM_PER_BLOCK = 2048 // input.element_size()
    N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

    grid = (M, N_BLOCK)

    # ---   launch kernel   --- #

    # Launch kernel
    if deterministic:
        range_reduce_deter_kernel[grid](
            input,
            output_,
            input_ranges,
            unique_ordered_out_ranges,
            cu_range_sizes,
            row_map,
            out2inp_range_map,
            input_stride,
            output_stride,
            out2inp_range_map_stride,
            N,
            N_BLOCK,
            ELEM_PER_BLOCK,
        )
    else:
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

    # ---   post-process output   --- #

    # If transposed earlier, transpose back
    if dim != 0:
        output_ = output_.transpose(0, dim)

    if need_to_copy:
        output.data.copy_(output_)

    return output
