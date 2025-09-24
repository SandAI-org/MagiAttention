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
from triton.language.extra import libdevice

from magi_attention.common.enum import OutMaybeWithLSE, ReduceOp
from magi_attention.utils import is_fp_dtype_at_least, max_fp_dtype, nvtx

from .utils import _calc_cu_range_sizes, _calc_out2inp_range_map, _calc_ranges_row_map

__all__ = ["range_reduce"]


torch2triton_dtype_map = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


@triton.jit
def range_sum_reduce_nondeter_kernel(
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
def range_sum_reduce_deter_kernel(
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
    reduce_dtype: tl.constexpr,
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

    # load output
    if not is_last_block:
        out = tl.load(curr_out_ptr + cols)
    else:
        out = tl.load(curr_out_ptr + cols, mask=cols < elem_in_last_block)
    out = out.to(reduce_dtype)

    # reduce input
    out2inp_range_map_start = (
        out2inp_range_map_ptr + range_idx * out2inp_range_map_stride
    )
    for idx in tl.range(0, out2inp_range_map_stride):
        inp_range_idx = tl.load(out2inp_range_map_start + idx)
        if inp_range_idx == -1:  # placeholder
            pass
        else:
            input_range_start = tl.load(input_ranges_ptr + inp_range_idx * 2)
            inp_idx = (
                input_range_start + row_idx_in_range
            ) * input_stride + block_idx_in_row * ELEM_PER_BLOCK
            curr_inp_ptr = input_ptr + inp_idx

            # load input
            if not is_last_block:
                inp = tl.load(curr_inp_ptr + cols)
            else:
                inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)
            inp = inp.to(reduce_dtype)

            # add to output
            out += inp

    # store reduced output
    if not is_last_block:
        tl.store(curr_out_ptr + cols, out)
    else:
        tl.store(curr_out_ptr + cols, out, mask=cols < elem_in_last_block)


@triton.jit
def range_avg_reduce_kernel(
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
    reduce_dtype: tl.constexpr,
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

    # load output
    if not is_last_block:
        out = tl.load(curr_out_ptr + cols)
    else:
        out = tl.load(curr_out_ptr + cols, mask=cols < elem_in_last_block)
    out = out.to(reduce_dtype)

    # reduce input
    cnt = 0.0
    out2inp_range_map_start = (
        out2inp_range_map_ptr + range_idx * out2inp_range_map_stride
    )
    for idx in tl.range(0, out2inp_range_map_stride):
        inp_range_idx = tl.load(out2inp_range_map_start + idx)
        if inp_range_idx == -1:  # placeholder
            pass
        else:
            cnt += 1.0
            input_range_start = tl.load(input_ranges_ptr + inp_range_idx * 2)
            inp_idx = (
                input_range_start + row_idx_in_range
            ) * input_stride + block_idx_in_row * ELEM_PER_BLOCK
            curr_inp_ptr = input_ptr + inp_idx

            # load input
            if not is_last_block:
                inp = tl.load(curr_inp_ptr + cols)
            else:
                inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)
            inp = inp.to(reduce_dtype)

            # add to output
            out += inp

    # scale by count
    if cnt > 1.0:
        out /= cnt

    # store reduced output
    if not is_last_block:
        tl.store(curr_out_ptr + cols, out)
    else:
        tl.store(curr_out_ptr + cols, out, mask=cols < elem_in_last_block)


@triton.jit
def _safe_subtract_exp(a, b):
    c = tl.exp(a - b)
    # resolve nan to zero causing by "-inf" - "-inf"
    c = tl.zeros_like(c) if libdevice.isnan(c) else c

    return c


@triton.jit
def range_lse_reduce_kernel(
    input_ptr,
    input_lse_ptr,
    output_ptr,
    output_lse_ptr,
    input_ranges_ptr,
    output_ranges_ptr,
    cu_range_sizes_ptr,
    row_map_ptr,
    out2inp_range_map_ptr,
    input_stride_s,
    input_stride_nh,
    input_lse_stride_s,
    output_stride_s,
    output_stride_nh,
    output_lse_stride_s,
    out2inp_range_map_stride,
    N: tl.constexpr,
    N_BLOCK: tl.constexpr,
    ELEM_PER_BLOCK: tl.constexpr,
    reduce_dtype: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_idx_in_row = tl.program_id(2)

    range_idx = tl.load(row_map_ptr + row_idx)
    cu_range_size = tl.load(cu_range_sizes_ptr + range_idx)
    row_idx_in_range = row_idx - cu_range_size
    is_last_block = block_idx_in_row == N_BLOCK - 1
    elem_in_last_block = N - block_idx_in_row * ELEM_PER_BLOCK
    cols = tl.arange(0, ELEM_PER_BLOCK)

    output_range_start = tl.load(output_ranges_ptr + range_idx * 2)
    out_idx = (
        (output_range_start + row_idx_in_range) * output_stride_s
        + head_idx * output_stride_nh
        + block_idx_in_row * ELEM_PER_BLOCK
    )
    curr_out_ptr = output_ptr + out_idx

    out_lse_idx = (
        output_range_start + row_idx_in_range
    ) * output_lse_stride_s + head_idx
    curr_out_lse_ptr = output_lse_ptr + out_lse_idx

    # load output
    if not is_last_block:
        out = tl.load(curr_out_ptr + cols)
    else:
        out = tl.load(curr_out_ptr + cols, mask=cols < elem_in_last_block)
    out = out.to(reduce_dtype)

    # load output lse
    out_lse = tl.load(curr_out_lse_ptr).to(reduce_dtype)

    # reduce input and input lse
    out2inp_range_map_start = (
        out2inp_range_map_ptr + range_idx * out2inp_range_map_stride
    )
    for idx in tl.range(0, out2inp_range_map_stride):
        inp_range_idx = tl.load(out2inp_range_map_start + idx)
        if inp_range_idx == -1:  # placeholder
            pass
        else:
            input_range_start = tl.load(input_ranges_ptr + inp_range_idx * 2)
            inp_idx = (
                (input_range_start + row_idx_in_range) * input_stride_s
                + head_idx * input_stride_nh
                + block_idx_in_row * ELEM_PER_BLOCK
            )
            curr_inp_ptr = input_ptr + inp_idx
            inp_lse_idx = (
                input_range_start + row_idx_in_range
            ) * input_lse_stride_s + head_idx
            curr_inp_lse_ptr = input_lse_ptr + inp_lse_idx

            # load input
            if not is_last_block:
                inp = tl.load(curr_inp_ptr + cols)
            else:
                inp = tl.load(curr_inp_ptr + cols, mask=cols < elem_in_last_block)
            inp = inp.to(reduce_dtype)

            # load input lse
            inp_lse = tl.load(curr_inp_lse_ptr).to(reduce_dtype)

            # correct lse
            # formula derivation:
            # reduced_lse = log(exp(lse1) + exp(lse2))
            #             = lse1 + log(1 + exp(lse2 - lse1))
            #             = max_lse + log(1 + exp(min_lse - max_lse))
            #             = max_lse + log1p(exp(min_lse - max_lse))
            min_lse = tl.minimum(inp_lse, out_lse)
            max_lse = tl.maximum(inp_lse, out_lse)
            reduced_lse = max_lse + libdevice.log1p(
                _safe_subtract_exp(min_lse, max_lse)
            )

            # get reduce weights
            out_weight = _safe_subtract_exp(out_lse, reduced_lse)
            inp_weight = _safe_subtract_exp(inp_lse, reduced_lse)

            # reduce output
            out = out_weight * out + inp_weight * inp

            # reduce output lse
            out_lse = reduced_lse

    # store reduced output
    if not is_last_block:
        tl.store(curr_out_ptr + cols, out)
    else:
        tl.store(curr_out_ptr + cols, out, mask=cols < elem_in_last_block)
        # NOTE: only last block need to store reduced output lse
        # since lse is shared across all blocks in a row
        # which also indicates the lse is reduced duplicately by N_BLOCK times
        tl.store(curr_out_lse_ptr, out_lse)


@nvtx.instrument_nvtx
def range_reduce(
    input: torch.Tensor,
    output: torch.Tensor,
    input_ranges: torch.Tensor,
    output_ranges: torch.Tensor,
    dim: int = 0,
    deterministic: bool = False,
    reduce_op: ReduceOp = "sum",
    reduce_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> OutMaybeWithLSE:
    """
    Reduce values from input tensor to output tensor based on specified ranges.

    Args:
        input (torch.Tensor): Source tensor to reduce from
        output (torch.Tensor): Destination tensor to reduce into
        input_ranges (torch.Tensor): Tensor of [start, end] ranges in the input
        output_ranges (torch.Tensor): Tensor of [start, end] ranges in the output
        dim (int, optional): Dimension along which to perform the reduction. Default is 0.
        deterministic(bool, optional): Whether to enable deterministic mode
        reduce_op (Literal["sum", "avg", "weight", "lse"]): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction

            NOTE:
                1. if reduce_op is "avg", we will sum-reduce to the output tensor and apply average division afterwards,
                    so the user should guarantee that the output tensor is initialized to zero
                    otherwise the semantics will be incorrect unless the user intentionally does this
                2. if reduce_op is "lse", the user is required to pass "input_lse" and "output_lse",
                    and we only support input/output has shape [seqlen, num_heads, head_dim]
                    while input_lse/output_lse has shape [seqlen, num_heads] for now
        reduce_dtype (torch.dtype): the dtype for the reduction.
            Defaults to None to use: maximum precision of the input/output and fp32

            NOTE: this is only used for those deterministic kernels, and for non-deterministic kernels,
            the dtype will always be the same dtype as the input/output
        input_lse (torch.Tensor | None): the log-sum-exp tensor for the input tensor,
            only required and used if reduce_op is "lse"
        output_lse (torch.Tensor | None): the log-sum-exp tensor for the output tensor,
            only required and used if reduce_op is "lse"

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
                NOTE: this is only used in deterministic mode

    Returns:
        tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        The output tensor with the corrected lse after reduction if reduce_op is "lse",
        otherwise only the output tensor after reduction
    """
    # NOTE: only sum-reduce has non-deterministic kernel by now
    deterministic |= reduce_op != "sum"
    is_lse_reduce = reduce_op == "lse"

    # check
    assert (
        input_ranges.shape == output_ranges.shape
    ), f"{input_ranges=} and {output_ranges=} must have the same shape"
    if is_lse_reduce:
        assert (
            input_lse is not None and output_lse is not None
        ), "lse reduction requires input_lse and output_lse"
        assert is_fp_dtype_at_least(input_lse, torch.float32) and is_fp_dtype_at_least(
            output_lse, torch.float32
        ), "lse reduction requires input_lse and output_lse at least float32"
        assert input_lse.ndim == output_lse.ndim == 2, (
            "lse reduction requires input and output must be 2D tensors "
            "with the shape: [seqlen, nheads]"
        )
        assert input.ndim == output.ndim == 3, (
            "lse reduction requires input and output must be 3D tensors "
            "with the shape: [seqlen, nheads, head_dim]"
        )

    # Return directly if empty tensor
    if input_ranges.shape[0] == 0 or input.numel() == 0:
        if is_lse_reduce:
            return output, output_lse
        return output

    # ---   calculate meta   --- #

    # Determine the reduce dtype
    reduce_dtype = reduce_dtype or max_fp_dtype(input.dtype, torch.float32)
    reduce_dtype = torch2triton_dtype_map[reduce_dtype]

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
        # sanity check
        assert out2inp_range_map.size(0) == unique_ordered_out_ranges.size(0)

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
    # sanity check
    if deterministic:
        assert cu_range_sizes.size(0) == unique_ordered_out_ranges.size(0) + 1
    else:
        assert cu_range_sizes.size(0) == input_ranges.size(0) + 1

    # Calculate row_map if not provided
    row_map = kwargs.pop("row_map", None)
    if row_map is None:
        row_map = _calc_ranges_row_map(
            unique_ordered_out_ranges if deterministic else input_ranges,
            total_size,
        )
    else:
        row_map = row_map.contiguous()
    # sanity check
    assert row_map.size(0) == total_size

    # ---   pre-process input/output   --- #

    output_ = output
    need_copy_output = False
    if is_lse_reduce:
        output_lse_ = output_lse
        need_copy_output_lse = False

    # Handle the case when dim is not 0
    if dim != 0:
        need_copy_output = True
        input = input.transpose(0, dim).contiguous()
        output_ = output_.transpose(0, dim).contiguous()
        if is_lse_reduce:
            need_copy_output_lse = True
            input_lse = input_lse.transpose(0, dim).contiguous()  # type: ignore[union-attr]
            output_lse_ = output_lse_.transpose(0, dim).contiguous()  # type: ignore[union-attr]
    else:
        need_copy_output |= not output.is_contiguous()
        input = input.contiguous()
        output_ = output_.contiguous()
        if is_lse_reduce:
            need_copy_output_lse |= not output_lse.is_contiguous()  # type: ignore[union-attr]
            input_lse = input_lse.contiguous()  # type: ignore[union-attr]
            output_lse_ = output_lse_.contiguous()  # type: ignore[union-attr]

    if not deterministic and output.dtype == torch.bfloat16:
        # NOTE: in non-deterministic mode, we will use triton atomic op
        # which does not support bfloat16, w.r.t. the issue:
        # https://github.com/pytorch/pytorch/issues/97016
        output_ = output_.to(torch.float32)
        need_copy_output = True

    # Calculate stride
    if is_lse_reduce:
        input_stride_s, input_stride_nh, _ = input.stride()
        output_stride_s, output_stride_nh, _ = output_.stride()
        input_lse_stride_s, _ = input_lse.stride()  # type: ignore[union-attr]
        output_lse_stride_s, _ = output_lse_.stride()  # type: ignore[union-attr]
    else:
        input_stride = input.stride(0)
        output_stride = output_.stride(0)

    # ---   calculate grid size   --- #

    if is_lse_reduce:
        M = total_size  # seqlen
        H = input.shape[1]  # num_heads
        N = input.numel() // input.shape[0] // H  # head dim

        ELEM_PER_BLOCK = 128 // input.element_size()
        N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

        grid = (M, H, N_BLOCK)
    else:
        M = total_size  # seqlen
        N = input.numel() // input.shape[0]  # hidden dim

        ELEM_PER_BLOCK = 2048 // input.element_size()
        N_BLOCK = triton.cdiv(N, ELEM_PER_BLOCK)

        grid = (M, N_BLOCK)  # type: ignore[assignment]

    # ---   launch kernel   --- #

    match reduce_op:
        case "sum":
            if deterministic:
                range_sum_reduce_deter_kernel[grid](
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
                    reduce_dtype,
                )
            else:
                range_sum_reduce_nondeter_kernel[grid](
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
        case "avg":
            range_avg_reduce_kernel[grid](
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
                reduce_dtype,
            )
        case "lse":
            range_lse_reduce_kernel[grid](
                input,
                input_lse,
                output_,
                output_lse_,
                input_ranges,
                unique_ordered_out_ranges,
                cu_range_sizes,
                row_map,
                out2inp_range_map,
                input_stride_s,
                input_stride_nh,
                input_lse_stride_s,
                output_stride_s,
                output_stride_nh,
                output_lse_stride_s,
                out2inp_range_map_stride,
                N,
                N_BLOCK,
                ELEM_PER_BLOCK,
                reduce_dtype,
            )
        case _:
            raise ValueError(f"Invalid reduce_op {reduce_op}")

    # ---   post-process output   --- #

    # If transposed earlier, transpose back
    if dim != 0:
        output_ = output_.transpose(0, dim)
        if is_lse_reduce:
            output_lse_ = output_lse_.transpose(0, dim)  # type: ignore[union-attr]

    if need_copy_output:
        output.data.copy_(output_)

    if is_lse_reduce:
        if need_copy_output_lse:
            output_lse.data.copy_(output_lse_)  # type: ignore[union-attr]
        return output, output_lse

    return output
