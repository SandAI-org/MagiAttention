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

import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Range fill
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Range gather
# ---------------------------------------------------------------------------


@triton.jit
def range_gather_per_range_kernel(
    input_ptr,
    output_ptr,
    ranges_ptr,
    cu_range_sizes_ptr,
    input_stride,
    output_stride,
    N_PER_ROW: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    NUM_TILE_CTAS: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr = 4,
):
    # Same tile loop as the original kernel (tl.range + base+offset pointers).
    # Extra 1D CTAs only stripe tiles: CTA (range, t) owns tiles t, t+K, ...
    # A 2D one-tile-per-CTA kernel faulted on Muon byte-wise fat ranges.
    pid = tl.program_id(0)
    range_idx = pid // NUM_TILE_CTAS
    tile_cta = pid % NUM_TILE_CTAS

    cu_range_size = tl.load(cu_range_sizes_ptr + range_idx)
    range_start = tl.load(ranges_ptr + range_idx * 2)
    range_end = tl.load(ranges_ptr + range_idx * 2 + 1)
    range_size = range_end - range_start

    num_row_blocks = (range_size + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    if tile_cta >= num_row_blocks:
        return

    row_offs = tl.arange(0, ROWS_PER_BLOCK)[:, None]
    col_offs = tl.arange(0, N_PER_ROW)[None, :]
    input_offs = (row_offs * input_stride) + col_offs
    output_offs = (row_offs * output_stride) + col_offs
    col_mask = (col_offs < input_stride) & (col_offs < output_stride)

    inp_idx = range_start * input_stride
    out_idx = cu_range_size * output_stride
    curr_inp_ptr = input_ptr + inp_idx
    curr_out_ptr = output_ptr + out_idx

    num_iters = (num_row_blocks - tile_cta + NUM_TILE_CTAS - 1) // NUM_TILE_CTAS
    for i in tl.range(num_iters, loop_unroll_factor=UNROLL_FACTOR):
        row_block_idx = tile_cta + i * NUM_TILE_CTAS
        row_start = row_block_idx * ROWS_PER_BLOCK
        inp_ptr_this_block = curr_inp_ptr + row_start * input_stride
        out_ptr_this_block = curr_out_ptr + row_start * output_stride

        row_mask = row_offs + row_start < range_size
        mask = row_mask & col_mask

        inp = tl.load(
            inp_ptr_this_block + input_offs,
            mask=mask,
        )
        tl.store(
            out_ptr_this_block + output_offs,
            inp,
            mask=mask,
            cache_modifier=".cs",  # cache streaming, since accessed once
        )


@triton.jit
def range_gather_per_row_kernel(
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


# ---------------------------------------------------------------------------
# Range reduce
# ---------------------------------------------------------------------------


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
    cnt = 1.0  # in acc_reduce mode, the old value in out counts 1
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
