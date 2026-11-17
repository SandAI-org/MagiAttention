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


@triton.jit
def _safe_subtract_exp(a, b):
    val = tl.exp(a - b)
    return tl.where(val != val, 0.0, val)  # nan -> 0.0


@triton.jit
def correct_out_lse_kernel(
    input_ptr,
    input_lse_ptr,
    output_ptr,
    output_lse_ptr,
    output_stride_s,
    output_stride_nh,
    output_stride_hd,
    input_stride_s,
    input_stride_nh,
    input_stride_hd,
    output_lse_stride_s,
    output_lse_stride_nh,
    input_lse_stride_s,
    input_lse_stride_nh,
    M,
    M_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    reduce_dtype: tl.constexpr,
):
    row_block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    curr_m_start = row_block_idx * M_BLOCK
    cols = tl.arange(0, N_BLOCK)[None, :]
    rows = tl.arange(0, M_BLOCK)[:, None]
    row_mask = curr_m_start + rows < M

    out_idx = curr_m_start * output_stride_s + head_idx * output_stride_nh
    curr_out_ptr = output_ptr + out_idx
    out_offs = rows * output_stride_s + cols * output_stride_hd

    inp_idx = curr_m_start * input_stride_s + head_idx * input_stride_nh
    curr_inp_ptr = input_ptr + inp_idx
    inp_offs = rows * input_stride_s + cols * input_stride_hd

    out_lse_idx = curr_m_start * output_lse_stride_s + head_idx * output_lse_stride_nh
    curr_out_lse_ptr = output_lse_ptr + out_lse_idx
    out_lse_offs = rows * output_lse_stride_s

    inp_lse_idx = curr_m_start * input_lse_stride_s + head_idx * input_lse_stride_nh
    curr_inp_lse_ptr = input_lse_ptr + inp_lse_idx
    inp_lse_offs = rows * input_lse_stride_s

    # load output
    out = tl.load(
        curr_out_ptr + out_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load output lse
    out_lse = tl.load(
        curr_out_lse_ptr + out_lse_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load input
    inp = tl.load(
        curr_inp_ptr + inp_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # load input lse
    inp_lse = tl.load(
        curr_inp_lse_ptr + inp_lse_offs,
        mask=row_mask,
    ).to(reduce_dtype)

    # correct lse
    # formula derivation:
    # reduced_lse = log(exp(lse1) + exp(lse2))
    #             = lse1 + log(1 + exp(lse2 - lse1))
    #             = max_lse + log(1 + exp(min_lse - max_lse))
    #             = max_lse + log1p(exp(min_lse - max_lse))
    min_lse = tl.minimum(inp_lse, out_lse)
    max_lse = tl.maximum(inp_lse, out_lse)
    reduced_lse = max_lse + libdevice.log1p(_safe_subtract_exp(min_lse, max_lse))

    # get reduce weights: exp(lse_i - reduced_lse)
    out_weight = _safe_subtract_exp(out_lse, reduced_lse)
    inp_weight = _safe_subtract_exp(inp_lse, reduced_lse)

    # reduce output: w1 * out + w2 * inp
    out = out_weight * out + inp_weight * inp

    # reduce output lse
    out_lse = reduced_lse

    # store reduced output
    tl.store(curr_out_ptr + out_offs, out, mask=row_mask)

    # store reduced output lse
    tl.store(curr_out_lse_ptr + out_lse_offs, out_lse, mask=row_mask)
