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

"""Forward postprocess for the atomic-reduction path.

Port of flash_fwd_postprocess_kernel.h: rows whose merged LSE stayed -inf get
O written to zero (the uncovered/empty-row contract), and the attention sink
is folded in globally — one rescale per row, so the locked epilogue never has
to reason about sink x overlap.  No dtype conversion happens here: like the
C++ path, atomic-reduction O stays fp32 for the caller.
"""

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from quack.compile_utils import make_fake_tensor as fake_tensor

from .cache_utils import get_jit_cache


class FFAFwdPostProcess:
    def __init__(
        self,
        o_dtype: Type[cutlass.Numeric],
        head_dim_v: int,
        has_sink: bool,
        tile_m: int = 128,
    ):
        self.o_dtype = o_dtype
        self.head_dim_v = head_dim_v
        self.has_sink = has_sink
        self.tile_m = tile_m
        # 16B vectors over the row; O rows are dtype-contiguous.
        self.vec_elems = 128 // o_dtype.width

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mSink: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        assert (mSink is not None) == self.has_sink
        total_q = mO.shape[0]
        num_head = mO.shape[1]
        grid = (cute.ceil_div(total_q, self.tile_m), num_head, 1)
        self.kernel(mO, mLSE, mSink).launch(
            grid=grid, block=[self.tile_m, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mSink: Optional[cute.Tensor],
    ):
        tidx = cute.arch.thread_idx()[0]
        block, head_idx = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        row = block * self.tile_m + tidx
        total_q = mO.shape[0]

        if row < total_q:
            lse = Float32(mLSE[head_idx, row])
            is_empty = lse == -Float32.inf

            o_scale = Float32(0.0)
            if const_expr(self.has_sink):
                assert mSink is not None
                # One sink logit per head: lse_sink = log(exp(sink)) = sink.
                lse_sink = Float32(mSink[head_idx])
                LOG2_E = math.log2(math.e)
                LN2 = math.log(2.0)
                lse_final = lse_sink
                if not is_empty:
                    lse_hi = cutlass.max(lse, lse_sink)
                    lse_lo = cutlass.min(lse, lse_sink)
                    lse_ratio = cute.math.exp2(
                        (lse_lo - lse_hi) * LOG2_E, fastmath=True
                    )
                    lse_final = (
                        lse_hi
                        + cute.math.log2(Float32(1.0 + lse_ratio), fastmath=True) * LN2
                    )
                    o_scale = cute.math.exp2(
                        (lse - lse_final) * LOG2_E, fastmath=True
                    )
                mLSE[head_idx, row] = lse_final

            if is_empty or const_expr(self.has_sink):
                gO_row = mO[row, head_idx, None]
                num_vecs = self.head_dim_v // self.vec_elems
                for i in cutlass.range(num_vecs, unroll_full=True):
                    gO_vec = cute.local_tile(gO_row, (self.vec_elems,), (i,))
                    rO = cute.make_rmem_tensor((self.vec_elems,), self.o_dtype)
                    if const_expr(self.has_sink):
                        cute.autovec_copy(gO_vec, rO)
                        for j in cutlass.range(self.vec_elems, unroll_full=True):
                            rO[j] = self.o_dtype(Float32(rO[j]) * o_scale)
                    else:
                        rO.fill(0)
                    cute.autovec_copy(rO, gO_vec)


def _compile_fwd_postprocess(
    o_torch_dtype: torch.dtype,
    head_dim_v: int,
    has_sink: bool,
):
    from magi_attention.utils.dtype import to_cute_dtype

    cache_key = (o_torch_dtype, head_dim_v, has_sink)
    cache = get_jit_cache("ffa_fwd_postprocess")
    if cache_key not in cache:
        o_dtype = to_cute_dtype(o_torch_dtype)
        obj = FFAFwdPostProcess(o_dtype, head_dim_v, has_sink=has_sink)
        sym = cute.sym_int
        total_q, num_head = sym(), sym()
        div = 128 // o_dtype.width
        mO = fake_tensor(o_dtype, (total_q, num_head, head_dim_v), divisibility=div)
        mLSE = fake_tensor(Float32, (num_head, total_q), divisibility=1)
        mSink = (
            fake_tensor(cutlass.BFloat16, (num_head,), divisibility=1)
            if has_sink
            else None
        )
        cache[cache_key] = cute.compile(
            obj,
            mO,
            mLSE,
            mSink,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return cache[cache_key]


def fwd_postprocess(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor | None,
) -> None:
    """In-place: zero O rows whose LSE is -inf, fold the sink into O/LSE."""
    compiled = _compile_fwd_postprocess(out.dtype, out.shape[-1], sink is not None)
    compiled(out, lse, sink)
