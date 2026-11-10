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

# mypy: disable-error-code="attr-defined"

"""Forward postprocess for the range kernels.

Every caller ends with O = 0 and the sink-only LSE (``-inf`` without a sink)
on the rows no relation wrote, which is what a subsequent merge reads. Two
in-place variants produce that state:

- the atomic merge reduces partial O/LSE from every relation that covers a Q
  row, so those rows hold neither the sink nor the final normalization and it
  folds the sink into them;
- the direct store writes every covered row once with the sink already folded
  in, so only the uncovered rows are filled.
"""

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr
from quack.compile_utils import make_fake_tensor as fake_tensor

from .cache_utils import get_jit_cache


class FFAFwdPostProcess:
    """Fill the O/LSE rows that hold no result.

    ``has_sink`` makes every uncovered row carry ``lse_sink``.
    ``sink_in_written_rows`` says the rows holding a result already include the
    sink, so those rows stay untouched; otherwise their LSE is merged with
    ``lse_sink`` and O is rescaled to match.
    """

    def __init__(
        self,
        o_dtype: Type[cutlass.Numeric],
        head_dim_v: int,
        has_sink: bool,
        sink_in_written_rows: bool,
        tile_m: int = 128,
    ):
        assert (
            has_sink or not sink_in_written_rows
        ), "sink_in_written_rows requires a sink"
        self.o_dtype = o_dtype
        self.head_dim_v = head_dim_v
        self.has_sink = has_sink
        self.sink_in_written_rows = sink_in_written_rows
        self.tile_m = tile_m
        # 16B vectors over the row; O rows are dtype-contiguous.
        self.vec_elems = 128 // o_dtype.width

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mLSESink: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        assert (mLSESink is not None) == self.has_sink
        total_q = mO.shape[0]
        num_head = mO.shape[1]
        grid = (cute.ceil_div(total_q, self.tile_m), num_head, 1)
        self.kernel(mO, mLSE, mLSESink).launch(
            grid=grid, block=[self.tile_m, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mLSESink: Optional[cute.Tensor],
    ):
        tidx = cute.arch.thread_idx()[0]
        block, head_idx = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        row = block * self.tile_m + tidx
        total_q = mO.shape[0]

        if row < total_q:
            lse = Float32(mLSE[row, head_idx])
            is_empty = lse == -Float32.inf

            o_scale = Float32(0.0)
            if const_expr(self.has_sink):
                assert mLSESink is not None
                lse_sink = Float32(mLSESink[head_idx])
                if is_empty:
                    # No relation wrote this row: only the sinks contribute.
                    mLSE[row, head_idx] = lse_sink
                elif const_expr(not self.sink_in_written_rows):
                    LOG2_E = math.log2(math.e)
                    LN2 = math.log(2.0)
                    lse_hi = cutlass.max(lse, lse_sink)
                    lse_lo = cutlass.min(lse, lse_sink)
                    lse_ratio = cute.math.exp2(
                        (lse_lo - lse_hi) * LOG2_E, fastmath=True
                    )
                    lse_final = (
                        lse_hi
                        + cute.math.log2(Float32(1.0 + lse_ratio), fastmath=True) * LN2
                    )
                    mLSE[row, head_idx] = lse_final
                    o_scale = cute.math.exp2((lse - lse_final) * LOG2_E, fastmath=True)

            gO_row = mO[row, head_idx, None]
            num_vecs = self.head_dim_v // self.vec_elems
            if is_empty:
                # Store zeros without reading O: an uncovered row's O is
                # uninitialized and may hold NaN/Inf, which a 0 scale keeps.
                for i in cutlass.range(num_vecs, unroll_full=True):
                    gO_vec = cute.local_tile(gO_row, (self.vec_elems,), (i,))
                    rO = cute.make_rmem_tensor((self.vec_elems,), self.o_dtype)
                    rO.fill(0)
                    cute.autovec_copy(rO, gO_vec)
            if const_expr(self.has_sink):
                if const_expr(not self.sink_in_written_rows):
                    if not is_empty:
                        for i in cutlass.range(num_vecs, unroll_full=True):
                            gO_vec = cute.local_tile(gO_row, (self.vec_elems,), (i,))
                            rO = cute.make_rmem_tensor((self.vec_elems,), self.o_dtype)
                            cute.autovec_copy(gO_vec, rO)
                            for j in cutlass.range(self.vec_elems, unroll_full=True):
                                rO[j] = self.o_dtype(Float32(rO[j]) * o_scale)
                            cute.autovec_copy(rO, gO_vec)


def _compile_fwd_postprocess(
    o_torch_dtype: torch.dtype,
    head_dim_v: int,
    has_sink: bool,
    sink_in_written_rows: bool,
):
    from magi_attention.utils.dtype import to_cute_dtype

    cache_key = (o_torch_dtype, head_dim_v, has_sink, sink_in_written_rows)
    cache = fwd_postprocess.compile_cache
    if cache_key not in cache:
        o_dtype = to_cute_dtype(o_torch_dtype)
        obj = FFAFwdPostProcess(
            o_dtype,
            head_dim_v,
            has_sink=has_sink,
            sink_in_written_rows=sink_in_written_rows,
        )
        sym = cute.sym_int
        total_q, num_head = sym(), sym()
        div = 128 // o_dtype.width
        mO = fake_tensor(o_dtype, (total_q, num_head, head_dim_v), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, num_head), divisibility=1)
        mLSESink = (
            fake_tensor(Float32, (num_head,), divisibility=1) if has_sink else None
        )
        cache[cache_key] = cute.compile(
            obj,
            mO,
            mLSE,
            mLSESink,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return cache[cache_key]


def fwd_postprocess(
    out: torch.Tensor,
    lse: torch.Tensor,
    lse_sink: torch.Tensor | None,
    *,
    sink_in_written_rows: bool = False,
) -> None:
    """In-place: zero O where LSE is -inf and fold the sink into O/LSE.

    Args:
        out: ``[total_q, num_head, head_dim_v]`` O buffer, fp32/fp16/bf16.
        lse: fp32 ``[total_q, num_head]`` log-sum-exp; a row is uncovered when
            its entry is ``-inf``.
        lse_sink: fp32 ``[num_head]``, each head's log-sum-exp over its sink
            logits, or None without a sink. An uncovered row stores it as its
            LSE.
        sink_in_written_rows: the rows holding a result already include the
            sink, so they are left as they are. Without a sink no row holds
            one, and the two settings do the same work.
    """
    assert lse_sink is None or lse_sink.dtype == torch.float32
    # A sink in the result rows only changes what the covered rows need; the
    # uncovered rows of a sink-free call are zeroed by either setting, so both
    # share one compiled variant.
    sink_in_written_rows = sink_in_written_rows and lse_sink is not None
    compiled = _compile_fwd_postprocess(
        out.dtype, out.shape[-1], lse_sink is not None, sink_in_written_rows
    )
    compiled(out, lse, lse_sink)


fwd_postprocess.compile_cache = get_jit_cache("ffa_fwd_postprocess")
