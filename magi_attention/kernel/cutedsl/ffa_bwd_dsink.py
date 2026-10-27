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


"""Reduce sink gradients over every physical query row, so rows outside the q
ranges also contribute.

Each CTA computes per-sink partials from ``-sum(O * dO) * exp(sink - LSE)``.
The last CTA per head reduces the partials in a fixed order and writes dsink.
"""

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor

from magi_attention.utils.dtype import to_cute_dtype

from .cache_utils import get_jit_cache
from .cutedsl_utils import warp_reduce

# Query rows per CTA, with one row per thread.
DSINK_TILE_M = 128


class FFABwdDSink:
    """Compute delta once per row and reduce each sink with shared memory independent of n_sink."""

    def __init__(self, o_dtype: Type[cutlass.Numeric], head_dim_v: int):
        self.o_dtype = o_dtype
        self.head_dim_v = head_dim_v
        self.tile_m = DSINK_TILE_M
        self.num_warps = self.tile_m // cute.arch.WARP_SIZE
        # Vectorized O/dO loads (16 bytes)
        self.vec_elems = 128 // o_dtype.width
        assert head_dim_v % self.vec_elems == 0
        assert self.tile_m % cute.arch.WARP_SIZE == 0

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (total_q, num_head, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mLSE: cute.Tensor,  # (total_q, num_head) fp32
        mSink: cute.Tensor,  # (n_sink, num_head) fp32
        mdSink: cute.Tensor,  # (n_sink, num_head) fp32, arbitrary strides
        mPartial: cute.Tensor,  # (num_m_blocks, n_sink, num_head) fp32
        mCounter: cute.Tensor,  # (num_head,) int32, initially zero
        stream: cuda.CUstream = None,
    ):
        # Cover every query row once, independent of attention ranges.
        grid = (cute.ceil_div(mO.shape[0], self.tile_m), mO.shape[1], 1)
        self.kernel(mO, mdO, mLSE, mSink, mdSink, mPartial, mCounter).launch(
            grid=grid, block=[self.tile_m, 1, 1], stream=stream
        )

    @cute.jit
    def block_sum(self, value: Float32, sWarpSum: cute.Tensor, tidx: Int32):
        """Return the block sum in thread 0; retire shared reads before reuse."""
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        value = warp_reduce(value, lambda a, b: a + b)
        if lane_idx == 0:
            sWarpSum[warp_idx] = value
        cute.arch.barrier()
        result = Float32(0.0)
        if tidx == 0:
            for warp in cutlass.range_constexpr(self.num_warps):
                result += sWarpSum[warp]
        cute.arch.barrier()
        return result

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mSink: cute.Tensor,
        mdSink: cute.Tensor,
        mPartial: cute.Tensor,
        mCounter: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        m_block, head_idx = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        row = m_block * self.tile_m + tidx
        total_q = mO.shape[0]
        num_m_blocks = cute.ceil_div(total_q, self.tile_m)
        n_sink = mSink.shape[0]

        smem = cutlass.utils.SmemAllocator()
        sWarpSum = smem.allocate_tensor(
            Float32, cute.make_layout(self.num_warps), byte_alignment=16
        )
        sLast = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)

        # Empty rows contribute zero without reading potentially NaN O/dO.
        lse = -Float32.inf
        delta = Float32(0.0)
        if row < total_q:
            lse = Float32(mLSE[row, head_idx])
            if lse != -Float32.inf:
                gO_row = mO[row, head_idx, None]
                gdO_row = mdO[row, head_idx, None]
                num_vecs = self.head_dim_v // self.vec_elems
                for vec_idx in cutlass.range(num_vecs, unroll_full=True):
                    gO_vec = cute.local_tile(gO_row, (self.vec_elems,), (vec_idx,))
                    gdO_vec = cute.local_tile(gdO_row, (self.vec_elems,), (vec_idx,))
                    rO = cute.make_rmem_tensor((self.vec_elems,), self.o_dtype)
                    rdO = cute.make_rmem_tensor((self.vec_elems,), self.o_dtype)
                    cute.autovec_copy(gO_vec, rO)
                    cute.autovec_copy(gdO_vec, rdO)
                    for elem_idx in cutlass.range(self.vec_elems, unroll_full=True):
                        delta += Float32(rO[elem_idx]) * Float32(rdO[elem_idx])

        for sink_idx in cutlass.range(n_sink):
            row_grad = Float32(0.0)
            sink = Float32(mSink[sink_idx, head_idx])
            if lse != -Float32.inf and sink != -Float32.inf:
                probability = cute.math.exp2(
                    (sink - lse) * math.log2(math.e), fastmath=False
                )
                row_grad = -delta * probability
            block_grad = self.block_sum(row_grad, sWarpSum, tidx)
            if tidx == 0:
                if num_m_blocks == 1:
                    mdSink[sink_idx, head_idx] = block_grad
                else:
                    mPartial[m_block, sink_idx, head_idx] = block_grad

        if num_m_blocks > 1:
            if tidx == 0:
                # This thread writes every partial. The acq_rel add releases
                # this CTA's partials and acquires those of every CTA that
                # added before it.
                completed = cute.arch.atomic_add(
                    (mCounter.iterator + head_idx).llvm_ptr,
                    Int32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
                sLast[0] = Int32(completed == num_m_blocks - 1)
            # Share the last writer's acquired visibility with all reducer threads.
            cute.arch.barrier()
            if sLast[0] != 0:
                for sink_idx in cutlass.range(n_sink):
                    thread_grad = Float32(0.0)
                    for block in cutlass.range(tidx, num_m_blocks, self.tile_m):
                        thread_grad += mPartial[block, sink_idx, head_idx]
                    grad = self.block_sum(thread_grad, sWarpSum, tidx)
                    if tidx == 0:
                        mdSink[sink_idx, head_idx] = grad


_COMPILE_CACHE = get_jit_cache("bwd_dsink")


def _compile_bwd_dsink(o_torch_dtype: torch.dtype, head_dim_v: int):
    cache_key = (o_torch_dtype, head_dim_v)
    cache = _COMPILE_CACHE
    if cache_key not in cache:
        o_dtype = to_cute_dtype(o_torch_dtype)
        kernel = FFABwdDSink(o_dtype, head_dim_v)
        sym = cute.sym_int
        total_q, num_head, n_sink = sym(), sym(), sym()
        vec_elems = 128 // o_dtype.width
        mO = fake_tensor(
            o_dtype, (total_q, num_head, head_dim_v), divisibility=vec_elems
        )
        mdO = fake_tensor(
            o_dtype, (total_q, num_head, head_dim_v), divisibility=vec_elems
        )
        mLSE = fake_tensor(Float32, (total_q, num_head), divisibility=1)
        mSink = fake_tensor(Float32, (n_sink, num_head), divisibility=1)
        mdSink = cute.runtime.make_fake_tensor(
            Float32,
            (n_sink, num_head),
            stride=(cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        mPartial = fake_tensor(Float32, (sym(), n_sink, num_head), divisibility=1)
        mCounter = fake_tensor(Int32, (sym(),), divisibility=1)
        cache[cache_key] = cute.compile(
            kernel,
            mO,
            mdO,
            mLSE,
            mSink,
            mdSink,
            mPartial,
            mCounter,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return cache[cache_key]


def bwd_dsink(
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    dsink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute FP32 sink gradients of shape ``[n_sink, num_head]``.

    ``out`` and ``dout`` have shape ``[total_q, num_head, head_dim_v]`` and
    share a dtype. ``lse`` is FP32 ``[total_q, num_head]`` including sinks;
    ``sink`` is FP32 ``[n_sink, num_head]`` with any ``n_sink``. Rows with
    ``lse == -inf`` contribute zero and their ``out``/``dout`` are not read.
    A head whose sink logits are all ``-inf`` gets a zero gradient. A supplied
    ``dsink`` is overwritten.
    """
    total_q, num_head, head_dim_v = out.shape
    n_sink = sink.shape[0]
    assert out.dtype == dout.dtype
    assert lse.dtype == torch.float32 and sink.dtype == torch.float32
    assert lse.shape == (total_q, num_head) and sink.shape == (n_sink, num_head)
    if dsink is None:
        dsink = torch.empty_like(sink)
    else:
        assert dsink.dtype == torch.float32 and dsink.shape == sink.shape
    if total_q == 0:
        return dsink.zero_()
    if n_sink == 0 or num_head == 0:
        return dsink
    out, dout, lse, sink = [t.contiguous() for t in (out, dout, lse, sink)]

    num_m_blocks = (total_q + DSINK_TILE_M - 1) // DSINK_TILE_M
    partial = torch.empty(
        num_m_blocks if num_m_blocks > 1 else 0,
        n_sink,
        num_head,
        dtype=torch.float32,
        device=out.device,
    )
    # Same-stream initialization also resets completion counts on graph replay.
    counter = torch.zeros(
        num_head if num_m_blocks > 1 else 0, dtype=torch.int32, device=out.device
    )
    compiled = _compile_bwd_dsink(out.dtype, head_dim_v)
    compiled(out, dout, lse, sink, dsink, partial, counter)
    return dsink
