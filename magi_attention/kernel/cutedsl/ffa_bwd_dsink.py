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


"""Backward reduction for shared sink logits (``sh`` layout)."""

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from quack.compile_utils import make_fake_tensor as fake_tensor

from magi_attention.utils.dtype import to_cute_dtype

from .cache_utils import get_jit_cache
from .cutedsl_utils import warp_reduce


class FFABwdDSink:
    """Reduce one query tile per head, with one query row per thread."""

    def __init__(
        self,
        o_dtype: Type[cutlass.Numeric],
        head_dim_v: int,
        n_sink: int,
        tile_m: int = 128,
    ):
        self.o_dtype = o_dtype
        self.head_dim_v = head_dim_v
        self.n_sink = n_sink
        self.tile_m = tile_m
        self.num_warps = tile_m // cute.arch.WARP_SIZE
        # Vectorized O/dO loads (16 bytes)
        self.vec_elems = 128 // o_dtype.width
        assert head_dim_v % self.vec_elems == 0
        assert tile_m % cute.arch.WARP_SIZE == 0
        # One writer per sink in the block reduction
        assert n_sink <= tile_m

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (total_q, num_head, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mLSE: cute.Tensor,  # (total_q, num_head) fp32
        mSink: cute.Tensor,  # (n_sink, num_head) fp32
        mdSinkPartial: cute.Tensor,  # (n_sink, num_head, num_m_blocks) fp32
        stream: cuda.CUstream = None,
    ):
        total_q = mO.shape[0]
        num_head = mO.shape[1]
        # Cover every query row once, independent of attention ranges
        grid = (cute.ceil_div(total_q, self.tile_m), num_head, 1)
        self.kernel(mO, mdO, mLSE, mSink, mdSinkPartial).launch(
            grid=grid, block=[self.tile_m, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mSink: cute.Tensor,
        mdSinkPartial: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        m_block, head_idx = cute.arch.block_idx()[0], cute.arch.block_idx()[1]
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        row = m_block * self.tile_m + tidx
        total_q = mO.shape[0]

        smem = cutlass.utils.SmemAllocator()
        sWarpDSink = smem.allocate_tensor(
            Float32, cute.make_layout((self.n_sink, self.num_warps)), byte_alignment=16
        )

        # Compute row gradients
        # NOTES: tail rows and rows with LSE == -inf contribute zero
        rdSink = cute.make_rmem_tensor((self.n_sink,), Float32)
        rdSink.fill(0.0)
        if row < total_q:
            lse = Float32(mLSE[row, head_idx])
            if lse != -Float32.inf:
                gO_row = mO[row, head_idx, None]
                gdO_row = mdO[row, head_idx, None]
                # Compute delta = sum(O * dO)
                delta = Float32(0.0)
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

                # Compute row-wise dsink = -exp(sink - lse) * delta
                LOG2_E = math.log2(math.e)
                for sink_idx in cutlass.range_constexpr(self.n_sink):
                    p_sink = cute.math.exp2(
                        (Float32(mSink[sink_idx, head_idx]) - lse) * LOG2_E,
                        fastmath=False,
                    )
                    rdSink[sink_idx] = -delta * p_sink

        # Reduce dsink (warp-level)
        for sink_idx in cutlass.range_constexpr(self.n_sink):
            rdSink[sink_idx] = warp_reduce(rdSink[sink_idx], lambda a, b: a + b)
        if lane_idx == 0:
            for sink_idx in cutlass.range_constexpr(self.n_sink):
                sWarpDSink[sink_idx, warp_idx] = rdSink[sink_idx]

        # Reduce dsink (block-level)
        cute.arch.barrier()
        if tidx < self.n_sink:
            dsink_sum = Float32(0.0)
            for warp in cutlass.range_constexpr(self.num_warps):
                dsink_sum += sWarpDSink[tidx, warp]
            mdSinkPartial[tidx, head_idx, m_block] = dsink_sum


_COMPILE_CACHE = get_jit_cache("bwd_dsink")


def _compile_bwd_dsink(o_torch_dtype: torch.dtype, head_dim_v: int, n_sink: int):
    cache_key = (o_torch_dtype, head_dim_v, n_sink)
    cache = _COMPILE_CACHE
    if cache_key not in cache:
        o_dtype = to_cute_dtype(o_torch_dtype)
        kernel = FFABwdDSink(o_dtype, head_dim_v, n_sink)
        sym = cute.sym_int
        total_q, num_head, num_m_blocks = sym(), sym(), sym()
        vec_elems = 128 // o_dtype.width
        mO = fake_tensor(
            o_dtype, (total_q, num_head, head_dim_v), divisibility=vec_elems
        )
        mdO = fake_tensor(
            o_dtype, (total_q, num_head, head_dim_v), divisibility=vec_elems
        )
        mLSE = fake_tensor(Float32, (total_q, num_head), divisibility=1)
        mSink = fake_tensor(Float32, (n_sink, num_head), divisibility=1)
        mdSinkPartial = fake_tensor(
            Float32, (n_sink, num_head, num_m_blocks), divisibility=1
        )
        cache[cache_key] = cute.compile(
            kernel,
            mO,
            mdO,
            mLSE,
            mSink,
            mdSinkPartial,
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
    tile_m: int = 128,
) -> torch.Tensor:
    """Compute FP32 sink gradients of shape ``[n_sink, num_head]``.

    ``out`` and ``dout`` have shape ``[total_q, num_head, head_dim_v]`` and
    share a dtype. ``lse`` is FP32 ``[total_q, num_head]`` including sinks;
    ``sink`` is FP32 ``[n_sink, num_head]``. A supplied ``dsink`` is overwritten.
    """
    total_q, num_head, head_dim_v = out.shape
    n_sink = sink.shape[0]
    assert out.dtype == dout.dtype
    assert lse.dtype == torch.float32 and sink.dtype == torch.float32
    assert lse.shape == (total_q, num_head) and sink.shape == (n_sink, num_head)
    out, dout, lse, sink = [t.contiguous() for t in (out, dout, lse, sink)]

    num_m_blocks = (total_q + tile_m - 1) // tile_m
    partial = torch.empty(
        n_sink, num_head, num_m_blocks, dtype=torch.float32, device=out.device
    )
    compiled = _compile_bwd_dsink(out.dtype, head_dim_v, n_sink)
    compiled(out, dout, lse, sink, partial)

    # Reduce dsink (grid-level)
    # NOTES: partials are indexed by query block, not CTA completion order
    if dsink is None:
        return partial.sum(dim=-1)
    assert dsink.dtype == torch.float32 and dsink.shape == sink.shape
    torch.sum(partial, dim=-1, out=dsink)
    return dsink
