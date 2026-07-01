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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

# mypy: disable-error-code="union-attr,index,operator,assignment,attr-defined"

# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
#
# Computes D_i = (dO_i * O_i).sum(dim=-1), optionally adjusted for LSE gradient:
#   D'_i = D_i - dLSE_i
# This works because in the backward pass:
#   dS_ij = P_ij * (dP_ij - D_i)                     [standard]
# When LSE is differentiable, d(loss)/d(S_ij) gets an extra term dLSE_i * P_ij
# (since d(LSE_i)/d(S_ij) = P_ij), giving:
#   dS_ij = P_ij * (dP_ij - D_i) + dLSE_i * P_ij
#         = P_ij * (dP_ij - (D_i - dLSE_i))
# So the main backward kernel is unchanged; we just replace D with D' = D - dLSE here.

import math
import operator
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import Arch, BaseDSL

# isort: split
from quack import copy_utils, layout_utils
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils
from .cache_utils import get_jit_cache
from .ffa_utils import make_fake_bwd_tensors
from .seqlen_info import SeqlenInfo
from .tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FFABwdPreProcess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int = 128,
        num_threads: int = 256,
        use_padded_offsets: bool = True,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a
        self.dtype = dtype
        self.tile_m = tile_m
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim = head_dim
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.num_threads = num_threads
        self.use_padded_offsets = use_padded_offsets

    def _check_tile(self) -> None:
        """Validate the kernel config (dtype, head dim, threads)."""
        if self.dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise ValueError(f"Only Float16/BFloat16 is supported, got {self.dtype}")
        if self.head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {self.head_dim}")
        if self.num_threads % 32 != 0:
            raise ValueError(
                f"num_threads must be a multiple of 32, got {self.num_threads}"
            )
        # num_threads must cover tile_m for multiplying lse with log2.
        if self.num_threads < self.tile_m:
            raise ValueError(
                f"num_threads ({self.num_threads}) must be >= tile_m ({self.tile_m})"
            )

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_v_padded % 128 == 0
            else (
                64
                if self.head_dim_v_padded % 64 == 0
                else (32 if self.head_dim_v_padded % 32 == 0 else 16)
            )
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // Float32.width
        assert (
            self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            Float32, self.num_threads, num_copy_elems_dQaccum
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (batch, seqlen, nheads, head_dim_v) or (total_q, nheads, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mPdPsum: cute.Tensor,  # (batch, nheads, seqlen_padded) or (nheads, total_q_padded)
        mLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mLSElog2: Optional[cute.Tensor],  # same shape as mPdPsum
        # (batch, nheads, seqlen_padded * head_dim_v) or (nheads, total_q_padded * head_dim_v)
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],  # (batch + 1,)
        mSeqUsedQ: Optional[cute.Tensor],  # (batch,)
        mdLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            if const_expr(mdQaccum.element_type not in [Float32]):
                raise TypeError("dQaccum tensor must be Float32")
        if const_expr(mLSE is not None):
            assert (
                mLSElog2 is not None
            ), "If mLSE is provided, mLSElog2 must also be provided"
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        if const_expr(mdLSE is not None):
            if const_expr(mdLSE.element_type not in [Float32]):
                raise TypeError("dLSE tensor must be Float32")

        self._check_tile()
        self._setup_attributes()

        # (batch, nheads, seqlen) -> (seqlen, nheads, batch) or (total_q, nheads) -> (nheads, total_q)
        transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mPdPsum = layout_utils.select(mPdPsum, transpose)
        if const_expr(mLSE is not None):
            mLSE = layout_utils.select(mLSE, transpose)
            mLSElog2 = layout_utils.select(mLSElog2, transpose)
        if const_expr(mdLSE is not None):
            mdLSE = layout_utils.select(mdLSE, transpose)
        if const_expr(mdQaccum is not None):
            mdQaccum = layout_utils.select(mdQaccum, transpose)

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
            num_head = mO.shape[1]
            num_batch = mCuSeqlensQ.shape[0] - 1
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[0]

        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mO.shape[1], self.tile_m),
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[2],
            total_q=mO.shape[0],
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            mdLSE,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        # This kernel is launched with use_pdl=True, so the GPU may start executing it in
        # "prologue" mode while the previous stream kernel is still running. We must wait
        # before touching any upstream GMEM outputs (mO, mdO, mLSE); otherwise we risk
        # reading a partially-written dout, which silently corrupts dpsum = sum(O * dO) and
        # propagates to dQ/dK via dS = P * (dP - dpsum).
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen = SeqlenInfo.create(
                batch_idx, mO.shape[1], mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m
            )
            mO_cur = seqlen.offset_batch(mO, batch_idx, dim=0)[None, head_idx, None]
            mdO_cur = seqlen.offset_batch(mdO, batch_idx, dim=0)[None, head_idx, None]
            # Stats buffers (dpsum/lse_log2) are always consumed with padded q-offsets
            # on the generic backward path (mdQaccum is present). Keep dedicated hd256
            # behavior controlled by self.use_padded_offsets.
            stats_use_padded_offsets = self.use_padded_offsets
            if const_expr(mdQaccum is not None):
                stats_use_padded_offsets = True
            mPdPsum_cur = seqlen.offset_batch(
                mPdPsum, batch_idx, dim=2, padded=stats_use_padded_offsets
            )[None, head_idx]
            headdim_v = mO_cur.shape[cute.rank(mO_cur) - 1]
            seqlen_q = seqlen.seqlen
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            # Each copy will use the same predicate
            copy = partial(copy_utils.copy, pred=tOpO)

            tOrO = cute.make_rmem_tensor_like(tOgO)
            tOrdO = cute.make_rmem_tensor_like(tOgdO)
            if const_expr(self.check_hdim_v_oob):
                tOrO.fill(0.0)
                tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            for m in cutlass.range(cute.size(tOrO.shape[1]), unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                # This is bc the entries of t0OcO are known at compile time.
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    copy(tOgO[None, m, None], tOrO[None, m, None])
                    copy(tOgdO[None, m, None], tOrdO[None, m, None])
            # O and dO loads are done; signal that the next kernel can start.
            # Correctness is ensured by griddepcontrol_wait() in bwd_sm90 before it reads our outputs.
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            # Sum across the "k" dimension
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            pdpsum = cutedsl_utils.warp_reduce(
                pdpsum, operator.add, width=threads_per_row
            )
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # If dLSE is provided, compute D' = D - dLSE (see module docstring for derivation).
            gdLSE = None
            if const_expr(mdLSE is not None):
                mdLSE_cur = seqlen.offset_batch(mdLSE, batch_idx, dim=2)[None, head_idx]
                gdLSE = cute.local_tile(mdLSE_cur, (self.tile_m,), (m_block,))

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]
                    gPdPsum[row] = PdPsum_val

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                mdQaccum_cur = seqlen.offset_batch(
                    mdQaccum,
                    batch_idx,
                    dim=2,
                    padded=True,
                    multiple=self.head_dim_padded,
                )[None, head_idx]
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            if const_expr(mLSE is not None):
                mLSElog2_cur = seqlen.offset_batch(
                    mLSElog2, batch_idx, dim=2, padded=stats_use_padded_offsets
                )[None, head_idx]
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dq_accum,
    use_padded_offsets,
):
    """Compile bwd preprocess kernel using cute fake tensors (no real GPU tensors needed)."""
    (
        mQ,
        mK,
        mV,
        mO,
        mdO,
        mdQ,
        mdK,
        mdV,
        mLSE,
        mLSElog2,
        mPdPsum,
        mdQaccum,
        mdKaccum,
        mdVaccum,
    ) = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = (
        fake_tensor(cutlass.Int32, (batchp1,), divisibility=1)
        if has_cuseqlens_q
        else None
    )
    mSequsedQ = (
        fake_tensor(cutlass.Int32, (batch,), divisibility=1) if has_seqused_q else None
    )
    mdLSE = (
        fake_tensor(cutlass.Float32, mLSE.shape, divisibility=1) if has_dlse else None
    )
    mdQaccum = mdQaccum if has_dq_accum else None
    fa_bwd_pre = FFABwdPreProcess(
        dtype, head_dim, head_dim_v, m_block_size, use_padded_offsets=use_padded_offsets
    )
    return cute.compile(
        fa_bwd_pre,
        mO,
        mdO,
        mPdPsum,
        mLSE,
        mLSElog2,
        mdQaccum,
        mCuSeqlensQ,
        mSequsedQ,
        mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def bwd_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    dpsum: torch.Tensor,
    lse: torch.Tensor,
    lse_log2: torch.Tensor,
    dq_accum: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None,
    seqused_q: torch.Tensor | None,
    dlse: torch.Tensor | None,
    dtype: torch.dtype,
    head_dim: int,
    head_dim_v: int,
    m_block_size: int,
    use_padded_offsets: bool = True,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        is_varlen,
        seqused_q is not None,
        dlse is not None,
        dq_accum is not None,
        use_padded_offsets,
    )
    if compile_key not in bwd_preprocess.compile_cache:
        bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(
            *compile_key
        )
    bwd_preprocess.compile_cache[compile_key](
        out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
    )


bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")
