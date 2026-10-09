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

# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.

# pyright: reportInvalidTypeForm=false

"""SM100 FFA backward for index-sparse (DSA) attention.

Q tiles are the outer loop and gathered K tiles the inner loop, i.e. the
inner-loop-K kernel with the contiguous K/V TMA replaced by a token gather.

Sparsity semantics:

- q block ``b`` covers q tokens ``[b * tile_m, min((b + 1) * tile_m, total_q))``
  of every q head; there are no q/k ranges.
- ``index_sparse_indices[b, h_kv, :]`` lists the global K token ids in
  ``[0, total_k)`` attended by q block ``b`` for kv head ``h_kv`` (shared by all
  q heads of that group). Valid ids come first and are pairwise distinct; the
  tail is padded with -1, and ``topk_len[b, h_kv]`` counts the valid ids.

Single CTA, CLC-persistent, warp specialized (20 warps / 640 threads):

- warp 0-3   ``load_KV``: cp.async gather of the indexed K/V rows per K tile
- warp 4-7   ``compute``: T2R S/dP, softmax -> P / dS into smem, dQ epilogue per Q tile
- warp 8-11  ``reduce`` dV: T2R dV^T, atomic add into the indexed fp32 dV rows
- warp 12-15 ``reduce`` dK: T2R dK^T, atomic add into the indexed fp32 dK rows
- warp 16    ``mma``: all tcgen05 GEMMs, TMEM alloc/free
- warp 17    ``load``: TMA Q/dO and per-lane LSE/dPsum once per Q tile
- warp 18    CLC scheduler producer
- warp 19    empty (register donor)

Per K tile the MMA warp issues S = Q @ K^T, dP = dO @ V^T, dV^T = dO^T @ P,
dQ += dS @ K (then releases K), dK^T = Q^T @ dS.
"""

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import torch
from cutlass import Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import ClcDynamicPersistentTileScheduler, LayoutEnum

# isort: split
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

import magi_attention.kernel.cutedsl as magiattn_cutedsl
from magi_attention.utils.dtype import to_cute_dtype

from . import cutedsl_utils
from . import pipeline as ffa_pipeline
from . import sm100_utils
from .cache_utils import get_jit_cache
from .cutedsl_utils import ThreadCooperativeGroup, to_cute_tensor
from .ffa_bwd_postprocess import bwd_postprocess_rowmajor
from .ffa_bwd_preprocess import bwd_preprocess
from .ffa_utils import maybe_contiguous
from .named_barrier import NamedBarrierBwdSm100Index
from .seqlen_info import SeqlenInfoQK
from .tile_scheduler import (
    ClcState,
    SchedulingMode,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)


class FFABwdSm100Index:
    arch = 100
    # Q/K/V/dO all share this head dim and one load / one GEMM covers it whole.
    # dV^T / dK^T put head_dim on the MMA M axis (lanes), which the 1-CTA
    # tcgen05 MMA only allows as 128; the TMEM plan below is sized for it.
    head_dim = 128

    def __init__(
        self,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 64,
        kv_stage: int = 3,
        debug_print: bool = False,
    ):
        assert tile_m == 128, "tile_m must be 128 (one TMEM lane per Q row)"
        # reduce_dKV keeps one K row id per lane for every 32 columns
        assert tile_n % cute.arch.WARP_SIZE == 0 and tile_n <= 128

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.kv_stage = kv_stage
        self.qhead_per_kvhead = qhead_per_kvhead

        # CTA tiler: (Q, K, HD)
        self.cta_tiler = (tile_m, tile_n, self.head_dim)

        # S = Q @ K.T => (tileQ128,tileK64,tileHD128)
        self.mma_tiler_qk = (tile_m, tile_n, self.head_dim)
        # dP = dO @ V.T => (tileQ128,tileK64,tileHD128)
        self.mma_tiler_dov = (tile_m, tile_n, self.head_dim)
        # dV.T = dO.T @ P => (tileHD128,tileK64,tileQ128)
        self.mma_tiler_dop = (self.head_dim, tile_n, tile_m)
        # dK.T = Q.T @ dS => (tileHD128,tileK64,tileQ128)
        self.mma_tiler_qds = (self.head_dim, tile_n, tile_m)
        # dQ = dS @ K => (tileQ128,tileHD128,tileK64)
        self.mma_tiler_dsk = (tile_m, self.head_dim, tile_n)

        self.acc_dtype = Float32
        self.cluster_shape_mnk = (1, 1, 1)

        self.load_KV_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7)
        self.reduce_dV_warp_ids = (8, 9, 10, 11)
        self.reduce_dK_warp_ids = (12, 13, 14, 15)
        self.mma_warp_id = 16
        self.load_warp_id = 17
        self.clc_scheduler_warp_id = 18
        self.empty_warp_id = 19

        # 20 warps -> 640 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.load_KV_warp_ids,
                *self.compute_warp_ids,
                *self.reduce_dV_warp_ids,
                *self.reduce_dK_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.clc_scheduler_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100Index.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.dQ_epi_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100Index.dQEpilogue),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM buffer distribution (columns), every buffer owns its columns:
        # dQ [0, HD) stays resident for the whole Q tile,
        # S / dP / dV.T / dK.T are refreshed per K tile.
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_dQ_offset = 0
        self.tmem_S_offset = self.tmem_dQ_offset + self.head_dim
        self.tmem_dP_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dV_offset = self.tmem_dP_offset + self.tile_n
        self.tmem_dK_offset = self.tmem_dV_offset + self.tile_n
        assert self.tmem_dK_offset + self.tile_n <= self.tmem_alloc_cols

        # setmaxnreg is warpgroup-uniform: WG0 load_KV, WG1 compute,
        # WG2/WG3 reduce, WG4 mma/load/clc/empty.
        self.num_regs_load_KV = 64
        self.num_regs_compute = 120
        self.num_regs_reduce = 128
        self.num_regs_other = 40
        assert (
            self.num_regs_load_KV
            + self.num_regs_compute
            + 2 * self.num_regs_reduce
            + self.num_regs_other
        ) * 128 <= 64 * 1024

        # CLC persistent scheduling
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.CLC

        self.buffer_align_bytes = 1024

        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[bwd_sm100_index_init] "
            print()
            print(f"{prefix}Initialized FFABwdSm100Index with: ")
            print(f"{prefix}{self.head_dim=} | {self.qhead_per_kvhead=}")
            print(f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.kv_stage=}")
            print(f"{prefix}{self.mma_tiler_qk=} | {self.mma_tiler_dov=}")
            print(
                f"{prefix}{self.mma_tiler_dop=} | {self.mma_tiler_qds=} | {self.mma_tiler_dsk=}"
            )
            print(
                f"{prefix}{self.tmem_dQ_offset=} | {self.tmem_S_offset=} | {self.tmem_dP_offset=} | "
                f"{self.tmem_dV_offset=} | {self.tmem_dK_offset=}"
            )
            print(
                f"{prefix}{self.num_regs_load_KV=} | {self.num_regs_compute=} | "
                f"{self.num_regs_reduce=} | {self.num_regs_other=}"
            )
            print()

    def _setup_attributes(self):
        self.QdO_stage = 1
        self.LSE_stage = 1
        self.single_stage = 1

        # T2R column chunk: one `tcgen05.ld.32x32b.x32` per chunk
        self.t2r_ncol = 32
        assert self.tile_n % self.t2r_ncol == 0
        assert self.head_dim % self.t2r_ncol == 0
        # reduce_dKV: chunk c of a K tile maps to the lane-distributed row ids of register c
        assert self.t2r_ncol == cute.arch.WARP_SIZE

        # K/V gather: 128-bit cp.async, one 128B swizzle row (64 elems) per thread row
        self.gather_copy_bits = 128
        self.gather_copy_elems = self.gather_copy_bits // self.k_dtype.width
        self.gather_threads_per_row = 64 // self.gather_copy_elems
        self.num_load_KV_threads = cute.arch.WARP_SIZE * len(self.load_KV_warp_ids)
        assert self.head_dim % 64 == 0
        assert (
            self.tile_n % (self.num_load_KV_threads // self.gather_threads_per_row) == 0
        )

        self.cta_group = tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # --- S = Q @ K.T with (K, K) major ---
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_qk[:2],
        )

        # --- dP = dO @ V.T with (K, K) major ---
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dov[:2],
        )

        # --- dV.T = dO.T @ P with (MN, MN) major ---
        # A = dO.T (HD, Q) read from sdO, B = P (K, Q) read from sP, both MN-major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dop[:2],
        )

        # --- dK.T = Q.T @ dS with (MN, MN) major ---
        # A = Q.T (HD, Q) read from sQ, B = dS (K, Q) read from sdS, both MN-major
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_qds[:2],
        )

        # --- dQ = dS @ K with (K, MN) major ---
        # A = dS (Q, K) K-major, B = K (HD, K) MN-major read from sK
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )

        return tiled_mma_S, tiled_mma_dP, tiled_mma_dV, tiled_mma_dK, tiled_mma_dQ

    def _setup_smem_layout(self):
        # NOTE: every aliased pair below shares the same physical SW128 layout:
        # a K-major view tiles the (8 x 64) atom along M first, an MN-major view
        # tiles the (64 x 8) atom along K first, so both store
        # [64-wide chunk][other dim][64] and only the logical view differs.

        # --- S = Q @ K.T / dK.T = Q.T @ dS ---

        # sQ: operand A (K-major) of S, (tileQ128,tileHD128)
        self.sQ_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S, self.mma_tiler_qk, self.q_dtype, self.QdO_stage
        )
        # sQt: operand A (MN-major) of dK.T, (tileHD128,tileQ128), aliases sQ
        self.sQt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK, self.mma_tiler_qds, self.q_dtype, self.QdO_stage
        )
        # sK: operand B (K-major) of S, (tileK64,tileHD128)
        self.sK_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        # sKt: operand B (MN-major) of dQ, (tileHD128,tileK64), aliases sK
        self.sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.k_dtype, self.kv_stage
        )

        # --- dP = dO @ V.T / dV.T = dO.T @ P ---

        # sdO: operand A (K-major) of dP, (tileQ128,tileHD128)
        self.sdO_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP, self.mma_tiler_dov, self.do_dtype, self.QdO_stage
        )
        # sdOt: operand A (MN-major) of dV.T, (tileHD128,tileQ128), aliases sdO
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV, self.mma_tiler_dop, self.do_dtype, self.QdO_stage
        )
        # sV: operand B (K-major) of dP, (tileK64,tileHD128)
        self.sV_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP, self.mma_tiler_dov, self.v_dtype, self.kv_stage
        )

        # --- P / dS ---

        # sP: operand B (MN-major) of dV.T, (tileK64,tileQ128)
        self.sP_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV, self.mma_tiler_dop, self.ds_dtype, self.single_stage
        )
        # sdS: operand A (K-major) of dQ, (tileQ128,tileK64)
        self.sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ, self.mma_tiler_dsk, self.ds_dtype, self.single_stage
        )
        # sdSt: operand B (MN-major) of dK.T, (tileK64,tileQ128), aliases sdS
        self.sdSt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK, self.mma_tiler_qds, self.ds_dtype, self.single_stage
        )
        # sP_epi / sdS_epi: row-major (tileQ128,tileK64) R2S views of sP / sdS
        self.sPdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            self.single_stage,
        )

        # --- dQ epilogue ---

        # sdQ: row-major (tileQ128,tileHD128), reuses the sP + sdS bytes
        self.dQ_epi_tile = (self.tile_m, self.head_dim)
        self.sdQ_layout = sm100_utils_basic.make_smem_layout_epi(
            self.dq_dtype, LayoutEnum.ROW_MAJOR, self.dQ_epi_tile, 1
        )

        # --- LSE / dPsum ---

        self.sLSE_layout = cute.make_layout((self.tile_m,))
        self.sdPsum_layout = cute.make_layout((self.tile_m,))

        for a, b in (
            (self.sQ_layout, self.sQt_layout),
            (self.sK_layout, self.sKt_layout),
            (self.sdO_layout, self.sdOt_layout),
            (self.sdS_layout, self.sdSt_layout),
            (self.sP_layout, self.sdS_layout),
        ):
            assert cute.cosize(a) == cute.cosize(b), "aliased smem views differ in size"

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        mdKacc: cute.Tensor,
        mdVacc: cute.Tensor,
        softmax_scale: Float32,
        mQRanges: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: cute.Tensor,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mdO/mdQ/mdKacc/mdVacc/mLSE/mdPsum tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.dq_dtype = mdQ.element_type
        self.ds_dtype = self.q_dtype

        # (sq, nhq, hd) -> (sq, hd, nhq)
        mQ, mdO, mdQ = [layout_utils.select(t, mode=[0, 2, 1]) for t in (mQ, mdO, mdQ)]
        # (sk, nhk, hd) -> (sk, hd, nhk)
        mK, mV = [layout_utils.select(t, mode=[0, 2, 1]) for t in (mK, mV)]
        # (nhq, sq_padded) -> (sq_padded, nhq)
        mLSE, mdPsum = [layout_utils.select(t, mode=[1, 0]) for t in (mLSE, mdPsum)]

        # (nhk, sk_padded * hd) -> (sk_padded, hd, nhk), row-major fp32 accumulators
        mdKacc, mdVacc = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[1] // self.head_dim, self.head_dim, t.shape[0]),
                    stride=(self.head_dim, 1, t.stride[0]),
                ),
            )
            for t in (mdKacc, mdVacc)
        ]

        # ///////////////////////////////////////////////////////////////////////////////
        # Set up attributes, tiled MMA and SMEM layouts
        # ///////////////////////////////////////////////////////////////////////////////

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        self.cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )

        # --- Make tiled TMA G2S-copy of Q/dO (K/V are gathered by cp.async) ---

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S = Q @ K.T: Q as operand A
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            self.tiled_mma_S,
            self.cta_layout_vmnk.shape,
        )
        # dP = dO @ V.T: dO as operand A
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_dov,
            self.tiled_mma_dP,
            self.cta_layout_vmnk.shape,
        )

        # --- Make tiled TMA S2G-copy of dQ ---

        # 4D descriptor with the base shifted back by total_q rows: the row mode
        # spans total_q rows and the 4th mode re-adds rows at unit row stride, so
        # tile rows past the end of the current Q range exceed the row extent
        # and are clipped by TMA instead of clobbering the next range.
        total_q_dq = mdQ.shape[0]
        row_stride_dq = mdQ.layout.stride[0]
        mdQ_desc = cute.make_tensor(
            (mdQ.iterator - Int64(total_q_dq) * row_stride_dq).align(16),
            cute.make_layout(
                (total_q_dq, mdQ.shape[1], mdQ.shape[2], 1 << 26),
                stride=(
                    row_stride_dq,
                    mdQ.layout.stride[1],
                    mdQ.layout.stride[2],
                    row_stride_dq,
                ),
            ),
        )
        tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdQ_desc,
            cute.select(self.sdQ_layout, mode=[0, 1]),
            self.dQ_epi_tile,
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        # mQRanges must be the single range [0, total_q), so the scheduled
        # m_block is also the q block row of mIndices / mTopkLen.
        TileScheduler = SingleTileVarlenScheduler
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=cute.size(mQRanges.shape[0]),
            num_splits=1,
            seqlen_k=cute.size(mK.shape[0]),
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mQRanges=mQRanges,
            # CLC only supports the prefix-sum decode over disjoint Q ranges.
            max_outer_range_width=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.q_dtype.width // 8,
            is_persistent=False,  # persistence comes from CLC, not grid striding
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args, scheduling_mode=self.scheduling_mode
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        sPdS_elems = max(
            cute.cosize(self.sP_layout) + cute.cosize(self.sdS_layout),
            cute.cosize(self.sdQ_layout) * self.dq_dtype.width // self.ds_dtype.width,
        )
        clc_response_size = self.sched_stages * 4
        clc_mbar_size = self.sched_stages * 2

        @cute.struct
        class SharedStorage:
            # ---  mbarriers for pipelines ---

            QdO_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.QdO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.LSE_stage]
            K_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.kv_stage]
            V_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.kv_stage]
            S_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            dP_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            P_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            dS_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            dV_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            dK_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]
            dQ_mbar_ptr: cute.struct.MemRange[Int64, 2 * self.single_stage]

            # --- CLC buffers ---

            # PipelineClcFetchAsync expects 2 * sched_stages mbarriers (full + empty).
            clc_mbar_ptr: cute.struct.MemRange[Int64, clc_mbar_size]
            # CLC response storage (16 bytes per stage, stored as 4 Int32s).
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size], 16
            ]

            # --- tmem ptr ---

            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf_ptr: Int32

            # --- smem tensors ---

            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            # sP | sdS, reused as sdQ at the end of each Q tile
            sPdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, sPdS_elems],
                self.buffer_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sLSE_layout)],
                128,
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdPsum_layout)],
                128,
            ]

        self.shared_storage = SharedStorage

        # --- Make others ---

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[bwd_sm100_index_call] "
            print()
            print(f"{prefix}tiled_mma_S: {self.tiled_mma_S}")
            print(f"{prefix}tiled_mma_dP: {self.tiled_mma_dP}")
            print(f"{prefix}tiled_mma_dV: {self.tiled_mma_dV}")
            print(f"{prefix}tiled_mma_dK: {self.tiled_mma_dK}")
            print(f"{prefix}tiled_mma_dQ: {self.tiled_mma_dQ}")
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sQt_layout: {self.sQt_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sKt_layout: {self.sKt_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sdO_layout: {self.sdO_layout}")
            print(f"{prefix}sdOt_layout: {self.sdOt_layout}")
            print(f"{prefix}sP_layout: {self.sP_layout}")
            print(f"{prefix}sdS_layout: {self.sdS_layout}")
            print(f"{prefix}sdSt_layout: {self.sdSt_layout}")
            print(f"{prefix}sPdS_epi_layout: {self.sPdS_epi_layout}")
            print(f"{prefix}sdQ_layout: {self.sdQ_layout}")
            print(f"{prefix}threads_per_cta: {self.threads_per_cta}")
            print(f"{prefix}smem bytes: {self.shared_storage.size_in_bytes()}")  # type: ignore[attr-defined]
            print()

        # --- Launch the kernel ---

        self.kernel(
            tma_tensor_Q,
            mK,
            mV,
            tma_tensor_dO,
            tma_tensor_dQ,
            mLSE,
            mdPsum,
            mdKacc,
            mdVacc,
            mQRanges,
            mIndices,
            mTopkLen,
            tma_atom_Q,
            tma_atom_dO,
            tma_atom_dQ,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sP_layout,
            self.sdS_layout,
            self.sdSt_layout,
            self.sPdS_epi_layout,
            self.sdQ_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=None,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdQ: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdKacc: cute.Tensor,
        mdVacc: cute.Tensor,
        mQRanges: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sPdS_epi_layout: cute.ComposedLayout,
        sdQ_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Set up before warp specialization
        # /////////////////////////////////////////////////////////////////////////////

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # --- Prefetch TMA descriptor ---

        if warp_idx == self.load_warp_id:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_dO,
                tma_atom_dQ,
            ):
                cpasync.prefetch_descriptor(tma_atom)

        # --- Alloc smem storage and fetch ptrs ---

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # --- Alloc tmem alloc/dealloc barrier ---

        # NOTE: only the mma warp drives tmem alloc/dealloc; the compute and reduce
        # warps also arrive on this barrier so the mma warp knows tmem is no longer
        # in use before it deallocates.
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100Index.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.compute_warp_ids,
                    *self.reduce_dV_warp_ids,
                    *self.reduce_dK_warp_ids,
                )
            ),
        )
        tmem = cutlass.utils.TmemAllocator(
            alloc_result_dst_smem_ptr=storage.tmem_holding_buf_ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # --- Make pipeline cooperative groups ---

        # One arrive per warp (elected), except the TMA / UMMA agents.
        load_warp = ThreadCooperativeGroup(1)
        load_KV_warps = ThreadCooperativeGroup(len(self.load_KV_warp_ids))
        mma_warp = ThreadCooperativeGroup(1)
        compute_warps = ThreadCooperativeGroup(len(self.compute_warp_ids))
        reduce_dV_warps = ThreadCooperativeGroup(len(self.reduce_dV_warp_ids))
        reduce_dK_warps = ThreadCooperativeGroup(len(self.reduce_dK_warp_ids))

        # --- Make pipelines ---

        # Q/dO pipeline (load -> MMA), once per Q tile
        pipeline_QdO = ffa_pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.QdO_mbar_ptr.data_ptr(),
            num_stages=self.QdO_stage,
            producer_group=load_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # LSE/dPsum pipeline (load -> compute), once per Q tile
        pipeline_LSE = pipeline.PipelineAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.LSE_stage,
            producer_group=load_warp,
            consumer_group=compute_warps,
            defer_sync=True,
        )
        # K / V pipelines (load_KV cp.async gather -> MMA), per K tile
        pipeline_K = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.K_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=load_KV_warps,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_V = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.V_mbar_ptr.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=load_KV_warps,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # S / dP pipelines (MMA -> compute)
        pipeline_S = pipeline.PipelineUmmaAsync.create(
            num_stages=self.single_stage,
            producer_group=mma_warp,
            consumer_group=compute_warps,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dP = pipeline.PipelineUmmaAsync.create(
            num_stages=self.single_stage,
            producer_group=mma_warp,
            consumer_group=compute_warps,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # P / dS pipelines (compute -> MMA), sP / sdS in smem
        pipeline_P = pipeline.PipelineAsyncUmma.create(
            num_stages=self.single_stage,
            producer_group=compute_warps,
            consumer_group=mma_warp,
            barrier_storage=storage.P_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dS = pipeline.PipelineAsyncUmma.create(
            num_stages=self.single_stage,
            producer_group=compute_warps,
            consumer_group=mma_warp,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # dV / dK pipelines (MMA -> reduce)
        pipeline_dV = pipeline.PipelineUmmaAsync.create(
            num_stages=self.single_stage,
            producer_group=mma_warp,
            consumer_group=reduce_dV_warps,
            barrier_storage=storage.dV_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dK = pipeline.PipelineUmmaAsync.create(
            num_stages=self.single_stage,
            producer_group=mma_warp,
            consumer_group=reduce_dK_warps,
            barrier_storage=storage.dK_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # dQ pipeline (MMA -> compute), once per Q tile
        pipeline_dQ = pipeline.PipelineUmmaAsync.create(
            num_stages=self.single_stage,
            producer_group=mma_warp,
            consumer_group=compute_warps,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # --- Cluster arrive after mbarrier init ---

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # --- Make smem tensors ---

        # sQ: (MMA_sA=(128,16),MMA_Q1,MMA_HD=(4,2),stage1) K-major
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # sQt: MN-major view of sQ (operand A of dK.T = Q.T @ dS)
        sQt = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype),
            sQt_layout.outer,
        )
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        # sdOt: MN-major view of sdO (operand A of dV.T = dO.T @ P)
        sdOt = cute.make_tensor(
            cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
            sdOt_layout.outer,
        )
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sKt: MN-major view of sK (operand B of dQ = dS @ K)
        sKt = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sKt_layout.inner, dtype=self.k_dtype),
            sKt_layout.outer,
        )
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)

        # sP / sdS share the sPdS buffer, sdQ reuses all of it
        sPdS_ptr = storage.sPdS.data_ptr()
        sP = cute.make_tensor(
            cute.recast_ptr(sPdS_ptr, sP_layout.inner, dtype=self.ds_dtype),
            sP_layout.outer,
        )
        sP_epi = cute.make_tensor(
            cute.recast_ptr(sPdS_ptr, sPdS_epi_layout.inner, dtype=self.ds_dtype),
            sPdS_epi_layout.outer,
        )
        sdS_ptr = sPdS_ptr + cute.cosize(sP_layout)
        sdS = cute.make_tensor(
            cute.recast_ptr(sdS_ptr, sdS_layout.inner, dtype=self.ds_dtype),
            sdS_layout.outer,
        )
        sdSt = cute.make_tensor(
            cute.recast_ptr(sdS_ptr, sdSt_layout.inner, dtype=self.ds_dtype),
            sdSt_layout.outer,
        )
        sdS_epi = cute.make_tensor(
            cute.recast_ptr(sdS_ptr, sPdS_epi_layout.inner, dtype=self.ds_dtype),
            sPdS_epi_layout.outer,
        )
        sdQ = cute.make_tensor(
            cute.recast_ptr(sPdS_ptr, sdQ_layout.inner, dtype=self.dq_dtype),
            sdQ_layout.outer,
        )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)

        # --- Make tmem fragments of tS / tdP / tdV / tdK / tdQ ---

        # NOTE: we always request all 512 columns of tmem, so the allocation
        # starts at column 0 and the fragments can be built from a fake ptr.
        tmem_ptr = cute.make_ptr(
            self.acc_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        thr_mma_S = tiled_mma_S.get_slice(0)
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        # tStS: (MMA_tC=(tileQ128,tileK64),MMA_Q1,MMA_K1)
        tStS = thr_mma_S.make_fragment_C(
            thr_mma_S.partition_shape_C(self.mma_tiler_qk[:2])
        )
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # tdPtdP: (MMA_tC=(tileQ128,tileK64),MMA_Q1,MMA_K1)
        tdPtdP = thr_mma_dP.make_fragment_C(
            thr_mma_dP.partition_shape_C(self.mma_tiler_dov[:2])
        )
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # tdVtdV: (MMA_tC=(tileHD128,tileK64),MMA_HD1,MMA_K1)
        tdVtdV = thr_mma_dV.make_fragment_C(
            thr_mma_dV.partition_shape_C(self.mma_tiler_dop[:2])
        )
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        # tdKtdK: (MMA_tC=(tileHD128,tileK64),MMA_HD1,MMA_K1)
        tdKtdK = thr_mma_dK.make_fragment_C(
            thr_mma_dK.partition_shape_C(self.mma_tiler_qds[:2])
        )
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        # tdQtdQ: (MMA_tC=(tileQ128,tileHD128),MMA_Q1,MMA_HD1)
        tdQtdQ = thr_mma_dQ.make_fragment_C(
            thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        )
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        # --- Make other info ---

        # Q side only: K rows are addressed globally through mIndices
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mQRanges=mQRanges,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )

        # --- Cluster wait before tensor memory alloc ---

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # --- Make CLC tile scheduler ---

        clc_pipeline_producer_group = ThreadCooperativeGroup(1)
        # Every warp in the CTA consumes each CLC response
        clc_pipeline_consumer_group = ThreadCooperativeGroup(self.threads_per_cta)
        clc = ClcState.create(
            hw_scheduler=ClcDynamicPersistentTileScheduler.create(
                self.tile_scheduler_cls.clc_problem_shape(tile_sched_params),
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                storage.clc_response.data_ptr(),
            ),
            pipeline=pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_mbar_ptr.data_ptr(),
                num_stages=self.sched_stages,
                producer_group=clc_pipeline_producer_group,
                consumer_group=clc_pipeline_consumer_group,
                tx_count=16,
                cta_layout_vmnk=cta_layout_vmnk,
            ),
            consumer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.sched_stages
            ),
            producer_state=pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.sched_stages
            ),
        )
        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params, clc=clc)
        assert isinstance(
            tile_scheduler, TileSchedulerProtocol
        ), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # ///////////////////////////////////////////////////////////////////////////////
        #  CLC Scheduler Warp / Empty Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.clc_scheduler_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.clc_scheduler_warp(tile_scheduler)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.empty_warp(tile_scheduler)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load Warp (Q / dO / LSE / dPsum)
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                mQ,
                mdO,
                mLSE,
                mdPsum,
                sQ,
                sdO,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_dO,
                pipeline_QdO,
                pipeline_LSE,
                SeqlenInfoCls,
                tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load KV Warps (K / V)
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.load_KV_warp_ids[0]
            and warp_idx <= self.load_KV_warp_ids[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_load_KV)
            self.load_KV(
                mK,
                mV,
                mIndices,
                mTopkLen,
                sK,
                sV,
                pipeline_K,
                pipeline_V,
                tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # --- Alloc and retrieve tmem buffer ---

            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # --- Enter mma loop ---

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                sP,
                sdS,
                sdSt,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_QdO,
                pipeline_K,
                pipeline_V,
                pipeline_S,
                pipeline_dP,
                pipeline_P,
                pipeline_dS,
                pipeline_dV,
                pipeline_dK,
                pipeline_dQ,
                mTopkLen,
                tile_scheduler,
            )

            # --- Dealloc tmem buffer ---

            tmem.relinquish_alloc_permit()
            tmem.wait_for_alloc()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            self.compute_loop(
                thr_mma_S,
                thr_mma_dQ,
                tStS,
                tdPtdP,
                tdQtdQ,
                sLSE,
                sdPsum,
                sP_epi,
                sdS_epi,
                sdQ,
                mdQ,
                tma_atom_dQ,
                pipeline_LSE,
                pipeline_S,
                pipeline_dP,
                pipeline_P,
                pipeline_dS,
                pipeline_dQ,
                softmax_scale,
                softmax_scale_log2,
                mTopkLen,
                SeqlenInfoCls,
                tile_scheduler,
            )

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Reduce WarpGroups (dV / dK)
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.reduce_dV_warp_ids[0]
            and warp_idx <= self.reduce_dK_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_reduce)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            if warp_idx <= self.reduce_dV_warp_ids[-1]:
                self.reduce_dKV(
                    thr_mma_dV,
                    tdVtdV,
                    mdVacc,
                    mIndices,
                    mTopkLen,
                    pipeline_dV,
                    tile_scheduler,
                )
            else:
                self.reduce_dKV(
                    thr_mma_dK,
                    tdKtdK,
                    mdKacc,
                    mIndices,
                    mTopkLen,
                    pipeline_dK,
                    tile_scheduler,
                )

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

    @cute.jit
    def _num_n_blocks(self, topk_len: Int32) -> Int32:
        """Number of gathered K tiles in the inner loop of one Q tile."""
        return (topk_len + self.tile_n - 1) // self.tile_n

    @cute.jit
    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_QdO: ffa_pipeline.PipelineTmaUmma,
        pipeline_LSE: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
    ):
        lane = cute.arch.lane_idx()

        # --- Init producer pipeline states ---

        producer_state_QdO = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.QdO_stage
        )
        producer_state_LSE = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.LSE_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)

            # //////////////////////////////////////////////
            #  Make gQ/gdO/gLSE/gdPsum
            # //////////////////////////////////////////////

            # mQ_cur / mdO_cur: (seqQ,HD)
            mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mdO_cur = seqlen_info.offset_batch_Q(mdO, batch_idx, dim=3)[
                None, None, head_idx
            ]
            # gQ / gdO: (tileQ128,tileHD128,restQ)
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0)
            )
            gdO = cute.local_tile(
                mdO_cur, cute.select(self.mma_tiler_dov, mode=[0, 2]), (None, 0)
            )
            # gLSE / gdPsum: (tileQ128,restQ)
            mLSE_cur = seqlen_info.offset_batch_Q(mLSE, batch_idx, dim=2)[
                None, head_idx
            ]
            mdPsum_cur = seqlen_info.offset_batch_Q(mdPsum, batch_idx, dim=2)[
                None, head_idx
            ]
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

            # //////////////////////////////////////////////
            #  TMA partition and G2S-load fn for sQ/sdO
            # //////////////////////////////////////////////

            tSgQ = thr_mma_S.partition_A(gQ)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                src_tensor=tSgQ,
                dst_tensor=sQ,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_QdO)
            tdPgdO = thr_mma_dP.partition_A(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                src_tensor=tdPgdO,
                dst_tensor=sdO,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_QdO)

            # --- Q / dO (once per Q tile, resident across the K loop) ---

            pipeline_QdO.producer_acquire(producer_state_QdO)
            load_Q(m_block, producer_state=producer_state_QdO)
            load_dO(m_block, producer_state=producer_state_QdO)
            pipeline_QdO.producer_commit(producer_state_QdO)
            producer_state_QdO.advance()

            # --- LSE / dPsum (token-space stats are only 4B aligned) ---

            pipeline_LSE.producer_acquire(producer_state_LSE)
            gLSE_cur = gLSE[None, m_block]
            gdPsum_cur = gdPsum[None, m_block]
            for i in cutlass.range_constexpr(self.tile_m // cute.arch.WARP_SIZE):
                row = lane + i * cute.arch.WARP_SIZE
                sLSE[row] = gLSE_cur[row]
                sdPsum[row] = gdPsum_cur[row]
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_LSE.producer_commit(producer_state_LSE)
            producer_state_LSE.advance()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def _flatten_kv_smem(self, sX: cute.Tensor) -> cute.Tensor:
        """K-major (MMA=(tileK,16),1,restHD,stage) smem -> (tileK,HD,stage) row view."""
        return cute.make_tensor(
            sX.iterator,
            cute.make_layout(
                (sX.shape[0][0], (sX.shape[0][1], sX.shape[2]), sX.shape[3]),
                stride=(
                    sX.stride[0][0],
                    (sX.stride[0][1], sX.stride[2]),
                    sX.stride[3],
                ),
            ),
        )

    @cute.jit
    def _gather_rows(
        self,
        tiled_gather: cute.TiledCopy,
        mX_cur: cute.Tensor,
        tXsX: cute.Tensor,
        tXcX: cute.Tensor,
        tXrRow: cute.Tensor,
    ):
        """Issue the cp.async copies of this thread's gathered rows into one smem stage."""
        for m in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            # 16B-aligned row base: rows are head_dim contiguous elems
            row_ptr = cute.make_ptr(
                mX_cur.element_type,
                cutedsl_utils.elem_pointer(mX_cur, (tXrRow[m], 0)).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            # gX_row: (CPY_ELEMS,HD/CPY_ELEMS)
            gX_row = cute.tiled_divide(
                cute.make_tensor(row_ptr, cute.make_layout((self.head_dim,))),
                (self.gather_copy_elems,),
            )
            for k in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                tXsX_mk = tXsX[None, m, k]
                chunk = tXcX[0, 0, k][1] // self.gather_copy_elems
                tXgX_mk = cute.make_tensor(gX_row[None, chunk].iterator, tXsX_mk.layout)
                cute.copy(tiled_gather, tXgX_mk, tXsX_mk)

    @cute.jit
    def load_KV(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        pipeline_K: pipeline.PipelineAsyncUmma,
        pipeline_V: pipeline.PipelineAsyncUmma,
        tile_scheduler: TileSchedulerProtocol,
    ):
        """Gather the indexed K/V rows of every K tile with cp.async (all load_KV warps)."""
        tidx = cute.arch.thread_idx()[0] % self.num_load_KV_threads

        # --- Make the cp.async gather copy over (tileK64,tileHD128) row views ---

        gather_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.k_dtype,
            num_bits_per_copy=self.gather_copy_bits,
        )
        # (16 rows, 8 threads) x (1, 8 elems): 16 rows x one 128B swizzle row per copy
        tiled_gather = cute.make_tiled_copy_tv(
            gather_copy_atom,
            cute.make_ordered_layout(
                (
                    self.num_load_KV_threads // self.gather_threads_per_row,
                    self.gather_threads_per_row,
                ),
                order=(1, 0),
            ),
            cute.make_layout((1, self.gather_copy_elems)),
        )
        thr_gather = tiled_gather.get_slice(tidx)
        # tKsK / tVsV: (CPY_ATOM,CPY_K4,CPY_HD2,stage)
        tKsK = thr_gather.partition_D(self._flatten_kv_smem(sK))
        tVsV = thr_gather.partition_D(self._flatten_kv_smem(sV))
        # tKVcKV: (CPY_ATOM,CPY_K4,CPY_HD2) of (row, col) coords in the K tile
        tKVcKV = thr_gather.partition_S(
            cute.make_identity_tensor((self.tile_n, self.head_dim))
        )
        num_rows_per_thr = cute.size(tKsK, mode=[1])

        # --- Init producer pipeline states ---

        producer_state_K = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        producer_state_V = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, _, _ = work_tile.tile_idx
            head_idx_kv = head_idx // self.qhead_per_kvhead
            topk_len = mTopkLen[m_block, head_idx_kv]
            n_block_max = self._num_n_blocks(topk_len)

            # mK_cur / mV_cur: (seqK,HD) over all K tokens, rows picked by mIdx_cur
            mK_cur = mK[None, None, head_idx_kv]
            mV_cur = mV[None, None, head_idx_kv]
            # mIdx_cur: (topk,) global K row ids of this Q tile
            mIdx_cur = mIndices[m_block, head_idx_kv, None]

            # //////////////////////////////////////////////
            #  K loop: gathered K/V tiles
            # //////////////////////////////////////////////

            for n_block in cutlass.range(n_block_max, unroll=1):
                col_limit = topk_len - n_block * self.tile_n

                # Padding rows gather K/V row 0: masked columns still need finite
                # K/V, otherwise dQ = dS @ K picks up 0 * NaN from stale smem.
                tKVrRow = cute.make_rmem_tensor((num_rows_per_thr,), Int32)
                for m in cutlass.range_constexpr(num_rows_per_thr):
                    row = tKVcKV[0, m, 0][0]
                    row_idx = Int32(0)
                    if row < col_limit:
                        row_idx = mIdx_cur[n_block * self.tile_n + row]
                    tKVrRow[m] = row_idx

                # --- Gather K ---

                pipeline_K.producer_acquire(producer_state_K)
                self._gather_rows(
                    tiled_gather,
                    mK_cur,
                    tKsK[None, None, None, producer_state_K.index],
                    tKVcKV,
                    tKVrRow,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

                # Commit sK to be full: make the cp.async writes visible to UMMA
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_K.producer_commit(producer_state_K)
                producer_state_K.advance()

                # --- Gather V (same rows as K) ---

                pipeline_V.producer_acquire(producer_state_V)
                self._gather_rows(
                    tiled_gather,
                    mV_cur,
                    tVsV[None, None, None, producer_state_V.index],
                    tKVcKV,
                    tKVrRow,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)

                # Commit sV to be full: make the cp.async writes visible to UMMA
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_V.producer_commit(producer_state_V)
                producer_state_V.advance()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sdSt: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_QdO: ffa_pipeline.PipelineTmaUmma,
        pipeline_K: pipeline.PipelineAsyncUmma,
        pipeline_V: pipeline.PipelineAsyncUmma,
        pipeline_S: pipeline.PipelineUmmaAsync,
        pipeline_dP: pipeline.PipelineUmmaAsync,
        pipeline_P: pipeline.PipelineAsyncUmma,
        pipeline_dS: pipeline.PipelineAsyncUmma,
        pipeline_dV: pipeline.PipelineUmmaAsync,
        pipeline_dK: pipeline.PipelineUmmaAsync,
        pipeline_dQ: pipeline.PipelineUmmaAsync,
        mTopkLen: cute.Tensor,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # --- Make GEMM fragments & define GEMM funcs ---

        # S = Q @ K.T
        tSrQ = tiled_mma_S.make_fragment_A(sQ)
        tSrK = tiled_mma_S.make_fragment_B(sK)
        mma_s_qk_fn = partial(
            sm100_utils.gemm_w_idx, tiled_mma_S, tStS, tSrQ, tSrK, zero_init=True
        )
        # dP = dO @ V.T
        tdPrdO = tiled_mma_dP.make_fragment_A(sdO)
        tdPrV = tiled_mma_dP.make_fragment_B(sV)
        mma_dp_dov_fn = partial(
            sm100_utils.gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrdO, tdPrV, zero_init=True
        )
        # dV.T = dO.T @ P
        tdVrdOt = tiled_mma_dV.make_fragment_A(sdOt)
        tdVrP = tiled_mma_dV.make_fragment_B(sP)
        mma_dv_dop_fn = partial(
            sm100_utils.gemm_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrdOt,
            tdVrP,
            A_idx=0,
            B_idx=0,
            zero_init=True,
        )
        # dK.T = Q.T @ dS
        tdKrQt = tiled_mma_dK.make_fragment_A(sQt)
        tdKrdS = tiled_mma_dK.make_fragment_B(sdSt)
        mma_dk_qds_fn = partial(
            sm100_utils.gemm_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrQt,
            tdKrdS,
            A_idx=0,
            B_idx=0,
            zero_init=True,
        )
        # dQ += dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrKt = tiled_mma_dQ.make_fragment_B(sKt)
        mma_dq_dsk_fn = partial(
            sm100_utils.gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrKt, A_idx=0
        )

        # --- Init pipeline states ---

        consumer_state_QdO = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.QdO_stage
        )
        consumer_state_K = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        consumer_state_V = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        consumer_state_P = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dS = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_state_S = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dP = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dV = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dK = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dQ = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, _, _ = work_tile.tile_idx
            topk_len = mTopkLen[m_block, head_idx // self.qhead_per_kvhead]
            n_block_max = self._num_n_blocks(topk_len)

            # Wait for sQ/sdO to be full (resident for the whole K loop)
            pipeline_QdO.consumer_wait(consumer_state_QdO)

            # Acquire tdQ to be empty (previous Q tile's dQ consumed by compute)
            pipeline_dQ.producer_acquire(producer_state_dQ)

            for n_block in cutlass.range(n_block_max, unroll=1):
                # --- GEMM: S = Q @ K.T ---

                pipeline_K.consumer_wait(consumer_state_K)
                pipeline_S.producer_acquire(producer_state_S)
                mma_s_qk_fn(A_idx=0, B_idx=consumer_state_K.index)
                pipeline_S.producer_commit(producer_state_S)
                producer_state_S.advance()

                # --- GEMM: dP = dO @ V.T ---

                pipeline_V.consumer_wait(consumer_state_V)
                pipeline_dP.producer_acquire(producer_state_dP)
                mma_dp_dov_fn(A_idx=0, B_idx=consumer_state_V.index)
                pipeline_dP.producer_commit(producer_state_dP)
                producer_state_dP.advance()

                # Release sV to be empty
                pipeline_V.consumer_release(consumer_state_V)
                consumer_state_V.advance()

                # --- GEMM: dV.T = dO.T @ P ---

                pipeline_P.consumer_wait(consumer_state_P)
                pipeline_dV.producer_acquire(producer_state_dV)
                mma_dv_dop_fn()
                pipeline_dV.producer_commit(producer_state_dV)
                producer_state_dV.advance()

                # Release sP to be empty
                pipeline_P.consumer_release(consumer_state_P)
                consumer_state_P.advance()

                # --- GEMM: dQ += dS @ K ---

                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dq_dsk_fn(B_idx=consumer_state_K.index, zero_init=n_block == 0)

                # Release sK to be empty
                pipeline_K.consumer_release(consumer_state_K)
                consumer_state_K.advance()

                # --- GEMM: dK.T = Q.T @ dS ---

                pipeline_dK.producer_acquire(producer_state_dK)
                mma_dk_qds_fn()
                pipeline_dK.producer_commit(producer_state_dK)
                producer_state_dK.advance()

                # Release sdS to be empty
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

            # Commit tdQ to be full; this commit tracks every MMA issued above,
            # so the compute warps may also reuse sP/sdS as sdQ once it lands.
            pipeline_dQ.producer_commit(producer_state_dQ)
            producer_state_dQ.advance()

            # Release sQ/sdO to be empty
            pipeline_QdO.consumer_release(consumer_state_QdO)
            consumer_state_QdO.advance()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def _make_t2r(
        self,
        thr_mma: cute.ThrMma,
        tAcc: cute.Tensor,
        mma_tile_mn: cute.Tile,
        tidx: Int32,
    ):
        """Split a (128, N) accumulator into T2R chunks of `t2r_ncol` columns.

        Returns the tiled copy, the per-thread tmem source ``(ATOM, 1, 1, CHUNK)``,
        and the matching (row, col) coordinates.
        """
        chunk_layout = cute.make_layout((mma_tile_mn[0], self.t2r_ncol))
        tAcc_i = cute.logical_divide(tAcc, chunk_layout)
        cAcc = thr_mma.partition_C(cute.make_identity_tensor(mma_tile_mn))
        cAcc_i = cute.logical_divide(cAcc, chunk_layout)
        # `tcgen05.ld.sync.aligned.32x32b.x32`: one row (lane) x 32 cols per thread
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.t2r_ncol)),
            self.acc_dtype,
        )
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tAcc_i[(None, None), 0])
        thr_t2r = tiled_t2r.get_slice(tidx)
        tAcc_t2r = thr_t2r.partition_S(tAcc_i[(None, None), None])
        cAcc_t2r = thr_t2r.partition_D(cAcc_i[(None, None), None])
        return tiled_t2r, thr_t2r, tAcc_t2r, cAcc_t2r

    @cute.jit
    def _make_r2s(
        self,
        thr_mma: cute.ThrMma,
        tiled_t2r: cute.TiledCopy,
        thr_t2r: cute.core.ThrCopy,
        sEpi: cute.Tensor,
        mma_tile_mn: cute.Tile,
        dtype: cutlass.Constexpr,
    ):
        """R2S copy of T2R chunks into a row-major swizzled smem tile."""
        chunk_layout = cute.make_layout((mma_tile_mn[0], self.t2r_ncol))
        tAccsEpi = thr_mma.partition_C(sEpi)
        tAccsEpi_i = cute.logical_divide(tAccsEpi, chunk_layout)
        # position-independent partition: the chunk offsets straddle swizzle atoms
        tAccsEpi_r2s = copy_utils.partition_D_position_independent(
            thr_t2r, tAccsEpi_i[(None, None), None]
        )
        smem_store_atom = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, dtype, self.acc_dtype, tiled_t2r
        )
        tiled_r2s = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r)
        return tiled_r2s, tAccsEpi_r2s

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sP_epi: cute.Tensor,
        sdS_epi: cute.Tensor,
        sdQ: cute.Tensor,
        mdQ: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        pipeline_LSE: pipeline.PipelineAsync,
        pipeline_S: pipeline.PipelineUmmaAsync,
        pipeline_dP: pipeline.PipelineUmmaAsync,
        pipeline_P: pipeline.PipelineAsyncUmma,
        pipeline_dS: pipeline.PipelineAsyncUmma,
        pipeline_dQ: pipeline.PipelineUmmaAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        mTopkLen: cute.Tensor,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
    ):
        # --- Set up thread info ---

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        )

        # --- Make T2R copies for S / dP / dQ and R2S copies for P / dS / dQ ---

        # tStS_t2r / tdPtdP_t2r: (T2R_ATOM=((32,32),1),1,1,CHUNK2)
        # tScS_t2r: (T2R_ATOM=(32,1),1,1,CHUNK2) of (row, col) coords
        tiled_t2r_S, thr_t2r_S, tStS_t2r, tScS_t2r = self._make_t2r(
            thr_mma_S, tStS, self.mma_tiler_qk[:2], tidx
        )
        _, _, tdPtdP_t2r, _ = self._make_t2r(
            thr_mma_S, tdPtdP, self.mma_tiler_qk[:2], tidx
        )
        tiled_t2r_dQ, thr_t2r_dQ, tdQtdQ_t2r, _ = self._make_t2r(
            thr_mma_dQ, tdQtdQ, self.mma_tiler_dsk[:2], tidx
        )
        tiled_r2s_PdS, tPsP_r2s = self._make_r2s(
            thr_mma_S,
            tiled_t2r_S,
            thr_t2r_S,
            sP_epi[None, None, 0],
            self.mma_tiler_qk[:2],
            self.ds_dtype,
        )
        _, tdSsdS_r2s = self._make_r2s(
            thr_mma_S,
            tiled_t2r_S,
            thr_t2r_S,
            sdS_epi[None, None, 0],
            self.mma_tiler_qk[:2],
            self.ds_dtype,
        )
        sdQ_2d = sdQ[None, None, 0]
        tiled_r2s_dQ, tdQsdQ_r2s = self._make_r2s(
            thr_mma_dQ,
            tiled_t2r_dQ,
            thr_t2r_dQ,
            sdQ_2d,
            self.mma_tiler_dsk[:2],
            self.dq_dtype,
        )
        num_chunks_S = cute.size(tScS_t2r, mode=[3])
        num_chunks_dQ = cute.size(tdQtdQ_t2r, mode=[3])

        # Every element a thread holds belongs to one Q row (its TMEM lane)
        row_idx = tScS_t2r[0][0]

        # --- Init consumer / producer pipeline states ---

        consumer_state_LSE = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.LSE_stage
        )
        consumer_state_S = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dP = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dQ = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        producer_state_P = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dS = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            topk_len = mTopkLen[m_block, head_idx // self.qhead_per_kvhead]
            n_block_max = self._num_n_blocks(topk_len)

            # Rows at or past total_q are masked to P = dS = 0
            row_valid = row_idx < seqlen_info.seqlen_q - m_block * self.tile_m

            # --- S2R LSE / dPsum of this thread's row ---

            pipeline_LSE.consumer_wait(consumer_state_LSE)
            row_lse = sLSE[row_idx]
            row_dpsum = sdPsum[row_idx]

            for n_block in cutlass.range(n_block_max, unroll=1):
                # Gathered columns at or past topk_len are masked to P = dS = 0
                col_limit = topk_len - n_block * self.tile_n

                # //////////////////////////////////////////////
                #  Softmax-fwd: rP = exp2(rS * scale_log2 - rLSE)
                #  and R2S copy rP to sP
                # //////////////////////////////////////////////

                # Wait for tS to be full, T2R copy tS to rS
                pipeline_S.consumer_wait(consumer_state_S)
                tSrS = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                for c in cutlass.range_constexpr(num_chunks_S):
                    cute.copy(tiled_t2r_S, tStS_t2r[None, 0, 0, c], tSrS[None, 0, 0, c])

                # Release tS to be empty
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_S.consumer_release(consumer_state_S)
                consumer_state_S.advance()

                # Apply mask and softmax-fwd (rS now holds rP in fp32)
                for i in cutlass.range(cute.size(tSrS), unroll_full=True):
                    p = cute.math.exp2(
                        tSrS[i] * softmax_scale_log2 - row_lse, fastmath=True
                    )
                    is_valid = row_valid and tScS_t2r[i][1] < col_limit
                    tSrS[i] = p if is_valid else Float32(0.0)

                # Acquire sP to be empty (previous dV GEMM done), R2S copy rP to sP
                pipeline_P.producer_acquire(producer_state_P)
                for c in cutlass.range_constexpr(num_chunks_S):
                    copy_utils.cvt_copy(
                        tiled_r2s_PdS, tSrS[None, 0, 0, c], tPsP_r2s[None, 0, 0, c]
                    )

                # Commit sP to be full
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_P.producer_commit(producer_state_P)
                producer_state_P.advance()

                # //////////////////////////////////////////////
                #  Softmax-bwd: rdS = rP * (rdP - rdPsum) * scale
                #  after T2R copy tdP to rdP, then R2S copy rdS to sdS
                # //////////////////////////////////////////////

                # Wait for tdP to be full, acquire sdS to be empty
                pipeline_dP.consumer_wait(consumer_state_dP)
                pipeline_dS.producer_acquire(producer_state_dS)
                for c in cutlass.range_constexpr(num_chunks_S):
                    tdPrdP = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, 0, c].shape, Float32
                    )
                    cute.copy(tiled_t2r_S, tdPtdP_t2r[None, 0, 0, c], tdPrdP)
                    tSrP_cur = tSrS[None, 0, 0, c]
                    for i in cutlass.range(cute.size(tdPrdP), unroll_full=True):
                        tdPrdP[i] = (
                            tSrP_cur[i] * (tdPrdP[i] - row_dpsum) * softmax_scale
                        )
                    copy_utils.cvt_copy(
                        tiled_r2s_PdS, tdPrdP, tdSsdS_r2s[None, 0, 0, c]
                    )

                # Release tdP to be empty
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dP.consumer_release(consumer_state_dP)
                consumer_state_dP.advance()

                # Commit sdS to be full
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dS.producer_commit(producer_state_dS)
                producer_state_dS.advance()

            # Release sLSE/sdPsum to be empty
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_LSE.consumer_release(consumer_state_LSE)
            consumer_state_LSE.advance()

            # //////////////////////////////////////////////
            #  dQ epilogue: T2R tdQ -> R2S sdQ -> TMA S2G
            # //////////////////////////////////////////////

            # Wait for tdQ to be full; it also implies every MMA reading sP/sdS
            # of this Q tile has retired, so their bytes can be reused as sdQ.
            pipeline_dQ.consumer_wait(consumer_state_dQ)
            for c in cutlass.range_constexpr(num_chunks_dQ):
                tdQrdQ = cute.make_rmem_tensor(tScS_t2r[None, 0, 0, 0].shape, Float32)
                if n_block_max > 0:
                    cute.copy(tiled_t2r_dQ, tdQtdQ_t2r[None, 0, 0, c], tdQrdQ)
                else:
                    # empty K range: dQ is zero and tdQ was never written
                    tdQrdQ.fill(0.0)
                copy_utils.cvt_copy(tiled_r2s_dQ, tdQrdQ, tdQsdQ_r2s[None, 0, 0, c])

            # Release tdQ to be empty
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_dQ.consumer_release(consumer_state_dQ)
            consumer_state_dQ.advance()

            # Make R2S stores visible to the TMA S2G copy
            cute.arch.fence_view_async_shared()
            self.dQ_epi_barrier.arrive_and_wait()

            if warp_idx == self.compute_warp_ids[0]:
                # Place the tile at row total_q - h0 (h0 = rows left in this range
                # from the tile start) with 4th-mode offset offset_q + len_q, so
                # the tile starts at row offset_q + m_block * tile_m and rows past
                # the range end are clipped by TMA.
                len_q = seqlen_info.seqlen_q
                h0 = len_q - m_block * self.tile_m
                blk = cute.domain_offset(
                    (mdQ.shape[0] - h0, 0, head_idx, seqlen_info.offset_q + len_q),
                    mdQ,
                )
                gdQ = cute.local_tile(blk[(None, None, 0, 0)], self.dQ_epi_tile, (0, 0))
                tdQsdQ_tma, tdQgdQ_tma = cpasync.tma_partition(
                    tma_atom_dQ,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sdQ_2d, 0, 2),
                    cute.group_modes(gdQ, 0, 2),
                )
                cute.copy(tma_atom_dQ, tdQsdQ_tma, tdQgdQ_tma)
                cute.arch.cp_async_bulk_commit_group()
                # Drain the TMA smem read before sP/sdS are rewritten
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            self.dQ_epi_barrier.arrive_and_wait()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

    @cute.jit
    def reduce_dKV(
        self,
        thr_mma: cute.ThrMma,
        tAcc: cute.Tensor,
        mAcc: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: cute.Tensor,
        pipeline_acc: pipeline.PipelineUmmaAsync,
        tile_scheduler: TileSchedulerProtocol,
    ):
        """T2R dV.T (or dK.T) per K tile and atomically add it into the gathered fp32 rows.

        Each thread owns one head-dim lane and ``tile_n`` K columns, so a warp's
        atomics at a fixed column hit 32 consecutive fp32 words of one K row.
        """
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * 4)
        lane = cute.arch.lane_idx()

        # tAcc_t2r: (T2R_ATOM=((32,32),1),1,1,CHUNK2)
        # tAcccAcc_t2r: (T2R_ATOM=(32,1),1,1,CHUNK2) of (hd, col) coords
        tiled_t2r, _, tAcc_t2r, tAcccAcc_t2r = self._make_t2r(
            thr_mma, tAcc, self.mma_tiler_dop[:2], tidx
        )
        num_chunks = cute.size(tAcccAcc_t2r, mode=[3])
        hd_idx = tAcccAcc_t2r[0][0]

        consumer_state = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, _, _ = work_tile.tile_idx
            head_idx_kv = head_idx // self.qhead_per_kvhead
            topk_len = mTopkLen[m_block, head_idx_kv]
            n_block_max = self._num_n_blocks(topk_len)

            # mAcc_cur: (seqK,HD) row-major fp32 over all K tokens
            mAcc_cur = mAcc[None, None, head_idx_kv]
            # mIdx_cur: (topk,) global K row ids of this Q tile
            mIdx_cur = mIndices[m_block, head_idx_kv, None]

            for n_block in cutlass.range(n_block_max, unroll=1):
                col_limit = topk_len - n_block * self.tile_n

                # Prefetch the K row ids of this tile, column c * 32 + l on lane l
                tIrRow = cute.make_rmem_tensor((num_chunks,), Int32)
                for c in cutlass.range_constexpr(num_chunks):
                    col = c * self.t2r_ncol + lane
                    row_idx = Int32(0)
                    if col < col_limit:
                        row_idx = mIdx_cur[n_block * self.tile_n + col]
                    tIrRow[c] = row_idx

                # Wait for tAcc to be full, T2R copy it to registers
                pipeline_acc.consumer_wait(consumer_state)
                tAccrAcc = cute.make_rmem_tensor(tAcccAcc_t2r.shape, Float32)
                for c in cutlass.range_constexpr(num_chunks):
                    cute.copy(
                        tiled_t2r, tAcc_t2r[None, 0, 0, c], tAccrAcc[None, 0, 0, c]
                    )

                # Release tAcc to be empty before the atomics
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_acc.consumer_release(consumer_state)
                consumer_state.advance()

                # Atomic add the valid columns into their gathered K rows
                for c in cutlass.range_constexpr(num_chunks):
                    tAccrAcc_c = tAccrAcc[None, 0, 0, c]
                    tAcccAcc_c = tAcccAcc_t2r[None, 0, 0, c]
                    for i in cutlass.range(cute.size(tAccrAcc_c), unroll_full=True):
                        col = tAcccAcc_c[i][1]
                        # shuffle_sync needs every lane: keep it outside the predicate
                        row_idx = cute.arch.shuffle_sync(
                            tIrRow[c], col - c * self.t2r_ncol
                        )
                        if col < col_limit:
                            cutedsl_utils.atomic_add_fp32(
                                tAccrAcc_c[i],
                                cutedsl_utils.elem_pointer(mAcc_cur, (row_idx, hd_idx)),
                            )

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()


def ffa_bwd_sm100_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    tile_m: int = 128,
    tile_n: int = 64,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass of index-sparse (DSA) attention on SM100.

    Args:
        index_sparse_indices: int32 ``(ceil(total_q / tile_m), num_heads_kv, topk)``.
            Row ``[b, h_kv]`` holds the distinct global K token ids attended by
            q tokens ``[b * tile_m, (b + 1) * tile_m)`` of every q head mapped to
            kv head ``h_kv``; valid ids come first and the tail is padded with -1.

    Returns:
        A tuple of (dQ, dK, dV) with the same shapes and dtypes as q, k, v.
    """
    head_dim = FFABwdSm100Index.head_dim
    assert (
        q.shape[-1] == k.shape[-1] == v.shape[-1] == head_dim
    ), f"only head_dim == {head_dim} is supported"
    assert q.ndim == k.ndim == v.ndim == 3
    assert out.shape == dout.shape == q.shape
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype
    assert all(
        t.device == q.device for t in (k, v, out, dout, lse, index_sparse_indices)
    )
    assert lse.shape == (q.shape[0], q.shape[1]) and lse.dtype == torch.float32
    assert lse.stride(-1) == 1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    q, k, v, out, dout, lse, index_sparse_indices = [
        maybe_contiguous(t) for t in (q, k, v, out, dout, lse, index_sparse_indices)
    ]
    total_q, num_head, _ = q.shape
    total_k, num_head_kv, _ = v.shape
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    qhead_per_kvhead = num_head // num_head_kv
    num_q_blocks = (total_q + tile_m - 1) // tile_m
    assert index_sparse_indices.dtype == torch.int32
    assert index_sparse_indices.ndim == 3 and index_sparse_indices.shape[:2] == (
        num_q_blocks,
        num_head_kv,
    ), (
        f"index_sparse_indices must be (num_q_blocks={num_q_blocks}, "
        f"num_heads_kv={num_head_kv}, topk), got {tuple(index_sparse_indices.shape)}"
    )
    device = q.device
    dtype = to_cute_dtype(q.dtype)
    kv_stage = 3

    # Valid ids precede the -1 padding, so the count is the per-row inner length
    topk_len = (index_sparse_indices >= 0).sum(dim=-1, dtype=torch.int32)

    # One range over every token: the scheduler, dPsum preprocess, dQ epilogue
    # and dK/dV postprocess all run their full-mask single-batch paths.
    q_ranges = torch.tensor([[0, total_q]], dtype=torch.int32, device=device)
    k_ranges = torch.tensor([[0, total_k]], dtype=torch.int32, device=device)

    # --- Preprocess: dPsum = (o * dout).sum(-1), lse_log2 = lse * log2(e) ---

    # Row-major token-space stats + one tile guard for the tail tile's TMA box.
    total_q_rounded_padded = (num_q_blocks + 1) * tile_m
    # Zeros, not empty: the tail tile reads the padding rows.
    dpsum = torch.zeros(
        num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )
    lse_log2 = torch.zeros(
        num_head, total_q_rounded_padded, dtype=torch.float32, device=device
    )
    bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        None,  # dq_accum: dQ is stored directly
        None,  # cu_seqlens_q
        None,  # seqused_q
        None,  # dlse
        dtype,
        head_dim,
        head_dim,
        tile_m,
        use_padded_offsets=False,
        q_ranges=q_ranges,
        max_seqlen_q=total_q,
    )

    # --- Allocate outputs / accumulators ---

    # Empty: every (Q tile, q head) stores its dQ rows, and the postprocess
    # writes every dK/dV row (untouched rows come out as zero).
    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    total_k_rounded_padded = ((total_k + tile_n - 1) // tile_n + 1) * tile_n
    dk_accum = torch.zeros(
        num_head_kv,
        total_k_rounded_padded * head_dim,
        dtype=torch.float32,
        device=device,
    )
    dv_accum = torch.zeros(
        num_head_kv,
        total_k_rounded_padded * head_dim,
        dtype=torch.float32,
        device=device,
    )

    # --- Main kernel ---

    compile_key = (
        dtype,
        qhead_per_kvhead,
        tile_m,
        tile_n,
        kv_stage,
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
    )
    compile_cache = ffa_bwd_sm100_index.compile_cache  # type: ignore[attr-defined]
    if compile_key not in compile_cache:
        ffa_bwd_obj = FFABwdSm100Index(
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=tile_m,
            tile_n=tile_n,
            kv_stage=kv_stage,
            debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
        )
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq)
        ]
        lse_log2_tensor, dpsum_tensor, dk_accum_tensor, dv_accum_tensor = [
            to_cute_tensor(t) for t in (lse_log2, dpsum, dk_accum, dv_accum)
        ]
        q_ranges_tensor, indices_tensor, topk_len_tensor = [
            to_cute_tensor(t, assumed_align=4)
            for t in (q_ranges, index_sparse_indices, topk_len)
        ]
        compile_cache[compile_key] = cute.compile(
            ffa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_tensor,
            dk_accum_tensor,
            dv_accum_tensor,
            softmax_scale,
            q_ranges_tensor,
            indices_tensor,
            topk_len_tensor,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq,
        dk_accum,
        dv_accum,
        softmax_scale,
        q_ranges,
        index_sparse_indices,
        topk_len,
    )

    # --- Postprocess: fp32 dK/dV accumulators -> dk/dv ---

    # softmax_scale is already folded into dS, so dK needs no extra scale.
    bwd_postprocess_rowmajor(dk_accum, dk, k_ranges, total_k, 1.0)
    bwd_postprocess_rowmajor(dv_accum, dv, k_ranges, total_k, 1.0)

    return dq, dk, dv


ffa_bwd_sm100_index.compile_cache = get_jit_cache("bwd_sm100_index")  # type: ignore[attr-defined]
