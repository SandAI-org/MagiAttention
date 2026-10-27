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

"""SM100 forward for index-sparse (DSA) attention, ``ffa_fwd_sm100_index``.

Given ``index_sparse_indices`` with shape
``(ceil(total_q / q_block_size), num_heads_kv, topk)`` of global K token ids (-1 padded),
computes token-major ``out`` and natural-log ``lse``.

Kernel structure (single CTA, 16 warps):

- A 128-row Q tile is ``q_block_size`` query tokens x all ``G`` query heads of one
  KV head (PackGQA row order, head fastest), so one tile shares one index row.
  The kernel therefore requires ``q_block_size * G == 128``.
- Three load warps gather the K/V token rows of every 128-slot tile with 16 B
  ``cp.async`` copies (two 256 B rows per warp instruction; shuffle of the table
  row, 64-bit address, validity compare, copy with an immediate destination
  offset), compute the SW128 swizzle in software and arrive on the stage barrier
  themselves after ``cp.async.wait_group`` and a proxy fence.  Invalid slots
  (``-1``, out-of-range id, ``slot >= topk_len``) are zero-filled and masked to
  ``-inf`` by the softmax warps from validity words read off the index row.
- Two softmax warpgroups alternate over consecutive K tiles (``q_stage = 1``
  ping-pong in the N direction), S and P share TMEM, O is accumulated in two
  TMEM stages and combined at the end of the work item by the correction warps,
  which also issue the O TMA store.

Supported: BF16 / FP16, head_dim 128, ``G`` dividing 128, non-causal, SM 10.x.
"""

import math
from functools import partial
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint32, const_expr, pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum, SmemAllocator, TmemAllocator

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
from .ffa_utils import maybe_contiguous
from .mask import mask_r2p_lambda
from .named_barrier import NamedBarrierFwdSm100
from .pack_gqa import pack_gqa_layout
from .softmax import SoftmaxSm100
from .tile_scheduler import (
    SchedulingMode,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)

# Beyond this |row_max * scale_log2| the packed FFMA x * s - max * s loses the exactness at the
# max; the warp then subtracts the max first (see ``scale_subtract_rowmax_guarded``).
FFMA_MAX_SCALED_ABS = 2048.0


# =============================================================================
# Low-level bridges: zero-filling cp.async, opaque mov, barrier subsets
# =============================================================================
@dsl_user_op
def cp_async_16_zfill(
    dst_smem_addr: cutlass.Int32,
    src_gmem_addr: cutlass.Int64,
    row_id: cutlass.Int32,
    dst_imm_offset: int = 0,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """Copy 16 B from global to shared memory at ``dst_smem_addr + dst_imm_offset``.

    Writes 16 zero bytes instead when ``row_id < 0``.

    The sign of ``row_id`` drives the instruction's ``ignore-src`` predicate (one ``setp`` inside the
    asm, no branch).  An ignored source is not read, so ``src_gmem_addr`` need not be mapped.
    ``dst_imm_offset`` must be a Python int (it is emitted as the instruction's immediate).
    Completion is tracked by the calling thread's ``cp.async`` groups. The unhinted
    instruction avoids an uninitialized uniform-register policy operand in the
    nvidia-cutlass-dsl 4.4.x lowering of ``L2::cache_hint`` on sm_103a.
    """
    assert isinstance(dst_imm_offset, int) and 0 <= dst_imm_offset < (
        1 << 24
    ), "immediate offset out of range"
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(dst_smem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(src_gmem_addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row_id).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $2, 0;\n\t"
        f"cp.async.cg.shared.global [$0+{dst_imm_offset}], [$1], 16, p;\n}}",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def opaque_u32(
    value: cutlass.Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> cutlass.Uint32:
    """Return ``value`` through a ``mov.b32`` the optimizer cannot see through.

    Used once per kernel on the gather's row pitch so the 64-bit product with the row id is a single
    ``mul.wide.u32`` (``IMAD.WIDE.U32`` with the base as addend) instead of a 64 x 64 multiply.
    """
    res = llvm.inline_asm(
        T.i32(),
        [cutlass.Uint32(value).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(res)


class _SubsetNamedBarrier(ffa_pipeline.NamedBarrier):
    """CTA named barrier whose participants occupy only selected warp roles (non-aligned flavour)."""

    @dsl_user_op
    def arrive(self, *, loc=None, ip=None):
        self.arrive_w_index(Int32(0), loc=loc, ip=ip)

    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None):
        self.arrive_and_wait_w_index(Int32(0), loc=loc, ip=ip)

    @dsl_user_op
    def arrive_w_index(self, index: Int32, *, loc=None, ip=None):
        nvvm.barrier_cta_arrive(
            Int32(self.barrier_id + index).ir_value(loc=loc, ip=ip),
            Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_and_wait_w_index(self, index: Int32, *, loc=None, ip=None):
        nvvm.barrier_cta_sync(
            Int32(self.barrier_id + index).ir_value(loc=loc, ip=ip),
            thread_count=Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )


@cute.jit
def scale_subtract_rowmax_guarded(
    softmax: SoftmaxSm100, acc_S_row: cute.Tensor, row_max: Float32
) -> None:
    """``x * scale_log2 - row_max * scale_log2`` as one packed FFMA per pair, exact at the max.

    Beyond ``FFMA_MAX_SCALED_ABS`` the fused form can round ``x == row_max`` away from 0; the warp
    then subtracts the max first (``vote_any_sync`` keeps the branch warp-uniform).
    """
    assert (
        cute.size(acc_S_row.shape) % 2 == 0
    ), "acc_S_row must have an even number of elements"
    scale_log2 = softmax.scale_log2
    row_max_scaled = row_max * scale_log2
    neg_row_max_scaled = -row_max_scaled
    if cute.arch.vote_any_sync(
        cutlass.max(row_max_scaled, -row_max_scaled) >= FFMA_MAX_SCALED_ABS
    ):
        neg_row_max_scaled = Float32(0.0)
        neg_row_max = -row_max
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.add_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]), (neg_row_max, neg_row_max)
            )
    for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
        acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
            (acc_S_row[i], acc_S_row[i + 1]),
            (scale_log2, scale_log2),
            (neg_row_max_scaled, neg_row_max_scaled),
        )


# =============================================================================
# Public Kernel Class
# =============================================================================
class FFAFwdSm100Index:
    head_dim = 128
    tile_m = 128
    tile_n = 128
    is_persistent_default = True
    # 64-element (128 B) chunks of a 16-bit row: the unit of the SW128 K-major smem tile.
    gather_chunk_elems = 64
    # Three warps issue the K/V cp.async gathers (warps 13-15); the epilogue TMA store is
    # folded into the correction warps, and the persistent scheduler runs in static mode.
    gather_warps = 3
    # Keep at most one cp.async group in flight per gather thread before arriving on the
    # oldest stage barrier so the MMA warp receives completed K/V stages without extra slack.
    cp_async_depth = 1

    def __init__(
        self,
        head_dim: int = 128,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        debug_print: bool = False,
    ):
        assert (
            head_dim == self.head_dim
        ), f"the SM100 index-sparse forward supports head_dim {self.head_dim} only"
        self.head_dim_padded = head_dim
        self.head_dim_v_padded = head_dim
        self.m_block_size = self.tile_m
        self.n_block_size = self.tile_n
        self.q_stage = 1
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to begin the P @ V MMA, then write the rest of P and signal again. This allows overlap
        # between computing the last columns of P and the P @ V MMA.
        self.split_P_arrive = (self.n_block_size // 4 * 3) // 32 * 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert (
            self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f
        ), "Only SM 10.x and 11.x are supported"

        self.cta_tiler = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.m_block_size,
            self.head_dim_v_padded,
            self.n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.is_persistent = self.is_persistent_default
        # All warp roles are occupied; static scheduling needs no dedicated scheduler warp.
        self.scheduling_mode = SchedulingMode.STATIC
        self.qhead_per_kvhead = qhead_per_kvhead
        assert (
            self.m_block_size % self.qhead_per_kvhead == 0
        ), "qhead_per_kvhead must divide the 128-row tile"
        # A 128-row tile is q_block_size query tokens x all query heads of one KV head (PackGQA row
        # order, head fastest), so tile m shares indices row m, which covers query tokens
        # [m * q_block_size, (m + 1) * q_block_size).
        self.pack_gqa = qhead_per_kvhead > 1
        self.q_block_size = self.m_block_size // self.qhead_per_kvhead
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        self.enable_ex2_emu = not is_sm103

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        # Gather warps occupy warps 13-15 of warpgroup 3 (setmaxnreg is warpgroup-uniform),
        # and the correction warps issue the O TMA store.
        self.load_warp_ids = tuple(range(16 - self.gather_warps, 16))
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
            )
        )

        self.s_stage = 2  # Always 2: for q_stage=1 it's n-direction
        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.s_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 64, 192

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.enable_ex2_emu:
            self.num_regs_softmax = 184
            self.num_regs_correction = 88
            self.num_regs_other = 56
        else:
            # Give warpgroup 3 (MMA + 3 gather warps) 96 registers: the cp.async gather loop's
            # per-row shuffle, 64-bit address and validity compare spill to local memory at 72,
            # and those spill loads are the kernel's dominant long-scoreboard stall.
            self.num_regs_softmax = 168
            self.num_regs_correction = 72
            self.num_regs_other = 96
        assert (
            2 * self.num_regs_softmax + self.num_regs_correction + self.num_regs_other
        ) * 128 <= 64 * 1024

        self.buffer_align_bytes = 1024
        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[fwd_sm100_index_init] "
            print()
            print(f"{prefix}Initialized FFAFwdSm100Index with: ")
            print(
                f"{prefix}{self.head_dim=} | {self.qhead_per_kvhead=} | {self.q_block_size=}"
            )
            print(
                f"{prefix}{self.m_block_size=} | {self.n_block_size=} | {self.s_stage=}"
            )
            print(
                f"{prefix}{self.tmem_s_offset=} | {self.tmem_p_offset=} | {self.tmem_o_offset=}"
            )
            print(
                f"{prefix}{self.num_regs_softmax=} | {self.num_regs_correction=} | {self.num_regs_other=}"
            )
            print()

    def _setup_attributes(self):
        """Set up pipeline stage counts from the shared-memory footprint."""
        smem_size_q = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_padded
            * self.q_dtype.width
            // 8
        )
        smem_size_o = (
            self.q_stage
            * self.m_block_size
            * self.head_dim_v_padded
            * self.o_dtype.width
            // 8
        )
        smem_size_q_o = smem_size_q + smem_size_o
        smem_size_kv_per_stage = (
            self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        )
        self.kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        assert self.s_stage >= self.q_stage
        depth = self.cp_async_depth
        assert (
            1 <= depth <= self.kv_stage - 2
        ), "cp_async_depth must leave the MMA warp at least one stage"
        self.cp_async_depth_eff = depth

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (s_q, h_q, d)
        mK: cute.Tensor,  # (s_k, h_kv, d), contiguous
        mV: cute.Tensor,  # (s_k, h_kv, d), contiguous
        mO: cute.Tensor,  # (s_q, h_q, d)
        mLSE: Optional[cute.Tensor],  # (s_q, h_q), fp32, natural log
        softmax_scale: Float32,
        mIndices: cute.Tensor,  # (num_q_blocks, h_kv, topk) int32, -1 padded
        mTopkLen: Optional[cute.Tensor],  # (num_q_blocks, h_kv) int32 or None
        nheads_kv: Int32,
        stream: cuda.CUstream,
    ):
        """Launch the index-sparse attention forward.

        Q/K/V/O use token-major 3-D layouts; LSE uses ``(S_q, H_q)``.
        PackGQA folds the query heads of one KV head into the row mode. K/V are
        viewed as ``(S_k * H_kv, D)`` tables for cp.async gathers. ``mIndices``
        contains global K token ids; optional ``mTopkLen`` bounds each index row.
        """
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [
            cutedsl_utils.assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)
        ]
        seqlen_q, seqlen_k = mQ.shape[0], mK.shape[0]
        mQ, mO = [layout_utils.select(t, mode=[0, 2, 1]) for t in (mQ, mO)]
        mQ, mO = [
            cute.make_tensor(t.iterator, cute.append(t.layout, cute.make_layout(1)))
            for t in (mQ, mO)
        ]
        if const_expr(mLSE is not None):
            assert mLSE is not None
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.append(mLSE.layout, cute.make_layout(1))
            )
        mK, mV = [
            cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (t.shape[0] * t.shape[1], t.shape[2]),
                    stride=(t.stride[1], t.stride[2]),
                ),
            )
            for t in (mK, mV)
        ]
        num_splits = Int32(1)

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        # Tuned defaults for the polynomial ex2 emulation path.
        self.ex2_emu_freq = 10 if const_expr(self.enable_ex2_emu) else 0
        self.ex2_emu_start_frg = 0

        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        assert self.q_dtype == self.k_dtype == self.v_dtype, "Q/K/V must share a dtype"
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )
        # The gather writes rows at hand-computed smem offsets, so pin the operand layouts it
        # assumes: K-major SW128 (K) and MN-major SW128 (V) tiles of 128 x 128 16-bit elements are
        # the same bytes -- 64-element chunk c of token row r lives at c * 8192 + r * 64 elements.
        self.kv_stage_elems = cute.cosize(sK_layout) // self.kv_stage
        self.chunk_stride_elems = self.n_block_size * self.gather_chunk_elems
        assert cute.cosize(sV_layout) == cute.cosize(sK_layout)
        assert self.kv_stage_elems == self.n_block_size * self.head_dim_padded
        assert self.head_dim_padded % self.gather_chunk_elems == 0

        if const_expr(self.pack_gqa):
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(
                self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2])
            ),
        }
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        # TMA load for Q and TMA store for O
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
        )
        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
        )

        TileScheduler = (
            StaticPersistentTileScheduler
            if const_expr(self.is_persistent)
            else SingleTileScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            seqlen_k,
            mQ.shape[1],
            self.head_dim_v_padded,
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_stage * 2]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, cute.cosize(sO_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = cutedsl_utils.compute_softmax_scale_log2(
            softmax_scale, None
        )

        if const_expr(self.debug_print):
            prefix = "[fwd_sm100_index_call] "
            print()
            print(f"{prefix}tiled_mma_qk: {tiled_mma_qk}")
            print(f"{prefix}tiled_mma_pv: {tiled_mma_pv}")
            print(f"{prefix}kv_stage: {self.kv_stage}")
            print(f"{prefix}sQ_layout: {sQ_layout}")
            print(f"{prefix}sK_layout: {sK_layout}")
            print(f"{prefix}sV_layout: {sV_layout}")
            print(f"{prefix}tP_layout: {tP_layout}")
            print(f"{prefix}sO_layout: {sO_layout}")
            print(f"{prefix}threads_per_cta: {self.threads_per_cta}")
            print(f"{prefix}smem bytes: {self.shared_storage.size_in_bytes()}")  # type: ignore[attr-defined]
            print()

        # Enqueue the kernel on the caller's stream.
        self.kernel(
            mQ,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            mIndices,
            mTopkLen,
            seqlen_q,
            seqlen_k,
            nheads_kv,
            mK,
            mV,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        seqlen_q: Int32,
        seqlen_k: Int32,
        nheads_kv: Int32,
        mK: cute.Tensor,  # (s_k * h_kv, d) row table, read directly by the cp.async gather
        mV: cute.Tensor,  # (s_k * h_kv, dv)
    ):
        """Device kernel coordinating the warp-specialized forward pipeline:
        1. Load warps: load Q with TMA and gather indexed K/V token rows with cp.async
        2. MMA warp: issue Q @ K^T and P @ V UMMA operations
        3. Softmax warpgroups: compute online softmax and mask invalid index slots
        4. Correction warps: rescale partial O accumulators, combine O0/O1, and store O/LSE
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_O):
                cpasync.prefetch_descriptor(tma_atom)

        # Alloc
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        pipeline_q = ffa_pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        # Every gather thread arrives on the full barrier itself once its cp.async group for the
        # stage has completed and been fenced into the async proxy; the MMA warp releases with
        # tcgen05.commit after the stage's MMA reads.
        gather_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.load_warp_ids)
        )
        pipeline_kv = ffa_pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=gather_threads,
            consumer_group=mma_warp,
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = ffa_pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads,
            defer_sync=True,
        )
        pipeline_p_lastsplit = ffa_pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_warps,
            consumer_group=mma_warp,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = ffa_pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_sm_stats = ffa_pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # NamedBarrier for per-warp softmax-to-correction stats signaling
        sm_stats_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )

        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # Raw K/V stage base for the cp.async gather (resolved outside the warp-role branches:
        # struct field access is a Python-object expression the DSL cannot trace inside a dynamic `if`).
        sKV_base_ptr = storage.sK.data_ptr()
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.s_stage * self.m_block_size * 2)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # Always request all 512 columns of tmem, so allocation starts at column 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.s_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)

        pipeline_init_wait()

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                mQ,
                sQ,
                sKV_base_ptr,
                tma_atom_Q,
                pipeline_q,
                pipeline_kv,
                tile_scheduler,
                mIndices,
                mTopkLen,
                seqlen_k,
                nheads_kv,
                mK,
                mV,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                tile_scheduler,
                mIndices,
                mTopkLen,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.softmax1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                tile_scheduler=tile_scheduler,
                mIndices=mIndices,
                mTopkLen=mTopkLen,
                seqlen_k=seqlen_k,
            )

            stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
            softmax_loop(stage=stage, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                softmax_scale_log2,
                seqlen_q,
                tile_scheduler,
                mIndices,
                mTopkLen,
                mO,
                tma_atom_O,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        sKV_base_ptr: cute.Pointer,
        tma_atom_Q: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        tile_scheduler: "TileSchedulerProtocol",
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        seqlen_k: Int32,
        nheads_kv: Int32,
        mK: cute.Tensor,
        mV: cute.Tensor,
    ):
        """Load warps: TMA for Q, cp.async gather for the indexed K/V token rows (one tile per stage)."""
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        lane_idx = tidx % cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_rank = warp_idx - self.load_warp_ids[0]
        sKV_base_addr = Int32(sKV_base_ptr.toint())
        two_w, lane_base, src_base_K, row_bytes, dst_rel = self.cp_async_lane_constants(
            mK, lane_idx, warp_rank
        )
        _, _, src_base_V, _, _ = self.cp_async_lane_constants(mV, lane_idx, warp_rank)
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 1)
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            # With PackGQA, head_idx is the KV head; without it qhead_per_kvhead == 1 and the two agree.
            topk_eff, block_iter_count, process_tile = self.tile_counts(
                mIndices, mTopkLen, m_block, head_idx
            )
            gather = partial(
                self.gather_KV_cp_async,
                sKV_base_addr=sKV_base_addr,
                mIndices=mIndices,
                m_block=m_block,
                head_idx=head_idx,
                lane_idx=lane_idx,
                two_w=two_w,
                lane_base=lane_base,
                row_bytes=row_bytes,
                dst_rel=dst_rel,
                pipeline_kv=pipeline_kv,
            )
            load_K = partial(gather, mK, src_base=src_base_K)
            load_V = partial(gather, mV, src_base=src_base_V, prefetch_tile=None)
            resolve = partial(
                self.resolve_row_ids,
                topk_eff=topk_eff,
                seqlen_k=seqlen_k,
                nheads_kv=nheads_kv,
                head_idx=head_idx,
                lane_idx=lane_idx,
            )

            if process_tile:
                # Row ids are resolved once per tile at its K load and kept for its V load two loads
                # later: `b` holds the newest pending tile, `a` the one before it, `n0..n3` the raw token
                # ids prefetched for the next K load.
                n0, n1, n2, n3 = self.load_tile_toks(
                    mIndices, m_block, head_idx, block_iter_count - 1, lane_idx
                )
                b0, b1, b2, b3 = resolve((n0, n1, n2, n3), block_iter_count - 1)
                n0, n1, n2, n3 = load_K(
                    row_ids=(b0, b1, b2, b3),
                    prefetch_tile=block_iter_count - 2,
                    producer_state=kv_producer_state,
                )  # K0
                a0, a1, a2, a3 = b0, b1, b2, b3
                if (
                    const_expr(len(self.load_warp_ids) == 1)
                    or warp_idx == self.load_warp_ids[0]
                ):
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                kv_producer_state.advance()

                # q_stage=1 intra-warp overlap
                # Load order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2
                q_producer_phase ^= 1

                # Prologue: K[N-2]; prefetch K[N-3]'s ids (tile 0 if none).
                b0, b1, b2, b3 = resolve((n0, n1, n2, n3), block_iter_count - 2)
                n0, n1, n2, n3 = load_K(
                    row_ids=(b0, b1, b2, b3),
                    prefetch_tile=cutlass.max(block_iter_count - 3, Int32(0)),
                    producer_state=kv_producer_state,
                )
                kv_producer_state.advance()
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: the tile resolved two loads ago
                    load_V(row_ids=(a0, a1, a2, a3), producer_state=kv_producer_state)
                    kv_producer_state.advance()
                    # K[N-3-i]; prefetch K[N-4-i]'s ids
                    a0, a1, a2, a3 = b0, b1, b2, b3
                    b0, b1, b2, b3 = resolve((n0, n1, n2, n3), block_iter_count - 3 - i)
                    n0, n1, n2, n3 = load_K(
                        row_ids=(b0, b1, b2, b3),
                        prefetch_tile=cutlass.max(block_iter_count - 4 - i, Int32(0)),
                        producer_state=kv_producer_state,
                    )
                    kv_producer_state.advance()
                # Epilogue: V[1] then V[0]
                load_V(row_ids=(a0, a1, a2, a3), producer_state=kv_producer_state)
                kv_producer_state.advance()
                load_V(row_ids=(b0, b1, b2, b3), producer_state=kv_producer_state)
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            work_tile = tile_scheduler.advance_to_next_work()
            # End of persistent scheduler loop

        # Every gather thread owns cp.async groups and full-barrier arrives: drain them, then all
        # threads wait for the MMA warp's last releases.
        self.cp_async_drain(pipeline_kv, kv_producer_state)
        pipeline_kv.producer_tail(kv_producer_state)
        if (
            const_expr(len(self.load_warp_ids) == 1)
            or warp_idx == self.load_warp_ids[0]
        ):
            # This is equivalent to pipeline_q.producer_tail
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        q_smem_base = sm100_utils.make_smem_desc_base(
            cute.recast_layout(128, sQ.element_type.width, sQ.layout[0]),
            sQ.iterator.type.swizzle_type,
            sm100_utils.Major.K,
        )
        k_smem_base = sm100_utils.make_smem_desc_base(
            cute.recast_layout(128, sK.element_type.width, sK.layout[0]),
            sK.iterator.type.swizzle_type,
            sm100_utils.Major.K,
        )
        q_smem_start = [
            sm100_utils.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator)
            for stage in range(self.q_stage)
        ]

        sm100_utils.declare_ptx_smem_desc(
            q_smem_start[self.q_stage - 1],
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="fwd_index_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fwd_index_qk_mma_idesc")
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="fwd_index_q_smem_desc",
                idesc_var_name="fwd_index_qk_mma_idesc",
                smem_offset=0,
                zero_init=True,
            )
            for stage in range(self.s_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
            )
            for stage in range(self.s_stage)
        ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        # Pipeline s_p_o phases for stage 0 and stage 1.
        # These phases must persist across tiles
        # so that mbarrier phase stays in sync when block_iter_count varies.
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            topk_eff, block_iter_count, process_tile = self.tile_counts(
                mIndices, mTopkLen, m_block, head_idx
            )

            if process_tile:
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # GEMM order: S0=Q@K0, S1=Q@K1, {O0+=P0@V0, S0=Q@K2}, {O1+=P1@V1, S1=Q@K3}, ...
                # ================================================================

                # Prologue: wait Q0
                pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                gemm_Si[0](
                    smem_desc_start_b=sm100_utils.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                gemm_Si[1](
                    smem_desc_start_b=sm100_utils.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(1)  # signal S1 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # Flat loop: N-2 iterations, alternating stage 0/1
                block_loop_count = block_iter_count - 2
                O_acc_s0 = False  # per-stage accumulate flags
                O_acc_s1 = False
                # Pre-declare loop variables for DSL type tracking
                Vi_index = mma_kv_consumer_state.index
                Ki_index = mma_kv_consumer_state.index
                mma_kv_release_state = mma_kv_consumer_state.clone()
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                sK_cur = sK[None, None, None, Ki_index]

                pair_count = (
                    block_loop_count // 2
                )  # N even => block_loop_count = N-2 is even
                for i in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(stage == 0):
                            phase_cur, O_acc_cur = phase_s0, O_acc_s0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                        # Wait V
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index = mma_kv_consumer_state.index
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        mma_kv_consumer_state.advance()
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, phase_cur)
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_acc_cur,
                            mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                                stage
                            )
                            if self.split_P_arrive > 0
                            else None,
                            mbar_phase=phase_cur,
                        )
                        # Overlap the independent K wait with the issued PV MMA.
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index = mma_kv_consumer_state.index
                        sK_cur = sK[None, None, None, Ki_index]
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_utils.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        # Release V and K
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index = mma_kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    pipeline_s_p_o.producer_acquire_w_index_phase(epi_s, epi_p)
                    gemm_Pi[epi_stage_constexpr](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=epi_zi,
                        mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(
                            epi_s
                        )
                        if self.split_P_arrive > 0
                        else None,
                        mbar_phase=epi_p,
                    )
                    pipeline_o_acc.producer_commit_w_index(epi_s)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                # Epilogue did one acquire per stage; advance phases for next tile
                phase_s0 ^= 1
                phase_s1 ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        tile_scheduler: "TileSchedulerProtocol",
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        seqlen_k: Int32,
    ):
        """Process alternating K tiles and publish each row's softmax statistics to correction."""
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids))
        )
        lane_idx = tidx % cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            topk_eff, block_iter_count, has_work = self.tile_counts(
                mIndices, mTopkLen, m_block, head_idx
            )

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
            )

            # Always acquire pipeline_sm_stats to stay in sync with correction
            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            if has_work:
                # block_iter_count is even: each WG processes exactly half the tiles
                # WG0 (stage=0): logical tiles N-1, N-3, ... (stride 2)
                # WG1 (stage=1): logical tiles N-2, N-4, ... (stride 2)
                # Every slot >= topk_eff (so the whole phantom tile that even rounding adds), every
                # -1 and every out-of-range id is masked to -inf (the gather zero-filled those rows).
                wg_count = block_iter_count // 2
                logical_first = block_iter_count - 1 - stage

                # Tiles are processed in descending logical order, so the tiles that can contain an
                # invalid slot (the padded tail, slots >= topk_len, the phantom tile, or anything from the
                # first invalid id on) come first.  They take the masked softmax step; the remaining tiles
                # take the mask-free step in a loop whose body has no branch, which the compiler needs to
                # schedule the step well.
                t_mask = self.first_masked_tile(
                    mIndices, m_block, head_idx, topk_eff, seqlen_k, lane_idx
                )
                # This WG's tiles are logical_first - 2 k; the masked ones are those >= t_mask.
                m_wg = cutlass.min(
                    cutlass.max(
                        (logical_first - t_mask) // self.s_stage + Int32(1), Int32(0)
                    ),
                    wg_count,
                )
                if m_wg > Int32(0):
                    w0, w1, w2, w3 = self.tile_valid_mask(
                        mIndices,
                        m_block,
                        head_idx,
                        logical_first,
                        topk_eff,
                        seqlen_k,
                        lane_idx,
                    )
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        mask_fn=partial(apply_valid_mask, w0=w0, w1=w1, w2=w2, w3=w3),
                        is_first=True,
                    )
                    for n_tile in cutlass.range(m_wg - 1, unroll=1):
                        logical_n = logical_first - self.s_stage * (n_tile + 1)
                        v0, v1, v2, v3 = self.tile_valid_mask(
                            mIndices,
                            m_block,
                            head_idx,
                            logical_n,
                            topk_eff,
                            seqlen_k,
                            lane_idx,
                        )
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            mask_fn=partial(
                                apply_valid_mask, w0=v0, w1=v1, w2=v2, w3=v3
                            ),
                        )
                    for n_tile in cutlass.range(wg_count - m_wg, unroll=1):
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase, sm_stats_producer_phase
                        )
                else:
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase, sm_stats_producer_phase, is_first=True
                    )
                    for n_tile in cutlass.range(wg_count - 1, unroll=1):
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase, sm_stats_producer_phase
                        )

                sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                sScale[
                    tidx + stage * self.m_block_size + self.s_stage * self.m_block_size
                ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            else:
                # Empty tile: arrive barrier once (synthetic "no work" signal for correction)
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: "SoftmaxSm100",
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32]:
        """Mask one score tile, update online statistics, and publish unnormalized P in two splits."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r)  # type: ignore[misc]
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        # One packed FFMA per pair (x * scale_log2 - row_max * scale_log2); the warp-uniform guard
        # falls back to subtract-then-scale beyond |row_max * scale_log2| >= 2048.
        scale_subtract_rowmax_guarded(softmax, tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        # Accumulate the row sum before waiting for the correction warps to free the stats slot.
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        softmax_scale_log2: Float32,
        seqlen_q: Int32,
        tile_scheduler: "TileSchedulerProtocol",
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        mO: cute.Tensor,
        tma_atom_O: Optional[cute.CopyAtom],
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx_abs = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx = warp_idx_abs % 4
        # With the epilogue folded into the correction warps, warp 8 issues the O TMA store and the
        # four warps synchronise on this named barrier around the sO writes.
        is_store_warp = warp_idx_abs == self.correction_warp_ids[0]
        corr_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.Epilogue),
            num_threads=cute.arch.WARP_SIZE * len(self.correction_warp_ids),
        )

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            mO_cur = mO[None, None, None, batch_idx][None, None, head_idx]
            tiler_gO = (self.mma_tiler_pv[0], self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]
            store_O_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO
            )

            stats = [(Float32(0.0), -Float32.inf, True)] * self.s_stage

            topk_eff, block_iter_count, has_work = self.tile_counts(
                mIndices, mTopkLen, m_block, head_idx
            )

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # Wait for both softmax warp groups.
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                # q_stage=1 correction loop
                corr_pair_count = (block_iter_count - 2) // 2
                # Rescale the two alternating stages as a pair.
                for i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv, tOtO[None, None, None, stage], tidx, scale
                            )
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.s_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                # N even: no remainder. Release final sm_stats stage 1.
                pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    row_max = sScale[
                        tidx
                        + stage * self.m_block_size
                        + self.s_stage * self.m_block_size
                    ]
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # row_max is in original S space (unscaled). To compute rescale
                # factors we need exp2((row_max - max_combined) * scale_log2).
                # For empty/padding stages (zero_or_nan=True), row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1 = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = (
                    max_combined if max_combined != -Float32.inf else Float32(0.0)
                )
                scale0 = (
                    cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan0
                    else Float32(0.0)
                )
                scale1 = (
                    cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan1
                    else Float32(0.0)
                )
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                combined_zero_or_nan = (
                    sum_combined == 0.0 or sum_combined != sum_combined
                )
                inv_sum = cute.arch.rcp_approx(
                    sum_combined if not combined_zero_or_nan else 1.0
                )
                final_scale0 = scale0 * inv_sum
                final_scale1 = scale1 * inv_sum

                # Wait for both O accumulators from MMA warp
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )
                self.correction_acquire_sO(corr_barrier, is_store_warp)
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    final_scale0,
                    final_scale1,
                    sO[None, None, 0],
                )
                # Release both O buffers in tmem
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_s_p_o.consumer_release_w_index(stage)
                self.correction_commit_sO(corr_barrier, is_store_warp, store_O_fn)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
            else:
                # Empty tile (block_count == 0): sync pipelines and write O=0.
                # Match softmax's 1 barrier arrive per stage.
                for stage_idx in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=stage_idx * 4 + warp_idx
                    )
                    pipeline_sm_stats.consumer_release_w_index(stage_idx)
                sm_stats_consumer_phase ^= 1
                self.correction_acquire_sO(corr_barrier, is_store_warp)
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                self.correction_commit_sO(corr_barrier, is_store_warp, store_O_fn)

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                # q_stage=1: compute combined LSE from two partial softmax stats
                m_tile_idx = m_block
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]
                rm0_lse = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1_lse = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0_lse, rm1_lse)
                max_safe = (
                    max_combined if max_combined != -Float32.inf else Float32(0.0)
                )
                s0 = (
                    cute.math.exp2(
                        (rm0_lse - max_safe) * softmax_scale_log2, fastmath=True
                    )
                    if not zero_or_nan0
                    else Float32(0.0)
                )
                s1 = (
                    cute.math.exp2(
                        (rm1_lse - max_safe) * softmax_scale_log2, fastmath=True
                    )
                    if not zero_or_nan1
                    else Float32(0.0)
                )
                sum_comb = row_sum0 * s0 + row_sum1 * s1
                comb_zero_or_nan = sum_comb == 0.0 or sum_comb != sum_comb
                LN2 = math.log(2.0)
                lse = (
                    (
                        max_safe * softmax_scale_log2
                        + cute.math.log2(sum_comb, fastmath=True)
                    )
                    * LN2
                    if not comb_zero_or_nan
                    else -Float32.inf
                )
                packed_seqlen_q = seqlen_q * self.qhead_per_kvhead
                if tidx < packed_seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse

            # Advance to next tile
            work_tile = tile_scheduler.advance_to_next_work()
        # End of persistent scheduler loop

        if is_store_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def correction_acquire_sO(
        self,
        corr_barrier: pipeline.NamedBarrier,
        is_store_warp: Boolean,
    ):
        """Wait for the previous TMA store to finish reading sO before all writers reuse it."""
        if is_store_warp:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        corr_barrier.arrive_and_wait()

    @cute.jit
    def correction_commit_sO(
        self,
        corr_barrier: pipeline.NamedBarrier,
        is_store_warp: Boolean,
        store_O_fn: Callable,
    ):
        """Synchronize the fenced sO writers before warp 8 issues its TMA store."""
        corr_barrier.arrive_and_wait()
        if is_store_warp:
            store_O_fn(src_idx=0, dst_idx=0)
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale an FP32 partial O accumulator in TMEM when its softmax row maximum changes."""
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tOtO_i = cute.composition(
            tOtO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.composition(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_rmem_tensor(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_combine(
        self,
        thr_mma: cute.ThrMma,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tidx: Int32,
        scale0: Float32,
        scale1: Float32,
        sO: cute.Tensor,
    ):
        """Combine the alternating tiles' FP32 O accumulators and convert the result into sO."""
        corr_tile_size = 8 * 32 // self.o_dtype.width
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO0_i = cute.logical_divide(
            tOtO0, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOtO1_i = cute.logical_divide(
            tOtO1, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO0_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO0_t2r = thr_tmem_load.partition_S(tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO0_t2r_i = tOtO0_t2r[None, 0, 0, i]
            tOtO1_t2r_i = tOtO1_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            frg_shape = tOcO_t2r[None, 0, 0, i].shape
            tOrO0_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            # When both scales are 0 (empty tile), skip tmem reads to avoid 0*NaN=NaN.
            is_zero_output = scale0 == Float32(0.0) and scale1 == Float32(0.0)
            if not is_zero_output:
                cute.copy(tiled_tmem_load, tOtO0_t2r_i, tOrO0_frg)
                cute.copy(tiled_tmem_load, tOtO1_t2r_i, tOrO1_frg)
                # Combined: O = O0 * scale0 + O1 * scale1
                for j in cutlass.range(0, cute.size(tOrO0_frg), 2, unroll_full=True):
                    o0_a, o0_b = cute.arch.mul_packed_f32x2(
                        (tOrO0_frg[j], tOrO0_frg[j + 1]), (scale0, scale0)
                    )
                    o1_a, o1_b = cute.arch.mul_packed_f32x2(
                        (tOrO1_frg[j], tOrO1_frg[j + 1]), (scale1, scale1)
                    )
                    tOrO0_frg[j], tOrO0_frg[j + 1] = cute.arch.add_packed_f32x2(
                        (o0_a, o0_b), (o1_a, o1_b)
                    )
            else:
                tOrO0_frg.fill(Float32(0.0))
            copy_utils.cvt_copy(tiled_smem_store, tOrO0_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def tile_counts(
        self,
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        m_block: Int32,
        head_idx: Int32,
    ) -> Tuple[Int32, Int32, Boolean]:
        """Effective index count, even-rounded 128-slot tile count, and whether the tile has work.

        Rounding the tile count up to an even number keeps the two softmax warpgroups' alternation
        static; the phantom tile only holds slots >= topk_eff, which gather as
        zero-filled rows and are masked to -inf.
        """
        topk = Int32(mIndices.shape[2])
        topk_eff = topk
        if const_expr(mTopkLen is not None):
            topk_eff = cutlass.min(
                cutlass.max(mTopkLen[m_block, head_idx], Int32(0)), topk  # type: ignore[index]
            )
        raw_tile_count = (topk_eff + self.n_block_size - 1) // self.n_block_size
        block_iter_count = (raw_tile_count + 1) & ~1
        process_tile = raw_tile_count > Int32(0)
        return topk_eff, block_iter_count, process_tile

    @cute.jit
    def first_masked_tile(
        self,
        mIndices: cute.Tensor,
        m_block: Int32,
        head_idx: Int32,
        topk_eff: Int32,
        seqlen_k: Int32,
        lane_idx: Int32,
    ) -> Int32:
        """Index of the first 128-slot tile holding an invalid slot (``-1``, out-of-range id, or ``slot >= topk_eff``).

        One warp scans the indices row in 128-slot passes (one aligned int4 per lane, the wrapper
        guarantees ``topk % 4 == 0`` and 16 B rows), keeps the smallest invalid slot per lane and
        reduces across the warp once; the passes are unrolled so their loads overlap.  Tiles below the
        result are fully valid and need no mask; tiles from it on take the masked softmax step.
        """
        topk = Int32(mIndices.shape[2])
        cand = topk_eff
        n_pass = (topk_eff + Int32(self.n_block_size - 1)) // self.n_block_size
        for p in cutlass.range(n_pass, unroll=16):
            slot = p * self.n_block_size + lane_idx * 4
            toks = _ldg_indices_int4(
                mIndices, m_block, head_idx, cutlass.min(slot, topk - Int32(4))
            )
            for j in cutlass.range_constexpr(4):
                sj = slot + Int32(j)
                invalid = sj < topk_eff and (toks[j] < Int32(0) or toks[j] >= seqlen_k)
                cand = cutlass.min(cand, sj if invalid else topk_eff)
        for off in cutlass.range_constexpr(5):
            cand = cutlass.min(cand, cute.arch.shuffle_sync_bfly(cand, 1 << off))
        return cand // self.n_block_size

    @cute.jit
    def tile_valid_mask(
        self,
        mIndices: cute.Tensor,
        m_block: Int32,
        head_idx: Int32,
        logical_tile: Int32,
        topk_eff: Int32,
        seqlen_k: Int32,
        lane_idx: Int32,
    ) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
        """Four validity words for one 128-slot tile: bit l of word s is slot s * 32 + l.

        Each warp loads the indices row directly (one coalesced 128 B load per word), so the mask
        never depends on shared memory the gather pipeline may already have recycled.
        """
        slot_base = logical_tile * self.n_block_size
        words = []
        for s in cutlass.range_constexpr(self.n_block_size // cute.arch.WARP_SIZE):
            slot = slot_base + s * cute.arch.WARP_SIZE + lane_idx
            token = Int32(-1)
            if slot < topk_eff:
                token = mIndices[m_block, head_idx, slot]
            valid = token >= Int32(0) and token < seqlen_k
            words.append(Uint32(cute.arch.vote_ballot_sync(valid)))
        return words[0], words[1], words[2], words[3]

    @cute.jit
    def load_tile_toks(
        self,
        mIndices: cute.Tensor,
        m_block: Int32,
        head_idx: Int32,
        logical_tile: Int32,
        lane_idx: Int32,
    ):
        """Raw token ids of one 128-slot tile, four coalesced words: lane ``l`` gets slots ``32 r + l``.

        The slot is clamped to the last index column so the phantom tile and the dummy prefetch after
        the last load stay in bounds; validity is decided at use time from the unclamped slot.
        """
        topk = Int32(mIndices.shape[2])
        slot_base = logical_tile * self.n_block_size
        toks = []
        for r in cutlass.range_constexpr(self.n_block_size // cute.arch.WARP_SIZE):
            slot = slot_base + Int32(r * cute.arch.WARP_SIZE) + lane_idx
            toks.append(mIndices[m_block, head_idx, cutlass.min(slot, topk - Int32(1))])
        return toks[0], toks[1], toks[2], toks[3]

    @cute.jit
    def resolve_row_ids(
        self,
        toks: Tuple[Int32, Int32, Int32, Int32],
        logical_tile: Int32,
        topk_eff: Int32,
        seqlen_k: Int32,
        nheads_kv: Int32,
        head_idx: Int32,
        lane_idx: Int32,
    ):
        """Table rows of one tile from its raw token ids: ``token * nheads_kv + head_idx``, or ``-1`` for an invalid slot.

        An invalid slot is ``-1``, an id outside ``[0, seqlen_k)`` or ``slot >= topk_eff``; the gather
        uses the sign as the cp.async ``ignore-src`` predicate and zero-fills the row (see
        :meth:`_cp_async_row_pair`).
        """
        slot_base = logical_tile * self.n_block_size
        rows = []
        for r in cutlass.range_constexpr(self.n_block_size // cute.arch.WARP_SIZE):
            slot = slot_base + Int32(r * cute.arch.WARP_SIZE) + lane_idx
            tok = toks[r]
            valid = slot < topk_eff and tok >= Int32(0) and tok < seqlen_k
            rows.append(tok * nheads_kv + head_idx if valid else Int32(-1))
        return rows[0], rows[1], rows[2], rows[3]

    @cute.jit
    def cp_async_lane_constants(
        self, mKV: cute.Tensor, lane_idx: Int32, warp_rank: Int32
    ):
        """Per-lane invariants of the cp.async gather (see :meth:`gather_KV_cp_async`).

        Returns ``(two_w, lane_base, src_base, row_bytes, dst_rel)``: ``row_even = 2 * gather_warps * i + two_w``
        for iteration ``i``; ``lane_base = two_w + half`` is the shuffle source lane before the
        per-iteration constant; ``src_base`` is table row 0 plus this lane's 16 B column; ``dst_rel[j]`` is
        the destination of iteration ``i`` with ``i % period == j`` relative to the stage base
        (``chunk * 16384 + (two_w + half) * 128 + ((seg ^ half) ^ (row_even % 8)) * 16``, the
        ``row_even * 128`` part being the instruction's immediate).
        """
        elem_bytes = self.k_dtype.width // 8
        rows_per_iter = 2 * self.gather_warps
        seg = lane_idx % 16  # 16 B segment of the 256 B row
        half = lane_idx // 16  # which of the warp's two rows
        two_w = 2 * warp_rank
        lane_base = two_w + half
        src_base = Int64(mKV.iterator.toint()) + Int64(seg * 16)
        row_bytes = opaque_u32(cutlass.Uint32(mKV.stride[0] * elem_bytes))
        base = (seg // 8) * Int32(self.chunk_stride_elems * elem_bytes) + (
            two_w + half
        ) * 128
        period = 8 // math.gcd(rows_per_iter, 8)
        swz_half = (seg % 8) ^ half
        dst_rel = tuple(
            base + (swz_half ^ ((Int32(rows_per_iter * j) + two_w) % 8)) * 16
            for j in range(period)
        )
        return two_w, lane_base, src_base, row_bytes, dst_rel

    @cute.jit
    def gather_KV_cp_async(
        self,
        mKV: cute.Tensor,
        sKV_base_addr: Int32,
        mIndices: cute.Tensor,
        m_block: Int32,
        head_idx: Int32,
        lane_idx: Int32,
        two_w: Int32,
        lane_base: Int32,
        src_base: Int64,
        row_bytes: cutlass.Uint32,
        dst_rel: tuple,
        row_ids: Tuple[Int32, Int32, Int32, Int32],
        prefetch_tile: Optional[Int32],
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
    ):
        """Gather one 128-token K or V tile into the stage of ``producer_state`` with cp.async.

        ``two_w``, ``lane_base``, ``src_base``, ``row_bytes`` and ``dst_rel`` are the per-lane invariants
        from :meth:`cp_async_lane_constants` (hoisted out of the load loop).

        ``row_ids`` are the tile's table rows (:meth:`resolve_row_ids`, computed once per tile and
        reused for its K and V loads).  When ``prefetch_tile`` is given, that tile's raw token ids
        are loaded first and returned so their latency overlaps this tile's acquire and issue.
        The rows of invalid slots (``row_id < 0``) are zero-filled through the cp.async ``ignore-src``
        predicate: the softmax masks them to ``-inf``, and V must be zero there because ``P = 0``
        times an Inf/NaN table row would be NaN.

        Each warp instruction copies two full 256 B rows (lanes 0-15 one row, lanes 16-31 the next,
        16 B per lane).  Iteration ``i`` of warp ``w`` takes rows ``2 * gather_warps * i + 2 w`` and
        ``+1``; everything that depends on ``w`` is folded into a few per-load registers so each copy
        is four instructions: one shuffle (table row of the lane's row), one 32 x 32 -> 64 bit
        multiply-add (source address: both factors unsigned 32-bit values, the row pitch made opaque by
        :func:`opaque_u32`, so it is a single ``IMAD.WIDE.U32`` with the base as addend), one compare
        feeding the ``ignore-src`` predicate, and the ``cp.async`` itself, whose destination is a
        per-load register plus an immediate:

        * the 32-slot word holding the row's token is a compile-time constant except in the
          iterations where the warps straddle a word boundary (one select there);
        * the shuffle source lane is ``(row_even + half) % 32``; ``shfl.idx`` masks the lane to 5
          bits, so the compile-time part is simply added to ``2 w + half``;
        * the SW128 term ``(seg ^ (row % 8)) * 16`` has ``row_even % 8`` periodic in ``i``, so the few
          distinct destination bases are precomputed per load
          (``seg ^ (row % 8) == (seg ^ half) ^ (row_even % 8)``) and ``row_even * 128`` is the immediate.

        The destination is the SW128 K-major tile the UMMA descriptor expects: byte
        ``chunk * 16384 + row * 128 + ((seg ^ (row % 8)) * 16)`` for 16 B segment ``seg`` of the
        64-element chunk ``chunk`` (stage bases are 1024 B aligned, so the XOR on the absolute
        address reduces to ``row % 8``).

        Every gather thread then commits one cp.async group for the stage, waits until at most
        ``cp_async_depth`` groups are pending, fences its completed writes into the async proxy and
        arrives on the full barrier of the stage issued ``cp_async_depth`` loads ago.  Each thread
        commits exactly one group per load, so the per-thread group count tracks the stage order.
        """
        n_rows = self.n_block_size
        elem_bytes = self.k_dtype.width // 8
        gw = self.gather_warps
        rows_per_iter = 2 * gw
        n_iters = (n_rows + rows_per_iter - 1) // rows_per_iter

        # Prefetch the next K tile's token ids before anything that can stall.
        next_toks = None
        if const_expr(prefetch_tile is not None):
            next_toks = self.load_tile_toks(
                mIndices, m_block, head_idx, prefetch_tile, lane_idx
            )

        # Every gather thread waits for the MMA warp to release the stage.
        pipeline_kv.producer_acquire(producer_state)

        # Destination bases for this stage: stage offset plus the per-lane chunk / row / swizzle terms.
        period = len(dst_rel)
        stage_base = sKV_base_addr + producer_state.index * Int32(
            self.kv_stage_elems * elem_bytes
        )
        dsts = [stage_base + dst_rel[j] for j in range(period)]

        for i in cutlass.range_constexpr(n_iters):
            row_even_lo = (
                rows_per_iter * i
            )  # row_even for warp 0; warp w adds 2 w (< rows_per_iter)
            row_even_hi = row_even_lo + rows_per_iter - 2
            words = n_rows // cute.arch.WARP_SIZE
            v_lo = row_ids[row_even_lo // cute.arch.WARP_SIZE]
            v_hi = row_ids[min(row_even_hi // cute.arch.WARP_SIZE, words - 1)]
            if const_expr(row_even_hi < n_rows):
                self._cp_async_row_pair(
                    row_even_lo,
                    row_even_hi,
                    v_lo,
                    v_hi,
                    lane_base,
                    two_w,
                    src_base,
                    row_bytes,
                    dsts[i % period],
                )
            else:
                # Only the warps whose rows exist issue this iteration (warp-uniform branch).
                if Int32(row_even_lo) + two_w < Int32(n_rows):
                    self._cp_async_row_pair(
                        row_even_lo,
                        row_even_hi,
                        v_lo,
                        v_hi,
                        lane_base,
                        two_w,
                        src_base,
                        row_bytes,
                        dsts[i % period],
                    )

        depth = self.cp_async_depth_eff
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(depth)
        cute.arch.fence_view_async_shared()
        if producer_state.count >= Int32(depth):
            prev = producer_state.index - Int32(depth)
            pipeline_kv.producer_commit_w_index(
                prev + Int32(self.kv_stage) if prev < Int32(0) else prev
            )
        return next_toks

    @cute.jit
    def _cp_async_row_pair(
        self,
        row_even_lo: cutlass.Constexpr[int],
        row_even_hi: cutlass.Constexpr[int],
        v_lo: Int32,
        v_hi: Int32,
        lane_base: Int32,
        two_w: Int32,
        src_base: Int64,
        row_bytes: cutlass.Uint32,
        dst: Int32,
    ):
        """One warp instruction of the gather: lanes 0-15 copy row ``row_even``, lanes 16-31 ``row_even + 1``.

        ``v_lo``/``v_hi`` are this lane's table-row words for the 32-slot words that ``row_even_lo`` and
        ``row_even_hi`` (warp 0's and the last warp's rows) fall in; ``dst`` already holds the stage,
        chunk, row parity and swizzle term, ``row_even_lo * 128`` goes in as the immediate offset.
        A negative ``row_id`` (invalid slot) makes the copy write zeros without reading the source,
        so its (out-of-table) address is never dereferenced.
        """
        warp_size = cute.arch.WARP_SIZE
        word_lo = row_even_lo // warp_size
        word_hi = row_even_hi // warp_size
        v = v_lo
        if const_expr(word_hi != word_lo):
            # The warps straddle a 32-slot boundary in this iteration: warp w is in the upper word
            # when row_even_lo + 2 w >= (word_lo + 1) * 32.
            v = (
                v_hi
                if two_w >= Int32((word_lo + 1) * warp_size - row_even_lo)
                else v_lo
            )
        # shfl.idx keeps the low 5 bits of the source lane, so no modulo is needed.
        row_id = cute.arch.shuffle_sync(
            v, offset=lane_base + Int32(row_even_lo % warp_size)
        )
        src = src_base + Int64(cutlass.Uint32(row_id)) * Int64(row_bytes)
        cp_async_16_zfill(dst, src, row_id, row_even_lo * 128)

    @cute.jit
    def cp_async_drain(
        self,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
    ):
        """Complete all outstanding cp.async groups and arrive on the stages still pending.

        ``producer_state`` points past the last issued load; the newest ``min(count, depth)`` loads
        have not been committed yet.
        """
        depth = self.cp_async_depth_eff
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        for k in cutlass.range_constexpr(depth):
            if producer_state.count > Int32(k):
                pipeline_kv.producer_commit_w_index(
                    (producer_state.index + Int32(self.kv_stage - 1 - k))
                    % self.kv_stage
                )


def _ldg_indices_int4(
    indices: cute.Tensor, m_block: Int32, head_idx: Int32, slot: Int32
) -> Tuple[Int32, Int32, Int32, Int32]:
    """Load indices[m_block, head_idx, slot:slot+4] as one aligned int4 (slot % 4 == 0, 16 B aligned rows)."""
    ptr = indices.iterator + cute.crd2idx((m_block, head_idx, slot), indices.layout)
    ptr = cute.make_ptr(Int32, ptr.toint(), indices.memspace, assumed_align=16)
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [ptr.llvm_ptr],
        "{\n\t"
        ".reg .b64 v<2>;\n\t"
        "ld.global.nc.v2.u64 {v0, v1}, [$4];\n\t"
        "mov.b64 {$0, $1}, v0;\n\t"
        "mov.b64 {$2, $3}, v1;\n\t"
        "}\n",
        ",".join(["=r"] * 4 + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i])) for i in range(4))


@cute.jit
def apply_valid_mask(
    acc_S: cute.Tensor,
    w0: Uint32,
    w1: Uint32,
    w2: Uint32,
    w3: Uint32,
) -> None:
    """Mask the tile columns whose validity bit is clear (word s covers columns s*32..s*32+31)."""
    mask_r2p_lambda(
        acc_S,
        lambda s: (w0, w1, w2, w3)[s],
        rank1=True,
    )


# =============================================================================
# Python entry point
# =============================================================================
def ffa_fwd_sm100_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    q_block_size: int = 1,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of index-sparse (DSA) attention on SM100.

    Args:
        q: ``(total_q, num_heads_q, 128)`` fp16 / bf16.
        k, v: ``(total_k, num_heads_kv, 128)`` contiguous, same dtype as ``q``.
        index_sparse_indices: int32 ``(ceil(total_q / q_block_size), num_heads_kv, topk)``.
            Row ``[b, h_kv]`` lists the global K token ids attended by q tokens
            ``[b * q_block_size, (b + 1) * q_block_size)`` of every q head of kv head
            ``h_kv``; valid ids come first and the tail is padded with -1 (ids
            ``>= total_k`` are also skipped).
            ``topk`` must be a multiple of 4.
        q_block_size: q tokens per index row.  The kernel handles one index row per
            128-row Q tile of ``q_block_size`` tokens x ``G = num_heads_q / num_heads_kv``
            heads, so ``q_block_size * G == 128`` is required (16 for 8:1 GQA, 128 for
            MHA).
        out, lse: optional preallocated ``(total_q, num_heads_q, 128)`` / fp32
            ``(total_q, num_heads_q)`` outputs.

    Returns:
        ``(out, lse)``: ``lse`` is the natural-log logsumexp of the scaled scores; a row
        that attends no valid token gets ``out = 0`` and ``lse = -inf``.
    """
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
    total_q, num_heads_q, head_dim = q.shape
    total_k, num_heads_kv, _ = k.shape
    assert head_dim == 128, "only head_dim == 128 is supported"
    assert k.shape == v.shape == (total_k, num_heads_kv, head_dim)
    assert q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
    assert (
        num_heads_q % num_heads_kv == 0
    ), "num_heads_q must be divisible by num_heads_kv"
    group = num_heads_q // num_heads_kv
    assert q_block_size * group == FFAFwdSm100Index.tile_m, (
        f"q_block_size * (num_heads_q / num_heads_kv) must be {FFAFwdSm100Index.tile_m}, "
        f"got {q_block_size} * {group}"
    )
    num_q_blocks = (total_q + q_block_size - 1) // q_block_size
    assert index_sparse_indices.dtype == torch.int32
    assert index_sparse_indices.ndim == 3 and index_sparse_indices.shape[:2] == (
        num_q_blocks,
        num_heads_kv,
    ), (
        f"index_sparse_indices must be (num_q_blocks={num_q_blocks}, "
        f"num_heads_kv={num_heads_kv}, topk), got {tuple(index_sparse_indices.shape)}"
    )
    topk = index_sparse_indices.shape[2]
    assert topk > 0 and topk % 4 == 0, "topk must be a positive multiple of 4"
    assert total_k * num_heads_kv < 2**31, "total_k * num_heads_kv must fit in int32"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    q, k, v, index_sparse_indices = [
        maybe_contiguous(t) for t in (q, k, v, index_sparse_indices)
    ]
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(total_q, num_heads_q, dtype=torch.float32, device=q.device)
    assert out.shape == q.shape and out.dtype == q.dtype
    assert lse.shape == (total_q, num_heads_q) and lse.dtype == torch.float32
    assert out.is_contiguous() and lse.is_contiguous()

    assert k.is_contiguous() and v.is_contiguous(), "K/V must be contiguous row tables"

    dtype = to_cute_dtype(q.dtype)
    compile_key = (
        dtype,
        group,
        torch.cuda.get_device_capability(q.device),
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
    )
    compile_cache = ffa_fwd_sm100_index.compile_cache  # type: ignore[attr-defined]
    if compile_key not in compile_cache:
        kernel = FFAFwdSm100Index(
            head_dim,
            qhead_per_kvhead=group,
            debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
        )
        q_tensor, k_tensor, v_tensor, out_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        indices_tensor = to_cute_tensor(index_sparse_indices)
        compile_cache[compile_key] = cute.compile(
            kernel,
            q_tensor,
            k_tensor,
            v_tensor,
            out_tensor,
            lse_tensor,
            float(softmax_scale),
            indices_tensor,
            None,  # topk_len: the padding -1 marks the unused slots
            cutlass.Int32(num_heads_kv),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        out,
        lse,
        float(softmax_scale),
        index_sparse_indices,
        None,
        cutlass.Int32(num_heads_kv),
    )
    return out, lse


ffa_fwd_sm100_index.compile_cache = get_jit_cache("fwd_sm100_index")  # type: ignore[attr-defined]
