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

# mypy: disable-error-code="no-redef,union-attr,index,attr-defined,assignment,arg-type,has-type,misc"
# pyright: reportInvalidTypeForm=false

import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass import Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

# isort: split
import quack.activation
from quack import copy_utils, layout_utils
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils
from . import pipeline as ffa_pipeline
from . import sm100_utils
from .block_info import BlockInfo
from .cutedsl_utils import ThreadCooperativeGroup
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .named_barrier import NamedBarrierBwdSm100
from .seqlen_info import SeqlenInfoQK
from .softmax import apply_score_mod_bwd_inner, apply_score_mod_inner
from .sparse_utils import (
    BlockSparseTensors,
    get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm100,
)
from .tile_scheduler import (
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)


class FFABwdSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        mask_type: int = MT_MAP.full,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        spt: Optional[bool] = None,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
        debug_print: bool = False,
    ):
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        assert self.tile_hdim <= 128 or (
            self.tile_hdim == 192 and self.tile_hdimv == 128
        )
        assert self.tile_hdimv <= 128

        self.use_2cta_instrs = bool(
            use_2cta_instrs
            and cluster_size == 2
            and score_mod is None
            and score_mod_bwd is None
            and mask_mod is None
        )
        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        assert (
            self.tile_hdim != 192 or self.use_2cta_instrs
        ), "Must use 2CTA for hdim 192"

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)

        # S.T = K @ Q.T => (tileK128*CTA2,tileQ128,tileHD128)
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP.T = V @ dO.T => (tileK128*CTA2,tileQ128,tileHD128)
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO => (tileK128*CTA2,tileHD128,tileQ128)
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q => (tileK128*CTA2,tileHD128,tileQ128)
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K => (tileQ128,tileHD128,tileK128*CTA2)
        # NOTE: for 2-CTA mode, reduction is along cluster-wide tileK dim.
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * self.cta_group_size)

        self.acc_dtype = Float32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.deterministic = deterministic
        self.spt_override = spt

        # Score mod and mask mod support
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        # For score_mod, use vec_size=1 (like forward) to handle per-element indices
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        # Generally slower to use store dS in smem for dK, and doesn't work for 2cta
        self.use_smem_dS_for_mma_dK = False

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM buffer distribution
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        if self.use_2cta_instrs and self.tile_hdim == 192 and self.tile_hdimv == 128:
            assert self.tile_m == 128
            assert self.tile_n == 128
            self.tmem_dV_offset = 0
            self.tmem_dK_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_S_offset = self.tmem_dK_offset + self.tile_hdim
            self.tmem_P_offset = self.tmem_S_offset  # overlap with S
            self.tmem_dP_offset = 512 - self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
            self.tmem_dQ_offset = 512 - self.tile_hdim // 2
        else:
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # embedded in left-half of S

            self.tmem_dV_offset = self.tmem_S_offset + self.tile_n

            self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_dS_offset = self.tmem_dP_offset  # embedded in left-half of dP

            # NOTE:
            # 1. in 1-CTA mode: tdQ with shape (tileQ,tileHD) is fully overlapped with tdP,
            #   where dP(i) GEMM waits until dQ(i-1) GEMM finished and tdQ(i-1) consumed by dQacc wg
            # 2. in 2-CTA mode: tdQ with shape (tileQ//2,tileHD) is embedded in the right-half of S
            #   where S(i) GEMM waits until dQ(i-1) GEMM finished and tdQ(i-1) consumed by dQacc wg
            self.tmem_dQ_offset = (
                (self.tmem_S_offset + (self.tile_hdim // 2))
                if self.use_2cta_instrs  # 2-CTA: embedded in right-half of S
                else self.tmem_dP_offset  # 1-CTA: fully overlapped with dP
            )

            self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m

        if (not self.is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
            self.num_regs_compute = 136
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        else:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 136
            self.num_regs_compute = 136 if self.use_2cta_instrs else 144
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        self.num_regs_empty = 24

        if const_expr(self.tile_hdim == 192):
            if not self.is_causal and not is_local:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load
            else:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= self.tmem_alloc_cols
        )

        self.buffer_align_bytes = 1024

        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[bwd_sm100_init] "
            print()
            print(f"{prefix}Initialized FFABwdSm100 with: ")
            print(
                f"{prefix}{self.tile_hdim=} | {self.tile_hdimv=} | {self.qhead_per_kvhead=}"
            )
            print(
                f"{prefix}{self.mask_type=} | {self.is_causal=} | {self.is_local=} | "
                f"{self.is_persistent=} | {self.deterministic=}"
            )
            print(
                f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.use_2cta_instrs=} | {self.cta_group_size=}"
            )
            print(f"{prefix}{self.cluster_shape_mn=}")
            print(f"{prefix}{self.mma_tiler_kq=} | {self.mma_tiler_vdo=}")
            print(
                f"{prefix}{self.mma_tiler_pdo=} | {self.mma_tiler_dsq=} | {self.mma_tiler_dsk=}"
            )
            print(f"{prefix}{self.reduce_warp_ids=} | {self.compute_warp_ids=}")
            print(
                f"{prefix}{self.mma_warp_id=} | {self.load_warp_id=} | {self.relay_warp_id=} | {self.empty_warp_id=}"
            )
            print(
                f"{prefix}{self.tmem_S_offset=} | {self.tmem_P_offset=} | {self.tmem_dV_offset=}"
            )
            print(
                f"{prefix}{self.tmem_dP_offset=} | {self.tmem_dQ_offset=} | {self.tmem_dK_offset=} | {self.tmem_dS_offset=}"
            )
            print(
                f"{prefix}{self.num_regs_reduce=} | {self.num_regs_compute=} | {self.num_regs_load=} | {self.num_regs_mma=}"
            )
            print(
                f"{prefix}{self.score_mod=} | {self.score_mod_bwd=} | {self.mask_mod=}"
            )
            print()

    @property
    def is_causal(self) -> bool:
        return self.mask_type == MT_MAP.causal

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs else 2
        self.dO_stage = 1
        self.single_stage = 1
        self.sdKVaccum_stage = 2

        # Determine number of tma reduce adds per dQacc mma
        # TODO: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.use_2cta_instrs and self.tile_hdim == 192:
            self.dQ_reduce_ncol_t2r = 32
            self.dQ_reduce_ncol = 24 if not self.is_causal else 32
            self.sdQacc_stage = 2 if not self.is_causal else 1
        else:
            if self.use_2cta_instrs:
                self.dQ_reduce_ncol = 16 if self.deterministic else 8
                self.sdQacc_stage = 2 if self.deterministic else 4
                self.dQ_reduce_ncol_t2r = 32
            else:
                self.dQ_reduce_ncol = 32
                self.sdQacc_stage = 64 // self.dQ_reduce_ncol
                self.dQ_reduce_ncol_t2r = self.dQ_reduce_ncol

        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQacc_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQacc_reduce_stage_cta = self.dQacc_reduce_stage // self.cta_group_size
        self.dQacc_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1

        # Determine number of tma reduce adds
        # for dKacc and dVacc epilogue (must divide hdim_per_wg)
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)

        # CTA group for MMA operations
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[bwd_sm100_setup_attributes] "
            print()
            print(
                f"{prefix}{self.Q_stage=} | {self.dO_stage=} | "
                f"{self.single_stage=} | {self.sdKVaccum_stage=}"
            )
            print(
                f"{prefix}{self.dQ_reduce_ncol=} | {self.dQ_reduce_ncol_t2r=} | "
                f"{self.sdQacc_stage=}"
            )
            print(
                f"{prefix}{self.dQacc_reduce_stage=} | {self.dQacc_reduce_stage_t2r=}"
            )
            print(
                f"{prefix}{self.cluster_reduce_dQ=} | {self.dK_reduce_ncol=} | "
                f"{self.cta_group=}"
            )
            print()

    def _get_tiled_mma(self):
        # --- S.T = K @ Q.T with (K, K) major ---

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )

        # --- dP.T = V @ dO.T with (K, K) major ---

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )

        # --- dV = P.T @ dO with (K, MN) major ---

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,  # tP
        )

        # --- dK = dS.T @ Q with (K, MN) major ---

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=(
                tcgen05.OperandSource.SMEM
                if const_expr(self.use_smem_dS_for_mma_dK)
                else tcgen05.OperandSource.TMEM
            ),  # tdS or sdS
        )

        # --- dQ = dS @ K with (MN, MN) major ---

        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (128,128,16)
        # TV Layout A:     (2,(64,16)):(64,(1,128))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(64,128)):(64,(1,128))
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )

        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # --- S.T = K @ Q.T with (K, K) major ---

        # sK: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_HD=(4,2)):((64,1),0,(16,8192))
        # sQ: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD=(4,2),stageQ):((64,1),0,(16,4096),0)
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )

        # --- dP.T = V @ dO.T with (K, K) major ---

        # sV: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_HD=(4,2)):((64,1),0,(16,8192))
        # sdOt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD=(4,2),stagedO):((64,1),0,(16,4096),0)
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )

        # --- dV += P.T @ dO with (K, MN) major ---

        # tPt: S<3,4,3> o 0 o (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        # sdO: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD=8,stagedO):((1,64),0,1024,0)
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )

        # --- dK += dS.T @ Q with (K, MN) major ---

        # sdSt: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        # tdSt: S<3,4,3> o 0 o (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        # sQt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD8,stageQ):((1,64),0,1024,0)
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )

        # --- dQ = dS @ K with (MN, MN) major ---

        # sdS: S<3,4,3> o 0 o (MMA_sA=(64,16),MMA_K1,MMA_Q16):((1,64),0,1024)
        # sKt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD16):((1,64),0,1024)
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))

        # --- Make other smem layouts ---

        # sdS_xchg: (tileK128,tileQ128//2):(1,128)
        # sdQacc: (tileQ128*RedColdQ,stagedQ):(1,1024)
        # sLSE: (tileQ128,stageQ):(1,128)
        # sdPsum: (tileQ128,stagedO):(1,128)
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))
        self.sdQacc_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQacc_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )

        # --- Make epilogue smem layouts ---

        self.sdK_epi_tile = (  # (tileK128, 64) or (tileK128, 32) if hd is small
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (  # (tileK128, 64) or (tileK128, 32) if hd is small
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),
        )

        self.num_epi_stages = max(
            1, (self.tile_hdim // 2) // self.sdK_epi_tile[1]
        )  # hd64 gets 1 stage
        self.num_epi_stages_v = max(
            1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1]
        )  # hd64 gets 1 stage
        self.sdK_flat_epi_tile = (
            self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        )
        self.sdV_flat_epi_tile = (
            self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        )
        self.num_compute_wgs = len(self.compute_warp_ids) // 4

        # sdK_epi: S<3,4,3> o 0 o (EPI_K=(8,16),EPI_HD=(64,1),stageEPI=(1,2)):((64,512),(1,0),(0,8192))
        # sdV_epi: S<3,4,3> o 0 o (EPI_K=(8,16),EPI_HD=(64,1),stageEPI=(1,2)):((64,512),(1,0),(0,8192))
        if const_expr(not self.dKV_postprocess):
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                self.num_compute_wgs,
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                self.num_compute_wgs,
            )
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQacc: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mdO/mdQacc/mdK/mdV/mLSE/mdPsum tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqacc_dtype = mdQacc.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        self.use_tma_store = not (
            self.qhead_per_kvhead == 1 and mCuSeqlensK is not None
        )
        self.dKV_postprocess = self.qhead_per_kvhead > 1

        if const_expr(self.dKV_postprocess):
            assert (
                self.dk_dtype.width == 32
            ), "Must accumulate dK in float precision for GQA"
            assert (
                self.dv_dtype.width == 32
            ), "Must accumulate dV in float precision for GQA"

        mdQacc, mdK, mdV = [
            cutedsl_utils.assume_tensor_aligned(t) for t in (mdQacc, mdK, mdV)
        ]

        # --- Make mQ/mdO ---

        # (b, sq, nhq, hd) -> (sq, hd, nhq, b)
        # or (sq, nhq, hd) -> (sq, hd, nhq) if there's cu_seqlens_q
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]

        # (sq, hd, nhq, b) --> (hd, sq, nhq, b)
        # or (sq, hd, nhq) -> (hd, sq, nhq) if there's cu_seqlens_q
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)  # => actually dO.T

        # --- Make mK/mV ---

        # (b, sk, nhk, hd) -> (sk, hd, nhk, b)
        # or (sk, nhk, hd) -> (sk, hd, nhk) if there's cu_seqlens_k
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]

        # --- Make mLSE/mdPsum ---

        # (b, nhq, sq) --> (sq, nhq, b)
        # or (nhq, sq) --> (sq, nhq) if there's cu_seqlens_q
        LSE_dPsum_dQacc_transpose = (
            [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        )
        mLSE, mdPsum = [
            layout_utils.select(t, mode=LSE_dPsum_dQacc_transpose)
            for t in (mLSE, mdPsum)
        ]

        # --- Make mdQacc ---

        # (b, nhq, sq*hd) --> (sq*hd, nhq, b)
        # or (nhq, sq*hd) --> (sq*hd, nhq) if there's cu_seqlens_q
        mdQacc = layout_utils.select(mdQacc, mode=LSE_dPsum_dQacc_transpose)

        # --- Make mdK/mdV ---

        # (b, sk, nhk, hd) -> (sk, hd, nhk, b)
        # or (sk, nhk, hd) -> (sk, hd, nhk) if there's cu_seqlens_k
        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
            )
        mdK, mdV = [
            layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)
        ]

        # --- Make semaphore for mdQ/mdK/mdV ---

        # NOTE: this is only used for deterministic mode

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        # ///////////////////////////////////////////////////////////////////////////////
        # Set up attributes
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Set up attributes ---

        self._setup_attributes()

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tiled MMA, tiled TMA copy, and SMEM layouts
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tiled MMA ---

        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()

        # --- Make smem layout ---

        self._setup_smem_layout()

        # --- Make tiled TMA S2G-copy of dK/dV ---

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cta_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and not self.dKV_postprocess):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        # --- Make tiled R2S-copy of dK/dV ---

        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout(
                (128, 1), order=(1, 0)
            )  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        # --- Make tiled TMA G2S-copy of Q/K/V/dO ---

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cta_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cta_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cta_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cta_layout_vmnk.shape,
        )

        # --- Make tiled TMA G2S-copy of Qt/dOt/Kt for 2-CTA ---

        # NOTE: in 2-CTA mode, to accurately apply GEMM for dK = dS.T @ Q and dQ = dS @ K,
        # we need reload sQt with shape (tileHD128//2, tileQ128) per CTA
        # and sKt (tileHD128//2, tileK128*CTA2) per CTA

        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cta_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs):
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cta_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs):
            Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mnk, self.tiled_mma_dQ.thr_id
            )
            tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                Kt_tma_op,
                layout_utils.select(mK, mode=transpose_sh_k),
                cute.select(self.sKt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsk,
                self.tiled_mma_dQ,
                self.cta_layout_vmnk.shape,
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        )
        self.tma_copy_bytes["dKacc"] = (
            self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        )
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = (
            self.tma_copy_bytes["dS"] // 2
        )  # Half of sdS for all2all exchange

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        if const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = (self.is_causal or self.is_local) and self.deterministic
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic

        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batches
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,  # pack_gqa disabled for bwd
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,  # persistent mode not tested
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)

        self.tile_scheduler_cls = TileScheduler

        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        if const_expr(self.use_2cta_instrs):
            assert (
                sdV_bytes <= sV_bytes
            ), "sdV doesn't fit in sV storage allocation (2-CTA)"
            assert (
                sdK_bytes <= sK_bytes
            ), "sdK doesn't fit in sK storage allocation (2-CTA)"

            sQt_size = (
                cute.cosize(self.sQt_layout) if const_expr(self.tile_hdim <= 128) else 0
            )
            sdOt_size = (
                cute.cosize(self.sdOt_layout)
                if const_expr(self.tile_hdim <= 128)
                else 0
            )
            sdS_xchg_size = (
                cute.cosize(self.sdS_xchg_layout)
                if const_expr(self.tile_hdim <= 128)
                else 0
            )

            @cute.struct
            class SharedStorage:
                # ---  mbarriers for pipelines ---

                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, 2 * self.sdKVaccum_stage
                ]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQacc_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQacc_reduce_stage // 2
                ]

                # --- tmem ptr ---

                # Tmem dealloc cluster mbarrier
                tmem_dealloc_mbar_ptr: Int64
                # Tmem holding buffer ptr
                tmem_holding_buf_ptr: Int32

                # --- 2-CTA mbarrier ptrs ---

                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQacc_empty_mbar_ptr: cutlass.Int64

                # --- smem tensors ---

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
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
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[
                        self.dpsum_dtype, cute.cosize(self.sdPsum_layout)
                    ],
                    128,
                ]
                sdQacc: cute.struct.Align[
                    cute.struct.MemRange[
                        self.dqacc_dtype, cute.cosize(self.sdQacc_layout)
                    ],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]

        else:

            @cute.struct
            class SharedStorage:
                # ---  mbarriers for pipelines ---

                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, 2 * self.sdKVaccum_stage
                ]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQacc_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQacc_reduce_stage // 2
                ]

                # --- tmem ptr ---

                # Tmem dealloc cluster mbarrier
                tmem_dealloc_mbar_ptr: Int64
                # Tmem holding buffer ptr
                tmem_holding_buf_ptr: Int32

                # --- smem tensors ---

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
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
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[
                        self.dpsum_dtype, cute.cosize(self.sdPsum_layout)
                    ],
                    128,
                ]
                sdQacc: cute.struct.Align[
                    cute.struct.MemRange[
                        self.dqacc_dtype, cute.cosize(self.sdQacc_layout)
                    ],
                    self.buffer_align_bytes,
                ]

        self.shared_storage = SharedStorage

        # --- Make others ---

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            # Without score_mod: bake scale into log2
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            # With score_mod: score_mod applied to S * softmax_scale, then use LOG2_E only
            softmax_scale_log2 = LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(self.use_2cta_instrs):
            assert blocksparse_tensors is None, (
                "2-CTA mode does not support block sparsity. "
                "Please create kernel with use_2cta_instrs=False for block sparse attention."
            )
        if const_expr(self.use_block_sparsity or aux_tensors is not None):
            assert all(
                x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            ), "Variable sequence length is not supported yet for blocksparse or aux tensors in bwd"

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[bwd_sm100_call] "

            print()
            print(f"{prefix}tiled_mma_S: {self.tiled_mma_S}")
            print()
            print(f"{prefix}tiled_mma_dP: {self.tiled_mma_dP}")
            print()
            print(f"{prefix}tiled_mma_dV: {self.tiled_mma_dV}")
            print()
            print(f"{prefix}tiled_mma_dK: {self.tiled_mma_dK}")
            print()
            print(f"{prefix}tiled_mma_dQ: {self.tiled_mma_dQ}")
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sKt_layout: {self.sKt_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sdOt_layout: {self.sdOt_layout}")
            print(f"{prefix}sdO_layout: {self.sdO_layout}")
            print(f"{prefix}tP_layout: {self.tP_layout}")
            print(f"{prefix}sdSt_layout: {self.sdSt_layout}")
            print(f"{prefix}sdS_layout: {self.sdS_layout}")
            print(f"{prefix}tdS_layout: {self.tdS_layout}")
            print(f"{prefix}sdS_xchg_layout: {self.sdS_xchg_layout}")
            print(f"{prefix}sLSE_layout: {self.sLSE_layout}")
            print(f"{prefix}sdPsum_layout: {self.sdPsum_layout}")
            print(f"{prefix}sQt_layout: {self.sQt_layout}")
            print(f"{prefix}sdQacc_layout: {self.sdQacc_layout}")
            print(f"{prefix}sdK_layout: {self.sdK_layout}")
            print(f"{prefix}sdV_layout: {self.sdV_layout}")
            print(f"{prefix}sdK_epi_tile: {self.sdK_epi_tile}")
            print(f"{prefix}sdV_epi_tile: {self.sdV_epi_tile}")
            print(f"{prefix}sdK_flat_epi_tile: {self.sdK_flat_epi_tile}")
            print(f"{prefix}sdV_flat_epi_tile: {self.sdV_flat_epi_tile}")
            print()
            print(f"{prefix}num_epi_stages: {self.num_epi_stages}")
            print(f"{prefix}num_epi_stages_v: {self.num_epi_stages_v}")
            print(f"{prefix}num_compute_wgs: {self.num_compute_wgs}")
            print(f"{prefix}dKV_postprocess: {self.dKV_postprocess}")
            print(f"{prefix}use_tma_store: {self.use_tma_store}")
            print(
                f"{prefix}use_2cta_instrs: {self.use_2cta_instrs} | "
                f"use_block_sparsity: {self.use_block_sparsity}"
            )
            print(f"{prefix}threads_per_cta: {self.threads_per_cta}")
            print(f"{prefix}num_mcast_ctas_b: {self.num_mcast_ctas_b}")
            print(f"{prefix}is_q_do_mcast: {self.is_q_do_mcast}")
            print()

            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
            cute.printf(prefix + "mdO.layout: {}", mdO.layout)
            cute.printf(prefix + "mLSE.layout: {}", mLSE.layout)
            cute.printf(prefix + "mdPsum.layout: {}", mdPsum.layout)
            cute.printf(prefix + "mdV.layout: {}", mdV.layout)
            cute.printf(prefix + "mdK.layout: {}", mdK.layout)
            cute.printf(prefix + "mdQacc.layout: {}", mdQacc.layout)
            cute.printf("")
            cute.printf(prefix + "grid_dim: {}", grid_dim)
            cute.printf(
                prefix + "softmax_scale_log2={} softmax_scale={}",
                softmax_scale_log2,
                softmax_scale,
            )
            cute.printf("")

        # --- Launch the kernel ---

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdQacc,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQacc_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            tile_sched_params,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk
            if cute.size(self.cluster_shape_mnk) > 1
            else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQacc: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQacc_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Set up before warp specialization
        # /////////////////////////////////////////////////////////////////////////////

        # --- Set up thread info ---

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0
        cta_layout_vmnk = (
            cute.tiled_divide(  # (CTA_V(2),CTA_M1,CTA_N1,CTA_K1):((1),0,0,0)
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_S.thr_id.shape,),
            )
        )

        # Used only for debug print
        # guarded by const_expr so zero overhead when debug_print=False
        is_print_block = const_expr(self.debug_print) and (
            (bidx == 0) and (bidy == 0) and (bidz == 0)
        )
        is_print_thread = const_expr(self.debug_print) and (
            (tidx == 0) and is_print_block
        )

        # --- Prefetch TMA descriptor ---

        if warp_idx == self.load_warp_id:  # only one warp is enough
            for tma_atom in (
                tma_atom_Q,
                tma_atom_Qt,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dOt,
                tma_atom_dK,
                tma_atom_dV,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # --- Alloc smem storage and fetch ptrs ---

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Cluster mbarrier ptrs
        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            dQacc_empty_mbar_ptr = storage.dQacc_empty_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
            dQacc_empty_mbar_ptr = None

        # Pipeline mbarrier ptrs
        S_mbar_ptr = storage.S_mbar_ptr.data_ptr()
        dP_mbar_ptr = storage.dP_mbar_ptr.data_ptr()
        dKV_mbar_ptr = storage.dKV_mbar_ptr.data_ptr()
        dQ_mbar_ptr = storage.dQ_mbar_ptr.data_ptr()
        dS_mbar_ptr = storage.dS_mbar_ptr.data_ptr()
        LSE_mbar_ptr = storage.LSE_mbar_ptr.data_ptr()
        dPsum_mbar_ptr = storage.dPsum_mbar_ptr.data_ptr()
        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        dO_mbar_ptr = storage.dO_mbar_ptr.data_ptr()
        if const_expr(self.use_2cta_instrs):
            Qt_mbar_ptr = storage.Qt_mbar_ptr.data_ptr()
            Kt_mbar_ptr = storage.Kt_mbar_ptr.data_ptr()

        # tmem buf/dealloc ptrs
        tmem_holding_buf_ptr = storage.tmem_holding_buf_ptr
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr

        # --- Cluster mbarrier initialization ---

        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQacc_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQacc_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        # --- Alloc tmem alloc/dealloc barrier ---

        # NOTE: Only the mma warp drives tmem alloc/dealloc, and TmemAllocator internally
        # initializes the dealloc mbar only for the mma warp (covering both CTAs in a
        # 2-CTA cluster). This means the dealloc mbar alone cannot block until compute
        # and reduce warps finish using tmem. Therefore, all three warp groups
        # (mma + compute + reduce) must arrive on this shared barrier, giving the
        # mma warp a safe signal that tmem is no longer in use. And only by that point,
        # the mma warp (in 2-CTA cluster) can wait on dealloc mbar before it deallocates.

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            alloc_result_dst_smem_ptr=tmem_holding_buf_ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar_ptr,
        )

        # --- Make pipeline cooperative groups ---

        load_warp = ThreadCooperativeGroup(len([self.load_warp_id]))
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        mma_warp_mcast = ThreadCooperativeGroup(
            len([self.mma_warp_id]) * self.num_mcast_ctas_b,
        )
        compute_warps = ThreadCooperativeGroup(len(self.compute_warp_ids))
        compute_warps_cluster = ThreadCooperativeGroup(
            len(self.compute_warp_ids) * self.cta_group_size,
        )
        reduce_warps_cluster = ThreadCooperativeGroup(
            len(self.reduce_warp_ids) * self.cta_group_size,
        )

        # --- Make pipelines ---

        # S/P pipeline (MMA -> compute:softmax_fwd)
        pipeline_S_P = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=mma_warp,
            consumer_group=compute_warps_cluster,
            barrier_storage=S_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # dP pipeline (MMA -> compute:softmax_bwd)
        pipeline_dP = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=mma_warp,
            consumer_group=compute_warps_cluster,
            barrier_storage=dP_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # dKV pipeline (MMA -> compute)
        pipeline_dKV = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=mma_warp,
            consumer_group=compute_warps_cluster,
            barrier_storage=dKV_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # dO pipeline (MMA -> reduce)
        pipeline_dQ = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=mma_warp,
            consumer_group=reduce_warps_cluster,
            barrier_storage=dQ_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # dS pipeline (compute:softmax_bwd -> MMA)
        pipeline_dS = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=compute_warps_cluster,
            consumer_group=mma_warp,
            barrier_storage=dS_mbar_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # Load LSE pipeline (load -> compute:softmax_fwd)
        pipeline_LSE = pipeline.PipelineTmaAsync.create(
            barrier_storage=LSE_mbar_ptr,
            num_stages=self.Q_stage,
            producer_group=load_warp,
            consumer_group=compute_warps,
            tx_count=self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )

        # Load dPsum pipeline (load -> compute:softmax_bwd)
        pipeline_dPsum = pipeline.PipelineTmaAsync.create(
            barrier_storage=dPsum_mbar_ptr,
            num_stages=self.dO_stage,
            producer_group=load_warp,
            consumer_group=compute_warps,
            tx_count=self.tma_copy_bytes["dPsum"],
            defer_sync=True,
        )

        # Load Q pipeline (load -> MMA)
        pipeline_Q = ffa_pipeline.PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=self.Q_stage,
            producer_group=load_warp,
            consumer_group=mma_warp_mcast,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # Load Qt/Kt pipeline (load -> MMA)
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = ffa_pipeline.PipelineTmaUmma.create(
                    barrier_storage=Qt_mbar_ptr,
                    num_stages=self.Q_stage,
                    producer_group=load_warp,
                    consumer_group=mma_warp_mcast,
                    tx_count=self.tma_copy_bytes["Q"],
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = ffa_pipeline.PipelineTmaUmma.create(
                barrier_storage=Kt_mbar_ptr,
                num_stages=self.single_stage,
                producer_group=load_warp,
                consumer_group=mma_warp_mcast,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        # Load dO pipeline (load -> MMA)
        pipeline_dO = ffa_pipeline.PipelineTmaUmma.create(
            barrier_storage=dO_mbar_ptr,
            num_stages=self.dO_stage,
            producer_group=load_warp,
            consumer_group=mma_warp_mcast,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # --- Cluster arrive after mbarrier init ---

        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        # --- Make smem tensors of sQ/sK/sV/sdO/sdS/sLSE/sdPsum/sdQacc/sdK/sdV ---

        # sQ: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD=(4,2),stageQ):((64,1),0,(16,4096),0)
        # (operand B of S.T = K @ Q.T)
        sQ = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype
        )
        # sQt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD8,stageQ):((1,64),0,1024,0)
        # (Q transposed, operand B of dK += dS.T @ Q)
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sQt = storage.sQt.get_tensor(
                sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
            )
        else:
            # 1-CTA / large hdim: sQt aliases sQ's smem (strip swizzle, recast)
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype),
                sQt_layout.outer,
            )
        # sK: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_HD=(4,2)):((64,1),0,(16,8192))
        # (operand A of S.T = K @ Q.T)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sKt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD16):((1,64),0,1024)
        # (K transposed, operand B of dQ = dS @ K)
        if const_expr(self.use_2cta_instrs):
            sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        else:
            # 1-CTA: sKt aliases sK's smem (strip swizzle, recast)
            sKt = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer
            )
        # sV: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_HD=(4,2)):((64,1),0,(16,8192))
        # (operand A of dP.T = V @ dO.T)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        # sdSt: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        # (dS transposed, operand A of dK += dS.T @ Q)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        # sdS: S<3,4,3> o 0 o (MMA_sA=(64,16),MMA_Q1,MMA_K16):((1,64),0,1024)
        # (dS, operand A of dQ = dS @ K; aliases sdSt's smem, strip swizzle, recast)
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer
        )
        # sdS_xchg: (tileK128,tileQ128//2):(1,128)
        # (2-CTA only: scratch for all2all exchanging dS halves across 2-CTA group)
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim <= 128):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                # large hdim: reuse sdQacc smem for the dS exchange buffer
                sdS_xchg = storage.sdQacc.get_tensor(
                    sdS_xchg_layout, dtype=self.ds_dtype
                )
        else:
            sdS_xchg = None

        # sdO: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD8,stagedO):((1,64),0,1024,0)
        # (operand B of dV += P.T @ dO)
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        # sdOt: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_Q1,MMA_HD=(4,2),stagedO):((64,1),0,(16,4096),0)
        # (dO transposed, operand B of dP.T = V @ dO.T)
        if const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
            sdOt = storage.sdOt.get_tensor(
                sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
            )
        else:
            # 1-CTA / large hdim: sdOt aliases sdO's smem (strip swizzle, recast)
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        # sLSE: (tileQ128,stageQ):(1,128)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        # sdPsum: (tileQ128,stagedO):(1,128)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        # sdK_epi / sdV_epi: S<3,4,3> o 0 o (EPI_K=(8,16),EPI_HD=(64,1),stageEPI=(1,2)):((64,512),(1,0),(0,8192))
        # (dK/dV epilogue staging; reuse sK/sV (2-CTA) or sQ/sdO (1-CTA) smem)
        if const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(
                    sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
                )
                sdK = storage.sK.get_tensor(
                    sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
                )
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # sdQacc: (tileQ128*RedColdQ,stagedQ):(1,1024)
        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQacc = storage.sdQacc.get_tensor(sdQacc_layout)

        # --- Make tmem fragments of tS/tP / tdP / tdV / tdK/tdS / tdQ ---

        # NOTE: `tmem_ptr` + `make_fragment_C` returns a fake tensor with tmem col offset always at 0,
        # by right we need to explicitly retrieve tmem_ptr with `cute.arch.retrieve_tmem_ptr`.
        # But we know that we always request 512 columns of tmem, so we know that it must start at 0.

        # tStS: (MMA_tC=(row128,col128),MMA_K1,MMA_Q1):((65536,1),0,0)
        tmem_ptr = cute.make_ptr(
            self.acc_dtype, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # tdPtdP: (MMA_tC=(row128,col128),MMA_K1,MMA_Q1):((65536,1),0,0)
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # tdVtdV: (MMA_tC=(row128,col128),MMA_K1,MMA_HD1):((65536,1),0,0)
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        # tdKtdK: (MMA_tC=(row128,col128),MMA_K1,MMA_HD1):((65536,1),0,0)
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        # tdQtdQ: (MMA_tC=(intraRow64,(col64,interRow2)),MMA_Q1,MMA_HD1):((65536,(1,4194304)),0,0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(mma_tile_coord_v)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)
        # tP: (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype),
            tP_layout.outer,
        )
        # tdS: (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((64,1),0,(16,8192))
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype),
            tdS_layout.outer,
        )

        # --- Make other info dataclass ---

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n
            * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

        # --- Cluster wait before tensor memory alloc ---

        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # --- Make tile scheduler ---

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(
            tile_scheduler, TileSchedulerProtocol
        ), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread:
                prefix = "[bwd_sm100_kernel_setup] "
                cute.printf("")
                cute.printf(prefix + "cta_layout_vmnk: {}", cta_layout_vmnk)
                cute.printf(
                    prefix + "tile_m={} tile_n={} tile_hdim={} tile_hdimv={}",
                    self.tile_m,
                    self.tile_n,
                    self.tile_hdim,
                    self.tile_hdimv,
                )
                cute.printf(
                    prefix + "softmax_scale={} softmax_scale_log2={}",
                    softmax_scale,
                    softmax_scale_log2,
                )
                cute.printf("")
                cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                cute.printf(prefix + "sK.layout: {}", sK.layout)
                cute.printf(prefix + "sV.layout: {}", sV.layout)
                cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                cute.printf(prefix + "sdS.layout: {}", sdS.layout)
                cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                cute.printf(prefix + "sdQacc.layout: {}", sdQacc.layout)
                cute.printf("")
                cute.printf(prefix + "tStS.layout: {}", tStS.layout)
                cute.printf(prefix + "tdPtdP.layout: {}", tdPtdP.layout)
                cute.printf(prefix + "tdVtdV.layout: {}", tdVtdV.layout)
                cute.printf(prefix + "tdKtdK.layout: {}", tdKtdK.layout)
                cute.printf(prefix + "tdQtdQ.layout: {}", tdQtdQ.layout)
                cute.printf(prefix + "tP.layout: {}", tP.layout)
                cute.printf(prefix + "tdS.layout: {}", tdS.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        #  Empty Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Relay Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.relay_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(
                self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty
            )

            # --- Enter relay loop ---

            # NOTE: 2-CTA only

            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # --- Enter load loop ---

            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cta_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                tile_scheduler,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
                is_print_block=is_print_block,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA Warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # --- Alloc and retrieve tmem buffer ---

            tmem.allocate(self.tmem_alloc_cols)  # alias for `cute.arch.alloc_tmem`
            tmem.wait_for_alloc()  # alias for `tmem_alloc_barrier.arrive_and_wait`
            tmem_ptr = tmem.retrieve_ptr(  # alias for `cute.arch.retrieve_tmem_ptr`
                self.acc_dtype
            )

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
                tP,
                sdSt,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_leader_mbar_ptr,
                pipeline_Q,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                tile_scheduler,
                is_leader_cta,
                blocksparse_tensors,
                is_print_block=is_print_block,
            )

            # --- Dealloc tmem buffer ---

            tmem.relinquish_alloc_permit()  # alias for `cute.arch.relinquish_tmem_alloc_permit`
            tmem.wait_for_alloc()  # alias for `tmem_alloc_barrier.arrive_and_wait`
            tmem.free(
                tmem_ptr
            )  # alias for `deallc_mbar.arrive_wait + cute.arch.dealloc_tmem`

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute WarpGroups
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            # --- Increase rmem usage ---

            cute.arch.setmaxregister_increase(self.num_regs_compute)

            # --- Wait and retrieve tmem buffer ---

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # --- Enter compute loop ---

            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                sdS,
                sdS_xchg,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                dS_cluster_full_mbar_ptr,
                dQacc_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                tile_scheduler,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                aux_tensors,
                fastdiv_mods,
                blocksparse_tensors,
                is_print_block=is_print_block,
            )

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Reduce WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            # --- Increase rmem usage ---

            cute.arch.setmaxregister_increase(self.num_regs_reduce)

            # --- Wait and retrieve tmem buffer ---

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            # --- Enter reduce loop ---

            self.dQacc_reduce(
                mdQacc,
                sdQacc,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQacc_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                tile_scheduler,
                mdQ_semaphore,
                blocksparse_tensors,
                is_print_block=is_print_block,
            )

            # --- Arrive mma warp's tmem dealloc ---

            tmem_alloc_barrier.arrive()

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
    ):
        """Relay warp (2-CTA only): forward the peer CTA's dS-ready signal to the
        leader CTA's MMA warp.

        In 2-CTA mode, each tileQ is split in half across the cluster and the two
        CTAs each compute their own half of ``dS`` (see ``sdS_xchg`` with shape
        ``(tileK, tileM // 2)``). The ``dK += dS.T @ Q`` step uses a 2-CTA tcgen05
        UMMA whose accumulation runs along the cluster-wide tileK dim, so the
        leader CTA (rank 0) cannot issue the MMA until *both* CTAs' dS halves are
        present in its smem.

        The leader's MMA warp blocks on ``dS_cluster_leader_mbar``, which is
        initialized with an arrival count of 2 (one per dS half). This relay warp
        bridges the cross-CTA handshake: it waits on the local
        ``dS_cluster_full_mbar`` (signalled once the peer's dS half has been
        exchanged into place) and then performs a single cross-CTA arrive on the
        leader's ``dS_cluster_leader_mbar`` (``peer_cta_rank_in_cluster=0``),
        contributing the peer's "half ready" count.

        Why a dedicated warp:
          - The cross-CTA mbarrier arrive needs a worker to bridge
            "local completion -> notify remote leader".
          - Pulling this blocking/forwarding off the compute and MMA warps avoids
            serializing them on cross-CTA synchronization.
          - It is extremely lightweight (registers dropped via
            ``setmaxregister_decrease``); in 1-CTA mode this warp is skipped
            entirely (guarded by ``const_expr(self.use_2cta_instrs)``).

        It runs the same persistent tile-scheduler loop as the other warps,
        forwarding one signal per ``m_block`` iteration within each tile.
        """
        dS_cluster_phase = Int32(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, _, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block // self.cluster_shape_mnk[0]
            )

            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )
            if process_tile:
                num_iters = m_block_max - m_block_min
                for _ in cutlass.range(num_iters, unroll=1):
                    # Wait for sdS_xchg to be full from peer CTA
                    cute.arch.mbarrier_wait(
                        dS_cluster_full_mbar_ptr, phase=dS_cluster_phase
                    )

                    # Arrive the mma warp of the leader CTA
                    # to notify it that both half of sdS from both CTAs
                    # are ready and can issue dQ GEMM
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            dS_cluster_leader_mbar_ptr,
                            peer_cta_rank_in_cluster=Int32(0),
                        )

                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        pipeline_Q: ffa_pipeline.PipelineTmaUmma,
        pipeline_Qt: ffa_pipeline.PipelineTmaUmma,
        pipeline_Kt: ffa_pipeline.PipelineTmaUmma,
        pipeline_dO: ffa_pipeline.PipelineTmaUmma,
        pipeline_LSE: pipeline.PipelineTmaAsync,
        pipeline_dPsum: pipeline.PipelineTmaAsync,
        cta_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
        is_print_block: bool = False,
    ):
        tidx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        copy_stats_fn = partial(cute.copy, copy_atom_stats)
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_vmnk, (0, None, 0, 0)).shape
        )

        # --- Init producer pipeline states ---

        producer_state_Q_LSE = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Qt = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_Q_Qt = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_O_Ot = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dPsum = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # --- Compute TMA multicast mask for Q/dO ---

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead
            n_block_cta_group = n_block // self.cta_group_size

            # //////////////////////////////////////////////
            #  Make gQ/gK/gV/gdO/gLSE/gdPsum
            # //////////////////////////////////////////////

            # mQ_cur: (seqQ,HD):(1@1,1@0)
            # mK_cur: (seqK,HD):(1@1,1@0)
            # mV_cur: (seqK,HD):(1@1,1@0)
            # mdO_cur: (HD,seqQ):(1@0,1@1) => actually dO.T
            mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mK_cur = seqlen_info.offset_batch_K(mK, batch_idx, dim=3)[
                None, None, head_idx_kv
            ]
            mV_cur = seqlen_info.offset_batch_K(mV, batch_idx, dim=3)[
                None, None, head_idx_kv
            ]
            if const_expr(not seqlen_info.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset(
                    (0, seqlen_info.offset_q), mdO[None, None, head_idx]
                )
            # gQ: (tileQ128,tileHD128,restQ):(1@1,1@0,128@1)
            # gK: (tileK128*CTA2,tileHD128):(1@1,1@0)
            # gV: (tileK128*CTA2,tileHD128):(1@1,1@0)
            # gdO: (tileHD128,tileQ128,restQ):(1@0,1@1,128@1) => actually dO.T
            # where: restQ = seqQ // tileQ
            gQ = cute.local_tile(
                mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0)
            )
            gK = cute.local_tile(
                mK_cur,
                cute.select(self.mma_tiler_kq, mode=[0, 2]),
                (n_block_cta_group, 0),
            )
            gV = cute.local_tile(
                mV_cur,
                cute.select(self.mma_tiler_vdo, mode=[0, 2]),
                (n_block_cta_group, 0),
            )
            gdO = cute.local_tile(
                mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None)
            )

            # mLSE_cur: (seqQ):(1)
            # mdPsum_cur: (seqQ):(1)
            mLSE_cur = seqlen_info.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            mdPsum_cur = seqlen_info.offset_batch_Q(
                mdPsum, batch_idx, dim=2, padded=True
            )[None, head_idx]
            # gLSE: (tileQ128,restQ):(1,128)
            # gdPsum: (tileQ128,restQ):(1,128)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

            # mQt_cur: (HD,seqQ):(1@0,1@1)
            # mKt_cur: (HD,seqK):(1@0,1@1)
            # mdOt_cur: (seqQ,HD):(1@1,1@0) => actually dO
            if const_expr(self.use_2cta_instrs):
                if const_expr(not seqlen_info.has_cu_seqlens_q):
                    mQt_cur = mQt[None, None, head_idx, batch_idx]
                    mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                else:
                    mQt_cur = cute.domain_offset((0, seqlen_info.offset_q, 0), mQt)[
                        None, None, head_idx
                    ]
                    mdOt_cur = cute.domain_offset((seqlen_info.offset_q, 0, 0), mdOt)[
                        None, None, head_idx
                    ]
                if const_expr(not seqlen_info.has_cu_seqlens_k):
                    mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
                else:
                    mKt_cur = cute.domain_offset((0, seqlen_info.offset_k, 0), mKt)[
                        None, None, head_idx_kv
                    ]

            # gQt: (tileHD128,tileQ128,restQ):(1@0,1@1,128@1)
            # gKt: (tileHD128,tileK128*CTA2):(1@0,1@1)
            # gdOt: (tileQ128,tileHD128,restQ):(1@1,1@0,128@1) => actually dO
            gQt = None
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(
                    mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                )
            gKt = None
            if const_expr(self.use_2cta_instrs):
                gKt = cute.local_tile(
                    mKt_cur,
                    cute.select(self.mma_tiler_dsk, mode=[1, 2]),
                    (0, n_block_cta_group),
                )
            gdOt = None
            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(
                    mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                )

            # //////////////////////////////////////////////
            #  TMA Partition gQ/gK/gV/gdO and
            #  define G2S-load fn for sQ/sK/sV/sdO
            # //////////////////////////////////////////////

            # S.T = K @ Q.T => load sK/sQ
            # tSgK: (MMA_sA=(128,16),MMA_K1,MMA_HD8):((1@1,1@0),0,16@0)
            # tSgQ: (MMA_sB=(64,16),MMA_Q1,MMA_HD8,restQ):((1@1,1@0),0,16@0,128@1)
            tSgK = thr_mma_S.partition_A(gK)
            tSgQ = thr_mma_S.partition_B(gQ)
            # tKgK: (TMA_ATOM=((64,128),2)):(((1@0,1@1),64@0))
            # tKsK: (TMA_ATOM=((8192,2))):((1,8192))
            load_K, tKsK, tKgK = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                cta_coord=block_in_cluster_coord_vmnk[2],
                cta_layout=a_cta_layout,
                src_tensor=tSgK,
                dst_tensor=sK,
                single_stage=True,  # remove the rest/stage dim
            )
            # tQgQ: (TMA_ATOM=((64,64),2),restQ):(((1@0,1@1),64@0),128@1)
            # tQsQ: (TMA_ATOM=(4096,2),stageQ):((1,4096),0)
            load_Q, tQsQ, tQgQ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # dP = V @ dO.T => load sV/sdOt
            # tdPgV: ((128,16),1,8):((1@1,1@0),0,16@0)
            # tVgV: (((64,128),2)):(((1@0,1@1),64@0))
            # tVsV: ((8192,1),6):((1,0),8192)
            tdPgV = thr_mma_dP.partition_A(gV)
            load_V, tVsV, tVgV = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                src_tensor=tdPgV,
                dst_tensor=sV,
                single_stage=True,
            )

            # tdPgdOt: (MMA_sB=(64,16),MMA_Q1,MMA_HD8,restQ):((1@1,1@0),0,16@0,128@1)
            # tdOgdOt: (TMA_ATOM=((64,64),2),restQ):(((1@0,1@1),64@0),128@1)
            # tdOsdOt: (TMA_ATOM=(4096,2),stageQ):((1,4096),0)
            if const_expr(tma_atom_dOt is not None):
                tdPgdOt = thr_mma_dP.partition_B(gdOt)
                load_dOt, tdOsdOt, tdOgdOt = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdOt,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # dV += P.T @ dO => load sdO
            # tdVgdO: (MMA_sB=(64,16),tileHD1,tileQ8,restQ):((1@0,1@1),0,16@1,128@1)
            # tdOgdO: (TMA_ATOM=((64,128),1),restQ):(((1@0,1@1),0),128@1)
            # tdOsdO: (TMA_ATOM=(8192,1),stageQ):((1,0),0)
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, tdOsdO, tdOgdO = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # dK += dS.T @ Q => sQt
            # NOTE: in 2-CTA mode, we need separate Qt load
            # tdKgQt: (MMA_sB=(64,16),tileHD1,tileQ8,restQ):((1@0,1@1),0,16@1,128@1)
            # tdQgQt: (TMA_ATOM=((64,128),1),restQ):(((1@0,1@1),0),128@1)
            # tdQsQt: (TMA_ATOM=(8192,1),stageQ):((1,0),0)
            if const_expr(tma_atom_Qt is not None):
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, tdQsQt, tdQgQt = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # dQ = dS @ K => sKt
            # tdQgKt: (MMA_sB=(64,16),tileHD1,tileQ8,restQ):((1@0,1@1),0,16@1)
            # tdKgKt: (TMA_ATOM=((64,256),1),restQ):(((1@0,1@1),0))
            # tdKsKt: (TMA_ATOM=(16384,1),stageQ):((1,0))
            if const_expr(self.use_2cta_instrs):
                tdQgKt = thr_mma_dQ.partition_B(gKt)
                load_Kt, tdKsKt, tdKgKt = copy_utils.tma_get_copy_fn(
                    tma_atom_Kt,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    tdQgKt,
                    sKt,
                    single_stage=True,
                )

            # --- Debug print ---

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm100_load] "

                    cute.printf("")
                    cute.printf(
                        prefix + "Q_stage={} dO_stage={} single_stage={}",
                        self.Q_stage,
                        self.dO_stage,
                        self.single_stage,
                    )
                    cute.printf("")

                    # --- gmem source tensors (mX_cur) ---

                    cute.printf("")
                    cute.printf(prefix + "mQ_cur.layout: {}", mQ_cur.layout)
                    cute.printf(prefix + "mK_cur.layout: {}", mK_cur.layout)
                    cute.printf(prefix + "mV_cur.layout: {}", mV_cur.layout)
                    cute.printf(prefix + "mdO_cur.layout: {}", mdO_cur.layout)
                    cute.printf(prefix + "mLSE_cur.layout: {}", mLSE_cur.layout)
                    cute.printf(prefix + "mdPsum_cur.layout: {}", mdPsum_cur.layout)
                    if const_expr(self.use_2cta_instrs):
                        cute.printf(prefix + "mQt_cur.layout: {}", mQt_cur.layout)
                        cute.printf(prefix + "mdOt_cur.layout: {}", mdOt_cur.layout)
                        cute.printf(prefix + "mKt_cur.layout: {}", mKt_cur.layout)
                    cute.printf("")

                    # --- tiled gmem tensors (gX) ---

                    cute.printf("")
                    cute.printf(prefix + "gQ.layout: {}", gQ.layout)
                    cute.printf(prefix + "gK.layout: {}", gK.layout)
                    cute.printf(prefix + "gV.layout: {}", gV.layout)
                    cute.printf(prefix + "gdO.layout: {}", gdO.layout)
                    cute.printf(prefix + "gLSE.layout: {}", gLSE.layout)
                    cute.printf(prefix + "gdPsum.layout: {}", gdPsum.layout)
                    if const_expr(tma_atom_dOt is not None):
                        cute.printf(prefix + "gdOt.layout: {}", gdOt.layout)
                    if const_expr(tma_atom_Qt is not None):
                        cute.printf(prefix + "gQt.layout: {}", gQt.layout)
                    if const_expr(self.use_2cta_instrs):
                        cute.printf(prefix + "gKt.layout: {}", gKt.layout)
                    cute.printf("")

                    # --- mma-partitioned gmem tensors (tYgX) ---

                    cute.printf("")
                    cute.printf(prefix + "tSgK.layout: {}", tSgK.layout)
                    cute.printf(prefix + "tSgQ.layout: {}", tSgQ.layout)
                    cute.printf(prefix + "tdPgV.layout: {}", tdPgV.layout)
                    if const_expr(tma_atom_dOt is not None):
                        cute.printf(prefix + "tdPgdOt.layout: {}", tdPgdOt.layout)
                    cute.printf(prefix + "tdVgdO.layout: {}", tdVgdO.layout)
                    if const_expr(tma_atom_Qt is not None):
                        cute.printf(prefix + "tdKgQt.layout: {}", tdKgQt.layout)
                    if const_expr(self.use_2cta_instrs):
                        cute.printf(prefix + "tdQgKt.layout: {}", tdQgKt.layout)
                    cute.printf("")

                    # --- tma-partitioned smem/gmem tensors (tXsX/tXgX) ---

                    cute.printf("")
                    cute.printf(prefix + "tKsK.layout: {}", tKsK.layout)
                    cute.printf(prefix + "tKgK.layout: {}", tKgK.layout)
                    cute.printf(prefix + "tQsQ.layout: {}", tQsQ.layout)
                    cute.printf(prefix + "tQgQ.layout: {}", tQgQ.layout)
                    cute.printf(prefix + "tVsV.layout: {}", tVsV.layout)
                    cute.printf(prefix + "tVgV.layout: {}", tVgV.layout)
                    if const_expr(tma_atom_dOt is not None):
                        cute.printf(prefix + "tdOsdOt.layout: {}", tdOsdOt.layout)
                        cute.printf(prefix + "tdOgdOt.layout: {}", tdOgdOt.layout)
                    if const_expr(tma_atom_dO is not None):
                        cute.printf(prefix + "tdOsdO.layout: {}", tdOsdO.layout)
                        cute.printf(prefix + "tdOgdO.layout: {}", tdOgdO.layout)
                    if const_expr(tma_atom_Qt is not None):
                        cute.printf(prefix + "tdQsQt.layout: {}", tdQsQt.layout)
                        cute.printf(prefix + "tdQgQt.layout: {}", tdQgQt.layout)
                    if const_expr(self.use_2cta_instrs):
                        cute.printf(prefix + "tdKsKt.layout: {}", tdKsKt.layout)
                        cute.printf(prefix + "tdKgKt.layout: {}", tdKgKt.layout)
                    cute.printf("")

                    # --- smem dest tensors (sX) ---
                    cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                    cute.printf(prefix + "sQt.layout: {}", sQt.layout)
                    cute.printf(prefix + "sK.layout: {}", sK.layout)
                    cute.printf(prefix + "sV.layout: {}", sV.layout)
                    cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                    cute.printf(prefix + "sdOt.layout: {}", sdOt.layout)
                    cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                    cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                    cute.printf("")

            # //////////////////////////////////////////////
            #  G2S-load sQ/sK/sV/sdO/sLSE/sdPsum
            # //////////////////////////////////////////////

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                # NOTE: some tiles might be empty due to block sparsity
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)
            else:
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )

            if process_tile:
                if const_expr(self.use_block_sparsity):  # TODO: review the logics
                    (
                        producer_state_Q_LSE,
                        producer_state_dO_dPsum,
                    ) = produce_block_sparse_q_loads_bwd_sm100(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        producer_state_Q_LSE,
                        producer_state_dO_dPsum,
                        pipeline_Q,
                        pipeline_LSE,
                        pipeline_dO,
                        pipeline_dPsum,
                        load_K,
                        load_V,
                        load_Q,
                        load_dO,
                        copy_stats_fn,
                        gLSE,
                        sLSE,
                        gdPsum,
                        sdPsum,
                        self.tma_copy_bytes["K"],
                        self.tma_copy_bytes["V"],
                        should_load_Q=should_load_Q,
                        should_load_dO=should_load_dO,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                else:
                    first_m_block = m_block_min

                    # TODO: review the logics
                    if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                        # Prologue
                        assert should_load_Q and should_load_dO
                        # K & Q (for S)
                        pipeline_Q.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_K(
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q_Qt
                            )
                        )
                        load_Q(first_m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Q.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()
                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_LSE)
                        with cute.arch.elect_one():
                            copy_stats_fn(
                                gLSE[None, first_m_block],
                                sLSE[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(
                                    producer_state_LSE
                                ),
                            )
                        producer_state_LSE.advance()

                        # dOt + V, for dP.T = V @ dO.T
                        pipeline_dO.producer_acquire(
                            producer_state_O_Ot,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_O_Ot
                            )
                        )
                        load_dOt(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()
                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dPsum)
                        with cute.arch.elect_one():
                            copy_stats_fn(
                                gdPsum[None, first_m_block],
                                sdPsum[None, producer_state_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dPsum
                                ),
                            )
                        producer_state_dPsum.advance()

                        # Qt, for dK = dS.T @ Q
                        pipeline_Qt.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_Qt(first_m_block, producer_state=producer_state_Q_Qt)
                        load_Kt(
                            tma_bar_ptr=pipeline_Qt.producer_get_barrier(
                                producer_state_Q_Qt
                            )
                        )
                        pipeline_Qt.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dO, for dV = P.T @ dO
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dO(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                        # Mainloop
                        # 2CTA: [lse | Q | dOt | dPsum | Qt | dO]
                        for m_block in cutlass.range(
                            m_block_min + 1, m_block_max, unroll=1
                        ):
                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            with cute.arch.elect_one():
                                copy_stats_fn(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_LSE
                                    ),
                                )
                            producer_state_LSE.advance()

                            # Q
                            pipeline_Q.producer_acquire(producer_state_Q_Qt)
                            load_Q(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Q.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dPsum)
                            with cute.arch.elect_one():
                                copy_stats_fn(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dPsum
                                    ),
                                )
                            producer_state_dPsum.advance()

                            # dOt, for dP.T = V @ dO.T
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dOt(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                            load_Qt(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Qt.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dO, for dV = P.T @ dO
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dO(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()
                    else:
                        # --- Prologue: load K,V,Kt/Q0/dO0,dOt0/LSE0,dPsum0 ---

                        # Load Q0,K,LSE0
                        if const_expr(should_load_Q):
                            # Load Q0,K
                            pipeline_Q.producer_acquire(
                                producer_state_Q_LSE,
                                # expect sQ + sK
                                extra_tx_count=self.tma_copy_bytes["K"],
                            )
                            load_K(
                                tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                    producer_state_Q_LSE
                                )
                            )
                            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # Load LSE0
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            with cute.arch.elect_one():
                                copy_stats_fn(
                                    gLSE[None, first_m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                            producer_state_Q_LSE.advance()

                        # Load V, dO0, dOt0, dPsum0
                        if const_expr(should_load_dO):
                            # Load V, dO0, dOt0
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                # expect sV + sdO (+ sdOt)
                                extra_tx_count=self.tma_copy_bytes["V"]
                                + self.tma_copy_bytes["dO"]
                                if const_expr(tma_atom_dOt is not None)
                                else self.tma_copy_bytes["V"],
                            )
                            load_V(
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                )
                            )
                            load_dO(
                                first_m_block, producer_state=producer_state_dO_dPsum
                            )
                            if const_expr(tma_atom_dOt is not None):
                                load_dOt(
                                    first_m_block,
                                    producer_state=producer_state_dO_dPsum,
                                )
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # Load dPsum0
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            with cute.arch.elect_one():
                                copy_stats_fn(
                                    gdPsum[None, first_m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                            producer_state_dO_dPsum.advance()

                        # Load Kt
                        if const_expr(self.use_2cta_instrs):
                            pipeline_Kt.producer_acquire(producer_state_Kt)
                            load_Kt(
                                tma_bar_ptr=pipeline_Kt.producer_get_barrier(
                                    producer_state_Kt
                                )
                            )
                            pipeline_Kt.producer_commit(producer_state_Kt)
                            producer_state_Kt.advance()

                        # --- Mainloop: load Q(i),Qt(i-1)/dO(i),dOt(i)/LSE(i),dPsum(i) ---

                        for m_block in cutlass.range(
                            m_block_min + 1, m_block_max, unroll=1
                        ):
                            # Load Qt(i-1), Q(i), LSE(i)
                            if const_expr(should_load_Q):
                                # Load Qt(i-1)
                                if const_expr(tma_atom_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(
                                        m_block - 1, producer_state=producer_state_Qt
                                    )
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                # Load Q(i)
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                load_Q(m_block, producer_state=producer_state_Q_LSE)
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # Load LSE(i)
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                with cute.arch.elect_one():
                                    copy_stats_fn(
                                        gLSE[None, m_block],
                                        sLSE[None, producer_state_Q_LSE.index],
                                        mbar_ptr=pipeline_LSE.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                    )
                                producer_state_Q_LSE.advance()

                            # Load dO(i), dOt(i), dPsum(i)
                            if const_expr(should_load_dO):
                                # Load dO(i), dOt(i)
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    # expect sdO (+ sdOt)
                                    extra_tx_count=self.tma_copy_bytes["dO"]
                                    if const_expr(tma_atom_dOt is not None)
                                    else 0,
                                )
                                load_dO(m_block, producer_state=producer_state_dO_dPsum)
                                if const_expr(tma_atom_dOt is not None):
                                    load_dOt(
                                        m_block, producer_state=producer_state_dO_dPsum
                                    )
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # Load dPsum(i)
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                with cute.arch.elect_one():
                                    copy_stats_fn(
                                        gdPsum[None, m_block],
                                        sdPsum[None, producer_state_dO_dPsum.index],
                                        mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                    )
                                producer_state_dO_dPsum.advance()

                        # --- Epilogue: load Qt(-1) ---

                        # Load Qt(-1)
                        if const_expr(should_load_Q):
                            if const_expr(tma_atom_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(
                                    m_block_max - 1, producer_state=producer_state_Qt
                                )
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                # --- Producer tail ---

                if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                    pipeline_Q.producer_tail(producer_state_Q_Qt)
                    pipeline_LSE.producer_tail(producer_state_LSE)
                    pipeline_dO.producer_tail(producer_state_O_Ot)
                    pipeline_dPsum.producer_tail(producer_state_dPsum)
                else:
                    if const_expr(should_load_Q):
                        pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                        if const_expr(tma_atom_Qt is not None):
                            pipeline_Qt.producer_tail(producer_state_Qt)
                    if const_expr(should_load_dO):
                        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            # Advance to next KV tile
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

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
        tP: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_Q: ffa_pipeline.PipelineTmaUmma,
        pipeline_Qt: ffa_pipeline.PipelineTmaUmma,
        pipeline_Kt: ffa_pipeline.PipelineTmaUmma,
        pipeline_dO: ffa_pipeline.PipelineTmaUmma,
        pipeline_S_P: pipeline.PipelineUmmaAsync,
        pipeline_dS: pipeline.PipelineAsyncUmma,
        pipeline_dKV: pipeline.PipelineUmmaAsync,
        pipeline_dP: pipeline.PipelineUmmaAsync,
        pipeline_dQ: pipeline.PipelineUmmaAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        is_leader_cta: cutlass.Boolean,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        is_print_block: bool = False,
    ):
        cta_group = pipeline_S_P.cta_group

        # --- Make GEMM fragments & define GEMM funcs ---

        # FIXME: For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower than putting them here.
        # Partition smem / tmem tensors

        # --- S.T = K @ Q.T with (K, K) major ---

        # tSrK: (MMA_ATOM1,MMA_K1,MMA_HD=(4,2)):(0,0,(2,1024))
        # tSrQ: (MMA_ATOM1,MMA_Q1,MMA_HD=(4,2),stageQ):(0,0,(2,512),0)
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        mma_s_qk_fn = partial(
            sm100_utils.gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=self.cta_group_size,
        )

        # --- dP.T = V @ dO.T with (K, K) major ---

        # tdPrV: (MMA_ATOM1,MMA_K1,MMA_HD=(4,2)):(0,0,(2,1024))
        # tdPrdOt: (MMA_ATOM1,MMA_Q1,MMA_HD=(4,2),stageQ):(0,0,(2,512),0)
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        mma_dp_vdo_fn = partial(
            sm100_utils.gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )

        # --- dK = dS.T @ Q with (K, MN) major ---

        # tdKrdS: (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((131072,1),0,(16,64))
        # tdKrQ: (MMA_ATOM1,MMA_Q1,MMA_HD8,stageQ):(0,0,128,0)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        if const_expr(self.use_smem_dS_for_mma_dK and not self.use_2cta_instrs):
            # NOTE: For 2-CTA, dS (dK mma) MUST come from TMEM (cannot use SMEM)
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)  # From SMEM
            mma_dk_dsq_fn = partial(
                sm100_utils.gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ
            )
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM
            # Need to explicitly pass in tA_addr for correctness
            mma_dk_dsq_fn = partial(
                sm100_utils.gemm_ptx_w_idx,
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                sA=None,
                sB=sQt,
                tA_addr=self.tmem_dS_offset,
                cta_group=self.cta_group_size,
            )

        # --- dQ = dS @ K with (MN, MN) major ---

        # tdQrdS: (MMA_ATOM1,MMA_K1,MMA_Q16):(0,0,128)
        # tdQrK: (MMA_ATOM1,MMA_K1,MMA_HD16):(0,0,128)
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        mma_dq_dsk_fn = partial(
            sm100_utils.gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=2 if const_expr(self.use_2cta_instrs) else 1,
        )

        # --- dV = P.T @ dO with (K, MN) major ---

        # tdVrP: (MMA_tA=(128,16),MMA_K1,MMA_Q=(4,2)):((131072,1),0,(16,64))
        # tdVrdO: (MMA_ATOM1,MMA_Q1,MMA_HD=8,stageQ):(0,0,128,0)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        mma_dv_pdo_fn = partial(
            sm100_utils.gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
            cta_group=self.cta_group_size,
        )

        # --- Init pipeline states and phases ---

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        consumer_state_Qt = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Q = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        consumer_state_dS = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        dS_cluster_phase = Int32(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block // self.cluster_shape_mnk[0]
            )

            if const_expr(self.use_block_sparsity):  # TODO: review the logics
                block_iter_count = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                block_iter_count = m_block_max - m_block_min
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )

            # --- Debug print ---

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm100_mma] "
                    cute.printf("")
                    cute.printf(
                        prefix + "tile_m={} tile_n={} tile_hdim={} tile_hdimv={}",
                        self.tile_m,
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                    )
                    cute.printf("")
                    cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                    cute.printf(prefix + "sK.layout: {}", sK.layout)
                    cute.printf(prefix + "sV.layout: {}", sV.layout)
                    cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                    cute.printf(prefix + "sdS.layout: {}", sdS.layout)
                    cute.printf(prefix + "sdSt.layout: {}", sdSt.layout)
                    cute.printf(prefix + "tP.layout: {}", tP.layout)
                    cute.printf(prefix + "tdS.layout: {}", tdS.layout)
                    cute.printf("")
                    cute.printf(prefix + "tStS.layout: {}", tStS.layout)
                    cute.printf(prefix + "tdPtdP.layout: {}", tdPtdP.layout)
                    cute.printf(prefix + "tdVtdV.layout: {}", tdVtdV.layout)
                    cute.printf(prefix + "tdKtdK.layout: {}", tdKtdK.layout)
                    cute.printf(prefix + "tdQtdQ.layout: {}", tdQtdQ.layout)
                    cute.printf("")
                    cute.printf(prefix + "tSrK.layout: {}", tSrK.layout)
                    cute.printf(prefix + "tSrQ.layout: {}", tSrQ.layout)
                    cute.printf(prefix + "tdPrV.layout: {}", tdPrV.layout)
                    cute.printf(prefix + "tdPrdOt.layout: {}", tdPrdOt.layout)
                    cute.printf(prefix + "tdKrdS.layout: {}", tdKrdS.layout)
                    cute.printf(prefix + "tdKrQ.layout: {}", tdKrQ.layout)
                    cute.printf(prefix + "tdQrdS.layout: {}", tdQrdS.layout)
                    cute.printf(prefix + "tdQrK.layout: {}", tdQrK.layout)
                    cute.printf(prefix + "tdVrdO.layout: {}", tdVrdO.layout)
                    cute.printf(prefix + "tdVrP.layout: {}", tdVrP.layout)
                    cute.printf("")

            # TODO: review the logics
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    accumulate_dV = False

                    # -----------------------------------------------------------
                    # MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dP.T = V    @ dO.T
                    # 3. dK   = dS.T @ Q
                    # 4. dV   = P.T  @ dO
                    # 5. dQ   = dS   @ K

                    main_loop_iters = m_block_max - m_block_min

                    # empty waits
                    # pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dQ tmem overlaps with S
                        mma_s_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_S_P.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP tmem overlaps with S
                        mma_dp_vdo_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP -> dS
                        mma_dk_dsq_fn(
                            B_idx=consumer_state_Q.index, zero_init=not accumulate_dK
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_dv_pdo_fn(
                            B_idx=consumer_state_dO.index, zero_init=not accumulate_dV
                        )
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(
                            dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase
                        )
                        mma_dq_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # Commit tdV/tdK to be full
                    # to notify the epilogue to store them to gmem
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(
                        0, pipeline_dKV.producer_mask, cta_group
                    )
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(
                        1, pipeline_dKV.producer_mask, cta_group
                    )
                    producer_phase_dKV ^= 1
            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and process_tile:
                    # //////////////////////////////////////////////
                    #  Prologue: GEMM for S(0),dP(0),dV(0)
                    # //////////////////////////////////////////////

                    accumulate_dK = False

                    # --- GEMM: S(0).T = K @ Q(0).T ---

                    # Wait for K/sQ(0) to be full
                    pipeline_Q.consumer_wait(consumer_state_Q)

                    # Acquire for tS(0) to be empty
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tS(0)
                    mma_s_qk_fn(B_idx=consumer_state_Q.index)

                    # Commit tS(0) to be full
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # Release sQ(0) to be empty
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # --- GEMM: dP(0).T = V @ dO(0).T ---

                    # Wait for V/sdO(0)/sdOt(0) to be full
                    pipeline_dO.consumer_wait(consumer_state_dO)

                    # Acquire tdP(0) to be empty
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tdP(0)
                    mma_dp_vdo_fn(B_idx=consumer_state_dO.index)

                    # Commit tdP(0) to be full
                    pipeline_dP.sync_object_full.arrive(
                        0, pipeline_dP.producer_mask, cta_group
                    )
                    producer_phase_acc ^= 1

                    # --- GEMM: dV(0) = P(0).T @ dO(0) ---

                    # Wait for tP(0) to be full
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tdV(0)
                    mma_dv_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)

                    # Release sdO(0) to be empty
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    # Wait for sKt to be full
                    pipeline_Kt.consumer_wait(consumer_state_Kt)

                    # //////////////////////////////////////////////
                    #  Mainloop: GEMM for S(i),dK(i-1),dP(i),dQ(i-1),dV(i)
                    # //////////////////////////////////////////////

                    main_loop_iters = (
                        block_iter_count
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min
                    )

                    for i in cutlass.range(1, main_loop_iters, unroll=1):
                        # --- GEMM: S(i).T = K @ Q(i).T ---

                        # Wait for sQ(i) to be full
                        pipeline_Q.consumer_wait(consumer_state_Q)

                        # Acquire tdQ(i-1) to be empty
                        # since tdQ is embedded in right-half of tdS
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)

                        # Issue UMMA for tS(i)
                        # NOTE: we don't need to wait for tS(i) to be empty
                        # since we've read tP(i-1) for dV(i) GEMM in last iter
                        mma_s_qk_fn(B_idx=consumer_state_Q.index)

                        # Commit tS(i) to be full
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                        # Release sQ(i) to be empty
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # --- GEMM: dK(i-1) = dS(i-1).T @ Q(i-1) ---

                        # Wait for sQt(i-1) to be full
                        pipeline_Qt.consumer_wait(consumer_state_Qt)

                        # Wait for dS(i-1) to be full
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                        # Issue UMMA for tdK(i-1)
                        mma_dk_dsq_fn(
                            B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK
                        )

                        # Release sQt(i-1) to be empty
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        # --- GEMM: dP(i) = V @ dO(i).T ---

                        # Wait for sdO(i)/sdOt(i) to be full
                        pipeline_dO.consumer_wait(consumer_state_dO)

                        # Issue UMMA for tdP(i)
                        mma_dp_vdo_fn(B_idx=consumer_state_dO.index)

                        # Commit tdP(i) to be full
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )

                        # --- GEMM: dQ(i-1) = dS(i-1) @ K ---

                        # Wait for tdSt(i-1) from both CTAs to be full
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(
                            dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase
                        )

                        # Issue UMMA for tdQ(i-1)
                        mma_dq_dsk_fn()

                        # Commit tdQ(i-1) to be full
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )

                        # Release tdSt(i-1) to be empty
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1
                        producer_phase_acc ^= 1

                        # --- GEMM: dV(i) = P(i).T @ dO(i) ---

                        # Wait for tP(i) to be full
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                        # Issue UMMA for tdV(i)
                        mma_dv_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)

                        # Release sdO(i) to be empty
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    # //////////////////////////////////////////////
                    #  Epilogue: GEMM for dK(-1),dQ(-1)
                    # //////////////////////////////////////////////

                    # --- Commit dV is ready ---

                    # Release tP(-1) to be empty
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # Acquire dV buffer to be empty
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)

                    # Commit dV(-1) to be full
                    # to notify the compute wgs to write dV back to gmem
                    pipeline_dKV.sync_object_full.arrive(
                        0, pipeline_dKV.producer_mask, cta_group
                    )

                    # --- GEMM: dK(-1) = dS(-1).T @ Q(-1) ---
                    # --- Commit dK is ready ---

                    # Acquire dK buffer to be empty
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # Wait for sQt(-1) to be full
                    pipeline_Qt.consumer_wait(consumer_state_Qt)

                    # Wait for dS(-1) to be full
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tdK(-1)
                    mma_dk_dsq_fn(
                        B_idx=consumer_state_Qt.index, zero_init=not accumulate_dK
                    )

                    # Release sQt(-1) to be empty
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()

                    # Commit dK to be full
                    # to notify the compute wgs to write dK back to gmem
                    pipeline_dKV.sync_object_full.arrive(
                        1, pipeline_dKV.producer_mask, cta_group
                    )
                    producer_phase_dKV ^= 1

                    # --- GEMM: dQ(-1) = dS(-1) @ K ---

                    # Wait for dSt(-1) to be full
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(
                        dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase
                    )

                    # Acquire tdQ(-1) to be empty
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)

                    # Issue UMMA for tdQ(-1)
                    mma_dq_dsk_fn()

                    # Commit tdQ(-1) to be full
                    pipeline_dQ.sync_object_full.arrive(
                        0, pipeline_dQ.producer_mask, cta_group
                    )

                    # Release tdSt(-1) to be empty
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # Release sKt to be empty
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_Kt.advance()

                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1
                    producer_phase_acc ^= 1
            else:
                if is_leader_cta and process_tile:
                    # //////////////////////////////////////////////
                    #  Prologue: GEMM for S(0),dP(0),dV(0)
                    # //////////////////////////////////////////////

                    accumulate_dK = False

                    # --- GEMM: S(0).T = K @ Q(0).T ---

                    # Wait for K/sQ(0) to be full
                    handle_Q = pipeline_Q_consumer.wait_and_advance()

                    # Acquire for tS(0) to be empty
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tS(0)
                    mma_s_qk_fn(B_idx=handle_Q.index)

                    # Commit tS(0) to be full
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # NOTE: we don't release sQ(0) until dK(0) GEMM is done
                    # for the first iter in the mainloop

                    # --- GEMM: dP(0).T = V @ dO(0).T ---

                    # Wait for V/sdO(0) to be full
                    pipeline_dO.consumer_wait(consumer_state_dO)

                    # Acquire tdP(0) to be empty
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    # Acquire tdQ(0) to be empty
                    # prepared for dQ(0) GEMM in the mainloop
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tdP(0)
                    mma_dp_vdo_fn(B_idx=consumer_state_dO.index)

                    # Commit tdP(0) to be full
                    pipeline_dP.sync_object_full.arrive(
                        0, pipeline_dP.producer_mask, cta_group
                    )
                    producer_phase_acc ^= 1

                    # --- GEMM: dV(0) = P(0).T @ dO(0) ---

                    # Wait for tP(0) to be full
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                    # Issue UMMA for tdV(0)
                    mma_dv_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)

                    # Release sdO(0) to be empty
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    # //////////////////////////////////////////////
                    #  Mainloop: GEMM for S(i),dK(i-1),dQ(i-1),dP(i),dV(i)
                    # //////////////////////////////////////////////

                    # NOTE: For block sparsity, we use block_iter_count;
                    # for dense, use m_block range, MMA doesn't need actual m_block indices,
                    # just the iteration count
                    main_loop_iters = (
                        block_iter_count
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min
                    )

                    handle_Q_next = handle_Q
                    for i in cutlass.range(1, main_loop_iters, unroll=1):
                        # --- GEMM: S(i).T = K @ Q(i).T ---

                        # Wait for sQ(i) to be full
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()

                        # Issue UMMA for tS(i)
                        # NOTE: we don't need to wait for tS(i) to be empty
                        # since we've read tP(i-1) for dV(i) GEMM in last iter
                        mma_s_qk_fn(B_idx=handle_Q_next.index)

                        # Commit tS(i) to be full
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                        # --- GEMM: dK(i-1) = dS(i-1).T @ Q(i-1) ---

                        # Wait for tdS(i-1) to be full
                        pipeline_dS.consumer_wait(consumer_state_dS)

                        # Issue UMMA for tdK(i-1)
                        # NOTE: we don't need to wait for sQ(i-1) to be full
                        # since we've read sQ(i-1) for S(i-1) GEMM
                        mma_dk_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)

                        # Release sQ(i-1) to be empty
                        accumulate_dK = True
                        handle_Q.release()

                        # --- GEMM: dQ(i-1) = dS(i-1) @ K ---

                        # Issue UMMA for tdQ(i-1)
                        # NOTE:
                        #   1. we don't need to wait for tdS(i-1) to be full
                        #       since we've read tdS(i-1) for dK(i-1) GEMM
                        #   2. we don't need to wait for tdQ(i-1) to be empty
                        #       since it's ready before dP(i-1) GEMM
                        mma_dq_dsk_fn()

                        # Commit tdQ(i-1) to be full
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )

                        # Release tdS(i-1) to be empty
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()

                        # --- GEMM: dP(i) = V @ dO(i).T ---

                        # Wait for sdO(i) to be full
                        pipeline_dO.consumer_wait(consumer_state_dO)

                        # Acquire tdQ(i-1) to be empty => tdP(i) to be empty
                        # NOTE: in 1-CTA mode, tdQ is overlapped with tdP
                        # so when tdQ(i-1) is consumed by dQacc warp,
                        # its tmem buffer is empty for tdP(i)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)

                        # Issue UMMA for tdP(i)
                        mma_dp_vdo_fn(B_idx=consumer_state_dO.index)

                        # Commit tdP(i) to be full
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )
                        producer_phase_acc ^= 1

                        # --- GEMM: dV(i) = P(i).T @ dO(i) ---

                        # Wait for tP(i) to be full
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)

                        # Issue UMMA for tdV(i)
                        mma_dv_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)

                        # Release sdO(i) to be empty
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        handle_Q = handle_Q_next

                    # //////////////////////////////////////////////
                    #  Epilogue: GEMM for dK(-1),dQ(-1)
                    # //////////////////////////////////////////////

                    # --- Commit dV is ready ---

                    # Release tP(-1) to be empty
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # Acquire dV buffer to be empty
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)

                    # Commit dV(-1) to be full
                    # to notify the compute wgs to write dV back to gmem
                    pipeline_dKV.sync_object_full.arrive(
                        0, pipeline_dKV.producer_mask, cta_group
                    )

                    # --- GEMM: dK(-1) = dS(-1).T @ Q(-1) ---
                    # --- Commit dK is ready ---

                    # Acquire dK buffer to be empty
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # Wait for tdS(-1) to be full
                    pipeline_dS.consumer_wait(consumer_state_dS)

                    # Issue UMMA for tdK(-1)
                    mma_dk_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)

                    # Commit dK to be full
                    # to notify the compute wgs to write dK back to gmem
                    pipeline_dKV.sync_object_full.arrive(
                        1, pipeline_dKV.producer_mask, cta_group
                    )
                    producer_phase_dKV ^= 1

                    # --- GEMM: dQ(-1) = dS(-1) @ K ---

                    # Issue UMMA for tdQ(-1)
                    mma_dq_dsk_fn()

                    # Commit dQ(-1) to be full
                    pipeline_dQ.sync_object_full.arrive(
                        0, pipeline_dQ.producer_mask, cta_group
                    )
                    handle_Q.release()

                    # Release tdS(-1) to be empty
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()
                    producer_phase_acc ^= 1

            # Advance to next KV tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        # --- Producer tail ---

        # FIXME: Currently it hangs if we have this S_P.producer_tail,
        # need to investigate why.

        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (
                    None,
                ) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (
                    None,
                ) * (rank - 4)
        return t[coord]

    @cute.jit
    def apply_score_mod_fwd(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply forward score modification for SM100 backward pass."""
        # In bwd, S is computed as K @ Q.T so dimensions are (tile_n, tile_m)
        cS = cute.make_identity_tensor((self.tile_n, self.tile_m))
        cS = cute.domain_offset((n_block * self.tile_n, m_block * self.tile_m), cS)
        tScS = thr_mma_S.partition_C(cS)
        tScS_idx = thr_copy_t2r.partition_D(tScS)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_idx,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        """Apply backward score modification (joint graph) for SM100."""
        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            index_tensor,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        sdS_xchg: cute.Tensor,
        pipeline_LSE: pipeline.PipelineTmaAsync,
        pipeline_dPsum: pipeline.PipelineTmaAsync,
        pipeline_S_P: pipeline.PipelineUmmaAsync,
        pipeline_dS: pipeline.PipelineAsyncUmma,
        pipeline_dKV: pipeline.PipelineUmmaAsync,
        pipeline_dP: pipeline.PipelineUmmaAsync,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQacc_empty_mbar_ptr: cute.Pointer,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        AttentionMaskCls: Callable[..., AttentionMask],
        tile_scheduler: TileSchedulerProtocol,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        is_print_block: bool = False,
    ):
        # --- Set up thread info ---

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        )
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        # --- Expand sLSE/sdPsum to 2D and transpose ---

        # sLSE: (tileQ128,stageQ):(1,128)
        # sLSE_2D_: ((tileQ128,tileK128,stageQ):(1,0,128)
        # sLSE_2D: (tileK128,tileQ128,stageQ):(0,1,0)
        # sdPsum: (tileQ128,stage_dO):(1,128)
        # sdPsum_2D_: ((tileQ128,tileK128,stage_dO):(1,0,128)
        # sdPsum_2D: (tileK128,tileQ128,stage_dO):(0,1,0)
        sLSE_2D_ = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D_ = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sLSE_2D = layout_utils.transpose_view(sLSE_2D_)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D_)

        # --- Make tmem (coord) tensor of tP/tdS ---

        # tStS: (MMA_tC=(tileK128,tileQ128),MMA_K1,MMA_Q1):((65536,1),0,0)
        # tStP: (MMA_tC=(tileK128,tileQFP32View64),MMA_K1,MMA_Q1):((65536,1),0,0)
        # tScS: (MMA_tC=(tileK128,tileQ128),MMA_K1,MMA_Q1):((1@0,1@1),0,0)
        # tScP: (MMA_tC=(tileK128,tileQFP32View64),MMA_K1,MMA_Q1):((1@0,1@1),0,0)
        tileQFP32View = self.cta_tiler[1] // Float32.width * self.v_dtype.width
        tStP = cute.composition(
            tStS, (cute.make_layout((self.tile_n, tileQFP32View)), 1, 1)
        )
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(
            tScS, (cute.make_layout((self.tile_n, tileQFP32View)), 1, 1)
        )
        # tdPtdP: (MMA_tC=(tileK128,tileQ128),MMA_K1,MMA_Q1):((65536,1),0,0)
        # tdPtdS: (MMA_tC=(tileK128,tileQFP32View64),MMA_K1,MMA_Q1):((65536,1),0,0)
        # tdPcdP: (MMA_tC=(tileK128,tileQ128),MMA_K1,MMA_Q1):((1@0,1@1),0,0)
        # tdPcdS: (MMA_tC=(tileK128,tileQFP32View64),MMA_K1,MMA_Q1):((1@0,1@1),0,0)
        tdPtdS = cute.composition(
            # bf16 tdS embeded in fp32 tdP cols
            tdPtdP,
            (cute.make_layout((self.tile_n, tileQFP32View)), 1, 1),
        )
        tdPcdP = thr_mma_dP.partition_C(
            cute.make_identity_tensor(self.mma_tiler_vdo[:2])
        )
        tdPcdS = cute.composition(
            tdPcdP, (cute.make_layout((self.tile_n, tileQFP32View)), 1, 1)
        )

        # --- Make T2R tiled copy for S/dP ---

        # T2R copy atom of `tcgen05.ld.sync.aligned.32x32b.x32`
        # layout_src_tv=(32,1024):(0,1) => (row32,col32) cells in tmem per warp
        # layout_dst_tv=(32,32):(32,1) => 32 fp32 elems in rmem per thread
        tmem_load_atom = cute.make_copy_atom(
            # NOTE: 2-CTA assumes: repetiton should always be 32 & 16
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )

        # T2R tiled copy atom:
        # layout_src_tv_tiled=((32,4,numWG2),((32,32),1)):((0,1,128),((4,256),0))
        #   => 4 x (row32,col32) cells in tmem per warp group
        # layout_dst_tv_tiled=((32,4,numWG2),(32,1)):((256,1,128),(4,0))
        #   => still 32 fp32 elems in rmem per thread, but tiled in 2 warp groups
        thr_copy_t2r = sm100_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(
            tidx
        )

        # tStS_t2r: (T2R_CPY_ATOM=((32,32),1),CPY_Q2,MMA_K1,MMA_Q1):(((1,65536),0),64,0,0)
        # tdPtdP_t2r: (T2R_CPY_ATOM=((32,32),1),CPY_Q2,MMA_K1,MMA_Q1):(((1,65536),0),64,0,0)
        # tScS_t2r: (T2R_CPY_ATOM=((32,1),1),CPY_Q2,MMA_K1,MMA_Q1):((1@1,0),64@1,0,0)
        # t0ScS_t2r: (T2R_CPY_ATOM=((32,1),1),CPY_Q2,MMA_K1,MMA_Q1):((1@1,0),64@1,0,0)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)

        # tSsLSE: (T2R_CPY_ATOM=(32,1),CPY_Q2,MMA_K1,MMA_Q1,stageQ):((1,0),64,0,0,0)
        # tSsdPsum: (T2R_CPY_ATOM=(32,1),CPY_Q2,MMA_K1,MMA_Q1,stage_dO):((1,0),64,0,0,0)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))

        num_cpy_stages = cute.size(tScS_t2r, mode=[1])  # CPY_Q2

        # --- Make R2T tiled copy for P/dS ---

        # R2T copy atom of `tcgen05.st.sync.aligned.32x32b.x16`
        # layout_src_tv=(32,16):(16,1) => 16 fp32 elems in rmem per thread
        # layout_dst_tv=(32,512):(0,1) => (row32,col16) cells in tmem per warp
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # R2T tiled copy atom:
        # layout_src_tv_tiled=((32,4,numWG2),(16,1)):((128,1,64),(4,0))
        #   => still 16 fp32 elems in rmem per thread, but tiled in 2 warp groups
        # layout_dst_tv_tiled=((32,4,numWG2),((16,32),1)):((0,1,64),((4,128),0))
        #   => 4 x (row32,col16) cells in tmem per warp group
        thr_copy_r2t = sm100_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(
            tidx
        )

        # tScP_r2t: (T2R_CPY_ATOM=(16,1),CPY_Q2,MMA_K1,MMA_Q1):((1@1,0),32@1,0,0)
        # tStP_r2t: (T2R_CPY_ATOM=((16,32),1),CPY_Q2,MMA_K1,MMA_Q1):(((1,65536),0),32,0,0)
        # tdPtdS_r2t: (T2R_CPY_ATOM=((16,32),1),CPY_Q2,MMA_K1,MMA_Q1):(((1,65536),0),32,0,0)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)

        # --- Make R2S tiled copy for dS ---

        # R2S copy atom of `universal copy`
        # layout_src_tv=(1,1):(0,0)
        # layout_dst_tv=(1,1):(0,0)
        #
        # NOTE: stmatrix is NOT selected here because all stmatrix conditions require num_dp=16,
        # but for N-major output (typical case), get_tmem_load_op selects Ld32x32b (num_dp=32).
        # With num_dp=32, get_smem_store_op falls through all stmatrix branches so returns CopyUniversalOp.
        # Only M-major output would pick Ld16x256b with num_dp=16, enabling `stmatrix.m8n8.x4`.
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, tiled_tmem_load=thr_copy_t2r
        )

        # R2S tiled copy atom:
        # layout_src_tv_tiled=((32,4,2),(1,32)):((256,1,128),(0,4))
        #   => still 16 fp32 elems in rmem per thread, but tiled in 2 warp groups
        # layout_dst_tv_tiled=((32,4,2),(1,32)):((256,1,128),(0,4))
        #   => each thread writes its 16 fp32 elems from rmem to smem
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(
            tidx
        )

        # --- Make sdS epilogue smem tensor (R2S dst) ---

        # NOTE: This part is a bit iffy, we might be making a lot of assumptions here

        # sdS_epi: S<3,4,3> o 0 o ((EPI_Q=(8,16),EPI_K=(64,2))):(((64,512),(1,8192)))
        # tRS_sdS: (R2S_CPY_ATOM=(1,32),CPY_Q=(2)):((0,1),(8192))
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_layout = cute.slice_(sdS_epi_layout.outer, (None, None, 0))
        # NOTE: Need to group into 1 mode to be compatible with thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        # e.g. we assume the swizzle (i.e. layout.inner) stays the same
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        if const_expr(self.use_2cta_instrs):
            # sdS_xchg_epi: S<3,4,3> o 0 o ((EPI_Q=(8,16),EPI_K=(64,2))):(((64,512),(1,8192)))
            sdS_xchg_epi = cute.make_tensor(
                cute.recast_ptr(sdS_xchg.iterator, sdS_epi_layout.inner), sdS_layout
            )
            # tRS_sdS_xchg: (R2S_CPY_ATOM=(1,32),CPY_Q=(2)):((0,1),(8192))
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        # 2-CTA all2all exchange:
        # CTA 0 exchanges stage 1 (bottom half of (tileQ//2,tileK)) of sdS with CTA 1,
        # while CTA 1 exchanges stage 0 (top half of (tileQ//2,tileK)) of sdS with CTA 0.
        # then CTA0/1 both have a (tileQ//2,tileK*CTA2) tile of sdS for dQ GEMM.
        exchange_stage = (
            cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)
        )

        # --- Init consumer / producer pipeline states ---

        consumer_state_S_P_dP = (
            ffa_pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
                pipeline.PipelineUserType.Consumer, 1
            )
        )
        producer_state_dS = (
            ffa_pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
                pipeline.PipelineUserType.Producer, 1
            )
        )
        consumer_state_dKV = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block // self.cluster_shape_mnk[0]
            )

            # --- Define attn mask apply fn ---

            mask = AttentionMaskCls(seqlen_info)
            n_block_for_cluster = n_block // self.cta_group_size
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block_for_cluster,
                # TODO: condition mask_seqlen
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                mask_mod=self.mask_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )

            prefetch_LSE = False
            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            loop_count = m_block_max - m_block_min
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )

            # TODO: review the logics
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)

            # --- Debug print ---

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm100_compute] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block_min={} m_block_max={} loop_count={}",
                        m_block_min,
                        m_block_max,
                        loop_count,
                    )
                    cute.printf(
                        prefix + "num_wg={} tidx={} warp_idx={} cta_rank_in_cluster={}",
                        num_wg,
                        tidx,
                        warp_idx,
                        cta_rank_in_cluster,
                    )
                    cute.printf(prefix + "num_cpy_stages={}", num_cpy_stages)
                    cute.printf("")
                    cute.printf(prefix + "sLSE_2D_: {}", sLSE_2D_.layout)
                    cute.printf(prefix + "sdPsum_2D_: {}", sdPsum_2D_.layout)
                    cute.printf(prefix + "sLSE_2D.layout: {}", sLSE_2D.layout)
                    cute.printf(prefix + "sdPsum_2D.layout: {}", sdPsum_2D.layout)
                    cute.printf(prefix + "tSsLSE.layout: {}", tSsLSE.layout)
                    cute.printf(prefix + "tSsdPsum.layout: {}", tSsdPsum.layout)
                    cute.printf("")
                    cute.printf(prefix + "tdVtdV.layout: {}", tdVtdV.layout)
                    cute.printf(prefix + "tdKtdK.layout: {}", tdKtdK.layout)
                    cute.printf("")
                    cute.printf(prefix + "tStS.layout: {}", tStS.layout)
                    cute.printf(prefix + "tStP.layout: {}", tStP.layout)
                    cute.printf(prefix + "tScS.layout: {}", tScS.layout)
                    cute.printf(prefix + "tScP.layout: {}", tScP.layout)
                    cute.printf("")
                    cute.printf(prefix + "tdPtdP.layout: {}", tdPtdP.layout)
                    cute.printf(prefix + "tdPtdS.layout: {}", tdPtdS.layout)
                    cute.printf(prefix + "tdPcdP.layout: {}", tdPcdP.layout)
                    cute.printf(prefix + "tdPcdS.layout: {}", tdPcdS.layout)
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_load_atom.layout_src_tv,
                        tmem_load_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_copy_t2r: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_copy_t2r.layout_src_tv_tiled,
                        thr_copy_t2r.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_store_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_store_atom.layout_src_tv,
                        tmem_store_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_copy_r2t: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_copy_r2t.layout_src_tv_tiled,
                        thr_copy_r2t.layout_dst_tv_tiled,
                    )
                    cute.printf(
                        prefix + "copy_atom_r2s: layout_src_tv={} layout_dst_tv={}",
                        copy_atom_r2s.layout_src_tv,
                        copy_atom_r2s.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "thr_copy_r2s: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                        thr_copy_r2s.layout_src_tv_tiled,
                        thr_copy_r2s.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(prefix + "tStS_t2r.layout: {}", tStS_t2r.layout)
                    cute.printf(prefix + "tdPtdP_t2r.layout: {}", tdPtdP_t2r.layout)
                    cute.printf(prefix + "tScS_t2r.layout: {}", tScS_t2r.layout)
                    cute.printf(prefix + "t0ScS_t2r.layout: {}", t0ScS_t2r.layout)
                    cute.printf(prefix + "tScP_r2t.layout: {}", tScP_r2t.layout)
                    cute.printf(prefix + "tStP_r2t.layout: {}", tStP_r2t.layout)
                    cute.printf(prefix + "tdPtdS_r2t.layout: {}", tdPtdS_r2t.layout)
                    cute.printf(prefix + "tRS_sdS.layout: {}", tRS_sdS.layout)
                    cute.printf("")
                    cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                    cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                    cute.printf(prefix + "sdS.layout: {}", sdS.layout)
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_load_atom.layout_src_tv,
                        tmem_load_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix + "tmem_store_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_store_atom.layout_src_tv,
                        tmem_store_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix + "copy_atom_r2s: layout_src_tv={} layout_dst_tv={}",
                        copy_atom_r2s.layout_src_tv,
                        copy_atom_r2s.layout_dst_tv,
                    )
                    cute.printf("")
                    cute.printf(prefix + "sdS_epi_layout: {}", sdS_epi_layout)
                    cute.printf(prefix + "sdS_layout: {}", sdS_layout)
                    cute.printf("")
                    if const_expr(self.use_2cta_instrs):
                        cute.printf(prefix + "exchange_stage: {}", exchange_stage)
                        cute.printf(
                            prefix + "sdS_xchg_epi_layout: {}", sdS_xchg_epi.layout
                        )
                        cute.printf(
                            prefix + "tRS_sdS_xchg.layout: {}", tRS_sdS_xchg.layout
                        )
                        cute.printf("")

            # --- Mainloop for softmax fwd/bwd ---

            # NOTE: For block sparsity: iterate over sparse m_block count
            # and derive actual m_block from Q_IDX/FULL_Q_IDX tensors.
            # For dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                is_full_block = False

                # TODO: review the logics
                if const_expr(self.use_block_sparsity):
                    m_block, is_full_block = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )

                # //////////////////////////////////////////////
                #  S2R copy sLSE & T2R copy tS to rLSE/rS
                # //////////////////////////////////////////////

                # Wait for sLSE to be full
                pipeline_LSE.consumer_wait(consumer_state_LSE)

                # S2R copy sLSE to rLSE if to prefetch and not shuffle
                tSrLSE_s2r = cute.make_rmem_tensor(
                    tScS_t2r[None, 0, 0, 0].shape, Float32
                )
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(
                        tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r
                    )

                # Wait for tS to be full
                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)

                # T2R copy tS to rS
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                if const_expr(self.tile_hdim == 192):
                    # TODO: review the logics
                    # Signal S tmem load completion using pipeline_S_P when hdim 192
                    # dP is overlapped with S
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                elif const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
                    # Signal S tmem load completion using pipeline_dS when 2cta hdim 128
                    if iter_idx > 0:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            # Commit tdS to be full for prev iter in 2-CTA mode
                            pipeline_dS.producer_commit(producer_state_dS)
                        producer_state_dS.advance()

                # TODO: review the logics
                if const_expr(self.score_mod_bwd is not None):
                    tSrS_pre = cute.make_fragment_like(tSrS_t2r)
                    cute.autovec_copy(tSrS_t2r, tSrS_pre)
                # TODO: review the logics
                if const_expr(self.score_mod is not None):
                    # Apply score_mod FIRST -> matches forward
                    self.apply_score_mod_fwd(
                        tSrS_t2r,
                        thr_copy_t2r,
                        thr_mma_S,
                        batch_idx,
                        head_idx,
                        m_block,
                        n_block,
                        softmax_scale,
                        seqlen_info,
                        aux_tensors,
                        fastdiv_mods,
                    )

                # //////////////////////////////////////////////
                #  Apply mask on rS
                # //////////////////////////////////////////////

                check_m_boundary = (m_block + 1) * self.tile_m > seqlen_info.seqlen_q
                mask_fn(
                    tSrS_t2r,
                    m_block=m_block,
                    is_full_block=is_full_block,
                    check_m_boundary=check_m_boundary,
                )

                # //////////////////////////////////////////////
                #  Softmax-fwd: rP = exp(rS - rLSE)
                #  and R2T copy rP to tP
                # //////////////////////////////////////////////

                lane_idx = cute.arch.lane_idx()
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_cpy_stages):  # CPY_Q2
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]

                    # S2R copy sLSE(i) if not to prefetch
                    if const_expr(not self.shuffle_LSE):
                        if const_expr(stage > 0 or not prefetch_LSE):
                            cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                        tSrLSE = tSrLSE_s2r
                    else:
                        tSrLSE = tSsLSE_cur[lane_idx]

                    # Apply softmax-fwd: F = rS - rLSE, P = exp(F)
                    for v in cutlass.range_constexpr(  # T2R_CPY_ATOM32 // 2
                        cute.size(tSrS_t2r, mode=[0]) // 2
                    ):
                        if const_expr(not self.shuffle_LSE):
                            lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        else:
                            lse_pair = (
                                cutedsl_utils.shuffle_sync(tSrLSE, offset=2 * v),
                                cutedsl_utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                            )

                        # Apply F = rS * scale - rLSE = fma(rS, scale, -rLSE)
                        (
                            tSrS_cur[2 * v],
                            tSrS_cur[2 * v + 1],
                        ) = cute.arch.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )

                        # Apply P = exp2(F)
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(
                            tSrS_cur[2 * v + 1], fastmath=True
                        )

                    # Type cast from rS to rP
                    cutedsl_utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])

                    # Fence and sync before R2T store
                    # TODO(REVIEW): why only the first stage needs this
                    if const_expr(stage == 0):
                        cute.arch.fence_view_async_tmem_load()
                        # Without this barrier, we could have 1 warp writing to P in tmem while
                        # another warp is still reading S from tmem.
                        self.compute_sync_barrier.arrive_and_wait()

                    # R2T copy rP to tP
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                # Fence and sync all R2T store done
                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()

                # TODO: review the logics
                if const_expr(not self.tile_hdim == 192):
                    # Signal tmem store P completion with pipeline_S_P
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)

                # Release sLSE(i) to be empty
                # NOTE: Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()

                # //////////////////////////////////////////////
                #  Softmax-bwd: rdS.T = rP.T * (rdP.T - rdPsum)
                #  after T2R copy tdP to rdP
                #  and then R2T/R2S copy rdS to tdS/sdS
                #  and DS2S copy sdS_exg to peer CTA in 2-CTA mode
                # //////////////////////////////////////////////

                # Wait for sdPsum/tdP to be full
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)

                # Apply softmax-bwd: rdS.T = rP.T * (rdP.T - rdPsum)
                # after T2R copy tdP to rdP, and then R2T copy rdS to tdS
                for stage in cutlass.range_constexpr(num_cpy_stages):  # CPY_Q2
                    # T2R copy tdP to rdP
                    tdPrdP_t2r = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, None, None].shape, Float32
                    )
                    cute.copy(
                        thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r
                    )

                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()

                    # NOTE: tSrS_t2r stores rP for now
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]

                    # S2R copy sdPsum to rdPsum
                    tSsdPsum_cur = tSsdPsum[
                        None, stage, 0, 0, consumer_state_dPsum.index
                    ]
                    if const_expr(not self.shuffle_dPsum):
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    else:
                        tSrdPsum = tSsdPsum_cur[lane_idx]

                    # Apply softmax-bwd: rdS = rP * (rdP - rdPsum)
                    for v in cutlass.range_constexpr(
                        cute.size(tdPrdP_t2r, mode=[0]) // 2
                    ):
                        if const_expr(not self.shuffle_dPsum):
                            dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        else:
                            dPsum_pair = (
                                cutedsl_utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                cutedsl_utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                            )
                        (
                            tdPrdP_cur[2 * v],
                            tdPrdP_cur[2 * v + 1],
                        ) = quack.activation.sub_packed_f32x2(
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                        )
                        (
                            tdPrdP_cur[2 * v],
                            tdPrdP_cur[2 * v + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )

                    # TODO: review the logics
                    if const_expr(self.score_mod_bwd is not None):
                        tSrS_pre_cur = tSrS_pre[None, stage, 0, 0]
                        cS_bwd = cute.make_identity_tensor((self.tile_n, self.tile_m))
                        cS_bwd = cute.domain_offset(
                            (n_block * self.tile_n, m_block * self.tile_m), cS_bwd
                        )
                        tScS_bwd = thr_mma_S.partition_C(cS_bwd)
                        tScS_idx_bwd = thr_copy_t2r.partition_D(tScS_bwd)
                        tScS_idx_cur = tScS_idx_bwd[None, stage, 0, 0]
                        self.apply_score_mod_bwd(
                            tdPrdP_cur,
                            tSrS_pre_cur,
                            tScS_idx_cur,
                            batch_idx,
                            head_idx,
                            softmax_scale,
                            seqlen_info,
                            aux_tensors,
                            fastdiv_mods,
                        )
                        # Zero out OOB positions (kv_idx >= seqlen_k) after score_mod_bwd
                        for i in cutlass.range(cute.size(tdPrdP_cur), unroll_full=True):
                            kv_idx = tScS_idx_cur[i][0]
                            tdPrdP_cur[i] = (
                                0.0 if kv_idx >= seqlen_info.seqlen_k else tdPrdP_cur[i]
                            )

                    # Type convert from rdP to rdS
                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    cutedsl_utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)

                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                        if const_expr(self.use_2cta_instrs):
                            tdPrdS_xchg = cute.make_fragment_like(
                                tdPrdS_cvt, self.ds_dtype
                            )

                    # --- R2T copy rdS to tdS ---

                    if const_expr(
                        not self.use_smem_dS_for_mma_dK or self.use_2cta_instrs
                    ):
                        tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                        cute.copy(
                            thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0]
                        )

                    # --- R2S copy rdS to sdS ---

                    # NOTE: For 2-CTA, keep exchange stage in registers,
                    # and write non-exchange to sdS
                    if const_expr(self.use_2cta_instrs):
                        if exchange_stage == stage:
                            cute.autovec_copy(tdPrdS_cvt, tdPrdS_xchg)
                        else:
                            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
                    else:
                        cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])

                if const_expr(not self.use_smem_dS_for_mma_dK):
                    cute.arch.fence_view_async_tmem_store()

                if const_expr(self.use_2cta_instrs):
                    # use pipeline_dP to signal tmem store of dS
                    with cute.arch.elect_one():
                        # Release tdP to be empty in 2-CTA mode
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                consumer_state_S_P_dP.advance()

                # Copy exchange registers to sdS_xchg buffer
                if const_expr(self.use_2cta_instrs):
                    # when hdim 192, sdQacc overlapped with sdS_xchg
                    if const_expr(self.tile_hdim == 192):
                        cute.arch.mbarrier_wait(
                            dQacc_empty_mbar_ptr, phase=producer_state_dS.phase
                        )
                    cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                cute.arch.fence_view_async_shared()
                self.compute_sync_barrier.arrive_and_wait()

                # Release sdPsum to be empty
                # Normally we'd need syncwarp here since only 1 thread will signal in
                # consumer_release, but we already have the self.compute_sync_barrier before this
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()

                # when 2cta hdim 128, pipeline_dS also signals S tmem load completion so is deferred
                if const_expr(not (self.use_2cta_instrs and self.tile_hdim == 128)):
                    with cute.arch.elect_one():
                        # Commit tdS to be full for this iter if not using 2-CTA
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                # DS2S copy from sdS_xchg to peer's sdS buffer in 2-CTA mode
                if const_expr(self.use_2cta_instrs):
                    stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                    stage_copy_elems = const_expr(
                        stage_copy_bytes // (self.ds_dtype.width // 8)
                    )
                    if tidx == 0:
                        peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                        smem_src_ptr = sdS_xchg.iterator
                        # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                        smem_dst_ptr = (
                            sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                        )
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )
                        sm100_utils.cpasync_bulk_s2cluster(
                            smem_src_ptr,
                            smem_dst_ptr,
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )

            # Commit tdS to be full for last iter in 2-CTA mode
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                if process_tile:
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # --- Epilogue for dKV store ---

            if process_tile:
                if const_expr(not self.use_tma_store):
                    # when self.qhead_per_kvhead == 1 and mCuSeqlensK is not None
                    # Non-TMA store dK/dV
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen_info,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        is_print_block=is_print_block,
                    )
                else:  # TMA store dK/dV
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    # TMA store dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen_info,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                        "V",
                        is_print_block=is_print_block,
                    )
                    # TMA store dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen_info,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale if const_expr(not self.dKV_postprocess) else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                        "K",
                        is_print_block=is_print_block,
                    )

            # TODO: review the logics
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity,
            # no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_local or self.is_varlen_q):
                    should_zero_dKV = m_block_min >= m_block_max
                if const_expr(self.use_block_sparsity):
                    # For block sparsity, zero when no m_blocks contribute to this n_block
                    if not process_tile:
                        should_zero_dKV = True

                if should_zero_dKV:
                    # For 2-CTA: use cluster-wide tile size (cta_group_size * tile_n)
                    cluster_tile_n = self.tile_n * self.cta_group_size
                    n_block_for_tile = n_block // self.cta_group_size
                    gmem_tiled_copy_zero_dK = sm100_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = sm100_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen_info.offset_batch_K(mdV, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    mdK_cur = seqlen_info.offset_batch_K(mdK, batch_idx, dim=3)[
                        None, None, head_idx
                    ]
                    gdK = cute.local_tile(
                        mdK_cur, (cluster_tile_n, self.tile_hdim), (n_block_for_tile, 0)
                    )
                    gdV = cute.local_tile(
                        mdV_cur,
                        (cluster_tile_n, self.tile_hdimv),
                        (n_block_for_tile, 0),
                    )
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(
                        tdVgdV[None, 0, 0]
                    )
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if (
                                row_idx
                                < seqlen_info.seqlen_k
                                - cluster_tile_n * n_block_for_tile
                            ):
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(
                                        gmem_tiled_copy_zero_dK,
                                        zero,
                                        tdKgdK[None, i, j],
                                    )
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if (
                                row_idx
                                < seqlen_info.seqlen_k
                                - cluster_tile_n * n_block_for_tile
                            ):
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(
                                        gmem_tiled_copy_zero_dV,
                                        zero,
                                        tdVgdV[None, i, j],
                                    )

            # Advance to next KV tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def _dq_semaphore_lock_value(
        self,
        iter_idx: Int32,
        curr_q_cnt: Int32,
        curr_dq_write_order: Optional[cute.Tensor],
        curr_dq_write_order_full: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        seqlen,
        m_block: Int32,
        n_block: Int32,
    ) -> Int32:
        lock_value = n_block
        if const_expr(self.spt):
            n_block_max_for_m_block = block_info.get_n_block_max_for_m_block(
                seqlen, m_block
            )
            lock_value = n_block_max_for_m_block - 1 - n_block
        if const_expr(self.use_block_sparsity):
            assert blocksparse_tensors is not None
            if const_expr(blocksparse_tensors.dq_write_order is not None):
                sparse_iter = iter_idx // self.subtile_factor
                if sparse_iter < curr_q_cnt:
                    assert curr_dq_write_order is not None
                    lock_value = curr_dq_write_order[sparse_iter]
                else:
                    assert curr_dq_write_order_full is not None
                    lock_value = curr_dq_write_order_full[sparse_iter - curr_q_cnt]
        return lock_value

    @cute.jit
    def dQacc_reduce(
        self,
        mdQacc: cute.Tensor,
        sdQacc: cute.Tensor,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: pipeline.PipelineUmmaAsync,
        dQacc_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        mdQ_semaphore: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        is_print_block: bool = False,
    ):
        # --- Set up thread info ---

        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(
            cute.arch.warp_idx() % len(self.reduce_warp_ids)
        )
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        # --- Make T2R tiled copy of dQacc ---

        # T2R copy atom of `tcgen05.ld.sync.aligned.32x32b.x32`
        # layout_src_tv=(32,1024):(0,1) => (row32,col32) cells in tmem per warp
        # layout_dst_tv=(32,32):(32,1) => 32 fp32 elems in rmem per thread
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)),
            Float32,
        )

        # T2R tiled copy atom:
        # layout_src_tv_tiled=((32,4),(1024,1)):((0,1),(4,0))
        #   => 4 x (row32,col32) cells in tmem per warp group
        # layout_dst_tv_tiled=((32,4),(32,1)):((128,1),(4,0))
        #   => still 32 fp32 elems in rmem per thread, but tiled in a warp group
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)

        # tdQtdQ: (MMA_tC=(intraRow64,(col64,interRow2)),MMA_Q1,MMA_HD1):((65536,(1,4194304)),0,0)
        # tdQtdQ_t2r: (T2R_CPY_ATOM=((col32,row32),1),CPY_HD2,MMA_Q1,MMA_HD1):(((1,65536),0),32,0,0)
        # tdQcdQ: (MMA_tC=(64,128),MMA_Q1,MMA_HD1):((1@0,1@1),0,0)
        # tdQrdQ_t2r: (T2R_CPY_ATOM=(32,1),CPY_HD2,MMA_Q1,MMA_HD1)
        # tdQrdQ: (redHDCol8,restRedHDCTA8)
        # where restRedHDCTA = tileHD // redHDCol // CTA2
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(
            cute.make_identity_tensor(self.mma_tiler_dsk[:2])  # (tileQ128,tileHD128)
        )
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        tdQrdQ_shape = (
            self.dQ_reduce_ncol,
            self.tile_hdim // self.cta_group_size // self.dQ_reduce_ncol,
        )

        # NOTE: in 2-CTA mode, each CTA rank will reduce half dQacc along tileQ
        # since each CTA holds a (tileQ//2,tileHD) slice of the full dQacc tile,
        # e.g. each restRedHDCTA8 stages in total restRedHD16 stages
        stage_offset_cta = (
            self.dQacc_reduce_stage_cta * cta_rank_in_cluster
            if const_expr(self.use_2cta_instrs)
            else 0
        )

        # --- Make R2S tiled copy of dQacc ---

        # R2S tiled copy atom:
        # layout_src_tv=(1,4):(0,1)
        # layout_dst_tv=(1,4):(0,1)
        # layout_src_tv_tiled=(128,(4,1)):(4,(1,0))
        # layout_dst_tv_tiled=(128,(4,1)):(4,(1,0))
        thr_copy_dQacc_r2s = copy_utils.tiled_copy_1d(
            self.dqacc_dtype,
            num_reduce_threads,
            num_copy_elems=128 // self.dqacc_dtype.width,  # 128B => 4 fp32
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQacc_r2s.partition_D(sdQacc)

        # --- Init pipeline states ---

        dQ_consumer_state = ffa_pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = ffa_pipeline.make_pipeline_state(
            ffa_pipeline.PipelineUserType.Producer, self.sdQacc_stage
        )
        read_flag = const_expr(not self.deterministic)

        # /////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // self.cta_group_size  # for 2CTA
            seqlen_info = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block_cta_group
            )

            # --- Make gdQacc ---

            # mdQacc_cur: (seqQ*tileHD):(1)
            if const_expr(not seqlen_info.has_cu_seqlens_q):
                mdQacc_cur = mdQacc[None, head_idx, batch_idx]
            else:
                mdQacc_cur = cute.domain_offset(
                    (seqlen_info.padded_offset_q * self.tile_hdim,),
                    mdQacc[None, head_idx],
                )

            # gdQacc_: (tileQ128*tileHD128,restQ):(1,16384)
            gdQacc_ = cute.local_tile(
                mdQacc_cur, (self.tile_m * self.tile_hdim,), (None,)
            )

            # gdQacc: (tileQ128*redHDCol8,restRedHD16,restQ):(1,1024,16384)
            # where restRedHD = tileHD // redHDCol
            gdQacc = cute.flat_divide(
                gdQacc_, (self.tile_m * self.tile_hdim // self.dQacc_reduce_stage,)
            )

            if const_expr(self.deterministic):
                assert mdQ_semaphore is not None
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            delay_semaphore_release = (
                not self.tile_hdim == 192 and not self.use_block_sparsity
            )

            loop_count = m_block_max - m_block_min
            process_tile = (
                const_expr(not self.is_local and not self.is_varlen_q)
                or m_block_min < m_block_max
            )

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            curr_dq_write_order = None
            curr_dq_write_order_full = None

            # TODO: review the logics
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)
            if const_expr(self.deterministic and self.use_block_sparsity):
                assert blocksparse_tensors is not None
                if const_expr(blocksparse_tensors.dq_write_order is not None):
                    assert blocksparse_tensors.dq_write_order is not None
                    curr_dq_write_order = blocksparse_tensors.dq_write_order[
                        batch_idx, head_idx, n_block, None
                    ]
                    if const_expr(blocksparse_tensors.dq_write_order_full is not None):
                        assert blocksparse_tensors.dq_write_order_full is not None
                        curr_dq_write_order_full = (
                            blocksparse_tensors.dq_write_order_full[
                                batch_idx, head_idx, n_block, None
                            ]
                        )

            # --- Debug print ---

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm100_dQacc_reduce] "
                    cute.printf("")
                    cute.printf(
                        prefix + "tidx={} warp_idx={} cta_rank_in_cluster={}",
                        tidx,
                        warp_idx,
                        cta_rank_in_cluster,
                    )
                    cute.printf(
                        prefix + "dQ_reduce_ncol_t2r={}, stage_offset_cta={}",
                        self.dQ_reduce_ncol_t2r,
                        stage_offset_cta,
                    )
                    cute.printf(
                        prefix + "dQacc_reduce_stage={} dQacc_reduce_stage_cta={}",
                        self.dQacc_reduce_stage,
                        self.dQacc_reduce_stage_cta,
                    )
                    cute.printf(
                        prefix + "dQacc_reduce_stage_t2r={}",
                        self.dQacc_reduce_stage_t2r,
                    )
                    cute.printf(
                        prefix + "delay_semaphore_release={}", delay_semaphore_release
                    )
                    cute.printf(
                        prefix + "loop_count={} m_block_min={} m_block_max={}",
                        loop_count,
                        m_block_min,
                        m_block_max,
                    )
                    cute.printf("")
                    cute.printf(prefix + "tdQtdQ.layout: {}", tdQtdQ.layout)
                    cute.printf(prefix + "tdQtdQ_t2r.layout: {}", tdQtdQ_t2r.layout)
                    cute.printf(prefix + "tdQrdQ_t2r.shape: {}", tdQrdQ_t2r_shape)
                    cute.printf(prefix + "tdQrdQ.shape: {}", tdQrdQ_shape)
                    cute.printf(prefix + "tdQcdQ.layout: {}", tdQcdQ.layout)
                    cute.printf(prefix + "tdQsdQ.layout: {}", tdQsdQ.layout)
                    cute.printf(prefix + "sdQacc.layout: {}", sdQacc.layout)
                    cute.printf("")
                    cute.printf(
                        prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                        tmem_load_atom.layout_src_tv,
                        tmem_load_atom.layout_dst_tv,
                    )
                    cute.printf(
                        prefix + "thr_copy_t2r: layout_src_tv_tiled={}",
                        thr_copy_t2r.layout_src_tv_tiled,
                    )
                    cute.printf(
                        prefix + "thr_copy_t2r: layout_dst_tv_tiled={}",
                        thr_copy_t2r.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(
                        prefix + "thr_copy_dQacc_r2s: layout_src_tv={}",
                        thr_copy_dQacc_r2s.layout_src_tv,
                    )
                    cute.printf(
                        prefix + "thr_copy_dQacc_r2s: layout_dst_tv={}",
                        thr_copy_dQacc_r2s.layout_dst_tv,
                    )
                    cute.printf(
                        prefix + "thr_copy_dQacc_r2s: layout_src_tv_tiled={}",
                        thr_copy_dQacc_r2s.layout_src_tv_tiled,
                    )
                    cute.printf(
                        prefix + "thr_copy_dQacc_r2s: layout_dst_tv_tiled={}",
                        thr_copy_dQacc_r2s.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(prefix + "mdQacc_cur.layout={}", mdQacc_cur.layout)
                    cute.printf(prefix + "gdQacc_cur.layout={}", gdQacc_.layout)
                    cute.printf(prefix + "gdQacc.layout={}", gdQacc.layout)
                    cute.printf("")

            # --- dQacc reduce mainloop ---

            # NOTE: For block sparsity: iterate over sparse m_block count
            # and derive actual m_block from Q_IDX/FULL_Q_IDX tensors.
            # For dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                m_block_oob_upper = False

                # TODO: review the logics
                if const_expr(self.use_block_sparsity):
                    m_block, _ = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    m_block_oob_upper = m_block >= m_block_max

                # Wait for tdQ(i) to be full
                pipeline_dQ.consumer_wait(dQ_consumer_state)

                # T2R copy dQacc
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()

                # Release tdQ(i) to be empty
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                if m_block_max > 0:
                    m_block = cutlass.min(m_block, m_block_max - 1)
                gdQacc_cur = gdQacc[None, None, m_block]

                tdQrdQ = cute.make_tensor(tdQrdQ_t2r.iterator, tdQrdQ_shape)

                for stage in cutlass.range_constexpr(
                    cute.size(tdQrdQ, mode=[1])
                ):  # restRedHDCTA8
                    # R2S copy dQacc
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    tdQrdQ_r2s = cute.make_tensor(
                        tdQrdQ[None, stage].iterator, tdQsdQ_r2s.shape
                    )
                    cute.copy(thr_copy_dQacc_r2s, tdQrdQ_r2s, tdQsdQ_r2s)

                    # Proxy fence to make sure generic smem store is visible to TMA
                    cute.arch.fence_view_async_shared()

                    # Semaphore acquire
                    # TODO: review the logics
                    if const_expr(self.deterministic and stage == 0):
                        if not m_block_oob_upper:
                            lock_value = self._dq_semaphore_lock_value(
                                iter_idx,
                                curr_q_cnt,
                                curr_dq_write_order,
                                curr_dq_write_order_full,
                                blocksparse_tensors,
                                block_info,
                                seqlen_info,
                                m_block,
                                n_block_cta_group,
                            )
                            cutedsl_utils.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                lock_value,
                            )

                    # Sync before S2G copy
                    self.reduce_sync_barrier.arrive_and_wait()

                    # S2G copy dQacc (TMA atomic reduce)
                    if is_tma_warp and not m_block_oob_upper:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQacc[None, smem_idx].iterator,
                                gdQacc_cur[None, stage + stage_offset_cta].iterator,
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.sdQacc_stage - 1, read=read_flag
                        )
                    elif is_tma_warp:
                        # Drain pending TMA stores so SMEM buffers are safe to reuse
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)

                    # Sync after S2G copy
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()

                    # TODO: review the logics
                    if const_expr(
                        self.deterministic and stage == 0 and delay_semaphore_release
                    ):
                        if m_block > m_block_min:
                            cutedsl_utils.arrive_inc(
                                mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                1,
                            )

                # TODO: review the logics
                if const_expr(self.tile_hdim == 192):
                    if const_expr(self.sdQacc_stage > 1):
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dQacc_empty_mbar_ptr)

                # Semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                # TODO: review the logics
                if const_expr(self.deterministic and not delay_semaphore_release):
                    if const_expr(self.sdQacc_stage > 1 and not self.tile_hdim == 192):
                        if is_tma_warp and not m_block_oob_upper:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    if not m_block_oob_upper:
                        cutedsl_utils.arrive_inc(
                            mdQ_semaphore_cur[m_block, None].iterator,
                            tidx,
                            cta_rank_in_cluster,
                            1,
                        )

            if process_tile:
                if is_tma_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                self.reduce_sync_barrier.arrive_and_wait()
                # final semaphore release
                if const_expr(self.deterministic and delay_semaphore_release):
                    cutedsl_utils.arrive_inc(
                        mdQ_semaphore_cur[(m_block_max - 1, None)].iterator,
                        tidx,
                        cta_rank_in_cluster,
                        1,
                    )

            if const_expr(
                self.deterministic
                and not self.spt
                and not self.use_block_sparsity
                and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    cutedsl_utils.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        tidx,
                        cta_rank_in_cluster,
                        1,
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen_info: SeqlenInfoQK,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: pipeline.PipelineUmmaAsync,
        consumer_state_dKV: pipeline.PipelineState,
        softmax_scale: Float32,
        is_print_block: bool = False,
    ):
        # --- Set up thread info ---

        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = seqlen_info.offset_batch_K(mdV, batch_idx, dim=3)[
            None, None, head_idx
        ]
        mdK_cur = seqlen_info.offset_batch_K(mdK, batch_idx, dim=3)[
            None, None, head_idx
        ]

        # --- T2R tiled copy tdV to rdV ---

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)
        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)

        # Wait for tdV to be full
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        # --- R2G tiled copy rdV to gdV ---

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(
            mdV_cur, (self.mma_tiler_pdo[0], self.tile_hdimv), (None, 0)
        )
        gdV_tile = gdV[None, None, n_block // self.cta_group_size]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        if tidx < seqlen_info.seqlen_k - self.tile_n * n_block:
            cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            # Release tdV to be empty
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # --- T2R tiled copy tdK to rdK ---

        # Wait for tdK to be full
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        # --- R2G tiled copy rdK to gdK ---

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(
            mdK_cur, (self.mma_tiler_dsq[0], self.tile_hdim), (None, 0)
        )
        gdK_tile = gdK[None, None, n_block // self.cta_group_size]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        if tidx < seqlen_info.seqlen_k - self.tile_n * n_block:
            cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            # Release tdK to be empty
            pipeline_dKV.consumer_release(consumer_state_dKV)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if (wg_idx == 0) and (tidx == 0) and is_print_block:
                prefix = "[epilogue_dKV] "
                cute.printf("")
                cute.printf(
                    prefix
                    + "tidx={} warp_idx={} wg_idx={} n_block={} softmax_scale={}",
                    tidx,
                    warp_idx,
                    wg_idx,
                    n_block,
                    softmax_scale,
                )
                cute.printf("")
                cute.printf(
                    prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                    tmem_load_atom.layout_src_tv,
                    tmem_load_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "tiled_tmem_ld_dV: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    tiled_tmem_ld_dV.layout_src_tv_tiled,
                    tiled_tmem_ld_dV.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix
                    + "tiled_tmem_ld_dK: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    tiled_tmem_ld_dK.layout_src_tv_tiled,
                    tiled_tmem_ld_dK.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix + "atom_universal_copy: layout_src_tv={} layout_dst_tv={}",
                    atom_universal_copy.layout_src_tv,
                    atom_universal_copy.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "tiled_gmem_store_dV: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    tiled_gmem_store_dV.layout_src_tv_tiled,
                    tiled_gmem_store_dV.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix
                    + "tiled_gmem_store_dK: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    tiled_gmem_store_dK.layout_src_tv_tiled,
                    tiled_gmem_store_dK.layout_dst_tv_tiled,
                )
                cute.printf("")
                cute.printf(prefix + "tdVtdV.layout: {}", tdVtdV.layout)
                cute.printf(prefix + "tdVtdV_t2r.layout: {}", tdVtdV_t2r.layout)
                cute.printf(prefix + "tdVcdV.layout: {}", tdVcdV.layout)
                cute.printf(prefix + "tdVcdV_t2r.layout: {}", tdVcdV_t2r.layout)
                cute.printf(prefix + "tdVrdV_t2r.layout: {}", tdVrdV_t2r.layout)
                cute.printf(prefix + "tdVrdV_r2s.layout: {}", tdVrdV_r2s.layout)
                cute.printf(prefix + "gdV.layout: {}", gdV.layout)
                cute.printf(prefix + "tdVgdV.layout: {}", tdVgdV.layout)
                cute.printf(prefix + "tdVgdV_r2g.layout: {}", tdVgdV_r2g.layout)
                cute.printf("")
                cute.printf(prefix + "tdKtdK.layout: {}", tdKtdK.layout)
                cute.printf(prefix + "tdKtdK_t2r.layout: {}", tdKtdK_t2r.layout)
                cute.printf(prefix + "tdKcdK.layout: {}", tdKcdK.layout)
                cute.printf(prefix + "tdKcdK_t2r.layout: {}", tdKcdK_t2r.layout)
                cute.printf(prefix + "tdKrdK_t2r.layout: {}", tdKrdK_t2r.layout)
                cute.printf(prefix + "tdKrdK_r2s.layout: {}", tdKrdK_r2s.layout)
                cute.printf(prefix + "gdK.layout: {}", gdK.layout)
                cute.printf(prefix + "tdKgdK.layout: {}", tdKgdK.layout)
                cute.printf(prefix + "tdKgdK_r2g.layout: {}", tdKgdK_r2g.layout)
                cute.printf("")

        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen_info: SeqlenInfoQK,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: pipeline.PipelineUmmaAsync,
        consumer_state_dKV: pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
        is_print_block: bool = False,
    ) -> pipeline.PipelineState:
        # --- Set up thread info ---

        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        # --- Make sdK/sdV ---

        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile
            if const_expr(K_or_V == "K")
            else self.sdV_flat_epi_tile
        )
        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # R2S tiled copy
        # layout_src_tv=(1,8):(0,1) | layout_dst_tv=(1,8):(0,1)
        # layout_src_tv_tiled=(128,(8,1)):(1,(128,0))
        # layout_dst_tv_tiled=(128,(8,1)):(1,(128,0))

        # sdKV: (EPI_K=(8,16),EPI_HD=(64,1)):((64,512),(1,0))
        # tdKVsdKV_r2s: (R2S_CPY_ATOM=(8,1),CPY_K1,CPY_HD8):((1,0),0,8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        # --- Make gdK/gdV ---

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            assert not seqlen_info.has_cu_seqlens_k, "varlen uses non tma store path"
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(  # (tileK128,tileHD128)
                mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
            )
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tileK,tileHD/2)
            gdKV_epi = cute.local_tile(  # (tileK,EPI_HD64,tileHD/2/EPI_HD64)
                gdKV, epi_tile, (0, None)
            )
        else:
            if const_expr(not seqlen_info.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset(
                    (seqlen_info.padded_offset_k * tile_hdim,), mdKV[None, head_idx_kv]
                )
            gdKV_p = cute.local_tile(  # (tileK*tileHD,)
                mdKV_cur, (self.tile_n * tile_hdim,), (n_block,)
            )
            gdKV = cute.logical_divide(  # (tileK*tileHD/2,)
                gdKV_p, (self.tile_n * tile_hdim // num_wg,)
            )[((None, wg_idx),)]
            gdKV_epi = cute.flat_divide(  # (tileK*tileHD/2/stageEPI, stageEPI)
                gdKV, (flat_epi_tile,)
            )

        # --- Make mdK/mdV semaphore ---

        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]
        read_flag = const_expr(not deterministic_KV)

        # --- TMA partition gdK/gdV and sdK/sdV ---

        if const_expr(not self.dKV_postprocess):
            # tdKVsdKV: ((8192,1)):((1,0))
            # tdKVgdKV: (((64,128),1),stageEPI):(((1@0,1@1),0),0)
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            if const_expr(K_or_V == "K"):
                assert (
                    num_epi_stages == self.num_epi_stages
                ), "Epi stage calculation is wrong (K)"
            else:
                assert (
                    num_epi_stages == self.num_epi_stages_v
                ), "Epi stage calculation is wrong (V)"
        else:
            num_epi_stages = (
                self.num_epi_stages
                if const_expr(K_or_V == "K")
                else self.num_epi_stages_v
            )

        # Make T2R copy atom for tdK/tdV
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)),
            Float32,
        )

        # Wait for tdK/tdV to be full
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # Semaphore acquire
        if const_expr(deterministic_KV):
            cutedsl_utils.wait_eq(
                mdKV_semaphore_cur.iterator,
                tidx,
                wg_idx,
                head_idx % self.qhead_per_kvhead,
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        # --- Epilogue loop ---

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # T2R copy tdK/tdV to rdK/rdV
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(
                tidx
            )
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[
                None, None, 0, 0
            ]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[
                None, None, 0, 0
            ]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert (
                cute.size(tdKVrdKV_t2r)
                == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE
            ), "RMEM<->TMEM fragment size mismatch"

            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # Scale rdK/rdV if needed
            if const_expr(scale is not None):
                for i in cutlass.range(
                    cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True
                ):
                    (
                        tdKVrdKV_t2r[2 * i],
                        tdKVrdKV_t2r[2 * i + 1],
                    ) = cute.arch.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )

            # Type convert rdK/rdV
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # R2S copy rdK/rdV to sdK/sdV
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)

            # Proxy fence and barrier to make sure SMEM store is visible to
            # all threads in this wg before TMA store
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # S2G copy sdK/sdV to gdK/gdV
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    # If qhead_per_kvhead == 1, we only need to TMA store
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:  # otherwise, we need to TMA atomic reduce
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)

                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx,
                number_of_threads=128 + cute.arch.WARP_SIZE,
            )

        # Semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            cutedsl_utils.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            # Release tdK or tdV to be empty
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # --- Debug print ---

        if const_expr(self.debug_print):
            if (wg_idx == 0) and (tidx == 0) and is_print_block:
                prefix = f"[epilogue_{K_or_V}_tma] "
                cute.printf("")
                cute.printf(
                    prefix + "tidx={} wg_idx={} num_wg={} n_block={} scale={}",
                    tidx,
                    wg_idx,
                    num_wg,
                    n_block,
                    scale,
                )
                cute.printf(
                    prefix + "tile_hdim={} dK_reduce_ncol={} num_epi_stages={}",
                    tile_hdim,
                    self.dK_reduce_ncol,
                    num_epi_stages,
                )
                cute.printf("")
                cute.printf(
                    prefix + "tmem_load_atom: layout_src_tv={} layout_dst_tv={}",
                    tmem_load_atom.layout_src_tv,
                    tmem_load_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "thr_copy_t2r: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    thr_copy_t2r.layout_src_tv_tiled,
                    thr_copy_t2r.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix + "thr_copy_r2s_dKV: layout_src_tv={} layout_dst_tv={}",
                    thr_copy_r2s_dKV.layout_src_tv,
                    thr_copy_r2s_dKV.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "thr_copy_r2s_dKV: layout_src_tv_tiled={} layout_dst_tv_tiled={}",
                    thr_copy_r2s_dKV.layout_src_tv_tiled,
                    thr_copy_r2s_dKV.layout_dst_tv_tiled,
                )
                cute.printf("")
                cute.printf(prefix + "tdKVtdKV.layout: {}", tdKVtdKV.layout)
                cute.printf(prefix + "tdKVtdKV_t2r.layout: {}", tdKVtdKV_t2r.layout)
                cute.printf(prefix + "tdKVcdKV.layout: {}", tdKVcdKV.layout)
                cute.printf(prefix + "tdKVcdKV_t2r.layout: {}", tdKVcdKV_t2r.layout)
                cute.printf(prefix + "tdKVrdKV_t2r.layout: {}", tdKVrdKV_t2r.layout)
                cute.printf(prefix + "tdKVrdKV.layout: {}", tdKVrdKV.layout)
                cute.printf(prefix + "tdKVrdKV_r2s.layout: {}", tdKVrdKV_r2s.layout)
                cute.printf("")
                cute.printf(prefix + "sdKV.layout: {}", sdKV.layout)
                cute.printf(prefix + "tdKVsdKV_r2s.layout: {}", tdKVsdKV_r2s.layout)
                cute.printf("")
                cute.printf(prefix + "mdKV_cur.layout: {}", mdKV_cur.layout)
                cute.printf(prefix + "gdKV_p.layout: {}", gdKV_p.layout)
                cute.printf(prefix + "gdKV.layout: {}", gdKV.layout)
                cute.printf(prefix + "gdKV_epi.layout: {}", gdKV_epi.layout)
                if const_expr(not self.dKV_postprocess):
                    cute.printf(prefix + "tdKVsdKV.layout: {}", tdKVsdKV.layout)
                    cute.printf(prefix + "tdKVgdKV.layout: {}", tdKVgdKV.layout)
                cute.printf("")

        return consumer_state_dKV
