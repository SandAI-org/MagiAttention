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

# mypy: disable-error-code="arg-type,union-attr,index,misc,no-redef,assignment,attr-defined"
# pyright: reportInvalidTypeForm=false

import math
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Boolean, Float32, Int32, const_expr, pipeline
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase
from quack.sm90_utils import gemm_w_idx, gemm_zero_init

from . import cutedsl_utils
from . import pipeline as ffa_pipeline
from .block_info import BlockInfo
from .cutedsl_utils import ThreadCooperativeGroup
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .named_barrier import NamedBarrierBwd
from .seqlen_info import SeqlenInfoQK
from .softmax import apply_score_mod_bwd_inner, apply_score_mod_inner
from .sparse_utils import (
    BlockSparseTensors,
    consume_block_sparse_mma_bwd_sm90,
    dQacc_store_block_sparse_bwd_sm90,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm90,
)
from .tile_scheduler import (
    SingleTileLPTBwdScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)


class FFABwdSm90:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        mask_type: int = MT_MAP.full,
        is_local: bool = False,
        deterministic: bool = False,
        tile_m: int = 64,
        tile_n: int = 128,
        Q_stage: int = 2,
        dO_stage: int = 2,
        PdS_stage: int = 2,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 2,
        AtomLayoutMdQ: int = 1,
        num_threads: int = 384,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        subtile_factor: cutlass.Constexpr[int] = 1,
        dQ_single_wg: bool = False,
        debug_print: bool = False,
    ):
        self.dtype = dtype

        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )

        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.qhead_per_kvhead = qhead_per_kvhead
        self.mask_type = mask_type
        self.is_local = is_local
        self.deterministic = deterministic
        self.tile_m = tile_m  # tileQ64
        self.tile_n = tile_n  # tileK128
        self.num_threads = num_threads  # 384 (3 WGs = 1 producer + 2 MMA)
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE  # 12

        self.Q_stage = Q_stage
        self.dO_stage = dO_stage
        self.PdS_stage = PdS_stage
        assert self.dO_stage in [1, self.Q_stage]
        assert self.PdS_stage in [1, self.Q_stage]

        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB

        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ

        self.num_wg_load = 1
        self.num_wg_mma = (self.num_threads // 128) - self.num_wg_load
        self.mma_dkv_is_rs = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_wg_mma
            and SdP_swapAB
            and not dKV_swapAB
        )

        self.V_in_regs = V_in_regs

        # May be overridden in __call__ for varlen inputs.
        if qhead_per_kvhead > 1:
            assert self.same_hdim_kv, "GQA backward requires head_dim == head_dim_v"
            assert self.num_wg_mma == 2, "GQA backward assumes 2 warp groups"

        # These are tuned for speed
        # Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share
        # them and then shuffle to get the value whenever we need? This can reduce register
        # pressure when SdP_swapAB, where each thread needs to keep statistics for (kBlockM / 4)
        # rows. If !SdP_swapAB, each thread only needs to keep statistics for 2 rows.
        self.shuffle_LSE = self.SdP_swapAB and self.tile_hdim <= 64
        self.shuffle_dPsum = self.SdP_swapAB and self.tile_hdim <= 64

        self.buffer_align_bytes = 1024
        self.num_warps_per_wg = 4
        self.num_threads_per_wg = self.num_warps_per_wg * cute.arch.WARP_SIZE

        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.subtile_factor = subtile_factor
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4

        self.qk_acc_dtype = Float32

        # dQ_single_wg: WG0 computes the full dQ GEMM, WG1 skips it.
        # Only valid for 2 MMA warp groups.
        if dQ_single_wg:
            assert self.num_wg_mma == 2, "dQ_single_wg only supports 2 warp groups"

        self.num_wg_dQ = 1 if dQ_single_wg else self.num_wg_mma

        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[bwd_sm90_init] "
            print()
            print(f"{prefix}Initialized FFABwdSm90 with: ")
            print(
                f"{prefix}{self.dtype=} | {self.tile_hdim=} | {self.tile_hdimv=} | {self.qhead_per_kvhead=}"
            )
            print(
                f"{prefix}{self.mask_type=} | {self.is_causal=} | {self.is_local=} | {self.deterministic=}"
            )
            print(
                f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.num_threads=} | {self.num_warps=}"
            )
            print(f"{prefix}{self.Q_stage=} | {self.dO_stage=} | {self.PdS_stage=}")
            print(
                f"{prefix}{self.SdP_swapAB=} | {self.dKV_swapAB=} | {self.dQ_swapAB=}"
            )
            print(
                f"{prefix}{self.AtomLayoutMSdP=} | {self.AtomLayoutNdKV=} | {self.AtomLayoutMdQ=}"
            )
            print(
                f"{prefix}{self.num_wg_mma=} | {self.num_wg_load=} | "
                f"{self.num_wg_dQ=} | {self.mma_dkv_is_rs=} | {self.V_in_regs=}"
            )
            print(f"{prefix}{self.shuffle_LSE=} | {self.shuffle_dPsum=}")
            print(
                f"{prefix}{self.score_mod=} | {self.score_mod_bwd=} | {self.mask_mod=}"
            )
            print(
                f"{self.num_warps_per_wg=} | "
                f"{self.num_threads_per_wg=} | "
                f"{self.buffer_align_bytes=}"
            )
            print()

    @property
    def is_causal(self) -> bool:
        return self.mask_type == MT_MAP.causal

    def _check_tile(self) -> None:
        """Validate the kernel config (dtype, head dims, tile sizes, threads)."""
        if self.dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise ValueError(f"Only Float16/BFloat16 is supported, got {self.dtype}")
        if self.head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {self.head_dim}")
        if self.head_dim_v % 8 != 0:
            raise ValueError(
                f"head_dim_v must be a multiple of 8, got {self.head_dim_v}"
            )
        if self.tile_n % 16 != 0:
            raise ValueError(f"tile_n must be a multiple of 16, got {self.tile_n}")
        if self.num_threads % 32 != 0:
            raise ValueError(
                f"num_threads must be a multiple of 32, got {self.num_threads}"
            )

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQacc_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQacc_type not in [Float32]):
            raise TypeError("dQacc tensor must be Float32")
        if const_expr(self.qhead_per_kvhead == 1):
            if const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError(
                    "mdK and mdV tensors must have the same data type as mQ"
                )
        else:
            if const_expr(not (mdK_type == mdV_type == Float32)):
                raise TypeError(
                    "mdKaccum and mdVaccum tensors must have the data type Float32"
                )
        assert mQ_type == self.dtype

    def _get_tiled_mma(self):
        maybe_swap_mn = (
            lambda shape, swap: (shape[1], shape[0], *shape[2:]) if swap else shape
        )

        # Tiled MMA for S = Q @ K.T / dP = dO @ V.T
        # Thr Layout VMNK: (128,2,1,1):(1,128,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,64,16)
        # TV Layout A:     (128,(64,16)):(0,(1,64))
        # TV Layout B:     (128,(64,16)):(0,(1,64))
        # TV Layout C:     ((4,8,4),(2,2,8)):((128,1,16),(64,8,512))
        atom_layout_SdP = (
            self.AtomLayoutMSdP,
            self.num_wg_mma // self.AtomLayoutMSdP,
            1,
        )
        tiler_mn_SdP = (
            self.tile_m // atom_layout_SdP[0],
            self.tile_n // atom_layout_SdP[1],
        )
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[1] if not self.SdP_swapAB else tiler_mn_SdP[0]),
        )

        # Tiled MMA for dV = P.T @ dO / dK = dS.T @ Q
        # Thr Layout VMNK: (128,2,1,1):(1,128,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,128,16)
        # TV Layout A:     ((4,8,4),(2,2,2)):((128,1,16),(64,8,512))
        # TV Layout B:     (128,(128,16)):(0,(1,128))
        # TV Layout C:     ((4,8,4),(2,2,16)):((128,1,16),(64,8,512))
        atom_layout_dKV = (
            self.AtomLayoutNdKV,
            self.num_wg_mma // self.AtomLayoutNdKV,
            1,
        )
        tiler_mn_dK = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdim // atom_layout_dKV[1],
        )
        tiler_mn_dV = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdimv // atom_layout_dKV[1],
        )
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.MN
                if not self.mma_dkv_is_rs
                else warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1] if not self.dKV_swapAB else tiler_mn_d[0]),
                a_source=warpgroup.OperandSource.RMEM
                if self.mma_dkv_is_rs
                else warpgroup.OperandSource.SMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]

        # Tiled MMA for dQ = dS @ K
        # Thr Layout VMNK: (128,1,2,1):(1,0,128,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,64,16)
        # TV Layout A:     (128,(64,16)):(0,(1,64))
        # TV Layout B:     (128,(64,16)):(0,(1,64))
        # TV Layout C:     ((4,8,4),(2,2,8)):((128,1,16),(64,8,512))
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (
            self.tile_m // atom_layout_dQ[0],
            self.tile_hdim // atom_layout_dQ[1],
        )
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN
            if not self.dQ_swapAB
            else warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1] if not self.dQ_swapAB else tiler_mn_dQ[0]),
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQacc_struct = [
            cute.struct.Align[
                cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQacc_layout, Float32),
            ]
        ]

        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = (
            cute.cosize(self.sPdS_layout) if const_expr(not self.mma_dkv_is_rs) else 0
        )
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.Q_stage
            ],
            128,
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.dO_stage
            ],
            128,
        ]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cosize_sP], self.buffer_align_bytes
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cosize_sdS], self.buffer_align_bytes
            ]
            sdQacc: sdQacc_struct

        self.shared_storage_cls = SharedStorageQKV

    def _setup_attributes(self):
        # --- Set up tiled MMA ---

        (
            self.tiled_mma_SdP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()

        self.num_mma_threads = self.tiled_mma_SdP.size
        assert self.num_threads == self.num_mma_threads + self.num_threads_per_wg
        self.num_producer_threads = (
            self.num_wg_load * cute.arch.WARP_SIZE
        )  # only first warp in producer WG
        self.load_warp_ids = list(range(0, self.num_wg_load * self.num_warps_per_wg))
        self.mma_warp_ids = list(
            range(self.num_wg_load * self.num_warps_per_wg, self.num_warps)
        )

        # --- Set up registers ---

        REG_LIMIT = 504 if self.num_wg_mma == 2 else 512
        if const_expr(self.num_wg_mma == 2):
            if const_expr(self.num_wg_dQ == 1):
                self.num_mma_regs_wg0 = 256
                self.num_mma_regs_wg1 = 224
            else:
                self.num_mma_regs_wg0 = 240
                self.num_mma_regs_wg1 = 240
            self.num_mma_regs = self.num_mma_regs_wg0  # for backward compat
            self.num_producer_regs = 24
            assert (
                self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs
                <= REG_LIMIT
            )
        else:  # 3 warp groups
            self.num_mma_regs_wg0 = 160
            self.num_mma_regs_wg1 = 160
            self.num_mma_regs = 160
            self.num_producer_regs = 32
            assert (
                self.num_mma_regs_wg0 * self.num_wg_mma + self.num_producer_regs
                <= REG_LIMIT
            )

        if const_expr(self.debug_print):
            # NOTE: we might need extra registers for load warp to debug print
            # otherwise, it will raise illegal instruction error
            num_regs_for_print = 24
            self.num_producer_regs += num_regs_for_print
            self.num_mma_regs -= num_regs_for_print
            self.num_mma_regs_wg0 -= num_regs_for_print
            self.num_mma_regs_wg1 -= num_regs_for_print

        # We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
        # Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
        # The M dimension (tile_m) doesn't matter for the layout, only the K dimension
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            # Need to set major_mode_size (mms) to accommodate Q and Q.T
            sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, shape, stage, mms
            )
            for shape, stage, mms in [
                (
                    (self.tile_m, self.tile_hdim),
                    self.Q_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
                (
                    (self.tile_m, self.tile_hdimv),
                    self.dO_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ

        # Accomodate both K and K.T
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        # There's only V, no V.T, so layout is normal
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )

        # Accomodate both S and S.T
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQacc_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )

        # dQacc R->S
        self.r2s_tiled_copy_dQacc = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            # thr_layout
            cute.make_layout((self.num_threads_per_wg, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),  # val_layout
        )
        # dKVaccum for GQA epilogue - reuses sV+sK memory recast as f32
        # TODO: assert that sVaccum and sKaccum don't overflow smem

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[bwd_sm90_setup_attributes] "
            print()
            print(f"{prefix}num_mma_threads: {self.num_mma_threads}")
            print(f"{prefix}num_producer_threads: {self.num_producer_threads}")
            print(f"{prefix}{self.load_warp_ids=} | {self.mma_warp_ids=}")
            print(f"{prefix}num_mma_regs: {self.num_mma_regs}")
            print(f"{prefix}num_mma_regs_wg0: {self.num_mma_regs_wg0}")
            print(f"{prefix}num_mma_regs_wg1: {self.num_mma_regs_wg1}")
            print(f"{prefix}num_producer_regs: {self.num_producer_regs}")
            print()
            print(f"{prefix}tiled_mma_SdP: {self.tiled_mma_SdP}")
            print()
            print(f"{prefix}tiled_mma_dK: {self.tiled_mma_dK}")
            print()
            print(f"{prefix}tiled_mma_dV: {self.tiled_mma_dV}")
            print()
            print(f"{prefix}tiled_mma_dQ: {self.tiled_mma_dQ}")
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sdO_layout: {self.sdO_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sPdS_layout: {self.sPdS_layout}")
            print(f"{prefix}sdQacc_layout: {self.sdQacc_layout}")
            print()

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
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Set up attributes
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Checks ---

        self._check_tile()
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQacc, mdK, mdV)
            )
        )

        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        # For GQA (qhead_per_kvhead > 1), multiple Q heads accumulate into the same dK/dV,
        # so we need the float32 accum path + postprocess.
        # For varlen_k with qhead_per_kvhead == 1, we use ragged TMA tensors.
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None

        # --- Set up attributes ---

        self._setup_attributes()

        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mO/mLSE tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQacc, mdK, mdV = [
            cutedsl_utils.assume_tensor_aligned(t)
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQacc, mdK, mdV)
        ]

        # Non-varlen inputs are (b, s, n, h), varlen inputs are (s, n, h).
        # We convert both to a seqlen-major view with head-dim second.
        # Each tensor may have different rank when Q is padded (seqused_q) but K/V are unpadded (cu_seqlens_k).
        def _qkv_transpose(t):
            return layout_utils.select(
                t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1]
            )

        # mQ/mdO: (sQ,HD,nhQ,batch):(nhQ*HD,1,HD,sQ*nhQ*HD)
        # mK/mV:  (sK,HD,nhK,batch):(nhK*HD,1,HD,sK*nhK*HD)
        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        if const_expr(self.qhead_per_kvhead == 1):
            # mdK/mdV: (sK,HD,nhK,batch):(nhK*HD,1,HD,sK*nhK*HD)
            mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        else:
            # Accum tensors are (b, n, s*h) for non-varlen and (n, s*h) for varlen.
            # mdK/mdV: (sK*HD,nhK,batch):(1,sK*HD,sK*HD*nhK)
            accum_transpose = [2, 1, 0] if cute.rank(mdK.shape) == 3 else [1, 0]
            mdK, mdV = [layout_utils.select(t, accum_transpose) for t in (mdK, mdV)]
        # Non-varlen stats are (b, n, s), varlen stats are (n, s).
        # mLSE/mdPsum: (sQpad,nhQ,batch):(1,sQpad,sQpad*nhQ)
        # mdQacc:    (sQpad*HD,nhQ,batch):(1,sQpad*HD,sQpad*HD*nhQ)
        LSE_dPsum_dQacc_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE, mdPsum, mdQacc = [
            layout_utils.select(t, LSE_dPsum_dQacc_transpose)
            for t in (mLSE, mdPsum, mdQacc)
        ]

        # (batch, num_head, num_m_blocks, cluster_size) -> (num_m_blocks, cluster_size, num_head, batch)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=[2, 3, 1, 0])
        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=[2, 3, 1, 0])
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        # ///////////////////////////////////////////////////////////////////////////////
        # Make TMA tiled copy atom and tensors
        # ///////////////////////////////////////////////////////////////////////////////

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
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
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = (
            self.tile_n * self.tile_hdimv * Float32.width // 8
        )

        # tma_atom_Q: layout_src_tv=(1,tileM*tileHD):(0,1), layout_dst_tv=(1,tileM*tileHD):(0,1)
        # tma_tensor_Q: (sQ,HD,nhQ,batch):(1@1,1@0,1@2,1@3)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        # tma_atom_K: layout_src_tv=(1,tileN*tileHD):(0,1), layout_dst_tv=(1,tileN*tileHD):(0,1)
        # tma_tensor_K: (sK,HD,nhK,batch):(1@1,1@0,1@2,1@3)
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        # tma_atom_V: layout_src_tv=(1,tileN*tileHD):(0,1), layout_dst_tv=(1,tileN*tileHD):(0,1)
        # tma_tensor_V: (sK,HD,nhK,batch):(1@1,1@0,1@2,1@3)
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        # tma_atom_dO: layout_src_tv=(1,tileM*tileHD):(0,1), layout_dst_tv=(1,tileM*tileHD):(0,1)
        # tma_tensor_dO: (sQ,HD,nhQ,batch):(1@1,1@0,1@2,1@3)
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        if const_expr(self.qhead_per_kvhead == 1):
            mdK_tma = (
                copy_utils.create_ragged_tensor_for_tma(
                    mdK, ragged_dim=0, ptr_shift=True
                )
                if self.varlen_k
                else mdK
            )
            mdV_tma = (
                copy_utils.create_ragged_tensor_for_tma(
                    mdV, ragged_dim=0, ptr_shift=True
                )
                if self.varlen_k
                else mdV
            )
            # tma_atom_dK: layout_src_tv=(1,tileN*tileHD):(0,1), layout_dst_tv=(1,tileN*tileHD):(0,1)
            # tma_tensor_dK: (sK,HD,nhK,batch):(1@1,1@0,1@2,1@3)
            tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdK_tma,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
            )
            # tma_atom_dV: layout_src_tv=(1,tileN*tileHD):(0,1), layout_dst_tv=(1,tileN*tileHD):(0,1)
            # tma_tensor_dV: (sK,HD,nhK,batch):(1@1,1@0,1@2,1@3)
            tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                mdV_tma,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
            )
        else:
            tma_atom_dK = tma_atom_dV = tma_tensor_dK = tma_tensor_dV = None

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        if const_expr(mCuSeqlensK is not None or mSeqUsedK is not None):
            self.tile_scheduler_cls = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            self.tile_scheduler_cls = SingleTileLPTBwdScheduler
        else:
            self.tile_scheduler_cls = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK.shape[0] - 1),  # num_batch
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q or total_q for seqlen_k
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # Swapping the role of Q & K
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            head_swizzle=self.deterministic,
        )
        tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
            tile_sched_args
        )
        grid_dim = self.tile_scheduler_cls.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        self._get_shared_storage_cls()

        # --- Make others ---

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        fastdiv_mods = None
        if const_expr(aux_tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[bwd_sm90_call] "

            cute.printf("")
            cute.printf(
                prefix + "use_block_sparsity: {} | varlen_q: {} | varlen_k: {} | ",
                self.use_block_sparsity,
                self.is_varlen_q,
                self.varlen_k,
            )
            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
            cute.printf(prefix + "mdO.layout: {}", mdO.layout)
            cute.printf(prefix + "mLSE.layout: {}", mLSE.layout)
            cute.printf(prefix + "mdPsum.layout: {}", mdPsum.layout)
            cute.printf(prefix + "mdQacc.layout: {}", mdQacc.layout)
            cute.printf(prefix + "mdK.layout: {}", mdK.layout)
            cute.printf(prefix + "mdV.layout: {}", mdV.layout)
            cute.printf("")
            cute.printf(
                prefix + "tma_copy_bytes: Q={}, K={}, V={}, dO={}",
                self.tma_copy_bytes["Q"],
                self.tma_copy_bytes["K"],
                self.tma_copy_bytes["V"],
                self.tma_copy_bytes["dO"],
            )
            cute.printf(
                prefix + "tma_copy_bytes: LSE={}, dPsum={}, dQ={}, dKacc={}, dVacc={}",
                self.tma_copy_bytes["LSE"],
                self.tma_copy_bytes["dPsum"],
                self.tma_copy_bytes["dQ"],
                self.tma_copy_bytes["dKacc"],
                self.tma_copy_bytes["dVacc"],
            )
            cute.printf("")
            cute.printf(
                prefix + "tma_atom_Q: layout_src_tv={}, layout_dst_tv={}",
                tma_atom_Q.layout_src_tv,
                tma_atom_Q.layout_dst_tv,
            )
            cute.printf(prefix + "tma_tensor_Q.layout: {}", tma_tensor_Q.layout)
            cute.printf(
                prefix + "tma_atom_K: layout_src_tv={}, layout_dst_tv={}",
                tma_atom_K.layout_src_tv,
                tma_atom_K.layout_dst_tv,
            )
            cute.printf(prefix + "tma_tensor_K.layout: {}", tma_tensor_K.layout)
            cute.printf(
                prefix + "tma_atom_V: layout_src_tv={}, layout_dst_tv={}",
                tma_atom_V.layout_src_tv,
                tma_atom_V.layout_dst_tv,
            )
            cute.printf(prefix + "tma_tensor_V.layout: {}", tma_tensor_V.layout)
            cute.printf(
                prefix + "tma_atom_dO: layout_src_tv={}, layout_dst_tv={}",
                tma_atom_dO.layout_src_tv,
                tma_atom_dO.layout_dst_tv,
            )
            cute.printf(prefix + "tma_tensor_dO.layout: {}", tma_tensor_dO.layout)
            if const_expr(self.qhead_per_kvhead == 1):
                cute.printf(
                    prefix + "tma_atom_dK: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_dK.layout_src_tv,
                    tma_atom_dK.layout_dst_tv,
                )
                cute.printf(prefix + "tma_tensor_dK.layout: {}", tma_tensor_dK.layout)
                cute.printf(
                    prefix + "tma_atom_dV: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_dV.layout_src_tv,
                    tma_atom_dV.layout_dst_tv,
                )
                cute.printf(prefix + "tma_tensor_dV.layout: {}", tma_tensor_dV.layout)
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
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK if const_expr(self.qhead_per_kvhead == 1) else mdK,
            tma_tensor_dV if const_expr(self.qhead_per_kvhead == 1) else mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQacc,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQacc_layout,
            self.r2s_tiled_copy_dQacc,
            self.tiled_mma_SdP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            aux_tensors,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQacc: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQacc_layout: cute.Layout,
        r2s_tiled_copy_dQacc: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Set up before warp specialization
        # /////////////////////////////////////////////////////////////////////////////

        # --- Set up thread info ---

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Used only for debug print
        # guarded by const_expr so zero overhead when debug_print=False
        bidx, bidy, bidz = cute.arch.block_idx()
        is_print_block = const_expr(self.debug_print) and (
            (bidx == 0) and (bidy == 0) and (bidz == 0)
        )
        is_print_thread = const_expr(self.debug_print) and (
            (tidx == 127) and is_print_block
        )

        # --- Prefetch TMA descriptor ---

        if warp_idx == 0:
            for atom in [
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dK,
                tma_atom_dV,
            ]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)

        # --- Alloc smem storage and fetch ptrs ---

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)

        # --- Make pipelines ---

        pipeline_producer_group = ThreadCooperativeGroup(1)
        pipeline_consumer_group = ThreadCooperativeGroup(
            self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = ffa_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"] + self.tma_copy_bytes["LSE"],
            defer_sync=True,
        )
        pipeline_dO = ffa_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"] + self.tma_copy_bytes["dPsum"],
            defer_sync=False,
        )

        # --- Make smem tensors of sQ/sK/sV/sO/sP/sdS/sLSE/sdPsum/sdQacc ---

        # sQ:  ((ATOM_Q8,LAY_tileQ8),(ATOM_HD64,LAY_tileHD2),STAGE_Q=(1,2)):((64,512),(1,4096),(0,8192))
        # sdO: ((ATOM_Q8,LAY_tileQ8),(ATOM_HD64,LAY_tileHD2),STAGE_dO=(1,2)):((64,512),(1,4096),(0,8192))
        # sK:  ((ATOM_K8,LAY_tileK16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sV:  ((ATOM_K8,LAY_tileK16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sP/sdS: ((ATOM_Q8,LAY_tileQ8),(ATOM_K64,LAY_tileK2),STAGE_PdS=(1,2)):((64,512),(1,4096),(0,8192))
        # sLSE/sdPsum:   (tileQ64,STAGE_Q=2):(1,64)
        # sdQacc: (tileQ*tileHD//2=4096,STAGE=2):(1,4096)
        sQ: cute.Tensor = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner
        )
        sdO: cute.Tensor = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner
        )
        sK: cute.Tensor = storage.sK.get_tensor(
            sK_layout.outer, swizzle=sK_layout.inner
        )
        sV: cute.Tensor = storage.sV.get_tensor(
            sV_layout.outer, swizzle=sV_layout.inner
        )
        sP: cute.Tensor | None = None
        if const_expr(not self.mma_dkv_is_rs):
            sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS: cute.Tensor = storage.sdS.get_tensor(
            sPdS_layout.outer, swizzle=sPdS_layout.inner
        )
        sLSE: cute.Tensor = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum: cute.Tensor = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdQacc: cute.Tensor = storage.sdQacc.get_tensor(sdQacc_layout)

        # --- Make other info dataclass ---

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
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
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )

        # --- Make tile scheduler ---

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(
            tile_scheduler, TileSchedulerProtocol
        ), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread:
                prefix = "[bwd_sm90_kernel_setup] "
                cute.printf("")
                cute.printf(prefix + "warp_idx={} tidx={}", warp_idx, tidx)
                cute.printf("")
                cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                cute.printf(prefix + "sK.layout: {}", sK.layout)
                cute.printf(prefix + "sV.layout: {}", sV.layout)
                if const_expr(sP is not None):
                    cute.printf(prefix + "sP.layout: {}", sP.layout)
                cute.printf(prefix + "sdS.layout: {}", sdS.layout)
                cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                cute.printf(prefix + "sdQacc.layout: {}", sdQacc.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.load_warp_ids[-1]:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)

            if warp_idx == 0:  # load
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    pipeline_Q,
                    pipeline_dO,
                    block_info,
                    SeqlenInfoCls,
                    tile_scheduler,
                    blocksparse_tensors,
                    qhead_per_kvhead_divmod,
                    is_print_block,
                )
            elif warp_idx == 1:  # dQacc atomic reduce
                self.dQacc_store(
                    mdQacc,
                    sdQacc,
                    block_info,
                    tile_scheduler,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    mdQ_semaphore,
                    is_print_block,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA WarpGroups
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.mma_warp_ids[0]:  # Consumer
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQacc,
                pipeline_Q,
                pipeline_dO,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQacc,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                tile_scheduler,
                aux_tensors,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )

            if const_expr(self.num_wg_dQ == self.num_wg_mma):
                # Both WGs compute dQ
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True, is_print_block=is_print_block)
            else:
                # WG0 computes dQ, WG1 skips it
                warp_idx_in_mma = (
                    cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    - self.mma_warp_ids[0] * cute.arch.WARP_SIZE
                )
                if warp_idx_in_mma < self.num_warps_per_wg:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                    self.mma(*mma_args, is_dQ_wg=True, is_print_block=is_print_block)
                else:
                    cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                    self.mma(*mma_args, is_dQ_wg=False, is_print_block=is_print_block)

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: pipeline.PipelineAsync,
        pipeline_dO: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        is_print_block: bool = False,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % len(
            self.load_warp_ids
        )
        tidx, _, _ = cute.arch.thread_idx()

        is_load_warp = warp_idx_in_wg == 0

        if is_load_warp:
            producer_state_Q = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.Q_stage
            )
            producer_state_dO = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dO_stage
            )

            # ///////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduler loop
            # ///////////////////////////////////////////////////////////////////////////////
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # --- Get current tile info ---

                n_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen_info = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx
                    if const_expr(self.qhead_per_kvhead == 1)
                    else head_idx // qhead_per_kvhead_divmod
                )

                m_block_min, m_block_max = block_info.get_m_block_min_max(
                    seqlen_info, n_block
                )

                if const_expr(not self.use_block_sparsity):
                    total_m_block_cnt = m_block_max - m_block_min
                    process_tile = (
                        const_expr(not self.is_local and not self.is_varlen_q)
                        or m_block_min < m_block_max
                    )
                else:
                    total_m_block_cnt = get_total_q_block_count_bwd(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                    )
                    process_tile = total_m_block_cnt > Int32(0)

                # Used only for debug print
                is_print_thread_and_tile = const_expr(self.debug_print) and (
                    (tidx == 0)
                    and is_print_block
                    and (n_block == 0)
                    and (head_idx == 0)
                    and (batch_idx == 0)
                )

                # //////////////////////////////////////////////
                #  Make gQ/gK/gV/gdO/gLSE/gdPsum
                # //////////////////////////////////////////////

                # mK_cur/mV_cur: (sK,HD):(1@1,1@0)
                mK_cur = seqlen_info.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen_info.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]

                # gK/gV: (tileK,tileHD):(1@1,1@0)
                gK = cute.local_tile(
                    mK_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
                )
                gV = cute.local_tile(
                    mV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0)
                )

                # mQ_cur/mdO_cur: (sQ,HD):(1@1,1@0)
                # mLSE_cur/mdPsum_cur: (sQpad):(1)
                mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mLSE_cur = seqlen_info.offset_batch_Q(
                    mLSE, batch_idx, dim=2, padded=True
                )[None, head_idx]
                mdO_cur = seqlen_info.offset_batch_Q(mdO, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                mdPsum_cur = seqlen_info.offset_batch_Q(
                    mdPsum, batch_idx, dim=2, padded=True
                )[None, head_idx]

                # gQ/gdO: (tileQ,tileHD,restQ):(1@1,1@0,tileQ@1)
                # gLSE/gdPsum: (tileQ,restQ):(1,tileQ)
                # where restQ = sQ // tileQ
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (None, 0))
                gdO = cute.local_tile(
                    mdO_cur, (self.tile_m, self.tile_hdimv), (None, 0)
                )
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))

                # //////////////////////////////////////////////
                #  Make TMA load partial fns
                # //////////////////////////////////////////////

                load_K, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK, single_stage=True
                )
                load_V, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV, single_stage=True
                )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ
                )
                load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
                load_dO, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO, 0, cute.make_layout(1), gdO, sdO
                )
                load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
                load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                load_LSE = copy_utils.tma_producer_copy_fn(load_LSE, pipeline_Q)
                load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)
                load_dPsum = copy_utils.tma_producer_copy_fn(load_dPsum, pipeline_dO)

                # --- Debug print ---

                if const_expr(self.debug_print):
                    if is_print_thread_and_tile:
                        prefix = "[bwd_sm90_load] "
                        cute.printf("")
                        cute.printf(
                            prefix + "n_block={}, head_idx={}, batch_idx={}",
                            n_block,
                            head_idx,
                            batch_idx,
                        )
                        cute.printf(
                            prefix
                            + "m_block_min={}, m_block_max={}, total_m_block_cnt={}",
                            m_block_min,
                            m_block_max,
                            total_m_block_cnt,
                        )
                        cute.printf(prefix + "process_tile={}", process_tile)
                        cute.printf("")
                        cute.printf(prefix + "mQ_cur.layout: {}", mQ_cur.layout)
                        cute.printf(prefix + "mK_cur.layout: {}", mK_cur.layout)
                        cute.printf(prefix + "mV_cur.layout: {}", mV_cur.layout)
                        cute.printf(prefix + "mdO_cur.layout: {}", mdO_cur.layout)
                        cute.printf(prefix + "mLSE_cur.layout: {}", mLSE_cur.layout)
                        cute.printf(prefix + "mdPsum_cur.layout: {}", mdPsum_cur.layout)
                        cute.printf("")
                        cute.printf(prefix + "gQ.layout: {}", gQ.layout)
                        cute.printf(prefix + "gK.layout: {}", gK.layout)
                        cute.printf(prefix + "gV.layout: {}", gV.layout)
                        cute.printf(prefix + "gdO.layout: {}", gdO.layout)
                        cute.printf(prefix + "gLSE.layout: {}", gLSE.layout)
                        cute.printf(prefix + "gdPsum.layout: {}", gdPsum.layout)
                        cute.printf("")
                        cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                        cute.printf(prefix + "sK.layout: {}", sK.layout)
                        cute.printf(prefix + "sV.layout: {}", sV.layout)
                        cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                        cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                        cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                        cute.printf("")

                # //////////////////////////////////////////////
                #  G2S-load sQ/sK/sV/sdO/sLSE/sdPsum
                # //////////////////////////////////////////////

                if process_tile:
                    if const_expr(not self.use_block_sparsity):
                        first_m_block = m_block_min
                        pipeline_Q.producer_acquire(
                            producer_state_Q, extra_tx_count=self.tma_copy_bytes["K"]
                        )
                        load_K(
                            tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                producer_state_Q
                            )
                        )
                        load_Q(first_m_block, producer_state=producer_state_Q)
                        # Wait for bwd preprocess to finish writing LSE and dPsum
                        cute.arch.griddepcontrol_wait()
                        load_LSE(first_m_block, producer_state=producer_state_Q)
                        producer_state_dO_cur = (
                            producer_state_dO
                            if const_expr(self.Q_stage != self.dO_stage)
                            else producer_state_Q
                        )
                        pipeline_dO.producer_acquire(
                            producer_state_dO_cur,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(
                            tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                producer_state_dO_cur
                            )
                        )
                        load_dO(first_m_block, producer_state=producer_state_dO_cur)
                        load_dPsum(first_m_block, producer_state=producer_state_dO_cur)
                        producer_state_Q.advance()
                        producer_state_dO.advance()

                        for m_block in cutlass.range(
                            m_block_min + 1, m_block_max, unroll=1
                        ):
                            pipeline_Q.producer_acquire(producer_state_Q)
                            load_Q(m_block, producer_state=producer_state_Q)
                            load_LSE(m_block, producer_state=producer_state_Q)
                            producer_state_dO_cur = (
                                producer_state_dO
                                if const_expr(self.Q_stage != self.dO_stage)
                                else producer_state_Q
                            )
                            pipeline_dO.producer_acquire(producer_state_dO_cur)
                            load_dO(m_block, producer_state=producer_state_dO_cur)
                            load_dPsum(m_block, producer_state=producer_state_dO_cur)
                            producer_state_Q.advance()
                            producer_state_dO.advance()
                    else:  # block sparse load (TODO: review the logics)
                        (
                            producer_state_Q,
                            producer_state_dO,
                        ) = produce_block_sparse_q_loads_bwd_sm90(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q,
                            producer_state_dO,
                            pipeline_Q,
                            pipeline_dO,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            load_LSE,
                            load_dPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            Q_stage_eq_dO_stage=(self.Q_stage == self.dO_stage),
                            subtile_factor=self.subtile_factor,
                            m_block_max=m_block_max,
                        )

                # Advance to next K/V tile
                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def apply_score_mod(
        self,
        acc_S: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        n_block: Int32,
        softmax_scale: Float32,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        # NOTE: SdP_swapAB: swapAB transposes the tile, so use (n, m) indexing
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m)
            if self.SdP_swapAB
            else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
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
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor: cute.Tensor,
        score_tensor: cute.Tensor,
        thr_mma_SdP: cute.ThrMma,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        n_block: Int32,
        softmax_scale: Float32,
        seqlen_info: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
    ):
        cS = cute.make_identity_tensor(
            (self.tile_n, self.tile_m)
            if self.SdP_swapAB
            else (self.tile_m, self.tile_n)
        )
        cS = cute.domain_offset(
            (n_block * self.tile_n, m_block * self.tile_m)
            if self.SdP_swapAB
            else (m_block * self.tile_m, n_block * self.tile_n),
            cS,
        )
        tScS = thr_mma_SdP.partition_C(cS)

        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            tScS,
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
            qhead_per_kvhead=self.qhead_per_kvhead,
            transpose_indices=self.SdP_swapAB,
        )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQacc: cute.Tensor,
        pipeline_Q: pipeline.PipelineAsync,
        pipeline_dO: pipeline.PipelineAsync,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQacc: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        AttentionMaskCls: Callable[..., AttentionMask],
        tile_scheduler: TileSchedulerProtocol,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        is_print_block: bool = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tidx -= self.mma_warp_ids[0] * cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_wg)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_wg
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled MMA partitions with partial MMA fns
        # ///////////////////////////////////////////////////////////////////////////////

        # --- S = Q @ K.T ---

        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))

        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        # tSrQ: (MMA_ATOM=1,MMA_Q1,MMA_HD=(4,2),STAGE_Q=(1,2)):(0,0,(2,512),(0,1024))
        # tSrK: (MMA_ATOM=1,MMA_K1,MMA_HD=(4,2)):(0,0,(2,1024))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_S[:2],
            tSrQ,
            tSrK,
            swap_AB=self.SdP_swapAB,
        )

        # tLSEsLSE:   ((ATOM_Q2,MMA_Q8,1),STAGE2):((1,8,0),64)
        # tLSEsdPsum: ((ATOM_Q2,MMA_Q8,1),STAGE2):((1,8,0),64)
        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        # When shuffle=True, rows are distributed across 8 quads (4 threads each) within a warp.
        # Each thread loads only ceil(num_rows/8) values;
        shfl_copy = copy_utils.tiled_copy_1d(
            sLSE.element_type, num_threads=8, num_copy_elems=2
        )
        if const_expr(self.shuffle_LSE):
            tLSEsLSE = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(
                tLSEsLSE
            )
            # ((2, 1), 1, 2) -> (((2, 1), 1), 2)
            tLSEsLSE = cute.group_modes(tLSEsLSE, 0, 2)
        if const_expr(self.shuffle_dPsum):
            tLSEsdPsum = shfl_copy.get_slice(cute.arch.lane_idx() // 4).partition_S(
                tLSEsdPsum
            )
            tLSEsdPsum = cute.group_modes(tLSEsdPsum, 0, 2)

        # --- dP = dO @ V.T ---

        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        # tdPrdO: (MMA_ATOM=1,MMA_Q1,MMA_HD=(4,2),STAGE_dO=(1,2)):(0,0,(2,512),(0,1024))
        # tdPrV:  (MMA_ATOM=1,MMA_K1,MMA_HD=(4,2)):(0,0,(2,1024))
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_dP[:2],
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )

        # --- dV += P.T @ dO ---

        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))

        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        # acc_dV:  (MMA_ATOM=(2,2,16),MMA_K1,MMA_HD1):((1,2,4),0,0)
        # tdVrdOt: (MMA_ATOM=1,MMA_HD1,MMA_Q4,STAGE_dO=(1,2)):(0,0,128,(0,1024))
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn = partial(
                gemm_w_idx,
                tiled_mma_dV,
                acc_dV,
                tdVrPt,
                tdVrdOt,
                swap_AB=self.dKV_swapAB,
            )
        else:
            mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)

        # --- dK += dS.T @ Q ---

        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))

        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        # acc_dK:  (MMA_ATOM=(2,2,16),MMA_K1,MMA_HD1):((1,2,4),0,0)
        # tdKrdSt: (MMA_ATOM=(2,2,2),MMA_K1,MMA_Q4):((1,2,4),0,8)
        # tdKrQt:  (MMA_ATOM=1,MMA_HD1,MMA_Q4,STAGE_Q=(1,2)):(0,0,128,(0,1024))
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        if const_expr(not self.mma_dkv_is_rs):
            mma_dsq_fn = partial(
                gemm_w_idx,
                tiled_mma_dK,
                acc_dK,
                tdKrdSt,
                tdKrQt,
                swap_AB=self.dKV_swapAB,
            )
        else:
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)

        # --- dQ = dS @ K ---

        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = warp_group_idx if const_expr(self.num_wg_dQ > 1) else 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))

        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            # tdQrdS: (MMA_ATOM=1,MMA_Q1,MMA_K=(4,2),STAGE_dS=(1,2)):(0,0,(2,512),(0,1024))
            # tdQrKt: (MMA_ATOM=1,MMA_HD1,MMA_K8):(0,0,128)
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # R2S tiled copy atom and partition of P/dS/dQacc
        # ///////////////////////////////////////////////////////////////////////////////

        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sP if const_expr(not self.SdP_swapAB) else sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdS if const_expr(not self.SdP_swapAB) else sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )

        tdQsdQacc = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQacc = r2s_tiled_copy_dQacc.get_slice(tidx)
            # tdQsdQacc: ((CPY_ATOM=(4,1)),CPY_Q8,1):((1,0),512,0)
            tdQsdQacc = smem_thr_copy_dQacc.partition_D(sdQacc)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make others before persistent tile scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////

        PdS_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )
        score_mod_fn = partial(
            self.apply_score_mod,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )
        score_mod_bwd_fn = partial(
            self.apply_score_mod_bwd,
            thr_mma_SdP=thr_mma_SdP,
            softmax_scale=softmax_scale,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )

        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQacc=tdQsdQacc,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            # acc_dV=acc_dV,
            # acc_dK=acc_dK,
            is_dQ_wg=is_dQ_wg,
        )

        consumer_state_Q = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block
            )

            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
            else:
                total_m_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_m_block_cnt > Int32(0)

            # --- Make mask object and score-mod fn ---

            mask = AttentionMaskCls(seqlen_info)
            score_mod_fn_cur = partial(
                score_mod_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen_info,
            )
            score_mod_bwd_fn_cur = partial(
                score_mod_bwd_fn,
                batch_idx=batch_idx,
                head_idx=head_idx,
                n_block=n_block,
                seqlen_info=seqlen_info,
            )

            # --- Debug print ---

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm90_mma] "
                    cute.printf("")
                    cute.printf(
                        prefix + "n_block={} head_idx={} batch_idx={}",
                        n_block,
                        head_idx,
                        batch_idx,
                    )
                    cute.printf("")
                    cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                    cute.printf(prefix + "sK.layout: {}", sK.layout)
                    cute.printf(prefix + "sV.layout: {}", sV.layout)
                    cute.printf(prefix + "sdO.layout: {}", sdO.layout)
                    if const_expr(sP is not None):
                        cute.printf(prefix + "sP.layout: {}", sP.layout)
                    cute.printf(prefix + "sdS.layout: {}", sdS.layout)
                    cute.printf(prefix + "sLSE.layout: {}", sLSE.layout)
                    cute.printf(prefix + "sdPsum.layout: {}", sdPsum.layout)
                    cute.printf("")
                    cute.printf(prefix + "tLSEsLSE.layout: {}", tLSEsLSE.layout)
                    cute.printf(prefix + "tLSEsdPsum.layout: {}", tLSEsdPsum.layout)
                    cute.printf("")
                    # S = Q @ K.T
                    cute.printf(prefix + "tSrQ.layout: {}", tSrQ.layout)
                    cute.printf(prefix + "tSrK.layout: {}", tSrK.layout)
                    cute.printf("")
                    # dP = dO @ V.T
                    cute.printf(prefix + "tdPrdO.layout: {}", tdPrdO.layout)
                    cute.printf(prefix + "tdPrV.layout: {}", tdPrV.layout)
                    cute.printf("")
                    # dV += P.T @ dO
                    cute.printf(prefix + "acc_dV.layout: {}", acc_dV.layout)
                    if const_expr(sP is not None):
                        cute.printf(prefix + "tdVrPt.layout: {}", tdVrPt.layout)
                    cute.printf(prefix + "tdVrdOt.layout: {}", tdVrdOt.layout)
                    cute.printf("")
                    # dK += dS.T @ Q
                    cute.printf(prefix + "acc_dK.layout: {}", acc_dK.layout)
                    cute.printf(prefix + "tdKrdSt.layout: {}", tdKrdSt.layout)
                    cute.printf(prefix + "tdKrQt.layout: {}", tdKrQt.layout)
                    cute.printf("")
                    # dQ = dS @ K
                    if const_expr(is_dQ_wg):
                        cute.printf(prefix + "tdQrdS.layout: {}", tdQrdS.layout)
                        cute.printf(prefix + "tdQrKt.layout: {}", tdQrKt.layout)
                        cute.printf(prefix + "tdQsdQacc.layout: {}", tdQsdQacc.layout)
                        cute.printf("")

            # --- Mainloop ---

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                        mask_local=self.is_local,
                        mask_mod=self.mask_mod,
                        aux_tensors=aux_tensors,
                        fastdiv_mods=fastdiv_mods,
                    )
                    dKV_accumulate = False
                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                            m_block,
                            consumer_state_Q,
                            consumer_state_dO,
                            mask_fn=mask_fn,
                            score_mod_fn=score_mod_fn_cur,
                            score_mod_bwd_fn=score_mod_bwd_fn_cur,
                            dKV_accumulate=dKV_accumulate,
                            is_print_thread_and_tile=(
                                is_print_thread_and_tile and m_block == m_block_min
                            ),
                        )
                        dKV_accumulate = True
                else:  # block sparse mma (TODO: review the logics)
                    (
                        consumer_state_Q,
                        consumer_state_dO,
                    ) = consume_block_sparse_mma_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        consumer_state_Q,
                        consumer_state_dO,
                        mma_one_m_block_all,
                        mask,
                        self.mask_mod,
                        is_causal=self.is_causal,
                        is_local=self.is_local,
                        thr_mma_SdP=thr_mma_SdP,
                        score_mod_fn=score_mod_fn_cur,
                        score_mod_bwd_fn=score_mod_bwd_fn_cur,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                        aux_tensors=aux_tensors,
                        fastdiv_mods=fastdiv_mods,
                    )

                # --- Epilogue ---

                if const_expr(self.qhead_per_kvhead == 1):
                    acc_dK.store(acc_dK.load() * softmax_scale)
                self.epilogue_dKV(
                    acc_dV,
                    mdV,
                    sV,
                    acc_dK,
                    mdK,
                    sK,
                    seqlen_info,
                    tma_atom_dK,
                    tma_atom_dV,
                    tiled_mma_dK,
                    tiled_mma_dV,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                    qhead_per_kvhead_divmod,
                    mdK_semaphore,
                    mdV_semaphore,
                    is_print_thread_and_tile,
                )
            else:  # KV tile with zero Q blocks produces no dK/dV; write zeros.
                if const_expr(
                    self.use_block_sparsity or self.is_local or self.is_varlen_q
                ):
                    acc_dK.fill(0.0)
                    acc_dV.fill(0.0)
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen_info,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                    )

            # Advance to next K/V tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if warp_idx == self.mma_warp_ids[0]:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """Retrieve the statistic for a given accumulator row.

        When shuffle=False, direct register indexing.
        When shuffle=True, warp shuffle from the thread group that holds the value.
        """
        if const_expr(not shuffle):
            return tSrS[row]

        # tSrS: (((2, 1), 1), 1)), distributed across 8 threads in the warp
        vecsize = cute.size(tSrS, mode=[0, 0])  # 2
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))

        # register index: 0, 1, 0, 1, ..., 2, 3, 2, 3, ...
        return cutedsl_utils.shuffle_sync(
            tSrS[idx0 + idx1 * vecsize], offset=off * 4 + (lane % 4)
        )

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: pipeline.PipelineState | ffa_pipeline.PipelineStateSimple,
        consumer_state_dO: pipeline.PipelineState | ffa_pipeline.PipelineStateSimple,
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: pipeline.PipelineAsync,
        pipeline_dO: pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQacc: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        score_mod_fn: Optional[Callable] = None,
        score_mod_bwd_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
        is_print_thread_and_tile: bool = False,
    ):
        """Process one m_block of the bwd inner loop for a fixed KV tile.

        This is the core bwd sub-process: it fuses 5 GEMMs and 2 pointwise ops over
        one (m_block) x (n_block) tile, accumulating dK/dV across m_blocks and writing
        dQ to smem (handed off to dQacc_store):
          (1) [GEMM 1] S   = Q @ K^T
          (2) [GEMM 2] dP  = dO @ V^T
          (3) [Pointwise 1] P  = exp2(S * scale - LSE)
          (4) [Pointwise 2] dS = P * (dP - dPsum)
          (5) [GEMM 3] dV += P^T @ dO
          (6) [GEMM 4] dQ  = dS @ K        (is_dQ_wg only)
          (7) [GEMM 5] dK += dS^T @ Q
        """
        consumer_state_dO_cur = (
            consumer_state_Q
            if const_expr(self.Q_stage == self.dO_stage)
            else consumer_state_dO
        )
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = (
            consumer_state_dO_cur.index if const_expr(self.dO_stage > 1) else 0
        )
        smem_idx_PdS = smem_idx_Q if const_expr(self.PdS_stage > 1) else 0

        # --- Apply S = Q @ K.T ---

        pipeline_Q.consumer_wait(
            consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
        )
        # acc_S:   (MMA_ATOM=(2,2,8),MMA_Q1,MMA_K1):((1,2,4),0,0)
        # tLSErLSE: ((ATOM_Q2,MMA_Q8,1)):((1,2,0))
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        # If shuffle_LSE, OOB reads are OK since sLSE is already padded
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])

        # --- Apply dP = dO @ V.T ---

        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        # acc_dP: (MMA_ATOM=(2,2,8),MMA_Q1,MMA_K1):((1,2,4),0,0)
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)

        if const_expr(self.score_mod_bwd is not None):
            acc_S_pre = cute.make_fragment_like(acc_S)
            cute.autovec_copy(acc_S, acc_S_pre)

        if const_expr(self.score_mod is not None):
            score_mod_fn(acc_S, m_block=m_block)

        # --- Apply P = softmax(S) = exp(S - LSE) ---

        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        # acc_S_mn: ((ATOM_Q2,MMA_Q8,1),(ATOM_K2,1)):((1,4,0),(2,0))
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])

        # Convert P from f32 -> f16
        # tdVrP: (MMA_ATOM=(2,2,2),MMA_M1,MMA_K=(4,1)):((1,2,4),0,(8,0))
        tdVrP = cutedsl_utils.cvt_f16(
            layout_utils.reshape_acc_to_frgA(acc_S), self.dtype
        )

        # R2S for P
        if const_expr(not self.mma_dkv_is_rs):
            # sync to ensure P has already been used in the previous iteration before overwriting
            if const_expr(self.PdS_stage == 1):
                PdS_barrier.arrive_and_wait()
            copy_P_r2s(tdVrP, dst_idx=smem_idx_PdS)

        # --- Apply softmax bwd: dS = P*(dP - dPsum) ---

        warpgroup.wait_group(0)
        # acc_dP_mn: ((ATOM_Q2,MMA_Q8,1),(ATOM_K2,1)):((1,4,0),(2,0))
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(
                tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum
            )
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

        if const_expr(self.score_mod_bwd is not None):
            score_mod_bwd_fn(acc_dP, acc_S_pre, m_block=m_block)

        # Convert dS from f32 -> f16
        # tdKrdS: (MMA_ATOM=(2,2,2),MMA_M1,MMA_K=(4,1)):((1,2,4),0,(8,0))
        tdKrdS = cutedsl_utils.cvt_f16(
            layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype
        )

        # NOTE: If there's double buffering on dS, we don't need to sync here.
        # Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        # But because both WGs have to sync at the end of the loop and double buffering,
        # this race condition is not possible.
        # This sync is to ensure:
        #  (1) P is written in case of !mma_dkv_is_rs and
        #  (2) dS is already read by the Mma in the previous iteration in case of mma_dkv_is_rs.
        if const_expr(
            not self.mma_dkv_is_rs or (self.PdS_stage == 1 and self.mma_dkv_is_rs)
        ):
            cute.arch.fence_view_async_shared()
            PdS_barrier.arrive_and_wait()

        # R2S for dS
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)

        # --- dV += P.T @ dO ---

        if const_expr(not self.mma_dkv_is_rs):
            mma_pdo_fn(
                A_idx=smem_idx_PdS,
                B_idx=smem_idx_dO,
                zero_init=not dKV_accumulate,
                wg_wait=-1,
            )
        else:
            mma_pdo_fn(
                tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
            )

        # Proxy fence to make sure sdS is written before it's read by WGMMA
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        if const_expr(is_dQ_wg):
            # --- Apply dQ = dS @ K ---

            # acc_dQ: (MMA_ATOM=(2,2,8),MMA_Q1,MMA_HD1):((1,2,4),0,0)
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(
                consumer_state_dO_cur
            )  # release dO as dV mma is done

            # --- Apply dK += dS.T @ Q ---

            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS,
                    B_idx=smem_idx_Q,
                    zero_init=not dKV_accumulate,
                    wg_wait=1,
                )
            else:
                mma_dsq_fn(
                    tCrA=tdKrdS,
                    B_idx=smem_idx_Q,
                    zero_init=not dKV_accumulate,
                    wg_wait=1,
                )

            # dQ R2S: wait for dQacc_store to free the smem buffer, then write dQ to smem
            # When dQ_single_wg, only WG0 enters here so warp_group_idx == 0
            cute.arch.barrier(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_wg + cute.arch.WARP_SIZE,
            )
            tdQrdQacc_flat = cute.make_tensor(
                acc_dQ.iterator, cute.make_layout(tdQsdQacc.shape)
            )
            cute.autovec_copy(tdQrdQacc_flat, tdQsdQacc)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_wg + cute.arch.WARP_SIZE,
            )

            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:  # WG1 skips dQ, only does dV wait + dK
            # --- Apply dK += dS.T @ Q ---
            if const_expr(not self.mma_dkv_is_rs):
                mma_dsq_fn(
                    A_idx=smem_idx_PdS,
                    B_idx=smem_idx_Q,
                    zero_init=not dKV_accumulate,
                    wg_wait=1,
                )
            else:
                mma_dsq_fn(
                    tCrA=tdKrdS,
                    B_idx=smem_idx_Q,
                    zero_init=not dKV_accumulate,
                    wg_wait=1,
                )
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)

        consumer_state_Q.advance()
        consumer_state_dO.advance()

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm90_mma_one_m_block] "
                cute.printf("")
                cute.printf(prefix + "m_block={}", m_block)
                cute.printf(
                    prefix + "smem_idx_Q={} smem_idx_dO={} smem_idx_PdS={}",
                    smem_idx_Q,
                    smem_idx_dO,
                    smem_idx_PdS,
                )
                cute.printf("")
                # S = Q @ K.T
                cute.printf(prefix + "acc_S.layout: {}", acc_S.layout)
                cute.printf(prefix + "tLSErLSE.layout: {}", tLSErLSE.layout)
                # dP = dO @ V.T
                cute.printf(prefix + "acc_dP.layout: {}", acc_dP.layout)
                cute.printf(prefix + "tLSErdPsum.layout: {}", tLSErdPsum.layout)
                cute.printf("")
                # P = exp2(S * scale - LSE) / dS = P * (dP - dPsum)
                cute.printf(prefix + "acc_S_mn.layout: {}", acc_S_mn.layout)
                cute.printf(prefix + "acc_dP_mn.layout: {}", acc_dP_mn.layout)
                cute.printf(prefix + "tdVrP.layout: {}", tdVrP.layout)
                cute.printf(prefix + "tdKrdS.layout: {}", tdKrdS.layout)
                cute.printf("")
                # dQ = dS @ K
                if const_expr(is_dQ_wg):
                    cute.printf(prefix + "acc_dQ.layout: {}", acc_dQ.layout)
                    cute.printf("")

        return consumer_state_Q, consumer_state_dO

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        is_print_thread_and_tile: bool = False,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue),
            num_threads=self.num_mma_threads,
        )

        # --- Write dK/dV back to gmem ---

        if const_expr(self.qhead_per_kvhead == 1):  # store
            # mdK_cur/mdV_cur: (sK,HD):(1@1,1@0)
            mdK_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=3, ragged=self.varlen_k
            )[None, None, head_idx]
            mdV_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=3, ragged=self.varlen_k
            )[None, None, head_idx]
            # gdK: (tileK128,tileHD128):(1@1,1@0)
            # gdV: (tileK128,tileHD128):(1@1,1@0)
            gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
            gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdimv), (n_block, 0))
            store_dK, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
            )
            store_dV, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
            )
            sdV = (
                sV
                if const_expr(not self.dKV_swapAB)
                else layout_utils.transpose_view(sV)
            )
            sdK = (
                sK
                if const_expr(not self.dKV_swapAB)
                else layout_utils.transpose_view(sK)
            )
            copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dV,
                sdV,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_dK,
                sdK,
                tidx,
                transpose=self.dKV_swapAB,
                position_independent=True,
            )
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dV_r2s(acc_dV, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dV()
                cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
            copy_dK_r2s(acc_dK, dst_idx=None)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                store_dK()
                cute.arch.cp_async_bulk_commit_group()
        else:  # atomic add
            deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
            sdKaccum_shape0 = self.tile_n * self.tile_hdim // self.num_wg_mma
            sdVaccum_shape0 = self.tile_n * self.tile_hdimv // self.num_wg_mma
            sdKaccum_layout = cute.make_layout((sdKaccum_shape0, self.num_wg_mma))
            sdVaccum_layout = cute.make_layout((sdVaccum_shape0, self.num_wg_mma))
            head_idx_kv = head_idx // qhead_per_kvhead_divmod
            if const_expr(deterministic_KV):
                assert mdK_semaphore is not None
                assert mdV_semaphore is not None
                mdK_semaphore_cur = mdK_semaphore[n_block, None, head_idx_kv, batch_idx]
                mdV_semaphore_cur = mdV_semaphore[n_block, None, head_idx_kv, batch_idx]
                lock_value = head_idx % self.qhead_per_kvhead
            # mdKaccum_cur: (sKpad*tileHD):(1)
            # mdVaccum_cur: (sKpad*tileHD):(1)
            mdKaccum_cur = seqlen.offset_batch_K(
                mdK, batch_idx, dim=2, padded=True, multiple=self.tile_hdim
            )[None, head_idx_kv]
            mdVaccum_cur = seqlen.offset_batch_K(
                mdV, batch_idx, dim=2, padded=True, multiple=self.tile_hdimv
            )[None, head_idx_kv]
            # gdKaccum: (tileK*tileHD//num_wg=8192,num_wg=2):(1,8192)
            # gdVaccum: (tileK*tileHD//num_wg=8192,num_wg=2):(1,8192)
            gdKaccum_ = cute.local_tile(
                mdKaccum_cur, (self.tile_n * self.tile_hdim,), (n_block,)
            )
            gdKaccum = cute.flat_divide(gdKaccum_, (sdKaccum_shape0,))
            gdVaccum_ = cute.local_tile(
                mdVaccum_cur, (self.tile_n * self.tile_hdimv,), (n_block,)
            )
            gdVaccum = cute.flat_divide(gdVaccum_, (sdVaccum_shape0,))
            # These two overlap each other
            # sdKaccum: (tileK*tileHD//num_wg=8192,num_wg=2):(1,8192)
            # sdVaccum: (tileK*tileHD//num_wg=8192,num_wg=2):(1,8192)
            sVaccum_ptr = cute.recast_ptr(sV.iterator, dtype=Float32)
            sdKaccum = cute.make_tensor(sVaccum_ptr, sdKaccum_layout)
            sdVaccum = cute.make_tensor(sVaccum_ptr, sdVaccum_layout)
            tiled_copy_dKVaccum_r2s = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
                ),
                cute.make_layout((self.num_threads_per_wg, self.num_wg_mma)),
                cute.make_layout(128 // Float32.width),
            )
            thr_copy_dKVaccum_r2s = tiled_copy_dKVaccum_r2s.get_slice(tidx)
            # tdKsdKaccum: ((CPY_ATOM=(4,1)),CPY_K16,1):((1,0),512,0)
            # tdVsdVaccum: ((CPY_ATOM=(4,1)),CPY_K16,1):((1,0),512,0)
            tdKsdKaccum = thr_copy_dKVaccum_r2s.partition_D(sdKaccum)
            tdVsdVaccum = thr_copy_dKVaccum_r2s.partition_D(sdVaccum)

            read_flag = const_expr(not deterministic_KV)
            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                cutedsl_utils.wait_eq(mdK_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdKrdKaccum_flat = cute.make_tensor(acc_dK.iterator, tdKsdKaccum.shape)
            cute.autovec_copy(tdKrdKaccum_flat, tdKsdKaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKaccum[None, wg_idx].iterator,
                            gdKaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dKacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()

            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            if const_expr(deterministic_KV):
                cutedsl_utils.arrive_inc(mdK_semaphore_cur.iterator, tidx, 0, 1)
                cutedsl_utils.wait_eq(mdV_semaphore_cur.iterator, tidx, 0, lock_value)
            epi_barrier.arrive_and_wait()
            tdVrdVaccum_flat = cute.make_tensor(acc_dV.iterator, tdVsdVaccum.shape)
            cute.autovec_copy(tdVrdVaccum_flat, tdVsdVaccum)
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            if warp_idx == 4:
                with cute.arch.elect_one():
                    for wg_idx in cutlass.range_constexpr(self.num_wg_mma):
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdVaccum[None, wg_idx].iterator,
                            gdVaccum[None, wg_idx].iterator,
                            self.tma_copy_bytes["dVacc"] // self.num_wg_mma,
                        )
                cute.arch.cp_async_bulk_commit_group()
            if const_expr(deterministic_KV):
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cutedsl_utils.arrive_inc(mdV_semaphore_cur.iterator, tidx, 0, 1)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm90_epilogue_dKV] "
                cute.printf("")
                cute.printf(
                    prefix + "n_block={} head_idx={} batch_idx={}",
                    n_block,
                    head_idx,
                    batch_idx,
                )
                cute.printf("")
                cute.printf(prefix + "acc_dK.layout: {}", acc_dK.layout)
                cute.printf(prefix + "acc_dV.layout: {}", acc_dV.layout)
                cute.printf(prefix + "sK.layout: {}", sK.layout)
                cute.printf(prefix + "sV.layout: {}", sV.layout)
                cute.printf("")
                if const_expr(self.qhead_per_kvhead == 1):
                    # store path: TMA store sK/sV -> gdK/gdV
                    cute.printf(prefix + "gdK.layout: {}", gdK.layout)
                    cute.printf(prefix + "gdV.layout: {}", gdV.layout)
                    cute.printf("")
                else:
                    # atomic-add path: GQA accum via cp.async.bulk reduce add
                    cute.printf(prefix + "head_idx_kv={}", head_idx_kv)
                    cute.printf(prefix + "mdKaccum_cur.layout: {}", mdKaccum_cur.layout)
                    cute.printf(prefix + "mdVaccum_cur.layout: {}", mdVaccum_cur.layout)
                    cute.printf(prefix + "gdKaccum.layout: {}", gdKaccum.layout)
                    cute.printf(prefix + "gdVaccum.layout: {}", gdVaccum.layout)
                    cute.printf(prefix + "sdKaccum.layout: {}", sdKaccum.layout)
                    cute.printf(prefix + "sdVaccum.layout: {}", sdVaccum.layout)
                    cute.printf(prefix + "tdKsdKaccum.layout: {}", tdKsdKaccum.layout)
                    cute.printf(prefix + "tdVsdVaccum.layout: {}", tdVsdVaccum.layout)
                    cute.printf("")

    @cute.jit
    def dQacc_store(
        self,
        mdQacc: cute.Tensor,
        sdQacc: cute.Tensor,
        block_info: BlockInfo,
        tile_scheduler: TileSchedulerProtocol,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        is_print_block: bool = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_local_tidx = tidx % cute.arch.WARP_SIZE
        read_flag = const_expr(not self.deterministic)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (warp_local_tidx == 0)
                and is_print_block
                and (n_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            # mdQacc_cur: (sQpad*tileHD):(1)
            if const_expr(not seqlen_info.has_cu_seqlens_q):
                mdQacc_cur = mdQacc[None, head_idx, batch_idx]
            else:
                mdQacc_cur = cute.domain_offset(
                    (seqlen_info.padded_offset_q * self.tile_hdim,),
                    mdQacc[None, head_idx],
                )
            # gdQacc: ((tileQ*tileHD//num_wg_dQ=4096,num_wg_dQ),restQ):((1,4096),8192)
            # where restQ = sQpad // tileQ
            gdQacc = cute.local_tile(
                mdQacc_cur,
                (
                    cute.make_layout(
                        (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
                    ),
                ),
                (None,),
            )

            if const_expr(mdQ_semaphore is not None):
                # mdQ_semaphore is (num_m_blocks, cluster_size, num_head, batch) after transpose
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen_info, n_block
            )
            if const_expr(not self.use_block_sparsity):
                process_tile = (
                    const_expr(not self.is_local and not self.is_varlen_q)
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min
            else:
                total_block_cnt = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    subtile_factor=self.subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = total_block_cnt > Int32(0)

            # --- Debug print ---

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[bwd_sm90_dQacc_store] "
                    cute.printf("")
                    cute.printf(
                        prefix + "n_block={} head_idx={} batch_idx={}",
                        n_block,
                        head_idx,
                        batch_idx,
                    )
                    cute.printf(
                        prefix + "m_block_min={} m_block_max={}",
                        m_block_min,
                        m_block_max,
                    )
                    if const_expr(not self.use_block_sparsity):
                        cute.printf(prefix + "loop_count={}", loop_count)
                    cute.printf(prefix + "process_tile={}", process_tile)
                    cute.printf("")
                    cute.printf(prefix + "mdQacc_cur.layout: {}", mdQacc_cur.layout)
                    cute.printf(prefix + "gdQacc.layout: {}", gdQacc.layout)
                    cute.printf(prefix + "sdQacc.layout: {}", sdQacc.layout)
                    cute.printf("")

            # --- Store dQacc ---

            if process_tile:
                if const_expr(not self.use_block_sparsity):
                    for iter_idx in cutlass.range(loop_count, unroll=1):
                        m_block = m_block_min + iter_idx
                        m_block_safe = m_block

                        num_dQ_chunks = self.num_wg_dQ
                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            if const_expr(not self.deterministic):
                                # If deterministic, we already waited at the end of the prev iter
                                cute.arch.cp_async_bulk_wait_group(
                                    num_dQ_chunks - 1 - warp_group_idx, read=read_flag
                                )
                            cute.arch.barrier_arrive(
                                barrier_id=int(NamedBarrierBwd.dQEmptyWG0)
                                + warp_group_idx,
                                number_of_threads=self.num_threads_per_wg
                                + cute.arch.WARP_SIZE,
                            )

                        # Semaphore acquire: wait for prior n_blocks to finish writing this m_block
                        if const_expr(self.deterministic):
                            if const_expr(self.spt):
                                (
                                    _,
                                    n_block_max_for_m_block,
                                ) = block_info.get_n_block_min_max(
                                    seqlen_info, m_block_safe
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block
                            else:
                                lock_value = n_block
                            cutedsl_utils.wait_eq(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                lock_value,
                            )

                        for warp_group_idx in cutlass.range_constexpr(num_dQ_chunks):
                            cute.arch.barrier(
                                barrier_id=int(NamedBarrierBwd.dQFullWG0)
                                + warp_group_idx,
                                number_of_threads=self.num_threads_per_wg
                                + cute.arch.WARP_SIZE,
                            )
                            with cute.arch.elect_one():
                                copy_utils.cpasync_reduce_bulk_add_f32(
                                    sdQacc[None, warp_group_idx].iterator,
                                    gdQacc[
                                        (None, warp_group_idx), m_block_safe
                                    ].iterator,
                                    self.tma_copy_bytes["dQ"],
                                )
                            cute.arch.cp_async_bulk_commit_group()

                        # Semaphore release: signal that this n_block is done with this m_block
                        if const_expr(self.deterministic):
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                            cutedsl_utils.arrive_inc(
                                mdQ_semaphore_cur[(m_block_safe, None)].iterator,
                                warp_local_tidx,
                                0,  # flag_offset
                                1,
                            )
                else:  # block sparse dQacc (TODO: review the logics)
                    assert (
                        not self.deterministic
                    ), "Deterministic not implemented for block-sparse backward"
                    dQacc_store_block_sparse_bwd_sm90(
                        blocksparse_tensors,
                        batch_idx,
                        head_idx,
                        n_block,
                        sdQacc,
                        gdQacc,
                        subtile_factor=self.subtile_factor,
                        m_block_max=m_block_max,
                        num_dQ_warp_groups=self.num_wg_dQ,
                        num_threads_per_warp_group=self.num_threads_per_wg,
                        tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                    )

            # For local masking + deterministic (non-spt): signal remaining m_blocks
            # that this n_block won't visit, so they don't deadlock waiting.
            if const_expr(
                self.deterministic
                and not self.spt
                and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    cutedsl_utils.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator,
                        warp_local_tidx,
                        0,  # flag_offset
                        1,
                    )

            # Advance to next K/V tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.deterministic):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
