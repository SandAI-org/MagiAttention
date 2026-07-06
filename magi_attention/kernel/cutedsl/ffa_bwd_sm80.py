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

# pyright: reportInvalidTypeForm=false


import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp

# isort: split
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils, sm80_utils
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .seqlen_info import SeqlenInfoQK
from .sparse_utils import BlockSparseTensors
from .tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FFABwdSm80:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_stages_Q: int = 2,
        num_stages_dO: int = 2,
        num_threads: int = 256,
        pack_gqa: bool = False,
        mask_type: int = MT_MAP.full,
        SdP_swapAB: bool = False,
        dKV_swapAB: bool = False,
        dQ_swapAB: bool = False,
        AtomLayoutMSdP: int = 1,
        AtomLayoutNdKV: int = 8,
        AtomLayoutMdQ: int = 1,
        V_in_regs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        debug_print: bool = False,
    ):
        self.dtype = dtype

        # NOTE: Pad head_dim to a multiple of 32 (stricter than fwd's 16) due to
        # backward kernel register layout requirements for dQ/dK/dV accumulation
        hdim_multiple_of = 32
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )

        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded

        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size  # tileQ64
        self.n_block_size = n_block_size  # tileK128
        self.num_threads = num_threads  # 256 (2 WGs)
        self.num_mma_warps = self.num_threads // cute.arch.WARP_SIZE  # 8

        self.pack_gqa = pack_gqa
        self.mask_type = mask_type
        self.num_stages_Q = num_stages_Q
        self.num_stages_dO = num_stages_dO

        self.SdP_swapAB = SdP_swapAB
        self.dKV_swapAB = dKV_swapAB
        self.dQ_swapAB = dQ_swapAB
        self.AtomLayoutMSdP = AtomLayoutMSdP
        self.AtomLayoutNdKV = AtomLayoutNdKV
        self.AtomLayoutMdQ = AtomLayoutMdQ
        self.Mma_dKV_is_RS = (
            AtomLayoutMSdP == 1
            and AtomLayoutNdKV == self.num_mma_warps
            and SdP_swapAB
            and not dKV_swapAB
        )
        self.V_in_regs = V_in_regs
        self.share_QV_smem = V_in_regs
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd

        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[bwd_sm80_init] "
            print()
            print(f"{prefix}Initialized FFABwdSm80 with: ")
            print(
                f"{prefix}{self.dtype=} | {self.head_dim_padded=} | "
                + f"{self.head_dim_v_padded=} | {self.qhead_per_kvhead=} | "
                + f"{self.check_hdim_oob=} | {self.check_hdim_v_oob=}"
            )
            print(f"{prefix}{self.mask_type=} | {self.is_causal=} | {self.pack_gqa=}")
            print(
                f"{prefix}{self.m_block_size=} | {self.n_block_size=} | "
                + f"{self.num_threads=} | {self.num_mma_warps=}"
            )
            print(f"{prefix}{self.num_stages_Q=} | {self.num_stages_dO=}")
            print(
                f"{prefix}{self.SdP_swapAB=} | {self.dKV_swapAB=} | {self.dQ_swapAB=}"
            )
            print(
                f"{prefix}{self.AtomLayoutMSdP=} | {self.AtomLayoutNdKV=} | {self.AtomLayoutMdQ=}"
            )
            print(f"{prefix}{self.Mma_dKV_is_RS=} | {self.V_in_regs=}")
            print(f"{prefix}{self.score_mod=} | {self.score_mod_bwd=}")
            print()

    @property
    def is_causal(self) -> bool:
        return self.mask_type == MT_MAP.causal

    @property
    def smem_capacity_arch(self) -> str:
        """SMEM-capacity bucket used by ``_check_tile``.

        Subclasses that reuse the SM80 MMA but have a different SMEM budget
        override this (e.g. FFABwdSm120 -> "sm_120").
        """
        return "sm_80"

    def _check_tile(self) -> None:
        """Validate the kernel config (dtype, head dims, tile sizes, threads, SMEM)."""
        if self.dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise ValueError(f"Only Float16/BFloat16 is supported, got {self.dtype}")
        if self.head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {self.head_dim}")
        if self.head_dim_v % 8 != 0:
            raise ValueError(
                f"head_dim_v must be a multiple of 8, got {self.head_dim_v}"
            )
        if self.n_block_size % 16 != 0:
            raise ValueError(
                f"n_block_size must be a multiple of 16, got {self.n_block_size}"
            )
        if self.num_threads % 32 != 0:
            raise ValueError(
                f"num_threads must be a multiple of 32, got {self.num_threads}"
            )
        # SMEM usage: Q tile + dO tile + K tile + V tile.
        smem_usage_Q = self.m_block_size * self.head_dim * self.num_stages_Q * 2
        smem_usage_dO = self.m_block_size * self.head_dim_v * self.num_stages_dO * 2
        smem_usage_K = self.n_block_size * self.head_dim * 2
        smem_usage_V = self.n_block_size * self.head_dim_v * 2
        smem_usage_QV = (
            max(smem_usage_Q, smem_usage_V)
            if self.V_in_regs
            else (smem_usage_Q + smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_dO + smem_usage_K
        smem_capacity = utils_basic.get_smem_capacity_in_bytes(self.smem_capacity_arch)
        if smem_usage > smem_capacity:
            raise ValueError(
                f"SMEM usage {smem_usage} B exceeds {self.smem_capacity_arch} "
                f"capacity {smem_capacity} B"
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
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        if cutlass.const_expr(not (mQ_type == mK_type == mV_type == mdO_type)):
            raise TypeError("All tensors must have the same data type")
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            if cutlass.const_expr(not (mdK_type == mdV_type == mQ_type)):
                raise TypeError(
                    "mdK and mdV tensors must have the same data type as mQ"
                )
        else:
            if cutlass.const_expr(not (mdK_type == mdV_type == cutlass.Float32)):
                raise TypeError(
                    "mdKaccum and mdVaccum tensors must have the data type Float32"
                )
        if cutlass.const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if cutlass.const_expr(mLSE_type not in [cutlass.Float32]):
            raise TypeError("LSE tensor must be Float32")
        if cutlass.const_expr(mdPsum_type not in [cutlass.Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if cutlass.const_expr(mdQacc_type not in [cutlass.Float32]):
            raise TypeError("dQacc tensor must be Float32")
        if cutlass.const_expr(mCuSeqlensQ_type not in [None, cutlass.Int32]):
            raise TypeError("cuSeqlensQ tensor must be Int32")
        if cutlass.const_expr(mCuSeqlensK_type not in [None, cutlass.Int32]):
            raise TypeError("cuSeqlensK tensor must be Int32")
        if cutlass.const_expr(mSeqUsedQ_type not in [None, cutlass.Int32]):
            raise TypeError("SeqUsedQ tensor must be Int32")
        if cutlass.const_expr(mSeqUsedK_type not in [None, cutlass.Int32]):
            raise TypeError("SeqUsedK tensor must be Int32")
        assert mQ_type == self.dtype

    def _get_tiled_mma(self):
        # Tiled MMA for S=Q*K.T / dP=dO*V.T
        # Thr Layout VMNK: (32,1,8,1):(1,0,32,0)
        # Permutation MNK: (16:1,128:1,16:1)
        # MMA Atom
        # ThrID:           32:1
        # Shape MNK:       (16,8,16)
        # TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
        # TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
        # TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
        AtomLayoutSdP = (  # (1, 8, 1) or (8, 1, 1) if SdP_swapAB
            (self.AtomLayoutMSdP, self.num_mma_warps // self.AtomLayoutMSdP, 1)
            if cutlass.const_expr(not self.SdP_swapAB)
            else (self.num_mma_warps // self.AtomLayoutMSdP, self.AtomLayoutMSdP, 1)
        )
        tiled_mma_sdp = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutSdP,
            # (16, 128, 16) or (128, 16, 16) if SdP_swapAB
            permutation_mnk=(AtomLayoutSdP[0] * 16, AtomLayoutSdP[1] * 16, 16),
        )

        # Tiled MMA for dK=dS.T*Q / dV=P.T*dO
        # Thr Layout VMNK: (32,8,1,1):(1,32,0,0)
        # Permutation MNK: (128:1,16:1,16:1)
        # MMA Atom
        # ThrID:           32:1
        # Shape MNK:       (16,8,16)
        # TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
        # TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
        # TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
        AtomLayoutdKV = (  # (8, 1, 1) or (1, 8, 1) if dKV_swapAB
            (self.AtomLayoutNdKV, self.num_mma_warps // self.AtomLayoutNdKV, 1)
            if cutlass.const_expr(not self.dKV_swapAB)
            else (self.num_mma_warps // self.AtomLayoutNdKV, self.AtomLayoutNdKV, 1)
        )
        tiled_mma_dkv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdKV,
            # (128, 16, 16) or (16, 128, 16) if dKV_swapAB
            permutation_mnk=(AtomLayoutdKV[0] * 16, AtomLayoutdKV[1] * 16, 16),
        )

        # Tiled MMA for dQ=dS*K
        # Thr Layout VMNK: (32,1,8,1):(1,0,32,0)
        # Permutation MNK: (16:1,128:1,16:1)
        # MMA Atom
        # ThrID:           32:1
        # Shape MNK:       (16,8,16)
        # TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
        # TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
        # TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
        AtomLayoutdQ = (  # (1, 8, 1) or (8, 1, 1) if dQ_swapAB
            (self.AtomLayoutMdQ, self.num_mma_warps // self.AtomLayoutMdQ, 1)
            if cutlass.const_expr(not self.dQ_swapAB)
            else (self.num_mma_warps // self.AtomLayoutMdQ, self.AtomLayoutMdQ, 1)
        )
        tiled_mma_dq = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, cutlass.Float32, (16, 8, 16)),
            AtomLayoutdQ,
            # (16, 128, 16) or (128, 16, 16) if dQ_swapAB
            permutation_mnk=(AtomLayoutdQ[0] * 16, AtomLayoutdQ[1] * 16, 16),
        )

        # --- Debug print ---

        if cutlass.const_expr(self.debug_print):
            prefix = "[bwd_sm80_get_tiled_mma] "
            print()
            print(f"{prefix}AtomLayoutSdP: {AtomLayoutSdP}")
            print(f"{prefix}AtomLayoutdKV: {AtomLayoutdKV}")
            print(f"{prefix}AtomLayoutdQ: {AtomLayoutdQ}")
            print()
            print(f"{prefix}tiled_mma_sdp: {tiled_mma_sdp}")
            print()
            print(f"{prefix}tiled_mma_dkv: {tiled_mma_dkv}")
            print()
            print(f"{prefix}tiled_mma_dq: {tiled_mma_dq}")
            print()

        return tiled_mma_sdp, tiled_mma_dkv, tiled_mma_dq

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024
            ]
            for layout in (
                self.sQ_layout,
                self.sK_layout,
                self.sV_layout,
                self.sdO_layout,
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]
        sLSE_struct, sdPsum_struct = [
            cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(layout)], 128
            ]
            for layout in (self.sLSE_layout, self.sLSE_layout)
        ]
        sP_struct, sdS_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 128
            ]
            for layout in (self.sPdS_layout, self.sPdS_layout)
        ]

        @cute.struct
        class SharedStorageSeparateQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQ_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct
            # TODO: optimize the case where there's no sP/sdS

        @cute.struct
        class SharedStorageSharedQV:
            sK: sK_struct
            sV: sV_struct
            sQ: sQV_struct
            sdO: sdO_struct
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sP: sP_struct
            sdS: sdS_struct

        self.shared_storage_cls = (
            SharedStorageSeparateQV
            if cutlass.const_expr(not self.share_QV_smem)
            else SharedStorageSharedQV
        )

    def _setup_attributes(self):
        # --- Set up tiled MMA ---

        (
            self.tiled_mma_sdp,
            self.tiled_mma_dkv,
            self.tiled_mma_dq,
        ) = self._get_tiled_mma()

        # --- Set up smem layout: sQ/sK/sV/sdO/sLSE ---

        # sQ/sdO: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ8),(ATOM_HD64,LAY_tileHD2),(1,1)):((64,512),(1,4096),(0,0))
        # sK/sV: S<3,3,3> o 0 o ((ATOM_K8,LAY_tileK16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sP/sdS: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ8),(ATOM_K64,LAY_tileK2)):((64,512),(1,4096))
        # sLSE/sdPsum: (tileQ64,1):(1,64) | sLSEMma/sdPsumMma: (tileQ64,tileHD128,1):(1,0,64)
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(
            self.dtype, self.head_dim_padded
        )
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self.m_block_size, self.head_dim_padded, self.num_stages_Q),
            (0, 1, 2),
        )
        sK_layout_atom = sQ_layout_atom
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.n_block_size, self.head_dim_padded),
            (0, 1),
        )
        sV_layout_atom = sm80_utils.get_smem_layout_atom(
            self.dtype, self.head_dim_v_padded
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.n_block_size, self.head_dim_v_padded),
            (0, 1),
        )
        sdO_layout_atom = sV_layout_atom
        self.sdO_layout = cute.tile_to_shape(
            sdO_layout_atom,
            (self.m_block_size, self.head_dim_v_padded, self.num_stages_dO),
            (0, 1, 2),
        )
        # TODO(REVIEW): do we set swizzle to be 3 here explicitly?
        sPdS_layout_atom = sm80_utils.get_smem_layout_atom(
            self.dtype, self.n_block_size
        )
        self.sPdS_layout = cute.tile_to_shape(
            sPdS_layout_atom,
            (self.m_block_size, self.n_block_size),
            (0, 1),
        )

        # We set stride to be multiple of 64 so that if ShuffleLSE,
        # even if threads read from sLSE but out of bounds,
        # it's still a valid smem address.
        self.sLSE_layout = cute.make_layout(
            (self.m_block_size, self.num_stages_Q),
            stride=(1, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout = cute.make_layout(
            (self.m_block_size, self.n_block_size, self.num_stages_Q),
            stride=(1, 0, cute.round_up(self.m_block_size, 64)),
        )
        sLSEMma_layout_transposed = cute.make_layout(
            (self.n_block_size, self.m_block_size, self.num_stages_Q),
            stride=(0, 1, cute.round_up(self.m_block_size, 64)),
        )
        self.sLSEMma_layout = (
            sLSEMma_layout if not self.SdP_swapAB else sLSEMma_layout_transposed
        )

        # --- Set up G2S/S2G/R2G tiled copy ---

        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = (
            universal_copy_bits // self.dtype.width
        )  # 8 elems per copy atom
        async_copy_elems_accum = universal_copy_bits // cutlass.Float32.width

        # Value layouts for all copies: (1,8):(0,1) => 8 bf16 elements per thread
        vQKVdO_layout = cute.make_layout((1, async_copy_elems))

        # atom_async_copy: G2S copy atom for Q/K/V/dO load with `cp.async`
        # layout_src_tv: (1,8):(0,1) => 8 bf16 elements per thread
        # layout_dst_tv: (1,8):(0,1) => 8 bf16 elements per thread
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # atom_async_copy_accum: G2S copy atom for LSE load with `cp.async`
        # layout_src_tv=(1,4):(0,1) => 4 float32 elements per thread
        # layout_dst_tv=(1,4):(0,1) => 4 float32 elements per thread
        if cutlass.const_expr(not self.varlen_q):
            atom_async_copy_accum = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.Float32,
                num_bits_per_copy=universal_copy_bits,
            )
        else:
            async_copy_elems_accum = 1
            atom_async_copy_accum = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=cutlass.Float32.width,
            )

        # atom_universal_copy: universal copy atom for dQacc/dK/dV store with `st.global`
        # layout_src_tv: (1,8):(0,1) => 8 bf16 elements per thread
        # layout_dst_tv: (1,8):(0,1) => 8 bf16 elements per thread
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # tQ/tK: (32,8):(8,1)
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert (
            self.num_threads % tQK_shape_dim_1 == 0
        ), "num_threads must be divisible by tQK_shape_dim_1"
        tQK_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )

        # tV/tdO: (32,8):(8,1)
        tVdO_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        assert (
            self.num_threads % tVdO_shape_dim_1 == 0
        ), "num_threads must be divisible by tVdO_shape_dim_1"
        tVdO_layout = cute.make_ordered_layout(
            (self.num_threads // tVdO_shape_dim_1, tVdO_shape_dim_1),
            order=(1, 0),
        )

        # G2S async tiled_copy_QKVdO:
        # layout_src_tv_tiled=((8,32),(8,1)):((256,1),(32,0))
        # layout_dst_tv_tiled=((8,32),(8,1)):((256,1),(32,0))
        self.gmem_tiled_copy_QK = cute.make_tiled_copy_tv(
            atom_async_copy, tQK_layout, vQKVdO_layout
        )
        self.gmem_tiled_copy_VdO = cute.make_tiled_copy_tv(
            atom_async_copy, tVdO_layout, vQKVdO_layout
        )

        # G2S async tiled_copy_LSE:
        # layout_src_tv_tiled=(256,(4,1)):(4,(1,0))
        # layout_dst_tv_tiled=(256,(4,1)):(4,(1,0))
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_accum,
            cute.make_layout(self.num_threads),
            cute.make_layout(async_copy_elems_accum),
        )

        # R2G universal tiled_copy_dK/dV:
        # layout_src_tv_tiled=(256,(1,1)):(1,(0,0))
        # layout_dst_tv_tiled=(256,(1,1)):(1,(0,0))
        self.gmem_tiled_copy_dK = cute.make_tiled_copy_tv(
            atom_universal_copy, tQK_layout, vQKVdO_layout
        )
        self.gmem_tiled_copy_dV = cute.make_tiled_copy_tv(
            atom_universal_copy, tVdO_layout, vQKVdO_layout
        )

        # R2G universal atomic tiled_copy_dOacc:
        # layout_src_tv_tiled=(256,(1,1)):(1,(0,0))
        # layout_dst_tv_tiled=(256,(1,1)):(1,(0,0))
        self.gmem_tiled_copy_dQacc = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float32,
                num_bits_per_copy=cutlass.Float32.width,
            ),
            cute.make_layout(self.num_threads),
            cute.make_layout(1),
        )
        if cutlass.const_expr(self.qhead_per_kvhead > 1):
            self.gmem_tiled_copy_dK = self.gmem_tiled_copy_dQacc
            self.gmem_tiled_copy_dV = self.gmem_tiled_copy_dQacc

        # --- Debug print ---

        if cutlass.const_expr(self.debug_print):
            prefix = "[bwd_sm80_setup_attributes] "
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sdO_layout: {self.sdO_layout}")
            print(f"{prefix}sPdS_layout: {self.sPdS_layout}")
            print(f"{prefix}sLSE_layout: {self.sLSE_layout}")
            print(f"{prefix}sLSEMma_layout: {self.sLSEMma_layout}")
            print()
            print(f"{prefix}tQK_layout: {tQK_layout}")
            print(f"{prefix}tVdO_layout: {tVdO_layout}")
            print(f"{prefix}vQKVdO_layout: {vQKVdO_layout}")
            print()
            print(
                f"{prefix}atom_async_copy: "
                f"layout_src_tv={atom_async_copy.layout_src_tv} | "
                f"layout_dst_tv={atom_async_copy.layout_dst_tv}"
            )
            print(
                f"{prefix}atom_universal_copy: "
                f"layout_src_tv={atom_universal_copy.layout_src_tv} | "
                f"layout_dst_tv={atom_universal_copy.layout_dst_tv}"
            )
            print(
                f"{prefix}atom_async_copy_accum: "
                f"layout_src_tv={atom_async_copy_accum.layout_src_tv} | "
                f"layout_dst_tv={atom_async_copy_accum.layout_dst_tv}"
            )
            print()
            print(
                f"{prefix}gmem_tiled_copy_QK: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_QK.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_QK.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_VdO: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_VdO.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_VdO.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_dK: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_dK.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_dK.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_dV: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_dV.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_dV.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_LSE: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_LSE.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_LSE.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_dQacc: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_dQacc.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_dQacc.layout_dst_tv_tiled}"
            )
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
        softmax_scale: cutlass.Float32,
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

        assert (
            window_size_left is None and window_size_right is None
        ), "Sliding window is not supported yet for sm80"
        assert (
            mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None
        ), "Determinism is not supported yet for sm80"
        assert aux_tensors is None, "Aux tensors are not supported yet for sm80"
        assert (
            blocksparse_tensors is None
        ), "Blocksparse tensors are not supported yet for sm80"

        self._check_tile()
        self._check_type(
            *(  # type: ignore[arg-type]
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    mdQacc,
                    mdK,
                    mdV,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )

        # --- Set up attributes ---

        self.varlen_q = mCuSeqlensQ is not None
        self._setup_attributes()

        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mO/mLSE tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        mQ, mK, mV, mdO, mLSE, mdPsum, mdQacc, mdK, mdV = [
            cutedsl_utils.assume_tensor_aligned(t)
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQacc, mdK, mdV)
        ]

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        self.tile_scheduler_cls = (
            SingleTileVarlenScheduler
            if cutlass.const_expr(mCuSeqlensK is not None)
            else SingleTileScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            # Uses seqlen k, etc. since main bwd kernel's blocks are over n
            num_block=cute.ceil_div(mK.shape[1], self.n_block_size),
            num_head=(
                mQ.shape[1]
                if cutlass.const_expr(mCuSeqlensQ is not None)
                else mQ.shape[2]
            ),
            num_batch=(
                mCuSeqlensK.shape[0] - 1  # type: ignore[union-attr]
                if cutlass.const_expr(mCuSeqlensK is not None)
                else mK.shape[0]
            ),
            num_splits=1,
            seqlen_k=0,
            headdim=mK.shape[2],
            headdim_v=mV.shape[2],
            total_q=mK.shape[0],
            tile_shape_mn=(self.n_block_size, self.m_block_size),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if cutlass.const_expr(self.pack_gqa)
            else 1,
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
        )
        tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
            tile_sched_args
        )
        grid_dim = self.tile_scheduler_cls.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        self._get_shared_storage_cls()

        # --- Make others ---

        # NB: keep softmax_scale as a real Float32 (the kernel param is typed
        # cutlass.Float32). compute_softmax_scale_log2 would null out softmax_scale
        # when score_mod is None, which then fails None->Float32 conversion at launch.
        LOG2_E = math.log2(math.e)
        if cutlass.const_expr(self.score_mod is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            softmax_scale_log2 = LOG2_E

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if cutlass.const_expr(self.debug_print):
            prefix = "[bwd_sm80_call] "

            print()
            print(f"{prefix}m_block_size: {self.m_block_size}")
            print(f"{prefix}n_block_size: {self.n_block_size}")
            print(f"{prefix}num_threads: {self.num_threads}")
            print(f"{prefix}num_stages_Q: {self.num_stages_Q}")
            print(f"{prefix}num_stages_dO: {self.num_stages_dO}")
            print(f"{prefix}pack_gqa: {self.pack_gqa}")
            print()

            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
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
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQacc,
            mdK,
            mdV,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale,
            softmax_scale_log2,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdO_layout,
            self.sPdS_layout,
            self.sLSE_layout,
            self.sLSEMma_layout,
            self.gmem_tiled_copy_QK,
            self.gmem_tiled_copy_VdO,
            self.gmem_tiled_copy_dK,
            self.gmem_tiled_copy_dV,
            self.gmem_tiled_copy_LSE,
            self.gmem_tiled_copy_dQacc,
            self.tiled_mma_sdp,
            self.tiled_mma_dkv,
            self.tiled_mma_dq,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=self.shared_storage_cls.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
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
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sLSEMma_layout: cute.Layout,
        gmem_tiled_copy_QK: cute.TiledCopy,
        gmem_tiled_copy_VdO: cute.TiledCopy,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        gmem_tiled_copy_dQacc: cute.TiledCopy,
        tiled_mma_sdp: cute.TiledMma,
        tiled_mma_dkv: cute.TiledMma,
        tiled_mma_dq: cute.TiledMma,
        tile_sched_params: ParamsBase,
    ):
        # /////////////////////////////////////////////////////////////////////////////
        #  Set up thread and tile
        # /////////////////////////////////////////////////////////////////////////////

        # --- Set up thread info ---

        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Used only for debug print
        # guarded by const_expr so zero overhead when debug_print=False
        is_print_block = const_expr(self.debug_print) and (
            (bidx == 0) and (bidy == 0) and (bidz == 0)
        )
        is_print_thread = const_expr(self.debug_print) and (
            (tidx == 63) and is_print_block
        )

        # --- Set up work tile ---

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        n_block, head_idx, batch_idx, _ = work_tile.tile_idx

        # --- Set up seqlen info ---

        seqlen_info = SeqlenInfoQK.create(
            batch_idx,
            mQ.shape[1],
            mK.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.m_block_size,
            tile_n=self.n_block_size,
        )

        # TODO: return early if m_block_max == 0
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.m_block_size)
        m_block_min = 0
        if cutlass.const_expr(self.is_causal):
            m_block_min = max(
                (
                    n_block * self.n_block_size
                    + seqlen_info.seqlen_q
                    - seqlen_info.seqlen_k
                )
                // self.m_block_size,
                m_block_min,
            )

        # NOTE: Start async loads of the last mn-tile, where we take care of the mn residue
        m_block = m_block_min

        d_head = mQ.shape[cute.rank(mQ) - 1]
        d_head_v = mdO.shape[cute.rank(mdO) - 1]
        head_idx_kv = (
            head_idx // self.qhead_per_kvhead
            if cutlass.const_expr(not self.pack_gqa)
            else head_idx
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make gmem tiles for Q/K/V/dO/LSE/dPsum/dQacc
        # ///////////////////////////////////////////////////////////////////////////////

        # mQ_cur/mdO_cur: (sQ,HD):(HD*nhQ,1)
        # mK_cur/mV_cur: (sK,HD):(HD*nhK,1)
        # mLSE_cur/mdPsum_cur: (sQ):(1)
        # mdQacc_cur: (sQ*HD):(1)
        blkQ_shape = (self.m_block_size, self.head_dim_padded)
        blkK_shape = (self.n_block_size, self.head_dim_padded)
        blkV_shape = (self.n_block_size, self.head_dim_v_padded)
        blkdO_shape = (self.m_block_size, self.head_dim_v_padded)
        if cutlass.const_expr(not seqlen_info.has_cu_seqlens_q):
            mQ_cur = mQ[batch_idx, None, head_idx, None]
            mLSE_cur = mLSE[batch_idx, head_idx, None]
            mdO_cur = mdO[batch_idx, None, head_idx, None]
            mdPsum_cur = mdPsum[batch_idx, head_idx, None]
            mdQacc_cur = mdQacc[batch_idx, head_idx, None]
        else:
            padded_offset_q = seqlen_info.padded_offset_q
            mQ_cur = cute.domain_offset(
                (seqlen_info.offset_q, 0), mQ[None, head_idx, None]
            )
            mLSE_cur = cute.domain_offset((padded_offset_q,), mLSE[head_idx, None])
            mdO_cur = cute.domain_offset(
                (seqlen_info.offset_q, 0), mdO[None, head_idx, None]
            )
            mdPsum_cur = cute.domain_offset((padded_offset_q,), mdPsum[head_idx, None])
            mdQacc_cur = cute.domain_offset(
                (padded_offset_q * self.head_dim_padded,), mdQacc[head_idx, None]
            )
        if cutlass.const_expr(not seqlen_info.has_cu_seqlens_k):
            mK_cur, mV_cur = [t[batch_idx, None, head_idx_kv, None] for t in (mK, mV)]
        else:
            mK_cur, mV_cur = [
                cute.domain_offset(
                    (seqlen_info.offset_k, 0), t[None, head_idx_kv, None]
                )
                for t in (mK, mV)
            ]

        # gQ/gdO: (tileQ64,tileHD128,restQ):(HD*nhQ,1,HD*nhQ*tileQ)
        # gK/gV: (tileK128,tileHD128):(HD*nhK,1)
        # gLSE/gdPsum: (tileQ64,restQ):(1,tileQ)
        # gdQacc: (tileQ64*tileHD128,restQ):(1,tileQ*tileHD)
        # where restQ = sQ // tileQ64
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (None, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (n_block, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (n_block, 0))
        gdO = cute.local_tile(mdO_cur, blkdO_shape, (None, 0))
        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (None,))
        gdPsum = cute.local_tile(mdPsum_cur, (self.m_block_size,), (None,))
        gdQacc = cute.local_tile(
            mdQacc_cur, (self.m_block_size * self.head_dim_padded,), (None,)
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Alloc smem storage and make smem tensors for sQ/sK/sV
        # ///////////////////////////////////////////////////////////////////////////////

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)

        # sQ/sdO: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ8),(ATOM_HD64,LAY_tileHD2),(1,1)):((64,512),(1,4096),(0,0))
        # sK/sV: S<3,3,3> o 0 o ((ATOM_K8,LAY_tileK16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sP/sdS: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ8),(ATOM_K64,LAY_tileK2)):((64,512),(1,4096))
        # sLSE/sdPsum: (tileQ64,1):(1,64) | sLSEMma/sdPsumMma: (tileQ64,tileHD128,1):(1,0,64)
        sQ: cute.Tensor = storage.sQ.get_tensor(sQ_layout)
        sK: cute.Tensor = storage.sK.get_tensor(sK_layout)
        if cutlass.const_expr(not self.share_QV_smem):
            sV: cute.Tensor = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
            )
        sdO: cute.Tensor = storage.sdO.get_tensor(sdO_layout)
        sP: cute.Tensor = storage.sP.get_tensor(sPdS_layout)
        sdS: cute.Tensor = storage.sdS.get_tensor(sPdS_layout)
        sLSE: cute.Tensor = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum: cute.Tensor = storage.sdPsum.get_tensor(sLSE_layout)
        sLSEMma: cute.Tensor = storage.sLSE.get_tensor(sLSEMma_layout)
        sdPsumMma: cute.Tensor = storage.sdPsum.get_tensor(sLSEMma_layout)
        # Transpose view of tensors for tiled mma
        sQt, sdOt, sKt, sPt, sdSt = [
            layout_utils.transpose_view(t) for t in (sQ, sdO, sK, sP, sdS)
        ]
        # Reuse sK/sV buffer for sdK/sdV
        sdK = cute.make_tensor(sK.iterator, sK_layout)
        sdV = cute.make_tensor(sV.iterator, sV_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # G2S/S2G tiled copy partitions for Q/K/V/dO/LSE/dPsum/dQacc
        # ///////////////////////////////////////////////////////////////////////////////

        # tQgQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,restQ):((1,0),32768,64,65536)
        # tQsQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,STAGE=(1,1)):((1,0),2048,4096,(0,0))
        # tKgK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),8192,64)
        # tKsK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),2048,8192)
        gmem_thr_copy_QK = gmem_tiled_copy_QK.get_slice(tidx)
        tQgQ = gmem_thr_copy_QK.partition_S(gQ)
        tQsQ = gmem_thr_copy_QK.partition_D(sQ)
        tKgK = gmem_thr_copy_QK.partition_S(gK)
        tKsK = gmem_thr_copy_QK.partition_D(sK)

        # tVgV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),8192,64)
        # tVsV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),2048,8192)
        # tdOgdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,restQ):((1,0),32768,64,65536)
        # tdOsdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,STAGE=(1,1)):((1,0),2048,4096,(0,0))
        gmem_thr_copy_VdO = gmem_tiled_copy_VdO.get_slice(tidx)
        tVgV = gmem_thr_copy_VdO.partition_S(gV)
        tVsV = gmem_thr_copy_VdO.partition_D(sV)
        tdOgdO = gmem_thr_copy_VdO.partition_S(gdO)
        tdOsdO = gmem_thr_copy_VdO.partition_D(sdO)

        # tLSEgLSE: (CPY_ATOM=(4,1),CPY_Q1,restQ):((1,0),0,64)
        # tLSEsLSE: (CPY_ATOM=(4,1),CPY_Q1,STAGE1):((1,0),0,64)
        # tLSEgdPsum: (CPY_ATOM=(4,1),CPY_Q1,restQ):((1,0),0,64)
        # tLSEsdPsum: (CPY_ATOM=(4,1),CPY_Q1,STAGE1):((1,0),0,64)
        gmem_thr_copy_lse = gmem_tiled_copy_LSE.get_slice(tidx)
        tLSEgLSE = gmem_thr_copy_lse.partition_S(gLSE)
        tLSEsLSE = gmem_thr_copy_lse.partition_D(sLSE)
        tLSEgdPsum = gmem_thr_copy_lse.partition_S(gdPsum)
        tLSEsdPsum = gmem_thr_copy_lse.partition_D(sdPsum)

        # tdQgdQacc: (CPY_ATOM=(1,1),CPY_V32,restQ):((0,0),256,8192)
        gmem_thr_copy_dQacc = gmem_tiled_copy_dQacc.get_slice(tidx)
        tdQgdQacc = gmem_thr_copy_dQacc.partition_S(gdQacc)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////

        # tSrQ: (MMA_ATOM=(2,2,2),MMA_Q4,MMA_HD=((2,2),2)):((1,2,4),8,((64,128),32))
        # tSrK: (MMA_ATOM=(2,2),MMA_K2,MMA_HD=((2,2),2)):((1,2),4,((16,32),8))
        # tdPrdO: (MMA_ATOM=(2,2,2),MMA_Q4,MMA_HD=((2,2),2)):((1,2,4),8,((64,128),32))
        # tdPrV: (MMA_ATOM=(2,2),MMA_K2,MMA_HD=((2,2),2)):((1,2),4,((16,32),8))
        thr_mma_sdp = tiled_mma_sdp.get_slice(tidx)
        tSrQ = cutedsl_utils.mma_make_fragment_A(
            sQ[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB
        )
        tSrK = cutedsl_utils.mma_make_fragment_B(
            sK, thr_mma_sdp, swapAB=self.SdP_swapAB
        )
        tdPrdO = cutedsl_utils.mma_make_fragment_A(
            sdO[None, None, 0], thr_mma_sdp, swapAB=self.SdP_swapAB
        )
        tdPrV = cutedsl_utils.mma_make_fragment_B(
            sV, thr_mma_sdp, swapAB=self.SdP_swapAB
        )

        # tdVrP: (MMA_ATOM=(2,2,2),MMA_K1,MMA_Q4):((1,2,4),0,8)
        # tdVrdO: (MMA_ATOM=(2,2),MMA_HD=(8,2),MMA_Q4):((1,2),(4,128),32)
        # tdKrdS: (MMA_ATOM=(2,2,2),MMA_K1,MMA_Q4):((1,2,4),0,8)
        # tdKrQ: (MMA_ATOM=(2,2),MMA_HD=(8,2),MMA_Q4):((1,2),(4,128),32)
        thr_mma_dkv = tiled_mma_dkv.get_slice(tidx)
        tdVrP = cutedsl_utils.mma_make_fragment_A(
            sPt, thr_mma_dkv, swapAB=self.dKV_swapAB
        )
        tdVrdO = cutedsl_utils.mma_make_fragment_B(
            sdOt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB
        )
        tdKrdS = cutedsl_utils.mma_make_fragment_A(
            sdSt, thr_mma_dkv, swapAB=self.dKV_swapAB
        )
        tdKrQ = cutedsl_utils.mma_make_fragment_B(
            sQt[None, None, 0], thr_mma_dkv, swapAB=self.dKV_swapAB
        )

        # acc_dK/dV: (MMA_ATOM=(2,2),MMA_K1,MMA_HD16):((1,2),0,4)
        acc_shape_dK = thr_mma_dkv.partition_shape_C(
            (self.n_block_size, self.head_dim_padded)
        )
        acc_shape_dV = thr_mma_dkv.partition_shape_C(
            (self.n_block_size, self.head_dim_v_padded)
        )
        acc_dK = cute.make_rmem_tensor(acc_shape_dK, cutlass.Float32)
        acc_dV = cute.make_rmem_tensor(acc_shape_dV, cutlass.Float32)
        acc_dK.fill(0.0)
        acc_dV.fill(0.0)

        # tdQrdS: (MMA_ATOM=(2,2,2),MMA_Q4,MMA_K=((2,2),2)):((1,2,4),8,((64,128),32))
        # tdQrK: (MMA_ATOM=(2,2),MMA_HD2,MMA_K8):((1,2),32,4)
        thr_mma_dq = tiled_mma_dq.get_slice(tidx)
        tdQrdS = cutedsl_utils.mma_make_fragment_A(
            sdS, thr_mma_dq, swapAB=self.dQ_swapAB
        )
        tdQrK = cutedsl_utils.mma_make_fragment_B(
            sKt, thr_mma_dq, swapAB=self.dQ_swapAB
        )

        # tSsLSEMma_/tSsdPsumMma_: (MMA_ATOM=(ATOM_Q2,ATOM_K2),MMA_Q4,MMA_K2,STAGE1):((0,8),16,0,64)
        # tSsLSEMma/tSsdPsumMma: (MMA_ATOM=(ATOM_Q2,MMA_Q4),1):((8,16),0)
        tSsLSEMma_ = thr_mma_sdp.partition_C(sLSEMma)
        tSsdPsumMma_ = thr_mma_sdp.partition_C(sdPsumMma)
        LSEslice = (
            (None, 0, None)
            if cutlass.const_expr(not self.SdP_swapAB)
            else (0, None, None)
        )
        tSsLSEMma = layout_utils.reshape_acc_to_mn(tSsLSEMma_)[LSEslice]
        tSsdPsumMma = layout_utils.reshape_acc_to_mn(tSsdPsumMma_)[LSEslice]

        # ///////////////////////////////////////////////////////////////////////////////
        # Make S2R/R2S tiled copy and partitions for Q/K/V/dO/P/dS
        # ///////////////////////////////////////////////////////////////////////////////

        # S2R copy atom for Q/K/V/dO with `ldmatrix.sync.aligned.m8n8.x4` => m32xn8
        # layout_src_tv=(32,8):(8,1)
        # layout_dst_tv=(32,(2,4)):(2,(1,64))
        smem_copy_atom = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )

        # S2R copy atom for Pt/dSt/Qt/dOt/Kt with `ldmatrix.sync.aligned.m8n8.x4.trans` => m8xn32
        # layout_src_tv=(32,8):(8,1)
        # layout_dst_tv=((4,8),(1,2,4)):((16,1),(1,8,64))
        smem_copy_atom_trans = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )

        # R2S copy atom for P/dS with universal `st.shared`
        # layout_src_tv=(1,2):(0,1)
        # layout_dst_tv=(1,2):(0,1)
        r2s_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            # TODO(REVIEW): what's the number of bits? What if SdP_swapAB
            num_bits_per_copy=2 * self.dtype.width,
        )

        # tSsQ/tdPsdO: (CPY_ATOM=(8,1),CPY_Q4,CPY_HD=((2,2),2),STAGE=(1,1)):((1,0),1024,((-16,-32),4096),(0,0))
        smem_thr_copy_QdO = cutedsl_utils.make_tiled_copy_A(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        tSsQ = smem_thr_copy_QdO.partition_S(sQ)
        tdPsdO = smem_thr_copy_QdO.partition_S(sdO)

        # tSsK/tdPsV: (CPY_ATOM=(8,1),CPY_K1,CPY_HD=((2,2),2)):((1,0),0,((-16,-32),8192))
        smem_thr_copy_KV = cutedsl_utils.make_tiled_copy_B(
            smem_copy_atom, tiled_mma_sdp, swapAB=self.SdP_swapAB
        ).get_slice(tidx)
        tSsK = smem_thr_copy_KV.partition_S(sK)
        tdPsV = smem_thr_copy_KV.partition_S(sV)

        # tdVsPt/tdKsdSt: (CPY_ATOM=(8,1),CPY_K1,CPY_Q4):((1,0),0,1024)
        # TODO(REVIEW): should this be smem_copy_atom_transposed?
        smem_thr_copy_PdSt = cutedsl_utils.make_tiled_copy_A(
            smem_copy_atom_trans, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        tdVsPt = smem_thr_copy_PdSt.partition_S(sPt)
        tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt)

        # tdVsdOt/tdKsQt: (CPY_ATOM=(8,1),CPY_HD=((2,2),2),CPY_Q4,STAGE=(1,1)):((1,0),((-16,-32),4096),1024,(0,0))
        smem_thr_copy_QdOt = cutedsl_utils.make_tiled_copy_B(
            smem_copy_atom_trans, tiled_mma_dkv, swapAB=self.dKV_swapAB
        ).get_slice(tidx)
        tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt)
        tdKsQt = smem_thr_copy_QdOt.partition_S(sQt)

        # tdQsdS: (CPY_ATOM=(8,1),CPY_Q4,CPY_K=((2,2),2)):((1,0),1024,((-16,-32),4096))
        smem_thr_copy_dS = cutedsl_utils.make_tiled_copy_A(
            smem_copy_atom, tiled_mma_dq, swapAB=self.dQ_swapAB
        ).get_slice(tidx)
        tdQsdS = smem_thr_copy_dS.partition_S(sdS)

        # tdQsKt: (CPY_ATOM=(8,1),CPY_HD1,CPY_K8):((1,0),0,1024)
        smem_thr_copy_Kt = cutedsl_utils.make_tiled_copy_B(
            smem_copy_atom_trans, tiled_mma_dq, swapAB=self.dQ_swapAB
        ).get_slice(tidx)
        tdQsKt = smem_thr_copy_Kt.partition_S(sKt)

        # tPsP/tdSsdS: (CPY_ATOM=(2,(2,2)),CPY_Q4,CPY_K1):((1,(512,4096)),1024,0)
        r2s_thr_copy_PdS = cute.make_tiled_copy_C(
            r2s_copy_atom,
            tiled_mma_sdp,
        ).get_slice(tidx)
        tPsP = r2s_thr_copy_PdS.partition_D(sP)
        tdSsdS = r2s_thr_copy_PdS.partition_D(sdS)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make predicate tensors for Q/LSE G2S loads
        # ///////////////////////////////////////////////////////////////////////////////

        # cQ: (tileQ64,tileHD128):(1@0,1@1)
        # tQcQ/t0QcQ/tdOcdO/t0dOcdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2):((1@1,0),32@0,64@1)
        # cLSE: (tileQ64):(1@0)
        # tLSEcLSE/tdPsumcdPsum: (CPY_ATOM=(4,1),CPY_Q=(1)):((1@0,0),(0))
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQcQ = gmem_thr_copy_QK.partition_S(cQ)
        t0QcQ = gmem_thr_copy_QK.get_slice(0).partition_S(cQ)
        if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
            tdOcdO = tQcQ
            t0dOcdO = t0QcQ
        else:
            cdO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
            tdOcdO = gmem_thr_copy_VdO.partition_S(cdO)
            t0dOcdO = gmem_thr_copy_VdO.get_slice(0).partition_S(cdO)
        cLSE = cute.make_identity_tensor((self.m_block_size,))
        tLSEcLSE = gmem_thr_copy_lse.partition_S(cLSE)
        tdPsumcdPsum = tLSEcLSE

        # tQpQ/tdOpdO: (ATOM_REST_V1,CPY_Q2,CPY_HD2):(2,0,1) => the same predicate along CPY_Q
        tQpQ = cutedsl_utils.predicate_k(tQcQ, limit=d_head)
        if cutlass.const_expr(self.same_hdim_kv):
            tdOpdO = tQpQ
        else:
            tdOpdO = cutedsl_utils.predicate_k(tdOcdO, limit=d_head_v)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make partial functions for Q,LSE/dO,dPsum loads ---

        load_Q_LSE = partial(
            self.load_Q_LSE,
            gmem_tiled_copy_QK,
            gmem_tiled_copy_LSE,
            tQgQ,
            tQsQ,
            tQcQ,
            t0QcQ,
            tQpQ,
            tLSEgLSE,
            tLSEsLSE,
            tLSEcLSE,
            seqlen=seqlen_info.seqlen_q,
        )
        load_dO_dPsum = partial(
            self.load_dO_dPsum,
            gmem_tiled_copy_VdO,
            gmem_tiled_copy_LSE,
            tdOgdO,
            tdOsdO,
            tdOcdO,
            t0dOcdO,
            tdOpdO,
            tLSEgdPsum,
            tLSEsdPsum,
            tdPsumcdPsum,
            seqlen=seqlen_info.seqlen_q,
        )

        # --- Make partial functions for compute_one_m_block ---

        mma_params = SimpleNamespace(
            thr_mma_sdp=thr_mma_sdp,
            thr_mma_dkv=thr_mma_dkv,
            thr_mma_dq=thr_mma_dq,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tdPrdO=tdPrdO,
            tdPrV=tdPrV,
            tdVrP=tdVrP,
            tdVrdO=tdVrdO,
            tdKrdS=tdKrdS,
            tdKrQ=tdKrQ,
            tdQrdS=tdQrdS,
            tdQrK=tdQrK,
            acc_dK=acc_dK,
            acc_dV=acc_dV,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_QdO=smem_thr_copy_QdO,
            smem_thr_copy_KV=smem_thr_copy_KV,
            smem_thr_copy_PdSt=smem_thr_copy_PdSt,
            smem_thr_copy_QdOt=smem_thr_copy_QdOt,
            smem_thr_copy_dS=smem_thr_copy_dS,
            smem_thr_copy_Kt=smem_thr_copy_Kt,
            r2s_thr_copy_PdS=r2s_thr_copy_PdS,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tdPsdO=tdPsdO,
            tdPsV=tdPsV,
            tSsLSEMma=tSsLSEMma,
            tSsdPsumMma=tSsdPsumMma,
            tPsP=tPsP,
            tdSsdS=tdSsdS,
            tdVsPt=tdVsPt,
            tdVsdOt=tdVsdOt,
            tdKsdSt=tdKsdSt,
            tdKsQt=tdKsQt,
            tdQsdS=tdQsdS,
            tdQsKt=tdQsKt,
        )
        gmem_copy_params = SimpleNamespace(
            gmem_thr_copy_dQacc=gmem_thr_copy_dQacc, tdQgdQacc=tdQgdQacc
        )
        compute_one_m_block = partial(
            self.compute_one_m_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            gmem_copy_params=gmem_copy_params,
            load_Q_LSE=load_Q_LSE,
            load_dO_dPsum=load_dO_dPsum,
            m_block_max=m_block_max,
            softmax_scale=softmax_scale,
            softmax_scale_log2=softmax_scale_log2,
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread:
                prefix = "[bwd_sm80_kernel_setup] "
                cute.printf("")
                cute.printf(
                    prefix
                    + "bidx={}, bidy={}, bidz={}, tidx={}, n_block={}, head_idx={}, batch_idx={}",
                    bidx,
                    bidy,
                    bidz,
                    tidx,
                    n_block,
                    head_idx,
                    batch_idx,
                )
                cute.printf(
                    prefix + "m_block_min={}, m_block_max={}",
                    m_block_min,
                    m_block_max,
                )
                cute.printf("")
                cute.printf(prefix + "mQ_cur: {}", mQ_cur.layout)
                cute.printf(prefix + "mK_cur: {}", mK_cur.layout)
                cute.printf(prefix + "mV_cur: {}", mV_cur.layout)
                cute.printf(prefix + "mdO_cur: {}", mdO_cur.layout)
                cute.printf(prefix + "mLSE_cur: {}", mLSE_cur.layout)
                cute.printf(prefix + "mdPsum_cur: {}", mdPsum_cur.layout)
                cute.printf(prefix + "mdQacc_cur: {}", mdQacc_cur.layout)
                cute.printf("")
                cute.printf(prefix + "gQ: {}", gQ.layout)
                cute.printf(prefix + "gK: {}", gK.layout)
                cute.printf(prefix + "gV: {}", gV.layout)
                cute.printf(prefix + "gdO: {}", gdO.layout)
                cute.printf(prefix + "gLSE: {}", gLSE.layout)
                cute.printf(prefix + "gdPsum: {}", gdPsum.layout)
                cute.printf(prefix + "gdQacc: {}", gdQacc.layout)
                cute.printf("")
                cute.printf(prefix + "sQ: {}", sQ.layout)
                cute.printf(prefix + "sK: {}", sK.layout)
                cute.printf(prefix + "sV: {}", sV.layout)
                cute.printf(prefix + "sdO: {}", sdO.layout)
                cute.printf(prefix + "sP: {}", sP.layout)
                cute.printf(prefix + "sdS: {}", sdS.layout)
                cute.printf(prefix + "sLSE: {}", sLSE.layout)
                cute.printf(prefix + "sdPsum: {}", sdPsum.layout)
                cute.printf("")
                cute.printf(prefix + "sQt: {}", sQt.layout)
                cute.printf(prefix + "sdOt: {}", sdOt.layout)
                cute.printf(prefix + "sKt: {}", sKt.layout)
                cute.printf(prefix + "sPt: {}", sPt.layout)
                cute.printf(prefix + "sdSt: {}", sdSt.layout)
                cute.printf("")
                cute.printf(prefix + "tQgQ: {}", tQgQ.layout)
                cute.printf(prefix + "tQsQ: {}", tQsQ.layout)
                cute.printf(prefix + "tKgK: {}", tKgK.layout)
                cute.printf(prefix + "tKsK: {}", tKsK.layout)
                cute.printf(prefix + "tVgV: {}", tVgV.layout)
                cute.printf(prefix + "tVsV: {}", tVsV.layout)
                cute.printf(prefix + "tLSEgLSE: {}", tLSEgLSE.layout)
                cute.printf(prefix + "tLSEsLSE: {}", tLSEsLSE.layout)
                cute.printf(prefix + "tLSEgdPsum: {}", tLSEgdPsum.layout)
                cute.printf(prefix + "tLSEsdPsum: {}", tLSEsdPsum.layout)
                cute.printf(prefix + "tdOgdO: {}", tdOgdO.layout)
                cute.printf(prefix + "tdOsdO: {}", tdOsdO.layout)
                cute.printf(prefix + "tdQgdQacc: {}", tdQgdQacc.layout)
                cute.printf("")
                cute.printf(prefix + "acc_dK: {}", acc_dK.layout)
                cute.printf(prefix + "acc_dV: {}", acc_dV.layout)
                cute.printf(prefix + "tSrQ: {}", tSrQ.layout)
                cute.printf(prefix + "tSrK: {}", tSrK.layout)
                cute.printf(prefix + "tdPrdO: {}", tdPrdO.layout)
                cute.printf(prefix + "tdPrV: {}", tdPrV.layout)
                cute.printf(prefix + "tdVrP: {}", tdVrP.layout)
                cute.printf(prefix + "tdVrdO: {}", tdVrdO.layout)
                cute.printf(prefix + "tdKrdS: {}", tdKrdS.layout)
                cute.printf(prefix + "tdKrQ: {}", tdKrQ.layout)
                cute.printf(prefix + "tdQrdS: {}", tdQrdS.layout)
                cute.printf(prefix + "tdQrK: {}", tdQrK.layout)
                cute.printf(prefix + "tSsLSEMma_: {}", tSsLSEMma_.layout)
                cute.printf(prefix + "tSsdPsumMma_: {}", tSsdPsumMma_.layout)
                cute.printf(prefix + "tSsLSEMma: {}", tSsLSEMma.layout)
                cute.printf(prefix + "tSsdPsumMma: {}", tSsdPsumMma.layout)
                cute.printf("")
                cute.printf(
                    prefix + "smem_copy_atom: layout_src_tv={}, layout_dst_tv={}",
                    smem_copy_atom.layout_src_tv,
                    smem_copy_atom.layout_dst_tv,
                )
                cute.printf(
                    prefix + "smem_copy_atom_trans: layout_src_tv={}, layout_dst_tv={}",
                    smem_copy_atom_trans.layout_src_tv,
                    smem_copy_atom_trans.layout_dst_tv,
                )
                cute.printf(
                    prefix + "r2s_copy_atom: layout_src_tv={}, layout_dst_tv={}",
                    r2s_copy_atom.layout_src_tv,
                    r2s_copy_atom.layout_dst_tv,
                )
                cute.printf(prefix + "tSsQ: {}", tSsQ.layout)
                cute.printf(prefix + "tSsK: {}", tSsK.layout)
                cute.printf(prefix + "tdPsdO: {}", tdPsdO.layout)
                cute.printf(prefix + "tdPsV: {}", tdPsV.layout)
                cute.printf(prefix + "tdVsPt: {}", tdVsPt.layout)
                cute.printf(prefix + "tdVsdOt: {}", tdVsdOt.layout)
                cute.printf(prefix + "tdKsdSt: {}", tdKsdSt.layout)
                cute.printf(prefix + "tdKsQt: {}", tdKsQt.layout)
                cute.printf(prefix + "tdQsdS: {}", tdQsdS.layout)
                cute.printf(prefix + "tdQsKt: {}", tdQsKt.layout)
                cute.printf(prefix + "tPsP: {}", tPsP.layout)
                cute.printf(prefix + "tdSsdS: {}", tdSsdS.layout)
                cute.printf("")
                cute.printf(prefix + "cQ: {}", cQ.layout)
                cute.printf(prefix + "tQcQ: {}", tQcQ.layout)
                cute.printf(prefix + "t0QcQ: {}", t0QcQ.layout)
                cute.printf(prefix + "tdOcdO: {}", tdOcdO.layout)
                cute.printf(prefix + "t0dOcdO: {}", t0dOcdO.layout)
                cute.printf(prefix + "cLSE: {}", cLSE.layout)
                cute.printf(prefix + "tLSEcLSE: {}", tLSEcLSE.layout)
                cute.printf(prefix + "tdPsumcdPsum: {}", tdPsumcdPsum.layout)
                cute.printf(prefix + "tQpQ: {}", tQpQ.layout)
                cute.printf(prefix + "tdOpdO: {}", tdOpdO.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue: Load sK/sV, and one full stages of sQ/sLSE/sdO/sdPsum
        # ///////////////////////////////////////////////////////////////////////////////

        # Load sV
        self.load_V(
            gmem_thr_copy_VdO,
            tVgV,
            tVsV,
            n_block,
            seqlen=seqlen_info.seqlen_k,
            headdim=d_head_v,
            is_print_thread_and_tile=is_print_thread,
        )
        if cutlass.const_expr(self.V_in_regs):
            cute.arch.cp_async_commit_group()

        # Load sK
        self.load_K(
            gmem_thr_copy_QK,
            tKgK,
            tKsK,
            n_block,
            seqlen=seqlen_info.seqlen_k,
            headdim=d_head,
            is_print_thread_and_tile=is_print_thread,
        )
        cute.arch.cp_async_commit_group()

        # S2R copy sV to rV if V_in_regs
        if cutlass.const_expr(self.V_in_regs):
            # Wait for sV load to finish before S2R copy
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()

            # S2R copy rotated V from smem buffer that Q/V share to rmem
            tdPrV_copy_view = smem_thr_copy_KV.retile(tdPrV)
            cute.copy(smem_thr_copy_KV, tdPsV, tdPrV_copy_view)

            # Make sure all threads have read smem before loading Q
            cute.arch.barrier()

        # Load sQ,sLSE/sdO,sdPsum for one full stages
        assert self.num_stages_Q >= self.num_stages_dO
        for stage in cutlass.range_constexpr(self.num_stages_Q):
            if cutlass.const_expr(
                self.num_stages_Q == 1 or stage < self.num_stages_Q - 1
            ):
                if stage == 0 or m_block + stage < m_block_max:
                    load_Q_LSE(
                        m_block + stage,
                        smem_pipe_write_q=stage,
                        is_print_thread_and_tile=is_print_thread and stage == 0,
                    )
                cute.arch.cp_async_commit_group()

            if cutlass.const_expr(stage < self.num_stages_dO):
                if stage == 0 or m_block + stage < m_block_max:
                    load_dO_dPsum(
                        m_block + stage,
                        smem_pipe_write_q=stage,
                        is_print_thread_and_tile=is_print_thread and stage == 0,
                    )
                cute.arch.cp_async_commit_group()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop: Compute each m block iteration of
        #   1. recompute with softmax: S=Q*K^T, P=softmax(S)=exp(S - LSE)
        #   2. backward before softmax: dV=P^T*dO, dP=dO*V^T
        #   3. backward of softmax: dS=P*(dP-sum(dP*P))=P*(dP-sum(dO*O))=P*(dP-dPsum))
        #   4. backward after softmax: dK=dS^T*Q, dQ=dS*K
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make mask object and partial fn ---

        # NOTE: use_r2p=False because the SM80 backward SdP MMA tiles the N (key)
        # dimension across multiple warp-columns (n_block_size=128 over 8 warps),
        # which the R2P bitmask fast path does not handle (it ignores each warp's
        # column offset). Thus fall back to the layout-agnostic per-column mask path.
        mask = AttentionMask(
            self.m_block_size, self.n_block_size, seqlen_info, use_r2p=False
        )
        mask_fn = partial(
            mask.apply_mask,
            n_block=n_block,
            thr_mma=thr_mma_sdp,
            batch_idx=batch_idx,
            head_idx=head_idx,
            mask_seqlen=True,
            mask_causal=self.is_causal,
        )

        # --- compute each m block iteration ---

        smem_pipe_read_q = cutlass.Int32(0)
        smem_pipe_read_do = cutlass.Int32(0)
        smem_pipe_write_q = cutlass.Int32(self.num_stages_Q - 1)
        smem_pipe_write_do = cutlass.Int32(0)
        for m_tile in cutlass.range(m_block_min, m_block_max, unroll=1):
            compute_one_m_block(
                m_tile,
                smem_pipe_read_q,
                smem_pipe_read_do,
                smem_pipe_write_q,
                smem_pipe_write_do,
                mask_fn=mask_fn,
                is_print_thread_and_tile=is_print_thread and m_tile == m_block_min,
            )
            smem_pipe_read_q = self.advance_pipeline(
                smem_pipe_read_q, self.num_stages_Q
            )
            smem_pipe_read_do = self.advance_pipeline(
                smem_pipe_read_do, self.num_stages_dO
            )
            smem_pipe_write_q = self.advance_pipeline(
                smem_pipe_write_q, self.num_stages_Q
            )
            smem_pipe_write_do = self.advance_pipeline(
                smem_pipe_write_do, self.num_stages_dO
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////

        # NOTE: If GQA, we scale dK in the postprocessing kernel instead
        if cutlass.const_expr(self.qhead_per_kvhead == 1):
            acc_dK.store(acc_dK.load() * softmax_scale)

        self.epilogue(
            acc_dK,
            acc_dV,
            mdK,
            mdV,
            sdK,
            sdV,
            gmem_tiled_copy_dK,
            gmem_tiled_copy_dV,
            tiled_mma_dkv,
            tidx,
            n_block,
            head_idx,
            batch_idx,
            seqlen_info,
            d_head,
            d_head_v,
            is_print_thread_and_tile=is_print_thread,
        )

    @cute.jit
    def compute_one_m_block(
        self,
        m_block: cutlass.Int32,
        smem_pipe_read_q: cutlass.Int32,
        smem_pipe_read_do: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        smem_pipe_write_do: cutlass.Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        load_Q_LSE: Callable,
        load_dO_dPsum: Callable,
        m_block_max: cutlass.Int32,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        mask_fn: Optional[Callable] = None,
        is_print_thread_and_tile: bool = False,
    ):
        # Define some helper functions
        # NOTE:
        #   1. if num_stages_Q > 1, we load next Q/LSE tile after dP MMA
        #       and next dO/dPsum tile after dQ MMA
        #   2. if num_stages_Q == 1, we load next Q/LSE tile after dQ MMA
        #       and next dO/dPsum tile after dK MMA
        def load_Q_LSE_next():
            m_block_next = m_block + (
                self.num_stages_Q - 1
                if cutlass.const_expr(self.num_stages_Q > 1)
                else 1
            )
            if m_block_next < m_block_max:
                load_Q_LSE(m_block_next, smem_pipe_write_q)
            cute.arch.cp_async_commit_group()

        def load_dO_dPsum_next():
            if m_block + self.num_stages_dO < m_block_max:
                load_dO_dPsum(m_block + self.num_stages_dO, smem_pipe_write_do)
            cute.arch.cp_async_commit_group()

        # --- Apply S = Q*K^T ---

        # Zero-init acc_S
        # acc_S: (MMA_ATOM=(2,2),MMA_Q4,MMA_K2):((1,2),4,16)
        acc_shape_SdP = mma_params.thr_mma_sdp.partition_shape_C(
            (self.m_block_size, self.n_block_size)
            if cutlass.const_expr(not self.SdP_swapAB)
            else (self.n_block_size, self.m_block_size)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_S.fill(0.0)

        # Wait for this Q/LSE tile
        cute.arch.cp_async_wait_group(
            1 if cutlass.const_expr(self.num_stages_Q > 1) else 0
        )
        cute.arch.barrier()

        # Issue MMA for S = Q*K^T
        sm80_utils.gemm(
            tiled_mma=mma_params.thr_mma_sdp,
            acc=acc_S,
            tCrA=mma_params.tSrQ,
            tCrB=mma_params.tSrK,
            tCsA=smem_copy_params.tSsQ[
                None,
                None,
                None,
                smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0,
            ],
            tCsB=smem_copy_params.tSsK,
            smem_thr_copy_A=smem_copy_params.smem_thr_copy_QdO,
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_KV,
            swap_AB=self.SdP_swapAB,
        )

        # --- Apply P = softmax(S) = exp(S - LSE) ---

        # Reshape from `zipped_divide` view to `logical_divide` view
        # acc_S/P_mn: ((ATOM_Q2,MMA_Q4),(ATOM_K2,MMA_K2)):((2,4),(1,16))
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_P, acc_P_mn = acc_S, acc_S_mn
        num_rows = cute.size(acc_S_mn, mode=[0])

        acc_S_pre_mn = None
        if const_expr(self.score_mod_bwd is not None):
            # Fork acc_S_pre from acc_S for score_mod_bwd, which needs the original S before softmax
            # and acc_S will store P in-place
            acc_S_pre = cute.make_fragment_like(acc_S)
            acc_S_pre.store(acc_S.load())
            acc_S_pre_mn = layout_utils.reshape_acc_to_mn(acc_S_pre)

        # S2R copy LSE
        # tSsLSEMma: (MMA_ATOM=(ATOM_Q2,MMA_Q4),1):((8,16),0)
        # tLSErLSE: (ATOM_Q2,MMA_Q4):((1,2))
        tLSErLSE: cute.Tensor = cute.make_fragment_like(
            smem_copy_params.tSsLSEMma[None, 0]
        )
        cute.autovec_copy(
            smem_copy_params.tSsLSEMma[
                None,
                smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0,
            ],
            tLSErLSE,
        )
        assert cute.size(tLSErLSE) == num_rows  # same number of rows

        # Apply score_mod if provided
        if cutlass.const_expr(self.score_mod is not None):
            assert self.score_mod is not None  # mypy
            for r in cutlass.range(num_rows, unroll_full=True):  # loop over rows
                acc_S_mn[r, None].store(
                    self.score_mod(
                        acc_S_mn[r, None].load() * softmax_scale,
                        0,
                        0,
                        0,
                        0,
                        None,
                        [],
                    )
                )

        # Apply mask if provided
        if cutlass.const_expr(mask_fn is not None):
            assert mask_fn is not None  # mypy
            mask_fn(acc_S, m_block=m_block)

        # Apply softmax with LSE: P = exp(S - LSE)
        for r in cutlass.range(num_rows, unroll_full=True):  # loop over rows
            acc_P_mn[r, None].store(
                cute.math.exp2(
                    acc_S_mn[r, None].load() * softmax_scale_log2 - tLSErLSE[r],
                    fastmath=True,
                )
            )

        # --- Apply dP = dO*V^T ---

        # Zero-init acc_dP
        # acc_dP: (MMA_ATOM=(2,2),MMA_Q4,MMA_K2):((1,2),4,16)
        acc_dP = cute.make_rmem_tensor(acc_shape_SdP, cutlass.Float32)
        acc_dP.fill(0.0)

        # Wait for this dO/dPsum tile
        cute.arch.cp_async_wait_group(
            1 if cutlass.const_expr(self.num_stages_dO > 1) else 0
        )
        cute.arch.barrier()

        # Issue MMA for dP = dO*V^T
        # with next Q/LSE tile load if num_stages_Q > 1
        sm80_utils.gemm(
            tiled_mma=mma_params.thr_mma_sdp,
            acc=acc_dP,
            tCrA=mma_params.tdPrdO,
            tCrB=mma_params.tdPrV,
            tCsA=smem_copy_params.tdPsdO[
                None,
                None,
                None,
                smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0,
            ],
            tCsB=smem_copy_params.tdPsV,
            smem_thr_copy_A=smem_copy_params.smem_thr_copy_QdO,
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_KV,
            hook_fn=load_Q_LSE_next
            if cutlass.const_expr(self.num_stages_Q > 1)
            else None,
            swap_AB=self.SdP_swapAB,
        )

        # --- Apply softmax bwd: dS = P*(dP - dPsum) ---

        # acc_dP/dS_mn: ((ATOM_Q2,MMA_Q4),(ATOM_K2,MMA_K2)):((2,4),(1,16))
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP)
        acc_dS, acc_dS_mn = acc_dP, acc_dP_mn

        # S2R copy dPsum
        # tSsdPsumMma: (MMA_ATOM=(ATOM_Q2,MMA_Q4),1):((8,16),0)
        # tLSErdPsum: (ATOM_Q2,MMA_Q4):((1,2))
        tLSErdPsum: cute.Tensor = cute.make_fragment_like(
            smem_copy_params.tSsdPsumMma[None, 0]
        )
        cute.autovec_copy(
            smem_copy_params.tSsdPsumMma[
                None,
                smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0,
            ],
            tLSErdPsum,
        )
        assert cute.size(tLSErdPsum) == num_rows  # same number of rows

        # Apply dS = P*(dP - dPsum)
        for r in cutlass.range(num_rows, unroll_full=True):  # loop over rows
            acc_dS_row = acc_P_mn[r, None].load() * (
                acc_dS_mn[r, None].load() - tLSErdPsum[r]
            )

            # Apply score_mod_bwd if provided
            if cutlass.const_expr(self.score_mod_bwd is not None):
                assert self.score_mod_bwd is not None  # mypy
                assert acc_S_pre_mn is not None  # mypy
                acc_dS_row = self.score_mod_bwd(
                    acc_dS_row,
                    acc_S_pre_mn[r, None].load() * softmax_scale,
                    0,
                    0,
                    0,
                    0,
                    None,
                    [],
                )

            acc_dS_mn[r, None].store(acc_dS_row)

        # --- Prepare P/dS for dK/dV MMA ---

        # Make rP/rdS from acc_P/acc_dS with dtype cast,
        # and R2S copy to smem if RS MMA is required
        # rP: (MMA_ATOM=(2,2),MMA_Q4,MMA_K2):((1,2),4,16)
        # tdVrP: (MMA_ATOM=(2,2,2),MMA_K1,MMA_Q4):((1,2,4),0,8)
        rP = cute.make_fragment_like(acc_P, self.dtype)
        rP.store(acc_P.load().to(self.dtype))

        # R2S first and S2R back later in the gemm fn
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            # tPrP: (CPY_ATOM=(2,(2,2)),CPY_Q4,CPY_K1):((1,(2,16)),4,0)
            # tPsP: (CPY_ATOM=(2,(2,2)),CPY_Q4,CPY_K1):((1,(512,4096)),1024,0)
            tPrP = smem_copy_params.r2s_thr_copy_PdS.retile(rP)
            cute.copy(smem_copy_params.r2s_thr_copy_PdS, tPrP, smem_copy_params.tPsP)

        tdVrP = (
            # NOTE: if Mma_dKV_is_RS, we directly use rP
            # otherwise, we use the pre-allocated tdVrP rmem buffer
            # and it will S2R copy tPsP (who's been R2S copied from rP above) to tdVrP in the gemm fn
            layout_utils.reshape_acc_to_frgA(rP)
            if cutlass.const_expr(self.Mma_dKV_is_RS)
            else mma_params.tdVrP
        )

        # rdS: (MMA_ATOM=(2,2),MMA_Q4,MMA_K2):((1,2),4,16)
        # tdKrdS: (MMA_ATOM=(2,2,2),MMA_K1,MMA_Q4):((1,2,4),0,8)
        rdS = cute.make_fragment_like(acc_dS, self.dtype)
        rdS.store(acc_dS.load().to(self.dtype))

        # R2S first and S2R back later in the gemm fn
        # NOTE: For hdim 64, It's faster to write to smem_dS first before the dV MMA
        if cutlass.const_expr(not self.Mma_dKV_is_RS):
            # Make sure P's R2S-copy above is done
            # before its S2R-copy in the gemm fn below
            # TODO(REVIEW): why put it here instead of just before dV MMA ?
            cute.arch.barrier()

            # tdSrdS: (CPY_ATOM=(2,(2,2)),CPY_Q4,CPY_K1):((1,(2,16)),4,0)
            # tdSsdS: (CPY_ATOM=(2,(2,2)),CPY_Q4,CPY_K1):((1,(512,4096)),1024,0)
            tdSrdS = smem_copy_params.r2s_thr_copy_PdS.retile(rdS)
            cute.copy(
                smem_copy_params.r2s_thr_copy_PdS, tdSrdS, smem_copy_params.tdSsdS
            )

        tdKrdS = (
            # NOTE: if Mma_dKV_is_RS, we directly use rdS
            # otherwise, we use the pre-allocated tdKrdS rmem buffer
            # and it will S2R copy tPtdSrdSsP (who's been R2S copied from rdS above) to tdKrdS in the gemm fn
            layout_utils.reshape_acc_to_frgA(rdS)
            if cutlass.const_expr(self.Mma_dKV_is_RS)
            else mma_params.tdKrdS
        )

        # --- Apply dV = P.T*dO ---

        sm80_utils.gemm(
            tiled_mma=mma_params.thr_mma_dkv,
            acc=mma_params.acc_dV,
            tCrA=tdVrP,
            tCrB=mma_params.tdVrdO,
            tCsA=smem_copy_params.tdVsPt,
            tCsB=smem_copy_params.tdVsdOt[
                None,
                None,
                None,
                smem_pipe_read_do if cutlass.const_expr(self.num_stages_dO > 1) else 0,
            ],
            smem_thr_copy_A=smem_copy_params.smem_thr_copy_PdSt,
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
        )

        # --- Apply dQ = dS*K ---

        # Make sure dS's R2S-copy above is done
        # before its S2R-copy in the gemm fn below
        cute.arch.barrier()

        # acc_dQ: (MMA_ATOM=(2,2),MMA_Q4,MMA_HD2):((1,2),4,16)
        # acc_dQ_atomic: (CPY_ATOM=(1,4),CPY_Q4,CPY_HD2):((0,1),4,16)
        # tdQgdQacc_atomic: (CPY_ATOM=(1,1),REST_V32):((0,0),256)
        acc_shape_dQ = mma_params.thr_mma_dq.partition_shape_C(
            (self.m_block_size, self.head_dim_padded)
            if cutlass.const_expr(not self.dQ_swapAB)
            else (self.head_dim_padded, self.m_block_size)
        )
        acc_dQ = cute.make_rmem_tensor(acc_shape_dQ, cutlass.Float32)
        acc_dQ_atomic = gmem_copy_params.gmem_thr_copy_dQacc.retile(acc_dQ)
        num_atomic_elems = cute.size(acc_dQ_atomic)
        tdQgdQacc_atomic = gmem_copy_params.tdQgdQacc[None, None, m_block]
        assert (
            cute.size(tdQgdQacc_atomic) == num_atomic_elems
        )  # same number of elements

        def dQ_mma_with_atomic_add(hook_fn):
            # Zero-init acc_dQ
            acc_dQ.fill(0.0)

            # Issue MMA for dQ = dS*K
            sm80_utils.gemm(
                tiled_mma=mma_params.thr_mma_dq,
                acc=acc_dQ,
                tCrA=mma_params.tdQrdS,
                tCrB=mma_params.tdQrK,
                tCsA=smem_copy_params.tdQsdS,
                tCsB=smem_copy_params.tdQsKt,
                smem_thr_copy_A=smem_copy_params.smem_thr_copy_dS,
                smem_thr_copy_B=smem_copy_params.smem_thr_copy_Kt,
                swap_AB=self.dQ_swapAB,
                hook_fn=hook_fn,
            )

            # Atomic-add acc_dQ to dQaccum
            for i in cutlass.range(num_atomic_elems, unroll_full=True):
                cutedsl_utils.atomic_add_fp32(
                    acc_dQ_atomic[i], cutedsl_utils.elem_pointer(tdQgdQacc_atomic, i)
                )

        # Apply dQ MMA with next dO/dPsum tile load
        if cutlass.const_expr(self.num_stages_Q > 1):
            dQ_mma_with_atomic_add(load_dO_dPsum_next)

        # --- Apply dK = dS^T*Q ---

        # Issue MMA for dK = dS^T*Q
        # with next dO/dPsum tile load if num_stages_Q == 1
        sm80_utils.gemm(
            tiled_mma=mma_params.thr_mma_dkv,
            acc=mma_params.acc_dK,
            tCrA=tdKrdS,
            tCrB=mma_params.tdKrQ,
            tCsA=smem_copy_params.tdKsdSt,
            tCsB=smem_copy_params.tdKsQt[
                None,
                None,
                None,
                smem_pipe_read_q if cutlass.const_expr(self.num_stages_Q > 1) else 0,
            ],
            smem_thr_copy_A=smem_copy_params.smem_thr_copy_PdSt,
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_QdOt,
            A_in_regs=self.Mma_dKV_is_RS,
            swap_AB=self.dKV_swapAB,
            hook_fn=load_dO_dPsum_next
            if cutlass.const_expr(self.num_stages_Q == 1)
            else None,
        )

        # If num_stages_Q == 1, we want to do Mma_dK first
        # so Q.T is used and we can apply dQ MMA with next Q/LSE tile load
        if cutlass.const_expr(self.num_stages_Q == 1):
            cute.arch.barrier()  # Make sure Q.T is read
            dQ_mma_with_atomic_add(load_Q_LSE_next)

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_compute_one_m_block] "
                cute.printf("")
                cute.printf(
                    prefix
                    + f"Mma_dKV_is_RS={self.Mma_dKV_is_RS}, "
                    + f"SdP_swapAB={self.SdP_swapAB}, "
                    + f"dKV_swapAB={self.dKV_swapAB}, "
                    + f"dQ_swapAB={self.dQ_swapAB}"
                )
                cute.printf(
                    prefix
                    + "m_block={}, smem_pipe_read_q={}, smem_pipe_read_do={}, "
                    + "smem_pipe_write_q={}, smem_pipe_write_do={}",
                    m_block,
                    smem_pipe_read_q,
                    smem_pipe_read_do,
                    smem_pipe_write_q,
                    smem_pipe_write_do,
                )
                # MMA S = Q*K^T
                cute.printf(prefix + "acc_S: {}", acc_S.layout)
                cute.printf(prefix + "acc_S_mn: {}", acc_S_mn.layout)
                cute.printf(prefix + "tLSErLSE: {}", tLSErLSE.layout)
                cute.printf(prefix + "num_rows: {}", num_rows)
                # MMA dP = dO*V^T
                cute.printf(prefix + "acc_dP: {}", acc_dP.layout)
                cute.printf(prefix + "acc_dP_mn: {}", acc_dS_mn.layout)
                cute.printf(prefix + "tLSErdPsum: {}", tLSErdPsum.layout)
                # S/dP -> P/dS
                cute.printf(prefix + "rP: {}", rP.layout)
                cute.printf(prefix + "rdS: {}", rdS.layout)
                # MMA dV = P^T*dO, dK = dS^T*Q
                cute.printf(prefix + "acc_dV: {}", mma_params.acc_dV.layout)
                cute.printf(prefix + "acc_dK: {}", mma_params.acc_dK.layout)
                cute.printf(prefix + "tdVrP: {}", tdVrP.layout)
                cute.printf(prefix + "tdKrdS: {}", tdKrdS.layout)
                if cutlass.const_expr(not self.Mma_dKV_is_RS):
                    cute.printf(prefix + "tPrP: {}", tPrP.layout)
                    cute.printf(prefix + "tPsP: {}", smem_copy_params.tPsP.layout)
                    cute.printf(prefix + "tdSrdS: {}", tdSrdS.layout)
                    cute.printf(prefix + "tdSsdS: {}", smem_copy_params.tdSsdS.layout)
                # MMA dQ = dS*K
                cute.printf(prefix + "acc_dQ: {}", acc_dQ.layout)
                cute.printf(prefix + "acc_dQ_atomic: {}", acc_dQ_atomic.layout)
                cute.printf(prefix + "tdQgdQacc_atomic: {}", tdQgdQacc_atomic.layout)
                cute.printf(prefix + "tdQrdS: {}", mma_params.tdQrdS.layout)
                cute.printf(prefix + "tdQrK: {}", mma_params.tdQrK.layout)
                cute.printf(prefix + "num_atomic_elems: {}", num_atomic_elems)
                cute.printf("")

    @cute.jit
    def epilogue(
        self,
        acc_dK: cute.Tensor,
        acc_dV: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sdK: cute.Tensor,
        sdV: cute.Tensor,
        gmem_tiled_copy_dK: cute.TiledCopy,
        gmem_tiled_copy_dV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        n_block: cutlass.Int32,
        num_head: cutlass.Int32,
        batch_size: cutlass.Int32,
        seqlen_info: SeqlenInfoQK,
        d_head: cutlass.Int32,
        d_head_v: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        batch_idx = batch_size
        head_idx_kv = (
            num_head // self.qhead_per_kvhead
            if cutlass.const_expr(not self.pack_gqa)
            else num_head
        )

        gmem_thr_copy_dK = gmem_tiled_copy_dK.get_slice(tidx)
        gmem_thr_copy_dV = gmem_tiled_copy_dV.get_slice(tidx)

        if cutlass.const_expr(self.qhead_per_kvhead == 1):  # vectorized store
            # Make rdK/rdV from acc_dK/dV with dtype cast
            # acc_dK/dV: (MMA_ATOM=(2,2),MMA_K1,MMA_HD16):((1,2),0,4)
            # rdK/rdV: (MMA_ATOM=(2,2),MMA_K1,MMA_HD16):((1,2),0,4)
            rdK = cute.make_fragment_like(acc_dK, self.dtype)
            rdK.store(acc_dK.load().to(self.dtype))
            rdV = cute.make_fragment_like(acc_dV, self.dtype)
            rdV.store(acc_dV.load().to(self.dtype))

            # Make sure all threads have finished reading sK and sV,
            # before we overwrite smem with dK and dV
            cute.arch.barrier()

            # R2S copy atom for dK/dV with universal `st.shared`
            # layout_src_tv=(1,2):(0,1)
            # layout_dst_tv=(1,2):(0,1)
            smem_copy_atom_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dtype,
                num_bits_per_copy=2 * self.dtype.width,
            )

            # R2S tiled copy of dK/dV with universal `st.shared`
            # layout_src_tv_tiled=((4,8,8),(2,(2,2))):((256,1,16),(128,(8,1024)))
            # layout_dst_tv_tiled=((4,8,8),(2,(2,2))):((256,1,16),(128,(8,1024)))
            smem_thr_copy_dKV = cute.make_tiled_copy_C(
                smem_copy_atom_dKV, tiled_mma
            ).get_slice(tidx)

            # Partition for R2S copy dK/dV
            # taccdKrdK/taccdVrdV: (CPY_ATOM=(2,4),CPY_K1,CPY_HD8):((1,2),0,8)
            # taccdKsdK/taccdVsdV: (CPY_ATOM=(2,(2,2)),CPY_K1,CPY_HD=((2,2),2)):((1,(512,-8)),0,((-16,-32),8192))
            taccdKrdK: cute.Tensor = smem_thr_copy_dKV.retile(rdK)
            taccdVrdV: cute.Tensor = smem_thr_copy_dKV.retile(rdV)
            taccdKsdK = smem_thr_copy_dKV.partition_D(sdK)
            taccdVsdV = smem_thr_copy_dKV.partition_D(sdV)

            # R2S copy dK/dV
            cute.copy(smem_copy_atom_dKV, taccdKrdK, taccdKsdK)
            cute.copy(smem_copy_atom_dKV, taccdVrdV, taccdVsdV)

            # Make gdK/gdV and rdK/rdV for R2G copy
            if cutlass.const_expr(not seqlen_info.has_cu_seqlens_k):
                mdK_cur, mdV_cur = [
                    t[batch_idx, None, head_idx_kv, None] for t in (mdK, mdV)
                ]
            else:
                mdK_cur, mdV_cur = [
                    cute.domain_offset(
                        (seqlen_info.offset_k, 0), t[None, head_idx_kv, None]
                    )
                    for t in (mdK, mdV)
                ]

            # mdK/dV_cur: (sK,HD):(HD*nhK,1)
            # gdK/dV: (tileK128,tileHD128):(HD*nhK,1)
            blkdK_shape = (self.n_block_size, self.head_dim_padded)
            blkdV_shape = (self.n_block_size, self.head_dim_v_padded)
            gdK = cute.local_tile(mdK_cur, blkdK_shape, (n_block, 0))
            gdV = cute.local_tile(mdV_cur, blkdV_shape, (n_block, 0))

            # Partition for S2R copy dK/dV
            # tdKsdK/tdVsdV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),2048,8192)
            # tdKgdK/tdVgdV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),32768,64)
            # tdKrdK/tdVrdV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),16,8)
            tdKsdK = gmem_thr_copy_dK.partition_S(sdK)
            tdVsdV = gmem_thr_copy_dV.partition_S(sdV)
            tdKgdK = gmem_thr_copy_dK.partition_D(gdK)
            tdVgdV = gmem_thr_copy_dV.partition_D(gdV)
            tdKrdK = cute.make_fragment_like(tdKgdK, self.dtype)
            tdVrdV = cute.make_fragment_like(tdVgdV, self.dtype)

            # Make sure R2S copy of dK/dV is finished
            # before we start S2R copy
            cute.arch.barrier()

            # S2R copy dK/dV from smem back to rmem for wider vectorization
            # TODO: Need to check OOB when reading from smem if kBlockN isn't evenly tiled
            cute.autovec_copy(tdKsdK, tdKrdK)
            cute.autovec_copy(tdVsdV, tdVrdV)

            # Make predicates of dK/dV for R2G copy
            # cdK: (tileK128,tileHD128):(1@0,1@1)
            # tdKcdK/t0dKcdK/tdVcdV/t0dVcdV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),32@0,64@1)
            # tdKpdK/tdVpdV: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1)  => the same predicate along CPY_K
            cdK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
            tdKcdK = gmem_thr_copy_dK.partition_S(cdK)
            t0dKcdK = gmem_tiled_copy_dK.get_slice(0).partition_S(cdK)
            if cutlass.const_expr(self.head_dim_padded == self.head_dim_v_padded):
                tdVcdV = tdKcdK
                t0dVcdV = t0dKcdK
            else:
                cdV = cute.make_identity_tensor(
                    (self.n_block_size, self.head_dim_v_padded)
                )
                tdVcdV = gmem_thr_copy_dV.partition_S(cdV)
                t0dVcdV = gmem_tiled_copy_dV.get_slice(0).partition_S(cdV)
            tdKpdK = cutedsl_utils.predicate_k(tdKcdK, limit=d_head)
            if cutlass.const_expr(self.same_hdim_kv):
                tdVpdV = tdKpdK
            else:
                tdVpdV = cutedsl_utils.predicate_k(tdVcdV, limit=d_head_v)

            # R2G copy dK/dV
            sk_limit = seqlen_info.seqlen_k - n_block * self.n_block_size
            for rest_m in cutlass.range_constexpr(
                cute.size(tdKrdK.shape[1])
            ):  # loop over CPY_K
                if t0dKcdK[0, rest_m, 0][0] < sk_limit - tdKcdK[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dK,
                        tdKrdK[None, rest_m, None],
                        tdKgdK[None, rest_m, None],
                        pred=tdKpdK[None, rest_m, None]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None,
                    )
            for rest_m in cutlass.range_constexpr(
                cute.size(tdVrdV.shape[1])
            ):  # loop over CPY_K
                if t0dVcdV[0, rest_m, 0][0] < sk_limit - tdVcdV[0][0]:
                    cute.copy(
                        gmem_tiled_copy_dV,
                        tdVrdV[None, rest_m, None],
                        tdVgdV[None, rest_m, None],
                        pred=tdVpdV[None, rest_m, None]
                        if cutlass.const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:  # qhead_per_kvhead > 1 => atomic add
            # Make fp32 gdK/gdV buffer for atomic add
            if cutlass.const_expr(not seqlen_info.has_cu_seqlens_k):
                mdK_cur, mdV_cur = [t[batch_idx, head_idx_kv, None] for t in (mdK, mdV)]
            else:
                padded_offset_k = seqlen_info.padded_offset_k
                mdK_cur = cute.domain_offset(
                    (padded_offset_k * self.head_dim_padded,), mdK[head_idx_kv, None]
                )
                mdV_cur = cute.domain_offset(
                    (padded_offset_k * self.head_dim_v_padded,), mdV[head_idx_kv, None]
                )

            # mdK/dV_cur: (sK*HD):(1)
            # gdK/dV: (tileK128*tileHD128):(1)
            gdK = cute.local_tile(
                mdK_cur, (self.n_block_size * self.head_dim_padded,), (n_block,)
            )
            gdV = cute.local_tile(
                mdV_cur, (self.n_block_size * self.head_dim_v_padded,), (n_block,)
            )

            # tdKgdKaccum/tdVgdVaccum: (CPY_ATOM=(1,1),REST_V=(64)):((0,0),(256))
            tdKgdKaccum = gmem_thr_copy_dK.partition_S(gdK)
            tdVgdVaccum = gmem_thr_copy_dV.partition_S(gdV)

            # acc_dK/dV: (MMA_ATOM=(2,2),MMA_K1,MMA_HD16):((1,2),0,4)
            # acc_dK/dV_atomic: (CPY_ATOM=(1,4),CPY_K1,CPY_HD16):((0,1),0,4)
            acc_dK_atomic: cute.Tensor = gmem_thr_copy_dK.retile(acc_dK)
            acc_dV_atomic: cute.Tensor = gmem_thr_copy_dV.retile(acc_dV)
            assert cute.size(acc_dK_atomic) == cute.size(tdKgdKaccum)
            assert cute.size(acc_dV_atomic) == cute.size(tdVgdVaccum)

            # R2G atomic add dK/dV to fp32 gmem buffer
            for i in cutlass.range(cute.size(acc_dV_atomic), unroll_full=True):
                cutedsl_utils.atomic_add_fp32(
                    acc_dV_atomic[i], cutedsl_utils.elem_pointer(tdVgdVaccum, i)
                )
            for i in cutlass.range(cute.size(acc_dK_atomic), unroll_full=True):
                cutedsl_utils.atomic_add_fp32(
                    acc_dK_atomic[i], cutedsl_utils.elem_pointer(tdKgdKaccum, i)
                )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_epilogue] "
                cute.printf("")
                cute.printf(prefix + f"qhead_per_kvhead={self.qhead_per_kvhead}")
                cute.printf(
                    prefix + "n_block={}, head_idx_kv={}, batch_idx={}",
                    n_block,
                    head_idx_kv,
                    batch_idx,
                )
                cute.printf(prefix + "acc_dK: {}", acc_dK.layout)
                cute.printf(prefix + "acc_dV: {}", acc_dV.layout)
                cute.printf(prefix + "mdK_cur: {}", mdK_cur.layout)
                cute.printf(prefix + "mdV_cur: {}", mdV_cur.layout)
                cute.printf(prefix + "gdK: {}", gdK.layout)
                cute.printf(prefix + "gdV: {}", gdV.layout)
                cute.printf("")

                if cutlass.const_expr(self.qhead_per_kvhead == 1):
                    cute.printf(prefix + "(smem store path)")
                    cute.printf(prefix + "rdK: {}", rdK.layout)
                    cute.printf(prefix + "rdV: {}", rdV.layout)
                    cute.printf(prefix + "sdK: {}", sdK.layout)
                    cute.printf(prefix + "sdV: {}", sdV.layout)
                    cute.printf("")
                    cute.printf(
                        prefix
                        + "smem_copy_atom_dKV: layout_src_tv={}, layout_dst_tv={}",
                        smem_copy_atom_dKV.layout_src_tv,
                        smem_copy_atom_dKV.layout_dst_tv,
                    )
                    cute.printf(
                        prefix
                        + "smem_thr_copy_dKV: layout_src_tv_tiled={}, layout_dst_tv_tiled={}",
                        smem_thr_copy_dKV.layout_src_tv_tiled,
                        smem_thr_copy_dKV.layout_dst_tv_tiled,
                    )
                    cute.printf("")
                    cute.printf(prefix + "taccdKrdK: {}", taccdKrdK.layout)
                    cute.printf(prefix + "taccdVrdV: {}", taccdVrdV.layout)
                    cute.printf(prefix + "taccdKsdK: {}", taccdKsdK.layout)
                    cute.printf(prefix + "taccdVsdV: {}", taccdVsdV.layout)
                    cute.printf("")
                    cute.printf(prefix + "tdKsdK: {}", tdKsdK.layout)
                    cute.printf(prefix + "tdKgdK: {}", tdKgdK.layout)
                    cute.printf(prefix + "tdVsdV: {}", tdVsdV.layout)
                    cute.printf(prefix + "tdVgdV: {}", tdVgdV.layout)
                    cute.printf(prefix + "tdKrdK: {}", tdKrdK.layout)
                    cute.printf(prefix + "tdVrdV: {}", tdVrdV.layout)
                    cute.printf("")
                    cute.printf(prefix + "cdK: {}", cdK.layout)
                    cute.printf(prefix + "tdKcdK: {}", tdKcdK.layout)
                    cute.printf(prefix + "t0dKcdK: {}", t0dKcdK.layout)
                    cute.printf(prefix + "tdVcdV: {}", tdVcdV.layout)
                    cute.printf(prefix + "t0dVcdV: {}", t0dVcdV.layout)
                    cute.printf(prefix + "tdKpdK: {}", tdKpdK.layout)
                    cute.printf(prefix + "tdVpdV: {}", tdVpdV.layout)
                else:
                    cute.printf(prefix + "(atomic add path)")
                    cute.printf(prefix + "tdKgdKaccum: {}", tdKgdKaccum.layout)
                    cute.printf(prefix + "tdVgdVaccum: {}", tdVgdVaccum.layout)
                    cute.printf(prefix + "acc_dK_atomic: {}", acc_dK_atomic.layout)
                    cute.printf(prefix + "acc_dV_atomic: {}", acc_dV_atomic.layout)
                cute.printf("")

    @cute.jit
    def advance_pipeline(self, pipeline_index, num_stages: cutlass.Constexpr):
        return pipeline_index + 1 if pipeline_index < num_stages - 1 else 0

    @cute.jit
    def load_K(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshot kBlockM/kBlockN when we load Q/K ?
        is_even_n_smem_k = self.n_block_size % gmem_thr_copy.tiler_mn[0].shape == 0

        # tKgK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),8192,64)
        # tKsK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),2048,8192)

        # cK: (tileK128,tileHD128):(1@0,1@1)
        # tKcK/t0KcK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),32@0,64@1)
        # tKpK: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1) => the same predicate along CPY_K
        cK = cute.make_identity_tensor((self.n_block_size, self.head_dim_padded))
        tKcK: cute.Tensor = gmem_thr_copy.partition_S(cK)
        t0KcK = gmem_thr_copy.get_slice(0).partition_S(cK)
        tKpK = cutedsl_utils.predicate_k(tKcK, limit=headdim)

        sk_limit = seqlen - block * self.n_block_size
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if (
                is_even_n_smem_k
                or n < cute.size(tKsK.shape[1]) - 1
                or tKcK[0, n, 0][0] < self.n_block_size
            ):
                # NOTE: Instead of using `tKcK[0, n, 0][0] < sk_limit`,
                # we use t0KcK to compare with `sk_limit - tKcK[0][0]`
                # to make the left hand side a compile-time constant expression.
                predicate_n = t0KcK[0, n, 0][0] < sk_limit - tKcK[0][0]
                predicate = cute.make_fragment_like(tKpK[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (
                            tKpK[i, n, k]
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and predicate_n

                cute.copy(
                    gmem_thr_copy,
                    tKgK[None, n, None],
                    tKsK[None, n, None],
                    pred=predicate,
                )

            # NOTE: We need to clear the sK smem tiles
            # since we'll use sKt for mma_dq

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_load_K] "
                cute.printf("")
                cute.printf(
                    prefix + "block={}, seqlen={}, headdim={}",
                    block,
                    seqlen,
                    headdim,
                )
                cute.printf(
                    prefix + "sk_limit={}, is_even_n_smem_k={}",
                    sk_limit,
                    is_even_n_smem_k,
                )
                cute.printf("")
                cute.printf(prefix + "tKgK: {}", tKgK.layout)
                cute.printf(prefix + "tKsK: {}", tKsK.layout)
                cute.printf(prefix + "cK: {}", cK.layout)
                cute.printf(prefix + "tKcK: {}", tKcK.layout)
                cute.printf(prefix + "t0KcK: {}", t0KcK.layout)
                cute.printf(prefix + "tKpK: {}", tKpK.layout)
                cute.printf("")

    @cute.jit
    def load_V(
        self,
        gmem_thr_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
        headdim: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshot kBlockN when we load V ?
        is_even_n_smem_v = self.n_block_size % gmem_thr_copy.tiler_mn[0].shape == 0

        # tVgV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),8192,64)
        # tVsV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1,0),2048,8192)

        # cV: (tileK128,tileHD128):(1@0,1@1)
        # tVcV/t0VcV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),32@0,64@1)
        # tVpV: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1) => the same predicate along CPY_K
        cV = cute.make_identity_tensor((self.n_block_size, self.head_dim_v_padded))
        tVcV: cute.Tensor = gmem_thr_copy.partition_S(cV)
        t0VcV = gmem_thr_copy.get_slice(0).partition_S(cV)
        tVpV = cutedsl_utils.predicate_k(tVcV, limit=headdim)

        sk_limit = seqlen - block * self.n_block_size
        for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
            # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
            if (
                is_even_n_smem_v
                or n < cute.size(tVsV.shape[1]) - 1
                or tVcV[0, n, 0][0] < self.n_block_size
            ):
                # NOTE: Instead of using `tVcV[0, n, 0][0] < sk_limit`,
                # we use t0VcV to compare with `sk_limit - tVcV[0][0]`
                # to make the left hand side a compile-time constant expression.
                predicate_n = t0VcV[0, n, 0][0] < sk_limit - tVcV[0][0]
                predicate = cute.make_fragment_like(tVpV[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (
                            tVpV[i, n, k]
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and predicate_n
                cute.copy(
                    gmem_thr_copy,
                    tVgV[None, n, None],
                    tVsV[None, n, None],
                    pred=predicate,
                )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_load_V] "
                cute.printf("")
                cute.printf(
                    prefix + "block={}, seqlen={}, headdim={}",
                    block,
                    seqlen,
                    headdim,
                )
                cute.printf(
                    prefix + "sk_limit={}, is_even_n_smem_v={}",
                    sk_limit,
                    is_even_n_smem_v,
                )
                cute.printf("")
                cute.printf(prefix + "cV: {}", cV.layout)
                cute.printf(prefix + "tVgV: {}", tVgV.layout)
                cute.printf(prefix + "tVsV: {}", tVsV.layout)
                cute.printf(prefix + "tVcV: {}", tVcV.layout)
                cute.printf(prefix + "t0VcV: {}", t0VcV.layout)
                cute.printf(prefix + "tVpV: {}", tVpV.layout)
                cute.printf("")

    @cute.jit
    def load_Q_LSE(
        self,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        tQcQ: cute.Tensor,
        t0QcQ: cute.Tensor,
        tQpQ: cute.Tensor,
        tLSEgLSE: cute.Tensor,
        tLSEsLSE: cute.Tensor,
        tLSEcLSE: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshot kBlockM when we load Q ?
        is_even_m_smem_q = self.m_block_size % gmem_tiled_copy_Q.tiler_mn[0].shape == 0

        # tQgQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,restQ):((1,0),32768,64,65536)
        # tQsQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,STAGE=(1,1)):((1,0),2048,4096,(0,0))
        # tQcQ/t0QcQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2):((1@1,0),32@0,64@1)
        # tQpQ: (ATOM_REST_V1,CPY_Q2,CPY_HD2):(2,0,1) => the same predicate along CPY_Q
        # tLSEgLSE: (CPY_ATOM=(4,1),CPY_Q1,restQ):((1,0),0,64)
        # tLSEsLSE: (CPY_ATOM=(4,1),CPY_Q1,STAGE1):((1,0),0,64)
        # tLSEcLSE: (CPY_ATOM=(4,1),CPY_Q=(1)):((1@0,0),(0))

        sq_limit = seqlen - block * self.m_block_size
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if (
                is_even_m_smem_q
                or m < cute.size(tQsQ.shape[1]) - 1
                or tQcQ[0, m, 0][0] < self.m_block_size
            ):
                # NOTE: Instead of using `tQcQ[0, m, 0][0] < sq_limit`,
                # we use t0QcQ to compare with `sq_limit - tQcQ[0][0]`
                # to make the left hand side a compile-time constant expression.
                predicate_m = t0QcQ[0, m, 0][0] < sq_limit - tQcQ[0][0]
                predicate = cute.make_fragment_like(tQpQ[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (
                            tQpQ[i, m, k]
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and predicate_m
                cute.copy(
                    gmem_tiled_copy_Q,
                    tQgQ[None, m, None, block],
                    tQsQ[
                        None,
                        m,
                        None,
                        smem_pipe_write_q
                        if cutlass.const_expr(self.num_stages_Q) > 1
                        else 0,
                    ],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tLSEsLSE.shape[1])):
            if tLSEcLSE[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_LSE,
                    tLSEgLSE[None, m, block],
                    tLSEsLSE[
                        None,
                        m,
                        smem_pipe_write_q
                        if cutlass.const_expr(self.num_stages_Q > 1)
                        else 0,
                    ],
                )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_load_Q_LSE] "
                cute.printf("")
                cute.printf(
                    prefix + "block={}, smem_pipe_write_q={}, seqlen={}",
                    block,
                    smem_pipe_write_q,
                    seqlen,
                )
                cute.printf(
                    prefix + "sq_limit={}, is_even_m_smem_q={}",
                    sq_limit,
                    is_even_m_smem_q,
                )
                cute.printf("")
                cute.printf(prefix + "tQgQ: {}", tQgQ.layout)
                cute.printf(prefix + "tQsQ: {}", tQsQ.layout)
                cute.printf(prefix + "tQcQ: {}", tQcQ.layout)
                cute.printf(prefix + "t0QcQ: {}", t0QcQ.layout)
                cute.printf(prefix + "tQpQ: {}", tQpQ.layout)
                cute.printf(prefix + "tLSEgLSE: {}", tLSEgLSE.layout)
                cute.printf(prefix + "tLSEsLSE: {}", tLSEsLSE.layout)
                cute.printf(prefix + "tLSEcLSE: {}", tLSEcLSE.layout)
                cute.printf("")

    @cute.jit
    def load_dO_dPsum(
        self,
        gmem_tiled_copy_dO: cute.TiledCopy,
        gmem_tiled_copy_dPsum: cute.TiledCopy,
        tdOgdO: cute.Tensor,
        tdOsdO: cute.Tensor,
        tdOcdO: cute.Tensor,
        t0dOcdO: cute.Tensor,
        tdOpdO: cute.Tensor,
        tdPsumgdPsum: cute.Tensor,
        tdPsumsdPsum: cute.Tensor,
        tdPsumcdPsum: cute.Tensor,
        block: cutlass.Int32,
        smem_pipe_write_q: cutlass.Int32,
        seqlen: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshot kBlockM when we load dO ?
        is_even_m_smem_do = (
            self.m_block_size % gmem_tiled_copy_dO.tiler_mn[0].shape == 0
        )

        # tdOgdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,restQ):((1,0),32768,64,65536)
        # tdOsdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2,STAGE=(1,1)):((1,0),2048,4096,(0,0))
        # tdOcdO/t0dOcdO: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD2):((1@1,0),32@0,64@1)
        # tdOpdO: (ATOM_REST_V1,CPY_Q2,CPY_HD2):(2,0,1) => the same predicate along CPY_Q
        # tLSEgdPsum: (CPY_ATOM=(4,1),CPY_Q1,restQ):((1,0),0,64)
        # tLSEsdPsum: (CPY_ATOM=(4,1),CPY_Q1,STAGE1):((1,0),0,64)
        # tdPsumcdPsum: (CPY_ATOM=(4,1),CPY_Q=(1)):((1@0,0),(0))

        sq_limit = seqlen - block * self.m_block_size
        for m in cutlass.range_constexpr(cute.size(tdOsdO.shape[1])):
            # If kBlockM doesn't evenly divide the tiled copy, only the last `m` needs to be checked
            if (
                is_even_m_smem_do
                or m < cute.size(tdOsdO.shape[1]) - 1
                or tdOcdO[0, m, 0][0] < self.m_block_size
            ):
                # NOTE: Instead of using `tdOcdO[0, m, 0][0] < sq_limit`,
                # we use t0dOcdO to compare with `sq_limit - tdOcdO[0][0]`
                # to make the left hand side a compile-time constant expression.
                predicate_m = t0dOcdO[0, m, 0][0] < sq_limit - tdOcdO[0][0]
                predicate = cute.make_fragment_like(tdOpdO[None, 0, None])
                for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                    for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                        predicate[i, k] = (
                            tdOpdO[i, m, k]
                            if cutlass.const_expr(self.check_hdim_oob)
                            else True
                        ) and predicate_m
                cute.copy(
                    gmem_tiled_copy_dO,
                    tdOgdO[None, m, None, block],
                    tdOsdO[
                        None,
                        m,
                        None,
                        smem_pipe_write_q
                        if cutlass.const_expr(self.num_stages_dO > 1)
                        else 0,
                    ],
                    pred=predicate,
                )
            # We need to clear the sQ smem tiles since we'll use sQt for mma_dK
        # We made sure LSE length is padded so we read `kBlockM` elements so that all
        # elements in sLSE are filled. Without this we might have uninitialized sLSE values.
        for m in cutlass.range_constexpr(cute.size(tdPsumgdPsum.shape[1])):
            if tdPsumcdPsum[0, m][0] < self.m_block_size:
                cute.copy(
                    gmem_tiled_copy_dPsum,
                    tdPsumgdPsum[None, m, block],
                    tdPsumsdPsum[
                        None,
                        m,
                        smem_pipe_write_q
                        if cutlass.const_expr(self.num_stages_dO > 1)
                        else 0,
                    ],
                )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[bwd_sm80_load_dO_dPsum] "
                cute.printf("")
                cute.printf(
                    prefix + "block={}, smem_pipe_write_q={}, seqlen={}",
                    block,
                    smem_pipe_write_q,
                    seqlen,
                )
                cute.printf(
                    prefix + "sq_limit={}, is_even_m_smem_do={}",
                    sq_limit,
                    is_even_m_smem_do,
                )
                cute.printf("")
                cute.printf(prefix + "tdOgdO: {}", tdOgdO.layout)
                cute.printf(prefix + "tdOsdO: {}", tdOsdO.layout)
                cute.printf(prefix + "tdOcdO: {}", tdOcdO.layout)
                cute.printf(prefix + "t0dOcdO: {}", t0dOcdO.layout)
                cute.printf(prefix + "tdOpdO: {}", tdOpdO.layout)
                cute.printf(prefix + "tdPsumgdPsum: {}", tdPsumgdPsum.layout)
                cute.printf(prefix + "tdPsumsdPsum: {}", tdPsumsdPsum.layout)
                cute.printf(prefix + "tdPsumcdPsum: {}", tdPsumcdPsum.layout)
                cute.printf("")
