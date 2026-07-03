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
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import Arch, BaseDSL

# isort: split
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils, sm80_utils
from .block_info import BlockInfo
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .named_barrier import NamedBarrierFwd
from .pack_gqa import PackGQA
from .seqlen_info import SeqlenInfoQK
from .softmax import Softmax, apply_score_mod_inner
from .sparse_utils import BlockSparseTensors
from .tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FFAFwdSm80:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        mask_type: int = MT_MAP.full,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int | None = None,
        debug_print: bool = False,
    ):
        self.dtype = dtype

        # Pad head_dim to a multiple of 16 as k_block_size
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
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m  # tileQ128
        self.tile_n = tile_n  # tileK64

        self.num_threads = num_threads  # 128 (1 WG)
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE  # 4

        self.num_stages = num_stages
        self.q_subtile_factor = q_subtile_factor
        self.Q_in_regs = Q_in_regs  # False
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32

        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        if self.vec_size > 2:
            raise ValueError(
                f"score_mod vec_size {self.vec_size} not supported on sm80/90/120 "
                "due to accumulator thread ownership pattern."
            )

        self.buffer_align_bytes = 1024

        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[fwd_sm80_init] "
            print()
            print(f"{prefix}Initialized FFAFwdSm80 with: ")
            print(f"{prefix}{self.dtype=} | {self.arch=} | {self.qhead_per_kvhead=}")
            print(
                f"{prefix}{self.tile_hdim=} | {self.tile_hdimv=} | "
                f"{self.check_hdim_oob=} | {self.check_hdim_v_oob=}"
            )
            print(
                f"{prefix}{self.mask_type=} | {self.is_causal=} | {self.is_local=} | {self.pack_gqa=}"
            )
            print(
                f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.num_stages=} | {self.num_threads=}"
            )
            print(f"{prefix}{self.Q_in_regs=} | {self.q_subtile_factor=}")
            print(f"{prefix}{self.score_mod=} | {self.mask_mod=}")
            print(f"{prefix}{self.vec_size=} | {has_aux_tensors=}")
            print(f"{prefix}{self.buffer_align_bytes=}")
            print()

    @property
    def arch(self) -> Arch:
        """The DSL arch enum of the active compilation target."""
        return BaseDSL._get_dsl().get_arch_enum()

    @property
    def arch_num(self) -> int:
        """The DSL arch number of the active compilation target."""
        return self.arch.major * 10 + self.arch.minor

    @property
    def is_causal(self) -> bool:
        return self.mask_type == MT_MAP.causal

    @property
    def smem_capacity_arch(self) -> str:
        """SMEM-capacity bucket used by ``_check_tile``.

        Subclasses that reuse the SM80 MMA but have a different SMEM budget
        override this (e.g. FFAFwdSm120 -> "sm_120").
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
        if self.tile_n % 16 != 0:
            raise ValueError(f"tile_n must be a multiple of 16, got {self.tile_n}")
        if self.num_threads % 32 != 0:
            raise ValueError(
                f"num_threads must be a multiple of 32, got {self.num_threads}"
            )
        # Twice the M block must be divisible by the thread count.
        if (self.tile_m * 2) % self.num_threads != 0:
            raise ValueError(
                f"tile_m * 2 ({self.tile_m * 2}) must be divisible by num_threads "
                f"({self.num_threads})"
            )
        # SMEM usage: Q tile + (K tile + V tile); K and V use the same tile size.
        smem_usage_Q = self.tile_m * self.head_dim * 2
        smem_usage_K = self.tile_n * self.head_dim * self.num_stages * 2
        smem_usage_V = self.tile_n * self.head_dim_v * self.num_stages * 2
        smem_usage_QV = (
            max(smem_usage_Q, smem_usage_V)
            if self.Q_in_regs
            else (smem_usage_Q + smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
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
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type == mO_type == self.dtype)):
            raise TypeError(
                f"All tensors must have the same data type as the kernel dtype: {self.dtype}"
            )
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")

    def _get_smem_layout_atom(self):
        # S<3,3,3> o 0 o (8,64):(64,1)
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdimv)
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = (
            None  # always rP for sm80, since MMA needs all A,B to be in rmem
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm80_get_smem_layout_atom] "
            print()
            print(f"{prefix}sQ_layout_atom: {sQ_layout_atom}")
            print(f"{prefix}sK_layout_atom: {sK_layout_atom}")
            print(f"{prefix}sV_layout_atom: {sV_layout_atom}")
            print(f"{prefix}sO_layout_atom: {sO_layout_atom}")
            print(f"{prefix}sP_layout_atom: {sP_layout_atom}")
            print()

        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        # Tiled MMA for S=Q*K.T
        # Thr Layout VMNK: (32,4,1,1):(1,32,0,0)
        # Permutation MNK: (64:1,16:1,16:1)
        # MMA Atom
        # ThrID:           32:1
        # Shape MNK:       (16,8,16)
        # TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
        # TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
        # TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            atom_layout_mnk=(self.num_warps, 1, 1),
            permutation_mnk=(self.num_warps * 16, 16, 16),  # warps slice along m-dim
        )

        # Tiled MMA for O=P*V
        # Thr Layout VMNK: (32,4,1,1):(1,32,0,0)
        # Permutation MNK: (64:1,16:1,16:1)
        # MMA Atom
        # ThrID:           32:1
        # Shape MNK:       (16,8,16)
        # TV Layout A:     ((4,8),(2,2,2)):((32,1),(16,8,128))
        # TV Layout B:     ((4,8),(2,2)):((16,1),(8,64))
        # TV Layout C:     ((4,8),(2,2)):((32,1),(16,8))
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            atom_layout_mnk=(self.num_warps, 1, 1),
            permutation_mnk=(self.num_warps * 16, 16, 16),  # warps slice along m-dim
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)],
                self.buffer_align_bytes,
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]

        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], self.buffer_align_bytes
        ]

        @cute.struct
        class SharedStorageQKV:  # QKV
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:  # QK, sharing V with Q, if Q_in_regs
            sQ: sQV_struct
            sK: sK_struct

        self.shared_storage_cls = (
            SharedStorageSharedQV if const_expr(self.Q_in_regs) else SharedStorageQKV
        )

    def _setup_attributes(self):
        # --- Set up tiled MMA ---

        self.tiled_mma_qk, self.tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = self.tiled_mma_qk.size

        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads

        # --- Set up smem layout: sQ/sK/sV ---

        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()

        # sQ/sO: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sK/sV: S<3,3,3> o 0 o ((ATOM_K8,LAY_tileK8),(ATOM_HD64,LAY_tileHD2),STAGE=(1,1)):((64,512),(1,4096),(0,0))
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            # (tileQ128, tileHD128)
            (self.tile_m, self.tile_hdim),
            (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            # (tileK64, tileHD128, numStages)
            (self.tile_n, self.tile_hdim, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            # (tileK64, tileHD128, numStages)
            (self.tile_n, self.tile_hdimv, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom,
            # (tileQ128, tileHD128)
            (self.tile_m, self.tile_hdimv),
            (0, 1),
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n),
                (0, 1),
            )
        else:
            self.sP_layout = None

        # --- Set up G2S/S2G/R2G tiled copy ---

        # Thread layouts for copies
        universal_copy_bits = 128  # 16B per copy atom
        async_copy_elems = (
            universal_copy_bits // self.dtype.width
        )  # 8 elems per copy atom

        # Value layouts for all copies: (1,8):(0,1) => 8 bf16 elements per thread
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout

        # atom_async_copy: G2S copy atom for Q/K/V load with `cp.async`
        # layout_src_tv: (1,8):(0,1) => 8 bf16 elements per thread
        # layout_dst_tv: (1,8):(0,1) => 8 bf16 elements per thread
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # atom_universal_copy: universal copy atom for O store with `st.global`
        # layout_src_tv: (1,8):(0,1) => 8 bf16 elements per thread
        # layout_dst_tv: (1,8):(0,1) => 8 bf16 elements per thread
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        # tQ/tK: (16,8):(8,1)
        tQK_shape_dim_1 = (
            sQ_layout_atom.outer.shape[1] // async_copy_elems
        )  # 8 for smem_blk=64
        assert (
            self.num_Q_load_threads % tQK_shape_dim_1 == 0
        ), "num_Q_load_threads must be divisible by tQK_shape_dim_1"
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.tile_m % tQ_layout.shape[0] == 0

        # tK: (16,8):(8,1)
        assert (
            self.num_producer_threads % tQK_shape_dim_1 == 0
        ), "num_producer_threads must be divisible by tQK_shape_dim_1"
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )

        # tV: (16,8):(8,1)
        tV_shape_dim_1 = (
            sV_layout_atom.outer.shape[1] // async_copy_elems
        )  # 8 for smem_blk=64
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )

        # tO: (16,8):(8,1)
        # TODO: need a different thread layout for O if O dtype is not the same as V dtype
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.tile_m % tO_layout.shape[0] == 0

        # G2S async tiled_copy_QKV:
        # layout_src_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        # layout_dst_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            atom_async_copy, tQ_layout, vQKV_layout
        )
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(
            atom_async_copy, tK_layout, vQKV_layout
        )
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
            atom_async_copy, tV_layout, vQKV_layout
        )

        # R2G universal tiled_copy_O:
        # layout_src_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        # layout_dst_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm80_setup_attributes] "
            print()
            print(f"{prefix}{self.num_producer_threads=}")
            print(f"{prefix}{self.num_Q_load_threads=}")
            print(f"{prefix}{self.num_epilogue_threads=}")
            print(f"{prefix}{self.num_mma_threads=}")
            print()
            print(f"{prefix}tiled_mma_qk: {self.tiled_mma_qk}")
            print(f"{prefix}tiled_mma_pv: {self.tiled_mma_pv}")
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sO_layout: {self.sO_layout}")
            print(f"{prefix}sP_layout: {self.sP_layout}")
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
            print()
            print(f"{prefix}tQK_shape_dim_1: {tQK_shape_dim_1}")
            print(f"{prefix}tV_shape_dim_1: {tV_shape_dim_1}")
            print(f"{prefix}tQ_layout: {tQ_layout}")
            print(f"{prefix}tK_layout: {tK_layout}")
            print(f"{prefix}tV_layout: {tV_layout}")
            print(f"{prefix}tO_layout: {tO_layout}")
            print()
            print(
                f"{prefix}gmem_tiled_copy_Q: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_Q.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_Q.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_K: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_K.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_K.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_V: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_V.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_V.layout_dst_tv_tiled}"
            )
            print(
                f"{prefix}gmem_tiled_copy_O: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_O.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_O.layout_dst_tv_tiled}"
            )
            print()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # ///////////////////////////////////////////////////////////////////////////////
        # Set up attributes
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Checks ---

        assert mPageTable is None, "Page table is not supported yet for sm80"
        assert learnable_sink is None, "Learnable sink is not supported yet for sm80"
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
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )

        # --- Set up attributes ---

        self._setup_attributes()

        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mO/mLSE tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        # Layout permutation of Q/K/V/O:
        # 4D non-varlen: (b, s, nh, hd) -> (s, hd, nh, b)
        # 3D varlen: (t, nh, hd) -> (t, hd, nh)
        mQ, mK, mV, mO = [
            cutedsl_utils.assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)
        ]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mQ, mO = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=QO_layout_transpose)
            )
            for t in (mQ, mO)
        ]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]

        # Layout permutation of LSE:
        # 3D non-varlen: (b, nh, s) -> (s, nh, b)
        # 2D varlen: (nh, t) -> (t, nh)
        if const_expr(mLSE is not None):
            assert mLSE is not None  # mypy
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        self.tile_scheduler_cls = (
            SingleTileVarlenScheduler
            if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
            else SingleTileScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=(
                mCuSeqlensQ.shape[0] - 1  # type: ignore[union-attr]
                if const_expr(mCuSeqlensQ is not None)
                else mQ.shape[3]
            ),
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )
        tile_sched_params = self.tile_scheduler_cls.to_underlying_arguments(
            tile_sched_args
        )
        grid_dim = self.tile_scheduler_cls.get_grid_shape(tile_sched_params)

        # --- Make smem storage ---

        self._get_shared_storage_cls()

        # --- Make others ---

        softmax_scale_log2, softmax_scale = cutedsl_utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        fastdiv_mods = cutedsl_utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm80_call] "

            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
            cute.printf(prefix + "mO.layout: {}", mO.layout)
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
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            self.tiled_mma_qk,
            self.tiled_mma_pv,
            tile_sched_params,
            aux_tensors,
            fastdiv_mods,
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
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
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
        m_block, num_head, batch_size, _ = work_tile.tile_idx

        # --- Set up block info ---

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )

        # --- Set up seqlen info ---

        seqlen_info = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen_info, m_block)

        # NOTE:
        # 1. Start async loads KV of the last mn-tile, where we take care of the mn residue
        # 2. For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
        #   `seqlen_q=seqlen_k=0 and n_block_max=0`. Therefore, we clamp to 0
        #   so we don't use a negative block index for K/V loads, and the load/store predicates
        #   already guard all memory accesses when seqlen is 0.
        n_block = cutlass.max(n_block_max - 1, 0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make gmem tiles for Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////

        # mQ_cur: (sQ,HD):(HD*nhQ,1)
        # mK_cur/mV_cur: (sK,HD):(HD*nhK,1)
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        num_head_kv = num_head // self.qhead_per_kvhead
        if const_expr(not seqlen_info.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            mQ_cur = cute.domain_offset(
                (seqlen_info.offset_q, 0), mQ[None, None, num_head]
            )
        if const_expr(not seqlen_info.has_cu_seqlens_k):
            mK_cur = mK[None, None, num_head_kv, batch_size]
            mV_cur = mV[None, None, num_head_kv, batch_size]
        else:
            mK_cur = cute.domain_offset(
                (seqlen_info.offset_k, 0), mK[None, None, num_head_kv]
            )
            mV_cur = cute.domain_offset(
                (seqlen_info.offset_k, 0), mV[None, None, num_head_kv]
            )

        # gQ: (tileQ128,tileHD128):(HD*nhQ,1)
        # gK/gV: (tileK64,tileHD128,restK):(HD*nhK,1,HD*nhK*tileK)
        # where restK = sK // tileK64
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))  # slice out this m_block
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))  # all n blocks
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))  # all n blocks

        # ///////////////////////////////////////////////////////////////////////////////
        # Alloc smem storage and make smem tensors for sQ/sK/sV
        # ///////////////////////////////////////////////////////////////////////////////

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)

        # sQ: S<3,3,3> o 0 o ((ATOM_Q8,LAY_tileQ16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sK: S<3,3,3> o 0 o ((ATOM_K8,LAY_tileK8),(ATOM_HD64,LAY_tileHD2),STAGE=(1,1)):((64,512),(1,4096),(0,0))
        # sV: S<3,3,3> o 0 o ((ATOM_K8,LAY_tileK8),(ATOM_HD64,LAY_tileHD2),STAGE=(1,1)):((64,512),(1,4096),(0,0))
        # sVt: S<3,3,3> o 0 o ((ATOM_HD64,LAY_tileHD2),(ATOM_K8,LAY_tileK8),STAGE=(1,1)):((1,4096),(64,512),(0,0))
        sQ: cute.Tensor = storage.sQ.get_tensor(sQ_layout)
        sK: cute.Tensor = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV: cute.Tensor = storage.sV.get_tensor(sV_layout)
        else:  # Q/V shares the same smem buffer
            sV = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
            )
        # Transpose view of V to tensor with layout (tileHD, tileK) for tiled mma pv
        sVt = layout_utils.transpose_view(sV)
        # Reuse sQ buffer for O's S2G store
        sO = cute.make_tensor(sQ.iterator, sO_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # G2S tiled copy partitions for K/V
        # ///////////////////////////////////////////////////////////////////////////////

        # tKsK/tVsV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,STAGE=(1,1)):((1,0),1024,4096,(0,0))
        # tKgK/tVgV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,restK):((1,0),4096,64,16384)
        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////

        # tSrQ: (MMA_ATOM=(2,2,2),MMA_Q2,MMA_HD=((2,2),2)):((1,2,4),8,((32,64),16))
        # tSrK: (MMA_ATOM=(2,2),MMA_K8,MMA_HD=((2,2),2)):((1,2),4,((64,128),32))
        # tOrVt: (MMA_ATOM=(2,2),MMA_HD=(8,2),MMA_K4):((1,2),(4,128),32)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ: cute.Tensor = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK: cute.Tensor = thr_mma_qk.make_fragment_B(
            thr_mma_qk.partition_B(sK[None, None, 0])
        )
        tOrVt: cute.Tensor = thr_mma_pv.make_fragment_B(
            thr_mma_pv.partition_B(sVt[None, None, 0])
        )

        # acc_O: (MMA_ATOM=(2,2),MMA_Q2,MMA_HD16):((1,2),4,8)
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make S2R tiled copy and partitions for Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////

        # S2R copy atom for Q/K with `ldmatrix.sync.aligned.m8n8.x4` => m32xn8
        # layout_src_tv=(32,8):(8,1)
        # layout_dst_tv=(32,(2,4)):(2,(1,64))
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )

        # S2R copy atom for V with `ldmatrix.sync.aligned.m8n8.x4.trans` => m8xn32
        # layout_src_tv=(32,8):(8,1)
        # layout_dst_tv=((4,8),(1,2,4)):((16,1),(1,8,64))
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )

        # S2R tiled copy for Q
        # layout_src_tv_tiled=((16,2,4),(8,1)):((1,512,16),(64,0))
        # layout_dst_tv_tiled=((4,8,4),((2,2,2),1)):((128,1,16),((64,8,512),0))
        smem_thr_copy_Q = cutedsl_utils.make_tiled_copy_A(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)

        # S2R tiled copy for K
        # layout_src_tv_tiled=((8,2,2,4),(8,1)):((1,128,8,0),(16,0))
        # layout_dst_tv_tiled=((4,8,4),((2,2,2),1)):((32,1,0),((16,128,8),0))
        smem_thr_copy_K = cutedsl_utils.make_tiled_copy_B(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)

        # S2R tiled copy for V
        # layout_src_tv_tiled=((16,2,4),(8,1)):((16,8,0),(1,0))
        # layout_dst_tv_tiled=((4,8,4),((2,2,2),1)):((32,1,0),((16,128,8),0))
        smem_thr_copy_V = cutedsl_utils.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)

        # Partition src of sQ/sK/sVt for S2R tiled copy
        # tSsQ: (CPY_ATOM=(8,1),CPY_Q2,CPY_HD=((2,2),2)):((1,0),4096,((-16,-32),8192))
        # tSsK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD=((2,2),2),STAGE=(1,1)):((1,0),1024,((-16,-32),4096),(0,0))
        # tOsVt: (CPY_ATOM=(8,1),CPY_HD=((2,2),2),CPY_K4,STAGE=(1,1)):((1,0),((-16,-32),4096),1024,(0,0))
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make predicate tensors for K/V G2S loads
        # ///////////////////////////////////////////////////////////////////////////////

        # cK: (tileK64,tileHD128):(1@0,1@1)
        # tKcK/t0KcK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        # tVcV/t0VcV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)

        # tKpK/tVpV: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1) => the same predicate along CPY_K
        tKpK = cutedsl_utils.predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = cutedsl_utils.predicate_k(tVcV, limit=mV.shape[1])

        # ///////////////////////////////////////////////////////////////////////////////
        # Make others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make softmax object ---

        softmax = Softmax.create(  # shape: (atom_v_m * rest_m)
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        # --- Make partial functions for K/V loads ---

        load_K = partial(
            self.load_K,
            gmem_tiled_copy_K,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            seqlen=seqlen_info.seqlen_k,
        )
        load_V = partial(
            self.load_V,
            gmem_tiled_copy_V,
            tVgV,
            tVsV,
            tVcV,
            t0VcV,
            tVpV,
            seqlen=seqlen_info.seqlen_k,
        )

        # --- Make partial functions for compute_one_n_block ---

        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
        )
        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
            score_mod=self.score_mod,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods,
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread:
                prefix = "[fwd_sm80_kernel_setup] "
                cute.printf("")
                cute.printf(
                    prefix
                    + "bidx={}, bidy={}, bidz={}, tidx={}, m_block={}, num_head={}, batch_size={}",
                    bidx,
                    bidy,
                    bidz,
                    tidx,
                    m_block,
                    num_head,
                    batch_size,
                )
                cute.printf(
                    prefix + "n_block_min={}, n_block_max={}, n_block={}",
                    n_block_min,
                    n_block_max,
                    n_block,
                )
                cute.printf("")
                cute.printf(prefix + "mQ_cur: {}", mQ_cur.layout)
                cute.printf(prefix + "mK_cur: {}", mK_cur.layout)
                cute.printf(prefix + "mV_cur: {}", mV_cur.layout)
                cute.printf("")
                cute.printf(prefix + "gQ: {}", gQ.layout)
                cute.printf(prefix + "gK: {}", gK.layout)
                cute.printf(prefix + "gV: {}", gV.layout)
                cute.printf("")
                cute.printf(prefix + "sQ: {}", sQ.layout)
                cute.printf(prefix + "sK: {}", sK.layout)
                cute.printf(prefix + "sV: {}", sV.layout)
                cute.printf(prefix + "sVt: {}", sVt.layout)
                cute.printf("")
                cute.printf(prefix + "tKsK: {}", tKsK.layout)
                cute.printf(prefix + "tKgK: {}", tKgK.layout)
                cute.printf(prefix + "tVsV: {}", tVsV.layout)
                cute.printf(prefix + "tVgV: {}", tVgV.layout)
                cute.printf("")
                cute.printf(prefix + "tSrQ: {}", tSrQ.layout)
                cute.printf(prefix + "tSrK: {}", tSrK.layout)
                cute.printf(prefix + "tOrVt: {}", tOrVt.layout)
                cute.printf(prefix + "acc_O: {}", acc_O.layout)
                cute.printf("")
                cute.printf(
                    prefix + "smem_copy_atom_QK: layout_src_tv={}, layout_dst_tv={}",
                    smem_copy_atom_QK.layout_src_tv,
                    smem_copy_atom_QK.layout_dst_tv,
                )
                cute.printf(
                    prefix + "smem_copy_atom_V: layout_src_tv={}, layout_dst_tv={}",
                    smem_copy_atom_V.layout_src_tv,
                    smem_copy_atom_V.layout_dst_tv,
                )
                cute.printf(
                    prefix
                    + "smem_thr_copy_Q: layout_src_tv_tiled={}, layout_dst_tv_tiled={}",
                    smem_thr_copy_Q.layout_src_tv_tiled,
                    smem_thr_copy_Q.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix
                    + "smem_thr_copy_K: layout_src_tv_tiled={}, layout_dst_tv_tiled={}",
                    smem_thr_copy_K.layout_src_tv_tiled,
                    smem_thr_copy_K.layout_dst_tv_tiled,
                )
                cute.printf(
                    prefix
                    + "smem_thr_copy_V: layout_src_tv_tiled={}, layout_dst_tv_tiled={}",
                    smem_thr_copy_V.layout_src_tv_tiled,
                    smem_thr_copy_V.layout_dst_tv_tiled,
                )
                cute.printf("")
                cute.printf(prefix + "tSsQ: {}", tSsQ.layout)
                cute.printf(prefix + "tSsK: {}", tSsK.layout)
                cute.printf(prefix + "tOsVt: {}", tOsVt.layout)
                cute.printf("")
                cute.printf(prefix + "cK: {}", cK.layout)
                cute.printf(prefix + "tKcK: {}", tKcK.layout)
                cute.printf(prefix + "t0KcK: {}", t0KcK.layout)
                cute.printf(prefix + "tKpK: {}", tKpK.layout)
                cute.printf(prefix + "tVcV: {}", tVcV.layout)
                cute.printf(prefix + "t0VcV: {}", t0VcV.layout)
                cute.printf(prefix + "tVpV: {}", tVpV.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue: Load sQ, and one full stages of sK/sV
        # ///////////////////////////////////////////////////////////////////////////////

        # NOTE:
        # 1. If Q_in_regs, we load Q, then load 1 stage of K,
        #     then load the rest stages of K and all stages of V except the last stage of V,
        #     after we (optionally) rotate Q and read from smem buffer that Q/V share to rmem.
        # 2. Otherwise, we load Q, then load all stages of K/V except the last stage of V,
        #     then (optionally) rotate Q.

        # Load sQ
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(
            gmem_thr_copy_Q,
            gQ,
            sQ,
            block=m_block,
            seqlen=seqlen_info.seqlen_q,
            headdim=mQ.shape[1],
            is_print_thread_and_tile=is_print_thread,
        )
        cute.arch.cp_async_commit_group()

        # Load sK0, wait for sQ load to finish
        # and S2R copy sQ to rQ, if Q_in_regs
        if const_expr(self.Q_in_regs):
            load_K(
                n_block,
                smem_pipe_write=0,
                need_predicates=True,
                is_print_thread_and_tile=is_print_thread,
            )
            cute.arch.cp_async_commit_group()

            # Wait for sQ load to finish before S2R copy
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()

            # S2R copy rotated Q from smem buffer that Q/V share to rmem
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

            # Make sure all threads have read smem before loading V
            cute.arch.barrier()

        # Load sK/sV for the rest of stages
        for stage in cutlass.range_constexpr(self.num_stages):
            # Load Ki, i ∈ {1,...,num_stages-1} (∪ {0} if !Q_in_regs)
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                        is_print_thread_and_tile=is_print_thread and stage == 0,
                    )
                cute.arch.cp_async_commit_group()

            # Load Vi, i ∈ {0,...,num_stages-2}
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                        is_print_thread_and_tile=is_print_thread and stage == 0,
                    )
                cute.arch.cp_async_commit_group()

        # Wait for sQ load to finish before mainloop
        # if not Q_in_regs, otherwise, it's been waited before S2R copy already
        if const_expr(not self.Q_in_regs):
            # group cnt: load_Q: 1, load_K: num_stages, load_V: num_stages-1
            # thus we wait for `num_stages * 2 - 1` groups
            # to allow all stages of K/V loads on the fly
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop: Compute each n block iteration of
        #   1. forward before softmax: S = Q*K^T
        #   2. forward of softmax: P = softmax(S)
        #   3. forward after softmax: O = P*V
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make mask object and partial fn ---

        # NOTE: For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of tile_n.
        # We also need masking on S if it's causal, for the last several blocks.

        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen_info,
            window_size_left,
            window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_tensors=aux_tensors,
            fastdiv_mods=fastdiv_mods
            if const_expr(self.mask_mod is not None)
            else None,
        )

        # --- First iteration with seqlen masking ---

        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(
            n_block,
            smem_pipe_read,
            smem_pipe_write,
            is_first_n_block=True,
            seqlen_info=seqlen_info,
            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
            is_print_thread_and_tile=is_print_thread,
        )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # --- Next couple of iterations with causal/local masking ---

        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = (
                block_info.get_n_block_min_causal_local_mask(
                    seqlen_info, m_block, n_block_min
                )
            )
            for n_tile in cutlass.range(
                n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
            ):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                    seqlen_info=seqlen_info,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # --- The remaining iterations without masking ---

        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1,
                smem_pipe_read,
                smem_pipe_write,
                seqlen_info=seqlen_info,
                is_first_n_block=False,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # --- Final normalize acc_O by row_sum and calculate LSE ---

        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue: Store O and LSE to gmem
        # ///////////////////////////////////////////////////////////////////////////////

        self.epilogue(
            acc_O,
            softmax.row_sum,  # lse
            mO,
            mLSE,
            sO,
            seqlen_info,
            gmem_tiled_copy_O,
            tiled_mma_pv,
            tidx,
            m_block,
            num_head,
            batch_size,
            is_print_thread_and_tile=is_print_thread,
        )

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # tQsQ: (CPY_ATOM=(8,1),CPY_Q8,CPY_HD2):((1,0),1024,8192)
        # tQgQ: (CPY_ATOM=(8,1),CPY_Q8,CPY_HD2):((1,0),16384,64)
        tQsQ: cute.Tensor = gmem_thr_copy.partition_D(sQ)
        tQgQ: cute.Tensor = gmem_thr_copy.partition_S(gQ)

        # cQ: (tileQ128,tileHD128):(1@0,1@1)
        # tQcQ/t0QcQ: (CPY_ATOM=(8,1),CPY_Q8,CPY_HD2):((1@1,0),16@0,64@1)
        # tQpQ: (ATOM_REST_V1,CPY_Q8,CPY_HD2):(2,0,1) => the same predicate along CPY_Q
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ: cute.Tensor = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = cutedsl_utils.predicate_k(tQcQ, limit=headdim)

        sq_limit = seqlen - block * self.tile_m
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):  # loop over CPY_Q8
            # NOTE: Instead of using `tQcQ[0, m, 0][0] < sq_limit`,
            # we use t0QcQ to compare with `sq_limit - tQcQ[0][0]`
            # to make the left hand side a compile-time constant expression.
            if t0QcQ[0, m, 0][0] < sq_limit - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None]
                    if const_expr(self.check_hdim_oob)
                    else None,
                )

        # NOTE: We don't need to clear the sQ smem tiles
        # since we'll only write out the valid outputs

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm80_load_Q] "
                cute.printf("")
                cute.printf(
                    prefix + "block={}, seqlen={}, headdim={}",
                    block,
                    seqlen,
                    headdim,
                )
                cute.printf(prefix + "cQ: {}", cQ.layout)
                cute.printf(prefix + "tQsQ: {}", tQsQ.layout)
                cute.printf(prefix + "tQgQ: {}", tQgQ.layout)
                cute.printf(prefix + "tQcQ: {}", tQcQ.layout)
                cute.printf(prefix + "t0QcQ: {}", t0QcQ.layout)
                cute.printf(prefix + "tQpQ: {}", tQpQ.layout)
                cute.printf("")

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshoot kBlockN when we load K ?
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0

        # tKgK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,restK):((1,0),4096,64,16384)
        # tKsK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,(1,1)):((1,0),1024,4096,(0,0))
        # tKcK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        # t0KcK: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        # tKpK: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1)

        if const_expr(need_predicates or not is_even_n_smem_k):
            if const_expr(is_even_n_smem_k):  # even but need predicates
                sk_limit = seqlen - block * self.tile_n
            else:  # not even
                sk_limit = (
                    cutlass.min(seqlen - block * self.tile_n, self.tile_n)
                    if const_expr(need_predicates)
                    else self.tile_n
                )

            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                # NOTE: Instead of using `tKcK[0, n, 0][0] < sk_limit`,
                # we use t0KcK to compare with `sk_limit - tKcK[0][0]`
                # to make the left hand side a compile-time constant expression.
                if t0KcK[0, n, 0][0] < sk_limit - tKcK[0][0]:
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=tKpK[None, n, None]
                        if const_expr(self.check_hdim_oob)
                        else None,
                    )
        else:  # even and no predicates needed
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

        # NOTE: We don't need to clear the sK smem tiles
        # since we'll mask out the scores anyway.

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm80_load_K] "
                cute.printf("")
                cute.printf(
                    prefix
                    + f"need_predicates={need_predicates}, "
                    + f"is_even_n_smem_k={is_even_n_smem_k}"
                )
                cute.printf(
                    prefix + "block={}, smem_pipe_write={}, seqlen={}",
                    block,
                    smem_pipe_write,
                    seqlen,
                )
                cute.printf(prefix + "tKgK: {}", tKgK.layout)
                cute.printf(prefix + "tKsK: {}", tKsK.layout)
                cute.printf(prefix + "tKcK: {}", tKcK.layout)
                cute.printf(prefix + "t0KcK: {}", t0KcK.layout)
                cute.printf(prefix + "tKpK: {}", tKpK.layout)
                cute.printf("")

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
        is_print_thread_and_tile: bool = False,
    ):
        # TODO(REVIEW): Do we need to check if we overshoot kBlockN when we load V ?
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0

        # tVgV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,restK):((1,0),4096,64,16384)
        # tVsV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2,(1,1)):((1,0),1024,4096,(0,0))
        # tVcV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        # t0VcV: (CPY_ATOM=(8,1),CPY_K4,CPY_HD2):((1@1,0),16@0,64@1)
        # tVpV: (ATOM_REST_V1,CPY_K4,CPY_HD2):(2,0,1)

        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if (
                    is_even_n_smem_v
                    or n < cute.size(tVsV.shape[1]) - 1
                    or tVcV[0, n, 0][0] < self.tile_n
                ):
                    predicate = (
                        tVpV[None, n, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None
                    )
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(
                                cute.size(predicate.shape[0])
                            ):
                                predicate[i, k] = (
                                    tVpV[i, n, k]
                                    if const_expr(self.check_hdim_v_oob)
                                    else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVgV[None, n, None, block],
                        tVsV[
                            None,
                            n,
                            None,
                            smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                        ],
                        pred=predicate,
                    )
        else:  # even and no predicates needed
            cute.copy(
                gmem_tiled_copy,
                tVgV[None, None, None, block],
                tVsV[
                    None,
                    None,
                    None,
                    smem_pipe_write if const_expr(self.num_stages > 1) else 0,
                ],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm80_load_V] "
                cute.printf("")
                cute.printf(
                    prefix
                    + f"need_predicates={need_predicates}, "
                    + f"is_even_n_smem_v={is_even_n_smem_v}"
                )
                cute.printf(
                    prefix + "block={}, smem_pipe_write={}, seqlen={}",
                    block,
                    smem_pipe_write,
                    seqlen,
                )
                cute.printf(prefix + "tVgV: {}", tVgV.layout)
                cute.printf(prefix + "tVsV: {}", tVsV.layout)
                cute.printf(prefix + "tVcV: {}", tVcV.layout)
                cute.printf(prefix + "t0VcV: {}", t0VcV.layout)
                cute.printf(prefix + "tVpV: {}", tVpV.layout)
                cute.printf("")

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
        mask_fn: Callable | None = None,
        is_first_n_block: cutlass.Constexpr[Boolean] = False,
        check_inf: cutlass.Constexpr[Boolean] = True,
        is_print_thread_and_tile: bool = False,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """

        # Define some helper functions
        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    # need predicates for the first tile
                    need_predicates=is_first_n_block and self.num_stages == 1,
                    is_print_thread_and_tile=const_expr(self.num_stages == 1)
                    and is_print_thread_and_tile,
                )
            cute.arch.cp_async_commit_group()

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(
                    n_block - self.num_stages, smem_pipe_write, need_predicates=False
                )
            cute.arch.cp_async_commit_group()

        # --- Apply S = Q*K^T ---

        # Zero-init acc_S
        # acc_S: (MMA_ATOM=(2,2),MMA_Q2,MMA_K8):((1,2),4,8)
        # acc_O: (MMA_ATOM=(2,2),MMA_Q2,MMA_HD16):((1,2),4,8)
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C(
            (self.tile_m, self.tile_n)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)

        # Wait for this K tile and load next V tile
        sync()
        load_V_next()

        # Issue MMA for S = Q * K^T, after S2R copy sQ/sK to rQ/rK
        sm80_utils.gemm(
            tiled_mma=mma_params.thr_mma_qk,
            acc=acc_S,
            tCrA=mma_params.tSrQ,
            tCrB=mma_params.tSrK,
            tCsA=smem_copy_params.tSsQ,
            tCsB=smem_copy_params.tSsK[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_thr_copy_A=smem_copy_params.smem_thr_copy_Q,
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_K,
            A_in_regs=self.Q_in_regs,
        )

        # --- Apply P = softmax(S) ---

        # Apply score_mod if provided
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                acc_S,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen_info=seqlen_info,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )

        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        # For single stage, wait for this V tile and load next K tile here
        # to overlap with softmax below since the pipeline is empty right now
        if const_expr(self.num_stages == 1):
            sync()  # TODO(REVIEW): should we always delay the wait for V until MMA for O ?
            load_K_next()

        # Apply mask_fn if provided
        if const_expr(mask_fn is not None):
            assert mask_fn is not None  # mypy
            mask_fn(acc_S, n_block=n_block)

        # Apply softmax to acc_S and rescale acc_O
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        softmax.rescale_O(mma_params.acc_O, row_scale)

        # Make rP from rS with dtype cast and layout reshape
        # rP: (MMA_ATOM_rC=(2,2),MMA_Q2,MMA_K8):((1,2),4,8)
        # tOrP: (MMA_ATOM_rA=(2,2,2),MMA_Q2,MMA_K4):((1,2,8),4,16)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)

        # --- Apply O = P*V ---

        # For multi stage, we wait for this V tile and load next K tile here
        # to overlap with MMA for O below
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()

        # Issue MMA for O = P * V, after S2R copy sV to rV
        sm80_utils.gemm_rs(
            tiled_mma=mma_params.thr_mma_pv,
            acc=mma_params.acc_O,
            tCrA=tOrP,
            tCrB=mma_params.tOrVt,
            tCsB=smem_copy_params.tOsVt[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_thr_copy_B=smem_copy_params.smem_thr_copy_V,
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm80_compute_one_n_block] "
                cute.printf("")
                cute.printf(
                    prefix
                    + f"is_first_n_block={is_first_n_block}, check_inf={check_inf}"
                )
                cute.printf(
                    prefix + "n_block={}, smem_pipe_read={}, smem_pipe_write={}",
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                )
                cute.printf(prefix + "acc_S: {}", acc_S.layout)
                cute.printf(prefix + "acc_O: {}", mma_params.acc_O.layout)
                cute.printf(prefix + "rP: {}", rP.layout)
                cute.printf(prefix + "tOrP: {}", tOrP.layout)
                cute.printf("")

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk: cute.TiledMma,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        softmax_scale: cutlass.Float32 | None,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

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
            seqlen_info=seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen_info: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        m_block: cutlass.Int32,
        head_idx: cutlass.Int32,
        batch_idx: cutlass.Int32,
        is_print_thread_and_tile: bool = False,
    ):
        # Make rO from acc_O with dtype cast
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))

        # Make sure all threads have finished reading smem
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )

        # R2S copy rO to sO
        if const_expr(self.arch >= Arch.sm_90):
            # HACK: Ampere warp-MMA path. The O accumulator comes from the warp MMA whose
            # permutation_mnk N=16 places two n8 atoms per 16-wide head_dim_v tile.
            # cute.make_tiled_copy_C with the StMatrix atom (selected by get_smem_
            # store_atom whenever this kernel is *compiled* for sm90+, even though it
            # uses the Ampere MMA) mis-maps the second n8 atom, corrupting the high 8
            # of every 16 head_dim_v columns. Store via the MMA's native C partition
            # instead, which handles the N=16 permutation correctly on any target.
            taccOsO = tiled_mma.get_slice(tidx).partition_C(sO)
            cute.autovec_copy(rO, taccOsO)
        else:
            smem_copy_atom_O = cutedsl_utils.get_smem_store_atom(
                self.arch_num, self.dtype
            )
            smem_thr_copy_O = cute.make_tiled_copy_C(
                smem_copy_atom_O, tiled_mma
            ).get_slice(tidx)

            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )

        # R2G copy rLSE to gLSE
        if const_expr(mLSE is not None):
            mLSE_cur = seqlen_info.offset_batch_Q(mLSE, batch_idx, dim=2)[
                None, head_idx
            ]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(gLSE_expanded)
                )
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(
                    thr_mma.get_slice(0).partition_C(cO)
                )
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(
                        cute.size(taccOgLSE.shape[1]), unroll_full=True
                    ):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen_info.seqlen_q
                            - m_block * self.tile_m
                            - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(
                    mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen_info.seqlen_q
                )

        # Make sure sO store is ready
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )

        # S2R copy sO back to rO for wider vectorization
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)

        # R2G copy rO to gO
        sq_limit = seqlen_info.seqlen_q - m_block * self.tile_m
        mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3, ragged=False)[
            None, None, head_idx
        ]
        if const_expr(not self.pack_gqa):
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = cutedsl_utils.predicate_k(tOcO, limit=mO.shape[1])

            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if t0OcO[0, rest_m, 0][0] < sq_limit - tOcO[0][0]:
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen_info.seqlen_q
            )

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm80_epilogue] "
                cute.printf("")
                cute.printf(
                    prefix + "m_block={}, head_idx={}, batch_idx={}",
                    m_block,
                    head_idx,
                    batch_idx,
                )
                cute.printf(prefix + "acc_O: {}", acc_O.layout)
                cute.printf(prefix + "rO: {}", rO.layout)
                cute.printf(prefix + "sO: {}", sO.layout)
                cute.printf(prefix + "cO: {}", cO.layout)
                cute.printf(prefix + "mO_cur: {}", mO_cur.layout)
                cute.printf("")

    @cute.jit
    def advance_pipeline(self, pipeline_index: Int32) -> Int32:
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0
