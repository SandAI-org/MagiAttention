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

# mypy: disable-error-code="arg-type,union-attr,index,misc,assignment"
# pyright: reportInvalidTypeForm=false

# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Literal, Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import Float32, Int32, const_expr, pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

# isort: split
from quack import copy_utils, layout_utils, sm90_utils
from quack.cute_dsl_utils import ParamsBase

from . import cutedsl_utils
from . import pipeline as ffa_pipeline
from .block_info import BlockInfo
from .cutedsl_utils import ThreadCooperativeGroup
from .ffa_utils import MT_MAP
from .mask import AttentionMask
from .named_barrier import NamedBarrierFwd
from .pack_gqa import PackGQA, make_packgqa_tiled_tma_atom, pack_gqa_layout
from .paged_kv import PagedKVManager
from .seqlen_info import SeqlenInfoQK
from .softmax import Softmax, apply_score_mod_inner
from .sparse_utils import (
    BlockSparseTensors,
    consume_block_sparse_loads,
    produce_block_sparse_loads,
)
from .tile_scheduler import (
    SingleTileLPTScheduler,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)


class FFAFwdSm90:
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
        num_stages: int = 2,
        Q_in_regs: bool = False,
        intra_wg_overlap: bool = True,
        mma_pv_is_rs: bool = True,
        paged_kv_non_tma: bool = False,
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
        assert not paged_kv_non_tma or not (
            self.check_hdim_oob or self.check_hdim_v_oob
        ), "Paged KV does not support irregular head dim"

        self.qhead_per_kvhead = qhead_per_kvhead
        self.mask_type = mask_type
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m  # tileQ128
        self.tile_n = tile_n  # tileK128

        self.intra_wg_overlap = intra_wg_overlap
        self.mma_pv_is_rs = mma_pv_is_rs
        self.use_tma_KV = not paged_kv_non_tma
        self.cluster_shape_mn = (1, 1)
        self.num_warps_per_wg = 4
        self.num_threads_per_wg = self.num_warps_per_wg * cute.arch.WARP_SIZE

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
            prefix = "[fwd_sm90_init] "
            print()
            print(f"{prefix}Initialized FFAFwdSm90 with: ")
            print(f"{prefix}{self.dtype=} | {self.arch=} | {self.qhead_per_kvhead=}")
            print(
                f"{prefix}{self.tile_hdim=} | {self.tile_hdimv=} | "
                f"{self.check_hdim_oob=} | {self.check_hdim_v_oob=}"
            )
            print(
                f"{prefix}{self.mask_type=} | {self.is_causal=} | "
                f"{self.is_local=} | {self.pack_gqa=}"
            )
            print(f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.num_stages=}")
            print(f"{prefix}{self.Q_in_regs=} | {self.q_subtile_factor=}")
            print(f"{prefix}{self.score_mod=} | {self.mask_mod=}")
            print(f"{prefix}{self.vec_size=} | {has_aux_tensors=}")
            print(
                f"{prefix}{self.intra_wg_overlap=} | "
                f"{self.mma_pv_is_rs=} | {self.use_tma_KV=} | "
                f"{self.cluster_shape_mn=} | "
            )
            print(
                f"{self.num_warps_per_wg=} | "
                f"{self.num_threads_per_wg=} | "
                f"{self.buffer_align_bytes=}"
            )
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
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.dtype, self.tile_n
                ),
                self.dtype,
            )
        else:
            sP_layout_atom = None

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm90_get_smem_layout_atom] "
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
        # Thr Layout VMNK: (128,2,1,1):(1,128,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,128,16)
        # TV Layout A:     (128,(64,16)):(0,(1,64))
        # TV Layout B:     (128,(128,16)):(0,(1,128))
        # TV Layout C:     ((4,8,4),(2,2,16)):((128,1,16),(64,8,512))
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )

        # Tiled MMA for O=P*V
        # Thr Layout VMNK: (128,2,1,1):(1,128,0,0)
        # Permutation MNK: (_,_,_)
        # MMA Atom
        # ThrID:           128:1
        # Shape MNK:       (64,128,16)
        # TV Layout A:     ((4,8,4),(2,2,2)):((128,1,16),(64,8,512))
        # TV Layout B:     (128,(128,16)):(0,(1,128))
        # TV Layout C:     ((4,8,4),(2,2,16)):((128,1,16),(64,8,512))
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(
                self.tile_m // 64,
                1,
                1,
            ),  # Might need (1, 2, 1) for hdim 512
            tiler_mn=(64, self.tile_hdimv),
            a_source=warpgroup.OperandSource.RMEM
            if self.mma_pv_is_rs
            else warpgroup.OperandSource.SMEM,
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
        cosize_sP = (
            cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        )
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sP], self.buffer_align_bytes
        ]

        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        self.shared_storage_cls = (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

    def _setup_attributes(self):
        # --- Set up tiled MMA ---

        self.tiled_mma_qk, self.tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = self.tiled_mma_qk.size

        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_wg
        assert self.num_wg_mma in [1, 2, 3]

        self.num_wg_load = 1
        self.num_producer_threads = (
            self.num_wg_load * cute.arch.WARP_SIZE
        )  # only first warp in producer WG
        self.num_threads = self.num_threads_per_wg * (
            self.num_wg_mma + self.num_wg_load
        )  # 384 (3 WGs = 1 producer + 2 MMA)
        self.num_warps = self.num_threads // cute.arch.WARP_SIZE  # 12
        self.load_warp_ids = list(range(0, self.num_wg_load * self.num_warps_per_wg))
        self.mma_warp_ids = list(
            range(self.num_wg_load * self.num_warps_per_wg, self.num_warps)
        )
        self.num_Q_load_threads = self.num_threads_per_wg  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads

        # --- Set up smem layout: sQ/sK/sV ---

        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()

        # --- Set up G2S/S2G/R2G tiled copy (legacy from sm80) ---

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

        # tQ: (16,8):(8,1)
        tQ_shape_dim_1 = (
            sQ_layout_atom.outer.shape[1] // async_copy_elems
        )  # 8 for smem_blk=64
        assert (
            self.num_Q_load_threads % tQ_shape_dim_1 == 0
        ), "num_Q_load_threads must be divisible by tQ_shape_dim_1"
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQ_shape_dim_1, tQ_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.tile_m % tQ_layout.shape[0] == 0

        # tO: (32,8):(8,1)
        tO_shape_dim_1 = (
            sO_layout_atom.outer.shape[1] // async_copy_elems
        )  # 8 for smem_blk=64
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.tile_m % tO_layout.shape[0] == 0

        # G2S async tiled_copy_Q:
        # layout_src_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        # layout_dst_tv_tiled=((8,16),(8,1)):((128,1),(16,0))
        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            atom_async_copy, tQ_layout, vQKV_layout
        )

        # R2G universal tiled_copy_O:
        # layout_src_tv_tiled=(8,32),(8,1)):((256,1),(32,0))
        # layout_dst_tv_tiled=(8,32),(8,1)):((256,1),(32,0))
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

        # --- Set up other attrs ---

        self.use_scheduler_barrier = (
            (self.num_wg_mma >= 2 and self.tile_hdim <= 128)
            if const_expr(self.intra_wg_overlap)
            else (self.num_wg_mma == 2)
        )
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        self.use_tma_O = self.use_tma_Q
        self.rescale_O_before_gemm = self.tile_hdimv > 128 and self.intra_wg_overlap

        # --- Set up registers ---

        self.num_mma_regs, self.num_producer_regs = {
            1: (256, 56),
            2: (240, 24),
            3: (160, 32),
        }[self.num_wg_mma]

        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(
            self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)
        ):
            self.num_mma_regs, self.num_producer_regs = 224, 40

        if const_expr(self.debug_print):
            # NOTE: we need extra registers for load warp to debug print
            # otherwise, it will raise illegal instruction error
            num_regs_for_print = 24
            self.num_producer_regs += num_regs_for_print
            self.num_mma_regs -= num_regs_for_print

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm90_setup_attributes] "
            print()
            print(f"{prefix}{self.num_threads=} | {self.num_warps=}")
            print(f"{prefix}{self.num_wg_mma=} | {self.num_wg_load=}")
            print(f"{prefix}{self.num_mma_threads=}")
            print(f"{prefix}{self.num_producer_threads=}")
            print(f"{prefix}{self.num_Q_load_threads=}")
            print(f"{prefix}{self.num_epilogue_threads=}")
            print(f"{prefix}{self.load_warp_ids=} | {self.mma_warp_ids=}")
            print()
            print(f"{prefix}{self.num_mma_regs=} | {self.num_producer_regs=}")
            print(f"{prefix}{self.use_scheduler_barrier=}")
            print(f"{prefix}{self.use_tma_Q=} | {self.use_tma_KV=} | {self.use_tma_O=}")
            print(f"{prefix}{self.rescale_O_before_gemm=}")
            print()
            print(f"{prefix}tiled_mma_qk: {self.tiled_mma_qk}")
            print(f"{prefix}tiled_mma_pv: {self.tiled_mma_pv}")
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
            print(f"{prefix}tQ_shape_dim_1: {tQ_shape_dim_1}")
            print(f"{prefix}tO_shape_dim_1: {tO_shape_dim_1}")
            print(f"{prefix}tQ_layout: {tQ_layout}")
            print(f"{prefix}tO_layout: {tO_layout}")
            print()
            print(
                f"{prefix}gmem_tiled_copy_Q: "
                f"layout_src_tv_tiled={self.gmem_tiled_copy_Q.layout_src_tv_tiled} | "
                f"layout_dst_tv_tiled={self.gmem_tiled_copy_Q.layout_dst_tv_tiled}"
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
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
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

        self._check_type(
            *(
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

        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        # --- Set up attributes ---

        self._setup_attributes()

        # ///////////////////////////////////////////////////////////////////////////////
        # Make mQ/mK/mV/mO/mLSE tensors
        # with layout transformations for specific memory access patterns
        # ///////////////////////////////////////////////////////////////////////////////

        # mQ: (sQ,HD,nhQ,batch):(nhQ*HD,1,HD,sQ*nhQ*HD)
        # mK: (sK,HD,nhK,batch):(nhK*HD,1,HD,sK*nhK*HD)
        # mV: (sK,HD,nhK,batch):(nhK*HD,1,HD,sK*nhK*HD)
        # mO: (sQ,HD,nhQ,batch):(nhQ*HD,1,HD,sQ*nhQ*HD)
        mQ, mK, mV, mO = [
            cutedsl_utils.assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)
        ]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]

        # mLSE: (sQ,nhQ,batch):(1,sQ,sQ*nhQ)
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        # mQ/mO_packgqa: ((nhG,sQ),HD,nhK,batch):((HD,nhQ*HD),1,nhG*HD,sQ*nhQ*HD)
        # mLSE_packgqa: ((nhG,sQ),nhK,batch):((sQ,1),sQ*nhG,sQ*nhQ)
        # where nhG = nhQ / nhK = self.qhead_per_kvhead
        mQ_og, mO_og, mLSE_og = mQ, mO, mLSE
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        # ///////////////////////////////////////////////////////////////////////////////
        # Set up smem layout: sQ/sK/sV
        # ///////////////////////////////////////////////////////////////////////////////

        # sQ/sO: S<3,4,3> o 0 o ((ATOM_Q8,LAY_tileQ16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sK/sV: S<3,4,3> o 0 o ((ATOM_K8,LAY_tileK8),(ATOM_HD64,LAY_tileHD2),STAGE=(1,2)):((64,512),(1,8192),(0,16384))
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(
                mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage
            )
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                (mO, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make TMA tiled copy atom and tensors
        # ///////////////////////////////////////////////////////////////////////////////

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(
                make_packgqa_tiled_tma_atom,
                qhead_per_kvhead=self.qhead_per_kvhead,
                head_idx=2,
            )
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )

        # tma_atom_Q: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_Q: ((nhG,sQ),HD,nhK,batch):((1@1,1@2),1@0,4@1,1@3)
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            g2s_copy_op_Q = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                g2s_copy_op_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )

        # tma_atom_K: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_K: (sK,HD,nhK,batch):(1@1,1@0,1@2,1@3)
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(self.use_tma_KV):
            g2s_copy_op_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                g2s_copy_op_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                g2s_copy_op_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )

        # tma_atom_O: layout_src_tv=(1,8192):(0,1), layout_dst_tv=(1,8192):(0,1)
        # tma_tensor_O: ((nhG,sQ),HD,nhK,batch):((1@1,1@2),1@0,4@1,1@3)
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            s2g_copy_op_O = cpasync.CopyBulkTensorTileS2GOp()
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                s2g_copy_op_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )

        # ///////////////////////////////////////////////////////////////////////////////
        # Make tile scheduler class/args, SMEM storage, and others
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Make tile scheduler class/args ---

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            self.tile_scheduler_cls = SingleTileVarlenScheduler
        else:
            self.tile_scheduler_cls = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
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
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = cutedsl_utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Launch the kernel
        # ///////////////////////////////////////////////////////////////////////////////

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm90_call] "

            cute.printf("")
            cute.printf(prefix + "mQ.layout: {}", mQ.layout)
            cute.printf(prefix + "mK.layout: {}", mK.layout)
            cute.printf(prefix + "mV.layout: {}", mV.layout)
            cute.printf(prefix + "mO.layout: {}", mO.layout)
            cute.printf(prefix + "mQ_og.layout: {}", mQ_og.layout)
            cute.printf(prefix + "mO_og.layout: {}", mO_og.layout)
            if const_expr(mLSE is not None):
                cute.printf(prefix + "mLSE.layout: {}", mLSE.layout)
                cute.printf(prefix + "mLSE_og.layout: {}", mLSE_og.layout)
            cute.printf("")
            cute.printf(prefix + "sQ_layout: {}", self.sQ_layout)
            cute.printf(prefix + "sK_layout: {}", self.sK_layout)
            cute.printf(prefix + "sV_layout: {}", self.sV_layout)
            cute.printf(prefix + "sO_layout: {}", self.sO_layout)
            cute.printf(prefix + "sP_layout: {}", self.sP_layout)
            cute.printf("")
            cute.printf(
                prefix + "tma_copy_bytes: Q={}, K={}, V={}",
                self.tma_copy_bytes["Q"],
                self.tma_copy_bytes["K"],
                self.tma_copy_bytes["V"],
            )
            if const_expr(self.use_tma_Q):
                cute.printf(
                    prefix + "tma_atom_Q: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_Q.layout_src_tv,
                    tma_atom_Q.layout_dst_tv,
                )
                cute.printf(prefix + "tma_tensor_Q.layout: {}", tma_tensor_Q.layout)
            if const_expr(self.use_tma_KV):
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
            if const_expr(self.use_tma_O):
                cute.printf(
                    prefix + "tma_atom_O: layout_src_tv={}, layout_dst_tv={}",
                    tma_atom_O.layout_src_tv,
                    tma_atom_O.layout_dst_tv,
                )
                cute.printf(prefix + "tma_tensor_O.layout: {}", tma_tensor_O.layout)
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
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_O,
            self.tiled_mma_qk,
            self.tiled_mma_pv,
            tile_sched_params,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
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
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
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
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # --- Alloc smem storage and fetch ptrs ---

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        # --- Make pipelines ---

        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_wg)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = ffa_pipeline.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = ffa_pipeline.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = ffa_pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = ffa_pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = ffa_pipeline.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = ffa_pipeline.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # --- Cluster arrive after mbarrier init ---

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # --- Make smem tensors of sQ/sK/sV/sO/sP ---

        # sQ: ((ATOM_Q8,LAY_tileQ16),(ATOM_HD64,LAY_tileHD2)):((64,512),(1,8192))
        # sK: ((ATOM_K8,LAY_tileK16),(ATOM_HD64,LAY_tileHD2),STAGE_K=(1,2)):((64,512),(1,8192),(0,16384))
        # sV: ((ATOM_K8,LAY_tileK16),(ATOM_HDV64,LAY_tileHD2),STAGE_V=(1,2)):((64,512),(1,8192),(0,16384))
        # sVt: ((ATOM_HDV64,LAY_tileHD2),(ATOM_K8,LAY_tileK16),STAGE_V=(1,2)):((1,8192),(64,512),(0,16384))
        # sO: ((ATOM_Q8,LAY_tileQ16),(ATOM_HDV64,LAY_tileHD2)):((64,512),(1,8192))
        sQ: cute.Tensor = storage.sQ.get_tensor(
            sQ_layout.outer, swizzle=sQ_layout.inner
        )
        sK: cute.Tensor = storage.sK.get_tensor(
            sK_layout.outer, swizzle=sK_layout.inner
        )
        if const_expr(not self.Q_in_regs):
            sV: cute.Tensor = storage.sV.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner
            )
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt: cute.Tensor = layout_utils.transpose_view(sV)
        sP: cute.Tensor | None = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # Reuse sQ's buffer for sO
        sO: cute.Tensor = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype
        )

        # --- Make other info dataclass ---

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
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0]
            if const_expr(not self.pack_gqa)
            else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            mCuTotalMBlocks=(
                blocksparse_tensors.cu_total_m_blocks
                if blocksparse_tensors is not None
                else None
            ),
            mCuBlockIdxOffsets=(
                blocksparse_tensors.cu_block_idx_offsets
                if blocksparse_tensors is not None
                else None
            ),
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )

        # --- Cluster wait before warp specialization ---

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- Make tile scheduler ---

        tile_scheduler = self.tile_scheduler_cls.create(tile_sched_params)
        assert isinstance(
            tile_scheduler, TileSchedulerProtocol
        ), f"tile_scheduler is not a TileSchedulerProtocol: {type(tile_scheduler)}"

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread:
                prefix = "[fwd_sm90_kernel_setup] "
                cute.printf("")
                cute.printf(prefix + "warp_idx={} tidx={}", warp_idx, tidx)
                cute.printf(
                    prefix + "num_threads={} num_producer_regs={} num_mma_regs={}",
                    self.num_threads,
                    self.num_producer_regs,
                    self.num_mma_regs,
                )
                cute.printf("")
                cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                cute.printf(prefix + "sK.layout: {}", sK.layout)
                cute.printf(prefix + "sV.layout: {}", sV.layout)
                cute.printf(prefix + "sVt.layout: {}", sVt.layout)
                cute.printf(prefix + "sO.layout: {}", sO.layout)
                if const_expr(sP is not None):
                    cute.printf(prefix + "sP.layout: {}", sP.layout)
                cute.printf("")

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load WarpGroup
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.load_warp_ids[-1]:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)

            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_Q,
                mPageTable,
                blocksparse_tensors,
                block_info,
                SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                is_print_block=is_print_block,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA WarpGroups
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.mma_warp_ids[0]:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                pipeline_q,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                tile_scheduler,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
                is_print_block=is_print_block,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        mPageTable: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        tile_scheduler: TileSchedulerProtocol,
        is_print_block: bool = False,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % len(
            self.load_warp_ids
        )
        tidx, _, _ = cute.arch.thread_idx()

        # NOTE: TMA: only warp 0 loads | cp_async: all warps load
        # When not use_tma_Q, all 128 producer threads participate in Q loading.
        is_load_warp = warp_idx_in_wg == 0 or const_expr(
            not self.use_tma_KV or not self.use_tma_Q
        )

        # KV loading restricted to warp 0 for TMA, all warps for non-TMA KV
        is_kv_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_KV)

        if is_load_warp:
            q_producer_phase = Int32(1)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )

            # ///////////////////////////////////////////////////////////////////////////////
            #  Persistent tile scheduler loop
            # ///////////////////////////////////////////////////////////////////////////////
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                # --- Get current tile info ---

                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen_info = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )

                # Used only for debug print
                is_print_thread_and_tile = const_expr(self.debug_print) and (
                    (tidx == 0)
                    and is_print_block
                    and (m_block == 0)
                    and (head_idx == 0)
                    and (batch_idx == 0)
                )

                # //////////////////////////////////////////////
                #  Make gQ/gK/gV with TMA load partial fns
                # //////////////////////////////////////////////

                # mQ_cur: ((nhG,sQ),HD):((1@1,1@2),1@0)
                # gQ: ((nhG,tileQ128//nhG),tileHD128):((1@1,1@2),1@0)
                mQ_cur = seqlen_info.offset_batch_Q(mQ, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                load_Q = None
                if const_expr(self.use_tma_Q):
                    gQ = cute.local_tile(
                        mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0)
                    )
                    load_Q, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                    )

                paged_kv_manager = None
                tma_load_K_fn = None
                tma_load_V_fn = None
                if const_expr(self.use_tma_KV):
                    if const_expr(mPageTable is not None):  # TODO: review the logics
                        # Paged TMA: keep page dimension indexable
                        mK_cur = mK[None, None, head_idx_kv, None]
                        mV_cur = mV[None, None, head_idx_kv, None]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (0, 0, None)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (0, 0, None)
                        )
                    else:
                        # mK_cur/mV_cur: (sK,HD):(1@1,1@0)
                        # gK/gV: (tileK128,tileHD128,restK):(1@1,1@0,128@1)
                        # where restK = sK // tileK128
                        mK_cur = seqlen_info.offset_batch_K(mK, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        mV_cur = seqlen_info.offset_batch_K(mV, batch_idx, dim=3)[
                            None, None, head_idx_kv
                        ]
                        gK = cute.local_tile(
                            mK_cur, (self.tile_n, self.tile_hdim), (None, 0)
                        )
                        gV = cute.local_tile(
                            mV_cur, (self.tile_n, self.tile_hdimv), (None, 0)
                        )

                    tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K, 0, cute.make_layout(1), gK, sK
                    )
                    tma_load_K_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_K_fn, pipeline_k
                    )
                    tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V, 0, cute.make_layout(1), gV, sV
                    )
                    tma_load_V_fn = copy_utils.tma_producer_copy_fn(
                        tma_load_V_fn, pipeline_v
                    )
                else:  # cp.async path
                    paged_kv_manager = PagedKVManager.create(
                        mPageTable,
                        mK,
                        mV,
                        FastDivmodDivisor(mK.shape[0]),
                        batch_idx,
                        head_idx_kv,
                        tidx,
                        seqlen_info.seqlen_k,
                        0,  # leftpad_k
                        self.tile_n,
                        self.tile_hdim,
                        self.tile_hdimv,
                        self.num_threads_per_wg,
                        mK.element_type,
                        arch=self.arch_num,
                    )

                load_K = partial(
                    self.load_KV,
                    tma_load_K_fn,
                    paged_kv_manager,
                    sK,
                    pipeline_kv=pipeline_k,
                    K_or_V="K",
                )
                load_V = partial(
                    self.load_KV,
                    tma_load_V_fn,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_v,
                    K_or_V="V",
                )

                # --- Debug print ---

                if const_expr(self.debug_print):
                    if is_print_thread_and_tile:
                        prefix = "[fwd_sm90_load] "
                        cute.printf("")
                        cute.printf(
                            prefix + "m_block={} head_idx={} batch_idx={}",
                            m_block,
                            head_idx,
                            batch_idx,
                        )
                        cute.printf(prefix + "mQ_cur.layout: {}", mQ_cur.layout)
                        if const_expr(self.use_tma_KV):
                            cute.printf(prefix + "mK_cur.layout: {}", mK_cur.layout)
                            cute.printf(prefix + "mV_cur.layout: {}", mV_cur.layout)
                        cute.printf("")
                        if const_expr(self.use_tma_Q):
                            cute.printf(prefix + "gQ.layout: {}", gQ.layout)
                        if const_expr(self.use_tma_KV):
                            cute.printf(prefix + "gK.layout: {}", gK.layout)
                            cute.printf(prefix + "gV.layout: {}", gV.layout)
                        cute.printf("")
                        cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                        cute.printf(prefix + "sK.layout: {}", sK.layout)
                        cute.printf(prefix + "sV.layout: {}", sV.layout)
                        cute.printf("")

                # //////////////////////////////////////////////
                #  G2S-load sQ/sK/sV
                # //////////////////////////////////////////////

                pack_gqa = None
                if const_expr(not self.use_tma_Q):
                    pack_gqa = PackGQA(
                        self.tile_m,
                        self.tile_hdim,
                        self.check_hdim_oob,
                        self.qhead_per_kvhead,
                    )

                if const_expr(not self.use_block_sparsity):
                    n_block_min, n_block_max = block_info.get_n_block_min_max(
                        seqlen_info, m_block
                    )

                    # Clamp n_block to 0 when n_block_max == 0
                    # (can happen with causal + pack_gqa when seqlen_k < tile_n).
                    # TMA handles n_block=-1 gracefully (fills zeros),
                    # but cp.async would crash on out-of-bounds page table access.
                    n_block = (
                        n_block_max - 1
                        if const_expr(self.use_tma_KV)
                        else cutlass.max(n_block_max - 1, 0)
                    )
                    page_idx = (
                        mPageTable[batch_idx, n_block]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )

                    # --- Prologue: load Q,K0 ---

                    # First iteration: load K on pipeline_k, Q on pipeline_q
                    if is_kv_load_warp:
                        pipeline_k.producer_acquire(kv_producer_state)
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                        load_K(
                            block=n_block,
                            producer_state=kv_producer_state,
                            page_idx=page_idx,
                        )
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen_info.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1

                    # --- Mainloop/Epilogue: load Ki,Vi ---

                    if is_kv_load_warp:
                        if const_expr(not self.intra_wg_overlap or not self.use_tma_KV):
                            # --- Mainloop0: load V0 ---

                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()

                            # --- Mainloop1: load Ki,Vi ---

                            for i in cutlass.range(
                                n_block_max - 1 - n_block_min, unroll=1
                            ):
                                n_block = n_block_max - 1 - i - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(
                                        mPageTable is not None and self.use_tma_KV
                                    )
                                    else None
                                )
                                if const_expr(not self.use_tma_KV):
                                    paged_kv_manager.load_page_table(n_block)
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state)
                                load_V(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                kv_producer_state.advance()
                        else:
                            # --- Mainloop: load Ki,V(i-1) ---

                            for i in cutlass.range(
                                n_block_max - 1 - n_block_min, unroll=1
                            ):
                                n_block_prev = n_block_max - i - 1
                                n_block = n_block_prev - 1
                                page_idx = (
                                    mPageTable[batch_idx, n_block]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                page_idx_prev = (
                                    mPageTable[batch_idx, n_block_prev]
                                    if const_expr(mPageTable is not None)
                                    else None
                                )
                                kv_producer_state_prev = kv_producer_state.clone()
                                kv_producer_state.advance()
                                pipeline_k.producer_acquire(kv_producer_state)
                                load_K(
                                    block=n_block,
                                    producer_state=kv_producer_state,
                                    page_idx=page_idx,
                                )
                                pipeline_v.producer_acquire(kv_producer_state_prev)
                                load_V(
                                    block=n_block_prev,
                                    producer_state=kv_producer_state_prev,
                                    page_idx=page_idx_prev,
                                )

                            # --- Epilogue: load V(-1) ---

                            n_block = n_block_min
                            page_idx = (
                                mPageTable[batch_idx, n_block]
                                if const_expr(mPageTable is not None)
                                else None
                            )
                            pipeline_v.producer_acquire(kv_producer_state)
                            load_V(
                                block=n_block,
                                producer_state=kv_producer_state,
                                page_idx=page_idx,
                            )
                            kv_producer_state.advance()
                else:  # block sparse load (TODO: review the logics)
                    # Block sparsity: use TMA closures directly (not paged)
                    # Load Q on pipeline_q, separate from K/V pipeline
                    if const_expr(self.use_tma_Q):
                        if warp_idx_in_wg == 0:
                            pipeline_q.producer_acquire_w_index_phase(
                                0, q_producer_phase
                            )
                            load_Q(
                                tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0)
                            )
                            q_producer_phase ^= 1
                    else:
                        pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                        pack_gqa.load_Q(
                            mQ_cur,
                            sQ,
                            gmem_tiled_copy_Q,
                            tidx,
                            m_block,
                            seqlen_info.seqlen_q,
                        )
                        cute.arch.cp_async_commit_group()
                        pipeline_q.producer_commit_w_index(0)
                        q_producer_phase ^= 1
                    if is_kv_load_warp:
                        kv_producer_state = produce_block_sparse_loads(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            m_block,
                            seqlen_info,
                            kv_producer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            self.intra_wg_overlap,
                            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                            self.q_subtile_factor
                            if self.q_subtile_factor is not None
                            else 1,
                        )

                # Advance to next Q tile
                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # NOTE: Producer tail is only useful for cluster to avoid early exit of blocks.
            # We only need producer_tail on V since that's the last that's loaded,
            # and we don't need it for Q (no cluster) and K.
            if is_kv_load_warp:
                pipeline_v.producer_tail(kv_producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_load_fn: Optional[Callable],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
    ):
        if const_expr(self.use_tma_KV):
            src_idx = block if const_expr(page_idx is None) else page_idx
            tma_load_fn(src_idx=src_idx, producer_state=producer_state)
        else:  # cp.async
            paged_kv_manager.load_KV(
                block, sX[None, None, producer_state.index], K_or_V
            )
            cute.arch.cp_async_commit_group()

        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable[..., SeqlenInfoQK],
        AttentionMaskCls: Callable[..., AttentionMask],
        tile_scheduler: TileSchedulerProtocol,
        blocksparse_tensors: Optional[BlockSparseTensors],
        aux_tensors: Optional[list],
        fastdiv_mods=None,
        is_print_block: bool = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tidx -= self.mma_warp_ids[0] * cute.arch.WARP_SIZE
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_wg)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_wg
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled MMA partitions with partial MMA fns
        # ///////////////////////////////////////////////////////////////////////////////

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))

        # --- S = Q @ K.T ---

        # tSrQ: (MMA_ATOM=1,MMA_Q1,MMA_HD=(4,2)):(0,0,(2,1024))
        # tSrK: (MMA_ATOM=1,MMA_K1,MMA_HD=(4,2),STAGE_K=(1,2)):(0,0,(2,1024),(0,2048))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init,
            tiled_mma_qk,
            (self.tile_m, self.tile_n),
            tSrQ,
            tSrK,
        )

        # --- O = P @ V ---

        # acc_O:  (MMA_ATOM=(2,2,16),MMA_Q1,MMA_HD1):((1,2,4),0,0)
        # tOrP:   (MMA_ATOM=(2,2,2),MMA_Q1,MMA_K8):((1,2,4),0,8)
        # tOrVt:  (MMA_ATOM=1,MMA_HD1,MMA_K8,STAGE_V=(1,2)):(0,0,128,(0,2048))
        acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # R2S tiled copy atom and partition of P
        # ///////////////////////////////////////////////////////////////////////////////

        smem_copy_atom_P = cutedsl_utils.get_smem_store_atom(self.arch_num, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(
            smem_copy_atom_P, tiled_mma_qk
        ).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        # ///////////////////////////////////////////////////////////////////////////////
        # Make others before persistent tile scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////

        # Arrive between MMA WGs
        if const_expr(self.use_scheduler_barrier):
            if cutedsl_utils.canonical_warp_group_idx(sync=False) == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_wg,
                )

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )

        # Make softmax object
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        # For RescaleOBeforeGemm: persistent scores_scale across iterations
        scores_scale = None
        if const_expr(self.rescale_O_before_gemm):
            scores_scale = cute.make_rmem_tensor_like(softmax.row_max, Float32)

        # Make partial functions for MMA and first/last half block processing
        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
            scores_scale=scores_scale,
        )
        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
            scores_scale=scores_scale,
            softmax=softmax,
            acc_O=acc_O,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # ///////////////////////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # --- Get current tile info ---

            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen_info = SeqlenInfoCls(batch_idx)

            # Used only for debug print
            is_print_thread_and_tile = const_expr(self.debug_print) and (
                (tidx == 128)
                and is_print_block
                and (m_block == 0)
                and (head_idx == 0)
                and (batch_idx == 0)
            )

            # --- Make mask fn and score-mod fn ---

            # Recompute fastdiv_mods if necessary for varlen with aux_tensors
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen_info.has_cu_seqlens_q or seqlen_info.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None
                and (seqlen_info.has_cu_seqlens_k or seqlen_info.has_seqused_k)
            )
            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen_info.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen_info.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen_info)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )

            # --- Make MMA one n-block fn ---

            mma_one_n_block = partial(
                mma_one_n_block_all,
                seqlen=seqlen_info,
                softmax=softmax,
                score_mod_fn=score_mod_fn,
            )

            # For performance reason, we separate out two kinds of iterations:
            # those that need masking on S, and those that don't.
            # We need masking on S for the very last block when K and V has length not multiple of tile_n.
            # We also need masking on S if it's causal, for the last several blocks.
            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen_info, m_block
            )
            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            # softmax.reset()  # Don't need reset as we explicitly call softmax w is_first=True
            O_should_accumulate = False

            # --- Debug print ---

            if const_expr(self.debug_print):
                if is_print_thread_and_tile:
                    prefix = "[fwd_sm90_mma] "
                    cute.printf("")
                    cute.printf(
                        prefix + "m_block={} head_idx={} batch_idx={}",
                        m_block,
                        head_idx,
                        batch_idx,
                    )
                    cute.printf("")
                    cute.printf(prefix + "sQ.layout: {}", sQ.layout)
                    cute.printf(prefix + "sK.layout: {}", sK.layout)
                    cute.printf(prefix + "sVt.layout: {}", sVt.layout)
                    cute.printf(prefix + "sO.layout: {}", sO.layout)
                    cute.printf("")
                    cute.printf(prefix + "tSrQ.layout: {}", tSrQ.layout)
                    cute.printf(prefix + "tSrK.layout: {}", tSrK.layout)
                    cute.printf(prefix + "acc_O.layout: {}", acc_O.layout)
                    cute.printf(prefix + "tOrP.layout: {}", tOrP.layout)
                    cute.printf(prefix + "tOrVt.layout: {}", tOrVt.layout)
                    cute.printf("")

            # --- Mainloop ---

            if const_expr(not self.use_block_sparsity):
                # --- First ("half") iteration with seqlen masking ---

                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_first_half_block(
                        n_block=n_block_max - 1,
                        seqlen=seqlen_info,
                        kv_consumer_state=kv_consumer_state,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod),
                        score_mod_fn=score_mod_fn,
                        is_first_block=True,
                        is_print_thread_and_tile=is_print_thread_and_tile,
                    )
                else:
                    self.warp_scheduler_barrier_sync()
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1,
                        seqlen=seqlen_info,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                        is_first_n_block=True,
                        mask_fn=partial(
                            mask_fn, mask_mod=self.mask_mod, mask_seqlen=True
                        ),
                    )
                    O_should_accumulate = True

                n_block_max -= 1

                # --- Next couple of iterations with causal/local masking ---

                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = (
                        block_info.get_n_block_min_causal_local_mask(
                            seqlen_info, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen_info,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_causal_local_mask
                    )

                # --- The remaining iterations have no masking ---

                n_block_min_before_local_mask = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen_info, m_block, n_block_min
                    )
                )

                for n_tile in cutlass.range(
                    n_block_max - n_block_min_before_local_mask, unroll=1
                ):
                    kv_consumer_state = mma_one_n_block(
                        kv_consumer_state,
                        n_block=n_block_max - 1 - n_tile,
                        seqlen=seqlen_info,
                        mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        mask_fn=partial(
                            mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                        ),
                        is_print_thread_and_tile=(
                            is_print_thread_and_tile and n_tile == 0
                        ),
                    )
                    O_should_accumulate = True

                # --- Separate iterations with local masking on the left ---

                if const_expr(
                    self.is_local and block_info.window_size_left is not None
                ):
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_before_local_mask
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            seqlen=seqlen_info,
                            mma_pv_fn=partial(
                                mma_pv_fn, zero_init=not O_should_accumulate
                            ),
                            mask_fn=partial(
                                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
                            ),
                        )
                        O_should_accumulate = True

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # --- Last "half" iteration ---

                if const_expr(self.intra_wg_overlap):
                    kv_consumer_state = process_last_half_block(
                        kv_consumer_state=kv_consumer_state,
                        zero_init=not O_should_accumulate,
                        is_print_thread_and_tile=is_print_thread_and_tile,
                    )
                    O_should_accumulate = True
                else:
                    self.warp_scheduler_barrier_arrive()

            else:  # block sparse mma (TODO: review the logics)
                (
                    kv_consumer_state,
                    O_should_accumulate,
                    processed_any,
                ) = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    seqlen_info,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    mask_fn,
                    score_mod_fn,
                    O_should_accumulate,
                    self.mask_mod,
                    fastdiv_mods,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

                # Release Q pipeline so the producer can load the next tile's Q
                pipeline_q.consumer_release_w_index(0)

                # Handle empty case (when no blocks to process)
                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            q_consumer_phase ^= 1

            # --- Apply attention sink ---

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:  # Each thread might have a different sink value due to different q_head
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])

            # --- Final normalize acc_O by row_sum and calculate LSE ---

            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)

            # --- Epilogue ---

            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen_info,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            # Advance to next Q tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def first_half_block_overlap(
        self,
        n_block: Int32,
        mma_qk_fn: Callable,
        kv_consumer_state,
        pipeline_k,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        acc_O: Optional[cute.Tensor] = None,
        mask_fn: Callable = None,
        score_mod_fn: Optional[Callable] = None,
        is_first_block: bool = False,
        is_print_thread_and_tile: bool = False,
    ):
        """Processes the first half block when using intra-warpgroup-overlap.

        Sub-process (the "first half" of an n_block, overlapped with the next QK GEMM):
          1. S = Q @ K.T   (wait K full, GEMM, release K)
          2. score_mod + seqlen mask on S
          3. online softmax  -> row_scale
          4. convert P (fp32 S -> dtype) and copy to smem for the PV GEMM
          5. (RescaleOBeforeGemm) init acc_O / stash row_scale
        """

        # --- S = Q @ K.T ---

        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=0)
        pipeline_k.consumer_release(kv_consumer_state)

        # --- Score mod + mask ---

        # Apply score modification if present
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)

        # Apply mask; mask_seqlen always True for first block
        # Caveat: if full block further right than mask block, seqlen masking is redundant;
        # however, masking is being applied anyway, so essentially no perf hit
        mask_fn(acc_S, n_block=n_block, mask_seqlen=True)

        # --- Online softmax ---

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_block)

        # --- Convert P and copy to smem ---

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        tOrP_cur.store(tOrP_acc.load().to(self.dtype))

        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
            # Fence and barrier to make smem store visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        # --- RescaleOBeforeGemm: init acc_O ---

        # For RescaleOBeforeGemm: initialize acc_O
        if const_expr(self.rescale_O_before_gemm):
            acc_O.fill(0.0)
            scores_scale.store(row_scale.load())

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm90_first_half_block_overlap] "
                cute.printf("")
                cute.printf(prefix + "n_block={}", n_block)
                cute.printf(prefix + "acc_S.layout: {}", acc_S.layout)
                cute.printf(prefix + "row_scale.layout: {}", row_scale.layout)
                cute.printf("")

        return kv_consumer_state

    @cute.jit
    def last_half_block_overlap(
        self,
        kv_consumer_state,
        pipeline_v,
        mma_pv_fn: Callable,
        zero_init: bool,
        scores_scale: Optional[cute.Tensor] = None,
        softmax: Optional[Softmax] = None,
        acc_O: Optional[cute.Tensor] = None,
        is_print_thread_and_tile: bool = False,
    ):
        """Processes the final PV GEMM when using intra-warpgroup-overlap.

        Sub-process (the "last half": the dangling O += P @ V for the final n_block):
          1. (RescaleOBeforeGemm) rescale acc_O by the stashed scores_scale
          2. O += P @ V   (wait V full, GEMM, release V, advance pipeline state)
        """

        # For RescaleOBeforeGemm: rescale O before the final PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        # --- O += P @ V ---

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        mma_pv_fn(B_idx=kv_consumer_state.index, zero_init=zero_init, wg_wait=0)
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm90_last_half_block_overlap] "
                cute.printf("")
                cute.printf(prefix + "zero_init={}", zero_init)
                cute.printf(prefix + "acc_O.layout: {}", acc_O.layout)
                cute.printf("")

        return kv_consumer_state

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | ffa_pipeline.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,  # not used
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        is_print_thread_and_tile: bool = False,
    ):
        """Process one full n_block (non-overlap path).

        Sub-process:
          1. S = Q @ K.T   (wait K full, GEMM, scheduler-barrier handoff, release K)
          2. score_mod + mask on S
          3. online softmax  -> row_scale
          4. convert P (fp32 S -> dtype) and copy to smem
          5. rescale acc_O by row_scale
          6. O += P @ V   (wait V full, GEMM, release V, advance pipeline state)
        """

        # --- S = Q @ K.T ---

        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # --- Score mod + mask ---

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        # --- Online softmax ---

        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )

        # --- Convert P and copy to smem ---

        # if cute.arch.thread_idx()[0] == 0: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        cutedsl_utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)

        # --- Rescale O ---

        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV

        # --- O += P @ V ---

        pipeline_v.consumer_wait(
            smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()

        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm90_mma_one_n_block] "
                cute.printf("")
                cute.printf(prefix + "n_block={}", n_block)
                cute.printf(prefix + "acc_S.layout: {}", acc_S.layout)
                cute.printf(prefix + "row_scale.layout: {}", row_scale.layout)
                cute.printf(prefix + "acc_O.layout: {}", acc_O.layout)
                cute.printf("")

        return smem_pipe_read

    @cute.jit
    def mma_one_n_block_intrawg_overlap(
        self,
        smem_pipe_read: pipeline.PipelineState | ffa_pipeline.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        check_inf: cutlass.Constexpr = True,
        is_print_thread_and_tile: bool = False,
    ):
        """Process one n_block with intra-warpgroup overlap.

        Unlike mma_one_n_block, the QK GEMM of THIS block is issued together with the
        PV GEMM of the PREVIOUS block (whose P is already in smem), so the two GEMMs of
        adjacent blocks overlap inside the same warp group:
          1. issue S(i) = Q @ K(i).T   (wg_wait=-1, don't block)
          2. (RescaleOBeforeGemm) rescale acc_O while QK is in flight
          3. issue O += P(i-1) @ V(i-1)  (wg_wait=-1, don't block)
          4. wait QK(i) -> score_mod + mask + online softmax -> convert P(i) to smem
          5. wait PV(i-1) done, release V(i-1)
        The dangling PV(last) is later flushed by last_half_block_overlap.
        """

        # --- Issue S(i) = Q @ K(i).T (overlapped) ---

        smem_pipe_read_v = smem_pipe_read.clone()
        smem_pipe_read.advance()
        pipeline_k.consumer_wait(
            smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read)
        )
        self.warp_scheduler_barrier_sync()
        # S = Q @ K.T
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        # RescaleOBeforeGemm: rescale O while QK GEMM is in flight, before PV GEMM
        if const_expr(self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, scores_scale)

        # --- Issue O += P(i-1) @ V(i-1) (overlapped) ---

        pipeline_v.consumer_wait(
            smem_pipe_read_v, pipeline_v.consumer_try_wait(smem_pipe_read_v)
        )
        # O += P @ V
        mma_pv_fn(B_idx=smem_pipe_read_v.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(1)
        pipeline_k.consumer_release(smem_pipe_read)

        # --- Score mod + mask on S(i) ---

        # handle score mods and masking
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        # if cute.arch.thread_idx()[0] == 128: cute.print_tensor(layout_utils.reshape_acc_to_mn(acc_S))

        # --- Online softmax + wait PV(i-1) ---

        row_scale = softmax.online_softmax(acc_S, check_inf=check_inf)
        warpgroup.wait_group(0)
        pipeline_v.consumer_release(smem_pipe_read_v)

        # --- Convert P(i) and copy to smem ---

        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        # tOrP_cur.store(tOrP_acc.load().to(self.dtype))
        # the "to(self.dtype)" conversion fails to vectorize for block sizes other
        # than 128 x 128, i.e. it calls convert on 1 fp32 element at a time instead of
        # 2 elements. So we just call ptx directly.
        cutedsl_utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        if const_expr(not self.rescale_O_before_gemm):
            softmax.rescale_O(acc_O, row_scale)
        if const_expr(self.rescale_O_before_gemm):
            scores_scale.store(row_scale.load())
        if const_expr(not self.mma_pv_is_rs):
            # Fence and barrier to make sure smem store is visible to WGMMA
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV

        # --- Debug print ---

        if const_expr(self.debug_print):
            if is_print_thread_and_tile:
                prefix = "[fwd_sm90_mma_one_n_block_intrawg_overlap] "
                cute.printf("")
                cute.printf(prefix + "n_block={}", n_block)
                cute.printf(prefix + "acc_S.layout: {}", acc_S.layout)
                cute.printf(prefix + "row_scale.layout: {}", row_scale.layout)
                cute.printf(prefix + "acc_O.layout: {}", acc_O.layout)
                cute.printf("")

        return smem_pipe_read

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
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
            seqlen_info=seqlen,
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
        tma_atom_O: Optional[cute.CopyAtom],
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
        smem_copy_atom_O = cutedsl_utils.get_smem_store_atom(self.arch_num, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(
            tidx
        )

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

        # S2G copy sO to gO
        ragged = self.use_tma_O and (
            seqlen_info.has_cu_seqlens_q or seqlen_info.has_seqused_q
        )
        mO_cur = seqlen_info.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
            None, None, head_idx
        ]
        if const_expr(self.use_tma_O):
            # Proxy fence to ensure smem writes are visible to TMA
            cute.arch.fence_view_async_shared()

            # Notify the TMA-S2G warp that the sO store is ready for TMA copy
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )

            # S2G copy sO to gO via TMA
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:  # last warp in the producer WG
                # Wait for sO store to be ready for TMA copy
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )

                # Issue TMA store for O
                store_O()

                # Commit and wait for TMA store to be finished (at least reading sO)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        else:
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
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                )
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
                prefix = "[fwd_sm90_epilogue] "
                cute.printf("")
                cute.printf(
                    prefix + "m_block={}, head_idx={}, batch_idx={}",
                    m_block,
                    head_idx,
                    batch_idx,
                )
                cute.printf(prefix + f"use_tma_O={self.use_tma_O}, {ragged=}")
                cute.printf(prefix + "acc_O: {}", acc_O.layout)
                cute.printf(prefix + "rO: {}", rO.layout)
                cute.printf(prefix + "sO: {}", sO.layout)
                cute.printf(prefix + "cO: {}", cO.layout)
                cute.printf(prefix + "mO_cur: {}", mO_cur.layout)
                cute.printf("")

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + cutedsl_utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_wg,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = cutedsl_utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_wg,
            )
