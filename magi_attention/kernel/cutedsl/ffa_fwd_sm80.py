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

# mypy: disable-error-code="union-attr,arg-type,no-redef,attr-defined,misc,assignment"
# pyright: reportInvalidTypeForm=false

# SM80 (Ampere) forward pass for flash attention, extracted from flash_fwd.py.


from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import warp
from quack import layout_utils

from .cutedsl_utils import assume_tensor_aligned

# isort: split
from .legacy import ampere_helpers as sm80_utils
from .legacy import utils
from .legacy.block_info import BlockInfo
from .legacy.block_sparsity import BlockSparseTensors
from .legacy.flash_fwd import FlashAttentionForwardBase
from .legacy.mask import AttentionMask
from .legacy.seqlen_info import SeqlenInfoQK
from .legacy.softmax import Softmax, apply_score_mod_inner
from .legacy.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


class FFAFwdSm80(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        debug_print: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug_print = debug_print

        if self.debug_print:
            prefix = "[fwd_sm80_init] "
            print()
            print(f"{prefix}Initialized FFAFwdSm80 with: ")
            print(
                f"{prefix}{self.dtype=} | {self.tile_hdim=} | {self.tile_hdimv=} | {self.qhead_per_kvhead=}"
            )
            print(f"{prefix}{self.is_causal=} | {self.is_local=} | {self.pack_gqa=}")
            print(
                f"{prefix}{self.tile_m=} | {self.tile_n=} | {self.num_stages=} | {self.num_threads=}"
            )
            print(f"{prefix}{self.Q_in_regs=} | {self.q_subtile_factor=}")
            print(f"{prefix}{self.score_mod=} | {self.mask_mod=}")
            print(f"{prefix}{self.arch=}")
            print()

    def _get_smem_layout_atom(self):
        sQ_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdimv)
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cosize_sQV], 1024
        ]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        return (
            SharedStorageQKV
            if const_expr(not self.Q_in_regs)
            else SharedStorageSharedQV
        )

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
        aux_tensors=None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
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
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # SM80 (Ampere) has no TMA; always use the plain cp.async epilogue path,
        # even when this kernel is forced to run on a newer (sm90+) device.
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Layout permutation: 4D non-varlen vs 3D varlen
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
        if const_expr(mLSE is not None):
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
        # TileScheduler for varlen, simple grid for non-varlen
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else mQ.shape[3]
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(mQ.shape[0], self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
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
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors
        )

        # --- Debug print ---

        if const_expr(self.debug_print):
            prefix = "[fwd_sm80_call] "

            print()
            print(f"{prefix}tiled_mma_qk: {tiled_mma_qk}")
            print()
            print(f"{prefix}tiled_mma_pv: {tiled_mma_pv}")
            print()
            print(f"{prefix}sQ_layout: {self.sQ_layout}")
            print(f"{prefix}sK_layout: {self.sK_layout}")
            print(f"{prefix}sV_layout: {self.sV_layout}")
            print(f"{prefix}sO_layout: {self.sO_layout}")
            print(f"{prefix}sP_layout: {self.sP_layout}")
            print(f"{prefix}num_stages: {self.num_stages}")
            print(f"{prefix}pack_gqa: {self.pack_gqa}")
            print(f"{prefix}num_threads: {self.num_threads}")
            print()

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
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
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
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_tensors=None,
        fastdiv_mods=None,
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, _ = work_tile.tile_idx

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
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        # For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
        # seqlen_q=seqlen_k=0 and n_block_max=0.  Clamp to 0 so we don't use a
        # negative block index for K/V loads; the load/store predicates already
        # guard all memory accesses when seqlen is 0.
        n_block = cutlass.max(n_block_max - 1, 0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        num_head_kv = num_head // self.qhead_per_kvhead
        if const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, None, num_head])
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, num_head_kv, batch_size]
            mV_cur = mV[None, None, num_head_kv, batch_size]
        else:
            mK_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mK[None, None, num_head_kv]
            )
            mV_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mV[None, None, num_head_kv]
            )
        gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_Q = utils.make_tiled_copy_A(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_K = utils.make_tiled_copy_B(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_V = utils.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
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
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

        # shape: (atom_v_m * rest_m)
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        # group parameters for compute_one_n_block
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
        load_K = partial(
            self.load_K,
            gmem_tiled_copy_K,
            tKgK,
            tKsK,
            tKcK,
            t0KcK,
            tKpK,
            seqlen=seqlen.seqlen_k,
        )
        load_V = partial(
            self.load_V,
            gmem_tiled_copy_V,
            tVgV,
            tVsV,
            tVcV,
            t0VcV,
            tVpV,
            seqlen=seqlen.seqlen_k,
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

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        self.load_Q(
            gmem_thr_copy_Q,
            gQ,
            sQ,
            m_block,
            seqlen=seqlen.seqlen_q,
            headdim=mQ.shape[1],
        )
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in cutlass.range_constexpr(self.num_stages):
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= 0:
                    load_K(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= 0:
                    load_V(
                        n_block - stage,
                        smem_pipe_write=stage,
                        need_predicates=stage == 0,
                    )
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of tile_n.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
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

        # First iteration with seqlen masking
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        compute_one_n_block(
            n_block,
            smem_pipe_read,
            smem_pipe_write,
            is_first_n_block=True,
            seqlen=seqlen,
            mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
        )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # Next couple of iterations with causal masking
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = (
                block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
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
                    seqlen=seqlen,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # The remaining iterations have no masking
        for n_tile in cutlass.range(n_block, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1,
                smem_pipe_read,
                smem_pipe_write,
                seqlen=seqlen,
                is_first_n_block=False,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False),
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # TODO: local

        # normalize acc_O by row_sum and calculate the lse
        row_scale = softmax.finalize()
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # reuse sQ's data iterator
        sO = cute.make_tensor(sQ.iterator, sO_layout)
        self.epilogue(
            acc_O,
            softmax.row_sum,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            num_head,
            batch_size,
        )

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
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """

        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C(
            (self.tile_m, self.tile_n)
        )
        acc_S = cute.make_fragment(acc_shape_S, Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()

        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= 0:
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        load_V_next()
        sm80_utils.gemm(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
            # hook_fn=load_V_next,
            A_in_regs=self.Q_in_regs,
        )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )

        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= 0:
                load_K(
                    n_block - self.num_stages, smem_pipe_write, need_predicates=False
                )
            cute.arch.cp_async_commit_group()

        # wait for smem tile V for O
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            mma_params.tOrVt,
            smem_copy_params.tOsVt[
                None,
                None,
                None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ],
            smem_copy_params.smem_thr_copy_V,
            # hook_fn=load_K_next,
        )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()

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
