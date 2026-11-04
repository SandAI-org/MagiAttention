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

# pyright: reportInvalidTypeForm=false

"""SM120 forward for index-sparse (DSA) attention, ``ffa_fwd_sm120_index``.

Same contract as ``ffa_fwd_sm100_index``: ``index_sparse_indices`` has shape
``(ceil(total_q / q_block_size), num_heads_kv, topk)`` of global K token ids (-1 padded); the
kernel computes token-major ``out`` and natural-log ``lse``.

SM120 (RTX 50xx / RTX PRO 6000) has no TMEM / UMMA / wgmma: the tensor cores are fed by warp-level
``mma.sync.m16n8k16`` (BF16 x BF16 -> F32), and a CTA gets at most 99 KiB of shared memory. The
kernel is therefore a FlashAttention-2 style warp-specialized design:

* A tile is 128 Q rows of one kv head: ``q_block_size`` query tokens x the ``G`` query heads
  of the kv head (PackGQA, head fastest), so the whole tile shares one index row.
* Without ``topk_len`` every tile has the same ``n_kv`` 128-slot KV blocks, and the grid is a
  stream-K schedule: one CTA per SM, each owning an equal contiguous range of the
  ``num_tiles * n_kv`` (tile, KV block) units, so there is no wave tail. A tile split between
  CTAs is published as fp32 pieces (normalized O + log2 LSE) to a workspace; the CTA that brings
  the tile's atomic block counter to ``n_kv`` merges the pieces after its own segments. No CTA
  waits on another. With ``topk_len`` the grid is one CTA per tile.
* Warps 0-7 (MMA warps) each own 16 Q rows and run an independent online-softmax loop:
  Q lives in registers as MMA A fragments for the whole kernel, S = Q K^T is accumulated in
  registers, P is repacked in place into the A fragments of P V, and O stays in registers.
* Warp 8 (gather warp) reads the index row and gathers K and V token rows with 16 B
  ``cp.async`` (zero-filled through the ``ignore-src`` predicate for invalid slots) into a ring of
  ``kv_stages`` 32 KiB stages, K and V of one 128-slot tile in consecutive stages. Completion is
  signalled through ``cp.async.mbarrier.arrive.noinc`` so the gather warp never waits on its own
  copies; the MMA warps release a stage right after the last ``ldmatrix`` that reads it.
* Stage layout: two 64-column (128 B) chunks of 128 token rows with the 128 B XOR swizzle
  ``16 B segment ^ (row % 8)``, so the ``ldmatrix`` (K) and ``ldmatrix.trans`` (V) reads of
  eight rows hit eight distinct bank groups.
* Invalid slots (``-1``, an id outside ``[0, total_k)``, ``slot >= topk_eff``) are masked to
  ``-inf`` from four validity words per tile that the gather warp ballots and stores next to the
  stage; a tile whose words are all ones skips the mask.

Supported: BF16 / FP16, head_dim 128, ``G`` dividing 128, non-causal, SM 12.x.
"""

import math
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

import magi_attention.kernel.cutedsl as magiattn_cutedsl
from magi_attention.utils.dtype import to_cute_dtype

from .cache_utils import get_jit_cache
from .cutedsl_utils import to_cute_tensor
from .dsl_compat import SmemAllocator
from .ffa_utils import aligned_rows, maybe_contiguous

LN2 = math.log(2.0)
LOG2_E = math.log2(math.e)


# =============================================================================
# Inline PTX bridges
# =============================================================================
def _asm(ret_type, operands, text, constraints, side_effects, loc, ip):
    return llvm.inline_asm(
        ret_type,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_init(addr: Int32, count: int, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(count).ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_wait(addr: Int32, phase: Int32, *, loc=None, ip=None) -> None:
    """Spin until the phase with parity ``phase`` of the mbarrier at ``addr`` has completed."""
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n\t"
        "@P1 bra DONE;\n\t"
        "bra LAB_WAIT;\n\t"
        "DONE:\n\t}",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def mbar_arrive(addr: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def cp_async_mbar_arrive_noinc(addr: Int32, *, loc=None, ip=None) -> None:
    """Arrive on the mbarrier once all prior cp.async of this thread have landed."""
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None) -> None:
    _asm(None, [], "cp.async.wait_all;", "", True, loc, ip)


@dsl_user_op
def cp_async_16_zfill(
    dst_smem_addr: Int32,
    src_gmem_addr: Int64,
    row_id: Int32,
    dst_imm_offset: int = 0,
    *,
    loc=None,
    ip=None,
) -> None:
    """16 B global -> shared copy; writes zeros without reading the source when ``row_id < 0``."""
    assert isinstance(dst_imm_offset, int) and 0 <= dst_imm_offset < (1 << 24)
    _asm(
        None,
        [
            Int32(dst_smem_addr).ir_value(loc=loc, ip=ip),
            Int64(src_gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(row_id).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.lt.s32 p, $2, 0;\n\t"
        f"cp.async.cg.shared.global [$0+{dst_imm_offset}], [$1], 16, p;\n}}",
        "r,l,r",
        True,
        loc,
        ip,
    )


def _struct4(res, ty):
    return tuple(llvm.extractvalue(ty, res, [i]) for i in range(4))


@dsl_user_op
def ldsm_x4(addr: Int32, imm: int = 0, *, loc=None, ip=None) -> Tuple[Int32, ...]:
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0, $1, $2, $3}}, [$4+{imm}];",
        "=r,=r,=r,=r,r",
        False,
        loc,
        ip,
    )
    return tuple(Int32(v) for v in _struct4(res, T.i32()))


@dsl_user_op
def ldsm_x4_trans(addr: Int32, imm: int = 0, *, loc=None, ip=None) -> Tuple[Int32, ...]:
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        f"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {{$0, $1, $2, $3}}, [$4+{imm}];",
        "=r,=r,=r,=r,r",
        False,
        loc,
        ip,
    )
    return tuple(Int32(v) for v in _struct4(res, T.i32()))


@dsl_user_op
def mma_16816(
    ab_type: str,
    a: Tuple[Int32, Int32, Int32, Int32],
    b0: Int32,
    b1: Int32,
    c: Tuple[Float32, Float32, Float32, Float32],
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, ...]:
    """``mma.sync.aligned.m16n8k16.row.col.f32.{ab}.{ab}.f32``."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (*a, b0, b1)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in c],
        f"mma.sync.aligned.m16n8k16.row.col.f32.{ab_type}.{ab_type}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        False,
        loc,
        ip,
    )
    return tuple(Float32(v) for v in _struct4(res, T.f32()))


@dsl_user_op
def cvt_pack(ab_type: str, lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Two f32 -> one packed 16-bit pair, ``lo`` in the low half."""
    return Int32(
        _asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            f"cvt.rn.{ab_type}x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_b32_pred(addr: Int64, pred: Boolean, *, loc=None, ip=None) -> Int32:
    """32-bit read-only global load, zero when ``pred`` is false (the address is then not read)."""
    return Int32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(pred).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@p ld.global.nc.b32 $0, [$1];\n\t}",
            "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def st_global_b32_pred(
    addr: Int64, value: Int32, pred: Boolean, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}",
        "l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_f32_pred(
    addr: Int64, value: Float32, pred: Boolean, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\t@p st.global.f32 [$0], $1;\n\t}",
        "l,f,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_shared_v4(addr: Int32, *, loc=None, ip=None) -> Tuple[Int32, ...]:
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return tuple(Int32(v) for v in _struct4(res, T.i32()))


@dsl_user_op
def st_shared_v4(
    addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (addr, w0, w1, w2, w3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def shfl_bfly_f32(x: Float32, offset: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def lg2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "lg2.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def opaque_i32(value: Int32, *, loc=None, ip=None) -> Int32:
    """Return ``value`` through a ``mov.b32`` the optimizer cannot see through.

    Used to recompute the epilogue's row pointers instead of letting CSE keep the prologue's copy
    live in registers across the whole KV loop.
    """
    return Int32(
        _asm(
            T.i32(),
            [Int32(value).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None) -> None:
    _asm(None, [], "fence.acq_rel.gpu;", "", True, loc, ip)


@dsl_user_op
def atom_add_gpu(addr: Int64, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.add.relaxed.gpu.global.s32 $0, [$1], $2;",
            "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def st_global_v4_f32(addr: Int64, v0, v1, v2, v3, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (v0, v1, v2, v3)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_v2_f32(addr: Int64, v0, v1, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (v0, v1)],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_cg_v4_f32(addr: Int64, *, loc=None, ip=None) -> Tuple[Float32, ...]:
    """L2-only load: the data was written by another CTA during this kernel."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        True,
        loc,
        ip,
    )
    return tuple(Float32(v) for v in _struct4(res, T.f32()))


@dsl_user_op
def ld_global_cg_v2_f32(addr: Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        True,
        loc,
        ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [i])) for i in range(2))


@dsl_user_op
def st_shared_b32(addr: Int32, value: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_shared_b32(addr: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


# =============================================================================
# Kernel
# =============================================================================
class FFAFwdSm120Index:
    head_dim = 128
    tile_m = 128
    tile_n = 128
    # topk must be a multiple of this, as for the SM100 forward: index rows start 16 B aligned.
    topk_align = 4
    # Each MMA warp owns 16 of the 128 tile rows.
    num_mma_warps = 8
    # 3 x 32 KiB stages = 96 KiB, the most the 99 KiB opt-in shared memory holds.
    kv_stages = 3
    # Stream-K partial piece: 128 x 128 fp32 O in MMA fragment order, plus 128 x 2 fp32 rows of
    # log2-domain LSE (each thread stores its two rows).
    piece_o_bytes = 128 * 128 * 4
    piece_lse_bytes = 256 * 2 * 4

    def __init__(
        self,
        qhead_per_kvhead: int,
        ids_all_valid: bool = False,
        stream_k: bool = True,
        two_pieces: bool = False,
    ):
        assert self.tile_m % qhead_per_kvhead == 0
        self.G = qhead_per_kvhead
        self.q_block_size = self.tile_m // qhead_per_kvhead
        self.ids_all_valid = ids_all_valid
        # Stream-K needs the same number of 128-slot KV blocks in every tile, i.e. no topk_len.
        self.stream_k = stream_k
        # Every split tile is split across exactly two CTAs (each CTA owns >= n_kv - 1 units),
        # so the merge is straight-line code.
        self.two_pieces = two_pieces
        self.load_warp_id = self.num_mma_warps
        # The register file is split per SM sub-partition (16K regs each, warps assigned round
        # robin), so a 9th warp would cap every thread at 168 registers. Instead launch three
        # full warpgroups and rebalance with setmaxnreg. setmaxnreg only moves registers inside
        # the CTA's launch allocation (384 x 168), so 256 x num_regs_mma + 128 x num_regs_load
        # must not exceed it, or the increase blocks forever. Warpgroup 2 holds the gather warp
        # and three idle warps (setmaxnreg is warpgroup-wide).
        self.threads_per_cta = 32 * (self.num_mma_warps + 4)
        self.num_mma_threads = 32 * self.num_mma_warps
        self.num_regs_mma = 232
        self.num_regs_load = 32
        assert 256 * self.num_regs_mma + 128 * self.num_regs_load <= 384 * 168
        # Bytes of one K or V stage (128 rows x 128 x 16 bit) and of one 64-column chunk of it.
        self.stage_bytes = self.tile_n * self.head_dim * 2
        self.chunk_bytes = self.tile_n * 128
        # Full barrier: 32 noinc cp.async arrivals + 1 plain arrival publishing the mask words.
        self.full_arrivals = 33
        self.empty_arrivals = self.num_mma_warps
        # Named barrier 0 is __syncthreads.
        self.mma_barrier_id = 1

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (s_q, h_q, d)
        mK: cute.Tensor,  # (s_k, h_kv, d)
        mV: cute.Tensor,  # (s_k, h_kv, d)
        mO: cute.Tensor,  # (s_q, h_q, d)
        mLSE: cute.Tensor,  # (s_q, h_q) fp32
        softmax_scale: Float32,
        mIndices: cute.Tensor,  # (num_q_blocks, h_kv, topk) int32
        mTopkLen: Optional[cute.Tensor],  # (num_q_blocks, h_kv) int32 or None
        mWorkspace: Optional[cute.Tensor],  # stream-K pieces, uint8
        mCounter: Optional[cute.Tensor],  # (num_q_blocks * h_kv,) int32, kept zeroed
        nheads_kv: Int32,
        num_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.dtype = mQ.element_type
        self.ab_type = "bf16" if const_expr(self.dtype == cutlass.BFloat16) else "f16"

        @cute.struct
        class SharedStorage:
            full_bar: cute.struct.MemRange[Int64, self.kv_stages]
            empty_bar: cute.struct.MemRange[Int64, self.kv_stages]
            mask: cute.struct.Align[
                cute.struct.MemRange[Uint32, self.kv_stages * 4], 16
            ]
            flag: cute.struct.MemRange[Int32, 4]
            kv: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype, self.kv_stages * self.tile_n * self.head_dim
                ],
                1024,
            ]

        self.shared_storage = SharedStorage
        softmax_scale_log2 = softmax_scale * LOG2_E
        if const_expr(self.stream_k):
            grid = [num_ctas, 1, 1]
        else:
            grid = [cute.size(mIndices.shape[0]), nheads_kv, 1]
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mIndices,
            mTopkLen,
            mWorkspace,
            mCounter,
            softmax_scale_log2,
            softmax_scale,
            nheads_kv,
            num_ctas,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            smem=SharedStorage.size_in_bytes(),
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
        mLSE: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        mCounter: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        nheads_kv: Int32,
        num_ctas: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        bars = (
            Int32(storage.full_bar.data_ptr().toint()),
            Int32(storage.empty_bar.data_ptr().toint()),
            Int32(storage.mask.data_ptr().toint()),
            Int32(storage.kv.data_ptr().toint()),
            Int32(storage.flag.data_ptr().toint()),
        )

        if warp_idx == 0 and lane == 0:
            for s in cutlass.range_constexpr(self.kv_stages):
                mbar_init(bars[0] + 8 * s, self.full_arrivals)
                mbar_init(bars[1] + 8 * s, self.empty_arrivals)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # Each role runs its own segment loop right after its setmaxnreg: ptxas budgets registers
        # per region dominated by the setmaxnreg, so code shared by both warpgroups (one loop with
        # the roles branching inside) would be held to the 168-register launch allocation.
        if warp_idx >= self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx == self.load_warp_id:
                self.role_loop(
                    True,
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mIndices,
                    mTopkLen,
                    mWorkspace,
                    mCounter,
                    softmax_scale_log2,
                    softmax_scale,
                    nheads_kv,
                    num_ctas,
                    bidx,
                    bidy,
                    warp_idx,
                    lane,
                    bars,
                )
                cp_async_wait_all()
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            self.role_loop(
                False,
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                mIndices,
                mTopkLen,
                mWorkspace,
                mCounter,
                softmax_scale_log2,
                softmax_scale,
                nheads_kv,
                num_ctas,
                bidx,
                bidy,
                warp_idx,
                lane,
                bars,
            )

    @cute.jit
    def role_loop(
        self,
        is_load: cutlass.Constexpr[bool],
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mIndices: cute.Tensor,
        mTopkLen: Optional[cute.Tensor],
        mWorkspace: Optional[cute.Tensor],
        mCounter: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        nheads_kv: Int32,
        num_ctas: Int32,
        bidx: Int32,
        bidy: Int32,
        warp_idx: Int32,
        lane: Int32,
        bars,
    ):
        """Walk this CTA's segments (one tile's KV-block range each) as the gather or MMA role."""
        topk = Int32(mIndices.shape[2])
        n_kv = (topk + (self.tile_n - 1)) // self.tile_n
        num_m_blocks = Int32(mIndices.shape[0])
        frags = None
        if const_expr(not is_load):
            frags = (
                cute.make_rmem_tensor((32,), Int32),  # Q A-fragments
                cute.make_rmem_tensor(
                    (64,), Float32
                ),  # S / P (also the stream-K merge accumulator)
                cute.make_rmem_tensor((64,), Float32),  # O
            )
        stage = Int32(0)
        phase = Int32(0)
        # Split tiles of this CTA's first / last segment that it has to merge (-1: none).
        pend0 = Int32(-1)
        pend1 = Int32(-1)
        if const_expr(self.stream_k):
            # CTA c owns work units [c * U / P, (c + 1) * U / P), unit = (tile, 128-slot KV block).
            # The wrapper guarantees U < 2^31; see unit_start for why everything stays 32-bit.
            total_units = num_m_blocks * nheads_kv * n_kv
            u = self.unit_start(bidx, total_units, num_ctas)
            u_start = u
            u_end = self.unit_start(bidx + 1, total_units, num_ctas)
            while u < u_end:
                tile = u // n_kv
                kv0 = u - tile * n_kv
                kv1 = cutlass.min(n_kv, kv0 + (u_end - u))
                m_block = tile // nheads_kv
                head_kv = tile - m_block * nheads_kv
                if const_expr(is_load):
                    stage, phase = self.gather_segment(
                        mK,
                        mV,
                        mIndices,
                        m_block,
                        head_kv,
                        topk,
                        topk,
                        kv0,
                        kv1,
                        nheads_kv,
                        lane,
                        bars,
                        stage,
                        phase,
                    )
                else:
                    is_first = u == u_start
                    piece = Int32(bidx) * 2 + (0 if is_first else 1)
                    stage, phase, pending = self.mma_segment(
                        mQ,
                        mO,
                        mLSE,
                        mWorkspace,
                        mCounter,
                        m_block,
                        head_kv,
                        tile,
                        kv0,
                        kv1,
                        n_kv,
                        piece,
                        total_units,
                        num_ctas,
                        warp_idx,
                        lane,
                        bars,
                        frags,
                        stage,
                        phase,
                        softmax_scale_log2,
                        softmax_scale,
                    )
                    if is_first:
                        pend0 = pending
                    else:
                        pend1 = pending
                u = u + (kv1 - kv0)
            if const_expr(not is_load):
                # Merging after the segment loop keeps the merge's registers out of the hot loop.
                for pend in (pend0, pend1):
                    if pend >= 0:
                        self.stream_k_combine(
                            mQ,
                            mO,
                            mLSE,
                            mWorkspace,
                            frags[1],
                            pend,
                            n_kv,
                            nheads_kv,
                            total_units,
                            num_ctas,
                            warp_idx,
                            lane,
                            softmax_scale_log2,
                        )
        else:
            m_block = bidx
            head_kv = bidy
            topk_eff = topk
            if const_expr(mTopkLen is not None):
                topk_eff = cutlass.min(
                    cutlass.max(mTopkLen[m_block, head_kv], Int32(0)), topk
                )
            n_kv_eff = (topk_eff + (self.tile_n - 1)) // self.tile_n
            if const_expr(is_load):
                stage, phase = self.gather_segment(
                    mK,
                    mV,
                    mIndices,
                    m_block,
                    head_kv,
                    topk,
                    topk_eff,
                    Int32(0),
                    n_kv_eff,
                    nheads_kv,
                    lane,
                    bars,
                    stage,
                    phase,
                )
            else:
                stage, phase, _ = self.mma_segment(
                    mQ,
                    mO,
                    mLSE,
                    None,
                    None,
                    m_block,
                    head_kv,
                    Int32(0),
                    Int32(0),
                    n_kv_eff,
                    n_kv_eff,
                    Int32(0),
                    Int32(0),
                    num_ctas,
                    warp_idx,
                    lane,
                    bars,
                    frags,
                    stage,
                    phase,
                    softmax_scale_log2,
                    softmax_scale,
                )

    # -------------------------------------------------------------------------
    # Gather warp
    # -------------------------------------------------------------------------
    @cute.jit
    def load_tile_toks(self, mIndices, m_block, head_kv, tile, topk, lane):
        """Raw ids of one 128-slot tile: lane ``l`` gets slots ``32 r + l`` (clamped in bounds)."""
        base = tile * self.tile_n + lane
        last = topk - 1
        return tuple(
            mIndices[m_block, head_kv, cutlass.min(base + 32 * r, last)]
            for r in range(4)
        )

    @cute.jit
    def gather_stage(self, src_base: Int64, dst_base: Int32, rows, dst_rel, lane):
        """Copy the 128 rows of one K or V tile, two rows per warp instruction.

        Lanes 0-15 copy 16 B segment ``lane`` of row ``2 i``, lanes 16-31 of row ``2 i + 1``. The
        destination is chunk ``seg / 8`` of the row with segment ``(seg % 8) ^ (row % 8)``;
        ``(row % 8)`` only depends on ``i % 4`` and the lane's half, so it is folded into the four
        per-lane ``dst_rel`` bases and ``row * 128`` becomes the instruction's immediate.
        """
        half = lane >> 4
        for i in cutlass.range_constexpr(self.tile_n // 2):
            word = rows[(2 * i) // 32]
            row_id = cute.arch.shuffle_sync(word, offset=half + ((2 * i) % 32))
            src = src_base + (Int64(Uint32(row_id)) << 8)
            cp_async_16_zfill(dst_base + dst_rel[i % 4], src, row_id, 256 * i)

    @cute.jit
    def advance(self, stage: Int32, phase: Int32):
        stage = stage + 1
        if stage == self.kv_stages:
            stage = Int32(0)
            phase = phase ^ 1
        return stage, phase

    @cute.jit
    def gather_segment(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mIndices: cute.Tensor,
        m_block: Int32,
        head_kv: Int32,
        topk: Int32,
        topk_eff: Int32,
        kv0: Int32,
        kv1: Int32,
        nheads_kv: Int32,
        lane: Int32,
        bars,
        stage: Int32,
        phase: Int32,
    ):
        """Gather K/V of KV blocks ``[kv0, kv1)`` of one index row into the stage ring."""
        full_addr, empty_addr, mask_addr, kv_addr, _ = bars
        seqlen_k = Int32(mK.shape[0])
        seg = lane & 15
        half = lane >> 4
        k_base = Int64(mK.iterator.toint()) + Int64(seg * 16)
        v_base = Int64(mV.iterator.toint()) + Int64(seg * 16)
        dst_rel = tuple(
            (seg >> 3) * self.chunk_bytes
            + half * 128
            + (((seg & 7) ^ (2 * j + half)) << 4)
            for j in range(4)
        )
        t0, t1, t2, t3 = self.load_tile_toks(
            mIndices, m_block, head_kv, kv0, topk, lane
        )
        for j in cutlass.range(kv0, kv1, unroll=1):
            toks = (t0, t1, t2, t3)
            rows = []
            words = []
            for r in cutlass.range_constexpr(4):
                tok = toks[r]
                if const_expr(self.ids_all_valid):
                    rows.append(tok * nheads_kv + head_kv)
                else:
                    slot = j * self.tile_n + 32 * r + lane
                    valid = slot < topk_eff and tok >= 0 and tok < seqlen_k
                    rows.append(tok * nheads_kv + head_kv if valid else Int32(-1))
                    words.append(Int32(cute.arch.vote_ballot_sync(valid)))
            # Prefetch the next tile's ids before anything that can stall.
            nxt = cutlass.min(j + 1, kv1 - 1)
            t0, t1, t2, t3 = self.load_tile_toks(
                mIndices, m_block, head_kv, nxt, topk, lane
            )

            # K stage (also carries the validity words).
            mbar_wait(empty_addr + stage * 8, phase ^ 1)
            if const_expr(not self.ids_all_valid):
                if lane == 0:
                    st_shared_v4(mask_addr + stage * 16, *words)
            self.gather_stage(
                k_base, kv_addr + stage * self.stage_bytes, rows, dst_rel, lane
            )
            cp_async_mbar_arrive_noinc(full_addr + stage * 8)
            if lane == 0:
                mbar_arrive(full_addr + stage * 8)
            stage, phase = self.advance(stage, phase)

            # V stage.
            mbar_wait(empty_addr + stage * 8, phase ^ 1)
            self.gather_stage(
                v_base, kv_addr + stage * self.stage_bytes, rows, dst_rel, lane
            )
            cp_async_mbar_arrive_noinc(full_addr + stage * 8)
            if lane == 0:
                mbar_arrive(full_addr + stage * 8)
            stage, phase = self.advance(stage, phase)
        return stage, phase

    # -------------------------------------------------------------------------
    # MMA warps
    # -------------------------------------------------------------------------
    @cute.jit
    def row_coords(self, m_block, head_kv, row, seqlen_q, nheads_q):
        """(element offset of the row in Q/O, element offset in LSE, row in bounds)."""
        token = m_block * self.q_block_size + row // self.G
        head = head_kv * self.G + row % self.G
        lse_off = Int64(token) * Int64(nheads_q) + Int64(head)
        return lse_off * self.head_dim, lse_off, token < seqlen_q

    @cute.jit
    def unit_start(self, c: Int32, total_units: Int32, num_ctas: Int32) -> Int32:
        """First work unit of CTA ``c``: ``floor(c * U / P)``, exactly, in 32-bit arithmetic.

        NOTE: ``c * U`` does not fit 32 bits, and a 64-bit integer division compiles to a CALL
        whose register save/restore spills the MMA warps' fragments, so split U = q P + r:
        ``floor(c U / P) = c q + floor(c r / P)`` with ``c r < P^2``.
        """
        q = total_units // num_ctas
        r = total_units - q * num_ctas
        return c * q + (c * r) // num_ctas

    @cute.jit
    def owner_cta(self, unit: Int32, total_units: Int32, num_ctas: Int32) -> Int32:
        """The CTA whose unit range ``[unit_start(c), unit_start(c + 1))`` contains ``unit``.

        An fp32 estimate is off by at most one either way; two exact corrections fix it.
        NOTE: rcp, not an IEEE division: the division's slow path is a CALL, and a kernel with a
        CALL loses setmaxnreg's per-role register budgets (the MMA warps then spill).
        """
        c = Int32(
            Float32(unit)
            * Float32(num_ctas)
            * cute.arch.rcp_approx(Float32(total_units))
        )
        c = cutlass.max(cutlass.min(c, num_ctas - 1), Int32(0))
        if self.unit_start(c, total_units, num_ctas) > unit:
            c = c - 1
        if c + 1 < num_ctas:
            if self.unit_start(c + 1, total_units, num_ctas) <= unit:
                c = c + 1
        return c

    @cute.jit
    def mma_segment(
        self,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mWorkspace: Optional[cute.Tensor],
        mCounter: Optional[cute.Tensor],
        m_block: Int32,
        head_kv: Int32,
        tile: Int32,
        kv0: Int32,
        kv1: Int32,
        n_kv: Int32,
        piece: Int32,
        total_units: Int32,
        num_ctas: Int32,
        warp_idx: Int32,
        lane: Int32,
        bars,
        frags,
        stage: Int32,
        phase: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
    ):
        """Attend one tile's Q rows to KV blocks ``[kv0, kv1)``; store O/LSE or a stream-K piece."""
        full_addr, empty_addr, mask_addr, kv_addr, flag_addr = bars
        seqlen_q = Int32(mQ.shape[0])
        nheads_q = Int32(mQ.shape[1])
        g = lane >> 2
        t = lane & 3
        row0 = warp_idx * 16 + g
        q_off0, lse_off0, ok0 = self.row_coords(
            m_block, head_kv, row0, seqlen_q, nheads_q
        )
        q_off1, lse_off1, ok1 = self.row_coords(
            m_block, head_kv, row0 + 8, seqlen_q, nheads_q
        )
        col_bytes = Int64(t * 4)
        q_ptr0 = Int64(mQ.iterator.toint()) + q_off0 * 2 + col_bytes
        q_ptr1 = Int64(mQ.iterator.toint()) + q_off1 * 2 + col_bytes

        # Q as m16n8k16 A fragments: k-step kk holds (row g, k 2t), (g+8, 2t), (g, 2t+8), (g+8, 2t+8).
        tQ, tS, tO = frags
        for kk in cutlass.range_constexpr(8):
            tQ[kk * 4 + 0] = ld_global_b32_pred(q_ptr0 + kk * 32, ok0)
            tQ[kk * 4 + 1] = ld_global_b32_pred(q_ptr1 + kk * 32, ok1)
            tQ[kk * 4 + 2] = ld_global_b32_pred(q_ptr0 + kk * 32 + 16, ok0)
            tQ[kk * 4 + 3] = ld_global_b32_pred(q_ptr1 + kk * 32 + 16, ok1)

        tO.fill(0.0)

        # ldmatrix lane addresses (relative to the stage base; see the module docstring for the
        # swizzle). K, non-transposed x4 for n-frag pair p / k-step kk: lane l reads key
        # p*16 + (l/16)*8 + l%8, columns kk*16 + ((l/8)%2)*8.
        sw = lane & 7
        k_row = (((lane >> 4) << 3) + sw) * 128
        k_off = tuple(
            k_row + ((((c * 2) + ((lane >> 3) & 1)) ^ sw) << 4) for c in range(4)
        )
        # V, transposed x4 for key step kk / d-pair dp: lane l reads key kk*16 + ((l/8)%2)*8 + l%8,
        # columns dp*16 + (l/16)*8.
        v_row = ((((lane >> 3) & 1) << 3) + sw) * 128
        v_off = tuple(v_row + ((((c * 2) + (lane >> 4)) ^ sw) << 4) for c in range(4))

        neg_inf = Float32(-math.inf)
        m0 = neg_inf
        m1 = neg_inf
        l0 = Float32(0.0)
        l1 = Float32(0.0)
        for _j in cutlass.range(kv0, kv1, unroll=1):
            # ---------------- S = Q K^T ----------------
            mbar_wait(full_addr + stage * 8, phase)
            words = (Int32(-1), Int32(-1), Int32(-1), Int32(-1))
            if const_expr(not self.ids_all_valid):
                words = ld_shared_v4(mask_addr + stage * 16)
            sbase = kv_addr + stage * self.stage_bytes
            kaddr = tuple(sbase + k_off[c] for c in range(4))
            for kk in cutlass.range_constexpr(8):
                a = (tQ[kk * 4], tQ[kk * 4 + 1], tQ[kk * 4 + 2], tQ[kk * 4 + 3])
                for p in cutlass.range_constexpr(8):
                    b = ldsm_x4(kaddr[kk % 4], p * 2048 + (kk // 4) * self.chunk_bytes)
                    for h in cutlass.range_constexpr(2):
                        nf = 2 * p + h
                        if const_expr(kk == 0):
                            c = (Float32(0.0),) * 4
                        else:
                            c = (
                                tS[nf * 4],
                                tS[nf * 4 + 1],
                                tS[nf * 4 + 2],
                                tS[nf * 4 + 3],
                            )
                        d = mma_16816(self.ab_type, a, b[2 * h], b[2 * h + 1], c)
                        for e in cutlass.range_constexpr(4):
                            tS[nf * 4 + e] = d[e]
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty_addr + stage * 8)
            stage, phase = self.advance(stage, phase)

            # ---------------- mask + online softmax ----------------
            if const_expr(not self.ids_all_valid):
                self.apply_mask(tS, words, t)
            m0, l0 = self.softmax_rows(tS, tO, 0, m0, l0, softmax_scale_log2)
            m1, l1 = self.softmax_rows(tS, tO, 2, m1, l1, softmax_scale_log2)

            # ---------------- O += P V ----------------
            mbar_wait(full_addr + stage * 8, phase)
            sbase = kv_addr + stage * self.stage_bytes
            vaddr = tuple(sbase + v_off[c] for c in range(4))
            for kk in cutlass.range_constexpr(8):
                n0 = 2 * kk
                n1 = 2 * kk + 1
                a = (
                    cvt_pack(self.ab_type, tS[n0 * 4 + 0], tS[n0 * 4 + 1]),
                    cvt_pack(self.ab_type, tS[n0 * 4 + 2], tS[n0 * 4 + 3]),
                    cvt_pack(self.ab_type, tS[n1 * 4 + 0], tS[n1 * 4 + 1]),
                    cvt_pack(self.ab_type, tS[n1 * 4 + 2], tS[n1 * 4 + 3]),
                )
                for dp in cutlass.range_constexpr(8):
                    b = ldsm_x4_trans(
                        vaddr[dp % 4], kk * 2048 + (dp // 4) * self.chunk_bytes
                    )
                    for h in cutlass.range_constexpr(2):
                        df = 2 * dp + h
                        c = (tO[df * 4], tO[df * 4 + 1], tO[df * 4 + 2], tO[df * 4 + 3])
                        d = mma_16816(self.ab_type, a, b[2 * h], b[2 * h + 1], c)
                        for e in cutlass.range_constexpr(4):
                            tO[df * 4 + e] = d[e]
            cute.arch.sync_warp()
            if lane == 0:
                mbar_arrive(empty_addr + stage * 8)
            stage, phase = self.advance(stage, phase)

        # ---------------- epilogue ----------------
        m_block = opaque_i32(m_block)
        q_off0, lse_off0, ok0 = self.row_coords(
            m_block, head_kv, row0, seqlen_q, nheads_q
        )
        q_off1, lse_off1, ok1 = self.row_coords(
            m_block, head_kv, row0 + 8, seqlen_q, nheads_q
        )
        col_bytes = Int64(t * 4)
        o_ptr0 = Int64(mO.iterator.toint()) + q_off0 * 2 + col_bytes
        o_ptr1 = Int64(mO.iterator.toint()) + q_off1 * 2 + col_bytes
        lse_ptr0 = Int64(mLSE.iterator.toint()) + lse_off0 * 4
        lse_ptr1 = Int64(mLSE.iterator.toint()) + lse_off1 * 4
        inv0, lse2_0 = self.finalize_row(l0, m0, softmax_scale_log2)
        inv1, lse2_1 = self.finalize_row(l1, m1, softmax_scale_log2)
        # Split tile this CTA arrived last on and must merge, or -1.
        pending = Int32(-1)
        if const_expr(not self.stream_k):
            self.store_row(tO, 0, inv0, lse2_0 * LN2, o_ptr0, lse_ptr0, ok0, t)
            self.store_row(tO, 2, inv1, lse2_1 * LN2, o_ptr1, lse_ptr1, ok1, t)
        else:
            # The segment covers the whole tile iff it spans all n_kv KV blocks.
            if kv0 == 0 and kv1 == n_kv:
                self.store_row(tO, 0, inv0, lse2_0 * LN2, o_ptr0, lse_ptr0, ok0, t)
                self.store_row(tO, 2, inv1, lse2_1 * LN2, o_ptr1, lse_ptr1, ok1, t)
            else:
                # Pieces add their KV-block counts; the one that completes n_kv arrived last.
                done = self.stream_k_publish(
                    tO,
                    inv0,
                    inv1,
                    lse2_0,
                    lse2_1,
                    mWorkspace,
                    mCounter,
                    tile,
                    piece,
                    num_ctas,
                    warp_idx * 32 + lane,
                    flag_addr,
                    kv1 - kv0,
                    n_kv,
                )
                if done + (kv1 - kv0) == n_kv:
                    pending = tile
        return stage, phase, pending

    @cute.jit
    def stream_k_publish(
        self,
        tO: cute.Tensor,
        inv0: Float32,
        inv1: Float32,
        lse2_0: Float32,
        lse2_1: Float32,
        mWorkspace: cute.Tensor,
        mCounter: cute.Tensor,
        tile: Int32,
        piece: Int32,
        num_ctas: Int32,
        tid: Int32,
        flag_addr: Int32,
        n_blocks: Int32,
        n_kv: Int32,
    ) -> Int32:
        """Publish this CTA's piece of a split tile (``n_blocks`` KV blocks of it).

        Returns the tile's KV-block count published before this piece.

        A piece is the normalized partial O (fp32, MMA fragment order, 16 B per thread per vector
        so every store is coalesced) and the log2-domain LSE of the thread's two rows. Every
        thread fences its stores before the barrier, then one thread adds ``n_blocks`` to the tile
        counter; the CTA whose addition completes ``n_kv`` merges the tile
        (:meth:`stream_k_combine`) and resets the counter to 0, so the counter buffer stays
        zeroed between launches. No CTA ever waits on another.
        """
        ws_base = Int64(mWorkspace.iterator.toint())
        lse_region = ws_base + Int64(num_ctas) * 2 * self.piece_o_bytes
        po = ws_base + Int64(piece) * self.piece_o_bytes + Int64(tid * 16)
        for i in cutlass.range_constexpr(16):
            st_global_v4_f32(
                po + i * (self.num_mma_threads * 16),
                tO[4 * i + 0] * inv0,
                tO[4 * i + 1] * inv0,
                tO[4 * i + 2] * inv1,
                tO[4 * i + 3] * inv1,
            )
        st_global_v2_f32(
            lse_region + Int64(piece) * self.piece_lse_bytes + Int64(tid * 8),
            lse2_0,
            lse2_1,
        )
        fence_acq_rel_gpu()
        cute.arch.barrier(
            barrier_id=self.mma_barrier_id, number_of_threads=self.num_mma_threads
        )
        if tid == 0:
            counter_ptr = Int64(mCounter.iterator.toint()) + Int64(tile) * 4
            before = atom_add_gpu(counter_ptr, n_blocks)
            st_shared_b32(flag_addr, before)
            st_global_b32_pred(counter_ptr, Int32(0), before + n_blocks == n_kv)
        cute.arch.barrier(
            barrier_id=self.mma_barrier_id, number_of_threads=self.num_mma_threads
        )
        arrived = ld_shared_b32(flag_addr)
        # Keep the flag stable until every thread has read it (the next segment may rewrite it).
        cute.arch.barrier(
            barrier_id=self.mma_barrier_id, number_of_threads=self.num_mma_threads
        )
        return arrived

    @cute.jit
    def stream_k_combine(
        self,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mWorkspace: cute.Tensor,
        acc: cute.Tensor,
        tile: Int32,
        n_kv: Int32,
        nheads_kv: Int32,
        total_units: Int32,
        num_ctas: Int32,
        warp_idx: Int32,
        lane: Int32,
        softmax_scale_log2: Float32,
    ):
        """Merge every published piece of ``tile`` with weights ``2^(lse_i - lse)`` and store it.

        The tile's pieces come from CTAs ``c_first .. c_last``: every CTA after ``c_first`` starts
        inside the tile (piece ``2 c``), ``c_first`` holds piece ``2 c + 1`` unless it starts at
        the tile's first unit.
        """
        fence_acq_rel_gpu()
        tid = warp_idx * 32 + lane
        t = lane & 3
        seqlen_q = Int32(mQ.shape[0])
        nheads_q = Int32(mQ.shape[1])
        m_block = tile // nheads_kv
        head_kv = tile - m_block * nheads_kv
        row0 = warp_idx * 16 + (lane >> 2)
        q_off0, lse_off0, ok0 = self.row_coords(
            m_block, head_kv, row0, seqlen_q, nheads_q
        )
        q_off1, lse_off1, ok1 = self.row_coords(
            m_block, head_kv, row0 + 8, seqlen_q, nheads_q
        )
        first_unit = tile * n_kv
        c_first = self.owner_cta(first_unit, total_units, num_ctas)
        p_first = c_first * 2 + (
            1 if self.unit_start(c_first, total_units, num_ctas) < first_unit else 0
        )
        ws_base = Int64(mWorkspace.iterator.toint())
        lse_region = ws_base + Int64(num_ctas) * 2 * self.piece_o_bytes
        neg_inf = Float32(-math.inf)
        acc.fill(0.0)
        if const_expr(self.two_pieces):
            # NOTE: straight-line code. A runtime loop over the pieces here makes ptxas spill the
            # O accumulator inside the main KV loop (48 of its 64 registers).
            pieces = (p_first, c_first * 2 + 2)
            lses = [
                ld_global_cg_v2_f32(
                    lse_region + Int64(p) * self.piece_lse_bytes + Int64(tid * 8)
                )
                for p in pieces
            ]
            mx0 = fmax_f32(lses[0][0], lses[1][0])
            mx1 = fmax_f32(lses[0][1], lses[1][1])
            mu0 = Float32(0.0) if mx0 == neg_inf else mx0
            mu1 = Float32(0.0) if mx1 == neg_inf else mx1
            s0 = Float32(0.0)
            s1 = Float32(0.0)
            for j in cutlass.range_constexpr(2):
                w0 = cute.arch.exp2(lses[j][0] - mu0)
                w1 = cute.arch.exp2(lses[j][1] - mu1)
                s0 = s0 + w0
                s1 = s1 + w1
                self.accumulate_piece(acc, ws_base, pieces[j], tid, w0, w1)
        else:
            c_last = self.owner_cta(first_unit + n_kv - 1, total_units, num_ctas)
            mx0 = neg_inf
            mx1 = neg_inf
            for c in cutlass.range(c_first, c_last + 1, unroll=1):
                p = p_first if c == c_first else c * 2
                a, b = ld_global_cg_v2_f32(
                    lse_region + Int64(p) * self.piece_lse_bytes + Int64(tid * 8)
                )
                mx0 = fmax_f32(mx0, a)
                mx1 = fmax_f32(mx1, b)
            mu0 = Float32(0.0) if mx0 == neg_inf else mx0
            mu1 = Float32(0.0) if mx1 == neg_inf else mx1
            s0 = Float32(0.0)
            s1 = Float32(0.0)
            for c in cutlass.range(c_first, c_last + 1, unroll=1):
                p = p_first if c == c_first else c * 2
                a, b = ld_global_cg_v2_f32(
                    lse_region + Int64(p) * self.piece_lse_bytes + Int64(tid * 8)
                )
                wa = cute.arch.exp2(a - mu0)
                wb = cute.arch.exp2(b - mu1)
                s0 = s0 + wa
                s1 = s1 + wb
                self.accumulate_piece(acc, ws_base, p, tid, wa, wb)
        has0 = s0 > Float32(0.0)
        has1 = s1 > Float32(0.0)
        r0 = cute.arch.rcp_approx(s0) if has0 else Float32(0.0)
        r1 = cute.arch.rcp_approx(s1) if has1 else Float32(0.0)
        lse0 = (mu0 + lg2_approx(s0)) * LN2 if has0 else neg_inf
        lse1 = (mu1 + lg2_approx(s1)) * LN2 if has1 else neg_inf
        col_bytes = Int64(t * 4)
        o_base = Int64(mO.iterator.toint())
        lse_base = Int64(mLSE.iterator.toint())
        self.store_row(
            acc,
            0,
            r0,
            lse0,
            o_base + q_off0 * 2 + col_bytes,
            lse_base + lse_off0 * 4,
            ok0,
            t,
        )
        self.store_row(
            acc,
            2,
            r1,
            lse1,
            o_base + q_off1 * 2 + col_bytes,
            lse_base + lse_off1 * 4,
            ok1,
            t,
        )

    @cute.jit
    def accumulate_piece(
        self,
        acc: cute.Tensor,
        ws_base: Int64,
        piece: Int32,
        tid: Int32,
        w0: Float32,
        w1: Float32,
    ):
        pp = ws_base + Int64(piece) * self.piece_o_bytes + Int64(tid * 16)
        for i in cutlass.range_constexpr(16):
            v = ld_global_cg_v4_f32(pp + i * (self.num_mma_threads * 16))
            acc[4 * i + 0] = acc[4 * i + 0] + w0 * v[0]
            acc[4 * i + 1] = acc[4 * i + 1] + w0 * v[1]
            acc[4 * i + 2] = acc[4 * i + 2] + w1 * v[2]
            acc[4 * i + 3] = acc[4 * i + 3] + w1 * v[3]

    @cute.jit
    def apply_mask(self, tS: cute.Tensor, words, t: Int32):
        """Set the columns whose validity bit is clear to -inf (no-op for an all-valid tile).

        Column ``nf * 8 + 2 t + e`` of the tile is bit ``(nf % 4) * 8 + 2 t + e`` of word ``nf / 4``.
        """
        w0, w1, w2, w3 = words
        if (w0 & w1 & w2 & w3) != Int32(-1):
            sh = tuple(Uint32(w) >> Uint32(2 * t) for w in words)
            neg_inf = Float32(-math.inf)
            for nf in cutlass.range_constexpr(16):
                for e in cutlass.range_constexpr(2):
                    bit = (sh[nf // 4] >> Uint32((nf % 4) * 8 + e)) & Uint32(1)
                    keep = bit != Uint32(0)
                    tS[nf * 4 + e] = tS[nf * 4 + e] if keep else neg_inf
                    tS[nf * 4 + 2 + e] = tS[nf * 4 + 2 + e] if keep else neg_inf

    @cute.jit
    def softmax_rows(
        self,
        tS: cute.Tensor,
        tO: cute.Tensor,
        r: int,
        m_old: Float32,
        l_old: Float32,
        scale_log2: Float32,
    ):
        """Online softmax on the 32 columns of one of the thread's two rows (``r`` = 0 or 2)."""
        mx = fmax_f32(tS[r], tS[r + 1])
        for nf in cutlass.range_constexpr(1, 16):
            mx = fmax_f32(mx, fmax_f32(tS[nf * 4 + r], tS[nf * 4 + r + 1]))
        mx = fmax_f32(mx, shfl_bfly_f32(mx, 1))
        mx = fmax_f32(mx, shfl_bfly_f32(mx, 2))
        m_new = fmax_f32(m_old, mx)
        # A row with no valid column so far keeps m = -inf; subtracting 0 keeps exp2 at 0.
        m_use = Float32(0.0) if m_new == Float32(-math.inf) else m_new
        alpha = cute.arch.exp2((m_old - m_use) * scale_log2)
        neg_m_scaled = -(m_use * scale_log2)
        row_sum = Float32(0.0)
        for nf in cutlass.range_constexpr(16):
            for e in cutlass.range_constexpr(2):
                p = cute.arch.exp2(tS[nf * 4 + r + e] * scale_log2 + neg_m_scaled)
                tS[nf * 4 + r + e] = p
                row_sum = row_sum + p
        for df in cutlass.range_constexpr(16):
            tO[df * 4 + r] = tO[df * 4 + r] * alpha
            tO[df * 4 + r + 1] = tO[df * 4 + r + 1] * alpha
        return m_new, l_old * alpha + row_sum

    @cute.jit
    def finalize_row(self, l: Float32, m: Float32, scale_log2: Float32):
        """Quad-reduce the row sum; return (1 / l or 0, log2-domain LSE or -inf)."""
        l = l + shfl_bfly_f32(l, 1)
        l = l + shfl_bfly_f32(l, 2)
        has = l > Float32(0.0)
        inv = cute.arch.rcp_approx(l) if has else Float32(0.0)
        lse2 = m * scale_log2 + lg2_approx(l) if has else Float32(-math.inf)
        return inv, lse2

    @cute.jit
    def store_row(
        self,
        tO: cute.Tensor,
        r: int,
        inv: Float32,
        lse: Float32,
        o_ptr: Int64,
        lse_ptr: Int64,
        ok: Boolean,
        t: Int32,
    ):
        for df in cutlass.range_constexpr(16):
            packed = cvt_pack(
                self.ab_type, tO[df * 4 + r] * inv, tO[df * 4 + r + 1] * inv
            )
            st_global_b32_pred(o_ptr + df * 16, packed, ok)
        st_global_f32_pred(lse_ptr, lse, ok and t == 0)


# =============================================================================
# Python entry point
# =============================================================================
def ffa_fwd_sm120_index(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    q_block_size: int = 1,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    topk_len: Optional[torch.Tensor] = None,
    assume_valid_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass of index-sparse (DSA) attention on SM120.

    Args:
        q: ``(total_q, num_heads_q, 128)`` fp16 / bf16.
        k, v: ``(total_k, num_heads_kv, 128)``, same dtype as ``q``.
        index_sparse_indices: int32 ``(ceil(total_q / q_block_size), num_heads_kv, topk)``.
            Row ``[b, h_kv]`` lists the global K token ids attended by q tokens
            ``[b * q_block_size, (b + 1) * q_block_size)`` of every q head of kv head ``h_kv``;
            ``-1`` and ids outside ``[0, total_k)`` are skipped. ``topk`` must be a multiple of
            ``FFAFwdSm120Index.topk_align`` (4).
        q_block_size: q tokens per index row; ``q_block_size * (num_heads_q / num_heads_kv)``
            must be 128.
        out, lse: optional preallocated ``(total_q, num_heads_q, 128)`` / fp32
            ``(total_q, num_heads_q)`` outputs.
        topk_len: optional int32 ``(ceil(total_q / q_block_size), num_heads_kv)``; only the first
            ``topk_len[b, h_kv]`` slots of each index row are used. Rows then differ in length,
            so this disables the stream-K schedule (one CTA per tile instead).
        assume_valid_ids: assert that every entry of ``index_sparse_indices`` is in
            ``[0, total_k)``. Together with ``topk % 128 == 0`` and no ``topk_len`` this selects a
            compile-time path without validity ballots and softmax masking. A false assertion
            silently attends the wrong tokens.

    Returns:
        ``(out, lse)``: ``lse`` is the natural-log logsumexp of the scaled scores; a row that
        attends no valid token gets ``out = 0`` and ``lse = -inf``.
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
    tile_m = FFAFwdSm120Index.tile_m
    assert q_block_size * group == tile_m, (
        f"q_block_size * (num_heads_q / num_heads_kv) must be {tile_m}, "
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
    assert (
        topk > 0 and topk % FFAFwdSm120Index.topk_align == 0
    ), f"topk must be a positive multiple of {FFAFwdSm120Index.topk_align}"
    assert total_k * num_heads_kv < 2**31, "total_k * num_heads_kv must fit in int32"
    if topk_len is not None:
        assert topk_len.dtype == torch.int32 and topk_len.shape == (
            num_q_blocks,
            num_heads_kv,
        )
        topk_len = topk_len.contiguous()
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    # Positive scaling preserves score ordering for online softmax.
    assert (
        math.isfinite(softmax_scale) and softmax_scale > 0
    ), f"softmax_scale must be finite and positive, got {softmax_scale}"

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    # Index vector loads require 16 B row alignment.
    index_sparse_indices = aligned_rows(index_sparse_indices)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(total_q, num_heads_q, dtype=torch.float32, device=q.device)
    assert out.shape == q.shape and out.dtype == q.dtype and out.is_contiguous()
    assert lse.shape == (total_q, num_heads_q) and lse.dtype == torch.float32
    assert lse.is_contiguous()

    dtype = to_cute_dtype(q.dtype)
    stream_k = topk_len is None
    ids_all_valid = (
        bool(assume_valid_ids)
        and topk % FFAFwdSm120Index.tile_n == 0
        and topk_len is None
    )
    num_tiles = num_q_blocks * num_heads_kv
    workspace = counter = None
    num_ctas = 0
    two_pieces = False
    if stream_k:
        n_kv = (topk + FFAFwdSm120Index.tile_n - 1) // FFAFwdSm120Index.tile_n
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        assert num_tiles * n_kv < 2**31, "too many (tile, KV block) work units"
        num_ctas = min(num_sms, num_tiles * n_kv)
        two_pieces = num_tiles * n_kv // num_ctas >= n_kv - 1
        piece_bytes = FFAFwdSm120Index.piece_o_bytes + FFAFwdSm120Index.piece_lse_bytes
        workspace = torch.empty(
            num_ctas * 2 * piece_bytes, dtype=torch.uint8, device=q.device
        )
        # The counter belongs to this invocation, like the workspace it guards: a buffer shared
        # across calls would let two concurrent launches (two CUDA graphs replayed on different
        # streams, say) mix their completion counts while holding separate workspaces.
        counter = torch.zeros(num_tiles, dtype=torch.int32, device=q.device)
    compile_key = (
        dtype,
        group,
        torch.cuda.get_device_capability(q.device),
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
        ids_all_valid,
        stream_k,
        two_pieces,
    )
    compile_cache = ffa_fwd_sm120_index.compile_cache  # type: ignore[attr-defined]
    if compile_key not in compile_cache:
        kernel = FFAFwdSm120Index(
            group, ids_all_valid=ids_all_valid, stream_k=stream_k, two_pieces=two_pieces
        )
        q_t, k_t, v_t, out_t = [to_cute_tensor(t) for t in (q, k, v, out)]
        lse_t = to_cute_tensor(lse, assumed_align=4)
        idx_t = to_cute_tensor(index_sparse_indices)
        tl_t = None if stream_k else to_cute_tensor(topk_len, assumed_align=4)
        ws_t = to_cute_tensor(workspace) if stream_k else None
        cnt_t = to_cute_tensor(counter, assumed_align=4) if stream_k else None
        compile_cache[compile_key] = cute.compile(
            kernel,
            q_t,
            k_t,
            v_t,
            out_t,
            lse_t,
            float(softmax_scale),
            idx_t,
            tl_t,
            ws_t,
            cnt_t,
            cutlass.Int32(num_heads_kv),
            cutlass.Int32(num_ctas),
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
        topk_len,
        workspace,
        counter,
        cutlass.Int32(num_heads_kv),
        cutlass.Int32(num_ctas),
    )
    return out, lse


ffa_fwd_sm120_index.compile_cache = get_jit_cache("fwd_sm120_index")  # type: ignore[attr-defined]
