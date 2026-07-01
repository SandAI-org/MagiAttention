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

# mypy: disable-error-code="index,union-attr,assignment,arg-type"

import math
from enum import IntEnum
from typing import Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_helpers
from cutlass import Boolean, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Enumerations that match the HW encodings (values MUST stay identical)
# ---------------------------------------------------------------------------


class Major(IntEnum):  # matrix “layout” in the ISA docs
    K = 0
    MN = 1


class ScaleIn(IntEnum):  # negate flags
    One = 0
    Neg = 1


class Saturate(IntEnum):
    False_ = 0
    True_ = 1


class CFormat(IntEnum):  # 2-bit field (bits 4-5)
    F16 = 0
    F32 = 1
    S32 = 2


class F16F32Format(IntEnum):  # 3-bit field (A/B element type)
    F16 = 0
    BF16 = 1
    TF32 = 2


class S8Format(IntEnum):
    UINT8 = 0
    INT8 = 1


class MXF8F6F4Format(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


class LayoutType(IntEnum):  # occupies the top-3 bits [61:64)
    SWIZZLE_NONE = 0  # (a.k.a. “INTERLEAVE” in older docs)
    SWIZZLE_128B_BASE32B = 1
    SWIZZLE_128B = 2
    SWIZZLE_64B = 4
    SWIZZLE_32B = 6
    # values 3,5,7 are reserved / illegal for UMMA


# ---------------------------------------------------------------------------
# CUTLASS-type → encoding helpers
# ---------------------------------------------------------------------------


def to_UMMA_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 3-bit encoding for Matrix A/B.
    """
    if cutlass_type is cutlass.Int8:
        return S8Format.INT8
    # Unsigned 8-bit (if available in your CUTLASS build)
    if cutlass_type is cutlass.Uint8:
        return S8Format.UINT8
    # FP-16 / BF-16
    if cutlass_type is cutlass.Float16:
        return F16F32Format.F16
    if cutlass_type is cutlass.BFloat16:
        return F16F32Format.BF16
    # TensorFloat-32 (8-bit exponent, 10-bit mantissa packed in 19 bits)
    if cutlass_type is cutlass.TFloat32:
        return F16F32Format.TF32
    # Float-8 / Float-6 / Float-4 – add whenever CUTLASS exposes them
    if cutlass_type is cutlass.Float8E4M3FN:
        return MXF8F6F4Format.E4M3
    if cutlass_type is cutlass.Float8E5M2:
        return MXF8F6F4Format.E5M2
    raise TypeError(f"Unsupported CUTLASS scalar type for A/B: {cutlass_type!r}")


def to_C_format(cutlass_type) -> int:
    """
    Map a CUTLASS scalar class to the 2-bit accumulator encoding.
    """
    if cutlass_type is cutlass.Float16:
        return CFormat.F16
    if cutlass_type is cutlass.Float32:
        return CFormat.F32
    if cutlass_type is cutlass.Int32:
        return CFormat.S32
    raise TypeError(
        f"Unsupported CUTLASS scalar type for accumulator: {cutlass_type!r}"
    )


# ---------------------------------------------------------------------------
# Instruction-descriptor (idesc) builders for tcgen05 / UMMA MMA
#
# Build the 32-bit MMA instruction descriptor from CUTLASS scalar types.
# All matrix / accumulator operands must be passed as CUTLASS scalar classes
# (e.g. cutlass.Float16); raw integers are not accepted.
# ---------------------------------------------------------------------------

# Ported Cutlass code from C++ to Python:
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_hpp
# https://github.com/NVIDIA/cutlass/blob/main/include/cute/atom/mma_traits_sm100.hpp


def make_instr_desc(
    a_type,  # CUTLASS scalar class, e.g. cutlass.Int8
    b_type,
    c_type,
    M: int,  # 64, 128 or 256
    N: int,  # 8 … 256 (multiple of 8)
    a_major: Major,
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    """
    Build the 32-bit instruction descriptor for Blackwell MMA.
    All matrix/accumulator **types must be CUTLASS scalar classes** –
    passing integers is forbidden.
    """
    # --- encode element formats -------------------------------------------------
    a_fmt = int(to_UMMA_format(a_type))
    b_fmt = int(to_UMMA_format(b_type))
    c_fmt = int(to_C_format(c_type))

    # --- range checks on M/N -----------------------------------------------------
    if M not in (64, 128, 256):
        raise ValueError("M must be 64, 128 or 256")
    if N < 8 or N > 256 or (N & 7):
        raise ValueError("N must be a multiple of 8 in the range 8…256")

    m_dim = M >> 4  # 5-bit field
    n_dim = N >> 3  # 6-bit field

    # fmt: off
    # --- pack the bit-fields -----------------------------------------------------
    desc = 0
    desc |= (0                 & 0x3) << 0        # sparse_id2 (always 0 here)
    desc |= (int(is_sparse)    & 0x1) << 2        # sparse_flag
    desc |= (int(c_sat)        & 0x1) << 3        # saturate
    desc |= (c_fmt             & 0x3) << 4        # c_format
    desc |= (a_fmt             & 0x7) << 7        # a_format
    desc |= (b_fmt             & 0x7) << 10       # b_format
    desc |= (int(a_neg)        & 0x1) << 13       # a_negate
    desc |= (int(b_neg)        & 0x1) << 14       # b_negate
    desc |= (int(a_major)      & 0x1) << 15       # a_major
    desc |= (int(b_major)      & 0x1) << 16       # b_major
    desc |= (n_dim             & 0x3F) << 17      # n_dim (6 bits)
    desc |= (m_dim             & 0x1F) << 24      # m_dim (5 bits)
    desc |= (int(max_shift)    & 0x3) << 30       # max_shift (2 bits)
    # fmt: on

    return desc & 0xFFFF_FFFF  # ensure 32-bit result


def mma_op_to_idesc(op: cute.nvgpu.tcgen05.mma.MmaOp):
    return make_instr_desc(
        op.a_dtype,
        op.b_dtype,
        op.acc_dtype,
        op.shape_mnk[0],
        op.shape_mnk[1],
        Major.K
        if op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
        Major.K
        if op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
        else Major.MN,
    )


# ---------------------------------------------------------------------------
# Shared-memory descriptor helpers
#
# Resolve the SWIZZLE_* family from a tensor layout and build the 64-bit UMMA
# shared-memory descriptor (compile-time base + dynamic start address).
# ---------------------------------------------------------------------------


def _layout_type(swizzle: cute.Swizzle) -> LayoutType:
    B, M, S = swizzle.num_bits, swizzle.num_base, swizzle.num_shift

    if M == 4:  # Swizzle<*,4,3>
        if S != 3:
            raise ValueError("Unexpected swizzle shift – want S==3 for M==4")
        return {
            0: LayoutType.SWIZZLE_NONE,
            1: LayoutType.SWIZZLE_32B,
            2: LayoutType.SWIZZLE_64B,
            3: LayoutType.SWIZZLE_128B,
        }[
            B
        ]  # KeyError ⇒ invalid B→ raise
    if M == 5:  # Swizzle<2,5,2> (the only legal triple for M==5)
        if (B, S) != (2, 2):
            raise ValueError("Only Swizzle<2,5,2> supported for 128B_BASE32B")
        return LayoutType.SWIZZLE_128B_BASE32B

    # Any other (M,B,S) triple is not a UMMA-legal shared-memory layout
    raise ValueError("Unsupported swizzle triple for UMMA smem descriptor")


def make_smem_desc_base(
    layout: cute.Layout, swizzle: cute.Swizzle, major: Major
) -> int:
    """
    Convert a 2-D *shared-memory* Cute layout into the Blackwell 64-bit
    smem-descriptor, without the smem start address.
    layout must correspond to layout of an uint128 tensor.
    """
    # ------------------------------------------------------------------ meta
    layout_type = _layout_type(swizzle)  # resolve SWIZZLE_* family

    VERSION = 1  # bits 46–47
    LBO_MODE = 0  # bit  52
    BASE_OFFSET = 0  # bits 49–51   (CUTLASS always 0)

    # ---------------------------------------------------------- strides  (units: uint128_t = 16 B)
    swizzle_atom_mn_size = {
        LayoutType.SWIZZLE_NONE: 1,
        LayoutType.SWIZZLE_32B: 2,
        LayoutType.SWIZZLE_64B: 4,
        LayoutType.SWIZZLE_128B: 8,
        LayoutType.SWIZZLE_128B_BASE32B: 8,
    }[layout_type]

    if major is Major.MN:
        swizzle_atom_k_size = 4 if layout_type is LayoutType.SWIZZLE_128B_BASE32B else 8
        canonical_layout = cute.logical_divide(
            layout, (swizzle_atom_mn_size, swizzle_atom_k_size)
        )
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError(
                "Not a canonical UMMA_MN Layout: Expected profile failure."
            )
        stride_00 = canonical_layout.stride[0][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_00 != 1:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if stride_10 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_MN Layout: Expected stride failure.")
        stride_01, stride_11 = (
            canonical_layout.stride[0][1],
            canonical_layout.stride[1][1],
        )
        if layout_type is LayoutType.SWIZZLE_NONE:
            stride_byte_offset, leading_byte_offset = stride_01, stride_11
        else:
            stride_byte_offset, leading_byte_offset = stride_11, stride_01
    else:
        if layout_type == LayoutType.SWIZZLE_128B_BASE32B:
            raise ValueError("SWIZZLE_128B_BASE32B is invalid for Major-K")
        if not cute.size(layout.shape[0]) % 8 == 0:
            raise ValueError(
                "Not a canonical UMMA_K Layout: Expected MN-size multiple of 8."
            )
        canonical_layout = cute.logical_divide(layout, (8, 2))
        if not cute.is_congruent(canonical_layout, ((1, 1), (1, 1))):
            raise ValueError("Not a canonical UMMA_K Layout: Expected profile failure.")
        stride_00 = canonical_layout.stride[0][0]
        if stride_00 != swizzle_atom_mn_size:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_10 = canonical_layout.stride[1][0]
        if layout_type is not LayoutType.SWIZZLE_NONE and stride_10 != 1:
            raise ValueError("Not a canonical UMMA_K Layout: Expected stride failure.")
        stride_01 = canonical_layout.stride[0][1]
        stride_byte_offset, leading_byte_offset = stride_01, stride_10

    # ------------------------------------------------------------------ pack
    desc = 0
    # leading_byte_offset_  [16:30)
    desc |= (leading_byte_offset & 0x3FFF) << 16
    # stride_byte_offset_   [32:46)
    desc |= (stride_byte_offset & 0x3FFF) << 32
    # version_             [46:48)
    desc |= (VERSION & 0x3) << 46
    # base_offset_         [49:52)
    desc |= (BASE_OFFSET & 0x7) << 49
    # lbo_mode_            [52:53)
    desc |= (LBO_MODE & 0x1) << 52
    # layout_type_         [61:64)
    desc |= (int(layout_type) & 0x7) << 61

    return desc & 0xFFFF_FFFF_FFFF_FFFF  # force 64-bit width


def make_smem_desc_start_addr(start_addr: cute.Pointer) -> cutlass.Int32:
    # 14 bits, remove 4 LSB (bits 0-13 in desc)
    return (start_addr.toint() & 0x3FFFF) >> 4


def smem_desc_base_from_tensor(sA: cute.Tensor, major: Major) -> int:
    sA_swizzle = sA.iterator.type.swizzle_type
    return make_smem_desc_base(
        cute.recast_layout(128, sA.element_type.width, sA.layout[0]),
        sA_swizzle,
        major,
    )


# ---------------------------------------------------------------------------
# tcgen05 / UMMA GEMM emitters
#
# Issue MMA instructions over CUTLASS tensors: the MMA-kind dispatch helper,
# high-level cute.gemm wrappers, and low-level inline-PTX emitters that
# pre-stage the shared-memory descriptors.
# ---------------------------------------------------------------------------


def _tcgen05_mma_kind(op: cute.nvgpu.tcgen05.mma.MmaOp) -> str:
    if isinstance(op, tcgen05.mma.MmaF16BF16Op):
        return "f16"
    if isinstance(op, tcgen05.mma.MmaTF32Op):
        return "tf32"
    if isinstance(op, tcgen05.mma.MmaI8Op):
        return "i8"
    if isinstance(op, tcgen05.mma.MmaFP8Op):
        return "f8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF8Op):
        return "mxf8f6f4"
    if isinstance(op, tcgen05.mma.MmaMXF4Op):
        return "mxf4"
    if isinstance(op, tcgen05.mma.MmaMXF4NVF4Op):
        return "mxf4nvf4"
    raise TypeError(f"Unsupported tcgen05 MMA op kind: {type(op).__name__}")


@cute.jit
def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    swap_AB: bool = False,
    num_unroll_groups: int = 1,
) -> None:
    if const_expr(swap_AB):
        return gemm_w_idx(
            tiled_mma, acc, tCrB, tCrA, B_idx, A_idx, zero_init=zero_init, swap_AB=False
        )
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]

        mma_atom = cute.make_mma_atom(tiled_mma.op)
        for k in cutlass.range(
            cute.size(tCrA.shape[2]),
            unroll=cute.size(tCrA.shape[2]) // num_unroll_groups,
        ):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
            cute.gemm(mma_atom, acc, rA[None, None, k], rB[None, None, k], acc)


@cute.jit
def gemm_ptx_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    **kwargs,
) -> None:
    rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
    rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
    sA_cur = None
    if const_expr(sA is not None):
        sA_cur = sA if const_expr(A_idx is None) else sA[None, None, None, A_idx]
    sB_cur = sB if const_expr(B_idx is None) else sB[None, None, None, B_idx]
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    acc_tmem_addr = acc.iterator.toint()
    gemm_ptx_partial(
        mma_atom.op,
        acc_tmem_addr,
        rA,
        rB,
        sA_cur,
        sB_cur,
        zero_init=zero_init,
        cta_group=cta_group,
        **kwargs,
    )


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or k != 0)
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


def i64_to_i32x2(i: int) -> Tuple[int, int]:
    """Convert a 64-bit integer to a tuple of two 32-bit integers."""
    return i & 0xFFFF_FFFF, (i >> 32) & 0xFFFF_FFFF


@cute.jit
def gemm_ptx(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else None
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo) | make_smem_desc_start_addr(
            sA[None, None, 0].iterator
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo) | make_smem_desc_start_addr(
        sB[None, None, 0].iterator
    )
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(not is_ts):
            smem_desc_a_lo = smem_desc_start_a_lo + (
                (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            )
        smem_desc_b_lo = smem_desc_start_b_lo + (
            (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        )
        # with cute.arch.elect_one():
        #     cute.printf("smem_desc_a_lo = {}, smem_desc_b_lo = {}", smem_desc_a_lo, smem_desc_b_lo)
        #     cute.printf("smem_desc_a_lo_correct = {}, smem_desc_b_lo_correct = {}", smem_desc_a_lo_correct, smem_desc_b_lo_correct)  # noqa: E501
        with cute.arch.elect_one():
            if const_expr(not is_ts):
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        smem_desc_a_lo.ir_value(),
                        smem_desc_b_lo.ir_value(),
                        Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
                    ".reg .b32 idesc;\n\t"
                    f"mov.b32 idesc, {hex(idesc)};\n\t"
                    f"mov.b64 smem_desc_a, {{$1, {hex(smem_desc_a_hi)}}};\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            else:
                llvm.inline_asm(
                    None,
                    [
                        acc.iterator.toint().ir_value(),
                        tCrA[None, None, k].iterator.toint().ir_value(),
                        smem_desc_b_lo.ir_value(),
                        Int32(not zero_init or k != 0).ir_value(),
                    ],
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    ".reg .b64 smem_desc_b;\n\t"
                    f"mov.b64 smem_desc_b, {{$2, {hex(smem_desc_b_hi)}}};\n\t"
                    "setp.ne.b32 p, $3, 0;\n\t"
                    f"tcgen05.mma.cta_group::1.kind::{kind} [$0], [$1], smem_desc_b, {hex(idesc)}, p;\n\t"
                    "}\n",
                    "r,r,r,r",
                    has_side_effects=True,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )


@cute.jit
def gemm_ptx_loop(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    if const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * sA.element_type.width // 8) >> 4
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in cutlass.range_constexpr(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1]
        for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * sB.element_type.width // 8) >> 4
        for k in cutlass.range_constexpr(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1]
        for k in cutlass.range_constexpr(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 smem_desc_a_lo, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                acc.iterator.toint().ir_value(),
                Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                Int32(smem_desc_start_b_lo).ir_value(),
                Int32(not zero_init).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            "mov.b32 tmem_a, $1;\n\t"
            "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} [$0], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [$0], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} "
                    f"[$0], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in cutlass.range_constexpr(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    # sA_offset: Int32 = 0,
    # acc_offset: Int32 = 0,
    tA_addr: Optional[Int32] = None,
    cta_group: int = 1,
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo | make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
        # ) + sA_offset
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
            f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                    f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            # "r,r,r",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            # Int32(cute.arch.make_warp_uniform(tCrA[None, None, 0].iterator.toint())).ir_value(),
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert (
                mbar_phase is not None
            ), "mbar_phase must be provided when mbar_ptr is not None"
            assert (
                split_arrive is not None
            ), "split_arrive must be provided when mbar_ptr is not None"
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
            f"[tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                    f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    cute.size(tCrA.shape[2])
                    if const_expr(mbar_ptr is None)
                    else split_arrive_idx,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                        f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_partial1(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: cutlass.Constexpr[int],
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA_base_addr_for_desc: Int32,
    sA_addr_offset_for_desc: cutlass.Constexpr[int],
    sA_stage: Int32,
    sB_base_addr_for_desc: Int32,
    sB_addr_offset_for_desc: cutlass.Constexpr[int],
    sB_stage: Int32,
    sA_layout: Optional[cute.Layout],
    sB_layout: Optional[cute.Layout],
    sA_swizzle: Optional[cute.Swizzle],
    sB_swizzle: cute.Swizzle,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert (
            sA_layout is not None
        ), "sA_layout must be provided when a_src is not TMEM"
        assert (
            sA_swizzle is not None
        ), "sA_swizzle must be provided when a_src is not TMEM"
    idesc: int = const_expr(mma_op_to_idesc(op))
    kind = _tcgen05_mma_kind(op)
    if const_expr(not is_ts):
        smem_desc_base_a: int = const_expr(
            make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                Major.K
                if const_expr(
                    op.a_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K
                )
                else Major.MN,
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b: int = const_expr(
        make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            Major.K
            if const_expr(op.b_major_mode == cute.nvgpu.tcgen05.mma.OperandMajorMode.K)
            else Major.MN,
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)
    mask = [Int32(0)] * 4

    if const_expr(not is_ts):
        offset_a = [
            (cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 8) >> 4
            for k in range(cute.size(tCrA.shape[2]))
        ]
    else:
        offset_a = [
            cute.crd2idx((0, 0, k), sA_layout) * op.a_dtype.width // 32
            for k in range(cute.size(tCrA.shape[2]))
        ]
    offset_a_diff = [
        offset_a[k] - offset_a[k - 1] for k in range(1, cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        (cute.crd2idx((0, 0, k), sB_layout) * op.b_dtype.width // 8) >> 4
        for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        # smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | make_smem_desc_start_addr(sA[None, None, 0].iterator))
        smem_desc_start_a_lo = const_expr(smem_desc_base_a_lo)
    else:
        smem_desc_start_a_lo = None
    # smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | make_smem_desc_start_addr(sB[None, None, 0].iterator))
    smem_desc_start_b_lo = const_expr(smem_desc_base_b_lo)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                # Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(sA_base_addr_for_desc).ir_value(),
                Int32(sA_stage).ir_value(),
                # Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(sB_base_addr_for_desc).ir_value(),
                Int32(sB_stage).ir_value(),
                Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            # "mov.b32 smem_desc_a_lo, $0;\n\t"
            # f"add.u32 smem_desc_a_lo, $0, {hex(smem_desc_start_a_lo)};\n\t"
            f"mad.lo.u32 smem_desc_a_lo, $1, {hex(sA_addr_offset_for_desc)}, $0;\n\t"
            # "mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mad.lo.u32 smem_desc_b_lo, $3, {hex(sB_addr_offset_for_desc)}, $2;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $4, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} "
            f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} "
                    f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, {{$5, $6, $7, $8}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(tCrA[None, None, 0].iterator.toint()).ir_value(),
                Int32(smem_desc_start_b_lo).ir_value(),
                Int32(not zero_init).ir_value(),
                mask[0].ir_value(),
                mask[1].ir_value(),
                mask[2].ir_value(),
                mask[3].ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_a, $1;\n\t"
            f"mov.b32 smem_desc_b_lo, $2;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $3, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} "
            f"[$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::{kind} "
                    f"[$0], [tmem_a], smem_desc_b, idesc, {{$4, $5, $6, $7}}, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def gemm_ptx_precomputed(
    acc_tmem_addr: Int32,
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_start_b: Int32,
    idesc: int,
    smem_desc_base_a: Optional[int],
    smem_desc_base_b: int,
    tCrA_layout: cute.Layout,
    tCrB_layout: cute.Layout,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    kind: str = "f16",
) -> None:
    # acc_tmem_addr += acc_offset
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    else:
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)

    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # else cute.recast_layout(32, tCrA.element_type.width, tCrA_layout)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    offset_a_diff = [offset_a[k] - offset_a[k - 1] for k in range(1, num_k_tile)]
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]
    offset_b_diff = [offset_b[k] - offset_b[k - 1] for k in range(1, num_k_tile)]

    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
        # smem_desc_start_a_lo = smem_desc_start_a
    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                # acc.iterator.toint().ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
            f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 smem_desc_a_lo, smem_desc_a_lo, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                    f"[tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            # "r,r,r",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        input_args = [
            Int32(cute.arch.make_warp_uniform(smem_desc_start_a)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(not zero_init).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert (
                mbar_phase is not None
            ), "mbar_phase must be provided when mbar_ptr is not None"
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            # [
            #     # acc.iterator.toint().ir_value(),
            #     Int32(tCrA_layout[None, None, 0].iterator.toint()).ir_value(),
            #     Int32(smem_desc_start_b_lo).ir_value(),
            #     Int32(not zero_init).ir_value(),
            # ],
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            "setp.ne.b32 p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
            f"[tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    # f"add.u32 tmem_a, tmem_a, {hex(offset_a_diff[k - 1])};\n\t"
                    # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, 1;\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                    f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    num_k_tile if const_expr(mbar_ptr is None) else num_k_tile // 4 * 3,
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        # f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                        f"[tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(num_k_tile // 4 * 3, num_k_tile)
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_base_a: Optional[int],
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # else cute.recast_layout(32, tCrA.element_type.width, tCrA_layout)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, {{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp, var_name: str = "idesc"
) -> None:
    idesc = const_expr(mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t" f"mov.b32 {var_name}, {hex(idesc)};\n\t",  # noqa
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    # idesc: int,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
    cta_group: int = 1,
    kind: str = "f16",
) -> None:
    is_ts = False
    num_k_tile = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]

    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    pred_str = "p" if isinstance(zero_init, Boolean) else "0" if zero_init else "1"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(not zero_init).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t" ".reg .pred leader_thread;\n\t" ".reg .pred p;\n\t"
            # ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            # ".reg .b64 smem_desc_b;\n\t"
            f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            # f"mov.b32 idesc, {hex(idesc)};\n\t"
            # f"mov.b32 tmem_acc, {hex(acc_tmem_addr)};\n\t"
            f"mov.b32 tmem_acc, $2;\n\t"
            "mov.b32 smem_desc_b_lo_start, $0;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
            f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
            f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            + "".join(
                (
                    f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "setp.ne.b32 p, $1, 0;\n\t"
            # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b, idesc, {pred_str};\n\t"  # noqa: E501
            f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
            f"[tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
            + "".join(
                (
                    # f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    # f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    # f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    # f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    # f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, idesc, 1;\n\t"  # noqa: E501
                    # f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b, {idesc_var_name}, 1;\n\t"  # noqa: E501
                    f"@leader_thread tcgen05.mma.cta_group::{cta_group}.kind::{kind} "
                    f"[tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, {idesc_var_name}, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


# ---------------------------------------------------------------------------
# Copy helpers (SM100-specific tiled / tmem / cluster copies)
# ---------------------------------------------------------------------------


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    major_mode_size: int,
    num_threads: int,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom, num_wg: int = 1, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_dp, num_bits, num_rep, _ = blackwell_helpers.get_tmem_copy_properties(
        tmem_copy_atom
    )
    assert num_dp == 32
    assert num_bits == 32
    tiler_mn = (cute.make_layout((128 * num_rep * num_wg // 32, 32), stride=(32, 1)),)
    layout_tv = cute.make_layout(
        ((32, 4, num_wg), (num_rep, 32)),
        stride=((0, 1, 4 * num_rep), (4, 4 * num_rep * num_wg)),
    )
    return cute.make_tiled_copy(tmem_copy_atom, layout_tv, tiler_mn)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_src_ptr_i32 = smem_src_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_dst_ptr_i32 = set_block_rank(
        smem_dst_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_dst_ptr_i32,
            smem_src_ptr_i32,
            mbar_ptr_i32,
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
