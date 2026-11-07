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

import inspect
import math
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Type, overload

import cutlass
import cutlass.cute as cute
import torch
from cutlass import (
    BFloat16,
    Boolean,
    Float32,
    Int16,
    Int32,
    Int64,
    const_expr,
    pipeline,
)
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import NumericMeta, T, dsl_user_op

from .dsl_compat import FastDivmodDivisor

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)

load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile

# NVVM result-type inference depends on both the DSL and CUDA backend versions.
_FMAX_REQUIRES_RESULT_TYPE = "res" in inspect.signature(nvvm.fmax).parameters
_ATOMICRMW_REQUIRES_RESULT_TYPE = "res" in inspect.signature(nvvm.atomicrmw).parameters


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster_size=cluster_size
    )


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(
        s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1]
    )
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(
        t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t))
    )


def to_cute_tensor(
    t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True
):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    # NOTE: torch 2.9.1 doesn't support fp8 via DLPack but 2.11.0 nightly does
    # currently export raw bytes as uint8 and tell cutlass correct type
    # can directly export as fp8 when torch supports it
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = from_dlpack(
            t.view(torch.uint8).detach(),
            assumed_align=assumed_align,
            enable_tvm_ffi=enable_tvm_ffi,
        )
        tensor.element_type = (
            cutlass.Float8E4M3FN
            if t.dtype == torch.float8_e4m3fn
            else cutlass.Float8E5M2
        )
    else:
        tensor = from_dlpack(
            t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi
        )
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI, tailored to FlexAttention aux tensors.
    This allows the user to specify alignment and leading dimension for aux tensors used in
    custom score_mod callables.
    """
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def get_aux_tensor_metadata(aux_tensors):
    return tuple(
        (
            getattr(t, "__assumed_align__", 0),
            getattr(t, "__leading_dim__", -1),
            hasattr(t, "__leading_dim__"),
        )
        for t in aux_tensors
    )


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())


# credit: monellz (https://github.com/NVIDIA/cutlass/issues/2658#issuecomment-3630564264)
def dump_kernel_attributes(compiled_kernel):
    import torch
    from cuda.bindings import driver
    from cutlass.utils import HardwareInfo

    device_id = torch.cuda.current_device()
    hardware_info = HardwareInfo(device_id=device_id)
    cubin_data = compiled_kernel.artifacts.CUBIN
    assert (
        cubin_data is not None
    ), "cubin_data is None, need '--keep-cubin' option when compiling"
    cuda_library = hardware_info._checkCudaErrors(
        driver.cuLibraryLoadData(cubin_data, None, None, 0, None, None, 0)
    )
    kernels = hardware_info._checkCudaErrors(
        driver.cuLibraryEnumerateKernels(1, cuda_library)
    )
    kernel = hardware_info._checkCudaErrors(driver.cuKernelGetFunction(kernels[0]))
    # more metrics: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1g5e92a1b0d8d1b82cb00dcfb2de15961b  # noqa: E501
    local_size_bytes = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            kernel,
        )
    )
    num_regs = hardware_info._checkCudaErrors(
        driver.cuFuncGetAttribute(
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS,
            kernel,
        )
    )

    print("--- Kernel Info ---")
    print(f"local_size_bytes: {local_size_bytes}")
    print(f"num_regs: {num_regs}")
    print("--- End Kernel Info ---")


# ---------------------------------------------------------------------------
# DSL helper user ops (device-side helpers used by the ffa kernels)
# ---------------------------------------------------------------------------

LOG2_E = math.log2(math.e)

# Obtained from sollya:
# fpminimax(exp(x * log(2.0)), 1, [|1,24...|],[0;1],relative);
POLY_EX2 = {
    0: (1.0),
    1: (
        1.0,
        0.922497093677520751953125,
    ),
    2: (
        1.0,
        0.6657850742340087890625,
        0.330107033252716064453125,
    ),
    3: (
        1.0,
        0.695146143436431884765625,
        0.227564394474029541015625,
        0.077119089663028717041015625,
    ),
    4: (
        1.0,
        0.693042695522308349609375,
        0.2412912547588348388671875,
        5.2225358784198760986328125e-2,
        1.3434938155114650726318359375e-2,
    ),
    5: (
        1.0,
        0.693151414394378662109375,
        0.24016360938549041748046875,
        5.5802188813686370849609375e-2,
        9.01452265679836273193359375e-3,
        1.86810153536498546600341796875e-3,
    ),
}


def compute_softmax_scale_log2(softmax_scale, score_mod):
    """Compute softmax_scale_log2 and adjusted softmax_scale based on whether score_mod is used.

    When score_mod is None, fold the log2(e) factor into softmax_scale_log2 and set softmax_scale
    to None. When score_mod is present, keep softmax_scale separate so it can be applied before
    the score_mod, and set softmax_scale_log2 to just the change-of-base constant.

    Returns (softmax_scale_log2, softmax_scale).
    """
    if const_expr(score_mod is None):
        return softmax_scale * LOG2_E, None
    else:
        return LOG2_E, softmax_scale


def compute_fastdiv_mods(
    mQ, mK, qhead_per_kvhead, pack_gqa, aux_tensors, mPageTable=None
):
    """Compute FastDivmodDivisor pairs for aux_tensors index computation.

    Returns a (seqlen_q_divmod, seqlen_k_divmod) tuple, or None if aux_tensors is None.
    """
    if const_expr(aux_tensors is None):
        return None
    seqlen_q = cute.size(mQ.shape[0]) // (
        qhead_per_kvhead if const_expr(pack_gqa) else 1
    )
    seqlen_k = (
        cute.size(mK.shape[0])
        if const_expr(mPageTable is None)
        else mK.shape[0] * mPageTable.shape[1]
    )
    return (FastDivmodDivisor(seqlen_q), FastDivmodDivisor(seqlen_k))


def make_tiled_copy_A(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)


def make_tiled_copy_B(
    copy_atom: cute.CopyAtom,
    tiled_mma: cute.TiledMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.TiledCopy:
    if const_expr(swapAB):
        return cute.make_tiled_copy_A(copy_atom, tiled_mma)
    else:
        return cute.make_tiled_copy_B(copy_atom, tiled_mma)


def mma_make_fragment_A(
    smem: cute.Tensor,
    thr_mma: cute.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor,
    thr_mma: cute.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_A(smem, thr_mma)
    else:
        return thr_mma.make_fragment_B(thr_mma.partition_B(smem))


def get_smem_store_atom(
    arch: cutlass.Constexpr[int],
    element_type: Type[cute.Numeric],
    transpose: bool = False,
) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=2 * element_type.width,
        )
    else:
        return cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
            element_type,
        )


@cute.jit
def warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@dsl_user_op
def cta_exit(*, loc=None, ip=None) -> None:
    """PTX `exit;`. Cluster-safe only if every CTA of the cluster takes it,
    before any barrier a peer could wait on."""
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32 | None = None,
    *,
    loc=None,
    ip=None,
) -> Float32:
    result_type = {"res": T.f32()} if _FMAX_REQUIRES_RESULT_TYPE else {}
    return Float32(
        nvvm.fmax(
            a=Float32(a).ir_value(loc=loc, ip=ip),
            b=Float32(b).ir_value(loc=loc, ip=ip),
            c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
            loc=loc,
            ip=ip,
            **result_type,
        )
    )


@cute.jit
def fmax_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        # if const_expr(init_val is None):
        #     init_val = -Float32.if
        # return x.reduce(cute.ReductionOp.MAX, init_val, 0)
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        # local_max = [res[0], res[1]]
        # for i in cutlass.range_constexpr(2, cute.size(x.shape), 2):
        #     local_max[0] = fmax(local_max[0], res[i + 0])
        #     local_max[1] = fmax(local_max[1], res[i + 1])
        # local_max[0] = fmax(local_max[0], local_max[1])
        # return local_max[0] if const_expr(init_val is None) else fmax(local_max[0], init_val)
        local_max = [res[0], res[1], res[2], res[3]]
        for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
            local_max[0] = fmax(local_max[0], res[i + 0])
            local_max[1] = fmax(local_max[1], res[i + 1])
            local_max[2] = fmax(local_max[2], res[i + 2])
            local_max[3] = fmax(local_max[3], res[i + 3])
        local_max[0] = fmax(local_max[0], local_max[1])
        local_max[2] = fmax(local_max[2], local_max[3])
        local_max[0] = fmax(local_max[0], local_max[2])
        return (
            local_max[0]
            if const_expr(init_val is None)
            else fmax(local_max[0], init_val)
        )
    else:
        # [2025-06-15] x.reduce only seems to use 50% 3-input max and 50% 2-input max
        # We instead force the 3-input max.
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_max_0 = (
            fmax(init_val, res[0], res[1])
            if const_expr(init_val is not None)
            else fmax(res[0], res[1])
        )
        local_max = [
            local_max_0,
            fmax(res[2], res[3]),
            fmax(res[4], res[5]),
            fmax(res[6], res[7]),
        ]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_max[0] = fmax(local_max[0], res[i], res[i + 1])
            local_max[1] = fmax(local_max[1], res[i + 2], res[i + 3])
            local_max[2] = fmax(local_max[2], res[i + 4], res[i + 5])
            local_max[3] = fmax(local_max[3], res[i + 6], res[i + 7])
        local_max[0] = fmax(local_max[0], local_max[1])
        return fmax(local_max[0], local_max[2], local_max[3])


@cute.jit
def fadd_reduce(
    x: cute.TensorSSA,
    init_val: float | Float32 | None = None,
    arch: cutlass.Constexpr[int] = 80,
) -> Float32:
    if const_expr(arch < 100 or cute.size(x.shape) % 8 != 0):
        if const_expr(init_val is None):
            init_val = Float32.zero
        return x.reduce(cute.ReductionOp.ADD, init_val, 0)
        # res = cute.make_rmem_tensor(x.shape, Float32)
        # res.store(x)
        # local_sum = [res[0], res[1], res[2], res[3]]
        # for i in cutlass.range_constexpr(4, cute.size(x.shape), 4):
        #     local_sum[0] += res[i + 0]
        #     local_sum[1] += res[i + 1]
        #     local_sum[2] += res[i + 2]
        #     local_sum[3] += res[i + 3]
        # local_sum[0] += local_sum[1]
        # local_sum[2] += local_sum[3]
        # local_sum[0] += local_sum[2]
        # return local_sum[0] if const_expr(init_val is None) else local_sum[0] + init_val
    else:
        res = cute.make_rmem_tensor(x.shape, Float32)
        res.store(x)
        local_sum_0 = (
            cute.arch.add_packed_f32x2((init_val, 0.0), (res[0], res[1]))
            # cute.arch.add_packed_f32x2((init_val / 2, init_val / 2), (res[0], res[1]))
            if const_expr(init_val is not None)
            else (res[0], res[1])
        )
        local_sum = [local_sum_0, (res[2], res[3]), (res[4], res[5]), (res[6], res[7])]
        for i in cutlass.range_constexpr(8, cute.size(x.shape), 8):
            local_sum[0] = cute.arch.add_packed_f32x2(
                local_sum[0], (res[i + 0], res[i + 1])
            )
            local_sum[1] = cute.arch.add_packed_f32x2(
                local_sum[1], (res[i + 2], res[i + 3])
            )
            local_sum[2] = cute.arch.add_packed_f32x2(
                local_sum[2], (res[i + 4], res[i + 5])
            )
            local_sum[3] = cute.arch.add_packed_f32x2(
                local_sum[3], (res[i + 6], res[i + 7])
            )
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[1])
        local_sum[2] = cute.arch.add_packed_f32x2(local_sum[2], local_sum[3])
        local_sum[0] = cute.arch.add_packed_f32x2(local_sum[0], local_sum[2])
        return local_sum[0][0] + local_sum[0][1]


sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
)


@dsl_user_op
def atomic_add_fp32(
    a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    result_type = {"res": T.f32()} if _ATOMICRMW_REQUIRES_RESULT_TYPE else {}
    nvvm.atomicrmw(
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(),
        **result_type,
    )


@dsl_user_op
def _atomic_max_fp32_bits(
    a: Float32, gmem_ptr: cute.Pointer, signed: bool, *, loc=None, ip=None
) -> None:
    bits = llvm.bitcast(T.i32(), Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result_type = {"res": T.i32()} if _ATOMICRMW_REQUIRES_RESULT_TYPE else {}
    nvvm.atomicrmw(
        op=nvvm.AtomicOpKind.MAX if signed else nvvm.AtomicOpKind.UMIN,
        ptr=gmem_ptr.llvm_ptr,
        a=bits,
        **result_type,
    )


@cute.jit
def atomic_max_fp32(a: Float32, gmem_ptr: cute.Pointer) -> None:
    """Atomic ``*gmem_ptr = max(*gmem_ptr, a)`` on an fp32 location.

    PTX has no float atomic max. On the IEEE bit pattern a non-negative float
    orders like a signed int and a negative one like a reversed unsigned int,
    so a signed max / unsigned min pair covers both signs as long as the
    location and ``a`` are not NaN (``-inf`` is fine as the initial value).
    """
    if a >= 0.0:
        _atomic_max_fp32_bits(a, gmem_ptr, True)
    else:
        _atomic_max_fp32_bits(a, gmem_ptr, False)


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    """Return a predicate tensor for the "k" dimension of ``tAcA``,
    where ``tAcA`` is a tensor with shape

    ``(CPY_ATOM=(ATOM_V,REST_V),CPY_MN,CPY_K)``.

    NOTE: we only compute predicates for the "k" dimension,
    and use "if" on the mn dimension, to reduce register pressure.
    """

    # tAcA: (CPY_ATOM=(ATOM_V,REST_V),CPY_MN,CPY_K)
    # tApA: (REST_V,CPY_MN,CPY_K):(CPY_K,0,1)
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),  # REST_V
                cute.size(tAcA, mode=[1]),  # CPY_MN
                cute.size(tAcA, mode=[2]),  # CPY_K
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),  # (CPY_K,0,1)
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync):
        warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx


@cute.jit
def shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    # important: need stride 1 and not 0 for recast_tensor to work
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@dsl_user_op
def shl_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """
    Left-shift val by shift bits using PTX shl.b32 (sign-agnostic).

    Named ``shl_u32`` (not ``shl_b32``) because python type annotations
    distinguish signed/unsigned.

    PTX semantics (§9.7.8.8): "Shift amounts greater than the register width N
    are clamped to N."  So ``shl.b32 d, a, 32`` is well-defined and yields 0.

    This differs from C/C++ and LLVM IR, where shifting by >= the type width is
    undefined behavior.  CuTeDSL compiles through MLIR -> LLVM IR, so a plain
    Python-level ``Uint32(x) << Uint32(n)`` inherits LLVM's UB: the optimizer
    may treat the result as poison and eliminate dependent code.  Inline PTX
    bypasses the LLVM IR shift entirely — the instruction is emitted verbatim
    into PTX where clamping makes it safe for all shift amounts.
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shl.b32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def shr_u32(
    val: cutlass.Uint32, shift: cutlass.Uint32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """
    Unsigned right-shift val by shift bits using PTX shr.u32 (zero-fills).

    See ``shl_u32`` docstring for why inline PTX is used instead of plain
    CuTeDSL shift operators (LLVM shift-by-type-width UB).
    """
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Uint32(val).ir_value(loc=loc, ip=ip),
                cutlass.Uint32(shift).ir_value(loc=loc, ip=ip),
            ],
            "shr.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: Int32, lane: Optional[Int32] = None) -> Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, val = %d", cute.arch.thread_idx()[0] % 32, val)  # noqa: E501
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
        # if cute.arch.thread_idx()[0] >= 128 and cute.arch.thread_idx()[0] < 128 + 32 and cute.arch.block_idx()[0] == 0: cute.printf("tidx = %d, partial_sum = %d, val = %d", cute.arch.thread_idx()[0] % 32, partial_sum, val)  # noqa: E501
    return val


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32,
    b: float | Float32,
    to_dtype: Type,
    *,
    loc=None,
    ip=None,
) -> Int32:
    assert to_dtype in [
        cutlass.BFloat16,
        cutlass.Float16,
    ], "to_dtype must be BFloat16 or Float16"
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            f"cvt.rn.{'bf16x2' if to_dtype is cutlass.BFloat16 else 'f16x2'}.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@overload
def cvt_f16(src: cute.Tensor, dst: cute.Tensor) -> None:
    ...


@overload
def cvt_f16(src: cute.Tensor, dtype: Type[cute.Numeric]) -> cute.Tensor:
    ...


@cute.jit
def cvt_f16(src: cute.Tensor, dst_or_dtype):
    """Convert Float32 tensor to Float16/BFloat16.

    Args:
        src: Source tensor with Float32 element type
        dst_or_dtype: Either a destination tensor or a dtype (Float16/BFloat16)

    Returns:
        None if dst is a tensor, or a new tensor if dtype is provided
    """
    if const_expr(isinstance(dst_or_dtype, type)):
        # dtype variant: create new tensor and call the tensor variant
        dtype = dst_or_dtype
        dst = cute.make_rmem_tensor(src.shape, dtype)
        cvt_f16(src, dst)
        return dst
    else:
        # tensor variant: write to dst
        dst = dst_or_dtype
        assert cute.size(dst.shape) == cute.size(
            src.shape
        ), "dst and src must have the same size"
        assert cute.size(src.shape) % 2 == 0, "src must have an even number of elements"
        assert dst.element_type in [
            cutlass.BFloat16,
            cutlass.Float16,
        ], "dst must be BFloat16 or Float16"
        assert src.element_type is Float32, "src must be Float32"
        dst_i32 = cute.recast_tensor(dst, Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)


@dsl_user_op
@cute.jit
def evaluate_polynomial(
    x: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Float32:
    deg = len(poly) - 1
    out = poly[deg]
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = out * x + poly[i]
    return out


@dsl_user_op
@cute.jit
def evaluate_polynomial_2(
    x: Float32, y: Float32, poly: Tuple[Float32, ...], *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    deg = len(poly) - 1
    out = (poly[deg], poly[deg])
    for i in cutlass.range_constexpr(deg - 1, -1, -1):
        out = cute.arch.fma_packed_f32x2(out, (x, y), (poly[i], poly[i]))
    return out


@dsl_user_op
def add_round_down(
    x: float | Float32, y: float | Float32, *, loc=None, ip=None
) -> Float32:
    # There's probably a way to call llvm or nvvm to do this instead of ptx
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Float32(y).ir_value(loc=loc, ip=ip)],
            "add.rm.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def combine_int_frac_ex2(
    x_rounded: Float32, frac_ex2: Float32, *, loc=None, ip=None
) -> Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x_rounded).ir_value(loc=loc, ip=ip),
                Float32(frac_ex2).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t"
            ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\n\t"
            "mov.b32 x_rounded_i, $1;\n\t"
            "mov.b32 frac_ex_i, $2;\n\t"
            "shl.b32 x_rounded_e, x_rounded_i, 23;\n\t"
            # add.u32 generates IMAD instruction and add.s32 generates LEA instruction
            # IMAD uses the FMA pipeline and LEA uses the ALU pipeline, afaik
            "add.s32 out_i, x_rounded_e, frac_ex_i;\n\t" "mov.b32 $0, out_i;\n\t" "}\n",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_emulation(x: Float32, *, poly_degree: int = 3, loc=None, ip=None) -> Float32:
    assert poly_degree in POLY_EX2, f"Polynomial degree {poly_degree} not supported"
    # We assume x <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    x_clamped = cute.arch.fmax(x, -127.0)
    # We want to round down here, so that the fractional part is in [0, 1)
    x_rounded = add_round_down(x_clamped, fp32_round_int, loc=loc, ip=ip)
    # The integer floor of x is now in the last 8 bits of x_rounded
    # We assume the next 2 ops round to nearest even. The rounding mode is important.
    x_rounded_back = x_rounded - fp32_round_int
    x_frac = x_clamped - x_rounded_back
    x_frac_ex2 = evaluate_polynomial(x_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    return combine_int_frac_ex2(x_rounded, x_frac_ex2, loc=loc, ip=ip)


# TODO: check that the ex2_emulation_2 produces the same SASS as the ptx version
@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even. The rounding mode is important.
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out


@dsl_user_op
def e2e_asm2(x: Float32, y: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    out_f32x2 = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(y, loc=loc, ip=ip).ir_value()],
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, $2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, $3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 $0, r7;\n\t"
        "mov.b32 $1, r8;\n\t"
        "}\n",
        "=r,=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    out0 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [0], loc=loc, ip=ip))
    out1 = Float32(llvm.extractvalue(T.f32(), out_f32x2, [1], loc=loc, ip=ip))
    return out0, out1


@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """Convert a scalar to a cute TensorSSA of shape (1,) and given dtype"""
    vec = cute.make_rmem_tensor(1, dtype)
    vec[0] = a
    return vec.load()


def ssa_to_scalar(val):
    """Could inline but nice for reflecting the above api"""
    return val[0]


@cute.jit
def clz(x: Int32) -> Int32:
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


# ---------------------------------------------------------------------------
# Barrier helpers (cross-CTA semaphores via global-memory atomics)
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer,
    thread_idx: int | Int32,
    flag_offset: int,
    val: cutlass.Constexpr[Int32],
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)


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
    """Copy 16 B from global to shared memory at ``dst_smem_addr + dst_imm_offset``, or write
    16 zero bytes when ``row_id < 0``.

    ``row_id < 0`` sets the ``ignore-src`` predicate, so the source is not read and
    ``src_gmem_addr`` may be unmapped. ``dst_imm_offset`` must be a Python int: it is emitted as
    the instruction's immediate. Completion is tracked by the calling thread's ``cp.async``
    groups. ``.cg`` keeps the gather out of L1, since each row is read once per use.

    NOTE: an ``L2::cache_hint`` operand miscompiles under nvidia-cutlass-dsl 4.4.x on sm_103a
    (uninitialized policy register).
    """
    assert isinstance(dst_imm_offset, int) and 0 <= dst_imm_offset < (
        1 << 24
    ), "immediate offset out of range"
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


# ====================================================================
# Shared-memory helpers (from smem_ops)
# ====================================================================


@dsl_user_op
def tensor_atomic_add(
    tensor: cute.Tensor, idx: Int32, val: Int32, *, loc=None, ip=None
) -> Int32:
    """Atomic add val to tensor[idx], return old value. Works for both SMEM and GMEM."""
    ptr = elem_pointer(tensor, idx, loc=loc, ip=ip)
    return cute.arch.atomic_add(ptr.llvm_ptr, val, loc=loc, ip=ip)


@dsl_user_op
def u32_gt(a: Int32, b: Int32, *, loc=None, ip=None) -> Boolean:
    """Unsigned greater-than: a >u b."""
    from cutlass._mlir.dialects import arith as _arith

    return Boolean(
        _arith.CmpIOp(
            _arith.CmpIPredicate.ugt,
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        ).result
    )


@dsl_user_op
def u32_eq(a: Int32, b: Int32, *, loc=None, ip=None) -> Boolean:
    """Unsigned equal: a ==u b."""
    return Boolean(a == b)


# ====================================================================
# Float <-> Ordered uint conversion (DSL)
# ====================================================================


@dsl_user_op
def float_to_ordered(val: Float32, *, loc=None, ip=None) -> Int32:
    from cutlass._mlir.dialects import arith as _arith

    bits = Int32(
        _arith.BitcastOp(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip).result
    )
    thirty_one = _arith.ConstantOp(T.i32(), 31, loc=loc, ip=ip).result
    sign_mask = Int32(
        _arith.ShRSIOp(bits.ir_value(loc=loc, ip=ip), thirty_one, loc=loc, ip=ip).result
    )
    return bits ^ (sign_mask | Int32(0x80000000))


# ====================================================================
# BFloat16 <-> Ordered uint16 conversion (DSL)
# ====================================================================


@dsl_user_op
def bf16_to_ordered16(val: BFloat16, *, loc=None, ip=None) -> Int16:
    from cutlass._mlir.dialects import arith as _arith

    bits = Int16(
        _arith.BitcastOp(T.i16(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip).result
    )
    fifteen = _arith.ConstantOp(T.i16(), 15, loc=loc, ip=ip).result
    sign_mask = Int16(
        _arith.ShRSIOp(bits.ir_value(loc=loc, ip=ip), fifteen, loc=loc, ip=ip).result
    )
    return bits ^ (sign_mask | Int16(-32768))


@dsl_user_op
def int16_zext_to_int32(val: Int16, *, loc=None, ip=None) -> Int32:
    from cutlass._mlir.dialects import arith as _arith

    return Int32(
        _arith.ExtUIOp(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip).result
    )


# ====================================================================
# Vectorized GMEM -> SMEM load helpers
# ====================================================================

VEC_BF16 = 8
VEC_FP32 = 4


@dsl_user_op
def vec_load_bf16_to_ordered16(
    gmem_tensor: cute.Tensor,
    gmem_base: Int32,
    smem_tensor: cute.Tensor,
    smem_base: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """128-bit vectorized load: load 4xi32, bitcast each i32 to 2xbf16, convert to ordered16."""
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith as _arith
    from cutlass._mlir.dialects import vector as _vector

    ptr = elem_pointer(gmem_tensor, gmem_base, loc=loc, ip=ip)
    vec_i32_ty = ir.VectorType.get([4], Int32.mlir_type)
    vec_i32 = cute.arch.load(ptr.llvm_ptr, vec_i32_ty, loc=loc, ip=ip)
    fifteen = _arith.ConstantOp(T.i16(), 15, loc=loc, ip=ip).result
    sign_const = Int16(-32768)
    for j_val in range(4):
        word = _vector.extract(
            vec_i32, dynamic_position=[], static_position=[j_val], loc=loc, ip=ip
        )
        lo_bits = Int16(_arith.TruncIOp(T.i16(), word, loc=loc, ip=ip).result)
        hi_word = Int32(
            _arith.ShRUIOp(
                word,
                _arith.ConstantOp(T.i32(), 16, loc=loc, ip=ip).result,
                loc=loc,
                ip=ip,
            ).result
        )
        hi_bits = Int16(
            _arith.TruncIOp(
                T.i16(), hi_word.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
            ).result
        )
        for sub, bits in enumerate([lo_bits, hi_bits]):
            sign_mask = Int16(
                _arith.ShRSIOp(
                    bits.ir_value(loc=loc, ip=ip), fifteen, loc=loc, ip=ip
                ).result
            )
            ordered = bits ^ (sign_mask | sign_const)
            smem_tensor[smem_base + Int32(j_val * 2 + sub)] = ordered


@dsl_user_op
def vec_load_fp32_to_ordered32(
    gmem_tensor: cute.Tensor,
    gmem_base: Int32,
    smem_tensor: cute.Tensor,
    smem_base: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import arith as _arith
    from cutlass._mlir.dialects import vector as _vector

    ptr = elem_pointer(gmem_tensor, gmem_base, loc=loc, ip=ip)
    vec_ty = ir.VectorType.get([VEC_FP32], Float32.mlir_type)
    vec = cute.arch.load(ptr.llvm_ptr, vec_ty, loc=loc, ip=ip)
    for j_val in range(VEC_FP32):
        elem_ir = _vector.extract(
            vec, dynamic_position=[], static_position=[j_val], loc=loc, ip=ip
        )
        bits = Int32(_arith.BitcastOp(T.i32(), elem_ir, loc=loc, ip=ip).result)
        thirty_one = _arith.ConstantOp(T.i32(), 31, loc=loc, ip=ip).result
        sign_mask = Int32(
            _arith.ShRSIOp(
                bits.ir_value(loc=loc, ip=ip), thirty_one, loc=loc, ip=ip
            ).result
        )
        ordered = bits ^ (sign_mask | Int32(0x80000000))
        smem_tensor[smem_base + Int32(j_val)] = ordered


# ====================================================================
# GMEM synchronization primitives
# ====================================================================
# gmem_ld_acquire / gmem_red_release are prefixed so they do not shadow the
# pointer-based ld_acquire / red_release barrier helpers above.
#
# cute.arch.load / store / atomic_* get the raw LLVM pointer: they normalize a
# cute pointer through `to_llvm_ptr`, which the CUDA 13 build of CuTeDSL does
# not provide.


@dsl_user_op
def gmem_ld_acquire(tensor: cute.Tensor, idx: Int32, *, loc=None, ip=None) -> Int32:
    ptr = elem_pointer(tensor, idx, loc=loc, ip=ip)
    return cute.arch.load(
        ptr.llvm_ptr, Int32, sem="acquire", scope="gpu", loc=loc, ip=ip
    )


@dsl_user_op
def gmem_red_release(
    tensor: cute.Tensor, idx: Int32, val: Int32, *, loc=None, ip=None
) -> Int32:
    cute.arch.fence_acq_rel_gpu(loc=loc, ip=ip)
    ptr = elem_pointer(tensor, idx, loc=loc, ip=ip)
    return cute.arch.atomic_add(
        ptr.llvm_ptr, val, sem="relaxed", scope="gpu", loc=loc, ip=ip
    )


@dsl_user_op
def st_release(
    tensor: cute.Tensor, idx: Int32, val: Int32, *, loc=None, ip=None
) -> None:
    cute.arch.fence_acq_rel_gpu(loc=loc, ip=ip)
    ptr = elem_pointer(tensor, idx, loc=loc, ip=ip)
    cute.arch.store(ptr.llvm_ptr, val, sem="release", scope="gpu", loc=loc, ip=ip)


@dsl_user_op
def gmem_atomic_add_relaxed(
    tensor: cute.Tensor, idx: Int32, val: Int32, *, loc=None, ip=None
) -> Int32:
    ptr = elem_pointer(tensor, idx, loc=loc, ip=ip)
    return cute.arch.atomic_add(
        ptr.llvm_ptr, val, sem="relaxed", scope="gpu", loc=loc, ip=ip
    )


@dsl_user_op
def gmem_store_cs_i32(
    tensor: cute.Tensor, coord: cute.Coord, val: Int32, *, loc=None, ip=None
) -> None:
    """Store Int32 to GMEM with cache-streaming hint (st.cs)."""
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip)
    cute.arch.store(ptr.llvm_ptr, val, cop="cs", loc=loc, ip=ip)
