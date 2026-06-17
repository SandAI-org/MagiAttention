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

import math
from functools import lru_cache
from typing import Callable, Tuple, Type, overload

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, const_expr
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import NumericMeta, T, dsl_user_op

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))

load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


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
    thr_mma: cute.core.ThrMma,
    swapAB: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    if const_expr(swapAB):
        return mma_make_fragment_B(smem, thr_mma)
    else:
        return thr_mma.make_fragment_A(thr_mma.partition_A(smem))


def mma_make_fragment_B(
    smem: cute.Tensor,
    thr_mma: cute.core.ThrMma,
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
def atomic_add_fp32(
    a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(),
    )


@dsl_user_op
def elem_pointer(
    x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None
) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(
                tAcA[(0, rest_v), 0, rest_k][1], limit
            )
    return tApA


def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
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
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@dsl_user_op
def cvt_f16x2_f32(
    a: float | Float32, b: float | Float32, to_dtype: Type, *, loc=None, ip=None
) -> cutlass.Int32:
    assert to_dtype in [
        cutlass.BFloat16,
        cutlass.Float16,
    ], "to_dtype must be BFloat16 or Float16"
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
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
        dst = cute.make_fragment(src.shape, dtype)
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
        dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
        assert cute.size(dst_i32.shape) * 2 == cute.size(src.shape)
        for i in cutlass.range_constexpr(cute.size(dst_i32)):
            dst_i32[i] = cvt_f16x2_f32(src[2 * i], src[2 * i + 1], dst.element_type)
