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

import torch

__version__ = "1.0.0"


def fp_dtype_bits(
    dtype: torch.dtype,
) -> int:
    if dtype == torch.float4_e2m1fn_x2:
        # NOTE: _x2 suffix for packed representation
        # of two float4_e2m1f values into one byte
        # see issue: https://github.com/pytorch/pytorch/issues/146414
        return 4

    return torch.finfo(dtype).bits


def fp_dtype_eq(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) == fp_dtype_bits(dtype2)


def fp_dtype_ne(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) != fp_dtype_bits(dtype2)


def fp_dtype_lt(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) < fp_dtype_bits(dtype2)


def fp_dtype_le(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) <= fp_dtype_bits(dtype2)


def fp_dtype_gt(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) > fp_dtype_bits(dtype2)


def fp_dtype_ge(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
) -> bool:
    return fp_dtype_bits(dtype1) >= fp_dtype_bits(dtype2)


def max_fp_dtype(
    *dtypes: torch.dtype,
) -> torch.dtype:
    return max(dtypes, key=lambda dtype: fp_dtype_bits(dtype))


def min_fp_dtype(
    *dtypes: torch.dtype,
) -> torch.dtype:
    return min(dtypes, key=lambda dtype: fp_dtype_bits(dtype))


def to_higher_fp_dtype(
    tensor: torch.Tensor,
    lowest_precision: torch.dtype,
) -> torch.Tensor:
    if fp_dtype_lt(tensor.dtype, lowest_precision):
        return tensor.to(lowest_precision)
    return tensor


def to_triton_dtype(dtype: torch.dtype):
    """Map a :class:`torch.dtype` to its corresponding Triton language dtype.

    Args:
        dtype (torch.dtype): the torch dtype to convert. Supports floating-point
            (fp64/fp32/fp16/bf16/fp8-e4m3/fp8-e5m2), signed/unsigned integer
            (int64..int8, uint64..uint8) and boolean dtypes.

    Returns:
        triton.language.dtype: the equivalent Triton dtype.

    Raises:
        KeyError: if ``dtype`` is not in the supported mapping.
    """
    import triton.language as tl

    return {
        # float
        torch.float64: tl.float64,
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e4m3fn: tl.float8e4nv,
        torch.float8_e5m2: tl.float8e5,
        # int
        torch.int64: tl.int64,
        torch.int32: tl.int32,
        torch.int16: tl.int16,
        torch.int8: tl.int8,
        # uint
        torch.uint64: tl.uint64,
        torch.uint32: tl.uint32,
        torch.uint16: tl.uint16,
        torch.uint8: tl.uint8,
        # bool
        torch.bool: tl.int1,
    }[dtype]


def to_cute_dtype(dtype: torch.dtype):
    """Map a :class:`torch.dtype` to its corresponding CUTLASS (CuTe) dtype.

    Args:
        dtype (torch.dtype): the torch dtype to convert. Supports floating-point
            (fp64/fp32/fp16/bf16/fp8-e4m3/fp8-e5m2), signed/unsigned integer
            (int64..int8, uint64..uint8) and boolean dtypes.

    Returns:
        type: the equivalent ``cutlass`` dtype class.

    Raises:
        KeyError: if ``dtype`` is not in the supported mapping.
    """
    import cutlass

    return {
        # float
        torch.float64: cutlass.Float64,
        torch.float32: cutlass.Float32,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
        # int
        torch.int64: cutlass.Int64,
        torch.int32: cutlass.Int32,
        torch.int16: cutlass.Int16,
        torch.int8: cutlass.Int8,
        # uint
        torch.uint64: cutlass.Uint64,
        torch.uint32: cutlass.Uint32,
        torch.uint16: cutlass.Uint16,
        torch.uint8: cutlass.Uint8,
        # bool
        torch.bool: cutlass.Boolean,
    }[dtype]


def to_tilelang_dtype(dtype: torch.dtype):
    """
    Map a :class:`torch.dtype` to its corresponding TileLang dtype.

    Args:
        dtype (torch.dtype): the torch dtype to convert. Supports floating-point
            (fp64/fp32/fp16/bf16/fp8-e4m3/fp8-e5m2), signed/unsigned integer
            (int64..int8, uint64..uint8) and boolean dtypes.

    Returns:
        type: the equivalent TileLang dtype class.

    Raises:
        KeyError: if ``dtype`` is not in the supported mapping.
    """
    import tilelang.language as T

    return {
        # float
        torch.float64: T.float64,
        torch.float32: T.float32,
        torch.float16: T.float16,
        torch.bfloat16: T.bfloat16,
        torch.float8_e4m3fn: T.float8_e4m3fn,
        torch.float8_e5m2: T.float8_e5m2,
        # int
        torch.int64: T.int64,
        torch.int32: T.int32,
        torch.int16: T.int16,
        torch.int8: T.int8,
        # uint
        torch.uint64: T.uint64,
        torch.uint32: T.uint32,
        torch.uint16: T.uint16,
        torch.uint8: T.uint8,
        # bool
        torch.bool: T.bool,
    }[dtype]
