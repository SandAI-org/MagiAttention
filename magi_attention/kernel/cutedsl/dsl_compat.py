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

# CuTeDSL APIs whose current name does not exist in every supported
# nvidia-cutlass-dsl release.
#
# Each binding selects, by the installed DSL version, the current name where
# it exists and the deprecated name otherwise, so the kernels emit no DSL
# DeprecationWarning on a supported release. The kernels import these names
# from here instead of from cutlass. Every version check runs once, when this
# module is imported, and never inside a `@cute.jit` function.

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from magi_attention.utils.version import is_cutedsl_version_ge

# tcgen05.OperandMajorMode and warpgroup.OperandMajorMode are deprecated since
# 4.5.0, which adds the shared cute.nvgpu.OperandMajorMode; remove the branch
# when 4.4.2 support is dropped. Below 4.5.0 the two old enums are distinct and
# their members compare unequal, so each MMA family needs its own name.
if is_cutedsl_version_ge("4.5.0"):
    from cutlass.cute.nvgpu import OperandMajorMode as Tcgen05OperandMajorMode
    from cutlass.cute.nvgpu import OperandMajorMode as WarpgroupOperandMajorMode
else:
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode as Tcgen05OperandMajorMode
    from cutlass.cute.nvgpu.warpgroup import (
        OperandMajorMode as WarpgroupOperandMajorMode,
    )

# The ab_dtype form of the tcgen05 make_trivial_tiled_mma is deprecated since
# 4.5.0, which adds the form with separate a_dtype and b_dtype; remove the
# branch when 4.4.2 support is dropped.
_TCGEN05_MMA_HAS_SEPARATE_AB_DTYPES = is_cutedsl_version_ge("4.5.0")


def make_trivial_tiled_mma(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    a_leading_mode: Tcgen05OperandMajorMode,
    b_leading_mode: Tcgen05OperandMajorMode,
    acc_dtype: Type[cutlass.Numeric],
    cta_group: tcgen05.CtaGroup,
    mma_tiler_mn: tuple[int, int],
    a_source: tcgen05.OperandSource = tcgen05.OperandSource.SMEM,
) -> cute.TiledMma:
    """Build a tcgen05 tiled MMA from
    ``cutlass.utils.blackwell_helpers.make_trivial_tiled_mma``.

    Below nvidia-cutlass-dsl 4.5.0 the DSL takes one dtype for both operands,
    so ``a_dtype`` must equal ``b_dtype``; an unequal pair fails an assertion.
    """
    if _TCGEN05_MMA_HAS_SEPARATE_AB_DTYPES:
        return sm100_utils_basic.make_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_leading_mode,
            b_leading_mode,
            acc_dtype,
            cta_group,
            mma_tiler_mn,
            a_source,
        )
    assert a_dtype == b_dtype, (
        "nvidia-cutlass-dsl < 4.5.0 builds a tcgen05 MMA with one operand "
        f"dtype; got a_dtype={a_dtype}, b_dtype={b_dtype}"
    )
    return sm100_utils_basic.make_trivial_tiled_mma(
        a_dtype,
        a_leading_mode,
        b_leading_mode,
        acc_dtype,
        cta_group,
        mma_tiler_mn,
        a_source,
    )


# cute.FastDivmodDivisor is deprecated since 4.6.1, which adds
# cute.FastDivmodDivisorV2; remove the branch when 4.4.2 support is dropped.
# V2 crosses a kernel boundary as two MLIR values instead of one, so a
# hand-written __new_from_mlir_values__ must not assume one value per divisor.
if is_cutedsl_version_ge("4.6.1"):
    from cutlass.cute import FastDivmodDivisorV2 as FastDivmodDivisor
else:
    from cutlass.cute import FastDivmodDivisor

# cutlass.utils.LayoutEnum is deprecated since 4.8.0, which adds
# cutlass.tensor_utils.LayoutEnum; remove the branch when 4.7.0 support is
# dropped.
if is_cutedsl_version_ge("4.8.0"):
    from cutlass.tensor_utils import LayoutEnum
else:
    from cutlass.utils import LayoutEnum

# cutlass.utils.SmemAllocator is deprecated since 4.8.0, which adds
# cutlass.memory.SmemAllocator; remove the branch when 4.7.0 support is
# dropped.
if is_cutedsl_version_ge("4.8.0"):
    from cutlass.memory import SmemAllocator
else:
    from cutlass.utils import SmemAllocator

# cutlass.utils.TmemAllocator is deprecated since 4.8.0, which adds
# cutlass.memory.TmemAllocator; remove the branch when 4.7.0 support is
# dropped.
if is_cutedsl_version_ge("4.8.0"):
    from cutlass.memory import TmemAllocator
else:
    from cutlass.utils import TmemAllocator

# cutlass.utils.get_smem_capacity_in_bytes is deprecated since 4.8.0, which
# adds cutlass.memory.get_smem_capacity_in_bytes; remove the branch when 4.7.0
# support is dropped.
if is_cutedsl_version_ge("4.8.0"):
    from cutlass.memory import get_smem_capacity_in_bytes
else:
    from cutlass.utils import get_smem_capacity_in_bytes

__all__ = [
    "FastDivmodDivisor",
    "LayoutEnum",
    "SmemAllocator",
    "Tcgen05OperandMajorMode",
    "TmemAllocator",
    "WarpgroupOperandMajorMode",
    "get_smem_capacity_in_bytes",
    "make_trivial_tiled_mma",
]
