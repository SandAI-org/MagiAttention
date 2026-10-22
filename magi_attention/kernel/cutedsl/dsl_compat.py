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

# Bind non-deprecated APIs at import time for each supported DSL version.

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic

from magi_attention.utils.version import is_cutedsl_version_ge

# Before DSL 4.5, tcgen05 and warpgroup enum members compare unequal.
if is_cutedsl_version_ge("4.5.0"):
    from cutlass.cute.nvgpu import OperandMajorMode as Tcgen05OperandMajorMode
    from cutlass.cute.nvgpu import OperandMajorMode as WarpgroupOperandMajorMode
else:
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode as Tcgen05OperandMajorMode
    from cutlass.cute.nvgpu.warpgroup import (
        OperandMajorMode as WarpgroupOperandMajorMode,
    )

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
    """Build a tcgen05 MMA; DSL < 4.5 requires equal operand dtypes."""
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


# V2 uses two MLIR values; custom reconstruction must preserve both.
if is_cutedsl_version_ge("4.6.1"):
    from cutlass.cute import FastDivmodDivisorV2 as FastDivmodDivisor
else:
    from cutlass.cute import FastDivmodDivisor

# DSL 4.8 deprecates these cutlass.utils exports.
if is_cutedsl_version_ge("4.8.0"):
    from cutlass.tensor_utils import LayoutEnum
else:
    from cutlass.utils import LayoutEnum

if is_cutedsl_version_ge("4.8.0"):
    from cutlass.memory import SmemAllocator
else:
    from cutlass.utils import SmemAllocator

if is_cutedsl_version_ge("4.8.0"):
    from cutlass.memory import TmemAllocator
else:
    from cutlass.utils import TmemAllocator

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
