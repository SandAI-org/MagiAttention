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

from typing import TYPE_CHECKING, SupportsInt

import torch

from magi_attention.common.enum import (
    MagiAttentionKernelBackend,
    MagiAttentionPrecision,
)

if TYPE_CHECKING:
    from magi_attention.magi_attn_ext import KernelBarrier

_NativeKernelBarrier: type["KernelBarrier"] | None
try:
    from magi_attention import magi_attn_ext
except ImportError:
    _NativeKernelBarrier = None
else:
    _NativeKernelBarrier = magi_attn_ext.KernelBarrier


def create_kernel_barrier(target: SupportsInt) -> "KernelBarrier":
    """Construct a native barrier only when the extension is installed."""
    if _NativeKernelBarrier is None:
        raise ImportError("magi_attn_ext is required to create a kernel barrier")
    return _NativeKernelBarrier(target)


# --- backend x precision compatibility matrix ---
# FFA / FA4: only support bf16 / fp16 (hardware kernels)
# SDPA / SDPA_OL: support bf16 / fp16 / fp32 / fp64 (pure-torch)
_BACKEND_SUPPORTED_PRECISIONS: dict[
    MagiAttentionKernelBackend, set[MagiAttentionPrecision]
] = {
    MagiAttentionKernelBackend.FFA: {
        MagiAttentionPrecision.BF16,
        MagiAttentionPrecision.FP16,
    },
    MagiAttentionKernelBackend.FA4: {
        MagiAttentionPrecision.BF16,
        MagiAttentionPrecision.FP16,
    },
    MagiAttentionKernelBackend.SDPA: {
        MagiAttentionPrecision.BF16,
        MagiAttentionPrecision.FP16,
        MagiAttentionPrecision.FP32,
        MagiAttentionPrecision.FP64,
    },
    MagiAttentionKernelBackend.SDPA_OL: {
        MagiAttentionPrecision.BF16,
        MagiAttentionPrecision.FP16,
        MagiAttentionPrecision.FP32,
        MagiAttentionPrecision.FP64,
    },
    MagiAttentionKernelBackend.CUTEDSL: {
        MagiAttentionPrecision.BF16,
        MagiAttentionPrecision.FP16,
    },
}


def _validate_backend_precision(
    backend: MagiAttentionKernelBackend,
    precision: MagiAttentionPrecision | None,
    input_dtype: torch.dtype,
) -> None:
    """Validate the (backend, precision, input_dtype) combination at the
    entrance of dist_attn, raising early and clearly on illegal combos."""

    if precision is not None:
        supported = _BACKEND_SUPPORTED_PRECISIONS[backend]
        assert precision in supported, (
            f"MAGI_ATTENTION_PRECISION={precision.value} is not supported by "
            f"kernel backend {backend.value}. "
            f"Supported precisions: {sorted(p.value for p in supported)}"
        )
    else:
        _DTYPE_TO_PRECISION = {
            torch.bfloat16: MagiAttentionPrecision.BF16,
            torch.float16: MagiAttentionPrecision.FP16,
            torch.float32: MagiAttentionPrecision.FP32,
            torch.float64: MagiAttentionPrecision.FP64,
        }
        inferred = _DTYPE_TO_PRECISION.get(input_dtype)
        if inferred is not None:
            supported = _BACKEND_SUPPORTED_PRECISIONS[backend]
            assert inferred in supported, (
                f"Input dtype {input_dtype} is not supported by "
                f"kernel backend {backend.value}. "
                f"Supported precisions: {sorted(p.value for p in supported)}. "
                f"Set MAGI_ATTENTION_PRECISION to override."
            )


def _prepare_attn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    backend: MagiAttentionKernelBackend,
    precision: MagiAttentionPrecision | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    """Validate compute precision and adapt Q/K/V without changing global state."""
    output_dtype = q.dtype
    _validate_backend_precision(backend, precision, output_dtype)
    if precision is not None:
        compute_dtype = precision.to_torch_dtype()
        if compute_dtype != output_dtype:
            q, k, v = (tensor.to(compute_dtype) for tensor in (q, k, v))
    return q, k, v, output_dtype


def _restore_attn_output(out: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Restore the caller's output dtype while preserving the gradient path."""
    return out.to(output_dtype) if out.dtype != output_dtype else out
