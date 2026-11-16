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

# The DSL keeps the traced IR only when CUTE_DSL_KEEP_IR is set before cutlass
# is imported, so this runs as a script in a fresh process.
# Usage: python tmem_dealloc_mbarrier_test_utils.py

from __future__ import annotations

import json
from collections.abc import Iterator

import cutlass
import torch
from cutlass._mlir import ir

import magi_attention
from magi_attention.kernel.cutedsl import flex_flash_attn

RESULT_PREFIX = "TMEM_DEALLOC_MBARRIER_RESULT "

# The op name differs across the supported DSL versions.
_MBARRIER_INIT_OPS = frozenset({"nvvm.mbarrier.init", "nvvm.mbarrier.init.shared"})


def _operations(operation: ir.Operation) -> Iterator[ir.Operation]:
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _operations(child.operation)


def _mbarrier_init_arrival_counts(mlir: str) -> list[int]:
    arrival_counts = []
    with ir.Context():
        module = ir.Module.parse(mlir)
        for operation in _operations(module.operation):
            if operation.name not in _MBARRIER_INIT_OPS:
                continue
            constant = operation.operands[1].owner
            assert constant.name == "arith.constant", constant.name
            arrival_counts.append(ir.IntegerAttr(constant.attributes["value"]).value)
    return arrival_counts


def _trace_bwd() -> str:
    """The backward kernel is 2-CTA at head dim 128 unless 2-CTA is disabled."""
    torch.manual_seed(0)
    seqlen, num_heads, head_dim = 1024, 2, 128
    q, k, v = (
        torch.randn(
            (1, seqlen, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    )
    out, _ = flex_flash_attn.flex_flash_attn_func(q, k, v, mask_types=0)
    torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    torch.cuda.synchronize()

    compile_cache = flex_flash_attn._flex_flash_attn_bwd.compile_cache  # type: ignore[attr-defined]
    (compiled,) = compile_cache.cache.values()
    return compiled.__mlir__


def main() -> None:
    result = {
        "magi_attention_file": magi_attention.__file__,
        "cutlass_version": cutlass.__version__,
        "arrival_counts": _mbarrier_init_arrival_counts(_trace_bwd()),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
