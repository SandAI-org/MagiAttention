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

"""Fresh-process probe of allocator reconstruction, without an attention kernel.

CUTE_DSL_KEEP_IR must be set before importing Cutlass. This kernel owns exactly
one mbarrier: the allocator's two-CTA deallocation barrier. Inspecting all its
initializers therefore identifies that barrier without guessing by arrival count
among unrelated attention pipelines.
"""

import json
from collections.abc import Iterator

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cuda.bindings import driver as cuda
from cutlass._mlir import ir
from cutlass.cute.runtime import from_dlpack

import magi_attention
from magi_attention.kernel.cutedsl.dsl_compat import SmemAllocator, TmemAllocator

RESULT_PREFIX = "TMEM_DEALLOC_MBARRIER_RESULT "
_MBARRIER_INIT_OPS = frozenset({"nvvm.mbarrier.init", "nvvm.mbarrier.init.shared"})


class TmemAllocatorProbe:
    @cute.jit
    def __call__(self, done: cute.Tensor, stream: cuda.CUstream):
        self.kernel(done).launch(
            grid=(2, 1, 1), block=(32, 1, 1), cluster=(2, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, done: cute.Tensor):
        smem = SmemAllocator()
        result_ptr = smem.allocate(cutlass.Int32, byte_alignment=8)
        dealloc_ptr = smem.allocate(cutlass.Int64, byte_alignment=8)
        allocator = TmemAllocator(
            alloc_result_dst_smem_ptr=result_ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=32),
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=dealloc_ptr,
        )
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # A dynamic warp-role region rebuilds the allocator, exactly the
        # operation that used to reinitialize its live deallocation barrier.
        if warp == 0:
            allocator.allocate(32)
            allocator.wait_for_alloc()
            ptr = allocator.retrieve_ptr(cutlass.Float32)
            # Deliberately no work tile: both CTAs go straight to deallocation.
            allocator.relinquish_alloc_permit()
            allocator.free(ptr)
        if cute.arch.thread_idx()[0] == 0:
            done[cute.arch.block_idx()[0]] = 1


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
            if operation.name in _MBARRIER_INIT_OPS:
                constant = operation.operands[1].owner
                assert constant.name == "arith.constant", constant.name
                arrival_counts.append(
                    ir.IntegerAttr(constant.attributes["value"]).value
                )
    return arrival_counts


def main() -> None:
    done = torch.zeros(2, device="cuda", dtype=torch.int32)
    done_cute = from_dlpack(done)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    # Compile this specific probe directly; no attention wrapper/cache internals.
    compiled = cute.compile(TmemAllocatorProbe(), done_cute, stream)
    counts = _mbarrier_init_arrival_counts(compiled.__mlir__)
    # Fail before launching known-bad IR, so a repeated init need not hang a GPU.
    assert counts == [32], f"expected one deallocation barrier init, got {counts}"
    compiled(done_cute, stream)
    torch.cuda.synchronize()
    result = {
        "magi_attention_file": magi_attention.__file__,
        "cutlass_version": cutlass.__version__,
        "arrival_counts": counts,
        "completed_ctas": done.tolist(),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
