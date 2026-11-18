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

import math
import re
import tempfile
import unittest
from unittest import TestCase

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import NamedBarrier

from magi_attention.kernel.cutedsl.cutedsl_utils import (
    atomic_add_fp32,
    atomic_max_fp32,
    fmax,
)
from magi_attention.kernel.cutedsl.dsl_compat import SmemAllocator, TmemAllocator
from magi_attention.testing import parameterize
from magi_attention.testing.precision import assert_equal
from magi_attention.utils.arch import is_sm


@cute.kernel
def _fmax_kernel(
    a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_c: cutlass.Constexpr
):
    tidx, _, _ = cute.arch.thread_idx()
    out[tidx] = fmax(a[tidx], b[tidx], 0.5 if use_c else None)


@cute.jit
def _fmax(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_c: cutlass.Constexpr):
    _fmax_kernel(a, b, out, use_c).launch(grid=(1, 1, 1), block=(32, 1, 1))


def _check_fmax(use_c: bool) -> None:
    a = torch.linspace(-2, 2, 32, device="cuda")
    b = a.flip(0)
    out = torch.empty_like(a)
    tensors = [from_dlpack(t, enable_tvm_ffi=True) for t in (a, b, out)]
    compiled = cute.compile(_fmax, *tensors, use_c, options="--enable-tvm-ffi")
    compiled(a, b, out)
    expected = torch.maximum(a, b)
    if use_c:
        expected = expected.clamp_min(0.5)
    assert_equal(out, expected, test_case=f"_check_fmax(use_c={use_c})")


@cute.kernel
def _atomic_add_kernel(a: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    atomic_add_fp32(a[tidx], out.iterator)


@cute.jit
def _atomic_add(a: cute.Tensor, out: cute.Tensor):
    _atomic_add_kernel(a, out).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.kernel
def _atomic_max_kernel(a: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    atomic_max_fp32(a[tidx], out.iterator)


@cute.jit
def _atomic_max(a: cute.Tensor, out: cute.Tensor):
    _atomic_max_kernel(a, out).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def _rebuild_two_cta_allocator():
    _rebuild_two_cta_allocator_kernel().launch(
        grid=(2, 1, 1), block=(32, 1, 1), cluster=(2, 1, 1)
    )


@cute.kernel
def _rebuild_two_cta_allocator_kernel():
    smem = SmemAllocator()
    allocator = TmemAllocator(
        alloc_result_dst_smem_ptr=smem.allocate(cutlass.Int32, byte_alignment=8),
        barrier_for_retrieve=NamedBarrier(barrier_id=1, num_threads=32),
        allocator_warp_id=0,
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=smem.allocate(cutlass.Int64, byte_alignment=8),
    )
    if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:
        # Carry the allocator through a dynamic region, forcing reconstruction.
        allocator.allocate(32)
        allocator.wait_for_alloc()
        ptr = allocator.retrieve_ptr(cutlass.Float32)
        allocator.relinquish_alloc_permit()
        allocator.free(ptr)


class TestCutedslUtils(TestCase):
    """Device helpers the kernels call, compiled the way production compiles them."""

    @unittest.skipUnless(is_sm(100, 103), "requires SM100/103")
    def test_two_cta_allocator_rebuild_does_not_reinitialize_barrier(self):
        # DSL < 4.5 reconstructed via __init__, resetting a live barrier.
        # Inspect PTX without launching this potentially deadlocking kernel.
        with tempfile.TemporaryDirectory() as dump_dir:
            compiled = cute.compile(
                _rebuild_two_cta_allocator,
                options=f"--keep-ptx --dump-dir={dump_dir}",
            )
            ptx = compiled.__ptx__
            self.assertIsNotNone(ptx)
            inits = re.findall(r"mbarrier\.init[^;]*;", ptx)
            self.assertEqual(len(inits), 1, inits)

    def test_fmax(self):
        _check_fmax(use_c=False)

    # Separate method: a skipTest here must not skip the other cases.
    def test_fmax_three_inputs(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("Three-input fmax requires SM100 or newer")
        _check_fmax(use_c=True)

    def test_atomic_add_fp32(self):
        a = torch.arange(32, device="cuda", dtype=torch.float32)
        out = torch.zeros(1, device="cuda")
        tensors = [from_dlpack(t, enable_tvm_ffi=True) for t in (a, out)]
        compiled = cute.compile(_atomic_add, *tensors, options="--enable-tvm-ffi")
        compiled(a, out)
        assert_equal(out[0], a.sum(), test_case="test_atomic_add_fp32")

    @parameterize("bounds", [(-2.0, 2.0), (-3.0, -1.0)])
    def test_atomic_max_fp32(self, bounds: tuple[float, float]):
        a = torch.linspace(*bounds, 32, device="cuda")
        out = torch.full((1,), -math.inf, device="cuda")
        tensors = [from_dlpack(t, enable_tvm_ffi=True) for t in (a, out)]
        compiled = cute.compile(_atomic_max, *tensors, options="--enable-tvm-ffi")
        compiled(a, out)
        assert_equal(out[0], a.max(), test_case=f"test_atomic_max_fp32({bounds})")


if __name__ == "__main__":
    unittest.main()
