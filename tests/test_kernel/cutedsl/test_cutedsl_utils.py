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

from unittest import TestCase

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.cutedsl_utils import atomic_add_fp32, fmax
from magi_attention.testing import parameterize


@cute.kernel
def _fmax_kernel(
    a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_c: cutlass.Constexpr
):
    tidx, _, _ = cute.arch.thread_idx()
    out[tidx] = fmax(a[tidx], b[tidx], 0.5 if use_c else None)


@cute.jit
def _fmax(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, use_c: cutlass.Constexpr):
    _fmax_kernel(a, b, out, use_c).launch(grid=(1, 1, 1), block=(32, 1, 1))


def _check_fmax(use_c: bool, enable_tvm_ffi: bool) -> None:
    a = torch.linspace(-2, 2, 32, device="cuda")
    b = a.flip(0)
    out = torch.empty_like(a)
    tensors = [from_dlpack(t, enable_tvm_ffi=enable_tvm_ffi) for t in (a, b, out)]
    options = "--enable-tvm-ffi" if enable_tvm_ffi else ""
    compiled = cute.compile(_fmax, *tensors, use_c, options=options)
    compiled(*((a, b, out) if enable_tvm_ffi else tensors))
    expected = torch.maximum(a, b)
    if use_c:
        expected = expected.clamp_min(0.5)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@cute.kernel
def _atomic_add_kernel(a: cute.Tensor, out: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    atomic_add_fp32(a[tidx], out.iterator)


@cute.jit
def _atomic_add(a: cute.Tensor, out: cute.Tensor):
    _atomic_add_kernel(a, out).launch(grid=(1, 1, 1), block=(32, 1, 1))


class TestCutedslUtils(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")

    @parameterize("enable_tvm_ffi", [False, True])
    def test_fmax(self, enable_tvm_ffi):
        _check_fmax(use_c=False, enable_tvm_ffi=enable_tvm_ffi)

    # Separate method: a skipTest inside parameterize ends all remaining cases.
    @parameterize("enable_tvm_ffi", [False, True])
    def test_fmax_three_inputs(self, enable_tvm_ffi):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("Three-input fmax requires SM100 or newer")
        _check_fmax(use_c=True, enable_tvm_ffi=enable_tvm_ffi)

    @parameterize("enable_tvm_ffi", [False, True])
    def test_atomic_add_fp32(self, enable_tvm_ffi):
        a = torch.arange(32, device="cuda", dtype=torch.float32)
        out = torch.zeros(1, device="cuda")
        tensors = [from_dlpack(t, enable_tvm_ffi=enable_tvm_ffi) for t in (a, out)]
        options = "--enable-tvm-ffi" if enable_tvm_ffi else ""
        compiled = cute.compile(_atomic_add, *tensors, options=options)
        compiled(*((a, out) if enable_tvm_ffi else tensors))
        torch.testing.assert_close(out[0], a.sum(), rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
