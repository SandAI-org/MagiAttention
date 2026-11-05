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

"""Correctness tests for the CuTe DSL radix-select top-k kernel.

The kernel is verified against ``torch.topk``: values gathered by the returned
indices are sorted on both sides and compared with ``assert_close``. Index
identity, including tie-breaking, is checked with ``assert_equal``.

Run:
    pytest tests/test_kernel/cutedsl/test_topk.py -v
"""

from itertools import product
from unittest import skipUnless

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.cutedsl.topk import cutedsl_topk, cutedsl_topk_available
from magi_attention.testing import parameterize
from magi_attention.testing.precision import assert_close, assert_equal

_shapes = [
    # (batch, n, k)
    (1, 64, 8),
    (1, 256, 32),
    (2, 1024, 64),
    (4, 4096, 256),
    (8, 8192, 512),
    (4, 32000, 256),
    (2, 65536, 1024),
    (1, 100000, 5000),
    # Multi-CTA rows (n > SMEM capacity), including batches larger than the
    # number of CTA groups so each group loops over several rows.
    (2, 300000, 30000),
    (40, 300000, 3072),
    # Row lengths that are not a multiple of the 16-byte vector width.
    (3, 7, 3),
    (2, 1001, 100),
    (2, 300001, 3000),
]

_dtypes = [torch.float32, torch.bfloat16]

_cases = list(product(_shapes, _dtypes))

_tie_cases = list(product([(4, 4096, 256), (3, 300000, 30000)], _dtypes))


def _verify_topk_indices(x: torch.Tensor, k: int, indices: torch.Tensor) -> None:
    """Assert that *indices* identify the correct top-k elements of *x*."""
    ref_values = torch.topk(x, k, dim=-1, sorted=True).values.float()
    gathered = (
        torch.gather(x.float(), -1, indices.long()).sort(dim=-1, descending=True).values
    )
    assert_close(
        gathered,
        ref_values,
        atol=1e-5,
        rtol=1e-5,
        test_case="cutedsl_topk values",
    )


def _lowest_index_topk_ref(x: torch.Tensor, k: int) -> torch.Tensor:
    """Ascending top-k indices where ties at the k-th value keep the lowest indices."""
    kth = torch.topk(x, k, dim=-1).values[..., -1:]
    gt = x > kth
    eq = x == kth
    need_eq = k - gt.sum(dim=-1, keepdim=True)
    selected = gt | (eq & (eq.cumsum(dim=-1) <= need_eq))
    return selected.nonzero()[:, -1].view(*x.shape[:-1], k)


@skipUnless(torch.cuda.is_available(), "CUDA required")
@skipUnless(
    torch.cuda.is_available() and cutedsl_topk_available(),
    "cutedsl topk not available",
)
class TestCuteDSLTopK(TestCase):
    @parameterize("case", _cases)
    def test_cutedsl_topk(self, case):
        (batch, n, k), dtype = case

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(batch, n, dtype=dtype, device="cuda", generator=generator)

        indices = cutedsl_topk(x, k, sorted=False)
        _verify_topk_indices(x, k, indices)
        self.assertTrue(
            bool((indices[:, 1:] > indices[:, :-1]).all()),
            "rows not strictly ascending",
        )
        assert_equal(indices, cutedsl_topk(x, k), test_case="cutedsl_topk default args")
        indices_i32 = cutedsl_topk(x, k, out_dtype=torch.int32)
        self.assertEqual(indices_i32.dtype, torch.int32)
        assert_equal(
            indices_i32.long(),
            indices,
            test_case="cutedsl_topk int32 indices",
        )

    @parameterize("case", _tie_cases)
    def test_cutedsl_topk_ties(self, case):
        """Heavy ties at the pivot: the kernel must keep the lowest tied indices."""
        (batch, n, k), dtype = case

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randint(0, 16, (batch, n), device="cuda", generator=generator).to(
            dtype
        )

        indices = cutedsl_topk(x, k)
        assert_equal(
            indices,
            _lowest_index_topk_ref(x, k),
            test_case="cutedsl_topk lowest-index ties",
        )

    @parameterize("dtype", _dtypes)
    def test_cutedsl_topk_misaligned_base(self, dtype):
        """A contiguous view that starts one element into its storage."""
        batch, n, k = 4, 4096, 256

        generator = torch.Generator(device="cuda").manual_seed(42)
        storage = torch.randn(
            batch * n + 1, dtype=dtype, device="cuda", generator=generator
        )
        x = storage[1:].view(batch, n)
        self.assertTrue(x.data_ptr() % 16)

        _verify_topk_indices(x, k, cutedsl_topk(x, k))

    def test_cutedsl_topk_rejects_sorted(self):
        x = torch.randn(2, 1024, device="cuda")
        with self.assertRaises(ValueError):
            cutedsl_topk(x, 8, sorted=True)


if __name__ == "__main__":
    run_tests()
