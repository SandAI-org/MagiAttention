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
indices are sorted on both sides and compared with ``torch.testing.assert_close``.

Run:
    pytest tests/test_kernel/cutedsl/test_topk.py -v
"""

from itertools import product

import pytest
import torch

from magi_attention.kernel.cutedsl.topk import cutedsl_topk, cutedsl_topk_available

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not cutedsl_topk_available(),
        reason="cutedsl topk not available",
    ),
]

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


def _case_id(case) -> str:
    (b, n, k), dtype = case
    return f"b{b}_n{n}_k{k}_{str(dtype).split('.')[-1]}"


def _verify_topk_indices(x: torch.Tensor, k: int, indices: torch.Tensor) -> None:
    """Assert that *indices* identify the correct top-k elements of *x*."""
    ref_values = torch.topk(x, k, dim=-1, sorted=True).values.float()
    gathered = (
        torch.gather(x.float(), -1, indices.long()).sort(dim=-1, descending=True).values
    )
    torch.testing.assert_close(gathered, ref_values, atol=1e-5, rtol=1e-5)


def _lowest_index_topk_ref(x: torch.Tensor, k: int) -> torch.Tensor:
    """Ascending top-k indices where ties at the k-th value keep the lowest indices."""
    kth = torch.topk(x, k, dim=-1).values[..., -1:]
    gt = x > kth
    eq = x == kth
    need_eq = k - gt.sum(dim=-1, keepdim=True)
    selected = gt | (eq & (eq.cumsum(dim=-1) <= need_eq))
    return selected.nonzero()[:, -1].view(*x.shape[:-1], k)


@pytest.mark.parametrize("case", _cases, ids=[_case_id(c) for c in _cases])
def test_cutedsl_topk(case):
    (batch, n, k), dtype = case

    torch.manual_seed(42)
    x = torch.randn(batch, n, dtype=dtype, device="cuda")

    indices = cutedsl_topk(x, k, sorted=False)
    _verify_topk_indices(x, k, indices)
    assert bool((indices[:, 1:] > indices[:, :-1]).all()), "rows not strictly ascending"
    assert torch.equal(indices, cutedsl_topk(x, k))
    indices_i32 = cutedsl_topk(x, k, out_dtype=torch.int32)
    assert indices_i32.dtype == torch.int32
    assert torch.equal(indices_i32.long(), indices)


@pytest.mark.parametrize("shape", [(4, 4096, 256), (3, 300000, 30000)])
@pytest.mark.parametrize("dtype", _dtypes)
def test_cutedsl_topk_ties(shape, dtype):
    """Heavy ties at the pivot: the kernel must keep the lowest tied indices."""
    batch, n, k = shape

    torch.manual_seed(42)
    x = torch.randint(0, 16, (batch, n), device="cuda").to(dtype)

    indices = cutedsl_topk(x, k)
    assert torch.equal(indices, _lowest_index_topk_ref(x, k))


@pytest.mark.parametrize("dtype", _dtypes)
def test_cutedsl_topk_misaligned_base(dtype):
    """A contiguous view that starts one element into its storage."""
    batch, n, k = 4, 4096, 256

    torch.manual_seed(42)
    storage = torch.randn(batch * n + 1, dtype=dtype, device="cuda")
    x = storage[1:].view(batch, n)
    assert x.data_ptr() % 16

    _verify_topk_indices(x, k, cutedsl_topk(x, k))


def test_cutedsl_topk_rejects_sorted():
    x = torch.randn(2, 1024, device="cuda")
    with pytest.raises(ValueError):
        cutedsl_topk(x, 8, sorted=True)
