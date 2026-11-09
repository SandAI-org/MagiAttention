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

"""Correctness tests for ``cutedsl_topk_stream``, the CuTe DSL streaming top-k kernel
(bf16 / fp16, k <= 10240, n <= 2**21).

The kernel is verified against ``torch.topk``: values gathered by the returned
indices are sorted on both sides and compared with ``assert_close``. Index
identity, including tie-breaking, is checked with ``assert_equal``.

Run:
    pytest tests/test_kernel/cutedsl/test_topk_stream.py -v
"""

import random
from functools import partial
from itertools import product

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.kernel.cutedsl.topk_stream import MAX_K, MAX_N, cutedsl_topk_stream
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
    (2, 300000, 10240),
    (40, 300000, 3072),
    # Row lengths that are not a multiple of the 16-byte vector width.
    (3, 7, 3),
    (2, 1001, 100),
    (2, 300001, 3000),
]

_dtypes = [torch.bfloat16, torch.float16]

_cases = list(product(_shapes, _dtypes))

_tie_cases = list(product([(4, 4096, 256), (3, 300000, 10240)], _dtypes))


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
        test_case="cutedsl_topk_stream values",
    )


def _lowest_index_topk_ref(x: torch.Tensor, k: int) -> torch.Tensor:
    """Ascending top-k indices where ties at the k-th value keep the lowest indices."""
    kth = torch.topk(x, k, dim=-1).values[..., -1:]
    gt = x > kth
    eq = x == kth
    need_eq = k - gt.sum(dim=-1, keepdim=True)
    selected = gt | (eq & (eq.cumsum(dim=-1) <= need_eq))
    return selected.nonzero()[:, -1].view(*x.shape[:-1], k)


def _lowest_index_topk_ref_end(
    x: torch.Tensor, k: int, end: torch.Tensor
) -> torch.Tensor:
    """``_lowest_index_topk_ref`` of each row's first ``end`` elements, where NaN is never
    selected; a row with fewer than k selectable elements pads with -1."""
    rows, n = x.shape
    xv = x.float()
    end = end.long()
    valid = (torch.arange(n, device=x.device)[None, :] < end[:, None]) & ~xv.isnan()
    masked = torch.where(valid, xv, torch.full_like(xv, float("-inf")))
    kth = torch.topk(masked, k, dim=-1).values[:, -1:]
    gt = valid & (xv > kth)
    eq = valid & (xv == kth)
    need = k - gt.sum(-1, keepdim=True)
    selected = gt | (eq & (eq.cumsum(-1) <= need))
    selected = torch.where(valid.sum(-1, keepdim=True) < k, valid, selected)
    pos = selected.cumsum(-1) - 1
    out = torch.full((rows, k), -1, dtype=torch.int64, device=x.device)
    r_idx, c_idx = selected.nonzero(as_tuple=True)
    out[r_idx, pos[r_idx, c_idx]] = c_idx
    return out


def _stream_cases(dtype: torch.dtype, generator: torch.Generator):
    """Inputs for the streaming kernel's corner paths: heavy ties at the threshold, a row cut
    to k on overflow, a sampled threshold that is too high (second pass), -inf and signed
    zeros."""
    rows, n = 8, 294912
    randn = partial(torch.randn, rows, n, device="cuda", generator=generator)
    peak = randn()
    # Every sampled block (64 elements every 2048 for this n) is a peak, so the
    # threshold comes out too high.
    peak[(torch.arange(n, device="cuda") % 2048 < 64).expand(rows, n)] += 6.0
    neg_inf = randn()
    neg_inf[:, ::7] = float("-inf")
    signed_zero = torch.randint(
        0, 2, (rows, n), device="cuda", generator=generator
    ).float()
    yield "relu_sparse", torch.relu(randn() - 2.5).to(dtype)
    yield "quantized", (torch.round(randn() * 4) / 4).to(dtype)
    yield "sampled_peak", peak.to(dtype)
    yield "neg_inf", neg_inf.to(dtype)
    yield "signed_zero", (signed_zero * -0.0).to(dtype)


def _assert_topk_values_exact(
    x: torch.Tensor, k: int, indices: torch.Tensor, test_case: str
) -> None:
    """Gathered values, sorted as torch.topk sorts them, match exactly."""
    ref = torch.topk(x, k, dim=-1, sorted=True).values
    got = torch.gather(x, -1, indices.long()).sort(dim=-1, descending=True).values
    assert_equal(got, ref, test_case=test_case)


def _assert_topk_indices_match_torch(
    x: torch.Tensor, k: int, indices: torch.Tensor, test_case: str
) -> None:
    """Exact index identity against torch.topk; needs unique values per row."""
    ref = torch.topk(x, k, dim=-1, sorted=True)
    assert_equal(
        indices.sort(dim=-1).values,
        ref.indices.sort(dim=-1).values,
        test_case=test_case,
    )


_stream_dtypes = [torch.bfloat16, torch.float16]

# (batch, n, k) around tile, segment and sampling-stride boundaries, with k <= MAX_K and
# n <= MAX_N. The wrapper pads n to a multiple of 8.
_stream_exact_shapes = [
    (1, 1, 1),
    (1, 7, 3),
    (1, 8, 8),
    (2, 63, 17),
    (3, 64, 64),
    (2, 65, 1),
    (1, 255, 255),
    (1, 256, 256),
    (3, 257, 100),
    (1, 2047, 1024),
    (2, 2048, 2048),
    (1, 2049, 2048),
    (4, 65535, 4096),
    (3, 65536, 10000),
    (2, 65537, 4096),
    (2, 131072, 10240),
    (1, 294912, 4096),
    (5, 300001, 3000),
    (257, 8192, 512),
    (2, 2**20, 2048),
]
_stream_exact_cases = list(product(_stream_exact_shapes, _stream_dtypes))

# Values are exact integers of the dtype, so the top-k index set is unique and
# torch.topk's own indices are comparable.
_stream_unique_cases = [(torch.bfloat16, 4, 256, k) for k in (1, 17, 128, 256)] + [
    (torch.float16, 4, 2048, k) for k in (1, 513, 1024, 2048)
]

_stream_end_cases = [(dtype, k) for dtype in _stream_dtypes for k in (17, 4096, 10240)]

_stream_max_cases = [
    (torch.bfloat16, 1, MAX_N, MAX_K),
    (torch.float16, 1, MAX_N, MAX_K),
    (torch.bfloat16, 1, MAX_N - 1, MAX_K),
    (torch.float16, 1, MAX_N, 1),
    (torch.bfloat16, 2, MAX_N // 2, MAX_K),
]


def _stream_distributions(dtype: torch.dtype, generator: torch.Generator):
    """Broad value distributions for the streaming path. The NaN case is checked
    against the reference that masks NaN and pads short rows with -1."""
    rows, n = 8, 294912

    def randn() -> torch.Tensor:
        return torch.randn(rows, n, device="cuda", generator=generator)

    yield "randn", randn()
    yield "all_equal", torch.ones(rows, n, device="cuda")
    yield "two_values", torch.randint(
        0, 2, (rows, n), device="cuda", generator=generator
    ).float()
    yield "ramp", torch.linspace(-1.0, 1.0, n, device="cuda").expand(rows, n).clone()
    yield "reverse_ramp", (
        torch.linspace(1.0, -1.0, n, device="cuda").expand(rows, n).clone()
    )
    yield "relu_sparse", torch.relu(randn() - 2.5)
    yield "quantized", torch.round(randn() * 4) / 4
    signs = torch.randint(0, 2, (rows, n), device="cuda", generator=generator).bool()
    yield "signed_zero", torch.zeros(rows, n, device="cuda").masked_fill_(signs, -0.0)
    x = randn()
    x[:, ::7] = float("-inf")
    yield "neg_inf", x
    x = randn()
    x[:, ::97] = float("inf")
    yield "pos_inf", x
    x = randn()
    x[:, ::97] = float("inf")
    x[:, ::89] = float("-inf")
    yield "both_inf", x
    yield "all_neg_inf", torch.full((rows, n), float("-inf"), device="cuda")
    yield "all_pos_inf", torch.full((rows, n), float("inf"), device="cuda")
    scale = 1e30 if dtype == torch.bfloat16 else 1e4
    yield "large_magnitude", -randn().abs() * scale
    x = randn()
    x[:, 1000 : 1000 + 3 * 4096] += 6.0
    yield "local_peak", x
    x = randn()
    peak_blocks = (torch.arange(n, device="cuda") % 2048 < 64).expand(rows, n)
    x[peak_blocks] += 6.0
    yield "sampled_peak", x
    x = randn()
    x[:, ::7] = float("nan")
    yield "nan", x


# The sweep crosses row lengths and k values around tile, segment and sampling-stride
# boundaries with float distributions over many scales, bit-pattern distributions that
# cover denormals, zeros of both signs and infinities, ties straddling the k-th value, and
# NaN hotspots. k > n is skipped because cutedsl_topk_stream rejects it; cases with an odd k
# pass a random per-row ``end``.
_SWEEP_NS = [
    1,
    37,
    1602,
    32768,
    123245,
    225467,
    262144,
    418673,
    682965,
    998123,
    1048576,
    MAX_N - 1,
    MAX_N,
]
_SWEEP_KS = [1, 389, 512, 1024, 1231, 2048, 2132, 2333, 4096, 5000, 8191, 10000]
_SWEEP_KS.append(MAX_K)
_SWEEP_BATCHES = [3, 37, 121, 512, 4096]
# Rows per case are capped so one input holds at most this many elements.
_SWEEP_MAX_ELEMS = 1 << 25

_SWEEP_FLOAT_DISTS = {
    "normal": lambda x: x,
    "normal_plus_100": lambda x: x + 100,
    "normal_minus_100": lambda x: x - 100,
    "normal_x1e2": lambda x: x * 1e2,
    "normal_x1e3": lambda x: x * 1e3,
    "normal_x1e4": lambda x: x * 1e4,
    "normal_x1e-2": lambda x: x * 1e-2,
    "normal_x1e-3": lambda x: x * 1e-3,
    "normal_x1e-4": lambda x: x * 1e-4,
}

_SIGN_BIT = 0x8000
_INF_BITS = {torch.bfloat16: 0x7F80, torch.float16: 0x7C00}
_MIN_NORMAL_BITS = {torch.bfloat16: 0x0080, torch.float16: 0x0400}

# Inclusive bit-pattern ranges, given the +inf bit pattern of the dtype.
_SWEEP_BIT_DISTS = {
    "bits_positive": lambda inf: (0, inf - 1),
    "bits_negative": lambda inf: (_SIGN_BIT, _SIGN_BIT | (inf - 1)),
    "bits_zero": lambda inf: (0, 0),
    "bits_below_0x20": lambda inf: (0, 0x1F),
    "bits_below_0x1000": lambda inf: (0, 0xFFF),
}

_SWEEP_DISTS = [
    *_SWEEP_FLOAT_DISTS,
    *_SWEEP_BIT_DISTS,
    "pivot_ties_positive",
    "pivot_ties_negative",
    "nan_hotspot_positive",
    "nan_hotspot_negative",
]


def _bits_to(bits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """16-bit patterns held in int32 to a tensor of ``dtype`` with those bits."""
    return (
        torch.where(bits >= _SIGN_BIT, bits - 0x10000, bits).to(torch.int16).view(dtype)
    )


def _scatter_segments(
    rows: int, n: int, segments: list[tuple[int, int, int]], generator: torch.Generator
) -> torch.Tensor:
    """Each ``(count, lo, hi)`` segment fills ``count`` random positions of every row with
    bit patterns drawn uniformly from ``[lo, hi]``; the counts sum to ``n``."""
    perm = torch.rand(rows, n, device="cuda", generator=generator).argsort(dim=-1)
    bits = torch.empty(rows, n, dtype=torch.int32, device="cuda")
    start = 0
    for count, lo, hi in segments:
        if count > 0:
            values = torch.randint(
                lo,
                hi + 1,
                (rows, count),
                dtype=torch.int32,
                device="cuda",
                generator=generator,
            )
            bits.scatter_(1, perm[:, start : start + count], values)
        start += count
    assert start == n
    return bits


def _sweep_input(
    dist: str,
    dtype: torch.dtype,
    rows: int,
    n: int,
    k: int,
    generator: torch.Generator,
    rng: random.Random,
) -> torch.Tensor:
    if dist in _SWEEP_FLOAT_DISTS:
        x = torch.randn(rows, n, device="cuda", generator=generator)
        return _SWEEP_FLOAT_DISTS[dist](x).to(dtype)
    inf = _INF_BITS[dtype]
    neg_inf = _SIGN_BIT | inf
    if dist in _SWEEP_BIT_DISTS:
        lo, hi = _SWEEP_BIT_DISTS[dist](inf)
        bits = torch.randint(
            lo, hi + 1, (rows, n), dtype=torch.int32, device="cuda", generator=generator
        )
        return _bits_to(bits, dtype)
    if dist.startswith("pivot_ties"):
        # The pivot repeats around the k-th position: k - num_pivots values rank above
        # it and the rest rank below it.
        magnitude = rng.randint(_MIN_NORMAL_BITS[dtype], inf - 1)
        num_pivots = min(rng.randint(1, max(1, k * 11 // 10)), n)
        num_larger = min(max(k - num_pivots, 0), n - num_pivots)
        num_smaller = n - num_pivots - num_larger
        if dist == "pivot_ties_positive":
            pivot = magnitude
            larger, smaller = (pivot, inf), (_SIGN_BIT, neg_inf)
        else:
            pivot = _SIGN_BIT | magnitude
            larger, smaller = (0, inf), (pivot, neg_inf)
        segments = [
            (num_pivots, pivot, pivot),
            (num_larger, *larger),
            (num_smaller, *smaller),
        ]
        return _bits_to(_scatter_segments(rows, n, segments, generator), dtype)
    # NaN hotspots: up to 8 NaN per row; the rest spans both signs, NaN included.
    nan = inf + 1 if dist == "nan_hotspot_positive" else neg_inf + 1
    hot = min(8, n)
    num_positive = rng.randint(0, n - hot)
    segments = [
        (hot, nan, nan),
        (num_positive, 0, inf + 1),
        (n - hot - num_positive, _SIGN_BIT, neg_inf + 1),
    ]
    return _bits_to(_scatter_segments(rows, n, segments, generator), dtype)


def _assert_matches_torch_topk(
    x: torch.Tensor,
    k: int,
    end: torch.Tensor | None,
    indices: torch.Tensor,
    test_case: str,
) -> None:
    """Compare against torch.topk over each row's selectable elements: the first ``end``
    (all when ``end`` is None) that are not NaN. The sorted gathered values equal
    torch.topk's values exactly; rows whose k-th value is not tied select exactly
    torch.topk's indices; every row equals the reference that keeps the lowest tied
    indices and pads rows with fewer than k selectable elements with -1."""
    rows, n = x.shape
    if end is None:
        end = torch.full((rows,), n, dtype=torch.int32, device=x.device)
    xv = x.float()
    in_range = torch.arange(n, device=x.device)[None, :] < end.long()[:, None]
    valid = in_range & ~xv.isnan()
    masked = torch.where(valid, xv, torch.full_like(xv, float("-inf")))
    ref = torch.topk(masked, k, dim=-1, sorted=True)
    picked = indices.long()
    got = torch.where(
        picked >= 0,
        torch.gather(masked, -1, picked.clamp(min=0)),
        torch.full_like(ref.values, float("-inf")),
    ).sort(dim=-1, descending=True)
    assert_equal(
        got.values, ref.values, test_case=f"{test_case} values", print_no_mismatch=False
    )
    kth = ref.values[:, -1:]
    untied = (valid.sum(dim=-1) >= k) & (
        (masked == kth).sum(dim=-1) == (ref.values == kth).sum(dim=-1)
    )
    assert_equal(
        picked[untied],
        ref.indices[untied].sort(dim=-1).values,
        test_case=f"{test_case} torch indices",
        print_no_mismatch=False,
    )
    assert_equal(
        picked,
        _lowest_index_topk_ref_end(x, k, end),
        test_case=f"{test_case} lowest-index ties",
        print_no_mismatch=False,
    )


class TestCuteDSLTopKStream(TestCase):
    @parameterize("case", _cases)
    def test_cutedsl_topk(self, case):
        (batch, n, k), dtype = case

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(batch, n, dtype=dtype, device="cuda", generator=generator)

        indices = cutedsl_topk_stream(x, k, sorted=False)
        _verify_topk_indices(x, k, indices)
        self.assertTrue(
            bool((indices[:, 1:] > indices[:, :-1]).all()),
            "rows not strictly ascending",
        )
        assert_equal(
            indices,
            cutedsl_topk_stream(x, k),
            test_case="cutedsl_topk_stream default args",
        )
        indices_i32 = cutedsl_topk_stream(x, k, out_dtype=torch.int32)
        self.assertEqual(indices_i32.dtype, torch.int32)
        assert_equal(
            indices_i32.long(),
            indices,
            test_case="cutedsl_topk_stream int32 indices",
        )

    @parameterize("case", _tie_cases)
    def test_cutedsl_topk_ties(self, case):
        """Heavy ties at the pivot: the kernel must keep the lowest tied indices."""
        (batch, n, k), dtype = case

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randint(0, 16, (batch, n), device="cuda", generator=generator).to(
            dtype
        )

        indices = cutedsl_topk_stream(x, k)
        assert_equal(
            indices,
            _lowest_index_topk_ref(x, k),
            test_case="cutedsl_topk_stream lowest-index ties",
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

        _verify_topk_indices(x, k, cutedsl_topk_stream(x, k))

    @parameterize("dtype", _dtypes)
    def test_cutedsl_topk_concurrent_streams(self, dtype):
        """Launches overlapping on two streams each return their own input's top-k."""
        batch, n, k = 4, 300000, 3000
        num_iters = 20

        generator = torch.Generator(device="cuda").manual_seed(42)
        inputs = [
            torch.randn(batch, n, dtype=dtype, device="cuda", generator=generator)
            for _ in range(2)
        ]
        refs = [_lowest_index_topk_ref(x, k) for x in inputs]
        # JIT-compile before the timed window below.
        cutedsl_topk_stream(inputs[0], k)

        main_stream = torch.cuda.current_stream()
        streams = [torch.cuda.Stream() for _ in range(2)]
        outputs: list[list[torch.Tensor]] = [[], []]
        # Hold both streams until every launch is queued so the kernels overlap.
        torch.cuda._sleep(100_000_000)
        for s in streams:
            s.wait_stream(main_stream)
        for _ in range(num_iters):
            for i, s in enumerate(streams):
                with torch.cuda.stream(s):
                    outputs[i].append(cutedsl_topk_stream(inputs[i], k))
        for s in streams:
            main_stream.wait_stream(s)
        torch.cuda.synchronize()

        for i in range(2):
            for out in outputs[i]:
                assert_equal(
                    out, refs[i], test_case=f"cutedsl_topk_stream concurrent stream {i}"
                )

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("k", [4096, 10240])
    def test_cutedsl_topk_stream_corner_cases(self, dtype, k):
        """Ties, overflow cuts and the second pass of the streaming kernel."""
        generator = torch.Generator(device="cuda").manual_seed(42)
        for name, x in _stream_cases(dtype, generator):
            assert_equal(
                cutedsl_topk_stream(x, k),
                _lowest_index_topk_ref(x, k),
                test_case=f"cutedsl_topk_stream {name} {dtype}",
            )

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("n", [65536, 294912 + 3])
    @parameterize("k", [4096, 10000])
    def test_cutedsl_topk_end(self, dtype, n, k):
        """Per-row ``end``, including rows shorter than k (tail padded with -1)."""
        generator = torch.Generator(device="cuda").manual_seed(42)
        fixed = torch.tensor(
            [0, 1, 100, k - 1, k, k + 1, k + 904, 65537, n], device="cuda"
        ).clamp(max=n)
        random_end = torch.randint(k, n + 1, (24,), device="cuda", generator=generator)
        end = torch.cat([fixed, random_end]).to(torch.int32)
        x = torch.randn(end.numel(), n, dtype=dtype, device="cuda", generator=generator)
        for out_dtype in (torch.int64, torch.int32):
            indices = cutedsl_topk_stream(x, k, end=end, out_dtype=out_dtype)
            self.assertEqual(indices.dtype, out_dtype)
            assert_equal(
                indices.long(),
                _lowest_index_topk_ref_end(x, k, end),
                test_case=f"cutedsl_topk_stream end n={n} {dtype} {out_dtype}",
            )

    @parameterize("dtype", [torch.bfloat16, torch.float16])
    def test_cutedsl_topk_stream_nan(self, dtype):
        """NaN is never selected, on rows longer than k and on rows of at most k elements."""
        k, n = 512, 4096
        generator = torch.Generator(device="cuda").manual_seed(42)
        # Rows: short with NaN, exactly k, longer than k, and longer than k with fewer than
        # k selectable elements.
        end = torch.tensor([100, k, 600, n, n], dtype=torch.int32, device="cuda")
        x = torch.randn(end.numel(), n, device="cuda", generator=generator).to(dtype)
        x[0, [0, 3, 99]] = float("nan")
        x[1, [7, k - 1]] = float("nan")
        x[2, 100:300] = float("nan")
        x[3, ::5] = float("nan")
        x[4, : n - 400] = float("nan")
        for out_dtype in (torch.int64, torch.int32):
            indices = cutedsl_topk_stream(x, k, end=end, out_dtype=out_dtype)
            assert_equal(
                indices.long(),
                _lowest_index_topk_ref_end(x, k, end),
                test_case=f"cutedsl_topk_stream NaN {dtype} {out_dtype}",
            )

    def test_cutedsl_topk_rejects_unsupported_inputs(self):
        """fp32 input, rows longer than MAX_N, and k outside [1, min(MAX_K, n)]."""
        x = torch.randn(2, 1024, device="cuda")
        with self.assertRaises(ValueError):
            cutedsl_topk_stream(x, 8)
        long_rows = torch.randn(2, MAX_N + 8, device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            cutedsl_topk_stream(long_rows, 8)
        wide = torch.randn(2, 2 * MAX_K, device="cuda", dtype=torch.bfloat16)
        for k in (0, MAX_K + 1):
            with self.assertRaises(ValueError):
                cutedsl_topk_stream(wide, k)
        with self.assertRaises(ValueError):
            cutedsl_topk_stream(wide[:, :64], 65)

    def test_cutedsl_topk_rejects_sorted(self):
        x = torch.randn(2, 1024, device="cuda")
        with self.assertRaises(ValueError):
            cutedsl_topk_stream(x, 8, sorted=True)


class TestStreamTopKAgainstTorch(TestCase):
    """The streaming kernel against torch.topk, with exact value and index checks."""

    @parameterize("case", _stream_exact_cases)
    def test_stream_shapes_match_torch_exact(self, case):
        (batch, n, k), dtype = case
        self.assertLessEqual(k, MAX_K)
        self.assertLessEqual(n, MAX_N)

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(batch, n, dtype=dtype, device="cuda", generator=generator)
        indices = cutedsl_topk_stream(x, k)
        label = f"stream exact b={batch} n={n} k={k} {dtype}"
        _assert_topk_values_exact(x, k, indices, label)
        assert_equal(indices, _lowest_index_topk_ref(x, k), test_case=label)
        self.assertTrue(
            bool((indices[:, 1:] > indices[:, :-1]).all()),
            f"{label}: rows not strictly ascending",
        )
        indices_i32 = cutedsl_topk_stream(x, k, out_dtype=torch.int32)
        assert_equal(indices_i32.long(), indices, test_case=f"{label} int32")

    @parameterize("case", _stream_unique_cases)
    def test_stream_unique_values_match_torch_indices(self, case):
        dtype, batch, n, k = case
        generator = torch.Generator().manual_seed(42)
        values = torch.stack(
            [torch.randperm(n, generator=generator) for _ in range(batch)]
        )
        x = values.to(device="cuda", dtype=dtype)
        indices = cutedsl_topk_stream(x, k)
        label = f"stream unique {dtype} b={batch} n={n} k={k}"
        _assert_topk_indices_match_torch(x, k, indices, label)
        _assert_topk_values_exact(x, k, indices, label)

    @parameterize("case", _stream_end_cases)
    def test_stream_end_matches_torch(self, case):
        dtype, k = case
        n = 200000
        end = torch.tensor(
            [
                0,
                1,
                k - 1,
                k,
                k + 1,
                63,
                64,
                65,
                2047,
                2048,
                2049,
                65535,
                65536,
                65537,
                131071,
                131072,
                131073,
                n - 1,
                n,
                n + 123,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(end.numel(), n, dtype=dtype, device="cuda", generator=generator)
        indices = cutedsl_topk_stream(x, k, end=end)
        label = f"stream end {dtype} k={k}"
        assert_equal(
            indices,
            _lowest_index_topk_ref_end(x, k, end),
            test_case=label,
        )
        for r in range(end.numel()):
            e = min(int(end[r].item()), n)
            kk = min(k, e)
            if e < k:
                self.assertTrue(
                    bool((indices[r, e:] == -1).all()),
                    f"{label} row {r}: tail not padded with -1",
                )
            if kk == 0:
                continue
            got = torch.gather(x[r], 0, indices[r][indices[r] >= 0])
            ref = torch.topk(x[r, :e], kk, sorted=True).values
            assert_equal(
                got.sort(descending=True).values,
                ref,
                test_case=f"{label} row {r} end={e}",
            )

    @parameterize("dtype", _stream_dtypes)
    def test_stream_distributions_match_torch(self, dtype):
        k = 4096
        generator = torch.Generator(device="cuda").manual_seed(42)
        for name, values in _stream_distributions(dtype, generator):
            x = values.to(dtype)
            indices = cutedsl_topk_stream(x, k)
            label = f"stream dist {name} {dtype}"
            if name == "nan":
                end = torch.full(
                    (x.shape[0],), x.shape[1], dtype=torch.int32, device="cuda"
                )
                assert_equal(
                    indices,
                    _lowest_index_topk_ref_end(x, k, end),
                    test_case=label,
                )
                selected = torch.gather(x, -1, indices.long())
                self.assertFalse(
                    bool(torch.isnan(selected).any()), f"{label}: NaN selected"
                )
                masked = torch.where(
                    torch.isnan(x), torch.full_like(x, float("-inf")), x
                )
                _assert_topk_values_exact(masked, k, indices, label)
            else:
                _assert_topk_values_exact(x, k, indices, label)
                assert_equal(indices, _lowest_index_topk_ref(x, k), test_case=label)

    @parameterize("dtype", _stream_dtypes)
    @parameterize("dist", _SWEEP_DISTS)
    def test_stream_sweep_matches_torch(self, dtype, dist):
        """Every (n, k) pair of the sweep, one batch size each, against torch.topk."""
        generator = torch.Generator(device="cuda").manual_seed(42)
        rng = random.Random(f"{dtype}-{dist}")
        case_id = 0
        for n in _SWEEP_NS:
            for k in _SWEEP_KS:
                if k > n:
                    continue
                rows = min(
                    _SWEEP_BATCHES[case_id % len(_SWEEP_BATCHES)],
                    max(1, _SWEEP_MAX_ELEMS // n),
                )
                out_dtype = torch.int32 if case_id % 2 else torch.int64
                case_id += 1
                x = _sweep_input(dist, dtype, rows, n, k, generator, rng)
                end = None
                if k % 2:
                    end = torch.randint(
                        0,
                        n + 1,
                        (rows,),
                        dtype=torch.int32,
                        device="cuda",
                        generator=generator,
                    )
                indices = cutedsl_topk_stream(x, k, end=end, out_dtype=out_dtype)
                label = (
                    f"stream sweep {dist} {dtype} b={rows} n={n} k={k} "
                    f"end={end is not None} {out_dtype}"
                )
                self.assertEqual(indices.dtype, out_dtype, label)
                self.assertEqual(tuple(indices.shape), (rows, k), label)
                _assert_matches_torch_topk(x, k, end, indices, label)

    @parameterize("case", _stream_max_cases)
    def test_stream_max_bounds_match_torch(self, case):
        dtype, batch, n, k = case
        self.assertLessEqual(k, MAX_K)
        self.assertLessEqual(n, MAX_N)

        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(batch, n, dtype=dtype, device="cuda", generator=generator)
        indices = cutedsl_topk_stream(x, k)
        label = f"stream max b={batch} n={n} k={k} {dtype}"
        _assert_topk_values_exact(x, k, indices, label)
        assert_equal(indices, _lowest_index_topk_ref(x, k), test_case=label)

    @parameterize("dtype", _stream_dtypes)
    def test_stream_3d_matches_torch(self, dtype):
        generator = torch.Generator(device="cuda").manual_seed(42)
        x = torch.randn(2, 3, 8192, dtype=dtype, device="cuda", generator=generator)
        k = 1024
        indices = cutedsl_topk_stream(x, k)
        label = f"stream 3d {dtype}"
        _assert_topk_values_exact(x, k, indices, label)
        assert_equal(
            indices.reshape(-1, k),
            _lowest_index_topk_ref(x.reshape(-1, 8192), k),
            test_case=label,
        )

    @parameterize("dtype", _stream_dtypes)
    def test_stream_noncontiguous_matches_torch(self, dtype):
        generator = torch.Generator(device="cuda").manual_seed(42)
        base = torch.randn(1024, 8, dtype=dtype, device="cuda", generator=generator)
        x = base.T
        self.assertFalse(x.is_contiguous())
        k = 256
        indices = cutedsl_topk_stream(x, k)
        label = f"stream noncontig {dtype}"
        _assert_topk_values_exact(x, k, indices, label)
        assert_equal(indices, _lowest_index_topk_ref(x, k), test_case=label)


if __name__ == "__main__":
    run_tests()
