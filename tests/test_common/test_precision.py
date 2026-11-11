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

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from magi_attention.functional.utils import safe_softmax, safe_subtract
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import _check_one
from magi_attention.testing.precision import (
    ROUNDING_NOISE_PERTURBATION,
    calc_inf_norm,
    calc_rounding_noise_norm,
)

NEG_INF = float("-inf")
POS_INF = float("inf")


class TestPrecision(TestCase):
    @parameterize(
        "case",
        [
            (NEG_INF, NEG_INF, 0.5),
            (POS_INF, POS_INF, 0.5),
            (0.0, NEG_INF, POS_INF),
            (NEG_INF, 0.0, POS_INF),
            (POS_INF, NEG_INF, POS_INF),
            (math.nan, NEG_INF, math.nan),
            (math.nan, math.nan, math.nan),
        ],
    )
    def test_non_finite_entries_preserve_error(self, case: tuple[float, float, float]):
        actual, expected, error = case
        norm = calc_inf_norm(torch.tensor([actual, 1.5]), torch.tensor([expected, 1.0]))
        if math.isnan(error):
            self.assertTrue(math.isnan(norm))
        else:
            self.assertEqual(norm, error)

    @parameterize("values", [[3.0, 1000.0], []])
    def test_rounding_noise_ignores_uncovered_rows(self, values: list[float]):
        ref = torch.tensor([NEG_INF, *values])
        self.assertEqual(
            calc_rounding_noise_norm(ref, perturbation=ROUNDING_NOISE_PERTURBATION),
            calc_rounding_noise_norm(
                torch.tensor(values), perturbation=ROUNDING_NOISE_PERTURBATION
            )
            if values
            else 0.0,
        )

    @parameterize("value", [math.nan, POS_INF, NEG_INF])
    def test_reference_comparison_rejects_non_finite_output(self, value: float):
        ref = torch.ones(1024)
        actual = ref.clone()
        actual[0] = value
        with self.assertRaises(AssertionError):
            _check_one("out", actual, ref, ref, torch.bfloat16, self.id())

    @staticmethod
    def _reference_case(
        head_ref: float, head_actual: float, head_len: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(actual, ref_hi, ref_lo) of 4096 elements with max |ref_hi| 4, where
        head_ref and head_actual are given in units of max |ref_hi|.

        actual equals the bf16 rounding of ref_hi except in the first head_len
        elements. ref_lo misses every rtol on 41 elements (1%), which sets the
        mismatch threshold to 2%."""
        scale = 4.0
        ref = scale * torch.linspace(0.25, 1.0, 4096, dtype=torch.float64)
        ref[:head_len] = scale * head_ref
        actual = ref.to(torch.bfloat16).to(torch.float64)
        ref_lo = actual.clone()
        ref_lo[-41:] *= 1.5
        actual[:head_len] = scale * head_actual
        return actual, ref, ref_lo

    @parameterize("name", ["out", "dq", "dk", "dv"])
    def test_reference_comparison_accepts_fp32_noise_at_zero_reference(self, name: str):
        """Noise of 8 fp32 eps of max|ref| where the reference is exactly 0
        passes, although its 6.25% share of the elements is above the 2%
        mismatch threshold; with max|ref| at 4, an atol that does not scale
        with max|ref| would count this noise."""
        noise = 8 * torch.finfo(torch.float32).eps
        actual, ref, ref_lo = self._reference_case(0.0, noise)
        _check_one(name, actual, ref, ref_lo, torch.bfloat16, self.id())

    @parameterize("name", ["out", "dq", "dk", "dv"])
    def test_reference_comparison_rejects_error_at_zero_reference(self, name: str):
        """An error of 1e-3 * max|ref| at exact-zero references on 6.25% of the
        elements stays inside the Linf bound and fails the mismatch count."""
        actual, ref, ref_lo = self._reference_case(0.0, 1e-3)
        with self.assertRaisesRegex(AssertionError, "mismatch_ratio"):
            _check_one(name, actual, ref, ref_lo, torch.bfloat16, self.id())

    @parameterize("name", ["out", "dq", "dk", "dv"])
    def test_reference_comparison_rejects_mismatch_fraction(self, name: str):
        """A 50% relative error on small nonzero references passes on 1% of the
        elements and fails on 6.25%, around the 2% mismatch threshold."""
        actual, ref, ref_lo = self._reference_case(1e-3, 1.5e-3, head_len=41)
        _check_one(name, actual, ref, ref_lo, torch.bfloat16, self.id())
        actual, ref, ref_lo = self._reference_case(1e-3, 1.5e-3)
        with self.assertRaisesRegex(AssertionError, "mismatch_ratio"):
            _check_one(name, actual, ref, ref_lo, torch.bfloat16, self.id())


class TestSafeSubtract(TestCase):
    @parameterize("equal_inf_value", [None, 0.0, 2.0, NEG_INF])
    def test_infinities_and_nan(self, equal_inf_value: float | None):
        a = torch.tensor([NEG_INF, POS_INF, POS_INF, NEG_INF, 3.0, 4.0, math.nan])
        b = torch.tensor([NEG_INF, POS_INF, NEG_INF, POS_INF, 1.0, 4.0, math.nan])
        actual = safe_subtract(a, b, equal_inf_value=equal_inf_value)
        negative = NEG_INF if equal_inf_value is None else equal_inf_value
        positive = math.nan if equal_inf_value is None else equal_inf_value
        expected = torch.tensor(
            [negative, positive, POS_INF, NEG_INF, 2.0, 0.0, math.nan]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    @parameterize("equal_inf_value", [None, 0.0, 2.0, NEG_INF])
    def test_broadcast_gradients(self, equal_inf_value: float | None):
        a = torch.tensor([[NEG_INF], [POS_INF], [3.0]], requires_grad=True)
        b = torch.tensor([[NEG_INF, POS_INF, 1.0]], requires_grad=True)
        result = safe_subtract(a, b, equal_inf_value=equal_inf_value)
        da, db = torch.autograd.grad(result, (a, b), torch.ones_like(result))
        positive_grad = 3.0 if equal_inf_value is None else 2.0
        self.assertEqual(da, torch.tensor([[2.0], [positive_grad], [3.0]]))
        self.assertEqual(db, torch.tensor([[-2.0, -positive_grad, -3.0]]))

    def test_default_softmax_empty_rows(self):
        x = torch.tensor([[NEG_INF, NEG_INF], [0.0, 0.0]], requires_grad=True)
        lse = torch.tensor([[NEG_INF], [math.log(2.0)]], requires_grad=True)
        probabilities = safe_softmax(x, lse=lse)
        self.assertEqual(probabilities, torch.tensor([[0.0, 0.0], [0.5, 0.5]]))
        dx, dlse = torch.autograd.grad(probabilities.sum(), (x, lse))
        self.assertEqual(dx, torch.tensor([[0.0, 0.0], [0.5, 0.5]]))
        self.assertEqual(dlse, torch.tensor([[0.0], [-1.0]]))


if __name__ == "__main__":
    run_tests()
