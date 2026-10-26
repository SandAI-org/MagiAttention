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

from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import _check_one
from magi_attention.testing.precision import calc_inf_norm, calc_rounding_noise_norm

NEG_INF = float("-inf")
POS_INF = float("inf")


class TestPrecision(TestCase):
    @parameterize(
        "case",
        [
            (NEG_INF, NEG_INF, 0.5),
            (0.0, NEG_INF, POS_INF),
            (NEG_INF, 0.0, POS_INF),
            (POS_INF, NEG_INF, POS_INF),
            (math.nan, NEG_INF, math.nan),
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
            calc_rounding_noise_norm(ref),
            calc_rounding_noise_norm(torch.tensor(values)) if values else 0.0,
        )

    @parameterize("value", [math.nan, POS_INF, NEG_INF])
    def test_reference_comparison_rejects_non_finite_output(self, value: float):
        ref = torch.ones(1024)
        actual = ref.clone()
        actual[0] = value
        with self.assertRaises(AssertionError):
            _check_one("out", actual, ref, ref, torch.bfloat16, self.id())


if __name__ == "__main__":
    run_tests()
