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

from magi_attention.testing.precision import calc_inf_norm, calc_rounding_noise_norm

NEG_INF = float("-inf")
POS_INF = float("inf")


class TestNonFiniteLinfGate:
    """``-inf`` LSE rows that attend to no key must count as zero error when
    both sides agree and as ``inf`` error when they disagree."""

    def test_equal_neg_inf_rows_are_zero_error(self):
        ref = torch.tensor([NEG_INF, 1.0, 2.0])
        assert calc_inf_norm(ref.clone(), ref) == 0.0
        assert calc_inf_norm(torch.tensor([NEG_INF, 1.5, 2.0]), ref) == 0.5

    def test_finite_against_neg_inf_is_inf_error(self):
        ref = torch.tensor([NEG_INF, 1.0])
        assert calc_inf_norm(torch.tensor([0.0, 1.0]), ref) == POS_INF
        assert calc_inf_norm(ref, torch.tensor([0.0, 1.0])) == POS_INF

    def test_opposite_inf_signs_are_inf_error(self):
        assert (
            calc_inf_norm(torch.tensor([POS_INF, 1.0]), torch.tensor([NEG_INF, 1.0]))
            == POS_INF
        )

    def test_nan_output_is_not_zero_error(self):
        norm = calc_inf_norm(
            torch.tensor([float("nan"), 1.0]), torch.tensor([NEG_INF, 1.0])
        )
        assert math.isnan(norm)

    def test_rounding_noise_skips_non_finite_entries(self):
        ref = torch.tensor([NEG_INF, 3.0, 1000.0])
        noise = calc_rounding_noise_norm(ref)
        assert math.isfinite(noise)
        assert noise == calc_rounding_noise_norm(ref[1:])

    def test_rounding_noise_of_all_neg_inf_is_zero(self):
        assert calc_rounding_noise_norm(torch.tensor([NEG_INF, NEG_INF])) == 0.0
