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
import unittest

import pytest
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl.ffa_utils import normalize_softcap
from magi_attention.testing import parameterize


@pytest.mark.arch_independent
class TestNormalizeSoftcap(unittest.TestCase):
    def test_disabled_and_positive_values(self):
        self.assertIsNone(normalize_softcap(None))
        self.assertIsNone(normalize_softcap(0.0))
        self.assertEqual(normalize_softcap(30), 30.0)

    @parameterize("value", [-1.0, math.nan, math.inf, -math.inf])
    def test_rejects_negative_and_nonfinite_values(self, value):
        with self.assertRaisesRegex(ValueError, "softcap must be"):
            normalize_softcap(value)


if __name__ == "__main__":
    run_tests()
