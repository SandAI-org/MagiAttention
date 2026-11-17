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

import unittest

import pytest

from magi_attention.testing.flag_generator import FlagCombGenerator


class TestFlagGeneratorNoValidComb(unittest.TestCase):
    """Tests that do not require distributed environment."""

    def test_get_next_valid_comb_no_valid_raises(self):
        gen = FlagCombGenerator(
            flags=["a", "b"],
            strategy="sequential",
            cycle_times=1,
        )

        def always_invalid(comb, cfg):
            return False

        with pytest.raises(RuntimeError, match="FlagCombGenerator"):
            gen.get_next_valid_comb(
                test_config={"backend": "ffa"},
                is_valid_fn=always_invalid,
            )

    def test_get_next_valid_comb_exhausted_iterator_raises(self):
        gen = FlagCombGenerator(
            flags=["a"],
            strategy="constant",
            cycle_times=1,
        )

        def always_invalid(comb, cfg):
            return False

        with pytest.raises(RuntimeError, match="FlagCombGenerator"):
            gen.get_next_valid_comb(
                test_config={},
                is_valid_fn=always_invalid,
            )

    def test_get_next_valid_comb_finds_valid(self):
        gen = FlagCombGenerator(
            flags=["a"],
            strategy="sequential",
            cycle_times=1,
        )

        def only_true(comb, cfg):
            return comb.get("a") is True

        result = gen.get_next_valid_comb(
            test_config={},
            is_valid_fn=only_true,
        )
        assert result["a"] is True


if __name__ == "__main__":
    unittest.main()
