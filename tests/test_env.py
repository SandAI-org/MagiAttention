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


import os
import unittest
from unittest import TestCase, mock

from magi_attention import env
from magi_attention.testing.utils import switch_env

BWD_PACK_FACTOR_ENV = "MAGI_ATTENTION_BWD_PACK_FACTOR"


class TestBwdPackFactorEnv(TestCase):
    """Parsing of MAGI_ATTENTION_BWD_PACK_FACTOR."""

    def test_default_is_one(self):
        # The magi_attention.testing env helpers only set values, so removing
        # the variable needs mock.patch.dict, which restores it on exit.
        with mock.patch.dict(os.environ):
            os.environ.pop(BWD_PACK_FACTOR_ENV, None)
            self.assertEqual(env.general.bwd_pack_factor(), 1)

    def test_positive_integer(self):
        for value in ("1", "2", "4", "64"):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                self.assertEqual(env.general.bwd_pack_factor(), int(value), value)

    def test_invalid_value_raises(self):
        for value in ("0", "-2"):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                with self.assertRaisesRegex(ValueError, "positive integer", msg=value):
                    env.general.bwd_pack_factor()
        for value in ("four", "2.5", ""):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                with self.assertRaisesRegex(ValueError, "invalid literal", msg=value):
                    env.general.bwd_pack_factor()


if __name__ == "__main__":
    unittest.main()
