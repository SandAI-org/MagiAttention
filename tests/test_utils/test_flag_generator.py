# Copyright (c) 2025 SandAI. All Rights Reserved.
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

import random
import unittest
from typing import Any
from unittest import TestCase

import torch

from magi_attention.testing import parameterize
from magi_attention.testing.flag_generator import FlagCombGenerator, FlagCombStrategy


class TestFlagGenerator(TestCase):
    def setUp(self):
        super().setUp()

        random.seed(self.seed)

    @property
    def seed(self) -> int:
        return 42

    @parameterize(
        "flag_config",
        [
            {
                "name": "test1_all_boolean_flags",
                "flags": ["a", "b", "c", "a"],
                "cycle_times": 3,
                # answers
                "num_flags": 3,
                "num_combs": 8,
                "sequential_first_comb": {"a": False, "b": False, "c": False},
                "sequential_last_comb": {"a": True, "b": True, "c": True},
            },
            {
                "name": "test2_various_flags",
                "flags": ["a", "b", "c", "d"],
                "options": {
                    "a": [False, True],
                    "b": [1, 2, 3],
                    "c": [torch.bfloat16, torch.float16, torch.float32, torch.float64],
                },
                "defaults": {
                    "a": True,
                    "b": 2,
                    "c": torch.float32,
                },
                "cycle_times": 1,
                # answers
                "num_flags": 4,
                "num_combs": 48,
                "sequential_first_comb": {
                    "a": True,
                    "b": 2,
                    "c": torch.float32,
                    "d": False,
                },
                "sequential_last_comb": {
                    "a": False,
                    "b": 3,
                    "c": torch.float64,
                    "d": True,
                },
            },
        ],
    )
    @parameterize(
        "strategy",
        [
            "sequential",
            "random",
            # "heuristic",
        ],
    )
    def test_flag_generator(
        self,
        flag_config: dict[str, Any],
        strategy: FlagCombStrategy,
    ):
        name = f"[{flag_config['name']}]x[{strategy}]"
        generator = FlagCombGenerator(
            flags=flag_config["flags"],
            options=flag_config.get("options", {}),
            defaults=flag_config.get("defaults", {}),
            groups=flag_config.get("groups", []),
            strategy=strategy,
            cycle_times=flag_config.get("cycle_times", 1),
        )

        assert generator.num_flags == flag_config["num_flags"]
        assert generator.num_combs == flag_config["num_combs"]

        match strategy:
            case "sequential":
                first_comb = next(iter(generator))
                assert first_comb == flag_config["sequential_first_comb"]
                last_comb = next(reversed(generator))
                assert last_comb == flag_config["sequential_last_comb"]
            case "random":
                pass
            case "heuristic":
                pass

        # just print the attributes and combinations from the generator for check
        print(
            f"For {name}: {generator.flags=} | {generator.num_flags=} | {generator.num_combs=}"
        )
        print(f"For {name}: {generator.options=}")
        print(f"For {name}: {generator.defaults=}")
        print(f"For {name}: {generator.groups=}")

        for idx, flag_comb in enumerate(generator):
            print(f"For [{name}]: Comb {idx} => {flag_comb}")


if __name__ == "__main__":
    unittest.main()
