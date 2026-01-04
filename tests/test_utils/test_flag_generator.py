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

from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.flag_generator import FlagCombGenerator, FlagCombStrategy


class TestFlagGenerator(DistTestBase):
    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "flag_config",
        [
            {
                "name": "test1_all_boolean_flags",
                "flags": ["a", "b", "c", "d", "e", "f"],
                "groups": [("a", "b"), ("d", "e", "f")],
                "cycle_times": 2,
                # answers
                "num_flags": 6,
                "num_combs": 64,
                "constant_comb": {
                    "a": False,
                    "b": False,
                    "c": False,
                    "d": False,
                    "e": False,
                    "f": False,
                },
                "sequential_first_comb": {
                    "a": False,
                    "b": False,
                    "c": False,
                    "d": False,
                    "e": False,
                    "f": False,
                },
                "sequential_last_comb": {
                    "a": True,
                    "b": True,
                    "c": True,
                    "d": True,
                    "e": True,
                    "f": True,
                },
                "heuristic_first_combs": [
                    {
                        "a": False,
                        "b": False,
                        "c": False,
                        "d": False,
                        "e": False,
                        "f": False,
                    },
                    {"a": True, "b": True, "c": True, "d": True, "e": True, "f": True},
                    {
                        "a": True,
                        "b": False,
                        "c": False,
                        "d": False,
                        "e": False,
                        "f": False,
                    },
                    {
                        "a": False,
                        "b": True,
                        "c": False,
                        "d": False,
                        "e": False,
                        "f": False,
                    },
                    {
                        "a": False,
                        "b": False,
                        "c": True,
                        "d": False,
                        "e": False,
                        "f": False,
                    },
                    {
                        "a": False,
                        "b": False,
                        "c": False,
                        "d": True,
                        "e": False,
                        "f": False,
                    },
                    {
                        "a": False,
                        "b": False,
                        "c": False,
                        "d": False,
                        "e": True,
                        "f": False,
                    },
                    {
                        "a": False,
                        "b": False,
                        "c": False,
                        "d": False,
                        "e": False,
                        "f": True,
                    },
                ],
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
                "groups": [("b", "c")],
                "cycle_times": 1,
                # answers
                "num_flags": 4,
                "num_combs": 48,
                "constant_comb": {
                    "a": True,
                    "b": 2,
                    "c": torch.float32,
                    "d": False,
                },
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
                "heuristic_first_combs": [
                    {"a": True, "b": 2, "c": torch.float32, "d": False},
                    {"a": False, "b": 3, "c": torch.float64, "d": True},
                    {"a": False, "b": 2, "c": torch.float32, "d": False},
                    {"a": True, "b": 1, "c": torch.float32, "d": False},
                    {"a": True, "b": 3, "c": torch.float32, "d": False},
                    {"a": True, "b": 2, "c": torch.bfloat16, "d": False},
                    {"a": True, "b": 2, "c": torch.float16, "d": False},
                    {"a": True, "b": 2, "c": torch.float64, "d": False},
                    {"a": True, "b": 2, "c": torch.float32, "d": True},
                ],
            },
        ],
    )
    @parameterize(
        "strategy",
        [
            "constant",
            "sequential",
            "random",
            "heuristic",
        ],
    )
    def test_flag_generator(
        self,
        flag_config: dict[str, Any],
        strategy: FlagCombStrategy,
    ):
        name = f"[{flag_config['name']}]x[{strategy}]"
        flags = flag_config["flags"]
        options = flag_config.get("options", {})
        defaults = flag_config.get("defaults", {})
        groups = flag_config.get("groups", [])
        cycle_times = flag_config.get("cycle_times", 1)

        generator = FlagCombGenerator(
            flags=flags,
            options=options,
            defaults=defaults,
            groups=groups,
            strategy=strategy,
            cycle_times=cycle_times,
        )

        assert generator.num_flags == flag_config["num_flags"]
        assert generator.num_combs == flag_config["num_combs"]

        match strategy:
            case "constant":
                iterator = iter(generator)
                for _ in range(cycle_times):
                    assert next(iterator) == flag_config["constant_comb"]
                with pytest.raises(StopIteration):
                    next(iterator)
            case "sequential":
                first_comb = next(iter(generator))
                assert first_comb == flag_config["sequential_first_comb"]
                last_comb = next(reversed(generator))
                assert last_comb == flag_config["sequential_last_comb"]
            case "random":
                first_comb = next(iter(generator))
                first_comb = FlagCombGenerator.sync_group(
                    first_comb,
                    group=self.process_group,
                )

                first_comb_list = [None] * self.world_size
                dist.all_gather_object(
                    first_comb_list,
                    first_comb,
                    group=self.process_group,
                )

                first_comb0 = first_comb_list[0]
                for comb in first_comb_list:
                    assert comb == first_comb0
            case "heuristic":
                iterator = iter(generator)
                for comb in flag_config["heuristic_first_combs"]:
                    assert next(iterator) == comb

        if self.rank == 0:
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
    run_tests()
