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

import itertools
import math
import random
from typing import Any, Generator, Literal, TypeAlias

FlagCombStrategy: TypeAlias = Literal["random", "sequential", "heuristic"]


class FlagCombGenerator:
    """Flag Combination Generator for UniTest

    This is useful when you have a lot of flags
    influencing either the behavior or performance
    individually or simultaneously,

    but you don't want to test all the combinations but only the valuable ones
    to avoid excessive testing time
    """

    def __init__(
        self,
        flags: list[str],
        options: dict[str, list[Any]] = {},
        defaults: dict[str, Any] = {},
        groups: list[tuple[str, ...]] = [],
        strategy: FlagCombStrategy = "heuristic",
        seed: int = 42,
        cycle_times: int = -1,
    ):
        """

        Args:
            flags (list[str]): the flag name list.
            options (dict[str, list[Any]], optional):
                the options of values for the corresponding flag,
                where the non-provided flag will be treated as a boolean flag.
                Defaults to ``{}`` to see each flag as a boolean flag.
            defaults (dict[str, Any], optional):
                the default value for the corresponding flag,
                where the non-provided flag will be treated as a boolean flag and set to ``False`` by default.
                Defaults to ``{}`` to see each flag as a boolean flag and set all to ``False``.
            groups (list[tuple[str, ...]], optional):
                the group of flags, where the inter-group flags are semantically independent,
                and the intra-group flags are semantically related and require more combinations to be tested.
                Defaults to ``[]`` to see each flag as an independent flag.

                NOTE: it might not affect some strategies, such as ``"sequential"`` and ``"random"``.
            strategy (FlagCombStrategy, optional):
                the strategy to generate flag combinations.
                Defaults to ``"heuristic"`` to use some heuristics to generate flag combinations,
                to cover all the valuable combinations as quickly as possible.
            seed (int, optional):
                the seed for the random number generator. Defaults to ``42``.

                NOTE: it might not affect some strategies, such as ``"sequential"``.
            cycle_times (int, optional):
                the number of times to cycle through the flag combinations.
                Defaults to ``-1`` to cycle indefinitely.
        """
        # an ordered and deterministic version for `list(set(...))`
        self.flags = list(dict.fromkeys(flags).keys())

        # init options and defaults
        self.options = {flag: options.get(flag, [False, True]) for flag in self.flags}
        self.defaults = {
            flag: defaults.get(flag, self.options[flag][0]) for flag in self.flags
        }

        # check defaults and reorder options
        for flag in self.flags:
            option_list = self.options[flag]
            default_value = self.defaults[flag]
            assert default_value in option_list, (
                f"The default value for {flag=} ({default_value=}) "
                f"must be in the options ({option_list=})"
            )
            self.options[flag] = [default_value] + [
                v for v in option_list if v != default_value
            ]

        self.groups = groups
        self.strategy = strategy
        self.seed = seed

        assert (
            cycle_times > 0 or cycle_times == -1
        ), f"`cycle_times` must be greater than 0 or -1, but got {cycle_times=}"
        self.cycle_times = cycle_times

        self.comb_set: set[tuple[Any, ...]] = set()

    @property
    def num_flags(self) -> int:
        return len(self.flags)

    @property
    def num_combs(self) -> int:
        return math.prod(len(options) for options in self.options.values())

    def is_comb_covered(self, comb: tuple[Any, ...]) -> bool:
        return comb in self.comb_set

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        return self.iter()

    def __reversed__(self):
        return self.iter(reverse=True)

    def iter(self, reverse: bool = False) -> Generator[dict[str, Any], None, None]:
        if self.cycle_times == -1:
            while True:
                yield from self._iter(reverse)
        else:
            for _ in range(self.cycle_times):
                yield from self._iter(reverse)

    def _iter(self, reverse: bool = False) -> Generator[dict[str, Any], None, None]:
        self.comb_set = set()  # reset the comb set
        match self.strategy:
            case "sequential":
                yield from self._iter_sequential(reverse=reverse)
            case "random":
                yield from self._iter_random(reverse=reverse)
            case "heuristic":
                yield from self._iter_heuristic(reverse=reverse)

    def _iter_sequential(
        self, reverse: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        all_combs = itertools.product(*self.options.values())
        if reverse:
            for comb in reversed(list(all_combs)):
                if comb in self.comb_set:
                    continue
                self.comb_set.add(comb)
                yield from self._yield(comb)
        else:
            for comb in all_combs:
                if comb in self.comb_set:
                    continue
                self.comb_set.add(comb)
                yield from self._yield(comb)

    def _iter_random(
        self, reverse: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        random.seed(self.seed)

        option_lists = list(self.options.values())
        total_num_combs = self.num_combs

        while len(self.comb_set) < total_num_combs:
            comb = tuple([random.choice(option_list) for option_list in option_lists])

            if comb in self.comb_set:
                continue

            self.comb_set.add(comb)
            yield from self._yield(comb)

    def _iter_heuristic(
        self, reverse: bool = False
    ) -> Generator[dict[str, Any], None, None]:
        raise NotImplementedError

    def _yield(self, comb: tuple[Any, ...]) -> Generator[dict[str, Any], None, None]:
        yield dict(zip(self.flags, comb))
