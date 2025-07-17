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
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn

from magi_attention.common.enum import AttnOverlapMode, OverlapAlgType


@dataclass(frozen=True)
class OverlapAlg(ABC):
    """The abstract config/meta info dataclass for specific overlap algorithm"""

    @property
    @abstractmethod
    def type(self) -> OverlapAlgType:
        """The type enum of the overlap algorithm"""

    @property
    @abstractmethod
    def is_optimal(self) -> bool:
        """Whether the overlap algorithm is optimal"""


@dataclass(frozen=True)
class UniformOverlapAlg(OverlapAlg):
    """The config/meta info dataclass for the uniform overlap algorithm"""

    random_costs: bool = False
    random_seed: int | None = None

    @property
    def type(self) -> OverlapAlgType:
        return OverlapAlgType.UNIFORM

    @property
    def is_optimal(self) -> bool:
        """Whether the overlap algorithm is optimal"""
        return False


@dataclass(frozen=True)
class GreedyOverlapAlg(OverlapAlg):
    """The config/meta info dataclass for the greedy overlap algorithm"""

    @property
    def type(self) -> OverlapAlgType:
        return OverlapAlgType.GREEDY

    @property
    def is_optimal(self) -> bool:
        """Whether the overlap algorithm is optimal"""
        return False


@dataclass(frozen=True)
class OverlapConfig:
    """The config dataclass for multi-stage overlapping"""

    enable: bool = True  # if False, turn off the multi-stage overlapping mode

    mode: AttnOverlapMode = AttnOverlapMode.STATIC

    degree: int | None = 1
    dynamic_max_degree: int | None = (
        8  # only used in dynamic mode, if None, then no limit
    )

    min_chunk_size: int = 512
    max_num_chunks: int = 64

    # TODO: use another non-trivial alg as default in the future
    alg: OverlapAlg = UniformOverlapAlg()

    calc_cost_factor: float = (
        1.0  # define: calc_cost = calc_cost_factor * calc_area (unit: μs)
    )
    comm_cost_factor: float = (
        1.0  # define: comm_cost = comm_cost_factor * comm_size (unit: μs)
    )

    def __post_init__(self):
        if not self.enable:
            # HACK: force auto-set other attrs to disable mso
            object.__setattr__(self, "mode", AttnOverlapMode.STATIC)
            object.__setattr__(self, "degree", 1)
            object.__setattr__(self, "max_num_chunks", 1)

        if self.mode is AttnOverlapMode.STATIC:
            assert self.degree is not None, (
                "When using static overlap mode, "
                f"the {self.degree=} should be set explicitly."
            )
            assert (
                self.degree <= self.max_num_chunks
            ), f"The {self.max_num_chunks=} should be no less than {self.degree=}."
        elif self.mode is AttnOverlapMode.DYNAMIC:
            assert self.degree is None, (
                "When using dynamic overlap mode, "
                f"the {self.degree=} should not be set."
            )
            assert self.dynamic_max_degree is None or (
                self.dynamic_max_degree > 0
            ), f"The {self.dynamic_max_degree=} should be greater than 0 if specified."

        assert (
            self.min_chunk_size > 0
        ), f"The {self.min_chunk_size=} should be greater than 0."
        assert (
            self.max_num_chunks > 0
        ), f"The {self.max_num_chunks=} should be greater than 0."
        assert (
            self.calc_cost_factor > 0.0 and self.comm_cost_factor > 0.0
        ), f"The {self.calc_cost_factor=} and {self.comm_cost_factor=} should be both greater than 0."


@dataclass
class OverlapStageCost:
    """The comm/calc time cost of one overlap stage (unit: μs).
    NOTE: the launch timestamp of the calc cost can be at most
    as early as the finish timestamp of the comm cost in the timeline.
    """

    comm_cost: float = 0.0
    calc_cost: float = 0.0

    def __post_init__(self) -> None:
        assert (
            self.comm_cost >= 0.0 and self.calc_cost >= 0.0
        ), f"The comm cost ({self.comm_cost}) and calc cost ({self.calc_cost}) should be non-negative."


@dataclass
class OverlapSolution:
    """An overlap solution dataclass, made of several info as follows:
    1. overlap_degree: the number of remote overlap stages.
    2. partitions: the partitions of the stage costs, a list with length `overlap_degree`,
        each element of which is a list of stage indices in the `stage_costs`,
        among them any two elements are mutually exclusive.
        NOTE: we agree that the idx '0' is for host cost pair, and will be always put in the first partition
    3. overall_cost: the overall timeline cost unitl the last stage done scheduled by this partitions.
    """

    overlap_degree: int
    overall_cost: float
    partitions: list[list[int]]

    def __post_init__(self) -> None:
        assert (
            self.overlap_degree >= 0
        ), f"The overlap degree ({self.overlap_degree}) should be non-negative."

        assert (
            self.overall_cost >= 0.0
        ), f"The overall cost ({self.overall_cost}) should be non-negative."

        assert len(self.partitions) == self.overlap_degree, (
            f"The length of the partitions ({len(self.partitions)}) should be equal to "
            f"the overlap degree ({self.overlap_degree})."
        )


class OverlapSolver(nn.Module):
    """The implementation of the algorithms for multi-stage overlapping specified by `alg`."""

    def __init__(self, alg: OverlapAlg) -> None:
        super().__init__()

        self.alg = alg

        self.solve_func = {
            OverlapAlgType.UNIFORM: self._solve_with_uniform,
            OverlapAlgType.GREEDY: self._solve_with_greedy,
        }[self.alg.type]

        # return values
        self.best_solution: OverlapSolution = None  # type: ignore
        self.solution_dict: dict[int, OverlapSolution] = {}

    def solve(
        self,
        stage_costs: list[OverlapStageCost],
        overlap_degree: int | None = None,
        dynamic_max_degree: int | None = None,
    ) -> tuple[OverlapSolution, dict[int, OverlapSolution]]:
        """Solve the multi-stage overlapping problem.

        Args:
            stage_costs (list[OverlapStageCost]): the (calc cost, comm cost) pair for each latent overlap stage
                which consists of 1 host stage cost and n remote stage costs
            overlap_degree (int | None): the number of remote overlap stages
            dynamic_max_degree: the maximum overlap degree to try if `overlap_degree` is None, i.e. in dynamic mode

        Returns:
            best_solution (OverlapSolution): the best solution with the minimum overall cost
            solutions_dict (dict[int, OverlapSolution]): the solutions for different overlap degrees
        """

        assert len(stage_costs) >= 1, f"The {len(stage_costs)=} should be at least 1."

        assert (
            overlap_degree is None or overlap_degree > 0
        ), f"The {overlap_degree=} should be positive if specified."

        assert stage_costs[0].comm_cost == 0.0, (
            f"The {stage_costs[0]=} should be the host stage cost, "
            "whose comm cost must be zero."
        )

        self.solve_func(
            stage_costs=stage_costs,
            overlap_degree=overlap_degree,
            dynamic_max_degree=dynamic_max_degree,
            **asdict(self.alg),
        )

        assert len(self.solution_dict) > 0, "No solution is found"

        self.best_solution = self._get_best_solution_from_dict(self.solution_dict)

        return self.best_solution, self.solution_dict

    def _solve_with_uniform(
        self,
        stage_costs: list[OverlapStageCost],
        overlap_degree: int | None = None,
        dynamic_max_degree: int | None = None,
        **kwargs,
    ) -> None:
        """Uniformly partition the stage costs into `overlap_degree` merged stages
        which only serves as as a dummy but feasible solution
        for verification of correctness, instead of production.

        Args:
            stage_costs (list[OverlapStageCost]): the (calc cost, comm cost) pair for each latent overlap stage
            overlap_degree (int | None): the number of remote overlap stages
            **kwargs: the additional keyword arguments for the specific algorithm, including:
                - random_costs (bool): whether to randomly shuffle the stage costs
                - random_seed (int | None): the random seed for shuffling the stage costs if needed, None means no random seed

        e.g.
            1. if random, partitions might be: [[0,4,2],[num_stages-2,1,5], ...]
            2. otherwise, partitions might be: [[0,1,2],[3,4,5], ...]
        """

        random_costs = kwargs["random_costs"]
        random_seed = kwargs["random_seed"]

        def _solve_with_static_overlap_degree(overlap_degree: int) -> OverlapSolution:
            partitions: list[list[int]] = []

            #  ------    split the required overlap degree   ------ #
            # into two parts:
            #   1. the one to split the stages
            #   2. the one for the idle stages appended in the last

            if overlap_degree + 1 > num_stages:
                overlap_degree_idle = overlap_degree + 1 - num_stages
                overlap_degree = num_stages - 1
            else:
                overlap_degree_idle = 0

            #  ------    partition the stage idxs   ------ #

            if overlap_degree == 0:
                # NOTE: corner case when no remote stage is needed
                # then all overlap degree is idle, so we just fill the partitions manually
                partitions = [[] for _ in range(overlap_degree_idle)]
                partitions[0].append(0)  # host cost idx
            else:
                num_stages_in_one_degree = num_stages // overlap_degree
                num_stages_remain = num_stages % overlap_degree

                stage_idxs = list(range(1, num_stages))  # [1,2,3,4,5,...,num_stages-1]
                if random_costs:
                    if random_seed is not None:
                        random.seed(random_seed)
                    random.shuffle(stage_idxs)  # [4,2,num_stages-2,1,5, ...]
                # NOTE: agree that: 0 (host stage cost) has to be put at first: [0,4,2,num_stages-2,1,5,...]
                stage_idxs = [0] + stage_idxs

                start = 0
                for i in range(overlap_degree):
                    end = (
                        start
                        + num_stages_in_one_degree
                        + (1 if i < num_stages_remain else 0)
                    )
                    partitions.append(stage_idxs[start:end])
                    start = end

                #  ------    append the idle degree if needed   ------ #

                for _ in range(overlap_degree_idle):
                    partitions.append([])

            #  ------    reset the required overlap degree   ------ #

            overlap_degree += overlap_degree_idle

            #  ------    calc overall cost   ------ #

            overall_cost = self._calc_overall_cost(
                stage_costs=stage_costs,
                partitions=partitions,
                overlap_degree=overlap_degree,
            )

            #  ------    construct the solution   ------ #

            solution = OverlapSolution(
                overlap_degree=overlap_degree,
                overall_cost=overall_cost,
                partitions=partitions,
            )

            return solution

        num_stages = len(stage_costs)

        if overlap_degree is None:
            dynamic_max_degree = dynamic_max_degree or num_stages
            for overlap_degree in range(1, dynamic_max_degree + 1):
                solution = _solve_with_static_overlap_degree(overlap_degree)
                self.solution_dict[overlap_degree] = solution
        else:
            solution = _solve_with_static_overlap_degree(overlap_degree)
            self.solution_dict[overlap_degree] = solution

    def _solve_with_greedy(
        self,
        stage_costs: list[OverlapStageCost],
        overlap_degree: int | None = None,
        dynamic_max_degree: int | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError("TODO: implement the greedy algorithm")

    def _get_best_solution_from_dict(
        self,
        solution_dict: dict[int, OverlapSolution],
    ) -> OverlapSolution:
        return sorted(
            solution_dict.values(),
            # NOTE: the cmp key is bi-level:
            # 1. first level: minimize the overall cost (resolution as 1 μs)
            # 2. if the overall cost is approximately equal,
            #   then second level: minimize the overlap degree
            key=lambda sol: (round(sol.overall_cost), sol.overlap_degree),
        )[0]

    def _calc_overall_cost(
        self,
        stage_costs: list[OverlapStageCost],
        partitions: list[list[int]],
        overlap_degree: int,
    ) -> float:
        # HACK: for now, with the hypothesis that:
        # every internal comm/calc cost pair can be perfectly overlapped by the larger one
        # we just calc the overall cost as the sum of two parts:
        # 1. the sum of the maximum of ith comm cost and (i-1)th calc cost pair, for i in [0,1,...,overlap_degree-1]
        # 2. the last remote calc cost, with the index of -1

        overall_cost = 0.0
        for i in range(overlap_degree):
            if i == 0:
                # first remote comm cost overlapped with the host calc cost
                overall_cost += max(
                    # first remote comm cost
                    sum(stage_costs[idx].comm_cost for idx in partitions[0]),
                    # host calc cost
                    stage_costs[0].calc_cost,
                )
            else:  # ith remote comm cost overlapped with (i-1)th remote calc cost
                overall_cost += max(
                    # ith remote comm cost
                    sum(stage_costs[idx].comm_cost for idx in partitions[i]),
                    # (i-1)th remote calc cost
                    sum(
                        stage_costs[idx].calc_cost
                        for idx in partitions[i - 1]
                        if idx != 0
                    ),
                )

        # last remote calc cost
        overall_cost += sum(
            stage_costs[idx].calc_cost for idx in partitions[-1] if idx != 0
        )

        return overall_cost

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OverlapSolver):
            return False

        return (self.alg, self.best_solution, self.solution_dict) == (
            other.alg,
            other.best_solution,
            other.solution_dict,
        )
