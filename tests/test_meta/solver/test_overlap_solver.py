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
from unittest import TestCase

from magi_attention.meta.solver.overlap_solver import (
    OverlapSolver,
    OverlapStageCost,
    PipelineOverlapAlg,
    UniformOverlapAlg,
)
from magi_attention.testing import parameterize


class TestPipelineOverlapSolver(TestCase):
    @parameterize("degree", [1, 2, 3, 4])
    def test_partition_conserves_chunks_and_retains_uniform_candidate(self, degree):
        costs = [OverlapStageCost(calc_cost=3)] + [
            OverlapStageCost(comm_cost=2 + i % 3, calc_cost=5 + i * i)
            for i in range(16)
        ]
        solver = OverlapSolver(PipelineOverlapAlg())
        result, _ = solver.solve(costs, overlap_degree=degree)
        uniform, _ = OverlapSolver(UniformOverlapAlg()).solve(
            costs, overlap_degree=degree
        )
        self.assertEqual(
            sorted(i for part in result.partitions for i in part), list(range(17))
        )
        self.assertIn(0, result.partitions[0])
        self.assertTrue(all(any(i > 0 for i in part) for part in result.partitions))
        self.assertLessEqual(
            result.overall_cost, solver._pipeline_cost(costs, uniform.partitions)
        )
        repeated, _ = OverlapSolver(PipelineOverlapAlg()).solve(
            costs, overlap_degree=degree
        )
        self.assertEqual(result, repeated)

    def test_dynamic_degree_is_bounded_and_zero_costs_are_supported(self):
        costs = [OverlapStageCost() for _ in range(9)]
        result, candidates = OverlapSolver(PipelineOverlapAlg()).solve(
            costs, dynamic_max_degree=3
        )
        self.assertTrue(all(1 <= degree <= 3 for degree in candidates))
        self.assertEqual(result.overall_cost, 0)
        self.assertEqual(
            sorted(i for part in result.partitions for i in part), list(range(9))
        )


if __name__ == "__main__":
    unittest.main()
