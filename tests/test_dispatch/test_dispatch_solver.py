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

from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.solver.dispatch_solver import (
    BSDispatchAlg,
    DispatchData,
    DispatchJob,
    DispatchSolver,
    DPDispatchAlg,
    IOUAffinity,
    LBDispatchAlg,
    MinHeapDispatchAlg,
    SampleIDAffinity,
    SortedSequentialSelectAlg,
    ToppHeapDispatchAlg,
)
from magi_attention.testing import parameterize

WORLD_SIZE = 4
SEED = 42


class TestDispatchSolver(TestCase):
    def test_solve_with_lb(self):
        # --------------      init dispatch solver       -------------- #

        lb_alg = LBDispatchAlg()
        self.assertFalse(lb_alg.is_optimal)
        self.assertFalse(lb_alg.is_equal_num_workloads)
        self.assertFalse(lb_alg.is_partitions_returned)
        self.assertFalse(lb_alg.is_affinity_considered)
        lb_solver = DispatchSolver(alg=lb_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [2, 4, 3, 16, 13, 5, 9, 10]
        num_buckets = 4

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
        )

        # --------------      solve       -------------- #

        solution = lb_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #

        self.assertTrue(
            solution.minimax_workload == sum(job_workloads) / num_buckets == 15.5
        )
        self.assertTrue(len(solution.bucket_partitions) == 0)

    def test_solve_with_dp_and_bs(self):
        # --------------      init dispatch solver       -------------- #

        lb_solver = DispatchSolver(alg=LBDispatchAlg())  # baseline

        dp_alg = DPDispatchAlg()
        self.assertTrue(dp_alg.is_optimal)
        self.assertFalse(dp_alg.is_equal_num_workloads)
        self.assertFalse(dp_alg.is_partitions_returned)
        self.assertFalse(dp_alg.is_affinity_considered)
        dp_solver = DispatchSolver(alg=dp_alg)

        bs_alg = BSDispatchAlg()
        self.assertTrue(bs_alg.is_optimal)
        self.assertFalse(bs_alg.is_equal_num_workloads)
        self.assertTrue(bs_alg.is_partitions_returned)
        self.assertFalse(bs_alg.is_affinity_considered)
        bs_solver = DispatchSolver(alg=bs_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [2, 4, 3, 16, 13, 5, 9, 10]
        num_buckets = 4

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
        )

        # --------------      solve       -------------- #

        lb_solution = lb_solver.solve(
            dispatch_data=dispatch_data,
        )
        dp_solution = dp_solver.solve(
            dispatch_data=dispatch_data,
        )
        bs_solution = bs_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #

        self.assertTrue(
            dp_solution.minimax_workload
            == bs_solution.minimax_workload
            == 16
            >= lb_solution.minimax_workload
        )
        self.assertTrue(len(dp_solution.bucket_partitions) == 0)
        self.assertTrue(len(bs_solution.bucket_partitions) == num_buckets)

    def test_solve_with_minhp(self):
        # --------------      init dispatch solver       -------------- #

        bs_solver = DispatchSolver(alg=BSDispatchAlg())  # baseline

        minhp_alg = MinHeapDispatchAlg()
        self.assertFalse(minhp_alg.is_optimal)
        self.assertTrue(minhp_alg.is_equal_num_workloads)
        self.assertTrue(minhp_alg.is_partitions_returned)
        self.assertFalse(minhp_alg.is_affinity_considered)
        minhp_solver = DispatchSolver(alg=minhp_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [2, 4, 3, 16, 13, 5, 9, 10]
        num_buckets = 4
        num_jobs_in_each_bucket = len(job_workloads) // num_buckets

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
        )

        # --------------      solve       -------------- #

        bs_solution = bs_solver.solve(
            dispatch_data=dispatch_data,
        )
        minhp_solution = minhp_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #

        self.assertTrue(
            minhp_solution.minimax_workload == 18 >= bs_solution.minimax_workload
        )
        self.assertTrue(len(minhp_solution.bucket_partitions) == num_buckets)
        self.assertTrue(
            all(
                len(p) == num_jobs_in_each_bucket
                for p in minhp_solution.bucket_partitions
            )
        )

    def test_solve_with_topphp_and_sample_id(self):
        # --------------      init dispatch solver       -------------- #

        minhp_solver = DispatchSolver(alg=MinHeapDispatchAlg())  # baseline

        topphp_alg = ToppHeapDispatchAlg(top_p=0.5)
        self.assertFalse(topphp_alg.is_optimal)
        self.assertTrue(topphp_alg.is_equal_num_workloads)
        self.assertTrue(topphp_alg.is_partitions_returned)
        self.assertTrue(topphp_alg.is_affinity_considered)
        topphp_solver = DispatchSolver(alg=topphp_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [2, 4, 3, 16, 13, 5, 9, 10]
        job_affinities = []
        for i in range(len(job_workloads)):
            job_affinity = SampleIDAffinity()
            job_affinity.add_sample_id(i)
            job_affinities.append(job_affinity)  # each job contains a unique sample id

        num_buckets = 4
        num_jobs_in_each_bucket = len(job_workloads) // num_buckets

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,
            affinities=job_affinities,
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
        )

        # --------------      solve       -------------- #

        minhp_solution = minhp_solver.solve(
            dispatch_data=dispatch_data,
        )
        topphp_solution = topphp_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #

        self.assertTrue(
            topphp_solution.minimax_workload == 18 >= minhp_solution.minimax_workload
        )
        self.assertTrue(len(topphp_solution.bucket_partitions) == num_buckets)
        self.assertTrue(
            all(
                len(p) == num_jobs_in_each_bucket
                for p in topphp_solution.bucket_partitions
            )
        )

    def test_solve_with_topphp_and_iou(self):
        # --------------      init dispatch solver       -------------- #

        minhp_solver = DispatchSolver(alg=MinHeapDispatchAlg())  # baseline

        topphp_alg = ToppHeapDispatchAlg(top_p=0.5)
        self.assertFalse(topphp_alg.is_optimal)
        self.assertTrue(topphp_alg.is_equal_num_workloads)
        self.assertTrue(topphp_alg.is_partitions_returned)
        self.assertTrue(topphp_alg.is_affinity_considered)
        topphp_solver = DispatchSolver(alg=topphp_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [2, 4, 3, 16, 13, 5, 9, 10]
        job_affinities = []
        attn_ranges = AttnRanges.from_ranges(
            [
                [0, 2],
                [1, 6],
                [4, 7],
                [7, 15],
                [11, 22],
                [19, 23],
                [23, 30],
                [27, 32],
            ]
        )
        for i in range(len(job_workloads)):
            job_affinity = IOUAffinity()
            job_affinity.append(attn_ranges[i])
            job_affinities.append(job_affinity)  # each job contains a range

        num_buckets = 4
        num_jobs_in_each_bucket = len(job_workloads) // num_buckets

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,
            affinities=job_affinities,
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
        )

        # --------------      solve       -------------- #

        minhp_solution = minhp_solver.solve(
            dispatch_data=dispatch_data,
        )
        topphp_solution = topphp_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #

        self.assertTrue(
            topphp_solution.minimax_workload == 19 >= minhp_solution.minimax_workload
        )
        self.assertTrue(len(topphp_solution.bucket_partitions) == num_buckets)
        self.assertTrue(
            all(
                len(p) == num_jobs_in_each_bucket
                for p in topphp_solution.bucket_partitions
            )
        )
        self.assertTrue(
            topphp_solution.bucket_partitions == [[3, 0], [4, 5], [7, 6], [1, 2]]
        )

    @parameterize(
        "testcase",
        [
            [1.0, 32],
            [0.5, 28],
            [0.2, 32],
            [0.0, 28],
        ],
    )
    def test_solve_with_sorted_sequential(self, testcase: tuple[float, int]):
        # --------------      init dispatch solver       -------------- #

        minhp_solver = DispatchSolver(alg=MinHeapDispatchAlg())  # baseline

        allocation_ratio, minimax_workload = testcase
        sorted_ss_alg = SortedSequentialSelectAlg(allocation_ratio=allocation_ratio)
        self.assertFalse(sorted_ss_alg.is_optimal)
        self.assertTrue(sorted_ss_alg.is_equal_num_workloads)
        self.assertTrue(sorted_ss_alg.is_partitions_returned)
        self.assertFalse(sorted_ss_alg.is_affinity_considered)
        sorted_ss_solver = DispatchSolver(alg=sorted_ss_alg)

        # --------------      init jobs      -------------- #

        job_workloads = [8, 8, 8, 8, 8, 8, 4, 4]
        sample_id_affinity_list = [
            [0],
            [0],
            [0],
            [0],
            [0],
            [0, 1],
            [1],
            [1],
        ]
        sample_areas = [48, 8]
        job_affinities = []
        for i in range(len(job_workloads)):
            job_affinity = SampleIDAffinity.from_list(sample_id_affinity_list[i])
            job_affinities.append(job_affinity)  # each job contains a unique sample id

        num_buckets = 2
        num_jobs_in_each_bucket = len(job_workloads) // num_buckets

        jobs = DispatchJob.from_job_list(
            workloads=job_workloads,  # type: ignore
            affinities=job_affinities,  # type: ignore
        )

        dispatch_data: DispatchData = DispatchData(
            jobs=jobs,
            num_buckets=num_buckets,
            sample_areas=sample_areas,
        )

        # --------------      solve       -------------- #

        minhp_solution = minhp_solver.solve(
            dispatch_data=dispatch_data,
        )
        sorted_ss_solution = sorted_ss_solver.solve(
            dispatch_data=dispatch_data,
        )

        # --------------      check       -------------- #
        self.assertTrue(
            sorted_ss_solution.minimax_workload
            == minimax_workload
            >= minhp_solution.minimax_workload
        )
        self.assertTrue(len(sorted_ss_solution.bucket_partitions) == num_buckets)
        self.assertTrue(
            all(
                len(p) == num_jobs_in_each_bucket
                for p in sorted_ss_solution.bucket_partitions
            )
        )


if __name__ == "__main__":
    unittest.main()
