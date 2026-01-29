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

import heapq
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cmp_to_key
from typing import Callable, TypeVar

import torch.nn as nn

from magi_attention.common import AttnRange, AttnRanges
from magi_attention.common.enum import DispatchAlgType
from magi_attention.utils import argmax, argmin, argsort, is_list_type_all


@dataclass(frozen=True)
class DispatchAlg(ABC):
    """The abstract config/meta info dataclass for specific dispatch algorithm"""

    @property
    @abstractmethod
    def type(self) -> DispatchAlgType:
        """The type enum of the dispatch algorithm"""

    @property
    @abstractmethod
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""

    @property
    @abstractmethod
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""

    @property
    @abstractmethod
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""

    @property
    @abstractmethod
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""


@dataclass(frozen=True)
class LBDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the lower-bound dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.LOWER_BOUND

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return False

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return False

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class DPDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the dynamic programming dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.DYNAMIC_PROGRAMMING

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return True

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return False

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return False

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class BSDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the binary search dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.BINARY_SEARCH

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return True

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return False

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class MinHeapDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the min-heap dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.MIN_HEAP

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class BTPDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the backtracing pruning dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.BACKTRACKING_PRUNING

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return True

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class ToppHeapDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the topp-heap dispatch algorithm"""

    top_p: float = 0.0

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.TOPP_HEAP

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return True


@dataclass(frozen=True)
class RandomSelectDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the topp-heap dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.RANDOM_SELECT

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class SequentialDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the topp-heap dispatch algorithm"""

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.SEQUENTIAL_SELECT

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class BatchToppHeapDispatchAlg(DispatchAlg):
    """The config/meta info dataclass for the batch topp-heap dispatch algorithm"""

    top_p: float = 0.0
    num_of_select_chunk: int = 1

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.BATCH_TOPP_HEAP

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return True


@dataclass(frozen=True)
class SortedSequentialSelectAlg(DispatchAlg):
    """The config/meta info dataclass for the batch topp-heap dispatch algorithm"""

    allocation_ratio: float = 1.0

    @property
    def type(self) -> DispatchAlgType:
        return DispatchAlgType.SORTED_SEQUENTIAL_SELECT

    @property
    def is_optimal(self) -> bool:
        """Whether the dispatch algorithm is optimal"""
        return False

    @property
    def is_partitions_returned(self) -> bool:
        """Whether the dispatch partitions are returned"""
        return True

    @property
    def is_equal_num_workloads(self) -> bool:
        """Whether the number of workloads of each bucket are equal"""
        return True

    @property
    def is_affinity_considered(self) -> bool:
        """Whether the affinity is considered"""
        return False


@dataclass(frozen=True)
class DispatchConfig:
    """The config dataclass for load-balanced dispatching"""

    alg: DispatchAlg = MinHeapDispatchAlg()

    def __post_init__(self):
        pass


T = TypeVar("T", bound="BaseDispatchAffinity")


class BaseDispatchAffinity(ABC):
    """The base abstract class for dispatch affinity
    which defines two abstract methods to be implemented by subclasses:
        1. distance_to: Measure the distance between two dispatch affinities
        2. __repr__: A string representation of the affinity
    """

    @abstractmethod
    def distance_to(self: T, other: T) -> float:
        """Measure the distance between two dispatch affinities"""

    @abstractmethod
    def update(self: T, other: T) -> None:
        """Update self affinity with other affinity in-place"""

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover
        """A string representation of the affinity"""

    def make_cmp_key(self: T) -> Callable:
        """Make a cmp key function
        for comparing two affinities's distance to self
        """

        def close_to_self_cmp_func(a: T, b: T) -> int:
            """NOTE: the cmp func rule:
            1. return negative when a < b
            2. return zero when a == b
            3. return positive when a > b
            """
            dist = self.distance_to(a) - self.distance_to(b)
            return -1 if dist < 0 else 1 if dist > 0 else 0

        return cmp_to_key(close_to_self_cmp_func)

    def get_closest_affinity_idx(self: T, others: list[T]) -> int:
        """Get the idx of the affinity in 'others' that is closest to self"""
        return argmin(others, key=self.make_cmp_key())

    def get_farthest_affinity_idx(self: T, others: list[T]) -> int:
        """Get the idx of the affinity in 'others' that is farthest to self"""
        return argmax(others, key=self.make_cmp_key())


class SampleIDAffinity(BaseDispatchAffinity):
    """SampleIDAffinity with the dict mapping each added sample id to its count"""

    def __init__(self):
        self.sample_id_cnt_dict = defaultdict(int)

    @staticmethod
    def from_list(ids: list[int]) -> "SampleIDAffinity":
        affinity = SampleIDAffinity()
        for id in ids:
            affinity.add_sample_id(id)
        return affinity

    def add_sample_id(self, sample_id: int) -> None:
        assert sample_id >= 0
        self.sample_id_cnt_dict[sample_id] += 1

    def get_count(self, sample_id: int) -> int:
        return self.sample_id_cnt_dict[sample_id]

    def get_sample_id_by(self, mode="max_count") -> int:
        assert not self.is_empty(), "No sample id has been added to this affinity yet."

        sample_id = -1
        if mode == "max_count":
            sample_id = max(self.sample_id_cnt_dict, key=self.sample_id_cnt_dict.get)
        elif mode == "min_count":
            sample_id = min(self.sample_id_cnt_dict, key=self.sample_id_cnt_dict.get)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return sample_id

    def num_sample_ids(self, distinct=True) -> int:
        if distinct:
            return len(self.sample_id_cnt_dict)
        return sum(self.sample_id_cnt_dict.values())

    def is_empty(self) -> bool:
        return len(self.sample_id_cnt_dict) == 0

    def distance_to(self, other: "SampleIDAffinity") -> float:
        """The distance of self affinity to other affinity is defined as:
        the negative of the counts in other affinity
        of the sample id with the max counts in self affinity
        """
        sample_id_with_max_count = self.get_sample_id_by("max_count")
        sample_id_count_in_other = other.get_count(sample_id_with_max_count)
        return -sample_id_count_in_other

    def update(self, other: "SampleIDAffinity") -> None:
        """Update self affinity with other affinity in-place"""
        for sample_id, count in other.sample_id_cnt_dict.items():
            self.sample_id_cnt_dict[sample_id] += count

    def __repr__(self) -> str:  # pragma: no cover
        sample_id_to_count = dict(self.sample_id_cnt_dict)
        return f"sample id affinity: {sample_id_to_count=}"


class IOUAffinity(BaseDispatchAffinity):
    """IOUAffinity with the ranges that have added to chunk or bucket"""

    def __init__(self):
        self.iou_ranges = AttnRanges()

    def append(self, attn_range: AttnRange):
        self.iou_ranges.append(attn_range)

    def extend(self, attn_ranges: AttnRanges):
        self.iou_ranges.extend(attn_ranges)

    @staticmethod
    def from_ranges(ranges: AttnRanges) -> "IOUAffinity":
        iou_affinity = IOUAffinity()
        iou_affinity.extend(ranges)

        return iou_affinity

    def distance_to(self, other: "IOUAffinity") -> float:
        """The distance to other affinity is defined as:
        the length of the intersection
        """
        intersect_size = self.iou_ranges.intersect_size_with(
            other=other.iou_ranges,
        )
        return -intersect_size

    def update(self, other: "IOUAffinity") -> None:
        """Update self affinity with other affinity in-place"""
        self.iou_ranges.extend(other.iou_ranges)

    def __repr__(self) -> str:  # pragma: no cover
        return f"iou affinity: ranges={self.iou_ranges}"


@dataclass
class DispatchJob:
    job_id: int

    workload: float = 0.0
    affinity: BaseDispatchAffinity | None = None

    def __post_init__(self):
        assert self.workload >= 0

    @staticmethod
    def from_job_list(
        workloads: list[float],
        affinities: list[BaseDispatchAffinity] | None = None,
    ) -> list["DispatchJob"]:
        if affinities is None:
            affinities = [None] * len(workloads)  # type: ignore[list-item]

        assert len(workloads) == len(affinities) > 0, "workloads should not be empty"
        assert is_list_type_all(
            affinities, just_same=True
        ), "The affinities of the jobs are not of the same type or all None."

        jobs = []
        for job_id, (workload, affinity) in enumerate(zip(workloads, affinities)):
            jobs.append(
                DispatchJob(
                    job_id=job_id,
                    workload=workload,
                    affinity=affinity,
                )
            )

        return jobs


@dataclass
class DispatchData:
    """The dispatch data dataclass, made of several data as follows:
    1. jobs (list[DispatchJob]): the list of jobs to be dispatched in balance
    2. num_buckets (int): the number of buckets, denoted as 'k' in any specific algorithm

    Optional:
    1. sample_areas (list[int]): the areas of each sample
    """

    jobs: list[DispatchJob]
    num_buckets: int

    sample_areas: list[int] | None = None


@dataclass
class DispatchSolution:
    """The dispatch solution dataclass, made of several info as follows:
    1. minimax_workload: the minimum maximum workload of all buckets.
    2. bucket_partitions: the partitions of the job workloads, a list with length `num_buckets`,
        each element of which is a list of job indices in the `jobs`,
        among them any two elements are mutually exclusive.
    """

    minimax_workload: float
    bucket_partitions: list[list[int]]


class DispatchSolver(nn.Module):
    """The implementation of the algorithms for balanced dispatching specified by 'alg'.
    NOTE: now, the object "balance" is formulized as:
        "minimize the maximum workload"
        which might not be mathematically equivalent due to discreteness,
        as well as other constraints, including:
            1. should the number of jobs in each bucket be exactly equal ?
            2. should the affinity be considered ?
    """

    def __init__(self, alg: DispatchAlg) -> None:
        super().__init__()

        self.alg = alg

        self.solve_func = {
            DispatchAlgType.LOWER_BOUND: self._solve_with_lb,
            DispatchAlgType.DYNAMIC_PROGRAMMING: self._solve_with_dp,
            DispatchAlgType.BINARY_SEARCH: self._solve_with_bs,
            DispatchAlgType.MIN_HEAP: self._solve_with_minhp,
            DispatchAlgType.TOPP_HEAP: self._solve_with_topphp,
            DispatchAlgType.BACKTRACKING_PRUNING: self._solve_with_btp,
            DispatchAlgType.RANDOM_SELECT: self._solve_with_random,
            DispatchAlgType.SEQUENTIAL_SELECT: self._solve_with_sequential,
            DispatchAlgType.BATCH_TOPP_HEAP: self._solve_with_batch_topphp,
            DispatchAlgType.SORTED_SEQUENTIAL_SELECT: self._solve_with_sorted_ss,
        }[self.alg.type]

        # tmp values
        self.minimax_workload: float = 0.0
        self.bucket_workloads: list[float] = []
        self.bucket_partitions: list[list[int]] = []

        # return values
        self.solution: DispatchSolution = None  # type: ignore

    def solve(
        self,
        dispatch_data: DispatchData,
    ) -> DispatchSolution:
        """Dispatch jobs with workloads and optional affinities to 'num_buckets' buckets
        to make the workload of each bucket as balanced as possible

        Args:
            jobs (list[DispatchJob]): the list of jobs to be dispatched in balance
            num_buckets (int): the number of buckets, denoted as 'k' in any specific algorithm
        Returns:
            minimax_workload (float): the maximum workload of any bucket
            bucket_workloads (list[float], optional): the workload of each bucket
            bucket_partitions (list[list[int]], optional): the partition of jobs,
                where each element is a list of job indices,
                among them any two elements are mutually exclusive
        """

        assert dispatch_data.num_buckets > 0, "num_buckets should be greater than 0"

        self.solve_func(
            dispatch_data=dispatch_data,
            **asdict(self.alg),
        )

        self.solution = DispatchSolution(
            minimax_workload=self.minimax_workload,
            bucket_partitions=self.bucket_partitions,
        )

        return self.solution

    def _solve_with_lb(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Lower bound for the job partition to minimize the maximum workload
        NOTE: this algorithm is just to get the lower bound of the answer, which might be invalid

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments
        """
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        self.minimax_workload = sum(workloads) / num_buckets

    def _solve_with_bs(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Binary search for the job partition to minimize the maximum workload
        NOTE: this algorithm has no constraint on the number of jobs dispatched in each bucket

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(nlogn + log(S−M) x k**n), where:
                - n is the length of the jobs array
                - S is the sum of all elements in the jobs array
                - M is the maximum element in the jobs array
                - log(S−M) is the number of possible partitions
                - k**n is the number of possible combinations of partitions
            - NOTE: this is the worst-case time complexity,
                but in practice it is often much faster due to search pruning

        Space complexity:
            - O(n), where:
                - n is the length of the jobs array
                - and the space is mainly used by the recursive stack
        """

        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # sort jobs in descending order
        sorted_indices = argsort(workloads, key=lambda x: -x)
        workloads = [workloads[i] for i in sorted_indices]

        # init the range of binary search
        left = workloads[0]  # left: the maximum size of a single job
        right = sum(workloads)  # right: the sum of all jobs

        while left < right:
            mid = (left + right) // 2
            if self._bs_check(
                workloads, num_buckets, mid
            ):  # check if mid is a valid solution
                right = mid  # if yes, reduce the range to the right half
            else:
                left = mid + 1  # if no, increase the range to the left half

        self.minimax_workload = left
        self.bucket_workloads = [
            sum(workloads[i] for i in p) for p in self.bucket_partitions
        ]
        self.bucket_partitions = [
            [sorted_indices[i] for i in p] for p in self.bucket_partitions
        ]

    def _solve_with_dp(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Dynamic programming for the job partition to minimize the maximum workload
        NOTE: this algorithm has no constraint on the number of jobs dispatched in each bucket,
        and it only returns the scalar answer of the maximum workload, w/o neither the partition or the workloads

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(k x 3**n), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - firstly of all, we need O(2**n) time to pre-process the sum array / init the dp array
                - then, we need to go through a double for-loop to do dp transition,
                    - the outer for-loop has O(k) iterations
                    - the inner for-loop has O(2**n) states, each state will tranverse all of its substates,
                    - so there will be ∑(f:0->n) C(n,f) * 2**f = (1+2)**n = 3**n iterations,
                        where f is the number of 1s in the state
                - therefore, the total iterations in the transition is O(k x 3**n)
                - finally, the total time complexity is O(k x 3**n + 2**n) = O(k x 3**n)

        Space complexity:
            - O(2**n), where:
                - n is the length of the jobs array
                - firstly of all, we need a sum array, which costs O(2**n)
                - and a dp array, which costs O(2**n) if we do the state compression
        """

        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # init dp array, where:
        # dp(i)[j]: the maximum workload if we assign the jobs to the first i buckets
        # when the state (the assigned jobs' bit-map) is j
        m = 1 << len(workloads)  # m = 2**n
        dp = [0.0] * m

        # init: dp(0)[j] = sum[j] = sum(jobs in j)
        for i, v in enumerate(workloads):
            bit = 1 << i
            for j in range(bit):
                dp[bit | j] = dp[j] + v
        sum = dp.copy()

        # dp transition, where:
        # dp(i)[j] = min(max(dp(i-1)[j ^ s], sum(jobs in s))) for each s,
        # where s is the substate of j, (j ^ s) is the rest state
        # which indicates that we assign the jobs in s to the i-th bucket,
        # thus assign the jobs in (j ^ s) to the first (i-1)-th bucket
        for i in range(1, num_buckets):
            for j in range(m - 1, 0, -1):
                s = j
                while s:  # for each substate s of j until s = 0
                    dp[j] = min(dp[j], max(dp[j ^ s], sum[s]))
                    s = (s - 1) & j  # transfer to the next substate

        # dp(k-1)[m-1] is the answer, since we already assign all jobs to k buckets
        self.minimax_workload = dp[-1]

        # NOTE: since there's no traceback, we can't recover the partition, nor the workload of each bucket
        # self.bucket_workloads = ...
        # self.bucket_partitions = ...

    def _solve_with_btp(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Backtrack pruning algorithm for the job partition to minimize the maximum workload,
        with the hard constraint that the number of jobs dispatched in each bucket should be the same
        NOTE: the solution of this algorithm is optimal when the number of jobs dispatched in each bucket is same

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments
        """

        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        # sort jobs in descending order
        sorted_indices = argsort(workloads, key=lambda x: -x)
        workloads = [workloads[i] for i in sorted_indices]

        # init the job partition and its workloads
        bucket_nums = [0] * num_buckets
        bucket_workloads = [0.0] * num_buckets
        bucket_partitions: list[list[int]] = [[] for _ in range(num_buckets)]

        # init best bucket and best minimax workload
        best_bucket: list[list[int]] = [[] for _ in range(num_buckets)]
        best_minimax_workload = float("inf")

        def backtrack(
            index: int,
            current_max: float,
        ):
            nonlocal best_minimax_workload, best_bucket

            # when index is same as workloads length, all workloads are allocated
            # update the workload and bucket according to the value of the current max.
            if index == n:
                if current_max < best_minimax_workload:
                    best_minimax_workload = current_max
                    best_bucket = [list(bucket) for bucket in bucket_partitions]
                return

            for i in range(num_buckets):
                # If the current bucket has already reached the limit quantity, continue.
                if bucket_nums[i] >= bucket_num_limit:
                    continue

                # If the workload at the index-th position is assigned to the i-th bucket,
                # calculate the sum of the numbers in this bucket.
                new_sum = bucket_workloads[i] + workloads[index]
                new_max = max(new_sum, current_max)

                # If the sum in the current bucket has already exceeded that of the current optimal solution, perform pruning.
                if new_max >= best_minimax_workload:
                    continue

                # Assign the workload at the index-th position to the i-th bucket.
                bucket_workloads[i] += workloads[index]
                bucket_nums[i] += 1
                bucket_partitions[i].append(index)

                # Continue the calculation of the workload at the (index + 1)-th position
                backtrack(index + 1, new_max)

                # Backtrack and remove the workload at the index-th position from the i-th bucket.
                bucket_workloads[i] -= workloads[index]
                bucket_nums[i] -= 1
                bucket_partitions[i].pop()

                # If, after backtracking, the length of the current bucket is 0,
                # it indicates that the lengths of all buckets following the current one are also 0.
                # Due to symmetry, perform pruning to avoid redundant calculations.
                if bucket_nums[i] == 0:
                    break

        # Execute the backtracking pruning algorithm.
        backtrack(0, 0)

        self.minimax_workload = best_minimax_workload
        self.bucket_partitions = [[sorted_indices[i] for i in p] for p in best_bucket]
        self.bucket_workloads = [sum(workloads[i] for i in p) for p in best_bucket]

    def _solve_with_minhp(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Greedy algorithm using a min-heap for the job partition to minimize the maximum workload,
        with the hard constraint that the number of jobs dispatched in each bucket should be the same
        NOTE: this algorithm is not guaranteed to find the optimal solution
        for example, if the jobs.workload = [8, 7, 6, 5, 4, 2, 2, 2] and num_buckets is 2,
        the answer in greedy algorithm is [8, 5, 4, 2] and [7, 6, 2, 2] with total number 19 and 17
        the optimal solution is [8, 6, 2, 2] and [7, 5, 4, 2] all with total number 18

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments

        Time complexity:
            - O(nlogn + nklogk + k), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - firstly of all, we need O(nlogn) time to sort the jobs array
                - then, we need O(k) time to init the min-heap
                - then, we need O(nklogk) time to do the greedy assignment for each job
                - since each assignment at most costs O(klogk) times of heap-push and O(klogk) times of heap-pop

        Space complexity:
            - O(n + k), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - the job array needs O(n) space
                - the min-heap and other auxiliary data needs O(k) space
        """
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        # sort jobs in descending order
        # in order to unify test cases, it is not directly sorted in descending order.
        sorted_indices = argsort(workloads)[::-1]
        workloads = [workloads[i] for i in sorted_indices]

        # init the job partition and its workloads
        bucket_nums = [0] * num_buckets
        self.bucket_workloads = [0] * num_buckets
        self.bucket_partitions = [[] for _ in range(num_buckets)]

        # init the min-heap with size of k
        # where each heap element is a tuple: (cur_workload, bucket_index)
        heap = []
        for bucket_idx in range(num_buckets):
            heap.append((0.0, bucket_idx))
        heapq.heapify(heap)

        # define a helper function to assign a job to a bucket
        # with their idxs and the new workload to update the heap
        def _assign_job_to_bucket(
            job_idx: int,
            bucket_idx: int,
            new_workload: float,
        ) -> None:
            # assign the job to this bucket
            self.bucket_partitions[bucket_idx].append(job_idx)
            self.bucket_workloads[bucket_idx] = new_workload
            bucket_nums[bucket_idx] += 1
            # push the updated sum of this bucket back into the heap
            heapq.heappush(heap, (new_workload, bucket_idx))

        # assign each job to a bucket
        for job_idx, job_workload in enumerate(workloads):
            while (
                heap
            ):  # find the bucket with the minimum workload that is not full to assign this job
                cur_workload, bucket_idx = heapq.heappop(heap)
                if bucket_nums[bucket_idx] < bucket_num_limit:
                    # assign the job to this bucket
                    _assign_job_to_bucket(
                        job_idx=job_idx,
                        bucket_idx=bucket_idx,
                        new_workload=cur_workload + job_workload,
                    )
                    break
            else:  # if no bucket is found for this job (which shouldn't happen), raise an error
                raise RuntimeError(f"No bucket is found for the job {job_idx}.")

        self.minimax_workload = max(self.bucket_workloads)
        self.bucket_partitions = [
            [sorted_indices[i] for i in p] for p in self.bucket_partitions
        ]

    def _solve_with_topphp(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        """Greedy algorithm using a top-p min-heap for the job partition to minimize the maximum workload,
        with the constraints:
            1. the number of jobs dispatched in each bucket should be the same (hard)
            2. the affinity of the jobs in each bucket should be as close as possible (soft)
        NOTE: this algorithm is not guaranteed to find the optimal solution

        Args:
            dispatch_data (DispatchData): data config with dispatch params
            kwargs (dict | None): additional arguments, including:
                1. top_p (float, optional): the top-p value, ranging from [0., 1.]
                NOTE: this arg denotes the ratio of the utmost number of jobs fetched from the top of the min-heap each time
                more specifically, the utmost fetched job number is: 'max(1, ⌈num_buckets * top_p⌉)'

        Time complexity:
            - O(nlogn + mnklogk + k), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - m = max(1, ⌈num_buckets * top_p⌉) is the utmost number of jobs fetched from the top of the min-heap each time
                - firstly of all, we need O(nlogn) time to sort the jobs array
                - then, we need O(k) time to init the min-heap
                - then, we need O(mnklogk) time to do the greedy assignment for each job
                - since each assignment at most costs O(mklogk) times of heap-push and O(mklogk) times of heap-pop

        Space complexity:
            - O(n + k + m), where:
                - n is the length of the jobs array
                - k is the number of buckets
                - m = max(1, ⌈num_buckets * top_p⌉) is the utmost number of jobs fetched from the top of the min-heap each time
                - the job array needs O(n) space
                - the min-heap and other auxiliary data needs O(k) space
                - the affinity array needs O(m) space
        """

        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        top_p: float = kwargs.get("top_p", 0.0)
        assert 0.0 <= top_p <= 1.0

        # calcuate m
        m = max(1, math.ceil(num_buckets * top_p))

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        # get the affinity of each job
        affinities = [job.affinity for job in jobs if job.affinity is not None]
        assert (
            affinities
        ), "For top-p min-heap algorithm, we need the affinity for each job to be explicitly set."
        affinity_class = type(
            affinities[0]
        )  # all the affinity should be of the same type, checked in self.solve() already

        # sort workloads and affinities in descending order
        sorted_indices = argsort(workloads, key=lambda x: -x)
        workloads = [workloads[i] for i in sorted_indices]
        affinities = [affinities[i] for i in sorted_indices]

        # init the job partition and its workloads
        bucket_nums = [0] * num_buckets
        self.bucket_workloads = [0] * num_buckets
        self.bucket_partitions = [[] for _ in range(num_buckets)]

        # init the min-heap with size of k
        # where each heap element is a tuple: (cur_workload, bucket_index, bucket_affinity)
        heap = []
        for bucket_idx in range(num_buckets):
            heap.append((0.0, bucket_idx, affinity_class()))
        heapq.heapify(heap)

        # define a helper function to assign a job to a bucket
        # with their idxs and the new workload to update the heap
        def _assign_job_to_bucket(
            job_idx: int,
            bucket_idx: int,
            new_workload: float,
            new_affinity: BaseDispatchAffinity,
        ) -> None:
            # assign the job to this bucket
            self.bucket_partitions[bucket_idx].append(job_idx)
            self.bucket_workloads[bucket_idx] = new_workload
            bucket_nums[bucket_idx] += 1
            # push the updated sum of this bucket back into the heap
            heapq.heappush(heap, (new_workload, bucket_idx, new_affinity))

        # assign each job to a bucket
        for job_idx, (job_workload, job_affinity) in enumerate(
            zip(workloads, affinities)
        ):
            top_p_bucket_idxs: list[int] = []
            top_p_bucket_affinities: list[BaseDispatchAffinity] = []
            top_p_bucket_cur_workloads: list[float] = []
            while (
                heap
            ):  # find the top-p buckets with the minimum workload that is not full to assign this job
                cur_workload, bucket_idx, bucket_affinity = heapq.heappop(heap)
                if bucket_nums[bucket_idx] < bucket_num_limit:
                    top_p_bucket_idxs.append(bucket_idx)
                    top_p_bucket_affinities.append(bucket_affinity)
                    top_p_bucket_cur_workloads.append(cur_workload)

                if len(top_p_bucket_idxs) == m:
                    break

            # if no bucket is found for this job (which shouldn't happen), raise an error
            if len(top_p_bucket_idxs) == 0:
                raise RuntimeError(f"No bucket is found for the job {job_idx}.")

            # find the bucket whose affinity is closest to job affinity among top-p ones
            closest_topp_idx = job_affinity.get_closest_affinity_idx(
                top_p_bucket_affinities
            )

            # update this closest bucket
            closest_bucket_affinity = top_p_bucket_affinities[closest_topp_idx]
            closest_bucket_affinity.update(job_affinity)

            # assign the job to this closest bucket
            _assign_job_to_bucket(
                job_idx=job_idx,
                bucket_idx=top_p_bucket_idxs[closest_topp_idx],
                new_workload=top_p_bucket_cur_workloads[closest_topp_idx]
                + job_workload,
                new_affinity=closest_bucket_affinity,
            )

            # push the other top-p buckets back into heap w/o updating
            for topp_idx, (cur_workload, bucket_idx, bucket_affinity) in enumerate(
                zip(
                    top_p_bucket_cur_workloads,
                    top_p_bucket_idxs,
                    top_p_bucket_affinities,
                )
            ):
                if topp_idx != closest_topp_idx:
                    heapq.heappush(heap, (cur_workload, bucket_idx, bucket_affinity))

        self.minimax_workload = max(self.bucket_workloads)
        self.bucket_partitions = [
            [sorted_indices[i] for i in p] for p in self.bucket_partitions
        ]

    def _solve_with_random(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        chunk_idx = list(range(len(jobs)))
        random.shuffle(chunk_idx)

        self.bucket_partitions = [
            chunk_idx[i * bucket_num_limit : (i + 1) * bucket_num_limit]
            for i in range(num_buckets)
        ]

        workload_per_bucket = [
            [workloads[idx] for idx in partition]
            for partition in self.bucket_partitions
        ]
        self.minimax_workload = max([sum(workload) for workload in workload_per_bucket])

    def _solve_with_sequential(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        chunk_idx = list(range(len(jobs)))

        self.bucket_partitions = [
            chunk_idx[i * bucket_num_limit : (i + 1) * bucket_num_limit]
            for i in range(num_buckets)
        ]

        workload_per_bucket = [
            [workloads[idx] for idx in partition]
            for partition in self.bucket_partitions
        ]
        self.minimax_workload = max([sum(workload) for workload in workload_per_bucket])

    def _solve_with_batch_topphp(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        top_p: float = kwargs.get("top_p", 0.0)
        num_of_select_chunk: int = kwargs.get("num_of_select_chunk", 1)
        assert 0.0 <= top_p <= 1.0
        assert num_of_select_chunk >= 1

        # calcuate m
        m = max(1, math.ceil(num_buckets * top_p))

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        # get the affinity of each job
        affinities = [job.affinity for job in jobs if job.affinity is not None]
        assert (
            affinities
        ), "For top-p min-heap algorithm, we need the affinity for each job to be explicitly set."
        affinity_class = type(
            affinities[0]
        )  # all the affinity should be of the same type, checked in self.solve() already

        # sort workloads and affinities in descending order
        sorted_indices = argsort(workloads, key=lambda x: -x)
        workloads = [workloads[i] for i in sorted_indices]
        affinities = [affinities[i] for i in sorted_indices]

        # init the job partition and its workloads
        bucket_nums = [0] * num_buckets
        self.bucket_workloads = [0] * num_buckets
        self.bucket_partitions = [[] for _ in range(num_buckets)]

        # init the min-heap with size of k
        # where each heap element is a tuple: (cur_workload, bucket_index, bucket_affinity)
        heap = []
        for bucket_idx in range(num_buckets):
            heap.append((0.0, bucket_idx, affinity_class()))
        heapq.heapify(heap)

        bucket_iteration_times = (
            len(workloads) + num_of_select_chunk - 1
        ) // num_of_select_chunk

        # define a helper function to assign a job to a bucket
        # with their idxs and the new workload to update the heap
        def _assign_job_to_bucket(
            job_idx: int,
            bucket_idx: int,
            new_workload: float,
            new_affinity: BaseDispatchAffinity,
        ) -> None:
            # assign the job to this bucket
            self.bucket_partitions[bucket_idx].append(job_idx)
            self.bucket_workloads[bucket_idx] = new_workload
            bucket_nums[bucket_idx] += 1

        for iteration in range(bucket_iteration_times):
            # print(f"In interation {iteration}")
            job_idx_start, job_idx_end = (
                iteration * num_of_select_chunk,
                min((iteration + 1) * num_of_select_chunk, len(workloads)),
            )

            top_p_bucket_idxs: list[int] = []
            top_p_bucket_affinities: list[BaseDispatchAffinity] = []
            top_p_bucket_cur_workloads: list[float] = []
            num_of_chunks_can_store: int = 0
            while (
                heap
            ):  # find the top-p buckets with the minimum workload that is not full to assign this job
                cur_workload, bucket_idx, bucket_affinity = heapq.heappop(heap)
                if bucket_nums[bucket_idx] < bucket_num_limit:
                    top_p_bucket_idxs.append(bucket_idx)
                    top_p_bucket_affinities.append(bucket_affinity)
                    top_p_bucket_cur_workloads.append(cur_workload)
                    num_of_chunks_can_store += (
                        bucket_num_limit - bucket_nums[bucket_idx]
                    )

                if (
                    len(top_p_bucket_idxs) == m
                    and num_of_chunks_can_store >= num_of_select_chunk
                ):
                    break

            # if no bucket is found for this job (which shouldn't happen), raise an error
            if len(top_p_bucket_idxs) == 0:
                raise RuntimeError(f"No bucket is found for the job when {bucket_idx=}")

            for job_idx in range(job_idx_start, job_idx_end):
                (job_workload, job_affinity) = (workloads[job_idx], affinities[job_idx])

                # print(f"{job_idx=} {job_affinity=} {top_p_bucket_affinities=}")
                closest_topp_idx = job_affinity.get_closest_affinity_idx(
                    top_p_bucket_affinities
                )

                # update this closest bucket
                closest_bucket_affinity = top_p_bucket_affinities[closest_topp_idx]
                closest_bucket_affinity.update(job_affinity)
                top_p_bucket_cur_workloads[closest_topp_idx] += job_workload

                # assign the job to this closest bucket
                _assign_job_to_bucket(
                    job_idx=job_idx,
                    bucket_idx=top_p_bucket_idxs[closest_topp_idx],
                    new_workload=top_p_bucket_cur_workloads[closest_topp_idx],
                    new_affinity=closest_bucket_affinity,
                )

                if bucket_nums[top_p_bucket_idxs[closest_topp_idx]] >= bucket_num_limit:
                    heapq.heappush(
                        heap,
                        (
                            top_p_bucket_cur_workloads[closest_topp_idx],
                            top_p_bucket_idxs[closest_topp_idx],
                            closest_bucket_affinity,
                        ),
                    )
                    top_p_bucket_idxs.pop(closest_topp_idx)
                    top_p_bucket_affinities.pop(closest_topp_idx)
                    top_p_bucket_cur_workloads.pop(closest_topp_idx)

            # push the other top-p buckets back into heap w/o updating
            for topp_idx, (cur_workload, bucket_idx, bucket_affinity) in enumerate(
                zip(
                    top_p_bucket_cur_workloads,
                    top_p_bucket_idxs,
                    top_p_bucket_affinities,
                )
            ):
                heapq.heappush(heap, (cur_workload, bucket_idx, bucket_affinity))

        self.minimax_workload = max(self.bucket_workloads)
        self.bucket_partitions = [
            [sorted_indices[i] for i in p] for p in self.bucket_partitions
        ]

    def _solve_with_sorted_ss(
        self,
        dispatch_data: DispatchData,
        **kwargs,
    ) -> None:
        # get dispatch jobs from data config
        jobs: list[DispatchJob] = dispatch_data.jobs
        num_buckets: int = dispatch_data.num_buckets

        # get allocation ratio from dispatch alg
        allocation_ratio = kwargs.get("allocation_ratio", 1.0)
        assert 0.0 <= allocation_ratio <= 1.0

        sample_areas: list[int] = dispatch_data.sample_areas  # type: ignore
        assert (
            sample_areas is not None and len(sample_areas) > 0
        ), f"sorted sequential select solver need areas of each sample, but got {sample_areas=}"

        sample_ids_sorted = argsort(sample_areas, key=lambda x: -x)

        # get the workload of each job
        workloads = [job.workload for job in jobs]

        # check the job number constraint
        n = len(workloads)
        assert (
            n % num_buckets == 0
        ), f"The number of jobs ({n}) should be divisible by k ({num_buckets}) for this algorithm."
        bucket_num_limit = n // num_buckets

        # get the affinity of each job and assert check
        affinities: list[SampleIDAffinity] = [
            job.affinity for job in jobs if isinstance(job.affinity, SampleIDAffinity)
        ]
        assert (
            affinities
        ), "For sorted sequential select algorithm, we need the affinity for each job to be explicitly set."
        affinity_class = type(
            affinities[0]
        )  # all the affinity should be of the same type, checked in self.solve() already
        assert affinity_class == SampleIDAffinity
        assert len(affinities) == len(
            workloads
        ), "class of affinity for sorted sequential select must be all SampleIDAffinity."

        sample_to_slice_map: defaultdict[int, list[int]] = defaultdict(list)
        empty_chunk_idx: list[int] = []

        # calculate samples are chunked to which chunk and empty chunk
        for chunk_idx, affinity in enumerate(affinities):
            if len(affinity.sample_id_cnt_dict) == 0:
                empty_chunk_idx.append(chunk_idx)
                continue
            for sample_idx, _ in affinity.sample_id_cnt_dict.items():
                sample_to_slice_map[sample_idx].append(chunk_idx)

        # initialize heap to store bucket params
        heap = []
        for bucket_idx in range(num_buckets):
            heap.append((0.0, bucket_idx, affinity_class()))
        heapq.heapify(heap)

        def _assign_job_to_bucket(
            job_idx: int,
            bucket_idx: int,
            new_workload: float,
        ) -> None:
            # assign the job to this bucket
            self.bucket_partitions[bucket_idx].append(job_idx)
            self.bucket_workloads[bucket_idx] = new_workload
            bucket_nums[bucket_idx] += 1

        # init the job partition and its workloads
        bucket_nums = [0] * num_buckets
        self.bucket_workloads = [0] * num_buckets
        self.bucket_partitions = [[] for _ in range(num_buckets)]
        chunk_has_select = [False] * len(workloads)

        for sample_id in sample_ids_sorted:
            # get chunk_ids will be handled
            chunk_ids_list = sorted(sample_to_slice_map[sample_id])[::-1]
            max_chunks_per_allocation = max(
                1, math.ceil(allocation_ratio * len(chunk_ids_list))
            )
            cur_workload, bucket_idx, affinity = heapq.heappop(heap)
            chunks_assigned_per_allocation = 0
            for chunk_id in chunk_ids_list:
                # skip chunk has assigned
                if chunk_has_select[chunk_id]:
                    continue

                # assign chunk and calculate bucket workload
                cur_workload = cur_workload + workloads[chunk_id]
                _assign_job_to_bucket(
                    job_idx=chunk_id,
                    bucket_idx=bucket_idx,
                    new_workload=cur_workload,
                )
                chunks_assigned_per_allocation += 1
                chunk_has_select[chunk_id] = True

                # get new bucket if current bucket is full
                if bucket_nums[bucket_idx] == bucket_num_limit and len(heap) > 0:
                    cur_workload, bucket_idx, affinity = heapq.heappop(heap)
                    chunks_assigned_per_allocation = 0
                # push current bucket and get new bucket if assigned chunks exceed max chunks per allocation
                elif (
                    allocation_ratio < 1.0
                    and chunks_assigned_per_allocation == max_chunks_per_allocation
                ):
                    heapq.heappush(heap, (cur_workload, bucket_idx, affinity))
                    cur_workload, bucket_idx, affinity = heapq.heappop(heap)
                    chunks_assigned_per_allocation = 0

            if bucket_nums[bucket_idx] < bucket_num_limit:
                heapq.heappush(heap, (cur_workload, bucket_idx, affinity))

            if len(heap) == 0:
                break

        # assign empty chunks
        empty_chunk_idx.sort()
        empty_idx, empty_chunks_wait_to_assign = 0, len(empty_chunk_idx)
        while len(heap) > 0:
            cur_workload, bucket_idx, affinity = heapq.heappop(heap)
            # calculate how many chunks need to be allocated
            chunks_need_to_assign = bucket_num_limit - bucket_nums[bucket_idx]
            # the remaining number of empty chunks must be less than the number to be allocated
            assert chunks_need_to_assign <= empty_chunks_wait_to_assign
            empty_chunks_wait_to_assign -= chunks_need_to_assign
            # assign chunks
            for _ in range(chunks_need_to_assign):
                _assign_job_to_bucket(
                    job_idx=empty_chunk_idx[empty_idx],
                    bucket_idx=bucket_idx,
                    new_workload=cur_workload,
                )
                empty_idx += 1

        assert (
            len(heap) == 0
            and empty_chunks_wait_to_assign == 0
            and empty_idx == len(empty_chunk_idx)
        )
        for num in bucket_nums:
            assert num == bucket_num_limit

        self.minimax_workload = max(self.bucket_workloads)

    def _bs_backtrack(self, workloads: list[float], idx: int, limit: float) -> bool:
        """Inner function of '_solve_with_bs',
        to assign the current job with index 'idx' to some bucket whose workload does not exceed 'limit',
        and then recursively assign the remaining jobs,
        until all jobs have been assigned or no solution is found

        Args:
            workloads (list[float]): the workload of each job
            idx (int): the index of the current job
            limit (float): the maximum workload of each bucket
        Returns:
            bool: whether a solution is found
        """

        if idx >= len(workloads):
            return True  # all jobs have been assigned

        current_job_workload = workloads[idx]
        for partition in self.bucket_partitions:
            partition_sum = sum(workloads[idx] for idx in partition)

            if partition_sum + current_job_workload <= limit:
                partition.append(idx)
                if self._bs_backtrack(workloads, idx + 1, limit):
                    return True
                partition.pop()

            # if the current bucket has not been assigned any work,
            # or assigning the current work exactly reaches the limit,
            # there is no need to try more buckets
            if partition_sum == 0 or partition_sum + current_job_workload == limit:
                break

        return False

    def _bs_check(self, workloads: list[float], k: int, limit: float) -> bool:
        """Inner function of '_solve_with_bs',
        to check if it is possible to assign all jobs to k buckets,
        such that the workload of each bucket does not exceed 'limit'

        Args:
            jobs (list[float]): the workload of each job
            k (int): the number of buckets
            limit (float): the maximum workload of each bucket
        Returns:
            bool: whether a solution is found
        """
        self.bucket_partitions = [[] for _ in range(k)]
        return self._bs_backtrack(workloads, 0, limit)
