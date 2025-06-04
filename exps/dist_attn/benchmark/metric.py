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

import math

from exps.dist_attn.benchmark.enums import MetricsType
from magi_attention.common import AttnRanges
from magi_attention.meta.container.bucket import AttnBucket

NHQ = 48
NHK = 8
HD = 128
COMP_SPEED = 989  # 989 TFLOPS/s
FWD_COMP_MFU = 0.68
BWD_COMP_MFU = 0.45
COMM_SPEED = 50  # 50 GB/s
COMM_MFU = 0.55


class MetricsCalculator:
    """This is a calculator for multiple metrics, it can be used with
    calculate function with metric type and load balance result.
    """

    def __init__(self):
        self.calculate_function = {
            MetricsType.RANGES_INFORMATION_ENTROPY: self.calculate_ranges_information_entropy,
            MetricsType.AREA_INFORMATION_ENTROPY: self.calculate_area_information_entropy,
            MetricsType.REMOTE_NORMALIZED_VALUE: self.calculate_remote_normalized_values,
            MetricsType.MAX_AREA_DIVIDED_BY_TOTAL_AREA: self.calculate_max_area_divided_by_total_area,
            MetricsType.MAX_AREA_DIVIDED_BY_AVERAGE_AREA: self.calculate_max_area_divided_by_average_area,
            MetricsType.AREA_GINI_IMPURITY: self.calculate_area_gini_impurity,
            MetricsType.RANGES_GIMI_IMPURITY: self.calculate_ranges_gini_impurity,
            MetricsType.COST_MODEL: self.calculate_cost_model,
        }

    def calculate(
        self,
        metrics_type: MetricsType,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        return self.calculate_function[metrics_type](
            buckets=buckets,
            total_seqlen=total_seqlen,
            chunk_size=chunk_size,
        )

    def calculate_area_information_entropy(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = [bucket.area for bucket in buckets]
        total_workload = sum(workloads)
        if total_workload == 0:
            return 0.0
        propertys = [workload / total_workload for workload in workloads]

        entropy = 0.0
        for property in propertys:
            entropy -= math.log2(property) * property
        entropy /= math.log2(len(workloads))

        return entropy

    def calculate_ranges_information_entropy(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = []
        for bucket in buckets:
            k_ranges = bucket.k_ranges.merge()
            q_ranges = AttnRanges.from_ranges(
                [
                    [
                        chunk.chunk_id * chunk_size,  # type: ignore
                        (chunk.chunk_id + 1) * chunk_size,  # type: ignore
                    ]
                    for chunk in bucket.q_chunks
                ]
            )
            remote_ranges = k_ranges.find_hole_ranges(
                other_attn_ranges=q_ranges,
                is_self_merged=True,
            )
            workloads.append(remote_ranges.total_seqlen)

        total_workload = sum(workloads)
        if total_workload == 0:
            return 0.0
        propertys = [
            workload / total_workload if workload > 0 else 0.0000001
            for workload in workloads
        ]
        # print(f"ranges: {workloads=} {total_workload=} {propertys=}")

        entropy = 0.0
        for property in propertys:
            entropy -= math.log2(property) * property
        entropy /= math.log2(len(workloads))

        return entropy

    def calculate_remote_normalized_values(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        cp_size = len(buckets)
        workloads = []
        for bucket in buckets:
            k_ranges = bucket.k_ranges.merge()
            q_ranges = AttnRanges.from_ranges(
                [
                    [
                        chunk.chunk_id * chunk_size,  # type: ignore
                        (chunk.chunk_id + 1) * chunk_size,  # type: ignore
                    ]
                    for chunk in bucket.q_chunks
                ]
            )
            remote_ranges = k_ranges.find_hole_ranges(
                other_attn_ranges=q_ranges,
                is_self_merged=True,
            )
            workloads.append(remote_ranges.total_seqlen)

        assert total_seqlen % cp_size == 0
        remote_workload = (cp_size - 1) * total_seqlen // cp_size
        if remote_workload == 0:
            return 0
        return max(workloads) / remote_workload

    def calculate_max_area_divided_by_total_area(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = [bucket.area for bucket in buckets]
        return max(workloads) / sum(workloads)

    def calculate_max_area_divided_by_average_area(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = [bucket.area for bucket in buckets]
        cp_size = len(workloads)
        return max(workloads) * cp_size / sum(workloads)

    def calculate_area_gini_impurity(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = [bucket.area for bucket in buckets]
        total_workload = sum(workloads)
        if total_workload == 0:
            return 0.0
        propertys = [workload / total_workload for workload in workloads]
        gimi_impurity = 1.0
        for property in propertys:
            gimi_impurity -= property * property

        return gimi_impurity

    def calculate_ranges_gini_impurity(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        workloads = []
        for bucket in buckets:
            k_ranges = bucket.k_ranges.merge()
            q_ranges = AttnRanges.from_ranges(
                [
                    [
                        chunk.chunk_id * chunk_size,  # type: ignore
                        (chunk.chunk_id + 1) * chunk_size,  # type: ignore
                    ]
                    for chunk in bucket.q_chunks
                ]
            )
            remote_ranges = k_ranges.find_hole_ranges(
                other_attn_ranges=q_ranges,
                is_self_merged=True,
            )
            workloads.append(remote_ranges.total_seqlen)

        total_workload = sum(workloads)
        if total_workload == 0:
            return 0.0
        propertys = [workload / total_workload for workload in workloads]
        gimi_impurity = 1.0
        for property in propertys:
            gimi_impurity -= property * property

        return gimi_impurity

    def calculate_cost_model(
        self,
        buckets: list[AttnBucket],
        total_seqlen: int,
        chunk_size: int,
    ) -> float:
        def calculate_cost_model_for_one_rank(
            areas: int,
            ranges: int,
            nhq: int,
            nhk: int,
            hd: int,
        ) -> float:
            comp_volumn = 4 * areas * nhq * hd
            fwd_comp_volumn, bwd_comp_volumn = comp_volumn, comp_volumn * 2.5
            fwd_comp_time = fwd_comp_volumn / (COMP_SPEED * 10**12 * FWD_COMP_MFU)
            bwd_comp_time = bwd_comp_volumn / (COMP_SPEED * 10**12 * BWD_COMP_MFU)

            comm_volumn = 4 * ranges * nhk * hd
            fwd_comm_volumn, bwd_comm_volumn = comm_volumn, comm_volumn * 2
            fwd_comm_time = fwd_comm_volumn / (COMM_SPEED * 2**30 * COMM_MFU)
            bwd_comm_time = bwd_comm_volumn / (COMM_SPEED * 2**30 * COMM_MFU)

            # DEBUG
            # print(f"{fwd_comp_time=} {fwd_comm_time=} {bwd_comp_time=} {bwd_comm_time=}")
            # if fwd_comp_time < fwd_comm_time or bwd_comp_time < bwd_comm_time:
            #     print("Shorter!")
            # else:
            #     print("Bigger!")

            return max(fwd_comm_time, fwd_comp_time) + max(bwd_comm_time, bwd_comp_time)

        cost = 0.0
        for bucket in buckets:
            areas = bucket.area
            k_ranges = bucket.k_ranges.merge()
            q_ranges = AttnRanges.from_ranges(
                [
                    [
                        chunk.chunk_id * chunk_size,  # type: ignore
                        (chunk.chunk_id + 1) * chunk_size,  # type: ignore
                    ]
                    for chunk in bucket.q_chunks
                ]
            )
            remote_ranges = k_ranges.find_hole_ranges(
                other_attn_ranges=q_ranges,
                is_self_merged=True,
            )
            ranges = remote_ranges.total_seqlen
            cost_this_rank = calculate_cost_model_for_one_rank(
                areas=areas,
                ranges=ranges,
                nhq=NHQ,
                nhk=NHK,
                hd=HD,
            )
            cost = max(cost, cost_this_rank)

        return cost
