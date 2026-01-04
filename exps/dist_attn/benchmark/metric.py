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

import math
from dataclasses import dataclass
from typing import Optional

import torch

from exps.dist_attn.benchmark.enums import MetricsType
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta import make_global_bucket_from_qk_ranges
from magi_attention.meta.collection.calc_meta import CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta
from magi_attention.meta.container.bucket import AttnBucket


def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    head_dim: int,
) -> dict[str, float]:
    attn_area = make_global_bucket_from_qk_ranges(
        q_ranges,
        k_ranges,
        attn_mask_type,
        num_chunks=1,
        chunk_size=total_seqlen_q,
    ).area

    flops_fwd = 4 * attn_area * num_heads_q * head_dim
    flops_bwd = flops_fwd * 2.5  # 2.0(bwd) + 0.5(recompute)
    flops_1f1b = flops_fwd + flops_bwd

    return {
        "fwd": flops_fwd,
        "bwd": flops_bwd,
        "1f1b": flops_1f1b,
    }


NHQ = 48
NHK = 8
HD = 128
COMP_SPEED = 989  # 989 TFLOPS/s
FWD_COMP_MFU = 0.68
BWD_COMP_MFU = 0.45
COMM_SPEED = 50  # 50 GB/s
COMM_MFU = 0.55


@dataclass
class MetricData:
    comm_meta_list: list[CommMeta]
    calc_meta_list: list[CalcMeta]

    q_heads: Optional[int] = None
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    pass_type: Optional[str] = "fwd"

    fwd_cast_dtype: Optional[torch.dtype] = None  # fwd q, k, v
    fwd_reduce_dtype: Optional[torch.dtype] = None  # fwd o
    bwd_cast_dtype: Optional[torch.dtype] = None  # bwd q, k, v, o, do
    bwd_reduce_dtype: Optional[torch.dtype] = None  # bwd dq, dk, dv
    # Note: lse is fixed torch.float32


@dataclass
class MetricSet:
    computation_amount_list: Optional[list[list[float]]] = None
    comm_bytes_list: Optional[list[list[tuple[int, int]]]] = None


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


class MetricDataCalculator:
    @staticmethod
    def calculate(
        metrics_type: MetricsType,
        metric_data: MetricData,
    ) -> MetricSet:
        assert metrics_type in [MetricsType.COMPUTATION_AMOUNT, MetricsType.COMM_BYTES]

        calculate_function = {
            MetricsType.COMPUTATION_AMOUNT: MetricDataCalculator.calculate_computational_cost,
            MetricsType.COMM_BYTES: MetricDataCalculator.calculate_communication_cost,
        }[metrics_type]

        return calculate_function(metric_data=metric_data)

    @staticmethod
    def calculate_computational_cost(
        metric_data: MetricData,
    ) -> MetricSet:
        calc_meta_list: list[CalcMeta] = metric_data.calc_meta_list
        num_heads_q, head_dim, pass_type = (
            metric_data.q_heads,
            metric_data.head_dim,
            metric_data.pass_type,
        )
        assert num_heads_q is not None and head_dim is not None, (
            f"Need q_heads and head_dim to calculate computation cost, "
            f"but got {metric_data.q_heads=} and {metric_data.head_dim=}"
        )
        assert pass_type in [
            "fwd",
            "bwd",
            "1f1b",
        ], "Only support 'fwd', 'bwd' and '1f1b' pass!"
        computation_amount_list: list[list[float]] = []

        for calc_meta in calc_meta_list:
            computation_amount_per_rank: list[float] = []
            iter_attn_arg = [calc_meta.local_attn_arg] + calc_meta.remote_attn_args_list

            for attn_arg in iter_attn_arg:
                attn_mask_type = list(
                    map(AttnMaskType.from_int_type, attn_arg.attn_type_map)
                )

                if len(attn_arg.q_ranges) > 0:
                    flops_dict = calculate_attn_flops(
                        q_ranges=attn_arg.q_ranges,
                        k_ranges=attn_arg.k_ranges,
                        attn_mask_type=attn_mask_type,
                        total_seqlen_q=attn_arg.q_ranges.end,
                        num_heads_q=num_heads_q,
                        head_dim=head_dim,
                    )
                    computation_amount_per_rank.append(flops_dict[pass_type])
                else:
                    computation_amount_per_rank.append(0)

            computation_amount_list.append(computation_amount_per_rank)

        return MetricSet(computation_amount_list=computation_amount_list)

    @staticmethod
    def calculate_communication_cost(
        metric_data: MetricData,
    ) -> MetricSet:
        comm_meta_list: list[CommMeta] = metric_data.comm_meta_list
        num_heads_q, num_heads_kv, head_dim, pass_type = (
            metric_data.q_heads,
            metric_data.kv_heads,
            metric_data.head_dim,
            metric_data.pass_type,
        )
        assert (
            num_heads_q is not None
            and num_heads_kv is not None
            and head_dim is not None
        ), (
            f"Need q_heads, kv_heads and head_dim to calculate computation cost, "
            f"but got {metric_data.q_heads=}, {metric_data.kv_heads=} and {metric_data.head_dim=}"
        )
        assert pass_type in ["fwd", "bwd"], "Only support 'fwd' and 'bwd' pass!"

        fwd_cast_dtype, fwd_reduce_dtype, bwd_cast_dtype, bwd_reduce_dtype = (
            metric_data.fwd_cast_dtype,
            metric_data.fwd_reduce_dtype,
            metric_data.bwd_cast_dtype,
            metric_data.bwd_reduce_dtype,
        )
        assert (
            fwd_cast_dtype is not None
            and fwd_reduce_dtype is not None
            and bwd_cast_dtype is not None
            and bwd_reduce_dtype is not None
        ), (
            f"Need set dtype for cast and reduce in fwd and bwd pass, "
            f"but got {fwd_cast_dtype=}, {fwd_reduce_dtype=}, {bwd_cast_dtype=} and {bwd_reduce_dtype=}"
        )

        def dtype_nbytes(dtype: torch.dtype) -> int:
            return torch.tensor([], dtype=dtype).element_size()

        fwd_cast_dtype_bytes = dtype_nbytes(fwd_cast_dtype)
        fwd_reduce_dtype_bytes = dtype_nbytes(fwd_reduce_dtype)
        bwd_cast_dtype_bytes = dtype_nbytes(bwd_cast_dtype)
        bwd_reduce_dtype_bytes = dtype_nbytes(bwd_reduce_dtype)

        comm_bytes_list: list[list[tuple[int, int]]] = []

        for comm_meta in comm_meta_list:
            comm_bytes_this_rank: list[tuple[int, int]] = []

            kv_recv_tokens_num_list: list[
                int
            ] = comm_meta.num_remote_kv_tokens_per_stage
            qo_recv_tokens_num_list: list[
                int
            ] = comm_meta.num_remote_qo_tokens_per_stage

            kv_send_tokens_num_list: list[int] = []
            qo_send_tokens_num_list: list[int] = []

            num_of_stage = len(comm_meta.num_remote_qo_tokens_per_stage)

            for i in range(num_of_stage):
                kv_group_collective_arg = comm_meta.kv_group_collective_args_list[i]
                qo_group_collective_arg = comm_meta.qo_group_collective_args_list[i]

                if kv_group_collective_arg is not None:
                    kv_send_tokens_num = sum(
                        [
                            kv_group_collective_arg.input_split_size_list[j]
                            * len(kv_group_collective_arg.dst_indices_list[j])
                            for j in range(
                                len(kv_group_collective_arg.input_split_size_list)
                            )
                        ]
                    )
                else:
                    kv_send_tokens_num = 0  # type: ignore[unreachable]

                if qo_group_collective_arg is not None:
                    qo_send_tokens_num = sum(
                        [
                            qo_group_collective_arg.input_split_size_list[j]
                            * len(qo_group_collective_arg.dst_indices_list[j])
                            for j in range(
                                len(qo_group_collective_arg.input_split_size_list)
                            )
                        ]
                    )
                else:
                    qo_send_tokens_num = 0  # type: ignore[unreachable]

                kv_send_tokens_num_list.append(kv_send_tokens_num)
                qo_send_tokens_num_list.append(qo_send_tokens_num)

            for i in range(num_of_stage + 2):
                recv_bytes, send_bytes = 0, 0

                if pass_type == "fwd":
                    if i < num_of_stage:
                        send_q_bytes = (
                            qo_send_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * fwd_cast_dtype_bytes
                        )
                        send_kv_bytes = (
                            kv_send_tokens_num_list[i]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * fwd_cast_dtype_bytes
                        )

                        recv_q_bytes = (
                            qo_recv_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * fwd_cast_dtype_bytes
                        )
                        recv_kv_bytes = (
                            kv_recv_tokens_num_list[i]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * fwd_cast_dtype_bytes
                        )

                        send_bytes += send_q_bytes + send_kv_bytes
                        recv_bytes += recv_q_bytes + recv_kv_bytes

                    if i > 1:
                        send_o_bytes = (
                            qo_recv_tokens_num_list[i - 2]
                            * num_heads_q
                            * head_dim
                            * fwd_reduce_dtype_bytes
                        )
                        send_lse_bytes = (
                            qo_recv_tokens_num_list[i - 2] * num_heads_q * 4
                        )

                        recv_o_bytes = (
                            qo_send_tokens_num_list[i - 2]
                            * num_heads_q
                            * head_dim
                            * fwd_reduce_dtype_bytes
                        )
                        recv_lse_bytes = (
                            qo_send_tokens_num_list[i - 2] * num_heads_q * 4
                        )

                        send_bytes += send_o_bytes + send_lse_bytes
                        recv_bytes += recv_o_bytes + recv_lse_bytes

                elif pass_type == "bwd":
                    if i < num_of_stage:
                        send_q_bytes = (
                            qo_send_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * bwd_cast_dtype_bytes
                        )
                        send_kv_bytes = (
                            kv_send_tokens_num_list[i]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * bwd_cast_dtype_bytes
                        )
                        send_o_bytes = (
                            qo_send_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * 2
                            * bwd_cast_dtype_bytes
                        )  # o and do
                        send_lse_bytes = qo_send_tokens_num_list[i] * num_heads_q * 4

                        recv_q_bytes = (
                            qo_recv_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * bwd_cast_dtype_bytes
                        )
                        recv_kv_bytes = (
                            kv_recv_tokens_num_list[i]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * bwd_cast_dtype_bytes
                        )
                        recv_o_bytes = (
                            qo_recv_tokens_num_list[i]
                            * num_heads_q
                            * head_dim
                            * 2
                            * bwd_cast_dtype_bytes
                        )  # o and do
                        recv_lse_bytes = qo_recv_tokens_num_list[i] * num_heads_q * 4

                        send_bytes += (
                            send_q_bytes + send_kv_bytes + send_o_bytes + send_lse_bytes
                        )
                        recv_bytes += (
                            recv_q_bytes + recv_kv_bytes + recv_o_bytes + recv_lse_bytes
                        )

                    if i > 1:
                        send_dq_bytes = (
                            qo_recv_tokens_num_list[i - 2]
                            * num_heads_q
                            * head_dim
                            * bwd_reduce_dtype_bytes
                        )
                        send_dkv_bytes = (
                            kv_recv_tokens_num_list[i - 2]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * bwd_reduce_dtype_bytes
                        )

                        recv_dq_bytes = (
                            qo_send_tokens_num_list[i - 2]
                            * num_heads_q
                            * head_dim
                            * bwd_reduce_dtype_bytes
                        )
                        recv_dkv_bytes = (
                            kv_send_tokens_num_list[i - 2]
                            * num_heads_kv
                            * head_dim
                            * 2
                            * bwd_reduce_dtype_bytes
                        )

                        send_bytes += send_dq_bytes + send_dkv_bytes
                        recv_bytes += recv_dq_bytes + recv_dkv_bytes

                comm_bytes_this_rank.append((send_bytes, recv_bytes))

            comm_bytes_list.append(comm_bytes_this_rank)

        return MetricSet(comm_bytes_list=comm_bytes_list)
