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

from exps.dist_attn.benchmark.enums import MetricsType


def generate_seqlens(
    distribution: dict,
    total_seqlen: int,
    allow_zero: bool = False,
    rng: random.Random | None = None,
) -> list[int]:
    """Each time a number is sampled from the distribution and accumulated into the current length.
    Sampling is repeated until the length exceeds the total sequence length.

    Args:
        distribution (dict): a dictionary for describing data distribution, such as {(a, b): property}
        total_seqlen (int): total_seqlen for sampling
        allow_zero (bool): can zero be sampled
        rng (random.Random | None): independent random number generator, use random module if None

    Returns:
        list[int]: The sampled length, with the total length equal to total_seqlen.
    """
    # init random number generator
    rng = rng if rng is not None else random
    
    # normalize distribution
    total = sum(distribution.values())
    distribution = {k: v / total for k, v in distribution.items()}

    items = list(distribution.items())
    intervals = [item[0] for item in items]
    weights = [item[1] for item in items]

    seqlens = []
    current_total = 0

    while current_total < total_seqlen:
        remaining = total_seqlen - current_total

        # choose intervals according to weights
        selected_interval = rng.choices(intervals, weights=weights, k=1)[0]

        a, b = selected_interval
        # generate seqlen less than remaining and in the interval
        seqlen = rng.randint(int(a), int(b))
        seqlen = min(seqlen, remaining)

        if not allow_zero:
            seqlen = max(seqlen, 1)

        seqlens.append(seqlen)
        current_total += seqlen

    return seqlens


def generate_seqlen_for_one_time(
    distribution: dict,
    total_seqlen: int,
    allow_zero: bool = False,
    rng: random.Random | None = None,
) -> int:
    """Only sample for one time, rest is the same as above generate_seqlens function

    Returns:
        int: a seqlen for one time sample
    """
    # init random number generator
    rng = rng if rng is not None else random

    total = sum(distribution.values())
    distribution = {k: v / total for k, v in distribution.items()}

    items = list(distribution.items())
    intervals = [item[0] for item in items]
    weights = [item[1] for item in items]

    # choose intervals according to weights
    selected_interval = rng.choices(intervals, weights=weights, k=1)[0]

    a, b = selected_interval
    # generate seqlen less than remaining and in the interval
    seqlen = rng.randint(int(a), int(b))
    seqlen = min(seqlen, total_seqlen)

    if not allow_zero:
        seqlen = max(seqlen, 1)

    return seqlen


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    """transfer seqlens list to cu_seqlens, do not have check"""
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def varlen_long_seqlen_distribution():
    """data distribution for seqlen > 64k"""
    return {
        (0, 2 * 1024): 0.16,
        (2 * 1024, 4 * 1024): 0.05,
        (4 * 1024, 8 * 1024): 0.04,
        (8 * 1024, 16 * 1024): 0.06,
        (16 * 1024, 32 * 1024): 0.08,
        (32 * 1024, 64 * 1024): 0.21,
        (64 * 1024, 128 * 1024): 0.4,
        (128 * 1024, 256 * 1024): 0.2,
        (256 * 1024, 512 * 1024): 0.05,
        (512 * 1024, 1024 * 1024): 0.04,
        (1024 * 1024, 2048 * 1024): 0.01,
        (2048 * 1024, 4096 * 1024): 0.01,
    }


def varlen_short_seqlen_distribution():
    """data distribution for seqlen <= 64k"""
    return {
        (0, 0.25 * 1024): 0.0766,
        (0.25 * 1024, 0.5 * 1024): 0.1651,
        (0.5 * 1024, 0.75 * 1024): 0.1246,
        (0.75 * 1024, 1.0 * 1024): 0.0972,
        (1.0 * 1024, 1.5 * 1024): 0.1324,
        (1.5 * 1024, 2.0 * 1024): 0.0927,
        (2.0 * 1024, 2.5 * 1024): 0.0720,
        (2.5 * 1024, 3.0 * 1024): 0.0519,
        (3.0 * 1024, 3.5 * 1024): 0.0379,
        (3.5 * 1024, 4.0 * 1024): 0.0264,
        (4.0 * 1024, 5.0 * 1024): 0.0380,
        (5.0 * 1024, 6.0 * 1024): 0.0245,
        (6.0 * 1024, 7.0 * 1024): 0.0145,
        (7.0 * 1024, 8.0 * 1024): 0.0097,
        (8.0 * 1024, 10.0 * 1024): 0.0122,
        (10.0 * 1024, 15.0 * 1024): 0.0128,
        (15.0 * 1024, 20.0 * 1024): 0.0051,
        (20.0 * 1024, 64.0 * 1024): 0.0064,
    }


def add_to_data_dict_average(
    data: dict[str, dict[str, list[float]]],
    metrics_type: list[MetricsType],
    data_list: list,
) -> list:
    """After averaging the metrics calculated by multiple masks, convert them into a data_dict.

    Args:
        data (dict[str, dict[str, list[float]]]): data to handled, include multiple masks with multiple metrics
        metrics_type (list[MetricsType]): calculated metric types
        data_list (list): data_dict can be converted to excel or csv

    Returns:
        list: data_dict can be converted to excel or csv
    """
    for task_name, task_data in data.items():
        task_para = task_name.split("-")
        name, total_seqlen, chunk_size, cp_size = task_para
        for solver_name, solver_metrics in task_data.items():
            cur_data_dict = {
                "task_name": name,
                "total_seqlen": total_seqlen,
                "chunk_size": chunk_size,
                "cp_size": cp_size,
                "solver": solver_name,
            }

            for idx in range(len(solver_metrics)):
                metric_name = metrics_type[idx].name
                metric = solver_metrics[idx]
                cur_data_dict[metric_name] = str(metric)

            data_list.append(cur_data_dict)

    return data_list


def add_to_data_dict_without_average(
    data: dict,
    metrics_type: list[MetricsType],
    data_list: list,
) -> list:
    """Without averaging, retain the calculation results of all masks and convert them into a data_dict.

    Args:
        data (dict[str, dict[str, list[float]]]): data to handled, include multiple masks with multiple metrics
        metrics_type (list[MetricsType]): calculated metric types
        data_list (list): data_dict can be converted to excel or csv

    Returns:
        dict: data_dict can be converted to excel or csv
    """
    for task_name, task_tuple_list in data.items():
        task_para = task_name.split("-")
        name, total_seqlen, chunk_size, cp_size = task_para
        for task_tuple in task_tuple_list:
            q_ranges, k_ranges, is_causal_mapping, task_data = task_tuple
            cur_data_dict = {
                "task_name": name,
                "total_seqlen": total_seqlen,
                "chunk_size": chunk_size,
                "cp_size": cp_size,
                "q_ranges": q_ranges,
                "k_ranges": k_ranges,
                "is_causal_mapping": is_causal_mapping,
            }

            for solver_idx, (solver_name, solver_metrics) in enumerate(
                task_data.items()
            ):
                cur_data_dict["solver_" + str(solver_idx)] = solver_name

                for idx in range(len(solver_metrics)):
                    metric_name = metrics_type[idx].name + "_" + str(solver_idx)
                    metric = solver_metrics[idx]
                    cur_data_dict[metric_name] = metric

            data_list.append(cur_data_dict)

    return data_list
