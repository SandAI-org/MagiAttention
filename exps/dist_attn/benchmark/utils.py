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

import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np

import magi_attention
from exps.dist_attn.benchmark.enums import MetricsType


class DatasetSampler:
    def __init__(
        self,
        data_path: str,
        pack_len: int = 0,
        chunk_ratio: float = 0.25,
        is_binned: bool = True,
        seed: int = 42,
        drop_thres: int = -1,
    ):
        """
        This is a data sampler for context parallelism benchmarking.
        It generates variable-length (varlen) sample packs to simulate the dataset packing strategy
        used during training, while chunking excessively long sequences to avoid full-mask scenarios.

        args:
            data_path: file path for the dataset statistics results.
            pack_len: total length of each sample pack.
            chunk_ratio: ratio used to determine chunk size; sequences longer than `pack_len * chunk_ratio`
                are split into chunks of this length.
            is_binned: whether the dataset statistics are provided as intervals with counts (binned)
                or as individual lengths with counts.
            seed: random seed to shuffle.
            drop_thres: whether to drop sequences longer than `drop_thres` to generate short-varlen pack,
                -1: no drop by default.

        """
        assert (
            0 < chunk_ratio <= 1
        ), f"Invalid chunk ratio, expect (0,1], got {chunk_ratio}."
        assert pack_len > 0, f"Invalid pack size, got {pack_len}."
        assert os.path.exists(
            data_path
        ), f"Invalid data path, not exist, got {data_path}."

        self.pack_idx = 0
        self.sample_idx = 0
        self.pack_len = pack_len
        self.chunk_len = pack_len * chunk_ratio
        self.drop_thres = drop_thres
        self.rng = np.random.default_rng(seed)
        if is_binned:
            distribution = self.load_binned_distribution_len(data_path)
        else:
            distribution = self.load_distribution_len(data_path)
        self.pack_with_shuffle(distribution)

    def load_distribution_len(self, data_path: str) -> Dict[int, int]:
        """Individual lengths distribution
        Dataset statistics are provided as individual sample lengths with their corresponding counts.
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        distribution = {int(k): int(v) for k, v in raw.items()}
        return distribution

    def load_binned_distribution_len(self, data_path: str) -> Dict[int, int]:
        """Binned distribution
        Dataset statistics are provided as intervals (bins) with counts for each interval.
        """
        distribution = defaultdict(int)
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip column names
            for row in reader:
                range, num, *_ = row
                left, right = range.strip("[]").split(",")
                length = (int(left) + int(right)) // 2
                distribution[length] = int(num)
        return distribution

    def pack_with_shuffle(self, dlen2num: Dict[int, int]):
        """
        Process the dataset distribution by flattening it into individual samples,
        splitting samples longer than `chunk_len` into chunks, shuffling the samples,
        sequentially packing them into sample packs, and shuffling the packs.
        """

        # DE-BUG
        split_alignment = magi_attention.comm.native_grpcoll_split_alignment()

        # collect all samples
        lengths = np.array(list(dlen2num.keys()), dtype=np.int32)
        # DE-BUG
        lengths = (np.round(lengths / split_alignment) * split_alignment).astype(
            np.int32
        )
        nums = np.array(list(dlen2num.values()), dtype=np.int32)
        # generate all chunk sample lengths and shuffle
        if self.drop_thres > 0 and self.chunk_len > self.drop_thres:
            full_parts = np.empty((0,), dtype=np.int32)
        else:
            full_chunk = lengths // self.chunk_len
            full_sizes = full_chunk * nums
            full_parts = np.repeat(self.chunk_len, full_sizes.sum()).astype(np.int32)
        tails = lengths % self.chunk_len
        # DE-BUG
        tails = (np.round(tails / split_alignment) * split_alignment).astype(np.int32)
        tail_mask = tails > 0
        if self.drop_thres > 0:
            tail_mask = tail_mask & (tails <= self.drop_thres)
        tail_sizes = nums[tail_mask]
        tail_vals = tails[tail_mask]
        tail_parts = np.repeat(tail_vals, tail_sizes).astype(np.int32)
        samples = np.concatenate([full_parts, tail_parts])
        self.rng.shuffle(samples)
        samples = samples.tolist()

        # create all packs and shuffle
        pack_num = sum(samples) // self.pack_len
        sample_idx, tail = 0, 0
        packs = []
        for _ in range(pack_num):
            _pack = []
            left = self.pack_len
            if tail > 0:
                _pack.append(tail)
                left -= tail
                tail = 0
            while left > 0 and sample_idx < len(samples):
                if samples[sample_idx] <= left:
                    _pack.append(samples[sample_idx])
                    left -= samples[sample_idx]
                    sample_idx += 1
                else:
                    _pack.append(left)
                    tail = samples[sample_idx] - left
                    left = 0
                    sample_idx += 1
            if len(_pack) > 0:
                packs.append(_pack)
        self.rng.shuffle(packs)

        self.samples = samples
        self.packs = packs

    def generate_pack_samples(self) -> List[int]:
        """Get a sample pack."""
        if self.pack_idx >= len(self.packs):
            raise StopIteration
        pack = self.packs[self.pack_idx]
        self.pack_idx += 1
        return pack

    def generate_single_sample(self):
        """Get a single sample."""
        if self.sample_idx >= len(self.samples):
            raise StopIteration
        sample = self.samples[self.sample_idx]
        self.sample_idx += 1
        return sample

    def reset(self, re_shuffle=False):
        """reset and shuffle without re-pack"""
        self.pack_idx = 0
        self.sample_idx = 0
        if re_shuffle:
            self.rng.shuffle(self.samples)
            self.rng.shuffle(self.packs)


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


# TODO: to remove later


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
