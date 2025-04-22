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
from functools import partial
from itertools import accumulate, pairwise

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask


def seqlens2curanges(seqlens: list[int]):
    return list(pairwise(accumulate([0] + seqlens)))


def make_full_mask_score_mod():
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    return score_mod


def causal_block_mask_func(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def make_causal_mask_score_mod():
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            causal_block_mask_func(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def make_causal_block_mask(sq, sk):
    block_mask = create_block_mask(
        causal_block_mask_func,
        B=None,
        H=None,
        Q_LEN=sq,
        KV_LEN=sk,
    )

    return block_mask


def sliding_window_causal_mask_func(b, h, q_idx, kv_idx, window_size):
    causal_mask = q_idx >= kv_idx
    window_mask = q_idx - kv_idx <= window_size
    return causal_mask & window_mask


def make_sliding_window_causal_mask_score_mod(window_size):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                sliding_window_causal_mask_func,
                window_size=window_size,
            )(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def make_sliding_window_causal_block_mask(sq, sk, window_size):
    block_mask = create_block_mask(
        partial(
            sliding_window_causal_mask_func,
            window_size=window_size,
        ),
        B=None,
        H=None,
        Q_LEN=sq,
        KV_LEN=sk,
    )

    return block_mask


def varlen_full_mask(b, h, q_idx, kv_idx, document_id):
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return document_mask


def make_varlen_full_block_mask(sq, sk, document_id):
    block_mask = create_block_mask(
        partial(varlen_full_mask, document_id=document_id), 1, 1, sq, sk, device="cuda"
    )

    return block_mask


def make_varlen_full_sdpa_mask(sq, sk, document_id):
    sdpa_mask = create_mask(
        partial(varlen_full_mask, document_id=document_id), 1, 1, sq, sk, device="cuda"
    )

    return sdpa_mask


def make_varlen_full_mask_score_mod(document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(varlen_full_mask, document_id=document_id)(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def varlen_causal_mask(b, h, q_idx, kv_idx, document_id):
    causal_mask = q_idx >= kv_idx
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return causal_mask & document_mask


def make_varlen_causal_block_mask(sq, sk, document_id):
    block_mask = create_block_mask(
        partial(varlen_causal_mask, document_id=document_id),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return block_mask


def make_varlen_causal_sdpa_mask(sq, sk, document_id):
    sdpa_mask = create_mask(
        partial(varlen_causal_mask, document_id=document_id),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return sdpa_mask


def make_varlen_causal_mask_score_mod(document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(varlen_causal_mask, document_id=document_id)(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def generate_seqlens(distribution, total_seqlen):
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

        # filter intervals satisfy：a <= remaining and a < b
        available_intervals = []
        available_weights = []
        for interval, weight in zip(intervals, weights):
            a, b = interval
            if a < b and a <= remaining:
                available_intervals.append(interval)
                available_weights.append(weight)

        if not available_intervals:
            raise ValueError(
                f"No valid interval available for remaining length {remaining}"
            )

        # choose intervals according to weights
        selected_interval = random.choices(
            available_intervals, weights=available_weights, k=1
        )[0]

        a, b = selected_interval
        # generate seqlen less than remaining and in the interval
        max_val = min(b - 1, remaining)
        seqlen = random.randint(a, max_val)

        seqlens.append(seqlen)
        current_total += seqlen

    seqlens = [seqlen for seqlen in seqlens if seqlen > 0]

    return seqlens


def seqlens2cu_seqlens(seqlens: list[int]) -> list[int]:
    cu_seqlens = [0]
    for seqlen in seqlens:
        cu_seqlens.append(cu_seqlens[-1] + seqlen)
    return cu_seqlens


def curanges2document_id(cu_ranges):
    document_id = torch.zeros(cu_ranges[-1][1], dtype=torch.int32, device="cuda")
    for i, (start, end) in enumerate(cu_ranges):
        document_id[start:end] = i

    return document_id


__all__ = [
    "make_full_mask_score_mod",
    "make_causal_block_mask",
    "make_causal_mask_score_mod",
    "make_sliding_window_causal_block_mask",
    "make_sliding_window_causal_mask_score_mod",
    "make_varlen_full_block_mask",
    "make_varlen_full_sdpa_mask",
    "make_varlen_full_mask_score_mod",
    "make_varlen_causal_block_mask",
    "make_varlen_causal_sdpa_mask",
    "make_varlen_causal_mask_score_mod",
    "generate_seqlens",
    "seqlens2curanges",
    "seqlens2cu_seqlens",
    "curanges2document_id",
]
