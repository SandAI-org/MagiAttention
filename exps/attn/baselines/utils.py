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

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._calc_dispatch_meta import _calc_self_attn_areas


def calculate_attn_flops(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    num_heads_q: int,
    head_dim: int,
) -> dict[str, float]:
    attn_area = _calc_self_attn_areas(
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


def block_causal_block_mask_func(b, h, q_idx, kv_idx, block_size):
    block_idx_q = q_idx // block_size
    end_q_idx_in_this_block = (block_idx_q + 1) * block_size
    return kv_idx <= end_q_idx_in_this_block


def make_block_causal_mask_score_mod(block_size):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                block_causal_block_mask_func,
                block_size=block_size,
            )(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def make_block_causal_block_mask(sq, sk, block_size):
    block_mask = create_block_mask(
        partial(
            block_causal_block_mask_func,
            block_size=block_size,
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


def varlen_block_causal_mask(b, h, q_idx, kv_idx, block_size, document_id):
    block_causal_mask = block_causal_block_mask_func(
        None, None, q_idx, kv_idx, block_size
    )
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return block_causal_mask & document_mask


def make_varlen_block_causal_block_mask(sq, sk, block_size, document_id):
    block_mask = create_block_mask(
        partial(
            varlen_block_causal_mask,
            block_size=block_size,
            document_id=document_id,
        ),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return block_mask


def make_varlen_block_causal_sdpa_mask(sq, sk, block_size, document_id):
    sdpa_mask = create_mask(
        partial(
            varlen_block_causal_mask,
            block_size=block_size,
            document_id=document_id,
        ),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return sdpa_mask


def make_varlen_block_causal_mask_score_mod(block_size, document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                varlen_block_causal_mask,
                block_size=block_size,
                document_id=document_id,
            )(b, h, q_idx, kv_idx),
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

        # 筛选符合条件的区间：a <= remaining 且 a < b
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

        # 根据权重选择区间
        selected_interval = random.choices(
            available_intervals, weights=available_weights, k=1
        )[0]

        a, b = selected_interval
        # 生成该区间内且不超过remaining的长度
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


def generate_ranges_from_seqlen(seqlen, block_size, start_offset=0):
    num_blocks = (seqlen + block_size - 1) // block_size

    q_ranges = []
    k_ranges = []

    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, seqlen)

        q_ranges.append([start + start_offset, end + start_offset])
        k_ranges.append([start_offset, end + start_offset])

    return q_ranges, k_ranges


def generate_ranges_from_seqlens(seqlens: list[int], block_size: int):
    q_ranges = AttnRanges()
    k_ranges = AttnRanges()
    cu_seqlens = seqlens2cu_seqlens(seqlens)
    for seqlen, start_offset in zip(seqlens, cu_seqlens[:-1]):
        q_range_list, k_range_list = generate_ranges_from_seqlen(
            seqlen, block_size, start_offset
        )
        q_ranges.extend(AttnRanges.from_ranges(q_range_list))
        k_ranges.extend(AttnRanges.from_ranges(k_range_list))

    return q_ranges, k_ranges
