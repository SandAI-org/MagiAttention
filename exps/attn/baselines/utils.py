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

import os
import random
from functools import lru_cache, partial
from itertools import accumulate, pairwise
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from exps.dist_attn.benchmark.enums import FlashMaskType
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta import make_global_bucket_from_qk_ranges


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# availability check
def block_sparse_available(
    attn_impl: str,
    num_q_heads: int,
    num_kv_heads: int,
    q_block_size: int,
    k_block_size: int,
    wd: str,
) -> bool:
    """
    Check availability of different block sparse attention implementations.
    """
    if attn_impl == "flashinfer" or attn_impl == "ffa_swapab":
        # flashinfer and ffa_swapab doesn't support backward
        return wd == "fwd"

    if attn_impl == "ffa" or attn_impl == "flex":
        return True

    if q_block_size == k_block_size:  # equal block size
        if attn_impl == "vsa" or attn_impl == "vsa_triton":
            # currently vsa only supports block size == 64
            return num_q_heads == num_kv_heads and q_block_size == 64

        if attn_impl == "fa2_sparse":
            return (
                wd == "fwd" and q_block_size == 128
            )  # only support forward and 128 block size

    return False


def var_block_sparse_available(attn_impl: str, wd: str) -> bool:
    if attn_impl == "flashinfer":
        # flashinfer doesn't support variable block size
        return wd == "fwd"
    return True


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


def sliding_window_full_mask_func(b, h, q_idx, kv_idx, window_size):
    return (q_idx - kv_idx <= window_size) & (kv_idx - q_idx <= window_size)


def make_sliding_window_full_mask_score_mod(window_size):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                sliding_window_full_mask_func,
                window_size=window_size,
            )(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def make_sliding_window_full_block_mask(sq, sk, window_size):
    block_mask = create_block_mask(
        partial(
            sliding_window_full_mask_func,
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


def prefix_lm_causal_mask(b, h, q_idx, kv_idx, prefix_length):
    causal_mask = q_idx >= kv_idx
    prefix_mask = kv_idx <= prefix_length
    return causal_mask | prefix_mask


def make_prefix_lm_causal_block_mask(sq, sk, prefix_length):
    block_mask = create_block_mask(
        partial(prefix_lm_causal_mask, prefix_length=prefix_length),
        B=None,
        H=None,
        Q_LEN=sq,
        KV_LEN=sk,
    )

    return block_mask


def make_prefix_lm_causal_mask_score_mod(prefix_length):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(prefix_lm_causal_mask, prefix_length=prefix_length)(
                b, h, q_idx, kv_idx
            ),
            score,
            -float("inf"),
        )

    return score_mod


def share_question_mask(b, h, q_idx, kv_idx, document_id):
    share_mask = document_id[kv_idx] == 0
    first_causal = document_id[q_idx] != 0
    varlen_mask = varlen_causal_mask(b, h, q_idx, kv_idx, document_id)
    return (first_causal & share_mask) | varlen_mask


def make_share_question_block_mask(sq, sk, document_id):
    block_mask = create_block_mask(
        partial(share_question_mask, document_id=document_id),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return block_mask


def make_share_question_mask_score_mod(document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(share_question_mask, document_id=document_id)(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def causal_blockwise_mask(b, h, q_idx, kv_idx, document_id):
    blockwise_mask = document_id[q_idx] == document_id[-1]
    last_causal = document_id[kv_idx] != document_id[-1]
    varlen_mask = varlen_causal_mask(b, h, q_idx, kv_idx, document_id)
    return (last_causal & blockwise_mask) | varlen_mask


def make_causal_blockwise_block_mask(sq, sk, document_id):
    block_mask = create_block_mask(
        partial(causal_blockwise_mask, document_id=document_id),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return block_mask


def make_causal_blockwise_mask_score_mod(document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(causal_blockwise_mask, document_id=document_id)(
                b, h, q_idx, kv_idx
            ),
            score,
            -float("inf"),
        )

    return score_mod


def prefix_lm_varlen_mask(
    b, h, q_idx, kv_idx, prefix_length, document_id, cu_seqlens_kv
):
    prefix_mask = (kv_idx - cu_seqlens_kv[document_id[kv_idx]]) <= prefix_length
    document_mask = document_id[q_idx] == document_id[kv_idx]
    prefix_document_mask = prefix_mask & document_mask
    varlen_mask = varlen_causal_mask(b, h, q_idx, kv_idx, document_id)
    return varlen_mask | prefix_document_mask


def make_prefix_lm_varlen_block_mask(sq, sk, prefix_length, document_id, cu_seqlens_kv):
    block_mask = create_block_mask(
        partial(
            prefix_lm_varlen_mask,
            prefix_length=prefix_length,
            document_id=document_id,
            cu_seqlens_kv=cu_seqlens_kv,
        ),
        1,
        1,
        sq,
        sk,
        device="cuda",
    )

    return block_mask


def make_prefix_lm_varlen_mask_score_mod(prefix_length, document_id, cu_seqlens_kv):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                prefix_lm_varlen_mask,
                prefix_length=prefix_length,
                document_id=document_id,
                cu_seqlens_kv=cu_seqlens_kv,
            )(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def globle_sliding_window_mask_func(b, h, q_idx, kv_idx, window_size):
    sliding_window_mask = (q_idx - kv_idx <= window_size) & (
        kv_idx - q_idx <= window_size
    )
    global_mask = (q_idx < 2 * window_size) | (kv_idx < 2 * window_size)
    return sliding_window_mask | global_mask


def make_global_sliding_window_block_mask(sq, sk, window_size):
    block_mask = create_block_mask(
        partial(
            globle_sliding_window_mask_func,
            window_size=window_size,
        ),
        B=None,
        H=None,
        Q_LEN=sq,
        KV_LEN=sk,
    )

    return block_mask


def make_global_sliding_window_mask_score_mod(window_size):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                globle_sliding_window_mask_func,
                window_size=window_size,
            )(b, h, q_idx, kv_idx),
            score,
            -float("inf"),
        )

    return score_mod


def make_block_causal_varlen_mask(b, h, q_idx, kv_idx, block_size, document_id):
    block_idx = q_idx // block_size
    block_mask = kv_idx < (block_idx + 1) * block_size
    document_mask = document_id[q_idx] == document_id[kv_idx]
    return block_mask & document_mask


def make_block_causal_varlen_block_mask(sq, sk, block_size, document_id):
    block_mask = create_block_mask(
        partial(
            make_block_causal_varlen_mask,
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


def make_block_causal_varlen_score_mod(block_size, document_id):
    def score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(
            partial(
                make_block_causal_varlen_mask,
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

        # Filter for valid intervals: a <= remaining and a < b
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

        # Select an interval based on the weights
        selected_interval = random.choices(
            available_intervals, weights=available_weights, k=1
        )[0]

        a, b = selected_interval
        # Generate a length within the selected interval that does not exceed the remaining length
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


def generate_flashmask_indices(
    sq,
    sk,
    flash_mask_type,
    window_size=0,
    cu_ranges=None,
    prefix_length=0,
    block_size=1024,
):
    import paddle

    is_causal = True
    if flash_mask_type == FlashMaskType.FULL:
        LTS = paddle.to_tensor(
            [sq] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        UTE = paddle.to_tensor(
            [0] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.CAUSAL:
        LTS = paddle.to_tensor(
            [sq] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = LTS
    elif flash_mask_type == FlashMaskType.SLIDING_WINDOW_CAUSAL:
        LTS = paddle.arange(
            window_size + 1, sq + window_size + 1, dtype=paddle.int32
        ).reshape([1, 1, sk, 1])
        LTS = paddle.clip(LTS, max=sq)
        attn_mask_startend_row_indices = LTS
    elif flash_mask_type == FlashMaskType.SLIDING_WINDOW:
        LTS = paddle.arange(
            window_size + 1, sq + window_size + 1, dtype=paddle.int32
        ).reshape([1, 1, sk, 1])
        LTS = paddle.clip(LTS, max=sq)
        UTE = paddle.arange(-window_size, sq - window_size, dtype=paddle.int32).reshape(
            [1, 1, sk, 1]
        )
        UTE[:, :, : 1 + window_size, :] = 0
        attn_mask_startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.CAUSAL_DOCUMENT:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = LTS
    elif flash_mask_type == FlashMaskType.FULL_DOCUMENT:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, sk, 1])
        UTE = paddle.repeat_interleave(starts, lens).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.SHARE_QUESTION:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        ends[0] = sq
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = LTS
    elif flash_mask_type == FlashMaskType.CAUSAL_BLOCKWISE:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        ends[-2:] = sq
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, sk, 1])
        nstarts = paddle.full_like(starts, starts[-1])
        nstarts[-2:] = sq
        LTE = paddle.repeat_interleave(nstarts, lens).reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
    elif flash_mask_type == FlashMaskType.PREFIX_LM_CAUSAL:
        LTS = paddle.to_tensor(
            [sq] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        UTE = paddle.arange(0, sq, dtype=paddle.int32).reshape([1, 1, sk, 1])
        UTE[:, :, : prefix_length + 1, :] = 0
        attn_mask_startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.PREFIX_LM_DOCUMENT:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, sk, 1])
        LTE = paddle.arange(0, sq, dtype=paddle.int32)
        for i in range(len(starts)):
            s = cu_ranges[i][0]
            LTE[s : min(s + prefix_length + 1, sq)] = s
        LTE = LTE.reshape([1, 1, sk, 1])
        attn_mask_startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.GLOBAL_SLIDING_WINDOW:
        LTS = paddle.arange(
            window_size + 1, sq + window_size + 1, dtype=paddle.int32
        ).reshape([1, 1, sk, 1])
        LTS[:, :, : 2 * window_size, :] = sq
        LTE = paddle.to_tensor(
            [sq] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        UTS = paddle.to_tensor(
            [2 * window_size] * sk, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        ).reshape([1, 1, sk, 1])
        UTS[:, :, : 1 + 3 * window_size, :] = 0
        UTE = paddle.arange(-window_size, sq - window_size, dtype=paddle.int32).reshape(
            [1, 1, sk, 1]
        )
        UTE[:, :, : 1 + 3 * window_size, :] = 0
        attn_mask_startend_row_indices = paddle.concat([LTS, LTE, UTS, UTE], axis=-1)
        is_causal = False
    elif flash_mask_type == FlashMaskType.BLOCK_CAUSAL_DOCUMENT:
        cu_ranges_paddle = paddle.to_tensor(
            cu_ranges, dtype=paddle.int32, place=paddle.CUDAPlace(0)
        )
        starts, ends = cu_ranges_paddle[:, 0], cu_ranges_paddle[:, 1]
        lens = ends - starts
        LTS = paddle.repeat_interleave(ends, lens).reshape([1, 1, -1, 1])
        starts_expanded = paddle.repeat_interleave(starts, lens)
        global_idx = paddle.arange(0, sq, dtype=paddle.int32)
        relative_idx = global_idx - starts_expanded
        UTE = (
            paddle.floor_divide(relative_idx, block_size).to(paddle.int32) * block_size
            + starts_expanded
        )
        UTE = UTE.reshape([1, 1, -1, 1])
        attn_mask_startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        is_causal = False
        is_causal = False

    return attn_mask_startend_row_indices, is_causal


# ================ Utils for VSA ================
def get_vsa_mask_from_block_sparse_score(
    scores: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a block-wise attention score into a block-sparse index format
    that is compatible with FastVideo VSA (Video Sparse Attention).

    Args:
        scores (torch.Tensor): The attention scores of shape [b, h, num_q_blocks, num_kv_blocks].
        k (int): The number of key-value blocks each query block attends to.

    Returns:
        q2k_block_sparse_index: [bs, hq, num_q_blocks, k]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, hq, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to k).
        k2q_block_sparse_index: [bs, hk, num_kv_blocks, num_q_blocks]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, hk, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
    """

    device = scores.device
    # Ensure mask has batch dimension
    if scores.dim() == 3:  # Assuming [h, num_q_blocks, num_kv_blocks]
        scores = scores.unsqueeze(0)  # Add batch_size 1

    bs, h, num_q_blocks, num_kv_blocks = scores.shape
    # Ensure k is not larger than num_kv_blocks
    k = min(k, num_kv_blocks)

    # Get top-k indices for each q block
    _, q2k_block_sparse_index = torch.topk(scores, k, dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(torch.int32)

    # sort q2k_block_sparse_index
    q2k_block_sparse_index, _ = torch.sort(q2k_block_sparse_index, dim=-1)

    # All q blocks attend to exactly k kv blocks
    q2k_block_sparse_num = torch.full(
        (bs, h, num_q_blocks), k, dtype=torch.int32, device=device
    )

    # Fill in the mask based on the indices
    for b in range(bs):
        for head in range(h):
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx]

    # Create the reverse mapping (k2q)
    # First, initialize lists to collect q indices for each kv block
    k2q_indices_list: List[List[List[int]]] = [
        [[] for _ in range(num_kv_blocks)] for _ in range(bs * h)
    ]

    # Populate the lists based on q2k mapping
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for q_idx in range(num_q_blocks):
                kv_indices = q2k_block_sparse_index[b, head, q_idx].tolist()
                for kv_idx in kv_indices:
                    k2q_indices_list[flat_idx][kv_idx].append(q_idx)

    # Find the maximum number of q blocks that attend to any kv block
    max_q_per_kv = 0
    for flat_idx in range(bs * h):
        for kv_idx in range(num_kv_blocks):
            max_q_per_kv = max(max_q_per_kv, len(k2q_indices_list[flat_idx][kv_idx]))

    # Create tensors for k2q mapping
    k2q_block_sparse_index = torch.full(
        (bs, h, num_kv_blocks, max_q_per_kv), -1, dtype=torch.int32, device=device
    )
    k2q_block_sparse_num = torch.zeros(
        (bs, h, num_kv_blocks), dtype=torch.int32, device=device
    )

    # Fill the tensors
    for b in range(bs):
        for head in range(h):
            flat_idx = b * h + head
            for kv_idx in range(num_kv_blocks):
                q_indices = k2q_indices_list[flat_idx][kv_idx]
                num_q = len(q_indices)
                k2q_block_sparse_num[b, head, kv_idx] = num_q
                if num_q > 0:
                    k2q_block_sparse_index[b, head, kv_idx, :num_q] = torch.tensor(
                        q_indices, dtype=torch.int32, device=device
                    )

    return (
        q2k_block_sparse_index,
        q2k_block_sparse_num,
        k2q_block_sparse_index,
        k2q_block_sparse_num,
    )


# ================ Utils for flashinfer ================
def get_flashinfer_uniform_block_index(
    num_q_blocks: int,
    num_kv_blocks: int,
    seq_len_q: int,
    seq_len_k: int,
    num_kv_heads: int,
):
    # synthesize uniform block sizes
    block_row_sz = torch.ones(num_q_blocks, dtype=torch.int32) * (
        seq_len_q // num_q_blocks
    )
    block_row_sz[-1] = seq_len_q - (seq_len_q // num_q_blocks) * (num_q_blocks - 1)
    block_row_sz = block_row_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    block_col_sz = torch.ones(num_kv_blocks, dtype=torch.int32) * (
        seq_len_k // num_kv_blocks
    )
    block_col_sz[-1] = seq_len_k - (seq_len_k // num_kv_blocks) * (num_kv_blocks - 1)
    block_col_sz = block_col_sz.unsqueeze(0).repeat(num_kv_heads, 1)

    return block_row_sz, block_col_sz


# ================ Utils for flexattn ================
@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def flex_mask_mod(
    b_idx: torch.IntTensor,
    h_idx: torch.IntTensor,
    q_idx: torch.IntTensor,
    k_idx: torch.IntTensor,
    q_block_idx: torch.IntTensor,
    k_block_idx: torch.IntTensor,
    block_mask: torch.BoolTensor,
) -> torch.BoolTensor:
    """
    Block sparse mask for FlexAttention.
    Get each query/key token's block id, and return the corresponding mask value.

    q_block_idx: [num_heads, seq_len_q]
    k_block_idx: [num_heads, seq_len_k]
    block_mask: [num_heads, num_q_blocks, num_kv_blocks]
    """
    q_block_id = q_block_idx[h_idx, q_idx]
    k_block_id = k_block_idx[h_idx, k_idx]
    return block_mask[h_idx, q_block_id, k_block_id]


def get_var_block_idx(
    block_mask: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    bsz: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to get block indices for variable blocks.
    """
    bsz, num_heads, num_q_blocks, num_kv_blocks = block_mask.shape
    device = block_mask.device

    # Create a column of zeros for concatenation
    zeros_col_shape = (num_heads, 1)
    zeros = torch.zeros(zeros_col_shape, dtype=block_row_sz.dtype, device=device)

    # Calculate row (query) offsets
    row_cumsum = torch.cumsum(block_row_sz, dim=1)
    row_offsets = torch.cat([zeros, row_cumsum], dim=1)

    # Calculate column (key/value) offsets
    col_cumsum = torch.cumsum(block_col_sz, dim=1)
    col_offsets = torch.cat([zeros, col_cumsum], dim=1)

    row_offsets_list = row_offsets.tolist()
    col_offsets_list = col_offsets.tolist()

    q_block_idx = torch.zeros((num_heads, seq_len_q), dtype=torch.int32, device=device)
    k_block_idx = torch.zeros((num_heads, seq_len_k), dtype=torch.int32, device=device)

    for head_idx in range(num_heads):
        for q_block in range(num_q_blocks):
            q_start = row_offsets_list[head_idx][q_block]
            q_end = row_offsets_list[head_idx][q_block + 1]
            q_block_idx[head_idx, q_start:q_end] = q_block

        for k_block in range(num_kv_blocks):
            k_start = col_offsets_list[head_idx][k_block]
            k_end = col_offsets_list[head_idx][k_block + 1]
            k_block_idx[head_idx, k_start:k_end] = k_block

    return q_block_idx, k_block_idx


def get_uniform_block_idx(
    block_mask: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    bsz: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to get block indices for uniform blocks.
    """
    bsz, num_heads, num_q_blocks, num_kv_blocks = block_mask.shape
    device = block_mask.device

    assert seq_len_q % num_q_blocks == 0, "seq_len_q must be divisible by num_q_blocks"
    assert (
        seq_len_k % num_kv_blocks == 0
    ), "seq_len_k must be divisible by num_kv_blocks"

    q_block_size = seq_len_q // num_q_blocks
    k_block_size = seq_len_k // num_kv_blocks

    q_block_idx = torch.arange(0, seq_len_q, device=device) // q_block_size
    k_block_idx = torch.arange(0, seq_len_k, device=device) // k_block_size
    q_block_idx = q_block_idx.unsqueeze(0).repeat(num_heads, 1).to(torch.int32)
    k_block_idx = k_block_idx.unsqueeze(0).repeat(num_heads, 1).to(torch.int32)

    return q_block_idx, k_block_idx


def get_flex_mask_from_block_mask(
    block_mask: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    num_q_heads: int,
    num_kv_heads: int,
    block_row_sz: torch.Tensor = None,
    block_col_sz: torch.Tensor = None,
    bsz: int = 1,
):
    bsz, num_heads, num_q_blocks, num_kv_blocks = block_mask.shape
    assert num_heads == num_q_heads, "Block mask must be query-specific"
    device = block_mask.device

    if block_row_sz is None and block_col_sz is None:
        mode = "uniform"
    else:
        mode = "variable"

    if mode == "variable":
        block_col_sz = block_col_sz.repeat_interleave(
            num_q_heads // num_kv_heads, dim=0
        )
        q_block_idx, k_block_idx = get_var_block_idx(
            block_mask=block_mask,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            bsz=bsz,
        )
    elif mode == "uniform":
        q_block_idx, k_block_idx = get_uniform_block_idx(
            block_mask=block_mask, seq_len_q=seq_len_q, seq_len_k=seq_len_k, bsz=bsz
        )
    # TODO: assume batch size is 1 for now
    block_mask = block_mask.squeeze(0)

    mask_mod = partial(
        flex_mask_mod,
        q_block_idx=q_block_idx,
        k_block_idx=k_block_idx,
        block_mask=block_mask,
    )

    flex_mask = create_block_mask_cached(
        mask_mod, bsz, num_heads, seq_len_q, seq_len_k, device=device
    )

    return flex_mask
