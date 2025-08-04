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

import os
import random
from functools import partial
from itertools import accumulate, pairwise
from typing import List, Tuple

import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta._calc_dispatch_meta import _calc_self_attn_areas


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def generate_global_block_sparse_pattern(
    h, num_q_blocks, num_kv_blocks, sparsity_ratio, device="cuda"
):
    """
    Generates a global, arbitrary block-sparse pattern.

    In this pattern, connections are selected based on global scores, which means
    some q_blocks might not have any connections.

    Args:
        h (int): Number of attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        sparsity_ratio (float): The global proportion of connections to keep (e.g., 0.01 for 1%).
        device (str): The device to create the tensor on.

    Returns:
        block_sparse_mask (torch.Tensor): A boolean tensor mask of shape [h, num_q_blocks, num_kv_blocks].
        scores (torch.Tensor): A tensor of shape [h, num_q_blocks, num_kv_blocks] containing the random attention scores.
    """
    # 1. Generate random scores for all possible (q_block, kv_block) connections for each head.
    scores = torch.rand(h, num_q_blocks, num_kv_blocks, device=device)

    # 2. To perform a global top-k, flatten the q_block and kv_block dimensions for each head.
    # Shape changes from [h, num_q, num_k] to [h, num_q * num_k].
    flat_scores = scores.view(h, -1)

    # 3. Calculate the total number of connections 'k' to keep for each head.
    num_total_connections = num_q_blocks * num_kv_blocks
    k = int(num_total_connections * sparsity_ratio)
    k = max(1, k)  # Ensure at least one connection is kept.

    # 4. Perform a global top-k operation on the flattened scores to find the indices of the k highest-scoring connections.
    _, top_indices = torch.topk(flat_scores, k, dim=-1)

    # 5. Create a flattened boolean mask and set the positions corresponding to top_indices to True.
    flat_mask = torch.zeros_like(flat_scores, dtype=torch.bool)
    flat_mask.scatter_(dim=-1, index=top_indices, value=True)

    # 6. Reshape the flattened mask back to the 3D shape [h, num_q_blocks, num_kv_blocks].
    block_sparse_mask = flat_mask.view(h, num_q_blocks, num_kv_blocks)

    return block_sparse_mask, scores


def generate_headwise_4D_block_sparse_pattern(
    num_q_heads, num_q_blocks, num_kv_blocks, sparsity, device="cuda"
):
    """
    Generates a head-wise block sparse pattern. Each head gets its own random mask.

    Args:
        h (int): Number of attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        k (int): Number of key-value blocks each query block attends to.
        device (str): The device to create tensors on.

    Returns:
        torch.Tensor: A boolean tensor mask of shape [b, h, num_q_blocks, num_kv_blocks] where b = 1.
        scores (torch.Tensor): A tensor of shape [b, h, num_q_blocks, num_kv_blocks] containing the random attention scores.
    """
    k = max(1, int((sparsity) * num_kv_blocks))
    k = min(k, num_kv_blocks)

    # Create random scores for each query block for each head
    scores = torch.rand(num_q_heads, num_q_blocks, num_kv_blocks, device=device)

    # Get the indices of the top-k scoring key-value blocks for each query block per head per batch
    _, topk_indices = torch.topk(scores, k, dim=-1)

    # Create a boolean mask initialized to all False
    block_sparse_mask = torch.zeros(
        num_q_heads, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # Use scatter_ to efficiently set the corresponding positions to True based on indices
    block_sparse_mask.scatter_(2, topk_indices, True)

    block_sparse_mask = block_sparse_mask.unsqueeze(0)
    scores = scores.unsqueeze(0)

    return block_sparse_mask, scores


def generate_kv_headwise_4D_block_sparse_pattern(
    num_kv_heads, num_q_blocks, num_kv_blocks, sparsity, device="cuda"
):
    """
    Generates a block sparse pattern based on the number of KV heads for GQA.
    All query heads within a group share the same mask.

    Args:
        num_kv_heads (int): Number of Key-Value attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        sparsity (float): The density ratio of connections.
        device (str): The device to create tensors on.

    Returns:
        torch.Tensor: A boolean tensor mask of shape [b, h, num_q_blocks, num_kv_blocks] where b = 1.
        scores (torch.Tensor): A tensor of shape [b, h, num_q_blocks, num_kv_blocks] containing the random attention scores.
    """
    k = max(1, int(sparsity * num_kv_blocks))
    k = min(k, num_kv_blocks)

    # 1. Generate scores based on the number of KV heads, not Q heads
    scores = torch.rand(num_kv_heads, num_q_blocks, num_kv_blocks, device=device)

    # 2. Find top-k for each KV head
    _, topk_indices = torch.topk(scores, k, dim=-1)

    # 3. Create mask with the shape of KV heads
    block_sparse_mask = torch.zeros(
        num_kv_heads, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # 4. Scatter True values
    block_sparse_mask.scatter_(2, topk_indices, True)

    # 5. Add batch dimension
    block_sparse_mask = block_sparse_mask.unsqueeze(0)
    scores = scores.unsqueeze(0)

    return block_sparse_mask, scores


def generate_headwise_block_sparse_pattern(
    h, num_q_blocks, num_kv_blocks, sparsity, device="cuda"
):
    """
    Generates a head-wise block sparse pattern. Each head gets its own random mask.

    Args:
        h (int): Number of attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        k (int): Number of key-value blocks each query block attends to.
        device (str): The device to create tensors on.

    Returns:
        torch.Tensor: A boolean tensor mask of shape [h, num_q_blocks, num_kv_blocks].
    """
    k = max(1, int((sparsity) * num_kv_blocks))
    k = min(k, num_kv_blocks)

    # Create random scores for each query block for each head
    scores = torch.rand(h, num_q_blocks, num_kv_blocks, device=device)

    # Get the indices of the top-k scoring key-value blocks for each query block per head
    _, topk_indices = torch.topk(scores, k, dim=-1)

    # Create a boolean mask initialized to all False
    block_sparse_mask = torch.zeros(
        h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # Use scatter_ to efficiently set the corresponding positions to True based on indices
    block_sparse_mask.scatter_(2, topk_indices, True)

    return block_sparse_mask


def flatten_head_mask(mask_4d: torch.Tensor) -> torch.Tensor:
    """
    Flattens a q head-wise 4D block mask into a single 2D block mask.
    This creates a block-diagonal mask for the flattened Q, K, V tensors.

    Args:
        mask_3d (torch.Tensor): The input 3D mask of shape [h, num_q_blocks, num_k_blocks].

    Returns:
        torch.Tensor: The output 2D mask of shape [h * num_q_blocks, h * num_k_blocks].
    """
    b, h, num_q, num_k = mask_4d.shape
    num_q_flat = h * num_q
    num_k_flat = h * num_k

    # Find the coordinates of all True elements in the 3D mask (h_idx, q_idx, k_idx)
    b_indices, h_indices, q_indices, k_indices = torch.nonzero(mask_4d, as_tuple=True)

    # Map the 3D coordinates to the flattened 2D coordinates
    # q_flat_idx = q_idx + h_idx * num_q
    # k_flat_idx = k_idx + h_idx * num_k
    q_indices_flat = q_indices + h_indices * num_q
    k_indices_flat = k_indices + h_indices * num_k

    # Create an empty 2D mask and populate it
    mask_flat = torch.zeros(
        num_q_flat, num_k_flat, dtype=torch.bool, device=mask_4d.device
    )
    mask_flat[q_indices_flat, k_indices_flat] = True

    return mask_flat


def flatten_kvhead_mask(
    mask_4d: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """
    Flattens a kv head-wise 4D block mask into a single 2D block mask.
    This creates a block-diagonal mask for the flattened Q, K, V tensors.

    Args:
        mask_4d (torch.Tensor): The input 3D mask of shape [h, num_q_blocks, num_k_blocks].

    Returns:
        torch.Tensor: The output 2D mask of shape [h * num_q_blocks, h * num_k_blocks].
    """
    b, h_q, num_q, num_k = mask_4d.shape
    num_groups = num_q_heads // num_kv_heads

    # Find the coordinates of all True elements in the 3D mask (h_idx, q_idx, k_idx)
    b_indices, h_indices, q_indices, k_indices = torch.nonzero(mask_4d, as_tuple=True)

    q_indices_flat = q_indices + h_indices * num_q
    kv_head_indices = h_indices // num_groups
    k_indices_flat = k_indices + kv_head_indices * num_k

    # Create an empty 2D mask and populate it
    mask_flat = torch.zeros(
        num_q_heads * num_q,
        num_kv_heads * num_k,
        dtype=torch.bool,
        device=mask_4d.device,
    )

    mask_flat[q_indices_flat, k_indices_flat] = True

    return mask_flat


def generate_ranges_from_mask(
    block_mask: torch.Tensor, block_m: int, block_n: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates query and key sequence ranges from a 2D boolean block mask.

    For each `True` value at `block_mask[i, j]`, this function generates a
    corresponding query range [i * block_m, (i + 1) * block_m] and
    key range [j * block_n, (j + 1) * block_n].

    Args:
        block_mask (torch.Tensor): A 2D boolean tensor of shape [num_q_blocks, num_k_blocks].
        block_m (int): The size of each query block.
        block_n (int): The size of each key block.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - q_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the query ranges.
            - k_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the key ranges.
    """
    # 1. Find the coordinates (i, j) of all True elements
    true_indices = torch.nonzero(block_mask, as_tuple=False)

    if true_indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int32), torch.empty(
            (0, 2), dtype=torch.int32
        )

    # 2. Separate the row indices (q_block_indices) and column indices (k_block_indices)
    q_block_indices = true_indices[:, 0]
    k_block_indices = true_indices[:, 1]

    # 3. Vectorize the calculation of all q_ranges
    q_starts = q_block_indices * block_m
    q_ends = q_starts + block_m
    q_range_tensor = torch.stack([q_starts, q_ends], dim=1)

    # 4. Vectorize the calculation of all k_ranges
    k_starts = k_block_indices * block_n
    k_ends = k_starts + block_n
    k_range_tensor = torch.stack([k_starts, k_ends], dim=1)

    return q_range_tensor.int(), k_range_tensor.int()


def generate_gqa_ranges_from_3d_mask(
    mask_3d: torch.Tensor,
    block_m: int,
    block_n: int,
    num_q_heads: int,
    num_k_heads: int,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A more efficient function that directly generates the final q_ranges and k_ranges
    from a 3D head-wise mask, with native support for GQA.

    It avoids creating a large intermediate 2D mask, thus saving memory and computation.

    Args:
        mask_3d (torch.Tensor): A boolean mask of shape [num_q_heads, num_q_blocks, num_kv_blocks].
                                Note: The first dimension is the number of query heads.
        block_m (int): The size of a Q block.
        block_n (int): The size of a K/V block.
        num_q_heads (int): The total number of query heads.
        num_k_heads (int): The total number of key/value heads.
        seq_len (int): The original (non-flattened) sequence length.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The q_ranges and k_ranges that can be directly used in ffa_func.
    """
    # Check if GQA parameters are valid
    if num_q_heads % num_k_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_k_heads for GQA.")

    gqa_group_size = num_q_heads // num_k_heads

    # 1. Directly find the coordinates (q_head_idx, q_block_idx, k_block_idx) of all blocks
    #    where attention needs to be computed from the 3D mask.
    #    This is the key step, as we operate directly in the 3D space.
    q_head_indices, q_block_indices, k_block_indices = torch.nonzero(
        mask_3d, as_tuple=True
    )

    if q_head_indices.numel() == 0:
        return torch.empty(
            (0, 2), dtype=torch.long, device=mask_3d.device
        ), torch.empty((0, 2), dtype=torch.long, device=mask_3d.device)

    # 2. Calculate q_ranges
    #    Physical offset for q = q_head_idx * seq_len
    #    Intra-block offset for q = q_block_idx * block_m
    q_starts = q_head_indices * seq_len + q_block_indices * block_m
    q_ends = q_starts + block_m
    q_range_tensor = torch.stack([q_starts, q_ends], dim=1)

    # 3. Calculate k_ranges, taking GQA into account
    #    First, find the corresponding K/V head index for each Q head
    k_head_indices = q_head_indices // gqa_group_size

    #    Physical offset for k = k_head_idx * seq_len
    #    Intra-block offset for k = k_block_idx * block_n
    k_starts = k_head_indices * seq_len + k_block_indices * block_n
    k_ends = k_starts + block_n
    k_range_tensor = torch.stack([k_starts, k_ends], dim=1)

    return q_range_tensor.int(), k_range_tensor.int()


def get_sdpa_mask_from_block_sparse_mask(
    block_mask: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    block_size_q: int,
    block_size_k: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Converts a block-level sparse mask to an element-level boolean mask
    that is compatible with SDPA (scaled_dot_product_attention).

    Args:
        block_mask (torch.Tensor): The block mask of shape [H, num_q_blocks, num_k_blocks].
        seq_len_q (int): The full length of the query sequence.
        seq_len_k (int): The full length of the key/value sequence.
        block_size_q (int): The size of a Q block.
        block_size_k (int): The size of a K block.
        batch_size (int): The batch size.

    Returns:
        torch.Tensor: An SDPA-compatible mask of shape [B, H, S_q, S_k].
    """
    num_heads = block_mask.shape[1]
    device = block_mask.device

    # 1. Create a large 4D mask of the target shape, filled with False.
    #    This is our "canvas", where False means all positions are masked out by default.
    sdpa_mask = torch.zeros(
        (batch_size, num_heads, seq_len_q, seq_len_k), dtype=torch.bool, device=device
    )

    # 2. Efficiently find the coordinates (h, q_block, k_block) of all blocks to be activated.
    _, h_indices, qb_indices, kb_indices = torch.nonzero(block_mask, as_tuple=True)

    # 3. Iterate through all activated blocks.
    for h, qb, kb in zip(h_indices, qb_indices, kb_indices):
        # Calculate the start and end coordinates for this block in the element-level mask.
        q_start, q_end = qb * block_size_q, (qb + 1) * block_size_q
        k_start, k_end = kb * block_size_k, (kb + 1) * block_size_k

        # "Paint" the corresponding rectangular region on the canvas to True,
        # indicating that attention is allowed for these positions.
        sdpa_mask[:, h, q_start:q_end, k_start:k_end] = True

    return sdpa_mask


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


# ================ Utils for Variable Block Sparse Attention ================


def get_random_variable_block_mask(
    seq_len_q: int,
    seq_len_k: int,
    num_blocks_row: int,
    num_blocks_col: int,
    num_kv_heads: int,
    sparsity_ratio: float = 0.8,
    bsz: int = 1,
    device: torch.device | str = "cuda",
):
    """
    NOTE: generated by num_kv_heads, same as FlashInfer.
    """

    def random_partition_batch(
        seq_len: int,
        num_blocks: int,
        bsz: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        assert seq_len >= num_blocks
        sizes = torch.empty((bsz, num_blocks), dtype=dtype, device=device)
        for i in range(bsz):
            cut_pts = torch.randperm(seq_len - 1, device=device)[: num_blocks - 1] + 1
            cut_pts, _ = torch.sort(cut_pts)
            row_sizes = torch.diff(
                torch.cat(
                    (
                        torch.tensor([0], device=device),
                        cut_pts,
                        torch.tensor([seq_len], device=device),
                    )
                )
            )
            sizes[i] = row_sizes

        assert sizes.min() >= 1
        assert sizes.max() <= seq_len
        assert torch.all(sizes.sum(dim=-1) == seq_len)

        return sizes

    block_row_sz = random_partition_batch(
        seq_len_q, num_blocks_row, num_kv_heads, device=device
    )

    block_col_sz = random_partition_batch(
        seq_len_k, num_blocks_col, num_kv_heads, device=device
    )
    # TODO: modify to topk block selection
    block_mask_map = (
        torch.rand(num_kv_heads, num_blocks_row, num_blocks_col, device=device)
        < sparsity_ratio
    )
    block_mask_map = block_mask_map.unsqueeze(0)  # add batch dimension

    return block_mask_map, block_row_sz, block_col_sz


def generate_ranges_from_var_block_mask(
    block_mask: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates query and key sequence ranges from a 2D "flattened" variable-size block mask,
    assuming a varlen-style sequence concatenation across heads.

    This function interprets a 2D mask where heads are tiled, and their corresponding
    sequences are concatenated. The ranges for head `h` are offset by the total
    sequence length of all preceding heads.

    Args:
        block_mask (torch.Tensor): A 2D boolean tensor of shape
                                   [num_heads * num_q_blocks, num_heads * num_k_blocks].
        block_row_sz (torch.Tensor): A 2D tensor of shape [num_heads, num_q_blocks]
                                     defining the height of each query block per head.
        block_col_sz (torch.Tensor): A 2D tensor of shape [num_heads, num_k_blocks]
                                     defining the width of each key block per head.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - q_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2]
                                             listing the query ranges [start, end).
            - k_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2]
                                             listing the key ranges [start, end).
    """
    device = block_mask.device
    num_heads, num_q_blocks = block_row_sz.shape
    _, num_k_blocks = block_col_sz.shape

    # --- 1. Calculate intra-head and inter-head offsets ---

    # Intra-head offsets (offsets within each head's own sequence)
    zeros_col = torch.zeros((num_heads, 1), dtype=block_row_sz.dtype, device=device)
    row_offsets_intra = torch.cat([zeros_col, torch.cumsum(block_row_sz, dim=1)], dim=1)
    col_offsets_intra = torch.cat([zeros_col, torch.cumsum(block_col_sz, dim=1)], dim=1)

    # Inter-head offsets (the starting position of each head in the concatenated sequence)
    zero = torch.tensor([0], dtype=torch.long, device=device)
    q_len_per_head = torch.sum(block_row_sz, dim=1)
    k_len_per_head = torch.sum(block_col_sz, dim=1)
    q_head_start_offsets = torch.cat([zero, torch.cumsum(q_len_per_head, dim=0)[:-1]])
    k_head_start_offsets = torch.cat([zero, torch.cumsum(k_len_per_head, dim=0)[:-1]])

    # --- 2. Find the coordinates (flat_i, flat_j) from the 2D mask ---
    flat_q_indices, flat_k_indices = torch.nonzero(block_mask, as_tuple=True)

    # Handle case with no active blocks
    if flat_q_indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int32, device=device), torch.empty(
            (0, 2), dtype=torch.int32, device=device
        )

    # --- 3. Map flat indices back to head and block indices ---
    h_indices_q = flat_q_indices // num_q_blocks
    q_block_indices = flat_q_indices % num_q_blocks

    h_indices_k = flat_k_indices // num_k_blocks
    k_block_indices = flat_k_indices % num_k_blocks

    # --- 4. Filter out cross-head attention blocks ---
    intra_head_mask = h_indices_q == h_indices_k

    h_indices = h_indices_q[intra_head_mask]
    q_block_indices = q_block_indices[intra_head_mask]
    k_block_indices = k_block_indices[intra_head_mask]

    # --- 5. Gather ranges, applying both inter-head and intra-head offsets ---
    q_starts = (
        row_offsets_intra[h_indices, q_block_indices] + q_head_start_offsets[h_indices]
    )
    q_ends = (
        row_offsets_intra[h_indices, q_block_indices + 1]
        + q_head_start_offsets[h_indices]
    )
    q_range_tensor = torch.stack([q_starts, q_ends], dim=1)

    k_starts = (
        col_offsets_intra[h_indices, k_block_indices] + k_head_start_offsets[h_indices]
    )
    k_ends = (
        col_offsets_intra[h_indices, k_block_indices + 1]
        + k_head_start_offsets[h_indices]
    )
    k_range_tensor = torch.stack([k_starts, k_ends], dim=1)

    return q_range_tensor.int(), k_range_tensor.int()


def get_sdpa_mask_from_var_block_mask(
    block_mask: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    seq_len_q: int,
    seq_len_k: int,
    bsz: int = 1,
) -> torch.Tensor:
    """
    Generates a standard SDPA (Scaled Dot Product Attention) mask from a
    variable block sparse attention specification.

    This function converts a block-level sparse definition into an element-level
    dense boolean mask, which can be directly used with
    `torch.nn.functional.scaled_dot_product_attention`.

    Args:
        block_mask (Tensor): A boolean or integer tensor of shape
                             [num_heads, num_q_blocks, num_kv_blocks].
                             A value of 1 (True) indicates that the corresponding
                             block's attention should be computed.
        block_row_sz (Tensor): A tensor of shape [num_heads, num_q_blocks]
                               that defines the height of each query block.
        block_col_sz (Tensor): A tensor of shape [num_heads, num_kv_blocks]
                               that defines the width of each key/value block.
        seq_len_q (int): The total length of the query sequence.
        seq_len_k (int): The total length of the key/value sequence.
        bsz (int, optional): The batch size. Defaults to 1.

    Returns:
        Tensor: A boolean mask of shape [bsz, num_heads, seq_len_q, seq_len_k].
                A value of True allows attention computation, while False forbids it.
    """
    num_heads = block_mask.shape[0]
    device = block_mask.device

    # --- 1. Pre-calculate start and end offsets for each block ---
    # We use cumulative sum to find the boundaries of each block.
    # To easily get the start positions, we concatenate a zero column at the beginning
    # of the cumulative sum result.
    # e.g., cumsum([5, 4, 6]) -> [5, 9, 15]
    # After concatenation -> [0, 5, 9, 15]. The range for block `i` is from
    # offsets[i] to offsets[i+1].

    # Create a column of zeros for concatenation
    zeros_col_shape = (num_heads, 1)
    zeros = torch.zeros(zeros_col_shape, dtype=block_row_sz.dtype, device=device)

    # Calculate row (query) offsets
    row_cumsum = torch.cumsum(block_row_sz, dim=1)
    row_offsets = torch.cat([zeros, row_cumsum], dim=1)

    # Calculate column (key/value) offsets
    col_cumsum = torch.cumsum(block_col_sz, dim=1)
    col_offsets = torch.cat([zeros, col_cumsum], dim=1)

    # --- 2. Initialize the final SDPA mask ---
    sdpa_mask = torch.zeros(
        (bsz, num_heads, seq_len_q, seq_len_k), dtype=torch.bool, device=device
    )

    # --- 3. Efficiently find the coordinates (h, qb, kb) of all active blocks ---
    h_indices, qb_indices, kb_indices = torch.nonzero(block_mask, as_tuple=True)

    row_offsets_list = row_offsets.tolist()
    col_offsets_list = col_offsets.tolist()

    # --- 5. Iterate through all active blocks and populate the mask ---
    for h, qb, kb in zip(h_indices, qb_indices, kb_indices):
        # Use fast Python list indexing to get the block boundaries.
        q_start = row_offsets_list[h][qb]
        q_end = row_offsets_list[h][qb + 1]

        k_start = col_offsets_list[h][kb]
        k_end = col_offsets_list[h][kb + 1]

        # "Paint" the corresponding rectangular region to True,
        # indicating that attention is allowed for these positions.
        sdpa_mask[:, h, q_start:q_end, k_start:k_end] = True

    return sdpa_mask
