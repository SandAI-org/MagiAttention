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
        torch.Tensor: A boolean tensor mask of shape [h, num_q_blocks, num_kv_blocks].
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

    return block_sparse_mask


def generate_headwise_block_sparse_pattern(
    h, num_q_blocks, num_kv_blocks, k, device="cuda"
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


def flatten_head_mask(mask_3d: torch.Tensor) -> torch.Tensor:
    """
    Flattens a head-wise 3D block mask into a single 2D block mask.
    This creates a block-diagonal mask for the flattened Q, K, V tensors.

    Args:
        mask_3d (torch.Tensor): The input 3D mask of shape [h, num_q_blocks, num_k_blocks].

    Returns:
        torch.Tensor: The output 2D mask of shape [h * num_q_blocks, h * num_k_blocks].
    """
    h, num_q, num_k = mask_3d.shape
    num_q_flat = h * num_q
    num_k_flat = h * num_k

    # Find the coordinates of all True elements in the 3D mask (h_idx, q_idx, k_idx)
    h_indices, q_indices, k_indices = torch.nonzero(mask_3d, as_tuple=True)

    # Map the 3D coordinates to the flattened 2D coordinates
    # q_flat_idx = q_idx + h_idx * num_q
    # k_flat_idx = k_idx + h_idx * num_k
    q_indices_flat = q_indices + h_indices * num_q
    k_indices_flat = k_indices + h_indices * num_k

    # Create an empty 2D mask and populate it
    mask_flat = torch.zeros(
        num_q_flat, num_k_flat, dtype=torch.bool, device=mask_3d.device
    )
    mask_flat[q_indices_flat, k_indices_flat] = True

    return mask_flat


def generate_ranges_from_mask(
    block_mask: torch.Tensor, block_m: int, block_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - q_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the query ranges.
            - k_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the key ranges.
    """
    # 1. Find the coordinates (i, j) of all True elements
    true_indices = torch.nonzero(block_mask, as_tuple=False)

    if true_indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long), torch.empty(
            (0, 2), dtype=torch.long
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
        tuple[torch.Tensor, torch.Tensor]: The q_ranges and k_ranges that can be directly used in ffa_func.
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
    num_heads = block_mask.shape[0]
    device = block_mask.device

    # 1. Create a large 4D mask of the target shape, filled with False.
    #    This is our "canvas", where False means all positions are masked out by default.
    sdpa_mask = torch.zeros(
        (batch_size, num_heads, seq_len_q, seq_len_k), dtype=torch.bool, device=device
    )

    # 2. Efficiently find the coordinates (h, q_block, k_block) of all blocks to be activated.
    h_indices, qb_indices, kb_indices = torch.nonzero(block_mask, as_tuple=True)

    # 3. Iterate through all activated blocks.
    for h, qb, kb in zip(h_indices, qb_indices, kb_indices):
        # Calculate the start and end coordinates for this block in the element-level mask.
        q_start, q_end = qb * block_size_q, (qb + 1) * block_size_q
        k_start, k_end = kb * block_size_k, (kb + 1) * block_size_k

        # "Paint" the corresponding rectangular region on the canvas to True,
        # indicating that attention is allowed for these positions.
        sdpa_mask[:, h, q_start:q_end, k_start:k_end] = True

    return sdpa_mask
