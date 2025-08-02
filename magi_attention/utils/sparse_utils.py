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

import torch


def generate_headwise_4D_block_sparse_pattern(
    num_q_heads, num_q_blocks, num_kv_blocks, sparsity, device="cuda"
):
    """
    Generates a head-wise block sparse pattern. Each query head gets its own random mask.

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

    return block_sparse_mask, scores


def flatten_head_mask(mask_4d: torch.Tensor) -> torch.Tensor:
    """
    Flattens a q head-wise 4D block mask into a single 2D block mask.
    This creates a block-diagonal mask for the flattened Q, K, V tensors.

    Args:
        mask_4d (torch.Tensor): The input 4D mask of shape [b, h, num_q_blocks, num_k_blocks].

    Returns:
        torch.Tensor: The output 2D mask of shape [b * h * num_q_blocks, b * h * num_k_blocks].
    """
    b, h, num_q, num_k = mask_4d.shape
    num_q_flat = h * num_q
    num_k_flat = h * num_k

    # Find the coordinates of all True elements in the 3D mask (h_idx, q_idx, k_idx)
    _, h_indices, q_indices, k_indices = torch.nonzero(mask_4d, as_tuple=True)

    # Map the 4D coordinates to the flattened 2D coordinates
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
        mask_3d (torch.Tensor): The input 4D mask of shape [b, h, num_q_blocks, num_k_blocks].

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


def generate_ranges_from_block_mask(
    block_mask: torch.Tensor, block_m: int, block_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates query and key range tensor from a 2D boolean block mask.

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
