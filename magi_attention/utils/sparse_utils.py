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

from typing import Tuple

import torch

# ================ Utils for Block Sparse Attention ================


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


# ================ Utils for Variable Block Sparse Attention ================


def get_random_variable_block_mask(
    seq_len_q: int,
    seq_len_k: int,
    num_blocks_row: int,
    num_blocks_col: int,
    num_kv_heads: int,
    min_q_block_size: int = 128,
    min_kv_block_size: int = 128,
    sparsity_ratio: float = 0.8,
    bsz: int = 1,
    device: torch.device | str = "cuda",
):
    """
    Generates a random variable-size block sparse mask, ensuring minimum block sizes.

    NOTE: Block sizes (block_row_sz, block_col_sz) are generated per head,
          which is consistent with libraries like FlashInfer.
    """

    def random_partition_with_min_size(
        seq_len: int,
        num_blocks: int,
        min_block_size: int,
        batch_size: int,  # Represents num_heads or bsz
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """
        Partitions a sequence into random-sized blocks, with a guaranteed minimum size.
        This implementation is fully vectorized.
        """
        # 1. Validate that the partitioning is possible
        if seq_len < num_blocks * min_block_size:
            raise ValueError(
                f"Cannot partition seq_len {seq_len} into {num_blocks} blocks "
                f"with min_block_size {min_block_size}. "
                f"Required: {num_blocks * min_block_size}."
            )

        # 2. Distribute the "slack" length
        # First, give each block its minimum required size.
        # The remaining length is what we'll distribute randomly.
        extra_len = seq_len - num_blocks * min_block_size

        # 3. Generate random cut points for the extra length
        # We need to partition extra_len into num_blocks parts.
        # This is done by choosing num_blocks - 1 cut points from [0, extra_len].
        cut_pts = torch.randint(
            0, extra_len + 1, (batch_size, num_blocks - 1), device=device
        )
        cut_pts, _ = torch.sort(cut_pts, dim=-1)

        # 4. Calculate sizes of the extra partitions using torch.diff
        zeros = torch.zeros((batch_size, 1), dtype=cut_pts.dtype, device=device)
        extras = torch.full(
            (batch_size, 1), extra_len, dtype=cut_pts.dtype, device=device
        )

        boundaries = torch.cat([zeros, cut_pts, extras], dim=-1)
        extra_sizes = torch.diff(boundaries, dim=-1)

        # 5. Add the minimum size back to each block
        final_sizes = extra_sizes + min_block_size

        # Final assertions to ensure correctness
        assert final_sizes.min() >= min_block_size
        assert torch.all(final_sizes.sum(dim=-1) == seq_len)
        return final_sizes.to(dtype)

    # Generate block sizes for rows (queries) and columns (keys) for each head
    block_row_sz = random_partition_with_min_size(
        seq_len=seq_len_q,
        num_blocks=num_blocks_row,
        min_block_size=min_q_block_size,
        batch_size=num_kv_heads,
        device=device,
    )

    block_col_sz = random_partition_with_min_size(
        seq_len=seq_len_k,
        num_blocks=num_blocks_col,
        min_block_size=min_kv_block_size,
        batch_size=num_kv_heads,
        device=device,
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
    seq_len_q: int,
    seq_len_k: int,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
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
                             [batch_size, num_heads, num_q_blocks, num_kv_blocks].
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
    num_heads = block_mask.shape[1]
    device = block_mask.device

    # TODO: assume batch size is 1 for now
    block_mask = block_mask.squeeze(0)

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
