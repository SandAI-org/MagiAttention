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

# ================ Utils for Block Sparse Attention ================


def generate_block_sparse_pattern(
    num_q_heads: int,
    num_kv_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparsity: float,
    mode: str = "per_kv_head",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a head-wise block sparse pattern, supporting both MHA and GQA semantics.

    The final returned mask is always of shape [1, num_q_heads, num_q_blocks, num_kv_blocks].

    Args:
        num_q_heads (int): Total number of query attention heads.
        num_kv_heads (int): Total number of key-value attention heads.
        num_q_blocks (int): Number of query blocks per head.
        num_kv_blocks (int): Number of key-value blocks per head.
        sparsity (float): The density ratio of connections.
        mode (str("per_q_head", "per_kv_head")):
            - "per_q_head": Each query head gets a unique random mask (for MHA).
            - "per_kv_head": Query heads in the same group share a mask (for GQA).
        device (str): The device to create tensors on.

    Returns:
        torch.Tensor: A boolean tensor mask of shape [1, num_q_heads, num_q_blocks, num_kv_blocks].
        torch.Tensor: A tensor containing the random scores used for selection,
                      shape is [1, num_mask_heads, num_q_blocks, num_kv_blocks],
                      where num_mask_heads is num_q_heads or num_kv_heads based on mode.
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    k = max(1, int(sparsity * num_kv_blocks))
    k = min(k, num_kv_blocks)

    if mode == "per_q_head":
        # Each Q head gets its own mask. This is equivalent to GQA where num_groups=num_q_heads.
        num_mask_heads = num_q_heads
    elif mode == "per_kv_head":
        # Masks are generated per KV head.
        num_mask_heads = num_kv_heads
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 1. Create random scores based on the number of heads specified by the mode
    scores = torch.rand(num_mask_heads, num_q_blocks, num_kv_blocks, device=device)

    # 2. Get the indices of the top-k scoring key-value blocks
    _, topk_indices = torch.topk(scores, k, dim=-1)

    # 3. Create a boolean base mask initialized to all False
    base_mask = torch.zeros(
        num_mask_heads, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )

    # 4. Use scatter_ to efficiently set the corresponding positions to True
    base_mask.scatter_(2, topk_indices, True)

    # 5. Expand mask if generated at KV-head granularity for GQA
    if mode == "per_kv_head" and num_q_heads != num_kv_heads:
        num_groups = num_q_heads // num_kv_heads
        # Repeat the mask for each Q head in the group
        block_sparse_mask = torch.repeat_interleave(
            base_mask, repeats=num_groups, dim=0
        )
    else:
        block_sparse_mask = base_mask

    # 6. Add batch dimension
    block_sparse_mask = block_sparse_mask.unsqueeze(0)
    scores = scores.unsqueeze(0)

    return block_sparse_mask, scores


def flatten_block_mask(
    mask_4d: torch.Tensor, num_q_heads: int, num_kv_heads: int
) -> torch.Tensor:
    """
    Flattens a 4D block mask (MHA or GQA) into a single 2D block mask.

    This function correctly handles both standard multi-head attention (MHA),
    where num_q_heads == num_kv_heads, and grouped-query attention (GQA).

    Args:
        mask_4d (torch.Tensor): The input 4D mask of shape [B, num_q_heads, num_q_blocks, num_k_blocks].
                                B (batch size) is assumed to be 1.
        num_q_heads (int): Total number of query heads.
        num_kv_heads (int): Total number of key-value heads.

    Returns:
        torch.Tensor: The output 2D mask of shape
                      [num_q_heads * num_q_blocks, num_kv_heads * num_k_blocks].
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    b, h_q, num_q, num_k = mask_4d.shape
    if b != 1:
        # This implementation assumes a batch size of 1 for simplicity.
        # It can be extended if multi-batch support is needed.
        raise ValueError("Batch size for mask flattening must be 1.")
    if h_q != num_q_heads:
        raise ValueError(
            "Mask dimension mismatch: mask_4d.shape[1] should equal num_q_heads."
        )

    num_groups = num_q_heads // num_kv_heads
    num_q_flat = num_q_heads * num_q
    num_k_flat = num_kv_heads * num_k

    # Find the coordinates of all True elements in the mask
    # We ignore the batch dimension as we assume it's 1.
    _, h_indices_q, q_indices, k_indices = torch.nonzero(mask_4d, as_tuple=True)

    # Map the query head and block indices to a flat query index
    q_indices_flat = q_indices + h_indices_q * num_q

    # Determine the corresponding KV head index for each Q head
    h_indices_kv = h_indices_q // num_groups

    # Map the KV head and block indices to a flat key index
    k_indices_flat = k_indices + h_indices_kv * num_k

    # Create an empty 2D mask and populate it
    mask_flat = torch.zeros(
        num_q_flat, num_k_flat, dtype=torch.bool, device=mask_4d.device
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
    seqlen_q: int,
    seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Converts a block-level sparse mask to an element-level boolean mask
    that is compatible with SDPA (scaled_dot_product_attention).

    Args:
        block_mask (torch.Tensor): The block mask of shape [H, num_q_blocks, num_k_blocks].
        seqlen_q (int): The full length of the query sequence.
        seqlen_k (int): The full length of the key/value sequence.
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
        (batch_size, num_heads, seqlen_q, seqlen_k), dtype=torch.bool, device=device
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


def generate_variable_block_sparse_pattern(
    num_q_heads: int,
    num_kv_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparsity: float,
    mode: str = "per_kv_head",
    min_q_block_size: int = 64,
    min_kv_block_size: int = 64,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates a variable-size block sparse pattern with top-k sparsity,
    supporting both MHA and GQA semantics.

    Args:
        num_q_heads (int): The total number of query heads.
        num_kv_heads (int): The total number of key/value heads. For GQA, this is
            less than num_q_heads. For MHA, this must be equal to num_q_heads.
        seqlen_q (int): The sequence length of the query tensor.
        seqlen_k (int): The sequence length of the key/value tensor.
        num_q_blocks (int): The number of blocks to partition the query sequence into.
        num_kv_blocks (int): The number of blocks to partition the key/value sequence into.
        sparsity (float): The target sparsity level, between 0.0 and 1.0. This
            determines the number of KV blocks to keep for each row of Q blocks.
            Specifically, k = int(sparsity * num_kv_blocks).
        mode (str, optional): The mask generation mode. Defaults to "per_kv_head".
            - "per_kv_head": Generates one sparsity mask per KV head and repeats
              it for all associated Q heads. This is the standard and efficient
              approach for GQA.
            - "per_q_head": Generates a unique sparsity mask for each Q head. This
              is functionally equivalent to MHA.
        min_q_block_size (int, optional): The minimum size of each block along the
            query sequence. Defaults to 64.
        min_kv_block_size (int, optional): The minimum size of each block along the
            key/value sequence. Defaults to 64.
        device (str | torch.device, optional): The device on which to create
            the tensors. Defaults to "cuda".

    Raises:
        ValueError: If `num_q_heads` is not divisible by `num_kv_heads`.
        ValueError: If a sequence length cannot be partitioned into the specified
            number of blocks with the given minimum block size.
        ValueError: If an unknown `mode` is provided.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
        - final_mask (torch.Tensor): A boolean block-sparse attention mask of
          shape `(1, num_q_heads, num_q_blocks, num_kv_blocks)`. `True` indicates
          that a block should be computed.
        - final_block_row_sz (torch.Tensor): A tensor of shape
          `(num_q_heads, num_q_blocks)` containing the variable sizes of each
          query block for each query head.
        - final_block_col_sz (torch.Tensor): A tensor of shape
          `(num_kv_heads, num_kv_blocks)` containing the variable sizes of each
          key/value block for each key/value head.
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    # --- 2. Generate random block size layouts ---
    def random_partition_with_min_size(
        seqlen: int,
        num_blocks: int,
        min_block_size: int,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        if seqlen < num_blocks * min_block_size:
            raise ValueError(
                f"Cannot partition seqlen {seqlen} into {num_blocks} blocks with min_size {min_block_size}."
            )
        extra_len = seqlen - num_blocks * min_block_size
        cut_pts = torch.randint(
            0, extra_len + 1, (batch_size, num_blocks - 1), device=device
        )
        cut_pts, _ = torch.sort(cut_pts, dim=-1)
        zeros = torch.zeros((batch_size, 1), dtype=cut_pts.dtype, device=device)
        extras = torch.full(
            (batch_size, 1), extra_len, dtype=cut_pts.dtype, device=device
        )
        boundaries = torch.cat([zeros, cut_pts, extras], dim=-1)
        extra_sizes = torch.diff(boundaries, dim=-1)
        final_sizes = extra_sizes + min_block_size
        return final_sizes.to(dtype)

    # Generate row sizes. For GQA, we need to generate for each KV head and then expand.
    if mode == "per_kv_head":
        base_block_row_sz = random_partition_with_min_size(
            seqlen_q, num_q_blocks, min_q_block_size, num_kv_heads, device
        )
        final_block_row_sz = torch.repeat_interleave(
            base_block_row_sz, num_q_heads // num_kv_heads, dim=0
        )
    elif mode == "per_q_head":  # MHA mode
        final_block_row_sz = random_partition_with_min_size(
            seqlen_q, num_q_blocks, min_q_block_size, num_q_heads, device
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Generate col sizes. This should ALWAYS be based on KV heads.
    final_block_col_sz = random_partition_with_min_size(
        seqlen_k, num_kv_blocks, min_kv_block_size, num_kv_heads, device
    )

    # --- 3. Generate block mask using top-k ---
    # The base mask should match the granularity of the mode
    num_gen_mask_heads = num_kv_heads if mode == "per_kv_head" else num_q_heads
    k = max(1, int(sparsity * num_kv_blocks))
    k = min(k, num_kv_blocks)

    scores = torch.rand(num_gen_mask_heads, num_q_blocks, num_kv_blocks, device=device)
    _, topk_indices = torch.topk(scores, k, dim=-1)

    base_mask = torch.zeros(
        num_gen_mask_heads, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    base_mask.scatter_(2, topk_indices, True)

    # --- 4. Expand mask for GQA if necessary ---
    if mode == "per_kv_head" and num_q_heads != num_kv_heads:
        final_mask = torch.repeat_interleave(
            base_mask, num_q_heads // num_kv_heads, dim=0
        )
    else:
        final_mask = base_mask

    # --- 5. Add batch dimension and return ---
    final_mask = final_mask.unsqueeze(0)

    # The shapes are now perfectly aligned with the requirements of the downstream functions
    # final_mask: [1, num_q_heads, Nq, Nk]
    # final_block_row_sz: [num_q_heads, Nq]
    # final_block_col_sz: [num_kv_heads, Nk]
    return final_mask, final_block_row_sz, final_block_col_sz


def generate_ranges_from_var_block_mask(
    block_mask: torch.Tensor,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates query and key sequence ranges from a 2D "flattened" variable-size
    block mask, correctly handling both MHA and GQA scenarios.

    This function assumes sequences are concatenated across heads.

    Args:
        block_mask (torch.Tensor): A 2D boolean tensor of shape
                                   [num_q_heads * num_q_blocks, num_kv_heads * num_k_blocks].
                                   This mask should be generated by flatten_block_mask.
        block_row_sz (torch.Tensor): A 2D tensor of shape [num_q_heads, num_q_blocks]
                                     defining the height of each query block per head.
        block_col_sz (torch.Tensor): A 2D tensor of shape [num_kv_heads, num_k_blocks]
                                     defining the width of each key block per head.
        num_q_heads (int): Total number of query heads.
        num_kv_heads (int): Total number of key-value heads.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - q_range_tensor (torch.Tensor): Tensor of shape [num_active_blocks, 2]
                                             listing the query ranges [start, end).
            - k_range_tensor (torch.Tensor): Tensor of shape [num_active_blocks, 2]
                                             listing the key ranges [start, end).
    """
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    device = block_mask.device
    num_groups = num_q_heads // num_kv_heads

    # Extract block counts from the per-head size tensors
    _, num_q_blocks = block_row_sz.shape
    h_kv_for_col, num_k_blocks = block_col_sz.shape

    if h_kv_for_col != num_kv_heads:
        raise ValueError(
            "The head dimension of block_col_sz must be equal to num_kv_heads."
        )

    # --- 1. Find the coordinates (flat_i, flat_j) from the 2D mask ---
    flat_q_indices, flat_k_indices = torch.nonzero(block_mask, as_tuple=True)

    if flat_q_indices.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int32, device=device), torch.empty(
            (0, 2), dtype=torch.int32, device=device
        )

    # --- 2. Reverse-engineer head and block indices from flat indices ---
    # For Query side (related to num_q_heads)
    h_indices_q = flat_q_indices // num_q_blocks
    qb_indices = flat_q_indices % num_q_blocks

    # For Key side (related to num_kv_heads)
    h_indices_k = flat_k_indices // num_k_blocks
    kb_indices = flat_k_indices % num_k_blocks

    # --- 3. Filter out invalid cross-group attention blocks ---
    # A connection is valid only if the Q head belongs to the group of the K head.
    # e.g., in a GQA with group_size=4, Q heads 0,1,2,3 all map to KV head 0.
    # A connection from Q head 5 to KV head 0 would be invalid.
    valid_gqa_mask = (h_indices_q // num_groups) == h_indices_k

    h_indices_q = h_indices_q[valid_gqa_mask]
    qb_indices = qb_indices[valid_gqa_mask]
    h_indices_k = h_indices_k[valid_gqa_mask]
    kb_indices = kb_indices[valid_gqa_mask]

    # --- 4. Calculate intra-head and inter-head offsets ---
    # Intra-head offsets (offsets within each head's own sequence)
    zeros_q = torch.zeros((num_q_heads, 1), dtype=block_row_sz.dtype, device=device)
    row_offsets_intra = torch.cat([zeros_q, torch.cumsum(block_row_sz, dim=1)], dim=1)

    zeros_k = torch.zeros((num_kv_heads, 1), dtype=block_col_sz.dtype, device=device)
    col_offsets_intra = torch.cat([zeros_k, torch.cumsum(block_col_sz, dim=1)], dim=1)

    # Inter-head offsets (start position of each head in the concatenated sequence)
    zero_offset = torch.tensor([0], dtype=torch.long, device=device)
    q_len_per_head = torch.sum(block_row_sz, dim=1)
    k_len_per_head = torch.sum(block_col_sz, dim=1)
    q_head_start_offsets = torch.cat(
        [zero_offset, torch.cumsum(q_len_per_head, dim=0)[:-1]]
    )
    k_head_start_offsets = torch.cat(
        [zero_offset, torch.cumsum(k_len_per_head, dim=0)[:-1]]
    )

    # --- 5. Gather ranges, applying both inter-head and intra-head offsets ---
    q_starts = (
        row_offsets_intra[h_indices_q, qb_indices] + q_head_start_offsets[h_indices_q]
    )
    q_ends = (
        row_offsets_intra[h_indices_q, qb_indices + 1]
        + q_head_start_offsets[h_indices_q]
    )
    q_range_tensor = torch.stack([q_starts, q_ends], dim=1)

    k_starts = (
        col_offsets_intra[h_indices_k, kb_indices] + k_head_start_offsets[h_indices_k]
    )
    k_ends = (
        col_offsets_intra[h_indices_k, kb_indices + 1]
        + k_head_start_offsets[h_indices_k]
    )
    k_range_tensor = torch.stack([k_starts, k_ends], dim=1)

    return q_range_tensor.int(), k_range_tensor.int()


def get_sdpa_mask_from_var_block_mask(
    block_mask: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
    bsz: int = 1,
) -> torch.Tensor:
    """
    Generates a standard SDPA (Scaled Dot Product Attention) mask from a
    variable block sparse attention specification.

    This function is updated to correctly handle GQA where the number of heads
    in block_row_sz and block_col_sz might differ.
    """
    # --- 0. Determine correct head counts for Q and KV ---
    if block_mask.shape[0] != 1:
        raise ValueError("This implementation assumes batch size of block_mask is 1.")

    num_q_heads = block_mask.shape[1]
    num_kv_heads = block_col_sz.shape[0]
    device = block_mask.device

    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
        )

    num_groups = num_q_heads // num_kv_heads

    # --- 1. Pre-calculate start and end offsets for each block ---
    # Create a zero column for Q offsets
    zeros_q = torch.zeros((num_q_heads, 1), dtype=block_row_sz.dtype, device=device)
    row_offsets = torch.cat([zeros_q, torch.cumsum(block_row_sz, dim=1)], dim=1)

    # Create a zero column for K/V offsets, using the correct kv_head count
    zeros_k = torch.zeros((num_kv_heads, 1), dtype=block_col_sz.dtype, device=device)
    col_offsets = torch.cat([zeros_k, torch.cumsum(block_col_sz, dim=1)], dim=1)

    # --- 2. Initialize the final SDPA mask ---
    sdpa_mask = torch.zeros(
        (bsz, num_q_heads, seqlen_q, seqlen_k), dtype=torch.bool, device=device
    )

    # --- 3. Efficiently find the coordinates (h_q, qb, kb) of all active blocks ---
    # We squeeze the batch dim as we assume it's 1
    h_indices_q, qb_indices, kb_indices = torch.nonzero(
        block_mask.squeeze(0), as_tuple=True
    )

    # --- 4. Iterate through all active blocks and populate the mask ---
    for h_q, qb, kb in zip(h_indices_q, qb_indices, kb_indices):
        # Determine the corresponding KV head index for the current Q head
        h_kv = h_q // num_groups

        # Get element-level start/end positions for this block
        q_start = row_offsets[h_q, qb].item()
        q_end = row_offsets[h_q, qb + 1].item()

        # Use the correct kv_head index to get column offsets
        k_start = col_offsets[h_kv, kb].item()
        k_end = col_offsets[h_kv, kb + 1].item()

        # "Paint" the corresponding rectangular region to True
        sdpa_mask[:, h_q, q_start:q_end, k_start:k_end] = True

    return sdpa_mask
