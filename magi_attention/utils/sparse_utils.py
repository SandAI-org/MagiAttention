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

import torch
import triton
import triton.language as tl


@triton.jit
def block_mask_to_qk_ranges_kernel(
    # Pointers
    indices_ptr,  # input: [N, 4] int64 (b, h, q, k)
    q_ranges_ptr,  # output: [N, 2] int32
    k_ranges_ptr,  # output: [N, 2] int32
    # Strides
    stride_idx_0,
    stride_idx_1,  # indices strides
    stride_out_0,
    stride_out_1,  # output strides
    # Shapes & Block sizes
    num_q_blocks,  # Integer
    num_k_blocks,  # Integer
    block_m,  # Integer
    block_n,  # Integer
    n_elements,  # Total number of true blocks (N)
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # indices shape is [N, 4], layout is usually row-major.
    # We need columns 1 (h), 2 (q), 3 (k). Column 0 is batch (assumed 0).
    row_ptr = indices_ptr + offsets * stride_idx_0

    h_ptr = row_ptr + 1 * stride_idx_1
    q_ptr = row_ptr + 2 * stride_idx_1
    k_ptr = row_ptr + 3 * stride_idx_1

    # Load indices
    h = tl.load(h_ptr, mask=mask)
    q = tl.load(q_ptr, mask=mask)
    k = tl.load(k_ptr, mask=mask)

    # Core Logic (Flattening + Range Calculation)
    # Corresponding to: q_indices_flat = q_indices + h_indices * num_q
    #            q_starts = q_indices_flat * block_m

    q_global_idx = q + h * num_q_blocks
    q_start = q_global_idx * block_m
    q_end = q_start + block_m

    k_global_idx = k + h * num_k_blocks
    k_start = k_global_idx * block_n
    k_end = k_start + block_n

    # Store results
    # Output shape [N, 2]. Col 0 = start, Col 1 = end.

    q_out_row_ptr = q_ranges_ptr + offsets * stride_out_0
    tl.store(q_out_row_ptr + 0 * stride_out_1, q_start.to(tl.int32), mask=mask)
    tl.store(q_out_row_ptr + 1 * stride_out_1, q_end.to(tl.int32), mask=mask)

    k_out_row_ptr = k_ranges_ptr + offsets * stride_out_0
    tl.store(k_out_row_ptr + 0 * stride_out_1, k_start.to(tl.int32), mask=mask)
    tl.store(k_out_row_ptr + 1 * stride_out_1, k_end.to(tl.int32), mask=mask)


@triton.jit
def topk_indices_to_qk_ranges_kernel(
    topk_indices_ptr,  # input: [total_num_q_blocks, kv_heads, topk_k_blocks] int32
    q_ranges_ptr,  # output: [N, 2] int32
    k_ranges_ptr,  # output: [N, 2] int32
    num_q_blocks,
    num_topk,
    num_k_blocks,
    block_m,
    block_n,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load TopK Indices (Local K Block IDs)
    # Shape is [H, Q, K], flattened.
    local_k_ids = tl.load(topk_indices_ptr + offsets, mask=mask)

    # Calculate Global Q Ranges
    # Logic: Since input is [H, Q, K], dividing linear index by K (num_topk)
    # gives us the global Q block index (0, 0, ..., 1, 1, ..., Q, Q, ...).
    # This naturally handles the Head dimension (monotonically increasing).
    global_q_id = offsets // num_topk

    q_start = global_q_id * block_m
    q_end = q_start + block_m

    # Calculate Global K Ranges
    # Logic: We need to determine which Head we are in to apply the Head Offset.
    # Elements per head = num_q_blocks * num_topk
    elems_per_head = num_q_blocks * num_topk
    head_id = offsets // elems_per_head

    # Global K ID = (Head_ID * K_Stride) + Local_K_ID
    global_k_id = (head_id * num_k_blocks) + local_k_ids

    k_start = global_k_id * block_n
    k_end = k_start + block_n

    # Store Results
    # Output shape is [N, 2]. In flattened memory, this is 2*N elements.
    # Element 'i' maps to output indices '2*i' (start) and '2*i + 1' (end).
    out_offsets = offsets * 2

    # Store Q Ranges
    tl.store(q_ranges_ptr + out_offsets, q_start, mask=mask)
    tl.store(q_ranges_ptr + out_offsets + 1, q_end, mask=mask)

    # Store K Ranges
    tl.store(k_ranges_ptr + out_offsets, k_start, mask=mask)
    tl.store(k_ranges_ptr + out_offsets + 1, k_end, mask=mask)


# ================ Utils for Block Sparse Attention ================


def generate_block_sparse_pattern(
    num_q_heads: int,
    num_kv_heads: int,
    num_q_blocks: int,
    num_kv_blocks: int,
    sparsity: float,
    mode: str = "per_kv_head",
    sparse_format: str = "block_mask",
    device: str = "cuda",
    scores: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a Block Sparse Pattern, supporting return of Dense Block Mask or TopK Indices.

    Args:
        num_q_heads (int): Number of Q heads.
        num_kv_heads (int): Number of KV heads.
        num_q_blocks (int): Number of Q blocks.
        num_kv_blocks (int): Number of K blocks.
        sparsity (float): Sparsity level (determines the k value for topk).
        mode (str):
            - "per_q_head": Independent pattern per Q Head (only applies to 'block_mask' format or if Q=KV).
            - "per_kv_head": Shared pattern per KV Head group (GQA).
            - "shared": Shared pattern across all Heads.
        sparse_format (str):
            - "block_mask": Returns a boolean mask of shape [1, num_kv_heads, Q_blk, K_blk].
            - "topk": Returns an index tensor of shape [num_kv_heads, Q_blk, k].
        device (str): Device to place tensors on.

    Returns:
        output (torch.Tensor): Shape depends on sparse_format.
        scores (torch.Tensor): Original random scores, shape [num_mask_heads, Q_blk, K_blk].
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
    elif mode == "shared":
        num_mask_heads = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if scores is None:
        scores = torch.rand(num_mask_heads, num_q_blocks, num_kv_blocks, device=device)
    else:
        assert scores.shape == (
            num_mask_heads,
            num_q_blocks,
            num_kv_blocks,
        ), f"Provided scores shape {scores.shape} does not match expected {(num_mask_heads, num_q_blocks, num_kv_blocks)}"

    _, topk_indices = torch.topk(scores, k, dim=-1)

    if sparse_format == "topk":
        if mode == "per_q_head" and mode == "shared":
            raise ValueError("Not Supported")

        # if mode == "shared":
        # topk_indices = topk_indices.expand(num_kv_heads, -1, -1)
        topk_indices = topk_indices.contiguous().to(dtype=torch.int32)

        return topk_indices, scores

    elif sparse_format == "block_mask":
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(2, topk_indices, True)

        """
        if mode == "per_kv_head" and num_q_heads != num_kv_heads:
            num_groups = num_q_heads // num_kv_heads
            # Repeat the mask for each Q head in the group
            mask = torch.repeat_interleave(
                mask, repeats=num_groups, dim=0
            )
        """

        # Add Batch dimension
        block_mask = mask.unsqueeze(0)

        return block_mask, scores

    else:
        raise ValueError(f"Unknown sparse_format: {sparse_format}")


def flatten_block_mask_to_kv_shape(mask_kv_4d: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask_kv_4d: shape [1, num_kv_heads, num_q_blocks, num_k_blocks]

    Returns:
        mask_flat: shape [num_kv_heads * num_q_blocks, num_kv_heads * num_k_blocks]
    """
    b, h_kv, num_q, num_k = mask_kv_4d.shape
    if b != 1:
        raise ValueError("Batch size must be 1")

    num_q_flat = h_kv * num_q
    num_k_flat = h_kv * num_k

    _, h_indices, q_indices, k_indices = torch.nonzero(mask_kv_4d, as_tuple=True)

    q_indices_flat = q_indices + h_indices * num_q

    k_indices_flat = k_indices + h_indices * num_k

    mask_flat = torch.zeros(
        num_q_flat, num_k_flat, dtype=torch.bool, device=mask_kv_4d.device
    )
    mask_flat[q_indices_flat, k_indices_flat] = True

    return mask_flat


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
    # if h_q != num_q_heads:
    #    raise ValueError(
    #        "Mask dimension mismatch: mask_4d.shape[1] should equal num_q_heads."
    #    )

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


def generate_ranges_from_topk_indices_triton(
    topk_indices: torch.Tensor, block_m: int, block_n: int, num_k_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates SORTED query and key range tensors.

    Args:
        topk_indices: Shape [kv_heads, num_q_blocks, topk_k_blocks].
        block_m: Size of query block.
        block_n: Size of key block.
        num_k_blocks: Total K blocks per head (stride), used for calculating global K offsets.

    Returns:
        q_range_tensor: Sorted global Q ranges.
        k_range_tensor: Corresponding global K ranges.
    """
    num_kv_heads, num_q_blocks, num_topk = topk_indices.shape
    n_elements = topk_indices.numel()

    dtype = torch.int32
    q_range = torch.empty((n_elements, 2), device=topk_indices.device, dtype=dtype)
    k_range = torch.empty((n_elements, 2), device=topk_indices.device, dtype=dtype)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    topk_indices_to_qk_ranges_kernel[grid](
        topk_indices_ptr=topk_indices,
        q_ranges_ptr=q_range,
        k_ranges_ptr=k_range,
        num_q_blocks=num_q_blocks,
        num_topk=num_topk,
        num_k_blocks=num_k_blocks,
        block_m=block_m,
        block_n=block_n,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return q_range, k_range


def generate_ranges_from_topk_indices(
    topk_indices: torch.Tensor, block_m: int, block_n: int, num_k_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates SORTED query and key range tensors.

    Args:
        topk_indices: Shape [kv_heads, num_q_blocks, topk_k_blocks].
        block_m: Size of query block.
        block_n: Size of key block.
        num_k_blocks: Total K blocks per head (stride), used for calculating global K offsets.

    Returns:
        q_range_tensor: Sorted global Q ranges.
        k_range_tensor: Corresponding global K ranges.
    """
    # Original shape: [num_kv_heads, num_q_blocks, num_k]
    num_kv_heads, num_q_blocks, num_k = topk_indices.shape
    device = topk_indices.device

    # ================= 1. Generate Sorted Global Q Ranges =================
    # Generate Head IDs: [H, 1, 1] -> [0, 1, ..., H-1]
    head_ids = torch.arange(num_kv_heads, device=device, dtype=topk_indices.dtype).view(
        -1, 1, 1
    )

    # This naturally maps to: (Head0, Q0), (Head0, Q1)... (Head1, Q0)...
    global_q_ids = torch.arange(
        num_kv_heads * num_q_blocks, dtype=topk_indices.dtype, device=device
    )

    # Expand for the K dimension: Each Q block has 'num_k' entries.
    # [0, 1] -> [0, 0, ..., 1, 1, ...]
    global_q_ids_flat = global_q_ids.repeat_interleave(num_k)

    q_start = global_q_ids_flat * block_m
    q_end = q_start + block_m
    q_range_tensor = torch.stack([q_start, q_end], dim=-1)

    # ================= 2. Generate Corresponding Sorted Global K Ranges =================
    # K indices are already arranged as [H, Q, K] (due to the permutation at the start)

    # Calculate Head Base Offset: [H, 1, 1]
    head_offset_k = head_ids * num_k_blocks

    # Global K ID = Head_Offset + Local_K_ID
    # [H, 1, 1] + [H, Q, K] -> [H, Q, K]
    global_k_block_ids = head_offset_k + topk_indices

    # Flatten
    global_k_ids_flat = global_k_block_ids.reshape(-1)

    k_start = global_k_ids_flat * block_n
    k_end = k_start + block_n
    k_range_tensor = torch.stack([k_start, k_end], dim=-1)

    return q_range_tensor, k_range_tensor


def generate_ranges_from_block_mask_triton(
    block_mask: torch.Tensor, block_m: int, block_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates query and key range tensor from a boolean block mask.

    Args:
        block_mask (torch.Tensor): A boolean tensor of shape [bsz, kv_heads, num_q_blocks, num_k_blocks].
        block_m (int): The size of each query block.
        block_n (int): The size of each key block.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - q_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the query ranges.
            - k_range_tensor (torch.Tensor): Tensor of shape [num_true_blocks, 2] listing the key ranges.
    """
    assert block_mask.dim() == 4, "block_mask must be 4D [B, H, Q, K]"
    b, h, num_q, num_k = block_mask.shape

    # 1. Find all True indices in the block mask
    indices = torch.nonzero(block_mask)  # Shape: [N, 4] -> (batch, h, q, k)

    n_elements = indices.shape[0]
    if n_elements == 0:
        return (
            torch.empty((0, 2), dtype=torch.int32, device=block_mask.device),
            torch.empty((0, 2), dtype=torch.int32, device=block_mask.device),
        )

    q_ranges = torch.empty((n_elements, 2), dtype=torch.int32, device=block_mask.device)
    k_ranges = torch.empty((n_elements, 2), dtype=torch.int32, device=block_mask.device)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    block_mask_to_qk_ranges_kernel[grid](
        indices_ptr=indices,
        q_ranges_ptr=q_ranges,
        k_ranges_ptr=k_ranges,
        stride_idx_0=indices.stride(0),
        stride_idx_1=indices.stride(1),
        stride_out_0=q_ranges.stride(0),
        stride_out_1=q_ranges.stride(1),
        num_q_blocks=num_q,
        num_k_blocks=num_k,
        block_m=block_m,
        block_n=block_n,
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    return q_ranges, k_ranges


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


def get_sdpa_mask_from_topk_indices(
    topk_indices: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    num_q_heads: int,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Generates a full boolean mask suitable for SDPA (Scaled Dot Product Attention)
    based on Top-K block indices.

    Args:
        topk_indices (torch.Tensor): Shape [num_kv_heads, num_q_blocks, num_topk].
                                     Stores the allowed K-block indices for each Q-block.
        seqlen_q (int): Total length of the Query sequence.
        seqlen_k (int): Total length of the Key sequence.
        block_size_q (int): Block size for Query.
        block_size_k (int): Block size for Key.
        num_q_heads (int): Total number of Query heads.
        batch_size (int): Batch size.

    Returns:
        torch.Tensor: A boolean mask of shape [B, H_q, S_q, S_k] (True indicates allowed/unmasked).
    """
    num_kv_heads, num_q_blocks, num_topk = topk_indices.shape
    num_k_blocks = (seqlen_k + block_size_k - 1) // block_size_k
    device = topk_indices.device

    # 1. Construct Block-level mask: [num_kv_heads, num_q_blocks, num_k_blocks]
    # Initialize with all False
    block_mask = torch.zeros(
        (num_kv_heads, num_q_blocks, num_k_blocks), dtype=torch.bool, device=device
    )

    # Prepare indices to fill block_mask using advanced indexing
    # q_idx: [num_kv_heads, num_q_blocks, num_topk] -> corresponds to dim=1 of block_mask
    # (The view is [1, num_q_blocks, 1], broadcasting across heads and topk)
    q_idx = (
        torch.arange(num_q_blocks, device=device).view(1, -1, 1).expand_as(topk_indices)
    )

    # h_idx: [num_kv_heads, num_q_blocks, num_topk] -> corresponds to dim=0 of block_mask
    # (The view is [num_kv_heads, 1, 1], broadcasting across blocks and topk)
    h_idx = (
        torch.arange(num_kv_heads, device=device).view(-1, 1, 1).expand_as(topk_indices)
    )

    # Set the positions corresponding to topk indices to True
    # topk_indices corresponds to dim=2 of block_mask
    block_mask[h_idx, q_idx, topk_indices] = True

    # 2. Handle Head Grouping (GQA/MQA)
    # If num_q_heads > num_kv_heads, we need to repeat the mask for the shared KV heads
    num_groups = num_q_heads // num_kv_heads
    if num_groups > 1:
        block_mask = torch.repeat_interleave(block_mask, repeats=num_groups, dim=0)

    # 3. Expand to Element-level Mask: [B, H, S_q, S_k] (Intermediate steps)
    # Upsample along the block dimensions using repeat_interleave

    # Upsample Q dimension
    sdpa_mask = torch.repeat_interleave(block_mask, repeats=block_size_q, dim=1)
    # Upsample K dimension
    sdpa_mask = torch.repeat_interleave(sdpa_mask, repeats=block_size_k, dim=2)

    # 4. Crop to actual sequence lengths
    # (Handles cases where block_size does not perfectly divide seqlen)
    sdpa_mask = sdpa_mask[:, :seqlen_q, :seqlen_k]

    # 5. Add Batch Dimension
    # [H, S_q, S_k] -> [B, H, S_q, S_k]
    sdpa_mask = sdpa_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return sdpa_mask


def get_sdpa_mask_from_block_sparse_mask(
    block_mask: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    num_q_heads: int,
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
    num_kv_heads = block_mask.shape[1]
    num_groups = num_q_heads // num_kv_heads
    # Repeat the mask for each Q head in the group
    block_mask = torch.repeat_interleave(block_mask, repeats=num_groups, dim=1)
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


# TODO: add faster mask creation and refactor
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
        """
        base_block_row_sz = random_partition_with_min_size(
            seqlen_q, num_q_blocks, min_q_block_size, num_kv_heads, device
        )
        final_block_row_sz = torch.repeat_interleave(
            base_block_row_sz, num_q_heads // num_kv_heads, dim=0
        )
        """
        final_block_row_sz = random_partition_with_min_size(
            seqlen_q, num_q_blocks, min_q_block_size, num_kv_heads, device
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
    """
    if mode == "per_kv_head" and num_q_heads != num_kv_heads:
        final_mask = torch.repeat_interleave(
            base_mask, num_q_heads // num_kv_heads, dim=0
        )
    else:
        final_mask = base_mask
    """
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
    # num_groups = num_q_heads // num_kv_heads
    num_groups = 1  # for now, we donot care num_q_heads
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
    # zeros_q = torch.zeros((num_q_heads, 1), dtype=block_row_sz.dtype, device=device)
    zeros_q = torch.zeros((num_kv_heads, 1), dtype=block_row_sz.dtype, device=device)
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
    num_q_heads: int,
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
    num_kv_heads = block_mask.shape[1]
    num_groups = num_q_heads // num_kv_heads
    # Repeat the mask for each Q head in the group
    # since we get full head mask for sdpa mask.
    block_mask = torch.repeat_interleave(block_mask, repeats=num_groups, dim=1)
    block_row_sz = torch.repeat_interleave(block_row_sz, repeats=num_groups, dim=0)

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

# TODO: automatically get params for flex_flash_func with ref_block_size input
def autotune_params_from_ref_block(
    ref_block_size: tuple[int, int]
):
    """
    Autotune and return the best configuration for flex_flash_func based on the given reference block size.
    return tile_size, swapab, packgqa, sparse_load parameters.
    """
    pass

# FIXME: remove this function when autotuning from ref_block_size is ready
def choose_ref_block(
    block_size: tuple[int, int], swap_ab: bool = False, sparse_load: bool = False
) -> tuple[int, int]:
    """
    Choose the proper reference block size for different Q/K block sizes, currently for uniform block mask.

    Rules:
    - ref_q_block_size must be a multiple of 64 and >= q_block_size
    - ref_k_block_size must be a multiple of 16 and >= k_block_size
    """
    q_block_size, k_block_size = block_size
    if swap_ab:
        ref_k_block_size = 64
        if q_block_size in (8, 16, 32, 64):
            ref_q_block_size = q_block_size
        else:
            raise NotImplementedError(
                "SwapAB Attention q_block_size must in (8, 16, 32, 64)."
            )
    elif sparse_load:
        # TODO: support arbitrary Q block size
        return (128, 128)  # only support (128, 128) block size currently
    else:
        if pack_gqa and q_block_size < 64:
            assert qhead_per_khead is not None
            q_block_size = q_block_size * qhead_per_khead
        # Tile_M must be a multiple of 64
        if q_block_size >= 64:
            ref_q_block_size = min(128, ((q_block_size + 63) // 64) * 64)
        else:
            ref_q_block_size = 64

        # Tile_K must be a multiple of 16
        if k_block_size >= 16:
            ref_k_block_size = min(256, ((k_block_size + 15) // 16) * 16)
        else:
            ref_k_block_size = 16

    return (ref_q_block_size, ref_k_block_size)
