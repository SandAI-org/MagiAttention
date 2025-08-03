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

from collections import OrderedDict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

import magi_attention
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.mask import AttnMask
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges


class FixedLenDict(OrderedDict):
    """A fixed-length dictionary that evicts the least recently used item (LRU policy) when capacity is exceeded"""

    def __init__(self, max_size: int, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        # If key exists, delete it first (to ensure it moves to end)
        if key in self:
            del self[key]
        # If at max capacity, remove the oldest item
        elif len(self) >= self.max_size:
            self.popitem(last=False)
        # Insert new key-value pair (automatically added to end)
        super().__setitem__(key, value)

    def get(self, key, default=None):
        # Override get method to move accessed items to end (marking as recently used)
        if key in self:
            value = super().__getitem__(key)
            del self[key]
            super().__setitem__(key, value)
            return value
        return default

    def get_most_recent_key(self):
        """
        Gets and returns the most recently added or accessed key.
        If the dictionary is empty, returns None.
        """
        if not self:
            return None

        return next(reversed(self.keys()))


def compute_pad_size(
    total_seqlen_q: int,
    cp_size: int,
    chunk_size: int,
) -> int:
    """
    Compute the size to pad to the input tensor along the seqlen dim at last.

    Args:
        total_seqlen_q (int): seqlen of q.
        cp_size (int): The size of cp group.
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

    Returns:
        int: the number of tokens to pad.
    """

    # Validate sequence length
    block_requirement = chunk_size * cp_size
    tokens_to_pad = 0
    if (remainder := total_seqlen_q % block_requirement) != 0:
        tokens_to_pad = block_requirement - remainder

    return tokens_to_pad


def squash_batch_dim(x):
    x_merged = rearrange(x, "b s ... -> (b s) ...")
    return x_merged


def full_attention_to_varlen_attention(batch_size, seq_len):
    cu_seqlens_q = torch.arange(0, batch_size + 1) * seq_len
    cu_seqlens_k = cu_seqlens_q

    return cu_seqlens_q, cu_seqlens_k


def pad_at_dim(x, dim, pad_size, value=0, side="right"):
    pad = [0] * (2 * x.dim())
    pad_idx = -(dim + 1) * 2 + (0 if side == "left" else 1)
    pad[pad_idx] = pad_size
    return F.pad(x, pad=tuple(pad), mode="constant", value=value)


def unpad_at_dim(x, dim, pad_size):
    seq_len = x.size(dim)
    unpad_x = x.narrow(dim=0, start=0, length=seq_len - pad_size)
    return unpad_x


def from_mask(
    mask: list[list[int]] | torch.Tensor,
) -> "AttnMask":
    """
    The (less common) factory method to construct a AttnMask instance,
    with a 2d int32 mask tensor, where the nonzero cell indicates unmasked position,
    while the zero cell indicates masked position

    Args:
        mask (list[list[int]] | torch.Tensor): the 2d int32 mask tensor

    Returns:
        AttnMask: the attn mask instance
    """

    return AttnMask.from_mask(
        mask=mask,
    )


def apply_padding(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen: int,
    pad_size: int,
) -> tuple[AttnRanges, AttnRanges, list[AttnMaskType]]:
    """
    Appends padding to the attention ranges and updates the corresponding mask type.

    This function adds a padding query range at the end of `q_ranges`, a dummy key
    range to `k_ranges`, and appends a `FULL` attention mask type to maintain alignment.
    It is typically used when padding is required for alignment or block-wise processing.

    Args:
        q_ranges (AttnRanges): Query token ranges before padding.
        k_ranges (AttnRanges): Key token ranges before padding.
        attn_mask_type (list[AttnMaskType]): List of attention mask types corresponding to the ranges.
        total_seqlen (int): The total original sequence length (used to place the padding at the end).
        pad_size (int): The size of the padding to append.

    Returns:
        tuple[AttnRanges, AttnRanges, list[AttnMaskType]]:
            - Updated query ranges with padding added.
            - Updated key ranges with a dummy range for padding.
            - Updated attention mask type list with a FULL mask for the padding block.
    """
    q_range = AttnRanges.from_ranges(q_ranges.to_naive_ranges(), check=True)
    k_range = AttnRanges.from_ranges(k_ranges.to_naive_ranges(), check=True)
    attn_mask_types = [attn_mask_type[i] for i in range(len(attn_mask_type))]

    q_range.append(AttnRange(start=total_seqlen, end=total_seqlen + pad_size))
    k_range.append(AttnRange(start=0, end=0))
    attn_mask_types.append(AttnMaskType.FULL)

    return q_range, k_range, attn_mask_types


def init_hierarchical_mesh(
    world_size: int,
    world_size_inter_node: int,
    world_size_intra_node: int,
) -> DeviceMesh | None:
    """Generate device mesh for hierarchical comm

    Args:
        world_size (int): total world size for cp
        world_size_inter_node (int): inter-machine world size
        world_size_intra_node (int): in-machine world size

    Returns:
        Optional[DeviceMesh]: The device mesh object if using hierarchical
    """
    assert world_size == world_size_inter_node * world_size_intra_node, (
        f"world_size must be equal to inter_node * intra_node, "
        f"but got {world_size=}, {world_size_inter_node=} and {world_size_intra_node=}"
    )
    if magi_attention.comm.is_hierarchical_comm_enable():
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )
    else:
        device_mesh = None

    return device_mesh


def infer_attn_mask_from_sliding_window(
    q_range: AttnRange,
    k_range: AttnRange,
    window_size: list[int],
) -> tuple[AttnRanges, AttnRanges, list[AttnMaskType]]:
    """Convert only one sliding window masks into representations using q_range, k_range, and mask type.
    The mask type is specified using window_size.

    Args:
        q_range (AttnRange): q_range of this sliding window mask
        k_range (AttnRange): k_range of this sliding window mask
        window_size (list[int]): window_size of sliding window mask

    Returns:
        tuple[AttnRanges, AttnRanges, list[AttnMaskType]]: processed (q_ranges, k_ranges, masktypes) triple,
            sliding window mask have been cutted into triple representation.

    Example:
        Here's an example of ``infer_attn_mask_from_sliding_window``::

            >>> q_ranges, k_ranges, attn_mask_type = infer_attn_mask_from_sliding_window(
            ...     q_range=AttnRange.from_range([5, 15]),
            ...     k_range=AttnRange.from_range([5, 15]),
            ...     window_size=(2, 3),
            ... )

        The code above represents the sliding window mask within the ``[5, 15] x [5, 15]`` region
        with a window size of (2, 3).
    """
    assert len(window_size) == 2, "window size must be of 2 int"
    assert window_size[0] < k_range.seqlen and window_size[1] < k_range.seqlen, (
        "the num of window_size must be -1 or < k_range.seqlen",
        f"but got {window_size=}",
    )

    q_ranges_, k_ranges_ = AttnRanges(), AttnRanges()
    attn_mask_type_: list[AttnMaskType] = []

    # remove the invalid parts in q_range
    q_range_global = q_range
    if q_range.seqlen > k_range.seqlen:
        q_range_global = AttnRange(
            start=q_range.end - k_range.seqlen,
            end=q_range.end,
        )

    # When window_size is -1 or k_range.seqlen - 1, we increment it to avoid splitting the full mask in the result.
    left_window_size = (
        window_size[0]
        if window_size[0] != -1 and window_size[0] != k_range.seqlen - 1
        else k_range.seqlen
    )
    right_window_size = (
        window_size[1]
        if window_size[1] != -1 and window_size[1] != k_range.seqlen - 1
        else k_range.seqlen
    )
    # The principle of the algorithm is to first expand the sliding window mask into a bi-causal one,
    # and then use the slicing algorithm in the slice maker to cut the bi-causal mask to get the sliced sliding window mask.
    # And precisely because we are only simulating the expansion of the bi-causal mask,
    # the left and right window_size can exceed k_range.seqlen - 1 at this point. Compute the expanded bi-causal k_range here.
    slice_k_range_start = k_range.end - q_range_global.seqlen - left_window_size
    slice_k_range_end = k_range.end + right_window_size

    # Compute the region of the k_range that actually needs to be calculated.
    k_range_global = AttnRange(
        start=max(k_range.start, slice_k_range_start),
        end=k_range.end,
    )

    # The following is the logic for slicing the bi-causal mask in the slice maker.
    # First, define the variables needed for the slicing process.
    causal_start = slice_k_range_end - q_range_global.seqlen
    diff_len_of_k_range_minus_q_range = max(
        0,
        slice_k_range_end - slice_k_range_start - q_range_global.seqlen,
    )

    # calculate k_range exceed slice_start, the maxValue not exceed slice_q_range.seqlen
    range_start_exceed_slice_start = min(
        k_range_global.start - slice_k_range_start,
        q_range_global.seqlen,
    )
    range_end_exceed_slice_start = min(
        k_range_global.end - slice_k_range_start,
        q_range_global.seqlen,
    )

    # calculate k_range exceed causal start, the minValue not less than 0
    range_end_exceed_causal_start = max(0, k_range_global.end - causal_start)
    range_start_exceed_causal_start = max(0, k_range_global.start - causal_start)

    # Draw vertical lines from the two endpoints of k_range,
    # which intersect the two hypotenuses of the bi-causal mask at two points.
    # Calculate the vertical coordinates (heights) of these two intersection points,
    # and determine which point is above the other by comparison.
    short_length = min(range_start_exceed_slice_start, range_end_exceed_causal_start)
    long_length = max(range_start_exceed_slice_start, range_end_exceed_causal_start)

    # (part1) calculate q_range and k_range of causal slice
    causal_q_range_local = AttnRange(
        start=q_range_global.start + range_start_exceed_causal_start,
        end=q_range_global.start + short_length,
    )
    causal_k_range_local = AttnRange(
        start=k_range_global.start,
        end=min(
            k_range_global.end,
            k_range_global.start + diff_len_of_k_range_minus_q_range,
        ),
    )

    # (part2) calculate q_range of full or bi_causal slice
    full_or_bi_causal_q_range_local = AttnRange(
        start=q_range_global.start + short_length,
        end=q_range_global.start + long_length,
    )

    # (part3) calculate q_range and k_range of inv_causal slice
    inv_causal_q_range_local = AttnRange(
        start=q_range_global.start + long_length,
        end=q_range_global.start + range_end_exceed_slice_start,
    )
    inv_causal_k_range_local = AttnRange(
        start=max(
            k_range_global.start,
            k_range_global.end - diff_len_of_k_range_minus_q_range,
        ),
        end=k_range_global.end,
    )

    # exclude invalid causal slice
    if causal_q_range_local.seqlen > 0:
        q_ranges_.append(causal_q_range_local)
        k_ranges_.append(causal_k_range_local)
        attn_mask_type_.append(AttnMaskType.CAUSAL)

    # exclude invalid full or bi_causal slice
    if full_or_bi_causal_q_range_local.seqlen > 0:
        q_ranges_.append(full_or_bi_causal_q_range_local)
        k_ranges_.append(k_range_global)
        attn_mask_type_.append(
            AttnMaskType.FULL
            if range_start_exceed_slice_start > range_end_exceed_causal_start
            else AttnMaskType.BICAUSAL
        )

    # exclude invalid inv_causal slice
    if inv_causal_q_range_local.seqlen > 0:
        q_ranges_.append(inv_causal_q_range_local)
        k_ranges_.append(inv_causal_k_range_local)
        attn_mask_type_.append(AttnMaskType.INVCAUSAL)

    return q_ranges_, k_ranges_, attn_mask_type_
