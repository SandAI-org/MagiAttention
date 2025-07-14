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
    head_dim: int,
    chunk_size: int,
) -> int:
    """
    Get the size need to pad(for better performance).

    Args:
        total_seqlen_q (int): seqlen of q.
        cp_size (int): The size of cp group.
        head_dim (int): head dim for q k v.
        chunk_size (int): chunk size to chunk the permutable tensor

    Returns:
        tokens_to_pad (int): tokens need to pad.
    """
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 8")
    if head_dim > 192:
        raise ValueError(f"head_dim ({head_dim}) must be ≤ 192")

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
        device_mesh (DeviceMesh | None): return device_mesh if use hierarchical comm,
            return None if not.
    """
    assert world_size == world_size_inter_node * world_size_intra_node, (
        f"world_size must be equal to inter_node * intra_node, "
        f"but get {world_size=}, {world_size_inter_node=} and {world_size_intra_node=}"
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


def infer_attn_mask_from_window_size(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    window_size_list: list[list[int]],
) -> tuple[AttnRanges, AttnRanges, list[AttnMaskType]]:
    """Convert full, causal, and sliding window masks into representations using q_ranges, k_ranges, and mask types.
    The mask type is specified using window_size, and multiple masks can be processed simultaneously.

    Args:
        q_ranges (AttnRanges): q_range of masks
        k_ranges (AttnRanges): k_range of masks
        window_size_list (list[list[int]]): masktype of each (q_range, k_range) area,
            the mask type is specified using window_size.

    Returns:
        tuple[AttnRanges, AttnRanges, list[AttnMaskType]]: processed (q_ranges, k_ranges, masktypes) triple,
            sliding window mask have been cutted into triple representation.
    """
    processed_q_ranges: AttnRanges = AttnRanges()
    processed_k_ranges: AttnRanges = AttnRanges()
    attn_mask_type: list[AttnMaskType] = []

    for q_range, k_range, window_size in zip(q_ranges, k_ranges, window_size_list):
        if window_size == [-1, -1]:
            processed_q_ranges.append(q_range)
            processed_k_ranges.append(k_range)
            attn_mask_type.append(AttnMaskType.FULL)
        elif window_size == [-1, 0]:
            processed_q_ranges.append(q_range)
            processed_k_ranges.append(k_range)
            attn_mask_type.append(AttnMaskType.CAUSAL)
        elif window_size == [0, -1]:
            processed_q_ranges.append(q_range)
            processed_k_ranges.append(k_range)
            attn_mask_type.append(AttnMaskType.INVCAUSAL)
        else:
            # sliding window
            (
                sw_q_ranges,
                sw_k_ranges,
                sw_attn_mask_type,
            ) = infer_attn_mask_from_sliding_window(
                q_range=q_range,
                k_range=k_range,
                window_size=window_size,
            )
            processed_q_ranges.extend(sw_q_ranges)
            processed_k_ranges.extend(sw_k_ranges)
            attn_mask_type.extend(sw_attn_mask_type)

    return processed_q_ranges, processed_k_ranges, attn_mask_type


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
    """
    assert len(window_size) == 2, "window size must be of 2 int"
    assert window_size[0] < k_range.seqlen and window_size[1] < k_range.seqlen, (
        "the num of window_size must be -1 or < k_range.seqlen",
        f"but get {window_size=}",
    )

    q_ranges_, k_ranges_ = AttnRanges(), AttnRanges()
    attn_mask_type_: list[AttnMaskType] = []

    left_window_size = window_size[0] if window_size[0] != -1 else k_range.seqlen - 1
    right_window_size = window_size[1] if window_size[1] != -1 else k_range.seqlen - 1

    if left_window_size + right_window_size + 1 < k_range.seqlen:
        sliding_window_length = left_window_size = right_window_size + 1
        top_length = left_window_size + 1 if left_window_size > 0 else 0
        bottom_length = right_window_size + 1 if right_window_size > 0 else 0

        causal_q_range = AttnRange(
            start=q_range.start,
            end=q_range.start + top_length,
        )
        bi_causal_q_range = AttnRange(
            start=q_range.start + top_length,
            end=q_range.end - bottom_length,
        )
        inv_causal_q_range = AttnRange(
            start=q_range.end - bottom_length,
            end=q_range.end,
        )

        if causal_q_range.seqlen > 0:
            causal_k_range = AttnRange(
                start=k_range.start,
                end=k_range.start + sliding_window_length,
            )

            q_ranges_.append(causal_q_range)
            k_ranges_.append(causal_k_range)
            attn_mask_type_.append(AttnMaskType.CAUSAL)

        if bi_causal_q_range.seqlen > 0:
            q_ranges_.append(bi_causal_q_range)
            k_ranges_.append(k_range)
            attn_mask_type_.append(AttnMaskType.BICAUSAL)

        if inv_causal_q_range.seqlen > 0:
            inv_causal_k_range = AttnRange(
                start=k_range.end - sliding_window_length,
                end=k_range.end,
            )

            q_ranges_.append(inv_causal_q_range)
            k_ranges_.append(inv_causal_k_range)
            attn_mask_type_.append(AttnMaskType.INVCAUSAL)
    else:
        top_length = q_range.seqlen - right_window_size - 1
        bottom_length = q_range.seqlen - left_window_size - 1

        causal_q_range = AttnRange(
            start=q_range.start,
            end=q_range.start + top_length,
        )
        bi_causal_q_range = AttnRange(
            start=q_range.start + top_length,
            end=q_range.end - bottom_length,
        )
        inv_causal_q_range = AttnRange(
            start=q_range.end - bottom_length,
            end=q_range.end,
        )

        if causal_q_range.seqlen > 0:
            q_ranges_.append(causal_q_range)
            k_ranges_.append(k_range)
            attn_mask_type_.append(AttnMaskType.CAUSAL)

        if bi_causal_q_range.seqlen > 0:
            q_ranges_.append(bi_causal_q_range)
            k_ranges_.append(k_range)
            attn_mask_type_.append(AttnMaskType.FULL)

        if inv_causal_q_range.seqlen > 0:
            q_ranges_.append(inv_causal_q_range)
            k_ranges_.append(k_range)
            attn_mask_type_.append(AttnMaskType.INVCAUSAL)

    return q_ranges_, k_ranges_, attn_mask_type_
