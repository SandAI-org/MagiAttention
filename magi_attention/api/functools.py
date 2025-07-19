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
    Get the size need to pad (for better performance).

    Args:
        total_seqlen_q (int): seqlen of q.
        cp_size (int): The size of cp group.
        head_dim (int): head dim for q k v.
        chunk_size (int): chunk size to chunk the permutable tensor

    Returns:
        int: tokens need to pad.
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
