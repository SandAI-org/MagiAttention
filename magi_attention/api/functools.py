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

from magi_attention.common.mask import AttnMask


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


def compute_pad_size(total_seqlen_q, cp_size, head_dim):
    """
    Get the size need to pad(for better performance).
    args:
        total_seqlen_q: seqlen of q.
        cp_size: The size of cp group.
        head_dim: head dim for q k v.

    returns:
        tokens_to_pad: tokens need to pad.
        q_block_size: block size.
    """
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 8")
    if head_dim > 192:
        raise ValueError(f"head_dim ({head_dim}) must be ≤ 192")

    # for size the chunk_size is fixed as 1536
    chunk_size = 1536
    # Validate sequence length
    block_requirement = chunk_size * cp_size
    tokens_to_pad = 0
    if (remainder := total_seqlen_q % block_requirement) != 0:
        tokens_to_pad = block_requirement - remainder

    return tokens_to_pad, chunk_size


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
