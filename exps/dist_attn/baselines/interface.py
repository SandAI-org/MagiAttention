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

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List

import torch

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


class AttnBackend(Enum):
    TE = ("te",)
    FA3 = "fa3"


class AttnBaselineInterface(ABC):
    def __init__(
        self,
        cp_process_group: Dict,
        qkv_format: str,
        backend: AttnBackend,
    ):
        """
        Initialize the attention baseline interface with context parallel (CP) communication settings,
        QKV format description, and backend implementation details.

        Args:
            cp_process_group (Dict): A dictionary containing process group information used for context parallel
                communication. Keys typically include 'group', 'rank', and 'world_size'.
            qkv_format (str): The format specification of the QKV tensors, e.g., "qkv_interleaved" or "separate".
            backend (AttnBackend): Backend implementation identifier (e.g., FlashAttention, Torch-based kernel).
                Used to select the appropriate attention kernel and communication strategy.
        """
        pass

    @abstractmethod
    def dispatch(
        self,
        x_global: torch.Tensor,
        ranges: AttnRanges,
        valid_total_seqlen: int,  # required by AttnRanges.to_cu_seqlens
        name: str | List[str],  # key names for shard_meta
    ):
        """
        Dispatch the global tensor `x_global` along its sequence dimension according to `cu_seqlens` and meta information,
        and return the dispatched local tensor `x_local` that is shard-aligned with current cp rank.

        Args:
            x_global (torch.Tensor): The global input tensor with shape [total_seqlen, ...].
            cu_seqlens (torch.Tensor): Cumulative sequence lengths (CUDA tensor) describing per-sample offsets.
            host_cu_seqlens (List[int]): Host-side copy of `cu_seqlens` used for metadata construction and validation.
            name (str): Unique keys used to identify and store shard metadata.

        Returns:
            torch.Tensor: The dispatched local tensor with shape [local_seqlen, ...], specific to the current cp rank.
        """

    @abstractmethod
    def undispatch(
        self,
        x_local: torch.Tensor,
        name: str,  # key name for shard_meta
    ) -> torch.Tensor:
        """
        Reconstruct the global tensor `x_global` from local shard `x_local` using saved meta information under `name`.

        Args:
            x_local (torch.Tensor): The local tensor with shape [local_seqlen, ...] to be gathered from cp ranks.
            name (str): The key used to retrieve the corresponding shard metadata for reconstruction.

        Returns:
            torch.Tensor: The reconstructed global tensor with shape [total_seqlen, ...].
        """

    @abstractmethod
    def apply_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_type: AttnMaskType,
        dropout_p: float,
        softmax_scale: float,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the attention mechanism on inputs `q`, `k`, and `v`, with optional masking, dropout, and scaling.

        Args:
            q (torch.Tensor): Query tensor of shape [total_seqlen_q, num_heads, head_dim].
            k (torch.Tensor): Key tensor of shape [total_seqlen_k, num_heads, head_dim].
            v (torch.Tensor): Value tensor of shape [total_seqlen_k, num_heads, head_dim].
            attn_mask_type (AttnMaskType | list[AttnMaskType]): Attention mask type(s) per sample or batch-wide.
            dropout_p (float): Dropout probability applied to attention weights (0.0 to 1.0).
            softmax_scale (float): Scale applied before softmax; typically 1 / sqrt(head_dim).
            deterministic (bool): If True, disables dropout and enforces deterministic computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape [total_seqlen_q, num_heads, head_dim].
                - Log-sum-exp tensor for softmax, shape [batch_size, num_heads, max_seqlen_q].
        """
