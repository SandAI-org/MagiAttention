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

"""
Sparse attention optimization arguments for Flexible Flash Attention.

This module provides a dataclass for configuring sparse attention optimizations,
including swap_ab, pack_gqa, sparse_load, and other related parameters.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FfaSparseArgs:
    """Configuration arguments for sparse attention optimization in Flexible Flash Attention.

    This dataclass encapsulates all parameters related to sparse attention optimization,
    making it easier to manage and pass these parameters as a group.

    The FFA internal tile size is automatically selected by the kernel based on the
    ref_block_size parameter in flex_flash_attn_func, or can be manually specified
    through the choose_ref_block utility.

    Attributes:
        swap_ab: Whether to use swap_ab mode for optimizing performance when q_range size
            is small (<= 16). Defaults to False.
        pack_gqa: Whether to group query heads sharing the same KV head into a single
            computation block tile for small seqlen_q scenarios. This method significantly
            improves the computational efficiency of block sparse attention when seqlen_q
            is small. Defaults to False.
        sparse_load: Whether to enable sparse load mode for optimizing performance when
            k_range size is small (< 64). Must be used together with auto_range_merge=True
            for enhanced performance. Defaults to False.
        auto_range_merge: Whether to automatically merge k_ranges for the same q_range.
            This flag is useful for sparse attention scenarios. Defaults to False.
        disable_fwd_atomic_reduction: Whether to disable forward atomic reduction.
            If you can ensure q_ranges is non-overlapped, you can set this to True for
            better performance. Defaults to False.
        max_seqlen_q: Maximum sequence length for query. If provided, enables optimization
            for tile_scheduler. Most recommended to set this when using auto_range_merge
            (for block sparse attention). Defaults to None.

    Example:
        >>> # Create with default values
        >>> sparse_args = FfaSparseArgs()
        >>>
        >>> # Create with specific values
        >>> sparse_args = FfaSparseArgs(
        ...     auto_range_merge=True,
        ...     sparse_load=True,
        ...     max_seqlen_q=4096
        ... )
        >>>
        >>> # Auto-tune from ref_block_size
        >>> sparse_args = FfaSparseArgs.from_ref_block_size(
        ...     ref_block_size=(64, 32),
        ...     qhead_per_khead=4
        ... )
        >>>
        >>> # Use with flex_flash_attn_func
        >>> out, lse = flex_flash_attn_func(
        ...     q, k, v, q_ranges, k_ranges,
        ...     sparse_args=sparse_args
        ... )
    """

    ffa_tile_size: tuple[int, int] | None = (128, 128)
    swap_ab: bool = False
    pack_gqa: bool = False
    sparse_load: bool = False
    auto_range_merge: bool = False
    max_seqlen_q: int | None = None

    @classmethod
    def from_ref_block_size(
        cls,
        ref_block_size: tuple[int, int],
        qhead_per_khead: int,
    ) -> "FfaSparseArgs":
        """Auto-tune and create FfaSparseArgs from ref_block_size and qhead_per_khead.

        This method uses the choose_ref_block utility to automatically determine
        the best FFA internal parameters based on the given ref_block_size.

        The auto-tuning rules (from sparse_utils.choose_ref_block):
        - SwapAB and sparse load can't be enabled together
        - Prioritize sparse load and packGQA in small Q/K blocks
        - For k_block_size < 64:
            - sparse_load = True, internal tile size K = 128
        - For k_block_size >= 64:
            - internal tile size K is a multiple of 16, capped at 128
        - For q_block_size < 128:
            - pack_gqa = True if qhead_per_khead > 1
            - If q_block_size * qhead_per_khead <= 16: swap_ab = True
            - Otherwise: internal tile size M is a multiple of 64, capped at 128
        - For q_block_size >= 128:
            - pack_gqa = False, swap_ab = False
            - internal tile size M is a multiple of 64, capped at 128

        Args:
            ref_block_size: A tuple of (q_block_size, k_block_size) representing the
                most common (mode) user's Q/K block sizes.
            qhead_per_khead: The number of query heads per key head (GQA group size).

        Returns:
            FfaSparseArgs: An instance with auto-tuned parameters (excluding ffa_tile_size
                which is handled internally by the kernel).

        Example:
            >>> sparse_args = FfaSparseArgs.from_ref_block_size(
            ...     ref_block_size=(64, 32),
            ...     qhead_per_khead=4
            ... )
        """
        from magi_attention.utils.sparse_utils import choose_ref_block

        params = choose_ref_block(ref_block_size, qhead_per_khead)
        return cls(
            ffa_tile_size=params["ref_block_size"],
            swap_ab=params["swap_ab"],
            pack_gqa=params["pack_gqa"],
            sparse_load=params["sparse_load"],
            max_seqlen_q=ref_block_size[0],  # default to ref q_block_size
            auto_range_merge=True,  # default to True for sparse attention
        )

    def validate_tile_size(self) -> None:
        if self.ffa_tile_size is None:
            raise RuntimeError(
                "If manually setting FfaSparseArgs, ffa_tile_size must be specified."
            )
        kblock_m, kblock_n = self.ffa_tile_size
        if self.swap_ab:
            assert self.ffa_tile_size in (
                (8, 64),
                (16, 64),
                (32, 64),
                (64, 64),
            ), "ref_block_size must be (8, 64), (16, 64), (32, 64) or (64, 64) when swap_ab == True"
        elif self.sparse_load:
            assert (
                kblock_n == 128
            ), "sparse load requires kblock_n == 128 in ffa_tile_size"
        else:
            # TODO: K>128 support?
            assert kblock_m in (
                64,
                128,
                192,
            ), "ref_block_size: (kblock_m, kblock_n), kblock_m must be 64, 128 or 192 when swapab == False"
            assert (
                kblock_n % 16 == 0 and kblock_n <= 128
            ), "ref_block_size: (kblock_m, kblock_n), kblock_n <= 128 and kblock_n % 16 == 0 must be True"

    def validate(self) -> None:
        """Validate the sparse arguments configuration.

        Raises:
            RuntimeError: If the configuration is invalid.
        """
        self.validate_tile_size()
        if self.sparse_load and self.swap_ab:
            # FIXME: support both optimizations together
            raise RuntimeError("swap_ab and sparse_load cannot be enabled together.")
        if self.sparse_load and not self.auto_range_merge:
            raise RuntimeError(
                "When using sparse load, range merge must be enabled "
                "(set auto_range_merge to True)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert the sparse arguments to a dictionary.

        Returns:
            A dictionary containing all the sparse arguments.
        """
        return {
            "ffa_tile_size": self.ffa_tile_size,
            "swap_ab": self.swap_ab,
            "pack_gqa": self.pack_gqa,
            "sparse_load": self.sparse_load,
            "auto_range_merge": self.auto_range_merge,
            "max_seqlen_q": self.max_seqlen_q,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FfaSparseArgs":
        """Create FfaSparseArgs from a dictionary.

        Args:
            d: A dictionary containing sparse arguments.

        Returns:
            FfaSparseArgs: An instance created from the dictionary.
        """
        return cls(
            ffa_tile_size=d.get("ffa_tile_size", (128, 128)),
            swap_ab=d.get("swap_ab", False),
            pack_gqa=d.get("pack_gqa", False),
            sparse_load=d.get("sparse_load", False),
            auto_range_merge=d.get("auto_range_merge", False),
            max_seqlen_q=d.get("max_seqlen_q"),
        )


# Default instance with all optimizations disabled
DEFAULT_SPARSE_ARGS = FfaSparseArgs()
