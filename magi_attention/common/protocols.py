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
Protocol definitions for core data structures shared between
Python and C++ (pybind11) backends.

These protocols serve as the contract that both backends must satisfy.
They enable static type checkers (mypy/pyright) to verify interface
alignment without requiring explicit inheritance.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class AttnMaskTypeProtocol(Protocol):
    """Protocol for AttnMaskType enum."""

    @classmethod
    def from_int_type(cls, int_type: int) -> AttnMaskTypeProtocol:
        ...

    def to_int_type(self) -> int:
        ...


@runtime_checkable
class AttnRangeProtocol(Protocol):
    """Protocol for AttnRange — a half-open integer range [start, end)."""

    start: int
    end: int

    @property
    def seqlen(self) -> int:
        ...

    @staticmethod
    def from_range(attn_range: Any, check: bool = False) -> AttnRangeProtocol:
        ...

    def clone(self) -> AttnRangeProtocol:
        ...

    def offset(self, offset: int) -> AttnRangeProtocol:
        ...

    def truncate(
        self, start: int | None = None, end: int | None = None
    ) -> AttnRangeProtocol:
        ...

    def intersect(self, other: AttnRangeProtocol) -> AttnRangeProtocol:
        ...

    def intersect_size(self, other: AttnRangeProtocol) -> int:
        ...

    def union(self, other: AttnRangeProtocol) -> list[AttnRangeProtocol]:
        ...

    def union_size(self, other: AttnRangeProtocol) -> int:
        ...

    def diff_by(self, other: AttnRangeProtocol) -> list[AttnRangeProtocol]:
        ...

    def is_subrange_of(self, other: AttnRangeProtocol) -> bool:
        ...

    def is_overlap_with(self, other: AttnRangeProtocol) -> bool:
        ...

    def is_empty(self) -> bool:
        ...

    def is_valid_close(self, start: int | None = None, end: int | None = None) -> bool:
        ...

    def is_valid_open(self, start: int | None = None, end: int | None = None) -> bool:
        ...

    def check_valid(self, start: int | None = None, end: int | None = None) -> None:
        ...

    def to_naive_range(self) -> tuple:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


@runtime_checkable
class AttnRangesProtocol(Protocol):
    """Protocol for AttnRanges — a list of AttnRange objects."""

    @staticmethod
    def from_ranges(ranges: Any, check: bool = False) -> AttnRangesProtocol:
        ...

    @staticmethod
    def from_cu_seqlens(cu_seqlens: Sequence[int], seq_len: int) -> AttnRangesProtocol:
        ...

    def append(self, attn_range: AttnRangeProtocol, check: bool = False) -> None:
        ...

    def insert(
        self, idx: int, attn_range: AttnRangeProtocol, check: bool = False
    ) -> None:
        ...

    def extend(self, attn_ranges: AttnRangesProtocol, check: bool = False) -> None:
        ...

    def pop(self, idx: int = -1) -> AttnRangeProtocol:
        ...

    def clear_empty(self) -> AttnRangesProtocol:
        ...

    def clone(self) -> AttnRangesProtocol:
        ...

    def sort(self) -> AttnRangesProtocol:
        ...

    def merge(self) -> AttnRangesProtocol:
        ...

    def merge_with_split_alignment(
        self, split_alignment: int = 1
    ) -> AttnRangesProtocol:
        ...

    def chunk(self, chunk_size: int, check: bool = True) -> list[AttnRangesProtocol]:
        ...

    def truncate(
        self, start: int | None = None, end: int | None = None
    ) -> AttnRangesProtocol:
        ...

    def is_sorted(self) -> bool:
        ...

    def is_merged(self) -> bool:
        ...

    def is_non_overlap(self) -> bool:
        ...

    def is_cu_seqlens(self, seqlen: int) -> bool:
        ...

    def is_valid(self) -> bool:
        ...

    def check_valid(self) -> None:
        ...

    def is_empty(self) -> bool:
        ...

    def to_cu_seqlens(self, seq_len: int) -> list[int]:
        ...

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        ...

    def to_naive_ranges(self) -> Any:
        ...

    def make_range_local(
        self,
        other_attn_range: AttnRangeProtocol,
        is_self_merged: bool = False,
        prefix_offset: list[int] | None = None,
    ) -> tuple:
        ...

    def make_ranges_local(
        self,
        other_attn_ranges: AttnRangesProtocol,
        is_self_merged: bool = False,
    ) -> AttnRangesProtocol:
        ...

    def find_hole_ranges(
        self,
        other_attn_ranges: AttnRangesProtocol,
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> AttnRangesProtocol:
        ...

    def find_overlap_ranges(
        self,
        other_attn_ranges: AttnRangesProtocol,
        is_self_merged: bool = False,
        is_other_merged: bool = False,
    ) -> AttnRangesProtocol:
        ...

    def intersect_size(self) -> int:
        ...

    def intersect_size_with(self, other: AttnRangesProtocol) -> int:
        ...

    def union_size(self) -> int:
        ...

    def union_size_with(self, other: AttnRangesProtocol) -> int:
        ...

    @property
    def total_seqlen(self) -> int:
        ...

    @property
    def max_seqlen(self) -> int:
        ...

    @property
    def start(self) -> int:
        ...

    @property
    def end(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def points(self) -> list[int]:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: Any) -> Any:
        ...

    def __setitem__(self, idx: Any, value: Any) -> None:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


@runtime_checkable
class AttnRectangleProtocol(Protocol):
    """Protocol for AttnRectangle — a (q, k, d) rectangle with mask type."""

    @property
    def q_range(self) -> AttnRangeProtocol:
        ...

    @property
    def k_range(self) -> AttnRangeProtocol:
        ...

    @property
    def d_range(self) -> AttnRangeProtocol:
        ...

    def is_valid(
        self,
        q_range: AttnRangeProtocol | None = None,
        k_range: AttnRangeProtocol | None = None,
        d_range: AttnRangeProtocol | None = None,
    ) -> bool:
        ...

    def check_valid(
        self,
        q_range: AttnRangeProtocol | None = None,
        k_range: AttnRangeProtocol | None = None,
        d_range: AttnRangeProtocol | None = None,
    ) -> None:
        ...

    def get_valid_or_none(self) -> AttnRectangleProtocol | None:
        ...

    def shrink_q_range(self) -> bool:
        ...

    def shrink_k_range(self) -> bool:
        ...

    def shrink_d_range(self) -> bool:
        ...

    def clone(self) -> AttnRectangleProtocol:
        ...

    def area(self) -> int:
        ...

    def cut_q(
        self, cut_pos: int
    ) -> tuple[AttnRectangleProtocol | None, AttnRectangleProtocol | None]:
        ...

    def cut_k(
        self, cut_pos: int
    ) -> tuple[AttnRectangleProtocol | None, AttnRectangleProtocol | None]:
        ...

    def get_rect_within_q_segment(
        self, q_start: int, q_end: int
    ) -> AttnRectangleProtocol | None:
        ...

    def get_rect_within_k_segment(
        self, k_start: int, k_end: int
    ) -> AttnRectangleProtocol | None:
        ...

    def intersection_q_id_on_left_boundary(self) -> int:
        ...

    def intersection_q_id_on_right_boundary(self) -> int:
        ...

    def is_full(self) -> bool:
        ...

    def is_causal(self) -> bool:
        ...

    def is_inv_causal(self) -> bool:
        ...

    def is_bi_causal(self) -> bool:
        ...

    def to_qk_range_mask_type(self) -> list[tuple]:
        ...

    def __len__(self) -> int:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...


@runtime_checkable
class AttnRectanglesProtocol(Protocol):
    """Protocol for AttnRectangles — a list of AttnRectangle objects."""

    @staticmethod
    def from_ranges(
        q_ranges: Any,
        k_ranges: Any,
        mask_types: Any,
        check: bool = False,
    ) -> AttnRectanglesProtocol:
        ...

    def append(self, attn_rect: AttnRectangleProtocol, check: bool = False) -> None:
        ...

    def extend(self, attn_rects: AttnRectanglesProtocol, check: bool = False) -> None:
        ...

    def is_valid(self) -> bool:
        ...

    def check_valid(self) -> None:
        ...

    def is_empty(self) -> bool:
        ...

    def get_qo_ranges_union(self) -> AttnRangesProtocol:
        ...

    def get_kv_ranges_union(self) -> AttnRangesProtocol:
        ...

    def total_seqlen_qo(self) -> int:
        ...

    def total_seqlen_kv(self) -> int:
        ...

    def cut_q(
        self, cut_pos: int
    ) -> tuple[AttnRectanglesProtocol, AttnRectanglesProtocol]:
        ...

    def cut_k(
        self, cut_pos: int
    ) -> tuple[AttnRectanglesProtocol, AttnRectanglesProtocol]:
        ...

    def get_rects_within_q_segment(
        self, q_start: int, q_end: int
    ) -> AttnRectanglesProtocol:
        ...

    def get_rects_within_k_segment(
        self, k_start: int, k_end: int
    ) -> AttnRectanglesProtocol:
        ...

    def area(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: Any) -> Any:
        ...

    def __setitem__(self, idx: Any, value: Any) -> None:
        ...

    def __iter__(self) -> Iterator:
        ...

    def __eq__(self, other: Any) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __repr__(self) -> str:
        ...
