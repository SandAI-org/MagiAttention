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
MagiAttention CPP Extensions
"""
from __future__ import annotations
import collections.abc
import torch
import typing
__all__: list[str] = ['AttnMaskType', 'AttnRange', 'AttnRangeWithRank', 'AttnRanges', 'AttnRectangle', 'AttnRectangles', 'GroupCastRanges', 'KernelBarrier', 'argsort_ranges', 'binary_greedy_parallel_solve', 'check_valid_cu_seqlens', 'compute_sparse_load_metadata', 'cut_host_remote_buckets', 'destroy_event', 'elapsed_ms_event', 'expand_attn_ranges', 'is_valid_cu_seqlens', 'produce', 'reorder_ranges_and_attn_type_maps', 'start_event', 'stop_event', 'unique_consecutive_pairs']
class AttnMaskType:
    """
    Members:
    
      FULL
    
      CAUSAL
    
      INVCAUSAL
    
      BICAUSAL
    """
    BICAUSAL: typing.ClassVar[AttnMaskType]  # value = <AttnMaskType.BICAUSAL: 3>
    CAUSAL: typing.ClassVar[AttnMaskType]  # value = <AttnMaskType.CAUSAL: 1>
    FULL: typing.ClassVar[AttnMaskType]  # value = <AttnMaskType.FULL: 0>
    INVCAUSAL: typing.ClassVar[AttnMaskType]  # value = <AttnMaskType.INVCAUSAL: 2>
    __members__: typing.ClassVar[dict[str, AttnMaskType]]  # value = {'FULL': <AttnMaskType.FULL: 0>, 'CAUSAL': <AttnMaskType.CAUSAL: 1>, 'INVCAUSAL': <AttnMaskType.INVCAUSAL: 2>, 'BICAUSAL': <AttnMaskType.BICAUSAL: 3>}
    @staticmethod
    def from_int_type(int_type: typing.SupportsInt) -> AttnMaskType:
        """
        Convert an integer to the corresponding AttnMaskType enum member.
        """
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    def to_int_type(self) -> int:
        """
        Convert this AttnMaskType to its integer representation.
        """
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class AttnRange:
    """
    A dataclass to manage any indices range for attention computation
    """
    @staticmethod
    def from_range(attn_range: typing.Any, check: bool = False) -> AttnRange:
        """
        Construct an AttnRange from a tuple, list, or another AttnRange.
        """
    def __eq__(self, arg0: typing.Any) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, start: typing.SupportsInt, end: typing.SupportsInt) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __ne__(self, arg0: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def check_valid(self, start: typing.SupportsInt | None = None, end: typing.SupportsInt | None = None) -> None:
        """
        Validate the range and raise RangeError if invalid.
        """
    def clone(self) -> AttnRange:
        """
        Return a deep copy of this range.
        """
    def diff_by(self, other: AttnRange) -> list[AttnRange]:
        """
        Compute the set difference: other - self.
        """
    def intersect(self, other: AttnRange) -> AttnRange:
        """
        Return the intersection of this range with another.
        """
    def intersect_size(self, other: AttnRange) -> int:
        """
        Return the length of the intersection with another range.
        """
    def is_empty(self) -> bool:
        """
        Return True if the range has zero length (start == end).
        """
    def is_overlap_with(self, other: AttnRange) -> bool:
        """
        Return True if this range overlaps with another.
        """
    def is_subrange_of(self, other: AttnRange) -> bool:
        """
        Return True if this range is entirely contained within other.
        """
    def is_valid_close(self, start: typing.SupportsInt | None = None, end: typing.SupportsInt | None = None) -> bool:
        """
        Check validity as a closed interval [start, end].
        """
    def is_valid_open(self, start: typing.SupportsInt | None = None, end: typing.SupportsInt | None = None) -> bool:
        """
        Check validity as a half-open interval [start, end).
        """
    def offset(self, offset: typing.SupportsInt) -> AttnRange:
        """
        Return a new range shifted by the given offset.
        """
    def to_naive_range(self) -> tuple:
        """
        Convert to a plain (start, end) tuple.
        """
    def truncate(self, start: typing.SupportsInt | None = None, end: typing.SupportsInt | None = None) -> AttnRange:
        """
        Truncate this range to fit within [start, end).
        """
    def union(self, other: AttnRange) -> list[AttnRange]:
        """
        Return the union of this range with another.
        """
    def union_size(self, other: AttnRange) -> int:
        """
        Return the total length covered by the union with another range.
        """
    @property
    def end(self) -> int:
        """
        The end index of this range (exclusive).
        """
    @end.setter
    def end(self, arg0: typing.SupportsInt) -> None:
        ...
    @property
    def seqlen(self) -> int:
        """
        The length of this range in the axis.
        """
    @property
    def start(self) -> int:
        """
        The start index of this range.
        """
    @start.setter
    def start(self, arg0: typing.SupportsInt) -> None:
        ...
class AttnRangeWithRank(AttnRange):
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, rank_set: collections.abc.Set[typing.SupportsInt], start: typing.SupportsInt, end: typing.SupportsInt) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def clone(self) -> AttnRangeWithRank:
        ...
    @property
    def rank_set(self) -> set[int]:
        ...
    @rank_set.setter
    def rank_set(self, arg0: collections.abc.Set[typing.SupportsInt]) -> None:
        ...
    @property
    def seqlen(self) -> int:
        ...
class AttnRanges:
    """
    A dataclass to manage a list of 'AttnRange' objects for attention computation
    """
    _ranges: typing.Any
    @staticmethod
    def from_cu_seqlens(cu_seqlens: collections.abc.Sequence[typing.SupportsInt], seq_len: typing.SupportsInt) -> AttnRanges:
        """
        Construct AttnRanges from a cumulative sequence length array.
        """
    @staticmethod
    def from_ranges(ranges: typing.Any, check: bool = False) -> AttnRanges:
        """
        Construct AttnRanges from tuples, AttnRange list, or another AttnRanges.
        """
    def __eq__(self, arg0: typing.Any) -> bool:
        ...
    def __getitem__(self, index: typing.Any) -> typing.Any:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, index: typing.Any, range: typing.Any) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, attn_range: AttnRange, check: bool = False) -> None:
        """
        Add the attn_range to the end
        """
    def check_valid(self) -> None:
        """
        Validate all ranges and raise ValueError if any is invalid.
        """
    def chunk(self, chunk_size: typing.SupportsInt, check: bool = True) -> list[AttnRanges]:
        """
        Split ranges into chunks of at most chunk_size total tokens.
        """
    def clear_empty(self) -> AttnRanges:
        """
        Return a new AttnRanges with all empty (zero-length) ranges removed.
        """
    def clone(self) -> AttnRanges:
        """
        Clone the current AttnRanges efficiently
        """
    def extend(self, attn_ranges: AttnRanges, check: bool = False) -> None:
        """
        Extend this AttnRanges by appending all ranges from another AttnRanges.
        """
    def find_hole_ranges(self, other_attn_ranges: AttnRanges, is_self_merged: bool = False, is_other_merged: bool = False) -> AttnRanges:
        """
        Returns the result of self - other_attn_ranges (set difference).
        """
    def find_overlap_ranges(self, other_attn_ranges: AttnRanges, is_self_merged: bool = False, is_other_merged: bool = False) -> AttnRanges:
        """
        Returns the intersection of self and other_attn_ranges.
        """
    def insert(self, idx: typing.SupportsInt, attn_range: AttnRange, check: bool = False) -> None:
        """
        Insert the attn_range to the 'idx'-th position
        """
    def intersect_size(self) -> int:
        """
        Calculate the total size of overlapping parts between all attn_ranges.
        """
    def intersect_size_with(self, other: AttnRanges) -> int:
        """
        Calculate the total size of overlap between self and other.
        """
    def is_cu_seqlens(self, seqlen: typing.SupportsInt) -> bool:
        """
        Return True if ranges form a valid cu_seqlens partition of [0, seqlen).
        """
    def is_empty(self) -> bool:
        """
        Return True if there are no ranges in this collection.
        """
    def is_merged(self) -> bool:
        """
        Whether the ranges are merged, i.e. sorted and non-overlapping
        """
    def is_non_overlap(self) -> bool:
        """
        Whether any pair of the ranges have overlapped parts.
        """
    def is_sorted(self) -> bool:
        """
        Whether the ranges are sorted by 'attn_range.start' in ascending order
        """
    def is_valid(self) -> bool:
        """
        Return True if all ranges satisfy start <= end.
        """
    def make_range_local(self, other_attn_range: AttnRange, is_self_merged: bool = False, prefix_offset: typing.Any = None) -> tuple:
        """
        Map other_attn_range to the corresponding local range within self.
        """
    def make_ranges_local(self, other_attn_ranges: AttnRanges, is_self_merged: bool = False) -> AttnRanges:
        """
        Map each range in other_attn_ranges to the local ranges of self.
        """
    def merge(self) -> AttnRanges:
        """
        Merge the attn_ranges for the overlapped / tangent parts in ascending order by start.
        """
    def merge_with_split_alignment(self, split_alignment: typing.SupportsInt = 1) -> AttnRanges:
        """
        Merge the attn_ranges with split alignment in ascending order by start.
        """
    def pop(self, idx: typing.SupportsInt = -1) -> AttnRange:
        """
        Remove and return item at index (default last).
        """
    def sort(self) -> AttnRanges:
        """
        Sort the attn_ranges by start in ascending order.
        """
    def to_cu_seqlens(self, seq_len: typing.SupportsInt) -> list[int]:
        """
        Convert ranges to a cumulative sequence length list.
        """
    def to_naive_ranges(self) -> list:
        """
        Convert all ranges to a list of (start, end) tuples.
        """
    def to_tensor(self, device: typing.Any = 'cpu') -> torch.Tensor:
        """
        Convert ranges to an [N, 2] int32 tensor.
        """
    def truncate(self, start: typing.SupportsInt | None = None, end: typing.SupportsInt | None = None) -> AttnRanges:
        """
        Truncate each range to fit within [start, end), dropping empty results.
        """
    def union_size(self) -> int:
        """
        Return the total seqlen of all ranges (alias for total_seqlen).
        """
    def union_size_with(self, other: AttnRanges) -> int:
        """
        Return the combined total seqlen of self and other.
        """
    @property
    def end(self) -> int:
        """
        The maximum end index across all ranges.
        """
    @property
    def max_seqlen(self) -> int:
        """
        The maximum seqlen this ranges represent.
        """
    @property
    def points(self) -> list[int]:
        """
        The axis points covered by this ranges in ascending order and without duplicates.
        """
    @property
    def size(self) -> int:
        """
        The number of AttnRange objects in this collection.
        """
    @property
    def start(self) -> int:
        """
        The minimum start index across all ranges.
        """
    @property
    def total_seqlen(self) -> int:
        """
        The total seqlen this ranges represent.
        """
class AttnRectangle:
    """
    A dataclass to manage any indices rectangle like
    [start_q, end_q) [start_k, end_k) [start_d, end_d) mask_type
    for attention computation.
    d_range is d_index = k_index - q_index diagonal line range
    """
    k_range: AttnRange
    q_range: AttnRange
    def __eq__(self, arg0: AttnRectangle) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self, q_range: AttnRange, k_range: AttnRange, d_range: AttnRange | None = None, mask_type: AttnMaskType | int = ...) -> None:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def area(self) -> int:
        """
        Return the number of valid (q, k) integer points in this rectangle.
        """
    def check_valid(self, q_range: AttnRange | None = None, k_range: AttnRange | None = None, d_range: AttnRange | None = None) -> None:
        """
        Validate the rectangle and raise ValueError if invalid.
        """
    def clone(self) -> AttnRectangle:
        """
        Clone the current rectangle efficiently
        """
    def cut_k(self, cut_pos: typing.SupportsInt) -> tuple[AttnRectangle | None, AttnRectangle | None]:
        """
        Split the rectangle at a k-axis position.
        """
    def cut_q(self, cut_pos: typing.SupportsInt) -> tuple[AttnRectangle | None, AttnRectangle | None]:
        """
        Split the rectangle at a q-axis position.
        """
    def get_rect_within_k_segment(self, k_start: typing.SupportsInt, k_end: typing.SupportsInt) -> AttnRectangle | None:
        """
        Obtain the part of the current rectangle within the interval k range
        """
    def get_rect_within_q_segment(self, q_start: typing.SupportsInt, q_end: typing.SupportsInt) -> AttnRectangle | None:
        """
        Obtain the part of the current rectangle within the interval q range
        """
    def get_valid_or_none(self) -> AttnRectangle:
        """
        Return self if valid, otherwise None.
        """
    def intersection_q_id_on_left_boundary(self) -> int:
        """
        get k_start d_start intersection, which is q id on left boundary
        """
    def intersection_q_id_on_right_boundary(self) -> int:
        """
        get k_end d_end intersection, which is q id on right boundary
        """
    def is_bi_causal(self) -> bool:
        """
        Return True if this rectangle has a bi-causal attention pattern.
        """
    def is_causal(self) -> bool:
        """
        Return True if this rectangle has a causal attention pattern.
        """
    def is_full(self) -> bool:
        """
        Return True if this rectangle has a full (no mask) attention pattern.
        """
    def is_inv_causal(self) -> bool:
        """
        Return True if this rectangle has an inverse-causal attention pattern.
        """
    def is_valid(self, q_range: AttnRange | None = None, k_range: AttnRange | None = None, d_range: AttnRange | None = None) -> bool:
        """
        Return True if the rectangle has a valid non-empty area.
        """
    def shrink_d_range(self) -> bool:
        """
        Tighten d_range to the feasible diagonal bounds given q and k ranges.
        """
    def shrink_k_range(self) -> bool:
        """
        Tighten k_range to the feasible bounds given d and q ranges.
        """
    def shrink_q_range(self) -> bool:
        """
        Tighten q_range to the feasible bounds given d and k ranges.
        """
    def to_qk_range_mask_type(self) -> list[tuple[AttnRange, AttnRange, int]]:
        """
        Change rectangle to q k range and mask type style.
        """
    @property
    def d_range(self) -> AttnRange:
        ...
    @d_range.setter
    def d_range(self, arg1: AttnRange | None) -> None:
        ...
class AttnRectangles:
    """
    A dataclass to manage a list of 'AttnRectangle' objects for attention computation
    """
    @staticmethod
    def from_ranges(q_ranges: typing.Any, k_ranges: typing.Any, mask_types: list, check: bool = False) -> AttnRectangles:
        """
        Construct AttnRectangles from parallel lists of q_ranges, k_ranges, and mask_types.
        """
    def __eq__(self, arg0: AttnRectangles) -> bool:
        ...
    def __getitem__(self, arg0: typing.Any) -> typing.Any:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __hash__(self) -> int:
        ...
    def __init__(self) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[AttnRectangle]:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: typing.Any, arg1: typing.Any) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def append(self, attn_rect: AttnRectangle, check: bool = False) -> None:
        """
        Add the attn_rect to the end
        """
    def area(self) -> int:
        """
        Return the sum of areas of all rectangles.
        """
    def check_valid(self) -> None:
        """
        Validate all rectangles and raise ValueError if any is invalid.
        """
    def cut_k(self, cut_pos: typing.SupportsInt) -> tuple[AttnRectangles, AttnRectangles]:
        """
        Split all rectangles at a k-axis position.
        """
    def cut_q(self, cut_pos: typing.SupportsInt) -> tuple[AttnRectangles, AttnRectangles]:
        """
        Split all rectangles at a q-axis position.
        """
    def extend(self, attn_rects: AttnRectangles, check: bool = False) -> None:
        """
        Extend this collection by appending all rectangles from another AttnRectangles.
        """
    def get_kv_ranges_union(self) -> AttnRanges:
        """
        Return the merged union of all k_ranges across rectangles.
        """
    def get_qo_ranges_union(self) -> AttnRanges:
        """
        Return the merged union of all q_ranges across rectangles.
        """
    def get_rects_within_k_segment(self, k_start: typing.SupportsInt, k_end: typing.SupportsInt) -> AttnRectangles:
        """
        Return rectangles clipped to the k-axis segment [k_start, k_end).
        """
    def get_rects_within_q_segment(self, q_start: typing.SupportsInt, q_end: typing.SupportsInt) -> AttnRectangles:
        """
        Return rectangles clipped to the q-axis segment [q_start, q_end).
        """
    def is_empty(self) -> bool:
        """
        Return True if there are no rectangles in this collection.
        """
    def is_valid(self) -> bool:
        """
        Return True if all rectangles are valid.
        """
    def total_seqlen_kv(self) -> int:
        """
        Return the total key/value sequence length across all rectangles.
        """
    def total_seqlen_qo(self) -> int:
        """
        Return the total query/output sequence length across all rectangles.
        """
    @property
    def size(self) -> int:
        """
        The number of AttnRectangle objects in this collection.
        """
class GroupCastRanges(AttnRanges):
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, cp_size: typing.SupportsInt, ranges_per_rank: collections.abc.Sequence[AttnRanges], split: bool = True) -> None:
        ...
    def __iter__(self) -> collections.abc.Iterator[AttnRangeWithRank]:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def _split(self) -> None:
        ...
    @property
    def _ranges(self) -> typing.Any:
        """
        Internal list of AttnRangeWithRank objects
        """
    @_ranges.setter
    def _ranges(self, arg1: typing.Any) -> None:
        ...
class KernelBarrier:
    def __init__(self, target: typing.SupportsInt) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def get_value(self) -> int:
        """
        Get current kernel_barrier count value from GPU (for debugging)
        """
    def reset(self) -> None:
        """
        Reset the kernel_barrier count to 0
        """
    def synchronize(self) -> None:
        """
        Launch a spin kernel to wait until count >= target
        """
def argsort_ranges(arg0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Argsort [N,2] int32 tensor by first column, returning int32 indices
    """
def binary_greedy_parallel_solve(rects: typing.Any, host_ranges_q: list, host_ranges_k: list, num_heads_q: typing.SupportsInt, num_heads_kv: typing.SupportsInt, num_heads_group: typing.SupportsInt, bucket_per_rank: list, rank: typing.SupportsInt = -1, debug_print: bool = False) -> None:
    """
    Optimized Binary-Greedy-Parallel solver implemented in C++
    """
def check_valid_cu_seqlens(cu_seqlens: collections.abc.Sequence[typing.SupportsInt], seq_len: typing.SupportsInt) -> None:
    """
    Validate cu_seqlens and raise ValueError if invalid.
    """
def compute_sparse_load_metadata(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor | None, arg4: typing.SupportsInt) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute sparse load metadata (loop count and invalid count) for each unique Q range
    """
def cut_host_remote_buckets(bucket_this_rank: AttnRectangles, host_ranges_q_this_rank: AttnRanges, host_ranges_k_this_rank: AttnRanges) -> tuple:
    """
    Optimized version of calc_host_and_remote_bucket_this_rank
    """
def destroy_event() -> None:
    ...
def elapsed_ms_event(arg0: str) -> float:
    ...
def expand_attn_ranges(ranges: AttnRanges, stride: typing.SupportsInt, num_heads_group: typing.SupportsInt) -> AttnRanges:
    """
    Optimized version of _expand_attn_ranges for DynamicAttnSolver
    """
def is_valid_cu_seqlens(cu_seqlens: collections.abc.Sequence[typing.SupportsInt], seq_len: typing.SupportsInt) -> bool:
    """
    Check whether cu_seqlens is a valid cumulative sequence length array.
    """
def produce(kernel_barrier: KernelBarrier | None) -> None:
    """
    Launch a producer kernel to increment the kernel_barrier count by 1
    """
def reorder_ranges_and_attn_type_maps(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor | None, arg3: torch.Tensor, arg4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reorder [N,2] int32 ranges using vectorized int2 loads
    """
def start_event(arg0: str) -> None:
    ...
def stop_event(arg0: str) -> None:
    ...
def unique_consecutive_pairs(arg0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find unique (int, int) pairs from a pre-sorted [N,2] int32 CUDA tensor
    """
