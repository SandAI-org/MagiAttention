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

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import Iterator

import torch
import torch.distributed as dist

import magi_attention
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.collection.calc_meta import AttnArg, AttnCalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.meta.container.bucket import AttnBucket
from magi_attention.meta.container.chunk import AttnChunk
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice
from magi_attention.utils import nvtx, transpose_matrix

from .overlap_solver import OverlapConfig, OverlapSolver, OverlapStageCost


class AttnRangeWithRank(AttnRange):
    def __init__(self, rank_set: set[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_set = rank_set


class GroupCastRanges(AttnRanges):
    def __init__(
        self,
        cp_size: int,
        ranges_per_rank: list[AttnRanges],
        split: bool = True,
    ):
        super().__init__()

        assert len(ranges_per_rank) == cp_size
        self._ranges: list[AttnRangeWithRank] = []  # type: ignore

        for cp_rank, ranges in enumerate(ranges_per_rank):
            for r in ranges:
                self._ranges.append(
                    AttnRangeWithRank(rank_set={cp_rank}, start=r.start, end=r.end)
                )

        # sort by attn_range.start
        self._ranges.sort(key=lambda attn_range: attn_range.start)

        if split:
            self._split()

    @nvtx.instrument_nvtx
    def _split(self) -> None:
        """Split the ranges group as fragmented as possible"""

        if len(self._ranges) <= 1:
            return

        new_ranges: list[AttnRangeWithRank] = []

        # 对每个区间[p1,p2]判断它被哪些原始range覆盖
        all_points = self.points
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i + 1]

            # 找出所有覆盖这个区间的ranges
            cover_rank_set = set()
            for r in self._ranges:
                if r.start <= p1 and r.end >= p2:
                    cover_rank_set.update(r.rank_set)

            if cover_rank_set:  # 如果有range覆盖这个区间
                new_ranges.append(
                    AttnRangeWithRank(rank_set=cover_rank_set, start=p1, end=p2)
                )

        self._ranges = new_ranges

    def __iter__(self) -> Iterator[AttnRangeWithRank]:
        return iter(self._ranges)


@dataclass
class HostRankEntry:
    """
    HostRankEntry is a dataclass that contains the host q/k ranges and the remote k ranges,
    it is a key data structure for calculating the remote rank entry.

    Args:
        host_q_ranges_global: global q ranges for this rank, merged
        host_k_ranges_global: global k ranges for this rank, merged

        attn_calc_slice_global_list: contains all slices to be calculated on this rank,
            including all slices from both host_stage and remote_stage
        attn_calc_host_slice_local_list: slices that need to be calculated in the host_stage

        remote_k_ranges_global_hb_domain: global k ranges for the high-bandwidth domain, merged
        attn_calc_remote_slice_list_hb_domain: contains all slices in the remote_stage located in the high-bandwidth domain

        remote_k_ranges_global_lb_domain: global k ranges for the low-bandwidth domain, merged
        remote_k_ranges_global_per_chunk: global k ranges for each chunk in the low-bandwidth domain,
            these are the k ranges needed by the attn slices in this chunk.
            the remote_k_ranges_global for each chunk is merged
        attn_calc_remote_slice_list_per_chunk: contains slices that need to be calculated
            for each chunk in the low-bandwidth domain
    """

    host_q_ranges_global: AttnRanges
    host_k_ranges_global: AttnRanges
    attn_calc_slice_global_list: list[AttnSlice]
    attn_calc_host_slice_local_list: list[AttnSlice]

    # NOTE: Each element in this list is a MultiKAttnSlice, which contains
    # one q_range_global and multiple k_ranges_global, where all k_ranges_global
    # are located in the high-bandwidth domain
    remote_k_ranges_global_hb_domain: AttnRanges
    attn_calc_remote_slice_list_hb_domain: list[MultiKAttnSlice]

    remote_k_ranges_global_lb_domain: AttnRanges
    # NOTE: We only chunknize remote k_ranges located in the low-bandwidth domain,
    # so attn_calc_remote_slice_list_per_chunk only contains remote k_ranges in the low-bandwidth domain
    remote_k_ranges_global_per_chunk: list[AttnRanges]
    # NOTE: this is a special attr to support multi-stage overlap
    # each multik_slice of which contains a q_range_local and a k_ranges_global
    # where the k_ranges_global won't be made local until the multi-stage overlap problem solved
    attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]]

    def __post_init__(self):
        assert len(self.remote_k_ranges_global_per_chunk) == len(
            self.attn_calc_remote_slice_list_per_chunk
        ), (
            f"The number of chunks is inconsistent: "
            f"{len(self.remote_k_ranges_global_per_chunk)=}, {len(self.attn_calc_remote_slice_list_per_chunk)=}"
        )

    def get_host_calc_area(self) -> int:
        """Get the host calc area"""
        return sum(
            attn_slice.area for attn_slice in self.attn_calc_host_slice_local_list
        )

    def get_remote_calc_area(self, chunk_idx: int | None = None) -> int:
        """Get the remote calc area (w.r.t. a specific chunk)"""
        if chunk_idx is None:  # return the remote calc area for all chunks
            return sum(
                attn_slice.area
                for attn_slice in chain(*self.attn_calc_remote_slice_list_per_chunk)
            )
        return sum(
            attn_slice.area
            for attn_slice in self.attn_calc_remote_slice_list_per_chunk[chunk_idx]
        )

    def get_remote_comm_size(self, chunk_idx: int | None = None) -> int:
        """Get the remote comm size (w.r.t. a specific chunk)"""
        if chunk_idx is None:  # return the remote comm size for all chunks
            return sum(
                remote_k_ranges.total_seqlen
                for remote_k_ranges in self.remote_k_ranges_global_per_chunk
            )

        return self.remote_k_ranges_global_per_chunk[chunk_idx].total_seqlen


@dataclass
class RemoteRankEntry:
    """
    RemoteRankEntry is a dataclass that contains the remote k ranges and the local k ranges,
    it is a key data structure for calculating the transfer table.

    Args:
        host_k_ranges_global: k_ranges_global owned by the host rank, merged.
        remote_k_ranges_global: k_ranges_global owned by the remote rank, merged.
            Represents the remote kv needed by the host rank in the current overlap stage.

        attn_calc_remote_slice_local_list: Represents the attention calculations that the
            host rank needs to perform in the current overlap stage.
    """

    host_k_ranges_global: AttnRanges
    remote_k_ranges_global: AttnRanges

    attn_calc_remote_slice_local_list: list[AttnSlice]


class HostAttnSliceMaker:
    class CausalCaseKey(Enum):
        INVALID = "invalid"
        RECTANGLE = "full_rectangle"
        TRAPEZOID = "uncut_triangle_or_trapezoid"
        TRIANGLE = "cut_triangle_on_the_far_right"
        PENTAGON = "rotated_trapezoid_or_pentagon"

    def __init__(
        self,
        q_range_local: AttnRange,
        k_ranges_local: AttnRanges,
        k_ranges_global: AttnRanges,
        calc_k_range_global: AttnRange,
        mask_type_global: AttnMaskType,
    ):
        """
        Args:
            q_range_local (AttnRange): the host q range local
            k_ranges_local (AttnRanges): the host k ranges local
                which remains unmerged, and will be merged in the specific case
            k_ranges_global (AttnRanges): the host k ranges global
                which should be guaranteed to be non-empty from outside
            calc_k_range_global (AttnRange): the host k range global for the original calc slice
            mask_type_global (AttnMaskType): the attn mask type for the original calc slice
        """
        self.q_range_local = q_range_local
        self.k_ranges_local = k_ranges_local
        self.k_ranges_global = k_ranges_global
        self.calc_k_range_global = calc_k_range_global
        self.mask_type_global = mask_type_global

        # init for causal
        if self.mask_type_global is AttnMaskType.CAUSAL:
            self._init_causal()

    def _init_causal(self) -> None:
        self.last_k_range_global = self.k_ranges_global[-1]

        # ---- calc the start and end of the causal area ---- #

        if self.calc_k_range_global.seqlen > self.q_range_local.seqlen:
            # the causal mask of a trapezoid
            self.causal_mask_start = (
                self.calc_k_range_global.end - self.q_range_local.seqlen
            )
        else:
            # the causal mask of a triangle or a null slice
            self.causal_mask_start = self.calc_k_range_global.start

        self.causal_mask_end = self.calc_k_range_global.end

        self.exceed_causal_start = (
            self.last_k_range_global.start - self.causal_mask_start
        )

        # when q_range.seqlen > k_range.seqlen, exceed_causal_end is
        # just a part of the actual exceeded length
        self.exceed_causal_end = self.last_k_range_global.end - self.causal_mask_start
        # it needs to add the difference between the lengths of q_range and k_range in slice
        self.diff_len_of_q_range_minus_k_range = max(
            0,
            self.q_range_local.seqlen - self.calc_k_range_global.seqlen,
        )

        # ---- determine the causal case key ---- #

        self._init_causal_case_key()

    def _init_causal_case_key(self) -> None:
        self.causal_case_key = self.CausalCaseKey.INVALID
        if (
            self.last_k_range_global.start <= self.causal_mask_start
            and self.last_k_range_global.end <= self.causal_mask_start
        ):
            # case1: the area will be formed as a full rectangle mask
            self.causal_case_key = self.CausalCaseKey.RECTANGLE
        elif (
            self.last_k_range_global.start <= self.causal_mask_start
            and self.last_k_range_global.end == self.causal_mask_end
        ):
            # case2: the area will be formed as a normal causal mask,
            # i.e. an uncut triangle or a trapezoid
            self.causal_case_key = self.CausalCaseKey.TRAPEZOID
        elif (
            self.last_k_range_global.start > self.causal_mask_start
            and self.last_k_range_global.end == self.causal_mask_end
        ):
            # case3: the area will be formed as a cut triangle on the far right
            self.causal_case_key = self.CausalCaseKey.TRIANGLE
        elif (
            self.last_k_range_global.start <= self.causal_mask_start
            and self.causal_mask_start
            < self.last_k_range_global.end
            < self.causal_mask_end
        ):
            # this includes two cases:
            # case 4: the area of a rotated trapezoid or a pentagon,
            #   when q_range.seqlen <= k_range.seqlen in the slice
            # case 5: the area of a cut rotated trapezoid,
            #   when q_range.seqlen > k_range.seqlen in the slice
            self.causal_case_key = self.CausalCaseKey.PENTAGON

    @nvtx.instrument_nvtx
    def make(self) -> list[AttnSlice]:
        match self.mask_type_global:
            case AttnMaskType.FULL:
                attn_slices = self._make_slice_for_full_mask()
            case AttnMaskType.CAUSAL:
                attn_slices = self._make_slice_for_causal_mask()
            case _:
                raise ValueError(f"Got invalid mask type {self.mask_type_global=}.")

        return attn_slices

    def _make_slice_for_full_mask(self) -> list[AttnSlice]:
        """For full mask, just merge the k ranges local
        to be a single k range and form a single attn slice
        """

        k_range_local = self._merge_k_ranges_and_check(
            self.k_ranges_local,
            allow_empty=False,
        )

        return [
            AttnSlice(
                q_range=self.q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.FULL,
            )
        ]

    def _make_slice_for_causal_mask(self) -> list[AttnSlice]:
        """For causal mask, there're more than one cases to be considered"""

        match self.causal_case_key:
            case self.CausalCaseKey.RECTANGLE:
                attn_slices = self._make_slice_for_causal_rectangle_mask()
            case self.CausalCaseKey.TRAPEZOID:
                attn_slices = self._make_slice_for_causal_trapezoid_mask()
            case self.CausalCaseKey.TRIANGLE:
                attn_slices = self._make_slice_for_causal_triangle_mask()
            case self.CausalCaseKey.PENTAGON:
                attn_slices = self._make_slice_for_causal_pentagon_mask()
            case self.CausalCaseKey.INVALID:
                raise ValueError(
                    f"Got invalid range {self.last_k_range_global=} "
                    f"when {self.causal_mask_start=} and {self.causal_mask_end=}."
                )
            case _:
                raise ValueError(
                    f"Got invalid causal case key {self.causal_case_key=}."
                )

        return attn_slices

    def _make_slice_for_causal_rectangle_mask(self) -> list[AttnSlice]:
        """in such case, we just call the maker for full mask,
        since a causal rectangle mask equals to a full mask
        """

        return self._make_slice_for_full_mask()

    def _make_slice_for_causal_trapezoid_mask(self) -> list[AttnSlice]:
        """in such case, the whole mask will be a single
        normal causal mask after merged
        """

        k_range_local = self._merge_k_ranges_and_check(
            self.k_ranges_local,
            allow_empty=False,
        )

        return [
            AttnSlice(
                q_range=self.q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.CAUSAL,
            )
        ]

    def _make_slice_for_causal_triangle_mask(self) -> list[AttnSlice]:
        """in such case, the mask will be formed as two parts:
        part1: an optional full mask merged from the previous k ranges
        part2: a causal mask on the far right made by the last k range
        """

        attn_slices: list[AttnSlice] = []

        # part1 (optional): previous full mask
        previous_full_k_ranges_local = self.k_ranges_local[:-1]
        previous_full_k_range_local = self._merge_k_ranges_and_check(
            previous_full_k_ranges_local,
            allow_empty=True,
        )

        # TODO: The current solution is to divide the slice into a complete rectangle and a small triangle.
        # TODO: The slice can be cut into a rectangle and a trapezoid to ensure that the k_range is not divided.
        if not previous_full_k_range_local.is_empty():
            attn_slices.append(
                AttnSlice(
                    q_range=self.q_range_local,
                    k_range=previous_full_k_range_local,
                    mask_type=AttnMaskType.FULL,
                )
            )

        # part2: causal mask on the far right
        last_causal_k_range_local = self.k_ranges_local[-1]
        last_causal_q_range_local = AttnRange(
            start=self.q_range_local.end - self.last_k_range_global.seqlen,
            end=self.q_range_local.end,
        )
        attn_slices.append(
            AttnSlice(
                q_range=last_causal_q_range_local,
                k_range=last_causal_k_range_local,
                mask_type=AttnMaskType.CAUSAL,
            )
        )

        return attn_slices

    def _make_slice_for_causal_pentagon_mask(self) -> list[AttnSlice]:
        """in such case, the mask has to divide from the middle into two parts:
        part1: the top causal mask
        part2: the bottom full mask
        """

        attn_slices: list[AttnSlice] = []

        k_range_local = self._merge_k_ranges_and_check(
            self.k_ranges_local,
            allow_empty=False,
        )

        # part1: top causal mask
        top_causal_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            AttnSlice(
                q_range=top_causal_q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.CAUSAL,
            ),
        )

        # part2: bottom full mask
        bottom_full_q_range_local = AttnRange(
            start=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end,
        )
        attn_slices.append(
            AttnSlice(
                q_range=bottom_full_q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.FULL,
            ),
        )

        return attn_slices

    def _merge_k_ranges_and_check(
        self,
        k_ranges: AttnRanges,
        allow_empty: bool = False,
    ) -> AttnRange:
        k_ranges = k_ranges.merge()
        is_empty = k_ranges.is_empty()

        # sanity check
        if magi_attention.is_sanity_check_enable():
            # the local ranges are always contains only a single range after merged
            assert len(k_ranges) <= 1
            # unless it is empty
            assert not is_empty or allow_empty

        if is_empty:
            return AttnRange(0, 0)

        return k_ranges[0]


class RemoteAttnSliceMaker(HostAttnSliceMaker):
    def __init__(
        self,
        q_range_local: AttnRange,
        k_ranges_global: AttnRanges,
        calc_k_range_global: AttnRange,
        mask_type_global: AttnMaskType,
    ):
        """
        Args:
            q_range_local (AttnRange): the host q range local
            k_ranges_global (AttnRanges): the remote k ranges global
                which should be guaranteed to be non-empty from outside
            calc_k_range_global (AttnRange): the remote k range global for the original calc slice
            mask_type_global (AttnMaskType): the attn mask type for the original calc slice
        """
        super().__init__(
            q_range_local=q_range_local,
            k_ranges_local=AttnRanges(),  # just a placeholder, not used
            k_ranges_global=k_ranges_global,
            calc_k_range_global=calc_k_range_global,
            mask_type_global=mask_type_global,
        )
        del self.k_ranges_local  # this attr is not used, so del it

        self.batch_size = len(self.k_ranges_global)

    def _init_causal_case_key(self) -> None:
        super()._init_causal_case_key()

        if self.causal_case_key is self.CausalCaseKey.PENTAGON:
            self.special_pentagon_case_type = False
        elif self.causal_case_key is self.CausalCaseKey.INVALID:
            if (
                self.last_k_range_global.start > self.causal_mask_start
                and self.causal_mask_start
                < self.last_k_range_global.end
                < self.causal_mask_end
            ):
                # this contains special sub-type of cases for the pentagon cases
                # that will be just invalid in host slice maker
                self.causal_case_key = self.CausalCaseKey.PENTAGON
                self.special_pentagon_case_type = True

    def _make_slice_for_full_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """For full mask, we just wrap the args to a single multi-k attn slice"""

        return [
            MultiKAttnSlice(
                q_range=self.q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * self.batch_size,
            )
        ]

    def _make_slice_for_causal_rectangle_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of only several full rectangles

        thus we just call the maker for full mask
        """

        return self._make_slice_for_full_mask()

    def _make_slice_for_causal_trapezoid_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of several full rectangles
        plus a single normal causal mask, i.e. an uncut triangle or an uncut trapezoid

        thus the whole masks will be formed as
        the previous full masks plus a single normal causal mask in the last
        """

        return [
            MultiKAttnSlice(
                q_range=self.q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * (self.batch_size - 1)
                + [AttnMaskType.CAUSAL],
            )
        ]

    def _make_slice_for_causal_triangle_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of several full rectangles
        plus a cut triangle on the far bottom-right

        the whole masks can be divided from the middle into two parts:
            part1: the optional top full masks
            part2: the bottom previous full masks plus a single normal causal mask in the last
        """

        attn_slices: list[MultiKAttnSlice] = []

        triangle_start = self.q_range_local.end - self.last_k_range_global.seqlen

        # part1: optional top full masks
        full_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=triangle_start,
        )
        if self.batch_size > 1:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=full_q_range_local,
                    k_ranges=self.k_ranges_global[:-1],
                    mask_types=[AttnMaskType.FULL] * (self.batch_size - 1),
                )
            )

        # part2: bottom full masks + causal mask
        causal_q_range_local = AttnRange(
            start=triangle_start,
            end=self.q_range_local.end,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * (self.batch_size - 1)
                + [AttnMaskType.CAUSAL],
            )
        )

        return attn_slices

    def _make_slice_for_causal_pentagon_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """this includes three cases, where the area is made of several full rectangles
        plus either an uncut/cut rotated trapezoid or a pentagon

        we further dispatch them into two sub types of cases:
            normal type: the plused rotated trapezoid is uncut
            special type: the plused rotated trapezoid is cut
        """

        if self.special_pentagon_case_type:
            return self._make_slice_for_causal_pentagon_mask_special()
        else:
            return self._make_slice_for_causal_pentagon_mask_normal()

    def _make_slice_for_causal_pentagon_mask_normal(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """this normal type includes two cases:
            case1: the area is made of several full rectangles
                plus either an uncut rotated trapezoid or a pentagon,
                when q_range.seqlen <= k_range.seqlen in a slice
            case2: the area is made of a single cut rotated trapezoid,
                when q_range.seqlen > k_range.seqlen in a slice

        thus the whole masks can be divided from the middle into two parts:
            part1: the top previous full masks plus a single normal causal mask in the last
            part2: the bottom full masks
        """

        attn_slices: list[MultiKAttnSlice] = []

        # part1: top full masks + causal mask
        top_causal_q_range_local = AttnRange(
            start=self.q_range_local.start + self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=top_causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * (self.batch_size - 1)
                + [AttnMaskType.CAUSAL],
            )
        )

        # part2: bottom full masks
        bottom_full_q_range_local = AttnRange(
            start=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=bottom_full_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * self.batch_size,
            )
        )

        return attn_slices

    def _make_slice_for_causal_pentagon_mask_special(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """this special type includes two cases:
            case1: the area is made of several full rectangles
                plus a cut rotated trapezoid,
                when q_range.seqlen <= k_range.seqlen in a slice
            case2: the area is made of a single cut rotated trapezoid,
                when q_range.seqlen > k_range.seqlen in a slice
            NOTE: the case2 of the special type is the same as the case2 of the normal type
            we just handle each case2 with the same way as its corr. case1

        thus the whole masks can be divided from the middle into three parts:
            part1: the top optional full masks
            part2: the middle full masks plus a single normal causal mask in the last
            part3: the bottom full masks
        """

        attn_slices: list[MultiKAttnSlice] = []

        # part1: top optional full masks
        top_full_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.start
            + self.exceed_causal_start
            + self.diff_len_of_q_range_minus_k_range,
        )
        if self.batch_size > 1:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=top_full_q_range_local,
                    k_ranges=self.k_ranges_global[:-1],
                    mask_types=[AttnMaskType.FULL] * (self.batch_size - 1),
                )
            )

        # part2: middle full masks + causal mask
        mid_causal_q_range_local = AttnRange(
            start=self.q_range_local.start
            + self.exceed_causal_start
            + self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=mid_causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * (self.batch_size - 1)
                + [AttnMaskType.CAUSAL],
            )
        )

        # part3: bottom full masks
        bottom_full_q_range_local = AttnRange(
            start=self.q_range_local.start
            + self.exceed_causal_end
            + self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=bottom_full_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * self.batch_size,
            )
        )

        return attn_slices


@dataclass
class TransferInfo:
    k_ranges_global_recv_from_per_rank: list[AttnRanges]
    k_ranges_local_recv_from_per_rank: list[AttnRanges]
    k_ranges_global_send_to_per_rank: list[AttnRanges]
    k_ranges_local_send_to_per_rank: list[AttnRanges]

    group_cast_ranges_global_transfer: GroupCastRanges
    group_cast_ranges_local_send_to: GroupCastRanges


@dataclass
class TableEntry:
    """The entry dataclass for transfer table,
    where:
        1. k_ranges_global: global k ranges to send w.r.t. send rank's dispatch meta
        2. k_ranges_local_in_send_buf: local k ranges to send w.r.t. send rank's send buf
        3. k_ranges_local_in_recv_buf: local k ranges to send w.r.t. recv rank's recv buf
    """

    k_ranges_global: AttnRanges
    k_ranges_local_in_send_buf: AttnRanges
    k_ranges_local_in_recv_buf: AttnRanges


class TransferTable:
    """The transfer table class, maintaining [cp_size, cp_size] entries,
    where table[send_rank][recv_rank] is the send entry from send_rank to recv_rank

    Therefore:
        1. we can get the send args for group collective
            using 'k_ranges_local_in_send_buf' in the row of table[this_rank][...]
        2. we can get the recv args for group collective
            using 'k_ranges_local_in_recv_buf' in the column of table[...][this_rank]
    """

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        self._transfer_table: list[list[TableEntry]] = []

        # init each entry in the transfer table
        for send_rank in range(cp_size):
            self._transfer_table.append([])
            for recv_rank in range(cp_size):
                self._transfer_table[send_rank].append(
                    TableEntry(
                        k_ranges_global=AttnRanges(),
                        k_ranges_local_in_send_buf=AttnRanges(),
                        k_ranges_local_in_recv_buf=AttnRanges(),
                    )
                )

    # get
    def get_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_global

    def get_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf

    def get_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_recv_buf

    # append
    def append_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_global.append(k_range)

    def append_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf.append(
            k_range
        )

    # sort
    def sort_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global.sort()

    def sort_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf.sort()

    # make
    def make_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
        remote_k_ranges_global_for_recv_rank: AttnRanges,
    ) -> None:
        """Construct local k_ranges w.r.t. recv rank's recv buffer
        from host global k_ranges to send from send_rank to recv_rank
        and remote global k_ranges to recv from send_rank to recv_rank

        NOTE: this is the special attribute that should NOT be passed in from outside,
        but ONLY constructed internally
        """

        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_recv_buf = remote_k_ranges_global_for_recv_rank.make_ranges_local(
            self._transfer_table[send_rank][recv_rank].k_ranges_global,
            is_self_merged=True,
        )


class DistAttnSolver:
    """The dist-attn solver class to process dispatch meta for calc/comm meta"""

    @nvtx.instrument_nvtx
    def __init__(
        self,
        bucket_per_rank: list[AttnBucket],
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        cp_group: dist.ProcessGroup,
        high_bandwith_domain_size: int,
        overlap_config: OverlapConfig,
        cp_intra_group: dist.ProcessGroup | None = None,
        cp_inter_group: dist.ProcessGroup | None = None,
    ):
        assert dist.get_backend(cp_group) == dist.Backend.NCCL

        self.cp_rank = dist.get_rank(cp_group)
        self.cp_size = dist.get_world_size(cp_group)
        self.cp_group = cp_group
        self.cp_intra_group = cp_intra_group
        self.cp_inter_group = cp_inter_group
        self.shard_seqlen_q = dispatch_meta_q.shard_seqlen
        self.shard_seqlen_k = dispatch_meta_k.shard_seqlen
        self.high_bandwith_domain_size = high_bandwith_domain_size

        self.overlap_config = overlap_config
        self.overlap_solver = OverlapSolver(alg=self.overlap_config.alg)

        # NOTE: the real overlap degree should be determined in the later code:
        # 1. if overlap mode is static, then its real value equals to the one in the overlap config
        # 2. if overlap mode is dynamic, then its real value is determined by overlap solver
        self.overlap_degree: int = -1

        # NOTE: the real overlap chunk size and the number of chunks should be determined in the later code:
        # 1. if the remote length is not too long (num_chunks <= max_num_chunks),
        #   then use the 'min_chunk_size' to chunk it, i.e. overlap_chunk_size = min_chunk_size
        # 2. otherwise, use the 'max_num_chunks' to calc a larger chunk size
        #   and assign it as overlap_chunk_size, i.e. overlap_num_chunks = max_num_chunks
        self.overlap_chunk_size: int = -1
        self.overlap_num_chunks: int = -1

        # init host / remote q/k ranges global for this rank
        bucket_this_rank = bucket_per_rank[self.cp_rank]
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        ) = self._init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        # set some attributes that might be fetched from outside
        self.bucket = bucket_this_rank
        self.host_q_ranges_global = host_q_ranges_global_this_rank
        self.host_k_ranges_global = host_k_ranges_global_this_rank
        self.remote_k_ranges_global = remote_k_ranges_global_this_rank
        self.remote_k_ranges_global_hb_domain = remote_k_ranges_global_hb_domain
        self.remote_k_ranges_global_lb_domain = remote_k_ranges_global_lb_domain

        # init host rank entry for this rank
        self.host_rank_entry_this_rank = self._init_host_rank_entry_this_rank(
            host_q_ranges_global=host_q_ranges_global_this_rank,
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain=remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain=remote_k_ranges_global_lb_domain,
            attn_calc_slice_global_list=bucket_this_rank.attn_slices,
        )

        # init remote rank entry for each stage for this rank
        self.remote_rank_entry_per_stage_this_rank = (
            self._init_remote_rank_entry_per_stage_this_rank(
                self.host_rank_entry_this_rank
            )
        )

        # init for hb domain remote stage if available
        if self.high_bandwith_domain_size > 1:
            # init remote rank entry for high-bandwidth domain stage this rank
            remote_rank_entry_for_this_domain = (
                self._init_remote_rank_entry_this_domain(
                    host_rank_entry_this_rank=self.host_rank_entry_this_rank,
                )
            )
            # HACK: prepend remote rank entry for high-bandwidth domain as the first stage
            # and the overlap degree need plus 1, since hb remote rank entry is unaware by the overlap solver
            # FIXME: therefore, the overlap solver will wrongly use the remote comm in first lb domain stage
            # to overlap with the host calc, instead of the remote comm in hb domain
            self.remote_rank_entry_per_stage_this_rank.insert(
                0,
                remote_rank_entry_for_this_domain,
            )
            self.overlap_degree += 1

        # init remote rank entry for each rank for each stage
        self.remote_rank_entry_per_rank_per_stage = (
            self._init_remote_rank_entry_per_rank_per_stage(
                self.remote_rank_entry_per_stage_this_rank
            )
        )

        # init transfer table per stage
        self.transfer_table_per_stage: list[
            TransferTable
        ] = self._init_transfer_table_per_stage(
            self.remote_rank_entry_per_rank_per_stage,
        )

    @nvtx.instrument_nvtx
    def _init_host_remote_ranges_global_this_rank(
        self,
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        bucket_this_rank: AttnBucket,
    ) -> tuple[AttnRanges, AttnRanges, AttnRanges, AttnRanges, AttnRanges]:
        # init host q_ranges global for this rank
        host_q_ranges_global_this_rank = dispatch_meta_q.host_ranges_per_rank[
            self.cp_rank
        ].merge()

        # init host k_ranges global for this rank
        host_k_ranges_global_this_rank = dispatch_meta_k.host_ranges_per_rank[
            self.cp_rank
        ].merge()

        # init remote k_ranges global for this rank
        # NOTE: this only contains the remote k ranges that we need to calculate from
        remote_k_ranges_global_this_rank = bucket_this_rank.k_ranges.find_hole_ranges(
            host_k_ranges_global_this_rank,
            is_other_merged=True,
        )

        # split remote k_ranges global into high-bandwidth / low-bandwidth domain
        host_k_ranges_global_this_domain = (
            dispatch_meta_k.host_ranges_this_domain.merge()
        )
        remote_k_ranges_global_hb_domain = (
            remote_k_ranges_global_this_rank.find_overlap_ranges(
                host_k_ranges_global_this_domain,
                is_self_merged=True,
                is_other_merged=True,
            )
        )
        remote_k_ranges_global_lb_domain = (
            remote_k_ranges_global_this_rank.find_hole_ranges(
                host_k_ranges_global_this_domain,
                is_self_merged=True,
                is_other_merged=True,
            )
        )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            # check if merged successfully
            assert host_q_ranges_global_this_rank.is_merged()
            assert host_k_ranges_global_this_rank.is_merged()
            assert remote_k_ranges_global_this_rank.is_merged()
            assert remote_k_ranges_global_hb_domain.is_merged()
            assert remote_k_ranges_global_lb_domain.is_merged()

            # whether q_ranges and k_ranges are one-one mapping in attn calc
            attn_calc_q_ranges_global = bucket_this_rank.q_ranges
            attn_calc_k_ranges_global = bucket_this_rank.k_ranges
            assert len(attn_calc_q_ranges_global) == len(attn_calc_k_ranges_global), (
                f"The {len(attn_calc_q_ranges_global)=} should be equal to "
                f"{len(attn_calc_k_ranges_global)=}."
            )

            # check about high-bandwidth domain
            intersect_ranges_between_hb_lb = (
                remote_k_ranges_global_hb_domain.find_overlap_ranges(
                    remote_k_ranges_global_lb_domain,
                )
            )
            assert intersect_ranges_between_hb_lb.is_empty()  # they are orthogonal

            if self.high_bandwith_domain_size == 1:
                # in such case, host k ranges in this hb domain are exactly the host k ranges for this rank
                # then there'll be no remote k ranges in this hb domain, and
                # the remote k ranges in lb domain are exactly the remote k ranges for this rank
                assert (
                    host_k_ranges_global_this_rank == host_k_ranges_global_this_domain
                )
                assert remote_k_ranges_global_hb_domain.is_empty()
                assert (
                    remote_k_ranges_global_lb_domain == remote_k_ranges_global_this_rank
                )

        return (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
            remote_k_ranges_global_hb_domain,
            remote_k_ranges_global_lb_domain,
        )

    @nvtx.instrument_nvtx
    def _init_host_rank_entry_this_rank(
        self,
        host_q_ranges_global: AttnRanges,
        host_k_ranges_global: AttnRanges,
        remote_k_ranges_global_hb_domain: AttnRanges,
        remote_k_ranges_global_lb_domain: AttnRanges,
        attn_calc_slice_global_list: list[AttnSlice],
    ) -> HostRankEntry:
        """Initialize host rank entry for this rank"""

        # -------   chunk remote k ranges global  ------ #

        remote_k_ranges_global_per_chunk = self._chunk_remote_k_ranges_global(
            # NOTE: we only chunk the remote k ranges in the low-bandwidth domain
            # to be passed to overlap solver for multi-stage overlapping
            remote_k_ranges_global=remote_k_ranges_global_lb_domain,
        )

        # -------   calc attn calc host q ranges local  ------ #

        attn_calc_global_chunk = AttnChunk(q_slices=attn_calc_slice_global_list)
        attn_calc_host_q_ranges_local = host_q_ranges_global.make_ranges_local(
            attn_calc_global_chunk.q_ranges,
            is_self_merged=True,
        )

        # -------   make attn calc host/remote slices  ------ #

        attn_calc_host_slice_local_list: list[AttnSlice] = []
        attn_calc_remote_slice_list_hb_domain: list[MultiKAttnSlice] = []
        attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]] = [
            [] for _ in range(len(remote_k_ranges_global_per_chunk))
        ]

        for ith_attn_slice_global, ith_attn_calc_host_q_range_local in zip(
            attn_calc_slice_global_list,
            attn_calc_host_q_ranges_local,
        ):
            ith_attn_slice_global_mask_type: AttnMaskType = ith_attn_slice_global.mask_type  # type: ignore
            # HACK: wrap k range to k ranges,
            # to use the API of AttnRanges like find_overlap_ranges
            ith_attn_calc_k_ranges_global = AttnRanges()
            ith_attn_calc_k_ranges_global.append(ith_attn_slice_global.k_range)  # type: ignore

            # -------   make ith attn calc host slice local  ------ #

            self._make_ith_attn_calc_host_slice(
                host_k_ranges_global=host_k_ranges_global,
                attn_calc_host_slice_local_list=attn_calc_host_slice_local_list,
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
            )

            # -------   make ith attn calc remote slice per chunk  ------ #

            self._make_ith_attn_calc_remote_slice_per_chunk(
                remote_k_ranges_global_per_chunk=remote_k_ranges_global_per_chunk,
                attn_calc_remote_slice_list_per_chunk=attn_calc_remote_slice_list_per_chunk,
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
            )

            # -------   make ith attn calc remote slice for hb domain  ------ #

            self._make_ith_attn_calc_remote_slice(
                remote_k_ranges_global=remote_k_ranges_global_hb_domain,
                attn_calc_remote_slice_list=attn_calc_remote_slice_list_hb_domain,
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
            )

        host_rank_entry_this_rank = HostRankEntry(
            host_q_ranges_global=host_q_ranges_global,
            host_k_ranges_global=host_k_ranges_global,
            attn_calc_slice_global_list=attn_calc_slice_global_list,
            attn_calc_host_slice_local_list=attn_calc_host_slice_local_list,
            remote_k_ranges_global_hb_domain=remote_k_ranges_global_hb_domain,
            attn_calc_remote_slice_list_hb_domain=attn_calc_remote_slice_list_hb_domain,
            remote_k_ranges_global_lb_domain=remote_k_ranges_global_lb_domain,
            remote_k_ranges_global_per_chunk=remote_k_ranges_global_per_chunk,
            attn_calc_remote_slice_list_per_chunk=attn_calc_remote_slice_list_per_chunk,
        )

        # DE-BUG: log host_rank_entry_this_rank
        # from magi_attention.utils import write_rank
        # write_rank(repr(host_rank_entry_this_rank), "host_rank_entry_this_rank.log")

        return host_rank_entry_this_rank

    @nvtx.instrument_nvtx
    def _chunk_remote_k_ranges_global(
        self,
        remote_k_ranges_global: AttnRanges,
    ) -> list[AttnRanges]:
        """Chunk remote k ranges global for multi-stage overlap
        called in 'self._init_host_rank_entry_this_rank'
        """

        # determine the chunk size constrainted by min_chunk_size and max_num_chunks
        total_remote_k_seqlen = remote_k_ranges_global.total_seqlen
        num_chunks = (
            total_remote_k_seqlen + self.overlap_config.min_chunk_size - 1
        ) // self.overlap_config.min_chunk_size
        if num_chunks <= self.overlap_config.max_num_chunks:
            self.overlap_chunk_size = self.overlap_config.min_chunk_size
            self.overlap_num_chunks = num_chunks
        else:
            self.overlap_num_chunks = self.overlap_config.max_num_chunks
            self.overlap_chunk_size = (
                total_remote_k_seqlen + self.overlap_num_chunks - 1
            ) // self.overlap_num_chunks
            self.overlap_num_chunks = (
                total_remote_k_seqlen + self.overlap_chunk_size - 1
            ) // self.overlap_chunk_size

        # chunk the remote k ranges global for multi-stage overlapping
        remote_k_ranges_global_per_chunk: list[
            AttnRanges
        ] = remote_k_ranges_global.chunk(
            self.overlap_chunk_size, check=magi_attention.is_sanity_check_enable()
        )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert all(
                remote_k_ranges_global_ith_chunk.is_merged()
                for remote_k_ranges_global_ith_chunk in remote_k_ranges_global_per_chunk
            ), (
                f"Every remote_k_ranges_global for each chunk should be merged, "
                f"but {remote_k_ranges_global_per_chunk=}."
            )
            assert len(remote_k_ranges_global_per_chunk) == self.overlap_num_chunks, (
                f"{len(remote_k_ranges_global_per_chunk)=} should be equal "
                f"to {self.overlap_num_chunks=} with {self.overlap_chunk_size=}"
            )

        return remote_k_ranges_global_per_chunk

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_host_slice(
        self,
        host_k_ranges_global: AttnRanges,
        attn_calc_host_slice_local_list: list[AttnSlice],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
    ) -> None:
        """Make attn calc host slice local, appended to 'attn_calc_host_slice_local_list'
        called in 'self._init_host_rank_entry_this_rank'
        """

        # fine the overlap part with the global host k ranges
        # i.e. the global attn calc host k ranges
        ith_attn_calc_host_k_ranges_global = (
            ith_attn_calc_k_ranges_global.find_overlap_ranges(
                host_k_ranges_global,
                is_self_merged=True,
                is_other_merged=True,
            )
        )
        # no overlap ranges on host, nothing to do
        if len(ith_attn_calc_host_k_ranges_global) == 0:
            return
        # otherwise, make it local on global host k ranges but do NOT merge for now
        ith_attn_calc_host_k_ranges_local = host_k_ranges_global.make_ranges_local(
            ith_attn_calc_host_k_ranges_global,
            is_self_merged=True,
        )

        # make ith attn calc host slice local
        slice_maker = HostAttnSliceMaker(
            q_range_local=ith_attn_calc_host_q_range_local,
            k_ranges_local=ith_attn_calc_host_k_ranges_local,
            k_ranges_global=ith_attn_calc_host_k_ranges_global,
            calc_k_range_global=ith_attn_calc_k_ranges_global[0],
            mask_type_global=ith_attn_slice_global_mask_type,
        )
        attn_calc_host_slice_local_list.extend(slice_maker.make())

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_remote_slice_per_chunk(
        self,
        remote_k_ranges_global_per_chunk: list[AttnRanges],
        attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
    ) -> None:
        """Make attn calc remote slice for the given remote k ranges global in each chunk,
            and append to 'attn_calc_remote_slice_list_per_chunk'
            called in'self._init_host_rank_entry_this_rank'
        HACK: inplace operation for 'attn_calc_remote_slice_list_per_chunk' for the purpose of performance,
              need further refactor.
        """

        for j, jth_chunk_remote_k_ranges_global in enumerate(
            remote_k_ranges_global_per_chunk
        ):
            self._make_ith_attn_calc_remote_slice(
                remote_k_ranges_global=jth_chunk_remote_k_ranges_global,
                attn_calc_remote_slice_list=attn_calc_remote_slice_list_per_chunk[j],
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
            )

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_remote_slice(
        self,
        remote_k_ranges_global: AttnRanges,
        attn_calc_remote_slice_list: list[MultiKAttnSlice],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
    ) -> None:
        """Make ith attn calc remote slice for the given remote k ranges global,
            and append to given 'attn_calc_remote_slice_list',
            called in 'self._init_host_rank_entry_this_rank' directly for hb domain
            and in 'self._make_ith_attn_calc_remote_slice_per_chunk' for jth chunk of lb domain
        HACK: inplace operation for 'attn_calc_remote_slice_list' for the purpose of performance,
              need further refactor.
        """

        # find the overlap part in the global remote k ranges
        ith_attn_calc_remote_k_ranges_global = (
            ith_attn_calc_k_ranges_global.find_overlap_ranges(
                remote_k_ranges_global,
                is_self_merged=True,
                is_other_merged=True,
            )
        )

        # no overlap with and ith slice
        if len(ith_attn_calc_remote_k_ranges_global) == 0:
            return

        # make ith attn calc remote multik slice
        slice_maker = RemoteAttnSliceMaker(
            q_range_local=ith_attn_calc_host_q_range_local,
            k_ranges_global=ith_attn_calc_remote_k_ranges_global,
            calc_k_range_global=ith_attn_calc_k_ranges_global[0],
            mask_type_global=ith_attn_slice_global_mask_type,
        )
        attn_calc_remote_slice_list.extend(slice_maker.make())  # type: ignore[arg-type]

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_per_stage_this_rank(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[RemoteRankEntry]:
        """Initialize remote rank entry per overlap stage for this rank"""

        # -------   caculate calc/comm cost pairs  ------ #

        chunk_costs = self._calc_cost_pairs_per_chunk(
            host_rank_entry_this_rank=host_rank_entry_this_rank,
        )

        # ------    solve the multi-stage overlap problem   ------ #
        # ------    and get chunk partitions for each stage   ------ #

        cost_partitions = self._solve_multi_stage_overlap(
            chunk_costs=chunk_costs,
        )

        # ------    caculate remote rank entry for each stage   ------ #

        remote_rank_entry_per_stage_this_rank = self._calc_remote_rank_entry_per_stage(
            host_rank_entry_this_rank=host_rank_entry_this_rank,
            cost_partitions=cost_partitions,
        )

        return remote_rank_entry_per_stage_this_rank

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_per_rank_per_stage(
        self, remote_rank_entry_per_stage_this_rank: list[RemoteRankEntry]
    ) -> list[list[RemoteRankEntry]]:
        """Initialize remote rank entry per rank for each overlap stage"""

        # all gather remote rank entry per stage from each rank
        remote_rank_entry_per_stage_per_rank = [None] * self.cp_size

        with nvtx.add_nvtx_event("remote_rank_entry_ag"):
            dist.all_gather_object(
                remote_rank_entry_per_stage_per_rank,
                remote_rank_entry_per_stage_this_rank,
                group=self.cp_group,
            )

        # check shape to be [cp_size, overlap_degree]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(remote_rank_entry_per_stage_per_rank) == self.cp_size
                and len(remote_rank_entry_per_stage_per_rank[0]) == self.overlap_degree  # type: ignore
            ), f"{len(remote_rank_entry_per_stage_per_rank)=}, {self.cp_size=}, {self.overlap_degree=}"

        # transpose to be remote rank entry per rank for each stage
        remote_rank_entry_per_rank_per_stage = transpose_matrix(
            remote_rank_entry_per_stage_per_rank  # type: ignore
        )

        # check shape to be [overlap_degree, cp_size]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(remote_rank_entry_per_rank_per_stage) == self.overlap_degree  # type: ignore
                and len(remote_rank_entry_per_rank_per_stage[0]) == self.cp_size
            ), f"{len(remote_rank_entry_per_rank_per_stage)=}, {self.overlap_degree=}, {self.cp_size=}"

        return remote_rank_entry_per_rank_per_stage

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_this_domain(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> RemoteRankEntry:
        """Initialize remote rank entry for high-bandwidth domain
        Args:
            host_rank_entry_this_rank: HostRankEntry for this rank

        Returns:
            remote_rank_entry_for_hb_domain: RemoteRankEntry for high-bandwidth domain
        """

        slice_tuples = [
            (
                attn_slice.q_range,
                attn_slice.k_ranges,
                attn_slice.mask_types[-1]
                if len(attn_slice.mask_types) > 0
                else AttnMaskType.FULL,  # if empty, this is no use but a placeholder
            )
            for attn_slice in host_rank_entry_this_rank.attn_calc_remote_slice_list_hb_domain
        ]

        return self._make_remote_entry_for_one_stage(
            slice_tuples=slice_tuples,
            remote_k_ranges_global_this_stage=host_rank_entry_this_rank.remote_k_ranges_global_hb_domain,
            host_k_ranges_global_this_rank=host_rank_entry_this_rank.host_k_ranges_global,
        )

    @nvtx.instrument_nvtx
    def _calc_cost_pairs_per_chunk(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[OverlapStageCost]:
        """Calculate the calc/comm cost pairs for each chunk
        called in 'self._init_remote_rank_entry_per_stage_this_rank'
        """
        # 1-1. host comm cost (must be 0. since no comm needs to be waited)
        host_comm_cost = 0.0

        # 1-2. host calc cost
        host_calc_area = host_rank_entry_this_rank.get_host_calc_area()
        host_calc_cost = self.overlap_config.calc_cost_factor * host_calc_area

        # 2-1. remote comm cost for each chunk
        remote_comm_size_per_chunk = [
            host_rank_entry_this_rank.get_remote_comm_size(chunk_idx)
            for chunk_idx in range(self.overlap_num_chunks)
        ]
        remote_comm_cost_per_chunk = [
            self.overlap_config.comm_cost_factor * comm_size
            for comm_size in remote_comm_size_per_chunk
        ]

        # 2-2. remote calc cost for each chunk
        remote_calc_area_per_chunk = [
            host_rank_entry_this_rank.get_remote_calc_area(chunk_idx)
            for chunk_idx in range(self.overlap_num_chunks)
        ]
        remote_calc_cost_per_chunk = [
            self.overlap_config.calc_cost_factor * remote_calc_area
            for remote_calc_area in remote_calc_area_per_chunk
        ]

        # 3-1. construct the stage cost pairs for each chunk
        chunk_costs = [
            OverlapStageCost(
                comm_cost=comm_cost,
                calc_cost=calc_cost,
            )
            for comm_cost, calc_cost in zip(
                [host_comm_cost] + remote_comm_cost_per_chunk,
                [host_calc_cost] + remote_calc_cost_per_chunk,
            )
        ]

        # 3-2. sanity check
        assert (
            len(chunk_costs) == self.overlap_num_chunks + 1
        ), f"{len(chunk_costs)=}, {self.overlap_num_chunks=}"

        return chunk_costs

    @nvtx.instrument_nvtx
    def _solve_multi_stage_overlap(
        self,
        chunk_costs: list[OverlapStageCost],
    ) -> list[list[int]]:
        """Solve the multi-stage overlap problem with the overlap solver
        called in 'self._init_remote_rank_entry_per_stage_this_rank'

        Args:
            chunk_costs: list of OverlapStageCost, each element is the cost pair for one chunk

        Returns:
            cost_partitions: list of list of int, each element is the chunk partition for one stage
        """

        # overlap solver will return the solution with the partitions of the chunk costs,
        # which is a list with length 'overlap_degree',
        # where the ith elem is a chunk idx list that
        # contains the chunk idxs that need be processed together in the ith stage
        # e.g. [[0, 2], [1, 4], [3, 5]] for 6 chunks and 3 overlap degree
        best_solution, solution_dict = self.overlap_solver.solve(
            stage_costs=chunk_costs,
            overlap_degree=self.overlap_config.degree,
            dynamic_max_degree=self.overlap_config.dynamic_max_degree,
        )

        # get the cost partitions of 1 host cost pair and n remote cost pairs
        cost_partitions = best_solution.partitions
        # sanity check
        assert (
            0 in cost_partitions[0]
        ), f"The host cost with index 0 must be in the first partition, but got {cost_partitions=}"

        # get the overlap degree w.r.t the best solution
        best_overlap_degree_this_rank = best_solution.overlap_degree
        if self.overlap_config.mode is AttnOverlapMode.STATIC:
            if magi_attention.is_sanity_check_enable():
                assert best_overlap_degree_this_rank == self.overlap_config.degree, (
                    f"in static mode, {best_overlap_degree_this_rank=} "
                    f"should be equal to {self.overlap_config.degree=}"
                )

            # if static mode, then each rank already has the same overlap degree
            # so there's no need to reduce and the overlap degree is the same as the config
            self.overlap_degree = best_overlap_degree_this_rank
        elif self.overlap_config.mode is AttnOverlapMode.DYNAMIC:
            # if dynamic mode, the final overlap degree is the maximum one among ranks
            # so we need to reduce the best overlap degree among ranks
            with nvtx.add_nvtx_event("dynamic_overlap_degree_ar_max"):
                overlap_degree_reduce_tensor = torch.tensor(
                    best_overlap_degree_this_rank,
                    dtype=torch.int32,
                    device=torch.cuda.current_device(),
                )
                dist.all_reduce(
                    overlap_degree_reduce_tensor,
                    op=dist.ReduceOp.MAX,
                    group=self.cp_group,
                )
                final_overlap_degree = overlap_degree_reduce_tensor.item()

            for _ in range(best_overlap_degree_this_rank, final_overlap_degree):
                # HACK: for the rank with the best overlap degree < final overlap degree
                # we just append the idle stages to the last of the cost partitions
                cost_partitions.append([])

            self.overlap_degree = final_overlap_degree
        else:
            raise ValueError(f"Unknown overlap mode: {self.overlap_config.mode}")

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert (
                len(cost_partitions) == self.overlap_degree
            ), f"{len(cost_partitions)=}, {self.overlap_degree=}"

        # DE-BUG: log vars related to overlap solver I/O
        # from magi_attention.utils import write_rank
        # write_rank(
        #     msg=(
        #         f"{best_overlap_degree_this_rank=} | {self.overlap_degree=} | \n\n"
        #         f"{self.overlap_config=} | \n\n"
        #         f"{len(chunk_costs)=} | {chunk_costs=} | \n\n"
        #         f"{len(cost_partitions)=} | {cost_partitions=} | \n\n"
        #         f"{len(solution_dict)=} | {solution_dict=} | \n\n"
        #     ),
        #     path="overlap_solver_io.log",
        # )

        return cost_partitions

    @nvtx.instrument_nvtx
    def _calc_remote_rank_entry_per_stage(
        self,
        cost_partitions: list[list[int]],
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[RemoteRankEntry]:
        """Calculate the remote rank entry per stage for this rank
        called in 'self._init_remote_rank_entry_per_stage_this_rank'
        """

        remote_rank_entry_per_stage_this_rank: list[RemoteRankEntry] = []

        for cost_partition in cost_partitions:
            remote_rank_entry_per_stage_this_rank.append(
                self._calc_remote_rank_entry_for_one_stage(
                    cost_partiton=cost_partition,
                    host_rank_entry_this_rank=host_rank_entry_this_rank,
                )
            )

        # DE-BUG: log remote_rank_entry_per_stage_this_rank
        # from magi_attention.utils import write_rank
        # write_rank(
        #     repr(remote_rank_entry_per_stage_this_rank),
        #     "remote_rank_entry_per_stage_this_rank.log",
        # )

        return remote_rank_entry_per_stage_this_rank

    @nvtx.instrument_nvtx
    def _calc_remote_rank_entry_for_one_stage(
        self, cost_partiton: list[int], host_rank_entry_this_rank: HostRankEntry
    ) -> RemoteRankEntry:
        """Calculate the remote rank entry for one stage for this rank
        called in'self._calc_remote_rank_entry_per_stage'
        """

        # ------    merge the chunks into one overlap stage within each partition  ------ #
        # ------    and construct the remote rank entry for each stage   ------ #

        # init the args and some temp vars for remote rank entry
        remote_k_ranges_global_this_stage = AttnRanges()
        attn_calc_remote_slice_list_per_chunk_this_stage: list[
            list[MultiKAttnSlice]
        ] = []
        total_q_ranges_local_this_stage = AttnRanges()

        # find the remote_ k_ranges_global and remote_slice_local_list
        # within the chunks for this stage
        for cost_idx in cost_partiton:
            if cost_idx == 0:  # ignore the host cost
                continue
            chunk_idx = cost_idx - 1
            remote_k_ranges_global_this_stage.extend(
                host_rank_entry_this_rank.remote_k_ranges_global_per_chunk[chunk_idx]
            )
            attn_calc_remote_slice_list_per_chunk_this_stage.append(
                host_rank_entry_this_rank.attn_calc_remote_slice_list_per_chunk[
                    chunk_idx
                ]
            )
        remote_k_ranges_global_this_stage = remote_k_ranges_global_this_stage.merge()

        # add all q_ranges to total_q_ranges_this_bucket_local
        for attn_slice in chain(*attn_calc_remote_slice_list_per_chunk_this_stage):
            total_q_ranges_local_this_stage.append(attn_slice.q_range)
        total_q_ranges_boundary_local_this_stage = (
            total_q_ranges_local_this_stage.points
        )

        # get two dict as (key: q_range, value: k_ranges) and (key: q_range, value: masktype)
        (
            map_slice_q_range_to_k_ranges,
            map_slice_q_range_to_masktype,
        ) = self._calc_remote_rank_q_range_to_k_ranges_map(
            attn_calc_remote_slice_list_per_chunk_this_stage=attn_calc_remote_slice_list_per_chunk_this_stage,
            total_q_ranges_boundary_local_this_stage=total_q_ranges_boundary_local_this_stage,
        )

        # construct the attn_calc_remote_slice_local_list_this_stage
        q_range_k_ranges_tuples: list[tuple[AttnRange, AttnRanges]] = sorted(
            map_slice_q_range_to_k_ranges.items(),
            key=lambda t: t[0].start,  # sort by q_range.start
        )
        slice_tuples: list[tuple[AttnRange, AttnRanges, AttnMaskType]] = [
            (q_range, k_ranges, map_slice_q_range_to_masktype[q_range])
            for q_range, k_ranges in q_range_k_ranges_tuples
        ]

        return self._make_remote_entry_for_one_stage(
            slice_tuples=slice_tuples,
            remote_k_ranges_global_this_stage=remote_k_ranges_global_this_stage,
            host_k_ranges_global_this_rank=host_rank_entry_this_rank.host_k_ranges_global,
        )

    @nvtx.instrument_nvtx
    def _make_remote_entry_for_one_stage(
        self,
        slice_tuples: list[tuple[AttnRange, AttnRanges, AttnMaskType]],
        remote_k_ranges_global_this_stage: AttnRanges,
        host_k_ranges_global_this_rank: AttnRanges,
    ) -> RemoteRankEntry:
        """Make the remote entry for one stage
        called in 'self._calc_remote_rank_entry_for_one_stage' for each stage in lb domain
        and in 'self._init_remote_rank_entry_this_domain' for this hb domain

        Args:
            slice_tuples: each tuple contains a q_range and it's corresponding k_ranges and mask type
            remote_k_ranges_global_this_stage: the remote k ranges for this stage
            host_k_ranges_global_this_rank: the host k ranges for this rank

        Returns:
            attn_calc_remote_slice_local_list_this_stage: the remote entry for this stage
        """
        attn_calc_remote_slice_local_list_this_stage: list[AttnSlice] = []

        for q_range, k_ranges, mask_type in slice_tuples:
            k_ranges = remote_k_ranges_global_this_stage.make_ranges_local(
                k_ranges,
                is_self_merged=True,
            ).merge()

            for k_range in k_ranges:
                attn_calc_remote_slice_local_list_this_stage.append(
                    AttnSlice(
                        q_range=q_range,
                        k_range=k_range,
                        mask_type=mask_type,
                    )
                )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert remote_k_ranges_global_this_stage.is_merged()

        return RemoteRankEntry(
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global=remote_k_ranges_global_this_stage,
            attn_calc_remote_slice_local_list=attn_calc_remote_slice_local_list_this_stage,
        )

    @nvtx.instrument_nvtx
    def _calc_remote_rank_q_range_to_k_ranges_map(
        self,
        attn_calc_remote_slice_list_per_chunk_this_stage: list[list[MultiKAttnSlice]],
        total_q_ranges_boundary_local_this_stage: list[int],
    ) -> tuple[
        defaultdict[AttnRange, AttnRanges], defaultdict[AttnRange, AttnMaskType]
    ]:
        """Split the slice according to the boundary, and form maps from q_range to k_ranges and masktype.
        called in 'self._calc_remote_rank_entry_for_one_stage'
        """

        # init q_range->k_ranges map and q_range->masktype map
        map_slice_q_range_to_k_ranges: defaultdict[AttnRange, AttnRanges] = defaultdict(
            AttnRanges
        )
        map_slice_q_range_to_masktype: defaultdict[
            AttnRange, AttnMaskType
        ] = defaultdict(lambda: AttnMaskType.FULL)

        for slice in chain(*attn_calc_remote_slice_list_per_chunk_this_stage):
            # find the start and end index in the boundary list
            slice_q_range_start, slice_q_range_end = (
                slice.q_range.start,
                slice.q_range.end,
            )
            boundary_left_index = bisect_left(
                total_q_ranges_boundary_local_this_stage, slice_q_range_start
            )
            boundary_right_index = bisect_left(
                total_q_ranges_boundary_local_this_stage, slice_q_range_end
            )

            # traverse from computed boundary start to end
            for boundary_idx in range(boundary_left_index, boundary_right_index):
                boundary_start, boundary_end = (
                    total_q_ranges_boundary_local_this_stage[boundary_idx],
                    total_q_ranges_boundary_local_this_stage[boundary_idx + 1],
                )

                # create the segmented q_range.
                q_range_this_slice = AttnRange(start=boundary_start, end=boundary_end)

                if slice.mask_types[-1] == AttnMaskType.FULL:
                    # in the case of full, no need to handle k_ranges.
                    map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                        slice.k_ranges
                    )
                elif slice.mask_types[-1] == AttnMaskType.CAUSAL:
                    # in the case of causal, the end of the last range in k_ranges may need to be shortened
                    if magi_attention.is_sanity_check_enable():
                        assert (
                            slice_q_range_end == boundary_end
                        ), "slice_q_range_end should be always equal to boundary_end"

                    map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                        slice.k_ranges
                    )
                    map_slice_q_range_to_masktype[
                        q_range_this_slice
                    ] = AttnMaskType.CAUSAL
                else:
                    raise ValueError(
                        f"Only support 'full' and 'causal' mask, "
                        f"but get {slice.mask_types[-1]}"
                    )

        return (
            map_slice_q_range_to_k_ranges,
            map_slice_q_range_to_masktype,
        )

    @nvtx.instrument_nvtx
    def _init_transfer_table_per_stage(
        self,
        remote_rank_entry_per_rank_per_stage: list[list[RemoteRankEntry]],
    ) -> list[TransferTable]:
        """Initialize transfer table per stage for this rank"""

        transfer_table_per_stage: list[TransferTable] = []

        transfer_info_per_stage_this_rank: list[TransferInfo] = [
            self._init_transfer_info_this_rank_for_one_stage(
                remote_rank_entry_per_rank_this_stage
            )
            for remote_rank_entry_per_rank_this_stage in (
                remote_rank_entry_per_rank_per_stage
            )
        ]

        transfer_info_per_rank_per_stage = self._init_transfer_info_per_rank_per_stage(
            transfer_info_per_stage_this_rank
        )

        transfer_table_per_stage = [
            self._init_transfer_table_for_one_stage(
                remote_rank_entry_per_rank_this_stage,
                transfer_info_per_rank_this_stage,
            )
            for (
                remote_rank_entry_per_rank_this_stage,
                transfer_info_per_rank_this_stage,
            ) in zip(
                remote_rank_entry_per_rank_per_stage,
                transfer_info_per_rank_per_stage,
            )
        ]

        return transfer_table_per_stage

    @nvtx.instrument_nvtx
    def _init_transfer_table_for_one_stage(
        self,
        remote_rank_entry_per_rank: list[RemoteRankEntry],
        transfer_info_per_rank: list[TransferInfo],
    ) -> TransferTable:
        """Initialize transfer table for each overlap stage
        called in 'self._init_transfer_table_per_stage'
        """

        # init transfer table entry for each rank pair: (send_ranki, recv_rankj)
        transfer_table = TransferTable(cp_size=self.cp_size)

        # fill up transfer table
        for send_rank in range(self.cp_size):  # for each send_ranki
            transfer_info = transfer_info_per_rank[send_rank]
            group_cast_ranges_global_transfer = (
                transfer_info.group_cast_ranges_global_transfer
            )
            group_cast_ranges_local_send_to = (
                transfer_info.group_cast_ranges_local_send_to
            )

            # for each non-overlapped global/local k range that send_ranki needs to send to
            # we tranverse each dest recv_rankj to recv it in the set,
            # and append it to k_ranges_local_in_send_buf at the (send_ranki, recv_rankj) table entry
            for r in group_cast_ranges_global_transfer:
                k_range = AttnRange(start=r.start, end=r.end)
                if send_rank == self.cp_rank:  # the send row for this rank
                    for recv_rank in r.rank_set:
                        transfer_table.append_k_ranges_global(
                            send_rank=self.cp_rank,
                            recv_rank=recv_rank,
                            k_range=k_range,
                        )
                elif self.cp_rank in r.rank_set:  # the recv col for this rank
                    transfer_table.append_k_ranges_global(
                        send_rank=send_rank,
                        recv_rank=self.cp_rank,
                        k_range=k_range,
                    )

            for r in group_cast_ranges_local_send_to:
                k_range = AttnRange(start=r.start, end=r.end)
                if send_rank == self.cp_rank:  # the send row for this rank
                    for recv_rank in r.rank_set:
                        transfer_table.append_k_ranges_local_in_send_buf(
                            send_rank=self.cp_rank,
                            recv_rank=recv_rank,
                            k_range=k_range,
                        )
                elif self.cp_rank in r.rank_set:  # the recv col for this rank
                    transfer_table.append_k_ranges_local_in_send_buf(
                        send_rank=send_rank,
                        recv_rank=self.cp_rank,
                        k_range=k_range,
                    )

            # sort the k ranges in each table entry
            if send_rank == self.cp_rank:  # the send row for this rank
                for recv_rank in range(self.cp_size):
                    transfer_table.sort_k_ranges_global(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                    )
                    transfer_table.sort_k_ranges_local_in_send_buf(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                    )
                    # fill k_ranges_local_in_recv_buf
                    transfer_table.make_k_ranges_local_in_recv_buf(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                        remote_k_ranges_global_for_recv_rank=remote_rank_entry_per_rank[
                            recv_rank
                        ].remote_k_ranges_global,
                    )
            else:  # the recv col for this rank
                transfer_table.sort_k_ranges_global(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                transfer_table.sort_k_ranges_local_in_send_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                # fill k_ranges_local_in_recv_buf
                transfer_table.make_k_ranges_local_in_recv_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                    remote_k_ranges_global_for_recv_rank=remote_rank_entry_per_rank[
                        self.cp_rank
                    ].remote_k_ranges_global,
                )

        return transfer_table

    @nvtx.instrument_nvtx
    def _init_transfer_info_this_rank_for_one_stage(
        self,
        remote_rank_entry_per_rank: list[RemoteRankEntry],
    ) -> TransferInfo:
        """Initialize transfer info for this rank for certain stage
        called in 'self._init_transfer_table_per_stage'
        """

        host_k_ranges_global_this_rank = remote_rank_entry_per_rank[
            self.cp_rank
        ].host_k_ranges_global
        remote_k_ranges_global_this_rank = remote_rank_entry_per_rank[
            self.cp_rank
        ].remote_k_ranges_global

        # init k ranges global/local for send_to/recv_from per rank
        k_ranges_global_recv_from_per_rank: list[AttnRanges] = []
        k_ranges_local_recv_from_per_rank: list[AttnRanges] = []
        k_ranges_global_send_to_per_rank: list[AttnRanges] = []
        k_ranges_local_send_to_per_rank: list[AttnRanges] = []
        for rank in range(self.cp_size):
            if rank == self.cp_rank:  # no need to recv from / send to this rank
                k_ranges_global_recv_from_per_rank.append(AttnRanges())
                k_ranges_local_recv_from_per_rank.append(AttnRanges())
                k_ranges_global_send_to_per_rank.append(AttnRanges())
                k_ranges_local_send_to_per_rank.append(AttnRanges())
                continue

            # ----------    for k_ranges recv from     ---------- #

            # get the global k ranges that this rank needs to recv from current rank
            rank_host_k_ranges_global = remote_rank_entry_per_rank[
                rank
            ].host_k_ranges_global
            k_ranges_global_recv_from_rank = (
                remote_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_host_k_ranges_global,
                    is_self_merged=True,
                    is_other_merged=True,
                )
            )
            # make the global k ranges local w.r.t. self's recv buffer
            k_ranges_local_recv_from_rank = (
                remote_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_recv_from_rank,
                    is_self_merged=True,
                )
            )
            # add to recv transfer info for both global and local ones
            k_ranges_global_recv_from_per_rank.append(k_ranges_global_recv_from_rank)
            k_ranges_local_recv_from_per_rank.append(k_ranges_local_recv_from_rank)

            # ----------    for k_ranges send to     ---------- #

            # get the global k ranges that this rank needs to send to current rank
            rank_remote_k_ranges_global = remote_rank_entry_per_rank[
                rank
            ].remote_k_ranges_global
            k_ranges_global_send_to_rank = (
                host_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_remote_k_ranges_global,
                    is_self_merged=True,
                    is_other_merged=True,
                )
            )
            # make the global k ranges local w.r.t. self's send buffer
            k_ranges_local_send_to_rank = (
                host_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_send_to_rank,
                    is_self_merged=True,
                )
            )
            # add to send transfer info for both global and local ones
            k_ranges_global_send_to_per_rank.append(k_ranges_global_send_to_rank)
            k_ranges_local_send_to_per_rank.append(k_ranges_local_send_to_rank)

        # init group_cast_ranges for global/local k ranges that send_ranki needs to send to
        # which splits the local ranges into non-overlapped local ranges
        group_cast_ranges_global_transfer = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=k_ranges_global_send_to_per_rank,
        )
        group_cast_ranges_local_send_to = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=k_ranges_local_send_to_per_rank,
        )

        transfer_info_this_rank = TransferInfo(
            k_ranges_global_recv_from_per_rank=k_ranges_global_recv_from_per_rank,
            k_ranges_local_recv_from_per_rank=k_ranges_local_recv_from_per_rank,
            k_ranges_global_send_to_per_rank=k_ranges_global_send_to_per_rank,
            k_ranges_local_send_to_per_rank=k_ranges_local_send_to_per_rank,
            group_cast_ranges_global_transfer=group_cast_ranges_global_transfer,
            group_cast_ranges_local_send_to=group_cast_ranges_local_send_to,
        )

        return transfer_info_this_rank

    @nvtx.instrument_nvtx
    def _init_transfer_info_per_rank_per_stage(
        self,
        transfer_info_per_stage_this_rank: list[TransferInfo],
    ) -> list[list[TransferInfo]]:
        """Initialize transfer info per rank for each stage
        called in 'self._init_transfer_table_per_stage'
        """

        # all gather transfer info per stage from each rank
        transfer_info_per_stage_per_rank = [None] * self.cp_size
        with nvtx.add_nvtx_event("transfer_info_ag"):
            dist.all_gather_object(
                transfer_info_per_stage_per_rank,
                transfer_info_per_stage_this_rank,
                group=self.cp_group,
            )

        # check shape to be [cp_size, overlap_degree]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(transfer_info_per_stage_per_rank) == self.cp_size
                and len(transfer_info_per_stage_per_rank[0]) == self.overlap_degree  # type: ignore
            )

        # transpose to be transfer info per rank for each stage
        transfer_info_per_rank_per_stage = transpose_matrix(
            transfer_info_per_stage_per_rank  # type: ignore
        )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            # for each stage:
            #   for each rank pair (i≠j): (send_ranki, recv_rankj)
            #       whether the global k ranges that send_ranki needs to send to recv_rankj
            #       are equal to the ones that recv_rankj needs to recv from send_ranki
            for stage, transfer_info_per_rank in enumerate(
                transfer_info_per_rank_per_stage
            ):
                for send_rank in range(self.cp_size):
                    for recv_rank in range(self.cp_size):
                        if send_rank == recv_rank:
                            continue

                        send_info: TransferInfo = transfer_info_per_rank[send_rank]
                        recv_info: TransferInfo = transfer_info_per_rank[recv_rank]
                        k_ranges_global_recv_from_send_rank = (
                            recv_info.k_ranges_global_recv_from_per_rank[send_rank]
                        )
                        k_ranges_global_send_to_recv_rank = (
                            send_info.k_ranges_global_send_to_per_rank[recv_rank]
                        )

                        assert (
                            k_ranges_global_recv_from_send_rank
                            == k_ranges_global_send_to_recv_rank
                        ), (
                            f"The sanity check for transfer table at {stage=} failed:\n"
                            f"For rank pair ({send_rank=} {recv_rank=}), we got:\n"
                            f"{k_ranges_global_recv_from_send_rank=}\n"
                            f"{k_ranges_global_send_to_recv_rank=}"
                        )

        return transfer_info_per_rank_per_stage

    @nvtx.instrument_nvtx
    def calc_comm_meta(self) -> CommMeta:
        """Calculate communication meta for kv group collective"""

        num_remote_tokens_list: list[int] = []
        group_collective_args_list: list[GroupCollectiveArg] = []

        for transfer_table_this_stage, remote_rank_entry_per_rank_this_stage in zip(
            self.transfer_table_per_stage,
            self.remote_rank_entry_per_rank_per_stage,
        ):
            total_seqlen_host_k = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].host_k_ranges_global.total_seqlen

            num_remote_tokens = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].remote_k_ranges_global.total_seqlen

            group_collective_arg = self._calc_group_collective_arg(
                transfer_table_this_stage,
                total_seqlen_host_k,
            )

            num_remote_tokens_list.append(num_remote_tokens)
            group_collective_args_list.append(group_collective_arg)

        # build comm meta
        comm_meta = CommMeta(
            num_remote_tokens_per_stage=num_remote_tokens_list,
            group_collective_args_list=group_collective_args_list,
        )

        return comm_meta

    @nvtx.instrument_nvtx
    def _calc_group_collective_arg(
        self,
        transfer_table: TransferTable,
        total_seqlen_host_k: int,
    ) -> GroupCollectiveArg:
        """Calculate group collective args from one transfer table
        called in 'self.calc_comm_meta'
        """
        # retrieve group cast ranges for local k ranges that this rank needs to send to
        # which splits the local ranges into non-overlapped local ranges
        group_cast_ranges_local_send_to = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=[
                transfer_table.get_k_ranges_local_in_send_buf(
                    send_rank=self.cp_rank,
                    recv_rank=recv_rank,
                )
                for recv_rank in range(self.cp_size)
            ],
        )

        # calc input split size list with dst indices list
        input_split_size_list: list[int] = []
        dst_indices_list: list[list[int]] = []

        last_end = 0
        for r in group_cast_ranges_local_send_to:
            if r.start != last_end:  # [last_end, r.start) has no dest rank
                # FIXME: this branch is unreachable in the current test cases
                input_split_size_list.append(r.start - last_end)
                dst_indices_list.append([])

            input_split_size_list.append(r.seqlen)
            dst_indices_list.append(list(r.rank_set))
            last_end = r.end

        if last_end != total_seqlen_host_k:  # [last_end, seqlen) has no dest rank
            input_split_size_list.append(total_seqlen_host_k - last_end)
            dst_indices_list.append([])

        # retrieve group cast ranges for local k ranges that this rank needs to recv from
        group_cast_ranges_local_recv_from = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=[
                transfer_table.get_k_ranges_local_in_recv_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                for send_rank in range(self.cp_size)
            ],
            # NOTE: no need to split group cast ranges for recv
            split=False,
        )

        # calc output split size list with src index list
        output_split_size_list = []
        src_index_list = []

        if magi_attention.is_sanity_check_enable():
            # NOTE: as for group cast semantics,
            # there's only one src rank that sends the corr. data into
            # each non-overlapped range in recv buffer
            for r in group_cast_ranges_local_recv_from:
                assert len(r.rank_set) == 1

        for r in group_cast_ranges_local_recv_from:
            output_split_size_list.append(r.seqlen)
            src_index_list.append(r.rank_set.pop())

        # build group collective arg
        group_collective_arg = GroupCollectiveArg(
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            world_size=self.cp_size,
            intra_group=self.cp_intra_group,
            inter_group=self.cp_inter_group,
        )

        return group_collective_arg

    @nvtx.instrument_nvtx
    def calc_attn_calc_meta(self) -> AttnCalcMeta:
        """Calculate flex-flash-attention calculation meta"""

        if magi_attention.is_sanity_check_enable():
            # check local attn calc
            assert all(
                attn_slice is not None
                for attn_slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
            )

            # check remote attn calc for each overlap stage
            for (
                remote_rank_entry_this_stage_this_rank
            ) in self.remote_rank_entry_per_stage_this_rank:
                assert all(
                    attn_slice is not None
                    for attn_slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
                )

        # ---   build local attn args   --- #

        host_slice_local_list = (
            self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )
        local_attn_arg = AttnArg(
            q_ranges=AttnRanges.from_ranges(
                [attn_slice.q_range for attn_slice in host_slice_local_list]  # type: ignore[arg-type]
            ),
            k_ranges=AttnRanges.from_ranges(
                [attn_slice.k_range for attn_slice in host_slice_local_list]  # type: ignore[arg-type]
            ),
            is_causal_mapping=[
                attn_slice.mask_type == AttnMaskType.CAUSAL
                for attn_slice in host_slice_local_list
            ],
            shard_seqlen_q=self.shard_seqlen_q,
            total_area=sum(attn_slice.area for attn_slice in host_slice_local_list),
        )

        # ---   build remote attn args for each overlap stage   --- #

        remote_attn_args_list = []
        for (
            remote_rank_entry_this_stage_this_rank
        ) in self.remote_rank_entry_per_stage_this_rank:
            remote_slice_local_list = (
                remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
            )
            remote_attn_args_list.append(
                AttnArg(
                    q_ranges=AttnRanges.from_ranges(
                        [attn_slice.q_range for attn_slice in remote_slice_local_list]  # type: ignore[arg-type]
                    ),
                    k_ranges=AttnRanges.from_ranges(
                        [attn_slice.k_range for attn_slice in remote_slice_local_list]  # type: ignore[arg-type]
                    ),
                    is_causal_mapping=[
                        attn_slice.mask_type == AttnMaskType.CAUSAL
                        for attn_slice in remote_slice_local_list
                    ],
                    shard_seqlen_q=self.shard_seqlen_q,
                    total_area=sum(
                        attn_slice.area for attn_slice in remote_slice_local_list
                    ),
                )
            )

        # ---   build attn calc meta   --- #

        attn_calc_meta = AttnCalcMeta(
            local_attn_arg=local_attn_arg,
            remote_attn_args_list=remote_attn_args_list,
        )

        return attn_calc_meta

    def __repr__(self, title_len: int = 50) -> str:
        repr_contents = []

        repr_summary = self._repr_host_info(
            self.host_rank_entry_this_rank, title_len=title_len
        )
        repr_contents.append(repr_summary)

        for stage, (
            transfer_table_this_stage,
            remote_rank_entry_per_rank_this_stage,
        ) in enumerate(
            zip(
                self.transfer_table_per_stage,
                self.remote_rank_entry_per_rank_per_stage,
            )
        ):
            repr_this_stage = self._repr_remote_info_for_one_stage(
                stage,
                transfer_table_this_stage,
                remote_rank_entry_per_rank_this_stage,
                title_len=title_len,
            )

            repr_contents.append(repr_this_stage)

        # 末尾换行
        repr_contents.append("\n\n")

        return "\n\n".join(repr_contents)

    def _repr_host_info(
        self, host_rank_entry_this_rank: HostRankEntry, title_len: int = 50
    ) -> str:  # pragma: no cover
        repr_info = []

        # add summary info title
        stage_title = "  Host Info  "
        repr_info.append("\n" + "=" * title_len + stage_title + "=" * title_len + "\n")

        host_q_ranges_global = host_rank_entry_this_rank.host_q_ranges_global
        host_k_ranges_global = host_rank_entry_this_rank.host_k_ranges_global
        remote_k_ranges_global_hb_domain = (
            host_rank_entry_this_rank.remote_k_ranges_global_hb_domain
        )
        remote_k_ranges_global_lb_domain = (
            host_rank_entry_this_rank.remote_k_ranges_global_lb_domain
        )
        attn_calc_slice_global_list = (
            host_rank_entry_this_rank.attn_calc_slice_global_list
        )
        attn_calc_host_slice_local_list = (
            host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )

        repr_info.append(f"host_q_ranges_global: {host_q_ranges_global}")
        repr_info.append(f"host_k_ranges_global: {host_k_ranges_global}")
        repr_info.append(
            f"remote_k_ranges_global_hb_domain: {remote_k_ranges_global_hb_domain}"
        )
        repr_info.append(
            f"remote_k_ranges_global_lb_domain: {remote_k_ranges_global_lb_domain}"
        )
        repr_info.append(f"attn_calc_slice_global_list: {attn_calc_slice_global_list}")
        repr_info.append(
            f"attn_calc_host_slice_local_list: {attn_calc_host_slice_local_list}"
        )

        return "\n".join(repr_info)

    def _repr_remote_info_for_one_stage(
        self,
        stage: int,
        transfer_table_this_stage: TransferTable,
        remote_rank_entry_per_rank_this_stage: list[RemoteRankEntry],
        title_len: int = 50,
    ) -> str:  # pragma: no cover
        # 计算每个单元格需要的最大宽度
        cell_widths = [[0] * self.cp_size for _ in range(self.cp_size)]
        for send_rank in range(self.cp_size):
            for recv_rank in range(self.cp_size):
                send_str = f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}"
                recv_str = f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}"
                global_str = f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}"

                width = max(len(send_str), len(recv_str), len(global_str))
                cell_widths[send_rank][recv_rank] = width

        # 计算每列的最大宽度
        col_widths = [
            max(
                max(cell_widths[row][col] for row in range(self.cp_size)),
                len(
                    "host_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].host_k_ranges_global}"
                ),
                len(
                    "remote_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].remote_k_ranges_global}"
                ),
                len(
                    "attn_calc_remote_slice_local_list: "
                    f"{remote_rank_entry_per_rank_this_stage[col].attn_calc_remote_slice_local_list}"
                ),
            )
            for col in range(self.cp_size)
        ]

        # 计算表格的总宽度（考虑到每列分隔符 " | " 以及每行的"row xx |"前缀）
        table_width = (
            sum(col_widths) + 4 * (self.cp_size - 1) + 7
        )  # 每列间隔宽度4 + row xx | 前缀宽度为7

        # 构建表格
        repr_info_this_stage = []

        # 添加overlap stage title分割线
        stage_title = f"  Remote Info for Stage {stage}  "
        repr_info_this_stage.append(
            "\n" + "=" * title_len + stage_title + "=" * title_len + "\n"
        )

        # 添加列标题行（扩展为5行高度）
        repr_info_this_stage.append("\n" + "-" * table_width)

        # 第一行：列号
        header_cells = [f"col{j:2d}".center(col_widths[j]) for j in range(self.cp_size)]
        repr_info_this_stage.append("r/c   | " + " | ".join(header_cells) + " |")

        # 第二行：host_k_ranges_global
        host_cells = [
            f"host_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].host_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(host_cells) + " |")

        # 第三行：remote_k_ranges_global
        remote_cells = [
            f"remote_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].remote_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_cells) + " |")

        # 第四行：attn_calc_remote_slice_local_list
        remote_slice_cells = [
            "attn_calc_remote_slice_local_list: "
            f"{remote_rank_entry_per_rank_this_stage[j].attn_calc_remote_slice_local_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_slice_cells) + " |")

        # 添加分割线
        repr_info_this_stage.append("-" * table_width)

        # 添加每一行
        for send_rank in range(self.cp_size):
            # 处理每个单元格的三行内容
            cell_lines = []
            for recv_rank in range(self.cp_size):
                col_width = col_widths[recv_rank]
                cell_content = [
                    f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                ]
                cell_lines.append(cell_content)

            # 组装每行的三行内容
            for line_idx in range(3):
                prefix = f"row{send_rank:2d} |" if line_idx == 0 else "      |"
                line = [cell_lines[j][line_idx] for j in range(self.cp_size)]
                repr_info_this_stage.append(f"{prefix} " + " | ".join(line) + " |")

            repr_info_this_stage.append("-" * table_width)  # 每行后面添加分割线

        return "\n".join(repr_info_this_stage)
