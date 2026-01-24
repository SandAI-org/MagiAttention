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

from enum import Enum

import magi_attention
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice
from magi_attention.utils import nvtx


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
        if self.mask_type_global == AttnMaskType.CAUSAL:
            self._init_causal()
        elif self.mask_type_global == AttnMaskType.INVCAUSAL:
            self._init_inv_causal()
        elif self.mask_type_global == AttnMaskType.BICAUSAL:
            self._init_bi_causal()

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

        self.diff_len_of_k_range_minus_q_range = max(
            0,
            self.calc_k_range_global.seqlen - self.q_range_local.seqlen,
        )

        # ---- determine the causal case key ---- #

        self._init_causal_case_key()

    def _init_inv_causal(self) -> None:
        self.first_k_range_global = self.k_ranges_global[0]

        # ---- calc the start and end of the inv_causal area ---- #

        if self.calc_k_range_global.seqlen > self.q_range_local.seqlen:
            # the inv causal mask of a trapezoid
            self.inv_causal_mask_end = (
                self.calc_k_range_global.start + self.q_range_local.seqlen
            )
        else:
            # the inv causal mask of a triangle or a null slice
            self.inv_causal_mask_end = self.calc_k_range_global.end

        self.inv_causal_mask_start = self.calc_k_range_global.start

        self.start_distance_to_inv_causal_end = (
            self.inv_causal_mask_end - self.first_k_range_global.start
        )

        # when q_range.seqlen > k_range.seqlen, exceed_causal_end is
        # just a part of the actual exceeded length
        self.end_distance_to_inv_causal_end = (
            self.inv_causal_mask_end - self.first_k_range_global.end
        )
        # it needs to add the difference between the lengths of q_range and k_range in slice
        self.diff_len_of_q_range_minus_k_range = max(
            0,
            self.q_range_local.seqlen - self.calc_k_range_global.seqlen,
        )

        # ---- determine the inv causal case key ---- #

        self._init_inv_causal_key()

    def _init_bi_causal(self) -> None:
        # ---- calc the causal start and inv causal end of the bi_causal area ---- #

        self.diff_len_of_q_range_minus_k_range = max(
            0,
            self.q_range_local.seqlen - self.calc_k_range_global.seqlen,
        )

        self.diff_len_of_k_range_minus_q_range = max(
            0,
            self.calc_k_range_global.seqlen - self.q_range_local.seqlen,
        )

        self.inv_causal_end = self.calc_k_range_global.start + self.q_range_local.seqlen
        self.causal_start = self.calc_k_range_global.end - self.q_range_local.seqlen

        self.left_point = min(self.inv_causal_end, self.causal_start)
        self.right_point = max(self.inv_causal_end, self.causal_start)

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

    def _init_inv_causal_key(self) -> None:
        self.inv_causal_case_key = self.CausalCaseKey.INVALID
        if (
            self.first_k_range_global.start >= self.inv_causal_mask_end
            and self.first_k_range_global.end >= self.inv_causal_mask_end
        ):
            # case1: the area will be formed as a full rectangle mask
            self.inv_causal_case_key = self.CausalCaseKey.RECTANGLE
        elif (
            self.first_k_range_global.start == self.inv_causal_mask_start
            and self.first_k_range_global.end >= self.inv_causal_mask_end
        ):
            # case2: the area will be formed as a normal causal mask,
            # i.e. an uncut triangle or a trapezoid
            self.inv_causal_case_key = self.CausalCaseKey.TRAPEZOID
        elif (
            self.first_k_range_global.start == self.inv_causal_mask_start
            and self.first_k_range_global.end < self.inv_causal_mask_end
        ):
            self.inv_causal_case_key = self.CausalCaseKey.TRIANGLE
        elif (
            self.first_k_range_global.end >= self.inv_causal_mask_end
            and self.inv_causal_mask_start
            < self.first_k_range_global.start
            < self.inv_causal_mask_end
        ):
            # this includes two cases:
            # case 4: the area of a rotated trapezoid or a pentagon,
            #   when q_range.seqlen <= k_range.seqlen in the slice
            # case 5: the area of a cut rotated trapezoid,
            #   when q_range.seqlen > k_range.seqlen in the slice
            self.inv_causal_case_key = self.CausalCaseKey.PENTAGON

    @nvtx.instrument_nvtx
    def make(self) -> list[AttnSlice]:
        match self.mask_type_global:
            case AttnMaskType.FULL:
                attn_slices = self._make_slice_for_full_mask()
            case AttnMaskType.CAUSAL:
                attn_slices = self._make_slice_for_causal_mask()
            case AttnMaskType.INVCAUSAL:
                attn_slices = self._make_slice_for_inv_causal_mask()
            case AttnMaskType.BICAUSAL:
                attn_slices = self._make_slice_for_bi_causal_mask()
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

    def _make_slice_for_inv_causal_mask(self) -> list[AttnSlice]:
        """For inv causal mask, there're more than one cases to be considered"""

        match self.inv_causal_case_key:
            case self.CausalCaseKey.RECTANGLE:
                attn_slices = self._make_slice_for_inv_causal_rectangle_mask()
            case self.CausalCaseKey.TRAPEZOID:
                attn_slices = self._make_slice_for_inv_causal_trapezoid_mask()
            case self.CausalCaseKey.TRIANGLE:
                attn_slices = self._make_slice_for_inv_causal_triangle_mask()
            case self.CausalCaseKey.PENTAGON:
                attn_slices = self._make_slice_for_inv_causal_pentagon_mask()
            case self.CausalCaseKey.INVALID:
                raise ValueError(
                    f"Got invalid range {self.first_k_range_global=} "
                    f"when {self.inv_causal_mask_start=} and {self.inv_causal_mask_end=}."
                )
            case _:
                raise ValueError(
                    f"Got invalid causal case key {self.inv_causal_case_key=}."
                )

        return attn_slices

    def _make_slice_for_bi_causal_mask(self) -> list[AttnSlice]:
        """For bi causal mask, there're more than one cases to be considered"""

        total_attn_slices: list[AttnSlice] = []

        for index in range(len(self.k_ranges_global)):
            attn_slices = self._make_slice_for_bi_causal(
                index=index,
            )

            total_attn_slices.extend(attn_slices)

        return total_attn_slices

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
            start=self.q_range_local.start + self.diff_len_of_q_range_minus_k_range,
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

    def _make_slice_for_inv_causal_rectangle_mask(self) -> list[AttnSlice]:
        """in such case, we just call the maker for full mask,
        since a causal rectangle mask equals to a full mask
        """

        return self._make_slice_for_full_mask()

    def _make_slice_for_inv_causal_trapezoid_mask(self) -> list[AttnSlice]:
        """in such case, the whole mask will be a single
        normal inv causal mask after merged
        """

        k_range_local = self._merge_k_ranges_and_check(
            self.k_ranges_local,
            allow_empty=False,
        )

        return [
            AttnSlice(
                q_range=self.q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.INVCAUSAL,
            )
        ]

    def _make_slice_for_inv_causal_triangle_mask(self) -> list[AttnSlice]:
        """in such case, the mask will be formed as two parts:
        part1: a inv causal mask on the far right made by the first k range
        part2: an optional full mask merged from the later k ranges
        """

        attn_slices: list[AttnSlice] = []

        # part1: inv causal mask on the left
        first_inv_causal_k_range_local = self.k_ranges_local[0]
        first_inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.start + self.first_k_range_global.seqlen,
        )
        attn_slices.append(
            AttnSlice(
                q_range=first_inv_causal_q_range_local,
                k_range=first_inv_causal_k_range_local,
                mask_type=AttnMaskType.INVCAUSAL,
            )
        )

        # part2 (optional): subsequent full mask
        subsequent_full_k_ranges_local = self.k_ranges_local[1:]
        subsequent_full_k_range_local = self._merge_k_ranges_and_check(
            subsequent_full_k_ranges_local,
            allow_empty=True,
        )

        # TODO: The current solution is to divide the slice into a complete rectangle and a small triangle.
        # TODO: The slice can be cut into a rectangle and a trapezoid to ensure that the k_range is not divided.
        if not subsequent_full_k_range_local.is_empty():
            attn_slices.append(
                AttnSlice(
                    q_range=self.q_range_local,
                    k_range=subsequent_full_k_range_local,
                    mask_type=AttnMaskType.FULL,
                )
            )

        return attn_slices

    def _make_slice_for_inv_causal_pentagon_mask(self) -> list[AttnSlice]:
        """in such case, the mask has to divide from the middle into two parts:
        part1: the top full mask
        part2: the bottom inv causal mask
        """

        attn_slices: list[AttnSlice] = []

        k_range_local = self._merge_k_ranges_and_check(
            self.k_ranges_local,
            allow_empty=False,
        )

        # part1: top full mask
        top_full_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            AttnSlice(
                q_range=top_full_q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.FULL,
            ),
        )

        # part2: bottom causal mask
        bottom_causal_q_range_local = AttnRange(
            start=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            AttnSlice(
                q_range=bottom_causal_q_range_local,
                k_range=k_range_local,
                mask_type=AttnMaskType.INVCAUSAL,
            ),
        )

        return attn_slices

    def _make_slice_for_bi_causal(
        self,
        index: int,
    ) -> list[AttnSlice]:
        """For each k_range, split the bi-causal mask,
        and each bi-causal mask is split into at most three optional parts:
        part1: the top causal mask
        part2: the middle full or bi_causal mask
        part3: the bottom inv_causal mask
        """
        # get global and local k_ranges
        k_range_global = self.k_ranges_global[index]
        k_range_local = self.k_ranges_local[index]
        attn_slices: list[AttnSlice] = []

        # calculate k_range exceed slice_start, the maxValue not exceed slice_q_range.seqlen
        range_start_exceed_slice_start = min(
            k_range_global.start - self.calc_k_range_global.start,
            self.q_range_local.seqlen,
        )
        range_end_exceed_slice_start = min(
            k_range_global.end - self.calc_k_range_global.start,
            self.q_range_local.seqlen,
        )

        # calculate k_range exceed causal start, the minValue not less than 0
        range_end_exceed_causal_start = max(0, k_range_global.end - self.causal_start)
        range_start_exceed_causal_start = max(
            0, k_range_global.start - self.causal_start
        )

        # Draw vertical lines from the two endpoints of k_range,
        # which intersect the two hypotenuses of the bi-causal mask at two points.
        # Calculate the vertical coordinates (heights) of these two intersection points,
        # and determine which point is above the other by comparison.
        short_length = min(
            range_start_exceed_slice_start, range_end_exceed_causal_start
        )
        long_length = max(range_start_exceed_slice_start, range_end_exceed_causal_start)

        # (part1) calculate q_range and k_range of causal slice
        causal_q_range_local = AttnRange(
            start=self.q_range_local.start + range_start_exceed_causal_start,
            end=self.q_range_local.start + short_length,
        )
        causal_k_range_local = AttnRange(
            start=k_range_local.start,
            end=min(
                k_range_local.end,
                k_range_local.start + self.diff_len_of_k_range_minus_q_range,
            ),
        )

        # (part2) calculate q_range of full or bi_causal slice
        full_or_bi_causal_q_range_local = AttnRange(
            start=self.q_range_local.start + short_length,
            end=self.q_range_local.start + long_length,
        )

        # (part3) calculate q_range and k_range of inv_causal slice
        inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.start + long_length,
            end=self.q_range_local.start + range_end_exceed_slice_start,
        )
        inv_causal_k_range_local = AttnRange(
            start=max(
                k_range_local.start,
                k_range_local.end - self.diff_len_of_k_range_minus_q_range,
            ),
            end=k_range_local.end,
        )

        # exclude invalid causal slice
        if causal_q_range_local.seqlen > 0:
            attn_slices.append(
                AttnSlice(
                    q_range=causal_q_range_local,
                    k_range=causal_k_range_local,
                    mask_type=AttnMaskType.CAUSAL,
                )
            )

        # exclude invalid full or bi_causal slice
        if full_or_bi_causal_q_range_local.seqlen > 0:
            attn_slices.append(
                AttnSlice(
                    q_range=full_or_bi_causal_q_range_local,
                    k_range=k_range_local,
                    mask_type=AttnMaskType.FULL
                    if range_start_exceed_slice_start > range_end_exceed_causal_start
                    else AttnMaskType.BICAUSAL,
                )
            )

        # exclude invalid inv_causal slice
        if inv_causal_q_range_local.seqlen > 0:
            attn_slices.append(
                AttnSlice(
                    q_range=inv_causal_q_range_local,
                    k_range=inv_causal_k_range_local,
                    mask_type=AttnMaskType.INVCAUSAL,
                )
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

        if self.causal_case_key == self.CausalCaseKey.PENTAGON:
            self.special_pentagon_case_type = False
        elif self.causal_case_key == self.CausalCaseKey.INVALID:
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

    def _init_inv_causal_key(self):
        super()._init_inv_causal_key()

        if self.inv_causal_case_key == self.CausalCaseKey.PENTAGON:
            self.special_pentagon_case_type = False
        elif self.inv_causal_case_key == self.CausalCaseKey.INVALID:
            if (
                self.first_k_range_global.end < self.inv_causal_mask_end
                and self.inv_causal_mask_start
                < self.first_k_range_global.start
                < self.inv_causal_mask_end
            ):
                # this contains special sub-type of cases for the pentagon cases
                # that will be just invalid in host slice maker
                self.inv_causal_case_key = self.CausalCaseKey.PENTAGON
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

    def _make_slice_for_inv_causal_rectangle_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of only several full rectangles

        thus we just call the maker for full mask
        """

        return self._make_slice_for_full_mask()

    def _make_slice_for_inv_causal_trapezoid_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of several full rectangles
        plus a single normal inv causal mask, i.e. an uncut triangle or an uncut trapezoid

        thus the whole masks will be formed as
        the previous full masks plus a single normal inv causal mask in the first
        """

        return [
            MultiKAttnSlice(
                q_range=self.q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.INVCAUSAL]
                + [AttnMaskType.FULL] * (self.batch_size - 1),
            )
        ]

    def _make_slice_for_inv_causal_triangle_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """in such case, the area is made of several full rectangles
        plus a cut triangle on the far bottom-left

        the whole masks can be divided from the middle into two parts:
            part1: the top previous full masks plus a single normal inv causal mask in the first
            part2: the optional bottom full masks
        """

        attn_slices: list[MultiKAttnSlice] = []

        triangle_end = self.q_range_local.start + self.first_k_range_global.seqlen

        # part1: inv causal mask + top full masks
        inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=triangle_end,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=inv_causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.INVCAUSAL]
                + [AttnMaskType.FULL] * (self.batch_size - 1),
            )
        )

        # part2: optional bottom full masks
        full_q_range_local = AttnRange(
            start=triangle_end,
            end=self.q_range_local.end,
        )
        if self.batch_size > 1:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=full_q_range_local,
                    k_ranges=self.k_ranges_global[1:],
                    mask_types=[AttnMaskType.FULL] * (self.batch_size - 1),
                )
            )

        return attn_slices

    def _make_slice_for_inv_causal_pentagon_mask(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """this includes three cases, where the area is made of several full rectangles
        plus either an uncut/cut rotated trapezoid or a pentagon

        we further dispatch them into two sub types of cases:
            normal type: the plused rotated trapezoid is uncut
            special type: the plused rotated trapezoid is cut
        """

        if self.special_pentagon_case_type:
            return self._make_slice_for_inv_causal_pentagon_mask_special()
        else:
            return self._make_slice_for_inv_causal_pentagon_mask_normal()

    def _make_slice_for_inv_causal_pentagon_mask_normal(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
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

        # part1: top full masks
        top_full_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=top_full_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * self.batch_size,
            )
        )

        # part2: bottom full masks + causal mask
        bottom_inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=bottom_inv_causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.INVCAUSAL]
                + [AttnMaskType.FULL] * (self.batch_size - 1),
            )
        )

        return attn_slices

    def _make_slice_for_inv_causal_pentagon_mask_special(self) -> list[MultiKAttnSlice]:  # type: ignore[override]
        """this special type includes two cases:
            case1: the area is made of several full rectangles
                plus a cut rotated trapezoid,
                when q_range.seqlen <= k_range.seqlen in a slice
            case2: the area is made of a single cut rotated trapezoid,
                when q_range.seqlen > k_range.seqlen in a slice
            NOTE: the case2 of the special type is the same as the case2 of the normal type
            we just handle each case2 with the same way as its corr. case1

        thus the whole masks can be divided from the middle into three parts:
            part1: the top full masks
            part2: the middle full masks plus a single normal inv causal mask in the last
            part3: the bottom optional full masks
        """

        attn_slices: list[MultiKAttnSlice] = []

        # part1: top full masks
        top_full_q_range_local = AttnRange(
            start=self.q_range_local.start,
            end=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=top_full_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.FULL] * self.batch_size,
            )
        )

        # part2: middle inv causal mask + full masks
        mid_inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.end
            - self.start_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end
            - self.end_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
        )
        attn_slices.append(
            MultiKAttnSlice(
                q_range=mid_inv_causal_q_range_local,
                k_ranges=self.k_ranges_global,
                mask_types=[AttnMaskType.INVCAUSAL]
                + [AttnMaskType.FULL] * (self.batch_size - 1),
            )
        )

        # part3: bottom optional full masks
        bottom_full_q_range_local = AttnRange(
            start=self.q_range_local.end
            - self.end_distance_to_inv_causal_end
            - self.diff_len_of_q_range_minus_k_range,
            end=self.q_range_local.end,
        )
        if self.batch_size > 1:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=bottom_full_q_range_local,
                    k_ranges=self.k_ranges_global[1:],
                    mask_types=[AttnMaskType.FULL] * (self.batch_size - 1),
                )
            )

        return attn_slices

    def _make_slice_for_bi_causal(  # type: ignore[override]
        self,
        index: int,
    ) -> list[MultiKAttnSlice]:
        """For each k_range, split the bi-causal mask

        each bi-causal mask is split into at most three optional parts:
            part1: the top causal mask
            part2: the middle full or bi_causal mask
            part3: the bottom inv_causal mask
        """
        k_range_global: AttnRange = self.k_ranges_global[index]
        attn_slices: list[MultiKAttnSlice] = []

        # calculate k_range exceed slice_start, the maxValue not exceed slice_q_range.seqlen
        range_start_exceed_slice_start = min(
            k_range_global.start - self.calc_k_range_global.start,
            self.q_range_local.seqlen,
        )
        range_end_exceed_slice_start = min(
            k_range_global.end - self.calc_k_range_global.start,
            self.q_range_local.seqlen,
        )

        # calculate k_range exceed causal start, the minValue not less than 0
        range_end_exceed_causal_start = max(0, k_range_global.end - self.causal_start)
        range_start_exceed_causal_start = max(
            0, k_range_global.start - self.causal_start
        )

        # Draw vertical lines from the two endpoints of k_range,
        # which intersect the two hypotenuses of the bi-causal mask at two points.
        # Calculate the vertical coordinates (heights) of these two intersection points,
        # and determine which point is above the other by comparison.
        short_length = min(
            range_start_exceed_slice_start, range_end_exceed_causal_start
        )
        long_length = max(range_start_exceed_slice_start, range_end_exceed_causal_start)

        # (part1) calculate q_range and k_range of causal slice
        causal_q_range_local = AttnRange(
            start=self.q_range_local.start + range_start_exceed_causal_start,
            end=self.q_range_local.start + short_length,
        )
        causal_k_range_local = AttnRange(
            start=k_range_global.start,
            end=min(
                k_range_global.end,
                k_range_global.start + self.diff_len_of_k_range_minus_q_range,
            ),
        )

        # (part2) calculate q_range of full or bi_causal slice
        full_or_bi_causal_q_range_local = AttnRange(
            start=self.q_range_local.start + short_length,
            end=self.q_range_local.start + long_length,
        )

        # (part3) calculate q_range and k_range of inv_causal slice
        inv_causal_q_range_local = AttnRange(
            start=self.q_range_local.start + long_length,
            end=self.q_range_local.start + range_end_exceed_slice_start,
        )
        inv_causal_k_range_local = AttnRange(
            start=max(
                k_range_global.start,
                k_range_global.end - self.diff_len_of_k_range_minus_q_range,
            ),
            end=k_range_global.end,
        )

        # exclude invalid causal slice
        if causal_q_range_local.seqlen > 0:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=causal_q_range_local,
                    k_ranges=AttnRanges.from_ranges([causal_k_range_local]),
                    mask_types=[AttnMaskType.CAUSAL],
                )
            )

        # exclude invalid full or bi_causal slice
        if full_or_bi_causal_q_range_local.seqlen > 0:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=full_or_bi_causal_q_range_local,
                    k_ranges=AttnRanges.from_ranges([k_range_global]),
                    mask_types=[AttnMaskType.FULL]
                    if range_start_exceed_slice_start > range_end_exceed_causal_start
                    else [AttnMaskType.BICAUSAL],
                )
            )

        # exclude invalid inv_causal slice
        if inv_causal_q_range_local.seqlen > 0:
            attn_slices.append(
                MultiKAttnSlice(
                    q_range=inv_causal_q_range_local,
                    k_ranges=AttnRanges.from_ranges([inv_causal_k_range_local]),
                    mask_types=[AttnMaskType.INVCAUSAL],
                )
            )

        return attn_slices
