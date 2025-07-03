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

import numpy as np

from magi_attention.common import AttnRange
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


def add_range_to_array(
    array: np.ndarray,
    q_range: AttnRange,
    k_range: AttnRange,
    masktype: AttnMaskType = AttnMaskType.FULL,
    check: bool = False,
):
    # get start and end of range
    x_start, x_end = q_range.start, q_range.end
    y_start, y_end = k_range.start, k_range.end

    print(f"{q_range=} {k_range=}")
    if check:
        # check whether the current slice has been filled
        assert np.all(array[x_start:x_end, y_start:y_end] == 0), (
            f"Part of the area has been added," f"when {q_range=} and {k_range=}"
        )

    # fill the area according to the type of the mask.
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            if masktype == AttnMaskType.FULL:
                array[i][j] = 1
            elif masktype == AttnMaskType.CAUSAL:
                b = y_end - x_end
                fx = i + b
                if j <= fx:
                    array[i][j] = 1
            elif masktype == AttnMaskType.INVCAUSAL:
                b = y_start - x_start
                fx = i + b
                if j >= fx:
                    array[i][j] = 1
            elif masktype == AttnMaskType.BICAUSAL:
                causal_b = y_end - x_end
                f_causal = i + causal_b

                inv_causal_b = y_start - x_start
                f_inv_causal = i + inv_causal_b
                if j <= f_causal and j >= f_inv_causal:
                    array[i][j] = 1

    return array


def make_range_global(
    global_ranges: AttnRanges,
    local_range: AttnRange,
) -> AttnRanges:
    """convert local_range to global_ranges with base global_ranges

    Args:
        global_ranges (AttnRanges): the actual base global ranges
        local_range (AttnRange): range need to convert

    Returns:
        AttnRanges: converted multiple ranges since local range may
            be converted to multiple segments of ranges
    """
    assert local_range.seqlen <= global_ranges.total_seqlen

    ranges_ = AttnRanges()

    local_start, local_length = local_range.start, local_range.seqlen

    global_index = 0
    current_global_length = 0
    start_length = local_start

    while global_index < len(global_ranges):
        if global_ranges[global_index].seqlen <= start_length:
            start_length -= global_ranges[global_index].seqlen
            global_index += 1
        else:
            current_global_length = start_length
            break

    while global_index < len(global_ranges):
        if global_ranges[global_index].seqlen - current_global_length < local_length:
            range_ = AttnRange(
                start=global_ranges[global_index].start + current_global_length,
                end=global_ranges[global_index].end,
            )
            local_length = (
                local_length
                - global_ranges[global_index].seqlen
                + current_global_length
            )
            global_index += 1
            current_global_length = 0
            ranges_.append(range_)
        else:
            range_ = AttnRange(
                start=global_ranges[global_index].start + current_global_length,
                end=global_ranges[global_index].start
                + current_global_length
                + local_length,
            )
            ranges_.append(range_)
            break

    return ranges_


def determine_ith_range_masktype(
    i: int,
    length: int,
    masktype: AttnMaskType = AttnMaskType.FULL,
):
    """
    determine mask type in tests for Slice,
    when convert local range with one single masktype to global range with multi masktypes
    """
    if length == 1 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.BICAUSAL
    if i == 0 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.INVCAUSAL
    if i == length - 1 and masktype is AttnMaskType.BICAUSAL:
        return AttnMaskType.CAUSAL
    if i == 0 and masktype is AttnMaskType.INVCAUSAL:
        return AttnMaskType.INVCAUSAL
    if i == length - 1 and masktype is AttnMaskType.CAUSAL:
        return AttnMaskType.CAUSAL
    return AttnMaskType.FULL
