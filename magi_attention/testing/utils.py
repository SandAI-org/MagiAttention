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


# FIXME fix bugs and move to magi_attention/api/functools
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
        f"but got {window_size=}",
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
