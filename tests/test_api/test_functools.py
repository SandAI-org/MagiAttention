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

import unittest
from unittest import TestCase

import numpy as np

from magi_attention.api.functools import (
    apply_padding,
    compute_pad_size,
    infer_attn_mask_from_cu_seqlens,
    infer_attn_mask_from_sliding_window,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.testing import parameterize


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


def generate_sliding_window_mask(
    q_seqlen: int,
    k_seqlen: int,
    window_size: tuple[int, int],
) -> np.ndarray:
    left_window_size = window_size[0] if window_size[0] != -1 else k_seqlen - 1
    right_window_size = window_size[1] if window_size[1] != -1 else k_seqlen - 1
    q_idxs = (np.arange(q_seqlen) + (k_seqlen - q_seqlen))[:, None]  # [q_len, 1]
    k_idxs = np.arange(k_seqlen)[None, :]  # [1, k_len]
    mask = (k_idxs >= q_idxs - left_window_size) & (
        k_idxs <= q_idxs + right_window_size
    )

    return mask.astype(np.int32)  # [q_len, k_len]


def generate_sliding_window_mask_with_numpy(
    q_range: AttnRange,
    k_range: AttnRange,
    window_size: tuple[int, int],
    array_size: int,
) -> np.ndarray:
    mask = np.zeros((array_size, array_size), dtype=np.int32)

    q_seqlen, k_seqlen = min(q_range.seqlen, k_range.seqlen), k_range.seqlen

    mask[
        q_range.end - q_seqlen : q_range.end, k_range.end - k_seqlen : k_range.end
    ] = generate_sliding_window_mask(
        q_seqlen=q_seqlen,
        k_seqlen=k_seqlen,
        window_size=window_size,
    )

    return mask


def generate_varlen_sliding_window_mask_with_numpy(
    cu_seqlens: list[int],
    window_size: tuple[int, int],
    total_seqlen: int,
) -> np.ndarray:
    mask = np.zeros((total_seqlen, total_seqlen), dtype=np.int32)

    for i in range(len(cu_seqlens) - 1):
        varlen_seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
        mask[
            cu_seqlens[i] : cu_seqlens[i + 1], cu_seqlens[i] : cu_seqlens[i + 1]
        ] = generate_sliding_window_mask(
            q_seqlen=varlen_seqlen,
            k_seqlen=varlen_seqlen,
            window_size=window_size,
        )

    return mask


class TestFunctools(TestCase):
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase1",
                "total_seqlen": 1024,
                "cp_size": 6,
                "chunk_size": 208,
                "answer": 224,
            },
            {
                "name": "testcase2",
                "total_seqlen": 10244,
                "cp_size": 8,
                "chunk_size": 451,
                "answer": 580,
            },
            {
                "name": "testcase3",
                "total_seqlen": 3840,
                "cp_size": 10,
                "chunk_size": 128,
                "answer": 0,
            },
            {
                "name": "testcase4",
                "total_seqlen": 42817,
                "cp_size": 12,
                "chunk_size": 892,
                "answer": 10703,
            },
            {
                "name": "testcase5",
                "total_seqlen": 13000,
                "cp_size": 17,
                "chunk_size": 369,
                "answer": 5819,
            },
            {
                "name": "testcase6",
                "total_seqlen": 1789,
                "cp_size": 2,
                "chunk_size": 900,
                "answer": 11,
            },
        ],
    )
    def test_pad_size(
        self,
        testcase: dict,
    ):
        name: str = testcase["name"]
        total_seqlen: int = testcase["total_seqlen"]
        cp_size: int = testcase["cp_size"]
        chunk_size: int = testcase["chunk_size"]
        answer: int = testcase["answer"]

        result = compute_pad_size(
            total_seqlen_q=total_seqlen,
            cp_size=cp_size,
            chunk_size=chunk_size,
        )

        assert answer == result, f"wrong answer in {name} testcase"

    @parameterize(
        "testcase",
        [
            {
                "name": "testcase1",
                "q_ranges": [[0, 1024]],
                "k_ranges": [[0, 1024]],
                "attn_type_map": [0],
                "total_seqlen": 1024,
                "pad_size": 200,
                "ref_q_ranges": [
                    [0, 1024],
                    [1024, 1224],
                ],
                "ref_k_ranges": [
                    [0, 1024],
                    [0, 0],
                ],
                "ref_attn_type_map": [0, 0],
            },
            {
                "name": "testcase2",
                "q_ranges": [
                    [0, 1230],
                    [400, 2000],
                    [2500, 3000],
                ],
                "k_ranges": [
                    [0, 1024],
                    [1024, 2048],
                    [0, 2048],
                ],
                "attn_type_map": [0, 1, 2],
                "total_seqlen": 3100,
                "pad_size": 580,
                "ref_q_ranges": [
                    [0, 1230],
                    [400, 2000],
                    [2500, 3000],
                    [3100, 3680],
                ],
                "ref_k_ranges": [
                    [0, 1024],
                    [1024, 2048],
                    [0, 2048],
                    [0, 0],
                ],
                "ref_attn_type_map": [0, 1, 2, 0],
            },
            {
                "name": "testcase3",
                "q_ranges": [
                    [0, 20000],
                    [20000, 30000],
                    [30000, 80000],
                ],
                "k_ranges": [
                    [0, 10000],
                    [30000, 70000],
                    [40000, 60000],
                ],
                "attn_type_map": [3, 2, 0],
                "total_seqlen": 80000,
                "pad_size": 10000,
                "ref_q_ranges": [
                    [0, 20000],
                    [20000, 30000],
                    [30000, 80000],
                    [80000, 90000],
                ],
                "ref_k_ranges": [
                    [0, 10000],
                    [30000, 70000],
                    [40000, 60000],
                    [0, 0],
                ],
                "ref_attn_type_map": [3, 2, 0, 0],
            },
        ],
    )
    def test_apply_padding(
        self,
        testcase: dict,
    ):
        name: str = testcase["name"]
        q_ranges: AttnRanges = AttnRanges.from_ranges(testcase["q_ranges"])
        k_ranges: AttnRanges = AttnRanges.from_ranges(testcase["k_ranges"])
        attn_type_map: list[int] = testcase["attn_type_map"]
        total_seqlen: int = testcase["total_seqlen"]
        pad_size: int = testcase["pad_size"]

        ref_q_ranges: AttnRanges = AttnRanges.from_ranges(testcase["ref_q_ranges"])
        ref_k_ranges: AttnRanges = AttnRanges.from_ranges(testcase["ref_k_ranges"])
        ref_attn_type_map: list[int] = testcase["ref_attn_type_map"]

        attn_mask_type: list[AttnMaskType] = [
            [
                AttnMaskType.FULL,
                AttnMaskType.CAUSAL,
                AttnMaskType.INVCAUSAL,
                AttnMaskType.BICAUSAL,
            ][attn_type]
            for attn_type in attn_type_map
        ]
        ref_attn_mask_type: list[AttnMaskType] = [
            [
                AttnMaskType.FULL,
                AttnMaskType.CAUSAL,
                AttnMaskType.INVCAUSAL,
                AttnMaskType.BICAUSAL,
            ][attn_type]
            for attn_type in ref_attn_type_map
        ]

        q_ranges_padded, k_ranges_padded, attn_mask_type_padded = apply_padding(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen=total_seqlen,
            pad_size=pad_size,
        )

        assert (
            q_ranges_padded == ref_q_ranges
        ), f"in {name} case, {q_ranges_padded=} is not same as {ref_q_ranges=}"
        assert (
            k_ranges_padded == ref_k_ranges
        ), f"in {name} case, {k_ranges_padded=} is not same as {ref_k_ranges=}"
        assert (
            attn_mask_type_padded == ref_attn_mask_type
        ), f"in {name} case, {attn_mask_type_padded=} is not same as {ref_attn_mask_type=}"

    @parameterize(
        "testcase",
        [
            {
                "name": "testcase1",
                "q_range": AttnRange.from_range([5, 25]),
                "k_range": AttnRange.from_range([5, 25]),
                "range_size": 30,
            },
            {
                "name": "testcase2",
                "q_range": AttnRange.from_range([5, 15]),
                "k_range": AttnRange.from_range([7, 30]),
                "range_size": 35,
            },
            {
                "name": "testcase3",
                "q_range": AttnRange.from_range([5, 30]),
                "k_range": AttnRange.from_range([10, 20]),
                "range_size": 30,
            },
            {
                "name": "testcase4",
                "q_range": AttnRange.from_range([10, 20]),
                "k_range": AttnRange.from_range([5, 45]),
                "range_size": 50,
            },
            {
                "name": "testcase5",
                "q_range": AttnRange.from_range([3, 25]),
                "k_range": AttnRange.from_range([20, 24]),
                "range_size": 40,
            },
        ],
    )
    def test_infer_attn_mask_from_sliding_window(
        self,
        testcase: dict,
    ):
        case_name: str = testcase["name"]
        q_range: AttnRange = testcase["q_range"]
        k_range: AttnRange = testcase["k_range"]
        range_size: int = testcase["range_size"]

        name = f"in {case_name}, when {q_range=} x {k_range=} x" f"{range_size=}"

        seqlen = k_range.seqlen
        for i in range(-1, seqlen):
            for j in range(-1, seqlen):
                # calculate function answer
                mask = np.zeros((range_size, range_size))
                q_ranges, k_ranges, masktypes = infer_attn_mask_from_sliding_window(
                    q_range=q_range,
                    k_range=k_range,
                    window_size=(i, j),
                )

                # Accumulate all results into an array and perform validation.
                for sw_q_range, sw_k_range, sw_mask_type in zip(
                    q_ranges, k_ranges, masktypes
                ):
                    add_range_to_array(
                        array=mask,
                        q_range=sw_q_range,
                        k_range=sw_k_range,
                        masktype=sw_mask_type,
                        check=True,
                    )

                # calculate ref answer
                ref_mask = generate_sliding_window_mask_with_numpy(
                    q_range=q_range,
                    k_range=k_range,
                    window_size=(i, j),
                    array_size=range_size,
                )

                assert np.array_equal(
                    mask, ref_mask
                ), f"There's wrong when {name=} with window_size=({i}, {j})"

    @parameterize(
        "testcase",
        [
            {
                "name": "testcase1",
                "cu_seqlens": [0, 10, 20, 40, 60, 100, 150, 170, 180, 185],
                "window_size_length": 10,
            },
            {
                "name": "testcase2",
                "cu_seqlens": [0, 5, 16, 40, 56, 90, 150, 300, 800],
                "window_size_length": 23,
            },
            {
                "name": "testcase4",
                "cu_seqlens": [0, 15, 30, 45, 60, 75, 90],
                "window_size_length": 5,
            },
            {
                "name": "testcase4",
                "cu_seqlens": [0, 100, 146, 200, 221, 230, 234, 236],
                "window_size_length": 41,
            },
        ],
    )
    def test_infer_attn_mask_from_cu_seqlens(
        self,
        testcase: dict,
    ):
        case_name: str = testcase["name"]
        cu_seqlens: list[int] = testcase["cu_seqlens"]
        window_size_length: int = testcase["window_size_length"]
        total_seqlen = cu_seqlens[-1]

        name = f"in {case_name}, when {cu_seqlens=} x {window_size_length=}"

        for i in range(-1, window_size_length):
            for j in range(-1, window_size_length):
                # calculate function answer
                mask = np.zeros((total_seqlen, total_seqlen))
                q_ranges, k_ranges, masktypes = infer_attn_mask_from_cu_seqlens(
                    cu_seqlens=cu_seqlens,
                    window_size=(i, j),
                )

                # Accumulate all results into an array and perform validation.
                for sw_q_range, sw_k_range, sw_mask_type in zip(
                    q_ranges, k_ranges, masktypes
                ):
                    add_range_to_array(
                        array=mask,
                        q_range=sw_q_range,
                        k_range=sw_k_range,
                        masktype=sw_mask_type,
                        check=True,
                    )

                # calculate ref answer
                ref_mask = generate_varlen_sliding_window_mask_with_numpy(
                    cu_seqlens=cu_seqlens,
                    window_size=(i, j),
                    total_seqlen=total_seqlen,
                )

                assert np.array_equal(
                    mask, ref_mask
                ), f"There's wrong when {name=} with window_size=({i}, {j})"


if __name__ == "__main__":
    unittest.main()
