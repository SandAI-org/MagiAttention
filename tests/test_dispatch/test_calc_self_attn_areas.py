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

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.meta import make_global_bucket
from magi_attention.meta.container import AttnBucket, AttnChunk, AttnSlice
from magi_attention.testing import parameterize
from magi_attention.utils import argsort


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


class TestCalcSelfAttnAreas(TestCase):
    def test_calc_self_attn_areas_one(self):
        q_ranges = AttnRanges.from_ranges(
            [
                (0, 10),
                (10, 16),
                (16, 30),
                (30, 43),
                (43, 61),
                (61, 64),
            ],
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (0, 11),
                (5, 18),
                (18, 32),
                (32, 45),
                (45, 64),
                (53, 64),
            ]
        )
        attn_mask_type = [
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
        ]

        num_chunks = 8
        chunk_size = 8

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(0, 8),
                            k_range=AttnRange(0, 11),
                            _area=88,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=1,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(8, 10),
                            k_range=AttnRange(0, 11),
                            _area=22,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(10, 16),
                            k_range=AttnRange(5, 18),
                            _area=63,
                        ),
                    ],
                    sample_ids=[0, 1],
                ),
                AttnChunk(
                    chunk_id=2,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(16, 24),
                            k_range=AttnRange(18, 26),
                            _area=36,
                        ),
                    ],
                    sample_ids=[2],
                ),
                AttnChunk(
                    chunk_id=3,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(24, 30),
                            k_range=AttnRange(18, 32),
                            _area=69,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(30, 32),
                            k_range=AttnRange(32, 34),
                            _area=3,
                        ),
                    ],
                    sample_ids=[2, 3],
                ),
                AttnChunk(
                    chunk_id=4,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(32, 40),
                            k_range=AttnRange(32, 42),
                            _area=52,
                        ),
                    ],
                    sample_ids=[3],
                ),
                AttnChunk(
                    chunk_id=5,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(40, 43),
                            k_range=AttnRange(32, 45),
                            _area=36,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(43, 48),
                            k_range=AttnRange(45, 64),
                            _area=95,
                        ),
                    ],
                    sample_ids=[3, 4],
                ),
                AttnChunk(
                    chunk_id=6,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(48, 56),
                            k_range=AttnRange(45, 64),
                            _area=152,
                        ),
                    ],
                    sample_ids=[4],
                ),
                AttnChunk(
                    chunk_id=7,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(56, 61),
                            k_range=AttnRange(45, 64),
                            _area=95,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(61, 64),
                            k_range=AttnRange(53, 64),
                            _area=33,
                        ),
                    ],
                    sample_ids=[4, 5],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The test of testcase one is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    def test_calc_self_attn_areas_two(self):
        q_ranges = AttnRanges.from_ranges(
            [
                (0, 2),
                (2, 5),
                (5, 8),
                (8, 10),
                (10, 14),
                (14, 16),
                (16, 22),
                (22, 29),
                (29, 32),
            ],
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (0, 2),
                (2, 5),
                (5, 8),
                (8, 16),
                (8, 16),
                (8, 16),
                (4, 20),
                (17, 24),
                (19, 31),
            ]
        )
        attn_mask_type = [  # TODO: limited to all full attn masks for now
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
        ]

        num_chunks = 4
        chunk_size = 8

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(0, 2),
                            k_range=AttnRange(0, 2),
                            _area=3,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(2, 5),
                            k_range=AttnRange(2, 5),
                            _area=6,
                        ),
                        AttnSlice(
                            slice_id=2,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(5, 8),
                            k_range=AttnRange(5, 8),
                            _area=6,
                        ),
                    ],
                    sample_ids=[0, 1, 2],
                ),
                AttnChunk(
                    chunk_id=1,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(8, 10),
                            k_range=AttnRange(8, 16),
                            _area=16,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(10, 14),
                            k_range=AttnRange(8, 16),
                            _area=26,
                        ),
                        AttnSlice(
                            slice_id=2,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(14, 16),
                            k_range=AttnRange(8, 16),
                            _area=15,
                        ),
                    ],
                    sample_ids=[3, 4, 5],
                ),
                AttnChunk(
                    chunk_id=2,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(16, 22),
                            k_range=AttnRange(4, 20),
                            _area=81,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(22, 24),
                            k_range=AttnRange(17, 19),
                            _area=3,
                        ),
                    ],
                    sample_ids=[6, 7],
                ),
                AttnChunk(
                    chunk_id=3,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(24, 29),
                            k_range=AttnRange(17, 24),
                            _area=25,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(29, 32),
                            k_range=AttnRange(19, 31),
                            _area=36,
                        ),
                    ],
                    sample_ids=[7, 8],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The test in testcase one is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    def test_calc_self_attn_areas_all_full(self):
        q_ranges = AttnRanges.from_ranges(
            [
                (0, 8),
                (8, 22),
                (22, 38),
                (38, 51),
                (51, 56),
                (56, 64),
                (64, 72),
                (72, 89),
                (89, 96),
            ],
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (15, 30),
                (56, 74),
                (25, 87),
                (7, 58),
                (71, 90),
                (62, 90),
                (71, 90),
                (3, 96),
                (7, 49),
            ]
        )
        attn_mask_type = [AttnMaskType.FULL for _ in range(len(q_ranges))]

        num_chunks = 12
        chunk_size = 8

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(0, 8),
                            k_range=AttnRange(15, 30),
                            _area=120,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=1,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(8, 16),
                            k_range=AttnRange(56, 74),
                            _area=144,
                        ),
                    ],
                    sample_ids=[1],
                ),
                AttnChunk(
                    chunk_id=2,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(16, 22),
                            k_range=AttnRange(56, 74),
                            _area=108,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(22, 24),
                            k_range=AttnRange(25, 87),
                            _area=124,
                        ),
                    ],
                    sample_ids=[1, 2],
                ),
                AttnChunk(
                    chunk_id=3,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(24, 32),
                            k_range=AttnRange(25, 87),
                            _area=496,
                        ),
                    ],
                    sample_ids=[2],
                ),
                AttnChunk(
                    chunk_id=4,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(32, 38),
                            k_range=AttnRange(25, 87),
                            _area=372,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(38, 40),
                            k_range=AttnRange(7, 58),
                            _area=102,
                        ),
                    ],
                    sample_ids=[2, 3],
                ),
                AttnChunk(
                    chunk_id=5,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(40, 48),
                            k_range=AttnRange(7, 58),
                            _area=408,
                        ),
                    ],
                    sample_ids=[3],
                ),
                AttnChunk(
                    chunk_id=6,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(48, 51),
                            k_range=AttnRange(7, 58),
                            _area=153,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(51, 56),
                            k_range=AttnRange(71, 90),
                            _area=95,
                        ),
                    ],
                    sample_ids=[3, 4],
                ),
                AttnChunk(
                    chunk_id=7,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(56, 64),
                            k_range=AttnRange(62, 90),
                            _area=224,
                        ),
                    ],
                    sample_ids=[5],
                ),
                AttnChunk(
                    chunk_id=8,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(64, 72),
                            k_range=AttnRange(71, 90),
                            _area=152,
                        ),
                    ],
                    sample_ids=[6],
                ),
                AttnChunk(
                    chunk_id=9,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(72, 80),
                            k_range=AttnRange(3, 96),
                            _area=744,
                        ),
                    ],
                    sample_ids=[7],
                ),
                AttnChunk(
                    chunk_id=10,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(80, 88),
                            k_range=AttnRange(3, 96),
                            _area=744,
                        ),
                    ],
                    sample_ids=[7],
                ),
                AttnChunk(
                    chunk_id=11,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(88, 89),
                            k_range=AttnRange(3, 96),
                            _area=93,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.FULL,
                            q_range=AttnRange(89, 96),
                            k_range=AttnRange(7, 49),
                            _area=294,
                        ),
                    ],
                    sample_ids=[7, 8],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The test of all full is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    def test_calc_self_attn_areas_all_causal_1(self):
        q_ranges = AttnRanges.from_ranges(
            [
                (0, 8),
                (8, 24),
                (24, 38),
                (38, 57),
                (57, 83),
                (83, 92),
                (92, 96),
            ],
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (12, 20),
                (27, 43),
                (43, 48),
                (48, 67),
                (5, 74),
                (31, 86),
                (67, 96),
            ]
        )
        attn_mask_type = [AttnMaskType.CAUSAL for _ in range(len(q_ranges))]

        num_chunks = 12
        chunk_size = 8

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(0, 8),
                            k_range=AttnRange(12, 20),
                            _area=36,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=1,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(8, 16),
                            k_range=AttnRange(27, 35),
                            _area=36,
                        ),
                    ],
                    sample_ids=[1],
                ),
                AttnChunk(
                    chunk_id=2,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(16, 24),
                            k_range=AttnRange(27, 43),
                            _area=100,
                        ),
                    ],
                    sample_ids=[1],
                ),
                AttnChunk(chunk_id=3, q_slices=[]),
                AttnChunk(
                    chunk_id=4,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(33, 38),
                            k_range=AttnRange(43, 48),
                            _area=15,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(38, 40),
                            k_range=AttnRange(48, 50),
                            _area=3,
                        ),
                    ],
                    sample_ids=[2, 3],
                ),
                AttnChunk(
                    chunk_id=5,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(40, 48),
                            k_range=AttnRange(48, 58),
                            _area=52,
                        ),
                    ],
                    sample_ids=[3],
                ),
                AttnChunk(
                    chunk_id=6,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(48, 56),
                            k_range=AttnRange(48, 66),
                            _area=116,
                        ),
                    ],
                    sample_ids=[3],
                ),
                AttnChunk(
                    chunk_id=7,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(56, 57),
                            k_range=AttnRange(48, 67),
                            _area=19,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(57, 64),
                            k_range=AttnRange(5, 55),
                            _area=329,
                        ),
                    ],
                    sample_ids=[3, 4],
                ),
                AttnChunk(
                    chunk_id=8,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(64, 72),
                            k_range=AttnRange(5, 63),
                            _area=436,
                        ),
                    ],
                    sample_ids=[4],
                ),
                AttnChunk(
                    chunk_id=9,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(72, 80),
                            k_range=AttnRange(5, 71),
                            _area=500,
                        ),
                    ],
                    sample_ids=[4],
                ),
                AttnChunk(
                    chunk_id=10,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(80, 83),
                            k_range=AttnRange(5, 74),
                            _area=204,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(83, 88),
                            k_range=AttnRange(31, 82),
                            _area=245,
                        ),
                    ],
                    sample_ids=[4, 5],
                ),
                AttnChunk(
                    chunk_id=11,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(88, 92),
                            k_range=AttnRange(31, 86),
                            _area=214,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(92, 96),
                            k_range=AttnRange(67, 96),
                            _area=110,
                        ),
                    ],
                    sample_ids=[5, 6],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The test in all_causal_1 is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    def test_calc_self_attn_areas_all_causal_2(self):
        q_ranges = AttnRanges.from_ranges(
            [
                (0, 30),
                (30, 53),
                (53, 74),
                (74, 96),
            ],
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (0, 50),
                (61, 71),
                (34, 47),
                (57, 90),
            ]
        )
        attn_mask_type = [AttnMaskType.CAUSAL for _ in range(len(q_ranges))]

        num_chunks = 12
        chunk_size = 8

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(0, 8),
                            k_range=AttnRange(0, 28),
                            _area=196,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=1,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(8, 16),
                            k_range=AttnRange(0, 36),
                            _area=260,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=2,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(16, 24),
                            k_range=AttnRange(0, 44),
                            _area=324,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(
                    chunk_id=3,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(24, 30),
                            k_range=AttnRange(0, 50),
                            _area=285,
                        ),
                    ],
                    sample_ids=[0],
                ),
                AttnChunk(chunk_id=4, q_slices=[]),
                AttnChunk(
                    chunk_id=5,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(43, 48),
                            k_range=AttnRange(61, 66),
                            _area=15,
                        ),
                    ],
                    sample_ids=[1],
                ),
                AttnChunk(
                    chunk_id=6,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(48, 53),
                            k_range=AttnRange(61, 71),
                            _area=40,
                        ),
                    ],
                    sample_ids=[1],
                ),
                AttnChunk(
                    chunk_id=7,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(61, 64),
                            k_range=AttnRange(34, 37),
                            _area=6,
                        ),
                    ],
                    sample_ids=[2],
                ),
                AttnChunk(
                    chunk_id=8,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(64, 72),
                            k_range=AttnRange(34, 45),
                            _area=60,
                        ),
                    ],
                    sample_ids=[2],
                ),
                AttnChunk(
                    chunk_id=9,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(72, 74),
                            k_range=AttnRange(34, 47),
                            _area=25,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(74, 80),
                            k_range=AttnRange(57, 74),
                            _area=87,
                        ),
                    ],
                    sample_ids=[2, 3],
                ),
                AttnChunk(
                    chunk_id=10,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(80, 88),
                            k_range=AttnRange(57, 82),
                            _area=172,
                        ),
                    ],
                    sample_ids=[3],
                ),
                AttnChunk(
                    chunk_id=11,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(88, 96),
                            k_range=AttnRange(57, 90),
                            _area=236,
                        ),
                    ],
                    sample_ids=[3],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The testcase of causal_2 is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    def test_calc_self_attn_areas_one_line(self):
        q_ranges = AttnRanges.from_ranges(
            [(0, 1), (1, 10)],
        )
        k_ranges = AttnRanges.from_ranges([(0, 10), (0, 1)])
        attn_mask_type = [AttnMaskType.CAUSAL for _ in range(len(q_ranges))]

        num_chunks = 1
        chunk_size = 10

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        result_bucket = AttnBucket(
            q_chunks=[
                AttnChunk(
                    chunk_id=0,
                    q_slices=[
                        AttnSlice(
                            slice_id=0,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(0, 1),
                            k_range=AttnRange(0, 10),
                            _area=10,
                        ),
                        AttnSlice(
                            slice_id=1,
                            mask_type=AttnMaskType.CAUSAL,
                            q_range=AttnRange(9, 10),
                            k_range=AttnRange(0, 1),
                            _area=1,
                        ),
                    ],
                    sample_ids=[0, 1],
                ),
            ]
        )

        assert global_bucket == result_bucket, (
            f"The testcase of one line is not passed!\n"
            f"expect result={result_bucket}\n"
            f"but got {global_bucket}."
        )

    @parameterize(
        "testcase",
        [
            (
                AttnRanges.from_ranges(
                    [
                        (0, 8),
                        (8, 24),
                        (24, 38),
                        (38, 57),
                        (57, 83),
                        (83, 92),
                        (92, 96),
                    ],
                ),
                AttnRanges.from_ranges(
                    [
                        (12, 20),
                        (27, 43),
                        (43, 48),
                        (48, 67),
                        (5, 74),
                        (31, 86),
                        (67, 96),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        (0, 30),
                        (30, 53),
                        (53, 74),
                        (74, 96),
                    ],
                ),
                AttnRanges.from_ranges(
                    [
                        (0, 50),
                        (61, 71),
                        (34, 47),
                        (57, 90),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        (0, 10),
                        (10, 16),
                        (16, 30),
                        (30, 43),
                        (43, 61),
                        (61, 64),
                    ],
                ),
                AttnRanges.from_ranges(
                    [
                        (0, 11),
                        (5, 18),
                        (18, 32),
                        (32, 45),
                        (45, 64),
                        (53, 64),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        (0, 2),
                        (2, 5),
                        (5, 8),
                        (8, 10),
                        (10, 14),
                        (14, 16),
                        (16, 22),
                        (22, 29),
                        (29, 32),
                    ],
                ),
                AttnRanges.from_ranges(
                    [
                        (0, 2),
                        (2, 5),
                        (5, 8),
                        (8, 16),
                        (8, 16),
                        (8, 16),
                        (4, 20),
                        (17, 24),
                        (19, 31),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        [0, 8],
                        [0, 8],
                        [8, 16],
                        [16, 24],
                        [24, 32],
                        [32, 40],
                        [40, 48],
                        [0, 14],
                        [0, 16],
                        [18, 20],
                        [23, 42],
                    ]
                ),
                AttnRanges.from_ranges(
                    [
                        [0, 8],
                        [8, 16],
                        [0, 8],
                        [0, 8],
                        [0, 8],
                        [0, 8],
                        [0, 8],
                        [16, 20],
                        [20, 24],
                        [9, 10],
                        [33, 45],
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        (0, 8),
                        (24, 32),
                    ]
                ),
                AttnRanges.from_ranges(
                    [
                        (0, 8),
                        (24, 32),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        (0, 10),
                        (2, 5),
                        (7, 30),
                        (28, 40),
                        (29, 41),
                        (45, 48),
                        (45, 48),
                    ]
                ),
                AttnRanges.from_ranges(
                    [
                        (0, 8),
                        (8, 16),
                        (16, 24),
                        (0, 8),
                        (32, 34),
                        (0, 8),
                        (16, 18),
                    ]
                ),
            ),
            (
                AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [128, 256],
                        [256, 512],
                        [512, 1024],
                    ]
                ),
                AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 512],
                        [512, 1024],
                    ]
                ),
            ),
        ],
    )
    @parameterize(
        "masktype",
        [
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.INVCAUSAL,
            AttnMaskType.BICAUSAL,
        ],
    )
    @parameterize("chunk_size", [4, 8, 16])
    def test_calc_self_attn_areas(
        self,
        testcase: tuple[AttnRanges, AttnRanges],
        masktype: AttnMaskType,
        chunk_size: int,
    ):
        q_ranges, k_ranges = testcase
        attn_mask_type = [masktype] * len(q_ranges)

        assert q_ranges.end % chunk_size == 0
        num_chunks = q_ranges.end // chunk_size

        sorted_indices = argsort(q_ranges, key=lambda x: (x.start, x.end))
        q_ranges._ranges = [q_ranges[i] for i in sorted_indices]
        k_ranges._ranges = [k_ranges[i] for i in sorted_indices]
        attn_mask_type = [attn_mask_type[i] for i in sorted_indices]

        global_bucket = make_global_bucket(
            q_ranges,
            k_ranges,
            attn_mask_type,
            num_chunks,
            chunk_size,
        )

        assert (
            len(global_bucket.q_chunks) == num_chunks
        ), f"The num of chunks must be {num_chunks}, but got {len(global_bucket.q_chunks)}"

        answer = np.zeros((q_ranges.end, q_ranges.end), dtype=np.int32)
        result = np.zeros((q_ranges.end, q_ranges.end), dtype=np.int32)

        for q_range, k_range, masktype in zip(q_ranges, k_ranges, attn_mask_type):
            add_range_to_array(
                array=answer,
                q_range=q_range,
                k_range=k_range,
                masktype=masktype,
                check=True,
            )

        for chunk_index, chunk in enumerate(global_bucket.q_chunks):
            chunk_begin = chunk_index * chunk_size
            chunk_end = (chunk_index + 1) * chunk_size

            for slice in chunk.attn_slices:
                add_range_to_array(
                    array=result,
                    q_range=slice.q_range,  # type: ignore
                    k_range=slice.k_range,  # type: ignore
                    masktype=slice.mask_type,  # type: ignore
                    check=True,
                )
                q_range_start, q_range_end = slice.q_range.start, slice.q_range.end  # type: ignore

                assert (
                    chunk_begin <= q_range_start < chunk_end
                    and chunk_begin < q_range_end <= chunk_end
                )

        assert np.array_equal(answer, result), (
            f"There's wrong with {global_bucket=}, "
            f"when {q_ranges=}, {k_ranges=}, {attn_mask_type=} and {chunk_size=}"
        )


if __name__ == "__main__":
    unittest.main()
