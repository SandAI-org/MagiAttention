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

from magi_attention.api.functools import apply_padding, compute_pad_size
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.testing import parameterize


class TestCalcSelfAttnAreas(TestCase):
    @parameterize(
        "testcase",
        [
            {
                "name": "testcase1",
                "total_seqlen": 1024,
                "cp_size": 6,
                "head_dim": 128,
                "chunk_size": 208,
                "answer": 224,
            },
            {
                "name": "testcase2",
                "total_seqlen": 10244,
                "cp_size": 8,
                "head_dim": 128,
                "chunk_size": 451,
                "answer": 580,
            },
            {
                "name": "testcase3",
                "total_seqlen": 3840,
                "cp_size": 10,
                "head_dim": 128,
                "chunk_size": 128,
                "answer": 0,
            },
            {
                "name": "testcase4",
                "total_seqlen": 42817,
                "cp_size": 12,
                "head_dim": 128,
                "chunk_size": 892,
                "answer": 10703,
            },
            {
                "name": "testcase5",
                "total_seqlen": 13000,
                "cp_size": 17,
                "head_dim": 128,
                "chunk_size": 369,
                "answer": 5819,
            },
            {
                "name": "testcase6",
                "total_seqlen": 1789,
                "cp_size": 2,
                "head_dim": 128,
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
        head_dim: int = testcase["head_dim"]
        chunk_size: int = testcase["chunk_size"]
        answer: int = testcase["answer"]

        result = compute_pad_size(
            total_seqlen_q=total_seqlen,
            cp_size=cp_size,
            head_dim=head_dim,
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


if __name__ == "__main__":
    unittest.main()
