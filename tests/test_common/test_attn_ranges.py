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

import unittest
from types import ModuleType, SimpleNamespace

import pytest
import torch

from magi_attention.common.enum import AttnMaskType as PyAttnMaskType
from magi_attention.common.range import AttnRange as PyAttnRange
from magi_attention.common.ranges import AttnRanges as PyAttnRanges
from magi_attention.common.rectangle import AttnRectangle as PyAttnRectangle
from magi_attention.common.rectangles import AttnRectangles as PyAttnRectangles
from magi_attention.testing import parameterize

cpp_backend: ModuleType | None
try:
    from magi_attention import magi_attn_ext as cpp_backend
except ImportError:
    cpp_backend = None


class TestAttnRanges(unittest.TestCase):
    backend = "python"

    def setUp(self):
        if self.backend == "cpp":
            if cpp_backend is None:
                self.skipTest("magi_attn_ext not available")
            self.types = cpp_backend
        else:
            self.types = SimpleNamespace(
                AttnRange=PyAttnRange,
                AttnRanges=PyAttnRanges,
                AttnMaskType=PyAttnMaskType,
                AttnRectangle=PyAttnRectangle,
                AttnRectangles=PyAttnRectangles,
            )

    def test_init(self):
        """Test initialization"""
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        self.assertEqual(len(attn_ranges), 0)
        self.assertTrue(attn_ranges.is_empty())

    def test_is_valid(self):
        """Test validity check"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        # Empty ranges are always valid
        attn_ranges = AttnRanges()
        self.assertTrue(attn_ranges.is_valid())

        # Add valid ranges
        attn_ranges.append(AttnRange(0, 10))
        self.assertTrue(attn_ranges.is_valid())

    def test_check_valid(self):
        """Test validity verification"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        # All ranges are valid, should pass check
        attn_ranges.check_valid()

    def test_append(self):
        """Test adding ranges"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()

        # Test basic append
        attn_ranges.append(AttnRange(0, 10))
        self.assertEqual(len(attn_ranges), 1)
        self.assertEqual(attn_ranges[0], AttnRange(0, 10))

        # Test append with check
        attn_ranges.append(AttnRange(10, 20), check=True)
        self.assertEqual(len(attn_ranges), 2)

    def test_from_ranges(self):
        """Test creating AttnRanges from different types"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        # Create from AttnRanges
        original = AttnRanges()
        original.append(AttnRange(0, 10))
        original.append(AttnRange(20, 30))

        copied = AttnRanges.from_ranges(original)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)

        # Create from AttnRange list
        rect_range_list = [AttnRange(0, 10), AttnRange(20, 30)]
        attn_ranges = AttnRanges.from_ranges(rect_range_list)
        self.assertEqual(len(attn_ranges), 2)
        self.assertEqual(attn_ranges[0], AttnRange(0, 10))
        self.assertEqual(attn_ranges[1], AttnRange(20, 30))

        # Create from NaiveRanges
        naive_ranges = [(0, 10), (20, 30)]
        attn_ranges = AttnRanges.from_ranges(naive_ranges)
        self.assertEqual(len(attn_ranges), 2)
        self.assertEqual(attn_ranges[0], AttnRange(0, 10))
        self.assertEqual(attn_ranges[1], AttnRange(20, 30))

        # Test creation with check
        valid_ranges = [(0, 10), (20, 30)]
        attn_ranges = AttnRanges.from_ranges(valid_ranges, check=True)
        self.assertEqual(len(attn_ranges), 2)
        self.assertEqual(attn_ranges[0], AttnRange(0, 10))
        self.assertEqual(attn_ranges[1], AttnRange(20, 30))

    def test_to_naive_ranges(self):
        """Test conversion to naive ranges"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        naive_ranges = attn_ranges.to_naive_ranges()
        expected = [(0, 10), (20, 30)]
        self.assertEqual(naive_ranges, expected)

    def test_is_empty(self):
        """Test empty check"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        self.assertTrue(attn_ranges.is_empty())

        attn_ranges.append(AttnRange(0, 10))
        self.assertFalse(attn_ranges.is_empty())

    def test_len(self):
        """Test length"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        self.assertEqual(len(attn_ranges), 0)

        attn_ranges.append(AttnRange(0, 10))
        self.assertEqual(len(attn_ranges), 1)

        attn_ranges.append(AttnRange(20, 30))
        self.assertEqual(len(attn_ranges), 2)

    def test_getitem(self):
        """Test index access"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))
        attn_ranges.append(AttnRange(40, 50))

        # Test single index
        self.assertEqual(attn_ranges[0], AttnRange(0, 10))
        self.assertEqual(attn_ranges[1], AttnRange(20, 30))
        self.assertEqual(attn_ranges[2], AttnRange(40, 50))

        # Test slicing
        slice_result = attn_ranges[1:3]
        self.assertTrue(isinstance(slice_result, AttnRanges))
        self.assertEqual(len(slice_result), 2)
        self.assertEqual(slice_result[0], AttnRange(20, 30))
        self.assertEqual(slice_result[1], AttnRange(40, 50))

        # Test negative index
        self.assertEqual(attn_ranges[-1], AttnRange(40, 50))
        self.assertEqual(attn_ranges[-2], AttnRange(20, 30))

    def test_setitem(self):
        """Test index assignment"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        # Test single index assignment
        attn_ranges[0] = AttnRange(5, 15)
        self.assertEqual(attn_ranges[0], AttnRange(5, 15))

        # Test slice assignment
        new_ranges = AttnRanges()
        new_ranges.append(AttnRange(25, 35))
        new_ranges.append(AttnRange(45, 55))

        attn_ranges[1:3] = new_ranges
        self.assertEqual(len(attn_ranges), 3)
        self.assertEqual(attn_ranges[1], AttnRange(25, 35))
        self.assertEqual(attn_ranges[2], AttnRange(45, 55))

    def test_iter(self):
        """Test iteration"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        ranges_list = list(attn_ranges)
        self.assertEqual(len(ranges_list), 2)
        self.assertEqual(ranges_list[0], AttnRange(0, 10))
        self.assertEqual(ranges_list[1], AttnRange(20, 30))

    def test_eq(self):
        """Test equality"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges1 = AttnRanges()
        attn_ranges1.append(AttnRange(0, 10))
        attn_ranges1.append(AttnRange(20, 30))

        attn_ranges2 = AttnRanges()
        attn_ranges2.append(AttnRange(0, 10))
        attn_ranges2.append(AttnRange(20, 30))

        self.assertEqual(attn_ranges1, attn_ranges2)

        # Test different content
        attn_ranges3 = AttnRanges()
        attn_ranges3.append(AttnRange(0, 10))
        attn_ranges3.append(AttnRange(25, 35))

        self.assertNotEqual(attn_ranges1, attn_ranges3)

        # Test different type objects
        self.assertNotEqual(attn_ranges1, "not a rect ranges")

    def test_hash(self):
        """Test hashing"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges1 = AttnRanges()
        attn_ranges1.append(AttnRange(0, 10))
        attn_ranges1.append(AttnRange(20, 30))

        attn_ranges2 = AttnRanges()
        attn_ranges2.append(AttnRange(0, 10))
        attn_ranges2.append(AttnRange(20, 30))

        # Same content should have same hash value
        self.assertEqual(hash(attn_ranges1), hash(attn_ranges2))

        # Hash value should be hashable
        hash_value = hash(attn_ranges1)
        self.assertTrue(isinstance(hash_value, int))

    def test_repr(self):
        """Test string representation"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        # Test empty ranges
        empty_ranges = AttnRanges()
        self.assertEqual(repr(empty_ranges), "[[,)]")

        # Test non-empty ranges
        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        expected_repr = "[[0, 10), [20, 30)]"
        self.assertEqual(repr(attn_ranges), expected_repr)

    def test_edge_cases(self):
        """Test edge cases"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        # Test various operations on empty ranges
        empty_ranges = AttnRanges()

        # Empty ranges should be valid
        self.assertTrue(empty_ranges.is_valid())

        # Empty ranges should have special string representation
        self.assertEqual(repr(empty_ranges), "[[,)]")

        # Empty ranges should have computable hash value
        hash_value = hash(empty_ranges)
        self.assertTrue(isinstance(hash_value, int))

        # Test single range
        single_range = AttnRanges()
        single_range.append(AttnRange(5, 10))

        self.assertEqual(len(single_range), 1)
        self.assertEqual(single_range[0], AttnRange(5, 10))
        self.assertFalse(single_range.is_empty())

    def test_invalid_operations(self):
        """Test invalid operations"""
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges()
        attn_ranges.append(AttnRange(0, 10))
        attn_ranges.append(AttnRange(20, 30))

        # Test index out of bounds
        with pytest.raises(IndexError):
            _ = attn_ranges[5]

        # Test slice assignment length mismatch
        new_ranges = AttnRanges()
        new_ranges.append(AttnRange(25, 35))

        with pytest.raises(AssertionError):
            attn_ranges[1:3] = new_ranges  # Length mismatch

    def test_make_ranges_local(self):
        AttnRanges = self.types.AttnRanges

        # ---------    case1: w/o truncate     --------- #
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_ranges = AttnRanges.from_ranges([(2, 8), (25, 32), (42, 45)])
        local_ranges = ranges.make_ranges_local(other_ranges)
        self.assertEqual(
            local_ranges, AttnRanges.from_ranges([(2, 8), (15, 22), (27, 30)])
        )

    def test_make_ranges_local_raises(self):
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_ranges = AttnRanges.from_ranges([(2, 8), (25, 36), (42, 45)])
        with pytest.raises(
            ValueError,
            match=(
                "The attn_range \\[25, 36\\) is not in the \\(even merged\\)"
                " attn_ranges \\[\\[0, 10\\), \\[20, 35\\), \\[40, 50\\)\\]"
            ),
        ):
            ranges.make_ranges_local(other_ranges)

    def test_make_range_local(self):
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        # ---------    case1: w/o truncate     --------- #
        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])
        other_range = AttnRange(23, 32)
        local_range, _ = ranges.make_range_local(other_range)
        self.assertEqual(local_range, AttnRange(13, 22))

    def test_make_range_local_raises(self):
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges.from_ranges([(0, 10), (20, 30), (30, 35), (40, 50)])

        # test out-of-bounds range
        invalid_range = AttnRange(60, 70)
        with pytest.raises(
            ValueError,
            match=(
                "The attn_range \\[60, 70\\) is not in the \\(even merged\\)"
                " attn_ranges \\[\\[0, 10\\), \\[20, 35\\), \\[40, 50\\)\\]"
            ),
        ):
            ranges.make_range_local(invalid_range)

        # test partially overlapping range
        invalid_range = AttnRange(5, 25)
        with pytest.raises(
            ValueError,
            match=(
                "The attn_range \\[5, 25\\) is not in the \\(even merged\\)"
                " attn_ranges \\[\\[0, 10\\), \\[20, 35\\), \\[40, 50\\)\\]"
            ),
        ):
            ranges.make_range_local(invalid_range)

        # test multiple overlapping ranges
        invalid_range = AttnRange(25, 45)
        with pytest.raises(
            ValueError,
            match=(
                "The attn_range \\[25, 45\\) is not in the \\(even merged\\)"
                " attn_ranges \\[\\[0, 10\\), \\[20, 35\\), \\[40, 50\\)\\]"
            ),
        ):
            ranges.make_range_local(invalid_range)

    def test_find_hole_range(self):
        AttnRanges = self.types.AttnRanges

        ranges1_list = [
            [(0, 10), (20, 30), (40, 50)],
            [
                [0, 28800],
                [0, 28800],
                [0, 28800],
                [0, 57600],
                [0, 57600],
                [0, 57600],
                [0, 57600],
                [0, 86400],
                [0, 86400],
                [0, 86400],
                [0, 115200],
                [0, 115200],
                [0, 115200],
                [0, 115200],
                [115200, 144000],
                [115200, 144000],
                [115200, 144000],
                [115200, 144000],
                [144000, 172800],
                [144000, 172800],
                [144000, 172800],
                [144000, 172800],
                [172800, 201600],
                [172800, 201600],
                [172800, 201600],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [201600, 230400],
                [230400, 234040],
                [230400, 234040],
                [234040, 237680],
            ],
        ]

        ranges2_list = [
            [(5, 25), (35, 45)],
            [
                [4096, 6144],
                [12288, 14336],
                [20480, 22528],
                [30720, 32768],
                [38912, 40960],
                [49152, 51200],
                [57344, 57600],
                [57600, 59392],
                [77824, 79872],
                [83968, 86016],
                [92160, 94208],
                [94208, 96256],
                [104448, 106496],
                [112640, 114688],
                [118784, 120832],
                [120832, 122880],
                [131072, 133120],
                [139264, 141312],
                [147456, 149504],
                [149504, 151552],
                [157696, 159744],
                [165888, 167936],
                [182272, 184320],
                [186368, 188416],
                [194560, 196608],
                [202752, 204800],
                [210944, 212992],
                [219136, 221184],
                [227328, 229376],
                [229376, 230400],
                [230400, 231424],
                [233472, 234040],
                [234040, 235520],
            ],
        ]
        ans_list = [
            [(0, 5), (25, 30), (45, 50)],
            [
                [0, 4096],
                [6144, 12288],
                [14336, 20480],
                [22528, 30720],
                [32768, 38912],
                [40960, 49152],
                [51200, 57344],
                [59392, 77824],
                [79872, 83968],
                [86016, 92160],
                [96256, 104448],
                [106496, 112640],
                [114688, 118784],
                [122880, 131072],
                [133120, 139264],
                [141312, 147456],
                [151552, 157696],
                [159744, 165888],
                [167936, 182272],
                [184320, 186368],
                [188416, 194560],
                [196608, 202752],
                [204800, 210944],
                [212992, 219136],
                [221184, 227328],
                [231424, 233472],
                [235520, 237680],
            ],
        ]
        for ranges1_list, ranges2_list, ans_list in zip(
            ranges1_list, ranges2_list, ans_list
        ):
            ranges1 = AttnRanges.from_ranges(ranges1_list)
            ranges2 = AttnRanges.from_ranges(ranges2_list)
            hole_ranges = ranges1.find_hole_ranges(ranges2)
            self.assertEqual(hole_ranges, AttnRanges.from_ranges(ans_list))

    def test_find_overlap_range(self):
        AttnRanges = self.types.AttnRanges

        ranges1_list = [
            [(0, 10), (20, 30), (40, 50)],
        ]
        ranges2_list = [
            [(5, 25), (35, 45)],
        ]
        ans_list = [
            [(5, 10), (20, 25), (40, 45)],
        ]
        for ranges1_list, ranges2_list, ans_list in zip(
            ranges1_list, ranges2_list, ans_list
        ):
            ranges1 = AttnRanges.from_ranges(ranges1_list)
            ranges2 = AttnRanges.from_ranges(ranges2_list)
            overlap_ranges = ranges1.find_overlap_ranges(ranges2)
            self.assertEqual(overlap_ranges, AttnRanges.from_ranges(ans_list))

    def test_from_cu_seqlens(self):
        AttnRanges = self.types.AttnRanges

        # ---------    valid cu_seqlens     --------- #
        cu_seqlens = [0, 10, 20, 30, 40, 50]
        seq_len = 50
        ranges = AttnRanges.from_cu_seqlens(cu_seqlens, seq_len)
        self.assertEqual(
            ranges,
            AttnRanges.from_ranges([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]),
        )

        # ---------    empty cu_seqlens always valid     --------- #
        self.assertEqual(AttnRanges.from_cu_seqlens([], 0), AttnRanges())
        self.assertEqual(AttnRanges.from_cu_seqlens([], 5), AttnRanges())

        # ---------    more valid cu_seqlens     --------- #
        cu_seqlens = [0, 23, 49, 58, 89]
        ranges = AttnRanges.from_cu_seqlens(cu_seqlens, 89)
        self.assertEqual(ranges.to_cu_seqlens(89), cu_seqlens)

        cu_seqlens = [0, 89]
        ranges = AttnRanges.from_cu_seqlens(cu_seqlens, 89)
        self.assertEqual(ranges.to_cu_seqlens(89), cu_seqlens)

        cu_seqlens = [0]
        ranges = AttnRanges.from_cu_seqlens(cu_seqlens, 0)
        self.assertEqual(ranges.to_cu_seqlens(0), cu_seqlens)

        # ---------    invalid cu_seqlens w/o starting from 0     --------- #
        with pytest.raises(ValueError):
            AttnRanges.from_cu_seqlens([23, 49, 58, 89], 89)

        # ---------    invalid cu_seqlens w/o monotonically increasing     --------- #
        with pytest.raises(ValueError):
            AttnRanges.from_cu_seqlens([0, 50, 49, 58, 89], 89)
        with pytest.raises(ValueError):
            AttnRanges.from_cu_seqlens([0, 49, 49, 58, 89], 89)

        # ---------    invalid cu_seqlens w/o ending at seq_len     --------- #
        with pytest.raises(ValueError):
            AttnRanges.from_cu_seqlens([0, 23, 49, 58, 89], 90)

    def test_sort(self):
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges.from_ranges(
            [(0, 10), (30, 35), (20, 30), (10, 25), (40, 50)]
        )
        self.assertFalse(ranges.is_sorted())
        sorted_ranges = ranges.sort()
        self.assertEqual(
            sorted_ranges,
            AttnRanges.from_ranges([(0, 10), (10, 25), (20, 30), (30, 35), (40, 50)]),
        )
        self.assertTrue(sorted_ranges.is_sorted())

    def test_merge(self):
        AttnRanges = self.types.AttnRanges

        # case1
        ranges = AttnRanges.from_ranges(
            [(0, 10), (10, 20), (20, 30), (30, 35), (40, 50)]
        )
        self.assertFalse(ranges.is_merged())
        merged_ranges = ranges.merge()
        self.assertEqual(merged_ranges, AttnRanges.from_ranges([(0, 35), (40, 50)]))
        self.assertTrue(merged_ranges.is_merged())

        # case2
        ranges = AttnRanges.from_ranges([(0, 10), (6, 8)])
        self.assertFalse(ranges.is_merged())
        merged_ranges = ranges.merge()
        self.assertEqual(merged_ranges, AttnRanges.from_ranges([(0, 10)]))
        self.assertTrue(merged_ranges.is_merged())

    def test_merge_with_split_alignment(self):
        AttnRanges = self.types.AttnRanges

        # case1: basic alignment
        ranges = AttnRanges.from_ranges([(1, 9), (11, 19)])
        # alignment=10 -> [(0, 10), (10, 20)] -> merged to [(0, 20)]
        merged = ranges.merge_with_split_alignment(split_alignment=10)
        self.assertEqual(merged, AttnRanges.from_ranges([(0, 20)]))

        # case2: disjoint after alignment
        ranges = AttnRanges.from_ranges([(1, 9), (21, 29)])
        # alignment=10 -> [(0, 10), (20, 30)]
        merged = ranges.merge_with_split_alignment(split_alignment=10)
        self.assertEqual(merged, AttnRanges.from_ranges([(0, 10), (20, 30)]))

        # case3: overlap after alignment
        ranges = AttnRanges.from_ranges([(5, 15), (12, 25)])
        # alignment=10 -> [(0, 20), (10, 30)] -> merged to [(0, 30)]
        merged = ranges.merge_with_split_alignment(split_alignment=10)
        self.assertEqual(merged, AttnRanges.from_ranges([(0, 30)]))

        # case4: already aligned
        ranges = AttnRanges.from_ranges([(0, 10), (10, 20), (25, 30)])
        merged = ranges.merge_with_split_alignment(split_alignment=5)
        self.assertEqual(merged, AttnRanges.from_ranges([(0, 20), (25, 30)]))

    def test_non_overlap(self):
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges.from_ranges([(8, 14), (5, 10)])
        self.assertFalse(attn_ranges.is_non_overlap())

        attn_ranges = AttnRanges.from_ranges([(8, 14), (14, 15)])
        self.assertTrue(attn_ranges.is_non_overlap())

        attn_ranges = AttnRanges.from_ranges([(8, 14), (3, 7)])
        self.assertTrue(attn_ranges.is_non_overlap())

    def test_chunk(self):
        AttnRanges = self.types.AttnRanges

        # ----    case0: raise assert error when the ranges is overlapped   --- #

        attn_ranges = AttnRanges.from_ranges([(8, 14), (5, 10)])

        with pytest.raises(
            AssertionError,
            match="non-overlap",
        ):
            attn_ranges.chunk(chunk_size=8)

        # ---------    case1: a single range     --------- #
        attn_ranges = AttnRanges.from_ranges([(5, 10)])

        # chunk with a large chunk size
        self.assertEqual(attn_ranges.chunk(chunk_size=8), [attn_ranges])

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [AttnRanges.from_ranges([(5, 9)]), AttnRanges.from_ranges([(9, 10)])],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=1),
            [
                AttnRanges.from_ranges([(5, 6)]),
                AttnRanges.from_ranges([(6, 7)]),
                AttnRanges.from_ranges([(7, 8)]),
                AttnRanges.from_ranges([(8, 9)]),
                AttnRanges.from_ranges([(9, 10)]),
            ],
        )

        # ---------    case2: several small ranges     --------- #
        attn_ranges = AttnRanges.from_ranges([(2, 4), (6, 9), (10, 12), (14, 17)])

        # chunk with a xlarge chunk size
        self.assertEqual(attn_ranges.chunk(chunk_size=16), [attn_ranges])

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(2, 4), (6, 9), (10, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(2, 4), (6, 8)]),
                AttnRanges.from_ranges([(8, 9), (10, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(2, 4)]),
                AttnRanges.from_ranges([(6, 8)]),
                AttnRanges.from_ranges([(8, 9), (10, 11)]),
                AttnRanges.from_ranges([(11, 12), (14, 15)]),
                AttnRanges.from_ranges([(15, 17)]),
            ],
        )

        # -----    case3: several small ranges + one long range   ------ #
        attn_ranges = AttnRanges.from_ranges([(3, 6), (8, 9), (10, 12), (14, 21)])

        # chunk with a xlarge chunk size
        self.assertEqual(attn_ranges.chunk(chunk_size=16), [attn_ranges])

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(3, 6), (8, 9), (10, 12), (14, 16)]),
                AttnRanges.from_ranges([(16, 21)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(3, 6), (8, 9)]),
                AttnRanges.from_ranges([(10, 12), (14, 16)]),
                AttnRanges.from_ranges([(16, 20)]),
                AttnRanges.from_ranges([(20, 21)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(3, 5)]),
                AttnRanges.from_ranges([(5, 6), (8, 9)]),
                AttnRanges.from_ranges([(10, 12)]),
                AttnRanges.from_ranges([(14, 16)]),
                AttnRanges.from_ranges([(16, 18)]),
                AttnRanges.from_ranges([(18, 20)]),
                AttnRanges.from_ranges([(20, 21)]),
            ],
        )

        # -----    case4: one long range + several small ranges   ------ #
        attn_ranges = AttnRanges.from_ranges([(1, 2), (4, 12), (13, 15), (16, 19)])

        # chunk with a xlarge chunk size
        self.assertEqual(attn_ranges.chunk(chunk_size=16), [attn_ranges])

        # chunk with a large chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=8),
            [
                AttnRanges.from_ranges([(1, 2), (4, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 15), (16, 19)]),
            ],
        )

        # chunk with a medium chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=4),
            [
                AttnRanges.from_ranges([(1, 2), (4, 7)]),
                AttnRanges.from_ranges([(7, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 15), (16, 17)]),
                AttnRanges.from_ranges([(17, 19)]),
            ],
        )

        # chunk with a small chunk size
        self.assertEqual(
            attn_ranges.chunk(chunk_size=2),
            [
                AttnRanges.from_ranges([(1, 2), (4, 5)]),
                AttnRanges.from_ranges([(5, 7)]),
                AttnRanges.from_ranges([(7, 9)]),
                AttnRanges.from_ranges([(9, 11)]),
                AttnRanges.from_ranges([(11, 12), (13, 14)]),
                AttnRanges.from_ranges([(14, 15), (16, 17)]),
                AttnRanges.from_ranges([(17, 19)]),
            ],
        )

    def test_truncate(self):
        AttnRanges = self.types.AttnRanges

        attn_ranges = AttnRanges.from_ranges([(9, 15), (20, 30), (30, 35), (40, 50)])

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, attn_ranges)

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 57
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, attn_ranges)

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 25, None
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(
            trunc_ranges, AttnRanges.from_ranges([(25, 30), (30, 35), (40, 50)])
        )

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 31
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(
            trunc_ranges, AttnRanges.from_ranges([(9, 15), (20, 30), (30, 31)])
        )

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 25, 31
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_ranges, AttnRanges.from_ranges([(25, 30), (30, 31)]))

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 0, 5
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 53, 64
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

        # -----    case8: with left+right truncate but falling into a hole   ---- #
        trunc_start, trunc_end = 36, 39
        trunc_ranges = attn_ranges.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_ranges.is_empty())

    def test_intersect_size(self):
        AttnRanges = self.types.AttnRanges

        # Test empty AttnRanges
        empty_ranges = AttnRanges()
        self.assertEqual(empty_ranges.intersect_size(), 0)

        # Test single range case
        single_range = AttnRanges.from_ranges([(5, 10)])
        self.assertEqual(single_range.intersect_size(), 0)

        # Test multiple non-overlapping ranges
        non_overlap_ranges = AttnRanges.from_ranges([(5, 10), (15, 20), (25, 30)])
        self.assertEqual(non_overlap_ranges.intersect_size(), 0)

        # Test two partially overlapping ranges
        two_overlap_ranges = AttnRanges.from_ranges([(5, 15), (10, 20)])
        # 1 * [10, 15) = 5 = 5
        self.assertEqual(two_overlap_ranges.intersect_size(), 5)

        # Test three overlapping ranges
        three_overlap_ranges = AttnRanges.from_ranges([(5, 15), (10, 20), (12, 25)])
        # 1 * [10, 15) + 1 * [12, 20) = 5 + 8 = 13
        self.assertEqual(three_overlap_ranges.intersect_size(), 13)

        three_overlap_ranges_same = AttnRanges.from_ranges([(5, 15), (5, 15), (5, 15)])
        # 2 * [5, 15) = 10 + 10 = 20
        self.assertEqual(three_overlap_ranges_same.intersect_size(), 20)

        four_overlap_ranges_same = AttnRanges.from_ranges(
            [(5, 15), (5, 15), (5, 15), (5, 15)]
        )
        # 3 * [5, 15) = 10 + 10 + 10 = 30
        self.assertEqual(four_overlap_ranges_same.intersect_size(), 30)

        # Test contained ranges
        contained_ranges = AttnRanges.from_ranges([(5, 20), (8, 15)])
        # 1 * [8, 15) = 7 = 7
        self.assertEqual(contained_ranges.intersect_size(), 7)

        # Test multiple complex overlapping ranges
        complex_ranges = AttnRanges.from_ranges(
            [(0, 10), (5, 15), (8, 20), (18, 25), (22, 30)]
        )
        # 1 * [5, 10) + 1 * [8, 15) + 1 * [18, 20) + 1 * [22, 25) = 5 + 7 + 2 + 3 = 17
        self.assertEqual(complex_ranges.intersect_size(), 17)

        # Test adjacent but non-overlapping ranges
        adjacent_ranges = AttnRanges.from_ranges([(5, 10), (10, 15), (15, 20)])
        # 0 * [] = 0 = 0
        self.assertEqual(adjacent_ranges.intersect_size(), 0)

        # Test three ranges overlapping at a single point
        point_overlap_ranges = AttnRanges.from_ranges([(0, 10), (5, 15), (5, 20)])
        # 2 * [5, 10) + 1 * [5, 15) = 10 + 5 = 15
        self.assertEqual(point_overlap_ranges.intersect_size(), 15)

    def test_intersect_size_with(self):
        AttnRanges = self.types.AttnRanges

        # Test two empty ranges
        empty_ranges1 = AttnRanges()
        empty_ranges2 = AttnRanges()
        self.assertEqual(empty_ranges1.intersect_size_with(empty_ranges2), 0)

        # Test one empty range and one non-empty range
        non_empty_ranges = AttnRanges.from_ranges([(5, 10)])
        self.assertEqual(empty_ranges1.intersect_size_with(non_empty_ranges), 0)
        self.assertEqual(non_empty_ranges.intersect_size_with(empty_ranges1), 0)

        # Test two non-overlapping range sets
        ranges1 = AttnRanges.from_ranges([(5, 10), (15, 20)])
        ranges2 = AttnRanges.from_ranges([(25, 30), (35, 40)])
        self.assertEqual(ranges1.intersect_size_with(ranges2), 0)

        # Test partially overlapping range sets
        ranges1 = AttnRanges.from_ranges([(5, 15), (25, 35)])
        ranges2 = AttnRanges.from_ranges([(10, 20), (30, 40)])
        # Overlap regions [10, 15) and [30, 35), total 5 + 5 = 10
        self.assertEqual(ranges1.intersect_size_with(ranges2), 10)

        # Test fully contained range sets
        ranges1 = AttnRanges.from_ranges([(5, 25)])
        ranges2 = AttnRanges.from_ranges([(10, 20)])
        # Overlap region [10, 20), total 10
        self.assertEqual(ranges1.intersect_size_with(ranges2), 10)
        self.assertEqual(ranges2.intersect_size_with(ranges1), 10)

        # Test multiple complex overlapping ranges
        ranges1 = AttnRanges.from_ranges([(0, 10), (15, 25), (30, 40)])
        ranges2 = AttnRanges.from_ranges([(5, 20), (22, 35)])
        # Overlap regions [5, 10), [15, 20), [22, 25), and [30, 35), total 5 + 5 + 3 + 5 = 18
        self.assertEqual(ranges1.intersect_size_with(ranges2), 18)

        # Test identical range sets
        ranges = AttnRanges.from_ranges([(5, 10), (15, 20)])
        self.assertEqual(ranges.intersect_size_with(ranges), 10)

        # Test adjacent but non-overlapping ranges
        ranges1 = AttnRanges.from_ranges([(5, 10), (20, 25)])
        ranges2 = AttnRanges.from_ranges([(10, 15), (15, 20)])
        self.assertEqual(ranges1.intersect_size_with(ranges2), 0)

    def test_union_size(self):
        AttnRanges = self.types.AttnRanges

        # Test empty ranges
        empty_ranges1 = AttnRanges()
        empty_ranges2 = AttnRanges()
        self.assertEqual(empty_ranges1.union_size(), 0)
        self.assertEqual(empty_ranges2.union_size(), 0)

        # TODO(littsk): more tests

    def test_union_size_with(self):
        AttnRanges = self.types.AttnRanges

        r1 = AttnRanges.from_ranges([(0, 10), (20, 30)])
        r2 = AttnRanges.from_ranges([(5, 15)])
        self.assertEqual(r1.union_size_with(r2), 30)

        empty = AttnRanges()
        self.assertEqual(r1.union_size_with(empty), 20)
        self.assertEqual(empty.union_size_with(r1), 20)
        self.assertEqual(empty.union_size_with(empty), 0)

    def test_check_valid_and_is_valid(self):
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        valid = AttnRanges.from_ranges([(0, 10), (5, 15)])
        self.assertTrue(valid.is_valid())
        valid.check_valid()

        empty = AttnRanges()
        self.assertTrue(empty.is_valid())
        empty.check_valid()

        if self.backend == "python":
            invalid = AttnRanges()
            bad_range = AttnRange(0, 0)
            bad_range._start = 10
            bad_range._end = 5
            invalid._ranges.append(bad_range)
            self.assertFalse(invalid.is_valid())
            with pytest.raises((ValueError, RuntimeError)):
                invalid.check_valid()

    def test_insert_with_check(self):
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges()
        ranges.insert(0, AttnRange(0, 10), check=True)
        self.assertEqual(len(ranges), 1)

    def test_extend_with_check(self):
        AttnRanges = self.types.AttnRanges

        r1 = AttnRanges.from_ranges([(0, 10)])
        r2 = AttnRanges.from_ranges([(20, 30)])
        r1.extend(r2, check=True)
        self.assertEqual(len(r1), 2)

    def test_pop_empty_raises(self):
        AttnRanges = self.types.AttnRanges

        empty = AttnRanges()
        with pytest.raises(IndexError, match="pop from empty"):
            empty.pop()

    def test_clear_empty(self):
        AttnRange = self.types.AttnRange
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges()
        ranges.append(AttnRange(0, 10))
        ranges.append(AttnRange(5, 5))
        ranges.append(AttnRange(20, 30))
        ranges.append(AttnRange(7, 7))

        cleared = ranges.clear_empty()
        self.assertEqual(len(cleared), 2)
        self.assertEqual(cleared[0], AttnRange(0, 10))
        self.assertEqual(cleared[1], AttnRange(20, 30))

        all_empty = AttnRanges()
        all_empty.append(AttnRange(0, 0))
        self.assertEqual(len(all_empty.clear_empty()), 0)

    def test_empty_start_end_raises(self):
        AttnRanges = self.types.AttnRanges

        empty = AttnRanges()
        with pytest.raises((ValueError, RuntimeError)):
            _ = empty.start
        with pytest.raises((ValueError, RuntimeError)):
            _ = empty.end

    def test_to_tensor_empty(self):
        AttnRanges = self.types.AttnRanges

        empty = AttnRanges()
        t = empty.to_tensor()
        self.assertEqual(t.shape, (0, 2))
        self.assertEqual(t.dtype, torch.int32)

    def test_is_merged_unsorted(self):
        AttnRanges = self.types.AttnRanges

        unsorted = AttnRanges.from_ranges([(10, 20), (0, 5)])
        self.assertFalse(unsorted.is_merged())

    def test_is_cu_seqlens_not_start_zero(self):
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges.from_ranges([(5, 10), (10, 20)])
        self.assertFalse(ranges.is_cu_seqlens(20))

    @parameterize(
        "case",
        [
            ([(0, 2048), (2048, 4096)], 4096, True),
            ([(0, 8192)] * 4, 8192, True),
            ([(2048, 4096), (0, 2048)], 4096, True),
            ([(0, 4096), (6144, 8192)], 8192, False),
            ([(0, 4096)], 8192, False),
            ([(2048, 8192)], 8192, False),
            ([(0, 0), (0, 8192)], 8192, True),
            ([(0, 100)], 50, False),
            ([], 0, True),
            ([], 5, False),
        ],
    )
    def test_is_full_coverage(self, case):
        ranges_list, seqlen, expected = case
        if self.backend == "cpp":
            pytest.skip(
                "is_full_coverage is python-side only until magi_attn_ext syncs it"
            )
        AttnRanges = self.types.AttnRanges

        self.assertIs(
            AttnRanges.from_ranges(ranges_list).is_full_coverage(seqlen), expected
        )

    def test_max_seqlen(self):
        AttnRanges = self.types.AttnRanges

        ranges = AttnRanges.from_ranges([(0, 5), (10, 30), (40, 45)])
        self.assertEqual(ranges.max_seqlen, 20)

        empty = AttnRanges()
        self.assertEqual(empty.max_seqlen, 0)


class TestCppAttnRanges(TestAttnRanges):
    backend = "cpp"


if __name__ == "__main__":
    unittest.main()
