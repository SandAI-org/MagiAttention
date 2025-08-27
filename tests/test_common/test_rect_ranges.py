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

from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.common.rect_range import AttnRectRange
from magi_attention.common.rect_ranges import AttnRectRanges


class TestAttnRectRanges(TestCase):
    def test_init(self):
        """Test initialization"""
        rect_ranges = AttnRectRanges()
        self.assertEqual(len(rect_ranges), 0)
        self.assertTrue(rect_ranges.is_empty())

    def test_is_valid(self):
        """Test validity check"""
        # Empty ranges are always valid
        rect_ranges = AttnRectRanges()
        self.assertTrue(rect_ranges.is_valid())

        # Add valid ranges
        rect_ranges.append(AttnRectRange(0, 10))
        self.assertTrue(rect_ranges.is_valid())

    def test_check_valid(self):
        """Test validity verification"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        # All ranges are valid, should pass check
        rect_ranges.check_valid()

    def test_append(self):
        """Test adding ranges"""
        rect_ranges = AttnRectRanges()

        # Test basic append
        rect_ranges.append(AttnRectRange(0, 10))
        self.assertEqual(len(rect_ranges), 1)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))

        # Test append with check
        rect_ranges.append(AttnRectRange(10, 20), check=True)
        self.assertEqual(len(rect_ranges), 2)

    def test_from_ranges(self):
        """Test creating AttnRectRanges from different types"""
        # Create from AttnRectRanges
        original = AttnRectRanges()
        original.append(AttnRectRange(0, 10))
        original.append(AttnRectRange(20, 30))

        copied = AttnRectRanges.from_ranges(original)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)  # Should be deep copy

        # Create from AttnRectRange list
        rect_range_list = [AttnRectRange(0, 10), AttnRectRange(20, 30)]
        rect_ranges = AttnRectRanges.from_ranges(rect_range_list)
        self.assertEqual(len(rect_ranges), 2)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))

        # Create from AttnRange list
        attn_range_list = [AttnRange(0, 10), AttnRange(20, 30)]
        rect_ranges = AttnRectRanges.from_ranges(attn_range_list)
        self.assertEqual(len(rect_ranges), 2)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))

        # Create from NaiveRanges
        naive_ranges = [(0, 10), (20, 30)]
        rect_ranges = AttnRectRanges.from_ranges(naive_ranges)
        self.assertEqual(len(rect_ranges), 2)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))

        # Create from AttnRanges
        attn_ranges = AttnRanges.from_ranges([(0, 10), (20, 30)])
        rect_ranges = AttnRectRanges.from_ranges(attn_ranges)
        self.assertEqual(len(rect_ranges), 2)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))

        # Test creation with check
        valid_ranges = [(0, 10), (20, 30)]
        rect_ranges = AttnRectRanges.from_ranges(valid_ranges, check=True)
        self.assertEqual(len(rect_ranges), 2)
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))

    def test_to_naive_ranges(self):
        """Test conversion to naive ranges"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        naive_ranges = rect_ranges.to_naive_ranges()
        expected = [(0, 10), (20, 30)]
        self.assertEqual(naive_ranges, expected)

    def test_is_empty(self):
        """Test empty check"""
        rect_ranges = AttnRectRanges()
        self.assertTrue(rect_ranges.is_empty())

        rect_ranges.append(AttnRectRange(0, 10))
        self.assertFalse(rect_ranges.is_empty())

    def test_len(self):
        """Test length"""
        rect_ranges = AttnRectRanges()
        self.assertEqual(len(rect_ranges), 0)

        rect_ranges.append(AttnRectRange(0, 10))
        self.assertEqual(len(rect_ranges), 1)

        rect_ranges.append(AttnRectRange(20, 30))
        self.assertEqual(len(rect_ranges), 2)

    def test_getitem(self):
        """Test index access"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))
        rect_ranges.append(AttnRectRange(40, 50))

        # Test single index
        self.assertEqual(rect_ranges[0], AttnRectRange(0, 10))
        self.assertEqual(rect_ranges[1], AttnRectRange(20, 30))
        self.assertEqual(rect_ranges[2], AttnRectRange(40, 50))

        # Test slicing
        slice_result = rect_ranges[1:3]
        self.assertIsInstance(slice_result, AttnRectRanges)
        self.assertEqual(len(slice_result), 2)
        self.assertEqual(slice_result[0], AttnRectRange(20, 30))
        self.assertEqual(slice_result[1], AttnRectRange(40, 50))

        # Test negative index
        self.assertEqual(rect_ranges[-1], AttnRectRange(40, 50))
        self.assertEqual(rect_ranges[-2], AttnRectRange(20, 30))

    def test_setitem(self):
        """Test index assignment"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        # Test single index assignment
        rect_ranges[0] = AttnRectRange(5, 15)
        self.assertEqual(rect_ranges[0], AttnRectRange(5, 15))

        # Test slice assignment
        new_ranges = AttnRectRanges()
        new_ranges.append(AttnRectRange(25, 35))
        new_ranges.append(AttnRectRange(45, 55))

        rect_ranges[1:3] = new_ranges
        self.assertEqual(len(rect_ranges), 3)
        self.assertEqual(rect_ranges[1], AttnRectRange(25, 35))
        self.assertEqual(rect_ranges[2], AttnRectRange(45, 55))

    def test_iter(self):
        """Test iteration"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        ranges_list = list(rect_ranges)
        self.assertEqual(len(ranges_list), 2)
        self.assertEqual(ranges_list[0], AttnRectRange(0, 10))
        self.assertEqual(ranges_list[1], AttnRectRange(20, 30))

    def test_eq(self):
        """Test equality"""
        rect_ranges1 = AttnRectRanges()
        rect_ranges1.append(AttnRectRange(0, 10))
        rect_ranges1.append(AttnRectRange(20, 30))

        rect_ranges2 = AttnRectRanges()
        rect_ranges2.append(AttnRectRange(0, 10))
        rect_ranges2.append(AttnRectRange(20, 30))

        self.assertEqual(rect_ranges1, rect_ranges2)

        # Test different content
        rect_ranges3 = AttnRectRanges()
        rect_ranges3.append(AttnRectRange(0, 10))
        rect_ranges3.append(AttnRectRange(25, 35))

        self.assertNotEqual(rect_ranges1, rect_ranges3)

        # Test different type objects
        self.assertNotEqual(rect_ranges1, "not a rect ranges")

    def test_hash(self):
        """Test hashing"""
        rect_ranges1 = AttnRectRanges()
        rect_ranges1.append(AttnRectRange(0, 10))
        rect_ranges1.append(AttnRectRange(20, 30))

        rect_ranges2 = AttnRectRanges()
        rect_ranges2.append(AttnRectRange(0, 10))
        rect_ranges2.append(AttnRectRange(20, 30))

        # Same content should have same hash value
        self.assertEqual(hash(rect_ranges1), hash(rect_ranges2))

        # Hash value should be hashable
        hash_value = hash(rect_ranges1)
        self.assertIsInstance(hash_value, int)

    def test_repr(self):
        """Test string representation"""
        # Test empty ranges
        empty_ranges = AttnRectRanges()
        self.assertEqual(repr(empty_ranges), "[[,)]")

        # Test non-empty ranges
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        expected_repr = "[[0, 10), [20, 30)]"
        self.assertEqual(repr(rect_ranges), expected_repr)

    def test_edge_cases(self):
        """Test edge cases"""
        # Test various operations on empty ranges
        empty_ranges = AttnRectRanges()

        # Empty ranges should be valid
        self.assertTrue(empty_ranges.is_valid())

        # Empty ranges should have special string representation
        self.assertEqual(repr(empty_ranges), "[[,)]")

        # Empty ranges should have computable hash value
        hash_value = hash(empty_ranges)
        self.assertIsInstance(hash_value, int)

        # Test single range
        single_range = AttnRectRanges()
        single_range.append(AttnRectRange(5, 10))

        self.assertEqual(len(single_range), 1)
        self.assertEqual(single_range[0], AttnRectRange(5, 10))
        self.assertFalse(single_range.is_empty())

    def test_invalid_operations(self):
        """Test invalid operations"""
        rect_ranges = AttnRectRanges()
        rect_ranges.append(AttnRectRange(0, 10))
        rect_ranges.append(AttnRectRange(20, 30))

        # Test index out of bounds
        with self.assertRaises(IndexError):
            _ = rect_ranges[5]

        # Test slice assignment length mismatch
        new_ranges = AttnRectRanges()
        new_ranges.append(AttnRectRange(25, 35))

        with self.assertRaises(AssertionError):
            rect_ranges[1:3] = new_ranges  # Length mismatch


if __name__ == "__main__":
    unittest.main()
