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

import random
import unittest
from unittest import TestCase

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.rectangle import AttnRectangle
from magi_attention.common.rectangles import AttnRectangles


class TestAttnRectangles(TestCase):
    def setUp(self):
        """setup test environment"""
        self.rect1 = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 15)
        )
        self.rect2 = AttnRectangle(
            AttnRange(10, 20), AttnRange(20, 30), AttnRange(5, 15)
        )
        self.rect3 = AttnRectangle(
            AttnRange(5, 15), AttnRange(5, 25), AttnRange(-3, 17)
        )

    def test_init(self):
        """test init"""
        rects = AttnRectangles()
        self.assertEqual(len(rects), 0)
        self.assertTrue(rects.is_empty())
        self.assertTrue(rects.is_valid())

    def test_append(self):
        """test append rectangle"""
        rects = AttnRectangles()

        # test normal append
        rects.append(self.rect1)
        self.assertEqual(len(rects), 1)
        self.assertEqual(rects[0], self.rect1)

        # test append with check
        rects.append(self.rect2, check=True)
        self.assertEqual(len(rects), 2)

        # test append invalid rectangle (should raise exception)
        # create an invalid rectangle by directly modifying internal state
        invalid_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 15)
        )
        # directly modify internal state to make it invalid
        invalid_rect._q_range._start = 10
        invalid_rect._q_range._end = 5
        with self.assertRaises(ValueError):
            rects.append(invalid_rect, check=True)

    def test_extend(self):
        """test extend rectangle list"""
        rects1 = AttnRectangles()
        rects1.append(self.rect1)
        rects1.append(self.rect2)

        rects2 = AttnRectangles()
        rects2.append(self.rect3)

        # test normal extend
        rects1.extend(rects2)
        self.assertEqual(len(rects1), 3)
        self.assertEqual(rects1[2], self.rect3)

        # test extend with check
        rects3 = AttnRectangles()
        rects3.append(self.rect1)
        rects1.extend(rects3, check=True)
        self.assertEqual(len(rects1), 4)

    def test_from_ranges(self):
        """test from ranges"""
        # test using AttnRange
        q_ranges = [AttnRange(0, 10), AttnRange(10, 20)]
        k_ranges = [AttnRange(0, 20), AttnRange(20, 40)]
        mask_types = [AttnMaskType.FULL, AttnMaskType.CAUSAL]

        rects = AttnRectangles.from_ranges(q_ranges, k_ranges, mask_types)
        self.assertEqual(len(rects), 2)

        # test using integer mask_types
        int_mask_types = [0, 1, 2, 3]
        q_ranges_int = [AttnRange(0, 5)] * 4
        k_ranges_int = [AttnRange(0, 10)] * 4

        rects_int = AttnRectangles.from_ranges(
            q_ranges_int, k_ranges_int, int_mask_types
        )
        self.assertEqual(len(rects_int), 4)

        # test using AttnRange
        attn_ranges = [AttnRange(0, 10), AttnRange(10, 20)]
        rects_attn = AttnRectangles.from_ranges(attn_ranges, attn_ranges, [0, 0])
        self.assertEqual(len(rects_attn), 2)

        # test length mismatch exception
        with self.assertRaises(AssertionError):
            AttnRectangles.from_ranges([AttnRange(0, 10)], [AttnRange(0, 20)], [0, 1])

    def test_is_valid(self):
        """test validity check"""
        rects = AttnRectangles()
        self.assertTrue(rects.is_valid())  # empty list is always valid

        rects.append(self.rect1)
        self.assertTrue(rects.is_valid())

        # test invalid rectangle
        invalid_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 15)
        )
        # directly modify internal state to make it invalid
        invalid_rect._q_range._start = 10
        invalid_rect._q_range._end = 5
        rects._rects.append(invalid_rect)  # directly modify internal state
        self.assertFalse(rects.is_valid())

    def test_check_valid(self):
        """test validity check (raise exception)"""
        rects = AttnRectangles()
        rects.check_valid()  # empty list should pass

        rects.append(self.rect1)
        rects.check_valid()  # valid rectangle should pass

        # test invalid rectangle
        invalid_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 15)
        )
        # directly modify internal state to make it invalid
        invalid_rect._q_range._start = 10
        invalid_rect._q_range._end = 5
        rects._rects.append(invalid_rect)
        with self.assertRaises(ValueError):
            rects.check_valid()

    def test_get_qo_ranges_union(self):
        """test get Q/O ranges union"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        qo_ranges = rects.get_qo_ranges_union()
        self.assertEqual(qo_ranges.total_seqlen, 20)  # 0-10 + 10-20

        # test empty list
        empty_rects = AttnRectangles()
        empty_qo_ranges = empty_rects.get_qo_ranges_union()
        self.assertEqual(empty_qo_ranges.total_seqlen, 0)

    def test_get_kv_ranges_union(self):
        """test get K/V ranges union"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        kv_ranges = rects.get_kv_ranges_union()
        self.assertEqual(kv_ranges.total_seqlen, 30)  # 0-20 + 20-30

        # test empty list
        empty_rects = AttnRectangles()
        empty_kv_ranges = empty_rects.get_kv_ranges_union()
        self.assertEqual(empty_kv_ranges.total_seqlen, 0)

    def test_total_seqlen_qo(self):
        """test total seqlen Q/O"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        self.assertEqual(rects.total_seqlen_qo(), 20)

        # test empty list
        empty_rects = AttnRectangles()
        self.assertEqual(empty_rects.total_seqlen_qo(), 0)

    def test_total_seqlen_kv(self):
        """test total seqlen K/V"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        self.assertEqual(rects.total_seqlen_kv(), 30)

        # test empty list
        empty_rects = AttnRectangles()
        self.assertEqual(empty_rects.total_seqlen_kv(), 0)

    def test_cut_q(self):
        """test cut Q"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        # cut in the middle
        left_rects, right_rects = rects.cut_q(15)

        # check cutting result - adjust expected value based on actual cutting logic
        # rect1: [0, 10) x [0, 20)
        # rect2: [10, 20) x [20, 30)
        self.assertGreaterEqual(
            len(left_rects), 1
        )  # left_rects should have at least one rectangle
        self.assertGreaterEqual(
            len(right_rects), 1
        )  # right_rects should have at least one rectangle

        # check area conservation
        original_area = rects.area()
        left_area = left_rects.area()
        right_area = right_rects.area()
        self.assertEqual(original_area, left_area + right_area)

        # test empty list cutting
        empty_rects = AttnRectangles()
        left_empty, right_empty = empty_rects.cut_q(10)
        self.assertEqual(len(left_empty), 0)
        self.assertEqual(len(right_empty), 0)

    def test_cut_k(self):
        """test cut K"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        # cut in the middle
        left_rects, right_rects = rects.cut_k(25)

        # check cutting result - adjust expected value based on actual cutting logic
        # rect1: [0, 10) x [0, 20)
        # rect2: [10, 20) x [20, 30)
        self.assertGreaterEqual(
            len(left_rects), 1
        )  # left_rects should have at least one rectangle
        self.assertGreaterEqual(
            len(right_rects), 1
        )  # right_rects should have at least one rectangle

        # check area conservation
        original_area = rects.area()
        left_area = left_rects.area()
        right_area = right_rects.area()
        self.assertEqual(original_area, left_area + right_area)

        # test empty list cutting
        empty_rects = AttnRectangles()
        left_empty, right_empty = empty_rects.cut_k(10)
        self.assertEqual(len(left_empty), 0)
        self.assertEqual(len(right_empty), 0)

    def test_get_rects_within_q_segment(self):
        """test get rects within Q segment"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)
        rects.append(self.rect3)

        # get fully contained segment
        segment_rects = rects.get_rects_within_q_segment(5, 15)
        # rect1: [0, 10) - overlap with [5, 15) (0 < 15 and 10 > 5)
        # rect2: [10, 20) - overlap with [5, 15) (10 < 15 and 20 > 5)
        # rect3: [5, 15) - fully contained in [5, 15)
        self.assertEqual(
            len(segment_rects), 3
        )  # all three rectangles overlap with the segment

        # get partially overlapping segment
        segment_rects = rects.get_rects_within_q_segment(15, 25)
        # rect1: [0, 10) - not overlap (10 <= 15)
        # rect2: [10, 20) - overlap with [15, 25) (10 < 25 and 20 > 15)
        # rect3: [5, 15) - not overlap (15 >= 15)
        self.assertEqual(len(segment_rects), 1)  # only rect2

        # get non-overlapping segment
        segment_rects = rects.get_rects_within_q_segment(30, 40)
        self.assertEqual(len(segment_rects), 0)

        # test empty list
        empty_rects = AttnRectangles()
        empty_segment = empty_rects.get_rects_within_q_segment(0, 10)
        self.assertEqual(len(empty_segment), 0)

    def test_get_rects_within_k_segment(self):
        """test get rects within K segment"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)
        rects.append(self.rect3)

        # get fully contained segment
        segment_rects = rects.get_rects_within_k_segment(5, 25)
        # rect1: [0, 20) - overlap with [5, 25) (0 < 25 and 20 > 5)
        # rect2: [20, 30) - overlap with [5, 25) (20 < 25 and 30 > 5)
        # rect3: [5, 25) - fully contained in [5, 25)
        self.assertEqual(
            len(segment_rects), 3
        )  # all three rectangles overlap with the segment

        # get partially overlapping segment
        segment_rects = rects.get_rects_within_k_segment(25, 35)
        # rect1: [0, 20) - not overlap (20 <= 25)
        # rect2: [20, 30) - overlap with [25, 35) (20 < 35 and 30 > 25)
        # rect3: [5, 25) - not overlap (25 >= 25)
        self.assertEqual(len(segment_rects), 1)  # only rect2

        # get non-overlapping segment
        segment_rects = rects.get_rects_within_k_segment(40, 50)
        self.assertEqual(len(segment_rects), 0)

        # test empty list
        empty_rects = AttnRectangles()
        empty_segment = empty_rects.get_rects_within_k_segment(0, 10)
        self.assertEqual(len(empty_segment), 0)

    def test_area(self):
        """test total area calculation"""
        rects = AttnRectangles()
        self.assertEqual(rects.area(), 0)  # empty list area is 0

        rects.append(self.rect1)
        self.assertEqual(rects.area(), self.rect1.area())

        rects.append(self.rect2)
        expected_area = self.rect1.area() + self.rect2.area()
        self.assertEqual(rects.area(), expected_area)

    def test_properties(self):
        """test property access"""
        # test empty list
        empty_rects = AttnRectangles()
        self.assertEqual(empty_rects.size, 0)
        self.assertTrue(empty_rects.is_empty())  # empty list should be empty

        # test non-empty list
        rects = AttnRectangles()
        rects.append(self.rect1)
        self.assertEqual(rects.size, 1)
        self.assertFalse(
            rects.is_empty()
        )  # after adding elements, it should not be empty

    def test_indexing(self):
        """test indexing access"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)
        rects.append(self.rect3)

        # test integer indexing
        self.assertEqual(rects[0], self.rect1)
        self.assertEqual(rects[1], self.rect2)
        self.assertEqual(rects[2], self.rect3)

        # test slice indexing
        slice_rects = rects[1:3]
        self.assertEqual(len(slice_rects), 2)
        self.assertEqual(slice_rects[0], self.rect2)
        self.assertEqual(slice_rects[1], self.rect3)

        # test negative indexing
        self.assertEqual(rects[-1], self.rect3)
        self.assertEqual(rects[-2], self.rect2)

    def test_setitem(self):
        """test indexing assignment"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)

        # test integer indexing assignment
        new_rect = AttnRectangle(
            AttnRange(100, 110), AttnRange(100, 120), AttnRange(0, 20)
        )
        rects[0] = new_rect
        self.assertEqual(rects[0], new_rect)

        # test slice assignment
        new_rects = AttnRectangles()
        new_rects.append(new_rect)
        rects[1:2] = new_rects
        self.assertEqual(rects[1], new_rect)

    def test_iteration(self):
        """test iteration"""
        rects = AttnRectangles()
        rects.append(self.rect1)
        rects.append(self.rect2)
        rects.append(self.rect3)

        # test iteration
        rect_list = list(rects)
        self.assertEqual(len(rect_list), 3)
        self.assertEqual(rect_list[0], self.rect1)
        self.assertEqual(rect_list[1], self.rect2)
        self.assertEqual(rect_list[2], self.rect3)

        # test empty list iteration
        empty_rects = AttnRectangles()
        empty_list = list(empty_rects)
        self.assertEqual(len(empty_list), 0)

    def test_equality_and_hash(self):
        """test equality and hash"""
        rects1 = AttnRectangles()
        rects1.append(self.rect1)
        rects1.append(self.rect2)

        rects2 = AttnRectangles()
        rects2.append(self.rect1)
        rects2.append(self.rect2)

        rects3 = AttnRectangles()
        rects3.append(self.rect1)
        rects3.append(self.rect3)

        # test equality
        self.assertEqual(rects1, rects2)
        self.assertNotEqual(rects1, rects3)

        # test hash
        self.assertEqual(hash(rects1), hash(rects2))
        self.assertNotEqual(hash(rects1), hash(rects3))

        # test comparison with empty list
        empty_rects = AttnRectangles()
        self.assertNotEqual(rects1, empty_rects)

    def test_repr(self):
        """test string representation"""
        rects = AttnRectangles()

        # test empty list representation
        empty_repr = repr(rects)
        self.assertIn("[-1, -1) x [-1, -1): None", empty_repr)

        # test non-empty list representation
        rects.append(self.rect1)
        non_empty_repr = repr(rects)
        # check if contains rectangle information
        self.assertIn("[0, 10)", non_empty_repr)

    def test_random_cut_q_k(self):
        """test random cut q or k by area conservation method"""
        # random test 100 times
        for _ in range(100):
            rects = AttnRectangles()

            # random add rectangles
            num_rects = random.randint(0, 10)
            for _ in range(num_rects):
                q_start = random.randint(0, 99)
                q_end = random.randint(q_start + 1, 100)
                k_start = random.randint(0, 99)
                k_end = random.randint(k_start + 1, 100)

                # ensure d_range is valid, avoid shrink q k range
                d_min = k_start - (q_end - 1)
                d_mid_min = k_start - q_start
                d_mid_max = k_end - q_end
                if d_mid_min > d_mid_max:
                    d_mid_min, d_mid_max = d_mid_max, d_mid_min
                d_max = k_end - 1 - q_start

                # ensure d_range is in valid range
                if d_min <= d_mid_min and d_mid_max <= d_max:
                    d_start = random.randint(d_min, d_mid_min)
                    d_end = random.randint(max(d_mid_max, d_start), d_max)

                    if d_start <= d_end:  # only create rectangle when d_range is valid
                        rect = AttnRectangle(
                            AttnRange(q_start, q_end),
                            AttnRange(k_start, k_end),
                            AttnRange(d_start, d_end),
                        )
                        rects.append(rect)

            # test various operations
            if not rects.is_empty():
                # random cut q or k
                if rects.total_seqlen_qo() > 0:
                    cut_pos = random.randint(0, rects.total_seqlen_qo())
                    left, right = rects.cut_q(cut_pos)
                    self.assertEqual(rects.area(), left.area() + right.area())

                if rects.total_seqlen_kv() > 0:
                    cut_pos = random.randint(0, rects.total_seqlen_kv())
                    left, right = rects.cut_k(cut_pos)
                    self.assertEqual(rects.area(), left.area() + right.area())


if __name__ == "__main__":
    unittest.main()
