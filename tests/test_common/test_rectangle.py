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

import random
import unittest
from unittest import TestCase

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.rect_range import AttnRectRange
from magi_attention.common.rectangle import AttnRectangle


class TestAttnRectangle(TestCase):
    def test_init_with_attn_rect_range(self):
        """test init with AttnRectRange"""
        q_range = AttnRectRange(0, 10)
        k_range = AttnRectRange(0, 20)
        d_range = AttnRectRange(-5, 5)

        rect = AttnRectangle(q_range, k_range, d_range)
        self.assertEqual(rect.q_range, q_range)
        self.assertEqual(rect.k_range, k_range)
        self.assertEqual(rect.d_range, d_range)

    def test_init_with_attn_range(self):
        """test init with AttnRange"""
        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)
        d_range = AttnRange(0, 10)

        rect = AttnRectangle(q_range, k_range, d_range)
        self.assertIsInstance(rect.q_range, AttnRectRange)
        self.assertIsInstance(rect.k_range, AttnRectRange)
        self.assertIsInstance(rect.d_range, AttnRectRange)
        self.assertEqual(rect.q_range.start, 0)
        self.assertEqual(rect.q_range.end, 10)
        self.assertEqual(rect.k_range.start, 0)
        self.assertEqual(rect.k_range.end, 20)
        self.assertEqual(rect.d_range.start, 0)
        self.assertEqual(rect.d_range.end, 10)

    def test_init_with_large_d_range(self):
        """test init with large d_range"""
        q_range = AttnRectRange(0, 10)
        k_range = AttnRectRange(0, 20)
        d_range = AttnRectRange(-100, 100)

        rect = AttnRectangle(q_range, k_range, d_range)
        self.assertEqual(rect.d_range.start, -9)
        self.assertEqual(rect.d_range.end, 19)

    def test_causal_mask_d_range_adjustment(self):
        """test causal mask d_range adjustment"""
        q_range = AttnRectRange(0, 10)
        k_range = AttnRectRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.CAUSAL)
        # causal mask: d_range.end should be limited to k_range.end - q_range.end
        expected_d_end = k_range.end - q_range.end
        self.assertEqual(rect.d_range.end, expected_d_end)

    def test_bicausal_mask_d_range_adjustment(self):
        """test bicausal mask d_range adjustment"""
        q_range = AttnRectRange(0, 10)
        k_range = AttnRectRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.BICAUSAL)
        # bicausal mask: d_range.start and d_range.end should be adjusted
        expected_d_start = k_range.start - q_range.start
        expected_d_end = k_range.end - q_range.end
        self.assertEqual(rect.d_range.start, expected_d_start)
        self.assertEqual(rect.d_range.end, expected_d_end)

    def test_invcausal_mask_d_range_adjustment(self):
        """test invcausal mask d_range adjustment"""
        q_range = AttnRectRange(0, 10)
        k_range = AttnRectRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.INVCAUSAL)
        # invcausal mask: d_range.start should be limited to k_range.start - q_range.start
        expected_d_start = k_range.start - q_range.start
        self.assertEqual(rect.d_range.start, expected_d_start)

    def test_property_setters(self):
        """test property setters"""
        rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        # test q_range setter
        new_q_range = AttnRectRange(5, 15)
        rect.q_range = new_q_range
        self.assertEqual(rect.q_range, new_q_range)

        # test k_range setter
        new_k_range = AttnRectRange(5, 25)
        rect.k_range = new_k_range
        self.assertEqual(rect.k_range, new_k_range)

        # test d_range setter
        new_d_range = AttnRectRange(-10, 10)
        rect.d_range = new_d_range
        self.assertEqual(rect.d_range, new_d_range)

    def test_is_valid(self):
        """test validity check"""
        # valid rect
        valid_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )
        self.assertTrue(valid_rect.is_valid())

        # invalid rect - create invalid state by directly modifying attributes
        invalid_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )
        # directly modify attributes to make it invalid - use an existing invalid range
        existing_invalid_range = AttnRectRange(0, 1)
        existing_invalid_range._start = 10
        existing_invalid_range._end = 0
        invalid_rect._q_range = existing_invalid_range
        self.assertFalse(invalid_rect.is_valid())

    def test_check_valid(self):
        """test validity check (raise exception)"""
        # create a valid rect, then make it invalid by property setter
        rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        # set invalid value through property setter should raise exception
        # create an existing invalid range
        existing_invalid_range = AttnRectRange(0, 1)
        existing_invalid_range._start = 10
        existing_invalid_range._end = 0

        with self.assertRaises(ValueError):
            rect.q_range = existing_invalid_range

    def test_get_valid_or_none(self):
        """test get valid rect or None"""
        valid_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )
        self.assertEqual(valid_rect.get_valid_or_none(), valid_rect)

        # create an invalid rect (by directly modifying attributes)
        invalid_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )
        # directly modify attributes to make it invalid - use an existing invalid range
        existing_invalid_range = AttnRectRange(0, 1)
        existing_invalid_range._start = 10
        existing_invalid_range._end = 0
        invalid_rect._q_range = existing_invalid_range
        self.assertIsNone(invalid_rect.get_valid_or_none())

    def test_shrink_d_range(self):
        """test d_range shrink"""
        rect = AttnRectangle(
            AttnRectRange(10, 20), AttnRectRange(10, 30), AttnRectRange(-100, 100)
        )

        result = rect.shrink_d_range()
        self.assertTrue(result)
        self.assertEqual(rect.d_range.start, -9)
        self.assertEqual(rect.d_range.end, 19)

    def test_shrink_q_range(self):
        """test q_range shrink"""
        rect = AttnRectangle(
            AttnRectRange(0, 20), AttnRectRange(10, 15), AttnRectRange(-4, 4)
        )

        result = rect.shrink_q_range()
        self.assertTrue(result)
        self.assertGreaterEqual(rect.q_range.start, 6)
        self.assertLessEqual(rect.q_range.end, 19)

    def test_shrink_k_range(self):
        """test k_range shrink"""
        rect = AttnRectangle(
            AttnRectRange(10, 15), AttnRectRange(0, 20), AttnRectRange(-4, 4)
        )

        result = rect.shrink_k_range()
        self.assertTrue(result)
        self.assertGreaterEqual(rect.k_range.start, 6)
        self.assertLessEqual(rect.k_range.end, 19)

    def test_count_areas(self):
        """test count areas"""
        # test 100 times with random AttnRectangle
        for _ in range(100):
            # generate random AttnRectangle with constraints
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            d_min = k_start - (q_end - 1)
            # d_mid is the dialog line of top left corner and bottom right corner of the rectangle
            # d_mid used to make sure the q k range do not need shrink
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            # will shrink d_range automatically
            rand_rect = AttnRectangle(
                AttnRectRange(q_start, q_end),
                AttnRectRange(k_start, k_end),
                AttnRectRange(d_start, d_end),
            )

            # calculate standard area
            standard_area = 0
            for i in range(q_start, q_end):
                for j in range(k_start, k_end):
                    if j - i >= d_start and j - i <= d_end:
                        standard_area += 1

            # use count_areas method
            area = rand_rect.area()

            self.assertEqual(area, standard_area)

    def test_cut_q(self):
        """test cut q"""
        # random test and check area conservation
        for _ in range(100):
            # generate random q/k range
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)

            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            # generate d_range to make sure q/k range do not need shrink
            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRectRange(q_start, q_end),
                AttnRectRange(k_start, k_end),
                AttnRectRange(d_start, d_end),
            )

            # random cut in the middle (may at the boundary)
            cut_pos = random.randint(q_start, q_end)
            left_rect, right_rect = rand_rect.cut_q(cut_pos)

            # check if the cut is happened
            if left_rect is not None:
                self.assertLessEqual(left_rect.q_range.end, cut_pos)
            if right_rect is not None:
                self.assertGreaterEqual(right_rect.q_range.start, cut_pos)

            # area conservation: the sum of left and right areas should be equal to the original area
            total_area = rand_rect.area()
            left_area = left_rect.area() if left_rect is not None else 0
            right_area = right_rect.area() if right_rect is not None else 0
            self.assertEqual(total_area, left_area + right_area)

    def test_cut_k(self):
        """test cut k"""
        # random test and check area conservation
        for _ in range(100):
            # generate random q/k range
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)

            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            # generate d_range to make sure q/k range do not need shrink
            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRectRange(q_start, q_end),
                AttnRectRange(k_start, k_end),
                AttnRectRange(d_start, d_end),
            )

            # random cut in the middle (may at the boundary)
            cut_pos = random.randint(k_start, k_end)
            left_rect, right_rect = rand_rect.cut_k(cut_pos)

            # check if the cut is happened
            if left_rect is not None:
                self.assertLessEqual(left_rect.k_range.end, cut_pos)
            if right_rect is not None:
                self.assertGreaterEqual(right_rect.k_range.start, cut_pos)

            # area conservation: the sum of left and right areas should be equal to the original area
            total_area = rand_rect.area()
            left_area = left_rect.area() if left_rect is not None else 0
            right_area = right_rect.area() if right_rect is not None else 0
            self.assertEqual(total_area, left_area + right_area)

    def test_get_rect_within_q_segment(self):
        """test get rect within q segment"""
        rect = AttnRectangle(
            AttnRectRange(0, 20), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        # fully included segment
        segment_rect = rect.get_rect_within_q_segment(5, 15)
        self.assertIsNotNone(segment_rect)
        self.assertEqual(segment_rect.q_range.start, 5)
        self.assertEqual(segment_rect.q_range.end, 15)

        # non-overlapping segment
        segment_rect = rect.get_rect_within_q_segment(25, 35)
        self.assertIsNone(segment_rect)

    def test_get_rect_within_k_segment(self):
        """test get rect within k segment"""
        rect = AttnRectangle(
            AttnRectRange(0, 20), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        # fully included segment
        segment_rect = rect.get_rect_within_k_segment(5, 15)
        self.assertIsNotNone(segment_rect)
        self.assertEqual(segment_rect.k_range.start, 5)
        self.assertEqual(segment_rect.k_range.end, 15)

        # non-overlapping segment
        segment_rect = rect.get_rect_within_k_segment(25, 35)
        self.assertIsNone(segment_rect)

    def test_intersection_boundary_methods(self):
        """test intersection boundary methods"""
        rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        # left boundary intersection
        q_id_left = rect.intersection_q_id_on_left_boundary()
        self.assertEqual(q_id_left, 5)

        # right boundary intersection
        q_id_right = rect.intersection_q_id_on_right_boundary()
        self.assertEqual(q_id_right, 14)

    def test_mask_type_checks(self):
        """test mask type checks"""

        # random test
        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            q_range = AttnRectRange(q_start, q_end)
            k_range = AttnRectRange(k_start, k_end)

            d_min_full = k_start - (q_end - 1)
            d_max_full = k_end - 1 - q_start
            rect_full = AttnRectangle(
                q_range, k_range, AttnRectRange(d_min_full, d_max_full)
            )
            self.assertTrue(rect_full.is_full())

            rect_causal = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.CAUSAL)
            self.assertTrue(rect_causal.is_causal())

            rect_inv = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.INVCAUSAL)
            self.assertTrue(rect_inv.is_inv_causal())

            # BICAUSAL is only valid when q_range.seqlen <= k_range.seqlen
            if q_range.seqlen > k_range.seqlen:
                continue

            rect_bi = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.BICAUSAL)
            self.assertTrue(rect_bi.is_bi_causal())

    def test_to_qk_range_mask_type(self):
        """test convert to qk range and mask type"""
        # full mask
        full_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-100, 100)
        )
        result = full_rect.to_qk_range_mask_type()
        self.assertEqual(len(result), 1)
        q_range, k_range, mask_type = result[0]
        self.assertEqual(mask_type, 0)  # FULL

        # causal mask
        causal_rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), mask_type=AttnMaskType.CAUSAL
        )
        result = causal_rect.to_qk_range_mask_type()
        self.assertEqual(len(result), 1)
        q_range, k_range, mask_type = result[0]
        self.assertEqual(mask_type, 1)  # CAUSAL

        # random test
        for _ in range(100):
            # random q/k
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            # generate d_range in the same way as test_count_areas, ensure the rectangle is valid
            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRectRange(q_start, q_end),
                AttnRectRange(k_start, k_end),
                AttnRectRange(d_start, d_end),
            )

            parts = rand_rect.to_qk_range_mask_type()

            # area conservation: use the output (q_range, k_range, mask_type) to restore each sub-rectangle and sum the areas
            total_area = rand_rect.area()
            sum_area = 0
            for q_r, k_r, mask_t in parts:
                sub_rect = AttnRectangle(q_r, k_r, mask_type=mask_t)
                sum_area += sub_rect.area()

            self.assertEqual(total_area, sum_area)

    def test_equality_and_hash(self):
        """test equality and hash"""
        rect1 = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        rect2 = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        rect3 = AttnRectangle(
            AttnRectRange(1, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        self.assertEqual(rect1, rect2)
        self.assertNotEqual(rect1, rect3)
        self.assertEqual(hash(rect1), hash(rect2))
        self.assertNotEqual(hash(rect1), hash(rect3))

    def test_repr(self):
        """test string representation"""
        rect = AttnRectangle(
            AttnRectRange(0, 10), AttnRectRange(0, 20), AttnRectRange(-5, 5)
        )

        repr_str = repr(rect)
        # check if the string representation contains the correct range information
        self.assertIn("0, 10", repr_str)
        self.assertIn("0, 20", repr_str)
        self.assertIn("-5, 5", repr_str)


if __name__ == "__main__":
    unittest.main()
