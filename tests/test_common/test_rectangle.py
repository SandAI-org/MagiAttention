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
from types import ModuleType, SimpleNamespace

import pytest

from magi_attention.common.enum import AttnMaskType as PyAttnMaskType
from magi_attention.common.range import AttnRange as PyAttnRange
from magi_attention.common.ranges import AttnRanges as PyAttnRanges
from magi_attention.common.rectangle import AttnRectangle as PyAttnRectangle
from magi_attention.common.rectangles import AttnRectangles as PyAttnRectangles

cpp_backend: ModuleType | None
try:
    from magi_attention import magi_attn_ext as cpp_backend
except ImportError:
    cpp_backend = None


@pytest.mark.arch_independent
class TestAttnRectangle(unittest.TestCase):
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

    def test_init_with_attn_range(self):
        """test init with AttnRange"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)
        d_range = AttnRange(-5, 5)

        rect = AttnRectangle(q_range, k_range, d_range)
        self.assertTrue(isinstance(rect.q_range, AttnRange))
        self.assertTrue(isinstance(rect.k_range, AttnRange))
        self.assertTrue(isinstance(rect.d_range, AttnRange))
        self.assertEqual(rect.q_range.start, 0)
        self.assertEqual(rect.q_range.end, 10)
        self.assertEqual(rect.k_range.start, 0)
        self.assertEqual(rect.k_range.end, 15)
        self.assertEqual(rect.d_range.start, -5)
        self.assertEqual(rect.d_range.end, 5)

    def test_init_with_large_d_range(self):
        """test init with large d_range"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)
        d_range = AttnRange(-100, 100)

        rect = AttnRectangle(q_range, k_range, d_range)
        self.assertEqual(rect.d_range.start, -9)
        self.assertEqual(rect.d_range.end, 19)

    def test_causal_mask_d_range_adjustment(self):
        """test causal mask d_range adjustment"""
        AttnMaskType = self.types.AttnMaskType
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.CAUSAL)
        expected_d_end = k_range.end - q_range.end
        self.assertEqual(rect.d_range.end, expected_d_end)

    def test_bicausal_mask_d_range_adjustment(self):
        """test bicausal mask d_range adjustment"""
        AttnMaskType = self.types.AttnMaskType
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.BICAUSAL)
        expected_d_start = k_range.start - q_range.start
        expected_d_end = k_range.end - q_range.end
        self.assertEqual(rect.d_range.start, expected_d_start)
        self.assertEqual(rect.d_range.end, expected_d_end)

    def test_invcausal_mask_d_range_adjustment(self):
        """test invcausal mask d_range adjustment"""
        AttnMaskType = self.types.AttnMaskType
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        q_range = AttnRange(0, 10)
        k_range = AttnRange(0, 20)

        rect = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.INVCAUSAL)
        expected_d_start = k_range.start - q_range.start
        self.assertEqual(rect.d_range.start, expected_d_start)

    def test_property_setters(self):
        """test property setters"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        new_q_range = AttnRange(5, 15)
        rect.q_range = new_q_range
        self.assertEqual(rect.q_range, new_q_range)

        new_k_range = AttnRange(5, 25)
        rect.k_range = new_k_range
        self.assertEqual(rect.k_range, new_k_range)

        new_d_range = AttnRange(-10, 10)
        rect.d_range = new_d_range
        self.assertEqual(rect.d_range, new_d_range)

    def test_is_valid(self):
        """test validity check"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        valid_rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))
        self.assertTrue(valid_rect.is_valid())

        invalid_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5)
        )
        invalid_rect.q_range.start = 10
        invalid_rect.q_range.end = 0
        self.assertFalse(invalid_rect.is_valid())

    def test_check_valid(self):
        """test validity check (raise exception)"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        existing_invalid_range = AttnRange(0, 1)
        existing_invalid_range.start = 10
        existing_invalid_range.end = 0

        with pytest.raises((ValueError, RuntimeError)):
            rect.q_range = existing_invalid_range

    def test_get_valid_or_none(self):
        """test get valid rect or None"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        valid_rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))
        self.assertEqual(valid_rect.get_valid_or_none(), valid_rect)

        invalid_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5)
        )
        invalid_rect.q_range.start = 10
        invalid_rect.q_range.end = 0
        self.assertIs(invalid_rect.get_valid_or_none(), None)

    def test_shrink_d_range(self):
        """test d_range shrink"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(10, 20), AttnRange(10, 30), AttnRange(-100, 100))

        result = rect.shrink_d_range()
        self.assertTrue(result)
        self.assertEqual(rect.d_range.start, -9)
        self.assertEqual(rect.d_range.end, 19)

    def test_shrink_q_range(self):
        """test q_range shrink"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 20), AttnRange(10, 15), AttnRange(-4, 4))

        result = rect.shrink_q_range()
        self.assertTrue(result)
        self.assertTrue(rect.q_range.start >= 6)
        self.assertTrue(rect.q_range.end <= 19)

    def test_shrink_k_range(self):
        """test k_range shrink"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(10, 15), AttnRange(0, 20), AttnRange(-4, 4))

        result = rect.shrink_k_range()
        self.assertTrue(result)
        self.assertTrue(rect.k_range.start >= 6)
        self.assertTrue(rect.k_range.end <= 19)

    def test_count_areas(self):
        """test count areas"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRange(q_start, q_end),
                AttnRange(k_start, k_end),
                AttnRange(d_start, d_end),
            )

            standard_area = 0
            for i in range(q_start, q_end):
                for j in range(k_start, k_end):
                    if j - i >= d_start and j - i <= d_end:
                        standard_area += 1

            area = rand_rect.area()

            self.assertEqual(area, standard_area)

    def test_cut_q(self):
        """test cut q"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)

            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRange(q_start, q_end),
                AttnRange(k_start, k_end),
                AttnRange(d_start, d_end),
            )

            cut_pos = random.randint(q_start, q_end)
            left_rect, right_rect = rand_rect.cut_q(cut_pos)

            if left_rect is not None:
                self.assertTrue(left_rect.q_range.end <= cut_pos)
            if right_rect is not None:
                self.assertTrue(right_rect.q_range.start >= cut_pos)

            total_area = rand_rect.area()
            left_area = left_rect.area() if left_rect is not None else 0
            right_area = right_rect.area() if right_rect is not None else 0
            self.assertEqual(total_area, left_area + right_area)

    def test_cut_k(self):
        """test cut k"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)

            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRange(q_start, q_end),
                AttnRange(k_start, k_end),
                AttnRange(d_start, d_end),
            )

            cut_pos = random.randint(k_start, k_end)
            left_rect, right_rect = rand_rect.cut_k(cut_pos)

            if left_rect is not None:
                self.assertTrue(left_rect.k_range.end <= cut_pos)
            if right_rect is not None:
                self.assertTrue(right_rect.k_range.start >= cut_pos)

            total_area = rand_rect.area()
            left_area = left_rect.area() if left_rect is not None else 0
            right_area = right_rect.area() if right_rect is not None else 0
            self.assertEqual(total_area, left_area + right_area)

    def test_get_rect_within_q_segment(self):
        """test get rect within q segment"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 20), AttnRange(0, 20), AttnRange(-5, 5))

        segment_rect = rect.get_rect_within_q_segment(5, 15)
        self.assertIsNot(segment_rect, None)
        self.assertEqual(segment_rect.q_range.start, 5)
        self.assertEqual(segment_rect.q_range.end, 15)

        segment_rect = rect.get_rect_within_q_segment(25, 35)
        self.assertIs(segment_rect, None)

    def test_get_rect_within_k_segment(self):
        """test get rect within k segment"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 20), AttnRange(0, 20), AttnRange(-5, 5))

        segment_rect = rect.get_rect_within_k_segment(5, 15)
        self.assertIsNot(segment_rect, None)
        self.assertEqual(segment_rect.k_range.start, 5)
        self.assertEqual(segment_rect.k_range.end, 15)

        segment_rect = rect.get_rect_within_k_segment(25, 35)
        self.assertIs(segment_rect, None)

    def test_intersection_boundary_methods(self):
        """test intersection boundary methods"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        q_id_left = rect.intersection_q_id_on_left_boundary()
        self.assertEqual(q_id_left, 5)

        q_id_right = rect.intersection_q_id_on_right_boundary()
        self.assertEqual(q_id_right, 9)

    def test_mask_type_checks(self):
        """test mask type checks"""
        AttnMaskType = self.types.AttnMaskType
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            q_range = AttnRange(q_start, q_end)
            k_range = AttnRange(k_start, k_end)

            d_min_full = k_start - (q_end - 1)
            d_max_full = k_end - 1 - q_start
            rect_full = AttnRectangle(
                q_range, k_range, AttnRange(d_min_full, d_max_full)
            )
            self.assertTrue(rect_full.is_full())

            rect_causal = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.CAUSAL)
            self.assertTrue(rect_causal.is_causal())

            rect_inv = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.INVCAUSAL)
            self.assertTrue(rect_inv.is_inv_causal())

            if q_range.seqlen > k_range.seqlen:
                continue

            rect_bi = AttnRectangle(q_range, k_range, mask_type=AttnMaskType.BICAUSAL)
            self.assertTrue(rect_bi.is_bi_causal())

    def test_to_qk_range_mask_type(self):
        """test convert to qk range and mask type"""
        AttnMaskType = self.types.AttnMaskType
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        full_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), AttnRange(-100, 100)
        )
        result = full_rect.to_qk_range_mask_type()
        self.assertEqual(len(result), 1)
        q_range, k_range, mask_type = result[0]
        self.assertEqual(mask_type, 0)

        causal_rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 20), mask_type=AttnMaskType.CAUSAL
        )
        result = causal_rect.to_qk_range_mask_type()
        self.assertEqual(len(result), 1)
        q_range, k_range, mask_type = result[0]
        self.assertEqual(mask_type, 1)

        for _ in range(100):
            q_start = random.randint(0, 99)
            q_end = random.randint(q_start + 1, 100)
            k_start = random.randint(0, 99)
            k_end = random.randint(k_start + 1, 100)

            d_min = k_start - (q_end - 1)
            d_mid_min = k_start - q_start
            d_mid_max = k_end - q_end
            if d_mid_min > d_mid_max:
                d_mid_min, d_mid_max = d_mid_max, d_mid_min
            d_max = k_end - 1 - q_start
            d_start = random.randint(d_min, d_mid_min)
            d_end = random.randint(max(d_mid_max, d_start), d_max)

            rand_rect = AttnRectangle(
                AttnRange(q_start, q_end),
                AttnRange(k_start, k_end),
                AttnRange(d_start, d_end),
            )

            parts = rand_rect.to_qk_range_mask_type()
            self.assertTrue(len(parts) <= 3)

            total_area = rand_rect.area()
            sum_area = 0
            for q_r, k_r, mask_t in parts:
                sub_rect = AttnRectangle(q_r, k_r, mask_type=mask_t)
                sum_area += sub_rect.area()

            self.assertEqual(total_area, sum_area)

    def test_equality_and_hash(self):
        """test equality and hash"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect1 = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        rect2 = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        rect3 = AttnRectangle(AttnRange(1, 10), AttnRange(0, 20), AttnRange(-5, 5))

        self.assertEqual(rect1, rect2)
        self.assertNotEqual(rect1, rect3)
        self.assertEqual(hash(rect1), hash(rect2))
        self.assertNotEqual(hash(rect1), hash(rect3))

    def test_repr(self):
        """test string representation"""
        AttnRange = self.types.AttnRange
        AttnRectangle = self.types.AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 20), AttnRange(-5, 5))

        repr_str = repr(rect)
        self.assertIn("0, 10", repr_str)
        self.assertIn("0, 15", repr_str)
        self.assertIn("-5, 5", repr_str)


class TestCppAttnRectangle(TestAttnRectangle):
    backend = "cpp"


if __name__ == "__main__":
    unittest.main()
