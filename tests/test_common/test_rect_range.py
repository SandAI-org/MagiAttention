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

from magi_attention.common.range import AttnRange, RangeError
from magi_attention.common.rect_range import AttnRectRange


class TestAttnRectRange(TestCase):
    def test_simple_properties(self):
        # ---------    init an attn rect range     --------- #

        attn_rect_range = AttnRectRange(0, 10)
        self.assertEqual(attn_rect_range.start, 0)
        self.assertEqual(attn_rect_range.end, 10)
        self.assertEqual(attn_rect_range.seqlen, 10)
        self.assertEqual(len(attn_rect_range), 10)
        self.assertFalse(attn_rect_range.is_empty())

        # ---------    change its start     --------- #

        attn_rect_range.start = 4
        self.assertEqual(attn_rect_range.start, 4)
        self.assertEqual(attn_rect_range.end, 10)
        self.assertEqual(attn_rect_range.seqlen, 6)
        self.assertEqual(len(attn_rect_range), 6)

        # ---------    change its end     --------- #

        attn_rect_range.end = 12
        self.assertEqual(attn_rect_range.start, 4)
        self.assertEqual(attn_rect_range.end, 12)
        self.assertEqual(attn_rect_range.seqlen, 8)
        self.assertEqual(len(attn_rect_range), 8)

        # ---------    test empty range     --------- #

        attn_rect_range.start = 5
        attn_rect_range.end = 5
        self.assertEqual(attn_rect_range.start, 5)
        self.assertEqual(attn_rect_range.end, 5)
        self.assertEqual(attn_rect_range.seqlen, 0)
        self.assertEqual(len(attn_rect_range), 0)
        self.assertTrue(attn_rect_range.is_empty())

        # ---------    test read-only properties     --------- #

        with self.assertRaises(
            AttributeError,
            msg="The 'seqlen' property is read-only",
        ):
            attn_rect_range.seqlen = 3

        # ---------    test range equal with some other simple APIs    --------- #
        attn_rect_range2 = AttnRectRange(7, 9)
        self.assertNotEqual(attn_rect_range, attn_rect_range2)

        attn_rect_range3 = AttnRectRange(0, 0)
        self.assertTrue(attn_rect_range3.is_empty())
        self.assertNotEqual(
            attn_rect_range, attn_rect_range3
        )  # both empty, but not equal

        attn_rect_range4 = attn_rect_range3.offset(5)
        self.assertTrue(attn_rect_range4.is_empty())
        self.assertEqual(attn_rect_range, attn_rect_range4)

        naive_attn_rect_range4 = attn_rect_range4.to_naive_range()
        self.assertEqual(naive_attn_rect_range4, (5, 5))
        self.assertNotEqual(
            attn_rect_range, naive_attn_rect_range4
        )  # the same content, but not the same type
        attn_rect_range4_from_naive = AttnRectRange.from_range(
            naive_attn_rect_range4
        )  # another constructor, from naive range
        self.assertEqual(attn_rect_range, attn_rect_range4_from_naive)

    def test_inheritance_and_conversion(self):
        # ---------    test inheritance from AttnRange     --------- #
        attn_range = AttnRange(3, 7)
        attn_rect_range = AttnRectRange.from_parent(attn_range)

        self.assertEqual(attn_rect_range.start, 3)
        self.assertEqual(attn_rect_range.end, 7)
        self.assertEqual(attn_rect_range.seqlen, 4)

        # ---------    test conversion back to parent     --------- #
        parent_range = attn_rect_range.to_parent()
        self.assertEqual(parent_range, attn_range)
        self.assertIsInstance(parent_range, AttnRange)
        self.assertNotIsInstance(parent_range, AttnRectRange)

        # ---------    test from_range with AttnRange input     --------- #
        attn_rect_range2 = AttnRectRange.from_range(attn_range)
        self.assertEqual(attn_rect_range2.start, 3)
        self.assertEqual(attn_rect_range2.end, 7)
        self.assertIsInstance(attn_rect_range2, AttnRectRange)

    def test_truncate(self):
        attn_rect_range = AttnRectRange(9, 15)

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_rect_range)

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 20
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_rect_range)

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 11, None
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRectRange(11, 15))

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 13
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRectRange(9, 13))

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 11, 13
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRectRange(11, 13))

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 1, 7
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 17, 23
        trunc_range = attn_rect_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())

    def test_validation_methods(self):
        # ---------    test is_valid_close (closed interval)     --------- #
        attn_rect_range = AttnRectRange(5, 10)

        # Valid cases for closed interval [start, end]
        self.assertTrue(attn_rect_range.is_valid_close())  # 5 <= 10
        self.assertTrue(attn_rect_range.is_valid_close(3, 8))  # 3 <= 8
        self.assertTrue(attn_rect_range.is_valid_close(7, 7))  # 7 <= 7

        # Invalid cases for closed interval
        self.assertFalse(attn_rect_range.is_valid_close(8, 6))  # 8 > 6

        # ---------    test is_valid_open (open interval)     --------- #

        # Valid cases for open interval [start, end)
        self.assertTrue(attn_rect_range.is_valid_open())  # 5 < 10
        self.assertTrue(attn_rect_range.is_valid_open(3, 8))  # 3 < 8

        # Invalid cases for open interval
        self.assertFalse(attn_rect_range.is_valid_open(7, 7))  # 7 >= 7
        self.assertFalse(attn_rect_range.is_valid_open(8, 6))  # 8 > 6

        # ---------    test check_valid with closed interval rule     --------- #

        # Valid cases
        attn_rect_range.check_valid()  # Should not raise
        attn_rect_range.check_valid(3, 8)  # Should not raise
        attn_rect_range.check_valid(7, 7)  # Should not raise

        # Invalid cases
        with self.assertRaises(RangeError):
            attn_rect_range.check_valid(8, 6)

    def test_offset_and_operations(self):
        attn_rect_range = AttnRectRange(3, 8)

        # ---------    test offset     --------- #
        offset_range = attn_rect_range.offset(5)
        self.assertEqual(offset_range.start, 8)
        self.assertEqual(offset_range.end, 13)
        self.assertIsInstance(offset_range, AttnRectRange)

        # ---------    test negative offset     --------- #
        neg_offset_range = attn_rect_range.offset(-2)
        self.assertEqual(neg_offset_range.start, 1)
        self.assertEqual(neg_offset_range.end, 6)
        self.assertIsInstance(neg_offset_range, AttnRectRange)

        # ---------    test zero offset     --------- #
        zero_offset_range = attn_rect_range.offset(0)
        self.assertEqual(zero_offset_range, attn_rect_range)
        self.assertIsInstance(zero_offset_range, AttnRectRange)

    def test_edge_cases(self):
        # ---------    test zero length range     --------- #
        zero_range = AttnRectRange(5, 5)
        self.assertTrue(zero_range.is_empty())
        self.assertEqual(zero_range.seqlen, 0)
        self.assertEqual(len(zero_range), 0)

        # ---------    test single element range     --------- #
        single_range = AttnRectRange(5, 6)
        self.assertFalse(single_range.is_empty())
        self.assertEqual(single_range.seqlen, 1)
        self.assertEqual(len(single_range), 1)

    def test_from_range_interface(self):
        """Test the from_range static method with various input types and parameters"""

        # ---------    test from NaiveRange (tuple)     --------- #
        naive_range_tuple = (3, 8)
        rect_range_from_tuple = AttnRectRange.from_range(naive_range_tuple)
        self.assertEqual(rect_range_from_tuple.start, 3)
        self.assertEqual(rect_range_from_tuple.end, 8)
        self.assertIsInstance(rect_range_from_tuple, AttnRectRange)

        # ---------    test from NaiveRange (list)     --------- #
        naive_range_list = [5, 12]
        rect_range_from_list = AttnRectRange.from_range(naive_range_list)
        self.assertEqual(rect_range_from_list.start, 5)
        self.assertEqual(rect_range_from_list.end, 12)
        self.assertIsInstance(rect_range_from_list, AttnRectRange)

        # ---------    test from AttnRectRange instance     --------- #
        original_rect_range = AttnRectRange(7, 15)
        rect_range_from_instance = AttnRectRange.from_range(original_rect_range)
        self.assertEqual(rect_range_from_instance.start, 7)
        self.assertEqual(rect_range_from_instance.end, 15)
        self.assertIsInstance(rect_range_from_instance, AttnRectRange)
        # Verify it's a deep copy, not the same object
        self.assertIsNot(rect_range_from_instance, original_rect_range)

        # ---------    test from AttnRange instance     --------- #
        attn_range = AttnRange(2, 9)
        rect_range_from_attn = AttnRectRange.from_range(attn_range)
        self.assertEqual(rect_range_from_attn.start, 2)
        self.assertEqual(rect_range_from_attn.end, 9)
        self.assertIsInstance(rect_range_from_attn, AttnRectRange)

        # ---------    test with check=True for valid ranges     --------- #
        valid_range = (4, 10)
        rect_range_with_check = AttnRectRange.from_range(valid_range, check=True)
        self.assertEqual(rect_range_with_check.start, 4)
        self.assertEqual(rect_range_with_check.end, 10)

        # ---------    test with check=True for invalid ranges     --------- #
        invalid_range = (10, 4)  # start > end
        with self.assertRaises(RangeError):
            AttnRectRange.from_range(invalid_range, check=True)

        # ---------    test edge cases with from_range     --------- #
        # Empty range
        empty_range = (5, 5)
        rect_range_empty = AttnRectRange.from_range(empty_range)
        self.assertTrue(rect_range_empty.is_empty())
        self.assertEqual(rect_range_empty.start, 5)
        self.assertEqual(rect_range_empty.end, 5)

        # Single element range
        single_range = (3, 4)
        rect_range_single = AttnRectRange.from_range(single_range)
        self.assertFalse(rect_range_single.is_empty())
        self.assertEqual(rect_range_single.seqlen, 1)

        # Zero range
        zero_range = (0, 0)
        rect_range_zero = AttnRectRange.from_range(zero_range)
        self.assertTrue(rect_range_zero.is_empty())
        self.assertEqual(rect_range_zero.seqlen, 0)

        # ---------    test from_range preserves equality     --------- #
        range1 = AttnRectRange(3, 7)
        range2 = AttnRectRange.from_range(range1)
        self.assertEqual(range1, range2)

        # Should be equal after creating from naive range
        naive_range = (3, 7)
        range3 = AttnRectRange.from_range(naive_range)
        self.assertEqual(range1, range3)

        # ---------    test from_range with negative values     --------- #
        negative_range = (-5, -2)
        rect_range_negative = AttnRectRange.from_range(negative_range)
        self.assertEqual(rect_range_negative.start, -5)
        self.assertEqual(rect_range_negative.end, -2)
        self.assertEqual(rect_range_negative.seqlen, 3)

        # Negative range should pass validation with check=True (since -5 <= -2)
        rect_range_negative_checked = AttnRectRange.from_range(
            negative_range, check=True
        )
        self.assertEqual(rect_range_negative_checked.start, -5)
        self.assertEqual(rect_range_negative_checked.end, -2)


if __name__ == "__main__":
    unittest.main()
