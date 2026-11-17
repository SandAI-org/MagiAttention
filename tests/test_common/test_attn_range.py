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

from magi_attention.common.enum import AttnMaskType as PyAttnMaskType
from magi_attention.common.range import AttnRange as PyAttnRange
from magi_attention.common.range import RangeError
from magi_attention.common.ranges import AttnRanges as PyAttnRanges
from magi_attention.common.rectangle import AttnRectangle as PyAttnRectangle
from magi_attention.common.rectangles import AttnRectangles as PyAttnRectangles

cpp_backend: ModuleType | None
try:
    from magi_attention import magi_attn_ext as cpp_backend
except ImportError:
    cpp_backend = None


@pytest.mark.arch_independent
class TestAttnRange(unittest.TestCase):
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

    def test_simple_properties(self):
        AttnRange = self.types.AttnRange

        # ---------    init an attn range     --------- #

        attn_range = AttnRange(0, 10)
        self.assertEqual(attn_range.start, 0)
        self.assertEqual(attn_range.end, 10)
        self.assertEqual(attn_range.seqlen, 10)
        self.assertEqual(len(attn_range), 10)
        self.assertFalse(attn_range.is_empty())

        # ---------    change its start     --------- #

        attn_range.start = 4
        self.assertEqual(attn_range.start, 4)
        self.assertEqual(attn_range.end, 10)
        self.assertEqual(attn_range.seqlen, 6)
        self.assertEqual(attn_range.seqlen, 6)
        self.assertEqual(len(attn_range), 6)

        # ---------    change its end     --------- #

        attn_range.end = 12
        self.assertEqual(attn_range.start, 4)
        self.assertEqual(attn_range.end, 12)
        self.assertEqual(attn_range.seqlen, 8)
        self.assertEqual(attn_range.seqlen, 8)
        self.assertEqual(len(attn_range), 8)

        # ---------    test empty range     --------- #

        attn_range.start = 5
        attn_range.end = 5
        self.assertEqual(attn_range.start, 5)
        self.assertEqual(attn_range.end, 5)
        self.assertEqual(attn_range.seqlen, 0)
        self.assertEqual(attn_range.seqlen, 0)
        self.assertEqual(len(attn_range), 0)
        self.assertTrue(attn_range.is_empty())

        # ---------    test read-only properties     --------- #

        with pytest.raises(AttributeError):
            attn_range.seqlen = 3

        with pytest.raises(AttributeError):
            attn_range.seqlen = 3

        # ---------    test range equal with some other simple APIs    --------- #
        attn_range2 = AttnRange(7, 9)
        self.assertNotEqual(attn_range, attn_range2)

        attn_range3 = AttnRange(0, 0)
        self.assertTrue(attn_range3.is_empty())
        self.assertNotEqual(attn_range, attn_range3)

        attn_range4 = attn_range3.offset(5)
        self.assertTrue(attn_range4.is_empty())
        self.assertEqual(attn_range, attn_range4)

        naive_attn_range4 = attn_range4.to_naive_range()
        self.assertEqual(naive_attn_range4, (5, 5))
        self.assertNotEqual(attn_range, naive_attn_range4)
        attn_range4_from_naive = AttnRange.from_range(
            naive_attn_range4
        )  # another contructor, from naive range
        self.assertEqual(attn_range, attn_range4_from_naive)

    def test_set_ops(self):
        AttnRange = self.types.AttnRange

        # ---------    init several attn ranges     --------- #
        # the attn ranges below can be mapped into the axis as below:
        #      |---------------------| r1:(2,9)
        #               |-------| r2:(4,7)
        # |----| r3:(0,2)
        #                              |-------| r4:(10,13)
        #                         |---------| r5:(8,12)

        attn_range1 = AttnRange(2, 9)
        attn_range2 = AttnRange.from_range((4, 7))
        attn_range3 = AttnRange(0, 2)
        attn_range4 = AttnRange.from_range((10, 13))
        attn_range5 = AttnRange(8, 12)

        # ---------    test is subrange of     --------- #

        self.assertTrue(attn_range2.is_subrange_of(attn_range1))
        self.assertFalse(attn_range3.is_subrange_of(attn_range1))
        self.assertFalse(attn_range5.is_subrange_of(attn_range1))

        # ---------    test intersect     --------- #

        self.assertEqual(attn_range1.intersect(attn_range2), attn_range2)
        self.assertTrue(attn_range1.intersect(attn_range4).is_empty())
        self.assertEqual(attn_range1.intersect(attn_range5), AttnRange(8, 9))
        self.assertEqual(attn_range4.intersect(attn_range5), AttnRange(10, 12))
        self.assertTrue(attn_range4.is_overlap_with(attn_range5))
        self.assertFalse(attn_range2.is_overlap_with(attn_range5))

        # ---------    test diff by     --------- #

        # case1: two disjoint ranges
        attn_diff_ranges_13 = attn_range1.diff_by(attn_range3)
        self.assertEqual(len(attn_diff_ranges_13), 1)
        self.assertEqual(attn_diff_ranges_13[0], attn_range3)
        attn_diff_ranges_31 = attn_range3.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_31), 1)
        self.assertEqual(attn_diff_ranges_31[0], attn_range1)

        # case2: range and its sub range
        self.assertEqual(attn_range1.diff_by(attn_range2), [])
        attn_diff_ranges_21 = attn_range2.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_21), 2)
        self.assertEqual(attn_diff_ranges_21[0], AttnRange(2, 4))
        self.assertEqual(attn_diff_ranges_21[1], AttnRange(7, 9))

        # case3: overlapped ranges but not case2
        attn_diff_ranges_15 = attn_range1.diff_by(attn_range5)
        self.assertEqual(len(attn_diff_ranges_15), 1)
        self.assertEqual(attn_diff_ranges_15[0], AttnRange(9, 12))
        attn_diff_ranges_51 = attn_range5.diff_by(attn_range1)
        self.assertEqual(len(attn_diff_ranges_51), 1)
        self.assertEqual(attn_diff_ranges_51[0], AttnRange(2, 8))

    def test_truncate(self):
        AttnRange = self.types.AttnRange

        attn_range = AttnRange(9, 15)

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_range)

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 20
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, attn_range)

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 11, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(11, 15))

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(9, 13))

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 11, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertEqual(trunc_range, AttnRange(11, 13))

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 1, 7
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 17, 23
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        self.assertTrue(trunc_range.is_empty())

    def test_validation_methods(self):
        AttnRange = self.types.AttnRange

        # ---------    test is_valid_close (closed interval)     --------- #
        attn_range = AttnRange(5, 10)

        # Valid cases for closed interval [start, end]
        self.assertTrue(attn_range.is_valid_close())
        self.assertTrue(attn_range.is_valid_close(3, 8))
        self.assertTrue(attn_range.is_valid_close(7, 7))

        # Invalid cases for closed interval
        self.assertFalse(attn_range.is_valid_close(8, 6))

        # ---------    test is_valid_open (open interval)     --------- #

        # Valid cases for open interval [start, end)
        self.assertTrue(attn_range.is_valid_open())
        self.assertTrue(attn_range.is_valid_open(3, 8))

        # Invalid cases for open interval
        self.assertFalse(attn_range.is_valid_open(7, 7))
        self.assertFalse(attn_range.is_valid_open(8, 6))

        # ---------    test check_valid with closed interval rule     --------- #

        # Valid cases
        attn_range.check_valid()  # Should not raise
        attn_range.check_valid(3, 8)  # Should not raise
        attn_range.check_valid(7, 7)  # Should not raise

        # Invalid cases
        with pytest.raises(RangeError):
            attn_range.check_valid(8, 6)

    def test_offset_and_operations(self):
        AttnRange = self.types.AttnRange

        attn_rect_range = AttnRange(3, 8)

        # ---------    test offset     --------- #
        offset_range = attn_rect_range.offset(5)
        self.assertEqual(offset_range.start, 8)
        self.assertEqual(offset_range.end, 13)
        self.assertTrue(isinstance(offset_range, AttnRange))

        # ---------    test negative offset     --------- #
        neg_offset_range = attn_rect_range.offset(-2)
        self.assertEqual(neg_offset_range.start, 1)
        self.assertEqual(neg_offset_range.end, 6)
        self.assertTrue(isinstance(neg_offset_range, AttnRange))

        # ---------    test zero offset     --------- #
        zero_offset_range = attn_rect_range.offset(0)
        self.assertEqual(zero_offset_range, attn_rect_range)
        self.assertTrue(isinstance(zero_offset_range, AttnRange))

    def test_edge_cases(self):
        AttnRange = self.types.AttnRange

        # ---------    test zero length range     --------- #
        zero_range = AttnRange(5, 5)
        self.assertTrue(zero_range.is_empty())
        self.assertEqual(zero_range.seqlen, 0)
        self.assertEqual(len(zero_range), 0)

        # ---------    test single element range     --------- #
        single_range = AttnRange(5, 6)
        self.assertFalse(single_range.is_empty())
        self.assertEqual(single_range.seqlen, 1)
        self.assertEqual(len(single_range), 1)

    def test_from_range_with_check(self):
        AttnRange = self.types.AttnRange

        r = AttnRange.from_range((0, 10), check=True)
        self.assertEqual(r, AttnRange(0, 10))

        r2 = AttnRange.from_range(AttnRange(3, 7), check=True)
        self.assertEqual(r2, AttnRange(3, 7))

        r_empty = AttnRange.from_range((5, 5), check=True)
        self.assertTrue(r_empty.is_empty())

    def test_intersect_size(self):
        AttnRange = self.types.AttnRange

        r1 = AttnRange(0, 10)
        r2 = AttnRange(5, 15)
        self.assertEqual(r1.intersect_size(r2), 5)

        r3 = AttnRange(20, 30)
        self.assertEqual(r1.intersect_size(r3), 0)

        r4 = AttnRange(3, 7)
        self.assertEqual(r1.intersect_size(r4), 4)

        self.assertEqual(r1.intersect_size(r1), 10)

    def test_union(self):
        AttnRange = self.types.AttnRange

        r1 = AttnRange(0, 10)
        r2 = AttnRange(5, 15)
        result = r1.union(r2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], AttnRange(0, 15))

        r3 = AttnRange(20, 30)
        result_disjoint = r1.union(r3)
        self.assertEqual(len(result_disjoint), 2)

        r_sub = AttnRange(3, 7)
        result_sub = r1.union(r_sub)
        self.assertEqual(len(result_sub), 1)
        self.assertEqual(result_sub[0], r1)

        result_sub_rev = r_sub.union(r1)
        self.assertEqual(len(result_sub_rev), 1)
        self.assertEqual(result_sub_rev[0], r1)

        empty = AttnRange(5, 5)
        result_with_empty = r1.union(empty)
        self.assertEqual(len(result_with_empty), 2)

    def test_union_size(self):
        AttnRange = self.types.AttnRange

        r1 = AttnRange(0, 10)
        r2 = AttnRange(5, 15)
        self.assertEqual(r1.union_size(r2), 15)

        r3 = AttnRange(20, 30)
        self.assertEqual(r1.union_size(r3), 20)

        r_sub = AttnRange(3, 7)
        self.assertEqual(r1.union_size(r_sub), 10)


class TestCppAttnRange(TestAttnRange):
    backend = "cpp"


if __name__ == "__main__":
    unittest.main()
