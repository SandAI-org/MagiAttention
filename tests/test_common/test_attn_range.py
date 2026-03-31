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

import pytest


class TestAttnRange:
    def test_simple_properties(self, backend):
        from magi_attention.common import AttnRange

        # ---------    init an attn range     --------- #

        attn_range = AttnRange(0, 10)
        assert attn_range.start == 0
        assert attn_range.end == 10
        assert attn_range.seqlen == 10
        assert len(attn_range) == 10
        assert not attn_range.is_empty()

        # ---------    change its start     --------- #

        attn_range.start = 4
        assert attn_range.start == 4
        assert attn_range.end == 10
        assert attn_range.seqlen == 6
        assert attn_range.seqlen == 6
        assert len(attn_range) == 6

        # ---------    change its end     --------- #

        attn_range.end = 12
        assert attn_range.start == 4
        assert attn_range.end == 12
        assert attn_range.seqlen == 8
        assert attn_range.seqlen == 8
        assert len(attn_range) == 8

        # ---------    test empty range     --------- #

        attn_range.start = 5
        attn_range.end = 5
        assert attn_range.start == 5
        assert attn_range.end == 5
        assert attn_range.seqlen == 0
        assert attn_range.seqlen == 0
        assert len(attn_range) == 0
        assert attn_range.is_empty()

        # ---------    test read-only properties     --------- #

        with pytest.raises(AttributeError):
            attn_range.seqlen = 3

        with pytest.raises(AttributeError):
            attn_range.seqlen = 3

        # ---------    test range equal with some other simple APIs    --------- #
        attn_range2 = AttnRange(7, 9)
        assert attn_range != attn_range2

        attn_range3 = AttnRange(0, 0)
        assert attn_range3.is_empty()
        assert attn_range != attn_range3  # both empty, but not equal

        attn_range4 = attn_range3.offset(5)
        assert attn_range4.is_empty()
        assert attn_range == attn_range4

        naive_attn_range4 = attn_range4.to_naive_range()
        assert naive_attn_range4 == (5, 5)
        assert (
            attn_range != naive_attn_range4
        )  # the same content, but not the same type
        attn_range4_from_naive = AttnRange.from_range(
            naive_attn_range4
        )  # another contructor, from naive range
        assert attn_range == attn_range4_from_naive

    def test_set_ops(self, backend):
        from magi_attention.common import AttnRange

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

        assert attn_range2.is_subrange_of(attn_range1)
        assert not attn_range3.is_subrange_of(attn_range1)
        assert not attn_range5.is_subrange_of(attn_range1)

        # ---------    test intersect     --------- #

        assert attn_range1.intersect(attn_range2) == attn_range2
        assert attn_range1.intersect(attn_range4).is_empty()
        assert attn_range1.intersect(attn_range5) == AttnRange(8, 9)
        assert attn_range4.intersect(attn_range5) == AttnRange(10, 12)
        assert attn_range4.is_overlap_with(attn_range5)
        assert not attn_range2.is_overlap_with(attn_range5)

        # ---------    test diff by     --------- #

        # case1: two disjoint ranges
        attn_diff_ranges_13 = attn_range1.diff_by(attn_range3)
        assert len(attn_diff_ranges_13) == 1
        assert attn_diff_ranges_13[0] == attn_range3
        attn_diff_ranges_31 = attn_range3.diff_by(attn_range1)
        assert len(attn_diff_ranges_31) == 1
        assert attn_diff_ranges_31[0] == attn_range1

        # case2: range and its sub range
        assert attn_range1.diff_by(attn_range2) == []
        attn_diff_ranges_21 = attn_range2.diff_by(attn_range1)
        assert len(attn_diff_ranges_21) == 2
        assert attn_diff_ranges_21[0] == AttnRange(2, 4)
        assert attn_diff_ranges_21[1] == AttnRange(7, 9)

        # case3: overlapped ranges but not case2
        attn_diff_ranges_15 = attn_range1.diff_by(attn_range5)
        assert len(attn_diff_ranges_15) == 1
        assert attn_diff_ranges_15[0] == AttnRange(9, 12)
        attn_diff_ranges_51 = attn_range5.diff_by(attn_range1)
        assert len(attn_diff_ranges_51) == 1
        assert attn_diff_ranges_51[0] == AttnRange(2, 8)

    def test_truncate(self, backend):
        from magi_attention.common import AttnRange

        attn_range = AttnRange(9, 15)

        # ---------    case1: w/o truncate     --------- #
        trunc_start, trunc_end = None, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range == attn_range

        # ---------    case2: with dummy truncate     --------- #
        trunc_start, trunc_end = 0, 20
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range == attn_range

        # ---------    case3: with left truncate     --------- #
        trunc_start, trunc_end = 11, None
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range == AttnRange(11, 15)

        # ---------    case4: with right truncate     --------- #
        trunc_start, trunc_end = None, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range == AttnRange(9, 13)

        # ---------    case5: with left+right truncate     --------- #
        trunc_start, trunc_end = 11, 13
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range == AttnRange(11, 13)

        # -----    case6: with left+right truncate but too left   ---- #
        trunc_start, trunc_end = 1, 7
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range.is_empty()

        # -----    case7: with left+right truncate but too right   ---- #
        trunc_start, trunc_end = 17, 23
        trunc_range = attn_range.truncate(
            start=trunc_start,
            end=trunc_end,
        )
        assert trunc_range.is_empty()

    def test_validation_methods(self, backend):
        from magi_attention.common import AttnRange
        from magi_attention.common.range import RangeError

        # ---------    test is_valid_close (closed interval)     --------- #
        attn_range = AttnRange(5, 10)

        # Valid cases for closed interval [start, end]
        assert attn_range.is_valid_close()  # 5 <= 10
        assert attn_range.is_valid_close(3, 8)  # 3 <= 8
        assert attn_range.is_valid_close(7, 7)  # 7 <= 7

        # Invalid cases for closed interval
        assert not attn_range.is_valid_close(8, 6)  # 8 > 6

        # ---------    test is_valid_open (open interval)     --------- #

        # Valid cases for open interval [start, end)
        assert attn_range.is_valid_open()  # 5 < 10
        assert attn_range.is_valid_open(3, 8)  # 3 < 8

        # Invalid cases for open interval
        assert not attn_range.is_valid_open(7, 7)  # 7 >= 7
        assert not attn_range.is_valid_open(8, 6)  # 8 > 6

        # ---------    test check_valid with closed interval rule     --------- #

        # Valid cases
        attn_range.check_valid()  # Should not raise
        attn_range.check_valid(3, 8)  # Should not raise
        attn_range.check_valid(7, 7)  # Should not raise

        # Invalid cases
        with pytest.raises(RangeError):
            attn_range.check_valid(8, 6)

    def test_offset_and_operations(self, backend):
        from magi_attention.common import AttnRange

        attn_rect_range = AttnRange(3, 8)

        # ---------    test offset     --------- #
        offset_range = attn_rect_range.offset(5)
        assert offset_range.start == 8
        assert offset_range.end == 13
        assert isinstance(offset_range, AttnRange)

        # ---------    test negative offset     --------- #
        neg_offset_range = attn_rect_range.offset(-2)
        assert neg_offset_range.start == 1
        assert neg_offset_range.end == 6
        assert isinstance(neg_offset_range, AttnRange)

        # ---------    test zero offset     --------- #
        zero_offset_range = attn_rect_range.offset(0)
        assert zero_offset_range == attn_rect_range
        assert isinstance(zero_offset_range, AttnRange)

    def test_edge_cases(self, backend):
        from magi_attention.common import AttnRange

        # ---------    test zero length range     --------- #
        zero_range = AttnRange(5, 5)
        assert zero_range.is_empty()
        assert zero_range.seqlen == 0
        assert len(zero_range) == 0

        # ---------    test single element range     --------- #
        single_range = AttnRange(5, 6)
        assert not single_range.is_empty()
        assert single_range.seqlen == 1
        assert len(single_range) == 1
