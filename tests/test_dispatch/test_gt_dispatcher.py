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

import importlib
import sys
import unittest
from unittest import TestCase

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.mask import AttnMask
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import MinHeapDispatchAlg
from magi_attention.testing.gt_dispatcher import GroundTruthDispatcher
from magi_attention.testing.utils import switch_envvars


def reload_magi_modules():
    """Helper to reload magi_attention modules and update global names in this module."""
    importlib.reload(sys.modules["magi_attention.common.range"])
    importlib.reload(sys.modules["magi_attention.common.ranges"])
    importlib.reload(sys.modules["magi_attention.common.enum"])
    importlib.reload(sys.modules["magi_attention.common.mask"])
    importlib.reload(sys.modules["magi_attention.common"])
    importlib.reload(sys.modules["magi_attention.testing.gt_dispatcher"])
    import magi_attention.common
    import magi_attention.testing.gt_dispatcher

    # Update the global names in this test module
    test_module = sys.modules[__name__]
    test_module.AttnRange = magi_attention.common.AttnRange
    test_module.AttnRanges = magi_attention.common.AttnRanges
    test_module.AttnMaskType = magi_attention.common.enum.AttnMaskType
    test_module.AttnMask = magi_attention.common.AttnMask
    test_module.GroundTruthDispatcher = (
        magi_attention.testing.gt_dispatcher.GroundTruthDispatcher
    )
    return magi_attention.common


class TestGroundTruthDispatcher(TestCase):
    def setUp(self):
        # Ensure we are using the Python backend
        self.switch_back = switch_envvars(
            ["MAGI_ATTENTION_CPP_BACKEND"],
            enable_dict={"MAGI_ATTENTION_CPP_BACKEND": False},
        )
        reload_magi_modules()

    def tearDown(self):
        self.switch_back()

    def test_make_sub_mask_with_sub_area(self):
        # --------------      init sample meta      -------------- #

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 6),
                (6, 9),
                (9, 12),
                (12, 16),
            ]
        )
        k_ranges = AttnRanges.from_ranges(
            [
                (0, 4),
                (4, 12),
                (12, 15),
                (1, 13),
            ]
        )
        attn_mask_type = [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
        ]
        # --------------      init attn mask       -------------- #
        attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # use the end of sq
        )
        assert attn_mask.area == 82
        # ------    sub mask 1    ------ #
        sub_q_range = AttnRange.from_range((4, 13))
        sub_k_range = AttnRange.from_range((1, 13))
        sub_area = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        sub_attn_mask1 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask1.area == sub_area == 41
        assert sub_attn_mask1.q_ranges == AttnRanges.from_ranges(
            [[0, 2], [2, 5], [5, 8], [8, 9]]
        )
        assert sub_attn_mask1.k_ranges == AttnRanges.from_ranges(
            [[0, 3], [3, 11], [11, 12], [0, 9]]
        )
        assert sub_attn_mask1.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
            AttnMaskType.FULL,
        ]
        # ------    sub mask 2    ------ #
        sub_q_range = AttnRange.from_range((0, 14))
        sub_k_range = AttnRange.from_range((0, 7))
        sub_area = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        sub_attn_mask2 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask2.area == sub_area == 31
        assert sub_attn_mask2.q_ranges == AttnRanges.from_ranges(
            [[0, 6], [6, 9], [9, 12], [12, 14]]
        )
        assert sub_attn_mask2.k_ranges == AttnRanges.from_ranges(
            [[0, 4], [4, 7], [9, 9], [1, 7]]
        )
        assert sub_attn_mask2.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
        ]
        # ------    sub mask 3    ------ #
        sub_q_range = AttnRange.from_range((5, 16))
        sub_k_range = AttnRange.from_range((3, 11))
        sub_area = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        sub_attn_mask3 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask3.area == sub_area == 53
        assert sub_attn_mask3.q_ranges == AttnRanges.from_ranges(
            [[0, 1], [1, 4], [4, 7], [7, 9], [9, 11]]
        )
        assert sub_attn_mask3.k_ranges == AttnRanges.from_ranges(
            [[0, 1], [1, 8], [4, 4], [0, 8], [0, 8]]
        )
        assert sub_attn_mask3.attn_mask_type == [
            AttnMaskType.FULL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
        ]
        # ------    sub mask 4    ------ #
        sub_q_range = AttnRange.from_range((4, 12))
        sub_k_range = AttnRange.from_range((4, 12))
        sub_area = attn_mask.calc_sub_area(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        sub_attn_mask4 = attn_mask.make_sub_mask(
            q_range=sub_q_range,
            k_range=sub_k_range,
        )
        assert sub_attn_mask4.area == sub_area == 24
        assert sub_attn_mask4.q_ranges == AttnRanges.from_ranges(
            [[0, 2], [2, 5], [5, 8]]
        )
        assert sub_attn_mask4.k_ranges == AttnRanges.from_ranges(
            [[0, 0], [0, 8], [5, 5]]
        )
        assert sub_attn_mask4.attn_mask_type == [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
        ]

    def test_compute_self_attn_areas(self):
        # --------------      init sample meta      -------------- #

        q_ranges = AttnRanges.from_ranges(
            [
                (0, 6),
                (6, 9),
                (9, 12),
                (12, 16),
            ]
        )

        k_ranges = AttnRanges.from_ranges(
            [
                (0, 4),
                (4, 12),
                (12, 15),
                (1, 13),
            ]
        )

        attn_mask_type = [
            AttnMaskType.CAUSAL,
            AttnMaskType.FULL,
            AttnMaskType.CAUSAL,
            AttnMaskType.CAUSAL,
        ]

        chunk_size = 4

        # --------------      init attn mask       -------------- #

        attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # use the end of sq
        )

        # --------------      init gt dispatcher       -------------- #

        gt_dispatcher = GroundTruthDispatcher(alg=MinHeapDispatchAlg())
        global_bucket = gt_dispatcher._compute_self_attn_areas(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            chunk_size=chunk_size,
        )

        self.assertEqual(global_bucket.area, attn_mask.area)

        self_attn_mask = gt_dispatcher._self_attn_mask
        self.assertEqual(self_attn_mask, attn_mask)

        for chunk_mask, chunk in zip(
            gt_dispatcher._chunk_masks, global_bucket.q_chunks
        ):
            self.assertEqual(chunk_mask.area, chunk.area)


class TestCppGroundTruthDispatcher(TestGroundTruthDispatcher):
    def setUp(self):
        # Ensure we are using the C++ backend
        self.switch_back = switch_envvars(
            ["MAGI_ATTENTION_CPP_BACKEND"],
            enable_dict={"MAGI_ATTENTION_CPP_BACKEND": True},
        )
        common = reload_magi_modules()
        if not getattr(common, "USE_CPP_BACKEND", False):
            self.skipTest("C++ backend is not available")

    def tearDown(self):
        self.switch_back()


if __name__ == "__main__":
    unittest.main()
