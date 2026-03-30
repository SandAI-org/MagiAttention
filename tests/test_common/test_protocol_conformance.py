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

"""
Tests that both Python and C++ (pybind11) backends satisfy the Protocol
contracts defined in magi_attention.common.protocols.

This ensures the two backends remain interchangeable.
"""

import unittest

from magi_attention.common.protocols import (
    AttnMaskTypeProtocol,
    AttnRangeProtocol,
    AttnRangesProtocol,
    AttnRectangleProtocol,
    AttnRectanglesProtocol,
)


class TestPythonBackendProtocol(unittest.TestCase):
    """Verify that the pure-Python implementations satisfy the Protocols."""

    def test_attn_mask_type(self):
        from magi_attention.common.enum import AttnMaskType

        self.assertIsInstance(AttnMaskType.FULL, AttnMaskTypeProtocol)

    def test_attn_range(self):
        from magi_attention.common.range import AttnRange

        r = AttnRange(0, 10)
        self.assertIsInstance(r, AttnRangeProtocol)

    def test_attn_ranges(self):
        from magi_attention.common.range import AttnRange
        from magi_attention.common.ranges import AttnRanges

        rs = AttnRanges()
        rs.append(AttnRange(0, 10))
        self.assertIsInstance(rs, AttnRangesProtocol)

    def test_attn_rectangle(self):
        from magi_attention.common.enum import AttnMaskType
        from magi_attention.common.range import AttnRange
        from magi_attention.common.rectangle import AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 10), mask_type=AttnMaskType.FULL)
        self.assertIsInstance(rect, AttnRectangleProtocol)

    def test_attn_rectangles(self):
        from magi_attention.common.rectangles import AttnRectangles

        rects = AttnRectangles()
        self.assertIsInstance(rects, AttnRectanglesProtocol)


class TestCppBackendProtocol(unittest.TestCase):
    """Verify that the C++ pybind11 implementations satisfy the Protocols."""

    @classmethod
    def setUpClass(cls):
        try:
            import magi_attention.magi_attn_ext  # noqa: F401

            cls.ext_available = True
        except ImportError:
            cls.ext_available = False

    def setUp(self):
        if not self.ext_available:
            self.skipTest("magi_attn_ext not available")

    def test_attn_mask_type(self):
        from magi_attention.magi_attn_ext import AttnMaskType

        self.assertIsInstance(AttnMaskType.FULL, AttnMaskTypeProtocol)

    def test_attn_range(self):
        from magi_attention.magi_attn_ext import AttnRange

        r = AttnRange(0, 10)
        self.assertIsInstance(r, AttnRangeProtocol)

    def test_attn_ranges(self):
        from magi_attention.magi_attn_ext import AttnRange, AttnRanges

        rs = AttnRanges()
        rs.append(AttnRange(0, 10))
        self.assertIsInstance(rs, AttnRangesProtocol)

    def test_attn_rectangle(self):
        from magi_attention.magi_attn_ext import AttnMaskType, AttnRange, AttnRectangle

        rect = AttnRectangle(AttnRange(0, 10), AttnRange(0, 10), mask_type=AttnMaskType.FULL)
        self.assertIsInstance(rect, AttnRectangleProtocol)

    def test_attn_rectangles(self):
        from magi_attention.magi_attn_ext import AttnRectangles

        rects = AttnRectangles()
        self.assertIsInstance(rects, AttnRectanglesProtocol)


if __name__ == "__main__":
    unittest.main()
