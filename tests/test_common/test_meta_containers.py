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

import pytest


@pytest.fixture(autouse=True)
def _reload_slice_module():
    """Ensure AttnMaskType identity is consistent after conftest.py module reloads."""
    mods_to_reload = [
        "magi_attention.common.enum",
        "magi_attention.meta.container.slice",
        "magi_attention.meta.container.chunk",
        "magi_attention.meta.container.bucket",
    ]
    for mod_name in mods_to_reload:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
    yield


def _make_slice(slice_id, mask_type_str, q_start, q_end, k_start, k_end):
    """Helper to create AttnSlice with fresh imports to avoid enum identity issues after module reload."""
    from magi_attention.common import AttnRange
    from magi_attention.common.enum import AttnMaskType
    from magi_attention.meta.container.slice import AttnSlice

    mt = AttnMaskType(mask_type_str) if isinstance(mask_type_str, str) else mask_type_str
    return AttnSlice(
        slice_id=slice_id,
        mask_type=mt,
        q_range=AttnRange(q_start, q_end),
        k_range=AttnRange(k_start, k_end),
    )


class TestAttnSlice:
    def test_area_full_mask(self):
        s = _make_slice(0, "full", 0, 10, 0, 20)
        assert s.area == 200

    def test_area_causal_mask(self):
        s = _make_slice(0, "causal", 0, 5, 0, 10)
        assert s.area == (2 * 10 - 5 + 1) * 5 // 2

        s_tri = _make_slice(1, "causal", 0, 10, 0, 5)
        assert s_tri.area == (1 + 5) * 5 // 2

    def test_area_unsupported_mask_type(self):
        from magi_attention.common import AttnRange
        from magi_attention.meta.container.slice import AttnSlice

        s = AttnSlice(
            slice_id=0,
            mask_type="unsupported",
            q_range=AttnRange(0, 10),
            k_range=AttnRange(0, 10),
        )
        with pytest.raises(ValueError, match="Only support"):
            _ = s.area

    def test_area_setter(self):
        s = _make_slice(0, "full", 0, 10, 0, 10)
        s.area = 999
        assert s.area == 999

    def test_iou_with(self):
        s1 = _make_slice(0, "full", 0, 10, 0, 10)
        s2 = _make_slice(1, "full", 0, 10, 5, 15)
        iou = s1.iou_with(s2)
        assert iou == 5 / 15

    def test_eq_type_guard(self):
        s = _make_slice(0, "full", 0, 10, 0, 10)
        assert s != "not a slice"
        assert s != 42


class TestMultiKAttnSlice:
    def test_area_full_mask(self):
        from magi_attention.common import AttnRange, AttnRanges
        from magi_attention.common.enum import AttnMaskType
        from magi_attention.meta.container.slice import MultiKAttnSlice

        s = MultiKAttnSlice(
            q_range=AttnRange(0, 10),
            k_ranges=AttnRanges.from_ranges([(0, 5), (10, 20)]),
            mask_types=[AttnMaskType(v) for v in ["full", "full"]],
        )
        assert s.area == 10 * 5 + 10 * 10

    def test_area_unsupported_mask_type(self):
        from magi_attention.common import AttnRange, AttnRanges
        from magi_attention.meta.container.slice import MultiKAttnSlice

        s = MultiKAttnSlice(
            q_range=AttnRange(0, 10),
            k_ranges=AttnRanges.from_ranges([(0, 5)]),
            mask_types=["unsupported"],
        )
        with pytest.raises(ValueError, match="Only support"):
            _ = s.area

    def test_area_setter(self):
        from magi_attention.common import AttnRange, AttnRanges
        from magi_attention.common.enum import AttnMaskType
        from magi_attention.meta.container.slice import MultiKAttnSlice

        s = MultiKAttnSlice(
            q_range=AttnRange(0, 10),
            k_ranges=AttnRanges.from_ranges([(0, 5)]),
            mask_types=[AttnMaskType("full")],
        )
        s.area = 777
        assert s.area == 777


class TestAttnChunk:
    def test_iou_with_overlap(self):
        from magi_attention.meta.container.chunk import AttnChunk

        s1 = _make_slice(0, "full", 0, 10, 0, 20)
        s2 = _make_slice(1, "full", 0, 10, 10, 30)
        c1 = AttnChunk(chunk_id=0, q_slices=[s1])
        c2 = AttnChunk(chunk_id=1, q_slices=[s2])

        # intersect_size_with = 10, union_size_with = 20 + 20 = 40
        iou = c1.iou_with(c2)
        assert iou == pytest.approx(10 / 40)

    def test_iou_empty_k_ranges(self):
        from magi_attention.meta.container.chunk import AttnChunk

        c = AttnChunk(chunk_id=0, q_slices=[])
        assert c.iou == 0.0

    def test_eq_type_guard(self):
        from magi_attention.meta.container.chunk import AttnChunk

        c = AttnChunk(chunk_id=0, q_slices=[])
        assert c != "not a chunk"
        assert c != 42


class TestAttnBucket:
    def test_iou_with_overlap(self):
        from magi_attention.meta.container.bucket import AttnBucket
        from magi_attention.meta.container.chunk import AttnChunk

        s1 = _make_slice(0, "full", 0, 10, 0, 20)
        s2 = _make_slice(1, "full", 0, 10, 10, 30)
        c1 = AttnChunk(chunk_id=0, q_slices=[s1])
        c2 = AttnChunk(chunk_id=1, q_slices=[s2])
        b1 = AttnBucket(cp_rank=0, q_chunks=[c1])
        b2 = AttnBucket(cp_rank=1, q_chunks=[c2])

        # intersect_size_with = 10, union_size_with = 20 + 20 = 40
        iou = b1.iou_with(b2)
        assert iou == pytest.approx(10 / 40)

    def test_iou_empty(self):
        from magi_attention.meta.container.bucket import AttnBucket

        b = AttnBucket(cp_rank=0, q_chunks=[])
        assert b.iou == 0.0

    def test_eq_type_guard(self):
        from magi_attention.meta.container.bucket import AttnBucket

        b = AttnBucket(cp_rank=0, q_chunks=[])
        assert b != "not a bucket"
        assert b != 42
