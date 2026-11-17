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

This ensures the two backends remain interchangeable:
- Protocol isinstance checks verify method existence
- Signature comparison verifies parameter names match
"""

import inspect
import re

import pytest

from magi_attention.common.protocols import (
    AttnMaskTypeProtocol,
    AttnRangeProtocol,
    AttnRangesProtocol,
    AttnRectangleProtocol,
    AttnRectanglesProtocol,
)


class TestProtocolConformance:
    """Verify both backends satisfy the Protocol contracts (method existence)."""

    def test_attn_mask_type(self, backend):
        from magi_attention.common import AttnMaskType

        assert isinstance(AttnMaskType.FULL, AttnMaskTypeProtocol)

    def test_attn_range(self, backend):
        from magi_attention.common import AttnRange

        r = AttnRange(0, 10)
        assert isinstance(r, AttnRangeProtocol)

    def test_attn_ranges(self, backend):
        from magi_attention.common import AttnRange, AttnRanges

        rs = AttnRanges()
        rs.append(AttnRange(0, 10))
        assert isinstance(rs, AttnRangesProtocol)

    def test_attn_rectangle(self, backend):
        from magi_attention.common import AttnMaskType, AttnRange, AttnRectangle

        rect = AttnRectangle(
            AttnRange(0, 10), AttnRange(0, 10), mask_type=AttnMaskType.FULL
        )
        assert isinstance(rect, AttnRectangleProtocol)

    def test_attn_rectangles(self, backend):
        from magi_attention.common import AttnRectangles

        rects = AttnRectangles()
        assert isinstance(rects, AttnRectanglesProtocol)


def _get_param_names_python(cls, method_name):
    """Get parameter names from a Python method via inspect."""
    method = getattr(cls, method_name)
    sig = inspect.signature(method)
    return [p for p in sig.parameters if p != "self"]


def _get_param_names_pybind(cls, method_name):
    """Get parameter names from a pybind11 method by parsing its docstring.

    pybind11 embeds the signature in the first line of __doc__, e.g.:
        offset(self: ..., offset: ...) -> ...
    """
    method = getattr(cls, method_name)
    doc = getattr(method, "__doc__", None)
    if not doc:
        return None

    first_line = doc.split("\n")[0]
    match = re.match(r"[^(]*\(([^)]*)\)", first_line)
    if not match:
        return None

    params_str = match.group(1)
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if not part:
            continue
        name = part.split(":")[0].strip()
        if name == "self":
            continue
        params.append(name)

    return params


# Public methods to check per class (excluding dunder methods and properties)
_ATTN_RANGE_METHODS = [
    "from_range",
    "clone",
    "offset",
    "truncate",
    "intersect",
    "intersect_size",
    "union",
    "union_size",
    "diff_by",
    "is_subrange_of",
    "is_overlap_with",
    "is_empty",
    "is_valid_close",
    "is_valid_open",
    "check_valid",
    "to_naive_range",
]

_ATTN_RANGES_METHODS = [
    "from_ranges",
    "from_cu_seqlens",
    "append",
    "insert",
    "extend",
    "pop",
    "clear_empty",
    "clone",
    "sort",
    "merge",
    "merge_with_split_alignment",
    "chunk",
    "truncate",
    "is_sorted",
    "is_merged",
    "is_non_overlap",
    "is_cu_seqlens",
    "is_valid",
    "check_valid",
    "is_empty",
    "to_cu_seqlens",
    "to_tensor",
    "to_naive_ranges",
    "make_range_local",
    "make_ranges_local",
    "find_hole_ranges",
    "find_overlap_ranges",
    "intersect_size",
    "intersect_size_with",
    "union_size",
    "union_size_with",
]

_ATTN_RECTANGLE_METHODS = [
    "is_valid",
    "check_valid",
    "get_valid_or_none",
    "shrink_q_range",
    "shrink_k_range",
    "shrink_d_range",
    "clone",
    "area",
    "cut_q",
    "cut_k",
    "get_rect_within_q_segment",
    "get_rect_within_k_segment",
    "intersection_q_id_on_left_boundary",
    "intersection_q_id_on_right_boundary",
    "is_full",
    "is_causal",
    "is_inv_causal",
    "is_bi_causal",
    "to_qk_range_mask_type",
]

_ATTN_RECTANGLES_METHODS = [
    "from_ranges",
    "append",
    "extend",
    "is_valid",
    "check_valid",
    "is_empty",
    "get_qo_ranges_union",
    "get_kv_ranges_union",
    "total_seqlen_qo",
    "total_seqlen_kv",
    "cut_q",
    "cut_k",
    "get_rects_within_q_segment",
    "get_rects_within_k_segment",
    "area",
]

_ATTN_MASK_TYPE_METHODS = [
    "from_int_type",
    "to_int_type",
]


def _has_cpp_backend():
    try:
        import magi_attention.magi_attn_ext  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cpp_backend(), reason="magi_attn_ext not available")
class TestSignatureAlignment:
    """Verify that Python and C++ backends have identical parameter names."""

    @pytest.mark.parametrize("method_name", _ATTN_RANGE_METHODS)
    def test_attn_range_signature(self, method_name):
        from magi_attention.common.range import AttnRange as PyClass
        from magi_attention.magi_attn_ext import AttnRange as CppClass

        py_params = _get_param_names_python(PyClass, method_name)
        cpp_params = _get_param_names_pybind(CppClass, method_name)
        assert (
            cpp_params is not None
        ), f"Could not parse C++ signature for {method_name}"
        assert (
            py_params == cpp_params
        ), f"AttnRange.{method_name} param mismatch: python={py_params} cpp={cpp_params}"

    @pytest.mark.parametrize("method_name", _ATTN_RANGES_METHODS)
    def test_attn_ranges_signature(self, method_name):
        from magi_attention.common.ranges import AttnRanges as PyClass
        from magi_attention.magi_attn_ext import AttnRanges as CppClass

        py_params = _get_param_names_python(PyClass, method_name)
        cpp_params = _get_param_names_pybind(CppClass, method_name)
        assert (
            cpp_params is not None
        ), f"Could not parse C++ signature for {method_name}"
        assert (
            py_params == cpp_params
        ), f"AttnRanges.{method_name} param mismatch: python={py_params} cpp={cpp_params}"

    @pytest.mark.parametrize("method_name", _ATTN_RECTANGLE_METHODS)
    def test_attn_rectangle_signature(self, method_name):
        from magi_attention.common.rectangle import AttnRectangle as PyClass
        from magi_attention.magi_attn_ext import AttnRectangle as CppClass

        py_params = _get_param_names_python(PyClass, method_name)
        cpp_params = _get_param_names_pybind(CppClass, method_name)
        assert (
            cpp_params is not None
        ), f"Could not parse C++ signature for {method_name}"
        assert (
            py_params == cpp_params
        ), f"AttnRectangle.{method_name} param mismatch: python={py_params} cpp={cpp_params}"

    @pytest.mark.parametrize("method_name", _ATTN_RECTANGLES_METHODS)
    def test_attn_rectangles_signature(self, method_name):
        from magi_attention.common.rectangles import AttnRectangles as PyClass
        from magi_attention.magi_attn_ext import AttnRectangles as CppClass

        py_params = _get_param_names_python(PyClass, method_name)
        cpp_params = _get_param_names_pybind(CppClass, method_name)
        assert (
            cpp_params is not None
        ), f"Could not parse C++ signature for {method_name}"
        assert (
            py_params == cpp_params
        ), f"AttnRectangles.{method_name} param mismatch: python={py_params} cpp={cpp_params}"

    @pytest.mark.parametrize("method_name", _ATTN_MASK_TYPE_METHODS)
    def test_attn_mask_type_signature(self, method_name):
        from magi_attention.common.enum import AttnMaskType as PyClass
        from magi_attention.magi_attn_ext import AttnMaskType as CppClass

        py_params = _get_param_names_python(PyClass, method_name)
        cpp_params = _get_param_names_pybind(CppClass, method_name)
        assert (
            cpp_params is not None
        ), f"Could not parse C++ signature for {method_name}"
        assert (
            py_params == cpp_params
        ), f"AttnMaskType.{method_name} param mismatch: python={py_params} cpp={cpp_params}"
