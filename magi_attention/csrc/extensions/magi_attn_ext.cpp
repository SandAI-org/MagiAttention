/**********************************************************************************
 * Copyright (c) 2025-2026 SandAI. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *********************************************************************************/

#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "magi_attn_ext.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MagiAttention CPP Extensions";

  // Bind the Host-side management class to Python
  py::class_<magi_attn_ext::KernelBarrier>(m, "KernelBarrier")
      .def(py::init<int>(), py::arg("target"))
      .def("reset", &magi_attn_ext::KernelBarrier::reset, "Reset the kernel_barrier count to 0")
      .def("get_value", &magi_attn_ext::KernelBarrier::get_value, "Get current kernel_barrier count value from GPU (for debugging)")
      .def("synchronize", &magi_attn_ext::KernelBarrier::synchronize, "Launch a spin kernel to wait until count >= target")
      .def("__repr__", [](const magi_attn_ext::KernelBarrier& self) { return "<KernelBarrier>"; });

  // ==========================================
  // Global Producer Function Binding
  // ==========================================
  m.def("produce", &magi_attn_ext::produce, py::arg("kernel_barrier"), "Launch a producer kernel to increment the kernel_barrier count by 1");

  // AttnRange
  py::class_<magi_attn_ext::AttnRange>(m, "AttnRange")
      .def(py::init<int, int>(), py::arg("start"), py::arg("end"))
      .def_static("from_range", &magi_attn_ext::AttnRange::from_range, py::arg("attn_range"), py::arg("check") = false)
      .def_readwrite("start", &magi_attn_ext::AttnRange::start)
      .def_readwrite("end", &magi_attn_ext::AttnRange::end)
      .def_property_readonly("seqlen", &magi_attn_ext::AttnRange::seqlen)
      .def("is_valid", &magi_attn_ext::AttnRange::is_valid)
      .def("is_valid_open", &magi_attn_ext::AttnRange::is_valid_open_py, py::arg("start") = py::none(), py::arg("end") = py::none())
      .def("is_valid_close", &magi_attn_ext::AttnRange::is_valid_close_py, py::arg("start") = py::none(), py::arg("end") = py::none())
      .def("check_valid", &magi_attn_ext::AttnRange::check_valid_py, py::arg("start") = py::none(), py::arg("end") = py::none())
      .def("is_empty", &magi_attn_ext::AttnRange::is_empty)
      .def("is_subrange_of", &magi_attn_ext::AttnRange::is_subrange_of, py::arg("other"))
      .def("is_overlap_with", &magi_attn_ext::AttnRange::is_overlap_with, py::arg("other"))
      .def("clone", &magi_attn_ext::AttnRange::clone)
      .def("offset", &magi_attn_ext::AttnRange::offset, py::arg("offset"))
      .def("truncate", &magi_attn_ext::AttnRange::truncate_py, py::arg("start") = py::none(), py::arg("end") = py::none())
      .def("intersect", &magi_attn_ext::AttnRange::intersect, py::arg("other"))
      .def("intersect_size", &magi_attn_ext::AttnRange::intersect_size, py::arg("other"))
      .def("union", &magi_attn_ext::AttnRange::union_range, py::arg("other"))
      .def("union_size", &magi_attn_ext::AttnRange::union_size, py::arg("other"))
      .def("diff_by", &magi_attn_ext::AttnRange::diff_by, py::arg("other"))
      .def("to_naive_range", &magi_attn_ext::AttnRange::to_naive_range)
      .def("to_string", &magi_attn_ext::AttnRange::to_string)
      .def("__repr__", &magi_attn_ext::AttnRange::to_repr)
      .def("__eq__", &magi_attn_ext::AttnRange::eq_py)
      .def("__ne__", &magi_attn_ext::AttnRange::ne_py)
      .def("__len__", &magi_attn_ext::AttnRange::seqlen)
      .def("__hash__", &magi_attn_ext::AttnRange::hash_py)
      .def(py::pickle([](const magi_attn_ext::AttnRange& r) { return r.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::AttnRange::setstate_py(t); }));

  // AttnMaskType
  py::enum_<magi_attn_ext::AttnMaskType>(m, "AttnMaskType")
      .value("FULL", magi_attn_ext::AttnMaskType::FULL)
      .value("CAUSAL", magi_attn_ext::AttnMaskType::CAUSAL)
      .value("INVCAUSAL", magi_attn_ext::AttnMaskType::INV_CAUSAL)
      .value("BICAUSAL", magi_attn_ext::AttnMaskType::BI_CAUSAL)
      .export_values()
      .def("to_int_type", &magi_attn_ext::attn_mask_type_to_int)
      .def_static("from_int_type", &magi_attn_ext::attn_mask_type_from_int);

  // AttnRanges
  py::class_<magi_attn_ext::AttnRanges>(m, "AttnRanges")
      .def(py::init<>())
      .def(py::init<std::vector<magi_attn_ext::AttnRange>>(), py::arg("other_ranges"))
      .def_static("from_ranges", &magi_attn_ext::AttnRanges::from_ranges, py::arg("ranges"), py::arg("check") = false)
      .def(
          "append",
          py::overload_cast<int, int, bool>(&magi_attn_ext::AttnRanges::append),
          py::arg("start"),
          py::arg("end"),
          py::arg("check") = false,
          "Append an AttnRange using (start, end) to the AttnRanges")
      .def(
          "append",
          py::overload_cast<const magi_attn_ext::AttnRange&, bool>(&magi_attn_ext::AttnRanges::append),
          py::arg("range"),
          py::arg("check") = false,
          "Append an AttnRange using lvalue reference to the AttnRanges")
      .def(
          "extend",
          py::overload_cast<const std::vector<magi_attn_ext::AttnRange>&, bool>(&magi_attn_ext::AttnRanges::extend),
          py::arg("other_ranges"),
          py::arg("check") = false,
          "Extend the AttnRanges with another AttnRanges")
      .def(
          "extend",
          py::overload_cast<const magi_attn_ext::AttnRanges&, bool>(&magi_attn_ext::AttnRanges::extend),
          py::arg("other"),
          py::arg("check") = false,
          "Extend the AttnRanges with another AttnRanges")
      .def_property_readonly("size", &magi_attn_ext::AttnRanges::size, "Return the size (vector length) of the AttnRanges")
      .def("__len__", &magi_attn_ext::AttnRanges::size)
      .def("is_empty", &magi_attn_ext::AttnRanges::is_empty, "Check if the AttnRanges is empty")
      .def("insert", &magi_attn_ext::AttnRanges::insert, py::arg("idx"), py::arg("range"), py::arg("check") = false)
      .def("pop", &magi_attn_ext::AttnRanges::pop, py::arg("idx") = -1)
      .def("clear_empty", &magi_attn_ext::AttnRanges::clear_empty)
      .def("chunk", &magi_attn_ext::AttnRanges::chunk, py::arg("chunk_size"), py::arg("check") = true)
      .def("truncate", &magi_attn_ext::AttnRanges::truncate, py::arg("start") = py::none(), py::arg("end") = py::none())
      .def("is_sorted", &magi_attn_ext::AttnRanges::is_sorted)
      .def("is_merged", &magi_attn_ext::AttnRanges::is_merged)
      .def("find_hole_ranges", &magi_attn_ext::AttnRanges::find_hole_ranges, py::arg("other"), py::arg("is_self_merged") = false, py::arg("is_other_merged") = false)
      .def(
          "find_overlap_ranges",
          &magi_attn_ext::AttnRanges::find_overlap_ranges,
          py::arg("other"),
          py::arg("is_self_merged") = false,
          py::arg("is_other_merged") = false)
      .def_static("from_cu_seqlens", &magi_attn_ext::AttnRanges::from_cu_seqlens, py::arg("cu_seqlens"), py::arg("seqlen"))
      .def("clone", &magi_attn_ext::AttnRanges::clone)
      .def("intersect_size", &magi_attn_ext::AttnRanges::intersect_size)
      .def("intersect_size_with", &magi_attn_ext::AttnRanges::intersect_size_with, py::arg("other"))
      .def("union_size", &magi_attn_ext::AttnRanges::union_size)
      .def("union_size_with", &magi_attn_ext::AttnRanges::union_size_with, py::arg("other"))
      .def_property_readonly("max_seqlen", &magi_attn_ext::AttnRanges::max_seqlen)
      .def_property_readonly("start", &magi_attn_ext::AttnRanges::start_val)
      .def_property_readonly("end", &magi_attn_ext::AttnRanges::end_val)
      .def_property_readonly("points", &magi_attn_ext::AttnRanges::points)
      .def("is_non_overlap", &magi_attn_ext::AttnRanges::is_non_overlap)
      .def("is_cu_seqlens", &magi_attn_ext::AttnRanges::is_cu_seqlens, py::arg("seqlen"))
      .def("to_cu_seqlens", &magi_attn_ext::AttnRanges::to_cu_seqlens, py::arg("seqlen"))
      .def("to_tensor", &magi_attn_ext::AttnRanges::to_tensor, py::arg("device") = "cpu")
      .def("to_naive_ranges", &magi_attn_ext::AttnRanges::to_naive_ranges)
      .def_property_readonly("total_seqlen", &magi_attn_ext::AttnRanges::total_seqlen)
      .def("merge", &magi_attn_ext::AttnRanges::merge)
      .def("sort", &magi_attn_ext::AttnRanges::sort_ranges)
      .def("sort_ranges", &magi_attn_ext::AttnRanges::sort_ranges)
      .def("is_valid", &magi_attn_ext::AttnRanges::is_valid)
      .def("check_valid", &magi_attn_ext::AttnRanges::check_valid)
      .def("clear", &magi_attn_ext::AttnRanges::clear, "Clear the AttnRanges")
      .def("reserve", &magi_attn_ext::AttnRanges::reserve, py::arg("capacity"), "Reserve the AttnRanges to a specific capacity")
      .def(
          "make_range_local",
          &magi_attn_ext::AttnRanges::make_range_local_py,
          py::arg("other_attn_range"),
          py::arg("is_self_merged") = false,
          py::arg("prefix_offset") = py::none())
      .def("make_ranges_local", &magi_attn_ext::AttnRanges::make_ranges_local, py::arg("other_attn_ranges"), py::arg("is_self_merged") = false)
      .def("__getitem__", &magi_attn_ext::AttnRanges::getitem_py, py::arg("index"))
      .def("__setitem__", &magi_attn_ext::AttnRanges::setitem_py, py::arg("index"), py::arg("range"))
      .def("__iter__", &magi_attn_ext::AttnRanges::iter_py, py::keep_alive<0, 1>())
      .def("__eq__", &magi_attn_ext::AttnRanges::eq_py)
      .def("__hash__", &magi_attn_ext::AttnRanges::hash_py)
      .def("__repr__", &magi_attn_ext::AttnRanges::to_repr)
      .def_property(
          "_ranges",
          &magi_attn_ext::AttnRanges::get_ranges_py,
          &magi_attn_ext::AttnRanges::set_ranges_py,
          "Internal list of AttnRange objects (compatible with Python version)")
      .def(py::pickle([](const magi_attn_ext::AttnRanges& rs) { return rs.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::AttnRanges::setstate_py(t); }));

  // AttnRectangle
  py::class_<magi_attn_ext::AttnRectangle>(m, "AttnRectangle")
      .def(py::init<const magi_attn_ext::AttnRange&, const magi_attn_ext::AttnRange&, const magi_attn_ext::AttnRange&>(), py::arg("q"), py::arg("k"), py::arg("d"))
      .def(py::init<const magi_attn_ext::AttnRange&, const magi_attn_ext::AttnRange&, magi_attn_ext::AttnMaskType>(), py::arg("q"), py::arg("k"), py::arg("mask_type"))
      .def(
          py::init([](const magi_attn_ext::AttnRange& q, const magi_attn_ext::AttnRange& k, int mask_type) {
            return new magi_attn_ext::AttnRectangle(q, k, static_cast<magi_attn_ext::AttnMaskType>(mask_type));
          }),
          py::arg("q"),
          py::arg("k"),
          py::arg("mask_type"))
      .def_property("q_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_q_range, py::const_), &magi_attn_ext::AttnRectangle::set_q_range)
      .def_property("k_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_k_range, py::const_), &magi_attn_ext::AttnRectangle::set_k_range)
      .def_property("d_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_d_range, py::const_), &magi_attn_ext::AttnRectangle::set_d_range)
      .def("get_q_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_q_range, py::const_), py::return_value_policy::reference_internal)
      .def("get_k_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_k_range, py::const_), py::return_value_policy::reference_internal)
      .def("get_d_range", py::overload_cast<>(&magi_attn_ext::AttnRectangle::get_d_range, py::const_), py::return_value_policy::reference_internal)
      .def("is_valid", &magi_attn_ext::AttnRectangle::is_valid, py::arg("q") = nullptr, py::arg("k") = nullptr, py::arg("d") = nullptr)
      .def("check_valid", &magi_attn_ext::AttnRectangle::check_valid, py::arg("q") = nullptr, py::arg("k") = nullptr, py::arg("d") = nullptr)
      .def("get_valid_or_none", &magi_attn_ext::AttnRectangle::get_valid_or_none, py::return_value_policy::reference_internal)
      .def("shrink_q_range", &magi_attn_ext::AttnRectangle::shrink_q_range)
      .def("shrink_k_range", &magi_attn_ext::AttnRectangle::shrink_k_range)
      .def("shrink_d_range", &magi_attn_ext::AttnRectangle::shrink_d_range)
      .def("area", &magi_attn_ext::AttnRectangle::area)
      .def("clone", &magi_attn_ext::AttnRectangle::clone)
      .def("cut_q", &magi_attn_ext::AttnRectangle::cut_q, py::arg("cut_pos"))
      .def("cut_k", &magi_attn_ext::AttnRectangle::cut_k, py::arg("cut_pos"))
      .def("get_rect_within_q_segment", &magi_attn_ext::AttnRectangle::get_rect_within_q_segment, py::arg("q_start"), py::arg("q_end"))
      .def("get_rect_within_k_segment", &magi_attn_ext::AttnRectangle::get_rect_within_k_segment, py::arg("k_start"), py::arg("k_end"))
      .def("intersection_q_id_on_left_boundary", &magi_attn_ext::AttnRectangle::intersection_q_id_on_left_boundary)
      .def("intersection_q_id_on_right_boundary", &magi_attn_ext::AttnRectangle::intersection_q_id_on_right_boundary)
      .def("is_full", &magi_attn_ext::AttnRectangle::is_full)
      .def("is_causal", &magi_attn_ext::AttnRectangle::is_causal)
      .def("is_inv_causal", &magi_attn_ext::AttnRectangle::is_inv_causal)
      .def("is_bi_causal", &magi_attn_ext::AttnRectangle::is_bi_causal)
      .def("to_qk_range_mask_type", &magi_attn_ext::AttnRectangle::to_qk_range_mask_type)
      .def("__len__", [](const magi_attn_ext::AttnRectangle&) { return 1; })
      .def("__eq__", &magi_attn_ext::AttnRectangle::operator==)
      .def("__hash__", &magi_attn_ext::AttnRectangle::hash_py)
      .def("__repr__", &magi_attn_ext::AttnRectangle::to_repr)
      .def(py::pickle([](const magi_attn_ext::AttnRectangle& r) { return r.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::AttnRectangle::setstate_py(t); }));

  // AttnRectangles
  py::class_<magi_attn_ext::AttnRectangles>(m, "AttnRectangles")
      .def(py::init<>())
      .def_static(
          "from_ranges", &magi_attn_ext::AttnRectangles::from_ranges_py, py::arg("q_ranges"), py::arg("k_ranges"), py::arg("mask_types"), py::arg("check") = false)
      .def_property_readonly("size", &magi_attn_ext::AttnRectangles::size)
      .def("is_empty", &magi_attn_ext::AttnRectangles::is_empty)
      .def("is_valid", &magi_attn_ext::AttnRectangles::is_valid)
      .def("check_valid", &magi_attn_ext::AttnRectangles::check_valid)
      .def("clear", &magi_attn_ext::AttnRectangles::clear)
      .def("append", &magi_attn_ext::AttnRectangles::append_py, py::arg("rect"), py::arg("check") = false)
      .def("extend", &magi_attn_ext::AttnRectangles::extend_py, py::arg("other"), py::arg("check") = false)
      .def("get_qo_ranges_union", &magi_attn_ext::AttnRectangles::get_qo_ranges_union)
      .def("get_kv_ranges_union", &magi_attn_ext::AttnRectangles::get_kv_ranges_union)
      .def("total_seqlen_qo", &magi_attn_ext::AttnRectangles::total_seqlen_qo)
      .def("total_seqlen_kv", &magi_attn_ext::AttnRectangles::total_seqlen_kv)
      .def("get_rects_within_q_segment", &magi_attn_ext::AttnRectangles::get_rects_within_q_segment, py::arg("q_start"), py::arg("q_end"))
      .def("get_rects_within_k_segment", &magi_attn_ext::AttnRectangles::get_rects_within_k_segment, py::arg("k_start"), py::arg("k_end"))
      .def("cut_q", &magi_attn_ext::AttnRectangles::cut_q, py::arg("cut_pos"))
      .def("cut_k", &magi_attn_ext::AttnRectangles::cut_k, py::arg("cut_pos"))
      .def("__len__", &magi_attn_ext::AttnRectangles::size)
      .def("__getitem__", &magi_attn_ext::AttnRectangles::get_item)
      .def("__setitem__", &magi_attn_ext::AttnRectangles::set_item)
      .def("__repr__", &magi_attn_ext::AttnRectangles::to_repr)
      .def("__hash__", &magi_attn_ext::AttnRectangles::hash_py)
      .def("__eq__", [](const magi_attn_ext::AttnRectangles& self, const magi_attn_ext::AttnRectangles& other) { return self == other; })
      .def("area", &magi_attn_ext::AttnRectangles::area)
      .def(
          py::pickle(
              [](const magi_attn_ext::AttnRectangles& rs) { return rs.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::AttnRectangles::setstate_py(t); }));

  m.def("is_valid_cu_seqlens", &magi_attn_ext::is_valid_cu_seqlens, py::arg("cu_seqlens"), py::arg("seq_len"));
  m.def("check_valid_cu_seqlens", &magi_attn_ext::check_valid_cu_seqlens, py::arg("cu_seqlens"), py::arg("seq_len"));

  // AttnRangeWithRank
  py::class_<magi_attn_ext::AttnRangeWithRank, magi_attn_ext::AttnRange>(m, "AttnRangeWithRank")
      .def(py::init<std::set<int>, int, int>(), py::arg("rank_set"), py::arg("start"), py::arg("end"))
      .def_readwrite("rank_set", &magi_attn_ext::AttnRangeWithRank::rank_set)
      .def_property_readonly("seqlen", &magi_attn_ext::AttnRangeWithRank::get_seqlen)
      .def("__len__", &magi_attn_ext::AttnRangeWithRank::get_seqlen)
      .def("clone", &magi_attn_ext::AttnRangeWithRank::clone)
      .def("__repr__", &magi_attn_ext::AttnRangeWithRank::to_repr)
      .def(
          py::pickle(
              [](const magi_attn_ext::AttnRangeWithRank& r) { return r.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::AttnRangeWithRank::setstate_py(t); }));

  // GroupCastRanges
  py::class_<magi_attn_ext::GroupCastRanges, magi_attn_ext::AttnRanges>(m, "GroupCastRanges")
      .def(py::init<>())
      .def(py::init<int, const std::vector<magi_attn_ext::AttnRanges>&, bool>(), py::arg("cp_size"), py::arg("ranges_per_rank"), py::arg("split") = true)
      .def("_split", &magi_attn_ext::GroupCastRanges::_split)
      .def("__repr__", &magi_attn_ext::GroupCastRanges::to_repr)
      .def_property(
          "_ranges",
          &magi_attn_ext::GroupCastRanges::get_group_cast_ranges_py,
          &magi_attn_ext::GroupCastRanges::set_group_cast_ranges_py,
          "Internal list of AttnRangeWithRank objects")
      .def(
          "__iter__",
          [](const magi_attn_ext::GroupCastRanges& self) { return py::make_iterator(self.group_cast_ranges.begin(), self.group_cast_ranges.end()); },
          py::keep_alive<0, 1>())
      .def(
          py::pickle(
              [](const magi_attn_ext::GroupCastRanges& rs) { return rs.getstate_py(); }, [](py::tuple t) { return magi_attn_ext::GroupCastRanges::setstate_py(t); }));

  // Binary-Greedy-Parallel solve (optimized version that moves more logic to C++)
  m.def(
      "binary_greedy_parallel_solve",
      &magi_attn_ext::binary_greedy_parallel_solve,
      py::arg("rects"),
      py::arg("host_ranges_q"),
      py::arg("host_ranges_k"),
      py::arg("num_heads_q"),
      py::arg("num_heads_kv"),
      py::arg("num_heads_group"),
      py::arg("bucket_per_rank"),
      py::arg("rank") = -1,
      py::arg("debug_print") = false,
      "Optimized Binary-Greedy-Parallel solver implemented in C++");

  // Optimized version of calc_host_and_remote_bucket_this_rank
  m.def(
      "cut_host_remote_buckets",
      &magi_attn_ext::cut_host_remote_buckets,
      py::arg("bucket_this_rank"),
      py::arg("host_ranges_q_this_rank"),
      py::arg("host_ranges_k_this_rank"),
      "Optimized version of calc_host_and_remote_bucket_this_rank");

  // Optimized version of _expand_attn_ranges for DynamicAttnSolver
  m.def(
      "expand_attn_ranges",
      &magi_attn_ext::expand_attn_ranges,
      py::arg("ranges"),
      py::arg("stride"),
      py::arg("num_heads_group"),
      "Optimized version of _expand_attn_ranges for DynamicAttnSolver");
}
