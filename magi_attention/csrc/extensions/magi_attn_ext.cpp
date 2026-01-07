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
#include <torch/python.h>

#include "magi_attn_ext.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MagiAttention CPP Extensions";

  // AttnRange
  py::class_<magi_attn_ext::AttnRange>(m, "AttnRange")
      .def(py::init<int, int>(), py::arg("start"), py::arg("end"))
      .def_readwrite("start", &magi_attn_ext::AttnRange::start)
      .def_readwrite("end", &magi_attn_ext::AttnRange::end)
      .def("is_valid", &magi_attn_ext::AttnRange::is_valid)
      .def("is_empty", &magi_attn_ext::AttnRange::is_empty)
      .def("seqlen", &magi_attn_ext::AttnRange::seqlen)
      .def("__repr__", [](const magi_attn_ext::AttnRange& r) { return "AttnRange(start=" + std::to_string(r.start) + ", end=" + std::to_string(r.end) + ")"; });

  // AttnRanges
  py::class_<magi_attn_ext::AttnRanges>(m, "AttnRanges")
      .def(py::init<>())
      .def(py::init<std::vector<magi_attn_ext::AttnRange>>(), py::arg("other_ranges"))
      .def(
          "append",
          py::overload_cast<int, int>(&magi_attn_ext::AttnRanges::append),
          py::arg("start"),
          py::arg("end"),
          "Append an AttnRange using (start, end) to the AttnRanges")
      .def(
          "append",
          py::overload_cast<const magi_attn_ext::AttnRange&>(&magi_attn_ext::AttnRanges::append),
          py::arg("range"),
          "Append an AttnRange using lvalue reference to the AttnRanges")
      .def("extend", &magi_attn_ext::AttnRanges::extend, py::arg("other_ranges"), "Extend the AttnRanges with another AttnRanges")
      .def("size", &magi_attn_ext::AttnRanges::size, "Return the size (vector length) of the AttnRanges")
      .def("is_empty", &magi_attn_ext::AttnRanges::is_empty, "Check if the AttnRanges is empty")
      .def("clear", &magi_attn_ext::AttnRanges::clear, "Clear the AttnRanges")
      .def("reserve", &magi_attn_ext::AttnRanges::reserve, py::arg("capacity"), "Reserve the AttnRanges to a specific capacity")
      .def(
          "__getitem__",
          [](magi_attn_ext::AttnRanges& self, size_t index) -> magi_attn_ext::AttnRange& {
            if (index >= self.size()) {
              throw py::index_error();
            }
            return self.get()[index];
          },
          py::return_value_policy::reference_internal,
          py::arg("index"))
      .def(
          "__setitem__",
          [](magi_attn_ext::AttnRanges& self, size_t index, const magi_attn_ext::AttnRange& range) {
            if (index >= self.size()) {
              throw py::index_error();
            }
            self.get()[index] = range;
          },
          py::arg("index"),
          py::arg("range"))
      .def("__repr__", [](const magi_attn_ext::AttnRanges& self) {
        std::string repr = "AttnRanges([";
        for (size_t i = 0; i < self.size(); ++i) {
          const auto& r = self.get()[i];
          repr += "(" + std::to_string(r.start) + ", " + std::to_string(r.end) + ")";
          if (i < self.size() - 1)
            repr += ", ";
        }
        repr += "])";
        return repr;
      });

  // Calculate simplex edges for binary greedy parallel algorithm
  m.def(
      "calc_simplex_edges",
      &magi_attn_ext::calc_simplex_edges,
      py::arg("cp_size"),
      py::arg("rank_m"),
      py::arg("rank_n"),
      py::arg("comm_len_m"),
      py::arg("comm_len_n"),
      py::arg("sparse_solver_map"),
      py::arg("sparse_area_map"),
      py::arg("area_avg"),
      py::arg("num_heads_q"),
      py::arg("num_heads_kv"),
      "Calculate simplex edges for the binary greedy parallel algorithm");

  // Greedy selection algorithm
  m.def(
      "greedy_selection",
      &magi_attn_ext::greedy_selection,
      py::arg("node_num"),
      py::arg("edges"),
      py::arg("threshold"),
      "Solve range selection problem using a greedy algorithm");

  // Greedy max flow algorithm
  m.def(
      "greedy_max_flow",
      &magi_attn_ext::greedy_max_flow,
      py::arg("cp_size"),
      py::arg("simplex_edges"),
      py::arg("simplex_selected_edges"),
      py::arg("sparse_area_map"),
      py::arg("rank_m"),
      py::arg("rank_n"),
      py::arg("usp_choices"),
      py::arg("area_avg"),
      py::arg("unbalance_rate"),
      "Greedy algorithm to replace max flow for task assignment");

  // Core Binary-Greedy solver (C++ implementation of Python binary_greedy)
  m.def(
      "binary_greedy_solver",
      &magi_attn_ext::binary_greedy_solver,
      py::arg("cp_size"),
      py::arg("rank_m"),
      py::arg("rank_n"),
      py::arg("comm_len_m"),
      py::arg("comm_len_n"),
      py::arg("sparse_area_map"),
      py::arg("num_heads_q"),
      py::arg("num_heads_kv"),
      py::arg("usp_choices"),
      py::arg("rank") = -1,
      py::arg("debug_print") = false,
      "Core Binary-Greedy solver implemented in C++");
}
