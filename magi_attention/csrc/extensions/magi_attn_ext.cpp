/**********************************************************************************
 * Copyright (c) 2025 SandAI. All Rights Reserved.
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

#include "magi_ext.hpp"


namespace magi_attn_ext {

} // namespace magi_attn_ext


namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MagiAttention CPP Extensions";

    // AttnRange
    py::class_<magi_attn_ext::AttnRange>(m, "AttnRange")
        .def(py::init<int, int>(),
            py::arg("start"),
            py::arg("end"))
        .def_readwrite("start", &magi_attn_ext::AttnRange::start)
        .def_readwrite("end", &magi_attn_ext::AttnRange::end)
        .def("is_valid", &magi_attn_ext::AttnRange::is_valid)
        .def("is_empty", &magi_attn_ext::AttnRange::is_empty)
        .def("seqlen", &magi_attn_ext::AttnRange::seqlen)
        .def("__repr__", [](const magi_attn_ext::AttnRange& r) {
            return "AttnRange(start=" + std::to_string(r.start) +
                   ", end=" + std::to_string(r.end) + ")";
        });

    // AttnRanges
    py::class_<magi_attn_ext::AttnRanges>(m, "AttnRanges")
        .def(py::init<>())
        .def(py::init<std::vector<magi_attn_ext::AttnRange>>(),
            py::arg("other_ranges"))
        .def("append", py::overload_cast<int, int>(&magi_attn_ext::AttnRanges::append),
            py::arg("start"), py::arg("end"),
            "Append an AttnRange using (start, end) to the AttnRanges")
        .def("append", py::overload_cast<const magi_attn_ext::AttnRange&>(&magi_attn_ext::AttnRanges::append),
            py::arg("range"),
            "Append an AttnRange using lvalue reference to the AttnRanges")
        .def("extend", &magi_attn_ext::AttnRanges::extend,
            py::arg("other_ranges"),
            "Extend the AttnRanges with another AttnRanges")
        .def("size", &magi_attn_ext::AttnRanges::size,
            "Return the size (vector length) of the AttnRanges")
        .def("is_empty", &magi_attn_ext::AttnRanges::is_empty,
            "Check if the AttnRanges is empty")
        .def("clear", &magi_attn_ext::AttnRanges::clear,
            "Clear the AttnRanges")
        .def("reserve", &magi_attn_ext::AttnRanges::reserve,
            py::arg("capacity"),
            "Reserve the AttnRanges to a specific capacity")
        .def("__getitem__", [](magi_attn_ext::AttnRanges& self, size_t index) -> magi_attn_ext::AttnRange& {
            if (index >= self.size()) {
                throw py::index_error();
            }
            return self.get()[index];
        }, py::return_value_policy::reference_internal, py::arg("index"))
        .def("__setitem__", [](magi_attn_ext::AttnRanges& self, size_t index, const magi_attn_ext::AttnRange& range) {
            if (index >= self.size()) {
                throw py::index_error();
            }
            self.get()[index] = range;
        }, py::arg("index"), py::arg("range"))
        .def("__repr__", [](const magi_attn_ext::AttnRanges& self) {
            std::string repr = "AttnRanges([";
            for (size_t i = 0; i < self.size(); ++i) {
                const auto& r = self.get()[i];
                repr += "(" + std::to_string(r.start) + ", " + std::to_string(r.end) + ")";
                if (i < self.size() - 1) repr += ", ";
            }
            repr += "])";
            return repr;
        });
}
