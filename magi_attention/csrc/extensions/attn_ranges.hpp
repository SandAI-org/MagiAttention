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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace magi_attn_ext {

struct AttnRange {
  int start = 0;
  int end = 0;

  virtual ~AttnRange() = default;

  AttnRange(int start, int end) : start(start), end(end) {
    check_valid();
  }

  AttnRange() = default;
  AttnRange(const AttnRange&) = default;
  AttnRange& operator=(const AttnRange&) = default;

  static AttnRange from_range(const pybind11::object& attn_range_obj, bool check = false) {
    AttnRange res;
    // Use try-catch instead of isinstance to avoid infinite recursion
    // when checking attributes during type inspection
    try {
      AttnRange other = attn_range_obj.cast<AttnRange>();
      res = AttnRange(other.start, other.end);
    } catch (const pybind11::cast_error&) {
      // Not an AttnRange, try as sequence
      if (pybind11::isinstance<pybind11::tuple>(attn_range_obj) || pybind11::isinstance<pybind11::list>(attn_range_obj)) {
        pybind11::sequence seq = attn_range_obj.cast<pybind11::sequence>();
        if (seq.size() < 2) {
          throw std::runtime_error("AttnRange.from_range: sequence must have at least 2 elements");
        }
        int start_val = seq[0].cast<int>();
        int end_val = seq[1].cast<int>();
        res = AttnRange(start_val, end_val);
      } else {
        throw pybind11::type_error("AttnRange.from_range: unsupported type");
      }
    }

    if (check) {
      res.check_valid();
    }

    return res;
  }

  bool operator==(const AttnRange& range) const {
    return start == range.start && end == range.end;
  }

  bool operator!=(const AttnRange& other) const {
    return !(*this == other);
  }

  bool is_valid() const {
    return start <= end;
  }

  void check_valid(std::optional<int> start_val = std::nullopt, std::optional<int> end_val = std::nullopt) const {
    int check_start = start_val.has_value() ? start_val.value() : start;
    int check_end = end_val.has_value() ? end_val.value() : end;
    if (!(check_start <= check_end)) {
      throw std::runtime_error("The attn_range (" + std::to_string(check_start) + ", " + std::to_string(check_end) + ") is invalid against the rule: 'start <= end'");
    }
  }

  bool is_empty() const {
    return seqlen() == 0;
  }

  int seqlen() const {
    return end - start;
  }

  bool is_valid_open(std::optional<int> start_val = std::nullopt, std::optional<int> end_val = std::nullopt) const {
    int check_start = start_val.has_value() ? start_val.value() : start;
    int check_end = end_val.has_value() ? end_val.value() : end;
    return check_start < check_end;
  }

  bool is_valid_close(std::optional<int> start_val = std::nullopt, std::optional<int> end_val = std::nullopt) const {
    int check_start = start_val.has_value() ? start_val.value() : start;
    int check_end = end_val.has_value() ? end_val.value() : end;
    return check_start <= check_end;
  }

  bool is_subrange_of(const AttnRange& other) const {
    return start >= other.start && end <= other.end;
  }

  bool is_overlap_with(const AttnRange& other) const {
    return !(start >= other.end || end <= other.start);
  }

  AttnRange clone() const {
    return AttnRange(start, end);
  }

  AttnRange offset(int offset_val) const {
    return AttnRange(start + offset_val, end + offset_val);
  }

  AttnRange truncate(std::optional<int> start_val = std::nullopt, std::optional<int> end_val = std::nullopt) const {
    int new_start = start_val.has_value() ? std::max(start, start_val.value()) : start;
    int new_end = end_val.has_value() ? std::min(end, end_val.value()) : end;
    // NOTE: if new_start > new_end, then return empty range: [new_start, new_start)
    return AttnRange(new_start, std::max(new_start, new_end));
  }

  AttnRange intersect(const AttnRange& other) const {
    int new_start = std::max(start, other.start);
    int new_end = std::min(end, other.end);
    return AttnRange(std::min(new_start, new_end), new_end);
  }

  int intersect_size(const AttnRange& other) const {
    return intersect(other).seqlen();
  }

  std::vector<AttnRange> union_range(const AttnRange& other) const {
    // REVIEW: Is this interface correct?
    if (is_empty() || other.is_empty()) {
      return {*this, other};
    }

    if (is_subrange_of(other)) {
      return {other};
    }

    if (other.is_subrange_of(*this)) {
      return {*this};
    }

    if (is_overlap_with(other)) {
      AttnRange union_range_obj(std::min(start, other.start), std::max(end, other.end));
      return {union_range_obj};
    }

    return {*this, other};
  }

  int union_size(const AttnRange& other) const {
    auto union_ranges = union_range(other);
    int total_size = 0;
    for (const auto& r : union_ranges) {
      total_size += r.seqlen();
    }
    return total_size;
  }

  std::vector<AttnRange> diff_by(const AttnRange& other) const {
    // other - self
    std::vector<AttnRange> diff_ranges;

    AttnRange inter_range = intersect(other);

    if (inter_range == *this) { // self is a subrange of other
      diff_ranges.push_back(AttnRange(other.start, start));
      diff_ranges.push_back(AttnRange(end, other.end));
    } else if (inter_range == other) { // other is a subrange of self
      diff_ranges.push_back(AttnRange(other.start, other.start));
    } else if (inter_range.is_empty()) { // self and other are disjoint
      diff_ranges.push_back(other);
    } else { // self and other are overlapping, but neither of them cover the other
      if (other.start < start) {
        diff_ranges.push_back(AttnRange(other.start, start));
      } else {
        diff_ranges.push_back(AttnRange(end, other.end));
      }
    }

    // Filter out empty ranges
    std::vector<AttnRange> non_empty_ranges;
    for (const auto& diff_range : diff_ranges) {
      if (!diff_range.is_empty()) {
        non_empty_ranges.push_back(diff_range);
      }
    }

    return non_empty_ranges;
  }

  std::string to_string() const {
    return "[" + std::to_string(start) + ", " + std::to_string(end) + ")";
  }

  pybind11::tuple to_naive_range() const {
    return pybind11::make_tuple(start, end);
  }

  std::string to_repr() const {
    return "[" + std::to_string(start) + ", " + std::to_string(end) + ")";
  }

  // Python binding helper functions
  bool is_valid_open_py(pybind11::object start_obj, pybind11::object end_obj) const {
    std::optional<int> start_val = start_obj.is_none() ? std::nullopt : std::optional<int>(start_obj.cast<int>());
    std::optional<int> end_val = end_obj.is_none() ? std::nullopt : std::optional<int>(end_obj.cast<int>());
    return is_valid_open(start_val, end_val);
  }

  bool is_valid_close_py(pybind11::object start_obj, pybind11::object end_obj) const {
    std::optional<int> start_val = start_obj.is_none() ? std::nullopt : std::optional<int>(start_obj.cast<int>());
    std::optional<int> end_val = end_obj.is_none() ? std::nullopt : std::optional<int>(end_obj.cast<int>());
    return is_valid_close(start_val, end_val);
  }

  void check_valid_py(pybind11::object start_obj, pybind11::object end_obj) const {
    try {
      std::optional<int> start_val = start_obj.is_none() ? std::nullopt : std::optional<int>(start_obj.cast<int>());
      std::optional<int> end_val = end_obj.is_none() ? std::nullopt : std::optional<int>(end_obj.cast<int>());
      check_valid(start_val, end_val);
    } catch (const std::runtime_error& e) {
      // Convert std::runtime_error to Python RangeError
      pybind11::object range_error_class = pybind11::module_::import("magi_attention.common.range").attr("RangeError");
      // Create the exception instance using PyObject_CallFunction
      PyObject* exc = PyObject_CallFunction(range_error_class.ptr(), "s", e.what());
      if (exc) {
        PyErr_SetObject((PyObject*)Py_TYPE(exc), exc);
        Py_DECREF(exc);
      } else {
        // Fallback if call fails
        PyErr_SetString(PyExc_RuntimeError, e.what());
      }
      throw pybind11::error_already_set();
    }
  }

  AttnRange truncate_py(pybind11::object start_obj, pybind11::object end_obj) const {
    std::optional<int> start_val = start_obj.is_none() ? std::nullopt : std::optional<int>(start_obj.cast<int>());
    std::optional<int> end_val = end_obj.is_none() ? std::nullopt : std::optional<int>(end_obj.cast<int>());
    return truncate(start_val, end_val);
  }

  bool eq_py(pybind11::object other) const {
    // Use try-catch instead of isinstance to avoid infinite recursion
    try {
      AttnRange other_range = other.cast<AttnRange>();
      return operator==(other_range);
    } catch (const pybind11::cast_error&) {
      return false;
    }
  }

  bool ne_py(pybind11::object other) const {
    // Use try-catch instead of isinstance to avoid infinite recursion
    try {
      AttnRange other_range = other.cast<AttnRange>();
      return operator!=(other_range);
    } catch (const pybind11::cast_error&) {
      return true;
    }
  }

  size_t hash_py() const {
    return std::hash<int>{}(start) ^ (std::hash<int>{}(end) << 1);
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(start, end);
  }

  static AttnRange setstate_py(pybind11::tuple t) {
    if (t.size() != 2)
      throw std::runtime_error("Invalid state for AttnRange!");
    return AttnRange(t[0].cast<int>(), t[1].cast<int>());
  }
};

struct AttnRanges {
  std::vector<AttnRange> ranges;
  mutable std::map<int, std::pair<int, AttnRange>> local_range_cache;

  virtual ~AttnRanges() = default;

  AttnRanges() = default;

  explicit AttnRanges(std::vector<AttnRange>&& other_ranges) : ranges(std::move(other_ranges)) {}

  void append(int start, int end, bool check = false) {
    if (check) {
      if (!(start <= end)) {
        throw pybind11::value_error("The attn_range (" + std::to_string(start) + ", " + std::to_string(end) + ") is invalid against the rule: 'start <= end'");
      }
    }
    local_range_cache.clear();
    ranges.emplace_back(start, end);
  }

  void append(const AttnRange& range, bool check = false) {
    if (check) {
      range.check_valid();
    }
    local_range_cache.clear();
    ranges.push_back(range);
  }

  void append(AttnRange&& range) {
    local_range_cache.clear();
    ranges.emplace_back(std::move(range));
  }

  void insert(size_t idx, const AttnRange& range, bool check = false) {
    if (check) {
      range.check_valid();
    }
    local_range_cache.clear();
    if (idx >= ranges.size()) {
      ranges.push_back(range);
    } else {
      ranges.insert(ranges.begin() + idx, range);
    }
  }

  void extend(const std::vector<AttnRange>& other_ranges, bool check = false) {
    if (check) {
      for (const auto& r : other_ranges) {
        r.check_valid();
      }
    }
    local_range_cache.clear();
    ranges.insert(ranges.end(), other_ranges.begin(), other_ranges.end());
  }

  void extend(const AttnRanges& other, bool check = false) {
    extend(other.ranges, check);
  }

  AttnRange pop(int idx = -1) {
    if (ranges.empty()) {
      throw std::out_of_range("pop from empty AttnRanges");
    }
    local_range_cache.clear();
    size_t actual_idx = (idx < 0) ? (ranges.size() + idx) : static_cast<size_t>(idx);
    if (actual_idx >= ranges.size()) {
      throw std::out_of_range("pop index out of range");
    }
    AttnRange res = ranges[actual_idx];
    ranges.erase(ranges.begin() + actual_idx);
    return res;
  }

  AttnRanges clear_empty() const {
    AttnRanges non_empty_ranges;
    for (const auto& r : ranges) {
      if (!r.is_empty()) {
        non_empty_ranges.append(r);
      }
    }
    return non_empty_ranges;
  }

  std::vector<AttnRanges> chunk(int chunk_size, bool check = true) const {
    if (check) {
      if (!is_non_overlap()) {
        PyErr_SetString(PyExc_AssertionError, "the ranges should be non-overlap if needed to be chunked");
        throw pybind11::error_already_set();
      }
    }

    std::vector<AttnRanges> chunked_ranges_list;
    AttnRanges chunked_ranges;
    int cnt = 0;
    for (const auto& attn_range : ranges) {
      int seqlen = attn_range.seqlen();
      int start = attn_range.start;
      int new_cnt = cnt + seqlen;
      while (new_cnt >= chunk_size) {
        int seqlen_truc = chunk_size - cnt;
        int end = start + seqlen_truc;
        chunked_ranges.append(AttnRange(start, end));
        chunked_ranges_list.push_back(std::move(chunked_ranges));

        chunked_ranges = AttnRanges();
        new_cnt -= chunk_size;
        start = end;
        cnt = 0;
      }
      cnt = new_cnt;

      if (cnt > 0) {
        chunked_ranges.append(AttnRange(start, attn_range.end));
      }
    }

    if (!chunked_ranges.is_empty()) {
      chunked_ranges_list.push_back(std::move(chunked_ranges));
    }

    return chunked_ranges_list;
  }

  AttnRanges truncate(std::optional<int> start_opt = std::nullopt, std::optional<int> end_opt = std::nullopt) const {
    AttnRanges trunc_ranges;
    for (const auto& r : ranges) {
      AttnRange tr = r.truncate(start_opt, end_opt);
      if (tr.is_empty()) {
        continue;
      }
      trunc_ranges.append(tr);
    }
    return trunc_ranges;
  }

  bool is_sorted() const {
    for (size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i - 1].start > ranges[i].start) {
        return false;
      }
    }
    return true;
  }

  bool is_merged() const {
    if (!is_sorted())
      return false;
    for (size_t i = 1; i < ranges.size(); ++i) {
      if (!(ranges[i - 1].end < ranges[i].start)) {
        return false;
      }
    }
    return true;
  }

  const std::vector<AttnRange>& get() const {
    return ranges;
  }

  std::vector<AttnRange>& get() {
    return ranges;
  }

  AttnRange& at(size_t idx) {
    if (idx >= ranges.size()) {
      throw std::out_of_range("AttnRanges idx out of range");
    }
    return ranges[idx];
  }

  const AttnRange& at(size_t idx) const {
    if (idx >= ranges.size()) {
      throw std::out_of_range("AttnRanges idx out of range");
    }
    return ranges[idx];
  }

  AttnRange& operator[](size_t idx) {
    return ranges[idx];
  }

  const AttnRange& operator[](size_t idx) const {
    return ranges[idx];
  }

  size_t size() const {
    return ranges.size();
  }

  bool is_empty() const {
    return ranges.empty();
  }

  void clear() {
    local_range_cache.clear();
    ranges.clear();
  }

  void reserve(size_t capacity) {
    ranges.reserve(capacity);
  }

  bool operator==(const AttnRanges& other) const {
    return ranges == other.ranges;
  }

  bool operator!=(const AttnRanges& other) const {
    return !(*this == other);
  }

  AttnRanges clone() const {
    std::vector<AttnRange> new_ranges;
    new_ranges.reserve(ranges.size());
    for (const auto& r : ranges) {
      new_ranges.push_back(r.clone());
    }
    return AttnRanges(std::move(new_ranges));
  }

  static AttnRanges from_cu_seqlens(const std::vector<int>& cu_seqlens, int seqlen) {
    // Note: check_valid_cu_seqlens logic
    if (cu_seqlens.empty())
      return AttnRanges();
    if (cu_seqlens[0] != 0)
      throw std::invalid_argument("cu_seqlens[0] must be 0");
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
      if (cu_seqlens[i - 1] >= cu_seqlens[i]) {
        throw std::invalid_argument("cu_seqlens must be strictly increasing");
      }
    }
    if (cu_seqlens.back() != seqlen) {
      throw std::invalid_argument("cu_seqlens[-1] must be equal to seqlen");
    }

    AttnRanges res;
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
      res.append(AttnRange(cu_seqlens[i - 1], cu_seqlens[i]));
    }
    return res;
  }

  int intersect_size() const {
    if (ranges.empty() || ranges.size() == 1) {
      return 0;
    }

    std::vector<std::pair<int, int>> events;
    events.reserve(ranges.size() * 2);
    for (const auto& r : ranges) {
      events.push_back({r.start, 1});
      events.push_back({r.end, -1});
    }

    std::sort(events.begin(), events.end());

    int count = 0;
    int last_pos = 0;
    int overlap_size = 0;

    for (const auto& event : events) {
      int pos = event.first;
      int type = event.second;
      if (count > 1) {
        overlap_size += (count - 1) * (pos - last_pos);
      }
      count += type;
      last_pos = pos;
    }

    return overlap_size;
  }

  int intersect_size_with(const AttnRanges& other) const {
    AttnRanges total_ranges = clone();
    total_ranges.extend(other.ranges);

    AttnRanges overlap = find_overlap_ranges(other);
    AttnRanges merged_total = total_ranges.merge();
    AttnRanges non_overlap_ranges = merged_total.find_hole_ranges(overlap);

    AttnRanges intersec_ranges;
    for (const auto& r : total_ranges.ranges) {
      AttnRanges ranges_worker;
      ranges_worker.append(r);
      intersec_ranges.extend(ranges_worker.find_hole_ranges(non_overlap_ranges).ranges);
    }

    return intersec_ranges.intersect_size();
  }

  int union_size() const {
    return total_seqlen();
  }

  int union_size_with(const AttnRanges& other) const {
    return total_seqlen() + other.total_seqlen();
  }

  int max_seqlen() const {
    if (is_empty())
      return 0;
    int max_len = 0;
    for (const auto& r : ranges) {
      max_len = std::max(max_len, r.seqlen());
    }
    return max_len;
  }

  int start_val() const {
    if (is_empty())
      throw std::runtime_error("The ranges is empty, there is no start");
    int min_start = ranges[0].start;
    for (const auto& r : ranges) {
      min_start = std::min(min_start, r.start);
    }
    return min_start;
  }

  int end_val() const {
    if (is_empty())
      throw std::runtime_error("The ranges is empty, there is no end");
    int max_end = ranges[0].end;
    for (const auto& r : ranges) {
      max_end = std::max(max_end, r.end);
    }
    return max_end;
  }

  std::vector<int> points() const {
    std::map<int, bool> point_map;
    for (const auto& r : ranges) {
      point_map[r.start] = true;
      point_map[r.end] = true;
    }
    std::vector<int> res;
    res.reserve(point_map.size());
    for (auto const& [key, val] : point_map) {
      res.push_back(key);
    }
    return res;
  }

  AttnRanges find_hole_ranges(const AttnRanges& other, bool is_self_merged = false, bool is_other_merged = false) const {
    AttnRanges ranges1 = is_self_merged ? clone() : merge();
    AttnRanges ranges2 = is_other_merged ? other.clone() : other.merge();

    size_t p1 = 0;
    size_t p2 = 0;
    AttnRanges hole_ranges;

    while (p1 < ranges1.size() && p2 < ranges2.size()) {
      AttnRange& r1 = ranges1.ranges[p1];
      const AttnRange& r2 = ranges2.ranges[p2];

      bool r1_end_greater = r1.end > r2.end;

      if (r1.start < r2.start) {
        hole_ranges.append(AttnRange(r1.start, std::min(r1.end, r2.start)));
      }

      if (r1.start < r2.end) {
        r1.start = std::max(r1.start, r2.end);
        if (r1.start > r1.end)
          r1.start = r1.end; // Ensure validity
      }

      if (r1_end_greater) {
        p2++;
      } else {
        p1++;
      }
    }

    for (size_t i = p1; i < ranges1.size(); ++i) {
      hole_ranges.append(ranges1.ranges[i]);
    }

    return hole_ranges;
  }

  AttnRanges find_overlap_ranges(const AttnRanges& other, bool is_self_merged = false, bool is_other_merged = false) const {
    AttnRanges ranges1 = is_self_merged ? clone() : merge();
    AttnRanges ranges2 = is_other_merged ? other.clone() : other.merge();

    size_t p1 = 0;
    size_t p2 = 0;
    AttnRanges overlap_ranges;

    while (p1 < ranges1.size() && p2 < ranges2.size()) {
      const AttnRange& r1 = ranges1.ranges[p1];
      const AttnRange& r2 = ranges2.ranges[p2];

      if (r1.is_overlap_with(r2)) {
        overlap_ranges.append(r1.intersect(r2));
      }

      if (r1.end > r2.end) {
        p2++;
      } else {
        p1++;
      }
    }

    return overlap_ranges;
  }

  static AttnRanges from_ranges(const pybind11::object& obj, bool check) {
    AttnRanges ranges;
    // Use try-catch instead of isinstance to avoid infinite recursion
    try {
      ranges = obj.cast<AttnRanges>().clone();
    } catch (const pybind11::cast_error&) {
      // Not an AttnRanges C++ instance, try as sequence or list
      bool handled = false;
      try {
        if (pybind11::isinstance<pybind11::list>(obj) || pybind11::isinstance<pybind11::tuple>(obj)) {
          pybind11::sequence seq = obj.cast<pybind11::sequence>();
          for (auto item : seq) {
            ranges.append(AttnRange::from_range(pybind11::reinterpret_borrow<pybind11::object>(item)));
          }
          handled = true;
        } else {
          // Try to get _ranges if it's a Python-side AttnRanges
          pybind11::list list_obj = pybind11::getattr(obj, "_ranges").cast<pybind11::list>();
          for (auto item : list_obj) {
            ranges.append(AttnRange::from_range(pybind11::reinterpret_borrow<pybind11::object>(item)));
          }
          handled = true;
        }
      } catch (...) {
      }

      if (!handled) {
        throw pybind11::type_error("Unsupported type for AttnRanges.from_ranges");
      }
    }

    if (check) {
      ranges.check_valid();
    }
    return ranges;
  }

  torch::Tensor to_tensor(const pybind11::object& device_obj) const {
    std::string device_str = "cpu";
    if (pybind11::isinstance<pybind11::str>(device_obj)) {
      device_str = device_obj.cast<std::string>();
    } else if (pybind11::isinstance<pybind11::int_>(device_obj)) {
      device_str = "cuda:" + std::to_string(device_obj.cast<int>());
    } else {
      device_str = pybind11::str(device_obj).cast<std::string>();
    }

    if (is_empty()) {
      return torch::empty({0, 2}, torch::dtype(torch::kInt32).device(device_str));
    }
    std::vector<int> data;
    data.reserve(size() * 2);
    for (const auto& r : get()) {
      data.push_back(r.start);
      data.push_back(r.end);
    }
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor cpu_tensor = torch::from_blob(data.data(), {static_cast<long>(size()), 2}, options).clone();
    return cpu_tensor.to(device_str);
  }

  pybind11::list to_naive_ranges() const {
    pybind11::list res;
    for (const auto& r : get()) {
      res.append(pybind11::make_tuple(r.start, r.end));
    }
    return res;
  }

  AttnRange& at_index(size_t index) {
    if (index >= size()) {
      throw pybind11::index_error();
    }
    return get()[index];
  }

  const AttnRange& at_index(size_t index) const {
    if (index >= size()) {
      throw pybind11::index_error();
    }
    return get()[index];
  }

  void set_at_index(size_t index, const AttnRange& range) {
    if (index >= size()) {
      throw pybind11::index_error();
    }
    get()[index] = range;
  }

  // FIXME: getitem_py returns a copy of AttnRange, which may lead to unexpected behavior if modifying in-place.
  pybind11::object getitem_py(pybind11::object index) const {
    if (pybind11::isinstance<pybind11::slice>(index)) {
      pybind11::slice slice(index);
      size_t start, stop, step, slicelength;
      if (!slice.compute(size(), &start, &stop, &step, &slicelength))
        throw pybind11::error_already_set();
      AttnRanges result;
      result.reserve(slicelength);
      for (size_t i = 0; i < slicelength; ++i) {
        result.append(at_index(start));
        start += step;
      }
      return pybind11::cast(result);
    }

    long long idx = index.cast<long long>();
    if (idx < 0)
      idx += (long long)size();
    if (idx < 0 || idx >= (long long)size()) {
      throw pybind11::index_error();
    }
    return pybind11::cast(ranges[(size_t)idx]);
  }

  void setitem_py(pybind11::object index, pybind11::object value) {
    if (pybind11::isinstance<pybind11::slice>(index)) {
      pybind11::slice slice(index);

      // Use try-catch instead of isinstance to avoid infinite recursion
      AttnRanges* src_ptr = nullptr;
      try {
        src_ptr = &value.cast<AttnRanges&>();
      } catch (const pybind11::cast_error&) {
        throw pybind11::type_error("Assignment value must be AttnRanges");
      }
      auto& src = *src_ptr;

      // Strict check to match Python's assert idx.stop - idx.start == len(value)
      try {
        if (!slice.attr("start").is_none() && !slice.attr("stop").is_none()) {
          long long s_start = slice.attr("start").cast<long long>();
          long long s_stop = slice.attr("stop").cast<long long>();
          if (s_stop - s_start != (long long)src.size()) {
            PyErr_SetString(PyExc_AssertionError, "slice assignment size mismatch");
            throw pybind11::error_already_set();
          }
        }
      } catch (const pybind11::cast_error&) {
        // If casting fails, we'll rely on compute() below
      }

      size_t start, stop, step, slicelength;
      if (!slice.compute(size(), &start, &stop, &step, &slicelength))
        throw pybind11::error_already_set();

      local_range_cache.clear();
      if (step == 1) {
        // Standard slice assignment: can change size
        auto it_start = ranges.begin() + start;
        auto it_end = ranges.begin() + stop;
        ranges.erase(it_start, it_end);
        ranges.insert(ranges.begin() + start, src.ranges.begin(), src.ranges.end());
      } else {
        // Extended slice assignment: size must not change
        if (src.size() != slicelength) {
          throw pybind11::value_error("attempt to assign sequence of size " + std::to_string(src.size()) + " to extended slice of size " + std::to_string(slicelength));
        }
        for (size_t i = 0; i < slicelength; ++i) {
          ranges[start] = src[i];
          start += step;
        }
      }
      return;
    }

    long long idx = index.cast<long long>();
    if (idx < 0)
      idx += (long long)size();
    if (idx < 0 || idx >= (long long)size()) {
      throw pybind11::index_error();
    }
    set_at_index((size_t)idx, value.cast<AttnRange>());
  }

  pybind11::iterator iter_py() const {
    // Return iterator over internal ranges
    return pybind11::make_iterator(ranges.begin(), ranges.end());
  }

  bool eq_py(const pybind11::object& other) const {
    // Use try-catch instead of isinstance to avoid infinite recursion
    try {
      return *this == other.cast<AttnRanges>();
    } catch (const pybind11::cast_error&) {
      return false;
    }
  }

  size_t hash_py() const {
    size_t seed = 0;
    for (const auto& r : ranges) {
      seed ^= r.hash_py() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(ranges);
  }

  static AttnRanges setstate_py(pybind11::tuple t) {
    if (t.size() != 1)
      throw std::runtime_error("Invalid state for AttnRanges!");
    return AttnRanges(t[0].cast<std::vector<AttnRange>>());
  }

  std::string to_repr() const {
    if (is_empty()) {
      return "[[,)]";
    }
    std::string repr = "[";
    for (size_t i = 0; i < size(); ++i) {
      repr += ranges[i].to_repr();
      if (i < size() - 1)
        repr += ", ";
    }
    repr += "]";
    return repr;
  }

  bool is_valid() const {
    for (const auto& r : ranges) {
      if (!r.is_valid())
        return false;
    }
    return true;
  }

  void check_valid() const {
    if (!is_valid()) {
      throw pybind11::value_error("Some of the ranges is invalid against the rule: 'start <= end'");
    }
  }

  int total_seqlen() const {
    int total_seqlen = 0;
    for (const auto& range : ranges) {
      total_seqlen += range.seqlen();
    }
    return total_seqlen;
  }

  bool is_non_overlap() const {
    return total_seqlen() == merge().total_seqlen();
  }

  bool is_cu_seqlens(int seqlen) const {
    if (ranges.empty())
      return seqlen == 0;
    if (ranges[0].start != 0)
      return false;
    for (size_t i = 1; i < ranges.size(); ++i) {
      if (ranges[i - 1].end != ranges[i].start)
        return false;
    }
    if (ranges.back().end != seqlen)
      return false;
    return true;
  }

  std::vector<int> to_cu_seqlens(int seqlen) const {
    if (!is_cu_seqlens(seqlen)) {
      throw pybind11::value_error("The ranges can not be converted to cu_seqlens");
    }
    std::vector<int> cu_seqlens;
    cu_seqlens.reserve(ranges.size() + 1);
    cu_seqlens.push_back(0);
    for (const auto& r : ranges) {
      cu_seqlens.push_back(r.end);
    }
    return cu_seqlens;
  }

  AttnRanges sort_ranges() const {
    std::vector<AttnRange> sorted_ranges = ranges;

    std::stable_sort(sorted_ranges.begin(), sorted_ranges.end(), [](const AttnRange& a, const AttnRange& b) { return a.start < b.start; });

    AttnRanges attn_ranges(std::move(sorted_ranges));

    return attn_ranges;
  }

  AttnRanges merge() const {
    AttnRanges sorted_ranges_obj = sort_ranges();
    const auto& _ranges = sorted_ranges_obj.ranges;
    AttnRanges _merged_ranges;

    int start = std::numeric_limits<int>::min(), end = std::numeric_limits<int>::min();
    for (size_t i = 0; i < _ranges.size(); i++) {
      const AttnRange& attn_range = _ranges[i];
      if (start == std::numeric_limits<int>::min()) {
        start = attn_range.start;
        end = attn_range.end;
        _merged_ranges.append(AttnRange(start, end));
      } else if (attn_range.start > end) {
        start = attn_range.start;
        end = attn_range.end;
        _merged_ranges.append(AttnRange(start, end));
      } else if (attn_range.end > end) {
        end = attn_range.end;
        _merged_ranges.at(_merged_ranges.size() - 1).end = end;
      }
    }

    return _merged_ranges;
  }

  std::pair<AttnRange, AttnRange> make_range_local(const AttnRange& other_attn_range, bool is_self_merged = false, const std::vector<int>* prefix_offset_ptr = nullptr)
      const {
    AttnRanges merged_ranges_obj;
    const AttnRanges* merged_ranges_ptr;
    if (is_self_merged) {
      merged_ranges_ptr = this;
    } else {
      merged_ranges_obj = merge();
      merged_ranges_ptr = &merged_ranges_obj;
    }

    std::vector<int> prefix_offset;
    if (prefix_offset_ptr == nullptr) {
      prefix_offset.reserve(merged_ranges_ptr->size());
      int current_offset = 0;
      for (const auto& range : merged_ranges_ptr->ranges) {
        prefix_offset.push_back(current_offset);
        current_offset += range.seqlen();
      }
      prefix_offset_ptr = &prefix_offset;
    }

    int left = 0, right = (int)merged_ranges_ptr->size() - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      if (merged_ranges_ptr->ranges[mid].start > other_attn_range.start) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }
    int le_idx = right;

    if (le_idx < 0 || le_idx >= (int)merged_ranges_ptr->size()) {
      throw pybind11::value_error("The attn_range " + other_attn_range.to_repr() + " is not in the (even merged) attn_ranges " + merged_ranges_ptr->to_repr());
    }

    const AttnRange& target_range = merged_ranges_ptr->ranges[le_idx];
    if (other_attn_range.is_subrange_of(target_range)) {
      int start = (*prefix_offset_ptr)[le_idx] + other_attn_range.start - target_range.start;
      return {AttnRange(start, start + other_attn_range.seqlen()), target_range};
    } else {
      throw pybind11::value_error("The attn_range " + other_attn_range.to_repr() + " is not in the (even merged) attn_ranges " + merged_ranges_ptr->to_repr());
    }
  }

  AttnRanges make_ranges_local(const AttnRanges& other_attn_ranges, bool is_self_merged = false) const {
    AttnRanges local_ranges;
    AttnRanges merged_ranges_obj;
    const AttnRanges* merged_ranges_ptr;
    if (is_self_merged) {
      merged_ranges_ptr = this;
    } else {
      merged_ranges_obj = merge();
      merged_ranges_ptr = &merged_ranges_obj;
    }

    if (local_range_cache.empty() && merged_ranges_ptr->size() > 0) {
      int local_index = 0;
      for (const auto& range : merged_ranges_ptr->ranges) {
        local_range_cache[range.start] = {local_index, range};
        local_index += range.seqlen();
        local_range_cache[range.end] = {local_index, range};
      }
    }

    std::vector<int> prefix_offset;
    prefix_offset.reserve(merged_ranges_ptr->size());
    int current_offset = 0;
    for (const auto& range : merged_ranges_ptr->ranges) {
      prefix_offset.push_back(current_offset);
      current_offset += range.seqlen();
    }

    for (const auto& attn_range : other_attn_ranges.ranges) {
      AttnRange local_range;
      auto it_start = local_range_cache.find(attn_range.start);
      if (it_start != local_range_cache.end()) {
        int start = it_start->second.first;
        const AttnRange& target_range = it_start->second.second;
        if (!attn_range.is_subrange_of(target_range)) {
          throw std::runtime_error("target range should cover attn_range");
        }
        local_range = AttnRange(start, start + attn_range.seqlen());
        local_range_cache[attn_range.end] = {start + attn_range.seqlen(), target_range};
      } else {
        auto it_end = local_range_cache.find(attn_range.end);
        if (it_end != local_range_cache.end()) {
          int end = it_end->second.first;
          const AttnRange& target_range = it_end->second.second;
          if (!attn_range.is_subrange_of(target_range)) {
            throw std::runtime_error("target range should cover attn_range");
          }
          local_range = AttnRange(end - attn_range.seqlen(), end);
          local_range_cache[attn_range.start] = {end - attn_range.seqlen(), target_range};
        } else {
          auto res = merged_ranges_ptr->make_range_local(attn_range, true, &prefix_offset);
          local_range = res.first;
          const AttnRange& target_range = res.second;
          local_range_cache[attn_range.start] = {local_range.start, target_range};
          local_range_cache[attn_range.end] = {local_range.end, target_range};
        }
      }
      local_ranges.append(local_range);
    }
    return local_ranges;
  }

  pybind11::tuple make_range_local_py(const AttnRange& other, bool is_self_merged, pybind11::object prefix_offset_obj = pybind11::none()) const {
    const std::vector<int>* prefix_offset_ptr = nullptr;
    std::vector<int> prefix_offset;
    if (!prefix_offset_obj.is_none()) {
      prefix_offset = prefix_offset_obj.cast<std::vector<int>>();
      prefix_offset_ptr = &prefix_offset;
    }
    auto res = make_range_local(other, is_self_merged, prefix_offset_ptr);
    return pybind11::make_tuple(res.first, res.second);
  }

  std::string to_string() const {
    std::string result = "[";
    for (size_t i = 0; i < ranges.size(); i++) {
      result += ranges[i].to_string();
      if (i != ranges.size() - 1) {
        result += ", ";
      }
    }
    return result;
  }

  pybind11::object get_ranges_py() const {
    // Return a list of internal ranges
    pybind11::list result;
    for (const auto& r : ranges) {
      result.append(r);
    }
    return result;
  }

  void set_ranges_py(pybind11::object ranges_obj) {
    // Note: Removed the recursive self.attr("_ranges") = ranges_obj call
    // which caused infinite recursion because this method IS the setter for _ranges.

    // Try to update internal ranges if all items are AttnRange
    if (pybind11::isinstance<pybind11::list>(ranges_obj) || pybind11::isinstance<pybind11::tuple>(ranges_obj)) {
      pybind11::list ranges_list = ranges_obj.cast<pybind11::list>();
      bool all_attn_ranges = true;
      for (auto item : ranges_list) {
        // Use try-catch instead of isinstance to avoid infinite recursion
        try {
          item.cast<AttnRange>();
        } catch (const pybind11::cast_error&) {
          all_attn_ranges = false;
          break;
        }
      }
      if (all_attn_ranges) {
        // Update internal ranges as well for consistency
        clear();
        for (auto item : ranges_list) {
          append(item.cast<AttnRange>());
        }
      }
    }
  }
};

struct AttnRangeWithRank : AttnRange {
  std::set<int> rank_set;

  AttnRangeWithRank(std::set<int> rank_set, int start, int end) : AttnRange(start, end), rank_set(std::move(rank_set)) {}

  AttnRangeWithRank() = default;
  AttnRangeWithRank(const AttnRangeWithRank&) = default;
  AttnRangeWithRank& operator=(const AttnRangeWithRank&) = default;

  bool operator==(const AttnRangeWithRank& other) const {
    return AttnRange::operator==(other) && rank_set == other.rank_set;
  }

  AttnRangeWithRank clone() const {
    return AttnRangeWithRank(rank_set, start, end);
  }

  std::string to_repr() const {
    std::string res = "[" + std::to_string(start) + ", " + std::to_string(end) + "), rank_set={";
    bool first = true;
    for (int r : rank_set) {
      if (!first)
        res += ", ";
      res += std::to_string(r);
      first = false;
    }
    res += "}";
    return res;
  }

  int get_seqlen() const {
    return end - start;
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(rank_set, start, end);
  }

  static AttnRangeWithRank setstate_py(pybind11::tuple t) {
    if (t.size() != 3)
      throw std::runtime_error("Invalid state for AttnRangeWithRank!");
    return AttnRangeWithRank(t[0].cast<std::set<int>>(), t[1].cast<int>(), t[2].cast<int>());
  }
};

struct GroupCastRanges : AttnRanges {
  // We use the base class 'ranges' but we need to store AttnRangeWithRank.
  // Since AttnRanges stores AttnRange by value, we might need a separate vector
  // if we want to store rank_set.
  std::vector<AttnRangeWithRank> group_cast_ranges;
  int cp_size = 0;
  bool split = true;

  GroupCastRanges() = default;

  GroupCastRanges(int cp_size, const std::vector<AttnRanges>& ranges_per_rank, bool split = true) : cp_size(cp_size), split(split) {
    if (ranges_per_rank.size() != static_cast<size_t>(cp_size)) {
      throw std::runtime_error("GroupCastRanges: ranges_per_rank size mismatch");
    }

    for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
      for (const auto& r : ranges_per_rank[cp_rank].ranges) {
        group_cast_ranges.emplace_back(std::set<int>{cp_rank}, r.start, r.end);
      }
    }

    // sort by attn_range.start
    std::sort(group_cast_ranges.begin(), group_cast_ranges.end(), [](const AttnRangeWithRank& a, const AttnRangeWithRank& b) { return a.start < b.start; });

    if (split) {
      _split();
    }

    // Sync to base class ranges for common methods like total_seqlen, etc.
    sync_to_base();
  }

  void sync_to_base() {
    local_range_cache.clear();
    ranges.clear();
    for (const auto& r : group_cast_ranges) {
      ranges.push_back(AttnRange(r.start, r.end));
    }
  }

  void _split() {
    if (group_cast_ranges.size() <= 1) {
      return;
    }

    std::vector<AttnRangeWithRank> new_ranges;

    // Get all points
    std::set<int> points_set;
    for (const auto& r : group_cast_ranges) {
      points_set.insert(r.start);
      points_set.insert(r.end);
    }
    std::vector<int> points(points_set.begin(), points_set.end());

    for (size_t i = 0; i < points.size() - 1; ++i) {
      int p1 = points[i];
      int p2 = points[i + 1];

      std::set<int> cover_rank_set;
      for (const auto& r : group_cast_ranges) {
        if (r.start <= p1 && r.end >= p2) {
          cover_rank_set.insert(r.rank_set.begin(), r.rank_set.end());
        }
      }

      if (!cover_rank_set.empty()) {
        new_ranges.emplace_back(std::move(cover_rank_set), p1, p2);
      }
    }

    group_cast_ranges = std::move(new_ranges);
    sync_to_base();
  }

  pybind11::object get_group_cast_ranges_py() const {
    pybind11::list result;
    for (const auto& r : group_cast_ranges) {
      result.append(r);
    }
    return result;
  }

  void set_group_cast_ranges_py(pybind11::object ranges_obj) {
    if (pybind11::isinstance<pybind11::list>(ranges_obj) || pybind11::isinstance<pybind11::tuple>(ranges_obj)) {
      pybind11::list ranges_list = ranges_obj.cast<pybind11::list>();
      group_cast_ranges.clear();
      for (auto item : ranges_list) {
        group_cast_ranges.push_back(item.cast<AttnRangeWithRank>());
      }
      sync_to_base();
    }
  }

  std::string to_repr() const {
    if (group_cast_ranges.empty()) {
      return "[[,)]";
    }
    std::string repr = "[";
    for (size_t i = 0; i < group_cast_ranges.size(); ++i) {
      repr += group_cast_ranges[i].to_repr();
      if (i < group_cast_ranges.size() - 1)
        repr += ", ";
    }
    repr += "]";
    return repr;
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(cp_size, group_cast_ranges, split);
  }

  static GroupCastRanges setstate_py(pybind11::tuple t) {
    if (t.size() != 3)
      throw std::runtime_error("Invalid state for GroupCastRanges!");
    GroupCastRanges res;
    res.cp_size = t[0].cast<int>();
    res.group_cast_ranges = t[1].cast<std::vector<AttnRangeWithRank>>();
    res.split = t[2].cast<bool>();
    res.sync_to_base();
    return res;
  }
};

enum AttnMaskType { FULL = 0, CAUSAL = 1, INV_CAUSAL = 2, BI_CAUSAL = 3 };

inline int attn_mask_type_to_int(AttnMaskType self) {
  return static_cast<int>(self);
}

inline AttnMaskType attn_mask_type_from_int(int i) {
  return static_cast<AttnMaskType>(i);
}

inline bool is_valid_cu_seqlens(const std::vector<int>& cu_seqlens, int seq_len) {
  if (cu_seqlens.empty())
    return true;
  if (cu_seqlens[0] != 0)
    return false;
  for (size_t i = 1; i < cu_seqlens.size(); ++i) {
    if (cu_seqlens[i - 1] >= cu_seqlens[i])
      return false;
  }
  if (cu_seqlens.back() != seq_len)
    return false;
  return true;
}

inline void check_valid_cu_seqlens(const std::vector<int>& cu_seqlens, int seq_len) {
  if (!is_valid_cu_seqlens(cu_seqlens, seq_len)) {
    throw pybind11::value_error("The cu_seqlens is invalid against the rule: 'cu_seqlens[0] == 0', and 'cu_seqlens[i-1] < cu_seqlens[i]'");
  }
}

} // namespace magi_attn_ext
