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

#include <algorithm>
#include <cassert>
#include <optional>
#include <stdexcept>
#include <vector>
#include "attn_ranges.hpp"
#include "rectangle.hpp"

namespace magi_attn_ext {

class AttnRectangles {
 public:
  AttnRectangles() = default;

  explicit AttnRectangles(std::vector<AttnRectangle>&& other_ranges) : _rects(std::move(other_ranges)) {}

  AttnRectangles(const AttnRectangles& rect) {
    _rects = rect._rects;
  }

  AttnRectangles(AttnRectangles&&) = default;

  AttnRectangles& operator=(const AttnRectangles& other) {
    if (this != &other) {
      _rects = other._rects;
    }
    return *this;
  }

  AttnRectangles& operator=(AttnRectangles&& other) noexcept {
    if (this != &other) {
      _rects = std::move(other._rects);
    }
    return *this;
  }

  size_t size() const {
    return _rects.size();
  }

  bool is_empty() const {
    return _rects.empty();
  }

  bool is_valid() const {
    if (is_empty()) {
      return true;
    }

    for (const auto& rect : _rects) {
      if (!rect.is_valid()) {
        return false;
      }
    }
    return true;
  }

  void check_valid() const {
    if (!is_valid()) {
      throw std::invalid_argument("Some of the rects are invalid");
    }
  }

  void clear() {
    _rects.clear();
  }

  void append(const AttnRectangle& rect) {
    _rects.push_back(rect);
  }

  void append(AttnRectangle&& rect) {
    _rects.emplace_back(std::move(rect));
  }

  void append_py(const AttnRectangle& rect, bool check = false) {
    if (check) {
      rect.check_valid();
    }
    append(rect);
  }

  void extend(const std::vector<AttnRectangle>& other_rects) {
    _rects.insert(_rects.end(), other_rects.begin(), other_rects.end());
  }

  void extend(const AttnRectangles& other) {
    extend(other.get());
  }

  void extend_py(const AttnRectangles& other, bool check = false) {
    if (check) {
      other.check_valid();
    }
    extend(other);
  }

  const std::vector<AttnRectangle>& get() const {
    return _rects;
  }

  std::vector<AttnRectangle>& get() {
    return _rects;
  }

  AttnRectangle& at(size_t idx) {
    if (idx >= _rects.size()) {
      throw std::out_of_range("AttnRanges idx out of range");
    }
    return _rects[idx];
  }

  const AttnRectangle& at(size_t idx) const {
    if (idx >= _rects.size()) {
      throw std::out_of_range("AttnRanges idx out of range");
    }
    return _rects[idx];
  }

  AttnRectangle& operator[](size_t idx) {
    return _rects[idx];
  }

  const AttnRectangle& operator[](size_t idx) const {
    return _rects[idx];
  }

  static AttnRectangles from_ranges(AttnRanges& q_ranges, AttnRanges& k_ranges, std::vector<AttnMaskType>& mask_types) {
    if (!(q_ranges.size() == k_ranges.size() && q_ranges.size() == mask_types.size())) {
      throw std::invalid_argument("q_ranges, k_ranges, mask_types length should be equal");
    }

    AttnRectangles attn_rects;

    for (size_t i = 0; i < q_ranges.size(); i++) {
      AttnRange& q_range = q_ranges[i];
      AttnRange& k_range = k_ranges[i];
      AttnMaskType mask_type = mask_types[i];

      if (q_range.is_empty() or k_range.is_empty())
        continue;
      if (mask_type == AttnMaskType::BI_CAUSAL && q_range.seqlen() > k_range.seqlen())
        continue;

      attn_rects.append(AttnRectangle(q_range, k_range, std::nullopt, mask_type));
    }

    return attn_rects;
  }

  static AttnRectangles from_ranges_py(const pybind11::object& q_ranges_obj, const pybind11::object& k_ranges_obj, const pybind11::list& mask_types_py, bool check) {
    AttnRanges q_ranges = AttnRanges::from_ranges(q_ranges_obj, check);
    AttnRanges k_ranges = AttnRanges::from_ranges(k_ranges_obj, check);

    std::vector<AttnMaskType> mask_types;
    mask_types.reserve(mask_types_py.size());
    for (auto item : mask_types_py) {
      if (pybind11::isinstance<pybind11::int_>(item)) {
        mask_types.push_back(static_cast<AttnMaskType>(item.cast<int>()));
      } else {
        mask_types.push_back(item.cast<AttnMaskType>());
      }
    }

    return from_ranges(q_ranges, k_ranges, mask_types);
  }

  pybind11::object get_item(pybind11::object idx) {
    // Use try-catch or explicit type check for slice
    if (PySlice_Check(idx.ptr())) {
      pybind11::slice s = idx.cast<pybind11::slice>();
      size_t start, stop, step, slicelength;
      if (!s.compute(size(), &start, &stop, &step, &slicelength)) {
        throw pybind11::error_already_set();
      }
      AttnRectangles result;
      result.get().reserve(slicelength);
      for (size_t i = 0; i < slicelength; ++i) {
        result.append(_rects[start]);
        start += step;
      }
      return pybind11::cast(result);
    } else {
      try {
        long long index_ll = idx.cast<long long>();
        if (index_ll < 0) {
          index_ll += (long long)size();
        }
        if (index_ll < 0 || index_ll >= (long long)size()) {
          throw pybind11::index_error();
        }
        return pybind11::cast(&_rects[(size_t)index_ll]);
      } catch (const pybind11::cast_error&) {
        throw pybind11::type_error("Index must be an integer or slice");
      }
    }
  }

  void set_item(pybind11::object idx, pybind11::object value) {
    if (PySlice_Check(idx.ptr())) {
      pybind11::slice s = idx.cast<pybind11::slice>();
      size_t start, stop, step, slicelength;
      if (!s.compute(size(), &start, &stop, &step, &slicelength)) {
        throw pybind11::error_already_set();
      }

      // Use try-catch instead of isinstance to avoid infinite recursion
      AttnRectangles* vals_ptr = nullptr;
      try {
        vals_ptr = &value.cast<AttnRectangles&>();
      } catch (const pybind11::cast_error&) {
        throw pybind11::type_error("Assignment value must be AttnRectangles");
      }
      auto& vals = *vals_ptr;

      if (vals.size() != slicelength) {
        throw std::invalid_argument("slice length and value length mismatch");
      }
      for (size_t i = 0; i < slicelength; ++i) {
        _rects[start] = vals[i];
        start += step;
      }
    } else {
      try {
        long long index_ll = idx.cast<long long>();
        if (index_ll < 0) {
          index_ll += (long long)size();
        }
        if (index_ll < 0 || index_ll >= (long long)size()) {
          throw pybind11::index_error();
        }
        _rects[(size_t)index_ll] = value.cast<AttnRectangle>();
      } catch (const pybind11::cast_error&) {
        throw pybind11::type_error("Index must be an integer or slice");
      }
    }
  }

  std::string to_repr() const {
    if (is_empty()) {
      return "[-1, -1) x [-1, -1): None";
    }
    std::string res = "[";
    for (size_t i = 0; i < size(); ++i) {
      res += _rects[i].to_repr();
      if (i < size() - 1)
        res += ", ";
    }
    res += "]";
    return res;
  }

  size_t hash_py() const {
    size_t h = 0;
    for (const auto& rect : _rects) {
      h ^= rect.hash_py() + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(_rects);
  }

  static AttnRectangles setstate_py(pybind11::tuple t) {
    if (t.size() != 1)
      throw std::runtime_error("Invalid state for AttnRectangles!");
    return AttnRectangles(t[0].cast<std::vector<AttnRectangle>>());
  }

  AttnRanges get_qo_ranges_union() const {
    AttnRanges qo_ranges;
    for (auto& rect : _rects) {
      qo_ranges.append(rect.get_q_range());
    }
    return qo_ranges.merge();
  }

  AttnRanges get_kv_ranges_union() const {
    AttnRanges kv_ranges;
    for (auto& rect : _rects) {
      kv_ranges.append(rect.get_k_range());
    }
    return kv_ranges.merge();
  }

  int total_seqlen_qo() {
    return get_qo_ranges_union().total_seqlen();
  }

  int total_seqlen_kv() {
    return get_kv_ranges_union().total_seqlen();
  }

  std::pair<AttnRectangles, AttnRectangles> cut_q(int cut_pos) {
    AttnRectangles rects_left;
    AttnRectangles rects_right;
    rects_left.get().reserve(_rects.size());
    rects_right.get().reserve(_rects.size());
    for (AttnRectangle& rect : _rects) {
      auto [rect_left, rect_right] = rect.cut_q(cut_pos);
      if (rect_left.has_value()) {
        rects_left.append(std::move(rect_left.value()));
      }
      if (rect_right.has_value()) {
        rects_right.append(std::move(rect_right.value()));
      }
    }

    return std::pair<AttnRectangles, AttnRectangles>(std::move(rects_left), std::move(rects_right));
  }

  std::pair<AttnRectangles, AttnRectangles> cut_k(int cut_pos) {
    AttnRectangles rects_left;
    AttnRectangles rects_right;
    rects_left.get().reserve(_rects.size());
    rects_right.get().reserve(_rects.size());
    for (AttnRectangle& rect : _rects) {
      auto [rect_left, rect_right] = rect.cut_k(cut_pos);
      if (rect_left.has_value()) {
        rects_left.append(std::move(rect_left.value()));
      }
      if (rect_right.has_value()) {
        rects_right.append(std::move(rect_right.value()));
      }
    }

    return std::pair<AttnRectangles, AttnRectangles>(std::move(rects_left), std::move(rects_right));
  }

  AttnRectangles get_rects_within_q_segment(int q_start, int q_end) {
    AttnRectangles rects_in_seg;

    for (auto& rect : _rects) {
      auto rect_in_seg = rect.get_rect_within_q_segment(q_start, q_end);
      if (rect_in_seg.has_value()) {
        rects_in_seg.append(rect_in_seg.value());
      }
    }

    return rects_in_seg;
  }

  AttnRectangles get_rects_within_k_segment(int k_start, int k_end) {
    AttnRectangles rects_in_seg;

    for (auto& rect : _rects) {
      auto rect_in_seg = rect.get_rect_within_k_segment(k_start, k_end);
      if (rect_in_seg.has_value()) {
        rects_in_seg.append(rect_in_seg.value());
      }
    }

    return rects_in_seg;
  }

  long long area() {
    long long total_area = 0;
    for (auto& rect : _rects) {
      total_area += rect.area();
    }
    return total_area;
  }

  bool operator==(const AttnRectangles& other_rects) const {
    if (this->size() != other_rects.size()) {
      return false;
    }

    for (size_t i = 0; i < this->size(); i++) {
      if (other_rects[i] != (*this)[i]) {
        return false;
      }
    }

    return true;
  }

  bool operator!=(const AttnRectangles& other_rects) const {
    return !(*this == other_rects);
  }

  auto begin() {
    return _rects.begin();
  }
  auto end() {
    return _rects.end();
  }
  auto begin() const {
    return _rects.begin();
  }
  auto end() const {
    return _rects.end();
  }

 private:
  std::vector<AttnRectangle> _rects;
};

} // namespace magi_attn_ext
