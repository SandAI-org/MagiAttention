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
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include "attn_ranges.hpp"

namespace magi_attn_ext {

class AttnRectangle {
 public:
  AttnRectangle(const AttnRange& q, const AttnRange& k, const AttnRange& d) : q_range(q), k_range(k), d_range(d) {
    shrink_d_range();
    shrink_q_range();
    shrink_k_range();
    check_valid();
  }

  AttnRectangle(const AttnRange& q, const AttnRange& k, const AttnRange& d, AttnMaskType mask_type) : q_range(q), k_range(k), d_range(d) {
    if (mask_type == AttnMaskType::CAUSAL || mask_type == AttnMaskType::BI_CAUSAL) {
      d_range.end = std::min(d_range.end, k_range.end - q_range.end);
    } else {
      d_range.end = std::min(d_range.end, k_range.end - 1 - q_range.start);
    }

    if (mask_type == AttnMaskType::INV_CAUSAL || mask_type == AttnMaskType::BI_CAUSAL) {
      d_range.start = std::max(d_range.start, k_range.start - q_range.start);
    } else {
      d_range.start = std::max(d_range.start, k_range.start - (q_range.end - 1));
    }

    shrink_d_range();
    shrink_q_range();
    shrink_k_range();
    check_valid();
  }

  AttnRectangle(const AttnRange& q, const AttnRange& k, AttnMaskType mask_type)
      : AttnRectangle(q, k, AttnRange{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}, mask_type) {}

  AttnRectangle(const AttnRectangle& rec) : q_range(rec.q_range), k_range(rec.k_range), d_range(rec.d_range) {}

  AttnRectangle clone() const {
    return AttnRectangle(*this);
  }

  AttnRange& get_q_range() {
    return q_range;
  }

  const AttnRange& get_q_range() const {
    return q_range;
  }

  void set_q_range(const AttnRange& q) {
    check_valid(&q, nullptr, nullptr);
    q_range = q;
  }

  AttnRange& get_k_range() {
    return k_range;
  }

  const AttnRange& get_k_range() const {
    return k_range;
  }

  void set_k_range(const AttnRange& k) {
    check_valid(nullptr, &k, nullptr);
    k_range = k;
  }

  AttnRange& get_d_range() {
    return d_range;
  }

  const AttnRange& get_d_range() const {
    return d_range;
  }

  void set_d_range(const AttnRange& d) {
    check_valid(nullptr, nullptr, &d);
    d_range = d;
  }

  bool is_valid(const AttnRange* q = nullptr, const AttnRange* k = nullptr, const AttnRange* d = nullptr) const {
    const AttnRange& q_to_check = q ? *q : q_range;
    const AttnRange& k_to_check = k ? *k : k_range;
    const AttnRange& d_to_check = d ? *d : d_range;
    if (q_to_check.is_valid_open() && k_to_check.is_valid_open() && d_to_check.is_valid_close()) {
      return true;
    }
    return false;
  }

  void check_valid(const AttnRange* q = nullptr, const AttnRange* k = nullptr, const AttnRange* d = nullptr) const {
    if (!is_valid(q, k, d)) {
      const AttnRange& q_to_check = q ? *q : q_range;
      const AttnRange& k_to_check = k ? *k : k_range;
      const AttnRange& d_to_check = d ? *d : d_range;
      throw std::invalid_argument(
          std::string("Some of the ") + "q_range=" + q_to_check.to_string() + " " + "k_range=" + k_to_check.to_string() + " " + "d_range=" + d_to_check.to_string() +
          " is invalid, no area include.");
    }
  }

  AttnRectangle* get_valid_or_null() {
    return (is_valid() ? this : nullptr);
  }

  const AttnRectangle* get_valid_or_none() const {
    return is_valid() ? this : nullptr;
  }

  bool shrink_q_range() {
    // calc intersection of d_range end diagonal line & k_range start line
    int intersection_q_start = k_range.start - d_range.end;
    // calc instersection of d_range start diagonal line & k_range end line
    int intersection_q_end = k_range.end - d_range.start;
    q_range.start = std::max(q_range.start, intersection_q_start);
    q_range.end = std::min(q_range.end, intersection_q_end);
    return q_range.is_valid_open();
  }

  bool shrink_k_range() {
    // calc intersection of d_range start diagonal line & q_range start line
    int intersection_k_start = d_range.start + q_range.start;
    // calc intersection of d_range end diagonal line & q_range end line
    int intersection_k_end = d_range.end + q_range.end;
    k_range.start = std::max(k_range.start, intersection_k_start);
    k_range.end = std::min(k_range.end, intersection_k_end);
    return k_range.is_valid_open();
  }

  bool shrink_d_range() {
    int d_range_min = k_range.start - (q_range.end - 1);
    int d_range_max = (k_range.end - 1) - q_range.start;
    d_range.start = std::max(d_range.start, d_range_min);
    d_range.end = std::min(d_range.end, d_range_max);
    return d_range.is_valid_close();
  }

  std::pair<std::optional<AttnRectangle>, std::optional<AttnRectangle>> cut_q(int cut_pos) const {
    if (cut_pos <= q_range.start) {
      return {std::nullopt, *this};
    }
    if (cut_pos >= q_range.end) {
      return {*this, std::nullopt};
    }

    AttnRectangle left(*this);
    AttnRectangle right(*this);

    left.q_range.end = cut_pos;
    right.q_range.start = cut_pos;

    left.shrink_d_range();
    left.shrink_k_range();
    right.shrink_d_range();
    right.shrink_k_range();

    return {left, right};
  }

  std::pair<std::optional<AttnRectangle>, std::optional<AttnRectangle>> cut_k(int cut_pos) const {
    if (cut_pos <= k_range.start) {
      return {std::nullopt, *this};
    }
    if (cut_pos >= k_range.end) {
      return {*this, std::nullopt};
    }

    AttnRectangle left(*this);
    AttnRectangle right(*this);

    left.k_range.end = cut_pos;
    right.k_range.start = cut_pos;

    left.shrink_d_range();
    left.shrink_q_range();
    right.shrink_d_range();
    right.shrink_q_range();

    return {left, right};
  }

  std::optional<AttnRectangle> get_rect_within_q_segment(int q_start, int q_end) const {
    if (q_end <= q_range.start || q_start >= q_range.end) {
      return std::nullopt;
    }

    AttnRectangle rect_in_seg(*this);
    rect_in_seg.q_range.start = std::max(rect_in_seg.q_range.start, q_start);
    rect_in_seg.q_range.end = std::min(rect_in_seg.q_range.end, q_end);
    rect_in_seg.shrink_d_range();
    rect_in_seg.shrink_k_range();

    return rect_in_seg;
  }

  std::optional<AttnRectangle> get_rect_within_k_segment(int k_start, int k_end) const {
    if (k_end <= k_range.start || k_start >= k_range.end) {
      return std::nullopt;
    }

    AttnRectangle rect_in_seg(*this);
    rect_in_seg.k_range.start = std::max(rect_in_seg.k_range.start, k_start);
    rect_in_seg.k_range.end = std::min(rect_in_seg.k_range.end, k_end);
    rect_in_seg.shrink_d_range();
    rect_in_seg.shrink_q_range();

    return rect_in_seg;
  }

  int intersection_q_id_on_left_boundary() const {
    return k_range.start - d_range.start;
  }

  int intersection_q_id_on_right_boundary() const {
    return k_range.end - 1 - d_range.end;
  }

  bool is_full() const {
    return d_range.start <= k_range.start - (q_range.end - 1) && d_range.end >= k_range.end - 1 - q_range.start;
  }

  bool is_causal() const {
    return d_range.start <= k_range.start - (q_range.end - 1) && d_range.end == k_range.end - q_range.end;
  }

  bool is_inv_causal() const {
    return d_range.start == k_range.start - q_range.start && d_range.end >= k_range.end - 1 - q_range.start;
  }

  bool is_bi_causal() const {
    return d_range.start == k_range.start - q_range.start && d_range.end == k_range.end - q_range.end;
  }

  void append_qk_range_mask_type(std::vector<std::tuple<AttnRange, AttnRange, int>>& attn_arg) const {
    if (is_full()) {
      attn_arg.push_back({q_range, k_range, 0});
      return;
    }
    if (is_causal()) {
      attn_arg.push_back({q_range, k_range, 1});
      return;
    }
    if (is_inv_causal()) {
      attn_arg.push_back({q_range, k_range, 2});
      return;
    }
    if (is_bi_causal()) {
      attn_arg.push_back({q_range, k_range, 3});
      return;
    }

    int q_id_l = intersection_q_id_on_left_boundary();
    int q_id_r = intersection_q_id_on_right_boundary();

    if (q_id_l < q_range.start || q_id_l >= q_range.end || q_id_r < q_range.start || q_id_r >= q_range.end) {
      throw std::invalid_argument("rect without shrinkage call to_qk_range_mask_type");
    }

    if (q_id_l == q_range.end - 1) {
      const auto& [up_rect, down_rect] = cut_q(q_id_r + 1);
      if (up_rect.has_value()) {
        up_rect.value().append_qk_range_mask_type(attn_arg);
      }
      if (down_rect.has_value()) {
        down_rect.value().append_qk_range_mask_type(attn_arg);
      }
      return;
    }

    if (q_id_r == q_range.start) {
      const auto& [up_rect, down_rect] = cut_q(q_id_l);
      if (up_rect.has_value()) {
        up_rect.value().append_qk_range_mask_type(attn_arg);
      }
      if (down_rect.has_value()) {
        down_rect.value().append_qk_range_mask_type(attn_arg);
      }
      return;
    }

    if (q_id_r <= q_id_l) {
      const auto& [up_rect, down_rect] = cut_q(q_id_l);
      if (up_rect.has_value()) {
        up_rect.value().append_qk_range_mask_type(attn_arg);
      }
      if (down_rect.has_value()) {
        down_rect.value().append_qk_range_mask_type(attn_arg);
      }
      return;
    } else if (q_id_r == q_id_l + 1) {
      const auto& [up_rect, down_rect] = cut_q(q_id_r);
      if (up_rect.has_value()) {
        up_rect.value().append_qk_range_mask_type(attn_arg);
      }
      if (down_rect.has_value()) {
        down_rect.value().append_qk_range_mask_type(attn_arg);
      }
      return;
    } else {
      const auto& [up_rect, remaining_rect] = cut_q(q_id_l);
      if (up_rect.has_value()) {
        up_rect.value().append_qk_range_mask_type(attn_arg);
      }
      if (remaining_rect.has_value()) {
        const auto& [mid_rect, down_rect] = remaining_rect.value().cut_q(q_id_r + 1);
        if (mid_rect.has_value()) {
          mid_rect.value().append_qk_range_mask_type(attn_arg);
        }
        if (down_rect.has_value()) {
          down_rect.value().append_qk_range_mask_type(attn_arg);
        }
      }
      return;
    }
  }

  std::vector<std::tuple<AttnRange, AttnRange, int>> to_qk_range_mask_type() const {
    std::vector<std::tuple<AttnRange, AttnRange, int>> attn_arg;
    append_qk_range_mask_type(attn_arg);
    return attn_arg;
  }

  long long area() {
    return count_areas(q_range.start, q_range.end, k_range.start, k_range.end, d_range.start, d_range.end);
  }

  long long count_areas(int lq, int rq, int lk, int rk, int ld, int rd) {
    if (rq <= lq || rk <= lk || rd < ld)
      return 0;

    long long Q1 = lq, Q2 = rq - 1;
    long long K1 = lk, K2 = rk - 1;

    long long a = K1 - ld;
    long long b = K2 - rd;

    long long total = 0;

    if (a <= b) {
      // Interval 1
      long long q_start = Q1;
      long long q_end = std::min(a - 1, Q2);
      if (q_start <= q_end) {
        long long m = std::max(q_start, K1 - (long long)rd);
        long long n = q_end;
        if (m <= n) {
          long long c = (long long)rd - K1 + 1;
          long long sum_q = (long long)n * (n + 1) / 2 - (long long)(m - 1) * m / 2;
          long long sum_c = c * (n - m + 1);
          total += sum_q + sum_c;
        }
      }

      // Interval 2
      q_start = std::max(a, Q1);
      q_end = std::min(b, Q2);
      if (q_start <= q_end) {
        long long k_count = (long long)rd - ld + 1;
        total += k_count * (q_end - q_start + 1);
      }

      // Interval 3
      q_start = std::max(b + 1, Q1);
      q_end = Q2;
      if (q_start <= q_end) {
        long long m = q_start;
        long long n = std::min(q_end, K2 - (long long)ld);
        if (m <= n) {
          long long c = (long long)K2 - ld + 1;
          long long sum_c = c * (n - m + 1);
          long long sum_q = (long long)n * (n + 1) / 2 - (long long)(m - 1) * m / 2;
          total += sum_c - sum_q;
        }
      }
    } else {
      // When a > b

      // Interval 1
      long long q_start = Q1;
      long long q_end = std::min(b, Q2);
      if (q_start <= q_end) {
        long long m = std::max(q_start, K1 - (long long)rd);
        long long n = q_end;
        if (m <= n) {
          long long c = (long long)rd - K1 + 1;
          long long sum_q = (long long)n * (n + 1) / 2 - (long long)(m - 1) * m / 2;
          long long sum_c = c * (n - m + 1);
          total += sum_q + sum_c;
        }
      }

      // Interval 2
      q_start = std::max(b + 1, Q1);
      q_end = std::min(a - 1, Q2);
      if (q_start <= q_end) {
        if (K1 <= K2) {
          long long k_count = (long long)K2 - K1 + 1;
          total += k_count * (q_end - q_start + 1);
        }
      }

      // Interval 3
      q_start = std::max(a, Q1);
      q_end = Q2;
      if (q_start <= q_end) {
        long long m = q_start;
        long long n = std::min(q_end, K2 - (long long)ld);
        if (m <= n) {
          long long c = (long long)K2 - ld + 1;
          long long sum_c = c * (n - m + 1);
          long long sum_q = (long long)n * (n + 1) / 2 - (long long)(m - 1) * m / 2;
          total += sum_c - sum_q;
        }
      }
    }

    return total;
  }

  bool operator==(const AttnRectangle& other) const {
    return q_range == other.q_range && k_range == other.k_range && d_range == other.d_range;
  }

  bool operator!=(const AttnRectangle& other) const {
    return !(*this == other);
  }

  size_t hash_py() const {
    return q_range.hash_py() ^ (k_range.hash_py() << 1) ^ (d_range.hash_py() << 2);
  }

  pybind11::tuple getstate_py() const {
    return pybind11::make_tuple(q_range, k_range, d_range);
  }

  static AttnRectangle setstate_py(pybind11::tuple t) {
    if (t.size() != 3)
      throw std::runtime_error("Invalid state for AttnRectangle!");

    // Use try-catch or careful casting to avoid recursion during unpickling
    try {
      return AttnRectangle(t[0].cast<AttnRange>(), t[1].cast<AttnRange>(), t[2].cast<AttnRange>());
    } catch (const pybind11::cast_error& e) {
      throw std::runtime_error(std::string("AttnRectangle.setstate_py failed to cast elements to AttnRange: ") + e.what());
    }
  }

  std::string to_repr() const {
    return q_range.to_repr() + " x " + k_range.to_repr() + " x " + d_range.to_repr();
  }

 private:
  AttnRange q_range;
  AttnRange k_range;
  AttnRange d_range;
};

} // namespace magi_attn_ext
