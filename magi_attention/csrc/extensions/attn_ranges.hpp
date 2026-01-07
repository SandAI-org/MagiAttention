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

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace magi_attn_ext {

struct AttnRange {
  int start;
  int end;

  AttnRange(int start, int end) : start(start), end(end) {
    check_valid();
  }

  AttnRange() = default;
  AttnRange(const AttnRange&) = default;
  AttnRange& operator=(const AttnRange&) = default;

  bool operator==(const AttnRange& range) const {
    return start == range.start && end == range.end;
  }

  bool operator!=(const AttnRange& other) const {
        return !(*this == other);
  }

  bool is_valid() const {
    return start <= end;
  }

  void check_valid() const {
    if (!is_valid()) {
      throw std::runtime_error("AttnRange is invalid with start=" + std::to_string(start) + " and end=" + std::to_string(end));
    }
  }

  bool is_empty() const {
    return seqlen() == 0;
  }

  int seqlen() const {
    return end - start;
  }

  bool is_valid_open() const {
    return start < end;
  }

  bool is_valid_close() const {
    return start <= end;
  }

  std::string to_string() const {
    return "[" + std::to_string(start) + ", " + std::to_string(end) + ")";
  }
};

struct AttnRanges {
  std::vector<AttnRange> ranges;

  AttnRanges() = default;

  explicit AttnRanges(std::vector<AttnRange>&& other_ranges) : ranges(std::move(other_ranges)) {}

  void append(int start, int end) {
    ranges.emplace_back(start, end);
  }

  void append(const AttnRange& range) {
    ranges.push_back(range);
  }

  void append(AttnRange&& range) {
    ranges.emplace_back(std::move(range));
  }

  void extend(const std::vector<AttnRange>& other_ranges) {
    ranges.insert(ranges.end(), other_ranges.begin(), other_ranges.end());
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
    ranges.clear();
  }

  void reserve(size_t capacity) {
    ranges.reserve(capacity);
  }

  int total_seqlen() {
    int total_seqlen = 0;
    for(auto& range : ranges) {
      total_seqlen += range.seqlen();
    }
    return total_seqlen;
  }

  AttnRanges sort_ranges() {
    std::vector<AttnRange> sorted_ranges = ranges;

    sort(sorted_ranges.begin(), sorted_ranges.end(),
      [](const AttnRange& a, const AttnRange& b) {
        return a.start < b.start;
      }
    );

    AttnRanges attn_ranges(std::move(sorted_ranges));

    return attn_ranges;
  }

  AttnRanges merge() {
    AttnRanges _ranges = sort_ranges();
    AttnRanges _merged_ranges;

    int start = -1, end = -1;
    for (size_t i = 0; i < this->size(); i++) {
      AttnRange& attn_range = (*this)[i];
      if (start == -1) {
        start = attn_range.start;
        end = attn_range.end;
        _merged_ranges.append(AttnRange(start, end));
      }
      else if (attn_range.start > end) {
        start = attn_range.start;
        end = attn_range.end;
        _merged_ranges.append(AttnRange(start, end));
      }
      else if (attn_range.end > end) {
        end = attn_range.end;
        _merged_ranges[_merged_ranges.size() - 1].end = end;
      }
    }

    return _merged_ranges;
  }

  std::string to_string() {
    std::string result = "[";
    for (size_t i = 0; i < ranges.size(); i++) {
      result += ranges[i].to_string();
      if (i != ranges.size() - 1) {
        result += ", ";
      }
    }
    return result;
  }
};

enum AttnMaskType {
  FULL = 0,
  CAUSAL = 1,
  INV_CAUSAL = 2,
  BI_CAUSAL = 3
};

} // namespace magi_attn_ext
