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

#pragma once

#include <vector>

namespace magi_attn_ext {

struct AttnRange {
  int start;
  int end;

  AttnRange(int start, int end) : start(start), end(end) {
    check_valid();
  }

  bool is_valid() const {
    return start >= 0 && start <= end;
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
};

} // namespace magi_attn_ext
