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
#include "attn_ranges.hpp"
#include "rectangle.hpp"
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <cassert>

namespace magi_attn_ext {

class AttnRectangles {
public:
    AttnRectangles() = default;

    explicit AttnRectangles(std::vector<AttnRectangle>&& other_ranges) : _rects(std::move(other_ranges)) {}

    AttnRectangles(AttnRectangles& rect) {
        _rects = rect._rects;
    }

    AttnRectangles(AttnRectangles&&) = default;

    int size() { return _rects.size(); }

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

    void extend(const std::vector<AttnRectangle>& other_rects) {
        _rects.insert(_rects.end(), other_rects.begin(), other_rects.end());
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

    size_t size() const {
        return _rects.size();
    }

    static AttnRectangles from_ranges(
        AttnRanges& q_ranges,
        AttnRanges& k_ranges,
        std::vector<AttnMaskType>& mask_types
    ) {
        assert(
            q_ranges.size() == k_ranges.size() &&
            q_ranges.size() == mask_types.size() &&
            "q_ranges, k_ranges, mask_types length should be equal"
        );

        AttnRectangles attn_rects;

        for(size_t i = 0; i < q_ranges.size(); i++) {
            AttnRange& q_range = q_ranges[i];
            AttnRange& k_range = k_ranges[i];
            AttnMaskType mask_type = mask_types[i];

            if (q_range.is_empty() or k_range.is_empty())
                continue;
            if (mask_type == AttnMaskType::BI_CAUSAL && q_range.seqlen() > k_range.seqlen())
                continue;

            attn_rects.append(
                AttnRectangle(
                    q_range, k_range, mask_type
                )
            );
        }

        return attn_rects;
    }

    AttnRanges get_qo_ranges_union() const {
        AttnRanges qo_ranges;
        for(auto& rect : _rects) {
            qo_ranges.append(rect.get_q_range());
        }
        return qo_ranges.merge();
    }

    AttnRanges get_kv_ranges_union() const {
        AttnRanges kv_ranges;
        for(auto& rect : _rects) {
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
        for (AttnRectangle& rect : _rects) {
            auto [rect_left, rect_right] = rect.cut_q(cut_pos);
            if (rect_left.has_value()) {
                rects_left.append(rect_left.value());
            }
            if (rect_right.has_value()) {
                rects_right.append(rect_right.value());
            }
        }

        return std::pair<AttnRectangles, AttnRectangles>(
            std::move(rects_left),
            std::move(rects_right)
        );
    }

    std::pair<AttnRectangles, AttnRectangles> cut_k(int cut_pos) {
        AttnRectangles rects_left;
        AttnRectangles rects_right;
        for (AttnRectangle& rect : _rects) {
            auto [rect_left, rect_right] = rect.cut_k(cut_pos);
            if (rect_left.has_value()) {
                rects_left.append(rect_left.value());
            }
            if (rect_right.has_value()) {
                rects_right.append(rect_right.value());
            }
        }

        return std::pair<AttnRectangles, AttnRectangles>(
            std::move(rects_left),
            std::move(rects_right)
        );
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

    int area() {
        int total_area = 0;
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

    auto begin() { return _rects.begin(); }
    auto end() { return _rects.end(); }
    auto begin() const { return _rects.begin(); }
    auto end() const { return _rects.end(); }

private:
    std::vector<AttnRectangle> _rects;
};

}
