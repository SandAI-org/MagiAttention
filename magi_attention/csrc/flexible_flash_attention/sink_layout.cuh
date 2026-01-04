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

#include <map>
#include <stdexcept>

namespace flash {

enum class SinkLayout : uint8_t {
  SH = 0,
  SSH = 1,
  // SHD = 2, // TODO: support SHD
};

inline const std::map<std::string, SinkLayout>& get_str_to_sink_layout_map() {
  // NOTE: use `Magic Statics` to initialize the map
  // to avoid `Static Initialization Order Fiasco` and thread unsafety
  static const std::map<std::string, SinkLayout> map = {
      {"sh", SinkLayout::SH}, {"ssh", SinkLayout::SSH},
      // {"shd", SinkLayout::SHD} // TODO: support SHD
  };
  return map;
}

inline SinkLayout str_to_sink_layout(const std::string& op_str) {
  const auto& op_map = get_str_to_sink_layout_map();
  auto it = op_map.find(op_str);
  if (it != op_map.end()) {
    return it->second;
  } else {
    throw std::runtime_error("Unsupported SinkLayout string value: " + op_str);
  }
}

} // namespace flash
