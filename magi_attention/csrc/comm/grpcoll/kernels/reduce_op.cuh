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

#include <map>
#include <stdexcept>
#include "exception.cuh"

namespace magi_attn_comm::grpcoll {

enum class ReduceOp : uint8_t {
  SUM = 0,
  AVG = 1,
  LSE = 2,
};

inline const std::map<std::string, ReduceOp>& get_str_to_reduce_op_map() {
  // NOTES: use `Magic Statics` to initialize the map
  // to avoid `Static Initialization Order Fiasco` and thread unsafety
  static const std::map<std::string, ReduceOp> map = {{"sum", ReduceOp::SUM}, {"avg", ReduceOp::AVG}, {"lse", ReduceOp::LSE}};
  return map;
}

inline ReduceOp str_to_reduce_op(const std::string& op_str) {
  const auto& op_map = get_str_to_reduce_op_map();
  auto it = op_map.find(op_str);
  if (it != op_map.end()) {
    return it->second;
  } else {
    throw std::runtime_error("Unsupported ReduceOp string value: " + op_str);
  }
}

} // namespace magi_attn_comm::grpcoll
