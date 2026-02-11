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

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "utils.h"

namespace flash {

// We consolidate all the info related to sequence length here.
// This is so that we can do all the gmem reads once at the beginning of each tile,
// rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.

// TODO: add PackGQA template to this class
struct SeqlenInfo {
  int offset_q, offset_k;
  int seqlen_q, seqlen_k;
  int2 const* q_ranges;
  int2 const* k_ranges;

  CUTLASS_DEVICE
  SeqlenInfo() : offset_q(0), offset_k(0), seqlen_q(0), seqlen_k(0), q_ranges(nullptr), k_ranges(nullptr) {}

  CUTLASS_DEVICE
  SeqlenInfo(int const bidb, int2 const* q_ranges, int2 const* k_ranges) : q_ranges(q_ranges), k_ranges(k_ranges) {
    int2 q_range = q_ranges[bidb];
    int2 k_range = k_ranges[bidb];
    offset_q = q_range.x;
    offset_k = k_range.x;
    seqlen_q = q_range.y - q_range.x;
    seqlen_k = k_range.y - k_range.x;
  }

  CUTLASS_DEVICE
  void update_k(int const bidb) {
    int2 k_range = load_and_broadcast<2>(&k_ranges[bidb]);
    offset_k = k_range.x;
    seqlen_k = k_range.y - k_range.x;
  }

  CUTLASS_DEVICE
  void update_q(int const bidb) {
    int2 q_range = load_and_broadcast<2>(&q_ranges[bidb]);
    offset_q = q_range.x;
    seqlen_q = q_range.y - q_range.x;
  }
};

} // namespace flash
