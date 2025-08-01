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

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.
// TODO: Add distributed offset to DistributedSeqlenInfo
struct DistributedSeqlenInfo {
  int const offset_q, offset_k;
  int const seqlen_q, seqlen_k;

  CUTLASS_DEVICE
  DistributedSeqlenInfo(int const bidb, int const* const q_ranges, int const* const k_ranges)
      : offset_q(q_ranges[2 * bidb]),
        offset_k(k_ranges[2 * bidb]),
        seqlen_q(q_ranges[2 * bidb + 1] - q_ranges[2 * bidb]),
        seqlen_k(k_ranges[2 * bidb + 1] - k_ranges[2 * bidb]) {}
};

} // namespace flash
