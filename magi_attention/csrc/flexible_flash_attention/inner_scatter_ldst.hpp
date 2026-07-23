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

namespace flash {

// Scatter load/store thread topology (dtype-independent).
//
// Hardware anchors:
//   kBankRowBytes = 128  (SMEM bank row = one memory transaction)
//   kLaneBytes    = 16   (per-thread access: cp.async width for load,
//                          coalesced portion for store)
//   kThreadsPerGroup = 128/16 = 8 (threads jointly covering one bank row)
//
// Token-row partitioning:
//   kNumGroups = kNumThreads / 8   (e.g. 128→16, 32→4)
//   kTokensPerGroup = kTileSize / kNumGroups
//
// Head-dim tiling is dtype-dependent — compute at call site:
//   tiles_per_row  = kHeadDim * sizeof(T) / kBankRowBytes
//   elems_per_lane = kLaneBytes / sizeof(T)
//   elems_per_row  = kBankRowBytes / sizeof(T)
template <int kNumThreads_, int kTileSize_>
struct ScatterLdstGroup {
  static constexpr int kBankRowBytes = 128;
  static constexpr int kLaneBytes = 16;
  static constexpr int kThreadsPerGroup = kBankRowBytes / kLaneBytes;
  static constexpr int kNumGroups = kNumThreads_ / kThreadsPerGroup;
  static constexpr int kTokensPerGroup = kTileSize_ / kNumGroups;

  static_assert(kNumThreads_ % kThreadsPerGroup == 0, "kNumThreads must be a multiple of 8");
  static_assert(kTileSize_ % kNumGroups == 0, "kTileSize must be divisible by kNumGroups");
};

} // namespace flash
