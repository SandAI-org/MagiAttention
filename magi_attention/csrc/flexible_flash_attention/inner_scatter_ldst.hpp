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

// Thread decomposition for scatter load (cp.async) and scatter store (atomicAdd).
//
// Anchor: each thread processes kBytesPerLane = 16B per tile_idx iteration.
//   Load:  16B = one cp.async instruction (hardware width).
//   Store: 16B = kInnerElemsPerThread × sizeof(Inner) (work-division of 128B row among 8 lanes).
//
// Token-row partitioning:
//   kThreadsPerGroup = 128B / 16B = 8 threads jointly cover one SMEM bank row.
//   kNumGroups = kNumThreads / 8 → each group handles kTokensPerGroup rows.
//   Store inherits the same 8-thread grouping so that the same threads handle the same tokens.
//
// Head-dim tiling per row:
//   Outer (load):  kOuterTilesPerRow = kHeadDim / (128B / sizeof(Outer))
//   Inner (store): kInnerTilesPerRow = kHeadDim / (128B / sizeof(Inner))
//   Ratio = sizeof(Inner) / sizeof(Outer), e.g. float/bf16 = 2.
//
// Outer = GMEM activation (Q/K/V, bf16/fp16). Inner = accumulator (dQ/dK/dV, float).
template <int kNumThreads_, int kTileSize_, int kHeadDim_, int kOuterElemSize_, int kInnerElemSize_>
struct InnerScatterLdstGroup {
  static constexpr int kSmemBankRowBytes = 128;
  static constexpr int kBytesPerLane = 16;

  static constexpr int kNumThreads = kNumThreads_;
  static constexpr int kTileSize = kTileSize_;
  static constexpr int kHeadDim = kHeadDim_;

  // ── Token-row decomposition (shared by load and store) ──
  static constexpr int kThreadsPerGroup = kSmemBankRowBytes / kBytesPerLane;
  static constexpr int kNumGroups = kNumThreads / kThreadsPerGroup;
  static constexpr int kTokensPerGroup = kTileSize / kNumGroups;

  // ── Outer (cp.async load) ──
  static constexpr int kOuterElemsPerThread = kBytesPerLane / kOuterElemSize_;
  static constexpr int kOuterElemsPerBankRow = kSmemBankRowBytes / kOuterElemSize_;
  static constexpr int kOuterTilesPerRow = kHeadDim / kOuterElemsPerBankRow;

  // ── Inner (atomicAdd store) ──
  static constexpr int kInnerElemsPerThread = kBytesPerLane / kInnerElemSize_;
  static constexpr int kInnerElemsPerBankRow = kSmemBankRowBytes / kInnerElemSize_;
  static constexpr int kInnerTilesPerRow = kHeadDim / kInnerElemsPerBankRow;

  static_assert(kBytesPerLane % kOuterElemSize_ == 0);
  static_assert(kBytesPerLane % kInnerElemSize_ == 0);
  static_assert(kHeadDim % kOuterElemsPerBankRow == 0, "HeadDim must be a multiple of outer bank row elements");
  static_assert(kHeadDim % kInnerElemsPerBankRow == 0, "HeadDim must be a multiple of inner bank row elements");
  static_assert(kNumThreads % kThreadsPerGroup == 0);
  static_assert(kTileSize % kNumGroups == 0);
};

} // namespace flash
