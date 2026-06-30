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

#include "cutlass/arch/barrier.h"

namespace flash {

using named_barrier = cutlass::arch::NamedBarrier;
using resv_barrier = cutlass::arch::ReservedNamedBarriers;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Named Barrier Enums and Traits
////////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: since cutlass::arch::ReservedNamedBarriers already reserves barriers 0 to 7,
// we can only use at most 8 named barriers for our own purposes.
static constexpr uint32_t MaxNumRawNamedBarriers = named_barrier::HardwareMaxNumNamedBarriers;
static constexpr uint32_t MaxNumUserNamedBarriers = named_barrier::HardwareMaxNumNamedBarriers - named_barrier::ReservedNamedBarrierCount;

// Traits for different named barrier enums
template <typename BarrierEnum>
struct BarrierTraits {
  static constexpr bool kUseRawBarrier = false;
  static constexpr uint32_t kNumBarriers = 0;
  static constexpr uint32_t kMaxNumWGs = 0;
};

template <>
struct BarrierTraits<resv_barrier> {
  static constexpr bool kUseRawBarrier = true;
  static constexpr uint32_t kNumBarriers = 8;
  static constexpr uint32_t kMaxNumWGs = 1024 / cutlass::NumThreadsPerWarpGroup; // at most 1024 threads per block
};

enum class FwdNamedBarriers {
  QueryEmpty = 0,
  WarpSchedulerWG1 = 1,
  WarpSchedulerWG2 = 2,
  WarpSchedulerWG3 = 3,
  WarpGroupSwapAB1 = 4,
  WarpGroupSwapAB2 = 5,
  WarpGroupSwapAB3 = 6,
};

template <>
struct BarrierTraits<FwdNamedBarriers> {
  static constexpr bool kUseRawBarrier = false;
  static constexpr uint32_t kNumBarriers = 7;
  static constexpr uint32_t kMaxNumWGs = 3;
};

// k for outer-loop and q for inner-loop
enum class BwdNamedBarriersLoopQ {
  KVEmpty = 0,
  PdS = 1,
  dQEmptyWG1 = 2,
  dQEmptyWG2 = 3,
  dQEmptyWG3 = 4,
  dQFullWG1 = 5,
  dQFullWG2 = 6,
  dQFullWG3 = 7,
};

template <>
struct BarrierTraits<BwdNamedBarriersLoopQ> {
  static constexpr bool kUseRawBarrier = false;
  static constexpr uint32_t kNumBarriers = 8;
  static constexpr uint32_t kMaxNumWGs = 3;
};

// q for outer-loop and k for inner-loop
enum class BwdNamedBarriersLoopK {
  QdOEmpty = 6,
  PdS = 7,
  dVEmptyWG1 = 8,
  dVEmptyWG2 = 9,
  dVFullWG1 = 10,
  dVFullWG2 = 11,
  dKEmptyWG1 = 12,
  dKEmptyWG2 = 13,
  dKFullWG1 = 14,
  dKFullWG2 = 15,
};

template <>
struct BarrierTraits<BwdNamedBarriersLoopK> {
  // NOTE: since SwapBwdQKLoop is true, we require 10 barriers for 2 consumer WGs
  // which exceeds the maximum number of user-named barriers (8).
  // Therefore, we have to use raw barrier IDs in this case.
  // And to avoid potential conflicts with reserved barriers,
  // we use raw barrier IDs 6~15, overlapping reserved barriers with
  // 6 (TmemAllocBarrier) and 7 (Sm120MainloopBarrier),
  // which seems safe for Hopper and earlier architectures.
  // However, for Blackwell and later architectures, we need to be more cautious about this overlap.
  static constexpr bool kUseRawBarrier = true;
  static constexpr uint32_t kNumBarriers = 10;
  static constexpr uint32_t kMaxNumWGs = 2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Barrier Manager
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BarrierManager {
  // Sync on given barrier with the offset for certain warp group
  template <int KNumThreads, typename BarrierEnum>
  CUTLASS_DEVICE static void sync(BarrierEnum barrier, int warp_group_idx = 0) {
    uint32_t barrier_id = _get_barrier_id(barrier, warp_group_idx);
    _sync<KNumThreads>(barrier_id);
  }

  // Arrive on given barrier with the offset for certain warp group
  template <int KNumThreads, typename BarrierEnum>
  CUTLASS_DEVICE static void arrive(BarrierEnum barrier, int warp_group_idx = 0) {
    uint32_t barrier_id = _get_barrier_id(barrier, warp_group_idx);
    _arrive<KNumThreads>(barrier_id);
  }

  // Static check for the validity of the barrier enum
  template <typename BarrierEnum, int kNumWarpGroups>
  CUTLASS_DEVICE static constexpr bool check() {
    using Traits = BarrierTraits<BarrierEnum>;

    static_assert(
        Traits::kNumBarriers <= (Traits::kUseRawBarrier ? MaxNumRawNamedBarriers : MaxNumUserNamedBarriers), "Exceeding the maximum number of barriers allowed.");
    static_assert(Traits::kMaxNumWGs >= kNumWarpGroups, "Exceeding the maximum number of warp groups allowed.");

    return true; // a dummy return value to force compile-time evaluation
  }

  // Calculate the barrier ID offset for certain warp group
  // with a starting offset determined by whether raw barrier IDs are used
  template <typename BarrierEnum>
  CUTLASS_DEVICE static uint32_t _get_barrier_id(BarrierEnum barrier, int warp_group_idx = 0) {
    using Traits = BarrierTraits<BarrierEnum>;
    static constexpr uint32_t barrier_id_start_offset = Traits::kUseRawBarrier ? 0 : named_barrier::ReservedNamedBarrierCount;
    uint32_t barrier_id = static_cast<uint32_t>(barrier) + warp_group_idx + barrier_id_start_offset;
    return barrier_id;
  }

  // Inner barrier sync implementation
  // which is almost the same as `named_barrier::sync`
  // but allows using raw barrier IDs if BarrierTraits::kUseRawBarrier is true
  template <int KNumThreads>
  CUTLASS_DEVICE static void _sync(uint32_t barrier_id) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(KNumThreads));
    cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, KNumThreads, barrier_id);
  }

  // Inner barrier arrive implementation
  // which is almost the same as `named_barrier::arrive`
  // but allows using raw barrier IDs if BarrierTraits::kUseRawBarrier is true
  template <int KNumThreads>
  CUTLASS_DEVICE static void _arrive(uint32_t barrier_id) {
    cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, KNumThreads, barrier_id);
    asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(KNumThreads));
  }
};
} // namespace flash
