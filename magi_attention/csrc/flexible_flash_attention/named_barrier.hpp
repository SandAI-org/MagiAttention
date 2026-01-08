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

// cutlass::arch::NamedBarrier::sync/arrive are only enabled Sm90 even though they work
// for Sm80 as well. We reimplement them here, enabled for both Sm90 and Sm80.

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, uint32_t barrier_id_) {
  static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(resv_barrier::FirstUserBarrier);
  uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
  cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_sync(uint32_t num_threads, resv_barrier reserved_named_barriers) {
  uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
  asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
  cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, num_threads, barrier_id);
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, uint32_t barrier_id_) {
  static constexpr uint32_t ReservedNamedBarrierCount = static_cast<uint32_t>(resv_barrier::FirstUserBarrier);
  uint32_t barrier_id = barrier_id_ + ReservedNamedBarrierCount;
  cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

CUTLASS_DEVICE
static void named_barrier_arrive(uint32_t num_threads, resv_barrier reserved_named_barriers) {
  uint32_t barrier_id = static_cast<uint32_t>(reserved_named_barriers);
  cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads, barrier_id);
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Enumerates the reserved named barriers to avoid potential conflicts

// NOTE: since cutlass::arch::ReservedNamedBarriers already reserves barriers 0 to 7,
// we can only use at most 8 named barriers for our own purposes.
static const uint32_t MaxNumUserNamedBarriers = named_barrier::HardwareMaxNumNamedBarriers - named_barrier::ReservedNamedBarrierCount;

enum class FwdNamedBarriers {
  QueryEmpty = 0,
  WarpSchedulerWG1 = 1,
  WarpSchedulerWG2 = 2,
  WarpSchedulerWG3 = 3,
  WarpGroupSwapAB1 = 4,
  WarpGroupSwapAB2 = 5,
  WarpGroupSwapAB3 = 6,

  kNumBarriers = 7,
  kMaxNumWGs = 3,
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

  kNumBarriers = 8,
  kMaxNumWGs = 3,
};

// q for outer-loop and k for inner-loop
enum class BwdNamedBarriersLoopK {
  QdOEmpty = 0,
  PdS = 1,
  dVEmptyWG1 = 2,
  dVEmptyWG2 = 3,
  dVFullWG1 = 4,
  dVFullWG2 = 5,
  dKEmptyWG1 = 6,
  dKEmptyWG2 = 7,
  dKFullWG1 = 8,
  dKFullWG2 = 9,

  kNumBarriers = 10,
  kMaxNumWGs = 2,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Barrier Manager

struct BarrierManager {
  template <int KNumThreads, typename BarrierEnum>
  CUTLASS_DEVICE static void sync(BarrierEnum barrier, int warp_group_idx = 0) {
    uint32_t barrier_id = get_barrier_id(barrier, warp_group_idx);
    named_barrier::sync(KNumThreads, /*id=*/barrier_id);
  }

  template <int KNumThreads, typename BarrierEnum>
  CUTLASS_DEVICE static void arrive(BarrierEnum barrier, int warp_group_idx = 0) {
    uint32_t barrier_id = get_barrier_id(barrier, warp_group_idx);
    named_barrier::arrive(KNumThreads, /*id=*/barrier_id);
  }

  // Calculate the barrier ID offset for certain warp group
  template <typename BarrierEnum>
  CUTLASS_DEVICE static uint32_t get_barrier_id(BarrierEnum barrier, int warp_group_idx = 0) {
    uint32_t barrier_id = static_cast<uint32_t>(barrier) + warp_group_idx;
    return barrier_id;
  }

  template <typename BarrierEnum, int kNumWarpGroups>
  CUTLASS_DEVICE static constexpr bool check() {
    static_assert(static_cast<uint32_t>(BarrierEnum::kNumBarriers) <= MaxNumUserNamedBarriers, "Exceeding the maximum number of user defined named barriers allowed.");
    static_assert(static_cast<int>(BarrierEnum::kMaxNumWGs) >= kNumWarpGroups, "Exceeding the maximum number of warp groups allowed for the named barriers.");

    return true; // a dummy return value to force compile-time evaluation
  }
};

} // namespace flash
