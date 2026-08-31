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

#include <cutlass/cutlass.h>

namespace flash {

// Deterministic range-lock primitives shared by FWD epilogue, BWD mainloop, and BWD epilogue.
//
// Range locks use a two-slot layout per block: [arrive_num, counter].
// - deterministic_sync: spin-waits until range_lock[slot*2] reaches the expected value.
// - deterministic_arrive: atomically increments the counter; when it reaches 2 the
//   arrive_num is published and the counter is reset, unblocking the next sync.

CUTLASS_DEVICE
void deterministic_sync(int* range_lock, int bidh, int offset, int range_block_size, int num_heads, int left_range_sync_num, int right_range_sync_num) {
  int left_range_block_idx = offset / range_block_size;
  int right_range_block_idx = (offset + range_block_size - 1) / range_block_size;

  // sync_num == 0 means conflict_state found no predecessor batch for that range
  // block, so there is nothing to wait for. The two sides must be tested
  // separately: a tile straddling two range blocks can have a predecessor on one
  // side only, and a lock never returns to 0 once any batch has published to it.
  if (left_range_sync_num != 0) {
    int left_range_index = left_range_block_idx * num_heads + bidh;

#pragma unroll 1
    while (atomicCAS(&range_lock[left_range_index * 2], left_range_sync_num, left_range_sync_num) != left_range_sync_num) {
    }
  }

  if (left_range_block_idx != right_range_block_idx && right_range_sync_num != 0) {
    int right_range_index = right_range_block_idx * num_heads + bidh;

#pragma unroll 1
    while (atomicCAS(&range_lock[right_range_index * 2], right_range_sync_num, right_range_sync_num) != right_range_sync_num) {
    }
  }
}

CUTLASS_DEVICE
void deterministic_arrive(
    int* range_lock,
    int bidh,
    int offset,
    int range_block_size,
    int num_heads,
    int arrive_num,
    bool left_range_arrive_twice,
    bool right_range_arrive_twice) {
  int left_range_block_idx = offset / range_block_size;
  int left_range_index = left_range_block_idx * num_heads + bidh;
  int right_range_block_idx = (offset + range_block_size - 1) / range_block_size;
  int right_range_index = right_range_block_idx * num_heads + bidh;

  int add_cnt = right_range_arrive_twice ? 2 : 1;
  int tmp = atomicAdd(&range_lock[right_range_index * 2 + 1], add_cnt);
  if (tmp + add_cnt == 2) {
    atomicExch(&range_lock[right_range_index * 2 + 1], 0);
    atomicExch(&range_lock[right_range_index * 2], arrive_num);
  }

  add_cnt = left_range_arrive_twice ? 2 : 1;
  tmp = atomicAdd(&range_lock[left_range_index * 2 + 1], add_cnt);
  if (tmp + add_cnt == 2) {
    atomicExch(&range_lock[left_range_index * 2 + 1], 0);
    atomicExch(&range_lock[left_range_index * 2], arrive_num);
  }
}

} // namespace flash
