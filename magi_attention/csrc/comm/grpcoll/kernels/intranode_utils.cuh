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

#include <functional>
#include <optional>

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"

namespace magi_attn_comm::grpcoll::intranode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks>
DEVICE_INLINE void cached_notify_group_cast_func(const int* rank_prefix_matrix, size_t num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
  // Barrier before cleaning
  barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

  // Copy and clean
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto ptr = static_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
    ptr[i] = rank_prefix_matrix[i];
#pragma unroll
  for (size_t i = thread_id; i < num_memset_int; i += num_threads)
    ptr[kNumRanks * kNumRanks + i] = 0;

  // Barrier after cleaning
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks>
DEVICE_INLINE void cached_notify_group_reduce_func(size_t num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
  // Barrier before cleaning
  barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

  // Clean
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto ptr = static_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (size_t i = thread_id; i < num_memset_int; i += num_threads)
    ptr[i] = 0;

  // Barrier after cleaning
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

template <int kNumRanks>
DEVICE_INLINE void reset_send_head_before_group_reduce_func(int* send_head, int num_reduced_tokens, int num_channels, int channel_id) {
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto rank_id = thread_id / WARP_SIZE;
  const auto lane_id = thread_id % WARP_SIZE;
  if (rank_id >= kNumRanks)
    return;

  int token_start_idx, token_end_idx;
  get_channel_task_range(num_reduced_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

  /** NOTE: the process below is to find the correct next valid head `p`
   * for those `-1` entries, and in-place update them to the encoded `-p-1`
   * since in the group-reduce stage, the receivers need to update the `expected_head`
   * to next valid position by decoding with `-expected_head - 1` when they reach certain `-1` entry
   * and the reason of encoding `-p-1` is to maintain the `-1` entries still negative
   */
  int last_head = 1 << 25; // NOTE: `1 << 25` is a heuristic large number
#pragma unroll
  for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= WARP_SIZE) {
    int token_idx = token_idx_tail - lane_id, expected_head = 0;
    auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
    for (int i = 0; i < min(WARP_SIZE, token_idx_tail - token_start_idx + 1); ++i) {
      const int head = broadcast_in_warp(/*val=*/current_head, /*src_lane=*/i);
      if (head < 0) {
        if (lane_id == i)
          expected_head = encode(last_head);
      } else {
        last_head = head;
      }
    }
    if (current_head < 0 and token_idx >= token_start_idx)
      send_head[token_idx * kNumRanks + rank_id] = expected_head;
  }
}

} // namespace magi_attn_comm::grpcoll::intranode
