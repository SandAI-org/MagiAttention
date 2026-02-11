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

#include "configs.cuh"
#include "intranode_notify_kernel.cuh"
#include "intranode_utils.cuh"

namespace magi_attn_comm::grpcoll::intranode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast (Cached) Notify
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks, bool kRequireRecvCount>
__global__ void notify_group_cast_kernel(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_tokens,
    int num_channels,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix,
    size_t num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto lane_id = thread_id % WARP_SIZE, warp_id = thread_id / WARP_SIZE, num_warps = num_threads / WARP_SIZE;

  if (sm_id == 0) { // the first SM for counting the `num_recv_tokens`, which will return back to CPU, and calculating `rank_size_prefix`
    // Barrier first
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // `per_rank_buffer` has shape (kNumRanks, kNumRanks)
    // starting at the beginning of the buffer ptr of each rank
    int* per_rank_buffer;
    if (thread_id < kNumRanks)
      per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);

    // After this loop:
    //  - `per_rank_buffer[rank][i, j]`: the number of tokens from rank i to rank j
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i)
        per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
    }

    // Wait for all ranks to be finished
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Sum per-rank counts and return to CPU
    // Also pre-compute the prefix sum for data sending
    auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
    auto buffer_ptr_after_rank_prefix = local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 1; i < kNumRanks; ++i) {
        local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      }

      // Notify the host the total number of received tokens if required
      if constexpr (kRequireRecvCount) {
        if (thread_id == rank) {
          auto num_recv_tokens = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];

          // Self-rotated wait for the counter reset by the host
          auto start_time = clock64();
          while (true) {
            auto recv_counter_value = ld_volatile_global(grpcoll_recv_counter_mapped);
            if (recv_counter_value == -1)
              break;

            // Timeout check
            if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
              printf(
                  "grpcoll timeout for intranode notify_group_cast recv counter with thread: %d, rank: %d, num_recv_tokens: %d, recv_counter_value: %d\n",
                  thread_id,
                  rank,
                  num_recv_tokens,
                  recv_counter_value);
              trap();
            }
          }

          *grpcoll_recv_counter_mapped = num_recv_tokens;
        }
      }
    }

    __syncthreads();

#pragma unroll
    // Copy `rank_size_prefix` matrix to an individual tensor
    // where the original part left in buffer will be skipped in `group_cast`
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
      rank_prefix_matrix[i] = local_per_rank_buffer[i];

#pragma unroll
    // Extra memset for later channel metadata
    // including channel start/end offset, head and tail
    for (size_t i = thread_id; i < num_memset_int; i += num_threads)
      buffer_ptr_after_rank_prefix[i] = 0;

    // Barrier
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  } else { // the rest SMs are responsible for calculating the `channel_prefix_matrix`
    int dst_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens
      int count = 0;
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += WARP_SIZE)
        count += is_token_in_rank[i * kNumRanks + dst_rank];
      count = warp_reduce_sum(count);
      if (lane_id == 0)
        channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
    }
    __syncthreads();

    // Pre-compute prefix sum for all channels
    if (thread_id == 0) {
#pragma unroll
      for (int i = 1; i < num_channels; ++i)
        channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
    }
  }
}

void notify_group_cast(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_ranks,
    int num_tokens,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix,
    size_t num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int num_channels,
    bool require_recv_count) {
  constexpr int kNumThreads = 128;

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);

  RANKS_SWITCH(num_ranks, kNumRanks, [&] {
    BOOL_SWITCH(require_recv_count, kRequireRecvCount, [&] {
      LAUNCH_KERNEL(
          &cfg,
          notify_group_cast_kernel<kNumRanks, kRequireRecvCount>,
          num_tokens_per_rank,
          grpcoll_recv_counter_mapped,
          num_tokens,
          num_channels,
          is_token_in_rank,
          channel_prefix_matrix,
          rank_prefix_matrix,
          num_memset_int,
          buffer_ptrs,
          barrier_signal_ptrs,
          rank);
    });
  });
}

template <int kNumRanks>
__global__ void cached_notify_group_cast_kernel(const int* rank_prefix_matrix, size_t num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
  cached_notify_group_cast_func<kNumRanks>(rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank);
}

void cached_notify_group_cast(
    const int* rank_prefix_matrix,
    size_t num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(1, 128, stream);

  RANKS_SWITCH(num_ranks, kNumRanks, [&] {
    LAUNCH_KERNEL(&cfg, cached_notify_group_cast_kernel<kNumRanks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank);
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce Cached Notify
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks>
__global__ void cached_notify_group_reduce_kernel(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_reduced_tokens,
    size_t num_memset_int,
    int** barrier_signal_ptrs,
    int rank) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  if (sm_id == 0) {
    cached_notify_group_reduce_func<kNumRanks>(num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank);
  } else {
    const auto channel_id = sm_id - 1;
    reset_send_head_before_group_reduce_func<kNumRanks>(send_head, num_reduced_tokens, num_channels, channel_id);
  }
}

void cached_notify_group_reduce(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_reduced_tokens,
    size_t num_memset_int,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
  const int num_threads = std::max(128, WARP_SIZE * num_ranks);
  GRPCOLL_HOST_ASSERT(num_threads <= 1024);

  // The first SM is to barrier and clean flag
  // while the rest SMs are to reset send_head per channel
  const int num_sms = 1 + num_channels;
  GRPCOLL_HOST_ASSERT(num_sms <= num_channels * 2);
  SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream);

  RANKS_SWITCH(num_ranks, kNumRanks, [&] {
    LAUNCH_KERNEL(
        &cfg, cached_notify_group_reduce_kernel<kNumRanks>, buffer_ptrs, send_head, num_channels, num_reduced_tokens, num_memset_int, barrier_signal_ptrs, rank);
  });
}

template <int kNumRanks>
__global__ void reset_send_head_before_group_reduce_kernel(int* send_head, int num_channels, int num_reduced_tokens) {
  const auto channel_id = static_cast<int>(blockIdx.x);
  reset_send_head_before_group_reduce_func<kNumRanks>(send_head, num_reduced_tokens, num_channels, channel_id);
}

void reset_send_head_before_group_reduce(int* send_head, int num_channels, int num_reduced_tokens, int num_ranks, cudaStream_t stream) {
  const int num_threads = std::max(128, WARP_SIZE * num_ranks);
  GRPCOLL_HOST_ASSERT(num_threads <= 1024);

  // One SM per channel
  SETUP_LAUNCH_CONFIG(num_channels, num_threads, stream);

  RANKS_SWITCH(num_ranks, kNumRanks, [&] { LAUNCH_KERNEL(&cfg, reset_send_head_before_group_reduce_kernel<kNumRanks>, send_head, num_channels, num_reduced_tokens); });
}

} // namespace magi_attn_comm::grpcoll::intranode
