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

/**********************************************************************************
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *********************************************************************************/

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "reduce_op.cuh"
#include "utils.cuh"

namespace magi_attn_comm::grpcoll {

namespace intranode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// (Cached) Notify Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks>
__global__ void notify_group_cast_kernel(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_tokens,
    int num_channels,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix,
    int num_memset_int,
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
      for (int i = 1; i < kNumRanks; ++i)
        local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      if (thread_id == rank)
        *grpcoll_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
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
    for (int i = thread_id; i < num_memset_int; i += num_threads)
      buffer_ptr_after_rank_prefix[i] = 0;

    // Barrier
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  } else { // the rest SMs are responsible for calculating the `channel_prefix_matrix`, `is_token_in_rank`
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
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int num_channels) {
  constexpr int kNumThreads = 128;

#define NOTIFY_GROUP_CAST_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL(                             \
      &cfg,                                  \
      notify_group_cast_kernel<ranks>,       \
      num_tokens_per_rank,                   \
      grpcoll_recv_counter_mapped,           \
      num_tokens,                            \
      num_channels,                          \
      is_token_in_rank,                      \
      channel_prefix_matrix,                 \
      rank_prefix_matrix,                    \
      num_memset_int,                        \
      buffer_ptrs,                           \
      barrier_signal_ptrs,                   \
      rank);                                 \
  break

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
  SWITCH_RANKS(NOTIFY_GROUP_CAST_LAUNCH_CASE);
#undef NOTIFY_GROUP_CAST_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_group_cast_kernel(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
  // A simplified version for cached handles
  barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

  // Copy and clean
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto ptr = static_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
    ptr[i] = rank_prefix_matrix[i];
#pragma unroll
  for (int i = thread_id; i < num_memset_int; i += num_threads)
    ptr[kNumRanks * kNumRanks + i] = 0;

  // Barrier after cleaning
  barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void cached_notify_group_cast(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
#define CACHED_NOTIFY_GROUP_CAST_LAUNCH_CASE(ranks)                                                                                        \
  LAUNCH_KERNEL(&cfg, cached_notify_group_cast_kernel<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 128, stream);
  SWITCH_RANKS(CACHED_NOTIFY_GROUP_CAST_LAUNCH_CASE);
#undef CACHED_NOTIFY_GROUP_CAST_LAUNCH_CASE
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumDataGroups, int kNumRanks, int kNumThreads, int kWarpCopyUnrollStages, int kNumTMAStages, int kNumTMABytesPerWarp, bool kCastLSE>
GLOBAL_LAUNCH_BOUNDS(kNumThreads, 1)
void group_cast_kernel(
    /* 1st group of input / output data*/
    int4* recv_x,
    float* recv_lse,
    const int4* x,
    const float* lse,
    /* 2nd group of input / output data*/
    int4* recv_x_2nd,
    const int4* x_2nd,
    /* 3rd group of input / output data*/
    int4* recv_x_3rd,
    const int4* x_3rd,
    /* other metadata */
    int* recv_src_idx,
    int* recv_channel_offset,
    int* send_head,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    const int64_t* post_perm_idx,
    int num_tokens,
    int hidden_int4,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  // Get thread Info
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const bool is_sender = sm_id % 2 == 0; // even-numbered SMs are senders

  // Get rank Info
  const auto num_threads_per_rank = kNumThreads / kNumRanks; // Several warps are response for a single rank
  const auto responsible_rank = thread_id / num_threads_per_rank;
  const auto send_rank = is_sender ? rank : responsible_rank, recv_rank = is_sender ? responsible_rank : rank;

  // Get channel Info
  const auto num_channels = num_sms / 2, responsible_channel = sm_id / 2;
  const auto num_channels_total = num_channels * kNumRanks, channel_rank_offset = responsible_channel * kNumRanks + send_rank; // each rank has SM/2 channels
  const auto num_channel_tokens_total = num_channels_total * num_recv_buffer_tokens, channel_rank_token_offset = channel_rank_offset * num_recv_buffer_tokens;
  const auto responsible_rank_channel = responsible_rank * num_channels + responsible_channel;

  // Get buffer ptr of the recv rank
  // (the metadata of any pair of (sender, receiver) is all stored on the receiver side)
  // and jump across the temp rank prefix matrix, consumed in `notify_group_cast`
  // `rank_prefix_matrix`: shape=(kNumRanks, kNumRanks), dtype=int
  auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[recv_rank]) + kNumRanks * kNumRanks * sizeof(int));

  // Get channel metadata buffers
  // (senders are responsible for tails, and receivers are responsible for heads)
  //  `start_offset`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `end_offset`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `head_idx`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `tail_idx`: shape=(kNumChannels, kNumRanks), dtype=int
  auto channel_start_offset = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_end_offset = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_head_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_tail_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);

  // Get channel data buffers
  //  `x_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4), dtype=int4, alignment=sizeof(int4)
  //  `src_idx_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens), dtype=int
  //  `lse_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_heads), dtype=float
  //  NOTES: we gurantee that `num_heads` will be `0` if `kCastLSE` is `false` to make `lse_buffers` empty
  auto channel_x_buffers =
      Buffer<int4, sizeof(int4)>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4, /*elem_offset=*/channel_rank_token_offset * hidden_int4);
  auto channel_src_idx_buffers = Buffer<int>(ptr, /*num_elems=*/num_channel_tokens_total, /*elem_offset=*/channel_rank_token_offset);
  Buffer<float> channel_lse_buffers;
  if constexpr (kCastLSE)
    channel_lse_buffers = Buffer<float>(ptr, /*num_elems=*/num_channel_tokens_total * num_heads, /*elem_offset=*/channel_rank_token_offset * num_heads);

  // Get channel data buffers for other groups
  //  `x_buffers_2nd`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4), dtype=int4, alignment=sizeof(int4)
  //  `x_buffers_3rd`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4), dtype=int4, alignment=sizeof(int4)
  constexpr bool kIs2ndGroupExists = kNumDataGroups > 1;
  constexpr bool kIs3rdGroupExists = kNumDataGroups > 2;
  Buffer<int4, sizeof(int4)> channel_x_buffers_2nd;
  Buffer<int4, sizeof(int4)> channel_x_buffers_3rd;
  if constexpr (kIs2ndGroupExists)
    channel_x_buffers_2nd =
        Buffer<int4, sizeof(int4)>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4, /*elem_offset=*/channel_rank_token_offset * hidden_int4);
  if constexpr (kIs3rdGroupExists)
    channel_x_buffers_3rd =
        Buffer<int4, sizeof(int4)>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4, /*elem_offset=*/channel_rank_token_offset * hidden_int4);

  // Get copy info
#ifndef DISABLE_SM90_FEATURES
  // Get TMA copy info
  const int hidden_int4_per_stage = hidden_int4 / kNumTMAStages;
  const int hidden_bytes_per_stage = hidden_int4_per_stage * sizeof(int4);

  // Prepare TMA buffer in dynamic shared memory for this warp
  extern __shared__ __align__(1024) uint8_t smem_buffer[]; // REVIEW: why aligned to 1024 bytes ?
  auto tma_buffer = smem_buffer + warp_id * kNumTMABytesPerWarp;

  // Init the TMA stage and mbarrier for this warp
  uint32_t tma_stage = 0;
  auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + hidden_bytes_per_stage);
  if (lane_id == 0) { // the lane0 in this warp
    mbarrier_init(tma_mbarrier, 1);
    fence_view_async_shared();
    fence_barrier_init();
  }
  __syncwarp();
#endif

  if (is_sender) {
    // Ger send warp info
    // NOTES: the warps in one block are first divided into `kNumRanks` warp groups
    // where each warp group is responsible for one rank, with the group size of `num_send_warps / kNumRanks`
    constexpr int num_send_warps = kNumThreads / WARP_SIZE;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const auto send_warp_id_in_rank = warp_id % num_send_warps_per_rank;
    const auto max_num_used_slots_in_queue = num_recv_buffer_tokens - num_max_send_tokens;

    GRPCOLL_STATIC_ASSERT(kNumRanks <= WARP_SIZE, "Invalid number of ranks");
    GRPCOLL_STATIC_ASSERT(num_send_warps % kNumRanks == 0, "Invalid number of send warps");

    // Store the channel start_offset, end_offset from the channel_prefix_matrix
    if (lane_id == 0 and send_warp_id_in_rank == 0) { // the lane0 in the send warp0 for this rank
      // Send offset by code: `-value - 1`, e.g. 0 -> -1, 1 -> -2
      // NOTES: this is for distinguishing zero tokens
      // and the receiver will restore the real offset by: `-code - 1`

      // Send start offset code into the receiver's channel_start_offset buffer
      int value = responsible_channel > 0 ? channel_prefix_matrix[responsible_rank_channel - 1] : 0;
      st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1); // system scope, relaxed order

      // Send end offset code into the receiver's channel_end_offset buffer
      value = channel_prefix_matrix[responsible_rank_channel];
      st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1); // system scope, relaxed order
    }
    __syncwarp();

    // Get send tasks
    // i.e. the range of tokens [start_idx, end_idx) in `x` for the responsible channel
    // NOTES: this range does not distiguish the destination rank,
    // thus every warp in the block will get the same range
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

    // Iterate over all tokens and send by chunks (chunk_size=num_max_send_tokens)
    int cached_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Wait queue empty enough to send one chunk
      auto start_time = clock64();
      while (lane_id == 0) { // the lane0 in this warp
        // Load channel head idx stored by the receiver
        // NOTES: the head idxs received by each warp for the responsible rank might not be the same
        int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer()); // volatile load

        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        if (num_used_slots <= max_num_used_slots_in_queue)
          break; // the empty slots in recv queue is enough for this send chunk size

        // Check timeout
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("grpcoll timeout for group cast senders, rank=%d, responsible_channel=%d\n", rank, responsible_channel);
          trap();
        }
        // Rare cases to loop again
      }
      __syncwarp();

      // Send one chunk
      int chunk_token_idx = 0;
      while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
        // NOTES: for the same token, the warp assigned to save `send_head`
        // may be different from the warp assigned to send the following data
        const auto is_token_in_responsible_rank = is_token_in_rank[token_idx * kNumRanks + responsible_rank];

        // Pick (round-robin) one warp for the responsible rank and let its lane0 save the send head
        if (lane_id == 0 and token_idx % num_send_warps_per_rank == send_warp_id_in_rank)
          // send_head: shape=[num_tokens, num_ranks]:
          // send_head[i, r]: the offset in the corr. channel of send token i if it needs to be sent to rank r
          // since the cached_channel_tail_idx starts at 0, so when token_idx == token_start_idx for the corr. channel
          // the send_head[:, r] will be several cu_seqlens that look like:
          //     [0, 1, ... channel0_size, 0, 1, ... channel1_size, ...]
          // and if is_token_in_rank[i, r] == -1, then send_head[i, r] == -1 and should be ignored in the cu_seqlens above
          send_head[token_idx * kNumRanks + responsible_rank] = is_token_in_responsible_rank ? cached_channel_tail_idx : -1;

        // Skip if this token won't be sent to the responsible rank
        if (!is_token_in_responsible_rank) {
          token_idx++;
          continue;
        }

        // Get an empty slot in recv queue
        int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;

        // Pick (round-robin) one warp for the responsible rank to send this token
        if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
          // Get the token ptr in the send buffer and the recv queue
          int4* token_ptr_in_queue_array[kNumDataGroups];
          const int4* token_ptr_in_x_array[kNumDataGroups];

          // for the 1st group
          token_ptr_in_queue_array[0] = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
          token_ptr_in_x_array[0] = x + token_idx * hidden_int4;

          // for the 2nd group if exists
          if constexpr (kIs2ndGroupExists) {
            token_ptr_in_queue_array[1] = channel_x_buffers_2nd.buffer() + dst_slot_idx * hidden_int4;
            token_ptr_in_x_array[1] = x_2nd + token_idx * hidden_int4;
          }

          // for the 3rd group if exists
          if constexpr (kIs3rdGroupExists) {
            token_ptr_in_queue_array[2] = channel_x_buffers_3rd.buffer() + dst_slot_idx * hidden_int4;
            token_ptr_in_x_array[2] = x_3rd + token_idx * hidden_int4;
          }

#pragma unroll
          // Warp-copy this token from send buffer to the recv queue
          for (int i = 0; i < kNumDataGroups; ++i) {
            auto token_ptr_in_queue = token_ptr_in_queue_array[i];
            auto token_ptr_in_x = token_ptr_in_x_array[i];

            UNROLLED_WARP_COPY(
                /*UNROLL_FACTOR=*/kWarpCopyUnrollStages,
                /*LANE_ID=*/lane_id,
                /*N=*/hidden_int4,
                /*DST=*/token_ptr_in_queue,
                /*SRC=*/token_ptr_in_x,
                /*LD_FUNC=*/__ldg, // read-only load, REVIEW: why not use `ld_nc_global` here ?
                /*ST_FUNC=*/st_na_global // non-cached store
            );
          }

          // Copy channel src idx by lane0
          // which will be used to fill `recv_src_idx` by the receiver
          if (lane_id == 0)
            channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

          // Warp-strided copy `lse` to `channel_lse` by this warp
          // which is used to fill `recv_lse` by the receiver
          // if `kCastLSE` is `true`
          if constexpr (kCastLSE) {
#pragma unroll
            for (int i = lane_id; i < num_heads; i += WARP_SIZE)
              channel_lse_buffers[dst_slot_idx * num_heads + i] = __ldg(lse + token_idx * num_heads + i);
          }
        }

        // Update global token idx and the local token idx in this send chunk
        chunk_token_idx++, token_idx++;
      }

      // Sync all send warps for the responsible rank
      sync_warp_group(/*group_flag=*/responsible_rank, /*group_size=*/num_threads_per_rank);

      // Update tail idx for the responsible rank w.r.t. the responsible channel
      // NOTES: here all send warps for the responsible rank are supposed to share the same new tail
      // since they update it in the same way in the above loop, though they handle different tokens in a round-robin way
      if (lane_id == 0 and send_warp_id_in_rank == 0) // the lane0 in the send warp0 for the responsible rank
        st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx); // system scope, release order
    }
  } else {
    // Ger recv warp info
    // NOTES: the warps in one block are first divided into `kNumRanks` warp groups
    // where each warp group is responsible for one rank, with the group size of `num_recv_warps / kNumRanks`
    constexpr int num_recv_warps = kNumThreads / WARP_SIZE;
    constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
    constexpr int num_recv_threads_per_rank = num_recv_warps_per_rank * WARP_SIZE;
    const auto recv_thread_id = thread_id;
    const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
    const auto recv_warp_id_in_rank = recv_thread_id_in_rank / WARP_SIZE;

    GRPCOLL_STATIC_ASSERT(kNumRanks <= WARP_SIZE, "Invalid number of ranks");
    GRPCOLL_STATIC_ASSERT(num_recv_warps % kNumRanks == 0, "Invalid number of recv warps");

    // Get global rank offset for the responsible rank from the rank prefix matrix
    auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[recv_rank]);
    int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + recv_rank] : 0;

    // Load non-empty channel start/end offset stored by the sender by lane0 in each warp
    int total_offset, num_tokens_to_recv;
    while (lane_id == 0 and (total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0) // volatile load
      ;
    while (lane_id == 0 and (num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0) // volatile load
      ;
    if (lane_id == 0) {
      // Recover the real channel start/end offset from the code by `-code - 1`
      total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;

      // Store channel start offset to the `recv_channel_offset`
      if (recv_warp_id_in_rank == 0) // the lane0 in the recv warp0 for the responsible rank
        // Here, total_offset = channel_start_offset
        recv_channel_offset[responsible_rank_channel] = total_offset;

      // Here, num_tokens_to_recv = channel_end_offset - channel_start_offset
      // = num_tokens to recv for the responsible rank w.r.t. the responsible channel
      num_tokens_to_recv -= total_offset;
    }

    // Broadcast total_offset to other lanes
    total_offset = broadcast_in_warp(total_offset);
    // Here, total_offset = rank_offset + channel_start_offset
    // = the global token offset in the send buffer of the start token in the channel
    total_offset += rank_offset;

    // Broadcast num_tokens_to_recv to other lanes
    num_tokens_to_recv = broadcast_in_warp(num_tokens_to_recv);

    // Prepare shared tail idxs for each rank in static shared memory
    // NOTES: unlike the sender, the receiver must ensure that
    // the tail index hold by all warps for the responsible rank are the same
    // thus we cannot use `broadcast_in_warp` to sync the tail idx
    // but only utilize the shared memory to sync across warps
    __shared__ volatile int shared_channel_tail_idx[kNumRanks];

    // Recv tokens by rounds
    auto start_time = clock64();
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) { // non-empty tokens to recv in the queue
      // Wait for the queue to be non-empty
      while (recv_thread_id_in_rank == 0) { // the thread0 for the responsible rank, i.e. the lane0 in the recv warp0 for the responsible rank
        // Load channel tail idx stored by the sender
        cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer()); // system scope, acquire order

        // Check if the queue is non-empty
        if (cached_channel_head_idx != cached_channel_tail_idx) {
          // Store into shared memory to broadcast to all warps for the responsible rank later
          shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
          break;
        }

        // Check timeout
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("grpcoll timeout for group cast receivers, rank=%d, responsible_channel=%d, tokens_to_recv=%d\n", rank, responsible_channel, num_tokens_to_recv);
          trap();
        }
      }

      // Synchronize all warps for the responsible rank
      sync_warp_group(/*group_flag=*/responsible_rank, /*group_size=*/num_threads_per_rank);

      // Load the channel tail idx from the shared memory
      // which is the same for all warps for the responsible rank
      cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

      // Warp-copy received tokens from recv queue to recv buffer
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) { // warp-group strided
        // Determine the final dst token idx in the recv buffer
        auto token_idx_in_recv_x = static_cast<int64_t>(total_offset + chunk_idx); // original token idx in recv buffer in rank order
        token_idx_in_recv_x = post_perm_idx == nullptr ? token_idx_in_recv_x : post_perm_idx[token_idx_in_recv_x];

        // Get the token idx in the recv queue
        int token_idx_in_queue = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;

        // Get the token ptr in the recv buffer and the recv queue
        int4* token_ptr_in_recv_x_int4_array[kNumDataGroups];
        int4* token_ptr_in_queue_int4_array[kNumDataGroups];

        // for the 1st group
        token_ptr_in_recv_x_int4_array[0] = recv_x + token_idx_in_recv_x * hidden_int4;
        token_ptr_in_queue_int4_array[0] = channel_x_buffers.buffer() + token_idx_in_queue * hidden_int4;

        // for the 2nd group if exists
        if constexpr (kIs2ndGroupExists) {
          token_ptr_in_recv_x_int4_array[1] = recv_x_2nd + token_idx_in_recv_x * hidden_int4;
          token_ptr_in_queue_int4_array[1] = channel_x_buffers_2nd.buffer() + token_idx_in_queue * hidden_int4;
        }

        // for the 3rd group if exists
        if constexpr (kIs3rdGroupExists) {
          token_ptr_in_recv_x_int4_array[2] = recv_x_3rd + token_idx_in_recv_x * hidden_int4;
          token_ptr_in_queue_int4_array[2] = channel_x_buffers_3rd.buffer() + token_idx_in_queue * hidden_int4;
        }

        // Copy this token from recv queue to recv buffer
#ifndef DISABLE_SM90_FEATURES
#pragma unroll
        // TMA-copy
        for (int i = 0; i < kNumDataGroups; ++i) {
          auto token_ptr_in_recv_x_int4 = token_ptr_in_recv_x_int4_array[i];
          auto token_ptr_in_queue_int4 = token_ptr_in_queue_int4_array[i];

#pragma unroll
          // multiple TMA stages
          for (int j = 0; j < kNumTMAStages; ++j) {
            if (lane_id == 0) { // only the lane0 in this warp issues the TMA
              // Wait for all previous TMA stores to be finished
              // REVIEW: can we use multiple buffers for multiple stages ?
              tma_store_wait();

              // Load the token from recv queue to shared memory
              tma_load_1d(
                  /*smem_ptr=*/tma_buffer,
                  /*gmem_ptr=*/token_ptr_in_queue_int4 + j * hidden_int4_per_stage,
                  /*mbar_ptr=*/tma_mbarrier,
                  /*num_bytes=*/hidden_bytes_per_stage,
                  /*evict_first=*/true // evict the read-once token in recv queue first from L2 cache
              );

              // Barrier the last load above to be finished before the next store below
              // NOTES: TMA stage will be inplace updated
              mbarrier_arrive_and_expect_tx(/*mbar_ptr=*/tma_mbarrier, /*num_bytes=*/hidden_bytes_per_stage);
              mbarrier_wait(/*mbar_ptr=*/tma_mbarrier, /*stage=*/tma_stage, /*num_tma_stages=*/kNumTMAStages);

              // Store the token from shared memory to recv buffer
              tma_store_1d(
                  /*smem_ptr=*/tma_buffer,
                  /*gmem_ptr=*/token_ptr_in_recv_x_int4 + j * hidden_int4_per_stage,
                  /*num_bytes=*/hidden_bytes_per_stage,
                  /*evict_first=*/false);
            }
          }
        }
        __syncwarp();
#else
#pragma unroll
        // Warp-copy
        for (int i = 0; i < kNumDataGroups; ++i) {
          auto token_ptr_in_recv_x_int4 = token_ptr_in_recv_x_int4_array[i];
          auto token_ptr_in_queue_int4 = token_ptr_in_queue_int4_array[i];

          UNROLLED_WARP_COPY(
              /*UNROLL_FACTOR=*/kWarpCopyUnrollStages,
              /*LANE_ID=*/lane_id,
              /*N=*/hidden_int4,
              /*DST=*/token_ptr_in_recv_x_int4,
              /*SRC=*/token_ptr_in_queue_int4,
              /*LD_FUNC=*/ld_nc_global, // non-cached load
              /*ST_FUNC=*/st_na_global // non-cached store
          );
        }
#endif
      }

#pragma unroll 4
      // Thead-copy `channel_src_idx` stored by the sender to `recv_src_idx`
      // NOTES: here we don't apply `post_perm_idx` to `recv_src_idx`
      for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
           chunk_idx += num_recv_threads_per_rank) // warp-group strided
        recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] = ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

      // Thread-copy `channel_lse` stored by the sender to `recv_lse`
      // if `kCastLSE` is `true`
      if constexpr (kCastLSE) {
#pragma unroll 4
        for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_heads; i += num_recv_threads_per_rank) { // warp-group strided
          int chunk_idx = i / num_heads, head_idx = i % num_heads;

          // Determine the final destination token idx in the recv buffer
          auto token_idx_in_recv_x = static_cast<int64_t>(total_offset + chunk_idx);
          token_idx_in_recv_x = post_perm_idx == nullptr ? token_idx_in_recv_x : post_perm_idx[token_idx_in_recv_x];

          // Get the token ptr in the recv queue
          int token_idx_in_queue = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
          auto token_ptr_in_queue = channel_lse_buffers.buffer() + token_idx_in_queue * num_heads + head_idx;

          recv_lse[token_idx_in_recv_x * num_heads + head_idx] = ld_nc_global(token_ptr_in_queue);
        }
      }

      // Update head idx for the responsible rank w.r.t. the responsible channel
      cached_channel_head_idx += num_recv_tokens;

      // Update the total offset of the start token idx in next round
      total_offset += num_recv_tokens;

      // Sync all send warps for the responsible rank
      sync_warp_group(/*group_flag=*/responsible_rank, /*group_size=*/num_threads_per_rank);

      // Store the new channel head idx to inform the sender
      if (lane_id == 0 and recv_warp_id_in_rank == num_recv_warps_per_rank - 1) // the lane0 in the last recv warp for the responsible rank
        st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx); // system scope, relaxed order

      // Update the remaining number of tokens to recv
      num_tokens_to_recv -= num_recv_tokens;
    }

#ifndef DISABLE_SM90_FEATURES
    // Wait for all previous TMA stores to be finished
    if (lane_id == 0)
      tma_store_wait();
#endif
  }
}

template <int kNumDataGroups, int kNumRanks, int kNumWarps>
void launch_group_cast(
    /* 1st group of input / output data*/
    void* recv_x,
    float* recv_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* recv_x_2nd,
    const void* x_2nd,
    /* 3rd group of input / output data*/
    void* recv_x_3rd,
    const void* x_3rd,
    /* other metadata */
    int* recv_src_idx,
    int* recv_channel_offset,
    int* send_head,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    const int64_t* post_perm_idx,
    int num_tokens,
    int hidden_int4,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  constexpr int kNumThreads = kNumWarps * WARP_SIZE; // num threads per block
  constexpr int kWarpCopyUnrollStages = 5; // warp-copy unroll stages
  constexpr int kNumTMAStages = 4; // num TMA stages
  constexpr int kNumTMABytesPerWarp = 8192; // num bytes of TMA transfer per warp

#ifndef DISABLE_SM90_FEATURES
  GRPCOLL_HOST_ASSERT(hidden_int4 % kNumTMAStages == 0);
  int hidden_bytes_per_stage = (hidden_int4 / kNumTMAStages) * sizeof(int4);
  GRPCOLL_HOST_ASSERT(hidden_bytes_per_stage + sizeof(uint64_t) <= kNumTMABytesPerWarp); // TMA buffer + mbarrier per warp
  constexpr int smem_size = kNumTMABytesPerWarp * kNumWarps; // shared memory size = num bytes of TMA transfer per block
#endif

  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 && kNumDataGroups <= 3, "Invalid kNumDataGroups");

#define GROUP_CAST_LAUNCH_CASE(cast_lse)                                                                                                          \
  {                                                                                                                                               \
    auto kernel = group_cast_kernel<kNumDataGroups, kNumRanks, kNumThreads, kWarpCopyUnrollStages, kNumTMAStages, kNumTMABytesPerWarp, cast_lse>; \
    SET_SHARED_MEMORY_FOR_TMA(kernel);                                                                                                            \
    LAUNCH_KERNEL(                                                                                                                                \
        &cfg,                                                                                                                                     \
        kernel,                                                                                                                                   \
        reinterpret_cast<int4*>(recv_x),                                                                                                          \
        recv_lse,                                                                                                                                 \
        reinterpret_cast<const int4*>(x),                                                                                                         \
        lse,                                                                                                                                      \
        reinterpret_cast<int4*>(recv_x_2nd),                                                                                                      \
        reinterpret_cast<const int4*>(x_2nd),                                                                                                     \
        reinterpret_cast<int4*>(recv_x_3rd),                                                                                                      \
        reinterpret_cast<const int4*>(x_3rd),                                                                                                     \
        recv_src_idx,                                                                                                                             \
        recv_channel_offset,                                                                                                                      \
        send_head,                                                                                                                                \
        is_token_in_rank,                                                                                                                         \
        channel_prefix_matrix,                                                                                                                    \
        post_perm_idx,                                                                                                                            \
        num_tokens,                                                                                                                               \
        hidden_int4,                                                                                                                              \
        num_heads,                                                                                                                                \
        buffer_ptrs,                                                                                                                              \
        rank,                                                                                                                                     \
        num_max_send_tokens,                                                                                                                      \
        num_recv_buffer_tokens);                                                                                                                  \
  }

#define GROUP_CAST_CAST_LSE_LAUNCH_CASE \
  {                                     \
    if (num_heads == 0) {               \
      GROUP_CAST_LAUNCH_CASE(false);    \
    } else {                            \
      GROUP_CAST_LAUNCH_CASE(true);     \
    }                                   \
  }

  // Even-numbered SMs for sending, odd-numbered SMs for receiving
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  GROUP_CAST_CAST_LSE_LAUNCH_CASE;

#undef GROUP_CAST_CAST_LSE_LAUNCH_CASE
#undef GROUP_CAST_LAUNCH_CASE
}

void group_cast(
    /* 1st group of input / output data*/
    void* recv_x,
    float* recv_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* recv_x_2nd,
    const void* x_2nd,
    /* 3rd group of input / output data*/
    void* recv_x_3rd,
    const void* x_3rd,
    /* other metadata */
    int* recv_src_idx,
    int* recv_channel_offset,
    int* send_head,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    const int64_t* post_perm_idx,
    int num_tokens,
    int hidden_int4,
    int num_heads,
    int num_groups,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
#define LAUNCH_INTRANODE_GROUP_CAST(num_groups, num_ranks, num_warps) \
  {                                                                   \
    launch_group_cast<num_groups, num_ranks, num_warps>(              \
        recv_x,                                                       \
        recv_lse,                                                     \
        x,                                                            \
        lse,                                                          \
        recv_x_2nd,                                                   \
        x_2nd,                                                        \
        recv_x_3rd,                                                   \
        x_3rd,                                                        \
        recv_src_idx,                                                 \
        recv_channel_offset,                                          \
        send_head,                                                    \
        is_token_in_rank,                                             \
        channel_prefix_matrix,                                        \
        post_perm_idx,                                                \
        num_tokens,                                                   \
        hidden_int4,                                                  \
        num_heads,                                                    \
        buffer_ptrs,                                                  \
        rank,                                                         \
        stream,                                                       \
        num_sms,                                                      \
        num_max_send_tokens,                                          \
        num_recv_buffer_tokens);                                      \
  }                                                                   \
  break

#define GROUP_CAST_DATA_GROUPS_LAUNCH_CASE(...)                       \
  {                                                                   \
    SWITCH_DATA_GROUPS_3(LAUNCH_INTRANODE_GROUP_CAST, ##__VA_ARGS__); \
  }                                                                   \
  break

  SWITCH_RANKS_WITH_WARPS(GROUP_CAST_DATA_GROUPS_LAUNCH_CASE);

#undef GROUP_CAST_DATA_GROUPS_LAUNCH_CASE
#undef LAUNCH_INTRANODE_GROUP_CAST
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cached Notify Group Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumRanks>
__global__ void cached_notify_group_reduce_kernel(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  if (sm_id == 0) {
    // Barrier before cleaning
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = static_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
      ptr[i] = 0;

    // Barrier after cleaning
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  } else {
    const auto channel_id = sm_id - 1;
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto rank_id = thread_id / WARP_SIZE;
    const auto lane_id = thread_id % WARP_SIZE;
    if (rank_id >= kNumRanks)
      return;

    int token_start_idx, token_end_idx;
    get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

    /** NOTE: the process below is to find the correct next valid head `p`
     * for those `-1` entries, and in-place update them to the encoded `-p-1`
     * since in the group-reduce stage, the receivers need to update the `expected_head`
     * to next valid position by decoding with `-expected_head - 1` when they reach certain `-1` entry
     * and the reason of encoding `-p-1` is to maintain the `-1` entries still negative
     */
    int last_head = 1 << 25; // NOTES: `1 << 25` is a heuristic large number
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
}

void cached_notify_group_reduce(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
#define CACHED_NOTIFY_GROUP_REDUCE(ranks)                                                                                                                          \
  LAUNCH_KERNEL(&cfg, cached_notify_group_reduce_kernel<ranks>, buffer_ptrs, send_head, num_channels, num_recv_tokens, num_memset_int, barrier_signal_ptrs, rank); \
  break

  const int num_threads = std::max(128, WARP_SIZE * num_ranks);
  GRPCOLL_HOST_ASSERT(num_threads <= 1024);
  GRPCOLL_HOST_ASSERT(1 + num_channels <= num_channels * 2);
  SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
  SWITCH_RANKS(CACHED_NOTIFY_GROUP_REDUCE);
#undef CACHED_NOTIFY_GROUP_REDUCE
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

// FIXME: the register usage is spilled for both load/store in some template cases
template <
    typename dtype_t,
    typename comm_dtype_t,
    typename reduce_dtype_t,
    ReduceOp kReduceOp,
    int kNumDataGroups,
    int kNumRanks,
    int kNumThreads,
    int kWarpCopyUnrollStages,
    int kNumTMAStages,
    int kNumTMABytesPerWarp,
    int kMaxNumHeads,
    bool kAccReduce>
GLOBAL_LAUNCH_BOUNDS(kNumThreads, 1)
void group_reduce_kernel(
    /* 1st group of input / output data*/
    dtype_t* reduced_x,
    float* reduced_lse,
    const dtype_t* x,
    const float* lse,
    /* 2nd group of input / output data*/
    dtype_t* reduced_x_2nd,
    const dtype_t* x_2nd,
    /* other metadata */
    int* send_head,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    const int64_t* pre_perm_idx,
    int num_tokens,
    int num_recv_tokens,
    int hidden_size,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  // Get thread Info
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const bool is_sender = sm_id % 2 == 0; // even-numbered SMs are senders

  // Get channel Info
  const auto num_channels = num_sms / 2;
  const int responsible_channel = sm_id / 2;
  const auto num_channels_total = num_channels * kNumRanks;
  const auto num_channel_tokens_total = num_channels_total * num_recv_buffer_tokens;

  // Get dtype Info
  GRPCOLL_STATIC_ASSERT(sizeof(int4) % sizeof(dtype_t) == 0, "Invalid vectorization");
  GRPCOLL_STATIC_ASSERT(sizeof(dtype_t) % sizeof(comm_dtype_t) == 0, "Invalid (dtype_t, comm_dtype_t)");
  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  constexpr int kCommDtypePerDtype = sizeof(dtype_t) / sizeof(comm_dtype_t);
  constexpr int kCommDtypePerInt4 = kCommDtypePerDtype * kDtypePerInt4;
  const int hidden_int4 = hidden_size / kDtypePerInt4;
  const int hidden_int4_comm = hidden_int4 / kCommDtypePerDtype;

  // Cast from `dtype_t` to `int4`
  constexpr bool kIsLSEReduce = kReduceOp == ReduceOp::LSE;
  constexpr bool kIs2ndGroupExists = kNumDataGroups > 1;
  auto x_int4 = reinterpret_cast<const int4*>(x);
  auto reduced_x_int4 = reinterpret_cast<int4*>(reduced_x);
  const int4* x_2nd_int4;
  int4* reduced_x_2nd_int4;
  if constexpr (kIs2ndGroupExists) {
    x_2nd_int4 = reinterpret_cast<const int4*>(x_2nd);
    reduced_x_2nd_int4 = reinterpret_cast<int4*>(reduced_x_2nd);
  }

  // Get copy info
#ifndef DISABLE_SM90_FEATURES
  // Get TMA copy info
  // Prepare TMA buffer in dynamic shared memory for this warp
  extern __shared__ __align__(1024) uint8_t smem_buffer[]; // REVIEW: why aligned to 1024 bytes ?
  auto tma_buffer = smem_buffer + warp_id * kNumTMABytesPerWarp;
  constexpr int kTMAStoreBytesPerWarp = WARP_SIZE * kCommDtypePerDtype * sizeof(int4); // the number of bytes per warp when using tma store
#endif

  if (is_sender) {
    // Ger send warp info
    // NOTES: the warps in one block are first divided into `num_send_warps / kNumRanks` warp groups
    // where for every single warp group, each warp is responsible for each rank, with the group size of `kNumRanks`
    // REVIEW: why interleaved organized, instead of following the way in group_cast stage ?
    constexpr int num_send_warps = kNumThreads / WARP_SIZE;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    constexpr int num_send_threads_per_rank = num_send_warps_per_rank * WARP_SIZE;
    const auto send_warp_id = warp_id;
    const auto responsible_rank = (responsible_channel + send_warp_id) % kNumRanks; // REVIEW: why shifted by responsible_channel ?
    const auto send_warp_id_in_rank = send_warp_id / kNumRanks;

    GRPCOLL_STATIC_ASSERT(num_send_warps % kNumRanks == 0, "Invalid number of send warps");

    // Get buffer ptr of the recv rank
    // (the metadata of any pair of (sender, receiver) is all stored on the receiver side)
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[responsible_rank]));
    const auto channel_rank_offset = responsible_channel * kNumRanks + rank;
    const auto channel_rank_token_offset = channel_rank_offset * num_recv_buffer_tokens;
    const auto responsible_rank_channel = responsible_rank * num_channels + responsible_channel;

    // Get channel metadata buffers
    // (senders are responsible for tails, and receivers are responsible for heads)
    // `head_idx`: shape=(kNumChannels, kNumRanks), dtype=int
    // `tail_idx`: shape=(kNumChannels, kNumRanks), dtype=int
    auto channel_head_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, channel_rank_offset);

    // Get channel data buffers
    // `x_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4_comm), dtype=int4, alignment=sizeof(int4)
    // `src_idx_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens), dtype=int
    // `lse_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_heads), dtype=float
    auto channel_x_buffers =
        Buffer<int4, sizeof(int4)>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4_comm, /*elem_offset=*/channel_rank_token_offset * hidden_int4_comm);
    auto channel_src_idx_buffers = Buffer<int>(ptr, /*num_elems=*/num_channel_tokens_total, /*elem_offset=*/channel_rank_token_offset);
    Buffer<float> channel_lse_buffers;
    if constexpr (kIsLSEReduce)
      channel_lse_buffers = Buffer<float>(ptr, /*num_elems=*/num_channel_tokens_total * num_heads, /*elem_offset=*/channel_rank_token_offset * num_heads);

    // Get channel data buffers for other groups
    //  `x_buffers_2nd`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4_comm), dtype=int4, alignment=sizeof(int4)
    Buffer<int4, sizeof(int4)> channel_x_buffers_2nd;
    if constexpr (kIs2ndGroupExists)
      channel_x_buffers_2nd =
          Buffer<int4, sizeof(int4)>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4_comm, /*elem_offset=*/channel_rank_token_offset * hidden_int4_comm);

    // Get rank offset
    // NOTES: `rank_prefix_matrix`: shape=(kNumRanks, kNumRanks), dtype=int
    //  is the same as the one in group_cast stage
    //  thus rank_prefix_matrix[:, rank]: the token end offsets sent by each rank to this rank in group_cast stage
    //  then, [rank_prefix_matrix[responsible_rank-1, rank], rank_prefix_matrix[responsible_rank, rank]) is the range of tokens in `x`
    //  which we should return back to responsible_rank in group_reduce stage by all warp groups for the responsible rank in all group_reduce SMs
    int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;
    int num_rank_tokens = rank_prefix_matrix[responsible_rank * kNumRanks + rank] - rank_offset;

    // Get channel offset
    // NOTES: `channel_prefix_matrix`: shape=(kNumRanks, kNumChannels), dtype=int
    //  is actually the `recv_channel_prefix_matrix` in group_cast stage
    //  thus channel_prefix_matrix[responsible_rank, :]: the token start offsets recv by responsible_rank for each channel to this rank in group_cast stage
    //  then, [channel_prefix_matrix[responsible_rank, responsible_channel], channel_prefix_matrix[responsible_rank, responsible_channel+1])
    //  is the local range of the responsible channel, for the tokens in `x`, recv by responsible_rank in group_cast stage
    //  which we should return back to responsible_rank in group_reduce stage by the warp group for the responsible rank in this group_reduce SM
    int channel_offset = channel_prefix_matrix[responsible_rank_channel];
    int num_channel_tokens = (responsible_channel == num_channels - 1 ? num_rank_tokens : channel_prefix_matrix[responsible_rank_channel + 1]) - channel_offset;

    // Get send tasks, i.e. the range of tokens [start_idx, end_idx) in `x` for the responsible channel w.r.t. the responsible rank
    // NOTES: this range distiguishs the destination rank, which is different from the one in group_cast stage
    int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

    // Iterate over all tokens sent to the responsible rank for the responsible channel
    // and send by chunks (chunk_size=min(num_max_send_tokens, token_end_idx-token_idx))
    int current_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Calculate chunk size for this round
      int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
      int max_num_used_slots_in_queue = num_recv_buffer_tokens - num_round_tokens;

      // Wait queue empty enough to send one chunk
      auto start_time = clock64();
      while (lane_id == 0) { // the lane0 in this warp
        // Load channel head idx stored by the receiver
        // NOTES: the head idxs received by each warp for the responsible rank might not be the same
        int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer()); // volatile load

        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        if (num_used_slots <= max_num_used_slots_in_queue)
          break;

        // Check timeout
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("grpcoll timeout for group reduce senders, rank=%d, responsible_channel=%d\n", rank, responsible_channel);
          trap();
        }
        // Rare cases to loop again
      }
      __syncwarp();

#pragma unroll
      // Send one chunk of tokens to the responsible rank
      for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) { // warp-group strided
        // Get an empty slot in recv queue
        int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;
        int send_token_idx = token_idx + i;

        // Determine the actual src token idx in the send buffer
        auto token_idx_in_x = static_cast<int64_t>(send_token_idx);
        token_idx_in_x = pre_perm_idx == nullptr ? token_idx_in_x : pre_perm_idx[token_idx_in_x];

        // Get the token ptr in the send buffer and the recv queue
        int4* token_ptr_in_queue_array[kNumDataGroups];
        const int4* token_ptr_in_x_array[kNumDataGroups];

        // for the 1st group
        token_ptr_in_queue_array[0] = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4_comm;
        token_ptr_in_x_array[0] = x_int4 + token_idx_in_x * hidden_int4;

        // for the 2nd group if exists
        if constexpr (kIs2ndGroupExists) {
          token_ptr_in_queue_array[1] = channel_x_buffers_2nd.buffer() + dst_slot_idx * hidden_int4_comm;
          token_ptr_in_x_array[1] = x_2nd_int4 + token_idx_in_x * hidden_int4;
        }

#pragma unroll
        // Warp-copy this token from send buffer to the recv queue
        for (int j = 0; j < kNumDataGroups; ++j) {
          int4* token_ptr_in_queue_int4 = token_ptr_in_queue_array[j];
          const int4* token_ptr_in_x_int4 = token_ptr_in_x_array[j];

          if constexpr (std::is_same_v<dtype_t, comm_dtype_t>) { // same comm dtype as `x.dtype`, so each int4 in `x` should copy to one int4 in queue
            UNROLLED_WARP_COPY(
                /*UNROLL_FACTOR=*/kWarpCopyUnrollStages,
                /*LANE_ID=*/lane_id,
                /*N=*/hidden_int4,
                /*DST=*/token_ptr_in_queue_int4,
                /*SRC=*/token_ptr_in_x_int4,
                /*LD_FUNC=*/ld_nc_global, // non-cached load
                /*ST_FUNC=*/st_na_global // non-cached store
            );
          } else { // low precision comm dtype, so each `kCommDtypePerDtype` int4 in `x` should copy to one int4 in queue
            UNROLLED_WARP_CAST_COPY(
                /*UNROLL_FACTOR=*/kWarpCopyUnrollStages,
                /*LANE_ID=*/lane_id,
                /*N=*/hidden_int4_comm,
                /*M=*/kCommDtypePerDtype,
                /*DST=*/token_ptr_in_queue_int4,
                /*SRC=*/token_ptr_in_x_int4,
                /*LD_FUNC=*/ld_nc_global, // non-cached load
                /*ST_FUNC=*/st_na_global, // non-cached store
                /*CAST_FUNC=*/(vec_downcast</*dtype_t=*/dtype_t, /*lp_dtype_t=*/comm_dtype_t, /*vec_dtype_t=*/int4>) // downcast from dtype_t to comm_dtype_t, given a
                                                                                                                     // `kCommDtypePerDtype` int4 ptr
            );
          }
        }

        // Copy channel src idx by lane0
        // NOTES:
        //  1. `src_idx` is actually the `recv_src_idx` in group_cast stage
        //    thus src_idx[j] indicates the token idx in `reduced_x` for x[j] to reduce to
        //  2. since we've NOT applied `post_perm_idx` to `recv_src_idx` in group cast stage,
        //    here we don't apply `pre_perm_idx` to `src_idx` either
        if (lane_id == 0)
          channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + send_token_idx);

        // Copy `lse` from the actual token idx in the send lse buffer
        // if `kIsLSEReduce`
        if constexpr (kIsLSEReduce) {
          for (int h = lane_id; h < num_heads; h += WARP_SIZE)
            channel_lse_buffers[dst_slot_idx * num_heads + h] = __ldg(lse + token_idx_in_x * num_heads + h);
        }
      }

      // Update token idx, channel tail idx by chunk size for last round
      token_idx += num_round_tokens;
      current_channel_tail_idx += num_round_tokens;

      // Sync all send warps for the responsible rank
      sync_warp_group(/*group_flag=*/responsible_rank, /*group_size=*/num_send_threads_per_rank);

      // Store the channel tail idx to inform the receiver
      if (lane_id == 0 and send_warp_id_in_rank == 0) // the lane0 in the send warp0 for the responsible rank
        st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx); // system scope, release order
    }
  } else {
    // Ger recv warp info
    // NOTES: the first warp for updating the queue head, others for reduction
    constexpr int num_reduce_warps = kNumThreads / WARP_SIZE - 1;
    const int reduce_warp_id = warp_id - 1, responsible_rank = lane_id;
    GRPCOLL_STATIC_ASSERT(kNumRanks <= WARP_SIZE, "Invalid number of ranks");
    GRPCOLL_STATIC_ASSERT(num_reduce_warps > 0, "Invalid number of warps");

    // Get head info
    // NOTES: the `num_heads` will be 0 if `kReduceOp != ReduceOp::LSE`
    // and `max_num_heads` will be 1 if `kReduceOp != ReduceOp::LSE`
    constexpr int max_num_heads = kIsLSEReduce ? kMaxNumHeads : 1;
    const int head_dim = kIsLSEReduce ? hidden_size / num_heads : -1;

    // Prepare some static shared memory for flags
    __shared__ volatile int shared_warp_channel_head_idx[num_reduce_warps][kNumRanks]; // channel heads for each reduce warp, each rank w.r.t. the responsible channel
    __shared__ volatile int shared_channel_tail_idx[kNumRanks]; // channel tails for each rank w.r.t. the responsible channel
    __shared__ volatile bool shared_warp_retired[num_reduce_warps]; // retired flags for each reduce warp w.r.t. the responsible channel

    // Prepare some static shared memory for temporary lse buffers
    // which will be read frequently while reducing the hidden values of some single token
    // FIXME: the bank conflict is very severe for these buffers
    __shared__ reduce_dtype_t shared_reduced_lse_buf[num_reduce_warps][max_num_heads]; // reduced lse buffer for each head, each reduce warp
    __shared__ reduce_dtype_t
        shared_old_lse_rescale_weight_buf[num_reduce_warps][max_num_heads]; // the weight to rescale the old `reduced_lse` for each head, each reduce warp

    // Init the static shared memory buffers
    if (thread_id < num_reduce_warps)
      shared_warp_retired[thread_id] = false;
    if (lane_id < kNumRanks)
      shared_warp_channel_head_idx[reduce_warp_id][lane_id] = 0;
    if (thread_id < kNumRanks)
      shared_channel_tail_idx[thread_id] = 0;

    // Sync all recv warps
    sync_warp_group(/*group_flag=*/0, /*group_size=*/kNumThreads);

    if (warp_id == 0) { // warp0 for updating the queue head, where each lane handles one rank
      // Get head/tail ptr of the responsible rank in buffer of the recv rank
      //  `head_idx`: shape=(kNumChannels, kNumRanks), dtype=int
      //  `tail_idx`: shape=(kNumChannels, kNumRanks), dtype=int
      int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + responsible_rank;
      int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels_total;

      // Self-rotate to update the queue head and retire other reduce warps
      int last_head = 0;
      while (responsible_rank < kNumRanks) {
        // Check whether all reduce warps are retired
        bool retired = true;
#pragma unroll
        for (int i = 0; i < num_reduce_warps; ++i) {
          retired &= shared_warp_retired[i];
          if (!retired)
            break;
        }
        if (retired)
          break; // if all reduce warps are retired, this warp can retire as well

        // Load queue tail for the responsible rank w.r.t. the responsible channel
        shared_channel_tail_idx[responsible_rank] = ld_acquire_sys_global(channel_tail_idx_ptr); // system scope, acquire order

        // Get minimum head across all reduce warps for each rank
        // which will be stored into `channel_head_idx` to inform the sender
        /** NOTE: the reason to get the minimum head is that:
         *  different from group-cast sender/receiver and group-reduce sender,
         *  the group-reduce receiver does not and cannot partition a specific warp groups for each rank
         *  since it might need to reduce the partial hidden values of the same token from all src ranks
         *  therefore, for certain rank, the queue head might be updated by any reduce warps
         *  and thus we need to get the minimum head across all reduce warps
         */
        int min_head = INT_MAX;
#pragma unroll
        for (int i = 0; i < num_reduce_warps; ++i) {
          if (!shared_warp_retired[i])
            min_head = min(min_head, shared_warp_channel_head_idx[i][responsible_rank]);
        }

        // Store queue head for the responsible rank w.r.t. the responsible channel
        // if the minimum head across all reduce warps is larger than the last head
        // and update the last head as well
        if (min_head != INT_MAX and min_head > last_head)
          st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head); // system scope, relaxed order
      }
    } else { // other warps except than warp0 handle the reduction
      // Prepare channel data buffers for each src rank
      Buffer<int4, sizeof(int4)> channel_x_buffers[kNumRanks];

      // Prepare channel lse data buffers for each src rank
      constexpr int kLSEBufArrayLength = kIsLSEReduce ? kNumRanks : 1;
      Buffer<float> channel_lse_buffers[kLSEBufArrayLength];

      // Prepare channel data buffers for each src rank for other groups
      constexpr int k2ndGroupBufArrayLength = kIs2ndGroupExists ? kNumRanks : 1;
      Buffer<int4, sizeof(int4)> channel_x_buffers_2nd[k2ndGroupBufArrayLength];

#pragma unroll
      for (int curr_rank = 0; curr_rank < kNumRanks; ++curr_rank) {
        const auto channel_rank_offset = responsible_channel * kNumRanks + curr_rank;
        const auto channel_rank_token_offset = channel_rank_offset * num_recv_buffer_tokens;

        // Get buffer ptr of the recv rank
        // and jump across the `head_idx` and `tail_idx`, loaded by warp0
        //  `head_idx`: shape=(kNumChannels, kNumRanks), dtype=int
        //  `tail_idx`: shape=(kNumChannels, kNumRanks), dtype=int
        // TODO: move this ptr out of the loop, when the non-inplace updated Buffer is supported
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels_total * sizeof(int));

        // Get `channel_x_buffers` for curr rank
        // `x_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4_comm), dtype=int, alignment=sizeof(int4)
        channel_x_buffers[curr_rank] = Buffer<int4, sizeof(int4)>(ptr, num_channel_tokens_total * hidden_int4_comm, channel_rank_token_offset * hidden_int4_comm);

        // Jump across the `src_idx_buffers`, loaded by warp0
        //  `src_idx_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens), dtype=int
        ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channel_tokens_total * sizeof(int));

        // Get `channel_lse_buffers` for curr rank if `kIsLSEReduce`
        if constexpr (kIsLSEReduce)
          // Get `channel_lse_buffers` for curr rank
          // `lse_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_heads), dtype=float
          channel_lse_buffers[curr_rank] = Buffer<float>(ptr, num_channel_tokens_total * num_heads, channel_rank_token_offset * num_heads);

        // Get `channel_x_buffers_2nd` for curr rank if `kIs2ndGroupExists`
        // `x_buffers_2nd`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4_comm), dtype=int, alignment=sizeof(int4)
        if constexpr (kIs2ndGroupExists)
          channel_x_buffers_2nd[curr_rank] = Buffer<int4, sizeof(int4)>(ptr, num_channel_tokens_total * hidden_int4_comm, channel_rank_token_offset * hidden_int4_comm);
      }

      // Get reduce tasks
      // i.e. the range of tokens [start_idx, end_idx) in `reduced_x` for the responsible channel
      // NOTES: this range is exactly the same as the one in group_cast stage
      // so as to reduce the tokens from all source ranks
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

      // Iterate over all tokens to reduce to and reduce each from all src ranks
      for (int64_t token_idx = token_start_idx + reduce_warp_id; token_idx < token_end_idx; token_idx += num_reduce_warps) { // warp-group strided
        // Read expected head for each rank
        int expected_head = -1;
        if (responsible_rank < kNumRanks) // the first `kNumRanks` lanes in each reduce warp load the expected head for each rank
          // `send_head`: shape=(num_recv_tokens, kNumRanks), dtype=int
          //  is the one initialized in group_cast stage and updated in `cached_notify_group_reduce`
          //  where send_head[token_idx, r]: the token offset of token_idx for the responsible channel
          //  if it is sent to rank r in group_cast stage
          expected_head = ld_nc_global(send_head + token_idx * kNumRanks + responsible_rank); // non-cached load

        // Wait for the expected heads of this token for each rank to be all ready
        auto start_time = clock64();
        // NOTES: here we should check `expected_head >= 0` first
        // to avoid invalid `responsible_rank` when accessing `shared_channel_tail_idx`
        while (any_in_warp(/*pred=*/expected_head >= 0 and shared_channel_tail_idx[responsible_rank] <= expected_head)) {
          // Check timeout
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf("grpcoll timeout for group reduce receivers, rank=%d, responsible_channel=%d, expect_head=%d\n", rank, responsible_channel, expected_head);
            trap();
          }
        }
        __syncwarp();

        // Get src ranks and slot indices in the recv queue of each expected head for each rank
        int num_src_ranks = 0, src_rank_idxs[kNumRanks], slot_indices[kNumRanks];
#pragma unroll
        for (int curr_rank = 0; curr_rank < kNumRanks; ++curr_rank) {
          auto expected_head_cur_rank = broadcast_in_warp(/*val=*/expected_head, /*src_lane=*/curr_rank);
          if (expected_head_cur_rank >= 0) { // valid head
            slot_indices[num_src_ranks] = expected_head_cur_rank % num_recv_buffer_tokens;
            src_rank_idxs[num_src_ranks++] = curr_rank;
          }
        }

        // Wait for all previous TMA stores to be finished
        // i.e. wait for all hidden values of last token is reduced
        // and release all TMA slots for reducing the current token
#ifndef DISABLE_SM90_FEATURES
        if (lane_id == 0)
          tma_store_wait();
        __syncwarp();
#endif

        // Reduce `reduced_lse` from `channel_lse_buffers` first
        // if `kIsLSEReduce`
        if constexpr (kIsLSEReduce) {
          for (int h = lane_id; h < num_heads; h += WARP_SIZE) {
            auto reduced_lse_ptr = reduced_lse + token_idx * num_heads + h;

            // Load all recv partial lse for current head from all src ranks
            float recv_lses[kNumRanks];
#pragma unroll
            for (int j = 0; j < num_src_ranks; ++j)
              // NOTES：for now, the channel_lse_buffers will be read repeatedly during reduing the hidden values
              // since no shared memory is used, thus here we use `__ldg` instead of original `ld_nc_global`
              recv_lses[j] = __ldg(channel_lse_buffers[src_rank_idxs[j]].buffer() + slot_indices[j] * num_heads + h);

            // Initialize the high-precision reduce buffer
            reduce_dtype_t reduced_lse_val, old_lse_val;
            if constexpr (kAccReduce) { // if in `kAccReduce` mode, initialize `reduced_lse` with the old value
              reduced_lse_val = old_lse_val = static_cast<reduce_dtype_t>(*reduced_lse_ptr);
            } else { // else, initialize `reduced_lse` with -inf
              reduced_lse_val = get_neg_inf<reduce_dtype_t>();
            }

#pragma unroll
            // Apply lse reduce for each src rank
            for (int j = 0; j < num_src_ranks; ++j)
              lse_reduce<reduce_dtype_t, float>(/*reduced_lse=*/reduced_lse_val, /*src_lse=*/recv_lses[j]);
            auto reduced_lse_val_float = static_cast<float>(reduced_lse_val);

            // Store the reduced lse to shared memory buffer temporarily
            // which will be read later to reduce the hidden values
            shared_reduced_lse_buf[reduce_warp_id][h] = reduced_lse_val_float;

            // Store the weight to rescale the old `reduced_lse` for each head
            // which will be read later to reduce the hidden values
            // if in `kAccReduce` mode
            if constexpr (kAccReduce) {
              shared_old_lse_rescale_weight_buf[reduce_warp_id][h] = get_lse_rescale_weight(/*lse_to_rescale=*/old_lse_val, /*rescaled_lse=*/reduced_lse_val);
            }

            // Store the reduced lse to `reduced_lse` as well
            // REVIEW: is it necessary to use TMA copy here to optimize ?
            *reduced_lse_ptr = reduced_lse_val_float;
          }
          __syncwarp();
        }

        // Prepare the token start ptr in the reduced buffer and the recv queue
        int4* reduced_token_ptr_int4_array[kNumDataGroups];
        Buffer<int4, sizeof(int4)>* channel_x_buffers_ptr_array[kNumDataGroups];

        // for the 1st group
        reduced_token_ptr_int4_array[0] = reduced_x_int4 + token_idx * hidden_int4;
        channel_x_buffers_ptr_array[0] = channel_x_buffers;

        // for the 2nd group if exists
        if constexpr (kIs2ndGroupExists) {
          reduced_token_ptr_int4_array[1] = reduced_x_2nd_int4 + token_idx * hidden_int4;
          channel_x_buffers_ptr_array[1] = channel_x_buffers_2nd;
        }

        // Reduce this token by all the received partial token from all src ranks
        // NOTES: we guarantee that `hidden_int4_comm` is a multiple of `WARP_SIZE`
        // so that all lanes in warp will always enter the loop together
        // to make `__syncwarp` hang-free and tma store bytes constant
#pragma unroll
        for (int g = 0; g < kNumDataGroups; ++g) {
          int4* reduced_token_ptr_int4 = reduced_token_ptr_int4_array[g];
          auto channel_x_buffers_ptr = channel_x_buffers_ptr_array[g];

#pragma unroll
          for (int i = lane_id; i < hidden_int4_comm; i += WARP_SIZE) { // warp-strided
            // Get the hidden value ptr of `int_4` to reduce to in `reduced_x`
            int4* reduce_hidval_ptr_int4 = reduced_token_ptr_int4 + i * kCommDtypePerDtype;

            // Get the optional head idx, valid and used only when `kIsLSEReduce`
            // NOTES: we guarantee that each `kCommDtypePerInt4` of elems share the same head
            const int head_idx = kIsLSEReduce ? i * kCommDtypePerInt4 / head_dim : -1;

            // Load all recv partial hidden values from all src ranks
            // REVIEW: why use a temp buffer here instead of loading and reducing in one iteration ?
            int4 recv_hidval_int4[kNumRanks];
#pragma unroll
            for (int j = 0; j < num_src_ranks; ++j)
              recv_hidval_int4[j] = ld_nc_global(channel_x_buffers_ptr[src_rank_idxs[j]].buffer() + slot_indices[j] * hidden_int4_comm + i);

            // Prepare high-precision reduce buffer for this hidden value
            reduce_dtype_t hp_hidval_reduce_buf[kCommDtypePerInt4];

            // Initialize the high-precision reduce buffer
            if constexpr (kAccReduce) { // if in `kAccReduce` mode
              // Initialize the high-precision reduce buffer
              // with the old value in `reduced_x`
              auto reduce_hidval_ptr_dtype = reinterpret_cast<const dtype_t*>(reduce_hidval_ptr_int4);
              foreach_assign<reduce_dtype_t, dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, reduce_hidval_ptr_dtype);

              // Rescale the initial old value in advance
              // if `kIsLSEReduce`
              if constexpr (kIsLSEReduce) {
                reduce_dtype_t rescale_weight = shared_old_lse_rescale_weight_buf[reduce_warp_id][head_idx];
                foreach_mul<reduce_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, rescale_weight);
              }
            } else { // not in `kAccReduce` mode
              // Zero-initialize the high-precision reduce buffer
              foreach_fill<reduce_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, 0);
            }

            // Reduce all recv partial hidden values from all src ranks
            // to the high-precision reduce buffer
            if constexpr (kIsLSEReduce) {
              // FIXME: the bank conflict is very severe here,
              // since all lanes in one warp are very likely to share the same `head_idx`
              reduce_dtype_t reduced_lse_val = shared_reduced_lse_buf[reduce_warp_id][head_idx];
              for (int j = 0; j < num_src_ranks; ++j) {
                auto jth_recv_hidval_comm_dtype = reinterpret_cast<const comm_dtype_t*>(&recv_hidval_int4[j]);
                // TODO: optimize the repeated load of each head of lse with dynamic shared memory
                // but be careful of the high occupancy of the tma buffer
                auto jth_recv_lse = __ldg(channel_lse_buffers[src_rank_idxs[j]].buffer() + slot_indices[j] * num_heads + head_idx);
                foreach_reduce_lse<reduce_dtype_t, comm_dtype_t, float, kCommDtypePerInt4>(
                    hp_hidval_reduce_buf, reduced_lse_val, jth_recv_hidval_comm_dtype, jth_recv_lse);
              }
            } else if constexpr (kReduceOp == ReduceOp::SUM || kReduceOp == ReduceOp::AVG) {
#pragma unroll
              for (int j = 0; j < num_src_ranks; ++j) {
                auto jth_recv_hidval_comm_dtype = reinterpret_cast<const comm_dtype_t*>(&recv_hidval_int4[j]);
                foreach_reduce_add<reduce_dtype_t, comm_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, jth_recv_hidval_comm_dtype);
              }

              if constexpr (kReduceOp == ReduceOp::AVG) {
                auto num_reduces = num_src_ranks;
                if constexpr (kAccReduce) // if in `kAccReduce` mode, the old value also counts
                  ++num_reduces;
                if (num_reduces > 1) // average by dividing non-trivial `num_reduces`
                  foreach_div<reduce_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, static_cast<reduce_dtype_t>(num_reduces));
              }
            }

            // Cast the high-precision reduced value back to `dtype_t`
            int4 reduced_hidval_int4[kCommDtypePerDtype];
            dtype_t* reduced_hidval_ptr_dtype = reinterpret_cast<dtype_t*>(reduced_hidval_int4);
            foreach_assign<dtype_t, reduce_dtype_t, kCommDtypePerInt4>(reduced_hidval_ptr_dtype, hp_hidval_reduce_buf);

            // Copy the reduced hidden value to `reduced_x`
#ifndef DISABLE_SM90_FEATURES
            // Wait for the previous (num_tma_stages - 1) TMA stores to be finished
            // to release at least one TMA slot for the current hidden value
            if (lane_id == 0)
              tma_store_wait<kNumTMAStages - 1>();
            __syncwarp();

            // Copy the reduced hidden value to the TMA slot for current TMA stage
            const int tma_stage_idx = (i / WARP_SIZE) % kNumTMAStages;
            auto tma_ptr_int4_cur_stage = reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * WARP_SIZE * kCommDtypePerDtype;
#pragma unroll
            for (int l = 0; l < kCommDtypePerDtype; ++l)
              tma_ptr_int4_cur_stage[lane_id * kCommDtypePerDtype + l] = reduced_hidval_int4[l];

            // Fence TMA store to wait the TMA buffer for each lane to be ready
            // before issuing the next TMA store by lane0
            // NOTES: it's issued by all lanes, compared to other TMA ops which are only issued by lane0
            tma_store_fence();
            __syncwarp();

            // Store all the reduced hidden values for all lanes from TMA slot to `reduced_x`
            if (lane_id == 0) {
              tma_store_1d(
                  /*smem_ptr=*/tma_ptr_int4_cur_stage,
                  /*gmem_ptr=*/reduce_hidval_ptr_int4,
                  /*num_bytes=*/kTMAStoreBytesPerWarp,
                  /*evict_first=*/false);
            }
            __syncwarp();
#else
            foreach_assign<int4, int4, kCommDtypePerDtype>(reduce_hidval_ptr_int4, reduced_hidval_int4);
#endif
          }
        }

        // Update channel head idx for each rank to shared memory
        // which will be read by the warp0 to update the minimum head across all reduce warps
        /** NOTE: for those `-1` entries of the original `send_head` generated in group-cast stage,
         * we've already in-place updated them in `cached_notify_group_reduce` to the valid next position `p`,
         * but encoded to `-p-1` to maintain them still negative like `-1`
         * and we can decode the correct next expect head by `-expected_head - 1`
         */
        if (responsible_rank < kNumRanks)
          shared_warp_channel_head_idx[reduce_warp_id][responsible_rank] = expected_head < 0 ? decode(expected_head) : expected_head + 1;
      }

      // Retired this warp by toggling the retire flag
      __syncwarp();
      if (lane_id == 0)
        shared_warp_retired[reduce_warp_id] = true;

      // Wait for all previous TMA stores to be finished
#ifndef DISABLE_SM90_FEATURES
      if (lane_id == 0)
        tma_store_wait();
#endif
    }
  }
}

template <typename dtype_t, typename comm_dtype_t, typename reduce_dtype_t, int kNumDataGroups, int kNumRanks, int kNumWarps, bool kAccReduce>
void launch_group_reduce(
    /* 1st group of input / output data*/
    void* reduced_x,
    float* reduced_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* reduced_x_2nd,
    const void* x_2nd,
    /* other metadata */
    int* send_head,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    const int64_t* pre_perm_idx,
    int num_tokens,
    int num_recv_tokens,
    int hidden_size,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    ReduceOp reduce_op) {
  constexpr int kNumThreads = kNumWarps * WARP_SIZE; // num threads per block
  constexpr int kMaxNumHeads = 128; // the maximum number of heads supported for lse
  constexpr int kWarpCopyUnrollStages = 4; // warp-copy unroll stages
  constexpr int kCommDtypePerDtype = sizeof(dtype_t) / sizeof(comm_dtype_t); // num comm_dtype_t elems per dtype_t
  GRPCOLL_STATIC_ASSERT(kCommDtypePerDtype == 1 || kCommDtypePerDtype == 2 || kCommDtypePerDtype == 4, "Invalid (dtype_t, comm_dtype_t)");
  constexpr int kNumTMAStages = 8 / kCommDtypePerDtype; // num TMA stages
  constexpr int kNumTMABytesPerWarp = 4096; // num bytes of TMA transfer per warp

#ifndef DISABLE_SM90_FEATURES
  GRPCOLL_STATIC_ASSERT(kNumTMAStages * WARP_SIZE * kCommDtypePerDtype * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid TMA buffer count"); // TMA buffer per warp
  constexpr int smem_size = kNumTMABytesPerWarp * kNumWarps; // shared memory size = num bytes of TMA transfer per block
#endif

  // NOTES: when `kReduceOp != ReduceOp::LSE`,
  // num_heads should be 0 to let `lse_buffers` empty
  if (reduce_op != ReduceOp::LSE)
    GRPCOLL_HOST_ASSERT(num_heads == 0);
  else
    GRPCOLL_HOST_ASSERT(num_heads <= kMaxNumHeads);

  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 && kNumDataGroups <= 2, "Invalid kNumDataGroups");

#define GROUP_REDUCE_LAUNCH_CASE(reduce_op)        \
  {                                                \
    auto kernel = group_reduce_kernel<             \
        dtype_t,                                   \
        comm_dtype_t,                              \
        reduce_dtype_t,                            \
        reduce_op,                                 \
        kNumDataGroups,                            \
        kNumRanks,                                 \
        kNumThreads,                               \
        kWarpCopyUnrollStages,                     \
        kNumTMAStages,                             \
        kNumTMABytesPerWarp,                       \
        kMaxNumHeads,                              \
        kAccReduce>;                               \
    SET_SHARED_MEMORY_FOR_TMA(kernel);             \
    LAUNCH_KERNEL(                                 \
        &cfg,                                      \
        kernel,                                    \
        reinterpret_cast<dtype_t*>(reduced_x),     \
        reduced_lse,                               \
        reinterpret_cast<const dtype_t*>(x),       \
        lse,                                       \
        reinterpret_cast<dtype_t*>(reduced_x_2nd), \
        reinterpret_cast<const dtype_t*>(x_2nd),   \
        send_head,                                 \
        src_idx,                                   \
        rank_prefix_matrix,                        \
        channel_prefix_matrix,                     \
        pre_perm_idx,                              \
        num_tokens,                                \
        num_recv_tokens,                           \
        hidden_size,                               \
        num_heads,                                 \
        buffer_ptrs,                               \
        rank,                                      \
        num_max_send_tokens,                       \
        num_recv_buffer_tokens);                   \
  }                                                \
  break

  // Even-numbered SMs for sending, odd-numbered SMs for receiving
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_REDUCE_OPS(GROUP_REDUCE_LAUNCH_CASE);

#undef GROUP_REDUCE_REDUCE_OP_LAUNCH_CASE
#undef GROUP_REDUCE_LAUNCH_CASE
}

void group_reduce(
    /* 1st group of input / output data*/
    void* reduced_x,
    float* reduced_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* reduced_x_2nd,
    const void* x_2nd,
    /* other metadata */
    int* send_head,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    const int64_t* pre_perm_idx,
    int num_tokens,
    int num_recv_tokens,
    int hidden_size,
    int num_heads,
    int num_groups,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    bool acc_reduce,
    cudaDataType_t dtype,
    cudaDataType_t comm_dtype,
    ReduceOp reduce_op) {
#define LAUNCH_INTRANODE_GROUP_REDUCE(dtype, comm_dtype, reduce_dtype, num_groups, num_ranks, num_warps, acc_reduce) \
  {                                                                                                                  \
    launch_group_reduce<dtype, comm_dtype, reduce_dtype, num_groups, num_ranks, num_warps, acc_reduce>(              \
        reduced_x,                                                                                                   \
        reduced_lse,                                                                                                 \
        x,                                                                                                           \
        lse,                                                                                                         \
        reduced_x_2nd,                                                                                               \
        x_2nd,                                                                                                       \
        send_head,                                                                                                   \
        src_idx,                                                                                                     \
        rank_prefix_matrix,                                                                                          \
        channel_prefix_matrix,                                                                                       \
        pre_perm_idx,                                                                                                \
        num_tokens,                                                                                                  \
        num_recv_tokens,                                                                                             \
        hidden_size,                                                                                                 \
        num_heads,                                                                                                   \
        buffer_ptrs,                                                                                                 \
        rank,                                                                                                        \
        stream,                                                                                                      \
        num_sms,                                                                                                     \
        num_max_send_tokens,                                                                                         \
        num_recv_buffer_tokens,                                                                                      \
        reduce_op);                                                                                                  \
  }                                                                                                                  \
  break

#define GROUP_REDUCE_DTYPE_LAUNCH_CASE(...)                                                \
  {                                                                                        \
    SWITCH_DTYPES_COMM_DTYPES_REDUCE_DTYPES(LAUNCH_INTRANODE_GROUP_REDUCE, ##__VA_ARGS__); \
  }                                                                                        \
  break

#define GROUP_REDUCE_DATA_GROUPS_LAUNCH_CASE(...)                        \
  {                                                                      \
    SWITCH_DATA_GROUPS_2(GROUP_REDUCE_DTYPE_LAUNCH_CASE, ##__VA_ARGS__); \
  }

#define GROUP_REDUCE_ACC_REDUCE_LAUNCH_CASE(num_ranks, num_warps)        \
  {                                                                      \
    if (acc_reduce) {                                                    \
      GROUP_REDUCE_DATA_GROUPS_LAUNCH_CASE(num_ranks, num_warps, true);  \
    } else {                                                             \
      GROUP_REDUCE_DATA_GROUPS_LAUNCH_CASE(num_ranks, num_warps, false); \
    }                                                                    \
  }                                                                      \
  break

  SWITCH_RANKS_WITH_WARPS(GROUP_REDUCE_ACC_REDUCE_LAUNCH_CASE);

#undef GROUP_REDUCE_DTYPE_LAUNCH_CASE
#undef GROUP_REDUCE_DATA_GROUPS_LAUNCH_CASE
#undef GROUP_REDUCE_ACC_REDUCE_LAUNCH_CASE
#undef LAUNCH_INTRANODE_GROUP_REDUCE
}

} // namespace intranode

} // namespace magi_attn_comm::grpcoll
