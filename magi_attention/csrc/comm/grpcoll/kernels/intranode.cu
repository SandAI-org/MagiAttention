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
#include "utils.cuh"

namespace magi_attn_comm::grpcoll {

namespace intranode {

template <int kNumRanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    int num_channels,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto lane_id = thread_id % WARP_SIZE, warp_id = thread_id / WARP_SIZE, num_warps = num_threads / WARP_SIZE;

  if (sm_id == 0) {
    // Barrier first
    barrier_block<kNumRanks, true>(barrier_signal_ptrs, rank);

    int *per_rank_buffer, *per_expert_buffer;
    if (thread_id < kNumRanks) {
      // `per_rank_buffer` has shape (kNumRanks, kNumRanks)
      // starting at the beginning of the buffer ptr of each rank
      per_rank_buffer = static_cast<int*>(buffer_ptrs[thread_id]);
      per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
    }

    // After this loop:
    //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
    //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
    int num_experts_per_rank = num_experts / kNumRanks;
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i)
        per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
#pragma unroll
      for (int i = 0; i < num_experts_per_rank; ++i)
        per_expert_buffer[rank * num_experts_per_rank + i] = num_tokens_per_expert[thread_id * num_experts_per_rank + i];
    }

    // Wait for all ranks to be finished
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);

    // Sum per-rank counts and return to CPU
    // Also pre-compute the prefix sum for data sending
    auto local_per_rank_buffer = static_cast<int*>(buffer_ptrs[rank]);
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 1; i < kNumRanks; ++i)
        local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      if (thread_id == rank)
        *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
    }

    // Sum per-experts counts and return to CPU
    auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < num_experts_per_rank) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i)
        sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }
    __syncthreads();

// Copy rank size prefix matrix to another tensor
#pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
      rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

// Extra memset for later communication queue
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads)
      local_per_expert_buffer[i] = 0;

    // Barrier
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
  } else {
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

void notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks) \
  LAUNCH_KERNEL(                           \
      &cfg,                                \
      notify_dispatch<ranks>,              \
      num_tokens_per_rank,                 \
      moe_recv_counter_mapped,             \
      num_tokens_per_expert,               \
      moe_recv_expert_counter_mapped,      \
      num_experts,                         \
      num_tokens,                          \
      num_channels,                        \
      is_token_in_rank,                    \
      channel_prefix_matrix,               \
      rank_prefix_matrix_copy,             \
      num_memset_int,                      \
      expert_alignment,                    \
      buffer_ptrs,                         \
      barrier_signal_ptrs,                 \
      rank);                               \
  break

  constexpr int kNumThreads = 128;
  GRPCOLL_HOST_ASSERT(num_experts % num_ranks == 0);
  GRPCOLL_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
  SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank) {
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

void cached_notify_dispatch(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                                 \
  LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, barrier_signal_ptrs, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 128, stream);
  SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(/*max_threads_per_block=*/kNumThreads, /*min_blocks_per_sm=*/1) dispatch(
    int4* recv_x,
    float* recv_x_scales,
    int* recv_src_idx,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_channel_offset,
    int* send_head,
    const int64_t* post_perm_idx,
    const int4* x,
    const float* x_scales,
    const int64_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens,
    int num_worst_tokens,
    int hidden_int4,
    int num_topk,
    int num_experts,
    int num_scales,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  // Get thread Info
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const bool is_sender = sm_id % 2 == 0; // even-numbered SMs are senders
  GRPCOLL_DEVICE_ASSERT(num_sms % 2 == 0);

  // Get Rank Info
  const auto num_threads_per_rank = kNumThreads / kNumRanks; // Several warps are response for a single rank
  const auto responsible_rank = thread_id / num_threads_per_rank;
  const auto send_rank = is_sender ? rank : responsible_rank, recv_rank = is_sender ? responsible_rank : rank;

  // Get Channel Info
  const auto num_channels = num_sms / 2, responsible_channel = sm_id / 2;
  const auto num_channels_total = num_channels * kNumRanks, channel_rank_offset = responsible_channel * kNumRanks + send_rank; // each rank has SM/2 channels
  const auto num_channel_tokens_total = num_channels_total * num_recv_buffer_tokens, channel_rank_token_offset = channel_rank_offset * num_recv_buffer_tokens;
  const auto responsible_rank_channel = responsible_rank * num_channels + responsible_channel;

  // Get expert Info (TODO: remove the logic for experts)
  int num_experts_per_rank = num_experts / kNumRanks;
  GRPCOLL_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
  GRPCOLL_DEVICE_ASSERT(num_topk <= WARP_SIZE);
  GRPCOLL_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  GRPCOLL_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

  // Get metadata ptr after the rank prefix matrix
  // `rank_prefix_matrix`: shape=(kNumRanks, kNumRanks), dtype=int
  auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[recv_rank]) + kNumRanks * kNumRanks * sizeof(int));

  // Get channel metadata buffers
  // Senders are responsible for tails, and receivers are responsible for heads
  // And the metadata of any pair of (sender, receiver) is all stored on the receiver side
  //  `start_offset`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `end_offset`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `head_idx`: shape=(kNumChannels, kNumRanks), dtype=int
  //  `tail_idx`: shape=(kNumChannels, kNumRanks), dtype=int
  auto channel_start_offset = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_end_offset = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_head_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);
  auto channel_tail_idx = Buffer<int>(ptr, /*num_elems=*/num_channels_total, /*elem_offset=*/channel_rank_offset);

  // Get channel data buffers
  //  `x_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, hidden_int4), dtype=int4
  //  `src_idx_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens), dtype=int
  //  `topk_idx_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_topk), dtype=int64_t
  //  `topk_weights_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_topk), dtype=float
  //  `x_scales_buffers`: shape=(kNumChannels, kNumRanks, num_recv_buffer_tokens, num_scales), dtype=float
  auto channel_x_buffers = Buffer<int4>(ptr, /*num_elems=*/num_channel_tokens_total * hidden_int4, /*elem_offset=*/channel_rank_token_offset * hidden_int4);
  auto channel_src_idx_buffers = Buffer<int>(ptr, /*num_elems=*/num_channel_tokens_total, /*elem_offset=*/channel_rank_token_offset);
  auto channel_topk_idx_buffers = Buffer<int64_t>(ptr, /*num_elems=*/num_channel_tokens_total * num_topk, /*elem_offset=*/channel_rank_token_offset * num_topk);
  auto channel_topk_weights_buffers = Buffer<float>(ptr, /*num_elems=*/num_channel_tokens_total * num_topk, /*elem_offset=*/channel_rank_token_offset * num_topk);
  auto channel_x_scales_buffers = Buffer<float>(ptr, /*num_elems=*/num_channel_tokens_total * num_scales, /*elem_offset=*/channel_rank_token_offset * num_scales);

  // Get copy info
  const auto warp_copy_unroll_stages = 5; // TODO: test other stages and make it configurable
#ifndef DISABLE_SM90_FEATURES
  // Get TMA copy info
  const auto num_tma_stages = 2; // TODO: test other stages and make it configurable
  GRPCOLL_DEVICE_ASSERT(hidden_int4 % num_tma_stages == 0);
  auto hidden_int4_per_stage = hidden_int4 / num_tma_stages;

  auto hidden_bytes_per_stage = hidden_int4_per_stage * static_cast<int>(sizeof(int4));
  GRPCOLL_DEVICE_ASSERT(hidden_bytes_per_stage + sizeof(uint64_t) <= kNumTMABytesPerWarp); // tma buffer + mbarrier

  // Prepare TMA buffer in shared memory for this warp
  extern __shared__ __align__(1024) uint8_t smem_buffer[]; // Review: why aligned to 1024 bytes ?
  auto tma_buffer = smem_buffer + warp_id * kNumTMABytesPerWarp;

  // Init the TMA phase and mbarrier for this warp
  uint32_t tma_phase = 0;
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
    const auto num_send_warps = kNumThreads / WARP_SIZE;
    const auto num_send_warps_per_rank = num_send_warps / kNumRanks;
    const auto send_thread_id = thread_id;
    const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / WARP_SIZE;
    const auto max_num_used_slots_in_queue = num_recv_buffer_tokens - num_max_send_tokens;

    GRPCOLL_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
    GRPCOLL_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

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

    // Get send tasks, i.e. the range of tokens [start_idx, end_idx) for this channel
    // NOTES: this range does not distiguish the destination rank,
    // thus every thread in the block will get the same range
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
        int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer()); // volatile

        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        if (num_used_slots <= max_num_used_slots_in_queue)
          break; // the empty slots in recv queue is enough for this send chunk size

        // Check timeout
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("grpcoll timeout for dispatch senders, rank=%d, responsible_channel=%d\n", rank, responsible_channel);
          trap();
        }
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
        if (not is_token_in_responsible_rank) {
          token_idx++;
          continue;
        }

        // Get an empty slot in recv queue
        int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;

        // Pick (round-robin) one warp for the responsible rank to send this token
        if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
          // Warp-copy this token from send buffer to the recv queue
          // Review: why not use TMA copy here ?
          auto token_ptr_in_queue = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4; // token idx in the recv queue
          auto token_ptr_in_x = x + token_idx * hidden_int4; // global token idx in the send buffer
          UNROLLED_WARP_COPY(
              /*UNROLL_FACTOR=*/warp_copy_unroll_stages,
              /*LANE_ID=*/lane_id,
              /*N=*/hidden_int4,
              /*DST=*/token_ptr_in_queue,
              /*SRC=*/token_ptr_in_x,
              /*LD_FUNC=*/__ldg, // read-only load
              /*ST_FUNC=*/st_na_global // non-cached store
          );

          // Store channel src idx by lane0
          // which will be used to fill `recv_src_idx` by the receiver
          if (lane_id == 0)
            channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

          // Copy `topk_idx` and `topk_weights` to `channel_topk_idx` and `channel_topk_weights` respectively
          // by the first `num_topk` lanes in this warp
          // which is used to fill `recv_topk_idx` and `recv_topk_weights` by the receiver
          // TODO: remove this unused copy
          if (lane_id < num_topk) {
            // Top-k index
            int recv_expert_begin = responsible_rank * num_experts_per_rank, recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
            auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane_id);
            idx_value = (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
            channel_topk_idx_buffers[dst_slot_idx * num_topk + lane_id] = idx_value;

            // Top-k weights
            auto weight_value = __ldg(topk_weights + token_idx * num_topk + lane_id);
            weight_value = (idx_value >= 0) ? weight_value : 0.0f;
            channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = weight_value;
          }

#pragma unroll
          // Warp-strided copy `x_scales` to `channel_x_scales` by this warp
          // which is used to fill `recv_x_scales` by the receiver
          // TODO: remove this unused copy
          for (int i = lane_id; i < num_scales; i += WARP_SIZE) {
            auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
            channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + offset);
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
    const auto num_recv_warps = kNumThreads / WARP_SIZE;
    const auto num_recv_warps_per_rank = num_recv_warps / kNumRanks;
    const auto num_recv_threads_per_rank = num_recv_warps_per_rank * WARP_SIZE;
    const auto recv_thread_id = thread_id;
    const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
    const auto recv_warp_id_in_rank = recv_thread_id_in_rank / WARP_SIZE;
    GRPCOLL_DEVICE_ASSERT(kNumRanks <= WARP_SIZE);
    GRPCOLL_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

    // Get global rank offset for the responsible rank from the rank prefix matrix
    auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[recv_rank]);
    int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + recv_rank] : 0;

    // Load non-empty channel start/end offset stored by the sender by lane0 in each warp
    int total_offset, num_tokens_to_recv;
    while (lane_id == 0 and (total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0) // volatile
      ;
    while (lane_id == 0 and (num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0) // volatile
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
    total_offset = broadcast_warp(total_offset);
    // Here, total_offset = rank_offset + channel_start_offset
    // = the global token offset in the send buffer of the start token in the channel
    total_offset += rank_offset;

    // Broadcast num_tokens_to_recv to other lanes
    num_tokens_to_recv = broadcast_warp(num_tokens_to_recv);

    // Shared tail indices for each rank
    // NOTES: unlike the sender, the receiver must ensure that
    // the tail index hold by all warps for the responsible rank are the same
    // thus we cannot use `broadcast_warp` to sync the tail idx
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
          printf(
              "grpcoll timeout for dispatch receivers, rank=%d, responsible_channel=%d, tokens_remained_to_recv=%d\n", rank, responsible_channel, num_tokens_to_recv);
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
        // Determine the final destination token idx in the recv buffer
        auto token_idx_in_recv_x = static_cast<int64_t>(total_offset + chunk_idx); // original token idx in recv buffer in rank order
        token_idx_in_recv_x = post_perm_idx == nullptr ? token_idx_in_recv_x : post_perm_idx[token_idx_in_recv_x];

        // Get the token ptr in the recv buffer
        auto token_ptr_in_recv_x_int4 = recv_x + token_idx_in_recv_x * hidden_int4;

        // Get the token ptr in the recv queue
        int token_idx_in_queue = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        auto token_ptr_in_queue_int4 = channel_x_buffers.buffer() + token_idx_in_queue * hidden_int4;

        // Copy this token from recv queue to recv buffer
#ifndef DISABLE_SM90_FEATURES
#pragma unroll
        // TMA-copy
        for (int i = 0; i < num_tma_stages; ++i) // multiple TMA stages
          if (lane_id == 0) { // the lane0 in this warp issues the TMA
            // Wait for all previous TMA stores to be finished
            // Review: can we use double buffer for all stages ?
            tma_store_wait();

            // Load the token from recv queue to shared memory
            tma_load_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/token_ptr_in_queue_int4 + i * hidden_int4_per_stage,
                /*mbar_ptr=*/tma_mbarrier,
                /*num_bytes=*/hidden_bytes_per_stage,
                /*evict_first=*/true // evict the read-once token in recv queue first from L2 cache
            );

            // Barrier the last load above to be finished before the next store below
            // NOTES: TMA phase will be inplace updated
            mbarrier_arrive_and_expect_tx(/*mbar_ptr=*/tma_mbarrier, /*num_bytes=*/hidden_bytes_per_stage);
            mbarrier_wait(/*mbar_ptr=*/tma_mbarrier, /*phase=*/tma_phase, /*num_tma_stages=*/num_tma_stages);

            // Store the token from shared memory to recv buffer
            tma_store_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/token_ptr_in_recv_x_int4 + i * hidden_int4_per_stage,
                /*num_bytes=*/hidden_bytes_per_stage,
                /*evict_first=*/false);
          }
        __syncwarp();
#else
        // Warp-copy
        UNROLLED_WARP_COPY(
            /*UNROLL_FACTOR=*/warp_copy_unroll_stages,
            /*LANE_ID=*/lane_id,
            /*N=*/hidden_int4,
            /*DST=*/
            token_ptr_in_recv_x_int4
                /*SRC=*/token_ptr_in_queue_int4,
            /*LD_FUNC=*/ld_nc_global,
            /*ST_FUNC=*/st_na_global);
#endif
      }

#pragma unroll 4
      // Thead-copy `channel_src_idx` stored by the sender to `recv_src_idx`
      for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
           chunk_idx += num_recv_threads_per_rank) // warp-group strided
        recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] = ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

#pragma unroll 4
      // Thread-copy `channel_topk_idx` and `channel_topk_weights` stored by the sender
      // to `recv_topk_idx` and `recv_topk_weights` respectively
      // TODO: remove this unused copy
      for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += num_recv_threads_per_rank) { // warp-group strided
        int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
        int token_idx_in_queue = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
        auto buffer_idx = token_idx_in_queue * num_topk + token_topk_idx;
        recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
        recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
      }

#pragma unroll 4
      // Thread-copy `channel_x_scales` stored by the sender to `recv_x_scales`
      // TODO: remove this unused copy
      for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += num_recv_threads_per_rank) { // warp-group strided
        int chunk_idx = i / num_scales, scales_idx = i % num_scales;
        int token_idx_in_queue = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
            ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_queue * num_scales + scales_idx);
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

  // Post-process for `num_worst_tokens` mode
  if (num_worst_tokens > 0) {
    auto rank_prefix_matrix = static_cast<int*>(buffer_ptrs[rank]);
    const auto num_recv_tokens = rank_prefix_matrix[(kNumRanks - 1) * kNumRanks + rank];
    const auto clean_start = num_recv_tokens * num_topk + sm_id * kNumThreads;
    const auto clean_end = num_worst_tokens * num_topk;
    const auto clean_stride = num_sms * kNumThreads;
#pragma unroll
    // Clean unused `recv_topk_idx` as -1
    for (int i = clean_start + thread_id; i < clean_end; i += clean_stride)
      recv_topk_idx[i] = -1;
  }
}

void dispatch(
    void* recv_x,
    float* recv_x_scales,
    int* recv_src_idx,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_channel_offset,
    int* send_head,
    const int64_t* post_perm_idx,
    const void* x,
    const float* x_scales,
    const int64_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens,
    int num_worst_tokens,
    int hidden_int4,
    int num_topk,
    int num_experts,
    int num_scales,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens) {
  constexpr int kNumThreads = 768; // block size
  constexpr int kNumWarps = kNumThreads / WARP_SIZE; // num warps per block
  constexpr int kNumTMABytesPerWarp = 8192; // num bytes of tma transfer per warp
#ifndef DISABLE_SM90_FEATURES
  constexpr int smem_size = kNumTMABytesPerWarp * kNumWarps; // shared memory size = num bytes of tma transfer per block
#endif

  // Make sure never OOB
  GRPCOLL_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());

#define DISPATCH_LAUNCH_CASE(ranks)                                  \
  {                                                                  \
    auto kernel = dispatch<ranks, kNumThreads, kNumTMABytesPerWarp>; \
    SET_SHARED_MEMORY_FOR_TMA(kernel);                               \
    LAUNCH_KERNEL(                                                   \
        &cfg,                                                        \
        kernel,                                                      \
        reinterpret_cast<int4*>(recv_x),                             \
        recv_x_scales,                                               \
        recv_src_idx,                                                \
        recv_topk_idx,                                               \
        recv_topk_weights,                                           \
        recv_channel_offset,                                         \
        send_head,                                                   \
        post_perm_idx,                                               \
        reinterpret_cast<const int4*>(x),                            \
        x_scales,                                                    \
        topk_idx,                                                    \
        topk_weights,                                                \
        is_token_in_rank,                                            \
        channel_prefix_matrix,                                       \
        num_tokens,                                                  \
        num_worst_tokens,                                            \
        hidden_int4,                                                 \
        num_topk,                                                    \
        num_experts,                                                 \
        num_scales,                                                  \
        scale_token_stride,                                          \
        scale_hidden_stride,                                         \
        buffer_ptrs,                                                 \
        rank,                                                        \
        num_max_send_tokens,                                         \
        num_recv_buffer_tokens);                                     \
  }                                                                  \
  break

  // Even-numbered SMs for sending, odd-numbered SMs for receiving
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(
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

    // NOTES: `1 << 25` is a heuristic large number
    int last_head = 1 << 25;
#pragma unroll
    for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= WARP_SIZE) {
      int token_idx = token_idx_tail - lane_id, expected_head = 0;
      auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
      for (int i = 0; i < min(WARP_SIZE, token_idx_tail - token_start_idx + 1); ++i) {
        const int head = broadcast_warp(/*val=*/current_head, /*src_lane=*/i);
        if (head < 0) {
          if (lane_id == i)
            expected_head = -last_head - 1;
        } else {
          last_head = head;
        }
      }
      if (current_head < 0 and token_idx >= token_start_idx)
        send_head[token_idx * kNumRanks + rank_id] = expected_head;
    }
  }
}

void cached_notify_combine(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)                                                                                                                   \
  LAUNCH_KERNEL(&cfg, cached_notify_combine<ranks>, buffer_ptrs, send_head, num_channels, num_recv_tokens, num_memset_int, barrier_signal_ptrs, rank); \
  break

  const int num_threads = std::max(128, WARP_SIZE * num_ranks);
  GRPCOLL_HOST_ASSERT(num_ranks <= num_threads);
  GRPCOLL_HOST_ASSERT(num_threads <= 1024);
  GRPCOLL_HOST_ASSERT(1 + num_channels <= num_channels * 2);
  SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
  SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads, int kNumTMABytesPerWarp>
__global__ void __launch_bounds__(/*max_threads_per_block=*/kNumThreads, /*min_blocks_per_sm=*/1) combine(
    dtype_t* recv_x,
    float* recv_topk_weights,
    const dtype_t* x,
    const float* topk_weights,
    const dtype_t* bias_0,
    const dtype_t* bias_1,
    const int64_t* pre_perm_idx,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
    int hidden,
    int num_topk,
    void** buffer_ptrs,
    int rank,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    // TODO: make acc_reduce a template parameter
    bool acc_reduce) {
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto sm_id = static_cast<int>(blockIdx.x), lane_id = get_lane_id();
  const auto num_channels = num_sms / 2;
  const bool is_sender = sm_id % 2 == 0;
  const int responsible_channel = sm_id / 2;
  GRPCOLL_DEVICE_ASSERT(num_topk <= WARP_SIZE);

  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
  auto x_int4 = reinterpret_cast<const int4*>(x);
  auto bias_0_int4 = reinterpret_cast<const int4*>(bias_0);
  auto bias_1_int4 = reinterpret_cast<const int4*>(bias_1);
  auto recv_int4 = reinterpret_cast<int4*>(recv_x);

  // TMA stuffs
#ifndef DISABLE_SM90_FEATURES
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  auto tma_buffer = smem_buffer + (thread_id / WARP_SIZE) * kNumTMABytesPerWarp;
#endif

  if (is_sender) {
    // Workers for sending
    // Several warps are responsible for a single rank
    constexpr int num_send_warps_per_rank = (kNumThreads / WARP_SIZE) / kNumRanks;
    constexpr int num_send_warps = num_send_warps_per_rank * kNumRanks;
    const auto num_threads_per_rank = num_send_warps_per_rank * WARP_SIZE;
    const auto send_thread_id = thread_id;
    const auto send_warp_id = send_thread_id / WARP_SIZE;
    const auto send_rank_id = (responsible_channel + send_warp_id) % kNumRanks;
    const auto send_warp_id_in_rank = send_warp_id / kNumRanks;
    GRPCOLL_STATIC_ASSERT(num_send_warps * WARP_SIZE == kNumThreads, "Invalid warp count");

    // Calculate pointers by the specific layout
    auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[send_rank_id]));
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + rank;

    // Channel meta data
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers = Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_weights_buffers =
        Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);

    // Get tasks
    // NOTES: `channel_offset` is already shifted
    int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
    int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
    int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
    int num_channel_tokens =
        (responsible_channel == num_channels - 1 ? num_rank_tokens : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) - channel_offset;
    int token_start_idx = rank_offset + channel_offset, token_end_idx = rank_offset + channel_offset + num_channel_tokens;

    // Iterate over all tokens and send by chunks
    int current_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Check destination queue emptiness, or wait a buffer to be released (rare cases)
      auto start_time = clock64();
      int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
      while (lane_id == 0) {
        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens)
          break;

        // Rare cases to loop again
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("grpcoll timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
          trap();
        }
      }
      __syncwarp();

// Send by chunk
#pragma unroll
      for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
        // Get an empty slot
        int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

        // Copy data
        auto token_idx_in_x = static_cast<int64_t>(token_idx + i);
        token_idx_in_x = pre_perm_idx == nullptr ? token_idx_in_x : pre_perm_idx[token_idx_in_x];
        auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x = x_int4 + token_idx_in_x * hidden_int4;
        UNROLLED_WARP_COPY(4, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

        // Send source index
        if (lane_id == 0)
          channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

        // Send `topk_weights`
        if (num_topk > 0 and lane_id < num_topk)
          channel_topk_weights_buffers[dst_slot_idx * num_topk + lane_id] = __ldg(topk_weights + (token_idx + i) * num_topk + lane_id);
      }
      token_idx += num_round_tokens;
      current_channel_tail_idx += num_round_tokens;

      // Move tail index
      sync_warp_group(/*group_flag=*/send_rank_id, /*group_size=*/num_threads_per_rank);
      if (lane_id == 0 and send_warp_id_in_rank == 0)
        st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
    }
  } else {
    // Workers for receiving
    // One warp for moving the queue head, others for reduction
    constexpr int num_recv_warps = kNumThreads / WARP_SIZE;
    const auto recv_warp_id = thread_id / WARP_SIZE;
    GRPCOLL_DEVICE_ASSERT(kNumRanks <= WARP_SIZE and kNumThreads > WARP_SIZE);
    GRPCOLL_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % WARP_SIZE == 0);

    // Shared head, tail and retired flags for receiver warps
    __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
    __shared__ volatile int channel_tail_idx[kNumRanks];
    __shared__ volatile bool warp_retired[num_recv_warps];
    if (thread_id < num_recv_warps)
      warp_retired[thread_id] = false;
    if (lane_id < kNumRanks)
      warp_channel_head_idx[recv_warp_id][lane_id] = 0;
    if (thread_id < kNumRanks)
      channel_tail_idx[thread_id] = 0;
    asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

    if (thread_id < WARP_SIZE) {
      int* channel_head_idx_ptr = static_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + lane_id;
      int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

      // Queue head updater
      int last_head = 0;
      while (lane_id < kNumRanks) {
        // Check retired
        bool retired = true;
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i)
          retired = retired and warp_retired[i];
        if (retired)
          break;

        // Update queue tail
        channel_tail_idx[lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

        // Update minimum head
        int min_head = std::numeric_limits<int>::max();
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i)
          if (not warp_retired[i])
            min_head = min(min_head, warp_channel_head_idx[i][lane_id]);
        if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
          st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
      }
    } else {
      // Receivers
      // Channel metadata
      // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
      Buffer<int4> channel_x_buffers[kNumRanks];
      Buffer<float> channel_topk_weights_buffers[kNumRanks];

// Calculate pointers by the specific layout
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) {
        auto channel_rank_offset = responsible_channel * kNumRanks + i;
        auto num_channels_total = num_channels * kNumRanks;
        // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        auto ptr = reinterpret_cast<void*>(static_cast<int8_t*>(buffer_ptrs[rank]) + 2 * num_channels * kNumRanks * sizeof(int));

        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        channel_x_buffers[i] = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4, channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        ptr = reinterpret_cast<void*>(static_cast<int8_t*>(ptr) + num_channels_total * num_recv_buffer_tokens * sizeof(int));

        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        channel_topk_weights_buffers[i] =
            Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk, channel_rank_offset * num_recv_buffer_tokens * num_topk);
      }

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

      // Iterate over all tokens and combine
      for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx; token_idx += num_recv_warps - 1) {
        // Read expected head
        int expected_head = -1;
        if (lane_id < kNumRanks)
          expected_head = ld_nc_global(send_head + token_idx * kNumRanks + lane_id);

        auto start_time = clock64();
        while (__any_sync(0xffffffff, channel_tail_idx[lane_id] <= expected_head and expected_head >= 0)) {
          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf("grpcoll timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n", rank, responsible_channel, expected_head);
            trap();
          }
        }
        __syncwarp();

        // Broadcast current heads
        int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i) {
          auto expected_head_i = broadcast_warp(/*val=*/expected_head, /*src_lane=*/i);
          if (expected_head_i >= 0) {
            slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
            topk_ranks[num_topk_ranks++] = i;
          }
        }

        // Wait shared memory release
#ifndef DISABLE_SM90_FEATURES
        if (lane_id == 0)
          tma_store_wait();
        __syncwarp();
#endif

        // Reduce data with pipeline
        constexpr int kNumStages = 8;
        GRPCOLL_STATIC_ASSERT(kNumStages * WARP_SIZE * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid count");

#pragma unroll
        for (int i = lane_id; i < hidden_int4; i += WARP_SIZE) {
          auto dst_buf_ptr_int4 = recv_int4 + token_idx * hidden_int4 + i;

          // Read bias
          // TODO: make it as a template
          int4 bias_0_value_int4 = bias_0_int4 != nullptr ? __ldg(bias_0_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);
          int4 bias_1_value_int4 = bias_1_int4 != nullptr ? __ldg(bias_1_int4 + token_idx * hidden_int4 + i) : make_int4(0, 0, 0, 0);

          // Read buffers
          int4 recv_value_int4[kNumRanks];

#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j)
            recv_value_int4[j] = ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

          // Reduce bias
          float values[kDtypePerInt4];
          auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
          auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);

#pragma unroll
          for (int j = 0; j < kDtypePerInt4; ++j)
            values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);

          // Read acc out buffers if needed
          const dtype_t* acc_buf_dtypes = nullptr;
          if (acc_reduce) {
            acc_buf_dtypes = reinterpret_cast<const dtype_t*>(dst_buf_ptr_int4);
          }

// Reduce all-to-all results
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) {
            auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);

#pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++k)
              values[k] += static_cast<float>(recv_value_dtypes[k]);
          }
          if (acc_reduce) {
#pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++k)
              values[k] += static_cast<float>(acc_buf_dtypes[k]);
          }

          // Cast back to `dtype_t`
          int4 out_int4;
          auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
          for (int j = 0; j < kDtypePerInt4; ++j)
            out_dtypes[j] = static_cast<dtype_t>(values[j]);

#ifndef DISABLE_SM90_FEATURES
          // Wait TMA arrival
          if (lane_id == 0)
            tma_store_wait<kNumStages - 1>();
          __syncwarp();

          // Write into TMA buffer
          auto tma_stage_idx = (i / WARP_SIZE) % kNumStages;
          reinterpret_cast<int4*>(tma_buffer)[tma_stage_idx * WARP_SIZE + lane_id] = out_int4;

          // Issue TMA
          tma_store_fence();
          __syncwarp();
          if (lane_id == 0) {
            auto tma_bytes = min(WARP_SIZE, hidden_int4 - i) * static_cast<int>(sizeof(int4));
            tma_store_1d(reinterpret_cast<int4*>(tma_buffer) + tma_stage_idx * WARP_SIZE, dst_buf_ptr_int4, tma_bytes, false);
          }
          __syncwarp();
#else
          recv_int4[token_idx * hidden_int4 + i] = out_int4;
#endif
        }

        // Reduce `topk_weights`
        if (lane_id < num_topk) {
          float value = 0;
#pragma unroll
          for (int i = 0; i < num_topk_ranks; ++i)
            value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk + lane_id);
          recv_topk_weights[token_idx * num_topk + lane_id] = value;
        }

        // Update head
        if (lane_id < kNumRanks)
          warp_channel_head_idx[recv_warp_id][lane_id] = (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
      }

      // Retired
      __syncwarp();
      if (lane_id == 0)
        warp_retired[recv_warp_id] = true;

      // Make TMA store visible to the next kernel
#ifndef DISABLE_SM90_FEATURES
      if (lane_id == 0)
        tma_store_wait();
#endif
    }
  }
}

void combine(
    cudaDataType_t type,
    void* recv_x,
    float* recv_topk_weights,
    const void* x,
    const float* topk_weights,
    const void* bias_0,
    const void* bias_1,
    const int64_t* pre_perm_idx,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
    int hidden,
    int num_topk,
    void** buffer_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    bool acc_reduce) {
  constexpr int kNumThreads = 768;
  constexpr int kNumTMABytesPerWarp = 4096;
#ifndef DISABLE_SM90_FEATURES
  constexpr int smem_size = kNumTMABytesPerWarp * (kNumThreads / WARP_SIZE);
#endif

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                  \
  {                                                                        \
    auto kernel = combine<dtype, ranks, kNumThreads, kNumTMABytesPerWarp>; \
    SET_SHARED_MEMORY_FOR_TMA(kernel);                                     \
    LAUNCH_KERNEL(                                                         \
        &cfg,                                                              \
        kernel,                                                            \
        reinterpret_cast<dtype*>(recv_x),                                  \
        recv_topk_weights,                                                 \
        reinterpret_cast<const dtype*>(x),                                 \
        topk_weights,                                                      \
        reinterpret_cast<const dtype*>(bias_0),                            \
        reinterpret_cast<const dtype*>(bias_1),                            \
        pre_perm_idx,                                                      \
        src_idx,                                                           \
        rank_prefix_matrix,                                                \
        channel_prefix_matrix,                                             \
        send_head,                                                         \
        num_tokens,                                                        \
        num_recv_tokens,                                                   \
        hidden,                                                            \
        num_topk,                                                          \
        buffer_ptrs,                                                       \
        rank,                                                              \
        num_max_send_tokens,                                               \
        num_recv_buffer_tokens,                                            \
        acc_reduce);                                                       \
  }                                                                        \
  break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)               \
  SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
  break

  // Even-numbered blocks for sending, odd-numbered blocks for receiving
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);
  GRPCOLL_HOST_ASSERT(kNumThreads >= num_ranks * WARP_SIZE);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

} // namespace intranode

} // namespace magi_attn_comm::grpcoll
