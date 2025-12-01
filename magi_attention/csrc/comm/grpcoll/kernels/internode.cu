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

#include <functional>
#include <optional>

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace magi_attn_comm::grpcoll {

namespace internode {

extern nvshmem_team_t cpu_rdma_team;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Source Meta
///////////////////////////////////////////////////////////////////////////////////////////////////

struct SourceMeta {
  int src_rdma_rank, is_token_in_nvl_rank_bits;

  GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

  __forceinline__ SourceMeta() = default;

  // TODO: faster encoding
  DEVICE_INLINE SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
    src_rdma_rank = rdma_rank;
    is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
#pragma unroll
    for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
      is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
  }

  DEVICE_INLINE bool is_token_in_nvl_rank(int nvl_rank) const {
    return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
  }
};

GRPCOLL_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
  return sizeof(SourceMeta);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(const int num_rdma_ranks) {
  return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

constexpr int get_num_threads_dispatch(const int num_dispatch_rdma_sender_warps) {
  return (num_dispatch_rdma_sender_warps + 1 + NUM_MAX_NVL_PEERS) * WARP_SIZE;
}

constexpr int get_num_threads_combine(const int num_combine_forwarder_warps) {
  return (num_combine_forwarder_warps + 1) * WARP_SIZE;
}

HOST_DEVICE_INLINE int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
  return static_cast<int>(align(
      /*hidden_states*/ hidden_int4 * sizeof(int4) +
          /*source_meta*/ sizeof(SourceMeta) +
          /*fp8_scales*/ num_scales * sizeof(float) +
          /*topk_idx*/ num_topk_idx * sizeof(int) +
          /*topk_weights*/ num_topk_weights * sizeof(float),
      sizeof(int4)));
}

HOST_DEVICE_INLINE std::pair<int, int> get_rdma_clean_meta(
    int hidden_int4,
    int num_scales,
    int num_topk_idx,
    int num_topk_weights,
    int num_rdma_ranks,
    int num_rdma_recv_buffer_tokens,
    int num_channels) {
  // Return `int32_t` offset and count to clean
  return {
      (get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels) /
          sizeof(int),
      (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

HOST_DEVICE_INLINE std::pair<int, int> get_nvl_clean_meta(
    int hidden_int4,
    int num_scales,
    int num_topk_idx,
    int num_topk_weights,
    int num_rdma_ranks,
    int num_nvl_ranks,
    int num_nvl_recv_buffer_tokens,
    int num_channels,
    bool is_dispatch) {
  // Return `int32_t` offset and to clean
  GRPCOLL_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

  return {
      (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks * num_channels) / sizeof(int),
      num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
  };
}

template <bool kLowLatencyMode>
DEVICE_INLINE int get_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
  return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

template <bool kLowLatencyMode>
DEVICE_INLINE void nvshmem_sync_with_same_gpu_idx(const nvshmem_team_t& rdma_team) {
  kLowLatencyMode ? void(nvshmem_sync(rdma_team)) : nvshmem_sync_all();
}

template <bool kLowLatencyMode>
DEVICE_INLINE void wait_all_inflight_wrs_finished(const int num_threads, const int thread_id, const int num_rdma_ranks, const int rdma_rank, const int nvl_rank) {
  const auto qps_per_rdma_rank = ibgda_get_qps_per_rank();
  for (int i = thread_id; i < qps_per_rdma_rank * (num_rdma_ranks - 1); i += num_threads) {
    auto dst_rdma_rank = (i / qps_per_rdma_rank + rdma_rank + 1) % num_rdma_ranks;
    auto qp_id = i % qps_per_rdma_rank;
    nvshmemi_ibgda_quiet(get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), qp_id);
  }
  __syncthreads();
}

template <bool kLowLatencyMode, bool kSyncOnly = false>
DEVICE_INLINE void barrier_all(const int thread_id, const nvshmem_team_t rdma_team, int** barrier_signal_ptrs, const int nvl_rank) {
  if (thread_id == WARP_SIZE) // NOTES: we need at least 2 warps
    nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
  barrier_block<NUM_MAX_NVL_PEERS, kSyncOnly>(barrier_signal_ptrs, nvl_rank);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Notify Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <bool kLowLatencyMode, int kNumThreads, int kNumRDMARanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_rdma_rank,
    int* moe_recv_rdma_counter_mapped,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    const bool* is_token_in_rank,
    int num_tokens,
    int num_channels,
    const int rdma_clean_offset,
    const int rdma_num_int_clean,
    const int nvl_clean_offset,
    const int nvl_num_int_clean,
    int* rdma_channel_prefix_matrix,
    int* recv_rdma_rank_prefix_sum,
    int* gbl_channel_prefix_matrix,
    int* recv_gbl_rank_prefix_sum,
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    const nvshmem_team_t rdma_team) {
  const auto sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  constexpr int kNumWarps = kNumThreads / WARP_SIZE;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;
  GRPCOLL_STATIC_ASSERT(kNumWarps > 1, "Too few warps"); // for `barrier_all`
  GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= kNumThreads, "Invalid number of NVL peers");

  /** NOTE:
   * The first SM is responsible to:
   *  1. wait all previous inflight WRs finished
   *  2. clean the RDMA/NVL buffer
   *  3. switch meta data with other RDMA/NVL peers
   *  4. calculate meta tensors `recv_rdma_rank_prefix_sum` and `recv_gbl_rank_prefix_sum`
   *
   * Each of the rest SMs is responsible for one RDMA peer to:
   *  1. calculate meta tensors `rdma_channel_prefix_matrix` and `gbl_channel_prefix_matrix`
   */
  if (sm_id == 0) {
    // Wait until all previous inflight WRs for each QP of each RDMA peer are finished
    wait_all_inflight_wrs_finished<kLowLatencyMode>(kNumThreads, thread_id, kNumRDMARanks, rdma_rank, nvl_rank);

    // Barrier all first
    barrier_all<kLowLatencyMode, /*kSyncOnly=*/true>(thread_id, rdma_team, barrier_signal_ptrs, nvl_rank);

    // Get RDMA symmetric buffer for temporary meta data switch
    // `meta_elems_per_rdma_rank_int`:
    //  1. first `NUM_MAX_NVL_PEERS` elems: number of send/recv tokens for each NVL rank in this node
    //  2. next `num_rdma_experts` elems: number of send/recv tokens for each expert in this node
    //  3. last `1` elem: total number of send/recv tokens for this node
    auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
    const int meta_elems_per_rdma_rank_int = NUM_MAX_NVL_PEERS + num_rdma_experts + 1;
    auto rdma_recv_num_tokens_mixed = SymBuffer<int, /*kDecoupled=*/true>(rdma_buffer_ptr, /*num_elems=*/meta_elems_per_rdma_rank_int, /*num_ranks=*/kNumRDMARanks);

    // Clean up RDMA buffer of this rank for later meta data switch
    GRPCOLL_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += kNumThreads)
      rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

    // Copy send meta data of this RDMA rank to its local send buffer
    //  `num_tokens_per_rank`: shape=(num_ranks,), dtype=int
    //  `num_tokens_per_expert`: shape=(num_experts,), dtype=int
    //  `num_tokens_per_rdma_rank`: shape=(kNumRDMARanks,), dtype=int
    //  `rdma_recv_num_tokens_mixed.send_buffer/recv_buffer`: shape=(kNumRDMARanks, meta_elems_per_rdma_rank_int), dtype=int
    GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= kNumThreads, "Invalid number of RDMA peers");
#pragma unroll
    for (int r = thread_id; r < num_ranks; r += kNumThreads)
      rdma_recv_num_tokens_mixed.send_buffer(r / NUM_MAX_NVL_PEERS)[r % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[r];
#pragma unroll
    for (int e = thread_id; e < num_experts; e += kNumThreads)
      rdma_recv_num_tokens_mixed.send_buffer(e / num_rdma_experts)[NUM_MAX_NVL_PEERS + e % num_rdma_experts] = num_tokens_per_expert[e];
    if (thread_id < kNumRDMARanks)
      rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] = num_tokens_per_rdma_rank[thread_id];
    __syncthreads();

    // Copy send meta data of this RDMA rank from its local send buffer
    // to the remote recv buffer of each RDMA peer
    for (int r = warp_id; r < kNumRDMARanks; r += kNumWarps) {
      if (r != rdma_rank) { // r is RDMA peer, then copy through nvshmem
        nvshmemi_ibgda_put_nbi_warp</*kAlwaysDoPostSend=*/true>(
            /*req_rptr=*/reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank)),
            /*req_lptr=*/reinterpret_cast<uint64_t>(rdma_recv_num_tokens_mixed.send_buffer(r)),
            /*bytes=*/meta_elems_per_rdma_rank_int * sizeof(int),
            /*dst_pe=*/get_dst_rdma_rank<kLowLatencyMode>(r, nvl_rank),
            /*qp_id=*/0,
            /*lane_id=*/lane_id,
            /*message_idx=*/0);
      } else { // r is this RDMA rank, then copy through p2p
        UNROLLED_WARP_COPY(
            /*UNROLL_FACTOR=*/1,
            /*LANE_ID=*/lane_id,
            /*N=*/meta_elems_per_rdma_rank_int,
            /*DST=*/rdma_recv_num_tokens_mixed.recv_buffer(rdma_rank),
            /*SRC=*/rdma_recv_num_tokens_mixed.send_buffer(r),
            /*LD_FUNC=*/ld_volatile_global, // volatile load
            /*ST_FUNC=*/st_na_global // non-cached store
        );
      }
    }
    __syncthreads();

    // Wait all previous RDMA copies finished
    // TODO: more light fence or barrier or signaling
    if (thread_id < kNumRDMARanks and thread_id != rdma_rank)
      nvshmemi_ibgda_quiet(/*dst_pe=*/get_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), /*qp_id=*/0);
    __syncthreads();

    // Barrier RDMA team
    // TODO: overlap RDMA barrier and NVL cleaning
    if (thread_id == 0)
      nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    __syncthreads();

    // Get NVL buffers, sending buffer for dst NVL peer, and receiving buffer for this NVL rank
    //  `nvl_reduced_num_tokens_per_expert`: shape=(num_rdma_experts,), dtype=int
    //      `nvl_reduced_num_tokens_per_expert[e]`: the number of tokens received from all to expert `e` in this node
    //  `nvl_send_num_tokens_per_rank`: shape=(NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //      `nvl_send_num_tokens_per_rank[nvl_rank][r]`: the number of tokens sent from RDMA rank `r` via this NVL rank
    //  `nvl_send_num_tokens_per_expert`: shape=(NUM_MAX_NVL_PEERS, num_nvl_experts), dtype=int
    //      `nvl_send_num_tokens_per_expert[nvl_rank][e]`: the number of tokens sent from all to local expert `e` via this NVL rank
    //  `nvl_recv_num_tokens_per_rank`: shape=(NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //      `nvl_recv_num_tokens_per_rank[p][r]`: the number of tokens received from RDMA rank `r` via NVL rank `p` to this NVL rank
    //  `nvl_recv_num_tokens_per_expert`: shape=(NUM_MAX_NVL_PEERS, num_nvl_experts), dtype=int
    //      `nvl_recv_num_tokens_per_expert[p][e]`: the number of tokens received from all via NVL rank `p` to local expert `e` in this NVL rank
    auto nvl_recv_buffer = buffer_ptrs[nvl_rank], nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
    auto nvl_reduced_num_tokens_per_expert = Buffer<int>(nvl_recv_buffer, /*num_elems=*/num_rdma_experts).advance_also(nvl_send_buffer);
    auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS);
    auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, /*num_elems=*/num_nvl_experts, /*num_ranks=*/NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, /*num_elems=*/num_nvl_experts, /*num_ranks=*/NUM_MAX_NVL_PEERS);

    // Clean up NVL buffer of this NVL rank for later meta data switch
    auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
    GRPCOLL_DEVICE_ASSERT(
        nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes + nvl_send_num_tokens_per_expert.total_bytes <=
        nvl_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += kNumThreads)
      nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

    // Reduce number of received tokens per expert in this node from all
    // and copy into `nvl_reduced_num_tokens_per_expert`
    // TODO: maybe use NVSHMEM reduction
    if (thread_id < num_rdma_experts) { // NOTES: we check `num_rdma_experts <= kNumThreads` in host
      int sum = 0;
#pragma unroll
      for (int r = 0; r < kNumRDMARanks; ++r)
        sum += rdma_recv_num_tokens_mixed.recv_buffer(r)[NUM_MAX_NVL_PEERS + thread_id];
      nvl_reduced_num_tokens_per_expert[thread_id] = sum;
    }
    __syncthreads();

    // Reduce (prefix-summed) number of received tokens from each RDMA peer to this RDMA rank
    // and copy into `recv_rdma_rank_prefix_sum`: shape=(kNumRDMARanks,), dtype=int
    // as well as the total received number to the pinned `moe_recv_rdma_counter` to notify the host
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int r = 0; r < kNumRDMARanks; ++r) {
        sum += rdma_recv_num_tokens_mixed.recv_buffer(r)[NUM_MAX_NVL_PEERS + num_rdma_experts];
        recv_rdma_rank_prefix_sum[r] = sum;
      }
      while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1) // self-rotated wait for the counter reset by the host
        ;
      *moe_recv_rdma_counter_mapped = sum;
    }

    // P2P-copy to remote `nvl_send_num_tokens_per_rank` and `nvl_send_num_tokens_per_expert`
    // in NVL peer indicated by `thread_id`,
    // which hold the number of tokens sent from each RDMA rank / local expert resp. via this NVL rank
    if (thread_id < NUM_MAX_NVL_PEERS) {
#pragma unroll
      for (int r = 0; r < kNumRDMARanks; ++r)
        nvl_send_num_tokens_per_rank.buffer(nvl_rank)[r] = rdma_recv_num_tokens_mixed.recv_buffer(r)[thread_id];
#pragma unroll
      for (int e = 0; e < num_nvl_experts; ++e)
        nvl_send_num_tokens_per_expert.buffer(nvl_rank)[e] = nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + e];
    }

    // Barrier for NVL team and wait for all NVL meta data switch finished
    barrier_block<NUM_MAX_NVL_PEERS, /*kSyncOnly=*/false>(barrier_signal_ptrs, nvl_rank);

    // Reduce (prefix-summed) number of received tokens from each global rank to this rank
    // and copy into `recv_gbl_rank_prefix_sum`: shape=(kNumRanks,), dtype=int
    // as well as the total received number to the pinned `grpcoll_recv_counter` to notify the host
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int r = 0; r < num_ranks; ++r) {
        int src_rdma_rank = r / NUM_MAX_NVL_PEERS, src_nvl_rank = r % NUM_MAX_NVL_PEERS;
        sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
        recv_gbl_rank_prefix_sum[r] = sum;
      }
      while (ld_volatile_global(grpcoll_recv_counter_mapped) != -1) // self-rotated wait for the counter reset by the host
        ;
      *grpcoll_recv_counter_mapped = sum;
    }

    // Reduce number of received tokens for each local expert from all NVL ranks to this NVL rank
    // and copy the total received number to the pinned `moe_recv_expert_counter` to notify the host
    if (thread_id < num_nvl_experts) { // NOTES: we check `num_nvl_experts <= kNumThreads` in host
      int sum = 0;
#pragma unroll
      for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
        sum += nvl_recv_num_tokens_per_expert.buffer(r)[thread_id];
      while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1) // self-rotated wait for the counter reset by the host
        ;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }

    // Barrier all finally
    barrier_all<kLowLatencyMode, /*kSyncOnly=*/false>(thread_id, rdma_team, barrier_signal_ptrs, nvl_rank);
  } else {
    const int dst_rdma_rank = sm_id - 1;

    // Iterate over channels to calculate number of send tokens for each channel of dst RDMA peer
    // and initialize `gbl_channel_prefix_matrix` and `rdma_channel_prefix_matrix`
    GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    for (int channel_id = warp_id; channel_id < num_channels; channel_id += kNumWarps) { // each warp for one channel
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens for this channel
      // each lane gets partial number of tokens sent to each NVL rank in the dst RDMA node to `count_per_nvl_rank`
      // as well as the total number to `count_all_nvl_ranks` for part of tokens in this channel
      int count_all_nvl_ranks = 0, count_per_nvl_rank[NUM_MAX_NVL_PEERS] = {0};
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += WARP_SIZE) { // each lane for one token
        auto is_token_in_rank_uint64 = *reinterpret_cast<const uint64_t*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
        auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_uint64);
#pragma unroll
        for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j)
          count_per_nvl_rank[j] += is_token_in_rank_values[j];
        count_all_nvl_ranks += (is_token_in_rank_uint64 != 0); // NOTES: one `uint64_t` is 8 bytes to cover 8 bools for 8 NVL peers
      }

      // Warp reduce `count_per_nvl_rank` and `count_all_nvl_ranks` for this channel
      count_all_nvl_ranks = warp_reduce_sum(count_all_nvl_ranks);
#pragma unroll
      for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
        count_per_nvl_rank[r] = warp_reduce_sum(count_per_nvl_rank[r]);

      // Write `count_per_nvl_rank` and `count_all_nvl_ranks` into channel matrix by lane0
      //  `gbl_channel_prefix_matrix`: shape=(kNumRanks, kNumChannels), dtype=int
      //  `rdma_channel_prefix_matrix`: shape=(kNumRDMARanks, kNumChannels), dtype=int
      if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
          gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + r) * num_channels + channel_id] = count_per_nvl_rank[r];
        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = count_all_nvl_ranks;
      }
    }
    __syncthreads();

    // Make `rdma_channel_prefix_matrix` prefix-summed
    if (thread_id == 0) {
      auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
      make_prefix_sum(prefix_row, num_channels);
    }

    // Make `gbl_channel_prefix_matrix` prefix-summed
    if (thread_id < NUM_MAX_NVL_PEERS) {
      auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
      make_prefix_sum(prefix_row, num_channels);
    }
  }
}

void notify_dispatch(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_rdma_rank,
    int* moe_recv_rdma_counter_mapped,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    const bool* is_token_in_rank,
    int num_tokens,
    int num_channels,
    int hidden_int4,
    int num_scales,
    int num_topk,
    int* rdma_channel_prefix_matrix,
    int* recv_rdma_rank_prefix_sum,
    int* gbl_channel_prefix_matrix,
    int* recv_gbl_rank_prefix_sum,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_recv_tokens,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes,
    bool low_latency_mode) {
  constexpr int kNumThreads = 512;
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  const auto num_rdma_experts = num_experts / num_rdma_ranks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                                          \
  {                                                                                                                                                          \
    auto notify_dispatch_func = low_latency_mode ? notify_dispatch<true, kNumThreads, num_rdma_ranks> : notify_dispatch<false, kNumThreads, num_rdma_ranks>; \
    LAUNCH_KERNEL(                                                                                                                                           \
        &cfg,                                                                                                                                                \
        notify_dispatch_func,                                                                                                                                \
        num_tokens_per_rank,                                                                                                                                 \
        grpcoll_recv_counter_mapped,                                                                                                                         \
        num_ranks,                                                                                                                                           \
        num_tokens_per_rdma_rank,                                                                                                                            \
        moe_recv_rdma_counter_mapped,                                                                                                                        \
        num_tokens_per_expert,                                                                                                                               \
        moe_recv_expert_counter_mapped,                                                                                                                      \
        num_experts,                                                                                                                                         \
        is_token_in_rank,                                                                                                                                    \
        num_tokens,                                                                                                                                          \
        num_channels,                                                                                                                                        \
        rdma_clean_meta.first,                                                                                                                               \
        rdma_clean_meta.second,                                                                                                                              \
        nvl_clean_meta.first,                                                                                                                                \
        nvl_clean_meta.second,                                                                                                                               \
        rdma_channel_prefix_matrix,                                                                                                                          \
        recv_rdma_rank_prefix_sum,                                                                                                                           \
        gbl_channel_prefix_matrix,                                                                                                                           \
        recv_gbl_rank_prefix_sum,                                                                                                                            \
        rdma_buffer_ptr,                                                                                                                                     \
        buffer_ptrs,                                                                                                                                         \
        barrier_signal_ptrs,                                                                                                                                 \
        rank,                                                                                                                                                \
        cpu_rdma_team);                                                                                                                                      \
  }                                                                                                                                                          \
  break

  // Get clean meta
  auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
  auto nvl_clean_meta =
      get_nvl_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels, true);
  GRPCOLL_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
  GRPCOLL_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
  // REVIEW: why limited to INT_MAX ?
  GRPCOLL_HOST_ASSERT(num_rdma_bytes < INT_MAX);
  GRPCOLL_HOST_ASSERT(num_nvl_bytes < INT_MAX);
  GRPCOLL_HOST_ASSERT(num_rdma_experts <= kNumThreads);
  GRPCOLL_HOST_ASSERT(num_nvl_experts <= kNumThreads);

  // Launch kernel
  SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
  SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);

#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool kLowLatencyMode,
    int kNumRDMARanks,
    bool kCachedMode,
    int kNumTMABytesPerWarp,
    int kNumDispatchRDMASenderWarps,
    int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
GLOBAL_LAUNCH_BOUNDS(get_num_threads_dispatch(kNumDispatchRDMASenderWarps), 1)
void dispatch(
    int4* recv_x,
    float* recv_x_scales,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    SourceMeta* recv_src_meta,
    const int4* x,
    const float* x_scales,
    const int64_t* topk_idx,
    const float* topk_weights,
    int* send_rdma_head,
    int* send_nvl_head,
    int* recv_rdma_channel_prefix_matrix,
    int* recv_gbl_channel_prefix_matrix,
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    const bool* is_token_in_rank,
    int num_tokens,
    int hidden_int4,
    int num_scales,
    int num_topk,
    int num_experts,
    int scale_token_stride,
    int scale_hidden_stride,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks) {
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const bool is_forwarder = sm_id % 2 == 0;

  GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= WARP_SIZE, "Invalid number of RDMA peers");
  GRPCOLL_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms);

  /** NOTE: Determine warp role and its target rank
   * For Forwarder (Even SMs):
   *  1. the first `NUM_MAX_NVL_PEERS` warps are `kRDMAAndNVLForwarder`,
   *    forwarding the received tokens from all RDMA peers in RDMA recv buffer (as RDMA consumers)
   *    to each dst NVL peer in this node (as NVL producers),
   *    each warp for one NVL peer and each lane for one RDMA peer
   *  2. the rest warps are `kForwarderCoordinator`, but only the first one is active,
   *    and it is responsible for updating the minimum RDMA head consumed by all `kRDMAAndNVLForwarder` warps
   *
   * For Sender/Receiver (Odd SMs):
   *  1. the first `kNumDispatchRDMASenderWarps` warps are `kRDMASender`,
   *    copying the corr. channel of tokens to RDMA send buffer of this RDMA rank for each RDMA peer,
   *    each warp for one token in round-robin way
   *  2. the next warp is `kRDMASenderCoordinator`, issuing RDMA copy for tokens
   *    copied to the RDMA send buffer of this RDMA rank by `kRDMASender`
   *    to the RDMA recv buffer of each RDMA peer
   *  3. the rest `NUM_MAX_NVL_PEERS` warps are `kNVLReceivers`, each warp for one NVL peer
   */
  enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };
  const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
      }
    } else { // sender / receiver
      if (warp_id < kNumDispatchRDMASenderWarps) {
        return {WarpRole::kRDMASender, -1}; // Not applicable for RDMA senders
      } else if (warp_id == kNumDispatchRDMASenderWarps) {
        return {WarpRole::kRDMASenderCoordinator, -1}; // Not applicable for RDMA senders
      } else {
        return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
      }
    }
  }();
  const auto warp_role = role_meta.first;
  const auto target_rank = role_meta.second;

  // Get RDMA symmetric buffer
  //  `rdma_channel_data`: shape=(num_channels, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, num_bytes_per_token), dtype=uint8_t
  //  `rdma_channel_meta`: shape=(num_channels, kNumRDMARanks, num_meta_per_rdma_channel), dtype=int
  //  `rdma_channel_head`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
  //  `rdma_channel_tail`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
  const auto hidden_bytes = hidden_int4 * sizeof(int4), scale_bytes = num_scales * sizeof(float);
  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
  constexpr int num_meta_per_rdma_channel = 2 * (NUM_MAX_NVL_PEERS + 1); // (start, end) idx for dst RDMA peer (latter) + its each NVL rank (former)
  auto rdma_channel_data = SymBuffer<uint8_t, /*kDecoupled=*/true>(
      rdma_buffer_ptr,
      /*num_elems=*/num_max_rdma_chunked_recv_tokens * num_bytes_per_token,
      /*num_ranks=*/kNumRDMARanks,
      /*sm_id=*/channel_id,
      /*num_sms=*/num_channels);
  auto rdma_channel_meta = SymBuffer<int, /*kDecoupled=*/true>(
      rdma_buffer_ptr, /*num_elems=*/num_meta_per_rdma_channel, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);
  auto rdma_channel_head =
      SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);
  auto rdma_channel_tail =
      SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);

  // Get NVL buffer
  //  `nvl_channel_x`: shape=(num_channels, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_bytes_per_token), dtype=uint8_t
  //  `nvl_channel_prefix_start`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
  //  `nvl_channel_prefix_end`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
  //  `nvl_channel_head`: shape=(num_channels, NUM_MAX_NVL_PEERS), dtype=int
  //  `nvl_channel_tail`: shape=(num_channels, NUM_MAX_NVL_PEERS), dtype=int
  int rs_wr_rank = 0, ws_rr_rank = 0; // `rs_wr` denotes "Read for Senders, Write for Receivers", while `ws_rr` denotes "Write for Senders, Read for Receivers"
  if (warp_role == WarpRole::kRDMAAndNVLForwarder)
    // NVL forwarder/sender will read from this NVL rank and write to target NVL peer
    rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
  if (warp_role == WarpRole::kNVLReceivers)
    // NVL receiver will read from target NVL peer and write to this NVL rank
    rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;
  auto rs_wr_buffer_ptr = buffer_ptrs[rs_wr_rank], ws_rr_buffer_ptr = buffer_ptrs[ws_rr_rank];
  auto nvl_channel_x = AsymBuffer<uint8_t>(
                           ws_rr_buffer_ptr,
                           /*num_elems=*/num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                           /*num_ranks=*/NUM_MAX_NVL_PEERS,
                           /*sm_id=*/channel_id,
                           /*num_sms=*/num_channels,
                           /*offset=*/rs_wr_rank)
                           .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_start =
      AsymBuffer<int>(
          ws_rr_buffer_ptr, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_end =
      AsymBuffer<int>(
          ws_rr_buffer_ptr, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_head =
      AsymBuffer<int>(rs_wr_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/ws_rr_rank)
          .advance_also(ws_rr_buffer_ptr);
  auto nvl_channel_tail =
      AsymBuffer<int>(ws_rr_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);

  // Prepare RDMA sender warp synchronization
  //  `rdma_send_channel_lock`: the lock to mutex access `rdma_send_channel_tail` and `rdma_send_channel_window` for each RDMA rank
  //  `rdma_send_channel_tail`: the latest released tail for each RDMA rank
  //  `rdma_send_channel_window`: the ongoing 32 transactions' status for each RDMA rank
  //  `sync_rdma_sender_smem`: synchronize warps of `kRDMASender` and `kRDMASenderCoordinator`
  __shared__ int rdma_send_channel_lock[kNumRDMARanks];
  __shared__ int rdma_send_channel_tail[kNumRDMARanks];
  __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks]; // NOTES: each bit in one `uint32_t` corresponds to one transaction
  auto sync_rdma_sender_smem = []() { sync_warp_group(/*group_flag=*/0, /*group_size=*/(kNumDispatchRDMASenderWarps + 1) * WARP_SIZE); };

  // Prepare TMA buffer and init mbarrier
  // NOTES: TMA buffer is only used by `kRDMAAndNVLForwarder` and `kNVLReceivers`
  extern __shared__ __align__(1024) uint8_t smem_tma_buffer[]; // REVIEW: why aligned to 1024 bytes ?
  auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
  auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + hidden_bytes);
  uint32_t tma_phase = 0;
  if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and lane_id == 0) {
    mbarrier_init(tma_mbarrier, 1);
    fence_view_async_shared();
    fence_barrier_init();
  }
  __syncwarp();

  // Prepare NVL forwarder warp synchronization
  //  `forward_channel_head`: the RDMA head for each src RDMA peer of each dst NVL peer / `kRDMAAndNVLForwarder` warp
  //  `forward_channel_retired`: the retire flag for each `kRDMAAndNVLForwarder` warp
  //  `sync_forwarder_smem`: synchronize warps of `kRDMAAndNVLForwarder` and `kForwarderCoordinator` warps
  __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
  __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
  auto sync_forwarder_smem = []() { sync_warp_group(/*group_flag=*/1, /*group_size=*/(NUM_MAX_NVL_PEERS + 1) * WARP_SIZE); };

  // Warp-specialized working
  if (warp_role == WarpRole::kRDMASender) {
    // Get tasks of this channel to send tokens ranging in [token_start_idx, token_end_idx)
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

    // Copy `rdma_channel_meta` for each dst RDMA peer
    //  `gbl_channel_prefix_matrix`: shape=(kNumRanks, kNumChannels), dtype=int
    //    `gbl_channel_prefix_matrix[r][c]`: the prefix-summed number of tokens sent to global rank `r` by channel `c`
    //  `rdma_channel_prefix_matrix`: shape=(kNumRDMARanks, kNumChannels), dtype=int
    //    `rdma_channel_prefix_matrix[r][c]`: the prefix-summed number of tokens sent to RDMA rank `r` by channel `c`
    GRPCOLL_STATIC_ASSERT(num_meta_per_rdma_channel <= WARP_SIZE, "Invalid number of NVL peers");
    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
      auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) // NOTES: for this NVL rank, we directly write to recv buffer
                                                : rdma_channel_meta.send_buffer(dst_rdma_rank);
      if (lane_id < NUM_MAX_NVL_PEERS) { // the start token idx of this channel sent to each NVL rank for dst RDMA peer
        dst_ptr[lane_id] = encode(channel_id == 0 ? 0 : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels + channel_id - 1]);
      } else if (lane_id < NUM_MAX_NVL_PEERS * 2) { // the end token idx of this channel sent to each NVL rank for dst RDMA peer
        dst_ptr[lane_id] = encode(gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) * num_channels + channel_id]);
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2) { // the start token idx of this channel sent to dst RDMA peer
        dst_ptr[lane_id] = encode(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]);
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) { // the end token idx of this channel sent to dst RDMA peer
        dst_ptr[lane_id] = encode(rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id]);
      }
      __syncwarp();

      // RDMA Copy `rdma_channel_meta` to dst RDMA peer
      if (dst_rdma_rank != rdma_rank) {
        nvshmemi_ibgda_put_nbi_warp</*kAlwaysDoPostSend=*/true>(
            /*req_rptr=*/reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
            /*req_lptr=*/reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
            /*bytes=*/num_meta_per_rdma_channel * sizeof(int),
            /*dst_pe=*/get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
            /*qp_id=*/channel_id, // NOTES: each channel use its own qp
            /*lane_id=*/lane_id,
            /*message_idx=*/0);
      }
    }

    // Sync `kRDMASender` with `kRDMASenderCoordinator` to make sure the shared memory of
    // `rdma_send_channel_lock`, `rdma_send_channel_tail` and `rdma_send_channel_window`
    // are cleaned by `kRDMASenderCoordinator` before `kRDMASender` access them
    sync_rdma_sender_smem();

    // Iterate over tokens and copy into buffer
    int64_t token_idx;
    int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
    auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
    GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
      // Update `global_rdma_tail_idx`
      // by counting whether this token is sent to RDMA rank indicated by `lane_id`
      uint64_t is_token_in_rank_uint64 = 0;
      if (lane_id < kNumRDMARanks) {
        is_token_in_rank_uint64 = __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
        global_rdma_tail_idx += (is_token_in_rank_uint64 != 0); // NOTES: one `uint64_t` is 8 bytes to cover 8 bools for 8 NVL peers
      }
      __syncwarp();

      // Skip the token which does not belong to this warp
      // NOTES: each warp for one token in round-robin way
      if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
        continue;

      const auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

      // Wait the queue non-full, i.e. the remote buffer to be released
      auto start_time = clock64();
      while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
        cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id))); // volatile load

        // Timeout check
        if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
          printf(
              "grpcoll dispatch RDMA sender timeout, channel: %d, RDMA: %d, NVL: %d, dst RDMA rank: %d, head: %d, tail: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              lane_id,
              cached_rdma_channel_head,
              rdma_tail_idx);
          trap();
        }
      }
      __syncwarp();

      // Store RDMA head for combine stage to reduce
      //  `send_rdma_head`: shape=(num_tokens, kNumRDMARanks), dtype=int
      if (lane_id < kNumRDMARanks and not kCachedMode)
        send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

      // Broadcast tails
      SourceMeta src_meta;
      int num_topk_rdma_ranks = 0;
      void* dst_send_buffers[kNumTopkRDMARanks];
#pragma unroll
      // Broadcast info about the topk RDMA ranks this token will be sent to, including:
      //  1. the `src_meta` to tell which NVL ranks this token should be sent to in each RDMA peer
      //  2. the `dst_send_buffer` ptr of this token in each RDMA send buffer queue
      for (int r = 0, slot_idx; r < kNumRDMARanks; ++r) {
        if ((slot_idx = broadcast_in_warp(/*val=*/rdma_tail_idx, /*src_lane=*/r)) >= 0) {
          slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
          auto recv_is_token_in_rank_uint64 = broadcast_ptr_in_warp(/*ptr=*/is_token_in_rank_uint64, /*src_lane=*/r);
          auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
          if (lane_id == num_topk_rdma_ranks)
            src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
          dst_send_buffers[num_topk_rdma_ranks++] =
              reinterpret_cast<uint8_t*>(broadcast_ptr_in_warp(/*ptr=*/send_buffer, /*src_lane=*/r)) + slot_idx * num_bytes_per_token;
        }
      }
      GRPCOLL_DEVICE_ASSERT(num_topk_rdma_ranks <= kNumTopkRDMARanks); // REVIEW: why at most 8 RDMA peers to send to ?

      // Warp-copy the hidden value of this token
      // into each RDMA send buffer for each topk RDMA rank to send to
      auto st_topk_rdma_ranks = [=](const int hidden_offset, const int4& hidden_val_int4) {
#pragma unroll
        for (int j = 0; j < num_topk_rdma_ranks; ++j)
          st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + hidden_offset, hidden_val_int4);
      };
      UNROLLED_WARP_COPY(
          /*UNROLL_FACTOR=*/5,
          /*LANE_ID=*/lane_id,
          /*N=*/hidden_int4,
          /*DST=*/0,
          /*SRC=*/x + token_idx * hidden_int4,
          /*LD_FUNC=*/ld_nc_global, // non-cached load
          /*ST_FUNC=*/st_topk_rdma_ranks // non-cached store to each send buffer of each topk RDMA rank
      );

#pragma unroll
      // Offset the send buffers across the hidden states part
      for (int r = 0; r < num_topk_rdma_ranks; ++r)
        dst_send_buffers[r] = reinterpret_cast<int4*>(dst_send_buffers[r]) + hidden_int4;

#pragma unroll
      // Copy `x_scales`
      // into each RDMA send buffer for each topk RDMA rank to send to
      for (int i = lane_id; i < num_scales; i += WARP_SIZE) {
        auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
        auto value = ld_nc_global(x_scales + offset);
#pragma unroll
        for (int j = 0; j < num_topk_rdma_ranks; ++j)
          st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
      }

#pragma unroll
      // Offset the send buffers across the scales
      for (int r = 0; r < num_topk_rdma_ranks; ++r)
        dst_send_buffers[r] = reinterpret_cast<float*>(dst_send_buffers[r]) + num_scales;

      // Copy `src_meta`
      // into each RDMA send buffer for each topk RDMA rank to send to
      if (lane_id < num_topk_rdma_ranks)
        st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);

#pragma unroll
      // Offset the send buffers across the src_meta
      for (int r = 0; r < num_topk_rdma_ranks; ++r)
        dst_send_buffers[r] = reinterpret_cast<SourceMeta*>(dst_send_buffers[r]) + 1;

#pragma unroll
      // Copy `topk_idx` and `topk_weights`
      // into each RDMA send buffer for each topk RDMA rank to send to
      for (int i = lane_id; i < num_topk * num_topk_rdma_ranks; i += WARP_SIZE) {
        auto rank_idx = i / num_topk, copy_idx = i % num_topk;
        auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
        auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
        st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
        st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
      }
      __syncwarp();

      // Release the transaction in the window
      if (is_token_in_rank_uint64 != 0) {
        // Acquire lock first
        acquire_lock(rdma_send_channel_lock + lane_id);
        auto latest_tail = rdma_send_channel_tail[lane_id];
        auto window_slot = rdma_tail_idx - latest_tail;

        // If the window is already full,
        // release the lock to let other warps update the latest tail
        // and then retry to take up a valid window slot
        while (window_slot >= WARP_SIZE) {
          release_lock(rdma_send_channel_lock + lane_id);
          acquire_lock(rdma_send_channel_lock + lane_id);
          latest_tail = rdma_send_channel_tail[lane_id];
          window_slot = rdma_tail_idx - latest_tail;
        }

        // Mark the window slot as released by setting the corr. bit to 1
        auto window = rdma_send_channel_window[lane_id] | (1u << window_slot);

        // Update the latest tail and shift the window
        if (window_slot == 0) {
          // If all the window slot are set to 1, i.e. all released, then `~window` == 0, and num_empty_slots == WARP_SIZE
          // Otherwise, `__ffs(~window) - 1` will find the least slot idx set to 0, which equals to `num_empty_slots`, i.e. number of least slots all set to 1
          // e.g. if window == 0b01010111, then `~window` == 0b10101000, `__ffs(~window) - 1` == 3, since the least 3 slots in window are all set to 1
          auto num_empty_slots = (~window) == 0 ? WARP_SIZE : __ffs(~window) - 1;

          // Update the latest tail by `num_empty_slots`
          st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots); // CTA scope, release order

          // Shift the window by `num_empty_slots` bits
          // e.g. if window == 0b01010111, then `window >> num_empty_slots` == 0b00001010
          window >>= num_empty_slots;
        }
        rdma_send_channel_window[lane_id] = window;

        // Release lock
        release_lock(rdma_send_channel_lock + lane_id);
      }
      __syncwarp();
    }
  } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
    // Clean shared memory of
    // `rdma_send_channel_lock`, `rdma_send_channel_tail` and `rdma_send_channel_window`
    (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
    (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
    (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;

    // Sync `kRDMASender` with `kRDMASenderCoordinator` to make sure the shared memory of
    // `rdma_send_channel_lock`, `rdma_send_channel_tail` and `rdma_send_channel_window`
    // are cleaned by `kRDMASenderCoordinator` before `kRDMASender` access them
    sync_rdma_sender_smem();

    // Get number of tokens to send in this channel for the RDMA rank indicated by `lane_id`
    //  `rdma_channel_prefix_matrix`: shape=(kNumRDMARanks, kNumChannels), dtype=int
    int num_tokens_to_send = 0;
    if (lane_id < kNumRDMARanks) {
      num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
      if (channel_id > 0)
        num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
    }

    // Issue RDMA copy for all tokens
    // copied into each RDMA send buffer for each RDMA rank by `kRDMASender`
    int last_issued_tail = 0;
    auto start_time = clock64();
    while (any_in_warp(num_tokens_to_send > 0)) {
      // Timeout check
      if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
        printf(
            "grpcoll dispatch RDMA sender coordinator timeout, channel: %d, RDMA: %d, NVL %d, dst RDMA: %d, tail: %d, remaining: %d\n",
            channel_id,
            rdma_rank,
            nvl_rank,
            lane_id,
            last_issued_tail,
            num_tokens_to_send);
        trap();
      }

      // Iterate all RDMA ranks if there's any (remaining) token to send
      for (int r = 0, synced_num_tokens_to_send; r < kNumRDMARanks; ++r) {
        // To mitigate in-cast congestion, shuffle the starting index of target rank for different ranks and channels
        int dst_rdma_rank = (r + channel_id + rdma_rank) % kNumRDMARanks;
        synced_num_tokens_to_send = broadcast_in_warp(/*val=*/num_tokens_to_send, /*src_lane=*/dst_rdma_rank);
        if (synced_num_tokens_to_send == 0)
          continue;

        // Read the latest progress of `kRDMASender`
        // to get the number of tokens copied into the RDMA send buffer
        // NOTES: `rdma_send_channel_tail` does not need to be protected by lock
        auto processed_tail = broadcast_in_warp(/*val=*/ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank))); // CTA scope, acquire order
        auto synced_last_issued_tail = broadcast_in_warp(/*val=*/last_issued_tail, /*src_lane=*/dst_rdma_rank);
        auto num_tokens_processed = processed_tail - synced_last_issued_tail;

        // If the number of tokens to be processed is not enough as a chunk, skip for next round
        if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
          continue;

        // Issue RDMA copy for a chunk of tokens in this round
        // from the send buffer of this RDMA rank to the recv buffer of `dst_rdma_rank`
        auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
        GRPCOLL_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
        if (dst_rdma_rank != rdma_rank) { // dst RDMA peer
          auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
          GRPCOLL_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
          const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
          const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
          const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
          // TODO: try thread-level `put_nbi` ?
          nvshmemi_ibgda_put_nbi_warp</*kAlwaysDoPostSend=*/true>(
              /*req_rptr=*/dst_ptr,
              /*req_lptr=*/src_ptr,
              /*bytes=*/num_bytes_per_msg,
              /*dst_pe=*/get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
              /*qp_id=*/channel_id,
              /*lane_id=*/lane_id,
              /*message_idx=*/0);
        } else { // this RDMA rank
          // Already in its own recv buffer, so no need to copy
          memory_fence(); // NOTES: use lighter fence for local memory operations
        }
        __syncwarp();

        // Update last issued tails by last round of chunk size
        // as well as the `rdma_channel_tail` of `dst_rdma_rank` by atomic-add
        if (lane_id == dst_rdma_rank) {
          last_issued_tail += num_tokens_to_issue;
          num_tokens_to_send -= num_tokens_to_issue;
          nvshmemi_ibgda_amo_nonfetch_add(
              /*rptr=*/rdma_channel_tail.buffer(rdma_rank),
              /*value=*/num_tokens_to_issue,
              /*pe=*/get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
              /*qp_id=*/channel_id,
              /*is_local_copy=*/dst_rdma_rank == rdma_rank);
        }
        __syncwarp();
      }
    }
  } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
    const auto dst_nvl_rank = target_rank; // each warp for one dst NVL peer

    // Wait `rdma_channel_meta` to be ready for each RDMA peer
    // NOTES: each lane will ready specific `num_tokens_to_recv_from_rdma` and `rdma_token_start_idx` for each RDMA peer
    int num_tokens_to_recv_from_rdma = 0, rdma_token_start_idx = 0;
    auto start_time = clock64();
    if (lane_id < kNumRDMARanks) {
      while (true) {
        auto nvl_token_start_idx_encoded = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
        auto nvl_token_end_idx_encoded = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
        auto rdma_token_start_idx_encoded = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
        auto rdma_token_end_idx_encoded = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
        if (nvl_token_start_idx_encoded < 0 and nvl_token_end_idx_encoded < 0 and rdma_token_start_idx_encoded < 0 and
            rdma_token_end_idx_encoded < 0) { // all valid encoded values
          // Store encoded `nvl_token_start_idx` and `nvl_token_end_idx`
          // to `nvl_channel_prefix_start` and `nvl_channel_prefix_end` in target NVL peer
          const auto nvl_token_start_idx = decode(nvl_token_start_idx_encoded), nvl_token_end_idx = decode(nvl_token_end_idx_encoded);
          GRPCOLL_DEVICE_ASSERT(nvl_token_start_idx >= 0 and nvl_token_end_idx >= nvl_token_start_idx);
          st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, nvl_token_start_idx_encoded);
          st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, nvl_token_end_idx_encoded);

          // Get RDMA channel received token count
          rdma_token_start_idx = decode(rdma_token_start_idx_encoded);
          auto rdma_token_end_idx = decode(rdma_token_end_idx_encoded);
          num_tokens_to_recv_from_rdma = rdma_token_end_idx - rdma_token_start_idx;
          GRPCOLL_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);

          // Store `rdma_token_end_idx` for combine stage
          //  `recv_rdma_channel_prefix_matrix`: shape=[kNumRDMARanks, kNumChannels], dtype=int
          if (not kCachedMode)
            recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = rdma_token_end_idx;

          // Shift `rdma_token_start_idx` by RDMA rank offset
          //  `recv_rdma_rank_prefix_sum`: shape=[kNumRDMARanks,], dtype=int
          rdma_token_start_idx += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];

          break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "grpcoll dispatch RDMA and NVL forwarder timeout for RDMA channel meta, channel: %d, RDMA: %d, NVL: %d, src RDMA rank: %d, dst NVL rank: %d, encoded meta: %d, %d, %d, %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              lane_id,
              dst_nvl_rank,
              nvl_token_start_idx_encoded,
              nvl_token_end_idx_encoded,
              rdma_token_start_idx_encoded,
              rdma_token_end_idx_encoded);
          trap();
        }
      }
    }
    __syncwarp();

    // Shift cached head ptr to the first token in RDMA buffer to forward to dst NVL peer
    //  `send_nvl_head`: shape=[num_rdma_recv_tokens, NUM_MAX_NVL_PEERS], dtype=int
    send_nvl_head += rdma_token_start_idx * NUM_MAX_NVL_PEERS + dst_nvl_rank;

    // Sync shared memory of
    // `forward_channel_head` and `forward_channel_retired`
    // to make sure they are cleaned by `kForwarderCoordinator`
    sync_forwarder_smem();

    // Forward tokens from RDMA buffer to dst NVL buffer
    // NOTES: always start from the local rank
    int src_rdma_rank = sm_id % kNumRDMARanks, rdma_nvl_token_idx = 0;
    int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
    int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0;
    while (any_in_warp(num_tokens_to_recv_from_rdma > 0)) {
      // Wait NVL queue empty enough to forward a chunk of tokens
      // as a producer to read the NVL head and update the NVL tail later
      start_time = clock64();
      while (true) {
        const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
        if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
          break;

        // Read the NVL head updated by `kNVLReceivers`
        // REVIEW: here all lanes repeatly read the same `nvl_channel_head` ?
        cached_nvl_channel_head = broadcast_in_warp(/*val=*/ld_volatile_global(nvl_channel_head.buffer())); // volatile load

        // Timeout check
        if (lane_id == 0 and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "grpcoll dispatch RDMA and NVL forwarder timeout (NVL head check), channel: %d, RDMA: %d, NVL: %d, dst NVL rank: %d, head: %d, tail: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              ld_volatile_global(nvl_channel_head.buffer()),
              cached_nvl_channel_tail);
          trap();
        }
      }

      // Find next src RDMA peer to be forwarded (round-robin)
      // as a consumer to read the RDMA tail and update the RDMA head (by `kForwarderCoordinator`) later
      start_time = clock64();
      while (true) {
        src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
        if (broadcast_in_warp(/*val=*/num_tokens_to_recv_from_rdma, /*src_lane=*/src_rdma_rank) > 0) {
          if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
            cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank))); // system scope, acquire order
          if (broadcast_in_warp(/*val=*/cached_rdma_channel_tail > cached_rdma_channel_head, /*src_lane=*/src_rdma_rank))
            break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
          printf(
              "grpcoll dispatch RDMA and NVL forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA lane: %d, head: %d, tail: %d, expected: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              lane_id,
              cached_rdma_channel_head,
              cached_rdma_channel_tail,
              num_tokens_to_recv_from_rdma);
          trap();
        }
      }

      // Determine the RDMA head and tail for the src RDMA peer in this round
      auto src_rdma_head = broadcast_in_warp(/*val=*/cached_rdma_channel_head, /*src_lane=*/src_rdma_rank);
      auto src_rdma_tail = broadcast_in_warp(/*val=*/cached_rdma_channel_tail, /*src_lane=*/src_rdma_rank);

      // Iterate over every token from the src RDMA buffer between `src_rdma_head` and `src_rdma_tail`
      // and copy into dst NVL buffer through TMA
      for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
        // Get slot idx in the RDMA recv queue
        auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;

        // Get token ptr in RDMA recv buffer
        auto rdma_token_ptr = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;

        // Get `src_meta` for current token
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(rdma_token_ptr + hidden_bytes + scale_bytes)); // non-cached load

        // Decrement `num_tokens_to_recv_from_rdma`
        lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;

        // If this RDMA token does not need to forward to dst NVL peer, skip it
        // Otherwise, increment `rdma_nvl_token_idx` and update `send_nvl_head`
        // of current RDMA token for combine stage
        bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
        if (lane_id == src_rdma_rank) {
          auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
          rdma_nvl_token_idx += is_in_dst_nvl_rank;
          if (not kCachedMode)
            send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
        }
        if (!is_in_dst_nvl_rank)
          continue;

        // Get an empty slot in NVL queue
        // and increment `cached_nvl_channel_tail`
        int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;

        // Get token ptr in NVL buffer of dst NVL peer
        auto nvl_token_ptr = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

        // TMA-copy token from RDMA buffer to TMA buffer in shared memory
        if (lane_id == 0) { // issued by lane0
          tma_load_1d(
              /*smem_ptr=*/tma_buffer,
              /*gmem_ptr=*/rdma_token_ptr,
              /*mbar_ptr=*/tma_mbarrier,
              /*num_bytes=*/num_bytes_per_token,
              /*evict_first=*/false);
          mbarrier_arrive_and_expect_tx(/*mbar_ptr=*/tma_mbarrier, /*num_bytes=*/num_bytes_per_token);
        }
        __syncwarp();

        // Wait TMA load to be finished
        // and flip the `tma_phase` in-place
        mbarrier_wait(/*mbar_ptr=*/tma_mbarrier, /*stage=*/tma_phase);

        // TMA-copy token from TMA buffer in shared memory to NVL buffer
        if (lane_id == 0) { // issued by lane0
          tma_store_1d(
              /*smem_ptr=*/tma_buffer,
              /*gmem_ptr=*/nvl_token_ptr,
              /*num_bytes=*/num_bytes_per_token,
              /*evict_first=*/true);
        }
        __syncwarp();

        // Early stop when the NVL chunk is full
        if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
          src_rdma_tail = i + 1;

        // Wait TMA store to be finished
        tma_store_wait();
        __syncwarp();
      }

      // Sync RDMA head to shared memory
      // to let `kForwarderCoordinator` warp update the minimum RDMA head across all `kRDMAAndNVLForwarder` warps
      if (lane_id == src_rdma_rank)
        forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);
      __syncwarp();

      // Update NVL tail
      if (lane_id == 0)
        st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail); // system scope, release order
    }
    __syncwarp();

    // Retired this warp by toggling the `forward_channel_retired` in shared memory
    if (lane_id == 0)
      forward_channel_retired[dst_nvl_rank] = true;
  } else if (warp_role == WarpRole::kForwarderCoordinator) {
    // Since we only need the first warp as `kForwarderCoordinator`,
    // and extra warps should exit directly
    if (target_rank > 0)
      return;

    // Clean shared memory of
    // `forward_channel_head` and `forward_channel_retired`
    GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= WARP_SIZE, "Invalid number of NVL peers");
#pragma unroll
    for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += WARP_SIZE)
      forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
    if (lane_id < NUM_MAX_NVL_PEERS)
      forward_channel_retired[lane_id] = false;

    // Sync shared memory of
    // `forward_channel_head` and `forward_channel_retired`
    // before `kRDMAAndNVLForwarder` warps access them
    sync_forwarder_smem();

    // Loop minimum head in `forward_channel_head` and update RDMA head
    int last_head = 0, target_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
    while (true) {
      // Find minimum head recorded by `kRDMAAndNVLForwarder` warps
      int min_head = INT_MAX;
#pragma unroll
      for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
        if (!forward_channel_retired[r])
          min_head = min(min_head, forward_channel_head[r][target_rdma_rank]);
      if (all_in_warp(min_head == INT_MAX)) // all `kRDMAAndNVLForwarder` warps are retired
        break;

      // Update RDMA head by atomic add
      // REVIEW: why here we need to check `min_head >= last_head + num_max_rdma_chunked_send_tokens` ?
      if (min_head != INT_MAX and min_head >= last_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
        nvshmemi_ibgda_amo_nonfetch_add(
            /*rptr=*/rdma_channel_head.buffer(rdma_rank),
            /*value=*/min_head - last_head,
            /*pe=*/get_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
            /*qp_id=*/channel_id + num_channels,
            /*is_local_copy=*/lane_id == rdma_rank);
        last_head = min_head;
      }

      // Nanosleep and let other warps work
      // REVIEW: why here we need to nanosleep but not in intranode group cast ?
      __nanosleep(NUM_WAIT_NANOSECONDS);
    }
  } else { // WarpRole::kNVLReceivers
    // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
    const int src_nvl_rank = target_rank;
    const int local_expert_begin = rank * (num_experts / num_ranks);
    const int local_expert_end = local_expert_begin + (num_experts / num_ranks);

    // Load global rank offset for the src NVL rank in the RDMA peer indicated by `lane_id`
    //  `recv_gbl_rank_prefix_sum`: shape=(kNumRanks,), dtype=int
    int total_offset = 0;
    if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
      total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

    // Read channel offsets for the src NVL rank in the RDMA peer indicated by `lane_id`
    // and update total offset by the prefix start
    int start_offset = 0, end_offset = 0, num_tokens_to_recv;
    auto start_time = clock64();
    while (lane_id < kNumRDMARanks) {
      start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id); // volatile load
      end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id); // volatile load
      if (start_offset < 0 and end_offset < 0) { // all valid encoded offsets
        start_offset = decode(start_offset), end_offset = decode(end_offset);
        total_offset += start_offset;
        break;
      }

      // Timeout check
      if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
        printf(
            "grpcoll dispatch NVL receiver timeout, channel: %d, RDMA: %d, NVL: %d, src RDMA rank: %d, src NVL rank: %d, start: %d, end: %d\n",
            channel_id,
            rdma_rank,
            nvl_rank,
            lane_id,
            src_nvl_rank,
            start_offset,
            end_offset);
        trap();
      }
    }

    // Warp-reduce across all RDMA peers for the src NVL rank
    num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

    // Store `recv_gbl_channel_prefix_matrix` for combine stage
    //  `recv_gbl_channel_prefix_matrix`: shape=(kNumRanks, num_channels), dtype=int
    if (lane_id < kNumRDMARanks and not kCachedMode)
      recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] = total_offset;
    __syncwarp();

    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      // Wait for NVL recv queue to be non-empty
      start_time = clock64();
      while (true) {
        // Ready to copy
        if (cached_channel_head_idx != cached_channel_tail_idx)
          break;

        // Read the NVL tail updated by `kRDMAAndNVLForwarder`
        // REVIEW: here all lanes repeatedly load the same `nvl_channel_tail` ?
        cached_channel_tail_idx = broadcast_in_warp(/*val=*/ld_acquire_sys_global(nvl_channel_tail.buffer())); // system scope, acquire order

        // Timeout check
        if (lane_id == 0 and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "grpcoll dispatch NVL receiver timeout, channel: %d, RDMA: %d, NVL: %d, src NVL rank: %d, head: %d, tail: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              src_nvl_rank,
              cached_channel_head_idx,
              cached_channel_tail_idx);
          trap();
        }
      }

      // Iterate over the tokens in the NVL recv queue of this NVL rank
      // from `cached_channel_head_idx` to `cached_channel_tail_idx`
      // and copy into `recv_x`, as well as other data
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
        // Get slot idx in queue
        // and update `cached_channel_head_idx` for next token
        int slot_idx_in_queue = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;

        // Get token ptr in the NVL recv buffer of this NVL rank
        auto token_ptr_in_buffer = nvl_channel_x.buffer() + slot_idx_in_queue * num_bytes_per_token;

        // Load src meta to get the `src_rdma_rank` for this token
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(token_ptr_in_buffer + hidden_bytes + scale_bytes));

        // Get recv token idx in the `recv_x`
        int64_t token_idx_in_recv_x = broadcast_in_warp(/*val=*/total_offset, /*src_lane=*/src_meta.src_rdma_rank);

        // Increment `total_offset` of next token for `src_rdma_rank`
        (lane_id == src_meta.src_rdma_rank) ? (++total_offset) : 0;

        // Get TMA load bytes, including hidden states and scales
        bool scale_aligned = (scale_bytes % 16 == 0);
        auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);

        // TMA-copy token from NVL recv buffer to TMA buffer in shared memory
        // including hidden states and scales
        if (lane_id == 0) { // issued by lane0
          tma_load_1d(
              /*smem_ptr=*/tma_buffer,
              /*gmem_ptr=*/token_ptr_in_buffer,
              /*mbar_ptr=*/tma_mbarrier,
              /*num_bytes=*/tma_load_bytes,
              /*evict_first=*/true);
          mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);
        }
        __syncwarp();

        // Wait TMA load to be finished
        // and flip the `tma_phase` in-place
        mbarrier_wait(tma_mbarrier, tma_phase);

        // TMA-copy hidden states of the token from TMA buffer in shared memory to `recv_x`
        if (lane_id == 0) { // issued by lane0
          tma_store_1d(
              /*smem_ptr=*/tma_buffer,
              /*gmem_ptr=*/recv_x + token_idx_in_recv_x * hidden_int4,
              /*num_bytes=*/hidden_bytes,
              /*evict_first=*/false);
        }
        __syncwarp();

        // Copy scales of the token from TMA buffer in shared memory to `recv_x_scales`
        // TODO: make it as templated
        token_ptr_in_buffer += hidden_bytes;
        if (scale_aligned) { // if aligned to 16 bytes, use TMA copy
          tma_store_1d(
              /*smem_ptr=*/tma_buffer + hidden_bytes,
              /*gmem_ptr=*/recv_x_scales + token_idx_in_recv_x * num_scales,
              /*num_bytes=*/scale_bytes,
              /*evict_first=*/false);
        } else { // if not aligned, use warp copy
          UNROLLED_WARP_COPY(
              /*UNROLL_FACTOR=*/1,
              /*LANE_ID=*/lane_id,
              /*N=*/num_scales,
              /*DST=*/recv_x_scales + token_idx_in_recv_x * num_scales,
              /*SRC=*/reinterpret_cast<float*>(token_ptr_in_buffer),
              /*LD_FUNC=*/ld_nc_global,
              /*ST_FUNC=*/st_na_global);
        }

        // Copy src meta to `recv_src_meta` for combine stage
        token_ptr_in_buffer += scale_bytes;
        if (lane_id == 0 and not kCachedMode)
          st_na_global(recv_src_meta + token_idx_in_recv_x, src_meta); // non-cached store

        // Copy `topk_idx` and `topk_weights` to `recv_topk_idx` and `recv_topk_weights`
        token_ptr_in_buffer += sizeof(SourceMeta);
        if (lane_id < num_topk) {
          // Read topk idx/weight from recv buffer
          auto topk_idx_value = static_cast<int64_t>(ld_nc_global(reinterpret_cast<int*>(token_ptr_in_buffer) + lane_id));
          auto topk_weight_value = ld_nc_global(reinterpret_cast<float*>(token_ptr_in_buffer + sizeof(int) * num_topk) + lane_id);
          auto idx_in_recv_topk = token_idx_in_recv_x * num_topk + lane_id;

          // Transform and write to recv topk idx/weight
          topk_idx_value = (topk_idx_value >= local_expert_begin and topk_idx_value < local_expert_end) ? topk_idx_value - local_expert_begin : -1;
          topk_weight_value = topk_idx_value >= 0 ? topk_weight_value : 0.0f;
          st_na_global(recv_topk_idx + idx_in_recv_topk, topk_idx_value);
          st_na_global(recv_topk_weights + idx_in_recv_topk, topk_weight_value);
        }

        // Wait TMA store to be finished
        tma_store_wait();
        __syncwarp();
      }

      // Update NVL queue head
      if (lane_id == 0)
        st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx); // system scope, relaxed order
    }
  }
}

void dispatch(
    void* recv_x,
    float* recv_x_scales,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    void* recv_src_meta,
    const void* x,
    const float* x_scales,
    const int64_t* topk_idx,
    const float* topk_weights,
    int* send_rdma_head,
    int* send_nvl_head,
    int* recv_rdma_channel_prefix_matrix,
    int* recv_gbl_channel_prefix_matrix,
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    const bool* is_token_in_rank,
    int num_tokens,
    int hidden_int4,
    int num_scales,
    int num_topk,
    int num_experts,
    int scale_token_stride,
    int scale_hidden_stride,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks,
    bool is_cached_dispatch,
    cudaStream_t stream,
    int num_channels,
    bool low_latency_mode) {
  constexpr int kNumDispatchRDMASenderWarps = 7;
  constexpr int kNumTMABytesPerWarp = 16384;
  constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS;
  constexpr int kNumThreads = get_num_threads_dispatch(kNumDispatchRDMASenderWarps);
  constexpr int kNumWarps = kNumThreads / WARP_SIZE;
  GRPCOLL_STATIC_ASSERT(kNumWarps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS, "Invalid number of warps");

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                                                                 \
  {                                                                                                                                                          \
    auto dispatch_func = low_latency_mode ? (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>     \
                                                                : dispatch<true, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>)   \
                                          : (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>    \
                                                                : dispatch<false, num_rdma_ranks, false, kNumTMABytesPerWarp, kNumDispatchRDMASenderWarps>); \
    SET_SHARED_MEMORY_FOR_TMA(dispatch_func);                                                                                                                \
    LAUNCH_KERNEL(                                                                                                                                           \
        &cfg,                                                                                                                                                \
        dispatch_func,                                                                                                                                       \
        reinterpret_cast<int4*>(recv_x),                                                                                                                     \
        recv_x_scales,                                                                                                                                       \
        recv_topk_idx,                                                                                                                                       \
        recv_topk_weights,                                                                                                                                   \
        reinterpret_cast<SourceMeta*>(recv_src_meta),                                                                                                        \
        reinterpret_cast<const int4*>(x),                                                                                                                    \
        x_scales,                                                                                                                                            \
        topk_idx,                                                                                                                                            \
        topk_weights,                                                                                                                                        \
        send_rdma_head,                                                                                                                                      \
        send_nvl_head,                                                                                                                                       \
        recv_rdma_channel_prefix_matrix,                                                                                                                     \
        recv_gbl_channel_prefix_matrix,                                                                                                                      \
        rdma_channel_prefix_matrix,                                                                                                                          \
        recv_rdma_rank_prefix_sum,                                                                                                                           \
        gbl_channel_prefix_matrix,                                                                                                                           \
        recv_gbl_rank_prefix_sum,                                                                                                                            \
        is_token_in_rank,                                                                                                                                    \
        num_tokens,                                                                                                                                          \
        hidden_int4,                                                                                                                                         \
        num_scales,                                                                                                                                          \
        num_topk,                                                                                                                                            \
        num_experts,                                                                                                                                         \
        scale_token_stride,                                                                                                                                  \
        scale_hidden_stride,                                                                                                                                 \
        rdma_buffer_ptr,                                                                                                                                     \
        num_max_rdma_chunked_send_tokens,                                                                                                                    \
        num_max_rdma_chunked_recv_tokens,                                                                                                                    \
        buffer_ptrs,                                                                                                                                         \
        num_max_nvl_chunked_send_tokens,                                                                                                                     \
        num_max_nvl_chunked_recv_tokens,                                                                                                                     \
        rank,                                                                                                                                                \
        num_ranks);                                                                                                                                          \
  }                                                                                                                                                          \
  break

  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
  GRPCOLL_HOST_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
  GRPCOLL_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < INT_MAX); // Make sure never OOB
  GRPCOLL_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  GRPCOLL_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));
  GRPCOLL_HOST_ASSERT(num_topk <= WARP_SIZE);
  // NOTES: in case of splitting, the issued put at the end of the buffer
  GRPCOLL_HOST_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

  // Even-numbered SMs for forwarders, odd-numbered SMs for RDMA senders and NVL receivers
  const int num_sms = num_channels * 2;
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Cached Notify Group Cast / Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
__global__ void cached_notify(
    const int rdma_clean_offset,
    const int rdma_num_int_clean,
    const int nvl_clean_offset,
    const int nvl_num_int_clean,
    int* combined_rdma_head,
    int num_combined_tokens,
    int num_channels,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    int* combined_nvl_head,
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    bool is_cached_dispatch,
    const nvshmem_team_t rdma_team) {
  const auto sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const auto nvl_rank = rank % NUM_MAX_NVL_PEERS, num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS, rdma_rank = rank / NUM_MAX_NVL_PEERS;

  if (sm_id == 0) { // the first SM is responsible to wait all previous inflight WRs finished and then clean the RDMA/NVL buffer
    // Wait until all previous inflight WRs for each QP of each RDMA peer are finished
    wait_all_inflight_wrs_finished<kLowLatencyMode>(num_threads, thread_id, num_rdma_ranks, rdma_rank, nvl_rank);

    // Barrier all first
    barrier_all<kLowLatencyMode, /*kSyncOnly=*/true>(thread_id, rdma_team, barrier_signal_ptrs, nvl_rank);

    // Clean RDMA buffer of this RDMA rank
    auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads)
      rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

    // Clean NVL buffer of this NVL rank
    auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads)
      nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

    __syncthreads();

    // Barrier all finally
    barrier_all<kLowLatencyMode, /*kSyncOnly=*/false>(thread_id, rdma_team, barrier_signal_ptrs, nvl_rank);
  } else if (sm_id == 1) { // the second SM is responsible to reset the rdma head before combine
    // If this is a cached dispatch,
    // no need to reset the rdma head, just return
    if (is_cached_dispatch)
      return;

    // Reset the rdma head, iterating in reverse order
    // each warp is responsible for one channel
    // and each lane in any warp is responsible for one rdma rank of the corr. channel
    if (lane_id < num_rdma_ranks and warp_id < num_channels) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_combined_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

      // NOTES: `1 << 25` is a heuristic large number
      int last_head = 1 << 25;
      for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
        auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
        if (current_head < 0) {
          combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
        } else {
          last_head = current_head;
        }
      }
    }
  } else { // the rest of SMs are responsible to reset the nvl head before combine
    // If this is a cached dispatch,
    // no need to reset the nvl head, just return
    if (is_cached_dispatch)
      return;

    if (warp_id < num_channels) {
      constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);
      constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
      constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;
      GRPCOLL_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");

      // TMA stuffs
      extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
      auto tma_buffer = smem_tma_buffer + warp_id * kNumTMABytesPerWarp;
      auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
      uint32_t tma_phase = 0;
      if (lane_id == 0) {
        mbarrier_init(tma_mbarrier, 1);
        fence_view_async_shared();
        fence_barrier_init();
      }
      __syncwarp();

      for (int dst_rdma_rank = sm_id - 2; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 2) {
        // Iterate in reverse order
        int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
        int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
        int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
        token_start_idx += shift, token_end_idx += shift;

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
        for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
          auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);

          if (lane_id == 0) {
            tma_load_1d(tma_buffer, combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS, tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
            mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
          }
          mbarrier_wait(tma_mbarrier, tma_phase);
          __syncwarp();

          for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
            if (lane_id < NUM_MAX_NVL_PEERS) {
              auto current_head = reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
              if (current_head < 0) {
                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
              } else {
                last_head = current_head;
              }
            }
          }
          tma_store_fence();
          __syncwarp();

          if (lane_id == 0)
            tma_store_1d(tma_buffer, combined_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
          tma_store_wait();
          __syncwarp();
        }
      }
    }
  }
}

void cached_notify(
    int hidden_int4,
    int num_scales,
    int num_topk_idx,
    int num_topk_weights,
    int num_ranks,
    int num_channels,
    int num_combined_tokens,
    int* combined_rdma_head,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    int* combined_nvl_head,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_recv_tokens,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes,
    bool is_cached_dispatch,
    bool low_latency_mode) {
  const int num_threads = std::max(128, WARP_SIZE * num_channels), num_warps = num_threads / WARP_SIZE;
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  const int kNumTMABytesPerWarp = 8192;
  const int smem_size = kNumTMABytesPerWarp * num_warps;
  const int num_sms = num_channels * 2;

  // Get clean meta
  auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
  auto nvl_clean_meta = get_nvl_clean_meta(
      hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels, is_cached_dispatch);
  GRPCOLL_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
  GRPCOLL_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
  // REVIEW: why limited to INT_MAX ?
  GRPCOLL_HOST_ASSERT(num_rdma_bytes < INT_MAX);
  GRPCOLL_HOST_ASSERT(num_nvl_bytes < INT_MAX);
  GRPCOLL_HOST_ASSERT(num_sms > 3); // REVIEW: why num_sms > 3 ?
  GRPCOLL_HOST_ASSERT(num_warps > 1); // for `barrier_all`
  if (!is_cached_dispatch) {
    // for rdma head reset before combine
    GRPCOLL_HOST_ASSERT(num_warps >= num_channels);
    GRPCOLL_HOST_ASSERT(num_rdma_ranks <= WARP_SIZE);

    // for nvl head reset before combine
    GRPCOLL_HOST_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
    GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= WARP_SIZE, "Too many NVL peers");
  }

  // Launch kernel
  auto cached_notify_func = low_latency_mode ? cached_notify<true, kNumTMABytesPerWarp> : cached_notify<false, kNumTMABytesPerWarp>;
  SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream);
  SET_SHARED_MEMORY_FOR_TMA(cached_notify_func);
  LAUNCH_KERNEL(
      &cfg,
      cached_notify_func,
      rdma_clean_meta.first,
      rdma_clean_meta.second,
      nvl_clean_meta.first,
      nvl_clean_meta.second,
      combined_rdma_head,
      num_combined_tokens,
      num_channels,
      rdma_channel_prefix_matrix,
      rdma_rank_prefix_sum,
      combined_nvl_head,
      rdma_buffer_ptr,
      buffer_ptrs,
      barrier_signal_ptrs,
      rank,
      num_ranks,
      is_cached_dispatch,
      cpu_rdma_team);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int kNumRanks,
    bool kMaybeWithBias,
    typename dtype_t,
    int kMaxNumRanks,
    bool kUseTMA,
    int kNumStages,
    int kNumTMALoadBytes = 0,
    typename GetAddrFn,
    typename ReceiveTWFn>
__device__ int combine_token(
    bool is_token_in_rank,
    int head_idx,
    int lane_id,
    int hidden_int4,
    int num_topk,
    int4* combined_row,
    float* combined_topk_weights,
    const int4* bias_0_int4,
    const int4* bias_1_int4,
    int num_max_recv_tokens,
    const GetAddrFn& get_addr_fn,
    const ReceiveTWFn& recv_tw_fn,
    uint8_t* smem_ptr,
    uint32_t (&tma_phase)[kNumStages]) {
  constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

  // Broadcast current heads
  // Lane `i` holds the head of rank `i` and `is_token_in_rank`
  GRPCOLL_STATIC_ASSERT(kMaxNumRanks <= WARP_SIZE, "Too many ranks");
  int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
#pragma unroll
  for (int i = 0; i < kNumRanks; ++i)
    if (broadcast_in_warp(/*val=*/is_token_in_rank, /*src_lane=*/i)) {
      slot_indices[num_topk_ranks] = broadcast_in_warp(/*val=*/head_idx, /*src_lane=*/i) % num_max_recv_tokens;
      topk_ranks[num_topk_ranks++] = i;
    }
  GRPCOLL_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);
  GRPCOLL_STATIC_ASSERT(not(kUseTMA and kMaybeWithBias), "TMA cannot be used by receiver warps");
  GRPCOLL_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages now");

  // Reduce data
  if constexpr (kUseTMA) {
    constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;

    auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
      return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);
    };
    auto tma_store_buffer = [=](const int& i) -> int4* {
      return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + NUM_MAX_NVL_PEERS * kNumTMALoadBytes);
    };
    auto tma_mbarrier = [=](const int& i) -> uint64_t* {
      return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);
    };

    // Prefetch
    if (lane_id < num_topk_ranks)
      tma_load_1d(tma_load_buffer(0, lane_id), get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0), tma_mbarrier(0), kNumTMALoadBytes);
    mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
    __syncwarp();

    for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += WARP_SIZE, iter += 1) {
      const int stage_idx = iter % kNumStages;
      const int next_stage_idx = (iter + 1) % kNumStages;

      // Prefetch next stage
      if (shifted + WARP_SIZE < hidden_int4) {
        if (lane_id < num_topk_ranks)
          tma_load_1d(
              tma_load_buffer(next_stage_idx, lane_id),
              get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + WARP_SIZE),
              tma_mbarrier(next_stage_idx),
              kNumTMALoadBytes);
        mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();
      }

      mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);
      float values[kDtypePerInt4] = {0};
#pragma unroll
      for (int j = 0; j < num_topk_ranks; ++j) {
        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
#pragma unroll
        for (int k = 0; k < kDtypePerInt4; ++k)
          values[k] += static_cast<float>(recv_value_dtypes[k]);
      }

      tma_store_wait<kNumStages - 1>();
      auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
#pragma unroll
      for (int j = 0; j < kDtypePerInt4; ++j)
        out_dtypes[j] = static_cast<dtype_t>(values[j]);
      tma_store_fence();
      __syncwarp();

      if (lane_id == 0)
        tma_store_1d(tma_store_buffer(stage_idx), combined_row + shifted + lane_id, kNumTMALoadBytes);
      __syncwarp();
    }

    // Flush all writes
    tma_store_wait();
  } else {
#pragma unroll
    for (int i = lane_id; i < hidden_int4; i += WARP_SIZE) {
      // Read bias
      // TODO: make it as a finer-grained template
      int4 bias_0_value_int4, bias_1_value_int4;
      if constexpr (kMaybeWithBias) {
        bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
        bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
      }

      // Read buffers
      // TODO: maybe too many registers here
      int4 recv_value_int4[kMaxNumRanks];
#pragma unroll
      for (int j = 0; j < num_topk_ranks; ++j)
        recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));

      // Clean
      // Reduce bias
      float values[kDtypePerInt4] = {0};
      if constexpr (kMaybeWithBias) {
        auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
        auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
#pragma unroll
        for (int j = 0; j < kDtypePerInt4; ++j)
          values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
      }

// Reduce all-to-all results
#pragma unroll
      for (int j = 0; j < num_topk_ranks; ++j) {
        auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
#pragma unroll
        for (int k = 0; k < kDtypePerInt4; ++k)
          values[k] += static_cast<float>(recv_value_dtypes[k]);
      }

      // Cast back to `dtype_t` and write
      int4 out_int4;
      auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
      for (int j = 0; j < kDtypePerInt4; ++j)
        out_dtypes[j] = static_cast<dtype_t>(values[j]);
      st_na_global(combined_row + i, out_int4);
    }
  }

  // Reduce `topk_weights`
  if (lane_id < num_topk) {
    float value = 0;
#pragma unroll
    for (int i = 0; i < num_topk_ranks; ++i)
      value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
    st_na_global(combined_topk_weights + lane_id, value);
  }

  // Return the minimum top-k rank
  return topk_ranks[0];
}

// FIXME: the register usage is highly spilled for both load/store
template <
    bool kLowLatencyMode,
    int kNumRDMARanks,
    typename dtype_t,
    int kNumCombineForwarderWarps,
    int kNumTMABytesPerSenderWarp,
    int kNumTMABytesPerForwarderWarp,
    int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
    int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
    int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
    int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS>
GLOBAL_LAUNCH_BOUNDS(get_num_threads_combine(kNumForwarders), 1)
void combine(
    int4* combined_x,
    float* combined_topk_weights,
    const bool* is_combined_token_in_rank,
    const int4* x,
    const float* topk_weights,
    const int4* bias_0,
    const int4* bias_1,
    const int* combined_rdma_head,
    const int* combined_nvl_head,
    const SourceMeta* src_meta,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    int num_tokens,
    int num_combined_tokens,
    int hidden_size,
    int num_topk,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks) {
  const auto sm_id = static_cast<int>(blockIdx.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();
  const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const bool is_forwarder = sm_id % 2 == 1;

  const auto hidden_int4 = hidden_size / (sizeof(int4) / sizeof(dtype_t)), hidden_bytes = hidden_int4 * sizeof(int4);
  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, /*num_scales=*/0, /*num_topk_idx=*/0, num_topk);
  const auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;

  /** NOTE: Determine warp role and its target warp id
   * For Forwarder (Odd SMs):
   *  1. the first `kNumForwarders` warps are `kNVLAndRDMAForwarder`, ...
   *  2. the rest warp is `kCoordinator`, ...
   *
   * For Sender/Receiver (Even SMs):
   *  1. the first `NUM_MAX_NVL_PEERS` warps are `kNVLSender`, ...
   *  2. the next `kNumRDMAReceivers` warps are `kRDMAReceiver`, ...
   *  3. the rest warp is `kCoordinator`, ...
   */
  enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };
  auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (!is_forwarder) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kNVLSender, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else if (warp_id < kNumForwarders) {
        return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    } else {
      if (warp_id < kNumForwarders) {
        return {WarpRole::kNVLAndRDMAForwarder, (warp_id + channel_id) % kNumForwarders};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    }
  }();
  const auto warp_role = role_meta.first;
  const auto target_warp_id = role_meta.second;

  // Warp-specialized working
  if (warp_role == WarpRole::kNVLSender) {
    // NVL producers
    const auto dst_nvl_rank = target_warp_id;

    // NVL layouts
    // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
    auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
    auto nvl_channel_x =
        AsymBuffer<uint8_t>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
            .advance_also(local_buffer_ptr);
    auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank).advance_also(dst_buffer_ptr);
    auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_buffer_ptr);

    // TMA stuffs
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
    auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + hidden_bytes);
    uint32_t tma_phase = 0;
    if (lane_id == 0) {
      mbarrier_init(tma_mbarrier, 1);
      fence_view_async_shared();
      fence_barrier_init();
      GRPCOLL_DEVICE_ASSERT(hidden_bytes + sizeof(uint64_t) <= kNumTMABytesPerSenderWarp);
    }
    __syncwarp();

    // Get tasks for each RDMA lane
    int token_start_idx = 0, token_end_idx = 0;
    if (lane_id < kNumRDMARanks) {
      int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
      token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
      token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
    }
    __syncwarp();

    // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= WARP_SIZE, "Invalid number of RDMA peers");

    // Iterate over all tokens and send by chunks
    int current_rdma_idx = channel_id % kNumRDMARanks;
    while (true) {
      // Exit if possible
      if (all_in_warp(token_start_idx >= token_end_idx))
        break;

      // Decide the next RDMA buffer to send
      bool is_lane_ready = false;
      auto start_time = clock64();
      while (true) {
        int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
        is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
            num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
        if (any_in_warp(is_lane_ready))
          break;

        // Retry
        if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
          cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
          printf(
              "grpcoll combine NVL sender timeout, channel: %d, RDMA: %d, NVL: %d, dst NVL rank: %d, RDMA lane: %d, head: %d, tail: %d, start: %d, end: %d\n",
              channel_id,
              rdma_rank,
              nvl_rank,
              dst_nvl_rank,
              lane_id,
              ld_volatile_global(nvl_channel_head.buffer() + lane_id),
              cached_channel_tail_idx,
              token_start_idx,
              token_end_idx);
          trap();
        }
      }

      // Sync token start index and count
      for (int i = 0; i < kNumRDMARanks; ++i) {
        current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;
        if (broadcast_in_warp(/*val=*/(token_start_idx >= token_end_idx) or (not is_lane_ready), /*src_lane=*/current_rdma_idx))
          continue;

        // Sync token start index
        auto token_idx = static_cast<int64_t>(broadcast_in_warp(/*val=*/token_start_idx, /*src_lane=*/current_rdma_idx));
        int num_tokens_in_chunk = broadcast_in_warp(/*val=*/min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), /*src_lane=*/current_rdma_idx);

        // Send by chunk
        for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
          // Get an empty slot
          int dst_slot_idx = 0;
          if (lane_id == current_rdma_idx) {
            dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
            dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
          }
          dst_slot_idx = broadcast_in_warp(/*val=*/dst_slot_idx, /*src_lane=*/current_rdma_idx);

          // Load data
          auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
          auto shifted_x = x + token_idx * hidden_int4;
          if (lane_id == 0) {
            tma_store_wait();
            tma_load_1d(tma_buffer, shifted_x, tma_mbarrier, hidden_bytes);
            mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
          }
          __syncwarp();
          mbarrier_wait(tma_mbarrier, tma_phase);

          // Load source meta
          if (lane_id == num_topk)
            *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

          // Load `topk_weights`
          if (lane_id < num_topk)
            *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

          // Issue TMA store
          tma_store_fence();
          __syncwarp();
          if (lane_id == 0)
            tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
        }
        lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
      }

      // Move queue tail
      tma_store_wait();
      __syncwarp();
      if (lane_id < kNumRDMARanks and is_lane_ready)
        st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
    }
  } else { // warp_role == WarpRole::kNVLAndRDMAForwarder | warp_role == WarpRole::kRDMAReceiver | warp_role == WarpRole::kCoordinator
    // Combiners and coordinators
    // RDMA symmetric layout
    auto rdma_channel_data = SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    // NVL layouts
    void* local_nvl_buffer = buffer_ptrs[nvl_rank];
    void* nvl_buffers[NUM_MAX_NVL_PEERS];
#pragma unroll
    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
      nvl_buffers[i] = buffer_ptrs[i];
    auto nvl_channel_x = AsymBuffer<uint8_t>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, channel_id, num_channels)
                             .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_head =
        AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank).advance_also(local_nvl_buffer);
    auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels).advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

    // Combiner warp synchronization
    __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
    __shared__ volatile bool forwarder_retired[kNumForwarders];
    __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
    __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
    auto sync_forwarder_smem = [=]() { sync_warp_group(/*group_flag=*/0, /*group_size=*/(kNumForwarders + 1) * WARP_SIZE); };
    auto sync_rdma_receiver_smem = [=]() { sync_warp_group(/*group_flag=*/1, /*group_size=*/(kNumRDMAReceivers + 1) * WARP_SIZE); };

    if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
      // Receive from NVL ranks and forward to RDMA ranks
      // NOTES: this part is using "large warps" for each RDMA ranks
      const auto dst_rdma_rank = target_warp_id / kNumWarpsPerForwarder, sub_warp_id = target_warp_id % kNumWarpsPerForwarder;
      auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
      auto sync_large_warp = [=]() {
        if (kNumWarpsPerForwarder == 1) {
          __syncwarp();
        } else {
          sync_warp_group(/*group_flag=*/dst_rdma_rank + 2, /*group_size=*/kNumWarpsPerForwarder * WARP_SIZE);
        }
      };
      GRPCOLL_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

      // TMA stuffs
      constexpr int kNumStages = 2;
      constexpr int kNumTMALoadBytes = sizeof(int4) * WARP_SIZE;
      constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
      GRPCOLL_STATIC_ASSERT(kNumTMABufferBytesPerStage * kNumStages <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough");

      extern __shared__ __align__(1024) uint8_t smem_buffer[];
      auto smem_ptr = smem_buffer + target_warp_id * kNumStages * kNumTMABufferBytesPerStage;
      auto tma_mbarrier = [=](const int& i) {
        return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
      };
      uint32_t tma_phase[kNumStages] = {0};
      if (lane_id < kNumStages) {
        mbarrier_init(tma_mbarrier(lane_id), WARP_SIZE);
        fence_view_async_shared();
        fence_barrier_init();
      }
      __syncwarp();

      // Advance to the corresponding NVL buffer
      nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
      nvl_channel_head.advance(dst_rdma_rank);
      nvl_channel_tail.advance(dst_rdma_rank);

      // Clean shared memory and sync
      GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= WARP_SIZE, "Invalid number of NVL peers");
      lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[target_warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (forwarder_retired[target_warp_id] = false) : false;
      sync_forwarder_smem();

      // Get count and cached head
      int cached_nvl_channel_tail_idx = 0;
      int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
      int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
      num_tokens_to_combine -= num_tokens_prefix;
      num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
      combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

      // Iterate over all tokens and combine by chunks
      for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
        // Check destination queue emptiness, or wait a buffer to be released
        auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
        auto num_chunked_tokens = token_end_idx - token_start_idx;
        auto start_time = clock64();
        while (sub_warp_id == 0 and lane_id == 0) {
          // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
          // Here, `token_start_idx` is the actual tail
          int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
          if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
            break;

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "grpcoll combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, NVL: %d, dst RDMA rank: %d, head: %ld, tail: %d, chunked: %d\n",
                channel_id,
                rdma_rank,
                nvl_rank,
                dst_rdma_rank,
                ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)),
                token_start_idx,
                num_chunked_tokens);
            trap();
          }
        }
        sync_large_warp();

        // Combine and write to the RDMA buffer
        for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
          // Read expected head
          GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= WARP_SIZE, "Invalid number of RDMA peers");
          int expected_head = -1;
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);

          // Wait lanes to be ready
          start_time = clock64();
          while (cached_nvl_channel_tail_idx <= expected_head) {
            cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
              printf(
                  "grpcoll combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, NVL: %d, src NVL rank: %d, dst RDMA rank: %d, tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                  channel_id,
                  rdma_rank,
                  nvl_rank,
                  lane_id,
                  dst_rdma_rank,
                  cached_nvl_channel_tail_idx,
                  token_idx,
                  num_tokens_to_combine,
                  sub_warp_id,
                  kNumWarpsPerForwarder,
                  expected_head);
              trap();
            }
          }

          // Combine current token
          auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
          void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
          auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
            return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) + hidden_int4_idx;
          };
          auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
            return ld_nc_global(
                reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) + topk_idx);
          };
          combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes>(
              expected_head >= 0,
              expected_head,
              lane_id,
              hidden_int4,
              num_topk,
              static_cast<int4*>(shifted),
              reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
              nullptr,
              nullptr,
              num_max_nvl_chunked_recv_tokens_per_rdma,
              get_addr_fn,
              recv_tw_fn,
              smem_ptr,
              tma_phase);

          // Update head
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head < 0 ? (forwarder_nvl_head[target_warp_id][lane_id] = -expected_head - 1) : (forwarder_nvl_head[target_warp_id][lane_id] = expected_head + 1);
        }
        sync_large_warp();

        // Issue RDMA send
        if (sub_warp_id == kNumWarpsPerForwarder - 1) {
          if (dst_rdma_rank != rdma_rank) {
            auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
            const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
            const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
            nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg, get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank), channel_id, lane_id, 0);
          } else {
            memory_fence();
          }

          // Write new RDMA tail
          __syncwarp();
          if (lane_id == 0) {
            nvshmemi_ibgda_amo_nonfetch_add(
                rdma_channel_tail.buffer(rdma_rank),
                num_chunked_tokens,
                get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                channel_id,
                dst_rdma_rank == rdma_rank);
          }
        }
      }

      // Retired
      __syncwarp();
      if (lane_id == 0)
        forwarder_retired[target_warp_id] = true;
    } else if (warp_role == WarpRole::kRDMAReceiver) {
      // Receive from RDMA ranks and write to the output tensor
      // Clean shared memory and sync
      GRPCOLL_DEVICE_ASSERT(kNumRDMARanks <= WARP_SIZE);
      lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[target_warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (rdma_receiver_retired[target_warp_id] = false) : 0;
      sync_rdma_receiver_smem();

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over all tokens and combine
      int cached_channel_tail_idx = 0;
      for (int64_t token_idx = token_start_idx + target_warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
        // Read expected head
        GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= WARP_SIZE, "Invalid number of RDMA peers");
        int expected_head = -1;
        if (lane_id < kNumRDMARanks) {
          expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
          (expected_head < 0) ? (rdma_receiver_rdma_head[target_warp_id][lane_id] = -expected_head - 1)
                              : (rdma_receiver_rdma_head[target_warp_id][lane_id] = expected_head);
        }

        // Wait lanes to be ready
        auto start_time = clock64();
        while (cached_channel_tail_idx <= expected_head) {
          cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "grpcoll combine RDMA receiver timeout, channel: %d, RDMA: %d, NVL: %d, src RDMA rank: %d, tail: %d, waiting: %ld, expect: %d\n",
                channel_id,
                rdma_rank,
                nvl_rank,
                lane_id,
                cached_channel_tail_idx,
                token_idx,
                expected_head);
            trap();
          }
        }
        __syncwarp();

        // Combine current token
        auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
          return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) + hidden_int4_idx;
        };
        auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
          return ld_nc_global(
              reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
              topk_idx);
        };
        uint32_t dummy_tma_phases[2];
        combine_token<kNumRDMARanks, true, dtype_t, kNumTopkRDMARanks, false, 2>(
            expected_head >= 0,
            expected_head,
            lane_id,
            hidden_int4,
            num_topk,
            combined_x + token_idx * hidden_int4,
            combined_topk_weights + token_idx * num_topk,
            bias_0 == nullptr ? nullptr : bias_0 + token_idx * hidden_int4,
            bias_1 == nullptr ? nullptr : bias_1 + token_idx * hidden_int4,
            num_max_rdma_chunked_recv_tokens,
            get_addr_fn,
            recv_tw_fn,
            nullptr,
            dummy_tma_phases);
      }

      // Retired
      __syncwarp();
      if (lane_id == 0)
        rdma_receiver_retired[target_warp_id] = true;
    } else {
      // Coordinator
      // Sync shared memory status
      is_forwarder ? sync_forwarder_smem() : sync_rdma_receiver_smem();
      const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

      int last_rdma_head = 0;
      int last_nvl_head[kNumRDMARanks] = {0};
      int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
      int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
      GRPCOLL_STATIC_ASSERT(kNumCombineForwarderWarps <= WARP_SIZE, "Invalid number of forwarder warps");
      while (true) {
        // Retired
        if (not is_forwarder and all_in_warp(lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
          break;
        if (is_forwarder and all_in_warp(lane_id >= kNumForwarders or forwarder_retired[lane_id]))
          break;

        // Find minimum head for RDMA ranks
        if (!is_forwarder) {
          int min_head = INT_MAX;
#pragma unroll
          for (int i = 0; i < kNumRDMAReceivers; ++i)
            if (not rdma_receiver_retired[i])
              min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
          if (min_head != INT_MAX and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
            nvshmemi_ibgda_amo_nonfetch_add(
                rdma_channel_head.buffer(rdma_rank),
                min_head - last_rdma_head,
                get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                channel_id + num_channels,
                dst_rdma_rank == rdma_rank);
            last_rdma_head = min_head;
          }
        } else {
// Find minimum head for NVL ranks
#pragma unroll
          for (int i = 0; i < kNumRDMARanks; ++i) {
            int min_head = INT_MAX;
#pragma unroll
            for (int j = 0; j < num_warps_per_rdma_rank; ++j)
              if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
            if (min_head != INT_MAX and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS)
              st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
          }
        }

        // Nanosleep and let other warps work
        __nanosleep(NUM_WAIT_NANOSECONDS);
      }
    }
  }
}

void combine(
    cudaDataType_t type,
    void* combined_x,
    float* combined_topk_weights,
    const bool* is_combined_token_in_rank,
    const void* x,
    const float* topk_weights,
    const void* bias_0,
    const void* bias_1,
    const int* combined_rdma_head,
    const int* combined_nvl_head,
    const void* src_meta,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    int num_tokens,
    int num_combined_tokens,
    int hidden_size,
    int num_topk,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_channels,
    bool low_latency_mode) {
  constexpr int kNumCombineForwarderWarps = 24;
  constexpr int kNumTMABytesPerSenderWarp = 16384;
  constexpr int kNumTMABytesPerForwarderWarp = 9248; // REVIEW: why this unusual number ?
  constexpr int smem_size = std::max(kNumTMABytesPerSenderWarp * NUM_MAX_NVL_PEERS, kNumTMABytesPerForwarderWarp * kNumCombineForwarderWarps);

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                                                                                \
  {                                                                                                                                        \
    auto combine_func = low_latency_mode                                                                                                   \
        ? combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps, kNumTMABytesPerSenderWarp, kNumTMABytesPerForwarderWarp>   \
        : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps, kNumTMABytesPerSenderWarp, kNumTMABytesPerForwarderWarp>; \
    SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                               \
    LAUNCH_KERNEL(                                                                                                                         \
        &cfg,                                                                                                                              \
        combine_func,                                                                                                                      \
        reinterpret_cast<int4*>(combined_x),                                                                                               \
        combined_topk_weights,                                                                                                             \
        is_combined_token_in_rank,                                                                                                         \
        reinterpret_cast<const int4*>(x),                                                                                                  \
        topk_weights,                                                                                                                      \
        reinterpret_cast<const int4*>(bias_0),                                                                                             \
        reinterpret_cast<const int4*>(bias_1),                                                                                             \
        combined_rdma_head,                                                                                                                \
        combined_nvl_head,                                                                                                                 \
        reinterpret_cast<const SourceMeta*>(src_meta),                                                                                     \
        rdma_channel_prefix_matrix,                                                                                                        \
        rdma_rank_prefix_sum,                                                                                                              \
        gbl_channel_prefix_matrix,                                                                                                         \
        num_tokens,                                                                                                                        \
        num_combined_tokens,                                                                                                               \
        hidden_size,                                                                                                                       \
        num_topk,                                                                                                                          \
        rdma_buffer_ptr,                                                                                                                   \
        num_max_rdma_chunked_send_tokens,                                                                                                  \
        num_max_rdma_chunked_recv_tokens,                                                                                                  \
        buffer_ptrs,                                                                                                                       \
        num_max_nvl_chunked_send_tokens,                                                                                                   \
        num_max_nvl_chunked_recv_tokens,                                                                                                   \
        rank,                                                                                                                              \
        num_ranks);                                                                                                                        \
  }                                                                                                                                        \
  break

  const int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  const auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
  const int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
  const int num_threads = get_num_threads_combine(num_forwarder_warps), num_warps = num_threads / WARP_SIZE;
  GRPCOLL_HOST_ASSERT(num_rdma_ranks <= kNumCombineForwarderWarps);
  GRPCOLL_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % num_rdma_ranks == 0);
  GRPCOLL_HOST_ASSERT(num_warps == num_forwarder_warps + 1);
  GRPCOLL_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  GRPCOLL_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks > std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
  GRPCOLL_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);
  GRPCOLL_HOST_ASSERT(num_topk <= WARP_SIZE);
  GRPCOLL_HOST_ASSERT(type == CUDA_R_16BF);
  GRPCOLL_HOST_ASSERT(hidden_size % (sizeof(int4) / sizeof(nv_bfloat16)) == 0);

  // Even-numbered SMs for NVL senders and RDMA receivers, odd-numbered SMs for forwarders
  const int num_sms = num_channels * 2;
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream);
  SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode

} // namespace magi_attn_comm::grpcoll
