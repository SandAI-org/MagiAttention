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

#pragma once

#include <functional>
#include <optional>

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"
#include "reduce_op.cuh"
#include "utils.cuh"

namespace magi_attn_comm::grpcoll::internode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Source Meta
///////////////////////////////////////////////////////////////////////////////////////////////////

struct SourceMeta {
  // `src_rdma_rank`: the src RDMA peer to return to in group reduce stage
  // `is_token_in_nvl_rank_bits`: whether the token is in each NVL peer
  // REVIEW: why we need to keep the `is_token_in_nvl_rank_bits`,
  // instead of just keeping the src NVL peer directly ?
  int src_rdma_rank, is_token_in_nvl_rank_bits;

  GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

  __forceinline__ SourceMeta() = default;

  // TODO: faster encoding
  DEVICE_INLINE SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
    src_rdma_rank = rdma_rank;
    is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
#pragma unroll
    for (int r = 1; r < NUM_MAX_NVL_PEERS; ++r)
      is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[r] << r;
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

constexpr int static_min(const int a, const int b) {
  return a < b ? a : b;
}

constexpr int static_max(const int a, const int b) {
  return a > b ? a : b;
}

constexpr int get_num_max_src_rdma_ranks(const int num_rdma_ranks) {
  // REVIEW: why at most 8 RDMA ranks can be sent to in the original code ?
  // return static_min(num_rdma_ranks, NUM_MAX_NVL_PEERS);
  return num_rdma_ranks;
}

constexpr int get_num_threads_group_cast(const int num_group_cast_rdma_sender_warps) {
  return (num_group_cast_rdma_sender_warps + 1 + NUM_MAX_NVL_PEERS) * WARP_SIZE;
}

constexpr int get_num_threads_group_reduce(const int num_group_reduce_forwarder_warps) {
  return (num_group_reduce_forwarder_warps + 1) * WARP_SIZE;
}

HOST_DEVICE_INLINE int get_num_bytes_per_token(int hidden_int4, int num_heads) {
  return static_cast<int>(align(
      /*hidden_states=*/hidden_int4 * sizeof(int4) +
          /*lse*/ num_heads * sizeof(float) +
          /*source_meta=*/sizeof(SourceMeta),
      sizeof(int4)));
}

HOST_DEVICE_INLINE std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_heads, int num_rdma_ranks, int num_rdma_recv_buffer_tokens, int num_channels) {
  // Return `int32_t` offset and count to clean
  return {
      (get_num_bytes_per_token(hidden_int4, num_heads) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_channels) / sizeof(int),
      (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

HOST_DEVICE_INLINE std::pair<int, int> get_nvl_clean_meta(
    int hidden_int4,
    int num_heads,
    int num_rdma_ranks,
    int num_nvl_ranks,
    int num_nvl_recv_buffer_tokens,
    int num_channels,
    bool is_group_cast) {
  GRPCOLL_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

  // Return `int32_t` offset and to clean
  return {
      (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_heads) * num_nvl_ranks * num_channels) / sizeof(int),
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
  if (thread_id == WARP_SIZE) // REVIEW: why we need the second warp here ?
    nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
  barrier_block<NUM_MAX_NVL_PEERS, kSyncOnly>(barrier_signal_ptrs, nvl_rank);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast Timeout Check
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE void timeout_check_rdma_sender(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_rdma_rank,
    const int head,
    const int tail) {
  if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast RDMA sender timeout, channel: %d, RDMA: %d, NVL: %d, dst RDMA rank: %d, head: %d, tail: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_rdma_rank,
        head,
        tail);
    trap();
  }
}

DEVICE_INLINE void timeout_check_rdma_sender_coordinator(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_rdma_rank,
    const int tail,
    const int remain_tokens_to_send) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast RDMA sender coordinator timeout, channel: %d, RDMA: %d, NVL %d, dst RDMA: %d, tail: %d, remaining: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_rdma_rank,
        tail,
        remain_tokens_to_send);
    trap();
  }
}

DEVICE_INLINE void timeout_check_rdma2nvl_forwarder_rdma_meta(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int src_rdma_rank,
    const int dst_nvl_rank,
    const int nvl_token_start_idx_encoded,
    const int nvl_token_end_idx_encoded,
    const int rdma_token_start_idx_encoded,
    const int rdma_token_end_idx_encoded) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast RDMA and NVL forwarder timeout (RDMA meta), channel: %d, RDMA: %d, NVL: %d, src RDMA rank: %d, dst NVL rank: %d, encoded meta: (nvl start: %d, nvl end: %d, rdma start: %d, rdma end: %d)\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        src_rdma_rank,
        dst_nvl_rank,
        nvl_token_start_idx_encoded,
        nvl_token_end_idx_encoded,
        rdma_token_start_idx_encoded,
        rdma_token_end_idx_encoded);
    trap();
  }
}

DEVICE_INLINE void timeout_check_rdma2nvl_forwarder_nvl_head(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_nvl_rank,
    const int head,
    const int tail) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast RDMA and NVL forwarder timeout (NVL head), channel: %d, RDMA: %d, NVL: %d, dst NVL rank: %d, head: %d, tail: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_nvl_rank,
        head,
        tail);
    trap();
  }
}

DEVICE_INLINE void timeout_check_rdma2nvl_forwarder_rdma_head(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_nvl_rank,
    const int src_rdma_lane,
    const int head,
    const int tail,
    const int expected_num_tokens_to_recv) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast RDMA and NVL forwarder timeout (RDMA head), channel: %d, RDMA: %d, NVL: %d, dst NVL: %d, src RDMA lane: %d, head: %d, tail: %d, expected: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_nvl_rank,
        src_rdma_lane,
        head,
        tail,
        expected_num_tokens_to_recv);
    trap();
  }
}

DEVICE_INLINE void timeout_check_nvl_receiver_meta(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int src_rdma_rank,
    const int src_nvl_rank,
    const int start_offset,
    const int end_offset) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast NVL receiver timeout (meta), channel: %d, RDMA: %d, NVL: %d, src RDMA rank: %d, src NVL rank: %d, start: %d, end: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        src_rdma_rank,
        src_nvl_rank,
        start_offset,
        end_offset);
    trap();
  }
}

DEVICE_INLINE void timeout_check_nvl_receiver_tail(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int src_nvl_rank,
    const int head,
    const int tail) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_cast NVL receiver timeout (tail), channel: %d, RDMA: %d, NVL: %d, src NVL rank: %d, head: %d, tail: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        src_nvl_rank,
        head,
        tail);
    trap();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce Timeout Check
///////////////////////////////////////////////////////////////////////////////////////////////////

DEVICE_INLINE void timeout_check_nvl_sender(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_nvl_rank,
    const int src_rdma_lane,
    const int head,
    const int tail,
    const int token_start_idx,
    const int token_end_idx) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_reduce NVL sender timeout, channel: %d, RDMA: %d, NVL: %d, dst NVL rank: %d, src RDMA lane: %d, head: %d, tail: %d, start: %d, end: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_nvl_rank,
        src_rdma_lane,
        head,
        tail,
        token_start_idx,
        token_end_idx);
    trap();
  }
}

DEVICE_INLINE void timeout_check_nvl2rdma_forwarder_rdma_head(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int dst_rdma_rank,
    const uint64_t head,
    const int tail,
    const int num_chunked_tokens) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_reduce forwarder (RDMA head) timeout, channel: %d, RDMA: %d, NVL: %d, dst RDMA rank: %d, head: %ld, tail: %d, chunked: %d\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        dst_rdma_rank,
        head,
        tail,
        num_chunked_tokens);
    trap();
  }
}

DEVICE_INLINE void timeout_check_nvl2rdma_forwarder_nvl_head(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int src_nvl_rank,
    const int dst_rdma_rank,
    const int tail,
    const int token_idx,
    const int num_tokens_to_reduce,
    const int warp_idx_in_group,
    const int expected_head) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_reduce forwarder (NVL head) timeout, channel: %d, RDMA: %d, NVL: %d, src NVL rank: %d, dst RDMA rank: %d, tail: %d, token_info: (token idx: %d, num tokens: %d, warp idx: %d, expected head: %d)\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        src_nvl_rank,
        dst_rdma_rank,
        tail,
        token_idx,
        num_tokens_to_reduce,
        warp_idx_in_group,
        expected_head);
    trap();
  }
}

DEVICE_INLINE void timeout_check_rdma_recevier(
    const int64_t start_time,
    const int channel_id,
    const int rdma_rank,
    const int nvl_rank,
    const int src_rdma_lane,
    const int tail,
    const int token_idx,
    const int expected_head) {
  if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
    printf(
        "grpcoll group_reduce RDMA receiver timeout, channel: %d, RDMA: %d, NVL: %d, src RDMA lane: %d, tail: %d, token_info: (token idx: %d, expected head: %d)\n",
        channel_id,
        rdma_rank,
        nvl_rank,
        src_rdma_lane,
        tail,
        token_idx,
        expected_head);
    trap();
  }
}

} // namespace magi_attn_comm::grpcoll::internode
