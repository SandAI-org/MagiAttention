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
  // NOTE: in the original code, it uses at most 8 RDMA ranks can be sent to
  // which is probably due to highly spilled registers (at most 3KB) as `num_rdma_ranks` grows
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce Helper Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename reduce_dtype_t, bool kAccReduce, int kMaxNumHeads, int kNumReduceWarps, typename GetLSEFn>
__device__ void reduce_lse_in_warp(
    float* reduced_lse,
    const GetLSEFn& get_lse_fn,
    reduce_dtype_t shared_reduced_lse[kNumReduceWarps][kMaxNumHeads],
    reduce_dtype_t shared_old_lse_rescale_weight[kNumReduceWarps][kMaxNumHeads],
    int* src_rank_idxs,
    int* slot_indices,
    int num_src_ranks,
    int reduce_warp_id,
    int lane_id,
    int num_heads) {
#pragma unroll
  // Reduce lse from all src ranks
  for (int h = lane_id; h < num_heads; h += WARP_SIZE) {
    auto reduced_lse_ptr = reduced_lse + h;

    // Initialize the high-precision reduce buffer
    reduce_dtype_t reduced_lse_val, old_lse_val;
    if constexpr (kAccReduce) { // if in `kAccReduce` mode, initialize `reduced_lse` with the old value
      reduced_lse_val = old_lse_val = static_cast<reduce_dtype_t>(*reduced_lse_ptr);
    } else { // else, initialize `reduced_lse` with -inf
      reduced_lse_val = get_neg_inf<reduce_dtype_t>();
    }

#pragma unroll
    // Apply lse reduce for each src rank
    for (int j = 0; j < num_src_ranks; ++j) {
      lse_reduce<reduce_dtype_t, float>(/*reduced_lse=*/reduced_lse_val, /*src_lse=*/get_lse_fn(src_rank_idxs[j], slot_indices[j], h));
    }

    // Store the reduced lse to shared memory buffer temporarily
    // which will be read later to reduce the hidden values
    shared_reduced_lse[reduce_warp_id][h] = reduced_lse_val;

    // Store the rescale weight of old `reduced_lse` for each head
    // which will be read later to reduce the hidden values if in `kAccReduce` mode
    if constexpr (kAccReduce) {
      shared_old_lse_rescale_weight[reduce_warp_id][h] = get_lse_rescale_weight(/*lse_to_rescale=*/old_lse_val, /*rescaled_lse=*/reduced_lse_val);
    }

    // Store reduced lse to output buffer
    st_na_global(reduced_lse_ptr, static_cast<float>(reduced_lse_val)); // non-cached store
  }
}

template <
    typename dtype_t,
    typename comm_dtype_t,
    typename reduce_dtype_t,
    ReduceOp kReduceOp,
    bool kAccReduce,
    int kMaxNumHeads,
    int kNumReduceWarps,
    int kCommDtypePerInt4,
    typename ReceiveHidValFn,
    typename GetLSEFn>
__device__ void reduce_hidval_in_warp(
    int reduce_warp_id,
    int head_idx,
    int num_src_ranks,
    int num_ranks_to_reduce,
    int4* reduce_hidval_ptr_int4,
    reduce_dtype_t* hp_hidval_reduce_buf,
    int* src_rank_idxs,
    int* slot_indices,
    const ReceiveHidValFn& recv_hidval_int4_fn,
    const GetLSEFn& get_lse_fn,
    reduce_dtype_t shared_reduced_lse[kNumReduceWarps][kMaxNumHeads],
    reduce_dtype_t shared_old_lse_rescale_weight[kNumReduceWarps][kMaxNumHeads]) {
  constexpr bool kIsLSEReduce = kReduceOp == ReduceOp::LSE;

  // Initialize the high-precision reduce buffer
  if constexpr (kAccReduce) { // if in `kAccReduce` mode
    // Initialize the high-precision reduce buffer
    // with the old value in `reduced_token`
    auto reduce_hidval_ptr_dtype = reinterpret_cast<const dtype_t*>(reduce_hidval_ptr_int4);
    foreach_assign<reduce_dtype_t, dtype_t, kCommDtypePerInt4>(
        hp_hidval_reduce_buf, reduce_hidval_ptr_dtype); // NOTE: we have `kCommDtypePerInt4` of `dtype_t` elems to process

    // Rescale the initial old value in advance if `kIsLSEReduce`
    if constexpr (kIsLSEReduce) {
      reduce_dtype_t rescale_weight = shared_old_lse_rescale_weight[reduce_warp_id][head_idx];
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
    reduce_dtype_t reduced_lse_val = shared_reduced_lse[reduce_warp_id][head_idx];
    for (int j = 0; j < num_src_ranks; ++j) {
      auto jth_recv_hidval_comm_dtype = reinterpret_cast<const comm_dtype_t*>(recv_hidval_int4_fn(j));
      // TODO: optimize the repeated load of each head of lse with dynamic shared memory
      // but be careful of the high occupancy of the TMA buffer
      auto jth_recv_lse = get_lse_fn(src_rank_idxs[j], slot_indices[j], head_idx);
      foreach_reduce_lse<reduce_dtype_t, comm_dtype_t, float, kCommDtypePerInt4>(hp_hidval_reduce_buf, reduced_lse_val, jth_recv_hidval_comm_dtype, jth_recv_lse);
    }
  } else if constexpr (kReduceOp == ReduceOp::SUM || kReduceOp == ReduceOp::AVG) {
#pragma unroll
    // Reduce hidden states from src tokens in high precision
    for (int j = 0; j < num_src_ranks; ++j) {
      auto recv_value_dtypes = reinterpret_cast<const comm_dtype_t*>(recv_hidval_int4_fn(j));
      foreach_reduce_add<reduce_dtype_t, comm_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, recv_value_dtypes);
    }

    if constexpr (kReduceOp == ReduceOp::AVG) {
      auto num_reduces = kAccReduce ? num_ranks_to_reduce + 1 : num_ranks_to_reduce; // if in `kAccReduce` mode, the old value also counts
      if (num_reduces > 1) // average by dividing non-trivial `num_reduces`
        foreach_div<reduce_dtype_t, kCommDtypePerInt4>(hp_hidval_reduce_buf, static_cast<reduce_dtype_t>(num_reduces));
    }
  }
}

template <
    typename dtype_t,
    typename comm_dtype_t,
    typename reduce_dtype_t,
    ReduceOp kReduceOp,
    bool kAccReduce,
    bool kGlobalReduce,
    int kMaxNumHeads,
    int kNumReduceWarps,
    int kNumRanks,
    int kMaxNumSrcRanks,
    bool kUseTMA,
    int kNumTMAStages,
    int kNumTMALoadBytes,
    int kNumTMABufferBytesPerStage,
    typename GetHidValPtrFn,
    typename GetLSEFn>
__device__ void reduce_token_in_warp(
    bool is_token_in_rank,
    int token_idx_in_queue,
    int reduce_warp_id,
    int lane_id,
    int hidden_int4,
    int num_heads,
    int head_dim,
    int num_global_ranks,
    int4* reduced_token,
    float* reduced_lse,
    int num_max_recv_tokens,
    const bool* is_token_in_global_rank,
    const GetHidValPtrFn& get_hidval_ptr_fn,
    const GetLSEFn& get_lse_fn,
    reduce_dtype_t shared_reduced_lse[kNumReduceWarps][kMaxNumHeads],
    reduce_dtype_t shared_old_lse_rescale_weight[kNumReduceWarps][kMaxNumHeads],
    uint8_t* smem_ptr,
    uint32_t (&tma_phase)[kNumTMAStages]) {
  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  constexpr int kCommDtypePerDtype = sizeof(dtype_t) / sizeof(comm_dtype_t);
  constexpr int kCommDtypePerInt4 = kCommDtypePerDtype * kDtypePerInt4;
  constexpr bool kIsLSEReduce = kReduceOp == ReduceOp::LSE;
  const int hidden_int4_comm = hidden_int4 / kCommDtypePerDtype;
  GRPCOLL_STATIC_ASSERT(kReduceOp == ReduceOp::AVG or !kGlobalReduce, "Only support global reduce for ReduceOp::AVG");
  GRPCOLL_STATIC_ASSERT(kMaxNumSrcRanks <= WARP_SIZE, "Too many src ranks");

  // Count and collect src slots and corr. src ranks from all lanes
  // NOTES: lane `r` holds the `token_idx_in_queue` and `is_token_in_rank` of rank `r`
  int num_src_ranks = 0, src_rank_idxs[kMaxNumSrcRanks], slot_indices[kMaxNumSrcRanks];
#pragma unroll
  for (int r = 0; r < kNumRanks; ++r) {
    if (broadcast_in_warp(/*val=*/is_token_in_rank, /*src_lane=*/r)) {
      slot_indices[num_src_ranks] = broadcast_in_warp(/*val=*/token_idx_in_queue, /*src_lane=*/r) % num_max_recv_tokens;
      src_rank_idxs[num_src_ranks++] = r;
    }
  }
  GRPCOLL_DEVICE_ASSERT(num_src_ranks <= kMaxNumSrcRanks);

  // Get the actual number of ranks to reduce
  // which is only used for `kReduceOp == ReduceOp::AVG` by now
  int num_ranks_to_reduce = 0;
  if constexpr (kGlobalReduce) { // read and reduce from `is_token_in_global_rank`
    for (int r = lane_id; r < num_global_ranks; r += WARP_SIZE) {
      num_ranks_to_reduce += is_token_in_global_rank[r];
    }
    num_ranks_to_reduce = warp_reduce_sum(num_ranks_to_reduce);
  } else { // just use `num_src_ranks`
    num_ranks_to_reduce = num_src_ranks;
  }

  // Reduce lse from src tokens if `kIsLSEReduce`
  if constexpr (kIsLSEReduce) {
    reduce_lse_in_warp<reduce_dtype_t, kAccReduce, kMaxNumHeads, kNumReduceWarps>(
        /*reduced_lse=*/reduced_lse,
        /*get_lse_fn=*/get_lse_fn,
        /*shared_reduced_lse=*/shared_reduced_lse,
        /*shared_old_lse_rescale_weight=*/shared_old_lse_rescale_weight,
        /*src_rank_idxs=*/src_rank_idxs,
        /*slot_indices=*/slot_indices,
        /*num_src_ranks=*/num_src_ranks,
        /*reduce_warp_id=*/reduce_warp_id,
        /*lane_id=*/lane_id,
        /*num_heads=*/num_heads);
    __syncwarp();
  }

  // Reduce token from src ranks
  if constexpr (kUseTMA) {
    GRPCOLL_STATIC_ASSERT(kNumTMALoadBytes == sizeof(int4) * WARP_SIZE, "Invalid kNumTMALoadBytes");
    GRPCOLL_STATIC_ASSERT(
        kNumTMABufferBytesPerStage == align(kNumTMALoadBytes * (kMaxNumSrcRanks + 1) + /*mbarrier*/ sizeof(uint64_t), sizeof(int4)),
        "Invalid kNumTMABufferBytesPerStage");

    // Define functions to access TMA load/store buffers and mbarriers
    //  `tma_load_buffer`: shape=(kNumTMAStages, kMaxNumSrcRanks, kNumTMALoadBytes), dtype=int4
    //  `tma_store_buffer`: shape=(kNumTMAStages, kNumTMALoadBytes), dtype=int4
    //  `tma_mbarrier`: shape=(kNumTMAStages, 1), dtype=uint64_t
    auto tma_load_buffer = [=](const int& stage_idx, const int& src_rank_idx) -> int4* {
      return reinterpret_cast<int4*>(smem_ptr + stage_idx * kNumTMABufferBytesPerStage + src_rank_idx * kNumTMALoadBytes);
    };
    auto tma_store_buffer = [=](const int& stage_idx) -> int4* {
      return reinterpret_cast<int4*>(smem_ptr + stage_idx * kNumTMABufferBytesPerStage + kMaxNumSrcRanks * kNumTMALoadBytes);
    };
    auto tma_mbarrier = [=](const int& stage_idx) -> uint64_t* {
      return reinterpret_cast<uint64_t*>(smem_ptr + stage_idx * kNumTMABufferBytesPerStage + (kMaxNumSrcRanks + 1) * kNumTMALoadBytes);
    };

    // Prefetch the hidden states of src tokens for stage0 with TMA-load
    // each lane for one token, if `kNumTMAStages > 1`
    if constexpr (kNumTMAStages > 1) {
      if (lane_id < num_src_ranks) {
        tma_load_1d(
            /*smem_ptr=*/tma_load_buffer(0, lane_id),
            /*gmem_ptr=*/get_hidval_ptr_fn(src_rank_idxs[lane_id], slot_indices[lane_id], 0),
            /*mbar_ptr=*/tma_mbarrier(0),
            /*num_bytes=*/kNumTMALoadBytes,
            /*evict_first=*/true);
      }
      mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_src_ranks ? kNumTMALoadBytes : 0);
      __syncwarp();
    }

    // Loop over the whole hidden size in int4 and group_reduce
    // NOTES: hidden_int4 should be a multiple of WARP_SIZE
    // since we need to sync the whole warp in each loop, and copy WARP_SIZE of int4s to/from shared memory
    for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += WARP_SIZE, iter += 1) { // warp-strided loop
      // Get TMA stage info
      const int stage_idx = iter % kNumTMAStages;
      const int next_stage_idx = (iter + 1) % kNumTMAStages;

      // Get the hidden value ptr of `int_4` to reduce to in `reduced_token`
      int4* reduce_hidval_ptr_int4 = reduced_token + shifted;

      // Get the optional head idx, valid and used only when `kIsLSEReduce`
      // NOTES: we guarantee that each `kCommDtypePerInt4` of elems share the same head
      const int head_idx = kIsLSEReduce ? shifted / head_dim : -1;

      // Prefetch the hidden states of src tokens for next stage
      // each lane for one token, if `kNumTMAStages > 1`
      if constexpr (kNumTMAStages > 1) {
        if (shifted + WARP_SIZE < hidden_int4) { // not last iter
          if (lane_id < num_src_ranks) {
            tma_load_1d(
                /*smem_ptr=*/tma_load_buffer(next_stage_idx, lane_id),
                /*gmem_ptr=*/get_hidval_ptr_fn(src_rank_idxs[lane_id], slot_indices[lane_id], shifted + WARP_SIZE),
                /*mbar_ptr=*/tma_mbarrier(next_stage_idx),
                /*num_bytes=*/kNumTMALoadBytes,
                /*evict_first=*/true);
          }
          mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_src_ranks ? kNumTMALoadBytes : 0);
          __syncwarp();
        }
      } else { // Load the hidden states of src tokens for current stage if `kNumTMAStages == 1`
        if (lane_id < num_src_ranks) {
          tma_load_1d(
              /*smem_ptr=*/tma_load_buffer(stage_idx, lane_id),
              /*gmem_ptr=*/get_hidval_ptr_fn(src_rank_idxs[lane_id], slot_indices[lane_id], shifted),
              /*mbar_ptr=*/tma_mbarrier(stage_idx),
              /*num_bytes=*/kNumTMALoadBytes,
              /*evict_first=*/true);
        }
        mbarrier_arrive_and_expect_tx(tma_mbarrier(stage_idx), lane_id < num_src_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();
      }

      // Wait for the TMA load of current stage to be finished
      // for current hidden states of src tokens
      mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);

      // Prepare high-precision reduce buffer and recv_hidval_int4_fn
      reduce_dtype_t hp_hidval_reduce_buf[kCommDtypePerInt4];
      auto recv_hidval_int4_fn = [&](int src_rank_idx) -> int4* { return reinterpret_cast<int4*>(tma_load_buffer(stage_idx, src_rank_idx) + lane_id); };

      // High-precision reduce `recv_hidval_int4` from src ranks to `hp_hidval_reduce_buf` in high precision
      // maybe with old vlaues in `reduce_hidval_ptr_int4` if in `kAccReduce` mode
      reduce_hidval_in_warp<dtype_t, comm_dtype_t, reduce_dtype_t, kReduceOp, kAccReduce, kMaxNumHeads, kNumReduceWarps, kCommDtypePerInt4>(
          /*reduce_warp_id=*/reduce_warp_id,
          /*head_idx=*/head_idx,
          /*num_src_ranks=*/num_src_ranks,
          /*num_ranks_to_reduce=*/num_ranks_to_reduce,
          /*reduce_hidval_ptr_int4=*/reduce_hidval_ptr_int4,
          /*hp_hidval_reduce_buf=*/hp_hidval_reduce_buf,
          /*src_rank_idxs=*/src_rank_idxs,
          /*slot_indices=*/slot_indices,
          /*recv_hidval_int4_fn=*/recv_hidval_int4_fn,
          /*get_lse_fn=*/get_lse_fn,
          /*shared_reduced_lse=*/shared_reduced_lse,
          /*shared_old_lse_rescale_weight=*/shared_old_lse_rescale_weight);

      // Wait for the TMA store in previous round with `stage_idx` to be finished
      // allowing previous `kNumTMAStages - 1` stages to be inflight
      tma_store_wait<kNumTMAStages - 1>();

      // Cast and store the reduced hidden states of src tokens
      // from registers to TMA buffer in shared memory
      // GRPCOLL_STATIC_ASSERT(kCommDtypePerInt4 == kDtypePerInt4); // TODO: support lp comm dtype
      auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
      foreach_assign<dtype_t, reduce_dtype_t, kCommDtypePerInt4>(out_dtypes, hp_hidval_reduce_buf);

      // Fence all lanes to make sure their access to TMA buffer
      // in shared memory are visible to each other
      tma_store_fence();
      __syncwarp();

      // Store the reduced hidden states of src tokens for current stage
      // from TMA buffer in shared memory to output buffer
      if (lane_id == 0) { // issued by lane0 for all WARP_SIZE of reduced values
        tma_store_1d(
            /*smem_ptr=*/tma_store_buffer(stage_idx),
            /*gmem_ptr=*/reduce_hidval_ptr_int4,
            /*num_bytes=*/kNumTMALoadBytes,
            /*evict_first=*/true);
      }
      __syncwarp();
    }

    // Wait all TMA stores to be finished
    tma_store_wait();
  } else {
#pragma unroll
    for (int i = lane_id; i < hidden_int4_comm; i += WARP_SIZE) { // warp-strided loop
      // Get the hidden value ptr of `int_4` to reduce to in `reduced_token`
      int4* reduce_hidval_ptr_int4 = reduced_token + i * kCommDtypePerDtype;

      // Get the optional head idx, valid and used only when `kIsLSEReduce`
      // NOTES: we guarantee that each `kCommDtypePerInt4` of elems share the same head
      const int head_idx = kIsLSEReduce ? i * kCommDtypePerInt4 / head_dim : -1;

      // Read hidden states of src tokens
      // TODO: maybe too many registers here
      int4 recv_hidval_int4[kMaxNumSrcRanks];
#pragma unroll
      for (int j = 0; j < num_src_ranks; ++j) {
        recv_hidval_int4[j] = ld_nc_global(get_hidval_ptr_fn(src_rank_idxs[j], slot_indices[j], i * kCommDtypePerDtype));
      }

      // Prepare high-precision reduce buffer and recv_hidval_int4_fn
      reduce_dtype_t hp_hidval_reduce_buf[kCommDtypePerInt4];
      auto recv_hidval_int4_fn = [&](int src_rank_idx) -> int4* { return recv_hidval_int4 + src_rank_idx; };

      // High-precision reduce `recv_hidval_int4` from src ranks to `hp_hidval_reduce_buf` in high precision
      // maybe with old vlaues in `reduce_hidval_ptr_int4` if in `kAccReduce` mode
      reduce_hidval_in_warp<dtype_t, comm_dtype_t, reduce_dtype_t, kReduceOp, kAccReduce, kMaxNumHeads, kNumReduceWarps, kCommDtypePerInt4>(
          /*reduce_warp_id=*/reduce_warp_id,
          /*head_idx=*/head_idx,
          /*num_src_ranks=*/num_src_ranks,
          /*num_ranks_to_reduce=*/num_ranks_to_reduce,
          /*reduce_hidval_ptr_int4=*/reduce_hidval_ptr_int4,
          /*hp_hidval_reduce_buf=*/hp_hidval_reduce_buf,
          /*src_rank_idxs=*/src_rank_idxs,
          /*slot_indices=*/slot_indices,
          /*recv_hidval_int4_fn=*/recv_hidval_int4_fn,
          /*get_lse_fn=*/get_lse_fn,
          /*shared_reduced_lse=*/shared_reduced_lse,
          /*shared_old_lse_rescale_weight=*/shared_old_lse_rescale_weight);

      // Cast the high-precision reduced value back to `dtype_t`
      int4 out_int4[kCommDtypePerDtype];
      auto out_dtypes = reinterpret_cast<dtype_t*>(out_int4);
      foreach_assign<dtype_t, reduce_dtype_t, kCommDtypePerInt4>(out_dtypes, hp_hidval_reduce_buf);

#pragma unroll
      // Store reduced values to output buffer
      for (int l = 0; l < kCommDtypePerDtype; ++l) {
        st_na_global(reduce_hidval_ptr_int4 + l, out_int4[l]); // non-cached store
      }
    }
  }
}

} // namespace magi_attn_comm::grpcoll::internode
