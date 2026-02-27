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

#include "configs.cuh"
#include "internode_utils.cuh"

namespace cg = cooperative_groups;

namespace magi_attn_comm::grpcoll::internode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool kLowLatencyMode,
    bool kCachedMode,
    int kNumDataGroups,
    int kNumRDMARanks,
    int kNumMaxDstRDMARanks,
    int kNumTMABytesPerWarp,
    int kNumSenderWarps,
    int kWarpCopyUnrollStages,
    bool kCastLSE,
    bool kHasKernelBarrier,
    bool kIsCachedNotifyFused>
GLOBAL_LAUNCH_BOUNDS(get_num_threads_group_cast(kNumSenderWarps), 1)
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
    SourceMeta* recv_src_meta,
    int* send_rdma_head,
    int* send_nvl_head,
    int* recv_rdma_channel_prefix_matrix,
    int* recv_gbl_channel_prefix_matrix,
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    const bool* is_token_in_rank,
    const int64_t* post_perm_idx,
    int num_tokens,
    int hidden_int4,
    int num_heads,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks,
    /* other metadata for optional cached notify */
    const size_t rdma_clean_offset,
    const size_t rdma_num_int_clean,
    const size_t nvl_clean_offset,
    const size_t nvl_num_int_clean,
    int** barrier_signal_ptrs,
    const nvshmem_team_t rdma_team,
    /* other metadata for optional kernel barrier */
    magi_attn_ext::KernelBarrierView kernel_barrier_view) {
  // Optional kernel barrier arrive
  if constexpr (kHasKernelBarrier) {
    kernel_barrier_view.arrive();
  }

  // Get thread info
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();

  // Get rank and channel Info
  const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const bool is_forwarder = sm_id % 2 == 0;
  GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= WARP_SIZE, "Invalid number of RDMA peers");
  GRPCOLL_DEVICE_ASSERT(
      ibgda_get_state()->num_rc_per_pe == num_channels or ibgda_get_state()->num_rc_per_pe >= num_sms,
      "num_channels = %d must be equal to or larger than num_sms = %d divided by NUM_MAX_NVL_PEERS = %d",
      num_channels,
      num_sms,
      NUM_MAX_NVL_PEERS);

  // Optional cached notify and grid sync
  if constexpr (kCachedMode && kIsCachedNotifyFused) {
    auto grid = cg::this_grid();
    if (sm_id == 0) {
      cached_notify_func<kLowLatencyMode>(
          rdma_clean_offset,
          rdma_num_int_clean,
          nvl_clean_offset,
          nvl_num_int_clean,
          rdma_buffer_ptr,
          buffer_ptrs,
          barrier_signal_ptrs,
          num_threads,
          thread_id,
          kNumRDMARanks,
          rdma_rank,
          nvl_rank,
          rdma_team);
    }
    grid.sync();
  }

  // Get size info
  const auto hidden_bytes = hidden_int4 * sizeof(int4), lse_bytes = num_heads * sizeof(float);
  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_heads);
  const auto total_num_bytes_per_token = num_bytes_per_token + hidden_bytes * (kNumDataGroups - 1); // for other groups, we only transfer distinct hidden states
  constexpr int num_meta_per_rdma_channel = 2 * (NUM_MAX_NVL_PEERS + 1); // (start, end) idx for dst RDMA peer (latter) + its each NVL rank (former)
  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 and kNumDataGroups <= 3, "Invalid number of data groups");

  /** NOTE: Determine warp role and its target rank
   * For Forwarder (Even SMs):
   *  1. the first `NUM_MAX_NVL_PEERS` warps are `kRDMA2NVLForwarder`,
   *    forwarding the received tokens from all RDMA peers in RDMA recv buffer (as RDMA consumers)
   *    to each dst NVL peer in this node (as NVL producers),
   *    each warp for one NVL peer and each lane for one RDMA peer
   *  2. the rest warps are `kForwarderCoordinator`, but only the first one is active,
   *    and it is responsible for updating the minimum RDMA head consumed by all `kRDMA2NVLForwarder` warps
   *
   * For Sender/Receiver (Odd SMs):
   *  1. the first `kNumSenderWarps` warps are `kRDMASender`,
   *    copying the corr. channel of tokens to RDMA send buffer of this RDMA rank for each RDMA peer (as RDMA producers),
   *    each warp for one token in round-robin way
   *  2. the next warp is `kRDMASenderCoordinator`, issuing RDMA copy for tokens
   *    copied to the RDMA send buffer of this RDMA rank by `kRDMASender`
   *    to the RDMA recv buffer of each RDMA peer
   *  3. the rest `NUM_MAX_NVL_PEERS` warps are `kNVLReceivers`,
   *    copying the received tokens from all NVL peers to output buffer (as NVL consumers),
   *    each warp for one NVL peer
   */
  enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMA2NVLForwarder, kForwarderCoordinator, kNVLReceivers };
  const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kRDMA2NVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
      }
    } else { // sender / receiver
      if (warp_id < kNumSenderWarps) {
        return {WarpRole::kRDMASender, -1}; // Not applicable for RDMA senders
      } else if (warp_id == kNumSenderWarps) {
        return {WarpRole::kRDMASenderCoordinator, -1}; // Not applicable for RDMA senders
      } else {
        return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumSenderWarps) % NUM_MAX_NVL_PEERS};
      }
    }
  }();
  const auto warp_role = role_meta.first;
  const auto target_rank = role_meta.second;

  int rs_wr_rank = 0, ws_rr_rank = 0; // `rs_wr` denotes "Read for Senders, Write for Receivers", while `ws_rr` denotes "Write for Senders, Read for Receivers"
  if (warp_role == WarpRole::kRDMA2NVLForwarder) {
    // NVL forwarder/sender will read from this NVL rank and write to target NVL peer
    rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
  } else if (warp_role == WarpRole::kNVLReceivers) {
    // NVL receiver will read from target NVL peer and write to this NVL rank
    rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;
  }

  // Get RDMA data buffer
  // NOTE: for other data groups, we pack their hidden states together with the first for coalesce transfer if possible
  //  `rdma_channel_data_1st`: shape=(num_channels, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, num_bytes_per_token), dtype=uint8_t
  //  `rdma_channel_data_2nd`: shape=(num_channels, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, hidden_bytes), dtype=uint8_t
  auto rdma_channel_data = SymBuffer<uint8_t, /*kDecoupled=*/true>(
      rdma_buffer_ptr,
      /*num_elems=*/num_max_rdma_chunked_recv_tokens * total_num_bytes_per_token,
      /*num_ranks=*/kNumRDMARanks,
      /*sm_id=*/channel_id,
      /*num_sms=*/num_channels);

  // Get RDMA meta buffer
  //  `rdma_channel_meta`: shape=(num_channels, kNumRDMARanks, num_meta_per_rdma_channel), dtype=int
  //  `rdma_channel_head`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
  //  `rdma_channel_tail`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
  auto rdma_channel_meta = SymBuffer<int, /*kDecoupled=*/true>(
      rdma_buffer_ptr, /*num_elems=*/num_meta_per_rdma_channel, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);
  auto rdma_channel_head =
      SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);
  auto rdma_channel_tail =
      SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);

  // Get NVL data buffer
  // NOTE: for other data groups, we pack their hidden states together with the first for coalesce transfer if possible
  //  `nvl_channel_x_1st`: shape=(num_channels, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_bytes_per_token), dtype=uint8_t
  //  `nvl_channel_x_2nd`: shape=(num_channels, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, hidden_bytes), dtype=uint8_t
  auto rs_wr_buffer_ptr = buffer_ptrs[rs_wr_rank], ws_rr_buffer_ptr = buffer_ptrs[ws_rr_rank];
  auto nvl_channel_x = AsymBuffer<uint8_t>(
                           ws_rr_buffer_ptr,
                           /*num_elems=*/num_max_nvl_chunked_recv_tokens * total_num_bytes_per_token,
                           /*num_ranks=*/NUM_MAX_NVL_PEERS,
                           /*sm_id=*/channel_id,
                           /*num_sms=*/num_channels,
                           /*offset=*/rs_wr_rank)
                           .advance_also(rs_wr_buffer_ptr);

  // Get NVL meta buffer
  //  `nvl_channel_prefix_start`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
  //  `nvl_channel_prefix_end`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
  //  `nvl_channel_head`: shape=(num_channels, NUM_MAX_NVL_PEERS), dtype=int
  //  `nvl_channel_tail`: shape=(num_channels, NUM_MAX_NVL_PEERS), dtype=int
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
  __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks]; // NOTE: each bit in one `uint32_t` corresponds to one transaction
  auto sync_rdma_sender_smem = []() { sync_warp_group(/*group_flag=*/0, /*group_size=*/(kNumSenderWarps + 1) * WARP_SIZE); };

  // Prepare TMA buffer and init mbarrier
  // NOTE: TMA buffer is only used by `kRDMA2NVLForwarder` and `kNVLReceivers`
  extern __shared__ __align__(1024) uint8_t smem_tma_buffer[]; // REVIEW: why aligned to 1024 bytes ?
  auto tma_buffer = smem_tma_buffer + target_rank * kNumTMABytesPerWarp;
  auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
  uint32_t tma_phase = 0;
  if ((warp_role == WarpRole::kRDMA2NVLForwarder or warp_role == WarpRole::kNVLReceivers) and lane_id == 0) {
    mbarrier_init(tma_mbarrier, /*arrive_count=*/1); // only lane0 participates
    fence_view_async_shared();
    fence_barrier_init();
  }
  __syncwarp();

  // Prepare NVL forwarder warp synchronization
  //  `forward_channel_head`: the RDMA head for each src RDMA peer of each dst NVL peer / `kRDMA2NVLForwarder` warp
  //  `forward_channel_retired`: the retire flag for each `kRDMA2NVLForwarder` warp
  //  `sync_forwarder_smem`: synchronize warps of `kRDMA2NVLForwarder` and `kForwarderCoordinator` warps
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
    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumSenderWarps) {
      auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) // NOTE: for this NVL rank, we directly write to recv buffer
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
            /*qp_id=*/channel_id, // NOTE: each channel use its own qp
            /*lane_id=*/lane_id,
            /*message_idx=*/0);
      }
    }

    // Sync `kRDMASender` with `kRDMASenderCoordinator` to make sure the shared memory of
    // `rdma_send_channel_lock`, `rdma_send_channel_tail` and `rdma_send_channel_window`
    // are cleaned by `kRDMASenderCoordinator` before `kRDMASender` access them
    sync_rdma_sender_smem();

    // Prepare RDMA send buffer
    int64_t token_idx;
    int cached_rdma_channel_head = 0, global_rdma_tail_idx = 0;
    uint8_t* send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);

    // Iterate over tokens and copy into RDMA send buffer
    GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
      // Update `global_rdma_tail_idx`
      // by counting whether this token is sent to RDMA rank indicated by `lane_id`
      uint64_t is_token_in_rank_uint64 = 0;
      if (lane_id < kNumRDMARanks) {
        is_token_in_rank_uint64 = __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
        global_rdma_tail_idx += (is_token_in_rank_uint64 != 0); // NOTE: one `uint64_t` is 8 bytes to cover 8 bools for 8 NVL peers
      }
      __syncwarp();

      // Skip the token which does not belong to this warp
      // NOTE: each warp for one token in round-robin way
      if ((token_idx - token_start_idx) % kNumSenderWarps != warp_id)
        continue;

      const auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

      // Wait the queue non-full, i.e. the remote buffer to be released
      auto start_time = clock64();
      while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
        cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id))); // volatile load

        // Timeout check
        timeout_check_rdma_sender(start_time, channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
      }
      __syncwarp();

      // Store RDMA head for group_reduce stage to reduce
      //  `send_rdma_head`: shape=(num_tokens, kNumRDMARanks), dtype=int
      if (lane_id < kNumRDMARanks and !kCachedMode) {
        send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;
      }

      // Broadcast tails
      SourceMeta src_meta;
      int num_dst_rdma_ranks = 0;
      void* dst_send_buffers[kNumMaxDstRDMARanks];
#pragma unroll
      // Broadcast info about the dst RDMA ranks this token will be sent to, including:
      //  1. the `src_meta` to tell which NVL ranks this token should be sent to in each RDMA peer
      //  2. the `dst_send_buffer` ptr of this token in each RDMA send buffer queue
      for (int r = 0, slot_idx; r < kNumRDMARanks; ++r) {
        if ((slot_idx = broadcast_in_warp(/*val=*/rdma_tail_idx, /*src_lane=*/r)) >= 0) {
          slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
          auto recv_is_token_in_rank_uint64 = broadcast_ptr_in_warp(/*ptr=*/is_token_in_rank_uint64, /*src_lane=*/r);
          auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);

          // Prepare `src_meta`
          if (lane_id == num_dst_rdma_ranks) {
            src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
          }

          // Prepare `dst_send_buffer`
          dst_send_buffers[num_dst_rdma_ranks++] =
              reinterpret_cast<uint8_t*>(broadcast_ptr_in_warp(/*ptr=*/send_buffer, /*src_lane=*/r)) + slot_idx * total_num_bytes_per_token;
        }
      }
      GRPCOLL_DEVICE_ASSERT(
          num_dst_rdma_ranks <= kNumMaxDstRDMARanks,
          "num_dst_rdma_ranks = %d exceeds kNumMaxDstRDMARanks = %d",
          num_dst_rdma_ranks,
          kNumMaxDstRDMARanks); // REVIEW: why at most 8 RDMA peers to send to ?

#pragma unroll
      // Warp-copy the hidden value of this token for each data group
      // into each RDMA send buffer for each dst RDMA rank to send to
      for (int g = 0; g < kNumDataGroups; ++g) {
        auto st_dst_rdma_ranks = [=](const int hidden_offset, const int4& hidden_val_int4) {
#pragma unroll
          for (int j = 0; j < num_dst_rdma_ranks; ++j)
            st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + hidden_int4 * g + hidden_offset, hidden_val_int4);
        };
        auto x_ptr = (g == 0) ? x : ((g == 1) ? x_2nd : x_3rd);

        UNROLLED_WARP_COPY(
            /*UNROLL_FACTOR=*/kWarpCopyUnrollStages,
            /*LANE_ID=*/lane_id,
            /*N=*/hidden_int4,
            /*DST=*/0,
            /*SRC=*/x_ptr + token_idx * hidden_int4,
            /*LD_FUNC=*/ld_nc_global, // non-cached load
            /*ST_FUNC=*/st_dst_rdma_ranks // non-cached store to each send buffer of each dst RDMA rank
        );
      }

#pragma unroll
      // Offset the send buffers across the hidden states part
      for (int r = 0; r < num_dst_rdma_ranks; ++r)
        dst_send_buffers[r] = reinterpret_cast<int4*>(dst_send_buffers[r]) + hidden_int4 * kNumDataGroups;

#pragma unroll
      // Copy `lse`
      // into each RDMA send buffer for each dst RDMA rank to send to
      for (int i = lane_id; i < num_heads; i += WARP_SIZE) {
        auto offset = token_idx * num_heads + i;
        auto value = ld_nc_global(lse + offset);
#pragma unroll
        for (int j = 0; j < num_dst_rdma_ranks; ++j)
          st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
      }

#pragma unroll
      // Offset the send buffers across the lse
      for (int r = 0; r < num_dst_rdma_ranks; ++r)
        dst_send_buffers[r] = reinterpret_cast<float*>(dst_send_buffers[r]) + num_heads;

      // Copy `src_meta`
      // into each RDMA send buffer for each dst RDMA rank to send to
      if (lane_id < num_dst_rdma_ranks) {
        st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
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
      if (lane_id < kNumRDMARanks) {
        timeout_check_rdma_sender_coordinator(start_time, channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
      }

      // Iterate all RDMA ranks if there's any (remaining) token to send
      for (int r = 0, synced_num_tokens_to_send; r < kNumRDMARanks; ++r) {
        // To mitigate in-cast congestion, shuffle the starting index of target rank for different ranks and channels
        const int dst_rdma_rank = (r + channel_id + rdma_rank) % kNumRDMARanks;
        const int dst_pe = get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank);
        synced_num_tokens_to_send = broadcast_in_warp(/*val=*/num_tokens_to_send, /*src_lane=*/dst_rdma_rank);
        if (synced_num_tokens_to_send == 0)
          continue;

        // Read the latest progress of `kRDMASender`
        // to get the number of tokens copied into the RDMA send buffer
        // NOTE: `rdma_send_channel_tail` does not need to be protected by lock
        auto processed_tail = broadcast_in_warp(/*val=*/ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank))); // CTA scope, acquire order
        auto synced_last_issued_tail = broadcast_in_warp(/*val=*/last_issued_tail, /*src_lane=*/dst_rdma_rank);
        auto num_tokens_processed = processed_tail - synced_last_issued_tail;

        // If the number of tokens to be processed is not enough as a chunk, skip for next round
        if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
          continue;

        // Issue RDMA copy for a chunk of tokens in this round
        // from the send buffer of this RDMA rank to the recv buffer of `dst_rdma_rank`
        auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
        GRPCOLL_DEVICE_ASSERT(
            num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send,
            "Invalid number of tokens to issue, num_tokens_to_issue = %d, num_tokens_processed = %d, synced_num_tokens_to_send = %d",
            num_tokens_to_issue,
            num_tokens_processed,
            synced_num_tokens_to_send);
        if (dst_rdma_rank != rdma_rank) { // dst RDMA peer
          auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
          GRPCOLL_DEVICE_ASSERT(
              dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens,
              "Invalid destination slot index, dst_slot_idx = %d, num_tokens_to_issue = %d, num_max_rdma_chunked_recv_tokens = %d",
              dst_slot_idx,
              num_tokens_to_issue,
              num_max_rdma_chunked_recv_tokens);

          const size_t num_bytes_per_msg = total_num_bytes_per_token * num_tokens_to_issue;
          auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * total_num_bytes_per_token);
          auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * total_num_bytes_per_token);
          // REVIEW: maybe better to use thread-level `put_nbi` ?
          nvshmemi_ibgda_put_nbi_warp</*kAlwaysDoPostSend=*/true>(
              /*req_rptr=*/dst_ptr,
              /*req_lptr=*/src_ptr,
              /*bytes=*/num_bytes_per_msg,
              /*dst_pe=*/dst_pe,
              /*qp_id=*/channel_id,
              /*lane_id=*/lane_id,
              /*message_idx=*/0);
        } else { // this RDMA rank
          // Already in its own recv buffer, so no need to copy
          memory_fence(); // NOTE: use lighter fence for local memory operations
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
              /*pe=*/dst_pe,
              /*qp_id=*/channel_id,
              /*is_local_copy=*/dst_rdma_rank == rdma_rank);
        }
        __syncwarp();
      }
    }
  } else if (warp_role == WarpRole::kRDMA2NVLForwarder) {
    const auto dst_nvl_rank = target_rank; // each warp for one dst NVL peer

    // Wait `rdma_channel_meta` to be ready for each RDMA peer
    // NOTE: each lane will ready specific `num_tokens_to_recv_from_rdma` and `rdma_token_start_idx` for each RDMA peer
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
          GRPCOLL_DEVICE_ASSERT(
              nvl_token_start_idx >= 0 and nvl_token_end_idx >= nvl_token_start_idx,
              "Invalid NVL token idx, nvl_token_start_idx = %d, nvl_token_end_idx = %d",
              nvl_token_start_idx,
              nvl_token_end_idx);
          st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, nvl_token_start_idx_encoded);
          st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, nvl_token_end_idx_encoded);

          // Get RDMA channel received token count
          rdma_token_start_idx = decode(rdma_token_start_idx_encoded);
          auto rdma_token_end_idx = decode(rdma_token_end_idx_encoded);
          num_tokens_to_recv_from_rdma = rdma_token_end_idx - rdma_token_start_idx;
          GRPCOLL_DEVICE_ASSERT(
              num_tokens_to_recv_from_rdma >= 0,
              "Invalid number of tokens to receive from RDMA, num_tokens_to_recv_from_rdma = %d, rdma_token_start_idx = %d, rdma_token_end_idx = %d",
              num_tokens_to_recv_from_rdma,
              rdma_token_start_idx,
              rdma_token_end_idx);

          // Store `rdma_token_end_idx` for group_reduce stage
          //  `recv_rdma_channel_prefix_matrix`: shape=[kNumRDMARanks, kNumChannels], dtype=int
          if (!kCachedMode)
            recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = rdma_token_end_idx;

          // Shift `rdma_token_start_idx` by RDMA rank offset
          //  `recv_rdma_rank_prefix_sum`: shape=[kNumRDMARanks,], dtype=int
          rdma_token_start_idx += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];

          break;
        }

        // Timeout check
        timeout_check_rdma2nvl_forwarder_rdma_meta(
            start_time,
            channel_id,
            rdma_rank,
            nvl_rank,
            lane_id,
            dst_nvl_rank,
            nvl_token_start_idx_encoded,
            nvl_token_end_idx_encoded,
            rdma_token_start_idx_encoded,
            rdma_token_end_idx_encoded);
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
    // NOTE: always start from the local rank
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
        auto nvl_channel_head_this_lane = ld_volatile_global(nvl_channel_head.buffer()); // volatile load
        cached_nvl_channel_head = broadcast_in_warp(/*val=*/nvl_channel_head_this_lane);

        // Timeout check
        if (lane_id == 0) {
          timeout_check_rdma2nvl_forwarder_nvl_head(start_time, channel_id, rdma_rank, nvl_rank, dst_nvl_rank, nvl_channel_head_this_lane, cached_nvl_channel_tail);
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
        if (lane_id < kNumRDMARanks) {
          timeout_check_rdma2nvl_forwarder_rdma_head(
              start_time, channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_rdma_channel_head, cached_rdma_channel_tail, num_tokens_to_recv_from_rdma);
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
        auto rdma_token_ptr = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * total_num_bytes_per_token;

        // Get `src_meta` for current token
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(rdma_token_ptr + hidden_bytes * kNumDataGroups + lse_bytes)); // non-cached load

        // Decrement `num_tokens_to_recv_from_rdma`
        lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;

        // If this RDMA token does not need to forward to dst NVL peer, skip it
        // Otherwise, increment `rdma_nvl_token_idx` and update `send_nvl_head`
        // of current RDMA token for group_reduce stage
        bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
        if (lane_id == src_rdma_rank) {
          auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
          rdma_nvl_token_idx += is_in_dst_nvl_rank;
          if (!kCachedMode)
            send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
        }
        if (!is_in_dst_nvl_rank)
          continue;

        // Get an empty slot in NVL queue
        // and increment `cached_nvl_channel_tail`
        int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;

        // Get token ptr in NVL buffer of dst NVL peer
        auto nvl_token_ptr = nvl_channel_x.buffer() + dst_slot_idx * total_num_bytes_per_token;

#pragma unroll
        // TMA-copy token from RDMA buffer to TMA buffer in shared memory
        // NOTE: due to the shared memory size limit, we can not issue all in one go,
        // thus we issue the hidden states iteratively for each data group,
        // while the last data group also includes the lse and src_meta
        for (int g = 0; g < kNumDataGroups; ++g) {
          auto hidval_offsets = hidden_bytes * g;
          auto tma_copy_bytes = (g < kNumDataGroups - 1) ? hidden_bytes : num_bytes_per_token;

          if (lane_id == 0) { // issued by lane0
            // Wait for TMA-copy for previous data groups to be finished
            if (g > 0)
              tma_store_wait();

            tma_load_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/rdma_token_ptr + hidval_offsets,
                /*mbar_ptr=*/tma_mbarrier,
                /*num_bytes=*/tma_copy_bytes,
                /*evict_first=*/false);
            mbarrier_arrive_and_expect_tx(/*mbar_ptr=*/tma_mbarrier, /*num_bytes=*/tma_copy_bytes);
          }
          __syncwarp();

          // Wait TMA load to be finished
          // and flip the `tma_phase` in-place
          mbarrier_wait(/*mbar_ptr=*/tma_mbarrier, /*stage=*/tma_phase);

          // TMA-copy token from TMA buffer in shared memory to NVL buffer
          if (lane_id == 0) { // issued by lane0
            tma_store_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/nvl_token_ptr + hidval_offsets,
                /*num_bytes=*/tma_copy_bytes,
                /*evict_first=*/true);
          }
          __syncwarp();
        }

        // Early stop when the NVL chunk is full
        if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
          src_rdma_tail = i + 1;

        // Wait TMA store to be finished
        tma_store_wait();
        __syncwarp();
      }

      // Sync RDMA head to shared memory
      // to let `kForwarderCoordinator` warp update the minimum RDMA head across all `kRDMA2NVLForwarder` warps
      if (lane_id == src_rdma_rank)
        forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);
      __syncwarp();

      // Update NVL tail
      if (lane_id == 0)
        st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail); // system scope, release order
    }
    __syncwarp();

    // Mark this warp as retired
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
    // before `kRDMA2NVLForwarder` warps access them
    sync_forwarder_smem();

    // Loop minimum head in `forward_channel_head` and update RDMA head
    int last_head = 0, target_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
    while (true) {
      // Find minimum head recorded by `kRDMA2NVLForwarder` warps
      int min_head = INT_MAX;
#pragma unroll
      for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
        if (!forward_channel_retired[r])
          min_head = min(min_head, forward_channel_head[r][target_rdma_rank]);
      if (all_in_warp(min_head == INT_MAX)) // all `kRDMA2NVLForwarder` warps are retired
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
      timeout_check_nvl_receiver_meta(start_time, channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset);
    }

    // Warp-reduce across all RDMA peers for the src NVL rank
    num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

    // Store `recv_gbl_channel_prefix_matrix` for group_reduce stage
    //  `recv_gbl_channel_prefix_matrix`: shape=(kNumRanks, num_channels), dtype=int
    if (lane_id < kNumRDMARanks and !kCachedMode)
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

        // Read the NVL tail updated by `kRDMA2NVLForwarder`
        // REVIEW: here all lanes repeatedly load the same `nvl_channel_tail` ?
        cached_channel_tail_idx = broadcast_in_warp(/*val=*/ld_acquire_sys_global(nvl_channel_tail.buffer())); // system scope, acquire order

        // Timeout check
        if (lane_id == 0) {
          timeout_check_nvl_receiver_tail(start_time, channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
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
        auto token_ptr_in_buffer = nvl_channel_x.buffer() + slot_idx_in_queue * total_num_bytes_per_token;

        // Load src meta to get the `src_rdma_rank` for this token
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(token_ptr_in_buffer + hidden_bytes * kNumDataGroups + lse_bytes));

        // Get recv token idx in the `recv_x`
        int64_t recv_token_idx = broadcast_in_warp(/*val=*/total_offset, /*src_lane=*/src_meta.src_rdma_rank);

        // Determine the final dst token idx in the recv buffer
        auto token_idx_in_recv_x = post_perm_idx == nullptr ? recv_token_idx : post_perm_idx[recv_token_idx];

        // Increment `total_offset` of next token for `src_rdma_rank`
        (lane_id == src_meta.src_rdma_rank) ? (++total_offset) : 0;

        // Get TMA load bytes, including hidden states and lse
        bool lse_aligned = lse_bytes % 16 == 0; // REVIEW: why need to check 16-bytes alignment here ?

#pragma unroll
        // TMA-copy token from NVL recv buffer to TMA buffer in shared memory
        // NOTE: due to the shared memory size limit, we can not issue all in one go,
        // thus we issue the hidden states iteratively for each data group,
        // while the last data group also includes the lse if aligned
        for (int g = 0; g < kNumDataGroups; ++g) {
          auto hidval_offsets = hidden_bytes * g;
          auto tma_load_bytes = (g < kNumDataGroups - 1) ? hidden_bytes : (hidden_bytes + (lse_aligned ? lse_bytes : 0));
          auto recv_x_ptr = (g == 0) ? recv_x : ((g == 1) ? recv_x_2nd : recv_x_3rd);

          if (lane_id == 0) { // issued by lane0
            // Wait for TMA-copy for previous data groups to be finished
            if (g > 0)
              tma_store_wait();

            tma_load_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/token_ptr_in_buffer + hidval_offsets,
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
                /*gmem_ptr=*/recv_x_ptr + token_idx_in_recv_x * hidden_int4,
                /*num_bytes=*/hidden_bytes,
                /*evict_first=*/false);
          }
          __syncwarp();
        }

        // Copy lse of the token from TMA buffer in shared memory to `recv_lse`
        token_ptr_in_buffer += hidden_bytes * kNumDataGroups;
        if (lse_aligned) { // if aligned to 16 bytes, use TMA copy
          if (lane_id == 0) { // issued by lane0
            tma_store_1d(
                /*smem_ptr=*/tma_buffer + hidden_bytes,
                /*gmem_ptr=*/recv_lse + token_idx_in_recv_x * num_heads,
                /*num_bytes=*/lse_bytes,
                /*evict_first=*/false);
          }
          __syncwarp();
        } else { // if not aligned, use warp copy
          UNROLLED_WARP_COPY(
              /*UNROLL_FACTOR=*/1,
              /*LANE_ID=*/lane_id,
              /*N=*/num_heads,
              /*DST=*/recv_lse + token_idx_in_recv_x * num_heads,
              /*SRC=*/reinterpret_cast<float*>(token_ptr_in_buffer),
              /*LD_FUNC=*/ld_nc_global, // non-cached load
              /*ST_FUNC=*/st_na_global // non-cached store
          );
        }

        // Copy src meta to `recv_src_meta` for group_reduce stage
        // NOTE: here we don't apply `post_perm_idx` to `recv_src_meta`
        token_ptr_in_buffer += lse_bytes;
        if (lane_id == 0 and !kCachedMode) {
          st_na_global(recv_src_meta + recv_token_idx, src_meta); // non-cached store
        }

        // Wait TMA store to be finished
        tma_store_wait();
        __syncwarp();
      }

      // Update NVL queue head
      if (lane_id == 0) {
        st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx); // system scope, relaxed order
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce Kernel
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename dtype_t,
    typename comm_dtype_t,
    typename reduce_dtype_t,
    ReduceOp kReduceOp,
    bool kAccReduce,
    bool kLowLatencyMode,
    int kNumDataGroups,
    int kMaxNumHeads,
    int kNumRDMARanks,
    int kNumMaxSrcRDMARanks,
    int kNumTMAStages,
    int kNumTMALoadBytes,
    int kNumTMABufferBytesPerStage,
    int kNumTMABytesPerSenderWarp,
    int kNumTMABytesPerForwarderWarp,
    int kNumForwarderWarps,
    int kNumWarpsPerForwarder,
    int kNumForwarders,
    int kNumRDMAReceivers,
    bool kHasKernelBarrier,
    bool kIsCachedNotifyFused>
GLOBAL_LAUNCH_BOUNDS(get_num_threads_group_reduce(kNumForwarders), 1)
void group_reduce_kernel(
    /* 1st group of input / output data*/
    int4* reduced_x,
    float* reduced_lse,
    const int4* x,
    const float* lse,
    /* 2nd group of input / output data*/
    int4* reduced_x_2nd,
    const int4* x_2nd,
    /* other metadata */
    const bool* is_reduced_token_in_rank,
    const int* reduced_rdma_head,
    const int* reduced_nvl_head,
    const SourceMeta* src_meta,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* gbl_rank_prefix_sum,
    const int64_t* pre_perm_idx,
    int num_reduced_tokens,
    int hidden_size,
    int num_heads,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_send_tokens,
    int num_max_nvl_chunked_recv_tokens,
    int rank,
    int num_ranks,
    /* other metadata for optional cached notify */
    const size_t rdma_clean_offset,
    const size_t rdma_num_int_clean,
    const size_t nvl_clean_offset,
    const size_t nvl_num_int_clean,
    int** barrier_signal_ptrs,
    const nvshmem_team_t rdma_team,
    /* other metadata for optional kernel barrier */
    magi_attn_ext::KernelBarrierView kernel_barrier_view) {
  // Optional kernel barrier arrive
  if constexpr (kHasKernelBarrier) {
    kernel_barrier_view.arrive();
  }

  // Get thread info
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
  const auto num_threads = static_cast<int>(blockDim.x), thread_id = static_cast<int>(threadIdx.x);
  const auto warp_id = thread_id / WARP_SIZE, lane_id = get_lane_id();

  // Get rank and channel Info
  const auto num_channels = num_sms / 2, channel_id = sm_id / 2;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  const bool is_forwarder = sm_id % 2 == 1;

  // Optional cached notify and grid sync
  if constexpr (kIsCachedNotifyFused) {
    auto grid = cg::this_grid();
    if (sm_id == 0) {
      cached_notify_func<kLowLatencyMode>(
          rdma_clean_offset,
          rdma_num_int_clean,
          nvl_clean_offset,
          nvl_num_int_clean,
          rdma_buffer_ptr,
          buffer_ptrs,
          barrier_signal_ptrs,
          num_threads,
          thread_id,
          kNumRDMARanks,
          rdma_rank,
          nvl_rank,
          rdma_team);
    }
    grid.sync();
  }

  // Get size info
  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  constexpr int kCommDtypePerDtype = sizeof(dtype_t) / sizeof(comm_dtype_t);
  constexpr bool isLowPrecisionComm = !std::is_same_v<dtype_t, comm_dtype_t>;

  constexpr int kNumMaxSrcNVLRanks = NUM_MAX_NVL_PEERS;
  constexpr bool kIsLSEReduce = kReduceOp == ReduceOp::LSE;
  constexpr int max_num_heads = kIsLSEReduce ? kMaxNumHeads : 1;
  constexpr int max_num_shared_warps = static_max(kNumForwarders, kNumRDMAReceivers);
  constexpr int max_num_shared_src_ranks = static_max(kNumRDMARanks, NUM_MAX_NVL_PEERS);
  GRPCOLL_STATIC_ASSERT(max_num_shared_src_ranks <= WARP_SIZE, "Invalid number of RDMA peers");
  GRPCOLL_STATIC_ASSERT(kNumForwarders == kNumRDMAReceivers + NUM_MAX_NVL_PEERS, "Invalid number of forwarders and receivers");
  GRPCOLL_STATIC_ASSERT(kNumForwarders >= kNumRDMARanks, "Invalid number of forwarders");
  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 and kNumDataGroups <= 2, "Invalid number of data groups");

  const int hidden_int4 = hidden_size / kDtypePerInt4, hidden_bytes = hidden_int4 * sizeof(int4);
  const int hidden_int4_comm = hidden_int4 / kCommDtypePerDtype, hidden_bytes_comm = hidden_int4_comm * sizeof(int4);
  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_heads), num_bytes_per_token_comm = get_num_bytes_per_token(hidden_int4_comm, num_heads);
  const auto total_num_bytes_per_token_comm =
      num_bytes_per_token_comm + hidden_bytes_comm * (kNumDataGroups - 1); // for other groups, we only transfer distinct hidden states
  const auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks; // split NVL queue for each RDMA peer
  const int head_dim = kIsLSEReduce ? hidden_size / num_heads : -1;

  /** NOTE: Determine warp role and its target warp id
   * For Forwarder (Odd SMs):
   *  1. the first `kNumForwarders` warps are `kNVL2RDMAForwarder`,
   *    combining the received tokens from all NVL peers in this node (as NVL consumers),
   *    and then forwarding them to target RDMA peer (as RDMA producers),
   *    each warp for one token in round-robin way
   *  2. the rest warp is `kCoordinator`, to update the minimum NVL head for `kNVL2RDMAForwarder`
   *
   * For Sender/Receiver (Even SMs):
   *  1. the first `NUM_MAX_NVL_PEERS` warps are `kNVLSender`,
   *    copying the corr. channel of tokens to NVL buffer of each NVL peer in this node (as NVL producers),
   *    each warp for one NVL peer and each lane for one RDMA peer
   *  2. the next `kNumRDMAReceivers` warps are `kRDMAReceiver`,
   *    combining the received tokens from all RDMA peers and copy to output buffer (as RDMA consumers),
   *    each warp for one token in round-robin way
   *  3. the rest warp is `kCoordinator`, to update the minimum RDMA head for `kRDMAReceiver`
   */
  enum class WarpRole { kNVLSender, kNVL2RDMAForwarder, kRDMAReceiver, kCoordinator };
  auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
      if (warp_id < kNumForwarders) {
        return {WarpRole::kNVL2RDMAForwarder, (warp_id + channel_id) % kNumForwarders};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    } else {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kNVLSender, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else if (warp_id < kNumForwarders) {
        return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    }
  }();
  const auto warp_role = role_meta.first;
  const auto target_warp_id = role_meta.second;

  // Warp-specialized working
  if (warp_role == WarpRole::kNVLSender) {
    const auto dst_nvl_rank = target_warp_id;

    // Get NVL buffers
    //  `nvl_channel_x_1st`: shape=(num_channels, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_bytes_per_token_comm), dtype=uint8_t
    //  `nvl_channel_x_2nd`: shape=(num_channels, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, hidden_bytes_comm), dtype=uint8_t
    //  `nvl_channel_head`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //  `nvl_channel_tail`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    // NOTE: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
    auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
    auto nvl_channel_x = AsymBuffer<uint8_t>(
                             dst_buffer_ptr,
                             /*num_elems=*/num_max_nvl_chunked_recv_tokens * total_num_bytes_per_token_comm,
                             /*num_ranks=*/NUM_MAX_NVL_PEERS,
                             /*sm_id=*/channel_id,
                             /*num_sms=*/num_channels,
                             /*offset=*/nvl_rank)
                             .advance_also(local_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(
            local_buffer_ptr, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/dst_nvl_rank)
            .advance_also(dst_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(
            dst_buffer_ptr, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/nvl_rank)
            .advance_also(local_buffer_ptr);

    // Prepare TMA buffer and init mbarrier
    extern __shared__ __align__(1024) uint8_t smem_tma_buffer[]; // REVIEW: why aligned to 1024 bytes ?
    auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumTMABytesPerSenderWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
    if (lane_id == 0) {
      mbarrier_init(tma_mbarrier, /*arrive_count=*/1); // only lane0 participates
      fence_view_async_shared();
      fence_barrier_init();
    }
    __syncwarp();

    // Get tasks for each RDMA peer indicated by `lane_id`,
    // i.e. the [token_start_idx, token_end_idx] in `x` for each RDMA peer w.r.t. dst_nvl_rank
    // `gbl_channel_prefix_matrix`: shape=(kNumRanks, kNumChannels), dtype=int
    int token_start_idx = 0, token_end_idx = 0;
    // NOTE: since `token_start_idx` will be updated in the loop,
    // we need to use `[&]` to capture reference instead of `[=]`
    auto is_task_valid = [&]() { return token_start_idx < token_end_idx; };
    if (lane_id < kNumRDMARanks) {
      int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
      token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
      token_end_idx = (prefix_idx == num_channels * num_ranks - 1) ? gbl_rank_prefix_sum[num_ranks - 1] : gbl_channel_prefix_matrix[prefix_idx + 1];
    }
    __syncwarp();

    // Iterate over all tokens and send by chunks
    // NOTE: here the cached value of each lane is only responsible for a single RDMA buffer
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    int current_rdma_idx = channel_id % kNumRDMARanks;
    while (true) {
      // Exit if all lanes' tasks are invalid
      if (all_in_warp(!is_task_valid()))
        break;

      // Decide the next RDMA buffer to send
      bool is_lane_ready = false;
      auto start_time = clock64();
      while (true) {
        // Check if any NVL buffer for any RDMA peer indicated by `lane_id` is empty enough to send a chunk
        int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
        is_lane_ready = lane_id < kNumRDMARanks and is_task_valid() and num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
        if (any_in_warp(is_lane_ready))
          break;

        // Read NVL head and retry later
        if (lane_id < kNumRDMARanks and is_task_valid())
          cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id); // volatile load

        // Timeout check
        if (lane_id < kNumRDMARanks) {
          timeout_check_nvl_sender(
              start_time, channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_channel_head_idx, cached_channel_tail_idx, token_start_idx, token_end_idx);
        }
      }

      // Iterate over each RDMA peer and send in round-robin way
      for (int r = 0; r < kNumRDMARanks; ++r) {
        current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks;

        // If the task for current RDMA peer is not valid or is not ready to send, skip for next round
        if (broadcast_in_warp(/*val=*/(!is_task_valid()) or (!is_lane_ready), /*src_lane=*/current_rdma_idx))
          continue;

        // If not skipped, get and sync token start idx and chunk size for this round
        auto token_idx = static_cast<int64_t>(broadcast_in_warp(/*val=*/token_start_idx, /*src_lane=*/current_rdma_idx));
        int num_tokens_in_chunk = broadcast_in_warp(/*val=*/min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), /*src_lane=*/current_rdma_idx);

        // Send by chunk
        for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
          // Get and sync an empty slot in NVL queue
          // as well as increment `cached_channel_tail_idx`
          int dst_slot_idx = 0;
          if (lane_id == current_rdma_idx) {
            dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
            dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
          }
          dst_slot_idx = broadcast_in_warp(/*val=*/dst_slot_idx, /*src_lane=*/current_rdma_idx);

          // Determine the actual src token idx in the NVL buffer
          auto token_idx_in_x = pre_perm_idx == nullptr ? token_idx : pre_perm_idx[token_idx];

          // Get token ptr in NVL buffer of dst NVL peer
          auto token_ptr_in_buffer = nvl_channel_x.buffer() + dst_slot_idx * total_num_bytes_per_token_comm;

#pragma unroll
          for (int g = 0; g < kNumDataGroups; ++g) {
            auto x_ptr = (g == 0) ? x : x_2nd;
            auto buf_offsets = hidden_bytes_comm * g;
            auto tma_store_bytes = (g == kNumDataGroups - 1) ? num_bytes_per_token_comm : hidden_bytes_comm;

            // Get token ptr in `x` of dst NVL peer
            auto token_ptr_in_x = x_ptr + token_idx_in_x * hidden_int4;

            // TMA-copy token from `x` to TMA buffer in shared memory
            if (lane_id == 0) { // issued by lane0
              // Wait for all previous TMA stores to be finished
              tma_store_wait();

              tma_load_1d(
                  /*smem_ptr=*/tma_buffer,
                  /*gmem_ptr=*/token_ptr_in_x,
                  /*mbar_ptr=*/tma_mbarrier,
                  /*num_bytes=*/hidden_bytes, // NOTE: even for low-precision comm, we still need to copy `hidden_bytes` first
                  /*evict_first=*/true);
              mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
            }
            __syncwarp();

            // Wait TMA load to be finished
            // and flip the `tma_phase` in-place
            mbarrier_wait(tma_mbarrier, /*stage=*/tma_phase);

            // For low-precision comm, we need to downcast the hidden states
            // from `dtype_t` to `comm_dtype_t` in TMA buffer in shared memory
            // resulting in only previous `hidden_bytes_comm` left in loaded `hidden_bytes`
            if constexpr (isLowPrecisionComm) {
              auto hidval_int4_ptr = reinterpret_cast<int4*>(tma_buffer);
#pragma unroll
              for (int i = lane_id; i < hidden_int4_comm; i += WARP_SIZE) {
                hidval_int4_ptr[i] = vec_downcast</*dtype_t=*/dtype_t, /*lp_dtype_t=*/comm_dtype_t, /*vec_dtype_t=*/int4>(hidval_int4_ptr + i * kCommDtypePerDtype);
              }
              // For non-last data group, the fence is required here
              // since we use generic memory proxy above to downcast,
              // while no fence will be added below before issuing next TMA store
              if (g < kNumDataGroups - 1)
                tma_store_fence();
            }

            // For last data group, we also need to copy `src_meta` and `lse` to shared memory
            if (g == kNumDataGroups - 1) {
              // Copy src meta to shared memory
              // NOTE: since we've NOT applied `post_perm_idx` to `recv_src_meta` in group cast stage,
              //    here we don't apply `pre_perm_idx` to `src_meta` either
              if (lane_id == 0) {
                *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes_comm) = ld_nc_global(src_meta + token_idx); // non-cached load
              }

#pragma unroll
              // Copy `lse` to shared memory
              for (int h = lane_id; h < num_heads; h += WARP_SIZE) {
                *reinterpret_cast<float*>(tma_buffer + hidden_bytes_comm + sizeof(SourceMeta) + h * sizeof(float)) =
                    ld_nc_global(lse + token_idx_in_x * num_heads + h); // non-cached load
              }

              // Fence TMA store to wait the TMA buffer for each lane to be ready
              // before issuing the next TMA store by lane0
              // NOTE: it's issued by all lanes, compared to other TMA ops which are only issued by lane0
              tma_store_fence();
            }
            __syncwarp();

            // TMA-copy token from TMA buffer in shared memory
            // to NVL buffer of dst NVL peer
            if (lane_id == 0) {
              tma_store_1d(
                  /*smem_ptr=*/tma_buffer,
                  /*gmem_ptr=*/token_ptr_in_buffer + buf_offsets,
                  /*num_bytes=*/tma_store_bytes, // NOTE: only previous `num_bytes_per_token_comm` left
                  /*evict_first=*/false);
            }
          }
        }

        // Update token start idx for next round
        lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
      }

      // Wait for all previous TMA stores to be finished
      tma_store_wait();
      __syncwarp();

      // Update NVL tail of the dst NVL peer
      if (lane_id < kNumRDMARanks and is_lane_ready) {
        st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx); // system scope, release order
      }
    }
  } else { // warp_role == WarpRole::kNVL2RDMAForwarder | warp_role == WarpRole::kRDMAReceiver | warp_role == WarpRole::kCoordinator
    // Get RDMA buffer
    //  `rdma_channel_data_1st`: shape=(num_channels, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, num_bytes_per_token_comm), dtype=int8_t
    //  `rdma_channel_data_2nd`: shape=(num_channels, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, hidden_bytes_comm), dtype=int8_t
    //  `rdma_channel_head`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
    //  `rdma_channel_tail`: shape=(num_channels, kNumRDMARanks), dtype=uint64_t
    auto rdma_channel_data = SymBuffer<int8_t, /*kDecoupled=*/true>(
        rdma_buffer_ptr,
        /*num_elems=*/num_max_rdma_chunked_recv_tokens * total_num_bytes_per_token_comm,
        /*num_ranks=*/kNumRDMARanks,
        /*sm_id=*/channel_id,
        /*num_sms=*/num_channels);
    auto rdma_channel_head =
        SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);
    auto rdma_channel_tail =
        SymBuffer<uint64_t, /*kDecoupled=*/false>(rdma_buffer_ptr, /*num_elems=*/1, /*num_ranks=*/kNumRDMARanks, /*sm_id=*/channel_id, /*num_sms=*/num_channels);

    // Get NVL Buffer
    //  `nvl_channel_x_1st`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks, num_max_nvl_chunked_recv_tokens_per_rdma, num_bytes_per_token_comm), dtype=uint8_t
    //  `nvl_channel_x_2nd`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks, num_max_nvl_chunked_recv_tokens_per_rdma, hidden_bytes_comm), dtype=uint8_t
    //  `nvl_channel_head`: shape=(NUM_MAX_NVL_PEERS, num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //  `nvl_channel_tail`: shape=(num_channels, NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    void* local_nvl_buffer = buffer_ptrs[nvl_rank];
    void* nvl_buffers[NUM_MAX_NVL_PEERS];
#pragma unroll
    for (int r = 0; r < NUM_MAX_NVL_PEERS; ++r)
      nvl_buffers[r] = buffer_ptrs[r];
    auto nvl_channel_x = AsymBuffer<uint8_t>(
                             local_nvl_buffer,
                             /*num_elems=*/kNumRDMARanks * num_max_nvl_chunked_recv_tokens_per_rdma * total_num_bytes_per_token_comm,
                             /*num_ranks=*/NUM_MAX_NVL_PEERS,
                             /*sm_id=*/channel_id,
                             /*num_sms=*/num_channels)
                             .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_head =
        AsymBuffer<int, /*kNumRanks=*/NUM_MAX_NVL_PEERS>(
            nvl_buffers, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels, /*offset=*/nvl_rank)
            .advance_also(local_nvl_buffer);
    auto nvl_channel_tail =
        AsymBuffer<int>(local_nvl_buffer, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS, /*sm_id=*/channel_id, /*num_sms=*/num_channels)
            .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

    // Reducer warp synchronization
    // for kNVL2RDMAForwarder:
    //    `shared_head`: shape=(kNumForwarders, NUM_MAX_NVL_PEERS), dtype=int
    //    `warp_retired`: shape=(kNumForwarders), dtype=bool
    //    `sync_forwarder_smem`: synchronize warps of `kNumForwarders` and `kCoordinator` warps
    // for kRDMAReceiver:
    //    `shared_head`: shape=(kNumRDMAReceivers, kNumRDMARanks), dtype=int
    //    `warp_retired`: shape=(kNumRDMAReceivers), dtype=bool
    //    `sync_rdma_receiver_smem`: synchronize warps of `kRDMAReceiver` and `kCoordinator` warps
    __shared__ volatile int shared_head[max_num_shared_warps][max_num_shared_src_ranks];
    __shared__ volatile bool warp_retired[max_num_shared_warps];

    auto sync_forwarder_smem = [=]() { sync_warp_group(/*group_flag=*/0, /*group_size=*/(kNumForwarders + 1) * WARP_SIZE); };
    auto sync_rdma_receiver_smem = [=]() { sync_warp_group(/*group_flag=*/1, /*group_size=*/(kNumRDMAReceivers + 1) * WARP_SIZE); };

    // Prepare some static shared memory for temporary lse buffers
    // which will be read frequently while reducing the hidden values of some single token
    __shared__ reduce_dtype_t shared_reduced_lse_buf[max_num_shared_warps][max_num_heads]; // reduced lse buffer for each head and each warp
    __shared__ reduce_dtype_t
        shared_old_lse_rescale_weight_buf[max_num_shared_warps][max_num_heads]; // the rescale weight of old `reduced_lse` for each head and each warp

    if (warp_role == WarpRole::kNVL2RDMAForwarder) {
      // Determine warp group
      // NOTE: we have `kNumForwarders` warps as `kNVL2RDMAForwarder`
      // and each `kNumWarpsPerForwarder` warps forms as a warp group for one RDMA peer
      const auto dst_rdma_rank = target_warp_id / kNumWarpsPerForwarder, sub_warp_id = target_warp_id % kNumWarpsPerForwarder;
      auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
      auto sync_forwarder_warp_group = [=]() {
        if constexpr (kNumWarpsPerForwarder == 1) {
          __syncwarp();
        } else {
          sync_warp_group(/*group_flag=*/dst_rdma_rank + 2, /*group_size=*/kNumWarpsPerForwarder * WARP_SIZE);
        }
      };
      // NOTE: since `kNumForwarderWarps` is set to 24 when `kNumRDMARanks <= 24`,
      // so when `kNumRDMARanks in (12, 24]`, `kNumWarpsPerForwarder` will always be 1,
      // and if `kNumRDMARanks in (24, 32]`, `kNumForwarderWarps` will be set to 32 thus `kNumWarpsPerForwarder` will be still 1
      // thus no worry to reach the barrier limit (`kNumWarpsPerForwarder > 1` and `kNumRDMARanks > 14`)
      GRPCOLL_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= NUM_MAX_BARRIERS, "Barriers are not enough");

      // Prepare TMA buffer and init mbarrier
      // kNumTMABufferBytesPerStage:
      //    1. TMA load buffer: kNumMaxSrcNVLRanks * kNumTMALoadBytes
      //    2. TMA store buffer: 1 * kNumTMALoadBytes
      //    3. mbarrier: 1 * 8 bytes
      //    4. aligned to 16 bytes
      GRPCOLL_STATIC_ASSERT(kNumTMAStages * kNumTMABufferBytesPerStage <= kNumTMABytesPerForwarderWarp, "TMA buffer is not larger enough");

      extern __shared__ __align__(1024) uint8_t smem_buffer[]; // REVIEW: why aligned to 1024 bytes ?
      auto smem_ptr = smem_buffer + target_warp_id * kNumTMAStages * kNumTMABufferBytesPerStage;
      auto tma_mbarrier = [=](const int& stage_idx) {
        return reinterpret_cast<uint64_t*>(smem_ptr + stage_idx * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (kNumMaxSrcNVLRanks + 1));
      };
      uint32_t tma_phase[kNumTMAStages] = {0};
      if (lane_id < kNumTMAStages) {
        mbarrier_init(tma_mbarrier(lane_id), /*arrive_count=*/WARP_SIZE); // all lanes participate
        fence_view_async_shared();
        fence_barrier_init();
      }
      __syncwarp();

      // Advance (in-place) to the corr. offset for dst RDMA peer in each NVL buffer
      nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * total_num_bytes_per_token_comm);
      nvl_channel_head.advance(dst_rdma_rank);
      nvl_channel_tail.advance(dst_rdma_rank);

      // Clean shared memory and sync with `kCoordinator`
      lane_id < NUM_MAX_NVL_PEERS ? (shared_head[target_warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (warp_retired[target_warp_id] = false) : false;
      sync_forwarder_smem();

      // Get forward tasks
      //  `rdma_channel_prefix_matrix`: shape=(kNumRDMARanks, kNumChannels), dtype=int
      //  `rdma_rank_prefix_sum`: shape=(kNumRDMARanks,), dtype=int
      //  `reduced_nvl_head`: shape=(num_rdma_recv_tokens, NUM_MAX_NVL_PEERS), dtype=int
      int cached_nvl_channel_tail_idx = 0;
      int num_tokens_to_group_reduce = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
      int num_tokens_prefix = channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
      num_tokens_to_group_reduce -= num_tokens_prefix;
      num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
      reduced_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

      // Iterate over all tokens, and group_reduce + forward by chunks
      for (int token_start_idx = 0; token_start_idx < num_tokens_to_group_reduce; token_start_idx += num_max_rdma_chunked_send_tokens) {
        // Check destination queue emptiness, or wait a buffer to be released
        auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_group_reduce);
        auto num_chunked_tokens = token_end_idx - token_start_idx;

        // Wait for queue empty enough to group_reduce `num_chunked_tokens` tokens
        auto start_time = clock64();
        while (sub_warp_id == 0 and lane_id == 0) {
          // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
          // NOTE: `token_start_idx` is the actual tail here
          auto rdma_head_idx = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)); // volatile load
          int num_used_slots = token_start_idx - rdma_head_idx;
          if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
            break;

          // Timeout check
          timeout_check_nvl2rdma_forwarder_rdma_head(start_time, channel_id, rdma_rank, nvl_rank, dst_rdma_rank, rdma_head_idx, token_start_idx, num_chunked_tokens);
        }
        sync_forwarder_warp_group();

        // Reduce and write to the RDMA send buffer by this warp group,
        // each warp for one token in this chunk
        for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder) {
          // Read expected NVL head
          // each lane for one NVL peer
          int expected_head = -1;
          if (lane_id < NUM_MAX_NVL_PEERS) {
            expected_head = ld_nc_global(reduced_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id); // non-cached load
          }

          // Wait all NVL tails for each src NVL peer to be ready
          // which are updated by `kNVLSender`
          start_time = clock64();
          while (cached_nvl_channel_tail_idx <= expected_head) {
            cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id)); // system scope, acquire order

            // Timeout check
            if (lane_id < NUM_MAX_NVL_PEERS) {
              timeout_check_nvl2rdma_forwarder_nvl_head(
                  start_time,
                  channel_id,
                  rdma_rank,
                  nvl_rank,
                  lane_id,
                  dst_rdma_rank,
                  cached_nvl_channel_tail_idx,
                  token_idx,
                  num_tokens_to_group_reduce,
                  sub_warp_id,
                  expected_head);
            }
          }

          // Get an empty slot in RDMA queue
          auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;

          // Get token ptr in RDMA send buffer
          void* token_ptr_in_rdma_buffer = send_buffer + rdma_slot_idx * total_num_bytes_per_token_comm;

          // Define `get_hidval_ptr_fn` and `get_lse_fn`
          auto get_hidval_ptr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
            return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * total_num_bytes_per_token_comm) + hidden_int4_idx;
          };
          auto get_lse_fn = [&](int src_nvl_rank, int slot_idx, int head_idx) -> float {
            return ld_nc_global(
                reinterpret_cast<float*>(
                    nvl_channel_x.buffer(src_nvl_rank) + slot_idx * total_num_bytes_per_token_comm + hidden_bytes_comm * kNumDataGroups + sizeof(SourceMeta)) +
                head_idx);
          };

          // NOTE: for `kReduceOp == ReduceOp::AVG`,
          // it's incorrect to partial-reduce it by `kNVL2RDMAForwarder` in advance,
          // but leave it to `kRDMAReceiver` to global-reduce
          constexpr auto kForwardReduceOp = kReduceOp == ReduceOp::AVG ? ReduceOp::SUM : kReduceOp;

          // Reduce this token
          //  `head_idx`: the head idx of the token in NVL buffer
          //  `reduced_token`: the token ptr in output buffer to store the reduced token
          //  `reduced_lse`: the lse ptr in output buffer to store the reduced lse
          //  `num_max_recv_tokens`: the queue size of NVL buffer
          //  `get_hidval_ptr_fn`: get the hidden value ptr of the token in NVL buffer
          //  `get_lse_fn`: get lse of the token in NVL buffer
          reduce_token_in_warp</*dtype_t=*/comm_dtype_t, // NOTE: for forwarders, no need to upcast back to `dtype_t`
                               /*comm_dtype_t=*/comm_dtype_t,
                               /*reduced_dtype_t=*/reduce_dtype_t,
                               /*kReduceOp=*/kForwardReduceOp,
                               /*kAccReduce=*/false, // no need to acc-reduce for forwarder
                               /*kGlobalReduce=*/false, // no need to global-reduce for forwarder
                               /*kNumDataGroups=*/kNumDataGroups,
                               /*kMaxNumHeads=*/max_num_heads,
                               /*kNumReduceWarps=*/kNumForwarders,
                               /*kNumRanks=*/kNumMaxSrcNVLRanks,
                               /*kMaxNumSrcRanks=*/kNumMaxSrcNVLRanks,
                               /*kUseTMA=*/true,
                               /*kNumTMAStages=*/kNumTMAStages,
                               /*kNumTMALoadBytes=*/kNumTMALoadBytes,
                               /*kNumTMABufferBytesPerStage=*/kNumTMABufferBytesPerStage>(
              /*is_token_in_rank=*/expected_head >= 0,
              /*token_idx_in_queue=*/expected_head,
              /*reduce_warp_id=*/target_warp_id,
              /*lane_id=*/lane_id,
              /*hidden_int4=*/hidden_int4_comm,
              /*num_heads=*/num_heads,
              /*head_dim=*/head_dim,
              /*num_global_ranks=*/num_ranks,
              /*reduced_token=*/static_cast<int4*>(token_ptr_in_rdma_buffer),
              /*reduced_lse=*/reinterpret_cast<float*>(static_cast<int8_t*>(token_ptr_in_rdma_buffer) + hidden_bytes_comm * kNumDataGroups + sizeof(SourceMeta)),
              /*reduced_token_2nd=*/static_cast<int4*>(token_ptr_in_rdma_buffer) + hidden_int4_comm,
              /*num_max_recv_tokens=*/num_max_nvl_chunked_recv_tokens_per_rdma,
              /*is_token_in_global_rank=*/nullptr, // no need to global-reduce for forwarder
              /*get_hidval_ptr_fn=*/get_hidval_ptr_fn,
              /*get_lse_fn=*/get_lse_fn,
              /*shared_reduced_lse=*/shared_reduced_lse_buf,
              /*shared_old_lse_rescale_weight=*/shared_old_lse_rescale_weight_buf,
              /*smem_ptr=*/smem_ptr,
              /*tma_phase=*/tma_phase);

          // Update NVL head
          /** NOTE: for those `-1` entries of the original `reduced_nvl_head` generated in group-cast stage,
           * we've already in-place updated them in `cached_notify` to the valid next position `p`,
           * but encoded to `-p-1` to maintain them still negative like `-1`
           * and we can decode the correct next expect head by `-expected_head - 1`
           */
          if (lane_id < NUM_MAX_NVL_PEERS) {
            shared_head[target_warp_id][lane_id] = expected_head < 0 ? decode(expected_head) : (expected_head + 1);
          }
        }
        sync_forwarder_warp_group();

        // RDMA-copy this chunk of tokens from RDMA send buffer of this RDMA rank
        // to RDMA recv buffer of dst RDMA peer
        if (sub_warp_id == kNumWarpsPerForwarder - 1) { // issued by last warp in the warp group
          const int dst_pe = get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank);

          if (dst_rdma_rank != rdma_rank) { // dst RDMA peer
            auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
            const size_t num_bytes_per_msg = num_chunked_tokens * total_num_bytes_per_token_comm;
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * total_num_bytes_per_token_comm);
            const auto src_ptr = reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * total_num_bytes_per_token_comm);
            nvshmemi_ibgda_put_nbi_warp</*kAlwaysDoPostSend=*/true>(
                /*req_rptr=*/dst_ptr,
                /*req_lptr=*/src_ptr,
                /*bytes=*/num_bytes_per_msg,
                /*dst_pe=*/dst_pe,
                /*qp_id=*/channel_id,
                /*lane_id=*/lane_id,
                /*message_idx=*/0);
          } else { // this RDMA rank
            // Already in its own recv buffer, so no need to copy
            memory_fence(); // NOTE: use lighter fence for local memory operations
          }
          __syncwarp();

          // Update RDMA tail by atomic-add
          if (lane_id == 0) { // issued by lane0 of the last warp
            nvshmemi_ibgda_amo_nonfetch_add(
                /*rptr=*/rdma_channel_tail.buffer(rdma_rank),
                /*value=*/num_chunked_tokens,
                /*pe=*/dst_pe,
                /*qp_id=*/channel_id,
                /*is_local_copy=*/dst_rdma_rank == rdma_rank);
          }
        }
      }
      __syncwarp();

      // Mark this warp as retired
      if (lane_id == 0) {
        warp_retired[target_warp_id] = true;
      }
    } else if (warp_role == WarpRole::kRDMAReceiver) {
      // Clean shared memory and sync with the `kCoordinator` warp
      lane_id < kNumRDMARanks ? (shared_head[target_warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (warp_retired[target_warp_id] = false) : 0;
      sync_rdma_receiver_smem();

      // The same task as the `kRDMASender` in the group_cast stage
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_reduced_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over all tokens, and group_reduce + write to `reduced_x`
      // each warp for one token
      int cached_channel_tail_idx = 0;
      for (int64_t token_idx = token_start_idx + target_warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
        // Read expected head from each RDMA peer
        //  `reduced_rdma_head`: shape=[num_reduced_tokens, kNumRDMARanks]
        int expected_head = -1;
        if (lane_id < kNumRDMARanks) {
          expected_head = ld_nc_global(reduced_rdma_head + token_idx * kNumRDMARanks + lane_id); // non-cached load

          // Record RDMA head into shared memory
          // to let `kCoordinator` update the minimum one across all RDMA ranks
          // REVIEW: why not `expected_head + 1` like others when valid ?
          shared_head[target_warp_id][lane_id] = expected_head < 0 ? decode(expected_head) : expected_head;
        }

        // Wait the queue non-empty for each RDMA peer
        auto start_time = clock64();
        while (cached_channel_tail_idx <= expected_head) {
          // Read RDMA tail updated by `kNVL2RDMAForwarder`
          cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))); // system scope, acquire order

          // Timeout check
          timeout_check_rdma_recevier(start_time, channel_id, rdma_rank, nvl_rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
        }
        __syncwarp();

        // Define `get_hidval_ptr_fn` and `get_lse_fn`
        auto get_hidval_ptr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
          return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * total_num_bytes_per_token_comm) + hidden_int4_idx;
        };
        auto get_lse_fn = [&](int src_rdma_rank, int slot_idx, int head_idx) -> float {
          return ld_nc_global(
              reinterpret_cast<const float*>(
                  rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * total_num_bytes_per_token_comm + hidden_bytes_comm * kNumDataGroups + sizeof(SourceMeta)) +
              head_idx);
        };

        // NOTE: for `kReduceOp == ReduceOp::AVG`,
        // it's incorrect to partial-reduce it by `kNVL2RDMAForwarder`,
        // but let `kRDMAReceiver` global-reduce it here
        constexpr bool kGlobalReduce = kReduceOp == ReduceOp::AVG;

        // Reduce this token
        //  `head_idx`: the head idx of the token in RDMA buffer
        //  `reduced_token`: the token ptr in output buffer to store the reduced token
        //  `reduced_lse`: the lse ptr in output buffer to store the reduced lse
        //  `num_max_recv_tokens`: the queue size of RDMA buffer
        //  `get_hidval_ptr_fn`: get the hidden value ptr of the token in RDMA buffer
        //  `get_lse_fn`: get lse of the token in RDMA buffer
        uint32_t dummy_tma_phases[kNumTMAStages];
        reduce_token_in_warp</*dtype_t=*/dtype_t,
                             /*comm_dtype_t=*/comm_dtype_t,
                             /*reduced_dtype_t=*/reduce_dtype_t,
                             /*kReduceOp=*/kReduceOp,
                             /*kAccReduce=*/kAccReduce,
                             /*kGlobalReduce*/ kGlobalReduce,
                             /*kNumDataGroups=*/kNumDataGroups,
                             /*kMaxNumHeads=*/max_num_heads,
                             /*kNumReduceWarps=*/kNumRDMAReceivers,
                             /*kNumRanks=*/kNumRDMARanks,
                             /*kMaxNumSrcRanks=*/kNumMaxSrcRDMARanks,
                             /*kUseTMA=*/false,
                             /*kNumTMAStages=*/kNumTMAStages,
                             /*kNumTMALoadBytes=*/0,
                             /*kNumTMABufferBytesPerStage=*/0>(
            /*is_token_in_rank=*/expected_head >= 0,
            /*token_idx_in_queue=*/expected_head,
            /*reduce_warp_id=*/target_warp_id,
            /*lane_id=*/lane_id,
            /*hidden_int4=*/hidden_int4,
            /*num_heads=*/num_heads,
            /*head_dim=*/head_dim,
            /*num_global_ranks=*/num_ranks,
            /*reduced_token=*/reduced_x + token_idx * hidden_int4,
            /*reduced_lse=*/reduced_lse + token_idx * num_heads,
            /*reduced_token_2nd=*/reduced_x_2nd + token_idx * hidden_int4,
            /*num_max_recv_tokens=*/num_max_rdma_chunked_recv_tokens,
            /*is_token_in_global_rank=*/is_reduced_token_in_rank + token_idx * num_ranks,
            /*get_hidval_ptr_fn=*/get_hidval_ptr_fn,
            /*get_lse_fn=*/get_lse_fn,
            /*shared_reduced_lse=*/shared_reduced_lse_buf,
            /*shared_old_lse_rescale_weight=*/shared_old_lse_rescale_weight_buf,
            /*smem_ptr=*/nullptr,
            /*tma_phases=*/dummy_tma_phases);
      }
      __syncwarp();

      // Mark this warp as retired
      if (lane_id == 0) {
        warp_retired[target_warp_id] = true;
      }
    } else { // WarpRole::kCoordinator, for both forwarder and sender/receiver, only one warp
      // Sync with either `kNVL2RDMAForwarder` or `kRDMAReceiver` warps
      // to wait all shared memory cleaned before accessing
      is_forwarder ? sync_forwarder_smem() : sync_rdma_receiver_smem();
      constexpr auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;
      GRPCOLL_STATIC_ASSERT(kNumForwarders % kNumRDMARanks == 0, "Invalid number of forwarder warps");

      // Check retirement and update minimum head
      // for either `kNVL2RDMAForwarder` or `kRDMAReceiver`
      int last_rdma_head = 0;
      int last_nvl_head[kNumRDMARanks] = {0};
      int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
      int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
      GRPCOLL_STATIC_ASSERT(kNumForwarderWarps <= WARP_SIZE, "Invalid number of forwarder warps");
      while (true) {
        // Check if all warps of either `kNVL2RDMAForwarder` or `kRDMAReceiver` are retired
        if (is_forwarder) { // `kNVL2RDMAForwarder`
          if (all_in_warp(lane_id >= kNumForwarders or warp_retired[lane_id]))
            break;
        } else { // `kRDMAReceiver`
          if (all_in_warp(lane_id >= kNumRDMAReceivers or warp_retired[lane_id]))
            break;
        }

        // Update minimum head for either RDMA or NVL ranks
        if (is_forwarder) { // `kNVL2RDMAForwarder`
#pragma unroll
          for (int i = 0; i < kNumRDMARanks; ++i) {
            // Find minimum NVL head
            int min_head = INT_MAX;
#pragma unroll
            for (int j = 0; j < num_warps_per_rdma_rank; ++j) {
              if (!warp_retired[i * num_warps_per_rdma_rank + j])
                min_head = min(min_head, shared_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
            }

            // Update NVL head
            if (lane_id < NUM_MAX_NVL_PEERS and min_head != INT_MAX and min_head > last_nvl_head[i])
              st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head); // system scope, relaxed order
          }
        } else { // `kRDMAReceiver`
          // Find minimum RDMA head
          int min_head = INT_MAX;
#pragma unroll
          for (int i = 0; i < kNumRDMAReceivers; ++i) {
            if (!warp_retired[i]) {
              min_head = min(min_head, shared_head[i][dst_rdma_rank]);
            }
          }

          // Update RDMA head by atomic add
          // REVIEW: why need to check `min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens` ?
          if (lane_id < kNumRDMARanks and min_head != INT_MAX and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens) {
            nvshmemi_ibgda_amo_nonfetch_add(
                /*rptr=*/rdma_channel_head.buffer(rdma_rank),
                /*value=*/min_head - last_rdma_head,
                /*pe=*/get_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                /*qp_id=*/channel_id + num_channels,
                /*is_local_copy=*/dst_rdma_rank == rdma_rank);
            last_rdma_head = min_head;
          }
        }

        // Nanosleep and let either `kNVL2RDMAForwarder` or `kRDMAReceiver` warps work
        __nanosleep(NUM_WAIT_NANOSECONDS);
      }
    }
  }
}

} // namespace magi_attn_comm::grpcoll::internode
