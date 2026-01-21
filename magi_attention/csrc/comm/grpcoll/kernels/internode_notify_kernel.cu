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

#include "internode.cuh"

template <bool kLowLatencyMode, bool kRequireRecvCount, int kNumThreads, int kNumRDMARanks>
__global__ void notify_group_cast_kernel(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_rdma_rank,
    int* grpcoll_recv_rdma_counter_mapped,
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
    //  1. the first `NUM_MAX_NVL_PEERS` elems: number of send/recv tokens for each NVL rank in this node
    //  2. the last `1` elem: total number of send/recv tokens for this node
    auto rdma_buffer_ptr_int = static_cast<int*>(rdma_buffer_ptr);
    const int meta_elems_per_rdma_rank_int = NUM_MAX_NVL_PEERS + 1;
    auto rdma_recv_num_tokens_mixed = SymBuffer<int, /*kDecoupled=*/true>(rdma_buffer_ptr, /*num_elems=*/meta_elems_per_rdma_rank_int, /*num_ranks=*/kNumRDMARanks);

    // Clean up RDMA buffer of this rank for later meta data switch
    GRPCOLL_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += kNumThreads)
      rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

    // Copy send meta data of this RDMA rank to its local send buffer
    //  `num_tokens_per_rank`: shape=(num_ranks,), dtype=int
    //  `num_tokens_per_rdma_rank`: shape=(kNumRDMARanks,), dtype=int
    //  `rdma_recv_num_tokens_mixed.send_buffer/recv_buffer`: shape=(kNumRDMARanks, meta_elems_per_rdma_rank_int), dtype=int
    GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= kNumThreads, "Invalid number of RDMA peers");
#pragma unroll
    for (int r = thread_id; r < num_ranks; r += kNumThreads) {
      rdma_recv_num_tokens_mixed.send_buffer(r / NUM_MAX_NVL_PEERS)[r % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[r];
    }
    if (thread_id < kNumRDMARanks) {
      rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS] = num_tokens_per_rdma_rank[thread_id];
    }
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
    if (thread_id < kNumRDMARanks and thread_id != rdma_rank) {
      nvshmemi_ibgda_quiet(/*dst_pe=*/get_dst_rdma_rank<kLowLatencyMode>(thread_id, nvl_rank), /*qp_id=*/0);
    }
    __syncthreads();

    // Barrier RDMA team
    // TODO: overlap RDMA barrier and NVL cleaning
    if (thread_id == 0) {
      nvshmem_sync_with_same_gpu_idx<kLowLatencyMode>(rdma_team);
    }
    __syncthreads();

    // Get NVL buffers, sending buffer for dst NVL peer, and receiving buffer for this NVL rank
    //  `nvl_send_num_tokens_per_rank`: shape=(NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //      `nvl_send_num_tokens_per_rank[nvl_rank][r]`: the number of tokens sent from RDMA rank `r` via this NVL rank
    //  `nvl_recv_num_tokens_per_rank`: shape=(NUM_MAX_NVL_PEERS, kNumRDMARanks), dtype=int
    //      `nvl_recv_num_tokens_per_rank[p][r]`: the number of tokens received from RDMA rank `r` via NVL rank `p` to this NVL rank
    auto nvl_recv_buffer = buffer_ptrs[nvl_rank], nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
    auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, /*num_elems=*/kNumRDMARanks, /*num_ranks=*/NUM_MAX_NVL_PEERS);

    // Clean up NVL buffer of this NVL rank for later meta data switch
    auto nvl_buffer_ptr_int = static_cast<int*>(buffer_ptrs[nvl_rank]);
    GRPCOLL_DEVICE_ASSERT(nvl_send_num_tokens_per_rank.total_bytes <= nvl_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += kNumThreads)
      nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
    __syncthreads();

    // Reduce (prefix-summed) number of received tokens from each RDMA peer to this RDMA rank
    // and copy into `recv_rdma_rank_prefix_sum`: shape=(kNumRDMARanks,), dtype=int
    // as well as the total received number to the pinned `grpcoll_recv_rdma_counter` to notify the host
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int r = 0; r < kNumRDMARanks; ++r) {
        sum += rdma_recv_num_tokens_mixed.recv_buffer(r)[NUM_MAX_NVL_PEERS];
        recv_rdma_rank_prefix_sum[r] = sum;
      }

      // Notify the host the total number of received tokens via RDMA if required
      if constexpr (kRequireRecvCount) {
        // Self-rotated wait for the RDMA counter reset by the host
        auto start_time = clock64();
        while (true) {
          auto rdma_recv_counter_value = ld_volatile_global(grpcoll_recv_rdma_counter_mapped);
          if (rdma_recv_counter_value == -1)
            break;

          // Timeout check
          timeout_check_rdma_recv_counter(start_time, thread_id, sum, rdma_recv_counter_value, nvl_rank, rdma_rank);
        }

        *grpcoll_recv_rdma_counter_mapped = sum;
      }
    }

    // P2P-copy to remote `nvl_send_num_tokens_per_rank`
    // in NVL peer indicated by `thread_id`,
    // which hold the number of tokens sent from each RDMA rank resp. via this NVL rank
    if (thread_id < NUM_MAX_NVL_PEERS) {
#pragma unroll
      for (int r = 0; r < kNumRDMARanks; ++r)
        nvl_send_num_tokens_per_rank.buffer(nvl_rank)[r] = rdma_recv_num_tokens_mixed.recv_buffer(r)[thread_id];
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

      // Notify the host the total number of received tokens if required
      if constexpr (kRequireRecvCount) {
        // Self-rotated wait for the counter reset by the host
        auto start_time = clock64();
        while (true) {
          auto recv_counter_value = ld_volatile_global(grpcoll_recv_counter_mapped);
          if (recv_counter_value == -1)
            break;

          // Timeout check
          timeout_check_recv_counter(start_time, thread_id, sum, recv_counter_value, nvl_rank, rdma_rank);
        }

        *grpcoll_recv_counter_mapped = sum;
      }
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
        count_all_nvl_ranks += (is_token_in_rank_uint64 != 0); // NOTE: one `uint64_t` is 8 bytes to cover 8 bools for 8 NVL peers
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

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
__global__ void cached_notify_kernel(
    const int rdma_clean_offset,
    const int rdma_num_int_clean,
    const int nvl_clean_offset,
    const int nvl_num_int_clean,
    int* reduced_rdma_head,
    int num_reduced_tokens,
    int num_channels,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    int* reduced_nvl_head,
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    bool is_cached_group_cast,
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
  } else if (sm_id == 1) { // the second SM is responsible to reset the RDMA head before group_reduce
    // If this is a cached group_cast,
    // no need to reset the rdma head, just return
    if (is_cached_group_cast)
      return;

    // Reset the rdma head, iterating in reverse order
    // each warp is responsible for one channel
    // and each lane in any warp is responsible for one rdma rank of the corr. channel
    if (lane_id < num_rdma_ranks and warp_id < num_channels) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_reduced_tokens, num_channels, warp_id, token_start_idx, token_end_idx);

      // NOTE: `1 << 25` is a heuristic large number
      int last_head = 1 << 25;
      for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
        auto current_head = __ldg(reduced_rdma_head + token_idx * num_rdma_ranks + lane_id);
        if (current_head < 0) {
          reduced_rdma_head[token_idx * num_rdma_ranks + lane_id] = encode(last_head);
        } else {
          last_head = current_head;
        }
      }
    }
  } else { // the rest of SMs are responsible to reset the NVL head before group_reduce
    // If this is a cached group_cast,
    // no need to reset the nvl head, just return
    if (is_cached_group_cast)
      return;

    if (warp_id < num_channels) {
      const auto rest_sm_id = sm_id - 2, num_rest_sms = num_channels * 2 - 2;
      constexpr int tma_batch_size = kNumTMABytesPerWarp - sizeof(uint64_t);
      constexpr int num_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
      constexpr int num_tokens_per_batch = tma_batch_size / num_bytes_per_token;
      GRPCOLL_STATIC_ASSERT(num_bytes_per_token % 16 == 0, "num_bytes_per_token should be divisible by 16");

      // Prepare TMA buffer and init mbarrier
      extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
      auto tma_buffer = smem_tma_buffer + warp_id * kNumTMABytesPerWarp;
      auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
      uint32_t tma_phase = 0;
      if (lane_id == 0) {
        mbarrier_init(tma_mbarrier, /*arrive_count=*/1); // only lane0 participates
        fence_view_async_shared();
        fence_barrier_init();
      }
      __syncwarp();

      // Each rest SM for one dst RDMA peer
      for (int dst_rdma_rank = rest_sm_id; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_rest_sms) {
        // Iterate in reverse order
        int token_start_idx = warp_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id - 1];
        int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + warp_id];
        int rank_prefix = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
        token_start_idx += rank_prefix, token_end_idx += rank_prefix;

        int last_head = 1 << 25; // NOTE: `1 << 25` is a heuristic large number
        for (int batch_end_idx = token_end_idx; batch_end_idx > token_start_idx; batch_end_idx -= num_tokens_per_batch) {
          auto batch_start_idx = max(token_start_idx, batch_end_idx - num_tokens_per_batch);

          // TMA-copy original reduced NVL head to TMA buffer in shared memory
          if (lane_id == 0) {
            tma_load_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/reduced_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                /*mbar_ptr=*/tma_mbarrier,
                /*num_bytes=*/(batch_end_idx - batch_start_idx) * num_bytes_per_token,
                /*evict_first=*/true);
            mbarrier_arrive_and_expect_tx(tma_mbarrier, (batch_end_idx - batch_start_idx) * num_bytes_per_token);
          }

          // Wait for TMA-load to be finished
          mbarrier_wait(tma_mbarrier, tma_phase);
          __syncwarp();

          // Reset those `-1` entries of NVL head
          for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
            if (lane_id < NUM_MAX_NVL_PEERS) {
              auto current_head = reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
              if (current_head < 0) {
                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] = encode(last_head);
              } else {
                last_head = current_head;
              }
            }
          }

          // Fence all lanes to wait for all update to TMA buffer
          // in shared memory to be visible to each other
          // before issuing the next TMA-store
          tma_store_fence();
          __syncwarp();

          // TMA-copy updated NVL head from TMA buffer in shared memory to global memory
          if (lane_id == 0) {
            tma_store_1d(
                /*smem_ptr=*/tma_buffer,
                /*gmem_ptr=*/reduced_nvl_head + batch_start_idx * NUM_MAX_NVL_PEERS,
                /*num_bytes=*/(batch_end_idx - batch_start_idx) * num_bytes_per_token,
                /*evict_first=*/true);
          }

          // Wait for TMA-store to be finished
          tma_store_wait();
          __syncwarp();
        }
      }
    }
  }
}
