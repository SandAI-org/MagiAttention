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
#pragma once

#include "api.cuh"
#include "configs.cuh"
#include "internode_kernel.cuh"
#include "internode_notify_kernel.cuh"

namespace magi_attn_comm::grpcoll::internode {

extern nvshmem_team_t cpu_rdma_team;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumDataGroups, int kNumRDMARanks>
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
    void* recv_src_meta,
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
    int num_channels,
    bool is_cached_group_cast,
    cudaStream_t stream,
    /* other metadata for optional cached notify */
    size_t num_rdma_bytes,
    size_t num_nvl_bytes,
    int** barrier_signal_ptrs,
    /* other metadata for optional kernel barrier */
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier) {
  constexpr int kNumMaxDstRDMARanks = get_num_max_src_rdma_ranks(kNumRDMARanks);
  constexpr int kWarpCopyUnrollStages = 5;
  constexpr int kNumSenderWarps = 7;
  constexpr int kNumThreads = get_num_threads_group_cast(kNumSenderWarps);
  constexpr int kNumWarps = kNumThreads / WARP_SIZE;
  GRPCOLL_STATIC_ASSERT(kNumWarps == kNumSenderWarps + 1 + NUM_MAX_NVL_PEERS, "Invalid number of warps");

  constexpr int kNumTMABytesPerWarp = 27 * 1024; // 27KB
  constexpr int smem_size = kNumTMABytesPerWarp * NUM_MAX_NVL_PEERS; // 27KB * 8 = 216KB < 224KB, can hardly be raised up

  // Fused cached notify check
  size_t rdma_clean_offset = 0;
  size_t rdma_num_int_clean = 0;
  size_t nvl_clean_offset = 0;
  size_t nvl_num_int_clean = 0;
  if (is_cached_group_cast and barrier_signal_ptrs != nullptr) {
    // Get clean meta
    auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_heads, kNumDataGroups, kNumRDMARanks, num_max_rdma_chunked_recv_tokens, num_channels);
    auto nvl_clean_meta = get_nvl_clean_meta(
        hidden_int4, num_heads, kNumDataGroups, kNumRDMARanks, NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels, is_cached_group_cast);

    rdma_clean_offset = rdma_clean_meta.first;
    rdma_num_int_clean = rdma_clean_meta.second;
    nvl_clean_offset = nvl_clean_meta.first;
    nvl_num_int_clean = nvl_clean_meta.second;

    // Check if the buffer size is enough
    GRPCOLL_HOST_ASSERT((rdma_clean_offset + rdma_num_int_clean) * sizeof(int) <= num_rdma_bytes);
    GRPCOLL_HOST_ASSERT((nvl_clean_offset + nvl_num_int_clean) * sizeof(int) <= num_nvl_bytes);
  }

  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_heads);
  GRPCOLL_HOST_ASSERT(num_bytes_per_token + /*mbarrier*/ sizeof(uint64_t) <= kNumTMABytesPerWarp);
  // NOTE: in case of splitting, the issued put at the end of the buffer
  GRPCOLL_HOST_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

  // Even-numbered SMs for forwarders
  // odd-numbered SMs for RDMA senders and NVL receivers
  const int num_sms = num_channels * 2;
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);

  BOOL_SWITCH(kernel_barrier.has_value(), kHasKernelBarrier, [&] {
    auto kernel_barrier_view = [&]() {
      if constexpr (kHasKernelBarrier) {
        return kernel_barrier.value().view();
      } else {
        return magi_attn_ext::KernelBarrierView{};
      }
    }();

    BOOL_SWITCH(is_cached_group_cast, kCachedMode, [&] {
      BOOL_SWITCH(barrier_signal_ptrs != nullptr, kIsCachedNotifyFused, [&] {
        BOOL_SWITCH(num_heads != 0, kCastLSE, [&] {
          auto group_cast_func = group_cast_kernel<
              false, /*disable low_latency_mode to decrease compilation overhead*/
              kCachedMode,
              kNumDataGroups,
              kNumRDMARanks,
              kNumMaxDstRDMARanks,
              kNumTMABytesPerWarp,
              kNumSenderWarps,
              kWarpCopyUnrollStages,
              kCastLSE,
              kHasKernelBarrier,
              kIsCachedNotifyFused>;
          SET_SHARED_MEMORY_FOR_TMA(group_cast_func);
          LAUNCH_KERNEL(
              &cfg,
              group_cast_func,
              reinterpret_cast<int4*>(recv_x),
              recv_lse,
              reinterpret_cast<const int4*>(x),
              lse,
              reinterpret_cast<int4*>(recv_x_2nd),
              reinterpret_cast<const int4*>(x_2nd),
              reinterpret_cast<int4*>(recv_x_3rd),
              reinterpret_cast<const int4*>(x_3rd),
              reinterpret_cast<SourceMeta*>(recv_src_meta),
              send_rdma_head,
              send_nvl_head,
              recv_rdma_channel_prefix_matrix,
              recv_gbl_channel_prefix_matrix,
              rdma_channel_prefix_matrix,
              recv_rdma_rank_prefix_sum,
              gbl_channel_prefix_matrix,
              recv_gbl_rank_prefix_sum,
              is_token_in_rank,
              post_perm_idx,
              num_tokens,
              hidden_int4,
              num_heads,
              rdma_buffer_ptr,
              num_max_rdma_chunked_send_tokens,
              num_max_rdma_chunked_recv_tokens,
              buffer_ptrs,
              num_max_nvl_chunked_send_tokens,
              num_max_nvl_chunked_recv_tokens,
              rank,
              num_ranks,
              rdma_clean_offset,
              rdma_num_int_clean,
              nvl_clean_offset,
              nvl_num_int_clean,
              barrier_signal_ptrs,
              cpu_rdma_team,
              kernel_barrier_view);
        });
      });
    });
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename dtype_t,
    typename comm_dtype_t,
    typename reduce_dtype_t,
    int kNumDataGroups,
    int kNumRDMARanks,
    int kMaxNumHeads,
    int kNumForwarderWarps,
    int kNumTMAStages>
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
    const bool* is_reduced_token_in_rank,
    const int* reduced_rdma_head,
    const int* reduced_nvl_head,
    const void* src_meta,
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
    cudaStream_t stream,
    int num_channels,
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier,
    bool acc_reduce,
    ReduceOp reduce_op) {
  constexpr int kNumMaxSrcRDMARanks = get_num_max_src_rdma_ranks(kNumRDMARanks);
  constexpr int kNumWarpsPerForwarder = static_max(kNumForwarderWarps / kNumRDMARanks, 1);
  constexpr int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder; // approx to kNumForwarderWarps
  constexpr int kNumRDMAReceivers = kNumForwarders - NUM_MAX_NVL_PEERS;
  constexpr int num_threads = get_num_threads_group_reduce(kNumForwarders), num_warps = num_threads / WARP_SIZE;
  GRPCOLL_STATIC_ASSERT(kNumRDMARanks <= kNumForwarderWarps, "Invalid number of forwarder warps");
  GRPCOLL_STATIC_ASSERT(kNumForwarders > NUM_MAX_NVL_PEERS and kNumForwarders <= kNumForwarderWarps, "Invalid number of active forwarder warps");
  GRPCOLL_STATIC_ASSERT(num_warps == kNumForwarders + 1, "Invalid number of warps");

  constexpr int kNumTMABytesPerSenderWarp = 27 * 1024; // 27KB
  constexpr int kNumTMALoadBytes = sizeof(int4) * WARP_SIZE; // 512B, as a warp-copy unit, each lane for one int4
  constexpr int kNumTMABufferBytesPerStage = align(kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + /*mbarrier*/ sizeof(uint64_t), sizeof(int4)); // 4624B
  constexpr int kNumTMABytesPerForwarderWarp = kNumTMAStages * kNumTMABufferBytesPerStage;
  constexpr int smem_size = std::max(
      kNumTMABytesPerSenderWarp * NUM_MAX_NVL_PEERS, // 27KB * 8 = 216KB < 224KB, can hardly be raised up
      kNumTMABytesPerForwarderWarp * kNumForwarderWarps // 9248B * 24 = 216.75KB < 224KB, can hardly be raised up
  );

  // NOTE: when `kReduceOp != ReduceOp::LSE`,
  // num_heads should be 0 to let `lse_buffers` empty
  if (reduce_op != ReduceOp::LSE) {
    GRPCOLL_HOST_ASSERT(num_heads == 0);
  } else {
    GRPCOLL_HOST_ASSERT(num_heads <= kMaxNumHeads);
  }

  const int hidden_int4 = hidden_size / (sizeof(int4) / sizeof(dtype_t));
  // NOTE: we still need enough TMA load buffer for original dtype
  // before downcasting to comm_dtype_t, thus the maximum num_bytes_per_token is halved
  const auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_heads);
  GRPCOLL_HOST_ASSERT(num_bytes_per_token + /*mbarrier*/ sizeof(uint64_t) <= kNumTMABytesPerSenderWarp);
  GRPCOLL_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % kNumRDMARanks == 0);
  GRPCOLL_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / kNumRDMARanks > std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
  GRPCOLL_HOST_ASSERT(num_max_rdma_chunked_send_tokens >= kNumWarpsPerForwarder);

  // Even-numbered SMs for NVL senders and RDMA receivers
  // odd-numbered SMs for forwarders
  const int num_sms = num_channels * 2;
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream);

  BOOL_SWITCH(kernel_barrier.has_value(), kHasKernelBarrier, [&] {
    auto kernel_barrier_view = [&]() {
      if constexpr (kHasKernelBarrier) {
        return kernel_barrier.value().view();
      } else {
        return magi_attn_ext::KernelBarrierView{};
      }
    }();

    REDUCE_OP_SWITCH(reduce_op, kReduceOp, [&] {
      BOOL_SWITCH(acc_reduce, kAccReduce, [&] {
        auto group_reduce_func = group_reduce_kernel<
            dtype_t,
            comm_dtype_t,
            reduce_dtype_t,
            kReduceOp,
            kAccReduce,
            false, /*disable low_latency_mode to decrease compilation overhead*/
            kNumDataGroups,
            kMaxNumHeads,
            kNumRDMARanks,
            kNumMaxSrcRDMARanks,
            kNumTMAStages,
            kNumTMALoadBytes,
            kNumTMABufferBytesPerStage,
            kNumTMABytesPerSenderWarp,
            kNumTMABytesPerForwarderWarp,
            kNumForwarderWarps,
            kNumWarpsPerForwarder,
            kNumForwarders,
            kNumRDMAReceivers,
            kHasKernelBarrier>;
        SET_SHARED_MEMORY_FOR_TMA(group_reduce_func);
        LAUNCH_KERNEL(
            &cfg,
            group_reduce_func,
            reinterpret_cast<int4*>(reduced_x),
            reduced_lse,
            reinterpret_cast<const int4*>(x),
            lse,
            reinterpret_cast<int4*>(reduced_x_2nd),
            reinterpret_cast<const int4*>(x_2nd),
            is_reduced_token_in_rank,
            reduced_rdma_head,
            reduced_nvl_head,
            reinterpret_cast<const SourceMeta*>(src_meta),
            rdma_channel_prefix_matrix,
            rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            gbl_rank_prefix_sum,
            pre_perm_idx,
            num_reduced_tokens,
            hidden_size,
            num_heads,
            rdma_buffer_ptr,
            num_max_rdma_chunked_send_tokens,
            num_max_rdma_chunked_recv_tokens,
            buffer_ptrs,
            num_max_nvl_chunked_send_tokens,
            num_max_nvl_chunked_recv_tokens,
            rank,
            num_ranks,
            kernel_barrier_view);
      });
    });
  });
}

} // namespace magi_attn_comm::grpcoll::internode
