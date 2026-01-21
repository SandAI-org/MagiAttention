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

#include <optional>
#include <vector>

#include "reduce_op.cuh"
#include "kernel_barrier.cuh"

namespace magi_attn_comm::grpcoll {

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

} // namespace intranode

// Internode runtime
namespace internode {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

} // namespace internode

// Layout kernels
namespace layout {

void get_group_cast_meta(
    const int64_t* t2r_idx,
    int* num_tokens_per_rank,
    int* num_tokens_per_rdma_rank,
    bool* is_token_in_rank,
    int num_tokens,
    int num_ranks,
    cudaStream_t stream);

void get_a2av_perm_idx(const int64_t* output_split_sizes, const int64_t* src_idx, int64_t* perm_to_a2av_idx, int num_ranks, int num_splits, cudaStream_t stream);

} // namespace layout

// Intranode kernels
namespace intranode {

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
    int num_channels,
    bool require_recv_count);

void cached_notify_group_cast(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream);

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
    int num_recv_buffer_tokens,
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier);

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
    int num_recv_buffer_tokens,
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier
  ) {
  RANKS_WITH_WARPS_SWITCH(num_ranks, kNumRanks, kNumWarps, [&] {
    DATA_GROUPS_MAX3_SWITCH(num_groups, kNumDataGroups, [&] {
      launch_group_cast<kNumDataGroups, kNumRanks, kNumWarps>(
          recv_x,
          recv_lse,
          x,
          lse,
          recv_x_2nd,
          x_2nd,
          recv_x_3rd,
          x_3rd,
          recv_src_idx,
          recv_channel_offset,
          send_head,
          is_token_in_rank,
          channel_prefix_matrix,
          post_perm_idx,
          num_tokens,
          hidden_int4,
          num_heads,
          buffer_ptrs,
          rank,
          stream,
          num_sms,
          num_max_send_tokens,
          num_recv_buffer_tokens,
          kernel_barrier);
    });
  });
}

void cached_notify_group_reduce(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_reduced_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream);

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
    ReduceOp reduce_op);

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
    int num_reduced_tokens,
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
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier,
    bool acc_reduce,
    cudaDataType_t dtype,
    cudaDataType_t comm_dtype,
    ReduceOp reduce_op) {
  RANKS_WITH_WARPS_SWITCH(num_ranks, kNumRanks, kNumWarps, [&] {
    BOOL_SWITCH(acc_reduce, kAccReduce, [&] {
      DATA_GROUPS_MAX2_SWITCH(num_groups, kNumDataGroups, [&] {
        DTYPE_COMM_DTYPE_REDUCE_DTYPE_SWITCH(dtype, comm_dtype, T, T_COMM, T_REDUCE, [&] {
          launch_group_reduce<T, T_COMM, T_REDUCE, kNumDataGroups, kNumRanks, kNumWarps, kAccReduce>(
              reduced_x, reduced_lse, x, lse, reduced_x_2nd, x_2nd, send_head, src_idx,
              rank_prefix_matrix, channel_prefix_matrix, pre_perm_idx, num_reduced_tokens,
              hidden_size, num_heads, buffer_ptrs, rank, stream, num_sms, num_max_send_tokens,
              num_recv_buffer_tokens, reduce_op, kernel_barrier);
        });
      });
    });
  });
}

} // namespace intranode

// Internode kernels
namespace internode {

int get_source_meta_bytes();

void notify_group_cast(
    const int* num_tokens_per_rank,
    int* grpcoll_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_rdma_rank,
    int* grpcoll_recv_rdma_counter_mapped,
    const bool* is_token_in_rank,
    int num_tokens,
    int num_channels,
    int hidden_int4,
    int num_heads,
    int num_groups,
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
    bool require_recv_count);

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
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier);

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
    int num_groups,
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
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier) {
  RDMA_RANKS_SWITCH(num_ranks / NUM_MAX_NVL_PEERS, kNumRDMARanks, [&] {
      DATA_GROUPS_MAX3_SWITCH(num_groups, kNumDataGroups, [&] {
          launch_group_cast<kNumDataGroups, kNumRDMARanks>(
              recv_x,
              recv_lse,
              x,
              lse,
              recv_x_2nd,
              x_2nd,
              recv_x_3rd,
              x_3rd,
              recv_src_meta,
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
              num_channels,
              is_cached_group_cast,
              stream,
              kernel_barrier);
      });
  });
}

void cached_notify(
    int hidden_int4,
    int num_heads,
    int num_groups,
    int num_ranks,
    int num_channels,
    int num_reduced_tokens,
    int* reduced_rdma_head,
    const int* rdma_channel_prefix_matrix,
    const int* rdma_rank_prefix_sum,
    int* reduced_nvl_head,
    void* rdma_buffer_ptr,
    int num_max_rdma_chunked_recv_tokens,
    void** buffer_ptrs,
    int num_max_nvl_chunked_recv_tokens,
    int** barrier_signal_ptrs,
    int rank,
    cudaStream_t stream,
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes,
    bool is_cached_group_cast);

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
    ReduceOp reduce_op);

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
    int num_groups,
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
    cudaDataType_t dtype,
    cudaDataType_t comm_dtype,
    ReduceOp reduce_op) {
  RDMA_RANKS_WITH_FORWARDER_WARPS_SWITCH(num_ranks / NUM_MAX_NVL_PEERS, kNumRDMARanks, kNumWarps, [&] {
      DATA_GROUPS_MAX2_SWITCH(num_groups, kNumDataGroups, [&] {
          DTYPE_COMM_DTYPE_REDUCE_DTYPE_SWITCH(dtype, comm_dtype, T, T_COMM, T_REDUCE, [&] {
              auto launch_impl = [&](auto kNumTMAStages_, auto kMaxNumHeads_) {
                  constexpr static int kNumTMAStages = decltype(kNumTMAStages_)::value;
                  constexpr static int kMaxNumHeads = decltype(kMaxNumHeads_)::value;
                  launch_group_reduce<T, T_COMM, T_REDUCE, kNumDataGroups, kNumRDMARanks, kMaxNumHeads, kNumWarps, kNumTMAStages>(
                      reduced_x,
                      reduced_lse,
                      x,
                      lse,
                      reduced_x_2nd,
                      x_2nd,
                      is_reduced_token_in_rank,
                      reduced_rdma_head,
                      reduced_nvl_head,
                      src_meta,
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
                      stream,
                      num_channels,
                      kernel_barrier,
                      acc_reduce,
                      reduce_op);
              };

              if (num_heads <= 48) { /*only set max_num_heads=48 to reduce shared memory*/
                  if constexpr (kNumWarps > 24) { /*too many warps, then only num_tma_stages=1*/
                      launch_impl(std::integral_constant<int, 1>{}, std::integral_constant<int, 48>{});
                  } else { /*small num_heads and num_warps, num_tma_stages=2 is ok*/
                      launch_impl(std::integral_constant<int, 2>{}, std::integral_constant<int, 48>{});
                  }
              } else { /*try to set max_num_heads=128, then only num_tma_stages=1*/
                  if constexpr (std::is_same_v<T_REDUCE, double>) { /*double reduce dtype costs too much shared memory*/
                      if constexpr (kNumWarps > 24) { /*too many warps, then max_num_heads=86*/
                          launch_impl(std::integral_constant<int, 1>{}, std::integral_constant<int, 86>{});
                      } else { /*small num_warps, max_num_heads=120 is ok*/
                          launch_impl(std::integral_constant<int, 1>{}, std::integral_constant<int, 120>{});
                      }
                  } else { /*other reduce dtypes are ok to set max_num_heads=128*/
                      launch_impl(std::integral_constant<int, 1>{}, std::integral_constant<int, 128>{});
                  }
              }
          });
      });
  });
}

} // namespace internode

// Internode low-latency kernels
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0, int* clean_1, int num_clean_int_1, cudaStream_t stream);

void dispatch(
    void* packed_recv_x,
    void* packed_recv_x_scales,
    int* packed_recv_src_info,
    int64_t* packed_recv_layout_range,
    int* packed_recv_count,
    int* cumulative_local_expert_recv_stats,
    void* rdma_recv_x,
    int* rdma_recv_count,
    void* rdma_x,
    const void* x,
    const int64_t* topk_idx,
    int* next_clean,
    int num_next_clean_int,
    int num_tokens,
    int hidden_size,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    bool use_fp8,
    bool round_scale,
    bool use_ue8m0,
    void* workspace,
    int num_device_sms,
    cudaStream_t stream,
    int phases);

void combine(
    void* reduced_x,
    void* rdma_recv_x,
    int* rdma_recv_flag,
    void* rdma_send_x,
    const void* x,
    const int64_t* topk_idx,
    const float* topk_weights,
    const int* src_info,
    const int64_t* layout_range,
    int* next_clean,
    int num_next_clean_int,
    int num_reduced_tokens,
    int hidden_size,
    int num_max_dispatch_tokens_per_rank,
    int num_topk,
    int num_experts,
    int rank,
    int num_ranks,
    bool use_logfmt,
    void* workspace,
    int num_device_sms,
    cudaStream_t stream,
    int phases,
    bool zero_copy);

} // namespace internode_ll

} // namespace magi_attn_comm::grpcoll
