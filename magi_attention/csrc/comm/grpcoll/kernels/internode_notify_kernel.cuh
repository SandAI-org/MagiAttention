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

namespace magi_attn_comm::grpcoll::internode {

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
    const size_t rdma_clean_offset,
    const size_t rdma_num_int_clean,
    const size_t nvl_clean_offset,
    const size_t nvl_num_int_clean,
    int* rdma_channel_prefix_matrix,
    int* recv_rdma_rank_prefix_sum,
    int* gbl_channel_prefix_matrix,
    int* recv_gbl_rank_prefix_sum,
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    const nvshmem_team_t rdma_team);

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
    size_t num_rdma_bytes,
    size_t num_nvl_bytes,
    bool require_recv_count);

template <bool kLowLatencyMode, int kNumTMABytesPerWarp>
__global__ void cached_notify_kernel(
    const size_t rdma_clean_offset,
    const size_t rdma_num_int_clean,
    const size_t nvl_clean_offset,
    const size_t nvl_num_int_clean,
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
    const nvshmem_team_t rdma_team);

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
    size_t num_rdma_bytes,
    size_t num_nvl_bytes,
    bool is_cached_group_cast);

} // namespace magi_attn_comm::grpcoll::internode
