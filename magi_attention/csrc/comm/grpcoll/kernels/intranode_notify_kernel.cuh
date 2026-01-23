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

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "reduce_op.cuh"
#include "utils.cuh"

namespace magi_attn_comm::grpcoll::intranode {

template <int kNumRanks, bool kRequireRecvCount>
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
    int rank);

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

template <int kNumRanks>
__global__ void cached_notify_group_cast_kernel(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** barrier_signal_ptrs, int rank);

void cached_notify_group_cast(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream);

template <int kNumRanks>
__global__ void cached_notify_group_reduce_kernel(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_reduced_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank);

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

} // namespace magi_attn_comm::grpcoll::intranode
