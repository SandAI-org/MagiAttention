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

#pragma once

#include <vector>
#include "reduce_op.cuh"

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
    int num_sms);

void cached_notify_group_cast(
    const int* rank_prefix_matrix,
    int num_memset_int,
    void** buffer_ptrs,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream);

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
    const int64_t* post_perm_idx,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
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
    int num_recv_buffer_tokens);

void cached_notify_group_reduce(
    void** buffer_ptrs,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs,
    int rank,
    int num_ranks,
    cudaStream_t stream);

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
    const int64_t* pre_perm_idx,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
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
    bool acc_reduce,
    cudaDataType_t dtype,
    cudaDataType_t comm_dtype,
    ReduceOp reduce_op);

} // namespace intranode

// Internode kernels
namespace internode {

int get_source_meta_bytes();

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
    bool low_latency_mode);

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
    bool low_latency_mode);

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
    bool low_latency_mode);

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
    bool low_latency_mode);

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
    void* combined_x,
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
    int num_combined_tokens,
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
