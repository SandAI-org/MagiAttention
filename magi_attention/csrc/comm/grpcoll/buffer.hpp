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

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"
#include "kernels/reduce_op.cuh"

#include "config.hpp"
#include "event.hpp"
#include "meta.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME magi_attn_comm
#endif

namespace magi_attn_comm::grpcoll {

struct Buffer {
  GRPCOLL_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

 private:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;

  // NVLink Buffer
  int64_t num_nvl_bytes;
  void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;

  // NVSHMEM Buffer
  int64_t num_rdma_bytes;
  void* rdma_buffer_ptr = nullptr;

  // Device info and communication
  int device_id;
  int num_device_sms;
  int rank, rdma_rank, nvl_rank;
  int num_ranks, num_rdma_ranks, num_nvl_ranks;
  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

  // Stream for communication
  at::cuda::CUDAStream comm_stream;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Whether explicit `destroy()` is required.
  bool explicitly_destroy;
  // After `destroy()` be called, this flag will be true
  bool destroyed = false;

  // Barrier signals
  int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** barrier_signal_ptrs_gpu = nullptr;

  // Workspace
  void* workspace = nullptr;

  // Host-side rank-level counter for num_recv_tokens_this_rank
  volatile int* grpcoll_recv_counter = nullptr;
  // Device-side rank-level mapped counter for num_recv_tokens_this_rank
  int* grpcoll_recv_counter_mapped = nullptr;

  // TODO: remove this counter
  // Host-side expert-level counter for num_recv_tokens_per_local_expert_this_rank
  volatile int* moe_recv_expert_counter = nullptr;
  // Device-side expert-level mapped counter for num_recv_tokens_per_local_expert_this_rank
  int* moe_recv_expert_counter_mapped = nullptr;

  // TODO: rename this counter
  // Host-side RDMA-level counter for num_recv_tokens_this_node
  volatile int* moe_recv_rdma_counter = nullptr;
  // Device-side RDMA-level mapped counter for num_recv_tokens_this_node
  int* moe_recv_rdma_counter_mapped = nullptr;

 public:
  Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy);

  ~Buffer() noexcept(false);

  bool is_available() const;

  bool is_internode_available() const;

  int get_num_rdma_ranks() const;

  int get_rdma_rank() const;

  int get_root_rdma_rank(bool global) const;

  int get_local_device_id() const;

  pybind11::bytearray get_local_ipc_handle() const;

  pybind11::bytearray get_local_nvshmem_unique_id() const;

  torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

  torch::Stream get_comm_stream() const;

  void sync(
      const std::vector<int>& device_ids,
      const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
      const std::optional<pybind11::bytearray>& root_unique_id_opt);

  void destroy();

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<EventHandle>> get_group_cast_meta(
      const torch::Tensor& t2r_idx,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_comm_stream);

  std::tuple<
      /* 1st group of output data */
      torch::Tensor,
      std::optional<torch::Tensor>,
      /* 2nd group of output data */
      std::optional<torch::Tensor>,
      /* 3rd group of output data */
      std::optional<torch::Tensor>,
      /* handle */
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      torch::Tensor,
      /* event */
      std::optional<EventHandle>>
  intranode_group_cast(
      /* 1st group of input / output data*/
      const torch::Tensor& x,
      std::optional<torch::Tensor>& recv_x_buf,
      const std::optional<torch::Tensor>& lse,
      std::optional<torch::Tensor>& recv_lse_buf,
      /* 2nd group of input / output data*/
      const std::optional<torch::Tensor>& x_2nd,
      std::optional<torch::Tensor>& recv_x_buf_2nd,
      /* 3rd group of input / output data*/
      const std::optional<torch::Tensor>& x_3rd,
      std::optional<torch::Tensor>& recv_x_buf_3rd,
      /* other metadata */
      const std::optional<torch::Tensor>& num_tokens_per_rank,
      const torch::Tensor& is_token_in_rank,
      int cached_num_recv_tokens,
      const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
      const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
      const std::optional<torch::Tensor>& post_perm_idx,
      const Config& config,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_comm_stream);

  std::tuple<
      /* 1st group of output data */
      torch::Tensor,
      std::optional<torch::Tensor>,
      /* 2nd group of output data */
      std::optional<torch::Tensor>,
      /* event */
      std::optional<EventHandle>>
  intranode_group_reduce(
      /* 1st group of input / output data*/
      const torch::Tensor& x,
      std::optional<torch::Tensor>& reduced_x_buf,
      const std::optional<torch::Tensor>& lse,
      std::optional<torch::Tensor>& reduced_lse_buf,
      /* 2nd group of input / output data*/
      const std::optional<torch::Tensor>& x_2nd,
      std::optional<torch::Tensor>& reduced_x_buf_2nd,
      /* other metadata */
      const std::optional<torch::Tensor>& pre_perm_idx,
      const torch::Tensor& src_idx,
      const torch::Tensor& rank_prefix_matrix,
      const torch::Tensor& channel_prefix_matrix,
      const torch::Tensor& send_head,
      const Config& config,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_comm_stream,
      const std::string& reduce_op,
      bool acc_reduce,
      std::optional<c10::ScalarType> comm_dtype);

  std::tuple<
      torch::Tensor,
      /* handle */
      torch::Tensor,
      torch::Tensor,
      std::optional<torch::Tensor>,
      torch::Tensor,
      std::optional<torch::Tensor>,
      torch::Tensor,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      std::optional<torch::Tensor>,
      /* event */
      std::optional<EventHandle>>
  internode_group_cast(
      const torch::Tensor& x,
      std::optional<torch::Tensor>& recv_x_buf,
      const std::optional<torch::Tensor>& num_tokens_per_rank,
      const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
      const torch::Tensor& is_token_in_rank,
      const std::optional<torch::Tensor>& num_tokens_per_expert,
      int cached_num_recv_tokens,
      int cached_num_rdma_recv_tokens,
      const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
      const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
      const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
      const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
      const std::optional<torch::Tensor>& post_perm_idx,
      const Config& config,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_comm_stream);

  std::tuple<torch::Tensor, std::optional<EventHandle>> internode_group_reduce(
      const torch::Tensor& x,
      std::optional<torch::Tensor>& combined_x_buf,
      const torch::Tensor& src_meta,
      const torch::Tensor& is_combined_token_in_rank,
      const torch::Tensor& rdma_channel_prefix_matrix,
      const torch::Tensor& rdma_rank_prefix_sum,
      const torch::Tensor& gbl_channel_prefix_matrix,
      const torch::Tensor& combined_rdma_head,
      const torch::Tensor& combined_nvl_head,
      const std::optional<torch::Tensor>& pre_perm_idx,
      const Config& config,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_comm_stream,
      const std::string& reduce_op,
      bool acc_reduce);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_dispatch(
      const torch::Tensor& x,
      const torch::Tensor& topk_idx,
      const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool use_fp8,
      bool round_scale,
      bool use_ue8m0,
      bool async_op,
      bool return_recv_hook);

  std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
      const torch::Tensor& x,
      const torch::Tensor& topk_idx,
      const torch::Tensor& topk_weights,
      const torch::Tensor& src_info,
      const torch::Tensor& layout_range,
      int num_max_dispatch_tokens_per_rank,
      int num_experts,
      bool use_logfmt,
      bool zero_copy,
      bool async_op,
      bool return_recv_hook,
      const std::optional<torch::Tensor>& out = std::nullopt);

  torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;
};

} // namespace magi_attn_comm::grpcoll
