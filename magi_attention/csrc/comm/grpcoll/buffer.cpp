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

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>
#include <chrono>
#include <memory>

#include "buffer.hpp"

namespace py = pybind11;

namespace grpcoll = magi_attn_comm::grpcoll;

namespace magi_attn_comm::grpcoll {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Buffer Initialization
///////////////////////////////////////////////////////////////////////////////////////////////////

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)) {
  // Calculate metadata memory
  // NOTES: The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
  int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int); // host signal array for each nvl rank
  int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*); // host buffer ptr array to each buffer ptr for each nvl rank
  int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*); // host signal ptr array to each signal for each nvl rank

  // Common checks
  GRPCOLL_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= INT_MAX or num_rdma_bytes == 0));
  GRPCOLL_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= INT_MAX));
  GRPCOLL_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_PEERS or low_latency_mode));
  GRPCOLL_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
  if (num_rdma_bytes > 0)
    GRPCOLL_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

  // Get ranks
  CUDA_CHECK(cudaGetDevice(&device_id));
  rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
  GRPCOLL_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

  // Get device info
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
  num_device_sms = device_prop.multiProcessorCount;

  // Initialize intranode NVLink buffer (alloc local device buffer memory and set local IPC handles)
  if (num_nvl_bytes > 0) {
    // Allocate local NVLink buffer memory
    CUDA_CHECK(cudaMalloc(
        &buffer_ptrs[nvl_rank], // local nvl buffer_ptr
        num_nvl_bytes // the nvl buffer size set by the user
            + barrier_signal_bytes // the signal size
            + buffer_ptr_bytes // the buffer ptr array size
            + barrier_signal_ptr_bytes // the signal ptr array size
        ));

    // Initialize local NVLink IPC handle
    CUDA_CHECK(cudaIpcGetMemHandle(
        &ipc_handles[nvl_rank], // local nvl ipc handle
        buffer_ptrs[nvl_rank] // local nvl buffer_ptr
        ));

    // Set several local device ptrs into host ptr array
    // and set the device ptrs to global ptr arrays
    auto local_nvl_buffer_byte_ptr = static_cast<uint8_t*>(buffer_ptrs[nvl_rank]);

    // Set the host ptr to the local nvl signal
    int64_t local_nvl_buffer_byte_offs = num_nvl_bytes;
    barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(local_nvl_buffer_byte_ptr + local_nvl_buffer_byte_offs);

    // Set the device ptr to the buffer ptr array
    local_nvl_buffer_byte_offs += barrier_signal_bytes;
    buffer_ptrs_gpu = reinterpret_cast<void**>(local_nvl_buffer_byte_ptr + local_nvl_buffer_byte_offs);

    // Set the device ptr to the signal ptr array
    local_nvl_buffer_byte_offs += buffer_ptr_bytes;
    barrier_signal_ptrs_gpu = reinterpret_cast<int**>(local_nvl_buffer_byte_ptr + local_nvl_buffer_byte_offs);

    // Initialize local nvl signal to zero
    // NOTES: no need to synchronize here, since we will apply a full device sync in `sync`
    CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
  }

  // Initialize 32 MB device workspace
  CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

  // Initialize `num_recv_tokens_this_rank` counter (pinned host memory with its device ptr)
  CUDA_CHECK(cudaMallocHost(&grpcoll_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&grpcoll_recv_counter_mapped, const_cast<int*>(grpcoll_recv_counter), 0));
  *grpcoll_recv_counter = -1;

  // Initialize `num_recv_tokens_this_node` counter (pinned host memory with its device ptr)
  if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(&grpcoll_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&grpcoll_recv_rdma_counter_mapped, const_cast<int*>(grpcoll_recv_rdma_counter), 0));
    *grpcoll_recv_rdma_counter = -1;
  }
}

void Buffer::sync(
    const std::vector<int>& device_ids,
    const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
    const std::optional<py::bytearray>& root_unique_id_opt) {
  // Sync NVLink IPC handles, buffer ptrs, and signal ptrs
  if (num_nvl_bytes > 0) {
    GRPCOLL_HOST_ASSERT(num_ranks == device_ids.size());
    GRPCOLL_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());

    for (int i = 0, node_offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
      auto ith_nvl_rank = node_offset + i;

      // Get IPC handle for the ith nvl rank in this node
      GRPCOLL_HOST_ASSERT(all_gathered_handles[ith_nvl_rank].has_value());
      auto handle_str = std::string(all_gathered_handles[ith_nvl_rank].value());
      GRPCOLL_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);

      auto ith_ipc_handle = ipc_handles[i];
      if (ith_nvl_rank != rank) { // another peer
        // Copy IPC handle of the ith nvl rank to the handle array
        std::memcpy(ith_ipc_handle.reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);

        // Open the IPC handle of the ith nvl rank
        // and store its buffer ptr to the buffer ptr array
        CUDA_CHECK(cudaIpcOpenMemHandle(
            &buffer_ptrs[i],
            ith_ipc_handle,
            cudaIpcMemLazyEnablePeerAccess // Enable P2P Access
            ));

        // Store the signal ptr of the ith nvl rank to the signal ptr array
        barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
      } else { // current rank
        // Just check if the handle matches
        GRPCOLL_HOST_ASSERT(std::memcmp(ith_ipc_handle.reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
      }
    }

    // Here, all the device buffer ptr and device signal ptrs are ready from the all gathed NVLink IPC handles
    // but the ptr arrays themselves are still on the host, thus copy them to GPU
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));

    // Synchronize all GPU jobs to be finished with the CPU
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Sync NVSHMEM handles and allocate memory
#ifndef DISABLE_NVSHMEM
  if (num_rdma_bytes > 0) {
    // Get the root unique ID
    GRPCOLL_HOST_ASSERT(root_unique_id_opt.has_value());
    std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
    auto root_unique_id_str = root_unique_id_opt->cast<std::string>();

    // Copy the root unique ID
    std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());

    // Initialize NVSHMEM
    auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
    auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
    GRPCOLL_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));

    // Barrier all ranks in the same NVSHMEM team until initialized
    internode::barrier();

    // Allocate the NVSHMEM symmetric buffer for rdma transfer
    rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

    // Clean buffer (mainly for low-latency mode)
    CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

    // Barrier all ranks in the same NVSHMEM team until allocated
    internode::barrier();

    // Synchronize all GPU jobs to be finished with the CPU
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif

  // Make buffer available
  available = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Buffer Finalization
///////////////////////////////////////////////////////////////////////////////////////////////////

Buffer::~Buffer() noexcept(false) {
  if (not explicitly_destroy) {
    destroy();
  } else if (not destroyed) {
    printf("WARNING: destroy() was not called before grpcoll buffer destruction, which may leak resources.\n");
    fflush(stdout);
  }
}

void Buffer::destroy() {
  GRPCOLL_HOST_ASSERT(not destroyed);

  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());

  if (num_nvl_bytes > 0) {
    // Barrier
    intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Close remote IPC
    if (is_available()) {
      for (int i = 0; i < num_nvl_ranks; ++i)
        if (i != nvl_rank)
          CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
    }

    // Free local buffer and error flag
    CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
  }

  // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
  if (is_available() and num_rdma_bytes > 0) {
    CUDA_CHECK(cudaDeviceSynchronize());
    internode::barrier();
    internode::free(rdma_buffer_ptr);
    internode::finalize();
  }
#endif

  // Free workspace and counters
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(grpcoll_recv_counter)));

  destroyed = true;
  available = false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Core Kernel APIs
///////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<EventHandle>> Buffer::get_group_cast_meta(
    const torch::Tensor& t2r_idx,
    std::optional<EventHandle>& previous_event,
    bool async_op,
    bool allocate_on_comm_stream) {
  return Meta::get_group_cast_meta_from_t2r_idx(
      /*t2r_idx=*/t2r_idx,
      /*num_ranks=*/num_ranks,
      /*num_rdma_ranks=*/is_internode_available() ? num_rdma_ranks : 1,
      /*previous_event=*/previous_event,
      /*async_op=*/async_op,
      /*allocate_on_comm_stream=*/allocate_on_comm_stream,
      /*meta_stream=*/comm_stream);
}

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
Buffer::intranode_group_cast(
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
    bool allocate_on_comm_stream) {
  // REVIEW: should we release GIL here like internode ?
  // py::gil_scoped_release release;

  // Determine if we are using chunked mode
  bool cached_mode = cached_rank_prefix_matrix.has_value();

  // Get the number of data groups
  int num_groups = 1;
  if (x_2nd.has_value())
    ++num_groups;
  if (x_3rd.has_value()) {
    GRPCOLL_HOST_ASSERT(num_groups == 2);
    ++num_groups;
  }
  GRPCOLL_HOST_ASSERT(num_groups <= 3);

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  GRPCOLL_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
    GRPCOLL_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank.has_value());
  }

  // Type checks
  GRPCOLL_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
    GRPCOLL_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
  }

  // Shape and contiguous checks
  GRPCOLL_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  GRPCOLL_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0); // hidden comm bytes should be aligned with int4
  GRPCOLL_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
  GRPCOLL_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
    GRPCOLL_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
  }

  auto num_tokens = static_cast<int>(x.size(0)), hidden_size = static_cast<int>(x.size(1));
  // NOTES: actually, hidden size in int4 should be aligned with the number of TMA stages in the kernel
  // which we will verify later in the kernel launch function
  auto hidden_int4 = static_cast<int>(hidden_size * x.element_size() / sizeof(int4));
  if (num_groups > 1) {
    GRPCOLL_HOST_ASSERT(x_2nd->dim() == 2 and x_2nd->is_contiguous() and x_2nd->scalar_type() == x.scalar_type());
    GRPCOLL_HOST_ASSERT(x_2nd->size(0) == num_tokens and x_2nd->size(1) == hidden_size);
  }
  if (num_groups > 2) {
    GRPCOLL_HOST_ASSERT(x_3rd->dim() == 2 and x_3rd->is_contiguous() and x_3rd->scalar_type() == x.scalar_type());
    GRPCOLL_HOST_ASSERT(x_3rd->size(0) == num_tokens and x_3rd->size(1) == hidden_size);
  }

  // LSE checks
  float* lse_ptr = nullptr;
  int num_heads = 0; // NOTES: when lse is not provided, num_heads is set to 0 and consumes empty buffer
  if (lse.has_value()) {
    GRPCOLL_HOST_ASSERT(lse->dim() == 2 and lse->is_contiguous());
    GRPCOLL_HOST_ASSERT(lse->scalar_type() == torch::kFloat32);
    GRPCOLL_HOST_ASSERT(lse->size(0) == num_tokens);
    GRPCOLL_HOST_ASSERT(hidden_size % lse->size(1) == 0); // hidden size should be divisible by num_heads
    num_heads = static_cast<int>(lse->size(1));
    lse_ptr = lse->data_ptr<float>();
  }

  // Set current stream to comm stream if needed
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1;
  auto rank_prefix_matrix = torch::Tensor();
  auto channel_prefix_matrix = torch::Tensor();

  // Notify
  int num_memset_int = num_channels * num_ranks * 4; // clean channel start/end offset, head and tail
  if (cached_mode) {
    num_recv_tokens = cached_num_recv_tokens;
    rank_prefix_matrix = cached_rank_prefix_matrix.value();
    channel_prefix_matrix = cached_channel_prefix_matrix.value();

    // Copy rank prefix matrix and clean flags
    intranode::cached_notify_group_cast(
        /*rank_prefix_matrix=*/rank_prefix_matrix.data_ptr<int>(),
        /*num_memset_int=*/num_memset_int,
        /*buffer_ptrs=*/buffer_ptrs_gpu,
        /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
        /*rank=*/rank,
        /*num_ranks=*/num_ranks,
        /*stream=*/comm_stream);
  } else {
    rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

    // Reset the pinned counter to `-1`
    *grpcoll_recv_counter = -1;

    // Notify to clean the buffer, switch meta data and calculate meta tensors
    // TODO: make `notify_group_cast` an individual buffer API
    // to allow notifying in advance to enable cache mode amap
    intranode::notify_group_cast(
        /*num_tokens_per_rank=*/num_tokens_per_rank->data_ptr<int>(),
        /*grpcoll_recv_counter_mapped=*/grpcoll_recv_counter_mapped,
        /*num_ranks=*/num_ranks,
        /*num_tokens=*/num_tokens,
        /*is_token_in_rank=*/is_token_in_rank.data_ptr<bool>(),
        /*channel_prefix_matrix=*/channel_prefix_matrix.data_ptr<int>(),
        /*rank_prefix_matrix=*/rank_prefix_matrix.data_ptr<int>(),
        /*num_memset_int=*/num_memset_int,
        /*buffer_ptrs=*/buffer_ptrs_gpu,
        /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
        /*rank=*/rank,
        /*comm_stream=*/comm_stream,
        /*num_channels=*/num_channels);

    if (recv_x_buf.has_value()) {
      // if the recv buffer is given,
      // use its dim0 size as num_recv_tokens to avoid CPU sync
      num_recv_tokens = recv_x_buf->size(0);
    } else {
      // otherwise, synchronize num_recv_tokens
      auto start_time = std::chrono::high_resolution_clock::now();
      while (true) {
        // Read if num_recv_tokens is ready
        num_recv_tokens = static_cast<int>(*grpcoll_recv_counter);
        if (num_recv_tokens >= 0)
          break;

        // Timeout check
        if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
          throw std::runtime_error("grpcoll error: CPU recv timeout for intranode group cast");
      }
    }
  }

  // Allocate recv_x buffer
  auto recv_x = torch::Tensor();
  if (recv_x_buf.has_value()) {
    GRPCOLL_HOST_ASSERT(recv_x_buf->dim() == 2 and recv_x_buf->is_contiguous() and recv_x_buf->scalar_type() == x.scalar_type());
    GRPCOLL_HOST_ASSERT(recv_x_buf->size(0) == num_recv_tokens and recv_x_buf->size(1) == hidden_size);
    recv_x = recv_x_buf.value();
  } else {
    recv_x = torch::empty({num_recv_tokens, hidden_size}, x.options());
  }

  // Allocate 2nd recv_x buffer and assign the ptr if needed
  auto recv_x_2nd = std::optional<torch::Tensor>();
  void *x_ptr_2nd = nullptr, *recv_x_ptr_2nd = nullptr;
  if (num_groups > 1) {
    if (recv_x_buf_2nd.has_value()) {
      GRPCOLL_HOST_ASSERT(recv_x_buf_2nd->dim() == 2 and recv_x_buf_2nd->is_contiguous() and recv_x_buf_2nd->scalar_type() == x.scalar_type());
      GRPCOLL_HOST_ASSERT(recv_x_buf_2nd->size(0) == num_recv_tokens and recv_x_buf_2nd->size(1) == hidden_size);
      recv_x_2nd.emplace(recv_x_buf_2nd.value());
    } else {
      recv_x_2nd = torch::empty({num_recv_tokens, hidden_size}, x_2nd->options());
    }
    x_ptr_2nd = x_2nd->data_ptr();
    recv_x_ptr_2nd = recv_x_2nd->data_ptr();
  }

  // Allocate 3rd recv_x buffer and assign the ptr if needed
  auto recv_x_3rd = std::optional<torch::Tensor>();
  void *x_ptr_3rd = nullptr, *recv_x_ptr_3rd = nullptr;
  if (num_groups > 2) {
    if (recv_x_buf_3rd.has_value()) {
      GRPCOLL_HOST_ASSERT(recv_x_buf_3rd->dim() == 2 and recv_x_buf_3rd->is_contiguous() and recv_x_buf_3rd->scalar_type() == x.scalar_type());
      GRPCOLL_HOST_ASSERT(recv_x_buf_3rd->size(0) == num_recv_tokens and recv_x_buf_3rd->size(1) == hidden_size);
      recv_x_3rd.emplace(recv_x_buf_3rd.value());
    } else {
      recv_x_3rd = torch::empty({num_recv_tokens, hidden_size}, x_3rd->options());
    }
    x_ptr_3rd = x_3rd->data_ptr();
    recv_x_ptr_3rd = recv_x_3rd->data_ptr();
  }

  // Allocate metadata tensors
  auto recv_lse = std::optional<torch::Tensor>();
  auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
  auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
  auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

  // Assign ptr for post_perm_idx if needed
  int64_t* post_perm_idx_ptr = nullptr;
  if (post_perm_idx.has_value()) {
    GRPCOLL_HOST_ASSERT(post_perm_idx->scalar_type() == torch::kInt64);
    GRPCOLL_HOST_ASSERT(post_perm_idx->dim() == 1);
    GRPCOLL_HOST_ASSERT(post_perm_idx->size(0) == num_recv_tokens);
    post_perm_idx_ptr = post_perm_idx->data_ptr<int64_t>();
  }

  // Allocate recv_lse buffer and assign the ptr if needed
  float* recv_lse_ptr = nullptr;
  if (lse.has_value()) {
    if (recv_lse_buf.has_value()) {
      GRPCOLL_HOST_ASSERT(recv_lse_buf->dim() == 2 && recv_lse_buf->is_contiguous());
      GRPCOLL_HOST_ASSERT(recv_lse_buf->scalar_type() == torch::kFloat32);
      GRPCOLL_HOST_ASSERT(recv_lse_buf->size(0) == num_recv_tokens && recv_lse_buf->size(1) == num_heads);
      recv_lse.emplace(recv_lse_buf.value());
    } else {
      recv_lse = torch::empty({num_recv_tokens, num_heads}, lse->options());
    }
    recv_lse_ptr = recv_lse->data_ptr<float>();
  }

  // Check if the buffer size is enough
  GRPCOLL_HOST_ASSERT(
      num_ranks * num_ranks * sizeof(int) + // rank prefix matrix
          num_channels * num_ranks * sizeof(int) * 4 + // channel start/end offset + queue head/tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden_size * recv_x.element_size() * num_groups + // data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) + // src idx buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_heads * sizeof(float) + // lse buffer
          sizeof(int4) * num_groups // max padding bytes to align for vectorized data buffer
      <= num_nvl_bytes); // TODO: turn this assertion into the minimum bytes hint API for the user to determine the buffer size

  // Launch group_cast kernel
  /** FIXME: we find out the group_cast kernel cannot be picked until the ffa kernel is finished,
   * if the ffa kernel is picked first and the sm_margin is not large enough
   * e.g. if the group_cast kernel requires 24 SMs, then the pre-picked ffa kernel will have to give up at least 33 SMs,
   * otherwise the group_cast kernel will wait until the ffa kernel is finished,
   * the same phenomenon happens with the group_reduce kernel as well
   *
   * later, we've already figured out this phenomenon is due to
   * both cooperative launch pattern (which at least requires 24 SMs to be launched at the same time),
   * and also cluster launch pattern (which requires the SMs in one cluster to belong to the same TPC or GPC),
   * but we don't know how to exactly fix this issue by now
   */
  intranode::group_cast(
      /*recv_x=*/recv_x.data_ptr(),
      /*recv_lse=*/recv_lse_ptr,
      /*x=*/x.data_ptr(),
      /*lse=*/lse_ptr,
      /*recv_x_2nd=*/recv_x_ptr_2nd,
      /*x_2nd=*/x_ptr_2nd,
      /*recv_x_3rd=*/recv_x_ptr_3rd,
      /*x_3rd=*/x_ptr_3rd,
      /*recv_src_idx=*/recv_src_idx.data_ptr<int>(),
      /*recv_channel_offset=*/recv_channel_prefix_matrix.data_ptr<int>(),
      /*send_head=*/send_head.data_ptr<int>(),
      /*is_token_in_rank=*/is_token_in_rank.data_ptr<bool>(),
      /*channel_prefix_matrix=*/channel_prefix_matrix.data_ptr<int>(),
      /*post_perm_idx=*/post_perm_idx_ptr,
      /*num_tokens=*/num_tokens,
      /*hidden_int4=*/hidden_int4,
      /*num_heads=*/num_heads,
      /*num_groups=*/num_groups,
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*rank=*/rank,
      /*num_ranks=*/num_ranks,
      /*comm_stream=*/comm_stream,
      /*num_sms=*/config.num_sms,
      /*num_max_send_tokens=*/config.num_max_nvl_chunked_send_tokens,
      /*num_recv_buffer_tokens=*/config.num_max_nvl_chunked_recv_tokens);

  // Record or wait streams
  std::optional<EventHandle> event;
  if (async_op) { // Record tensors on non-allocated stream
    event = EventHandle(comm_stream);
    // record tensors
    for (auto& t : {x, recv_x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_src_idx, recv_channel_prefix_matrix, send_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream)
        t.record_stream(compute_stream);
    }
    // record optional tensors
    for (auto& to :
         {x_2nd, recv_x_2nd, x_3rd, recv_x_3rd, lse, recv_lse, num_tokens_per_rank, cached_channel_prefix_matrix, cached_rank_prefix_matrix, post_perm_idx}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else { // Wait tensors on compute stream
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back current stream to compute stream if needed
  if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(compute_stream);

  // Return values
  return {recv_x, recv_lse, recv_x_2nd, recv_x_3rd, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event};
}

std::tuple<
    /* 1st group of output data */
    torch::Tensor,
    std::optional<torch::Tensor>,
    /* 2nd group of output data */
    std::optional<torch::Tensor>,
    /* event */
    std::optional<EventHandle>>
Buffer::intranode_group_reduce(
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
    std::optional<c10::ScalarType> comm_dtype) {
  // Transfer reduce ops
  ReduceOp reduce_op_ = str_to_reduce_op(reduce_op);

  // Transfer dtypes and item sizes in bytes
  auto x_dtype = x.scalar_type();
  auto comm_dtype_ = comm_dtype.value_or(x_dtype);
  auto comm_elem_size = c10::elementSize(comm_dtype_);

  // Get the number of data groups
  int num_groups = 1;
  if (x_2nd.has_value())
    ++num_groups;
  GRPCOLL_HOST_ASSERT(num_groups <= 2);

  // Check tensors
  GRPCOLL_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  GRPCOLL_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.scalar_type() == torch::kInt32);

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  GRPCOLL_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;

  auto num_tokens = static_cast<int>(x.size(0)), hidden_size = static_cast<int>(x.size(1));
  auto num_reduced_tokens = static_cast<int>(send_head.size(0));
  GRPCOLL_HOST_ASSERT(src_idx.size(0) == num_tokens);
  GRPCOLL_HOST_ASSERT(send_head.size(1) == num_ranks);
  GRPCOLL_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
  GRPCOLL_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
  GRPCOLL_HOST_ASSERT((hidden_size * comm_elem_size) % sizeof(int4) == 0); // hidden comm bytes should be aligned with int4
  GRPCOLL_HOST_ASSERT(((hidden_size * comm_elem_size) / sizeof(int4)) % WARP_SIZE == 0); // hidden size in int4 should be aligned with warp size
  if (num_groups > 1) {
    GRPCOLL_HOST_ASSERT(x_2nd->dim() == 2 and x_2nd->is_contiguous() and x_2nd->scalar_type() == x_dtype);
    GRPCOLL_HOST_ASSERT(x_2nd->size(0) == num_tokens and x_2nd->size(1) == hidden_size);
  }

  // Set current stream to comm stream if needed
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Assign ptr for pre_perm_idx if needed
  int64_t* pre_perm_idx_ptr = nullptr;
  if (pre_perm_idx.has_value()) {
    GRPCOLL_HOST_ASSERT(pre_perm_idx->dim() == 1 && pre_perm_idx->is_contiguous());
    GRPCOLL_HOST_ASSERT(pre_perm_idx->scalar_type() == torch::kInt64);
    GRPCOLL_HOST_ASSERT(pre_perm_idx->size(0) == num_tokens);
    pre_perm_idx_ptr = pre_perm_idx->data_ptr<int64_t>();
  }

  // Allocate reduced_lse buffer and assign the ptr if needed
  int num_heads = 0; // NOTES: when `reduce_op != ReduceOp::LSE`, num_heads is set to 0 and consumes empty buffer
  auto reduced_lse = std::optional<torch::Tensor>();
  float *lse_ptr = nullptr, *reduced_lse_ptr = nullptr;
  if (lse.has_value()) {
    GRPCOLL_HOST_ASSERT(reduce_op_ == ReduceOp::LSE); // no point to transfer lse if reduce_op != ReduceOp::LSE
    GRPCOLL_HOST_ASSERT(lse->dim() == 2 and lse->is_contiguous());
    GRPCOLL_HOST_ASSERT(lse->scalar_type() == torch::kFloat32);
    GRPCOLL_HOST_ASSERT(lse->size(0) == num_tokens && hidden_size % lse->size(1) == 0); // hidden size should be divisible by num_heads
    num_heads = static_cast<int>(lse->size(1));
    auto head_dim = hidden_size / num_heads;
    GRPCOLL_HOST_ASSERT(head_dim % (sizeof(int4) / comm_elem_size) == 0); // each group of elems with dtype `comm_dtype` in one int4 should share the same head
    lse_ptr = lse->data_ptr<float>();

    if (reduced_lse_buf.has_value()) {
      GRPCOLL_HOST_ASSERT(reduced_lse_buf->dim() == 2 and reduced_lse_buf->is_contiguous());
      GRPCOLL_HOST_ASSERT(reduced_lse_buf->scalar_type() == lse->scalar_type());
      GRPCOLL_HOST_ASSERT(reduced_lse_buf->size(0) == num_reduced_tokens && reduced_lse_buf->size(1) == num_heads);
      reduced_lse.emplace(reduced_lse_buf.value());
    } else {
      GRPCOLL_HOST_ASSERT(!acc_reduce); // no point to acc_reduce if reduced_lse_buf is not provided
      /** NOTE: different from ep, for group-reduce with reduce_op == ReduceOp::LSE,
       * some token in reduced_lse might not reduce anything,
       * since the corr. token has no destination rank in the corr. group-cast
       * so we have to "-inf"-initialize reduced_lse, instead of empty initialization
       * however, we handle the "-inf" initialization inside the group_reduce kernel
       * thus here, we still use empty initialization
       */
      reduced_lse = torch::empty({num_reduced_tokens, num_heads}, lse->options());
    }

    reduced_lse_ptr = reduced_lse->data_ptr<float>();
  } else {
    GRPCOLL_HOST_ASSERT(reduce_op_ != ReduceOp::LSE); // lse must be provided when reduce_op == ReduceOp::LSE
  }

  // Launch barrier and reset queue head and tail
  // TODO: support notify_group_reduce when the group_reduce kernel is individually used
  // without relying on the symmetric group_cast called first and necessary handle given
  intranode::cached_notify_group_reduce(
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*send_head=*/send_head.data_ptr<int>(),
      /*num_channels=*/num_channels,
      /*num_recv_tokens=*/num_reduced_tokens,
      /*num_memset_int=*/num_channels * num_ranks * 2,
      /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
      /*rank=*/rank,
      /*num_ranks=*/num_ranks,
      /*comm_stream=*/comm_stream);

  // Allocate reduced_x buffer
  auto reduced_x = torch::Tensor();
  if (reduced_x_buf.has_value()) {
    GRPCOLL_HOST_ASSERT(reduced_x_buf->dim() == 2 and reduced_x_buf->is_contiguous() and reduced_x_buf->scalar_type() == x_dtype);
    GRPCOLL_HOST_ASSERT(reduced_x_buf->size(0) == num_reduced_tokens and reduced_x_buf->size(1) == hidden_size);
    reduced_x = reduced_x_buf.value();
  } else {
    GRPCOLL_HOST_ASSERT(!acc_reduce); // no point to acc_reduce if reduced_x_buf is not provided
    /** NOTE: different from ep, for group-reduce,
     * some token in reduced_x might not reduce anything,
     * since the corr. token has no destination rank in the corr. group-cast
     * so we have to zero-initialize reduced_x, instead of empty initialization
     * however, we handle the zero initialization inside the group_reduce kernel
     * thus here, we still use empty initialization
     */
    reduced_x = torch::empty({num_reduced_tokens, hidden_size}, x.options());
  }

  // Allocate 2nd reduced_x buffer and assign the ptr if needed
  auto reduced_x_2nd = std::optional<torch::Tensor>();
  void *x_ptr_2nd = nullptr, *reduced_x_ptr_2nd = nullptr;
  if (num_groups > 1) {
    if (reduced_x_buf_2nd.has_value()) {
      GRPCOLL_HOST_ASSERT(reduced_x_buf_2nd->dim() == 2 and reduced_x_buf_2nd->is_contiguous() and reduced_x_buf_2nd->scalar_type() == x_dtype);
      GRPCOLL_HOST_ASSERT(reduced_x_buf_2nd->size(0) == num_reduced_tokens and reduced_x_buf_2nd->size(1) == hidden_size);
      reduced_x_2nd.emplace(reduced_x_buf_2nd.value());
    } else {
      GRPCOLL_HOST_ASSERT(!acc_reduce);
      reduced_x_2nd = torch::empty({num_reduced_tokens, hidden_size}, x_2nd->options());
    }

    x_ptr_2nd = x_2nd->data_ptr();
    reduced_x_ptr_2nd = reduced_x_2nd->data_ptr();
  }

  // Check if the buffer size is enough
  GRPCOLL_HOST_ASSERT(
      num_channels * num_ranks * sizeof(int) * 2 + // queue head and tail
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden_size * comm_elem_size * num_groups + // data buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) + // src idx buffer
          num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_heads * sizeof(float) + // lse buffer
          sizeof(int4) * num_groups // max padding bytes to align for vectorized data buffer
      <= num_nvl_bytes);

  // Launch group_reduce kernel
  /** FIXME: we find out the group_reduce kernel cannot be picked until the ffa kernel is finished,
   * if the ffa kernel is picked first and the sm_margin is not large enough
   * e.g. if the group_reduce kernel requires 24 SMs, then the pre-picked ffa kernel will have to give up at least 33 SMs,
   * otherwise the group_reduce kernel will wait until the ffa kernel is finished,
   * the same phenomenon happens with the group_cast kernel as well
   *
   * later, we've already figured out this phenomenon is due to
   * both cooperative launch pattern (which at least requires 24 SMs to be launched at the same time),
   * and also cluster launch pattern (which requires the SMs in one cluster to belong to the same TPC or GPC),
   * but we don't know how to exactly fix this issue by now
   */
  intranode::group_reduce(
      /*reduced_x=*/reduced_x.data_ptr(),
      /*reduced_lse=*/reduced_lse_ptr,
      /*x=*/x.data_ptr(),
      /*lse=*/lse_ptr,
      /*reduced_x_2nd=*/reduced_x_ptr_2nd,
      /*x_2nd=*/x_ptr_2nd,
      /*send_head=*/send_head.data_ptr<int>(),
      /*src_idx=*/src_idx.data_ptr<int>(),
      /*rank_prefix_matrix=*/rank_prefix_matrix.data_ptr<int>(),
      /*channel_prefix_matrix=*/channel_prefix_matrix.data_ptr<int>(),
      /*pre_perm_idx=*/pre_perm_idx_ptr,
      /*num_tokens=*/num_tokens,
      /*num_recv_tokens=*/num_reduced_tokens,
      /*hidden_size=*/hidden_size,
      /*num_heads=*/num_heads,
      /*num_groups=*/num_groups,
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*rank=*/rank,
      /*num_ranks=*/num_ranks,
      /*comm_stream=*/comm_stream,
      /*num_sms=*/config.num_sms,
      /*num_max_send_tokens=*/config.num_max_nvl_chunked_send_tokens,
      /*num_recv_buffer_tokens=*/config.num_max_nvl_chunked_recv_tokens,
      /*acc_reduce=*/acc_reduce,
      /*dtype=*/at::cuda::ScalarTypeToCudaDataType(x_dtype),
      /*comm_dtype=*/at::cuda::ScalarTypeToCudaDataType(comm_dtype_),
      /*reduce_op=*/reduce_op_);

  // Record or wait streams
  std::optional<EventHandle> event;
  if (async_op) { // Record tensors on non-allocated stream
    event = EventHandle(comm_stream);
    // record tensors
    for (auto& t : {x, reduced_x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream)
        t.record_stream(compute_stream);
    }
    // record optional tensors
    for (auto& to : {x_2nd, reduced_x_2nd, lse, reduced_lse, pre_perm_idx}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else { // Wait tensors on compute stream
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back current stream to compute stream if needed
  if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(compute_stream);

  return {reduced_x, reduced_lse, reduced_x_2nd, event};
}

std::tuple<
    /* 1st group of output data */
    torch::Tensor,
    std::optional<torch::Tensor>,
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
Buffer::internode_group_cast(
    /* 1st group of input / output data*/
    const torch::Tensor& x,
    std::optional<torch::Tensor>& recv_x_buf,
    const std::optional<torch::Tensor>& lse,
    std::optional<torch::Tensor>& recv_lse_buf,
    /* other metadata */
    const std::optional<torch::Tensor>& num_tokens_per_rank,
    const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
    const torch::Tensor& is_token_in_rank,
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
    bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
  // In group_cast stage, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
  // If users of grpcoll need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
  // unless we release GIL here.
  py::gil_scoped_release release;

  // One channel use two SMs
  // one for forwarders, the other for (senders, receivers)
  const int num_channels = config.num_sms / 2;
  GRPCOLL_HOST_ASSERT(config.num_sms % 2 == 0);
  GRPCOLL_HOST_ASSERT(get_num_rdma_ranks() > 0 and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

  bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
    GRPCOLL_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
    GRPCOLL_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
    GRPCOLL_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank.has_value());
    GRPCOLL_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
  }

  // Type checks
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
    GRPCOLL_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
    GRPCOLL_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
    GRPCOLL_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    GRPCOLL_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
  }

  // Shape and contiguous checks
  GRPCOLL_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  GRPCOLL_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
  if (cached_mode) {
    GRPCOLL_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and cached_rdma_channel_prefix_matrix->size(1) == num_channels);
    GRPCOLL_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
    GRPCOLL_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and cached_gbl_channel_prefix_matrix->size(1) == num_channels);
    GRPCOLL_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
    GRPCOLL_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
  } else {
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    GRPCOLL_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
    GRPCOLL_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    GRPCOLL_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
  }

  auto num_tokens = static_cast<int>(x.size(0)), hidden_size = static_cast<int>(x.size(1)),
       hidden_int4 = static_cast<int>(hidden_size * x.element_size() / sizeof(int4));

  // LSE checks
  float* lse_ptr = nullptr;
  int num_heads = 0; // NOTES: when lse is not provided, num_heads is set to 0 and consumes empty buffer
  if (lse.has_value()) {
    GRPCOLL_HOST_ASSERT(lse->dim() == 2 and lse->is_contiguous());
    GRPCOLL_HOST_ASSERT(lse->scalar_type() == torch::kFloat32);
    GRPCOLL_HOST_ASSERT(lse->size(0) == num_tokens);
    GRPCOLL_HOST_ASSERT(hidden_size % lse->size(1) == 0); // hidden size should be divisible by num_heads
    num_heads = static_cast<int>(lse->size(1));
    lse_ptr = lse->data_ptr<float>();
  }

  // Set current stream to comm stream if needed
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
  auto rdma_channel_prefix_matrix = torch::Tensor();
  auto recv_rdma_rank_prefix_sum = torch::Tensor();
  auto gbl_channel_prefix_matrix = torch::Tensor();
  auto recv_gbl_rank_prefix_sum = torch::Tensor();

  // Notify
  if (cached_mode) {
    num_recv_tokens = cached_num_recv_tokens;
    num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
    rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
    recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
    gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
    recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

    // Just a barrier and clean flags
    internode::cached_notify(
        /*hidden_int4=*/hidden_int4,
        /*num_heads=*/num_heads,
        /*num_ranks=*/num_ranks,
        /*num_channels=*/num_channels,
        /*num_reduced_tokens=*/0,
        /*reduced_rdma_head=*/nullptr,
        /*rdma_channel_prefix_matrix=*/nullptr,
        /*rdma_rank_prefix_sum=*/nullptr,
        /*reduced_nvl_head=*/nullptr,
        /*rdma_buffer_ptr=*/rdma_buffer_ptr,
        /*num_max_rdma_chunked_recv_tokens=*/config.num_max_rdma_chunked_recv_tokens,
        /*buffer_ptrs=*/buffer_ptrs_gpu,
        /*num_max_nvl_chunked_recv_tokens=*/config.num_max_nvl_chunked_recv_tokens,
        /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
        /*rank=*/rank,
        /*stream=*/comm_stream,
        /*num_rdma_bytes=*/config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
        /*num_nvl_bytes=*/num_nvl_bytes,
        /*is_cached_group_cast=*/true,
        /*low_latency_mode=*/low_latency_mode);
  } else {
    rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Reset all the pinned counters to `-1`
    *grpcoll_recv_counter = -1, *grpcoll_recv_rdma_counter = -1;

    // Notify to clean RDMA/NVL buffers, switch meta data and calculate meta tensors for group cast
    // as well as set the pinned counters
    internode::notify_group_cast(
        /*num_tokens_per_rank=*/num_tokens_per_rank->data_ptr<int>(),
        /*grpcoll_recv_counter_mapped=*/grpcoll_recv_counter_mapped,
        /*num_ranks=*/num_ranks,
        /*num_tokens_per_rdma_rank=*/num_tokens_per_rdma_rank->data_ptr<int>(),
        /*grpcoll_recv_rdma_counter_mapped=*/grpcoll_recv_rdma_counter_mapped,
        /*is_token_in_rank=*/is_token_in_rank.data_ptr<bool>(),
        /*num_tokens=*/num_tokens,
        /*num_channels=*/num_channels,
        /*hidden_int4=*/hidden_int4,
        /*num_heads=*/num_heads,
        /*rdma_channel_prefix_matrix=*/rdma_channel_prefix_matrix.data_ptr<int>(),
        /*recv_rdma_rank_prefix_sum=*/recv_rdma_rank_prefix_sum.data_ptr<int>(),
        /*gbl_channel_prefix_matrix=*/gbl_channel_prefix_matrix.data_ptr<int>(),
        /*recv_gbl_rank_prefix_sum=*/recv_gbl_rank_prefix_sum.data_ptr<int>(),
        /*rdma_buffer_ptr=*/rdma_buffer_ptr,
        /*num_max_rdma_chunked_recv_tokens=*/config.num_max_rdma_chunked_recv_tokens,
        /*buffer_ptrs=*/buffer_ptrs_gpu,
        /*num_max_nvl_chunked_recv_tokens=*/config.num_max_nvl_chunked_recv_tokens,
        /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
        /*rank=*/rank,
        /*stream=*/comm_stream,
        /*num_rdma_bytes=*/config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
        /*num_nvl_bytes=*/num_nvl_bytes,
        /*low_latency_mode=*/low_latency_mode);

    // Synchronize total received tokens and received tokens for each RDMA peer
    // by let CPU wait for the pinned counters to be set
    // TODO: when `recv_x_buf` is provided, we can use its dim0 for `num_recv_tokens`
    // however, we still don't know `num_rdma_recv_tokens` to avoid CPU sync
    // one possible solution is to let user provide it, but had better find a better way
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
      // Read global count and each RDMA count
      num_recv_tokens = static_cast<int>(*grpcoll_recv_counter);
      num_rdma_recv_tokens = static_cast<int>(*grpcoll_recv_rdma_counter);
      bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);

      if (ready)
        break;

      // Timeout check
      if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS) {
        printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
        throw std::runtime_error("grpcoll error: CPU recv timeout for internode group cast");
      }
    }
  }

  // Allocate recv_x buffer
  auto recv_x = torch::Tensor();
  if (recv_x_buf.has_value()) {
    GRPCOLL_HOST_ASSERT(recv_x_buf->dim() == 2 && recv_x_buf->is_contiguous());
    GRPCOLL_HOST_ASSERT(recv_x_buf->scalar_type() == x.scalar_type());
    GRPCOLL_HOST_ASSERT(recv_x_buf->size(0) == num_recv_tokens and recv_x_buf->size(1) == hidden_size);
    recv_x = recv_x_buf.value();
  } else {
    recv_x = torch::empty({num_recv_tokens, hidden_size}, x.options());
  }

  // Assign ptr for post_perm_idx if needed
  int64_t* post_perm_idx_ptr = nullptr;
  if (post_perm_idx.has_value()) {
    GRPCOLL_HOST_ASSERT(post_perm_idx->scalar_type() == torch::kInt64);
    GRPCOLL_HOST_ASSERT(post_perm_idx->dim() == 1);
    GRPCOLL_HOST_ASSERT(post_perm_idx->size(0) == num_recv_tokens);
    post_perm_idx_ptr = post_perm_idx->data_ptr<int64_t>();
  }

  // Allocate new tensors
  auto recv_lse = std::optional<torch::Tensor>();
  auto recv_src_meta = std::optional<torch::Tensor>();
  auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
  auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
  auto send_rdma_head = std::optional<torch::Tensor>();
  auto send_nvl_head = std::optional<torch::Tensor>();
  if (not cached_mode) {
    recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
    recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
  }

  // Allocate recv_lse buffer and assign the ptr if needed
  float* recv_lse_ptr = nullptr;
  if (lse.has_value()) {
    if (recv_lse_buf.has_value()) {
      GRPCOLL_HOST_ASSERT(recv_lse_buf->dim() == 2 && recv_lse_buf->is_contiguous());
      GRPCOLL_HOST_ASSERT(recv_lse_buf->scalar_type() == torch::kFloat32);
      GRPCOLL_HOST_ASSERT(recv_lse_buf->size(0) == num_recv_tokens && recv_lse_buf->size(1) == num_heads);
      recv_lse.emplace(recv_lse_buf.value());
    } else {
      recv_lse = torch::empty({num_recv_tokens, num_heads}, lse->options());
    }
    recv_lse_ptr = recv_lse->data_ptr<float>();
  }

  // Launch group_cast kernel
  // NOTES: the buffer size checks are moved into the `.cu` file
  internode::group_cast(
      /*recv_x=*/recv_x.data_ptr(),
      /*recv_lse=*/recv_lse_ptr,
      /*x=*/x.data_ptr(),
      /*lse=*/lse_ptr,
      /*recv_src_meta=*/cached_mode ? nullptr : recv_src_meta->data_ptr(),
      /*send_rdma_head=*/cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
      /*send_nvl_head=*/cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
      /*recv_rdma_channel_prefix_matrix=*/cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
      /*recv_gbl_channel_prefix_matrix=*/cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
      /*rdma_channel_prefix_matrix=*/rdma_channel_prefix_matrix.data_ptr<int>(),
      /*recv_rdma_rank_prefix_sum=*/recv_rdma_rank_prefix_sum.data_ptr<int>(),
      /*gbl_channel_prefix_matrix=*/gbl_channel_prefix_matrix.data_ptr<int>(),
      /*recv_gbl_rank_prefix_sum=*/recv_gbl_rank_prefix_sum.data_ptr<int>(),
      /*is_token_in_rank=*/is_token_in_rank.data_ptr<bool>(),
      /*post_perm_idx=*/post_perm_idx_ptr,
      /*num_tokens=*/num_tokens,
      /*hidden_int4=*/hidden_int4,
      /*num_heads=*/num_heads,
      /*rdma_buffer_ptr=*/rdma_buffer_ptr,
      /*num_max_rdma_chunked_send_tokens=*/config.num_max_rdma_chunked_send_tokens,
      /*num_max_rdma_chunked_recv_tokens=*/config.num_max_rdma_chunked_recv_tokens,
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*num_max_nvl_chunked_send_tokens=*/config.num_max_nvl_chunked_send_tokens,
      /*num_max_nvl_chunked_recv_tokens=*/config.num_max_nvl_chunked_recv_tokens,
      /*rank=*/rank,
      /*num_ranks=*/num_ranks,
      /*is_cached_group_cast=*/cached_mode,
      /*stream=*/comm_stream,
      /*num_channels=*/num_channels,
      /*low_latency_mode=*/low_latency_mode);

  // Record or wait streams
  std::optional<EventHandle> event;
  if (async_op) { // Record tensors on non-allocated stream
    event = EventHandle(comm_stream);
    // record tensors
    for (auto& t : {x, is_token_in_rank, recv_x, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream)
        t.record_stream(compute_stream);
    }
    // record optional tensors
    for (auto& to :
         {lse,
          recv_lse,
          num_tokens_per_rank,
          num_tokens_per_rdma_rank,
          cached_rdma_channel_prefix_matrix,
          cached_recv_rdma_rank_prefix_sum,
          cached_gbl_channel_prefix_matrix,
          cached_recv_gbl_rank_prefix_sum,
          post_perm_idx,
          recv_rdma_channel_prefix_matrix,
          recv_gbl_channel_prefix_matrix,
          send_rdma_head,
          send_nvl_head,
          recv_src_meta}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else { // Wait tensors on compute stream
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back current stream to compute stream if needed
  if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(compute_stream);

  // Return values
  return {
      recv_x,
      recv_lse,
      rdma_channel_prefix_matrix,
      gbl_channel_prefix_matrix,
      recv_rdma_channel_prefix_matrix,
      recv_rdma_rank_prefix_sum,
      recv_gbl_channel_prefix_matrix,
      recv_gbl_rank_prefix_sum,
      recv_src_meta,
      send_rdma_head,
      send_nvl_head,
      event};
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
  return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>> Buffer::internode_group_reduce(
    const torch::Tensor& x,
    std::optional<torch::Tensor>& reduced_x_buf,
    const torch::Tensor& src_meta,
    const torch::Tensor& is_reduced_token_in_rank,
    const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum,
    const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& reduced_rdma_head,
    const torch::Tensor& reduced_nvl_head,
    const std::optional<torch::Tensor>& pre_perm_idx,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async_op,
    bool allocate_on_comm_stream,
    const std::string& reduce_op,
    bool acc_reduce,
    std::optional<c10::ScalarType> comm_dtype) {
  // Transfer reduce ops
  ReduceOp reduce_op_ = str_to_reduce_op(reduce_op);
  // TODO: support other reduce ops
  GRPCOLL_HOST_ASSERT(reduce_op_ == ReduceOp::SUM);

  // Transfer dtypes and item sizes in bytes
  auto x_dtype = x.scalar_type();
  auto comm_dtype_ = comm_dtype.value_or(x_dtype);
  auto comm_elem_size = c10::elementSize(comm_dtype_);

#ifndef DISABLE_NVSHMEM
  // One channel use two SMs
  // one for forwarders, the other for (senders, receivers)
  const int num_channels = config.num_sms / 2;
  GRPCOLL_HOST_ASSERT(config.num_sms % 2 == 0);

  // Shape and contiguous checks
  GRPCOLL_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  GRPCOLL_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
  GRPCOLL_HOST_ASSERT(is_reduced_token_in_rank.dim() == 2 and is_reduced_token_in_rank.is_contiguous() and is_reduced_token_in_rank.scalar_type() == torch::kBool);
  GRPCOLL_HOST_ASSERT(
      rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(reduced_rdma_head.dim() == 2 and reduced_rdma_head.is_contiguous() and reduced_rdma_head.scalar_type() == torch::kInt32);
  GRPCOLL_HOST_ASSERT(reduced_nvl_head.dim() == 2 and reduced_nvl_head.is_contiguous() and reduced_nvl_head.scalar_type() == torch::kInt32);

  const auto num_tokens = static_cast<int>(x.size(0)), num_reduced_tokens = static_cast<int>(is_reduced_token_in_rank.size(0));
  const auto hidden_size = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(hidden_size * comm_elem_size / sizeof(int4));
  GRPCOLL_HOST_ASSERT((hidden_size * comm_elem_size) % sizeof(int4) == 0); // hidden comm bytes should be aligned with int4
  GRPCOLL_HOST_ASSERT(hidden_int4 % WARP_SIZE == 0); // hidden size in int4 should be aligned with warp size
  GRPCOLL_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
  GRPCOLL_HOST_ASSERT(is_reduced_token_in_rank.size(1) == num_ranks);
  GRPCOLL_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
  GRPCOLL_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
  GRPCOLL_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
  GRPCOLL_HOST_ASSERT(reduced_rdma_head.dim() == 2 and reduced_rdma_head.size(0) == num_reduced_tokens and reduced_rdma_head.size(1) == num_rdma_ranks);
  GRPCOLL_HOST_ASSERT(reduced_nvl_head.dim() == 2 and reduced_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

  // Set current stream to comm stream if needed
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Assign ptr for pre_perm_idx if needed
  int64_t* pre_perm_idx_ptr = nullptr;
  if (pre_perm_idx.has_value()) {
    GRPCOLL_HOST_ASSERT(pre_perm_idx->dim() == 1 && pre_perm_idx->is_contiguous());
    GRPCOLL_HOST_ASSERT(pre_perm_idx->scalar_type() == torch::kInt64);
    GRPCOLL_HOST_ASSERT(pre_perm_idx->size(0) == num_tokens);
    pre_perm_idx_ptr = pre_perm_idx->data_ptr<int64_t>();
  }

  // Extra check for avoid-dead-lock design
  GRPCOLL_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  GRPCOLL_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

  // Launch barrier and reset queue head and tail
  internode::cached_notify(
      /*hidden_int4=*/hidden_int4,
      /*num_heads=*/0,
      /*num_ranks=*/num_ranks,
      /*num_channels=*/num_channels,
      /*num_reduced_tokens=*/num_reduced_tokens,
      /*reduced_rdma_head=*/reduced_rdma_head.data_ptr<int>(),
      /*rdma_channel_prefix_matrix=*/rdma_channel_prefix_matrix.data_ptr<int>(),
      /*rdma_rank_prefix_sum=*/rdma_rank_prefix_sum.data_ptr<int>(),
      /*reduced_nvl_head=*/reduced_nvl_head.data_ptr<int>(),
      /*rdma_buffer_ptr=*/rdma_buffer_ptr,
      /*num_max_rdma_chunked_recv_tokens=*/config.num_max_rdma_chunked_recv_tokens,
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*num_max_nvl_chunked_recv_tokens=*/config.num_max_nvl_chunked_recv_tokens,
      /*barrier_signal_ptrs=*/barrier_signal_ptrs_gpu,
      /*rank=*/rank,
      /*stream=*/comm_stream,
      /*num_rdma_bytes=*/config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
      /*num_nvl_bytes=*/num_nvl_bytes,
      /*is_cached_dispatch=*/false,
      /*low_latency_mode=*/low_latency_mode);

  // Allocate reduced_x buffer
  auto reduced_x = torch::Tensor();
  if (reduced_x_buf.has_value()) {
    GRPCOLL_HOST_ASSERT(reduced_x_buf->dim() == 2 and reduced_x_buf->is_contiguous());
    GRPCOLL_HOST_ASSERT(reduced_x_buf->scalar_type() == x_dtype);
    GRPCOLL_HOST_ASSERT(reduced_x_buf->size(0) == num_reduced_tokens and reduced_x_buf->size(1) == hidden_size);
    reduced_x = reduced_x_buf.value();
  } else {
    GRPCOLL_HOST_ASSERT(!acc_reduce); // no point to acc_reduce if reduced_x_buf is not provided
    /** NOTE: different from ep, for group-reduce,
     * some token in `reduced_x` might not reduce anything,
     * since the corr. token has no destination rank in the corr. group-cast
     * so we have to zero-initialize `reduced_x`, instead of empty initialization
     * however, we handle the zero initialization inside the group reduce kernel
     * thus here, we still use empty initialization
     */
    reduced_x = torch::empty({num_reduced_tokens, hidden_size}, x.options());
  }

  // Launch group reduce kernel
  internode::group_reduce(
      /*reduced_x=*/reduced_x.data_ptr(),
      /*reduced_lse=*/nullptr,
      /*x=*/x.data_ptr(),
      /*lse=*/nullptr,
      /*is_reduced_token_in_rank=*/is_reduced_token_in_rank.data_ptr<bool>(),
      /*reduced_rdma_head=*/reduced_rdma_head.data_ptr<int>(),
      /*reduced_nvl_head=*/reduced_nvl_head.data_ptr<int>(),
      /*src_meta=*/src_meta.data_ptr(),
      /*rdma_channel_prefix_matrix=*/rdma_channel_prefix_matrix.data_ptr<int>(),
      /*rdma_rank_prefix_sum=*/rdma_rank_prefix_sum.data_ptr<int>(),
      /*gbl_channel_prefix_matrix=*/gbl_channel_prefix_matrix.data_ptr<int>(),
      /*pre_perm_idx*/ pre_perm_idx_ptr,
      /*num_tokens=*/num_tokens,
      /*num_reduced_tokens=*/num_reduced_tokens,
      /*hidden_size=*/hidden_size,
      /*num_heads=*/0, // TODO: support lse reduce
      /*rdma_buffer_ptr=*/rdma_buffer_ptr,
      /*num_max_rdma_chunked_send_tokens=*/config.num_max_rdma_chunked_send_tokens,
      /*num_max_rdma_chunked_recv_tokens=*/config.num_max_rdma_chunked_recv_tokens,
      /*buffer_ptrs=*/buffer_ptrs_gpu,
      /*num_max_nvl_chunked_send_tokens=*/config.num_max_nvl_chunked_send_tokens,
      /*num_max_nvl_chunked_recv_tokens=*/config.num_max_nvl_chunked_recv_tokens,
      /*rank=*/rank,
      /*num_ranks=*/num_ranks,
      /*stream=*/comm_stream,
      /*num_channels=*/num_channels,
      /*low_latency_mode=*/low_latency_mode,
      /*acc_reduce=*/acc_reduce,
      /*dtype=*/at::cuda::ScalarTypeToCudaDataType(x_dtype),
      /*comm_dtype=*/at::cuda::ScalarTypeToCudaDataType(comm_dtype_),
      /*reduce_op=*/reduce_op_);

  // Record or wait streams
  std::optional<EventHandle> event;
  if (async_op) { // Record tensors on non-allocated stream
    event = EventHandle(comm_stream);
    // record tensors
    for (auto& t :
         {x,
          src_meta,
          is_reduced_token_in_rank,
          rdma_channel_prefix_matrix,
          rdma_rank_prefix_sum,
          gbl_channel_prefix_matrix,
          reduced_x,
          reduced_rdma_head,
          reduced_nvl_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream)
        t.record_stream(compute_stream);
    }
    // record optional tensors
    for (auto& to : {pre_perm_idx}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream)
        to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else { // Wait tensors on compute stream
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back current stream to compute stream if needed
  if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(compute_stream);

  // Return values
  return {reduced_x, event};
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
  return {};
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common Helper APIs
///////////////////////////////////////////////////////////////////////////////////////////////////

bool Buffer::is_available() const {
  return available;
}

bool Buffer::is_internode_available() const {
  return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
  return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
  return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
  return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
  return device_id;
}

py::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

py::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
  GRPCOLL_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
  auto unique_id = internode::get_unique_id();
  return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_local_buffer_tensor(const py::object& dtype, int64_t offset, bool use_rdma_buffer) const {
  torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
  auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
  auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
  auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
  return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
  return comm_stream;
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
  return true;
#else
  return false;
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Low Latency APIs
///////////////////////////////////////////////////////////////////////////////////////////////////

// NOTES: we remain original low-latency APIs here for future potential usage,
// which won't be exposed to users for now, but guaranteed its compatibility internally
std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_fp8,
    bool round_scale,
    bool use_ue8m0,
    bool async_op,
    bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
  GRPCOLL_HOST_ASSERT(low_latency_mode);

  // Tensor checks
  // By default using `ptp128c` FP8 cast
  GRPCOLL_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
  GRPCOLL_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
  GRPCOLL_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
  GRPCOLL_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
  GRPCOLL_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
  GRPCOLL_HOST_ASSERT(num_experts % num_ranks == 0);
  if (cumulative_local_expert_recv_stats.has_value()) {
    GRPCOLL_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
    GRPCOLL_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
    GRPCOLL_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
  }

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
  auto num_topk = static_cast<int>(topk_idx.size(1));
  auto num_local_experts = num_experts / num_ranks;

  // Buffer control
  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  GRPCOLL_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  GRPCOLL_HOST_ASSERT(not(async_op and return_recv_hook));
  if (not return_recv_hook)
    stream_wait(launch_stream, compute_stream);

  // Allocate packed tensors
  auto packed_recv_x =
      torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden}, x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
  auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  // Allocate column-majored scales
  auto packed_recv_x_scales = std::optional<torch::Tensor>();
  void* packed_recv_x_scales_ptr = nullptr;
  GRPCOLL_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

  if (use_fp8) {
    // TODO: support unaligned cases
    GRPCOLL_HOST_ASSERT(hidden % 512 == 0);
    if (not use_ue8m0) {
      packed_recv_x_scales =
          torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    } else {
      GRPCOLL_HOST_ASSERT(round_scale);
      packed_recv_x_scales =
          torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt).device(torch::kCUDA));
    }
    packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
    packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
  }

  // Kernel launch
  auto next_clean_meta = next_buffer.clean_meta();
  auto launcher = [=](int phases) {
    internode_ll::dispatch(
        packed_recv_x.data_ptr(),
        packed_recv_x_scales_ptr,
        packed_recv_src_info.data_ptr<int>(),
        packed_recv_layout_range.data_ptr<int64_t>(),
        packed_recv_count.data_ptr<int>(),
        cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
        buffer.dispatch_rdma_recv_data_buffer,
        buffer.dispatch_rdma_recv_count_buffer,
        buffer.dispatch_rdma_send_buffer,
        x.data_ptr(),
        topk_idx.data_ptr<int64_t>(),
        next_clean_meta.first,
        next_clean_meta.second,
        num_tokens,
        hidden,
        num_max_dispatch_tokens_per_rank,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        use_fp8,
        round_scale,
        use_ue8m0,
        workspace,
        num_device_sms,
        launch_stream,
        phases);
  };
  launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  // Wait streams
  std::optional<EventHandle> event;
  if (async_op) {
    // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
    // so in Python API, we must wrap all tensors into the event handle.
    event = EventHandle(launch_stream);
  } else if (not return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  // Receiver callback
  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook)
    recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  // Return values
  return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
  return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
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
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
  GRPCOLL_HOST_ASSERT(low_latency_mode);

  // Tensor checks
  GRPCOLL_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
  GRPCOLL_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
  GRPCOLL_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
  GRPCOLL_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
  GRPCOLL_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
  GRPCOLL_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
  GRPCOLL_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
  GRPCOLL_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
  GRPCOLL_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
  GRPCOLL_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
  GRPCOLL_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
  GRPCOLL_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
  GRPCOLL_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
  GRPCOLL_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
  GRPCOLL_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);
  auto hidden = static_cast<int>(x.size(2));
  auto num_topk = static_cast<int>(topk_weights.size(1));
  auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

  // Buffer control
  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  GRPCOLL_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  // Wait previous tasks to be finished
  // NOTES: the hook mode will always use the default stream
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  GRPCOLL_HOST_ASSERT(not(async_op and return_recv_hook));
  if (not return_recv_hook)
    stream_wait(launch_stream, compute_stream);

  // Allocate output tensor
  torch::Tensor combined_x;
  if (out.has_value()) {
    GRPCOLL_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
    GRPCOLL_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
    GRPCOLL_HOST_ASSERT(out->scalar_type() == x.scalar_type());
    combined_x = out.value();
  } else {
    combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
  }

  // Kernel launch
  auto next_clean_meta = next_buffer.clean_meta();
  auto launcher = [=](int phases) {
    internode_ll::combine(
        combined_x.data_ptr(),
        buffer.combine_rdma_recv_data_buffer,
        buffer.combine_rdma_recv_flag_buffer,
        buffer.combine_rdma_send_buffer,
        x.data_ptr(),
        topk_idx.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        src_info.data_ptr<int>(),
        layout_range.data_ptr<int64_t>(),
        next_clean_meta.first,
        next_clean_meta.second,
        num_combined_tokens,
        hidden,
        num_max_dispatch_tokens_per_rank,
        num_topk,
        num_experts,
        rank,
        num_ranks,
        use_logfmt,
        workspace,
        num_device_sms,
        launch_stream,
        phases,
        zero_copy);
  };
  launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  // Wait streams
  std::optional<EventHandle> event;
  if (async_op) {
    // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
    // so in Python API, we must wrap all tensors into the event handle.
    event = EventHandle(launch_stream);
  } else if (not return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  // Receiver callback
  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook)
    recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  // Return values
  return {combined_x, event, recv_hook};
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
  return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
  GRPCOLL_HOST_ASSERT(low_latency_mode);

  auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  auto clean_meta_0 = layout.buffers[0].clean_meta();
  auto clean_meta_1 = layout.buffers[1].clean_meta();

  auto check_boundary = [=](void* ptr, size_t num_bytes) {
    auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
    GRPCOLL_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
  };
  check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
  check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

  internode_ll::clean_low_latency_buffer(clean_meta_0.first, clean_meta_0.second, clean_meta_1.first, clean_meta_1.second, at::cuda::getCurrentCUDAStream());
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto dtype = torch::kBFloat16;
  auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

  GRPCOLL_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
  return torch::from_blob(
      buffer.combine_rdma_send_buffer_data_start,
      {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
      {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
      torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
#else
  GRPCOLL_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
  return {};
#endif
}

} // namespace magi_attn_comm::grpcoll

///////////////////////////////////////////////////////////////////////////////////////////////////
// GrpColl SubModule PyBind
///////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Magi Attention Communication Library";

  /**********************     GrpColl Sub-module     **********************/

  py::module_ grpcoll_submodule = m.def_submodule("grpcoll", "Group-Collective Communication Sub-Library based on DeepEP");
  // Config class
  py::class_<grpcoll::Config>(grpcoll_submodule, "Config")
      .def(
          py::init<int, int, int, int, int>(),
          py::arg("num_sms") = 20,
          py::arg("num_max_nvl_chunked_send_tokens") = 6,
          py::arg("num_max_nvl_chunked_recv_tokens") = 256,
          py::arg("num_max_rdma_chunked_send_tokens") = 6,
          py::arg("num_max_rdma_chunked_recv_tokens") = 256)
      .def("get_nvl_buffer_size_hint", &grpcoll::Config::get_nvl_buffer_size_hint)
      .def("get_rdma_buffer_size_hint", &grpcoll::Config::get_rdma_buffer_size_hint);

  // Eventhandle class
  py::class_<grpcoll::EventHandle>(grpcoll_submodule, "EventHandle").def(py::init<>()).def("current_stream_wait", &grpcoll::EventHandle::current_stream_wait);

  // Meta class
  py::class_<grpcoll::Meta>(grpcoll_submodule, "Meta")
      .def_static("get_group_cast_meta_from_t2r_idx", &grpcoll::Meta::get_group_cast_meta_from_t2r_idx)
      .def_static("get_a2av_perm_idx_from_src_idx", &grpcoll::Meta::get_a2av_perm_idx_from_src_idx);

  // Buffer class
  py::class_<grpcoll::Buffer>(grpcoll_submodule, "Buffer")
      .def(py::init<int, int, int64_t, int64_t, bool, bool>())
      .def("is_available", &grpcoll::Buffer::is_available)
      .def("get_num_rdma_ranks", &grpcoll::Buffer::get_num_rdma_ranks)
      .def("get_rdma_rank", &grpcoll::Buffer::get_rdma_rank)
      .def("get_root_rdma_rank", &grpcoll::Buffer::get_root_rdma_rank)
      .def("get_local_device_id", &grpcoll::Buffer::get_local_device_id)
      .def("get_local_ipc_handle", &grpcoll::Buffer::get_local_ipc_handle)
      .def("get_local_nvshmem_unique_id", &grpcoll::Buffer::get_local_nvshmem_unique_id)
      .def("get_local_buffer_tensor", &grpcoll::Buffer::get_local_buffer_tensor)
      .def("get_comm_stream", &grpcoll::Buffer::get_comm_stream)
      .def("sync", &grpcoll::Buffer::sync)
      .def("destroy", &grpcoll::Buffer::destroy)
      .def("get_group_cast_meta", &grpcoll::Buffer::get_group_cast_meta)
      .def("intranode_group_cast", &grpcoll::Buffer::intranode_group_cast)
      .def("intranode_group_reduce", &grpcoll::Buffer::intranode_group_reduce)
      .def("internode_group_cast", &grpcoll::Buffer::internode_group_cast)
      .def("internode_group_reduce", &grpcoll::Buffer::internode_group_reduce)
      .def("clean_low_latency_buffer", &grpcoll::Buffer::clean_low_latency_buffer)
      .def("low_latency_dispatch", &grpcoll::Buffer::low_latency_dispatch)
      .def("low_latency_combine", &grpcoll::Buffer::low_latency_combine)
      .def("get_next_low_latency_combine_buffer", &grpcoll::Buffer::get_next_low_latency_combine_buffer);

  // other functions
  grpcoll_submodule.def("get_low_latency_rdma_size_hint", &grpcoll::get_low_latency_rdma_size_hint);
  grpcoll_submodule.def("is_sm90_compiled", grpcoll::is_sm90_compiled);
}
