/**********************************************************************************
 * Copyright (c) 2025 SandAI. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *********************************************************************************/

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#include "cuda_check.h"

#define NUM_THREADS 256

/**
 * @brief Kernel to compute sparse load loop counts and invalid token counts for each unique Q range.
 *
 * @param k_ranges K ranges tensor [N, 2]
 * @param cu_k_ranges_num Cumulative count [unique_count+1]
 * @param sparse_loads Output: load loop counts [unique_count]
 * @param last_loop_invalid_count Output: number of invalid (masked) tokens in the last loop [unique_count]
 * @param is_equal Output: global consistency flag (1 if all k_ranges have same length, 0 otherwise)
 * @param tile_size Tile size
 * @param unique_count Number of unique Q ranges
 */
__global__ void compute_sparse_load_kernel(
    const int* k_ranges,
    const int* cu_k_ranges_num,
    int* sparse_loads,
    int* last_loop_invalid_count,
    int* is_equal, // Global consistency flag
    int tile_size,
    int unique_count) {
  int unique_idx = blockIdx.x;
  if (unique_idx >= unique_count)
    return;

  // Get the start and end indices for this unique Q range
  int start_idx = cu_k_ranges_num[unique_idx];
  int end_idx = cu_k_ranges_num[unique_idx + 1];
  int num_k_ranges = end_idx - start_idx;

  // Each thread processes some K ranges
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // Reference length from the first global range
  int ref_len = k_ranges[1] - k_ranges[0];

  int local_sum = 0;

  for (int i = tid; i < num_k_ranges; i += block_size) {
    int k_idx = start_idx + i;
    int k_start = k_ranges[k_idx * 2];
    int k_end = k_ranges[k_idx * 2 + 1];
    int len = k_end - k_start;

    local_sum += len;

    // check K range size equal
    if (len != ref_len) {
      // if already false, no need to do atomicExch again
      if (*(volatile int*)is_equal == 1) {
        atomicExch(is_equal, 0);
      }
    }
  }

  // Block-level reduction using shared memory
  __shared__ int shared_sum[NUM_THREADS];
  shared_sum[tid] = local_sum;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  // Thread 0 computes final result and writes output
  if (tid == 0) {
    int total_k_range = shared_sum[0];

    // Compute ceil_div(total_k_range, tile_size)
    int load_count = (total_k_range + tile_size - 1) / tile_size;
    sparse_loads[unique_idx] = load_count;

    // Calculate invalid tokens in the last loop (mask needed)
    // Logic: Total Capacity (aligned to tile size) - Actual Tokens
    int invalid_count = (load_count * tile_size) - total_k_range;

    last_loop_invalid_count[unique_idx] = invalid_count;
  }
}

/**
 * @brief Computes sparse load loop count and the invalid token count for the last loop.
 *
 * @return std::vector<torch::Tensor> {sparse_loads, last_loop_invalid_count}
 */
std::tuple<torch::Tensor, torch::Tensor, bool> compute_sparse_load_metadata(
    torch::Tensor k_ranges, // (n, 2)
    torch::Tensor cu_k_ranges_num, // (unique_count + 1, )
    torch::Tensor unique_count,
    int tile_size) {
  // Validate inputs
  TORCH_CHECK(k_ranges.is_cuda(), "k_ranges must be a CUDA tensor");
  TORCH_CHECK(cu_k_ranges_num.is_cuda(), "cu_k_ranges_num must be a CUDA tensor");
  TORCH_CHECK(unique_count.is_cuda(), "unique_count must be a CUDA tensor");
  TORCH_CHECK(k_ranges.dim() == 2 && k_ranges.size(1) == 2, "k_ranges must be [N, 2] tensor");
  TORCH_CHECK(k_ranges.scalar_type() == torch::kInt32, "k_ranges must be int32");
  TORCH_CHECK(cu_k_ranges_num.scalar_type() == torch::kInt32, "cu_k_ranges_num must be int32");
  TORCH_CHECK(unique_count.scalar_type() == torch::kInt32, "unique_count must be int32");
  TORCH_CHECK(k_ranges.is_contiguous(), "k_ranges must be contiguous");
  TORCH_CHECK(cu_k_ranges_num.is_contiguous(), "cu_k_ranges_num must be contiguous");
  TORCH_CHECK(tile_size > 0, "tile_size must be positive");

  // Get unique_count value from device to host
  int h_unique_count = unique_count.item<int>();
  int num_ranges = k_ranges.size(0);

  if (h_unique_count == 0) {
    auto opts = k_ranges.options().dtype(torch::kInt32);
    return {torch::empty({0}, opts), torch::empty({0}, opts), false};
  }

  // Allocate output tensors
  auto options = k_ranges.options().dtype(torch::kInt32);
  auto sparse_loads = torch::empty({h_unique_count}, options);
  auto last_loop_invalid_count = torch::empty({h_unique_count}, options); // Updated name
  auto is_equal_tensor = torch::full({1}, 1, options); // Init to True (1)

  // Launch kernel
  int threadsPerBlock = NUM_THREADS;
  int numBlocks = h_unique_count;

  compute_sparse_load_kernel<<<numBlocks, threadsPerBlock>>>(
      k_ranges.data_ptr<int>(),
      cu_k_ranges_num.data_ptr<int>(),
      sparse_loads.data_ptr<int>(),
      last_loop_invalid_count.data_ptr<int>(), // Pass updated pointer
      is_equal_tensor.data_ptr<int>(),
      tile_size,
      h_unique_count);

  CHECK_CUDA_KERNEL_LAUNCH();

  bool equal_k_range_size = (is_equal_tensor.item<int>() == 1);

  return {sparse_loads, last_loop_invalid_count, equal_k_range_size};
}
