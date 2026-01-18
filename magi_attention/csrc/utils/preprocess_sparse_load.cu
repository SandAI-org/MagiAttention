/**********************************************************************************
 * Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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
    const int* attn_type_map,
    int* sparse_loads,
    uint8_t* last_loop_invalid_count,
    int* is_equal, // Global consistency flag
    int tile_size,
    const int* unique_count) {
  int total_count = *unique_count;
  for (int unique_idx = blockIdx.x; unique_idx < total_count; unique_idx += gridDim.x) {
    const int2* k_ranges_vec = reinterpret_cast<const int2*>(k_ranges);

    // Get the start and end indices for this unique Q range
    int start_idx = cu_k_ranges_num[unique_idx];
    int end_idx = cu_k_ranges_num[unique_idx + 1];
    int num_k_ranges = end_idx - start_idx;

    // Each thread processes some K ranges
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Reference length from the first global range
    int2 ref_val = k_ranges_vec[0];
    int ref_len = ref_val.y - ref_val.x;

    int local_sum = 0;

    for (int i = tid; i < num_k_ranges; i += block_size) {
      int k_idx = start_idx + i;
      int2 range = k_ranges_vec[k_idx];
      int len = range.y - range.x;

      if (attn_type_map != nullptr) {
        int type = attn_type_map[k_idx];
        CUDA_KERNEL_ASSERT(type == 0 && "Sparse load only supports full attention in Q/K ranges!");
      }

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

      last_loop_invalid_count[unique_idx] = (uint8_t)invalid_count;
    }
    __syncthreads();
  }
}

/**
 * @brief Computes sparse load loop count, invalid token count for the last loop and
 * the flag of equal k range size.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_sparse_load_metadata(
    torch::Tensor k_ranges, // (n, 2)
    torch::Tensor cu_k_ranges_num, // (unique_count + 1, )
    torch::Tensor unique_count,
    torch::optional<torch::Tensor> attn_type_map,
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

  const int* attn_type_map_ptr = nullptr;
  if (attn_type_map.has_value() && attn_type_map.value().defined()) {
    auto t = attn_type_map.value();
    TORCH_CHECK(t.is_cuda(), "attn_type_map must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "attn_type_map must be int32");
    TORCH_CHECK(t.is_contiguous(), "attn_type_map must be contiguous");
    attn_type_map_ptr = t.data_ptr<int>();
  }

  int num_ranges = k_ranges.size(0);
  auto options = k_ranges.options().dtype(torch::kInt32);
  auto options_uint8 = k_ranges.options().dtype(torch::kUInt8);
  auto is_equal_tensor = torch::full({1}, 1, options); // Init to True (1)

  if (num_ranges == 0) {
    return {torch::empty({0}, options), torch::empty({0}, options_uint8), is_equal_tensor};
  }

  // Allocate output tensors
  auto sparse_loads = torch::empty({num_ranges}, options);
  // at most tile_size - 1 invalid tokens in the last loop, tile_size is fixed to 128 currently
  auto last_loop_invalid_count = torch::empty({num_ranges}, options_uint8);

  // Launch kernel
  int threadsPerBlock = NUM_THREADS;
  int numBlocks = std::min((int)num_ranges, 1024);

  compute_sparse_load_kernel<<<numBlocks, threadsPerBlock>>>(
      k_ranges.data_ptr<int>(),
      cu_k_ranges_num.data_ptr<int>(),
      attn_type_map_ptr,
      sparse_loads.data_ptr<int>(),
      last_loop_invalid_count.data_ptr<uint8_t>(),
      is_equal_tensor.data_ptr<int>(),
      tile_size,
      unique_count.data_ptr<int>());

  CHECK_CUDA_KERNEL_LAUNCH();

  return {sparse_loads, last_loop_invalid_count, is_equal_tensor};
}
