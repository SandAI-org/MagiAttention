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

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuda_check.h"

#define NUM_THREADS 256

/**
 * @brief Kernel to compute sparse load loop counts for each unique Q range.
 *
 * Each block processes one unique Q range:
 * 1. Threads in the block cooperatively sum all K range lengths for this unique Q range
 * 2. Compute ceil_div(total_k_range, tile_size) to get the sparse load count
 *
 * @param k_ranges K ranges tensor [N, 2], where each row is [k_start, k_end)
 * @param cu_k_ranges_num Cumulative count [unique_count+1], indices [cu_k_ranges_num[i], cu_k_ranges_num[i+1]) are K ranges for unique Q range i
 * @param sparse_loads Output tensor [unique_count] to store the load loop counts
 * @param tile_size Tile size for computing load counts
 * @param unique_count Number of unique Q ranges
 */
__global__ void compute_sparse_load_kernel(const int* k_ranges, const int* cu_k_ranges_num, int* sparse_loads, int tile_size, int unique_count) {
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

  // Thread-local accumulation
  int local_sum = 0;
  for (int i = tid; i < num_k_ranges; i += block_size) {
    int k_idx = start_idx + i;
    int k_start = k_ranges[k_idx * 2];
    int k_end = k_ranges[k_idx * 2 + 1];
    local_sum += (k_end - k_start);
  }

  // Block-level reduction using shared memory
  __shared__ int shared_sum[NUM_THREADS]; // Max 256 threads per block
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
  }
}

/**
 * @brief Computes the sparse load loop count for each unique Q range.
 *
 * For each unique Q range, this function:
 * 1. Sums the lengths of all associated K ranges
 * 2. Computes ceil_div(total_k_range, tile_size) to get the number of sparse loads needed
 *
 * @param k_ranges K ranges tensor [N, 2], dtype=int32
 * @param cu_k_ranges_num Cumulative count [unique_count+1], dtype=int32
 * @param unique_count Number of unique Q ranges (scalar tensor), dtype=int32
 * @param tile_size Tile size for computing load counts
 * @return torch::Tensor Sparse load loop counts [unique_count], dtype=int32
 */
torch::Tensor compute_sparse_load_loop_count(
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

  if (h_unique_count == 0) {
    return torch::empty({0}, k_ranges.options().dtype(torch::kInt32));
  }

  // Allocate output tensor
  auto sparse_loads = torch::empty({h_unique_count}, k_ranges.options().dtype(torch::kInt32));

  // Launch kernel: each block processes one unique Q range
  int threadsPerBlock = NUM_THREADS;
  int numBlocks = h_unique_count;

  compute_sparse_load_kernel<<<numBlocks, threadsPerBlock>>>(
      k_ranges.data_ptr<int>(), cu_k_ranges_num.data_ptr<int>(), sparse_loads.data_ptr<int>(), tile_size, h_unique_count);
  CHECK_CUDA_KERNEL_LAUNCH();

  return sparse_loads;
}
