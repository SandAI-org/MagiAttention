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
#include <cub/device/device_scan.cuh>
#include <vector>

#include "cuda_check.h"

// ===================================================================
// Data Structure for (int, int) Pairs
// ===================================================================
struct IntPair {
  int x;
  int y;

  __host__ __device__ bool operator<(const IntPair& other) const {
    if (x < other.x)
      return true;
    if (x > other.x)
      return false;
    return y < other.y;
  }
  __host__ __device__ bool operator!=(const IntPair& other) const {
    return x != other.x || y != other.y;
  }
};

// ===================================================================
// Custom Kernels
// ===================================================================
__global__ void mark_uniques(const IntPair* sorted_pairs, int* d_flags, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  bool is_unique = (idx == 0) || (sorted_pairs[idx] != sorted_pairs[idx - 1]);
  d_flags[idx] = is_unique ? 1 : 0;
}

__global__ void gather_uniques(const IntPair* sorted_pairs, const int* d_flags, const int* d_write_indices, IntPair* unique_pairs, int* unique_indices, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  if (d_flags[idx] == 1) {
    int write_idx = d_write_indices[idx];
    unique_pairs[write_idx] = sorted_pairs[idx];
    unique_indices[write_idx] = idx;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_consecutive_pairs_ext(torch::Tensor sorted_input_tensor) {
  // check input tensor
  TORCH_CHECK(sorted_input_tensor.is_cuda(), "Input tensor must be a CUDA tensor");
  TORCH_CHECK(sorted_input_tensor.dim() == 2 && sorted_input_tensor.size(1) == 2, "Input must be a [N, 2] tensor");
  TORCH_CHECK(sorted_input_tensor.scalar_type() == torch::kInt32, "Input tensor must be of type int32");
  TORCH_CHECK(sorted_input_tensor.is_contiguous(), "Input tensor must be contiguous");

  int n = sorted_input_tensor.size(0); // total elements
  if (n == 0) {
    return {
        torch::empty({0, 2}, sorted_input_tensor.options()),
        torch::empty({0}, sorted_input_tensor.options().dtype(torch::kInt32)),
        torch::empty({0}, sorted_input_tensor.options().dtype(torch::kInt32))};
  }

  // d_flags(i): whether the input tensor value is the first unique value at idx i.
  // d_write_indices(i):
  auto d_flags = torch::empty({n}, sorted_input_tensor.options().dtype(torch::kInt32));
  auto d_write_indices = torch::empty({n}, sorted_input_tensor.options().dtype(torch::kInt32));
  // 1. get int* pointer from input_tensor
  const int* raw_int_ptr = sorted_input_tensor.data_ptr<int>();
  // 2. reinterpret_cast to IntPair* pointer
  const IntPair* d_sorted_pairs_ptr = reinterpret_cast<const IntPair*>(raw_int_ptr);
  // ==========================================================

  // --- Pass 1: flag the unique_value ---
  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  mark_uniques<<<blocksPerGrid, threadsPerBlock>>>(d_sorted_pairs_ptr, d_flags.data_ptr<int>(), n);
  CHECK_CUDA_KERNEL_LAUNCH();

  // --- Pass 2: compute unique_count and exclusive_sum from d_flags  ---
  auto d_unique_count_out = d_flags.sum(torch::kInt32);

  void* d_temp_storage_scan = nullptr;
  size_t temp_storage_bytes_scan = 0;
  cub::Sum scan_op;
  int initial_value = 0;

  cub::DeviceScan::ExclusiveScan(d_temp_storage_scan, temp_storage_bytes_scan, d_flags.data_ptr<int>(), d_write_indices.data_ptr<int>(), scan_op, initial_value, n);
  auto d_temp_storage_tensor = torch::empty({(long)temp_storage_bytes_scan}, sorted_input_tensor.options().dtype(torch::kByte));
  // Get the raw pointer from this byte tensor. It now points to a memory block of the exact required size.
  d_temp_storage_scan = d_temp_storage_tensor.data_ptr();
  cub::DeviceScan::ExclusiveScan(d_temp_storage_scan, temp_storage_bytes_scan, d_flags.data_ptr<int>(), d_write_indices.data_ptr<int>(), scan_op, initial_value, n);
  CHECK_CUDA_KERNEL_LAUNCH();

  // --- out elements ---
  int h_unique_n = n; // set elements to total_element_num to avoid sync between cpu and gpu.

  // d_unique_pairs_out: The unique out tensor, elements with index >= d_unique_count_out are undefined value.
  // d_unique_indices_out: Original indices of the unique items
  auto d_unique_pairs_out = torch::empty({h_unique_n, 2}, sorted_input_tensor.options());
  auto d_unique_indices_out = torch::empty({h_unique_n}, sorted_input_tensor.options().dtype(torch::kInt32));

  // ---  Pass 3: Gather the unique items and their original indices ---
  if (h_unique_n > 0) {
    int* unique_pairs_out_ptr = d_unique_pairs_out.data_ptr<int>();
    IntPair* d_unique_out_ptr = reinterpret_cast<IntPair*>(unique_pairs_out_ptr);

    gather_uniques<<<blocksPerGrid, threadsPerBlock>>>(
        d_sorted_pairs_ptr, d_flags.data_ptr<int>(), d_write_indices.data_ptr<int>(), d_unique_out_ptr, d_unique_indices_out.data_ptr<int>(), n);
    CHECK_CUDA_KERNEL_LAUNCH();
  }

  return {d_unique_pairs_out, d_unique_indices_out, d_unique_count_out};
}
