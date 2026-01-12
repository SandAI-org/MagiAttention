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

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>

// ----------------------------------------------------------------------------
// Helper Kernels for Sort
// ----------------------------------------------------------------------------

__global__ void iota_kernel(int* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    data[i] = i;
  }
}

__global__ void extract_keys_kernel(const int* src, int* dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    dst[i] = src[i * 2];
  }
}

__global__ void check_is_sorted_kernel(const int* ranges, int n, int* is_sorted_flag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < n - 1; i += stride) {
    int curr_key = ranges[i * 2];
    int next_key = ranges[(i + 1) * 2];

    if (curr_key > next_key) {
      if (*is_sorted_flag != 0) {
        atomicExch(is_sorted_flag, 0);
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Main Sort Function using CUB
// ----------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor> argsort_ranges(torch::Tensor outer_ranges) {
  TORCH_CHECK(outer_ranges.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(outer_ranges.dim() == 2 && outer_ranges.size(1) == 2, "Input must be [N, 2]");
  TORCH_CHECK(outer_ranges.scalar_type() == torch::kInt32, "Input must be int32");

  int n = outer_ranges.size(0);
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(outer_ranges.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  int threads = 256;
  int blocks = std::min((n + threads - 1) / threads, 4096);

  auto is_sorted_tensor = torch::full({1}, 1, options);

  if (n > 1) {
    check_is_sorted_kernel<<<blocks, threads, 0, stream>>>(outer_ranges.data_ptr<int>(), n, is_sorted_tensor.data_ptr<int>());
  }

  // TODO: to early exit, we should implement RadixSort manually that supports direct exit
  // bool is_ordered = (is_sorted_tensor.item<int>() == 1);

  // // early exit for sorted ranges
  // if (is_ordered) {
  //   return std::make_tuple(torch::Tensor(), true);
  // }

  auto keys_in = torch::empty({n}, options);
  extract_keys_kernel<<<blocks, threads, 0, stream>>>(outer_ranges.data_ptr<int>(), keys_in.data_ptr<int>(), n);

  // argsort using int32
  auto values_in = torch::empty({n}, options);
  auto keys_out = torch::empty({n}, options);
  auto values_out = torch::empty({n}, options);

  iota_kernel<<<blocks, threads, 0, stream>>>(values_in.data_ptr<int>(), n);

  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  int* d_keys_in = keys_in.data_ptr<int>();
  int* d_keys_out = keys_out.data_ptr<int>();
  int* d_values_in = values_in.data_ptr<int>();
  int* d_values_out = values_out.data_ptr<int>();

  // Determine temporary device storage requirements
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 0, sizeof(int) * 8, stream);

  // Allocate temporary storage
  auto temp_storage = torch::empty({(long)temp_storage_bytes}, options.dtype(torch::kInt8));
  d_temp_storage = temp_storage.data_ptr();

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, n, 0, sizeof(int) * 8, stream);

  return std::make_tuple(values_out, is_sorted_tensor);
}

__global__ void reorder_ranges_kernel(
    const int2* __restrict__ outer_src, // [N] of int2 (Source)
    const int2* __restrict__ inner_src, // [N] of int2 (Source)
    const int* __restrict__ attn_src, // [N] of int (Source, optional)
    const int32_t* __restrict__ sorted_idx, // [N] (Indices)
    const int32_t* __restrict__ is_sorted, // [1] (Flag)
    int2* __restrict__ outer_dst, // [N] of int2 (Dest)
    int2* __restrict__ inner_dst, // [N] of int2 (Dest)
    int* __restrict__ attn_dst, // [N] of int (Dest, optional)
    const int n_elements,
    const bool has_attn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_elements) {
    bool sorted = (*is_sorted == 1);
    int32_t src_row = sorted ? i : sorted_idx[i];

    outer_dst[i] = outer_src[src_row];
    inner_dst[i] = inner_src[src_row];

    if (has_attn) {
      attn_dst[i] = attn_src[src_row];
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reorder_ranges_and_attn_type_maps(
    torch::Tensor outer_ranges,
    torch::Tensor inner_ranges,
    torch::optional<torch::Tensor> attn_type_map,
    torch::Tensor sorted_idx,
    torch::Tensor is_sorted) {
  TORCH_CHECK(outer_ranges.is_cuda() && inner_ranges.is_cuda() && sorted_idx.is_cuda(), "All tensors must be on CUDA");

  TORCH_CHECK(outer_ranges.is_contiguous() && inner_ranges.is_contiguous(), "Ranges tensors must be contiguous for vectorized access");

  TORCH_CHECK(outer_ranges.dim() == 2 && outer_ranges.size(1) == 2, "Outer range must be [N, 2]");
  TORCH_CHECK(inner_ranges.dim() == 2 && inner_ranges.size(1) == 2, "Inner range must be [N, 2]");

  TORCH_CHECK(outer_ranges.scalar_type() == torch::kInt32, "Outer ranges must be Int32");
  TORCH_CHECK(inner_ranges.scalar_type() == torch::kInt32, "Inner ranges must be Int32");
  TORCH_CHECK(sorted_idx.scalar_type() == torch::kInt32, "Sorted idx must be Int32 (standard torch argsort output)");
  TORCH_CHECK(is_sorted.size(0) == 1 && is_sorted.scalar_type() == torch::kInt32, "is_sorted must be a single Int32 tensor");

  int n_elements = sorted_idx.size(0);

  auto outer_dst = torch::empty_like(outer_ranges);
  auto inner_dst = torch::empty_like(inner_ranges);

  torch::Tensor attn_dst;
  const int* attn_src_ptr = nullptr;
  int* attn_dst_ptr = nullptr;
  bool has_attn = false;

  if (attn_type_map.has_value() && attn_type_map.value().defined()) {
    const auto& attn_t = attn_type_map.value();
    TORCH_CHECK(attn_t.is_cuda(), "attn_type_map must be on CUDA");
    TORCH_CHECK(attn_t.scalar_type() == torch::kInt32, "attn_type_map must be Int32");

    has_attn = true;
    attn_dst = torch::empty_like(attn_t);
    attn_src_ptr = attn_t.data_ptr<int>();
    attn_dst_ptr = attn_dst.data_ptr<int>();
  } else {
    attn_dst = torch::empty({}, outer_ranges.options());
  }

  const int threads = 256;
  const int blocks = (n_elements + threads - 1) / threads;

  reorder_ranges_kernel<<<blocks, threads>>>(
      reinterpret_cast<int2*>(outer_ranges.data_ptr<int>()),
      reinterpret_cast<int2*>(inner_ranges.data_ptr<int>()),
      attn_src_ptr,
      sorted_idx.data_ptr<int32_t>(),
      is_sorted.data_ptr<int32_t>(),
      reinterpret_cast<int2*>(outer_dst.data_ptr<int>()),
      reinterpret_cast<int2*>(inner_dst.data_ptr<int>()),
      attn_dst_ptr,
      n_elements,
      has_attn);

  return std::make_tuple(outer_dst, inner_dst, attn_dst);
}
