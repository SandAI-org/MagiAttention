// Copyright (c) 2025 SandAI. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cub/device/device_scan.cuh>
#include <vector>

#include "cuda_check.h"
#include "flash.h"


__global__ void tile_idx_to_work_tile(
  
) {
    
}

std::tuple<torch::Tensor> work_tile_divider_ext(
  torch::Tensor q,
  torch::Tensor k,
  torch::Tensor q_ranges,
  torch::Tensor k_ranges
) {
  int const batch_size = q_ranges.size(0);
  int const total_q = q.size(0);
  int const total_k = k.size(0);
  int const num_heads_qo = q.size(1);
  int const num_heads_kv = k.size(1);
  int const head_size = q.size(2);
  
  // Check q, k, v (dtype, device, layout)
  auto q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
  TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  TORCH_CHECK(q.dim() == 3, "query tensor must be a 3D tensor(total_q, num_heads_qo, head_size)");
  TORCH_CHECK(k.dim() == 3, "key tensor must be a 3D tensor(total_k, num_heads_kv, head_size)");
  TORCH_CHECK(v.dim() == 3, "value tensor must be a 3D tensor(total_k, num_heads_kv, head_size)");
  CHECK_SHAPE(q, total_q, num_heads_qo, head_size);
  CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
  CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
  TORCH_CHECK(q.stride(-1) == 1, "query tensor must have contiguous last dimension");
  TORCH_CHECK(k.stride(-1) == 1, "key tensor must have contiguous last dimension");
  TORCH_CHECK(v.stride(-1) == 1, "value tensor must have contiguous last dimension");

  // Check q_ranges (dtype, device, layout)
  TORCH_CHECK(q_ranges.dtype() == torch::kInt32, "q_ranges must have dtype torch.int32");
  CHECK_DEVICE(q_ranges);
  TORCH_CHECK(q_ranges.dim() == 2, "q_ranges must be a 2D tensor");
  TORCH_CHECK(q_ranges.size(1) == 2, "q_ranges must have 2 columns");
  CHECK_SHAPE(q_ranges, batch_size, 2);
  CHECK_CONTIGUOUS(q_ranges);

  // Check k_ranges (dtype, device, layout)
  CHECK_DEVICE(k_ranges);
  TORCH_CHECK(k_ranges.dtype() == torch::kInt32, "k_ranges must have dtype torch.int32");
  TORCH_CHECK(k_ranges.dim() == 2, "k_ranges must be a 2D tensor");
  TORCH_CHECK(k_ranges.size(1) == 2, "k_ranges must have 2 columns");
  CHECK_SHAPE(k_ranges, batch_size, 2);
  CHECK_CONTIGUOUS(k_ranges);

  

  return {q_ranges};
}
