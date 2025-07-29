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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h> // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "cuda_check.h"
#include "flash.h"
#include "work_tile_divider.hpp"

using namespace cute;

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

  using WorkTileDivider = flash::WorkTileDivider<cutlass::arch::Sm90, 64>;
  typename WorkTileDivider::Arguments divider_args{
    num_heads_qo,
    batch_size,
    static_cast<int*>(q_ranges.data_ptr()),
  };
  typename WorkTileDivider::Params divider_params = WorkTileDivider::to_underlying_arguments(divider_args);
  
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  dim3 grid_dims = WorkTileDivider::get_grid_shape(divider_params);
  dim3 block_dims = WorkTileDivider::get_block_shape();
  cutlass::kernel_launch<WorkTileDivider>(grid_dims, block_dims, 0 /*smem_size*/, stream, divider_params, false /*launch_with_pdl*/);

  return {q_ranges};
}
