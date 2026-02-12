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

#pragma once

#include <stdexcept>

#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "cuda_check.h"
#include "flash.h"
#include "flash_bwd_postprocess_kernel.h"
#include "static_switch.h"

using namespace cute;

template <typename TDkv, uint32_t kBlockN, uint32_t kHeadDim, class ArchTag>
void run_flash_bwd_dkv_postprocess(Flash_bwd_params& params, cudaStream_t stream) {
  using PostprocessKernel = flash::FlashAttnBwdDkvPostprocess<TDkv, kBlockN, kHeadDim, ArchTag>;

  typename PostprocessKernel::Params p;
  p.ptr_dK = static_cast<TDkv*>(params.dk_ptr);
  p.ptr_dV = static_cast<TDkv*>(params.dv_ptr);
  p.shape_dkv = make_shape((int32_t)params.total_k, (int32_t)params.d, (int32_t)params.h_kv);
  p.stride_dkv = make_stride((int64_t)params.dk_row_stride, _1{}, (int64_t)params.dk_head_stride);
  // use bwd_unique_count when AutoRangeMerge is true
  if (params.bwd_unique_count != nullptr) {
    p.k_ranges = params.merge_k_ranges;
    p.num_k_ranges = 0;
    p.num_k_ranges_ptr = params.bwd_unique_count;
  } else {
    p.k_ranges = params.k_ranges;
    p.num_k_ranges = params.b;
    p.num_k_ranges_ptr = nullptr;
  }

  dim3 grid(cute::ceil_div(params.total_k, kBlockN), params.h_kv, 1);
  dim3 block(kHeadDim, 1, 1);

  int smem_size = PostprocessKernel::SharedStorageSize;
  cutlass::device_kernel<PostprocessKernel><<<grid, block, smem_size, stream>>>(p);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename TDkv, uint32_t kHeadDim>
void run_flash_bwd_dkv_postprocess_(Flash_bwd_params& params, cudaStream_t stream) {
  static constexpr uint32_t kBlockN = 32;
  using ArchTag = cutlass::arch::Sm90;
  run_flash_bwd_dkv_postprocess<TDkv, kBlockN, kHeadDim, ArchTag>(params, stream);
}
