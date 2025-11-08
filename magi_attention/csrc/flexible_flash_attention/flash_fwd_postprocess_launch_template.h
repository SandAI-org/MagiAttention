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

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h> // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "cuda_check.h"
#include "flash.h"
#include "flash_fwd_postprocess_kernel.h"
#include "static_switch.h"

using namespace cute;

template <typename T_out, uint32_t kBlockM, uint32_t kHeadDim, bool Has_sink>
void run_flash_fwd_post_process(Flash_fwd_params& params, cudaStream_t stream) {
  using ArchTag = cutlass::arch::Sm90;
  using PostprocessKernel = flash::FlashAttnFwdPostprocess<T_out, kBlockM, kHeadDim, Has_sink, ArchTag>;

  auto kernel_params = PostprocessKernel::to_underlying_arguments(
      {// O
       static_cast<T_out*>(params.o_ptr),
       {params.total_q, params.d, params.h_qo}, // shape_O: [sq, hd, nhq]
       {params.o_row_stride, _1{}, params.o_head_stride}, // stride_O: [nhq*hd, 1, hd]
                                                          // LSE
       static_cast<float*>(params.softmax_lse_ptr),
       {params.h_qo, _1{}}, // stride_LSE: [nhq, 1]
                            // sink
       static_cast<float*>(params.sink_ptr),
       {params.total_sink, params.h_qo}, // shape_sink: [s_sink, nhq]
       {params.h_qo, _1{}}}); // stride_sink: [nhq, 1]

  dim3 grid_dims = PostprocessKernel::get_grid_shape(kernel_params);
  dim3 block_dims = PostprocessKernel::get_block_shape();

  auto kernel = cutlass::device_kernel<PostprocessKernel>;
  int smem_size = PostprocessKernel::SharedStorageSize;
  if (smem_size >= 48 * 1024) { // over the limitation (48KB) of static shared memory of H100
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  cutlass::kernel_launch<PostprocessKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, /*launch_with_pdl=*/false);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T_out, uint32_t kHeadDim>
void run_flash_fwd_post_process_(Flash_fwd_params& params, cudaStream_t stream) {
  // TODO: tuning block size
  static constexpr uint32_t kBlockM = 256;
  BOOL_SWITCH(params.has_sink(), Has_sink, [&] { run_flash_fwd_post_process<T_out, kBlockM, kHeadDim, Has_sink>(params, stream); });
}
