#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/device_kernel.h>  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "static_switch.h"
#include "flash.h"
#include "cuda_check.h"
#include "fast_zero_fill_kernel.h"

using namespace cute;

template <typename T_out, uint32_t kBlockM, uint32_t kHeadDim>
void run_fast_zero_fill(Flash_fwd_params& params, cudaStream_t stream) {
  using ArchTag = cutlass::arch::Sm90;
  using ZeroFillKernel = FastZeroFillKernel<T_out, kBlockM, kHeadDim, ArchTag>;

  auto kernel_params = ZeroFillKernel::to_underlying_arguments(
      {static_cast<T_out*>(params.o_ptr),
       {params.total_q, params.d, params.h_qo},
       {params.o_row_stride, _1{}, params.o_head_stride},
       static_cast<float*>(params.softmax_lse_ptr),
       {_1{}, params.total_q}});

  dim3 grid_dims = ZeroFillKernel::get_grid_shape(kernel_params);
  dim3 block_dims = ZeroFillKernel::get_block_shape();

  auto kernel = cutlass::device_kernel<ZeroFillKernel>;
  int smem_size = ZeroFillKernel::SharedStorageSize;
  if (smem_size >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  cutlass::kernel_launch<ZeroFillKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T_out, uint32_t kHeadDim>
void run_fast_zero_fill_(Flash_fwd_params& params, cudaStream_t stream) {
  // TODO: tuning block size
  static constexpr uint32_t kBlockM = 256;
  run_fast_zero_fill<T_out, kBlockM, kHeadDim>(params, stream);
}
