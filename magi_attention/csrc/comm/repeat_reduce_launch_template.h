#pragma once

#include <assert.h>
#include <stdlib.h>

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "group_collective.cuh"
#include "repeat_reduce_kernel.h"
#include "static_switch.h"


using namespace cute;

template<typename T_out, uint32_t kBlockM, uint32_t kBlockN>
void run_repeat_reduce(
    torch::cuda::nccl::GroupReducePostProcessArgs& args
) {
    using ArchTag = cutlass::arch::Sm90;
    using KernelClass = RepeatReduceKernel<T_out, kBlockM, kBlockN, ArchTag>;

    // unwrap group-reduce post-process args to repeat-reduce kernel args
    auto kernel_args = KernelClass::to_underlying_arguments({
        static_cast<T_out*>(args.recv_buffer), // ptr_O,
        {args.seqlen, args.stride0}, // shape_O: {seqlen, hidden_size},
        {args.stride0, _1{}}, // stride_O: {hidden_size, _1{}},

        static_cast<T_out*>(args.repeated_recv_buffer), // ptr_R,
        {args.repeated_seqlen, args.stride0}, // shape_R: {seqlen_r, hidden_size},
        {args.stride0, _1{}}, // stride_R: {hidden_size, _1{}},
        
        args.d_cu_split_size_list, // cu_split_size_o,
        args.d_split_size_list, // split_size_list,
        args.d_repeated_cu_split_size_list, // cu_split_size_r,
        args.d_num_repeats_list, // num_repeats_list,

        args.num_splits, // num_splits,
        args.max_split_size // max_split_size
    });

    dim3 grid_dims = KernelClass::get_grid_shape(kernel_args);
    dim3 block_dims = KernelClass::get_block_shape();

    auto kernel = cutlass::device_kernel<KernelClass>;
    int smem_size = KernelClass::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<KernelClass>(grid_dims, block_dims, smem_size, args.stream, kernel_args, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();
}


template<typename T_out, uint32_t kBlockN>
void repeat_reduce_cute_kernel(
    torch::cuda::nccl::GroupReducePostProcessArgs& args
) {
    /** TODO: tuning block size */
    static constexpr uint32_t kBlockM = 128;
    run_repeat_reduce<T_out, kBlockM, kBlockN>(args);
}
