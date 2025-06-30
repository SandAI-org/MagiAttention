#pragma once

#ifndef USE_NCCL
#define USE_NCCL
#endif

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#include "check.h"

namespace torch::cuda::nccl {

    struct GroupReduceMetaInfo {
        size_t seqlen;
        size_t num_splits;
        size_t repeated_seqlen;
        size_t max_split_size;
        std::vector<int64_t> num_repeats_list;
        std::vector<int64_t> cu_split_size_list;
        std::vector<int64_t> repeated_cu_split_size_list;
        std::vector<int64_t> repeated_recv_buffer_shape;
        
        GroupReduceMetaInfo(
            size_t seqlen,
            size_t num_splits, 
            size_t repeated_seqlen,
            size_t max_split_size,
            std::vector<int64_t> num_repeats_list,
            std::vector<int64_t> cu_split_size_list,
            std::vector<int64_t> repeated_cu_split_size_list,
            std::vector<int64_t> repeated_recv_buffer_shape
        ): 
            seqlen(seqlen),
            num_splits(num_splits),
            repeated_seqlen(repeated_seqlen),
            max_split_size(max_split_size),
            num_repeats_list(std::move(num_repeats_list)),
            cu_split_size_list(std::move(cu_split_size_list)),
            repeated_cu_split_size_list(std::move(repeated_cu_split_size_list)),
            repeated_recv_buffer_shape(std::move(repeated_recv_buffer_shape)) {}
    };


    struct GroupReducePostProcessArgs {
        void* recv_buffer;
        void* repeated_recv_buffer;

        const int64_t* d_split_size_list;
        const int64_t* d_num_repeats_list;
        const int64_t* d_cu_split_size_list;
        const int64_t* d_repeated_cu_split_size_list;

        size_t seqlen;
        size_t repeated_seqlen;
        size_t num_splits;
        size_t max_split_size;
        size_t stride0;

        c10::ScalarType type;
        cudaStream_t stream;

        GroupReducePostProcessArgs(
            void* recv_buffer,
            void* repeated_recv_buffer,
            const int64_t* d_split_size_list,
            const int64_t* d_num_repeats_list,
            const int64_t* d_cu_split_size_list,
            const int64_t* d_repeated_cu_split_size_list,
            size_t seqlen,
            size_t repeated_seqlen,
            size_t num_splits,
            size_t max_split_size,
            size_t stride0,
            c10::ScalarType type,
            cudaStream_t stream
        ): 
            recv_buffer(recv_buffer),
            repeated_recv_buffer(repeated_recv_buffer),
            d_split_size_list(d_split_size_list),
            d_num_repeats_list(d_num_repeats_list),
            d_cu_split_size_list(d_cu_split_size_list),
            d_repeated_cu_split_size_list(d_repeated_cu_split_size_list),
            seqlen(seqlen),
            repeated_seqlen(repeated_seqlen),
            num_splits(num_splits),
            max_split_size(max_split_size),
            stride0(stride0),
            type(type),
            stream(stream) {}
    };


    TORCH_CUDA_CPP_API void group_cast_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& dst_indices_list,
        const std::vector<int64_t>& src_index_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream
    );

    TORCH_CUDA_CPP_API void group_reduce_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        void* repeated_recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<int64_t>& dst_index_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream,
        GroupReducePostProcessArgs& args
    );

    GroupReduceMetaInfo compute_group_reduce_meta_info(
        const c10::IntArrayRef recv_buffer_shape,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        const int repeat_dim = 0
    );

} // namespace torch::cuda::nccl