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

#include <torch/extension.h>
#include <vector>

// --- CUDA Runtime API Headers ---
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Required Library Headers ---
#include <cub/device/device_scan.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// CUDA错误检查宏
#define CUDA_CHECK(err) { \
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err)); \
}

// ===================================================================
// Data Structure for (int, int) Pairs
// ===================================================================
struct IntPair {
    int x;
    int y;

    __host__ __device__
    bool operator<(const IntPair& other) const {
        if (x < other.x) return true;
        if (x > other.x) return false;
        return y < other.y;
    }
    __host__ __device__
    bool operator!=(const IntPair& other) const {
        return x != other.x || y != other.y;
    }
};

// ===================================================================
// KERNELS (保持不变)
// ===================================================================
__global__ void mark_uniques_kernel(const IntPair* sorted_pairs, int* d_flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool is_unique = (idx == 0) || (sorted_pairs[idx] != sorted_pairs[idx - 1]);
    d_flags[idx] = is_unique ? 1 : 0;
}

__global__ void move_uniques_kernel(const IntPair* sorted_pairs,
                                    const int* d_flags,
                                    const int* d_write_indices,
                                    IntPair* unique_pairs,
                                    int* unique_indices,
                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (d_flags[idx] == 1) {
        int write_idx = d_write_indices[idx];
        unique_pairs[write_idx] = sorted_pairs[idx];
        unique_indices[write_idx] = idx;
    }
}

// ===================================================================
// C++ Extension Main Function (已修正)
// ===================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> find_unique_pairs_ext(torch::Tensor sorted_input_tensor) {
    // 输入验证 (保持不变)
    TORCH_CHECK(sorted_input_tensor.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(sorted_input_tensor.dim() == 2 && sorted_input_tensor.size(1) == 2, "Input must be a [N, 2] tensor");
    TORCH_CHECK(sorted_input_tensor.scalar_type() == torch::kInt32, "Input tensor must be of type int32");
    TORCH_CHECK(sorted_input_tensor.is_contiguous(), "Input tensor must be contiguous");

    int n = sorted_input_tensor.size(0);
    if (n == 0) {
        return {torch::empty({0, 2}, sorted_input_tensor.options()), torch::empty({0}, sorted_input_tensor.options().dtype(torch::kInt32)), torch::empty({0}, sorted_input_tensor.options().dtype(torch::kInt32))};
    }

    // --- 设备内存分配 ---
    auto d_flags = torch::empty({n}, sorted_input_tensor.options().dtype(torch::kInt32));
    auto d_write_indices = torch::empty({n}, sorted_input_tensor.options().dtype(torch::kInt32));
    auto d_unique_count_out = torch::empty({1}, sorted_input_tensor.options().dtype(torch::kInt32));
    // ======================= 关键修正 =======================
    // 1. 获取一个PyTorch认识的、底层的int*指针
    const int* raw_int_ptr = sorted_input_tensor.data_ptr<int>();
    // 2. 使用reinterpret_cast将其转换为我们逻辑上需要的IntPair*指针
    const IntPair* d_sorted_pairs_ptr = reinterpret_cast<const IntPair*>(raw_int_ptr);
    // ==========================================================

    // --- Pass 1: 标记唯一项 ---
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    mark_uniques_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_sorted_pairs_ptr, d_flags.data_ptr<int>(), n);

    // --- 在GPU上计算唯一项总数 (并行归约) ---
    void* d_temp_storage_reduce = nullptr;
    size_t temp_storage_bytes_reduce = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage_reduce, temp_storage_bytes_reduce, d_flags.data_ptr<int>(), d_unique_count_out.data_ptr<int>(), n));
    CUDA_CHECK(cudaMalloc(&d_temp_storage_reduce, temp_storage_bytes_reduce));
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage_reduce, temp_storage_bytes_reduce, d_flags.data_ptr<int>(), d_unique_count_out.data_ptr<int>(), n));

    // --- 使用CUB进行扫描 ---
    void* d_temp_storage_scan = nullptr;
    size_t temp_storage_bytes_scan = 0;
    cub::Sum scan_op;
    int initial_value = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveScan(d_temp_storage_scan, temp_storage_bytes_scan, d_flags.data_ptr<int>(), d_write_indices.data_ptr<int>(), scan_op, initial_value, n));
    CUDA_CHECK(cudaMalloc(&d_temp_storage_scan, temp_storage_bytes_scan));
    CUDA_CHECK(cub::DeviceScan::ExclusiveScan(d_temp_storage_scan, temp_storage_bytes_scan, d_flags.data_ptr<int>(), d_write_indices.data_ptr<int>(), scan_op, initial_value, n));

    // --- 获取唯一项总数 ---
    int h_unique_n = n;

    /*
    if (n > 0) {
        int last_write_idx, last_flag;
        CUDA_CHECK(cudaMemcpy(&last_write_idx, d_write_indices.data_ptr<int>() + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_flag, d_flags.data_ptr<int>() + (n - 1), sizeof(int), cudaMemcpyDeviceToHost));
        h_unique_n = last_write_idx + last_flag;
    } */

    // 创建最终输出张量
    auto d_unique_pairs_out = torch::empty({h_unique_n, 2}, sorted_input_tensor.options());
    auto d_unique_indices_out = torch::empty({h_unique_n}, sorted_input_tensor.options().dtype(torch::kInt32));

    // --- 在GPU上计算唯一项总数 (并行归约) ---
    if (h_unique_n > 0) {
        // --- Pass 2: 移动数据和索引 ---
        int* unique_pairs_out_ptr = d_unique_pairs_out.data_ptr<int>();
        // 2. 使用reinterpret_cast将其转换为我们逻辑上需要的IntPair*指针
        IntPair* d_unique_out_ptr = reinterpret_cast<IntPair*>(unique_pairs_out_ptr);

        move_uniques_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_sorted_pairs_ptr,
            d_flags.data_ptr<int>(),
            d_write_indices.data_ptr<int>(),
            d_unique_out_ptr,
            d_unique_indices_out.data_ptr<int>(),
            n);
    }

    // --- 释放临时内存 ---
    cudaFree(d_temp_storage_scan);

    return {d_unique_pairs_out, d_unique_indices_out, d_unique_count_out};
}

// ===================================================================
// Pybind11 BINDINGS
// ===================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_unique_pairs", &find_unique_pairs_ext, "Finds unique (int, int) pairs from a pre-sorted tensor (CUDA extension)");
}
