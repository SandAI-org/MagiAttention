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

/**********************************************************************************
 * Copyright (c) 2025 DeepSeek. All Rights Reserved.
 *
 * Licensed under the MIT License.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *********************************************************************************/

#pragma once

#define WARP_SIZE 32
#define NUM_MAX_BARRIERS 16

#define NUM_MAX_NVL_PEERS 8
// Some warp roles require at least `kNumRDMARank` warps, such as the group reduce forwarder
// thus too large number of RDMA peers may cause the number of warps to exceed the limit.
// See issue: https://github.com/deepseek-ai/DeepEP/issues/237
#define NUM_MAX_RDMA_PEERS 20
#define NUM_MAX_PEERS (NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS)

#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_BUFFER_ALIGNMENT_BYTES 128

#define FINISHED_SUM_TAG 1024
#define NUM_WAIT_NANOSECONDS 500

#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__ // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef DISABLE_SM90_FEATURES
#include <cuda_fp8.h>
#else
// Ampere does not support FP8 features
#define __NV_E4M3 0
#define __NV_E5M2 1
typedef int __nv_fp8_interpretation_t;
typedef int __nv_fp8x4_e4m3;
typedef uint8_t __nv_fp8_storage_t;
#endif

#ifndef DISABLE_NVSHMEM
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <infiniband/mlx5dv.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#endif
