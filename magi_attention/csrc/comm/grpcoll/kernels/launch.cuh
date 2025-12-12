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

#include "configs.cuh"
#include "exception.cuh"
#include "reduce_op.cuh"

using namespace magi_attn_comm::grpcoll;

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
// TODO: dig into the effects on kernels by launch attributes
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                     \
  cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
  cudaLaunchAttribute attr[2];                                                \
  attr[0].id = cudaLaunchAttributeCooperative;                                \
  attr[0].val.cooperative = 1;                                                \
  attr[1].id = cudaLaunchAttributeClusterDimension;                           \
  attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1);                      \
  attr[1].val.clusterDim.y = 1;                                               \
  attr[1].val.clusterDim.z = 1;                                               \
  cfg.attrs = attr;                                                           \
  cfg.numAttrs = 2
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
  int __num_sms = (sms);                          \
  int __num_threads = (threads);                  \
  auto __stream = (stream)
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else
#define LAUNCH_KERNEL(config, kernel, ...)                                                \
  do {                                                                                    \
    kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__);                       \
    cudaError_t e = cudaGetLastError();                                                   \
    if (e != cudaSuccess) {                                                               \
      GrpCollException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
      fprintf(stderr, "%s\n", cuda_exception.what());                                     \
      throw cuda_exception;                                                               \
    }                                                                                     \
  } while (0)
#endif
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel)                                                                                   \
  GRPCOLL_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
  cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

#define SWITCH_RANKS(case_macro)                              \
  switch (num_ranks) {                                        \
    case 1:                                                   \
      case_macro(1);                                          \
    case 2:                                                   \
      case_macro(2);                                          \
    case 3:                                                   \
      case_macro(3);                                          \
    case 4:                                                   \
      case_macro(4);                                          \
    case 5:                                                   \
      case_macro(5);                                          \
    case 6:                                                   \
      case_macro(6);                                          \
    case 7:                                                   \
      case_macro(7);                                          \
    case 8:                                                   \
      case_macro(8);                                          \
    default:                                                  \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_ranks"); \
  }                                                           \
  while (false)

#define SWITCH_RANKS_WITH_WARPS(case_macro)                   \
  switch (num_ranks) {                                        \
    case 1:                                                   \
      case_macro(1, 24);                                      \
    case 2:                                                   \
      case_macro(2, 24);                                      \
    case 3:                                                   \
      case_macro(3, 24);                                      \
    case 4:                                                   \
      case_macro(4, 24);                                      \
    case 5:                                                   \
      case_macro(5, 20);                                      \
    case 6:                                                   \
      case_macro(6, 24);                                      \
    case 7:                                                   \
      case_macro(7, 21);                                      \
    case 8:                                                   \
      case_macro(8, 24);                                      \
    default:                                                  \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_ranks"); \
  }                                                           \
  while (false)

#define SWITCH_RDMA_RANKS_WITH_FORWARDER_WARPS(case_macro)         \
  switch (num_ranks / NUM_MAX_NVL_PEERS) {                         \
    case 2:                                                        \
      case_macro(2, 24);                                           \
    case 4:                                                        \
      case_macro(4, 24);                                           \
    case 8:                                                        \
      case_macro(8, 24);                                           \
    case 16:                                                       \
      case_macro(16, 24);                                          \
    case 32:                                                       \
      case_macro(32, 32);                                          \
    default:                                                       \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_rdma_ranks"); \
  }                                                                \
  while (false)

#define SWITCH_RDMA_RANKS(case_macro)                              \
  switch (num_ranks / NUM_MAX_NVL_PEERS) {                         \
    case 2:                                                        \
      case_macro(2);                                               \
    case 4:                                                        \
      case_macro(4);                                               \
    case 8:                                                        \
      case_macro(8);                                               \
    case 16:                                                       \
      case_macro(16);                                              \
    case 32:                                                       \
      case_macro(32);                                              \
    default:                                                       \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_rdma_ranks"); \
  }                                                                \
  while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)            \
  switch (num_ranks) {                                        \
    case 1:                                                   \
      case_macro(dtype, 1);                                   \
    case 2:                                                   \
      case_macro(dtype, 2);                                   \
    case 3:                                                   \
      case_macro(dtype, 3);                                   \
    case 4:                                                   \
      case_macro(dtype, 4);                                   \
    case 5:                                                   \
      case_macro(dtype, 5);                                   \
    case 6:                                                   \
      case_macro(dtype, 6);                                   \
    case 7:                                                   \
      case_macro(dtype, 7);                                   \
    case 8:                                                   \
      case_macro(dtype, 8);                                   \
    default:                                                  \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_ranks"); \
  }                                                           \
  while (false)

#define SWITCH_REDUCE_OPS(case_macro, ...)                    \
  switch (reduce_op) {                                        \
    case ReduceOp::SUM:                                       \
      case_macro(ReduceOp::SUM, ##__VA_ARGS__);               \
    case ReduceOp::AVG:                                       \
      case_macro(ReduceOp::AVG, ##__VA_ARGS__);               \
    case ReduceOp::LSE:                                       \
      case_macro(ReduceOp::LSE, ##__VA_ARGS__);               \
    default:                                                  \
      GRPCOLL_HOST_ASSERT(false and "Unsupported reduce op"); \
  }                                                           \
  while (false)

#define SWITCH_DTYPES(case_macro)                         \
  switch (dtype) {                                        \
    case CUDA_R_16BF:                                     \
      case_macro(nv_bfloat16, float);                     \
    case CUDA_R_16F:                                      \
      case_macro(half, float);                            \
    case CUDA_R_32F:                                      \
      case_macro(float, float);                           \
    case CUDA_R_64F:                                      \
      case_macro(double, double);                         \
    default:                                              \
      GRPCOLL_HOST_ASSERT(false and "Unsupported dtype"); \
  }                                                       \
  while (false)

#define SWITCH_DTYPES_REDUCE_DTYPES(case_macro)           \
  switch (dtype) {                                        \
    case CUDA_R_16BF:                                     \
      case_macro(nv_bfloat16, float);                     \
    case CUDA_R_16F:                                      \
      case_macro(half, float);                            \
    case CUDA_R_32F:                                      \
      case_macro(float, float);                           \
    case CUDA_R_64F:                                      \
      case_macro(double, double);                         \
    default:                                              \
      GRPCOLL_HOST_ASSERT(false and "Unsupported dtype"); \
  }                                                       \
  while (false)

// (dtype, comm_dtype, reduce_dtype)
// which satisfies: `comm_dtype <= dtype <= reduce_dtype`
#define SWITCH_DTYPES_COMM_DTYPES_REDUCE_DTYPES(case_macro, ...)                   \
  switch (dtype) {                                                                 \
    case CUDA_R_16BF:                                                              \
      GRPCOLL_HOST_ASSERT(comm_dtype == CUDA_R_16BF and "Unsupported comm dtype"); \
      case_macro(nv_bfloat16, nv_bfloat16, float, ##__VA_ARGS__);                  \
    case CUDA_R_16F:                                                               \
      GRPCOLL_HOST_ASSERT(comm_dtype == CUDA_R_16F and "Unsupported comm dtype");  \
      case_macro(half, half, float, ##__VA_ARGS__);                                \
    case CUDA_R_32F:                                                               \
      switch (comm_dtype) {                                                        \
        case CUDA_R_16BF:                                                          \
          case_macro(float, nv_bfloat16, float, ##__VA_ARGS__);                    \
        case CUDA_R_16F:                                                           \
          case_macro(float, half, float, ##__VA_ARGS__);                           \
        case CUDA_R_32F:                                                           \
          case_macro(float, float, float, ##__VA_ARGS__);                          \
        default:                                                                   \
          GRPCOLL_HOST_ASSERT(false and "Unsupported comm dtype");                 \
      }                                                                            \
      break;                                                                       \
    case CUDA_R_64F:                                                               \
      GRPCOLL_HOST_ASSERT(comm_dtype == CUDA_R_64F and "Unsupported comm dtype");  \
      case_macro(double, double, double, ##__VA_ARGS__);                           \
    default:                                                                       \
      GRPCOLL_HOST_ASSERT(false and "Unsupported dtype");                          \
  }                                                                                \
  while (false)

// TODO: support other hidden sizes
#define SWITCH_HIDDEN_SIZE(case_macro)                          \
  switch (hidden_size) {                                        \
    case 2048:                                                  \
      case_macro(2048);                                         \
    case 2560:                                                  \
      case_macro(2560);                                         \
    case 4096:                                                  \
      case_macro(4096);                                         \
    case 5120:                                                  \
      case_macro(5120);                                         \
    case 7168:                                                  \
      case_macro(7168);                                         \
    case 8192:                                                  \
      case_macro(8192);                                         \
    default:                                                    \
      GRPCOLL_HOST_ASSERT(false and "Unsupported hidden size"); \
  }                                                             \
  while (false)

#define SWITCH_DATA_GROUPS_3(case_macro, ...)                  \
  switch (num_groups) {                                        \
    case 1:                                                    \
      case_macro(1, ##__VA_ARGS__);                            \
    case 2:                                                    \
      case_macro(2, ##__VA_ARGS__);                            \
    case 3:                                                    \
      case_macro(3, ##__VA_ARGS__);                            \
    default:                                                   \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_groups"); \
  }                                                            \
  while (false)

#define SWITCH_DATA_GROUPS_2(case_macro, ...)                  \
  switch (num_groups) {                                        \
    case 1:                                                    \
      case_macro(1, ##__VA_ARGS__);                            \
    case 2:                                                    \
      case_macro(2, ##__VA_ARGS__);                            \
    default:                                                   \
      GRPCOLL_HOST_ASSERT(false and "Unsupported num_groups"); \
  }                                                            \
  while (false)
