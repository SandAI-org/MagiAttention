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

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
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

// TODO: support other num_ranks
#define SWITCH_RANKS(case_macro)                          \
  switch (num_ranks) {                                    \
    case 2:                                               \
      case_macro(2);                                      \
    case 4:                                               \
      case_macro(4);                                      \
    case 8:                                               \
      case_macro(8);                                      \
    default:                                              \
      GRPCOLL_HOST_ASSERT(false and "Unsupported ranks"); \
  }                                                       \
  while (false)

// TODO: support other RDMA num_ranks
#define SWITCH_RDMA_RANKS(case_macro)                          \
  switch (num_ranks / NUM_MAX_NVL_PEERS) {                     \
    case 2:                                                    \
      case_macro(2);                                           \
    case 4:                                                    \
      case_macro(4);                                           \
    case 8:                                                    \
      case_macro(8);                                           \
    case 16:                                                   \
      case_macro(16);                                          \
    default:                                                   \
      GRPCOLL_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
  }                                                            \
  while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)        \
  switch (num_ranks) {                                    \
    case 2:                                               \
      case_macro(dtype, 2);                               \
    case 4:                                               \
      case_macro(dtype, 4);                               \
    case 8:                                               \
      case_macro(dtype, 8);                               \
    default:                                              \
      GRPCOLL_HOST_ASSERT(false and "Unsupported ranks"); \
  }                                                       \
  while (false)

// TODO: support other dtypes
#define SWITCH_TYPES(case_macro)                         \
  switch (type) {                                        \
    case CUDA_R_16BF:                                    \
      case_macro(nv_bfloat16);                           \
    default:                                             \
      GRPCOLL_HOST_ASSERT(false and "Unsupported type"); \
  }                                                      \
  while (false)

// TODO: support other hidden sizes
#define SWITCH_HIDDEN(case_macro)                          \
  switch (hidden) {                                        \
    case 2048:                                             \
      case_macro(2048);                                    \
    case 2560:                                             \
      case_macro(2560);                                    \
    case 4096:                                             \
      case_macro(4096);                                    \
    case 5120:                                             \
      case_macro(5120);                                    \
    case 7168:                                             \
      case_macro(7168);                                    \
    case 8192:                                             \
      case_macro(8192);                                    \
    default:                                               \
      GRPCOLL_HOST_ASSERT(false and "Unsupported hidden"); \
  }                                                        \
  while (false)
