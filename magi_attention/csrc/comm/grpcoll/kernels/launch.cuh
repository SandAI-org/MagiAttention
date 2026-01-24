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

#define RANKS_SWITCH(NUM_RANKS, CONST_NAME, ...)               \
  [&] {                                                        \
    switch (NUM_RANKS) {                                       \
      case 1: {                                                \
        constexpr static int CONST_NAME = 1;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 2: {                                                \
        constexpr static int CONST_NAME = 2;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 3: {                                                \
        constexpr static int CONST_NAME = 3;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 4: {                                                \
        constexpr static int CONST_NAME = 4;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 5: {                                                \
        constexpr static int CONST_NAME = 5;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 6: {                                                \
        constexpr static int CONST_NAME = 6;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 7: {                                                \
        constexpr static int CONST_NAME = 7;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      case 8: {                                                \
        constexpr static int CONST_NAME = 8;                   \
        return __VA_ARGS__();                                  \
      }                                                        \
      default:                                                 \
        GRPCOLL_HOST_ASSERT(false && "Unsupported num_ranks"); \
    }                                                          \
  }()

#define RANKS_WITH_WARPS_SWITCH(NUM_RANKS, RANK_CONST, WARP_CONST, ...) \
  [&] {                                                                 \
    switch (NUM_RANKS) {                                                \
      case 1: {                                                         \
        constexpr static int RANK_CONST = 1;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 2: {                                                         \
        constexpr static int RANK_CONST = 2;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 3: {                                                         \
        constexpr static int RANK_CONST = 3;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 4: {                                                         \
        constexpr static int RANK_CONST = 4;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 5: {                                                         \
        constexpr static int RANK_CONST = 5;                            \
        constexpr static int WARP_CONST = 20;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 6: {                                                         \
        constexpr static int RANK_CONST = 6;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 7: {                                                         \
        constexpr static int RANK_CONST = 7;                            \
        constexpr static int WARP_CONST = 21;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case 8: {                                                         \
        constexpr static int RANK_CONST = 8;                            \
        constexpr static int WARP_CONST = 24;                           \
        return __VA_ARGS__();                                           \
      }                                                                 \
      default:                                                          \
        GRPCOLL_HOST_ASSERT(false && "Unsupported num_ranks");          \
    }                                                                   \
  }()

#define RDMA_RANKS_WITH_FORWARDER_WARPS_SWITCH(NUM_RANKS, RANK_CONST, WARP_CONST, ...) \
  [&] {                                                                                \
    switch (NUM_RANKS) {                                                               \
      case 2: {                                                                        \
        constexpr static int RANK_CONST = 2;                                           \
        constexpr static int WARP_CONST = 24;                                          \
        return __VA_ARGS__();                                                          \
      }                                                                                \
      case 4: {                                                                        \
        constexpr static int RANK_CONST = 4;                                           \
        constexpr static int WARP_CONST = 24;                                          \
        return __VA_ARGS__();                                                          \
      }                                                                                \
      case 8: {                                                                        \
        constexpr static int RANK_CONST = 8;                                           \
        constexpr static int WARP_CONST = 24;                                          \
        return __VA_ARGS__();                                                          \
      }                                                                                \
      case 16: {                                                                       \
        constexpr static int RANK_CONST = 16;                                          \
        constexpr static int WARP_CONST = 24;                                          \
        return __VA_ARGS__();                                                          \
      }                                                                                \
      case 32: {                                                                       \
        constexpr static int RANK_CONST = 32;                                          \
        constexpr static int WARP_CONST = 32;                                          \
        return __VA_ARGS__();                                                          \
      }                                                                                \
      default:                                                                         \
        GRPCOLL_HOST_ASSERT(false && "Unsupported num_rdma_ranks");                    \
    }                                                                                  \
  }()

#define RDMA_RANKS_SWITCH(NUM_RANKS, CONST_NAME, ...)               \
  [&] {                                                             \
    switch (NUM_RANKS) {                                            \
      case 2: {                                                     \
        constexpr static int CONST_NAME = 2;                        \
        return __VA_ARGS__();                                       \
      }                                                             \
      case 4: {                                                     \
        constexpr static int CONST_NAME = 4;                        \
        return __VA_ARGS__();                                       \
      }                                                             \
      case 8: {                                                     \
        constexpr static int CONST_NAME = 8;                        \
        return __VA_ARGS__();                                       \
      }                                                             \
      case 16: {                                                    \
        constexpr static int CONST_NAME = 16;                       \
        return __VA_ARGS__();                                       \
      }                                                             \
      case 32: {                                                    \
        constexpr static int CONST_NAME = 32;                       \
        return __VA_ARGS__();                                       \
      }                                                             \
      default:                                                      \
        GRPCOLL_HOST_ASSERT(false && "Unsupported num_rdma_ranks"); \
    }                                                               \
  }()

#define REDUCE_OP_SWITCH(REDUCE_OP, CONST_OP, ...)           \
  [&] {                                                      \
    if (REDUCE_OP == ReduceOp::SUM) {                        \
      constexpr static ReduceOp CONST_OP = ReduceOp::SUM;    \
      return __VA_ARGS__();                                  \
    } else if (REDUCE_OP == ReduceOp::AVG) {                 \
      constexpr static ReduceOp CONST_OP = ReduceOp::AVG;    \
      return __VA_ARGS__();                                  \
    } else if (REDUCE_OP == ReduceOp::LSE) {                 \
      constexpr static ReduceOp CONST_OP = ReduceOp::LSE;    \
      return __VA_ARGS__();                                  \
    } else {                                                 \
      GRPCOLL_HOST_ASSERT(false && "Unsupported reduce op"); \
    }                                                        \
  }()

#define DTYPE_SWITCH_IMPL(DTYPE, T, T_ACC, ...)        \
  [&] {                                                \
    if (DTYPE == CUDA_R_16BF) {                        \
      using T = nv_bfloat16;                           \
      using T_ACC = float;                             \
      return __VA_ARGS__();                            \
    }                                                  \
    if (DTYPE == CUDA_R_16F) {                         \
      using T = half;                                  \
      using T_ACC = float;                             \
      return __VA_ARGS__();                            \
    }                                                  \
    if (DTYPE == CUDA_R_32F) {                         \
      using T = float;                                 \
      using T_ACC = float;                             \
      return __VA_ARGS__();                            \
    }                                                  \
    if (DTYPE == CUDA_R_64F) {                         \
      using T = double;                                \
      using T_ACC = double;                            \
      return __VA_ARGS__();                            \
    }                                                  \
    GRPCOLL_HOST_ASSERT(false && "Unsupported dtype"); \
  }()

#define DTYPE_COMM_DTYPE_REDUCE_DTYPE_SWITCH(DTYPE, COMM_DTYPE, T, T_COMM, T_REDUCE, ...) \
  [&] {                                                                                   \
    if (DTYPE == CUDA_R_16BF) {                                                           \
      GRPCOLL_HOST_ASSERT(COMM_DTYPE == CUDA_R_16BF && "Unsupported comm dtype");         \
      using T = nv_bfloat16;                                                              \
      using T_COMM = nv_bfloat16;                                                         \
      using T_REDUCE = float;                                                             \
      return __VA_ARGS__();                                                               \
    } else if (DTYPE == CUDA_R_16F) {                                                     \
      GRPCOLL_HOST_ASSERT(COMM_DTYPE == CUDA_R_16F && "Unsupported comm dtype");          \
      using T = half;                                                                     \
      using T_COMM = half;                                                                \
      using T_REDUCE = float;                                                             \
      return __VA_ARGS__();                                                               \
    } else if (DTYPE == CUDA_R_32F) {                                                     \
      using T = float;                                                                    \
      using T_REDUCE = float;                                                             \
      if (COMM_DTYPE == CUDA_R_16BF) {                                                    \
        using T_COMM = nv_bfloat16;                                                       \
        return __VA_ARGS__();                                                             \
      }                                                                                   \
      if (COMM_DTYPE == CUDA_R_16F) {                                                     \
        using T_COMM = half;                                                              \
        return __VA_ARGS__();                                                             \
      }                                                                                   \
      if (COMM_DTYPE == CUDA_R_32F) {                                                     \
        using T_COMM = float;                                                             \
        return __VA_ARGS__();                                                             \
      }                                                                                   \
      GRPCOLL_HOST_ASSERT(false && "Unsupported comm dtype");                             \
    } else if (DTYPE == CUDA_R_64F) {                                                     \
      GRPCOLL_HOST_ASSERT(COMM_DTYPE == CUDA_R_64F && "Unsupported comm dtype");          \
      using T = double;                                                                   \
      using T_COMM = double;                                                              \
      using T_REDUCE = double;                                                            \
      return __VA_ARGS__();                                                               \
    } else {                                                                              \
      GRPCOLL_HOST_ASSERT(false && "Unsupported dtype");                                  \
    }                                                                                     \
  }()

// TODO: support other hidden sizes
#define HIDDEN_SIZE_SWITCH(HIDDEN_SIZE, CONST_NAME, ...)         \
  [&] {                                                          \
    switch (HIDDEN_SIZE) {                                       \
      case 2048: {                                               \
        constexpr static int CONST_NAME = 2048;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case 2560: {                                               \
        constexpr static int CONST_NAME = 2560;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case 4096: {                                               \
        constexpr static int CONST_NAME = 4096;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case 5120: {                                               \
        constexpr static int CONST_NAME = 5120;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case 7168: {                                               \
        constexpr static int CONST_NAME = 7168;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      case 8192: {                                               \
        constexpr static int CONST_NAME = 8192;                  \
        return __VA_ARGS__();                                    \
      }                                                          \
      default:                                                   \
        GRPCOLL_HOST_ASSERT(false && "Unsupported hidden size"); \
    }                                                            \
  }()

#define DATA_GROUPS_MAX2_SWITCH(NUM_GROUPS, CONST_NAME, ...) \
  [&] {                                                      \
    switch (NUM_GROUPS) {                                    \
      case 1: {                                              \
        constexpr static int CONST_NAME = 1;                 \
        return __VA_ARGS__();                                \
        break;                                               \
      }                                                      \
      case 2: {                                              \
        constexpr static int CONST_NAME = 2;                 \
        return __VA_ARGS__();                                \
        break;                                               \
      }                                                      \
      default:                                               \
        break;                                               \
    }                                                        \
    GRPCOLL_HOST_ASSERT(false && "Unsupported num_groups");  \
  }()

#define DATA_GROUPS_MAX3_SWITCH(NUM_GROUPS, CONST_NAME, ...) \
  [&] {                                                      \
    switch (NUM_GROUPS) {                                    \
      case 1: {                                              \
        constexpr static int CONST_NAME = 1;                 \
        return __VA_ARGS__();                                \
        break;                                               \
      }                                                      \
      case 2: {                                              \
        constexpr static int CONST_NAME = 2;                 \
        return __VA_ARGS__();                                \
        break;                                               \
      }                                                      \
      case 3: {                                              \
        constexpr static int CONST_NAME = 3;                 \
        return __VA_ARGS__();                                \
        break;                                               \
      }                                                      \
      default:                                               \
        break;                                               \
    }                                                        \
    GRPCOLL_HOST_ASSERT(false && "Unsupported num_groups");  \
  }()

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()
