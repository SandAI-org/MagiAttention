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

#include <cute/layout.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/kernel_launch.h>
#include <cutlass/numeric_types.h>

#include "flash.h"
#include "utils.h"

namespace flash {

using namespace cute;

/**
 * Kernel to zero out dK and dV rows that were not covered by the backward pass.
 * This is used when DisableBwdDkvAtomicReduction is true.
 * Threads are parallel in head_dim dimension,
 * so when zeroing a row, threads access contiguous dK/dV elements (stride 1).
 */
template <typename TDkv, uint32_t kBlockN, uint32_t kHeadDim, class ArchTag_>
class FlashAttnBwdDkvPostprocess {
 public:
  using ArchTag = ArchTag_;
  using TileShapeNK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;

  // (total_k, head_dim, num_heads_kv)
  using ShapeDkv = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideDkv = cute::Stride<int64_t, _1, int64_t>;

  static constexpr int SharedStorageSize = kBlockN * sizeof(int);
  static constexpr uint32_t MaxThreadsPerBlock = kHeadDim;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  struct Params {
    TDkv* ptr_dK;
    TDkv* ptr_dV;
    ShapeDkv shape_dkv;
    StrideDkv stride_dkv;
    int2* k_ranges;
    int num_k_ranges;
    int* num_k_ranges_ptr;
  };

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int32_t const n_block = blockIdx.x;
    int32_t const bidh = blockIdx.y;
    int32_t const thread_idx = threadIdx.x;

    int32_t const offset_n = n_block * kBlockN;
    int32_t const total_k = cute::get<0>(params.shape_dkv);
    int64_t const dk_row_stride = cute::get<0>(params.stride_dkv);
    int64_t const dk_head_stride = cute::get<2>(params.stride_dkv);

    // Shared memory for mask: [kBlockN]
    int* const mask_shmem = reinterpret_cast<int*>(smem_buf);
#pragma unroll
    for (int32_t i = thread_idx; i < kBlockN; i += blockDim.x) {
      int32_t const n_idx = offset_n + i;
      int valid = 0;
      if (n_idx < total_k) {
        // Binary search to check if n_idx is in any k_range
        int l = 0;
        int r = (params.num_k_ranges_ptr == nullptr) ? params.num_k_ranges : *params.num_k_ranges_ptr;
        while (l < r) {
          int mid = (l + r) / 2;
          if (params.k_ranges[mid].x <= n_idx) {
            l = mid + 1;
          } else {
            r = mid;
          }
        }
        // l-1 is the potential range index
        if (l > 0) {
          int2 range = params.k_ranges[l - 1];
          if (n_idx < range.y) {
            valid = 1;
          }
        }
      } else {
        valid = 1; // Prevent OOB access for n_idx >= total_k
      }
      mask_shmem[i] = valid;
    }
    __syncthreads();

    // Loop over kBlockN positions; threads parallel in head_dim for coalesced writes
    for (int32_t n = 0; n < kBlockN; ++n) {
      if (mask_shmem[n] != 0)
        continue;
      int32_t const row_idx = offset_n + n;

      // Zero dK and dV for this row: thread tid writes [tid] - contiguous, coalesced
      TDkv* dK_row = params.ptr_dK + row_idx * dk_row_stride + bidh * dk_head_stride;
      TDkv* dV_row = params.ptr_dV + row_idx * dk_row_stride + bidh * dk_head_stride;
      dK_row[thread_idx] = TDkv(0);
      dV_row[thread_idx] = TDkv(0);
    }
  }
};

} // namespace flash
