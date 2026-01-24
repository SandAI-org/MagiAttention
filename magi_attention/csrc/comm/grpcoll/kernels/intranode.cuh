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

#include "api.cuh"

#include "configs.cuh"
#include "intranode_kernel.cuh"
#include "intranode_notify_kernel.cuh"

namespace magi_attn_comm::grpcoll::intranode {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Cast
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNumDataGroups, int kNumRanks, int kNumWarps>
void launch_group_cast(
    /* 1st group of input / output data*/
    void* recv_x,
    float* recv_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* recv_x_2nd,
    const void* x_2nd,
    /* 3rd group of input / output data*/
    void* recv_x_3rd,
    const void* x_3rd,
    /* other metadata */
    int* recv_src_idx,
    int* recv_channel_offset,
    int* send_head,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    const int64_t* post_perm_idx,
    int num_tokens,
    int hidden_int4,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier) {
  constexpr int kNumThreads = kNumWarps * WARP_SIZE; // num threads per block
  constexpr int kWarpCopyUnrollStages = 5; // warp-copy unroll stages
  constexpr int kNumTMAStages = 4; // num TMA stages
  constexpr int kNumTMABytesPerWarp = 8192; // num bytes of TMA transfer per warp

#ifndef DISABLE_SM90_FEATURES
  GRPCOLL_HOST_ASSERT(hidden_int4 % kNumTMAStages == 0);
  int hidden_bytes_per_stage = (hidden_int4 / kNumTMAStages) * sizeof(int4);
  GRPCOLL_HOST_ASSERT(hidden_bytes_per_stage + /*mbarrier*/ sizeof(uint64_t) <= kNumTMABytesPerWarp); // TMA buffer + mbarrier per warp
  constexpr int smem_size = kNumTMABytesPerWarp * kNumWarps; // shared memory size = num bytes of TMA transfer per block
#endif

  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 && kNumDataGroups <= 3, "Invalid kNumDataGroups");
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  BOOL_SWITCH(num_heads != 0, kCastLSE, [&] {
    BOOL_SWITCH(kernel_barrier.has_value(), kHasKernelBarrier, [&] {
      auto kernel = group_cast_kernel<kNumDataGroups, kNumRanks, kNumThreads, kWarpCopyUnrollStages, kNumTMAStages, kNumTMABytesPerWarp, kCastLSE, kHasKernelBarrier>;

      auto kernel_barrier_view = [&]() {
        if constexpr (kHasKernelBarrier) {
          return kernel_barrier.value().view();
        } else {
          return magi_attn_ext::KernelBarrierView{};
        }
      }();

      SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
      SET_SHARED_MEMORY_FOR_TMA(kernel);

      LAUNCH_KERNEL(
          &cfg,
          kernel,
          reinterpret_cast<int4*>(recv_x),
          recv_lse,
          reinterpret_cast<const int4*>(x),
          lse,
          reinterpret_cast<int4*>(recv_x_2nd),
          reinterpret_cast<const int4*>(x_2nd),
          reinterpret_cast<int4*>(recv_x_3rd),
          reinterpret_cast<const int4*>(x_3rd),
          recv_src_idx,
          recv_channel_offset,
          send_head,
          is_token_in_rank,
          channel_prefix_matrix,
          post_perm_idx,
          num_tokens,
          hidden_int4,
          num_heads,
          buffer_ptrs,
          rank,
          num_max_send_tokens,
          num_recv_buffer_tokens,
          kernel_barrier_view);
    });
  });
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Group Reduce
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename dtype_t, typename comm_dtype_t, typename reduce_dtype_t, int kNumDataGroups, int kNumRanks, int kNumWarps, bool kAccReduce>
void launch_group_reduce(
    /* 1st group of input / output data*/
    void* reduced_x,
    float* reduced_lse,
    const void* x,
    const float* lse,
    /* 2nd group of input / output data*/
    void* reduced_x_2nd,
    const void* x_2nd,
    /* other metadata */
    int* send_head,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    const int64_t* pre_perm_idx,
    int num_reduced_tokens,
    int hidden_size,
    int num_heads,
    void** buffer_ptrs,
    int rank,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens,
    ReduceOp reduce_op,
    std::optional<magi_attn_ext::KernelBarrier>& kernel_barrier) {
  constexpr int kNumThreads = kNumWarps * WARP_SIZE; // num threads per block
  constexpr int kMaxNumHeads = 128; // the maximum number of heads supported for lse
  constexpr int kWarpCopyUnrollStages = 4; // warp-copy unroll stages
  constexpr int kCommDtypePerDtype = sizeof(dtype_t) / sizeof(comm_dtype_t); // num comm_dtype_t elems per dtype_t
  GRPCOLL_STATIC_ASSERT(kCommDtypePerDtype == 1 || kCommDtypePerDtype == 2 || kCommDtypePerDtype == 4, "Invalid (dtype_t, comm_dtype_t)");
  constexpr int kNumTMAStages = 8 / kCommDtypePerDtype; // num TMA stages
  constexpr int kNumTMABytesPerWarp = 4096; // num bytes of TMA transfer per warp

#ifndef DISABLE_SM90_FEATURES
  GRPCOLL_STATIC_ASSERT(kNumTMAStages * WARP_SIZE * kCommDtypePerDtype * sizeof(int4) <= kNumTMABytesPerWarp, "Invalid TMA buffer count"); // TMA buffer per warp
  constexpr int smem_size = kNumTMABytesPerWarp * kNumWarps; // shared memory size = num bytes of TMA transfer per block
#endif

  // NOTE: when `kReduceOp != ReduceOp::LSE`,
  // num_heads should be 0 to let `lse_buffers` empty
  if (reduce_op != ReduceOp::LSE) {
    GRPCOLL_HOST_ASSERT(num_heads == 0);
  } else {
    GRPCOLL_HOST_ASSERT(num_heads <= kMaxNumHeads);
  }

  GRPCOLL_STATIC_ASSERT(kNumDataGroups >= 1 && kNumDataGroups <= 2, "Invalid kNumDataGroups");

  // Even-numbered SMs for sending, odd-numbered SMs for receiving
  GRPCOLL_HOST_ASSERT(num_sms % 2 == 0);

  BOOL_SWITCH(kernel_barrier.has_value(), kHasKernelBarrier, [&] {
    magi_attn_ext::KernelBarrierView kernel_barrier_view = kHasKernelBarrier ? kernel_barrier.value().view() : magi_attn_ext::KernelBarrierView{};

    REDUCE_OP_SWITCH(reduce_op, kReduceOp, [&] {
      auto kernel = group_reduce_kernel<
          dtype_t,
          comm_dtype_t,
          reduce_dtype_t,
          kReduceOp,
          kNumDataGroups,
          kNumRanks,
          kNumThreads,
          kWarpCopyUnrollStages,
          kNumTMAStages,
          kNumTMABytesPerWarp,
          kMaxNumHeads,
          kAccReduce,
          kHasKernelBarrier>;

      SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
      SET_SHARED_MEMORY_FOR_TMA(kernel);

      LAUNCH_KERNEL(
          &cfg,
          kernel,
          reinterpret_cast<dtype_t*>(reduced_x),
          reduced_lse,
          reinterpret_cast<const dtype_t*>(x),
          lse,
          reinterpret_cast<dtype_t*>(reduced_x_2nd),
          reinterpret_cast<const dtype_t*>(x_2nd),
          send_head,
          src_idx,
          rank_prefix_matrix,
          channel_prefix_matrix,
          pre_perm_idx,
          num_reduced_tokens,
          hidden_size,
          num_heads,
          buffer_ptrs,
          rank,
          num_max_send_tokens,
          num_recv_buffer_tokens,
          kernel_barrier_view);
    });
  });
}

} // namespace magi_attn_comm::grpcoll::intranode
