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

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>

#include <cutlass/numeric_types.h>

#include "named_barrier.hpp"

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Softmax Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void thread_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
  for (int ni = 0; ni < size<1>(tensor); ni++) {
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
      summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
    }
  }
}

template <typename Engine0, typename Layout0, typename Operator>
__device__ __forceinline__ void warp_reduce_column_(Tensor<Engine0, Layout0>& tensor, Operator& op) {
// reduce column registers of mma accumulator layout in a warp
// after reduce, each thread has same result
#pragma unroll
  for (int ni = 0; ni < size<0>(tensor); ni++) {
#pragma unroll
    for (int offset = 4; offset <= 16; offset <<= 1) {
      tensor(ni) = op(tensor(ni), __shfl_xor_sync(uint32_t(-1), tensor(ni), offset));
    }
  }
}

template <int NumMmaWarpGroups, int kNRows, typename Engine0, typename Layout0, typename Operator>
__device__ __forceinline__ void warp_group_reduce_column_(Tensor<Engine0, Layout0>& tensor, Operator& op) {
  // reduce column registers of mma accumulator layout in a warp group
  // each thread have kNRows column, each row have 4 thread
  __shared__ float shared_result[NumMmaWarpGroups][4][kNRows * 4];
  // Get the current mma warp group index
  // -1 is because one warp group is the producer
  int const curr_WG = flash::canonical_warp_group_idx_nosync() - 1;
  int lane = threadIdx.x % 32;
  int warp_id = (threadIdx.x >> 5) & 0x3;
  // only use 4 threads per warp, to avoid tensor(runtime_index) access and use local memory
  if (lane < 4) {
#pragma unroll
    for (int i = 0; i < kNRows; i++) {
      // shmem index is the column id of mma result
      const int shmem_index = kNRows < 4 ? ((i >> 1) << 3) + (i & 0x1) + (lane << 1) : ((i >> 2) << 4) + (i & 0x3) + (lane << 2);
      shared_result[curr_WG][warp_id][shmem_index] = tensor(i);
    }
  }
  // Ensure that the shmem writes of the current thread
  // are visible to other threads within the same warp group
  __threadfence_block();
  // Sync on the current mma warp group's named barrier, wait for write
  cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpGroupSwapAB1) + curr_WG /*id*/);
#pragma unroll
  for (int ni = 0; ni < size<0>(tensor); ni++) {
    // global index is the column id of register in mma result
    int global_index = kNRows < 4 ? ((ni >> 1) << 3) | (ni & 0x1) | ((threadIdx.x & 0x3) << 1) : ((ni >> 2) << 4) | (ni & 0x3) | ((threadIdx.x & 0x3) << 2);
    // ver0
    // tensor(ni) =
    //     op(op(shared_result[curr_WG][0][global_index], shared_result[curr_WG][1][global_index]),
    //        op(shared_result[curr_WG][2][global_index], shared_result[curr_WG][3][global_index]));
    // ver1
    tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 1][global_index]);
    tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 2][global_index]);
    tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 3][global_index]);
    // ver2
    // if (lane < 4) {
    //   tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 1][global_index]);
    //   tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 2][global_index]);
    //   tensor(ni) = op(tensor(ni), shared_result[curr_WG][warp_id ^ 3][global_index]);
    // }
    // tensor(ni) = __shfl_sync(0xffffffff, tensor(ni), (lane & 0x3));
  }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void quad_allreduce_(Tensor<Engine0, Layout0>& dst, Tensor<Engine1, Layout1>& src, Operator& op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
  for (int i = 0; i < size(dst); i++) {
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& summary, Operator& op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template <bool zero_init = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& max) {
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}

template <bool zero_init = true, bool warp_reduce = true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1>& sum) {
  SumOp<float> sum_op;
  thread_reduce_<zero_init>(tensor, sum, sum_op);
  if constexpr (warp_reduce) {
    quad_allreduce_(sum, sum, sum_op);
  }
}

// Apply the exp to all the elements.
template <bool Scale_max = true, bool Check_inf = true, int Max_offset = 0, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
CUTLASS_DEVICE void scale_apply_exp2(Tensor<Engine0, Layout0>& tensor, Tensor<Engine1, Layout1> const& max, const float scale) {
  // For FP8, we can subtract max by 8.0 so that the value after exp2 is in the range of [0, 256].
  // This lets us use more of the FP8 range (instead of just [0, 1]) to reduce underflow.
  static constexpr float max_offset = float(Max_offset); // We can only template on int, not float
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    const float max_scaled =
        Check_inf ? (max(mi) == -INFINITY ? 0.f : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset) : (!Scale_max ? max(mi) : max(mi) * scale) - max_offset;
#pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni) {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)). This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
    }
  }
}

template <typename Element>
CUTLASS_DEVICE Element calc_lse(Element* x, size_t n) {
  static_assert(std::is_same_v<Element, float>, "Only support float");

  if (n == 0) {
    return -INFINITY;
  }

  Element lse = 0;
#pragma unroll
  for (int i = 0; i < n; ++i) {
    lse += expf(x[i]);
  }
  lse = logf(lse);

  return lse;
}

template <typename Element>
CUTLASS_DEVICE Element safe_sub(Element a, Element b) {
  if (a == -INFINITY && b == -INFINITY) {
    // (-Inf) - (-Inf) will result in NaN
    // but we expect it to be -Inf
    return -INFINITY;
  }
  return a - b;
}

template <typename Element>
CUTLASS_DEVICE Element softplus(Element x) {
  static_assert(std::is_same_v<Element, float>, "Only support float");
  return logf(1.f + expf(x));
}

template <typename Element>
CUTLASS_DEVICE Element correct_lse(Element lse1, Element lse2) {
  Element max_lse = max(lse1, lse2);
  Element min_lse = min(lse1, lse2);
  Element lse = max_lse + softplus(safe_sub(min_lse, max_lse));
  return lse;
}

template <typename Element>
CUTLASS_DEVICE Element calc_lse_rescale_weight(Element lse_to_rescale, Element rescaled_lse) {
  static_assert(std::is_same_v<Element, float>, "Only support float");
  Element rescale_weight = expf(safe_sub(lse_to_rescale, rescaled_lse));
  return rescale_weight;
}

template <typename Element>
CUTLASS_DEVICE Element safe_softmax(Element x, Element lse) {
  static_assert(std::is_same_v<Element, float>, "Only support float");
  Element softmax_x = expf(safe_sub(x, lse));
  return softmax_x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Softmax
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows, int Max_offset = 0, bool SwapAB = false>
struct Softmax {
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;
  float const softmax_scale_log2;

  CUTLASS_DEVICE Softmax(float const softmax_scale_log2_) : softmax_scale_log2(softmax_scale_log2_) {};

  template <bool Is_first, bool Check_inf = false, int NumMmaWarpGroups = 1, typename Tensor0>
  __forceinline__ __device__ TensorT max_get_scale(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(acc_s.layout()));
    static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
    TensorT scores_scale;
    if constexpr (Is_first) {
      if constexpr (!SwapAB) {
        flash::template reduce_max</*zero_init=*/true>(scores, row_max);
        cute::fill(scores_scale, 1.f);
      } else {
        MaxOp<float> max_op;
        thread_reduce_</*zero_init=*/true>(scores, row_max, max_op);
        warp_reduce_column_(row_max, max_op);
        warp_group_reduce_column_<NumMmaWarpGroups, kNRows>(row_max, max_op);
        cute::fill(scores_scale, 1.f);
      }
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      if constexpr (!SwapAB) {
        flash::template reduce_max</*zero_init=*/false>(scores, row_max);
      } else {
        // reduce max in column
        MaxOp<float> max_op;
        thread_reduce_</*zero_init=*/false>(scores, row_max, max_op);
        warp_reduce_column_(row_max, max_op);
        warp_group_reduce_column_<NumMmaWarpGroups, kNRows>(row_max, max_op);
      }
#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = !Check_inf ? row_max(mi) : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
        scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale(mi);
      }
    }
    return scores_scale;
  };

  template <bool Is_first, bool Check_inf = false, typename Tensor0>
  __forceinline__ __device__ void online_softmax(Tensor0& acc_s) {
    // Reshape acc_s from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(acc_s.layout()));
    static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
    flash::template scale_apply_exp2</*Scale_max=*/true, Check_inf, Max_offset>(scores, row_max, softmax_scale_log2);
    // We don't do the reduce across threads here since we don't need to use the row_sum.
    // We do that reduce at the end when we need to normalize the softmax.
    if constexpr (!SwapAB) {
      flash::reduce_sum</*zero_init=*/Is_first, /*warp_reduce=*/false>(scores, row_sum);
    } else {
      SumOp<float> sum_op;
      thread_reduce_</*zero_init=*/Is_first>(scores, row_sum, sum_op);
    }
  };

  template <int NumMmaWarpGroups = 1>
  __forceinline__ __device__ TensorT finalize(float const final_scale = 1.f) {
    SumOp<float> sum_op;
    if constexpr (!SwapAB) {
      quad_allreduce_(row_sum, row_sum, sum_op);
    } else {
      // reduce sum in column
      warp_reduce_column_(row_sum, sum_op);
      warp_group_reduce_column_<NumMmaWarpGroups, kNRows>(row_sum, sum_op);
    }
    TensorT scores_scale;
#pragma unroll
    for (int mi = 0; mi < size(row_sum); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
      scores_scale(mi) = inv_sum * final_scale;
      // For FP8, we might have scaled the output of exp by 2**8 so we need to divide sum by that amount.
      if constexpr (Max_offset != 0) {
        static constexpr float sum_scale = 1.f / float(1 << Max_offset);
        sum *= sum_scale;
      }
      row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
    }
    return scores_scale;
  };

  template <typename Tensor1>
  __forceinline__ __device__ void rescale_o(Tensor1& acc_o, TensorT const& scores_scale) {
    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(acc_o.layout()));
    static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
#pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
#pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale(mi);
      }
    }
  };
};

} // namespace flash
