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

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cassert>
#include <cstdint>
#include <cstdlib>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cute/tensor.hpp>

#include "cuda_check.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

// A wrapper for the kernel that is used to guard against compilation on
// architectures that will never use the kernel. The purpose of this is to
// reduce the size of the compiled binary.
// Adapted from
// https://github.com/vllm-project/vllm/blob/4d29e91be84d27ca313d657eee92c067439a4c23/csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cuh#L55
template <typename Kernel>
struct enable_sm90_or_later : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

template <typename Kernel>
struct enable_sm80_to_sm89 : Kernel {
  template <typename... Args>
  CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ <= 890)
    Kernel::operator()(std::forward<Args>(args)...);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const& x, float const& y) {
    return max(x, y);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x + y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// For SM90, convert acc_layout from ((2, 2, V), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
template <bool Transposed = false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) { // SM90
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = acc_layout;
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)));
    } else {
      return make_layout(make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
    }

  } else { // SM80
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{}); // ((2, 2), MMA_M, MMA_N)
    if constexpr (!Transposed) {
      return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    } else {
      return make_layout(make_layout(get<0, 0>(l), get<2>(l)), make_layout(get<0, 1>(l), get<1>(l)));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
// For SM90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((2, 2, 2), MMA_M, (N / 16, MMA_N))
// For SM90, FP8, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
template <typename MMA_Traits, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
  using X = Underscore;
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) { // SM90
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
      auto l = logical_divide(get<0, 2>(acc_layout), Tile<_2>{}); // ((2, N / 16))
      return make_layout(
          make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout), get<0, 0>(l)), get<1>(acc_layout), coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
    } else {
      static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
      static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
      static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
      auto l = logical_divide(get<0, 2>(acc_layout), Tile<Layout<Shape<_2, _2>>>{}); // (((2, 2), N / 32))
      // This combines the first two modes (<0, 0> and <0, 1>) into one mode.
      // Will require register shuffling later to be correct.
      return make_layout(
          make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
          get<1>(acc_layout),
          coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout)))); // ((4, 2, 2), MMA_M, N / 32 * MMA_N)
      // This combination is right but doesn't work with register shuffling.
      // return make_layout(make_layout(coalesce(make_layout(get<0, 0>(acc_layout), get<0, 0, 0>(l))), get<0,
      // 1>(acc_layout), get<0, 0, 1>(l)),
      //                    get<1>(acc_layout),
      //                    coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
    }
  } else { // SM80
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
      return acc_layout;
    } else {
      auto l = logical_divide(acc_layout, Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
      return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type_unsafe(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  static constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
  // Unsafe because we're returning a tensor with memory allocated on the stack. If the compiler does not
  // inline this function, then the memory might not be valid.
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const& tensor, Tensor<EngineOut, Layout>& out) {
  // Somehow if we allocate out inside this function and return it, e2e is slower and the output can be wrong.
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0, "Fragment size does not vectorize properly");
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  static_assert(size(frag) == size(out_frg));
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
#pragma unroll
  for (int i = 0; i < size(frag); ++i) {
    out_frg[i] = convert_op(frag[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
  if constexpr (A) {
    return mma.partition_fragment_A(tensor0);
  } else {
    return mma.partition_fragment_B(tensor0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init = false, int wg_wait = 0, bool SwapAB = false, int M_slice = -1, typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
CUTLASS_DEVICE void gemm(TiledMma& tiled_mma, Tensor0 const& tCrA, Tensor1 const& tCrB, Tensor2& tCrC) {
  if constexpr (M_slice >= 0) {
    static constexpr int MMA_M = decltype(size<1>(tCrC))::value;
    static_assert(M_slice < MMA_M);
    // After logical_divide, C has shape ((2,2,V), (MMA_M, 1), MMA_N)
    Tensor tCrC_slice = cute::logical_divide(tCrC, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
    if constexpr (!SwapAB) {
      Tensor tCrA_slice = cute::logical_divide(tCrA, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(tiled_mma, tCrA_slice, tCrB, tCrC_slice);
    } else {
      Tensor tCrB_slice = cute::logical_divide(tCrB, Shape<cute::Underscore, Int<MMA_M>>{})(_, make_coord(Int<M_slice>{}, _), _);
      gemm<zero_init, wg_wait, SwapAB, /*M_slice=*/-1>(tiled_mma, tCrA, tCrB_slice, tCrC_slice);
    }
  } else {
    constexpr bool Is_RS = !cute::is_base_of<GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value;
    // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take const
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1&>(tCrB));
      }
    }
    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    if constexpr (zero_init) {
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    }
    static constexpr int kNumKIters = CUTE_STATIC_V(size<2>(tCrA));
    static constexpr int kMaxKIters = 16;
    // Unroll the K mode manually to set scale D to 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < std::min(kNumKIters, kMaxKIters); ++k_block) {
      if constexpr (!SwapAB) {
        cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
      } else {
        cute::gemm(tiled_mma, tCrB(_, _, k_block), tCrA(_, _, k_block), tCrC);
      }
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    }
    // In the case of large kNumKIters, the compiler chooses to store the smem addresses
    // in registers, causing spills. This loop forces the compiler to recompute the addresses.
    if constexpr (kNumKIters > kMaxKIters) {
      // This will always be zero, just a way to force the compiler to recompute the smem
      // addresses. This results in USEL instructions. There's probably a better way to do this.
      int const k_offset = cutlass::canonical_warp_group_idx() < 128 ? 0 : 1;
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = kMaxKIters; k_block < kNumKIters; ++k_block) {
        if constexpr (!SwapAB) {
          cute::gemm(tiled_mma, tCrA(_, _, k_block + k_offset), tCrB(_, _, k_block + k_offset), tCrC);
        } else {
          cute::gemm(tiled_mma, tCrB(_, _, k_block + k_offset), tCrA(_, _, k_block + k_offset), tCrC);
        }
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
    }
    warpgroup_commit_batch();
    if constexpr (wg_wait >= 0) {
      warpgroup_wait<wg_wait>();
    }
    warpgroup_fence_operand(tCrC);
    if constexpr (Is_RS) {
      if constexpr (!SwapAB) {
        warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
      } else {
        warpgroup_fence_operand(const_cast<Tensor1&>(tCrB));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool A_in_regs = false,
    bool B_in_regs = false,
    bool SwapAB = false,
    typename Tensor0,
    typename Tensor1,
    typename Tensor2,
    typename Tensor3,
    typename Tensor4,
    typename TiledMma,
    typename TiledCopyA,
    typename TiledCopyB,
    typename ThrCopyA,
    typename ThrCopyB,
    typename Hook>
CUTLASS_DEVICE void gemm_sm80(
    Tensor0& acc,
    Tensor1& tCrA,
    Tensor2& tCrB,
    Tensor3 const& tCsA,
    Tensor4 const& tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B,
    Hook fn) {
  if constexpr (SwapAB) {
    gemm_sm80<B_in_regs, A_in_regs>(acc, tCrB, tCrA, tCsB, tCsA, tiled_mma, smem_tiled_copy_B, smem_tiled_copy_A, smem_thr_copy_B, smem_thr_copy_A, fn);
  } else {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc)); // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc)); // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view)); // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
    if (!A_in_regs) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
      if (i < size<2>(tCrA) - 1) {
        if (!A_in_regs) {
          cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
        }
        if (!B_in_regs) {
          cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
      }
      if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
        if (i == 0) {
          fn();
        }
      }
      cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename TiledMma, typename TiledCopy, typename ThrCopy>
CUTLASS_DEVICE void gemm_rs_sm80(
    Tensor0& acc,
    Tensor1& tCrA,
    Tensor2& tCrB,
    Tensor3 const& tCsB,
    TiledMma tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc)); // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB)); // MMA_K
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view)); // N
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool zero_init = false, typename Atom, typename TA, typename TB, typename TC>
CUTLASS_DEVICE void gemm_sm100(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  static constexpr int rA = decltype(rank(tA))::value;
  static constexpr int rB = decltype(rank(tB))::value;
  static constexpr int rC = decltype(rank(tC))::value;
  static_assert(rA == 3 && rB == 3 && rC == 3);

  if constexpr (zero_init) {
    atom.accumulate_ = decltype(atom.accumulate_)::Zero;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tA); k_block++) {
    cute::gemm(atom, tA(_, _, k_block), tB(_, _, k_block), tC);
    atom.accumulate_ = decltype(atom.accumulate_)::One;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg,
    UMMA::ScaleIn b_neg,
    class... TAs,
    class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<
        MMA_Atom<
            MMA_Traits<
                SM100_MMA_F8F6F4_SS,
                a_type,
                b_type,
                c_type,
                cute::C<M>,
                cute::C<N>,
                cute::integral_constant<UMMA::Major, a_major>,
                cute::integral_constant<UMMA::Major, b_major>,
                cute::integral_constant<UMMA::ScaleIn, a_neg>,
                cute::integral_constant<UMMA::ScaleIn, b_neg>>,
            TAs...>,
        TMs...>) {
  return TiledMMA<MMA_Atom<MMA_Traits<SM100_MMA_F8F6F4_TS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False>>, TAs...>, TMs...>{};
}

template <
    class a_type,
    class b_type,
    class c_type,
    int M,
    int N,
    UMMA::Major a_major,
    UMMA::Major b_major,
    UMMA::ScaleIn a_neg,
    UMMA::ScaleIn b_neg,
    class... TAs,
    class... TMs>
CUTE_HOST_DEVICE constexpr auto to_tiled_mma_sm100_ts(
    TiledMMA<MMA_Atom<SM100_MMA_F16BF16_SS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>, TAs...>, TMs...>) {
  return TiledMMA<MMA_Atom<SM100_MMA_F16BF16_TS<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg, UMMA::Saturate::False>, TAs...>, TMs...>{};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool Is_even_MN = true,
    bool Is_even_K = true,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    class CopyAtom,
    class TV,
    class Tiler,
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1,
    typename Engine2,
    typename Layout2,
    typename Engine3,
    typename Layout3>
CUTLASS_DEVICE void copy(
    TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
    Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D,
    Tensor<Engine2, Layout2> const& identity_MN,
    Tensor<Engine3, Layout3> const& predicate_K,
    const int max_MN = 0) {
  // Decay TiledCopy to CopyAtom
  auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
  auto has_with_bool = cute::is_valid([](auto t) -> void_t<decltype(declval<typename decltype(t)::Traits>().with(true))> {}, copy_atom);
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    bool predicate_mn = Is_even_MN || get<0>(identity_MN(_0{}, m, _0{})) < max_MN;
    if constexpr (Is_even_MN || !Clear_OOB_MN) {
      if (Is_even_MN || predicate_mn) {
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          if constexpr (Is_even_K || !Clear_OOB_K) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            }
          } else { // Clear_OOB_K == true && Is_even_K == false
            // If copy traits can be transformed with a predicate value, do it, otherwise branch here
            if constexpr (has_with_bool) {
              cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
            } else {
              if (predicate_K(k)) {
                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
              } else {
                cute::clear(D(_, m, k));
              }
            }
          }
        }
      }
    } else { // Clear_OOB_MN == true && Is_even_MN == false, also implies Clear_OOB_K == true
      if constexpr (!has_with_bool) {
        if (predicate_mn) {
#pragma unroll
          for (int k = 0; k < size<2>(S); ++k) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_K) {
              cute::clear(D(_, m, k));
            }
          }
        } else {
          cute::clear(D(_, m, _));
        }
      } else { // combine the mn predicate with the k predicate
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}

template <
    bool Is_even_MN = true,
    bool Is_even_K = true,
    bool Clear_OOB_MN = false,
    bool Clear_OOB_K = true,
    class CopyAtom,
    class TV,
    class Tiler,
    typename Engine0,
    typename Layout0,
    typename Engine1,
    typename Layout1,
    typename Engine2,
    typename Layout2,
    typename Engine3,
    typename Layout3,
    typename Engine4,
    typename Layout4>
CUTLASS_DEVICE void copy2(
    TiledCopy<CopyAtom, TV, Tiler> const& tiled_copy,
    Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D,
    Tensor<Engine2, Layout2> const& identity_MN,
    Tensor<Engine3, Layout3> const& predicate_K,
    Tensor<Engine4, Layout4> const& predicate_M) {
  // Decay TiledCopy to CopyAtom
  auto copy_atom = static_cast<CopyAtom const&>(tiled_copy);
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D)); // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D)); // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D)); // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
  auto has_with_bool = cute::is_valid([](auto t) -> void_t<decltype(declval<typename decltype(t)::Traits>().with(true))> {}, copy_atom);
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    bool predicate_mn = Is_even_MN || predicate_M(m);
    if constexpr (Is_even_MN || !Clear_OOB_MN) {
      if (Is_even_MN || predicate_mn) {
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          if constexpr (Is_even_K || !Clear_OOB_K) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            }
          } else { // Clear_OOB_K == true && Is_even_K == false
            // If copy traits can be transformed with a predicate value, do it, otherwise branch here
            if constexpr (has_with_bool) {
              cute::copy(copy_atom.with(predicate_K(k)), S(_, m, k), D(_, m, k));
            } else {
              if (predicate_K(k)) {
                cute::copy(copy_atom, S(_, m, k), D(_, m, k));
              } else {
                cute::clear(D(_, m, k));
              }
            }
          }
        }
      }
    } else { // Clear_OOB_MN == true && Is_even_MN == false, also implies Clear_OOB_K == true
      if constexpr (!has_with_bool) {
        if (predicate_mn) {
#pragma unroll
          for (int k = 0; k < size<2>(S); ++k) {
            if (Is_even_K || predicate_K(k)) {
              cute::copy(copy_atom, S(_, m, k), D(_, m, k));
            } else if (Clear_OOB_K) {
              cute::clear(D(_, m, k));
            }
          }
        } else {
          cute::clear(D(_, m, _));
        }
      } else { // combine the mn predicate with the k predicate
#pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(copy_atom.with(predicate_mn && (Is_even_K || predicate_K(k))), S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Byte permute and shuffle to match register layout of
// (FP8 downcasted) accumulator of GEMM-I to FP8 operand A of GEMM-II.
template <typename Fragment>
CUTLASS_DEVICE void permute_Aregs_fp8(Fragment& frag) {
  // frag has shape ((4, 2, 2), MMA_M, MMA_N), each element is 8 bits
  static_assert(decltype(size<0, 0>(frag))::value == 4);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(decltype(stride<0, 1>(frag))::value == 4);
  static_assert(sizeof(typename Fragment::value_type) == 1);

  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;
  int selector_upper = lane_03 ? 0x5410 : 0x1054;
  int selector_lower = lane_03 ? 0x7632 : 0x3276;

  static constexpr int upper_map[4] = {0, 3, 1, 2};
  // static constexpr int lower_map[4] = {1, 2, 0, 3};

  Tensor frag_64b = recast<uint2>(frag); // ((1, 1, 2), MMA_M, MMA_N)
#pragma unroll
  for (int i = 0; i < size(frag_64b); ++i) {
    uint32_t upper = frag_64b[i].x;
    uint32_t lower = frag_64b[i].y;
    uint32_t upper0 = lane_03 ? upper : lower;
    uint32_t lower0 = lane_03 ? lower : upper;
    upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
    // lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[quad_idx], 4);
    lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 1, 4);
    frag_64b[i].x = __byte_perm(upper0, lower0, selector_upper);
    frag_64b[i].y = __byte_perm(upper0, lower0, selector_lower);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment& frag) {
  // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 4);
  Tensor frag_64b = group_modes<1, 3>(recast<uint2>(frag)); // ((1, 2, N / 8), (MMA_M, MMA_N))
#pragma unroll
  for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
#pragma unroll
    for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
      cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi), frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8(Fragment& out) {
  // out has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
  static_assert(decltype(size<0, 0>(out))::value == 2);
  static_assert(decltype(size<0, 1>(out))::value == 2);
  static_assert(decltype(size<0, 2>(out))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(out))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 4);
  Tensor frag = group_modes<1, 3>(out); // ((2, 2, N / 8), (MMA_M, MMA_N))
#pragma unroll
  for (int mi = 0; mi < size<1>(frag); ++mi) {
#pragma unroll
    for (int j = 0; j < size<0, 1>(frag); ++j) {
#pragma unroll
      for (int i = 0; i < size<0, 2>(frag) / 2; ++i) {
        cutlass::swap(frag(make_coord(_1{}, j, 2 * i), mi), frag(make_coord(_0{}, j, 2 * i + 1), mi));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Fragment>
CUTLASS_DEVICE void permute_output_fp8_Vcolmajor(Fragment& frag) {
  // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 16 bits
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 2 || sizeof(typename Fragment::value_type) == 4);

  int quad_idx = threadIdx.x % 4;
  bool lane_03 = quad_idx == 0 || quad_idx == 3;

  static constexpr int upper_map[4] = {0, 2, 3, 1};
  // static constexpr int lower_map[4] = {2, 0, 1, 3};

  // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(frag); }
  using type2 = std::conditional_t<sizeof(typename Fragment::value_type) == 2, uint32_t, uint64_t>;
  Tensor frag_2 = group_modes<1, 3>(recast<type2>(frag)); // ((1, 2, N / 8), (MMA_M, MMA_N))
// if (blockIdx.x == 0 && threadIdx.x == 128) { print(frag); printf("\n"); print(frag_2); }
#pragma unroll
  for (int mi = 0; mi < size<1>(frag_2); ++mi) {
#pragma unroll
    for (int j = 0; j < size<0, 1>(frag_2); ++j) {
#pragma unroll
      for (int i = 0; i < size<0, 2>(frag_2) / 2; ++i) {
        type2 upper = frag_2(make_coord(_0{}, j, 2 * i), mi);
        type2 lower = frag_2(make_coord(_0{}, j, 2 * i + 1), mi);
        type2 upper0 = lane_03 ? upper : lower;
        type2 lower0 = lane_03 ? lower : upper;
        upper0 = __shfl_sync(uint32_t(-1), upper0, upper_map[quad_idx], 4);
        // lower0 = __shfl_sync(uint32_t(-1), lower0, lower_map[quad_idx], 4);
        lower0 = __shfl_sync(uint32_t(-1), lower0, upper_map[quad_idx] ^ 2, 4);
        frag_2(make_coord(_0{}, j, 2 * i), mi) = lane_03 ? upper0 : lower0;
        frag_2(make_coord(_0{}, j, 2 * i + 1), mi) = lane_03 ? lower0 : upper0;
      }
    }
  }
  // if (blockIdx.x == 0 && threadIdx.x == 128) { print_tensor(frag); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
CUTLASS_DEVICE void apply_softcap(Tensor<Engine, Layout>& tensor, float const softcap) {
#pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
  }
}

template <typename Engine, typename Layout>
CUTLASS_DEVICE auto calculate_dtanh(Tensor<Engine, Layout>& tensor) {
  Tensor out = make_fragment_like<float>(tensor);
#pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    out(i) = 1.f - (tensor(i) * tensor(i));
  }
  return out;
}

template <typename Element>
CUTLASS_DEVICE static constexpr auto sizeof_bytes_v() {
  return cutlass::sizeof_bits_v<Element> / 8;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Broadcast a value from a source lane to all lanes in the warp
template <class T>
CUTE_DEVICE T broadcast_in_warp(T val, int src_lane = 0) {
  return __shfl_sync(0xffffffff, val, src_lane);
}

// Count number of lanes in the warp where predicate is true
CUTE_DEVICE int count_in_warp(int predicate) {
  return __popc(__ballot_sync(0xffffffff, predicate));
}

// Inclusive prefix sum within a warp
template <class T>
CUTE_DEVICE T warp_prefix_sum(T val) {
  int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
    T partial_sum = __shfl_up_sync(0xffffffff, val, i);
    if (lane >= i) {
      val += partial_sum;
    }
  }
  return val;
}

// Get a uniform value across the warp by broadcasting from lane 0
template <class T>
CUTE_DEVICE T warp_uniform(T a) {
  return broadcast_in_warp(a, /*src_lane=*/0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE
int canonical_thread_idx_in_warpgroup_nosync() {
  return threadIdx.x % cutlass::NumThreadsPerWarpGroup;
}

CUTLASS_DEVICE
int canonical_thread_idx_in_warpgroup_sync() {
  return warp_uniform(canonical_thread_idx_in_warpgroup_nosync());
}

CUTLASS_DEVICE
int canonical_warp_idx_in_warpgroup_nosync() {
  return threadIdx.x / cutlass::NumThreadsPerWarp % cutlass::NumWarpsPerWarpGroup;
}

CUTLASS_DEVICE
int canonical_warp_idx_in_warpgroup_sync() {
  return warp_uniform(canonical_warp_idx_in_warpgroup_nosync());
}

CUTLASS_DEVICE
int canonical_warp_group_idx_nosync() {
  return threadIdx.x / cutlass::NumThreadsPerWarpGroup;
}

CUTLASS_DEVICE
int canonical_warp_group_idx_sync() {
  return warp_uniform(canonical_warp_group_idx_nosync());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Synchronize all threads in a cooperative group (CGA)
// or fall back to `__syncthreads()` for single-block cluster (i.e. normal CTA)
template <typename ClusterShape>
CUTLASS_DEVICE void sync_cga_threads() {
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }
}

// Get TMA multi-cast load metadata: multicast mask and cluster block ID
// RowwiseMask=true: multicast along M (x dim) mode for this N (y dim) load
// RowwiseMask=false: multicast along N (y dim) mode for this M (x dim) load
template <typename ClusterShape, typename GmemTiledCopy, bool RowwiseMask = true>
CUTLASS_DEVICE cute::tuple<uint16_t, uint32_t> get_tma_multi_cast_meta() {
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

  uint32_t cluster_block_id = RowwiseMask ? cluster_local_block_id.x : cluster_local_block_id.y;

  uint16_t mcast_mask = 0;
  if constexpr (cute::is_same_v<GmemTiledCopy, SM90_TMA_LOAD_MULTICAST>) {
    auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
    if constexpr (RowwiseMask) {
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    } else {
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
      }
    }
  }

  return {mcast_mask, cluster_block_id};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE
void cp_async_cacheglobal_l2_prefetch_256B(const void* src, void* dst, bool pred, int64_t cache_policy) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
  asm volatile("cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, %3;\n" ::"r"(dst_addr), "l"(src), "r"(pred ? 16 : 0), "l"(cache_policy));
}

CUTLASS_DEVICE
int64_t createpolicy_evict_last() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0; \n\t" : "=l"(res) :);
  return res;
}

CUTLASS_DEVICE
int64_t createpolicy_evict_first() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0; \n\t" : "=l"(res) :);
  return res;
}

// Define available cache operators matching PTX instructions
enum class LoadMode {
  Default, // Default cache behavior
  LDG, // read-only cache, .global.nc on some archs
  CG, // Cache at Global level (bypass L1, .global.cg)
  CA, // Cache at All levels (.global.ca)
  CS, // Cache Streaming (evict first, .global.cs)
  LU, // Last Use (.global.lu)
  CV // Cache Volatile (.global.cv)
};

// Helper to select the correct intrinsic function at compile time
template <LoadMode Mode, typename T>
__device__ __forceinline__ T load_from_global(const T* ptr) {
  if constexpr (Mode == LoadMode::LDG) {
    return __ldg(ptr);
  } else if constexpr (Mode == LoadMode::CG) {
    return __ldcg(ptr); // Bypass L1, useful if L1 is thrashing
  } else if constexpr (Mode == LoadMode::CA) {
    return __ldca(ptr);
  } else if constexpr (Mode == LoadMode::CS) {
    return __ldcs(ptr);
  } else if constexpr (Mode == LoadMode::LU) {
    return __ldlu(ptr);
  } else if constexpr (Mode == LoadMode::CV) {
    return __ldcv(ptr);
  } else {
    return *ptr; // Default load
  }
}

/**
 * @brief Generic load and broadcast function with controllable cache behavior.
 *
 * @tparam T The data type (e.g., float, float4, int2).
 * @tparam NumElem The number of elements in the vector type (1, 2, 3, or 4).
 * @tparam Mode The cache load mode.
 */
template <int NumElem, LoadMode Mode = LoadMode::Default, typename T>
__device__ __forceinline__ T load_and_broadcast(const T* ptr) {
  // Compile-time check to ensure supported vector sizes
  static_assert(NumElem >= 1 && NumElem <= 4, "Unsupported NumElem");

  // 1. Load data only on the first lane (Lane 0) of the warp.
  // Use bitwise AND for slightly better performance than modulo.
  T val = (threadIdx.x % 32 == 0) ? load_from_global<Mode>(ptr) : T{};

  // 2. Broadcast logic using if constexpr.
  if constexpr (NumElem == 1) {
    // Case for scalar types (float, int)
    return warp_uniform(val);
  } else if constexpr (NumElem == 2) {
    // Case for vector types with 2 components (float2, double2, int2)
    val.x = warp_uniform(val.x);
    val.y = warp_uniform(val.y);
    return val;
  } else if constexpr (NumElem == 3) {
    // Case for vector types with 3 components (float3, etc.)
    val.x = warp_uniform(val.x);
    val.y = warp_uniform(val.y);
    val.z = warp_uniform(val.z);
    return val;
  } else if constexpr (NumElem == 4) {
    // Case for vector types with 4 components (float4, int4)
    val.x = warp_uniform(val.x);
    val.y = warp_uniform(val.y);
    val.z = warp_uniform(val.z);
    val.w = warp_uniform(val.w);
    return val;
  }
  return val; // Should not be reached
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ void atomicMaxFloatOnlyIncrease(float* addr, float val) {
  if (isnan(val))
    return;
  unsigned int* ptr = (unsigned int*)addr;
  unsigned int old, assumed;
  old = *ptr;
  while (val > __uint_as_float(old)) {
    assumed = old;
    old = atomicCAS(ptr, assumed, __float_as_uint(val));
    if (assumed == old)
      break;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
