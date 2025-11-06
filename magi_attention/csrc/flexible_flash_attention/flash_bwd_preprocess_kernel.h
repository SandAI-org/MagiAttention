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

#include <stdexcept>

#include "cute/tensor.hpp"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, bool Clear_dQ, bool Clear_dK, bool Clear_dV, bool Has_sink, bool Deterministic>
class FlashAttnBwdPreprocess {
 public:
  // Type Aliases
  using TileShape_MK = TileShape_MK_;
  using ArchTag = ArchTag_;

  static_assert(
      std::is_same_v<Element, cutlass::half_t> && ArchTag::kMinComputeCapability >= 75 ||
      std::is_same_v<Element, cutlass::bfloat16_t> && ArchTag::kMinComputeCapability >= 80 ||
      std::is_same_v<Element, cutlass::float_e4m3_t> && ArchTag::kMinComputeCapability >= 89);

  static constexpr uint32_t MaxThreadsPerBlock = 256;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;
  static constexpr int SharedStorageSize = 0;
  static constexpr int kMaxSeqlenSink = Has_sink ? 8 : 1; // NOTE: we use dummy 1 to reduce memory usage if not `Has_sink`

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBlockM = get<0>(TileShape_MK{});
  static constexpr int kHeadDim = get<1>(TileShape_MK{});
  // We want kBlockKGmem to be a power of 2 so that when we do the summing,
  // it's just between threads in the same warp
  static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(MaxThreadsPerBlock % kGmemThreadsPerRow == 0, "MaxThreadsPerBlock must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<Shape<Int<MaxThreadsPerBlock / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals per load

  static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
  static_assert((kBlockM * kHeadDim / kGmemElemsPerLoadAccum) % MaxThreadsPerBlock == 0, "MaxThreadsPerBlock must divide kBlockM * kHeadDim / kGmemElemsPerLoadAccum");

  using GmemLayoutAtomAccum = Layout<Shape<Int<MaxThreadsPerBlock>>>;
  using GmemTiledCopyAccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      GmemLayoutAtomAccum{},
      Layout<Shape<Int<kGmemElemsPerLoadAccum>>>{})); // Val layout, 4 vals per store

  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>; // (sq, hd, nhq)
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  using ShapedPsum = cute::Shape<_4, int32_t, int32_t>; // (4, sq_rounded, nhq)
  using StridedPsum = cute::Stride<_1, _4, int64_t>;
  using ShapeLSE = cute::Shape<int32_t, int32_t>; // (sq, nhq)
  using StrideLSE = cute::Stride<int64_t, _1>;
  using ShapeSink = cute::Shape<int32_t, int32_t>; // (s_sink, nhq)
  using StrideSink = cute::Stride<int64_t, _1>;

  // Device side arguments
  struct Arguments {
    // O
    Element const* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    // dO
    Element const* ptr_dO;
    StrideO const stride_dO;
    // dPsum
    float* ptr_dPsum;
    ShapedPsum const shape_dPsum;
    StridedPsum const stride_dPsum;
    // LSE
    float const* ptr_LSE;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE;
    // LSE_log2
    float* ptr_LSE_log2;
    StridedPsum const stride_LSE_log2;
    // sink
    float* ptr_sink;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
    // dsink
    float* ptr_dsink;
    // meta
    int const total_q;
    int const total_sink;
  };

  // Kernel entry point API
  struct Params {
    // O
    Element const* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    // dO
    Element const* ptr_dO;
    StrideO const stride_dO;
    // dPsum
    float* ptr_dPsum;
    ShapedPsum const shape_dPsum;
    StridedPsum const stride_dPsum;
    // LSE
    float const* ptr_LSE;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE;
    // LSE_log2
    float* ptr_LSE_log2;
    StridedPsum const stride_LSE_log2;
    // sink
    float* ptr_sink = nullptr;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
    // dsink
    float* ptr_dsink = nullptr;
    // meta
    int const total_q;
    int const total_sink = 0;
  };

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    // Check total_sink
    if (args.total_sink > kMaxSeqlenSink) {
      throw std::runtime_error("Invalid seqlen sink: " + std::to_string(args.total_sink) + ", must be <= kMaxSeqlenSink: " + std::to_string(kMaxSeqlenSink));
    }

    return {
        // O
        args.ptr_O,
        args.shape_O,
        args.stride_O,
        // dO
        args.ptr_dO,
        args.stride_dO,
        // dPsum
        args.ptr_dPsum,
        args.shape_dPsum,
        args.stride_dPsum,
        // LSE
        args.ptr_LSE,
        args.shape_LSE,
        args.stride_LSE,
        // LSE_log2
        args.ptr_LSE_log2,
        args.stride_LSE_log2,
        // sink
        args.ptr_sink,
        args.shape_sink,
        args.stride_sink,
        // dsink
        args.ptr_dsink,
        // meta
        args.total_q,
        args.total_sink,
    };
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MK{});

    // one thread processes one row, thus we need kBlockM <= MaxThreadsPerBlock
    static_assert(kBlockM <= MaxThreadsPerBlock);

    // Get block coordinates
    int const m_block = blockIdx.y;
    int const bidh = blockIdx.z;

    // Get thread coordinates
    int const thread_idx = threadIdx.x;

    // Get seqlen info
    int const remain_valid_seqlen_q = params.total_q - m_block * kBlockM;
    bool const is_valid_row = thread_idx < remain_valid_seqlen_q && thread_idx < kBlockM;

    // Initialize the input tensors for O, dO and LSE
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh); // [sq, hd]
    Tensor gO = local_tile(cute::domain_offset(make_coord(0, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)
    Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh); // [sq, hd]
    Tensor gdO = local_tile(cute::domain_offset(make_coord(0, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh); // [sq,]
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(0), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block)); // (M,)

    // Load the LSE
    // NOTE: we mask the OOB lse as `inf`,
    // to make the subsequent calculation of OOB scores (exp(x - lse))
    // become exp(0 - `inf`) = exp(`-inf`) = 0
    float lse = is_valid_row ? gLSE(thread_idx) : INFINITY;

    // Initialize shared memory / register for sink
    __shared__ float shared_sink[kMaxSeqlenSink];
    float p_sink[kMaxSeqlenSink];

    // Load the sink
    if constexpr (Has_sink) {
      // Initialize the input tensor for sink
      Tensor mSink = make_tensor(make_gmem_ptr(params.ptr_sink), params.shape_sink, params.stride_sink)(_, bidh); // [s_sink,]

      // Load the sink to shared memory by first s_sink threads in the block
      if (thread_idx < params.total_sink) {
        shared_sink[thread_idx] = mSink(thread_idx);
      }
      __syncthreads();

#pragma unroll
      // Compute the `p_sink = exp(sink - lse)`
      // for this row with the corr. lse
      for (int si = 0; si < params.total_sink; ++si) {
        p_sink[si] = exp(shared_sink[si] - lse);
      }
    }

    // Initialize the tiled copy for O and dO
    GmemTiledCopy gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    // Partition the src thread global tensors for O and dO
    Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
    Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);

    // Construct identity layout of gO
    // and partition the dst thread global tensor of cO
    Tensor cO = cute::make_identity_tensor(TileShape_MK{}); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

    // Construct the predicate mask for head dim of pO
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO))); // (K,)
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); // hd_idx < hd
    }

    // Load the thread global tensors to register for O and dO by tiled copy
    // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
    Tensor tOrO = make_fragment_like(tOgO);
    Tensor tOrdO = make_fragment_like(tOgdO);
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
        /*tiled_copy=*/gmem_tiled_copy_O,
        /*S=*/tOgO,
        /*D=*/tOrO,
        /*identity_MN=*/tOcO,
        /*predicate_K=*/tOpO,
        /*max_MN=*/remain_valid_seqlen_q);
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
        /*tiled_copy=*/gmem_tiled_copy_O,
        /*S=*/tOgdO,
        /*D=*/tOrdO,
        /*identity_MN=*/tOcO,
        /*predicate_K=*/tOpO,
        /*max_MN=*/remain_valid_seqlen_q);

    // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
    // and upcast to float32
    Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
    Tensor tOrO_l = make_tensor(tOrO.data(), l);
    Tensor tOrO_l_fp32 = make_tensor_like<float>(tOrO_l); // (tM, tK)
    flash::convert_type_out(tOrO_l, tOrO_l_fp32);
    Tensor tOrdO_l = make_tensor(tOrdO.data(), l);
    Tensor tOrdO_l_fp32 = make_tensor_like<float>(tOrdO_l); // (tM, tK)
    flash::convert_type_out(tOrdO_l, tOrdO_l_fp32);

    // Compute `dPsum = sum(O * dO, dim=-1)`
    // and all reduce across the head dim
    Tensor dP_sum = make_tensor<float>(make_shape(size<0>(tOrO_l_fp32))); // (tM,)
#pragma unroll
    for (int mi = 0; mi < size<0>(tOrO_l_fp32); ++mi) {
      float dP_sum_cur = tOrdO_l_fp32(mi, 0) * tOrO_l_fp32(mi, 0);
#pragma unroll
      for (int ni = 1; ni < size<1>(tOrO_l_fp32); ni++) {
        dP_sum_cur += tOrdO_l_fp32(mi, ni) * tOrO_l_fp32(mi, ni);
      }
      flash::SumOp<float> sum_op;
      dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
    }

    // Initialize the output tensor for dPsum
    Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(0, _, bidh); // [sq,]
    Tensor gdPsum = local_tile(cute::domain_offset(make_coord(0), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block)); // (M,)

    // Store the reduced dPsum to output tensor
    // by the thread holding the head dim 0
    // NOTE: we make OOB dPsum as 0
    if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) { // hd_idx = 0
#pragma unroll
      for (int mi = 0; mi < size(dP_sum); ++mi) {
        int const row = get<0>(tOcO(_0{}, mi, _0{})); // row_idx
        gdPsum(row) = row < remain_valid_seqlen_q ? dP_sum(mi) : 0;
      }
    }

    // Initialize the output tensor for LSE_log2
    Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(0, _, bidh); // [sq,]
    Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(0), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block)); // (M,)

    // Scale and store the LSE to LSE_log2
    // NOTE: we reset the valid `-inf` to 0
    // to make the subsequent calculation of scores (exp(x - lse)) always correct
    // since when x = lse = `-inf`, the results would be NaN, but the expected result is `-inf`.
    // So instead, we reset `-inf` lse to 0 to make `-inf` - (`-inf`) become `-inf` - 0 = `-inf`
    if (is_valid_row) {
      gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
    }

    // Compute partial `dsink = p_sink * -dPsum` for this row
    // and then reduce-add it back (atomically / sequentially) to the global memory of dsink
    if constexpr (Has_sink) {
      // Make sure all writes to global memory in this block (esp. gdPsum)
      // before this point are completed
      __threadfence_block();

      if (is_valid_row) {
        // Initialize the output tensor for dsink
        Tensor mdSink = make_tensor(make_gmem_ptr(params.ptr_dsink), params.shape_sink, params.stride_sink)(_, bidh); // [s_sink,]

        // Load the negative dPsum for this row
        float neg_dPsum = -gdPsum(thread_idx);

#pragma unroll
        // Compute partial `dsink = p_sink * -dPsum`
        // and reduce-add to the global memory of dsink
        for (int si = 0; si < params.total_sink; ++si) {
          if constexpr (Deterministic) {
            // TODO: support sequential reduce-add for deterministic mode
          } else { // atomic reduce-add for non-deterministic mode
            atomicAdd(&mdSink(si), p_sink[si] * neg_dPsum);
          }
        }
      }
    }

    // if constexpr (Clear_dQ) {
    //     Tensor mdQaccum = make_tensor(make_gmem_ptr(params.ptr_dQaccum), params.shape_dQaccum,
    //     params.stride_dQaccum)(_, bidh, !is_varlen ? bidb : 0); Tensor gdQaccum =
    //     local_tile(cute::domain_offset(make_coord(seqlen_info.offset_padded * kHeadDim), mdQaccum), Shape<Int<kBlockM
    //     * kHeadDim>>{}, make_coord(m_block)); GmemTiledCopyAccum gmem_tiled_copy_dQaccum; auto gmem_thr_copy_dQaccum
    //     = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx); Tensor tdQgdQaccum =
    //     gmem_thr_copy_dQaccum.partition_D(gdQaccum); Tensor zero = make_fragment_like(tdQgdQaccum); clear(zero);
    //     cute::copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, zero, tdQgdQaccum);
    // }
  }
};

} // namespace flash
