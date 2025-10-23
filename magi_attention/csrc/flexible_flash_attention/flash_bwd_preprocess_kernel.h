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

#include "cute/tensor.hpp"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementAccum, class ArchTag_, bool Clear_dQ, bool Clear_dK, bool Clear_dV>
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

  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen_q, d, head)
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  using ShapedPsum = cute::Shape<_4, int32_t, int32_t>; // (4, total_seqlen, head)
  using StridedPsum = cute::Stride<_1, _4, int64_t>;
  using ShapeLSE = cute::Shape<int32_t, int32_t>; // (total_q, head)
  using StrideLSE = cute::Stride<int64_t, _1>;

  // Device side arguments
  struct Arguments {
    Element const* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    Element const* ptr_dO;
    StrideO const stride_dO;
    float* ptr_dPsum;
    ShapedPsum const shape_dPsum;
    StridedPsum const stride_dPsum;
    ShapeLSE const shape_LSE;
    float const* ptr_LSE;
    StrideLSE const stride_LSE;
    float* ptr_LSE_log2;
    StridedPsum const stride_LSE_log2;
    int2 const* q_ranges;
    int2 const* k_ranges;
    int const total_q;
    int const total_q_rounded;
  };

  // Kernel entry point API
  struct Params {
    Element const* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    Element const* ptr_dO;
    StrideO const stride_dO;
    float* ptr_dPsum;
    ShapedPsum const shape_dPsum;
    StridedPsum const stride_dPsum;
    ShapeLSE const shape_LSE;
    float const* ptr_LSE;
    StrideLSE const stride_LSE;
    float* ptr_LSE_log2;
    StridedPsum const stride_LSE_log2;
    int2 const* q_ranges = nullptr;
    int2 const* k_ranges = nullptr;
    int const total_q;
    int const total_q_rounded;
  };

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const& args) {
    return {
        args.ptr_O,
        args.shape_O,
        args.stride_O,
        args.ptr_dO,
        args.stride_dO,
        args.ptr_dPsum,
        args.shape_dPsum,
        args.stride_dPsum,
        args.shape_LSE,
        args.ptr_LSE,
        args.stride_LSE,
        args.ptr_LSE_log2,
        args.stride_LSE_log2,
        args.q_ranges,
        args.k_ranges,
        args.total_q,
        args.total_q_rounded,
    };
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
    static constexpr int kBlockM = get<0>(TileShape_MK{});

    // one thread processes one row, thus we need kBlockM <= MaxThreadsPerBlock
    static_assert(kBlockM <= MaxThreadsPerBlock);

    // Get block coordinates
    int const thread_idx = threadIdx.x;

    /**
     * NOTE: Here, we shift the batch size to the x-dimension because the z-dimension must be less than 65536.
     *  Thus once the batch size is too large (>= 65536), the z-dimension may overflow, causing an implicit
     * kernel-launch error. What's worse, this error may not be explicitly raised, resulting in the kernel being
     * skipped.
     */
    int const m_block = blockIdx.y;
    int const bidh = blockIdx.z;

    // TODO: remove to params
    // auto shape_LSE = select<0, 2>(params.shape_O);
    // Initialize the tensors for O, dO, and LSE
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh);
    Tensor gO = local_tile(cute::domain_offset(make_coord(0, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)

    Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh);
    Tensor gdO = local_tile(cute::domain_offset(make_coord(0, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh);
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(0), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block));

    // mask the oob lse as INFINITY.
    float lse = thread_idx < params.total_q - m_block * kBlockM && thread_idx < kBlockM ? gLSE(thread_idx) : INFINITY;

    // Initialize the tiled copy for O and dO
    GmemTiledCopy gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
    Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);

    // Construct identity layout for gO
    Tensor cO = cute::make_identity_tensor(TileShape_MK{}); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O);
    }

    // (8, kBlockM / 32, kHeadDim / 64) or (8, kBlockM / 16, kHeadDim / 128)
    Tensor tOrO = make_fragment_like(tOgO);
    Tensor tOrdO = make_fragment_like(tOgdO);
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
        gmem_tiled_copy_O, tOgO, tOrO, tOcO, tOpO, params.total_q - m_block * kBlockM);
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clearn_OOB_K=*/true>(
        gmem_tiled_copy_O, tOgdO, tOrdO, tOcO, tOpO, params.total_q - m_block * kBlockM);
    // if (threadIdx.x == 222) { printf("bidx = %d, bidy = %d, bidz = %d, seqlen_o = %d, m_block = %d, seqlen_o -
    // m_block * kBlockM = %d, tOgO addr = %p\n", blockIdx.x, blockIdx.y, blockIdx.z, seqlen_o, m_block, seqlen_o -
    // m_block * kBlockM, &tOgO(0));}

    // Reshape from e.g. (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, (8, kHeadDim / 64))
    Layout l = make_layout(get<1>(tOrO.layout()), make_layout(get<0>(tOrO.layout()), get<2>(tOrO.layout())));
    Tensor tOrO_l = make_tensor(tOrO.data(), l);
    Tensor o_fp32 = make_tensor_like<float>(tOrO_l);
    flash::convert_type_out(tOrO_l, o_fp32);
    Tensor tOrdO_l = make_tensor(tOrdO.data(), l);
    Tensor do_fp32 = make_tensor_like<float>(tOrdO_l);
    flash::convert_type_out(tOrdO_l, do_fp32);
    // Sum across the last dimension
    Tensor dP_sum = make_tensor<float>(make_shape(size<0>(o_fp32)));
#pragma unroll
    for (int mi = 0; mi < size<0>(o_fp32); ++mi) {
      float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
#pragma unroll
      for (int ni = 1; ni < size<1>(o_fp32); ni++) {
        dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
      }
      flash::SumOp<float> sum_op;
      dP_sum(mi) = flash::Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
    }

    Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_dPsum, params.stride_dPsum)(0, _, bidh); // total_q
    Tensor gdPsum = local_tile(cute::domain_offset(make_coord(0), mdPsum), Shape<Int<kBlockM>>{}, make_coord(m_block));

    if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) {
#pragma unroll
      for (int mi = 0; mi < size(dP_sum); ++mi) {
        int const row = get<0>(tOcO(_0{}, mi, _0{}));
        gdPsum(row) = row < params.total_q - m_block * kBlockM ? dP_sum(mi) : 0;
      }
    }

    Tensor mLSElog2 = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_dPsum, params.stride_LSE_log2)(0, _, bidh); // total_q
    Tensor gLSElog2 = local_tile(cute::domain_offset(make_coord(0), mLSElog2), Shape<Int<kBlockM>>{}, make_coord(m_block));

    // We should not write back -inf because the subsequent calculation of scores would involve -inf - (-inf), which
    // results in NaN.
    if (thread_idx < params.total_q_rounded - m_block * kBlockM && thread_idx < kBlockM) {
      gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
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
