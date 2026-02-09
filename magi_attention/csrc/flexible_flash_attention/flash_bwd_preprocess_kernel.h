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
#include "sink_layout.cuh"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <class TileShape_MK_, class Element, class ElementDq, class ArchTag_, bool ClearDq, bool Has_sink, SinkLayout kSinkLayout>
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
  // the first s_sink threads process the sink, thus we need kMaxSeqlenSink <= MaxThreadsPerBlock
  static_assert(kMaxSeqlenSink <= MaxThreadsPerBlock);

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(get<1>(TileShape_MK{}) % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kBlockM = get<0>(TileShape_MK{});
  static constexpr int kHeadDim = get<1>(TileShape_MK{});
  // one thread processes one row, thus we need kBlockM <= MaxThreadsPerBlock
  static_assert(kBlockM <= MaxThreadsPerBlock);
  static constexpr int kNumPartialSink = Has_sink ? MaxThreadsPerBlock : 1; // NOTE: we use dummy 1 to reduce memory usage if not `Has_sink`
  static constexpr int kWarpSize = 32;
  static_assert(!Has_sink || kNumPartialSink % kWarpSize == 0); // NOTE: we would demand the block size to be divisible by the warp size if `Has_sink`

  static_assert(kSinkLayout == SinkLayout::SH or kSinkLayout == SinkLayout::SSH, "Unsupported SinkLayout");
  using TileShapeMSink = cute::Shape<Int<kBlockM>, Int<kMaxSeqlenSink>>;

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

  static constexpr int kGmemElemsPerLoaddQ = sizeof(cute::uint128_t) / sizeof(ElementDq);
  static_assert((kBlockM * kHeadDim / kGmemElemsPerLoaddQ) % MaxThreadsPerBlock == 0, "MaxThreadsPerBlock must divide kBlockM * kHeadDim / kGmemElemsPerLoaddQ");

  using GmemLayoutAtomdQ = Layout<Shape<Int<MaxThreadsPerBlock>>>;
  using GmemTiledCopydQ = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementDq>{},
      GmemLayoutAtomdQ{},
      Layout<Shape<Int<kGmemElemsPerLoaddQ>>>{})); // Val layout, 4 or 8 vals per store

  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>; // (sq, hd, nhq)
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  using ShapedPsum = cute::Shape<_4, int32_t, int32_t>; // (4, sq_rounded, nhq)
  using StridedPsum = cute::Stride<_1, _4, int64_t>;
  using ShapeLSE = cute::Shape<int32_t, int32_t>; // (sq, nhq)
  using StrideLSE = cute::Stride<int64_t, _1>;
  using ShapeSink = cute::Shape<int32_t, int32_t, int32_t>; // (1, s_sink, nhq) or (sq, s_sink, nhq)
  using StrideSink = cute::Stride<int64_t, int64_t, _1>;
  using ShapedSinkReduceBuf = cute::Shape<int32_t, int32_t, int32_t>; // (num_m_block, s_sink, nhq)
  using StridedSinkReduceBuf = cute::Stride<int64_t, int64_t, _1>;

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
    // dQaccum
    void* ptr_dQaccum;
    ShapeO const shape_dQaccum;
    StrideO const stride_dQaccum;
    // sink
    float* ptr_sink;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
    // dsink
    float* ptr_dsink;
    float* ptr_dsink_reduce_buf;
    unsigned int* ptr_dsink_reduce_cnt; // one counter per head
    ShapedSinkReduceBuf const shape_dsink_reduce_buf;
    StridedSinkReduceBuf const stride_dsink_reduce_buf;
    // meta
    int const num_m_block;
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
    // dQaccum
    void* ptr_dQaccum;
    ShapeO const shape_dQaccum;
    StrideO const stride_dQaccum;
    // sink
    float* ptr_sink = nullptr;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
    // dsink
    float* ptr_dsink = nullptr;
    float* ptr_dsink_reduce_buf = nullptr;
    unsigned int* ptr_dsink_reduce_cnt = nullptr;
    ShapedSinkReduceBuf const shape_dsink_reduce_buf;
    StridedSinkReduceBuf const stride_dsink_reduce_buf;
    // meta
    int const num_m_block;
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
        // dQaccum
        args.ptr_dQaccum,
        args.shape_dQaccum,
        args.stride_dQaccum,
        // sink
        args.ptr_sink,
        args.shape_sink,
        args.stride_sink,
        // dsink
        args.ptr_dsink,
        args.ptr_dsink_reduce_buf,
        args.ptr_dsink_reduce_cnt,
        args.shape_dsink_reduce_buf,
        args.stride_dsink_reduce_buf,
        // meta
        args.num_m_block,
        args.total_q,
        args.total_sink,
    };
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
    // Get block coordinates
    int const m_block = blockIdx.y;
    int const bidh = blockIdx.z;

    // Get thread coordinates
    int const thread_idx = threadIdx.x;

    // Get seqlen info
    int const remain_valid_seqlen_q = params.total_q - m_block * kBlockM;
    bool const is_valid_row = thread_idx < kBlockM && thread_idx < remain_valid_seqlen_q;

    // Initialize the input tensors for O, dO, LSE, sink and dsink
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh); // [sq, hd]
    Tensor gO = local_tile(cute::domain_offset(make_coord(0, _0{}), mO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)
    Tensor mdO = make_tensor(make_gmem_ptr(params.ptr_dO), params.shape_O, params.stride_dO)(_, _, bidh); // [sq, hd]
    Tensor gdO = local_tile(cute::domain_offset(make_coord(0, _0{}), mdO), TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh); // [sq,]
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(0), mLSE), Shape<Int<kBlockM>>{}, make_coord(m_block)); // (M,)
    Tensor mSink = make_tensor(make_gmem_ptr(params.ptr_sink), params.shape_sink, params.stride_sink)(_, _, bidh); // [1, s_sink] or [sq, s_sink]
    Tensor gSink = local_tile(mSink, TileShapeMSink{}, make_coord(m_block, _0{})); // (M, MAX_SINK), used only when `kSinkLayout == SSH`
    Tensor mdSink = make_tensor(make_gmem_ptr(params.ptr_dsink), params.shape_sink, params.stride_sink)(_, _, bidh); // [1, s_sink] or [sq, s_sink]
    Tensor gdSink = local_tile(mdSink, TileShapeMSink{}, make_coord(m_block, _0{})); // (M, MAX_SINK), used only when `kSinkLayout == SSH`

    // Load the LSE
    // NOTE: we mask the OOB lse as `inf`,
    // to make the subsequent calculation of OOB scores (exp(x - lse))
    // become exp(x - `inf`) = exp(`-inf`) = 0
    float lse = is_valid_row ? gLSE(thread_idx) : INFINITY;

    // Initialize static shared memory buffer for p_sink and later partial dsink
    __shared__ float shared_pd_sink[kMaxSeqlenSink][kNumPartialSink];

    // Load the sink and compute p_sink
    if constexpr (Has_sink) {
      if constexpr (kSinkLayout == SinkLayout::SH) { // sink.shape = [1, s_sink]
        // Initialize static shared memory for sink
        __shared__ float shared_sink[kMaxSeqlenSink];

        // Load the sink to shared memory by first s_sink threads in the block
        if (thread_idx < params.total_sink) {
          shared_sink[thread_idx] = mSink(thread_idx);
        }
        __syncthreads();

        // Compute the `p_sink = exp(sink - lse)`
        // for this row with the corr. lse and store to shared memory
        // NOTE: the OOB part in shared_pd_sink will be zero since lse = inf,
        // which is necessary to safely reduce partial dsink later
#pragma unroll
        for (int si = 0; si < params.total_sink; ++si) {
          shared_pd_sink[si][thread_idx] = safe_softmax(shared_sink[si], lse);
        }
        __syncthreads();
      } else if constexpr (kSinkLayout == SinkLayout::SSH) { // sink.shape = [M, s_sink]
        float sink_this_row[kMaxSeqlenSink];

#pragma unroll
        // Load the sink to register for this row by this thread
        for (int si = 0; si < params.total_sink; ++si) {
          sink_this_row[si] = is_valid_row ? gSink(thread_idx, si) : 0.0f;
        }

        // Compute the `p_sink = exp(sink - lse)`
        // for this row with the corr. lse and store to shared memory
        // NOTE: the OOB part in shared_pd_sink will be zero since lse = inf,
        // which is necessary to safely reduce partial dsink later
#pragma unroll
        for (int si = 0; si < params.total_sink; ++si) {
          shared_pd_sink[si][thread_idx] = safe_softmax(sink_this_row[si], lse);
        }
        __syncthreads();
      }
    }

    // Initialize the tiled copy for O and dO
    GmemTiledCopy gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    // Partition the src global tensors for O and dO
    Tensor tOgO = gmem_thr_copy_O.partition_S(gO);
    Tensor tOgdO = gmem_thr_copy_O.partition_S(gdO);

    // Construct identity layout of gO for indexing convenience
    Tensor cO = cute::make_identity_tensor(TileShape_MK{}); // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);

    // Construct the predicate mask for head dim of pO
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO))); // (K,)
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O); // hd_idx < hd
    }

    // Load the global tensors to register for O and dO by tiled copy
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
    // and upcast to float32 for higher reduce precision
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
    // and also compute partial dsink if `Has_sink`
    // by the threads whose hd_idx == 0
    if (get<1>(tOcO(_0{}, _0{}, _0{})) == 0) { // hd_idx = 0
#pragma unroll
      for (int mi = 0; mi < size(dP_sum); ++mi) {
        int const row_idx = get<0>(tOcO(_0{}, mi, _0{})); // row_idx
        float dPsum_mi = row_idx < remain_valid_seqlen_q ? dP_sum(mi) : 0; // NOTE: we make OOB dPsum as 0
        gdPsum(row_idx) = dPsum_mi; // NOTE: the OOB part had better be set to 0

        // Compute `dsink = p_sink * -dPsum`
        if constexpr (Has_sink) {
          if constexpr (kSinkLayout == SinkLayout::SH) { // sink.shape = [1, s_sink]
#pragma unroll
            // Compute partial `dsink = p_sink * -dPsum` for this row
            // and store to the same shared memory as p_sink
            for (int si = 0; si < params.total_sink; ++si) {
              shared_pd_sink[si][row_idx] *= -dPsum_mi;
            }
          } else if constexpr (kSinkLayout == SinkLayout::SSH) { // sink.shape = [M, s_sink]
            if (row_idx < remain_valid_seqlen_q) {
#pragma unroll
              // Compute `dsink = p_sink * -dPsum` for this valid row
              // and store to the global memory
              for (int si = 0; si < params.total_sink; ++si) {
                gdSink(row_idx, si) = shared_pd_sink[si][row_idx] * -dPsum_mi;
              }
            }
          }
        }
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

    // Reduce partial dsink along the seqlen_q dim
    // if `Has_sink` and `kSinkLayout == SinkLayout::SH`
    if constexpr (Has_sink and kSinkLayout == SinkLayout::SH) {
      __syncthreads();
      if (params.num_m_block == 1) {
        // Since there is only one m block, just apply block reduction and store to dsink directly
        block_reduce_dsink(params, mdSink, shared_pd_sink, thread_idx);
      } else {
        Tensor mdSinkReduceBuf =
            make_tensor(make_gmem_ptr(params.ptr_dsink_reduce_buf), params.shape_dsink_reduce_buf, params.stride_dsink_reduce_buf)(_, _, bidh); // [num_m_block, s_sink]
        Tensor mdSinkReduceBufThisBlock = mdSinkReduceBuf(m_block, _); // [s_sink,]

        // Apply block-reduced dsink and store to reduce buffer for this block
        block_reduce_dsink(params, mdSinkReduceBufThisBlock, shared_pd_sink, thread_idx);

        // Mark the block-reduction of dsink by this block as done
        // and return a flag to indicate whether this is the last finished m block
        bool is_this_m_block_last_done = mark_this_m_block_done(params, bidh, thread_idx);

        // The last m block will load all the block-reduced dsink results from reduce buffer
        // then reduce them to store the final reduced dsink
        if (is_this_m_block_last_done) {
          global_reduce_dsink(params, mdSink, mdSinkReduceBuf, shared_pd_sink, thread_idx);
        }
      }
    }

    if constexpr (ClearDq) {
      Tensor mdQaccum = make_tensor(make_gmem_ptr(static_cast<ElementDq*>(params.ptr_dQaccum)), params.shape_dQaccum, params.stride_dQaccum);
      Tensor mdQaccum_head = mdQaccum(_, _, bidh);
      Tensor gdQaccum = local_tile(mdQaccum_head, TileShape_MK{}, make_coord(m_block, _0{})); // (M, K)

      GmemTiledCopydQ gmem_tiled_copy_dQaccum;
      auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
      Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);
      Tensor zero = make_fragment_like(tdQgdQaccum);
      clear(zero);

      Tensor cDQ = cute::make_identity_tensor(TileShape_MK{});
      Tensor tcDQ = gmem_thr_copy_dQaccum.partition_D(cDQ);

#pragma unroll
      for (int m = 0; m < size<1>(tdQgdQaccum); ++m) {
        for (int k = 0; k < size<2>(tdQgdQaccum); ++k) {
          if (get<0>(tcDQ(0, m, k)) < remain_valid_seqlen_q && get<1>(tcDQ(0, m, k)) < kHeadDim) {
            cute::copy(gmem_tiled_copy_dQaccum, zero(_, m, k), tdQgdQaccum(_, m, k));
          }
        }
      }
    }
  }

  CUTLASS_DEVICE float warp_reduce_dsink(unsigned int mask, float acc_dsink) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
      acc_dsink += __shfl_down_sync(mask, acc_dsink, offset);
    }
    return acc_dsink;
  }

  template <typename TensordSink>
  CUTLASS_DEVICE void block_reduce_dsink(Params const& params, TensordSink& dsink, float shared_dsink[kMaxSeqlenSink][kNumPartialSink], int const thread_idx) {
    static constexpr unsigned warp_mask = 0xffffffff;
    static constexpr int kNumWarps = kNumPartialSink / kWarpSize;
    __shared__ float shared_warp_reduced_dsink[kMaxSeqlenSink][kNumWarps];

    int const warp_idx = thread_idx / kWarpSize;
    int const land_idx = thread_idx % kWarpSize;

#pragma unroll
    for (int si = 0; si < params.total_sink; ++si) {
      __syncthreads();

      float acc_dsink = shared_dsink[si][thread_idx];

      // Accumulate warp-reduced dsink into lane0
      acc_dsink = warp_reduce_dsink(warp_mask, acc_dsink);

      // Store the warp-reduced dsink to shared memory by lane0
      if (land_idx == 0) {
        shared_warp_reduced_dsink[si][warp_idx] = acc_dsink;
      }
      __syncthreads();

      // Load the warp-reduced dsink from shared memory by the first `kNumWarps` threads
      // and apply final warp-reduce to accumulate block-reduced dsink into thread0
      const unsigned int ballot_warp_mask = __ballot_sync(warp_mask, thread_idx < kNumWarps);
      if (thread_idx < kNumWarps) {
        acc_dsink = shared_warp_reduced_dsink[si][thread_idx];
        acc_dsink = warp_reduce_dsink(ballot_warp_mask, acc_dsink);
      }

      // Stores block-reduced dsink by thread0
      if (thread_idx == 0) {
        dsink(si) = acc_dsink;
      }
    }
  }

  template <typename TensordSink, typename TensordSinkReduceBuf>
  CUTLASS_DEVICE void global_reduce_dsink(
      Params const& params,
      TensordSink& dsink,
      TensordSinkReduceBuf& dsink_reduce_buf,
      float shared_dsink[kMaxSeqlenSink][kNumPartialSink],
      int const thread_idx) {
    // Make sure the atomic operation is completed
    // before loading the dsink reduce buffer (acquire order)
    __threadfence(); // REVIEW: is this fence necessary ?

    // Load the block-reduced dsink from reduce buffer
    // and accumulate them to shared memory
#pragma unroll
    for (int si = 0; si < params.total_sink; ++si) {
      float acc_dsink = 0;
#pragma unroll
      for (int bi = thread_idx; bi < params.num_m_block; bi += kNumPartialSink) {
        acc_dsink += dsink_reduce_buf(bi, si);
      }
      shared_dsink[si][thread_idx] = acc_dsink;
    }
    __syncthreads();

    // Apply one more block reduction to store the final reduced dsink
    block_reduce_dsink(params, dsink, shared_dsink, thread_idx);
  }

  CUTLASS_DEVICE
  bool mark_this_m_block_done(Params const& params, int const bidh, int const thread_idx) const {
    // Make sure all writes to dsink reduce buffer in this block is visible to others (release order)
    __threadfence();
    __syncthreads();

    __shared__ bool is_this_m_block_last_done_shared;

    // Mark the block-reduction of dsink by this block as done
    // and get the flag to indicate whether this is the last finished m block
    if (thread_idx == 0) {
      unsigned int order = atomicInc(&params.ptr_dsink_reduce_cnt[bidh], params.num_m_block);
      is_this_m_block_last_done_shared = (order == params.num_m_block - 1);
    }

    __syncthreads();

    return is_this_m_block_last_done_shared;
  }
};

} // namespace flash
