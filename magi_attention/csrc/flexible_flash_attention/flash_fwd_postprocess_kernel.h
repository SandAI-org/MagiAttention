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

#pragma once

#include <stdexcept>

#include <cute/layout.hpp>

#include <cutlass/arch/grid_dependency_control.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include "sink_layout.cuh"
#include "softmax.h"
#include "utils.h"

namespace flash {
using namespace cute;

template <typename T_out_, uint32_t kBlockM_, uint32_t kHeadDim_, bool Has_sink, SinkLayout kSinkLayout, class ArchTag_>
class FlashAttnFwdPostprocess {
 public:
  using ArchTag = ArchTag_;
  using T_out = T_out_;
  static constexpr uint32_t kBlockM = kBlockM_;
  static constexpr uint32_t kHeadDim = kHeadDim_;

  using TileShapeMK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;

  // (seqlen_q, head_dim, num_heads_qo)
  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  // (seqlen_q, num_heads_qo)
  using ShapeLSE = cute::Shape<int32_t, int32_t>;
  using StrideLSE = cute::Stride<int64_t, _1>;
  // (1, seqlen_sink, num_heads_qo) or (seqlen_q, seqlen_sink, num_heads_qo)
  using ShapeSink = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideSink = cute::Stride<int64_t, int64_t, _1>;

  // These are for storing the output tensor without TMA (e.g., for setting output to zero)
  static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(T_out);
  static_assert(kHeadDim % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");

  // The "Row" below refers to a Head.
  // Bytes per head
  static constexpr int kBytePerRow = kHeadDim * sizeof(T_out);
  // Number of (128-byte, 64-byte, or 32-byte) blocks per head
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(T_out);
  // Number of threads required to collaboratively read/write one (128-byte, 64-byte, or 32-byte) block
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;

  // If PackGQA, we split the work of compute O_ptr among threads in the same row, so we need this to within a warp
  static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);

  // Number of epilogue threads must be a multiple of kGmemThreadsPerRow
  static_assert(kBlockM % kGmemThreadsPerRow == 0, "kBlockM must be a multiple of kGmemThreadsPerRow");

  // Layout of Epilogue threads, named GmemLayoutAtom
  using GmemLayoutAtom = cute::Layout<Shape<Int<kBlockM / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;

  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, T_out>{},
      GmemLayoutAtom{},
      cute::Layout<Shape<_1, Int<kGmemElemsPerStore>>>{})); // Val layout, 8 or 16 vals per store

  static constexpr int SharedStorageSize = 0;
  static constexpr uint32_t MaxThreadsPerBlock = kBlockM;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int kMaxSeqlenSink = Has_sink ? 8 : 1; // NOTE: we use dummy 1 to reduce memory usage if not `Has_sink`
  // the first s_sink threads process the sink, thus we need kMaxSeqlenSink <= MaxThreadsPerBlock
  static_assert(kMaxSeqlenSink <= MaxThreadsPerBlock);

  static_assert(kSinkLayout == SinkLayout::SH or kSinkLayout == SinkLayout::SSH, "Unsupported SinkLayout");
  using TileShapeMSink = cute::Shape<Int<kBlockM>, Int<kMaxSeqlenSink>>;
  static constexpr int kBlockMSink = kSinkLayout == SinkLayout::SH ? 1 : kBlockM;

  struct Arguments {
    // O
    T_out* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    // LSE
    float* ptr_LSE;
    StrideLSE const stride_LSE;
    // sink
    float* ptr_sink;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
  };

  struct Params {
    // O
    T_out* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    // LSE
    float* ptr_LSE;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE;
    // sink
    float* ptr_sink = nullptr;
    ShapeSink const shape_sink;
    StrideSink const stride_sink;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // Check seqlen_sink
    auto const seqlen_sink = get<1>(args.shape_sink);
    if (seqlen_sink > kMaxSeqlenSink) {
      throw std::runtime_error("Invalid seqlen sink: " + std::to_string(seqlen_sink) + ", must be <= kMaxSeqlenSink: " + std::to_string(kMaxSeqlenSink));
    }

    // (seqlen_q, num_heads_qo)
    auto const shape_LSE = select<0, 2>(args.shape_O);

    return {
        // O
        args.ptr_O,
        args.shape_O,
        args.stride_O,
        // LSE
        args.ptr_LSE,
        shape_LSE,
        args.stride_LSE,
        // sink
        args.ptr_sink,
        args.shape_sink,
        args.stride_sink};
  }

  static dim3 get_grid_shape(Params const& params) {
    int32_t const seqlen_o = cute::get<0>(params.shape_O);
    int32_t const num_heads_qo = cute::get<2>(params.shape_O);
    return dim3(cute::ceil_div(seqlen_o, kBlockM), num_heads_qo, 1);
  }

  static dim3 get_block_shape() {
    return dim3(kBlockM, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, [[maybe_unused]] char* smem_buf) {
    // Get block coordinates
    int32_t const block = blockIdx.x;
    int32_t const bidh = blockIdx.y;

    // Get thread coordinates
    int32_t const thread_idx = threadIdx.x;

    // Get offset and shape of the output tensor
    int32_t const offset_o = block * kBlockM;
    int32_t const seqlen_o = cute::get<0>(params.shape_O);
    int32_t const head_dim_o = cute::get<1>(params.shape_O);
    int32_t const seqlen_sink = cute::get<1>(params.shape_sink);

    // Get seqlen info
    int const remain_valid_seqlen_o = seqlen_o - offset_o;
    bool const is_valid_row = thread_idx < remain_valid_seqlen_o;

    // Initialize global tensors for O, LSE and sink
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh); // [sq, hd]
    Tensor gO = local_tile(mO, TileShapeMK{}, make_coord(block, _0{})); // (M, K)
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh); // [sq,]
    Tensor gLSE = local_tile(mLSE, cute::select<0>(TileShapeMK{}), make_coord(block)); // (M,)
    Tensor mSink = make_tensor(make_gmem_ptr(params.ptr_sink), params.shape_sink, params.stride_sink)(_, _, bidh); // [1, s_sink] or [sq, s_sink]
    Tensor gSink = local_tile(mSink, TileShapeMSink{}, make_coord(block, _0{})); // (M, MAX_SINK), used only when `kSinkLayout == SSH`

    // Initialize static shared memory for shared sink and lse_sink
    __shared__ float shared_sink[kBlockMSink][kMaxSeqlenSink];
    __shared__ float shared_lse_sink[kBlockMSink];

    // Load the sink and compute lse_sink
    if constexpr (Has_sink) {
      if constexpr (kSinkLayout == SinkLayout::SH) { // sink.shape = [1, s_sink]
        // Load the sink to shared memory
        // by first s_sink threads in the block
        if (thread_idx < seqlen_sink) {
          shared_sink[0][thread_idx] = mSink(thread_idx);
        }
        __syncthreads();

        // Compute the `lse_sink = log(sum(exp(sink)))`
        // by the thread0 in the block
        if (thread_idx == 0)
          shared_lse_sink[0] = calc_lse(shared_sink[0], seqlen_sink);
        __syncthreads();
      } else if constexpr (kSinkLayout == SinkLayout::SSH) { // sink.shape = [M, s_sink]
#pragma unroll
        // Load the sink to shared memory
        // for each row by each thread
        for (int si = 0; si < seqlen_sink; ++si) {
          shared_sink[thread_idx][si] = is_valid_row ? gSink(thread_idx, si) : 0.0f;
        }
        __syncthreads();

        // Compute the `lse_sink = log(sum(exp(sink)))`
        // for each row by each thread
        shared_lse_sink[thread_idx] = is_valid_row ? calc_lse(shared_sink[thread_idx], seqlen_sink) : -INFINITY;
        __syncthreads();
      }
    }

    // Initialize gmem_tiled_copy_O and gmem_thr_copy_O
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    // Get tOrO (source) and clear it to zero
    Tensor rO = make_tensor<T_out>(TileShapeMK{});
    Tensor tOrO = make_fragment_like(gmem_thr_copy_O.partition_S(rO));
    cute::clear(tOrO);

    // Get tOgO (destination) to be zero-filled
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // Get tOcO for easy access to row (seqlen) and col (head dim)
    Tensor cO = cute::make_identity_tensor(TileShapeMK{});
    Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

    // Get tLSErLSE and tOpO to indicate which row of O is needed to be set to zero
    // as well as tOpOk to indicate which col is needed to be set
    // and if `Has_sink`, get tOpOm_sink to indicate which row of O is needed to be rescaled
    auto tLSErLSE = make_fragment_like(make_tensor<float>(make_shape(size<1>(tOgO))));
    auto tOpOm = make_fragment_like(make_tensor<bool>(make_shape(size<1>(tOgO))));
    auto tOpOk = make_fragment_like(make_tensor<bool>(make_shape(size<2>(tOgO))));
    auto tOpOm_sink = make_fragment_like(make_tensor<bool>(make_shape(size<1>(tOgO))));

    // Clear tOpOm and tOpOk to false
    cute::clear(tOpOm);
    cute::clear(tOpOk);
    if constexpr (Has_sink) {
      // Clear tOpOm_sink to false if `Has_sink`
      cute::clear(tOpOm_sink);
    }

#pragma unroll
    // Load LSE into tLSErLSE
    // and set tOpOm to true if the LSE is -INFINITY
    // if `Has_sink`, set tOpOm_sink to true if the LSE is not -INFINITY and store rescaled LSE
    for (int32_t i = 0; i < size(tLSErLSE); ++i) {
      int32_t row_idx = get<0>(tOcO((_0{}, _0{}), i, _0{}));
      int32_t hd_idx = get<1>(tOcO((_0{}, _0{}), i, _0{}));
      if (row_idx < remain_valid_seqlen_o) {
        // Load LSE
        tLSErLSE(i) = gLSE(row_idx);
        tOpOm(i) = tLSErLSE(i) == -INFINITY;

        // Correct LSE and compute the rescale weights for O if `Has_sink`
        if constexpr (Has_sink) {
          // Set tOpOm_sink to true if the LSE is not -INFINITY
          tOpOm_sink(i) = !tOpOm(i);

          // Rescale LSE by lse_sink
          float corr_lse;
          if constexpr (kSinkLayout == SinkLayout::SH) { // sink.shape = [1, s_sink]
            corr_lse = tOpOm(i) ? shared_lse_sink[0] : correct_lse(tLSErLSE(i), shared_lse_sink[0]);
          } else if constexpr (kSinkLayout == SinkLayout::SSH) { // sink.shape = [M, s_sink]
            corr_lse = tOpOm(i) ? shared_lse_sink[row_idx] : correct_lse(tLSErLSE(i), shared_lse_sink[row_idx]);
          }

          // Store the rescale weight into tLSErLSE for later use to rescale O
          tLSErLSE(i) = tOpOm(i) ? 0 : calc_lse_rescale_weight(tLSErLSE(i), corr_lse);

          // Store rescaled LSE to global memory
          // by the thread whose hd_idx == 0
          if (hd_idx == 0)
            gLSE(row_idx) = corr_lse;
        }
      }
    }

#pragma unroll
    // Set tOpOk to true if the col, i.e. the head dim
    // is in the range of the output tensor
    for (int32_t i = 0; i < size(tOpOk); ++i) {
      int32_t col = get<1>(tOcO((_0{}, _0{}), _0{}, i));
      tOpOk(i) = col < head_dim_o;
    }

    /* DEBUG */
    // if (thread_idx == 0 && block == 0 && bidh == 0) {
    //     printf("kGmemElemsPerStore: %d. kBytePerRow: %d. kBlockKGmem: %d. kGmemThreadsPerRow: %d.\n",
    //     kGmemElemsPerStore, kBytePerRow, kBlockKGmem, kGmemThreadsPerRow);
    //     printf("=================================== tLSErLSE ===================================\n");
    //     print_tensor(tLSErLSE);
    //     printf("=================================== tOcO ===================================\n");
    //     print_tensor(tOcO);
    //     printf("=================================== tOpOm ===================================\n");
    //     print_tensor(tOpOm);
    //     printf("=================================== tOpOk ===================================\n");
    //     print_tensor(tOpOk);
    //     printf("=================================== tOgO ===================================\n");
    //     print_tensor(tOgO);
    //     printf("=================================== tOrO ===================================\n");
    //     print_tensor(tOrO);
    // }

    // Store the zero-filled tOrO to tOgO, where tOpOm and tOpOk are true
    // then the non-covered part of O is set to zero
    flash::copy2</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clearn_OOB_K=*/false>(
        /*tiled_copy=*/gmem_tiled_copy_O,
        /*S=*/tOrO,
        /*D=*/tOgO,
        /*identity_MN=*/tOcO,
        /*predicate_K=*/tOpOk,
        /*predicate_M=*/tOpOm);

    if constexpr (Has_sink) {
      // Retile the src/dst tensors to dst/src resp.
      Tensor tOgO_src = gmem_thr_copy_O.retile_S(tOgO);
      Tensor tOrO_dst = gmem_thr_copy_O.retile_D(tOrO); // NOTE: we reuse the zero-filled tOrO

      // Load the original tOgO to tOrOt, where tOpOm_sink and tOpOk are true
      flash::copy2</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clearn_OOB_K=*/false>(
          /*tiled_copy=*/gmem_tiled_copy_O,
          /*S=*/tOgO_src,
          /*D=*/tOrO_dst,
          /*identity_MN=*/tOcO,
          /*predicate_K=*/tOpOk,
          /*predicate_M=*/tOpOm_sink);

#pragma unroll
      // Rescale the tOrO by rescale_weight stored in tLSErLSE
      for (int m = 0; m < size<1>(tOrO_dst); ++m) {
        float rescale_weight_m = tLSErLSE(m);
        Tensor tOrO_dst_m = tOrO_dst(_, m, _);
#pragma unroll
        for (int i = 0; i < size(tOrO_dst_m); ++i) {
          tOrO_dst_m(i) = static_cast<T_out>(static_cast<float>(tOrO_dst_m(i)) * rescale_weight_m);
        }
      }

      // Store the rescaled tOrO to tOgO, where tOpOm_sink and tOpOk are true
      // then the covered part of O is rescaled by sink
      flash::copy2</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clearn_OOB_K=*/false>(
          /*tiled_copy=*/gmem_tiled_copy_O,
          /*S=*/tOrO,
          /*D=*/tOgO,
          /*identity_MN=*/tOcO,
          /*predicate_K=*/tOpOk,
          /*predicate_M=*/tOpOm_sink);
    }
  }
};
} // namespace flash
