// Copyright (c) 2025 SandAI. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cute/layout.hpp>

#include <cutlass/arch/grid_dependency_control.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include "utils.h"

namespace flash {
using namespace cute;

template <typename T_out_, uint32_t kBlockM_, uint32_t kHeadDim_, class ArchTag_>
class FastZeroFillKernel {
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
  using StrideLSE = cute::Stride<_1, int64_t>;

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

  struct Arguments {
    T_out* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
  };

  struct Params {
    T_out* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // (seqlen_q, num_heads_qo)
    auto const shape_LSE = select<0, 2>(args.shape_O);
    return {args.ptr_O, args.shape_O, args.stride_O, args.ptr_LSE, shape_LSE, args.stride_LSE};
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
  void operator()(Params const& params, char* smem_buf) {
    // Get block coordinates
    int32_t const block = blockIdx.x;
    int32_t const bidh = blockIdx.y;

    // Get thread coordinates
    int32_t const thread_idx = threadIdx.x;

    // Get offset and shape of the output tensor
    int32_t const offset_o = block * kBlockM;
    int32_t const seqlen_o = cute::get<0>(params.shape_O);
    int32_t const head_dim_o = cute::get<1>(params.shape_O);

    // Initialize global tensors
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh);
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh);
    Tensor gO = local_tile(mO, TileShapeMK{}, make_coord(block, _0{})); // (M, K)

    // Initialize gmem_tiled_copy_O and gmem_thr_copy_O
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    // Get tOrO (source) and clear it
    Tensor tOrO = make_fragment_like(gmem_thr_copy_O.partition_S(make_tensor<T_out>(TileShapeMK{})));
    cute::clear(tOrO);
    // Get tOgO (destination)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // Get tOcO for easy access to row and col
    Tensor tOcO = gmem_thr_copy_O.partition_S(cute::make_identity_tensor(TileShapeMK{}));

    // Get tLSErLSE and tOpO to indicate which is needed to be set to zero
    auto tOpOm = make_fragment_like(make_tensor<bool>(make_shape(size<1>(tOgO))));
    auto tOpOk = make_fragment_like(make_tensor<bool>(make_shape(size<2>(tOgO))));
    auto tLSErLSE = make_fragment_like(make_tensor<float>(make_shape(size<1>(tOgO))));
    // Clear tOpOm and tOpOk
    cute::clear(tOpOm);
    cute::clear(tOpOk);

#pragma unroll
    for (int32_t i = 0; i < size(tLSErLSE); ++i) {
      // Load LSE into tLSErLSE and set tOpOm to true if the LSE is -INFINITY
      int32_t row = get<0>(tOcO((_0{}, _0{}), i, _0{})) + offset_o;
      if (row < seqlen_o) {
        tLSErLSE(i) = mLSE(row);
        tOpOm(i) = tLSErLSE(i) == -INFINITY;
      } else {
        tLSErLSE(i) = -INFINITY;
        tOpOm(i) = false;
      }
    }

#pragma unroll
    for (int32_t i = 0; i < size(tOpOk); ++i) {
      // Set tOpOk to true if the col is in the range of the output tensor
      int32_t col = get<1>(tOcO((_0{}, _0{}), _0{}, i));
      tOpOk(i) = col < head_dim_o;
    }

    // Should we add __syncthreads() here?
    // __syncthreads();

    // Print debug info
    // if (thread_idx == 0 && block == 0 && bidh == 0) {
    //     printf("kGmemElemsPerStore: %d. kBytePerRow: %d. kBlockKGmem: %d. kGmemThreadsPerRow: %d.\n", kGmemElemsPerStore, kBytePerRow, kBlockKGmem,
    //     kGmemThreadsPerRow); printf("=================================== tLSErLSE ===================================\n"); print_tensor(tLSErLSE);
    //     printf("=================================== tOcO ===================================\n");
    //     print_tensor(tOcO);
    //     printf("=================================== tOpOm ===================================\n");
    //     print_tensor(tOpOm);
    //     printf("=================================== tOpOk ===================================\n");
    //     print_tensor(tOpOk);
    //     printf("=================================== tOrO ===================================\n");
    //     print_tensor(tOrO);
    // }

    flash::copy2<false, false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpOk, tOpOm);
  }
};
} // namespace flash
