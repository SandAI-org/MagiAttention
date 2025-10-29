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

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h> // For FastDivMod

#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <
    class TileShape_MNK_PV_,
    class ClusterShape_,
    class Element_,
    class ArchTag_,
    class BlockCoordType_,
    int NumEpilogueThreads_,
    bool DisableFwdAtomicReduction_,
    bool Deterministic_ = false,
    bool SwapAB_ = false>
struct CollectiveEpilogueFwd {
  // KblockM, Kheaddim, KblockN
  using TileShape_MNK_PV = TileShape_MNK_PV_;
  using ClusterShape = ClusterShape_;
  using Element = Element_;
  using ElementPartial = float;
  using ArchTag = ArchTag_;
  using BlockCoordType = BlockCoordType_;
  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool DisableFwdAtomicReduction = DisableFwdAtomicReduction_;
  static constexpr bool Deterministic = Deterministic_;
  static constexpr bool SwapAB = SwapAB_;

  static_assert(ArchTag::kMinComputeCapability >= 90 || CUTE_STATIC_V(size(ClusterShape{})) == 1);
  // static_assert(sizeof(Element) <= 2);

  static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
  static constexpr int kHeadDim = get<1>(TileShape_MNK_PV{});

  // when SwapAB == true, set the warp group overlap tileMMA size for kBlockM
  static constexpr int TileSize_kBlockM = kBlockM;
  // static constexpr int TileSize_kBlockM = kBlockM == 8 ? kBlockM : kBlockM / 2;
  using TileShape_MNK_PV_SwapAB_OP_SELECT = Shape<Int<kHeadDim>, Int<TileSize_kBlockM>, decltype(get<2>(TileShape_MNK_PV{}))>;

  using TileShape_MNK_PV_Active = std::conditional_t<SwapAB, TileShape_MNK_PV_SwapAB_OP_SELECT, TileShape_MNK_PV>;

  // TODO: Use finegrained TMA store for output, currently hardcoded to false
  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

  // These are for storing the output tensor without TMA (e.g., for setting output to zero)
  static constexpr int kGmemElemsPerStore = kBlockM >= 32 ? sizeof(cute::uint128_t) / sizeof(Element) : sizeof(cute::uint64_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerStore == 0, "Headdim must be a multiple of kGmemElemsPerStore");
  // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). We want each thread to have 4 elements
  // in the M direction and 2 elements in the K direction. In the case of PackGQA, this reduces the number of times
  // we need to call divmod.

  // The "Row" below refers to a Head.
  // Bytes per head
  static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
  // Number of (128-byte, 64-byte, or 32-byte) blocks per head
  static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
  // Number of threads required to collaboratively read/write one (128-byte, 64-byte, or 32-byte) block
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerStore;

  // If PackGQA, we split the work of compute O_ptr among threads in the same row, so we need this to within a warp
  // remove assert because no PackGQA
  // static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0);

  // Number of epilogue threads must be a multiple of kGmemThreadsPerRow
  static_assert(NumEpilogueThreads % kGmemThreadsPerRow == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRow");

  // Layout of Epilogue threads, named GmemLayoutAtom
  using GmemLayoutAtom = Layout<Shape<Int<NumEpilogueThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>, Stride<Int<kGmemThreadsPerRow>, _1>>;
  // kBlockM must be divisible by the 0-th dimension of GmemLayoutAtom to ensure correct tiling
  static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0, "kBlockM must be a multiple of NumEpilogueThreads / kGmemThreadsPerRow");

  using GmemTileCopyAtomO = std::
      conditional_t<kBlockM >= 32, Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>, Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, Element>>;

  using GmemTiledCopyO =
      decltype(make_tiled_copy(GmemTileCopyAtomO{}, GmemLayoutAtom{}, Layout<Shape<_1, Int<kGmemElemsPerStore>>>{})); // Val layout, 8 or 16 vals per store

  using SmemLayoutAtomOTMA =
      decltype(cutlass::gemm::collective::detail::
                   ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
  using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
  static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{}, Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));
  using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;

  // (seqlen_q, d, head)
  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  using ShapeLSE = cute::Shape<int32_t, int32_t>; // (seqlen_q, nheads_kv)
  using StrideLSE = cute::Stride<int64_t, _1>; // (seqlen_q, head)
  using CopyOpR2S = std::conditional_t<
      ArchTag::kMinComputeCapability >= 90,
      // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
      decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<StrideO, ElementPartial>()),
      AutoVectorizingCopyWithAssumedAlignment<128>>;

  // static constexpr size_t SmemAlignmentO = cutlass::detail::alignment_for_swizzle(SmemLayoutO{});
  // static_assert(SmemAlignmentO >= 128, "Require at least 128B alignment");
  // struct TensorStorage : cute::aligned_struct<SmemAlignmentO> {
  //     cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0, SmemAlignmentO> smem_o;
  // };
  struct TensorStorage : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
  };

  using TMA_O = decltype(make_tma_copy(
      GmemTiledCopyOTMA{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeO{}, StrideO{}),
      SmemLayoutOTMA{},
      select<0, 1>(TileShape_MNK_PV{}),
      _1{})); // no mcast for O

  // Host side kernel arguments
  struct Arguments {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
    int32_t const nheads;
    int32_t const nheads_kv;
    int* range_locks = nullptr;
    int2 const* q_ranges = nullptr;
    int2 const* k_ranges = nullptr;
    int* determin_range_locks = nullptr;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_O tma_store_O;
    int const nheads;
    int* range_locks = nullptr;
    int2 const* q_ranges = nullptr;
    int2 const* k_ranges = nullptr;
    int* determin_range_locks = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
    TMA_O tma_store_O = make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, select<0, 1>(TileShape_MNK_PV{}), _1{}); // no mcast

    int const qhead_per_khead = 1;

    // seqlen_q, num_heads_qo
    auto const shape_LSE = select<0, 2>(args.shape_O);
    return {
        args.ptr_O,
        args.shape_O,
        args.stride_O,
        args.ptr_LSE,
        shape_LSE,
        args.stride_LSE,
        cutlass::FastDivmod(qhead_per_khead),
        tma_store_O,
        args.nheads,
        args.range_locks,
        args.q_ranges,
        args.k_ranges,
        args.determin_range_locks};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    // cute::prefetch_tma_descriptor(params.tma_store_O.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  void acquire_lock(int* range_lock, int bidh, int offset, int q_block_size, int num_heads) {
    // Calculate lock index
    int block_idx1 = offset / q_block_size;
    int index_1 = block_idx1 * num_heads + bidh;

    // Check if we need a second lock
    int block_idx2 = (offset + q_block_size - 1) / q_block_size;
    bool need_second_lock = (block_idx1 != block_idx2);

// Acquire the first lock
#pragma unroll 1
    while (atomicCAS(&range_lock[index_1], 0, 1) != 0) {
      // #if __CUDA_ARCH__ >= 700
      //     __nanosleep(100);
      // #endif
    }

    // If we need a second lock
    if (need_second_lock) {
      int index_2 = block_idx2 * num_heads + bidh;

// Try to acquire the second lock
#pragma unroll 1
      while (atomicCAS(&range_lock[index_2], 0, 1) != 0) {
        // Temporarily release the first lock to avoid deadlock
        // atomicExch(&range_lock[index_1], 0);

        // Sleep briefly
        // #if __CUDA_ARCH__ >= 700
        //     __nanosleep(100);
        // #endif

        // Try to reacquire the first lock
        // while (atomicCAS(&range_lock[index_1], 0, 1) != 0) {
        //     printf("loop in first lock 2, bidh: %d, offset: %d, q_block_size: %d, num_heads: %d\n", bidh, offset, q_block_size, num_heads);
        //     // #if __CUDA_ARCH__ >= 700
        //     //     __nanosleep(100);
        //     // #endif
        // }
      }
    }
  }

  CUTLASS_DEVICE
  void release_lock(int* range_lock, int bidh, int offset, int q_block_size, int num_heads) {
    // Calculate lock indices
    int block_idx1 = offset / q_block_size;
    int index_1 = block_idx1 * num_heads + bidh;

    // Check if we need to release a second lock
    int block_idx2 = (offset + q_block_size - 1) / q_block_size;
    bool has_second_lock = (block_idx1 != block_idx2);

    // Release the second lock
    if (has_second_lock) {
      int index_2 = block_idx2 * num_heads + bidh;
      atomicExch(&range_lock[index_2], 0);
    }

    // Release the first lock
    atomicExch(&range_lock[index_1], 0);
  }

  CUTLASS_DEVICE
  void deterministic_sync(int* range_lock, int bidh, int offset, int q_block_size, int num_heads, int left_range_sync_num, int right_range_sync_num) {
    if (left_range_sync_num == 0 && right_range_sync_num == 0)
      return;

    // Calculate lock index
    int left_range_block_idx = offset / q_block_size;
    int left_range_index = left_range_block_idx * num_heads + bidh;
    int right_range_block_idx = (offset + q_block_size - 1) / q_block_size;

// Acquire the first lock
#pragma unroll 1
    while (atomicCAS(&range_lock[left_range_index * 2], left_range_sync_num, left_range_sync_num) != left_range_sync_num) {
    }

    // If we need a second lock
    if (left_range_block_idx != right_range_block_idx) {
      int right_range_index = right_range_block_idx * num_heads + bidh;

// Try to acquire the second lock
#pragma unroll 1
      while (atomicCAS(&range_lock[right_range_index * 2], right_range_sync_num, right_range_sync_num) != right_range_sync_num) {
      }
    }
  }

  CUTLASS_DEVICE
  void deterministic_arrive(
      int* range_lock,
      int bidh,
      int offset,
      int q_block_size,
      int num_heads,
      int arrive_num,
      bool left_range_arrive_twice,
      bool right_range_arrive_twice) {
    // Calculate lock indices
    int left_range_block_idx = offset / q_block_size;
    int left_range_index = left_range_block_idx * num_heads + bidh;
    int right_range_block_idx = (offset + q_block_size - 1) / q_block_size;
    int right_range_index = right_range_block_idx * num_heads + bidh;

    // Release the second lock
    int add_cnt = right_range_arrive_twice ? 2 : 1;
    int tmp = atomicAdd(&range_lock[right_range_index * 2 + 1], add_cnt);
    // each range_lock needs to arrive twice to make sure conflict batch has been completed
    if (tmp + add_cnt == 2) {
      atomicExch(&range_lock[right_range_index * 2 + 1], 0);
      atomicExch(&range_lock[right_range_index * 2], arrive_num);
    }

    // Release the first lock
    add_cnt = left_range_arrive_twice ? 2 : 1;
    tmp = atomicAdd(&range_lock[left_range_index * 2 + 1], add_cnt);
    if (tmp + add_cnt == 2) {
      atomicExch(&range_lock[left_range_index * 2 + 1], 0);
      atomicExch(&range_lock[left_range_index * 2], arrive_num);
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma>
  CUTLASS_DEVICE void store(
      Params const& params,
      FrgTensorO& tOrO,
      FrgTensorLSE& lse,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      BlockCoordType const& block_coord,
      flash::DistributedSeqlenInfo& seqlen_info) {
    // Get block coordinates for current job(tile)
    int m_block = get<0>(block_coord);
    int bidh = get<1>(block_coord);
    int bidb = get<2>(block_coord);

    // Get offset and seqlen for batch that current tile belongs to
    int offset_o = seqlen_info.offset_q;
    int seqlen_o = seqlen_info.seqlen_q;

    // Get warp group index for current thread
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);

    // Define Tensors for mO, gO, sO
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh);
    Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

    // Define sO as position independent swizzle tensor
    // Tensor sO_pi = cute::as_position_independent_swizzle_tensor(sO);

    // Define Tensor for mLSE
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh);
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset_o), mLSE), select<0>(TileShape_MNK_PV{}), make_coord(m_block));

    // Make sure all WGs have finished reading V
    // Technically we don't need this if we're not using smem, but the mainloop makes the assumption that
    // all epilogue threads sync at least once during the epilogue (so that we can start loading Q with
    // cp.async if we need).
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // Step 2: Write LSE from rmem -> gmem
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // (MMA,MMA_M,MMA_K)
    Tensor taccOcO = thread_mma.partition_C(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV_Active{})));
    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);

    // (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
    Tensor taccOcO_rowcol = make_tensor(taccOcO.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(taccOcO.layout()));
    Tensor taccOcO_slice = taccOcO_rowcol(_, _0{});

    // MMA_M
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_slice));

    // Predicate for skipping correction
    // NOTE: Correction only need when we use atomic reduction
    bool skip_correction = true;
    // Define lse_prev and lse_final
    auto lse_prev = make_fragment_like(lse);
    auto lse_final = make_fragment_like(lse);

    if constexpr (!DisableFwdAtomicReduction) {
      // Acquire range lock to prevent multiple threads from writing to gmem simultaneously
      if (thread_idx == 0) {
        if constexpr (Deterministic) {
          int left_range_conflict_msg = get<3>(block_coord);
          int right_range_conflict_msg = get<4>(block_coord);
          deterministic_sync(
              params.determin_range_locks, bidh, offset_o + m_block * kBlockM, kBlockM, params.nheads, left_range_conflict_msg >> 1, right_range_conflict_msg >> 1);
        }
        acquire_lock(params.range_locks, bidh, offset_o + m_block * kBlockM, kBlockM, params.nheads);
      }
      flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

#pragma unroll
      for (int mi = 0; mi < size(lse_prev); ++mi) {
        // Load lse_prev from gmem -> smem, and calculate lse_final
        int const row_block = [&]() {
          if constexpr (!SwapAB) {
            return get<0>(taccOcO_slice(mi));
          } else {
            return get<1>(taccOcO_slice(mi));
          }
        }();
        int const row_batch = m_block * kBlockM + row_block;
        if (row_batch >= seqlen_o) {
          lse(mi) = -INFINITY;
        }

        int const row_global = row_batch + offset_o;
        if (row_global < get<0>(params.shape_O)) {
          lse_prev(mi) = gLSE(row_block);

          if (lse_prev(mi) != -INFINITY) {
            // If there is any non-inf lse_prev, we cannot skip correction
            skip_correction = false;
          }

          lse_final(mi) = correct_lse(lse_prev(mi), lse(mi));
        }
      }

      // A workaround to ensure that all threads get the correct lse_final, low performance
      flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    } else {
      // If we don't use atomic reduction, we can just use lse directly
      for (int mi = 0; mi < size(lse_final); ++mi) {
        lse_final(mi) = lse(mi);
      }
    }

// Store correct lse_final to gmem
#pragma unroll
    for (int mi = 0; mi < size(lse_final); ++mi) {
      int const row_block = [&]() {
        if constexpr (!SwapAB) {
          return get<0>(taccOcO_slice(mi));
        } else {
          return get<1>(taccOcO_slice(mi));
        }
      }();
      int const row_batch = m_block * kBlockM + row_block;
      if (row_batch < seqlen_o) {
        int const col_id = [&]() {
          if constexpr (!SwapAB) {
            return get<1>(taccOcO_slice(_0{}));
          } else {
            return get<0>(taccOcO_slice(_0{}));
          }
        }();
        // Each column is written back by one thread
        if (col_id == 0) {
          gLSE(row_block) = lse_final(mi);
        }
      }
    }

    // Define tiled copy for O
    auto tiled_copy_O = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{}, tiled_mma);
    auto thr_copy_O = tiled_copy_O.get_thread_slice(thread_idx);

    if (!skip_correction) {
      // TODO: need reduce compute for pO, and add predicate k for pO
      Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})); // (BLK_M,BLK_K) -> (blk_m,blk_k)
      Tensor pO = make_tensor<bool>(make_shape(size<0>(cO), size<1>(cO)), make_stride(_1{}, _0{}));
      int bound = get<0>(params.shape_O) - (offset_o + m_block * kBlockM);
#pragma unroll
      for (int n = 0; n < size<0>(pO); ++n) {
        pO(n, _0{}) = get<0>(cO(n, _0{})) < bound;
      }
      Tensor tOpO = [&]() {
        if constexpr (!SwapAB) {
          return thr_copy_O.partition_D(pO);
        } else {
          auto pO_transposed = make_tensor(
              pO.data(),
              cute::make_layout(
                  cute::make_shape(get<1>(pO.layout().shape()), get<0>(pO.layout().shape())),
                  cute::make_stride(get<1>(pO.layout().stride()), get<0>(pO.layout().stride()))));
          return thr_copy_O.partition_D(pO_transposed);
        }
      }();

      // Define tOrPrevO, tOrPrevO_copy_view, tOgPrevO
      Tensor tOrPrevO = make_fragment_like(tOrO);
      Tensor tOrPrevO_copy_view = thr_copy_O.retile_D(tOrPrevO);
      Tensor tOgPrevO = [&]() {
        if constexpr (!SwapAB) {
          return thr_copy_O.partition_S(gO);
        } else {
          // When SwapAB is true, transpose gO by swapping shape and stride dimensions
          auto gO_transposed = make_tensor(
              gO.data(),
              cute::make_layout(
                  cute::make_shape(get<1>(gO.layout().shape()), get<0>(gO.layout().shape())),
                  cute::make_stride(get<1>(gO.layout().stride()), get<0>(gO.layout().stride()))));
          return thr_copy_O.partition_S(gO_transposed);
        }
      }();

      // Copy prev O from gmem to smem
      cute::copy_if(tOpO, tOgPrevO, tOrPrevO_copy_view);

      // Correct output
      Tensor tOrPrevO_rowcol = make_tensor(tOrPrevO.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tOrPrevO.layout()));
      Tensor tOrO_rowcol = make_tensor(tOrO.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tOrO.layout()));

      correct_output(tOrPrevO_rowcol, tOrO_rowcol, lse_prev, lse, lse_final);
    }

    // Initialize gmem_tiled_copy_O
    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);

    // Initialize tOcO and tOpO to predict OOB access
    Tensor tOcO = gmem_thr_copy_O.partition_D(cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})));
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOcO)));
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(params.shape_O);
    }

    // Initialize tOgO to store O to gmem
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // Convert tOrO to Element type and copy to smem
    {
      Tensor tOrFinalO = make_tensor_like<Element>(tOrO);
      flash::convert_type_out(tOrO, tOrFinalO);
      Tensor tOrO_copy_view = thr_copy_O.retile_S(tOrFinalO);
      Tensor tOsO = [&]() {
        if constexpr (!SwapAB) {
          return thr_copy_O.partition_D(sO);
        } else {
          // Create transposed view of sO by swapping shape and stride
          auto sO_transposed = make_tensor(
              sO.data(),
              cute::make_layout(
                  cute::make_shape(get<1>(sO.layout().shape()), get<0>(sO.layout().shape())),
                  cute::make_stride(get<1>(sO.layout().stride()), get<0>(sO.layout().stride()))));
          return thr_copy_O.partition_D(sO_transposed);
        }
      }();
      // Tensor tOsO = thr_copy_O.partition_D(sO_pi);
      cute::copy(tiled_copy_O, tOrO_copy_view, tOsO);
      flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    // Copy tOsO to tOrFinalO
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOrFinalO = make_fragment_like(tOsO);
    cute::copy(gmem_tiled_copy_O, tOsO, tOrFinalO);

    // Signal producer threads that smem_v is released
    if constexpr (ArchTag::kMinComputeCapability >= 90) {
      cutlass::arch::fence_view_async_shared();
#pragma unroll
      for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
        shared_storage.pipelines.barrier_O.arrive(cta_id);
      }
    }

    // cutlass::arch::fence_view_async_shared();
    // flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    // int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
    // if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
    //     // cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
    //     //                                     cutlass::arch::ReservedNamedBarriers::EpilogueBarrier)
    //     if (cute::elect_one_sync()) {
    //         #pragma unroll
    //         for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
    //             shared_storage.pipelines.barrier_O.arrive(cta_id);
    //         }
    //     }
    // }

    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrFinalO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM);

    // TODO: Fix TMA store
    // BUG: The following TMA code does not handle out-of-bounds access, needs to be fixed
    // {
    //     // TODO: move the following code out of braces
    //     cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
    //     flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    //     Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh);
    //     Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
    //     auto block_tma_O = params.tma_store_O.get_slice(_0{});
    //     Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
    //     Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)

    //     int warp_idx_sync = __shfl_sync(0xffffffff, thread_idx / cutlass::NumThreadsPerWarp, 0);
    //     if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
    //         // cutlass::arch::NamedBarrier::sync(NumEpilogueThreads + cutlass::NumThreadsPerWarp,
    //         //                                     cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    //         if (cute::elect_one_sync()) {
    //             cute::copy(params.tma_store_O, tOsO, tOgO);
    //             tma_store_arrive();
    //             tma_store_wait<0>();
    //             #pragma unroll
    //             for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
    //                 shared_storage.pipelines.barrier_O.arrive(cta_id);
    //             }
    //         }
    //     }
    // }

    if constexpr (!DisableFwdAtomicReduction) {
      // Make sure all writes to global memory before this point are completed
      __threadfence();
      flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (thread_idx == 0) {
        if constexpr (Deterministic) {
          int left_range_conflict_msg = get<3>(block_coord);
          int right_range_conflict_msg = get<4>(block_coord);
          int arrive_num = get<5>(block_coord) + 1;
          deterministic_arrive(
              params.determin_range_locks,
              bidh,
              offset_o + m_block * kBlockM,
              kBlockM,
              params.nheads,
              arrive_num,
              left_range_conflict_msg & 1,
              right_range_conflict_msg & 1);
        }
        release_lock(params.range_locks, bidh, offset_o + m_block * kBlockM, kBlockM, params.nheads);
      }
    }
  }

  CUTLASS_DEVICE void store_tail() {
    // Don't need to do tma_store_wait<0>() here since we already did in @store
  }

  CUTLASS_DEVICE ElementPartial correct_lse(ElementPartial lse_prev, ElementPartial lse_curr) {
    ElementPartial max_lse = max(lse_prev, lse_curr);
    ElementPartial min_lse = min(lse_prev, lse_curr);
    ElementPartial lse = max_lse + softplus(safe_sub(min_lse, max_lse));
    return lse;
  }

  CUTLASS_DEVICE ElementPartial softplus(ElementPartial x) {
    return logf(1.f + expf(x));
  }

  CUTLASS_DEVICE ElementPartial safe_sub(ElementPartial a, ElementPartial b) {
    if (a == -INFINITY && b == -INFINITY) {
      return -INFINITY;
    }
    return a - b;
  }

  template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
  CUTLASS_DEVICE void correct_output(
      Tensor<Engine0, Layout0>& prev_output,
      Tensor<Engine0, Layout0>& curr_output,
      Tensor<Engine1, Layout1> const& prev_lse,
      Tensor<Engine1, Layout1> const& curr_lse,
      Tensor<Engine1, Layout1> const& final_lse) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(curr_output) == size<0>(prev_output));
    CUTE_STATIC_ASSERT_V(size<1>(curr_output) == size<1>(prev_output));
    CUTE_STATIC_ASSERT_V(size<0>(curr_lse) == size<0>(prev_lse));
    CUTE_STATIC_ASSERT_V(size<0>(final_lse) == size<0>(prev_lse));
    CUTE_STATIC_ASSERT_V(size<0>(curr_output) == size<0>(final_lse));
#pragma unroll
    for (int mi = 0; mi < size<0>(curr_output); ++mi) {
      ElementPartial coeff_prev = expf(safe_sub(prev_lse(mi), final_lse(mi)));
      ElementPartial coeff_curr = expf(safe_sub(curr_lse(mi), final_lse(mi)));
#pragma unroll
      for (int ni = 0; ni < size<1>(curr_output); ++ni) {
        // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
        // max * log_2(e)). This allows the compiler to use the ffma
        // instruction instead of fadd and fmul separately.
        ElementPartial prev = (coeff_prev == 0.f) ? 0.f : coeff_prev * prev_output(mi, ni);
        ElementPartial curr = (coeff_curr == 0.f) ? 0.f : coeff_curr * curr_output(mi, ni);
        curr_output(mi, ni) = prev + curr;
      }
    }
  }

  // Write 0 to output and -inf to LSE
  CUTLASS_DEVICE void store_zero(Params const& params, int thread_idx, BlockCoordType const& block_coord, flash::DistributedSeqlenInfo& seqlen_info) {
    static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
    static_assert(kBlockM <= NumEpilogueThreads);

    // Get block coordinates for current job(tile)
    int m_block = get<0>(block_coord);
    int bidh = get<1>(block_coord);
    int bidb = get<2>(block_coord);
    // Get offset and seqlen for batch that current tile belongs to
    int offset_o = seqlen_info.offset_q;

    if constexpr (!DisableFwdAtomicReduction) {
      // Acquire range lock to prevent multiple threads from writing to gmem simultaneously
      if (thread_idx == 0) {
        if constexpr (Deterministic) {
          int left_range_conflict_msg = get<3>(block_coord);
          int right_range_conflict_msg = get<4>(block_coord);
          deterministic_sync(
              params.determin_range_locks, bidh, offset_o + m_block * kBlockM, kBlockM, params.nheads, left_range_conflict_msg >> 1, right_range_conflict_msg >> 1);
        }
      }
    }

    if constexpr (!DisableFwdAtomicReduction) {
      // Make sure all writes to global memory before this point are completed
      if (thread_idx == 0) {
        if constexpr (Deterministic) {
          int left_range_conflict_msg = get<3>(block_coord);
          int right_range_conflict_msg = get<4>(block_coord);
          int arrive_num = get<5>(block_coord) + 1;
          deterministic_arrive(
              params.determin_range_locks,
              bidh,
              offset_o + m_block * kBlockM,
              kBlockM,
              params.nheads,
              arrive_num,
              left_range_conflict_msg & 1,
              right_range_conflict_msg & 1);
        }
      }
    }
    flash::named_barrier_sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
};

} // namespace flash
