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

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h> // For FastDivMod

#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "seqlen.h"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;
namespace gcd = cutlass::gemm::collective::detail;
namespace ecd = cutlass::epilogue::collective::detail;

template <
    class TileShape_MNK_PV_,
    class ClusterShape_,
    class Element_,
    class ArchTag_,
    class BlockCoordType_,
    int NumEpilogueThreads_,
    bool DisableFwdAtomicReduction_,
    bool PackGQA_,
    int QheadPerKhead_,
    bool Deterministic_ = false,
    bool SwapAB_ = false,
    bool ReturnMaxLogits_ = false>
struct CollectiveEpilogueFwd {
  using TileShape_MNK_PV = TileShape_MNK_PV_;
  using ClusterShape = ClusterShape_;
  using Element = Element_;
  using ElementPartial = float;
  using ArchTag = ArchTag_;
  using BlockCoordType = BlockCoordType_;
  using resv_barrier = cutlass::arch::ReservedNamedBarriers;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90 || CUTE_STATIC_V(size(ClusterShape{})) == 1);

  static constexpr int NumEpilogueThreads = NumEpilogueThreads_;
  static constexpr bool DisableFwdAtomicReduction = DisableFwdAtomicReduction_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr int QheadPerKhead = QheadPerKhead_; // for non packgqa, QheadPerKhead is always 1.
  static constexpr bool Deterministic = Deterministic_;
  static constexpr bool SwapAB = SwapAB_;
  static constexpr bool ReturnMaxLogits = ReturnMaxLogits_;

  static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});
  static constexpr int kHeadDim = get<1>(TileShape_MNK_PV{});
  static constexpr int kBlockN = get<2>(TileShape_MNK_PV{});

  // when SwapAB == true, set the warp group overlap tileMMA size for kBlockM
  static constexpr int TileSize_kBlockM = kBlockM;
  // TileSize_kBlockM can be set as kBlockM/2 to enable two warp-group inter overlap, but now is disable because no gain.
  // static constexpr int TileSize_kBlockM = kBlockM == 8 ? kBlockM : kBlockM / 2;

  // TileShape_MNK_SwapAB_OP_SELECT use TileSize_kBlockM as n,
  // which use in tensor core ss_op_selector for inter warp group overlap
  // (splitting short q range when SwapAB is open).
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

  // The "Row" below refers to a Head. Bytes per head
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
      decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK_PV{})), decltype(cute::get<1>(TileShape_MNK_PV{}))>());
  using SmemLayoutOTMA = decltype(tile_to_shape(SmemLayoutAtomOTMA{}, select<0, 1>(TileShape_MNK_PV{})));
  static constexpr int kSwizzle = sizeof(Element) == 4 ? 2 : (kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1)));
  static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 3 : (sizeof(Element) == 2 ? 3 : 4);
  static constexpr int kSwizzleShift = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  // when sizeof(Element) == 4, we use Swizzle<2,3,2>, otherwize we use swizzle as fa3 to avoid bank conflict
  using SmemLayoutAtomO = decltype(composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleShift>{}, Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutOSTS = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_MNK_PV{})));

  // now we don't use TMA
  // using SmemLayoutO = std::conditional_t<ArchTag::kMinComputeCapability >= 90, SmemLayoutOTMA, SmemLayoutOSTS>;
  // when SwapAB is true, SmemLayoutOTMA has no bank conflict
  using SmemLayoutO = std::conditional_t<SwapAB, SmemLayoutOTMA, SmemLayoutOSTS>;

  // (seqlen_q, d, head)
  using ShapeO = cute::Shape<int32_t, int32_t, int32_t>;
  using ShapeOPacked = std::conditional_t<!PackGQA, ShapeO, cute::Shape<cute::Shape<cute::Int<QheadPerKhead>, int32_t>, int32_t, int32_t>>; // ((QheadPerKhead,
                                                                                                                                            // seqlen_q), d, nheads_kv)
  using StrideO = cute::Stride<int64_t, _1, int64_t>;
  using StrideOPacked = std::conditional_t<!PackGQA, StrideO, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t>>;

  using ShapeLSE = cute::Shape<int32_t, int32_t>; // (seqlen_q, nheads_qo)
  using StrideLSE = cute::Stride<int64_t, _1>; // (seqlen_q, head)
  using ShapeLSEPacked =
      std::conditional_t<!PackGQA, ShapeLSE, cute::Shape<cute::Shape<cute::Int<QheadPerKhead>, int32_t>, int32_t>>; // ((QheadPerKhead, seqlen_q), nheads_kv)
  using StrideLSEPacked = std::conditional_t<!PackGQA, StrideLSE, cute::Stride<cute::Stride<int32_t, int64_t>, int32_t>>;

  using CopyOpR2S = std::conditional_t<
      ArchTag::kMinComputeCapability >= 90,
      // cute::SM90_U32x4_STSM_N if Element size is 2 bytes (fp16, bf16)
      decltype(ecd::sm90_get_smem_store_op_for_accumulator<StrideO, ElementPartial>()),
      AutoVectorizingCopyWithAssumedAlignment<128>>;

  // static constexpr size_t SmemAlignmentO = cutlass::detail::alignment_for_swizzle(SmemLayoutO{});
  // static_assert(SmemAlignmentO >= 128, "Require at least 128B alignment");
  // struct TensorStorage : cute::aligned_struct<SmemAlignmentO> {
  //     cute::array_aligned<Element, Use_smem ? cute::cosize_v<SmemLayoutO> : 0, SmemAlignmentO> smem_o;
  // };
  static constexpr int NumMaxLogits = 128;
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
    float* ptr_max_logits = nullptr;
    float softmax_scale = 1.0f;
  };

  // Device side kernel params
  struct Params {
    Element* ptr_O;
    ShapeO const shape_O;
    ShapeOPacked const shape_O_packed;
    StrideO const stride_O;
    StrideOPacked const stride_O_packed;
    float* ptr_LSE;
    ShapeLSE const shape_LSE;
    ShapeLSEPacked const shape_LSE_packed;
    StrideLSE const stride_LSE;
    StrideLSEPacked const stride_LSE_packed;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_O tma_store_O;
    int const nheads;
    int const nheads_kv;
    int* range_locks = nullptr;
    int2 const* q_ranges = nullptr;
    int2 const* k_ranges = nullptr;
    int* determin_range_locks = nullptr;
    float* ptr_max_logits = nullptr;
    float softmax_scale = 1.0f;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
    TMA_O tma_store_O = make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutOTMA{}, select<0, 1>(TileShape_MNK_PV{}), _1{}); // no mcast

    // seqlen_q, num_heads_qo
    auto const shape_LSE = select<0, 2>(args.shape_O);
    auto const shape_LSE_packed =
        cute::conditional_return<!PackGQA>(shape_LSE, make_shape(make_shape(cute::Int<QheadPerKhead>{}, get<0>(args.shape_O)), args.nheads_kv));

    auto const stride_LSE_packed = cute::conditional_return<!PackGQA>(args.stride_LSE, make_stride(make_stride(1, get<0>(args.stride_LSE)), QheadPerKhead));

    auto const shape_O_packed = cute::conditional_return<!PackGQA>(
        args.shape_O, make_shape(make_shape(cute::Int<QheadPerKhead>{}, get<0>(args.shape_O)), get<1>(args.shape_O), args.nheads_kv));
    auto const stride_O_packed = cute::conditional_return<!PackGQA>(
        args.stride_O, make_stride(make_stride(get<2>(args.stride_O), get<0>(args.stride_O)), get<1>(args.stride_O), get<2>(args.stride_O) * QheadPerKhead));

    return {
        args.ptr_O,
        args.shape_O,
        shape_O_packed,
        args.stride_O,
        stride_O_packed,
        args.ptr_LSE,
        shape_LSE,
        shape_LSE_packed,
        args.stride_LSE,
        stride_LSE_packed,
        cutlass::FastDivmod(QheadPerKhead),
        tma_store_O,
        args.nheads,
        args.nheads_kv,
        args.range_locks,
        args.q_ranges,
        args.k_ranges,
        args.determin_range_locks,
        args.ptr_max_logits,
        args.softmax_scale};
  }

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
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

#pragma unroll 1
      // Try to acquire the second lock
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

#pragma unroll 1
    // Acquire the first lock
    while (atomicCAS(&range_lock[left_range_index * 2], left_range_sync_num, left_range_sync_num) != left_range_sync_num) {
    }

    // If we need a second lock
    if (left_range_block_idx != right_range_block_idx) {
      int right_range_index = right_range_block_idx * num_heads + bidh;

#pragma unroll 1
      // Try to acquire the second lock
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

  template <typename SharedStorage, typename FrgTensorO, typename FrgTensorLSE, typename TiledMma, typename... Args>
  CUTLASS_DEVICE void store(
      Params const& params,
      FrgTensorO& tOrO,
      FrgTensorLSE& lse,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      BlockCoordType const& block_coord,
      flash::DistributedSeqlenInfo& seqlen_info,
      Args&&... args) {
    // Get block coordinates for current job(tile)
    int m_block = get<0>(block_coord);
    int bidh = get<1>(block_coord);
    int bidb = get<2>(block_coord);

    // Get offset and seqlen for batch that current tile belongs to
    int offset_o = seqlen_info.offset_q;
    int seqlen_o = seqlen_info.seqlen_q;

    // Get warp group index for current thread
    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);

    // Define Tensors for mO, gO, sO
    Tensor mO = make_tensor(make_gmem_ptr(params.ptr_O), params.shape_O, params.stride_O)(_, _, bidh);
    Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
    Tensor sO = make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_o.data()), SmemLayoutO{});

    auto mOPacked = [&]() {
      if constexpr (!PackGQA) {
        return mO;
      } else {
        return make_tensor(make_gmem_ptr(params.ptr_O), make_layout(params.shape_O_packed, params.stride_O_packed))(_, _, bidh);
      }
    }();

    auto gOPacked = [&]() {
      if constexpr (!PackGQA) {
        return gO;
      } else {
        return local_tile(cute::domain_offset(make_coord(offset_o * QheadPerKhead, _0{}), mOPacked), select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
      }
    }();

    // Define sO as position independent swizzle tensor
    // Tensor sO_pi = cute::as_position_independent_swizzle_tensor(sO);

    // Define Tensor for mLSE
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE), params.shape_LSE, params.stride_LSE)(_, bidh);

    auto mLSEPacked = [&]() {
      if constexpr (!PackGQA) {
        return mLSE;
      } else {
        return make_tensor(make_gmem_ptr(params.ptr_LSE), make_layout(params.shape_LSE_packed, params.stride_LSE_packed))(_, bidh);
      }
    }();

    Tensor gLSE = local_tile(cute::domain_offset(make_coord(offset_o), mLSE), select<0>(TileShape_MNK_PV{}), make_coord(m_block));
    Tensor gLSEPacked = [&]() {
      if constexpr (!PackGQA) {
        return gLSE;
      } else {
        return local_tile(cute::domain_offset(make_coord(offset_o * QheadPerKhead), mLSEPacked), select<0>(TileShape_MNK_PV{}), make_coord(m_block));
      }
    }();

    // Make sure all WGs have finished reading V
    // Technically we don't need this if we're not using smem, but the mainloop makes the assumption that
    // all epilogue threads sync at least once during the epilogue (so that we can start loading Q with
    // cp.async if we need).
    BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

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
              params.determin_range_locks,
              bidh,
              offset_o * QheadPerKhead + m_block * kBlockM,
              kBlockM,
              !PackGQA ? params.nheads : params.nheads_kv,
              left_range_conflict_msg >> 1,
              right_range_conflict_msg >> 1);
        }
        acquire_lock(params.range_locks, bidh, offset_o * QheadPerKhead + m_block * kBlockM, kBlockM, !PackGQA ? params.nheads : params.nheads_kv);
      }
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

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
        if (row_batch >= seqlen_o * QheadPerKhead) {
          lse(mi) = -INFINITY;
        }

        int const row_global = row_batch + offset_o * QheadPerKhead;
        if (row_global < get<0>(params.shape_O) * QheadPerKhead) {
          if constexpr (!PackGQA) {
            lse_prev(mi) = gLSE(row_block);
          } else {
            lse_prev(mi) = gLSEPacked(row_block);
          }
          if (lse_prev(mi) != -INFINITY) {
            // If there is any non-inf lse_prev, we cannot skip correction
            skip_correction = false;
          }

          lse_final(mi) = correct_lse(lse_prev(mi), lse(mi));
        }
      }

      // A workaround to ensure that all threads get the correct lse_final, low performance
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
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

      if (row_batch < seqlen_o * QheadPerKhead) {
        int const col_id = [&]() {
          if constexpr (!SwapAB) {
            return get<1>(taccOcO_slice(_0{}));
          } else {
            return get<0>(taccOcO_slice(_0{}));
          }
        }();
        if (col_id == 0) {
          if constexpr (!PackGQA)
            gLSE(row_block) = lse_final(mi);
          else
            gLSEPacked(row_block) = lse_final(mi);
        }
      }
    }

    // Define tiled copy for O
    auto tiled_copy_O = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementPartial>{}, tiled_mma);
    auto thr_copy_O = tiled_copy_O.get_thread_slice(thread_idx);

    if (!__all_sync(0xFFFFFFFF, skip_correction)) {
      // Create identity tensor for coordinate mapping: (BLK_M, BLK_K) -> (m, k)
      // This maps the tile shape to logical indices.
      Tensor cO = cute::make_identity_tensor(select<0, 1>(TileShape_MNK_PV{})); // (BLK_M,BLK_K) -> (blk_m,blk_k)
      // TODO: add predicate k for pO
      // Initialize predicate tensor pO.
      // Stride is (_1, _0), meaning it varies along dimension 0 (M) but is broadcasted along dimension 1 (K).
      Tensor pO = make_tensor<bool>(make_shape(size<0>(cO), size<1>(cO)), make_stride(_1{}, _0{}));

      Tensor tOpO = [&]() {
        // Calculate the boundary limit for the M dimension (sequence length check).
        // bound = global_M_limit - current_block_offset
        int bound = get<0>(params.shape_O) * QheadPerKhead - (offset_o * QheadPerKhead + m_block * kBlockM);

        if constexpr (!SwapAB) {
          // ----------------------------------------------------------------
          // Case 1: Standard Layout (!SwapAB)
          // ----------------------------------------------------------------

          // Partition coordinates and predicates across threads
          Tensor tOcO = thr_copy_O.partition_D(cO);
          Tensor tOpO = thr_copy_O.partition_D(pO); // Shadowing intended for local scope calculation

#pragma unroll
          for (int mi = 0; mi < size<0, 1, 1>(tOpO); ++mi) {
            // 1. Check if the row index (M) is within the valid boundary.
            // 2. Check if the global LSE for this row is valid (not -inf), ensuring we don't process padded/masked tokens.
            auto coord = make_coord(make_coord(_0{}, make_coord(_0{}, mi, _0{})), _0{}, _0{});
            tOpO(coord) = get<0>(tOcO(coord)) < bound && gLSE(get<0>(tOcO(coord))) != -INFINITY;
          }

          /* DEBUG */
          // if (bidh == 0 && thread_idx == 128) {
          //   printf("thread_idx: %d, warp_group_idx: %d, m_block: %d, bidb: %d, bound: %d, physic_bound: %d, size<0, 1, 1>(tOpO): %lld\n", thread_idx, warp_group_idx,
          //   m_block, bidb, bound, bound + offset_o * QheadPerKhead + m_block * kBlockM, size<0, 1, 1>(tOpO)); printf("============================================
          //   tOcO m_block: %d bidb: %d ==============================\n", m_block, bidb); print_tensor(tOcO); printf("============================================
          //   tOpO m_block: %d bidb: %d ==============================\n", m_block, bidb); print_tensor(tOpO);
          // }

          return tOpO;

        } else {
          // ----------------------------------------------------------------
          // Case 2: Transposed Layout (SwapAB)
          // ----------------------------------------------------------------
          // When SwapAB is true, we need to transpose the coordinate and predicate tensors
          // before partitioning so that the thread mapping aligns correctly with the swapped axes.

          // Transpose cO: (M, K) -> (K, M) layout
          auto tO_transposed = make_tensor(
              cute::make_layout(
                  cute::make_shape(get<1>(cO.layout().shape()), get<0>(cO.layout().shape())),
                  cute::make_stride(get<1>(cO.layout().stride()), get<0>(cO.layout().stride()))));

          // Transpose pO similarly
          auto pO_transposed = make_tensor(
              pO.data(),
              cute::make_layout(
                  cute::make_shape(get<1>(pO.layout().shape()), get<0>(pO.layout().shape())),
                  cute::make_stride(get<1>(pO.layout().stride()), get<0>(pO.layout().stride()))));

          // Partition the transposed tensors
          Tensor tOcO_transposed = thr_copy_O.partition_D(tO_transposed);
          Tensor tOpO_transposed = thr_copy_O.partition_D(pO_transposed);

#pragma unroll
          for (int mi = 0; mi < size(tOpO_transposed); ++mi) {
            // Note: After transpose, get<1> corresponds to the original M dimension (row),
            // and get<0> corresponds to the K dimension (or N depending on context).
            tOpO_transposed(mi) = get<1>(tOcO_transposed(mi)) < bound && gLSE(get<0>(tOcO_transposed(mi))) != -INFINITY;
          }

          return tOpO_transposed;
        }
      }();

      // Define tOrPrevO, tOrPrevO_copy_view, tOgPrevO
      Tensor tOrPrevO = make_fragment_like(tOrO);
      Tensor tOrPrevO_copy_view = thr_copy_O.retile_D(tOrPrevO);
      auto tOgPrevO = [&]() {
        if constexpr (!SwapAB) {
          if constexpr (!PackGQA) {
            return thr_copy_O.partition_S(gO);
          } else {
            return thr_copy_O.partition_S(gOPacked);
          }
        } else {
          // When SwapAB is true, transpose gO by swapping shape and stride dimensions
          if constexpr (!PackGQA) {
            auto gO_transposed = make_tensor(
                gO.data(),
                cute::make_layout(
                    cute::make_shape(get<1>(gO.layout().shape()), get<0>(gO.layout().shape())),
                    cute::make_stride(get<1>(gO.layout().stride()), get<0>(gO.layout().stride()))));
            return thr_copy_O.partition_S(gO_transposed);
          } else {
            auto gOPacked_transposed = make_tensor(
                gOPacked.data(),
                cute::make_layout(
                    cute::make_shape(get<1>(gOPacked.layout().shape()), get<0>(gOPacked.layout().shape())),
                    cute::make_stride(get<1>(gOPacked.layout().stride()), get<0>(gOPacked.layout().stride()))));
            return thr_copy_O.partition_S(gOPacked_transposed);
          }
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
    Tensor tOgOPacked = gmem_thr_copy_O.partition_D(gOPacked);

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
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
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
    // BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
    // int warp_idx_sync = warp_uniform(thread_idx / cutlass::NumThreadsPerWarp);
    // if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
    //    // BarrierManager::sync<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);
    //     if (cute::elect_one_sync()) {
    //         #pragma unroll
    //         for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
    //             shared_storage.pipelines.barrier_O.arrive(cta_id);
    //         }
    //     }
    // }

    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    if constexpr (!PackGQA) {
      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_O, tOrFinalO, tOgO, tOcO, tOpO, seqlen_o - m_block * kBlockM);
    } else {
      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(      // the QheadPerKhead dim of  tOgOPacked should be known at compile time.
          gmem_tiled_copy_O, tOrFinalO, tOgOPacked, tOcO, tOpO, seqlen_o * QheadPerKhead - m_block * kBlockM);
    }

    // TODO: Fix TMA store
    // BUG: The following TMA code does not handle out-of-bounds access, needs to be fixed
    // {
    //     // TODO: move the following code out of braces
    //     cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
    //     BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

    //     Tensor mO = params.tma_store_O.get_tma_tensor(params.shape_O)(_, _, bidh);
    //     Tensor gO = local_tile(cute::domain_offset(make_coord(offset_o, _0{}), mO), select<0, 1>(TileShape_MNK_PV{}), make_coord(m_block, _0{}));
    //     auto block_tma_O = params.tma_store_O.get_slice(_0{});
    //     Tensor tOgO = block_tma_O.partition_D(gO);  // (TMA, TMA_M, TMA_K)
    //     Tensor tOsO = block_tma_O.partition_S(sO); // (TMA, TMA_M, TMA_K)

    //     int warp_idx_sync = warp_uniform(thread_idx / cutlass::NumThreadsPerWarp);
    //     if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
    //        // BarrierManager::sync<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);
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
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
      if (thread_idx == 0) {
        if constexpr (Deterministic) {
          int left_range_conflict_msg = get<3>(block_coord);
          int right_range_conflict_msg = get<4>(block_coord);
          int arrive_num = get<5>(block_coord) + 1;

          deterministic_arrive(
              params.determin_range_locks,
              bidh,
              offset_o * QheadPerKhead + m_block * kBlockM,
              kBlockM,
              !PackGQA ? params.nheads : params.nheads_kv,
              arrive_num,
              left_range_conflict_msg & 1,
              right_range_conflict_msg & 1);
        }
        release_lock(params.range_locks, bidh, offset_o * QheadPerKhead + m_block * kBlockM, kBlockM, !PackGQA ? params.nheads : params.nheads_kv);
      }
    }

    if constexpr (ReturnMaxLogits) {
      auto row_max = cute::get<0>(cute::make_tuple(std::forward<Args>(args)...));
      if constexpr (!PackGQA) {
        // no PackGQA, update max_logits for one head
        float thread_max = -INFINITY;
#pragma unroll
        for (int mi = 0; mi < size(row_max); ++mi) {
          int const row_block = [&]() {
            if constexpr (!SwapAB) {
              return get<0>(taccOcO_slice(mi));
            } else {
              return get<1>(taccOcO_slice(mi));
            }
          }();
          int const row_batch = m_block * kBlockM + row_block;
          if (row_batch < seqlen_o) {
            thread_max = max(thread_max, row_max(mi));
          }
        }

        // Warp reduce
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          thread_max = max(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));
        }

        if ((thread_idx % 32) == 0) {
          atomicMaxFloatOnlyIncrease(&shared_storage.tensors.smem_max_logits[bidh], thread_max);
        }
      } else {
        // PackGQA, flatten q_head_per_khead in seqlen dim
        // update multiple q_heads in a single store
        for (int mi = 0; mi < size(row_max); ++mi) {
          int const row_block = [&]() {
            if constexpr (!SwapAB) {
              return get<0>(taccOcO_slice(mi));
            } else {
              return get<1>(taccOcO_slice(mi));
            }
          }();
          int const row_batch = m_block * kBlockM + row_block;
          if (row_batch < seqlen_o * Qhead_per_khead) {
            // PackGQA qhead is contiguous, calculate the qhead index for the current row
            int const qhead_idx = bidh * Qhead_per_khead + row_block % Qhead_per_khead;
            atomicMaxFloatOnlyIncrease(&shared_storage.tensors.smem_max_logits[qhead_idx], row_max(mi));
          }
        }
      }
    }
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void store_tail(Params const& params, SharedStorage& shared_storage, int thread_idx) {
    if constexpr (ReturnMaxLogits) {
      // Ensure all threads have finished their atomic updates to shmem_max_logit
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

      // Use a loop to ensure all nheads are processed, even if NumEpilogueThreads < nheads
      for (int h = thread_idx; h < params.nheads; h += NumEpilogueThreads) {
        float block_max = shared_storage.tensors.smem_max_logits[h];
        if (block_max != -INFINITY) {
          atomicMaxFloatOnlyIncrease(static_cast<float*>(params.ptr_max_logits) + h, block_max * params.softmax_scale);
        }
      }
    }
    // Don't need to do tma_store_wait<0>() here since we already did in @store
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
      ElementPartial coeff_prev = calc_lse_rescale_weight(prev_lse(mi), final_lse(mi));
      ElementPartial coeff_curr = calc_lse_rescale_weight(curr_lse(mi), final_lse(mi));
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
              params.determin_range_locks,
              bidh,
              offset_o * QheadPerKhead + m_block * kBlockM,
              kBlockM,
              !PackGQA ? params.nheads : params.nheads_kv,
              left_range_conflict_msg >> 1,
              right_range_conflict_msg >> 1);
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
              offset_o * QheadPerKhead + m_block * kBlockM,
              kBlockM,
              !PackGQA ? params.nheads : params.nheads_kv,
              arrive_num,
              left_range_conflict_msg & 1,
              right_range_conflict_msg & 1);
        }
      }
    }
    BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
  }
};

} // namespace flash
