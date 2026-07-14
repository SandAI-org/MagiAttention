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
#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "deterministic.h"
#include "inner_ldst_mode.hpp"
#include "named_barrier.hpp"
#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;
namespace gcd = cutlass::gemm::collective::detail;

template <
    class TileShape_MNK_,
    class ElementDq_,
    class ElementDkv_,
    class ElementAccum_,
    class ArchTag_,
    class BlockCoordType_,
    bool dQ_swapAB_,
    bool dKV_swapAB_,
    int NumConsumerWarpGroups,
    int AtomLayoutMdQ,
    int AtomLayoutNdKV,
    bool OuterStoreNeedReduction_,
    int OuterStoreMode_,
    bool Deterministic_,
    bool BwdInnerLoopK_,
    bool PackGQA_,
    bool CatGQA_,
    int PackGQAFactor_,
    bool IndexSparse_,
    int SparseKBlockSize_>
struct CollectiveEpilogueBwd {
  using TileShape_MNK = TileShape_MNK_;
  using ElementDq = ElementDq_;
  using ElementDkv = ElementDkv_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  using BlockCoordType = BlockCoordType_;
  using SeqlenInfo_t = flash::SeqlenInfo;
  using resv_barrier = cutlass::arch::ReservedNamedBarriers;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr bool IsSameTypeDq = cute::is_same_v<ElementDq, ElementAccum>;
  static constexpr bool IsSameTypeDkv = cute::is_same_v<ElementDkv, ElementAccum>;
  static constexpr bool OuterStoreNeedReduction = OuterStoreNeedReduction_;
  static constexpr OuterStoreMode kOuterStoreMode = static_cast<OuterStoreMode>(OuterStoreMode_);

  static constexpr bool dQ_swapAB = dQ_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool Use_TMA = true; // SM90 only
  static constexpr bool Deterministic = Deterministic_;
  static constexpr bool BwdInnerLoopK = BwdInnerLoopK_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool CatGQA = CatGQA_;
  static constexpr bool FlattenGQA = PackGQA_ || CatGQA_;
  static constexpr int PackGQAFactor = PackGQAFactor_; // for non packgqa, PackGQAFactor is always 1.
  static constexpr bool IndexSparse = IndexSparse_;
  static constexpr int SparseKBlockSize = SparseKBlockSize_;
  // IndexSparse LoopQ: outer block = 1 K token in a kBlockN tile; only row 0 is valid.

  static constexpr int NumEpilogueThreads = NumConsumerWarpGroups * cutlass::NumThreadsPerWarpGroup;
  static constexpr int AtomLayoutMdKV = NumConsumerWarpGroups * (Use_TMA ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV;
  static constexpr int AtomLayoutNdQ = NumConsumerWarpGroups * (Use_TMA ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutMdQ;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  // TMA type for dQ: only used when OuterStoreNeedReduction=true (atomic reduce-add path).
  // When OuterStoreNeedReduction=false, store_dq() uses per-element flash::copy instead.
  using GmemTiledCopydQTMA = cute::SM90_TMA_REDUCE_ADD;
  using GmemTiledCopydKVTMA = std::conditional_t<!BwdInnerLoopK && !OuterStoreNeedReduction, cute::SM90_TMA_STORE, cute::SM90_TMA_REDUCE_ADD>;
  using BwdNamedBarriers = std::conditional_t<BwdInnerLoopK, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;
  static_assert(BarrierManager::check<BwdNamedBarriers, NumConsumerWarpGroups>());

  // These are for storing the output tensor without TMA (e.g., for setting output to zero)
  static constexpr int kGmemElemsPerLoadDq = sizeof(cute::uint128_t) / sizeof(ElementDq);
  static_assert(kHeadDim % kGmemElemsPerLoadDq == 0, "Headdim must be a multiple of kGmemElemsPerLoadDq");
  static constexpr int kGmemThreadsPerRowDq = cutlass::gcd(kHeadDim / kGmemElemsPerLoadDq, NumEpilogueThreads);
  static_assert(NumEpilogueThreads % kGmemThreadsPerRowDq == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRowDq");

  static constexpr int kGmemElemsPerLoadDkv = sizeof(cute::uint128_t) / sizeof(ElementDkv);
  static_assert(kHeadDim % kGmemElemsPerLoadDkv == 0, "Headdim must be a multiple of kGmemElemsPerLoadDkv");
  static constexpr int kGmemThreadsPerRowDkv = cutlass::gcd(kHeadDim / kGmemElemsPerLoadDkv, NumEpilogueThreads);
  static_assert(NumEpilogueThreads % kGmemThreadsPerRowDkv == 0, "NumEpilogueThreads must be a multiple of kGmemThreadsPerRowDkv");

  using GmemLayoutAtomDq = Layout<Shape<Int<NumEpilogueThreads / kGmemThreadsPerRowDq>, Int<kGmemThreadsPerRowDq>>, Stride<Int<kGmemThreadsPerRowDq>, _1>>;
  using GmemTiledCopydQ = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementDq>{},
      GmemLayoutAtomDq{},
      Layout<Shape<_1, Int<kGmemElemsPerLoadDq>>>{})); // Val layout, 8 or 16 vals per store

  using GmemLayoutAtomDkv = Layout<Shape<Int<NumEpilogueThreads / kGmemThreadsPerRowDkv>, Int<kGmemThreadsPerRowDkv>>, Stride<Int<kGmemThreadsPerRowDkv>, _1>>;
  using GmemTiledCopydKV = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementDkv>{},
      GmemLayoutAtomDkv{},
      Layout<Shape<_1, Int<kGmemElemsPerLoadDkv>>>{})); // Val layout, 8 or 16 vals per store

  using SmemLayoutAtomdQTMA = decltype(gcd::ss_smem_selector<
                                       GMMA::Major::K,
                                       ElementDq,
                                       // TODO: do we have to change this if dQ_swapAB is true?
                                       Int<kBlockM>,
                                       Int<kHeadDim / AtomLayoutNdQ>>());
  using SmemLayoutdQTMA = decltype(tile_to_shape(SmemLayoutAtomdQTMA{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutdQtTMA = decltype(cute::composition(SmemLayoutdQTMA{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));

  using SmemLayoutAtomdKVTMA = decltype(gcd::ss_smem_selector<
                                        GMMA::Major::K,
                                        ElementDkv,
                                        // TODO: do we have to change this if dKV_swapAB is true?
                                        Int<kBlockN>,
                                        Int<kHeadDim / AtomLayoutMdKV>>());
  using SmemLayoutdKVTMA = decltype(tile_to_shape(SmemLayoutAtomdKVTMA{}, select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVtTMA =
      decltype(cute::composition(SmemLayoutdKVTMA{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));

  using SmemLayoutAtomdQ = SmemLayoutAtomdQTMA;
  using SmemLayoutdQ = decltype(tile_to_shape(SmemLayoutAtomdQ{}, select<0, 2>(TileShape_MNK{})));
  using SmemLayoutdQt = decltype(cute::composition(SmemLayoutdQ{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));

  using SmemLayoutAtomdKV = SmemLayoutAtomdKVTMA;
  using SmemLayoutdKV = decltype(tile_to_shape(SmemLayoutAtomdKV{}, select<1, 2>(TileShape_MNK{})));
  using SmemLayoutdKVt = decltype(cute::composition(SmemLayoutdKV{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));

  using SmemCopyAtomdQ = Copy_Atom<cute::DefaultCopy, ElementDq>;
  using SmemCopyAtomdKV = Copy_Atom<cute::DefaultCopy, ElementDkv>;

  static constexpr size_t SmemAlignmentdQ = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdQ{}) : 128;
  static_assert(SmemAlignmentdQ >= 128, "Require at least 128B alignment");

  static constexpr size_t SmemAlignmentdKV = ArchTag::kMinComputeCapability >= 90 ? cutlass::detail::alignment_for_swizzle(SmemLayoutdKV{}) : 128;
  static_assert(SmemAlignmentdKV >= 128, "Require at least 128B alignment");

  struct TensorStorageLoopQ : cute::aligned_struct<SmemAlignmentdKV> {
    cute::array_aligned<ElementDkv, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dk;
    cute::array_aligned<ElementDkv, cute::cosize_v<SmemLayoutdKV>, SmemAlignmentdKV> smem_dv;
  };

  struct TensorStorageLoopK : cute::aligned_struct<SmemAlignmentdQ> {
    cute::array_aligned<ElementDq, cute::cosize_v<SmemLayoutdQ>, SmemAlignmentdQ> smem_dq;
  };

  using TensorStorage = std::conditional_t<BwdInnerLoopK, TensorStorageLoopK, TensorStorageLoopQ>;

  using ShapedQKV = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen, head_dim, num_heads)
  using StridedQKV = cute::Stride<int64_t, _1, int64_t>;

  // Packed shape/stride for dQ when PackGQA is enabled
  // ((PackGQAFactor, seqlen_q), head_dim, nheads_kv)
  using ShapedQPacked = std::conditional_t<!PackGQA, ShapedQKV, cute::Shape<cute::Shape<cute::Int<PackGQAFactor>, int32_t>, int32_t, int32_t>>;
  using StridedQPacked = std::conditional_t<!PackGQA, StridedQKV, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t>>;

  // Compile-time-selected dQ store layout/TMA (PackGQA picks packed variants).
  using ShapeDqStore = ShapedQPacked;
  using StrideDqStore = StridedQPacked;

  using TMA_dQ = std::conditional_t<
      Use_TMA,
      decltype(make_tma_copy(
          GmemTiledCopydQTMA{},
          make_tensor(make_gmem_ptr(static_cast<ElementDq*>(nullptr)), ShapedQKV{}, StridedQKV{}),
          SmemLayoutdQTMA{},
          select<0, 2>(TileShape_MNK{}),
          _1{})), // no mcast for dQ
      std::nullptr_t>;

  // Packed TMA for dQ when PackGQA is enabled
  using TMA_dQ_Packed = std::conditional_t<
      Use_TMA && PackGQA,
      decltype(make_tma_copy(
          GmemTiledCopydQTMA{},
          make_tensor(make_gmem_ptr(static_cast<ElementDq*>(nullptr)), ShapedQPacked{}, StridedQPacked{}),
          SmemLayoutdQTMA{},
          select<0, 2>(TileShape_MNK{}),
          _1{})), // no mcast for packed dQ
      std::nullptr_t>;

  using TMA_dQ_Store = std::conditional_t<PackGQA, TMA_dQ_Packed, TMA_dQ>;

  using TMA_dKV = std::conditional_t<
      Use_TMA,
      decltype(make_tma_copy(
          GmemTiledCopydKVTMA{},
          make_tensor(make_gmem_ptr(static_cast<ElementDkv*>(nullptr)), ShapedQKV{}, StridedQKV{}),
          SmemLayoutdKVTMA{},
          select<1, 2>(TileShape_MNK{}),
          _1{})), // no mcast for dKV
      std::nullptr_t>;

  // Host side kernel arguments
  struct Arguments {
    ElementDq* ptr_dQ; // q for outer-loop and k for inner-loop
    ShapedQKV const shape_dQ;
    StridedQKV const stride_dQ;
    ElementDkv* ptr_dK; // k for outer-loop and q for inner-loop
    ShapedQKV const shape_dK;
    StridedQKV const stride_dK;
    ElementDkv* ptr_dV; // k for outer-loop and q for inner-loop
    ShapedQKV const shape_dV;
    StridedQKV const stride_dV;
    int const num_heads_q;
    int const num_heads_kv;
    int2 const* q_ranges;
    int2 const* k_ranges;
    int* determin_range_locks = nullptr;
  };

  // Device side kernel params
  struct Params {
    ElementDq* ptr_dQ; // q for outer-loop and k for inner-loop
    ShapeDqStore const shape_dQ;
    StrideDqStore const stride_dQ;
    ElementDkv* ptr_dK; // k for outer-loop and q for inner-loop
    ShapedQKV const shape_dK;
    StridedQKV const stride_dK;
    ElementDkv* ptr_dV; // k for outer-loop and q for inner-loop
    ShapedQKV const shape_dV;
    StridedQKV const stride_dV;
    TMA_dQ_Store tma_store_dQ; // q for outer-loop and k for inner-loop
    TMA_dKV tma_store_dK; // k for outer-loop and q for inner-loop
    TMA_dKV tma_store_dV; // k for outer-loop and q for inner-loop
    int2 const* q_ranges;
    int2 const* k_ranges;
    cutlass::FastDivmod qhead_per_khead_divmod;
    int const nheads;
    int* determin_range_locks = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // shape_dQ/stride_dQ: packed layout when PackGQA, plain layout otherwise.
    auto const shape_dQ = cute::conditional_return<!PackGQA>(
        args.shape_dQ, make_shape(make_shape(cute::Int<PackGQAFactor>{}, get<0>(args.shape_dQ)), get<1>(args.shape_dQ), args.num_heads_kv));
    auto const stride_dQ = cute::conditional_return<!PackGQA>(
        args.stride_dQ, make_stride(make_stride(get<2>(args.stride_dQ), get<0>(args.stride_dQ)), get<1>(args.stride_dQ), get<2>(args.stride_dQ) * PackGQAFactor));

    Tensor mdQ = make_tensor(make_gmem_ptr(args.ptr_dQ), shape_dQ, stride_dQ);
    Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
    Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);

    TMA_dQ_Store tma_store_dQ = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(GmemTiledCopydQTMA{}, mdQ, SmemLayoutdQTMA{}, select<0, 2>(TileShape_MNK{}), _1{});
      } else {
        return nullptr;
      }
    }();
    TMA_dKV tma_store_dK = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(GmemTiledCopydKVTMA{}, mdK, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{});
      } else {
        return nullptr;
      }
    }();
    TMA_dKV tma_store_dV = [&] {
      if constexpr (Use_TMA) {
        return make_tma_copy(GmemTiledCopydKVTMA{}, mdV, SmemLayoutdKVTMA{}, select<1, 2>(TileShape_MNK{}), _1{});
      } else {
        return nullptr;
      }
    }();

    return {
        args.ptr_dQ,
        shape_dQ,
        stride_dQ,
        args.ptr_dK,
        args.shape_dK,
        args.stride_dK,
        args.ptr_dV,
        args.shape_dV,
        args.stride_dV,
        tma_store_dQ,
        tma_store_dK,
        tma_store_dV,
        args.q_ranges,
        args.k_ranges,
        /*qhead_per_khead_divmod=*/cutlass::FastDivmod(cute::ceil_div(args.num_heads_q, get<2>(args.shape_dK))),
        args.num_heads_kv,
        args.determin_range_locks};
  }

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA) {
      if constexpr (BwdInnerLoopK) {
        if constexpr (kOuterStoreMode == OuterStoreMode::Tma) {
          cute::prefetch_tma_descriptor(params.tma_store_dQ.get_tma_descriptor());
        }
      } else {
        if constexpr (kOuterStoreMode == OuterStoreMode::Tma) {
          cute::prefetch_tma_descriptor(params.tma_store_dK.get_tma_descriptor());
          cute::prefetch_tma_descriptor(params.tma_store_dV.get_tma_descriptor());
        }
      }
    }
  }

  // Perform a Consumer Epilogue -- TMA store for dK and dV
  // k for outer-loop and q for inner-loop
  template <typename SharedStorage, typename FrgTensorO, typename TiledMma, typename DetMsgT = cute::tuple<>>
  CUTLASS_DEVICE void store_dkv(
      Params const& params,
      FrgTensorO const& tdKrdK,
      FrgTensorO const& tdVrdV,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      BlockCoordType const& block_coord,
      DetMsgT const& det_msg = {}) {
    static_assert(!BwdInnerLoopK, "store_dkv() must be called when BwdInnerLoopK is false");

    // Get block coordinates for current job (tile)
    int n_block = get<0>(block_coord), bidh = get<1>(block_coord), bidb = get<2>(block_coord);

    int bidh_idx_in_group;
    int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);

    bidh_kv = cute::conditional_return<!FlattenGQA>(params.qhead_per_khead_divmod.div(bidh), bidh);
    bidh_idx_in_group = cute::conditional_return<!FlattenGQA>(params.qhead_per_khead_divmod.rem(bidh), 0);
    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKV{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKV{}));
    Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dk.data()), SmemLayoutdKVt{}));
    Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dv.data()), SmemLayoutdKVt{}));
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomdKV{}, tiled_mma);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(thread_idx);

    // Convert the type of tdVrdV and tdKrdK to ElementDkv if they are not the same as ElementAccum
    Tensor tdVrdV_out = [&] {
      if constexpr (IsSameTypeDkv) {
        return tdVrdV;
      } else {
        auto out = make_tensor_like<ElementDkv>(tdVrdV);
        flash::convert_type_out(tdVrdV, out);
        return out;
      }
    }();

    Tensor tdKrdK_out = [&] {
      if constexpr (IsSameTypeDkv) {
        return tdKrdK;
      } else {
        auto out = make_tensor_like<ElementDkv>(tdKrdK);
        flash::convert_type_out(tdKrdK, out);
        return out;
      }
    }();

    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(tdKrdK_out); // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(tdVrdV_out); // ((Atom,AtomNum), MMA_M, MMA_N)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dKV); print(sdK); printf("\n"); print(sdKt); printf("\n"); }

    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading K and V
    BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

    int offset_k;
    if constexpr ((IndexSparse && !BwdInnerLoopK)) {
      offset_k = bidb * SparseKBlockSize;
    } else {
      offset_k = get_batch_range(params.k_ranges, bidb).x;
    }

    // IndexSparse LoopQ with SparseKBlockSize < kBlockN: only partial rows valid in the tile,
    // TMA full-tile store would corrupt neighbors. Use per-element path with residual guard.
    // When SparseKBlockSize >= kBlockN the full tile is valid and can use TMA store.
    if constexpr (kOuterStoreMode == OuterStoreMode::Tma && !(IndexSparse && !BwdInnerLoopK)) {
      cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
      cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);

      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      BarrierManager::arrive<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);

      Tensor mdK = params.tma_store_dK.get_tma_tensor(params.shape_dK)(_, _, bidh_kv); // (seqlen_kv, head_dim)
      Tensor mdV = params.tma_store_dV.get_tma_tensor(params.shape_dK)(_, _, bidh_kv); // (seqlen_kv, head_dim)
      Tensor gdK = local_tile(domain_offset(make_coord(offset_k, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)
      Tensor gdV = local_tile(domain_offset(make_coord(offset_k, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)

      auto block_tma_dK = params.tma_store_dK.get_slice(_0{});
      auto block_tma_dV = params.tma_store_dV.get_slice(_0{});
      Tensor tdKgdK = block_tma_dK.partition_D(gdK); // (TMA, TMA_N, TMA_K)
      Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_N, TMA_K)
      Tensor tdVgdV = block_tma_dV.partition_D(gdV); // (TMA, TMA_N, TMA_K)
      Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_N, TMA_K)

      int warp_idx_sync = warp_uniform(thread_idx / cutlass::NumThreadsPerWarp);
      if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
        if constexpr (Deterministic) {
          if (cute::elect_one_sync()) {
            int left_range_conflict_msg = get<0>(det_msg);
            int right_range_conflict_msg = get<1>(det_msg);
            int arrive_num = get<2>(det_msg);
            // FlattenGQA (PackGQA or CatGQA): each work tile handles a single KV head,
            // so the per-Q-head interleaving dimension is not used; treat qheads_per_kheads as 1.
            int qheads_per_kheads = !FlattenGQA ? static_cast<int>(params.qhead_per_khead_divmod) : 1;
            int sync_num1 = bidh_idx_in_group ? arrive_num * qheads_per_kheads + bidh_idx_in_group : (left_range_conflict_msg >> 1) * qheads_per_kheads;
            int sync_num2 = bidh_idx_in_group ? arrive_num * qheads_per_kheads + bidh_idx_in_group : (right_range_conflict_msg >> 1) * qheads_per_kheads;
            deterministic_sync(params.determin_range_locks, bidh_kv, offset_k + n_block * kBlockN, kBlockN, params.nheads, sync_num1, sync_num2);
          }
        }
        BarrierManager::sync<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);
        if (cute::elect_one_sync()) {
          cute::copy(params.tma_store_dV, tdVsdV, tdVgdV);
          cute::copy(params.tma_store_dK, tdKsdK, tdKgdK);
          tma_store_arrive();
        }
      }

      tma_store_wait<0>();

      if constexpr (Deterministic) {
        if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1 && cute::elect_one_sync()) {
          int left_range_conflict_msg = get<0>(det_msg);
          int right_range_conflict_msg = get<1>(det_msg);
          // FlattenGQA (PackGQA or CatGQA): same reasoning as sync — treat qheads_per_kheads as 1
          int qheads_per_kheads = !FlattenGQA ? static_cast<int>(params.qhead_per_khead_divmod) : 1;
          int arrive_num = get<2>(det_msg);
          arrive_num = arrive_num * qheads_per_kheads + bidh_idx_in_group + 1;
          deterministic_arrive(
              params.determin_range_locks,
              bidh_kv,
              offset_k + n_block * kBlockN,
              kBlockN,
              params.nheads,
              arrive_num,
              left_range_conflict_msg & 1,
              right_range_conflict_msg & 1);
        }
      }
    } else {
      GmemTiledCopydKV gmem_tiled_copy_dKV;
      auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(thread_idx);

      Tensor mdK = make_tensor(make_gmem_ptr(params.ptr_dK), params.shape_dK, params.stride_dK)(_, _, bidh_kv);
      Tensor mdV = make_tensor(make_gmem_ptr(params.ptr_dV), params.shape_dV, params.stride_dV)(_, _, bidh_kv);

      Tensor gdK = local_tile(domain_offset(make_coord(offset_k, _0{}), mdK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)
      Tensor gdV = local_tile(domain_offset(make_coord(offset_k, _0{}), mdV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)

      // write back to smem to ensure layout compatibility with flash::copy
      Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));
      Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));
      cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);
      cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
      // make sure all WGs have finished writing to smem
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

      Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
      Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);
      Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
      Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);
      int residual_n;
      if constexpr ((IndexSparse && !BwdInnerLoopK)) {
        residual_n = SparseKBlockSize - n_block * kBlockN;
      } else {
        residual_n = get_batch_range(params.k_ranges, bidb).y - offset_k - n_block * kBlockN;
      }

      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdKsdK,
          tdKgdK,
          gmem_thr_copy_dKV.partition_D(make_identity_tensor(select<1, 2>(TileShape_MNK{}))),
          gmem_thr_copy_dKV.partition_D(make_tensor<bool>(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}))),
          residual_n);
      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dKV,
          tdVsdV,
          tdVgdV,
          gmem_thr_copy_dKV.partition_D(make_identity_tensor(select<1, 2>(TileShape_MNK{}))),
          gmem_thr_copy_dKV.partition_D(make_tensor<bool>(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}))),
          residual_n);
    }
  }

  // Perform a Consumer Epilogue -- TMA store for dQ
  // q for outer-loop and k for inner-loop
  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void store_dq(
      Params const& params,
      FrgTensorO const& tdQrdQ,
      SharedStorage& shared_storage,
      TiledMma tiled_mma,
      int thread_idx,
      BlockCoordType const& block_coord) {
    static_assert(BwdInnerLoopK, "store_dq() must be called when BwdInnerLoopK is true");
    static_assert(!Deterministic, "Deterministic mode is not supported yet");

    // Get block coordinates for current job (tile)
    int m_block = get<0>(block_coord), bidh = get<1>(block_coord), bidb = get<2>(block_coord);
    int offset_q = get_batch_range(params.q_ranges, bidb).x;

    // For PackGQA, bidh is already KV head index (scheduler uses num_heads_kv)
    // For non-PackGQA, bidh is Q head index
    int bidh_idx_in_group;
    int bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh) : bidh;
    Tensor sdQ = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dq.data()), SmemLayoutdQ{}));
    Tensor sdQt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.epilogue.smem_dq.data()), SmemLayoutdQt{}));

    auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomdQ{}, tiled_mma);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(thread_idx);

    // Convert the type of tdQrdQ to ElementDq if they are not the same as ElementAccum
    Tensor tdQrdQ_out = [&] {
      if constexpr (IsSameTypeDq) {
        return tdQrdQ;
      } else {
        auto out = make_tensor_like<ElementDq>(tdQrdQ);
        flash::convert_type_out(tdQrdQ, out);
        return out;
      }
    }();

    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(tdQrdQ_out); // ((Atom,AtomNum), MMA_M, MMA_N)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_dQ); print(sdQ); printf("\n"); print(sdQt); printf("\n"); }

    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(cute::conditional_return<!dQ_swapAB>(sdQ, sdQt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Make sure all WGs have finished reading Q
    BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);

    if constexpr (kOuterStoreMode != OuterStoreMode::Tma) {
      // Per-element direct store: outer Q range is unique per CTA (rangemerge / sparse),
      // so no atomic reduction needed.
      BarrierManager::sync<NumEpilogueThreads>(resv_barrier::EpilogueBarrier);

      GmemTiledCopydQ gmem_tiled_copy_dQ;
      auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(thread_idx);

      int const offset_q_scaled = offset_q * PackGQAFactor;
      Tensor mdQ = make_tensor(make_gmem_ptr(params.ptr_dQ), params.shape_dQ, params.stride_dQ)(_, _, bidh);
      Tensor gdQ = local_tile(domain_offset(make_coord(offset_q_scaled, _0{}), mdQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)

      Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
      Tensor tdQsdQ_e = gmem_thr_copy_dQ.partition_S(sdQ);
      int residual_m = get_batch_range(params.q_ranges, bidb).y * PackGQAFactor - offset_q_scaled - m_block * kBlockM;

      flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
          gmem_tiled_copy_dQ,
          tdQsdQ_e,
          tdQgdQ,
          gmem_thr_copy_dQ.partition_D(make_identity_tensor(select<0, 2>(TileShape_MNK{}))),
          gmem_thr_copy_dQ.partition_D(make_tensor<bool>(make_shape(Int<kBlockM>{}, Int<kHeadDim>{}))),
          residual_m);
    } else {
      // TMA atomic reduce-add: multiple CTAs may contribute to the same outer Q position.
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      BarrierManager::arrive<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);

      int warp_idx_sync = warp_uniform(thread_idx / cutlass::NumThreadsPerWarp);
      int const offset_q_scaled = offset_q * PackGQAFactor;
      Tensor mdQ = params.tma_store_dQ.get_tma_tensor(params.shape_dQ)(_, _, bidh);
      Tensor gdQ = local_tile(domain_offset(make_coord(offset_q_scaled, _0{}), mdQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)

      auto block_tma_dQ = params.tma_store_dQ.get_slice(_0{});
      Tensor tdQgdQ = block_tma_dQ.partition_D(gdQ); // (TMA, TMA_M, TMA_K)
      Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

      if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1) {
        BarrierManager::sync<NumEpilogueThreads + cutlass::NumThreadsPerWarp>(resv_barrier::EpilogueBarrier);
        if (cute::elect_one_sync()) {
          cute::copy(params.tma_store_dQ, tdQsdQ, tdQgdQ);
          tma_store_arrive();
        }
      }

      tma_store_wait<0>();
    }
  }

  CUTLASS_DEVICE void store_tail() {
    // if constexpr (Use_TMA) { tma_store_wait<0>(); }
  }

  // Write 0 to dK and dV
  // k for outer-loop and q for inner-loop
  template <typename DetMsgT = cute::tuple<>>
  CUTLASS_DEVICE void store_zero_dkv(Params const& params, int thread_idx, BlockCoordType const& block_coord, DetMsgT const& det_msg = {}) {
    if constexpr (Deterministic) {
      int warp_idx_sync = warp_uniform(thread_idx / cutlass::NumThreadsPerWarp);
      if (warp_idx_sync == NumEpilogueThreads / cutlass::NumThreadsPerWarp - 1 && cute::elect_one_sync()) {
        int n_block = get<0>(block_coord);
        int bidh = get<1>(block_coord);
        int bidb = get<2>(block_coord);
        int left_range_conflict_msg = get<0>(det_msg);
        int right_range_conflict_msg = get<1>(det_msg);
        int arrive_num = get<2>(det_msg);
        int bidh_idx_in_group;
        int bidh_kv = params.qhead_per_khead_divmod.divmod(bidh_idx_in_group, bidh);
        bidh_kv = cute::conditional_return<!FlattenGQA>(params.qhead_per_khead_divmod.div(bidh), bidh);
        bidh_idx_in_group = cute::conditional_return<!FlattenGQA>(params.qhead_per_khead_divmod.rem(bidh), 0);
        int offset_k = get_batch_range(params.k_ranges, bidb).x;
        // FlattenGQA (PackGQA or CatGQA): treat qheads_per_kheads as 1
        int qheads_per_kheads = !FlattenGQA ? static_cast<int>(params.qhead_per_khead_divmod) : 1;
        int sync_num1 = bidh_idx_in_group ? arrive_num * qheads_per_kheads + bidh_idx_in_group : (left_range_conflict_msg >> 1) * qheads_per_kheads;
        int sync_num2 = bidh_idx_in_group ? arrive_num * qheads_per_kheads + bidh_idx_in_group : (right_range_conflict_msg >> 1) * qheads_per_kheads;
        deterministic_sync(params.determin_range_locks, bidh_kv, offset_k + n_block * kBlockN, kBlockN, params.nheads, sync_num1, sync_num2);
        arrive_num = arrive_num * qheads_per_kheads + bidh_idx_in_group + 1;
        deterministic_arrive(
            params.determin_range_locks,
            bidh_kv,
            offset_k + n_block * kBlockN,
            kBlockN,
            params.nheads,
            arrive_num,
            left_range_conflict_msg & 1,
            right_range_conflict_msg & 1);
      }
    }
  }

  // Write 0 to dQ
  // q for outer-loop and k for inner-loop
  CUTLASS_DEVICE void store_zero_dq(Params const& params, int thread_idx, BlockCoordType const& block_coord) {
    if constexpr (Deterministic) {
      static_assert(!Deterministic, "Deterministic mode is not supported yet");
    }
  }
};

} // namespace flash
