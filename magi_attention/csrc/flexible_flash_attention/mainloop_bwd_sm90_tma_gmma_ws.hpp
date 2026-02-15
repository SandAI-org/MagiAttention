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

#include <cutlass/array.h>
#include <cutlass/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cute/tensor.hpp>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "block.h"
#include "copy_sm90_bulk_reduce.hpp"
#include "mask.h"
#include "named_barrier.hpp"
#include "seqlen.h"
#include "softmax.h"
#include "utils.h"

namespace flash {

using namespace cute;
namespace gcd = cutlass::gemm::collective::detail;

template <
    int Stages,
    int Stages_dO,
    int Stages_dS,
    class ClusterShape_,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Has_softcap_,
    bool Deterministic,
    bool RangeMerge_,
    bool SwapBwdQKLoop_,
    bool SparseLoad_,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 2,
    int AtomLayoutMdQ = 1,
    bool Mma_dP_is_RS = false>
struct CollectiveMainloopBwdSm90 {
  using ClusterShape = ClusterShape_;
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr int kStages = Stages;
  static constexpr int kStages_dO = Stages_dO;
  static constexpr int kStages_dS = Stages_dS;
  static_assert(kStages >= kStages_dO);
  static_assert(Stages_dS == 1 || Stages_dS == kStages);
  static_assert(!Mma_dP_is_RS || SdP_swapAB_); // If Mma_dP_is_RS, we need SdP_SwapAB

  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr bool SwapBwdQKLoop = SwapBwdQKLoop_;
  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
  static constexpr bool SparseLoad = SparseLoad_;
  static_assert(!SparseLoad || RangeMerge); // If SparseLoad, we need RangeMerge
  static_assert(!SparseLoad || SwapBwdQKLoop); // If SparseLoad, we need SwapBwdQKLoop

  using MainloopPipeline = std::conditional_t<!SparseLoad, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
  using TMAClusterBarrier_t = cutlass::arch::ClusterTransactionBarrier::ValueType;
  using BwdNamedBarriers = std::conditional_t<SwapBwdQKLoop, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;
  static_assert(BarrierManager::check<BwdNamedBarriers, NumMmaWarpGroups>());

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  using SeqlenInfo_t = flash::DistributedSeqlenInfo;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN>;

  static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
  static constexpr int AtomLayoutNSdP = NumMmaWarpGroups / AtomLayoutMSdP;
  static constexpr int AtomLayoutMdKV = NumMmaWarpGroups / AtomLayoutNdKV;
  static constexpr int AtomLayoutNdQ = NumMmaWarpGroups / AtomLayoutMdQ;

  static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
  // NOTE: with 1 producer loader, we also need 1 producer storer for dQ atomic reduce-add when disabling SwapBwdQKLoop,
  // however, when enabling SwapBwdQKLoop, we need 2 producer storers, each one for dK,dV atomic reduce-add respectively.
  // When enabling sparse load, 2 warps for sparse load and 2 warps for dK/dV store
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp * (SwapBwdQKLoop ? (!SparseLoad ? 3 : 4) : 2);
  // use two warps in producer WG for sparse load, because other two is for dK/dV store
  static constexpr int NumSparseLoadThreads = cutlass::NumThreadsPerWarp * 2;
  static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutMdKV == 1 && SdP_swapAB && !dKV_swapAB; // if dKV_swapAB, we can't use RS
  static constexpr bool Mma_dQ_is_RS = AtomLayoutNSdP == 1 && AtomLayoutNdQ == 1 && !SdP_swapAB && !dQ_swapAB; // If dQ_swapAB, we can't use RS

  // Const parameters for sparse load
  // A group of 8 threads load global memory together to form one memory transaction (8 * 16B = 128B)
  static constexpr int GroupSize = 8, NumGroups = NumSparseLoadThreads / GroupSize;
  // Number of rows (tokens) to load per group
  static constexpr int NumRowsPerGroup = kBlockN / NumGroups;
  static_assert(!SparseLoad || (NumRowsPerGroup == 4 || NumRowsPerGroup == 8), "Sparse load only supports 4 or 8 rows per group");

  static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
  static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

  // Define TiledMmaSdP and TiledMmadP for S=QK^T and dP=dOV^T
  using TileShapeAtomSdP = std::
      conditional_t<!SdP_swapAB, Shape<Int<kBlockM>, Int<kBlockN / AtomLayoutNSdP>, Int<kHeadDim>>, Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>>;
  using AtomLayoutSdP =
      std::conditional_t<!SdP_swapAB, Layout<Shape<Int<AtomLayoutMSdP>, Int<AtomLayoutNSdP>, _1>>, Layout<Shape<Int<AtomLayoutNSdP>, Int<AtomLayoutMSdP>, _1>>>;
  using TiledMmaSdP = decltype(cute::make_tiled_mma(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(), AtomLayoutSdP{}));
  using TiledMmadPRS = decltype(cute::make_tiled_mma(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(), AtomLayoutSdP{}));
  using TiledMmadP = std::conditional_t<!Mma_dP_is_RS, TiledMmaSdP, TiledMmadPRS>;
  static_assert(
      stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and stride<0>(typename TiledMmaSdP::BLayout{}) == 0,
      "Stride of the first mode of TiledMmaSdP must be 0");
  static_assert(
      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
      "Size of the first mode of TiledMmaSdP must be NumThreadsPerWarpGroup");

  // Define TiledMmadKV for dK=dS^TQ and dV = P^TdO
  using TileShapeAtomdKV = std::
      conditional_t<!dKV_swapAB, Shape<Int<kBlockN>, Int<kHeadDim / AtomLayoutMdKV>, Int<kBlockM>>, Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>>;
  using AtomLayoutdKV =
      std::conditional_t<!dKV_swapAB, Layout<Shape<Int<AtomLayoutNdKV>, Int<AtomLayoutMdKV>, _1>>, Layout<Shape<Int<AtomLayoutMdKV>, Int<AtomLayoutNdKV>, _1>>>;
  using TiledMmadKV = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dKV_is_RS,
          decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>()),
          decltype(GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdKV,
                   !dKV_swapAB ? PdSt_Major : GMMA::Major::MN,
                   !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())>{},
      AtomLayoutdKV{}));

  // Define TiledMmadQ for dQ=dSK
  using TileShapeAtomdQ = std::
      conditional_t<!dQ_swapAB, Shape<Int<kBlockM>, Int<kHeadDim / AtomLayoutNdQ>, Int<kBlockN>>, Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>>;
  using AtomLayoutdQ =
      std::conditional_t<!dQ_swapAB, Layout<Shape<Int<AtomLayoutMdQ>, Int<AtomLayoutNdQ>, _1>>, Layout<Shape<Int<AtomLayoutNdQ>, Int<AtomLayoutMdQ>, _1>>>;
  using TiledMmadQ = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Mma_dQ_is_RS,
          decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
          decltype(GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShapeAtomdQ,
                   !dQ_swapAB ? PdS_Major : GMMA::Major::MN,
                   !dQ_swapAB ? GMMA::Major::MN : PdS_Major>())>{},
      AtomLayoutdQ{}));

  // NOTE: we need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
  // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
  // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for the layout,
  // only the K dimension changes the layout.
  using SmemLayoutAtomQdO = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim / AtomLayoutMdKV>>()); // for dKV_Mma
  using SmemLayoutQ = std::conditional_t<
      SwapBwdQKLoop,
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, select<0, 2>(TileShape_MNK{}))), // (kBlockM, kHeadDim)
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})))>; // (kBlockM, kHeadDim, kStages)
  using SmemLayoutdO = std::conditional_t<
      SwapBwdQKLoop,
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, select<0, 2>(TileShape_MNK{}))), // (kBlockM, kHeadDim)
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages_dO>{})))>; // (kBlockM, kHeadDim, kStages_dO)

  using SmemLayoutAtomK = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim / AtomLayoutNdQ>>());
  using SmemLayoutK = std::conditional_t<
      SwapBwdQKLoop,
      decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{}))), // (kBlockN, kHeadDim, kStages)
      decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})))>; // (kBlockN, kHeadDim)

  using SmemLayoutAtomV = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
  using SmemLayoutV = std::conditional_t<
      SwapBwdQKLoop,
      decltype(tile_to_shape(SmemLayoutAtomV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{}))), // (kBlockN, kHeadDim, kStages)
      decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})))>; // (kBlockN, kHeadDim)

  using SmemLayoutAtomPdS = decltype(gcd::ss_smem_selector<PdS_Major, Element, Int<kBlockM / AtomLayoutMSdP>, Int<kBlockN / AtomLayoutNSdP>>());
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}), // (kBlockM, kBlockN, kStages_dS)
      std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // Need stride to be multiple of 32, otherwise we get error (misaligned address) when doing TMA if e.g. kBlockM=80
  // We set stride to be multiple of 64 so that if ShuffleLSE, even if threads read from sLSE but out of bounds,
  // it's still a valid smem address.
  static constexpr int LSEStageStride = 4 * cute::round_up(kBlockM, 64);
  using SmemLayoutLSE = std::conditional_t<
      SwapBwdQKLoop,
      cute::Layout<cute::Shape<_4, Int<kBlockM>>, cute::Stride<_1, _4>>, // (4, kBlockM)
      cute::Layout<cute::Shape<_4, Int<kBlockM>, Int<kStages>>, cute::Stride<_1, _4, Int<LSEStageStride>>>>; // (4, kBlockM, kStages)
  using SmemLayoutLSEMmaLoopQ = std::conditional_t<
      SdP_swapAB,
      cute::Layout<cute::Shape<_4, Int<kBlockN>, Int<kBlockM>, Int<kStages>>, cute::Stride<_1, _0, _4, Int<LSEStageStride>>>, // (4, kBlockN, kBlockM, kStages)
      cute::Layout<cute::Shape<_4, Int<kBlockM>, Int<kBlockN>, Int<kStages>>, cute::Stride<_1, _4, _0, Int<LSEStageStride>>>>; // (4, kBlockM, kBlockN, kStages)
  using SmemLayoutLSEMmaLoopK = std::conditional_t<
      SdP_swapAB,
      cute::Layout<cute::Shape<_4, Int<kBlockN>, Int<kBlockM>>, cute::Stride<_1, _0, _4>>, // (4, kBlockN, kBlockM)
      cute::Layout<cute::Shape<_4, Int<kBlockM>, Int<kBlockN>>, cute::Stride<_1, _4, _0>>>; // (4, kBlockM, kBlockN)
  using SmemLayoutLSEMma = std::conditional_t<SwapBwdQKLoop, SmemLayoutLSEMmaLoopK, SmemLayoutLSEMmaLoopQ>;

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt_ = std::conditional_t<
      SwapBwdQKLoop,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))), // (kHeadDim, kBlockM)
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages>{}), make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{})))>; // (kHeadDim,
                                                                                                                                                         // kBlockM,
                                                                                                                                                         // kStages)
  using SmemLayoutQt = decltype(cute::composition(SmemLayoutQ{}, SmemLayoutQt_{}));

  using SmemLayoutdOt_ = std::conditional_t<
      SwapBwdQKLoop,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))), // (kHeadDim, kBlockM)
      decltype(make_layout(
          make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dO>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{})))>; // (kHeadDim, kBlockM, kStages_dO)
  using SmemLayoutdOt = decltype(cute::composition(SmemLayoutdO{}, SmemLayoutdOt_{}));

  using SmemLayoutKt_ = std::conditional_t<
      SwapBwdQKLoop,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<kStages>{}), make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim>{}))), // (kHeadDim,
                                                                                                                                                        // kBlockN,
                                                                                                                                                        // kStages)
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{})))>; // (kHeadDim, kBlockN)
  using SmemLayoutKt = decltype(cute::composition(SmemLayoutK{}, SmemLayoutKt_{}));

  using SmemLayoutPdSt_ =
      decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}), make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{})));
  using SmemLayoutPdSt = decltype(cute::composition(SmemLayoutPdS{}, SmemLayoutPdSt_{}));

  // k for outer-loop and q for inner-loop
  // Thread layout, 256 or 384 threads per row
  // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each WG separately.
  using TileShape_dQaccum = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
  using R2SLayoutAtomdQaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdQaccum{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;
  using SmemLayoutAtomdQaccumTMA = decltype(gcd::ss_smem_selector<GMMA::Major::K, ElementAccum, Int<kBlockM>, Int<kHeadDim / AtomLayoutMdQ>>());
  using SmemLayoutdQaccumTMA = decltype(tile_to_shape(SmemLayoutAtomdQaccumTMA{}, TileShape_dQaccum{}));
  using SmemLayoutdQaccumtTMA =
      decltype(cute::composition(SmemLayoutdQaccumTMA{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));

  // q for outer-loop and k for inner-loop
  // Thread layout, 256 or 384 threads per row
  // We split into NumMmaWarpGroups so that we can do Bulk reduce add for each WG separately.
  using TileShape_dKVaccum = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
  using R2SLayoutAtomdKVaccum = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumMmaWarpGroups>>>;
  using R2STiledCopydKVaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdKVaccum{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdKVaccum = Layout<Shape<Int<kBlockN * kHeadDim / NumMmaWarpGroups>, Int<NumMmaWarpGroups>>>;
  using SmemLayoutAtomdKVaccumTMA = decltype(gcd::ss_smem_selector<GMMA::Major::K, ElementAccum, Int<kBlockN>, Int<kHeadDim / AtomLayoutNdKV>>());
  using SmemLayoutdKVaccumTMA = decltype(tile_to_shape(SmemLayoutAtomdKVaccumTMA{}, TileShape_dKVaccum{}));
  using SmemLayoutdKVaccumtTMA =
      decltype(cute::composition(SmemLayoutdKVaccumTMA{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));

  // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt / dSt.
  // If PdS_major is MN, then we need to "transpose" the write.
  static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
  using SmemCopyAtomPdS = Copy_Atom<
      std::conditional_t<
          (!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
          std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
          std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>>,
      Element>;

  using GmemTiledCopyQdO = std::conditional_t<SwapBwdQKLoop, cute::SM90_TMA_LOAD, decltype(gcd::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})))>;
  using GmemTiledCopyKV = std::conditional_t<SwapBwdQKLoop, decltype(gcd::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{}))), cute::SM90_TMA_LOAD>;
  using GmemTiledCopydQaccum = cute::SM90_TMA_REDUCE_ADD;
  using GmemTiledCopydKVaccum = cute::SM90_TMA_REDUCE_ADD;

  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen, head_dim, num_heads)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t>;
  using ShapeLSE = cute::Shape<_4, int32_t, int32_t>; // (4, seqlen_q, num_heads_q)
  using StrideLSE = cute::Stride<_1, _4, int64_t>;

  using TMA_QdO = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQdO{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      take<0, 2>(SmemLayoutQ{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along N mode for this M load, if any

  using TMA_K = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along M mode for this N load, if any

  using TMA_V = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      take<0, 2>(SmemLayoutV{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along M mode for this N load, if any

  // k for outer-loop and q for inner-loop
  using TMA_add_dQ = decltype(make_tma_copy(
      GmemTiledCopydQaccum{},
      make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      SmemLayoutdQaccumTMA{},
      TileShape_dQaccum{},
      _1{})); // no mcast for partial dQ

  // q for outer-loop and k for inner-loop
  using TMA_add_dKV = decltype(make_tma_copy(
      GmemTiledCopydKVaccum{},
      make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      SmemLayoutdKVaccumTMA{},
      TileShape_dKVaccum{},
      _1{})); // no mcast for partial dK,dV

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(kBlockM * kHeadDim * sizeof_bytes_v<Element>());
  static constexpr uint32_t TmaTransactionBytesdO = TmaTransactionBytesQ;
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(kBlockN * kHeadDim * sizeof_bytes_v<Element>());
  static constexpr uint32_t TmaTransactionBytesV = TmaTransactionBytesK;
  static constexpr uint32_t TmaTransactionBytesLSE = static_cast<uint32_t>(4 * kBlockM * sizeof_bytes_v<ElementAccum>());
  static constexpr uint32_t TmaTransactionBytesdPsum = TmaTransactionBytesLSE;
  static_assert(TmaTransactionBytesQ == TmaTransactionBytesdO, "TmaTransactionBytesQ must equal TmaTransactionBytesdO");
  static_assert(TmaTransactionBytesK == TmaTransactionBytesV, "TmaTransactionBytesK must equal TmaTransactionBytesV");
  static_assert(TmaTransactionBytesLSE == TmaTransactionBytesdPsum, "TmaTransactionBytesLSE must equal TmaTransactionBytesdPsum");

  // These are tuned for speed. They don't affect correctness.
  // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
  // this helps quite a bit to not have to do causal masking for most of the iterations.
  // For hdim 192, separating masking iterations results in register spills.
  static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
  // Do we keep the LSE and dPsum in each thread, or split them across 8 threads that share them
  // and then shuffle to get the value whenever we need? This can reduce register pressure when SdP_swapAB,
  // where each thread needs to keep statistics for (kBlockM / 4) rows.
  // If !SdP_swapAB, each thread only needs to keep statistic for 2 rows.
  static constexpr bool ShuffleLSE = SdP_swapAB && kHeadDim <= 128;
  static constexpr bool ShuffledPsum = SdP_swapAB && kHeadDim <= 128;
  static constexpr bool dQacc_use_TMA = kHeadDim < 256;
  static constexpr bool dKVacc_use_TMA = kHeadDim < 256;
  // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
  // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
  static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && !dQacc_use_TMA && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
  static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");
  static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS), "When enabling Slice_dQKV_Mma, we can't use Mma_dKV_is_RS");

  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
  static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
  static constexpr size_t SmemAlignmentV = !Mma_dP_is_RS ? SmemAlignmentQKVdO : cutlass::detail::alignment_for_swizzle(SmemLayoutV{});
  static constexpr size_t SmemAlignmentLSE = 128, SmemAlignmentdPsum = 128;
  static constexpr size_t maxSmemAlignment = cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO, SmemAlignmentV, SmemAlignmentLSE, SmemAlignmentdPsum);
  static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

  // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't line up with smem_k and smem_v due to alignment?
  using SmemdQacc_t = std::conditional_t<!dQacc_use_TMA, cute::array<ElementAccum, 0>, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdQaccumTMA>>>;
  using SmemdKVacc_t = std::conditional_t<!dKVacc_use_TMA, cute::array<ElementAccum, 0>, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutdKVaccumTMA>>>;
  using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;

  struct TensorStorageLoopQ : cute::aligned_struct<maxSmemAlignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentLSE> smem_lse;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentdPsum> smem_dpsum;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    SmemdQacc_t smem_dqacc;
  };

  struct TensorStorageLoopK : cute::aligned_struct<maxSmemAlignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentLSE> smem_lse;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentdPsum> smem_dpsum;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    SmemdKVacc_t smem_dkacc;
    SmemdKVacc_t smem_dvacc;
  };

  using TensorStorage = std::conditional_t<SwapBwdQKLoop, TensorStorageLoopK, TensorStorageLoopQ>;

  // Host side kernel arguments
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    Element* const ptr_V;
    StrideQKV const stride_V;
    Element const* const ptr_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQ; // k for outer-loop and q for inner-loop
    ShapeQKV const shape_dQ;
    StrideQKV const stride_dQ;
    ElementAccum* const ptr_dK; // q for outer-loop and k for inner-loop
    ShapeQKV const shape_dK;
    StrideQKV const stride_dK;
    ElementAccum* const ptr_dV; // q for outer-loop and k for inner-loop
    ShapeQKV const shape_dV;
    StrideQKV const stride_dV;
    float const* const ptr_LSE_log2;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE_log2;
    float const* const ptr_dPsum;
    StrideLSE const stride_dPsum;
    float const softmax_scale;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const cu_batches;
    int* dq_determin_conflict_state;
    int* dq_determin_range_locks;
    int const* const attn_type_map = nullptr;
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
  };

  // Device side kernel params
  struct Params {
    ShapeQKV const shape_Q;
    Element* const ptr_K; // used for sparse load calculating offsets
    ShapeQKV const shape_K;
    StrideQKV const stride_K; // used for sparse load calculating offsets
    Element* const ptr_V; // used for sparse load calculating offsets
    StrideQKV const stride_V; // used for sparse load calculating offsets
    ElementAccum* const ptr_dQ; // k for outer-loop and q for inner-loop
    ShapeQKV const shape_dQ;
    StrideQKV const stride_dQ;
    ElementAccum* const ptr_dK; // q for outer-loop and k for inner-loop
    ShapeQKV const shape_dK;
    StrideQKV const stride_dK;
    ElementAccum* const ptr_dV; // q for outer-loop and k for inner-loop
    ShapeQKV const shape_dV;
    StrideQKV const stride_dV;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_QdO tma_load_Q, tma_load_dO;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    TMA_add_dQ tma_add_dQ; // k for outer-loop and q for inner-loop
    TMA_add_dKV tma_add_dK; // q for outer-loop and k for inner-loop
    TMA_add_dKV tma_add_dV; // q for outer-loop and k for inner-loop
    float const* const ptr_LSE_log2;
    ShapeLSE const shape_LSE;
    StrideLSE const stride_LSE_log2;
    float const* const ptr_dPsum;
    StrideLSE const stride_dPsum;
    float const softmax_scale;
    float const softmax_scale_log2;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const cu_batches;
    int* dq_determin_conflict_state;
    int* dq_determin_range_locks;
    int const n_block_max_num;
    int const* const attn_type_map = nullptr;
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
  };

  template <bool IsProducer>
  struct BlockMeta {
    int const& m_block;
    int const& bidh;
    int const bidh_kv;
    int bidb;
    int end_batches;

    SeqlenInfo_t seqlen_info;
    flash::AttnType attn_type;
    int n_block_min;
    int n_block_max;

    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;

    template <typename SharedStorage>
    CUTLASS_DEVICE BlockMeta(Params const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage)
        : m_block(get<0>(block_coord)),
          bidh(get<1>(block_coord)),
          bidh_kv(params.qhead_per_khead_divmod.divide(bidh)),
          q_ranges(params.q_ranges),
          k_ranges(params.k_ranges),
          attn_type_map(params.attn_type_map) {
      bidb = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord)];
        } else {
          return get<2>(block_coord);
        }
      }();
      end_batches = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord) + 1];
        } else {
          return bidb + 1;
        }
      }();

      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
        auto [n_block_min_, n_block_max_] = BlockMN_t::get_n_block_min_max(seqlen_info, m_block, bidb, attn_type);
        n_block_min = n_block_min_;
        n_block_max = n_block_max_;
      }
    }

    CUTLASS_DEVICE
    void prefetch() {
      ++bidb;
      if constexpr (RangeMerge) {
        if (!is_finish()) {
          seqlen_info.update_k(bidb);
          attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
          auto [n_block_min_, n_block_max_] = BlockMN_t::get_n_block_min_max(seqlen_info, m_block, bidb, attn_type);
          n_block_min = n_block_min_;
          n_block_max = n_block_max_;
        }
      }
    }

    CUTLASS_DEVICE
    bool is_valid() {
      return n_block_min < n_block_max;
    }

    CUTLASS_DEVICE
    bool is_finish() {
      return bidb >= end_batches;
    }
  };

  // only used when SparseLoad=true
  struct SparseLoadBlockMeta {
    int const& m_block;
    int const& bidh;
    int const bidh_kv;
    int bidb;
    int end_batches;
    SeqlenInfo_t seqlen_info;
    flash::AttnType attn_type;

    // number of invalid tokens of the current group, control the token pointer of each group
    int num_invalid_token;
    int cur_k_range_indices[NumRowsPerGroup];
    int cur_k_range_inner_indices[NumRowsPerGroup];
    int token_indices[NumRowsPerGroup];
    int prev_token_indices[NumRowsPerGroup];
    int cur_loop;
    int loop_count;
    int stride_kv_s_kv;
    bool is_equal_k_range_size;
    int k_range_size; // size of all k ranges, only used when is_equal_k_range_size == true

    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;

    template <typename SharedStorage>
    CUTLASS_DEVICE SparseLoadBlockMeta(Params const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage, int thread_idx)
        : m_block(get<0>(block_coord)),
          bidh(get<1>(block_coord)),
          bidh_kv(params.qhead_per_khead_divmod.divide(bidh)), // for packgqa, bidh_kv is the actual head index
          q_ranges(params.q_ranges),
          k_ranges(params.k_ranges),
          attn_type_map(params.attn_type_map),
          is_equal_k_range_size(params.equal_k_range_size ? *params.equal_k_range_size == 1 : false),
          stride_kv_s_kv(get<0>(params.stride_K)) {
      bidb = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord)];
        } else {
          return get<2>(block_coord);
        }
      }();
      end_batches = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord) + 1];
        } else {
          return bidb + 1;
        }
      }();
      cur_loop = 0;
      loop_count = params.sparse_load_loop_count ? params.sparse_load_loop_count[get<2>(block_coord)] : 0;
      num_invalid_token = params.sparse_load_invalid_count ? params.sparse_load_invalid_count[get<2>(block_coord)] : 0;

      int last_idx = NumRowsPerGroup - 1;
      // initialize to the last valid token index
      cur_k_range_indices[last_idx] = end_batches - 1;
      cur_k_range_inner_indices[last_idx] = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x - 1;
      prev_token_indices[last_idx] = -1;

      if (is_equal_k_range_size) {
        k_range_size = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x;
      }

      int idx_in_sparse_load_group = thread_idx % NumSparseLoadThreads;
      int group_idx = idx_in_sparse_load_group / GroupSize;

      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        // metadata init
        // 1. search for the first token in the group
        int cnt = 0;
        // move to the last token index of each group
        int num_steps = (NumGroups - group_idx - 1) * NumRowsPerGroup;
        if (num_invalid_token) {
          if (num_steps >= num_invalid_token) {
            num_steps -= num_invalid_token;
            num_invalid_token = 0;
          } else {
            num_invalid_token -= num_steps;
            num_steps = 0;
          }
        }

        if (is_equal_k_range_size) {
          // equal size, we can compute the next token index directly with random access
          int n_k_ranges = num_steps / k_range_size;
          int n_k_range_inner = num_steps % k_range_size;

          // Check if the current inner index is sufficient to subtract n_k_range_inner
          if (cur_k_range_inner_indices[last_idx] >= n_k_range_inner) {
            // Sufficient; no borrow required
            cur_k_range_indices[last_idx] -= n_k_ranges;
            cur_k_range_inner_indices[last_idx] -= n_k_range_inner;
          } else {
            // Insufficient; need to borrow from the previous range
            // E.g., if current is 5 and we subtract 6 (n_k_range_inner=6), move to the previous range
            cur_k_range_indices[last_idx] -= (n_k_ranges + 1);
            // New inner index = (current value + block size) - value to subtract
            cur_k_range_inner_indices[last_idx] = cur_k_range_inner_indices[last_idx] + k_range_size - n_k_range_inner;
          }
        } else {
          while (cnt < num_steps) {
            int rest = num_steps - cnt;
            // Old: k_range size larger, move inner pointer
            // New: extra inner pointer to move
            if (cur_k_range_inner_indices[last_idx] + 1 > rest) {
              cur_k_range_inner_indices[last_idx] -= rest;
              break;
            } else {
              cur_k_range_indices[last_idx] -= 1;
              cnt += (cur_k_range_inner_indices[last_idx] + 1);
              // load previous K range, since we iterate from rightmost to leftmost
              int2 prev_k_range = k_ranges[cur_k_range_indices[last_idx]];
              cur_k_range_inner_indices[last_idx] = prev_k_range.y - prev_k_range.x - 1;
            }
          }
        }
        // 2. search for next NumRowsPerGroup tokens and compute token indices
        // K/V index: params.k_range[cur_k_range].x + cur_k_range_inner
        token_indices[last_idx] = (k_ranges[cur_k_range_indices[last_idx]].x + cur_k_range_inner_indices[last_idx]) * stride_kv_s_kv;

        CUTE_UNROLL
        for (int i = last_idx - 1; i >= 0; --i) {
          if (cur_k_range_inner_indices[i + 1] > 0) {
            // only move inner pointer
            cur_k_range_indices[i] = cur_k_range_indices[i + 1];
            cur_k_range_inner_indices[i] = cur_k_range_inner_indices[i + 1] - 1;
          } else {
            // move to previous krange
            cur_k_range_indices[i] = cur_k_range_indices[i + 1] - 1;
            int2 prev_k_range = k_ranges[cur_k_range_indices[i]];
            cur_k_range_inner_indices[i] = prev_k_range.y - prev_k_range.x - 1;
          }
          token_indices[i] = (k_ranges[cur_k_range_indices[i]].x + cur_k_range_inner_indices[i]) * stride_kv_s_kv;
        }

        // 3. corner case for boundary mask: move the valid token index ahead
        int offset = num_invalid_token % NumRowsPerGroup;
        switch (offset) {
          case 7:
            if constexpr (NumRowsPerGroup == 8) {
              token_indices[6] = token_indices[last_idx];
              token_indices[5] = token_indices[last_idx - 1];
              token_indices[4] = token_indices[last_idx - 2];
              token_indices[3] = token_indices[last_idx - 3];
              token_indices[2] = token_indices[last_idx - 4];
              token_indices[1] = token_indices[last_idx - 5];
              token_indices[0] = token_indices[last_idx - 6];
            }
            break;
          case 6:
            if constexpr (NumRowsPerGroup == 8) {
              token_indices[5] = token_indices[last_idx];
              token_indices[4] = token_indices[last_idx - 1];
              token_indices[3] = token_indices[last_idx - 2];
              token_indices[2] = token_indices[last_idx - 3];
              token_indices[1] = token_indices[last_idx - 4];
              token_indices[0] = token_indices[last_idx - 5];
            }
            break;
          case 5:
            if constexpr (NumRowsPerGroup == 8) {
              token_indices[4] = token_indices[last_idx];
              token_indices[3] = token_indices[last_idx - 1];
              token_indices[2] = token_indices[last_idx - 2];
              token_indices[1] = token_indices[last_idx - 3];
              token_indices[0] = token_indices[last_idx - 4];
            }
            break;
          case 4:
            if constexpr (NumRowsPerGroup == 8) {
              token_indices[3] = token_indices[last_idx];
              token_indices[2] = token_indices[last_idx - 1];
              token_indices[1] = token_indices[last_idx - 2];
              token_indices[0] = token_indices[last_idx - 3];
            }
            break;
          case 3:
            token_indices[2] = token_indices[last_idx];
            token_indices[1] = token_indices[last_idx - 1];
            token_indices[0] = token_indices[last_idx - 2];
            break;
          case 2:
            token_indices[1] = token_indices[last_idx];
            token_indices[0] = token_indices[last_idx - 1];
            break;
          case 1:
            token_indices[0] = token_indices[last_idx];
            break;
          default:
            break;
        }
      }
    }

    CUTLASS_DEVICE
    void prefetch() {
      ++cur_loop;
      // update previous token indices
      for (int i = 0; i < NumRowsPerGroup; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      // update token index for each thread
      if (!is_finish()) {
        // move pointer to the next token in the next tile
        int num_steps = kBlockN;
        int cnt = 0;
        int last_idx = NumRowsPerGroup - 1;

        if (num_invalid_token) {
          if (num_steps >= num_invalid_token) {
            num_steps -= num_invalid_token;
            num_invalid_token = 0;
          } else {
            num_invalid_token -= num_steps;
            num_steps = 0;
          }
        }

        if (is_equal_k_range_size) {
          // equal size, we can compute the next token index directly with random access
          int n_k_ranges = num_steps / k_range_size;
          int n_k_range_inner = num_steps % k_range_size;

          // Check if the current inner index is sufficient to subtract n_k_range_inner
          if (cur_k_range_inner_indices[last_idx] >= n_k_range_inner) {
            // Sufficient; no borrow required
            cur_k_range_indices[last_idx] -= n_k_ranges;
            cur_k_range_inner_indices[last_idx] -= n_k_range_inner;
          } else {
            // Insufficient; need to borrow from the previous range
            // E.g., if current is 5 and we subtract 6 (n_k_range_inner=6), move to the previous range
            cur_k_range_indices[last_idx] -= (n_k_ranges + 1);
            // New inner index = (current value + block size) - value to subtract
            cur_k_range_inner_indices[last_idx] = cur_k_range_inner_indices[last_idx] + k_range_size - n_k_range_inner;
          }
        } else {
          while (cnt < num_steps) {
            int rest = num_steps - cnt;
            // Old: k_range size larger, move inner pointer
            // New: extra inner pointer to move
            if (cur_k_range_inner_indices[last_idx] + 1 > rest) {
              cur_k_range_inner_indices[last_idx] -= rest;
              break;
            } else {
              cur_k_range_indices[last_idx] -= 1;
              cnt += (cur_k_range_inner_indices[last_idx] + 1);
              // load previous K range, since we iterate from rightmost to leftmost
              int2 prev_k_range = k_ranges[cur_k_range_indices[last_idx]];
              cur_k_range_inner_indices[last_idx] = prev_k_range.y - prev_k_range.x - 1;
            }
          }
        }

        // 2. search for next NumRowsPerGroup tokens and compute token indices
        // K/V index: params.k_range[cur_k_range].x + cur_k_range_inner
        token_indices[last_idx] = (k_ranges[cur_k_range_indices[last_idx]].x + cur_k_range_inner_indices[last_idx]) * stride_kv_s_kv;

        CUTE_UNROLL
        for (int i = last_idx - 1; i >= 0; --i) {
          if (cur_k_range_inner_indices[i + 1] > 0) {
            // only move inner pointer
            cur_k_range_indices[i] = cur_k_range_indices[i + 1];
            cur_k_range_inner_indices[i] = cur_k_range_inner_indices[i + 1] - 1;
          } else {
            // move to previous krange
            cur_k_range_indices[i] = cur_k_range_indices[i + 1] - 1;
            int2 prev_k_range = k_ranges[cur_k_range_indices[i]];
            cur_k_range_inner_indices[i] = prev_k_range.y - prev_k_range.x - 1;
          }
          token_indices[i] = (k_ranges[cur_k_range_indices[i]].x + cur_k_range_inner_indices[i]) * stride_kv_s_kv;
        }
      }
    }

    CUTLASS_DEVICE
    bool is_finish() {
      return cur_loop >= loop_count;
    }

    CUTLASS_DEVICE
    bool is_valid() {
      // blocks while applying sparse load are always valid
      return true;
    }
  };

  // only used when SparseLoad=true
  struct SparseMmaBlockMeta {
    int const& m_block;
    int const& bidh;
    int const bidh_kv;
    int bidb;
    int end_batches;
    SeqlenInfo_t seqlen_info;
    flash::AttnType attn_type;

    int cur_loop;
    int loop_count;
    int num_invalid_token;

    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;

    template <typename SharedStorage>
    CUTLASS_DEVICE SparseMmaBlockMeta(Params const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage)
        : m_block(get<0>(block_coord)),
          bidh(get<1>(block_coord)),
          bidh_kv(params.qhead_per_khead_divmod.divide(bidh)),
          q_ranges(params.q_ranges),
          k_ranges(params.k_ranges),
          attn_type_map(params.attn_type_map) {
      bidb = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord)];
        } else {
          return get<2>(block_coord);
        }
      }();
      end_batches = [&]() {
        if constexpr (RangeMerge) {
          return params.cu_batches[get<2>(block_coord) + 1];
        } else {
          return bidb + 1;
        }
      }();
      cur_loop = 0;
      loop_count = params.sparse_load_loop_count ? params.sparse_load_loop_count[get<2>(block_coord)] : 0;
      num_invalid_token = params.sparse_load_invalid_count ? params.sparse_load_invalid_count[get<2>(block_coord)] : 0;
      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
      }
    }

    CUTLASS_DEVICE
    void prefetch() {
      ++cur_loop;
    }

    CUTLASS_DEVICE
    bool is_finish() {
      return cur_loop >= loop_count;
    }

    CUTLASS_DEVICE
    bool is_valid() {
      // blocks while applying sparse load are always valid
      return true;
    }
  };

  static Params to_underlying_arguments(Arguments const& args) {
    if constexpr (Deterministic) {
      assert(args.dq_determin_conflict_state != nullptr);
      assert(args.dq_determin_range_locks != nullptr);
    }

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_QdO tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mQ, take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{});
    Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_Q, args.stride_dO);
    TMA_QdO tma_load_dO = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mdO, take<0, 2>(SmemLayoutdO{}), TileShape_MNK{}, ClusterShape{});
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{});
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
    TMA_V tma_load_V = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mV, take<0, 2>(SmemLayoutV{}), TileShape_MNK{}, ClusterShape{});

    Tensor mdQ = make_tensor(make_gmem_ptr(args.ptr_dQ), args.shape_dQ, args.stride_dQ);
    TMA_add_dQ tma_add_dQ = make_tma_copy(GmemTiledCopydQaccum{}, mdQ, SmemLayoutdQaccumTMA{}, TileShape_dQaccum{}, _1{});
    Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), args.shape_dK, args.stride_dK);
    TMA_add_dKV tma_add_dK = make_tma_copy(GmemTiledCopydKVaccum{}, mdK, SmemLayoutdKVaccumTMA{}, TileShape_dKVaccum{}, _1{});
    Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), args.shape_dV, args.stride_dV);
    TMA_add_dKV tma_add_dV = make_tma_copy(GmemTiledCopydKVaccum{}, mdV, SmemLayoutdKVaccumTMA{}, TileShape_dKVaccum{}, _1{});

    // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
    // Right after this, we multiply by log2(e) before applying exp2.
    // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
    // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
    // (assigning it to params.softmax_scale_log2).
    // In the backward, we need to multiply by
    // (1 - tanh^2) * softmax_scale / softcap_val * softcap_val = (1 - tanh^2) * softmax_scale.
    // Instead we multiply by (1 - tanh^2) and multiply dK and dV by params.softmax_scale
    // (the original softmax_scale) at the end.
    return {
        args.shape_Q,
        args.ptr_K,
        args.shape_K,
        args.stride_K,
        args.ptr_V,
        args.stride_V,
        args.ptr_dQ,
        args.shape_dQ,
        args.stride_dQ,
        args.ptr_dK,
        args.shape_dK,
        args.stride_dK,
        args.ptr_dV,
        args.shape_dV,
        args.stride_dV,
        /*qhead_per_khead_divmod=*/cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
        tma_load_Q,
        tma_load_dO,
        tma_load_K,
        tma_load_V,
        tma_add_dQ,
        tma_add_dK,
        tma_add_dV,
        args.ptr_LSE_log2,
        args.shape_LSE,
        args.stride_LSE_log2,
        args.ptr_dPsum,
        args.stride_dPsum,
        args.softmax_scale,
        /*softmax_scale_log2=*/!Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        /*softcap_val=*/!Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.q_ranges,
        args.k_ranges,
        args.cu_batches,
        args.dq_determin_conflict_state,
        args.dq_determin_range_locks,
        /*n_block_max_num=*/cute::ceil_div(get<0>(args.shape_K), kBlockN),
        args.attn_type_map,
        args.sparse_load_loop_count,
        args.sparse_load_invalid_count,
        args.equal_k_range_size};
  }

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    if constexpr (!SparseLoad) {
      cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }
  }

  // Perform a Producer Prologue/Mainloop -- TMA Load for K,V, with pipelining multi-stage TMA load for Q,dO,LSE,dPsum
  // k for outer-loop and q for inner-loop
  template <typename SharedStorage>
  CUTLASS_DEVICE bool load_with_loop_q(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write_q,
      PipelineState_dO& smem_pipe_write_do,
      SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      bool const has_valid_tile) {
    static_assert(!SwapBwdQKLoop, "load_with_loop_q() must be called when SwapBwdQKLoop is false");

    int n_block = get<0>(block_coord), bidh = get<1>(block_coord), bidb = get<2>(block_coord);
    int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
    SeqlenInfo_t seqlen_info{bidb, params.q_ranges, params.k_ranges};

    flash::AttnType attn_type = static_cast<flash::AttnType>(params.attn_type_map ? params.attn_type_map[bidb] : 0);
    auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(seqlen_info, n_block, bidb, attn_type);

    // It's possible to have m_block_max <= m_block_min,
    // where loading Q,dO might cause illegal memory access
    if (m_block_max <= m_block_min) {
      return false;
    }

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
    Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

    // prepare for TMA multicast meta
    auto [mcast_mask_qdo, cluster_block_id_qdo] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyQdO, /*RowwiseMask=*/false>();

    // Prepare the TMA loads
    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh); // (seqlen_q, head_dim)
    Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, bidh); // (seqlen_q, head_dim)
    Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, _, bidh); // (4, seqlen_q)
    Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, _, bidh); // (4, seqlen_q)

    Tensor gQ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (M, K, _)
    Tensor gdO = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (M, K, _)
    Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)
    Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)

    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(_0{}, seqlen_info.offset_q), mLSE), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, _)); // (4, M, _)
    Tensor gdPsum = local_tile(cute::domain_offset(make_coord(_0{}, seqlen_info.offset_q), mdPsum), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, _)); // (4, M, _)

    // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
    auto block_tma_Q = params.tma_load_Q.get_slice(cluster_block_id_qdo);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    // auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, block_rank_in_cluster, Layout<ClusterShape>{},
    //                                   group_modes<0, 2>(sQ), group_modes<0, 2>(gQ));  // (TMA, k), (TMA, PIPE)

    // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
    auto block_tma_dO = params.tma_load_dO.get_slice(cluster_block_id_qdo);
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
    // auto [tdOgdO, tdOsdO] = tma_partition(params.tma_load_dO, block_rank_in_cluster, Layout<ClusterShape>{},
    //                                   group_modes<0, 2>(sdO), group_modes<0, 2>(gdO));  // (TMA, k), (TMA, PIPE)

    Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
    Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
    Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
    Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));
    auto partition_K = tma_partition(params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x)); // (TMA), (TMA)
    auto partition_V = tma_partition(params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x)); // (TMA), (TMA)
    auto tKgK = get<0>(partition_K);
    auto tKsK = get<1>(partition_K);
    auto tVgV = get<0>(partition_V);
    auto tVsV = get<1>(partition_V);

    int m_block = m_block_min;
    int lane_predicate = cute::elect_one_sync();

    // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
    // int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
    // if (warp_idx_in_warpgroup == 0)
    //    BarrierManager::sync<NumMmaThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::KVEmpty);

    // Define lambda funcs to load Q,dO,K,V,LSE,dPsum
    auto load_Q_LSE = [&, mcast_mask_qdo = mcast_mask_qdo](int const m_block_idx) {
      pipeline_q.producer_acquire(smem_pipe_write_q);
      copy(
          params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
          tQgQ(_, m_block_idx),
          tQsQ(_, smem_pipe_write_q.index()));
      copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE(_, _, m_block_idx), sLSE(_, _, smem_pipe_write_q.index()));
    };

    auto load_dO_dPsum = [&, mcast_mask_qdo = mcast_mask_qdo](int const m_block_idx) {
      // If Q and dO have the same number of stages,
      // we can use the same pipeline state variable to reduce registers
      PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write_q, smem_pipe_write_do);
      pipeline_do.producer_acquire(smem_pipe_write_do_cur);
      copy(
          params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST),
          tdOgdO(_, m_block_idx),
          tdOsdO(_, smem_pipe_write_do_cur.index()));
      copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, _, m_block_idx), sdPsum(_, _, smem_pipe_write_do_cur.index()));
    };

    auto load_KV = [&]() {
      if (!has_valid_tile) {
        auto& barrier_KV = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_KV);
        shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
        // REVIEW: why not add `TMA::CacheHintSm90::EVICT_FIRST` hint here ?
        copy(params.tma_load_K.with(barrier_KV, /*mcast_mask=*/0), tKgK, tKsK);
        copy(params.tma_load_V.with(barrier_KV, /*mcast_mask=*/0), tVgV, tVsV);
      }
    };

    // Prologue: load first m block of Q,LSE and K,V for this n block
    if (lane_predicate) {
      load_Q_LSE(m_block);
      load_KV();
    }

    // MainLoop: load ith m block of dO,dPsum and (i+1)th m block of Q,LSE
    if (lane_predicate) {
#pragma unroll(kHeadDim < 256 ? 2 : 1)
      for (; m_block < m_block_max - 1; ++m_block) {
        load_dO_dPsum(m_block);

        if constexpr (!Q_dO_same_stages) {
          ++smem_pipe_write_do;
        }
        ++smem_pipe_write_q;

        load_Q_LSE(m_block + 1);
      }
    }

    // Epilogue: load last m block of dO,dPsum
    if (lane_predicate) {
      load_dO_dPsum(m_block);

      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_write_do;
      }
      ++smem_pipe_write_q;
    }

    // Update smem_pipe_write_do to smem_pipe_write_q if they share the same stages
    if constexpr (Q_dO_same_stages) {
      smem_pipe_write_do = smem_pipe_write_q;
    }

    return true;
  }

  // Perform a Producer Prologue/Mainloop -- TMA Load for Q,dO,LSE,dPsum, with pipelining multi-stage TMA load for K,V
  // q for outer-loop and k for inner-loop
  template <typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE bool load_with_loop_k(
      Params const& params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_v,
      SharedStorage& shared_storage,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      BlockMetaT& block_meta) {
    static_assert(SwapBwdQKLoop, "load_with_loop_k() must be called when SwapBwdQKLoop is true");

    // prepare for TMA multicast meta
    auto [mcast_mask_kv, cluster_block_id_kv] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyKV, /*RowwiseMask=*/true>();

    int const thread_idx = threadIdx.x % NumSparseLoadThreads;

    if constexpr (!SparseLoad) {
      while (!block_meta.is_finish() && !block_meta.is_valid()) {
        // Find the first valid block_meta
        block_meta.prefetch();
      }
    }

    if (block_meta.is_finish()) {
      // No valid block found
      return false;
    }

    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
    int lane_predicate = cute::elect_one_sync();
    auto is_tma_issue_thread = [&]() { return warp_idx_in_warpgroup == 0 && lane_predicate; };

    // Params for Sparse Load
    int64_t cache_policy = createpolicy_evict_last();
    int num_tiles = kHeadDim * sizeof(Element) / 128; // each tile load 128B
    int idx_in_sparse_load_group = thread_idx % NumSparseLoadThreads;
    int idx_in_group = idx_in_sparse_load_group % GroupSize;
    int group_idx = idx_in_sparse_load_group / GroupSize;

    // Wait for the MMA warpgroups to say that smem_q and smem_do are ready
    // int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
    // if (warp_idx_in_warpgroup == 0)
    //    BarrierManager::sync<NumMmaThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::QdOEmpty);

    // Define lambda funcs to load Q,dO,K,V,LSE,dPsum
    auto load_K = [&, mcast_mask_kv = mcast_mask_kv](int const n_block_idx, int offset_k) {
      if constexpr (!SparseLoad) {
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, block_meta.bidh_kv); // (seqlen_kv, head_dim)
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
        auto block_tma_K = params.tma_load_K.get_slice(cluster_block_id_kv);
        Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK)); // (TMA, k)
        Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK)); // (TMA, PIPE)

        if (is_tma_issue_thread()) {
          pipeline_k.producer_acquire(smem_pipe_write_k);
          copy(
              params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
              tKgK(_, n_block_idx),
              tKsK(_, smem_pipe_write_k.index()));
          ++smem_pipe_write_k;
        }
      } else {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        // Producer Ops. calculate src/dst offset based on token index, then cp.async load
        // K shape: (seqlen, head_dim, num_heads)
        // each thread in the same group has a offset 16B (8 elements)
        Element* ptr_gK_base = params.ptr_K + block_meta.bidh_kv * get<2>(params.stride_K) + idx_in_group * 8;
        // shared memory pointer
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});

        // loop over token indices
        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int token_idx = block_meta.token_indices[local_row];
          // loop over number of tiles to load one token
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            Element* dst_ptr = &sK(group_idx * NumRowsPerGroup + local_row, idx_in_group * 8 + tile_idx * 64, smem_pipe_write_k.index());
            cp_async_cacheglobal_l2_prefetch_256B(ptr_gK_base + token_idx + tile_idx * 64, dst_ptr, true, cache_policy);
          }
        }

        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;
      }
    };

    auto load_V = [&, mcast_mask_kv = mcast_mask_kv](int const n_block_idx, int offset_k) {
      if constexpr (!SparseLoad) {
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, block_meta.bidh_kv); // (seqlen_kv, head_dim)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
        auto block_tma_V = params.tma_load_V.get_slice(cluster_block_id_kv);
        Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV)); // (TMA, k)
        Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV)); // (TMA, PIPE)

        if (is_tma_issue_thread()) {
          pipeline_v.producer_acquire(smem_pipe_write_v);
          copy(
              params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
              tVgV(_, n_block_idx),
              tVsV(_, smem_pipe_write_v.index()));
          ++smem_pipe_write_v;
        }
      } else {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        // Producer Ops. calculate src/dst offset based on token index, then cp.async load
        // V shape: (seqlen, head_dim, num_heads)
        // each thread in the same group has a offset 16B (8 elements)
        Element* ptr_gV_base = params.ptr_V + block_meta.bidh_kv * get<2>(params.stride_V) + idx_in_group * 8;
        // shared memory pointer
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});

        // loop over token indices
        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int token_idx = block_meta.prev_token_indices[local_row];
          // loop over number of tiles to load one token
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            Element* dst_ptr = &sV(group_idx * NumRowsPerGroup + local_row, idx_in_group * 8 + tile_idx * 64, smem_pipe_write_v.index());
            cp_async_cacheglobal_l2_prefetch_256B(ptr_gV_base + token_idx + tile_idx * 64, dst_ptr, true, cache_policy);
          }
        }

        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      }
    };

    auto load_QdO_LSE_dPsum = [&]() {
      Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, block_meta.bidh); // (seqlen_q, head_dim)
      Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, block_meta.bidh); // (seqlen_q, head_dim)
      Tensor mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSE, params.stride_LSE_log2)(_, _, block_meta.bidh); // (4, seqlen_q)
      Tensor mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSE, params.stride_dPsum)(_, _, block_meta.bidh); // (4, seqlen_q)

      Tensor gQ = local_tile(
          domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(block_meta.m_block, _0{})); // (M, K)
      Tensor gdO = local_tile(
          domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(block_meta.m_block, _0{})); // (M, K)
      Tensor gLSE = local_tile(
          cute::domain_offset(make_coord(_0{}, block_meta.seqlen_info.offset_q), mLSE),
          make_shape(_4{}, Int<kBlockM>{}),
          make_coord(_0{}, block_meta.m_block)); // (4, M)
      Tensor gdPsum = local_tile(
          cute::domain_offset(make_coord(_0{}, block_meta.seqlen_info.offset_q), mdPsum),
          make_shape(_4{}, Int<kBlockM>{}),
          make_coord(_0{}, block_meta.m_block)); // (4, M)

      Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
      Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});

      Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
      Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

      // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
      auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)
      Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)

      // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
      auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
      Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO)); // (TMA)
      Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO)); // (TMA)

      if (is_tma_issue_thread()) {
        auto& barrier_QdO = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_QdO);
        shared_storage.pipelines.barrier_QdO.arrive_and_expect_tx(TmaTransactionBytesQ + TmaTransactionBytesdO + TmaTransactionBytesLSE + TmaTransactionBytesdPsum);
        // REVIEW: why not add `TMA::CacheHintSm90::EVICT_FIRST` hint here ?
        copy(params.tma_load_Q.with(barrier_QdO, /*mcast_mask=*/0), tQgQ, tQsQ);
        copy(params.tma_load_dO.with(barrier_QdO, /*mcast_mask=*/0), tdOgdO, tdOsdO);
        copy(bulk_copy.with(barrier_QdO), gLSE, sLSE);
        copy(bulk_copy.with(barrier_QdO), gdPsum, sdPsum);
      }
    };

    // Get n_block for kv
    int n_block;
    int prev_n_block;
    // Get offset for kv
    int offset_k = block_meta.seqlen_info.offset_k;
    int prev_offset_k = offset_k;
    // Get the minimum number of blocks to load
    int n_block_min;
    int n_block_max;
    if constexpr (!SparseLoad) {
      n_block = block_meta.n_block_min;
      n_block_max = block_meta.n_block_max;
      n_block_min = block_meta.n_block_min;
      prev_n_block = n_block;
    } else {
      n_block = 0;
      n_block_max = block_meta.loop_count;
    }

    // Prologue: load first n block of K and Q,dO,LSE,dPsum for this m block
    load_K(n_block, offset_k);
    load_QdO_LSE_dPsum();
    ++n_block;

    // MainLoop: load (i+1)th n block of K and ith n block of V
    do {
      // Prefetch the next block_meta
      block_meta.prefetch();
      if constexpr (SparseLoad) {
        n_block = block_meta.cur_loop;
      }
      if constexpr (!SparseLoad) {
#pragma unroll(kHeadDim < 256 ? 2 : 1)
        while (n_block < n_block_max) {
          load_V(prev_n_block, prev_offset_k);
          load_K(n_block, offset_k);
          // Step the previous n_block and offset_k
          prev_n_block = n_block;
          prev_offset_k = offset_k;
          // Increment n_block
          ++n_block;
        }

        // Step into the next block
        n_block = block_meta.n_block_min;
        offset_k = block_meta.seqlen_info.offset_k;
        n_block_max = block_meta.n_block_max;
      } else {
        if (n_block < n_block_max) {
          // Load interleaved K/V
          load_K(n_block, offset_k);
          load_V(prev_n_block, prev_offset_k);
        }
      }
    } while (!block_meta.is_finish() && block_meta.is_valid());

    // Epilogue: load last n block of V
    load_V(prev_n_block, prev_offset_k);

    return true;
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  // when q and do don't share the same stage
  // k for outer-loop and q for inner-loop
  CUTLASS_DEVICE void load_tail_with_loop_q(
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write_q,
      PipelineState_dO& smem_pipe_write_do) {
    static_assert(!SwapBwdQKLoop, "load_tail_with_loop_q() must be called when SwapBwdQKLoop is false");

    // Issue the epilogue waits
    if (cute::elect_one_sync()) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was still inverted from make_producer_start_state
       */
      pipeline_q.producer_tail(smem_pipe_write_q);
      pipeline_do.producer_tail(smem_pipe_write_do);
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  // q for outer-loop and k for inner-loop
  CUTLASS_DEVICE void load_tail_with_loop_k(
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_v) {
    static_assert(SwapBwdQKLoop, "load_tail_with_loop_k() must be called when SwapBwdQKLoop is true");

    // Issue the epilogue waits
    if constexpr (!SparseLoad) {
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
        /* This helps avoid early exit of blocks in Cluster
         * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
         * then would just be acquired since the phase was still inverted from make_producer_start_state
         */
        pipeline_k.producer_tail(smem_pipe_write_k);
        pipeline_v.producer_tail(smem_pipe_write_v);
      }
    } else {
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
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

  // Store partial dQ from SMEM to GMEM with TMA Atomic Reduce Add
  // k for outer-loop and q for inner-loop
  template <typename SharedStorage>
  CUTLASS_DEVICE void store_dq(Params const& params, SharedStorage& shared_storage, cute::tuple<int32_t, int32_t, int32_t> block_coord, int bidb_last = 0) {
    static_assert(!SwapBwdQKLoop, "store_dq() must be called when SwapBwdQKLoop is false");

    if constexpr (!dQacc_use_TMA) {
      return;
    }

    static constexpr int kBlockM = CollectiveMainloopBwdSm90::kBlockM;
    static constexpr int kBlockN = CollectiveMainloopBwdSm90::kBlockN;

    int n_block = get<0>(block_coord), bidh = get<1>(block_coord), bidb = get<2>(block_coord);
    SeqlenInfo_t seqlen_info{bidb, params.q_ranges, params.k_ranges};
    flash::AttnType attn_type = static_cast<flash::AttnType>(params.attn_type_map ? params.attn_type_map[bidb] : 0);
    auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(seqlen_info, n_block, bidb, attn_type);

    if constexpr (Deterministic) {
      // update conflict state of batches
      // bidb_last is the previous bidb, need to update conflict state of bidb_last ~ bidb
      // block_now is the block id of bidb, block_size = kBlock
      // params.q_ranges[bidb].x ~ params.q_ranges[bidb].y is the range of bidb
      int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
      uint32_t smid = blockIdx.x;
      uint32_t sm_stride = gridDim.x;
      int* conflict_state = params.dq_determin_conflict_state;
      // update missed batch's conflict state, loop for bidb_last ~ bidb
      while (bidb_last < bidb) {
        // bidb_last_l ~ bidb_last_r is the range of bidb_last
        int bidb_last_l = params.q_ranges[bidb_last].x, bidb_last_r = params.q_ranges[bidb_last].y;
        int l = bidb_last_l / kBlockM + lane; // bidb_last_l / kBlock is first block id
        int block_num = cute::ceil_div(bidb_last_r - bidb_last_l, kBlockM); // calc total block num of bidb_last
        int r = (bidb_last_l + block_num * kBlockM - 1) / kBlockM; // calc last block id
        // each threads of warp update conflict block id left ~ right
        // each batch's range will conflict with previous batch, which cover the same block id
        while (l <= r) {
          // conflict state[block id * sm_stride + smid] save the conflict info of this sm
          // conflict info is the previous conflict batch id + 1 (make it different to inital value 0)
          // conflict state == 0 means that there is no conflict batch, this batch is the first batch to add
          conflict_state[l * sm_stride + smid] = bidb_last + 1;
          l += cutlass::NumThreadsPerWarp;
        }
        bidb_last++;
      }
      __syncwarp();
    }

    int const last_n_block = cute::ceil_div(seqlen_info.seqlen_k, kBlockN) - 1;
    int const m_block_num = cute::ceil_div(seqlen_info.seqlen_q, kBlockM);
    bool const lane_predicate = cute::elect_one_sync();
    int const num_heads = get<2>(params.shape_Q);

    // batch i use [i * n_block_max_num + 1 , i * n_block_max_num + n_block_size - 1] for add rank of same qhead
    // except for the last n_block_id, the last is always (i + 1) * n_block_max_num
    auto m_block_sync = [&](int m_block_id) {
      uint32_t smid = blockIdx.x;
      uint32_t sm_stride = gridDim.x;
      // calc dq conflict range lock index
      int left_dq_conflict_index = seqlen_info.offset_q / kBlockM + m_block_id;
      int right_dq_conflict_index = (seqlen_info.offset_q + kBlockM - 1) / kBlockM + m_block_id;
      // the first n_block should wait for conflict batches
      // the others n_block should wait for previous n_block
      int sync_num1 = n_block == 0 ? params.dq_determin_conflict_state[left_dq_conflict_index * sm_stride + smid] * params.n_block_max_num
                                   : bidb * params.n_block_max_num + n_block;
      int sync_num2 = n_block == 0 ? params.dq_determin_conflict_state[right_dq_conflict_index * sm_stride + smid] * params.n_block_max_num
                                   : bidb * params.n_block_max_num + n_block;
      deterministic_sync(params.dq_determin_range_locks, bidh, seqlen_info.offset_q + m_block_id * kBlockM, kBlockM, num_heads, sync_num1, sync_num2);
    };

    auto m_block_arrive = [&](int m_block_id) {
      // calc arrive message: l_arrive_twice & r_arrive_twice
      // each range_lock needs to arrive twice to make sure conflict batch has been completed
      // because range_lock block and batch's block may start from a different offset
      bool l_arrive_twice = (m_block_id == 0) && (seqlen_info.offset_q % kBlockM != 0);
      bool r_arrive_twice = (m_block_id == m_block_num - 1) && (seqlen_info.offset_q % kBlockM != 0);
      // the last n_block arrive num is always (batch id + 1) * n_block_max_num
      int arrive_num = n_block == last_n_block ? (bidb + 1) * params.n_block_max_num : bidb * params.n_block_max_num + n_block + 1;
      deterministic_arrive(
          params.dq_determin_range_locks, bidh, seqlen_info.offset_q + m_block_id * kBlockM, kBlockM, num_heads, arrive_num, l_arrive_twice, r_arrive_twice);
    };

    // It's possible to have m_block_max <= m_block_min. Exit early
    if (m_block_max <= m_block_min) {
      if constexpr (Deterministic) {
        if (lane_predicate) {
          for (int m_block = 0; m_block < m_block_num; ++m_block) {
            m_block_sync(m_block);
            m_block_arrive(m_block);
          }
        }
      }
      return;
    }

    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
    Tensor mdQaccum = params.tma_add_dQ.get_tma_tensor(params.shape_dQ)(_, _, bidh); // (seqlen_q, head_dim)
    Tensor gdQaccum = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdQaccum), TileShape_dQaccum{}, make_coord(_, _0{})); // (M, K, _)

    auto block_tma_dQ = params.tma_add_dQ.get_slice(_0{});
    Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum); // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

    int m_block = m_block_min;
    if constexpr (Deterministic) {
      if (lane_predicate) {
        for (int m_block = 0; m_block < m_block_min; ++m_block) {
          m_block_sync(m_block);
          m_block_arrive(m_block);
        }
      }
    }

    auto store_dq_this_m_block = [&]() {
#pragma unroll
      // Sync at sdQ full barrier, to wait for all consumer WGs to finish dQ r2s-copy
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warpgroup_idx); // sdQ full, ready to copy to gmem
      }

      // Issue TMA copy from smem dQ to gmem dQ
      if (lane_predicate) {
        if constexpr (Deterministic) {
          m_block_sync(m_block);
        }
        cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
        tma_store_arrive();
        tma_store_wait<0>();
        if constexpr (Deterministic) {
          m_block_arrive(m_block);
        }
      }

      // Arrive at sdQ empty barrier, to inform all consumer WGs that sdQ is ready to be overwritten
      // NOTE: the for_each() function is required here to ensure `warpgroup_idx` is of type Int<x>.
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warpgroup_idx); // sdQ empty, ready to be overwritten
      }
    };

#pragma unroll 2
    for (; m_block < m_block_max; ++m_block) {
      store_dq_this_m_block();
    }

    if constexpr (Deterministic) {
      if (lane_predicate) {
        for (int m_block = m_block_max; m_block < m_block_num; ++m_block) {
          m_block_sync(m_block);
          m_block_arrive(m_block);
        }
      }
    }
  }

  // Store partial dK,dV from SMEM to GMEM with TMA Atomic Reduce Add
  // q for outer-loop and k for inner-loop
  template <typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE void store_dkv(Params const& params, SharedStorage& shared_storage, cute::tuple<int32_t, int32_t, int32_t> block_coord, BlockMetaT& block_meta) {
    static_assert(SwapBwdQKLoop, "store_dkv() must be called when SwapBwdQKLoop is true");
    static_assert(!Deterministic, "Deterministic mode is not supported yet");

    if constexpr (!dKVacc_use_TMA) {
      return;
    }

    bool const lane_predicate = cute::elect_one_sync();

    while (!block_meta.is_finish() && !block_meta.is_valid()) {
      // Find the first valid block_meta
      block_meta.prefetch();
    }

    if (block_meta.is_finish()) {
      // No valid block found
      return;
    }

    int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

    auto store_dv_this_n_block = [&](int const n_block_idx, int offset_k) {
      Tensor sdV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumTMA{});
      Tensor mdVaccum = params.tma_add_dV.get_tma_tensor(params.shape_dV)(_, _, block_meta.bidh_kv); // (seqlen_kv, head_dim)
      Tensor gdVaccum = local_tile(domain_offset(make_coord(offset_k, _0{}), mdVaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)

      auto block_tma_dV = params.tma_add_dV.get_slice(_0{});
      Tensor tdVgdV = block_tma_dV.partition_D(gdVaccum); // (TMA, TMA_N, TMA_K)
      Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_N, TMA_K)
#pragma unroll
      // Sync at sdV full barrier, to wait for all consumer WGs to finish dV r2s-copy
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warpgroup_idx); // sdV full, ready to copy to gmem
      }

      // Issue TMA copy from smem dV to gmem dV
      if (lane_predicate) {
        cute::copy(params.tma_add_dV, tdVsdV, tdVgdV(_, _, _, n_block_idx));
        tma_store_arrive();
        tma_store_wait<0>();
      }

      // Arrive at sdV empty barrier, to inform all consumer WGs that sdV is ready to be overwritten
      // NOTE: the for_each() function is required here to ensure `warpgroup_idx` is of type Int<x>.
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warpgroup_idx); // sdV empty, ready to be overwritten
      }
    };

    auto store_dk_this_n_block = [&](int const n_block_idx, int offset_k) {
      Tensor sdK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumTMA{});
      Tensor mdKaccum = params.tma_add_dK.get_tma_tensor(params.shape_dK)(_, _, block_meta.bidh_kv); // (seqlen_kv, head_dim)
      Tensor gdKaccum = local_tile(domain_offset(make_coord(offset_k, _0{}), mdKaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)

      auto block_tma_dK = params.tma_add_dK.get_slice(_0{});
      Tensor tdKgdK = block_tma_dK.partition_D(gdKaccum); // (TMA, TMA_N, TMA_K)
      Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_N, TMA_K)
#pragma unroll
      // Sync at sdK full barrier, to wait for all consumer WGs to finish dK r2s-copy
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warpgroup_idx); // sdK full, ready to copy to gmem
      }

      // Issue TMA copy from smem dK to gmem dK
      if (lane_predicate) {
        cute::copy(params.tma_add_dK, tdKsdK, tdKgdK(_, _, _, n_block_idx));
        tma_store_arrive();
        tma_store_wait<0>();
      }

      // Arrive at sdK empty barrier, to inform all consumer WGs that sdK is ready to be overwritten
      // NOTE: the for_each() function is required here to ensure `warpgroup_idx` is of type Int<x>.
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
            BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warpgroup_idx); // sdK empty, ready to be overwritten
      }
    };

    // Get n_block for kv
    int n_block = block_meta.n_block_min;
    // Get offset for kv
    int offset_k = block_meta.seqlen_info.offset_k;
    // Get the maximum number of blocks to load
    int n_block_max = block_meta.n_block_max;

    do {
#pragma unroll 2
      for (; n_block < n_block_max; ++n_block) {
        if (warp_idx_in_warpgroup == 2)
          store_dv_this_n_block(n_block, offset_k);
        else if (warp_idx_in_warpgroup == 3)
          store_dk_this_n_block(n_block, offset_k);
      }
      // Prefetch the next block_meta
      block_meta.prefetch();
      // Step into the next block
      n_block = block_meta.n_block_min;
      offset_k = block_meta.seqlen_info.offset_k;
      n_block_max = block_meta.n_block_max;
    } while (!block_meta.is_finish() && block_meta.is_valid());
  }

  // Initialize MMA consumers
  CUTLASS_DEVICE void mma_init() {
    if constexpr (SwapBwdQKLoop) { // q for outer-loop and k for inner-loop
      // We're not currently using this bc we're not using persistent scheduler
      // Tell producer that smem_q and smem_do are ready
      if constexpr (!SparseLoad) {
        BarrierManager::arrive<NumMmaThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::QdOEmpty);
      } else {
        BarrierManager::arrive<NumMmaThreads + NumSparseLoadThreads>(BwdNamedBarriers::QdOEmpty);
      }

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (dKVacc_use_TMA) {
        if (warp_idx_in_warpgroup == 0) {
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
              BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdV empty, ready to be overwritten
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
              BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdK empty, ready to be overwritten
        }
      }
    } else { // k for outer-loop and q for inner-loop
      // We're not currently using this bc we're not using persistent scheduler
      // Tell producer (warp 0) that smem_k and smem_v are ready
      BarrierManager::arrive<NumMmaThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::KVEmpty);

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (dQacc_use_TMA) {
        if (warp_idx_in_warpgroup == 0) {
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
              BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdQ empty, ready to be overwritten
        }
      }
    }
  }

  // Perform a Consumer Prologue/Mainloop -- WGMMA for S,dP,dQ,dK,dV with softmax for P,dS
  // k for outer-loop and q for inner-loop
  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE bool mma_with_loop_q(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_read_q,
      PipelineState_dO& smem_pipe_read_do,
      FrgTensordKV& tdKrdK,
      FrgTensordKV& tdVrdV,
      int thread_idx,
      int& work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage,
      bool const has_valid_tile) {
    static_assert(!SwapBwdQKLoop, "mma_with_loop_q() must be called when SwapBwdQKLoop is false");
    static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    // Get block coordinates and seqlen info
    int n_block = get<0>(block_coord), bidh = get<1>(block_coord), bidb = get<2>(block_coord);
    SeqlenInfo_t seqlen_info{bidb, params.q_ranges, params.k_ranges};
    int const seqlen_q = seqlen_info.seqlen_q, seqlen_k = seqlen_info.seqlen_k;
    flash::AttnType attn_type = static_cast<flash::AttnType>(params.attn_type_map ? params.attn_type_map[bidb] : 0);
    auto [m_block_min, m_block_max] = BlockMN_t::get_m_block_min_max(seqlen_info, n_block, bidb, attn_type);

    /* DEBUG */
    // if (bidh == 0 && thread_idx == 0) {
    //     printf("[BWD MMA] bidb: %d,  kBlockM: %d, kBlockN: %d, n_block: %d, m_block_min: %d, m_block_max: %d, attn_type: %d\n", bidb, kBlockM, kBlockN, n_block,
    //     m_block_min, m_block_max, attn_type);
    // }

    // It's possible to have m_block_max <= m_block_min. Exit early
    if (m_block_max <= m_block_min) {
      return false;
    }

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);

    Tensor sdQ = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{}));
    Tensor sdQt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumtTMA{}));

    Tensor sdPsumMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma = sLSEMma_full(_0{}, _, _, _); // slice dummy dim 0 with size of 4
    Tensor sdPsumMma = sdPsumMma_full(_0{}, _, _, _); // slice dummy dim 0 with size of 4

    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<NumMmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    TiledMmaSdP tiled_mma_SdP;
    TiledMmadP tiled_mma_dP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;
    auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS); print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP); printf("\n");
    // print(tdSsdS); printf("\n"); }

    auto r2s_tiled_copy_dQaccum = make_tiled_copy_C(Copy_Atom<DefaultCopy, ElementAccum>{}, tiled_mma_dQ);
    auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
    Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(cute::conditional_return<!dQ_swapAB>(sdQ, sdQt));

    /* DEBUG */
    // Tensor cdQsdQ = make_identity_tensor(SmemLayoutdQaccumTMA{}.shape());
    // Tensor tcdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(cdQsdQ);
    // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQaccum); printf("\n"); }

    // Allocate "fragments/descriptors"
    // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
    // because some partition_fragment_A/B don't compile.
    // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
    Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N, PIPE),
    // but we only take the col indices or row indices, depending on whether SdP_swapAB.
    Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)), // (2, MMA_M, PIPE)
        group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _, _))); // (2, V, MMA_N, PIPE)
    Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{}, _)), // (2, MMA_M, PIPE)
        group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _, _))); // (2, V, MMA_N, PIPE)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma); printf("\n"); print(tLSEsLSE); printf("\n"); }

    // If we want to split the stats among the 8 threads that share the same rows.
    static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    auto sync_dS_r2s = [&]() {
      cutlass::arch::fence_view_async_shared(); // proxy fence to make sure dS is written to shared memory before it's read by WGMMA
      BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
    };

    // For the case where we do atomicAdd directly to gdQaccum instead of using TMA
    Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQ)), params.shape_dQ, params.stride_dQ)(_, _, bidh); // (seqlen_q, head_dim)
    Tensor gdQaccum_ = local_tile(domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdQaccum), TileShape_dQaccum{}, make_coord(_, _0{})); // (M, K, _)
    Tensor gdQaccum = cute::flat_divide(gdQaccum_, make_shape(Int<kBlockM / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (M / WG, K, WG, 1, _)

    auto block_tma_dQ = params.tma_add_dQ.get_slice(_0{});
    Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum); // (TMA, TMA_M, TMA_K)
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

    /* DEBUG */
    // if (thread_idx == 0 && bidh == 0 && n_block == 0){
    //     printf("bidb: %d, offset_q: %d\n", bidb, seqlen_info.offset_q);
    //     printf("mdQaccum: "); print(mdQaccum); printf("\n");
    //     printf("gdQaccum_: "); print(gdQaccum_); printf("\n");
    //     printf("gdQaccum: "); print(gdQaccum); printf("\n");
    //     printf("tdQgdQ: "); print(tdQgdQ); printf("\n");
    //     printf("tdQsdQ: "); print(tdQsdQ); printf("\n");
    // }

    // We can reuse r2s_thr_copy_dQaccum for this partitioning
    Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum); printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum); printf("\n"); print(tdQgdQaccum);
    // printf("\n"); }

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask;

    int m_block = m_block_min;
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    // Wait until this n block of K,V loaded
    if (!has_valid_tile) {
      cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
      if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
        shared_storage.pipelines.barrier_KV.wait(work_idx % 2);
      }
    }

    if constexpr (Mma_dP_is_RS) { // guanrateed SdP_SwapAB, then only V needs to copy to registers
      using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
      Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
      Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
      cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
    }

    // Define backward step lambda func
    auto bwd_step = [&](int m_block, auto mask_fn, auto check_mask_lse_type) {
      static constexpr bool check_mask_lse = decltype(check_mask_lse_type)::value;

      // MMA1 (SS): apply S = QK^T or S^T = KQ^T if SdP_swapAB
      // after current m block of Q,LSE loaded
      // note that `tSrQ` stores Q and `tSrK` stores K, so:
      // case1. if SdP_swapAB, we apply S^T = KQ^T (passing Q,K to gemm, it swaps AB to K,Q and then transposes operand B to Q^T)
      // case2. if not SdP_swapAB, we apply S = QK^T (passing Q,K to gemm, it transposes operand B to K^T)
      Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      consumer_wait(pipeline_q, smem_pipe_read_q);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read_q.index()), tSrK, tSrS);

      // Copy LSE from shared memory to registers
      Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
      auto get_lse_scaled = [&](int const mi) {
        if constexpr (!ShuffleLSE) {
          return tLSErLSE(mi);
        } else {
          return broadcast_in_warp(tLSErLSE(mi / 8), /*src_lane=*/(mi % 8) * 4 + (thread_idx % 4));
        }
      };
      if constexpr (!ShuffleLSE) {
        cute::copy(tLSEsLSE(_, smem_pipe_read_q.index()), tLSErLSE);
      } else {
#pragma unroll
        for (int i = 0; i < kStatsPerThread; ++i) {
          // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
          tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8, smem_pipe_read_q.index());
        }
      }

      // MMA2 (SS): apply dP = dOV^T (or dP^T = VdO^T if SdP_swapAB)
      // after current m block of dO,dPsum loaded
      // note that `tdPrdO` stores dO and `tdPrV` stores V, so:
      // case1. if SdP_swapAB, we apply dP^T = VdO^T (passing dO,V to gemm, it swaps AB to V,dO and then transposes operand B to dO^T)
      // case2. if not SdP_swapAB, we apply dP = dOV^T (passing dO,V to gemm, it transposes operand B to V^T)
      Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
      PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read_q, smem_pipe_read_do);
      consumer_wait(pipeline_do, smem_pipe_read_do_cur);
      flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);

      // Apply softcap on `tSrS`, storing capped S (or S^T if SdP_swapAB)
      // after MMA1 finished
      warpgroup_wait<1>();
      if constexpr (Has_softcap) {
        flash::apply_softcap(tSrS, params.softcap_val);
      }

      // Reshape `tSrS` from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
      // and rename the transposed view as `scores`, storing S^T (or S if SdP_swapAB)
      Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));

      // Compute dtanh from `scores`, storing dtanh(S^T) (or dtanh(S) if SdP_swapAB)
      // NOTE: dtanh needs to happen before masking,
      // otherwise we get 1 - (-inf)^2 = NaN in the dtanh
      auto dtanh = [&] {
        if constexpr (Has_softcap)
          return flash::calculate_dtanh(scores);
        else
          return nullptr;
      }();

      // Apply mask on `tSrS`, storing masked S (or S^T if SdP_swapAB)
      mask_fn(tSrS, m_block);

      // Apply scaled softmax on `scores` in-place, storing P^T (or P if SdP_swapAB)
      // NOTE: since we cannot pad for each batch, we need to mask out the OOB LSE values
      // that might be read from other batch at each batch's last m block
      if constexpr (check_mask_lse) {
        // Create identity tensor for block shape
        auto thread_mma = TiledMmaSdP{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMmaSdP{}.get_thread_slice(_0{});

        static constexpr int Row = !SdP_swapAB ? 0 : 1;
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SdP_swapAB ? kBlockM : kBlockN>, Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(t0ScS.layout()));
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const seqlenq_row_limit = seqlen_q - m_block * kBlockM - thread_row_offset;

#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          bool const is_oob = int(get<Row>(t0ScS_rowcol(mi, _0{}))) >= seqlenq_row_limit;
          // NOTE: since the func requries warp sync, all lanes must call it first
          // even though some lanes' LSE values are not used due to OOB mask
          float lse_scaled = get_lse_scaled(mi);
          lse_scaled = is_oob ? cutlass::platform::numeric_limits<float>::infinity() : lse_scaled;
#pragma unroll
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
          }
        }
      } else { // guaranteed no OOB LSE read
#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          float const lse_scaled = get_lse_scaled(mi);
#pragma unroll
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
          }
        }
      }

      /* DEBUG */
      // Tensor scores_16 = make_tensor_like<Element>(tSrS);
      // flash::convert_type_out(tSrS, scores_16);
      // auto scores_16_copy = smem_thr_copy_PdS.retile_S(scores_16);
      // cute::copy(smem_tiled_copy_PdS, scores_16_copy, tdSsdS(_, _, _, cute::conditional_return<kStages_dS == 1>(_0{}, smem_pipe_read_q.index())));
      // BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
      // if (thread_idx == 0) {
      //   print_tensor(
      //     sP(_, _, cute::conditional_return<kStages_dS == 1>(_0{}, smem_pipe_read_q.index()))
      //   );
      // }

      // Copy dPsum from shared memory to registers
      Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum(_, _0{})), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
      auto get_dP_sum_cur = [&](int const mi) {
        if constexpr (!ShuffledPsum) {
          return tLSErdPsum(mi);
        } else {
          return broadcast_in_warp(tLSErdPsum(mi / 8), /*src_lane=*/(mi % 8) * 4 + (thread_idx % 4));
        }
      };
      if constexpr (!ShuffledPsum) {
        cute::copy(tLSEsdPsum(_, smem_pipe_read_do_cur.index()), tLSErdPsum);
      } else {
#pragma unroll
        for (int i = 0; i < kStatsPerThread; ++i) {
          tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8, smem_pipe_read_do_cur.index());
        }
      }

      // Reshape `tdPrdP` from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
      // and rename the view as `dS`, storing dP (or dP^T if SdP_swapAB)
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());

      // Apply softmax backward on `dS`, storing dS (or dS^T if SdP_swapAB)
      // after MMA2 finished
      warpgroup_wait<0>();
#pragma unroll
      for (int mi = 0; mi < size<0>(dS); ++mi) {
        float const dP_sum_cur = get_dP_sum_cur(mi);
#pragma unroll
        for (int ni = 0; ni < size<1>(dS); ++ni) {
          dS(mi, ni) = softmax_backward(/*P=*/scores(mi, ni), /*dP=*/dS(mi, ni), /*dPsum=*/dP_sum_cur);
          if constexpr (Has_softcap) {
            dS(mi, ni) *= dtanh(mi, ni);
          }
        }
      }

      // Downcast `tSrS` from ElementAccum to Element `rP`
      // storing the low-precision of P (or P^T if SdP_swapAB)
      // and copy to shared memory in `tPsP` for dV gemm if not Mma_dKV_is_RS
      // which is the view of `sP_pi` / `sP` (or `sPt_pi` / `sPt` if SdP_swapAB)
      Tensor rP = make_tensor_like<Element>(tSrS);
      flash::convert_type_out(tSrS, rP);
      if constexpr (!Mma_dKV_is_RS) {
        if constexpr (kStages_dS == 1) {
          // NOTE: we need to sync to make sure P has already been used in the previous iteration before writing new values
          BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
        }
        Tensor tPaP = smem_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index())));
      }

      // Downcast `tdPrdP` from ElementAccum to Element `rdS`
      // storing the low-precision of dS (or dS^T if SdP_swapAB)
      // and copy to shared memory in `tdSsdS` for dQ gemm (as well as dK gemm if not Mma_dKV_is_RS)
      // which is the view of `sdS` / `sdS_pi` (or `sdSt` / `sdSt_pi` if SdP_swapAB)
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_out(tdPrdP, rdS);
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        // NOTE: if there's double buffering on dS, we don't need to sync here.
        // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
        // But because both WGs have to sync at the end of the loop and double buffering,
        // this race condition is not possible.
        // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
        // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
        sync_dS_r2s();
      }
      // For hdim 64, It's faster to write to smem_dS first before the dV gemm
      Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS); // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index())));

      // Apply MMA for dQ,dK,dV
      if constexpr (!Slice_dQKV_Mma) { // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
        // MMA3 (RS or SS if not Mma_dKV_is_RS): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
        if constexpr (Mma_dKV_is_RS) {
          // if Mma_dKV_is_RS, it indicates SdP_swapAB and not dKV_swapAB
          // note that `rP` stores P^T and `tdVrdO` stores dO^T,
          // so we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
          Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
        } else {
          // if not Mma_dKV_is_RS, it indicates not SdP_swapAB or dKV_swapAB
          // note that `sPt` stores P^T and `tdVrdO` stores dO^T, so:
          // case1. if dKV_swapAB, we apply dV^T = dO^TP (passing P^T,dO^T to gemm, it swaps AB to dO^T,P^T and then transposes operand B to P)
          // case2. if not dKV_swapAB, we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
          Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
        }

        // MMA4 (SS): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
        // note that `tdQrdS` store dS, `tdQrK` store K^T, so:
        // case1. if dQ_swapAB, we apply dQ^T = K^TdS^T (passing dS,K^T to gemm, it swaps AB to K^T,dS and then transposes operand B to dS^T)
        // case2. if not dQ_swapAB, we apply dQ = dSK (passing dS,K^T to gemm, it transposes operand B to K)
        sync_dS_r2s();
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
        flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);

        // Release dO after MMA3 finished (wg_wait<1> in MMA4)
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        // MMA5 (RS or SS if not Mma_dKV_is_RS): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
        if constexpr (Mma_dKV_is_RS) {
          // if Mma_dKV_is_RS, it indicates SdP_swapAB and not dKV_swapAB
          // note that `rdS` stores dS^T and `tdKrQ` stores Q^T,
          // so we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
          Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
        } else {
          // if not Mma_dKV_is_RS, it indicates not SdP_swapAB or dKV_swapAB
          // note that `sdSt` stores dS^T and `tdKrQ` stores Q^T, so:
          // case1. if dKV_swapAB, we apply dK^T = Q^TdS (passing dS^T,Q^T to gemm, it swaps AB to Q^T,dS^T and then transposes operand B to dS)
          // case2. if not dKV_swapAB, we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
          Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
        }

        // Atomic reduce-add partial dQ
        // after MMA4 finished (wg_wait<1> in MMA5)
        if constexpr (dQacc_use_TMA) { // copy to shared memory first and let producer wap handle the TMA atomic reduce-add to global memory
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          // Sync at sdQ empty barrier, to wait until sdQ is ready to be overwritten
          BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
              BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdQ empty, ready to be overwritten

          // Copy dQ from registers to shared memory with softmax_scale applied
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          for (int dqi = 0; dqi < size(taccdQrdQ); ++dqi) {
            taccdQrdQ(dqi) *= params.softmax_scale;
          }
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);

          /* DEBUG */
          // if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
          //     printf("=================== before retile ===================\n");
          //     cute::print_tensor(tdQrdQ);
          //     printf("=================== after retile ===================\n");
          //     cute::print_tensor(taccdQrdQ);
          //     printf("=================== after copy ===================\n");
          //     cute::print_tensor(tdQsdQaccum);
          // }
          // Tensor cdQoob = make_identity_tensor(SmemLayoutdQaccumOOB{}.shape);
          // Tensor sdQoob = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumOOB{});
          // Tensor sdQoobP = make_tensor<bool>(SmemLayoutdQaccumOOB{}.shape, make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
          // Tensor tsdQoob = sdqoob.tile(TiledFillOOBLayout{});
          // Tensor tcdQoob = cdQoob.tile(TiledFillOOBLayout{});
          // int bound = seqlen_info.seqlen_q - m_block * kBlockM;
          // for (int i = 0; i < size<0>(tsdqoob); ++i){
          //     tsdqoob(i, _0{}, _0{}) = get<0>(tcdQoob(i, _0{}, _0{})) < bound;
          // }

          /* DEBUG */
          // if (m_block == m_block_max - 1){
          //     uint64_t seqlen_q = seqlen_info.seqlen_q;
          //     uint64_t bound = (seqlen_q - m_block * kBlockM) * kHeadDim / NumMmaWarpGroups;
          //     #pragma unroll
          //     for (int i = 0; i < size(tdQsdQaccum); ++i){
          //         if (get<0>(tcdQsdQaccum(i)) >= bound){
          //             tdQsdQaccum(i) = 0;
          //         }
          //     }
          //     if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
          //         printf("=================== tdQsdQaccum ===================\n");
          //         cute::print_tensor(tdQsdQaccum);
          //         printf("=================== bound ===================\n");
          //         printf("seqlen_q: %d, kHeadDim: %d, NumMmaWarpGroups: %d, m_block: %d, kBlockM: %d\n", seqlen_info.seqlen_q, kHeadDim, NumMmaWarpGroups, m_block,
          //         kBlockM); printf("=================== tcdQsdQaccum ===================\n"); cute::print_tensor(tcdQsdQaccum);
          //         printf("============================================\n");
          //     }
          // }

          // Fence and arrive at sdQ full barrier to notify producer warp dQ r2s-copy is finished for this consumer WG
          cutlass::arch::fence_view_async_shared(); // proxy fence to make sure dQ is written to shared memory before it's read by TMA
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
              BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warp_group_idx); // sdQ full, ready to copy to gmem
        } else { // directly atomic reduce-add to global memory
          // We can reuse r2s_thr_copy_dQaccum for this partitioning
          Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
          Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, _, _, m_block));

          // FIXME: size(tdQrdQ_atomic) and size(tdQgdQaccum_atomic) are not matched
          static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
            atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
          }
        }
      } else { // Slice_dQKV_Mma, and guaranteed not Mma_dKV_is_RS
        // MMA3-1 (SS, M_slice=0): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
        // note that `sPt` stores P^T and `tdVrdO` stores dO^T, so:
        // case1. if dKV_swapAB, we apply dV^T = dO^TP (passing P^T,dO^T to gemm, it swaps AB to dO^T,P^T and then transposes operand B to P)
        // case2. if not dKV_swapAB, we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
        Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(
            tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);

        // MMA4-1 (SS, M_slice=0): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
        // note that `tdQrdS` store dS, `tdQrK` store K^T, so:
        // case1. if dQ_swapAB, we apply dQ^T = K^TdS^T (passing dS,K^T to gemm, it swaps AB to K^T,dS and then transposes operand B to dS^T)
        // case2. if not dQ_swapAB, we apply dQ = dSK (passing dS,K^T to gemm, it transposes operand B to K)
        sync_dS_r2s();
        Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
        Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);

        // MMA3-2 (SS, M_slice=1): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(
            tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);

        // Atomic reduce-add partial dQ (M_slice=0) directly to global memory
        // after MMA4-1 finished (wg_wait<1> in MMA3-2)
        Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_dQaccum.retile_S(tdQrdQ));
        Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, _, _, m_block));
#pragma unroll
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        // MMA5-1 (SS, M_slice=0): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
        // note that `sdSt` stores dS^T and `tdKrQ` stores Q^T, so:
        // case1. if dKV_swapAB, we apply dK^T = Q^TdS (passing dS^T,Q^T to gemm, it swaps AB to Q^T,dS^T and then transposes operand B to dS)
        // case2. if not dKV_swapAB, we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
        Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
        Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_q.index()));
        flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(
            tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);

        // Release dO after MMA3-2 finished (wg_wait<1> in MMA5)
        pipeline_do.consumer_release(smem_pipe_read_do_cur);

        // MMA4-2 (SS, M_slice=1): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);

#pragma unroll
        // Atomic reduce-add partial dQ (M_slice=1) directly to global memory
        // after MMA4-1 finished (wg_wait<0> in MMA4-2)
        for (int i = size(tdQrdQ_atomic) / 2; i < size(tdQrdQ_atomic); ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }

        // MMA5-2 (SS, M_slice=1): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(
            tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read_q.index()), tdKrdK);
      }

      // Release Q after MMA5 finished
      warpgroup_wait<0>();
      pipeline_q.consumer_release(smem_pipe_read_q);

      // Update pipeline read state of Q,dO
      ++smem_pipe_read_q;
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_read_do;
      }
    };

    if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      // TODO: Handle causal part, can be optimized
    }

    // Define mask lambda func
    auto mask_fn = [&](auto& tSrS, int m_block) { mask.template apply</*Seqlenk_mask=*/true>(tSrS, m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k); };

    // Apply backward steps
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < m_block_max - 1; ++m_block) {
      bwd_step(m_block, mask_fn, /*check_mask_lse_type=*/cute::false_type{});
    }

    // Apply last epilogue step
    // NOTE: only the last m block needs to mask_lse
    bwd_step(m_block, mask_fn, /*check_mask_lse_type=*/cute::true_type{});

    if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      // TODO: Handle inv causal part, can be optimized
    }

    if constexpr (Q_dO_same_stages) {
      smem_pipe_read_do = smem_pipe_read_q;
    }
    return true;
  }

  // Perform a Consumer Prologue/Mainloop -- WGMMA for S,dP,dQ,dK,dV with softmax for P,dS
  // q for outer-loop and k for inner-loop
  template <typename SharedStorage, typename FrgTensordQ, typename BlockMetaT>
  CUTLASS_DEVICE bool mma_with_loop_k(
      Params const& params,
      MainloopPipeline pipeline_k,
      MainloopPipeline pipeline_v,
      PipelineState& smem_pipe_read_k,
      PipelineState& smem_pipe_read_v,
      FrgTensordQ& tdQrdQ,
      int thread_idx,
      int& work_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      BlockMetaT& block_meta,
      SharedStorage& shared_storage) {
    static_assert(SwapBwdQKLoop, "mma_with_loop_k() must be called when SwapBwdQKLoop is true");
    static_assert(is_rmem<FrgTensordQ>::value, "dQ tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    if constexpr (!SparseLoad) {
      // Get block coordinates and seqlen info
      while (!block_meta.is_finish() && !block_meta.is_valid()) {
        // Find the first valid block_meta
        block_meta.prefetch();
      }
    }

    if (block_meta.is_finish()) {
      // No valid block found
      return false;
    }

    int const seqlen_q = block_meta.seqlen_info.seqlen_q;
    int const m_block = block_meta.m_block;
    int const bidh = block_meta.bidh;
    bool const is_last_m_block_this_batch = seqlen_q - m_block * kBlockM <= kBlockM;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdS{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutPdSt{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);

    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumTMA{}));
    Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumtTMA{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumTMA{}));
    Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumtTMA{}));

    Tensor sdPsumMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma = sLSEMma_full(_0{}, _, _); // slice dummy dim 0 with size of 4
    Tensor sdPsumMma = sdPsumMma_full(_0{}, _, _); // slice dummy dim 0 with size of 4

    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<NumMmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    TiledMmaSdP tiled_mma_SdP;
    TiledMmadP tiled_mma_dP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;
    auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dP = tiled_mma_dP.get_slice(warp_group_thread_layout(warp_group_idx));
    auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi)); // ((Atom,AtomNum),PIPE_M,PIPE_N)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(smem_thr_copy_PdS); print(sP_pi); printf("\n"); print(sPt_pi); printf("\n"); print(tPsP); printf("\n");
    // print(tdSsdS); printf("\n"); }

    auto r2s_tiled_copy_dKVaccum = make_tiled_copy_C(Copy_Atom<DefaultCopy, ElementAccum>{}, tiled_mma_dKV);
    auto r2s_thr_copy_dKVaccum = r2s_tiled_copy_dKVaccum.get_thread_slice(thread_idx);
    Tensor tdKsdKaccum = r2s_thr_copy_dKVaccum.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));
    Tensor tdVsdVaccum = r2s_thr_copy_dKVaccum.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));

    /* DEBUG */
    // Tensor cdKVsdKV = make_identity_tensor(SmemLayoutdKVaccumTMA{}.shape());
    // Tensor tcdKVsdKVaccum = r2s_thr_copy_dKVaccum.partition_D(cdKVsdKV);
    // if (thread_idx == 0) { print(sdK); print(sdV); printf("\n"); print(tdKVsdKVaccum); printf("\n"); }

    // Allocate "fragments/descriptors"
    // We have to use the templated mma_partition_fragment_AB instead of cute::conditional_return or lambda,
    // because some partition_fragment_A/B don't compile.
    // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
    Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
    Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
    Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_dP, sV);
    Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
    Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
    Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
    Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

    // thread_mma_SdP.partition_C(sLSEMma) has shape ((2, 2, V), MMA_M, MMA_N),
    // but we only take the col indices or row indices, depending on whether SdP_swapAB.
    Tensor tLSEsLSE = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_0{}, _, _0{}), _, _0{})), // (2, MMA_M)
        group_modes<0, 3>(thread_mma_SdP.partition_C(sLSEMma)(make_coord(_, _0{}, _), _0{}, _))); // (2, V, MMA_N)
    Tensor tLSEsdPsum = cute::conditional_return<!SdP_swapAB>(
        group_modes<0, 2>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_0{}, _, _0{}), _, _0{})), // (2, MMA_M)
        group_modes<0, 3>(thread_mma_SdP.partition_C(sdPsumMma)(make_coord(_, _0{}, _), _0{}, _))); // (2, V, MMA_N)

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(sLSEMma); printf("\n"); print(tLSEsLSE); printf("\n"); }

    // If we want to split the stats among the 8 threads that share the same rows.
    static constexpr int kStatsPerThread = cute::ceil_div(decltype(size(tLSEsLSE))::value, 8);
    Tensor tLSErLSE = cute::conditional_return<!ShuffleLSE>(make_fragment_like(tLSEsLSE), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
    Tensor tLSErdPsum = cute::conditional_return<!ShuffledPsum>(make_fragment_like(tLSEsdPsum), make_tensor<ElementAccum>(Int<kStatsPerThread>{}));
    auto get_lse_scaled = [&](int const mi) {
      if constexpr (!ShuffleLSE) {
        return tLSErLSE(mi);
      } else {
        return broadcast_in_warp(tLSErLSE(mi / 8), /*src_lane=*/(mi % 8) * 4 + (thread_idx % 4));
      }
    };
    auto get_dP_sum_cur = [&](int const mi) {
      if constexpr (!ShuffledPsum) {
        return tLSErdPsum(mi);
      } else {
        return broadcast_in_warp(tLSErdPsum(mi / 8), /*src_lane=*/(mi % 8) * 4 + (thread_idx % 4));
      }
    };

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    auto sync_dS_r2s = [&]() {
      cutlass::arch::fence_view_async_shared(); // proxy fence to make sure dS is written to shared memory before it's read by WGMMA
      BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
    };

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask;

    int n_block;
    // Get seqlen for kv
    int seqlen_k = block_meta.seqlen_info.seqlen_k;
    // Get offset for kv
    int offset_k = block_meta.seqlen_info.offset_k;
    // Get the maximum number of blocks to calculate
    int n_block_max;

    if constexpr (!SparseLoad) {
      n_block = block_meta.n_block_min;
      n_block_max = block_meta.n_block_max;
    } else {
      n_block = 0;
      n_block_max = block_meta.loop_count;
    }

    // Get attention type for n_block
    flash::AttnType attn_type = block_meta.attn_type;
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    // Wait until this m block of Q,dO,LSE,dPsum loaded
    // and copy LSE,dPsum from shared memory to registers
    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_QdO.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.pipelines.barrier_QdO.wait(work_idx % 2);
    }

    // Copy LSE from shared memory to registers
    if constexpr (!ShuffleLSE) {
      cute::copy(tLSEsLSE, tLSErLSE);
    } else {
#pragma unroll
      for (int i = 0; i < kStatsPerThread; ++i) {
        // It's ok to read OOB, since we made sure sLSE is large enough and we won't use the OOB values
        tLSErLSE(i) = tLSEsLSE((thread_idx % 32) / 4 + i * 8);
      }
    }
    // Copy dPsum from shared memory to registers
    if constexpr (!ShuffledPsum) {
      cute::copy(tLSEsdPsum, tLSErdPsum);
    } else {
#pragma unroll
      for (int i = 0; i < kStatsPerThread; ++i) {
        // It's ok to read OOB, since we made sure sdPsum is large enough and we won't use the OOB values
        tLSErdPsum(i) = tLSEsdPsum((thread_idx % 32) / 4 + i * 8);
      }
    }

    if constexpr (Mma_dP_is_RS) {
      // NOTE: if Mma_dP_is_RS, then SdP_SwapAB must be true,
      // then we have to copy current n block of V to registers every iteration,
      // which seems unacceptable for loop-k settings
      static_assert(!Mma_dP_is_RS, "Mma_dP_is_RS is not supported yet when SwapBwdQKLoop is true.");
    }

    do {
      // define dK, dV, offset_k changed dynamically in the loop
      // For the case where we do atomicAdd directly to gdKaccum,gdVaccum instead of using TMA
      Tensor mdKaccum =
          make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dK)), params.shape_dK, params.stride_dK)(_, _, bidh); // (seqlen_kv, head_dim)
      Tensor gdKaccum_ = local_tile(domain_offset(make_coord(offset_k, _0{}), mdKaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)
      Tensor gdKaccum = cute::flat_divide(gdKaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

      Tensor mdVaccum =
          make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dV)), params.shape_dV, params.stride_dV)(_, _, bidh); // (seqlen_kv, head_dim)
      Tensor gdVaccum_ = local_tile(domain_offset(make_coord(offset_k, _0{}), mdVaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)
      Tensor gdVaccum = cute::flat_divide(gdVaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

      auto block_tma_dK = params.tma_add_dK.get_slice(_0{});
      Tensor tdKgdK = block_tma_dK.partition_D(gdKaccum); // (TMA, TMA_N, TMA_K)
      Tensor tdKsdK = block_tma_dK.partition_S(sdK); // (TMA, TMA_N, TMA_K)

      auto block_tma_dV = params.tma_add_dV.get_slice(_0{});
      Tensor tdVgdV = block_tma_dV.partition_D(gdVaccum); // (TMA, TMA_N, TMA_K)
      Tensor tdVsdV = block_tma_dV.partition_S(sdV); // (TMA, TMA_N, TMA_K)

      /* DEBUG */
      // if (thread_idx == 0 && bidh == 0 && m_block == 0){
      //     printf("bidb: %d, offset_k: %d\n", bidb, seqlen_info.offset_k);
      //     printf("mdKaccum: "); print(mdKaccum); printf("\n");
      //     printf("gdKaccum_: "); print(gdKaccum_); printf("\n");
      //     printf("gdKaccum: "); print(gdKaccum); printf("\n");
      //     printf("tdKgdK: "); print(tdKgdK); printf("\n");
      //     printf("tdKsdK: "); print(tdKsdK); printf("\n");
      //     printf("mdVaccum: "); print(mdVaccum); printf("\n");
      //     printf("gdVaccum_: "); print(gdVaccum_); printf("\n");
      //     printf("gdVaccum: "); print(gdVaccum); printf("\n");
      //     printf("tdVgdV: "); print(tdVgdV); printf("\n");
      //     printf("tdVsdV: "); print(tdVsdV); printf("\n");
      // }

      // We can reuse r2s_thr_copy_dKVaccum for this partitioning
      Tensor tdKgdKaccum = r2s_thr_copy_dKVaccum.partition_D(gdKaccum);
      Tensor tdVgdVaccum = r2s_thr_copy_dKVaccum.partition_D(gdVaccum);

      /* DEBUG */
      // if (blockIdx.x == 0 && threadIdx.x == 128) {
      // print(mdKaccum); printf("\n"); print(gdKaccum_); printf("\n"); print(gdKaccum); printf("\n"); print(tdKgdKaccum); printf("\n"); print(tdKsdK); printf("\n");
      // print(mdVaccum); printf("\n"); print(gdVaccum_); printf("\n"); print(gdVaccum); printf("\n"); print(tdVgdVaccum); printf("\n"); print(tdVsdV); printf("\n");
      // printf("\n"); }
      // Define backward step lambda func
      auto bwd_step = [&](int n_block, auto mask_fn, auto check_mask_lse_type) {
        static constexpr bool check_mask_lse = decltype(check_mask_lse_type)::value;

        // MMA1 (SS): apply S = QK^T (or S^T = KQ^T if SdP_swapAB)
        // after current n block of K loaded
        // note that `tSrQ` stores Q , `tSrK` stores K, so:
        // case1. if SdP_swapAB, we apply S^T = KQ^T (passing Q,K to gemm, it swaps AB to K,Q and then transposes operand B to Q^T)
        // case2. if not SdP_swapAB, we apply S = QK^T (passing Q,K to gemm, it transposes operand B to K^T)
        Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
        consumer_wait(pipeline_k, smem_pipe_read_k);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);

        // MMA2 (SS): apply dP = dOV^T (or dP^T = VdO^T if SdP_swapAB)
        // after current n block of V loaded
        // note that `tdPrdO` stores dO , `tdPrV` stores V, so:
        // case1. if SdP_swapAB, we apply dP^T = VdO^T (passing dO,V to gemm, it swaps AB to V,dO and then transposes operand B to dO^T)
        // case2. if not SdP_swapAB, we apply dP = dOV^T (passing dO,V to gemm, it transposes operand B to V^T)
        Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
        consumer_wait(pipeline_v, smem_pipe_read_v);
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_dP, tdPrdO, tdPrV(_, _, _, smem_pipe_read_v.index()), tdPrdP);

        // Apply softcap on `tSrS`, storing capped S (or S^T if SdP_swapAB)
        // after MMA1 finished
        warpgroup_wait<1>();
        if constexpr (Has_softcap) {
          flash::apply_softcap(tSrS, params.softcap_val);
        }

        // Reshape `tSrS` from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
        // and rename the transposed view as `scores`, storing S^T (or S if SdP_swapAB)
        Tensor scores = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tSrS.layout()));

        // Compute dtanh from `scores`, storing dtanh(S^T) (or dtanh(S) if SdP_swapAB)
        // NOTE: dtanh needs to happen before masking,
        // otherwise we get 1 - (-inf)^2 = NaN in the dtanh
        auto dtanh = [&] {
          if constexpr (Has_softcap)
            return flash::calculate_dtanh(scores);
          else
            return nullptr;
        }();

        // Apply mask on `tSrS`, storing masked S (or S^T if SdP_swapAB)
        mask_fn(tSrS, n_block);

        // Apply scaled softmax on `scores` in-place, storing P^T (or P if SdP_swapAB)
        // NOTE: since we cannot pad for each batch, we need to mask out the OOB LSE values
        // that might be read from other batch at each batch's last m block
        if constexpr (check_mask_lse) {
          // Create identity tensor for block shape
          auto thread_mma = TiledMmaSdP{}.get_thread_slice(thread_idx);
          auto thread0_mma = TiledMmaSdP{}.get_thread_slice(_0{});

          static constexpr int Row = !SdP_swapAB ? 0 : 1;
          Tensor cS = cute::make_identity_tensor(Shape<Int<!SdP_swapAB ? kBlockM : kBlockN>, Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
          Tensor tScS = thread_mma.partition_C(cS);
          Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
          Tensor t0ScS = thread0_mma.partition_C(cS);
          Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(t0ScS.layout()));
          int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
          int const seqlenq_row_limit = seqlen_q - m_block * kBlockM - thread_row_offset;

#pragma unroll
          for (int mi = 0; mi < size<0>(scores); ++mi) {
            bool const is_oob = int(get<Row>(t0ScS_rowcol(mi, _0{}))) >= seqlenq_row_limit;
            // NOTE: since the func requries warp sync, all lanes must call it first
            // even though some lanes' LSE values are not used due to OOB mask
            float lse_scaled = get_lse_scaled(mi);
            lse_scaled = is_oob ? cutlass::platform::numeric_limits<float>::infinity() : lse_scaled;
#pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
              scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
            }
          }
        } else { // guaranteed no OOB LSE read
#pragma unroll
          for (int mi = 0; mi < size<0>(scores); ++mi) {
            float const lse_scaled = get_lse_scaled(mi);
#pragma unroll
            for (int ni = 0; ni < size<1>(scores); ++ni) {
              scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
            }
          }
        }

        /* DEBUG */
        // Tensor scores_16 = make_tensor_like<Element>(tSrS);
        // flash::convert_type_out(tSrS, scores_16);
        // auto scores_16_copy = smem_thr_copy_PdS.retile_S(scores_16);
        // cute::copy(smem_tiled_copy_PdS, scores_16_copy, tdSsdS(_, _, _, cute::conditional_return<kStages_dS == 1>(_0{}, smem_pipe_read_k.index())));
        // BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
        // if (thread_idx == 0) {
        //   print_tensor(
        //     sP(_, _, cute::conditional_return<kStages_dS == 1>(_0{}, smem_pipe_read_k.index()))
        //   );
        // }

        // Reshape `tdPrdP` from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
        // and rename the view as `dS`, storing dP (or dP^T if SdP_swapAB)
        Tensor dS = make_tensor(tdPrdP.data(), scores.layout());

        // Release V after MMA2 finished
        // NOTE: this is different from loop-q settings, whose pipelined Q/dO are required at MMA5/MMA4 resp.
        // while pipelined V is only required at MMA2 in loop-k settings, thus can be released earlier
        warpgroup_wait<0>();
        pipeline_v.consumer_release(smem_pipe_read_v);

#pragma unroll
        // Apply softmax backward on `dS`, storing dS (or dS^T if SdP_swapAB)
        for (int mi = 0; mi < size<0>(dS); ++mi) {
          float const dP_sum_cur = get_dP_sum_cur(mi);
#pragma unroll
          for (int ni = 0; ni < size<1>(dS); ++ni) {
            dS(mi, ni) = softmax_backward(/*P=*/scores(mi, ni), /*dP=*/dS(mi, ni), /*dPsum=*/dP_sum_cur);
            if constexpr (Has_softcap) {
              dS(mi, ni) *= dtanh(mi, ni);
            }
          }
        }

        // Downcast `tSrS` from ElementAccum to Element `rP`
        // storing the low-precision of P (or P^T if SdP_swapAB)
        // and copy to shared memory in `tPsP` for dV gemm if not Mma_dKV_is_RS
        // which is the view of `sP_pi` / `sP` (or `sPt_pi` / `sPt` if SdP_swapAB)
        Tensor rP = make_tensor_like<Element>(tSrS);
        flash::convert_type_out(tSrS, rP);
        if constexpr (!Mma_dKV_is_RS) { // Copy P to shared memory for dK,dV gemm
          if constexpr (kStages_dS == 1) {
            // NOTE: we need to sync to make sure P has already been used in the previous iteration before writing new values
            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          }
          Tensor tPaP = smem_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
          cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index())));
        }

        // Downcast `tdPrdP` from ElementAccum to Element `rdS`
        // storing the low-precision of dS (or dS^T if SdP_swapAB)
        // and copy to shared memory in `tdSsdS` for dQ gemm (as well as dK gemm if not Mma_dKV_is_RS)
        // which is the view of `sdS` / `sdS_pi` (or `sdSt` / `sdSt_pi` if SdP_swapAB)
        Tensor rdS = make_tensor_like<Element>(tdPrdP);
        flash::convert_type_out(tdPrdP, rdS);
        if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
          // NOTE: if there's double buffering on dS, we don't need to sync here.
          // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
          // But because both WGs have to sync at the end of the loop and double buffering,
          // this race condition is not possible.
          // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
          // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
          sync_dS_r2s();
        }
        // For hdim 64, It's faster to write to smem_dS first before the dV gemm
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index())));

        // Apply MMA for dQ,dK,dV
        if constexpr (!Slice_dQKV_Mma) { // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
          // MMA3 (RS or SS if not Mma_dKV_is_RS): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
          Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
          if constexpr (Mma_dKV_is_RS) {
            // if Mma_dKV_is_RS, it indicates SdP_swapAB and not dKV_swapAB
            // note that `rP` stores P^T and `tdVrdO` stores dO^T,
            // so we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
            Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO, tdVrdV);
          } else {
            // if not Mma_dKV_is_RS, it indicates not SdP_swapAB or dKV_swapAB
            // note that `sPt` stores P^T and `tdVrdO` stores dO^T, so:
            // case1. if dKV_swapAB, we apply dV^T = dO^TP (passing P^T,dO^T to gemm, it swaps AB to dO^T,P^T and then transposes operand B to P)
            // case2. if not dKV_swapAB, we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
            Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
            Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrP_cur, tdVrdO, tdVrdV);
          }

          // MMA4 (RS or SS if not Mma_dKV_is_RS): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
          Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
          if constexpr (Mma_dKV_is_RS) {
            // if Mma_dKV_is_RS, it indicates SdP_swapAB and not dKV_swapAB
            // note that `rdS` stores dS^T and `tdKrQ` stores Q^T,
            // so we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
            Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
            flash::gemm</*zero_init=*/true, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ, tdKrdK);
          } else {
            sync_dS_r2s();
            // if not Mma_dKV_is_RS, it indicates not SdP_swapAB or dKV_swapAB
            // note that `sdSt` stores dS^T and `tdKrQ` stores Q^T, so:
            // case1. if dKV_swapAB, we apply dK^T = Q^TdS (passing dS^T,Q^T to gemm, it swaps AB to Q^T,dS^T and then transposes operand B to dS)
            // case2. if not dKV_swapAB, we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
            Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
            Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
            flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ, tdKrdK);
          }

          // Atomic reduce-add partial dV
          // after MMA3 finished (wg_wait<1> in MMA4)
          if constexpr (dKVacc_use_TMA) { // copy to shared memory first and let producer wap handle the TMA atomic reduce-add to global memory
            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

            // Sync at sdV empty barrier, to wait until sdV is ready to be overwritten
            BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
                BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdV empty, ready to be overwritten

            // Copy dV from registers to shared memory
            Tensor taccdVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV);
            cute::copy(r2s_tiled_copy_dKVaccum, taccdVrdV, tdVsdVaccum);

            /* DEBUG */
            // if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
            //     printf("=================== before retile ===================\n");
            //     cute::print_tensor(tdVrdV);
            //     printf("=================== after retile ===================\n");
            //     cute::print_tensor(taccdVrdV);
            //     printf("=================== after copy ===================\n");
            //     cute::print_tensor(tdVsdVaccum);
            // }
            // Tensor cdVoob = make_identity_tensor(SmemLayoutdVaccumOOB{}.shape);
            // Tensor sdVoob = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdVaccumOOB{});
            // Tensor sdVoobP = make_tensor<bool>(SmemLayoutdVaccumOOB{}.shape, make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
            // Tensor tsdVoob = sdVoob.tile(TiledFillOOBLayout{});
            // Tensor tcdVoob = cdVoob.tile(TiledFillOOBLayout{});
            // int bound = seqlen_k - n_block * kBlockN;
            // for (int i = 0; i < size<0>(tsdVoob); ++i){
            //     tsdVoob(i, _0{}, _0{}) = get<0>(tcdVoob(i, _0{}, _0{})) < bound;
            // }

            /* DEBUG */
            // if (n_block == n_block_max - 1){
            //     uint64_t bound = (seqlen_k - n_block * kBlockN) * kHeadDim / NumMmaWarpGroups;
            //     #pragma unroll
            //     for (int i = 0; i < size(tdVsdVaccum); ++i){
            //         if (get<0>(tcdVsdVaccum(i)) >= bound){
            //             tdVsdVaccum(i) = 0;
            //         }
            //     }
            //     if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
            //         printf("=================== tdVsdVaccum ===================\n");
            //         cute::print_tensor(tdVsdVaccum);
            //         printf("=================== bound ===================\n");
            //         printf("seqlen_k: %d, kHeadDim: %d, NumMmaWarpGroups: %d, n_block: %d, kBlockN: %d\n", kHeadDim, NumMmaWarpGroups, n_block,
            //         kBlockN); printf("=================== tcdVsdVaccum ===================\n"); cute::print_tensor(tcdVsdVaccum);
            //         printf("============================================\n");
            //     }
            // }

            // Fence and arrive at sdV full barrier to notify producer warp dV r2s-copy is finished for this consumer WG
            cutlass::arch::fence_view_async_shared(); // proxy fence to make sure dV is written before it's read by TMA
            BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
                BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warp_group_idx); // sdV full, ready to copy to gmem
          } else { // directly atomic reduce-add to global memory
            // We can reuse r2s_thr_copy_dKVaccum for this partitioning
            Tensor tdVrdV_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdVrdV));
            Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, _, _, n_block));

            // FIXME: size(tdVrdV_atomic) and size(tdVgdVaccum_atomic) are not matched
            static_assert(CUTE_STATIC_V(size(tdVrdV_atomic)) == CUTE_STATIC_V(size(tdVgdVaccum_atomic)));
#pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) {
              atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i));
            }
          }

          // MMA5 (SS): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
          // note that `tdQrdS` store dS, `tdQrK` store K^T, so:
          // case1. if dQ_swapAB, we apply dQ^T = K^TdS^T (passing dS,K^T to gemm, it swaps AB to K^T,dS and then transposes operand B to dS^T)
          // case2. if not dQ_swapAB, we apply dQ = dSK (passing dS,K^T to gemm, it transposes operand B to K)
          if constexpr (Mma_dKV_is_RS) {
            sync_dS_r2s();
          }
          Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);

          // Atomic reduce-add partial dK
          // after MMA4 finished (wg_wait<1> in MMA5)
          if constexpr (dKVacc_use_TMA) { // copy to shared memory first and let producer wap handle the TMA atomic reduce-add to global memory
            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

            // Sync at sdK empty barrier, to wait until sdK is ready to be overwritten
            BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
                BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warp_group_idx); // sdK empty, ready to be overwritten

            // Copy dK from registers to shared memory with softmax_scale applied
            Tensor taccdKrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK);
            for (int dki = 0; dki < size(taccdKrdK); ++dki) {
              taccdKrdK(dki) *= params.softmax_scale;
            }
            cute::copy(r2s_tiled_copy_dKVaccum, taccdKrdK, tdKsdKaccum);

            /* DEBUG */
            // if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
            //     printf("=================== before retile ===================\n");
            //     cute::print_tensor(tdKrdK);
            //     printf("=================== after retile ===================\n");
            //     cute::print_tensor(taccdKrdK);
            //     printf("=================== after copy ===================\n");
            //     cute::print_tensor(tdKsdKaccum);
            // }
            // Tensor cdKoob = make_identity_tensor(SmemLayoutdKaccumOOB{}.shape);
            // Tensor sdKoob = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdKaccumOOB{});
            // Tensor sdKoobP = make_tensor<bool>(SmemLayoutdKaccumOOB{}.shape, make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
            // Tensor tsdKoob = sdKoob.tile(TiledFillOOBLayout{});
            // Tensor tcdKoob = cdKoob.tile(TiledFillOOBLayout{});
            // int bound = seqlen_k - n_block * kBlockN;
            // for (int i = 0; i < size<0>(tsdKoob); ++i){
            //     tsdKoob(i, _0{}, _0{}) = get<0>(tcdKoob(i, _0{}, _0{})) < bound;
            // }

            /* DEBUG */
            // if (n_block == n_block_max - 1){
            //     uint64_t bound = (seqlen_k - n_block * kBlockN) * kHeadDim / NumMmaWarpGroups;
            //     #pragma unroll
            //     for (int i = 0; i < size(tdKsdKaccum); ++i){
            //         if (get<0>(tcdKsdKaccum(i)) >= bound){
            //             tdKsdKaccum(i) = 0;
            //         }
            //     }
            //     if (thread_idx == 0 && bidh == 0 && bidb == 0 && n_block == 0) {
            //         printf("=================== tdKsdKaccum ===================\n");
            //         cute::print_tensor(tdKsdKaccum);
            //         printf("=================== bound ===================\n");
            //         printf("seqlen_k: %d, kHeadDim: %d, NumMmaWarpGroups: %d, n_block: %d, kBlockN: %d\n", seqlen_k, kHeadDim, NumMmaWarpGroups, n_block,
            //         kBlockN); printf("=================== tcdKsdKaccum ===================\n"); cute::print_tensor(tcdKsdKaccum);
            //         printf("============================================\n");
            //     }
            // }

            // Fence and arrive at sdK full barrier to notify producer warp dK r2s-copy is finished for this consumer WG
            cutlass::arch::fence_view_async_shared(); // proxy fence to make sure dK is written before it's read by TMA
            BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + cutlass::NumThreadsPerWarp>(
                BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warp_group_idx); // sdK full, ready to copy to gmem
          } else { // directly atomic reduce-add to global memory
            // We can reuse r2s_thr_copy_dKVaccum for this partitioning
            Tensor tdKrdK_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdKrdK));
            Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, _, _, n_block));

            // FIXME: size(tdKrdK_atomic) and size(tdKgdKaccum_atomic) are not matched
            static_assert(CUTE_STATIC_V(size(tdKrdK_atomic)) == CUTE_STATIC_V(size(tdKgdKaccum_atomic)));
#pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) {
              atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i));
            }
          }
        } else { // Slice_dQKV_Mma, and guaranteed not Mma_dKV_is_RS
          // MMA3-1 (SS, M_slice=0): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
          // note that `sPt` stores P^T and `tdVrdO` stores dO^T, so:
          // case1. if dKV_swapAB, we apply dV^T = dO^TP (passing P^T,dO^T to gemm, it swaps AB to dO^T,P^T and then transposes operand B to P)
          // case2. if not dKV_swapAB, we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
          Tensor tdVrdV = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
          Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
          Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
          flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdVrP_cur, tdVrdO, tdVrdV);

          // MMA4-1 (SS, M_slice=0): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
          // note that `sdSt` stores dS^T and `tdKrQ` stores Q^T, so:
          // case1. if dKV_swapAB, we apply dK^T = Q^TdS (passing dS^T,Q^T to gemm, it swaps AB to Q^T,dS^T and then transposes operand B to dS)
          // case2. if not dKV_swapAB, we apply dK = dS^TQ (passing dS^T,Q^T to gemm, it transposes operand B to Q)
          sync_dS_r2s();
          Tensor tdKrdK = partition_fragment_C(tiled_mma_dKV, select<!dKV_swapAB ? 1 : 2, !dKV_swapAB ? 2 : 1>(TileShape_MNK{}));
          Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
          Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
          flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdKrdS_cur, tdKrQ, tdKrdK);

          // Atomic reduce-add partial dV (M_slice=0) directly to global memory
          // after MMA3-1 finished (wg_wait<1> in MMA4-1)
          Tensor tdVrdV_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdVrdV));
          Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, _, _, n_block));
#pragma unroll
          for (int i = 0; i < size(tdVrdV_atomic) / 2; ++i) {
            atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i));
          }

          // MMA3-2 (SS, M_slice=1): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
          flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO, tdVrdV);

          // Atomic reduce-add partial dK (M_slice=0) directly to global memory
          // after MMA4-1 finished (wg_wait<1> in MMA3-2)
          Tensor tdKrdK_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdKrdK));
          Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, _, _, n_block));
#pragma unroll
          for (int i = 0; i < size(tdKrdK_atomic) / 2; ++i) {
            atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i));
          }

          // MMA5-1 (SS, M_slice=0): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
          // note that `tdQrdS` store dS, `tdQrK` store K^T, so:
          // case1. if dQ_swapAB, we apply dQ^T = K^TdS^T (passing dS,K^T to gemm, it swaps AB to K^T,dS and then transposes operand B to dS^T)
          // case2. if not dQ_swapAB, we apply dQ = dSK (passing dS,K^T to gemm, it transposes operand B to K)
          Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return < kStages_dS == 1 > (_0{}, smem_pipe_read_k.index()));
          flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(
              tiled_mma_dQ, tdQrdS_cur, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);

#pragma unroll
          // Atomic reduce-add partial dV (M_slice=1) directly to global memory
          // after MMA3-2 finished (wg_wait<1> in MMA5-1)
          for (int i = size(tdVrdV_atomic) / 2; i < size(tdVrdV_atomic); ++i) {
            atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i));
          }

          // MMA4-2 (SS, M_slice=1): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
          flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ, tdKrdK);

#pragma unroll
          // Atomic reduce-add partial dK (M_slice=1) directly to global memory
          // after MMA4-2 finished (wg_wait<0> in MMA4-2)
          for (int i = size(tdKrdK_atomic) / 2; i < size(tdKrdK_atomic); ++i) {
            atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i));
          }

          // MMA5-2 (SS, M_slice=1): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
          flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(
              tiled_mma_dQ, tdQrdS_cur, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);
        }

        // Release K after MMA5 finished
        warpgroup_wait<0>();
        pipeline_k.consumer_release(smem_pipe_read_k);

        // Update pipeline read state of K,V
        ++smem_pipe_read_k;
        ++smem_pipe_read_v;
      };

      if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
        // TODO: Handle causal part, can be optimized
      }

      // Define mask lambda func
      auto mask_fn = [&](auto& tSrS, int n_block) {
        if constexpr (!SparseLoad) {
          mask.template apply</*Seqlenk_mask=*/true>(tSrS, m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
        } else {
          mask.template apply_sparse_load(tSrS, block_meta.num_invalid_token, thread_idx);
        }
      };

      // Apply backward steps
      // NOTE: only the last m block for the same batch needs to mask_lse
      if constexpr (!SparseLoad) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; n_block < n_block_max - 1; ++n_block) {
          if (is_last_m_block_this_batch)
            bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::true_type{});
          else
            bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::false_type{});
        }

        // Apply last epilogue step
        // NOTE: only the last m block for the same batch needs to mask_lse
        if (is_last_m_block_this_batch)
          bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::true_type{});
        else
          bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::false_type{});

        if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
          // TODO: Handle inv causal part, can be optimized
        }

        // Step into the next block
        block_meta.prefetch();
        n_block = block_meta.n_block_min;
        seqlen_k = block_meta.seqlen_info.seqlen_k;
        offset_k = block_meta.seqlen_info.offset_k;
        n_block_max = block_meta.n_block_max;
        attn_type = block_meta.attn_type;
      } else {
        if (n_block < n_block_max && attn_type == flash::AttnType::Full) {
          if (is_last_m_block_this_batch)
            bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::true_type{});
          else
            bwd_step(n_block, mask_fn, /*check_mask_lse_type=*/cute::false_type{});
        }

        // Step into the next block
        block_meta.prefetch();
        n_block = block_meta.cur_loop;
        attn_type = block_meta.attn_type;
      }
    } while (!block_meta.is_finish() && block_meta.is_valid());

    return true;
  }

  // Debug print some crucial configuration about mma
  // especially for the tiled mma definition
  CUTLASS_DEVICE void debug_print_mma(int block_idx = 0, int thread_idx = 128) {
    if (blockIdx.x == block_idx && threadIdx.x == thread_idx) {
      printf(
          "kBlockM=%d, kBlockN=%d, kHeadDim=%d | dQ_swapAB=%d, dKV_swapAB=%d, SdP_swapAB=%d | Mma_dQ_is_RS=%d, Mma_dKV_is_RS=%d, Mma_dP_is_RS=%d\n",
          kBlockM,
          kBlockN,
          kHeadDim,
          dQ_swapAB,
          dKV_swapAB,
          SdP_swapAB,
          Mma_dQ_is_RS,
          Mma_dKV_is_RS,
          Mma_dP_is_RS);

      TileShapeAtomdQ tile_shape_at_dQ;
      TiledMmadQ tiled_mma_dQ;
      TileShapeAtomdKV tile_shape_at_dKV;
      TiledMmadKV tiled_mma_dKV;
      TileShapeAtomSdP tile_shape_at_SdP;
      TiledMmaSdP tiled_mma_SdP;

      printf("tile_shape_at_dQ:\n");
      print(tile_shape_at_dQ);
      printf("\n");
      printf("tiled_mma_dQ:\n");
      print(tiled_mma_dQ);
      printf("\n");
      printf("\n");

      printf("tile_shape_at_dKV:\n");
      print(tile_shape_at_dKV);
      printf("\n");
      printf("tiled_mma_dKV:\n");
      print(tiled_mma_dKV);
      printf("\n");
      printf("\n");

      printf("tile_shape_at_SdP:\n");
      print(tile_shape_at_SdP);
      printf("\n");
      printf("tiled_mma_SdP:\n");
      print(tiled_mma_SdP);
      printf("\n");
      printf("\n");
    }
  }
};
} // namespace flash
