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
#include "block_meta.h"
#include "copy_sm90_bulk_reduce.hpp"
#include "deterministic.h"
#include "inner_ldst_mode.hpp"
#include "inner_scatter_ldst.hpp"
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
    bool BwdInnerLoopK_,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    bool PackGQA_,
    bool CatGQA_,
    bool RangeMerge_,
    bool BlockSparse_,
    bool IndexSparse_,
    bool InnerDirMaxToMin_,
    int MaskMode_,
    bool InnerStoreInProducer_,
    int InnerStoreMode_,
    int PackGQAFactor_,
    int NumConsumerWarpGroups,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    int Stages_V_,
    int Tma1dSmemRowPad_,
    int SparseKBlockSize_,
    int InnerLoadMode_,
    bool UnionDkvSmem_,
    int InnerStoreStages_,
    bool PerfDebugSkipVLoad_,
    bool PerfDebugSkipDvStore_,
    bool PerfDebugSkipDkStore_,
    bool PerfDebugSkipDvMma_>
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
  static constexpr int kStages_V = Stages_V_;
  static_assert(kStages >= kStages_dO);
  static_assert(Stages_dS == 1 || Stages_dS == kStages);
  static_assert(kStages_V >= 1 && kStages_V <= kStages);

  static constexpr bool PerfDebugSkipVLoad = PerfDebugSkipVLoad_;
  static constexpr bool PerfDebugSkipDvStore = PerfDebugSkipDvStore_;
  static constexpr bool PerfDebugSkipDkStore = PerfDebugSkipDkStore_;
  static constexpr bool PerfDebugSkipDvMma = PerfDebugSkipDvMma_;
  static constexpr bool UnionDkvSmem = UnionDkvSmem_;
  static constexpr int kInnerStoreStages = InnerStoreStages_;
  static_assert(kInnerStoreStages == 1 || kInnerStoreStages == 2, "kInnerStoreStages must be 1 or 2");

  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;
  static constexpr bool BwdInnerLoopK = BwdInnerLoopK_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool CatGQA = CatGQA_;
  static constexpr bool FlattenGQA = PackGQA_ || CatGQA_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr int PackGQAFactor = PackGQAFactor_;
  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
  static constexpr bool BlockSparse = BlockSparse_;
  static constexpr bool IndexSparse = IndexSparse_;
  static constexpr int SparseKBlockSize = SparseKBlockSize_;
  static_assert(!BlockSparse || RangeMerge); // If BlockSparse, we need RangeMerge
  static_assert(!(BlockSparse && IndexSparse));
  static_assert(!IndexSparse || SparseKBlockSize >= 1, "SparseKBlockSize must be >= 1 for IndexSparse");

  static constexpr bool InnerDirMaxToMin = InnerDirMaxToMin_;
  static constexpr int MaskMode = MaskMode_;
  // IsSparse: inner-loop direction data uses scatter load/store (vs TMA):
  // InnerLoopK (BwdInnerLoopK=true):  KV scatter when BlockSparse or IndexSparse
  // InnerLoopQ (BwdInnerLoopK=false): Q/dO scatter when BlockSparse or IndexSparse (inner_indices)
  static constexpr bool IsSparse = BlockSparse || IndexSparse;
  // InnerLoopQ scatter does not support CatGQA (the dense InnerLoopQ load iterates bidh_kv_cat
  // per merged sub-range; the scatter load path has no such loop).
  static_assert(!(IsSparse && !BwdInnerLoopK && CatGQA), "bwd InnerLoopQ scatter (block_sparse) does not support cat_gqa");

  // InnerStoreInProducer: who performs the inner-loop dX (dKV for InnerLoopK / dQ for InnerLoopQ)
  // store. Pure pass-through of the template/env toggle; the python JIT entry only emits
  // a non-default value for scatter configs (dense + consumer-store is valid in principle
  // -- the contiguous-atomicAdd consumer branch exists -- but is untested and currently
  // trips an nvcc ICE, so the entry point does not generate it).
  static constexpr bool InnerStoreInProducer = InnerStoreInProducer_;

  // ─── Inner-Loop Store Strategy (InnerStoreMode enum) ───
  // Tma:         2D TMA reduce-add full-tile from swizzled SMEM (default for dense paths)
  // Tma1d:       cp.reduce.async.bulk per-row from linear SMEM (scatter only)
  // AtomicAdd:   scalar atomicAdd from SMEM (scatter or dense fallback)
  // BypassSmem:  skip SMEM buffer entirely — consumer register atomicAdd to gmem
  static constexpr InnerStoreMode kInnerStoreMode = static_cast<InnerStoreMode>(InnerStoreMode_);
  static_assert(
      kInnerStoreMode == InnerStoreMode::Tma || kInnerStoreMode == InnerStoreMode::Tma1d || kInnerStoreMode == InnerStoreMode::AtomicAdd ||
      kInnerStoreMode == InnerStoreMode::BypassSmem);
  static_assert(IsSparse || kInnerStoreMode != InnerStoreMode::Tma1d, "Tma1d store requires scatter (sparse) path");

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  // ─── Inner-Loop Load Strategy (InnerLoadMode enum) ───
  // Tma:     physically contiguous tiles → TMA 2D descriptor (hardware scatter-free)
  // CpAsync: non-contiguous rows → cp.async per-row (software scatter)
  //
  // TMA requires tiles physically contiguous in gmem. Conditions:
  //   Dense: always contiguous.
  //   InnerLoopQ (inner=Q/dO): PackGQA packs heads into consecutive rows:
  //     BlockSparse: always contiguous (packed rows = token × heads, sequential).
  //     Both: contiguous only when one packed-Q token fills a tile (PackGQAFactor >= kBlockM).
  //     BlockSparse tiles span multiple ranges that may have gaps when PackGQAFactor < kBlockM.
  //   InnerLoopK (inner=K/V): each tile maps to one K block:
  //     BlockSparse: contiguous when kbs >= kBlockN (one block = one tile).
  //     IndexSparse: contiguous when kbs >= kBlockN AND PackGQAFactor >= kBlockM.
  static constexpr bool kInnerTilesContiguous = !IsSparse || (!BwdInnerLoopK && PackGQA && PackGQAFactor >= kBlockM) ||
      (BwdInnerLoopK && ((BlockSparse && SparseKBlockSize >= kBlockN) || (IndexSparse && PackGQA && PackGQAFactor >= kBlockM && SparseKBlockSize >= kBlockN)));
  static constexpr InnerLoadMode kInnerLoadMode = static_cast<InnerLoadMode>(InnerLoadMode_);
  static_assert(kInnerLoadMode != InnerLoadMode::Tma || kInnerTilesContiguous, "TMA inner load requires contiguous tiles");

  using MainloopPipeline =
      std::conditional_t<kInnerLoadMode == InnerLoadMode::Tma, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO =
      std::conditional_t<kInnerLoadMode == InnerLoadMode::Tma, typename cutlass::PipelineTmaAsync<kStages_dO>, typename cutlass::PipelineAsync<kStages_dO>>;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
  using MainloopPipeline_V =
      std::conditional_t<kInnerLoadMode == InnerLoadMode::Tma, typename cutlass::PipelineTmaAsync<kStages_V>, typename cutlass::PipelineAsync<kStages_V>>;
  using PipelineState_V = typename MainloopPipeline_V::PipelineState;
  using TMAClusterBarrier_t = cutlass::arch::ClusterTransactionBarrier::ValueType;
  using BwdNamedBarriers = std::conditional_t<BwdInnerLoopK, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;

  static_assert(BarrierManager::check<BwdNamedBarriers, NumConsumerWarpGroups>());

  using SeqlenInfo_t = flash::SeqlenInfo;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, PackGQA, PackGQAFactor>;

  static_assert(NumConsumerWarpGroups % AtomLayoutMSdP == 0);
  static_assert(NumConsumerWarpGroups % AtomLayoutNdKV == 0);
  static_assert(NumConsumerWarpGroups % AtomLayoutMdQ == 0);
  static constexpr int AtomLayoutNSdP = NumConsumerWarpGroups / AtomLayoutMSdP;
  static constexpr int AtomLayoutMdKV = NumConsumerWarpGroups / AtomLayoutNdKV;
  static constexpr int AtomLayoutNdQ = NumConsumerWarpGroups / AtomLayoutMdQ;

  static constexpr int NumConsumerThreads = NumConsumerWarpGroups * cutlass::NumThreadsPerWarpGroup;

  // ─── ProducerConsts: centralized producer warp role configuration ───
  struct ProducerConsts {
    // Loader warps: scatter (CpAsync) paths need multiple warps for cp.async bandwidth.
    // TMA paths need only 1 warp (hardware-initiated). When !InnerStoreInProducer,
    // no storer warps exist so all remaining producer warps go to loading.
    static constexpr int kInnerLoaderWarps = IsSparse ? (!InnerStoreInProducer ? 4 : 2) : 1;
    // DxStorer warps: 0 if !InnerStoreInProducer (consumer handles store),
    //                 else 2 (InnerLoopK: dK+dV) or 1 (InnerLoopQ: dQ)
    static constexpr int kInnerStorerWarps = !InnerStoreInProducer ? 0 : (BwdInnerLoopK ? 2 : 1);
    static constexpr int kTotalWarps = kInnerLoaderWarps + kInnerStorerWarps;

    // Thread counts (derived)
    static constexpr int kLoaderThreads = kInnerLoaderWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kStorerThreads = kInnerStorerWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kTotalThreads = kTotalWarps * cutlass::NumThreadsPerWarp;

    // Inner store barrier width = consumer WG + storer threads participating per direction.
    // InnerLoopQ: the single dQ storer warp (32 threads).
    // InnerLoopK sparse: both storer warps scatter each direction together (64 threads).
    // InnerLoopK dense: each direction is owned by exactly one warp (dV: warp 1, dK: warp 2)
    // via the early-return guards in store_dV/store_dK, so only 1 warp (32 threads) ever
    // reaches each bar.sync — counting both warps here would deadlock (the non-owning warp
    // never arrives). This also makes the dV and dK barrier chains independent, letting the
    // two TMA store warps run concurrently.
    static constexpr int kInnerStoreBarrierWarps = (BwdInnerLoopK && !IsSparse && InnerStoreInProducer) ? 1 : kInnerStorerWarps;
    static constexpr int kInnerStoreBarrierThreads = cutlass::NumThreadsPerWarpGroup + kInnerStoreBarrierWarps * cutlass::NumThreadsPerWarp;

    // Role predicates (warp_idx is 0-based within the producer warp group)
    static CUTLASS_DEVICE bool is_loader(int warp_idx) {
      return warp_idx < kInnerLoaderWarps;
    }
    static CUTLASS_DEVICE bool is_inner_storer(int warp_idx) {
      return InnerStoreInProducer && warp_idx >= kInnerLoaderWarps && warp_idx < kTotalWarps;
    }
    static CUTLASS_DEVICE bool is_leader_loader(int warp_idx) {
      return warp_idx == 0;
    }
  };

  // Thread counts promoted from ProducerConsts for use in barrier template args.
  static constexpr int NumProducerThreads = ProducerConsts::kTotalThreads;
  static constexpr int NumProducerLoaderThreads = ProducerConsts::kLoaderThreads;
  static constexpr int NumInnerStoreBarrierThreads = ProducerConsts::kInnerStoreBarrierThreads;

  static_assert(!InnerStoreInProducer || ProducerConsts::kInnerStorerWarps > 0);
  static_assert(InnerStoreInProducer || ProducerConsts::kInnerStorerWarps == 0);
  static_assert(ProducerConsts::kTotalWarps * cutlass::NumThreadsPerWarp == NumProducerThreads);

  // Inner tile size in tokens: kBlockN (KV) for InnerLoopK, kBlockM (Q/dO) for InnerLoopQ.
  static constexpr int kInnerTileSize = BwdInnerLoopK ? kBlockN : kBlockM;
  using ScatterLdst = ScatterLdstGroup<NumProducerLoaderThreads, kInnerTileSize>;
  static_assert(!IsSparse || kInnerTileSize % ScatterLdst::kNumGroups == 0, "Scatter requires kInnerTileSize divisible by NumLdstGroups");

  static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutMdKV == 1 && SdP_swapAB && !dKV_swapAB;
  static constexpr bool Mma_dQ_is_RS = AtomLayoutNSdP == 1 && AtomLayoutNdQ == 1 && !SdP_swapAB && !dQ_swapAB; // If dQ_swapAB, we can't use RS

  static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
  static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

  // Define TiledMmaSdP and TiledMmadP for S=QK^T and dP=dOV^T
  using TileShapeAtomSdP = std::
      conditional_t<!SdP_swapAB, Shape<Int<kBlockM>, Int<kBlockN / AtomLayoutNSdP>, Int<kHeadDim>>, Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>>;
  using AtomLayoutSdP =
      std::conditional_t<!SdP_swapAB, Layout<Shape<Int<AtomLayoutMSdP>, Int<AtomLayoutNSdP>, _1>>, Layout<Shape<Int<AtomLayoutNSdP>, Int<AtomLayoutMSdP>, _1>>>;
  using TiledMmaSdP = decltype(cute::make_tiled_mma(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(), AtomLayoutSdP{}));
  using TiledMmadP = TiledMmaSdP;
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
      BwdInnerLoopK,
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, select<0, 2>(TileShape_MNK{}))), // (kBlockM, kHeadDim)
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages>{})))>; // (kBlockM, kHeadDim, kStages)
  using SmemLayoutdO = std::conditional_t<
      BwdInnerLoopK,
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, select<0, 2>(TileShape_MNK{}))), // (kBlockM, kHeadDim)
      decltype(tile_to_shape(SmemLayoutAtomQdO{}, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}, Int<kStages_dO>{})))>; // (kBlockM, kHeadDim, kStages_dO)

  using SmemLayoutAtomK = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim / AtomLayoutNdQ>>());
  using SmemLayoutK = std::conditional_t<
      BwdInnerLoopK,
      decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{}))), // (kBlockN, kHeadDim, kStages)
      decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})))>; // (kBlockN, kHeadDim)

  using SmemLayoutAtomV = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
  using SmemLayoutV = std::conditional_t<
      BwdInnerLoopK,
      decltype(tile_to_shape(SmemLayoutAtomV{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages_V>{}))), // (kBlockN, kHeadDim, kStages_V)
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
      BwdInnerLoopK,
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
  using SmemLayoutLSEMma = std::conditional_t<BwdInnerLoopK, SmemLayoutLSEMmaLoopK, SmemLayoutLSEMmaLoopQ>;

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt_ = std::conditional_t<
      BwdInnerLoopK,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))), // (kHeadDim, kBlockM)
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages>{}), make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{})))>; // (kHeadDim,
                                                                                                                                                         // kBlockM,
                                                                                                                                                         // kStages)
  using SmemLayoutQt = decltype(cute::composition(SmemLayoutQ{}, SmemLayoutQt_{}));

  using SmemLayoutdOt_ = std::conditional_t<
      BwdInnerLoopK,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))), // (kHeadDim, kBlockM)
      decltype(make_layout(
          make_shape(Int<kHeadDim>{}, Int<kBlockM>{}, Int<kStages_dO>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{})))>; // (kHeadDim, kBlockM, kStages_dO)
  using SmemLayoutdOt = decltype(cute::composition(SmemLayoutdO{}, SmemLayoutdOt_{}));

  using SmemLayoutKt_ = std::conditional_t<
      BwdInnerLoopK,
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<kStages>{}), make_stride(Int<kBlockN>{}, _1{}, Int<kBlockN * kHeadDim>{}))), // (kHeadDim,
                                                                                                                                                        // kBlockN,
                                                                                                                                                        // kStages)
      decltype(make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{})))>; // (kHeadDim, kBlockN)
  using SmemLayoutKt = decltype(cute::composition(SmemLayoutK{}, SmemLayoutKt_{}));

  using SmemLayoutPdSt_ =
      decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}), make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{})));
  using SmemLayoutPdSt = decltype(cute::composition(SmemLayoutPdS{}, SmemLayoutPdSt_{}));

  // P only needs 1 stage (produced and consumed within the same inner iteration),
  // unlike dS which needs kStages_dS stages for cross-WG double buffering.
  using SmemLayoutP1 = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{}, _1{}),
      std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));
  using SmemLayoutP1t_ = decltype(make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, _1{}), make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{})));
  using SmemLayoutP1t = decltype(cute::composition(SmemLayoutP1{}, SmemLayoutP1t_{}));

  // k for outer-loop and q for inner-loop
  // Thread layout, 256 or 384 threads per row
  // We split into NumConsumerWarpGroups so that we can do Bulk reduce add for each WG separately.
  using TileShape_InnerDq = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
  using R2SLayoutAtomInnerDq = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumConsumerWarpGroups>>>;
  using R2STiledCopyInnerDq = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomInnerDq{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdQ = Layout<Shape<Int<kBlockM * kHeadDim / NumConsumerWarpGroups>, Int<NumConsumerWarpGroups>>>;
  using SmemLayoutAtomInnerDqSwizzled = decltype(gcd::ss_smem_selector<GMMA::Major::K, ElementAccum, Int<kBlockM>, Int<kHeadDim / AtomLayoutMdQ>>());
  using SmemLayoutdQSwizzled = decltype(tile_to_shape(SmemLayoutAtomInnerDqSwizzled{}, TileShape_InnerDq{}));
  using SmemLayoutdQtSwizzled =
      decltype(cute::composition(SmemLayoutdQSwizzled{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));

  // q for outer-loop and k for inner-loop
  // Thread layout, 256 or 384 threads per row
  // We split into NumConsumerWarpGroups so that we can do Bulk reduce add for each WG separately.
  using TileShape_InnerDkv = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
  using R2SLayoutAtomInnerDkv = Layout<Shape<Int<cutlass::NumThreadsPerWarpGroup>, Int<NumConsumerWarpGroups>>>;
  using R2STiledCopyInnerDkv = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomInnerDkv{},
      Layout<Shape<_4>>{})); // Val layout, 4 vals per store
  using SmemLayoutdKV = Layout<Shape<Int<kBlockN * kHeadDim / NumConsumerWarpGroups>, Int<NumConsumerWarpGroups>>>;
  using SmemLayoutAtomInnerDkvSwizzled = decltype(gcd::ss_smem_selector<GMMA::Major::K, ElementAccum, Int<kBlockN>, Int<kHeadDim / AtomLayoutNdKV>>());
  using SmemLayoutdKVSwizzled = decltype(tile_to_shape(SmemLayoutAtomInnerDkvSwizzled{}, TileShape_InnerDkv{}));
  using SmemLayoutdKVtSwizzled =
      decltype(cute::composition(SmemLayoutdKVSwizzled{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));

  // ─── Scatter dX store smem layouts ───
  // 1D cp.reduce.async.bulk needs each token row to be one LINEAR smem span. The swizzled
  // *inner TMA layouts cannot provide that: their SW128 atom (8 rows x 32 floats, column-major
  // tiled) physically interleaves one logical row into kHeadDim/32 separate 128B chunks that
  // are 8*128B apart. So the 1D bulk-reduce path uses a row-contiguous layout.
  // A 4-float (16B) row pad keeps rows 16B-aligned (bulk reduce requirement) while breaking
  // the worst r2s store bank conflicts (8-way unpadded -> <=2-way padded).
  // kInnerLoadMode == InnerLoadMode::Tma && IsSparse bypasses 1D bulk-reduce entirely (2D TMA reduce instead),
  // keeping the swizzled TMA layout → no bank conflicts, no padding needed.
  static constexpr int kTma1dSmemRowPad = Tma1dSmemRowPad_ >= 0 ? Tma1dSmemRowPad_ : 4; // floats; -1 = auto (default 4)
  // Store-side inner layouts: r2s writes and scatter-store reads go through these.
  // Defaults to SmemLayoutd*Swizzled; only switches to row-contiguous (linear + pad)
  // when Tma1d mode is active AND 2D TMA load is not available (fallback scatter).
  static constexpr bool kUseTma1dLinearDkv = (kInnerStoreMode == InnerStoreMode::Tma1d) && BwdInnerLoopK && kInnerLoadMode != InnerLoadMode::Tma;
  static constexpr bool kUseTma1dLinearDq = (kInnerStoreMode == InnerStoreMode::Tma1d) && !BwdInnerLoopK && kInnerLoadMode != InnerLoadMode::Tma;
  using SmemLayoutdKVStore =
      std::conditional_t<kUseTma1dLinearDkv, Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim + kTma1dSmemRowPad>, _1>>, SmemLayoutdKVSwizzled>;
  using SmemLayoutdQStore =
      std::conditional_t<kUseTma1dLinearDq, Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim + kTma1dSmemRowPad>, _1>>, SmemLayoutdQSwizzled>;
  using SmemLayoutdKVtStore =
      decltype(cute::composition(SmemLayoutdKVStore{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));
  using SmemLayoutdQtStore =
      decltype(cute::composition(SmemLayoutdQStore{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));
  static_assert(kHeadDim * sizeof(ElementAccum) % 16 == 0, "bulk reduce-add requires 16B-multiple row size");

  // If !SdP_swapAB, the MMA registers hold P / dS, otherwise they hold Pt / dSt.
  // If PdS_major is MN, then we need to "transpose" the write.
  static constexpr int kNumPdSStore = kBlockM * kBlockN / NumConsumerThreads;
  using SmemCopyAtomPdS = Copy_Atom<
      std::conditional_t<
          (!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
          std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
          std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>>,
      Element>;

  using GmemTiledCopyQdO = std::conditional_t<BwdInnerLoopK, cute::SM90_TMA_LOAD, decltype(gcd::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})))>;
  using GmemTiledCopyKV = std::conditional_t<BwdInnerLoopK, decltype(gcd::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{}))), cute::SM90_TMA_LOAD>;
  using GmemTiledCopyInnerDq = cute::SM90_TMA_REDUCE_ADD;
  using GmemTiledCopyInnerDkv = cute::SM90_TMA_REDUCE_ADD;

  using ShapeQKV = cute::Shape<int32_t, Int<kHeadDim>, int32_t>; // (seqlen, head_dim, num_heads)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t>;
  using ShapeLSE = cute::Shape<_4, int32_t, int32_t>; // (4, seqlen_q, num_heads_q)
  using StrideLSE = cute::Stride<_1, _4, int64_t>;

  // Define ShapeLSETMA and StrideLSETMA based on PackGQA and CatGQA,
  // which will be used for loading LSE and dPsum from global memory to shared memory
  using ShapeLSETMA = std::conditional_t<
      PackGQA,
      // (4, (qhead_per_khead, seqlen_q), nheads_kv)
      cute::Shape<_4, cute::Shape<cute::Int<PackGQAFactor>, int32_t>, int32_t>,
      std::conditional_t<
          CatGQA,
          // (4, seqlen_q, (qhead_per_khead, nheads_kv))
          cute::Shape<_4, int32_t, cute::Shape<cute::Int<PackGQAFactor>, int32_t>>,
          // (4, seqlen_q, num_heads_q)
          ShapeLSE>>;
  using StrideLSETMA = std::conditional_t<
      PackGQA,
      // (1, (head_stride, 4), head_stride * qhead_per_khead)
      cute::Stride<_1, cute::Stride<int64_t, _4>, int64_t>,
      std::conditional_t<
          CatGQA,
          // (1, 4, (head_stride, head_stride * qhead_per_khead))
          cute::Stride<_1, _4, cute::Stride<int64_t, int64_t>>,
          // (1, 4, head_stride)
          StrideLSE>>;

  // Define ShapeQdOdQTMA and StrideQdOdQTMA based on PackGQA and CatGQA,
  // which will be used for loading Q and dO from global memory to shared memory for TMA
  using ShapeQdOdQTMA = std::conditional_t<
      PackGQA,
      // Case 1: PackGQA is enabled
      // Shape: ((qhead_per_khead, seqlen), headdim, nheads_kv)
      cute::Shape<cute::Shape<cute::Int<PackGQAFactor>, int32_t>, Int<kHeadDim>, int32_t>,
      std::conditional_t<
          CatGQA,
          // Case 2: CatGQA is enabled
          // Shape: (seqlen, headdim, (qhead_per_khead, nheads_kv))
          cute::Shape<int32_t, Int<kHeadDim>, cute::Shape<cute::Int<PackGQAFactor>, int32_t>>,
          // Case 3: Default case (neither Pack nor Cat)
          ShapeQKV>>;
  using StrideQdOdQTMA = std::conditional_t<
      PackGQA,
      // Case 1: PackGQA is enabled
      // Stride corresponding to: ((qhead_per_khead, seqlen), headdim, nheads_kv)
      cute::Shape<cute::Shape<int64_t, int64_t>, _1, int64_t>,
      std::conditional_t<
          CatGQA,
          // Case 2: CatGQA is enabled
          // Stride corresponding to: (seqlen, headdim, (qhead_per_khead, nheads_kv))
          cute::Shape<int64_t, _1, cute::Shape<int64_t, int64_t>>,
          // Case 3: Default case
          StrideQKV>>;

  // Declare the TMA operand types for Q, dO, K, V, inner dQ and inner dKV.
  // TMA_QdO: non-packed path (flat shape), used when !PackGQA && !CatGQA
  using TMA_QdO = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQdO{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      take<0, 2>(SmemLayoutQ{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along N mode for this M load, if any
  // TMA_QdO_Packed: packed path (nested shape), used when PackGQA || CatGQA.
  // Uses make_tma_copy (not _A_sm90) with a 2D tile to avoid identity-compose
  // issues with nested shapes in make_tma_copy_A_sm90.
  using TMA_QdO_Packed = decltype(make_tma_copy(
      GmemTiledCopyQdO{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQdOdQTMA{}, StrideQdOdQTMA{}),
      take<0, 2>(SmemLayoutQ{}),
      select<0, 2>(TileShape_MNK{}),
      size<1>(ClusterShape{}))); // mcast along N
  using TMA_QdO_Store = std::conditional_t<FlattenGQA, TMA_QdO_Packed, TMA_QdO>;
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
      GmemTiledCopyInnerDq{},
      make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQdOdQTMA{}, StrideQdOdQTMA{}),
      SmemLayoutdQSwizzled{},
      TileShape_InnerDq{},
      _1{})); // no mcast for partial dQ

  // q for outer-loop and k for inner-loop
  using TMA_add_dKV = decltype(make_tma_copy(
      GmemTiledCopyInnerDkv{},
      make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQKV{}, StrideQKV{}),
      SmemLayoutdKVSwizzled{},
      TileShape_InnerDkv{},
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
  // dQ_use_smem gates the SMEM store infrastructure for the inner-loop dQ:
  // an smem buffer + r2s copy + handshake with a store agent. False only for hdim 256,
  // where smem cannot fit the buffer and the kernel falls back to direct per-thread
  // atomicAdd from registers (see Slice_dQKV_Mma below).
  static constexpr bool dQ_use_smem = kHeadDim < 256;
  // For dKV, the same gate is expressed directly via InnerStoreMode::BypassSmem (set from Python
  // when kHeadDim >= 256). All sparse scatter paths require kHeadDim < 256 (smem buffer).
  static_assert((kInnerStoreMode == InnerStoreMode::BypassSmem) == (kHeadDim >= 256), "BypassSmem <=> kHeadDim >= 256; enforce in Python prebuild");
  // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
  // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
  static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && !dQ_use_smem && dQ_swapAB && AtomLayoutMdQ == 1 && NumConsumerWarpGroups == 2;
  static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");
  static_assert(!(Slice_dQKV_Mma && Mma_dKV_is_RS), "When enabling Slice_dQKV_Mma, we can't use Mma_dKV_is_RS");

  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
  // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
  static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
  static constexpr size_t SmemAlignmentV = SmemAlignmentQKVdO;
  static constexpr size_t SmemAlignmentLSE = 128, SmemAlignmentdPsum = 128;
  static constexpr size_t maxSmemAlignment = cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO, SmemAlignmentV, SmemAlignmentLSE, SmemAlignmentdPsum);
  static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");

  // TODO: do we have to worry that smem_dk and smem_dv in the epilogue don't line up with smem_k and smem_v due to alignment?
  // Accum buffers are sized for the Store layout (the padded scatter layout is slightly larger
  // than the swizzled TMA layout; they alias the same buffer, only one is live per build).
  using SmemInnerDq_t = std::conditional_t<
      !dQ_use_smem,
      cute::array<ElementAccum, 0>,
      cute::array_aligned<ElementAccum, cute::max(cute::cosize_v<SmemLayoutdQSwizzled>, cute::cosize_v<SmemLayoutdQStore>)>>;
  using SmemInnerDkv_t = std::conditional_t<
      (kInnerStoreMode == InnerStoreMode::BypassSmem),
      cute::array<ElementAccum, 0>,
      cute::array_aligned<ElementAccum, cute::max(cute::cosize_v<SmemLayoutdKVSwizzled>, cute::cosize_v<SmemLayoutdKVStore>)>>;
  using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP1>, SmemAlignmentP>>;

  // ─── Per-iteration token-index slots in smem (single source of truth for scatter paths) ───
  // Only the inner (sparse-side) tensor has token indices at all — the outer tensor is dense
  // TMA-loaded — so the slots are sized by the inner tile (kInnerTileSize):
  //   InnerLoopQ (!BwdInnerLoopK): inner = Q  (kBlockM rows) → read by the dQ scatter store.
  //   InnerLoopK ( BwdInnerLoopK): inner = KV (kBlockN rows) → read by the dKV scatter store.
  //
  // SmemSparseInnerIndices: kStages stage-indexed slots, 1:1 with the inner-tensor pipeline
  // buffers (pipeline_q on InnerLoopQ, pipeline_k on InnerLoopK). Every access is protected
  // by an existing synchronization:
  //  - loader writes slot PipelineState::index() right after producer_acquire (stage held);
  //  - loader self-reads (dO/dPsum on InnerLoopQ, V on InnerLoopK) are same-warp program-ordered
  //    before any write that could reuse the slot;
  //  - consumer scatter stores (!InnerStoreInProducer) read their own read-state index()
  //    while still holding the stage (consumer_release comes after the scatter).
  using SmemSparseInnerIndices = std::conditional_t<IsSparse, cute::array<int, kInnerTileSize * kStages>, cute::array<int, 0>>;

  // SmemSparseStoreStagingIndices: fixed staging copies for the producer store warps
  // (InnerStoreInProducer only). Those warps have NO stage protection — the consumer releases
  // the stage right after arriving dXFull, so the loader may rewrite the pipeline slot while
  // the store warp still reads it. Consumer WG0 therefore copies the current slot here inside
  // the dXEmpty→dXFull window (stage still held there, and the store warp is blocked on dXFull).
  // InnerLoopK needs separate dV/dK staging ([staging_dv][staging_dk]): store_dV's dVEmpty
  // arrive lets the consumer's next-iteration dV r2s overwrite a shared staging while store_dK
  // would still be reading it. InnerLoopQ needs one ([staging_dq]). See .tmp/058 NOTES P7.
  static constexpr int kNumStoreStagingSlots = !InnerStoreInProducer ? 0 : (BwdInnerLoopK ? 2 : 1);
  using SmemSparseStoreStagingIndices = std::conditional_t<IsSparse, cute::array<int, kInnerTileSize * kNumStoreStagingSlots>, cute::array<int, 0>>;

  struct TensorStorageLoopQ : cute::aligned_struct<maxSmemAlignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentLSE> smem_lse;
    cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentdPsum> smem_dpsum;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    SmemInnerDq_t smem_inner_dq;
    SmemSparseInnerIndices smem_sparse_inner_indices;
    SmemSparseStoreStagingIndices smem_sparse_store_staging_indices;
  };

  // Empty placeholder for zero-sized SMEM fields (used with [[no_unique_address]])
  struct SmemP_Empty_ {
    CUTLASS_HOST_DEVICE Element* data() {
      return nullptr;
    }
    CUTLASS_HOST_DEVICE const Element* data() const {
      return nullptr;
    }
  };

  using SmemLSE_t = cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentLSE>;
  using SmemDPsum_t = cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentdPsum>;
  // InnerLoopK keeps P with kStages_dS stages (same as dS), unlike InnerLoopQ which uses 1-stage P.
  using SmemP_LoopK_t = std::conditional_t<Mma_dKV_is_RS, SmemP_Empty_, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;

  struct TensorStorageLoopK : cute::aligned_struct<maxSmemAlignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
    // dK and dV SMEM buffers: when UnionDkvSmem = false (default), uses separate buffers
    // allowing dV/dK r2s to overlap (independent barrier paths, +11% perf).
    // UnionDkvSmem = true: union saves 32 KB but forces serialized stores via the swapped
    // dVEmpty/dKEmpty barrier protocol in store_dkv().
    // kInnerStoreStages (1 or 2) controls pipeline depth; arrays of size 1 degenerate to single-buffer.
    struct SeparatedDkvSmem {
      SmemInnerDkv_t smem_inner_dk[kInnerStoreStages];
      SmemInnerDkv_t smem_inner_dv[kInnerStoreStages];
    };
    struct UnionedDkvSmem {
      struct Stage {
        union {
          SmemInnerDkv_t smem_inner_dk;
          SmemInnerDkv_t smem_inner_dv;
        };
      };
      Stage stages[kInnerStoreStages];
    };
    using DkvSmemStorage = std::conditional_t<UnionDkvSmem, UnionedDkvSmem, SeparatedDkvSmem>;
    DkvSmemStorage dkv_smem_storage;

    SmemSparseInnerIndices smem_sparse_inner_indices;
    SmemSparseStoreStagingIndices smem_sparse_store_staging_indices;
    // Zero-sized fields placed AFTER all data buffers so they fall in struct tail padding
    // (struct alignment from PdS swizzle is 1024B; core data sums to exactly N*1024).
    // [[no_unique_address]] on truly-empty types lets the compiler overlap them with
    // tail padding, avoiding a 1KB bump to the next alignment boundary.
    SmemLSE_t smem_lse;
    SmemDPsum_t smem_dpsum;
    [[no_unique_address]] SmemP_LoopK_t smem_p;
  };

  using TensorStorage = std::conditional_t<BwdInnerLoopK, TensorStorageLoopK, TensorStorageLoopQ>;

  // ─── inner dKV SMEM buffer accessors (uniform API across single/double-buffer + union/separate) ───
  // Takes the mainloop's TensorStorage (= TensorStorageLoopK / TensorStorageLoopQ).
  // Callers from the kernel level should pass shared_storage.tensors.mainloop.
  CUTLASS_DEVICE static auto* smem_inner_dk_ptr(TensorStorage& ts, int stage = 0) {
    auto& st = ts.dkv_smem_storage;
    if constexpr (UnionDkvSmem) {
      return st.stages[stage].smem_inner_dk.data();
    } else {
      return st.smem_inner_dk[stage].data();
    }
  }
  CUTLASS_DEVICE static auto* smem_inner_dv_ptr(TensorStorage& ts, int stage = 0) {
    auto& st = ts.dkv_smem_storage;
    if constexpr (UnionDkvSmem) {
      return st.stages[stage].smem_inner_dv.data();
    } else {
      return st.smem_inner_dv[stage].data();
    }
  }

  // Host side kernel arguments
  struct Arguments {
    /* ptr for Q, dO and dQ */
    Element const* const ptr_Q;
    Element const* const ptr_dO;
    ElementAccum* const ptr_dQ;
    /* Q, dO and dQ use same shape */
    ShapeQKV const shape_QdOdQ;
    /* Q, dO and dQ can use different stride */
    StrideQKV const stride_Q;
    StrideQKV const stride_dO;
    StrideQKV const stride_dQ;
    /* ptr for K, V, dK and dV */
    Element const* const ptr_K;
    Element const* const ptr_V;
    ElementAccum* const ptr_dK;
    ElementAccum* const ptr_dV;
    /* K, V use shape_KVdKdV; dK, dV use shape_dKdV */
    ShapeQKV const shape_KVdKdV;
    ShapeQKV const shape_dKdV;
    /* K, V, dK and dV can use different stride */
    StrideQKV const stride_K;
    StrideQKV const stride_V;
    StrideQKV const stride_dK;
    StrideQKV const stride_dV;
    /* ptr for LSE_log2 and dPsum */
    float const* const ptr_LSE_log2;
    float const* const ptr_dPsum;
    /* LSE_log2 and dPsum use same shape */
    ShapeLSE const shape_LSEdPsum;
    ;
    /* LSE_log2 and dPsum can use different stride */
    StrideLSE const stride_LSE;
    StrideLSE const stride_dPsum;
    /* other meta data used by kernel */
    float const softmax_scale;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map = nullptr;
    int const* const cu_batches = nullptr;
    int* dq_determin_conflict_state;
    int* dq_determin_range_locks;
    /* index_sparse */
    int const* const index_sparse_indices;
    int inner_indices_cnt;
  };

  // Device side kernel params
  struct Params {
    /* */
    ShapeQdOdQTMA const shape_QdOdQ;
    /* */
    Element const* const ptr_K;
    StrideQKV const stride_K;
    Element const* const ptr_V;
    StrideQKV const stride_V;
    ElementAccum* const ptr_dK;
    ElementAccum* const ptr_dV;
    ShapeQKV const shape_KVdKdV;
    ShapeQKV const shape_dKdV;
    StrideQKV const stride_dK;
    StrideQKV const stride_dV;
    /* */
    TMA_QdO_Store tma_load_Q, tma_load_dO;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    TMA_add_dQ tma_add_dQ;
    TMA_add_dKV tma_add_dK;
    TMA_add_dKV tma_add_dV;
    /* */
    float const* const ptr_LSE_log2;
    float const* const ptr_dPsum;
    ShapeLSETMA const shape_LSEdPsum;
    StrideLSETMA const stride_LSE;
    StrideLSETMA const stride_dPsum;
    /* */
    cutlass::FastDivmod qhead_per_khead_divmod;
    /* other meta data used by kernel */
    float const softmax_scale;
    float const softmax_scale_log2;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const n_block_max_num;
    int const* const attn_type_map = nullptr;
    int const* const cu_batches = nullptr;
    /* deterministic */
    int* dq_determin_conflict_state;
    int* dq_determin_range_locks;
    /* sparse load (InnerLoopQ: scatter Q/dO/dQ) */
    Element const* const ptr_Q;
    StrideQKV const stride_Q;
    Element const* const ptr_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQ;
    StrideQKV const stride_dQ;
    /* index_sparse */
    int const* const index_sparse_indices;
    int inner_indices_cnt;
  };

  // BlockSparse BlockMeta: templated on IsProducer and InnerLoopQ, other params fixed.
  template <bool IsProducer, bool InnerLoopQ>
  using BlockSparseBlockMetaT = flash::BlockSparseBlockMeta<
      IsProducer,
      RangeMerge,
      PackGQA,
      PackGQAFactor,
      ScatterLdst::kTokensPerGroup,
      ScatterLdst::kThreadsPerGroup,
      NumProducerThreads,
      InnerLoopQ ? kBlockM : kBlockN,
      InnerDirMaxToMin,
      InnerLoopQ>;

  template <bool IsProducer>
  using BlockSparseLoopKBlockMeta = BlockSparseBlockMetaT<IsProducer, false>;
  template <bool IsProducer>
  using BlockSparseLoopQBlockMeta = BlockSparseBlockMetaT<IsProducer, true>;

  static Params to_underlying_arguments(Arguments const& args) {
    if constexpr (Deterministic) {
      // In deterministic mode, we use atomic operations to update dQ,
      // which requires extra arguments to manage conflicts.
      // We assert that these arguments are not null.
      assert(args.dq_determin_conflict_state != nullptr);
      assert(args.dq_determin_range_locks != nullptr);
    }

    // Create shape for Q, dO and dQ
    auto const shape_QdOdQ = cute::conditional_return<PackGQA>(
        make_shape(
            make_shape(cute::Int<PackGQAFactor>{}, get<0>(args.shape_QdOdQ)), // (qhead_per_khead, seqlen)
            get<1>(args.shape_QdOdQ), // headdim
            get<2>(args.shape_KVdKdV) // nheads_kv
            ),
        cute::conditional_return<CatGQA>(
            make_shape(
                get<0>(args.shape_QdOdQ), // seqlen
                get<1>(args.shape_QdOdQ), // headdim
                make_shape(cute::Int<PackGQAFactor>{}, get<2>(args.shape_KVdKdV)) // (qhead_per_khead, nheads_kv)
                ),
            args.shape_QdOdQ));
    // Create stride for Q, dO and dQ
    auto const stride_Q = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), // (q_head_stride, row_stride)
            get<1>(args.stride_Q), // 1
            get<2>(args.stride_Q) * PackGQAFactor // qhead_per_khead * q_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_Q), // row_stride
                get<1>(args.stride_Q), // 1
                make_stride(get<2>(args.stride_Q), get<2>(args.stride_Q) * PackGQAFactor) // (q_head_stride, qhead_per_khead * q_head_stride)
                ),
            args.stride_Q));
    auto const stride_dO = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_dO), get<0>(args.stride_dO)), // (do_head_stride, row_stride)
            get<1>(args.stride_dO), // 1
            get<2>(args.stride_dO) * PackGQAFactor // qhead_per_khead * do_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_dO), // row_stride
                get<1>(args.stride_dO), // 1
                make_stride(get<2>(args.stride_dO), get<2>(args.stride_dO) * PackGQAFactor) // (do_head_stride, qhead_per_khead * do_head_stride)
                ),
            args.stride_dO));
    auto const stride_dQ = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_dQ), get<0>(args.stride_dQ)), // (dq_head_stride, row_stride)
            get<1>(args.stride_dQ), // 1
            get<2>(args.stride_dQ) * PackGQAFactor // qhead_per_khead * dq_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_dQ), // row_stride
                get<1>(args.stride_dQ), // 1
                make_stride(get<2>(args.stride_dQ), get<2>(args.stride_dQ) * PackGQAFactor) // (dq_head_stride, qhead_per_khead * dq_head_stride)
                ),
            args.stride_dQ));

    // Create TMA for loading Q and dO, and for adding to dQ.
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), make_layout(shape_QdOdQ, stride_Q));
    TMA_QdO_Store tma_load_Q = [&] {
      if constexpr (FlattenGQA) {
        return make_tma_copy(GmemTiledCopyQdO{}, mQ, take<0, 2>(SmemLayoutQ{}), select<0, 2>(TileShape_MNK{}), size<1>(ClusterShape{}));
      } else {
        return make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mQ, take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{});
      }
    }();
    Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), make_layout(shape_QdOdQ, stride_dO));
    TMA_QdO_Store tma_load_dO = [&] {
      if constexpr (FlattenGQA) {
        return make_tma_copy(GmemTiledCopyQdO{}, mdO, take<0, 2>(SmemLayoutdO{}), select<0, 2>(TileShape_MNK{}), size<1>(ClusterShape{}));
      } else {
        return make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mdO, take<0, 2>(SmemLayoutdO{}), TileShape_MNK{}, ClusterShape{});
      }
    }();
    // dQ TMA (add/store, not load) uses nested shape directly
    Tensor mdQ = make_tensor(make_gmem_ptr(args.ptr_dQ), make_layout(shape_QdOdQ, stride_dQ));
    TMA_add_dQ tma_add_dQ = make_tma_copy(GmemTiledCopyInnerDq{}, mdQ, SmemLayoutdQSwizzled{}, TileShape_InnerDq{}, _1{});

    /* DEBUG */
    // printf("====================== mQ: ======================\n");
    // cute::print(mQ.layout());
    // printf("\n====================== mdO: ======================\n");
    // cute::print(mdO.layout());
    // printf("\n====================== mdQ: ======================\n");
    // cute::print(mdQ.layout());

    // Create TMA for loading K and V (use shape_KVdKdV = original K/V shape)
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), make_layout(args.shape_KVdKdV, args.stride_K));
    TMA_K tma_load_K = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{});
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), make_layout(args.shape_KVdKdV, args.stride_V));
    TMA_V tma_load_V = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mV, take<0, 2>(SmemLayoutV{}), TileShape_MNK{}, ClusterShape{});
    // dK/dV TMA use shape_dKdV (same as shape_KVdKdV)
    Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), make_layout(args.shape_dKdV, args.stride_dK));
    TMA_add_dKV tma_add_dK = make_tma_copy(GmemTiledCopyInnerDkv{}, mdK, SmemLayoutdKVSwizzled{}, TileShape_InnerDkv{}, _1{});
    Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), make_layout(args.shape_dKdV, args.stride_dV));
    TMA_add_dKV tma_add_dV = make_tma_copy(GmemTiledCopyInnerDkv{}, mdV, SmemLayoutdKVSwizzled{}, TileShape_InnerDkv{}, _1{});

    /* DEBUG */
    // printf("====================== mK: ======================\n");
    // cute::print(mK.layout());
    // printf("====================== mV: ======================\n");
    // cute::print(mV.layout());
    // printf("====================== mdK: ======================\n");
    // cute::print(mdK.layout());
    // printf("====================== mdV: ======================\n");
    // cute::print(mdV.layout());

    // Create shape for LSE and dPsum
    auto const shape_LSEdPsum = cute::conditional_return<PackGQA>(
        make_shape(
            _4{},
            make_shape(cute::Int<PackGQAFactor>{}, get<1>(args.shape_LSEdPsum)), // (qhead_per_khead, seqlen_q)
            get<2>(args.shape_KVdKdV) // nheads_kv
            ),
        cute::conditional_return<CatGQA>(
            make_shape(
                _4{},
                get<1>(args.shape_LSEdPsum), // seqlen_q
                make_shape(cute::Int<PackGQAFactor>{}, get<2>(args.shape_KVdKdV)) // (qhead_per_khead, nheads_kv)
                ),
            args.shape_LSEdPsum));
    // Create stride for LSE and dPsum
    auto const stride_LSE = cute::conditional_return<PackGQA>(
        make_stride(
            _1{},
            make_stride(get<2>(args.stride_LSE), get<1>(args.stride_LSE)), // (head_stride, 4)
            get<2>(args.stride_LSE) * PackGQAFactor // (qhead_per_khead * head_stride)
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                _1{},
                get<1>(args.stride_LSE), // 4
                make_stride(get<2>(args.stride_LSE), get<2>(args.stride_LSE) * PackGQAFactor) // (head_stride, qhead_per_khead * head_stride)
                ),
            args.stride_LSE));
    auto const stride_dPsum = cute::conditional_return<PackGQA>(
        make_stride(
            _1{},
            make_stride(get<2>(args.stride_dPsum), _4{}), // (head_stride, 4)
            get<2>(args.stride_dPsum) * PackGQAFactor),
        cute::conditional_return<CatGQA>(
            make_stride(
                _1{},
                get<1>(args.stride_dPsum), // 4
                make_stride(get<2>(args.stride_dPsum), get<2>(args.stride_dPsum) * PackGQAFactor) // (head_stride, qhead_per_khead * head_stride)
                ),
            args.stride_dPsum));

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
        shape_QdOdQ,
        args.ptr_K,
        args.stride_K,
        args.ptr_V,
        args.stride_V,
        args.ptr_dK,
        args.ptr_dV,
        args.shape_KVdKdV,
        args.shape_dKdV,
        args.stride_dK,
        args.stride_dV,
        tma_load_Q,
        tma_load_dO,
        tma_load_K,
        tma_load_V,
        tma_add_dQ,
        tma_add_dK,
        tma_add_dV,
        args.ptr_LSE_log2,
        args.ptr_dPsum,
        shape_LSEdPsum,
        stride_LSE,
        stride_dPsum,
        /*qhead_per_khead_divmod=*/cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_QdOdQ), get<2>(args.shape_KVdKdV))),
        /*softmax_scale=*/args.softmax_scale,
        /*softmax_scale_log2=*/!Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        /*softcap_val=*/!Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        /*q_ranges=*/args.q_ranges,
        /*k_ranges=*/args.k_ranges,
        /*n_block_max_num=*/!BwdInnerLoopK ? cute::ceil_div(get<0>(args.shape_KVdKdV), kBlockN) : cute::ceil_div(get<0>(args.shape_QdOdQ), kBlockM),
        /*attn_type_map=*/args.attn_type_map,
        /*cu_batches=*/args.cu_batches,
        /*dq_determin_conflict_state=*/args.dq_determin_conflict_state,
        /*dq_determin_range_locks=*/args.dq_determin_range_locks,
        /*ptr_Q=*/args.ptr_Q,
        /*stride_Q=*/args.stride_Q,
        /*ptr_dO=*/args.ptr_dO,
        /*stride_dO=*/args.stride_dO,
        /*ptr_dQ=*/args.ptr_dQ,
        /*stride_dQ=*/args.stride_dQ,
        /*index_sparse_indices=*/args.index_sparse_indices,
        /*inner_indices_cnt=*/args.inner_indices_cnt};
  }

  // BlockMeta type alias — definition lives in block_meta.h
  // InnerLoopQ mapping:
  //   BwdInnerLoopK=true  → inner loop over n_block (InnerLoopK) → InnerLoopQ=false
  //   BwdInnerLoopK=false → inner loop over m_block (InnerLoopQ) → InnerLoopQ=true
  // So: InnerLoopQ = !BwdInnerLoopK
  template <bool IsProducer>
  using BlockMeta = flash::DenseBlockMeta<IsProducer, /*InnerLoopQ=*/!BwdInnerLoopK, RangeMerge, /*FlattenGQA=*/FlattenGQA, PackGQAFactor, SeqlenInfo_t, BlockMN_t>;

  // IndexSparse BlockMeta: templated on IsProducer and InnerLoopQ, other params fixed.
  template <bool IsProducer, bool InnerLoopQ>
  using IndexSparseBlockMetaT = flash::IndexSparseBlockMeta<
      IsProducer,
      PackGQA,
      PackGQAFactor,
      ScatterLdst::kTokensPerGroup,
      NumProducerThreads,
      ScatterLdst::kThreadsPerGroup,
      InnerLoopQ ? kBlockM : kBlockN,
      InnerDirMaxToMin,
      SparseKBlockSize,
      InnerLoopQ>;

  template <bool IsProducer>
  using IndexSparseLoopKBlockMeta = IndexSparseBlockMetaT<IsProducer, false>;
  template <bool IsProducer>
  using IndexSparseLoopQBlockMeta = IndexSparseBlockMetaT<IsProducer, true>;

  // Unified sparse aliases — dispatch to IndexSparse or BlockSparse based on template config.
  template <bool IsProducer>
  using SparseLoopQBlockMeta = std::conditional_t<IndexSparse, IndexSparseBlockMetaT<IsProducer, true>, BlockSparseBlockMetaT<IsProducer, true>>;
  template <bool IsProducer>
  using SparseLoopKBlockMeta = std::conditional_t<IndexSparse, IndexSparseBlockMetaT<IsProducer, false>, BlockSparseBlockMetaT<IsProducer, false>>;

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    // K/V TMA descriptors needed for dense and scatter-TMA paths (kInnerLoadMode == InnerLoadMode::Tma).
    // Only cp.async-only scatter (IndexSparse etc) skips the prefetch.
    if constexpr (!IsSparse || kInnerLoadMode == InnerLoadMode::Tma) {
      cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }
  }

  // ─── TMA 2D reduce-add: full-service helper for inner/outer TMA stores ───
  // Computes partition_S(smem), local_tile(model), partition_D(gmem), then issues
  // one TMA reduce-add instruction with arrive+wait.
  // Caller is responsible for thread predication and any pre/post Deterministic sync.
  // idx... selects the block position in the tiled gmem (supports 1D or 2D indexing for CatGQA).
  template <typename TmaCopy, typename SmemTensor, typename ModelTensor, typename TileShape, typename Coord, typename... Idx>
  CUTLASS_DEVICE static void tma_inner_store(
      TmaCopy const& tma,
      SmemTensor const& smem,
      ModelTensor&& model_tensor,
      TileShape tile_shape,
      Coord const& coord,
      Idx... idx) {
    auto block_tma = tma.get_slice(_0{});
    Tensor src = block_tma.partition_S(smem);
    Tensor gmem_tiled = local_tile(std::forward<ModelTensor>(model_tensor), tile_shape, coord);
    Tensor dst = block_tma.partition_D(gmem_tiled);
    cute::copy(tma, src, dst(_, _, _, idx...));
    tma_store_arrive();
    tma_store_wait<0>();
  }

  // ─── Unified scatter dX store (6 call sites funnel here) ───
  // Reduce-add kTileSize entries of SMEM store buffer into non-contiguous GMEM positions
  // addressed by sparse_indices[i]. Two modes via kInnerStoreMode:
  //   Tma1d:     per-entry cp.reduce.async.bulk (kHeadDim*4B, contiguous SMEM layout)
  //   AtomicAdd: per-element atomicAdd (kLaneBytes/sizeof(Accum) elems per thread)
  // kInnerStoreHeadPackFactor: GQA head-packing factor for compound indices (token_idx*G+g).
  //   dQ calls pass PackGQAFactor (compound_idx needs token/head decomposition);
  //   dK/dV calls pass 1 (plain token indices, no head packing).
  template <int kTileSize, int kNumThreads, int kInnerStoreHeadPackFactor, typename SmemStoreT>
  CUTLASS_DEVICE static void scatter_inner_store(
      SmemStoreT const& s_store,
      int const* sparse_indices,
      ElementAccum* gmem_base,
      int const stride_token,
      int const thread_idx,
      int const tile_offset = 0,
      int const stride_head = 0) {
    auto gmem_ptr = [&](int compound_idx) -> ElementAccum* {
      if constexpr (kInnerStoreHeadPackFactor != 1) {
        return gmem_base + static_cast<int64_t>(compound_idx / kInnerStoreHeadPackFactor) * stride_token +
            static_cast<int64_t>(compound_idx % kInnerStoreHeadPackFactor) * stride_head;
      } else {
        return gmem_base + static_cast<int64_t>(compound_idx) * stride_token;
      }
    };
    if constexpr ((kInnerStoreMode == InnerStoreMode::Tma1d)) {
      static constexpr int32_t kEntryBytes = kHeadDim * sizeof(ElementAccum);
      bool issued = false;
      for (int i = thread_idx; i < kTileSize; i += kNumThreads) {
        ElementAccum* const dst = gmem_ptr(sparse_indices[i]);
        cute::SM90_BULK_REDUCE_ADD::copy(&s_store(tile_offset + i, _0{}), dst, kEntryBytes);
        issued = true;
      }
      if (issued) {
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
      }
    } else {
      using Ldst = ScatterLdstGroup<kNumThreads, kTileSize>;
      static constexpr int kElemsPerLane = Ldst::kLaneBytes / sizeof(ElementAccum);
      static constexpr int kElemsPerRow = Ldst::kBankRowBytes / sizeof(ElementAccum);
      static constexpr int kTilesPerRow = kHeadDim / kElemsPerRow;
      int const group_idx = thread_idx / Ldst::kThreadsPerGroup;
      int const lane_in_group = thread_idx % Ldst::kThreadsPerGroup;
      CUTE_UNROLL
      for (int local_i = 0; local_i < Ldst::kTokensPerGroup; ++local_i) {
        int const i = group_idx * Ldst::kTokensPerGroup + local_i;
        ElementAccum* const dst = gmem_ptr(sparse_indices[i]);
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < kTilesPerRow; ++tile_idx) {
          int const col_base = lane_in_group * kElemsPerLane + tile_idx * kElemsPerRow;
          CUTE_UNROLL
          for (int v = 0; v < kElemsPerLane; ++v) {
            atomicAdd(dst + col_base + v, s_store(tile_offset + i, col_base + v));
          }
        }
      }
    }
  }

  // ─── CpAsync scatter load: load-side counterpart of scatter_inner_store ───
  // Scatter-loads kInnerTileSize tokens from non-contiguous GMEM positions into pipelined SMEM.
  // kHeadPackFactor: GQA compound index factor (>1 → compound_idx = token_idx*G+g decomposition).
  // Main tensor: Element-typed, kHeadDim wide. Thread decomposition mirrors scatter_inner_store.
  // ptr_main_base: pre-offset to current head + lane * (kLaneBytes/sizeof(Element)).
  template <int kHeadPackFactor = 1, typename SmemMainT>
  CUTLASS_DEVICE static void scatter_inner_load(
      SmemMainT& smem_main,
      Element const* ptr_main_base,
      int64_t stride_token,
      int64_t stride_head,
      int const* compound_indices,
      int stage,
      int thread_idx) {
    static constexpr int kElemsPerLane = ScatterLdst::kLaneBytes / sizeof(Element);
    static constexpr int kElemsPerRow = ScatterLdst::kBankRowBytes / sizeof(Element);
    static constexpr int kTilesPerRow = kHeadDim / kElemsPerRow;
    using CpAsyncCg = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
    CpAsyncCg const cp_async_cg{};
    int const group_lane = thread_idx % ScatterLdst::kThreadsPerGroup;
    int const group_idx = thread_idx / ScatterLdst::kThreadsPerGroup;

    CUTE_UNROLL
    for (int local_row = 0; local_row < ScatterLdst::kTokensPerGroup; ++local_row) {
      int smem_row = group_idx * ScatterLdst::kTokensPerGroup + local_row;
      int64_t token_offset;
      if constexpr (kHeadPackFactor > 1) {
        int const ci = compound_indices[smem_row];
        token_offset = static_cast<int64_t>(ci / kHeadPackFactor) * stride_token + (ci % kHeadPackFactor) * stride_head;
      } else {
        token_offset = static_cast<int64_t>(compound_indices[smem_row]) * stride_token;
      }
      CUTE_UNROLL
      for (int tile_idx = 0; tile_idx < kTilesPerRow; ++tile_idx) {
        if (group_lane * kElemsPerLane + tile_idx * kElemsPerRow < kHeadDim) {
          Element* dst_ptr = &smem_main(smem_row, group_lane * kElemsPerLane + tile_idx * kElemsPerRow, stage);
          auto g_src = make_tensor(make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_main_base + token_offset + tile_idx * kElemsPerRow)), Layout<_1>{});
          auto s_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(dst_ptr)), Layout<_1>{});
          cute::copy(cp_async_cg, g_src, s_dst);
        }
      }
    }
  }

  // Extended overload: main tensor + scalar tensor (LSE/dPsum, 4 floats per token).
  // scalar_offset_fn: callable taking compound_idx, returning byte offset into ptr_scalar_base.
  template <int kHeadPackFactor, typename SmemMainT, typename SmemScalarT, typename ScalarOffsetFn>
  CUTLASS_DEVICE static void scatter_inner_load(
      SmemMainT& smem_main,
      Element const* ptr_main_base,
      int64_t stride_token,
      int64_t stride_head,
      int const* compound_indices,
      int stage,
      int thread_idx,
      SmemScalarT& smem_scalar,
      float const* ptr_scalar_base,
      ScalarOffsetFn&& scalar_offset_fn) {
    scatter_inner_load<kHeadPackFactor>(smem_main, ptr_main_base, stride_token, stride_head, compound_indices, stage, thread_idx);

    using CpAsyncCg = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
    CpAsyncCg const cp_async_cg{};
    int const group_lane = thread_idx % ScatterLdst::kThreadsPerGroup;
    int const group_idx = thread_idx / ScatterLdst::kThreadsPerGroup;

    for (int i = group_lane; i < ScatterLdst::kTokensPerGroup; i += ScatterLdst::kThreadsPerGroup) {
      int smem_idx = group_idx * ScatterLdst::kTokensPerGroup + i;
      float* s_dst = &smem_scalar(_0{}, smem_idx, stage);
      auto g_src = make_tensor(make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_scalar_base + scalar_offset_fn(compound_indices[smem_idx]))), Layout<_1>{});
      auto s_dst_t = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(s_dst)), Layout<_1>{});
      cute::copy(cp_async_cg, g_src, s_dst_t);
    }
  }

  // Perform a Producer Prologue/Mainloop -- TMA Load for K,V, with pipelining multi-stage TMA load for Q,dO,LSE,dPsum
  // k for outer-loop and q for inner-loop
  // When BlockSparse (InnerLoopQ): Q/dO/LSE/dPsum are scatter-loaded via cp.async, K/V are still TMA.
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE bool load_with_loop_q(
      Params const& params,
      MainloopPipeline pipeline_q,
      MainloopPipeline_dO pipeline_do,
      PipelineState& smem_pipe_write_q,
      PipelineState_dO& smem_pipe_write_do,
      SharedStorage& shared_storage,
      BlockMetaT& block_meta) {
    // Compile Guard Clause
    static_assert(!BwdInnerLoopK, "load_with_loop_q() must be called when BwdInnerLoopK is false");
    // The BlockSparse scatter loader has no per-q-head (bidh_kv_cat) loop, so CatGQA cannot
    // be expressed on this path yet. PackGQA is supported by walking q_ranges in packed-row
    // space instead (see BlockSparseBlockMeta::kScatterScale).
    static_assert(!(BlockSparse && CatGQA), "BlockSparse InnerLoopQ does not support CatGQA");

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_tile_idx;
    int const bidh = block_meta.bidh;
    int const bidh_kv = block_meta.bidh_kv;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;
    int m_block;
    int bidh_kv_cat;

    // Prepare for TMA multicast meta
    auto [mcast_mask_qdo, cluster_block_id_qdo] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyQdO, /*RowwiseMask=*/false>();

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
    Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

    // For PackGQA, offset needs to be multiplied by PackGQAFactor
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * PackGQAFactor;

    // Prepare for TMA loads
    auto const mQdOdQLSEdPsum_coord = make_coord(_, _, cute::conditional_return<CatGQA>(make_coord(_, bidh), bidh));
    auto const gQdOdQ_coord = cute::conditional_return<CatGQA>(make_coord(_, _0{}, _), make_coord(_, _0{}));
    auto const gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
    // get_tma_tensor + local_tile: use packed TMA for PackGQA/CatGQA, non-packed otherwise
    auto mQ = params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
    auto mdO = params.tma_load_dO.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
    // (M, K, _); for CatGQA: (M, K, _, _)
    Tensor gQ = local_tile(domain_offset(gQdO_offset_q_coord, mQ), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
    // (M, K, _); for CatGQA: (M, K, _, _)
    Tensor gdO = local_tile(domain_offset(gQdO_offset_q_coord, mdO), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);

    Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)
    Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{})); // (N, K)

    auto mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSEdPsum, params.stride_LSE)(
        mQdOdQLSEdPsum_coord); // (4, seqlen_q); for CatGQA: (4, seqlen_q, qhead_per_khead)
    auto mdPsum = make_tensor(
        make_gmem_ptr(params.ptr_dPsum), params.shape_LSEdPsum, params.stride_dPsum)(mQdOdQLSEdPsum_coord); // (4, seqlen_q); for CatGQA: (4, seqlen_q, qhead_per_khead)

    auto const gLSEdPsum_coord = cute::conditional_return<CatGQA>(make_coord(_0{}, _, _), make_coord(_0{}, _));
    auto const LSEdPsum_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(_0{}, offset_q, _0{}), make_coord(_0{}, offset_q));
    Tensor gLSE =
        local_tile(cute::domain_offset(LSEdPsum_offset_q_coord, mLSE), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord); // (4, M, _); for CatGQA: (4, M, _, _)
    Tensor gdPsum =
        local_tile(cute::domain_offset(LSEdPsum_offset_q_coord, mdPsum), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord); // (4, M, _); for CatGQA: (4, M, _, _)

    auto block_tma_Q = params.tma_load_Q.get_slice(cluster_block_id_qdo);
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    auto block_tma_dO = params.tma_load_dO.get_slice(cluster_block_id_qdo);
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));

    auto rebind_Q_tiles = [&](SeqlenInfo_t const& si) {
      if constexpr (!RangeMerge) {
        return;
      }
      offset_q = !PackGQA ? si.offset_q : si.offset_q * PackGQAFactor;
      auto const qdo_off = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
      gQ = local_tile(domain_offset(qdo_off, mQ), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
      gdO = local_tile(domain_offset(qdo_off, mdO), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
      tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
      tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
      auto const lse_off = cute::conditional_return<CatGQA>(make_coord(_0{}, offset_q, _0{}), make_coord(_0{}, offset_q));
      gLSE = local_tile(cute::domain_offset(lse_off, mLSE), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
      gdPsum = local_tile(cute::domain_offset(lse_off, mdPsum), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
    };

    // Sparse TMA Q/dO/LSE/dPsum: use domain_offset per-call at the runtime-computed origin,
    // then index tile 0 (since origin already points to the exact tile start).
    // Dense path uses the pre-built tQgQ/tdOgdO/gLSE/gdPsum with m_block indexing.

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

    /* DEBUG */
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("m_block_min: %d, m_block_max: %d, n_block: %d, bidh: %d, bidb: %d, bidh_kv: %d\n", m_block_min, m_block_max, n_block, bidh, bidb, bidh_kv);
    //   printf("seqlen_q: %d, seqlen_k: %d\n", seqlen_info.seqlen_q, seqlen_info.seqlen_k);
    //   printf("offset_q: %d, offset_k: %d\n", seqlen_info.offset_q, seqlen_info.offset_k);
    //   printf("attn_type: %d\n", attn_type);
    //   printf("params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ)=\n");
    //   cute::print(params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ).layout());
    //   printf("\n======================= mQ: =======================\n");
    //   cute::print(mQ.layout());
    //   printf("\n======================= gQ: =======================\n");
    //   cute::print(gQ.layout());
    //   printf("\n======================= tQgQ: =======================\n");
    //   cute::print(tQgQ.layout());
    //   printf("\n======================= tQsQ: =======================\n");
    //   cute::print(tQsQ.layout());
    //   printf("\n======================= tdOgdO: =======================\n");
    //   cute::print(tdOgdO.layout());
    //   printf("\n======================= tdOsdO: =======================\n");
    //   cute::print(tdOsdO.layout());
    //   printf("\n======================= mLSE: =======================\n");
    //   cute::print(mLSE.layout());
    //   printf("\n======================= gLSE: =======================\n");
    //   cute::print(gLSE.layout());
    //   printf("\n======================= Smem LSE: =======================\n");
    //   cute::print(sLSE.layout());
    //   printf("\n======================= mPsum: =======================\n");
    //   cute::print(mdPsum.layout());
    //   printf("\n======================= gdPsum: =======================\n");
    //   cute::print(gdPsum.layout());
    //   printf("\n======================= Smem dPsum: =======================\n");
    //   cute::print(sdPsum.layout());
    //   printf("\n======================= tKgK: =======================\n");
    //   cute::print(tKgK.layout());
    //   printf("\n======================= tKsK: =======================\n");
    //   cute::print(tKsK.layout());
    //   printf("\n======================= tVgV: =======================\n");
    //   cute::print(tVgV.layout());
    //   printf("\n======================= tVsV: =======================\n");
    //   cute::print(tVsV.layout());
    // }

    // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
    // int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
    // if (warp_idx_in_warpgroup == 0)
    //    BarrierManager::sync<NumConsumerThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::KVEmpty);

    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    int const lane_predicate = cute::elect_one_sync();

    // ─── BlockSparse InnerLoopQ scatter infra (DCE'd on dense path) ───
    int const thread_idx = threadIdx.x % NumProducerLoaderThreads;
    static constexpr int kElemsPerLane = ScatterLdst::kLaneBytes / sizeof(Element);
    int const ldst_group_inner_idx = thread_idx % ScatterLdst::kThreadsPerGroup;
    int const ldst_group_idx = thread_idx / ScatterLdst::kThreadsPerGroup;
    // Compound index decomposition: compound_idx = token_idx * G + g (G = PackGQAFactor).
    // Splits compound_idx into token_idx * token_stride + g * head_stride.
    auto compound_idx_to_offset = [](int compound_idx, int64_t token_stride, int64_t head_stride) -> int64_t {
      if constexpr (PackGQA || CatGQA) {
        return (compound_idx / PackGQAFactor) * token_stride + (compound_idx % PackGQAFactor) * head_stride;
      } else {
        return compound_idx * token_stride;
      }
    };
    // Q/dO strides
    int64_t const stride_q_token = get<0>(params.stride_Q);
    int64_t const stride_do_token = get<0>(params.stride_dO);
    int64_t const stride_q_head = get<2>(params.stride_Q);
    int64_t const stride_do_head = get<2>(params.stride_dO);
    Element const* const ptr_gQ_base = params.ptr_Q + bidh * PackGQAFactor * stride_q_head + ldst_group_inner_idx * kElemsPerLane;
    Element const* const ptr_gdO_base = params.ptr_dO + bidh * PackGQAFactor * stride_do_head + ldst_group_inner_idx * kElemsPerLane;
    // LSE/dPsum strides: 4 floats per token.
    // kv_head_stride: stride between KV heads (bidh axis).
    // q_head_in_group_stride: stride between Q heads within one GQA group (0 when !PackGQA && !CatGQA).
    auto extract_lse_strides = [&](auto const& stride) -> cute::tuple<int64_t, int64_t> {
      int64_t kv_head_stride, q_head_in_group_stride;
      if constexpr (CatGQA) {
        kv_head_stride = get<2, 1>(stride);
        q_head_in_group_stride = get<2, 0>(stride);
      } else if constexpr (PackGQA) {
        kv_head_stride = get<2>(stride);
        q_head_in_group_stride = get<1, 0>(stride);
      } else {
        kv_head_stride = get<2>(stride);
        q_head_in_group_stride = int64_t(0);
      }
      return {kv_head_stride, q_head_in_group_stride};
    };
    auto [lse_kv_head_stride, lse_q_head_in_group_stride] = extract_lse_strides(params.stride_LSE);
    auto [dpsum_kv_head_stride, dpsum_q_head_in_group_stride] = extract_lse_strides(params.stride_dPsum);
    float const* const ptr_gLSE_base = params.ptr_LSE_log2 + bidh * lse_kv_head_stride;
    float const* const ptr_gdPsum_base = params.ptr_dPsum + bidh * dpsum_kv_head_stride;
    // LSE/dPsum per-token offset: token_stride=4 (fixed), head_stride from GQA decomposition.
    auto lse_row_offset = [&](int ci) -> int64_t { return compound_idx_to_offset(ci, 4, lse_q_head_in_group_stride); };
    auto dpsum_row_offset = [&](int ci) -> int64_t { return compound_idx_to_offset(ci, 4, dpsum_q_head_in_group_stride); };

    // ─── Load lambdas: two branches — TMA 2D (dense + contiguous sparse) and CpAsync (sparse fallback) ───
    // Q and dO share the same pipe slot when Q_dO_same_stages=true, so pipe advance
    // happens in load_dO_dPsum (the second of each pair) to keep the slot index in sync.
    auto load_Q_LSE = [&]() {
      if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
        // Sparse configs run multiple loader warps (sized for the CpAsync
        // scatter path), but only the leader loader warp may drive the Q/dO
        // TMA pipeline: an extra warp would producer_acquire and then early
        // return on `thread_idx != 0` BEFORE ++smem_pipe_write_q, leaving its
        // pipeline state at a stale phase that deadlocks producer_tail
        // whenever the total tile count has the wrong phase parity.
        if constexpr (IsSparse) {
          if (thread_idx >= cutlass::NumThreadsPerWarp)
            return;
        }
        if (!lane_predicate)
          return;
        pipeline_q.producer_acquire(smem_pipe_write_q);
        int const stage = smem_pipe_write_q.index();
        if constexpr (IsSparse) {
          if (thread_idx != 0)
            return;
          int const tile_first_compound_idx = block_meta.get_tile_first_compound_idx();
          shared_storage.tensors.mainloop.smem_sparse_inner_indices[stage * kBlockM] = tile_first_compound_idx;
          // domain_offset at the exact tile start; index tile 0
          auto const qdo_off = make_coord(tile_first_compound_idx, _0{});
          Tensor gQ_ = local_tile(domain_offset(qdo_off, mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
          Tensor tQgQ_ = group_modes<0, 3>(block_tma_Q.partition_S(gQ_));
          auto tma_Q_desc = params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          copy(tma_Q_desc, tQgQ_(_, 0), tQsQ(_, stage));
          auto const lse_off = make_coord(_0{}, tile_first_compound_idx);
          Tensor gLSE_ = local_tile(domain_offset(lse_off, mLSE), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, _0{}));
          copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE_, sLSE(_, _, stage));
        } else {
          auto tma_Q_desc = params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          if constexpr (CatGQA) {
            copy(tma_Q_desc, tQgQ(_, m_block, bidh_kv_cat), tQsQ(_, stage));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE(_, _, m_block, bidh_kv_cat), sLSE(_, _, stage));
          } else {
            copy(tma_Q_desc, tQgQ(_, m_block), tQsQ(_, stage));
            copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE(_, _, m_block), sLSE(_, _, stage));
          }
        }
      } else {
        // CpAsync scatter: all producer threads participate, per-token cp.async
        pipeline_q.producer_acquire(smem_pipe_write_q);
        int const stage = smem_pipe_write_q.index();
        int* const stage_indices = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[stage * kBlockM];
        block_meta.fill_token_indices(stage_indices, ldst_group_inner_idx, ldst_group_idx);
        __syncwarp();
        scatter_inner_load<PackGQAFactor>(sQ, ptr_gQ_base, stride_q_token, stride_q_head, stage_indices, stage, thread_idx, sLSE, ptr_gLSE_base, lse_row_offset);
        pipeline_q.producer_commit(smem_pipe_write_q, cutlass::arch::cpasync_barrier_arrive);
      }
    };

    auto load_dO_dPsum = [&]() {
      PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write_q, smem_pipe_write_do);
      if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
        // See load_Q_LSE: only the leader loader warp may touch the TMA pipeline.
        if constexpr (IsSparse) {
          if (thread_idx >= cutlass::NumThreadsPerWarp)
            return;
        }
        if (!lane_predicate)
          return;
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        if constexpr (IsSparse) {
          if (thread_idx != 0)
            return;
          int const tile_first_compound_idx = shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_write_q.index() * kBlockM];
          auto const qdo_off = make_coord(tile_first_compound_idx, _0{});
          Tensor gdO_ = local_tile(domain_offset(qdo_off, mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));
          Tensor tdOgdO_ = group_modes<0, 3>(block_tma_dO.partition_S(gdO_));
          auto tma_dO_desc = params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          copy(tma_dO_desc, tdOgdO_(_, 0), tdOsdO(_, smem_pipe_write_do_cur.index()));
          auto const dpsum_off = make_coord(_0{}, tile_first_compound_idx);
          Tensor gdPsum_ = local_tile(domain_offset(dpsum_off, mdPsum), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, _0{}));
          copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum_, sdPsum(_, _, smem_pipe_write_do_cur.index()));
        } else {
          auto tma_dO_desc = params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          if constexpr (CatGQA) {
            copy(tma_dO_desc, tdOgdO(_, m_block, bidh_kv_cat), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(
                bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)),
                gdPsum(_, _, m_block, bidh_kv_cat),
                sdPsum(_, _, smem_pipe_write_do_cur.index()));
          } else {
            copy(tma_dO_desc, tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum(_, _, m_block), sdPsum(_, _, smem_pipe_write_do_cur.index()));
          }
        }
      } else {
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        int const* const stage_indices = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_write_q.index() * kBlockM];
        scatter_inner_load<PackGQAFactor>(
            sdO, ptr_gdO_base, stride_do_token, stride_do_head, stage_indices, smem_pipe_write_do_cur.index(), thread_idx, sdPsum, ptr_gdPsum_base, dpsum_row_offset);
        pipeline_do.producer_commit(smem_pipe_write_do_cur, cutlass::arch::cpasync_barrier_arrive);
      }
      if constexpr (!Q_dO_same_stages) {
        ++smem_pipe_write_do;
      }
      ++smem_pipe_write_q;
    };

    auto load_KV = [&]() {
      // barrier_KV is init'd with numThreads=1, so exactly one thread may arrive.
      // thread_idx==0 selects one thread uniformly for dense (1 loader warp) and
      // BlockSparse (2 loader warps, where a per-warp elect_one_sync would give 2 arrivals).
      if (thread_idx != 0)
        return;
      auto& barrier_KV = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_KV);
      shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
      copy(params.tma_load_K.with(barrier_KV, /*mcast_mask=*/0), tKgK, tKsK);
      copy(params.tma_load_V.with(barrier_KV, /*mcast_mask=*/0), tVgV, tVsV);
    };

    auto load_body = [&]() {
      if constexpr (IsSparse) {
        // Scatter (BlockSparse / IndexSparse InnerLoopQ): one block per call, block_meta drives iteration
        load_Q_LSE();
        load_dO_dPsum();
      } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, PackGQAFactor); ++bidh_kv_cat) {
          rebind_Q_tiles(block_meta.seqlen_info);
          m_block = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_cnt);
          flash::iterate_range < kInnerDir,
              kHeadDim<256 ? 2 : 1>(
                  m_block,
                  block_meta.inner_block_min,
                  block_meta.inner_block_cnt,
                  [&] {
                    load_Q_LSE();
                    load_dO_dPsum();
                  });
        }
      }
    };

    // ─── Unified control flow ───
    // K/V are loaded once (fixed n_block), Q/dO are streamed across merged batches.
    if (block_meta.skip_to_first_valid()) {
      // Zero-ref tile (inner_block_cnt == 0): signal barrier_KV with 0 expected bytes
      // so the consumer's barrier_KV.wait() at the top of mma_with_loop_q can unblock.
      // Without this, the consumer deadlocks waiting for K/V data that was never loaded.
      if (thread_idx == 0) {
        shared_storage.pipelines.barrier_KV.arrive_and_expect_tx(0);
      }
      return false;
    }

    load_KV();

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        load_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
      }
    } else {
      load_body();
    }

    if constexpr (Q_dO_same_stages) {
      smem_pipe_write_do = smem_pipe_write_q;
    }

    return true;
  }

  // Perform a Producer Prologue/Mainloop -- TMA Load for Q,dO,LSE,dPsum, with pipelining multi-stage TMA load for K,V
  // q for outer-loop and k for inner-loop
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE bool load_with_loop_k(
      Params const& params,
      MainloopPipeline pipeline_k,
      MainloopPipeline_V pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState_V& smem_pipe_write_v,
      SharedStorage& shared_storage,
      BlockMetaT& block_meta) {
    // Compile Guard Clause
    static_assert(BwdInnerLoopK, "load_with_loop_k() must be called when BwdInnerLoopK is true");
    static_assert(!CatGQA, "load_with_loop_k() is not compatible with CatGQA");

    // BlockMeta: fixed per function call
    int const m_block = block_meta.outer_tile_idx;
    int const bidh = block_meta.bidh;
    int const bidh_kv = block_meta.bidh_kv;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sLSE = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSE{});
    Tensor sdPsum = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSE{});

    // prepare for TMA multicast meta
    auto [mcast_mask_kv, cluster_block_id_kv] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyKV, /*RowwiseMask=*/true>();

    // Prepare the TMA loads
    auto mQ = params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
    auto mdO = params.tma_load_dO.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
    Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    // For PackGQA, LSE/dPsum use packed shape/stride to correctly read data from multiple Q heads
    auto mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSEdPsum, params.stride_LSE)(_, _, bidh); // (4, seqlen_q)
    auto mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSEdPsum, params.stride_dPsum)(_, _, bidh); // (4, seqlen_q)

    // For PackGQA, offset needs to be multiplied by PackGQAFactor
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * PackGQAFactor;
    Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)
    Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)

    // For PackGQA, LSE/dPsum also use packed offset to match Q/dO's packed access pattern
    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(_0{}, offset_q), mLSE), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, m_block)); // (4, M)
    Tensor gdPsum = local_tile(cute::domain_offset(make_coord(_0{}, offset_q), mdPsum), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, m_block)); // (4, M)

    auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)

    auto block_tma_dO = params.tma_load_dO.get_slice(_0{});
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO)); // (TMA)
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO)); // (TMA)

    int const lane_predicate = cute::elect_one_sync();
    // BlockSparse/IndexSparse run the scatter load on 2 warps (warp 0 & 1), but the Q/dO/LSE/dPsum
    // TMA must be issued by a single warp only (warp 0), otherwise barrier_QdO's expect_tx is
    // counted twice and mismatches the consumer's wait. Dense only runs warp 0, so this is a no-op there.
    int const warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

    // ─── BlockSparse / IndexSparse scatter load lambdas ───
    // Loop-invariant scatter addressing hoisted out of the lambdas (computed once; unused &
    // DCE'd on the dense path). sK/sV are already shared at function scope above.
    // Use CuTe Copy_Atom for cp.async.cg (emits L2::128B). Benchmarked against bare-PTX
    // L2::cache_hint.L2::256B + evict_last: < 0.5% difference on BlockSparse MQA workloads.
    using CpAsyncCg = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
    CpAsyncCg const cp_async_cg{};
    int const thread_idx = threadIdx.x % NumProducerLoaderThreads;
    static constexpr int kElemsPerLane = ScatterLdst::kLaneBytes / sizeof(Element);
    static constexpr int kElemsPerRow = ScatterLdst::kBankRowBytes / sizeof(Element);
    static constexpr int kTilesPerRow = kHeadDim / kElemsPerRow;
    int const ldst_group_inner_idx = thread_idx % ScatterLdst::kThreadsPerGroup;
    int const ldst_group_idx = thread_idx / ScatterLdst::kThreadsPerGroup;
    int const stride_kv_row = get<0>(params.stride_K);
    int const stride_kv_row_v = get<0>(params.stride_V);
    Element const* const ptr_gK_base = params.ptr_K + bidh_kv * get<2>(params.stride_K) + ldst_group_inner_idx * kElemsPerLane;
    Element const* const ptr_gV_base = params.ptr_V + bidh_kv * get<2>(params.stride_V) + ldst_group_inner_idx * kElemsPerLane;

    // ─── Shared Q/dO/LSE/dPsum loading ───

    auto load_QdO_LSE_dPsum = [&]() {
      // Only warp 0's elected leader issues the QdO TMA (single-warp), see note above.
      if (!(warp_idx_in_warpgroup == 0 && lane_predicate))
        return;
      auto& barrier_QdO = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_QdO);
      shared_storage.pipelines.barrier_QdO.arrive_and_expect_tx(TmaTransactionBytesQ + TmaTransactionBytesdO + TmaTransactionBytesLSE + TmaTransactionBytesdPsum);
      copy(params.tma_load_Q.with(barrier_QdO, /*mcast_mask=*/0), tQgQ, tQsQ);
      copy(params.tma_load_dO.with(barrier_QdO, /*mcast_mask=*/0), tdOgdO, tdOsdO);
      copy(bulk_copy.with(barrier_QdO), gLSE, sLSE);
      copy(bulk_copy.with(barrier_QdO), gdPsum, sdPsum);
    };

    // ─── TMA setup for K/V ───
    auto block_tma_K = params.tma_load_K.get_slice(cluster_block_id_kv);
    Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK)); // (TMA, PIPE)

    auto block_tma_V = params.tma_load_V.get_slice(cluster_block_id_kv);
    Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV)); // (TMA, PIPE)

    // ─── TMA K/V load helper: domain_offset + partition, used by both dense and sparse paths ───
    auto tma_load_K_tile = [&](int origin, int block_idx, int stage) {
      Tensor gK_ = local_tile(domain_offset(make_coord(origin, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
      Tensor tKgK_ = group_modes<0, 3>(block_tma_K.partition_S(gK_));
      copy(
          params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
          tKgK_(_, block_idx),
          tKsK(_, stage));
    };
    auto tma_load_V_tile = [&](int origin, int block_idx, int stage) {
      Tensor gV_ = local_tile(domain_offset(make_coord(origin, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
      Tensor tVgV_ = group_modes<0, 3>(block_tma_V.partition_S(gV_));
      copy(
          params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
          tVgV_(_, block_idx),
          tVsV(_, stage));
    };

    // ─── load_K / load_V: two branches — TMA (dense + sparse) and CpAsync (sparse fallback) ───
    // When kStages_V != kStages, V pipeline's stage index differs from K's.
    // V needs K's stage to read smem_sparse_inner_indices (populated by load_K).
    int last_k_write_stage = 0;

    auto load_K = [&]() {
      if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
        if (!lane_predicate)
          return;
        pipeline_k.producer_acquire(smem_pipe_write_k);
        int const stage = smem_pipe_write_k.index();
        if constexpr (IsSparse) {
          if (thread_idx == 0) {
            // Round to kBlockN boundary: MaxToMin anchor is offset within the tile
            // by (kTokensPerGroup-1); division truncates this to the tile start.
            int const tile_first_compound_idx = (block_meta.get_tile_first_compound_idx() / kBlockN) * kBlockN;
            int* const stage_indices = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[stage * kBlockN];
            CUTE_UNROLL
            for (int r = 0; r < kBlockN; ++r) {
              stage_indices[r] = tile_first_compound_idx + r;
            }
            tma_load_K_tile(tile_first_compound_idx, 0, stage);
          }
          last_k_write_stage = stage;
        } else {
          tma_load_K_tile(block_meta.seqlen_info.offset_k, block_meta.inner_block_idx, stage);
        }
        ++smem_pipe_write_k;
      } else {
        // CpAsync scatter: all producer threads participate, per-token cp.async.
        pipeline_k.producer_acquire(smem_pipe_write_k);
        int* const stage_indices = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_write_k.index() * kBlockN];
        block_meta.fill_token_indices(stage_indices, ldst_group_inner_idx, ldst_group_idx);
        __syncwarp();
        scatter_inner_load(sK, ptr_gK_base, static_cast<int64_t>(stride_kv_row), int64_t(0), stage_indices, smem_pipe_write_k.index(), thread_idx);
        last_k_write_stage = smem_pipe_write_k.index();
        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;
      }
    };

    auto load_V = [&]() {
      if constexpr (PerfDebugSkipVLoad) {
        // Debug: lightweight V load placeholder. Loads a FIXED block (block 0)
        // so the pipeline stays intact and PV MMA runs on real (but wrong) data.
        // After the first iteration the load hits L2 cache, minimizing GMEM bandwidth.
        if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
          if (!lane_predicate)
            return;
          pipeline_v.producer_acquire(smem_pipe_write_v);
          if constexpr (IsSparse) {
            if (thread_idx == 0) {
              tma_load_V_tile(0, 0, smem_pipe_write_v.index());
            }
          } else {
            tma_load_V_tile(block_meta.seqlen_info.offset_k, 0, smem_pipe_write_v.index());
          }
          ++smem_pipe_write_v;
        } else {
          // CpAsync debug: sequential rows from block 0 (no stage_indices).
          pipeline_v.producer_acquire(smem_pipe_write_v);
          CUTE_UNROLL
          for (int local_row = 0; local_row < ScatterLdst::kTokensPerGroup; ++local_row) {
            int smem_row = ldst_group_idx * ScatterLdst::kTokensPerGroup + local_row;
            int token_idx = smem_row * stride_kv_row_v;
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < kTilesPerRow; ++tile_idx) {
              if (ldst_group_inner_idx * kElemsPerLane + tile_idx * kElemsPerRow < kHeadDim) {
                Element* dst_ptr = &sV(smem_row, ldst_group_inner_idx * kElemsPerLane + tile_idx * kElemsPerRow, smem_pipe_write_v.index());
                auto gV_src = make_tensor(make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_gV_base + token_idx + tile_idx * kElemsPerRow)), Layout<_1>{});
                auto sV_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(dst_ptr)), Layout<_1>{});
                cute::copy(cp_async_cg, gV_src, sV_dst);
              }
            }
          }
          pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
          ++smem_pipe_write_v;
        }
      } else if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
        if (!lane_predicate)
          return;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        if constexpr (IsSparse) {
          if (thread_idx == 0) {
            int const tile_first_compound_idx = shared_storage.tensors.mainloop.smem_sparse_inner_indices[last_k_write_stage * kBlockN];
            tma_load_V_tile(tile_first_compound_idx, 0, smem_pipe_write_v.index());
          }
        } else {
          tma_load_V_tile(block_meta.seqlen_info.offset_k, block_meta.inner_block_idx, smem_pipe_write_v.index());
        }
        ++smem_pipe_write_v;
      } else {
        // CpAsync scatter: reuse K's token indices from last_k_write_stage.
        pipeline_v.producer_acquire(smem_pipe_write_v);
        int const* const stage_indices = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[last_k_write_stage * kBlockN];
        scatter_inner_load(sV, ptr_gV_base, static_cast<int64_t>(stride_kv_row_v), int64_t(0), stage_indices, smem_pipe_write_v.index(), thread_idx);
        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      }
    };

    auto load_body = [&]() {
      if constexpr (IsSparse) {
        load_K();
        load_V();
      } else {
        flash::iterate_range < kInnerDir,
            kHeadDim<256 ? 2 : 1>(
                block_meta.inner_block_idx,
                block_meta.inner_block_min,
                block_meta.inner_block_cnt,
                [&] {
                  load_K();
                  load_V();
                });
      }
    };

    // ─── Unified control flow ───
    // Q/dO/LSE/dPsum are loaded once (fixed m_block), K/V are streamed across merged batches.
    if (block_meta.skip_to_first_valid())
      return false;

    block_meta.template update_block_cur<kInnerDir>();
    load_QdO_LSE_dPsum();

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        load_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
        block_meta.template update_block_cur<kInnerDir>();
      }
    } else {
      load_body();
    }

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
    static_assert(!BwdInnerLoopK, "load_tail_with_loop_q() must be called when BwdInnerLoopK is false");

    // PipelineAsync (kCpAsync): all threads must arrive.
    // PipelineTmaAsync (kTmaDense/kTma2D): single-thread arrive suffices.
    // In TMA mode only the leader loader warp drives the pipeline (see
    // load_Q_LSE); extra sparse loader warps hold virgin pipeline states
    // whose phase parity would deadlock producer_tail, so they must not
    // participate here. elect_one_sync() alone is insufficient: it elects
    // one lane PER WARP.
    bool const is_leader_loader_warp = (threadIdx.x % NumProducerLoaderThreads) < cutlass::NumThreadsPerWarp;
    if (kInnerLoadMode != InnerLoadMode::Tma || (is_leader_loader_warp && cute::elect_one_sync())) {
      pipeline_q.producer_tail(smem_pipe_write_q);
      pipeline_do.producer_tail(smem_pipe_write_do);
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  // q for outer-loop and k for inner-loop
  CUTLASS_DEVICE void load_tail_with_loop_k(
      MainloopPipeline pipeline_k,
      MainloopPipeline_V pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState_V& smem_pipe_write_v) {
    static_assert(BwdInnerLoopK, "load_tail_with_loop_k() must be called when BwdInnerLoopK is true");

    // PipelineAsync (kCpAsync): all threads must arrive.
    // PipelineTmaAsync (kTma): single-thread arrive suffices.
    if (kInnerLoadMode != InnerLoadMode::Tma || cute::elect_one_sync()) {
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  // Store partial dQ from SMEM to GMEM with TMA Atomic Reduce Add
  // k for outer-loop and q for inner-loop
  // Scatter path: token indices come from the fixed staging area, copied there by
  // consumer WG0 under the dQEmpty/dQFull handshake (no pipeline state needed here).
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE void store_dq(Params const& params, SharedStorage& shared_storage, BlockMetaT& block_meta) {
    static_assert(!BwdInnerLoopK, "store_dq() must be called when BwdInnerLoopK is false");

    // !InnerStoreInProducer: dQ store is handled by the MMA consumer threads, not by producer.
    if constexpr (!InnerStoreInProducer) {
      return;
    }

    if constexpr (!dQ_use_smem) {
      return;
    }

    static constexpr int kBlockM = CollectiveMainloopBwdSm90::kBlockM;
    static constexpr int kBlockN = CollectiveMainloopBwdSm90::kBlockN;

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_tile_idx;
    int const bidh = block_meta.bidh;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;
    // BlockMeta: reassigned per RangeMerge batch in while(true)
    flash::AttnType attn_type;
    int m_block_min;
    int m_block_max;
    int offset_q;
    int last_n_block;
    // PackGQA: Q heads packed into seqlen → m_block_num includes PackGQAFactor factor.
    // CatGQA: Q heads stay in head dim → m_block_num is based on raw seqlen_q.
    int m_block_num = cute::ceil_div(seqlen_info.seqlen_q * cute::conditional_return<PackGQA>(PackGQAFactor, 1), kBlockM);
    int bidb_last = 0;

    bool const lane_predicate = cute::elect_one_sync();
    int const num_heads = [&]() {
      if constexpr (CatGQA) {
        return get<2, 1>(params.shape_QdOdQ);
      } else {
        return get<2>(params.shape_QdOdQ);
      }
    }();

    // batch i use [i * n_block_max_num + 1 , i * n_block_max_num + n_block_size - 1] for add rank of same qhead
    // except for the last n_block_id, the last is always (i + 1) * n_block_max_num
    // PackGQA: offset_q is already scaled by PackGQAFactor (set in the main loop below),
    // so we use offset_q here to keep conflict indices consistent with the packed m_block range.
    auto m_block_sync = [&](int m_block_id) {
      uint32_t smid = blockIdx.x;
      uint32_t sm_stride = gridDim.x;
      // calc dq conflict range lock index
      int left_dq_conflict_index = offset_q / kBlockM + m_block_id;
      int right_dq_conflict_index = (offset_q + kBlockM - 1) / kBlockM + m_block_id;
      // the first n_block should wait for conflict batches
      // the others n_block should wait for previous n_block
      int sync_num1 = n_block == 0 ? params.dq_determin_conflict_state[left_dq_conflict_index * sm_stride + smid] * params.n_block_max_num
                                   : bidb * params.n_block_max_num + n_block;
      int sync_num2 = n_block == 0 ? params.dq_determin_conflict_state[right_dq_conflict_index * sm_stride + smid] * params.n_block_max_num
                                   : bidb * params.n_block_max_num + n_block;
      deterministic_sync(params.dq_determin_range_locks, bidh, offset_q + m_block_id * kBlockM, kBlockM, num_heads, sync_num1, sync_num2);
    };

    auto m_block_arrive = [&](int m_block_id) {
      // calc arrive message: l_arrive_twice & r_arrive_twice
      // each range_lock needs to arrive twice to make sure conflict batch has been completed
      // because range_lock block and batch's block may start from a different offset
      bool l_arrive_twice = (m_block_id == 0) && (offset_q % kBlockM != 0);
      bool r_arrive_twice = (m_block_id == m_block_num - 1) && (offset_q % kBlockM != 0);
      // the last n_block arrive num is always (batch id + 1) * n_block_max_num
      int arrive_num = n_block == last_n_block ? (bidb + 1) * params.n_block_max_num : bidb * params.n_block_max_num + n_block + 1;
      deterministic_arrive(params.dq_determin_range_locks, bidh, offset_q + m_block_id * kBlockM, kBlockM, num_heads, arrive_num, l_arrive_twice, r_arrive_twice);
    };

    auto const mQdOdQLSEdPsum_coord = make_coord(_, _, cute::conditional_return<CatGQA>(make_coord(_, bidh), bidh));
    auto const gQdOdQ_coord = cute::conditional_return<CatGQA>(make_coord(_, _0{}, _), make_coord(_, _0{}));
    // Dense TMA store view (swizzled); scatter store reads through the Store layout view
    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQSwizzled{});
    Tensor sdQ_store = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQStore{});
    Tensor mdQ_reduce = params.tma_add_dQ.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);

    // Scatter store addressing for producer store warp (32 threads).
    // PackGQA: smem_sparse_inner_indices hold compound indices (token_idx*G + g); bidh is the
    // kv head, so the head base and per-token decomposition are scaled by G (see scatter_inner_store).
    static constexpr int kdQHeadPackFactor = PackGQA ? PackGQAFactor : 1;
    [[maybe_unused]] int const store_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;
    [[maybe_unused]] int const stride_dq_token = get<0>(params.stride_dQ);
    [[maybe_unused]] int const stride_dq_head = get<2>(params.stride_dQ);
    [[maybe_unused]] ElementAccum* const ptr_dQ_base = params.ptr_dQ + bidh * kdQHeadPackFactor * static_cast<int64_t>(get<2>(params.stride_dQ));

    auto store_inner_dq = [&](int const m_block, int const bidh_kv_cat, int const off_q) {
#pragma unroll
      // Sync at sdQ full barrier, to wait for all consumer WGs to finish dQ r2s-copy
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }

      if constexpr (kInnerLoadMode == InnerLoadMode::Tma && IsSparse) {
        // 2D TMA reduce: entire tile written in one TMA reduce-add instruction.
        // Use domain_offset to shift to the exact sparse origin (mirrors load at L1512-1513).
        if (lane_predicate) {
          int const tile_first_compound_idx = shared_storage.tensors.mainloop.smem_sparse_store_staging_indices[0];
          auto const dq_sparse_off = make_coord(tile_first_compound_idx, _0{});
          tma_inner_store(params.tma_add_dQ, sdQ, domain_offset(dq_sparse_off, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord, 0);
        }
      } else if constexpr (IsSparse) {
        // Per-row 1D bulk reduce fallback for non-MQA scatter
        scatter_inner_store<kBlockM, cutlass::NumThreadsPerWarp, kdQHeadPackFactor>(
            sdQ_store,
            shared_storage.tensors.mainloop.smem_sparse_store_staging_indices.data(),
            ptr_dQ_base,
            stride_dq_token,
            store_thread_idx,
            /*tile_offset=*/0,
            stride_dq_head);
      } else {
        // Dense TMA reduce
        if (lane_predicate) {
          if constexpr (Deterministic) {
            if (!CatGQA || bidh_kv_cat == 0) {
              m_block_sync(m_block);
            }
          }
          auto const gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(off_q, _0{}, _0{}), make_coord(off_q, _0{}));
          if constexpr (CatGQA) {
            tma_inner_store(params.tma_add_dQ, sdQ, domain_offset(gQdO_offset_q_coord, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord, m_block, bidh_kv_cat);
          } else {
            tma_inner_store(params.tma_add_dQ, sdQ, domain_offset(gQdO_offset_q_coord, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord, m_block);
          }

          if constexpr (Deterministic) {
            if constexpr (CatGQA) {
              if (bidh_kv_cat == PackGQAFactor - 1) {
                m_block_arrive(m_block);
              }
            } else {
              m_block_arrive(m_block);
            }
          }
        }
      }

      // Arrive at sdQ empty barrier
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
      }
    };

    // Deterministic: forward sync+arrive signals for m_blocks that have no actual dQ data,
    // ensuring downstream consumers don't deadlock waiting for signals from skipped blocks.
    auto deterministic_pass_through = [&](int from, int to) {
      if constexpr (Deterministic) {
        if (lane_predicate) {
          for (int m_block = from; m_block < to; ++m_block) {
            m_block_sync(m_block);
            m_block_arrive(m_block);
          }
        }
      }
    };

    // Deterministic: update conflict state for batches between bidb_last and bidb.
    // Each SM tracks which batch last wrote to each m_block-aligned dQ region, so that
    // m_block_sync for n_block==0 knows which batch's arrive signal to wait for.
    auto update_conflict_state = [&](int bidb_last, int bidb_cur) {
      if constexpr (Deterministic) {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        uint32_t smid = blockIdx.x;
        uint32_t sm_stride = gridDim.x;
        int* conflict_state = params.dq_determin_conflict_state;
        // update missed batch's conflict state, loop for bidb_last ~ bidb
        while (bidb_last < bidb_cur) {
          // bidb_last_l ~ bidb_last_r is the range of bidb_last
          // PackGQA: q_ranges stores original offsets, but dQ conflict_state is indexed
          // by packed offsets (seqlen_q * PackGQAFactor), so we must scale accordingly.
          int bidb_last_l = params.q_ranges[bidb_last].x, bidb_last_r = params.q_ranges[bidb_last].y;
          if constexpr (PackGQA) {
            bidb_last_l *= PackGQAFactor;
            bidb_last_r *= PackGQAFactor;
          }
          int l = bidb_last_l / kBlockM + lane; // bidb_last_l / kBlock is first block id
          int block_num = cute::ceil_div(bidb_last_r - bidb_last_l, kBlockM); // calc total block num of bidb_last
          int r = (bidb_last_l + block_num * kBlockM - 1) / kBlockM; // calc last block id
          // each threads of warp update conflict block id left ~ right
          // each batch's range will conflict with previous batch, which cover the same block id
          while (l <= r) {
            // conflict state[block id * sm_stride + smid] save the conflict info of this sm
            conflict_state[l * sm_stride + smid] = bidb_last + 1;
            l += cutlass::NumThreadsPerWarp;
          }
          bidb_last++;
        }
        __syncwarp();
      }
    };

    auto store_body = [&]() {
      if constexpr (IsSparse) {
        // Scatter path uses the sparse consumer BlockMeta (one inner m_block per store_body
        // call; prefetch() advances a single block — mirrors store_dkv). The handshake count
        // thus matches the MMA consumer's ceil(gathered_tokens / kBlockM) tile count exactly.
        // m_block / bidh_kv_cat / off_q are unused by the scatter branch.
        store_inner_dq(block_meta.inner_block_idx, 0, 0);
      } else {
        m_block_min = block_meta.inner_block_min;
        m_block_max = block_meta.inner_block_cnt;
        seqlen_info = block_meta.seqlen_info;
        bidb = block_meta.bidb;
        attn_type = block_meta.attn_type;
        offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * PackGQAFactor;
        last_n_block = cute::ceil_div(seqlen_info.seqlen_k, kBlockN) - 1;
        m_block_num = cute::ceil_div(seqlen_info.seqlen_q * cute::conditional_return<PackGQA>(PackGQAFactor, 1), kBlockM);

        update_conflict_state(bidb_last, bidb);
        bidb_last = bidb;

        deterministic_pass_through(0, m_block_min);

        for (int bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, PackGQAFactor); ++bidh_kv_cat) {
          int m_block = flash::init_block_cur<kInnerDir>(m_block_min, m_block_max);
          flash::iterate_range<kInnerDir, 2>(m_block, m_block_min, m_block_max, [&] { store_inner_dq(m_block, bidh_kv_cat, offset_q); });
        }

        deterministic_pass_through(m_block_max, m_block_num);
      }
    };

    // ─── Unified control flow ───
    if (block_meta.skip_to_first_valid()) {
      // Tile entirely invalid: deterministic path still needs to arrive all range locks.
      deterministic_pass_through(0, m_block_num);
      return;
    }

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        store_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
      }
    } else {
      store_body();
    }
  }

  // Store partial dK,dV from SMEM to GMEM with TMA Atomic Reduce Add
  // q for outer-loop and k for inner-loop
  // Scatter path: token indices come from the per-direction staging areas (dV then dK),
  // copied there by consumer WG0 under the respective Empty/Full handshakes.
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE void store_dkv(Params const& params, SharedStorage& shared_storage, BlockMetaT& block_meta) {
    static_assert(BwdInnerLoopK, "store_dkv() must be called when BwdInnerLoopK is true");
    static_assert(!Deterministic, "Deterministic mode is not supported yet");

    if constexpr (kInnerStoreMode == InnerStoreMode::BypassSmem) {
      return;
    }

    // ─── Definitions hoisted to function top: shared by the Dense TMA-store path and the
    //     BlockSparse/IndexSparse scatter-store path. All are pure (layout / scalar) computations,
    //     so whatever is unused on a given path is DCE'd away (no runtime cost / no descriptor deref). ───
    // BlockMeta: fixed per function call
    int const bidh_kv = block_meta.bidh_kv;

    bool const lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

    // smem dK/dV SMEM buffers: scatter store reads the Store layout view, dense TMA the swizzled one.
    // For double-buffer (kInnerStoreStages >= 2), these are reconstructed per-iteration with the
    // current store_stage. Here we initialize to stage 0 for the initial setup.
    int store_stage = 0;
    auto make_sdK = [&](int stg) { return make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop, stg)), SmemLayoutdKVStore{}); };
    auto make_sdV = [&](int stg) { return make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop, stg)), SmemLayoutdKVStore{}); };
    auto make_sdK_tma = [&](int stg) { return make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop, stg)), SmemLayoutdKVSwizzled{}); };
    auto make_sdV_tma = [&](int stg) { return make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop, stg)), SmemLayoutdKVSwizzled{}); };
    Tensor sdK = make_sdK(0);
    Tensor sdV = make_sdV(0);
    Tensor sdK_tma = make_sdK_tma(0);
    Tensor sdV_tma = make_sdV_tma(0);

    // Dense TMA reduce-add setup (uses shape_dKdV which includes pool dimension)
    Tensor mdK_reduce = params.tma_add_dK.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);
    Tensor mdV_reduce = params.tma_add_dV.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);

    // BlockSparse / IndexSparse scatter-store addressing
    int const thread_idx = threadIdx.x % NumProducerLoaderThreads;
    int const stride_dV_token = get<0>(params.stride_dV);
    int const stride_dK_token = get<0>(params.stride_dK);
    ElementAccum* const ptr_gdV_base = params.ptr_dV + bidh_kv * get<2>(params.stride_dV);
    ElementAccum* const ptr_gdK_base = params.ptr_dK + bidh_kv * get<2>(params.stride_dK);
    int const* const idx_staging = [&]() -> int const* {
      if constexpr (IsSparse) {
        // Dedicated staging array: [staging_dv][staging_dk]
        return shared_storage.tensors.mainloop.smem_sparse_store_staging_indices.data();
      } else {
        return nullptr;
      }
    }();

    // ─── Unified store_dV / store_dK: scatter vs TMA reduce-add ───
    // Dense: only warp 1 in dV store, warp 2 in dK store (barrier width = 1 warp).
    // IsSparse: all scatter-store threads participate in both; the token indices come
    // from the smem slots written by the loader (single source of truth, no re-stepping here).
    auto store_dV = [&]() {
      if constexpr (!IsSparse) {
        if (warp_idx_in_warpgroup != 1)
          return;
      }
      // Wait for consumer to signal dV R2S complete (or empty handshake for perf-debug skip).
      // Must always sync here to prevent warp 1 from racing ahead of warp 2 in iterate_range.
#pragma unroll
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }
      if constexpr (!PerfDebugSkipDvStore) {
        if constexpr (kInnerLoadMode == InnerLoadMode::Tma && IsSparse) {
          if (lane_predicate && warp_idx_in_warpgroup == ProducerConsts::kInnerLoaderWarps) {
            tma_inner_store(params.tma_add_dV, sdV_tma, mdV_reduce, TileShape_InnerDkv{}, make_coord(_, _0{}), idx_staging[0] / kBlockN);
          }
        } else if constexpr (IsSparse) {
          scatter_inner_store<kBlockN, NumProducerLoaderThreads, /*kInnerStoreHeadPackFactor=*/1>(
              sdV, &idx_staging[0 * kBlockN], ptr_gdV_base, stride_dV_token, thread_idx);
        } else {
          if (lane_predicate) {
            tma_inner_store(
                params.tma_add_dV,
                sdV_tma,
                domain_offset(make_coord(block_meta.seqlen_info.offset_k, _0{}), mdV_reduce),
                TileShape_InnerDkv{},
                make_coord(_, _0{}),
                block_meta.inner_block_idx);
          }
        }
      } // !PerfDebugSkipDvStore
      // Union: signal dKEmpty (TMA dV done → consumer can r2s dK into shared buffer)
      // Un-union: signal dVEmpty (TMA dV done → consumer can r2s next dV into its own buffer)
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        if constexpr (!UnionDkvSmem) {
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
        } else {
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
        }
      }
    };

    auto store_dK = [&]() {
      if constexpr (!IsSparse) {
        if (warp_idx_in_warpgroup != 2)
          return;
      }
#pragma unroll
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }
      if constexpr (!PerfDebugSkipDkStore) {
        if constexpr (kInnerLoadMode == InnerLoadMode::Tma && IsSparse) {
          if (lane_predicate && warp_idx_in_warpgroup == ProducerConsts::kInnerLoaderWarps) {
            tma_inner_store(params.tma_add_dK, sdK_tma, mdK_reduce, TileShape_InnerDkv{}, make_coord(_, _0{}), idx_staging[kBlockN] / kBlockN);
          }
        } else if constexpr (IsSparse) {
          scatter_inner_store<kBlockN, NumProducerLoaderThreads, /*kInnerStoreHeadPackFactor=*/1>(
              sdK, &idx_staging[1 * kBlockN], ptr_gdK_base, stride_dK_token, thread_idx);
        } else {
          if (lane_predicate) {
            tma_inner_store(
                params.tma_add_dK,
                sdK_tma,
                domain_offset(make_coord(block_meta.seqlen_info.offset_k, _0{}), mdK_reduce),
                TileShape_InnerDkv{},
                make_coord(_, _0{}),
                block_meta.inner_block_idx);
          }
        }
      } // !PerfDebugSkipDkStore
      // Union: signal dVEmpty (TMA dK done → consumer can r2s next dV into shared buffer)
      // Un-union: signal dKEmpty (TMA dK done → consumer can r2s next dK into its own buffer)
      for (int warpgroup_idx = 0; warpgroup_idx < NumConsumerWarpGroups; ++warpgroup_idx) {
        if constexpr (!UnionDkvSmem) {
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
        } else {
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
        }
      }
    };

    // Double-buffer stage switch: rebind SMEM tensor views to the current store_stage buffer
    // so that store_dV/store_dK read from the correct R2S target written by the consumer.
    auto update_store_bufs = [&]() {
      if constexpr (kInnerStoreStages >= 2) {
        sdK = make_sdK(store_stage);
        sdV = make_sdV(store_stage);
        sdK_tma = make_sdK_tma(store_stage);
        sdV_tma = make_sdV_tma(store_stage);
      }
    };
    auto advance_store_stage = [&]() {
      if constexpr (kInnerStoreStages >= 2) {
        store_stage = (store_stage + 1) % kInnerStoreStages;
      }
    };

    // One inner tile's store: rebind buffers to the current stage, drain dV+dK, advance.
    // The stage must advance per inner tile (not per store_body) to stay in lockstep with
    // the consumer's consumer_store_stage, which advances after each tile's dK R2S. The
    // dense path iterates multiple inner tiles per store_body via iterate_range.
    auto store_tile = [&]() {
      update_store_bufs();
      // NOTE(058 P2a-2): an overlapped dV/dK variant (defer the dV bulk wait until after the
      // dK issue via staged tma_store_wait<1>/<0>) was implemented and benched: zero gain on
      // sparseload-loopk / indexattn-loopk (159/161 TF unchanged) — the store warps' wait is
      // not on the critical path once bulk reduce is enabled. Reverted to keep the simple
      // sequential form; see .tmp/058-fwd-tokenidx/NOTES.md.
      store_dV();
      store_dK();
      advance_store_stage();
    };

    auto store_body = [&]() {
      if constexpr (IsSparse) {
        store_tile();
      } else {
        flash::iterate_range<kInnerDir, 2>(block_meta.inner_block_idx, block_meta.inner_block_min, block_meta.inner_block_cnt, [&] { store_tile(); });
      }
    };

    // ─── Unified control flow ───
    if (block_meta.skip_to_first_valid())
      return;

    block_meta.template update_block_cur<kInnerDir>();

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        store_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
        block_meta.template update_block_cur<kInnerDir>();
      }
    } else {
      store_body();
    }
  }

  // Initialize MMA consumers
  CUTLASS_DEVICE void mma_init() {
    if constexpr (BwdInnerLoopK) { // q for outer-loop and k for inner-loop
      // Tell producer that smem_q and smem_do are ready
      BarrierManager::arrive<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::QdOEmpty);

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (InnerStoreInProducer && (kInnerStoreMode != InnerStoreMode::BypassSmem)) {
        // Initial arrive: smem_dkv_smem is initially empty.
        // Union: only dVEmpty (dK r2s waits for first TMA dV via dKEmpty from store_dV).
        // Un-union: both dVEmpty and dKEmpty (separate buffers → both r2s start immediately).
        if (warp_idx_in_warpgroup == 0 || (IsSparse && warp_idx_in_warpgroup == 1)) {
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);
          if constexpr (!UnionDkvSmem) {
            BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warp_group_idx);
          }
        }
      }
    } else { // k for outer-loop and q for inner-loop
      // We're not currently using this bc we're not using persistent scheduler
      // Tell producer (warp 0) that smem_k and smem_v are ready
      BarrierManager::arrive<NumConsumerThreads + NumProducerLoaderThreads>(BwdNamedBarriers::KVEmpty);

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (dQ_use_smem) {
        if constexpr (!InnerStoreInProducer) {
          // Consumer handles dQ store: all threads in WG arrive (no separate store warp)
          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
        } else {
          if (warp_idx_in_warpgroup == 0) {
            BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
          }
        }
      }
    }
  }

  // Perform a Consumer Prologue/Mainloop -- WGMMA for S,dP,dQ,dK,dV with softmax for P,dS
  // k for outer-loop and q for inner-loop
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename FrgTensordKV, typename BlockMetaT>
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
      BlockMetaT& block_meta,
      SharedStorage& shared_storage) {
    static_assert(!BwdInnerLoopK, "mma_with_loop_q() must be called when BwdInnerLoopK is false");
    static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_tile_idx;
    int const bidh = block_meta.bidh;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * PackGQAFactor;
    // BlockMeta: per-batch values accessed directly via block_meta.inner_block_min/max,
    // block_meta.seqlen_info.seqlen_q/k, block_meta.attn_type.

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sQt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdOt{});
    Tensor sKt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutKt{});

    // P uses 1-stage layout (produced+consumed within same iter); dS uses kStages_dS for double buffering
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP1{});
    Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
    Tensor sPt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP1t{});
    Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
    Tensor sdS = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdS{});
    Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
    Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()), SmemLayoutPdSt{});
    Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);

    // r2s write targets use the Store layout (aliases the swizzled TMA layout unless the
    // bulk-reduce scatter path swaps in the row-contiguous layout)
    Tensor sdQ = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQStore{}));
    Tensor sdQt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQtStore{}));

    Tensor sdPsumMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma = sLSEMma_full(_0{}, _, _, _); // slice dummy dim 0 with size of 4
    Tensor sdPsumMma = sdPsumMma_full(_0{}, _, _, _); // slice dummy dim 0 with size of 4

    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<NumConsumerWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

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

    auto r2s_tiled_copy_inner_dq = make_tiled_copy_C(Copy_Atom<DefaultCopy, ElementAccum>{}, tiled_mma_dQ);
    auto r2s_thr_copy_inner_dq = r2s_tiled_copy_inner_dq.get_thread_slice(thread_idx);
    Tensor tdQsdQ = r2s_thr_copy_inner_dq.partition_D(cute::conditional_return<!dQ_swapAB>(sdQ, sdQt));

    /* DEBUG */
    // Tensor cdQsdQ = make_identity_tensor(SmemLayoutdQSwizzled{}.shape());
    // Tensor tcdQsinner dQ = r2s_thr_copy_inner_dq.partition_D(cdQsdQ);
    // if (thread_idx == 0) { print(sdQ); printf("\n"); print(tdQsdQ); printf("\n"); }

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
      BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
    };

    // For the case where we do atomicAdd directly to gdQ_reduce instead of using TMA
    auto const mQdOdQLSEdPsum_coord = make_coord(_, _, cute::conditional_return<CatGQA>(make_coord(_, bidh), bidh));
    auto const gQdOdQ_coord = cute::conditional_return<CatGQA>(make_coord(_, _0{}, _), make_coord(_, _0{}));
    Tensor mdQ_reduce = params.tma_add_dQ.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
    auto const gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
    Tensor gdQ_reduce_ = local_tile(domain_offset(gQdO_offset_q_coord, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord); // (M, K, _)
    Tensor gdQ_reduce = cute::flat_divide(gdQ_reduce_, make_shape(Int<kBlockM / NumConsumerWarpGroups>{}, Int<kHeadDim>{})); // (M / WG, K, WG, 1, _)
    // We can reuse r2s_thr_copy_inner_dq for this partitioning
    Tensor tdQgdQ_reduce = r2s_thr_copy_inner_dq.partition_D(gdQ_reduce);

    auto rebind_dQ_reduce_tiles = [&]() {
      if constexpr (!RangeMerge) {
        return;
      }
      int const new_offset_q = !PackGQA ? block_meta.seqlen_info.offset_q : block_meta.seqlen_info.offset_q * PackGQAFactor;
      if constexpr (!dQ_use_smem) {
        auto const new_gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(new_offset_q, _0{}, _0{}), make_coord(new_offset_q, _0{}));
        gdQ_reduce_ = local_tile(domain_offset(new_gQdO_offset_q_coord, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord);
        gdQ_reduce = cute::flat_divide(gdQ_reduce_, make_shape(Int<kBlockM / NumConsumerWarpGroups>{}, Int<kHeadDim>{}));
        tdQgdQ_reduce = r2s_thr_copy_inner_dq.partition_D(gdQ_reduce);
      }
    };

    /* DEBUG */
    // if (thread_idx == 0 && bidh == 0 && n_block == 0){
    //     printf("bidb: %d, offset_q: %d\n", bidb, seqlen_info.offset_q);
    //     printf("gdQ_reduce_: "); print(gdQ_reduce_); printf("\n");
    //     printf("gdQ_reduce: "); print(gdQ_reduce); printf("\n");
    // }
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask;

    // Wait until this n block of K,V loaded
    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.pipelines.barrier_KV.wait(work_idx % 2);
    }

    Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));

    // Define backward step lambda func
    auto bwd_step = [&](int m_block, auto mask_fn, auto /*is_no_mask*/ = cute::false_type{}) {
      bool const is_last_m_block_this_batch = [&]() {
        if constexpr (BlockSparse) {
          return m_block == block_meta.padding_block() && block_meta.num_invalid_token > 0;
        } else {
          return (m_block == block_meta.inner_block_cnt - 1);
        }
      }();

      // MMA1 (SS): apply S = QK^T or S^T = KQ^T if SdP_swapAB
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
      mask_fn(m_block);

      // Apply scaled softmax on `scores` in-place, storing P^T (or P if SdP_swapAB)
      // NOTE: since we cannot pad for each batch, we need to mask out the OOB LSE values
      // that might be read from other batch at each batch's last m block
      if (is_last_m_block_this_batch) {
        auto thread_mma = TiledMmaSdP{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMmaSdP{}.get_thread_slice(_0{});

        static constexpr int Row = !SdP_swapAB ? 0 : 1;
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SdP_swapAB ? kBlockM : kBlockN>, Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(t0ScS.layout()));
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const seqlenq_row_limit = [&]() {
          if constexpr (BlockSparse) {
            if constexpr (BlockMetaT::kPaddingAtLowEnd) {
              return block_meta.num_invalid_token - thread_row_offset;
            } else {
              return kBlockM - block_meta.num_invalid_token - thread_row_offset;
            }
          } else {
            int const seqlen_q_packed_local = !PackGQA ? block_meta.seqlen_info.seqlen_q : block_meta.seqlen_info.seqlen_q * PackGQAFactor;
            return seqlen_q_packed_local - m_block * kBlockM - thread_row_offset;
          }
        }();

#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          int const row_pos = int(get<Row>(t0ScS_rowcol(mi, _0{})));
          bool is_oob;
          if constexpr (BlockSparse) {
            if constexpr (BlockMetaT::kPaddingAtLowEnd) {
              is_oob = row_pos < seqlenq_row_limit;
            } else {
              is_oob = row_pos >= seqlenq_row_limit;
            }
          } else {
            is_oob = row_pos >= seqlenq_row_limit;
          }
          float lse_scaled = get_lse_scaled(mi);
          lse_scaled = is_oob ? cutlass::platform::numeric_limits<float>::infinity() : lse_scaled;
#pragma unroll
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
          }
        }
      } else {
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
      // BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
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
        // P uses 1-stage buffer: always sync to ensure prev iter's MMA3 consumed P
        BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
        Tensor tPaP = smem_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, _0{}));
      }

      // Downcast `tdPrdP` from ElementAccum to Element `rdS`
      // storing the low-precision of dS (or dS^T if SdP_swapAB)
      // and copy to shared memory in `tdSsdS` for dQ gemm (as well as dK gemm if not Mma_dKV_is_RS)
      // which is the view of `sdS` / `sdS_pi` (or `sdSt` / `sdSt_pi` if SdP_swapAB)
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_out(tdPrdP, rdS);
      if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
        // SS mode: fence+barrier to make P writes visible before MMA3 reads P from SMEM.
        // RS mode + single-stage dS: protect dS from prev-iter MMA4/5 overlap.
        // RS mode + multi-stage dS: both P (in regs) and dS (double-buffered) need no sync.
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
          Tensor tdVrP_cur = tdVrP(_, _, _, _0{}); // P is 1-stage
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
        if constexpr (dQ_use_smem) {
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          // Sync at sdQ empty barrier (producer store mode only): wait until the store warp
          // finished the previous tile's store. The consumer store mode instead relies on its
          // trailing cross-WG sync below to guarantee sdQ is free.
          if constexpr (InnerStoreInProducer) {
            BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
            if constexpr (IsSparse) {
              // Copy this tile's token indices from the stage slot (still held — Q is
              // released only after MMA5) into the staging area for the store warp.
              // Protected by the dQEmpty/dQFull handshake; WG0 alone suffices since the
              // store warp syncs every WG's dQFull before reading.
              if (warp_group_idx == 0) {
                int const* const src = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_q.index() * kBlockM];
                int* const dst = shared_storage.tensors.mainloop.smem_sparse_store_staging_indices.data();
                for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockM; r += cutlass::NumThreadsPerWarpGroup) {
                  dst[r] = src[r];
                }
              }
            }
          }

          // Copy dQ from registers to shared memory with softmax_scale applied
          Tensor taccdQrdQ = r2s_thr_copy_inner_dq.retile_S(tdQrdQ);
          for (int dqi = 0; dqi < size(taccdQrdQ); ++dqi) {
            taccdQrdQ(dqi) *= params.softmax_scale;
          }
          cute::copy(r2s_tiled_copy_inner_dq, taccdQrdQ, tdQsdQ);
          cutlass::arch::fence_view_async_shared();

          if constexpr (!InnerStoreInProducer) {
            // Consumer store path: the consumer WGs reduce-add dQ from SMEM to global dQ.
            // The dQ MMA may split the head dim across WGs (e.g. dQ_swapAB with
            // AtomLayoutNdQ=2: each WG computes ALL kBlockM token rows but only half the
            // columns), so a token row in sdQ mixes both WGs' r2s writes. The sync must
            // therefore be cross-WG (mirrors the InnerLoopK consumer dKV store); per-WG
            // dQFull/dQEmpty barriers are NOT sufficient.
            BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);

            if constexpr (kInnerLoadMode == InnerLoadMode::Tma && IsSparse) {
              // Sparse TMA reduce: use domain_offset at exact sparse origin (mirrors load at L1512).
              if (thread_idx == 0) {
                Tensor sdQ_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQSwizzled{});
                int const compound_idx = shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_q.index() * kBlockM];
                auto const dq_sparse_off = make_coord(compound_idx, _0{});
                tma_inner_store(params.tma_add_dQ, sdQ_tma, domain_offset(dq_sparse_off, mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord, 0);
              }
            } else if constexpr (IsSparse) {
              // BlockSparse scatter reduce: per-row 1D bulk reduce
              Tensor sdQ_inner = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQStore{});
              static constexpr int kdQHeadPackFactor = PackGQA ? PackGQAFactor : 1;
              int const stride_dq_token = get<0>(params.stride_dQ);
              int const stride_dq_head = get<2>(params.stride_dQ);
              ElementAccum* const ptr_dQ_base = params.ptr_dQ + block_meta.bidh * kdQHeadPackFactor * static_cast<int64_t>(get<2>(params.stride_dQ));
              int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
              int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
              scatter_inner_store<kBlockM, NumConsumerThreads, kdQHeadPackFactor>(
                  sdQ_inner,
                  &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_q.index() * kBlockM],
                  ptr_dQ_base,
                  stride_dq_token,
                  flat_thread_idx,
                  /*tile_offset=*/0,
                  stride_dq_head);
            } else {
              // Dense TMA reduce: consumer-side dQ store via TMA reduce-add
              static_assert(!CatGQA, "Consumer dQ TMA store for CatGQA not yet implemented; use InnerStoreInProducer");
              if (thread_idx == 0) {
                Tensor sdQ_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_inner_dq.data()), SmemLayoutdQSwizzled{});
                tma_inner_store(params.tma_add_dQ, sdQ_tma, domain_offset(make_coord(offset_q, _0{}), mdQ_reduce), TileShape_InnerDq{}, gQdOdQ_coord, m_block);
              }
            }

            // Cross-WG sync: all scatter reads done before the next iteration's r2s overwrites sdQ
            BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
          }
          if constexpr (InnerStoreInProducer) {
            // Producer store path: signal the producer store warp (TMA or scatter) that
            // sdQ is full; pairs with the dQEmpty sync at the top of this block.
            BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warp_group_idx);
          }
        } else { // directly atomic reduce-add to global memory
          static_assert(!(IsSparse && !BwdInnerLoopK), "BlockSparse InnerLoopQ requires dQ_use_smem (kHeadDim <= 128)");
          // We can reuse r2s_thr_copy_inner_dq for this partitioning
          Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_inner_dq.retile_S(tdQrdQ));
          Tensor tdQgdQ_reduce_atomic = recast<float4>(tdQgdQ_reduce(_, _, _, _, _, m_block));

          // FIXME: size(tdQrdQ_atomic) and size(tdQgdQ_reduce_atomic) are not matched
          static_assert(CUTE_STATIC_V(size(tdQrdQ_atomic)) == CUTE_STATIC_V(size(tdQgdQ_reduce_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
            atomicAdd(&tdQgdQ_reduce_atomic(i), tdQrdQ_atomic(i));
          }
        }
      } else { // Slice_dQKV_Mma, and guaranteed not Mma_dKV_is_RS
        // MMA3-1 (SS, M_slice=0): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
        // note that `sPt` stores P^T and `tdVrdO` stores dO^T, so:
        // case1. if dKV_swapAB, we apply dV^T = dO^TP (passing P^T,dO^T to gemm, it swaps AB to dO^T,P^T and then transposes operand B to P)
        // case2. if not dKV_swapAB, we apply dV = P^TdO (passing P^T,dO^T to gemm, it transposes operand B to dO)
        Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
        Tensor tdVrP_cur = tdVrP(_, _, _, _0{}); // P is 1-stage
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
        Tensor tdQrdQ_atomic = recast<float4>(r2s_thr_copy_inner_dq.retile_S(tdQrdQ));
        Tensor tdQgdQ_reduce_atomic = recast<float4>(tdQgdQ_reduce(_, _, _, _, _, m_block));
#pragma unroll
        for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) {
          atomicAdd(&tdQgdQ_reduce_atomic(i), tdQrdQ_atomic(i));
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
          atomicAdd(&tdQgdQ_reduce_atomic(i), tdQrdQ_atomic(i));
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

    // Unified MMA body: iterates over all m_blocks in the range with a single bwd_step instantiation.
    auto mma_body = [&]() {
      if constexpr (IsSparse) {
        // Scatter (BlockSparse / IndexSparse InnerLoopQ): one Q block per call, block_meta drives iteration.
        // InnerLoopQ needs both padding masks: rows are the scattered Q tokens
        // (last tile may hold fewer than kBlockM valid tokens) and columns are
        // the contiguous K window (last n_block may overhang seqlen_k) —
        // symmetric with InnerLoopK, where the roles of rows/columns are swapped.
        bool const need_row_mask = block_meta.inner_block_idx == block_meta.padding_block() && block_meta.num_invalid_token > 0;
        int const num_invalid_k_token = !BwdInnerLoopK ? cute::max(0, (block_meta.outer_tile_idx + 1) * kBlockN - block_meta.seqlen_info.seqlen_k) : 0;
        bool const need_col_mask = num_invalid_k_token > 0;
        auto combined_mask_fn = [&](int /*m_blk*/) {
          if (need_col_mask) {
            mask.template apply_padding_mask(tSrS, num_invalid_k_token, thread_idx);
          }
          if (need_row_mask) {
            mask.template apply_padding_mask_row<BlockMetaT::kPaddingAtLowEnd>(tSrS, block_meta.num_invalid_token, thread_idx);
          }
        };
        auto sparse_no_mask_fn = [&](int /*m_blk*/) {};
        if (need_row_mask || need_col_mask) {
          bwd_step(block_meta.inner_block_idx, combined_mask_fn, cute::false_type{});
        } else {
          bwd_step(block_meta.inner_block_idx, sparse_no_mask_fn, cute::false_type{});
        }
      } else {
        rebind_dQ_reduce_tiles();

        for (int bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, PackGQAFactor); ++bidh_kv_cat) {
          if constexpr (MaskMode == 0) {
            // MaskMode 0 (regular): direct apply every block with Seqlenk_mask=true.
            auto mask_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/true, PackGQA, PackGQAFactor>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            int mb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_cnt);
            flash::iterate_range<kInnerDir>(mb, block_meta.inner_block_min, block_meta.inner_block_cnt, [&] { bwd_step(mb, mask_fn, cute::false_type{}); });
          } else if constexpr (MaskMode == 1) {
            // MaskMode 1 (dispatch): 3-lambda zone splitting (compile-time).
            auto boundary_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/true, PackGQA, PackGQAFactor>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            auto regular_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/false, PackGQA, PackGQAFactor>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            auto no_mask_fn = [&](int /*m_block*/) {};
            int mb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_cnt);
            flash::mask_dispatch<kBlockM, kBlockN, PackGQA, PackGQAFactor, flash::DispatchAxis::M, kInnerDir>(
                mb,
                block_meta.inner_block_min,
                block_meta.inner_block_cnt,
                n_block,
                block_meta.seqlen_info.seqlen_q,
                block_meta.seqlen_info.seqlen_k,
                block_meta.attn_type,
                bwd_step,
                boundary_fn,
                regular_fn,
                no_mask_fn);
          } else {
            // MaskMode 2 (unified): mask_dispatch_unified with runtime zone dispatch.
            flash::mask_dispatch_unified<kBlockM, kBlockN, PackGQA, PackGQAFactor, flash::DispatchAxis::M, kInnerDir>(block_meta, mask, tSrS, thread_idx, bwd_step);
          }
        }
      }
    };

    // ─── Unified MMA control flow ─── (mma_with_loop_q)
    if (block_meta.skip_to_first_valid())
      return false;

    block_meta.template update_block_cur<kInnerDir>();

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        mma_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
        block_meta.template update_block_cur<kInnerDir>();
      }
    } else {
      mma_body();
    }

    if constexpr (Q_dO_same_stages) {
      smem_pipe_read_do = smem_pipe_read_q;
    }

    return true;
  }

  // Perform a Consumer Prologue/Mainloop -- WGMMA for S,dP,dQ,dK,dV with softmax for P,dS
  // q for outer-loop and k for inner-loop
  template <flash::DispatchDirection kInnerDir, typename SharedStorage, typename FrgTensordQ, typename BlockMetaT>
  CUTLASS_DEVICE bool mma_with_loop_k(
      Params const& params,
      MainloopPipeline pipeline_k,
      MainloopPipeline_V pipeline_v,
      PipelineState& smem_pipe_read_k,
      PipelineState_V& smem_pipe_read_v,
      FrgTensordQ& tdQrdQ,
      int thread_idx,
      int& work_idx,
      BlockMetaT& block_meta,
      SharedStorage& shared_storage) {
    static_assert(BwdInnerLoopK, "mma_with_loop_k() must be called when BwdInnerLoopK is true");
    static_assert(!CatGQA, "mma_with_loop_k() is not implemented for CatGQA");
    static_assert(is_rmem<FrgTensordQ>::value, "dQ tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    // BlockMeta: fixed per function call
    int const m_block = block_meta.outer_tile_idx;
    int const bidh = block_meta.bidh;
    int const bidh_kv = block_meta.bidh_kv;
    int const seqlen_q = block_meta.seqlen_info.seqlen_q;
    int const seqlen_q_packed = !PackGQA ? seqlen_q : seqlen_q * PackGQAFactor;
    bool const is_last_m_block_this_batch = seqlen_q_packed - m_block * kBlockM <= kBlockM;

    // BlockMeta: per-batch values accessed directly via block_meta.inner_block_min/max,
    // block_meta.seqlen_info.seqlen_k, block_meta.attn_type.

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

    // r2s write targets use the Store layout (aliases the swizzled TMA layout unless the
    // bulk-reduce scatter path swaps in the row-contiguous layout).
    // Stage 0 tensors used as defaults; for DB, stage-specific tensors are constructed at R2S time.
    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVStore{}));
    Tensor sdKt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVtStore{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVStore{}));
    Tensor sdVt = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVtStore{}));

    Tensor sdPsumMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dpsum.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma_full = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_lse.data()), SmemLayoutLSEMma{});
    Tensor sLSEMma = sLSEMma_full(_0{}, _, _); // slice dummy dim 0 with size of 4
    Tensor sdPsumMma = sdPsumMma_full(_0{}, _, _); // slice dummy dim 0 with size of 4

    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);
    Layout warp_group_thread_layout = make_layout(make_shape(Int<NumConsumerWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

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

    auto r2s_tiled_copy_inner_dkv = make_tiled_copy_C(Copy_Atom<DefaultCopy, ElementAccum>{}, tiled_mma_dKV);
    auto r2s_thr_copy_inner_dkv = r2s_tiled_copy_inner_dkv.get_thread_slice(thread_idx);
    Tensor tdKsdK = r2s_thr_copy_inner_dkv.partition_D(cute::conditional_return<!dKV_swapAB>(sdK, sdKt));
    Tensor tdVsdV = r2s_thr_copy_inner_dkv.partition_D(cute::conditional_return<!dKV_swapAB>(sdV, sdVt));

    // Double-buffer store stage counter for InnerStoreInProducer R2S path.
    // Selects which inner dKV SMEM buffer to write (consumer) and read (producer).
    int consumer_store_stage = 0;
    auto make_r2s_dv_target = [&](int stg) {
      auto* p = smem_inner_dv_ptr(shared_storage.tensors.mainloop, stg);
      Tensor s1 = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(p), SmemLayoutdKVStore{}));
      Tensor s2 = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(p), SmemLayoutdKVtStore{}));
      return r2s_thr_copy_inner_dkv.partition_D(cute::conditional_return<!dKV_swapAB>(s1, s2));
    };
    auto make_r2s_dk_target = [&](int stg) {
      auto* p = smem_inner_dk_ptr(shared_storage.tensors.mainloop, stg);
      Tensor s1 = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(p), SmemLayoutdKVStore{}));
      Tensor s2 = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(p), SmemLayoutdKVtStore{}));
      return r2s_thr_copy_inner_dkv.partition_D(cute::conditional_return<!dKV_swapAB>(s1, s2));
    };
    // Pre-built per-stage targets with compile-time-foldable base offsets. Building the
    // target inside the hot loop from the runtime `consumer_store_stage` defeats ptxas
    // alignment analysis and devectorizes the R2S copy (STS.128 → STS.32, ~3x shared_st
    // instructions, measured -24% at 32K). A two-way branch on the stage keeps each
    // branch's address statically aligned.
    static_assert(kInnerStoreStages <= 2, "R2S stage dispatch below assumes at most 2 store stages");
    auto tdVsdVaccum_s0 = make_r2s_dv_target(0);
    auto tdVsdVaccum_s1 = make_r2s_dv_target(kInnerStoreStages - 1);
    auto tdKsdKaccum_s0 = make_r2s_dk_target(0);
    auto tdKsdKaccum_s1 = make_r2s_dk_target(kInnerStoreStages - 1);

    /* DEBUG */
    // Tensor cdKVsdKV = make_identity_tensor(SmemLayoutdKVSwizzled{}.shape());
    // Tensor tcdKVsdKV = r2s_thr_copy_inner_dkv.partition_D(cdKVsdKV);
    // if (thread_idx == 0) { print(sdK); print(sdV); printf("\n"); print(tcdKVsdKV); printf("\n"); }

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
      BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
    };

    int const offset_k = block_meta.seqlen_info.offset_k;

    // For the case where we do atomicAdd directly to gdK_reduce,gdV_reduce instead of using TMA
    Tensor mdK_reduce = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dK)), params.shape_dKdV, params.stride_dK)(_, _, bidh_kv);
    Tensor gdK_reduce_ = local_tile(domain_offset(make_coord(offset_k, _0{}), mdK_reduce), TileShape_InnerDkv{}, make_coord(_, _0{})); // (N, K, _)
    Tensor gdK_reduce = cute::flat_divide(gdK_reduce_, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

    Tensor mdV_reduce = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dV)), params.shape_dKdV, params.stride_dV)(_, _, bidh_kv);
    Tensor gdV_reduce_ = local_tile(domain_offset(make_coord(offset_k, _0{}), mdV_reduce), TileShape_InnerDkv{}, make_coord(_, _0{})); // (N, K, _)
    Tensor gdV_reduce = cute::flat_divide(gdV_reduce_, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

    // TMA-layout smem views for consumer sparse TMA stores (created fresh at each call site via tma_inner_store helper)

    // We can reuse r2s_thr_copy_inner_dkv for this partitioning
    Tensor tdKgdK_reduce = r2s_thr_copy_inner_dkv.partition_D(gdK_reduce);
    Tensor tdVgdV_reduce = r2s_thr_copy_inner_dkv.partition_D(gdV_reduce);

    auto rebind_dKV_reduce_tiles = [&]() {
      if constexpr (!RangeMerge) {
        return;
      }
      int const new_offset_k = block_meta.seqlen_info.offset_k;
      if constexpr (kInnerStoreMode == InnerStoreMode::BypassSmem) {
        gdK_reduce_ = local_tile(domain_offset(make_coord(new_offset_k, _0{}), mdK_reduce), TileShape_InnerDkv{}, make_coord(_, _0{}));
        gdK_reduce = cute::flat_divide(gdK_reduce_, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{}));
        gdV_reduce_ = local_tile(domain_offset(make_coord(new_offset_k, _0{}), mdV_reduce), TileShape_InnerDkv{}, make_coord(_, _0{}));
        gdV_reduce = cute::flat_divide(gdV_reduce_, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{}));
        tdKgdK_reduce = r2s_thr_copy_inner_dkv.partition_D(gdK_reduce);
        tdVgdV_reduce = r2s_thr_copy_inner_dkv.partition_D(gdV_reduce);
      }
    };

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) {
    // print(mdK_reduce); printf("\n"); print(gdK_reduce_); printf("\n"); print(gdK_reduce); printf("\n"); print(tdKgdK_reduce); printf("\n"); print(tdKsdK);
    // printf("\n"); print(mdV_reduce); printf("\n"); print(gdV_reduce_); printf("\n"); print(gdV_reduce); printf("\n"); print(tdVgdV_reduce); printf("\n");
    // print(tdVsdV); printf("\n"); printf("\n"); }

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask;

    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    // Wait until this m block of Q,dO,LSE,dPsum loaded
    // and copy LSE,dPsum from shared memory to registers.
    // This is a first-batch-only operation wrapped in a lambda for use inside while(true).
    auto wait_QdO_and_copy_LSE_dPsum = [&]() {
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
    };

    Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));

    // Define backward step lambda func
    auto bwd_step = [&](int n_block, auto mask_fn, auto /*is_no_mask*/ = cute::false_type{}) {
      // MMA1 (SS): apply S = QK^T (or S^T = KQ^T if SdP_swapAB)
      // after current n block of K loaded
      // note that `tSrQ` stores Q , `tSrK` stores K, so:
      // case1. if SdP_swapAB, we apply S^T = KQ^T (passing Q,K to gemm, it swaps AB to K,Q and then transposes operand B to Q^T)
      // case2. if not SdP_swapAB, we apply S = QK^T (passing Q,K to gemm, it transposes operand B to K^T)
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
      mask_fn(n_block);

      // Apply scaled softmax on `scores` in-place, storing P^T (or P if SdP_swapAB)
      // NOTE: since we cannot pad for each batch, we need to mask out the OOB LSE values
      // that might be read from other batch at each batch's last m block
      if (is_last_m_block_this_batch) {
        auto thread_mma = TiledMmaSdP{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMmaSdP{}.get_thread_slice(_0{});

        static constexpr int Row = !SdP_swapAB ? 0 : 1;
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SdP_swapAB ? kBlockM : kBlockN>, Int<!SdP_swapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(t0ScS.layout()));
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const seqlenq_row_limit = seqlen_q_packed - m_block * kBlockM - thread_row_offset;

#pragma unroll
        for (int mi = 0; mi < size<0>(scores); ++mi) {
          bool const is_oob = int(get<Row>(t0ScS_rowcol(mi, _0{}))) >= seqlenq_row_limit;
          float lse_scaled = get_lse_scaled(mi);
          lse_scaled = is_oob ? cutlass::platform::numeric_limits<float>::infinity() : lse_scaled;
#pragma unroll
          for (int ni = 0; ni < size<1>(scores); ++ni) {
            scores(mi, ni) = unsafe_softmax_log2(scores(mi, ni) * params.softmax_scale_log2, lse_scaled);
          }
        }
      } else {
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
      // BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
      // if (thread_idx == 0) {
      //   print_tensor(
      //     sP(_, _, cute::conditional_return<kStages_dS == 1>(_0{}, smem_pipe_read_k.index()))
      //   );
      // }

      // Reshape `tdPrdP` from ((2, 2, V), MMA_N, MMA_M) to (nrow=(2, V, MMA_M), ncol=(2, MMA_N))
      // and rename the view as `dS`, storing dP (or dP^T if SdP_swapAB)
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());

      // Wait for MMA2 to finish (V consumed). Release V immediately — V and dS
      // have separate SMEM buffers, so the producer can load V[j+1] while dS is used.
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
          BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
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
        if constexpr (PerfDebugSkipDvMma) {
          // Debug: skip dV MMA entirely; tdVrdV stays zero-initialized.
          // No WGMMA issued, so MMA4's wg_wait=1 is trivially satisfied (0 pending <= 1).
        } else if constexpr (Mma_dKV_is_RS) {
          Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
          flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_dKV, tdVrP, tdVrdO, tdVrdV);
        } else {
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
        if constexpr ((kInnerStoreMode == InnerStoreMode::BypassSmem)) {
          if constexpr (!(PerfDebugSkipDvStore || PerfDebugSkipDvMma)) {
            static_assert(
                !(kInnerStoreMode == InnerStoreMode::BypassSmem) || !dKV_swapAB,
                "(kInnerStoreMode == InnerStoreMode::BypassSmem) scatter requires !dKV_swapAB (kHeadDim<=128)");
            Tensor taccdVrdV = r2s_thr_copy_inner_dkv.retile_S(tdVrdV);
            if constexpr (IsSparse) {
              int const* const tidx = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
              int const stride_dV_token = get<0>(params.stride_dV);
              auto* ptr_gdV = reinterpret_cast<ElementAccum*>(params.ptr_dV) + bidh_kv * get<2>(params.stride_dV);
              Tensor cdKV = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
              Tensor cdKV_div = cute::flat_divide(cdKV, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{}));
              Tensor thr_cdKV = r2s_thr_copy_inner_dkv.partition_D(cdKV_div);
#pragma unroll
              for (int i = 0; i < size(taccdVrdV); ++i) {
                auto coord = thr_cdKV(i);
                int row = get<0>(coord);
                int col = get<1>(coord);
                int token_idx = tidx[row];
                atomicAdd(ptr_gdV + (token_idx)*stride_dV_token + col, taccdVrdV(i));
              }
            } else {
              Tensor tdVgdV_reduce_cur = tdVgdV_reduce(_, _, _, _, _, n_block);
#pragma unroll
              for (int i = 0; i < size(taccdVrdV); ++i) {
                atomicAdd(&tdVgdV_reduce_cur(i), taccdVrdV(i));
              }
            }
          } // !PerfDebugSkipDvStore
        } else if constexpr (InnerStoreInProducer) {
          // Path 2: R2S → smem, signal producer store warp for TMA/scatter GMEM writeback.
          // SkipDvStore: skip R2S but keep barrier handshake to prevent producer warp racing.
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);

          if constexpr (!PerfDebugSkipDvStore) {
            if constexpr (IsSparse) {
              if (warp_group_idx == 0) {
                int const* const src = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
                int* const dst = shared_storage.tensors.mainloop.smem_sparse_store_staging_indices.data(); // staging_dv
                for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockN; r += cutlass::NumThreadsPerWarpGroup) {
                  dst[r] = src[r];
                }
              }
            }

            Tensor taccdVrdV = r2s_thr_copy_inner_dkv.retile_S(tdVrdV);
            if constexpr (kInnerStoreStages >= 2) {
              if (consumer_store_stage == 0) {
                cute::copy(r2s_tiled_copy_inner_dkv, taccdVrdV, tdVsdVaccum_s0);
              } else {
                cute::copy(r2s_tiled_copy_inner_dkv, taccdVrdV, tdVsdVaccum_s1);
              }
            } else {
              cute::copy(r2s_tiled_copy_inner_dkv, taccdVrdV, tdVsdV);
            }

            cutlass::arch::fence_view_async_shared();
          } // !PerfDebugSkipDvStore

          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warp_group_idx);
        } else if constexpr (IsSparse) {
          // Path 3: Consumer self-store (sparse): R2S → smem, sync, GMEM write, sync
          static_assert(kInnerStoreMode != InnerStoreMode::BypassSmem, "Consumer scatter dKV requires smem SMEM buffer buffer (kHeadDim < 256)");

          Tensor taccdVrdV = r2s_thr_copy_inner_dkv.retile_S(tdVrdV);
          cute::copy(r2s_tiled_copy_inner_dkv, taccdVrdV, tdVsdV);
          cutlass::arch::fence_view_async_shared();

          BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);

          if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
            // Contiguous sparse: TMA 2D reduce (thread 0 only)
            if (thread_idx == 0) {
              Tensor sdV_tma_c = make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVSwizzled{});
              int const compound_idx = shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
              tma_inner_store(
                  params.tma_add_dV,
                  sdV_tma_c,
                  params.tma_add_dV.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv),
                  TileShape_InnerDkv{},
                  make_coord(_, _0{}),
                  compound_idx / kBlockN);
            }
          } else {
            // Non-contiguous sparse: scatter store (all consumer threads)
            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
            int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
            int const stride_dV_token = get<0>(params.stride_dV);
            ElementAccum* const ptr_gdV_base = params.ptr_dV + bidh_kv * get<2>(params.stride_dV);
            Tensor sdV_store = make_tensor(make_smem_ptr(smem_inner_dv_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVStore{});
            scatter_inner_store<kBlockN, NumConsumerThreads, /*kInnerStoreHeadPackFactor=*/1>(
                sdV_store,
                &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN],
                ptr_gdV_base,
                stride_dV_token,
                flat_thread_idx);
          }

          BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
        } else {
          // Path 4: Dense consumer float4 atomicAdd (register→GMEM directly, no smem)
          Tensor tdVrdV_atomic = recast<float4>(r2s_thr_copy_inner_dkv.retile_S(tdVrdV));
          Tensor tdVgdV_reduce_atomic = recast<float4>(tdVgdV_reduce(_, _, _, _, _, n_block));
          static_assert(CUTE_STATIC_V(size(tdVrdV_atomic)) == CUTE_STATIC_V(size(tdVgdV_reduce_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdVrdV_atomic); ++i) {
            atomicAdd(&tdVgdV_reduce_atomic(i), tdVrdV_atomic(i));
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
        if constexpr ((kInnerStoreMode == InnerStoreMode::BypassSmem)) {
          if constexpr (!PerfDebugSkipDkStore) {
            Tensor taccdKrdK = r2s_thr_copy_inner_dkv.retile_S(tdKrdK);
#pragma unroll
            for (int dki = 0; dki < size(taccdKrdK); ++dki) {
              taccdKrdK(dki) *= params.softmax_scale;
            }
            if constexpr (IsSparse) {
              int const* const tidx = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
              int const stride_dK_token = get<0>(params.stride_dK);
              auto* ptr_gdK = reinterpret_cast<ElementAccum*>(params.ptr_dK) + bidh_kv * get<2>(params.stride_dK);
              Tensor cdKV = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
              Tensor cdKV_div = cute::flat_divide(cdKV, make_shape(Int<kBlockN / NumConsumerWarpGroups>{}, Int<kHeadDim>{}));
              Tensor thr_cdKV = r2s_thr_copy_inner_dkv.partition_D(cdKV_div);
#pragma unroll
              for (int i = 0; i < size(taccdKrdK); ++i) {
                auto coord = thr_cdKV(i);
                int row = get<0>(coord);
                int col = get<1>(coord);
                int token_idx = tidx[row];
                atomicAdd(ptr_gdK + (token_idx)*stride_dK_token + col, taccdKrdK(i));
              }
            } else {
              Tensor tdKgdK_reduce_cur = tdKgdK_reduce(_, _, _, _, _, n_block);
#pragma unroll
              for (int i = 0; i < size(taccdKrdK); ++i) {
                atomicAdd(&tdKgdK_reduce_cur(i), taccdKrdK(i));
              }
            }
          } // !PerfDebugSkipDkStore
        } else if constexpr (InnerStoreInProducer) {
          // Write to smem, signal producer store warp for TMA/scatter reduce-add
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          BarrierManager::sync<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warp_group_idx);

          if constexpr (!PerfDebugSkipDkStore) {
            if constexpr (IsSparse) {
              if (warp_group_idx == 0) {
                int const* const src = &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
                int* const dst = &shared_storage.tensors.mainloop.smem_sparse_store_staging_indices[kBlockN]; // staging_dk
                for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockN; r += cutlass::NumThreadsPerWarpGroup) {
                  dst[r] = src[r];
                }
              }
            }

            Tensor taccdKrdK = r2s_thr_copy_inner_dkv.retile_S(tdKrdK);
            for (int dki = 0; dki < size(taccdKrdK); ++dki) {
              taccdKrdK(dki) *= params.softmax_scale;
            }
            if constexpr (kInnerStoreStages >= 2) {
              if (consumer_store_stage == 0) {
                cute::copy(r2s_tiled_copy_inner_dkv, taccdKrdK, tdKsdKaccum_s0);
              } else {
                cute::copy(r2s_tiled_copy_inner_dkv, taccdKrdK, tdKsdKaccum_s1);
              }
            } else {
              cute::copy(r2s_tiled_copy_inner_dkv, taccdKrdK, tdKsdK);
            }

            cutlass::arch::fence_view_async_shared();
          } // !PerfDebugSkipDkStore

          BarrierManager::arrive<NumInnerStoreBarrierThreads>(BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warp_group_idx);

          // Advance consumer store stage for double-buffer
          if constexpr (kInnerStoreStages >= 2) {
            consumer_store_stage = (consumer_store_stage + 1) % kInnerStoreStages;
          }
        } else if constexpr (IsSparse) {
          // Path 3: Consumer self-store (sparse): scale + R2S → smem, sync, GMEM write, sync
          static_assert(kInnerStoreMode != InnerStoreMode::BypassSmem, "Consumer scatter dKV requires smem SMEM buffer buffer (kHeadDim < 256)");

          Tensor taccdKrdK = r2s_thr_copy_inner_dkv.retile_S(tdKrdK);
          for (int dki = 0; dki < size(taccdKrdK); ++dki) {
            taccdKrdK(dki) *= params.softmax_scale;
          }
          cute::copy(r2s_tiled_copy_inner_dkv, taccdKrdK, tdKsdK);
          cutlass::arch::fence_view_async_shared();

          BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);

          if constexpr (kInnerLoadMode == InnerLoadMode::Tma) {
            // Contiguous sparse: TMA 2D reduce (thread 0 only)
            if (thread_idx == 0) {
              Tensor sdK_tma_c = make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVSwizzled{});
              int const compound_idx = shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN];
              tma_inner_store(
                  params.tma_add_dK,
                  sdK_tma_c,
                  params.tma_add_dK.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv),
                  TileShape_InnerDkv{},
                  make_coord(_, _0{}),
                  compound_idx / kBlockN);
            }
          } else {
            // Non-contiguous sparse: scatter store (all consumer threads)
            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
            int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
            int const stride_dK_token = get<0>(params.stride_dK);
            ElementAccum* const ptr_gdK_base = params.ptr_dK + bidh_kv * get<2>(params.stride_dK);
            Tensor sdK_store = make_tensor(make_smem_ptr(smem_inner_dk_ptr(shared_storage.tensors.mainloop)), SmemLayoutdKVStore{});
            scatter_inner_store<kBlockN, NumConsumerThreads, /*kInnerStoreHeadPackFactor=*/1>(
                sdK_store,
                &shared_storage.tensors.mainloop.smem_sparse_inner_indices[smem_pipe_read_k.index() * kBlockN],
                ptr_gdK_base,
                stride_dK_token,
                flat_thread_idx);
          }

          BarrierManager::sync<NumConsumerThreads>(BwdNamedBarriers::PdS);
        } else {
          // Path 4: Dense consumer float4 atomicAdd (register→GMEM directly, no smem)
          Tensor tdKrdK_atomic = recast<float4>(r2s_thr_copy_inner_dkv.retile_S(tdKrdK));
          Tensor tdKgdK_reduce_atomic = recast<float4>(tdKgdK_reduce(_, _, _, _, _, n_block));
          static_assert(CUTE_STATIC_V(size(tdKrdK_atomic)) == CUTE_STATIC_V(size(tdKgdK_reduce_atomic)));
#pragma unroll
          for (int i = 0; i < size(tdKrdK_atomic); ++i) {
            atomicAdd(&tdKgdK_reduce_atomic(i), tdKrdK_atomic(i));
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
        Tensor tdVrdV_atomic = recast<float4>(r2s_thr_copy_inner_dkv.retile_S(tdVrdV));
        Tensor tdVgdV_reduce_atomic = recast<float4>(tdVgdV_reduce(_, _, _, _, _, n_block));
#pragma unroll
        for (int i = 0; i < size(tdVrdV_atomic) / 2; ++i) {
          atomicAdd(&tdVgdV_reduce_atomic(i), tdVrdV_atomic(i));
        }

        // MMA3-2 (SS, M_slice=1): apply dV = P^TdO (or dV^T = dO^TP if dKV_swapAB)
        flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO, tdVrdV);

        // Atomic reduce-add partial dK (M_slice=0) directly to global memory
        // after MMA4-1 finished (wg_wait<1> in MMA3-2)
        Tensor tdKrdK_atomic = recast<float4>(r2s_thr_copy_inner_dkv.retile_S(tdKrdK));
        Tensor tdKgdK_reduce_atomic = recast<float4>(tdKgdK_reduce(_, _, _, _, _, n_block));
#pragma unroll
        for (int i = 0; i < size(tdKrdK_atomic) / 2; ++i) {
          atomicAdd(&tdKgdK_reduce_atomic(i), tdKrdK_atomic(i));
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
          atomicAdd(&tdVgdV_reduce_atomic(i), tdVrdV_atomic(i));
        }

        // MMA4-2 (SS, M_slice=1): apply dK = dS^TQ (or dK^T = Q^TdS if dKV_swapAB)
        flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ, tdKrdK);

#pragma unroll
        // Atomic reduce-add partial dK (M_slice=1) directly to global memory
        // after MMA4-2 finished (wg_wait<0> in MMA4-2)
        for (int i = size(tdKrdK_atomic) / 2; i < size(tdKrdK_atomic); ++i) {
          atomicAdd(&tdKgdK_reduce_atomic(i), tdKrdK_atomic(i));
        }

        // MMA5-2 (SS, M_slice=1): apply dQ = dSK (or dQ^T = K^TdS^T if dQ_swapAB)
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(
            tiled_mma_dQ, tdQrdS_cur, tdQrK(_, _, _, smem_pipe_read_k.index()), tdQrdQ);
      }

      // Release K after MMA5 finished (V already released after MMA2).
      warpgroup_wait<0>();
      pipeline_k.consumer_release(smem_pipe_read_k);

      // Update pipeline read state of K,V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
    };

    // --- Mask lambdas ---
    auto padding_mask_fn = [&](int /*n_blk*/) {
      if constexpr (IsSparse) {
        mask.template apply_padding_mask<BlockMetaT::kPaddingAtLowEnd>(tSrS, block_meta.num_invalid_token, thread_idx);
      }
    };
    auto sparse_no_mask_fn = [&](int /*n_blk*/) {};

    // Unified MMA body: scatter processes one n_block per call;
    // dense iterates over all n_blocks in the range with a single bwd_step instantiation.
    auto mma_body = [&]() {
      if constexpr (IsSparse) {
        if (block_meta.inner_block_idx == block_meta.padding_block() && block_meta.num_invalid_token > 0) {
          bwd_step(block_meta.inner_block_idx, padding_mask_fn, cute::false_type{});
        } else {
          bwd_step(block_meta.inner_block_idx, sparse_no_mask_fn, cute::false_type{});
        }
        return;
      }
      rebind_dKV_reduce_tiles();

      if constexpr (MaskMode == 0) {
        // MaskMode 0 (regular): direct apply every block with Seqlenk_mask=true.
        auto mask_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/true, PackGQA, PackGQAFactor>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        int nb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_cnt);
        flash::iterate_range<kInnerDir>(nb, block_meta.inner_block_min, block_meta.inner_block_cnt, [&] { bwd_step(nb, mask_fn, cute::false_type{}); });
      } else if constexpr (MaskMode == 1) {
        // MaskMode 1 (dispatch): 3-lambda zone splitting.
        auto boundary_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/true, PackGQA, PackGQAFactor>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        auto regular_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/false, PackGQA, PackGQAFactor>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        auto no_mask_fn = [&](int /*n_blk*/) {};
        int nb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_cnt);
        flash::mask_dispatch<kBlockM, kBlockN, PackGQA, PackGQAFactor, flash::DispatchAxis::N, kInnerDir>(
            nb,
            block_meta.inner_block_min,
            block_meta.inner_block_cnt,
            m_block,
            seqlen_q,
            block_meta.seqlen_info.seqlen_k,
            block_meta.attn_type,
            bwd_step,
            boundary_fn,
            regular_fn,
            no_mask_fn);
      } else {
        // MaskMode 2 (unified): mask_dispatch_unified with runtime zone dispatch.
        flash::mask_dispatch_unified<kBlockM, kBlockN, PackGQA, PackGQAFactor, flash::DispatchAxis::N, kInnerDir>(block_meta, mask, tSrS, thread_idx, bwd_step);
      }
    };

    // --- Unified MMA control flow ---
    block_meta.skip_to_first_valid();
    if (block_meta.is_finish())
      return false;

    block_meta.template update_block_cur<kInnerDir>();
    wait_QdO_and_copy_LSE_dPsum();

    if constexpr (BlockMetaT::NeedsBatchLoop) {
      while (true) {
        mma_body();
        block_meta.prefetch();
        if (block_meta.skip_to_first_valid())
          break;
        block_meta.template update_block_cur<kInnerDir>();
      }
    } else {
      mma_body();
    }

    return true;
  }

  // Debug print some crucial configuration about mma
  // especially for the tiled mma definition
  CUTLASS_DEVICE void debug_print_mma(int block_idx = 0, int thread_idx = 128) {
    if (blockIdx.x == block_idx && threadIdx.x == thread_idx) {
      printf(
          "kBlockM=%d, kBlockN=%d, kHeadDim=%d | dQ_swapAB=%d, dKV_swapAB=%d, SdP_swapAB=%d | Mma_dQ_is_RS=%d, Mma_dKV_is_RS=%d\n",
          kBlockM,
          kBlockN,
          kHeadDim,
          dQ_swapAB,
          dKV_swapAB,
          SdP_swapAB,
          Mma_dQ_is_RS,
          Mma_dKV_is_RS);

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
