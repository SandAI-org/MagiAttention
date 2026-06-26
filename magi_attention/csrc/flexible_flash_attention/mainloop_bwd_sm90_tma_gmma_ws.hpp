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
    bool SwapBwdQKLoop_,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    bool PackGQA_,
    bool CatGQA_,
    bool RangeMerge_,
    bool BlockSparse_,
    bool IndexSparse_,
    bool UseMaskDispatch_,
    bool InnerDirMaxToMin_,
    int MaskMode_,
    bool InnerDxStoreInProducer_,
    bool SparseInnerDxReduceUseTma_,
    int QheadPerKhead_,
    int NumMmaWarpGroups,
    int AtomLayoutMSdP,
    int AtomLayoutNdKV,
    int AtomLayoutMdQ,
    bool Mma_dP_is_RS,
    int Stages_V_ = Stages,
    int ScatterPad_ = -1,
    bool LseDpsumUnionDKVacc_ = false,
    bool DkvaccBypassSmem_ = false,
    int KBlockSize_ = 1,
    bool ForceMmaDkvSS_ = false>
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
  static_assert(!Mma_dP_is_RS || SdP_swapAB_); // If Mma_dP_is_RS, we need SdP_SwapAB

  static constexpr bool LseDpsumUnionDKVacc = LseDpsumUnionDKVacc_;
  static constexpr bool DkvaccBypassSmem = DkvaccBypassSmem_;

  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;
  static constexpr bool SwapBwdQKLoop = SwapBwdQKLoop_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr bool CatGQA = CatGQA_;
  static constexpr bool FlattenGQA = PackGQA_ || CatGQA_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr int QheadPerKhead = QheadPerKhead_;
  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;
  static constexpr bool BlockSparse = BlockSparse_;
  static constexpr bool IndexSparse = IndexSparse_;
  static constexpr int KBlockSize = KBlockSize_;
  static_assert(!BlockSparse || RangeMerge); // If BlockSparse, we need RangeMerge
  static_assert(!(BlockSparse && IndexSparse));
  static_assert(!IndexSparse || KBlockSize >= 1, "KBlockSize must be >= 1 for IndexSparse");

  static constexpr bool UseMaskDispatch = UseMaskDispatch_;
  static constexpr bool InnerDirMaxToMin = InnerDirMaxToMin_;
  static constexpr int MaskMode = MaskMode_;
  // InnerUseScatter: inner-loop direction data uses scatter load/store (vs TMA):
  // LoopK (SwapBwdQKLoop=true):  KV scatter when BlockSparse or IndexSparse
  // LoopQ (SwapBwdQKLoop=false): Q/dO scatter when BlockSparse or IndexSparse (inv_indices)
  static constexpr bool InnerUseScatter = BlockSparse || IndexSparse;
  // LoopQ scatter does not support CatGQA (the dense LoopQ load iterates bidh_kv_cat
  // per merged sub-range; the scatter load path has no such loop).
  static_assert(!(InnerUseScatter && !SwapBwdQKLoop && CatGQA), "bwd LoopQ scatter (block_sparse) does not support cat_gqa");

  // InnerDxStoreInProducer: who performs the inner-loop dX (dKV for LoopK / dQ for LoopQ)
  // store. Pure pass-through of the template/env toggle; the python JIT entry only emits
  // a non-default value for scatter configs (dense + consumer-store is valid in principle
  // -- the contiguous-atomicAdd consumer branch exists -- but is untested and currently
  // trips an nvcc ICE, so the entry point does not generate it).
  static constexpr bool InnerDxStoreInProducer = InnerDxStoreInProducer_;

  // SparseInnerDxReduceUseTma: scatter dX store uses per-row cp.reduce.async.bulk (one bulk
  // reduce-add per token row) instead of per-4B scalar atomicAdd. Requires a row-contiguous
  // smem accum layout (see SmemLayoutdKVaccumStore / SmemLayoutdQaccumStore below).
  // Pure pass-through; only meaningful on scatter paths. Guarded below because a dense build
  // with this flag set would silently mismatch the r2s-write (flat) vs TMA-read (swizzled)
  // accum layouts and corrupt results.
  static constexpr bool SparseInnerDxReduceUseTma = SparseInnerDxReduceUseTma_;
  static_assert(!SparseInnerDxReduceUseTma_ || InnerUseScatter, "SparseInnerDxReduceUseTma requires a scatter path (BlockSparse / IndexSparse)");

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  // ─── Inner-Loop Load/Store Strategy ───
  // Whether TMA 2D is used for inner-loop loads (Q/dO in LoopQ, K/V in LoopK)
  // and stores (dK/dV reduce in LoopQ, dQ reduce in LoopK).
  // TMA 2D requires physically contiguous tiles in global memory:
  //   Dense: always contiguous → TMA.
  //   LoopQ sparse (inner=Q/dO): PackGQA makes tiles contiguous when:
  //     - BlockSparse: always (packed rows = consecutive tokens × heads).
  //     - IndexSparse: QheadPerKhead >= kBlockM (one Q token fills the tile).
  //   LoopK sparse (inner=K/V): K tiles are contiguous when:
  //     - BlockSparse: KBlockSize >= kBlockN (range covers full tile).
  //     - IndexSparse: PackGQA + QheadPerKhead >= kBlockM + KBlockSize >= kBlockN.
  // Fallback: per-row cp.async scatter (PipelineAsync, all 64 loader threads).
  static constexpr bool Use_TMA_Inner = !InnerUseScatter || // Non-scatter (Dense): always TMA
      (!SwapBwdQKLoop && PackGQA && (!IndexSparse || QheadPerKhead >= kBlockM)) ||
      (SwapBwdQKLoop && ((BlockSparse && KBlockSize >= kBlockN) || (IndexSparse && PackGQA && QheadPerKhead >= kBlockM && KBlockSize >= kBlockN)));
  static constexpr bool Use_CpAsync_Inner = !Use_TMA_Inner;

  using MainloopPipeline = std::conditional_t<Use_TMA_Inner, typename cutlass::PipelineTmaAsync<kStages>, typename cutlass::PipelineAsync<kStages>>;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipeline_dO = std::conditional_t<Use_TMA_Inner, typename cutlass::PipelineTmaAsync<kStages_dO>, typename cutlass::PipelineAsync<kStages_dO>>;
  using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
  using MainloopPipeline_V = std::conditional_t<Use_TMA_Inner, typename cutlass::PipelineTmaAsync<kStages_V>, typename cutlass::PipelineAsync<kStages_V>>;
  using PipelineState_V = typename MainloopPipeline_V::PipelineState;
  using TMAClusterBarrier_t = cutlass::arch::ClusterTransactionBarrier::ValueType;
  using BwdNamedBarriers = std::conditional_t<SwapBwdQKLoop, BwdNamedBarriersLoopK, BwdNamedBarriersLoopQ>;

  static_assert(BarrierManager::check<BwdNamedBarriers, NumMmaWarpGroups>());

  using SeqlenInfo_t = flash::SeqlenInfo;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, PackGQA, QheadPerKhead>;

  static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
  static_assert(NumMmaWarpGroups % AtomLayoutMdQ == 0);
  static constexpr int AtomLayoutNSdP = NumMmaWarpGroups / AtomLayoutMSdP;
  static constexpr int AtomLayoutMdKV = NumMmaWarpGroups / AtomLayoutNdKV;
  static constexpr int AtomLayoutNdQ = NumMmaWarpGroups / AtomLayoutMdQ;

  static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

  // ─── ProducerWarpRoles: centralized producer warp role configuration ───
  // Replaces scattered magic-number warp_idx checks and hand-written barrier widths.
  struct ProducerWarpRoles {
    // Loader warps: scatter paths need 2 warps for cp.async bandwidth, else 1 (TMA)
    static constexpr int kNumLoaderWarps = InnerUseScatter ? 2 : 1;
    // DxStorer warps: 0 if !InnerDxStoreInProducer (consumer handles store),
    //                 else 2 (LoopK: dK+dV) or 1 (LoopQ: dQ)
    static constexpr int kNumDxStorerWarps = !InnerDxStoreInProducer ? 0 : (SwapBwdQKLoop ? 2 : 1);
    static constexpr int kNumTotalWarps = kNumLoaderWarps + kNumDxStorerWarps;

    // Thread counts (derived)
    static constexpr int kLoaderThreads = kNumLoaderWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kDxStorerThreads = kNumDxStorerWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kTotalThreads = kNumTotalWarps * cutlass::NumThreadsPerWarp;

    // Per-direction store thread count for dKV barriers:
    //   InnerUseScatter: all DxStorer warps participate in BOTH dV and dK → kDxStorerThreads
    //   Dense LoopK: warp1→dV, warp2→dK, each barrier involves 1 warp → NumThreadsPerWarp
    static constexpr int kPerDirDkvStoreThreads = !InnerDxStoreInProducer ? 0 : (InnerUseScatter ? kDxStorerThreads : cutlass::NumThreadsPerWarp);

    // Barrier participant counts (THE source of truth)
    // Outer-empty (KVEmpty for LoopQ, QdOEmpty for LoopK): loader threads only
    static constexpr int kOuterEmptyBarrierThreads = kLoaderThreads;
    // dQ barrier (LoopQ): consumer WG + dQ store warp (if InnerDxStoreInProducer)
    static constexpr int kDqBarrierThreads = cutlass::NumThreadsPerWarpGroup + (SwapBwdQKLoop ? 0 : kDxStorerThreads);
    // dKV per-direction barrier (LoopK): consumer WG + per-dir store threads
    static constexpr int kDkvBarrierThreads = cutlass::NumThreadsPerWarpGroup + (SwapBwdQKLoop ? kPerDirDkvStoreThreads : 0);

    // Role predicates (warp_idx is 0-based within the producer warp group)
    static CUTLASS_DEVICE bool is_loader(int warp_idx) {
      return warp_idx < kNumLoaderWarps;
    }
    static CUTLASS_DEVICE bool is_dx_storer(int warp_idx) {
      return InnerDxStoreInProducer && warp_idx >= kNumLoaderWarps && warp_idx < kNumTotalWarps;
    }
    static CUTLASS_DEVICE bool is_leader_loader(int warp_idx) {
      return warp_idx == 0;
    }
  };

  // Aliases for backward compatibility with existing code references
  static constexpr int NumProducerThreads = ProducerWarpRoles::kTotalThreads;
  static constexpr int NumBlockSparseThreads = cutlass::NumThreadsPerWarp * 2;
  // Per-direction store barrier width (NOT total DxStorer threads)
  static constexpr int NumdKVStoreThreads = ProducerWarpRoles::kPerDirDkvStoreThreads;
  static constexpr int NumKVEmptyProducerThreads = ProducerWarpRoles::kOuterEmptyBarrierThreads;
  static constexpr int NumdQBarrierThreads = ProducerWarpRoles::kDqBarrierThreads;

  static_assert(!InnerDxStoreInProducer || ProducerWarpRoles::kNumDxStorerWarps > 0);
  static_assert(InnerDxStoreInProducer || ProducerWarpRoles::kNumDxStorerWarps == 0);
  static_assert(ProducerWarpRoles::kNumTotalWarps * cutlass::NumThreadsPerWarp == NumProducerThreads);

  // Const parameters for scatter load/store
  static constexpr int kCpAsyncTransactionBytes = 128;
  static constexpr int GroupSize = kCpAsyncTransactionBytes / 16;
  static constexpr int NumGroups = NumBlockSparseThreads / GroupSize;
  // Only the inner (scatter-side) tensor is gathered row-by-row, so the per-group row
  // count is sized by the inner tile: kBlockN (KV) for LoopK, kBlockM (Q/dO) for LoopQ.
  // The smem token-index array (SmemTokenIndices_t below) is sized by the same dimension.
  static constexpr int kInnerScatterRows = SwapBwdQKLoop ? kBlockN : kBlockM;
  static constexpr int NumRowsPerGroup = kInnerScatterRows / NumGroups;
  static constexpr int NumCpAsyncTilesPerRow = kHeadDim * sizeof(Element) / kCpAsyncTransactionBytes;
  static constexpr int kStoreVecWidth = kCpAsyncTransactionBytes / (GroupSize * sizeof(ElementAccum));
  static constexpr int kNumStoreTiles = kHeadDim / (GroupSize * kStoreVecWidth);

  static_assert(!InnerUseScatter || kInnerScatterRows % NumGroups == 0, "Scatter requires the inner tile rows divisible by NumGroups");

  static constexpr bool Mma_dKV_is_RS = !ForceMmaDkvSS_ && AtomLayoutMSdP == 1 && AtomLayoutMdKV == 1 && SdP_swapAB && !dKV_swapAB;
  static constexpr bool Mma_dQ_is_RS = AtomLayoutNSdP == 1 && AtomLayoutNdQ == 1 && !SdP_swapAB && !dQ_swapAB; // If dQ_swapAB, we can't use RS

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

  // ─── Scatter dX store smem layouts ───
  // 1D cp.reduce.async.bulk needs each token row to be one LINEAR smem span. The swizzled
  // *accumTMA layouts cannot provide that: their SW128 atom (8 rows x 32 floats, column-major
  // tiled) physically interleaves one logical row into kHeadDim/32 separate 128B chunks that
  // are 8*128B apart. So the 1D bulk-reduce path uses a row-contiguous layout.
  // A 4-float (16B) row pad keeps rows 16B-aligned (bulk reduce requirement) while breaking
  // the worst r2s store bank conflicts (8-way unpadded -> <=2-way padded).
  // Use_TMA_Inner && InnerUseScatter bypasses 1D bulk-reduce entirely (2D TMA reduce instead),
  // keeping the swizzled TMA layout → no bank conflicts, no padding needed.
  static constexpr int kScatterAccRowPad = ScatterPad_ >= 0 ? ScatterPad_ : 4; // floats; -1 = auto (default 4)
  using SmemLayoutdKVaccumScatter = Layout<Shape<Int<kBlockN>, Int<kHeadDim>>, Stride<Int<kHeadDim + kScatterAccRowPad>, _1>>;
  using SmemLayoutdQaccumScatter = Layout<Shape<Int<kBlockM>, Int<kHeadDim>>, Stride<Int<kHeadDim + kScatterAccRowPad>, _1>>;
  // Store-side accum layouts: r2s writes and scatter-store reads go through these.
  // They alias SmemLayoutd*accumTMA unless the 1D bulk-reduce path is active.
  // When Use_TMA_Inner && InnerUseScatter, 2D TMA reduce reads the swizzled layout natively.
  // The flat Scatter layout only applies when SparseInnerDxReduceUseTma && Use_CpAsync_Inner
  // (1D per-row bulk reduce fallback), and only for the inner dX of this loop.
  using SmemLayoutdKVaccumStore = std::conditional_t<SparseInnerDxReduceUseTma && SwapBwdQKLoop && Use_CpAsync_Inner, SmemLayoutdKVaccumScatter, SmemLayoutdKVaccumTMA>;
  using SmemLayoutdQaccumStore = std::conditional_t<SparseInnerDxReduceUseTma && !SwapBwdQKLoop && Use_CpAsync_Inner, SmemLayoutdQaccumScatter, SmemLayoutdQaccumTMA>;
  using SmemLayoutdKVaccumtStore =
      decltype(cute::composition(SmemLayoutdKVaccumStore{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), make_stride(Int<kBlockN>{}, _1{}))));
  using SmemLayoutdQaccumtStore =
      decltype(cute::composition(SmemLayoutdQaccumStore{}, make_layout(make_shape(Int<kHeadDim>{}, Int<kBlockM>{}), make_stride(Int<kBlockM>{}, _1{}))));
  static_assert(kHeadDim * sizeof(ElementAccum) % 16 == 0, "bulk reduce-add requires 16B-multiple row size");

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

  using ShapeQKV = cute::Shape<int32_t, Int<kHeadDim>, int32_t>; // (seqlen, head_dim, num_heads)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t>;
  using ShapeLSE = cute::Shape<_4, int32_t, int32_t>; // (4, seqlen_q, num_heads_q)
  using StrideLSE = cute::Stride<_1, _4, int64_t>;

  // Define ShapeLSETMA and StrideLSETMA based on PackGQA and CatGQA,
  // which will be used for loading LSE and dPsum from global memory to shared memory
  using ShapeLSETMA = std::conditional_t<
      PackGQA,
      // (4, (qhead_per_khead, seqlen_q), nheads_kv)
      cute::Shape<_4, cute::Shape<cute::Int<QheadPerKhead>, int32_t>, int32_t>,
      std::conditional_t<
          CatGQA,
          // (4, seqlen_q, (qhead_per_khead, nheads_kv))
          cute::Shape<_4, int32_t, cute::Shape<cute::Int<QheadPerKhead>, int32_t>>,
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
      cute::Shape<cute::Shape<cute::Int<QheadPerKhead>, int32_t>, Int<kHeadDim>, int32_t>,
      std::conditional_t<
          CatGQA,
          // Case 2: CatGQA is enabled
          // Shape: (seqlen, headdim, (qhead_per_khead, nheads_kv))
          cute::Shape<int32_t, Int<kHeadDim>, cute::Shape<cute::Int<QheadPerKhead>, int32_t>>,
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

  // Declare the TMA operand types for Q, dO, K, V, dQaccum and dKVaccum.
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
      make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQdOdQTMA{}, StrideQdOdQTMA{}),
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
  // dQacc/dKVacc_use_TMA gate the entire smem-accumulator store infrastructure for the
  // inner-loop dX: an smem accum buffer + r2s copy + handshake with a store agent
  // (producer store warp or consumer scatter). The names are historical — "TMA" refers
  // to the default cp.reduce.async.bulk store from that smem buffer. They are false only
  // for hdim 256, where smem cannot fit the accum buffer and the kernel falls back to
  // direct per-thread atomicAdd from registers (see Slice_dQKV_Mma below); all scatter
  // paths require them to be true (hdim <= 128 in practice).
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
  // Accum buffers are sized for the Store layout (the padded scatter layout is slightly larger
  // than the swizzled TMA layout; they alias the same buffer, only one is live per build).
  using SmemdQacc_t = std::conditional_t<
      !dQacc_use_TMA,
      cute::array<ElementAccum, 0>,
      cute::array_aligned<ElementAccum, cute::max(cute::cosize_v<SmemLayoutdQaccumTMA>, cute::cosize_v<SmemLayoutdQaccumStore>)>>;
  using SmemdKVacc_t = std::conditional_t<
      DkvaccBypassSmem || !dKVacc_use_TMA,
      cute::array<ElementAccum, 0>,
      cute::array_aligned<ElementAccum, cute::max(cute::cosize_v<SmemLayoutdKVaccumTMA>, cute::cosize_v<SmemLayoutdKVaccumStore>)>>;
  using SmemP_t = std::conditional_t<Mma_dKV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP1>, SmemAlignmentP>>;

  // ─── Per-iteration token-index slots in smem (single source of truth for scatter paths) ───
  // kStages stage-indexed slots, 1:1 with the inner-tensor pipeline buffers (pipeline_q on
  // LoopQ, pipeline_k on LoopK). Every access is protected by an existing synchronization:
  //  - loader writes slot PipelineState::index() right after producer_acquire (stage held);
  //  - loader self-reads (dO/dPsum on LoopQ, V on LoopK) are same-warp program-ordered
  //    before any write that could reuse the slot;
  //  - consumer scatter stores (!InnerDxStoreInProducer) read their own read-state index()
  //    while still holding the stage (consumer_release comes after the scatter).
  // The producer store warps (InnerDxStoreInProducer) have NO stage protection — the consumer
  // releases the stage right after arriving dXFull, so the loader may rewrite the slot while
  // the store warp still reads it. They therefore read fixed STAGING areas instead, copied
  // from the current slot by consumer WG0 inside the dXEmpty→dXFull window (stage still held
  // there, and the store warp is blocked on dXFull). LoopK needs separate dV/dK staging:
  // store_dV's dVEmpty arrive lets the consumer's next-iteration dV r2s overwrite a shared
  // staging while store_dK would still be reading it. See .tmp/058 NOTES P7.
  static constexpr int kIdxStagingSlots = !InnerDxStoreInProducer ? 0 : (SwapBwdQKLoop ? 2 : 1);

  // Only the inner (scatter-side) tensor has token indices at all — the outer tensor is dense
  // TMA-loaded — so a single array suffices, sized by the inner tile:
  //   LoopQ (!SwapBwdQKLoop): inner = Q  (kBlockM rows) → read by the dQ scatter store.
  //   LoopK ( SwapBwdQKLoop): inner = KV (kBlockN rows) → read by the dKV scatter store.
  // Layout: [kStages stage slots][staging_dq | staging_dv, staging_dk].
  using SmemTokenIndices_t = std::conditional_t<InnerUseScatter, cute::array<int, kInnerScatterRows*(kStages + kIdxStagingSlots)>, cute::array<int, 0>>;

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
    SmemTokenIndices_t smem_token_indices;
  };

  // Empty placeholders for zero-sized SMEM fields (used with [[no_unique_address]])
  struct SmemLSE_Empty_ {};
  struct SmemDPsum_Empty_ {};
  struct SmemP_Empty_ {
    CUTLASS_HOST_DEVICE Element* data() {
      return nullptr;
    }
    CUTLASS_HOST_DEVICE const Element* data() const {
      return nullptr;
    }
  };

  static constexpr bool LseDpsumUnionEffective = LseDpsumUnionDKVacc && !DkvaccBypassSmem;
  using SmemLSE_t = std::conditional_t<LseDpsumUnionEffective, SmemLSE_Empty_, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentLSE>>;
  using SmemDPsum_t =
      std::conditional_t<LseDpsumUnionEffective, SmemDPsum_Empty_, cute::array_aligned<ElementAccum, cute::cosize_v<SmemLayoutLSE>, SmemAlignmentdPsum>>;
  // LoopK keeps P with kStages_dS stages (same as dS), unlike LoopQ which uses 1-stage P.
  using SmemP_LoopK_t = std::conditional_t<Mma_dKV_is_RS, SmemP_Empty_, cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP>>;

  // TMA 1D staging buffer: linear landing zone for SM90_BULK_COPY_G2S before rearrange
  // to swizzled smem layout. Only allocated when scatter path uses TMA 1D (Use_CpAsync_Inner).
  // LoopK: kBlockN rows of K/V. LoopQ: kBlockM rows of Q/dO.
  static constexpr int kTma1dStagingElems = Use_CpAsync_Inner ? kInnerScatterRows * kHeadDim : 0;
  using Tma1dStaging_t = cute::array_aligned<Element, kTma1dStagingElems, 128>;

  struct TensorStorageLoopK : cute::aligned_struct<maxSmemAlignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentV> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
    // dK and dV accumulators share SMEM via union: stores are serialized (dV r2s→TMA
    // completes before dK r2s starts), enforced by the swapped dVEmpty/dKEmpty barrier
    // protocol in store_dkv(). Saves 32 KB for larger kBlockM tiles.
    // When LseDpsumUnionDKVacc: the first bytes of smem_dkacc are also aliased as
    // smem_lse (512B) + smem_dpsum (512B) during outer-loop LSE/dPsum TMA loads.
    union {
      SmemdKVacc_t smem_dkacc;
      SmemdKVacc_t smem_dvacc;
    };
    SmemTokenIndices_t smem_token_indices;
    Tma1dStaging_t smem_tma1d_staging;
    // Zero-sized fields placed AFTER all data buffers so they fall in struct tail padding
    // (struct alignment from PdS swizzle is 1024B; core data sums to exactly N*1024).
    // [[no_unique_address]] on truly-empty types lets the compiler overlap them with
    // tail padding, avoiding a 1KB bump to the next alignment boundary.
    [[no_unique_address]] SmemLSE_t smem_lse;
    [[no_unique_address]] SmemDPsum_t smem_dpsum;
    [[no_unique_address]] SmemP_LoopK_t smem_p;
  };

  using TensorStorage = std::conditional_t<SwapBwdQKLoop, TensorStorageLoopK, TensorStorageLoopQ>;

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
    /* K, V use shape_KVdKdV; dK, dV use shape_dKdV (may differ when pool_count > 1) */
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
    int index_sparse_max_topk;
    /* dKV pool: reduce L2 contention by splitting dK/dV into pool_count slices */
    int pool_count;
    int pool_seqlen_k;
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
    TMA_QdO tma_load_Q, tma_load_dO;
    TMA_QdO_Packed tma_load_Q_packed, tma_load_dO_packed;
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
    /* sparse load (LoopQ: scatter Q/dO/dQ) */
    Element const* const ptr_Q;
    StrideQKV const stride_Q;
    Element const* const ptr_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQ;
    StrideQKV const stride_dQ;
    /* index_sparse */
    int const* const index_sparse_indices;
    int index_sparse_max_topk;
    /* dKV pool */
    int pool_count;
    int pool_seqlen_k;
  };

  // BlockSparse LoopK producer (used by load and store). token_indices stores raw IDs; stride multiplication is in the load/store lambdas.
  using BlockSparseLoopKBlockMeta = flash::BlockSparseBlockMeta</*IsProducer=*/true,
                                                                RangeMerge,
                                                                PackGQA,
                                                                QheadPerKhead,
                                                                NumRowsPerGroup,
                                                                GroupSize,
                                                                NumProducerThreads,
                                                                kBlockN,
                                                                InnerDirMaxToMin,
                                                                /*IsLoopQ=*/false>;

  // BlockSparse LoopK consumer (used by mma), no token_indices arrays
  using SparseMmaLoopKBlockMeta = flash::BlockSparseBlockMeta</*IsProducer=*/false,
                                                              RangeMerge,
                                                              PackGQA,
                                                              QheadPerKhead,
                                                              NumRowsPerGroup,
                                                              GroupSize,
                                                              NumProducerThreads,
                                                              kBlockN,
                                                              InnerDirMaxToMin,
                                                              /*IsLoopQ=*/false>;

  // BlockSparse LoopQ producer: scatter Q/dO, token_indices = Q positions
  using BlockSparseLoopQBlockMeta = flash::BlockSparseBlockMeta</*IsProducer=*/true,
                                                                RangeMerge,
                                                                PackGQA,
                                                                QheadPerKhead,
                                                                NumRowsPerGroup,
                                                                GroupSize,
                                                                NumProducerThreads,
                                                                kBlockM,
                                                                InnerDirMaxToMin,
                                                                /*IsLoopQ=*/true>;

  // BlockSparse LoopQ consumer: no token_indices arrays
  using SparseMmaLoopQBlockMeta = flash::BlockSparseBlockMeta</*IsProducer=*/false,
                                                              RangeMerge,
                                                              PackGQA,
                                                              QheadPerKhead,
                                                              NumRowsPerGroup,
                                                              GroupSize,
                                                              NumProducerThreads,
                                                              kBlockM,
                                                              InnerDirMaxToMin,
                                                              /*IsLoopQ=*/true>;

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
            make_shape(cute::Int<QheadPerKhead>{}, get<0>(args.shape_QdOdQ)), // (qhead_per_khead, seqlen)
            get<1>(args.shape_QdOdQ), // headdim
            get<2>(args.shape_KVdKdV) // nheads_kv
            ),
        cute::conditional_return<CatGQA>(
            make_shape(
                get<0>(args.shape_QdOdQ), // seqlen
                get<1>(args.shape_QdOdQ), // headdim
                make_shape(cute::Int<QheadPerKhead>{}, get<2>(args.shape_KVdKdV)) // (qhead_per_khead, nheads_kv)
                ),
            args.shape_QdOdQ));
    // Create stride for Q, dO and dQ
    auto const stride_Q = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), // (q_head_stride, row_stride)
            get<1>(args.stride_Q), // 1
            get<2>(args.stride_Q) * QheadPerKhead // qhead_per_khead * q_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_Q), // row_stride
                get<1>(args.stride_Q), // 1
                make_stride(get<2>(args.stride_Q), get<2>(args.stride_Q) * QheadPerKhead) // (q_head_stride, qhead_per_khead * q_head_stride)
                ),
            args.stride_Q));
    auto const stride_dO = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_dO), get<0>(args.stride_dO)), // (do_head_stride, row_stride)
            get<1>(args.stride_dO), // 1
            get<2>(args.stride_dO) * QheadPerKhead // qhead_per_khead * do_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_dO), // row_stride
                get<1>(args.stride_dO), // 1
                make_stride(get<2>(args.stride_dO), get<2>(args.stride_dO) * QheadPerKhead) // (do_head_stride, qhead_per_khead * do_head_stride)
                ),
            args.stride_dO));
    auto const stride_dQ = cute::conditional_return<PackGQA>(
        make_stride(
            make_stride(get<2>(args.stride_dQ), get<0>(args.stride_dQ)), // (dq_head_stride, row_stride)
            get<1>(args.stride_dQ), // 1
            get<2>(args.stride_dQ) * QheadPerKhead // qhead_per_khead * dq_head_stride
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                get<0>(args.stride_dQ), // row_stride
                get<1>(args.stride_dQ), // 1
                make_stride(get<2>(args.stride_dQ), get<2>(args.stride_dQ) * QheadPerKhead) // (dq_head_stride, qhead_per_khead * dq_head_stride)
                ),
            args.stride_dQ));

    // Create TMA for loading Q and dO, and for adding to dQ.
    // Non-packed TMA: uses flat shape/stride via make_tma_copy_A_sm90.
    Tensor mQ_flat = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_QdOdQ, args.stride_Q);
    TMA_QdO tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mQ_flat, take<0, 2>(SmemLayoutQ{}), TileShape_MNK{}, ClusterShape{});
    Tensor mdO_flat = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_QdOdQ, args.stride_dO);
    TMA_QdO tma_load_dO = make_tma_copy_A_sm90(GmemTiledCopyQdO{}, mdO_flat, take<0, 2>(SmemLayoutdO{}), TileShape_MNK{}, ClusterShape{});
    // Packed TMA: uses nested shape/stride via make_tma_copy (not _A_sm90)
    // to avoid identity-compose issues with hierarchical shapes.
    auto mQ_packed = [&]() {
      if constexpr (!PackGQA && !CatGQA) {
        return mQ_flat;
      } else {
        return make_tensor(make_gmem_ptr(args.ptr_Q), make_layout(shape_QdOdQ, stride_Q));
      }
    }();
    TMA_QdO_Packed tma_load_Q_packed = make_tma_copy(GmemTiledCopyQdO{}, mQ_packed, take<0, 2>(SmemLayoutQ{}), select<0, 2>(TileShape_MNK{}), size<1>(ClusterShape{}));
    auto mdO_packed = [&]() {
      if constexpr (!PackGQA && !CatGQA) {
        return mdO_flat;
      } else {
        return make_tensor(make_gmem_ptr(args.ptr_dO), make_layout(shape_QdOdQ, stride_dO));
      }
    }();
    TMA_QdO_Packed tma_load_dO_packed =
        make_tma_copy(GmemTiledCopyQdO{}, mdO_packed, take<0, 2>(SmemLayoutdO{}), select<0, 2>(TileShape_MNK{}), size<1>(ClusterShape{}));
    // dQ TMA (add/store, not load) uses nested shape directly
    Tensor mdQ = make_tensor(make_gmem_ptr(args.ptr_dQ), make_layout(shape_QdOdQ, stride_dQ));
    TMA_add_dQ tma_add_dQ = make_tma_copy(GmemTiledCopydQaccum{}, mdQ, SmemLayoutdQaccumTMA{}, TileShape_dQaccum{}, _1{});

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
    // dK/dV TMA use shape_dKdV (= pool_count * seqlen_k when pool > 1)
    Tensor mdK = make_tensor(make_gmem_ptr(args.ptr_dK), make_layout(args.shape_dKdV, args.stride_dK));
    TMA_add_dKV tma_add_dK = make_tma_copy(GmemTiledCopydKVaccum{}, mdK, SmemLayoutdKVaccumTMA{}, TileShape_dKVaccum{}, _1{});
    Tensor mdV = make_tensor(make_gmem_ptr(args.ptr_dV), make_layout(args.shape_dKdV, args.stride_dV));
    TMA_add_dKV tma_add_dV = make_tma_copy(GmemTiledCopydKVaccum{}, mdV, SmemLayoutdKVaccumTMA{}, TileShape_dKVaccum{}, _1{});

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
            make_shape(cute::Int<QheadPerKhead>{}, get<1>(args.shape_LSEdPsum)), // (qhead_per_khead, seqlen_q)
            get<2>(args.shape_KVdKdV) // nheads_kv
            ),
        cute::conditional_return<CatGQA>(
            make_shape(
                _4{},
                get<1>(args.shape_LSEdPsum), // seqlen_q
                make_shape(cute::Int<QheadPerKhead>{}, get<2>(args.shape_KVdKdV)) // (qhead_per_khead, nheads_kv)
                ),
            args.shape_LSEdPsum));
    // Create stride for LSE and dPsum
    auto const stride_LSE = cute::conditional_return<PackGQA>(
        make_stride(
            _1{},
            make_stride(get<2>(args.stride_LSE), get<1>(args.stride_LSE)), // (head_stride, 4)
            get<2>(args.stride_LSE) * QheadPerKhead // (qhead_per_khead * head_stride)
            ),
        cute::conditional_return<CatGQA>(
            make_stride(
                _1{},
                get<1>(args.stride_LSE), // 4
                make_stride(get<2>(args.stride_LSE), get<2>(args.stride_LSE) * QheadPerKhead) // (head_stride, qhead_per_khead * head_stride)
                ),
            args.stride_LSE));
    auto const stride_dPsum = cute::conditional_return<PackGQA>(
        make_stride(
            _1{},
            make_stride(get<2>(args.stride_dPsum), _4{}), // (head_stride, 4)
            get<2>(args.stride_dPsum) * QheadPerKhead),
        cute::conditional_return<CatGQA>(
            make_stride(
                _1{},
                get<1>(args.stride_dPsum), // 4
                make_stride(get<2>(args.stride_dPsum), get<2>(args.stride_dPsum) * QheadPerKhead) // (head_stride, qhead_per_khead * head_stride)
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
        tma_load_Q_packed,
        tma_load_dO_packed,
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
        /*n_block_max_num=*/!SwapBwdQKLoop ? cute::ceil_div(get<0>(args.shape_KVdKdV), kBlockN) : cute::ceil_div(get<0>(args.shape_QdOdQ), kBlockM),
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
        /*index_sparse_max_topk=*/args.index_sparse_max_topk,
        /*pool_count=*/args.pool_count,
        /*pool_seqlen_k=*/args.pool_seqlen_k};
  }

  // BlockMeta type alias — definition lives in block_meta.h
  // InnerLoopQ mapping:
  //   SwapBwdQKLoop=true  → inner loop over n_block (LoopK) → InnerLoopQ=false
  //   SwapBwdQKLoop=false → inner loop over m_block (LoopQ) → InnerLoopQ=true
  // So: InnerLoopQ = !SwapBwdQKLoop
  template <bool IsProducer>
  using BlockMeta = flash::DenseBlockMeta<IsProducer, /*InnerLoopQ=*/!SwapBwdQKLoop, RangeMerge, /*FlattenGQA=*/FlattenGQA, QheadPerKhead, SeqlenInfo_t, BlockMN_t>;

  // IndexSparse LoopK: outer=Q token, inner=K from forward topk indices
  template <bool IsProducer>
  using IndexSparseLoadBlockMeta = flash::IndexSparseBlockMeta<
      IsProducer,
      RangeMerge,
      PackGQA,
      QheadPerKhead,
      NumRowsPerGroup,
      NumProducerThreads,
      GroupSize,
      kBlockN,
      InnerDirMaxToMin,
      KBlockSize,
      /*IsLoopQ=*/false>;

  // IndexSparse LoopQ: outer=K block, inner=Q from inv_indices
  template <bool IsProducer>
  using IndexSparseInvLoadBlockMeta = flash::IndexSparseBlockMeta<
      IsProducer,
      /*RangeMerge=*/false,
      PackGQA,
      QheadPerKhead,
      NumRowsPerGroup,
      NumProducerThreads,
      GroupSize,
      kBlockM,
      InnerDirMaxToMin,
      KBlockSize,
      /*IsLoopQ=*/true>;

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (!PackGQA && !CatGQA) {
      cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
    } else {
      cute::prefetch_tma_descriptor(params.tma_load_Q_packed.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_dO_packed.get_tma_descriptor());
    }
    // K/V TMA descriptors needed for dense and scatter-TMA paths (Use_TMA_Inner).
    // Only cp.async-only scatter (IndexSparse etc) skips the prefetch.
    if constexpr (!InnerUseScatter || Use_TMA_Inner) {
      cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }
  }

  // ─── Unified scatter dX row store: per-row bulk reduce-add vs per-4B scalar atomicAdd ───
  // All scatter store call sites (LoopK consumer dV/dK, LoopK producer store_dV/store_dK,
  // LoopQ consumer dQ, LoopQ producer store_dQ) funnel into this helper.
  // Rows [row_offset, row_offset + kRows) of s_acc are reduce-added into gmem rows
  // token_idx = smem_token_indices[row] (caller pre-offsets the slot base).
  //
  // SparseInnerDxReduceUseTma=true: issuer thread t owns rows {t, t+kNumThreads, ...} and issues one
  // cp.reduce.async.bulk per row (kHeadDim*4B linear, no L2 sector waste); each issuer commits
  // and waits its own bulk group before returning, so smem is reusable on return. Threads with
  // thread_idx >= kRows issue nothing. Requires the row-contiguous *Store smem layout.
  //
  // SparseInnerDxReduceUseTma=false: original scalar geometry — GroupSize threads per row, each
  // covering kStoreVecWidth floats per store tile via per-4B atomicAdd.
  //
  // kRowPackScale > 1 (LoopQ + PackGQA dQ store): smem_token_indices hold PACKED rows
  // p = token * G + g; the gmem row decomposes as token*stride_row + g*stride_head
  // (q heads within a token are stride_head apart, not row-contiguous when nheads_kv > 1).
  template <int kRows, int kNumThreads, int kRowPackScale, typename SmemAccT>
  CUTLASS_DEVICE static void scatter_reduce_store_rows(
      SmemAccT const& s_acc, // Store-layout accum view, indexed at (row_offset + row, col)
      int const* smem_token_indices, // kRows ints, indexed at [row]
      ElementAccum* gmem_base, // dX base pointer for this head
      int const stride_row, // gmem row stride in elements
      int const thread_idx, // flat issuer index in [0, kNumThreads)
      int const row_offset = 0,
      int const stride_head = 0) { // gmem head stride in elements (kRowPackScale > 1 only)
    auto gmem_row = [&](int token_idx) -> ElementAccum* {
      if constexpr (kRowPackScale != 1) {
        return gmem_base + static_cast<int64_t>(token_idx / kRowPackScale) * stride_row + static_cast<int64_t>(token_idx % kRowPackScale) * stride_head;
      } else {
        return gmem_base + static_cast<int64_t>(token_idx) * stride_row;
      }
    };
    if constexpr (SparseInnerDxReduceUseTma) {
      static constexpr int32_t kRowBytes = kHeadDim * sizeof(ElementAccum);
      bool issued = false;
      for (int row = thread_idx; row < kRows; row += kNumThreads) {
        ElementAccum* const dst = gmem_row(smem_token_indices[row]);
        cute::SM90_BULK_REDUCE_ADD::copy(&s_acc(row_offset + row, _0{}), dst, kRowBytes);
        issued = true;
      }
      if (issued) {
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
      }
    } else {
      static_assert(kNumThreads % GroupSize == 0);
      static constexpr int kNumGroups_ = kNumThreads / GroupSize;
      static_assert(kRows % kNumGroups_ == 0, "scalar scatter store requires kRows divisible by thread groups");
      static constexpr int kRowsPerGroup_ = kRows / kNumGroups_;
      int const group_idx = thread_idx / GroupSize;
      int const idx_in_group = thread_idx % GroupSize;
      CUTE_UNROLL
      for (int local_row = 0; local_row < kRowsPerGroup_; ++local_row) {
        int const row = group_idx * kRowsPerGroup_ + local_row;
        ElementAccum* const dst = gmem_row(smem_token_indices[row]);
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < kNumStoreTiles; ++tile_idx) {
          int const col_base = idx_in_group * kStoreVecWidth + tile_idx * GroupSize * kStoreVecWidth;
          CUTE_UNROLL
          for (int v = 0; v < kStoreVecWidth; ++v) {
            atomicAdd(dst + col_base + v, s_acc(row_offset + row, col_base + v));
          }
        }
      }
    }
  }

  // Perform a Producer Prologue/Mainloop -- TMA Load for K,V, with pipelining multi-stage TMA load for Q,dO,LSE,dPsum
  // k for outer-loop and q for inner-loop
  // When BlockSparse (LoopQ): Q/dO/LSE/dPsum are scatter-loaded via cp.async, K/V are still TMA.
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
    static_assert(!SwapBwdQKLoop, "load_with_loop_q() must be called when SwapBwdQKLoop is false");
    // The BlockSparse scatter loader has no per-q-head (bidh_kv_cat) loop, so CatGQA cannot
    // be expressed on this path yet. PackGQA is supported by walking q_ranges in packed-row
    // space instead (see BlockSparseBlockMeta::kScatterScale).
    static_assert(!(BlockSparse && CatGQA), "BlockSparse LoopQ does not support CatGQA");

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_block;
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

    // For PackGQA, offset needs to be multiplied by QheadPerKhead
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * QheadPerKhead;

    // Prepare for TMA loads
    auto const mQdOdQLSEdPsum_coord = make_coord(_, _, cute::conditional_return<CatGQA>(make_coord(_, bidh), bidh));
    auto const gQdOdQ_coord = cute::conditional_return<CatGQA>(make_coord(_, _0{}, _), make_coord(_, _0{}));
    auto const gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
    // get_tma_tensor + local_tile: use packed TMA for PackGQA/CatGQA, non-packed otherwise
    auto mQ = [&]() {
      if constexpr (PackGQA || CatGQA) {
        return params.tma_load_Q_packed.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
      } else {
        return params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
      }
    }();
    auto mdO = [&]() {
      if constexpr (PackGQA || CatGQA) {
        return params.tma_load_dO_packed.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
      } else {
        return params.tma_load_dO.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
      }
    }();
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

    auto block_tma_Q = [&]() {
      if constexpr (PackGQA || CatGQA) {
        return params.tma_load_Q_packed.get_slice(cluster_block_id_qdo);
      } else {
        return params.tma_load_Q.get_slice(cluster_block_id_qdo);
      }
    }();
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
    auto block_tma_dO = [&]() {
      if constexpr (PackGQA || CatGQA) {
        return params.tma_load_dO_packed.get_slice(cluster_block_id_qdo);
      } else {
        return params.tma_load_dO.get_slice(cluster_block_id_qdo);
      }
    }();
    Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
    Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));

    auto rebind_Q_tiles = [&](SeqlenInfo_t const& si) {
      if constexpr (!RangeMerge) {
        return;
      }
      offset_q = !PackGQA ? si.offset_q : si.offset_q * QheadPerKhead;
      auto const qdo_off = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
      gQ = local_tile(domain_offset(qdo_off, mQ), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
      gdO = local_tile(domain_offset(qdo_off, mdO), select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
      tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
      tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
      auto const lse_off = cute::conditional_return<CatGQA>(make_coord(_0{}, offset_q, _0{}), make_coord(_0{}, offset_q));
      gLSE = local_tile(cute::domain_offset(lse_off, mLSE), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
      gdPsum = local_tile(cute::domain_offset(lse_off, mdPsum), make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
    };

    // Use_TMA_Inner && InnerUseScatter: absolute-coordinate TMA tensors (no domain_offset by offset_q)
    // for issuing TMA loads at runtime-computed token positions.
    auto tQgQ_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        auto gQ_abs = local_tile(mQ, select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
        return group_modes<0, 3>(block_tma_Q.partition_S(gQ_abs));
      } else {
        return cute::make_tuple(); // DCE placeholder
      }
    }();
    auto tdOgdO_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        auto gdO_abs = local_tile(mdO, select<0, 2>(TileShape_MNK{}), gQdOdQ_coord);
        return group_modes<0, 3>(block_tma_dO.partition_S(gdO_abs));
      } else {
        return cute::make_tuple();
      }
    }();
    auto gLSE_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        return local_tile(mLSE, make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
      } else {
        return cute::make_tuple();
      }
    }();
    auto gdPsum_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        return local_tile(mdPsum, make_shape(_4{}, Int<kBlockM>{}), gLSEdPsum_coord);
      } else {
        return cute::make_tuple();
      }
    }();

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
    //    BarrierManager::sync<NumMmaThreads + cutlass::NumThreadsPerWarp>(BwdNamedBarriers::KVEmpty);

    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    int const lane_predicate = cute::elect_one_sync();

    // ─── BlockSparse LoopQ scatter infra (DCE'd on dense path) ───
    using CpAsyncCg = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, cute::uint128_t>;
    CpAsyncCg const cp_async_cg{};
    int const thread_idx = threadIdx.x % NumBlockSparseThreads;
    int const idx_in_group = thread_idx % GroupSize;
    int const group_idx = thread_idx / GroupSize;
    // PackGQA: the token-index slots hold PACKED rows p = token * G + g (g = q-head
    // within the kv group, G = QheadPerKhead); bidh is then the kv head index, so head
    // bases are scaled by G. Q/dO/dQ rows decompose as token*row_stride + g*head_stride
    // (heads within a token are head_stride apart, NOT row-contiguous when nheads_kv > 1).
    static constexpr int kQPackScale = PackGQA ? QheadPerKhead : 1;
    int64_t const stride_q_row = get<0>(params.stride_Q);
    int64_t const stride_do_row = get<0>(params.stride_dO);
    int64_t const stride_q_head = get<2>(params.stride_Q);
    int64_t const stride_do_head = get<2>(params.stride_dO);
    Element const* const ptr_gQ_base = params.ptr_Q + bidh * kQPackScale * stride_q_head + idx_in_group * 8;
    Element const* const ptr_gdO_base = params.ptr_dO + bidh * kQPackScale * stride_do_head + idx_in_group * 8;
    // Decompose a packed row into a gmem element offset (plain token * row_stride when !PackGQA)
    auto packed_row_offset = [&](int p, int64_t stride_token, int64_t stride_head) -> int64_t {
      if constexpr (PackGQA) {
        return (p / kQPackScale) * stride_token + (p % kQPackScale) * stride_head;
      } else {
        return p * stride_token;
      }
    };
    // LSE/dPsum per-token row is always 4 floats. PackGQA: params.stride_LSE/stride_dPsum
    // are the packed nested strides (1, (head_stride, 4), G*head_stride), so get<2> already
    // contains the G factor for the kv-head base, and get<1,0> is the raw per-head stride.
    float const* const ptr_gLSE_base = params.ptr_LSE_log2 + bidh * get<2>(params.stride_LSE);
    float const* const ptr_gdPsum_base = params.ptr_dPsum + bidh * get<2>(params.stride_dPsum);
    auto lse_row_offset = [&](int p) -> int64_t {
      if constexpr (PackGQA) {
        return (p % kQPackScale) * get<1, 0>(params.stride_LSE) + (p / kQPackScale) * 4;
      } else {
        return p * 4;
      }
    };
    auto dpsum_row_offset = [&](int p) -> int64_t {
      if constexpr (PackGQA) {
        return (p % kQPackScale) * get<1, 0>(params.stride_dPsum) + (p / kQPackScale) * 4;
      } else {
        return p * 4;
      }
    };

    // Define lambda funcs to load Q,dO,K,V,LSE,dPsum
    // Each lambda is self-contained: lane_predicate guard + acquire + TMA copy (dense)
    // or multi-thread scatter cp.async (BlockSparse).
    // Q and dO share the same pipe slot when Q_dO_same_stages=true, so pipe advance
    // happens in load_dO_dPsum (the second of each pair) to keep the slot index in sync.
    auto load_Q_LSE = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        // All elected threads (lane_predicate) participate in pipe state advance,
        // but only thread 0 has the correct block_meta cursor and issues TMA.
        // (Each thread's block_meta.cur_range_inner_idx is group-offset-shifted
        // during construction; only thread 0's group_idx=0 → correct first row.)
        if (!lane_predicate)
          return;
        pipeline_q.producer_acquire(smem_pipe_write_q);
        if (thread_idx == 0) {
          int const stage = smem_pipe_write_q.index();
          int const packed_first_row = block_meta.get_packed_first_row();
          shared_storage.tensors.mainloop.smem_token_indices[stage * kBlockM] = packed_first_row;
          int const m_block_abs = packed_first_row / kBlockM;
          auto tma_Q_desc = params.tma_load_Q_packed.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          copy(tma_Q_desc, tQgQ_abs(_, m_block_abs), tQsQ(_, stage));
          copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE_abs(_, _, m_block_abs), sLSE(_, _, stage));
        }
      } else if constexpr (InnerUseScatter) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        int const stage = smem_pipe_write_q.index();
        int* const idx_slot = &shared_storage.tensors.mainloop.smem_token_indices[stage * kBlockM];
        block_meta.fill_token_indices(idx_slot, idx_in_group, group_idx);
        __syncwarp();
        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int smem_row = group_idx * NumRowsPerGroup + local_row;
          int64_t token_offset = packed_row_offset(idx_slot[smem_row], stride_q_row, stride_q_head);
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < NumCpAsyncTilesPerRow; ++tile_idx) {
            Element* dst_ptr = &sQ(smem_row, idx_in_group * 8 + tile_idx * 64, stage);
            auto gQ_src = make_tensor(make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_gQ_base + token_offset + tile_idx * 64)), Layout<_1>{});
            auto sQ_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(dst_ptr)), Layout<_1>{});
            cute::copy(cp_async_cg, gQ_src, sQ_dst);
          }
        }
        for (int i = idx_in_group; i < NumRowsPerGroup; i += GroupSize) {
          float* lse_dst = &sLSE(_0{}, group_idx * NumRowsPerGroup + i, stage);
          auto gLSE_src = make_tensor(
              make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_gLSE_base + lse_row_offset(idx_slot[group_idx * NumRowsPerGroup + i]))), Layout<_1>{});
          auto sLSE_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(lse_dst)), Layout<_1>{});
          cute::copy(cp_async_cg, gLSE_src, sLSE_dst);
        }
        pipeline_q.producer_commit(smem_pipe_write_q, cutlass::arch::cpasync_barrier_arrive);
      } else {
        if (!lane_predicate)
          return;
        pipeline_q.producer_acquire(smem_pipe_write_q);
        auto tma_Q_desc = [&]() {
          if constexpr (PackGQA || CatGQA) {
            return params.tma_load_Q_packed.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          } else {
            return params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          }
        }();
        if constexpr (CatGQA) {
          copy(tma_Q_desc, tQgQ(_, m_block, bidh_kv_cat), tQsQ(_, smem_pipe_write_q.index()));
          copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE(_, _, m_block, bidh_kv_cat), sLSE(_, _, smem_pipe_write_q.index()));
        } else {
          copy(tma_Q_desc, tQgQ(_, m_block), tQsQ(_, smem_pipe_write_q.index()));
          copy(bulk_copy.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q)), gLSE(_, _, m_block), sLSE(_, _, smem_pipe_write_q.index()));
        }
      }
    };

    auto load_dO_dPsum = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        if (!lane_predicate)
          return;
        PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write_q, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        if (thread_idx == 0) {
          int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_write_q.index() * kBlockM];
          int const m_block_abs = packed_first_row / kBlockM;
          auto tma_dO_desc = params.tma_load_dO_packed.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          copy(tma_dO_desc, tdOgdO_abs(_, m_block_abs), tdOsdO(_, smem_pipe_write_do_cur.index()));
          copy(bulk_copy.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur)), gdPsum_abs(_, _, m_block_abs), sdPsum(_, _, smem_pipe_write_do_cur.index()));
        }
        if constexpr (!Q_dO_same_stages) {
          ++smem_pipe_write_do;
        }
        ++smem_pipe_write_q;
      } else if constexpr (InnerUseScatter) {
        PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write_q, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        int const* const idx_slot = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_write_q.index() * kBlockM];
        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int smem_row = group_idx * NumRowsPerGroup + local_row;
          int64_t token_offset = packed_row_offset(idx_slot[smem_row], stride_do_row, stride_do_head);
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < NumCpAsyncTilesPerRow; ++tile_idx) {
            Element* dst_ptr = &sdO(smem_row, idx_in_group * 8 + tile_idx * 64, smem_pipe_write_do_cur.index());
            auto gdO_src = make_tensor(make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_gdO_base + token_offset + tile_idx * 64)), Layout<_1>{});
            auto sdO_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(dst_ptr)), Layout<_1>{});
            cute::copy(cp_async_cg, gdO_src, sdO_dst);
          }
        }
        for (int i = idx_in_group; i < NumRowsPerGroup; i += GroupSize) {
          float* dpsum_dst = &sdPsum(_0{}, group_idx * NumRowsPerGroup + i, smem_pipe_write_do_cur.index());
          auto gdPsum_src = make_tensor(
              make_gmem_ptr(reinterpret_cast<cute::uint128_t const*>(ptr_gdPsum_base + dpsum_row_offset(idx_slot[group_idx * NumRowsPerGroup + i]))), Layout<_1>{});
          auto sdPsum_dst = make_tensor(make_smem_ptr(reinterpret_cast<cute::uint128_t*>(dpsum_dst)), Layout<_1>{});
          cute::copy(cp_async_cg, gdPsum_src, sdPsum_dst);
        }
        pipeline_do.producer_commit(smem_pipe_write_do_cur, cutlass::arch::cpasync_barrier_arrive);
        if constexpr (!Q_dO_same_stages) {
          ++smem_pipe_write_do;
        }
        ++smem_pipe_write_q;
      } else {
        if (!lane_predicate)
          return;
        PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write_q, smem_pipe_write_do);
        pipeline_do.producer_acquire(smem_pipe_write_do_cur);
        auto tma_dO_desc = [&]() {
          if constexpr (PackGQA || CatGQA) {
            return params.tma_load_dO_packed.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          } else {
            return params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo, TMA::CacheHintSm90::EVICT_LAST);
          }
        }();
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
        if constexpr (!Q_dO_same_stages) {
          ++smem_pipe_write_do;
        }
        ++smem_pipe_write_q;
      }
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
      if constexpr (InnerUseScatter) {
        // Scatter (BlockSparse / IndexSparse LoopQ): one block per call, block_meta drives iteration
        load_Q_LSE();
        load_dO_dPsum();
      } else {
        CUTLASS_PRAGMA_NO_UNROLL
        for (bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, QheadPerKhead); ++bidh_kv_cat) {
          rebind_Q_tiles(block_meta.seqlen_info);
          m_block = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_max);
          flash::iterate_range < kInnerDir,
              kHeadDim<256 ? 2 : 1>(
                  m_block,
                  block_meta.inner_block_min,
                  block_meta.inner_block_max,
                  [&] {
                    load_Q_LSE();
                    load_dO_dPsum();
                  });
        }
      }
    };

    // ─── Unified control flow ───
    // K/V are loaded once (fixed n_block), Q/dO are streamed across merged batches.
    if (block_meta.skip_to_first_valid())
      return false;

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
    static_assert(SwapBwdQKLoop, "load_with_loop_k() must be called when SwapBwdQKLoop is true");
    static_assert(!CatGQA, "load_with_loop_k() is not compatible with CatGQA");

    // BlockMeta: fixed per function call
    int const m_block = block_meta.outer_block;
    int const bidh = block_meta.bidh;
    int const bidh_kv = block_meta.bidh_kv;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()), SmemLayoutdO{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
    ElementAccum* lse_smem_ptr;
    ElementAccum* dpsum_smem_ptr;
    if constexpr (LseDpsumUnionEffective) {
      lse_smem_ptr = reinterpret_cast<ElementAccum*>(shared_storage.tensors.mainloop.smem_dkacc.data());
      dpsum_smem_ptr = lse_smem_ptr + cute::cosize_v<SmemLayoutLSE>;
    } else {
      lse_smem_ptr = shared_storage.tensors.mainloop.smem_lse.data();
      dpsum_smem_ptr = shared_storage.tensors.mainloop.smem_dpsum.data();
    }
    Tensor sLSE = make_tensor(make_smem_ptr(lse_smem_ptr), SmemLayoutLSE{});
    Tensor sdPsum = make_tensor(make_smem_ptr(dpsum_smem_ptr), SmemLayoutLSE{});

    // prepare for TMA multicast meta
    auto [mcast_mask_kv, cluster_block_id_kv] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyKV, /*RowwiseMask=*/true>();

    // Prepare the TMA loads: use packed TMA for PackGQA, non-packed otherwise
    auto mQ = [&]() {
      if constexpr (PackGQA) {
        return params.tma_load_Q_packed.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
      } else {
        return params.tma_load_Q.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
      }
    }();
    auto mdO = [&]() {
      if constexpr (PackGQA) {
        return params.tma_load_dO_packed.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
      } else {
        return params.tma_load_dO.get_tma_tensor(params.shape_QdOdQ)(_, _, bidh);
      }
    }();
    Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_KVdKdV)(_, _, bidh_kv); // (seqlen_kv, head_dim)
    // For PackGQA, LSE/dPsum use packed shape/stride to correctly read data from multiple Q heads
    auto mLSE = make_tensor(make_gmem_ptr(params.ptr_LSE_log2), params.shape_LSEdPsum, params.stride_LSE)(_, _, bidh); // (4, seqlen_q)
    auto mdPsum = make_tensor(make_gmem_ptr(params.ptr_dPsum), params.shape_LSEdPsum, params.stride_dPsum)(_, _, bidh); // (4, seqlen_q)

    // For PackGQA, offset needs to be multiplied by QheadPerKhead
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * QheadPerKhead;
    Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor gK = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)
    Tensor gV = local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)

    // For PackGQA, LSE/dPsum also use packed offset to match Q/dO's packed access pattern
    auto bulk_copy = Copy_Traits<SM90_BULK_COPY_AUTO>{};
    Tensor gLSE = local_tile(cute::domain_offset(make_coord(_0{}, offset_q), mLSE), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, m_block)); // (4, M)
    Tensor gdPsum = local_tile(cute::domain_offset(make_coord(_0{}, offset_q), mdPsum), make_shape(_4{}, Int<kBlockM>{}), make_coord(_0{}, m_block)); // (4, M)

    auto block_tma_Q = [&]() {
      if constexpr (PackGQA) {
        return params.tma_load_Q_packed.get_slice(_0{});
      } else {
        return params.tma_load_Q.get_slice(_0{});
      }
    }();
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)

    auto block_tma_dO = [&]() {
      if constexpr (PackGQA) {
        return params.tma_load_dO_packed.get_slice(_0{});
      } else {
        return params.tma_load_dO.get_slice(_0{});
      }
    }();
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
    int const thread_idx = threadIdx.x % NumBlockSparseThreads;
    int const idx_in_group = thread_idx % GroupSize;
    int const group_idx = thread_idx / GroupSize;
    int const stride_kv_row = get<0>(params.stride_K);
    int const stride_kv_row_v = get<0>(params.stride_V);
    Element const* const ptr_gK_base = params.ptr_K + bidh_kv * get<2>(params.stride_K) + idx_in_group * 8;
    Element const* const ptr_gV_base = params.ptr_V + bidh_kv * get<2>(params.stride_V) + idx_in_group * 8;

    // TMA 1D base pointers (without idx_in_group offset — each row is a full contiguous load)
    Element const* const ptr_gK_base_tma1d = params.ptr_K + bidh_kv * get<2>(params.stride_K);
    Element const* const ptr_gV_base_tma1d = params.ptr_V + bidh_kv * get<2>(params.stride_V);
    int tma1d_phase = 0;

    // ─── Shared Q/dO/LSE/dPsum loading ───

    auto load_QdO_LSE_dPsum = [&]() {
      // Only warp 0's elected leader issues the QdO TMA (single-warp), see note above.
      if (!(warp_idx_in_warpgroup == 0 && lane_predicate))
        return;
      auto& barrier_QdO = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_QdO);
      shared_storage.pipelines.barrier_QdO.arrive_and_expect_tx(TmaTransactionBytesQ + TmaTransactionBytesdO + TmaTransactionBytesLSE + TmaTransactionBytesdPsum);
      if constexpr (PackGQA) {
        copy(params.tma_load_Q_packed.with(barrier_QdO, /*mcast_mask=*/0), tQgQ, tQsQ);
        copy(params.tma_load_dO_packed.with(barrier_QdO, /*mcast_mask=*/0), tdOgdO, tdOsdO);
      } else {
        copy(params.tma_load_Q.with(barrier_QdO, /*mcast_mask=*/0), tQgQ, tQsQ);
        copy(params.tma_load_dO.with(barrier_QdO, /*mcast_mask=*/0), tdOgdO, tdOsdO);
      }
      copy(bulk_copy.with(barrier_QdO), gLSE, sLSE);
      copy(bulk_copy.with(barrier_QdO), gdPsum, sdPsum);
    };

    // ─── TMA setup for K/V (dense and scatter-TMA paths) ───
    auto block_tma_K = params.tma_load_K.get_slice(cluster_block_id_kv);
    Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK)); // (TMA, k)
    Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK)); // (TMA, PIPE)

    auto block_tma_V = params.tma_load_V.get_slice(cluster_block_id_kv);
    Tensor tVgV = group_modes<0, 3>(block_tma_V.partition_S(gV)); // (TMA, k)
    Tensor tVsV = group_modes<0, 3>(block_tma_V.partition_D(sV)); // (TMA, PIPE)

    // Use_TMA_Inner && InnerUseScatter (LoopK scatter TMA): absolute-coordinate
    // K/V tensors for issuing TMA at runtime-computed positions from block_meta.
    auto tKgK_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        auto gK_abs = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        return group_modes<0, 3>(block_tma_K.partition_S(gK_abs));
      } else {
        return cute::make_tuple();
      }
    }();
    auto tVgV_abs = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        auto gV_abs = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        return group_modes<0, 3>(block_tma_V.partition_S(gV_abs));
      } else {
        return cute::make_tuple();
      }
    }();

    // ─── Unified load_K / load_V: scatter vs TMA ───
    // When kStages_V != kStages, V pipeline's stage index differs from K's.
    // V needs K's stage to read smem_token_indices (populated by load_K).
    int last_k_write_stage = 0;

    auto load_K = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        // Scatter TMA: BlockSparse K tiles are contiguous within each k_range.
        // All elected threads participate in pipe state; only thread 0 issues TMA.
        if (!lane_predicate)
          return;
        pipeline_k.producer_acquire(smem_pipe_write_k);
        if (thread_idx == 0) {
          int const stage = smem_pipe_write_k.index();
          int const packed_first_row = block_meta.get_packed_first_row();
          // Fill contiguous token indices (needed by consumer's dKV scatter store)
          int* const idx_slot = &shared_storage.tensors.mainloop.smem_token_indices[stage * kBlockN];
          CUTE_UNROLL
          for (int r = 0; r < kBlockN; ++r) {
            idx_slot[r] = packed_first_row + r;
          }
          int const n_block_abs = packed_first_row / kBlockN;
          copy(
              params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
              tKgK_abs(_, n_block_abs),
              tKsK(_, stage));
        }
        last_k_write_stage = smem_pipe_write_k.index();
        ++smem_pipe_write_k;
      } else if constexpr (InnerUseScatter) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        int* const idx_slot = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_write_k.index() * kBlockN];
        block_meta.fill_token_indices(idx_slot, idx_in_group, group_idx);
        __syncwarp();

        // ─── TMA 1D bulk load path ───
        // Each of NumBlockSparseThreads threads issues 1 bulk copy for its own row.
        // thread_idx == group_idx * NumRowsPerGroup + idx_in_group (== thread_idx for 64/8/8).
        Element* staging = shared_storage.tensors.mainloop.smem_tma1d_staging.data();
        static constexpr int kRowBytes = kHeadDim * static_cast<int>(sizeof(Element));
        static constexpr int kTotalTxBytes = kBlockN * kRowBytes;
        auto* staging_mbar = &shared_storage.pipelines.tma1d_staging_mbar;

        if (thread_idx == 0) {
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(staging_mbar, kTotalTxBytes);
        }
        {
          int my_row = thread_idx;
          int token_offset = idx_slot[my_row] * stride_kv_row;
          void const* src = reinterpret_cast<void const*>(ptr_gK_base_tma1d + token_offset);
          void* dst = reinterpret_cast<void*>(staging + my_row * kHeadDim);
          cute::SM90_BULK_COPY_G2S::copy(src, staging_mbar, dst, kRowBytes);
        }
        cutlass::arch::ClusterTransactionBarrier::wait(staging_mbar, tma1d_phase);
        tma1d_phase ^= 1;

        // Phase 2: Rearrange from linear staging to swizzled SmemLayoutK
        static constexpr int kElemsPerU128 = 16 / static_cast<int>(sizeof(Element));
        static constexpr int kU128PerRow = kHeadDim / kElemsPerU128;
        int const stage = smem_pipe_write_k.index();
        for (int row = thread_idx; row < kBlockN; row += NumBlockSparseThreads) {
          CUTE_UNROLL
          for (int chunk = 0; chunk < kU128PerRow; ++chunk) {
            int col = chunk * kElemsPerU128;
            cute::uint128_t val = *reinterpret_cast<cute::uint128_t const*>(staging + row * kHeadDim + col);
            *reinterpret_cast<cute::uint128_t*>(&sK(row, col, stage)) = val;
          }
        }

        last_k_write_stage = smem_pipe_write_k.index();
        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;
      } else {
        if (!lane_predicate)
          return;
        Tensor gK_ = local_tile(domain_offset(make_coord(block_meta.seqlen_info.offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor tKgK_ = group_modes<0, 3>(block_tma_K.partition_S(gK_));
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(
            params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tKgK_(_, block_meta.inner_block_cur),
            tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
      }
    };

    auto load_V = [&]() {
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        if (!lane_predicate)
          return;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        if (thread_idx == 0) {
          int const v_stage = smem_pipe_write_v.index();
          // V reads token indices from K's stage slot (they may differ when kStages_V < kStages)
          int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[last_k_write_stage * kBlockN];
          int const n_block_abs = packed_first_row / kBlockN;
          copy(
              params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
              tVgV_abs(_, n_block_abs),
              tVsV(_, v_stage));
        }
        ++smem_pipe_write_v;
      } else if constexpr (InnerUseScatter) {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        int const* const idx_slot = &shared_storage.tensors.mainloop.smem_token_indices[last_k_write_stage * kBlockN];

        // ─── TMA 1D bulk load path for V ───
        Element* staging = shared_storage.tensors.mainloop.smem_tma1d_staging.data();
        static constexpr int kRowBytes = kHeadDim * static_cast<int>(sizeof(Element));
        static constexpr int kTotalTxBytes = kBlockN * kRowBytes;
        auto* staging_mbar = &shared_storage.pipelines.tma1d_staging_mbar;

        if (thread_idx == 0) {
          cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(staging_mbar, kTotalTxBytes);
        }
        {
          int my_row = thread_idx;
          int token_offset = idx_slot[my_row] * stride_kv_row_v;
          void const* src = reinterpret_cast<void const*>(ptr_gV_base_tma1d + token_offset);
          void* dst = reinterpret_cast<void*>(staging + my_row * kHeadDim);
          cute::SM90_BULK_COPY_G2S::copy(src, staging_mbar, dst, kRowBytes);
        }
        cutlass::arch::ClusterTransactionBarrier::wait(staging_mbar, tma1d_phase);
        tma1d_phase ^= 1;

        // Phase 2: Rearrange from linear staging to swizzled SmemLayoutV
        static constexpr int kElemsPerU128 = 16 / static_cast<int>(sizeof(Element));
        static constexpr int kU128PerRow = kHeadDim / kElemsPerU128;
        int const v_stage = smem_pipe_write_v.index();
        for (int row = thread_idx; row < kBlockN; row += NumBlockSparseThreads) {
          CUTE_UNROLL
          for (int chunk = 0; chunk < kU128PerRow; ++chunk) {
            int col = chunk * kElemsPerU128;
            cute::uint128_t val = *reinterpret_cast<cute::uint128_t const*>(staging + row * kHeadDim + col);
            *reinterpret_cast<cute::uint128_t*>(&sV(row, col, v_stage)) = val;
          }
        }

        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      } else {
        if (!lane_predicate)
          return;
        Tensor gV_ = local_tile(domain_offset(make_coord(block_meta.seqlen_info.offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
        Tensor tVgV_ = group_modes<0, 3>(block_tma_V.partition_S(gV_));
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(
            params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tVgV_(_, block_meta.inner_block_cur),
            tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    };

    auto load_body = [&]() {
      if constexpr (InnerUseScatter) {
        load_K();
        load_V();
      } else {
        flash::iterate_range < kInnerDir,
            kHeadDim<256 ? 2 : 1>(
                block_meta.inner_block_cur,
                block_meta.inner_block_min,
                block_meta.inner_block_max,
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
    static_assert(!SwapBwdQKLoop, "load_tail_with_loop_q() must be called when SwapBwdQKLoop is false");

    // PipelineAsync (kCpAsync): all threads must arrive.
    // PipelineTmaAsync (kTmaDense/kTma2D): single-thread arrive suffices.
    if (!Use_TMA_Inner || cute::elect_one_sync()) {
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
    static_assert(SwapBwdQKLoop, "load_tail_with_loop_k() must be called when SwapBwdQKLoop is true");

    // PipelineAsync (kCpAsync): all threads must arrive.
    // PipelineTmaAsync (kTma): single-thread arrive suffices.
    if (!Use_TMA_Inner || cute::elect_one_sync()) {
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
    static_assert(!SwapBwdQKLoop, "store_dq() must be called when SwapBwdQKLoop is false");

    // !InnerDxStoreInProducer: dQ store is handled by the MMA consumer threads, not by producer.
    if constexpr (!InnerDxStoreInProducer) {
      return;
    }

    if constexpr (!dQacc_use_TMA) {
      return;
    }

    static constexpr int kBlockM = CollectiveMainloopBwdSm90::kBlockM;
    static constexpr int kBlockN = CollectiveMainloopBwdSm90::kBlockN;

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_block;
    int const bidh = block_meta.bidh;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;
    // BlockMeta: reassigned per RangeMerge batch in while(true)
    flash::AttnType attn_type;
    int m_block_min;
    int m_block_max;
    int offset_q;
    int last_n_block;
    // PackGQA: Q heads packed into seqlen → m_block_num includes QheadPerKhead factor.
    // CatGQA: Q heads stay in head dim → m_block_num is based on raw seqlen_q.
    int m_block_num = cute::ceil_div(seqlen_info.seqlen_q * cute::conditional_return<PackGQA>(QheadPerKhead, 1), kBlockM);
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
    // PackGQA: offset_q is already scaled by QheadPerKhead (set in the main loop below),
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
    Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
    Tensor sdQ_store = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumStore{});
    Tensor mdQaccum = params.tma_add_dQ.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
    auto block_tma_dQ = params.tma_add_dQ.get_slice(_0{});
    Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ);

    // Scatter store addressing for producer store warp (32 threads).
    // PackGQA: smem_token_indices hold packed rows p = token*G + g; bidh is the kv head,
    // so the head base and the per-row decompose are scaled by G (see scatter_reduce_store_rows).
    static constexpr int kdQPackScale = PackGQA ? QheadPerKhead : 1;
    [[maybe_unused]] int const store_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarp;
    [[maybe_unused]] int const stride_dq_row = get<0>(params.stride_dQ);
    [[maybe_unused]] int const stride_dq_head = get<2>(params.stride_dQ);
    [[maybe_unused]] ElementAccum* const ptr_dQ_base = params.ptr_dQ + bidh * kdQPackScale * static_cast<int64_t>(get<2>(params.stride_dQ));

    auto store_dQ_this_m_block = [&](int const m_block, int const bidh_kv_cat, int const off_q) {
#pragma unroll
      // Sync at sdQ full barrier, to wait for all consumer WGs to finish dQ r2s-copy
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<NumdQBarrierThreads>(BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }

      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        // 2D TMA reduce: entire tile written in one TMA reduce-add instruction.
        // LoopQ: all kBlockM packed rows belong to one physical token (QheadPerKhead >= kBlockM).
        if (lane_predicate) {
          int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[kStages * kBlockM];
          int const m_block_abs = packed_first_row / kBlockM;
          Tensor gdQaccum_abs = local_tile(mdQaccum, TileShape_dQaccum{}, gQdOdQ_coord);
          Tensor tdQgdQ_abs = block_tma_dQ.partition_D(gdQaccum_abs);
          cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ_abs(_, _, _, m_block_abs));
          tma_store_arrive();
          tma_store_wait<0>();
        }
      } else if constexpr (InnerUseScatter) {
        // Per-row 1D bulk reduce fallback for non-MQA scatter
        scatter_reduce_store_rows<kBlockM, cutlass::NumThreadsPerWarp, kdQPackScale>(
            sdQ_store,
            &shared_storage.tensors.mainloop.smem_token_indices[kStages * kBlockM],
            ptr_dQ_base,
            stride_dq_row,
            store_thread_idx,
            /*row_offset=*/0,
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
          Tensor gdQaccum = local_tile(domain_offset(gQdO_offset_q_coord, mdQaccum), TileShape_dQaccum{}, gQdOdQ_coord);
          Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum);
          if constexpr (CatGQA) {
            cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block, bidh_kv_cat));
          } else {
            cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
          }

          tma_store_arrive();
          tma_store_wait<0>();
          if constexpr (Deterministic) {
            if constexpr (CatGQA) {
              if (bidh_kv_cat == QheadPerKhead - 1) {
                m_block_arrive(m_block);
              }
            } else {
              m_block_arrive(m_block);
            }
          }
        }
      }

      // Arrive at sdQ empty barrier
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<NumdQBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
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
          // by packed offsets (seqlen_q * QheadPerKhead), so we must scale accordingly.
          int bidb_last_l = params.q_ranges[bidb_last].x, bidb_last_r = params.q_ranges[bidb_last].y;
          if constexpr (PackGQA) {
            bidb_last_l *= QheadPerKhead;
            bidb_last_r *= QheadPerKhead;
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
      if constexpr (InnerUseScatter) {
        // Scatter path uses the sparse consumer BlockMeta (one inner m_block per store_body
        // call; prefetch() advances a single block — mirrors store_dkv). The handshake count
        // thus matches the MMA consumer's ceil(gathered_tokens / kBlockM) tile count exactly.
        // m_block / bidh_kv_cat / off_q are unused by the scatter branch.
        store_dQ_this_m_block(block_meta.inner_block_cur, 0, 0);
      } else {
        m_block_min = block_meta.inner_block_min;
        m_block_max = block_meta.inner_block_max;
        seqlen_info = block_meta.seqlen_info;
        bidb = block_meta.bidb;
        attn_type = block_meta.attn_type;
        offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * QheadPerKhead;
        last_n_block = cute::ceil_div(seqlen_info.seqlen_k, kBlockN) - 1;
        m_block_num = cute::ceil_div(seqlen_info.seqlen_q * cute::conditional_return<PackGQA>(QheadPerKhead, 1), kBlockM);

        update_conflict_state(bidb_last, bidb);
        bidb_last = bidb;

        deterministic_pass_through(0, m_block_min);

        for (int bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, QheadPerKhead); ++bidh_kv_cat) {
          int m_block = flash::init_block_cur<kInnerDir>(m_block_min, m_block_max);
          flash::iterate_range<kInnerDir, 2>(m_block, m_block_min, m_block_max, [&] { store_dQ_this_m_block(m_block, bidh_kv_cat, offset_q); });
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
    static_assert(SwapBwdQKLoop, "store_dkv() must be called when SwapBwdQKLoop is true");
    static_assert(!Deterministic, "Deterministic mode is not supported yet");

    if constexpr (!dKVacc_use_TMA || DkvaccBypassSmem) {
      return;
    }

    // ─── Definitions hoisted to function top: shared by the Dense TMA-store path and the
    //     BlockSparse/IndexSparse scatter-store path. All are pure (layout / scalar) computations,
    //     so whatever is unused on a given path is DCE'd away (no runtime cost / no descriptor deref). ───
    // BlockMeta: fixed per function call
    int const bidh_kv = block_meta.bidh_kv;

    // dKV pool offset: each CTA writes to its own pool slice to reduce L2 contention
    int const pool_offset = params.pool_count > 1 ? (blockIdx.x % params.pool_count) * params.pool_seqlen_k : 0;

    bool const lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

    // smem dK/dV accumulators: scatter store reads the Store layout view, dense TMA the swizzled one
    Tensor sdK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumStore{});
    Tensor sdV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumStore{});
    Tensor sdK_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumTMA{});
    Tensor sdV_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumTMA{});

    // Dense TMA reduce-add setup (uses shape_dKdV which includes pool dimension)
    Tensor mdKaccum = params.tma_add_dK.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);
    Tensor mdVaccum = params.tma_add_dV.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);
    auto block_tma_dK = params.tma_add_dK.get_slice(_0{});
    Tensor tdKsdK = block_tma_dK.partition_S(sdK_tma); // (TMA, TMA_N, TMA_K)
    auto block_tma_dV = params.tma_add_dV.get_slice(_0{});
    Tensor tdVsdV = block_tma_dV.partition_S(sdV_tma); // (TMA, TMA_N, TMA_K)

    // BlockSparse / IndexSparse scatter-store addressing (pool_offset applied to base pointers)
    int const thread_idx = threadIdx.x % NumBlockSparseThreads;
    int const stride_dV_row = get<0>(params.stride_dV);
    int const stride_dK_row = get<0>(params.stride_dK);
    ElementAccum* const ptr_gdV_base = params.ptr_dV + bidh_kv * get<2>(params.stride_dV) + pool_offset * stride_dV_row;
    ElementAccum* const ptr_gdK_base = params.ptr_dK + bidh_kv * get<2>(params.stride_dK) + pool_offset * stride_dK_row;
    int const* const idx_staging = [&]() -> int const* {
      if constexpr (InnerUseScatter) {
        // [staging_dv][staging_dk] right after the kStages stage slots
        return shared_storage.tensors.mainloop.smem_token_indices.data() + kStages * kBlockN;
      } else {
        return nullptr;
      }
    }();

    // ─── Unified store_dV / store_dK: scatter vs TMA reduce-add ───
    // Dense: only warp 1 in dV store, warp 2 in dK store (barrier width = 1 warp).
    // InnerUseScatter: all scatter-store threads participate in both; the token indices come
    // from the smem slots written by the loader (single source of truth, no re-stepping here).
    auto store_dV = [&]() {
      if constexpr (!InnerUseScatter) {
        if (warp_idx_in_warpgroup != 1)
          return;
      }
#pragma unroll
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        // 2D TMA reduce: entire dV tile (absolute coordinates + pool offset).
        // When InnerUseScatter, multiple DxStorer warps enter this path; gate on
        // the first DxStorer warp to issue exactly one TMA reduce-add per tile.
        if (lane_predicate && warp_idx_in_warpgroup == ProducerWarpRoles::kNumLoaderWarps) {
          int const packed_first_row = idx_staging[0];
          int const n_block_abs = (pool_offset + packed_first_row) / kBlockN;
          Tensor gdVaccum_abs = local_tile(mdVaccum, TileShape_dKVaccum{}, make_coord(_, _0{}));
          Tensor tdVgdV_abs = block_tma_dV.partition_D(gdVaccum_abs);
          cute::copy(params.tma_add_dV, tdVsdV, tdVgdV_abs(_, _, _, n_block_abs));
          tma_store_arrive();
          tma_store_wait<0>();
        }
      } else if constexpr (InnerUseScatter) {
        scatter_reduce_store_rows<kBlockN, NumBlockSparseThreads, /*kRowPackScale=*/1>(sdV, &idx_staging[0 * kBlockN], ptr_gdV_base, stride_dV_row, thread_idx);
      } else {
        if (lane_predicate) {
          Tensor gdVaccum =
              local_tile(domain_offset(make_coord(pool_offset + block_meta.seqlen_info.offset_k, _0{}), mdVaccum), TileShape_dKVaccum{}, make_coord(_, _0{}));
          Tensor tdVgdV = block_tma_dV.partition_D(gdVaccum);
          cute::copy(params.tma_add_dV, tdVsdV, tdVgdV(_, _, _, block_meta.inner_block_cur));
          tma_store_arrive();
          tma_store_wait<0>();
        }
      }
      // Signal dKEmpty (not dVEmpty): smem_dkacc/dvacc are unioned — after TMA dV
      // finishes reading, the consumer can safely r2s dK into the shared buffer.
      // The consumer's dV r2s is gated by dVEmpty (signaled after TMA dK below).
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
      }
    };

    auto store_dK = [&]() {
      if constexpr (!InnerUseScatter) {
        if (warp_idx_in_warpgroup != 2)
          return;
      }
#pragma unroll
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warpgroup_idx);
      }
      if constexpr (Use_TMA_Inner && InnerUseScatter) {
        // 2D TMA reduce: entire dK tile (absolute coordinates + pool offset).
        // Same single-warp gate as store_dV above.
        if (lane_predicate && warp_idx_in_warpgroup == ProducerWarpRoles::kNumLoaderWarps) {
          int const packed_first_row = idx_staging[kBlockN];
          int const n_block_abs = (pool_offset + packed_first_row) / kBlockN;
          Tensor gdKaccum_abs = local_tile(mdKaccum, TileShape_dKVaccum{}, make_coord(_, _0{}));
          Tensor tdKgdK_abs = block_tma_dK.partition_D(gdKaccum_abs);
          cute::copy(params.tma_add_dK, tdKsdK, tdKgdK_abs(_, _, _, n_block_abs));
          tma_store_arrive();
          tma_store_wait<0>();
        }
      } else if constexpr (InnerUseScatter) {
        scatter_reduce_store_rows<kBlockN, NumBlockSparseThreads, /*kRowPackScale=*/1>(sdK, &idx_staging[1 * kBlockN], ptr_gdK_base, stride_dK_row, thread_idx);
      } else {
        if (lane_predicate) {
          Tensor gdKaccum =
              local_tile(domain_offset(make_coord(pool_offset + block_meta.seqlen_info.offset_k, _0{}), mdKaccum), TileShape_dKVaccum{}, make_coord(_, _0{}));
          Tensor tdKgdK = block_tma_dK.partition_D(gdKaccum);
          cute::copy(params.tma_add_dK, tdKsdK, tdKgdK(_, _, _, block_meta.inner_block_cur));
          tma_store_arrive();
          tma_store_wait<0>();
        }
      }
      // Signal dVEmpty (not dKEmpty): after TMA dK finishes reading, the consumer
      // can safely r2s dV of the next iteration into the shared buffer.
      for (int warpgroup_idx = 0; warpgroup_idx < NumMmaWarpGroups; ++warpgroup_idx) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warpgroup_idx);
      }
    };

    auto store_body = [&]() {
      if constexpr (InnerUseScatter) {
        // NOTE(058 P2a-2): an overlapped dV/dK variant (defer the dV bulk wait until after the
        // dK issue via staged tma_store_wait<1>/<0>) was implemented and benched: zero gain on
        // sparseload-loopk / indexattn-loopk (159/161 TF unchanged) — the store warps' wait is
        // not on the critical path once bulk reduce is enabled. Reverted to keep the simple
        // sequential form; see .tmp/058-fwd-tokenidx/NOTES.md.
        store_dV();
        store_dK();
      } else {
        flash::iterate_range<kInnerDir, 2>(block_meta.inner_block_cur, block_meta.inner_block_min, block_meta.inner_block_max, [&] {
          store_dV();
          store_dK();
        });
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
    if constexpr (SwapBwdQKLoop) { // q for outer-loop and k for inner-loop
      // Tell producer that smem_q and smem_do are ready
      BarrierManager::arrive<NumMmaThreads + NumKVEmptyProducerThreads>(BwdNamedBarriers::QdOEmpty);

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (dKVacc_use_TMA && InnerDxStoreInProducer && !DkvaccBypassSmem) {
        // Initial arrive on behalf of store warps: smem_dkvacc is initially empty.
        // Only dVEmpty gets initial arrive — the first r2s dV can proceed immediately.
        // dKEmpty is NOT pre-arrived: the first r2s dK must wait for the producer's
        // TMA dV to finish (store_dV signals dKEmpty), because smem_dkacc and
        // smem_dvacc share memory via union.
        if (warp_idx_in_warpgroup == 0 || (InnerUseScatter && warp_idx_in_warpgroup == 1)) {
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);
        }
      }
    } else { // k for outer-loop and q for inner-loop
      // We're not currently using this bc we're not using persistent scheduler
      // Tell producer (warp 0) that smem_k and smem_v are ready
      BarrierManager::arrive<NumMmaThreads + NumKVEmptyProducerThreads>(BwdNamedBarriers::KVEmpty);

      int warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();

      if constexpr (dQacc_use_TMA) {
        if constexpr (!InnerDxStoreInProducer) {
          // Consumer handles dQ store: all threads in WG arrive (no separate store warp)
          BarrierManager::arrive<NumdQBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
        } else {
          if (warp_idx_in_warpgroup == 0) {
            BarrierManager::arrive<NumdQBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
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
    static_assert(!SwapBwdQKLoop, "mma_with_loop_q() must be called when SwapBwdQKLoop is false");
    static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    // BlockMeta: fixed per function call
    int const n_block = block_meta.outer_block;
    int const bidh = block_meta.bidh;
    int bidb = block_meta.bidb;
    SeqlenInfo_t seqlen_info = block_meta.seqlen_info;
    int offset_q = !PackGQA ? seqlen_info.offset_q : seqlen_info.offset_q * QheadPerKhead;
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
    Tensor sdQ = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumStore{}));
    Tensor sdQt =
        cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumtStore{}));

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
    auto const mQdOdQLSEdPsum_coord = make_coord(_, _, cute::conditional_return<CatGQA>(make_coord(_, bidh), bidh));
    auto const gQdOdQ_coord = cute::conditional_return<CatGQA>(make_coord(_, _0{}, _), make_coord(_, _0{}));
    Tensor mdQaccum = params.tma_add_dQ.get_tma_tensor(params.shape_QdOdQ)(mQdOdQLSEdPsum_coord);
    auto const gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(offset_q, _0{}, _0{}), make_coord(offset_q, _0{}));
    Tensor gdQaccum_ = local_tile(domain_offset(gQdO_offset_q_coord, mdQaccum), TileShape_dQaccum{}, gQdOdQ_coord); // (M, K, _)
    Tensor gdQaccum = cute::flat_divide(gdQaccum_, make_shape(Int<kBlockM / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (M / WG, K, WG, 1, _)
    // We can reuse r2s_thr_copy_dQaccum for this partitioning
    Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);

    auto rebind_dQ_accum_tiles = [&]() {
      if constexpr (!RangeMerge) {
        return;
      }
      int const new_offset_q = !PackGQA ? block_meta.seqlen_info.offset_q : block_meta.seqlen_info.offset_q * QheadPerKhead;
      if constexpr (!dQacc_use_TMA) {
        auto const new_gQdO_offset_q_coord = cute::conditional_return<CatGQA>(make_coord(new_offset_q, _0{}, _0{}), make_coord(new_offset_q, _0{}));
        gdQaccum_ = local_tile(domain_offset(new_gQdO_offset_q_coord, mdQaccum), TileShape_dQaccum{}, gQdOdQ_coord);
        gdQaccum = cute::flat_divide(gdQaccum_, make_shape(Int<kBlockM / NumMmaWarpGroups>{}, Int<kHeadDim>{}));
        tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
      }
    };

    /* DEBUG */
    // if (thread_idx == 0 && bidh == 0 && n_block == 0){
    //     printf("bidb: %d, offset_q: %d\n", bidb, seqlen_info.offset_q);
    //     printf("gdQaccum_: "); print(gdQaccum_); printf("\n");
    //     printf("gdQaccum: "); print(gdQaccum); printf("\n");
    // }
    // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask;

    // Wait until this n block of K,V loaded
    cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.pipelines.barrier_KV.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.pipelines.barrier_KV.wait(work_idx % 2);
    }

    if constexpr (Mma_dP_is_RS) { // guanrateed SdP_SwapAB, then only V needs to copy to registers
      using SmemCopyAtomV = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_V = make_tiled_copy_A(SmemCopyAtomV{}, tiled_mma_dP);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
      Tensor tdPrV_copy_view = smem_thr_copy_V.retile_D(tdPrV);
      Tensor tdPsV_copy_view = smem_thr_copy_V.partition_S(cute::as_position_independent_swizzle_tensor(sV));
      cute::copy(smem_tiled_copy_V, tdPsV_copy_view, tdPrV_copy_view);
    }

    Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));

    // Define backward step lambda func
    auto bwd_step = [&](int m_block, auto mask_fn, auto /*is_no_mask*/ = cute::false_type{}) {
      bool const is_last_m_block_this_batch = [&]() {
        if constexpr (BlockSparse) {
          return m_block == block_meta.padding_block() && block_meta.num_invalid_token > 0;
        } else {
          return (m_block == block_meta.inner_block_max - 1);
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
            return kBlockM - block_meta.num_invalid_token - thread_row_offset;
          } else {
            int const seqlen_q_packed_local = !PackGQA ? block_meta.seqlen_info.seqlen_q : block_meta.seqlen_info.seqlen_q * QheadPerKhead;
            return seqlen_q_packed_local - m_block * kBlockM - thread_row_offset;
          }
        }();

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
        // P uses 1-stage buffer: always sync to ensure prev iter's MMA3 consumed P
        BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
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
        if constexpr (dQacc_use_TMA) {
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          // Sync at sdQ empty barrier (producer store mode only): wait until the store warp
          // finished the previous tile's store. The consumer store mode instead relies on its
          // trailing cross-WG sync below to guarantee sdQ is free.
          if constexpr (InnerDxStoreInProducer) {
            BarrierManager::sync<NumdQBarrierThreads>(BwdNamedBarriers::dQEmptyWG1, /*warp_group_idx=*/warp_group_idx);
            if constexpr (InnerUseScatter) {
              // Copy this tile's token indices from the stage slot (still held — Q is
              // released only after MMA5) into the staging area for the store warp.
              // Protected by the dQEmpty/dQFull handshake; WG0 alone suffices since the
              // store warp syncs every WG's dQFull before reading.
              if (warp_group_idx == 0) {
                int const* const src = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_q.index() * kBlockM];
                int* const dst = &shared_storage.tensors.mainloop.smem_token_indices[kStages * kBlockM];
                for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockM; r += cutlass::NumThreadsPerWarpGroup) {
                  dst[r] = src[r];
                }
              }
            }
          }

          // Copy dQ from registers to shared memory with softmax_scale applied
          Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
          for (int dqi = 0; dqi < size(taccdQrdQ); ++dqi) {
            taccdQrdQ(dqi) *= params.softmax_scale;
          }
          cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
          cutlass::arch::fence_view_async_shared();

          if constexpr (!InnerDxStoreInProducer) {
            // Consumer store path: the consumer WGs reduce-add dQ from SMEM to global dQ.
            // The dQ MMA may split the head dim across WGs (e.g. dQ_swapAB with
            // AtomLayoutNdQ=2: each WG computes ALL kBlockM token rows but only half the
            // columns), so a token row in sdQ mixes both WGs' r2s writes. The sync must
            // therefore be cross-WG (mirrors the LoopK consumer dKV store); per-WG
            // dQFull/dQEmpty barriers are NOT sufficient.
            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);

            if constexpr (Use_TMA_Inner && InnerUseScatter) {
              // IndexSparse TMA reduce: single thread issues one 2D TMA reduce-add instruction
              if (thread_idx == 0) {
                Tensor sdQ_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
                auto block_tma_dQ_c = params.tma_add_dQ.get_slice(_0{});
                Tensor tdQsdQ_c = block_tma_dQ_c.partition_S(sdQ_tma);

                int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_q.index() * kBlockM];
                int const m_block_abs = packed_first_row / kBlockM;
                Tensor gdQaccum_abs = local_tile(mdQaccum, TileShape_dQaccum{}, gQdOdQ_coord);
                Tensor tdQgdQ_abs = block_tma_dQ_c.partition_D(gdQaccum_abs);
                cute::copy(params.tma_add_dQ, tdQsdQ_c, tdQgdQ_abs(_, _, _, m_block_abs));
                tma_store_arrive();
                tma_store_wait<0>();
              }
            } else if constexpr (InnerUseScatter) {
              // BlockSparse scatter reduce: per-row 1D bulk reduce
              Tensor sdQ_acc = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumStore{});
              static constexpr int kdQPackScale = PackGQA ? QheadPerKhead : 1;
              int const stride_dq_row = get<0>(params.stride_dQ);
              int const stride_dq_head = get<2>(params.stride_dQ);
              ElementAccum* const ptr_dQ_base = params.ptr_dQ + block_meta.bidh * kdQPackScale * static_cast<int64_t>(get<2>(params.stride_dQ));
              int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
              int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
              scatter_reduce_store_rows<kBlockM, NumMmaThreads, kdQPackScale>(
                  sdQ_acc,
                  &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_q.index() * kBlockM],
                  ptr_dQ_base,
                  stride_dq_row,
                  flat_thread_idx,
                  /*row_offset=*/0,
                  stride_dq_head);
            } else {
              // Dense TMA reduce: consumer-side dQ store via TMA reduce-add
              static_assert(!CatGQA, "Consumer dQ TMA store for CatGQA not yet implemented; use InnerDxStoreInProducer");
              if (thread_idx == 0) {
                Tensor sdQ_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
                auto block_tma_dQ_c = params.tma_add_dQ.get_slice(_0{});
                Tensor tdQsdQ_c = block_tma_dQ_c.partition_S(sdQ_tma);
                auto const gQdO_off = make_coord(offset_q, _0{});
                Tensor gdQaccum_c = local_tile(domain_offset(gQdO_off, mdQaccum), TileShape_dQaccum{}, gQdOdQ_coord);
                Tensor tdQgdQ_c = block_tma_dQ_c.partition_D(gdQaccum_c);
                cute::copy(params.tma_add_dQ, tdQsdQ_c, tdQgdQ_c(_, _, _, m_block));
                tma_store_arrive();
                tma_store_wait<0>();
              }
            }

            // Cross-WG sync: all scatter reads done before the next iteration's r2s overwrites sdQ
            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          }
          if constexpr (InnerDxStoreInProducer) {
            // Producer store path: signal the producer store warp (TMA or scatter) that
            // sdQ is full; pairs with the dQEmpty sync at the top of this block.
            BarrierManager::arrive<NumdQBarrierThreads>(BwdNamedBarriers::dQFullWG1, /*warp_group_idx=*/warp_group_idx);
          }
        } else { // directly atomic reduce-add to global memory
          static_assert(!(InnerUseScatter && !SwapBwdQKLoop), "BlockSparse LoopQ requires dQacc_use_TMA (kHeadDim <= 128)");
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

    // Unified MMA body: iterates over all m_blocks in the range with a single bwd_step instantiation.
    auto mma_body = [&]() {
      if constexpr (InnerUseScatter) {
        // Scatter (BlockSparse / IndexSparse LoopQ): one Q block per call, block_meta drives iteration.
        // LoopQ needs both padding masks: rows are the scattered Q tokens
        // (last tile may hold fewer than kBlockM valid tokens) and columns are
        // the contiguous K window (last n_block may overhang seqlen_k) —
        // symmetric with LoopK, where the roles of rows/columns are swapped.
        bool const need_row_mask = block_meta.inner_block_cur == block_meta.padding_block() && block_meta.num_invalid_token > 0;
        int const num_invalid_k_token = !SwapBwdQKLoop ? cute::max(0, (block_meta.outer_block + 1) * kBlockN - block_meta.seqlen_info.seqlen_k) : 0;
        bool const need_col_mask = num_invalid_k_token > 0;
        auto combined_mask_fn = [&](int /*m_blk*/) {
          if (need_col_mask) {
            mask.template apply_padding_mask(tSrS, num_invalid_k_token, thread_idx);
          }
          if (need_row_mask) {
            mask.template apply_padding_mask_row(tSrS, block_meta.num_invalid_token, thread_idx);
          }
        };
        auto sparse_no_mask_fn = [&](int /*m_blk*/) {};
        if (need_row_mask || need_col_mask) {
          bwd_step(block_meta.inner_block_cur, combined_mask_fn, cute::false_type{});
        } else {
          bwd_step(block_meta.inner_block_cur, sparse_no_mask_fn, cute::false_type{});
        }
      } else {
        rebind_dQ_accum_tiles();

        for (int bidh_kv_cat = 0; bidh_kv_cat < cute::conditional_return<!CatGQA>(1, QheadPerKhead); ++bidh_kv_cat) {
          if constexpr (MaskMode == 0) {
            // MaskMode 0 (regular): direct apply every block with Seqlenk_mask=true.
            auto mask_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/true, PackGQA, QheadPerKhead>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            int mb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_max);
            flash::iterate_range<kInnerDir>(mb, block_meta.inner_block_min, block_meta.inner_block_max, [&] { bwd_step(mb, mask_fn, cute::false_type{}); });
          } else if constexpr (MaskMode == 1) {
            // MaskMode 1 (dispatch): 3-lambda zone splitting (compile-time).
            auto boundary_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/true, PackGQA, QheadPerKhead>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            auto regular_fn = [&](int m_block) {
              mask.template apply</*Seqlenk_mask=*/false, PackGQA, QheadPerKhead>(
                  tSrS, m_block, n_block, block_meta.attn_type, thread_idx, block_meta.seqlen_info.seqlen_q, block_meta.seqlen_info.seqlen_k);
            };
            auto no_mask_fn = [&](int /*m_block*/) {};
            int mb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_max);
            flash::mask_dispatch<kBlockM, kBlockN, PackGQA, QheadPerKhead, flash::DispatchAxis::M, kInnerDir>(
                mb,
                block_meta.inner_block_min,
                block_meta.inner_block_max,
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
            flash::mask_dispatch_unified<kBlockM, kBlockN, PackGQA, QheadPerKhead, flash::DispatchAxis::M, kInnerDir>(block_meta, mask, tSrS, thread_idx, bwd_step);
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
    static_assert(SwapBwdQKLoop, "mma_with_loop_k() must be called when SwapBwdQKLoop is true");
    static_assert(!CatGQA, "mma_with_loop_k() is not implemented for CatGQA");
    static_assert(is_rmem<FrgTensordQ>::value, "dQ tensor must be rmem resident.");

    /* DEBUG */
    // debug_print_mma();

    // BlockMeta: fixed per function call
    int const m_block = block_meta.outer_block;
    int const bidh = block_meta.bidh;
    int const bidh_kv = block_meta.bidh_kv;
    int const seqlen_q = block_meta.seqlen_info.seqlen_q;
    int const seqlen_q_packed = !PackGQA ? seqlen_q : seqlen_q * QheadPerKhead;
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
    // bulk-reduce scatter path swaps in the row-contiguous layout)
    Tensor sdK = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumStore{}));
    Tensor sdKt =
        cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumtStore{}));
    Tensor sdV = cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumStore{}));
    Tensor sdVt =
        cute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumtStore{}));

    ElementAccum* lse_smem_ptr_c;
    ElementAccum* dpsum_smem_ptr_c;
    if constexpr (LseDpsumUnionEffective) {
      lse_smem_ptr_c = reinterpret_cast<ElementAccum*>(shared_storage.tensors.mainloop.smem_dkacc.data());
      dpsum_smem_ptr_c = lse_smem_ptr_c + cute::cosize_v<SmemLayoutLSE>;
    } else {
      lse_smem_ptr_c = shared_storage.tensors.mainloop.smem_lse.data();
      dpsum_smem_ptr_c = shared_storage.tensors.mainloop.smem_dpsum.data();
    }
    Tensor sdPsumMma_full = make_tensor(make_smem_ptr(dpsum_smem_ptr_c), SmemLayoutLSEMma{});
    Tensor sLSEMma_full = make_tensor(make_smem_ptr(lse_smem_ptr_c), SmemLayoutLSEMma{});
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

    int const offset_k = block_meta.seqlen_info.offset_k;

    // dKV pool offset: each CTA writes to its own pool slice
    int const pool_offset = params.pool_count > 1 ? (blockIdx.x % params.pool_count) * params.pool_seqlen_k : 0;

    // For the case where we do atomicAdd directly to gdKaccum,gdVaccum instead of using TMA
    Tensor mdKaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dK)), params.shape_dKdV, params.stride_dK)(_, _, bidh_kv);
    Tensor gdKaccum_ = local_tile(domain_offset(make_coord(pool_offset + offset_k, _0{}), mdKaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)
    Tensor gdKaccum = cute::flat_divide(gdKaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

    Tensor mdVaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dV)), params.shape_dKdV, params.stride_dV)(_, _, bidh_kv);
    Tensor gdVaccum_ = local_tile(domain_offset(make_coord(pool_offset + offset_k, _0{}), mdVaccum), TileShape_dKVaccum{}, make_coord(_, _0{})); // (N, K, _)
    Tensor gdVaccum = cute::flat_divide(gdVaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{})); // (N / WG, K, WG, 1, _)

    // TMA partitions go through the swizzled TMA-layout views (dense path only; the r2s
    // targets sdK/sdV above may carry the row-contiguous Store layout instead)
    Tensor sdK_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumTMA{});
    Tensor sdV_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumTMA{});
    auto block_tma_dK = params.tma_add_dK.get_slice(_0{});
    Tensor tdKgdK = block_tma_dK.partition_D(gdKaccum); // (TMA, TMA_N, TMA_K)
    Tensor tdKsdK = block_tma_dK.partition_S(sdK_tma); // (TMA, TMA_N, TMA_K)

    auto block_tma_dV = params.tma_add_dV.get_slice(_0{});
    Tensor tdVgdV = block_tma_dV.partition_D(gdVaccum); // (TMA, TMA_N, TMA_K)
    Tensor tdVsdV = block_tma_dV.partition_S(sdV_tma); // (TMA, TMA_N, TMA_K)

    // We can reuse r2s_thr_copy_dKVaccum for this partitioning
    Tensor tdKgdKaccum = r2s_thr_copy_dKVaccum.partition_D(gdKaccum);
    Tensor tdVgdVaccum = r2s_thr_copy_dKVaccum.partition_D(gdVaccum);

    auto rebind_dKV_accum_tiles = [&]() {
      if constexpr (!RangeMerge) {
        return;
      }
      int const new_offset_k = block_meta.seqlen_info.offset_k;
      if constexpr (!dKVacc_use_TMA || (DkvaccBypassSmem && !InnerUseScatter)) {
        gdKaccum_ = local_tile(domain_offset(make_coord(pool_offset + new_offset_k, _0{}), mdKaccum), TileShape_dKVaccum{}, make_coord(_, _0{}));
        gdKaccum = cute::flat_divide(gdKaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{}));
        gdVaccum_ = local_tile(domain_offset(make_coord(pool_offset + new_offset_k, _0{}), mdVaccum), TileShape_dKVaccum{}, make_coord(_, _0{}));
        gdVaccum = cute::flat_divide(gdVaccum_, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{}));
        tdKgdKaccum = r2s_thr_copy_dKVaccum.partition_D(gdKaccum);
        tdVgdVaccum = r2s_thr_copy_dKVaccum.partition_D(gdVaccum);
      }
    };

    /* DEBUG */
    // if (blockIdx.x == 0 && threadIdx.x == 128) {
    // print(mdKaccum); printf("\n"); print(gdKaccum_); printf("\n"); print(gdKaccum); printf("\n"); print(tdKgdKaccum); printf("\n"); print(tdKsdK); printf("\n");
    // print(mdVaccum); printf("\n"); print(gdVaccum_); printf("\n"); print(gdVaccum); printf("\n"); print(tdVgdVaccum); printf("\n"); print(tdVsdV); printf("\n");
    // printf("\n"); }

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

    if constexpr (Mma_dP_is_RS) {
      // NOTE: if Mma_dP_is_RS, then SdP_SwapAB must be true,
      // then we have to copy current n block of V to registers every iteration,
      // which seems unacceptable for loop-k settings
      static_assert(!Mma_dP_is_RS, "Mma_dP_is_RS is not supported yet when SwapBwdQKLoop is true.");
    }

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
      // BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
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
        if constexpr (DkvaccBypassSmem) {
          // F6 bypass: atomicAdd from registers directly to global (no SMEM roundtrip)
          static_assert(!DkvaccBypassSmem || !dKV_swapAB, "DkvaccBypassSmem scatter requires !dKV_swapAB (kHeadDim<=128)");
          Tensor taccdVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV);
          if constexpr (InnerUseScatter) {
            int const* const tidx = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
            int const stride_dV_row = get<0>(params.stride_dV);
            auto* ptr_gdV = reinterpret_cast<ElementAccum*>(params.ptr_dV) + bidh_kv * get<2>(params.stride_dV);
            Tensor cdKV = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
            Tensor cdKV_div = cute::flat_divide(cdKV, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{}));
            Tensor thr_cdKV = r2s_thr_copy_dKVaccum.partition_D(cdKV_div);
#pragma unroll
            for (int i = 0; i < size(taccdVrdV); ++i) {
              auto coord = thr_cdKV(i);
              int row = get<0>(coord);
              int col = get<1>(coord);
              int token_idx = tidx[row];
              atomicAdd(ptr_gdV + (pool_offset + token_idx) * stride_dV_row + col, taccdVrdV(i));
            }
          } else {
            Tensor tdVrdV_atomic = recast<float4>(taccdVrdV);
            Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, _, _, n_block));
#pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) {
              atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i));
            }
          }
        } else if constexpr (dKVacc_use_TMA && InnerDxStoreInProducer) {
          // Write to smem, signal producer store warp for TMA/scatter reduce-add
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);

          if constexpr (InnerUseScatter) {
            if (warp_group_idx == 0) {
              int const* const src = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
              int* const dst = &shared_storage.tensors.mainloop.smem_token_indices[kStages * kBlockN];
              for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockN; r += cutlass::NumThreadsPerWarpGroup) {
                dst[r] = src[r];
              }
            }
          }

          Tensor taccdVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV);
          cute::copy(r2s_tiled_copy_dKVaccum, taccdVrdV, tdVsdVaccum);

          cutlass::arch::fence_view_async_shared();
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVFullWG1, /*warp_group_idx=*/warp_group_idx);
        } else {
          // Consumer store path: dispatch on the store mechanism
          if constexpr (Use_TMA_Inner && InnerUseScatter) {
            static_assert(dKVacc_use_TMA, "Consumer scatter dKV requires smem accumulator buffer (kHeadDim < 256)");

            Tensor taccdVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV);
            cute::copy(r2s_tiled_copy_dKVaccum, taccdVrdV, tdVsdVaccum);
            cutlass::arch::fence_view_async_shared();

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);

            if (thread_idx == 0) {
              Tensor sdV_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumTMA{});
              auto block_tma_dV_c = params.tma_add_dV.get_slice(_0{});
              Tensor tdVsdV_c = block_tma_dV_c.partition_S(sdV_tma);

              int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
              int const n_block_abs = (pool_offset + packed_first_row) / kBlockN;
              Tensor mdVaccum_c = params.tma_add_dV.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);
              Tensor gdVaccum_abs = local_tile(mdVaccum_c, TileShape_dKVaccum{}, make_coord(_, _0{}));
              Tensor tdVgdV_abs = block_tma_dV_c.partition_D(gdVaccum_abs);
              cute::copy(params.tma_add_dV, tdVsdV_c, tdVgdV_abs(_, _, _, n_block_abs));
              tma_store_arrive();
              tma_store_wait<0>();
            }

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          } else if constexpr (InnerUseScatter) {
            static_assert(dKVacc_use_TMA, "Consumer scatter dKV requires smem accumulator buffer (kHeadDim < 256)");

            Tensor taccdVrdV = r2s_thr_copy_dKVaccum.retile_S(tdVrdV);
            cute::copy(r2s_tiled_copy_dKVaccum, taccdVrdV, tdVsdVaccum);
            cutlass::arch::fence_view_async_shared();

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);

            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
            int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
            int const stride_dV_row = get<0>(params.stride_dV);
            ElementAccum* const ptr_gdV_base = params.ptr_dV + bidh_kv * get<2>(params.stride_dV) + pool_offset * stride_dV_row;
            Tensor sdV_store = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dvacc.data()), SmemLayoutdKVaccumStore{});
            scatter_reduce_store_rows<kBlockN, NumMmaThreads, /*kRowPackScale=*/1>(
                sdV_store, &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN], ptr_gdV_base, stride_dV_row, flat_thread_idx);

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          } else {
            Tensor tdVrdV_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdVrdV));
            Tensor tdVgdVaccum_atomic = recast<float4>(tdVgdVaccum(_, _, _, _, _, n_block));
            static_assert(CUTE_STATIC_V(size(tdVrdV_atomic)) == CUTE_STATIC_V(size(tdVgdVaccum_atomic)));
#pragma unroll
            for (int i = 0; i < size(tdVrdV_atomic); ++i) {
              atomicAdd(&tdVgdVaccum_atomic(i), tdVrdV_atomic(i));
            }
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
        if constexpr (DkvaccBypassSmem) {
          // F6 bypass: atomicAdd dK from registers directly to global
          Tensor taccdKrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK);
#pragma unroll
          for (int dki = 0; dki < size(taccdKrdK); ++dki) {
            taccdKrdK(dki) *= params.softmax_scale;
          }
          if constexpr (InnerUseScatter) {
            int const* const tidx = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
            int const stride_dK_row = get<0>(params.stride_dK);
            auto* ptr_gdK = reinterpret_cast<ElementAccum*>(params.ptr_dK) + bidh_kv * get<2>(params.stride_dK);
            Tensor cdKV = make_identity_tensor(make_shape(Int<kBlockN>{}, Int<kHeadDim>{}));
            Tensor cdKV_div = cute::flat_divide(cdKV, make_shape(Int<kBlockN / NumMmaWarpGroups>{}, Int<kHeadDim>{}));
            Tensor thr_cdKV = r2s_thr_copy_dKVaccum.partition_D(cdKV_div);
#pragma unroll
            for (int i = 0; i < size(taccdKrdK); ++i) {
              auto coord = thr_cdKV(i);
              int row = get<0>(coord);
              int col = get<1>(coord);
              int token_idx = tidx[row];
              atomicAdd(ptr_gdK + (pool_offset + token_idx) * stride_dK_row + col, taccdKrdK(i));
            }
          } else {
            Tensor tdKrdK_atomic = recast<float4>(taccdKrdK);
            Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, _, _, n_block));
#pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) {
              atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i));
            }
          }
        } else if constexpr (dKVacc_use_TMA && InnerDxStoreInProducer) {
          // Write to smem, signal producer store warp for TMA/scatter reduce-add
          int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;

          BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dKEmptyWG1, /*warp_group_idx=*/warp_group_idx);

          if constexpr (InnerUseScatter) {
            if (warp_group_idx == 0) {
              int const* const src = &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
              int* const dst = &shared_storage.tensors.mainloop.smem_token_indices[(kStages + 1) * kBlockN];
              for (int r = thread_idx % cutlass::NumThreadsPerWarpGroup; r < kBlockN; r += cutlass::NumThreadsPerWarpGroup) {
                dst[r] = src[r];
              }
            }
          }

          Tensor taccdKrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK);
          for (int dki = 0; dki < size(taccdKrdK); ++dki) {
            taccdKrdK(dki) *= params.softmax_scale;
          }
          cute::copy(r2s_tiled_copy_dKVaccum, taccdKrdK, tdKsdKaccum);

          cutlass::arch::fence_view_async_shared();
          BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dKFullWG1, /*warp_group_idx=*/warp_group_idx);
        } else {
          // Consumer store path: dispatch on the store mechanism
          if constexpr (Use_TMA_Inner && InnerUseScatter) {
            static_assert(dKVacc_use_TMA, "Consumer scatter dKV requires smem accumulator buffer (kHeadDim < 256)");

            Tensor taccdKrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK);
            for (int dki = 0; dki < size(taccdKrdK); ++dki) {
              taccdKrdK(dki) *= params.softmax_scale;
            }
            cute::copy(r2s_tiled_copy_dKVaccum, taccdKrdK, tdKsdKaccum);
            cutlass::arch::fence_view_async_shared();

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);

            if (thread_idx == 0) {
              Tensor sdK_tma = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumTMA{});
              auto block_tma_dK_c = params.tma_add_dK.get_slice(_0{});
              Tensor tdKsdK_c = block_tma_dK_c.partition_S(sdK_tma);

              int const packed_first_row = shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN];
              int const n_block_abs = (pool_offset + packed_first_row) / kBlockN;
              Tensor mdKaccum_c = params.tma_add_dK.get_tma_tensor(params.shape_dKdV)(_, _, bidh_kv);
              Tensor gdKaccum_abs = local_tile(mdKaccum_c, TileShape_dKVaccum{}, make_coord(_, _0{}));
              Tensor tdKgdK_abs = block_tma_dK_c.partition_D(gdKaccum_abs);
              cute::copy(params.tma_add_dK, tdKsdK_c, tdKgdK_abs(_, _, _, n_block_abs));
              tma_store_arrive();
              tma_store_wait<0>();
            }

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          } else if constexpr (InnerUseScatter) {
            static_assert(dKVacc_use_TMA, "Consumer scatter dKV requires smem accumulator buffer (kHeadDim < 256)");

            Tensor taccdKrdK = r2s_thr_copy_dKVaccum.retile_S(tdKrdK);
            for (int dki = 0; dki < size(taccdKrdK); ++dki) {
              taccdKrdK(dki) *= params.softmax_scale;
            }
            cute::copy(r2s_tiled_copy_dKVaccum, taccdKrdK, tdKsdKaccum);
            cutlass::arch::fence_view_async_shared();

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);

            int const warp_group_idx = flash::canonical_warp_group_idx_nosync() - 1;
            int const wg_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
            int const flat_thread_idx = warp_group_idx * cutlass::NumThreadsPerWarpGroup + wg_thread_idx;
            int const stride_dK_row = get<0>(params.stride_dK);
            ElementAccum* const ptr_gdK_base = params.ptr_dK + bidh_kv * get<2>(params.stride_dK) + pool_offset * stride_dK_row;
            Tensor sdK_store = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_dkacc.data()), SmemLayoutdKVaccumStore{});
            scatter_reduce_store_rows<kBlockN, NumMmaThreads, /*kRowPackScale=*/1>(
                sdK_store, &shared_storage.tensors.mainloop.smem_token_indices[smem_pipe_read_k.index() * kBlockN], ptr_gdK_base, stride_dK_row, flat_thread_idx);

            BarrierManager::sync<NumMmaThreads>(BwdNamedBarriers::PdS);
          } else {
            Tensor tdKrdK_atomic = recast<float4>(r2s_thr_copy_dKVaccum.retile_S(tdKrdK));
            Tensor tdKgdKaccum_atomic = recast<float4>(tdKgdKaccum(_, _, _, _, _, n_block));
            static_assert(CUTE_STATIC_V(size(tdKrdK_atomic)) == CUTE_STATIC_V(size(tdKgdKaccum_atomic)));
#pragma unroll
            for (int i = 0; i < size(tdKrdK_atomic); ++i) {
              atomicAdd(&tdKgdKaccum_atomic(i), tdKrdK_atomic(i));
            }
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

      // Release K after MMA5 finished (V already released after MMA2).
      warpgroup_wait<0>();
      pipeline_k.consumer_release(smem_pipe_read_k);

      // Update pipeline read state of K,V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
    };

    // --- Mask lambdas ---
    auto padding_mask_fn = [&](int /*n_blk*/) {
      if constexpr (InnerUseScatter) {
        mask.template apply_padding_mask(tSrS, block_meta.num_invalid_token, thread_idx);
      }
    };
    auto sparse_no_mask_fn = [&](int /*n_blk*/) {};

    // Unified MMA body: scatter processes one n_block per call;
    // dense iterates over all n_blocks in the range with a single bwd_step instantiation.
    auto mma_body = [&]() {
      if constexpr (InnerUseScatter) {
        if (block_meta.inner_block_cur == block_meta.padding_block() && block_meta.num_invalid_token > 0) {
          bwd_step(block_meta.inner_block_cur, padding_mask_fn, cute::false_type{});
        } else {
          bwd_step(block_meta.inner_block_cur, sparse_no_mask_fn, cute::false_type{});
        }
        return;
      }
      rebind_dKV_accum_tiles();

      if constexpr (MaskMode == 0) {
        // MaskMode 0 (regular): direct apply every block with Seqlenk_mask=true.
        auto mask_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/true, PackGQA, QheadPerKhead>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        int nb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_max);
        flash::iterate_range<kInnerDir>(nb, block_meta.inner_block_min, block_meta.inner_block_max, [&] { bwd_step(nb, mask_fn, cute::false_type{}); });
      } else if constexpr (MaskMode == 1) {
        // MaskMode 1 (dispatch): 3-lambda zone splitting.
        auto boundary_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/true, PackGQA, QheadPerKhead>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        auto regular_fn = [&](int n_blk) {
          mask.template apply</*Seqlenk_mask=*/false, PackGQA, QheadPerKhead>(
              tSrS, m_block, n_blk, block_meta.attn_type, thread_idx, seqlen_q, block_meta.seqlen_info.seqlen_k);
        };
        auto no_mask_fn = [&](int /*n_blk*/) {};
        int nb = flash::init_block_cur<kInnerDir>(block_meta.inner_block_min, block_meta.inner_block_max);
        flash::mask_dispatch<kBlockM, kBlockN, PackGQA, QheadPerKhead, flash::DispatchAxis::N, kInnerDir>(
            nb,
            block_meta.inner_block_min,
            block_meta.inner_block_max,
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
        flash::mask_dispatch_unified<kBlockM, kBlockN, PackGQA, QheadPerKhead, flash::DispatchAxis::N, kInnerDir>(block_meta, mask, tSrS, thread_idx, bwd_step);
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

    // When LSE/dPsum are unioned with dKVacc, the loader will TMA-load new
    // LSE/dPsum into smem_dkacc for the next work tile. We must ensure the
    // store warp has finished TMA-storing the last dK (which also lives in
    // smem_dkacc) before the loader overwrites it. dVEmpty is signaled by
    // the store warp after each dK TMA completes, so an extra sync here
    // drains the last pending arrive. We then re-arrive on behalf of the
    // store warp so the next tile's first dVEmpty sync is pre-satisfied
    // (mirroring mma_init).
    if constexpr (LseDpsumUnionEffective && dKVacc_use_TMA && InnerDxStoreInProducer) {
      int const warp_idx_in_wg = canonical_warp_idx_in_warpgroup_sync();
      BarrierManager::sync<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);
      if (warp_idx_in_wg == 0 || (InnerUseScatter && warp_idx_in_wg == 1)) {
        BarrierManager::arrive<cutlass::NumThreadsPerWarpGroup + NumdKVStoreThreads>(BwdNamedBarriers::dVEmptyWG1, /*warp_group_idx=*/warp_group_idx);
      }
    }

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
