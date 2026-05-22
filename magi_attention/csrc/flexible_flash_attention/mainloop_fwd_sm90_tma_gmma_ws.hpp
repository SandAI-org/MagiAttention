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
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cute/tensor.hpp>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "block.h"
#include "block_meta.h"
#include "mask.h"
#include "named_barrier.hpp"
#include "seqlen.h"
#include "sm90_pipeline_no_cluster.hpp"
#include "utils.h"

namespace flash {

using namespace cute;
namespace gcd = cutlass::gemm::collective::detail;

template <
    int Stages,
    class ClusterShape_,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Has_softcap_,
    bool MmaPV_is_RS_,
    bool IntraWGOverlap_,
    bool RangeMerge_,
    bool PackGQA_,
    int QheadPerKhead_,
    bool SwapAB_,
    bool SparseLoad_,
    bool IndexAttn_>
struct CollectiveMainloopFwdSm90 {
  using ClusterShape = ClusterShape_;
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  using TMAClusterBarrier_t = cutlass::arch::ClusterTransactionBarrier::ValueType;

  // Sanity check
  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr int kStages = Stages;
  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool MmaPV_is_RS = MmaPV_is_RS_;
  static constexpr bool IntraWGOverlap = IntraWGOverlap_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr bool SwapAB = SwapAB_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr int QheadPerKhead = QheadPerKhead_;
  static constexpr bool SparseLoad = SparseLoad_;
  static constexpr bool IndexAttn = IndexAttn_;
  static_assert(!(SparseLoad && IndexAttn), "SparseLoad and IndexAttn cannot be enabled at the same time");

  // Get the block size and head dimension from the TileShapeMNK for code readability
  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  // when SwapAB == true, set the warp group overlap tileMMA size for kBlockM
  static constexpr int TileSize_kBlockM = kBlockM;
  // TileSize_kBlockM can be set as kBlockM/2 to enable two warp-group inter overlap, but now is disable because no gain.
  // static constexpr int TileSize_kBlockM = kBlockM == 8 ? kBlockM : kBlockM / 2;

  // TileShapeMNK for mma qv: kBlockM, kBlockN, kHeadDim
  // (kBlockM, kHeadDim) @ (kHeadDim, kBlockN) -> (kBlockM, kBlockN)
  using TileShape_MNK_QV = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

  // (kBlockN, kHeadDim) @ (kHeadDim, kBlockM) -> (kBlockN, kBlockM)
  using TileShape_MNK_SwapAB = Shape<Int<kBlockN>, Int<kBlockM>, Int<kHeadDim>>;

  using TileShape_MNK_SwapAB_OP_SELECT = Shape<Int<kBlockN>, Int<TileSize_kBlockM>, Int<kHeadDim>>;

  // TileShapeMNK for mma pv: kBlockM, kHeadDim, kBlockN
  // (kBlockM, kBlockN) @ (kBlockN, kHeadDim) -> (kBlockM, kHeadDim)
  using TileShape_MNK_PV = Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;

  // (kHeadDim, kBlockN) @ (kBlockN, kBlockM) -> (kHeadDim, kBlockM)
  using TileShape_MNK_PV_SwapAB = Shape<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;

  // TileShape_MNK_SwapAB_OP_SELECT use TileSize_kBlockM as n,
  // which use in tensor core ss_op_selector for inter warp group overlap
  // (splitting short q range when SwapAB is open).
  using TileShape_MNK_PV_SwapAB_OP_SELECT = Shape<Int<kHeadDim>, Int<TileSize_kBlockM>, Int<kBlockN>>;

  using TileShape_MNK_PV_Active = std::conditional_t<SwapAB, TileShape_MNK_PV_SwapAB, TileShape_MNK_PV>;

  // By default, we use TMA for Q and KV to get better performance
  static constexpr bool Use_TMA_Q = true;
  static constexpr bool Use_TMA_KV = !SparseLoad && !IndexAttn ? true : false;
  static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
  // NOTE: SwapAB + IndexAttn is now allowed; small kBlockM (8/16) uses SwapAB
  // while IndexAttn handles indirect KV loading via cp.async.

  // By default, V is always row-major
  static constexpr GMMA::Major MmaMajorV = GMMA::Major::MN;
  static constexpr GMMA::Major TmaMajorV = GMMA::Major::MN;

  using SeqlenInfo_t = flash::SeqlenInfo;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, PackGQA, QheadPerKhead>;

  // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
  // Leaving this option here for reference.
  static constexpr bool MmaQK_is_RS = false;

  // without sparse load, use one warp to produce Q and KV
  // with sparse load (SparseLoad or IndexAttn), use one warpgroup to produce KV with cp.async, use one thread to produce Q with TMA
  static constexpr int NumProducerThreads = !(SparseLoad || IndexAttn) ? cutlass::NumThreadsPerWarp : cutlass::NumThreadsPerWarpGroup;

  // Const parameters for IndexAttn
  // SMEM bank row width: 32 banks * 4 bytes = 128 bytes
  static constexpr int kCpAsyncTransactionBytes = 128;
  // A group of 8 threads load global memory together to form one memory transaction (8 * 16B = 128B)
  static constexpr int GroupSize = kCpAsyncTransactionBytes / 16; // 16B per cp.async instruction
  static constexpr int NumGroups = NumProducerThreads / GroupSize;
  // Number of rows (tokens) to load per group
  static constexpr int NumRowsPerGroup = kBlockN / NumGroups;
  static_assert(
      !IndexAttn || (NumRowsPerGroup <= GroupSize && kBlockN % NumGroups == 0),
      "Sparse KV requires kBlockN divisible by NumGroups and NumRowsPerGroup <= GroupSize");
  static_assert(
      !SparseLoad || (NumRowsPerGroup <= GroupSize && kBlockN % NumGroups == 0),
      "Sparse KV (SparseLoad) requires kBlockN divisible by NumGroups and NumRowsPerGroup <= GroupSize");

  using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;

  // warp group overlap pipeline
  using AtomLayoutQK_SwapAB = Layout<Shape<_1, Int<kBlockM / TileSize_kBlockM>, _1>>;

  // Use if constexpr to avoid instantiating the unused QK branch that can trigger static asserts.
  static constexpr auto make_tiled_mma_qk_active() {
    if constexpr (SwapAB) {
      // TiledMmaQK_SwapAB
      // Q @ K is always SS when SwapAB
      return cute::make_tiled_mma(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_SwapAB_OP_SELECT>(), AtomLayoutQK_SwapAB{});
    } else {
      // TiledMmaQK
      return cute::make_tiled_mma(
          std::conditional_t<
              !MmaQK_is_RS,
              decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
              decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>())>{},
          AtomLayoutQK{});
    }
  }

  using TiledMmaQK_Active = decltype(make_tiled_mma_qk_active());

  // Atom layout for PV is the same as QK
  using AtomLayoutPV = AtomLayoutQK;
  using AtomLayoutPV_SwapAB = AtomLayoutQK_SwapAB;
  // permutate V @ P, divide kHeadDim
  // (kHeadDim, kBlockN) @ (kBlockN, kBlockM) -> (kHeadDim, kBlockM)
  using PermutationPV_SwapAB = Tile<Int<kHeadDim>, Int<kBlockM>, Int<kBlockN>>;

  // Use if constexpr to avoid instantiating unused PV branches that can trigger static asserts
  static constexpr auto make_tiled_mma_pv_active() {
    if constexpr (SwapAB) {
      // TileShape_MNK_PV_SwapAB
      // V @ P is always SS when SwapAB
      return cute::make_tiled_mma(
          GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV_SwapAB_OP_SELECT, MmaMajorV, GMMA::Major::MN>(),
          AtomLayoutPV_SwapAB{},
          PermutationPV_SwapAB{});
    } else {
      // TileShape_MNK_PV
      return cute::make_tiled_mma(
          std::conditional_t<
              !MmaPV_is_RS,
              decltype(GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
              decltype(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())>{},
          AtomLayoutPV{});
    }
  }

  using TiledMmaPV_Active = decltype(make_tiled_mma_pv_active());

  // REVIEW: do we still need TiledMmaPV_RS any more ?
  // no use so note it down
  // using TiledMmaPV_RS =
  //     decltype(cute::make_tiled_mma(GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(), AtomLayoutPV{}));

  // do pv must be larger than qk or not ?
  static constexpr int NumMmaThreadsQK = size(TiledMmaQK_Active{});
  static constexpr int NumMmaThreads = size(TiledMmaPV_Active{});
  static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
  static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
  static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
  static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
  static_assert(BarrierManager::check<FwdNamedBarriers, NumMmaWarpGroups>());

  // Get the smem layout for Q
  using SmemLayoutAtomQ = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kHeadDim>>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{}))); // (kBlockM, kHeadDim)

  // Get the smem layout for K
  using SmemLayoutAtomK = decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockN>, Int<kHeadDim>>());
  using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{}))); // (kBlockN, kHeadDim, kStages)

  // Get the smem layout for V transpose
  using SmemLayoutAtomVt = decltype(gcd::ss_smem_selector<TmaMajorV, Element, Int<kHeadDim>, decltype(cute::get<2>(TileShape_MNK_PV_Active{}))>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV_Active{}), Int<kStages>{}), // (kHeadDim, kBlockN, kStages)
      std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // Get the smem layout for V transpose for mma
  using SmemLayoutAtomVtMma = decltype(gcd::ss_smem_selector<MmaMajorV, Element, Int<kHeadDim>, decltype(cute::get<2>(TileShape_MNK_PV_Active{}))>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV_Active{}), Int<kStages>{}),
      std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // Get the smem layout for P, used when MmaPV_is_RS is false
  using SmemLayoutAtomP = std::conditional_t<
      !SwapAB,
      decltype(gcd::ss_smem_selector<GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockN>>()),
      decltype(gcd::ss_smem_selector<GMMA::Major::MN, Element, Int<kBlockM>, Int<kBlockN>>())>;
  using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{}))); // (kBlockM, kBlockN)
  // use SM90_U32x2_STSM_N when TileSize_kBlockM == 8
  // because P matrix's TiledCopy needs enough vals for selected CopyAtom
  // TiledNumVal{} % AtomNumVal{} == 0
  using SmemCopyAtomP = std::conditional_t<TileSize_kBlockM == 8, Copy_Atom<cute::SM90_U32x2_STSM_N, Element>, Copy_Atom<cute::SM90_U32x4_STSM_N, Element>>;

  // Get TMA copy op for Q and KV
  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = decltype(gcd::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

  // Set the shape and stride for Q and KV
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen, head_dim, num_heads)
  using StrideQK = cute::Stride<int64_t, _1, int64_t>;
  using StrideV = StrideQK;

  using ShapeQPackedTMA = std::conditional_t<
      !PackGQA,
      ShapeQKV,
      cute::Shape<cute::Shape<cute::Int<QheadPerKhead>, int32_t>, int32_t, int32_t> // ((qhead_per_khead, seqlen), headdim, khead)
      >;
  using StrideQPackedTMA = std::conditional_t<
      !PackGQA,
      StrideQK,
      cute::Shape<cute::Shape<int64_t, int64_t>, _1, int64_t> // ((qhead_per_khead, seqlen), headdim, khead)
      >;

  using TMA_Q = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
      SmemLayoutQ{},
      TileShape_MNK{},
      ClusterShape{})); // no mcast for Q

  using TMA_Q_Packed = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQPackedTMA{}, StrideQPackedTMA{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      ClusterShape{}));

  using TMA_K = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along M mode for this N load, if any

  using TMA_V = decltype(make_tma_copy( // REVIEW: why not use `make_tma_copy_B_sm90` for V ?
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2>(StrideV{})),
      take<0, 2>(SmemLayoutVt{}),
      select<1, 2>(TileShape_MNK_PV{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * sizeof_bytes_v<Element>());
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * sizeof_bytes_v<Element>());
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{})) * sizeof_bytes_v<Element>());
  static_assert(TmaTransactionBytesK == TmaTransactionBytesV, "TmaTransactionBytesK must equal TmaTransactionBytesV");

  using PipelineTmaAsync =
      std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
  using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineV = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using PipelineState = cutlass::PipelineState<kStages>;

  // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
  // and have sQ being position_independent_swizzle_tensor.
  // If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
  static constexpr size_t SmemAlignmentQ = !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
  static constexpr size_t SmemAlignmentK = Use_TMA_KV ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutK{});
  static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
  static constexpr size_t maxSmemAlignmentWithoutP = cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose);
  static constexpr size_t maxSmemAlignmentWithP = cute::max(maxSmemAlignmentWithoutP, SmemAlignmentP);
  static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
  static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

  using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
  // Sometimes even with SmemP_t = cute::array<Element, 0>, putting it in the TensorStorage struct causes
  // smem size to go from 227KB to 228KB and we get "invalid argument".

  struct TensorStorageWithoutP : cute::aligned_struct<maxSmemAlignmentWithoutP, _0> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
  };

  struct TensorStorageWithP : cute::aligned_struct<maxSmemAlignmentWithP, _0> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
    SmemP_t smem_p;
  };

  using TensorStorage = std::conditional_t<MmaPV_is_RS, TensorStorageWithoutP, TensorStorageWithP>;

  static constexpr size_t SmemAlignmentVt = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static constexpr size_t SmemAlignmentV = cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
  static_assert(SmemAlignmentVt >= 128 and SmemAlignmentV >= 128, "Require at least 128B alignment");

  // These are tuned for speed. They don't affect correctness.
  // UseSchedulerBarrier can let multiple warp groups launch tensors in order
  static constexpr bool UseSchedulerBarrier = (IntraWGOverlap ? (NumMmaWarpGroups >= 2) && (kHeadDim <= 128) : NumMmaWarpGroups == 2);
  static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 && IntraWGOverlap;

  // Host side kernel arguments
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    Element* const ptr_K; // not Element const* since we might append to KV cache in-place
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim;
    StrideV const stride_V;
    float const softmax_scale;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;
    int const* const cu_batches;
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
    int const* const index_attn_indices;
    int const index_attn_max_topk;
  };

  // Device side kernel params
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    ShapeQPackedTMA const shape_Q_packed;
    StrideQK const stride_Q;
    StrideQPackedTMA const stride_Q_packed;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    int32_t const headdim;
    StrideV const stride_V;
    cutlass::FastDivmod qhead_per_khead_divmod;
    TMA_Q tma_load_Q;
    TMA_Q_Packed tma_load_Q_packed;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    float const softmax_scale_log2;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;
    int const* const cu_batches;
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
    int const* const index_attn_indices;
    int const index_attn_max_topk;
  };

  // BlockMeta type aliases — definitions live in block_meta.h
  template <bool IsProducer>
  using BlockMeta = flash::DenseBlockMeta<IsProducer, /*InnerLoopQ=*/false, RangeMerge, /*FlattenGQA=*/PackGQA, QheadPerKhead, SeqlenInfo_t, BlockMN_t>;

  // SparseLoad producer (used by load)
  using SparseLoadBlockMeta = flash::
      SparseLoadBlockMeta</*IsProducer=*/true, RangeMerge, PackGQA, QheadPerKhead, SeqlenInfo_t, NumRowsPerGroup, NumGroups, GroupSize, NumProducerThreads, kBlockN>;

  // SparseLoad consumer (used by mma), replaces old SparseMmaBlockMeta
  using SparseMmaBlockMeta = flash::
      SparseLoadBlockMeta</*IsProducer=*/false, RangeMerge, PackGQA, QheadPerKhead, SeqlenInfo_t, NumRowsPerGroup, NumGroups, GroupSize, NumProducerThreads, kBlockN>;

  template <bool IsProducer>
  using IndexAttnBlockMeta = flash::IndexAttnBlockMeta<IsProducer, RangeMerge, PackGQA, QheadPerKhead, NumRowsPerGroup, NumProducerThreads, GroupSize, kBlockN>;

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{});
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K = make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{});
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), make_shape(args.headdim, get<0>(args.shape_K), get<2>(args.shape_K)), select<1, 0, 2>(args.stride_V));
    TMA_V tma_load_V = make_tma_copy(GmemTiledCopyKV{}, mV, take<0, 2>(SmemLayoutVt{}), select<1, 2>(TileShape_MNK_PV{}), size<0>(ClusterShape{}));

    auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
        args.shape_Q,
        make_shape(
            make_shape(cute::Int<QheadPerKhead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.shape_Q), // headdim
            get<2>(args.shape_K) // numhead_k
            ));

    auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
        args.stride_Q,
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.stride_Q), // headdim
            get<2>(args.stride_Q) * QheadPerKhead));

    auto mQPacked = [&]() {
      if constexpr (!PackGQA) {
        return mQ;
      } else {
        return make_tensor(
            make_gmem_ptr(args.ptr_Q),
            make_layout(
                make_shape(
                    make_shape(cute::Int<QheadPerKhead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
                    get<1>(args.shape_Q), // headdim
                    get<2>(args.shape_K) // numhead_k
                    ),
                stride_Q_packed));
      }
    }();

    TMA_Q_Packed tma_load_Q_packed = make_tma_copy(GmemTiledCopyQ{}, mQPacked, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), ClusterShape{});

    // If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
    // Right after this, we multiply by log2(e) before applying exp2.
    // To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
    // (assigning it to params.softcap_val) and pre-multiply softcap_val * log2(e)
    // (assigning it to params.softmax_scale_log2).
    return {
        args.ptr_Q,
        args.shape_Q,
        shape_Q_packed,
        args.stride_Q,
        stride_Q_packed,
        args.ptr_K,
        args.shape_K,
        args.stride_K,
        args.ptr_V,
        args.headdim,
        args.stride_V,
        /*qhead_per_khead_divmod=*/cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
        tma_load_Q,
        tma_load_Q_packed,
        tma_load_K,
        tma_load_V,
        /*softmax_scale_log2=*/!Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        /*softcap_val=*/!Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.q_ranges,
        args.k_ranges,
        args.attn_type_map,
        args.cu_batches,
        args.sparse_load_loop_count,
        args.sparse_load_invalid_count,
        args.equal_k_range_size,
        args.index_attn_indices,
        args.index_attn_max_topk};
  }

  // Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (Use_TMA_Q) {
      if constexpr (!PackGQA)
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
      else
        cute::prefetch_tma_descriptor(params.tma_load_Q_packed.get_tma_descriptor());
    }
    if constexpr (Use_TMA_KV) {
      cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
    }
  }

  template <typename SchedulerPrefetch, typename SharedStorage, typename BlockMetaT>
  CUTLASS_DEVICE bool load(
      Params const& params,
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_v,
      SharedStorage& shared_storage,
      SchedulerPrefetch const& scheduler_prefetch,
      BlockMetaT& block_meta,
      int& work_idx,
      int const thread_idx = 0) {
    // If this is true, we're guaranteed that only the first warp will execute this function
    static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
    // get thread_idx in the producer thread group
    // int const thread_idx = threadIdx.x % NumProducerThreads;

    // prepare for TMA multicast meta
    auto [mcast_mask_kv, cluster_block_id_kv] = get_tma_multi_cast_meta<ClusterShape, GmemTiledCopyKV, /*RowwiseMask=*/true>();

    int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
    auto is_tma_issue_thread = [&]() { return (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync(); };

    // as_position_independent_swizzle_tensor makes address calculation easier when we do LDSM & STSM to transpose.
    // But it requires smem_vt and smem_v to be aligned to e.g 512 bytes.
    // get thread_idx in the producer thread group
    // int const thread_idx = threadIdx.x % NumProducerThreads;
    // Only one thread in one warp within a warp group needs to issue the TMA load instruction

    // ─── Load Q (TMA, shared by both paths) ───
    // Define lambda funcs to load Q,K,V
    auto load_Q = [&]() {
      Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, block_meta.bidh); // (seqlen_q, head_dim)
      Tensor mQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return params.tma_load_Q_packed.get_tma_tensor(params.shape_Q_packed)(_, _, block_meta.bidh);
        } else {
          return mQ;
        }
      }();

      Tensor gQ = local_tile(
          domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(block_meta.m_block, _0{})); // (M, K)
      Tensor gQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return local_tile(
              domain_offset(
                  make_coord(block_meta.seqlen_info.offset_q * QheadPerKhead, _0{}),
                  mQ_Packed), // for packgqa, we need multiple qhead_per_khead for offset of seqlen;
              select<0, 2>(TileShape_MNK{}),
              make_coord(block_meta.m_block, _0{})); // (M // qhead_per_khead, K, qhead_per_khead)
        } else {
          return gQ;
        }
      }();

      // NOTE: tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
      auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
      auto block_tma_Q_Packed = params.tma_load_Q_packed.get_slice(_0{});
      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)
      Tensor tQgQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return group_modes<0, 3>(block_tma_Q_Packed.partition_S(gQ_Packed));
        } else {
          return tQgQ;
        }
      }();
      Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
      Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)

      if constexpr (Use_TMA_Q) {
        // Wait for the MMA warpgroups to signal that smem_q is ready
        if constexpr (!SparseLoad && !IndexAttn) {
          if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
            BarrierManager::sync<NumMmaThreadsQK + cutlass::NumThreadsPerWarp>(FwdNamedBarriers::QueryEmpty);
          }
        } else {
          BarrierManager::sync<NumMmaThreadsQK + NumProducerThreads>(FwdNamedBarriers::QueryEmpty);
        }

        if (is_tma_issue_thread()) {
          auto& barrier_Q = reinterpret_cast<TMAClusterBarrier_t&>(shared_storage.pipelines.barrier_Q);
          shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);

          if constexpr (PackGQA) {
            auto tma_desc = params.tma_load_Q_packed.with(
                reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q),
                /*mcast_mask=*/0,
                TMA::CacheHintSm90::EVICT_FIRST);
            copy(tma_desc, tQgQ_Packed, tQsQ);
          } else {
            auto tma_desc = params.tma_load_Q.with(
                reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q),
                /*mcast_mask=*/0,
                TMA::CacheHintSm90::EVICT_FIRST);
            copy(tma_desc, tQgQ, tQsQ);
          }
        }
      }
    };

    // ─── Define K/V load lambdas for both paths ───

    // SparseLoad/IndexAttn: cp.async scatter-load (reads token_indices from block_meta)
    auto load_K_scatter = [&]() {
      if constexpr (SparseLoad || IndexAttn) {
        int64_t cache_policy = createpolicy_evict_last();
        int num_tiles = kHeadDim * sizeof(Element) / kCpAsyncTransactionBytes;
        int idx_in_warpgroup = threadIdx.x % NumProducerThreads;
        int idx_in_group = idx_in_warpgroup % GroupSize;
        int group_idx = idx_in_warpgroup / GroupSize;
        int stride_kv = get<0>(params.stride_K);

        pipeline_k.producer_acquire(smem_pipe_write_k);
        Element* ptr_gK_base = params.ptr_K + block_meta.bidh_kv * get<2>(params.stride_K) + idx_in_group * 8;
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});

        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int token_offset = block_meta.token_indices[local_row] * stride_kv;
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            Element* dst_ptr = &sK(group_idx * NumRowsPerGroup + local_row, idx_in_group * 8 + tile_idx * 64, smem_pipe_write_k.index());
            cp_async_cacheglobal_l2_prefetch_256B(ptr_gK_base + token_offset + tile_idx * 64, dst_ptr, true, cache_policy);
          }
        }
        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;
      }
    };

    auto load_V_scatter = [&]() {
      if constexpr (SparseLoad || IndexAttn) {
        int64_t cache_policy = createpolicy_evict_last();
        int num_tiles = kHeadDim * sizeof(Element) / kCpAsyncTransactionBytes;
        int idx_in_warpgroup = threadIdx.x % NumProducerThreads;
        int idx_in_group = idx_in_warpgroup % GroupSize;
        int group_idx = idx_in_warpgroup / GroupSize;
        int stride_kv = get<0>(params.stride_V);

        pipeline_v.producer_acquire(smem_pipe_write_v);
        Element* ptr_gV_base = params.ptr_V + block_meta.bidh_kv * get<2>(params.stride_V) + idx_in_group * 8;
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

        CUTE_UNROLL
        for (int local_row = 0; local_row < NumRowsPerGroup; ++local_row) {
          int token_offset = block_meta.prev_token_indices[local_row] * stride_kv;
          CUTE_UNROLL
          for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            Element* dst_ptr = &sVt(idx_in_group * 8 + tile_idx * 64, group_idx * NumRowsPerGroup + local_row, smem_pipe_write_v.index());
            cp_async_cacheglobal_l2_prefetch_256B(ptr_gV_base + token_offset + tile_idx * 64, dst_ptr, true, cache_policy);
          }
        }
        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      }
    };

    // Dense: TMA load K/V (only called from Dense path below, never from SparseLoad/IndexAttn)
    auto load_K = [&, mcast_mask_kv = mcast_mask_kv, cluster_block_id_kv = cluster_block_id_kv](int const n_block_idx, int const offset_k) {
      Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, block_meta.bidh_kv);
      Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
      Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});

      auto block_tma_K = params.tma_load_K.get_slice(cluster_block_id_kv);
      Tensor tKgK = group_modes<0, 3>(block_tma_K.partition_S(gK));
      Tensor tKsK = group_modes<0, 3>(block_tma_K.partition_D(sK));

      if (is_tma_issue_thread()) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(
            params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tKgK(_, n_block_idx),
            tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
      }
    };

    auto load_V = [&, mcast_mask_kv = mcast_mask_kv, cluster_block_id_kv = cluster_block_id_kv](int const n_block_idx, int const offset_k) {
      auto shape_Vt = make_shape(params.headdim, get<0>(params.shape_K), get<2>(params.shape_K));

      Tensor mVt = params.tma_load_V.get_tma_tensor(shape_Vt)(_, _, block_meta.bidh_kv);
      Tensor gVt = local_tile(domain_offset(make_coord(_0{}, offset_k), mVt), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _));
      Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

      auto block_tma_Vt = params.tma_load_V.get_slice(cluster_block_id_kv);
      Tensor tVgVt = group_modes<0, 3>(block_tma_Vt.partition_S(gVt));
      Tensor tVsVt = group_modes<0, 3>(block_tma_Vt.partition_D(sVt));

      if (is_tma_issue_thread()) {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(
            params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tVgVt(_, n_block_idx),
            tVsVt(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    };

    // ─── First-batch-only prologue lambda ───
    auto load_Q_and_wait_barrier_O = [&]() {
      load_Q();
      shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    };

    // ─── SparseLoad / IndexAttn path ───
    if constexpr (SparseLoad || IndexAttn) {
      static_assert(IntraWGOverlap, "SparseLoad/IndexAttn FWD load requires IntraWGOverlap=true");

      bool is_first_batch = true;
      while (true) {
        block_meta.skip_to_first_valid();
        if (block_meta.is_finish())
          break;

        if (is_first_batch) {
          // Prologue: load first K block and Q
          load_K_scatter();
          load_Q_and_wait_barrier_O();
          is_first_batch = false;
        }

        // Prefetch advances cur_loop and updates token_indices/prev_token_indices
        int n_block_max = block_meta.loop_count;
        while (true) {
          block_meta.prefetch();
          int n_block = block_meta.cur_loop;
          if (n_block < n_block_max) {
            load_K_scatter();
            load_V_scatter();
          } else {
            break;
          }
        }
      }

      // Epilogue: load the last V block
      if (!is_first_batch) {
        load_V_scatter();
      }
      return !is_first_batch;
    }

    // ─── Dense path ───

    int n_block, n_block_min, offset_k;
    int prev_n_block, prev_offset_k;

    bool is_first_batch = true;
    while (true) {
      block_meta.skip_to_first_valid();
      if (block_meta.is_finish())
        break;

      if (is_first_batch) {
        load_Q_and_wait_barrier_O();
        is_first_batch = false;
      } else if constexpr (IntraWGOverlap) {
        // Cross-batch V: the previous batch's last V was deferred (IntraWGOverlap
        // loads V one step behind K). Emit it now before starting new batch's K.
        load_V(prev_n_block, prev_offset_k);
      }

      // Read per-batch variables
      n_block = block_meta.n_block_max - 1;
      offset_k = block_meta.seqlen_info.offset_k;
      n_block_min = block_meta.n_block_min;

      // Load first K (+ V if !IntraWGOverlap) for this batch
      load_K(n_block, offset_k);
      if constexpr (!IntraWGOverlap) {
        load_V(n_block, offset_k);
      }
      prev_n_block = n_block;
      prev_offset_k = offset_k;
      --n_block;

      // Inner n_block loop (right-to-left)
#pragma unroll(Use_TMA_KV ? 2 : 1)
      while (n_block >= n_block_min) {
        if constexpr (IntraWGOverlap) {
          load_K(n_block, offset_k);
          load_V(prev_n_block, prev_offset_k);
        } else {
          load_K(n_block, offset_k);
          load_V(n_block, offset_k);
        }
        prev_n_block = n_block;
        prev_offset_k = offset_k;
        --n_block;
      }

      block_meta.prefetch();
    }

    // Epilogue: load last V (IntraWGOverlap only)
    if constexpr (IntraWGOverlap) {
      if (!is_first_batch) {
        load_V(prev_n_block, prev_offset_k);
      }
    }

    return !is_first_batch;
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load_tail(
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_write_k,
      PipelineState& smem_pipe_write_v,
      SharedStorage& shared_storage,
      int const work_idx) {
    // If we don't wait for barrier_O here, when using Cluster, CTA0 might exit early and CTA1 will
    // try to arrive on barrier_O of CTA0, causing "unspecified launch failure".
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    if (!(SparseLoad || IndexAttn)) {
      int warp_idx_in_warpgroup = canonical_warp_idx_in_warpgroup_sync();
      // Issue the epilogue waits
      // TODO: check if this should be called by 1 thread or more
      if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
        /* This helps avoid early exit of blocks in Cluster
         *  Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
         *  then would just be acquired since the phase was still inverted from make_producer_start_state
         */
        pipeline_k.producer_tail(smem_pipe_write_k);
        pipeline_v.producer_tail(smem_pipe_write_v);
      }
    } else {
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      // Get the current mma warp group index
      // -1 is because one warp group is the producer
      int const curr_WG = flash::canonical_warp_group_idx_nosync() - 1;

      // Sync on the current mma warp group's named barrier
      BarrierManager::sync<2 * cutlass::NumThreadsPerWarpGroup>(FwdNamedBarriers::WarpSchedulerWG1, /*warp_group_idx=*/curr_WG);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    if constexpr (UseSchedulerBarrier) {
      // We have NamedBarrier for up to 3 WGs and 2 WGs is the minimum
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

      // Get the current mma warp group index
      int const curr_WG = flash::canonical_warp_group_idx_nosync() - 1;

      // Get the next mma warp group index
      // If there are 2 mma warp groups: the next mma warp group index is 1 - curr_WG
      // If there are 3 mma warp groups:
      //   if curr_WG is 0, the next mma warp group index is 1
      //   if curr_WG is 1, the next mma warp group index is 2
      //   if curr_WG is 2, the next mma warp group index is 0
      int const next_WG = NumMmaWarpGroups == 2 ? 1 - curr_WG : (curr_WG < NumMmaWarpGroups - 1 ? curr_WG + 1 : 0);

      // Arrive on the next mma warp group's named barrier
      BarrierManager::arrive<2 * cutlass::NumThreadsPerWarpGroup>(FwdNamedBarriers::WarpSchedulerWG1, /*warp_group_idx=*/next_WG);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Get the current warp group index, since one warp group is producer, the warp group index for mma starts from 1
    int warp_group_idx = flash::canonical_warp_group_idx_nosync();

    // Tell producers that smem_q is ready to be loaded
    if constexpr (!(SparseLoad || IndexAttn)) {
      BarrierManager::arrive<NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads)>(FwdNamedBarriers::QueryEmpty);
    } else {
      BarrierManager::arrive<NumMmaThreadsQK + NumProducerThreads>(FwdNamedBarriers::QueryEmpty);
    }

    if constexpr (UseSchedulerBarrier) {
      // We have NamedBarrier for up to 3 WGs (why 3 WGs ?)
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

      // WG1 is the smallest warp group used for mma, so it needs the very first signal to start
      if (warp_group_idx == 1) {
        BarrierManager::arrive<2 * cutlass::NumThreadsPerWarpGroup>(FwdNamedBarriers::WarpSchedulerWG1);
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename Softmax, typename ScoresScale, typename BlockMetaT>
  CUTLASS_DEVICE bool mma(
      Params const& params,
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_read_k,
      PipelineState& smem_pipe_read_v,
      FrgTensorO& tOrO,
      Softmax& softmax,
      ScoresScale& scores_scale,
      int const thread_idx,
      int& work_idx,
      BlockMetaT& block_meta,
      SharedStorage& shared_storage) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    static constexpr int kBlockM = CollectiveMainloopFwdSm90::kBlockM;
    static constexpr int kBlockN = CollectiveMainloopFwdSm90::kBlockN;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVtMma{});
    Tensor sP = [&] {
      if constexpr (MmaPV_is_RS) {
        // We might not have smem_p if !MmaPV_is_RS, just use smem_q as a placeholder since we don't use it
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutP{});
      } else {
        return make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()), SmemLayoutP{});
      }
    }();

    TiledMmaQK_Active tiled_mma_qk;
    TiledMmaPV_Active tiled_mma_pv;

    if constexpr (!MmaQK_is_RS) {
      static_assert(
          stride<0>(typename TiledMmaQK_Active::ALayout{}) == 0 and stride<0>(typename TiledMmaQK_Active::BLayout{}) == 0 and
              size<0>(typename TiledMmaQK_Active::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
              size<0>(typename TiledMmaQK_Active::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
          "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    }

    static constexpr int MmaWarpGroups = size(TiledMmaPV_Active{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    // Get the mma warp group index of the current thread, start from 0
    int warp_group_idx = warp_uniform(thread_idx / cutlass::NumThreadsPerWarpGroup);

    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors"
    auto tSrQ = [&]() {
      if constexpr (!SwapAB) {
        return wg_mma_qk.partition_fragment_A(sQ);
      } else {
        return wg_mma_qk.partition_fragment_B(sQ);
      }
    }();
    auto tSrK = [&]() {
      if constexpr (!SwapAB) {
        return wg_mma_qk.partition_fragment_B(sK);
      } else {
        return wg_mma_qk.partition_fragment_A(sK);
      }
    }();
    Tensor tOrV = [&]() {
      if constexpr (!SwapAB) {
        return wg_mma_pv.partition_fragment_B(sV);
      } else {
        return wg_mma_pv.partition_fragment_A(sV);
      }
    }();
    Tensor tOsP = [&]() {
      if constexpr (!SwapAB) {
        return wg_mma_pv.partition_fragment_A(sP);
      } else {
        return wg_mma_pv.partition_fragment_B(sP);
      }
    }();
    // if p is in registers, do we still need this step ?
    Tensor tPsP = [&]() {
      if constexpr (!SwapAB) {
        // Normal mode: keep original tensor construction logic
        return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));
      } else {
        // SwapAB mode: transpose sP layout to enable transposed write when tOrP is written to tPsP
        // sP is a shared memory tensor with layout (kBlockN, kBlockM), we need to transpose it to (kBlockM, kBlockN)
        auto sP_transposed = make_tensor(
            sP.data(),
            cute::make_layout(
                cute::make_shape(get<1>(sP.layout().shape()), get<0>(sP.layout().shape())),
                cute::make_stride(get<1>(sP.layout().stride()), get<0>(sP.layout().stride()))));
        return smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP_transposed));
      }
    }();

    // Allocate S(Q@K) fragment
    Tensor tSrS = [&]() {
      if constexpr (!SwapAB) {
        return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
      } else {
        return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK_SwapAB{}));
      }
    }();

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    auto consumer_release = [](auto& pipeline, auto& smem_pipe_read) {
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    };

    // Softcapping needs to happen before masking since if we apply after masking, softcapping
    // can turn -inf to e.g. -50.0, which can affect the attention softmax.
    auto scoremod_premask_fn = [&](auto& tSrS) {
      if constexpr (Has_softcap) {
        flash::apply_softcap(tSrS, params.softcap_val);
      }
    };

    auto write_P_to_smem = [&](auto& tOrP) { cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP); };

    auto arrive_on_P_write_barrier = [&] {
      cutlass::arch::fence_view_async_shared();
      __syncwarp(); // Only need syncwarp since each warp is using its own P values for MmaPV
    };

    auto& barrier_Q = shared_storage.pipelines.barrier_Q;

    if constexpr (MmaQK_is_RS) {
      // MmaQK_is_RS is always false, so we never enter this branch
      using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(cute::as_position_independent_swizzle_tensor(sQ));
      cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }

    flash::Mask<kBlockM, kBlockN, TiledMmaQK_Active, SwapAB> mask;

    // Dense-path mask functions (compiled away when SparseLoad/IndexAttn=true)
    auto boundary_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {
      mask.template apply</*Seqlenk_mask=*/true, PackGQA, QheadPerKhead>(tSrS, block_meta.m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
    };
    auto no_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) { /*do nothing*/ };
    auto regular_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {
      mask.template apply</*Seqlenk_mask=*/false, PackGQA, QheadPerKhead>(tSrS, block_meta.m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
    };

    int n_block_max, n_block, seqlen_k, n_block_min;
    flash::AttnType attn_type;

    Tensor tOrP = [&]() {
      if constexpr (TileSize_kBlockM == 8) {
        return make_tensor_like<Element>(make_tensor(tSrS.data(), tSrS.layout()));
      } else {
        return make_tensor_like<Element>(make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV_Active>(tSrS.layout())));
      }
    }();

    auto gemm_pv = [&]() {
      if constexpr (!SwapAB) {
        if constexpr (MmaPV_is_RS) {
          flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        } else {
          flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOsP, tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
        }
      } else {
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, tOrV(_, _, _, smem_pipe_read_v.index()), tOsP, tOrO);
      }
    };

    auto mma_head = [&](int const n_block, auto const& attn_type, int const& seqlen_k, bool is_first_ever) {
      consumer_wait(pipeline_k, smem_pipe_read_k);
      if constexpr (!SwapAB) {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      } else {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrK(_, _, _, smem_pipe_read_k.index()), tSrQ, tSrS);
      }
      warpgroup_wait<0>();
      consumer_release(pipeline_k, smem_pipe_read_k);

      scoremod_premask_fn(tSrS);

      if constexpr (SparseLoad || IndexAttn) {
        mask.apply_padding_mask(tSrS, block_meta.num_invalid_token, thread_idx);
      } else {
        boundary_mask_fn(tSrS, n_block, attn_type, block_meta.seqlen_info.seqlen_q, seqlen_k);
      }

      if (is_first_ever) {
        cute::copy(softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true, NumMmaWarpGroups>(tSrS), scores_scale);
        softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
      } else {
        cute::copy(softmax.template max_get_scale</*Is_first=*/false, /*Check_inf=*/true, NumMmaWarpGroups>(tSrS), scores_scale);
        softmax.template online_softmax</*Is_first=*/false, /*Check_inf=*/true>(tSrS);
        if constexpr (!RescaleOBeforeGemm) {
          softmax.rescale_o(tOrO, scores_scale);
        }
      }

      if constexpr (TileSize_kBlockM == 8) {
        Tensor tOrP_acc = make_tensor(tSrS.data(), tSrS.layout());
        flash::convert_type_out(tOrP_acc, tOrP);
      } else {
        Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV_Active>(tSrS.layout()));
        flash::convert_type_out(tOrP_acc, tOrP);
      }

      if constexpr (!MmaPV_is_RS) {
        write_P_to_smem(tOrP);
        arrive_on_P_write_barrier();
      }
    };

    auto mma_tail = [&]() {
      if constexpr (RescaleOBeforeGemm) {
        softmax.rescale_o(tOrO, scores_scale);
      }

      consumer_wait(pipeline_v, smem_pipe_read_v);
      gemm_pv();

      cute::copy(softmax.template finalize<NumMmaWarpGroups>(), scores_scale);

      warpgroup_wait<0>();
      consumer_release(pipeline_v, smem_pipe_read_v);
      ++work_idx;

      softmax.rescale_o(tOrO, scores_scale);
    };

    auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
      static constexpr bool Check_inf = decltype(check_inf_type)::value;

      Tensor tSrS = [&]() {
        if constexpr (!SwapAB) {
          return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
        } else {
          return partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK_SwapAB{}));
        }
      }();

      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_k, smem_pipe_read_k);
      }

      warp_scheduler_barrier_sync();

      if constexpr (!SwapAB) {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      } else {
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrK(_, _, _, smem_pipe_read_k.index()), tSrQ, tSrS);
      }

      if constexpr (RescaleOBeforeGemm) {
        softmax.rescale_o(tOrO, scores_scale);
      }

      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_v, smem_pipe_read_v);
      }

      gemm_pv();

      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      consumer_release(pipeline_k, smem_pipe_read_k);

      scoremod_premask_fn(tSrS);

      // For Dense, mask_fn is boundary/regular/no_mask depending on causal partition.
      // For SparseLoad/IndexAttn, callers always pass no_mask_fn (padding is
      // handled once in mma_head via apply_padding_mask on the last physical block).
      mask_fn(tSrS, n_block, attn_type, block_meta.seqlen_info.seqlen_q, seqlen_k);

      cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf, NumMmaWarpGroups>(tSrS), scores_scale);
      softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);

      warpgroup_wait<0>();
      consumer_release(pipeline_v, smem_pipe_read_v);

      convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);

      if constexpr (!MmaPV_is_RS) {
        write_P_to_smem(tOrP);
      }

      if constexpr (!RescaleOBeforeGemm) {
        softmax.rescale_o(tOrO, scores_scale);
      }

      if constexpr (!MmaPV_is_RS) {
        arrive_on_P_write_barrier();
      }
    };

    bool is_first_batch = true;
    while (true) {
      block_meta.skip_to_first_valid();
      if (block_meta.is_finish())
        break;

      n_block_max = block_meta.n_block_max;
      n_block = n_block_max - 1;
      seqlen_k = block_meta.seqlen_info.seqlen_k;
      n_block_min = block_meta.n_block_min;
      attn_type = block_meta.attn_type;

      if (is_first_batch) {
        barrier_Q.wait(work_idx % 2);
        mma_head(n_block, attn_type, seqlen_k, /*is_first_ever=*/true);
        --n_block;
        is_first_batch = false;
      }

      if constexpr (SparseLoad || IndexAttn) {
        while (true) {
          block_meta.prefetch();
          if (block_meta.is_finish())
            break;
          n_block = n_block_max - 1 - block_meta.cur_loop;
          if (n_block >= n_block_min) {
            fwd_step(n_block, no_mask_fn, cute::true_type{} /*check_inf*/);
          }
        }
      } else {
        flash::apply_causal_partition<kBlockM, kBlockN>(
            n_block,
            n_block_min,
            block_meta.m_block,
            block_meta.seqlen_info.seqlen_q,
            seqlen_k,
            attn_type,
            [&](int nb, auto mask_fn, auto is_no_mask) {
              using CheckInf = std::conditional_t<decltype(is_no_mask)::value, cute::false_type, cute::true_type>;
              fwd_step(nb, mask_fn, CheckInf{});
            },
            boundary_mask_fn,
            regular_mask_fn,
            no_mask_fn);

        block_meta.prefetch();
      }
    }

    if (is_first_batch) {
      return false;
    }

    if constexpr (!SparseLoad && !IndexAttn) {
      BarrierManager::arrive<NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads)>(FwdNamedBarriers::QueryEmpty);
    } else {
      BarrierManager::arrive<NumMmaThreadsQK + NumProducerThreads>(FwdNamedBarriers::QueryEmpty);
    }

    mma_tail();

    return true;
  }
};

} // namespace flash
