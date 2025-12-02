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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cute/tensor.hpp>

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "block.h"
#include "mask.h"
#include "named_barrier.hpp"
#include "seqlen.h"
#include "sm90_pipeline_no_cluster.hpp"
#include "utils.h"

#define BLOCK_SIZE 256

namespace flash {

using namespace cute;

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
    int Qhead_per_khead_>
struct CollectiveMainloopFwdSm90 {
  static constexpr int kStages = Stages;
  using ClusterShape = ClusterShape_;

  // kBlockM, kBlockN, kHeadDim
  using TileShape_MNK = TileShape_MNK_;
  // (kBlockM, kHeadDim) @ (kHeadDim, kBlockN) -> (kBlockM, kBlockN)
  using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), decltype(get<2>(TileShape_MNK{}))>;

  // TileShapeMNK for mma pv: kBlockM, kHeadDim, kBlockN
  // (kBlockM, kBlockN) @ (kBlockN, kHeadDim) -> (kBlockM, kHeadDim)
  using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<2>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{}))>;

  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool MmaPV_is_RS = MmaPV_is_RS_;
  static constexpr bool IntraWGOverlap = IntraWGOverlap_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr int qhead_per_khead = Qhead_per_khead_;

  // By default, we use TMA for Q and KV to get better performance
  static constexpr bool Use_TMA_Q = true;
  static constexpr bool Use_TMA_KV = true;

  // Sanity check
  static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // By default, V is always row-major
  static constexpr cute::GMMA::Major MmaMajorV = GMMA::Major::MN;
  static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::MN;

  // Get the block size and head dimension from the TileShapeMNK for code readability
  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  using SeqlenInfo_t = flash::DistributedSeqlenInfo;
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN>;

  // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
  // Leaving this option here for reference.
  static constexpr bool MmaQK_is_RS = false;
  using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      std::conditional_t<
          !MmaQK_is_RS,
          decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
          decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>())>{},
      AtomLayoutQK{}));

  // Atom layout for PV is the same as QK
  using AtomLayoutPV = AtomLayoutQK;
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      std::conditional_t<
          !MmaPV_is_RS,
          decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>()),
          decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>())>{},
      AtomLayoutPV{}));

  // REVIEW: do we still need TiledMmaPV_RS any more ?
  using TiledMmaPV_RS =
      decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(), AtomLayoutPV{}));

  // do pv must be larger than qk or not ?
  static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
  static constexpr int NumMmaThreads = size(TiledMmaPV{});
  // use one warp to produce Q and KV
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;
  static_assert(NumMmaThreadsQK % cutlass::NumThreadsPerWarpGroup == 0);
  static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
  static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
  // in which case should we use 3 warp groups ?
  static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

  // Get the smem layout for Q
  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::
                                       ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{}))); // kBlockM, kHeadim

  // Get the smem layout for K
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::
                                       ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtomK{}, make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{}))); // kBlockN, kHeadDim, kStages

  // Get the smem layout for V transpose
  using SmemLayoutAtomVt =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element, Int<kHeadDim>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}), // kHeadDim, kBlockN, kStages
      std::conditional_t<TmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // Get the smem layout for V transpose for mma?????? wtf
  using SmemLayoutAtomVtMma =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element, Int<kHeadDim>, decltype(cute::get<2>(TileShape_MNK_PV{}))>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(Int<kHeadDim>{}, shape<2>(TileShape_MNK_PV{}), Int<kStages>{}),
      std::conditional_t<MmaMajorV == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

  // Get the smem layout for P, used when MmaPV_is_RS is false
  using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::
                                       ss_smem_selector<GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));
  using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

  // Get TMA copy op for Q and KV
  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

  // Set the shape and stride for Q and KV
  using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t>; // (seqlen, head_dim, num_heads)
  // using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<int32_t, int32_t, int32_t, int32_t>>;
  // using ShapeQPacked = make_shape(make_shape(_4{}, int32_t{}), int32_t{}, int32_t{});
  using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<int32_t, int32_t, int32_t, int32_t>>; // (seqlen, headdim, khead, qhead_per_khead)
  /*using ShapeQPackedTMA = std::conditional_t<
    !PackGQA,
    ShapeQKV,
    cute::Shape<cute::Shape<cute::get<3>(ShapeQPacked), cute::get<0>(ShapeQPacked)>, cute::get<1>(ShapeQPacked), cute::get<2>(ShapeQPacked)> // ((qhead_per_khead,
  seqlen), headdim, khead)
  >;*/
  using ShapeQPackedTMA = std::conditional_t<
      !PackGQA,
      ShapeQKV,
      cute::Shape<cute::Shape<cute::Int<qhead_per_khead>, int32_t>, int32_t, int32_t> // ((qhead_per_khead, seqlen), headdim, khead)
      >;
  using StrideQK = cute::Stride<int64_t, _1, int64_t>;
  using StrideV = StrideQK;
  using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Shape<int64_t, _1, int64_t, int64_t>>; // (seqlen, headdim, khead, qhead_per_khead)
  /*using StrideQPackedTMA = std::conditional_t<
    !PackGQA,
    StrideQK,
    cute::Shape<cute::Shape<get<3>(StrideQPacked), get<0>(StridedQPacked)>, get<1>(StrideQPacked), get<2>(StrideQPacked)> // ((qhead_per_khead, seqlen), headdim, khead)
  >;*/
  using StrideQPackedTMA = std::conditional_t<
      !PackGQA,
      StrideQK,
      cute::Shape<cute::Shape<int64_t, int64_t>, _1, int64_t> // ((qhead_per_khead, seqlen), headdim, khead)
      >;
  // using TileShape_mNK = make_shape(cute::get<0>(TileShape_MNK{}) / ())

  using TMA_Q = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
      SmemLayoutQ{},
      TileShape_MNK{},
      ClusterShape{}));

  // static
  /*
  using shapeQ_Packed_2d = decltype(make_shape(make_shape(int32_t{}, int32_t{}), int32_t{}, int32_t{}));
  using shapeQ_Packed_2d_copy = decltype(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)),
      make_layout(
          make_shape(make_shape(_4{}, int32_t{}), int32_t{}, int32_t{}),
          make_stride(make_stride(get<3>(StrideQPacked{}), get<0>(StrideQPacked{})), get<1>(StrideQPacked{}), get<2>(StrideQPacked{})))));
  */
  using TMA_Q_Packed = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQPackedTMA{}, StrideQPackedTMA{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      ClusterShape{}));
  // using TMA_Q_Packed_2D = decltype(make_tma_copy(GmemTiledCopyQ{}, shapeQ_Packed_2d_copy{}, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), ClusterShape{}));

  using TMA_K = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQK{}),
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along M mode for this N load, if any

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, select<1, 0, 2>(StrideV{})), // headdim, seqlen, numhead;
      take<0, 2>(SmemLayoutVt{}), // headdim, kstages
      select<1, 2>(TileShape_MNK_PV{}), // k, N
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutVt{})) * cutlass::sizeof_bits_v<Element> / 8);

  using PipelineTmaAsync =
      std::conditional_t<CUTE_STATIC_V(size(ClusterShape{})) == 1, typename cutlass::PipelineTmaAsyncNoCluster<kStages>, typename cutlass::PipelineTmaAsync<kStages>>;
  using MainloopPipelineK = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineV = std::conditional_t<Use_TMA_KV, PipelineTmaAsync, typename cutlass::PipelineAsync<kStages>>;
  using PipelineState = cutlass::PipelineState<kStages>;

  // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q to be aligned
  // and have sQ being position_independent_swizzle_tensor.
  // If !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want smem_k and smem_v to be aligned.
  static constexpr size_t SmemAlignmentQ = !MmaQK_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
  static constexpr size_t SmemAlignmentK = 128;
  static constexpr size_t SmemAlignmentVtNoTranspose = cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static_assert(SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 && SmemAlignmentVtNoTranspose >= 128, "Require at least 128B alignment");
  static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
  static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

  using SmemP_t = std::conditional_t<MmaPV_is_RS, cute::array<Element, 0>, cute::array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
  // Sometimes even with SmemP_t = cute::array<Element, 0>, putting it in the TensorStorage struct causes
  // smem size to go from 227KB to 228KB and we get "invalid argument".

  struct TensorStorageWithoutP : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose), _0> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVtNoTranspose> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK> smem_k;
  };

  struct TensorStorageWithP : cute::aligned_struct<cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentVtNoTranspose, SmemAlignmentP), _0> {
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
  };

  // Device side kernel params
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    // ShapeQPacked const shape_Q_packed;
    ShapeQPackedTMA const shape_Q_packed;
    StrideQK const stride_Q;
    // StrideQPacked const stride_Q_packed;
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

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{}); // no mcast for Q

    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K =
        make_tma_copy_B_sm90(GmemTiledCopyKV{}, mK, take<0, 2>(SmemLayoutK{}), TileShape_MNK{}, ClusterShape{}); // mcast along M mode for this N load, if any
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), make_shape(args.headdim, get<0>(args.shape_K), get<2>(args.shape_K)), select<1, 0, 2>(args.stride_V));
    TMA_V tma_load_V = make_tma_copy(
        GmemTiledCopyKV{}, mV, take<0, 2>(SmemLayoutVt{}), select<1, 2>(TileShape_MNK_PV{}), size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    // int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
    auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
        args.shape_Q,
        make_shape(
            make_shape(cute::Int<qhead_per_khead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.shape_Q), // headdim
            get<2>(args.shape_K) // numhead_k
            ));

    auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
        args.stride_Q,
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.stride_Q), // headdim
            get<2>(args.stride_Q) * qhead_per_khead));

    auto mQPacked = [&]() {
      if constexpr (!PackGQA) {
        return mQ;
      } else {
        return make_tensor(
            make_gmem_ptr(args.ptr_Q),
            make_layout(
                make_shape(
                    make_shape(cute::Int<qhead_per_khead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
                    get<1>(args.shape_Q), // headdim
                    get<2>(args.shape_K) // numhead_k
                    ),
                stride_Q_packed));
      }
    }();

    TMA_Q_Packed tma_load_Q_packed = make_tma_copy(GmemTiledCopyQ{}, mQPacked, SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), ClusterShape{});

    // print(tma_load_Q_packed);
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
        cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))), // qhead_per_khead_divmod
        tma_load_Q,
        tma_load_Q_packed,
        tma_load_K,
        tma_load_V,
        !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.q_ranges,
        args.k_ranges,
        args.attn_type_map,
        args.cu_batches};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
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
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
      BlockMetaT& block_meta,
      int& work_idx) {
    // If this is true, we're guaranteed that only the first warp will execute this function
    static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;

    // prepare for cluster launch
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }

    while (!block_meta.is_finish() && !block_meta.is_valid()) {
      // Find the first valid block_meta
      block_meta.prefetch();
    }

    if (block_meta.is_finish()) {
      // No valid block found
      return false;
    }

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    auto is_tma_issue_thread = [&]() { return (SingleProducerWarp || warp_idx_in_warpgroup == 0) && cute::elect_one_sync(); };

    // as_position_independent_swizzle_tensor makes address calculation easier when we do LDSM & STSM to transpose.
    // But it requires smem_vt and smem_v to be aligned to e.g 512 bytes.
    // get thread_idx in the producer thread group
    // int const thread_idx = threadIdx.x % NumProducerThreads;
    // Only one thread in one warp within a warp group needs to issue the TMA load instruction

    // Define utility lambdas to load Q and KV
    auto load_Q = [&]() {
      auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
      auto block_tma_Q_Packed = params.tma_load_Q_packed.get_slice(_0{});
      Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, block_meta.bidh);

      Tensor mQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return params.tma_load_Q_packed.get_tma_tensor(params.shape_Q_packed)(_, _, block_meta.bidh);
        } else {
          return mQ;
        }
      }();

      Tensor gQ = local_tile(
          domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(block_meta.m_block, _0{})); // (M, K)

      // todo: replace 4 as qhead_per_khead here.
      Tensor gQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return local_tile(
              domain_offset(make_coord(block_meta.seqlen_info.offset_q * qhead_per_khead, _0{}), mQ_Packed), // we need multiple qhead_per_khead for offset of seqlen;
              select<0, 2>(TileShape_MNK{}),
              make_coord(block_meta.m_block, _0{})); // (M // qhead_per_khead, K, qhead_per_khead)
        } else {
          return gQ;
        }
      }();
      if (block_meta.bidh == 0 && threadIdx.x == 0 && block_meta.m_block == 0 && block_meta.bidb == 1) {
        // print_tensor(gQ_Packed_2d);
      }

      Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)

      Tensor tQgQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return group_modes<0, 3>(block_tma_Q_Packed.partition_S(gQ_Packed));
        } else {
          return tQgQ;
        }
      }();
      /*
      if constexpr (PackGQA) {
        if (block_meta.bidh == 0 && threadIdx.x == 0 && block_meta.m_block == 0 && block_meta.bidb == 0) {
          auto raw_ptr_Q = make_gmem_ptr(params.ptr_Q);

          Tensor mQ_Packed_2D = [&]() {
            if constexpr (PackGQA) {
              return params.tma_load_Q_Packed_2d.get_tma_tensor(params.shape_Q_Packed_2d)(_, _, block_meta.bidh);
            }
          }();
        /*
        Tensor gQ_Data_Tile = [&]() {
          if constexpr (PackGQA) {
            return local_tile(
              domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}, _0{}), mQ_Data),
              make_shape(params.qhead_per_khead_divmod.divide(get<0>(TileShape_MNK{})), get<2>(TileShape_MNK{}), get<3>(params.shape_Q_Packed)),
              make_coord(block_meta.m_block, _0{}, _0{}));
            }
        }(); */
      /*
      Tensor gQ_Data_2d_Tile = [&]() {
        return local_tile(
          domain_offset(make_coord(block_meta.seqlen_info.offset_q, _0{}), mQ_Packed_2D),
          select<0, 2>(TileShape_MNK{}),
          make_coord(block_meta.m_block, _0{}));
        );
      } */

      /*printf("\n[DEBUG] Actual Values in Global Memory for current tile:\n");
      print_tensor(gQ_Data_Tile(_, 0, _));
      printf("\n");
}
} */

      // Tensor tQgQ_Packed = !PackGQA ? tQgQ : group_modes<0, 4>(block_tma_Q.partition_S(gQ_Packed)); // (TMA)
      Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
      // Tensor sQ_Packed = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQPacked{});
      Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)
      if constexpr (Use_TMA_Q) {
        // Wait for the MMA warpgroups to signal that smem_q is ready
        if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
          cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + cutlass::NumThreadsPerWarp, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        }

        if (is_tma_issue_thread()) {
          shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);

          if constexpr (PackGQA) {
            auto tma_desc = params.tma_load_Q_packed.with(
                reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q),
                0 /*mcast_mask*/,
                TMA::CacheHintSm90::EVICT_FIRST);
            // if (block_meta.bidh == 0 && block_meta.m_block == 0 && threadIdx.x == 0) {
            //   print("PackGQA TMA desc     : "); print(tma_desc); print("\n");
            // }
            copy(tma_desc, tQgQ_Packed, tQsQ);
          } else {
            auto tma_desc = params.tma_load_Q.with(
                reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q),
                0 /*mcast_mask*/,
                TMA::CacheHintSm90::EVICT_FIRST);
            copy(tma_desc, tQgQ, tQsQ);
          }
        }
      }
    };

    auto load_K = [&](int const n_block_idx, auto& smem_pipe_write, int offset_k) {
      // Prepare the TMA load K
      auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
      Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, block_meta.bidh_kv);
      Tensor gK_TMA = local_tile(domain_offset(make_coord(offset_k, _0{}), mK_TMA), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _, _)
      // tma_partition doesn't handle position_independent_swizzle_tensor correctly, so we need to do it manually
      Tensor tKgK_TMA = group_modes<0, 3>(block_tma_K.partition_S(gK_TMA)); // (TMA, k)

      Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
      Tensor tKsK_TMA = group_modes<0, 3>(block_tma_K.partition_D(sK)); // (TMA, PIPE)

      if (is_tma_issue_thread()) {
        pipeline_k.producer_acquire(smem_pipe_write);
        copy(
            params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tKgK_TMA(_, n_block_idx),
            tKsK_TMA(_, smem_pipe_write.index()));
        ++smem_pipe_write;
      }
    };

    auto load_V = [&](int const n_block_idx, auto& smem_pipe_write, int offset_k) {
      // Prepare the TMA load V
      // Shape of V is (headdim, total_tokens, head_kv, batch)
      auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
      auto shape_V = make_shape(params.headdim, get<0>(params.shape_K), get<2>(params.shape_K));

      Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(shape_V)(_, _, block_meta.bidh_kv);
      Tensor gVt_TMA = local_tile(domain_offset(make_coord(_0{}, offset_k), mVt_TMA), select<1, 2>(TileShape_MNK_PV{}), make_coord(_0{}, _)); // (K, N, _, _)
      Tensor tVgVt_TMA = group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA)); // (TMA, k)

      Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});
      Tensor tVsVt_TMA = group_modes<0, 3>(block_tma_V.partition_D(sVt)); // (TMA, PIPE)

      if (is_tma_issue_thread()) {
        pipeline_v.producer_acquire(smem_pipe_write);
        copy(
            params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write), mcast_mask_kv, TMA::CacheHintSm90::EVICT_LAST),
            tVgVt_TMA(_, n_block_idx),
            tVsVt_TMA(_, smem_pipe_write.index()));
        ++smem_pipe_write;
      }
    };

    // Get n_block for kv
    int n_block = block_meta.n_block_max - 1;
    int prev_n_block = n_block;
    // Get offset for kv
    int offset_k = block_meta.seqlen_info.offset_k;
    int prev_offset_k = offset_k;
    // Get the minimum number of blocks to load
    int n_block_min = block_meta.n_block_min;

    // Prologue
    if constexpr (IntraWGOverlap) {
      load_K(n_block, smem_pipe_write_k, offset_k);
      load_Q();
      shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    } else {
      load_K(n_block, smem_pipe_write_k, offset_k);
      load_Q();
      shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
      load_V(n_block, smem_pipe_write_v, offset_k);
    }
    --n_block;

// Mainloop, load K and V interleaved
#pragma unroll 2
    do {
      // Prefetch the next block_meta
      block_meta.prefetch();

      // Loop until we reach the end of the current block
#pragma unroll(Use_TMA_KV ? 2 : 1)
      while (n_block >= n_block_min) {
        if constexpr (IntraWGOverlap) {
          load_K(n_block, smem_pipe_write_k, offset_k);
          load_V(prev_n_block, smem_pipe_write_v, prev_offset_k);
        } else {
          load_K(n_block, smem_pipe_write_k, offset_k);
          load_V(n_block, smem_pipe_write_v, offset_k);
        }

        // Step the previous n_block and offset_k
        prev_n_block = n_block;
        prev_offset_k = offset_k;
        // Decrement n_block
        --n_block;
      }

      // Step into the next block
      n_block = block_meta.n_block_max - 1;
      offset_k = block_meta.seqlen_info.offset_k;
      n_block_min = block_meta.n_block_min;
    } while (!block_meta.is_finish() && block_meta.is_valid());

    // Epilogue, load the tail V
    load_V(prev_n_block, smem_pipe_write_v, prev_offset_k);

    return true;
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
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
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
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      // Get the current mma warp group index
      // -1 is because one warp group is the producer
      int const curr_WG = flash::canonical_warp_group_idx_nosync() - 1;

      // Sync on the current mma warp group's named barrier
      cutlass::arch::NamedBarrier::sync(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + curr_WG /*id*/);
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
      cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) + next_WG /*id*/);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Get the current warp group index, since one warp group is producer, the warp group index for mma starts from 1
    int warp_group_idx = flash::canonical_warp_group_idx_nosync();

    // Tell producers that smem_q is ready to be loaded
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);

    if constexpr (UseSchedulerBarrier) {
      // We have NamedBarrier for up to 3 WGs (why 3 WGs ?)
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

      // WG1 is the smallest warp group used for mma, so it needs the very first signal to start
      if (warp_group_idx == 1) {
        cutlass::arch::NamedBarrier::arrive(2 * cutlass::NumThreadsPerWarpGroup, static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
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
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
      BlockMetaT& block_meta,
      SharedStorage& shared_storage) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

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

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;

    if constexpr (!MmaQK_is_RS) {
      static_assert(
          stride<0>(typename TiledMmaQK::ALayout{}) == 0 and stride<0>(typename TiledMmaQK::BLayout{}) == 0 and
              size<0>(typename TiledMmaQK::ALayout{}) == cutlass::NumThreadsPerWarpGroup and size<0>(typename TiledMmaQK::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
          "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    }

    static constexpr int MmaWarpGroups = size(TiledMmaPV{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}), make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    // Get the mma warp group index of the current thread, start from 0
    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    auto wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma_qk);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = wg_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = wg_mma_qk.partition_fragment_B(sK);
    Tensor tOrV = wg_mma_pv.partition_fragment_B(sV);
    Tensor tOsP = wg_mma_pv.partition_fragment_A(sP);
    // if p is in registers, do we still need this step ?
    Tensor tPsP = smem_thr_copy_P.partition_D(cute::as_position_independent_swizzle_tensor(sP));

    // Allocate S(Q@K) fragment
    Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

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

    while (!block_meta.is_finish() && !block_meta.is_valid()) {
      // Find the first valid block_meta
      block_meta.prefetch();
    }

    if (block_meta.is_finish()) {
      // No valid block found
      return false;
    }

    /** DEBUG **/
    // if (block_meta.bidb == 0 && block_meta.bidh == 0 && thread_idx == 0 && block_meta.m_block == 0) {
    //   printf(
    //       "initial block_meta: m_block: %d, n_block_min: %d, n_block_max: %d, seqlen_q: %d, seqlen_k: %d, attn_type: %d\n",
    //       block_meta.m_block,
    //       block_meta.n_block_min,
    //       block_meta.n_block_max,
    //       block_meta.seqlen_info.seqlen_q,
    //       block_meta.seqlen_info.seqlen_k,
    //       block_meta.attn_type
    //   );
    // }

    // Get n_block for kv
    int n_block = block_meta.n_block_max - 1;
    // Get seqlen for kv
    int seqlen_k = block_meta.seqlen_info.seqlen_k;
    // Get the minimum number of blocks to calculate
    int n_block_min = block_meta.n_block_min;
    // Get attention type for n_block
    flash::AttnType attn_type = block_meta.attn_type;
    // Get mask for n_block
    flash::Mask<kBlockM, kBlockN, TiledMmaQK> mask;
    // Whether the boundary masking is finished
    bool finish_boundary = true;

    // According to the specific attn_type, define three types of mask_fn, used in different situations
    // boundary_mask_fn: mask for boundary block, for the rightmost block in a tile job
    auto bypass_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {};
    auto boundary_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {
      mask.template apply<true /*Seqlenk_mask*/>(tSrS, block_meta.m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
    };
    // no_mask_fn: no mask, for full attention block in a tile job
    auto no_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {
      if constexpr (RangeMerge) {
        if (!finish_boundary) {
          boundary_mask_fn(tSrS, n_block, attn_type, seqlen_q, seqlen_k);
          finish_boundary = true;
        }
      } else {
        bypass_fn(tSrS, n_block, attn_type, seqlen_q, seqlen_k);
      }
    };
    // regular_mask_fn: mask for specific attention type block, for all other blocks in a tile job
    auto regular_mask_fn = [&](auto& tSrS, int n_block, auto const& attn_type, int const& seqlen_q, int const& seqlen_k) {
      if constexpr (RangeMerge) {
        if (!finish_boundary) {
          boundary_mask_fn(tSrS, n_block, attn_type, seqlen_q, seqlen_k);
          finish_boundary = true;
        } else {
          mask.template apply<false /*Seqlenk_mask*/>(tSrS, block_meta.m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
        }
      } else {
        mask.template apply<false /*Seqlenk_mask*/>(tSrS, block_meta.m_block, n_block, attn_type, thread_idx, seqlen_q, seqlen_k);
      }
    };

    /* ================================================= Prologue ================================================= */
    // Wait for the Q to be loaded
    barrier_Q.wait(work_idx % 2);
    // Wait for first block of k to be loaded
    consumer_wait(pipeline_k, smem_pipe_read_k);

    if (threadIdx.x == 128 && block_meta.bidh == 1 && block_meta.m_block == 0 && block_meta.bidb == 1) {
      // printf("m_block: %d\n", block_meta.m_block);
      // if (block_meta.seqlen_info.offset_q == 30) {
      // printf("offset_q: %d\n", block_meta.seqlen_info.offset_q);
      // print_tensor(sQ);
      // }
      // print_tensor(sQ_flattened);
    }

    // launch Q @ K of n_block and wait for it to finish
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
    warpgroup_wait<0>();

    // The first block of k has been consumed, notify producer that this buffer can be reused
    consumer_release(pipeline_k, smem_pipe_read_k);

    // Apply score-modification-function(currently only support softcap) before mask
    scoremod_premask_fn(tSrS);

    /** DEBUG **/
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ tSrS m_block: %d ==============================\n", m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS m_block: %d ==============================\n", m_block);
    // }

    // Apply mask
    boundary_mask_fn(tSrS, n_block, attn_type, block_meta.seqlen_info.seqlen_q, seqlen_k);

    /** DEBUG **/
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ tSrS after mask m_block: %d ==============================\n", m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS after mask m_block: %d ==============================\n", m_block);
    // }

    // Get row-max and row-sum of tSrS
    cute::copy(softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS), scores_scale);

    /** DEBUG **/
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ scores_scale m_block: %d ==============================\n", m_block);
    //     print_tensor(scores_scale);
    //     printf("============================================ scores_scale m_block: %d ==============================\n", m_block);
    // }

    // Apply exponential to tSrS
    softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);

    /** DEBUG **/
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ tSrS after online_softmax m_block: %d ==============================\n", m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS after online_softmax m_block: %d ==============================\n", m_block);
    // }

    // Convert layout and type from tSrS to tOrP which will be used in MmaPV
    Tensor tOrP = [&]() {
      Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMmaPV>(tSrS.layout()));
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      convert_type_out(tOrP_acc, tOrP);
      return tOrP;
    }();

    // Write tOrP to smem
    if constexpr (!MmaPV_is_RS) {
      write_P_to_smem(tOrP);
      // what's the purpose of this fence?
      arrive_on_P_write_barrier();
    }

    --n_block;

/* ================================================= Mainloop ================================================= */
#pragma unroll 2
    do {
      // Each step does Q @ K for iter n_block, P @ V for iter n_block + 1, and softmax for iter n_block.
      auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
        // Forward step: perform gemm0 (Q@K), gemm1 (P@V) and softmax in an interleaved fashion

        // Extract the boolean value from the check_inf_type template parameter to determine if we need to check for infinity values
        static constexpr bool Check_inf = decltype(check_inf_type)::value;

        // Partition the fragment C tensor into a new tensor tSrS, which is used to store the result of the Q@K matrix multiplication for n_block
        Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

        // If UseSchedulerBarrier is not enabled, all threads need to call consumer_wait, otherwise only threads in the 0th mma warp group call consumer_wait
        if (!UseSchedulerBarrier || warp_group_idx == 0) {
          consumer_wait(pipeline_k, smem_pipe_read_k);
        }

        // Sync on the current mma warp group's named barrier, and wait for the previous mma warp group to finish
        warp_scheduler_barrier_sync();

        // Do Q @ K of n_block
        flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);

        if constexpr (RescaleOBeforeGemm) {
          softmax.rescale_o(tOrO, scores_scale);
        }

        if (!UseSchedulerBarrier || warp_group_idx == 0) {
          // Wait for v to be loaded into shared memory
          consumer_wait(pipeline_v, smem_pipe_read_v);
        }

        // Do p @ v of n_block + 1
        flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
            tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);

        // Arrive on the next mma warp group's named barrier
        warp_scheduler_barrier_arrive();

        // Only wait for the Q @ K of n_block to finish
        warpgroup_wait<1>();

        // Signal that the current stage's K smem has been used up, can continue loading subsequent K
        consumer_release(pipeline_k, smem_pipe_read_k);

        // Apply score-modification-function(currently only support softcap) before mask
        scoremod_premask_fn(tSrS);

        // Apply mask
        mask_fn(tSrS, n_block, attn_type, block_meta.seqlen_info.seqlen_q, seqlen_k);

        // if (bidb == 1 && bidh == 0 && thread_idx == 255 && m_block == 1) {
        //     printf("============================================ tSrS fwd before online_softmax m_block: %d ==============================\n", m_block);
        //     print_tensor(tSrS);
        //     printf("============================================ tSrS fwd before online_softmax m_block: %d ==============================\n", m_block);
        // }

        // Get row-max and row-sum of tSrS
        cute::copy(softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS), scores_scale);

        // Apply exponential to tSrS (need to subtract row max)
        softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
        // if (bidb == 1 && bidh == 0 && thread_idx == 255 && m_block == 1) {
        //     printf("============================================ tSrS fwd after online_softmax m_block: %d ==============================\n", m_block);
        //     print_tensor(tSrS);
        //     printf("============================================ tSrS fwd after online_softmax m_block: %d ==============================\n", m_block);
        // }

        // Wait for P @ V of n_block + 1 to finish
        warpgroup_wait<0>();

        // Signal that the current stage's V smem has been used up, can continue loading subsequent V
        consumer_release(pipeline_v, smem_pipe_read_v);

        // Convert layout and type from tSrS to tOrP
        convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);

        // Write tOrP to smem
        if constexpr (!MmaPV_is_RS) {
          write_P_to_smem(tOrP);
        }

        // Only rescale tOrO if RescaleOBeforeGemm is not enabled
        if constexpr (!RescaleOBeforeGemm) {
          softmax.rescale_o(tOrO, scores_scale);
        }

        // what's the purpose of this fence?
        if constexpr (!MmaPV_is_RS) {
          arrive_on_P_write_barrier();
        }
      };

      // Prefetch the next block_meta
      block_meta.prefetch();

      if (n_block >= n_block_min && seqlen_k % kBlockN == 0 && attn_type == flash::AttnType::Full) {
        // If seqlen_k is a multiple of kBlockN, we can skip the boundary mask for the first n_block
        fwd_step(n_block, bypass_fn, cute::true_type{} /*check_inf*/);
        --n_block;
        finish_boundary = true;
      }

      if (n_block >= n_block_min) {
        if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
          int const m_idx_min = block_meta.m_block * kBlockM;
          int const n_block_min_causal_local_mask = std::max(n_block_min, (m_idx_min + seqlen_k - block_meta.seqlen_info.seqlen_q) / kBlockN);
#pragma unroll 1
          for (; n_block >= n_block_min_causal_local_mask; --n_block) {
            fwd_step(n_block, regular_mask_fn, cute::true_type{} /*check_inf*/);
          }
        }

        // Calculate the number of iterations needed before the left boundary of inv-causal and bi-causal, where we can skip applying mask to speed up
        int const m_idx_max = (block_meta.m_block + 1) * kBlockM;
        int const n_block_min_before_inv_causal_mask =
            attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal ? n_block_min : cute::ceil_div(m_idx_max, kBlockN);
        // Skip applying mask to the iterations before the left boundary of inv-causal and bi-causal, where we can skip applying mask to speed up
#pragma unroll 1
        for (; n_block >= n_block_min_before_inv_causal_mask; --n_block) {
          fwd_step(n_block, no_mask_fn, cute::false_type{} /*check_inf*/);
        }

        // Separate masking iterations on the left for inv-causal and bi-causal attention, because they are both top-left aligned
        if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
#pragma unroll 1
          for (; n_block >= n_block_min; --n_block) {
            fwd_step(n_block, regular_mask_fn, cute::true_type{} /*check_inf*/);
          }
        }
      }

      // Step into the next block
      n_block = block_meta.n_block_max - 1;
      seqlen_k = block_meta.seqlen_info.seqlen_k;
      n_block_min = block_meta.n_block_min;
      attn_type = block_meta.attn_type;
      finish_boundary = []() {
        if constexpr (RangeMerge) {
          return false;
        } else {
          return true;
        }
      }();
    } while (!block_meta.is_finish() && block_meta.is_valid());

    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreadsQK + (Use_TMA_Q ? cutlass::NumThreadsPerWarp : NumProducerThreads), static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);

    // Only rescale tOrO if RescaleOBeforeGemm is enabled
    if constexpr (RescaleOBeforeGemm) {
      softmax.rescale_o(tOrO, scores_scale);
    }

    // Signal that the current stage's V smem has been used up, can continue loading subsequent V
    consumer_wait(pipeline_v, smem_pipe_read_v);

    // Do P @ V for the most left n_block
    flash::gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma_pv, cute::conditional_return<MmaPV_is_RS>(tOrP, tOsP), tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);

    // Get the final scores_scale
    cute::copy(softmax.finalize(), scores_scale);

    // if (bidb == 1 && bidh == 0 && thread_idx == 255 && m_block == 1) {
    //     printf("============================================ tOrO m_block: %d ==============================\n", m_block);
    //     print_tensor(tOrO);
    //     printf("============================================ scores_scale m_block: %d ==============================\n", m_block);
    //     print_tensor(scores_scale);
    //     printf("============================================ tOrO m_block: %d ==============================\n", m_block);
    // }

    // Wait for P @ V of the most left n_block to finish
    warpgroup_wait<0>();

    // Signal that the current stage's V smem has been used up, can continue loading subsequent V
    consumer_release(pipeline_v, smem_pipe_read_v);
    ++work_idx;

    // Rescale tOrO
    softmax.rescale_o(tOrO, scores_scale);
    return true;
  }
};

} // namespace flash
