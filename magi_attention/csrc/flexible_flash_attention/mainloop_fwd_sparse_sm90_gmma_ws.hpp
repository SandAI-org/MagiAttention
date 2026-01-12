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
    bool PackGQA_ = false,
    int Qhead_per_khead_ = 1>
struct CollectiveMainloopSparseFwdSm90 {
  static constexpr int kStages = Stages;
  using ClusterShape = ClusterShape_;

  // kBlockM, kBlockN, kHeadDim
  using TileShape_MNK = TileShape_MNK_;

  // TileShapeMNK for mma qv: kBlockM, kBlockN, kHeadDim
  // (kBlockM, kHeadDim) @ (kHeadDim, kBlockN) -> (kBlockM, kBlockN)
  using TileShape_MNK_QV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{})), decltype(get<2>(TileShape_MNK{}))>;

  // TileShapeMNK for mma pv: kBlockM, kHeadDim, kBlockN
  // (kBlockM, kBlockN) @ (kBlockN, kHeadDim) -> (kBlockM, kHeadDim)
  using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), decltype(get<2>(TileShape_MNK{})), decltype(get<1>(TileShape_MNK{}))>;
  // interface consistency with non-sparse mainloop
  using TileShape_MNK_PV_Active = TileShape_MNK_PV;

  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  static constexpr bool Has_softcap = Has_softcap_;
  static constexpr bool MmaPV_is_RS = MmaPV_is_RS_;
  static constexpr bool IntraWGOverlap = IntraWGOverlap_;
  static constexpr bool RangeMerge = RangeMerge_;
  static constexpr bool PackGQA = PackGQA_;
  static constexpr int Qhead_per_khead = Qhead_per_khead_;
  static constexpr bool SwapAB = false;
  static constexpr bool SparseLoad = true;

  // By default, we use TMA for Q and KV to get better performance
  static constexpr bool Use_TMA_Q = true;
  // When using sparse load, use cp.async to load K/V instead of TMA
  static constexpr bool Use_TMA_KV = false;

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
  using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, PackGQA, Qhead_per_khead>;

  // Register bandwidth is actually a bottleneck so we don't want Q to be in registers.
  // Leaving this option here for reference.
  static constexpr bool MmaQK_is_RS = false;

  // Const parameters for sparse load
  // A group of 8 threads load global memory together to form one memory transaction (8 * 16B = 128B)
  static constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128 / GROUP_SIZE;
  // Number of rows (tokens) to load per group
  static constexpr int NUM_ROWS_PER_GROUP = kBlockN / NUM_GROUPS;
  static_assert(NUM_ROWS_PER_GROUP == 8, "When sparse load, only support 8 tokens load for each group");

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
  // interface consistency with non-sparse mainloop
  using TiledMmaPV_Active = TiledMmaPV;

  // REVIEW: do we still need TiledMmaPV_RS any more ?
  using TiledMmaPV_RS =
      decltype(cute::make_tiled_mma(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_PV, GMMA::Major::K, MmaMajorV>(), AtomLayoutPV{}));

  // do pv must be larger than qk or not ?
  static constexpr int NumMmaThreadsQK = size(TiledMmaQK{});
  static constexpr int NumMmaThreads = size(TiledMmaPV{});
  // use one warpgroup to produce KV with cp.async, use one thread to produce Q with TMA
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
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
  using ShapeQPackedTMA = std::conditional_t<
      !PackGQA,
      ShapeQKV,
      cute::Shape<cute::Shape<cute::Int<Qhead_per_khead>, int32_t>, int32_t, int32_t> // ((qhead_per_khead, seqlen), headdim, khead)
      >;
  using StrideQK = cute::Stride<int64_t, _1, int64_t>;
  using StrideV = StrideQK;

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
      ClusterShape{}));

  using TMA_Q_Packed = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQPackedTMA{}, StrideQPackedTMA{}),
      SmemLayoutQ{},
      select<0, 2>(TileShape_MNK{}),
      ClusterShape{}));

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);

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
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
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
    float const softmax_scale_log2;
    float const softcap_val;
    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;
    int const* const cu_batches;
    int const* const sparse_load_loop_count;
    uint8_t const* const sparse_load_invalid_count;
    int const* const equal_k_range_size;
  };

  template <bool IsProducer>
  struct SparseBlockMeta {
    int const& m_block;
    int const& bidh;
    int const bidh_kv;
    int bidb;
    int end_batches;
    SeqlenInfo_t seqlen_info;
    flash::AttnType attn_type;

    // number of invalid tokens of the current group, control the token pointer of each group
    int num_invalid_token;
    int cur_k_range_indices[NUM_ROWS_PER_GROUP];
    int cur_k_range_inner_indices[NUM_ROWS_PER_GROUP];
    int token_indices[NUM_ROWS_PER_GROUP];
    int prev_token_indices[NUM_ROWS_PER_GROUP];
    int cur_loop;
    int loop_count;
    int stride_kv_s_kv;
    bool is_equal_k_range_size;
    int k_range_size; // size of all k ranges, only used when is_equal_k_range_size == true

    int2 const* const q_ranges;
    int2 const* const k_ranges;
    int const* const attn_type_map;

    template <typename SharedStorage>
    CUTLASS_DEVICE SparseBlockMeta(Params const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage, int thread_idx)
        : m_block(get<0>(block_coord)),
          bidh(get<1>(block_coord)),
          bidh_kv(!PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh), // for packgqa, bidh_kv is the actual head index
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

      int last_idx = NUM_ROWS_PER_GROUP - 1;
      // initialize to the last valid token index
      cur_k_range_indices[last_idx] = end_batches - 1;
      cur_k_range_inner_indices[last_idx] = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x - 1;
      prev_token_indices[last_idx] = -1;

      if (is_equal_k_range_size) {
        k_range_size = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x;
      }

      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GROUP_SIZE;

      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        // metadata init
        // 1. search for the first token in the group
        int cnt = 0;
        // move to the last token index of each group
        int num_steps = (NUM_GROUPS - group_idx - 1) * NUM_ROWS_PER_GROUP;
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
        // 2. search for next NUM_ROWS_PER_GROUP tokens and compute token indices
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
        int offset = num_invalid_token % NUM_ROWS_PER_GROUP;
        switch (offset) {
          case 0:
            break;
          case 1:
            token_indices[0] = token_indices[last_idx];
            break;
          case 2:
            token_indices[1] = token_indices[last_idx];
            token_indices[0] = token_indices[last_idx - 1];
            break;
          case 3:
            token_indices[2] = token_indices[last_idx];
            token_indices[1] = token_indices[last_idx - 1];
            token_indices[0] = token_indices[last_idx - 2];
            break;
          case 4:
            token_indices[3] = token_indices[last_idx];
            token_indices[2] = token_indices[last_idx - 1];
            token_indices[1] = token_indices[last_idx - 2];
            token_indices[0] = token_indices[last_idx - 3];
            break;
          case 5:
            token_indices[4] = token_indices[last_idx];
            token_indices[3] = token_indices[last_idx - 1];
            token_indices[2] = token_indices[last_idx - 2];
            token_indices[1] = token_indices[last_idx - 3];
            token_indices[0] = token_indices[last_idx - 4];
            break;
          case 6:
            token_indices[5] = token_indices[last_idx];
            token_indices[4] = token_indices[last_idx - 1];
            token_indices[3] = token_indices[last_idx - 2];
            token_indices[2] = token_indices[last_idx - 3];
            token_indices[1] = token_indices[last_idx - 4];
            token_indices[0] = token_indices[last_idx - 5];
            break;
          case 7:
            token_indices[6] = token_indices[last_idx];
            token_indices[5] = token_indices[last_idx - 1];
            token_indices[4] = token_indices[last_idx - 2];
            token_indices[3] = token_indices[last_idx - 3];
            token_indices[2] = token_indices[last_idx - 4];
            token_indices[1] = token_indices[last_idx - 5];
            token_indices[0] = token_indices[last_idx - 6];
            break;
          default:
            break;
        }
        // NOTE: significant register spill due to dynamic indexing!
        // if (offset > 0) {
        //   int cnt = 0;
        //   for (int i = offset - 1; i >= 0; i--) {
        //     token_indices[i] = token_indices[last_idx - cnt];
        //     cnt++;
        //   }
        // }

        // ======DEBUG=======
        // if (m_block == 0) {
        //   if (group_idx == 3) {
        //     printf("Token indices for %d thread in group %d: \n", threadIdx.x, group_idx);
        //     for (int i = 0; i < NUM_ROWS_PER_GROUP; ++i) {
        //       printf("%d, ", token_indices[i] / stride_kv_s_kv);
        //     }
        //     printf("\n");
        //   }
        // }
      }
    }

    CUTLASS_DEVICE
    void prefetch() {
      ++cur_loop;
      // update previous token indices
      for (int i = 0; i < NUM_ROWS_PER_GROUP; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      // update token index for each thread
      if (!is_finish()) {
        int num_threads = NumProducerThreads;
        int num_steps = num_threads; // move pointer to the next token, for each thread
        int cnt = 0;
        int last_idx = NUM_ROWS_PER_GROUP - 1;

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

        // 2. search for next NUM_ROWS_PER_GROUP tokens and compute token indices
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

        // ======DEBUG=======
        // if (m_block == 0) {
        //   if (group_idx == 3) {
        //     printf("Token indices for %d thread in group %d: \n", threadIdx.x, group_idx);
        //     for (int i = 0; i < NUM_ROWS_PER_GROUP; ++i) {
        //       printf("%d, ", token_indices[i] / stride_kv_s_kv);
        //     }
        //     printf("\n");
        //     printf("Previous Token indices for %d thread in group %d: \n", threadIdx.x, group_idx);
        //     for (int i = 0; i < NUM_ROWS_PER_GROUP; ++i) {
        //       printf("%d, ", prev_token_indices[i] / stride_kv_s_kv);
        //     }
        //   }
        // }
      }
    }

    CUTLASS_DEVICE
    bool is_finish() {
      return cur_loop >= loop_count;
    }
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

    int cur_loop;
    int loop_count;
    int num_invalid_token;

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
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy_A_sm90(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, TileShape_MNK{}, ClusterShape{}); // no mcast for Q

    auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
        args.shape_Q,
        make_shape(
            make_shape(cute::Int<Qhead_per_khead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.shape_Q), // headdim
            get<2>(args.shape_K) // numhead_k
            ));

    auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
        args.stride_Q,
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), // (qhead_per_khead, seqlen)
            get<1>(args.stride_Q), // headdim
            get<2>(args.stride_Q) * Qhead_per_khead));

    auto mQPacked = [&]() {
      if constexpr (!PackGQA) {
        return mQ;
      } else {
        return make_tensor(
            make_gmem_ptr(args.ptr_Q),
            make_layout(
                make_shape(
                    make_shape(cute::Int<Qhead_per_khead>{}, get<0>(args.shape_Q)), // (qhead_per_khead, seqlen)
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
        cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))), // qhead_per_khead_divmod
        tma_load_Q,
        tma_load_Q_packed,
        !Has_softcap ? float(args.softmax_scale * M_LOG2E) : float(args.softcap_val * M_LOG2E),
        !Has_softcap ? 0.f : args.softmax_scale / args.softcap_val,
        args.q_ranges,
        args.k_ranges,
        args.attn_type_map,
        args.cu_batches,
        args.sparse_load_loop_count,
        args.sparse_load_invalid_count,
        args.equal_k_range_size};
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
      int& work_idx,
      int const thread_idx) {
    if (block_meta.is_finish()) {
      // No more blocks to process
      return false;
    }

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    auto is_tma_issue_thread = [&]() { return (warp_idx_in_warpgroup == 0) && cute::elect_one_sync(); };

    // Define utility lambdas to load Q (TMA load)
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

      Tensor gQ_Packed = [&]() {
        if constexpr (PackGQA) {
          return local_tile(
              domain_offset(
                  make_coord(block_meta.seqlen_info.offset_q * Qhead_per_khead, _0{}),
                  mQ_Packed), // for packgqa, we need multiple qhead_per_khead for offset of seqlen;
              select<0, 2>(TileShape_MNK{}),
              make_coord(block_meta.m_block, _0{})); // (M // qhead_per_khead, K, qhead_per_khead)
        } else {
          return gQ;
        }
      }();

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
        // if (warp_idx_in_warpgroup == 0) {
        cutlass::arch::NamedBarrier::sync(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
        // }
        if (is_tma_issue_thread()) {
          shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);

          if constexpr (PackGQA) {
            auto tma_desc = params.tma_load_Q_packed.with(
                reinterpret_cast<typename cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.pipelines.barrier_Q),
                0 /*mcast_mask*/,
                TMA::CacheHintSm90::EVICT_FIRST);

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

    int64_t cache_policy = createpolicy_evict_last();

    int num_tiles = kHeadDim * sizeof(Element) / 128; // each tile load 128B
    int idx_in_warpgroup = thread_idx % 128;
    int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
    int group_idx = idx_in_warpgroup / GROUP_SIZE;

    // ======Coalesced Load======
    auto load_K = [&](auto& smem_pipe_write) {
      pipeline_k.producer_acquire(smem_pipe_write);
      // Producer Ops. calculate src/dst offset based on token index, then cp.async load
      // K shape: (seqlen, head_dim, num_heads)
      // each thread in the same group has a offset 16B (8 elements)
      Element* ptr_gK_base = params.ptr_K + block_meta.bidh_kv * get<2>(params.stride_K) + idx_in_group * 8;
      // shared memory pointer
      Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});

      // loop over token indices
      CUTE_UNROLL
      for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
        int token_idx = block_meta.token_indices[local_row];
        // loop over number of tiles to load one token
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
          Element* dst_ptr = &sK(group_idx * NUM_ROWS_PER_GROUP + local_row, idx_in_group * 8 + tile_idx * 64, smem_pipe_write.index());
          cp_async_cacheglobal_l2_prefetch_256B(ptr_gK_base + token_idx + tile_idx * 64, dst_ptr, true, cache_policy);
        }
      }

      pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write;

      // ======DEBUG========
      // if (block_meta.m_block == 0 && threadIdx.x == 0 && block_meta.cur_loop == 0) {
      //   printf("shared memory sK: \n");
      //   cute::print_tensor(sK);

      //   printf("global memory gK: \n");
      //   for (int i = 0; i < NUM_ROWS_PER_GROUP; ++i) {
      //     int token_idx = block_meta.token_indices[i];
      //     printf("Token %d: \n", token_idx / get<0>(params.stride_K));
      //     for (int j = 0; j < kHeadDim; ++j) {
      //       printf("%.2f, ", static_cast<float>(ptr_gK_base[token_idx + j]));
      //     }
      //     printf("\n");
      //   }
      // }
    };

    auto load_V = [&](auto& smem_pipe_write) {
      pipeline_v.producer_acquire(smem_pipe_write);
      // Producer Ops. calculate src/dst offset based on token index, then cp.async load
      // V shape: (seqlen, head_dim, num_heads)
      // each thread in the same group has a offset 16B (8 elements)
      Element* ptr_gV_base = params.ptr_V + block_meta.bidh_kv * get<2>(params.stride_V) + idx_in_group * 8;
      // shared memory pointer
      Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

      // loop over token indices
      CUTE_UNROLL
      for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
        int token_idx = block_meta.prev_token_indices[local_row];
        // loop over number of tiles to load one token
        CUTE_UNROLL
        for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
          Element* dst_ptr = &sVt(idx_in_group * 8 + tile_idx * 64, group_idx * NUM_ROWS_PER_GROUP + local_row, smem_pipe_write.index());
          cp_async_cacheglobal_l2_prefetch_256B(ptr_gV_base + token_idx + tile_idx * 64, dst_ptr, true, cache_policy);
        }
      }

      pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write;
    };

    // Prologue
    int n_block_max = block_meta.loop_count;
    if constexpr (IntraWGOverlap) {
      load_K(smem_pipe_write_k);
      load_Q();
      shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    }

    do {
      // Prefetch the next block_meta
      block_meta.prefetch();
      int n_block = block_meta.cur_loop;

      if (n_block < n_block_max) {
        // Load interleaved K/V
        if constexpr (IntraWGOverlap) {
          load_K(smem_pipe_write_k);
          load_V(smem_pipe_write_v);
        }
      }

    } while (!block_meta.is_finish());

    // Epilogue, load the tail V
    load_V(smem_pipe_write_v);

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
    pipeline_k.producer_tail(smem_pipe_write_k);
    pipeline_v.producer_tail(smem_pipe_write_v);
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
    cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);

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

    if (block_meta.is_finish()) {
      // No more blocks to process
      return false;
    }

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
    int n_block_max = block_meta.loop_count;
    int n_block = 0;
    // // Get attention type for n_block
    flash::AttnType attn_type = block_meta.attn_type;
    // Get mask for n_block
    flash::Mask<kBlockM, kBlockN, TiledMmaQK> mask;

    /* ================================================= Prologue ================================================= */
    // Wait for the Q to be loaded
    barrier_Q.wait(work_idx % 2);
    // Wait for first block of k to be loaded
    consumer_wait(pipeline_k, smem_pipe_read_k);

    // launch Q @ K of n_block and wait for it to finish
    flash::gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
    warpgroup_wait<0>();
    // if (block_meta.bidb == 0 && block_meta.m_block == 0 && thread_idx == 0) {
    //     printf("============================================ tSrS m_block: %d ==============================\n", block_meta.m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS m_block: %d ==============================\n", block_meta.m_block);
    // }

    // The first block of k has been consumed, notify producer that this buffer can be reused
    consumer_release(pipeline_k, smem_pipe_read_k);

    // Apply score-modification-function(currently only support softcap) before mask
    scoremod_premask_fn(tSrS);
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ tSrS m_block: %d ==============================\n", m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS m_block: %d ==============================\n", m_block);
    // }

    // Apply boundary mask
    mask.template apply_sparse_load(tSrS, block_meta.num_invalid_token, thread_idx);
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ tSrS after mask m_block: %d ==============================\n", m_block);
    //     print_tensor(tSrS);
    //     printf("============================================ tSrS after mask m_block: %d ==============================\n", m_block);
    // }

    // if (block_meta.bidb == 0 && block_meta.m_block == 0 && thread_idx == 0) {
    //     printf("============================================ tSrS after mask m_block: %d, thread_idx: %d ==============================\n", block_meta.m_block,
    //     thread_idx); print_tensor(tSrS); printf("============================================ tSrS after mask m_block: %d, thread_idx: %d
    //     ==============================\n", block_meta.m_block, thread_idx);
    // }
    // Get row-max and row-sum of tSrS
    cute::copy(softmax.template max_get_scale</*Is_first=*/true, /*Check_inf=*/true>(tSrS), scores_scale);
    // if (bidb == 0 && bidh == 0 && thread_idx == 0 && m_block == 0) {
    //     printf("============================================ scores_scale m_block: %d ==============================\n", m_block);
    //     print_tensor(scores_scale);
    //     printf("============================================ scores_scale m_block: %d ==============================\n", m_block);
    // }

    // Apply exponential to tSrS
    softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(tSrS);
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

    ++n_block; // iterate from 0 to loop_count - 1

/* ================================================= Mainloop ================================================= */
#pragma unroll 2
    do {
      // Each step does Q @ K for iter n_block, P @ V for iter n_block + 1, and softmax for iter n_block.
      auto fwd_step = [&](int const n_block, auto check_inf_type) {
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

        // Do p @ v of n_block - 1
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

        // Apply mask (no operation here because sparse load only supports full attention types)

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
      n_block = block_meta.cur_loop;

      if (n_block < n_block_max && attn_type == flash::AttnType::Full) {
        fwd_step(n_block, cute::true_type{} /*check_inf*/);
        ++n_block;
      }

      // Step into the next block
      attn_type = block_meta.attn_type;
    } while (!block_meta.is_finish());

    cutlass::arch::NamedBarrier::arrive(NumMmaThreadsQK + NumProducerThreads, static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);

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
