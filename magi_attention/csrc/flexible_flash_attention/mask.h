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
#include <cutlass/fast_math.h> // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

// Enumeration for different attention types
enum class AttnType {
  Full = 0,
  Causal = 1,
  InvCausal = 2,
  BiCausal = 3,
};

// Mask struct for applying attention masks
template <int kBlockM, int kBlockN, typename TiledMma, bool SwapAB = false>
struct Mask {
  // Apply mask to the tensor tSrS based on attention type and sequence lengths
  template <bool Seqlenk_mask = false, bool PackGQA = false, int QheadPerKhead = 1, typename Engine, typename Layout>
  CUTLASS_DEVICE void apply(
      Tensor<Engine, Layout>& tSrS,
      const int m_block,
      const int n_block,
      const flash::AttnType attn_type,
      const int thread_idx,
      const int seqlen_q,
      const int seqlen_k) const {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

    static constexpr int Row = !SwapAB ? 0 : 1;
    static constexpr int Col = !SwapAB ? 1 : 0;

    // Create identity tensor for block shape
    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

    // Use the column indices of thread0 for comparison, known at compile time
    int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
    int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;

    // Handle right boundary
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) {
      if constexpr (Seqlenk_mask) { // Mask based on column
#pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
#pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
    } else if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      if constexpr (!SwapAB) {
        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
        static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
        // Might get out of bounds but will be checked later
        int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
#pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
          int const physical_row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
          // for packgqa, the actual row index need to divide by QheadPerKhead
          int const logical_row_idx = !PackGQA ? physical_row_idx : (physical_row_idx / QheadPerKhead);
          int const col_limit_right = !Seqlenk_mask ? logical_row_idx + causal_row_offset : __viaddmin_s32(logical_row_idx, causal_row_offset, seqlenk_col_limit);

#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else {
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const dist = seqlen_k - seqlen_q;

#pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
          // Calculate absolute global Key index
          int const global_k = col0 + n_block * kBlockN + thread_col_offset;

          // Calculate logical query limit: Q_logical >= K_logical - (Sk - Sq)
          // Convert to physical limit: limit * QheadPerKhead
          // Transform to local coordinate: - m_block_offset - thread_offset
          int const row_limit_global = (global_k - dist) * (!PackGQA ? 1 : QheadPerKhead);
          int const row_limit_bottom = row_limit_global - m_block * kBlockM - thread_row_offset;

#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            // Mask if Key is OOB or Q_phys < limit
            if (global_k >= seqlen_k || int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_bottom) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
    }

    // Handle left boundary
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
      // No left boundary mask needed for Full or Causal
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      if constexpr (!SwapAB) {
#pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
          int const physical_row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
          int const logical_row_idx = !PackGQA ? physical_row_idx : (physical_row_idx / QheadPerKhead);
          int const col_limit_left = logical_row_idx - n_block * kBlockN - thread_col_offset;

#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) < col_limit_left) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else {
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));

#pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
          // Calculate absolute global Key index
          int const global_k = col0 + n_block * kBlockN + thread_col_offset;

          // Determine the maximum valid global Query index (Row Limit)
          // InvCausal implies we keep the Upper Triangle where Q_logical <= K_logical.
          // With PackGQA, one Key corresponds to 'G' Query heads (G = QheadPerKhead).
          // Therefore, for a specific Key 'K', the valid physical Query range extends
          // to the last head in the group: Max_Q_phys = K * G + (G - 1).
          int const row_limit_global = !PackGQA ? global_k : (global_k * QheadPerKhead + (QheadPerKhead - 1));

          // Transform global limit to local coordinate relative to the thread block/warp
          int const row_limit_bottom = row_limit_global - m_block * kBlockM - thread_row_offset;

#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            // Mask if K is OOB or Q_phys > limit
            if (global_k >= seqlen_k || int(get<Row>(t0ScS_rowcol(m, _0{}))) > row_limit_bottom) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
    }
  };

  template <typename Engine, typename Layout>
  CUTLASS_DEVICE void apply_padding_mask(Tensor<Engine, Layout>& tSrS, int num_invalid_token, int thread_idx) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

    static constexpr int Col = !SwapAB ? 1 : 0;

    // Create identity tensor for block shape
    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

    // Use the column indices of thread0 for comparison, known at compile time
    int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
    int const seqlenk_col_limit = kBlockN - num_invalid_token - thread_col_offset;

#pragma unroll
    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
      if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
#pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
          tSrS_rowcol(m, n) = -INFINITY;
        }
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Unified mask dispatch: partitions iteration space into Causal/InvCausal diagonal
// and no-mask regions based on attention type, then dispatches appropriate mask_fn.
//
// Template parameters:
//   Axis: N = iterate over K-blocks (fixed m_block), M = iterate over Q-blocks (fixed n_block)
//   Direction: MinToMax = ascending, MaxToMin = descending
//   PackGQA/QheadPerKhead: only used when Axis=M (m-direction needs packed/logical conversion)
//
// seqlen_q: always LOGICAL (= seqlen_info.seqlen_q, unscaled by PackGQA).
//
// step_fn(block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for no-mask zones, cute::false_type otherwise
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class DispatchAxis { N, M };
enum class DispatchDirection { MinToMax, MaxToMin };

template <
    int kBlockM,
    int kBlockN,
    bool SparseLoad,
    bool IndexAttn,
    bool PackGQA,
    int QheadPerKhead,
    DispatchAxis Axis,
    DispatchDirection Direction,
    typename StepFn,
    typename BoundaryMaskFn,
    typename RegularMaskFn,
    typename NoMaskFn>
CUTLASS_DEVICE void mask_dispatch(
    int block_start,
    int block_end,
    int fixed_block,
    int seqlen_q,
    int seqlen_k,
    flash::AttnType attn_type,
    StepFn&& step_fn,
    BoundaryMaskFn&& boundary_fn,
    RegularMaskFn&& regular_fn,
    NoMaskFn&& no_mask_fn) {
  // SparseLoad/IndexAttn: iterate all blocks with no_mask
  if constexpr (SparseLoad || IndexAttn) {
    if constexpr (Direction == DispatchDirection::MaxToMin) {
      step_fn(block_start, no_mask_fn, cute::true_type{});
    } else {
      for (int b = block_start; b < block_end; ++b)
        step_fn(b, no_mask_fn, cute::true_type{});
    }
    return;
  }

  // Empty range check
  if constexpr (Direction == DispatchDirection::MaxToMin) {
    if (block_start < block_end)
      return;
  } else {
    if (block_start >= block_end)
      return;
  }

  // ─── Axis-dependent constants ───
  // Outer block size (the axis we're iterating)
  constexpr int kBlockOuter = (Axis == DispatchAxis::N) ? kBlockN : kBlockM;
  // Fixed block size (the other axis)
  constexpr int kBlockFixed = (Axis == DispatchAxis::N) ? kBlockM : kBlockN;

  // For M-axis with PackGQA: physical seqlen_q = logical * QheadPerKhead
  int const seqlen_q_packed = (Axis == DispatchAxis::M && PackGQA) ? seqlen_q * QheadPerKhead : seqlen_q;

  // Seqlen on the traversal axis (for boundary/alignment detection)
  int const seqlen_outer = (Axis == DispatchAxis::N) ? seqlen_k : seqlen_q_packed;

  // ─── Boundary detection ───
  // N-axis: rightmost n_block may have K columns beyond seqlen_k
  // M-axis: last m_block may have Q rows beyond seqlen_q_packed, PLUS fixed n_block may be K-boundary
  bool const last_block_is_boundary = (seqlen_outer % kBlockOuter != 0) || (attn_type != flash::AttnType::Full);
  // M-axis extra: if the fixed n_block is the last K-block, every m_block needs seqlen_k masking
  bool const cross_axis_boundary = (Axis == DispatchAxis::M) && ((fixed_block + 1) * kBlockN > seqlen_k);

  // ─── Causal/InvCausal diagonal boundaries ───
  // Which mask restricts at "small end" vs "large end" depends on axis:
  //   N-axis (min→max): small=InvCausal, large=Causal
  //   N-axis (max→min): large(first)=Causal, small(last)=InvCausal (same geometry, reversed traversal)
  //   M-axis (min→max): small=Causal, large=InvCausal

  // For N-axis: boundaries in logical Q space
  //   causal_boundary: (fixed_block * kBlockFixed + seqlen_k - seqlen_q) / kBlockOuter
  //   inv_boundary: ceil_div((fixed_block + 1) * kBlockFixed, kBlockOuter)
  // For M-axis: boundaries in packed M space
  //   causal_boundary: ceil_div(((fixed_block+1)*kBlockN - (seqlen_k - seqlen_q)) * PackGQA_factor, kBlockM)
  //   inv_boundary: (fixed_block*kBlockN + 1) * PackGQA_factor / kBlockM

  int causal_end, inv_start;

  if constexpr (Axis == DispatchAxis::N) {
    // N-axis: Causal restricts at large n (right side), InvCausal at small n (left side)
    causal_end = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
        ? max(block_start, (fixed_block * kBlockM + seqlen_k - seqlen_q) / kBlockN)
        : block_start;
    inv_start = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
        ? min(block_end, cute::ceil_div((fixed_block + 1) * kBlockM, kBlockN))
        : block_end;
  } else {
    // M-axis: Causal restricts at small m (top), InvCausal at large m (bottom)
    int const pack_factor = PackGQA ? QheadPerKhead : 1;
    int const causal_no_mask_val = ((fixed_block + 1) * kBlockN - (seqlen_k - seqlen_q)) * pack_factor;
    causal_end = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
        ? (causal_no_mask_val <= 0 ? block_start : min(block_end, cute::ceil_div(causal_no_mask_val, kBlockM)))
        : block_start;
    inv_start = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
        ? min(block_end, (fixed_block * kBlockN + 1) * pack_factor / kBlockM)
        : block_end;
  }

  // ─── Traversal ───
  // For min→max: stages are [start..inv_start) [inv_start..causal_end) [causal_end..end)
  //   N-axis min→max: InvCausal(left) → NoMask → Causal(right) → Boundary(last)
  //   M-axis min→max: Causal(top) → NoMask → InvCausal(bottom) → Boundary(last)
  // For max→min: stages are Boundary(first) → Causal(right) → NoMask → InvCausal(left)

  if constexpr (Direction == DispatchDirection::MinToMax) {
    int block = block_start;
    int const last_block = block_end - 1;

    if constexpr (Axis == DispatchAxis::N) {
      // N-axis min→max: InvCausal → NoMask → Causal → Boundary
      // Stage 1: InvCausal left
      if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; block < inv_start; ++block)
          step_fn(block, regular_fn, cute::false_type{});
      }
      // Stage 2: no-mask fast path
      int const no_mask_end = (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) ? last_block : min(last_block, causal_end);
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block < no_mask_end; ++block)
        step_fn(block, no_mask_fn, cute::true_type{});
      // Stage 3: Causal right
      if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; block < last_block; ++block)
          step_fn(block, regular_fn, cute::false_type{});
      }
      // Stage 4: boundary (rightmost n_block)
      if (block < block_end) {
        if (seqlen_k % kBlockN == 0 && attn_type == flash::AttnType::Full)
          step_fn(block, no_mask_fn, cute::false_type{});
        else
          step_fn(block, boundary_fn, cute::false_type{});
      }
    } else {
      // M-axis min→max: Causal → NoMask → InvCausal → Boundary
      // Stage 1: Causal top
      if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
        int const end1 = min(causal_end, last_block);
        CUTLASS_PRAGMA_NO_UNROLL
        for (; block < end1; ++block)
          step_fn(block, regular_fn, cute::false_type{});
      }
      // Stage 2: no-mask fast path (or boundary_fn if cross_axis_boundary)
      {
        int const no_mask_end = min(inv_start, last_block);
        if (cross_axis_boundary) {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; block < no_mask_end; ++block)
            step_fn(block, boundary_fn, cute::false_type{});
        } else {
          CUTLASS_PRAGMA_NO_UNROLL
          for (; block < no_mask_end; ++block)
            step_fn(block, no_mask_fn, cute::true_type{});
        }
      }
      // Stage 3: InvCausal bottom
      if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
        CUTLASS_PRAGMA_NO_UNROLL
        for (; block < last_block; ++block)
          step_fn(block, regular_fn, cute::false_type{});
      }
      // Stage 4: last m_block boundary
      if (block == last_block) {
        if (!cross_axis_boundary && seqlen_q_packed % kBlockM == 0 && attn_type == flash::AttnType::Full)
          step_fn(block, no_mask_fn, cute::false_type{});
        else
          step_fn(block, boundary_fn, cute::false_type{});
      }
    }
  } else {
    // Direction == MaxToMin (FWD path: N-axis only)
    static_assert(Axis == DispatchAxis::N, "MaxToMin only supported for N-axis");
    int block = block_start; // starts at max (rightmost)

    // Stage 1: boundary (rightmost n_block)
    if (seqlen_k % kBlockN == 0 && attn_type == flash::AttnType::Full)
      step_fn(block, no_mask_fn, cute::false_type{});
    else
      step_fn(block, boundary_fn, cute::false_type{});
    --block;

    // Stage 2: Causal right
    if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= causal_end; --block)
        step_fn(block, regular_fn, cute::false_type{});
    }

    // Stage 3: no-mask fast path
    int const no_mask_min = (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) ? block_end : inv_start;
    CUTLASS_PRAGMA_NO_UNROLL
    for (; block >= no_mask_min; --block)
      step_fn(block, no_mask_fn, cute::true_type{});

    // Stage 4: InvCausal left
    if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= block_end; --block)
        step_fn(block, regular_fn, cute::false_type{});
    }
  }
}

//
// Traverses n_blocks right-to-left (max→min) through 4 stages:
//   1. Boundary block (rightmost, seqlen_k may not align to kBlockN)
//   2. Causal diagonal (Causal/BiCausal top-right)
//   3. No-mask fast path (zero mask overhead)
//   4. InvCausal left boundary (InvCausal/BiCausal bottom-left)
//
// For SparseLoad/IndexAttn: early return with no_mask (padding handled in mma_head).
//
// seqlen_q: always LOGICAL (= seqlen_info.seqlen_q, unscaled by PackGQA).
//
// step_fn(n_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for stage 3, cute::false_type otherwise
//     (enables compile-time branching, e.g. FWD's check_inf optimization)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, bool SparseLoad, bool IndexAttn, typename StepFn, typename BoundaryMaskFn, typename RegularMaskFn, typename NoMaskFn>
CUTLASS_DEVICE void n_block_max_to_min_mask_dispatch(
    int n_block,
    int n_block_min,
    int m_block,
    int seqlen_q,
    int seqlen_k,
    flash::AttnType attn_type,
    StepFn&& step_fn,
    BoundaryMaskFn&& boundary_fn,
    RegularMaskFn&& regular_fn,
    NoMaskFn&& no_mask_fn) {
  mask_dispatch<kBlockM, kBlockN, SparseLoad, IndexAttn, false, 1, DispatchAxis::N, DispatchDirection::MaxToMin>(
      n_block,
      n_block_min,
      m_block,
      seqlen_q,
      seqlen_k,
      attn_type,
      static_cast<StepFn&&>(step_fn),
      static_cast<BoundaryMaskFn&&>(boundary_fn),
      static_cast<RegularMaskFn&&>(regular_fn),
      static_cast<NoMaskFn&&>(no_mask_fn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// n_block_min_to_max_mask_dispatch: Unified mask dispatch for BWD loop_k (n_block min→max).
//
// Traverses n_blocks left-to-right (min→max) through 4 stages:
//   1. InvCausal left boundary (InvCausal/BiCausal bottom-left)
//   2. No-mask fast path (zero mask overhead)
//   3. Causal diagonal (Causal/BiCausal top-right)
//   4. Boundary block (rightmost, seqlen_k may not align to kBlockN)
//
// For SparseLoad/IndexAttn: early return with no_mask (padding handled separately).
//
// seqlen_q: always LOGICAL (= seqlen_info.seqlen_q, unscaled by PackGQA).
//
// step_fn(n_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for stage 2, cute::false_type otherwise
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, bool SparseLoad, bool IndexAttn, typename StepFn, typename BoundaryMaskFn, typename RegularMaskFn, typename NoMaskFn>
CUTLASS_DEVICE void n_block_min_to_max_mask_dispatch(
    int n_block_min,
    int n_block_max,
    int m_block,
    int seqlen_q,
    int seqlen_k,
    flash::AttnType attn_type,
    StepFn&& step_fn,
    BoundaryMaskFn&& boundary_fn,
    RegularMaskFn&& regular_fn,
    NoMaskFn&& no_mask_fn) {
  mask_dispatch<kBlockM, kBlockN, SparseLoad, IndexAttn, false, 1, DispatchAxis::N, DispatchDirection::MinToMax>(
      n_block_min,
      n_block_max,
      m_block,
      seqlen_q,
      seqlen_k,
      attn_type,
      static_cast<StepFn&&>(step_fn),
      static_cast<BoundaryMaskFn&&>(boundary_fn),
      static_cast<RegularMaskFn&&>(regular_fn),
      static_cast<NoMaskFn&&>(no_mask_fn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// m_block_min_to_max_mask_dispatch: Unified mask dispatch for BWD loop_q (m_block min→max).
//
// Traverses m_blocks top-to-bottom (min→max in Q dimension) with fixed n_block.
//
// seqlen_q: always LOGICAL (= seqlen_info.seqlen_q, unscaled by PackGQA).
//   Physical seqlen is derived internally as seqlen_q * QheadPerKhead when PackGQA.
//
// step_fn(m_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for no-mask zones, cute::false_type otherwise
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int kBlockM,
    int kBlockN,
    bool SparseLoad,
    bool IndexAttn,
    bool PackGQA,
    int QheadPerKhead,
    typename StepFn,
    typename BoundaryMaskFn,
    typename RegularMaskFn,
    typename NoMaskFn>
CUTLASS_DEVICE void m_block_min_to_max_mask_dispatch(
    int m_block_min,
    int m_block_max,
    int n_block,
    int seqlen_q,
    int seqlen_k,
    flash::AttnType attn_type,
    StepFn&& step_fn,
    BoundaryMaskFn&& boundary_fn,
    RegularMaskFn&& regular_fn,
    NoMaskFn&& no_mask_fn) {
  mask_dispatch<kBlockM, kBlockN, SparseLoad, IndexAttn, PackGQA, QheadPerKhead, DispatchAxis::M, DispatchDirection::MinToMax>(
      m_block_min,
      m_block_max,
      n_block,
      seqlen_q,
      seqlen_k,
      attn_type,
      static_cast<StepFn&&>(step_fn),
      static_cast<BoundaryMaskFn&&>(boundary_fn),
      static_cast<RegularMaskFn&&>(regular_fn),
      static_cast<NoMaskFn&&>(no_mask_fn));
}
} // namespace flash
