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

  // Empty range check: MaxToMin traverses [block_end, block_start] (high-to-low),
  // so empty when block_start < block_end.
  if constexpr (Direction == DispatchDirection::MaxToMin) {
    if (block_start < block_end)
      return;
  } else {
    if (block_start >= block_end)
      return;
  }

  // ─── Axis-dependent constants ───
  constexpr int kBlockOuter = (Axis == DispatchAxis::N) ? kBlockN : kBlockM;

  // For M-axis with PackGQA: physical seqlen_q = logical * QheadPerKhead
  int const seqlen_q_packed = (Axis == DispatchAxis::M && PackGQA) ? seqlen_q * QheadPerKhead : seqlen_q;

  // Seqlen on the traversal axis (for boundary/alignment detection)
  int const seqlen_outer = (Axis == DispatchAxis::N) ? seqlen_k : seqlen_q_packed;

  // M-axis extra: if the fixed n_block is the last K-block, every m_block needs seqlen_k masking
  bool const cross_axis_boundary = (Axis == DispatchAxis::M) && ((fixed_block + 1) * kBlockN > seqlen_k);

  // ─── Causal/InvCausal diagonal boundaries ───
  // When a mask is inactive, its boundary defaults to range start/end so the
  // corresponding traversal stage is empty.
  int const range_min = (Direction == DispatchDirection::MaxToMin) ? block_end : block_start;
  int const range_max = (Direction == DispatchDirection::MaxToMin) ? block_start : block_end;
  int causal_end, inv_start;

  if constexpr (Axis == DispatchAxis::N) {
    causal_end = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
        ? max(range_min, (fixed_block * kBlockM + seqlen_k - seqlen_q) / kBlockN)
        : range_max;
    inv_start = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
        ? min(range_max, cute::ceil_div((fixed_block + 1) * kBlockM, kBlockN))
        : range_min;
  } else {
    int const pack_factor = PackGQA ? QheadPerKhead : 1;
    int const causal_no_mask_val = ((fixed_block + 1) * kBlockN - (seqlen_k - seqlen_q)) * pack_factor;
    causal_end = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
        ? (causal_no_mask_val <= 0 ? range_min : min(range_max, cute::ceil_div(causal_no_mask_val, kBlockM)))
        : range_min;
    inv_start = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
        ? min(range_max, (fixed_block * kBlockN + 1) * pack_factor / kBlockM)
        : range_max;
  }

  // ─── Unified stage boundaries ───
  // Traversal stages (in MinToMax order): small-end → no-mask → large-end → boundary
  // MaxToMin reverses: boundary → large-end → no-mask → small-end
  //   N-axis: small-end = InvCausal, large-end = Causal
  //   M-axis: small-end = Causal,    large-end = InvCausal
  int const small_end = (Axis == DispatchAxis::N) ? inv_start : causal_end;
  int const large_end = (Axis == DispatchAxis::N) ? causal_end : inv_start;
  bool const has_small_mask = (Axis == DispatchAxis::N) ? (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
                                                        : (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal);
  bool const has_large_mask = (Axis == DispatchAxis::N) ? (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
                                                        : (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal);

  auto handle_boundary = [&](int block) {
    if (!cross_axis_boundary && seqlen_outer % kBlockOuter == 0 && attn_type == flash::AttnType::Full)
      step_fn(block, no_mask_fn, cute::false_type{});
    else
      step_fn(block, boundary_fn, cute::false_type{});
  };

  // ─── Traversal ───
  if constexpr (Direction == DispatchDirection::MinToMax) {
    int block = block_start;
    int const last_block = block_end - 1;

    // Stage 1: small-end mask
    if (has_small_mask) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block < min(small_end, last_block); ++block)
        step_fn(block, regular_fn, cute::false_type{});
    }
    // Stage 2: no-mask (or cross-axis boundary for M-axis)
    {
      int const no_mask_end = min(large_end, last_block);
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
    // Stage 3: large-end mask
    if (has_large_mask) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block < last_block; ++block)
        step_fn(block, regular_fn, cute::false_type{});
    }
    // Stage 4: boundary (last block)
    if (block == last_block)
      handle_boundary(block);
  } else {
    int block = block_start;

    // Stage 1: boundary (first/max block)
    handle_boundary(block);
    --block;
    // Stage 2: large-end mask
    if (has_large_mask) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= large_end; --block)
        step_fn(block, regular_fn, cute::false_type{});
    }
    // Stage 3: no-mask (or cross-axis boundary for M-axis)
    if (cross_axis_boundary) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= small_end; --block)
        step_fn(block, boundary_fn, cute::false_type{});
    } else {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= small_end; --block)
        step_fn(block, no_mask_fn, cute::true_type{});
    }
    // Stage 4: small-end mask
    if (has_small_mask) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; block >= block_end; --block)
        step_fn(block, regular_fn, cute::false_type{});
    }
  }
}

} // namespace flash
