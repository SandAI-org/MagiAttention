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

template <DispatchDirection Dir>
CUTLASS_DEVICE int init_cursor(int lo, int hi) {
  if constexpr (Dir == DispatchDirection::MaxToMin) { return hi - 1; }
  else { return lo; }
}

template <DispatchDirection Dir, int Unroll = 1, typename BodyFn>
CUTLASS_DEVICE void iterate_range(int& cursor, int lo, int hi, BodyFn body) {
  if constexpr (Dir == DispatchDirection::MaxToMin) {
#pragma unroll Unroll
    while (cursor >= lo) { body(); --cursor; }
  } else {
#pragma unroll Unroll
    for (; cursor < hi;) { body(); ++cursor; }
  }
}

template <
    int kBlockM,
    int kBlockN,
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
  // Empty range check: MaxToMin traverses [block_end, block_start] (high-to-low),
  // so empty when block_start < block_end.
  if constexpr (Direction == DispatchDirection::MaxToMin) {
    if (block_start < block_end)
      return;
  } else {
    if (block_start >= block_end)
      return;
  }

  // ─── Constants ───
  constexpr bool is_N = (Axis == DispatchAxis::N);
  constexpr int kBlockOuter = is_N ? kBlockN : kBlockM;
  int const pack_factor = (!is_N && PackGQA) ? QheadPerKhead : 1;
  int const seqlen_outer = is_N ? seqlen_k : seqlen_q * pack_factor;
  bool const cross_axis_boundary = !is_N && ((fixed_block + 1) * kBlockN > seqlen_k);

  // ─── Diagonal boundaries → stage bounds ───
  // Inactive masks default to lo/hi so their traversal stage is empty.
  // N-axis: small = inv_causal,  large = causal
  // M-axis: small = causal,      large = inv_causal
  int const lo = min(block_start, block_end);
  int const hi = max(block_start, block_end);
  bool const has_causal = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal);
  bool const has_inv    = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal);

  int small_end, large_end;
  if constexpr (is_N) {
    small_end = has_inv    ? min(hi, cute::ceil_div((fixed_block + 1) * kBlockM, kBlockN)) : lo;
    large_end = has_causal ? max(lo, (fixed_block * kBlockM + seqlen_k - seqlen_q) / kBlockN) : hi;
  } else {
    int const cv = ((fixed_block + 1) * kBlockN - (seqlen_k - seqlen_q)) * pack_factor;
    small_end = has_causal ? (cv <= 0 ? lo : min(hi, cute::ceil_div(cv, kBlockM))) : lo;
    large_end = has_inv    ? min(hi, (fixed_block * kBlockN + 1) * pack_factor / kBlockM) : hi;
  }
  bool const has_small_mask = is_N ? has_inv : has_causal;
  bool const has_large_mask = is_N ? has_causal : has_inv;

  auto handle_boundary = [&](int block) {
    if (!cross_axis_boundary && seqlen_outer % kBlockOuter == 0 && attn_type == flash::AttnType::Full)
      step_fn(block, no_mask_fn, cute::false_type{});
    else
      step_fn(block, boundary_fn, cute::false_type{});
  };

  // ─── Traversal (unified for both directions) ───
  // MinToMax order: small_mask → no_mask → large_mask → boundary(last)
  // MaxToMin order: boundary(first) → large_mask → no_mask → small_mask
  // These are mirror images; we define stage bounds in traversal order.
  constexpr bool is_m2M = (Direction == DispatchDirection::MinToMax);
  int const last_block = block_end - 1;
  int block = block_start;

  // Boundary at direction-dependent extreme
  if constexpr (!is_m2M) {
    handle_boundary(block);
    --block;
  }

  // Stage boundaries in traversal order (first encountered → last encountered)
  int const first_end  = is_m2M ? small_end : large_end;
  int const second_end = is_m2M ? large_end : small_end;
  bool const has_first_mask = is_m2M ? has_small_mask : has_large_mask;
  bool const has_last_mask  = is_m2M ? has_large_mask : has_small_mask;
  int const tail_lo = is_m2M ? 0 : block_end;

  // Stage 1: mask nearest the starting end
  if (has_first_mask) {
    iterate_range<Direction>(block, first_end, min(first_end, last_block), [&]{
      step_fn(block, regular_fn, cute::false_type{});
    });
  }
  // Stage 2: no-mask (or cross-axis boundary for M-axis)
  if (cross_axis_boundary) {
    iterate_range<Direction>(block, second_end, min(second_end, last_block), [&]{
      step_fn(block, boundary_fn, cute::false_type{});
    });
  } else {
    iterate_range<Direction>(block, second_end, min(second_end, last_block), [&]{
      step_fn(block, no_mask_fn, cute::true_type{});
    });
  }
  // Stage 3: mask nearest the ending end
  if (has_last_mask) {
    iterate_range<Direction>(block, tail_lo, last_block, [&]{
      step_fn(block, regular_fn, cute::false_type{});
    });
  }
  // Boundary at end (MinToMax only)
  if constexpr (is_m2M) {
    if (block == last_block)
      handle_boundary(block);
  }
}

} // namespace flash
