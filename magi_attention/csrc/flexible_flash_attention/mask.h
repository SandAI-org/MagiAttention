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
// n_block_high_low_mask_dispatch: Unified mask dispatch for FWD (high-to-low n_block traversal).
//
// Traverses n_blocks right-to-left (high→low) through 4 stages:
//   1. Boundary block (rightmost, seqlen_k may not align to kBlockN)
//   2. Causal diagonal (Causal/BiCausal top-right)
//   3. No-mask fast path (zero mask overhead)
//   4. InvCausal left boundary (InvCausal/BiCausal bottom-left)
//
// For SparseLoad/IndexAttn: early return with no_mask (padding handled in mma_head).
//
// step_fn(n_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for stage 3, cute::false_type otherwise
//     (enables compile-time branching, e.g. FWD's check_inf optimization)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, bool SparseLoad, bool IndexAttn,
          typename StepFn, typename BoundaryMaskFn, typename RegularMaskFn, typename NoMaskFn>
CUTLASS_DEVICE void n_block_high_low_mask_dispatch(
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
  if constexpr (SparseLoad || IndexAttn) {
    step_fn(n_block, no_mask_fn, cute::true_type{});
    return;
  }

  if (n_block < n_block_min)
    return;

  // Stage 1: boundary (rightmost block, seqlen_k may not align to kBlockN)
  if (seqlen_k % kBlockN == 0 && attn_type == flash::AttnType::Full)
    step_fn(n_block, no_mask_fn, cute::false_type{});
  else
    step_fn(n_block, boundary_fn, cute::false_type{});
  --n_block;

  // Stage 2: causal diagonal (Causal/BiCausal top-right)
  if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
    int const n_min_causal = max(n_block_min, (m_block * kBlockM + seqlen_k - seqlen_q) / kBlockN);
    CUTLASS_PRAGMA_NO_UNROLL
    for (; n_block >= n_min_causal; --n_block)
      step_fn(n_block, regular_fn, cute::false_type{});
  }

  // Stage 3: no-mask fast path (full attention, zero overhead)
  int const n_min_inv = (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) ? n_block_min : cute::ceil_div((m_block + 1) * kBlockM, kBlockN);
  CUTLASS_PRAGMA_NO_UNROLL
  for (; n_block >= n_min_inv; --n_block)
    step_fn(n_block, no_mask_fn, cute::true_type{});

  // Stage 4: inv-causal left boundary (InvCausal/BiCausal bottom-left)
  if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (; n_block >= n_block_min; --n_block)
      step_fn(n_block, regular_fn, cute::false_type{});
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// n_block_low_high_mask_dispatch: Unified mask dispatch for BWD loop_k (low-to-high n_block).
//
// Traverses n_blocks left-to-right (low→high) through 4 stages:
//   1. InvCausal left boundary (InvCausal/BiCausal bottom-left)
//   2. No-mask fast path (zero mask overhead)
//   3. Causal diagonal (Causal/BiCausal top-right)
//   4. Boundary block (rightmost, seqlen_k may not align to kBlockN)
//
// For SparseLoad/IndexAttn: early return with no_mask (padding handled separately).
//
// step_fn(n_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for stage 2, cute::false_type otherwise
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, bool SparseLoad, bool IndexAttn,
          typename StepFn, typename BoundaryMaskFn, typename RegularMaskFn, typename NoMaskFn>
CUTLASS_DEVICE void n_block_low_high_mask_dispatch(
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
  if constexpr (SparseLoad || IndexAttn) {
    for (int n_block = n_block_min; n_block < n_block_max; ++n_block)
      step_fn(n_block, no_mask_fn, cute::true_type{});
    return;
  }

  if (n_block_min >= n_block_max)
    return;

  int n_block = n_block_min;

  // Stage 1: inv-causal left boundary (InvCausal/BiCausal bottom-left)
  if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
    int const n_max_inv = min(n_block_max, cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    CUTLASS_PRAGMA_NO_UNROLL
    for (; n_block < n_max_inv; ++n_block)
      step_fn(n_block, regular_fn, cute::false_type{});
  }

  // Stage 2: no-mask fast path (full attention, zero overhead)
  int const n_max_causal = (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal)
      ? n_block_max - 1
      : min(n_block_max - 1, (m_block * kBlockM + seqlen_k - seqlen_q) / kBlockN);
  CUTLASS_PRAGMA_NO_UNROLL
  for (; n_block < n_max_causal; ++n_block)
    step_fn(n_block, no_mask_fn, cute::true_type{});

  // Stage 3: causal diagonal (Causal/BiCausal top-right)
  if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (; n_block < n_block_max - 1; ++n_block)
      step_fn(n_block, regular_fn, cute::false_type{});
  }

  // Stage 4: boundary (rightmost block, seqlen_k may not align to kBlockN)
  if (n_block < n_block_max) {
    if (seqlen_k % kBlockN == 0 && attn_type == flash::AttnType::Full)
      step_fn(n_block, no_mask_fn, cute::false_type{});
    else
      step_fn(n_block, boundary_fn, cute::false_type{});
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// m_block_low_high_mask_dispatch: Unified mask dispatch for BWD loop_q (low-to-high m_block).
//
// Traverses m_blocks low-to-high (top→bottom in Q dimension) with fixed n_block.
// Geometry (for fixed n_block, traversing m_block from small to large):
//   - Causal (j <= i+offset): restricts at TOP (small m), visible at BOTTOM (large m)
//   - InvCausal (j >= i): visible at TOP (small m), restricts at BOTTOM (large m)
//
// Stages:
//   1. Causal diagonal (top — small m where Causal restricts) → regular_fn
//   2. No-mask fast path (between Causal end and InvCausal start) → no_mask_fn
//   3. InvCausal diagonal (bottom — large m where InvCausal restricts) → regular_fn
//   4. Boundary (last m_block): seqlen_q may not align to kBlockM
//
// For SparseLoad/IndexAttn: early return with no_mask.
//
// step_fn(m_block, mask_fn, is_no_mask_stage):
//   - mask_fn: one of {boundary_fn, regular_fn, no_mask_fn}
//   - is_no_mask_stage: cute::true_type for no-mask zones, cute::false_type otherwise
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, bool SparseLoad, bool IndexAttn, bool PackGQA, int QheadPerKhead,
          typename StepFn, typename BoundaryMaskFn, typename RegularMaskFn, typename NoMaskFn>
CUTLASS_DEVICE void m_block_low_high_mask_dispatch(
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
  if constexpr (SparseLoad || IndexAttn) {
    for (int m_block = m_block_min; m_block < m_block_max; ++m_block)
      step_fn(m_block, no_mask_fn, cute::true_type{});
    return;
  }

  if (m_block_min >= m_block_max)
    return;

  int m_block = m_block_min;
  int const last_m = m_block_max - 1;

  // n_block is the fixed K-block for this loop_q iteration.
  // If n_block is the last K-block and seqlen_k is not aligned, we need
  // seqlen_k boundary masking even in "no causal mask" regions.
  bool const n_is_boundary = ((n_block + 1) * kBlockN > seqlen_k);

  // For PackGQA, seqlen_q is already physical (= logical * QheadPerKhead).
  // With PackGQA: logical_m = physical_m / QheadPerKhead.
  int const seqlen_q_logical = !PackGQA ? seqlen_q : seqlen_q / QheadPerKhead;

  // ─── Geometry for Causal (j <= i + offset, offset = seqlen_k - seqlen_q): ───
  // Going low-to-high in m_block (fixed n_block):
  //   - small m (top): some j > i+offset → Causal DIAGONAL (needs mask)
  //   - large m (bottom): all j <= i+offset → FULLY VISIBLE (no mask)
  // m_causal_end: first m_block where ALL positions pass causal.
  //   Condition: (n+1)*N-1 <= m*M + offset → m >= ceil(((n+1)*N - offset) / M)
  //   With PackGQA: offset in logical space, m in packed space.
  int const causal_no_mask_val = ((n_block + 1) * kBlockN - (seqlen_k - seqlen_q_logical)) * (!PackGQA ? 1 : QheadPerKhead);
  int const m_causal_end = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal)
      ? (causal_no_mask_val <= 0 ? m_block_min : min(m_block_max, cute::ceil_div(causal_no_mask_val, kBlockM)))
      : m_block_min;

  // ─── Geometry for InvCausal (j >= i): ───
  // Going low-to-high in m_block (fixed n_block):
  //   - small m (top): all j >= i → FULLY VISIBLE (no mask)
  //   - large m (bottom): some j < i → InvCausal DIAGONAL (needs mask)
  // m_inv_start: first m_block where InvCausal starts restricting (diagonal begins).
  //   Block fully visible when: min(j) >= max(i) → n*N >= (m+1)*M - 1
  //   → m < (n*N + 1) / M, so m_inv_start = (n*N + 1) / M
  //   For PackGQA: logical_i = physical_i / QheadPerKhead, so n*N >= (m+1)*M/QPK - 1
  //   → (m+1)*M <= (n*N + 1) * QPK → m_inv_start = (n*N + 1) * QPK / M
  int const m_inv_start = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal)
      ? min(m_block_max, (n_block * kBlockN + 1) * (!PackGQA ? 1 : QheadPerKhead) / kBlockM)
      : m_block_max;

  // Stage 1: Causal diagonal region (top — small m, Causal restricts)
  if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
    int const end1 = min(m_causal_end, last_m);
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < end1; ++m_block)
      step_fn(m_block, regular_fn, cute::false_type{});
  }

  // Stage 2: No-mask fast path (between Causal end and InvCausal start)
  {
    int const no_mask_end = min(m_inv_start, last_m);
    if (n_is_boundary) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < no_mask_end; ++m_block)
        step_fn(m_block, boundary_fn, cute::false_type{});
    } else {
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < no_mask_end; ++m_block)
        step_fn(m_block, no_mask_fn, cute::true_type{});
    }
  }

  // Stage 3: InvCausal diagonal region (bottom — large m, InvCausal restricts)
  if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < last_m; ++m_block)
      step_fn(m_block, regular_fn, cute::false_type{});
  }

  // Stage 4: last m_block (boundary — seqlen_q may not align to kBlockM)
  if (m_block == last_m) {
    if (!n_is_boundary && seqlen_q % kBlockM == 0 && attn_type == flash::AttnType::Full)
      step_fn(m_block, no_mask_fn, cute::false_type{});
    else
      step_fn(m_block, boundary_fn, cute::false_type{});
  }
}
} // namespace flash
