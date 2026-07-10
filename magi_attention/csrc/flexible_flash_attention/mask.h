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
template <int kBlockM, int kBlockN, typename TiledMma, bool SwapAB>
struct Mask {
  // Apply mask to the tensor tSrS based on attention type and sequence lengths
  template <bool Seqlenk_mask, bool PackGQA, int PackGQAFactor, typename Engine, typename Layout>
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
          // for packgqa, the actual row index need to divide by PackGQAFactor
          int const logical_row_idx = !PackGQA ? physical_row_idx : (physical_row_idx / PackGQAFactor);
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
          // Convert to physical limit: limit * PackGQAFactor
          // Transform to local coordinate: - m_block_offset - thread_offset
          int const row_limit_global = (global_k - dist) * (!PackGQA ? 1 : PackGQAFactor);
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
          int const logical_row_idx = !PackGQA ? physical_row_idx : (physical_row_idx / PackGQAFactor);
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
          // With PackGQA, one Key corresponds to 'G' Query heads (G = PackGQAFactor).
          // Therefore, for a specific Key 'K', the valid physical Query range extends
          // to the last head in the group: Max_Q_phys = K * G + (G - 1).
          int const row_limit_global = !PackGQA ? global_k : (global_k * PackGQAFactor + (PackGQAFactor - 1));

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

  // Mask invalid columns (N-side padding, for BlockSparse LoopK / IndexSparse).
  // MaskLow=false (default): masks columns >= kBlockN - num_invalid (high end, for MinToMax).
  // MaskLow=true: masks columns < num_invalid (low end, for MaxToMin where padding
  //   occupies the beginning of the tile due to the reversed scatter fill order).
  template <bool MaskLow = false, typename Engine, typename Layout>
  CUTLASS_DEVICE void apply_padding_mask(Tensor<Engine, Layout>& tSrS, int num_invalid_token, int thread_idx) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

    static constexpr int Col = !SwapAB ? 1 : 0;

    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

    int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));

    if constexpr (MaskLow) {
      int const low_col_limit = num_invalid_token - thread_col_offset;
#pragma unroll
      for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
        if (int(get<Col>(t0ScS_rowcol(_0{}, n))) < low_col_limit) {
#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            tSrS_rowcol(m, n) = -INFINITY;
          }
        }
      }
    } else {
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
  }

  // Mask invalid rows (M-side padding, for BlockSparse LoopQ).
  // MaskLow=false (default): masks rows >= kBlockM - num_invalid (high end, MinToMax).
  // MaskLow=true: masks rows < num_invalid (low end, MaxToMin).
  template <bool MaskLow = false, typename Engine, typename Layout>
  CUTLASS_DEVICE void apply_padding_mask_row(Tensor<Engine, Layout>& tSrS, int num_invalid_token, int thread_idx) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

    static constexpr int Row = !SwapAB ? 0 : 1;

    Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

    int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));

    if constexpr (MaskLow) {
      int const low_row_limit = num_invalid_token - thread_row_offset;
#pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < low_row_limit) {
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            tSrS_rowcol(m, n) = -INFINITY;
          }
        }
      }
    } else {
      int const seqlenq_row_limit = kBlockM - num_invalid_token - thread_row_offset;
#pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        if (int(get<Row>(t0ScS_rowcol(m, _0{}))) >= seqlenq_row_limit) {
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            tSrS_rowcol(m, n) = -INFINITY;
          }
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
//   PackGQA/PackGQAFactor: only used when Axis=M (m-direction needs packed/logical conversion)
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
CUTLASS_DEVICE int init_block_cur(int lo, int hi) {
  if constexpr (Dir == DispatchDirection::MaxToMin) {
    return hi - 1;
  } else {
    return lo;
  }
}

template <DispatchDirection Dir>
CUTLASS_DEVICE void advance_block_cur(int& block_cur) {
  if constexpr (Dir == DispatchDirection::MaxToMin) {
    --block_cur;
  } else {
    ++block_cur;
  }
}

template <DispatchDirection Dir, int Unroll = 1, typename BodyFn>
CUTLASS_DEVICE void iterate_range(int& cursor, int lo, int hi, BodyFn body) {
  if constexpr (Dir == DispatchDirection::MaxToMin) {
#pragma unroll Unroll
    while (cursor >= lo) {
      body();
      --cursor;
    }
  } else {
#pragma unroll Unroll
    for (; cursor < hi;) {
      body();
      ++cursor;
    }
  }
}

// Single-instantiation mask dispatch: computes zone boundaries internally,
// applies mask directly via MaskT::apply, and calls step_fn exactly once per block.
// This avoids code bloat from multiple step_fn instantiations with different mask types.
//
// WARNING: BWD does NOT use this function. When inlined into BWD's consumer warp group,
// the 14+ local variables (zone boundaries, runtime flags) exhaust the 224-register budget
// and cause register spill (local_ld/st). BWD uses a direct mask_fn(tSrS, block) instead.
// This function is retained for FWD use only.
//
// BlockMetaT must provide: outer_tile_idx, inner_block_min, inner_block_cnt,
//   seqlen_info.seqlen_q, seqlen_info.seqlen_k, attn_type.
// MaskT must provide: apply<Seqlenk_mask, PackGQA, PackGQAFactor>(tSrS, m_block, n_block, ...).
template <
    int kBlockM,
    int kBlockN,
    bool PackGQA,
    int PackGQAFactor,
    DispatchAxis Axis,
    DispatchDirection Direction,
    typename BlockMetaT,
    typename MaskT,
    typename TensorS,
    typename StepFn>
CUTLASS_DEVICE void mask_dispatch_unified(BlockMetaT const& block_meta, MaskT const& mask, TensorS& tSrS, int thread_idx, StepFn&& step_fn) {
  constexpr bool is_N = (Axis == DispatchAxis::N);
  constexpr int kBlockOuter = is_N ? kBlockN : kBlockM;

  int const block_lo = block_meta.inner_block_min;
  int const block_hi = block_meta.inner_block_cnt;
  int const fixed_block = block_meta.outer_tile_idx;
  int const seqlen_q = block_meta.seqlen_info.seqlen_q;
  int const seqlen_k = block_meta.seqlen_info.seqlen_k;
  auto const attn_type = block_meta.attn_type;

  int const pack_factor = (!is_N && PackGQA) ? PackGQAFactor : 1;
  int const seqlen_outer = is_N ? seqlen_k : seqlen_q * pack_factor;
  bool const cross_axis_boundary = !is_N && ((fixed_block + 1) * kBlockN > seqlen_k);

  bool const has_causal = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal);
  bool const has_inv = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal);

  // Zone boundaries
  int small_end, large_end;
  if constexpr (is_N) {
    small_end = has_inv ? min(block_hi, cute::ceil_div((fixed_block + 1) * kBlockM, kBlockN)) : block_lo;
    large_end = has_causal ? max(block_lo, (fixed_block * kBlockM + seqlen_k - seqlen_q) / kBlockN) : block_hi;
  } else {
    int const cv = ((fixed_block + 1) * kBlockN - (seqlen_k - seqlen_q)) * pack_factor;
    small_end = has_causal ? (cv <= 0 ? block_lo : min(block_hi, cute::ceil_div(cv, kBlockM))) : block_lo;
    large_end = has_inv ? min(block_hi, (fixed_block * kBlockN + 1) * pack_factor / kBlockM) : block_hi;
  }
  bool const has_small_mask = is_N ? has_inv : has_causal;
  bool const has_large_mask = is_N ? has_causal : has_inv;

  int const last_block = block_hi - 1;
  bool const last_block_no_mask = !cross_axis_boundary && (seqlen_outer % kBlockOuter == 0) && (attn_type == flash::AttnType::Full);

  // Internal mask application: dispatches m_block/n_block based on Axis.
  // Axis=N: fixed_block is m_block, block is n_block.
  // Axis=M: block is m_block, fixed_block is n_block.
  auto apply_mask = [&](int block, bool seqlenk_mask) {
    int const m_blk = is_N ? fixed_block : block;
    int const n_blk = is_N ? block : fixed_block;
    if (seqlenk_mask)
      mask.template apply</*Seqlenk_mask=*/true, PackGQA, PackGQAFactor>(tSrS, m_blk, n_blk, attn_type, thread_idx, seqlen_q, seqlen_k);
    else
      mask.template apply</*Seqlenk_mask=*/false, PackGQA, PackGQAFactor>(tSrS, m_blk, n_blk, attn_type, thread_idx, seqlen_q, seqlen_k);
  };

  // Unified mask function: ONE lambda type, runtime dispatch.
  // seqlenk_mask=true  → boundary_fn (diagonal mask + seqlen clipping)
  // seqlenk_mask=false → regular_fn  (diagonal mask only)
  // else (no call)     → no_mask_fn  (fully visible, skip mask entirely)
  auto unified_mask_fn = [&](int block) {
    if (block == last_block && !last_block_no_mask) {
      apply_mask(block, /*seqlenk_mask=*/true); // boundary: last block may have seqlen padding
    } else if ((has_small_mask && block < small_end) || (has_large_mask && block >= large_end)) {
      apply_mask(block, /*seqlenk_mask=*/false); // diagonal zone: causal/inv_causal mask
    } else if (cross_axis_boundary) {
      apply_mask(block, /*seqlenk_mask=*/true); // M-axis: fixed n_block exceeds seqlen_k
    }
    // else: no mask needed — block is fully visible
  };

  // Use inner_block_idx as traversal start (may differ from init_block_cur after head processing).
  // FWD: mma_head already processed the first block and advanced inner_block_idx,
  //      so we must start from the advanced position, not from block_lo/block_hi.
  // BWD: no mma_head pre-processing, inner_block_idx == init_block_cur(block_lo, block_hi).
  int block_cur = block_meta.inner_block_idx;
  iterate_range<Direction>(block_cur, block_lo, block_hi, [&] { step_fn(block_cur, unified_mask_fn, cute::false_type{}); });
}

// mask_dispatch: 3-lambda variant with compile-time zone splitting.
// Each zone (boundary / diagonal-mask / no-mask) invokes a dedicated lambda,
// giving the compiler full specialization → zero runtime branching in the inner loop.
// Preferred for FWD Dense where compute-bound kernels benefit from branch-free hot paths.
template <
    int kBlockM,
    int kBlockN,
    bool PackGQA,
    int PackGQAFactor,
    DispatchAxis Axis,
    DispatchDirection Direction,
    typename StepFn,
    typename BoundaryMaskFn,
    typename RegularMaskFn,
    typename NoMaskFn>
CUTLASS_DEVICE void mask_dispatch(
    int block_cur,
    int block_lo,
    int block_hi,
    int fixed_block,
    int seqlen_q,
    int seqlen_k,
    flash::AttnType attn_type,
    StepFn&& step_fn,
    BoundaryMaskFn&& boundary_fn,
    RegularMaskFn&& regular_fn,
    NoMaskFn&& no_mask_fn) {
  if constexpr (Direction == DispatchDirection::MaxToMin) {
    if (block_cur < block_lo)
      return;
  } else {
    if (block_cur >= block_hi)
      return;
  }

  constexpr bool is_N = (Axis == DispatchAxis::N);
  constexpr int kBlockOuter = is_N ? kBlockN : kBlockM;
  int const pack_factor = (!is_N && PackGQA) ? PackGQAFactor : 1;
  int const seqlen_outer = is_N ? seqlen_k : seqlen_q * pack_factor;
  bool const cross_axis_boundary = !is_N && ((fixed_block + 1) * kBlockN > seqlen_k);

  bool const has_causal = (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal);
  bool const has_inv = (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal);

  int small_end, large_end;
  if constexpr (is_N) {
    small_end = has_inv ? min(block_hi, cute::ceil_div((fixed_block + 1) * kBlockM, kBlockN)) : block_lo;
    large_end = has_causal ? max(block_lo, (fixed_block * kBlockM + seqlen_k - seqlen_q) / kBlockN) : block_hi;
  } else {
    int const cv = ((fixed_block + 1) * kBlockN - (seqlen_k - seqlen_q)) * pack_factor;
    small_end = has_causal ? (cv <= 0 ? block_lo : min(block_hi, cute::ceil_div(cv, kBlockM))) : block_lo;
    large_end = has_inv ? min(block_hi, (fixed_block * kBlockN + 1) * pack_factor / kBlockM) : block_hi;
  }
  bool const has_small_mask = is_N ? has_inv : has_causal;
  bool const has_large_mask = is_N ? has_causal : has_inv;

  auto handle_boundary = [&](int block) {
    if (!cross_axis_boundary && seqlen_outer % kBlockOuter == 0 && attn_type == flash::AttnType::Full)
      step_fn(block, no_mask_fn, cute::false_type{});
    else
      step_fn(block, boundary_fn, cute::false_type{});
  };

  constexpr bool is_m2M = (Direction == DispatchDirection::MinToMax);
  int const last_block = block_hi - 1;

  if constexpr (!is_m2M) {
    handle_boundary(block_cur);
    --block_cur;
  }

  int const first_end = is_m2M ? small_end : large_end;
  int const second_end = is_m2M ? large_end : small_end;
  bool const has_first_mask = is_m2M ? has_small_mask : has_large_mask;
  bool const has_last_mask = is_m2M ? has_large_mask : has_small_mask;
  int const tail_lo = is_m2M ? 0 : block_lo;

  if (has_first_mask) {
    iterate_range<Direction>(block_cur, first_end, min(first_end, last_block), [&] { step_fn(block_cur, regular_fn, cute::false_type{}); });
  }
  if (cross_axis_boundary) {
    iterate_range<Direction>(block_cur, second_end, min(second_end, last_block), [&] { step_fn(block_cur, boundary_fn, cute::false_type{}); });
  } else {
    iterate_range<Direction>(block_cur, second_end, min(second_end, last_block), [&] { step_fn(block_cur, no_mask_fn, cute::true_type{}); });
  }
  if (has_last_mask) {
    iterate_range<Direction>(block_cur, tail_lo, last_block, [&] { step_fn(block_cur, regular_fn, cute::false_type{}); });
  }
  if constexpr (is_m2M) {
    if (block_cur == last_block)
      handle_boundary(block_cur);
  }
}

} // namespace flash
