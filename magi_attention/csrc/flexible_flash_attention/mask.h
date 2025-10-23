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
  template <bool Seqlenk_mask = false, typename Engine, typename Layout>
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
          int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
          int const col_limit_right = !Seqlenk_mask ? row_idx + causal_row_offset : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else {
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const causal_row_offset = seqlenk_col_limit - (seqlen_q - m_block * kBlockM - thread_row_offset);
        // row + sk - n_block * kBlockN - thread_col_offset - sq + m_block * kBlockM + thread_row_offset < col0
        // row + m_block * kBlockM + thread_row_offset - (sq - sk) < col0 + n_block * kBlockN + thread_col_offset
#pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
          // If col0 is beyond the column limit, mask out the entire column by setting row limit to kBlockM
          int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) {
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
          int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
          int const col_limit_left = row_idx - n_block * kBlockN - thread_col_offset;
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) < col_limit_left) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      } else {
        int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
        int const inv_causal_row_offset = seqlenk_col_limit + m_block * kBlockM + thread_row_offset - seqlen_k;
        // row + sk - n_block * kBlockN - thread_col_offset + m_block * kBlockM + thread_row_offset >= col0
        // row + m_block * kBlockM + thread_row_offset > col0 + n_block * kBlockN + thread_col_offset
#pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
          int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
          int const row_limit_bottom = col0 >= seqlenk_col_limit ? 0 : col0 - inv_causal_row_offset;
#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) > row_limit_bottom) {
              tSrS_rowcol(m, n) = -INFINITY;
            }
          }
        }
      }
    }
  };
};
} // namespace flash
