/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

enum class AttnType {
    Full = 0,
    Causal = 1,
    InvCausal = 2,
    BiCausal = 3,
};

template <int kBlockM, int kBlockN, typename TiledMma, bool SwapAB=false>
struct Mask {
    int const thread_idx;
    int const seqlen_q, seqlen_k;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
    {};

    template <bool Seqlenk_mask=false, typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block, const flash::AttnType attn_type) const {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;

        // 处理右边界
        if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        }
        else if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                // Might get OOB but it's ok since we'll check it later
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
                    int const col_limit_right = !Seqlenk_mask
                        ? row_idx + causal_row_offset
                        : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
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
                    // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                    // row limit to be kBlockM.
                    int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        }

        // 处理左边界
        if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
        }
        else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
            if constexpr (!SwapAB) {
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    int const row_idx = get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM;
                    int const col_limit_left = row_idx - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        if (int(get<Col>(t0ScS_rowcol(_0{}, n))) < col_limit_left) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
            else {
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
                        if (int(get<Row>(t0ScS_rowcol(m, _0{}))) > row_limit_bottom) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        }
    };

};

} // namespace flash
