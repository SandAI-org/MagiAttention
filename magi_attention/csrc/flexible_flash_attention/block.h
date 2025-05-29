/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "mask.h"

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN>
struct BlockMN {

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(
        SeqlenInfo_t const& seqlen_info,
        int const m_block, int const bidb,
        flash::AttnType const attn_type=flash::AttnType::Full) 
    {

        int const seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) {
            // do nothing
        }
        else if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
            int m_idx_max = (m_block + 1) * kBlockM;
            // TODO: check off-by-1 error
            n_block_max = std::min(n_block_max,
                                   cute::ceil_div(m_idx_max + seqlen_k - seqlen_q, kBlockN));
        }
        int n_block_min = 0;
        if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
            n_block_min = 0;
        }
        else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
            int m_idx_min = m_block * kBlockM;
            // TODO: Check if there's a better way to compute this
            n_block_min = m_idx_min > seqlen_k ? n_block_max : m_idx_min / kBlockN;
        }
        return {n_block_min, n_block_max};

        // if constexpr (Is_local) {
        //     int m_idx_min = m_block * kBlockM;
        //     if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
        //     n_block_min = std::max(int(0), (m_idx_min + seqlen_k - seqlen_q - window_size_left) / kBlockN);
        // }
        // // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        // if constexpr (Split) {
        //     uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
        //     int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
        //     int split_idx_actual = split_idx & 0x0000FFFF;
        //     int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
        //     int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
        //     n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
        //     n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
        //     // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, num_splits_dynamic = %d, num_splits_actual = %d, num_n_blocks_per_split = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, num_splits_dynamic, num_splits_actual, num_n_blocks_per_split, n_block_min, n_block_max); }
        // }
        // // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        // return {n_block_min, n_block_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            flash::AttnType const attn_type) 
    {

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
            // do nothing
        }
        else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
            // TODO: Need better way to compute this
            m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN, kBlockM));
        }
        int m_block_min = 0;
        if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
            m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k) / kBlockM);
        }
        else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::Full) {
            // do nothing
        }
        return {m_block_min, m_block_max};
    }

};

} // namespace flash
