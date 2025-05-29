/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.

template <bool Varlen, int kBlock>
struct SeqlenInfo {

    int const offset, offset_padded;
    int const seqlen;

    CUTLASS_DEVICE
    SeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const seqused)
        : offset(!Varlen || cu_seqlens == nullptr ? 0 : cu_seqlens[bidb])
        , offset_padded(!Varlen || cu_seqlens == nullptr ? 0 : (cu_seqlens[bidb] + bidb * kBlock) / kBlock * kBlock)
        , seqlen(!Varlen
                 ? seqlen_static
                 : (seqused ? seqused[bidb] : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static)))
    {
    }

};

template <bool Varlen, int kBlock>
struct SeqlenInfoBwd {
    // Reivew(littsk): offset_padded 是啥？
    int const offset, offset_padded;
    int const seqlen;

    CUTLASS_DEVICE
    SeqlenInfoBwd(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const q_ranges, int const* const seqused)
        : offset(!Varlen || (cu_seqlens == nullptr && q_ranges == nullptr) ? 0 : (q_ranges ? q_ranges[2 * bidb] : cu_seqlens[bidb]))
        , offset_padded(!Varlen || (cu_seqlens == nullptr && q_ranges == nullptr) ? 0 : ((q_ranges ? q_ranges[2 * bidb] : cu_seqlens[bidb]) + bidb * kBlock) / kBlock * kBlock)
        , seqlen(!Varlen
                 ? seqlen_static
                 : (seqused ? seqused[bidb] :
                    (q_ranges ? q_ranges[2 * bidb + 1] - q_ranges[2 * bidb]
                    : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static))))
    {
    }

};

template <bool Varlen, int kBlock>
struct SeqlenInfoFwd {

    int const offset, offset_padded;
    int const seqlen;

    CUTLASS_DEVICE
    SeqlenInfoFwd(int const bidb, int const seqlen_static, int const* const cu_seqlens, int const* const q_ranges, int const* const seqused)
        : offset(!Varlen || (cu_seqlens == nullptr && q_ranges == nullptr) ? 0 : (q_ranges ? q_ranges[2 * bidb] : cu_seqlens[bidb]))
        , offset_padded(!Varlen || (cu_seqlens == nullptr && q_ranges == nullptr) ? 0 : ((q_ranges? q_ranges[2 * bidb] : cu_seqlens[bidb]) + bidb * kBlock) / kBlock * kBlock)
        , seqlen(!Varlen
                 ? seqlen_static
                 : (seqused ? seqused[bidb] :
                    (q_ranges ? q_ranges[2 * bidb + 1] - q_ranges[2 * bidb]
                    : (cu_seqlens ? cu_seqlens[bidb + 1] - cu_seqlens[bidb] : seqlen_static))))
    {
    }

};

template <bool Varlen, int kBlockM>
struct SeqlenInfoQK {

    int const offset_q, offset_k, offset_q_padded;
    int const seqlen_q, seqlen_k;

    CUTLASS_DEVICE
    SeqlenInfoQK(int const bidb, int const seqlen_q_static, int const seqlen_k_static,
                 int const* const cu_seqlens_q, int const* const cu_seqlens_k,
                 int const* const q_ranges, int const* const k_ranges,
                 int const* const seqused_q, int const* const seqused_k
                 )
        : offset_q(!Varlen || (cu_seqlens_q == nullptr && q_ranges == nullptr) ? 0 : (q_ranges ? q_ranges[2 * bidb] : cu_seqlens_q[bidb]))
        , offset_k(!Varlen || (cu_seqlens_k == nullptr && k_ranges == nullptr) ? 0 : (k_ranges ? k_ranges[2 * bidb] : cu_seqlens_k[bidb]))
        // If varlen, the layout for dPSum, LSE_log2, and dQaccum is that we pad each sequence in the batch
        // by an extra kBlockM, so that the write for each sequence doesn't touch the next sequence.
        // Sequence i starts at cu_seqlens[i] + i * kBlockM and ends at cu_seqlens[i + 1] + i * kBlockM
        // However, the start must align to multiples of kBlockM.
        , offset_q_padded(!Varlen || (cu_seqlens_q == nullptr && q_ranges == nullptr) ? 0 : ((q_ranges ? q_ranges[2 * bidb] : cu_seqlens_q[bidb]) + bidb * kBlockM) / kBlockM * kBlockM)
        , seqlen_q(!Varlen
                   ? seqlen_q_static
                   : (seqused_q ? seqused_q[bidb] :
                      (q_ranges ? q_ranges[2 * bidb + 1] - q_ranges[2 * bidb]
                      : (cu_seqlens_q ? cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb] : seqlen_q_static))))
        , seqlen_k(!Varlen
                   ? seqlen_k_static
                   : (seqused_k ? seqused_k[bidb] :
                      (k_ranges ? k_ranges[2 * bidb + 1] - k_ranges[2 * bidb]
                      : (cu_seqlens_k ? cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb] : seqlen_k_static))))
    {
    }

};

// TODO: Add distributed offset to DistributedSeqlenInfo
struct DistributedSeqlenInfo {
    int const offset_q, offset_k;
    int const seqlen_q, seqlen_k;

    CUTLASS_DEVICE
    DistributedSeqlenInfo(int const bidb, int const* const q_ranges, int const* const k_ranges)
        : offset_q(q_ranges[2 * bidb])
        , offset_k(k_ranges[2 * bidb])
        , seqlen_q(q_ranges[2 * bidb + 1] - q_ranges[2 * bidb])
        , seqlen_k(k_ranges[2 * bidb + 1] - k_ranges[2 * bidb])
    {
    }
};

} // namespace flash
