/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

// We consolidate all the info related to sequence length here. This is so that we can do all
// the gmem reads once at the beginning of each tile, rather than having to repeat these reads
// to compute various things like n_block_min, n_block_max, etc.
// TODO: Add distributed offset to DistributedSeqlenInfo
struct DistributedSeqlenInfo {
  int const offset_q, offset_k;
  int const seqlen_q, seqlen_k;

  CUTLASS_DEVICE
  DistributedSeqlenInfo(int const bidb, int const* const q_ranges, int const* const k_ranges)
      : offset_q(q_ranges[2 * bidb]),
        offset_k(k_ranges[2 * bidb]),
        seqlen_q(q_ranges[2 * bidb + 1] - q_ranges[2 * bidb]),
        seqlen_k(k_ranges[2 * bidb + 1] - k_ranges[2 * bidb]) {}
};

} // namespace flash
