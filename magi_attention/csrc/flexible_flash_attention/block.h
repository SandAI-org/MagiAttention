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

#include "mask.h"

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool PackGQA = false, int Qhead_per_khead = 1>
struct BlockMN {
  static CUTLASS_DEVICE cute::tuple<int, int> get_n_block_min_max(
      SeqlenInfo_t const& seqlen_info,
      int const m_block,
      int const bidb,
      flash::AttnType const attn_type = flash::AttnType::Full) {
    int const seqlen_k = seqlen_info.seqlen_k;
    int const seqlen_q = seqlen_info.seqlen_q;
    int n_block_max = cute::ceil_div(seqlen_k, kBlockN);

    // for packgqa, the actual m_idx_max should be divided by Qhead_per_khead
    int m_idx_max_logical = !PackGQA ? (m_block + 1) * kBlockM : cute::ceil_div((m_block + 1) * kBlockM, Qhead_per_khead);

    int m_idx_min_logical = !PackGQA ? m_block * kBlockM : (m_block * kBlockM / Qhead_per_khead);

    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) {
      // do nothing
    } else if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      int m_idx_max = std::min(seqlen_q, m_idx_max_logical);
      // TODO: check off-by-1 error
      n_block_max = std::min(n_block_max, cute::ceil_div(std::max(0, m_idx_max + seqlen_k - seqlen_q), kBlockN));
    }
    int n_block_min = 0;
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
      n_block_min = 0;
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      // TODO: Check if there's a better way to compute this
      n_block_min = m_idx_min_logical >= seqlen_k ? n_block_max : m_idx_min_logical / kBlockN;
    }
    return {n_block_min, n_block_max};
  }

  // For backward with packgqa, m_block is in packed space, n_block is in logical space
  static CUTLASS_DEVICE cute::tuple<int, int> get_m_block_min_max(SeqlenInfo_t const& seqlen_info, int const n_block, int const bidb, flash::AttnType const attn_type) {
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;

    // For PackGQA, the packed seqlen_q is seqlen_q * Qhead_per_khead
    int const seqlen_q_packed = !PackGQA ? seqlen_q : seqlen_q * Qhead_per_khead;
    int m_block_max = cute::ceil_div(seqlen_q_packed, kBlockM);

    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
      // do nothing
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      // n_idx_max in logical space (max n_idx for this n_block)
      int n_idx_max = std::min(seqlen_k, (n_block + 1) * kBlockN);
      // For PackGQA, convert to packed m space: m_idx_packed = m_idx_logical * Qhead_per_khead
      // For InvCausal (m >= n), m_idx must be >= n_idx, so in packed space: m_idx_packed >= n_idx * Qhead_per_khead
      int m_idx_max_packed = !PackGQA ? n_idx_max : n_idx_max * Qhead_per_khead;
      m_block_max = std::min(m_block_max, cute::ceil_div(m_idx_max_packed, kBlockM));
    }

    int m_block_min = 0;
    if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      // For Causal (m <= n + offset), where offset = seqlen_q - seqlen_k
      // Given n_block, n_idx_min = n_block * kBlockN
      // m_idx_min (logical) = n_idx_min + seqlen_q - seqlen_k
      int m_idx_min_logical = n_block * kBlockN + seqlen_q - seqlen_k;
      // For PackGQA, convert to packed space
      int m_idx_min_packed = !PackGQA ? m_idx_min_logical : m_idx_min_logical * Qhead_per_khead;
      m_block_min = std::max(m_block_min, m_idx_min_packed / kBlockM);
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::Full) {
      // do nothing
    }

    return {m_block_min, m_block_max};
  }
};

} // namespace flash
