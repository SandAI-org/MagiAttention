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

#include "mask.h"

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN>
struct BlockMN {
  static CUTLASS_DEVICE cute::tuple<int, int> get_n_block_min_max(
      SeqlenInfo_t const& seqlen_info,
      int const m_block,
      int const bidb,
      flash::AttnType const attn_type = flash::AttnType::Full) {
    int const seqlen_k = seqlen_info.seqlen_k;
    int const seqlen_q = seqlen_info.seqlen_q;
    int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::InvCausal) {
      // do nothing
    } else if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      int m_idx_max = (m_block + 1) * kBlockM;
      // TODO: check off-by-1 error
      n_block_max = std::min(n_block_max, cute::ceil_div(m_idx_max + seqlen_k - seqlen_q, kBlockN));
    }
    int n_block_min = 0;
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
      n_block_min = 0;
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      int m_idx_min = m_block * kBlockM;
      // TODO: Check if there's a better way to compute this
      n_block_min = m_idx_min > seqlen_k ? n_block_max : m_idx_min / kBlockN;
    }
    return {n_block_min, n_block_max};
  }

  static CUTLASS_DEVICE cute::tuple<int, int> get_m_block_min_max(SeqlenInfo_t const& seqlen_info, int const n_block, int const bidb, flash::AttnType const attn_type) {
    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;
    int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
    if (attn_type == flash::AttnType::Full || attn_type == flash::AttnType::Causal) {
      // do nothing
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::BiCausal) {
      // TODO: Need better way to compute this
      m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN, kBlockM));
    }
    int m_block_min = 0;
    if (attn_type == flash::AttnType::Causal || attn_type == flash::AttnType::BiCausal) {
      m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k) / kBlockM);
    } else if (attn_type == flash::AttnType::InvCausal || attn_type == flash::AttnType::Full) {
      // do nothing
    }
    return {m_block_min, m_block_max};
  }
};

} // namespace flash
