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

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

#include "block.h"
#include "mask.h"
#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// DenseBlockMeta: Unified FWD + BWD Dense path BlockMeta.
//
// InnerLoopQ=false  =>  FWD / BWD-LoopK (inner loop over n_block/K).
// InnerLoopQ=true   =>  BWD-LoopQ       (inner loop over m_block/Q).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IsProducer, bool InnerLoopQ, bool RangeMerge, bool FlattenGQA, int QheadPerKhead, typename SeqlenInfo_t, typename BlockMN_t>
struct DenseBlockMeta {
  // All fields are by-value (no reference data members) to avoid register spilling to stack.
  // When !RangeMerge, the batch loop runs exactly once; mark it so callers can elide the while(true).
  static constexpr bool NeedsBatchLoop = RangeMerge;

  int const outer_block; // m_block when !InnerLoopQ, n_block when InnerLoopQ
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;

  SeqlenInfo_t seqlen_info;
  flash::AttnType attn_type;
  int inner_block_min; // n_block_min when !InnerLoopQ, m_block_min when InnerLoopQ
  int inner_block_max; // n_block_max when !InnerLoopQ, m_block_max when InnerLoopQ
  int inner_block_cur;

  int2 const* const q_ranges;
  int2 const* const k_ranges;
  int const* const attn_type_map;

  template <typename ParamsT, typename BlockCoordT, typename SharedStorage>
  CUTLASS_DEVICE DenseBlockMeta(ParamsT const& params, BlockCoordT const& block_coord, SharedStorage& shared_storage, int thread_idx = 0)
      : outer_block(get<0>(block_coord)),
        bidh(get<1>(block_coord)),
        // When FlattenGQA (PackGQA or CatGQA), the scheduler assigns bidh as
        // the kv-head index directly. Otherwise bidh is the q-head index and
        // we need to divide by QheadPerKhead to get bidh_kv.
        bidh_kv(!FlattenGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh),
        q_ranges(params.q_ranges),
        k_ranges(params.k_ranges),
        attn_type_map(params.attn_type_map) {
    bidb = [&]() {
      if constexpr (RangeMerge) {
        return load_and_broadcast<1>(&params.cu_batches[get<2>(block_coord)]);
      } else {
        return get<2>(block_coord);
      }
    }();

    end_batches = [&]() {
      if constexpr (RangeMerge) {
        return load_and_broadcast<1>(&params.cu_batches[get<2>(block_coord) + 1]);
      } else {
        return bidb + 1;
      }
    }();

    if (!is_finish()) {
      seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
      update_attn_and_bounds();
    }
  }

  CUTLASS_DEVICE
  void update_attn_and_bounds() {
    attn_type = static_cast<flash::AttnType>(attn_type_map ? load_and_broadcast<1>(&attn_type_map[bidb]) : 0);
    auto [min_, max_] = InnerLoopQ ? BlockMN_t::get_m_block_min_max(seqlen_info, outer_block, bidb, attn_type)
                                   : BlockMN_t::get_n_block_min_max(seqlen_info, outer_block, bidb, attn_type);
    inner_block_min = min_;
    inner_block_max = max_;
  }

  CUTLASS_DEVICE
  void prefetch() {
    ++bidb;
    if constexpr (RangeMerge) {
      if (!is_finish()) {
        if constexpr (!InnerLoopQ) {
          seqlen_info.update_k(bidb);
        } else {
          seqlen_info.update_q(bidb);
        }
        update_attn_and_bounds();
      }
    }
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_block, bidh, bidb);
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return inner_block_min < inner_block_max;
  }

  CUTLASS_DEVICE
  bool is_finish() {
    return bidb >= end_batches;
  }

  template <flash::DispatchDirection Dir>
  CUTLASS_DEVICE void update_block_cur() {
    inner_block_cur = flash::init_block_cur<Dir>(inner_block_min, inner_block_max);
  }

  CUTLASS_DEVICE
  bool skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
    return is_finish();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SparseLoadBlockMeta: Unified producer/consumer via IsProducer template parameter.
// Replaces both old SparseLoadBlockMeta AND SparseMmaBlockMeta.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool IsProducer,
    bool RangeMerge,
    bool PackGQA,
    int QheadPerKhead,
    int NumRowsPerGroup_,
    int GroupSize_,
    int NumProducerThreads_,
    int kBlockN_,
    bool InnerDirMaxToMin_>
struct SparseLoadBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr bool NeedsBatchLoop = true;

  int const outer_block;
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;
  flash::SeqlenInfo seqlen_info;
  flash::AttnType attn_type;

  int num_invalid_token;
  int inner_block_cur;
  int inner_block_max;

  static constexpr int inner_block_min = 0;

  int2 const* const q_ranges;
  int2 const* const k_ranges;
  int const* const attn_type_map;

  // Anchor scalars: persist across prefetch() calls
  int anchor_range_idx;
  int anchor_inner_idx;
  int k_range_size;

  // prev_anchor: only needed for FWD IntraWGOverlap V-stagger (2 regs, DCE'd in BWD)
  int prev_anchor_range_idx;
  int prev_anchor_inner_idx;

  // On-the-fly token index computation from anchor (equal k_range_size only).
  // SparseLoad always uses block-sparse patterns with uniform k_range_size.
  CUTLASS_DEVICE int compute_token_index_from(int range_idx, int inner_idx, int offset) const {
    int r = range_idx, i = inner_idx;
    if constexpr (!InnerDirMaxToMin_) {
      int total = i + offset;
      r += total / k_range_size;
      i = total % k_range_size;
      if (r >= end_batches) { r = end_batches - 1; i = k_ranges[r].y - k_ranges[r].x - 1; }
    } else {
      int total = i - offset;
      if (total >= 0) {
        i = total;
      } else {
        int borrow = (-total - 1) / k_range_size + 1;
        r -= borrow;
        i = total + borrow * k_range_size;
      }
      if (r < bidb) { r = bidb; i = 0; }
    }
    return k_ranges[r].x + i;
  }

  CUTLASS_DEVICE int get_token_index(int offset) const {
    return compute_token_index_from(anchor_range_idx, anchor_inner_idx, offset);
  }

  CUTLASS_DEVICE int get_prev_token_index(int offset) const {
    return compute_token_index_from(prev_anchor_range_idx, prev_anchor_inner_idx, offset);
  }

  template <typename ParamsT, typename SharedStorage>
  CUTLASS_DEVICE SparseLoadBlockMeta(
      ParamsT const& params,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
      SharedStorage& shared_storage,
      int thread_idx = 0)
      : outer_block(get<0>(block_coord)),
        bidh(get<1>(block_coord)),
        bidh_kv(!PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh),
        q_ranges(params.q_ranges),
        k_ranges(params.k_ranges),
        attn_type_map(params.attn_type_map) {
    bidb = [&]() {
      if constexpr (RangeMerge) {
        return params.cu_batches[get<2>(block_coord)];
      } else {
        return get<2>(block_coord);
      }
    }();
    end_batches = [&]() {
      if constexpr (RangeMerge) {
        return params.cu_batches[get<2>(block_coord) + 1];
      } else {
        return bidb + 1;
      }
    }();

    // SparseLoad always uses equal k_range_size (block-sparse patterns)
    int total_k_tokens = (end_batches - bidb) * (k_ranges[bidb].y - k_ranges[bidb].x);
    inner_block_max = (total_k_tokens + kBlockN_ - 1) / kBlockN_;
    num_invalid_token = inner_block_max * kBlockN_ - total_k_tokens;
    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);

    if constexpr (IsProducer) {
      k_range_size = k_ranges[bidb].y - k_ranges[bidb].x;

      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GroupSize_;

      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          anchor_range_idx = end_batches - 1;
          anchor_inner_idx = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x - 1;
          int num_steps = kBlockN_ - (group_idx + 1) * NumRowsPerGroup_;
          advance_anchor(num_steps);
        } else {
          anchor_range_idx = bidb;
          anchor_inner_idx = 0;
          int num_steps = group_idx * NumRowsPerGroup_;
          advance_anchor(num_steps);
        }
      }
      prev_anchor_range_idx = anchor_range_idx;
      prev_anchor_inner_idx = anchor_inner_idx;
    } else {
      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
      }
    }
  }

  // Advance anchor by num_steps (equal k_range_size only, O(1) arithmetic).
  CUTLASS_DEVICE
  void advance_anchor(int num_steps) {
    static_assert(IsProducer, "advance_anchor() is producer-only");

    int n_k_ranges = num_steps / k_range_size;
    int n_k_range_inner = num_steps % k_range_size;

    if constexpr (InnerDirMaxToMin_) {
      if (anchor_inner_idx >= n_k_range_inner) {
        anchor_range_idx -= n_k_ranges;
        anchor_inner_idx -= n_k_range_inner;
      } else {
        anchor_range_idx -= (n_k_ranges + 1);
        anchor_inner_idx += k_range_size - n_k_range_inner;
      }
    } else {
      int remaining = k_range_size - 1 - anchor_inner_idx;
      if (remaining >= n_k_range_inner) {
        anchor_range_idx += n_k_ranges;
        anchor_inner_idx += n_k_range_inner;
      } else {
        anchor_range_idx += (n_k_ranges + 1);
        anchor_inner_idx = n_k_range_inner - remaining - 1;
      }
    }

    // Clamp anchor to valid bounds
    if constexpr (InnerDirMaxToMin_) {
      if (anchor_range_idx < bidb) {
        anchor_range_idx = bidb;
        anchor_inner_idx = 0;
      }
    } else {
      if (anchor_range_idx >= end_batches) {
        anchor_range_idx = end_batches - 1;
        int2 last = k_ranges[end_batches - 1];
        anchor_inner_idx = last.y - last.x - 1;
      }
    }
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_block, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    flash::advance_block_cur<kDir>(inner_block_cur);
    if constexpr (IsProducer) {
      prev_anchor_range_idx = anchor_range_idx;
      prev_anchor_inner_idx = anchor_inner_idx;
      if (!is_finish()) {
        advance_anchor(kBlockN_);
      }
    }
  }

  CUTLASS_DEVICE
  bool is_finish() {
    if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
      return inner_block_cur < inner_block_min;
    } else {
      return inner_block_cur >= inner_block_max;
    }
  }

  CUTLASS_DEVICE
  int padding_block() const {
    return inner_block_max - 1;
  }

  template <flash::DispatchDirection>
  CUTLASS_DEVICE void update_block_cur() {}

  CUTLASS_DEVICE
  bool is_valid() {
    return !is_finish();
  }

  CUTLASS_DEVICE
  bool skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
    return is_finish();
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// IndexAttnBlockMeta: Sparse block metadata for index-based attention.
// Producer-only arrays (token_indices, prev_token_indices, group_token_ptr)
// are zero-length when !IsProducer to save registers.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool IsProducer,
    bool RangeMerge,
    bool PackGQA,
    int QheadPerKhead,
    int NumRowsPerGroup_,
    int NumProducerThreads_,
    int GroupSize_,
    int kBlockN_,
    bool InnerDirMaxToMin_>
struct IndexAttnBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  // IndexAttn always iterates multiple blocks; batch loop is always needed.
  static constexpr bool NeedsBatchLoop = true;

  int const outer_block;
  int const bidh;
  int const bidh_kv;
  int bidb;

  flash::SeqlenInfo seqlen_info;

  flash::AttnType attn_type = flash::AttnType::Full;
  int end_batches;

  int token_indices[IsProducer ? NumRowsPerGroup_ : 0];
  int prev_token_indices[IsProducer ? NumRowsPerGroup_ : 0];

  CUTLASS_DEVICE int get_token_index(int offset) const {
    return token_indices[offset];
  }

  CUTLASS_DEVICE int get_prev_token_index(int offset) const {
    return prev_token_indices[offset];
  }

  int inner_block_cur;
  int inner_block_max;
  int num_invalid_token;
  static constexpr int inner_block_min = 0;

  int const* group_token_ptr;

  template <typename ParamsT, typename SharedStorage>
  CUTLASS_DEVICE IndexAttnBlockMeta(ParamsT const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage, int thread_idx = 0)
      : outer_block(get<0>(block_coord)), bidh(get<1>(block_coord)), bidh_kv(!PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh), group_token_ptr(nullptr) {
    bidb = [&]() {
      if constexpr (RangeMerge) {
        return params.cu_batches[get<2>(block_coord)];
      } else {
        return get<2>(block_coord);
      }
    }();

    seqlen_info.offset_q = bidb;
    seqlen_info.seqlen_q = 1;

    int unique_idx = get<2>(block_coord);
    int max_topk = params.index_attn_max_topk;
    int const* row_ptr = params.index_attn_indices + static_cast<int64_t>(unique_idx) * max_topk;

    int actual_topk = max_topk;
    for (int i = max_topk - 1; i >= 0 && row_ptr[i] < 0; --i)
      --actual_topk;

    seqlen_info.seqlen_k = actual_topk;
    inner_block_max = (actual_topk + kBlockN_ - 1) / kBlockN_;
    num_invalid_token = inner_block_max * kBlockN_ - actual_topk;
    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);
    end_batches = bidb + 1;

    if constexpr (IsProducer) {
      int aligned_total = inner_block_max * kBlockN_;
      int group_idx = (thread_idx % NumProducerThreads_) / GroupSize_;
      int group_offset;
      if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
        group_offset = (aligned_total - kBlockN_) + group_idx * NumRowsPerGroup_;
      } else {
        group_offset = group_idx * NumRowsPerGroup_;
      }
      group_token_ptr = row_ptr + group_offset;

      CUTE_UNROLL
      for (int i = 0; i < NumRowsPerGroup_; ++i) {
        prev_token_indices[i] = -1;
      }

      if (!is_finish()) {
        CUTE_UNROLL
        for (int i = 0; i < NumRowsPerGroup_; ++i) {
          int id = group_token_ptr[i];
          token_indices[i] = (id >= 0) ? id : 0;
        }
      }
    }
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_block, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    flash::advance_block_cur<kDir>(inner_block_cur);
    if constexpr (IsProducer) {
      CUTE_UNROLL
      for (int i = 0; i < NumRowsPerGroup_; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      if (!is_finish()) {
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          group_token_ptr -= kBlockN_;
        } else {
          group_token_ptr += kBlockN_;
        }
        CUTE_UNROLL
        for (int i = 0; i < NumRowsPerGroup_; ++i) {
          int id = group_token_ptr[i];
          token_indices[i] = (id >= 0) ? id : 0;
        }
      }
    }
  }

  CUTLASS_DEVICE
  bool is_finish() {
    if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
      return inner_block_cur < inner_block_min;
    } else {
      return inner_block_cur >= inner_block_max;
    }
  }

  CUTLASS_DEVICE
  int padding_block() const {
    return inner_block_max - 1;
  }

  template <flash::DispatchDirection>
  CUTLASS_DEVICE void update_block_cur() {}

  CUTLASS_DEVICE bool is_valid() {
    return true;
  }

  CUTLASS_DEVICE
  bool skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
    return is_finish();
  }
};

} // namespace flash
