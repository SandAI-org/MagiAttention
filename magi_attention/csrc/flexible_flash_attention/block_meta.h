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

  CUTLASS_DEVICE
  void skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
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
    typename SeqlenInfo_t,
    int NumRowsPerGroup_,
    int NumGroups_,
    int GroupSize_,
    int NumProducerThreads_,
    int kBlockN_>
struct SparseLoadBlockMeta {
  // SparseLoad always iterates multiple blocks; batch loop is always needed.
  static constexpr bool NeedsBatchLoop = true;

  int const outer_block; // always m_block for SparseLoad (FWD only)
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;
  SeqlenInfo_t seqlen_info;
  flash::AttnType attn_type;

  int num_invalid_token;
  int cur_loop;
  int inner_block_max; // total number of sparse load iterations (was loop_count)

  static constexpr int inner_block_min = 0;

  int2 const* const q_ranges;
  int2 const* const k_ranges;
  int const* const attn_type_map;

  // Producer-only arrays (zero-length when !IsProducer)
  int cur_k_range_indices[IsProducer ? NumRowsPerGroup_ : 0];
  int cur_k_range_inner_indices[IsProducer ? NumRowsPerGroup_ : 0];
  int token_indices[IsProducer ? NumRowsPerGroup_ : 0];
  int prev_token_indices[IsProducer ? NumRowsPerGroup_ : 0];
  bool is_equal_k_range_size;
  int k_range_size;

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
        attn_type_map(params.attn_type_map),
        is_equal_k_range_size(params.equal_k_range_size ? *params.equal_k_range_size == 1 : false) {
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
    cur_loop = 0;
    inner_block_max = params.sparse_load_loop_count ? params.sparse_load_loop_count[get<2>(block_coord)] : 0;
    num_invalid_token = params.sparse_load_invalid_count ? params.sparse_load_invalid_count[get<2>(block_coord)] : 0;

    if constexpr (IsProducer) {
      constexpr int last_idx = NumRowsPerGroup_ - 1;
      cur_k_range_indices[last_idx] = end_batches - 1;
      cur_k_range_inner_indices[last_idx] = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x - 1;
      prev_token_indices[last_idx] = -1;

      if (is_equal_k_range_size) {
        k_range_size = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x;
      }

      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GroupSize_;

      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        int num_steps = (NumGroups_ - group_idx - 1) * NumRowsPerGroup_;
        advance_producer(num_steps);

        // Move valid token indices to front when there are invalid tokens
        int offset = num_invalid_token % NumRowsPerGroup_;
        for (int i = 0; i < offset; ++i) {
          token_indices[offset - 1 - i] = token_indices[last_idx - i];
        }
      }
    } else {
      // Consumer path
      if (!is_finish()) {
        seqlen_info = SeqlenInfo_t{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
      }
    }
  }

  // Retreat the k-range cursor by num_steps tokens, deducting invalid tokens
  // first, then update cur_k_range_indices/cur_k_range_inner_indices and fill
  // token_indices from last_idx backwards.
  CUTLASS_DEVICE
  void advance_producer(int num_steps) {
    static_assert(IsProducer, "advance_producer() is producer-only");
    constexpr int last_idx = NumRowsPerGroup_ - 1;

    if (num_invalid_token) {
      if (num_steps >= num_invalid_token) {
        num_steps -= num_invalid_token;
        num_invalid_token = 0;
      } else {
        num_invalid_token -= num_steps;
        num_steps = 0;
      }
    }

    if (is_equal_k_range_size) {
      int n_k_ranges = num_steps / k_range_size;
      int n_k_range_inner = num_steps % k_range_size;

      if (cur_k_range_inner_indices[last_idx] >= n_k_range_inner) {
        cur_k_range_indices[last_idx] -= n_k_ranges;
        cur_k_range_inner_indices[last_idx] -= n_k_range_inner;
      } else {
        cur_k_range_indices[last_idx] -= (n_k_ranges + 1);
        cur_k_range_inner_indices[last_idx] = cur_k_range_inner_indices[last_idx] + k_range_size - n_k_range_inner;
      }
    } else {
      int cnt = 0;
      while (cnt < num_steps) {
        int rest = num_steps - cnt;
        if (cur_k_range_inner_indices[last_idx] + 1 > rest) {
          cur_k_range_inner_indices[last_idx] -= rest;
          break;
        } else {
          cur_k_range_indices[last_idx] -= 1;
          cnt += (cur_k_range_inner_indices[last_idx] + 1);
          int2 prev_k_range = k_ranges[cur_k_range_indices[last_idx]];
          cur_k_range_inner_indices[last_idx] = prev_k_range.y - prev_k_range.x - 1;
        }
      }
    }

    token_indices[last_idx] = k_ranges[cur_k_range_indices[last_idx]].x + cur_k_range_inner_indices[last_idx];

    CUTE_UNROLL
    for (int i = last_idx - 1; i >= 0; --i) {
      if (cur_k_range_inner_indices[i + 1] > 0) {
        cur_k_range_indices[i] = cur_k_range_indices[i + 1];
        cur_k_range_inner_indices[i] = cur_k_range_inner_indices[i + 1] - 1;
      } else {
        cur_k_range_indices[i] = cur_k_range_indices[i + 1] - 1;
        int2 prev_k_range = k_ranges[cur_k_range_indices[i]];
        cur_k_range_inner_indices[i] = prev_k_range.y - prev_k_range.x - 1;
      }
      token_indices[i] = k_ranges[cur_k_range_indices[i]].x + cur_k_range_inner_indices[i];
    }
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_block, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    ++cur_loop;
    if constexpr (IsProducer) {
      for (int i = 0; i < NumRowsPerGroup_; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      if (!is_finish()) {
        advance_producer(kBlockN_);
      }
    }
  }

  CUTLASS_DEVICE
  bool is_finish() {
    return cur_loop >= inner_block_max;
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return !is_finish();
  }

  CUTLASS_DEVICE
  void skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// IndexAttnBlockMeta: Sparse block metadata for index-based attention.
// Producer-only arrays (token_indices, prev_token_indices, group_token_ptr)
// are zero-length when !IsProducer to save registers.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IsProducer, bool RangeMerge, bool PackGQA, int QheadPerKhead, int NumRowsPerGroup_, int NumProducerThreads_, int GroupSize_, int kBlockN_>
struct IndexAttnBlockMeta {
  // IndexAttn always iterates multiple blocks; batch loop is always needed.
  static constexpr bool NeedsBatchLoop = true;

  int const outer_block;
  int const bidh;
  int const bidh_kv;
  int bidb;

  flash::SeqlenInfo seqlen_info;

  flash::AttnType attn_type = flash::AttnType::Full;
  int inner_block_min = 0;
  int end_batches;

  int token_indices[IsProducer ? NumRowsPerGroup_ : 0];
  int prev_token_indices[IsProducer ? NumRowsPerGroup_ : 0];

  int cur_loop;
  int inner_block_max;
  int num_invalid_token;

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
    cur_loop = 0;
    inner_block_max = (actual_topk + kBlockN_ - 1) / kBlockN_;
    num_invalid_token = inner_block_max * kBlockN_ - actual_topk;
    end_batches = bidb + 1;

    if constexpr (IsProducer) {
      int aligned_total = inner_block_max * kBlockN_;
      int group_idx = (thread_idx % NumProducerThreads_) / GroupSize_;
      int group_offset = (aligned_total - kBlockN_) + group_idx * NumRowsPerGroup_;
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
    ++cur_loop;
    if constexpr (IsProducer) {
      CUTE_UNROLL
      for (int i = 0; i < NumRowsPerGroup_; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      if (!is_finish()) {
        group_token_ptr -= kBlockN_;
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
    return cur_loop >= inner_block_max;
  }

  CUTLASS_DEVICE bool is_valid() {
    return true;
  }

  CUTLASS_DEVICE
  void skip_to_first_valid() {
    while (!is_valid() && !is_finish()) {
      prefetch();
    }
  }
};

} // namespace flash
