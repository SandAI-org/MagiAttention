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
  // SparseLoad always iterates multiple blocks; batch loop is always needed.
  static constexpr bool NeedsBatchLoop = true;

  int const outer_block; // always m_block for SparseLoad (FWD only)
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;
  flash::SeqlenInfo seqlen_info;
  flash::AttnType attn_type;

  int num_invalid_token;
  int inner_block_cur;
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
        is_equal_k_range_size(params.equal_k_range_size) {
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
    // Compute inner_block_max and num_invalid_token in-kernel
    // (replaces Python-side compute_sparse_load_metadata precomputation).
    // is_equal_k_range_size is passed from host (default true for the common case),
    // so the equal path collapses the per-batch summation into a single multiply.
    int total_k_tokens;
    if (is_equal_k_range_size) {
      total_k_tokens = (end_batches - bidb) * (k_ranges[bidb].y - k_ranges[bidb].x);
    } else {
      total_k_tokens = 0;
      for (int i = bidb; i < end_batches; ++i) {
        total_k_tokens += k_ranges[i].y - k_ranges[i].x;
      }
    }
    inner_block_max = (total_k_tokens + kBlockN_ - 1) / kBlockN_;
    num_invalid_token = inner_block_max * kBlockN_ - total_k_tokens;
    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);

    if constexpr (IsProducer) {
      constexpr int last_idx = NumRowsPerGroup_ - 1;
      prev_token_indices[last_idx] = -1;

      if (is_equal_k_range_size) {
        k_range_size = k_ranges[bidb].y - k_ranges[bidb].x;
      }

      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GroupSize_;

      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          // Start from the last token of the last batch (max end)
          cur_k_range_indices[last_idx] = end_batches - 1;
          cur_k_range_inner_indices[last_idx] = k_ranges[end_batches - 1].y - k_ranges[end_batches - 1].x - 1;

          int num_steps = kBlockN_ - (group_idx + 1) * NumRowsPerGroup_;
          advance_and_fill(num_steps);
        } else {
          // Start from the first token of the first batch (min end)
          cur_k_range_indices[0] = bidb;
          cur_k_range_inner_indices[0] = 0;

          int num_steps = group_idx * NumRowsPerGroup_;
          advance_and_fill(num_steps);
        }
      }
    } else {
      // Consumer path
      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
      }
    }
  }

  // Clamp index to the nearest valid boundary if it overflowed.
  // MaxToMin can underflow below bidb; MinToMax can overflow past end_batches.
  // Clamped positions load a duplicated valid token; apply_padding_mask sets
  // their attention scores to -inf so they contribute zero after softmax.
  CUTLASS_DEVICE
  void clamp_to_boundary(int idx) {
    if constexpr (InnerDirMaxToMin_) {
      if (cur_k_range_indices[idx] < bidb) {
        cur_k_range_indices[idx] = bidb;
        cur_k_range_inner_indices[idx] = 0;
      }
    } else {
      if (cur_k_range_indices[idx] >= end_batches) {
        cur_k_range_indices[idx] = end_batches - 1;
        int2 last = k_ranges[end_batches - 1];
        cur_k_range_inner_indices[idx] = last.y - last.x - 1;
      }
    }
  }

  // Step one token in the traversal direction: backward (MaxToMin) or forward (MinToMax).
  // Clamps dst to boundary if it overflows (see clamp_to_boundary).
  CUTLASS_DEVICE
  void step_one_token(int dst, int src) {
    if constexpr (!InnerDirMaxToMin_) {
      int2 r = k_ranges[cur_k_range_indices[src]];
      if (cur_k_range_inner_indices[src] + 1 < r.y - r.x) {
        cur_k_range_indices[dst] = cur_k_range_indices[src];
        cur_k_range_inner_indices[dst] = cur_k_range_inner_indices[src] + 1;
      } else {
        cur_k_range_indices[dst] = cur_k_range_indices[src] + 1;
        cur_k_range_inner_indices[dst] = 0;
      }
    } else {
      if (cur_k_range_inner_indices[src] > 0) {
        cur_k_range_indices[dst] = cur_k_range_indices[src];
        cur_k_range_inner_indices[dst] = cur_k_range_inner_indices[src] - 1;
      } else {
        cur_k_range_indices[dst] = cur_k_range_indices[src] - 1;
        cur_k_range_inner_indices[dst] = 0;
        // Read range size only if index is still valid (clamp handles OOB below)
        if (cur_k_range_indices[dst] >= bidb) {
          int2 r = k_ranges[cur_k_range_indices[dst]];
          cur_k_range_inner_indices[dst] = r.y - r.x - 1;
        }
      }
    }
    clamp_to_boundary(dst);
  }

  // Advance the anchor cursor by num_steps tokens (borrow/carry arithmetic),
  // then fill all NumRowsPerGroup_ token_indices from the anchor outward via step_one_token.
  CUTLASS_DEVICE
  void advance_and_fill(int num_steps) {
    static_assert(IsProducer, "advance_and_fill() is producer-only");

    // Anchor index: MaxToMin starts from the high end, MinToMax from the low end
    constexpr int anchor = InnerDirMaxToMin_ ? NumRowsPerGroup_ - 1 : 0;

    // Advance anchor cursor by num_steps (equal-range O(1) fast path)
    if (is_equal_k_range_size) {
      int n_k_ranges = num_steps / k_range_size;
      int n_k_range_inner = num_steps % k_range_size;

      if constexpr (InnerDirMaxToMin_) {
        if (cur_k_range_inner_indices[anchor] >= n_k_range_inner) {
          cur_k_range_indices[anchor] -= n_k_ranges;
          cur_k_range_inner_indices[anchor] -= n_k_range_inner;
        } else {
          cur_k_range_indices[anchor] -= (n_k_ranges + 1);
          cur_k_range_inner_indices[anchor] += k_range_size - n_k_range_inner;
        }
      } else {
        int remaining = k_range_size - 1 - cur_k_range_inner_indices[anchor];
        if (remaining >= n_k_range_inner) {
          cur_k_range_indices[anchor] += n_k_ranges;
          cur_k_range_inner_indices[anchor] += n_k_range_inner;
        } else {
          cur_k_range_indices[anchor] += (n_k_ranges + 1);
          cur_k_range_inner_indices[anchor] = n_k_range_inner - remaining - 1;
        }
      }
    } else {
      // Unequal-range slow path: step one range at a time
      int cnt = 0;
      if constexpr (InnerDirMaxToMin_) {
        while (cnt < num_steps && cur_k_range_indices[anchor] >= bidb) {
          int rest = num_steps - cnt;
          if (cur_k_range_inner_indices[anchor] + 1 > rest) {
            cur_k_range_inner_indices[anchor] -= rest;
            break;
          }
          cnt += (cur_k_range_inner_indices[anchor] + 1);
          cur_k_range_indices[anchor] -= 1;
          if (cur_k_range_indices[anchor] < bidb)
            break;
          int2 r = k_ranges[cur_k_range_indices[anchor]];
          cur_k_range_inner_indices[anchor] = r.y - r.x - 1;
        }
      } else {
        while (cnt < num_steps && cur_k_range_indices[anchor] < end_batches) {
          int rest = num_steps - cnt;
          int2 r = k_ranges[cur_k_range_indices[anchor]];
          int remaining = r.y - r.x - 1 - cur_k_range_inner_indices[anchor];
          if (remaining >= rest) {
            cur_k_range_inner_indices[anchor] += rest;
            break;
          }
          cnt += (remaining + 1);
          cur_k_range_indices[anchor] += 1;
          cur_k_range_inner_indices[anchor] = 0;
        }
      }
    }

    // Clamp anchor to valid range [bidb, end_batches) if it overflowed
    clamp_to_boundary(anchor);

    token_indices[anchor] = k_ranges[cur_k_range_indices[anchor]].x + cur_k_range_inner_indices[anchor];

    // Fill remaining positions: each is one token away from the previous
    CUTE_UNROLL
    for (int j = 1; j < NumRowsPerGroup_; ++j) {
      int dst = InnerDirMaxToMin_ ? (NumRowsPerGroup_ - 1 - j) : j;
      int src = InnerDirMaxToMin_ ? (NumRowsPerGroup_ - j) : (j - 1);
      step_one_token(dst, src);
      token_indices[dst] = k_ranges[cur_k_range_indices[dst]].x + cur_k_range_inner_indices[dst];
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
      for (int i = 0; i < NumRowsPerGroup_; ++i) {
        prev_token_indices[i] = token_indices[i];
      }
      if (!is_finish()) {
        advance_and_fill(kBlockN_);
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
