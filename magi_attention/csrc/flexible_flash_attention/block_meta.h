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

#include <type_traits>

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
//
// IsLoopQ_=false (LoopK): outer=Q (TMA), inner=KV (scatter), token_indices = KV positions
// IsLoopQ_=true  (LoopQ): outer=KV (TMA), inner=Q (scatter),  token_indices = Q positions
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
    bool InnerDirMaxToMin_,
    bool IsLoopQ_ = false,
    int OuterBlockSize_ = 0>
struct SparseLoadBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr bool NeedsBatchLoop = true;
  static constexpr bool IsLoopQ = IsLoopQ_;
  static constexpr int InnerBlockSize = kBlockN_;

  int const outer_block; // m_block for LoopK, n_block for LoopQ
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;
  flash::SeqlenInfo seqlen_info;
  flash::AttnType attn_type;

  int num_invalid_token;
  // LoopQ: invalid K columns when kBlockN > seqlen_k for this bidb's K range
  int num_invalid_k_token = 0;
  int inner_block_cur;
  int inner_block_max;

  static constexpr int inner_block_min = 0;

  int2 const* const q_ranges;
  int2 const* const k_ranges;
  int const* const attn_type_map;

  // Producer-only traversal state: ONLY the anchor cursor (one (range, inner) pair
  // per thread) lives in registers. Per-row token indices are written straight into
  // the smem stage slot by fill_token_indices() — no per-row register array at all.
  // (Register arrays with dynamic indexing made nvcc spill to local memory.)
  int cur_range_idx;
  int cur_range_inner_idx;
  bool is_equal_range_size;
  int range_size;

  // The ranges for the scatter dimension: k_ranges for LoopK, q_ranges for LoopQ
  CUTLASS_DEVICE
  int2 const* scatter_ranges() const {
    if constexpr (IsLoopQ) {
      return q_ranges;
    } else {
      return k_ranges;
    }
  }

  // LoopQ + PackGQA: Q heads are folded into the row dimension, so the scatter
  // walk operates in PACKED-ROW space — every q_range endpoint is scaled by
  // QheadPerKhead. token_indices then hold packed rows p = token * G + g, where
  // g is the q-head index within the kv group (packed coordinate (g, token) with
  // g fastest, matching shape ((qhead_per_khead, seqlen), ...)). LoopK scatters
  // KV rows, which are never head-packed, so the scale is 1 there.
  static constexpr int kScatterScale = (IsLoopQ_ && PackGQA) ? QheadPerKhead : 1;

  // Read scatter range i with endpoints scaled to packed-row space.
  CUTLASS_DEVICE
  int2 scatter_range(int i) const {
    int2 r = scatter_ranges()[i];
    if constexpr (kScatterScale != 1) {
      r.x *= kScatterScale;
      r.y *= kScatterScale;
    }
    return r;
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
        attn_type_map(params.attn_type_map),
        is_equal_range_size([&]() {
          if constexpr (IsLoopQ) {
            return params.equal_q_range_size;
          } else {
            return params.equal_k_range_size;
          }
        }()) {
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

    // LoopQ validity: outer_block (n_block) must be within this bidb's K range.
    // Without this check, tiles with n_block outside the K range would compute
    // spurious attention between Q tokens and an unrelated K block.
    if constexpr (IsLoopQ_ && OuterBlockSize_ > 0) {
      int seqlen_k = k_ranges[bidb].y - k_ranges[bidb].x;
      if (outer_block * OuterBlockSize_ >= seqlen_k) {
        inner_block_max = 0;
        num_invalid_token = 0;
        inner_block_cur = 0;
        if constexpr (IsProducer) {
          seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        }
        return;
      }
      // K-dimension padding: when OuterBlockSize > seqlen_k remainder,
      // apply_padding_mask handles the excess columns.
      int k_block_end = (outer_block + 1) * OuterBlockSize_;
      num_invalid_k_token = k_block_end > seqlen_k ? k_block_end - seqlen_k : 0;
    }

    int total_tokens;
    if (is_equal_range_size) {
      int2 const r0 = scatter_range(bidb);
      total_tokens = (end_batches - bidb) * (r0.y - r0.x);
    } else {
      total_tokens = 0;
      for (int i = bidb; i < end_batches; ++i) {
        int2 const r = scatter_range(i);
        total_tokens += r.y - r.x;
      }
    }
    inner_block_max = (total_tokens + InnerBlockSize - 1) / InnerBlockSize;
    num_invalid_token = inner_block_max * InnerBlockSize - total_tokens;
    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);

    if constexpr (IsProducer) {
      if (is_equal_range_size) {
        int2 const r0 = scatter_range(bidb);
        range_size = r0.y - r0.x;
      }

      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GroupSize_;

      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        // Position the anchor at this group's first row of the first tile.
        // MaxToMin: anchor = HIGH end of the group's rows; MinToMax: LOW end.
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          int2 const r_last = scatter_range(end_batches - 1);
          cur_range_idx = end_batches - 1;
          cur_range_inner_idx = r_last.y - r_last.x - 1;
          advance_token_idx(cur_range_idx, cur_range_inner_idx, kBlockN_ - (group_idx + 1) * NumRowsPerGroup_);
        } else {
          cur_range_idx = bidb;
          cur_range_inner_idx = 0;
          advance_token_idx(cur_range_idx, cur_range_inner_idx, group_idx * NumRowsPerGroup_);
        }
      }
    } else {
      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);
      }
    }
  }

  // Advance a (range, inner) cursor by num_steps tokens in the traversal direction
  // (backward for MaxToMin, forward for MinToMax), then clamp to the nearest valid
  // boundary on overflow. Clamped positions load a duplicated valid token;
  // apply_padding_mask sets their attention scores to -inf so they contribute zero.
  // Equal-range sizes take an O(1) div/mod fast path; unequal sizes walk range by range.
  CUTLASS_DEVICE
  void advance_token_idx(int& range_idx, int& inner_idx, int num_steps) const {
    if (is_equal_range_size) {
      int n_ranges = num_steps / range_size;
      int n_range_inner = num_steps % range_size;

      if constexpr (InnerDirMaxToMin_) {
        if (inner_idx >= n_range_inner) {
          range_idx -= n_ranges;
          inner_idx -= n_range_inner;
        } else {
          range_idx -= (n_ranges + 1);
          inner_idx += range_size - n_range_inner;
        }
      } else {
        int remaining = range_size - 1 - inner_idx;
        if (remaining >= n_range_inner) {
          range_idx += n_ranges;
          inner_idx += n_range_inner;
        } else {
          range_idx += (n_ranges + 1);
          inner_idx = n_range_inner - remaining - 1;
        }
      }
    } else {
      // Unequal-range slow path: step one range at a time
      int cnt = 0;
      if constexpr (InnerDirMaxToMin_) {
        while (cnt < num_steps && range_idx >= bidb) {
          int rest = num_steps - cnt;
          if (inner_idx + 1 > rest) {
            inner_idx -= rest;
            break;
          }
          cnt += (inner_idx + 1);
          range_idx -= 1;
          if (range_idx < bidb)
            break;
          int2 r = scatter_range(range_idx);
          inner_idx = r.y - r.x - 1;
        }
      } else {
        while (cnt < num_steps && range_idx < end_batches) {
          int rest = num_steps - cnt;
          int2 r = scatter_range(range_idx);
          int remaining = r.y - r.x - 1 - inner_idx;
          if (remaining >= rest) {
            inner_idx += rest;
            break;
          }
          cnt += (remaining + 1);
          range_idx += 1;
          inner_idx = 0;
        }
      }
    }

    // Clamp to valid range [bidb, end_batches) if the cursor overflowed
    if constexpr (InnerDirMaxToMin_) {
      if (range_idx < bidb) {
        range_idx = bidb;
        inner_idx = 0;
      }
    } else {
      if (range_idx >= end_batches) {
        range_idx = end_batches - 1;
        int2 last = scatter_range(end_batches - 1);
        inner_idx = last.y - last.x - 1;
      }
    }
  }

  // Write this group's NumRowsPerGroup_ token indices for the CURRENT tile into the
  // smem stage slot (rows [group_idx*NumRowsPerGroup_, +NumRowsPerGroup_)).
  // Called after producer_acquire (the held stage makes the slot writable). Lane j of
  // the group computes row j (strided by GroupSize_) from a cursor copy of the anchor —
  // O(1) per row on the equal-range fast path. Caller must __syncwarp() before reading
  // the slot back (writer lanes ≠ reader lanes, but always within the same warp).
  CUTLASS_DEVICE
  void fill_token_indices(int* slot_rows, int idx_in_group, int group_idx) const {
    static_assert(IsProducer, "fill_token_indices() is producer-only");
    int* const group_rows = slot_rows + group_idx * NumRowsPerGroup_;
    for (int j = idx_in_group; j < NumRowsPerGroup_; j += GroupSize_) {
      int range_idx = cur_range_idx;
      int inner_idx = cur_range_inner_idx;
      advance_token_idx(range_idx, inner_idx, j);
      // MaxToMin walks backward from the high-end anchor: j steps back = row (last - j)
      int const dst = InnerDirMaxToMin_ ? (NumRowsPerGroup_ - 1 - j) : j;
      group_rows[dst] = scatter_range(range_idx).x + inner_idx;
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
      if (!is_finish()) {
        advance_token_idx(cur_range_idx, cur_range_inner_idx, kBlockN_);
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
// Helper: extract NHK from mainloop params.
// FWD has shape_K = (seqlen, headdim, nhk); BWD has shape_KVdKdV with same layout.
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace detail {
template <typename P, typename = void>
struct nhk_of {
  CUTLASS_DEVICE static int get(P const& p) {
    return cute::get<2>(p.shape_KVdKdV);
  }
};
template <typename P>
struct nhk_of<P, std::void_t<decltype(std::declval<P>().shape_K)>> {
  CUTLASS_DEVICE static int get(P const& p) {
    return cute::get<2>(p.shape_K);
  }
};
} // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
// IndexAttnBlockMeta: Sparse block metadata for index-based attention.
// Producer-only state (group_token_ptr) is unused when !IsProducer; token indices are
// written straight into the smem stage slot by fill_token_indices() (no register array).
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

  int inner_block_cur;
  int inner_block_max;
  int num_invalid_token;
  static constexpr int inner_block_min = 0;

  int const* group_token_ptr;
  // NHK and kv_head for logical→physical index conversion
  int nhk;
  int kv_head_local;

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
    // indices store logical token positions; recover physical row in-kernel
    nhk = detail::nhk_of<ParamsT>::get(params);
    kv_head_local = unique_idx % nhk;

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
    }
  }

  // Write this group's NumRowsPerGroup_ token indices for the CURRENT tile into the
  // smem stage slot. Same contract as SparseLoadBlockMeta::fill_token_indices():
  // called after producer_acquire; caller must __syncwarp() before reading back.
  CUTLASS_DEVICE
  void fill_token_indices(int* slot_rows, int idx_in_group, int group_idx) const {
    static_assert(IsProducer, "fill_token_indices() is producer-only");
    int* const group_rows = slot_rows + group_idx * NumRowsPerGroup_;
    for (int j = idx_in_group; j < NumRowsPerGroup_; j += GroupSize_) {
      int const id = group_token_ptr[j];
      // logical position → physical row: pos * NHK + kv_head
      group_rows[j] = (id >= 0) ? id * nhk + kv_head_local : 0;
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
      if (!is_finish()) {
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          group_token_ptr -= kBlockN_;
        } else {
          group_token_ptr += kBlockN_;
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
