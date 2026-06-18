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
    bool IsLoopQ_>
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
  // SparseLoad contract: all scatter-dim ranges have one uniform size (block-mask
  // generated ranges are uniform by construction; asserted at the Python entry).
  // This is what makes the O(1) div/mod cursor arithmetic in advance_token_idx valid.
  int range_size;

  // Read scatter range i with endpoints scaled to packed-row space.
  // LoopQ + PackGQA: Q heads are folded into the row dimension, so the scatter
  // walk operates in PACKED-ROW space — every endpoint is scaled by QheadPerKhead.
  // token_indices then hold packed rows p = token * G + g, where g is the q-head
  // index within the kv group. LoopK scatters KV rows (never head-packed, scale 1).
  CUTLASS_DEVICE
  int2 packed_range(int i) const {
    int2 r = (IsLoopQ ? q_ranges : k_ranges)[i];
    if constexpr (IsLoopQ_ && PackGQA) {
      r.x *= QheadPerKhead;
      r.y *= QheadPerKhead;
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

    // NOTE: no outer_block-vs-K-range validity check is needed for LoopQ.
    // auto_range_merge dedups with unique_consecutive_pairs, so every sub-batch
    // in a merged group shares one exact K window, and the scheduler derives the
    // outer (n_block) tile count from that same window — outer_block is always
    // in range. The K residual-column mask is computed at the use site in the
    // mainloop from seqlen_info (symmetric with LoopK's Q residual mask).

    int2 const r0 = packed_range(bidb);
    range_size = r0.y - r0.x;
    int const total_tokens = (end_batches - bidb) * range_size;
    inner_block_max = (total_tokens + InnerBlockSize - 1) / InnerBlockSize;
    num_invalid_token = inner_block_max * InnerBlockSize - total_tokens;
    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);

    if constexpr (IsProducer) {
      int idx_in_warpgroup = thread_idx % 128;
      int group_idx = idx_in_warpgroup / GroupSize_;

      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        // Position the anchor at this group's first row of the first tile.
        // MaxToMin: anchor = HIGH end of the group's rows; MinToMax: LOW end.
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          int2 const r_last = packed_range(end_batches - 1);
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
  // The uniform range size (SparseLoad contract) makes this O(1) div/mod arithmetic.
  CUTLASS_DEVICE
  void advance_token_idx(int& range_idx, int& inner_idx, int num_steps) const {
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

    // Clamp to valid range [bidb, end_batches) if the cursor overflowed
    if constexpr (InnerDirMaxToMin_) {
      if (range_idx < bidb) {
        range_idx = bidb;
        inner_idx = 0;
      }
    } else {
      if (range_idx >= end_batches) {
        range_idx = end_batches - 1;
        int2 last = packed_range(end_batches - 1);
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
      group_rows[dst] = packed_range(range_idx).x + inner_idx;
    }
  }

  // Return the absolute packed row index for the first row of the current tile.
  // Used by InnerLoadMode::kTma (scatter) to compute TMA tile coordinates.
  CUTLASS_DEVICE
  int get_packed_first_row() const {
    static_assert(IsProducer, "get_packed_first_row() is producer-only");
    return packed_range(cur_range_idx).x + cur_range_inner_idx;
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
// IndexAttnBlockMeta: Unified sparse block metadata for index-based attention.
// Handles both LoopK and LoopQ via the IsLoopQ_ template parameter (like SparseLoadBlockMeta).
//
// IsLoopQ_=false (LoopK): outer=Q token (bidb), inner=K from forward topk indices
//   fill_token_indices fills K physical rows: id * nhk + kv_head_local
// IsLoopQ_=true  (LoopQ): outer=K block (bidb), inner=Q from inv_indices
//   fill_token_indices fills Q packed rows: q_token * QheadPerKhead + sub_head
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool IsProducer,
    bool RangeMerge,
    bool PackGQA,
    int QheadPerKhead,
    int NumRowsPerGroup_,
    int NumProducerThreads_,
    int GroupSize_,
    int kInnerBlockSize_,
    bool InnerDirMaxToMin_,
    int KBlockSize_ = 1,
    bool IsLoopQ_ = false>
struct IndexAttnBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr int kKBlockSize = KBlockSize_;
  static constexpr bool IsLoopQ = IsLoopQ_;
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
  int nhk;
  // LoopK: kv_head for K physical row conversion; LoopQ: unused (Q packed rows are head-agnostic)
  int head_local;

  template <typename ParamsT, typename SharedStorage>
  CUTLASS_DEVICE IndexAttnBlockMeta(ParamsT const& params, cute::tuple<int32_t, int32_t, int32_t> const& block_coord, SharedStorage& shared_storage, int thread_idx = 0)
      : outer_block(get<0>(block_coord)),
        bidh(get<1>(block_coord)),
        bidh_kv(IsLoopQ ? bidh : (!PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh)),
        group_token_ptr(nullptr) {
    bidb = [&]() {
      if constexpr (RangeMerge && !IsLoopQ) {
        return params.cu_batches[get<2>(block_coord)];
      } else {
        return get<2>(block_coord);
      }
    }();

    nhk = detail::nhk_of<ParamsT>::get(params);

    int max_topk = params.index_attn_max_topk;
    int const* row_ptr;
    int actual_topk;

    if constexpr (!IsLoopQ) {
      // ── LoopK: bidb = Q token, indices = forward topk (Q→K) ──
      seqlen_info.offset_q = bidb;
      seqlen_info.seqlen_q = 1;

      int unique_idx = get<2>(block_coord);
      head_local = unique_idx % nhk;
      row_ptr = params.index_attn_indices + static_cast<int64_t>(unique_idx) * max_topk;

      actual_topk = max_topk;
      for (int i = max_topk - 1; i >= 0 && row_ptr[i] < 0; --i)
        --actual_topk;

      int effective_k = actual_topk * kKBlockSize;
      seqlen_info.seqlen_k = effective_k;
      inner_block_max = (effective_k + kInnerBlockSize_ - 1) / kInnerBlockSize_;
      num_invalid_token = inner_block_max * kInnerBlockSize_ - effective_k;
    } else {
      // ── LoopQ: bidb = K block, indices = inv_indices (K→Q) ──
      seqlen_info.offset_k = bidb * kKBlockSize;
      seqlen_info.seqlen_k = kKBlockSize;
      head_local = bidh;

      // inv_indices layout: (num_k_blocks, nhk * inv_topk_per_head)
      int inv_topk_per_head = max_topk / nhk;
      row_ptr = params.index_attn_indices + static_cast<int64_t>(bidb) * max_topk + static_cast<int64_t>(bidh) * inv_topk_per_head;
      int max_inv_topk = inv_topk_per_head;

      actual_topk = max_inv_topk;
      for (int i = max_inv_topk - 1; i >= 0 && row_ptr[i] < 0; --i)
        --actual_topk;

      int total_q_packed_rows = actual_topk * QheadPerKhead;
      seqlen_info.offset_q = 0;
      seqlen_info.seqlen_q = total_q_packed_rows;
      inner_block_max = (total_q_packed_rows + kInnerBlockSize_ - 1) / kInnerBlockSize_;
      num_invalid_token = inner_block_max * kInnerBlockSize_ - total_q_packed_rows;
    }

    inner_block_cur = flash::init_block_cur<kDir>(inner_block_min, inner_block_max);
    end_batches = bidb + 1;

    if constexpr (IsProducer) {
      if constexpr (!IsLoopQ && kKBlockSize <= 1) {
        // Token-level LoopK: pointer walks kInnerBlockSize_ entries per tile
        int aligned_total = inner_block_max * kInnerBlockSize_;
        int group_idx = (thread_idx % NumProducerThreads_) / GroupSize_;
        int group_offset;
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          group_offset = (aligned_total - kInnerBlockSize_) + group_idx * NumRowsPerGroup_;
        } else {
          group_offset = group_idx * NumRowsPerGroup_;
        }
        group_token_ptr = row_ptr + group_offset;
      } else {
        // Block-level LoopK or any LoopQ: absolute indexing from row_ptr
        group_token_ptr = row_ptr;
      }
    }
  }

  // Fill token indices into the smem stage slot for the CURRENT tile.
  // LoopK: fills K physical rows (id * nhk + kv_head_local)
  // LoopQ: fills Q packed rows (q_token * QheadPerKhead + sub_head)
  CUTLASS_DEVICE
  void fill_token_indices(int* slot_rows, int idx_in_group, int group_idx) const {
    static_assert(IsProducer, "fill_token_indices() is producer-only");
    int* const group_rows = slot_rows + group_idx * NumRowsPerGroup_;

    if constexpr (!IsLoopQ) {
      // ── LoopK: fill K positions ──
      if constexpr (kKBlockSize <= 1) {
        for (int j = idx_in_group; j < NumRowsPerGroup_; j += GroupSize_) {
          int const id = group_token_ptr[j];
          group_rows[j] = (id >= 0) ? id * nhk + head_local : 0;
        }
      } else {
        int tile_base = inner_block_cur * kInnerBlockSize_;
        for (int j = idx_in_group; j < NumRowsPerGroup_; j += GroupSize_) {
          int token_pos = tile_base + group_idx * NumRowsPerGroup_ + j;
          int block_idx = token_pos / kKBlockSize;
          int offset_in_block = token_pos % kKBlockSize;
          int block_id = (block_idx < seqlen_info.seqlen_k / kKBlockSize) ? group_token_ptr[block_idx] : -1;
          int logical_k = block_id * kKBlockSize + offset_in_block;
          group_rows[j] = (block_id >= 0) ? logical_k * nhk + head_local : 0;
        }
      }
    } else {
      // ── LoopQ: fill Q packed rows ──
      int tile_first_packed_row = inner_block_cur * kInnerBlockSize_;
      int base = tile_first_packed_row + group_idx * NumRowsPerGroup_;
      int total_q_packed = seqlen_info.seqlen_q;
      int max_inv_topk_val = total_q_packed / QheadPerKhead;

      for (int j = idx_in_group; j < NumRowsPerGroup_; j += GroupSize_) {
        int packed_row = base + j;
        if (packed_row < total_q_packed) {
          int q_token_local_idx = packed_row / QheadPerKhead;
          int sub_head = packed_row % QheadPerKhead;
          int q_token = (q_token_local_idx < max_inv_topk_val) ? group_token_ptr[q_token_local_idx] : -1;
          group_rows[j] = (q_token >= 0) ? q_token * QheadPerKhead + sub_head : 0;
        } else {
          group_rows[j] = 0;
        }
      }
    }
  }

  // LoopQ only: absolute packed row for TMA coordinate computation
  CUTLASS_DEVICE
  int get_packed_first_row() const {
    static_assert(IsProducer && IsLoopQ, "get_packed_first_row() is LoopQ producer-only");
    int packed_row = inner_block_cur * kInnerBlockSize_;
    int q_token_local_idx = packed_row / QheadPerKhead;
    int sub_head_offset = packed_row % QheadPerKhead;
    int q_token = (q_token_local_idx < seqlen_info.seqlen_q / QheadPerKhead) ? group_token_ptr[q_token_local_idx] : -1;
    return (q_token >= 0) ? q_token * QheadPerKhead + sub_head_offset : 0;
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_block, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    flash::advance_block_cur<kDir>(inner_block_cur);
    if constexpr (IsProducer && !IsLoopQ && kKBlockSize <= 1) {
      // Token-level LoopK: sliding window — advance pointer
      if (!is_finish()) {
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          group_token_ptr -= kInnerBlockSize_;
        } else {
          group_token_ptr += kInnerBlockSize_;
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
