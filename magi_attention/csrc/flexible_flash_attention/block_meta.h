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
// InnerLoopQ=false  =>  FWD / BWD-InnerLoopK (inner loop over n_block/K).
// InnerLoopQ=true   =>  BWD-InnerLoopQ  (inner loop over m_block/Q).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IsProducer, bool InnerLoopQ, bool RangeMerge, bool FlattenGQA, int PackGQAFactor, typename SeqlenInfo_t, typename BlockMN_t>
struct DenseBlockMeta {
  // All fields are by-value (no reference data members) to avoid register spilling to stack.
  // When !RangeMerge, the batch loop runs exactly once; mark it so callers can elide the while(true).
  static constexpr bool NeedsBatchLoop = RangeMerge;

  int const outer_tile_idx; // m_block when !InnerLoopQ, n_block when InnerLoopQ
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;

  SeqlenInfo_t seqlen_info;
  flash::AttnType attn_type;
  int inner_block_min; // n_block_min when !InnerLoopQ, m_block_min when InnerLoopQ
  int inner_block_cnt; // n_block_max when !InnerLoopQ, m_block_max when InnerLoopQ
  int inner_block_idx;

  int2 const* const q_ranges;
  int2 const* const k_ranges;
  int const* const attn_type_map;

  template <typename ParamsT, typename BlockCoordT, typename SharedStorage>
  CUTLASS_DEVICE DenseBlockMeta(ParamsT const& params, BlockCoordT const& block_coord, SharedStorage& shared_storage, int thread_idx = 0)
      : outer_tile_idx(get<0>(block_coord)),
        bidh(get<1>(block_coord)),
        // When FlattenGQA (PackGQA or CatGQA), the scheduler assigns bidh as
        // the kv-head index directly. Otherwise bidh is the q-head index and
        // we need to divide by PackGQAFactor to get bidh_kv.
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
    auto [min_, max_] = InnerLoopQ ? BlockMN_t::get_m_block_min_max(seqlen_info, outer_tile_idx, bidb, attn_type)
                                   : BlockMN_t::get_n_block_min_max(seqlen_info, outer_tile_idx, bidb, attn_type);
    inner_block_min = min_;
    inner_block_cnt = max_;
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
    return cute::make_tuple(outer_tile_idx, bidh, bidb);
  }

  CUTLASS_DEVICE
  bool is_valid() {
    return inner_block_min < inner_block_cnt;
  }

  CUTLASS_DEVICE
  bool is_finish() {
    return bidb >= end_batches;
  }

  template <flash::DispatchDirection Dir>
  CUTLASS_DEVICE void update_block_cur() {
    inner_block_idx = flash::init_block_cur<Dir>(inner_block_min, inner_block_cnt);
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
// BlockSparseBlockMeta: Unified producer/consumer via IsProducer template parameter.
// Used by both producer (IsProducer=true) and consumer (IsProducer=false) aliases.
//
// IsInnerLoopQ_=false (InnerLoopK): outer=Q (TMA), inner=KV (scatter), token_indices = KV positions
// IsInnerLoopQ_=true  (InnerLoopQ): outer=KV (TMA), inner=Q (scatter),  token_indices = Q positions
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool IsProducer,
    bool RangeMerge,
    bool PackGQA,
    int PackGQAFactor,
    int NumTokensPerLdstGroup_,
    int LdstGroupSize_,
    int NumProducerThreads_,
    int kBlockN_,
    bool InnerDirMaxToMin_,
    bool IsInnerLoopQ_>
struct BlockSparseBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr bool NeedsBatchLoop = true;
  static constexpr bool IsInnerLoopQ = IsInnerLoopQ_;
  static constexpr int InnerBlockSize = kBlockN_;

  int const outer_tile_idx; // m_block for InnerLoopK, n_block for InnerLoopQ
  int const bidh;
  int const bidh_kv;
  int bidb;
  int end_batches;
  flash::SeqlenInfo seqlen_info;
  flash::AttnType attn_type;

  int num_invalid_token;
  int inner_block_idx;
  int inner_block_cnt;

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
  // BlockSparse contract: all scatter-dim ranges have one uniform size (block-mask
  // generated ranges are uniform by construction; asserted at the Python entry).
  // This is what makes the O(1) div/mod cursor arithmetic in advance_token_idx valid.
  int range_size;

  // Read scatter range i with endpoints scaled to packed-row space.
  // InnerLoopQ + PackGQA: Q heads are folded into the row dimension, so the scatter
  // walk operates in PACKED-ROW space — every endpoint is scaled by PackGQAFactor.
  // token_indices then hold packed rows p = token * G + g, where g is the q-head
  // index within the kv group. InnerLoopK scatters KV rows (never head-packed, scale 1).
  CUTLASS_DEVICE
  int2 packed_range(int i) const {
    int2 r = (IsInnerLoopQ ? q_ranges : k_ranges)[i];
    if constexpr (IsInnerLoopQ && PackGQA) {
      r.x *= PackGQAFactor;
      r.y *= PackGQAFactor;
    }
    return r;
  }

  template <typename ParamsT, typename SharedStorage>
  CUTLASS_DEVICE BlockSparseBlockMeta(
      ParamsT const& params,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
      SharedStorage& shared_storage,
      int thread_idx = 0)
      : outer_tile_idx(get<0>(block_coord)),
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

    // No outer_tile_idx bounds check needed: the persistent scheduler computes tile
    // count from the outer-dimension range, so outer_tile_idx is always in-range
    // (symmetric for both InnerLoopQ and InnerLoopK).

    int2 const r0 = packed_range(bidb);
    range_size = r0.y - r0.x;
    int const total_tokens = (end_batches - bidb) * range_size;
    inner_block_cnt = (total_tokens + InnerBlockSize - 1) / InnerBlockSize;
    // Tile padding: last tile may extend past actual tokens; mask zeros these rows.
    num_invalid_token = inner_block_cnt * InnerBlockSize - total_tokens;
    inner_block_idx = flash::init_block_cur<kDir>(inner_block_min, inner_block_cnt);

    if constexpr (IsProducer) {
      int idx_in_warpgroup = thread_idx % 128;
      int ldst_group_idx = idx_in_warpgroup / LdstGroupSize_;

      if (!is_finish()) {
        seqlen_info = flash::SeqlenInfo{bidb, q_ranges, k_ranges};
        attn_type = static_cast<flash::AttnType>(attn_type_map ? attn_type_map[bidb] : 0);

        // Position the anchor at this group's first row of the first tile.
        // MaxToMin: anchor = HIGH end of the group's rows; MinToMax: LOW end.
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          int2 const r_last = packed_range(end_batches - 1);
          cur_range_idx = end_batches - 1;
          cur_range_inner_idx = r_last.y - r_last.x - 1;
          advance_token_idx(cur_range_idx, cur_range_inner_idx, kBlockN_ - (ldst_group_idx + 1) * NumTokensPerLdstGroup_);
        } else {
          cur_range_idx = bidb;
          cur_range_inner_idx = 0;
          advance_token_idx(cur_range_idx, cur_range_inner_idx, ldst_group_idx * NumTokensPerLdstGroup_);
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
  // The uniform range size (BlockSparse contract) makes this O(1) div/mod arithmetic.
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

  // Write this group's NumTokensPerLdstGroup_ token indices for the CURRENT tile into the
  // smem stage slot (rows [ldst_group_idx*NumTokensPerLdstGroup_, +NumTokensPerLdstGroup_)).
  // Called after producer_acquire (the held stage makes the slot writable). Lane j of
  // the group computes row j (strided by LdstGroupSize_) from a cursor copy of the anchor —
  // O(1) per row on the equal-range fast path. Caller must __syncwarp() before reading
  // the slot back (writer lanes ≠ reader lanes, but always within the same warp).
  CUTLASS_DEVICE
  void fill_token_indices(int* slot_rows, int token_idx_in_ldst_group, int ldst_group_idx) const {
    static_assert(IsProducer, "fill_token_indices() is producer-only");
    int* const group_rows = slot_rows + ldst_group_idx * NumTokensPerLdstGroup_;
    for (int j = token_idx_in_ldst_group; j < NumTokensPerLdstGroup_; j += LdstGroupSize_) {
      int range_idx = cur_range_idx;
      int inner_idx = cur_range_inner_idx;
      advance_token_idx(range_idx, inner_idx, j);
      // MaxToMin walks backward from the high-end anchor: j steps back = row (last - j)
      int const dst = InnerDirMaxToMin_ ? (NumTokensPerLdstGroup_ - 1 - j) : j;
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
    return cute::make_tuple(outer_tile_idx, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    flash::advance_block_cur<kDir>(inner_block_idx);
    if constexpr (IsProducer) {
      if (!is_finish()) {
        advance_token_idx(cur_range_idx, cur_range_inner_idx, kBlockN_);
      }
    }
  }

  CUTLASS_DEVICE
  bool is_finish() {
    if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
      return inner_block_idx < inner_block_min;
    } else {
      return inner_block_idx >= inner_block_cnt;
    }
  }

  CUTLASS_DEVICE
  int padding_block() const {
    return inner_block_cnt - 1;
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
// IndexSparseBlockMeta: Unified sparse block metadata for index-based attention.
// Handles both InnerLoopK and InnerLoopQ via the IsInnerLoopQ_ template parameter (like BlockSparseBlockMeta).
//
// Template params:
//   InnerBlockSize_    — inner scatter tile size (kBlockN for InnerLoopK, kBlockM for InnerLoopQ)
//   SparseKBlockSize_  — sparse K block size (kbs): how many K tokens each index entry covers
//                        (1 = token-level index sparse, ≥InnerBlockSize = block-level)
//
// IsInnerLoopQ_=false (InnerLoopK): outer=Q token (bidb), inner=K from forward topk indices
//   fill_token_indices fills K logical rows (k_token); head offset handled by ptr_gK_base
//   bidb = block_coord[2] = q_token; kv_head from scheduler bidh (intergroup)
//   indices layout: (total_q, nhk, max_topk) — per-head slice via bidh_kv * max_topk
// IsInnerLoopQ_=true  (InnerLoopQ): outer=K block (bidb), inner=Q from inner_indices
//   fill_token_indices fills Q packed rows: q_token * PackGQAFactor + sub_head
//   indices layout: (num_k_blocks, nhk, inner_max_topk)
//
// Fields (all const, computed in constructor init-list):
//   outer_tile_idx — tile index along the outer loop dimension
//   bidh        — head index from scheduler (= KV-head when PackGQA; = Q-head when !PackGQA)
//   bidb        — batch/token index (Q token for InnerLoopK, K block for InnerLoopQ)
//   nhk         — total KV-head count
//   bidh_kv     — resolved KV-head for this CTA (consistent with DenseBlockMeta/BlockSparseBlockMeta)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    bool IsProducer,
    bool PackGQA,
    int PackGQAFactor,
    int NumTokensPerLdstGroup_,
    int NumProducerThreads_,
    int LdstGroupSize_,
    int InnerBlockSize_,
    bool InnerDirMaxToMin_,
    int SparseKBlockSize_,
    bool IsInnerLoopQ_>
struct IndexSparseBlockMeta {
  static constexpr auto kDir = InnerDirMaxToMin_ ? flash::DispatchDirection::MaxToMin : flash::DispatchDirection::MinToMax;
  static constexpr int InnerBlockSize = InnerBlockSize_;
  static constexpr int SparseKBlockSize = SparseKBlockSize_;
  static constexpr bool IsInnerLoopQ = IsInnerLoopQ_;
  static constexpr bool NeedsBatchLoop = true;

  // ─── Scheduler-assigned coordinates (const, computed in init-list) ───
  int const outer_tile_idx;
  int const bidh; // Head index from scheduler (KV-head when PackGQA, Q-head otherwise)
  int const bidb;
  int const nhk; // Total KV-head count
  int const bidh_kv; // Resolved KV-head for this CTA (matches DenseBlockMeta/BlockSparseBlockMeta)

  // ─── Mutable state (set in constructor body) ───
  flash::SeqlenInfo seqlen_info;

  flash::AttnType attn_type = flash::AttnType::Full;
  int end_batches;

  int inner_block_idx;
  int inner_block_cnt;
  int num_invalid_token;
  static constexpr int inner_block_min = 0;

  int const* sparse_indices_ptr = nullptr;

  template <typename ParamsT, typename SharedStorage>
  CUTLASS_DEVICE IndexSparseBlockMeta(
      ParamsT const& params,
      cute::tuple<int32_t, int32_t, int32_t> const& block_coord,
      SharedStorage& shared_storage,
      int thread_idx = 0)
      : outer_tile_idx(get<0>(block_coord)), bidh(get<1>(block_coord)), bidb(get<2>(block_coord)), nhk(detail::nhk_of<ParamsT>::get(params)), bidh_kv([&]() -> int {
          if constexpr (PackGQA) {
            return bidh;
          } else {
            return params.qhead_per_khead_divmod.divide(bidh);
          }
        }()) {
    int const max_topk = params.index_sparse_max_topk;
    int const* row_ptr;
    int actual_topk;

    // Common: compute row_ptr and actual_topk (both branches use same layout)
    row_ptr = params.index_sparse_indices + static_cast<int64_t>(bidb) * nhk * max_topk + static_cast<int64_t>(bidh_kv) * max_topk;
    actual_topk = max_topk;
    for (int i = max_topk - 1; i >= 0 && row_ptr[i] < 0; --i)
      --actual_topk;

    int total_inner_tokens;
    if constexpr (!IsInnerLoopQ) {
      // ── InnerLoopK: bidb = Q token, indices = forward topk (Q→K) ──
      seqlen_info.offset_q = bidb;
      seqlen_info.seqlen_q = 1;
      total_inner_tokens = actual_topk * SparseKBlockSize;
      seqlen_info.seqlen_k = total_inner_tokens;
    } else {
      // ── InnerLoopQ: bidb = K block, indices = inner_indices (K→Q) ──
      seqlen_info.offset_k = bidb * SparseKBlockSize;
      seqlen_info.seqlen_k = SparseKBlockSize;
      total_inner_tokens = PackGQA ? actual_topk * PackGQAFactor : actual_topk;
      seqlen_info.offset_q = 0;
      seqlen_info.seqlen_q = total_inner_tokens;
    }

    inner_block_cnt = (total_inner_tokens + InnerBlockSize - 1) / InnerBlockSize;
    num_invalid_token = inner_block_cnt * InnerBlockSize - total_inner_tokens;

    inner_block_idx = flash::init_block_cur<kDir>(inner_block_min, inner_block_cnt);
    end_batches = bidb + 1;

    if constexpr (IsProducer) {
      if constexpr (!IsInnerLoopQ && SparseKBlockSize <= 1) {
        // Token-level InnerLoopK: pointer walks InnerBlockSize entries per tile
        int aligned_total = inner_block_cnt * InnerBlockSize;
        int ldst_group_idx = (thread_idx % NumProducerThreads_) / LdstGroupSize_;
        int group_offset;
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          group_offset = (aligned_total - InnerBlockSize) + ldst_group_idx * NumTokensPerLdstGroup_;
        } else {
          group_offset = ldst_group_idx * NumTokensPerLdstGroup_;
        }
        sparse_indices_ptr = row_ptr + group_offset;
      } else {
        // Block-level InnerLoopK or InnerLoopQ: absolute indexing from row_ptr
        sparse_indices_ptr = row_ptr;
      }
    }
  }

  // Fill token indices into the smem stage slot for the CURRENT tile.
  // InnerLoopK: fills logical K positions (head offset handled by ptr_gK_base)
  // InnerLoopQ: fills Q packed rows (q_token * PackGQAFactor + sub_head)
  CUTLASS_DEVICE
  void fill_token_indices(int* slot_rows, int token_idx_in_ldst_group, int ldst_group_idx) const {
    static_assert(IsProducer, "fill_token_indices() is producer-only");
    int* const group_rows = slot_rows + ldst_group_idx * NumTokensPerLdstGroup_;

    if constexpr (!IsInnerLoopQ) {
      // ── InnerLoopK: fill logical K positions ──
      // Head selection is done by ptr_gK_base = K_ptr + bidh_kv * head_stride,
      // so row indices are positions within the per-head K space [0, total_k).
      if constexpr (SparseKBlockSize <= 1) {
        for (int j = token_idx_in_ldst_group; j < NumTokensPerLdstGroup_; j += LdstGroupSize_) {
          int const k_token = sparse_indices_ptr[j];
          group_rows[j] = (k_token >= 0) ? k_token : 0;
        }
      } else {
        int tile_base = inner_block_idx * InnerBlockSize;
        for (int j = token_idx_in_ldst_group; j < NumTokensPerLdstGroup_; j += LdstGroupSize_) {
          int token_pos = tile_base + ldst_group_idx * NumTokensPerLdstGroup_ + j;
          int block_idx = token_pos / SparseKBlockSize;
          int offset_in_block = token_pos % SparseKBlockSize;
          int block_id = (block_idx < seqlen_info.seqlen_k / SparseKBlockSize) ? sparse_indices_ptr[block_idx] : -1;
          int logical_k = block_id * SparseKBlockSize + offset_in_block;
          group_rows[j] = (block_id >= 0) ? logical_k : 0;
        }
      }
    } else {
      // ── InnerLoopQ: fill Q rows ──
      int tile_first_row = inner_block_idx * InnerBlockSize;
      int base = tile_first_row + ldst_group_idx * NumTokensPerLdstGroup_;
      int total_q = seqlen_info.seqlen_q;

      if constexpr (PackGQA) {
        // PackGQA: interleave Q heads — packed row = token * G + sub_head
        int max_inner_topk_val = total_q / PackGQAFactor;
        for (int j = token_idx_in_ldst_group; j < NumTokensPerLdstGroup_; j += LdstGroupSize_) {
          int packed_row = base + j;
          if (packed_row < total_q) {
            int q_token_local_idx = packed_row / PackGQAFactor;
            int sub_head = packed_row % PackGQAFactor;
            int q_token = (q_token_local_idx < max_inner_topk_val) ? sparse_indices_ptr[q_token_local_idx] : -1;
            group_rows[j] = (q_token >= 0) ? q_token * PackGQAFactor + sub_head : 0;
          } else {
            group_rows[j] = 0;
          }
        }
      } else {
        // !PackGQA: each bidh selects one Q head — raw Q token indices only
        for (int j = token_idx_in_ldst_group; j < NumTokensPerLdstGroup_; j += LdstGroupSize_) {
          int local_idx = base + j;
          if (local_idx < total_q) {
            int q_token = sparse_indices_ptr[local_idx];
            group_rows[j] = (q_token >= 0) ? q_token : 0;
          } else {
            group_rows[j] = 0;
          }
        }
      }
    }
  }

  // Absolute packed row for TMA coordinate computation.
  // InnerLoopQ: maps inner_block_idx through inner_indices → absolute packed Q row.
  // InnerLoopK: maps inner_block_idx through index_sparse_indices → absolute packed K row.
  // NOTE: The bounds check (local_idx < seqlen_q) handles tile tail padding —
  // inner_block_cnt * InnerBlockSize may exceed actual token count; out-of-bounds
  // positions return 0 (a safe dummy row). The consumer's mask uses num_invalid_token
  // to zero these padding rows' attention scores, and the epilogue skips writing them.
  CUTLASS_DEVICE
  int get_packed_first_row() const {
    static_assert(IsProducer, "get_packed_first_row() is producer-only");
    if constexpr (IsInnerLoopQ) {
      if constexpr (PackGQA) {
        int packed_row = inner_block_idx * InnerBlockSize;
        int q_token_local_idx = packed_row / PackGQAFactor;
        int sub_head_offset = packed_row % PackGQAFactor;
        int q_token = (q_token_local_idx < seqlen_info.seqlen_q / PackGQAFactor) ? sparse_indices_ptr[q_token_local_idx] : -1;
        return (q_token >= 0) ? q_token * PackGQAFactor + sub_head_offset : 0;
      } else {
        int local_idx = inner_block_idx * InnerBlockSize;
        int q_token = (local_idx < seqlen_info.seqlen_q) ? sparse_indices_ptr[local_idx] : -1;
        return (q_token >= 0) ? q_token : 0;
      }
    } else {
      static_assert(SparseKBlockSize >= InnerBlockSize, "InnerLoopK get_packed_first_row() requires kbs >= kBlockN");
      return get_n_block_abs() * InnerBlockSize;
    }
  }

  // LoopK block-level only: absolute K block index for TMA tile load.
  // When kbs >= kBlockN, each inner_block_idx maps to exactly one K block index
  // from index_sparse_indices, and the tile is physically contiguous.
  CUTLASS_DEVICE
  int get_n_block_abs() const {
    static_assert(IsProducer && !IsInnerLoopQ && SparseKBlockSize >= InnerBlockSize, "get_n_block_abs() requires block-level LoopK with kbs >= kBlockN");
    int tiles_per_kblock = SparseKBlockSize / InnerBlockSize;
    int kblock_idx = inner_block_idx / tiles_per_kblock;
    int tile_within_kblock = inner_block_idx % tiles_per_kblock;
    int block_id = sparse_indices_ptr[kblock_idx];
    return (block_id >= 0) ? block_id * tiles_per_kblock + tile_within_kblock : 0;
  }

  CUTLASS_DEVICE
  auto get_epilogue_coord() const {
    return cute::make_tuple(outer_tile_idx, bidh, bidb);
  }

  CUTLASS_DEVICE
  void prefetch() {
    flash::advance_block_cur<kDir>(inner_block_idx);
    if constexpr (IsProducer && !IsInnerLoopQ && SparseKBlockSize <= 1) {
      // Token-level LoopK: sliding window — advance pointer
      if (!is_finish()) {
        if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
          sparse_indices_ptr -= InnerBlockSize;
        } else {
          sparse_indices_ptr += InnerBlockSize;
        }
      }
    }
  }

  CUTLASS_DEVICE
  bool is_finish() {
    if constexpr (kDir == flash::DispatchDirection::MaxToMin) {
      return inner_block_idx < inner_block_min;
    } else {
      return inner_block_idx >= inner_block_cnt;
    }
  }

  CUTLASS_DEVICE
  int padding_block() const {
    return inner_block_cnt - 1;
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
