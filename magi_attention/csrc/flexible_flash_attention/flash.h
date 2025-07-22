/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#include <torch/extension.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads.
  int h_qo, h_kv;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  using index_t = int64_t;

  // The O matrix (output).
  void* __restrict__ o_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // The pointer to the softmax sum.
  void* __restrict__ softmax_lse_ptr;
  void* __restrict__ softmax_lseaccum_ptr;

  // The dimensions.
  // b = q_ranges.shape[0]; seqlen_q: max_seqlen_q, seqlen_k: max_seqlen_k
  int b, merge_batch_size, max_seqlen_q, max_seqlen_k, max_seqlen_knew, d, max_seqlen_q_rounded, max_seqlen_k_rounded, d_rounded, rotary_dim;
  int total_q, total_k, total_knew;
  int b_k; // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
  int dv, dv_rounded; // For the case where V headdim is different from Q/K headdim

  // The scaling factors for the kernel.
  float scale_softmax;
  float softcap;

  // array of length b holding the starting index and ending index of each sequence.
  // only used for flex flash attention.
  int* __restrict__ q_ranges;
  int* __restrict__ k_ranges;
  int* __restrict__ attn_type_map;
  int* __restrict__ merge_q_ranges;
  int* __restrict__ qk_map;
  int* __restrict__ merge_k_ranges;
  int* __restrict__ bwd_kq_map;

  at::ScalarType compute_type;
  at::ScalarType out_type;

  bool disable_fwd_atomic_reduction;

  int* __restrict__ tile_count_semaphore;
  int* __restrict__ range_locks;
  // int * __restrict__ num_m_blocks_ptr;
  // int * __restrict__ num_n_blocks_ptr;
  int* __restrict__ num_splits_dynamic_ptr;

  // For deterministic
  int* __restrict__ determin_range_locks;
  int* __restrict__ determin_conflict_state;
  int arch;
  int num_sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
  using index_t = int64_t;

  at::ScalarType dkv_type;

  bool disable_bwd_dkv_atomic_reduction;

  // The dO and dQKV matrices.
  void* __restrict__ do_ptr;
  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  // To accumulate dQ
  void* __restrict__ dq_accum_ptr;
  void* __restrict__ dk_accum_ptr;
  void* __restrict__ dv_accum_ptr;

  // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
  // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
  // dv_accum_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  // The pointer to the softmax d sum.
  void* __restrict__ dsoftmax_sum;
  void* __restrict__ softmax_lse_log2_ptr;

  int* __restrict__ dq_determin_conflict_state;
  int* __restrict__ dq_determin_range_locks;

  bool deterministic;
  index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Arch, typename T, typename T_out, int kHeadDim, bool Has_softcap, bool DisableFwdAtomicReduction>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

template <int Arch, typename T, typename T_out, int kHeadDim, bool Has_softcap, bool DisableBwdDkvAtomicReduction>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream);

template <typename T_out, uint32_t kHeadDim>
void run_fast_zero_fill_(Flash_fwd_params& params, cudaStream_t stream);
