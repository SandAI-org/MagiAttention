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
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#include <torch/extension.h>

#include "sink_layout.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;
  void* __restrict__ sink_ptr;

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

  // Dimensions params
  int b, d, d_rounded;
  int total_q, total_k, total_sink;

  // The scaling factors for the kernel.
  float scale_softmax;
  float softcap;
  flash::SinkLayout sink_layout;

  // Ranges params (The triplet determines the specific computation)
  int2* __restrict__ q_ranges;
  int2* __restrict__ k_ranges;
  int* __restrict__ attn_type_map;

  // RangeMerge params
  int merge_batch_size;
  int2* __restrict__ merge_q_ranges;
  int* __restrict__ qk_map;
  int* __restrict__ unique_count;

  // Dtype params
  at::ScalarType compute_type;
  at::ScalarType out_type;

  // Performance tuning params
  bool disable_fwd_atomic_reduction;
  int* __restrict__ range_locks;

  // Deterministic params
  bool deterministic;
  int* __restrict__ determin_range_locks;
  int* __restrict__ determin_conflict_state;

  // Kernel utility params
  int arch;
  int num_sm;
  int* __restrict__ tile_count_semaphore;

  bool has_sink() const {
    return total_sink > 0;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
  using index_t = int64_t;

  // Dimensions params
  int total_q_rounded, num_m_block;

  // RangeMerge params
  int2* __restrict__ merge_k_ranges;
  int* __restrict__ bwd_kq_map;
  int* __restrict__ bwd_unique_count;

  // Dtype params
  at::ScalarType dkv_type;

  // The dO, dQ, dK and dV matrices.
  void* __restrict__ do_ptr;
  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  // The dsink-related matrices and workspace
  void* __restrict__ dsink_ptr;
  void* __restrict__ dsink_reduce_buf_ptr;
  void* __restrict__ dsink_reduce_cnt_ptr;

  // To accumulate dQ
  void* __restrict__ dq_accum_ptr;
  void* __restrict__ dk_accum_ptr;
  void* __restrict__ dv_accum_ptr;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  index_t do_row_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t do_head_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  // The pointer to the softmax d sum.
  void* __restrict__ dsoftmax_sum;
  void* __restrict__ softmax_lse_log2_ptr;

  // Performance tuning params
  bool disable_bwd_dkv_atomic_reduction;

  // Deterministic params
  int* __restrict__ dq_determin_conflict_state;
  int* __restrict__ dq_determin_range_locks;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Arch, typename T, typename T_out, int kHeadDim, bool Has_softcap, bool DisableFwdAtomicReduction, bool Deterministic>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);

template <int Arch, typename T, typename T_out, int kHeadDim, bool Has_softcap, bool DisableBwdDkvAtomicReduction, bool Deterministic, bool ProfileMode>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream);

template <typename T_out, uint32_t kHeadDim>
void run_flash_fwd_post_process_(Flash_fwd_params& params, cudaStream_t stream);
