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

#include "flex_flash_common.hpp"

void set_params_fprop(
    Flash_fwd_params& params,
    const size_t b,
    const size_t max_seqlen_q,
    const size_t max_seqlen_k,
    const size_t max_seqlen_q_rounded,
    const size_t max_seqlen_k_rounded,
    const size_t total_q,
    const size_t total_k,
    const size_t h_qo,
    const size_t h_kv,
    const size_t d,
    const size_t d_rounded,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    at::Tensor kernel_out,
    void* q_ranges_d,
    void* k_ranges_d,
    void* range_locks_d,
    bool deterministic,
    void* determin_range_locks_d,
    void* determin_conflict_state_d,
    void* attn_type_map_d,
    int merge_batch_size,
    void* merge_q_ranges_d,
    void* qk_map_d,
    void* unique_count_d,
    void* softmax_lse_d,
    float softmax_scale,
    void* tile_count_semaphore_d,
    float const softcap,
    int const sm_margin,
    bool const disable_fwd_atomic_reduction) {
  params = {};
  params.compute_type = q.scalar_type();
  params.out_type = kernel_out.scalar_type();
  params.disable_fwd_atomic_reduction = disable_fwd_atomic_reduction;
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_ptr = kernel_out.data_ptr();
  params.o_row_stride = kernel_out.stride(-3);
  params.o_head_stride = kernel_out.stride(-2);
  params.q_ranges = static_cast<int*>(q_ranges_d);
  params.k_ranges = static_cast<int*>(k_ranges_d);
  params.attn_type_map = static_cast<int*>(attn_type_map_d);
  params.merge_q_ranges = static_cast<int*>(merge_q_ranges_d);
  params.qk_map = static_cast<int*>(qk_map_d);
  params.unique_count = static_cast<int*>(unique_count_d);
  params.range_locks = static_cast<int*>(range_locks_d);
  params.tile_count_semaphore = static_cast<int*>(tile_count_semaphore_d);
  params.deterministic = deterministic;
  params.determin_range_locks = static_cast<int*>(determin_range_locks_d);
  params.determin_conflict_state = static_cast<int*>(determin_conflict_state_d);
  params.softmax_lse_ptr = softmax_lse_d;
  params.b = b;
  params.merge_batch_size = merge_batch_size;
  params.h_qo = h_qo;
  params.h_kv = h_kv;
  params.max_seqlen_q = max_seqlen_q;
  params.max_seqlen_k = max_seqlen_k;
  params.max_seqlen_q_rounded = max_seqlen_q_rounded;
  params.max_seqlen_k_rounded = max_seqlen_k_rounded;
  params.total_q = total_q;
  params.total_k = total_k;
  params.d = d;
  params.d_rounded = d_rounded;
  params.scale_softmax = softmax_scale;
  params.softcap = softcap;
  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
}

void set_params_dgrad(
    Flash_bwd_params& params,
    const size_t b,
    const size_t max_seqlen_q,
    const size_t max_seqlen_k,
    const size_t max_seqlen_q_rounded,
    const size_t max_seqlen_k_rounded,
    const size_t total_q,
    const size_t total_k,
    const size_t h_qo,
    const size_t h_kv,
    const size_t d,
    const size_t d_rounded,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor out,
    const at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    void* q_ranges_d,
    void* k_ranges_d,
    void* attn_type_map_d,
    int merge_batch_size,
    void* merge_k_ranges_d,
    void* bwd_kq_map_d,
    void* bwd_unique_count_d,
    void* softmax_lse_d,
    void* softmax_lse_log2_d,
    void* dsoftmax_sum_d,
    float softmax_scale,
    void* tile_count_semaphore_d,
    const float softcap,
    bool const deterministic,
    void* determin_range_locks_d,
    void* determin_conflict_state_d,
    void* dq_determin_conflict_state_d,
    void* dq_determin_range_locks_d,
    int const sm_margin,
    bool const disable_bwd_dkv_atomic_reduction) {
  set_params_fprop(
      params,
      b,
      max_seqlen_q,
      max_seqlen_k,
      max_seqlen_q_rounded,
      max_seqlen_k_rounded,
      total_q,
      total_k,
      h_qo,
      h_kv,
      d,
      d_rounded,
      q,
      k,
      v,
      out,
      /*q_ranges_d*/ q_ranges_d,
      /*k_ranges_d*/ k_ranges_d,
      /*range_locks_d*/ nullptr,
      /*deterministic*/ deterministic,
      /*determin_range_locks_d*/ determin_range_locks_d,
      /*determin_conflict_state_d*/ determin_conflict_state_d,
      /*attn_type_map_d*/ attn_type_map_d,
      /*merge_batch_size*/ merge_batch_size,
      /*merge_q_ranges_d*/ nullptr,
      /*qk_map_d*/ nullptr,
      /*unique_count_d*/ nullptr,
      /*softmax_lse_d*/ softmax_lse_d,
      /*softmax_scale*/ softmax_scale,
      /*tile_count_semaphore_d*/ tile_count_semaphore_d,
      /*softcap*/ softcap,
      /*sm_margin*/ sm_margin,
      /*disable_fwd_atomic_reduction*/ false);

  params.merge_k_ranges = static_cast<int*>(merge_k_ranges_d);
  params.bwd_kq_map = static_cast<int*>(bwd_kq_map_d);
  params.bwd_unique_count = static_cast<int*>(bwd_unique_count_d);
  params.disable_bwd_dkv_atomic_reduction = disable_bwd_dkv_atomic_reduction;
  params.compute_type = dout.scalar_type();
  params.dkv_type = dk.scalar_type();
  params.do_ptr = dout.data_ptr();
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.dq_row_stride = dq.stride(-3);
  params.dk_row_stride = dk.stride(-3);
  params.dv_row_stride = dv.stride(-3);
  params.dq_head_stride = dq.stride(-2);
  params.dk_head_stride = dk.stride(-2);
  params.dv_head_stride = dv.stride(-2);
  params.softmax_lse_log2_ptr = softmax_lse_log2_d;
  params.dsoftmax_sum = dsoftmax_sum_d;
  params.dq_determin_conflict_state = static_cast<int*>(dq_determin_conflict_state_d);
  params.dq_determin_range_locks = static_cast<int*>(dq_determin_range_locks_d);
}

void run_fast_zero_fill(Flash_fwd_params& params, cudaStream_t stream) {
  OUT_DTYPE_SWITCH(params.out_type, TOut, [&] {
#ifndef FLASHATTENTION_DISABLE_HDIM64
    if (params.d <= 64) {
      return run_fast_zero_fill_<TOut, 64>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
    if (params.d <= 128) {
      return run_fast_zero_fill_<TOut, 128>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
    if (params.d <= 192) {
      return run_fast_zero_fill_<TOut, 192>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
    if (params.d <= 256) {
      return run_fast_zero_fill_<TOut, 256>(params, stream);
    }
#endif
  });
}

int get_max_headdim() {
#ifndef FLASHATTENTION_DISABLE_HDIM256
  return 256;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  return 192;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  return 128;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM64
  return 64;
#endif
  return 0;
}

int round_up_headdim(int head_size) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) {
    return 64;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) {
    return 128;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  if (head_size <= 192) {
    return 192;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
  if (head_size <= 256) {
    return 256;
  }
#endif
  return 256;
}

// // utility kernel binding (declared in flash.h). Use explicit function type to avoid overload ambiguity.
// using UniquePairsFn = std::tuple<at::Tensor, at::Tensor, at::Tensor> (*)(at::Tensor);

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("unique_consecutive_pairs", static_cast<UniquePairsFn>(&unique_consecutive_pairs_ext), "Finds unique (int, int) pairs from a pre-sorted tensor (CUDA extension)");
// }
