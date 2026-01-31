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

#include "flex_flash_common.hpp"

void set_params_fprop(
    Flash_fwd_params& params,
    const size_t b,
    const size_t total_q,
    const size_t total_k,
    const size_t total_sink,
    const size_t h_qo,
    const size_t h_kv,
    const size_t d,
    const size_t d_rounded,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor sink,
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
    void* sparse_load_loop_count_d,
    void* sparse_load_invalid_count_d,
    void* equal_k_range_size_d,
    void* softmax_lse_d,
    void* max_logit_d,
    float const softmax_scale,
    void* tile_count_semaphore_d,
    float const softcap,
    flash::SinkLayout const sink_layout,
    int const sm_margin,
    bool const disable_fwd_atomic_reduction,
    int const max_seqlen_q,
    bool const has_max_seqlen_q,
    int const blocks_per_batch,
    int const tiles_per_batch_per_intergroup,
    int const max_tile_idx) {
  // Reset the parameters
  params = {};

  // Set dimensions
  params.b = b;
  params.h_qo = h_qo;
  params.h_kv = h_kv;
  params.total_q = total_q;
  params.total_k = total_k;
  params.total_sink = total_sink;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the compute and output types for the kernel.
  // Compute type is the type of the input tensors.
  // Output type is the type of the output tensor.
  params.compute_type = q.scalar_type();
  params.out_type = kernel_out.scalar_type();

  params.disable_fwd_atomic_reduction = disable_fwd_atomic_reduction;

  // Set the pointers of Q, K, V, sink
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.sink_ptr = sink.data_ptr();

  // Set the strides of Q, K, V
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);

  // Set the pointer of O
  params.o_ptr = kernel_out.data_ptr();

  // Set the strides of O
  // All stride are in elements, not bytes.
  params.o_row_stride = kernel_out.stride(-3);
  params.o_head_stride = kernel_out.stride(-2);

  // Set other pointers
  params.q_ranges = static_cast<int2*>(q_ranges_d);
  params.k_ranges = static_cast<int2*>(k_ranges_d);
  params.attn_type_map = static_cast<int*>(attn_type_map_d);

  // Set auto range merge
  params.merge_q_ranges = static_cast<int2*>(merge_q_ranges_d);
  params.qk_map = static_cast<int*>(qk_map_d);
  params.unique_count = static_cast<int*>(unique_count_d);
  params.merge_batch_size = merge_batch_size;
  params.sparse_load_loop_count = static_cast<int*>(sparse_load_loop_count_d);
  params.sparse_load_invalid_count = static_cast<uint8_t*>(sparse_load_invalid_count_d);
  params.equal_k_range_size = static_cast<int*>(equal_k_range_size_d);

  // Set kernel utility pointers
  params.range_locks = static_cast<int*>(range_locks_d);
  params.tile_count_semaphore = static_cast<int*>(tile_count_semaphore_d);

  // Set deterministic and it's pointers
  params.deterministic = deterministic;
  params.determin_range_locks = static_cast<int*>(determin_range_locks_d);
  params.determin_conflict_state = static_cast<int*>(determin_conflict_state_d);

  // Set softmax
  params.softmax_lse_ptr = softmax_lse_d;
  params.scale_softmax = softmax_scale;
  params.softcap = softcap;
  params.sink_layout = sink_layout;

  // Set max logit
  params.max_logit_ptr = max_logit_d;

  // Set the architecture and number of SMs to used in the kernel.
  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

  // Set optimization params for tile scheduling
  params.max_seqlen_q = max_seqlen_q;
  params.has_max_seqlen_q = has_max_seqlen_q;
  params.blocks_per_batch = blocks_per_batch;
  params.tiles_per_batch_per_intergroup = tiles_per_batch_per_intergroup;
  params.max_tile_idx = max_tile_idx;
}

void set_params_dgrad(
    Flash_bwd_params& params,
    const size_t b,
    const size_t total_q,
    const size_t total_k,
    const size_t total_q_rounded,
    const size_t num_m_block,
    const size_t total_sink,
    const size_t h_qo,
    const size_t h_kv,
    const size_t d,
    const size_t d_rounded,
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    const at::Tensor sink,
    const at::Tensor out,
    const at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    at::Tensor dsink,
    at::Tensor dsink_reduce_buf,
    at::Tensor dsink_reduce_cnt,
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
    flash::SinkLayout const sink_layout,
    int const sm_margin,
    bool const disable_bwd_dkv_atomic_reduction) {
  set_params_fprop(
      params,
      b,
      total_q,
      total_k,
      total_sink,
      h_qo,
      h_kv,
      d,
      d_rounded,
      q,
      k,
      v,
      sink,
      out,
      /*q_ranges_d=*/q_ranges_d,
      /*k_ranges_d=*/k_ranges_d,
      /*range_locks_d=*/nullptr,
      /*deterministic=*/deterministic,
      /*determin_range_locks_d=*/determin_range_locks_d,
      /*determin_conflict_state_d=*/determin_conflict_state_d,
      /*attn_type_map_d=*/attn_type_map_d,
      /*merge_batch_size=*/merge_batch_size,
      /*merge_q_ranges_d=*/nullptr,
      /*qk_map_d=*/nullptr,
      /*unique_count_d=*/nullptr,
      /*sparse_load_loop_count_d*/ nullptr,
      /*sparse_load_invalid_count_d*/ nullptr,
      /*equal_k_range_size_d=*/nullptr,
      /*softmax_lse_d=*/softmax_lse_d,
      /*max_logit_d=*/nullptr,
      /*softmax_scale=*/softmax_scale,
      /*tile_count_semaphore_d=*/tile_count_semaphore_d,
      /*softcap=*/softcap,
      /*sink_layout=*/sink_layout,
      /*sm_margin=*/sm_margin,
      /*disable_fwd_atomic_reduction=*/false);

  // Set backward-specific dimensions
  params.total_q_rounded = total_q_rounded;
  params.num_m_block = num_m_block;

  // Set backward-specific pointers and flags
  params.merge_k_ranges = static_cast<int2*>(merge_k_ranges_d);
  params.bwd_kq_map = static_cast<int*>(bwd_kq_map_d);
  params.bwd_unique_count = static_cast<int*>(bwd_unique_count_d);
  params.disable_bwd_dkv_atomic_reduction = disable_bwd_dkv_atomic_reduction;

  // HACK: override compute_type
  params.compute_type = dout.scalar_type();
  params.dkv_type = dk.scalar_type();

  // Set pointers and strides for dout, dq, dk, dv
  params.do_ptr = dout.data_ptr();
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.dsink_ptr = dsink.data_ptr();
  params.dsink_reduce_buf_ptr = dsink_reduce_buf.data_ptr();
  params.dsink_reduce_cnt_ptr = dsink_reduce_cnt.data_ptr();
  params.dq_row_stride = dq.stride(-3);
  params.dk_row_stride = dk.stride(-3);
  params.dv_row_stride = dv.stride(-3);
  params.dq_head_stride = dq.stride(-2);
  params.dk_head_stride = dk.stride(-2);
  params.dv_head_stride = dv.stride(-2);

  // Set softmax_lse_log2_ptr and dsoftmax_sum
  params.softmax_lse_log2_ptr = softmax_lse_log2_d;
  params.dsoftmax_sum = dsoftmax_sum_d;

  // Set the deterministic flag for dq path
  params.dq_determin_conflict_state = static_cast<int*>(dq_determin_conflict_state_d);
  params.dq_determin_range_locks = static_cast<int*>(dq_determin_range_locks_d);
}

void run_flash_fwd_post_process(Flash_fwd_params& params, cudaStream_t stream) {
  // Fast zero-fill for output accumulator if needed by kernel configuration
  OUT_DTYPE_SWITCH(params.out_type, TOut, [&] {
#ifndef FLASHATTENTION_DISABLE_HDIM64
    if (params.d <= 64) {
      return run_flash_fwd_post_process_<TOut, 64>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
    if (params.d <= 128) {
      return run_flash_fwd_post_process_<TOut, 128>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
    if (params.d <= 192) {
      return run_flash_fwd_post_process_<TOut, 192>(params, stream);
    }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
    if (params.d <= 256) {
      return run_flash_fwd_post_process_<TOut, 256>(params, stream);
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
