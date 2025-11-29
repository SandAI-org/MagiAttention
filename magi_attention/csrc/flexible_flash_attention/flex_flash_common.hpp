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

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include <cute/numeric/arithmetic_tuple.hpp>
#include <cutlass/numeric_types.h>

#include "cuda_check.h"
#include "flash.h"
#include "sink_layout.cuh"
#include "static_switch.h"
#include "tile_size.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

int get_max_headdim();
int round_up_headdim(int head_size);

void run_flash_fwd_post_process(Flash_fwd_params& params, cudaStream_t stream);

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
    void* softmax_lse_d,
    float softmax_scale,
    void* tile_count_semaphore_d,
    float const softcap = 0.f,
    flash::SinkLayout const sink_layout = flash::SinkLayout::SH,
    int const sm_margin = 0,
    bool const disable_fwd_atomic_reduction = false);

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
    const float softcap = 0.f,
    bool const deterministic = false,
    void* determin_range_locks_d = nullptr,
    void* determin_conflict_state_d = nullptr,
    void* dq_determin_conflict_state_d = nullptr,
    void* dq_determin_range_locks_d = nullptr,
    flash::SinkLayout const sink_layout = flash::SinkLayout::SH,
    int const sm_margin = 0,
    bool const disable_bwd_dkv_atomic_reduction = false);
