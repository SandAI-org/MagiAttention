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
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "event.hpp"
#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace magi_attn_comm::grpcoll {
struct Meta {
  static std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<EventHandle>> get_group_cast_meta_from_t2r_idx(
      const torch::Tensor& t2r_idx,
      int num_ranks,
      int num_rdma_ranks,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_meta_stream,
      std::optional<at::cuda::CUDAStream> meta_stream) {
    GRPCOLL_HOST_ASSERT(t2r_idx.dim() == 2);
    GRPCOLL_HOST_ASSERT(t2r_idx.is_contiguous());
    GRPCOLL_HOST_ASSERT(num_ranks > 0);

    // Get meta stream
    at::cuda::CUDAStream meta_stream_ = meta_stream.has_value() ? meta_stream.value() : at::cuda::getStreamFromPool();

    // Allocate all tensors on meta stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_meta_stream) {
      GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
      at::cuda::setCurrentCUDAStream(meta_stream_);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(meta_stream_, previous_event.value());
    } else {
      stream_wait(meta_stream_, compute_stream);
    }

    auto num_tokens = static_cast<int>(t2r_idx.size(0));
    GRPCOLL_HOST_ASSERT(t2r_idx.size(1) == num_ranks);
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (num_rdma_ranks > 1)
      num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_group_cast_meta(
        /*t2r_idx=*/t2r_idx.data_ptr<int64_t>(),
        /*num_tokens_per_rank=*/num_tokens_per_rank.data_ptr<int>(),
        /*num_tokens_per_rdma_rank=*/num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
        /*is_token_in_rank=*/is_token_in_rank.data_ptr<bool>(),
        /*num_tokens=*/num_tokens,
        /*num_ranks=*/num_ranks,
        /*stream=*/meta_stream_);

    // Wait streams
    std::optional<EventHandle> event;
    if (async_op) {
      event = EventHandle(meta_stream_);
      // record tensors on meta stream
      for (auto& t : {t2r_idx, num_tokens_per_rank, is_token_in_rank}) {
        t.record_stream(meta_stream_);
        if (allocate_on_meta_stream)
          t.record_stream(compute_stream);
      }
      // record optional tensors on meta stream
      for (auto& to : {num_tokens_per_rdma_rank}) {
        to.has_value() ? to->record_stream(meta_stream_) : void();
        if (allocate_on_meta_stream)
          to.has_value() ? to->record_stream(compute_stream) : void();
      }
    } else {
      stream_wait(compute_stream, meta_stream_);
    }

    // Switch back compute stream
    if (allocate_on_meta_stream)
      at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank, event};
  }

  static std::tuple<torch::Tensor, std::optional<EventHandle>> get_a2av_perm_idx_from_src_idx(
      const torch::Tensor& output_split_sizes,
      const torch::Tensor& src_idx,
      int num_tokens,
      int num_ranks,
      std::optional<EventHandle>& previous_event,
      bool async_op,
      bool allocate_on_meta_stream,
      std::optional<at::cuda::CUDAStream> meta_stream) {
    GRPCOLL_HOST_ASSERT(output_split_sizes.dim() == 1 && src_idx.dim() == 1);
    GRPCOLL_HOST_ASSERT(output_split_sizes.is_contiguous() && src_idx.is_contiguous());
    GRPCOLL_HOST_ASSERT(output_split_sizes.size(0) == src_idx.size(0));
    GRPCOLL_HOST_ASSERT(num_ranks > 0);

    // Get meta stream
    at::cuda::CUDAStream meta_stream_ = meta_stream.has_value() ? meta_stream.value() : at::cuda::getStreamFromPool();

    // Allocate all tensors on meta stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_meta_stream) {
      GRPCOLL_HOST_ASSERT(previous_event.has_value() and async_op);
      at::cuda::setCurrentCUDAStream(meta_stream_);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
      stream_wait(meta_stream_, previous_event.value());
    } else {
      stream_wait(meta_stream_, compute_stream);
    }

    auto num_splits = static_cast<int>(src_idx.size(0));
    auto perm_to_a2av_idx = torch::empty({num_tokens}, dtype(torch::kInt64).device(torch::kCUDA));

    layout::get_a2av_perm_idx(
        // input ptr
        output_split_sizes.data_ptr<int64_t>(),
        src_idx.data_ptr<int64_t>(),
        // output ptr
        perm_to_a2av_idx.data_ptr<int64_t>(),
        // meta
        num_ranks,
        num_splits,
        // stream
        meta_stream_);

    // Wait streams
    std::optional<EventHandle> event;
    if (async_op) {
      event = EventHandle(meta_stream_);
      for (auto& t : {output_split_sizes, src_idx, perm_to_a2av_idx}) {
        t.record_stream(meta_stream_);
        if (allocate_on_meta_stream)
          t.record_stream(compute_stream);
      }
    } else {
      stream_wait(compute_stream, meta_stream_);
    }

    // Switch back compute stream
    if (allocate_on_meta_stream)
      at::cuda::setCurrentCUDAStream(compute_stream);

    return {perm_to_a2av_idx, event};
  }
};

} // namespace magi_attn_comm::grpcoll
