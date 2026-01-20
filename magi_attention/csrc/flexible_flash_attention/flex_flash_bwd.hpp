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

#include <torch/version.h>

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace pybind11::detail {
template <>
struct type_caster<at::ScalarType> {
 public:
  PYBIND11_TYPE_CASTER(at::ScalarType, _("torch.dtype"));
  type_caster() : value(at::kFloat) {}
  bool load(handle src, bool) {
    PyObject* obj = src.ptr();
    if (THPDtype_Check(obj)) {
      value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
      return true;
    }
    return false;
  }
  static handle cast(const at::ScalarType& src, return_value_policy /* policy */, handle /* parent */) {
    return Py_NewRef(torch::getTHPDtype(src));
  }
};
} // namespace pybind11::detail
#endif

#include "flex_flash_common.hpp"

// void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream) {
//   ARCH_SWITCH(params.arch, Arch, [&] {
//     SOFTCAP_SWITCH(params.softcap > 0.f, Has_softcap, [&] {
//       BOOL_SWITCH(params.disable_bwd_dkv_atomic_reduction, DisableBwdDkvAtomicReduction, [&] {
//         COMPUTE_DTYPE_SWITCH(params.compute_type, TCompute, [&] {
//           OUT_DTYPE_SWITCH(params.dkv_type, TDkv, [&] {
// #ifndef FLASHATTENTION_DISABLE_HDIM64
//             if (params.d <= 64) {
//               return run_mha_bwd_<Arch, TCompute, TDkv, 64, Has_softcap, DisableBwdDkvAtomicReduction>(params, stream);
//             }
// #endif
// #ifndef FLASHATTENTION_DISABLE_HDIM96
//             if (params.d <= 96) {
//               return run_mha_bwd_<Arch, TCompute, TDkv, 96, Has_softcap, DisableBwdDkvAtomicReduction>(params, stream);
//             }
// #endif
// #ifndef FLASHATTENTION_DISABLE_HDIM128
//             if (params.d <= 128) {
//               return run_mha_bwd_<Arch, TCompute, TDkv, 128, Has_softcap, DisableBwdDkvAtomicReduction>(params, stream);
//             }
// #endif
// #ifndef FLASHATTENTION_DISABLE_HDIM192
//             if (params.d <= 192) {
//               return run_mha_bwd_<Arch, TCompute, TDkv, 192, Has_softcap, DisableBwdDkvAtomicReduction>(params, stream);
//             }
// #endif
// #ifndef FLASHATTENTION_DISABLE_HDIM256
//             if (params.d <= 256) {
//               return run_mha_bwd_<Arch, TCompute, TDkv, 256, Has_softcap, DisableBwdDkvAtomicReduction>(params, stream);
//             }
// #endif
//           });
//         });
//       });
//     });
//   });
// }

template <bool Deterministic = false, bool DisableDkvAtomic = false, bool SwapBwdQKLoop = false>
std::tuple<Flash_bwd_params, at::Tensor, at::Tensor, at::Tensor, at::Tensor> prepare_mha_bwd(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& sink_,
    const at::Tensor& out,
    std::optional<const at::Tensor>& dq_,
    std::optional<const at::Tensor>& dk_,
    std::optional<const at::Tensor>& dv_,
    std::optional<const at::Tensor>& dsink_,
    const at::Tensor& softmax_lse,
    const at::Tensor& q_ranges,
    const at::Tensor& k_ranges,
    std::optional<const at::Tensor>& attn_type_map_,
    std::optional<const at::Tensor>& merge_k_ranges_,
    std::optional<const at::Tensor>& bwd_kq_map_,
    std::optional<const at::Tensor>& bwd_unique_count_,
    float const softmax_scale,
    float const softcap,
    std::optional<at::ScalarType> dq_type_,
    std::optional<at::ScalarType> dk_type_,
    std::optional<at::ScalarType> dv_type_,
    const std::string& sink_layout_,
    int const sm_margin) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif

  // Check compute capability
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm9x = dprops->major >= 9;
  // Check compute capability
  TORCH_CHECK(is_sm9x, "Flexible Flash Attention only supports Hopper GPUs or newer.");

  int batch_size = q_ranges.size(0);
  int const total_q = q.size(0);
  int const total_k = k.size(0);
  int const num_heads_qo = q.size(1);
  int const num_heads_kv = k.size(1);
  int const head_size = q.size(2);
  auto opts = q.options();

  // Check q, k, v, out, dout (dtype, device, layout)
  auto q_type = q.scalar_type();
  TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_type && v.dtype() == q_type && out.dtype() == q_type && dout.dtype() == q_type);
  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(out);
  CHECK_DEVICE(dout);
  CHECK_SHAPE(q, total_q, num_heads_qo, head_size);
  CHECK_SHAPE(out, total_q, num_heads_qo, head_size);
  CHECK_SHAPE(dout, total_q, num_heads_qo, head_size);
  CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
  CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 && out.stride(-1) == 1 && dout.stride(-1) == 1);
  TORCH_CHECK(!DisableDkvAtomic or num_heads_qo == num_heads_kv, "disable_bwd_dkv_atomic_reduction can only be set with MHA, instead of GQA or MQA");

  // check softmax_lse (dtype, device, layout)
  TORCH_CHECK(softmax_lse.dtype() == at::kFloat);
  CHECK_DEVICE(softmax_lse);
  CHECK_SHAPE(softmax_lse, total_q, num_heads_qo);
  TORCH_CHECK(softmax_lse.stride(-1) == 1);

  TORCH_CHECK(q_ranges.dtype() == torch::kInt32 && k_ranges.dtype() == torch::kInt32);
  CHECK_DEVICE(q_ranges);
  CHECK_DEVICE(k_ranges);
  CHECK_SHAPE(q_ranges, batch_size, 2);
  CHECK_SHAPE(k_ranges, batch_size, 2);
  CHECK_CONTIGUOUS(q_ranges);
  CHECK_CONTIGUOUS(k_ranges);

  // Init attn_type_map
  at::Tensor attn_type_map;
  bool const has_attn_type_map = attn_type_map_.has_value();
  if (has_attn_type_map) {
    attn_type_map = attn_type_map_.value();
    TORCH_CHECK(attn_type_map.dtype() == torch::kInt32);
    CHECK_DEVICE(attn_type_map);
    CHECK_SHAPE(attn_type_map, batch_size);
    CHECK_CONTIGUOUS(attn_type_map);
  }

  // Transfer sink_layout
  flash::SinkLayout sink_layout = flash::str_to_sink_layout(sink_layout_);
  int const total_sink = sink_.has_value() ? (sink_layout == flash::SinkLayout::SSH ? sink_->size(1) : sink_->size(0)) : 0;

  // Init optional sink
  at::Tensor sink;
  if (sink_.has_value()) {
    sink = sink_.value();
  } else {
    // Create a dummy empty sink tensor with zero size
    switch (sink_layout) {
      case flash::SinkLayout::SH:
        sink = torch::empty({total_sink, num_heads_qo}, opts.dtype(at::kFloat));
        break;
      case flash::SinkLayout::SSH:
        sink = torch::empty({total_q, total_sink, num_heads_qo}, opts.dtype(at::kFloat));
        break;
      default:
        TORCH_CHECK(false, "Unsupported sink layout");
    }
  }
  TORCH_CHECK(sink.scalar_type() == at::kFloat, "sink must has dtype float");
  CHECK_DEVICE(sink);
  switch (sink_layout) {
    case flash::SinkLayout::SH:
      TORCH_CHECK(sink.dim() == 2, "sink must be 2D for SH layout");
      CHECK_SHAPE(sink, total_sink, num_heads_qo);
      CHECK_CONTIGUOUS(sink);
      break;
    case flash::SinkLayout::SSH:
      TORCH_CHECK(sink.dim() == 3, "sink must be 3D for SSH layout");
      CHECK_SHAPE(sink, total_q, total_sink, num_heads_qo);
      CHECK_CONTIGUOUS(sink);
      break;
    default:
      TORCH_CHECK(false, "Unsupported sink layout");
  }

  // Check merge_k_ranges, bwd_kq_map (dtype, device, layout) if given
  int merge_batch_size = batch_size;
  at::Tensor merge_k_ranges;
  at::Tensor bwd_kq_map;
  at::Tensor bwd_unique_count;
  bool const has_merge_k_ranges = merge_k_ranges_.has_value();
  bool const has_bwd_kq_map = bwd_kq_map_.has_value();
  bool const has_bwd_unique_count = bwd_unique_count_.has_value();
  if (has_merge_k_ranges) {
    merge_k_ranges = merge_k_ranges_.value();
    // HACK
    merge_batch_size = merge_k_ranges.size(0);
    // Check dtype, device and layout
    TORCH_CHECK(merge_k_ranges.dtype() == torch::kInt32);
    CHECK_DEVICE(merge_k_ranges);
    CHECK_SHAPE(merge_k_ranges, merge_batch_size, 2);
    CHECK_CONTIGUOUS(merge_k_ranges);
  }
  if (has_bwd_kq_map) {
    bwd_kq_map = bwd_kq_map_.value();
    // Check dtype, device and layout
    TORCH_CHECK(bwd_kq_map.dtype() == torch::kInt32);
    CHECK_DEVICE(bwd_kq_map);
    CHECK_SHAPE(bwd_kq_map, merge_batch_size + 1);
  }
  TORCH_CHECK(has_merge_k_ranges == has_bwd_kq_map, "merge_k_ranges and bwd_kq_map must be both given or both not given");
  if (has_bwd_unique_count) {
    bwd_unique_count = bwd_unique_count_.value();
    // Check dtype, device and layout
    TORCH_CHECK(bwd_unique_count.dtype() == torch::kInt32);
    CHECK_DEVICE(bwd_unique_count);
    CHECK_SHAPE(bwd_unique_count);
  }
  TORCH_CHECK(
      (has_merge_k_ranges == has_bwd_kq_map && has_bwd_kq_map == has_bwd_unique_count),
      "merge_k_ranges, bwd_kq_map, and bwd_unique_count must all be provided together or all be omitted");

  int const max_headdim = get_max_headdim();
  TORCH_CHECK(head_size % 8 == 0 && head_size <= max_headdim);
  TORCH_CHECK(num_heads_qo % num_heads_kv == 0);
  int element_size = (q_type == at::ScalarType::BFloat16) ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t);
  int const kBlockM = std::get<0>(tile_size_bwd_sm90<SwapBwdQKLoop>(head_size, element_size, softcap > 0.0));
  int const kBlockN = std::get<1>(tile_size_bwd_sm90<SwapBwdQKLoop>(head_size, element_size, softcap > 0.0));
  // Get rounded max_seqlen
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };

  // Determine output dtype for dq
  at::ScalarType dq_type = dq_type_.has_value() ? dq_type_.value() : (dq_.has_value() ? dq_.value().scalar_type() : at::ScalarType::Float);
  TORCH_CHECK(dq_type == at::ScalarType::Float);
  if (dq_.has_value()) {
    TORCH_CHECK(dq_.value().scalar_type() == dq_type, "dq must have the same dtype as dq_type (if given)");
  }
  // Determine output dtype for dk
  at::ScalarType dk_type =
      dk_type_.has_value() ? dk_type_.value() : (dk_.has_value() ? dk_.value().scalar_type() : (!DisableDkvAtomic ? at::ScalarType::Float : q_type));
  TORCH_CHECK(
      dk_type == at::ScalarType::Float || dk_type == at::ScalarType::BFloat16 || dk_type == at::ScalarType::Half,
      "Flexible Flash Attention only supports float, bf16 and fp16 for dk");
  if (dk_.has_value()) {
    TORCH_CHECK(dk_.value().scalar_type() == dk_type, "dk must have the same dtype as dk_type (if given)");
  }
  // Determine output dtype for dv
  at::ScalarType dv_type =
      dv_type_.has_value() ? dv_type_.value() : (dv_.has_value() ? dv_.value().scalar_type() : (!DisableDkvAtomic ? at::ScalarType::Float : q_type));
  TORCH_CHECK(
      dv_type == at::ScalarType::Float || dv_type == at::ScalarType::BFloat16 || dv_type == at::ScalarType::Half,
      "Flexible Flash Attention only supports float, bf16 and fp16 for dv");
  if (dv_.has_value()) {
    TORCH_CHECK(dv_.value().scalar_type() == dv_type, "dv must have the same dtype as dv_type (if given)");
  }
  // Check dk_type is same as dv_type
  TORCH_CHECK(dk_type == dv_type, "dk and dv must have the same dtype");

  // Init gradient tensors to return
  // dq, dk, dv, dsink are output tensors, if they are not given, we will create them.
  // If they are given, we will check dtype, device and layout.
  at::Tensor dq, dk, dv, dsink;
  if (dq_.has_value()) {
    dq = dq_.value();
    TORCH_CHECK(dq.scalar_type() == dq_type, "dq must have the same dtype as dq_type (if given)");
    CHECK_DEVICE(dq);
    CHECK_SHAPE(dq, total_q, num_heads_qo, head_size);
    TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
  } else {
    dq = torch::zeros_like(q, opts.dtype(dq_type));
  }
  if (dk_.has_value()) {
    dk = dk_.value();
    TORCH_CHECK(dk.dtype() == dk_type, "dk must have the same dtype as dk_type (if given)");
    CHECK_DEVICE(dk);
    CHECK_SHAPE(dk, total_k, num_heads_kv, head_size);
    TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
  } else {
    dk = torch::zeros_like(k, opts.dtype(dk_type));
  }
  if (dv_.has_value()) {
    dv = dv_.value();
    TORCH_CHECK(dv.dtype() == dv_type, "dv must have the same dtype as dv_type (if given)");
    CHECK_DEVICE(dv);
    CHECK_SHAPE(dv, total_k, num_heads_kv, head_size);
    TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
  } else {
    dv = torch::zeros_like(v, opts.dtype(dv_type));
  }
  if (sink_.has_value()) {
    if (dsink_.has_value()) {
      dsink = dsink_.value();
    } else {
      dsink = torch::zeros_like(sink_.value());
    }
  } else {
    // Create a dummy empty dsink tensor with zero size
    dsink = torch::empty_like(sink);
  }
  TORCH_CHECK(dsink.dtype() == sink.dtype(), "dsink must have the same dtype as sink");
  TORCH_CHECK(dsink.layout() == sink.layout(), "dsink must have the same layout as sink");
  CHECK_DEVICE(dsink);
  CHECK_CONTIGUOUS(dsink);

  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  // Create the rounded total_q to allocate dPsum and LSE_log2
  // to avoid OOB when loading them at the query tensor's seqlen boundary
  int const total_q_rounded = round_multiple(total_q + kBlockM - 1, kBlockM);

  // Create dsink_reduce_buf, which is a workspace to store the block-reduced dsink results
  // NOTE: when sink is not provided or sink_layout is SSH or there's only a single m block,
  // dsink_reduce_buf is a dummy empty tensor consuming no memory overhead
  int const num_m_block = cute::ceil_div(total_q_rounded, kBlockM);
  at::Tensor dsink_reduce_buf =
      torch::empty({num_m_block * (num_m_block > 1), total_sink * (sink_layout != flash::SinkLayout::SSH), num_heads_qo}, opts.dtype(torch::kFloat));

  // Create dsink_reduce_cnt, which is a semaphore to count the number of blocks
  // who's finished block reduction of dsink into dsink_reduce_buf for each head, so as to find the last finished block
  // which will reduce across blocks over dsink_reduce_buf to get the final reduced dsink
  // NOTE: when sink is not provided or sink_layout is SSH or there's only a single m block,
  // dsink_reduce_cnt is a dummy zero tensor consuming no memory overhead
  at::Tensor dsink_reduce_cnt =
      torch::zeros({num_heads_qo * (num_m_block > 1) * (total_sink > 0) * (sink_layout != flash::SinkLayout::SSH)}, opts.dtype(torch::kUInt32));

  // NOTE: we add a new dimension (4) for TMA alignment (16 bytes)
  // actually, we only use index 0 of the new dimension (4).
  at::Tensor softmax_d = torch::empty({num_heads_qo, total_q_rounded, 4}, opts.dtype(torch::kFloat));
  at::Tensor softmax_lse_log2 = torch::empty({num_heads_qo, total_q_rounded, 4}, opts.dtype(torch::kFloat));

  // Create tile_count_semaphore tensor, used to count the number of tiles
  at::Tensor tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
  at::Tensor determin_range_locks = torch::empty({(total_k + kBlockN - 1) / kBlockN + 1, num_heads_kv * 2}, opts.dtype(torch::kInt32));
  at::Tensor dq_determin_range_locks = torch::empty({(total_q + kBlockM - 1) / kBlockM + 1, num_heads_qo * 2}, opts.dtype(torch::kInt32));
  // Initialize determin_conflict_state, num_sm rows, ceil_div(total_k, kBlockN) + 1 columns
  int const num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
  at::Tensor determin_conflict_state = torch::empty({num_sm, (total_k + kBlockN - 1) / kBlockN + 1}, opts.dtype(torch::kInt32));
  at::Tensor dq_determin_conflict_state = torch::empty({num_sm, (total_q + kBlockM - 1) / kBlockM + 1}, opts.dtype(torch::kInt32));

  // If deterministic is enabled, we need to zero out the out_accum tensor and conflict state
  if constexpr (Deterministic) {
    determin_range_locks.zero_();
    determin_conflict_state.zero_();
    dq_determin_range_locks.zero_();
    dq_determin_conflict_state.zero_();
  }

  Flash_bwd_params params;
  set_params_dgrad(
      params,
      batch_size,
      total_q,
      total_k,
      total_q_rounded,
      num_m_block,
      total_sink,
      num_heads_qo,
      num_heads_kv,
      head_size,
      round_up_headdim(head_size),
      q,
      k,
      v,
      sink,
      out,
      dout, // input tensors
      dq,
      dk,
      dv,
      dsink, // output tensors
      dsink_reduce_buf,
      dsink_reduce_cnt, // workspace tensors
      /*q_ranges=*/q_ranges.data_ptr(),
      /*k_ranges=*/k_ranges.data_ptr(),
      /*attn_type_map=*/has_attn_type_map ? attn_type_map.data_ptr() : nullptr,
      /*merge_batch_size=*/merge_batch_size,
      /*merge_k_ranges=*/has_merge_k_ranges ? merge_k_ranges.data_ptr() : nullptr,
      /*bwd_kq_map=*/has_bwd_kq_map ? bwd_kq_map.data_ptr() : nullptr,
      /*bwd_unique_count=*/has_bwd_unique_count ? bwd_unique_count.data_ptr() : nullptr,
      /*softmax_lse=*/softmax_lse.data_ptr(),
      /*softmax_lse_log2=*/softmax_lse_log2.data_ptr(),
      /*dsoftmax_sum=*/softmax_d.data_ptr(),
      /*softmax_scale=*/softmax_scale,
      /*tile_count_semaphore=*/tile_count_semaphore.data_ptr(),
      /*softcap=*/softcap,
      /*deterministic=*/Deterministic,
      /*determin_range_locks=*/determin_range_locks.data_ptr(),
      /*determin_conflict_state=*/determin_conflict_state.data_ptr(),
      /*dq_determin_conflict_state=*/dq_determin_conflict_state.data_ptr(),
      /*dq_determin_range_locks=*/dq_determin_range_locks.data_ptr(),
      /*sink_layout=*/sink_layout,
      /*sm_margin=*/sm_margin,
      /*disable_bwd_dkv_atomic_reduction=*/DisableDkvAtomic);

  return {params, dq, dk, dv, dsink};
}
