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

std::tuple<Flash_fwd_params, at::Tensor, at::Tensor> prepare_mha_fwd(
     const at::Tensor& q,
     const at::Tensor& k,
     const at::Tensor& v,
     std::optional<at::Tensor>& out_,
     std::optional<at::Tensor>& softmax_lse_,
     const at::Tensor& q_ranges,
     const at::Tensor& k_ranges,
     int max_seqlen_q,
     int max_seqlen_k,
     std::optional<const at::Tensor>& attn_type_map_,
     std::optional<const at::Tensor>& merge_q_ranges_,
     std::optional<const at::Tensor>& qk_map_,
     std::optional<const at::Tensor>& unique_count_,
     float const softmax_scale,
     float const softcap,
     bool const disable_fwd_atomic_reduction,
     std::optional<at::ScalarType> out_type_,
     bool const deterministic,
     int const sm_margin) {
   auto dprops = at::cuda::getCurrentDeviceProperties();
   bool is_sm9x = dprops->major >= 9;
   TORCH_CHECK(is_sm9x, "Flexible Flash Attention only supports Hopper GPUs or newer.");

   int const batch_size = q_ranges.size(0);
   int const total_q = q.size(0);
   int const total_k = k.size(0);
   int const num_heads_qo = q.size(1);
   int const num_heads_kv = k.size(1);
   int const head_size = q.size(2);

   auto q_type = q.scalar_type();
   TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16, "Flexible Flash Attention only supports fp16 and bf16 data type");
   TORCH_CHECK(k.scalar_type() == q_type, "query and key must have the same dtype");
   TORCH_CHECK(v.scalar_type() == q_type, "query and value must have the same dtype");
   CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
   TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q/k/v must be 3D");
   CHECK_SHAPE(q, total_q, num_heads_qo, head_size);
   CHECK_SHAPE(k, total_k, num_heads_kv, head_size);
   CHECK_SHAPE(v, total_k, num_heads_kv, head_size);
   TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "q/k/v last dim must be contiguous");

   TORCH_CHECK(q_ranges.dtype() == torch::kInt32 && k_ranges.dtype() == torch::kInt32, "ranges must be int32");
   CHECK_DEVICE(q_ranges); CHECK_DEVICE(k_ranges);
   CHECK_SHAPE(q_ranges, batch_size, 2);
   CHECK_SHAPE(k_ranges, batch_size, 2);
   CHECK_CONTIGUOUS(q_ranges); CHECK_CONTIGUOUS(k_ranges);

   at::Tensor attn_type_map; bool const has_attn_type_map = attn_type_map_.has_value();
   if (has_attn_type_map) {
     attn_type_map = attn_type_map_.value();
     CHECK_DEVICE(attn_type_map);
     TORCH_CHECK(attn_type_map.dtype() == torch::kInt32);
     CHECK_SHAPE(attn_type_map, batch_size);
     CHECK_CONTIGUOUS(attn_type_map);
   }

   int merge_batch_size = batch_size; at::Tensor merge_q_ranges; at::Tensor qk_map; at::Tensor unique_count;
   bool const has_merge_q_ranges = merge_q_ranges_.has_value();
   bool const has_qk_map = qk_map_.has_value();
   bool const has_unique_count = unique_count_.has_value();
   if (has_merge_q_ranges) {
     merge_q_ranges = merge_q_ranges_.value();
     TORCH_CHECK(merge_q_ranges.dtype() == torch::kInt32); CHECK_DEVICE(merge_q_ranges);
     merge_batch_size = merge_q_ranges.size(0); CHECK_CONTIGUOUS(merge_q_ranges);
   }
   if (has_qk_map) {
     qk_map = qk_map_.value();
     TORCH_CHECK(qk_map.dtype() == torch::kInt32); CHECK_DEVICE(qk_map); CHECK_SHAPE(qk_map, merge_batch_size); CHECK_CONTIGUOUS(qk_map);
   }
   if (has_unique_count) {
     unique_count = unique_count_.value();
     TORCH_CHECK(unique_count.dtype() == torch::kInt32); CHECK_DEVICE(unique_count); CHECK_SHAPE(unique_count); CHECK_CONTIGUOUS(unique_count);
   }
   TORCH_CHECK((has_merge_q_ranges == has_qk_map && has_qk_map == has_unique_count), "merge_q_ranges/qk_map/unique_count must be provided together");

   int const max_headdim = get_max_headdim();
   TORCH_CHECK(head_size <= max_headdim);
   TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
   TORCH_CHECK(num_heads_qo % num_heads_kv == 0, "Number of heads in key/value must divide number of heads in query");

   auto opts = q.options();
   auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
   int const head_size_rounded = round_up_headdim(head_size);
   int const max_seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
   int const max_seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

   at::cuda::CUDAGuard device_guard{(char)q.get_device()};

   at::Tensor softmax_lse;
   if (softmax_lse_.has_value()) {
     softmax_lse = softmax_lse_.value();
     TORCH_CHECK(softmax_lse.scalar_type() == at::kFloat); CHECK_DEVICE(softmax_lse); CHECK_SHAPE(softmax_lse, total_q, num_heads_qo); CHECK_CONTIGUOUS(softmax_lse);
   } else {
     softmax_lse = torch::full({num_heads_qo, total_q}, -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
   }

   at::ScalarType out_type;
   if (out_type_.has_value()) out_type = out_type_.value();
   else if (out_.has_value()) out_type = out_.value().scalar_type();
   else out_type = !disable_fwd_atomic_reduction ? at::kFloat : q_type;
   TORCH_CHECK(out_type == at::kFloat || out_type == at::kBFloat16 || out_type == at::kHalf);

   at::Tensor out;
   if (out_.has_value()) {
     out = out_.value();
     TORCH_CHECK(out.scalar_type() == out_type); CHECK_DEVICE(out); CHECK_SHAPE(out, total_q, num_heads_qo, head_size); TORCH_CHECK(out.stride(-1) == 1);
   } else {
     out = torch::empty_like(q, opts.dtype(out_type));
   }

   int element_size = (q_type == at::ScalarType::BFloat16) ? sizeof(cutlass::bfloat16_t) : sizeof(cutlass::half_t);
   int const kBlockM = std::get<0>(tile_size_fwd_sm90(head_size, element_size, softcap > 0.0));
   at::Tensor range_locks = torch::empty({(total_q + kBlockM - 1) / kBlockM + 1, num_heads_qo}, opts.dtype(torch::kInt32));
   at::Tensor tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
   if (!disable_fwd_atomic_reduction) range_locks.zero_();

   at::Tensor determin_range_locks = torch::empty({(total_q + kBlockM - 1) / kBlockM + 1, num_heads_qo * 2}, opts.dtype(torch::kInt32));
   int const num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
   at::Tensor determin_conflict_state = torch::empty({num_sm, (total_q + kBlockM - 1) / kBlockM + 1}, opts.dtype(torch::kInt32));
   if (deterministic) { determin_range_locks.zero_(); determin_conflict_state.zero_(); }

   Flash_fwd_params params;
   set_params_fprop(
       params,
       batch_size,
       max_seqlen_q,
       max_seqlen_k,
       max_seqlen_q_rounded,
       max_seqlen_k_rounded,
       total_q,
       total_k,
       num_heads_qo,
       num_heads_kv,
       head_size,
       head_size_rounded,
       q,
       k,
       v,
       out,
       q_ranges.data_ptr(),
       k_ranges.data_ptr(),
       range_locks.data_ptr(),
       deterministic,
       deterministic ? determin_range_locks.data_ptr() : nullptr,
       deterministic ? determin_conflict_state.data_ptr() : nullptr,
       has_attn_type_map ? attn_type_map.data_ptr() : nullptr,
       merge_batch_size,
       has_merge_q_ranges ? merge_q_ranges.data_ptr() : nullptr,
       has_qk_map ? qk_map.data_ptr() : nullptr,
       has_unique_count ? unique_count.data_ptr() : nullptr,
       softmax_lse.data_ptr(),
       softmax_scale,
       tile_count_semaphore.data_ptr(),
       softcap,
       sm_margin,
       disable_fwd_atomic_reduction);

   return {params, out, softmax_lse};
}
