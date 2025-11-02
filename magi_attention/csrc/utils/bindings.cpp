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

#include <torch/extension.h>
#include <tuple>
#include "profile_utils.h"
// Forward declaration; implemented in unique_consecutive_pairs.cu
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> unique_consecutive_pairs_ext(torch::Tensor sorted_input_tensor);
// Forward declaration; implemented in preprocess_sparse_load.cu
torch::Tensor compute_sparse_load_loop_count(torch::Tensor k_ranges, torch::Tensor cu_k_ranges_num, torch::Tensor unique_count, int tile_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("unique_consecutive_pairs", &unique_consecutive_pairs_ext, "Find unique (int, int) pairs from a pre-sorted [N,2] int32 CUDA tensor");
  m.def("compute_sparse_load_loop_count", &compute_sparse_load_loop_count, "Compute sparse load loop counts for each unique Q range");
  m.def("start_event", &MagiEvents::start, "");
  m.def("stop_event", &MagiEvents::stop, "");
  m.def("elapsed_ms_event", &MagiEvents::elapsed_ms, "");
  m.def("destroy_event", &MagiEvents::destroy, "");
}
