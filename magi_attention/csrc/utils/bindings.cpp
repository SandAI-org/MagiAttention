/**********************************************************************************
 * Copyright (c) 2025 SandAI. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0
 *********************************************************************************/

#include <torch/extension.h>
#include <tuple>

// Forward declaration; implemented in unique_consecutive_pairs.cu
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
unique_consecutive_pairs_ext(torch::Tensor sorted_input_tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "unique_consecutive_pairs",
      &unique_consecutive_pairs_ext,
      "Find unique (int, int) pairs from a pre-sorted [N,2] int32 CUDA tensor");
}


