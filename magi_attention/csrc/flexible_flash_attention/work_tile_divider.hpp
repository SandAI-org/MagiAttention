// Copyright (c) 2025 SandAI. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/pipeline/pipeline.hpp>

#include <cutlass/arch/grid_dependency_control.h>

namespace flash {

using namespace cute;

template <class ArchTag_, int kBlock>
class WorkTileDivider {
 public:
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t MaxThreadsPerBlock = 1024;
  using ArchTag = ArchTag_;
  // Host side kernel arguments
  struct Arguments {
    int const num_heads;
    int const num_batches;
    int* const ranges = nullptr;
  };

  // Device side kernel params
  struct Params {
    int num_heads;
    int num_batches;
    int* const ranges;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_heads, args.num_batches, args.ranges};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    return dim3(132, 1, 1);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {

  }
};

} // namespace flash