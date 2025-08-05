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
  static constexpr uint32_t MaxThreadsPerBlock = 256;
  using ArchTag = ArchTag_;
  // Host side kernel arguments
  struct Arguments {
    int const num_batches;
    int num_heads_qo;
    int num_heads_kv;
    int max_seqlen;
    int* const tile_ranges = nullptr;
    int* const loop_ranges = nullptr;
    int* const job_list = nullptr;
    int* const arrangement = nullptr;
  };

  // Device side kernel params
  struct Params {
    int num_batches;
    int num_heads_kv;
    int num_heads_qo_per_kv;
    int job_max_num_per_batch;
    int job_max_num;
    int* const tile_ranges;
    int* const loop_ranges;
    int* const job_list;
    int* const arrangement;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    const int job_max_num_per_batch = (args.max_seqlen + kBlock - 1) / kBlock;
    return {args.num_batches,
            args.num_heads_kv,
            args.num_heads_qo / args.num_heads_kv,
            job_max_num_per_batch,
            args.num_batches * args.num_heads_qo * job_max_num_per_batch,
            args.tile_ranges, args.loop_ranges,
            args.job_list, args.arrangement};
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3 get_grid_shape(Params const& params) {
    return dim3((params.job_max_num + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock, 1, 1);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    int job_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (job_id >= params.job_max_num) return;
    int arrange[4] = {params.arrangement[0], params.arrangement[1], params.arrangement[2], params.arrangement[3]};
    int tmp[4] = {params.num_batches, params.num_heads_kv, params.num_heads_qo_per_kv, params.job_max_num_per_batch};
    int mod[4], id[4];
    mod[arrange[0]] = tmp[0];
    mod[arrange[1]] = tmp[1];
    mod[arrange[2]] = tmp[2];
    mod[arrange[3]] = tmp[3];
    int output_index = job_id;
    id[3] = job_id % mod[3];
    job_id /= mod[3];
    id[2] = job_id % mod[2];
    job_id /= mod[2];
    id[1] = job_id % mod[1];
    job_id /= mod[1];
    id[0] = job_id % mod[0];
    job_id /= mod[0];
    int batch_id = id[arrange[0]];
    int kv_inner_head_id = id[arrange[1]];
    int qo_intra_head_Id = id[arrange[2]];
    int nm_block = id[arrange[3]];
    params.job_list[output_index * 4 + 0] = batch_id;
    params.job_list[output_index * 4 + 1] = kv_inner_head_id;
    params.job_list[output_index * 4 + 2] = qo_intra_head_Id;
    params.job_list[output_index * 4 + 3] = nm_block;
  }
};

} // namespace flash