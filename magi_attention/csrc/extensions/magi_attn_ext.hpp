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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>

#include "attn_ranges.hpp"
#include "rectangles.hpp"
#include "kernel_barrier.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME magi_attn_ext
#endif

namespace magi_attn_ext {
// Binary-Greedy-Parallel solve
void binary_greedy_parallel_solve(
    pybind11::object& rects,
    const pybind11::list& host_ranges_q,
    const pybind11::list& host_ranges_k,
    int num_heads_q,
    int num_heads_kv,
    int num_heads_group,
    pybind11::list& bucket_per_rank,
    int rank,
    bool debug_print);

// Optimized version of dynamic solver calc_host_and_remote_bucket_this_rank
pybind11::tuple cut_host_remote_buckets(const AttnRectangles& bucket_this_rank, const AttnRanges& host_ranges_q_this_rank, const AttnRanges& host_ranges_k_this_rank);

// Optimized version of _expand_attn_ranges for DynamicAttnSolver
AttnRanges expand_attn_ranges(const AttnRanges& ranges, int stride, int num_heads_group);

} // namespace magi_attn_ext
