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

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME magi_attn_ext
#endif

namespace magi_attn_ext {
// Calculate simplex edges for the binary greedy parallel algorithm
// Returns a list of tuples: (uj, vj, wj, cj, is_qo, tag)
pybind11::list calc_simplex_edges(
    int cp_size,
    const pybind11::list& rank_m,
    const pybind11::list& rank_n,
    const pybind11::list& comm_len_m,
    const pybind11::list& comm_len_n,
    const pybind11::list& sparse_solver_map,
    const pybind11::list& sparse_area_map,
    double area_avg,
    int num_heads_q,
    int num_heads_kv);

// Greedy selection algorithm for edge selection
// Returns a list of indices of selected edges
pybind11::list greedy_selection(int node_num, const pybind11::list& edges, double threshold);

// Greedy max flow algorithm for task assignment
// Returns a tuple: (is_feasible, assignment_result)
pybind11::tuple greedy_max_flow(
    int cp_size,
    const pybind11::list& simplex_edges,
    const pybind11::list& simplex_selected_edges,
    const pybind11::list& sparse_area_map,
    const pybind11::list& rank_m,
    const pybind11::list& rank_n,
    const pybind11::list& usp_choices,
    double area_avg,
    double unbalance_rate);

// Core Binary-Greedy solver (C++ implementation of Python binary_greedy)
// Returns solver_map: list[tuple[int, int, int]]
pybind11::list binary_greedy_solver(
    int cp_size,
    const pybind11::list& rank_m,
    const pybind11::list& rank_n,
    const pybind11::list& comm_len_m,
    const pybind11::list& comm_len_n,
    const pybind11::list& sparse_area_map,
    int num_heads_q,
    int num_heads_kv,
    const pybind11::list& usp_choices,
    int rank,
    bool debug_print);
} // namespace magi_attn_ext
