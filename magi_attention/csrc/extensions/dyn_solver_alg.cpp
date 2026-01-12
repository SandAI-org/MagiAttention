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

#include "magi_attn_ext.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>
#if __cplusplus >= 201703L
#include <execution>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace magi_attn_ext {

// C++ internal data structures for avoiding Python type conversions
struct Edge {
  int u;
  int v;
  double weight;
  int cost;
  bool is_qo;
  int tag;
};

struct EdgeSortEntry {
  double score;
  int index;
  int u;
  int v;
  int cost;

  // For descending order sort
  bool operator>(const EdgeSortEntry& other) const {
    if (score != other.score) {
      return score > other.score;
    }
    return index < other.index;
  }
};

struct Assignment {
  int i;
  int j;
  int rank;
};

// C++ version of calc_simplex_edges (no Python type conversions)
static std::vector<Edge> calc_simplex_edges_cpp(
    int cp_size,
    const std::vector<int>& rank_m_vec,
    const std::vector<int>& rank_n_vec,
    const std::vector<int>& comm_len_m_vec,
    const std::vector<int>& comm_len_n_vec,
    const std::vector<int>& solver_assigned_ranks,
    const std::vector<std::tuple<int, int, long long>>& sparse_area_vec,
    double area_avg,
    int num_heads_q,
    int num_heads_kv) {
  int m = static_cast<int>(rank_m_vec.size());
  int n = static_cast<int>(rank_n_vec.size());
  int num_areas = static_cast<int>(sparse_area_vec.size());

  // Pre-compute constant values
  double area_avg_const = area_avg / cp_size * 0.05;
  double unsolved_weight_factor = 0.95 / cp_size;

  // Group sparse area map by row and column for fast lookup
  std::vector<std::vector<std::tuple<int, long long, int>>> row_areas(m);
  std::vector<std::vector<std::tuple<int, long long, int>>> col_areas(n);
  std::vector<long long> row_sums(m, 0);
  std::vector<long long> col_sums(n, 0);

#ifdef _OPENMP
  std::vector<int> row_counts(m, 0);
  std::vector<int> col_counts(n, 0);
#pragma omp parallel for
  for (int idx = 0; idx < num_areas; ++idx) {
#pragma omp atomic
    row_counts[std::get<0>(sparse_area_vec[idx])]++;
#pragma omp atomic
    col_counts[std::get<1>(sparse_area_vec[idx])]++;
  }

  for (int i = 0; i < m; ++i)
    row_areas[i].resize(row_counts[i]);
  for (int j = 0; j < n; ++j)
    col_areas[j].resize(col_counts[j]);

  std::vector<int> row_pos(m, 0);
  std::vector<int> col_pos(n, 0);
#pragma omp parallel for
  for (int idx = 0; idx < num_areas; ++idx) {
    int i = std::get<0>(sparse_area_vec[idx]);
    int j = std::get<1>(sparse_area_vec[idx]);
    long long area = std::get<2>(sparse_area_vec[idx]);

    int r_idx, c_idx;
#pragma omp atomic capture
    r_idx = row_pos[i]++;
#pragma omp atomic capture
    c_idx = col_pos[j]++;

    row_areas[i][r_idx] = std::make_tuple(j, area, idx);
    col_areas[j][c_idx] = std::make_tuple(i, area, idx);
#pragma omp atomic
    row_sums[i] += area;
#pragma omp atomic
    col_sums[j] += area;
  }
#else
  // Process sparse_area_vec: (i, j, area)
  for (int idx = 0; idx < num_areas; ++idx) {
    int i = std::get<0>(sparse_area_vec[idx]);
    int j = std::get<1>(sparse_area_vec[idx]);
    long long area = std::get<2>(sparse_area_vec[idx]);

    row_areas[i].emplace_back(j, area, idx);
    col_areas[j].emplace_back(i, area, idx);
    row_sums[i] += area;
    col_sums[j] += area;
  }
#endif

  // Pre-calculate total number of edges
  int per_node_edges = cp_size - 1;
  std::size_t total_edges = static_cast<std::size_t>(m + n) * static_cast<std::size_t>(per_node_edges);
  std::vector<Edge> edges(total_edges);

  // Process Q and KV edges in parallel
#ifdef _OPENMP
// Extreme optimization: Combine Q and KV processing into a single parallel loop (m + n)
// This allows for better load balancing across threads between QO and KV tasks.
#pragma omp parallel
  {
    std::vector<double> comm_weight_list(cp_size);
#pragma omp for schedule(dynamic)
    for (int idx = 0; idx < m + n; ++idx) {
      if (idx < m) {
        // --- Process Q edges (Row i) ---
        int i = idx;
        int q_comm_cost = comm_len_m_vec[i] * num_heads_q;
        std::fill(comm_weight_list.begin(), comm_weight_list.end(), 0.0);
        double comm_weight_solved_sum = 0.0;

        for (const auto& entry : row_areas[i]) {
          double comm_weight = std::get<1>(entry);
          int global_idx = std::get<2>(entry);
          int assigned_rank = solver_assigned_ranks[global_idx];

          if (assigned_rank != -1) {
            comm_weight_list[assigned_rank] += comm_weight;
            comm_weight_solved_sum += comm_weight;
          }
        }

        double comm_weight_sum = row_sums[i];
        double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
        int rank_m_i = rank_m_vec[i];
        double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

        int edge_idx = i * per_node_edges;
        for (int rank = 0; rank < cp_size; ++rank) {
          if (rank != rank_m_i) {
            double edges_weight = (comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
            edges[edge_idx++] = {rank_m_i, rank, edges_weight, q_comm_cost, true, i};
          }
        }
      } else {
        // --- Process KV edges (Column j) ---
        int j = idx - m;
        int k_comm_cost = comm_len_n_vec[j] * num_heads_kv;
        std::fill(comm_weight_list.begin(), comm_weight_list.end(), 0.0);
        double comm_weight_solved_sum = 0.0;

        for (const auto& entry : col_areas[j]) {
          double comm_weight = std::get<1>(entry);
          int global_idx = std::get<2>(entry);
          int assigned_rank = solver_assigned_ranks[global_idx];

          if (assigned_rank != -1) {
            comm_weight_list[assigned_rank] += comm_weight;
            comm_weight_solved_sum += comm_weight;
          }
        }

        double comm_weight_sum = col_sums[j];
        double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
        int rank_n_j = rank_n_vec[j];
        double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

        int edge_idx = m * per_node_edges + j * per_node_edges;
        for (int rank = 0; rank < cp_size; ++rank) {
          if (rank != rank_n_j) {
            double edges_weight = (comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
            edges[edge_idx++] = {rank, rank_n_j, edges_weight, k_comm_cost, false, j};
          }
        }
      }
    }
  }
#else
  // Fallback for non-OpenMP builds
  std::vector<double> comm_weight_list(cp_size);
  int edge_ptr = 0;

  // Q edges
  for (int i = 0; i < m; ++i) {
    int q_comm_cost = comm_len_m_vec[i] * num_heads_q;
    std::fill(comm_weight_list.begin(), comm_weight_list.end(), 0.0);
    double comm_weight_solved_sum = 0.0;

    for (const auto& entry : row_areas[i]) {
      double comm_weight = std::get<1>(entry);
      int global_idx = std::get<2>(entry);
      int assigned_rank = solver_assigned_ranks[global_idx];

      if (assigned_rank != -1) {
        comm_weight_list[assigned_rank] += comm_weight;
        comm_weight_solved_sum += comm_weight;
      }
    }

    double comm_weight_sum = row_sums[i];
    double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
    int rank_m_i = rank_m_vec[i];
    double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

    for (int rank = 0; rank < cp_size; ++rank) {
      if (rank != rank_m_i) {
        double edges_weight = (comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
        edges[edge_ptr++] = {rank_m_i, rank, edges_weight, q_comm_cost, true, i};
      }
    }
  }

  // KV edges
  for (int j = 0; j < n; ++j) {
    int k_comm_cost = comm_len_n_vec[j] * num_heads_kv;
    std::fill(comm_weight_list.begin(), comm_weight_list.end(), 0.0);
    double comm_weight_solved_sum = 0.0;

    for (const auto& entry : col_areas[j]) {
      double comm_weight = std::get<1>(entry);
      int global_idx = std::get<2>(entry);
      int assigned_rank = solver_assigned_ranks[global_idx];

      if (assigned_rank != -1) {
        comm_weight_list[assigned_rank] += comm_weight;
        comm_weight_solved_sum += comm_weight;
      }
    }

    double comm_weight_sum = col_sums[j];
    double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
    int rank_n_j = rank_n_vec[j];
    double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

    for (int rank = 0; rank < cp_size; ++rank) {
      if (rank != rank_n_j) {
        double edges_weight = (comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
        edges[edge_ptr++] = {rank, rank_n_j, edges_weight, k_comm_cost, false, j};
      }
    }
  }
#endif

  return edges;
}

// C++ version of greedy_selection (no Python type conversions)
static std::vector<int> greedy_selection_cpp(int node_num, const std::vector<Edge>& edges, double threshold) {
  int num_edges = static_cast<int>(edges.size());
  if (num_edges == 0) {
    return std::vector<int>();
  }

  // Sort edges by cost-effectiveness (weight / cost)
  std::vector<EdgeSortEntry> sorted_edges;
  sorted_edges.resize(num_edges); // Pre-allocate for parallel access

  // Parallelize score calculation
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int j = 0; j < num_edges; ++j) {
    double score = edges[j].weight / std::max(static_cast<double>(edges[j].cost), 1e-6);
    sorted_edges[j] = {score, j, edges[j].u, edges[j].v, edges[j].cost};
  }

  // Sort by score in descending order using a Top-K + local sort strategy
  // to reduce the cost of fully sorting all edges.
  auto sort_comp = [](const EdgeSortEntry& a, const EdgeSortEntry& b) { return a > b; };

  // Heuristic: only fully sort a Top-K prefix that is most likely to be used.
  int topk = num_edges;
  if (num_edges > 2048) {
    int min_k = 2048;
    int ratio_k = static_cast<int>(static_cast<double>(num_edges) * 0.5);
    topk = std::min(num_edges, std::max(min_k, ratio_k));
  }

  if (topk == num_edges) {
    // Fallback: full sort
#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
    std::sort(std::execution::par_unseq, sorted_edges.begin(), sorted_edges.end(), sort_comp);
#else
    std::sort(sorted_edges.begin(), sorted_edges.end(), sort_comp);
#endif
  } else {
    auto topk_end = sorted_edges.begin() + topk;
    std::nth_element(sorted_edges.begin(), topk_end, sorted_edges.end(), sort_comp);

#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
    std::sort(std::execution::par_unseq, sorted_edges.begin(), topk_end, sort_comp);
#else
    std::sort(sorted_edges.begin(), topk_end, sort_comp);
#endif
  }

  std::vector<int> selected_edges;
  selected_edges.reserve(num_edges / 4); // Heuristic
  std::vector<double> node_costs(node_num, 0.0);

  // Greedy loop with contiguous memory access (sorted_edges is compact)
  for (const auto& entry : sorted_edges) {
    int uj = entry.u;
    int vj = entry.v;
    int cj = entry.cost;

    // Check if adding this edge exceeds the threshold for either node
    if (node_costs[uj] + cj <= threshold && node_costs[vj] + cj <= threshold) {
      node_costs[uj] += cj;
      node_costs[vj] += cj;
      selected_edges.push_back(entry.index);
    }
  }

  return selected_edges;
}

// Separate selection logic from sorting for caching
static std::vector<int> greedy_selection_from_sorted(int node_num, const std::vector<EdgeSortEntry>& sorted_edges, double threshold) {
  int num_edges = static_cast<int>(sorted_edges.size());
  if (num_edges == 0) {
    return std::vector<int>();
  }

  std::vector<int> selected_edges;
  selected_edges.reserve(num_edges / 4);
  std::vector<double> node_costs(node_num, 0.0);

  for (const auto& entry : sorted_edges) {
    int uj = entry.u;
    int vj = entry.v;
    int cj = entry.cost;

    if (node_costs[uj] + cj <= threshold && node_costs[vj] + cj <= threshold) {
      node_costs[uj] += cj;
      node_costs[vj] += cj;
      selected_edges.push_back(entry.index);
    }
  }

  return selected_edges;
}

// C++ version of greedy_max_flow (no Python type conversions)
static std::pair<bool, std::vector<Assignment>> greedy_max_flow_cpp(
    int cp_size,
    const std::vector<Edge>& simplex_edges,
    const std::vector<int>& simplex_selected_edges,
    const std::vector<std::tuple<int, int, long long>>& sparse_area_vec,
    const std::vector<int>& rank_m_vec,
    const std::vector<int>& rank_n_vec,
    const std::vector<int>& usp_choices_vec,
    double area_avg,
    double unbalance_rate,
    int rank = -1,
    bool debug_print = false) {
  int m = static_cast<int>(rank_m_vec.size());
  int n = static_cast<int>(rank_n_vec.size());
  double max_allowed_load = area_avg * std::max(unbalance_rate, 1.0);
  int num_areas = static_cast<int>(sparse_area_vec.size());

  // 1. Precompute the index mask for each rank that allows processing
  using Mask = std::vector<bool>;
  std::vector<Mask> qo_masks(m, Mask(cp_size, false));
  std::vector<Mask> kv_masks(n, Mask(cp_size, false));

  for (int i = 0; i < m; ++i) {
    int r = rank_m_vec[i];
    qo_masks[i][r] = true;
  }
  for (int j = 0; j < n; ++j) {
    int r = rank_n_vec[j];
    kv_masks[j][r] = true;
  }

  int num_selected = static_cast<int>(simplex_selected_edges.size());
  for (int sel_idx = 0; sel_idx < num_selected; ++sel_idx) {
    int idx = simplex_selected_edges[sel_idx];
    const Edge& edge = simplex_edges[idx];
    int uj = edge.u;
    int vj = edge.v;
    bool is_qo = edge.is_qo;
    int tag = edge.tag;

    if (is_qo) {
      if (uj == rank_m_vec[tag]) {
        qo_masks[tag][vj] = true;
      }
    } else {
      if (vj == rank_n_vec[tag]) {
        kv_masks[tag][uj] = true;
      }
    }
  }

  // 2. Prepare task data and calculate "degree" (number of allowed ranks)
  struct Task {
    int idx;
    int i;
    int j;
    long long area;
    Mask mask;
    int degree;
  };

  std::vector<Task> tasks;
  tasks.resize(num_areas); // Pre-allocate for parallel access

  // Use atomic flag to track errors (boundary check or degree==0)
  std::atomic<bool> has_error(false);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int idx = 0; idx < num_areas; ++idx) {
    int i = std::get<0>(sparse_area_vec[idx]);
    int j = std::get<1>(sparse_area_vec[idx]);

    // Compute mask and degree
    Mask mask(cp_size, false);
    int degree = 0;
    for (int r = 0; r < cp_size; ++r) {
      if (qo_masks[i][r] && kv_masks[j][r]) {
        mask[r] = true;
        ++degree;
      }
    }

    // Check if degree is zero
    if (degree == 0) {
      has_error.store(true, std::memory_order_relaxed);
      // Mark as invalid task (degree = -1)
      tasks[idx] = {idx, i, j, std::get<2>(sparse_area_vec[idx]), mask, -1};
      continue;
    }

    // Store valid task result directly at index idx
    tasks[idx] = {idx, i, j, std::get<2>(sparse_area_vec[idx]), mask, degree};
  }

  // Check for errors after parallel loop
  if (has_error.load(std::memory_order_relaxed)) {
    std::vector<Assignment> sparse_res(num_areas);
    for (int k = 0; k < num_areas; ++k) {
      sparse_res[k] = {std::get<0>(sparse_area_vec[k]), std::get<1>(sparse_area_vec[k]), -1};
    }
    return std::make_pair(false, sparse_res);
  }

  // If no errors, all tasks should be valid (degree > 0)
  // No need to filter, tasks are already correctly sized and populated

  // 3. Sort by degree in ascending order
  std::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
    if (a.degree != b.degree) {
      return a.degree < b.degree;
    }
    if (a.area != b.area) {
      return a.area > b.area;
    }
    return false;
  });

  // 4. Greedy assignment
  std::vector<long long> rank_loads(cp_size, 0);
  std::vector<Assignment> sparse_res(num_areas);
  // Initialize with placeholders
  for (int k = 0; k < num_areas; ++k) {
    sparse_res[k] = {0, 0, -1};
  }

  for (const auto& task : tasks) {
    const auto& mask = task.mask;
    long long area = task.area;
    int i = task.i;
    int j = task.j;

    // check local rank
    int local_rank = -1;
    if (rank_m_vec[i] == rank_n_vec[j]) {
      local_rank = rank_m_vec[i];
    }

    int assign_rank;
    if (local_rank != -1) {
      assign_rank = local_rank;
    } else {
      int best_rank = -1;
      // usp greedy:
      int usp_rank = usp_choices_vec[i];
      if (usp_rank >= 0 && usp_rank < cp_size && mask[usp_rank]) {
        if (rank_loads[usp_rank] < area_avg) {
          best_rank = usp_rank;
        }
      }
      // ring greedy:
      int ring_rank = rank_m_vec[i];
      if (best_rank == -1 && ring_rank >= 0 && ring_rank < cp_size && mask[ring_rank]) {
        if (rank_loads[ring_rank] < area_avg) {
          best_rank = ring_rank;
        }
      }
      // regular greedy
      if (best_rank == -1) {
        double min_load = std::numeric_limits<double>::infinity();
        for (int r = 0; r < cp_size; ++r) {
          if (mask[r]) {
            if (rank_loads[r] < min_load) {
              min_load = rank_loads[r];
              best_rank = r;
            }
          }
        }
      }
      assign_rank = best_rank;
    }

    rank_loads[assign_rank] += area;
    sparse_res[task.idx] = {i, j, assign_rank};
  }

  // 4.5. Adjustment Step: Local Refinement
  for (int iter = 0; iter < 5; ++iter) {
    for (const auto& task : tasks) {
      int idx = task.idx;
      Assignment& curr_res = sparse_res[idx];
      int curr_rank = curr_res.rank;
      long long area = task.area;
      const auto& mask = task.mask;

      long long curr_load = rank_loads[curr_rank];

      if (curr_load < max_allowed_load) {
        continue;
      }

      int best_target_rank = -1;
      double min_target_load = curr_load;

      for (int r = 0; r < cp_size; ++r) {
        if (mask[r] && r != curr_rank) {
          double new_potential_load = rank_loads[r] + area;
          if (new_potential_load < min_target_load) {
            min_target_load = new_potential_load;
            best_target_rank = r;
          }
        }
      }

      if (best_target_rank != -1) {
        rank_loads[curr_rank] -= area;
        rank_loads[best_target_rank] += area;
        sparse_res[idx].rank = best_target_rank;
      }
    }
  }

  // 5. Check if the load upper limit constraint is satisfied
  bool is_feasible = true;
  for (int r = 0; r < cp_size; ++r) {
    if (rank_loads[r] > max_allowed_load) {
      is_feasible = false;
      break;
    }
  }

  return std::make_pair(is_feasible, sparse_res);
}

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
    int num_heads_kv) {
  int m = static_cast<int>(rank_m.size());
  int n = static_cast<int>(rank_n.size());

  // Pre-extract Python lists to C++ vectors to avoid repeated cast operations
  std::vector<int> rank_m_vec(m);
  std::vector<int> rank_n_vec(n);
  std::vector<int> comm_len_m_vec(m);
  std::vector<int> comm_len_n_vec(n);

  for (int i = 0; i < m; ++i) {
    rank_m_vec[i] = rank_m[i].cast<int>();
    comm_len_m_vec[i] = comm_len_m[i].cast<int>();
  }
  for (int j = 0; j < n; ++j) {
    rank_n_vec[j] = rank_n[j].cast<int>();
    comm_len_n_vec[j] = comm_len_n[j].cast<int>();
  }

  // Pre-extract sparse_solver_map to avoid repeated Python object access
  int num_areas = static_cast<int>(sparse_area_map.size());
  std::vector<int> solver_assigned_ranks(num_areas, -1);
  for (int idx = 0; idx < num_areas; ++idx) {
    pybind11::tuple solver_entry = sparse_solver_map[idx].cast<pybind11::tuple>();
    solver_assigned_ranks[idx] = solver_entry[2].cast<int>();
  }

  // Pre-compute constant values
  double area_avg_const = area_avg / cp_size * 0.05;
  double unsolved_weight_factor = 0.95 / cp_size;

  // Group sparse area map by row and column for fast lookup
  // Each entry in row_areas/col_areas is (other_idx, area, global_idx)
  std::vector<std::vector<std::tuple<int, long long, int>>> row_areas(m);
  std::vector<std::vector<std::tuple<int, long long, int>>> col_areas(n);
  std::vector<long long> row_sums(m, 0);
  std::vector<long long> col_sums(n, 0);

  // Process sparse_area_map: (i, j, area)
  for (int idx = 0; idx < num_areas; ++idx) {
    pybind11::tuple area_entry = sparse_area_map[idx].cast<pybind11::tuple>();
    int i = area_entry[0].cast<int>();
    int j = area_entry[1].cast<int>();
    long long area = area_entry[2].cast<long long>();

    row_areas[i].emplace_back(j, area, idx);
    col_areas[j].emplace_back(i, area, idx);
    row_sums[i] += area;
    col_sums[j] += area;
  }

  // Pre-calculate total number of edges:
  // Q edges: m rows * (cp_size - 1) edges per row (excluding self)
  // KV edges: n cols * (cp_size - 1) edges per col (excluding self)
  int per_node_edges = cp_size - 1;
  std::size_t total_edges = static_cast<std::size_t>(m + n) * static_cast<std::size_t>(per_node_edges);
  pybind11::list edges(total_edges);
  std::size_t edge_idx = 0;

  // Process Q edges (for each row i)
  std::vector<double> q_comm_weight_list(cp_size);
  for (int i = 0; i < m; ++i) {
    int q_comm_cost = comm_len_m_vec[i] * num_heads_q;
    std::fill(q_comm_weight_list.begin(), q_comm_weight_list.end(), 0.0);
    double comm_weight_solved_sum = 0.0;

    for (const auto& entry : row_areas[i]) {
      double comm_weight = std::get<1>(entry);
      int global_idx = std::get<2>(entry);
      int assigned_rank = solver_assigned_ranks[global_idx];

      if (assigned_rank != -1) {
        // local q o communication: rank_m[i] -> solver_map[global_idx]
        q_comm_weight_list[assigned_rank] += comm_weight;
        comm_weight_solved_sum += comm_weight;
      }
    }

    double comm_weight_sum = row_sums[i];
    double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
    int rank_m_i = rank_m_vec[i];
    double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

    for (int rank = 0; rank < cp_size; ++rank) {
      if (rank != rank_m_i) {
        // simplex weight function
        double edges_weight = (q_comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
        edges[edge_idx++] = pybind11::make_tuple(rank_m_i, rank, edges_weight, q_comm_cost, true, i);
      }
    }
  }

  // Process KV edges (for each column j)
  std::vector<double> kv_comm_weight_list(cp_size);
  for (int j = 0; j < n; ++j) {
    int k_comm_cost = comm_len_n_vec[j] * num_heads_kv;
    std::fill(kv_comm_weight_list.begin(), kv_comm_weight_list.end(), 0.0);
    double comm_weight_solved_sum = 0.0;

    for (const auto& entry : col_areas[j]) {
      double comm_weight = std::get<1>(entry);
      int global_idx = std::get<2>(entry);
      int assigned_rank = solver_assigned_ranks[global_idx];

      if (assigned_rank != -1) {
        // local k v communication: rank_n[j] -> solver_map[global_idx]
        kv_comm_weight_list[assigned_rank] += comm_weight;
        comm_weight_solved_sum += comm_weight;
      }
    }

    double comm_weight_sum = col_sums[j];
    double comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum;
    int rank_n_j = rank_n_vec[j];
    double unsolved_contribution = comm_weight_unsolved_sum * unsolved_weight_factor;

    for (int rank = 0; rank < cp_size; ++rank) {
      if (rank != rank_n_j) {
        // simplex weight function
        double edges_weight = (kv_comm_weight_list[rank] + unsolved_contribution) * 0.95 + area_avg_const;
        edges[edge_idx++] = pybind11::make_tuple(rank, rank_n_j, edges_weight, k_comm_cost, false, j);
      }
    }
  }

  return edges;
}

pybind11::list greedy_selection(int node_num, const pybind11::list& edges, double threshold) {
  int num_edges = static_cast<int>(edges.size());
  if (num_edges == 0) {
    return pybind11::list();
  }

  // Pre-extract edge data to C++ vectors for better performance
  std::vector<int> uj_vec(num_edges);
  std::vector<int> vj_vec(num_edges);
  std::vector<double> wj_vec(num_edges);
  std::vector<int> cj_vec(num_edges);

  for (int j = 0; j < num_edges; ++j) {
    pybind11::tuple edge = edges[j].cast<pybind11::tuple>();
    uj_vec[j] = edge[0].cast<int>();
    vj_vec[j] = edge[1].cast<int>();
    wj_vec[j] = edge[2].cast<double>();
    cj_vec[j] = edge[3].cast<int>();
  }

  // Sort edges by cost-effectiveness (weight / cost)
  std::vector<std::pair<double, int>> indexed_edges;
  indexed_edges.reserve(num_edges);
  for (int j = 0; j < num_edges; ++j) {
    double score = wj_vec[j] / std::max(static_cast<double>(cj_vec[j]), 1e-6);
    indexed_edges.emplace_back(score, j);
  }

  // Sort by score in descending order
  // When scores are equal, sort by index in ascending order to ensure deterministic behavior
  std::sort(indexed_edges.begin(), indexed_edges.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second; // Stable tie-breaker: smaller index first
  });

  pybind11::list selected_edges;
  std::vector<double> node_costs(node_num, 0.0);

  for (const auto& pair : indexed_edges) {
    int j = pair.second;
    int uj = uj_vec[j];
    int vj = vj_vec[j];
    int cj = cj_vec[j];

    // Check if adding this edge exceeds the threshold for either node
    if (node_costs[uj] + cj <= threshold && node_costs[vj] + cj <= threshold) {
      node_costs[uj] += cj;
      node_costs[vj] += cj;
      selected_edges.append(j);
    }
  }

  return selected_edges;
}

pybind11::tuple greedy_max_flow(
    int cp_size,
    const pybind11::list& simplex_edges,
    const pybind11::list& simplex_selected_edges,
    const pybind11::list& sparse_area_map,
    const pybind11::list& rank_m,
    const pybind11::list& rank_n,
    const pybind11::list& usp_choices,
    double area_avg,
    double unbalance_rate) {
  int m = static_cast<int>(rank_m.size());
  int n = static_cast<int>(rank_n.size());
  double max_allowed_load = area_avg * std::max(unbalance_rate, 1.0);
  int num_areas = static_cast<int>(sparse_area_map.size());

  // Pre-extract Python lists to C++ vectors
  std::vector<int> rank_m_vec(m);
  std::vector<int> rank_n_vec(n);
  std::vector<int> usp_choices_vec(m);

  for (int i = 0; i < m; ++i) {
    rank_m_vec[i] = rank_m[i].cast<int>();
    usp_choices_vec[i] = usp_choices[i].cast<int>();
  }
  for (int j = 0; j < n; ++j) {
    rank_n_vec[j] = rank_n[j].cast<int>();
  }

  // Pre-extract simplex_edges
  int num_simplex_edges = static_cast<int>(simplex_edges.size());
  std::vector<int> edge_uj(num_simplex_edges);
  std::vector<int> edge_vj(num_simplex_edges);
  std::vector<bool> edge_is_qo(num_simplex_edges);
  std::vector<int> edge_tag(num_simplex_edges);

  for (int idx = 0; idx < num_simplex_edges; ++idx) {
    pybind11::tuple edge = simplex_edges[idx].cast<pybind11::tuple>();
    edge_uj[idx] = edge[0].cast<int>();
    edge_vj[idx] = edge[1].cast<int>();
    edge_is_qo[idx] = edge[4].cast<bool>();
    edge_tag[idx] = edge[5].cast<int>();
  }

  // Pre-extract sparse_area_map
  std::vector<int> area_i(num_areas);
  std::vector<int> area_j(num_areas);
  std::vector<long long> area_value(num_areas);

  for (int idx = 0; idx < num_areas; ++idx) {
    pybind11::tuple area_entry = sparse_area_map[idx].cast<pybind11::tuple>();
    area_i[idx] = area_entry[0].cast<int>();
    area_j[idx] = area_entry[1].cast<int>();
    area_value[idx] = area_entry[2].cast<long long>();
  }

  // 1. Precompute the index mask for each rank that allows processing
  using Mask = std::vector<bool>;
  std::vector<Mask> qo_masks(m, Mask(cp_size, false));
  std::vector<Mask> kv_masks(n, Mask(cp_size, false));

  for (int i = 0; i < m; ++i) {
    int r = rank_m_vec[i];
    qo_masks[i][r] = true;
  }
  for (int j = 0; j < n; ++j) {
    int r = rank_n_vec[j];
    kv_masks[j][r] = true;
  }

  int num_selected = static_cast<int>(simplex_selected_edges.size());
  for (int sel_idx = 0; sel_idx < num_selected; ++sel_idx) {
    int idx = simplex_selected_edges[sel_idx].cast<int>();
    int uj = edge_uj[idx];
    int vj = edge_vj[idx];
    bool is_qo = edge_is_qo[idx];
    int tag = edge_tag[idx];

    if (is_qo) {
      if (uj == rank_m_vec[tag]) {
        qo_masks[tag][vj] = true;
      }
    } else {
      if (vj == rank_n_vec[tag]) {
        kv_masks[tag][uj] = true;
      }
    }
  }

  // 2. Prepare task data and calculate "degree" (number of allowed ranks)
  struct Task {
    int idx;
    int i;
    int j;
    long long area;
    Mask mask;
    int degree;
  };

  std::vector<Task> tasks;
  tasks.reserve(num_areas);

  for (int idx = 0; idx < num_areas; ++idx) {
    int i = area_i[idx];
    int j = area_j[idx];

    Mask mask(cp_size, false);
    int degree = 0;
    for (int r = 0; r < cp_size; ++r) {
      if (qo_masks[i][r] && kv_masks[j][r]) {
        mask[r] = true;
        ++degree;
      }
    }
    if (degree == 0) {
      pybind11::list sparse_res;
      for (int k = 0; k < num_areas; ++k) {
        sparse_res.append(pybind11::make_tuple(area_i[k], area_j[k], -1));
      }
      return pybind11::make_tuple(false, sparse_res);
    }
    tasks.push_back({idx, i, j, area_value[idx], mask, degree});
  }

  // 3. Sort by degree in ascending order
  std::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
    if (a.degree != b.degree) {
      return a.degree < b.degree;
    }
    if (a.area != b.area) {
      return a.area > b.area;
    }
    return false;
  });

  // 4. Greedy assignment
  std::vector<double> rank_loads(cp_size, 0.0);
  pybind11::list sparse_res;
  for (int k = 0; k < num_areas; ++k) {
    sparse_res.append(pybind11::make_tuple(0, 0, -1));
  }

  for (const auto& task : tasks) {
    const auto& mask = task.mask;
    long long area = task.area;
    int i = task.i;
    int j = task.j;

    int local_rank = -1;
    if (rank_m_vec[i] == rank_n_vec[j]) {
      local_rank = rank_m_vec[i];
    }

    int assign_rank;
    if (local_rank != -1) {
      assign_rank = local_rank;
    } else {
      int best_rank = -1;
      int usp_rank = usp_choices_vec[i];
      if (usp_rank >= 0 && usp_rank < cp_size && mask[usp_rank]) {
        if (rank_loads[usp_rank] < area_avg) {
          best_rank = usp_rank;
        }
      }
      int ring_rank = rank_m_vec[i];
      if (best_rank == -1 && ring_rank >= 0 && ring_rank < cp_size && mask[ring_rank]) {
        if (rank_loads[ring_rank] < area_avg) {
          best_rank = ring_rank;
        }
      }
      if (best_rank == -1) {
        double min_load = std::numeric_limits<double>::infinity();
        for (int r = 0; r < cp_size; ++r) {
          if (mask[r]) {
            if (rank_loads[r] < min_load) {
              min_load = rank_loads[r];
              best_rank = r;
            }
          }
        }
      }
      assign_rank = best_rank;
    }

    rank_loads[assign_rank] += area;
    sparse_res[task.idx] = pybind11::make_tuple(i, j, assign_rank);
  }

  // 4.5. Adjustment Step: Local Refinement
  for (int iter = 0; iter < 5; ++iter) {
    for (const auto& task : tasks) {
      int idx = task.idx;
      pybind11::tuple curr_res = sparse_res[idx].cast<pybind11::tuple>();
      int curr_i = curr_res[0].cast<int>();
      int curr_j = curr_res[1].cast<int>();
      int curr_rank = curr_res[2].cast<int>();
      long long area = task.area;
      const auto& mask = task.mask;

      double curr_load = rank_loads[curr_rank];

      if (curr_load < max_allowed_load) {
        continue;
      }

      int best_target_rank = -1;
      double min_target_load = curr_load;

      for (int r = 0; r < cp_size; ++r) {
        if (mask[r] && r != curr_rank) {
          double new_potential_load = rank_loads[r] + area;
          if (new_potential_load < min_target_load) {
            min_target_load = new_potential_load;
            best_target_rank = r;
          }
        }
      }

      if (best_target_rank != -1) {
        rank_loads[curr_rank] -= area;
        rank_loads[best_target_rank] += area;
        sparse_res[idx] = pybind11::make_tuple(curr_i, curr_j, best_target_rank);
      }
    }
  }

  // 5. Check if the load upper limit constraint is satisfied
  bool is_feasible = true;
  for (int r = 0; r < cp_size; ++r) {
    if (rank_loads[r] > max_allowed_load + 1e-6) {
      is_feasible = false;
      break;
    }
  }

  return pybind11::make_tuple(is_feasible, sparse_res);
}

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
    bool debug_print) {
  // Convert Python types to C++ types once at the beginning
  int m = static_cast<int>(rank_m.size());
  int n = static_cast<int>(rank_n.size());
  int num_areas = static_cast<int>(sparse_area_map.size());

  std::vector<int> rank_m_vec(m);
  std::vector<int> rank_n_vec(n);
  std::vector<int> comm_len_m_vec(m);
  std::vector<int> comm_len_n_vec(n);
  std::vector<int> usp_choices_vec(m);
  std::vector<std::tuple<int, int, long long>> sparse_area_vec(num_areas);

  for (int i = 0; i < m; ++i) {
    rank_m_vec[i] = rank_m[i].cast<int>();
    comm_len_m_vec[i] = comm_len_m[i].cast<int>();
    usp_choices_vec[i] = usp_choices[i].cast<int>();
  }
  for (int j = 0; j < n; ++j) {
    rank_n_vec[j] = rank_n[j].cast<int>();
    comm_len_n_vec[j] = comm_len_n[j].cast<int>();
  }
  for (int idx = 0; idx < num_areas; ++idx) {
    pybind11::tuple area_entry = sparse_area_map[idx].cast<pybind11::tuple>();
    int i = area_entry[0].cast<int>();
    int j = area_entry[1].cast<int>();
    long long area = area_entry[2].cast<long long>();
    sparse_area_vec[idx] = std::make_tuple(i, j, area);
  }

  // 1. Compute threshold (max comm cost per rank)
  double threshold = 0.0;
  int total_comm_m = 0;
  for (int i = 0; i < m; ++i) {
    total_comm_m += comm_len_m_vec[i];
  }
  int total_comm_n = 0;
  for (int j = 0; j < n; ++j) {
    total_comm_n += comm_len_n_vec[j];
  }
  threshold += static_cast<double>(num_heads_q) * static_cast<double>(total_comm_m) * 2.0;
  threshold += static_cast<double>(num_heads_kv) * static_cast<double>(total_comm_n) * 2.0;

  // 2. area_avg from sparse_area_map: (i, j, area)
  long long area_sum = 0;
  for (int idx = 0; idx < num_areas; ++idx) {
    area_sum += std::get<2>(sparse_area_vec[idx]);
  }
  double area_avg = (cp_size > 0) ? (static_cast<double>(area_sum) / static_cast<double>(cp_size)) : 0.0;

  // 3. Current solver state for generating simplex edges, initially all -1
  std::vector<int> solver_prev_ranks(num_areas, -1);

  int total_iters = 0;
  double t_greedy_solve = 0.0;
  double t_calc_edges = 0.0;
  double t_greedy_select = 0.0;

  std::vector<Assignment> solver_map;
  std::vector<Assignment> solver_try;
  std::vector<Assignment> best_map;
  bool has_best_map = false;

  // Binary search for threshold
  double low = 0.0;
  double high = threshold;
  double unbalance_rate = 1.10;
  int max_iters = 20;
  int max_attempts = 1;
  double eps = 1e-2;

  bool edges_dirty = true;
  std::vector<Edge> edges;
  std::vector<EdgeSortEntry> sorted_edges;

  auto t2 = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  for (int iter_idx = 0; iter_idx < max_iters; ++iter_idx) {
    double mid = (low + high) / 2.0;
    total_iters += 1;

    bool success = false;
    std::vector<int> selected_edges;
    solver_try.clear();
    std::vector<int> solver_state_ranks = solver_prev_ranks;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      // Regenerate simplex edges based on current solver state
      auto t_calc_start = std::chrono::high_resolution_clock::now();
      if (edges_dirty) {
        edges = calc_simplex_edges_cpp(
            cp_size, rank_m_vec, rank_n_vec, comm_len_m_vec, comm_len_n_vec, solver_state_ranks, sparse_area_vec, area_avg, num_heads_q, num_heads_kv);
      }
      auto t_calc_end = std::chrono::high_resolution_clock::now();
      t_calc_edges += std::chrono::duration<double>(t_calc_end - t_calc_start).count();

      auto t_greedy_start = std::chrono::high_resolution_clock::now();
      if (iter_idx == 0) {
        // In the first iteration, mid is very large, greedy will choose all edges.
        // We can skip it to save time.
        int num_edges = static_cast<int>(edges.size());
        selected_edges.clear();
        selected_edges.reserve(num_edges);
        for (int e = 0; e < num_edges; ++e) {
          selected_edges.push_back(e);
        }
      } else {
        if (edges_dirty) {
          int num_edges = static_cast<int>(edges.size());
          sorted_edges.resize(num_edges);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int j = 0; j < num_edges; ++j) {
            double score = edges[j].weight / std::max(static_cast<double>(edges[j].cost), 1e-6);
            sorted_edges[j] = {score, j, edges[j].u, edges[j].v, edges[j].cost};
          }

          auto sort_comp = [](const EdgeSortEntry& a, const EdgeSortEntry& b) { return a > b; };
          int topk = num_edges;
          if (num_edges > 2048) {
            int min_k = 2048;
            int ratio_k = static_cast<int>(static_cast<double>(num_edges) * 0.5);
            topk = std::min(num_edges, std::max(min_k, ratio_k));
          }

          if (topk == num_edges) {
#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
            std::sort(std::execution::par_unseq, sorted_edges.begin(), sorted_edges.end(), sort_comp);
#else
            std::sort(sorted_edges.begin(), sorted_edges.end(), sort_comp);
#endif
          } else {
            auto topk_end = sorted_edges.begin() + topk;
            std::nth_element(sorted_edges.begin(), topk_end, sorted_edges.end(), sort_comp);
#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
            std::sort(std::execution::par_unseq, sorted_edges.begin(), topk_end, sort_comp);
#else
            std::sort(sorted_edges.begin(), topk_end, sort_comp);
#endif
          }
          edges_dirty = false;
        }
        selected_edges = greedy_selection_from_sorted(cp_size, sorted_edges, mid);
      }
      auto t_greedy_end = std::chrono::high_resolution_clock::now();
      t_greedy_select += std::chrono::duration<double>(t_greedy_end - t_greedy_start).count();

      if (rank == 0 && debug_print) {
        int num_edges = static_cast<int>(edges.size());
        int num_selected = static_cast<int>(selected_edges.size());
        std::cout << "    - Greedy selected edges: " << num_selected << " / " << num_edges << std::endl;
      }

      auto t_start = std::chrono::high_resolution_clock::now();
      auto flow_res =
          greedy_max_flow_cpp(cp_size, edges, selected_edges, sparse_area_vec, rank_m_vec, rank_n_vec, usp_choices_vec, area_avg, unbalance_rate, rank, debug_print);
      auto t_end = std::chrono::high_resolution_clock::now();
      t_greedy_solve += std::chrono::duration<double>(t_end - t_start).count();

      success = flow_res.first;
      solver_try = flow_res.second;

      if (rank == 0 && debug_print) {
        std::cout << "    - Iter " << total_iters << ": mid=" << mid << ", success=" << (success ? "True" : "False") << std::endl;
      }

      if (success) {
        // Use this assignment as input for next attempt
        solver_state_ranks.clear();
        solver_state_ranks.reserve(num_areas);
        for (const auto& assignment : solver_try) {
          solver_state_ranks.push_back(assignment.rank);
        }
        edges_dirty = true;
        break;
      }
    }

    if (success) {
      best_map = solver_try;
      has_best_map = true;
      high = mid;
      solver_prev_ranks.clear();
      solver_prev_ranks.reserve(num_areas);
      for (const auto& assignment : solver_try) {
        solver_prev_ranks.push_back(assignment.rank);
      }
      edges_dirty = true;
    } else {
      low = mid;
    }

    if (high - low <= eps * high && low > 0.0) {
      break;
    }
  }

  double t_iter = 0.0;
  if (rank == 0) {
    auto t_now = std::chrono::high_resolution_clock::now();
    t_iter = std::chrono::duration<double>(t_now - t2).count();
  }

  auto t3 = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  if (has_best_map) {
    solver_map = best_map;
  } else if (solver_try.size() > 0) {
    solver_map = solver_try;
  } else {
    // Convert solver_prev_ranks to Assignment vector
    solver_map.clear();
    solver_map.reserve(num_areas);
    for (int idx = 0; idx < num_areas; ++idx) {
      int i = std::get<0>(sparse_area_vec[idx]);
      int j = std::get<1>(sparse_area_vec[idx]);
      solver_map.push_back({i, j, solver_prev_ranks[idx]});
    }
  }
  bool success = has_best_map;

  double t_post = 0.0;
  if (rank == 0) {
    auto t_now = std::chrono::high_resolution_clock::now();
    t_post = std::chrono::duration<double>(t_now - t3).count();
  }

  if (!success) {
    if (rank == 0 && debug_print) {
      std::cout << "[BinaryGreedyParallelDynamicAttnAlgorithm] network flow failed, "
                << "fallback to Q owner assignment" << std::endl;
    }
    // fallback to Q owner assignment
    for (int idx = 0; idx < num_areas; ++idx) {
      if (solver_map[idx].rank == -1) {
        int i = solver_map[idx].i;
        int q_owner = rank_m_vec[i];
        solver_map[idx].rank = q_owner;
      }
    }
  }

  // Convert C++ result back to Python list
  pybind11::list solver_map_py;
  for (const auto& assignment : solver_map) {
    solver_map_py.append(pybind11::make_tuple(assignment.i, assignment.j, assignment.rank));
  }

  if (rank == 0 && debug_print) {
    std::cout << "    - Iteration:  " << t_iter << "s (iters: " << total_iters << ")" << std::endl;
    std::cout << "        - Calc Edges: " << t_calc_edges << "s" << std::endl;
    std::cout << "        - Greedy Select: " << t_greedy_select << "s" << std::endl;
    std::cout << "        - Greedy Solve: " << t_greedy_solve << "s" << std::endl;
    std::cout << "    - Post-process: " << t_post << "s" << std::endl;
  }

  return solver_map_py;
}

// Helper function for recursive split matching Python logic
static void split_grid_recursive(
    AttnRectangles current_rects,
    int q_start,
    int q_end,
    int k_start,
    int k_end,
    bool prefer_q,
    const std::vector<std::pair<AttnRange, int>>& indexed_host_ranges_q,
    const std::vector<std::pair<AttnRange, int>>& indexed_host_ranges_k,
    std::vector<std::tuple<int, int, AttnRectangles>>& results) {
  if (current_rects.is_empty()) {
    return;
  }

  int nq = q_end - q_start;
  int nk = k_end - k_start;

  if (nq == 1 && nk == 1) {
    results.emplace_back(q_start, k_start, std::move(current_rects));
    return;
  }

  // Decide split axis: alternate unless one dimension is exhausted
  bool split_q = prefer_q;
  if (nq <= 1) {
    split_q = false;
  } else if (nk <= 1) {
    split_q = true;
  }

  if (split_q) {
    int mid = nq / 2;
    int mid_pos = indexed_host_ranges_q[q_start + mid].first.start;
    auto [left_rects, right_rects] = current_rects.cut_q(mid_pos);
    split_grid_recursive(std::move(left_rects), q_start, q_start + mid, k_start, k_end, !prefer_q, indexed_host_ranges_q, indexed_host_ranges_k, results);
    split_grid_recursive(std::move(right_rects), q_start + mid, q_end, k_start, k_end, !prefer_q, indexed_host_ranges_q, indexed_host_ranges_k, results);
  } else {
    int mid = nk / 2;
    int mid_pos = indexed_host_ranges_k[k_start + mid].first.start;
    auto [left_rects, right_rects] = current_rects.cut_k(mid_pos);
    split_grid_recursive(std::move(left_rects), q_start, q_end, k_start, k_start + mid, !prefer_q, indexed_host_ranges_q, indexed_host_ranges_k, results);
    split_grid_recursive(std::move(right_rects), q_start, q_end, k_start + mid, k_end, !prefer_q, indexed_host_ranges_q, indexed_host_ranges_k, results);
  }
}

// Get grid rectangles using a KD-tree style alternating split strategy
pybind11::list get_grid_rects(AttnRectangles& rects, const pybind11::list& indexed_host_ranges_q_py, const pybind11::list& indexed_host_ranges_k_py) {
  auto convert_to_indexed = [](const pybind11::list& host_ranges_py) {
    std::vector<std::pair<AttnRange, int>> indexed;
    for (int idx = 0; idx < (int)host_ranges_py.size(); ++idx) {
      pybind11::tuple entry = host_ranges_py[idx].cast<pybind11::tuple>();
      indexed.emplace_back(entry[0].cast<AttnRange>(), entry[1].cast<int>());
    }
    return indexed;
  };

  auto indexed_host_ranges_q = convert_to_indexed(indexed_host_ranges_q_py);
  auto indexed_host_ranges_k = convert_to_indexed(indexed_host_ranges_k_py);

  std::vector<std::tuple<int, int, AttnRectangles>> results;
  split_grid_recursive(rects, 0, (int)indexed_host_ranges_q.size(), 0, (int)indexed_host_ranges_k.size(), true, indexed_host_ranges_q, indexed_host_ranges_k, results);

  pybind11::list py_results;
  for (auto& res : results) {
    py_results.append(pybind11::make_tuple(std::get<0>(res), std::get<1>(res), std::move(std::get<2>(res))));
  }
  return py_results;
}

// --- Helper functions for Python to C++ data conversion ---
static AttnRange python_to_cpp_range(pybind11::handle obj) {
  try {
    return obj.cast<AttnRange>();
  } catch (const pybind11::cast_error&) {
    // If not a C++ AttnRange, try if it's a sequence (tuple/list)
    if (pybind11::isinstance<pybind11::sequence>(obj)) {
      pybind11::sequence seq = obj.cast<pybind11::sequence>();
      if (seq.size() >= 2) {
        return AttnRange(seq[0].cast<int>(), seq[1].cast<int>());
      }
    }
    // Fallback ONLY for Python-side AttnRange objects that are not the C++ class
    // We use getattr with a catch to be safe
    try {
      int s = pybind11::getattr(obj, "start").cast<int>();
      int e = pybind11::getattr(obj, "end").cast<int>();
      return AttnRange(s, e);
    } catch (...) {
      throw pybind11::type_error("Cannot convert Python object to AttnRange");
    }
  }
}

static AttnRectangle python_to_cpp_rectangle(pybind11::handle obj) {
  try {
    return obj.cast<AttnRectangle>();
  } catch (const pybind11::cast_error&) {
    // Fallback for Python-side AttnRectangle objects
    try {
      return AttnRectangle(
          python_to_cpp_range(pybind11::getattr(obj, "q_range")),
          python_to_cpp_range(pybind11::getattr(obj, "k_range")),
          python_to_cpp_range(pybind11::getattr(obj, "d_range")));
    } catch (...) {
      throw pybind11::type_error("Cannot convert Python object to AttnRectangle");
    }
  }
}

void binary_greedy_parallel_solve(
    pybind11::object& rects_py,
    const pybind11::list& host_ranges_q_py,
    const pybind11::list& host_ranges_k_py,
    int num_heads_q,
    int num_heads_kv,
    int num_heads_group,
    pybind11::list& bucket_per_rank,
    int rank,
    bool debug_print) {
  auto t0 = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  // --- Preprocess Analysis ---
  double t_p1 = 0.0, t_p2 = 0.0, t_p3 = 0.0;

  // 1. Convert Python rects (AttnRectangles) to C++ AttnRectangles
  auto tp1_start = std::chrono::high_resolution_clock::now();
  AttnRectangles rects;
  try {
    rects = rects_py.cast<AttnRectangles>();
  } catch (const pybind11::cast_error&) {
    // Try to handle as a list of rects or an object with _rects
    pybind11::list py_rect_list;
    bool handled = false;
    try {
      if (pybind11::isinstance<pybind11::list>(rects_py)) {
        py_rect_list = rects_py.cast<pybind11::list>();
        handled = true;
      } else {
        // Use getattr with catch instead of hasattr to avoid recursion traps
        py_rect_list = pybind11::getattr(rects_py, "_rects").cast<pybind11::list>();
        handled = true;
      }
    } catch (...) {
    }

    if (handled) {
      for (auto handle : py_rect_list) {
        rects.append(python_to_cpp_rectangle(handle));
      }
    } else {
      throw pybind11::type_error("rects must be AttnRectangles or list of AttnRectangle");
    }
  }
  if (rank == 0)
    t_p1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tp1_start).count();

  // 2. Preprocess: Extract indexed_host_ranges and compute rank_m/n, comm_len_m/n
  auto tp2_start = std::chrono::high_resolution_clock::now();
  auto convert_to_indexed = [](const pybind11::list& host_ranges_py) {
    std::vector<std::pair<AttnRange, int>> indexed;
    int num_ranks = (int)host_ranges_py.size();
    for (int idx = 0; idx < num_ranks; ++idx) {
      pybind11::object ranges_obj = host_ranges_py[idx];
      // Use try-catch instead of isinstance to avoid infinite recursion
      try {
        const auto& cpp_ranges = ranges_obj.cast<const AttnRanges&>();
        for (const auto& r : cpp_ranges.get()) {
          indexed.emplace_back(r, idx);
        }
      } catch (const pybind11::cast_error&) {
        // Handle as Python AttnRanges or list
        try {
          pybind11::list py_range_list;
          if (pybind11::isinstance<pybind11::list>(ranges_obj)) {
            py_range_list = ranges_obj.cast<pybind11::list>();
          } else {
            py_range_list = pybind11::getattr(ranges_obj, "_ranges").cast<pybind11::list>();
          }
          for (auto handle : py_range_list) {
            indexed.emplace_back(python_to_cpp_range(handle), idx);
          }
        } catch (...) {
          throw pybind11::type_error("host_ranges element must be AttnRanges or list of AttnRange");
        }
      }
    }
    std::stable_sort(indexed.begin(), indexed.end(), [](const auto& a, const auto& b) { return a.first.start < b.first.start; });
    return indexed;
  };

  auto indexed_host_ranges_q = convert_to_indexed(host_ranges_q_py);
  auto indexed_host_ranges_k = convert_to_indexed(host_ranges_k_py);
  if (rank == 0)
    t_p2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tp2_start).count();

  // 3. Compute usp_choices
  auto tp3_start = std::chrono::high_resolution_clock::now();
  int m = (int)indexed_host_ranges_q.size();
  int n = (int)indexed_host_ranges_k.size();
  int cp_size = (int)bucket_per_rank.size();

  std::vector<int> rank_m(m);
  std::vector<int> comm_len_m(m);
  for (int i = 0; i < m; ++i) {
    rank_m[i] = indexed_host_ranges_q[i].second;
    comm_len_m[i] = indexed_host_ranges_q[i].first.seqlen();
  }

  std::vector<int> rank_n(n);
  std::vector<int> comm_len_n(n);
  for (int j = 0; j < n; ++j) {
    rank_n[j] = indexed_host_ranges_k[j].second;
    comm_len_n[j] = indexed_host_ranges_k[j].first.seqlen();
  }

  int intra_group_num = std::gcd(cp_size, num_heads_group);
  int num_ranges_per_group = (intra_group_num > 0) ? (m / intra_group_num) : 0;
  std::vector<int> usp_choices(m);
  for (int i = 0; i < m; ++i) {
    int group_idx = (num_ranges_per_group > 0) ? (i / num_ranges_per_group) : 0;
    int host_rank = rank_m[i];
    usp_choices[i] = (intra_group_num > 0) ? ((host_rank / intra_group_num) * intra_group_num + group_idx) : 0;
  }
  if (rank == 0)
    t_p3 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tp3_start).count();

  auto t_preprocess_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  // Pre-declare containers for computation results
  std::vector<std::tuple<int, int, AttnRectangles>> sparse_grid_rects;
  std::vector<Assignment> final_map;
  int num_areas = 0;

  // Timing variables used inside the GIL-released block
  double t_calc_edges = 0.0;
  double t_greedy_sel = 0.0;
  double t_sort_edges = 0.0;
  double t_select_edges = 0.0;
  double t_max_flow = 0.0;
  double t_loop_other = 0.0;
  auto t_grid_end = t_preprocess_end;
  auto t_bg_start = t_preprocess_end;
  auto t_bg_init_end = t_preprocess_end;
  auto t_loop_end = t_preprocess_end;
  auto t_binary_greedy_end = t_preprocess_end;

  // --- Release GIL for heavy computation ---
  {
    pybind11::gil_scoped_release release;

    // 4. Grid Func: split_grid_recursive
    sparse_grid_rects.reserve(m * n / 2); // Heuristic
    split_grid_recursive(rects, 0, m, 0, n, true, indexed_host_ranges_q, indexed_host_ranges_k, sparse_grid_rects);

    if (rank == 0)
      t_grid_end = std::chrono::high_resolution_clock::now();

    // 5. Binary Greedy logic
    if (rank == 0)
      t_bg_start = std::chrono::high_resolution_clock::now();

    num_areas = (int)sparse_grid_rects.size();
    std::vector<std::tuple<int, int, long long>> sparse_area_vec(num_areas);
    long long area_sum = 0;
    for (int idx = 0; idx < num_areas; ++idx) {
      int i = std::get<0>(sparse_grid_rects[idx]);
      int j = std::get<1>(sparse_grid_rects[idx]);
      long long area = std::get<2>(sparse_grid_rects[idx]).area();
      sparse_area_vec[idx] = std::make_tuple(i, j, area);
      area_sum += area;
    }
    double area_avg = (cp_size > 0) ? (static_cast<double>(area_sum) / (double)cp_size) : 0.0;

    double threshold = 0.0;
    long long total_comm_m = 0;
    for (int v : comm_len_m)
      total_comm_m += v;
    long long total_comm_n = 0;
    for (int v : comm_len_n)
      total_comm_n += v;
    threshold += (double)num_heads_q * (double)total_comm_m * 2.0;
    threshold += (double)num_heads_kv * (double)total_comm_n * 2.0;

    if (rank == 0)
      t_bg_init_end = std::chrono::high_resolution_clock::now();

    std::vector<int> solver_prev_ranks(num_areas, -1);
    std::vector<Assignment> best_map;
    std::vector<Assignment> solver_try;
    bool has_best_map = false;
    double low = 0.0, high = threshold;
    double unbalance_rate = 1.10;
    double eps = 1e-2;

    bool edges_dirty = true;
    std::vector<Edge> edges;
    std::vector<EdgeSortEntry> sorted_edges;

    for (int iter_idx = 0; iter_idx < 20; ++iter_idx) {
      auto t_iter_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

      double mid = (low + high) / 2.0;
      std::vector<int> solver_state_ranks = solver_prev_ranks;

      auto t_calc_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
      if (edges_dirty) {
        edges = calc_simplex_edges_cpp(cp_size, rank_m, rank_n, comm_len_m, comm_len_n, solver_state_ranks, sparse_area_vec, area_avg, num_heads_q, num_heads_kv);
      }
      auto t_calc_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

      std::vector<int> selected_edges;
      auto t_sel_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
      if (iter_idx == 0) {
        selected_edges.resize(edges.size());
        std::iota(selected_edges.begin(), selected_edges.end(), 0);
      } else {
        auto t_sort_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
        if (edges_dirty) {
          int num_edges = static_cast<int>(edges.size());
          sorted_edges.resize(num_edges);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int j = 0; j < num_edges; ++j) {
            double score = edges[j].weight / std::max(static_cast<double>(edges[j].cost), 1e-6);
            sorted_edges[j] = {score, j, edges[j].u, edges[j].v, edges[j].cost};
          }

          auto sort_comp = [](const EdgeSortEntry& a, const EdgeSortEntry& b) { return a > b; };
          int topk = num_edges;
          if (num_edges > 2048) {
            int min_k = 2048;
            int ratio_k = static_cast<int>(static_cast<double>(num_edges) * 0.5);
            topk = std::min(num_edges, std::max(min_k, ratio_k));
          }

          if (topk == num_edges) {
#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
            std::sort(std::execution::par_unseq, sorted_edges.begin(), sorted_edges.end(), sort_comp);
#else
            std::sort(sorted_edges.begin(), sorted_edges.end(), sort_comp);
#endif
          } else {
            auto topk_end = sorted_edges.begin() + topk;
            std::nth_element(sorted_edges.begin(), topk_end, sorted_edges.end(), sort_comp);
#if __cplusplus >= 201703L && defined(__cpp_lib_parallel_algorithm)
            std::sort(std::execution::par_unseq, sorted_edges.begin(), topk_end, sort_comp);
#else
            std::sort(sorted_edges.begin(), topk_end, sort_comp);
#endif
          }
          edges_dirty = false;
        }
        auto t_sort_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

        auto t_select_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
        selected_edges = greedy_selection_from_sorted(cp_size, sorted_edges, mid);
        auto t_select_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

        if (rank == 0) {
          t_sort_edges += std::chrono::duration<double>(t_sort_end - t_sort_start).count();
          t_select_edges += std::chrono::duration<double>(t_select_end - t_select_start).count();
        }
      }
      auto t_sel_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

      auto t_flow_start = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};
      auto flow_res = greedy_max_flow_cpp(cp_size, edges, selected_edges, sparse_area_vec, rank_m, rank_n, usp_choices, area_avg, unbalance_rate, rank, false);
      auto t_flow_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

      solver_try = std::move(flow_res.second);
      if (flow_res.first) {
        best_map = solver_try;
        has_best_map = true;
        high = mid;
        solver_prev_ranks.clear();
        solver_prev_ranks.reserve(num_areas);
        for (const auto& a : best_map)
          solver_prev_ranks.push_back(a.rank);
        edges_dirty = true; // Ranks changed, need to recompute edges and sort
      } else {
        low = mid;
        // edges_dirty remains false, no need to recompute edges and sort
      }

      if (rank == 0) {
        t_calc_edges += std::chrono::duration<double>(t_calc_end - t_calc_start).count();
        t_greedy_sel += std::chrono::duration<double>(t_sel_end - t_sel_start).count();
        t_max_flow += std::chrono::duration<double>(t_flow_end - t_flow_start).count();
        auto t_iter_end = std::chrono::high_resolution_clock::now();
        t_loop_other += std::chrono::duration<double>(t_iter_end - t_iter_start).count() -
            (std::chrono::duration<double>(t_calc_end - t_calc_start).count() + std::chrono::duration<double>(t_sel_end - t_sel_start).count() +
             std::chrono::duration<double>(t_flow_end - t_flow_start).count());
      }

      if (high - low <= eps * high && low > 0.0)
        break;
    }

    if (rank == 0)
      t_loop_end = std::chrono::high_resolution_clock::now();

    if (has_best_map) {
      final_map = std::move(best_map);
    } else if (!solver_try.empty()) {
      final_map = std::move(solver_try);
    } else {
      final_map.reserve(num_areas);
      for (int idx = 0; idx < num_areas; ++idx) {
        int i = std::get<0>(sparse_grid_rects[idx]);
        int j = std::get<1>(sparse_grid_rects[idx]);
        final_map.push_back({i, j, -1});
      }
    }

    // Fallback for unassigned tasks
    for (int idx = 0; idx < num_areas; ++idx) {
      if (final_map[idx].rank == -1) {
        final_map[idx].rank = rank_m[final_map[idx].i];
      }
    }

    if (rank == 0)
      t_binary_greedy_end = std::chrono::high_resolution_clock::now();
  }
  // --- GIL is re-acquired here ---

  // 6. Apply result to bucket_per_rank (Handle both C++ and Python types)
  pybind11::module_ py_range_mod;
  pybind11::module_ py_rect_mod;
  bool py_mods_loaded = false;

  int fast_path_count = 0;
  int fallback_path_count = 0;
  double t_fast_path = 0.0;
  double t_fallback_path = 0.0;

  // Optimization: Cache C++ bucket pointers to avoid expensive pybind11::cast in the loop
  int cp_size_actual = static_cast<int>(bucket_per_rank.size());
  std::vector<AttnRectangles*> cached_cpp_buckets(cp_size_actual, nullptr);
  for (int r = 0; r < cp_size_actual; ++r) {
    // Use try-catch instead of isinstance to avoid infinite recursion
    try {
      cached_cpp_buckets[r] = &bucket_per_rank[r].cast<AttnRectangles&>();
    } catch (const pybind11::cast_error&) {
      cached_cpp_buckets[r] = nullptr;
    }
  }

  for (int idx = 0; idx < num_areas; ++idx) {
    int assigned_rank = final_map[idx].rank;
    if (assigned_rank != -1 && assigned_rank < cp_size_actual) {
      AttnRectangles& rect_to_add = std::get<2>(sparse_grid_rects[idx]);

      if (cached_cpp_buckets[assigned_rank]) {
        // Ultra-fast path: Direct pointer access to C++ container
        auto t_start = std::chrono::high_resolution_clock::now();
        AttnRectangles* cpp_bucket = cached_cpp_buckets[assigned_rank];
        for (size_t r_idx = 0; r_idx < rect_to_add.size(); ++r_idx) {
          cpp_bucket->append(rect_to_add.at(r_idx));
        }
        if (rank == 0)
          t_fast_path += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
        fast_path_count++;
      } else {
        // Fallback path: Python object manipulation
        auto t_start = std::chrono::high_resolution_clock::now();
        pybind11::object bucket = bucket_per_rank[assigned_rank];
        if (!py_mods_loaded) {
          py_range_mod = pybind11::module_::import("magi_attention.common.range");
          py_rect_mod = pybind11::module_::import("magi_attention.common.rectangle");
          py_mods_loaded = true;
        }

        for (size_t r_idx = 0; r_idx < rect_to_add.size(); ++r_idx) {
          const auto& r = rect_to_add.at(r_idx);
          pybind11::object py_q = py_range_mod.attr("AttnRange")(r.get_q_range().start, r.get_q_range().end);
          pybind11::object py_k = py_range_mod.attr("AttnRange")(r.get_k_range().start, r.get_k_range().end);
          pybind11::object py_d = py_range_mod.attr("AttnRange")(r.get_d_range().start, r.get_d_range().end);
          pybind11::object py_rect = py_rect_mod.attr("AttnRectangle")(py_q, py_k, py_d);
          bucket.attr("append")(py_rect);
        }
        if (rank == 0)
          t_fallback_path += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_start).count();
        fallback_path_count++;
      }
    }
  }

  auto t_end = (rank == 0) ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

  if (rank == 0 && debug_print) {
    double t_preprocess = std::chrono::duration<double>(t_preprocess_end - t0).count();
    double t_grid = std::chrono::duration<double>(t_grid_end - t_preprocess_end).count();
    double t_bg_total = std::chrono::duration<double>(t_binary_greedy_end - t_grid_end).count();
    double t_return = std::chrono::duration<double>(t_end - t_binary_greedy_end).count();
    double total = std::chrono::duration<double>(t_end - t0).count();

    std::cout << "    - Preprocess (Total): " << t_preprocess << "s" << std::endl;
    std::cout << "        * P1 (Rects):     " << t_p1 << "s" << std::endl;
    std::cout << "        * P2 (HostRanges): " << t_p2 << "s" << std::endl;
    std::cout << "        * P3 (RankChoices): " << t_p3 << "s" << std::endl;
    std::cout << "    - Grid Func:  " << t_grid << "s" << std::endl;
    std::cout << "    - Binary Greedy (Total): " << t_bg_total << "s" << std::endl;
    std::cout << "        * BG Init:      " << std::chrono::duration<double>(t_bg_init_end - t_bg_start).count() << "s" << std::endl;
    std::cout << "        * Calc Edges:   " << t_calc_edges << "s" << std::endl;
    std::cout << "        * Greedy Sel:   " << t_greedy_sel << "s" << std::endl;
    std::cout << "            - Sort Edges: " << t_sort_edges << "s" << std::endl;
    std::cout << "            - Select Edges: " << t_select_edges << "s" << std::endl;
    std::cout << "        * Max Flow:     " << t_max_flow << "s" << std::endl;
    std::cout << "        * Loop Other:   " << t_loop_other << "s" << std::endl;
    std::cout << "        * BG Post:      " << std::chrono::duration<double>(t_binary_greedy_end - t_loop_end).count() << "s" << std::endl;
    std::cout << "    - Return (Total):     " << t_return << "s" << std::endl;
    std::cout << "        * Fast Path (Count: " << fast_path_count << "): " << t_fast_path << "s" << std::endl;
    std::cout << "        * Fallback Path (Count: " << fallback_path_count << "): " << t_fallback_path << "s" << std::endl;
    std::cout << "[BinaryGreedyParallelDynamicAttnAlgorithm] solve elapsed time: " << total << "s" << std::endl;
  }
}

} // namespace magi_attn_ext
