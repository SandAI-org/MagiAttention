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
#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
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

struct CompactEdge {
  double score;
  int index;

  // For descending order sort
  bool operator>(const CompactEdge& other) const {
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
  // Process sequentially to ensure row_areas and col_areas are deterministic
  for (int idx = 0; idx < num_areas; ++idx) {
    int i = std::get<0>(sparse_area_vec[idx]);
    int j = std::get<1>(sparse_area_vec[idx]);
    long long area = std::get<2>(sparse_area_vec[idx]);

    int r_idx = row_pos[i]++;
    int c_idx = col_pos[j]++;

    row_areas[i][r_idx] = std::make_tuple(j, area, idx);
    col_areas[j][c_idx] = std::make_tuple(i, area, idx);
    row_sums[i] += area;
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

// Separate selection logic from sorting for caching
static std::vector<int> greedy_selection_from_sorted(int node_num, const std::vector<Edge>& edges, const std::vector<CompactEdge>& sorted_edges, double threshold) {
  int num_edges = static_cast<int>(sorted_edges.size());
  if (num_edges == 0) {
    return std::vector<int>();
  }

  std::vector<int> selected_edges;
  selected_edges.reserve(num_edges / 4);
  std::vector<double> node_costs(node_num, 0.0);

  for (const auto& entry : sorted_edges) {
    const auto& edge = edges[entry.index];
    int uj = edge.u;
    int vj = edge.v;
    int cj = edge.cost;

    if (node_costs[uj] + cj <= threshold && node_costs[vj] + cj <= threshold) {
      node_costs[uj] += cj;
      node_costs[vj] += cj;
      selected_edges.push_back(entry.index);
    }
  }

  return selected_edges;
}

// Helper for efficient bitset operations with dynamic size
struct FastMask {
  uint64_t* bits = nullptr;
  int num_words = 0;

  FastMask() = default;
  FastMask(uint64_t* p, int nw) : bits(p), num_words(nw) {}

  inline void set(int r) {
    bits[r >> 6] |= (1ULL << (r & 63));
  }
  inline bool test(int r) const {
    return (bits[r >> 6] >> (r & 63)) & 1;
  }

  template <typename F>
  inline void for_each_set_bit(F&& f) const {
    for (int k = 0; k < num_words; ++k) {
      uint64_t word = bits[k];
      while (word) {
        int r = (k << 6) + __builtin_ctzll(word);
        f(r);
        word &= (word - 1); // Clear the least significant set bit
      }
    }
  }
};

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
  int num_words = (cp_size + 63) >> 6;

  // 1. Precompute the index mask for each rank that allows processing
  // Use a single contiguous buffer for all masks to avoid thousands of allocations
  std::vector<uint64_t> qo_kv_masks_storage((m + n) * num_words, 0);
  auto get_qo_mask = [&](int i) { return FastMask(&qo_kv_masks_storage[i * num_words], num_words); };
  auto get_kv_mask = [&](int j) { return FastMask(&qo_kv_masks_storage[(m + j) * num_words], num_words); };

  for (int i = 0; i < m; ++i) {
    get_qo_mask(i).set(rank_m_vec[i]);
  }
  for (int j = 0; j < n; ++j) {
    get_kv_mask(j).set(rank_n_vec[j]);
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
        get_qo_mask(tag).set(vj);
      }
    } else {
      if (vj == rank_n_vec[tag]) {
        get_kv_mask(tag).set(uj);
      }
    }
  }

  // 2. Prepare task data and calculate "degree" (number of allowed ranks)
  struct Task {
    int idx;
    int i;
    int j;
    long long area;
    FastMask mask;
    int degree;
  };

  std::vector<Task> tasks(num_areas);
  std::vector<uint64_t> task_masks_storage(num_areas * num_words, 0);

  // Use atomic flag to track errors (boundary check or degree==0)
  std::atomic<bool> has_error(false);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int idx = 0; idx < num_areas; ++idx) {
    int i = std::get<0>(sparse_area_vec[idx]);
    int j = std::get<1>(sparse_area_vec[idx]);

    FastMask task_mask(&task_masks_storage[idx * num_words], num_words);
    int degree = 0;
    FastMask qo_mask = get_qo_mask(i);
    FastMask kv_mask = get_kv_mask(j);

    for (int k = 0; k < num_words; ++k) {
      uint64_t common = qo_mask.bits[k] & kv_mask.bits[k];
      task_mask.bits[k] = common;
      degree += __builtin_popcountll(common);
    }

    if (degree == 0) {
      has_error.store(true, std::memory_order_relaxed);
      tasks[idx] = {idx, i, j, std::get<2>(sparse_area_vec[idx]), task_mask, -1};
    } else {
      tasks[idx] = {idx, i, j, std::get<2>(sparse_area_vec[idx]), task_mask, degree};
    }
  }

  // Check for errors after parallel loop
  if (has_error.load(std::memory_order_relaxed)) {
    std::vector<Assignment> sparse_res(num_areas);
    for (int k = 0; k < num_areas; ++k) {
      sparse_res[k] = {std::get<0>(sparse_area_vec[k]), std::get<1>(sparse_area_vec[k]), -1};
    }
    return std::make_pair(false, sparse_res);
  }

  // 3. Sort by degree in ascending order
#ifdef _OPENMP
  __gnu_parallel::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
#else
  std::stable_sort(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
#endif
    if (a.degree != b.degree)
      return a.degree < b.degree;
    if (a.area != b.area)
      return a.area > b.area;
    return false;
  });

  // 4. Greedy assignment
  std::vector<long long> rank_loads(cp_size, 0);
  std::vector<Assignment> sparse_res(num_areas);
  for (int k = 0; k < num_areas; ++k)
    sparse_res[k] = {0, 0, -1};

  for (const auto& task : tasks) {
    const auto& mask = task.mask;
    long long area = task.area;
    int i = task.i;
    int j = task.j;

    int assign_rank = -1;
    if (rank_m_vec[i] == rank_n_vec[j]) {
      assign_rank = rank_m_vec[i];
    } else {
      // usp greedy
      int usp_rank = usp_choices_vec[i];
      if (usp_rank >= 0 && usp_rank < cp_size && mask.test(usp_rank) && rank_loads[usp_rank] < area_avg) {
        assign_rank = usp_rank;
      }
      // ring greedy
      if (assign_rank == -1) {
        int ring_rank = rank_m_vec[i];
        if (ring_rank >= 0 && ring_rank < cp_size && mask.test(ring_rank) && rank_loads[ring_rank] < area_avg) {
          assign_rank = ring_rank;
        }
      }
      // regular greedy: optimized search
      if (assign_rank == -1) {
        long long min_load = std::numeric_limits<long long>::max();
        mask.for_each_set_bit([&](int r) {
          if (rank_loads[r] < min_load) {
            min_load = rank_loads[r];
            assign_rank = r;
          }
        });
      }
    }

    rank_loads[assign_rank] += area;
    sparse_res[task.idx] = {i, j, assign_rank};
  }

  // 4.5. Adjustment Step: Local Refinement
  for (int iter = 0; iter < 5; ++iter) {
    bool any_overloaded = false;
    for (int r = 0; r < cp_size; ++r) {
      if (rank_loads[r] > max_allowed_load) {
        any_overloaded = true;
        break;
      }
    }
    if (!any_overloaded)
      break;

    bool changed = false;
    for (const auto& task : tasks) {
      int idx = task.idx;
      Assignment& curr_res = sparse_res[idx];
      int curr_rank = curr_res.rank;
      long long curr_load = rank_loads[curr_rank];

      if (curr_load <= max_allowed_load)
        continue;

      int best_target_rank = -1;
      long long min_target_load = curr_load;
      long long area = task.area;

      task.mask.for_each_set_bit([&](int r) {
        if (r == curr_rank)
          return;
        long long new_potential_load = rank_loads[r] + area;
        if (new_potential_load < min_target_load) {
          min_target_load = new_potential_load;
          best_target_rank = r;
        }
      });

      if (best_target_rank != -1) {
        rank_loads[curr_rank] -= area;
        rank_loads[best_target_rank] += area;
        sparse_res[idx].rank = best_target_rank;
        changed = true;
      }
    }
    if (!changed)
      break;
  }

  // 5. Feasibility check
  bool is_feasible = true;
  for (int r = 0; r < cp_size; ++r) {
    if (rank_loads[r] > max_allowed_load) {
      is_feasible = false;
      break;
    }
  }

  return std::make_pair(is_feasible, sparse_res);
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
  // --- Preprocess Analysis ---

  // 1. Convert Python rects (AttnRectangles) to C++ AttnRectangles
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

  // 2. Preprocess: Extract indexed_host_ranges and compute rank_m/n, comm_len_m/n
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

  // 3. Compute usp_choices
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

  // Pre-declare containers for computation results
  std::vector<std::tuple<int, int, AttnRectangles>> sparse_grid_rects;
  std::vector<Assignment> final_map;
  int num_areas = 0;

  // --- Release GIL for heavy computation ---
  {
    pybind11::gil_scoped_release release;

    // 4. Grid Func: split_grid_recursive
    sparse_grid_rects.reserve(m * n / 2); // Heuristic
    split_grid_recursive(rects, 0, m, 0, n, true, indexed_host_ranges_q, indexed_host_ranges_k, sparse_grid_rects);

    // 5. Binary Greedy logic
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

    std::vector<int> solver_prev_ranks(num_areas, -1);
    std::vector<Assignment> best_map;
    std::vector<Assignment> solver_try;
    bool has_best_map = false;
    double low = 0.0, high = threshold;
    double unbalance_rate = 1.10;
    double eps = 1e-2;

    bool edges_dirty = true;
    std::vector<Edge> edges;
    std::vector<CompactEdge> sorted_edges;

    for (int iter_idx = 0; iter_idx < 20; ++iter_idx) {
      double mid = (low + high) / 2.0;
      std::vector<int> solver_state_ranks = solver_prev_ranks;

      if (edges_dirty) {
        edges = calc_simplex_edges_cpp(cp_size, rank_m, rank_n, comm_len_m, comm_len_n, solver_state_ranks, sparse_area_vec, area_avg, num_heads_q, num_heads_kv);
      }

      std::vector<int> selected_edges;
      if (iter_idx == 0) {
        selected_edges.resize(edges.size());
        std::iota(selected_edges.begin(), selected_edges.end(), 0);
      } else {
        if (edges_dirty) {
          int num_edges = static_cast<int>(edges.size());
          sorted_edges.resize(num_edges);
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int j = 0; j < num_edges; ++j) {
            double score = edges[j].weight / std::max(static_cast<double>(edges[j].cost), 1e-6);
            sorted_edges[j] = {score, j};
          }

          auto sort_comp = [](const CompactEdge& a, const CompactEdge& b) { return a > b; };
          int topk = num_edges;
          if (num_edges > 2048) {
            int min_k = 2048;
            int ratio_k = static_cast<int>(static_cast<double>(num_edges) * 0.5);
            topk = std::min(num_edges, std::max(min_k, ratio_k));
          }

          if (topk == num_edges) {
#ifdef _OPENMP
            __gnu_parallel::sort(sorted_edges.begin(), sorted_edges.end(), sort_comp);
#else
            std::sort(sorted_edges.begin(), sorted_edges.end(), sort_comp);
#endif
          } else {
            auto topk_end = sorted_edges.begin() + topk;
            std::nth_element(sorted_edges.begin(), topk_end, sorted_edges.end(), sort_comp);

#ifdef _OPENMP
            __gnu_parallel::sort(sorted_edges.begin(), topk_end, sort_comp);
#else
            std::sort(sorted_edges.begin(), topk_end, sort_comp);
#endif
          }
          edges_dirty = false;
        }

        selected_edges = greedy_selection_from_sorted(cp_size, edges, sorted_edges, mid);
      }

      auto flow_res = greedy_max_flow_cpp(cp_size, edges, selected_edges, sparse_area_vec, rank_m, rank_n, usp_choices, area_avg, unbalance_rate, rank, false);

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

      if (high - low <= eps * high && low > 0.0)
        break;
    }

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
  }
  // --- GIL is re-acquired here ---

  // 6. Apply result to bucket_per_rank (Handle both C++ and Python types)
  pybind11::module_ py_range_mod;
  pybind11::module_ py_rect_mod;
  bool py_mods_loaded = false;

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
        AttnRectangles* cpp_bucket = cached_cpp_buckets[assigned_rank];
        for (size_t r_idx = 0; r_idx < rect_to_add.size(); ++r_idx) {
          cpp_bucket->append(rect_to_add.at(r_idx));
        }
      } else {
        // Fallback path: Python object manipulation
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
      }
    }
  }
}

pybind11::tuple cut_host_remote_buckets(const AttnRectangles& bucket_this_rank, const AttnRanges& host_ranges_q_this_rank, const AttnRanges& host_ranges_k_this_rank) {
  AttnRectangles host_bucket;
  AttnRectangles remote_bucket;

  int cut_pos_q = 0;
  AttnRectangles rest_rects_q = bucket_this_rank;

  for (const auto& host_range_q : host_ranges_q_this_rank.get()) {
    if (cut_pos_q != host_range_q.start) {
      cut_pos_q = host_range_q.start;
      auto pair = rest_rects_q.cut_q(cut_pos_q);
      remote_bucket.extend(pair.first);
      rest_rects_q = std::move(pair.second);
    }
    cut_pos_q = host_range_q.end;
    auto pair = rest_rects_q.cut_q(cut_pos_q);
    AttnRectangles rest_rects_k = std::move(pair.first);
    rest_rects_q = std::move(pair.second);

    int cut_pos_k = 0;
    for (const auto& host_range_k : host_ranges_k_this_rank.get()) {
      if (cut_pos_k != host_range_k.start) {
        cut_pos_k = host_range_k.start;
        auto pair_k = rest_rects_k.cut_k(cut_pos_k);
        remote_bucket.extend(pair_k.first);
        rest_rects_k = std::move(pair_k.second);
      }
      cut_pos_k = host_range_k.end;
      auto pair_k = rest_rects_k.cut_k(cut_pos_k);
      host_bucket.extend(pair_k.first);
      rest_rects_k = std::move(pair_k.second);
    }
    remote_bucket.extend(rest_rects_k);
  }
  remote_bucket.extend(rest_rects_q);

  return pybind11::make_tuple(std::move(host_bucket), std::move(remote_bucket));
}

AttnRanges expand_attn_ranges(const AttnRanges& ranges, int stride, int num_heads_group) {
  AttnRanges new_ranges;
  new_ranges.reserve(ranges.size() * num_heads_group);
  for (int i = 0; i < num_heads_group; ++i) {
    int offset = i * stride;
    for (const auto& r : ranges.get()) {
      new_ranges.append(AttnRange(r.start + offset, r.end + offset));
    }
  }
  return new_ranges;
}

} // namespace magi_attn_ext
