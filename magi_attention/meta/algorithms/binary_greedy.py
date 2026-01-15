# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class BinaryGreedyDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The Binary-Greedy dynamic dispatch algorithm implementation (Copy of FastSNF)"""

    def __init__(self, debug_print: bool = False):
        """
        The init method of the Binary-Greedy dynamic dispatch algorithm
        """
        self.debug_print = debug_print

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.BINARY_GREEDY

    def _get_grid_rects(
        self,
        rects: AttnRectangles,
        indexed_host_ranges_q: list[tuple[AttnRange, int]],
        indexed_host_ranges_k: list[tuple[AttnRange, int]],
    ) -> list[tuple[int, int, AttnRectangles]]:
        """
        Get grid rectangles using a KD-tree style alternating split strategy.
        Returns a list of (q_idx, k_idx, rects) for non-empty rects.
        """

        def split_grid(
            current_rects: AttnRectangles,
            q_idx_range: tuple[int, int],
            k_idx_range: tuple[int, int],
            prefer_q: bool,
        ) -> list[tuple[int, int, AttnRectangles]]:
            # Pruning: if no more rects, return empty list
            if not current_rects:
                return []

            q_start, q_end = q_idx_range
            k_start, k_end = k_idx_range
            nq, nk = q_end - q_start, k_end - k_start

            if nq == 1 and nk == 1:
                return [(q_start, k_start, current_rects)]

            # Decide split axis: alternate unless one dimension is exhausted
            split_q = prefer_q
            if nq <= 1:
                split_q = False
            elif nk <= 1:
                split_q = True

            if split_q:
                mid = nq // 2
                mid_pos = indexed_host_ranges_q[q_start + mid][0].start
                left_rects, right_rects = current_rects.cut_q(mid_pos)
                res = split_grid(
                    left_rects, (q_start, q_start + mid), (k_start, k_end), not prefer_q
                )
                res.extend(
                    split_grid(
                        right_rects,
                        (q_start + mid, q_end),
                        (k_start, k_end),
                        not prefer_q,
                    )
                )
                return res
            else:
                mid = nk // 2
                mid_pos = indexed_host_ranges_k[k_start + mid][0].start
                left_rects, right_rects = current_rects.cut_k(mid_pos)
                res = split_grid(
                    left_rects, (q_start, q_end), (k_start, k_start + mid), not prefer_q
                )
                res.extend(
                    split_grid(
                        right_rects,
                        (q_start, q_end),
                        (k_start + mid, k_end),
                        not prefer_q,
                    )
                )
                return res

        grid_rects = split_grid(
            rects,
            (0, len(indexed_host_ranges_q)),
            (0, len(indexed_host_ranges_k)),
            True,
        )
        return grid_rects

    def _calc_simplex_edges(
        self,
        cp_size: int,
        rank_m: list[int],
        rank_n: list[int],
        comm_len_m: list[int],
        comm_len_n: list[int],
        sparse_solver_map: list[tuple[int, int, int]],
        sparse_area_map: list[tuple[int, int, int]],
        area_avg: float,
        num_heads_q: int,
        num_heads_kv: int,
    ) -> list[tuple[int, int, float, int, bool, int]]:
        edges = []
        m = len(rank_m)
        n = len(rank_n)

        # Group sparse area map by row and column for fast lookup
        # Each entry in row_areas/col_areas is (other_idx, area, global_idx)
        row_areas: list[list[tuple[int, int, int]]] = [[] for _ in range(m)]
        col_areas: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
        row_sums = [0] * m
        col_sums = [0] * n

        for idx, (i, j, area) in enumerate(sparse_area_map):
            row_areas[i].append((j, area, idx))
            col_areas[j].append((i, area, idx))
            row_sums[i] += area
            col_sums[j] += area

        for i in range(m):
            q_comm_cost = comm_len_m[i] * num_heads_q
            comm_weight_list = [0 for _ in range(cp_size)]
            comm_weight_solved_sum = 0

            for j, comm_weight, global_idx in row_areas[i]:
                assigned_rank = sparse_solver_map[global_idx][2]
                if assigned_rank != -1:
                    # local q o communication: rank_m[i] -> solver_map[global_idx]
                    comm_weight_list[assigned_rank] += comm_weight
                    comm_weight_solved_sum += comm_weight

            comm_weight_sum = row_sums[i]
            comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum

            selected_ranks = [r for r in range(cp_size) if r != rank_m[i]]

            for rank in selected_ranks:
                # simplex weight function
                edges_weight = (
                    comm_weight_list[rank] + comm_weight_unsolved_sum / cp_size
                ) * 0.95 + area_avg / cp_size * 0.05
                edges.append((rank_m[i], rank, edges_weight, q_comm_cost, True, i))

        for j in range(n):
            k_comm_cost = comm_len_n[j] * num_heads_kv
            comm_weight_list = [0 for _ in range(cp_size)]
            comm_weight_solved_sum = 0

            for i, comm_weight, global_idx in col_areas[j]:
                assigned_rank = sparse_solver_map[global_idx][2]
                if assigned_rank != -1:
                    # local k v communication: rank_n[j] -> solver_map[global_idx]
                    comm_weight_list[assigned_rank] += comm_weight
                    comm_weight_solved_sum += comm_weight

            comm_weight_sum = col_sums[j]
            comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum

            selected_ranks = [r for r in range(cp_size) if r != rank_n[j]]

            for rank in selected_ranks:
                # simplex weight function
                edges_weight = (
                    comm_weight_list[rank] + comm_weight_unsolved_sum / cp_size
                ) * 0.95 + area_avg / cp_size * 0.05
                edges.append((rank, rank_n[j], edges_weight, k_comm_cost, False, j))
        # TODO: Distinguish between send and recv
        # rank * heads * (rank - 1) * 2 num_edges
        return edges

    def _GreedySelection(
        self,
        node_num: int,
        edges: list[tuple[int, int, float, int, bool, int]],
        threshold: float,
    ) -> list[int]:
        """
        Solve range selection problem using a greedy algorithm for faster execution.

        Args:
            node_num: Number of nodes (number of ranks)
            edges: List of edges, each edge is a tuple (uj, vj, wj, cj)
                - uj: Sending rank (node index)
                - vj: Receiving rank (node index)
                - wj: Edge value (weight)
                - cj: Edge cost (communication volume)
            threshold: Cost upper limit for each node

        Returns:
            List of indices of selected edges
        """
        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Sort edges by cost-effectiveness (weight / cost)
        # We use wj / max(cj, 1) as a simple heuristic
        indexed_edges = []
        for j in range(num_edges):
            uj, vj, wj, cj, _, _ = edges[j]
            score = wj / max(cj, 1e-6)
            indexed_edges.append((score, j))

        # Sort by score in descending order
        indexed_edges.sort(key=lambda x: x[0], reverse=True)

        selected_edges = []
        node_costs = [0.0] * node_num

        for _, j in indexed_edges:
            uj, vj, wj, cj, _, _ = edges[j]
            # Check if adding this edge exceeds the threshold for either node
            if node_costs[uj] + cj <= threshold and node_costs[vj] + cj <= threshold:
                node_costs[uj] += cj
                node_costs[vj] += cj
                selected_edges.append(j)

        return selected_edges

    def _GreedyMaxFlow(
        self,
        cp_size: int,
        simplex_edges: list[tuple[int, int, float, int, bool, int]],
        simplex_selected_edges: list[int],
        sparse_area_map: list[tuple[int, int, int]],
        rank_m: list[int],
        rank_n: list[int],
        usp_choices: list[int],
        area_avg: float,
        unbalance_rate: float,
    ) -> tuple[bool, list[tuple[int, int, int]], int, int]:
        """
        Greedy algorithm to replace max flow:
        1. Process tasks with small "selection space" (fewer allowed ranks) first.
        2. Always assign tasks to the candidate rank with the smallest current load.
        """
        m = len(rank_m)
        n = len(rank_n)
        max_allowed_load = area_avg * max(unbalance_rate, 1.0)

        # 1. Precompute the index mask for each rank that allows processing
        qo_masks = [0] * m
        kv_masks = [0] * n
        for i in range(m):
            qo_masks[i] |= 1 << rank_m[i]
        for j in range(n):
            kv_masks[j] |= 1 << rank_n[j]

        for idx in simplex_selected_edges:
            uj, vj, _, _, is_qo, tag = simplex_edges[idx]
            if is_qo:
                if uj == rank_m[tag]:
                    qo_masks[tag] |= 1 << vj
            else:
                if vj == rank_n[tag]:
                    kv_masks[tag] |= 1 << uj

        # 2. Prepare task data and calculate "degree" (number of allowed ranks)
        tasks = []
        for idx, (i, j, area) in enumerate(sparse_area_map):
            mask = qo_masks[i] & kv_masks[j]
            if mask == 0:
                res: list[tuple[int, int, int]] = [
                    (i, j, -1) for i, j, _ in sparse_area_map
                ]
                return False, res, 0, 0

            # Calculate the number of 1s in the mask (i.e., degree)
            degree = bin(mask).count("1")
            tasks.append(
                {
                    "idx": idx,
                    "i": i,
                    "j": j,
                    "area": area,
                    "mask": mask,
                    "degree": degree,
                }
            )

        # 3. Sort by degree in ascending order (tasks with smaller selection space first)
        tasks.sort(key=lambda x: (x["degree"], -x["area"]))

        # 4. Greedy assignment
        rank_loads = [0.0] * cp_size
        sparse_res: list[tuple[int, int, int]] = [(-1, -1, -1)] * len(sparse_area_map)

        # priority greedy: if the task (i, j) has the same rank_m[i] and rank_n[j] and the rank is in the mask,
        # it is a local task, assign to the local rank first
        for task in tasks:
            mask = task["mask"]
            area = task["area"]
            i = task["i"]
            j = task["j"]

            # check local rank
            local_rank = -1
            if rank_m[i] == rank_n[j]:
                local_rank = rank_m[i]

            if local_rank != -1:
                # local assignment priority
                assign_rank = local_rank
            else:
                best_rank = -1
                # usp greedy:
                if (mask >> usp_choices[i]) & 1:
                    if rank_loads[usp_choices[i]] < area_avg:
                        best_rank = usp_choices[i]
                # ring greedy:
                if best_rank == -1 and (mask >> rank_m[i]) & 1:
                    if rank_loads[rank_m[i]] < area_avg:
                        best_rank = rank_m[i]
                # regular greedy
                if best_rank == -1:
                    min_load = float("inf")
                    for r in range(cp_size):
                        if (mask >> r) & 1:
                            if rank_loads[r] < min_load:
                                min_load = rank_loads[r]
                                best_rank = r
                assign_rank = best_rank

            rank_loads[assign_rank] += area
            sparse_res[task["idx"]] = (i, j, assign_rank)

        # 4.5. Adjustment Step: Local Refinement
        # Iterate through tasks and try to move them to a rank that would have a lower load after the move.
        for _ in range(5):
            for task in tasks:
                idx = task["idx"]
                curr_i, curr_j, curr_rank = sparse_res[idx]
                area = task["area"]
                mask = task["mask"]

                curr_load = rank_loads[curr_rank]

                if curr_load < max_allowed_load:
                    continue

                best_target_rank = -1
                min_target_load = curr_load  # Only move if it improves balance

                for r in range(cp_size):
                    if (mask >> r) & 1 and r != curr_rank:
                        # If we move this task to rank 'r', its new load would be:
                        new_potential_load = rank_loads[r] + area
                        if new_potential_load < min_target_load:
                            min_target_load = new_potential_load
                            best_target_rank = r

                if best_target_rank != -1:
                    # Perform the swap
                    rank_loads[curr_rank] -= area
                    rank_loads[best_target_rank] += area
                    sparse_res[idx] = (curr_i, curr_j, best_target_rank)

        # 5. Check if the load upper limit constraint is satisfied
        # (corresponding to the feasible flow judgment in the original algorithm)
        # If unbalance_rate is negative or too small, it may not meet the demand,
        # here maintain the same judgment as the original logic
        is_feasible = all(load <= max_allowed_load + 1e-6 for load in rank_loads)

        # Return format: (is_feasible, assignment result, number of simulated nodes, number of simulated edges)
        # The number of nodes and edges in the greedy algorithm is not meaningful, return 0
        return is_feasible, sparse_res, 0, 0

    def solve(
        self,
        rects: AttnRectangles,
        host_ranges_q: list[AttnRanges],
        host_ranges_k: list[AttnRanges],
        num_heads_q: int,
        num_heads_kv: int,
        num_heads_group: int,
        bucket_per_rank: list[AttnRectangles],
    ) -> None:
        """
        The solve method of the Fast-Simplex-Network-Flow dynamic dispatch algorithm

        Args:
            rects: The attention rectangles
            host_ranges_q: The Q ranges of each rank
            host_ranges_k: The K ranges of each rank
            num_heads_q: The number of Q heads
            num_heads_kv: The number of KV heads
            bucket_per_rank: The buckets of each rank
        """
        # get the host rank list of Q and K
        cp_size = len(bucket_per_rank)
        rank_m = []
        rank_n = []
        comm_len_m = []
        comm_len_n = []

        indexed_host_ranges_q = []
        for idx, intervals in enumerate(host_ranges_q):
            indexed_host_ranges_q.extend([(interval, idx) for interval in intervals])
        indexed_host_ranges_q.sort(key=lambda x: x[0].start)
        for interval, idx in indexed_host_ranges_q:
            rank_m.append(idx)
            comm_len_m.append(interval.seqlen)

        indexed_host_ranges_k = []
        for idx, intervals in enumerate(host_ranges_k):
            indexed_host_ranges_k.extend([(interval, idx) for interval in intervals])
        indexed_host_ranges_k.sort(key=lambda x: x[0].start)
        for interval, idx in indexed_host_ranges_k:
            rank_n.append(idx)
            comm_len_n.append(interval.seqlen)

        # use USP choices as basic huristic solution
        intra_group_num = math.gcd(cp_size, num_heads_group)
        num_ranges_per_group = len(rank_m) // intra_group_num
        usp_choices = []
        for i, host_rank in enumerate(rank_m):
            group_idx = i // num_ranges_per_group
            usp_rank = (host_rank // intra_group_num) * intra_group_num + group_idx
            usp_choices.append(usp_rank)

        # get the grid rects
        sparse_grid_rects = self._get_grid_rects(
            rects, indexed_host_ranges_q, indexed_host_ranges_k
        )

        # get the area
        sparse_area_map: list[tuple[int, int, int]] = [
            (idx, jdx, rect.area()) for idx, jdx, rect in sparse_grid_rects
        ]

        # max comm cost per rank
        threshold = 0.0
        threshold += num_heads_q * sum(comm_len_m) * 2
        threshold += num_heads_kv * sum(comm_len_n) * 2

        area_avg = sum(area for _, _, area in sparse_area_map) / cp_size

        best_map: list[tuple[int, int, int]] | None = None

        # Current solver state for generating simplex edges, initially all -1
        solver_prev = [(i, j, -1) for i, j, _ in sparse_area_map]

        solver_map: list[tuple[int, int, int]] = []
        solver_try: list[tuple[int, int, int]] = []

        # Binary search for threshold: if successful, shrink upper bound; if failed, raise lower bound
        low, high = 0.0, threshold

        unbalance_rate = 1.10
        max_iters = 20
        # Testing shows that 1 attempt is enough to succeed, multiple attempts won't help if it fails
        max_attempts = 1
        eps = 1e-2

        for _ in range(max_iters):
            mid = (low + high) / 2

            success = False
            selected_edges = []
            solver_try = []
            # Each threshold tries at most max_attempts times,
            # if failed, use previous result to iteratively correct solver_map and retry
            solver_state = list(solver_prev)
            for _attempt in range(max_attempts):
                # Regenerate simplex edges based on current solver state
                edges = self._calc_simplex_edges(
                    cp_size,
                    rank_m,
                    rank_n,
                    comm_len_m,
                    comm_len_n,
                    solver_state,
                    sparse_area_map,
                    area_avg,
                    num_heads_q,
                    num_heads_kv,
                )

                if _ == 0:
                    # In the first iteration, mid is very large, greedy will choose all edges.
                    # We can skip it to save time.
                    selected_edges = list(range(len(edges)))
                else:
                    selected_edges = self._GreedySelection(
                        cp_size,
                        edges,
                        mid,
                    )

                success, solver_try, nf_nodes, nf_edges = self._GreedyMaxFlow(
                    cp_size,
                    edges,
                    selected_edges,
                    sparse_area_map,
                    rank_m,
                    rank_n,
                    usp_choices,
                    area_avg,
                    unbalance_rate,
                )

                if success:
                    # Use this assignment as input for next attempt
                    solver_state = solver_try
                    break

            if success:
                best_map = solver_try

                # TODO Use the actual threshold to shrink high more aggressively
                # The effect was not good, so it wasn't done
                high = mid
                solver_prev = solver_try
            else:
                low = mid
                # early stop to fasten binary search
                # break
            if high - low <= eps * high and low > 0:
                break

        solver_map = (
            best_map
            if best_map is not None
            else (solver_try if solver_try else solver_prev)
        )
        success = best_map is not None

        if not success:
            for idx, (i, j, area) in enumerate(sparse_area_map):
                if solver_map[idx][2] == -1:
                    solver_map[idx] = (i, j, rank_m[i])

        # Calculate result stage
        for idx, (i, j, rect) in enumerate(sparse_grid_rects):
            assigned_rank = solver_map[idx][2]
            if assigned_rank != -1:
                bucket_per_rank[assigned_rank].extend(rect)

        # Statistics for load balancing (for debugging)
        # rank_area = [0.0 for _ in range(cp_size)]
        # for idx, (_, _, area) in enumerate(sparse_area_map):
        #     assigned_rank = solver_map[idx][2]
        #     if assigned_rank != -1:
        #         rank_area[assigned_rank] += area

        # Verify that the total assigned area matches the total input area (for debugging)
        # total_assigned_area = sum(rank_area)
        # total_input_area = rects.area()
        # assert (
        #     abs(total_assigned_area - total_input_area) < 1e-3
        # ), f"Total assigned area {total_assigned_area} does not match input area {total_input_area}"

        # max_area = max(rank_area) if rank_area else 0.0
        # actual_unbalance_rate = max_area / area_avg if area_avg > 0 else 0.0
