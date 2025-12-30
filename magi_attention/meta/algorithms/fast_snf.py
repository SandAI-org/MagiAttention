# Copyright (c) 2025 SandAI. All Rights Reserved.
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
from collections import deque

try:
    import pulp
except ImportError:
    pulp = None

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class FastSNFDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The Fast-Simplex-Network-Flow dynamic dispatch algorithm implementation"""

    def __init__(self):
        """
        The init method of the Fast-Simplex-Network-Flow dynamic dispatch algorithm
        """
        pass

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.FAST_SIMPLEX_NETWORK_FLOW

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
        enable_topk: bool = False,
    ) -> list[tuple[int, int, float, int, bool, int]]:
        edges = []
        m = len(rank_m)
        n = len(rank_n)

        # Group sparse area map by row and column for fast lookup
        # Each entry in row_areas/col_areas is (other_idx, area, global_idx)
        row_areas: list[list[tuple[int, int, int]]] = [[] for _ in range(m)]
        col_areas: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
        row_sums = [0.0] * m
        col_sums = [0.0] * n

        for idx, (i, j, area) in enumerate(sparse_area_map):
            row_areas[i].append((j, area, idx))
            col_areas[j].append((i, area, idx))
            row_sums[i] += area
            col_sums[j] += area

        # Pre-calculate k_row and k_col if enable_topk is True
        if enable_topk:
            qk_rate = num_heads_q / num_heads_kv
            k_row = max(1, min(cp_size, 2 * int(math.sqrt(cp_size / qk_rate))))
            k_col = max(
                1, min(cp_size, 2 * int(math.sqrt(cp_size / qk_rate) * qk_rate))
            )

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

            if enable_topk:
                # Top-K selection for rows
                weighted_ranks = []
                for r, w in enumerate(comm_weight_list):
                    if w > 0 and r != rank_m[i]:
                        weighted_ranks.append((w, r))

                if len(weighted_ranks) > k_row:
                    weighted_ranks.sort(key=lambda x: x[0], reverse=True)
                    selected_ranks = [r for w, r in weighted_ranks[:k_row]]
                else:
                    selected_ranks = [r for w, r in weighted_ranks]
                    # Fill with some more ranks to reach k_row to ensure enough choices
                    # Distribute choices using offset
                    for offset in range(cp_size):
                        r = (i + offset) % cp_size
                        if r != rank_m[i] and r not in selected_ranks:
                            selected_ranks.append(r)
                            if len(selected_ranks) >= k_row:
                                break
            else:
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

            if enable_topk:
                # Top-K selection for columns
                weighted_ranks = []
                for r, w in enumerate(comm_weight_list):
                    if w > 0 and r != rank_n[j]:
                        weighted_ranks.append((w, r))

                if len(weighted_ranks) > k_col:
                    weighted_ranks.sort(key=lambda x: x[0], reverse=True)
                    selected_ranks = [r for w, r in weighted_ranks[:k_col]]
                else:
                    selected_ranks = [r for w, r in weighted_ranks]
                    # Fill with some more ranks to reach k_col
                    for offset in range(cp_size):
                        r = (j + offset) % cp_size
                        if r != rank_n[j] and r not in selected_ranks:
                            selected_ranks.append(r)
                            if len(selected_ranks) >= k_col:
                                break
            else:
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

    def _Simplex(
        self,
        node_num: int,
        edges: list[tuple[int, int, float, int, bool, int]],
        threshold: float,
    ) -> list[int]:
        """
        Solve range selection problem using integer linear programming

        Args:
            node_num: Number of nodes (number of ranks)
            edges: List of edges, each edge is a tuple (uj, vj, wj, cj)
                - uj: Sending rank (node index)
                - vj: Receiving rank (node index)
                - wj: Edge value (weight, can be any value, to be designed later)
                - cj: Edge cost (communication volume)
            threshold: Cost upper limit for each node

        Returns:
            List of indices of selected edges
        """
        if pulp is None:
            raise ImportError(
                "PuLP is required for Simplex algorithm. "
                "Please install it with: pip install pulp"
            )

        num_edges = len(edges)
        if num_edges == 0:
            return []

        # Create integer linear programming problem
        # Maximize objective function
        prob = pulp.LpProblem("Edge_Selection", pulp.LpMaximize)

        # Decision variables: x[j] = 1 means edge j is selected, x[j] = 0 means not selected
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(num_edges)]

        # Objective function: maximize the sum of weights of selected edges
        prob += pulp.lpSum([edges[j][2] * x[j] for j in range(num_edges)])

        # Constraint: total cost of each node does not exceed K
        # For each node i, calculate the sum of costs of all edges connected to it
        for i in range(node_num):
            # Find all edges connected to node i
            connected_edges = []
            for j in range(num_edges):
                uj, vj, wj, cj, _, _ = edges[j]
                if uj == i or vj == i:
                    connected_edges.append((j, cj))

            if connected_edges:
                # Add constraint: sum of costs of all edges connected to node i <= K
                prob += (
                    pulp.lpSum([cj * x[j] for j, cj in connected_edges]) <= threshold
                )

        # Solve the problem: set time limit to avoid long hangs
        # timeLimit is in seconds, can be adjusted as needed (e.g., 5 or 10)
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=0.010, threads=1)
        prob.solve(solver)

        # Check solution status: allow non-optimal but feasible solutions (e.g., interrupted by timeLimit)
        status = pulp.LpStatus[prob.status]
        if status not in ("Optimal", "Not Solved", "Undefined"):
            # Infeasible / Unbounded are directly considered as failures
            return []

        # Extract indices of selected edges
        selected_edges = []
        for j in range(num_edges):
            if pulp.value(x[j]) == 1:
                selected_edges.append(j)

        return selected_edges

    def _MaxFlow(
        self,
        cp_size: int,
        simplex_edges: list[tuple[int, int, float, int, bool, int]],
        simplex_selected_edges: list[int],
        sparse_area_map: list[tuple[int, int, int]],
        rank_m: list[int],
        rank_n: list[int],
        area_avg: float,
        unbalance_rate: float,
    ) -> tuple[bool, list[tuple[int, int, int]], int, int]:
        """
        Optimized maximum flow algorithm: uses Dinic's algorithm on grouped jobs.
        """
        m = len(rank_m)
        n = len(rank_n)

        # 1. Precompute candidate Rank sets
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

        # 2. Job Grouping
        groups: dict[int, int] = {}
        job_to_group_key = []
        for idx, (i, j, area) in enumerate(sparse_area_map):
            mask = qo_masks[i] & kv_masks[j]
            key = mask
            if key not in groups:
                groups[key] = 0
            groups[key] += area
            job_to_group_key.append(key)

        sorted_group_keys = sorted(groups.keys())
        group_key_to_idx = {key: idx for idx, key in enumerate(sorted_group_keys)}
        num_groups = len(sorted_group_keys)

        # 3. Build Graph
        source = 0
        sink = 1
        rank_node_start = 2
        group_node_start = rank_node_start + cp_size
        total_nodes = group_node_start + num_groups

        graph: list[list[list]] = [[] for _ in range(total_nodes)]
        edge_count = 0

        def add_edge(u: int, v: int, cap: float) -> None:
            nonlocal edge_count
            if cap <= 1e-6:
                return
            # [to_node, capacity, cost(0), rev_idx, orig_cap]
            graph[u].append([v, float(cap), 0, len(graph[v]), float(cap)])
            graph[v].append([u, 0.0, 0, len(graph[u]) - 1, 0.0])
            edge_count += 1

        # Source -> Rank (Load Balancing)
        total_cap = area_avg * max(unbalance_rate, 1.0)
        for r in range(cp_size):
            add_edge(source, rank_node_start + r, total_cap)

        # Rank -> Group -> Sink
        total_job_area = 0.0
        for g_idx, mask in enumerate(sorted_group_keys):
            area = groups[mask]
            total_job_area += area
            group_node = group_node_start + g_idx

            for r in range(cp_size):
                if (mask >> r) & 1:
                    add_edge(rank_node_start + r, group_node, area)

            add_edge(group_node, sink, area)

        if total_job_area <= 1e-9:
            return True, [], total_nodes, edge_count

        # 4. Dinic's Algorithm
        level = [-1] * total_nodes

        def bfs():
            for i in range(total_nodes):
                level[i] = -1
            level[source] = 0
            q = deque([source])
            while q:
                u = q.popleft()
                for v, cap, cost, rev, _ in graph[u]:
                    if cap > 1e-9 and level[v] == -1:
                        level[v] = level[u] + 1
                        q.append(v)
            return level[sink] != -1

        def dfs(u, limit, ptr):
            if u == sink or limit <= 1e-9:
                return limit
            pushed = 0.0
            graph_u = graph[u]
            while ptr[u] < len(graph_u):
                edge = graph_u[ptr[u]]
                v, cap, cost, rev, _ = edge
                if level[v] == level[u] + 1 and cap > 1e-9:
                    tr = dfs(v, min(limit - pushed, cap), ptr)
                    if tr > 1e-9:
                        edge[1] -= tr
                        graph[v][rev][1] += tr
                        pushed += tr
                        if pushed >= limit - 1e-9:
                            return pushed
                ptr[u] += 1
            return pushed

        max_flow = 0.0
        while bfs():
            ptr = [0] * total_nodes
            while True:
                pushed = dfs(source, float("inf"), ptr)
                if pushed <= 1e-9:
                    break
                max_flow += pushed

        # 5. Result Recovery
        rank_to_group_flow = [[0.0] * num_groups for _ in range(cp_size)]
        for g_idx in range(num_groups):
            group_node = group_node_start + g_idx
            for edge in graph[group_node]:
                v, cap_rem, _, _, orig_cap = edge
                if rank_node_start <= v < group_node_start:
                    rank_to_group_flow[v - rank_node_start][g_idx] = cap_rem

        sparse_res = []
        for idx, (i, j, area) in enumerate(sparse_area_map):
            key = job_to_group_key[idx]
            g_idx = group_key_to_idx[key]

            best_rank = -1
            max_f = -1.0
            for r in range(cp_size):
                f = rank_to_group_flow[r][g_idx]
                if f > area:
                    best_rank = r
                    break
                if f > max_f:
                    max_f = f
                    best_rank = r

            if best_rank != -1:
                rank_to_group_flow[best_rank][g_idx] -= area
            sparse_res.append((i, j, best_rank))

        return (
            abs(max_flow - total_job_area) <= 1e-3,
            sparse_res,
            total_nodes,
            edge_count,
        )

    def _NetworkFlow(
        self,
        cp_size: int,
        simplex_edges: list[tuple[int, int, float, int, bool, int]],
        simplex_selected_edges: list[int],
        sparse_area_map: list[tuple[int, int, int]],
        rank_m: list[int],
        rank_n: list[int],
        area_avg: float,
        unbalance_rate: float,
        comm_len_m: list[int],
        comm_len_n: list[int],
        num_heads_q: int,
        num_heads_kv: int,
        use_cost_flow: bool = True,
    ) -> tuple[bool, list[tuple[int, int, int]], int, int]:
        """
        # Optimized network flow algorithm: dramatically reduces graph size and accelerates solving via job grouping.
        """
        if not use_cost_flow:
            return self._MaxFlow(
                cp_size,
                simplex_edges,
                simplex_selected_edges,
                sparse_area_map,
                rank_m,
                rank_n,
                area_avg,
                unbalance_rate,
            )

        m = len(rank_m)
        n = len(rank_n)

        # 1. Precompute candidate Rank sets for each row and column using bitmasks
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

        # 2. Job grouping: jobs with the same (allowed_mask, owner_rank) are merged
        # key: (mask, owner_rank)
        groups: dict[tuple[int, int], int] = {}
        job_to_group_key = []
        for idx, (i, j, area) in enumerate(sparse_area_map):
            mask = qo_masks[i] & kv_masks[j]
            key = (mask, rank_m[i] if rank_m[i] == rank_n[j] else -1)
            if key not in groups:
                groups[key] = 0
            groups[key] += area
            job_to_group_key.append(key)

        sorted_group_keys = sorted(groups.keys())
        group_key_to_idx = {key: idx for idx, key in enumerate(sorted_group_keys)}
        num_groups = len(sorted_group_keys)

        # 3. Build the reduced network flow graph
        source = 0
        sink = 1
        rank_node_start = 2
        group_node_start = rank_node_start + cp_size
        total_nodes = group_node_start + num_groups

        graph: list[list[list]] = [[] for _ in range(total_nodes)]
        edge_count = 0

        def add_edge(u: int, v: int, cap: float, cost: int) -> None:
            nonlocal edge_count
            if cap <= 1e-6:
                return
            # [to_node, capacity, cost, rev_idx, orig_cap]
            graph[u].append([v, float(cap), int(cost), len(graph[v]), float(cap)])
            graph[v].append([u, 0.0, -int(cost), len(graph[u]) - 1, 0.0])
            edge_count += 1

        # Source -> Rank (Load constraints)
        base_cap = area_avg
        extra_cap = area_avg * max(unbalance_rate - 1.0, 0.0)
        for r in range(cp_size):
            rn = rank_node_start + r
            add_edge(source, rn, base_cap, 0)
            if extra_cap > 1e-6:
                add_edge(source, rn, extra_cap, 1)

        # Rank -> Group -> Sink (Job assignment)
        total_job_area = 0.0
        for g_idx, key in enumerate(sorted_group_keys):
            mask, owner = key
            area = groups[key]
            total_job_area += area
            group_node = group_node_start + g_idx

            # Only Ranks identified in the mask can process this group of jobs
            for r in range(cp_size):
                if (mask >> r) & 1:
                    cost = 0 if (r == owner) else 1
                    add_edge(rank_node_start + r, group_node, area, cost)

            add_edge(group_node, sink, area, 0)

        if total_job_area <= 1e-9:
            return True, [], total_nodes, edge_count

        # 4. Min-cost max-flow solver (SPFA + Dinic-style DFS)
        def min_cost_max_flow() -> tuple[float, float]:
            flow = 0.0
            cost = 0.0
            potential = [0.0] * total_nodes
            INF_DIST = 1e15

            while True:
                dist_map = [INF_DIST] * total_nodes
                dist_map[source] = 0
                q = deque([source])
                in_queue = [False] * total_nodes
                in_queue[source] = True

                while q:
                    u = q.popleft()
                    in_queue[u] = False
                    dist_u, pot_u = dist_map[u], potential[u]
                    for v, cap, cost_uv, rev, _ in graph[u]:
                        if cap > 1e-9:
                            reduced_c = cost_uv + pot_u - potential[v]
                            if dist_map[v] > dist_u + reduced_c + 1e-9:
                                dist_map[v] = dist_u + reduced_c
                                if not in_queue[v]:
                                    q.append(v)
                                    in_queue[v] = True

                if dist_map[sink] >= INF_DIST / 2:
                    break
                for i in range(total_nodes):
                    if dist_map[i] < INF_DIST / 2:
                        potential[i] += dist_map[i]

                ptr = [0] * total_nodes
                vis = [False] * total_nodes

                def dfs(u, limit):
                    if u == sink or limit <= 1e-9:
                        return limit
                    vis[u] = True
                    pushed = 0.0
                    for i in range(ptr[u], len(graph[u])):
                        ptr[u] = i
                        v, cap, cost_uv, rev, _ = graph[u][i]
                        if (
                            not vis[v]
                            and cap > 1e-9
                            and abs(cost_uv + potential[u] - potential[v]) < 1e-9
                        ):
                            tr = dfs(v, min(limit - pushed, cap))
                            if tr > 1e-9:
                                graph[u][i][1] -= tr
                                graph[v][rev][1] += tr
                                nonlocal cost
                                cost += tr * cost_uv
                                pushed += tr
                                if pushed >= limit - 1e-9:
                                    vis[u] = False
                                    return pushed
                    vis[u] = False
                    ptr[u] = len(graph[u])
                    return pushed

                while True:
                    p = dfs(source, float("inf"))
                    if p <= 1e-9:
                        break
                    flow += p
            return flow, cost

        # Solve for flow
        max_flow, _ = min_cost_max_flow()

        # 5. Result recovery: map group flow back to original jobs
        # rank_to_group_flow[rank_idx][group_idx]
        rank_to_group_flow = [[0.0] * num_groups for _ in range(cp_size)]
        for g_idx in range(num_groups):
            group_node = group_node_start + g_idx
            for edge in graph[group_node]:
                v, cap_rem, _, _, orig_cap = edge
                # Reverse edge flow = initial capacity (0) - residual capacity (negative),
                # which is the flow pushed by the forward edge
                if rank_node_start <= v < group_node_start:
                    rank_to_group_flow[v - rank_node_start][g_idx] = cap_rem

        sparse_res = []
        for idx, (i, j, area) in enumerate(sparse_area_map):
            key = job_to_group_key[idx]
            g_idx = group_key_to_idx[key]

            # Greedy assignment: extract flow from the rank with the most residual flow in the group
            best_rank = -1
            max_f = -1.0
            for r in range(cp_size):
                f = rank_to_group_flow[r][g_idx]
                if f > area:
                    best_rank = r
                    break
                if f > max_f:
                    max_f = f
                    best_rank = r

            if best_rank != -1:
                rank_to_group_flow[best_rank][g_idx] -= area
            sparse_res.append((i, j, best_rank))

        return (
            abs(max_flow - total_job_area) <= 1e-3,
            sparse_res,
            total_nodes,
            edge_count,
        )

    def solve(
        self,
        rects: AttnRectangles,
        host_ranges_q: list[AttnRanges],
        host_ranges_k: list[AttnRanges],
        num_heads_q: int,
        num_heads_kv: int,
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

        # get the grid rects
        sparse_grid_rects = self._get_grid_rects(
            rects, indexed_host_ranges_q, indexed_host_ranges_k
        )

        # get the area
        sparse_area_map = [
            (idx, jdx, rect.area()) for idx, jdx, rect in sparse_grid_rects
        ]

        # solve the grid
        solver_map: list[tuple[int, int, int]] = []

        # max comm cost per rank
        threshold = 0.0
        threshold += num_heads_q * sum(comm_len_m) * 2
        threshold += num_heads_kv * sum(comm_len_n) * 2

        area_avg = sum(area for _, _, area in sparse_area_map) / cp_size
        unbalance_rate = 1.000

        # Binary search for threshold: if successful, shrink upper bound; if failed, raise lower bound
        low, high = 0.0, threshold
        best_map: list[tuple[int, int, int]] | None = None
        best_edges: list[tuple[int, int, float, int, bool, int]] | None = None
        best_selected_edges: list[int] | None = None
        max_iters = 20
        # Testing shows that 1 attempt is enough to succeed, multiple attempts won't help if it fails
        max_attempts = 1
        eps = 1e-2
        # Current solver state for generating simplex edges, initially all -1
        solver_prev = [(i, j, -1) for i, j, _ in sparse_area_map]

        for _ in range(max_iters):
            mid = (low + high) / 2

            success = False
            selected_edges = []
            solver_try: list[tuple[int, int, int]] = []
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
                    enable_topk=(_ != 0),
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

                # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
                use_cost_flow = False

                success, solver_try, nf_nodes, nf_edges = self._NetworkFlow(
                    cp_size,
                    edges,
                    selected_edges,
                    sparse_area_map,
                    rank_m,
                    rank_n,
                    area_avg,
                    unbalance_rate,
                    comm_len_m,
                    comm_len_n,
                    num_heads_q,
                    num_heads_kv,
                    use_cost_flow,
                )

                if success:
                    # Use this assignment as input for next attempt
                    solver_state = solver_try
                    break

            if success:
                best_map = solver_try
                best_edges = edges
                best_selected_edges = selected_edges

                # TODO Use the actual threshold to shrink high more aggressively
                # The effect was not good, so it wasn't done

                high = mid
                solver_prev = solver_try
            else:
                low = mid

            if high - low <= eps * high and low > 0:
                break

        if best_map is not None:
            # Use optimal assignment obtained from binary search
            solver_map = best_map

            # Re-calculate edges and selected_edges using the best_map and final high
            edges = self._calc_simplex_edges(
                cp_size,
                rank_m,
                rank_n,
                comm_len_m,
                comm_len_n,
                solver_map,
                sparse_area_map,
                area_avg,
                num_heads_q,
                num_heads_kv,
                enable_topk=True,
            )

            selected_edges = self._GreedySelection(
                cp_size,
                edges,
                high,
            )

            # Try to use local as much as possible, run cost flow assignment again
            # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
            unbalance_rate = 1.000
            use_cost_flow = True

            success, solver_try, nf_nodes, nf_edges = self._NetworkFlow(
                cp_size,
                edges,
                selected_edges,
                sparse_area_map,
                rank_m,
                rank_n,
                area_avg,
                unbalance_rate,
                comm_len_m,
                comm_len_n,
                num_heads_q,
                num_heads_kv,
                use_cost_flow,
            )
            if not success:
                # use last success result (best_edges and best_selected_edges) to run cost flow assignment again
                assert best_edges is not None
                assert best_selected_edges is not None
                success, solver_try, nf_nodes, nf_edges = self._NetworkFlow(
                    cp_size,
                    best_edges,
                    best_selected_edges,
                    sparse_area_map,
                    rank_m,
                    rank_n,
                    area_avg,
                    unbalance_rate,
                    comm_len_m,
                    comm_len_n,
                    num_heads_q,
                    num_heads_kv,
                    use_cost_flow,
                )

            if success:
                solver_map = solver_try
            else:
                solver_map = best_map

        else:
            # Record last attempt result for debugging
            solver_map = solver_try
            success = False

        if not success:
            for idx, (i, j, area) in enumerate(sparse_area_map):
                if solver_map[idx][2] == -1:
                    solver_map[idx] = (i, j, rank_m[i])

        # Calculate result stage
        for idx, (i, j, rect) in enumerate(sparse_grid_rects):
            assigned_rank = solver_map[idx][2]
            if assigned_rank != -1:
                bucket_per_rank[assigned_rank].extend(rect)
