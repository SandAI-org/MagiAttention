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

import time
from collections import deque

import torch.distributed as dist

try:
    import pulp
except ImportError:
    pulp = None

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class FastSNFDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The Fast-Simplex-Network-Flow dynamic dispatch algorithm implementation"""

    def __init__(self, debug_print: bool = False):
        """
        The init method of the Fast-Simplex-Network-Flow dynamic dispatch algorithm
        """
        self.debug_print = debug_print

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.FAST_SIMPLEX_NETWORK_FLOW

    def _get_grid_rects(
        self,
        rects: AttnRectangles,
        indexed_host_ranges_q: list[tuple[AttnRange, int]],
        indexed_host_ranges_k: list[tuple[AttnRange, int]],
    ) -> list[list[AttnRectangles]]:
        """
        Get grid rectangles using a KD-tree style alternating split strategy
        """

        def split_grid(
            current_rects: AttnRectangles,
            q_ranges: list[tuple[AttnRange, int]],
            k_ranges: list[tuple[AttnRange, int]],
            prefer_q: bool,
        ) -> list[list[AttnRectangles]]:
            # Pruning: if no more rects, return empty AttnRectangles grid
            if not current_rects:
                return [[AttnRectangles() for _ in k_ranges] for _ in q_ranges]

            nq, nk = len(q_ranges), len(k_ranges)
            if nq == 1 and nk == 1:
                return [[current_rects]]

            # Decide split axis: alternate unless one dimension is exhausted
            split_q = prefer_q
            if nq <= 1:
                split_q = False
            elif nk <= 1:
                split_q = True

            if split_q:
                mid = nq // 2
                mid_pos = q_ranges[mid][0].start
                left_rects, right_rects = current_rects.cut_q(mid_pos)
                return split_grid(
                    left_rects, q_ranges[:mid], k_ranges, not prefer_q
                ) + split_grid(right_rects, q_ranges[mid:], k_ranges, not prefer_q)
            else:
                mid = nk // 2
                mid_pos = k_ranges[mid][0].start
                left_rects, right_rects = current_rects.cut_k(mid_pos)
                left_res = split_grid(
                    left_rects, q_ranges, k_ranges[:mid], not prefer_q
                )
                right_res = split_grid(
                    right_rects, q_ranges, k_ranges[mid:], not prefer_q
                )
                # Combine column results
                return [row_l + row_r for row_l, row_r in zip(left_res, right_res)]

        grid_rects = split_grid(
            rects, indexed_host_ranges_q, indexed_host_ranges_k, True
        )
        return grid_rects

    def _calc_simplex_edges(
        self,
        cp_size: int,
        rank_m: list[int],
        rank_n: list[int],
        comm_len_m: list[int],
        comm_len_n: list[int],
        solver_map: list[list[int]],
        area_map: list[list[int]],
        area_avg: float,
        num_heads_q: int,
        num_heads_kv: int,
    ) -> list[tuple[int, int, float, int, bool, int]]:
        edges = []
        m = len(rank_m)
        n = len(rank_n)
        for i in range(m):
            q_comm_cost = comm_len_m[i] * num_heads_q * 2
            comm_weight_list = [0 for _ in range(cp_size)]
            comm_weight_solved_sum = 0
            comm_weight_sum = 0
            for j in range(n):
                comm_weight = area_map[i][j]
                comm_weight_sum += comm_weight
                if solver_map[i][j] != -1:
                    # local q o communication: rank_m[i] -> solver_map[i][j]
                    comm_weight_list[solver_map[i][j]] += comm_weight
                    comm_weight_solved_sum += comm_weight
            comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum
            for rank in range(cp_size):
                # simplex weight function
                edges_weight = (
                    comm_weight_list[rank] + comm_weight_unsolved_sum / cp_size
                ) * 0.95 + area_avg / cp_size * 0.05
                if rank != rank_m[i]:
                    edges.append((rank_m[i], rank, edges_weight, q_comm_cost, True, i))

        for j in range(n):
            k_comm_cost = comm_len_n[j] * num_heads_kv
            comm_weight_list = [0 for _ in range(cp_size)]
            comm_weight_solved_sum = 0
            comm_weight_sum = 0
            for i in range(m):
                comm_weight = area_map[i][j]
                comm_weight_sum += comm_weight
                if solver_map[i][j] != -1:
                    # local k v communication: rank_n[j] -> solver_map[i][j]
                    comm_weight_list[solver_map[i][j]] += comm_weight
                    comm_weight_solved_sum += comm_weight
            comm_weight_unsolved_sum = comm_weight_sum - comm_weight_solved_sum
            for rank in range(cp_size):
                # simplex weight function
                edges_weight = (
                    comm_weight_list[rank] + comm_weight_unsolved_sum / cp_size
                ) * 0.95 + area_avg / cp_size * 0.05
                if rank != rank_n[j]:
                    edges.append((rank, rank_n[j], edges_weight, k_comm_cost, False, j))
        # TODO: Distinguish between send and recv
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

    def _NetworkFlow(
        self,
        cp_size: int,
        simplex_edges: list[tuple[int, int, float, int, bool, int]],
        simplex_selected_edges: list[int],
        area_map: list[list[int]],
        solver_map: list[list[int]],
        rank_m: list[int],
        rank_n: list[int],
        area_avg: float,
        unbalance_rate: float,
        comm_len_m: list[int],
        comm_len_n: list[int],
        num_heads_q: int,
        num_heads_kv: int,
        use_cost_flow: bool = True,
        log_stats: bool = False,
    ) -> tuple[bool, int, int]:
        """
        Build minimum cost maximum flow network based on simplex selection results,
        determine the execution rank for each job.

        Args:
            use_cost_flow: Whether to use cost flow. If False, all costs are set to 0,
                equivalent to maximum flow.
            log_stats: Whether to return graph statistics.

        Network structure:
        - Source s -> rank nodes: capacity is area_avg * unbalance_rate,
            cost determined by use_cost_flow
        - rank nodes -> job(i,j) nodes: capacity is area_map[i][j],
            cost determined by communication cost and use_cost_flow
        - job(i,j) nodes -> sink t: capacity is area_map[i][j], cost 0
        """
        # Decompose edges selected by simplex into qo/kv feasible sets
        m = len(area_map)
        n = len(area_map[0]) if m > 0 else 0
        qo_range_candidates: list[set[int]] = [set() for _ in range(m)]
        kv_range_candidates: list[set[int]] = [set() for _ in range(n)]

        for idx in simplex_selected_edges:
            uj, vj, _, _, is_qo, tag = simplex_edges[idx]
            if is_qo:
                # qo_range side: q of row i belongs to rank_m[tag], need uj == rank_m[tag]
                if uj == rank_m[tag]:
                    qo_range_candidates[tag].add(vj)
            else:
                # kv_range side: kv of column j belongs to rank_n[tag], need vj == rank_n[tag]
                if vj == rank_n[tag]:
                    kv_range_candidates[tag].add(uj)
        # Add local rank
        for i in range(m):
            qo_range_candidates[i].add(rank_m[i])
        for j in range(n):
            kv_range_candidates[j].add(rank_n[j])

        # ---- Build minimum cost maximum flow graph ----
        # Identify active jobs and assign IDs to avoid string node names
        active_jobs: list[tuple[int, int]] = []
        job_node_map = [[-1] * n for _ in range(m)]
        for i in range(m):
            area_row = area_map[i]
            for j in range(n):
                if area_row[j] > 1e-3:
                    job_node_map[i][j] = len(active_jobs)
                    active_jobs.append((i, j))
        num_jobs = len(active_jobs)

        # Node IDs: source=0, sink=1, ranks: 2..2+cp_size-1, jobs: 2+cp_size..
        source = 0
        sink = 1
        rank_node_start = 2
        job_node_start = rank_node_start + cp_size
        total_nodes = job_node_start + num_jobs

        # Use adjacency list: graph[u] = list[[v, cap, cost, rev_idx, orig_cap]]
        graph: list[list[list]] = [[] for _ in range(total_nodes)]
        edge_count = 0

        def add_edge(u: int, v: int, cap: float, cost: int) -> None:
            nonlocal edge_count
            if cap <= 1e-3:
                return
            # Forward edge
            graph[u].append([v, float(cap), int(cost), len(graph[v]), float(cap)])
            # Reverse edge
            graph[v].append([u, 0.0, -int(cost), len(graph[u]) - 1, 0.0])
            edge_count += 1

        # Source to rank:
        base_cap = area_avg
        extra_cap = area_avg * max(unbalance_rate - 1.0, 0.0)
        for r in range(cp_size):
            rn = rank_node_start + r
            # Base load edge
            add_edge(source, rn, base_cap, 0)
            # Extra load edge (if unbalance_rate > 1)
            if extra_cap > 0:
                extra_cost = 1 if use_cost_flow else 0
                add_edge(source, rn, extra_cap, extra_cost)

        total_job_area = 0.0
        # rank to job, then to sink
        for idx, (i, j) in enumerate(active_jobs):
            area = area_map[i][j]
            total_job_area += area
            job_node = job_node_start + idx

            # Only when both qo/kv are satisfied can rank execute this job
            allowed_ranks = qo_range_candidates[i].intersection(kv_range_candidates[j])

            # Greedy: put local_rank first in traversal order
            local_rank = rank_m[i] if rank_m[i] == rank_n[j] else None
            if local_rank is not None and local_rank in allowed_ranks:
                rank_order = [local_rank]
                for r in allowed_ranks:
                    if r != local_rank:
                        rank_order.append(r)
            else:
                rank_order = list(allowed_ranks)

            for r in rank_order:
                if use_cost_flow:
                    total_cost = 0 if (r == rank_m[i] or r == rank_n[j]) else 1
                else:
                    total_cost = 0
                add_edge(rank_node_start + r, job_node, area, total_cost)
            add_edge(job_node, sink, area, 0)

        node_count = total_nodes
        if total_job_area == 0:
            return True, node_count, edge_count

        # ---- Maximum flow solvers ----
        def max_flow_dinic() -> float:
            """Dinic's algorithm for faster maximum flow (when costs are not needed)"""
            level = [-1] * total_nodes

            def bfs():
                nonlocal level
                level = [-1] * total_nodes
                level[source] = 0
                q = deque([source])
                while q:
                    u = q.popleft()
                    for v, cap, cost, rev, orig_cap in graph[u]:
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
                    v, cap, cost, rev, orig_cap = edge
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

            total_flow = 0.0
            while bfs():
                ptr = [0] * total_nodes
                while True:
                    pushed = dfs(source, float("inf"), ptr)
                    if pushed <= 1e-9:
                        break
                    total_flow += pushed
            return total_flow

        def min_cost_max_flow() -> tuple[float, float]:
            """Optimized Min-Cost Max-Flow using SPFA with potentials and Dinic-style augmentation."""
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
                    dist_u = dist_map[u]
                    pot_u = potential[u]
                    for v, cap_uv, cost_uv, rev, _ in graph[u]:
                        if cap_uv > 1e-9:
                            reduced_c = cost_uv + pot_u - potential[v]
                            if dist_map[v] > dist_u + reduced_c + 1e-9:
                                dist_map[v] = dist_u + reduced_c
                                if not in_queue[v]:
                                    q.append(v)
                                    in_queue[v] = True

                if dist_map[sink] >= INF_DIST / 2:
                    break

                # Update potentials
                for i in range(total_nodes):
                    if dist_map[i] < INF_DIST / 2:
                        potential[i] += dist_map[i]

                # Dinic-style DFS for multi-path augmentation
                ptr = [0] * total_nodes
                vis = [False] * total_nodes

                def dfs(u, limit):
                    if u == sink or limit <= 1e-9:
                        return limit
                    vis[u] = True
                    pushed = 0.0
                    graph_u = graph[u]
                    while ptr[u] < len(graph_u):
                        edge = graph_u[ptr[u]]
                        v, cap_uv, cost_uv, rev, _ = edge
                        if (
                            not vis[v]
                            and cap_uv > 1e-9
                            and abs(cost_uv + potential[u] - potential[v]) < 1e-9
                        ):
                            tr = dfs(v, min(limit - pushed, cap_uv))
                            if tr > 1e-9:
                                edge[1] -= tr
                                graph[v][rev][1] += tr
                                nonlocal cost
                                cost += tr * cost_uv
                                pushed += tr
                                if pushed >= limit - 1e-9:
                                    vis[u] = False
                                    return pushed
                        ptr[u] += 1
                    vis[u] = False
                    return pushed

                while True:
                    p = dfs(source, float("inf"))
                    if p <= 1e-9:
                        break
                    flow += p

            return flow, cost

        if use_cost_flow:
            max_flow, _ = min_cost_max_flow()
        else:
            max_flow = max_flow_dinic()

        # Recover rank->job flow from residual graph to update solver_map
        for idx, (i, j) in enumerate(active_jobs):
            job_node = job_node_start + idx
            best_rank = -1
            best_flow_rate = -1.0

            # Flow from rank r to job_node is stored in the reverse edge capacity in graph[job_node]
            for edge in graph[job_node]:
                v = edge[0]
                # If neighbor is a rank node
                if rank_node_start <= v < job_node_start:
                    rank_idx = v - rank_node_start
                    flow_rj = edge[1]  # Capacity of reverse edge is the flow pushed
                    if flow_rj > 1e-6:
                        flow_rate = flow_rj / area_map[i][j]
                        if flow_rate > best_flow_rate:
                            best_flow_rate = flow_rate
                            best_rank = rank_idx
            solver_map[i][j] = best_rank

        return abs(max_flow - total_job_area) <= 1e-3, node_count, edge_count

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
        # Get rank number for distributed training (only in debug mode)
        rank = -1
        if dist.is_initialized():
            rank = dist.get_rank()

        # measure solver execution time on rank 0
        start_time = time.perf_counter() if rank == 0 else None

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
        grid_rects = self._get_grid_rects(
            rects, indexed_host_ranges_q, indexed_host_ranges_k
        )

        m = len(grid_rects)
        n = len(grid_rects[0])

        # solve the grid
        solver_map = [[-1 for _ in range(n)] for _ in range(m)]
        area_map = [[-1 for _ in range(n)] for _ in range(m)]

        # initialize the solver_map
        for i in range(m):
            for j in range(n):
                area_map[i][j] = grid_rects[i][j].area()

        threshold = 0.0
        threshold += num_heads_q * sum(comm_len_m) * cp_size
        threshold += num_heads_kv * sum(comm_len_n) * cp_size

        area_avg = sum(sum(row) for row in area_map) / cp_size
        unbalance_rate = 1.001

        # Binary search for threshold: if successful, shrink upper bound; if failed, raise lower bound
        low, high = 0.0, threshold
        best_map: list[list[int]] | None = None
        best_edges: list[tuple[int, int, float, int, bool, int]] | None = None
        best_selected_edges: list[int] | None = None
        max_iters = 20
        # Testing shows that 1 attempt is enough to succeed, multiple attempts won't help if it fails
        max_attempts = 1
        eps = 1e-3
        # Current solver state for generating simplex edges, initially all -1
        solver_prev = [[-1 for _ in range(n)] for _ in range(m)]

        total_iters = 0

        for _ in range(max_iters):
            total_iters += 1
            mid = (low + high) / 2

            success = False
            selected_edges = []
            solver_try = [[-1 for _ in range(n)] for _ in range(m)]
            # Each threshold tries at most max_attempts times,
            # if failed, use previous result to iteratively correct solver_map and retry
            solver_state = [row[:] for row in solver_prev]
            for _attempt in range(max_attempts):
                # Regenerate simplex edges based on current solver state
                edges = self._calc_simplex_edges(
                    cp_size,
                    rank_m,
                    rank_n,
                    comm_len_m,
                    comm_len_n,
                    solver_state,
                    area_map,
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

                solver_try = [[-1 for _ in range(n)] for _ in range(m)]
                # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
                use_cost_flow = False

                success, nf_nodes, nf_edges = self._NetworkFlow(
                    cp_size,
                    edges,
                    selected_edges,
                    area_map,
                    solver_try,
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
                    solver_state = [row[:] for row in solver_try]
                    break

            if success:
                best_map = solver_try
                best_edges = edges
                best_selected_edges = selected_edges

                # TODO The following code may affect the partitioning result.
                # Keep the code for experiment purpose
                """
                # Calculate solver_try actual threshold to accelerate binary search
                actual_rank_costs = [0.0 for _ in range(cp_size)]

                # QO communication costs: per row
                for i in range(m):
                    unique_ranks_in_row = set()
                    for j in range(n):
                        r = solver_try[i][j]
                        if r != -1 and area_map[i][j] > 1e-9:
                            unique_ranks_in_row.add(r)

                    for r in unique_ranks_in_row:
                        if r != rank_m[i]:
                            cost = comm_len_m[i] * num_heads_q
                            actual_rank_costs[r] += cost
                            actual_rank_costs[rank_m[i]] += cost

                # KV communication costs: per column
                for j in range(n):
                    unique_ranks_in_col = set()
                    for i in range(m):
                        r = solver_try[i][j]
                        if r != -1 and area_map[i][j] > 1e-9:
                            unique_ranks_in_col.add(r)

                    for r in unique_ranks_in_col:
                        if r != rank_n[j]:
                            cost = comm_len_n[j] * num_heads_kv
                            actual_rank_costs[r] += cost
                            actual_rank_costs[rank_n[j]] += cost

                solver_try_max_threshold = (
                    max(actual_rank_costs) if actual_rank_costs else 0.0
                )

                # Use the actual threshold to shrink high more aggressively
                high = min(mid, solver_try_max_threshold)
                """
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
                area_map,
                area_avg,
                num_heads_q,
                num_heads_kv,
            )

            selected_edges = self._GreedySelection(
                cp_size,
                edges,
                high,
            )

            # Try to use local as much as possible, run cost flow assignment again
            solver_try = [[-1 for _ in range(n)] for _ in range(m)]
            # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
            unbalance_rate = 1.000
            use_cost_flow = True

            success, nf_nodes, nf_edges = self._NetworkFlow(
                cp_size,
                edges,
                selected_edges,
                area_map,
                solver_try,
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
                solver_try = [[-1 for _ in range(n)] for _ in range(m)]
                assert best_edges is not None
                assert best_selected_edges is not None
                success, nf_nodes, nf_edges = self._NetworkFlow(
                    cp_size,
                    best_edges,
                    best_selected_edges,
                    area_map,
                    solver_try,
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
            selected_edges = selected_edges  # From last iteration
            success = False

        if not success:
            if rank == 0 and self.debug_print:
                print(
                    "[FastSNFDynamicAttnAlgorithm] network flow failed, fallback to Q owner assignment"
                )
            for i in range(m):
                for j in range(n):
                    if grid_rects[i][j].area() > 0 and solver_map[i][j] == -1:
                        solver_map[i][j] = rank_m[i]
                    else:
                        solver_map[i][j] = -1

        # Calculate result stage
        for i in range(m):
            for j in range(n):
                if solver_map[i][j] != -1:
                    bucket_per_rank[solver_map[i][j]].extend(grid_rects[i][j])

        # Statistics for load balancing only for debugging

        # rank_area = [0.0 for _ in range(cp_size)]
        # for i in range(m):
        #     for j in range(n):
        #         if solver_map[i][j] != -1:
        #             rank_area[solver_map[i][j]] += area_map[i][j]

        # max_area = max(rank_area) if rank_area else 0.0
        # unbalance_rate = max_area / area_avg if area_avg > 0 else 0.0

        if rank == 0 and start_time is not None and self.debug_print:
            elapsed = time.perf_counter() - start_time
            print(f"[FastSNFDynamicAttnAlgorithm] solve elapsed time: {elapsed:.6f}s")
