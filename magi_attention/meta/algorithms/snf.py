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
from collections import defaultdict, deque

import torch.distributed as dist

try:
    import pulp
except ImportError:
    pulp = None

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


# FIXME: SNF might dispatch the whole local stage into multiple ones,
# which is incompatible with attn sink
class SNFDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The Simplex-Network-Flow dynamic dispatch algorithm implementation"""

    def __init__(self, debug_print: bool = False):
        """
        The init method of the Simplex-Network-Flow dynamic dispatch algorithm
        """
        self.debug_print = debug_print

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.SIMPLEX_NETWORK_FLOW

    def _get_grid_rects(
        self,
        rects: AttnRectangles,
        indexed_host_ranges_q: list[tuple[AttnRange, int]],
        indexed_host_ranges_k: list[tuple[AttnRange, int]],
    ) -> list[list[AttnRectangles]]:
        """
        Get grid rectangles by cutting Q and K tiles

        Args:
            rects: The attention rectangles
            indexed_host_ranges_q: List of (Q range, host rank) tuples
            indexed_host_ranges_k: List of (K range, host rank) tuples

        Returns:
            Grid rectangles organized as list[list[AttnRectangles]]
        """
        rest_rects = rects
        grid_rects = []
        # cut Q tiles
        q_cut_pos = 0
        for item in indexed_host_ranges_q:
            q_tile_range: AttnRange = item[0]
            if q_cut_pos != q_tile_range.start:
                q_cut_pos = q_tile_range.start
                no_rects, rest_rects = rest_rects.cut_q(cut_pos=q_cut_pos)
                if len(no_rects) > 0:
                    raise ValueError(f"No rectangles are cut at position {q_cut_pos}")
            q_cut_pos = q_tile_range.end
            q_tile_rects, rest_rects = rest_rects.cut_q(cut_pos=q_cut_pos)
            # cut K tiles
            q_tile_list = []
            k_cut_pos = 0
            for item in indexed_host_ranges_k:
                k_tile_range: AttnRange = item[0]
                if k_cut_pos != k_tile_range.start:
                    k_cut_pos = k_tile_range.start
                    no_rects, q_tile_rects = q_tile_rects.cut_k(cut_pos=k_cut_pos)
                    if len(no_rects) > 0:
                        raise ValueError(
                            f"No rectangles are cut at position {k_cut_pos}"
                        )
                k_cut_pos = k_tile_range.end
                q_k_tile_rects, q_tile_rects = q_tile_rects.cut_k(cut_pos=k_cut_pos)
                q_tile_list.append(q_k_tile_rects)

            grid_rects.append(q_tile_list)
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
            q_comm_cost = comm_len_m[i] * num_heads_q
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
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=0.5, threads=1)
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
        unbanlance_rate: float,
        comm_len_m: list[int],
        comm_len_n: list[int],
        num_heads_q: int,
        num_heads_kv: int,
        use_cost_flow: bool = True,
    ) -> bool:
        """
        Build minimum cost maximum flow network based on simplex selection results,
        determine the execution rank for each job.

        Args:
            use_cost_flow: Whether to use cost flow. If False, all costs are set to 0,
                equivalent to maximum flow.

        Network structure:
        - Source s -> rank nodes: capacity is area_avg * unbanlance_rate,
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
                # qo_range side: q of row i belongs to rank_m[i], need uj == rank_m[tag]
                if uj == rank_m[tag]:
                    qo_range_candidates[tag].add(vj)
            else:
                # kv_range side: kv of column j belongs to rank_n[j], need vj == rank_n[tag]
                if vj == rank_n[tag]:
                    kv_range_candidates[tag].add(uj)
        # Add local rank
        for i in range(m):
            qo_range_candidates[i].add(rank_m[i])
        for j in range(n):
            kv_range_candidates[j].add(rank_n[j])

        # ---- Build minimum cost maximum flow graph ----
        # Use adjacency list: graph[u] = list[[v, cap, cost, rev_idx, orig_cap]]
        graph: dict[str, list[list]] = defaultdict(list)

        def add_edge(u: str, v: str, cap: float, cost: float) -> None:
            if cap <= 0:
                return
            # Forward edge
            graph[u].append([v, cap, cost, len(graph[v]), cap])
            # Reverse edge
            graph[v].append([u, 0.0, -cost, len(graph[u]) - 1, 0.0])

        source = "s"
        sink = "t"

        # Source to rank:
        # - First edge: capacity area_avg, cost 0, represents "base" load,
        #   encourages uniform distribution
        # - Second edge: capacity area_avg * (unbanlance_rate - 1),
        #   cost temporarily set to area_avg, indicates that extra load will
        #   incur higher cost, can replace area_avg with better cost function later
        rank_nodes = [f"r{r}" for r in range(cp_size)]
        base_cap = area_avg
        extra_cap = area_avg * max(unbanlance_rate - 1.0, 0.0)
        for rn in rank_nodes:
            # Base load edge
            add_edge(source, rn, base_cap, 0.0)
            # Extra load edge (if unbanlance_rate > 1)
            # TODO: Use better cost function
            if extra_cap > 0:
                # extra_cost = area_avg if use_cost_flow else 0.0
                extra_cost = 1.0 if use_cost_flow else 0.0
                add_edge(source, rn, extra_cap, extra_cost)

        total_job_area = 0.0
        job_nodes: list[list[str | None]] = [[None for _ in range(n)] for _ in range(m)]

        # rank to job, then to sink
        for i in range(m):
            for j in range(n):
                area = area_map[i][j]
                if area <= 1e-3:
                    continue
                total_job_area += area
                job_node = f"j{i}_{j}"
                job_nodes[i][j] = job_node

                # Only when both qo/kv are satisfied can rank execute this job
                allowed_ranks = qo_range_candidates[i].intersection(
                    kv_range_candidates[j]
                )

                # Greedy: put local_rank first in traversal order,
                # let the cost flow algorithm decide whether to use it
                local_rank = rank_m[i] if rank_m[i] == rank_n[j] else None
                if local_rank is not None and local_rank in allowed_ranks:
                    rank_order = [local_rank] + [
                        r for r in allowed_ranks if r != local_rank
                    ]
                else:
                    rank_order = list(allowed_ranks)

                for r in rank_order:
                    if use_cost_flow:
                        # TODO: Use better cost function based on communication cost,
                        # rank holding qo/kv sequences, rank load situation, etc.
                        # Calculate cost: if rank holds qo/kv sequences, cost is 0, otherwise communication cost
                        # qo communication cost: if rank != rank_m[i], need communication,
                        # cost = comm_len_m[i] * num_heads_q
                        # qo_cost = 0.0 if r == rank_m[i] else comm_len_m[i] * num_heads_q
                        # kv communication cost: if rank != rank_n[j], need communication,
                        # cost = comm_len_n[j] * num_heads_kv
                        # kv_cost = (
                        #     0.0 if r == rank_n[j] else comm_len_n[j] * num_heads_kv
                        # )
                        # Total cost = qo communication cost + kv communication cost
                        # total_cost = qo_cost + kv_cost
                        total_cost = 0.0 if (r == rank_m[i] or r == rank_n[j]) else 1.0
                    else:
                        # When cost flow is disabled, all costs are set to 0, equivalent to maximum flow
                        total_cost = 0.0
                    add_edge(f"r{r}", job_node, area, total_cost)
                add_edge(job_node, sink, area, 0.0)

        if total_job_area == 0:
            return True

        # ---- Minimum cost maximum flow (currently all costs are 0, equivalent to maximum flow) ----
        def min_cost_max_flow() -> tuple[float, float]:
            flow = 0.0
            cost = 0.0
            # Potential (if positive costs are set later, can use Dijkstra + potential optimization)
            potential: dict[str, float] = defaultdict(float)

            while True:
                # Use SPFA / Bellman-Ford to find shortest path (cost)
                dist: dict[str, float] = defaultdict(lambda: float("inf"))
                in_queue: dict[str, bool] = defaultdict(bool)
                prev_v: dict[str, str] = {}
                prev_e: dict[str, int] = {}

                dist[source] = 0.0
                q = deque([source])
                in_queue[source] = True

                while q:
                    u = q.popleft()
                    in_queue[u] = False
                    for idx, (v, cap_uv, cost_uv, rev, _) in enumerate(graph[u]):
                        if cap_uv <= 1e-9:
                            continue
                        # Effective cost with potential
                        new_dist = dist[u] + cost_uv + potential[u] - potential[v]
                        if new_dist < dist[v] - 1e-9:
                            dist[v] = new_dist
                            prev_v[v] = u
                            prev_e[v] = idx
                            if not in_queue[v]:
                                q.append(v)
                                in_queue[v] = True

                if sink not in prev_v:
                    break  # Cannot find augmenting path to sink

                # Update potential (here all costs are 0, potential is always 0,
                # but keep structure for future extension)
                for node in dist:
                    if dist[node] < float("inf"):
                        potential[node] += dist[node]

                # Calculate augmentable flow for this iteration
                add_flow = float("inf")
                v = sink
                while v != source:
                    u = prev_v[v]
                    e_idx = prev_e[v]
                    add_flow = min(add_flow, graph[u][e_idx][1])
                    v = u

                if add_flow <= 1e-9:
                    break

                # Backtrack to update residual network and accumulate cost
                v = sink
                while v != source:
                    u = prev_v[v]
                    e_idx = prev_e[v]
                    rev_idx = graph[u][e_idx][3]
                    graph[u][e_idx][1] -= add_flow
                    graph[v][rev_idx][1] += add_flow
                    cost += add_flow * graph[u][e_idx][2]
                    v = u

                flow += add_flow

            return flow, cost

        max_flow, _ = min_cost_max_flow()

        # Recover rank->job flow from residual graph to update solver_map,
        # select rank with maximum flow_rate
        for i in range(m):
            for j in range(n):
                job_node_val = job_nodes[i][j]
                if job_node_val is None or area_map[i][j] <= 0:
                    solver_map[i][j] = -1
                    continue
                job_node = job_node_val
                best_rank = -1
                best_flow_rate = -1.0
                for r in range(cp_size):
                    rn = f"r{r}"
                    for v, cap_uv, _, _, orig_cap in graph[rn]:
                        if v != job_node or orig_cap <= 0:
                            continue
                        # Flow = initial capacity - current capacity
                        flow_rj = orig_cap - cap_uv
                        if flow_rj > 1e-6:
                            flow_rate = flow_rj / area_map[i][j]
                            if flow_rate > best_flow_rate:
                                best_flow_rate = flow_rate
                                best_rank = r
                # if rank == 0 and best_flow_rate < 0.9:
                #     print(
                #         f"i: {i} j: {j} flow_rj: {flow_rj}, "
                #         f"best_flow_rate: {best_flow_rate}, best_rank: {best_rank}"
                #     )
                solver_map[i][j] = best_rank

        return abs(max_flow - total_job_area) <= 1e-3

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
        The solve method of the Simplex-Network-Flow dynamic dispatch algorithm

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
        unbanlance_rate = 1.001

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

        for _ in range(max_iters):
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
                selected_edges = self._Simplex(
                    cp_size,
                    edges,
                    mid,
                )
                solver_try = [[-1 for _ in range(n)] for _ in range(m)]
                # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
                use_cost_flow = False
                success = self._NetworkFlow(
                    cp_size,
                    edges,
                    selected_edges,
                    area_map,
                    solver_try,
                    rank_m,
                    rank_n,
                    area_avg,
                    unbanlance_rate,
                    comm_len_m,
                    comm_len_n,
                    num_heads_q,
                    num_heads_kv,
                    use_cost_flow,
                )
                if success:
                    if rank == 0 and self.debug_print:
                        print(f"iter: {_} attempt: {_attempt} success: {success}")
                    # Use this assignment as input for next attempt
                    solver_state = [row[:] for row in solver_try]
                    break

            if rank == 0 and self.debug_print:
                print(
                    f"iter: {_} low: {low}, high: {high}, mid: {mid}, success: {success}"
                )
            if success:
                best_map = solver_try
                best_edges = edges
                best_selected_edges = selected_edges
                high = mid
                solver_prev = solver_try
            else:
                low = mid
            if high - low <= eps * high and low > 0:
                break

        if best_map is not None:
            # Use optimal assignment obtained from binary search

            # solver_map = best_map
            if best_edges is None:
                raise RuntimeError("best_edges is None but best_map is not None")
            edges = best_edges
            selected_edges = (
                best_selected_edges if best_selected_edges is not None else []
            )
            # success = True

            # Try to use local as much as possible, run cost flow assignment again
            solver_try = [[-1 for _ in range(n)] for _ in range(m)]
            # Cost flow switch: True uses cost flow, False uses maximum flow (all costs are 0)
            unbalance_rate = 1.000
            use_cost_flow = True
            success = self._NetworkFlow(
                cp_size,
                edges,
                selected_edges,
                area_map,
                solver_try,
                rank_m,
                rank_n,
                area_avg,
                unbanlance_rate,
                comm_len_m,
                comm_len_n,
                num_heads_q,
                num_heads_kv,
                use_cost_flow,
            )
            solver_map = solver_try
            if not success:
                raise RuntimeError("Final Network Flow failed unexpectedly")
        else:
            # Record last attempt result for debugging
            solver_map = solver_try
            selected_edges = selected_edges  # From last iteration
            success = False

        if not success:
            if self.debug_print:
                print(
                    "[SNFDynamicAttnAlgorithm] network flow failed, fallback to Q owner assignment"
                )

        # TODO: Implement Simplex-Network-Flow algorithm here
        # This is a placeholder implementation that assigns remaining jobs
        # to the rank that owns the Q tile (rank_m[i])
        for i in range(m):
            for j in range(n):
                if solver_map[i][j] == -1 and grid_rects[i][j].area() > 0:
                    if self.debug_print:
                        print(f"no assign job: i: {i} j: {j}")
                    solver_map[i][j] = rank_m[i]

        # Calculate result stage
        for i in range(m):
            for j in range(n):
                if solver_map[i][j] != -1:
                    bucket_per_rank[solver_map[i][j]].extend(grid_rects[i][j])

        # Statistics for load balancing
        rank_area = [0.0 for _ in range(cp_size)]
        for i in range(m):
            for j in range(n):
                if solver_map[i][j] != -1:
                    rank_area[solver_map[i][j]] += area_map[i][j]

        max_area = max(rank_area) if rank_area else 0.0
        unbalance_rate = max_area / area_avg if area_avg > 0 else 0.0

        if rank == 0 and self.debug_print:
            print(
                f"[SNFDynamicAttnAlgorithm] load balance: "
                f"max_area={max_area:.2f}, area_avg={area_avg:.2f}, "
                f"unbalance_rate={unbalance_rate:.4f}"
            )

        if rank == 0 and start_time is not None and self.debug_print:
            elapsed = time.perf_counter() - start_time
            print(f"[SNFDynamicAttnAlgorithm] solve elapsed time: {elapsed:.6f}s")
