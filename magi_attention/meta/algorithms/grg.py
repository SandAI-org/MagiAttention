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

import copy
import random

from magi_attention.common import AttnRange, AttnRanges, AttnRectangles
from magi_attention.common.enum import DynamicAttnAlgType

from .base import DynamicAttnAlgorithm


class GRGDynamicAttnAlgorithm(DynamicAttnAlgorithm):
    """The greedy-random-grid dynamic dispatch algorithm implementation"""

    def __init__(self):
        """
        The init method of the greedy-random-grid dynamic dispatch algorithm
        """
        pass

    @property
    def type(self) -> DynamicAttnAlgType:
        return DynamicAttnAlgType.GREEDY_RANDOM_GRID

    def _get_grid_rects(
        self,
        rects: AttnRectangles,
        indexed_host_ranges_q: list[tuple[AttnRange, int]],
        indexed_host_ranges_k: list[tuple[AttnRange, int]],
    ) -> list[list[AttnRectangles]]:
        # print(f"rects: {rects}")
        # print(f"indexed_host_ranges_q: {indexed_host_ranges_q}")
        # print(f"indexed_host_ranges_k: {indexed_host_ranges_k}")
        rest_rects = rects
        grid_rects = []
        # cut Q tiles
        q_cut_pos = 0
        for item in indexed_host_ranges_q:
            q_tile_range: AttnRange = item[0]
            # q_host_rank: int = item[1]
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
                # k_host_rank: int = item[1]
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

    def _eval_greedy_algorithm(
        self,
        cp_size: int,
        m: int,
        n: int,
        rank_m: list[int],
        rank_n: list[int],
        comm_len_m: list[int],
        comm_len_n: list[int],
        eval_solver_map: list[list[int]],
        area_map: list[list[int]],
    ) -> float:
        # calc the area max
        area_per_rank = [0 for _ in range(cp_size)]
        for i in range(m):
            for j in range(n):
                rank = eval_solver_map[i][j]
                if rank != -1:
                    area_per_rank[rank] += area_map[i][j]

        area_max = max(area_per_rank)

        # calc the communication max
        comm_max = 0.0
        qo_lhrc: list[set] = [set() for _ in range(m)]
        kv_lhrc: list[set] = [set() for _ in range(n)]
        qo_lcrh: list[set] = [set() for _ in range(cp_size)]
        kv_lcrh: list[set] = [set() for _ in range(cp_size)]
        for i in range(m):
            for j in range(n):
                rank = eval_solver_map[i][j]
                if rank != -1:
                    if rank_m[i] != rank:
                        qo_lhrc[i].add(rank)
                        qo_lcrh[rank].add(i)
                    if rank_n[j] != rank:
                        kv_lhrc[j].add(rank)
                        kv_lcrh[rank].add(j)

        qo_lhrc_len = [0 for _ in range(cp_size)]
        kv_lhrc_len = [0 for _ in range(cp_size)]
        qo_lcrh_len = [0 for _ in range(cp_size)]
        kv_lcrh_len = [0 for _ in range(cp_size)]
        # calc the communication length of local-hold-remote-calc
        for i in range(m):
            qo_lhrc_len[rank_m[i]] += len(qo_lhrc[i]) * comm_len_m[i]
        for i in range(n):
            kv_lhrc_len[rank_n[i]] += len(kv_lhrc[i]) * comm_len_n[i]
        # calc the communication length of local-calc-remote-hold
        for i in range(cp_size):
            for j in qo_lcrh[i]:
                qo_lcrh_len[i] += comm_len_m[j]
            for j in kv_lcrh[i]:
                kv_lcrh_len[i] += comm_len_n[j]

        for i in range(cp_size):
            fwd_send_len = qo_lhrc_len[i] + kv_lhrc_len[i] * 2 + qo_lcrh_len[i]
            fwd_recv_len = qo_lcrh_len[i] + kv_lcrh_len[i] * 2 + qo_lhrc_len[i]
            comm_max = max(comm_max, fwd_send_len, fwd_recv_len)

        # cost function: communication cost max * area max
        return comm_max * area_max

    def _greedy_algorithm(
        self,
        m: int,
        n: int,
        cp_size: int,
        rank_m: list[int],
        rank_n: list[int],
        comm_len_m: list[int],
        comm_len_n: list[int],
        solver_map: list[list[int]],
        area_map: list[list[int]],
        grid_rects: list[list[AttnRectangles]],
        job_list: list[tuple[int, int]],
    ) -> list[list[int]]:
        # initialize the row rank set, col rank set, rank calc area, and total area
        row: list[set] = [set() for _ in range(m)]
        col: list[set] = [set() for _ in range(n)]
        rank_calc_area = [0 for _ in range(cp_size)]
        total_area = 0
        for i in range(m):
            for j in range(n):
                total_area += area_map[i][j]
                if solver_map[i][j] != -1:
                    rank = solver_map[i][j]
                    row[i].add(rank)
                    col[j].add(rank)
                    rank_calc_area[rank] += area_map[i][j]
        output_map = copy.deepcopy(solver_map)

        # greedy algorithm
        for i, j in job_list:
            intersection = row[i] & col[j]
            rank_choice_list = [rank for rank in intersection]
            rank_choice = -1
            if len(rank_choice_list) > 0:
                # greedy choices: no communication cost
                # rank appears in both rows and columns
                # TODO: dynamic fix max_area
                max_area = (
                    max([rank_calc_area[rank] for rank in rank_choice_list]) * 1.05
                )
                weights_list = [
                    max_area - rank_calc_area[rank] for rank in rank_choice_list
                ]
                rank_choice = random.choices(
                    rank_choice_list, weights=weights_list, k=1
                )[0]
            else:
                # no rank appears in both rows and columns
                # take a random rank
                # make some heuristic choices
                rank_choice_list = [i for i in range(cp_size)]
                max_area = (
                    max([rank_calc_area[rank] for rank in rank_choice_list]) * 1.05
                )
                # normalize the weights
                sum_area_weight = sum(
                    [max_area - rank_calc_area[rank] for rank in rank_choice_list]
                )
                weights_list = [
                    (max_area - rank_calc_area[rank]) / sum_area_weight
                    for rank in rank_choice_list
                ]
                # normalize the communication length weight
                sum_comm_len_weight = sum([comm_len_m[rank] for rank in row[i]]) + sum(
                    [comm_len_n[rank] for rank in col[j]]
                )
                for rank in row[i]:
                    weights_list[rank] += comm_len_m[rank] / sum_comm_len_weight
                for rank in col[j]:
                    weights_list[rank] += comm_len_n[rank] / sum_comm_len_weight
                rank_choice = random.choices(
                    rank_choice_list, weights=weights_list, k=1
                )[0]
            output_map[i][j] = rank_choice
            row[i].add(rank_choice)
            col[j].add(rank_choice)
            rank_calc_area[rank_choice] += area_map[i][j]

        return output_map

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
        The solve method of the greedy-random-grid dynamic dispatch algorithm

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
        grid_rects = self._get_grid_rects(
            rects, indexed_host_ranges_q, indexed_host_ranges_k
        )

        # for i in range(len(grid_rects)):
        #     for j in range(len(grid_rects[i])):
        #         print(f"grid_rects[{i}][{j}]: {grid_rects[i][j]}")

        m = len(grid_rects)
        n = len(grid_rects[0])

        # solve the grid
        solver_map = [[-1 for _ in range(n)] for _ in range(m)]
        area_map = [[-1 for _ in range(n)] for _ in range(m)]
        job_list = []

        # initialize the solver_map and job_list
        for i in range(m):
            for j in range(n):
                area_map[i][j] = grid_rects[i][j].area()
                if rank_m[i] == rank_n[j]:
                    # greedy choices: host job, no communication cost
                    solver_map[i][j] = rank_m[i]
                else:
                    solver_map[i][j] = -1
                    if grid_rects[i][j].area() > 0:
                        job_list.append((i, j))

        # random initialize stage
        random_times = 100000 // m // n
        local_eval = float("inf")
        for _ in range(random_times):
            new_solver_map = self._greedy_algorithm(
                m,
                n,
                cp_size,
                rank_m,
                rank_n,
                comm_len_m,
                comm_len_n,
                solver_map,
                area_map,
                grid_rects,
                job_list,
            )
            new_solver_eval = self._eval_greedy_algorithm(
                cp_size,
                m,
                n,
                rank_m,
                rank_n,
                comm_len_m,
                comm_len_n,
                new_solver_map,
                area_map,
            )
            if new_solver_eval < local_eval:
                local_eval = new_solver_eval
                local_optimal_solver_map = new_solver_map
            random.shuffle(job_list)

        # refinement stage
        refine_times = 100000 // m // n
        for iter in range(refine_times):
            random.shuffle(job_list)
            solver_map = copy.deepcopy(local_optimal_solver_map)
            refinement_position_num = (
                len(job_list) * (refine_times - iter) // refine_times
            )
            if refinement_position_num == 0:
                break
            # TODO: make some heuristics positions choices
            for pos_id in range(refinement_position_num):
                i, j = job_list[pos_id]
                solver_map[i][j] = -1
            partial_job_list = job_list[:refinement_position_num]

            # process greedy algorithm on partial job list
            random_times = 10
            for _ in range(random_times):
                random.shuffle(partial_job_list)
                new_solver_map = self._greedy_algorithm(
                    m,
                    n,
                    cp_size,
                    rank_m,
                    rank_n,
                    comm_len_m,
                    comm_len_n,
                    solver_map,
                    area_map,
                    grid_rects,
                    partial_job_list,
                )
                new_solver_eval = self._eval_greedy_algorithm(
                    cp_size,
                    m,
                    n,
                    rank_m,
                    rank_n,
                    comm_len_m,
                    comm_len_n,
                    new_solver_map,
                    area_map,
                )
                if new_solver_eval < local_eval:
                    local_eval = new_solver_eval
                    local_optimal_solver_map = new_solver_map

        # print(f"local_optimal_solver_map: {local_optimal_solver_map}")

        # calc result stage
        for i in range(m):
            for j in range(n):
                if local_optimal_solver_map[i][j] != -1:
                    bucket_per_rank[local_optimal_solver_map[i][j]].extend(
                        grid_rects[i][j]
                    )
