#
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
#


import copy
import random


def eval_greedy_algorithm(
    solver_map: list,
    cp_size: int,
    qk_rate: float,
) -> float:
    comm_max = 0.0
    qo_lhrc: list[set] = [set() for _ in range(cp_size)]
    kv_lhrc: list[set] = [set() for _ in range(cp_size)]
    qo_lcrh: list[set] = [set() for _ in range(cp_size)]
    kv_lcrh: list[set] = [set() for _ in range(cp_size)]
    for i in range(cp_size):
        for j in range(cp_size):
            rank = solver_map[i][j]
            if i != rank:
                qo_lhrc[i].add(rank)
                qo_lcrh[rank].add(i)
            if j != rank:
                kv_lhrc[j].add(rank)
                kv_lcrh[rank].add(j)
    for i in range(cp_size):
        fwd_send_len = (
            qk_rate * len(qo_lhrc[i]) + len(kv_lhrc[i]) * 2 + qk_rate * len(qo_lcrh[i])
        )
        fwd_recv_len = (
            qk_rate * len(qo_lcrh[i]) + len(kv_lcrh[i]) * 2 + qk_rate * len(qo_lhrc[i])
        )
        comm_max = max(fwd_send_len, fwd_recv_len)
    return comm_max


def greedy_algorithm(
    solver_map: list,
    job_list: list,
    cp_size: int,
) -> list:
    row: list[set] = [set() for _ in range(cp_size)]
    col: list[set] = [set() for _ in range(cp_size)]
    cnt = [0 for _ in range(cp_size)]
    for i in range(cp_size):
        for j in range(cp_size):
            if solver_map[i][j] != -1:
                rank = solver_map[i][j]
                row[i].add(rank)
                col[j].add(rank)
                cnt[rank] += 1
    rest_option = []
    for i in range(cp_size):
        rest_option.extend([i] * (cp_size - cnt[i]))
    output_map = copy.deepcopy(solver_map)
    for i, j in job_list:
        intersection = row[i] & col[j]
        rank_choice_list = [rank for rank in intersection if cnt[rank] < cp_size]
        rank_choice = -1
        if len(rank_choice_list) > 0:
            # rank appears in both rows and columns
            # allowing tasks to be allocated without communication overhead
            weights_list = [cp_size - cnt[rank] for rank in rank_choice_list]
            rank_choice = random.choices(rank_choice_list, weights=weights_list, k=1)[0]
        else:
            # no rank appears in both rows and columns
            # take a random rank
            rank_choice = random.choice(rest_option)
            row[i].add(rank_choice)
            col[j].add(rank_choice)
        rest_option.remove(rank_choice)
        cnt[rank_choice] += 1
        output_map[i][j] = rank_choice
    return output_map


def solve_full_mask_by_greedy_algorithm(
    seqlen_q: int,
    seqlen_k: int,
    cp_size: int,
    qk_rate: float,
) -> list:
    solver_map = [[-1 for _ in range(cp_size)] for _ in range(cp_size)]
    for i in range(cp_size):
        solver_map[i][i] = i
    job_list = []
    for i in range(cp_size):
        for j in range(cp_size):
            if i != j:
                job_list.append((i, j))
    # random initialize stage
    random_times = cp_size * cp_size
    local_optimal_solution = greedy_algorithm(solver_map, job_list, cp_size)
    local_eval = eval_greedy_algorithm(local_optimal_solution, cp_size, qk_rate)
    for _ in range(random_times):
        random.shuffle(job_list)
        new_solver_map = greedy_algorithm(solver_map, job_list, cp_size)
        new_solver_eval = eval_greedy_algorithm(new_solver_map, cp_size, qk_rate)
        if new_solver_eval < local_eval:
            print(new_solver_eval, local_eval)
            local_optimal_solution = new_solver_map
            local_eval = new_solver_eval
    # refinement stage
    refine_times = cp_size * cp_size
    for iter in range(refine_times):
        random.shuffle(job_list)
        solver_map = copy.deepcopy(local_optimal_solution)
        refinement_position_num = len(job_list) * (refine_times - iter) // refine_times
        if refinement_position_num == 0:
            break
        for pos_id in range(refinement_position_num):
            i, j = job_list[pos_id]
            solver_map[i][j] = -1
        partial_job_list = job_list[:refinement_position_num]
        random_times = cp_size
        for _ in range(random_times):
            random.shuffle(partial_job_list)
            new_solver_map = greedy_algorithm(solver_map, partial_job_list, cp_size)
            new_solver_eval = eval_greedy_algorithm(new_solver_map, cp_size, qk_rate)
            if new_solver_eval < local_eval:
                print(new_solver_eval, local_eval)
                local_optimal_solution = new_solver_map
                local_eval = new_solver_eval
    return local_optimal_solution


if __name__ == "__main__":
    # num_heads_q = 12        # number of attention (query) heads
    # num_heads_kv = 4       # number of key/value heads (GQA)
    # head_dim = 128         # dimension of each attention head
    # dtype = torch.bfloat16
    # device = "cuda"

    # bucket = [
    #     [
    #         [[0, 1024]],
    #         [[1024, 2048]],
    #         [AttnMaskType.FULL],
    #     ],
    #     [
    #         [[1024, 2048]],
    #         [[1024, 2048]],
    #         [AttnMaskType.FULL],
    #     ],
    #     [
    #         [[2048, 3072]],
    #         [[2048, 3072]],
    #         [AttnMaskType.FULL],
    #     ],
    #     [
    #         [[3072, 4096]],
    #         [[3072, 4096]],
    #         [AttnMaskType.FULL],
    #     ],
    # ]
    # local_ranges = [[[0, 1024]], [[1024, 2048]], [[2048, 3072]], [[3072, 4096]]]

    # kBlockM = 128
    # kBlockN = 128
    # eval_solver_result(bucket, local_ranges, kBlockM, kBlockN)
    cp_size = 64
    qk_rate = 1.0
    seqlen_q = 1024
    seqlen_k = 1024

    my_map = solve_full_mask_by_greedy_algorithm(seqlen_q, seqlen_k, cp_size, qk_rate)
    print("my solution:")
    for line in my_map:
        print(line)
    baseline_map = [[i for _ in range(cp_size)] for i in range(cp_size)]
    eval_baseline = eval_greedy_algorithm(baseline_map, cp_size, qk_rate)
    eval_my = eval_greedy_algorithm(my_map, cp_size, qk_rate)
    print(f"baseline_comm: {eval_baseline}")
    print(f"my_comm: {eval_my}")
    print(f"comm_cut_rate: {eval_my / eval_baseline}")
    cnt = [0 for _ in range(cp_size)]
    for line in my_map:
        for rank in line:
            cnt[rank] += 1
    print("load_balance_state:")
    print(cnt)
