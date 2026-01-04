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

import os

import torch
import torch.distributed._symmetric_memory as symm_mem

from magi_attention.comm.primitive.device_a2av import OnDeviceA2AVMgr, on_device_a2av
from magi_attention.comm.primitive.grpcoll.utils import are_all_ranks_on_same_host
from magi_attention.utils import clearup_dist_env, rprint_rank, setup_dist_env

# --------------      setup       -------------- #

(
    rank,
    local_rank,
    world_size,
    num_nodes,
    num_local_ranks,
    world_group,
    device,
    seed,
) = setup_dist_env(base_seed=0, seed_bias=lambda rank: rank)

assert world_size == 4

assert are_all_ranks_on_same_host(
    world_group
), "inter-node group is not supported for ngc-2505 version of symm_mem"

# `set_backend` and inter-node a2av is not supported for ngc-2505 version of symm_mem
# which is enabled when nvshmem is integrated as a backend from this pr: https://github.com/pytorch/pytorch/pull/151261
# Set NVSHMEM as SymmMem backend
# symm_mem.set_backend("NVSHMEM")

world_group_name = world_group.group_name
symm_mem.enable_symm_mem_for_group(world_group_name)
assert symm_mem.is_symm_mem_enabled_for_group(world_group_name)

hidden_size = 8
max_seqlen = 16
topk = 2
overflow = 2  # Assuming worst case, 2x tokens are routed to one EP rank
dtype = torch.bfloat16

# --------------      test       -------------- #

successed = True

# ---- prepare test case ---- #

test_case = os.environ["TEST_CASE"]

rprint_rank(f"Running test case: {test_case}\n")

match test_case:
    case "naive_a2a":
        send_buffer_per_rank = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ]
        expected_recv_buffer_per_rank = [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
        ]
        input_split_size_list_per_rank = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        output_split_size_list_per_rank = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        dst_indices_list_per_rank = [
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
        ]
        src_index_list_per_rank = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    case "naive_a2a_v":
        send_buffer_per_rank = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29, 30, 31],
        ]
        expected_recv_buffer_per_rank = [
            [0, 1, 8, 16, 17, 18, 24, 25],
            [2, 3, 9, 10, 11, 19, 26],
            [4, 5, 12, 20, 27, 28],
            [6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31],
        ]
        input_split_size_list_per_rank = [
            [2, 2, 2, 2],
            [1, 3, 1, 3],
            [3, 1, 1, 3],
            [2, 1, 2, 3],
        ]
        output_split_size_list_per_rank = [
            [2, 1, 3, 2],
            [2, 3, 1, 1],
            [2, 1, 1, 2],
            [2, 3, 3, 3],
        ]
        dst_indices_list_per_rank = [
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
            [[0], [1], [2], [3]],
        ]
        src_index_list_per_rank = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    case _:
        raise ValueError(f"Unknown test case: {test_case}")

# prepare meta args for this rank
input_split_size_list = input_split_size_list_per_rank[rank]
output_split_size_list = output_split_size_list_per_rank[rank]
dst_indices_list = dst_indices_list_per_rank[rank]
src_index_list = src_index_list_per_rank[rank]

# prepare buffers
send_buffer = (
    torch.tensor(
        send_buffer_per_rank[rank],
        dtype=dtype,
        device=device,
    )
    .view(-1, 1)
    .expand(-1, hidden_size)
)
rprint_rank(f"{send_buffer=}\n")

expected_recv_buffer = (
    torch.tensor(
        expected_recv_buffer_per_rank[rank],
        dtype=dtype,
        device=device,
    )
    .view(-1, 1)
    .expand(-1, hidden_size)
)
rprint_rank(f"{expected_recv_buffer=}\n")

recv_buffer = torch.full_like(
    expected_recv_buffer,
    fill_value=-1,
    dtype=dtype,
    device=device,
)
rprint_rank(f"{recv_buffer=}\n")


# ---- initialize the OnDeviceA2AV ---- #

OnDeviceA2AVMgr.initialize(
    max_input_seqlen=max_seqlen * topk,
    hidden_size=hidden_size,
    dtype=dtype,
    device=device,
    overflow=overflow,
)


# ---- prepare input_splits tensor on device ---- #

input_splits = torch.tensor(input_split_size_list, dtype=torch.int64, device=device)
rprint_rank(f"{input_splits=} | {input_split_size_list=}\n")


# ---- launch all-to-all-v triton kernel ---- #

gathered_tokens, output_splits = on_device_a2av(
    input=send_buffer,
    # input_splits=input_splits, # pass tensor directly
    input_splits=input_split_size_list,  # pass list and transfer to tensor inside
    output=recv_buffer,
    # received_num_tokens=-1, # unknown mode, causing a gpu-cpu sync
    received_num_tokens=sum(output_split_size_list),  # known mode
    group=world_group,
)


# ---- assert close to expected output_splits ---- #

rprint_rank(f"{output_splits=} | {output_split_size_list=}\n")

try:
    assert output_splits.tolist() == output_split_size_list
    rprint_rank("PASS for output_splits\n")
except AssertionError:
    rprint_rank("FAIL for output_splits\n")
    successed = False


# ---- assert close to expected output ---- #

rprint_rank(f"{gathered_tokens=}\n{expected_recv_buffer=}\n")

try:
    torch.testing.assert_close(gathered_tokens, expected_recv_buffer)
    rprint_rank("PASS for output\n")
except AssertionError:
    rprint_rank("FAIL for output\n")
    successed = False


# ---- check inplace ---- #

try:
    assert gathered_tokens.data_ptr() == recv_buffer.data_ptr()
    rprint_rank("PASS for inplace\n")
except AssertionError:
    rprint_rank("FAIL for inplace\n")
    successed = False


# ---- finalize the OnDeviceA2AV ---- #

OnDeviceA2AVMgr.finalize()


# --------------      clearup env       -------------- #

clearup_dist_env()


# --------------      exit       -------------- #

exit(0 if successed else 1)
