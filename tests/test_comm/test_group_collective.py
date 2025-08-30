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

from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.comm.primitive import group_cast_collective, group_reduce_collective
from magi_attention.comm.primitive.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
    sanity_check_for_group_reduce_meta_args_per_rank,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import assert_close
from magi_attention.testing.utils import switch_envvar_context


class TestGroupCollective(DistTestBase):
    def init_pg(self):
        super().init_pg()

        # -----    set up for hier comm   ---- #

        self._switch_hier_comm_context = partial(
            switch_envvar_context, envvar_name="MAGI_ATTENTION_HIERARCHICAL_COMM"
        )

        world_size_inter_node, world_size_intra_node = {
            1: (1, 1),
            2: (1, 2),
            3: (3, 1),
            4: (2, 2),
            5: (1, 5),
            6: (3, 2),
            7: (1, 7),
            8: (2, 4),
        }[self.world_size]
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size_inter_node, world_size_intra_node),
            mesh_dim_names=("inter", "intra"),
        )
        self.intra_group = device_mesh.get_group("intra")
        self.inter_group = device_mesh.get_group("inter")

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        # TODO: add test cases for world size > 4
        "test_case",
        [
            {
                "name": "naive_a2a",
                "world_size": 4,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "expected_recv_buffer_per_rank": [
                    [0, 4, 8, 12],
                    [1, 5, 9, 13],
                    [2, 6, 10, 14],
                    [3, 7, 11, 15],
                ],
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "output_split_size_list_per_rank": [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "dst_indices_list_per_rank": [
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                ],
                "src_index_list_per_rank": [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ],
            },
            {
                "name": "naive_a2a_v",
                "world_size": 4,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20, 21, 22, 23],
                    [24, 25, 26, 27, 28, 29, 30, 31],
                ],
                "expected_recv_buffer_per_rank": [
                    [0, 1, 8, 16, 17, 18, 24, 25],
                    [2, 3, 9, 10, 11, 19, 26],
                    [4, 5, 12, 20, 27, 28],
                    [6, 7, 13, 14, 15, 21, 22, 23, 29, 30, 31],
                ],
                "input_split_size_list_per_rank": [
                    [2, 2, 2, 2],
                    [1, 3, 1, 3],
                    [3, 1, 1, 3],
                    [2, 1, 2, 3],
                ],
                "output_split_size_list_per_rank": [
                    [2, 1, 3, 2],
                    [2, 3, 1, 1],
                    [2, 1, 1, 2],
                    [2, 3, 3, 3],
                ],
                "dst_indices_list_per_rank": [
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                ],
                "src_index_list_per_rank": [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ],
            },
            {
                "name": "normal_group_cast_case1",
                "world_size": 4,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 3, 6, 7, 14, 15],
                    [4, 5, 8, 9],
                ],
                "input_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                "output_split_size_list_per_rank": [
                    [1, 1, 1],
                    [2, 2, 1, 1, 1],
                    [1, 1, 2, 2],
                    [1, 1, 1, 1],
                ],
                "dst_indices_list_per_rank": [
                    [[1], [1, 2], [2]],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
                "src_index_list_per_rank": [
                    [1, 2, 3],
                    [0, 2, 0, 3, 3],
                    [0, 0, 1, 3],
                    [1, 1, 2, 2],
                ],
            },
            {
                "name": "normal_group_cast_case2",
                "world_size": 4,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13, 0, 1],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 6, 7, 14, 15],
                    [8, 9, 4, 5],
                ],
                "input_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                "output_split_size_list_per_rank": [
                    [1, 1, 1, 2],
                    [2, 2, 1, 1, 1],
                    [1, 2, 2],
                    [1, 1, 1, 1],
                ],
                "dst_indices_list_per_rank": [
                    [[0, 1], [1, 2], []],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
                "src_index_list_per_rank": [
                    [1, 2, 3, 0],
                    [0, 2, 0, 3, 3],
                    [0, 1, 3],
                    [2, 2, 1, 1],
                ],
            },
        ],
    )
    @parameterize("use_hier_comm", [False, True])
    @parameterize("async_op", [True])  # skip async_op=False to speed up
    def test_group_cast_collective(
        self,
        test_case: dict[str, Any],
        async_op: bool,
        use_hier_comm: bool,
    ):
        # skip for unmatched world size
        if self.world_size != test_case["world_size"]:
            return

        # skip for hier comm
        if use_hier_comm:
            if not async_op:
                return

        # sanity check for meta args per rank
        input_split_size_list_per_rank = test_case["input_split_size_list_per_rank"]
        output_split_size_list_per_rank = test_case["output_split_size_list_per_rank"]
        dst_indices_list_per_rank = test_case["dst_indices_list_per_rank"]
        src_index_list_per_rank = test_case["src_index_list_per_rank"]
        sanity_check_for_group_cast_meta_args_per_rank(
            input_split_size_list_per_rank=input_split_size_list_per_rank,
            output_split_size_list_per_rank=output_split_size_list_per_rank,
            dst_indices_list_per_rank=dst_indices_list_per_rank,
            src_index_list_per_rank=src_index_list_per_rank,
            world_size=self.world_size,
            check_nccl_send_recv=True,
        )

        # prepare meta args for this rank
        input_split_size_list = input_split_size_list_per_rank[self.rank]
        output_split_size_list = output_split_size_list_per_rank[self.rank]
        dst_indices_list = dst_indices_list_per_rank[self.rank]
        src_index_list = src_index_list_per_rank[self.rank]

        # prepare buffers
        send_buffer = torch.tensor(
            test_case["send_buffer_per_rank"][self.rank],
            dtype=self.dtype,
            device=self.device,
        )
        expected_recv_buffer = torch.tensor(
            test_case["expected_recv_buffer_per_rank"][self.rank],
            dtype=self.dtype,
            device=self.device,
        )
        recv_buffer = torch.full_like(
            expected_recv_buffer,
            fill_value=-1,
            dtype=self.dtype,
            device=self.device,
        )

        # run group-cast comm kernel
        with self._switch_hier_comm_context(enable=use_hier_comm):
            assert (
                not use_hier_comm or magi_attention.comm.is_hierarchical_comm_enable()
            )
            work = group_cast_collective(
                input=send_buffer,
                output=recv_buffer,
                input_split_size_list=input_split_size_list,
                output_split_size_list=output_split_size_list,
                dst_indices_list=dst_indices_list,
                src_index_list=src_index_list,
                group=self.process_group,
                async_op=async_op,
                # NOTE: args below for hierarchical comm
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )

            # post process
            recv_buffer = work.wait_post_process(recv_buffer)

        # check results
        err_msg_list = []
        try:
            assert_close(
                recv_buffer,
                expected_recv_buffer,
                atol=1e-8,
                rtol=1e-6,
                test_case="group-cast recv buffer",
            )
        except Exception as e:
            err_msg_list.append(
                f"Group-Cast collective has failed due to error: \n{e}\n"
                f"with: \n{recv_buffer=}\n{expected_recv_buffer=}\n"
            )

        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        # TODO: add test cases for world size > 4
        "test_case",
        [
            {
                "name": "naive_a2a_like_sum_reduce",
                "world_size": 4,
                "reduce_op": "sum",
                "send_buffer_per_rank": [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                ],
                "recv_buffer_before_reduce_per_rank": [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                ],
                "expected_recv_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [3, 4, 5, 6],
                ],
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "output_split_size_list_per_rank": [
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                "dst_index_list_per_rank": [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ],
                "src_indices_list_per_rank": [
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                    [[0], [1], [2], [3]],
                ],
            },
            {
                "name": "normal_group_sum_reduce",
                "world_size": 4,
                "reduce_op": "sum",
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21],
                ],
                "recv_buffer_before_reduce_per_rank": [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                ],
                "expected_recv_buffer_per_rank": [
                    [9, 11, 22, 20],
                    [19, 20, 15, 16],
                    [23, 25, 10, 11],
                    [14, 17, 19, 20],
                ],
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 2],
                    [2, 2, 1, 1, 1],
                    [1, 2, 2],
                    [1, 1, 1, 1, 1],
                ],
                "output_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                "dst_index_list_per_rank": [
                    [1, 2, 3, 0],
                    [0, 2, 0, 3, 3],
                    [0, 1, 3],
                    [1, 1, 0, 2, 2],
                ],
                "src_indices_list_per_rank": [
                    [[0, 1], [1, 2], [3]],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
            },
            {
                "name": "normal_group_avg_reduce",
                "world_size": 4,
                "reduce_op": "avg",
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21],
                ],
                "recv_buffer_before_reduce_per_rank": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "expected_recv_buffer_per_rank": [
                    [4, 5, 10.5, 19],
                    [17, 9, 13, 14],
                    [20, 11, 7, 8],
                    [10, 6.5, 15, 16],
                ],
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 2],
                    [2, 2, 1, 1, 1],
                    [1, 2, 2],
                    [1, 1, 1, 1, 1],
                ],
                "output_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                "dst_index_list_per_rank": [
                    [1, 2, 3, 0],
                    [0, 2, 0, 3, 3],
                    [0, 1, 3],
                    [1, 1, 0, 2, 2],
                ],
                "src_indices_list_per_rank": [
                    [[0, 1], [1, 2], [3]],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
            },
            {
                "name": "normal_group_lse_reduce",
                "world_size": 4,
                "reduce_op": "lse",
                "num_heads": 2,
                "head_dim": 3,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21],
                ],
                "send_lse_buffer_per_rank": [
                    [0, 1, 2, -1, -2],
                    [0.5, 1.5, 2.5, 0.0, -1.5, -2.5, -3.5],
                    [0.25, 1.25, 2.25, -1.25, -2.25],
                    [0.75, 1.75, 2.75, -1.75, -2.75],
                ],
                "recv_buffer_before_reduce_per_rank": [
                    [1, 1, 1, 1],
                    [2, 2, 2, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                ],
                "recv_lse_buffer_before_reduce_per_rank": [
                    [float("-inf"), float("-inf"), float("-inf"), float("-inf")],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [-1, -1, -1, -1],
                ],
                "expected_recv_buffer_per_rank": [
                    [4.6250, 5.9375, 11.5625, 19.0],
                    [12.1875, 13.6250, 10.5625, 12.8750],
                    [4.0312, 2.2188, 6.2812, 4.3438],
                    [5.0938, 2.1250, 8.8125, 6.6875],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [0.7014, 1.5298, 0.4102, 2.75],
                    [1.1369, 2.0483, 1.5019, 2.3502],
                    [1.0620, 1.7048, 2.7014, 1.3133],
                    [-0.7986, 2.0525, -0.4241, -0.7481],
                ],
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 2],
                    [2, 2, 1, 1, 1],
                    [1, 2, 2],
                    [1, 1, 1, 1, 1],
                ],
                "output_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                "dst_index_list_per_rank": [
                    [1, 2, 3, 0],
                    [0, 2, 0, 3, 3],
                    [0, 1, 3],
                    [1, 1, 0, 2, 2],
                ],
                "src_indices_list_per_rank": [
                    [[0, 1], [1, 2], [3]],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
            },
        ],
    )
    @parameterize("use_hier_comm", [False, True])
    @parameterize("deterministic", [False, True])
    @parameterize("async_op", [True])  # skip async_op=False to speed up
    def test_group_reduce_collective(
        self,
        test_case: dict[str, Any],
        use_hier_comm: bool,
        deterministic: bool,
        async_op: bool,
    ):
        test_case_name = test_case["name"]
        reduce_op = test_case["reduce_op"]
        is_lse_reduce = reduce_op == "lse"

        # skip for unmatched world size
        if self.world_size != test_case["world_size"]:
            return

        # skip for hier comm
        if use_hier_comm:
            # TODO: support hier comm as a sync op
            if not async_op:
                return
            # TODO: support hier comm for avg/lse reduce
            if reduce_op != "sum":
                return

        # sanity check for meta args per rank
        input_split_size_list_per_rank = test_case["input_split_size_list_per_rank"]
        output_split_size_list_per_rank = test_case["output_split_size_list_per_rank"]
        dst_index_list_per_rank = test_case["dst_index_list_per_rank"]
        src_indices_list_per_rank = test_case["src_indices_list_per_rank"]
        sanity_check_for_group_reduce_meta_args_per_rank(
            input_split_size_list_per_rank=input_split_size_list_per_rank,
            output_split_size_list_per_rank=output_split_size_list_per_rank,
            dst_index_list_per_rank=dst_index_list_per_rank,
            src_indices_list_per_rank=src_indices_list_per_rank,
            world_size=self.world_size,
            check_nccl_send_recv=True,
        )

        # prepare meta args for this rank
        input_split_size_list = input_split_size_list_per_rank[self.rank]
        output_split_size_list = output_split_size_list_per_rank[self.rank]
        dst_index_list = dst_index_list_per_rank[self.rank]
        src_indices_list = src_indices_list_per_rank[self.rank]

        # prepare buffers
        send_buffer = torch.tensor(
            test_case["send_buffer_per_rank"][self.rank],
            dtype=self.dtype,
            device=self.device,
        )
        recv_buffer_before_reduce = torch.tensor(
            test_case["recv_buffer_before_reduce_per_rank"][self.rank],
            dtype=self.dtype,
            device=self.device,
        )
        expected_recv_buffer = torch.tensor(
            test_case["expected_recv_buffer_per_rank"][self.rank],
            dtype=self.dtype,
            device=self.device,
        )
        if is_lse_reduce:
            # for now, lse-reduce requires strictly on shape:
            # send/recv buffer: [seqlen, num_heads, head_dim]
            # send/recv lse buffer: [seqlen, num_heads]
            nh, hd = test_case["num_heads"], test_case["head_dim"]

            send_buffer = send_buffer.repeat_interleave(repeats=nh * hd, dim=0).reshape(
                -1, nh, hd
            )
            recv_buffer_before_reduce = recv_buffer_before_reduce.repeat_interleave(
                repeats=nh * hd, dim=0
            ).reshape(-1, nh, hd)
            expected_recv_buffer = expected_recv_buffer.repeat_interleave(
                repeats=nh * hd, dim=0
            ).reshape(-1, nh, hd)

            send_lse_buffer = (
                torch.tensor(
                    test_case["send_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=nh, dim=0)
                .reshape(-1, nh)
            )
            recv_lse_buffer_before_reduce = (
                torch.tensor(
                    test_case["recv_lse_buffer_before_reduce_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=nh, dim=0)
                .reshape(-1, nh)
            )
            expected_recv_lse_buffer = (
                torch.tensor(
                    test_case["expected_recv_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=nh, dim=0)
                .reshape(-1, nh)
            )
            post_process_inputs = (
                recv_buffer_before_reduce,
                recv_lse_buffer_before_reduce,
            )
        else:
            send_lse_buffer = None
            recv_lse_buffer_before_reduce = None
            expected_recv_lse_buffer = None
            post_process_inputs = (recv_buffer_before_reduce,)  # type: ignore[assignment]

        # run group-reduce comm kernel
        with self._switch_hier_comm_context(enable=use_hier_comm):
            assert (
                not use_hier_comm or magi_attention.comm.is_hierarchical_comm_enable()
            )
            work = group_reduce_collective(
                input=send_buffer,
                output=recv_buffer_before_reduce,
                input_split_size_list=input_split_size_list,
                output_split_size_list=output_split_size_list,
                dst_index_list=dst_index_list,
                src_indices_list=src_indices_list,
                group=self.process_group,
                async_op=async_op,
                reduce_op=reduce_op,
                input_lse=send_lse_buffer,
                output_lse=recv_lse_buffer_before_reduce,
                # NOTE: args below for hierarchical comm
                intra_group=self.intra_group,
                inter_group=self.inter_group,
                deterministic=deterministic,
            )

        # post process
        post_process_outputs = work.wait_post_process(*post_process_inputs)

        # check results
        err_msg_list = []
        try:
            recv_buffer_after_reduce = (
                post_process_outputs[0] if is_lse_reduce else post_process_outputs
            )
            assert_close(
                recv_buffer_after_reduce,
                expected_recv_buffer,
                atol=1e-8,
                rtol=1e-4,
                test_case="group-reduce recv buffer",
            )
        except Exception as e:
            err_msg_list.append(
                f"Group-Reduce collective {test_case_name=} recv buffer failed due to error: \n{e}\n"
                f"with: \n{recv_buffer_after_reduce=}\n{expected_recv_buffer=}\n"
            )
        if is_lse_reduce:
            try:
                recv_lse_buffer_after_reduce = post_process_outputs[1]
                assert_close(
                    recv_lse_buffer_after_reduce,
                    expected_recv_lse_buffer,
                    atol=1e-8,
                    rtol=1e-4,
                    test_case="group-reduce recv lse buffer",
                )
            except Exception as e:
                err_msg_list.append(
                    f"Group-Reduce collective {test_case_name=} recv lse buffer failed due to error: \n{e}\n"
                    f"with: \n{recv_lse_buffer_after_reduce=}\n{expected_recv_lse_buffer=}\n"
                )

        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))


if __name__ == "__main__":
    run_tests()
