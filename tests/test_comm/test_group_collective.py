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

import os
from contextlib import contextmanager
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


class TestGroupCollectiveWithWorldSize4(DistTestBase):
    def init_pg(self):
        super().init_pg()

        # -----    set up for hier comm   ---- #

        self.hier_comm_env_variable = "MAGI_ATTENTION_HIERARCHICAL_COMM"

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
        return torch.int32

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
        with self._switch_hier_comm(enable=use_hier_comm):
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
        self.assertTrue(
            torch.equal(recv_buffer, expected_recv_buffer),
            msg=f"Group-Cast collective has failed: {recv_buffer=} != {expected_recv_buffer=}",
        )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        # TODO: add test cases for world size > 4
        "test_case",
        [
            {
                "name": "naive_a2a_like_reduce",
                "world_size": 4,
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
                "name": "normal_group_reduce",
                "world_size": 4,
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
                    [8, 10, 21, 19],
                    [17, 18, 13, 14],
                    [20, 22, 7, 8],
                    [10, 13, 15, 16],
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
    @parameterize("async_op", [True])  # skip async_op=False to speed up
    def test_group_reduce_collective(
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

        # run group-reduce comm kernel
        with self._switch_hier_comm(enable=use_hier_comm):
            work = group_reduce_collective(
                input=send_buffer,
                output=recv_buffer_before_reduce,
                input_split_size_list=input_split_size_list,
                output_split_size_list=output_split_size_list,
                dst_index_list=dst_index_list,
                src_indices_list=src_indices_list,
                group=self.process_group,
                async_op=async_op,
                # NOTE: args below for hierarchical comm
                intra_group=self.intra_group,
                inter_group=self.inter_group,
            )

        # post process
        recv_buffer_after_reduce = work.wait_post_process(recv_buffer_before_reduce)

        # check results
        self.assertTrue(
            torch.equal(recv_buffer_after_reduce, expected_recv_buffer),
            msg=(
                f"Group-Reduce collective has failed: "
                f"{recv_buffer_after_reduce=} != {expected_recv_buffer=}",
            ),
        )

    @contextmanager
    def _switch_hier_comm(self, enable: bool = False):
        old_value = os.environ.get(self.hier_comm_env_variable, "0")
        os.environ[self.hier_comm_env_variable] = "1" if enable else "0"
        if enable:  # sanity check
            assert magi_attention.comm.is_hierarchical_comm_enable()
        yield
        os.environ[self.hier_comm_env_variable] = old_value


class TestGroupCollectiveWithWorldSize6(TestGroupCollectiveWithWorldSize4):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_group_cast_collective(self, *args, **kwargs):
        super().test_group_cast_collective(*args, **kwargs)

    @skip_if_lt_x_gpu(6)
    def test_group_reduce_collective(self, *args, **kwargs):
        super().test_group_reduce_collective(*args, **kwargs)


class TestGroupCollectiveWithWorldSize8(TestGroupCollectiveWithWorldSize4):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_group_cast_collective(self, *args, **kwargs):
        super().test_group_cast_collective(*args, **kwargs)

    @skip_if_lt_x_gpu(8)
    def test_group_reduce_collective(self, *args, **kwargs):
        super().test_group_reduce_collective(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
