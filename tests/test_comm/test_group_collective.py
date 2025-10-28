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

from magi_attention.api.functools import pad_at_dim
from magi_attention.comm.primitive.grpcoll import group_cast, group_reduce
from magi_attention.comm.primitive.grpcoll._config import GrpCollConfig
from magi_attention.comm.primitive.grpcoll._handle import (
    GrpCollHandle,
    GrpCollInterHandle,
    GrpCollIntraHandle,
)
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.comm.primitive.grpcoll.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
    sanity_check_for_group_reduce_meta_args_per_rank,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import assert_close
from magi_attention.testing.utils import switch_envvar_context
from magi_attention.utils import wrap_to_list


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

        # -----    set up for native grpcoll   ---- #

        self._switch_native_grpcoll_context = partial(
            switch_envvar_context, envvar_name="MAGI_ATTENTION_NATIVE_GRPCOLL"
        )

        grpcoll_mgr.register_buffer(
            group=self.process_group,
            config=GrpCollConfig(
                num_sms=self.num_sms_for_native_grpcoll,
                nvl_chunk_size=8,
                nvl_buffer_size=256,
                rdma_chunk_size=8,
                rdma_buffer_size=256,
                num_nvl_bytes=int(1e9),
                num_rdma_bytes=0,
            ),
        )
        grpcoll_mgr.check_registered(group=self.process_group)

    def destroy_pg(self):
        grpcoll_mgr.release_buffer(group=self.process_group)
        grpcoll_mgr.check_released(group=self.process_group)

        super().destroy_pg()

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
        # TODO: add test cases for world size > 4
        return 4

    @property
    def num_heads(self) -> int:
        return 4

    @property
    def head_dim(self) -> int:
        return 128

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def num_sms_for_native_grpcoll(self) -> int:
        return 20

    @property
    def num_channels_for_native_grpcoll(self) -> int:
        return self.num_sms_for_native_grpcoll // 2

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
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
                "name": "normal_group_cast_case2_with_lse",
                "world_size": 4,
                "cast_lse": True,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "send_lse_buffer_per_rank": [
                    [0.1, 1.2, 2.3, 3.4],
                    [4.5, 5.6, 6.7, 7.8],
                    [8.9, 9.1, 10.2, 11.3],
                    [12.4, 13.5, 14.6, 15.7],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13, 0, 1],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 6, 7, 14, 15],
                    [8, 9, 4, 5],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [5.6, 9.1, 13.5, 0.1, 1.2],
                    [0.1, 1.2, 10.2, 11.3, 2.3, 12.4, 13.5],
                    [2.3, 6.7, 7.8, 14.6, 15.7],
                    [8.9, 9.1, 4.5, 5.6],
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
            {
                "name": "group_cast_with_max_output_seqlen_with_lse",
                "world_size": 4,
                "cast_lse": True,
                "max_output_seqlen": 16,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "send_lse_buffer_per_rank": [
                    [0.1, 1.2, 2.3, 3.4],
                    [4.5, 5.6, 6.7, 7.8],
                    [8.9, 9.1, 10.2, 11.3],
                    [12.4, 13.5, 14.6, 15.7],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13, 0, 1],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 6, 7, 14, 15],
                    [8, 9, 4, 5],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [5.6, 9.1, 13.5, 0.1, 1.2],
                    [0.1, 1.2, 10.2, 11.3, 2.3, 12.4, 13.5],
                    [2.3, 6.7, 7.8, 14.6, 15.7],
                    [8.9, 9.1, 4.5, 5.6],
                ],
                "input_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                # right pad some `0`
                "output_split_size_list_per_rank": [
                    [1, 1, 1, 2, 0, 0],
                    [2, 2, 1, 1, 1, 0],
                    [1, 2, 2, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                ],
                "dst_indices_list_per_rank": [
                    [[0, 1], [1, 2], []],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
                # right pad some `world_size`
                "src_index_list_per_rank": [
                    [1, 2, 3, 0, 4, 4],
                    [0, 2, 0, 3, 3, 4],
                    [0, 1, 3, 4, 4, 4],
                    [2, 2, 1, 1, 4, 4],
                ],
            },
            {
                "name": "group_cast_with_double_groups_with_lse",
                "world_size": 4,
                "cast_lse": True,
                "num_groups": 2,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "send_buffer_2nd_per_rank": [
                    [0.2, 1.4, 2.6, 3.8],
                    [4.1, 5.3, 6.5, 7.7],
                    [8.2, 9.4, 10.6, 11.8],
                    [12.1, 13.3, 14.5, 15.7],
                ],
                "send_lse_buffer_per_rank": [
                    [0.1, 1.2, 2.3, 3.4],
                    [4.5, 5.6, 6.7, 7.8],
                    [8.9, 9.1, 10.2, 11.3],
                    [12.4, 13.5, 14.6, 15.7],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13, 0, 1],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 6, 7, 14, 15],
                    [8, 9, 4, 5],
                ],
                "expected_recv_buffer_2nd_per_rank": [
                    [5.3, 9.4, 13.3, 0.2, 1.4],
                    [0.2, 1.4, 10.6, 11.8, 2.6, 12.1, 13.3],
                    [2.6, 6.5, 7.7, 14.5, 15.7],
                    [8.2, 9.4, 4.1, 5.3],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [5.6, 9.1, 13.5, 0.1, 1.2],
                    [0.1, 1.2, 10.2, 11.3, 2.3, 12.4, 13.5],
                    [2.3, 6.7, 7.8, 14.6, 15.7],
                    [8.9, 9.1, 4.5, 5.6],
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
            {
                "name": "group_cast_with_triple_groups_with_lse",
                "world_size": 4,
                "cast_lse": True,
                "num_groups": 3,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                "send_buffer_2nd_per_rank": [
                    [0.2, 1.4, 2.6, 3.8],
                    [4.1, 5.3, 6.5, 7.7],
                    [8.2, 9.4, 10.6, 11.8],
                    [12.1, 13.3, 14.5, 15.7],
                ],
                "send_buffer_3rd_per_rank": [
                    [0.1, 1.3, 2.5, 3.7],
                    [4.2, 5.4, 6.6, 7.8],
                    [8.1, 9.3, 10.5, 11.7],
                    [12.2, 13.4, 14.6, 15.8],
                ],
                "send_lse_buffer_per_rank": [
                    [0.1, 1.2, 2.3, 3.4],
                    [4.5, 5.6, 6.7, 7.8],
                    [8.9, 9.1, 10.2, 11.3],
                    [12.4, 13.5, 14.6, 15.7],
                ],
                "expected_recv_buffer_per_rank": [
                    [5, 9, 13, 0, 1],
                    [0, 1, 10, 11, 2, 12, 13],
                    [2, 6, 7, 14, 15],
                    [8, 9, 4, 5],
                ],
                "expected_recv_buffer_2nd_per_rank": [
                    [5.3, 9.4, 13.3, 0.2, 1.4],
                    [0.2, 1.4, 10.6, 11.8, 2.6, 12.1, 13.3],
                    [2.6, 6.5, 7.7, 14.5, 15.7],
                    [8.2, 9.4, 4.1, 5.3],
                ],
                "expected_recv_buffer_3rd_per_rank": [
                    [5.4, 9.3, 13.4, 0.1, 1.3],
                    [0.1, 1.3, 10.5, 11.7, 2.5, 12.2, 13.4],
                    [2.5, 6.6, 7.8, 14.6, 15.8],
                    [8.1, 9.3, 4.2, 5.4],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [5.6, 9.1, 13.5, 0.1, 1.2],
                    [0.1, 1.2, 10.2, 11.3, 2.3, 12.4, 13.5],
                    [2.3, 6.7, 7.8, 14.6, 15.7],
                    [8.9, 9.1, 4.5, 5.6],
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
    @parameterize(
        "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64]
    )
    @parameterize("use_hier_comm", [False, True])
    @parameterize("use_native_grpcoll", [False, True])
    @parameterize("async_op", [False, True])
    def test_group_cast(
        self,
        test_case: dict[str, Any],
        dtype: torch.dtype,
        use_hier_comm: bool,
        use_native_grpcoll: bool,
        async_op: bool,
    ):
        cast_lse = test_case.get("cast_lse", False)
        max_output_seqlen = test_case.get("max_output_seqlen", None)
        num_groups = test_case.get("num_groups", 1)
        assert num_groups <= 3

        test_case_name = (
            f"{test_case['name']=} x {dtype=} x "
            f"{use_hier_comm=} x {use_native_grpcoll=} x {async_op=} x "
            f"{cast_lse=} x {max_output_seqlen=}"
        )

        # skip for unmatched world size
        if self.world_size != test_case["world_size"]:
            return

        # skip when enabling hier comm
        if use_hier_comm:
            # TODO: support hier comm as a sync op
            if not async_op:
                return
            # TODO: support hier comm with native grpcoll
            if use_native_grpcoll:
                return
            # TODO: support hier comm with cast_lse
            if cast_lse:
                return

        # skip when max_output_seqlen is given
        if max_output_seqlen is not None:
            # TODO: support max_output_seqlen for other implementations
            if not use_native_grpcoll:
                return

        # skip when num_groups > 1
        if num_groups > 1:
            # TODO: support num_groups > 1 for other implementations
            if not use_native_grpcoll:
                return

        # prepare meta args per rank
        input_split_size_list_per_rank = test_case["input_split_size_list_per_rank"]
        output_split_size_list_per_rank = test_case["output_split_size_list_per_rank"]
        dst_indices_list_per_rank = test_case["dst_indices_list_per_rank"]
        src_index_list_per_rank = test_case["src_index_list_per_rank"]

        # sanity check for meta args per rank
        if max_output_seqlen is None:
            # NOTE: when max_output_seqlen is given, the original sanity check will fail thus skipped
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

        input_seqlen = sum(input_split_size_list)
        actual_output_seqlen = sum(output_split_size_list)
        if max_output_seqlen is not None:
            assert actual_output_seqlen <= max_output_seqlen

        # prepare buffers
        send_buffer = (
            torch.tensor(
                test_case["send_buffer_per_rank"][self.rank],
                dtype=dtype,
                device=self.device,
            )
            .repeat_interleave(repeats=self.hidden_size, dim=0)
            .reshape(-1, self.num_heads, self.head_dim)
        )
        expected_recv_buffer = (
            torch.tensor(
                test_case["expected_recv_buffer_per_rank"][self.rank],
                dtype=dtype,
                device=self.device,
            )
            .repeat_interleave(repeats=self.hidden_size, dim=0)
            .reshape(-1, self.num_heads, self.head_dim)
        )
        recv_buffer = torch.full_like(
            expected_recv_buffer,
            fill_value=-1,
            dtype=dtype,
            device=self.device,
        )

        if max_output_seqlen is not None:
            recv_buffer = pad_at_dim(
                recv_buffer,
                pad_size=max_output_seqlen - actual_output_seqlen,
                dim=0,
                value=-1,
            )
            assert recv_buffer.shape[0] == max_output_seqlen

        if num_groups > 1:
            send_buffer = [send_buffer]
            expected_recv_buffer = [expected_recv_buffer]
            recv_buffer = [recv_buffer]

            send_buffer.append(
                torch.tensor(
                    test_case["send_buffer_2nd_per_rank"][self.rank],
                    dtype=dtype,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.hidden_size, dim=0)
                .reshape(-1, self.num_heads, self.head_dim)
            )
            expected_recv_buffer.append(
                torch.tensor(
                    test_case["expected_recv_buffer_2nd_per_rank"][self.rank],
                    dtype=dtype,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.hidden_size, dim=0)
                .reshape(-1, self.num_heads, self.head_dim)
            )
            recv_buffer.append(recv_buffer[0].clone())

            if num_groups > 2:
                send_buffer.append(
                    torch.tensor(
                        test_case["send_buffer_3rd_per_rank"][self.rank],
                        dtype=dtype,
                        device=self.device,
                    )
                    .repeat_interleave(repeats=self.hidden_size, dim=0)
                    .reshape(-1, self.num_heads, self.head_dim)
                )
                expected_recv_buffer.append(
                    torch.tensor(
                        test_case["expected_recv_buffer_3rd_per_rank"][self.rank],
                        dtype=dtype,
                        device=self.device,
                    )
                    .repeat_interleave(repeats=self.hidden_size, dim=0)
                    .reshape(-1, self.num_heads, self.head_dim)
                )
                recv_buffer.append(recv_buffer[0].clone())

        if cast_lse:
            # prepare lse buffer with shape [seqlen, num_heads]
            send_lse_buffer = (
                torch.tensor(
                    test_case["send_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.num_heads, dim=0)
                .reshape(-1, self.num_heads)
            )
            expected_recv_lse_buffer = (
                torch.tensor(
                    test_case["expected_recv_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.num_heads, dim=0)
                .reshape(-1, self.num_heads)
            )
            recv_lse_buffer = torch.full_like(
                expected_recv_lse_buffer,
                fill_value=-1,
                dtype=torch.float32,
                device=self.device,
            )
            if max_output_seqlen is not None:
                recv_lse_buffer = pad_at_dim(
                    recv_lse_buffer,
                    pad_size=max_output_seqlen - actual_output_seqlen,
                    dim=0,
                    value=-1,
                )
                assert recv_lse_buffer.shape[0] == max_output_seqlen

            post_process_inputs = (
                recv_buffer,
                recv_lse_buffer,
            )
        else:
            send_lse_buffer = None
            recv_lse_buffer = None
            expected_recv_lse_buffer = None

            post_process_inputs = (recv_buffer,)  # type: ignore[assignment]

        # prepare for native grpcoll
        native_grpcoll_handle_dict: dict[str, GrpCollHandle | None]
        if use_native_grpcoll:
            native_grpcoll_handle_dict = {"group_cast": None}
        else:
            native_grpcoll_handle_dict = {}

        # run group-cast comm kernel
        with self._switch_hier_comm_context(
            enable=use_hier_comm
        ), self._switch_native_grpcoll_context(enable=use_native_grpcoll):
            work = group_cast(
                input=send_buffer,
                output=recv_buffer,
                input_split_sizes=input_split_size_list,
                output_split_sizes=output_split_size_list,
                dst_indices=dst_indices_list,
                src_index=src_index_list,
                group=self.process_group,
                async_op=async_op,
                cast_lse=cast_lse,
                input_lse=send_lse_buffer,
                output_lse=recv_lse_buffer,
                # kwargs below for hier comm
                intra_group=self.intra_group,
                inter_group=self.inter_group,
                # kwargs below for native grpcoll
                native_grpcoll_handle_dict=native_grpcoll_handle_dict,
            )

        # post process
        post_process_outputs = work.wait_post_process(*post_process_inputs)

        # check results
        err_msg_list = []
        try:
            recv_buffer = post_process_outputs[0] if cast_lse else post_process_outputs
            recv_buffer = wrap_to_list(recv_buffer)
            expected_recv_buffer = wrap_to_list(expected_recv_buffer)
            for ith_recv_buffer, ith_expected_recv_buffer in zip(
                recv_buffer, expected_recv_buffer
            ):
                assert_close(
                    ith_recv_buffer[:actual_output_seqlen],
                    ith_expected_recv_buffer,
                    atol=1e-8,
                    rtol=1e-6,
                    test_case="group-cast recv buffer",
                )
        except Exception as e:
            err_msg_list.append(
                f"For group-cast: {test_case_name=}, recv buffer is failed due to error: \n{e}\n"
                f"with: \n{recv_buffer=}\n{expected_recv_buffer=}\n"
            )
        if cast_lse:
            try:
                recv_lse_buffer = post_process_outputs[1]
                assert_close(
                    recv_lse_buffer[:actual_output_seqlen],
                    expected_recv_lse_buffer,
                    atol=1e-8,
                    rtol=1e-6,
                    test_case="group-cast recv lse buffer",
                )
            except Exception as e:
                err_msg_list.append(
                    f"For group-cast: {test_case_name=}, recv lse buffer is failed due to error: \n{e}\n"
                    f"with: \n{recv_lse_buffer=}\n{expected_recv_lse_buffer=}\n"
                )

        # raise error if any
        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))

        # check for native grpcoll
        if use_native_grpcoll:
            for handle in native_grpcoll_handle_dict.values():
                assert handle is not None and isinstance(handle, GrpCollHandle)
                self._check_native_grpcoll_handle(
                    handle=handle,
                    num_tokens=input_seqlen,
                    num_recv_tokens=max_output_seqlen or actual_output_seqlen,
                )

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "test_case",
        [
            # TODO: test acc_reduce=False
            {
                "name": "naive_a2a_like_sum_reduce",
                "world_size": 4,
                "reduce_op": "sum",
                "acc_reduce": True,
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
                "acc_reduce": True,
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
                "acc_reduce": True,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21],
                ],
                "recv_buffer_before_reduce_per_rank": [
                    [1, 2, 0, 1],
                    [1, 0, 1, 0],
                    [4, 2, 3, 4],
                    [2, 2, 3, 2],
                ],
                "expected_recv_buffer_per_rank": [
                    [3, 4, 7, 10],
                    [9, 6, 7, 7],
                    [12, 8, 5, 6],
                    [6, 5, 9, 9],
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
                "acc_reduce": True,
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
                    [4.6351, 5.9414, 11.5559, 19.0],
                    [12.1877, 13.6155, 10.5503, 12.8558],
                    [4.0215, 2.2208, 6.2703, 4.3447],
                    [5.0946, 2.1294, 8.8161, 6.6724],
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
            {
                "name": "group_lse_reduce_with_max_input_seqlen",
                "world_size": 4,
                "reduce_op": "lse",
                "acc_reduce": True,
                "max_input_seqlen": 16,
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
                    [4.6351, 5.9414, 11.5559, 19.0],
                    [12.1877, 13.6155, 10.5503, 12.8558],
                    [4.0215, 2.2208, 6.2703, 4.3447],
                    [5.0946, 2.1294, 8.8161, 6.6724],
                ],
                "expected_recv_lse_buffer_per_rank": [
                    [0.7014, 1.5298, 0.4102, 2.75],
                    [1.1369, 2.0483, 1.5019, 2.3502],
                    [1.0620, 1.7048, 2.7014, 1.3133],
                    [-0.7986, 2.0525, -0.4241, -0.7481],
                ],
                # right pad some `0`
                "input_split_size_list_per_rank": [
                    [1, 1, 1, 2, 0, 0],
                    [2, 2, 1, 1, 1, 0],
                    [1, 2, 2, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                ],
                "output_split_size_list_per_rank": [
                    [2, 1, 1],
                    [1, 1, 2],
                    [1, 1, 2],
                    [1, 1, 2],
                ],
                # right pad some `world_size`
                "dst_index_list_per_rank": [
                    [1, 2, 3, 0, 4, 4],
                    [0, 2, 0, 3, 3, 4],
                    [0, 1, 3, 4, 4, 4],
                    [1, 1, 0, 2, 2, 4],
                ],
                "src_indices_list_per_rank": [
                    [[0, 1], [1, 2], [3]],
                    [[3], [0, 3], [2]],
                    [[3], [0, 3], [1]],
                    [[1], [0, 1], [2]],
                ],
            },
            # TODO: test multiple groups with lse reduce
            {
                "name": "group_sum_reduce_with_double_groups",
                "world_size": 4,
                "reduce_op": "sum",
                "acc_reduce": True,
                "num_groups": 2,
                "send_buffer_per_rank": [
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9, 10, 11],
                    [12, 13, 14, 15, 16],
                    [17, 18, 19, 20, 21],
                ],
                "send_buffer_2nd_per_rank": [
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
                "recv_buffer_before_reduce_2nd_per_rank": [
                    [0, 0, 0, 0],
                    [-1, -1, -1, -1],
                    [-2, -2, -2, -2],
                    [-3, -3, -3, -3],
                ],
                "expected_recv_buffer_per_rank": [
                    [9, 11, 22, 20],
                    [19, 20, 15, 16],
                    [23, 25, 10, 11],
                    [14, 17, 19, 20],
                ],
                "expected_recv_buffer_2nd_per_rank": [
                    [8, 10, 21, 19],
                    [16, 17, 12, 13],
                    [18, 20, 5, 6],
                    [7, 10, 12, 13],
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
    @parameterize(
        "dtypes",
        [  # (dtype, comm_dtype)
            (torch.bfloat16, torch.bfloat16),
            (torch.float16, None),
            (torch.float32, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.float32, torch.float16),
            (torch.float64, None),
        ],
    )
    @parameterize("use_hier_comm", [False, True])
    @parameterize("use_native_grpcoll", [False, True])
    @parameterize("deterministic", [False, True])
    @parameterize("async_op", [False, True])
    def test_group_reduce(
        self,
        test_case: dict[str, Any],
        dtypes: tuple[torch.dtype, torch.dtype | None],  # (dtype, comm_dtype)
        use_hier_comm: bool,
        use_native_grpcoll: bool,
        deterministic: bool,
        async_op: bool,
    ):
        dtype, comm_dtype = dtypes
        reduce_op = test_case["reduce_op"]
        acc_reduce = test_case["acc_reduce"]
        is_lse_reduce = reduce_op == "lse"
        max_input_seqlen = test_case.get("max_input_seqlen", None)

        test_case_name = (
            f"{test_case['name']=} x {dtype=} x {comm_dtype=} x "
            f"{use_hier_comm=} x {use_native_grpcoll=} x "
            f"{deterministic=} x {async_op=} x "
            f"{reduce_op=} x {acc_reduce=} x {max_input_seqlen=}"
        )
        num_groups = test_case.get("num_groups", 1)
        assert num_groups <= 3

        # skip for unmatched world size
        if self.world_size != test_case["world_size"]:
            return

        # skip when enabling hier comm
        if use_hier_comm:
            # TODO: support hier comm as a sync op
            if not async_op:
                return
            # TODO: support hier comm for other reduce ops
            if reduce_op != "sum":
                return
            # TODO: support hier comm with native grpcoll
            if use_native_grpcoll:
                return

        # skip when enabling grpcoll
        if use_native_grpcoll:
            # for now, native grpcoll is always deterministic
            if not deterministic:
                return

        # skip when max_input_seqlen is given
        if max_input_seqlen is not None:
            # TODO: support max_input_seqlen for other implementations
            if not use_native_grpcoll:
                return

        # skip when dtype != comm_dtype
        if comm_dtype is not None and dtype != comm_dtype:
            # TODO: support specific comm dtype for other implementations
            if not use_native_grpcoll:
                return

        # skip when num_groups > 1
        if num_groups > 1:
            # TODO: support num_groups > 1 for other implementations
            if not use_native_grpcoll:
                return

        # prepare meta args per rank
        input_split_size_list_per_rank = test_case["input_split_size_list_per_rank"]
        output_split_size_list_per_rank = test_case["output_split_size_list_per_rank"]
        dst_index_list_per_rank = test_case["dst_index_list_per_rank"]
        src_indices_list_per_rank = test_case["src_indices_list_per_rank"]

        # sanity check for meta args per rank
        if max_input_seqlen is None:
            # NOTE: when max_input_seqlen is given, the original sanity check will fail thus skipped
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

        actual_input_seqlen = sum(input_split_size_list)
        output_seqlen = sum(output_split_size_list)
        if max_input_seqlen is not None:
            assert actual_input_seqlen <= max_input_seqlen

        # prepare buffers
        send_buffer = (
            torch.tensor(
                test_case["send_buffer_per_rank"][self.rank],
                dtype=dtype,
                device=self.device,
            )
            .repeat_interleave(repeats=self.hidden_size, dim=0)
            .reshape(-1, self.num_heads, self.head_dim)
        )
        recv_buffer_before_reduce = (
            torch.tensor(
                test_case["recv_buffer_before_reduce_per_rank"][self.rank],
                dtype=dtype,
                device=self.device,
            )
            .repeat_interleave(repeats=self.hidden_size, dim=0)
            .reshape(-1, self.num_heads, self.head_dim)
        )
        expected_recv_buffer = (
            torch.tensor(
                test_case["expected_recv_buffer_per_rank"][self.rank],
                dtype=dtype,
                device=self.device,
            )
            .repeat_interleave(repeats=self.hidden_size, dim=0)
            .reshape(-1, self.num_heads, self.head_dim)
        )

        if max_input_seqlen is not None:
            send_buffer = pad_at_dim(
                send_buffer,
                pad_size=max_input_seqlen - actual_input_seqlen,
                dim=0,
                value=-1,
            )
            assert send_buffer.shape[0] == max_input_seqlen

        if num_groups > 1:
            send_buffer = [send_buffer]
            recv_buffer_before_reduce = [recv_buffer_before_reduce]
            expected_recv_buffer = [expected_recv_buffer]

            send_buffer.append(
                torch.tensor(
                    test_case["send_buffer_2nd_per_rank"][self.rank],
                    dtype=dtype,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.hidden_size, dim=0)
                .reshape(-1, self.num_heads, self.head_dim)
            )
            recv_buffer_before_reduce.append(
                torch.tensor(
                    test_case["recv_buffer_before_reduce_2nd_per_rank"][self.rank],
                    dtype=dtype,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.hidden_size, dim=0)
                .reshape(-1, self.num_heads, self.head_dim)
            )
            expected_recv_buffer.append(
                torch.tensor(
                    test_case["expected_recv_buffer_2nd_per_rank"][self.rank],
                    dtype=dtype,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.hidden_size, dim=0)
                .reshape(-1, self.num_heads, self.head_dim)
            )

            if num_groups > 2:
                send_buffer.append(
                    torch.tensor(
                        test_case["send_buffer_3rd_per_rank"][self.rank],
                        dtype=dtype,
                        device=self.device,
                    )
                    .repeat_interleave(repeats=self.hidden_size, dim=0)
                    .reshape(-1, self.num_heads, self.head_dim)
                )
                recv_buffer_before_reduce.append(
                    torch.tensor(
                        test_case["recv_buffer_before_reduce_3rd_per_rank"][self.rank],
                        dtype=dtype,
                        device=self.device,
                    )
                    .repeat_interleave(repeats=self.hidden_size, dim=0)
                    .reshape(-1, self.num_heads, self.head_dim)
                )
                expected_recv_buffer.append(
                    torch.tensor(
                        test_case["expected_recv_buffer_3rd_per_rank"][self.rank],
                        dtype=dtype,
                        device=self.device,
                    )
                    .repeat_interleave(repeats=self.hidden_size, dim=0)
                    .reshape(-1, self.num_heads, self.head_dim)
                )

        if is_lse_reduce:
            # prepare lse buffer with shape [seqlen, num_heads]
            send_lse_buffer = (
                torch.tensor(
                    test_case["send_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.num_heads, dim=0)
                .reshape(-1, self.num_heads)
            )
            if max_input_seqlen is not None:
                send_lse_buffer = pad_at_dim(
                    send_lse_buffer,
                    pad_size=max_input_seqlen - actual_input_seqlen,
                    dim=0,
                    value=-1,
                )
                assert send_lse_buffer.shape[0] == max_input_seqlen
            recv_lse_buffer_before_reduce = (
                torch.tensor(
                    test_case["recv_lse_buffer_before_reduce_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.num_heads, dim=0)
                .reshape(-1, self.num_heads)
            )
            expected_recv_lse_buffer = (
                torch.tensor(
                    test_case["expected_recv_lse_buffer_per_rank"][self.rank],
                    dtype=torch.float32,
                    device=self.device,
                )
                .repeat_interleave(repeats=self.num_heads, dim=0)
                .reshape(-1, self.num_heads)
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

        # prepare for native grpcoll
        native_grpcoll_handle_dict: dict[str, GrpCollHandle | None]
        if use_native_grpcoll:
            native_grpcoll_handle_dict = {"group_reduce": None}
        else:
            native_grpcoll_handle_dict = {}

        # run group-reduce comm kernel
        with self._switch_hier_comm_context(
            enable=use_hier_comm
        ), self._switch_native_grpcoll_context(enable=use_native_grpcoll):
            work = group_reduce(
                input=send_buffer,
                output=recv_buffer_before_reduce,
                input_split_sizes=input_split_size_list,
                output_split_sizes=output_split_size_list,
                dst_index=dst_index_list,
                src_indices=src_indices_list,
                group=self.process_group,
                async_op=async_op,
                reduce_op=reduce_op,
                acc_reduce=acc_reduce,
                comm_dtype=comm_dtype,
                input_lse=send_lse_buffer,
                output_lse=recv_lse_buffer_before_reduce,
                deterministic=deterministic,
                # kwargs below for hier comm
                intra_group=self.intra_group,
                inter_group=self.inter_group,
                # kwargs below for native grpcoll
                native_grpcoll_handle_dict=native_grpcoll_handle_dict,
            )

        # post process
        post_process_outputs = work.wait_post_process(*post_process_inputs)

        # check results
        err_msg_list = []
        try:
            recv_buffer_after_reduce = (
                post_process_outputs[0] if is_lse_reduce else post_process_outputs
            )
            recv_buffer_after_reduce = wrap_to_list(recv_buffer_after_reduce)
            expected_recv_buffer = wrap_to_list(expected_recv_buffer)
            for ith_recv_buffer_after_reduce, ith_expected_recv_buffer in zip(
                recv_buffer_after_reduce, expected_recv_buffer
            ):
                assert_close(
                    ith_recv_buffer_after_reduce,
                    ith_expected_recv_buffer,
                    atol=1e-8 if dtype != torch.float16 else 1e-4,
                    rtol=1e-4 if dtype != torch.float16 else 5e-3,
                    test_case="group-reduce recv buffer",
                )
        except Exception as e:
            err_msg_list.append(
                f"For group-reduce: {test_case_name=}, recv buffer is failed due to error: \n{e}\n"
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
                    f"For group-reduce: {test_case_name=}, recv lse buffer is failed due to error: \n{e}\n"
                    f"with: \n{recv_lse_buffer_after_reduce=}\n{expected_recv_lse_buffer=}\n"
                )

        # raise error if any
        if err_msg_list:
            raise AssertionError("\n".join(err_msg_list))

        # check for native grpcoll
        if use_native_grpcoll:
            for handle in native_grpcoll_handle_dict.values():
                assert handle is not None and isinstance(handle, GrpCollHandle)
                self._check_native_grpcoll_handle(
                    handle=handle,
                    num_tokens=output_seqlen,
                    num_recv_tokens=max_input_seqlen or actual_input_seqlen,
                )

    def _check_native_grpcoll_handle(
        self,
        handle: GrpCollHandle,
        num_tokens: int,
        num_recv_tokens: int,
    ):
        """Check native grpcoll handle
        NOTE: the `num_tokens` is the input seqlen / output seqlen for group cast / group reduce
        while the `num_recv_tokens` is the (max) output seqlen / (max) input seqlen for group cast / group reduce
        """
        match handle:
            case GrpCollIntraHandle():
                assert handle.rank_prefix_matrix.shape == (
                    self.world_size,
                    self.world_size,
                )
                assert handle.channel_prefix_matrix.shape == (
                    self.world_size,
                    self.num_channels_for_native_grpcoll,
                )
                assert handle.recv_channel_prefix_matrix.shape == (
                    self.world_size,
                    self.num_channels_for_native_grpcoll,
                )
                assert handle.recv_src_idx.shape == (num_recv_tokens,)
                assert handle.is_token_in_rank.shape == (num_tokens, self.world_size)
                assert handle.send_head.shape == (num_tokens, self.world_size)
            case GrpCollInterHandle():
                # TODO: add check for GrpCollInterHandle
                raise NotImplementedError(
                    "GrpCollInterHandle check is not implemented yet"
                )
            case _:
                raise ValueError(f"Invalid handle type: {type(handle)}")


if __name__ == "__main__":
    run_tests()
