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
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.api.functools import compute_pad_size, pad_at_dim
from magi_attention.api.magi_attn_interface import (
    get_position_ids,
    magi_attn_flex_dispatch,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms

NAME = "name"
SKIP_WORLD_SIZE = "skip_world_size"
NUM_HEADS = 1
HEAD_DIM = 64


class TestPositionIdsWithWorldSize1(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]
        self.gloo_groups = [
            dist.new_group(list(range(self.world_size)), backend="gloo")
            for _ in range(1)
        ]

        # NOTE: test using sdpa backend with fp64 dtype support
        os.environ["MAGI_ATTENTION_SDPA_BACKEND"] = "1"

        # -----    set up for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable() and self.world_size in (
            4,
            6,
            8,
        ):
            world_size_inter_node, world_size_intra_node = {
                4: (2, 2),
                6: (3, 2),
                8: (2, 4),
            }[self.world_size]
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            self.device_mesh = None

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def gloo_group(self) -> dist.ProcessGroup:
        return self.gloo_groups[0]

    @property
    def world_size(self) -> int:
        return 1

    @property
    def seed(self) -> int:
        return 42

    @with_comms
    @parameterize(
        # TODO: test more diverse and complicated attn mask
        "attn_config",
        [
            # full attn with seqlen 1k and batchsize 2
            {
                NAME: "full_attn_2k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 2048],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 2048],
                    ]
                ),
                "is_causal_mapping": [True, True],
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
            },
            # full attn with seqlen 2k and batchsize 3
            {
                NAME: "full_attn_6k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                    ]
                ),
                "is_causal_mapping": [True, True, True],
                "total_seqlen_q": 6144,
                "total_seqlen_k": 6144,
            },
            # varlen full attn with total seqlen 1050
            {
                NAME: "flex_full_attn_1050",
                SKIP_WORLD_SIZE: [4, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 1050],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
            },
            # varlen block causal with total seqlen 1920
            {
                NAME: "varlen_block_causal_1920",
                SKIP_WORLD_SIZE: [7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 768],
                        [768, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1920],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 768],
                        [768, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1920],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 1920,
                "total_seqlen_k": 1920,
            },
            # varlen block causal with total seqlen 840
            {
                NAME: "varlen_block_causal_4200",
                SKIP_WORLD_SIZE: [4, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 640],
                        [640, 1280],
                        [1280, 1920],
                        [1920, 2560],
                        [2560, 3200],
                        [3200, 3840],
                        [3840, 4200],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 640],
                        [640, 1280],
                        [1280, 1920],
                        [1920, 2560],
                        [2560, 3200],
                        [3200, 3840],
                        [3840, 4200],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 4200,
                "total_seqlen_k": 4200,
            },
        ],
    )
    @parameterize(
        "head_dim",
        [64, 80],
    )
    def test_position_ids(
        self,
        attn_config: dict[str, Any],
        head_dim: int,
    ):
        # -----    skip for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable():
            if self.world_size not in (4, 6, 8):
                # skip for invalid world size
                # when hierarchical comm is enabled
                return

        attn_config_ = deepcopy(attn_config)

        q_ranges: AttnRanges = attn_config_["q_ranges"]
        k_ranges: AttnRanges = attn_config_["k_ranges"]
        is_causal_mapping: list[bool] = attn_config_["is_causal_mapping"]
        total_seqlen_q: int = attn_config_["total_seqlen_q"]
        total_seqlen_k: int = attn_config_["total_seqlen_k"]

        device = torch.cuda.current_device()
        dist_attn_config = DistAttnConfig()

        #   -----   init input   -----   #
        global_x = torch.randn(
            total_seqlen_q, head_dim, device=device, requires_grad=True
        )

        #   -----   dispatch along seqlen dim   -----   #
        cp_size = dist.get_world_size(self.nccl_group)
        pad_size, _ = compute_pad_size(total_seqlen_q, cp_size, head_dim)
        global_x_padded = pad_at_dim(global_x, 0, pad_size)
        local_x_padded, dist_attn_runtime_key = magi_attn_flex_dispatch(
            global_x,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[
                AttnMaskType.CAUSAL if is_causal else AttnMaskType.FULL
                for is_causal in is_causal_mapping
            ],
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            head_dim=head_dim,
            pad_size=pad_size,
            cp_group=None
            if magi_attention.is_hierarchical_comm_enable()
            else self.nccl_group,
            cp_mesh=self.device_mesh,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=dist_attn_config,
        )

        #  -----  get position_ids and check  -----  #
        position_ids = get_position_ids(dist_attn_runtime_key)
        position_ids = position_ids[
            position_ids < total_seqlen_q - 1
        ]  # remove padded id
        valid_length = position_ids.size(0)

        self.assertTrue(
            torch.equal(local_x_padded[:valid_length], global_x_padded[position_ids])
        )


class TestPositionIdsWithWorldSize2(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize3(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize4(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize5(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize6(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize7(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


class TestPositionIdsWithWorldSize8(TestPositionIdsWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_position_ids(self, *args, **kwargs):
        super().test_position_ids(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
