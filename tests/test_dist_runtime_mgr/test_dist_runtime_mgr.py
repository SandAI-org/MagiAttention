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

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
import magi_attention.testing
from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms


class TestDistAttnRuntimeMgr(DistTestBase):
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

        # -----    set up for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable():
            assert self.world_size == 4
            world_size_inter_node, world_size_intra_node = 2, 2
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
        return 4

    @property
    def seed(self) -> int:
        return 42

    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "test_config",
        [
            # full attn with total seqlen 14k
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 14336],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 14336],
                    ]
                ),
                "xattn_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 14336],
                    ]
                ),
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1222],
                    ]
                ),
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "total_seqlen_xattn_k": 1222,
                "chunk_size": 512,
            },
            # varlen full attn with total seqlen 12k
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "xattn_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1222],
                        [0, 1222],
                        [1222, 1558],
                        [1558, 1894],
                        [1894, 2230],
                        [2230, 2566],
                    ]
                ),
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "total_seqlen_xattn_k": 2566,
                "chunk_size": 512,
            },
            # varlen full attn with total seqlen 12k with overlapped q_ranges
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [2048, 4096],
                        [4096, 6144],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [10240, 12288],
                        [4096, 6144],
                        [10240, 12288],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "xattn_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                    ]
                ),
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1222],
                        [0, 1222],
                        [1222, 1558],
                        [1558, 1894],
                        [1894, 2230],
                        [2230, 2566],
                    ]
                ),
                "is_causal_mapping": [False] * 8,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "total_seqlen_xattn_k": 2566,
                "chunk_size": 512,
            },
            {
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 9720],
                        [9720, 19440],
                        [19440, 29160],
                        [29160, 38880],
                        [38880, 48600],
                        [48600, 58320],
                        [58320, 68040],
                        [68040, 77760],
                        [77760, 87480],
                        [87480, 97200],
                        [97200, 106920],
                        [106920, 116640],
                        [116640, 126360],
                        [126360, 136080],
                        [136080, 145800],
                        [145800, 155520],
                        [77760, 87480],
                        [87480, 97200],
                        [97200, 106920],
                        [106920, 116640],
                        [116640, 126360],
                        [126360, 136080],
                        [136080, 145800],
                        [145800, 155520],
                        [155520, 165240],
                        [165240, 174960],
                        [174960, 184680],
                        [184680, 194400],
                        [194400, 204120],
                        [204120, 213840],
                        [213840, 223560],
                        [223560, 233280],
                        [233280, 243000],
                        [243000, 252720],
                        [252720, 262440],
                        [262440, 272160],
                        [272160, 281880],
                        [281880, 291600],
                        [223560, 233280],
                        [233280, 243000],
                        [243000, 252720],
                        [252720, 262440],
                        [262440, 272160],
                        [272160, 281880],
                        [281880, 291600],
                        [291600, 293220],
                        [293220, 294912],
                        [293220, 294912],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 9720],
                        [0, 19440],
                        [0, 29160],
                        [0, 38880],
                        [0, 48600],
                        [0, 58320],
                        [0, 68040],
                        [0, 77760],
                        [0, 0],
                        [0, 9720],
                        [0, 19440],
                        [0, 29160],
                        [0, 38880],
                        [0, 48600],
                        [0, 58320],
                        [0, 68040],
                        [77760, 87480],
                        [87480, 97200],
                        [97200, 106920],
                        [106920, 116640],
                        [116640, 126360],
                        [126360, 136080],
                        [136080, 145800],
                        [145800, 155520],
                        [155520, 165240],
                        [155520, 174960],
                        [155520, 184680],
                        [155520, 194400],
                        [155520, 204120],
                        [155520, 213840],
                        [155520, 223560],
                        [155520, 155520],
                        [155520, 165240],
                        [155520, 174960],
                        [155520, 184680],
                        [155520, 194400],
                        [155520, 204120],
                        [155520, 213840],
                        [223560, 233280],
                        [233280, 243000],
                        [243000, 252720],
                        [252720, 262440],
                        [262440, 272160],
                        [272160, 281880],
                        [281880, 291600],
                        [291600, 293220],
                        [291600, 291600],
                        [293220, 294912],
                    ]
                ),
                "xattn_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 9720],
                        [9720, 19440],
                        [19440, 29160],
                        [29160, 38880],
                        [38880, 48600],
                        [48600, 58320],
                        [58320, 68040],
                        [68040, 77760],
                        [77760, 87480],
                        [87480, 97200],
                        [97200, 106920],
                        [106920, 116640],
                        [116640, 126360],
                        [126360, 136080],
                        [136080, 145800],
                        [145800, 155520],
                        [155520, 165240],
                        [165240, 174960],
                        [174960, 184680],
                        [184680, 194400],
                        [194400, 204120],
                        [204120, 213840],
                        [213840, 223560],
                        [223560, 233280],
                        [233280, 243000],
                        [243000, 252720],
                        [252720, 262440],
                        [262440, 272160],
                        [272160, 281880],
                        [281880, 291600],
                        [291600, 293220],
                        [293220, 294912],
                    ]
                ),
                "xattn_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 153],
                        [153, 306],
                        [306, 459],
                        [459, 612],
                        [612, 765],
                        [765, 918],
                        [918, 1071],
                        [1071, 1224],
                        [1224, 1274],
                        [1274, 1324],
                        [1324, 1374],
                        [1374, 1424],
                        [1424, 1474],
                        [1474, 1524],
                        [1524, 1574],
                        [1574, 1624],
                        [1624, 1674],
                        [1674, 1724],
                        [1724, 1774],
                        [1774, 1824],
                        [1824, 1874],
                        [1874, 1924],
                        [1924, 1974],
                        [1974, 2024],
                        [2024, 2074],
                        [2074, 2124],
                        [2124, 2174],
                        [2174, 2224],
                        [2224, 2274],
                        [2274, 2324],
                        [2324, 2745],
                        [2745, 2795],
                    ]
                ),
                "total_seqlen_q": 294912,
                "total_seqlen_k": 294912,
                "total_seqlen_xattn_k": 2795,
                "chunk_size": 1536,
            },
        ],
    )
    def test_update_xattn_k_ranges(
        self,
        test_config: dict[str, Any],
    ):
        q_ranges: AttnRanges = test_config["q_ranges"]
        k_ranges: AttnRanges = test_config["k_ranges"]
        xattn_q_ranges: AttnRanges = test_config["xattn_q_ranges"]
        xattn_k_ranges: AttnRanges = test_config["xattn_k_ranges"]
        total_seqlen_q: int = test_config["total_seqlen_q"]
        total_seqlen_k: int = test_config["total_seqlen_k"]
        total_seqlen_xattn_k: int = test_config["total_seqlen_xattn_k"]
        chunk_size: int = test_config["chunk_size"]

        dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            cp_mesh=self.device_mesh,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=DistAttnConfig(),
        )

        host_xattn_attn_arg: AttnArg = dist_attn_runtime_mgr.get_xattn_args(
            xattn_q_ranges,
            xattn_k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(xattn_k_ranges),
            return_host_only=True,
        )

        total_q = torch.randn(
            total_seqlen_q,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_k = torch.randn(
            total_seqlen_xattn_k,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_v = torch.randn(
            total_seqlen_xattn_k,
            1,
            128,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        dist.all_reduce(total_q, group=self.nccl_group)
        dist.all_reduce(xattn_k, group=self.nccl_group)
        dist.all_reduce(xattn_v, group=self.nccl_group)

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)

        total_o_ref, _ = flex_flash_attn_func(
            q=total_q,
            k=xattn_k,
            v=xattn_v,
            q_ranges=xattn_q_ranges.to_tensor(device=torch.cuda.current_device()),
            k_ranges=xattn_k_ranges.to_tensor(device=torch.cuda.current_device()),
            attn_type_map=torch.zeros(
                len(xattn_q_ranges),
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            ),
            max_seqlen_q=xattn_q_ranges.max_seqlen,
            max_seqlen_k=xattn_k_ranges.max_seqlen,
        )

        local_o, _ = flex_flash_attn_func(
            q=local_q,
            k=xattn_k,
            v=xattn_v,
            **host_xattn_attn_arg.to_ffa_args(is_bwd=False),
        )

        total_o = dist_attn_runtime_mgr.undispatch_qo(local_o)

        magi_attention.testing.assert_close(
            total_o,
            total_o_ref,
        )

        total_xattn_attn_arg: AttnArg = dist_attn_runtime_mgr.get_xattn_args(
            xattn_q_ranges,
            xattn_k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(xattn_q_ranges),
            return_host_only=False,
        )

        total_o, _ = flex_flash_attn_func(
            q=total_q,
            k=xattn_k,
            v=xattn_v,
            **total_xattn_attn_arg.to_ffa_args(is_bwd=False),
        )

        magi_attention.testing.assert_close(
            total_o,
            total_o_ref,
        )


if __name__ == "__main__":
    run_tests()
