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
                "is_causal_mapping": [False],
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
                "is_causal_mapping": [False] * 6,
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
