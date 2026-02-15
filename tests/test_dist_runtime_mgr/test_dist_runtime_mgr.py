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

from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
import magi_attention.testing
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode, AttnRole
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    SequentialDispatchAlg,
)
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeMgr,
    init_dist_attn_runtime_mgr,
)
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import EPSILON
from magi_attention.testing.utils import switch_sdpa_backend_decorator
from magi_attention.utils import make_attn_mask_from_ffa_args


class TestDistAttnRuntimeMgr(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend="nccl")
            for _ in range(2)
        ]

        # -----    set up for hier comm   ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
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
    def device(self) -> int:
        return torch.cuda.current_device()

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
        # -----    skip for fa4 backend   ---- #

        if magi_attention.is_fa4_backend_enable():
            # TODO: support dynamic solver/qo comm for fa4 backend
            return

        q_ranges: AttnRanges = test_config["q_ranges"]
        k_ranges: AttnRanges = test_config["k_ranges"]
        xattn_q_ranges: AttnRanges = test_config["xattn_q_ranges"]
        xattn_k_ranges: AttnRanges = test_config["xattn_k_ranges"]
        total_seqlen_q: int = test_config["total_seqlen_q"]
        total_seqlen_k: int = test_config["total_seqlen_k"]
        total_seqlen_xattn_k: int = test_config["total_seqlen_xattn_k"]
        chunk_size: int = test_config["chunk_size"]

        num_heads_q = 1
        num_heads_kv = 1
        head_dim = 128

        dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(q_ranges),
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            cp_mesh=self.device_mesh,
            dist_attn_config=DistAttnConfig(),
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
        )

        host_xattn_attn_arg: AttnArg = dist_attn_runtime_mgr.get_xattn_args(
            xattn_q_ranges,
            xattn_k_ranges,
            attn_mask_type=[AttnMaskType.FULL] * len(xattn_k_ranges),
            return_host_only=True,
        )

        total_q = torch.randn(
            total_seqlen_q,
            num_heads_q,
            head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_k = torch.randn(
            total_seqlen_xattn_k,
            num_heads_kv,
            head_dim,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
        )
        xattn_v = torch.randn(
            total_seqlen_xattn_k,
            num_heads_kv,
            head_dim,
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
            test_case="self-attn forward out",
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
            test_case="cross-attn forward out",
        )

    @switch_sdpa_backend_decorator
    @skip_if_lt_x_gpu(4)
    @with_comms
    @parameterize(
        "test_config",
        [
            # causal attn with total seqlen 1k
            {
                "test_case": "causal attn with total seqlen 1k",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "attn_mask_type": [AttnMaskType.CAUSAL],
                "dist_attn_config": DistAttnConfig(),
                "dispatch_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "dispatch_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "dispatch_attn_mask_type": [AttnMaskType.FULL],
                "dispatch_dist_attn_config": DistAttnConfig(),
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 32,
            },
            # varlen full attn with total seqlen 1k
            {
                "test_case": "varlen full attn with total seqlen 1k",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 896],
                        [896, 1024],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [896, 1024],
                        [768, 896],
                        [640, 768],
                        [512, 640],
                        [384, 512],
                        [256, 384],
                        [128, 256],
                        [0, 128],
                    ]
                ),
                "attn_mask_type": [AttnMaskType.FULL] * 8,
                "dist_attn_config": DistAttnConfig(
                    dispatch_config=DispatchConfig(
                        alg=MinHeapDispatchAlg(),
                    ),
                    overlap_config=OverlapConfig(
                        enable=True,
                        mode=AttnOverlapMode.STATIC,
                        degree=4,
                    ),
                ),
                "dispatch_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "dispatch_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                    ]
                ),
                "dispatch_attn_mask_type": [AttnMaskType.FULL],
                "dispatch_dist_attn_config": DistAttnConfig(),
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # varlen block causal with total seqlen 960
            {
                "test_case": "varlen block causal with total seqlen 960",
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 960],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [0, 256],
                        [0, 384],
                        [0, 512],
                        [512, 640],
                        [512, 768],
                        [768, 960],
                    ]
                ),
                "attn_mask_type": [AttnMaskType.FULL] * 7,
                "dist_attn_config": DistAttnConfig(
                    dispatch_config=DispatchConfig(
                        alg=MinHeapDispatchAlg(),
                    ),
                    overlap_config=OverlapConfig(
                        enable=True,
                        mode=AttnOverlapMode.DYNAMIC,
                        degree=None,
                    ),
                ),
                "dispatch_q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 384],
                        [384, 512],
                        [512, 768],
                        [768, 960],
                    ]
                ),
                "dispatch_dist_attn_config": DistAttnConfig(
                    dispatch_config=DispatchConfig(
                        alg=SequentialDispatchAlg(),
                    ),
                    overlap_config=OverlapConfig(
                        enable=True,
                        mode=AttnOverlapMode.STATIC,
                        degree=2,
                    ),
                ),
                "dispatch_k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [0, 384],
                        [0, 512],
                        [512, 768],
                        [768, 960],
                    ]
                ),
                "dispatch_attn_mask_type": [AttnMaskType.FULL] * 2
                + [AttnMaskType.CAUSAL] * 3,
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 16,
            },
        ],
    )
    def test_ref_dispatch_meta(
        self,
        test_config: dict[str, Any],
    ):
        q_ranges: AttnRanges = test_config["q_ranges"]
        k_ranges: AttnRanges = test_config["k_ranges"]
        attn_mask_type: list[AttnMaskType] = test_config["attn_mask_type"]
        dist_attn_config: DistAttnConfig = test_config["dist_attn_config"]
        dispatch_q_ranges: AttnRanges = test_config["dispatch_q_ranges"]
        dispatch_k_ranges: AttnRanges = test_config["dispatch_k_ranges"]
        dispatch_attn_mask_type: list[AttnMaskType] = test_config[
            "dispatch_attn_mask_type"
        ]
        dispatch_dist_attn_config: DistAttnConfig = test_config[
            "dispatch_dist_attn_config"
        ]
        total_seqlen_q: int = test_config["total_seqlen_q"]
        total_seqlen_k: int = test_config["total_seqlen_k"]
        chunk_size: int = test_config["chunk_size"]

        num_heads_q = 16
        num_heads_kv = 4
        head_dim = 128

        # use dispatch mask to init dist attn runtime mgr
        dispatch_dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=dispatch_q_ranges,
            k_ranges=dispatch_k_ranges,
            attn_mask_type=dispatch_attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            cp_mesh=self.device_mesh,
            dist_attn_config=dispatch_dist_attn_config,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
        )

        # extract the dispatch meta as the ref
        ref_dispatch_meta_q = dispatch_dist_attn_runtime_mgr.dispatch_meta_q
        ref_dispatch_meta_k = dispatch_dist_attn_runtime_mgr.dispatch_meta_k

        # use the real mask to init dist attn runtime mgr
        # with ref dispatch meta w.r.t. the dispatch mask
        dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            cp_mesh=self.device_mesh,
            dist_attn_config=dist_attn_config,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            ref_dispatch_meta_q=ref_dispatch_meta_q,
            ref_dispatch_meta_k=ref_dispatch_meta_k,
        )

        # check attributes
        assert dist_attn_runtime_mgr.dispatch_meta_q == ref_dispatch_meta_q
        assert dist_attn_runtime_mgr.dispatch_meta_k == ref_dispatch_meta_k
        assert torch.equal(
            dist_attn_runtime_mgr.get_position_ids(attn_role=AttnRole.QUERY),
            dispatch_dist_attn_runtime_mgr.get_position_ids(attn_role=AttnRole.QUERY),
        )
        assert torch.equal(
            dist_attn_runtime_mgr.get_position_ids(attn_role=AttnRole.KEY),
            dispatch_dist_attn_runtime_mgr.get_position_ids(attn_role=AttnRole.KEY),
        )

        # check calc_attn results
        # NOTE: native grpcoll does not support fp64
        if not magi_attention.comm.is_native_grpcoll_enable():
            self._calc_attn_with_mgr_and_assert_close_to_ref(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                head_dim=head_dim,
                dist_attn_runtime_mgr=dist_attn_runtime_mgr,
                dtype=torch.float64,
                test_case=test_config["test_case"],
            )

    def _calc_attn_with_mgr_and_assert_close_to_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
        total_seqlen_q: int,
        total_seqlen_k: int,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        dist_attn_runtime_mgr: DistAttnRuntimeMgr,
        dtype: torch.dtype = torch.float64,
        run_bwd: bool = True,
        test_case: str = "",
    ):
        total_q = torch.randn(
            total_seqlen_q,
            num_heads_q,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=run_bwd,
        )
        total_k = torch.randn(
            total_seqlen_k,
            num_heads_kv,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=run_bwd,
        )
        total_v = torch.randn(
            total_seqlen_k,
            num_heads_kv,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=run_bwd,
        )
        dist.all_reduce(total_q.data, group=self.nccl_group)
        dist.all_reduce(total_k.data, group=self.nccl_group)
        dist.all_reduce(total_v.data, group=self.nccl_group)

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

        local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

        total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

        if run_bwd:
            grad_total_out = torch.randn_like(total_out).detach()
            dist.all_reduce(grad_total_out.data, group=self.nccl_group)
            total_out.backward(grad_total_out)
            grad_total_q, grad_total_k, grad_total_v = (
                total_q.grad,
                total_k.grad,
                total_v.grad,
            )
        else:
            grad_total_q = None
            grad_total_k = None
            grad_total_v = None
            grad_total_out = None

        attn_type_map: list[int] = list(map(AttnMaskType.to_int_type, attn_mask_type))

        self._assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            total_q=total_q,
            total_k=total_k,
            total_v=total_v,
            total_out=total_out,
            grad_total_q=grad_total_q,
            grad_total_k=grad_total_k,
            grad_total_v=grad_total_v,
            grad_total_out=grad_total_out,
            run_bwd=run_bwd,
            test_case=test_case,
        )

    def _assert_close_to_torch_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        grad_total_q: torch.Tensor | None,
        grad_total_k: torch.Tensor | None,
        grad_total_v: torch.Tensor | None,
        grad_total_out: torch.Tensor | None,
        run_bwd: bool,
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance threshold  ---- #

        o_atol = EPSILON
        o_rtol = EPSILON

        dq_atol = EPSILON
        dq_rtol = EPSILON

        dk_atol = EPSILON
        dk_rtol = EPSILON

        dv_atol = EPSILON
        dv_rtol = EPSILON

        # -----   build attn mask   ---- #

        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref. torch ref with high precision (fp64)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_high_precision, _ = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            sink_layout="sh",
            high_precision=True,
            return_lse=False,
        )

        if run_bwd:
            total_out_ref_high_precision.backward(grad_total_out)
            (
                grad_total_q_ref_high_precision,
                grad_total_k_ref_high_precision,
                grad_total_v_ref_high_precision,
            ) = (
                total_q.grad,
                total_k.grad,
                total_v.grad,
            )

        # -----   init error message list   ---- #

        err_msg_list: list[str] = []

        # -----   assert close for fwd out   ---- #

        try:
            magi_attention.testing.assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        if run_bwd:
            # -----   assert close for bwd dq   ---- #

            try:
                magi_attention.testing.assert_close(
                    grad_total_q,
                    grad_total_q_ref_high_precision,
                    atol=dq_atol,
                    rtol=dq_rtol,
                    test_case=f"{test_case} => dq",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # -----   assert close for bwd dk   ---- #

            try:
                magi_attention.testing.assert_close(
                    grad_total_k,
                    grad_total_k_ref_high_precision,
                    atol=dk_atol,
                    rtol=dk_rtol,
                    test_case=f"{test_case} => dk",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # -----   assert close for bwd dv   ---- #

            try:
                magi_attention.testing.assert_close(
                    grad_total_v,
                    grad_total_v_ref_high_precision,
                    atol=dv_atol,
                    rtol=dv_rtol,
                    test_case=f"{test_case} => dv",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))


if __name__ == "__main__":
    run_tests()
