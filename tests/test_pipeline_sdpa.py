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

import random
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
import magi_attention.testing
from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.comm.primitive.grpcoll._buffer import GrpCollBuffer
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_mgr
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    GrpCollConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import (
    NAME,
    SKIP_WORLD_SIZE,
    DistTestBase,
    with_comms,
)
from magi_attention.testing.precision import (
    EPSILON,
    H100_MATMUL_MFU,
    H100_NVLINK_A2A_BWU,
    H100_NVLINK_BANDWIDTH,
    H100_TFLOPS_16,
    ref_attn_func,
)
from magi_attention.testing.utils import switch_sdpa_backend_decorator
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_attn_mask_from_ffa_args,
    get_calc_cost_factor,
    get_comm_cost_factor,
    str2seed,
    sync_rng,
)


class TestPipelineSDPABaseWithWorldSize1(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend=self.backend)
            for _ in range(2)
        ]

        # -----    set up for hier comm   ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
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
            self.device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
        else:
            self.device_mesh = None

        # -----    set up for native grpcoll   ---- #

        if magi_attention.comm.is_native_grpcoll_enable():
            for nccl_group in self.nccl_groups:
                grpcoll_mgr.register_buffer(
                    group=nccl_group,
                    config=GrpCollConfig(
                        num_nvl_bytes=int(2e9)
                        * self.world_size
                        // 8,  # 2GB for 8 ranks
                    ),
                )
                grpcoll_mgr.check_registered(group=nccl_group)

    def destroy_pg(self):
        # -----    clean up for native grpcoll   ---- #

        if magi_attention.comm.is_native_grpcoll_enable():
            for nccl_group in self.nccl_groups:
                grpcoll_mgr.release_buffer(group=nccl_group)
                grpcoll_mgr.check_released(group=nccl_group)

        super().destroy_pg()

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 1

    @switch_sdpa_backend_decorator
    @with_comms
    @parameterize(
        "attn_config",
        [
            # full attn with total seqlen 1k
            {
                NAME: "full_attn_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges([[0, 1024]]),
                "k_ranges": AttnRanges.from_ranges([[0, 1024]]),
                "attn_type_mapping": [0],
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 32,
            },
            # varlen full attn with total seqlen 1050
            {
                NAME: "varlen_full_attn_1050",
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
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
                "chunk_size": 5,
            },
            # varlen full attn with total seqlen 1k
            # but reverse k ranges
            {
                NAME: "reverse_varlen_full_attn_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # varlen block causal with total seqlen 960
            {
                NAME: "varlen_block_causal_960",
                SKIP_WORLD_SIZE: [7, 8],
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
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 16,
            },
            # varlen block causal with total seqlen 840
            {
                NAME: "varlen_block_causal_840",
                SKIP_WORLD_SIZE: [4, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 840],
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
                        [768, 840],
                    ]
                ),
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 840,
                "total_seqlen_k": 840,
                "chunk_size": 4,
            },
            # varlen block causal with total seqlen 1k
            # but reverse k ranges
            {
                NAME: "reverse_varlen_block_causal_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                        [512, 1024],
                        [512, 896],
                        [512, 768],
                        [512, 640],
                        [0, 512],
                        [0, 384],
                        [0, 256],
                        [0, 128],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # varlen block causal with total seqlen 1k
            # but as upper diagonal matrices
            {
                NAME: "upper_diagonal_varlen_block_causal_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                        [0, 512],
                        [128, 512],
                        [256, 512],
                        [384, 512],
                        [512, 1024],
                        [640, 1024],
                        [768, 1024],
                        [896, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # block sliding-window full with total seqlen 1k
            {
                NAME: "block_slide_window_full_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                        [0, 384],
                        [0, 512],
                        [0, 640],
                        [128, 768],
                        [256, 896],
                        [384, 1024],
                        [512, 1024],
                        [640, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # block sliding-window causal with total seqlen 1k
            {
                NAME: "block_slide_window_causal_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                        [0, 128],
                        [0, 256],
                        [0, 384],
                        [128, 512],
                        [256, 640],
                        [384, 768],
                        [512, 896],
                        [640, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # block sliding-window causal with total seqlen 1k
            # but reverse k ranges
            {
                NAME: "reverse_block_slide_window_causal_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
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
                        [640, 1024],
                        [512, 896],
                        [384, 768],
                        [256, 640],
                        [128, 512],
                        [0, 384],
                        [0, 256],
                        [0, 128],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # share question mask with total seqlen 1k + overlapped q ranges
            {
                NAME: "share_question_1k_with_q_overlap",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [128, 256],
                        [256, 512],
                        [512, 1024],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 128],
                        [128, 256],
                        [256, 512],
                        [512, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 4,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # varlen block causal with total seqlen 1k + overlapped q ranges
            {
                NAME: "varlen_block_causal_1k_with_q_overlap",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [128, 1024],
                        [256, 1024],
                        [384, 1024],
                        [512, 1024],
                        [640, 1024],
                        [768, 1024],
                        [896, 1024],
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
                        [768, 896],
                        [896, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 8,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # several random mask in overlap q_range with 1k mask
            {
                NAME: "random_overlap_mask_1k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 64],
                        [96, 128],
                        [32, 96],
                        [128, 320],
                        [192, 256],
                        [200, 448],
                        [700, 896],
                        [768, 832],
                        [640, 800],
                        [896, 1024],
                        [928, 960],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 32],
                        [16, 64],
                        [96, 128],
                        [128, 256],
                        [256, 384],
                        [384, 512],
                        [512, 640],
                        [640, 768],
                        [768, 896],
                        [896, 960],
                        [960, 1024],
                    ]
                ),
                "attn_type_mapping": [0] * 11,
                "total_seqlen_q": 1024,
                "total_seqlen_k": 1024,
                "chunk_size": 128,
            },
            # varlen block causal with total seqlen 840 + overlapped q ranges
            {
                NAME: "varlen_block_causal_840_with_q_overlap",
                SKIP_WORLD_SIZE: [4, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 512],
                        [128, 512],
                        [256, 512],
                        [384, 512],
                        [512, 840],
                        [640, 840],
                        [768, 840],
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
                        [768, 840],
                    ]
                ),
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 840,
                "total_seqlen_k": 840,
                "chunk_size": 4,
            },
            # half-inv block diagonal with total seqlen 1050
            # + interleaved overlapped q ranges
            {
                NAME: "half_inv_block_diagonal_1050_with_interleave_q_overlap",
                SKIP_WORLD_SIZE: [4, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 420],
                        [210, 630],
                        [420, 840],
                        [630, 1050],
                        [0, 210],
                        [840, 1050],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 210],
                        [210, 420],
                        [420, 630],
                        [630, 840],
                        [840, 1050],
                        [840, 1050],
                    ]
                ),
                "attn_type_mapping": [0] * 6,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
                "chunk_size": 5,
            },
        ],
    )
    @parameterize(
        # TODO:
        #   1. test non-trivial algorithms
        #   2. profile real comm/calc factors
        "overlap_config",
        [
            # disable multi-stage overlap
            {
                NAME: "disable_mso",
                "enable": False,
            },
            # static, overlap degree = 1, min chunk size = 15
            {
                NAME: "static_od1_cz15",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 1,
                "min_chunk_size": 15,
                "max_num_chunks": 60,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
            # static, overlap degree = 4, min chunk size = 23
            {
                NAME: "static_od4_cz23",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 13,
                "max_num_chunks": 52,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
            # dynamic, min chunk size = 56, no max overlap degree limit
            {
                NAME: "dynamic_cz56",
                "enable": True,
                "mode": AttnOverlapMode.DYNAMIC,
                "degree": None,
                "dynamic_max_degree": None,
                "min_chunk_size": 12,
                "max_num_chunks": 65,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
        ],
    )
    @parameterize(
        "num_heads",
        [(6, 1)],  # mqa
    )
    @parameterize(
        "head_dim",
        [64],
    )
    @parameterize(
        "dtype",
        [torch.float64],
    )
    @parameterize(
        "random_type_mapping",
        [False, True],
    )
    def test_pipeline_sdpa(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: tuple[int, int],  # (nhq, nhkv)
        head_dim: int,
        dtype: torch.dtype,
        random_type_mapping: bool,
        run_bwd: bool = True,
    ):
        # NOTE: test pipeline using sdpa does not need profile mode
        # thus we always enable sanity check mode
        assert magi_attention.is_sanity_check_enable()
        assert magi_attention.is_sdpa_backend_enable()

        # -----    skip for world size   ---- #

        if (
            attn_config.get(SKIP_WORLD_SIZE, [])
            and self.world_size in attn_config[SKIP_WORLD_SIZE]
        ):
            return

        # -----    skip for mso   ---- #

        if magi_attention.comm.is_qo_comm_enable():
            # TODO: support mso for qo comm
            if overlap_config[NAME] != "disable_mso":
                return

        # -----    skip for native grpcoll   ---- #

        if magi_attention.comm.is_native_grpcoll_enable():
            hidden_size_kv = num_heads[1] * head_dim
            if hidden_size_kv % GrpCollBuffer.get_hidden_size_alignment(dtype) != 0:
                return

            # TODO: for now, native grpcoll only supports fp32 lse comm
            # thus it cannot pass this test requiring fp64 lse
            if magi_attention.comm.is_qo_comm_enable():
                return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=} | \n\n{overlap_config=}"

        test_case = (
            f"world_size=[{self.world_size}] x "
            f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x "
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})] x "
            f"random_causal_mapping=[{random_type_mapping}] x "
        )
        test_case_seed = str2seed(test_case)

        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        attn_type_mapping: list[int] = attn_config["attn_type_mapping"]
        if random_type_mapping:
            # NOTE: to test causal mapping, we design a mode to just use random `attn_type_mapping`
            # instead of hard-coded config in the test cases
            with sync_rng(seed=test_case_seed):
                attn_type_mapping = [
                    random.choice([0, 1, 2, 3]) for _ in attn_type_mapping
                ]

        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]
        num_heads_q, num_heads_kv = num_heads
        softmax_scale = (  # choose softmax_scale by rule
            None if test_case_seed % 2 == 0 else (1 / head_dim)
        )
        softcap = 0.0  # not supported for test

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(
                # TODO: test other dispatch algs
                alg=MinHeapDispatchAlg()
            ),
            overlap_config=OverlapConfig(
                **{k: v for k, v in overlap_config.items() if k not in (NAME,)},
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=num_heads_q,
                    head_dim=head_dim,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(self.world_size),
                ),
            ),
            # NOTE: this config is useless for this test
            # since we already register/release buffer in `init_pg`/`destroy_pg`
            grpcoll_config=GrpCollConfig(),
        )

        # -----    run pipeline test   ---- #

        # -----   init attn_mask_type ----- #

        attn_mask_type: list[AttnMaskType] = list(
            map(AttnMaskType.from_int_type, attn_type_mapping)
        )

        # -----    init dist attn runtime mgr   ---- #

        dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=dist_attn_config,
            cp_mesh=self.device_mesh,
        )
        # HACK: seperate cp group for group-reduce
        dist_attn_runtime_mgr.dist_attn_runtime.cp_group_gr = self.nccl_groups[1]

        # -----   init global qkv   ---- #

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

        # -----   dispatch global qkv to local qkv   ---- #

        local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

        # -----   run dist attn forward on local qkv for local out/lse   ---- #

        local_out, local_lse = dist_attn_runtime_mgr.calc_attn(
            q=local_q,
            k=local_k,
            v=local_v,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )

        # -----   undispatch local out/lse to global out/lse   ---- #

        total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
        total_lse = dist_attn_runtime_mgr.undispatch_qo(local_lse)

        # -----   run backward   ---- #

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
            grad_total_out = None
            grad_total_q, grad_total_k, grad_total_v = None, None, None

        # -----   assert close to torch ref   ---- #

        self._assert_close_to_torch_ref(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_mapping,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            softmax_scale=softmax_scale,
            softcap=softcap,
            total_q=total_q,
            total_k=total_k,
            total_v=total_v,
            total_out=total_out,
            total_lse=total_lse,
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
        softmax_scale: float | None,
        softcap: float,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        total_lse: torch.Tensor,
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

        lse_atol = EPSILON
        lse_rtol = EPSILON

        dq_atol = EPSILON
        dq_rtol = EPSILON

        dk_atol = EPSILON
        dk_rtol = EPSILON

        dv_atol = EPSILON
        dv_rtol = EPSILON

        # -----   build attn mask   ---- #

        mask = get_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref. torch ref with high precision (fp64)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_high_precision, total_lse_ref_high_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            softmax_scale=softmax_scale,
            softcap=softcap,
            layout="thd",
            high_precision=True,
            return_lse=True,
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

        # -----   assert close for fwd lse   ---- #

        try:
            magi_attention.testing.assert_close(
                total_lse,
                total_lse_ref_high_precision,
                atol=lse_atol,
                rtol=lse_rtol,
                test_case=f"{test_case} => lse",
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


class TestPipelineSDPAWithWorldSize2(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize3(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize4(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize5(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize6(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize7(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


class TestPipelineSDPAWithWorldSize8(TestPipelineSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_pipeline_sdpa(self, *args, **kwargs):
        super().test_pipeline_sdpa(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
