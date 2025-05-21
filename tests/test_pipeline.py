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
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import (
    EPSILON,
    calc_inf_norm,
    extract_mismatch_info,
    torch_attn_ref,
)
from magi_attention.utils import get_attn_mask_from_ranges, str2seed, sync_rng
from magi_attention.utils._utils import is_list_value_all

# tell if using profile mode
profile_mode = os.environ.get("MAGI_ATTENTION_UNITEST_PROFILE_MODE", "0") == "1"

PROFILE_ONLY = "profile_only"
NAME = "name"
SKIP_WORLD_SIZE = "skip_world_size"


IB_BANDWIDTH = 50e9  # 500 GB/s, single-end

# H100 spec: https://www.nvidia.com/en-us/data-center/h100/
H100_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H100_NVLINK_BANDWIDTH = 450e9  # 450 GB/s, single-end

# H800 spec: https://chaoqing-i.com/upload/20231128/NVIDIA%20H800%20GPU%20Datasheet.pdf
H800_TFLOPS_16 = 989.5e12  # 989 teraFLOPS
H800_NVLINK_BANDWIDTH = 200e9  # 200 GB/s, single-end

# A100 spec: https://www.nvidia.com/en-us/data-center/a100/
A100_TFLOPS_16 = 312e12  # 312 teraFLOPS
A100_NVLINK_BANDWIDTH = 300e9  # 300 GB/s, single-end


# assuming that:
#   num_heads (nh) = 1, head_dim (hd) = 128
#   mfu = 0.5, bwu = 0.6
#   cp = 4, a2a_corr_factor = (cp-1)/cp = 0.75
#   unit: μs
NUM_HEADS = 1
HEAD_DIM = 128
MFU = 0.5
BWU = 0.6
A2A_CORR_FACTOR = 0.75
SEC_RATIO = 1e6  # 1s = 1e6 μs

# formula:
#   calc cost factor = 2 * 2 * nh * hd / TFLOPS / mfu * sec_ratio
#   comm cost factor = 2 * nh * hd / BANDWIDTH / a2a_corr_factor / bwu * sec_ratio
# then:
CALC_COST_FACTOR = 2 * 2 * NUM_HEADS * HEAD_DIM / H800_TFLOPS_16 / MFU * SEC_RATIO
INTRA_NODE_COMM_COST_FACTOR = (
    2 * NUM_HEADS * HEAD_DIM / H800_NVLINK_BANDWIDTH / A2A_CORR_FACTOR / BWU * SEC_RATIO
)
INTER_NODE_COMM_COST_FACTOR = (
    2 * NUM_HEADS * HEAD_DIM / IB_BANDWIDTH / A2A_CORR_FACTOR / BWU * SEC_RATIO
)


class TestPipelineBaseWithWorldSize1(DistTestBase):
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

        if magi_attention.is_hierarchical_comm_enable() and self.world_size in [
            4,
            6,
            8,
        ]:
            world_size_inter_node, world_size_intra_node = {
                4: (2, 2),
                6: (3, 2),
                8: (2, 4),
            }[self.world_size]
            device_mesh = init_device_mesh(
                device_type="cuda",
                mesh_shape=(world_size_inter_node, world_size_intra_node),
                mesh_dim_names=("inter", "intra"),
            )
            self.intra_group = device_mesh.get_group("intra")
            self.inter_group = device_mesh.get_group("inter")
        else:
            self.intra_group = None
            self.inter_group = None

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
            # full attn with total seqlen 14k
            {
                NAME: "full_attn_14k",
                SKIP_WORLD_SIZE: [3, 5, 6, 8],
                "q_ranges": AttnRanges.from_ranges([[0, 14336]]),
                "k_ranges": AttnRanges.from_ranges([[0, 14336]]),
                "is_causal_mapping": [False],
                "total_seqlen_q": 14336,
                "total_seqlen_k": 14336,
                "chunk_size": 512,
            },
            # varlen full attn with total seqlen 12k
            {
                NAME: "varlen_full_attn_12k",
                SKIP_WORLD_SIZE: [5, 7],
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
                "is_causal_mapping": [False] * 6,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # varlen block causal with total seqlen 15k
            {
                NAME: "varlen_block_causal_15k",
                SKIP_WORLD_SIZE: [4, 7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 15360],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 4096],
                        [0, 6144],
                        [0, 8192],
                        [8192, 10240],
                        [8192, 12288],
                        [12288, 15360],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 15360,
                "total_seqlen_k": 15360,
                "chunk_size": 512,
            },
            # varlen block causal with total seqlen 17k
            {
                NAME: "varlen_block_causal_17k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [12288, 17808],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 4096],
                        [0, 6144],
                        [0, 8192],
                        [8192, 10240],
                        [8192, 12288],
                        [12288, 17808],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 17808,
                "total_seqlen_k": 17808,
                "chunk_size": 1113,
            },
            # varlen block causal with total seqlen 10k + overlapped q ranges
            {
                NAME: "varlen_block_causal_10k_with_q_overlap",
                SKIP_WORLD_SIZE: [3, 6, 7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 10240],
                        [1280, 10240],
                        [2560, 10240],
                        [3840, 10240],
                        [5120, 10240],
                        [6400, 10240],
                        [7680, 10240],
                        [8960, 10240],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1280],
                        [1280, 2560],
                        [2560, 3840],
                        [3840, 5120],
                        [5120, 6400],
                        [6400, 7680],
                        [7680, 8960],
                        [8960, 10240],
                    ]
                ),
                "is_causal_mapping": [False] * 8,
                "total_seqlen_q": 10240,
                "total_seqlen_k": 10240,
                "chunk_size": 512,
            },
            # varlen block causal with total seqlen 12k + overlapped q ranges
            {
                NAME: "varlen_block_causal_12k_with_q_overlap",
                SKIP_WORLD_SIZE: [5, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 8192],
                        [2048, 8192],
                        [4096, 8192],
                        [6144, 8192],
                        [8192, 12288],
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
                "is_causal_mapping": [False] * 6,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # half-inv block diagonal with total seqlen 10k
            # + interleaved overlapped q ranges
            {
                NAME: "varlen_block_causal_12k_with_q_overlap",
                SKIP_WORLD_SIZE: [2, 4, 5, 6, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 3072],
                        [1536, 4608],
                        [3072, 6144],
                        [4608, 7680],
                        [6144, 9216],
                        [7680, 10752],
                        [9216, 10752],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1536],
                        [1536, 3072],
                        [3072, 4608],
                        [4608, 6144],
                        [6144, 7680],
                        [7680, 9216],
                        [9216, 10752],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 10752,
                "total_seqlen_k": 10752,
                "chunk_size": 512,
            },
            # NOTE: profile only case
            # full attn with total seqlen 140k
            # {
            #     PROFILE_ONLY: True,
            #     NAME: "full_attn_140k",
            #     SKIP_WORLD_SIZE: [1, 2, 3, 5, 6, 7, 8],
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 143360],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 143360],
            #         ]
            #     ),
            #     "is_causal_mapping": [False],
            #     "total_seqlen_q": 143360,
            #     "total_seqlen_k": 143360,
            #     "chunk_size": 2048,
            # },
            # NOTE: profile only case
            # varlen block causal with total seqlen 144k
            {
                PROFILE_ONLY: True,
                NAME: "varlen_block_causal_144k",
                SKIP_WORLD_SIZE: [1, 2, 3, 5, 6, 7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 20480],
                        [20480, 40960],
                        [40960, 61440],
                        [61440, 81920],
                        [81920, 102400],
                        [102400, 122880],
                        [122880, 147456],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 20480],
                        [0, 40960],
                        [0, 61440],
                        [0, 81920],
                        [81920, 102400],
                        [81920, 122880],
                        [122880, 147456],
                    ]
                ),
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 147456,
                "total_seqlen_k": 147456,
                "chunk_size": 4096,
            },
        ],
    )
    @parameterize(
        # TODO:
        #   1. test non-trivial algorithms
        #   2. profile real comm/calc factors
        "overlap_config",
        [
            # disable multi-stage overlap to roll back to the original code
            {
                NAME: "disable_mso",
                "enable": False,
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 1, min chunk size = 1023
            {
                NAME: "static_od1_cz1023",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 1,
                "min_chunk_size": 1023,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 2, min chunk size = 513
            {
                NAME: "static_od2_cz513",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 2,
                "min_chunk_size": 513,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 4, min chunk size = 253
            {
                NAME: "static_od4_cz253",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 253,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # dynamic, min chunk size = 256, no max overlap degree limit
            {
                NAME: "dynamic_cz256",
                "enable": True,
                "mode": AttnOverlapMode.DYNAMIC,
                "degree": None,
                "dynamic_max_degree": None,
                "min_chunk_size": 256,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # NOTE: profile only case
            # static, overlap degree = 4, min chunk size = 512, max num chunks = 64
            {
                PROFILE_ONLY: True,
                NAME: "static_d4",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 4,
                "min_chunk_size": 512,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # NOTE: profile only case
            # dynamic, min chunk size = 512, max num chunks = 64, max overlap degree = 8
            # {
            #     PROFILE_ONLY: True,
            #     NAME: "dynamic_md8",
            #     "enable": True,
            #     "mode": AttnOverlapMode.DYNAMIC,
            #     "degree": None,
            #     "dynamic_max_degree": 8,
            #     "min_chunk_size": 512,
            #     "max_num_chunks": 64,
            #     "alg": UniformOverlapAlg(
            #         random_costs=True,
            #         random_seed=42,
            #     ),
            #     "calc_cost_factor": CALC_COST_FACTOR,
            #     "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            # },
        ],
    )
    @parameterize(
        "num_heads",
        [NUM_HEADS],
    )
    @parameterize(
        "head_dim",
        [HEAD_DIM],
    )
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    @parameterize(
        "random_causal_mapping",
        [False, True],
    )
    @parameterize(
        "high_bandwith_domain_size",
        [1, 2, 4, 8],
    )
    def test_pipeline(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        random_causal_mapping: bool,
        high_bandwith_domain_size: int,
    ):
        # -----    switch mode   ---- #

        if profile_mode:
            prof_iters, prof_start_iter, prof_end_iter = 10, 4, 6
        else:
            prof_iters, prof_start_iter, prof_end_iter = 1, -1, -1
            assert magi_attention.is_sanity_check_enable()

        if profile_mode ^ attn_config.get(PROFILE_ONLY, False):
            return
        if profile_mode ^ overlap_config.get(PROFILE_ONLY, False):
            return

        # -----    skip for world size   ---- #

        if (
            attn_config.get(SKIP_WORLD_SIZE, [])
            and self.world_size in attn_config[SKIP_WORLD_SIZE]
        ):
            return
        if (
            self.world_size % high_bandwith_domain_size != 0
            or high_bandwith_domain_size > self.world_size
        ):
            # skip for invalid high_bandwith_domain_size
            return

        # -----    skip for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable():
            if self.world_size not in [4, 6, 8]:
                # skip for invalid world size
                # when hierarchical comm is enabled
                return
            if high_bandwith_domain_size > 1:
                return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=} | \n\n{overlap_config=}"

        test_case = (
            f"world_size=[{self.world_size}] x high_bandwith_domain_size=[{high_bandwith_domain_size}] x "
            f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x "
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})] x "
            f"random_causal_mapping=[{random_causal_mapping}]"
        )

        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        is_causal_mapping: list[bool] = attn_config["is_causal_mapping"]
        if random_causal_mapping:
            # NOTE: to test causal mapping, we design a mode to just use random `is_causal_mapping`
            # instead of hard-coded config in the test cases
            with sync_rng(seed=str2seed(test_case)):
                is_causal_mapping = [
                    random.choice([True, False]) for _ in is_causal_mapping
                ]

        # -----    skip for overlapped q_range with causal mask  ---- #

        if not q_ranges.is_non_overlap() and not is_list_value_all(
            is_causal_mapping, False
        ):
            return

        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]

        device = torch.cuda.current_device()

        dist_attn_config = DistAttnConfig(
            # TODO: test top-p minhp dispatch alg
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                **{
                    k: v
                    for k, v in overlap_config.items()
                    if k not in (NAME, PROFILE_ONLY)
                }
            ),
            high_bandwith_domain_size=high_bandwith_domain_size,
            deterministic=False,
        )

        # -----    run pipeline test   ---- #

        for iter in range(prof_iters):
            # -----    profile control if using profile mode   ---- #

            if profile_mode:
                if self.rank == 0 and iter == prof_start_iter:
                    torch.cuda.profiler.start()
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
                if self.rank == 0 and iter == prof_end_iter:
                    torch.cuda.profiler.stop()

            # -----    barrier at the beginning of each iteration   ---- #

            dist.barrier()
            torch.cuda.synchronize()

            # -----    init dist attn runtime mgr   ---- #

            dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=[
                    AttnMaskType.CAUSAL if is_causal else AttnMaskType.FULL
                    for is_causal in is_causal_mapping
                ],
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                chunk_size=chunk_size,
                cp_group=self.nccl_group,
                is_same_source=True,
                is_q_permutable=True,
                is_k_permutable=True,
                dist_attn_config=dist_attn_config,
                cp_intra_group=self.intra_group,
                cp_inter_group=self.inter_group,
            )
            if not magi_attention.is_hierarchical_comm_enable():
                # HACK: double cp group for kv/dkv
                dist_attn_runtime_mgr.dist_attn_runtime.cp_group_dkv = self.nccl_groups[
                    1
                ]

            # -----   init global qkv   ---- #

            total_q = torch.randn(
                total_seqlen_q,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            total_k = torch.randn(
                total_seqlen_k,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            total_v = torch.randn(
                total_seqlen_k,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            dist.all_reduce(total_q.data, group=self.nccl_group)
            dist.all_reduce(total_k.data, group=self.nccl_group)
            dist.all_reduce(total_v.data, group=self.nccl_group)

            # -----   dispatch global qkv to local qkv   ---- #

            local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
            local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
            local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

            # -----   run dist attn forward on local qkv for local o   ---- #

            local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

            # -----   undispatch local o to global o   ---- #

            total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

            # -----   run backward   ---- #

            grad_total_out = torch.randn_like(total_out).detach()
            dist.all_reduce(grad_total_out.data, group=self.nccl_group)
            total_out.backward(grad_total_out)
            grad_total_q, grad_total_k, grad_total_v = (
                total_q.grad,
                total_k.grad,
                total_v.grad,
            )

            # -----   assert close if not using profile mode   ---- #

            if not profile_mode:
                # -----   assert close to torch ref   ---- #

                self.assert_close_to_torch_ref(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    is_causal_mapping=is_causal_mapping,
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
                    dtype=dtype,
                    test_case=test_case,
                )

    def assert_close_to_torch_ref(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        is_causal_mapping: list[bool],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_out: torch.Tensor,
        grad_total_q: torch.Tensor,
        grad_total_k: torch.Tensor,
        grad_total_v: torch.Tensor,
        grad_total_out: torch.Tensor,
        dtype: torch.dtype,
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance threshold  ---- #

        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)

        # NOTE: an experimental value from magi_attention testing
        mismatch_thres_ratio: float = 2.0
        # NOTE: an experimental value from fa testing
        norm_rtol_ratio: float = 2.0

        # -----   build attn mask   ---- #

        mask = get_attn_mask_from_ranges(
            q_ranges=q_ranges.to_naive_ranges(),
            k_ranges=k_ranges.to_naive_ranges(),
            is_causal_mapping=is_causal_mapping,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
        )

        # -----   ref1. torch ref with high precision (fp32)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_high_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=True,
        )
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

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None

        total_out_ref_low_precision = torch_attn_ref(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            layout="thd",
            high_precision=False,
        )
        total_out_ref_low_precision.backward(grad_total_out)
        (
            grad_total_q_ref_low_precision,
            grad_total_k_ref_low_precision,
            grad_total_v_ref_low_precision,
        ) = (
            total_q.grad,
            total_k.grad,
            total_v.grad,
        )

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
        out_ref_norm = calc_inf_norm(
            total_out_ref_low_precision, total_out_ref_high_precision
        )
        self.assertLessEqual(
            out_norm,
            norm_rtol_ratio * out_ref_norm,
            msg=f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        o_thres = self._extract_mismatch_threshold_ref(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        magi_attention.testing.assert_close(
            total_out,
            total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_threshold=o_thres,
            test_case=f"{test_case} => o",
        )

        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
        dq_ref_norm = calc_inf_norm(
            grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
        )
        self.assertLessEqual(
            dq_norm,
            norm_rtol_ratio * dq_ref_norm,
            msg=f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dq_thres = self._extract_mismatch_threshold_ref(
            actual=grad_total_q_ref_low_precision,
            expected=grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        magi_attention.testing.assert_close(
            grad_total_q,
            grad_total_q_ref_high_precision,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_threshold=dq_thres,
            test_case=f"{test_case} => dq",
        )

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
        dk_ref_norm = calc_inf_norm(
            grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
        )
        self.assertLessEqual(
            dk_norm,
            norm_rtol_ratio * dk_ref_norm,
            msg=f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dk_thres = self._extract_mismatch_threshold_ref(
            actual=grad_total_k_ref_low_precision,
            expected=grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        magi_attention.testing.assert_close(
            grad_total_k,
            grad_total_k_ref_high_precision,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_threshold=dk_thres,
            test_case=f"{test_case} => dk",
        )

        # -----   assert close for bwd dv   ---- #

        # fa style with Linf norm
        dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
        dv_ref_norm = calc_inf_norm(
            grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
        )
        self.assertLessEqual(
            dv_norm,
            norm_rtol_ratio * dv_ref_norm,
            msg=f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dv_thres = self._extract_mismatch_threshold_ref(
            actual=grad_total_v_ref_low_precision,
            expected=grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        magi_attention.testing.assert_close(
            grad_total_v,
            grad_total_v_ref_high_precision,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_threshold=dv_thres,
            test_case=f"{test_case} => dv",
        )

    def _extract_mismatch_threshold_ref(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        atol: float,
        rtol: float,
        mismatch_thres_ratio: float = 1.0,
    ) -> float:
        mismatch_threshold_ref = 0.0
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        except AssertionError as e:
            error_msg = str(e)
            _, _, mismatch_threshold_ref = extract_mismatch_info(error_msg)

        return min(max(mismatch_threshold_ref * mismatch_thres_ratio, 0.0), 1.0)


class TestPipelineWithWorldSize2(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize3(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize4(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize5(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize6(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize7(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


class TestPipelineWithWorldSize8(TestPipelineBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_pipeline(self, *args, **kwargs):
        super().test_pipeline(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
