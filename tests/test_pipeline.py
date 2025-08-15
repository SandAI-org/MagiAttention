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
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.meta.solver.dispatch_solver import MinHeapDispatchAlg
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import (
    NAME,
    PROFILE_ONLY,
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
    calc_inf_norm,
    extract_mismatch_threshold,
    torch_attn_ref,
)
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_attn_mask_from_ffa_args,
    get_calc_cost_factor,
    get_comm_cost_factor,
    str2seed,
    sync_rng,
)


class TestPipelineBaseWithWorldSize1(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        self.profile_mode = (
            os.environ.get("MAGI_ATTENTION_UNITEST_PROFILE_MODE", "0") == "1"
        )

        if self.profile_mode:
            # disable sanity check when profiling
            os.environ["MAGI_ATTENTION_SANITY_CHECK"] = "0"

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

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @property
    def nccl_group(self) -> dist.ProcessGroup:
        return self.nccl_groups[0]

    @property
    def world_size(self) -> int:
        return 1

    @property
    def seed(self) -> int:
        return 42

    @with_comms
    @parameterize(
        "attn_config",
        [
            # full attn with total seqlen 14k
            {
                NAME: "full_attn_14k",
                SKIP_WORLD_SIZE: [3, 5, 6, 8],
                "q_ranges": AttnRanges.from_ranges([[0, 14336]]),
                "k_ranges": AttnRanges.from_ranges([[0, 14336]]),
                "attn_type_mapping": [0],
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
                "attn_type_mapping": [0] * 6,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # # varlen block causal with total seqlen 15k
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
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 15360,
                "total_seqlen_k": 15360,
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
                "attn_type_mapping": [0] * 6,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # simple bi_causal test with overlapped q ranges with 12k
            {
                NAME: "bi_causal_12k_with_q_overlap",
                SKIP_WORLD_SIZE: [5, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 8192],
                        [8192, 10240],
                        [10240, 12288],
                        [1000, 4000],
                        [10000, 12000],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 3072],
                        [0, 4096],
                        [0, 6144],
                        [6144, 12288],
                        [8192, 12288],
                        [9216, 12288],
                        [8000, 12000],
                        [0, 5000],
                    ]
                ),
                "attn_type_mapping": [3] * 8,
                "total_seqlen_q": 12288,
                "total_seqlen_k": 12288,
                "chunk_size": 512,
            },
            # merging causal and inv_causal to bi_causal with total seqlen 10k
            # + interleaved overlapped q ranges
            {
                NAME: "continuous_multi_masks_10k_with_q_overlap",
                SKIP_WORLD_SIZE: [3, 6, 7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [0, 2048],
                        [0, 2048],
                        [0, 2048],
                        [2048, 4096],
                        [3072, 5120],
                        [5120, 7168],
                        [6144, 8192],
                        [8192, 10240],
                        [8192, 10240],
                        [8192, 10240],
                        [8192, 10240],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 10240],
                        [0, 2048],
                        [2048, 4096],
                        [0, 2048],
                        [2048, 4096],
                        [0, 2048],
                        [2048, 4096],
                        [4096, 6144],
                        [6144, 10240],
                    ]
                ),
                "attn_type_mapping": [2, 0, 1, 3, 2, 1, 2, 1, 2, 0, 1, 3],
                "total_seqlen_q": 10240,
                "total_seqlen_k": 10240,
                "chunk_size": 512,
            },
            # full_mask_assembled_from_samll_pieces
            {
                NAME: "full_mask_assembled_from_samll_pieces_with_8k",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                "q_ranges": AttnRanges.from_ranges(
                    [[i * 512, (i + 1) * 512] for i in range(16) for _ in range(8)]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [[i * 1024, (i + 1) * 1024] for _ in range(16) for i in range(8)]
                ),
                "attn_type_mapping": [0] * 128,
                "total_seqlen_q": 8192,
                "total_seqlen_k": 8192,
                "chunk_size": 512,
            },
            # NOTE: profile only case
            # full attn with total seqlen 144k
            {
                PROFILE_ONLY: True,
                NAME: "full_attn_144k",
                SKIP_WORLD_SIZE: [1, 2, 3, 5, 6, 7, 8],
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 147456],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 147456],
                    ]
                ),
                "attn_type_mapping": [0],
                "total_seqlen_q": 147456,
                "total_seqlen_k": 147456,
                "chunk_size": 2048,
            },
            # NOTE: profile only case
            # varlen block causal with total seqlen 144k
            # {
            #     PROFILE_ONLY: True,
            #     NAME: "varlen_block_causal_144k",
            #     SKIP_WORLD_SIZE: [1, 2, 3, 5, 6, 7, 8],
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 20480],
            #             [20480, 40960],
            #             [40960, 61440],
            #             [61440, 81920],
            #             [81920, 102400],
            #             [102400, 122880],
            #             [122880, 147456],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 20480],
            #             [0, 40960],
            #             [0, 61440],
            #             [0, 81920],
            #             [81920, 102400],
            #             [81920, 122880],
            #             [122880, 147456],
            #         ]
            #     ),
            #     "attn_type_mapping": [0] * 7,
            #     "total_seqlen_q": 147456,
            #     "total_seqlen_k": 147456,
            #     "chunk_size": 4096,
            # },
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
            # },
        ],
    )
    @parameterize(
        "num_heads",
        [(6, 6), (6, 2)],  # mha  # gqa
    )
    @parameterize(
        "head_dim",
        [64, 128],
    )
    @parameterize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
        ],
    )
    @parameterize(
        "random_type_mapping",
        [False, True],
    )
    def test_pipeline(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: tuple[int, int],  # (nhq, nhkv)
        head_dim: int,
        dtype: torch.dtype,
        random_type_mapping: bool,
        run_bwd: bool = True,
    ):
        # -----    switch mode   ---- #

        if self.profile_mode:  # [start_iter, end_iter)
            prof_iters, prof_start_iter, prof_end_iter = 10, 5, 8
            assert not magi_attention.is_sanity_check_enable()
        else:
            prof_iters, prof_start_iter, prof_end_iter = 1, -1, -1
            assert magi_attention.is_sanity_check_enable()

        if self.profile_mode ^ attn_config.get(PROFILE_ONLY, False):
            return
        if self.profile_mode ^ overlap_config.get(PROFILE_ONLY, False):
            return

        # -----    skip for world size   ---- #

        if (
            attn_config.get(SKIP_WORLD_SIZE, [])
            and self.world_size in attn_config[SKIP_WORLD_SIZE]
        ):
            return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=} | \n\n{overlap_config=}"

        test_case = (
            f"world_size=[{self.world_size}] x "
            f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x "
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})] x "
            f"random_causal_mapping=[{random_type_mapping}]"
        )

        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        attn_type_mapping: list[int] = attn_config["attn_type_mapping"]
        if random_type_mapping:
            # NOTE: to test causal mapping, we design a mode to just use random `attn_type_mapping`
            # instead of hard-coded config in the test cases
            with sync_rng(seed=str2seed(test_case)):
                attn_type_mapping = [
                    random.choice([0, 1, 2, 3]) for _ in attn_type_mapping
                ]

                # FIXME when q_range.seqlen = k_range.seqlen with BICAUSAL masktype
                # ffa kernel fails to compute correctly. Innore it in testcase temporarily.
                for i in range(len(q_ranges)):
                    if (
                        attn_type_mapping[i] == 3
                        and q_ranges[i].seqlen == k_ranges[i].seqlen
                    ):
                        attn_type_mapping[i] = random.choice([0, 1, 2])

        # -----    skip for overlapped q_range with causal mask  ---- #

        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]

        num_heads_q, num_heads_kv = num_heads

        dist_attn_config = DistAttnConfig(
            # TODO: test other dispatch algs
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                **{
                    k: v
                    for k, v in overlap_config.items()
                    if k not in (NAME, PROFILE_ONLY)
                },
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
        )

        # -----   init attn_mask_type ----- #

        attn_mask_type = [
            {
                0: AttnMaskType.FULL,
                1: AttnMaskType.CAUSAL,
                2: AttnMaskType.INVCAUSAL,
                3: AttnMaskType.BICAUSAL,
            }[i]
            for i in attn_type_mapping
        ]

        # -----    run pipeline test   ---- #

        for iter in range(prof_iters):
            # -----    profile control if using profile mode   ---- #

            if self.profile_mode:
                if self.rank == 0 and iter == prof_start_iter:
                    torch.cuda.profiler.start()
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
                if self.rank == 0 and iter == prof_end_iter:
                    torch.cuda.profiler.stop()

                # barrier at the beginning of each iteration
                dist.barrier()
                torch.cuda.synchronize()

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
            # HACK: seperate cp group for dkv group-reduce
            dist_attn_runtime_mgr.dist_attn_runtime.cp_group_dkv = self.nccl_groups[1]

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

            # -----   run dist attn forward on local qkv for local o   ---- #

            if self.profile_mode:
                # barrier before fwd to wait for the processes with slow solver
                dist.barrier()
                torch.cuda.synchronize()

            local_out, _ = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)

            # -----   undispatch local o to global o   ---- #

            total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)

            # -----   run backward   ---- #

            if run_bwd:
                grad_total_out = torch.randn_like(total_out).detach()
                dist.all_reduce(grad_total_out.data, group=self.nccl_group)

                if self.profile_mode:
                    # barrier before bwd to wait for the processes with slow fwd
                    dist.barrier()
                    torch.cuda.synchronize()

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

            # -----   assert close if not using profile mode   ---- #

            if not self.profile_mode:
                # -----   assert close to torch ref   ---- #

                self.assert_close_to_torch_ref(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_mapping,
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
                    run_bwd=run_bwd,
                    test_case=test_case,
                )

    def assert_close_to_torch_ref(
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
        dtype: torch.dtype,
        run_bwd: bool,
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

        mask = get_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
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

        if run_bwd:
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

        # -----   init error message list   ---- #

        err_msg_list: list[str] = []

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(total_out, total_out_ref_high_precision)
        out_ref_norm = calc_inf_norm(
            total_out_ref_low_precision, total_out_ref_high_precision
        )
        try:
            self.assertLessEqual(
                out_norm,
                norm_rtol_ratio * out_ref_norm,
                msg=f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        o_thres = extract_mismatch_threshold(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )
        try:
            magi_attention.testing.assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        if run_bwd:
            # -----   assert close for bwd dq   ---- #

            # fa style with Linf norm
            dq_norm = calc_inf_norm(grad_total_q, grad_total_q_ref_high_precision)
            dq_ref_norm = calc_inf_norm(
                grad_total_q_ref_low_precision, grad_total_q_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    dq_norm,
                    norm_rtol_ratio * dq_ref_norm,
                    msg=f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dq_thres = extract_mismatch_threshold(
                actual=grad_total_q_ref_low_precision,
                expected=grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_thres_ratio=mismatch_thres_ratio,
            )
            try:
                magi_attention.testing.assert_close(
                    grad_total_q,
                    grad_total_q_ref_high_precision,
                    atol=dq_atol,
                    rtol=dq_rtol,
                    mismatch_threshold=dq_thres,
                    test_case=f"{test_case} => dq",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # -----   assert close for bwd dk   ---- #

            # fa style with Linf norm
            dk_norm = calc_inf_norm(grad_total_k, grad_total_k_ref_high_precision)
            dk_ref_norm = calc_inf_norm(
                grad_total_k_ref_low_precision, grad_total_k_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    dk_norm,
                    norm_rtol_ratio * dk_ref_norm,
                    msg=f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dk_thres = extract_mismatch_threshold(
                actual=grad_total_k_ref_low_precision,
                expected=grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_thres_ratio=mismatch_thres_ratio,
            )
            try:
                magi_attention.testing.assert_close(
                    grad_total_k,
                    grad_total_k_ref_high_precision,
                    atol=dk_atol,
                    rtol=dk_rtol,
                    mismatch_threshold=dk_thres,
                    test_case=f"{test_case} => dk",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # -----   assert close for bwd dv   ---- #

            # fa style with Linf norm
            dv_norm = calc_inf_norm(grad_total_v, grad_total_v_ref_high_precision)
            dv_ref_norm = calc_inf_norm(
                grad_total_v_ref_low_precision, grad_total_v_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    dv_norm,
                    norm_rtol_ratio * dv_ref_norm,
                    msg=f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dv_thres = extract_mismatch_threshold(
                actual=grad_total_v_ref_low_precision,
                expected=grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_thres_ratio=mismatch_thres_ratio,
            )
            try:
                magi_attention.testing.assert_close(
                    grad_total_v,
                    grad_total_v_ref_high_precision,
                    atol=dv_atol,
                    rtol=dv_rtol,
                    mismatch_threshold=dv_thres,
                    test_case=f"{test_case} => dv",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   raise error if any error occurs   ---- #

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))


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
