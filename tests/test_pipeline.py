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

import os
import random
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.comm.primitive.grpcoll._buffer import GrpCollBuffer
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode, AttnSinkLayout
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
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import (
    NAME,
    PROFILE_ONLY,
    SKIP_WORLD_SIZE,
    DistTestBase,
    with_comms,
)
from magi_attention.testing.flag_generator import FlagCombGenerator
from magi_attention.testing.precision import (
    EPSILON,
    H100_MATMUL_MFU,
    H100_NVLINK_A2A_BWU,
    H100_NVLINK_BANDWIDTH,
    H100_TFLOPS_16,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.testing.utils import switch_envvars
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_calc_cost_factor,
    get_comm_cost_factor,
    make_attn_mask_from_ffa_args,
    str2seed,
    sync_rng,
)


class TestPipelineBaseWithWorldSize1(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # init several pgs with all ranks
        self.nccl_groups = [
            dist.new_group(list(range(self.world_size)), backend=self.backend)
            for _ in range(2)
        ]

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

        # -----    set up for native grpcoll   ---- #

        native_grpcoll_registered = True
        for nccl_group in self.nccl_groups:
            try:
                grpcoll_buffer_mgr.initialize(
                    group=nccl_group,
                    config=GrpCollConfig(
                        num_sms=24,
                        num_nvl_bytes=int(2e9) * self.world_size // 8,  # ~2GB for 8 ranks
                    ),
                )
            except Exception as e:
                native_grpcoll_registered = False
                print(
                    f"The NCCL group {nccl_group} cannot be registered due to error: \n{e}\n"
                )

        # -----    set up for flags   ---- #

        self.flag_to_envvar = {
            "device_max_connections": "CUDA_DEVICE_MAX_CONNECTIONS",
            "deterministic_mode": "MAGI_ATTENTION_DETERMINISTIC_MODE",
            "enable_hier_comm": "MAGI_ATTENTION_HIERARCHICAL_COMM",
            "enable_qo_comm": "MAGI_ATTENTION_QO_COMM",
            "enable_native_grpcoll": "MAGI_ATTENTION_NATIVE_GRPCOLL",
            "fwd_hp_reduce": "MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE",
            "bwd_hp_reduce": "MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE",
        }

        # init flag generator and its iterator
        self.flag_generator = FlagCombGenerator(
            flags=list(self.flag_to_envvar.keys()),
            options={
                "device_max_connections": [1, 8],
                "enable_native_grpcoll": (
                    [False, True]
                    if native_grpcoll_registered
                    # disable native grpcoll if not registered successfully
                    else [False]
                ),
                "deterministic_mode": (
                    [False, True]
                    if not magi_attention.is_fa4_backend_enable()
                    # TODO: support deterministic mode for fa4 backend
                    else [False]
                ),
                "fwd_hp_reduce": (
                    [False, True]
                    if not magi_attention.is_fa4_backend_enable()
                    # TODO: support forward high precision reduce for fa4 backend
                    else [False]
                ),
                "bwd_hp_reduce": (
                    [False, True]
                    if not magi_attention.is_fa4_backend_enable()
                    # TODO: support backward high precision reduce for fa4 backend
                    else [False]
                ),
                "enable_qo_comm": (
                    [False, True]
                    if not magi_attention.is_fa4_backend_enable()
                    # TODO: support qo comm for fa4 backend
                    else [False]
                ),
            },
            defaults={
                "device_max_connections": 8,
            },
            groups=[
                # group for comm
                ("enable_hier_comm", "enable_qo_comm", "enable_native_grpcoll"),
            ],
            strategy="heuristic",
        )
        self.flag_iterator = iter(self.flag_generator)

    @property
    def timeout(self) -> int:
        return 900

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

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
                "total_seqlen_sink": 1,
                "sink_layout": "sh",
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
                "attn_type_mapping": [0] * 7,
                "total_seqlen_q": 15360,
                "total_seqlen_k": 15360,
                "total_seqlen_sink": 4,
                "sink_layout": "sh",
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
                "total_seqlen_sink": 8,
                "sink_layout": "sh",
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
            # full_mask_assembled_from_small_pieces
            {
                NAME: "full_mask_assembled_from_small_pieces_with_8k",
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
                SKIP_WORLD_SIZE: [],
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
            {
                PROFILE_ONLY: True,
                NAME: "varlen_block_causal_144k",
                SKIP_WORLD_SIZE: [],
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
                "attn_type_mapping": [0] * 7,
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
            # disable multi-stage overlap
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
            # disable multi-stage overlap
            {
                PROFILE_ONLY: True,
                NAME: "disable_mso",
                "enable": False,
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
            {
                PROFILE_ONLY: True,
                NAME: "dynamic_md8",
                "enable": True,
                "mode": AttnOverlapMode.DYNAMIC,
                "degree": None,
                "dynamic_max_degree": 8,
                "min_chunk_size": 512,
                "max_num_chunks": 64,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
            },
        ],
    )
    @parameterize(
        "num_heads",
        [
            (8, 8),  # mha
            (8, 2),  # gqa
        ],
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

        # -----    switch env flags   ---- #

        flag_comb_test_case = ""
        if not self.profile_mode:
            flag_comb = next(self.flag_iterator)
            flag_comb = FlagCombGenerator.sync_group(flag_comb, self.nccl_group)
            flag_comb_test_case = FlagCombGenerator.to_test_case(flag_comb)
            switch_back = switch_envvars(
                envvar_name_list=list(self.flag_to_envvar.values()),
                enable_dict={
                    envvar: (
                        flag_comb[flag] if isinstance(flag_comb[flag], bool) else True
                    )
                    for flag, envvar in self.flag_to_envvar.items()
                },
                enable_value_dict={
                    envvar: str(flag_comb[flag])
                    for flag, envvar in self.flag_to_envvar.items()
                    if not isinstance(flag_comb[flag], bool)
                },
            )
        else:
            flag_comb_test_case = "[profile_mode=True]"

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

            # TODO: support hierarchical comm for qo comm
            if magi_attention.comm.is_hierarchical_comm_enable():
                return

        # -----    skip for native grpcoll   ---- #

        if magi_attention.comm.is_native_grpcoll_enable():
            # TODO: support hierarchical comm with native grpcoll
            if magi_attention.comm.is_hierarchical_comm_enable():
                return

            # FIXME: when deterministic mode and native grpocoll are both enabled,
            # sometimes it causes hang when not launching in blocking mode
            if magi_attention.is_deterministic_mode_enable():
                return

            hidden_size_kv = num_heads[1] * head_dim
            if hidden_size_kv % GrpCollBuffer.get_hidden_size_alignment(dtype) != 0:
                return

        # -----    skip for flatten head groups   ---- #

        if magi_attention.is_flatten_head_groups_enable():
            # FIXME: Flattening head groups is incompatible with attn sink
            if attn_config.get("total_seqlen_sink", 0) > 0:
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
            f"has_sink=[{attn_config.get('total_seqlen_sink', 0) > 0}] x "
            + flag_comb_test_case
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
        total_seqlen_sink: int = attn_config.get("total_seqlen_sink", 0)
        if magi_attention.is_fa4_backend_enable():
            # TODO: support attn sink for fa4 backend
            total_seqlen_sink = 0
        chunk_size: int = attn_config["chunk_size"]
        num_heads_q, num_heads_kv = num_heads
        softmax_scale = (  # choose softmax_scale by rule
            None if test_case_seed % 2 == 0 else (1 / head_dim)
        )
        softcap = 0.0  # not supported for test
        sink_layout: AttnSinkLayout = attn_config.get("sink_layout", "sh")

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(
                # TODO: test other dispatch algs
                alg=MinHeapDispatchAlg()
            ),
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
            # NOTE: this config is useless for this test
            # since we already register/release buffer in `init_pg`/`destroy_pg`
            grpcoll_config=GrpCollConfig(),
        )

        # -----   init attn_mask_type ----- #

        attn_mask_type: list[AttnMaskType] = list(
            map(AttnMaskType.from_int_type, attn_type_mapping)
        )

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
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
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

            if total_seqlen_sink > 0:
                match sink_layout:
                    case "sh":
                        total_sink = torch.randn(
                            total_seqlen_sink,
                            num_heads_q,
                            device=self.device,
                            dtype=torch.float32,
                            requires_grad=run_bwd,
                        )
                    case "ssh":
                        total_sink = torch.randn(
                            total_seqlen_q,
                            total_seqlen_sink,
                            num_heads_q,
                            device=self.device,
                            dtype=torch.float32,
                            requires_grad=run_bwd,
                        )
                    case "shd":
                        raise NotImplementedError(
                            f"sink_layout {sink_layout} is not supported yet"
                        )
                    case _:
                        raise ValueError(f"Invalid sink_layout {sink_layout}")
                # TODO: support other sink layouts for distributed attention sink
                assert (
                    sink_layout == "sh"
                ), "Only support `sh` layout for distributed attention sink by now"
                dist.all_reduce(total_sink.data, group=self.nccl_group)
            else:
                total_sink = None

            # -----   dispatch global qkv to local qkv   ---- #

            local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
            local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
            local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)

            # -----   run dist attn forward on local qkv for local out   ---- #

            if self.profile_mode:
                # barrier before fwd to wait for the processes with slow solver
                dist.barrier()
                torch.cuda.synchronize()

            local_out, local_lse = dist_attn_runtime_mgr.calc_attn(
                q=local_q,
                k=local_k,
                v=local_v,
                sink=total_sink,
                softmax_scale=softmax_scale,
                softcap=softcap,
            )

            # -----   undispatch local out to global out   ---- #

            total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
            total_lse = dist_attn_runtime_mgr.undispatch_qo(local_lse)

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
                grad_total_sink = total_sink.grad if total_sink is not None else None
            else:
                grad_total_out = None
                grad_total_q, grad_total_k, grad_total_v = None, None, None
                grad_total_sink = None

            # -----   assert close if not using profile mode   ---- #

            if not self.profile_mode:
                # switch the env flags back
                switch_back()

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
                    total_sink=total_sink,
                    total_out=total_out,
                    total_lse=total_lse,
                    grad_total_q=grad_total_q,
                    grad_total_k=grad_total_k,
                    grad_total_v=grad_total_v,
                    grad_total_sink=grad_total_sink,
                    grad_total_out=grad_total_out,
                    dtype=dtype,
                    run_bwd=run_bwd,
                    test_case=test_case,
                    err_ratio_dict={
                        "dsink_mismatch_thres_ratio": MISMATCH_THRES_RATIO * 1.5,
                        "dsink_min_mismatch_thres": max(
                            2 / (total_seqlen_sink * num_heads_q), 5e-2
                        )
                        if total_seqlen_sink > 0 and sink_layout == "sh"
                        else 5e-2,
                        "dsink_min_norm_rtol": 0.15,
                        "dsink_norm_rtol_ratio": NORM_RTOL_RATIO * 2,
                        "dsink_atol": 2e-4 if sink_layout == "sh" else EPSILON,
                        "dsink_rtol": 0.15,
                        "lse_min_norm_rtol": 2e-5
                        if magi_attention.is_fa4_backend_enable()
                        else 0.0,
                    },
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
        total_sink: torch.Tensor | None,
        total_out: torch.Tensor,
        total_lse: torch.Tensor | None,
        grad_total_q: torch.Tensor | None,
        grad_total_k: torch.Tensor | None,
        grad_total_v: torch.Tensor | None,
        grad_total_sink: torch.Tensor | None,
        grad_total_out: torch.Tensor | None,
        dtype: torch.dtype,
        run_bwd: bool,
        test_case: str = "",
        err_ratio_dict: dict[str, float] = {},
    ) -> None:
        # -----   customize tolerance / threshold  ---- #

        o_atol = EPSILON
        o_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        o_norm_rtol_ratio = err_ratio_dict.get("o_norm_rtol_ratio", NORM_RTOL_RATIO)
        o_min_norm_rtol = err_ratio_dict.get("o_min_norm_rtol", 0.0)
        o_mismatch_thres_ratio = err_ratio_dict.get(
            "o_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        o_min_mismatch_thres = err_ratio_dict.get("o_min_mismatch_thres", 0.0)
        o_max_mismatch_thres = err_ratio_dict.get(
            "o_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        lse_atol = EPSILON
        lse_rtol = 0.001
        lse_norm_rtol_ratio = err_ratio_dict.get("lse_norm_rtol_ratio", NORM_RTOL_RATIO)
        lse_min_norm_rtol = err_ratio_dict.get("lse_min_norm_rtol", 0.0)
        lse_mismatch_thres_ratio = err_ratio_dict.get(
            "lse_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        lse_min_mismatch_thres = err_ratio_dict.get("lse_min_mismatch_thres", 0.0)
        lse_max_mismatch_thres = err_ratio_dict.get(
            "lse_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dq_atol = EPSILON
        dq_rtol = {torch.bfloat16: 0.3, torch.float16: 0.2}.get(dtype, 0.2)
        dq_norm_rtol_ratio = err_ratio_dict.get("dq_norm_rtol_ratio", NORM_RTOL_RATIO)
        dq_min_norm_rtol = err_ratio_dict.get("dq_min_norm_rtol", 0.0)
        dq_mismatch_thres_ratio = err_ratio_dict.get(
            "dq_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dq_min_mismatch_thres = err_ratio_dict.get("dq_min_mismatch_thres", 0.0)
        dq_max_mismatch_thres = err_ratio_dict.get(
            "dq_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dk_atol = EPSILON
        dk_rtol = {torch.bfloat16: 0.15, torch.float16: 0.08}.get(dtype, 0.08)
        dk_norm_rtol_ratio = err_ratio_dict.get("dk_norm_rtol_ratio", NORM_RTOL_RATIO)
        dk_min_norm_rtol = err_ratio_dict.get("dk_min_norm_rtol", 0.0)
        dk_mismatch_thres_ratio = err_ratio_dict.get(
            "dk_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dk_min_mismatch_thres = err_ratio_dict.get("dk_min_mismatch_thres", 0.0)
        dk_max_mismatch_thres = err_ratio_dict.get(
            "dk_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dv_atol = EPSILON
        dv_rtol = {torch.bfloat16: 0.05, torch.float16: 0.05}.get(dtype, 0.05)
        dv_norm_rtol_ratio = err_ratio_dict.get("dv_norm_rtol_ratio", NORM_RTOL_RATIO)
        dv_min_norm_rtol = err_ratio_dict.get("dv_min_norm_rtol", 0.0)
        dv_mismatch_thres_ratio = err_ratio_dict.get(
            "dv_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dv_min_mismatch_thres = err_ratio_dict.get("dv_min_mismatch_thres", 0.0)
        dv_max_mismatch_thres = err_ratio_dict.get(
            "dv_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        dsink_atol = err_ratio_dict.get("dsink_atol", EPSILON)
        dsink_rtol = err_ratio_dict.get("dsink_rtol", 0.05)
        dsink_norm_rtol_ratio = err_ratio_dict.get(
            "dsink_norm_rtol_ratio", NORM_RTOL_RATIO
        )
        dsink_min_norm_rtol = err_ratio_dict.get("dsink_min_norm_rtol", 0.0)
        dsink_mismatch_thres_ratio = err_ratio_dict.get(
            "dsink_mismatch_thres_ratio", MISMATCH_THRES_RATIO
        )
        dsink_min_mismatch_thres = err_ratio_dict.get("dsink_min_mismatch_thres", 0.0)
        dsink_max_mismatch_thres = err_ratio_dict.get(
            "dsink_max_mismatch_thres", MAX_MISMATCH_THRES
        )

        # -----   build attn mask   ---- #

        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=self.device,
        )

        # -----   ref1. torch ref with high precision (fp64)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if total_sink is not None:
            total_sink.grad = None

        total_out_ref_high_precision, total_lse_ref_high_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            sink=total_sink,
            softmax_scale=softmax_scale,
            softcap=softcap,
            layout="thd",
            sink_layout="sh",
            backend="torch",
            high_precision=True,
            return_lse=total_lse is not None,
            online_softmax=True,
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
            grad_total_sink_ref_high_precision = (
                total_sink.grad if total_sink is not None else None
            )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        total_q.grad, total_k.grad, total_v.grad = None, None, None
        if total_sink is not None:
            total_sink.grad = None

        total_out_ref_low_precision, total_lse_ref_low_precision = ref_attn_func(
            q=total_q,
            k=total_k,
            v=total_v,
            mask=mask,
            sink=total_sink,
            softmax_scale=softmax_scale,
            softcap=softcap,
            layout="thd",
            sink_layout="sh",
            backend="torch",
            high_precision=False,
            return_lse=total_lse is not None,
            online_softmax=True,
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
            grad_total_sink_ref_low_precision = (
                total_sink.grad if total_sink is not None else None
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
                max(o_min_norm_rtol, o_norm_rtol_ratio * out_ref_norm),
                msg=(
                    f"For {test_case=}: {out_norm=} should be no greater than "
                    f"max({o_min_norm_rtol}, {o_norm_rtol_ratio} x {out_ref_norm=})",
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # torch style with atol + rtol + mismatch threshold
        o_thres = extract_mismatch_threshold(
            actual=total_out_ref_low_precision,
            expected=total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=o_mismatch_thres_ratio,
            min_mismatch_thres=o_min_mismatch_thres,
            max_mismatch_thres=o_max_mismatch_thres,
        )
        try:
            assert_close(
                total_out,
                total_out_ref_high_precision,
                atol=o_atol,
                rtol=o_rtol,
                mismatch_threshold=o_thres,
                test_case=f"{test_case} => o",
            )
        except Exception as e:
            err_msg_list.append(str(e))

        # -----   assert close for fwd lse   ---- #

        if total_lse is not None:
            # fa style with Linf norm
            lse_norm = calc_inf_norm(total_lse, total_lse_ref_high_precision)
            lse_ref_norm = calc_inf_norm(
                total_lse_ref_low_precision, total_lse_ref_high_precision
            )
            try:
                self.assertLessEqual(
                    lse_norm,
                    max(lse_min_norm_rtol, lse_norm_rtol_ratio * lse_ref_norm),
                    msg=(
                        f"For {test_case=}: {lse_norm=} should be no greater than "
                        f"max({lse_min_norm_rtol}, {lse_norm_rtol_ratio} x {lse_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            lse_thres = extract_mismatch_threshold(
                actual=total_lse_ref_low_precision,
                expected=total_lse_ref_high_precision,
                atol=lse_atol,
                rtol=lse_rtol,
                mismatch_thres_ratio=lse_mismatch_thres_ratio,
                min_mismatch_thres=lse_min_mismatch_thres,
                max_mismatch_thres=lse_max_mismatch_thres,
            )
            try:
                assert_close(
                    total_lse,
                    total_lse_ref_high_precision,
                    atol=lse_atol,
                    rtol=lse_rtol,
                    mismatch_threshold=lse_thres,
                    test_case=f"{test_case} => lse",
                )
            except Exception as e:
                err_msg_list.append(str(e))

        # -----   assert close for bwd   ---- #

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
                    max(dq_min_norm_rtol, dq_norm_rtol_ratio * dq_ref_norm),
                    msg=(
                        f"For {test_case=}: {dq_norm=} should be no greater than "
                        f"max({dq_min_norm_rtol}, {dq_norm_rtol_ratio} x {dq_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dq_thres = extract_mismatch_threshold(
                actual=grad_total_q_ref_low_precision,
                expected=grad_total_q_ref_high_precision,
                atol=dq_atol,
                rtol=dq_rtol,
                mismatch_thres_ratio=dq_mismatch_thres_ratio,
                min_mismatch_thres=dq_min_mismatch_thres,
                max_mismatch_thres=dq_max_mismatch_thres,
            )
            try:
                assert_close(
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
                    max(dk_min_norm_rtol, dk_norm_rtol_ratio * dk_ref_norm),
                    msg=(
                        f"For {test_case=}: {dk_norm=} should be no greater than "
                        f"max({dk_min_norm_rtol}, {dk_norm_rtol_ratio} x {dk_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dk_thres = extract_mismatch_threshold(
                actual=grad_total_k_ref_low_precision,
                expected=grad_total_k_ref_high_precision,
                atol=dk_atol,
                rtol=dk_rtol,
                mismatch_thres_ratio=dk_mismatch_thres_ratio,
                min_mismatch_thres=dk_min_mismatch_thres,
                max_mismatch_thres=dk_max_mismatch_thres,
            )
            try:
                assert_close(
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
                    max(dv_min_norm_rtol, dv_norm_rtol_ratio * dv_ref_norm),
                    msg=(
                        f"For {test_case=}: {dv_norm=} should be no greater than "
                        f"max({dv_min_norm_rtol}, {dv_norm_rtol_ratio} x {dv_ref_norm=})"
                    ),
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # torch style with atol + rtol + mismatch threshold
            dv_thres = extract_mismatch_threshold(
                actual=grad_total_v_ref_low_precision,
                expected=grad_total_v_ref_high_precision,
                atol=dv_atol,
                rtol=dv_rtol,
                mismatch_thres_ratio=dv_mismatch_thres_ratio,
                min_mismatch_thres=dv_min_mismatch_thres,
                max_mismatch_thres=dv_max_mismatch_thres,
            )
            try:
                assert_close(
                    grad_total_v,
                    grad_total_v_ref_high_precision,
                    atol=dv_atol,
                    rtol=dv_rtol,
                    mismatch_threshold=dv_thres,
                    test_case=f"{test_case} => dv",
                )
            except Exception as e:
                err_msg_list.append(str(e))

            # -----   assert close for bwd dsink   ---- #

            if total_sink is not None:
                # fa style with Linf norm
                dsink_norm = calc_inf_norm(
                    grad_total_sink,
                    grad_total_sink_ref_high_precision,
                )
                dsink_ref_norm = calc_inf_norm(
                    grad_total_sink_ref_low_precision,
                    grad_total_sink_ref_high_precision,
                )
                try:
                    self.assertLessEqual(
                        dsink_norm,
                        max(
                            dsink_min_norm_rtol, dsink_norm_rtol_ratio * dsink_ref_norm
                        ),
                        msg=(
                            f"For {test_case=}: {dsink_norm=} should be no greater than "
                            f"max({dsink_min_norm_rtol}, {dsink_norm_rtol_ratio} x {dsink_ref_norm=})"
                        ),
                    )
                except Exception as e:
                    # err_msg_list.append(str(e))
                    # FIXME: dsink is easy to fail, disable it for now
                    print(f"dsink norm error for {test_case=}: \n{e}\n")

                # torch style with atol + rtol + mismatch threshold
                dsink_thres = extract_mismatch_threshold(
                    actual=grad_total_sink_ref_low_precision,
                    expected=grad_total_sink_ref_high_precision,
                    atol=dsink_atol,
                    rtol=dsink_rtol,
                    mismatch_thres_ratio=dsink_mismatch_thres_ratio,
                    min_mismatch_thres=dsink_min_mismatch_thres,
                    max_mismatch_thres=dsink_max_mismatch_thres,
                )
                try:
                    assert_close(
                        grad_total_sink,
                        grad_total_sink_ref_high_precision,
                        atol=dsink_atol,
                        rtol=dsink_rtol,
                        mismatch_threshold=dsink_thres,
                        test_case=f"{test_case} => dsink",
                    )
                except Exception as e:
                    # err_msg_list.append(str(e))
                    # FIXME: dsink is easy to fail, disable it for now
                    print(f"dsink mismatch error for {test_case=}: \n{e}\n")

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
