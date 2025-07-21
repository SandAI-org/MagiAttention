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
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.api.functools import (
    apply_padding,
    compute_pad_size,
    full_attention_to_varlen_attention,
)
from magi_attention.api.magi_attn_interface import (
    DistAttnRuntimeDict,
    magi_attn_flex_dispatch,
    magi_attn_varlen_dispatch,
)
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeMgr,
    init_dist_attn_runtime_mgr,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.utils._utils import is_list_value_all

NAME = "name"
SKIP_WORLD_SIZE = "skip_world_size"
INTERFACE = "interface"

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
HEAD_DIM = 64
DTYPE = torch.float64
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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, embed_dim, bias=False, dtype=DTYPE)
        self.k_proj = nn.Linear(hidden_dim, embed_dim, bias=False, dtype=DTYPE)
        self.v_proj = nn.Linear(hidden_dim, embed_dim, bias=False, dtype=DTYPE)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, k, v = [
            rearrange(
                e,
                "s (nh hd) -> s nh hd",
                hd=self.head_dim,
            )
            for e in (q, k, v)
        ]

        return q, k, v


# TODO: rewrite the specific ut for magi_attn_interface
# instead of a fork of `test_pipeline_sdpa`
class TestInterfaceSDPABaseWithWorldSize1(DistTestBase):
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
                NAME: "full_attn_1k_bs2",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                INTERFACE: "magi_attn",
                "batch_size": 2,
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
                "attn_type_mapping": 1,
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
                "chunk_size": 1024,
            },
            # full attn with seqlen 2k and batchsize 3
            {
                NAME: "full_attn_2k_bs3",
                SKIP_WORLD_SIZE: [3, 5, 6, 7],
                INTERFACE: "magi_attn",
                "batch_size": 3,
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
                "attn_type_mapping": [0, 0, 0],
                "total_seqlen_q": 6144,
                "total_seqlen_k": 6144,
                "chunk_size": 1536,
            },
            # varlen full attn with total seqlen 1050
            {
                NAME: "flex_full_attn_1050",
                SKIP_WORLD_SIZE: [4, 8],
                INTERFACE: "magi_attn_flex",
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
                "attn_type_mapping": 0,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
                "chunk_size": 257,
                "use_str_masktype": False,
            },
            {
                NAME: "varlen_full_attn_1050",
                SKIP_WORLD_SIZE: [4, 8],
                INTERFACE: "magi_attn_varlen",
                "cu_seqlens_q": torch.tensor(
                    [0, 128, 256, 384, 512, 640, 768, 1050], dtype=torch.int32
                ),
                "cu_seqlens_k": torch.tensor(
                    [0, 128, 256, 384, 512, 640, 768, 1050], dtype=torch.int32
                ),
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
                "chunk_size": 568,
            },
            # varlen block causal with total seqlen 960
            {
                NAME: "varlen_block_causal_960",
                SKIP_WORLD_SIZE: [7, 8],
                INTERFACE: "magi_attn_flex",
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
                "attn_type_mapping": [0, 1, 2, 3, 1, 2, 3],
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 568,
                "use_str_masktype": True,
            },
            # cp_mesh and cp_group are both set, raise ValueError
            {
                NAME: "cp_mesh and cp_group testcase",
                SKIP_WORLD_SIZE: [1, 2, 3, 5, 7],
                INTERFACE: "set_mesh_and_group",
                "q_ranges": AttnRanges.from_ranges([[0, 960]]),
                "k_ranges": AttnRanges.from_ranges([[0, 960]]),
                "attn_type_mapping": [0],
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 568,
            },
            # test for invalid masktype
            {
                NAME: "cp_mesh and cp_group testcase",
                SKIP_WORLD_SIZE: [3, 5, 7],
                INTERFACE: "test_for_invalid_mask",
                "q_ranges": AttnRanges.from_ranges([[0, 960]]),
                "k_ranges": AttnRanges.from_ranges([[0, 960]]),
                "attn_type_mapping": [0],
                "attn_mask_type": ["casual"],
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
                "chunk_size": 324,
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
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
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
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
        ],
    )
    @parameterize(
        "num_heads",
        [NUM_HEADS],
    )
    @parameterize(
        "head_dim",
        [64, 80],
    )
    @parameterize(
        "dtype",
        [DTYPE],
    )
    @parameterize(
        "high_bandwith_domain_size",
        [1],  # TODO: this feature'll probably be deprecated soon
    )
    def test_interface_sdpa(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        high_bandwith_domain_size: int,
    ):
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

        assert magi_attention.is_sanity_check_enable()

        # -----    skip for hier comm   ---- #

        if magi_attention.comm.is_hierarchical_comm_enable():
            if high_bandwith_domain_size > 1:
                return

        # -----    construct test case name   ---- #

        assert (
            NAME in attn_config and NAME in overlap_config
        ), f"{attn_config=} | \n\n{overlap_config=}"

        test_case = (
            f"world_size=[{self.world_size}] x high_bandwith_domain_size=[{high_bandwith_domain_size}] x "
            f"attn_config=[{attn_config[NAME]}] x overlap_config=[{overlap_config[NAME]}] x "
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})]"
        )
        # -----    contruct config from test cases   ---- #

        q_ranges: AttnRanges = attn_config["q_ranges"]
        k_ranges: AttnRanges = attn_config["k_ranges"]
        interface: str = attn_config["interface"]
        attn_type_mapping: int | list[int] = attn_config["attn_type_mapping"]
        total_seqlen_q: int = attn_config["total_seqlen_q"]
        total_seqlen_k: int = attn_config["total_seqlen_k"]
        chunk_size: int = attn_config["chunk_size"]

        device = torch.cuda.current_device()

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                **{k: v for k, v in overlap_config.items() if k not in (NAME,)}
            ),
            high_bandwith_domain_size=high_bandwith_domain_size,
            deterministic=False,
        )

        # ----- init input data and module ----- #
        x = torch.randn(
            total_seqlen_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )

        # --------- calculate pad size --------- #
        cp_size = dist.get_world_size(self.nccl_group)
        pad_size = compute_pad_size(total_seqlen_q, cp_size, head_dim, chunk_size)

        # ------ calculate attn_mask_type ------ #
        if isinstance(attn_type_mapping, list):
            attn_mask_type = [
                [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.INVCAUSAL,
                    AttnMaskType.BICAUSAL,
                ][attn_type]
                for attn_type in attn_type_mapping
            ]
        else:
            attn_mask_type = [
                [  # type: ignore[assignment]
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.INVCAUSAL,
                    AttnMaskType.BICAUSAL,
                ][attn_type_mapping]
            ] * len(q_ranges)

        if interface == "magi_attn":
            assert is_list_value_all(
                attn_mask_type, AttnMaskType.FULL
            ) or is_list_value_all(
                attn_mask_type, AttnMaskType.CAUSAL
            ), "we need to check varlen interface, which supports full or causal now"
            is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

            batch_size = attn_config["batch_size"]
            cu_seqlens_q, cu_seqlens_k = full_attention_to_varlen_attention(
                batch_size, attn_config["total_seqlen_q"] // batch_size
            )

            _, dist_attn_runtime_key = magi_attn_varlen_dispatch(
                x,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim=head_dim,
                pad_size=pad_size,
                chunk_size=chunk_size,
                cp_group=None
                if magi_attention.comm.is_hierarchical_comm_enable()
                else self.nccl_group,
                cp_mesh=self.device_mesh,
                causal=is_causal,
                dist_attn_config=dist_attn_config,
            )

        if interface == "magi_attn_varlen":
            assert is_list_value_all(
                attn_mask_type, AttnMaskType.FULL
            ) or is_list_value_all(
                attn_mask_type, AttnMaskType.CAUSAL
            ), "we need to check varlen interface, which supports full or causal now"
            is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

            cu_seqlens_q = attn_config["cu_seqlens_q"]
            cu_seqlens_k = attn_config["cu_seqlens_k"]
            _, dist_attn_runtime_key = magi_attn_varlen_dispatch(
                x,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim=head_dim,
                pad_size=pad_size,
                chunk_size=chunk_size,
                cp_group=None
                if magi_attention.comm.is_hierarchical_comm_enable()
                else self.nccl_group,
                cp_mesh=self.device_mesh,
                causal=is_causal,
                dist_attn_config=dist_attn_config,
            )

        if interface == "magi_attn_flex":
            use_str_masktype: bool = attn_config["use_str_masktype"]
            _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                x,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=[masktype.value for masktype in attn_mask_type]
                if use_str_masktype
                else attn_mask_type,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
                head_dim=head_dim,
                pad_size=pad_size,
                chunk_size=chunk_size,
                cp_group=None
                if magi_attention.comm.is_hierarchical_comm_enable()
                else self.nccl_group,
                cp_mesh=self.device_mesh,
                dist_attn_config=dist_attn_config,
            )

        if interface == "set_mesh_and_group":
            if magi_attention.comm.is_hierarchical_comm_enable():
                with pytest.raises(ValueError):
                    _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                        x,
                        q_ranges=q_ranges,
                        k_ranges=k_ranges,
                        attn_mask_type=attn_mask_type,
                        total_seqlen_q=total_seqlen_q,
                        total_seqlen_k=total_seqlen_k,
                        head_dim=head_dim,
                        pad_size=pad_size,
                        chunk_size=chunk_size,
                        cp_group=self.nccl_group,
                        cp_mesh=self.device_mesh,
                        dist_attn_config=dist_attn_config,
                    )
            return

        if interface == "test_for_invalid_mask":
            invalid_mask_type = attn_config["attn_mask_type"]
            with pytest.raises(ValueError):
                _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                    x,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_mask_type=invalid_mask_type,
                    total_seqlen_q=total_seqlen_q,
                    total_seqlen_k=total_seqlen_k,
                    head_dim=head_dim,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group=None
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    cp_mesh=self.device_mesh,
                    dist_attn_config=dist_attn_config,
                )
            return

        # -----    compute dist attn runtime mgr   ---- #
        dist_attn_runtime_mgr: DistAttnRuntimeMgr = DistAttnRuntimeDict[
            dist_attn_runtime_key
        ]

        # -------   calc ref_attn_runtime_mgr -------- #
        if pad_size > 0:
            q_ranges, k_ranges, attn_mask_type = apply_padding(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen=total_seqlen_q,
                pad_size=pad_size,
            )

        ref_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q + pad_size,
            total_seqlen_k=total_seqlen_k + pad_size,
            chunk_size=chunk_size,
            cp_group=self.nccl_group,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=dist_attn_config,
            cp_mesh=self.device_mesh,
        )

        assert (
            dist_attn_runtime_mgr == ref_attn_runtime_mgr
        ), f"the answer is not correct when {test_case}"


class TestInterfaceSDPAWithWorldSize2(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize3(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize4(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize5(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize6(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize7(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


class TestInterfaceSDPAWithWorldSize8(TestInterfaceSDPABaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_interface_sdpa(self, *args, **kwargs):
        super().test_interface_sdpa(*args, **kwargs)


if __name__ == "__main__":
    run_tests()
