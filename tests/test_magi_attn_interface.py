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
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
import magi_attention.testing
from magi_attention.api.functools import (
    compute_pad_size,
    full_attention_to_varlen_attention,
    squash_batch_dim,
)
from magi_attention.api.magi_attn_interface import (
    DistAttnRuntimeDict,
    calc_attn,
    magi_attn_flex_dispatch,
    magi_attn_varlen_dispatch,
    undispatch,
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
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import EPSILON, torch_attn_ref
from magi_attention.utils import get_attn_mask_from_ranges

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


# TODO: merge this test script with `test_pipeline_sdpa.py`
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
                "is_causal_mapping": True,
                "total_seqlen_q": 2048,
                "total_seqlen_k": 2048,
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
                "is_causal_mapping": [False, False, False],
                "total_seqlen_q": 6144,
                "total_seqlen_k": 6144,
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
                "is_causal_mapping": False,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
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
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 1050,
                "total_seqlen_k": 1050,
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
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 960,
                "total_seqlen_k": 960,
            },
            # varlen block causal with total seqlen 840
            {
                NAME: "varlen_block_causal_840",
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
                "is_causal_mapping": [False] * 7,
                "total_seqlen_q": 840,
                "total_seqlen_k": 840,
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
                "calc_cost_factor": CALC_COST_FACTOR,
                "comm_cost_factor": INTRA_NODE_COMM_COST_FACTOR,
            },
            # static, overlap degree = 2, min chunk size = 27
            {
                NAME: "static_od2_cz27",
                "enable": True,
                "mode": AttnOverlapMode.STATIC,
                "degree": 2,
                "min_chunk_size": 14,
                "max_num_chunks": 44,
                "alg": UniformOverlapAlg(
                    random_costs=True,
                    random_seed=42,
                ),
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
        [1, 2, 4, 8],
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

        # NOTE: test pipeline using sdpa does not need profile mode
        # thus we always enable sanity check mode
        assert magi_attention.is_sanity_check_enable()

        # -----    skip for hier comm   ---- #

        if magi_attention.is_hierarchical_comm_enable():
            if self.world_size not in (4, 6, 8):
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
            f"dtype=[{dtype}] x (nh,hd)=[({num_heads},{head_dim})]"
        )
        # -----    contruct config from test cases   ---- #

        attn_config_ = deepcopy(attn_config)

        q_ranges: AttnRanges = attn_config_["q_ranges"]
        k_ranges: AttnRanges = attn_config_["k_ranges"]
        interface: str = attn_config_[INTERFACE]
        is_causal_mapping: bool | list[bool] = attn_config_["is_causal_mapping"]
        total_seqlen_q: int = attn_config_["total_seqlen_q"]
        total_seqlen_k: int = attn_config_["total_seqlen_k"]

        device = torch.cuda.current_device()

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
            overlap_config=OverlapConfig(
                **{k: v for k, v in overlap_config.items() if k not in (NAME,)}
            ),
            high_bandwith_domain_size=high_bandwith_domain_size,
            deterministic=False,
        )

        # -----    run pipeline test   ---- #

        # ----- init input data and module---- #
        x = torch.randn(
            total_seqlen_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        MHA = MultiHeadAttention(head_dim, head_dim, num_heads).to(device)

        # calculate pad size;
        cp_size = dist.get_world_size(self.nccl_group)
        pad_size, _ = compute_pad_size(total_seqlen_q, cp_size, head_dim)

        if interface == "magi_attn":
            batch_size = attn_config_["batch_size"]
            x_with_batch = torch.randn(
                batch_size,
                attn_config_["total_seqlen_q"] // batch_size,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )

            x.data.copy_(squash_batch_dim(x_with_batch))

            cu_seqlens_q, cu_seqlens_k = full_attention_to_varlen_attention(
                batch_size, attn_config_["total_seqlen_q"] // batch_size
            )

            assert self.device_mesh is not None
            x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
                x,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim=head_dim,
                pad_size=pad_size,
                cp_group=None
                if magi_attention.is_hierarchical_comm_enable()
                else self.nccl_group,
                cp_mesh=self.device_mesh,
                causal=is_causal_mapping[0]
                if isinstance(is_causal_mapping, list)
                else is_causal_mapping,
                dist_attn_config=dist_attn_config,
            )

        if interface == "magi_attn_varlen":
            cu_seqlens_q = attn_config_["cu_seqlens_q"]
            cu_seqlens_k = attn_config_["cu_seqlens_k"]
            x_padded, dist_attn_runtime_key = magi_attn_varlen_dispatch(
                x,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim=head_dim,
                pad_size=pad_size,
                cp_group=None
                if magi_attention.is_hierarchical_comm_enable()
                else self.nccl_group,
                cp_mesh=self.device_mesh,
                causal=is_causal_mapping[0]
                if isinstance(is_causal_mapping, list)
                else is_causal_mapping,
                dist_attn_config=dist_attn_config,
            )

        if interface == "magi_attn_flex":
            attn_mask_type: AttnMaskType | list[AttnMaskType]
            if isinstance(is_causal_mapping, list):
                attn_mask_type = [
                    AttnMaskType.CAUSAL if is_causal else AttnMaskType.FULL
                    for is_causal in is_causal_mapping
                ]
            else:
                attn_mask_type = (
                    AttnMaskType.CAUSAL if is_causal_mapping else AttnMaskType.FULL
                )

            x_padded, dist_attn_runtime_key = magi_attn_flex_dispatch(
                x,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
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

        # -----    init dist attn runtime key   ---- #
        dist_attn_runtime_mgr: DistAttnRuntimeMgr = DistAttnRuntimeDict[
            dist_attn_runtime_key
        ]

        # HACK: seperate cp group for dkv group-reduce
        dist_attn_runtime_mgr.dist_attn_runtime.cp_group_dkv = self.nccl_groups[1]

        # -----   get local qkv   ---- #
        local_q, local_k, local_v = MHA.forward(x_padded)

        # -----   run dist attn forward on local qkv for local o   ---- #
        local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)

        # -----   undispatch local o to global o   ---- #
        total_out = undispatch(local_out, dist_attn_runtime_key)

        # -----   run backward   ---- #
        grad_total_out = torch.randn_like(total_out).detach()
        dist.all_reduce(grad_total_out.data, group=self.nccl_group)
        total_out.backward(grad_total_out)

        # -----   assert close to torch ref   ---- #
        self.assert_close_to_torch_ref(
            x=x,
            mha=MHA,
            x_grad=x.grad,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            is_causal_mapping=is_causal_mapping,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            total_out=total_out,
            grad_total_out=grad_total_out,
            test_case=test_case,
        )

    def assert_close_to_torch_ref(
        self,
        x: torch.Tensor,
        mha: MultiHeadAttention,
        x_grad: torch.Tensor,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        is_causal_mapping: bool | list[bool],
        total_seqlen_q: int,
        total_seqlen_k: int,
        total_out: torch.Tensor,
        grad_total_out: torch.Tensor,
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance threshold  ---- #

        o_atol = EPSILON
        o_rtol = EPSILON

        dq_atol = EPSILON
        dq_rtol = EPSILON
        # -----   build attn mask   ---- #

        mask = get_attn_mask_from_ranges(
            q_ranges=q_ranges.to_naive_ranges(),
            k_ranges=k_ranges.to_naive_ranges(),
            is_causal_mapping=is_causal_mapping,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
        )

        # -----   get total qkv   ---- #

        x.grad = None
        total_q, total_k, total_v = mha.forward(x)

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

        # -----   assert close for fwd out   ---- #

        magi_attention.testing.assert_close(
            total_out,
            total_out_ref_high_precision,
            atol=o_atol,
            rtol=o_rtol,
            test_case=f"{test_case} => o",
        )

        # -----   assert close for bwd dx   ---- #

        magi_attention.testing.assert_close(
            x_grad,
            x.grad,
            atol=dq_atol,
            rtol=dq_rtol,
            test_case=f"{test_case} => dq",
        )


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
