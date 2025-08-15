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

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.api.functools import (
    apply_padding,
    compute_pad_size,
    infer_varlen_mask_from_batch,
    pad_at_dim,
)
from magi_attention.api.magi_attn_interface import (
    calc_attn,
    dispatch,
    dist_attn_runtime_dict,
    get_position_ids,
    magi_attn_flex_dispatch,
    magi_attn_flex_key,
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
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeMgr,
    init_dist_attn_runtime_mgr,
)
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import (
    INTERFACE,
    NAME,
    SKIP_WORLD_SIZE,
    DistTestBase,
    with_comms,
)
from magi_attention.testing.precision import (
    H100_MATMUL_MFU,
    H100_NVLINK_A2A_BWU,
    H100_NVLINK_BANDWIDTH,
    H100_TFLOPS_16,
)
from magi_attention.testing.utils import switch_deterministic_mode_decorator
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_calc_cost_factor,
    get_comm_cost_factor,
    is_list_value_all,
)


class TestInterfaceBaseWithWorldSize1(DistTestBase):
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
        [(6, 2)],  # gqa
    )
    @parameterize(
        "head_dim",
        [128],
    )
    @parameterize(
        "dtype",
        [torch.bfloat16],
    )
    def test_interface(
        self,
        attn_config: dict[str, Any],
        overlap_config: dict[str, Any],
        num_heads: tuple[int, int],  # (nhq, nhkv)
        head_dim: int,
        dtype: torch.dtype,
    ):
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
        num_heads_q, num_heads_kv = num_heads

        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
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
        )

        # ----- init input data and module ----- #

        x = torch.randn(
            total_seqlen_q,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=True,
        )

        # --------- calculate pad size --------- #

        pad_size = compute_pad_size(total_seqlen_q, self.world_size, chunk_size)

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

        # ------ test interface ------ #

        match interface:
            case "magi_attn":
                assert is_list_value_all(
                    attn_mask_type, AttnMaskType.FULL
                ) or is_list_value_all(
                    attn_mask_type, AttnMaskType.CAUSAL
                ), "we need to check varlen interface, which supports full or causal now"
                is_causal = attn_mask_type[0] == AttnMaskType.CAUSAL

                batch_size = attn_config["batch_size"]
                cu_seqlens_q, cu_seqlens_k = infer_varlen_mask_from_batch(
                    batch_size, attn_config["total_seqlen_q"] // batch_size
                )

                _, dist_attn_runtime_key = magi_attn_varlen_dispatch(
                    x,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    causal=is_causal,
                    dist_attn_config=dist_attn_config,
                )
            case "magi_attn_varlen":
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
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    causal=is_causal,
                    dist_attn_config=dist_attn_config,
                )
            case "magi_attn_flex":
                use_str_masktype: bool = attn_config["use_str_masktype"]
                local_x_padded, dist_attn_runtime_key = magi_attn_flex_dispatch(
                    x,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_mask_type=[masktype.value for masktype in attn_mask_type]
                    if use_str_masktype
                    else attn_mask_type,
                    total_seqlen_q=total_seqlen_q,
                    total_seqlen_k=total_seqlen_k,
                    pad_size=pad_size,
                    chunk_size=chunk_size,
                    cp_group_or_mesh=self.device_mesh
                    if magi_attention.comm.is_hierarchical_comm_enable()
                    else self.nccl_group,
                    dist_attn_config=dist_attn_config,
                )
            case "set_mesh_and_group":
                if magi_attention.comm.is_hierarchical_comm_enable():
                    with pytest.raises(AssertionError):
                        _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                            x,
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_mask_type=attn_mask_type,
                            total_seqlen_q=total_seqlen_q,
                            total_seqlen_k=total_seqlen_k,
                            pad_size=pad_size,
                            chunk_size=chunk_size,
                            cp_group_or_mesh=self.nccl_group,
                            dist_attn_config=dist_attn_config,
                        )
                else:
                    with pytest.raises(ValueError):
                        _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                            x,
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_mask_type=attn_mask_type,
                            total_seqlen_q=total_seqlen_q,
                            total_seqlen_k=total_seqlen_k,
                            pad_size=pad_size,
                            chunk_size=chunk_size,
                            cp_group_or_mesh=self.device_mesh,
                            dist_attn_config=dist_attn_config,
                        )
                return
            case "test_for_invalid_mask":
                invalid_mask_type = attn_config["attn_mask_type"]
                with pytest.raises(ValueError):
                    _, dist_attn_runtime_key = magi_attn_flex_dispatch(
                        x,
                        q_ranges=q_ranges,
                        k_ranges=k_ranges,
                        attn_mask_type=invalid_mask_type,
                        total_seqlen_q=total_seqlen_q,
                        total_seqlen_k=total_seqlen_k,
                        pad_size=pad_size,
                        chunk_size=chunk_size,
                        cp_group_or_mesh=self.device_mesh
                        if magi_attention.comm.is_hierarchical_comm_enable()
                        else self.nccl_group,
                        dist_attn_config=dist_attn_config,
                    )
                return

        # -----    compute dist attn runtime mgr   ---- #

        dist_attn_runtime_mgr: DistAttnRuntimeMgr = dist_attn_runtime_dict[
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

        # -------   check mgr equality to ref -------- #

        assert (
            dist_attn_runtime_mgr == ref_attn_runtime_mgr
        ), f"the answer is not correct when {test_case=}"

        # -------   test position ids -------- #

        if interface == "magi_attn_flex":
            global_x_padded = pad_at_dim(x, 0, pad_size)

            #  -----  get position_ids and check  -----  #

            position_ids = get_position_ids(dist_attn_runtime_key)
            position_ids = position_ids[
                position_ids < total_seqlen_q - 1
            ]  # remove padded id
            valid_length = position_ids.size(0)

            self.assertTrue(
                torch.equal(
                    local_x_padded[:valid_length], global_x_padded[position_ids]
                )
            )


class TestInterfaceWithWorldSize2(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize3(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 3

    @skip_if_lt_x_gpu(3)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize4(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize5(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 5

    @skip_if_lt_x_gpu(5)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize6(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 6

    @skip_if_lt_x_gpu(6)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize7(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 7

    @skip_if_lt_x_gpu(7)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)


class TestInterfaceWithWorldSize8(TestInterfaceBaseWithWorldSize1):
    @property
    def world_size(self) -> int:
        return 8

    @skip_if_lt_x_gpu(8)
    def test_interface(self, *args, **kwargs):
        super().test_interface(*args, **kwargs)

    @skip_if_lt_x_gpu(8)
    @with_comms
    @switch_deterministic_mode_decorator(enable=True)
    def test_compiled_magiattn(self):
        # --- Define attention config --- #

        total_seqlen = 32 * 1024  # 32k tokens
        num_heads_q = 48  # number of attention (query) heads
        num_heads_kv = 8  # number of key/value heads (GQA)
        head_dim = 128  # dimension of each attention head
        dtype = torch.bfloat16  # attention activation / computation dtype
        chunk_size = 512  # chunk size
        embed_dim = 4096  # token embedding tensor

        # --- Initialize MagiAttention meta configs for customized attention mask --- #

        q_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],  # 0~4k
                [4096, 8192],  # 4k~8k
                [8192, 12288],  # 8k~12k
                [12288, 16384],  # 12k~16k
                [16384, 20480],  # 16k~20k
                [20480, 24576],  # 20k~24k
                [24576, 28672],  # 24k~28k
                [28672, 32768],  # 28k~32k
            ]
        )
        k_ranges = AttnRanges.from_ranges(
            [
                [0, 4096],  # 0~4k
                [0, 8192],  # 0~8k
                [0, 12288],  # 0~12k
                [0, 16384],  # 0~16k
                [0, 20480],  # 0~20k
                [0, 24576],  # 0~24k
                [0, 28672],  # 0~28k
                [0, 32768],  # 0~32k
            ]
        )
        attn_mask_type = [AttnMaskType.FULL] * len(q_ranges)
        total_seqlen_q = total_seqlen_k = total_seqlen
        pad_size = compute_pad_size(  # pad embeds along seqlen dim for better performance
            total_seqlen_q=total_seqlen_q,
            cp_size=self.world_size,  # assuming we only have 1-dim context parallelism (cp)
            chunk_size=chunk_size,
        )

        global_dout = torch.randn(
            total_seqlen, num_heads_q, head_dim, device=self.device, dtype=dtype
        )
        dist.all_reduce(global_dout, group=self.process_group)

        q_proj = nn.Linear(
            embed_dim, num_heads_q * head_dim, dtype=dtype, device=self.device
        )
        k_proj = nn.Linear(
            embed_dim, num_heads_kv * head_dim, dtype=dtype, device=self.device
        )
        v_proj = nn.Linear(
            embed_dim, num_heads_kv * head_dim, dtype=dtype, device=self.device
        )

        # --- Compute magi_attn runtime key --- #

        magi_attn_runtime_key = magi_attn_flex_key(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            pad_size=pad_size,
            chunk_size=chunk_size,
            cp_group_or_mesh=self.process_group,  # assuming we only have 1-dim context parallelism (cp)
        )

        total_out_ref, dx_ref = None, None
        for iter in range(6):
            use_compiled_magiattn = iter % 2 == 1

            torch.manual_seed(self.seed + iter // 2)
            x = torch.randn(
                total_seqlen,
                embed_dim,
                device=self.device,
                dtype=dtype,
                requires_grad=True,
            )
            dist.all_reduce(x.data, group=self.process_group)

            # --- Dispatch and pad --- #

            local_x = dispatch(x, key=magi_attn_runtime_key)

            # --- Simulate QKV projection --- #

            local_q = q_proj(local_x).view(-1, num_heads_q, head_dim)
            local_k = k_proj(local_x).view(-1, num_heads_kv, head_dim)
            local_v = v_proj(local_x).view(-1, num_heads_kv, head_dim)

            # --- Apply compiled magi_attn func --- #

            # NOTE: since torch.compile does not support async dist comm,
            # we can not compile it with fullgraph=True
            magiattn_func = (
                torch.compile(fullgraph=False)(calc_attn)
                if use_compiled_magiattn
                else calc_attn
            )
            local_out, _ = magiattn_func(
                q=local_q,
                k=local_k,
                v=local_v,
                key=magi_attn_runtime_key,
            )

            # --- Undispatch and unpad --- #

            total_out = undispatch(
                x=local_out,
                key=magi_attn_runtime_key,
            )

            total_out.backward(global_dout)

            dx = x.grad

            if use_compiled_magiattn:
                assert total_out_ref is not None and dx_ref is not None
                torch.testing.assert_close(total_out, total_out_ref)
                torch.testing.assert_close(dx, dx_ref)
            else:
                total_out_ref = total_out.detach().clone()
                dx_ref = dx.detach().clone()


if __name__ == "__main__":
    run_tests()
