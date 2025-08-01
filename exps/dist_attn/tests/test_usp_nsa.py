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

import unittest
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from exps.dist_attn.baselines.nsa import VarlenNSA
from exps.dist_attn.baselines.shard import ParallelMode, get_usp_pg, set_seed
from exps.dist_attn.baselines.usp_nsa import USPAllGatherNSA
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms

# from magi_attention.testing.precision import calc_inf_norm, extract_mismatch_info


def collect_global_grad(attn, grad, ranges):
    grad_part = attn.dispatch(grad, ranges)
    grad_global = attn.undispatch(grad_part)
    return grad_global


def clone_linear(linear: torch.nn.Linear):
    new_linear = torch.nn.Linear(
        linear.in_features,
        linear.out_features,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new_linear.load_state_dict(linear.state_dict())
    return new_linear


class TestUspNsaAttn(DistTestBase):
    def init_pg(self) -> None:
        super().init_pg()

        # -----    set up for hier comm   ---- #

        cp_pg_meta = {
            ParallelMode.RING: 2,
            ParallelMode.ULYSESS: 2,
        }
        world_size = 4
        pg_sizes = tuple(cp_pg_meta.values())
        pg_names = tuple(cp_pg_meta.keys())
        mesh = torch.arange(0, world_size).reshape(pg_sizes)
        deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=pg_names)

        # init several pgs with all ranks
        self.nccl_groups = get_usp_pg(deivce_mesh)

    @property
    def world_size(self) -> int:
        return 4

    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    @with_comms
    @parameterize(
        "attn_mask_config",
        [
            # {
            #     "name": "full_2k",
            #     "seqlen": 2048,
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 2048],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 2048],
            #         ]
            #     ),
            #     "attn_mask_type": AttnMaskType.FULL,
            # },
            {
                "name": "varlen_full_4k",
                "seqlen": 4096,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 4096],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 1024],
                        [1024, 4096],
                    ],
                ),
                "attn_mask_type": AttnMaskType.FULL,
            },
            # {
            #     "name": "varlen_full_4k",
            #     "seqlen": 4096,
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 1024],
            #             [1024, 3072],
            #             [3072, 4096],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 1024],
            #             [1024, 3072],
            #             [3072, 4096]
            #         ],
            #     ),
            #     "attn_mask_type": AttnMaskType.FULL,
            # },
            # {
            #     "name": "varlen_full_2k",
            #     "seqlen": 2048,
            #     "q_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 256],
            #             [256, 512],
            #             [512, 1024],
            #             [1024, 1280],
            #             [1280, 1536],
            #             [1536, 1792],
            #             [1792, 2048],
            #         ]
            #     ),
            #     "k_ranges": AttnRanges.from_ranges(
            #         [
            #             [0, 256],
            #             [256, 512],
            #             [512, 1024],
            #             [1024, 1280],
            #             [1280, 1536],
            #             [1536, 1792],
            #             [1792, 2048],
            #         ],
            #     ),
            #     "attn_mask_type": AttnMaskType.FULL,
            # },
        ],
    )
    @parameterize(
        "nsa_config",
        [
            {
                "l_slc": 64,
                "l_cmp": 64,
                "stride": 64,
                "slc_topk": 16,
                "block_size_q": 32,
                "window_size_left": 4,
                "window_size_right": 4,
            }
        ],
    )
    @parameterize(
        "model_config",
        [
            {
                "name": "mha_nh8_hd128",
                "num_heads_q": 8,
                "num_heads_kv": 8,
                "head_dim": 128,
            },
            # {
            #     "name": "gqa_nhq8_nhkv4_hd128",
            #     "num_heads_q": 8,
            #     "num_heads_kv": 4,
            #     "head_dim": 128,
            # },
        ],
    )
    @parameterize("deterministic", [True])
    @parameterize("dtype", [torch.float16])
    def test_usp_nsa_attn(
        self,
        attn_mask_config: dict[str, Any],
        nsa_config: dict[str, Any],
        model_config: dict[str, Any],
        deterministic: bool,
        dtype: torch.dtype,
    ):
        set_seed(self.seed)

        # -----    init test data   ---- #

        seqlen = attn_mask_config["seqlen"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        device = self.device

        l_slc = nsa_config["l_slc"]
        l_cmp = nsa_config["l_cmp"]
        stride = nsa_config["stride"]
        slc_topk = nsa_config["slc_topk"]
        block_size_q = nsa_config["block_size_q"]
        window_size_left = nsa_config["window_size_left"]
        window_size_right = nsa_config["window_size_right"]

        test_case = (
            f"[{'usp_nsa'}]"
            f"[{attn_mask_config['name']}]"
            f"[{model_config['name']}]"
            f"[{dtype=}]"
            f"[{l_cmp=}]"
            f"[{l_slc=}]"
            f"[{stride=}]"
            f"[{slc_topk=}]"
            f"[{block_size_q=}]"
            f"[{window_size_left=}]"
            f"[{window_size_right=}]"
        )
        print(test_case)

        q = torch.randn(
            (seqlen, num_heads_q, head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        k = torch.randn(
            (seqlen, num_heads_kv, head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        v = torch.randn(
            (seqlen, num_heads_kv, head_dim),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        dout = torch.randn((seqlen, num_heads_q, head_dim), device=device, dtype=dtype)

        dist.broadcast(q.data, src=0)
        dist.broadcast(k.data, src=0)
        dist.broadcast(v.data, src=0)
        dist.broadcast(dout.data, src=0)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # -----    init attn module   ---- #

        set_seed(self.seed)
        attn = USPAllGatherNSA(
            self.nccl_groups,
            l_cmp,
            l_slc,
            slc_topk,
            stride,
            block_size_q,
            head_dim,
            dtype,
            device,
        )

        # -----    dispatch   ---- #

        q_local = attn.dispatch(q, q_ranges)
        k_local = attn.dispatch(k, k_ranges)
        v_local = attn.dispatch(v, k_ranges)

        q_global = attn.undispatch(q_local)
        assert torch.equal(q, q_global)

        # -----    pre compute   ---- #

        attn.pre_compute_attn_runtime_meta(
            q_ranges, k_ranges, window_size_left, window_size_right, self.device
        )

        # -----    forward   ---- #

        out, lse = attn.apply_attn(
            q_local,
            k_local,
            v_local,
            None,  # type: ignore[arg-type]
            # window_size_left,
            # window_size_right,
            deterministic,
        )

        # -----    backward   ---- #

        out_global = attn.undispatch(out)
        out_global.backward(dout)

        dq_global = collect_global_grad(attn, q.grad, q_ranges)
        dk_global = collect_global_grad(attn, k.grad, k_ranges)
        dv_global = collect_global_grad(attn, v.grad, k_ranges)
        # print(f"{dq_global.shape=},{dk_global.shape=},{dv_global.shape=}")

        # -----    assert close nsa ref   ---- #

        # To avoid grad accumulation/interference, use .detach() and clone() for inputs,
        # and run the reference computation in a fresh context.

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        dout_ref = dout.detach().clone()

        set_seed(self.seed)
        nsa_varlen = VarlenNSA(
            hidden_dim=head_dim,
            d=stride,
            l_cmp=l_cmp,
            l_slc=l_slc,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            block_size_q=block_size_q,
            slc_top_k=slc_topk,
            dtype=dtype,
            device=self.device,
        )

        output_ref = nsa_varlen(q_ref, k_ref, v_ref, q_ranges, k_ranges, None, False)
        output_ref.backward(dout_ref)

        print(f"max diff out: {torch.abs(output_ref - out_global).max()}")
        print(f"max diff dq: {torch.abs(q_ref.grad - dq_global).max()}")
        print(f"max diff dk: {torch.abs(k_ref.grad - dk_global).max()}")
        print(f"max diff dv: {torch.abs(v_ref.grad - dv_global).max()}")

        # print(f"{idx.shape=},{idx_ref.shape=}")
        # assert torch.equal(idx, idx_ref)

        # assert torch.allclose(output_ref, out_global, atol=1e-2, rtol=1e-2), "Output mismatch"
        # assert torch.allclose(q_ref.grad, dq_global, atol=1e-2, rtol=1e-2), "dq mismatch"
        # assert torch.allclose(k_ref.grad, dk_global, atol=1e-2, rtol=1e-2), "dk mismatch"
        # assert torch.allclose(v_ref.grad, dv_global, atol=1e-2, rtol=1e-2), "dv mismatch"


if __name__ == "__main__":
    unittest.main()
