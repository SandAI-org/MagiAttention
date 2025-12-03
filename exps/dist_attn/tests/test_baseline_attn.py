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

import json
import unittest
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention.testing
from exps.dist_attn.baselines.loongtrain import LoongTrain
from exps.dist_attn.baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from exps.dist_attn.baselines.shard import (
    ParallelMode,
    get_loongtrain_pg,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    set_seed,
)
from exps.dist_attn.baselines.ulysess import Ulysess
from exps.dist_attn.baselines.usp import USP
from exps.dist_attn.baselines.utils_cp import AttnBackend
from exps.dist_attn.tests.test_utils import (
    collect_global_grad,
    get_attn_mask_from_cu_seqlens,
    ref_torch_sdpa_func,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.testing import parameterize
from magi_attention.testing.dist_common import DistTestBase, with_comms
from magi_attention.testing.precision import calc_inf_norm, extract_mismatch_info


class AttnImpl(Enum):
    ULYSSESS = 1
    RING_P2P = 2
    RING_ALLGATHER = 3
    USP = 4
    LOONGTRAIN = 5


global_pg_groups = {}  # type: ignore
global_loongtrain_pg_groups = {}  # type: ignore


def get_device_mesh(pg_meta, world_size):
    pg_sizes = tuple(pg_meta.values())
    pg_names = tuple(pg_meta.keys())
    mesh = torch.arange(0, world_size).reshape(pg_sizes)
    deivce_mesh = DeviceMesh("cuda", mesh=mesh, mesh_dim_names=pg_names)
    return deivce_mesh


def check_pg_exist(pg_meta, pg_groups):
    key = json.dumps({str(k): v for k, v in pg_meta.items()})
    if key in pg_groups:
        return pg_groups[key], key
    else:
        return None, key


class TestBaselineAttn(DistTestBase):
    @property
    def seed(self):
        return 42

    @property
    def device(self):
        return torch.cuda.current_device()

    @property
    def world_size(self) -> int:
        return 8

    def assert_close_to_sdpa_ref(
        self,
        total_q: torch.Tensor,
        total_k: torch.Tensor,
        total_v: torch.Tensor,
        total_dout: torch.Tensor,
        dist_attn_out: torch.Tensor,
        dist_attn_dq: torch.Tensor,
        dist_attn_dk: torch.Tensor,
        dist_attn_dv: torch.Tensor,
        dtype: torch.dtype,
        mask: torch.Tensor,
        test_case: str = "",
    ) -> None:
        # -----   customize tolerance threshold  ---- #

        EPSILON = 1e-7

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
        norm_rtol_ratio: float = 3.0

        # -----   ref1. fa3 with high precision (fp32)   ---- #

        out_ref_32, dq_ref_32, dk_ref_32, dv_ref_32 = ref_torch_sdpa_func(
            total_q, total_k, total_v, total_dout, mask, True
        )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        out_ref_16, dq_ref_16, dk_ref_16, dv_ref_16 = ref_torch_sdpa_func(
            total_q, total_k, total_v, total_dout, mask, False
        )

        if dist.get_rank() == 0:
            print(f"max diff out: {torch.abs(out_ref_16 - dist_attn_out).max()}")
            print(f"max diff dq: {torch.abs(dq_ref_16 - dist_attn_dq).max()}")
            print(f"max diff dk: {torch.abs(dk_ref_16 - dist_attn_dk).max()}")
            print(f"max diff dv: {torch.abs(dv_ref_16 - dist_attn_dv).max()}")

        # -----   assert close for fwd out   ---- #

        # fa style with Linf norm
        out_norm = calc_inf_norm(dist_attn_out, out_ref_32)
        out_ref_norm = calc_inf_norm(out_ref_16, out_ref_32)
        self.assertLessEqual(
            out_norm,
            norm_rtol_ratio * out_ref_norm,
            msg=f"For {test_case=}: {out_norm=} should be no greater than {norm_rtol_ratio}x of {out_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        o_thres = self._extract_mismatch_threshold_ref(
            actual=out_ref_16,
            expected=out_ref_32,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )

        magi_attention.testing.assert_close(
            dist_attn_out,
            out_ref_32,
            atol=o_atol,
            rtol=o_rtol,
            mismatch_threshold=o_thres,
            test_case=f"{test_case} => o",
        )

        # -----   assert close for bwd dq   ---- #

        # fa style with Linf norm
        dq_norm = calc_inf_norm(dist_attn_dq, dq_ref_32)
        dq_ref_norm = calc_inf_norm(dq_ref_16, dq_ref_32)
        self.assertLessEqual(
            dq_norm,
            norm_rtol_ratio * dq_ref_norm,
            msg=f"For {test_case=}: {dq_norm=} should be no greater than {norm_rtol_ratio}x of {dq_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dq_thres = self._extract_mismatch_threshold_ref(
            actual=dq_ref_16,
            expected=dq_ref_32,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )

        magi_attention.testing.assert_close(
            dist_attn_dq,
            dq_ref_16,
            atol=dq_atol,
            rtol=dq_rtol,
            mismatch_threshold=dq_thres,
            test_case=f"{test_case} => dq",
        )

        # -----   assert close for bwd dk   ---- #

        # fa style with Linf norm
        dk_norm = calc_inf_norm(dist_attn_dk, dk_ref_32)
        dk_ref_norm = calc_inf_norm(dk_ref_16, dk_ref_32)
        self.assertLessEqual(
            dk_norm,
            norm_rtol_ratio * dk_ref_norm,
            msg=f"For {test_case=}: {dk_norm=} should be no greater than {norm_rtol_ratio}x of {dk_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dk_thres = self._extract_mismatch_threshold_ref(
            actual=dk_ref_16,
            expected=dk_ref_32,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )

        magi_attention.testing.assert_close(
            dist_attn_dk,
            dk_ref_32,
            atol=dk_atol,
            rtol=dk_rtol,
            mismatch_threshold=dk_thres,
            test_case=f"{test_case} => dk",
        )

        # -----   assert close for bwd dv   ---- #
        # fa style with Linf norm
        dv_norm = calc_inf_norm(dist_attn_dv, dv_ref_32)
        dv_ref_norm = calc_inf_norm(dv_ref_16, dv_ref_32)
        self.assertLessEqual(
            dv_norm,
            norm_rtol_ratio * dv_ref_norm,
            msg=f"For {test_case=}: {dv_norm=} should be no greater than {norm_rtol_ratio}x of {dv_ref_norm=}",
        )

        # torch style with atol + rtol + mismatch threshold
        dv_thres = self._extract_mismatch_threshold_ref(
            actual=dv_ref_16,
            expected=dv_ref_32,
            atol=dv_atol,
            rtol=dv_rtol,
            mismatch_thres_ratio=mismatch_thres_ratio,
        )

        magi_attention.testing.assert_close(
            dist_attn_dv,
            dv_ref_32,
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

    @with_comms
    @parameterize(
        "impl_config",
        [
            {
                "name": AttnImpl.ULYSSESS,
                "cp_pg_meta": {
                    ParallelMode.ULYSESS: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.RING_ALLGATHER,
                "cp_pg_meta": {
                    ParallelMode.RING: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.RING_P2P,
                "cp_pg_meta": {
                    ParallelMode.RING: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.USP,
                "cp_pg_meta": {
                    ParallelMode.RING: 8,
                    ParallelMode.ULYSESS: 1,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.LOONGTRAIN,
                "cp_pg_meta": {
                    ParallelMode.RING: 4,
                    ParallelMode.ULYSESS: 2,
                },
                "world_size": 8,
                "window_num": 2,
            },
        ],
    )
    @parameterize(
        "attn_mask_config",
        [
            {
                "name": "full_4k",
                "seqlen": 4095,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4095],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 4095],
                    ]
                ),
                "attn_mask_type": AttnMaskType.FULL,
            },
            {
                "name": "varlen_full_4k",
                "seqlen": 4095,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4095],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2048],
                        [2048, 4095],
                    ],
                ),
                "attn_mask_type": AttnMaskType.FULL,
            },
            {
                "name": "varlen_causal_2k",
                "seqlen": 2047,
                "q_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2047],
                    ]
                ),
                "k_ranges": AttnRanges.from_ranges(
                    [
                        [0, 256],
                        [256, 512],
                        [512, 1024],
                        [1024, 1280],
                        [1280, 1536],
                        [1536, 1792],
                        [1792, 2047],
                    ],
                ),
                "attn_mask_type": AttnMaskType.CAUSAL,
            },
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
            {
                "name": "gqa_nhq16_nhkv8_hd128",
                "num_heads_q": 16,
                "num_heads_kv": 8,
                "head_dim": 128,
            },
            {
                "name": "gqa_nhq4_nhkv2_hd64",
                "num_heads_q": 4,
                "num_heads_kv": 2,
                "head_dim": 64,
            },
            {
                "name": "mha_nh1_hd64",
                "num_heads_q": 1,
                "num_heads_kv": 1,
                "head_dim": 128,
            },
        ],
    )
    @parameterize("attn_backend", [AttnBackend.FA3, AttnBackend.TE])
    @parameterize("dropout", [0.0])
    @parameterize("deterministic", [True])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    def test_baseline_attn(
        self,
        impl_config: dict[str, Any],
        attn_mask_config: dict[str, Any],
        model_config: dict[str, Any],
        attn_backend: AttnBackend,
        dropout: float,
        deterministic: bool,
        dtype: torch.dtype,
    ):
        # -----    init distributed environment   ---- #
        global global_pg_groups
        global global_loongtrain_pg_groups
        set_seed(self.seed)

        TO_TEST = impl_config["name"]
        world_size = impl_config["world_size"]
        cp_pg_meta = impl_config["cp_pg_meta"]

        # -----    test ring attn   ---- #
        if TO_TEST == AttnImpl.RING_ALLGATHER or TO_TEST == AttnImpl.RING_P2P:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_ring_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test ulysess   ---- #
        elif TO_TEST == AttnImpl.ULYSSESS:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_ulysess_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test usp   ---- #
        elif TO_TEST == AttnImpl.USP:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_usp_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test loongtrain   ---- #
        elif TO_TEST == AttnImpl.LOONGTRAIN:
            # NOTE: param for loongtrain double ring-attention
            window_num = impl_config["window_num"]
            rank = dist.get_rank()
            # NOTE: using pytest to run this test, so we can not use os.environ.get("RANK", 0)
            assert world_size % window_num == 0
            cp_group, key = check_pg_exist(cp_pg_meta, global_loongtrain_pg_groups)
            if cp_group is None:
                cp_group = get_loongtrain_pg(cp_pg_meta, window_num, rank)
                global_loongtrain_pg_groups[key] = cp_group

        # -----    init test data   ---- #

        seqlen = attn_mask_config["seqlen"]
        q_ranges: AttnRanges = attn_mask_config["q_ranges"]
        k_ranges: AttnRanges = attn_mask_config["k_ranges"]
        attn_mask_type = attn_mask_config["attn_mask_type"]
        causal = attn_mask_type == AttnMaskType.CAUSAL

        num_heads_q = model_config["num_heads_q"]
        num_heads_kv = model_config["num_heads_kv"]
        head_dim = model_config["head_dim"]
        if TO_TEST in [AttnImpl.ULYSSESS, AttnImpl.USP, AttnImpl.LOONGTRAIN]:
            a2a_size = cp_pg_meta[ParallelMode.ULYSESS]
            if (
                num_heads_q < a2a_size
                or num_heads_kv < a2a_size
                or num_heads_q % a2a_size != 0
                or num_heads_kv % a2a_size != 0
            ):
                return

        device = self.device
        qkv_format = "thd"

        test_case = f"[{TO_TEST}][{attn_backend}][{attn_mask_config['name']}][{model_config['name']}][{dtype=}]"
        if dist.get_rank() == 0:
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

        if TO_TEST == AttnImpl.RING_ALLGATHER:
            attn = RingAttnAllGather(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.RING_P2P:
            attn = RingAttnP2P(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.ULYSSESS:
            attn = Ulysess(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [device]
        elif TO_TEST == AttnImpl.USP:
            attn = USP(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.LOONGTRAIN:
            attn = LoongTrain(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]

        torch.cuda.synchronize()
        dist.barrier()

        # -----    dispatch   ---- #

        q_local = attn.dispatch(q, q_ranges, seqlen, "q")
        k_local = attn.dispatch(k, k_ranges, seqlen, "k")
        v_local = attn.dispatch(v, k_ranges, seqlen, "v")

        # -----    pre compute   ---- #

        attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

        # -----    forward   ---- #

        out, lse = attn.apply_attn(
            q_local,
            k_local,
            v_local,
            attn_mask_type,
            dropout,
            None,  # type: ignore[arg-type]
            deterministic,
        )

        # -----    backward   ---- #

        out_global = attn.undispatch(out, "q")
        out_global.backward(dout)

        dq_global = collect_global_grad(attn, q.grad, q_ranges, seqlen, "dq")
        dk_global = collect_global_grad(attn, k.grad, k_ranges, seqlen, "dk")
        dv_global = collect_global_grad(attn, v.grad, k_ranges, seqlen, "dv")

        # -----    assert close torch sdpa ref   ---- #

        cu_seqlens_q = torch.tensor(
            q_ranges.to_cu_seqlens(seqlen), device=device, dtype=torch.int32
        )
        cu_seqlens_kv = torch.tensor(
            k_ranges.to_cu_seqlens(seqlen), device=device, dtype=torch.int32
        )
        mask = get_attn_mask_from_cu_seqlens(cu_seqlens_q, cu_seqlens_kv, causal)
        self.assert_close_to_sdpa_ref(
            q, k, v, dout, out_global, dq_global, dk_global, dv_global, dtype, mask
        )
        torch.cuda.synchronize()
        dist.barrier()

    @with_comms
    @parameterize(
        "impl_config",
        [
            {
                "name": AttnImpl.ULYSSESS,
                "cp_pg_meta": {
                    ParallelMode.ULYSESS: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.RING_ALLGATHER,
                "cp_pg_meta": {
                    ParallelMode.RING: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.RING_P2P,
                "cp_pg_meta": {
                    ParallelMode.RING: 8,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.USP,
                "cp_pg_meta": {
                    ParallelMode.RING: 4,
                    ParallelMode.ULYSESS: 2,
                },
                "world_size": 8,
            },
            {
                "name": AttnImpl.LOONGTRAIN,
                "cp_pg_meta": {
                    ParallelMode.RING: 4,
                    ParallelMode.ULYSESS: 2,
                },
                "world_size": 8,
                "window_num": 2,
            },
        ],
    )
    @parameterize("attn_mask_type", [AttnMaskType.FULL, AttnMaskType.CAUSAL])
    @parameterize("dtype", [torch.float16, torch.bfloat16])
    @parameterize("attn_backend", [AttnBackend.FA3, AttnBackend.TE])
    def test_random_sample_baseline_attn(
        self,
        impl_config: dict[str, Any],
        attn_mask_type: AttnMaskType,
        dtype: torch.dtype,
        attn_backend: AttnBackend,
    ):
        # -----    init distributed environment   ---- #
        global global_pg_groups
        global global_loongtrain_pg_groups
        set_seed(self.seed)

        TO_TEST = impl_config["name"]
        world_size = impl_config["world_size"]
        cp_pg_meta = impl_config["cp_pg_meta"]

        # -----    test ring attn   ---- #
        if TO_TEST == AttnImpl.RING_ALLGATHER or TO_TEST == AttnImpl.RING_P2P:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_ring_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test ulysess   ---- #
        elif TO_TEST == AttnImpl.ULYSSESS:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_ulysess_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test usp   ---- #
        elif TO_TEST == AttnImpl.USP:
            cp_group, key = check_pg_exist(cp_pg_meta, global_pg_groups)
            if cp_group is None:
                device_shard = get_device_mesh(cp_pg_meta, world_size)
                cp_group = get_usp_pg(device_shard)
                global_pg_groups[key] = cp_group

        # -----    test loongtrain   ---- #
        elif TO_TEST == AttnImpl.LOONGTRAIN:
            # NOTE: param for loongtrain double ring-attention
            window_num = impl_config["window_num"]
            rank = dist.get_rank()
            # NOTE: using pytest to run this test, so we can not use os.environ.get("RANK", 0)
            assert world_size % window_num == 0
            cp_group, key = check_pg_exist(cp_pg_meta, global_loongtrain_pg_groups)
            if cp_group is None:
                cp_group = get_loongtrain_pg(cp_pg_meta, window_num, rank)
                global_loongtrain_pg_groups[key] = cp_group

        # -----    init test data   ---- #

        seqlen, num_heads_q, num_heads_kv, head_dim = 4095, 16, 8, 128
        causal = attn_mask_type == AttnMaskType.CAUSAL
        device = self.device
        qkv_format = "thd"
        rng = np.random.default_rng(42)
        all_seqlens = [
            np.sort(rng.integers(0, 4095, size=10)).tolist() for i in range(3)
        ]

        test_case = f"[{TO_TEST}][{attn_mask_type}][{attn_backend}][{dtype=}]"
        if dist.get_rank() == 0:
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

        if TO_TEST == AttnImpl.RING_ALLGATHER:
            attn = RingAttnAllGather(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.RING_P2P:
            attn = RingAttnP2P(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.ULYSSESS:
            attn = Ulysess(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [device]
        elif TO_TEST == AttnImpl.USP:
            attn = USP(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]
        elif TO_TEST == AttnImpl.LOONGTRAIN:
            attn = LoongTrain(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )  # type: ignore
            cal_runtime_args = [attn_mask_type, device]

        torch.cuda.synchronize()
        dist.barrier()

        for i in range(3):
            seqlens = [0] + all_seqlens[i] + [seqlen]
            seqlens = [[seqlens[j], seqlens[j + 1]] for j in range(len(seqlens) - 1)]
            q_ranges = AttnRanges.from_ranges(seqlens)
            k_ranges = AttnRanges.from_ranges(seqlens)

            # -----    dispatch   ---- #
            q.grad = None
            k.grad = None
            v.grad = None
            q_local = attn.dispatch(q, q_ranges, seqlen, "q")
            k_local = attn.dispatch(k, k_ranges, seqlen, "k")
            v_local = attn.dispatch(v, k_ranges, seqlen, "v")

            # -----    pre compute   ---- #

            attn.pre_compute_attn_runtime_meta(*cal_runtime_args)

            # -----    forward   ---- #

            out, lse = attn.apply_attn(
                q_local,
                k_local,
                v_local,
                attn_mask_type,
                0.0,
                None,  # type: ignore[arg-type]
                True,
            )

            # -----    backward   ---- #

            out_global = attn.undispatch(out, "q")
            out_global.backward(dout)

            dq_global = collect_global_grad(attn, q.grad, q_ranges, seqlen, "dq")
            dk_global = collect_global_grad(attn, k.grad, k_ranges, seqlen, "dk")
            dv_global = collect_global_grad(attn, v.grad, k_ranges, seqlen, "dv")

            # -----    assert close torch sdpa ref   ---- #

            cu_seqlens_q = torch.tensor(
                q_ranges.to_cu_seqlens(seqlen), device=device, dtype=torch.int32
            )
            cu_seqlens_kv = torch.tensor(
                k_ranges.to_cu_seqlens(seqlen), device=device, dtype=torch.int32
            )
            mask = get_attn_mask_from_cu_seqlens(cu_seqlens_q, cu_seqlens_kv, causal)
            self.assert_close_to_sdpa_ref(
                q, k, v, dout, out_global, dq_global, dk_global, dv_global, dtype, mask
            )

            torch.cuda.synchronize()
            dist.barrier()


if __name__ == "__main__":
    unittest.main()
