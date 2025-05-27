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
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from enum import Enum

import torch
import torch.distributed as dist
from baselines.attn import AttnBackend, get_max_seqlen
from baselines.ring_attn import RingAttnAllGather, RingAttnP2P
from baselines.shard import (
    ParallelMode,
    get_ring_pg,
    get_ulysess_pg,
    get_usp_pg,
    init_distributed,
    set_seed,
)
from baselines.ulysess import Ulysess
from baselines.usp import USP
from test_utils import (
    collect_global_grad,
    generate_test_data,
    get_attn_mask_from_cu_seqlens,
    test_fa3_varlen_func,
    test_torch_sdpa_func,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase

import magi_attention
import magi_attention.testing
from magi_attention.common.enum import AttnMaskType
from magi_attention.testing.precision import (
    EPSILON,
    calc_inf_norm,
    extract_mismatch_info,
)


class AttnImpl(Enum):
    ULYSSESS = 1
    RING_P2P = 2
    RING_ALLGATHER = 3
    USP = 4


class MyAttnTest(MultiProcessTestCase):
    def __init__(self, TO_TEST=AttnImpl.RING_ALLGATHER):
        self.TO_TEST = TO_TEST

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

        # -----   ref1. fa3 with high precision (fp32)   ---- #

        out_ref_32, dq_ref_32, dk_ref_32, dv_ref_32 = test_torch_sdpa_func(
            total_q, total_k, total_v, total_dout, mask, True
        )

        # -----   ref2. torch ref with low precision (fp16/bf16)   ---- #

        out_ref_16, dq_ref_16, dk_ref_16, dv_ref_16 = test_torch_sdpa_func(
            total_q, total_k, total_v, total_dout, mask, False
        )

        # print(f"max diff out: {torch.abs(out_ref_16 - dist_attn_out).max()}")
        # print(f"max diff dq: {torch.abs(dq_ref_16 - dist_attn_dq).max()}")
        # print(f"max diff dk: {torch.abs(dk_ref_16 - dist_attn_dk).max()}")
        # print(f"max diff dv: {torch.abs(dv_ref_16 - dist_attn_dv).max()}")

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

    def test_attn(self):
        # init distributed environment
        set_seed(42)

        # -----    test ring or all-gather   ---- #
        if self.TO_TEST == AttnImpl.RING_ALLGATHER or self.TO_TEST == AttnImpl.RING_P2P:
            cp_pg_meta = {
                ParallelMode.RING: 4,
            }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_ring_pg(device_shard)

        # -----    test ulysess   ---- #
        elif self.TO_TEST == AttnImpl.ULYSSESS:
            cp_pg_meta = {
                ParallelMode.ULYSESS: 4,
            }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_ulysess_pg(device_shard)

        # -----    test usp   ---- #
        elif self.TO_TEST == AttnImpl.USP:
            cp_pg_meta = {
                ParallelMode.ULYSESS: 2,
                ParallelMode.RING: 2,
            }
            # ulysess [0,1] or ring [0,1]
            # cp_pg_meta = {
            #     ParallelMode.RING: 2,
            #     ParallelMode.ULYSESS: 2,
            # }
            world_size = 4
            device_shard = init_distributed(world_size=world_size, pg_meta=cp_pg_meta)
            cp_group = get_usp_pg(device_shard)

        # -----    set test param   ---- #

        device = torch.cuda.current_device()
        batch_size = 3
        total_seqlen = 4096
        h = 16
        d = 128
        dtype = torch.float16
        # NUM_SAMPLES = batch_size
        qkv_format = "thd"
        deterministic = True
        dropout = 0.0
        attn_mask_type = AttnMaskType.CAUSAL
        causal = attn_mask_type == AttnMaskType.CAUSAL
        attn_backend = AttnBackend.TE

        # -----    init attn module   ---- #

        if self.TO_TEST == AttnImpl.RING_ALLGATHER:
            attn = RingAttnAllGather(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
        elif self.TO_TEST == AttnImpl.RING_P2P:
            attn = RingAttnP2P(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
        elif self.TO_TEST == AttnImpl.ULYSSESS:
            attn = Ulysess(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )
        elif self.TO_TEST == AttnImpl.USP:
            attn = USP(
                cp_process_group=cp_group, qkv_format=qkv_format, backend=attn_backend
            )

        # -----    init test data   ---- #

        q, k, v, dout, cu_seqlens, cu_seqlens_padded = generate_test_data(
            batch_size, total_seqlen, h, d, dtype, qkv_format, device
        )

        dist.broadcast(q.data, src=0)
        dist.broadcast(k.data, src=0)
        dist.broadcast(v.data, src=0)
        dist.broadcast(dout.data, src=0)
        dist.broadcast(cu_seqlens.data, src=0)
        dist.broadcast(cu_seqlens_padded.data, src=0)

        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # print(f"{q.shape=},{q.requires_grad=}")
        # print(f"{k.shape=},{k.requires_grad=}")
        # print(f"{v.shape=},{v.requires_grad=}")

        host_cu_seqlens = cu_seqlens.tolist()
        # print(f"{cu_seqlens=}")

        # -----    dispatch   ---- #

        q_local = attn.dispatch(q, cu_seqlens, host_cu_seqlens, "q")
        k_local = attn.dispatch(k, cu_seqlens, host_cu_seqlens, "k")
        v_local = attn.dispatch(v, cu_seqlens, host_cu_seqlens, "v")

        # -----    forward   ---- #

        # print(f"{q_local.shape=},{q_local.requires_grad=}")

        q_global = attn.undispatch(q_local, "q")
        assert torch.equal(q_global, q)

        out, lse = attn.apply_attn(
            q_local,
            k_local,
            v_local,
            attn_mask_type,
            dropout,
            None,
            deterministic,
        )

        # print(f"{out.shape=}")

        out_global = attn.undispatch(out, "q")

        # -----    backward   ---- #

        # print(f"{out_global.shape=},{out_global.requires_grad=}")
        out_global.backward(dout)

        dq_global = collect_global_grad(attn, q.grad, cu_seqlens, host_cu_seqlens, "dq")
        dk_global = collect_global_grad(attn, k.grad, cu_seqlens, host_cu_seqlens, "dk")
        dv_global = collect_global_grad(attn, v.grad, cu_seqlens, host_cu_seqlens, "dv")

        # -----    simple fa3 test    ---- #

        max_seqlen = get_max_seqlen(host_cu_seqlens)
        (
            test_out_ref,
            test_lse_ref,
            test_dq_ref,
            test_dk_ref,
            test_dv_ref,
        ) = test_fa3_varlen_func(
            q,
            k,
            v,
            dout,
            cu_seqlens,
            cu_seqlens,
            cu_seqlens_padded,
            cu_seqlens_padded,
            max_seqlen,
            max_seqlen,
            causal,
            qkv_format,
            deterministic,
        )

        print(f"max diff out: {torch.abs(test_out_ref - out_global).max()}")
        print(f"max diff dq: {torch.abs(test_dq_ref - dq_global).max()}")
        print(f"max diff dk: {torch.abs(test_dk_ref - dk_global).max()}")
        print(f"max diff dv: {torch.abs(test_dv_ref - dv_global).max()}")

        # -----    assert close torch sdpa ref   ---- #

        mask = get_attn_mask_from_cu_seqlens(cu_seqlens, cu_seqlens, causal)
        self.assert_close_to_sdpa_ref(
            q, k, v, dout, out_global, dq_global, dk_global, dv_global, dtype, mask
        )

        # torch.testing.assert_close(out_global, test_out_ref, atol=o_atol, rtol=o_rtol)
        # torch.testing.assert_close(dq_global, test_dq_ref, atol=dq_atol, rtol=dq_rtol)
        # torch.testing.assert_close(dk_global, test_dk_ref, atol=dk_atol, rtol=dk_rtol)
        # torch.testing.assert_close(dv_global, test_dv_ref, atol=dv_atol, rtol=dv_rtol)

        # -------------------       clearup env   ------------------- #

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    test = MyAttnTest(AttnImpl.USP)
    test.test_attn()
