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

"""Correctness tests for per-range Full/Causal mask_types on SM100.

Covers per-range ``mask_types: int32[B]`` varlen fwd+bwd against a torch SDPA
reference with a block-diagonal attn mask built from the per-range map.
"""

import random

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.dist_common import DistTestBase, with_run_in_mp
from magi_attention.testing.precision import (
    EPSILON,
    MAX_MISMATCH_THRES,
    MISMATCH_THRES_RATIO,
    NORM_RTOL_RATIO,
    assert_close,
    calc_inf_norm,
    extract_mismatch_threshold,
)
from magi_attention.utils import make_attn_mask_from_ffa_args

_RTOL = {
    "o": {torch.bfloat16: 0.05, torch.float16: 0.05},
    "dq": {torch.bfloat16: 0.3, torch.float16: 0.2},
    "dk": {torch.bfloat16: 0.15, torch.float16: 0.08},
    "dv": {torch.bfloat16: 0.05, torch.float16: 0.05},
}

_MIN_MISMATCH_THRES = {
    "o": 5e-3,
    "dq": 1e-2,
    "dk": 1e-2,
    "dv": 5e-3,
}


class TestFfaPerRangeMask(DistTestBase):
    @property
    def seed(self) -> int:
        return 42

    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    @property
    def timeout(self) -> int:
        return 600

    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    def _compare(
        self,
        name: str,
        actual: torch.Tensor,
        ref_hi: torch.Tensor,
        ref_lo: torch.Tensor,
        rtol: float,
        test_case: str,
        err_msg_list: list[str],
    ) -> None:
        norm = calc_inf_norm(actual, ref_hi)
        ref_norm = calc_inf_norm(ref_lo, ref_hi)
        try:
            self.assertLessEqual(
                norm,
                NORM_RTOL_RATIO * ref_norm,
                msg=(
                    f"For {test_case=}: {name} {norm=} should be no greater than "
                    f"{NORM_RTOL_RATIO} x {ref_norm=}"
                ),
            )
        except Exception as e:
            err_msg_list.append(str(e))

        thres = extract_mismatch_threshold(
            actual=ref_lo,
            expected=ref_hi,
            atol=EPSILON,
            rtol=rtol,
            mismatch_thres_ratio=MISMATCH_THRES_RATIO,
            min_mismatch_thres=_MIN_MISMATCH_THRES[name],
            max_mismatch_thres=MAX_MISMATCH_THRES,
        )
        try:
            assert_close(
                actual,
                ref_hi,
                atol=EPSILON,
                rtol=rtol,
                mismatch_threshold=thres,
                test_case=f"{test_case} => {name}",
                print_rank=-1,
            )
        except Exception as e:
            err_msg_list.append(str(e))

    def assert_close_to_torch_ref(
        self,
        *,
        q_thd: torch.Tensor,
        k_thd: torch.Tensor,
        v_thd: torch.Tensor,
        do_thd: torch.Tensor,
        out_thd: torch.Tensor,
        dq_thd: torch.Tensor,
        dk_thd: torch.Tensor,
        dv_thd: torch.Tensor,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_type_map: list[int],
        total_seqlen_q: int,
        total_seqlen_k: int,
        dtype: torch.dtype,
        test_case: str,
    ) -> None:
        mask = make_attn_mask_from_ffa_args(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            device=q_thd.device,
        )

        def _ref(high_precision: bool):
            q_ref = q_thd.clone().detach().requires_grad_()
            k_ref = k_thd.clone().detach().requires_grad_()
            v_ref = v_thd.clone().detach().requires_grad_()
            out_ref, _ = ref_attn_func(
                q=q_ref,
                k=k_ref,
                v=v_ref,
                mask=mask,
                layout="thd",
                backend="sdpa",
                high_precision=high_precision,
            )
            dq_ref, dk_ref, dv_ref = torch.autograd.grad(
                out_ref, (q_ref, k_ref, v_ref), do_thd
            )
            return out_ref, dq_ref, dk_ref, dv_ref

        out_hi, dq_hi, dk_hi, dv_hi = _ref(high_precision=True)
        out_lo, dq_lo, dk_lo, dv_lo = _ref(high_precision=False)

        err_msg_list: list[str] = []
        for name, actual, ref_hi, ref_lo in [
            ("o", out_thd, out_hi, out_lo),
            ("dq", dq_thd, dq_hi, dq_lo),
            ("dk", dk_thd, dk_hi, dk_lo),
            ("dv", dv_thd, dv_hi, dv_lo),
        ]:
            self._compare(
                name=name,
                actual=actual,
                ref_hi=ref_hi,
                ref_lo=ref_lo,
                rtol=_RTOL[name][dtype],
                test_case=test_case,
                err_msg_list=err_msg_list,
            )

        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

    ALL_TYPES = [MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal]

    def _run_case(
        self,
        *,
        name: str,
        seqlen_q: int,
        seqlen_k: int,
        d: int,
        mha_type: str,
        dtype: torch.dtype,
        attn_type_map: list[int],
    ) -> None:
        """Launch one per-range case and compare every output to the torch ref."""
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("Per-range mask_types requires SM100/SM110")

        device = self.device
        seed = self.seed + seqlen_q + seqlen_k + d
        torch.random.manual_seed(seed)
        random.seed(seed)

        batch_size = len(attn_type_map)
        nheads = 4
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]

        q = torch.randn(
            batch_size * seqlen_q, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()

        cu_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32
        )
        cu_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
        )
        test_case = (
            f"[RANK {self.rank}][{name}]"
            f"[{seqlen_q=}][{seqlen_k=}][{d=}][{mha_type=}][{dtype=}]"
            f"[{attn_type_map=}]"
        )

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=torch.stack([cu_q[:-1], cu_q[1:]], dim=1),
            k_ranges=torch.stack([cu_k[:-1], cu_k[1:]], dim=1),
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
        )
        g = torch.randn_like(out)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)

        for tensor_name, tensor in (("o", out), ("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertFalse(
                tensor.isnan().any(),
                msg=f"For {test_case}: {tensor_name} contains NaN",
            )

        self.assert_close_to_torch_ref(
            q_thd=q.detach(),
            k_thd=k.detach(),
            v_thd=v.detach(),
            do_thd=g,
            out_thd=out,
            dq_thd=dq,
            dk_thd=dk,
            dv_thd=dv,
            q_ranges=AttnRanges.from_ranges(
                [[i * seqlen_q, (i + 1) * seqlen_q] for i in range(batch_size)]
            ),
            k_ranges=AttnRanges.from_ranges(
                [[i * seqlen_k, (i + 1) * seqlen_k] for i in range(batch_size)]
            ),
            attn_type_map=attn_type_map,
            total_seqlen_q=batch_size * seqlen_q,
            total_seqlen_k=batch_size * seqlen_k,
            dtype=dtype,
            test_case=test_case,
        )

    @with_run_in_mp
    @parameterize("dtype", [torch.bfloat16, torch.float16])
    @parameterize("mha_type", ["mha", "gqa"])
    @parameterize("d", [64, 128])
    @parameterize("seqlen", [64, 128])
    def test_per_range_all_mask_types_fwd_bwd(self, seqlen, d, mha_type, dtype):
        # Sk = 2 * Sq keeps every type non-degenerate: with Sq == Sk BiCausal
        # collapses to the identity mask, whose dQ/dK are analytically zero, and
        # a relative comparison against exact zero is meaningless.
        self._run_case(
            name="all_mask_types",
            seqlen_q=seqlen,
            seqlen_k=seqlen * 2,
            d=d,
            mha_type=mha_type,
            dtype=dtype,
            attn_type_map=self.ALL_TYPES,
        )

    @with_run_in_mp
    @parameterize("seqlen_q", [1, 127, 128, 129, 255, 256, 257])
    def test_per_range_tile_boundary_lengths(self, seqlen_q):
        # Sweep lengths straddling the 128-wide tiles. Sk keeps a margin over Sq
        # so the BiCausal band stays wide enough to carry real gradients.
        self._run_case(
            name="tile_boundary",
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_q + 128,
            d=64,
            mha_type="mha",
            dtype=torch.bfloat16,
            attn_type_map=self.ALL_TYPES,
        )

    @with_run_in_mp
    @parameterize("seqlen_k", [64, 127])
    def test_per_range_fully_masked_rows(self, seqlen_k):
        # Sq > Sk leaves Q rows with no visible key: Causal masks the leading
        # rows, InvCausal the trailing ones, and BiCausal the whole range (its
        # band is empty once Sk < Sq). Reference and kernel must both yield
        # zeros rather than NaN.
        self._run_case(
            name="fully_masked",
            seqlen_q=256,
            seqlen_k=seqlen_k,
            d=64,
            mha_type="mha",
            dtype=torch.bfloat16,
            attn_type_map=self.ALL_TYPES,
        )

    @with_run_in_mp
    @parameterize("seqlen_q", [1, 255, 256])
    def test_per_range_single_key_column(self, seqlen_q):
        """``Sk == 1``: every visible row sees exactly one key, so ``p == 1`` and
        ``dS = p * (dP - dPsum) == 0``. dQ/dK are therefore analytically zero, and
        the reference produces exact zeros -- a relative comparison against them is
        ill-posed, so assert the analytic invariant directly instead.
        """
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("Per-range mask_types requires SM100/SM110")

        device = self.device
        torch.random.manual_seed(self.seed + seqlen_q)
        attn_type_map = self.ALL_TYPES
        batch_size, seqlen_k, nheads, d = len(attn_type_map), 1, 4, 64
        dtype = torch.bfloat16

        q = torch.randn(
            batch_size * seqlen_q, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k = torch.randn(
            batch_size * seqlen_k, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            batch_size * seqlen_k, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        cu_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32
        )
        cu_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
        )

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=torch.stack([cu_q[:-1], cu_q[1:]], dim=1),
            k_ranges=torch.stack([cu_k[:-1], cu_k[1:]], dim=1),
            mask_types=torch.tensor(attn_type_map, device=device, dtype=torch.int32),
            max_seqlen_q=seqlen_q,
            max_seqlen_k=seqlen_k,
        )
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), torch.randn_like(out))

        test_case = f"[RANK {self.rank}][single_key][{seqlen_q=}]"
        for name, tensor in (("o", out), ("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertFalse(
                tensor.isnan().any(), msg=f"For {test_case}: {name} contains NaN"
            )
        for name, tensor in (("dq", dq), ("dk", dk)):
            self.assertLess(
                tensor.abs().max().item(),
                1e-3,
                msg=f"For {test_case}: {name} must be analytically zero",
            )



    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    def test_per_range_deterministic_bwd(self, mha_type):
        """deterministic bwd on the per-range path must be bit-reproducible.

        Exercised because ``spt = (is_causal or is_local) and deterministic`` is
        true here (per-range forces ``causal``), which routes dQ's semaphore lock
        value through ``get_n_block_max_for_m_block_per_range``.

        Sk is kept large so a Q tile spans many K tiles: with few writers the
        non-deterministic path happens to be stable too and the test would not
        distinguish the two.
        """
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("Per-range mask_types requires SM100/SM110")

        device = self.device
        torch.random.manual_seed(self.seed)
        attn_type_map = self.ALL_TYPES
        batch_size = len(attn_type_map)
        seqlen_q, seqlen_k, d = 128, 1024, 64
        nheads = 4
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]
        dtype = torch.bfloat16

        q = torch.randn(
            batch_size * seqlen_q, nheads, d, device=device, dtype=dtype
        ).requires_grad_()
        k = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            batch_size * seqlen_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        g = torch.randn(batch_size * seqlen_q, nheads, d, device=device, dtype=dtype)
        cu_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32
        )
        cu_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
        )
        mask_types = torch.tensor(attn_type_map, device=device, dtype=torch.int32)

        def run():
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=torch.stack([cu_q[:-1], cu_q[1:]], dim=1),
                k_ranges=torch.stack([cu_k[:-1], cu_k[1:]], dim=1),
                mask_types=mask_types,
                max_seqlen_q=seqlen_q,
                max_seqlen_k=seqlen_k,
                deterministic=True,
            )
            return (out,) + torch.autograd.grad(out, (q, k, v), g)

        test_case = f"[RANK {self.rank}][deterministic][{mha_type=}]"
        first = run()
        for name, tensor in zip(("o", "dq", "dk", "dv"), first):
            self.assertFalse(
                tensor.isnan().any(), msg=f"For {test_case}: {name} contains NaN"
            )
        for _ in range(2):
            for name, expected, actual in zip(("o", "dq", "dk", "dv"), first, run()):
                self.assertTrue(
                    torch.equal(expected, actual),
                    msg=(
                        f"For {test_case}: {name} is not bit-reproducible across "
                        f"deterministic runs (max diff "
                        f"{(expected.float() - actual.float()).abs().max().item():.3e})"
                    ),
                )



if __name__ == "__main__":
    run_tests()
