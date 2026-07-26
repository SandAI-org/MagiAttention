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
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP, merge_ranges
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



    @with_run_in_mp
    @parameterize("mha_type", ["mha", "gqa"])
    def test_per_range_scattered_ranges(self, mha_type):
        """Arbitrary non-overlapping ranges: non-zero starts, holes, shuffled row
        order, and Q/K geometries decoupled. Covered rows are gather-compared
        against the reference; uncovered rows are undefined until the fwd
        postprocess lands (2E1) and are exercised by the sentinel test below.
        """
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("True q/k ranges require SM100/SM110")

        device = self.device
        torch.random.manual_seed(self.seed)
        total_q, total_k = 1024, 1536
        nheads = 4
        nheads_kv = {"mha": nheads, "gqa": 2}[mha_type]
        d, dtype = 64, torch.bfloat16
        q_rows = [[640, 897], [64, 191], [320, 448]]
        k_rows = [[1024, 1324], [0, 256], [512, 767]]
        attn_type_map = [MT_MAP.causal, MT_MAP.bi_causal, MT_MAP.inv_causal]

        q = torch.randn(total_q, nheads, d, device=device, dtype=dtype).requires_grad_()
        k = torch.randn(
            total_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        v = torch.randn(
            total_k, nheads_kv, d, device=device, dtype=dtype
        ).requires_grad_()
        q_ranges_t = torch.tensor(q_rows, device=device, dtype=torch.int32)
        k_ranges_t = torch.tensor(k_rows, device=device, dtype=torch.int32)
        mask_types = torch.tensor(attn_type_map, device=device, dtype=torch.int32)

        def run(deterministic):
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                q_ranges=q_ranges_t,
                k_ranges=k_ranges_t,
                mask_types=mask_types,
                max_seqlen_q=max(b - a for a, b in q_rows),
                max_seqlen_k=max(b - a for a, b in k_rows),
                deterministic=deterministic,
            )
            return (out,) + torch.autograd.grad(out, (q, k, v), g)

        g = torch.randn(total_q, nheads, d, device=device, dtype=dtype)
        out, dq, dk, dv = run(deterministic=False)

        test_case = f"[RANK {self.rank}][scattered][{mha_type=}]"
        for name, t in (("o", out), ("dq", dq), ("dk", dk), ("dv", dv)):
            covered = t[
                torch.cat(
                    [
                        torch.arange(a, b, device=device)
                        for a, b in (q_rows if name in ("o", "dq") else k_rows)
                    ]
                )
            ]
            self.assertFalse(
                covered.isnan().any(), msg=f"For {test_case}: {name} contains NaN"
            )

        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_rows),
            k_ranges=AttnRanges.from_ranges(k_rows),
            attn_type_map=attn_type_map,
            total_seqlen_q=total_q,
            total_seqlen_k=total_k,
            device=device,
        )

        def _ref(high_precision):
            qr, kr, vr = (t.clone().detach().requires_grad_() for t in (q, k, v))
            out_ref, _ = ref_attn_func(
                q=qr,
                k=kr,
                v=vr,
                mask=mask,
                layout="thd",
                backend="sdpa",
                high_precision=high_precision,
            )
            return (out_ref,) + torch.autograd.grad(out_ref, (qr, kr, vr), g)

        hi, lo = _ref(True), _ref(False)
        qsel = torch.cat([torch.arange(a, b, device=device) for a, b in q_rows])
        ksel = torch.cat([torch.arange(a, b, device=device) for a, b in k_rows])
        err_msg_list: list[str] = []
        for name, actual, ref_hi, ref_lo, sel in (
            ("o", out, hi[0], lo[0], qsel),
            ("dq", dq, hi[1], lo[1], qsel),
            ("dk", dk, hi[2], lo[2], ksel),
            ("dv", dv, hi[3], lo[3], ksel),
        ):
            self._compare(
                name=name,
                actual=actual[sel],
                ref_hi=ref_hi[sel],
                ref_lo=ref_lo[sel],
                rtol=_RTOL[name][dtype],
                test_case=test_case,
                err_msg_list=err_msg_list,
            )
        if err_msg_list:
            raise AssertionError("\n\n".join(err_msg_list))

        # Bit-compare covered rows only: uncovered rows come from torch.empty
        # and are different garbage on every allocation (undefined until 2E1).
        sels = {"o": qsel, "dq": qsel, "dk": ksel, "dv": ksel}
        first = run(deterministic=True)
        for _ in range(2):
            for name, expected, actual in zip(
                ("o", "dq", "dk", "dv"), first, run(deterministic=True)
            ):
                self.assertTrue(
                    torch.equal(expected[sels[name]], actual[sels[name]]),
                    msg=f"For {test_case}: {name} not bit-reproducible",
                )

    @with_run_in_mp
    @parameterize("dummy", [0])
    def test_per_range_kernel_leaves_uncovered_rows_untouched(self, dummy):
        """With holes, no kernel stage may write a row outside every range.

        Output buffers are prefilled with a sentinel through the internal entry
        points (the public API allocates internally); uncovered rows must come
        back bit-identical. The *values* of uncovered rows are defined only once
        the fwd postprocess lands (2E1).
        """
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("True q/k ranges require SM100/SM110")

        from magi_attention.kernel.cutedsl.ffa_utils import MaskMode
        from magi_attention.kernel.cutedsl.flex_flash_attn import (
            _flex_flash_attn_bwd,
            _flex_flash_attn_fwd,
        )

        device = self.device
        torch.random.manual_seed(self.seed)
        total_q, total_k, nheads, d = 512, 768, 2, 64
        dtype = torch.bfloat16
        q_rows = [[320, 449], [64, 191]]
        k_rows = [[512, 768], [128, 384]]
        q_ranges_t = torch.tensor(q_rows, device=device, dtype=torch.int32)
        k_ranges_t = torch.tensor(k_rows, device=device, dtype=torch.int32)

        q = torch.randn(total_q, nheads, d, device=device, dtype=dtype)
        k = torch.randn(total_k, nheads, d, device=device, dtype=dtype)
        v = torch.randn(total_k, nheads, d, device=device, dtype=dtype)
        s_o, s_lse, s_g = 3.140625, 12345.0, 2.75  # exactly representable

        out = torch.full((total_q, nheads, d), s_o, device=device, dtype=dtype)
        lse = torch.full((nheads, total_q), s_lse, device=device, dtype=torch.float32)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_type=MT_MAP.causal,
            mask_mode=MaskMode.STATIC_CAUSAL,
            max_seqlen_q=max(b - a for a, b in q_rows),
            max_seqlen_k=max(b - a for a, b in k_rows),
        )

        q_cov = torch.zeros(total_q, dtype=torch.bool, device=device)
        for a, b in q_rows:
            q_cov[a:b] = True
        k_cov = torch.zeros(total_k, dtype=torch.bool, device=device)
        for a, b in k_rows:
            k_cov[a:b] = True

        test_case = f"[RANK {self.rank}][sentinel]"
        self.assertTrue(
            (out[~q_cov] == s_o).all(),
            msg=f"{test_case}: fwd wrote uncovered O rows",
        )
        self.assertTrue(
            (lse[:, ~q_cov] == s_lse).all(),
            msg=f"{test_case}: fwd wrote uncovered LSE rows",
        )
        self.assertFalse((out[q_cov] == s_o).all(), msg=f"{test_case}: O not written")

        dout = torch.full((total_q, nheads, d), s_g, device=device, dtype=dtype)
        dq = torch.full_like(q, s_o)
        dk = torch.full_like(k, s_o)
        dv = torch.full_like(v, s_o)
        _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq=dq,
            dk=dk,
            dv=dv,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_type=MT_MAP.causal,
            mask_mode=MaskMode.STATIC_CAUSAL,
            max_seqlen_q=max(b - a for a, b in q_rows),
            max_seqlen_k=max(b - a for a, b in k_rows),
        )
        self.assertTrue(
            (dq[~q_cov] == s_o).all(), msg=f"{test_case}: bwd wrote uncovered dQ rows"
        )
        self.assertTrue(
            (dk[~k_cov] == s_o).all(), msg=f"{test_case}: bwd wrote uncovered dK rows"
        )
        self.assertTrue(
            (dv[~k_cov] == s_o).all(), msg=f"{test_case}: bwd wrote uncovered dV rows"
        )

    @with_run_in_mp
    @parameterize(
        "layout",
        ["same_q_disjoint_k", "unaligned_partial", "three_writers_mixed", "gqa"],
    )
    def test_fwd_atomic_overlapping_q_ranges(self, layout):
        """Overlapping Q ranges must merge into one softmax under range locks.

        Runs the fwd atomic-reduction path (fp32 O, LSE prefilled -inf) and
        compares covered rows against a single fp64 softmax over each row's
        visible-key union; uncovered rows must keep their sentinel.
        """
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("fwd atomic reduction requires SM100/SM110")

        from magi_attention.kernel.cutedsl.ffa_utils import MaskMode
        from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd

        device = self.device
        torch.random.manual_seed(self.seed)
        total, d = 1024, 128
        configs = {
            "same_q_disjoint_k": (
                [[0, 128], [0, 128]],
                [[128, 320], [320, 512]],
                [0, 0],
                (4, 4),
            ),
            "unaligned_partial": (
                [[0, 192], [128, 320]],
                [[512, 704], [704, 896]],
                [0, 0],
                (4, 4),
            ),
            "three_writers_mixed": (
                [[0, 128], [64, 256], [64, 128]],
                [[0, 128], [256, 512], [896, 1024]],
                [1, 0, 2],
                (4, 4),
            ),
            "gqa": (
                [[0, 192], [128, 320]],
                [[512, 704], [704, 896]],
                [0, 1],
                (8, 2),
            ),
        }
        q_rows, k_rows, types, (H, Hkv) = configs[layout]
        qr = torch.tensor(q_rows, device=device, dtype=torch.int32)
        kr = torch.tensor(k_rows, device=device, dtype=torch.int32)
        mt = torch.tensor(types, device=device, dtype=torch.int32)
        q = torch.randn(total, H, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(total, Hkv, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(total, Hkv, d, device=device, dtype=torch.bfloat16)

        s_o = 3.140625
        out = torch.full((total, H, d), s_o, device=device, dtype=torch.float32)
        lse = torch.full((H, total), float("-inf"), device=device, dtype=torch.float32)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            out=out,
            lse=lse,
            q_ranges=qr,
            k_ranges=kr,
            mask_mode=MaskMode.PER_RANGE,
            mask_types_tensor=mt,
            max_seqlen_q=total,
            max_seqlen_k=total,
            disable_fwd_atomic_reduction=False,
        )

        mask = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_rows),
            k_ranges=AttnRanges.from_ranges(k_rows),
            attn_type_map=types,
            total_seqlen_q=total,
            total_seqlen_k=total,
            device=device,
        )
        covered = mask.any(dim=1)
        qf = q.to(torch.float64).transpose(0, 1)
        kf = k.to(torch.float64).transpose(0, 1)
        vf = v.to(torch.float64).transpose(0, 1)
        if Hkv != H:
            kf = kf.repeat_interleave(H // Hkv, dim=0)
            vf = vf.repeat_interleave(H // Hkv, dim=0)
        s = qf @ kf.transpose(-1, -2) / (d**0.5)
        s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
        lse_ref = torch.logsumexp(s, dim=-1)
        o_ref = torch.softmax(s, dim=-1).nan_to_num(0.0) @ vf

        test_case = f"[RANK {self.rank}][fwd_atomic:{layout}]"
        o_got = out.to(torch.float64)[covered]
        o_exp = o_ref.transpose(0, 1)[covered]
        rel = (o_got - o_exp).abs().max().item() / o_exp.abs().max().item()
        lse_err = (
            (lse.to(torch.float64) - lse_ref)[:, covered].abs().max().item()
        )
        self.assertLess(rel, 5e-3, msg=f"{test_case}: O mismatch (rel {rel:.3e})")
        self.assertLess(lse_err, 1e-3, msg=f"{test_case}: LSE mismatch")
        self.assertTrue(
            (out[~covered] == s_o).all(),
            msg=f"{test_case}: atomic fwd wrote uncovered O rows",
        )
        self.assertTrue(
            (lse[:, ~covered] == float("-inf")).all(),
            msg=f"{test_case}: atomic fwd wrote uncovered LSE rows",
        )

    @with_run_in_mp
    @parameterize("dummy", [0])
    def test_fwd_atomic_matches_direct_on_disjoint(self, dummy):
        """On non-overlapping input the atomic path must be bit-equal to the
        direct-write path: skip_correction folds coeff_cur == 1.0 exactly."""
        major = torch.cuda.get_device_capability()[0]
        if major not in (10, 11):
            self.skipTest("fwd atomic reduction requires SM100/SM110")

        from magi_attention.kernel.cutedsl.ffa_utils import MaskMode
        from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_fwd

        device = self.device
        torch.random.manual_seed(self.seed + 1)
        B, s, H, d = 4, 256, 4, 128
        total = B * s
        cu = torch.arange(0, total + 1, s, device=device, dtype=torch.int32)
        qr = torch.stack([cu[:-1], cu[1:]], 1).contiguous()
        mt = torch.ones(B, device=device, dtype=torch.int32)
        q = torch.randn(total, H, d, device=device, dtype=torch.bfloat16)
        k = torch.randn(total, H, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(total, H, d, device=device, dtype=torch.bfloat16)

        common = dict(
            q_ranges=qr,
            k_ranges=qr.clone(),
            mask_mode=MaskMode.PER_RANGE,
            mask_types_tensor=mt,
            max_seqlen_q=s,
            max_seqlen_k=s,
        )
        out_direct, lse_direct = _flex_flash_attn_fwd(q, k, v, **common)
        out_atomic, lse_atomic = _flex_flash_attn_fwd(
            q, k, v, disable_fwd_atomic_reduction=False, **common
        )

        test_case = f"[RANK {self.rank}][fwd_atomic:disjoint]"
        self.assertTrue(
            torch.equal(out_atomic.to(torch.bfloat16), out_direct),
            msg=f"{test_case}: O not bit-equal to the direct path",
        )
        self.assertTrue(
            torch.equal(lse_atomic, lse_direct),
            msg=f"{test_case}: LSE not bit-equal to the direct path",
        )

    @with_run_in_mp
    @parameterize("dummy", [0])
    def test_ws_offsets_cache(self, dummy):
        import gc

        from magi_attention.kernel.cutedsl import ffa_utils

        dev = torch.cuda.current_device()
        ranges = torch.tensor([[0, 100], [100, 300]], device=dev, dtype=torch.int32)

        t1 = ffa_utils.cached_ranges_workspace_offsets(ranges, 64)
        self.assertIs(ffa_utils.cached_ranges_workspace_offsets(ranges, 64), t1)
        self.assertIsNot(ffa_utils.cached_ranges_workspace_offsets(ranges, 128), t1)

        ranges.add_(0)  # version bump must invalidate
        t2 = ffa_utils.cached_ranges_workspace_offsets(ranges, 64)
        self.assertIsNot(t2, t1)
        self.assertTrue(torch.equal(t2, t1))
        self.assertTrue(
            torch.equal(t2, torch.tensor([0, 128], device=dev, dtype=torch.int32))
        )

        key = id(ranges)
        del ranges, t1, t2
        gc.collect()
        self.assertNotIn(key, ffa_utils._ws_offsets_cache)

    @with_run_in_mp
    @parameterize("dummy", [0])
    def test_merge_ranges_contract(self, dummy):
        """Torch-native merge_ranges must reproduce the magi_attn_ext contract."""
        dev = torch.cuda.current_device()

        def t(data, dtype=torch.int32):
            return torch.tensor(data, device=dev, dtype=dtype)

        # The documented example of the ext helper, verbatim.
        merged, s_outer, s_inner, s_types, qk_map, count = merge_ranges(
            t([[20, 30], [10, 20], [10, 20], [20, 30]]),
            t([[100, 110], [120, 130], [140, 150], [160, 170]]),
            t([0, 1, 0, 0]),
        )
        self.assertTrue(torch.equal(merged, t([[10, 20], [20, 30], [0, 0], [0, 0]])))
        self.assertTrue(
            torch.equal(s_outer, t([[10, 20], [10, 20], [20, 30], [20, 30]]))
        )
        self.assertTrue(
            torch.equal(
                s_inner, t([[120, 130], [140, 150], [100, 110], [160, 170]])
            )
        )
        self.assertTrue(torch.equal(s_types, t([1, 0, 0, 0])))
        self.assertTrue(torch.equal(qk_map, t([0, 2, 0, 0])))
        self.assertEqual(count.item(), 2)

        # All-unique input: merge is the sorted identity.
        outer = t([[30, 40], [0, 10], [10, 30]])
        inner = t([[0, 5], [5, 9], [9, 12]])
        types = t([2, 3, 0])
        merged, s_outer, s_inner, s_types, qk_map, count = merge_ranges(
            outer, inner, types
        )
        self.assertTrue(torch.equal(merged, t([[0, 10], [10, 30], [30, 40]])))
        self.assertTrue(torch.equal(merged, s_outer))
        self.assertTrue(torch.equal(s_inner, t([[5, 9], [9, 12], [0, 5]])))
        self.assertTrue(torch.equal(s_types, t([3, 0, 2])))
        self.assertTrue(torch.equal(qk_map, t([0, 1, 2])))
        self.assertEqual(count.item(), 3)

        # All rows share one outer range; sort must stay stable.
        merged, s_outer, s_inner, s_types, qk_map, count = merge_ranges(
            t([[5, 8], [5, 8], [5, 8]]),
            t([[1, 2], [3, 4], [5, 6]]),
            t([1, 2, 3]),
        )
        self.assertTrue(torch.equal(merged, t([[5, 8], [0, 0], [0, 0]])))
        self.assertTrue(torch.equal(s_inner, t([[1, 2], [3, 4], [5, 6]])))
        self.assertTrue(torch.equal(s_types, t([1, 2, 3])))
        self.assertTrue(torch.equal(qk_map, t([0, 0, 0])))
        self.assertEqual(count.item(), 1)


if __name__ == "__main__":
    run_tests()
