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

from dataclasses import dataclass
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.functional import cutedsl_ffa
from magi_attention.kernel.cutedsl.ffa_utils import get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_bwd
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing.utils import switch_envvar_context

RANGE_MERGE_ENV = "MAGI_ATTENTION_RANGE_MERGE"
NUM_HEADS = 2
# A multiple of 32: with range_merge=False the kernel host then uses the dense
# dQacc layout exactly when disable_fwd_atomic_reduction is set.
HEAD_DIM = 128


@dataclass(frozen=True)
class RangeCase:
    q_ranges: list[list[int]]
    k_ranges: list[list[int]]
    total_q: int
    total_k: int
    # q ranges after the bwd merge sorts the rows by k start.
    kernel_q_ranges: list[list[int]]
    expect_direct_dq: bool


CASES = {
    # Batches 2 and 3 map to the same dense dQacc slot.
    "slot_collision": RangeCase(
        q_ranges=[[0, 128], [128, 256], [256, 384], [384, 512]],
        k_ranges=[[128, 256], [384, 512], [256, 384], [0, 128]],
        total_q=512,
        total_k=512,
        kernel_q_ranges=[[384, 512], [0, 128], [256, 384], [128, 256]],
        expect_direct_dq=False,
    ),
    # Slots do not collide, but rows [512, 640) are uncovered, so the
    # binary-search hole zeroing runs on unsorted starts.
    "hole_zeroing": RangeCase(
        q_ranges=[[0, 100], [100, 300], [300, 350], [350, 512]],
        k_ranges=[[256, 512], [0, 128], [384, 450], [100, 300]],
        total_q=640,
        total_k=512,
        kernel_q_ranges=[[100, 300], [350, 512], [0, 100], [300, 350]],
        expect_direct_dq=False,
    ),
    # The merge keeps the order, so the direct store stays enabled.
    "order_kept": RangeCase(
        q_ranges=[[0, 128], [128, 256], [256, 384], [384, 512]],
        k_ranges=[[0, 128], [128, 256], [256, 384], [384, 512]],
        total_q=512,
        total_k=512,
        kernel_q_ranges=[[0, 128], [128, 256], [256, 384], [384, 512]],
        expect_direct_dq=True,
    ),
}


class TestCutedslBwdUnderRangeMerge(TestCase):
    """Backward range ordering under MAGI_ATTENTION_RANGE_MERGE=1.

    calc_meta reorders backward rows by k start. disable_fwd_atomic_reduction
    requires the q ranges passed to the kernel to be sorted and disjoint.
    """

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the range bwd direct dQ store exists only on SM100/SM110")

    def _run_case(self, name: str) -> None:
        case = CASES[name]
        with switch_envvar_context(RANGE_MERGE_ENV, enable=True):
            attn_arg = AttnArg(
                q_ranges=AttnRanges.from_ranges(case.q_ranges),
                k_ranges=AttnRanges.from_ranges(case.k_ranges),
                attn_type_map=[0] * len(case.q_ranges),
            )
            gen = torch.Generator(device="cuda").manual_seed(1234)

            def randn(seqlen: int) -> torch.Tensor:
                return torch.randn(
                    seqlen,
                    NUM_HEADS,
                    HEAD_DIM,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )

            q, k, v = randn(case.total_q), randn(case.total_k), randn(case.total_k)
            do = randn(case.total_q)
            out, lse = cutedsl_ffa.cutedsl_fwd(
                q, k, v, attn_arg, softmax_scale=None, softcap=0.0
            )
            with mock.patch.object(
                cutedsl_ffa, "_flex_flash_attn_bwd", wraps=_flex_flash_attn_bwd
            ) as bwd:
                dq, dk, dv, _ = cutedsl_ffa.cutedsl_bwd(
                    do, q, k, v, out, lse, attn_arg, softmax_scale=None, softcap=0.0
                )

        kw = bwd.call_args.kwargs
        kernel_q = kw["q_ranges"]
        # The merge produced the row order this case targets.
        self.assertEqual(kernel_q.tolist(), case.kernel_q_ranges, name)

        direct_dq = kw["disable_fwd_atomic_reduction"]

        # The specialization that ran: no in-kernel merge, so the dense dQacc
        # layout follows disable_fwd_atomic_reduction.
        self.assertFalse(kw["range_merge"], name)
        self.assertEqual(direct_dq, case.expect_direct_dq, name)

        # fp32 reference over the union of (q, k) pairs.
        mask = torch.zeros(case.total_q, case.total_k, dtype=torch.bool, device="cuda")
        for (qs, qe), (ks, ke) in zip(case.q_ranges, case.k_ranges):
            mask[qs:qe, ks:ke] = True
        qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
        s = torch.einsum("qhd,khd->hqk", qf, kf) * HEAD_DIM**-0.5
        p = torch.softmax(s.masked_fill(~mask[None], float("-inf")), -1)
        torch.einsum("hqk,khd->qhd", p.nan_to_num(0.0), vf).backward(do.float())
        for grad_name, got, ref in (
            ("dq", dq, qf.grad),
            ("dk", dk, kf.grad),
            ("dv", dv, vf.grad),
        ):
            torch.testing.assert_close(
                got.float(),
                ref,
                atol=2e-2,
                rtol=0.0,
                msg=lambda m: f"{name} {grad_name}: {m}",
            )

    def test_slot_collision(self):
        self._run_case("slot_collision")

    def test_hole_zeroing(self):
        self._run_case("hole_zeroing")

    def test_order_kept(self):
        self._run_case("order_kept")


if __name__ == "__main__":
    run_tests()
