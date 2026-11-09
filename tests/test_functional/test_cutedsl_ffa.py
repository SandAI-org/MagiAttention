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

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.common import AttnRanges
from magi_attention.functional import cutedsl_ffa
from magi_attention.kernel.cutedsl.ffa_utils import get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import _flex_flash_attn_bwd
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import parameterize
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
    # Rows 1 and 2 share k start 0 and keep their input order.
    "k_start_tie": RangeCase(
        q_ranges=[[256, 384], [128, 256], [0, 128]],
        k_ranges=[[256, 384], [0, 256], [0, 128]],
        total_q=384,
        total_k=384,
        kernel_q_ranges=[[128, 256], [0, 128], [256, 384]],
        expect_direct_dq=False,
    ),
}

# (q_ranges, k_ranges) row sets; the flag test runs every row order of each set.
ROW_SETS: list[tuple[list[list[int]], list[list[int]]]] = [
    # shared k starts
    (
        [[256, 384], [128, 256], [0, 128]],
        [[256, 384], [0, 256], [0, 128]],
    ),
    # k rows form a partition and include an empty row
    (
        [[128, 256], [0, 128], [256, 384]],
        [[0, 128], [128, 128], [128, 256]],
    ),
    # k rows form a partition
    (
        [[0, 64], [64, 192], [192, 256], [256, 512]],
        [[256, 512], [0, 64], [192, 256], [64, 192]],
    ),
    # overlapping q rows
    (
        [[0, 128], [64, 192], [192, 256]],
        [[0, 64], [0, 64], [64, 128]],
    ),
    # holes in both q and k
    (
        [[0, 64], [128, 192]],
        [[0, 64], [128, 192]],
    ),
]


def _permute_rows(
    q_rows: list[list[int]], k_rows: list[list[int]]
) -> list[tuple[list[list[int]], list[list[int]]]]:
    return [
        ([q_rows[i] for i in order], [k_rows[i] for i in order])
        for order in itertools.permutations(range(len(q_rows)))
    ]


ROW_CASES = [
    row_case
    for q_rows, k_rows in ROW_SETS
    for row_case in _permute_rows(q_rows, k_rows)
]


def _make_attn_arg(
    q_rows: list[list[int]], k_rows: list[list[int]], range_merge: bool = True
) -> AttnArg:
    with switch_envvar_context(RANGE_MERGE_ENV, enable=range_merge):
        return AttnArg(
            q_ranges=AttnRanges.from_ranges(q_rows),
            k_ranges=AttnRanges.from_ranges(k_rows),
            attn_type_map=[0] * len(q_rows),
        )


def _end_if(ranges: AttnRanges, predicate: Callable[[int], bool]) -> int | None:
    return ranges.end if predicate(ranges.end) else None


@contextmanager
def _sync_error_mode():
    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


class TestCutedslBwdUnderRangeMerge(TestCase):
    """Backward range ordering under MAGI_ATTENTION_RANGE_MERGE=1.

    calc_meta reorders backward rows by k start. disable_fwd_atomic_reduction
    requires the q ranges passed to the kernel to be sorted and disjoint.
    """

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the range bwd direct dQ store exists only on SM100/SM110")

    @staticmethod
    def _make_inputs(case: RangeCase) -> tuple[torch.Tensor, ...]:
        """Return (q, k, v, do) for the case."""
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

        return (
            randn(case.total_q),
            randn(case.total_k),
            randn(case.total_k),
            randn(case.total_q),
        )

    @staticmethod
    def _fwd(
        attn_arg: AttnArg, q, k, v
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with switch_envvar_context(RANGE_MERGE_ENV, enable=True):
            return cutedsl_ffa.cutedsl_fwd(
                q, k, v, attn_arg, softmax_scale=None, softcap=0.0
            )

    @staticmethod
    def _bwd(attn_arg: AttnArg, q, k, v, do, out, lse):
        with switch_envvar_context(RANGE_MERGE_ENV, enable=True):
            return cutedsl_ffa.cutedsl_bwd(
                do, q, k, v, out, lse, attn_arg, softmax_scale=None, softcap=0.0
            )

    @staticmethod
    def _reference_grads(case: RangeCase, q, k, v, do) -> tuple[torch.Tensor, ...]:
        """fp32 reference over the union of (q, k) pairs."""
        mask = torch.zeros(case.total_q, case.total_k, dtype=torch.bool, device="cuda")
        for (qs, qe), (ks, ke) in zip(case.q_ranges, case.k_ranges):
            mask[qs:qe, ks:ke] = True
        qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
        s = torch.einsum("qhd,khd->hqk", qf, kf) * HEAD_DIM**-0.5
        p = torch.softmax(s.masked_fill(~mask[None], float("-inf")), -1)
        torch.einsum("hqk,khd->qhd", p.nan_to_num(0.0), vf).backward(do.float())
        return qf.grad, kf.grad, vf.grad

    def _run_case(self, name: str) -> None:
        case = CASES[name]
        attn_arg = _make_attn_arg(case.q_ranges, case.k_ranges)
        q, k, v, do = self._make_inputs(case)
        out, lse, _ = self._fwd(attn_arg, q, k, v)
        with mock.patch.object(
            cutedsl_ffa, "_flex_flash_attn_bwd", wraps=_flex_flash_attn_bwd
        ) as bwd:
            dq, dk, dv, _ = self._bwd(attn_arg, q, k, v, do, out, lse)

        kw = bwd.call_args.kwargs
        self.assertEqual(kw["q_ranges"].tolist(), case.kernel_q_ranges, name)
        self.assertFalse(kw["range_merge"], name)
        self.assertEqual(
            kw["disable_fwd_atomic_reduction"], case.expect_direct_dq, name
        )

        ref_grads = self._reference_grads(case, q, k, v, do)
        for grad_name, got, ref in zip(("dq", "dk", "dv"), (dq, dk, dv), ref_grads):
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

    def test_k_start_tie(self):
        self._run_case("k_start_tie")

    def test_bwd_has_no_device_sync(self):
        case = CASES["hole_zeroing"]
        q, k, v, do = self._make_inputs(case)
        # Compile the kernels outside the checked region.
        warmup_arg = _make_attn_arg(case.q_ranges, case.k_ranges)
        out, lse, _ = self._fwd(warmup_arg, q, k, v)
        self._bwd(warmup_arg, q, k, v, do, out, lse)

        attn_arg = _make_attn_arg(case.q_ranges, case.k_ranges)
        torch.cuda.synchronize()
        with _sync_error_mode():
            self._bwd(attn_arg, q, k, v, do, out, lse)


class TestAttnArgBwdArgs(TestCase):
    """AttnArg's host range flags match the bwd range tensors it builds."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("the bwd range tensors are built on a CUDA device")

    @parameterize("range_merge", [False, True])
    @parameterize("rows", ROW_CASES)
    def test_range_flags_match_tensors(self, range_merge: bool, rows):
        q_rows, k_rows = rows
        attn_arg = _make_attn_arg(q_rows, k_rows, range_merge)
        ffa_args = attn_arg.to_ffa_args(is_bwd=True)
        q_ranges = AttnRanges.from_ranges(ffa_args["q_ranges"].tolist())
        k_ranges = AttnRanges.from_ranges(ffa_args["k_ranges"].tolist())
        msg = f"{range_merge=} {q_rows=} {k_rows=}"

        self.assertEqual(attn_arg.bwd_args_q_sorted, q_ranges.is_sorted(), msg)
        self.assertEqual(
            attn_arg.bwd_args_q_coverage_end,
            _end_if(q_ranges, q_ranges.is_full_coverage),
            msg,
        )
        self.assertEqual(
            attn_arg.bwd_args_k_partition_end,
            _end_if(k_ranges, k_ranges.is_cu_seqlens),
            msg,
        )


if __name__ == "__main__":
    run_tests()
