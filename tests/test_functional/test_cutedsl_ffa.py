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
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention import env
from magi_attention.common import AttnRanges
from magi_attention.functional import cutedsl_ffa
from magi_attention.kernel.cutedsl.ffa_utils import get_device_arch
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import record_bwd_kernels
from magi_attention.testing.utils import switch_env, switch_envvar_context

RANGE_MERGE_ENV = "MAGI_ATTENTION_RANGE_MERGE"
BWD_PACK_FACTOR_ENV = "MAGI_ATTENTION_BWD_PACK_FACTOR"
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


@dataclass(frozen=True)
class FwdStoreCase:
    q_ranges: list[list[int]]
    k_ranges: list[list[int]]
    total_q: int
    total_k: int
    range_merge: bool
    expect_direct_store: bool
    expect_fill_uncovered_rows: bool


FWD_STORE_CASES = {
    "sorted_partition": FwdStoreCase(
        q_ranges=[[0, 128], [128, 384], [384, 512]],
        k_ranges=[[256, 512], [0, 256], [128, 384]],
        total_q=512,
        total_k=512,
        range_merge=False,
        expect_direct_store=True,
        expect_fill_uncovered_rows=False,
    ),
    # Rows [128, 256) and [384, 512) lie outside every q range.
    "uncovered_rows": FwdStoreCase(
        q_ranges=[[0, 128], [256, 384]],
        k_ranges=[[0, 256], [256, 512]],
        total_q=512,
        total_k=512,
        range_merge=False,
        expect_direct_store=True,
        expect_fill_uncovered_rows=True,
    ),
    # Rows [0, 64), [192, 256) and [384, 512) lie outside every q range.
    "uncovered_at_both_ends": FwdStoreCase(
        q_ranges=[[64, 192], [256, 384]],
        k_ranges=[[0, 256], [256, 512]],
        total_q=512,
        total_k=512,
        range_merge=False,
        expect_direct_store=True,
        expect_fill_uncovered_rows=True,
    ),
    "unsorted_rows": FwdStoreCase(
        q_ranges=[[256, 512], [0, 256]],
        k_ranges=[[0, 256], [256, 512]],
        total_q=512,
        total_k=512,
        range_merge=False,
        expect_direct_store=False,
        expect_fill_uncovered_rows=False,
    ),
    # The fwd range merge in AttnArg sorts the rows by q start.
    "unsorted_rows_under_range_merge": FwdStoreCase(
        q_ranges=[[256, 512], [0, 256]],
        k_ranges=[[0, 256], [256, 512]],
        total_q=512,
        total_k=512,
        range_merge=True,
        expect_direct_store=True,
        expect_fill_uncovered_rows=False,
    ),
    "overlapping_rows": FwdStoreCase(
        q_ranges=[[0, 256], [128, 512]],
        k_ranges=[[0, 256], [256, 512]],
        total_q=512,
        total_k=512,
        range_merge=False,
        expect_direct_store=False,
        expect_fill_uncovered_rows=False,
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


def _fold_env(
    backend: str, cat_gqa: bool, pack_factor: int, range_merge: bool = False
) -> dict[str, str]:
    """Environment variables that decide AttnArg's direct dK/dV grant and the
    cutedsl backward GQA fold; the legacy backend toggles stay off so that
    MAGI_ATTENTION_KERNEL_BACKEND alone selects the backend."""
    return {
        "MAGI_ATTENTION_KERNEL_BACKEND": backend,
        "MAGI_ATTENTION_SDPA_BACKEND": "0",
        "MAGI_ATTENTION_FA4_BACKEND": "0",
        "MAGI_ATTENTION_CATGQA": "1" if cat_gqa else "0",
        BWD_PACK_FACTOR_ENV: str(pack_factor),
        RANGE_MERGE_ENV: "1" if range_merge else "0",
    }


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


class TestCutedslFwdStorePath(TestCase):
    """cutedsl_fwd direct-stores O only for sorted, disjoint q rows.

    Rows outside every q range must hold zero O and the sink-only LSE, because
    the dist merge reads every row of the partial (out, lse).
    """

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("the fp32 range fwd exists only on SM100/SM110")

    @staticmethod
    def _make_inputs(
        case: FwdStoreCase, num_heads_q: int, num_heads_kv: int, num_sinks: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (q, k, v, sink) for the case."""
        gen = torch.Generator(device="cuda").manual_seed(1234)

        def randn(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
            return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

        q = randn(case.total_q, num_heads_q, HEAD_DIM)
        k = randn(case.total_k, num_heads_kv, HEAD_DIM)
        v = randn(case.total_k, num_heads_kv, HEAD_DIM)
        sink = randn(num_sinks, num_heads_q, dtype=torch.float32) if num_sinks else None
        return q, k, v, sink

    @staticmethod
    def _reference(
        case: FwdStoreCase, q, k, v, sink
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 (out, lse, max_logits) over the union of (q, k) pairs."""
        mask = torch.zeros(case.total_q, case.total_k, dtype=torch.bool, device="cuda")
        for (qs, qe), (ks, ke) in zip(case.q_ranges, case.k_ranges):
            mask[qs:qe, ks:ke] = True
        qhead_per_kvhead = q.shape[1] // k.shape[1]
        kf = k.float().repeat_interleave(qhead_per_kvhead, dim=1)
        vf = v.float().repeat_interleave(qhead_per_kvhead, dim=1)
        s = torch.einsum("qhd,khd->hqk", q.float(), kf) * HEAD_DIM**-0.5
        s = s.masked_fill(~mask[None], float("-inf"))
        max_logits = s.amax(dim=(1, 2))
        lse = torch.logsumexp(s, dim=-1)
        if sink is not None:
            lse = torch.logaddexp(lse, torch.logsumexp(sink, dim=0)[:, None])
        p = torch.exp(s - lse[..., None]).nan_to_num(0.0)
        out = torch.einsum("hqk,khd->qhd", p, vf)
        return out, lse.T, max_logits

    def _run_case(
        self,
        name: str,
        num_heads_q: int = NUM_HEADS,
        num_heads_kv: int = NUM_HEADS,
        num_sinks: int = 0,
    ) -> None:
        case = FWD_STORE_CASES[name]
        attn_arg = _make_attn_arg(case.q_ranges, case.k_ranges, case.range_merge)
        q, k, v, sink = self._make_inputs(case, num_heads_q, num_heads_kv, num_sinks)
        with mock.patch.object(
            cutedsl_ffa, "_flex_flash_attn_fwd", wraps=_flex_flash_attn_fwd
        ) as fwd:
            out, lse, max_logits = cutedsl_ffa.cutedsl_fwd(
                q,
                k,
                v,
                attn_arg,
                softmax_scale=None,
                softcap=0.0,
                sink=sink,
                return_max_logits=True,
            )

        kw = fwd.call_args.kwargs
        self.assertEqual(
            kw["disable_fwd_atomic_reduction"], case.expect_direct_store, name
        )
        self.assertEqual(
            kw["fill_uncovered_rows"], case.expect_fill_uncovered_rows, name
        )
        self.assertEqual(out.dtype, torch.float32, name)

        ref_out, ref_lse, ref_max_logits = self._reference(case, q, k, v, sink)
        for tensor_name, got, ref in (
            ("out", out, ref_out),
            ("lse", lse, ref_lse),
            ("max_logits", max_logits, ref_max_logits),
        ):
            torch.testing.assert_close(
                got,
                ref,
                atol=2e-2,
                rtol=0.0,
                msg=lambda m: f"{name} {tensor_name}: {m}",
            )

    @parameterize("name", list(FWD_STORE_CASES))
    def test_store_path(self, name: str):
        self._run_case(name)

    @parameterize("name", ["sorted_partition", "uncovered_rows"])
    def test_store_path_with_gqa_and_sink(self, name: str):
        self._run_case(name, num_heads_q=4, num_heads_kv=2, num_sinks=2)


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


class TestAttnArgFwdArgs(TestCase):
    """AttnArg's host range flags match the fwd range tensors it builds."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("the fwd range tensors are built on a CUDA device")

    @parameterize("range_merge", [False, True])
    @parameterize("rows", ROW_CASES)
    def test_range_flags_match_tensors(self, range_merge: bool, rows):
        q_rows, k_rows = rows
        attn_arg = _make_attn_arg(q_rows, k_rows, range_merge)
        ffa_args = attn_arg.to_ffa_args(is_bwd=False)
        q_ranges = AttnRanges.from_ranges(ffa_args["q_ranges"].tolist())
        msg = f"{range_merge=} {q_rows=} {k_rows=}"

        self.assertEqual(attn_arg.fwd_args_q_sorted, q_ranges.is_sorted(), msg)
        self.assertEqual(
            attn_arg.fwd_args_q_coverage_end,
            _end_if(q_ranges, q_ranges.is_full_coverage),
            msg,
        )


class TestBwdPackFactorEnv(TestCase):
    """Parsing of MAGI_ATTENTION_BWD_PACK_FACTOR."""

    def test_default_is_one(self):
        # The magi_attention.testing env helpers only set values, so removing
        # the variable needs mock.patch.dict, which restores it on exit.
        with mock.patch.dict(os.environ):
            os.environ.pop(BWD_PACK_FACTOR_ENV, None)
            self.assertEqual(env.general.bwd_pack_factor(), 1)

    def test_positive_integer(self):
        for value in ("1", "2", "4", "64"):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                self.assertEqual(env.general.bwd_pack_factor(), int(value), value)

    def test_invalid_value_raises(self):
        for value in ("0", "-2"):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                with self.assertRaisesRegex(ValueError, "positive integer", msg=value):
                    env.general.bwd_pack_factor()
        for value in ("four", "2.5", ""):
            with switch_env(BWD_PACK_FACTOR_ENV, value):
                with self.assertRaisesRegex(ValueError, "invalid literal", msg=value):
                    env.general.bwd_pack_factor()


class TestAttnArgDirectDkvGrant(TestCase):
    """AttnArg grants the direct dK/dV store (``disable_bwd_dkv_atomic_reduction``)
    to sorted, disjoint k ranges only under CatGQA, or under a bwd pack factor
    above 1 on the cutedsl backend, the only backend that reads
    MAGI_ATTENTION_BWD_PACK_FACTOR."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("the bwd range tensors are built on a CUDA device")

    @staticmethod
    def _grant(
        q_rows: list[list[int]],
        k_rows: list[list[int]],
        *,
        backend: str,
        cat_gqa: bool = False,
        pack_factor: int = 1,
        range_merge: bool = False,
    ) -> bool:
        with switch_env(_fold_env(backend, cat_gqa, pack_factor, range_merge)):
            attn_arg = AttnArg(
                q_ranges=AttnRanges.from_ranges(q_rows),
                k_ranges=AttnRanges.from_ranges(k_rows),
                attn_type_map=[0] * len(q_rows),
            )
        return attn_arg.disable_bwd_dkv_atomic_reduction

    @parameterize("backend", ["cutedsl", "ffa", "sdpa", "sdpa_ol", "fa4"])
    def test_grant_by_backend(self, backend: str):
        rows = [[0, 128], [128, 256]]
        self.assertFalse(self._grant(rows, rows, backend=backend), backend)
        self.assertTrue(self._grant(rows, rows, backend=backend, cat_gqa=True), backend)
        self.assertEqual(
            self._grant(rows, rows, backend=backend, pack_factor=4),
            backend == "cutedsl",
            backend,
        )

    def test_pack_factor_keeps_the_k_range_conditions(self):
        q_rows = [[0, 128], [128, 256]]
        for k_rows in ([[0, 128], [64, 192]], [[128, 256], [0, 128]]):
            self.assertFalse(
                self._grant(q_rows, k_rows, backend="cutedsl", pack_factor=4),
                str(k_rows),
            )
        self.assertFalse(
            self._grant(
                q_rows, q_rows, backend="cutedsl", pack_factor=4, range_merge=True
            )
        )


class TestCutedslBwdPackFactor(TestCase):
    """The dist cutedsl backward takes MAGI_ATTENTION_BWD_PACK_FACTOR, and the
    direct dK/dV store that calc_meta grants for it reaches the kernel."""

    NUM_HEADS_Q, NUM_HEADS_KV = 8, 2

    def setUp(self) -> None:
        if get_device_arch()[1] not in (10, 11):
            self.skipTest("bwd_pack_factor > 1 requires SM100/SM110")

    @staticmethod
    def _reference_grads(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        do: torch.Tensor,
        rows: list[list[int]],
    ) -> tuple[torch.Tensor, ...]:
        """fp32 GQA reference over the union of full (q, k) rectangles."""
        mask = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool, device="cuda")
        for start, end in rows:
            mask[start:end, start:end] = True
        group = q.shape[1] // k.shape[1]
        qf, kf, vf = (t.float().requires_grad_(True) for t in (q, k, v))
        kr, vr = (t.repeat_interleave(group, dim=1) for t in (kf, vf))
        s = torch.einsum("qhd,khd->hqk", qf, kr) * HEAD_DIM**-0.5
        p = torch.softmax(s.masked_fill(~mask[None], float("-inf")), -1)
        torch.einsum("hqk,khd->qhd", p.nan_to_num(0.0), vr).backward(do.float())
        return qf.grad, kf.grad, vf.grad

    def test_env_pack_factor_reaches_the_larkhaven(self):
        rows = [[0, 200], [200, 520]]
        total = rows[-1][1]
        gen = torch.Generator(device="cuda").manual_seed(1234)

        def randn(num_heads: int) -> torch.Tensor:
            return torch.randn(
                total,
                num_heads,
                HEAD_DIM,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )

        q, k, v, do = (
            randn(self.NUM_HEADS_Q),
            randn(self.NUM_HEADS_KV),
            randn(self.NUM_HEADS_KV),
            randn(self.NUM_HEADS_Q),
        )
        with switch_env(_fold_env("cutedsl", cat_gqa=False, pack_factor=4)):
            attn_arg = AttnArg(
                q_ranges=AttnRanges.from_ranges(rows),
                k_ranges=AttnRanges.from_ranges(rows),
                attn_type_map=[0] * len(rows),
            )
            out, lse, _ = cutedsl_ffa.cutedsl_fwd(
                q, k, v, attn_arg, softmax_scale=None, softcap=0.0
            )
            with record_bwd_kernels() as built, mock.patch.object(
                cutedsl_ffa, "_flex_flash_attn_bwd", wraps=_flex_flash_attn_bwd
            ) as bwd:
                dq, dk, dv, _ = cutedsl_ffa.cutedsl_bwd(
                    do, q, k, v, out, lse, attn_arg, softmax_scale=None, softcap=0.0
                )

        kw = bwd.call_args.kwargs
        self.assertEqual(kw["bwd_pack_factor"], 4)
        self.assertFalse(kw["cat_gqa"])
        self.assertTrue(kw["disable_bwd_dkv_atomic_reduction"])
        (bwd_kernel,) = built
        fold = bwd_kernel.gqa_fold
        self.assertEqual((fold.pack_factor, fold.head_loop), (4, False))
        self.assertFalse(bwd_kernel.dKV_postprocess)

        ref_grads = self._reference_grads(q, k, v, do, rows)
        for grad_name, got, ref in zip(("dq", "dk", "dv"), (dq, dk, dv), ref_grads):
            torch.testing.assert_close(
                got.float(),
                ref,
                atol=2e-2,
                rtol=2e-2,
                msg=lambda m: f"{grad_name}: {m}",
            )


if __name__ == "__main__":
    run_tests()
