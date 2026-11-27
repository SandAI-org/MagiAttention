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

import math
import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import plan_range_merge_bwd
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    CuteDSLDistTestBase,
    HeadConfig,
    Relations,
    assert_cuda_graph_matches_eager,
    assert_sequence_close,
    assert_sequence_equal,
    is_sm100_family,
    make_qkv,
    overlapping_relations,
    record_bwd_kernels,
    record_fwd_kernels,
    seed_test_case,
    sync_debug_error,
    to_ranges_tensor,
)
from magi_attention.testing.dist_common import with_run_in_mp
from magi_attention.testing.precision import assert_close, assert_equal
from magi_attention.testing.utils import switch_env
from magi_attention.utils.general import make_slice_mask_from_ffa_attn_type

_DTYPE = torch.bfloat16
_HEAD_DIM = 128
_NUM_HEAD_KV = 2
_TOL = 1e-2
_LSE_TOL = 1e-3
_GRAD_TOL = 3e-2
_ALL_MASKS = (MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal)
_MANY_RANGES = 212
_OUT_LSE_NAMES = ("out", "lse")
_GRAD_NAMES = ("dq", "dk", "dv")
_PUBLIC_API_NAMES = ("out", "lse", "dq", "dk", "dv", "dsink")


@dataclass(frozen=True)
class _FwdCase:
    group: int = 2
    out_dtype: torch.dtype = torch.float32
    pack_gqa: bool = False


@dataclass(frozen=True)
class _BwdCase:
    group: int = 1
    cat_gqa: bool = False
    two_cta: bool = False
    head_dim: int = _HEAD_DIM
    direct_dkv: bool = True
    range_merge: bool = False


@dataclass(frozen=True)
class _BwdProblem:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    do: torch.Tensor
    relations: Relations
    args: dict

    def tensors(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (self.q, self.k, self.v, self.out, self.lse, self.do)

    def run(self, **kwargs) -> tuple[torch.Tensor, ...]:
        return _flex_flash_attn_bwd(
            self.q,
            self.k,
            self.v,
            self.out,
            self.lse,
            self.do,
            **self.args,
            deterministic=True,
            **kwargs,
        )[:3]

    def ref_grads(self) -> tuple[torch.Tensor, ...]:
        return _ref_grads(self.q, self.k, self.v, self.do, self.relations)


def _with_2cta(enabled: bool) -> switch_env:
    return switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA", not enabled)


# ---------------------------------------------------------------------------
# Bitwise / closeness assertions, unrolled per tensor: no loops over test
# dimensions anywhere in this file.
# ---------------------------------------------------------------------------


def _assert_fwd_close(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    *,
    test_case: str = "",
) -> None:
    assert_close(
        out.float(),
        out_ref.float(),
        atol=_TOL,
        rtol=_TOL,
        test_case=f"{test_case} => out",
        print_rank=-1,
        print_no_mismatch=False,
    )
    # Rows attending nothing (or covered by no relation) carry lse -inf on
    # both sides; compare the finite pattern and the finite values instead of
    # the raw tensors, where -inf - -inf is nan.
    finite = torch.isfinite(lse_ref)
    assert_equal(
        torch.isfinite(lse),
        finite,
        test_case=f"{test_case} => lse_finite",
        print_rank=-1,
        print_no_mismatch=False,
    )
    assert_close(
        lse[finite],
        lse_ref[finite],
        atol=_LSE_TOL,
        rtol=_LSE_TOL,
        test_case=f"{test_case} => lse",
        print_rank=-1,
        print_no_mismatch=False,
    )


def _bitwise_across_margins(
    run: Callable[[int], Sequence[torch.Tensor]],
    margins: tuple[int, int, int],
    names: Sequence[str],
    test_case: str,
) -> Sequence[torch.Tensor]:
    """One kernel invocation per margin, straight-line: rerun at the base
    margin, plus the two other scheduler budgets, all bitwise identical."""
    base = run(margins[0])
    assert_sequence_equal(run(margins[0]), base, names, f"{test_case}[rerun]")
    assert_sequence_equal(
        run(margins[1]), base, names, f"{test_case}[margin={margins[1]}]"
    )
    assert_sequence_equal(
        run(margins[2]), base, names, f"{test_case}[margin={margins[2]}]"
    )
    return base


def _relations_mask(
    total_q: int, total_k: int, relations: Relations, device: int | str
) -> torch.Tensor:
    mask = torch.zeros((total_q, total_k), dtype=torch.bool, device=device)
    for (qs, qe), (ks, ke), mask_type in relations:
        if qe <= qs or ke <= ks:
            continue
        rel = make_slice_mask_from_ffa_attn_type(
            seqlen_q=qe - qs,
            seqlen_k=ke - ks,
            attn_type_idx=mask_type,
            device=device,
        )
        assert not (mask[qs:qe, ks:ke] & rel).any(), "relations share a pair"
        mask[qs:qe, ks:ke] |= rel
    return mask


def _ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    relations: Relations,
    sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = _relations_mask(q.shape[0], k.shape[0], relations, q.device)
    out, meta = ref_attn_func(
        q=q.detach().float(),
        k=k.detach().float(),
        v=v.detach().float(),
        mask=mask,
        sink=sink,
        layout="thd",
        backend="torch",
        high_precision=True,
        return_lse=True,
    )
    assert meta.lse is not None
    return out, meta.lse


def _ref_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    relations: Relations,
) -> tuple[torch.Tensor, ...]:
    mask = _relations_mask(q.shape[0], k.shape[0], relations, q.device)
    q_ref, k_ref, v_ref = (t.detach().float().requires_grad_() for t in (q, k, v))
    out, _ = ref_attn_func(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        mask=mask,
        layout="thd",
        backend="torch",
        high_precision=True,
    )
    return torch.autograd.grad(out, (q_ref, k_ref, v_ref), do.float())


# ---------------------------------------------------------------------------
# Relation fixtures (input data construction)
# ---------------------------------------------------------------------------


def _shared_k_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([5, 265], [7, 407], MT_MAP.full),
            ([105, 355], [600, 900], MT_MAP.full),
            ([305, 565], [37, 437], MT_MAP.full),
            ([405, 655], [613, 913], MT_MAP.causal),
            ([605, 865], [67, 467], MT_MAP.full),
            ([705, 955], [626, 926], MT_MAP.full),
            ([905, 1165], [97, 497], MT_MAP.full),
            ([1005, 1255], [639, 939], MT_MAP.causal),
        ],
        1260,
        944,
    )


def _merged_k_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([600, 700], [300, 557], MT_MAP.causal),
            ([5, 305], [0, 300], MT_MAP.causal),
            ([0, 143], [557, 700], MT_MAP.full),
            ([1, 301], [700, 1000], MT_MAP.causal),
            ([4, 304], [0, 300], MT_MAP.inv_causal),
            ([0, 200], [1000, 1000], MT_MAP.full),
            ([100, 357], [300, 557], MT_MAP.full),
            ([50, 50], [557, 700], MT_MAP.full),
            ([0, 300], [700, 1000], MT_MAP.inv_causal),
            ([400, 450], [0, 300], MT_MAP.full),
            ([250, 600], [557, 700], MT_MAP.inv_causal),
        ],
        700,
        1000,
    )


def _merged_k_relations_2cta_adjacent() -> tuple[Relations, int, int]:
    """Eight K groups of 160 rows after a full [0, 256) relation. Each group
    holds four 128-row pairs whose starts are 64 apart, so adjacent pairs of a
    group share a Q block; masks are inv_causal, bi_causal, bi_causal, causal."""
    masks = (MT_MAP.inv_causal, MT_MAP.bi_causal, MT_MAP.bi_causal, MT_MAP.causal)
    relations: Relations = [([0, 3072], [0, 256], MT_MAP.full)]
    for g in range(8):
        q0, k0 = g * 384, 256 + g * 160
        for p, mask in enumerate(masks):
            relations.append(([q0 + p * 64, q0 + p * 64 + 128], [k0, k0 + 160], mask))
    return relations, 3072, 1536


def _many_overlapping_relations() -> tuple[Relations, int, int]:
    """256 disjoint (qb, kb) rectangles across 16x16 unaligned bands so every
    physical Q and K block collects more than 32 writers."""
    bands = 16
    relations: Relations = []
    for g in range(bands * bands):
        qb = g % bands
        kb = (g // bands + qb) % bands
        relations.append(
            (
                [qb * 48 + 3, qb * 48 + 47],
                [kb * 48 + 5, kb * 48 + 49],
                _ALL_MASKS[g % len(_ALL_MASKS)],
            )
        )
    return relations, bands * 48 + 16, bands * 48 + 16


def _many_merged_k_relations(num_groups: int) -> tuple[Relations, int, int]:
    k_step, total_q = 32, 256
    relations: Relations = []
    for g in range(num_groups):
        k_range = [g * k_step, (g + 1) * k_step]
        q0 = (g * 17) % 128
        relations.append(([q0, q0 + 40], k_range, _ALL_MASKS[g % len(_ALL_MASKS)]))
        relations.append(([q0 + 64, q0 + 112], k_range, MT_MAP.causal))
    return relations, total_q, num_groups * k_step + 32


_DISJOINT_Q_RELATIONS: Relations = [
    ([3, 132], [0, 290], MT_MAP.full),
    ([140, 410], [290, 560], MT_MAP.causal),
    ([420, 420], [560, 620], MT_MAP.full),
    ([430, 690], [620, 620], MT_MAP.full),
    ([700, 965], [620, 900], MT_MAP.inv_causal),
    ([970, 1240], [900, 1170], MT_MAP.bi_causal),
]

_EMPTY_KEYLESS_RELATIONS: Relations = [
    ([1, 130], [0, 200], MT_MAP.full),
    ([50, 50], [200, 260], MT_MAP.full),
    ([100, 300], [260, 260], MT_MAP.full),
    ([127, 400], [260, 500], MT_MAP.causal),
    ([129, 130], [500, 520], MT_MAP.full),
    ([0, 257], [520, 900], MT_MAP.full),
]

_STRADDLING_RELATIONS: Relations = [
    ([0, 390], [10, 270], MT_MAP.bi_causal),
    ([0, 390], [270, 530], MT_MAP.bi_causal),
    ([400, 790], [270, 530], MT_MAP.bi_causal),
    ([400, 520], [530, 600], MT_MAP.bi_causal),
    ([7, 267], [600, 860], MT_MAP.causal),
    ([6, 266], [600, 860], MT_MAP.inv_causal),
    ([19, 535], [860, 1140], MT_MAP.full),
]

_GRAPH_DISJOINT_RELATIONS: Relations = [
    ([5, 270], [0, 310], MT_MAP.causal),
    ([280, 550], [120, 420], MT_MAP.inv_causal),
    ([560, 830], [200, 510], MT_MAP.full),
]


class _RangeDetHelpers:
    if TYPE_CHECKING:
        # A runtime definition would shadow the host class's `device` in the MRO.
        @property
        def device(self) -> int:
            ...

    @property
    def num_sm(self) -> int:
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def _margins(self) -> tuple[int, int, int]:
        """The full grid, 8 CTAs and 2 CTAs. Every fixture's grid bound exceeds
        8, so the three margins launch three grid sizes, and 2 CTAs hold one
        cluster for the 1-CTA and the 2-CTA kernels alike."""
        return (0, self.num_sm - 8, self.num_sm - 2)

    def _qkv(
        self,
        total_q: int,
        total_k: int,
        group: int,
        *,
        head_dim: int = _HEAD_DIM,
        num_head_kv: int = _NUM_HEAD_KV,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        heads = HeadConfig("det", num_head_kv * group, num_head_kv, head_dim)
        q, k, v = make_qkv((total_q,), (total_k,), heads, _DTYPE, self.device)
        return q.detach(), k.detach(), v.detach()

    def _range_args(self, relations: Relations) -> dict:
        q_ranges = [r[0] for r in relations]
        k_ranges = [r[1] for r in relations]
        return dict(
            q_ranges=to_ranges_tensor(q_ranges, self.device),
            k_ranges=to_ranges_tensor(k_ranges, self.device),
            mask_types=torch.tensor(
                [r[2] for r in relations], dtype=torch.int32, device=self.device
            ),
            max_seqlen_q=max(max(e - s for s, e in q_ranges), 1),
            max_seqlen_k=max(max(e - s for s, e in k_ranges), 1),
        )

    def _det_fwd(self, q, k, v, relations: Relations, **kwargs):
        return _flex_flash_attn_fwd(
            q,
            k,
            v,
            **self._range_args(relations),
            disable_fwd_atomic_reduction=False,
            deterministic=True,
            **kwargs,
        )

    def _bwd_problem(
        self,
        relations: Relations,
        total_q: int,
        total_k: int,
        group: int,
        *,
        head_dim: int = _HEAD_DIM,
        num_head_kv: int = _NUM_HEAD_KV,
        disable_fwd_atomic: bool = False,
    ) -> _BwdProblem:
        q, k, v = self._qkv(
            total_q, total_k, group, head_dim=head_dim, num_head_kv=num_head_kv
        )
        args = self._range_args(relations)
        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            **args,
            disable_fwd_atomic_reduction=disable_fwd_atomic,
            deterministic=True,
        )
        return _BwdProblem(
            q, k, v, out.to(q.dtype), lse, torch.randn_like(q), relations, args
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class _SingleGpuDetTestBase(_RangeDetHelpers, TestCase):
    @property
    def device(self) -> int:
        return torch.cuda.current_device()


class _SingleRankDetTestBase(_RangeDetHelpers, CuteDSLDistTestBase):
    """One rank: a test without parameter cases has nothing to spread over the
    GPUs, and its subprocess keeps the dist base's hang timeout."""

    @property
    def world_size(self) -> int:
        return 1


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicHost(_SingleGpuDetTestBase):
    def test_tile_counter_bound_counts_overlapping_ranges(self):
        seed_test_case("test_tile_counter_bound_counts_overlapping_ranges")
        # 2**24 ranges give 2**33 fwd and bwd tiles, four times the int32 tile
        # counter, so a tile twice as large still overflows it.
        num_ranges, rows = 2**24, 4096
        ranges = torch.tensor([0, rows], dtype=torch.int32, device=self.device).expand(
            num_ranges, 2
        )
        args = dict(
            q_ranges=ranges.contiguous(),
            k_ranges=ranges.contiguous(),
            mask_types=0,
            max_seqlen_q=rows,
            max_seqlen_k=rows,
        )
        q, k, v = self._qkv(rows, rows, group=1, num_head_kv=16)
        with self.assertRaises(ValueError):
            _flex_flash_attn_fwd(
                q, k, v, **args, disable_fwd_atomic_reduction=False, deterministic=True
            )
        q, k, v = self._qkv(rows, rows, group=1, num_head_kv=32)
        lse = torch.zeros(rows, q.shape[1], device=self.device)
        with _with_2cta(True), self.assertRaises(ValueError):
            _flex_flash_attn_bwd(
                q,
                k,
                v,
                q,
                lse,
                q,
                **args,
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )

    def test_debug_mode_rejects_max_seqlen_k_below_the_longest_k_range(self):
        seed_test_case("test_debug_mode_rejects_max_seqlen_k_below_longest_k")
        relations: Relations = [
            ([0, 100], [0, 300], MT_MAP.full),
            ([50, 200], [300, 400], MT_MAP.causal),
        ]
        q, k, v = self._qkv(200, 400, group=1)
        lse = torch.zeros(200, q.shape[1], device=self.device)
        args = self._range_args(relations)
        with switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", True):
            with self.assertRaises(ValueError):
                _flex_flash_attn_bwd(
                    q,
                    k,
                    v,
                    q,
                    lse,
                    q,
                    **{**args, "max_seqlen_k": 299},
                    disable_bwd_dkv_atomic_reduction=False,
                    deterministic=True,
                )

    def test_persistent_budget_validation(self):
        seed_test_case("test_persistent_budget_validation")
        relations, total_q, total_k = _shared_k_relations()
        prob = self._bwd_problem(relations, total_q, total_k, group=2)
        with _with_2cta(True), self.assertRaises(ValueError):
            prob.run(
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 1,
            )
        with self.assertRaises(ValueError):
            _flex_flash_attn_fwd(
                prob.q,
                prob.k,
                prob.v,
                **prob.args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                sm_margin=self.num_sm,
            )
        with _with_2cta(True):
            grads = prob.run(
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 2,
            )
        assert_sequence_close(
            grads, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, "budget_validation"
        )

    def test_deterministic_path_activation(self):
        """The single whitebox check of this suite: ``deterministic=True``
        reaches the fwd and the bwd range kernels, so the bitwise tests cannot
        pass on a path that ignores the flag."""
        seed_test_case("test_deterministic_path_activation")
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = self._qkv(total_q, total_k, group=2)
        with record_fwd_kernels() as built_fwd:
            self._det_fwd(q, k, v, relations)
        self.assertTrue(built_fwd[0].deterministic)
        prob = self._bwd_problem(relations, total_q, total_k, group=2)
        with record_bwd_kernels() as built_bwd:
            prob.run(disable_bwd_dkv_atomic_reduction=False)
        self.assertTrue(built_bwd[0].deterministic)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicFwd(_RangeDetHelpers, CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize(
        "case",
        [
            _FwdCase(group=2, out_dtype=torch.float32),
            _FwdCase(group=2, out_dtype=torch.bfloat16),
            _FwdCase(group=4, out_dtype=torch.float32, pack_gqa=True),
        ],
    )
    def test_fwd_bitwise_across_margins(self, case: _FwdCase):
        test_case = f"[fwd_bitwise][{case}]"
        seed_test_case(test_case)
        relations, total_q, total_k = overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = self._qkv(total_q, total_k, case.group)
        kwargs = {"out_dtype": case.out_dtype, "pack_gqa": case.pack_gqa}
        out, lse = _bitwise_across_margins(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m, **kwargs),
            self._margins(),
            _OUT_LSE_NAMES,
            test_case,
        )
        self.assertEqual(out.dtype, case.out_dtype)
        out_ref, lse_ref = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)

    @with_run_in_mp
    @parameterize("pack_gqa", [False, True])
    def test_fwd_empty_and_keyless_relations(self, pack_gqa: bool):
        test_case = f"[fwd_empty_keyless][pack={pack_gqa}]"
        seed_test_case(test_case)
        q, k, v = self._qkv(420, 900, group=2)
        out, lse = _bitwise_across_margins(
            lambda m: self._det_fwd(
                q, k, v, _EMPTY_KEYLESS_RELATIONS, sm_margin=m, pack_gqa=pack_gqa
            ),
            self._margins(),
            _OUT_LSE_NAMES,
            test_case,
        )
        out_ref, lse_ref = _ref_fwd(q, k, v, _EMPTY_KEYLESS_RELATIONS)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicFwdSingleRank(_SingleRankDetTestBase):
    @with_run_in_mp
    def test_fwd_caller_buffer_accumulation_with_sink(self):
        test_case = "[fwd_caller_buffer_sink]"
        seed_test_case(test_case)
        first_call: Relations = [
            ([0, 140], [0, 300], MT_MAP.full),
            ([70, 260], [300, 500], MT_MAP.causal),
        ]
        second_call: Relations = [
            ([3, 250], [500, 800], MT_MAP.full),
            ([128, 129], [800, 820], MT_MAP.full),
        ]
        q, k, v = self._qkv(260, 820, group=2)
        num_head = q.shape[1]
        sink = torch.randn(2, num_head, device=self.device, dtype=torch.float32)

        def run(sm_margin: int):
            out = torch.zeros(260, num_head, _HEAD_DIM, device=self.device)
            lse = torch.full((260, num_head), -math.inf, device=self.device)
            self._det_fwd(q, k, v, first_call, out=out, lse=lse, sm_margin=sm_margin)
            self._det_fwd(
                q, k, v, second_call, out=out, lse=lse, sink=sink, sm_margin=sm_margin
            )
            return out, lse

        out, lse = _bitwise_across_margins(
            run, self._margins(), _OUT_LSE_NAMES, test_case
        )
        out_ref, lse_ref = _ref_fwd(q, k, v, first_call + second_call, sink)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)

    @with_run_in_mp
    def test_fwd_many_overlapping_ranges_bitwise(self):
        test_case = "[fwd_many_overlapping_ranges]"
        seed_test_case(test_case)
        relations, total_q, total_k = _many_overlapping_relations()
        q, k, v = self._qkv(total_q, total_k, group=1, num_head_kv=1)
        out, lse = _bitwise_across_margins(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m),
            self._margins(),
            _OUT_LSE_NAMES,
            test_case,
        )
        out_ref, lse_ref = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicBwd(_RangeDetHelpers, CuteDSLDistTestBase):
    def _check_bwd_bitwise_and_close(
        self,
        case: _BwdCase,
        relations: Relations,
        total_q: int,
        total_k: int,
        test_case: str,
    ) -> Sequence[torch.Tensor]:
        prob = self._bwd_problem(
            relations, total_q, total_k, case.group, head_dim=case.head_dim
        )
        kwargs = dict(
            disable_bwd_dkv_atomic_reduction=case.direct_dkv,
            cat_gqa=case.cat_gqa,
            range_merge=case.range_merge,
        )
        with _with_2cta(case.two_cta):
            first = _bitwise_across_margins(
                lambda m: prob.run(sm_margin=m, **kwargs),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )
        return first

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=False),
            _BwdCase(group=2, cat_gqa=True, two_cta=True),
        ],
    )
    def test_bwd_dq_bitwise_across_margins(self, case: _BwdCase):
        test_case = f"[bwd_dq_bitwise][{case}]"
        seed_test_case(test_case)
        relations, total_q, total_k = overlapping_relations(
            [129, 300, 1000, 77], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        self._check_bwd_bitwise_and_close(case, relations, total_q, total_k, test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=False, direct_dkv=False),
            _BwdCase(group=2, two_cta=False, direct_dkv=False),
            _BwdCase(group=2, cat_gqa=True, two_cta=True, direct_dkv=False),
        ],
    )
    def test_bwd_reduced_dkv_bitwise_across_margins(self, case: _BwdCase):
        test_case = f"[bwd_reduced_dkv_bitwise][{case}]"
        seed_test_case(test_case)
        self._check_bwd_bitwise_and_close(case, *_shared_k_relations(), test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=False, direct_dkv=False),
            _BwdCase(group=2, cat_gqa=True, two_cta=True, direct_dkv=False),
        ],
    )
    def test_bwd_disjoint_q_dense_dqacc_bitwise_across_margins(self, case: _BwdCase):
        test_case = f"[bwd_disjoint_dense_dqacc][{case}]"
        seed_test_case(test_case)
        prob = self._bwd_problem(
            _DISJOINT_Q_RELATIONS,
            1260,
            1180,
            case.group,
            head_dim=case.head_dim,
            disable_fwd_atomic=True,
        )
        with _with_2cta(case.two_cta):
            first = _bitwise_across_margins(
                lambda m: prob.run(
                    sm_margin=m,
                    disable_fwd_atomic_reduction=True,
                    disable_bwd_dkv_atomic_reduction=case.direct_dkv,
                    cat_gqa=case.cat_gqa,
                ),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=2, cat_gqa=True, two_cta=False, head_dim=128),
            _BwdCase(group=2, cat_gqa=True, two_cta=True, head_dim=128),
            _BwdCase(group=2, cat_gqa=True, two_cta=False, head_dim=64),
        ],
    )
    def test_bwd_range_merge_bitwise_across_margins(self, case: _BwdCase):
        test_case = f"[bwd_range_merge][{case}]"
        seed_test_case(test_case)
        prob = self._bwd_problem(
            *_merged_k_relations(), case.group, head_dim=case.head_dim
        )
        plan = plan_range_merge_bwd(
            prob.args["q_ranges"], prob.args["k_ranges"], prob.args["mask_types"]
        )
        with _with_2cta(case.two_cta):
            first = _bitwise_across_margins(
                lambda m: prob.run(
                    sm_margin=m,
                    range_merge=plan,
                    cat_gqa=case.cat_gqa,
                    disable_bwd_dkv_atomic_reduction=True,
                ),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=2, cat_gqa=True, two_cta=False, direct_dkv=False),
            _BwdCase(
                group=2, cat_gqa=True, two_cta=True, direct_dkv=True, range_merge=True
            ),
        ],
    )
    def test_bwd_straddling_masked_blocks_all_mask_types(self, case: _BwdCase):
        test_case = f"[bwd_straddling_masked][merge={case.range_merge}]"
        seed_test_case(test_case)
        self._check_bwd_bitwise_and_close(
            case, _STRADDLING_RELATIONS, 800, 1150, test_case
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicBwdSingleRank(_SingleRankDetTestBase):
    @with_run_in_mp
    def test_bwd_empty_and_keyless_relations(self):
        test_case = "[bwd_empty_keyless]"
        seed_test_case(test_case)
        prob = self._bwd_problem(_EMPTY_KEYLESS_RELATIONS, 420, 900, group=1)
        first = _bitwise_across_margins(
            lambda m: prob.run(sm_margin=m),
            self._margins(),
            _GRAD_NAMES,
            test_case,
        )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    def test_bwd_disjoint_dq_caller_buffer_accumulation(self):
        test_case = "[bwd_disjoint_dq_acc]"
        seed_test_case(test_case)
        prob = self._bwd_problem(
            _DISJOINT_Q_RELATIONS, 1260, 1180, 2, disable_fwd_atomic=True
        )
        dq_base = torch.randn(prob.q.shape, device=self.device, dtype=torch.float32)
        with _with_2cta(True):
            first = prob.run(
                sm_margin=0, disable_fwd_atomic_reduction=True, cat_gqa=True
            )
            dq_acc_0 = prob.run(
                sm_margin=0,
                disable_fwd_atomic_reduction=True,
                cat_gqa=True,
                dq=dq_base.clone(),
            )[0]
            dq_acc_1 = prob.run(
                sm_margin=self.num_sm - 2,
                disable_fwd_atomic_reduction=True,
                cat_gqa=True,
                dq=dq_base.clone(),
            )[0]
        assert_equal(
            dq_acc_0,
            dq_acc_1,
            test_case=f"{test_case}[across_margins]",
            print_no_mismatch=False,
        )
        assert_close(
            dq_acc_0,
            dq_base + first[0],
            atol=1e-4,
            rtol=1e-4,
            test_case=f"{test_case}[sum]",
            print_rank=-1,
            print_no_mismatch=False,
        )

    @with_run_in_mp
    def test_bwd_range_merge_2cta_adjacent_pair_conflicts_bitwise(self):
        test_case = "[bwd_2cta_adjacent_conflicts]"
        seed_test_case(test_case)
        prob = self._bwd_problem(*_merged_k_relations_2cta_adjacent(), group=2)
        with _with_2cta(True):
            first = _bitwise_across_margins(
                lambda m: prob.run(
                    sm_margin=m,
                    range_merge=True,
                    cat_gqa=True,
                    disable_bwd_dkv_atomic_reduction=True,
                ),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    def test_bwd_many_overlapping_ranges_bitwise(self):
        test_case = "[bwd_many_overlapping]"
        seed_test_case(test_case)
        prob = self._bwd_problem(
            *_many_overlapping_relations(),
            group=1,
            num_head_kv=1,
            disable_fwd_atomic=True,
        )
        with _with_2cta(True):
            first = _bitwise_across_margins(
                lambda m: prob.run(sm_margin=m, disable_bwd_dkv_atomic_reduction=False),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    def test_bwd_many_merged_k_ranges_bitwise(self):
        test_case = "[bwd_many_merged_k]"
        seed_test_case(test_case)
        prob = self._bwd_problem(
            *_many_merged_k_relations(_MANY_RANGES), group=1, num_head_kv=1
        )
        plan = plan_range_merge_bwd(
            prob.args["q_ranges"], prob.args["k_ranges"], prob.args["mask_types"]
        )
        first = _bitwise_across_margins(
            lambda m: prob.run(
                sm_margin=m,
                range_merge=plan,
                disable_bwd_dkv_atomic_reduction=True,
            ),
            self._margins(),
            _GRAD_NAMES,
            test_case,
        )
        assert_sequence_close(
            first, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    def test_bwd_empty_ranges_single_cluster(self):
        test_case = "[bwd_empty_single_cluster]"
        seed_test_case(test_case)
        relations: Relations = (
            [([0, 128], [0, 200], MT_MAP.full)]
            + [([150, 150], [200, 200], MT_MAP.full)] * 61
            + [([160, 288], [220, 420], MT_MAP.causal)]
        )
        prob = self._bwd_problem(relations, 300, 440, group=1, num_head_kv=1)
        # num_sm - 1 leaves a single CTA, which only holds one cluster when
        # the launch is single-cluster.
        with _with_2cta(False):
            grads = prob.run(
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 1,
            )
        assert_sequence_close(
            grads, prob.ref_grads(), _GRAD_NAMES, _GRAD_TOL, test_case
        )

    @with_run_in_mp
    def test_bwd_reads_no_range_back_outside_debug_mode(self):
        seed_test_case("[bwd_no_range_sync]")
        # Built up front: copying host range lists to the device syncs.
        ordered = self._bwd_problem(
            *overlapping_relations([129, 300], coverage=2, mask_types=[MT_MAP.full]),
            group=1,
        )
        disjoint = self._bwd_problem(
            [
                ([3, 140], [0, 300], MT_MAP.full),
                ([145, 420], [300, 620], MT_MAP.causal),
            ],
            430,
            620,
            group=1,
        )

        def run_all() -> None:
            ordered.run(disable_bwd_dkv_atomic_reduction=False)
            disjoint.run(
                disable_bwd_dkv_atomic_reduction=True,
                disable_fwd_atomic_reduction=True,
            )

        with switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", False):
            run_all()
            torch.cuda.synchronize()
            with sync_debug_error():
                run_all()

    @with_run_in_mp
    def test_public_api_fwd_bwd_bitwise_with_sink(self):
        test_case = "[public_api_fwd_bwd_sink]"
        seed_test_case(test_case)
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = self._qkv(total_q, total_k, group=2)
        sink = torch.randn(2, q.shape[1], device=self.device, dtype=torch.float32)
        do = torch.randn_like(q)

        def run(sm_margin: int):
            q_in = q.detach().clone().requires_grad_()
            k_in = k.detach().clone().requires_grad_()
            v_in = v.detach().clone().requires_grad_()
            sink_in = sink.detach().clone().requires_grad_()
            out, meta = flex_flash_attn_func(
                q_in,
                k_in,
                v_in,
                **self._range_args(relations),
                sink=sink_in,
                deterministic=True,
                sm_margin=sm_margin,
            )
            out.backward(do)
            return (
                out.detach(),
                meta.lse,
                q_in.grad,
                k_in.grad,
                v_in.grad,
                sink_in.grad,
            )

        with _with_2cta(True):
            _bitwise_across_margins(run, self._margins(), _PUBLIC_API_NAMES, test_case)

    @with_run_in_mp
    def test_cuda_graph_replay_packed_fwd_bitwise(self):
        test_case = "[cuda_graph][packed_fwd]"
        seed_test_case(test_case)
        relations, total_q, total_k = overlapping_relations(
            [129, 300], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        args = self._range_args(relations)

        def fn(q, k, v):
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                pack_gqa=True,
            )

        data0 = self._qkv(total_q, total_k, group=4)
        data1 = self._qkv(total_q, total_k, group=4)
        assert_cuda_graph_matches_eager(
            fn,
            data0,
            [data1, data0, data1],
            test_case=test_case,
        )

    @with_run_in_mp
    def test_cuda_graph_replay_bwd_disjoint_dense_dqacc_bitwise(self):
        test_case = "[cuda_graph][bwd_disjoint_dense_dqacc]"
        seed_test_case(test_case)
        args = self._range_args(_GRAPH_DISJOINT_RELATIONS)

        def fn(*tensors):
            return _flex_flash_attn_bwd(
                *tensors,
                **args,
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
                disable_fwd_atomic_reduction=True,
                cat_gqa=True,
            )[:3]

        prob0 = self._bwd_problem(
            _GRAPH_DISJOINT_RELATIONS, 850, 520, group=2, disable_fwd_atomic=True
        )
        prob1 = self._bwd_problem(
            _GRAPH_DISJOINT_RELATIONS, 850, 520, group=2, disable_fwd_atomic=True
        )
        with _with_2cta(True):
            assert_cuda_graph_matches_eager(
                fn,
                prob0.tensors(),
                [prob1.tensors(), prob0.tensors(), prob1.tensors()],
                test_case=test_case,
            )


if __name__ == "__main__":
    run_tests()
