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
import os
import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence
from unittest import TestCase, mock

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention.kernel.cutedsl import flex_flash_attn_func
from magi_attention.kernel.cutedsl.cache_utils import JITCache
from magi_attention.kernel.cutedsl.ffa_bwd_sm100 import FFABwdSm100
from magi_attention.kernel.cutedsl.ffa_utils import MT_MAP
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import (
    plan_range_merge,
    plan_range_merge_bwd,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.cutedsl_probes import (
    DelayPoint,
    delay_injection,
    record_visits,
)
from magi_attention.testing.cutedsl_utils import (
    SM100_FAMILY_REQUIRED,
    CuteDSLDistTestBase,
    HeadConfig,
    is_sm100_family,
    make_qkv,
    record_bwd_kernels,
    record_fwd_kernels,
    seed_test_case,
    to_ranges_tensor,
)
from magi_attention.testing.dist_common import with_run_in_mp
from magi_attention.testing.precision import assert_close, assert_equal
from magi_attention.testing.utils import switch_env
from magi_attention.utils.general import make_slice_mask_from_ffa_attn_type

_DTYPE = torch.bfloat16
_HEAD_DIM = 128
_DET_HEAD_DIMS = [64, 128]
_NUM_HEAD_KV = 2
_TOL = 1e-2
_LSE_TOL = 1e-3
_GRAD_TOL = 3e-2
_ALL_MASKS = (MT_MAP.full, MT_MAP.causal, MT_MAP.inv_causal, MT_MAP.bi_causal)
_STRESS_REPEATS = int(os.environ.get("MAGI_ATTENTION_TEST_2CTA_STRESS_REPEATS", "0"))

Relations = list[tuple[list[int], list[int], int]]


@dataclass(frozen=True)
class _FwdCase:
    out_dtype: torch.dtype = torch.float32
    group: int = 1
    tol: float = _TOL


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

    def run(self, *, direct_dkv: bool = True, **kwargs) -> tuple[torch.Tensor, ...]:
        return _flex_flash_attn_bwd(
            self.q,
            self.k,
            self.v,
            self.out,
            self.lse,
            self.do,
            **self.args,
            disable_bwd_dkv_atomic_reduction=direct_dkv,
            deterministic=True,
            **kwargs,
        )[:3]

    def ref_grads(self) -> tuple[torch.Tensor, ...]:
        return _ref_grads(self.q, self.k, self.v, self.do, self.relations)


# ─────────────────────────────────────────────────────────────────────
# Assertions, environment helpers, and references
# ─────────────────────────────────────────────────────────────────────


def _with_2cta(enabled: bool) -> switch_env:
    return switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA", not enabled)


def _assert_bitwise_equal(
    got: Sequence[torch.Tensor], want: Sequence[torch.Tensor], test_case: str = ""
) -> None:
    for idx, (g, w) in enumerate(zip(got, want, strict=True)):
        assert_equal(
            g,
            w,
            test_case=f"{test_case}[{idx}]" if test_case else f"tensor[{idx}]",
            print_rank=-1,
            print_no_mismatch=False,
        )


def _assert_fwd_close(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    *,
    tol: float = _TOL,
    lse_tol: float = _LSE_TOL,
    check_finite: bool = False,
    test_case: str = "",
) -> None:
    assert_close(
        out.float(),
        out_ref.float(),
        atol=tol,
        rtol=tol,
        test_case=f"{test_case} => out",
        print_rank=-1,
        print_no_mismatch=False,
    )
    if check_finite:
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
            atol=lse_tol,
            rtol=lse_tol,
            test_case=f"{test_case} => lse",
            print_rank=-1,
            print_no_mismatch=False,
        )
    else:
        assert_close(
            lse,
            lse_ref,
            atol=lse_tol,
            rtol=lse_tol,
            test_case=f"{test_case} => lse",
            print_rank=-1,
            print_no_mismatch=False,
        )


def _assert_grads_close(
    grads: Sequence[torch.Tensor], want: Sequence[torch.Tensor], test_case: str = ""
) -> None:
    for name, got, ref in zip(("dq", "dk", "dv"), grads, want, strict=True):
        assert_close(
            got.float(),
            ref.float(),
            atol=_GRAD_TOL,
            rtol=_GRAD_TOL,
            test_case=f"{test_case} => {name}",
            print_rank=-1,
            print_no_mismatch=False,
        )


def _assert_bitwise_across_margins(
    run: Callable[[int], Sequence[torch.Tensor]],
    margins: Sequence[int],
    test_case: str = "",
) -> Sequence[torch.Tensor]:
    first = run(margins[0])
    for margin in margins[1:]:
        _assert_bitwise_equal(run(margin), first, f"{test_case}[margin={margin}]")
    return first


def _assert_cuda_graph_matches_eager(
    fn: Callable[..., Sequence[torch.Tensor]],
    data_0: Sequence[torch.Tensor],
    data_1: Sequence[torch.Tensor],
    test_case: str = "",
) -> None:
    eager_0 = fn(*data_0)
    eager_1 = fn(*data_1)
    static = [t.clone() for t in data_0]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(*static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn(*static)
    for step, (inputs, want) in enumerate(
        ((data_1, eager_1), (data_0, eager_0), (data_1, eager_1))
    ):
        for dst, src in zip(static, inputs, strict=True):
            dst.copy_(src)
        graph.replay()
        _assert_bitwise_equal(static_out, want, f"{test_case}[replay={step}]")


_torch_empty_like = torch.empty_like


def _nan_empty_like(*args, **kwargs) -> torch.Tensor:
    t = _torch_empty_like(*args, **kwargs)
    return t.fill_(float("nan")) if t.is_floating_point() else t


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
    *,
    return_max_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        return_max_logits=return_max_logits,
    )
    assert meta.lse is not None
    return out, meta.lse, meta.max_logits


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


# ─────────────────────────────────────────────────────────────────────
# Relation fixtures
# ─────────────────────────────────────────────────────────────────────


def _overlapping_relations(
    len_qs: list[int], coverage: int, mask_types: Sequence[int]
) -> tuple[Relations, int, int]:
    relations: Relations = []
    q_cursor, k_cursor, i = 0, 0, 0
    for len_q in len_qs:
        q_start = q_cursor + 3
        for c in range(coverage):
            len_k = len_q + 37
            relations.append(
                (
                    [q_start + c, q_start + c + len_q],
                    [k_cursor, k_cursor + len_k],
                    mask_types[i % len(mask_types)],
                )
            )
            k_cursor += len_k
            i += 1
        q_cursor = q_start + coverage + len_q
    return relations, q_cursor + 5, k_cursor


def _shared_k_relations() -> tuple[Relations, int, int]:
    relations: Relations = []
    for i in range(4):
        relations.append(
            ([5 + 300 * i, 265 + 300 * i], [7 + 30 * i, 407 + 30 * i], MT_MAP.full)
        )
        relations.append(
            (
                [105 + 300 * i, 355 + 300 * i],
                [600 + 13 * i, 900 + 13 * i],
                MT_MAP.causal if i % 2 else MT_MAP.full,
            )
        )
    return relations, 1260, 944


def _merged_q_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([131, 600], [100, 400], MT_MAP.full),
            ([0, 131], [0, 300], MT_MAP.full),
            ([601, 1000], [900, 1200], MT_MAP.causal),
            ([0, 131], [300, 500], MT_MAP.causal),
            ([600, 601], [0, 50], MT_MAP.full),
            ([131, 600], [600, 900], MT_MAP.inv_causal),
            ([0, 131], [530, 530], MT_MAP.full),
            ([0, 131], [500, 530], MT_MAP.full),
        ],
        1000,
        1200,
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


def _merged_k_relations_heavy_middle() -> tuple[Relations, int, int]:
    relations, total_q, _ = _merged_k_relations()
    relations = [
        (q_range, [580, 700] if k_range == [557, 700] else k_range, mask_type)
        for q_range, k_range, mask_type in relations
    ] + [
        ([q_start, q_end], [300, 557], MT_MAP.full)
        for q_start, q_end in ((0, 50), (50, 100), (357, 480), (480, 600))
    ]
    return relations, total_q, 1100


def _merged_k_relations_2cta_adjacent(
    num_groups: int = 8, pairs_per_group: int = 4
) -> tuple[Relations, int, int]:
    len_k, base_k = 160, 256
    span = 64 * pairs_per_group + 128
    total_q = span * num_groups
    relations: Relations = [([0, total_q], [0, base_k], MT_MAP.full)]
    for g in range(num_groups):
        k_range = [base_k + len_k * g, base_k + len_k * (g + 1)]
        for p in range(pairs_per_group):
            q_start = span * g + 64 * p
            if pairs_per_group == 1:
                mask_type = MT_MAP.full
            elif p == 0:
                mask_type = MT_MAP.inv_causal
            elif p == pairs_per_group - 1:
                mask_type = MT_MAP.causal
            else:
                mask_type = MT_MAP.bi_causal
            relations.append(([q_start, q_start + 128], list(k_range), mask_type))
    return relations, total_q, base_k + len_k * num_groups


def _many_overlapping_q_relations(num_ranges: int) -> tuple[Relations, int, int]:
    q_len, k_step, total_q = 48, 32, 256
    relations: Relations = [
        (
            [(g * 19) % (total_q - q_len), (g * 19) % (total_q - q_len) + q_len],
            [g * k_step, (g + 1) * k_step],
            _ALL_MASKS[g % len(_ALL_MASKS)],
        )
        for g in range(num_ranges)
    ]
    return relations, total_q, num_ranges * k_step + 32


def _many_merged_k_relations(num_groups: int) -> tuple[Relations, int, int]:
    k_step, total_q = 32, 256
    relations: Relations = []
    for g in range(num_groups):
        k_range = [g * k_step, (g + 1) * k_step]
        q0 = (g * 17) % 128
        relations.append(([q0, q0 + 40], k_range, _ALL_MASKS[g % len(_ALL_MASKS)]))
        relations.append(([q0 + 64, q0 + 112], k_range, MT_MAP.causal))
    return relations, total_q, num_groups * k_step + 32


def _many_overlapping_k_relations(num_ranges: int) -> tuple[Relations, int, int]:
    q_len, k_len, total_k = 16, 48, 256
    relations: Relations = [
        (
            [g * q_len, (g + 1) * q_len],
            [(g * 13) % (total_k - k_len), (g * 13) % (total_k - k_len) + k_len],
            _ALL_MASKS[g % len(_ALL_MASKS)],
        )
        for g in range(num_ranges)
    ]
    return relations, num_ranges * q_len, total_k


_DISJOINT_Q_RELATIONS: dict[bool, tuple[Relations, int]] = {
    True: (
        [
            ([3, 132], [0, 290], MT_MAP.full),
            ([140, 410], [290, 560], MT_MAP.causal),
            ([420, 420], [560, 620], MT_MAP.full),
            ([430, 690], [620, 620], MT_MAP.full),
            ([700, 965], [620, 900], MT_MAP.inv_causal),
            ([970, 1240], [900, 1170], MT_MAP.bi_causal),
        ],
        1180,
    ),
    False: (
        [
            ([3, 132], [10, 300], MT_MAP.full),
            ([140, 410], [150, 430], MT_MAP.causal),
            ([420, 420], [200, 260], MT_MAP.full),
            ([430, 690], [430, 430], MT_MAP.full),
            ([700, 965], [100, 380], MT_MAP.inv_causal),
            ([970, 1240], [250, 520], MT_MAP.bi_causal),
        ],
        550,
    ),
}

_ONE_INDEX_ROW_RELATIONS: Relations = [([0, 64], [0, 64], MT_MAP.full)]

_BWD_UNIT_RELATIONS: dict[int, Relations] = {
    1: [([0, 300], [0, 200], MT_MAP.full)],
    5: [([0, 300], [0, 768], MT_MAP.full), ([100, 400], [800, 1300], MT_MAP.causal)],
}


# ─────────────────────────────────────────────────────────────────────
# Shared test helpers
# ─────────────────────────────────────────────────────────────────────


class _RangeDetHelpers:
    if TYPE_CHECKING:

        @property
        def device(self) -> int:
            ...

    @property
    def num_sm(self) -> int:
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def _fwd_margins(self) -> tuple[int, ...]:
        return (0, 1, self.num_sm - 7, self.num_sm - 1, 0)

    def _bwd_margins(self, cluster: int, mid: int = 4) -> tuple[int, ...]:
        return (0, 1, self.num_sm - mid * cluster, self.num_sm - cluster, 0)

    def _many_ranges(self) -> int:
        return self.num_sm + 64

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


# ─────────────────────────────────────────────────────────────────────
# Host validation and persistent SM grid budgets
# ─────────────────────────────────────────────────────────────────────


class TestFFARangeDeterministicHost(_SingleGpuDetTestBase):
    def test_tile_counter_bound_counts_overlapping_ranges(self):
        seed_test_case("test_tile_counter_bound_counts_overlapping_ranges")
        num_ranges, rows = 2**22, 4096
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
        with self.assertRaisesRegex(ValueError, "tile counter"):
            _flex_flash_attn_fwd(
                q, k, v, **args, disable_fwd_atomic_reduction=False, deterministic=True
            )
        q, k, v = self._qkv(rows, rows, group=1, num_head_kv=32)
        lse = torch.zeros(rows, q.shape[1], device=self.device)
        with _with_2cta(True), self.assertRaisesRegex(ValueError, "tile counter"):
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
        self.assertEqual(args["max_seqlen_k"], 300)

        class KernelReached(Exception):
            pass

        def bwd(max_seqlen_k: int) -> None:
            _flex_flash_attn_bwd(
                q,
                k,
                v,
                q,
                lse,
                q,
                **{**args, "max_seqlen_k": max_seqlen_k},
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )

        with switch_env(
            "MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", True
        ), mock.patch.object(
            _flex_flash_attn_bwd, "compile_cache", JITCache()
        ), mock.patch.object(
            FFABwdSm100, "__init__", side_effect=KernelReached
        ):
            with self.assertRaisesRegex(
                ValueError, r"max_seqlen_k=299 < longest k range 300"
            ):
                bwd(299)
            with self.assertRaises(KernelReached):
                bwd(300)


class TestFFARangePersistentGrid(_SingleGpuDetTestBase):
    def _budget_error(self, sm_margin: int, cluster_size: int):
        return self.assertRaisesRegex(
            ValueError,
            rf"\({self.num_sm} - {sm_margin}\).*cluster_size={cluster_size}",
        )

    def _bwd_2cta(self, prob: _BwdProblem, sm_margin: int):
        with _with_2cta(True):
            return _flex_flash_attn_bwd(
                *prob.tensors(),
                **prob.args,
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=sm_margin,
            )[:3]

    def test_bwd_2cta_rejects_a_budget_below_one_cluster(self):
        seed_test_case("test_bwd_2cta_rejects_a_budget_below_one_cluster")
        prob = self._bwd_problem(*_shared_k_relations(), group=2)
        for margin in (self.num_sm - 1, self.num_sm):
            with record_bwd_kernels() as built, self._budget_error(margin, 2):
                self._bwd_2cta(prob, sm_margin=margin)
            self.assertEqual(built, [])

    def test_bwd_2cta_runs_on_one_cluster(self):
        seed_test_case("test_bwd_2cta_runs_on_one_cluster")
        prob = self._bwd_problem(*_shared_k_relations(), group=2)
        with record_bwd_kernels() as built:
            grads = self._bwd_2cta(prob, sm_margin=self.num_sm - 2)
        (kernel,) = built
        self.assertTrue(kernel.use_2cta_instrs and kernel.is_persistent)
        _assert_grads_close(grads, prob.ref_grads(), "bwd_2cta_one_cluster")

    def test_fwd_rejects_a_budget_below_one_cluster(self):
        seed_test_case("test_fwd_rejects_a_budget_below_one_cluster")
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = self._qkv(total_q, total_k, group=2)
        with record_fwd_kernels() as built, self._budget_error(self.num_sm, 1):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                **self._range_args(relations),
                disable_fwd_atomic_reduction=False,
                sm_margin=self.num_sm,
            )
        self.assertEqual(built, [])

        q, k, v = self._qkv(1024, 1024, group=1)
        ranges = to_ranges_tensor([[0, 512], [512, 1024]], self.device)
        with _with_2cta(True), record_fwd_kernels() as built, self._budget_error(
            self.num_sm - 1, 2
        ):
            _flex_flash_attn_fwd(
                q,
                k,
                v,
                q_ranges=ranges,
                k_ranges=ranges,
                mask_types=MT_MAP.full,
                max_seqlen_q=512,
                max_seqlen_k=512,
                disable_fwd_atomic_reduction=True,
                sm_margin=self.num_sm - 1,
            )
        self.assertEqual(built, [])


# ─────────────────────────────────────────────────────────────────────
# Forward deterministic range tests
# ─────────────────────────────────────────────────────────────────────


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicFwd(_RangeDetHelpers, CuteDSLDistTestBase):
    def _assert_bitwise_packed(
        self, q, k, v, relations: Relations, test_case: str
    ) -> Sequence[torch.Tensor]:
        with record_fwd_kernels() as built:
            result = _assert_bitwise_across_margins(
                lambda m: self._det_fwd(q, k, v, relations, pack_gqa=True, sm_margin=m),
                self._fwd_margins(),
                test_case,
            )
        kernel = built[0]
        self.assertTrue(kernel.deterministic and kernel.pack_gqa)
        self.assertEqual(kernel.qhead_per_kvhead, q.shape[1] // k.shape[1])
        return result

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _FwdCase(torch.float32, group=1),
            _FwdCase(torch.float32, group=2),
            _FwdCase(torch.bfloat16, group=2),
        ],
    )
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_bitwise_across_runs_and_sm_margin(self, case: _FwdCase, head_dim: int):
        test_case = f"[fwd_bitwise][{case}][hd={head_dim}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = self._qkv(total_q, total_k, case.group, head_dim=head_dim)
        with record_fwd_kernels() as built:
            out, lse = _assert_bitwise_across_margins(
                lambda m: self._det_fwd(
                    q, k, v, relations, out_dtype=case.out_dtype, sm_margin=m
                ),
                self._fwd_margins(),
                test_case,
            )
        kernel = built[0]
        self.assertTrue(kernel.deterministic and kernel.is_persistent)
        self.assertEqual(kernel.cluster_shape_mn, (1, 1))
        self.assertEqual(out.dtype, case.out_dtype)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)

    @with_run_in_mp
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_empty_keyless_and_protection_block_relations(self, head_dim: int):
        test_case = f"[fwd_empty_keyless][hd={head_dim}]"
        seed_test_case(test_case)
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([256, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v = self._qkv(420, 900, group=2, head_dim=head_dim)
        out, lse = _assert_bitwise_across_margins(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m),
            self._fwd_margins(),
            test_case,
        )
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(
            out, lse, out_ref, lse_ref, check_finite=True, test_case=test_case
        )

    @with_run_in_mp
    def test_fwd_caller_buffer_accumulation_with_sink(self):
        if self.rank != 0:
            return
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

        out, lse = _assert_bitwise_across_margins(run, self._fwd_margins(), test_case)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, first_call + second_call, sink)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)

    @with_run_in_mp
    def test_fwd_max_logits(self):
        if self.rank != 0:
            return
        test_case = "[fwd_max_logits]"
        seed_test_case(test_case)
        relations, total_q, total_k = _overlapping_relations(
            [200, 45], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = self._qkv(total_q, total_k, group=2)

        def run(sm_margin: int):
            max_logits = torch.full((q.shape[1],), -math.inf, device=self.device)
            out, lse = self._det_fwd(
                q, k, v, relations, max_logits=max_logits, sm_margin=sm_margin
            )
            return out, lse, max_logits

        out, lse, max_logits = _assert_bitwise_across_margins(
            run, self._fwd_margins(), test_case
        )
        out_ref, lse_ref, max_ref = _ref_fwd(q, k, v, relations, return_max_logits=True)
        assert max_ref is not None
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)
        assert_close(
            max_logits,
            max_ref,
            atol=1e-3,
            rtol=1e-4,
            test_case=f"{test_case} => max_logits",
            print_rank=-1,
            print_no_mismatch=False,
        )

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _FwdCase(torch.float32, group=1, tol=_TOL),
            _FwdCase(torch.float32, group=2, tol=_TOL),
            _FwdCase(torch.bfloat16, group=2, tol=2 * _TOL),
        ],
    )
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_fwd_range_merge_bitwise_across_runs_and_sm_margin(
        self, case: _FwdCase, head_dim: int
    ):
        test_case = f"[fwd_range_merge][{case}][hd={head_dim}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _merged_q_relations()
        q, k, v = self._qkv(total_q, total_k, case.group, head_dim=head_dim)
        args = self._range_args(relations)
        plan = plan_range_merge(args["q_ranges"], args["k_ranges"], args["mask_types"])

        def merge_fwd(sm_margin: int, range_merge=True):
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=True,
                range_merge=range_merge,
                deterministic=True,
                out_dtype=case.out_dtype,
                sm_margin=sm_margin,
            )

        with record_fwd_kernels() as built:
            merge_fwd(0)
        (kernel,) = built
        self.assertTrue(kernel.range_merge)
        self.assertFalse(kernel.deterministic)
        with record_fwd_kernels() as reserved_built:
            merge_fwd(self.num_sm - 7)
        (reserved_kernel,) = reserved_built
        self.assertTrue(reserved_kernel.range_merge and reserved_kernel.is_persistent)
        self.assertFalse(reserved_kernel.deterministic)

        out, lse = _assert_bitwise_across_margins(
            merge_fwd, self._fwd_margins(), test_case
        )
        _assert_bitwise_equal(
            merge_fwd(self.num_sm - 7, plan), (out, lse), f"{test_case}[plan]"
        )
        self.assertEqual(out.dtype, case.out_dtype)
        out_atomic, lse_atomic = self._det_fwd(q, k, v, relations)
        _assert_fwd_close(
            out, lse, out_atomic, lse_atomic, tol=case.tol, test_case=test_case
        )
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, tol=case.tol, test_case=test_case)

    @with_run_in_mp
    @parameterize("group", [2, 4])
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_packed_fwd_bitwise_across_runs_and_sm_margin(
        self, group: int, head_dim: int
    ):
        test_case = f"[packed_fwd_bitwise][group={group}][hd={head_dim}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        q, k, v = self._qkv(total_q, total_k, group, head_dim=head_dim)
        out, lse = self._assert_bitwise_packed(q, k, v, relations, test_case)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case=test_case)
        out_unpacked, lse_unpacked = self._det_fwd(q, k, v, relations, pack_gqa=False)
        _assert_fwd_close(
            out,
            lse,
            out_unpacked,
            lse_unpacked,
            tol=1e-3,
            lse_tol=1e-4,
            test_case=f"{test_case}[vs_unpacked]",
        )

    @with_run_in_mp
    @parameterize("group", [2, 4])
    def test_packed_fwd_empty_keyless_and_block_boundary_relations(self, group: int):
        test_case = f"[packed_fwd_empty_keyless][group={group}]"
        seed_test_case(test_case)
        relations: Relations = [
            ([61, 200], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([64, 128], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        q, k, v = self._qkv(420, 900, group=group)
        out, lse = self._assert_bitwise_packed(q, k, v, relations, test_case)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(
            out, lse, out_ref, lse_ref, check_finite=True, test_case=test_case
        )

    @with_run_in_mp
    def test_packed_fwd_cuda_graph_replay_matches_eager_bitwise(self):
        if self.rank != 0:
            return
        test_case = "[packed_fwd_cuda_graph]"
        seed_test_case(test_case)
        relations, total_q, total_k = _overlapping_relations(
            [129, 300], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        args = self._range_args(relations)

        def fwd(q, k, v):
            return _flex_flash_attn_fwd(
                q,
                k,
                v,
                **args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                pack_gqa=True,
            )

        _assert_cuda_graph_matches_eager(
            fwd,
            self._qkv(total_q, total_k, group=4),
            self._qkv(total_q, total_k, group=4),
            test_case,
        )

    @with_run_in_mp
    def test_fwd_with_more_ranges_than_index_rows_bitwise(self):
        if self.rank != 0:
            return
        test_case = "[fwd_more_ranges_than_index_rows]"
        seed_test_case(test_case)
        relations, total_q, total_k = _many_overlapping_q_relations(self._many_ranges())
        q, k, v = self._qkv(total_q, total_k, group=1, num_head_kv=1)
        with record_fwd_kernels() as built:
            self._det_fwd(q, k, v, _ONE_INDEX_ROW_RELATIONS)
            first = self._det_fwd(q, k, v, relations)
        # One compiled kernel reads a one-row and a many-row mWriteIndex.
        (kernel,) = built
        self.assertTrue(kernel.deterministic)
        _assert_bitwise_equal(self._det_fwd(q, k, v, relations), first, test_case)
        out_ref, lse_ref, _ = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(first[0], first[1], out_ref, lse_ref, test_case=test_case)


# ─────────────────────────────────────────────────────────────────────
# Backward deterministic range tests
# ─────────────────────────────────────────────────────────────────────


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicBwd(_RangeDetHelpers, CuteDSLDistTestBase):
    def _assert_bwd_bitwise_and_close(
        self,
        case: _BwdCase,
        relations: Relations,
        total_q: int,
        total_k: int,
        test_case: str,
    ) -> Sequence[torch.Tensor]:
        cluster = 2 if case.two_cta else 1
        prob = self._bwd_problem(
            relations, total_q, total_k, case.group, head_dim=case.head_dim
        )
        with _with_2cta(case.two_cta), record_bwd_kernels() as built:
            first = _assert_bitwise_across_margins(
                lambda m: prob.run(
                    direct_dkv=case.direct_dkv, cat_gqa=case.cat_gqa, sm_margin=m
                ),
                self._bwd_margins(cluster),
                test_case,
            )
        kernel = built[0]
        self.assertEqual(kernel.use_2cta_instrs, case.two_cta)
        self.assertTrue(kernel.is_persistent)
        _assert_grads_close(first, prob.ref_grads(), test_case)
        return first

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=True, head_dim=128),
            _BwdCase(group=1, two_cta=False, head_dim=128),
            _BwdCase(group=2, cat_gqa=True, two_cta=True, head_dim=128),
            _BwdCase(group=2, cat_gqa=True, two_cta=False, head_dim=128),
            _BwdCase(group=1, two_cta=False, head_dim=64),
            _BwdCase(group=2, cat_gqa=True, two_cta=False, head_dim=64),
        ],
    )
    def test_bwd_dq_bitwise_across_sm_margin(self, case: _BwdCase):
        test_case = f"[bwd_dq_bitwise][{case}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _overlapping_relations(
            [129, 300, 1000, 77], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        self._assert_bwd_bitwise_and_close(case, relations, total_q, total_k, test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=True, head_dim=128, direct_dkv=False),
            _BwdCase(group=2, two_cta=True, head_dim=128, direct_dkv=False),
            _BwdCase(group=2, two_cta=False, head_dim=128, direct_dkv=False),
            _BwdCase(
                group=2, cat_gqa=True, two_cta=True, head_dim=128, direct_dkv=False
            ),
            _BwdCase(group=1, two_cta=False, head_dim=64, direct_dkv=False),
            _BwdCase(group=2, two_cta=False, head_dim=64, direct_dkv=False),
            _BwdCase(
                group=2, cat_gqa=True, two_cta=False, head_dim=64, direct_dkv=False
            ),
        ],
    )
    def test_bwd_reduced_dkv_bitwise_across_sm_margin(self, case: _BwdCase):
        test_case = f"[bwd_reduced_dkv_bitwise][{case}]"
        seed_test_case(test_case)
        self._assert_bwd_bitwise_and_close(case, *_shared_k_relations(), test_case)

    @with_run_in_mp
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_bwd_empty_and_keyless_relations(self, head_dim: int):
        test_case = f"[bwd_empty_keyless][hd={head_dim}]"
        seed_test_case(test_case)
        relations: Relations = [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([127, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ]
        prob = self._bwd_problem(relations, 420, 900, group=1, head_dim=head_dim)
        first = _assert_bitwise_across_margins(
            lambda m: prob.run(sm_margin=m), (0, self.num_sm - 2, 0), test_case
        )
        _assert_grads_close(first, prob.ref_grads(), test_case)

    @with_run_in_mp
    def test_bwd_reads_no_range_back_outside_debug_mode(self):
        if self.rank != 0:
            return
        seed_test_case("[bwd_no_range_sync]")
        # Built up front: copying host range lists to the device syncs.
        ordered = self._bwd_problem(
            *_overlapping_relations([129, 300], coverage=2, mask_types=[MT_MAP.full]),
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
        merged = self._bwd_problem(*_merged_k_relations(), group=1)
        merge_plan = plan_range_merge_bwd(
            merged.args["q_ranges"], merged.args["k_ranges"], merged.args["mask_types"]
        )

        def run_all() -> None:
            ordered.run(direct_dkv=False)
            disjoint.run(direct_dkv=True, disable_fwd_atomic_reduction=True)
            merged.run(direct_dkv=True, range_merge=merge_plan)

        with switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", False):
            run_all()
            torch.cuda.synchronize()
            prev_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                run_all()
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

    def _check_bwd_range_merge(
        self,
        case: _BwdCase,
        relations: Relations,
        total_q: int,
        total_k: int,
        test_case: str,
        empty_like: Callable[..., torch.Tensor] = torch.empty_like,
    ) -> Sequence[torch.Tensor]:
        cluster = 2 if case.two_cta else 1
        prob = self._bwd_problem(
            relations, total_q, total_k, case.group, head_dim=case.head_dim
        )
        plan = plan_range_merge_bwd(
            prob.args["q_ranges"], prob.args["k_ranges"], prob.args["mask_types"]
        )

        def merge_bwd(sm_margin: int, range_merge=True):
            with mock.patch("torch.empty_like", empty_like):
                return prob.run(
                    range_merge=range_merge, cat_gqa=case.cat_gqa, sm_margin=sm_margin
                )

        with _with_2cta(case.two_cta):
            with record_bwd_kernels() as built:
                first = merge_bwd(0)
            (kernel,) = built
            self.assertTrue(
                kernel.range_merge and kernel.deterministic and kernel.is_persistent
            )
            self.assertEqual(kernel.use_2cta_instrs, case.two_cta)
            self.assertEqual(kernel.dq_pair_handoff, case.two_cta)
            self.assertFalse(kernel.dKV_postprocess)
            for margin in self._bwd_margins(cluster)[1:]:
                _assert_bitwise_equal(
                    merge_bwd(margin), first, f"{test_case}[margin={margin}]"
                )
            _assert_bitwise_equal(
                merge_bwd(self.num_sm - 4 * cluster, plan), first, f"{test_case}[plan]"
            )
            unmerged = prob.run(direct_dkv=False, cat_gqa=case.cat_gqa)
        _assert_grads_close(first, unmerged, f"{test_case}[vs_unmerged]")
        _assert_grads_close(first, prob.ref_grads(), test_case)
        return first

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, head_dim=64, two_cta=False),
            _BwdCase(group=2, cat_gqa=True, head_dim=64, two_cta=False),
            _BwdCase(group=1, head_dim=128, two_cta=False),
            _BwdCase(group=2, cat_gqa=True, head_dim=128, two_cta=False),
            _BwdCase(group=1, head_dim=128, two_cta=True),
            _BwdCase(group=2, cat_gqa=True, head_dim=128, two_cta=True),
        ],
    )
    def test_bwd_range_merge_bitwise_across_sm_margin(self, case: _BwdCase):
        test_case = f"[bwd_range_merge][{case}]"
        seed_test_case(test_case)
        self._check_bwd_range_merge(case, *_merged_k_relations(), test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, head_dim=128, two_cta=False),
            _BwdCase(group=2, cat_gqa=True, head_dim=128, two_cta=True),
        ],
    )
    def test_bwd_range_merge_heavy_middle_group_and_k_rows_in_no_group(
        self, case: _BwdCase
    ):
        test_case = f"[bwd_range_merge_heavy_middle][{case}]"
        seed_test_case(test_case)
        _, dk, dv = self._check_bwd_range_merge(
            case,
            *_merged_k_relations_heavy_middle(),
            test_case,
            empty_like=_nan_empty_like,
        )
        for lo, hi in ((557, 580), (1000, 1100)):
            assert_equal(
                dk[lo:hi], torch.zeros_like(dk[lo:hi]), print_no_mismatch=False
            )
            assert_equal(
                dv[lo:hi], torch.zeros_like(dv[lo:hi]), print_no_mismatch=False
            )

    def _assert_2cta_handoff_under_delays(
        self,
        relations: Relations,
        total_q: int,
        total_k: int,
        cat_gqa: bool,
        points: Sequence[DelayPoint],
        margins: Sequence[int],
        seed: int,
        test_case: str,
        repeats: int = 1,
    ) -> None:
        prob = self._bwd_problem(relations, total_q, total_k, group=2 if cat_gqa else 1)

        def merge_bwd(sm_margin: int):
            return prob.run(range_merge=True, cat_gqa=cat_gqa, sm_margin=sm_margin)

        with _with_2cta(True):
            with record_bwd_kernels() as built:
                first = merge_bwd(0)
            (kernel,) = built
            self.assertTrue(kernel.use_2cta_instrs and kernel.dq_pair_handoff)
            for point in points:
                with delay_injection(point, seed=seed, max_ns=20_000):
                    for sm_margin in margins:
                        for _ in range(repeats):
                            _assert_bitwise_equal(
                                merge_bwd(sm_margin),
                                first,
                                f"{test_case}[{point.value}][margin={sm_margin}]",
                            )
        _assert_grads_close(first, prob.ref_grads(), test_case)

    @with_run_in_mp
    @parameterize("cat_gqa", [False, True])
    def test_bwd_range_merge_2cta_adjacent_pair_conflicts_bitwise(self, cat_gqa: bool):
        test_case = f"[bwd_2cta_adjacent_conflicts][cat_gqa={cat_gqa}]"
        seed_test_case(test_case)
        self._assert_2cta_handoff_under_delays(
            *_merged_k_relations_2cta_adjacent(),
            cat_gqa=cat_gqa,
            points=(DelayPoint.WRITE_ORDER_WAIT,),
            margins=(self.num_sm - 2,),
            seed=7,
            test_case=test_case,
        )

    @unittest.skipUnless(
        _STRESS_REPEATS > 0, "set MAGI_ATTENTION_TEST_2CTA_STRESS_REPEATS to run"
    )
    @with_run_in_mp
    @parameterize("cat_gqa", [False, True])
    def test_bwd_range_merge_2cta_adjacent_pair_conflicts_stress(self, cat_gqa: bool):
        """Set MAGI_ATTENTION_TEST_2CTA_STRESS_REPEATS to repeat each SM budget."""
        test_case = f"[bwd_2cta_adjacent_stress][cat_gqa={cat_gqa}]"
        seed_test_case(test_case)
        self._assert_2cta_handoff_under_delays(
            *_merged_k_relations_2cta_adjacent(),
            cat_gqa=cat_gqa,
            points=(DelayPoint.WRITE_ORDER_WAIT,),
            margins=(0, self.num_sm - 2),
            seed=7,
            test_case=test_case,
            repeats=_STRESS_REPEATS,
        )

    @with_run_in_mp
    @parameterize("cat_gqa", [False, True])
    def test_bwd_range_merge_2cta_pair_handoff_with_mixed_masks(self, cat_gqa: bool):
        test_case = f"[bwd_2cta_pair_handoff_mixed_masks][cat_gqa={cat_gqa}]"
        seed_test_case(test_case)
        relations: Relations = [
            ([0, 900], [0, 256], MT_MAP.full),
            ([0, 128], [256, 440], MT_MAP.full),
            ([200, 315], [256, 440], MT_MAP.causal),
            ([320, 515], [256, 440], MT_MAP.bi_causal),
            ([520, 648], [256, 440], MT_MAP.inv_causal),
            ([584, 712], [256, 440], MT_MAP.causal),
            ([768, 890], [256, 440], MT_MAP.full),
        ]
        self._assert_2cta_handoff_under_delays(
            relations,
            900,
            450,
            cat_gqa=cat_gqa,
            points=(DelayPoint.WRITE_ORDER_WAIT, DelayPoint.PAIR_ARRIVE),
            margins=(0, self.num_sm - 2),
            seed=23,
            test_case=test_case,
        )

    @with_run_in_mp
    def test_public_api_fwd_bwd_bitwise_with_sink(self):
        if self.rank != 0:
            return
        test_case = "[public_api_fwd_bwd_sink]"
        seed_test_case(test_case)
        relations, total_q, total_k = _shared_k_relations()
        q, k, v = self._qkv(total_q, total_k, group=2)
        sink = torch.randn(2, q.shape[1], device=self.device, dtype=torch.float32)
        do = torch.randn_like(q)

        def run(sm_margin: int):
            leaves = [t.detach().clone().requires_grad_() for t in (q, k, v, sink)]
            out, meta = flex_flash_attn_func(
                *leaves[:3],
                **self._range_args(relations),
                sink=leaves[3],
                deterministic=True,
                sm_margin=sm_margin,
            )
            out.backward(do)
            return [out.detach(), meta.lse, *(t.grad for t in leaves)]

        with _with_2cta(True):
            _assert_bitwise_across_margins(run, (0, 1, self.num_sm - 2, 0), test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(range_merge=False, two_cta=False, head_dim=64),
            _BwdCase(range_merge=False, two_cta=False, head_dim=128),
            _BwdCase(range_merge=False, two_cta=True, head_dim=128),
            _BwdCase(range_merge=True, two_cta=False, head_dim=128),
            _BwdCase(range_merge=True, two_cta=True, head_dim=128),
        ],
    )
    def test_bwd_straddling_masked_blocks_all_mask_types(self, case: _BwdCase):
        test_case = f"[bwd_straddling_masked][{case}]"
        seed_test_case(test_case)
        cluster = 2 if case.two_cta else 1
        relations: Relations = [
            ([0, 390], [10, 270], MT_MAP.bi_causal),
            ([0, 390], [270, 530], MT_MAP.bi_causal),
            ([400, 790], [270, 530], MT_MAP.bi_causal),
            ([400, 520], [530, 600], MT_MAP.bi_causal),
            ([7, 267], [600, 860], MT_MAP.causal),
            ([6, 266], [600, 860], MT_MAP.inv_causal),
            ([19, 535], [860, 1140], MT_MAP.full),
        ]
        prob = self._bwd_problem(relations, 800, 1150, group=2, head_dim=case.head_dim)

        def run(sm_margin: int):
            return prob.run(
                direct_dkv=case.range_merge,
                range_merge=case.range_merge,
                cat_gqa=True,
                sm_margin=sm_margin,
            )

        with _with_2cta(case.two_cta):
            first = _assert_bitwise_across_margins(
                run,
                (0, 1, self.num_sm - 2 * cluster, self.num_sm - cluster),
                test_case,
            )
            with delay_injection(DelayPoint.WRITE_ORDER_WAIT, seed=11, max_ns=15_000):
                _assert_bitwise_equal(
                    run(self.num_sm - 2 * cluster), first, f"{test_case}[delay]"
                )
        _assert_grads_close(first, prob.ref_grads(), test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase(group=1, two_cta=True, head_dim=128, direct_dkv=True),
            _BwdCase(group=1, two_cta=False, head_dim=128, direct_dkv=False),
            _BwdCase(
                group=2, cat_gqa=True, two_cta=True, head_dim=128, direct_dkv=False
            ),
            _BwdCase(group=2, two_cta=True, head_dim=128, direct_dkv=False),
            _BwdCase(group=1, two_cta=False, head_dim=64, direct_dkv=True),
            _BwdCase(
                group=2, cat_gqa=True, two_cta=False, head_dim=64, direct_dkv=False
            ),
        ],
    )
    def test_bwd_disjoint_q_dense_dqacc_bitwise_across_sm_margin(self, case: _BwdCase):
        test_case = f"[bwd_disjoint_dense_dqacc][{case}]"
        seed_test_case(test_case)
        cluster = 2 if case.two_cta else 1
        relations, total_k = _DISJOINT_Q_RELATIONS[case.direct_dkv]
        prob = self._bwd_problem(
            relations,
            1260,
            total_k,
            case.group,
            head_dim=case.head_dim,
            disable_fwd_atomic=True,
        )

        def run(sm_margin: int, dq_init: torch.Tensor | None = None):
            return prob.run(
                direct_dkv=case.direct_dkv,
                dq=dq_init,
                disable_fwd_atomic_reduction=True,
                cat_gqa=case.cat_gqa,
                sm_margin=sm_margin,
            )

        with _with_2cta(case.two_cta):
            with record_bwd_kernels() as built:
                first = run(0)
            (kernel,) = built
            self.assertTrue(
                kernel.use_dense_dqacc_for_ranges and kernel.range_dq_semaphore
            )
            self.assertFalse(kernel.range_dq_write_order or kernel.dQ_rowmajor_accum)
            for margin in (1, self.num_sm - 2 * cluster, self.num_sm - cluster):
                _assert_bitwise_equal(
                    run(margin), first, f"{test_case}[margin={margin}]"
                )
            with delay_injection(DelayPoint.START, seed=19, max_ns=20_000):
                _assert_bitwise_equal(run(0), first, f"{test_case}[delay]")
            dq_base = torch.randn(prob.q.shape, device=self.device, dtype=torch.float32)
            dq_acc_0 = run(0, dq_init=dq_base.clone())[0]
            dq_acc_1 = run(self.num_sm - cluster, dq_init=dq_base.clone())[0]
            assert_equal(
                dq_acc_0,
                dq_acc_1,
                test_case=f"{test_case}[dq_acc]",
                print_no_mismatch=False,
            )
            assert_close(
                dq_acc_0,
                dq_base + first[0],
                atol=1e-4,
                rtol=1e-4,
                test_case=f"{test_case}[dq_acc_sum]",
                print_rank=-1,
                print_no_mismatch=False,
            )
        _assert_grads_close(first, prob.ref_grads(), test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [_BwdCase(head_dim=64, two_cta=False), _BwdCase(head_dim=128, two_cta=True)],
    )
    def test_bwd_range_merge_cuda_graph_replay_matches_eager_bitwise(
        self, case: _BwdCase
    ):
        test_case = f"[bwd_range_merge_cuda_graph][{case}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _merged_k_relations()
        args = self._range_args(relations)
        plan = plan_range_merge_bwd(
            args["q_ranges"], args["k_ranges"], args["mask_types"]
        )
        data = [
            self._bwd_problem(
                relations, total_q, total_k, group=2, head_dim=case.head_dim
            ).tensors()
            for _ in range(2)
        ]

        def bwd(*tensors):
            return _flex_flash_attn_bwd(
                *tensors,
                **args,
                range_merge=plan,
                disable_bwd_dkv_atomic_reduction=True,
                cat_gqa=True,
                deterministic=True,
            )[:3]

        # Replay inside record_bwd_kernels so the swapped cache keeps the module.
        with _with_2cta(case.two_cta), record_bwd_kernels() as built:
            _assert_cuda_graph_matches_eager(bwd, data[0], data[1], test_case)
            (kernel,) = built
            self.assertEqual(kernel.use_2cta_instrs, case.two_cta)
            self.assertTrue(kernel.range_merge and kernel.deterministic)

    @with_run_in_mp
    @parameterize(
        "case",
        [_BwdCase(head_dim=64, two_cta=False), _BwdCase(head_dim=128, two_cta=True)],
    )
    def test_bwd_disjoint_q_dense_dqacc_cuda_graph_replay(self, case: _BwdCase):
        test_case = f"[bwd_disjoint_dense_dqacc_graph][{case}]"
        seed_test_case(test_case)
        relations: Relations = [
            ([5, 270], [0, 310], MT_MAP.causal),
            ([280, 550], [120, 420], MT_MAP.inv_causal),
            ([560, 830], [200, 510], MT_MAP.full),
        ]
        args = self._range_args(relations)

        def bwd(*tensors):
            return _flex_flash_attn_bwd(
                *tensors,
                **args,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=False,
                cat_gqa=True,
                deterministic=True,
            )[:3]

        with _with_2cta(case.two_cta), record_bwd_kernels() as built:
            data = [
                self._bwd_problem(
                    relations,
                    850,
                    520,
                    group=2,
                    head_dim=case.head_dim,
                    disable_fwd_atomic=True,
                ).tensors()
                for _ in range(2)
            ]
            _assert_cuda_graph_matches_eager(bwd, data[0], data[1], test_case)
            (kernel,) = built
            self.assertTrue(kernel.range_dq_semaphore)
            self.assertEqual(kernel.use_2cta_instrs, case.two_cta)

    @with_run_in_mp
    @parameterize("head_dim", _DET_HEAD_DIMS)
    def test_cuda_graph_replay_matches_eager_bitwise(self, head_dim: int):
        test_case = f"[fwd_bwd_cuda_graph][hd={head_dim}]"
        seed_test_case(test_case)
        relations, total_q, total_k = _shared_k_relations()
        args = self._range_args(relations)

        def fwd_and_bwd(*tensors):
            fwd_out, fwd_lse = _flex_flash_attn_fwd(
                *tensors[:3],
                **args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
            )
            grads = _flex_flash_attn_bwd(
                *tensors,
                **args,
                disable_bwd_dkv_atomic_reduction=False,
                deterministic=True,
            )[:3]
            return (fwd_out, fwd_lse, *grads)

        with _with_2cta(True):
            data = [
                self._bwd_problem(
                    relations, total_q, total_k, group=2, head_dim=head_dim
                ).tensors()
                for _ in range(2)
            ]
            _assert_cuda_graph_matches_eager(fwd_and_bwd, data[0], data[1], test_case)

    @with_run_in_mp
    @parameterize("mode", ["dq_and_dkv", "range_merge", "overlapping_k"])
    def test_bwd_with_more_ranges_than_index_rows_bitwise(self, mode: str):
        test_case = f"[bwd_more_ranges_than_index_rows][{mode}]"
        seed_test_case(test_case)
        num_ranges = self._many_ranges()
        if mode == "dq_and_dkv":
            prob = self._bwd_problem(
                *_many_overlapping_q_relations(num_ranges), group=1, num_head_kv=1
            )
            with _with_2cta(True):
                with record_bwd_kernels() as built:
                    # One compiled kernel reads a one-row and a many-row mWriteIndex.
                    _flex_flash_attn_bwd(
                        *prob.tensors(),
                        **self._range_args(_ONE_INDEX_ROW_RELATIONS),
                        disable_bwd_dkv_atomic_reduction=False,
                        deterministic=True,
                    )
                    first = prob.run(direct_dkv=False)
                (kernel,) = built
                self.assertTrue(
                    kernel.range_dq_write_order and kernel.range_dkv_write_order
                )
                self.assertFalse(kernel.range_merge)
                _assert_bitwise_equal(prob.run(direct_dkv=False), first, test_case)
        elif mode == "range_merge":
            prob = self._bwd_problem(
                *_many_merged_k_relations(num_ranges), group=1, num_head_kv=1
            )
            plan = plan_range_merge_bwd(
                prob.args["q_ranges"], prob.args["k_ranges"], prob.args["mask_types"]
            )
            with record_bwd_kernels() as built:
                first = prob.run(direct_dkv=True, range_merge=plan)
            (kernel,) = built
            self.assertTrue(kernel.range_merge and kernel.range_dq_write_order)
            self.assertFalse(kernel.range_dkv_write_order)
            _assert_bitwise_equal(
                prob.run(direct_dkv=True, range_merge=plan), first, test_case
            )
        else:
            prob = self._bwd_problem(
                *_many_overlapping_k_relations(num_ranges),
                group=1,
                num_head_kv=1,
                disable_fwd_atomic=True,
            )
            with record_bwd_kernels() as built:
                first = prob.run(direct_dkv=False)
            (kernel,) = built
            self.assertTrue(kernel.range_dkv_write_order)
            _assert_bitwise_equal(prob.run(direct_dkv=False), first, test_case)
        _assert_grads_close(first, prob.ref_grads(), test_case)

    @with_run_in_mp
    @parameterize("num_units", sorted(_BWD_UNIT_RELATIONS))
    def test_bwd_2cta_peers_consume_matching_units(self, num_units: int):
        test_case = f"[bwd_2cta_peers][T={num_units}]"
        seed_test_case(test_case)
        cluster_size, num_clusters = 2, 4
        prob = self._bwd_problem(
            _BWD_UNIT_RELATIONS[num_units], 400, 1400, group=1, num_head_kv=1
        )
        valid = num_units * cluster_size
        with _with_2cta(True), record_visits(
            valid + 16 * cluster_size, cluster_size
        ) as (visits, clusters), record_bwd_kernels() as built:
            grads = prob.run(
                direct_dkv=False, sm_margin=self.num_sm - cluster_size * num_clusters
            )
        (kernel,) = built
        self.assertTrue(kernel.use_2cta_instrs)
        _assert_grads_close(grads, prob.ref_grads(), test_case)

        visits, clusters = visits.cpu(), clusters.cpu()
        warps = int(visits[0])
        self.assertGreater(warps, 0)
        assert_equal(
            visits[:valid],
            torch.full_like(visits[:valid], warps),
            test_case=test_case,
            print_no_mismatch=False,
        )
        assert_equal(
            visits[valid:],
            torch.zeros_like(visits[valid:]),
            test_case=test_case,
            print_no_mismatch=False,
        )
        per_unit = clusters[:valid].view(num_units, cluster_size)
        assert_equal(
            per_unit % warps,
            torch.zeros_like(per_unit),
            test_case=test_case,
            print_no_mismatch=False,
        )
        assert_equal(
            per_unit[:, 0], per_unit[:, 1], test_case=test_case, print_no_mismatch=False
        )


if __name__ == "__main__":
    run_tests()
