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
from magi_attention.kernel.cutedsl.range_merge import (
    RangeMergePlan,
    plan_range_merge_bwd,
)
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
_OUT_LSE_NAMES = ("out", "lse")
_GRAD_NAMES = ("dq", "dk", "dv")
_PUBLIC_API_NAMES = ("out", "lse", "dq", "dk", "dv", "dsink")


def _with_2cta(enabled: bool) -> switch_env:
    return switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA", not enabled)


def _assert_fwd_close(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    test_case: str,
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
    base = run(margins[0])
    assert_sequence_equal(run(margins[0]), base, names, f"{test_case}[rerun]")
    for m in margins[1:]:
        assert_sequence_equal(run(m), base, names, f"{test_case}[margin={m}]")
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
# Relation fixtures
# ---------------------------------------------------------------------------


def _overlapping_cov4() -> tuple[Relations, int, int]:
    return overlapping_relations(
        [129, 300, 1000, 77], coverage=4, mask_types=[MT_MAP.full, MT_MAP.causal]
    )


def _overlapping_cov3() -> tuple[Relations, int, int]:
    return overlapping_relations(
        [129, 300, 1000, 77], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
    )


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
    """Eight K groups whose four 128-row Q pairs shift by 64 rows per pair."""
    masks = (MT_MAP.inv_causal, MT_MAP.bi_causal, MT_MAP.bi_causal, MT_MAP.causal)
    relations: Relations = [([0, 3072], [0, 256], MT_MAP.full)]
    for g in range(8):
        q0, k0 = g * 384, 256 + g * 160
        for p, mask in enumerate(masks):
            relations.append(([q0 + p * 64, q0 + p * 64 + 128], [k0, k0 + 160], mask))
    return relations, 3072, 1536


def _many_overlapping_relations() -> tuple[Relations, int, int]:
    """256 disjoint (qb, kb) rectangles so every Q/K block gets >32 writers."""
    bands = 16
    relations: Relations = [
        (
            [(g % bands) * 48 + 3, (g % bands) * 48 + 47],
            [((g // bands + g) % bands) * 48 + 5, ((g // bands + g) % bands) * 48 + 49],
            _ALL_MASKS[g % len(_ALL_MASKS)],
        )
        for g in range(bands * bands)
    ]
    return relations, bands * 48 + 16, bands * 48 + 16


def _many_merged_k_relations() -> tuple[Relations, int, int]:
    num_groups, k_step, total_q = 212, 32, 256
    relations: Relations = []
    for g in range(num_groups):
        k_range = [g * k_step, (g + 1) * k_step]
        q0 = (g * 17) % 128
        relations.append(([q0, q0 + 40], k_range, _ALL_MASKS[g % len(_ALL_MASKS)]))
        relations.append(([q0 + 64, q0 + 112], k_range, MT_MAP.causal))
    return relations, total_q, num_groups * k_step + 32


def _disjoint_q_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([3, 132], [0, 290], MT_MAP.full),
            ([140, 410], [290, 560], MT_MAP.causal),
            ([420, 420], [560, 620], MT_MAP.full),
            ([430, 690], [620, 620], MT_MAP.full),
            ([700, 965], [620, 900], MT_MAP.inv_causal),
            ([970, 1240], [900, 1170], MT_MAP.bi_causal),
        ],
        1260,
        1180,
    )


def _empty_keyless_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([1, 130], [0, 200], MT_MAP.full),
            ([50, 50], [200, 260], MT_MAP.full),
            ([100, 300], [260, 260], MT_MAP.full),
            ([127, 400], [260, 500], MT_MAP.causal),
            ([129, 130], [500, 520], MT_MAP.full),
            ([0, 257], [520, 900], MT_MAP.full),
        ],
        420,
        900,
    )


def _straddling_relations() -> tuple[Relations, int, int]:
    return (
        [
            ([0, 390], [10, 270], MT_MAP.bi_causal),
            ([0, 390], [270, 530], MT_MAP.bi_causal),
            ([400, 790], [270, 530], MT_MAP.bi_causal),
            ([400, 520], [530, 600], MT_MAP.bi_causal),
            ([7, 267], [600, 860], MT_MAP.causal),
            ([6, 266], [600, 860], MT_MAP.inv_causal),
            ([19, 535], [860, 1140], MT_MAP.full),
        ],
        800,
        1150,
    )


@dataclass(frozen=True)
class _FwdCase:
    name: str
    fixture: Callable[[], tuple[Relations, int, int]]
    group: int = 2
    num_head_kv: int = _NUM_HEAD_KV
    out_dtype: torch.dtype = torch.float32
    pack_gqa: bool = False


@dataclass(frozen=True)
class _BwdCase:
    name: str
    fixture: Callable[[], tuple[Relations, int, int]]
    group: int = 1
    num_head_kv: int = _NUM_HEAD_KV
    cat_gqa: bool = False
    two_cta: bool = False
    head_dim: int = _HEAD_DIM
    direct_dkv: bool = True
    disable_fwd_atomic: bool = False
    range_merge: str = "off"  # "off", "auto", or "plan"


_BwdTensors = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class _RangeDetHelpers:
    if TYPE_CHECKING:

        @property
        def device(self) -> int:
            ...

    @property
    def num_sm(self) -> int:
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def _margins(self) -> tuple[int, int, int]:
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

    def _det_bwd(self, tensors, args, **kwargs):
        return _flex_flash_attn_bwd(*tensors, **args, deterministic=True, **kwargs)[:3]

    def _bwd_inputs(
        self,
        relations: Relations,
        total_q: int,
        total_k: int,
        group: int,
        *,
        head_dim: int = _HEAD_DIM,
        num_head_kv: int = _NUM_HEAD_KV,
        disable_fwd_atomic: bool = False,
    ) -> tuple[_BwdTensors, dict]:
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
        return (q, k, v, out.to(q.dtype), lse, torch.randn_like(q)), args


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicHost(_RangeDetHelpers, TestCase):
    @property
    def device(self) -> int:
        return torch.cuda.current_device()

    def test_host_bounds_and_debug_validation(self):
        seed_test_case("test_host_bounds_and_debug_validation")
        num_ranges, rows = 2**24, 4096
        ranges = (
            torch.tensor([0, rows], dtype=torch.int32, device=self.device)
            .expand(num_ranges, 2)
            .contiguous()
        )
        args = dict(
            q_ranges=ranges,
            k_ranges=ranges,
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
            self._det_bwd(
                (q, k, v, q, lse, q), args, disable_bwd_dkv_atomic_reduction=False
            )

        dbg_rels: Relations = [
            ([0, 100], [0, 300], MT_MAP.full),
            ([50, 200], [300, 400], MT_MAP.causal),
        ]
        q_d, k_d, v_d = self._qkv(200, 400, group=1)
        lse_d = torch.zeros(200, q_d.shape[1], device=self.device)
        dbg_args = {**self._range_args(dbg_rels), "max_seqlen_k": 299}
        with switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", True):
            with self.assertRaises(ValueError):
                self._det_bwd(
                    (q_d, k_d, v_d, q_d, lse_d, q_d),
                    dbg_args,
                    disable_bwd_dkv_atomic_reduction=False,
                )

    def test_persistent_budget_and_path_activation(self):
        seed_test_case("test_persistent_budget_and_path_activation")
        relations, total_q, total_k = _shared_k_relations()
        tensors, args = self._bwd_inputs(relations, total_q, total_k, group=2)
        q, k, v = tensors[:3]
        with record_fwd_kernels() as built_fwd:
            self._det_fwd(q, k, v, relations)
        self.assertTrue(built_fwd[0].deterministic)
        with record_bwd_kernels() as built_bwd:
            self._det_bwd(tensors, args, disable_bwd_dkv_atomic_reduction=False)
        self.assertTrue(built_bwd[0].deterministic)

        with _with_2cta(True), self.assertRaises(ValueError):
            self._det_bwd(
                tensors,
                args,
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 1,
            )
        with self.assertRaises(ValueError):
            self._det_fwd(q, k, v, relations, sm_margin=self.num_sm)
        with _with_2cta(True):
            grads = self._det_bwd(
                tensors,
                args,
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 2,
            )
        assert_sequence_close(
            grads, _ref_grads(q, k, v, tensors[5], relations), _GRAD_NAMES, _GRAD_TOL
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicSweeps(_RangeDetHelpers, CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize(
        "case",
        [
            _FwdCase("fp32", _overlapping_cov4),
            _FwdCase("bf16", _overlapping_cov4, out_dtype=torch.bfloat16),
            _FwdCase("pack_gqa", _overlapping_cov4, group=4, pack_gqa=True),
            _FwdCase("empty_unpacked", _empty_keyless_relations),
            _FwdCase("empty_packed", _empty_keyless_relations, pack_gqa=True),
            _FwdCase("many_256", _many_overlapping_relations, group=1, num_head_kv=1),
        ],
    )
    def test_fwd_bitwise_across_margins(self, case: _FwdCase):
        test_case = f"[fwd_det][{case.name}]"
        seed_test_case(test_case)
        relations, total_q, total_k = case.fixture()
        q, k, v = self._qkv(total_q, total_k, case.group, num_head_kv=case.num_head_kv)
        kwargs = {"out_dtype": case.out_dtype, "pack_gqa": case.pack_gqa}
        out, lse = _bitwise_across_margins(
            lambda m: self._det_fwd(q, k, v, relations, sm_margin=m, **kwargs),
            self._margins(),
            _OUT_LSE_NAMES,
            test_case,
        )
        self.assertEqual(out.dtype, case.out_dtype)
        out_ref, lse_ref = _ref_fwd(q, k, v, relations)
        _assert_fwd_close(out, lse, out_ref, lse_ref, test_case)

    @with_run_in_mp
    @parameterize(
        "case",
        [
            _BwdCase("dq_1cta", _overlapping_cov3),
            _BwdCase("dq_2cta_cat", _overlapping_cov3, 2, cat_gqa=True, two_cta=True),
            _BwdCase("dkv_mha", _shared_k_relations, direct_dkv=False),
            _BwdCase("dkv_gqa", _shared_k_relations, 2, direct_dkv=False),
            _BwdCase(
                "dkv_2cta_cat",
                _shared_k_relations,
                2,
                cat_gqa=True,
                two_cta=True,
                direct_dkv=False,
            ),
            _BwdCase(
                "disjoint_q_1cta",
                _disjoint_q_relations,
                direct_dkv=False,
                disable_fwd_atomic=True,
            ),
            _BwdCase(
                "disjoint_q_2cta",
                _disjoint_q_relations,
                2,
                cat_gqa=True,
                two_cta=True,
                direct_dkv=False,
                disable_fwd_atomic=True,
            ),
            _BwdCase(
                "rm_1cta_hd128",
                _merged_k_relations,
                2,
                cat_gqa=True,
                range_merge="plan",
            ),
            _BwdCase(
                "rm_2cta_hd128",
                _merged_k_relations,
                2,
                cat_gqa=True,
                two_cta=True,
                range_merge="plan",
            ),
            _BwdCase(
                "rm_1cta_hd64",
                _merged_k_relations,
                2,
                cat_gqa=True,
                head_dim=64,
                range_merge="plan",
            ),
            _BwdCase(
                "straddle_plain",
                _straddling_relations,
                2,
                cat_gqa=True,
                direct_dkv=False,
            ),
            _BwdCase(
                "straddle_rm_2cta",
                _straddling_relations,
                2,
                cat_gqa=True,
                two_cta=True,
                range_merge="auto",
            ),
            _BwdCase(
                "rm_2cta_pair_handoff",
                _merged_k_relations_2cta_adjacent,
                2,
                cat_gqa=True,
                two_cta=True,
                range_merge="auto",
            ),
            _BwdCase(
                "many_256_2cta",
                _many_overlapping_relations,
                1,
                1,
                two_cta=True,
                direct_dkv=False,
            ),
            _BwdCase("many_212_rm", _many_merged_k_relations, 1, 1, range_merge="plan"),
            _BwdCase("empty_keyless", _empty_keyless_relations),
        ],
    )
    def test_bwd_bitwise_across_margins(self, case: _BwdCase):
        test_case = f"[bwd_det][{case.name}]"
        seed_test_case(test_case)
        relations, total_q, total_k = case.fixture()
        tensors, args = self._bwd_inputs(
            relations,
            total_q,
            total_k,
            case.group,
            head_dim=case.head_dim,
            num_head_kv=case.num_head_kv,
            disable_fwd_atomic=case.disable_fwd_atomic,
        )
        rm_arg: bool | RangeMergePlan = False
        if case.range_merge == "auto":
            rm_arg = True
        elif case.range_merge == "plan":
            rm_arg = plan_range_merge_bwd(
                args["q_ranges"], args["k_ranges"], args["mask_types"]
            )
        kwargs = dict(
            disable_fwd_atomic_reduction=case.disable_fwd_atomic,
            disable_bwd_dkv_atomic_reduction=case.direct_dkv,
            cat_gqa=case.cat_gqa,
            range_merge=rm_arg,
        )
        with _with_2cta(case.two_cta):
            first = _bitwise_across_margins(
                lambda m: self._det_bwd(tensors, args, sm_margin=m, **kwargs),
                self._margins(),
                _GRAD_NAMES,
                test_case,
            )
        assert_sequence_close(
            first,
            _ref_grads(tensors[0], tensors[1], tensors[2], tensors[5], relations),
            _GRAD_NAMES,
            _GRAD_TOL,
            test_case,
        )


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestFFARangeDeterministicIntegration(_RangeDetHelpers, CuteDSLDistTestBase):
    @property
    def world_size(self) -> int:
        return 1

    @with_run_in_mp
    def test_caller_buffers_and_single_cluster(self):
        seed_test_case("[det_caller_buffers_and_single_cluster]")
        first_call: Relations = [
            ([0, 140], [0, 300], MT_MAP.full),
            ([70, 260], [300, 500], MT_MAP.causal),
        ]
        second_call: Relations = [
            ([3, 250], [500, 800], MT_MAP.full),
            ([128, 129], [800, 820], MT_MAP.full),
        ]
        q, k, v = self._qkv(260, 820, group=2)
        sink = torch.randn(2, q.shape[1], device=self.device, dtype=torch.float32)

        def run_fwd_acc(sm_margin: int):
            out = torch.zeros(260, q.shape[1], _HEAD_DIM, device=self.device)
            lse = torch.full((260, q.shape[1]), -math.inf, device=self.device)
            self._det_fwd(q, k, v, first_call, out=out, lse=lse, sm_margin=sm_margin)
            self._det_fwd(
                q, k, v, second_call, out=out, lse=lse, sink=sink, sm_margin=sm_margin
            )
            return out, lse

        out, lse = _bitwise_across_margins(
            run_fwd_acc, self._margins(), _OUT_LSE_NAMES, "[fwd_caller_acc]"
        )
        out_ref, lse_ref = _ref_fwd(q, k, v, first_call + second_call, sink)
        _assert_fwd_close(out, lse, out_ref, lse_ref, "[fwd_caller_acc]")

        disjoint_rels, dq_tot_q, dq_tot_k = _disjoint_q_relations()
        tensors, args = self._bwd_inputs(
            disjoint_rels, dq_tot_q, dq_tot_k, 2, disable_fwd_atomic=True
        )
        dq_base = torch.randn(tensors[0].shape, device=self.device, dtype=torch.float32)
        bwd_kw = dict(disable_fwd_atomic_reduction=True, cat_gqa=True)
        with _with_2cta(True):
            base_dq = self._det_bwd(tensors, args, sm_margin=0, **bwd_kw)[0]
            acc_0 = self._det_bwd(
                tensors, args, sm_margin=0, dq=dq_base.clone(), **bwd_kw
            )[0]
            acc_1 = self._det_bwd(
                tensors,
                args,
                sm_margin=self.num_sm - 2,
                dq=dq_base.clone(),
                **bwd_kw,
            )[0]
        assert_equal(acc_0, acc_1, test_case="[bwd_dq_acc]", print_no_mismatch=False)
        assert_close(
            acc_0,
            dq_base + base_dq,
            atol=1e-4,
            rtol=1e-4,
            test_case="[bwd_dq_acc_sum]",
            print_rank=-1,
            print_no_mismatch=False,
        )

        sparse_empty: Relations = (
            [([0, 128], [0, 200], MT_MAP.full)]
            + [([150, 150], [200, 200], MT_MAP.full)] * 61
            + [([160, 288], [220, 420], MT_MAP.causal)]
        )
        sc_t, sc_args = self._bwd_inputs(sparse_empty, 300, 440, group=1, num_head_kv=1)
        with _with_2cta(False):
            grads = self._det_bwd(
                sc_t,
                sc_args,
                disable_bwd_dkv_atomic_reduction=False,
                sm_margin=self.num_sm - 1,
            )
        assert_sequence_close(
            grads,
            _ref_grads(sc_t[0], sc_t[1], sc_t[2], sc_t[5], sparse_empty),
            _GRAD_NAMES,
            _GRAD_TOL,
        )

    @with_run_in_mp
    def test_sync_free_public_api_and_cuda_graphs(self):
        seed_test_case("[det_sync_public_api_graphs]")
        ov_rels, ov_q, ov_k = overlapping_relations(
            [129, 300], coverage=3, mask_types=[MT_MAP.full, MT_MAP.causal]
        )
        ord_t, ord_args = self._bwd_inputs(ov_rels, ov_q, ov_k, group=1)
        dis_rels, dis_q, dis_k = _disjoint_q_relations()
        dis_t, dis_args = self._bwd_inputs(
            dis_rels, dis_q, dis_k, group=2, disable_fwd_atomic=True
        )

        def run_sync_check() -> None:
            self._det_bwd(ord_t, ord_args, disable_bwd_dkv_atomic_reduction=False)
            self._det_bwd(
                dis_t,
                dis_args,
                disable_bwd_dkv_atomic_reduction=True,
                disable_fwd_atomic_reduction=True,
                cat_gqa=True,
            )

        with switch_env("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", False):
            run_sync_check()
            torch.cuda.synchronize()
            with sync_debug_error():
                run_sync_check()

        sk_rels, sk_q, sk_k = _shared_k_relations()
        q, k, v = self._qkv(sk_q, sk_k, group=2)
        sink = torch.randn(2, q.shape[1], device=self.device, dtype=torch.float32)
        do = torch.randn_like(q)

        def run_public(sm_margin: int):
            q_i, k_i, v_i, s_i = (
                t.detach().clone().requires_grad_() for t in (q, k, v, sink)
            )
            out, meta = flex_flash_attn_func(
                q_i,
                k_i,
                v_i,
                **self._range_args(sk_rels),
                sink=s_i,
                deterministic=True,
                sm_margin=sm_margin,
            )
            out.backward(do)
            return out.detach(), meta.lse, q_i.grad, k_i.grad, v_i.grad, s_i.grad

        with _with_2cta(True):
            _bitwise_across_margins(
                run_public, self._margins(), _PUBLIC_API_NAMES, "[public_api_sink]"
            )

        ov_args = self._range_args(ov_rels)
        f0 = self._qkv(ov_q, ov_k, group=4)
        f1 = self._qkv(ov_q, ov_k, group=4)
        assert_cuda_graph_matches_eager(
            lambda qi, ki, vi: _flex_flash_attn_fwd(
                qi,
                ki,
                vi,
                **ov_args,
                disable_fwd_atomic_reduction=False,
                deterministic=True,
                pack_gqa=True,
            ),
            f0,
            [f1, f0, f1],
            test_case="[cuda_graph_packed_fwd]",
        )

        dis_t2, _ = self._bwd_inputs(
            dis_rels, dis_q, dis_k, group=2, disable_fwd_atomic=True
        )
        with _with_2cta(True):
            assert_cuda_graph_matches_eager(
                lambda *ts: self._det_bwd(
                    ts,
                    dis_args,
                    disable_bwd_dkv_atomic_reduction=False,
                    disable_fwd_atomic_reduction=True,
                    cat_gqa=True,
                ),
                dis_t,
                [dis_t2, dis_t, dis_t2],
                test_case="[cuda_graph_bwd_disjoint]",
            )


if __name__ == "__main__":
    run_tests()
