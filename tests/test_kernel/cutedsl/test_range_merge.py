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

import unittest
from collections.abc import Sequence
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

from magi_attention import magi_attn_ext
from magi_attention.common import AttnRanges
from magi_attention.kernel.cutedsl import (
    MT_MAP,
    RangeMergePlan,
)
from magi_attention.kernel.cutedsl import (
    flex_flash_attn_func as cutedsl_flex_flash_attn_func,
)
from magi_attention.kernel.cutedsl import plan_range_merge, plan_range_merge_bwd
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import (
    bwd_range_merge_arg,
    merge_qk_ranges,
)
from magi_attention.testing import parameterize, ref_attn_func
from magi_attention.testing.cutedsl_utils import (
    MHA_HD64,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    assert_close_to_torch_ref,
    assert_cuda_graph_matches_eager,
    is_sm100_family,
    make_qkv,
    mask_types_arg,
    ranges_from_lengths,
    seed_test_case,
    to_ranges_tensor,
)
from magi_attention.testing.dist_common import with_run_in_mp
from magi_attention.testing.precision import assert_equal
from magi_attention.utils import make_attn_mask_from_ffa_args

# Unique-writer dK/dV, which the backward merge requires, is MHA-only.
DTYPE = torch.bfloat16

# (heads, mask pattern): one on-the-fly plan and one precomputed plan, at both
# supported head dims.
RANGE_MERGE_CASES = [
    (MHA_HD64, "full"),
    (MHA_HD128, "mixed"),
]

# Lengths off the tile sizes leave partial tiles inside every group.
Q_GROUP_LENGTHS = (257, 129, 193)
K_GROUP_LENGTHS = (129, 257, 193)
# (q group, k group) of each relation. Each q group pairs with several k
# groups (merged by the forward) and each k group with several q groups
# (merged by the backward). Distinct groups are disjoint, so the merged
# groups give O and dK/dV one writer per row.
RELATION_GROUPS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]

MASK_PATTERNS = {
    "full": [MT_MAP.full] * len(RELATION_GROUPS),
    # bi_causal / causal pairs leave K tiles without any Q block.
    "mixed": [
        MT_MAP.bi_causal,
        MT_MAP.causal,
        MT_MAP.full,
        MT_MAP.full,
        MT_MAP.inv_causal,
        MT_MAP.full,
    ],
}


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestRangeMerge(CuteDSLDistTestBase):
    @with_run_in_mp
    @parameterize("case", RANGE_MERGE_CASES)
    def test_range_merge_fwd_bwd(self, case: tuple[HeadConfig, str]):
        """``range_merge=True`` plans on the fly; a precomputed
        :class:`RangeMergePlan` carries both the Q- and the K-merge.

        The forward and backward are called separately: O has one writer per
        row after the Q-merge, but the relations repeat q ranges, so dQ keeps
        its reducing path (``disable_fwd_atomic_reduction`` only in the
        forward).
        """
        heads, pattern = case
        test_case = f"[test_range_merge_fwd_bwd][{pattern}][{heads.name}]"
        seed_test_case(test_case)
        q_groups = ranges_from_lengths(Q_GROUP_LENGTHS)
        k_groups = ranges_from_lengths(K_GROUP_LENGTHS)
        q_ranges = [q_groups[q_group] for q_group, _ in RELATION_GROUPS]
        k_ranges = [k_groups[k_group] for _, k_group in RELATION_GROUPS]
        attn_type_map = MASK_PATTERNS[pattern]
        q, k, v = (
            t.detach()
            for t in make_qkv(
                (sum(Q_GROUP_LENGTHS),),
                (sum(K_GROUP_LENGTHS),),
                heads,
                DTYPE,
                self.device,
            )
        )
        q_ranges_t = to_ranges_tensor(q_ranges, self.device)
        k_ranges_t = to_ranges_tensor(k_ranges, self.device)
        mask_types = mask_types_arg(attn_type_map, self.device)
        range_merge: bool | RangeMergePlan = (
            True
            if pattern == "full"
            else plan_range_merge(q_ranges_t, k_ranges_t, mask_types)
        )

        out, lse = _flex_flash_attn_fwd(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types,
            max_seqlen_q=max(Q_GROUP_LENGTHS),
            max_seqlen_k=max(K_GROUP_LENGTHS),
            disable_fwd_atomic_reduction=True,
            range_merge=range_merge,
        )
        do = torch.randn_like(out)
        dq, dk, dv, _ = _flex_flash_attn_bwd(
            q,
            k,
            v,
            out,
            lse,
            do,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            mask_types=mask_types,
            max_seqlen_q=max(Q_GROUP_LENGTHS),
            max_seqlen_k=max(K_GROUP_LENGTHS),
            disable_bwd_dkv_atomic_reduction=True,
            range_merge=bwd_range_merge_arg(range_merge),
        )

        assert_close_to_torch_ref(
            q=q,
            k=k,
            v=v,
            do=do,
            actual=AttnTensors(out=out, dq=dq, dk=dk, dv=dv),
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            test_case=test_case,
        )


def _plan_groups(
    merged_outer: torch.Tensor,
    sorted_inner: torch.Tensor,
    sorted_mask: torch.Tensor,
    cu_batches: torch.Tensor,
    num_groups: int | None = None,
) -> list[tuple[list[int], list[tuple[list[int], int]]]]:
    cu = cu_batches.tolist()
    inner, mask = sorted_inner.tolist(), sorted_mask.tolist()
    table_slots = merged_outer.shape[0]
    active_groups = table_slots if num_groups is None else num_groups
    assert len(cu) == table_slots + 1, (len(cu), table_slots)
    assert 0 <= active_groups <= table_slots, (active_groups, table_slots)
    assert cu[0] == 0 and cu[active_groups] == len(inner), (cu, active_groups)
    assert all(b > a for a, b in zip(cu[:active_groups], cu[1 : active_groups + 1])), cu
    assert all(c == len(inner) for c in cu[active_groups:]), cu
    return [
        (
            merged_outer[g].tolist(),
            list(zip(inner[cu[g] : cu[g + 1]], mask[cu[g] : cu[g + 1]])),
        )
        for g in range(active_groups)
    ]


class TestRangeMergePlan(TestCase):
    def _merge(self, outer, inner, mask_types):
        device = torch.cuda.current_device()
        outer_t = torch.tensor(outer, dtype=torch.int32, device=device).reshape(-1, 2)
        inner_t = torch.tensor(inner, dtype=torch.int32, device=device).reshape(-1, 2)
        mask_t = torch.tensor(mask_types, dtype=torch.int32, device=device)
        plan = merge_qk_ranges(outer_t, inner_t, mask_t)
        num_rows = outer_t.shape[0]
        self.assertIsNotNone(plan.unique_count)
        assert plan.unique_count is not None
        self.assertEqual(tuple(plan.unique_count.shape), (1,))
        self.assertEqual(plan.unique_count.dtype, torch.int32)
        num_groups = int(plan.unique_count.item())
        self.assertEqual(tuple(plan.merged_outer_ranges.shape), (num_rows, 2))
        self.assertEqual(tuple(plan.cu_batches.shape), (num_rows + 1,))
        self.assertEqual(tuple(plan.merged_outer_ranges_by_start.shape), (num_rows, 2))
        self.assertTrue(bool((plan.merged_outer_ranges[num_groups:] == 0).all()))
        return _plan_groups(
            plan.merged_outer_ranges,
            plan.sorted_inner_ranges,
            plan.sorted_mask_types,
            plan.cu_batches,
            num_groups=num_groups,
        )

    def test_groups_keep_the_rows_of_the_outer_sorted_grouping(self):
        outer = [[0, 256], [256, 512], [0, 256], [512, 513], [256, 512], [513, 640]]
        inner = [[0, 128], [0, 256], [128, 300], [0, 4096], [300, 301], [0, 640]]
        mask_types = [0, 1, 0, 3, 2, 1]
        device = torch.cuda.current_device()
        outer_t, inner_t, mask_t = (
            torch.tensor(x, dtype=torch.int32, device=device)
            for x in (outer, inner, mask_types)
        )
        sorted_idx, is_sorted = magi_attn_ext.argsort_ranges(outer_t)
        (
            sorted_outer,
            sorted_inner,
            sorted_mask,
        ) = magi_attn_ext.reorder_ranges_and_attn_type_maps(
            outer_t, inner_t, mask_t, sorted_idx, is_sorted
        )
        merged, cu, count = magi_attn_ext.unique_consecutive_pairs(sorted_outer)
        count = int(count.item())
        by_outer = _plan_groups(
            merged[:count], sorted_inner, sorted_mask, cu[: count + 1]
        )
        got = self._merge(outer, inner, mask_types)
        self.assertEqual(
            [g[0] for g in got], [[512, 513], [513, 640], [0, 256], [256, 512]]
        )
        self.assertEqual(
            sorted(got, key=lambda g: g[0]), sorted(by_outer, key=lambda g: g[0])
        )

    def test_empty_relation_table(self):
        device = torch.cuda.current_device()
        outer = torch.empty((0, 2), dtype=torch.int32, device=device)
        inner = torch.empty((0, 2), dtype=torch.int32, device=device)
        mask_types = torch.empty((0,), dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan = merge_qk_ranges(outer, inner, mask_types)
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
        self.assertEqual(tuple(plan.merged_outer_ranges.shape), (0, 2))
        self.assertEqual(tuple(plan.cu_batches.shape), (1,))
        self.assertIsNotNone(plan.unique_count)
        assert plan.unique_count is not None
        self.assertEqual(int(plan.unique_count.item()), 0)

    def test_merge_runs_without_host_sync(self):
        device = torch.cuda.current_device()
        outer = torch.tensor(
            [[0, 64], [64, 65], [0, 64], [128, 128]],
            dtype=torch.int32,
            device=device,
        )
        inner = torch.tensor(
            [[0, 64], [0, 4096], [64, 128], [32, 32]],
            dtype=torch.int32,
            device=device,
        )
        mask_types = torch.zeros(4, dtype=torch.int32, device=device)
        plan_range_merge(outer, inner, mask_types)
        torch.cuda.synchronize()
        prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan = merge_qk_ranges(outer, inner, mask_types)
            full_plan = plan_range_merge(outer, inner, mask_types)
            bwd_plan = plan_range_merge_bwd(outer, inner, mask_types)
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
        assert plan.unique_count is not None and full_plan.unique_count is not None
        assert bwd_plan.unique_count is not None
        self.assertEqual(int(plan.unique_count.item()), 3)
        self.assertEqual(int(full_plan.unique_count.item()), 3)
        self.assertEqual(int(bwd_plan.unique_count.item()), 4)


_CUTEDSL_RM_SYNC_GRAPH_CASES = [
    (64, False, 0, 1e-2),
    (64, False, 8, 1e-2),
    (64, True, 0, 0.0),
    (128, False, 0, 1e-2),
    (128, True, 0, 0.0),
]


def _poison_caching_allocator(
    shapes: list[tuple[int, ...]], dtype: torch.dtype
) -> None:
    blocks = [
        torch.full(shape, float("nan"), dtype=dtype, device="cuda") for shape in shapes
    ]
    del blocks


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestCuteDSLRangeMergeSyncAndGraph(TestCase):
    @parameterize("case", _CUTEDSL_RM_SYNC_GRAPH_CASES)
    def test_range_merge_fwd_bwd_sync_free_and_cuda_graph(
        self, case: tuple[int, bool, int, float]
    ):
        head_dim, deterministic, sm_margin, grad_tol = case
        torch.manual_seed(1234 + head_dim + int(deterministic) * 10 + sm_margin)
        dtype = torch.bfloat16
        hq, hk = 4, 2
        total_q, total_k = 448, 544
        max_seqlen_q, max_seqlen_k = 256, 384

        # [96, 96] in covered K [48, 177) exercises hole-cleanup sentinel padding.
        q_ranges_1 = [
            [32, 161],
            [32, 161],
            [32, 161],
            [192, 321],
            [192, 321],
            [384, 384],
            [384, 384],
        ]
        k_ranges_1 = [
            [48, 177],
            [224, 481],
            [96, 96],
            [48, 177],
            [224, 481],
            [48, 177],
            [500, 500],
        ]
        mask_types_1 = [
            MT_MAP.full,
            MT_MAP.causal,
            MT_MAP.full,
            MT_MAP.inv_causal,
            MT_MAP.bi_causal,
            MT_MAP.full,
            MT_MAP.full,
        ]

        # Second schedule keeps R=7 with new active group counts for mNumBatches.
        q_ranges_2 = [
            [16, 208],
            [16, 208],
            [16, 208],
            [16, 208],
            [240, 432],
            [240, 432],
            [240, 432],
        ]
        k_ranges_2 = [
            [32, 160],
            [176, 336],
            [352, 512],
            [200, 200],
            [32, 160],
            [176, 336],
            [352, 512],
        ]
        mask_types_2 = [
            MT_MAP.full,
            MT_MAP.causal,
            MT_MAP.inv_causal,
            MT_MAP.full,
            MT_MAP.inv_causal,
            MT_MAP.full,
            MT_MAP.bi_causal,
        ]

        qr = torch.tensor(q_ranges_1, dtype=torch.int32, device="cuda")
        kr = torch.tensor(k_ranges_1, dtype=torch.int32, device="cuda")
        am = torch.tensor(mask_types_1, dtype=torch.int32, device="cuda")

        q = torch.randn(
            total_q, hq, head_dim, dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            total_k, hk, head_dim, dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            total_k, hk, head_dim, dtype=dtype, device="cuda", requires_grad=True
        )
        do = torch.randn(total_q, hq, head_dim, dtype=dtype, device="cuda")

        def run_autograd(q_in, k_in, v_in, do_in):
            out, _ = cutedsl_flex_flash_attn_func(
                q_in,
                k_in,
                v_in,
                q_ranges=qr,
                k_ranges=kr,
                mask_types=am,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                deterministic=deterministic,
                sm_margin=sm_margin,
                cat_gqa=True,
            )
            dq_out, dk_out, dv_out = torch.autograd.grad(out, (q_in, k_in, v_in), do_in)
            return out, dq_out, dk_out, dv_out

        run_autograd(q, k, v, do)
        torch.cuda.synchronize()

        _poison_caching_allocator(
            [
                (total_q, hq, head_dim),
                (total_k, hk, head_dim),
                (total_k, hk, head_dim),
            ],
            dtype,
        )
        torch.cuda.synchronize()

        prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out, dq, dk, dv = run_autograd(q, k, v, do)
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
        torch.cuda.synchronize()

        self.assertFalse(bool(torch.isnan(dq).any()), "dq contains NaN")
        self.assertFalse(bool(torch.isnan(dk).any()), "dk contains NaN")
        self.assertFalse(bool(torch.isnan(dv).any()), "dv contains NaN")

        covered_q = torch.zeros(total_q, dtype=torch.bool, device="cuda")
        covered_q[32:161] = True
        covered_q[192:321] = True
        covered_k = torch.zeros(total_k, dtype=torch.bool, device="cuda")
        covered_k[48:177] = True
        covered_k[224:481] = True
        self.assertEqual(float(dq[~covered_q].abs().max().item()), 0.0)
        self.assertEqual(float(dk[~covered_k].abs().max().item()), 0.0)
        self.assertEqual(float(dv[~covered_k].abs().max().item()), 0.0)
        self.assertGreater(float(dk[96:177].abs().max().item()), 0.0)
        self.assertGreater(float(dv[96:177].abs().max().item()), 0.0)

        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        v_ref = v.detach().float().requires_grad_(True)
        mask_1 = make_attn_mask_from_ffa_args(
            q_ranges=AttnRanges.from_ranges(q_ranges_1),
            k_ranges=AttnRanges.from_ranges(k_ranges_1),
            attn_type_map=mask_types_1,
            total_seqlen_q=total_q,
            total_seqlen_k=total_k,
        )
        out_ref, _ = ref_attn_func(
            q_ref, k_ref, v_ref, mask_1, backend="torch", high_precision=True
        )
        out_ref.backward(do.float())
        torch.testing.assert_close(
            out[covered_q].float(), out_ref[covered_q], atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(dq.float(), q_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(dk.float(), k_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(dv.float(), v_ref.grad, atol=3e-2, rtol=3e-2)

        def run_fwd_bwd(q_in, k_in, v_in, do_in, qr_in, kr_in, am_in):
            out_g, lse_g = _flex_flash_attn_fwd(
                q_in,
                k_in,
                v_in,
                q_ranges=qr_in,
                k_ranges=kr_in,
                mask_types=am_in,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                disable_fwd_atomic_reduction=True,
                fill_uncovered_rows=True,
                range_merge=True,
                deterministic=deterministic,
                sm_margin=sm_margin,
            )
            dq_g, dk_g, dv_g, _ = _flex_flash_attn_bwd(
                q_in,
                k_in,
                v_in,
                out_g,
                lse_g,
                do_in,
                q_ranges=qr_in,
                k_ranges=kr_in,
                mask_types=am_in,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                deterministic=deterministic,
                sm_margin=sm_margin,
                cat_gqa=True,
                dq_type=dtype,
                dk_type=dtype,
                dv_type=dtype,
            )
            return out_g, lse_g, dq_g, dk_g, dv_g

        schedules = (
            (q_ranges_1, k_ranges_1, mask_types_1),
            (q_ranges_2, k_ranges_2, mask_types_2),
        )
        replay_inputs = [
            (
                torch.randn_like(q),
                torch.randn_like(k),
                torch.randn_like(v),
                torch.randn_like(do),
                torch.tensor(q_list, dtype=torch.int32, device="cuda"),
                torch.tensor(k_list, dtype=torch.int32, device="cuda"),
                torch.tensor(m_list, dtype=torch.int32, device="cuda"),
            )
            for q_list, k_list, m_list in schedules
        ]

        def check_replay(
            _step: int, got: Sequence[torch.Tensor], eager: Sequence[torch.Tensor]
        ) -> None:
            for idx in (0, 1):
                assert_equal(
                    got[idx], eager[idx], print_rank=-1, print_no_mismatch=False
                )
            for idx in (2, 3, 4):
                if grad_tol == 0.0:
                    assert_equal(
                        got[idx], eager[idx], print_rank=-1, print_no_mismatch=False
                    )
                else:
                    torch.testing.assert_close(
                        got[idx].float(),
                        eager[idx].float(),
                        atol=grad_tol,
                        rtol=grad_tol,
                    )

        replayed = assert_cuda_graph_matches_eager(
            run_fwd_bwd,
            (q.detach(), k.detach(), v.detach(), do.detach(), qr, kr, am),
            replay_inputs,
            check=check_replay,
        )
        for (
            (q_list, k_list, m_list),
            inputs,
            (got_out, _, got_dq, got_dk, got_dv),
        ) in zip(schedules, replay_inputs, replayed, strict=True):
            new_q, new_k, new_v, new_do = inputs[:4]
            q_r = new_q.detach().float().requires_grad_(True)
            k_r = new_k.detach().float().requires_grad_(True)
            v_r = new_v.detach().float().requires_grad_(True)
            mask_r = make_attn_mask_from_ffa_args(
                q_ranges=AttnRanges.from_ranges(q_list),
                k_ranges=AttnRanges.from_ranges(k_list),
                attn_type_map=m_list,
                total_seqlen_q=total_q,
                total_seqlen_k=total_k,
            )
            out_r, _ = ref_attn_func(
                q_r, k_r, v_r, mask_r, backend="torch", high_precision=True
            )
            out_r.backward(new_do.float())
            torch.testing.assert_close(got_out.float(), out_r, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(got_dq.float(), q_r.grad, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(got_dk.float(), k_r.grad, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(got_dv.float(), v_r.grad, atol=3e-2, rtol=3e-2)

    @parameterize("deterministic", [False, True])
    def test_all_empty_degenerate_ranges_sync_free_and_cuda_graph(
        self, deterministic: bool
    ):
        torch.manual_seed(99 + int(deterministic))
        dtype = torch.bfloat16
        total_q, total_k, heads, head_dim = 256, 256, 4, 64
        qr = torch.tensor(
            [[64, 64], [64, 64], [192, 192]], dtype=torch.int32, device="cuda"
        )
        kr = torch.tensor(
            [[32, 32], [128, 128], [128, 128]], dtype=torch.int32, device="cuda"
        )
        am = torch.zeros(3, dtype=torch.int32, device="cuda")

        q = torch.randn(total_q, heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(total_k, heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(total_k, heads, head_dim, dtype=dtype, device="cuda")
        do = torch.randn(total_q, heads, head_dim, dtype=dtype, device="cuda")

        def run_step(q_in, k_in, v_in, do_in):
            out_s, lse_s = _flex_flash_attn_fwd(
                q_in,
                k_in,
                v_in,
                q_ranges=qr,
                k_ranges=kr,
                mask_types=am,
                max_seqlen_q=0,
                max_seqlen_k=0,
                disable_fwd_atomic_reduction=True,
                fill_uncovered_rows=True,
                range_merge=True,
                deterministic=deterministic,
            )
            dq_s, dk_s, dv_s, _ = _flex_flash_attn_bwd(
                q_in,
                k_in,
                v_in,
                out_s,
                lse_s,
                do_in,
                q_ranges=qr,
                k_ranges=kr,
                mask_types=am,
                max_seqlen_q=0,
                max_seqlen_k=0,
                disable_fwd_atomic_reduction=True,
                disable_bwd_dkv_atomic_reduction=True,
                range_merge=True,
                deterministic=deterministic,
                dq_type=dtype,
                dk_type=dtype,
                dv_type=dtype,
            )
            return out_s, lse_s, dq_s, dk_s, dv_s

        run_step(q, k, v, do)
        _poison_caching_allocator([(total_q, heads, head_dim)] * 4, dtype)
        torch.cuda.synchronize()

        prev_sync_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            out, lse, dq, dk, dv = run_step(q, k, v, do)
        finally:
            torch.cuda.set_sync_debug_mode(prev_sync_debug_mode)
        torch.cuda.synchronize()

        def assert_zero_neginf(t_out, t_lse, t_dq, t_dk, t_dv) -> None:
            self.assertEqual(float(t_out.abs().max().item()), 0.0)
            self.assertTrue(bool(torch.isneginf(t_lse).all()))
            self.assertEqual(float(t_dq.abs().max().item()), 0.0)
            self.assertEqual(float(t_dk.abs().max().item()), 0.0)
            self.assertEqual(float(t_dv.abs().max().item()), 0.0)

        assert_zero_neginf(out, lse, dq, dk, dv)
        (g_out, g_lse, g_dq, g_dk, g_dv) = assert_cuda_graph_matches_eager(
            run_step, (q, k, v, do), [(q, k, v, do)]
        )[0]
        assert_zero_neginf(g_out, g_lse, g_dq, g_dk, g_dv)


if __name__ == "__main__":
    run_tests()
