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
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import (
    MHA_HD64,
    MHA_HD128,
    SM100_FAMILY_REQUIRED,
    AttnTensors,
    CuteDSLDistTestBase,
    HeadConfig,
    Relations,
    assert_close_to_torch_ref,
    assert_cuda_graph_matches_eager,
    is_sm100_family,
    make_qkv,
    mask_types_arg,
    ranges_from_lengths,
    seed_test_case,
    sync_debug_error,
    to_ranges_tensor,
)
from magi_attention.testing.dist_common import with_run_in_mp
from magi_attention.testing.precision import assert_equal

# Unique-writer dK/dV, which the backward merge requires, is MHA-only.
DTYPE = torch.bfloat16

# (heads, mask pattern, precomputed plan): one on-the-fly plan and one
# precomputed plan, at both supported head dims.
RANGE_MERGE_CASES = [
    (MHA_HD64, "full", False),
    (MHA_HD128, "mixed", True),
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
    def test_range_merge_fwd_bwd(self, case: tuple[HeadConfig, str, bool]):
        """``range_merge=True`` plans on the fly; a precomputed
        :class:`RangeMergePlan` carries both the Q- and the K-merge.

        The forward and backward are called separately: O has one writer per
        row after the Q-merge, but the relations repeat q ranges, so dQ keeps
        its reducing path (``disable_fwd_atomic_reduction`` only in the
        forward).
        """
        heads, pattern, precomputed = case
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
        plan = plan_range_merge(q_ranges_t, k_ranges_t, mask_types)
        range_merge: bool | RangeMergePlan = (True, plan)[precomputed]

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
        self.assertIsNotNone(plan.unique_count)
        assert plan.unique_count is not None
        num_groups = int(plan.unique_count.item())
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
        self.assertEqual(len(got), 4)
        self.assertEqual(
            sorted(got, key=lambda g: g[0]), sorted(by_outer, key=lambda g: g[0])
        )

    def test_empty_relation_table(self):
        device = torch.cuda.current_device()
        outer = torch.empty((0, 2), dtype=torch.int32, device=device)
        inner = torch.empty((0, 2), dtype=torch.int32, device=device)
        mask_types = torch.empty((0,), dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        with sync_debug_error():
            plan = merge_qk_ranges(outer, inner, mask_types)
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
        with sync_debug_error():
            plan = merge_qk_ranges(outer, inner, mask_types)
            full_plan = plan_range_merge(outer, inner, mask_types)
            bwd_plan = plan_range_merge_bwd(outer, inner, mask_types)
        assert plan.unique_count is not None and full_plan.unique_count is not None
        assert bwd_plan.unique_count is not None
        self.assertEqual(int(plan.unique_count.item()), 3)
        self.assertEqual(int(full_plan.unique_count.item()), 3)
        self.assertEqual(int(bwd_plan.unique_count.item()), 4)


_CUTEDSL_RM_SYNC_GRAPH_CASES = [
    (64, False, 1e-2),
    (64, True, 0.0),
    (128, True, 0.0),
]

# [96, 96] in covered K [48, 177) exercises hole-cleanup sentinel padding.
_SYNC_GRAPH_RELATIONS_1: Relations = [
    ([32, 161], [48, 177], MT_MAP.full),
    ([32, 161], [224, 481], MT_MAP.causal),
    ([32, 161], [96, 96], MT_MAP.full),
    ([192, 321], [48, 177], MT_MAP.inv_causal),
    ([192, 321], [224, 481], MT_MAP.bi_causal),
    ([384, 384], [48, 177], MT_MAP.full),
    ([384, 384], [500, 500], MT_MAP.full),
]

# Second schedule keeps R=7 with new active group counts for mNumBatches.
_SYNC_GRAPH_RELATIONS_2: Relations = [
    ([16, 208], [32, 160], MT_MAP.full),
    ([16, 208], [176, 336], MT_MAP.causal),
    ([16, 208], [352, 512], MT_MAP.inv_causal),
    ([16, 208], [200, 200], MT_MAP.full),
    ([240, 432], [32, 160], MT_MAP.inv_causal),
    ([240, 432], [176, 336], MT_MAP.full),
    ([240, 432], [352, 512], MT_MAP.bi_causal),
]


def _unpack_relations(
    relations: Relations, device: int | str
) -> tuple[
    list[list[int]],
    list[list[int]],
    list[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_list = [q_r for q_r, _, _ in relations]
    k_list = [k_r for _, k_r, _ in relations]
    m_list = [m for _, _, m in relations]
    return (
        q_list,
        k_list,
        m_list,
        to_ranges_tensor(q_list, device),
        to_ranges_tensor(k_list, device),
        torch.tensor(m_list, dtype=torch.int32, device=device),
    )


def _replay_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    relations: Relations,
) -> tuple[torch.Tensor, ...]:
    _, _, _, qr, kr, am = _unpack_relations(relations, q.device)
    return (
        torch.randn_like(q),
        torch.randn_like(k),
        torch.randn_like(v),
        torch.randn_like(do),
        qr,
        kr,
        am,
    )


def _run_merged_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    qr: torch.Tensor,
    kr: torch.Tensor,
    am: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    deterministic: bool,
    cat_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out, lse = _flex_flash_attn_fwd(
        q,
        k,
        v,
        q_ranges=qr,
        k_ranges=kr,
        mask_types=am,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        disable_fwd_atomic_reduction=True,
        fill_uncovered_rows=True,
        range_merge=True,
        deterministic=deterministic,
    )
    dq, dk, dv, _ = _flex_flash_attn_bwd(
        q,
        k,
        v,
        out,
        lse,
        do,
        q_ranges=qr,
        k_ranges=kr,
        mask_types=am,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        disable_fwd_atomic_reduction=True,
        disable_bwd_dkv_atomic_reduction=True,
        range_merge=True,
        deterministic=deterministic,
        cat_gqa=cat_gqa,
        dq_type=DTYPE,
        dk_type=DTYPE,
        dv_type=DTYPE,
    )
    return out, lse, dq, dk, dv


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
        self, case: tuple[int, bool, float]
    ):
        head_dim, deterministic, grad_tol = case
        torch.manual_seed(1234 + head_dim + int(deterministic) * 10)
        hq, hk = 4, 2
        total_q, total_k = 448, 544
        max_seqlen_q, max_seqlen_k = 256, 384

        q_list_1, k_list_1, m_list_1, qr, kr, am = _unpack_relations(
            _SYNC_GRAPH_RELATIONS_1, "cuda"
        )
        q_list_2, k_list_2, m_list_2, _, _, _ = _unpack_relations(
            _SYNC_GRAPH_RELATIONS_2, "cuda"
        )

        q = torch.randn(
            total_q, hq, head_dim, dtype=DTYPE, device="cuda", requires_grad=True
        )
        k = torch.randn(
            total_k, hk, head_dim, dtype=DTYPE, device="cuda", requires_grad=True
        )
        v = torch.randn(
            total_k, hk, head_dim, dtype=DTYPE, device="cuda", requires_grad=True
        )
        do = torch.randn(total_q, hq, head_dim, dtype=DTYPE, device="cuda")

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
            DTYPE,
        )
        torch.cuda.synchronize()

        with sync_debug_error():
            out, dq, dk, dv = run_autograd(q, k, v, do)
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

        tag = f"[range_merge_sync][hd={head_dim}][det={deterministic}]"
        assert_close_to_torch_ref(
            q=q.detach(),
            k=k.detach(),
            v=v.detach(),
            do=do,
            actual=AttnTensors(
                out=out.detach().masked_fill(~covered_q[:, None, None], 0),
                dq=dq,
                dk=dk,
                dv=dv,
            ),
            q_ranges=q_list_1,
            k_ranges=k_list_1,
            attn_type_map=m_list_1,
            test_case=f"{tag}[eager]",
        )

        def run_fwd_bwd(q_in, k_in, v_in, do_in, qr_in, kr_in, am_in):
            return _run_merged_fwd_bwd(
                q_in,
                k_in,
                v_in,
                do_in,
                qr_in,
                kr_in,
                am_in,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                deterministic=deterministic,
                cat_gqa=True,
            )

        replay_inputs = [
            _replay_input(q, k, v, do, _SYNC_GRAPH_RELATIONS_1),
            _replay_input(q, k, v, do, _SYNC_GRAPH_RELATIONS_2),
        ]

        def check_replay(
            _step: int, got: Sequence[torch.Tensor], eager: Sequence[torch.Tensor]
        ) -> None:
            # out/lse have one writer per row after the merge, so they are
            # bitwise stable in both modes; dQ reorders its reduction when
            # deterministic=False, hence the per-case tolerance.
            assert_equal(got[0], eager[0], print_rank=-1, print_no_mismatch=False)
            assert_equal(got[1], eager[1], print_rank=-1, print_no_mismatch=False)
            torch.testing.assert_close(
                got[2].float(), eager[2].float(), atol=grad_tol, rtol=grad_tol
            )
            torch.testing.assert_close(
                got[3].float(), eager[3].float(), atol=grad_tol, rtol=grad_tol
            )
            torch.testing.assert_close(
                got[4].float(), eager[4].float(), atol=grad_tol, rtol=grad_tol
            )

        replayed = assert_cuda_graph_matches_eager(
            run_fwd_bwd,
            (q.detach(), k.detach(), v.detach(), do.detach(), qr, kr, am),
            replay_inputs,
            check=check_replay,
        )
        assert_close_to_torch_ref(
            q=replay_inputs[0][0],
            k=replay_inputs[0][1],
            v=replay_inputs[0][2],
            do=replay_inputs[0][3],
            actual=AttnTensors(
                out=replayed[0][0],
                dq=replayed[0][2],
                dk=replayed[0][3],
                dv=replayed[0][4],
            ),
            q_ranges=q_list_1,
            k_ranges=k_list_1,
            attn_type_map=m_list_1,
            test_case=f"{tag}[replay=0]",
        )
        assert_close_to_torch_ref(
            q=replay_inputs[1][0],
            k=replay_inputs[1][1],
            v=replay_inputs[1][2],
            do=replay_inputs[1][3],
            actual=AttnTensors(
                out=replayed[1][0],
                dq=replayed[1][2],
                dk=replayed[1][3],
                dv=replayed[1][4],
            ),
            q_ranges=q_list_2,
            k_ranges=k_list_2,
            attn_type_map=m_list_2,
            test_case=f"{tag}[replay=1]",
        )

    @parameterize("deterministic", [False, True])
    def test_all_empty_degenerate_ranges_sync_free_and_cuda_graph(
        self, deterministic: bool
    ):
        torch.manual_seed(99 + int(deterministic))
        total_q, total_k, heads, head_dim = 256, 256, 4, 64
        qr = torch.tensor(
            [[64, 64], [64, 64], [192, 192]], dtype=torch.int32, device="cuda"
        )
        kr = torch.tensor(
            [[32, 32], [128, 128], [128, 128]], dtype=torch.int32, device="cuda"
        )
        am = torch.zeros(3, dtype=torch.int32, device="cuda")

        q = torch.randn(total_q, heads, head_dim, dtype=DTYPE, device="cuda")
        k = torch.randn(total_k, heads, head_dim, dtype=DTYPE, device="cuda")
        v = torch.randn(total_k, heads, head_dim, dtype=DTYPE, device="cuda")
        do = torch.randn(total_q, heads, head_dim, dtype=DTYPE, device="cuda")

        def run_step(q_in, k_in, v_in, do_in):
            return _run_merged_fwd_bwd(
                q_in,
                k_in,
                v_in,
                do_in,
                qr,
                kr,
                am,
                max_seqlen_q=0,
                max_seqlen_k=0,
                deterministic=deterministic,
            )

        run_step(q, k, v, do)
        _poison_caching_allocator([(total_q, heads, head_dim)] * 4, DTYPE)
        torch.cuda.synchronize()

        with sync_debug_error():
            out, lse, dq, dk, dv = run_step(q, k, v, do)
        torch.cuda.synchronize()

        def assert_zero_neginf(t_out, t_lse, t_dq, t_dk, t_dv) -> None:
            self.assertEqual(float(t_out.abs().max().item()), 0.0)
            self.assertTrue(bool(torch.isneginf(t_lse).all()))
            self.assertEqual(float(t_dq.abs().max().item()), 0.0)
            self.assertEqual(float(t_dk.abs().max().item()), 0.0)
            self.assertEqual(float(t_dv.abs().max().item()), 0.0)

        assert_zero_neginf(out, lse, dq, dk, dv)
        g_out, g_lse, g_dq, g_dk, g_dv = assert_cuda_graph_matches_eager(
            run_step, (q, k, v, do), [(q, k, v, do)]
        )[0]
        assert_zero_neginf(g_out, g_lse, g_dq, g_dk, g_dv)


if __name__ == "__main__":
    run_tests()
