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

import random
import unittest
from unittest import TestCase

import torch

from magi_attention import magi_attn_ext
from magi_attention.kernel.cutedsl import (
    MT_MAP,
)
from magi_attention.kernel.cutedsl import (
    flex_flash_attn_func as cutedsl_flex_flash_attn_func,
)
from magi_attention.kernel.cutedsl import plan_range_merge, plan_range_merge_bwd
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.range_merge import merge_qk_ranges
from magi_attention.testing import parameterize
from magi_attention.testing.cutedsl_utils import SM100_FAMILY_REQUIRED, is_sm100_family


class TestMergeRange(TestCase):
    @property
    def seed(self) -> int:
        return 42

    def merge_ranges_ref(
        self, outer_ranges: torch.Tensor, inner_ranges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_idx = torch.argsort(outer_ranges[:, 0], dim=0, stable=True)
        sorted_outer_ranges = outer_ranges[sorted_idx]
        sorted_inner_ranges = inner_ranges[sorted_idx]

        merge_outer_ranges, _, counts = torch.unique_consecutive(
            sorted_outer_ranges, dim=0, return_inverse=True, return_counts=True
        )
        range_map = torch.cumsum(counts, dim=0, dtype=torch.int32)

        return merge_outer_ranges, sorted_outer_ranges, sorted_inner_ranges, range_map

    def merge_ranges(
        self, outer_ranges: torch.Tensor, inner_ranges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_idx, is_sorted = magi_attn_ext.argsort_ranges(outer_ranges)
        (
            sorted_outer_ranges,
            sorted_inner_ranges,
            sorted_attn_type_map,
        ) = magi_attn_ext.reorder_ranges_and_attn_type_maps(
            outer_ranges, inner_ranges, None, sorted_idx, is_sorted
        )

        (
            merge_outer_ranges,
            range_map,
            unique_count,
        ) = magi_attn_ext.unique_consecutive_pairs(sorted_outer_ranges)

        range_map = range_map[1 : unique_count.item() + 1]

        return (
            merge_outer_ranges[: unique_count.item()],
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        )

    def test_simple_case(self):
        device = torch.cuda.current_device()

        outer_ranges_small = torch.tensor(
            [
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )
        inner_ranges_small = torch.tensor(
            [
                [0, 256],
                [256, 512],
                [512, 768],
                [768, 1024],
                [0, 512],
                [0, 768],
                [0, 1024],
                [0, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_small, inner_ranges_small)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_small, inner_ranges_small)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))

    def test_simple_sorted(self):
        device = torch.cuda.current_device()

        outer_ranges_sorted = torch.tensor(
            [
                [0, 256],
                [0, 256],
                [256, 512],
                [256, 512],
                [512, 768],
                [512, 768],
                [768, 1024],
                [768, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )
        inner_ranges = torch.tensor(
            [
                [0, 256],
                [0, 512],
                [256, 512],
                [0, 768],
                [512, 768],
                [0, 1024],
                [768, 1024],
                [0, 1024],
            ],
            dtype=torch.int32,
            device=device,
        )

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_sorted, inner_ranges)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_sorted, inner_ranges)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))

    def test_compilcate_case(self):
        NUM_PAIRS = 1024 * 1024
        KEY_X_MIN = 0
        KEY_X_MAX = 20
        KEY_Y_MIN = 0
        KEY_Y_MAX = 20
        device = torch.cuda.current_device()

        x_coords_tensor = torch.randint(
            low=KEY_X_MIN,
            high=KEY_X_MAX + 1,
            size=(NUM_PAIRS,),
            dtype=torch.int32,
            device=device,
        )
        y_coords_tensor = torch.randint(
            low=KEY_Y_MIN,
            high=KEY_Y_MAX + 1,
            size=(NUM_PAIRS,),
            dtype=torch.int32,
            device=device,
        )

        outer_ranges_large = torch.stack([x_coords_tensor, y_coords_tensor], dim=1)
        inner_ranges_large = outer_ranges_large.clone()

        (
            merge_outer_ranges_ref,
            sorted_outer_ranges_ref,
            sorted_inner_ranges_ref,
            range_map_ref,
        ) = self.merge_ranges_ref(outer_ranges_large, inner_ranges_large)
        (
            merge_outer_ranges,
            sorted_outer_ranges,
            sorted_inner_ranges,
            range_map,
        ) = self.merge_ranges(outer_ranges_large, inner_ranges_large)

        self.assertTrue(torch.equal(merge_outer_ranges, merge_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_outer_ranges, sorted_outer_ranges_ref))
        self.assertTrue(torch.equal(sorted_inner_ranges, sorted_inner_ranges_ref))
        self.assertTrue(torch.equal(range_map, range_map_ref))

    def _assert_unique_pairs_contract(
        self, sorted_pairs: torch.Tensor, pairs, cu, count
    ) -> None:
        """Valid groups match torch.unique_consecutive; the padded tail holds
        empty [0, 0] pairs and CSR pointers equal to n, i.e. empty groups."""
        n = sorted_pairs.shape[0]
        ref, counts = torch.unique_consecutive(sorted_pairs, dim=0, return_counts=True)
        u = ref.shape[0]
        self.assertEqual(int(count), u)
        self.assertEqual(tuple(pairs.shape), (n, 2))
        self.assertEqual(tuple(cu.shape), (n + 1,))
        self.assertTrue(torch.equal(pairs[:u], ref))
        ref_cu = torch.zeros(u + 1, dtype=torch.int32, device=cu.device)
        ref_cu[1:] = torch.cumsum(counts, dim=0)
        self.assertTrue(torch.equal(cu[: u + 1], ref_cu))
        self.assertTrue(bool((pairs[u:] == 0).all()))
        self.assertTrue(bool((cu[u:] == n).all()))

    def test_unique_consecutive_pairs_pads_tail(self):
        """Callers walk all n output slots, so the tail must be well defined
        even when the caching allocator hands back dirty memory."""
        device = torch.cuda.current_device()
        sorted_pairs = torch.tensor(
            [[0, 4], [0, 4], [0, 4], [4, 9], [4, 9], [9, 12], [9, 12], [9, 12]],
            dtype=torch.int32,
            device=device,
        )
        n = sorted_pairs.shape[0]
        for _ in range(4):
            # Freed blocks of the output sizes, filled with garbage, are what
            # the allocator reuses for the outputs below.
            dirty = [
                torch.full((n, 2), 0x7A5A5A5A, dtype=torch.int32, device=device),
                torch.full((n + 1,), -7, dtype=torch.int32, device=device),
            ]
            del dirty
            pairs, cu, count = magi_attn_ext.unique_consecutive_pairs(sorted_pairs)
            self._assert_unique_pairs_contract(sorted_pairs, pairs, cu, count)

    def test_unique_consecutive_pairs_in_cuda_graph(self):
        """Every launch goes to the caller's current stream: a launch on the
        legacy default stream would break stream capture."""
        device = torch.cuda.current_device()
        n = 1 << 12
        starts = torch.arange(n, dtype=torch.int32, device=device) // 3 * 8
        sorted_pairs = torch.stack([starts, starts + 8], dim=1)
        # Warm up on a side stream, as torch.cuda.graph requires.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            magi_attn_ext.unique_consecutive_pairs(sorted_pairs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pairs, cu, count = magi_attn_ext.unique_consecutive_pairs(sorted_pairs)
        graph.replay()
        torch.cuda.synchronize()
        self._assert_unique_pairs_contract(sorted_pairs, pairs, cu, count)


def _tiles(r: list[int]) -> int:
    return -(-(r[1] - r[0]) // 128)


def _work_tile_length(group: tuple[list[int], list[tuple[list[int], int]]]) -> int:
    return sum(_tiles(r) for r, _ in group[1])


def _launch_order_ref(
    outer: list[list[int]], inner: list[list[int]], mask_types: list[int]
) -> list[tuple[list[int], list[tuple[list[int], int]]]]:
    """Known-good RangeMerge group order: relations grouped by identical outer
    range (outer start ascending, equal starts in input order, consecutive
    equal ranges merged), then stably sorted by the bit length of the
    work-tile length (sum of the pairs' ceil(inner width / 128)), descending.
    Returns (outer range, [(inner range, mask type)]) per group."""
    order = sorted(range(len(outer)), key=lambda i: outer[i][0])
    groups: list[tuple[list[int], list[tuple[list[int], int]]]] = []
    for i in order:
        if groups and groups[-1][0] == outer[i]:
            groups[-1][1].append((inner[i], mask_types[i]))
        else:
            groups.append((outer[i], [(inner[i], mask_types[i])]))
    return sorted(
        groups,
        key=lambda g: _work_tile_length(g).bit_length(),
        reverse=True,
    )


def _plan_groups(
    merged_outer: torch.Tensor,
    sorted_inner: torch.Tensor,
    sorted_mask: torch.Tensor,
    cu_batches: torch.Tensor,
    num_groups: int | None = None,
) -> list[tuple[list[int], list[tuple[list[int], int]]]]:
    """Active groups of a merge table in table order; every active group has a pair."""
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


class TestRangeMergeLaunchOrder(TestCase):
    """merge_qk_ranges launches the merged groups heaviest work tile first."""

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

        sentinel = torch.iinfo(torch.int32).max
        by_start = plan.merged_outer_ranges_by_start.tolist()
        self.assertEqual([r[0] for r in by_start], sorted(r[0] for r in by_start))
        non_empty_active = sorted(
            [r for r in plan.merged_outer_ranges[:num_groups].tolist() if r[0] < r[1]],
            key=lambda r: r[0],
        )
        self.assertEqual(
            by_start,
            non_empty_active
            + [[sentinel, sentinel]] * (num_rows - len(non_empty_active)),
        )
        return _plan_groups(
            plan.merged_outer_ranges,
            plan.sorted_inner_ranges,
            plan.sorted_mask_types,
            plan.cu_batches,
            num_groups=num_groups,
        )

    def test_groups_follow_the_reference_order_on_random_relations(self):
        rng = random.Random(0)
        for _ in range(50):
            num_outer = rng.randint(1, 12)
            outer_pool = []
            for _ in range(num_outer):
                start = rng.randint(0, 2000)
                outer_pool.append([start, start + rng.randint(1, 600)])
            outer, inner, mask_types = [], [], []
            for _ in range(rng.randint(1, 40)):
                outer.append(list(rng.choice(outer_pool)))
                start = rng.randint(0, 4000)
                inner.append([start, start + rng.randint(0, 1500)])
                mask_types.append(rng.randint(0, 3))
            got = self._merge(outer, inner, mask_types)
            self.assertEqual(got, _launch_order_ref(outer, inner, mask_types))
            classes = [_work_tile_length(g).bit_length() for g in got]
            self.assertEqual(classes, sorted(classes, reverse=True))

    def test_groups_keep_the_rows_of_the_outer_sorted_grouping(self):
        """The permutation moves whole groups: each group holds the same rows,
        in the same order, as in the outer-sorted grouping of magi_attn_ext."""
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
            sorted(got, key=lambda g: g[0]), sorted(by_outer, key=lambda g: g[0])
        )
        self.assertEqual(
            got,
            [
                ([512, 513], [([0, 4096], 3)]),
                ([513, 640], [([0, 640], 1)]),
                ([0, 256], [([0, 128], 0), ([128, 300], 0)]),
                ([256, 512], [([0, 256], 1), ([300, 301], 2)]),
            ],
        )

    def test_a_one_row_group_against_every_key_launches_first(self):
        """A one-token Q row against all keys has the longest work tile, though
        its group is the narrowest and starts last."""
        seqlen, chunk = 16384, 1024
        outer = [[s, s + chunk] for s in range(0, seqlen - chunk, chunk)]
        inner = [[max(s - chunk, 0), s + chunk] for s, _ in outer]
        outer.append([seqlen - 1, seqlen])
        inner.append([0, seqlen])
        got = self._merge(outer, inner, [1] * len(outer))
        self.assertEqual(got[0], ([seqlen - 1, seqlen], [([0, seqlen], 1)]))

    def test_groups_within_a_factor_of_two_keep_the_outer_order(self):
        """Work-tile lengths 5 and 7 share a class and stay in outer order;
        length 8 starts the next class and launches first."""
        outer = [[0, 128], [128, 256], [256, 384]]
        inner = [[0, 640], [0, 896], [0, 1024]]
        got = self._merge(outer, inner, [0, 0, 0])
        self.assertEqual([g[0] for g in got], [[256, 384], [0, 128], [128, 256]])

    def test_empty_outer_and_zero_tile_groups_precede_padded_tail(self):
        """Active groups with empty inner ranges (tile_length == 0) or empty
        outer intervals stay in [0, G) ahead of padded tail slots [G, R), and
        merged_outer_ranges_by_start moves empty intervals to INT32_MAX."""
        outer = [[0, 256], [0, 256], [64, 64], [256, 384], [256, 384]]
        inner = [[0, 128], [128, 256], [0, 512], [96, 96], [96, 96]]
        mask_types = [0, 1, 0, 2, 3]
        got = self._merge(outer, inner, mask_types)
        self.assertEqual(got, _launch_order_ref(outer, inner, mask_types))
        self.assertEqual(
            got[-1],
            ([256, 384], [([96, 96], 2), ([96, 96], 3)]),
        )

    def test_empty_relation_table(self):
        """An empty [0, 2] relation table returns an empty plan without host sync."""
        device = torch.cuda.current_device()
        outer = torch.empty((0, 2), dtype=torch.int32, device=device)
        inner = torch.empty((0, 2), dtype=torch.int32, device=device)
        mask_types = torch.empty((0,), dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan = merge_qk_ranges(outer, inner, mask_types)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        self.assertEqual(tuple(plan.merged_outer_ranges.shape), (0, 2))
        self.assertEqual(tuple(plan.cu_batches.shape), (1,))
        self.assertIsNotNone(plan.unique_count)
        assert plan.unique_count is not None
        self.assertEqual(int(plan.unique_count.item()), 0)

    def test_merge_runs_without_host_sync(self):
        """merge_qk_ranges, plan_range_merge, and plan_range_merge_bwd run
        entirely on device without host synchronization."""
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
        # Warm up kernels before enabling strict sync debug mode.
        plan_range_merge(outer, inner, mask_types)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            plan = merge_qk_ranges(outer, inner, mask_types)
            full_plan = plan_range_merge(outer, inner, mask_types)
            bwd_plan = plan_range_merge_bwd(outer, inner, mask_types)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert plan.unique_count is not None and full_plan.unique_count is not None
        assert bwd_plan.unique_count is not None
        self.assertEqual(int(plan.unique_count.item()), 3)
        self.assertEqual(int(full_plan.unique_count.item()), 3)
        self.assertEqual(int(bwd_plan.unique_count.item()), 4)


def _make_mask(q_ranges, k_ranges, attn_type_map, total_q, total_k):
    """Build a full attention mask from ranges + attn_type_map for reference computation."""
    mask = torch.zeros(total_q, total_k, dtype=torch.bool, device="cuda")
    qa = torch.arange(total_q, device="cuda").unsqueeze(1)
    ka = torch.arange(total_k, device="cuda").unsqueeze(0)
    for (q0, q1), (k0, k1), atype in zip(q_ranges, k_ranges, attn_type_map):
        in_q = (qa >= q0) & (qa < q1)
        in_k = (ka >= k0) & (ka < k1)
        rect = in_q & in_k
        m_rel = qa - q0
        n_rel = ka - k0
        sq, sk = q1 - q0, k1 - k0
        if atype == 0:
            mask |= rect
        elif atype == 1:
            mask |= rect & ((m_rel + sk - sq) >= n_rel)
        elif atype == 2:
            mask |= rect & (n_rel >= m_rel)
        elif atype == 3:
            mask |= rect & ((m_rel + sk - sq) >= n_rel) & (n_rel >= m_rel)
    return mask


def _ref_attn(q, k, v, mask):
    """Python fp32 reference attention."""
    Q, Hq, D = q.shape
    _, Hk, _ = k.shape
    q2 = q.permute(1, 0, 2).float()
    k2 = k.repeat_interleave(Hq // Hk, dim=1).permute(1, 0, 2).float()
    v2 = v.repeat_interleave(Hq // Hk, dim=1).permute(1, 0, 2).float()
    s = torch.einsum("hmd,hnd->hmn", q2, k2) / (D**0.5)
    s = s.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = s.softmax(dim=-1)
    p = torch.where(mask.unsqueeze(0), p, torch.zeros_like(p))
    o = torch.einsum("hmn,hnd->hmd", p, v2)
    return o.permute(1, 0, 2)


_CUTEDSL_RM_SYNC_GRAPH_CASES = [
    (64, False, 0),
    (64, False, 8),
    (64, True, 0),
    (128, False, 0),
    (128, True, 0),
]


def _poison_caching_allocator(
    shapes: list[tuple[int, ...]], dtype: torch.dtype
) -> None:
    """Populate the CUDA caching allocator with NaN-filled blocks of ``shapes``."""
    blocks = [
        torch.full(shape, float("nan"), dtype=dtype, device="cuda") for shape in shapes
    ]
    del blocks


@unittest.skipUnless(is_sm100_family(), SM100_FAMILY_REQUIRED)
class TestCuteDSLRangeMergeSyncAndGraph(TestCase):
    """SM100/SM110 CuteDSL range_merge=True runs sync-free and captures in CUDAGraph."""

    @parameterize("case", _CUTEDSL_RM_SYNC_GRAPH_CASES)
    def test_range_merge_fwd_bwd_sync_free_and_cuda_graph(
        self, case: tuple[int, bool, int]
    ):
        head_dim, deterministic, sm_margin = case
        torch.manual_seed(1234 + head_dim + int(deterministic) * 10 + sm_margin)
        dtype = torch.bfloat16
        hq, hk = 4, 2
        total_q, total_k = 448, 544
        max_seqlen_q, max_seqlen_k = 256, 384

        # Uncovered Q holes: [0, 32), [161, 192), [321, 448).
        # Uncovered K holes: [0, 48), [177, 224), [481, 544).
        # [96, 96] lies inside covered K [48, 177] to exercise sentinel padding on
        # merged_outer_ranges_by_start; [384, 384] and [500, 500] are zero-tile groups.
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

        # Second schedule with the same relation count R=7 and different active group
        # counts (G_q=2, G_k=4) to exercise device-side mNumBatches inside a CUDAGraph.
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

        # Warm up JIT compilation before enabling strict sync debug mode.
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

        torch.cuda.set_sync_debug_mode("error")
        try:
            out, dq, dk, dv = run_autograd(q, k, v, do)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

        for name, grad in (("dq", dq), ("dk", dk), ("dv", dv)):
            self.assertFalse(bool(torch.isnan(grad).any()), f"{name} contains NaN")

        # Verify uncovered Q and K holes are strictly zeroed.
        for s, e in ((0, 32), (161, 192), (321, total_q)):
            self.assertEqual(float(dq[s:e].abs().max().item()), 0.0)
        for s, e in ((0, 48), (177, 224), (481, total_k)):
            self.assertEqual(float(dk[s:e].abs().max().item()), 0.0)
            self.assertEqual(float(dv[s:e].abs().max().item()), 0.0)
        # Verify covered K rows after degenerate [96, 96] were not zeroed as holes.
        self.assertGreater(float(dk[96:177].abs().max().item()), 0.0)
        self.assertGreater(float(dv[96:177].abs().max().item()), 0.0)

        # Reference check on covered Q rows and full gradients.
        q_ref = q.detach().float().requires_grad_(True)
        k_ref = k.detach().float().requires_grad_(True)
        v_ref = v.detach().float().requires_grad_(True)
        mask_1 = _make_mask(q_ranges_1, k_ranges_1, mask_types_1, total_q, total_k)
        out_ref = _ref_attn(q_ref, k_ref, v_ref, mask_1)
        out_ref.backward(do.float())
        covered_q = torch.zeros(total_q, dtype=torch.bool, device="cuda")
        covered_q[32:161] = True
        covered_q[192:321] = True
        torch.testing.assert_close(
            out[covered_q].float(), out_ref[covered_q], atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(dq.float(), q_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(dk.float(), k_ref.grad, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(dv.float(), v_ref.grad, atol=3e-2, rtol=3e-2)

        # CUDAGraph capture & replay of range_merge=True (fwd + bwd).
        static_q = q.detach().clone()
        static_k = k.detach().clone()
        static_v = v.detach().clone()
        static_do = do.detach().clone()
        static_qr = qr.clone()
        static_kr = kr.clone()
        static_am = am.clone()

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

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_fwd_bwd(
                static_q,
                static_k,
                static_v,
                static_do,
                static_qr,
                static_kr,
                static_am,
            )
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_outputs = run_fwd_bwd(
                static_q,
                static_k,
                static_v,
                static_do,
                static_qr,
                static_kr,
                static_am,
            )

        for q_list, k_list, m_list in (
            (q_ranges_1, k_ranges_1, mask_types_1),
            (q_ranges_2, k_ranges_2, mask_types_2),
        ):
            new_q = torch.randn_like(static_q)
            new_k = torch.randn_like(static_k)
            new_v = torch.randn_like(static_v)
            new_do = torch.randn_like(static_do)
            new_qr = torch.tensor(q_list, dtype=torch.int32, device="cuda")
            new_kr = torch.tensor(k_list, dtype=torch.int32, device="cuda")
            new_am = torch.tensor(m_list, dtype=torch.int32, device="cuda")

            static_q.copy_(new_q)
            static_k.copy_(new_k)
            static_v.copy_(new_v)
            static_do.copy_(new_do)
            static_qr.copy_(new_qr)
            static_kr.copy_(new_kr)
            static_am.copy_(new_am)

            graph.replay()
            torch.cuda.synchronize()
            got_tensors = [t.clone() for t in g_outputs]

            eager_tensors = run_fwd_bwd(
                new_q, new_k, new_v, new_do, new_qr, new_kr, new_am
            )
            torch.cuda.synchronize()

            if deterministic:
                for name, got_t, want_t in zip(
                    ("out", "lse", "dq", "dk", "dv"), got_tensors, eager_tensors
                ):
                    self.assertTrue(
                        torch.equal(got_t, want_t),
                        f"{name} CUDAGraph replay mismatch vs eager",
                    )
            else:
                # Forward direct store and LSE have a single writer per row.
                self.assertTrue(torch.equal(got_tensors[0], eager_tensors[0]))
                self.assertTrue(torch.equal(got_tensors[1], eager_tensors[1]))
                for got_t, want_t in zip(got_tensors[2:], eager_tensors[2:]):
                    torch.testing.assert_close(
                        got_t.float(), want_t.float(), atol=1e-2, rtol=1e-2
                    )

            q_r = new_q.detach().float().requires_grad_(True)
            k_r = new_k.detach().float().requires_grad_(True)
            v_r = new_v.detach().float().requires_grad_(True)
            mask_r = _make_mask(q_list, k_list, m_list, total_q, total_k)
            out_r = _ref_attn(q_r, k_r, v_r, mask_r)
            out_r.backward(new_do.float())
            torch.testing.assert_close(
                got_tensors[0].float(), out_r, atol=3e-2, rtol=3e-2
            )
            torch.testing.assert_close(
                got_tensors[2].float(), q_r.grad, atol=3e-2, rtol=3e-2
            )
            torch.testing.assert_close(
                got_tensors[3].float(), k_r.grad, atol=3e-2, rtol=3e-2
            )
            torch.testing.assert_close(
                got_tensors[4].float(), v_r.grad, atol=3e-2, rtol=3e-2
            )

    @parameterize("deterministic", [False, True])
    def test_all_empty_degenerate_ranges_sync_free_and_cuda_graph(
        self, deterministic: bool
    ):
        """All-empty [s, s) relations with max_seqlen_q=0, max_seqlen_k=0 run
        sync-free and capture in CUDAGraph, zeroing out, dq, dk, and dv."""
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

        torch.cuda.set_sync_debug_mode("error")
        try:
            out, lse, dq, dk, dv = run_step(q, k, v, do)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

        self.assertEqual(float(out.abs().max().item()), 0.0)
        self.assertTrue(bool(torch.isneginf(lse).all()))
        self.assertEqual(float(dq.abs().max().item()), 0.0)
        self.assertEqual(float(dk.abs().max().item()), 0.0)
        self.assertEqual(float(dv.abs().max().item()), 0.0)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            run_step(q, k, v, do)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_out, g_lse, g_dq, g_dk, g_dv = run_step(q, k, v, do)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(float(g_out.abs().max().item()), 0.0)
        self.assertTrue(bool(torch.isneginf(g_lse).all()))
        self.assertEqual(float(g_dq.abs().max().item()), 0.0)
        self.assertEqual(float(g_dk.abs().max().item()), 0.0)
        self.assertEqual(float(g_dv.abs().max().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
