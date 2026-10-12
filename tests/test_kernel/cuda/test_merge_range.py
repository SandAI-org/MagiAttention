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
import warnings
from unittest import TestCase

import torch

from magi_attention import magi_attn_ext
from magi_attention.kernel.cutedsl.range_merge import merge_qk_ranges


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
) -> list[tuple[list[int], list[tuple[list[int], int]]]]:
    """The groups of a merge table in table order; every group has a pair."""
    cu = cu_batches.tolist()
    inner, mask = sorted_inner.tolist(), sorted_mask.tolist()
    num_groups = merged_outer.shape[0]
    assert len(cu) == num_groups + 1, (len(cu), num_groups)
    assert cu[0] == 0 and cu[-1] == len(inner), cu
    assert all(b > a for a, b in zip(cu, cu[1:])), cu
    return [
        (
            merged_outer[g].tolist(),
            list(zip(inner[cu[g] : cu[g + 1]], mask[cu[g] : cu[g + 1]])),
        )
        for g in range(num_groups)
    ]


class TestRangeMergeLaunchOrder(TestCase):
    """merge_qk_ranges launches the merged groups heaviest work tile first."""

    def _merge(self, outer, inner, mask_types):
        device = torch.cuda.current_device()
        outer_t, inner_t, mask_t = (
            torch.tensor(x, dtype=torch.int32, device=device)
            for x in (outer, inner, mask_types)
        )
        plan = merge_qk_ranges(outer_t, inner_t, mask_t)
        by_start = plan.merged_outer_ranges_by_start.tolist()
        self.assertEqual([r[0] for r in by_start], sorted(r[0] for r in by_start))
        self.assertEqual(sorted(by_start), sorted(plan.merged_outer_ranges.tolist()))
        return _plan_groups(
            plan.merged_outer_ranges,
            plan.sorted_inner_ranges,
            plan.sorted_mask_types,
            plan.cu_batches,
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

    def test_merge_syncs_once_to_read_the_group_count(self):
        """Reading G is the only host sync; the launch order adds none."""
        device = torch.cuda.current_device()
        outer = torch.tensor(
            [[0, 64], [64, 65], [0, 64]], dtype=torch.int32, device=device
        )
        inner = torch.tensor(
            [[0, 64], [0, 4096], [64, 128]], dtype=torch.int32, device=device
        )
        mask_types = torch.zeros(3, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                merge_qk_ranges(outer, inner, mask_types)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs = [w for w in caught if "called a synchronizing" in str(w.message)]
        self.assertEqual(len(syncs), 1, [str(w.message) for w in syncs])


if __name__ == "__main__":
    unittest.main()
