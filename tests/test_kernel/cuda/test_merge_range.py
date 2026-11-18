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
from unittest import TestCase

import torch

from magi_attention import magi_attn_ext


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


if __name__ == "__main__":
    unittest.main()
