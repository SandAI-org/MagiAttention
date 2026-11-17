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
import random
import unittest
from unittest import TestCase

from magi_attention.meta.solver.various_chunk_solver import (
    VariousChunkRange,
    VariousChunkSolution,
    make_slices,
    pack_big_chunks,
    solve_various_chunk_dispatch,
)
from magi_attention.testing import parameterize

R = VariousChunkRange


def make_pnp_ranges(
    seed: int,
    num_samples: int,
    topk: int,
    video_chunk_size: int,
    tail_chunk_size: int,
):
    """Make packed samples, each with a DSA video range followed by a dense tail range,
    and the workload fn to estimate them:
        video q: attends topk video tokens + all tail tokens
        tail q: attends all video tokens + all tail tokens
    """
    rng = random.Random(seed)
    ranges: list[VariousChunkRange] = []
    video_starts: dict[int, int] = {}
    sample_lens: dict[int, tuple[int, int]] = {}
    offset = 0
    for sample_id in range(num_samples):
        video = rng.randint(16, 128) * video_chunk_size
        tail = rng.randint(256, 4096)
        video_starts[sample_id] = offset
        sample_lens[sample_id] = (video, tail)
        ranges.append(R((offset, offset + video), 0.0, video_chunk_size, sample_id))
        ranges.append(
            R((offset + video, offset + video + tail), 0.0, tail_chunk_size, sample_id)
        )
        offset += video + tail

    def workload_fn(r: VariousChunkRange) -> float:
        video, tail = sample_lens[r.sample_id]
        if r.start == video_starts[r.sample_id]:
            return r.seqlen * (min(topk, video) + tail)
        return r.seqlen * (video + tail)

    return ranges, workload_fn


def sequential_split_workloads(
    ranges: list[VariousChunkRange],
    workloads: list[float],
    num_groups: int,
) -> list[float]:
    """The baseline of the current nhk_ulysses: split the sequence into equal contiguous segments"""
    total_seqlen = ranges[-1].end
    seg_len = math.ceil(total_seqlen / num_groups)
    group_workloads = [0.0] * num_groups
    for r, workload in zip(ranges, workloads):
        for g in range(num_groups):
            lo, hi = max(r.start, g * seg_len), min(r.end, (g + 1) * seg_len)
            if lo < hi:
                group_workloads[g] += workload * (hi - lo) / r.seqlen
    return group_workloads


def imbalance_ratio(workloads: list[float]) -> float:
    return max(workloads) / (sum(workloads) / len(workloads))


class TestVariousChunkSolver(TestCase):
    def check_solution(
        self,
        ranges: list[VariousChunkRange],
        sol: VariousChunkSolution,
        num_groups: int,
        workloads: list[float] | None = None,
    ) -> None:
        if workloads is None:
            workloads = [r.weight for r in ranges]

        for group_list in (
            sol.group_q_ranges,
            sol.group_sample_ids,
            sol.group_slices,
            sol.group_seqlens,
            sol.group_pad_sizes,
            sol.group_workloads,
        ):
            self.assertEqual(len(group_list), num_groups)

        # coverage and atomicity: all slices of all groups are exactly the slices of the input
        all_slices = sorted(
            (s for g in sol.group_slices for s in g), key=lambda s: s.start
        )
        self.assertEqual(all_slices, make_slices(ranges, workloads))
        for s in all_slices:
            r = ranges[s.range_idx]
            self.assertEqual((s.start - r.start) % r.chunk_size, 0)
            self.assertEqual(s.end, min(s.start + r.chunk_size, r.end))

        # merged q_ranges are sorted, of a single sample each, and cover the group slices
        for g in range(num_groups):
            q_ranges = sol.group_q_ranges[g]
            self.assertEqual(len(q_ranges), len(sol.group_sample_ids[g]))
            self.assertEqual(sum(e - s for s, e in q_ranges), sol.group_seqlens[g])
            for (_, prev_end), (start, _) in zip(q_ranges, q_ranges[1:]):
                self.assertLessEqual(prev_end, start)
            for (start, end), sample_id in zip(q_ranges, sol.group_sample_ids[g]):
                for r in ranges:
                    if r.start < end and start < r.end:
                        self.assertEqual(r.sample_id, sample_id)
            self.assertEqual(
                sol.group_seqlens[g], sum(s.seqlen for s in sol.group_slices[g])
            )

        # capacity and padding
        if sol.num_overflow_slices == 0:
            for seqlen in sol.group_seqlens:
                self.assertLessEqual(seqlen, sol.capacity)
        self.assertEqual(sol.padded_seqlen, max(sol.group_seqlens))
        for seqlen, pad in zip(sol.group_seqlens, sol.group_pad_sizes):
            self.assertGreaterEqual(pad, 0)
            self.assertEqual(seqlen + pad, sol.padded_seqlen)

        self.assertAlmostEqual(sum(sol.group_workloads), sum(workloads), delta=1e-6)

    def test_make_slices(self):
        ranges = [R((0, 10), 10.0, 4, 0), R((10, 13), 6.0, 2, 0)]
        slices = make_slices(ranges, [r.weight for r in ranges])
        self.assertEqual(
            [(s.start, s.end) for s in slices],
            [(0, 4), (4, 8), (8, 10), (10, 12), (12, 13)],
        )
        self.assertEqual([s.workload for s in slices], [4.0, 4.0, 2.0, 4.0, 2.0])
        self.assertEqual([s.range_idx for s in slices], [0, 0, 0, 1, 1])

    def test_pack_big_chunks(self):
        big_chunk_size = 8
        ranges = [
            R((0, 10), 1.0, 3, 0),  # slices: 3, 3, 3, 1
            R((10, 12), 1.0, 2, 0),  # slices: 2, same sample as above
            R((12, 14), 1.0, 2, 1),  # new sample
            R((14, 24), 1.0, 10, 1),  # oversized slice
            R((24, 28), 1.0, 4, 1),
        ]
        slices = make_slices(ranges, [r.weight for r in ranges])
        big_chunks = pack_big_chunks(slices, big_chunk_size)

        self.assertEqual(
            [[(s.start, s.end) for s in c.slices] for c in big_chunks],
            [
                [(0, 3), (3, 6)],
                [(6, 9), (9, 10), (10, 12)],
                [(12, 14)],
                [(14, 24)],
                [(24, 28)],
            ],
        )
        self.assertEqual([c.sample_id for c in big_chunks], [0, 0, 1, 1, 1])
        self.assertEqual(
            [c.oversized for c in big_chunks], [False, False, False, True, False]
        )
        self.assertEqual([c.chunk_id for c in big_chunks], list(range(5)))
        for c in big_chunks:
            self.assertEqual(len({s.sample_id for s in c.slices}), 1)
            if not c.oversized:
                self.assertLessEqual(c.seqlen, big_chunk_size)

    def test_wave2_switch_group_in_middle(self):
        # slices: [0,2) [2,4) [4,6) [6,8) with workload 2 each (sample 0),
        #         [8,10) [10,12) with workload 20 each (sample 1)
        # big chunks: B0=[0,4), B1=[4,8), B2=[8,12), capacity = 6
        # wave 2 order: B2 -> g0, B0 -> g1, B1: [4,6) -> g1 (full), [6,8) -> g0
        ranges = [R((0, 8), 8.0, 2, 0), R((8, 12), 40.0, 2, 1)]
        sol = solve_various_chunk_dispatch(ranges, num_groups=2, big_chunk_size=4)

        self.check_solution(ranges, sol, num_groups=2)
        self.assertEqual(sol.capacity, 6)
        self.assertEqual(sol.group_q_ranges, [[(6, 8), (8, 12)], [(0, 6)]])
        self.assertEqual(sol.group_sample_ids, [[0, 1], [0]])
        self.assertEqual(sol.group_seqlens, [6, 6])
        self.assertEqual(sol.group_pad_sizes, [0, 0])
        self.assertEqual(sol.group_workloads, [42.0, 6.0])
        self.assertEqual(sol.num_overflow_slices, 0)

    def test_wave1_and_overflow(self):
        # big chunks: B0=[0,6) oversized, B1=[6,12) oversized, B2=[12,15)
        # align = min(6, 4) = 4, capacity = round_up(ceil(15 / 2), 4) = 8
        # wave 1: B0 -> g0, B1 -> g1; wave 2: B2 fits nowhere -> overflow to g0
        ranges = [
            R((0, 6), 6.0, 6, 0),
            R((6, 12), 60.0, 6, 1),
            R((12, 15), 3.0, 3, 2),
        ]
        sol = solve_various_chunk_dispatch(ranges, num_groups=2, big_chunk_size=4)

        self.check_solution(ranges, sol, num_groups=2)
        self.assertEqual(sol.capacity, 8)
        self.assertEqual(sol.group_q_ranges, [[(0, 6), (12, 15)], [(6, 12)]])
        self.assertEqual(sol.group_seqlens, [9, 6])
        self.assertEqual(sol.padded_seqlen, 9)
        self.assertEqual(sol.group_pad_sizes, [0, 3])
        self.assertEqual(sol.group_workloads, [9.0, 60.0])
        self.assertEqual(sol.num_overflow_slices, 1)

    def test_pad_align(self):
        ranges = [R((0, 8), 8.0, 2, 0), R((8, 12), 40.0, 2, 1)]
        sol = solve_various_chunk_dispatch(
            ranges, num_groups=2, big_chunk_size=4, pad_align=4
        )
        self.assertEqual(sol.padded_seqlen, 8)
        self.assertEqual(sol.group_pad_sizes, [2, 2])

    def test_workload_fn(self):
        ranges = [R((0, 8), 8.0, 2, 0), R((8, 12), 40.0, 2, 1)]
        default_sol = solve_various_chunk_dispatch(
            ranges, num_groups=2, big_chunk_size=4
        )

        # reverse the density: sample 0 becomes the expensive one
        def workload_fn(r: VariousChunkRange) -> float:
            return r.seqlen * (10.0 if r.sample_id == 0 else 1.0)

        sol = solve_various_chunk_dispatch(
            ranges, num_groups=2, big_chunk_size=4, workload_fn=workload_fn
        )
        self.check_solution(
            ranges, sol, num_groups=2, workloads=[workload_fn(r) for r in ranges]
        )
        self.assertNotEqual(sol.group_q_ranges, default_sol.group_q_ranges)
        # B0 -> g0, B1 -> g1, B2: [8,10) -> g0 (full), [10,12) -> g1
        self.assertEqual(sol.group_q_ranges, [[(0, 4), (8, 10)], [(4, 8), (10, 12)]])
        self.assertEqual(sol.group_sample_ids, [[0, 1], [0, 1]])
        self.assertEqual(sol.group_workloads, [42.0, 42.0])

    @parameterize("seed", [0, 1, 2, 3])
    @parameterize("num_groups", [2, 4, 8])
    @parameterize("video_chunk_size", [64, 128])
    def test_pnp_load_balance(self, seed: int, num_groups: int, video_chunk_size: int):
        ranges, workload_fn = make_pnp_ranges(
            seed=seed,
            num_samples=6,
            topk=2048,
            video_chunk_size=video_chunk_size,
            tail_chunk_size=64,
        )
        workloads = [workload_fn(r) for r in ranges]

        sol = solve_various_chunk_dispatch(
            ranges,
            num_groups=num_groups,
            workload_fn=workload_fn,
        )
        self.check_solution(ranges, sol, num_groups=num_groups, workloads=workloads)

        baseline_ratio = imbalance_ratio(
            sequential_split_workloads(ranges, workloads, num_groups)
        )
        solver_ratio = sol.imbalance_ratio
        self.assertLess(solver_ratio, baseline_ratio)
        self.assertLessEqual(solver_ratio, 1.1, f"{baseline_ratio=}, {solver_ratio=}")
        # the padding introduced by the capacity alignment is bounded
        self.assertLessEqual(max(sol.group_pad_sizes), sol.capacity // 4)

    def test_invalid_inputs(self):
        good = [R((0, 8), 8.0, 2, 0)]
        invalid_cases = [
            dict(ranges=[], num_groups=2),
            dict(ranges=good, num_groups=0),
            dict(ranges=good, num_groups=2, big_chunk_size=0),
            dict(ranges=good, num_groups=2, align=0),
            dict(ranges=good, num_groups=2, pad_align=0),
            dict(ranges=[R((4, 4), 1.0, 2, 0)], num_groups=2),
            dict(ranges=[R((0, 8), 1.0, 0, 0)], num_groups=2),
            dict(ranges=[R((0, 8), -1.0, 2, 0)], num_groups=2),
            dict(ranges=[R((0, 8), 1.0, 2, 0), R((6, 10), 1.0, 2, 0)], num_groups=2),
            dict(ranges=good, num_groups=2, workload_fn=lambda r: -1.0),
        ]
        for kwargs in invalid_cases:
            with self.assertRaises(ValueError, msg=f"{kwargs=}"):
                solve_various_chunk_dispatch(**kwargs)


if __name__ == "__main__":
    unittest.main()
