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

"""CPU model of the deterministic range-lock chain, checked before the kernels.

Models the protocol of the SM100 range deterministic design: writers merge
into physical row blocks in writer-number order, a writer's work tiles are
claimed as tickets, each logical row tile waits until every block it has
valid rows in carries its predecessor's number, then writes and arrives; a
(writer, block) publishes the writer number once its arrivals sum to 2.

The worker simulator enumerates every interleaving of a few workers (clusters)
with at most ``resident`` of them running at once, and reports whether some
interleaving can no longer progress, or a block merges writers out of order.
Imports neither torch nor a kernel module, so it runs without a GPU.
"""

import enum
import itertools
import random
import unittest
from dataclasses import dataclass


class WriterKind(enum.Enum):
    # The writer's logical tiles may sit in several work tiles (fwd O/LSE,
    # bwd dK/dV): a block covered by two of them takes weight 1 from each, a
    # block covered by one takes weight 2.
    MULTI_TILE = enum.auto()
    # All logical tiles sit in one work tile (bwd dQ, RangeMerge): that work
    # tile arrives once with weight 2 after its last tile touching the block.
    SINGLE_WORK_TILE = enum.auto()


@dataclass(frozen=True)
class Writer:
    number: int  # writer number, >= 1; 0 means "no predecessor"
    kind: WriterKind
    # work_tiles[i] is the list of logical tiles of work tile i, each the
    # valid rows [a, b) of that tile.
    work_tiles: tuple[tuple[tuple[int, int], ...], ...]


@dataclass(frozen=True)
class Step:
    """One logical tile of a work tile: wait, write, arrive."""

    writer: int
    waits: tuple[tuple[int, int], ...]  # (block, predecessor number)
    arrivals: tuple[tuple[int, int], ...]  # (block, weight)
    blocks: tuple[int, ...]


def logical_tiles(start: int, end: int, tile: int) -> tuple[tuple[int, int], ...]:
    """Valid rows of the tiles a range is cut into from its start."""
    return tuple((a, min(a + tile, end)) for a in range(start, end, tile))


def blocks_of(rows: tuple[int, int], block: int) -> tuple[int, ...]:
    a, b = rows
    return tuple(range(a // block, (b - 1) // block + 1)) if b > a else ()


def predecessors(writers: list[Writer], block: int) -> dict[tuple[int, int], int]:
    """(writer, block) -> number of the closest lower writer with an event on
    that block, 0 if none. Writers without tiles have no event."""
    covered: dict[int, list[int]] = {}
    for w in sorted(writers, key=lambda w: w.number):
        for tiles in w.work_tiles:
            for rows in tiles:
                for b in blocks_of(rows, block):
                    if not covered.setdefault(b, []) or covered[b][-1] != w.number:
                        covered[b].append(w.number)
    pred = {}
    for b, numbers in covered.items():
        for i, n in enumerate(numbers):
            pred[(n, b)] = numbers[i - 1] if i else 0
    return pred


def arrival_weights(
    writer: Writer, block: int
) -> list[list[tuple[tuple[int, int], ...]]]:
    """Per work tile, per logical tile, the (block, weight) arrivals."""
    out: list[list[tuple[tuple[int, int], ...]]] = []
    if writer.kind is WriterKind.MULTI_TILE:
        cover: dict[int, int] = {}
        for tiles in writer.work_tiles:
            for rows in tiles:
                for b in blocks_of(rows, block):
                    cover[b] = cover.get(b, 0) + 1
        for tiles in writer.work_tiles:
            out.append(
                [
                    tuple(
                        (b, 2 if cover[b] == 1 else 1) for b in blocks_of(rows, block)
                    )
                    for rows in tiles
                ]
            )
        return out
    assert len(writer.work_tiles) == 1
    (tiles,) = writer.work_tiles
    last: dict[int, int] = {}
    for i, rows in enumerate(tiles):
        for b in blocks_of(rows, block):
            last[b] = i
    out.append(
        [
            tuple((b, 2) for b in blocks_of(rows, block) if last[b] == i)
            for i, rows in enumerate(tiles)
        ]
    )
    return out


def compile_tickets(
    writers: list[Writer],
    ticket_order: list[tuple[int, int]],
    block: int,
    drop_empty_arrivals_of: frozenset[int] = frozenset(),
) -> list[tuple[Step, ...]]:
    """Steps of every ticket; ``ticket_order`` lists (writer number, work tile
    index). ``drop_empty_arrivals_of`` removes the arrivals of those writers
    (a writer that skips its zero-contribution events)."""
    by_number = {w.number: w for w in writers}
    pred = predecessors(writers, block)
    weights = {w.number: arrival_weights(w, block) for w in writers}
    tickets = []
    for number, unit in ticket_order:
        w = by_number[number]
        steps = []
        for rows, arrivals in zip(w.work_tiles[unit], weights[number][unit]):
            blocks = blocks_of(rows, block)
            steps.append(
                Step(
                    writer=number,
                    waits=tuple((b, pred[(number, b)]) for b in blocks),
                    arrivals=() if number in drop_empty_arrivals_of else arrivals,
                    blocks=blocks,
                )
            )
        tickets.append(tuple(steps))
    return tickets


class ClaimMode(enum.Enum):
    # Every ticket, the first included, comes from one counter (design R0).
    ATOMIC = enum.auto()
    # Worker i's first ticket is i; later claims start after the grid.
    RESERVED_FIRST = enum.auto()


@dataclass(frozen=True)
class Outcome:
    stuck: bool
    out_of_order: bool


def explore(
    tickets: list[tuple[Step, ...]],
    num_workers: int,
    resident: int,
    claim: ClaimMode,
    max_states: int = 200_000,
) -> Outcome:
    """Every interleaving of ``num_workers`` workers, at most ``resident`` of
    them running at once. A worker claims a ticket, runs its steps in order
    (a step runs once its waits hold), claims the next, and exits when no
    ticket is left."""
    num_blocks = 1 + max((b for t in tickets for s in t for b in s.blocks), default=0)
    UNSTARTED, RUNNING, DONE = 0, 1, 2

    def first_ticket(worker: int) -> int | None:
        return worker if worker < len(tickets) else None

    # worker state: (status, ticket or -1, step index)
    start_counter = 0 if claim is ClaimMode.ATOMIC else num_workers
    init = (
        start_counter,
        tuple((UNSTARTED, -1, 0) for _ in range(num_workers)),
        (0,) * num_blocks,  # published number
        (0,) * num_blocks,  # arrival count
        (0,) * num_blocks,  # highest writer that wrote the block
    )
    State = tuple[
        int,
        tuple[tuple[int, int, int], ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    seen: set[State] = set()
    stack: list[State] = [init]
    stuck = out_of_order = False
    while stack and len(seen) < max_states:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        counter, workers, published, count, written = state
        running = sum(1 for st, _, _ in workers if st == RUNNING)
        moves: list[tuple[int, int, tuple[int, int, int], Step | None]] = []
        for i, (st, ticket, step) in enumerate(workers):
            if st == UNSTARTED and running < resident:
                if claim is ClaimMode.RESERVED_FIRST:
                    t = first_ticket(i)
                    new_w = (RUNNING, t, 0) if t is not None else (DONE, -1, 0)
                    moves.append((counter, i, new_w, None))
                else:
                    moves.append((counter, i, (RUNNING, -1, 0), None))
            elif st == RUNNING and ticket == -1:
                if counter < len(tickets):
                    moves.append((counter + 1, i, (RUNNING, counter, 0), None))
                else:
                    moves.append((counter, i, (DONE, -1, 0), None))
            elif st == RUNNING:
                if step == len(tickets[ticket]):
                    moves.append((counter, i, (RUNNING, -1, 0), None))
                    continue
                s = tickets[ticket][step]
                if all(published[b] == p for b, p in s.waits):
                    moves.append((counter, i, (RUNNING, ticket, step + 1), s))
        if not moves:
            if any(st != DONE for st, _, _ in workers):
                stuck = True
            continue
        for new_counter, i, new_w, moved in moves:
            pub, cnt, wr = list(published), list(count), list(written)
            if moved is not None:
                for b in moved.blocks:
                    if wr[b] > moved.writer:
                        out_of_order = True
                    wr[b] = max(wr[b], moved.writer)
                for b, weight in moved.arrivals:
                    cnt[b] += weight
                    if cnt[b] == 2:
                        cnt[b] = 0
                        pub[b] = moved.writer
            ws = list(workers)
            ws[i] = new_w
            stack.append((new_counter, tuple(ws), tuple(pub), tuple(cnt), tuple(wr)))
    assert not stack, "state budget exhausted; shrink the case"
    return Outcome(stuck=stuck, out_of_order=out_of_order)


def range_writers(
    ranges: list[tuple[int, int]], tile: int, heads: int = 1
) -> list[Writer]:
    """MULTI_TILE writers (relation r, head g) numbered r * heads + g + 1,
    one work tile per logical tile (fwd stage tiles, bwd dK/dV K tiles)."""
    writers = []
    for r, (start, end) in enumerate(ranges):
        tiles = logical_tiles(start, end, tile)
        for g in range(heads):
            writers.append(
                Writer(
                    number=r * heads + g + 1,
                    kind=WriterKind.MULTI_TILE,
                    work_tiles=tuple((t,) for t in tiles),
                )
            )
    return writers


def head_major_order(writers: list[Writer]) -> list[tuple[int, int]]:
    return [
        (w.number, i)
        for w in sorted(writers, key=lambda w: w.number)
        for i in range(len(w.work_tiles))
    ]


class ConflictScanner:
    """A1 conflict state of one (execution slot, consumer role): before a
    tile of writer ``w`` is processed, every writer below ``w`` not yet
    recorded writes its number into the blocks it has events on; the tile
    then snapshots its predecessors from the column."""

    def __init__(self, writers: list[Writer], block: int):
        self._writers = sorted(writers, key=lambda w: w.number)
        self._block = block
        self._cursor = 0
        self.column: dict[int, int] = {}

    def snapshot(self, number: int, blocks: tuple[int, ...]) -> tuple[int, ...]:
        while (
            self._cursor < len(self._writers)
            and self._writers[self._cursor].number < number
        ):
            w = self._writers[self._cursor]
            for tiles in w.work_tiles:
                for rows in tiles:
                    for b in blocks_of(rows, self._block):
                        self.column[b] = w.number
            self._cursor += 1
        return tuple(self.column.get(b, 0) for b in blocks)


class TestRangeDeterministicProtocol(unittest.TestCase):
    _M = 8  # block height = logical tile height, small to keep states few

    def _run(self, writers, order, workers, resident, claim, **kw) -> Outcome:
        tickets = compile_tickets(writers, order, self._M, **kw)
        return explore(tickets, workers, resident, claim)

    def assertCompletes(self, outcome: Outcome) -> None:
        self.assertFalse(outcome.stuck, "some interleaving cannot progress")
        self.assertFalse(outcome.out_of_order, "a block merged out of order")

    def test_head_swizzle_order_deadlocks_and_head_major_completes(self):
        """bwd dK/dV without CatGQA: one relation, K rows [1, 2M + 1), two q
        heads of one kv head. Swizzle claims (h0,n0), (h1,n0), (h0,n1): the
        second ticket waits for writer h0 on block 1, which still lacks
        (h0,n1)."""
        writers = range_writers([(1, 2 * self._M + 1)], self._M, heads=2)
        h0, h1 = 1, 2
        swizzle = [(h0, 0), (h1, 0), (h0, 1), (h1, 1)]
        self.assertTrue(self._run(writers, swizzle, 1, 1, ClaimMode.ATOMIC).stuck)
        self.assertCompletes(
            self._run(writers, head_major_order(writers), 1, 1, ClaimMode.ATOMIC)
        )

    def test_reserved_first_ticket_deadlocks_and_atomic_claim_completes(self):
        """Two writers on one block, two workers, one resident at a time.
        With the first ticket reserved by worker index, worker 1 may start
        first and wait for ticket 0 that only the non-running worker 0 holds."""
        writers = range_writers([(0, self._M), (0, self._M)], self._M)
        order = head_major_order(writers)
        self.assertTrue(self._run(writers, order, 2, 1, ClaimMode.RESERVED_FIRST).stuck)
        self.assertCompletes(self._run(writers, order, 2, 1, ClaimMode.ATOMIC))

    def test_two_cta_halves_miscount_and_cluster_aggregation_counts_two(self):
        """2-CTA dQ: each CTA writes half of a logical Q tile. Q range
        [1, 2M + 1) puts three half tiles in block 1, so per-half counting
        sums to 3; aggregating the halves per logical tile sums to 2."""
        start, end = 1, 2 * self._M + 1
        halves = tuple(
            h
            for a, b in logical_tiles(start, end, self._M)
            for h in logical_tiles(a, b, self._M // 2)
        )
        per_half = Writer(1, WriterKind.MULTI_TILE, tuple((h,) for h in halves))
        sums: dict[int, int] = {}
        for tile_arrivals in arrival_weights(per_half, self._M):
            for arrivals in tile_arrivals:
                for b, weight in arrivals:
                    sums[b] = sums.get(b, 0) + weight
        self.assertEqual(sums[1], 3)
        (aggregated,) = range_writers([(start, end)], self._M)
        sums = {}
        for tile_arrivals in arrival_weights(aggregated, self._M):
            for arrivals in tile_arrivals:
                for b, weight in arrivals:
                    sums[b] = sums.get(b, 0) + weight
        self.assertEqual(set(sums.values()), {2})

    def test_guard_block_empty_range_and_zero_contribution(self):
        """Relation 0 [1, M + 2) ends one row into block 1, so its last tile's
        guard block 2 carries no event; relation 1 is empty and records
        nothing; relation 2 [2M, 3M) only waits for writers with events on
        block 2 (none). Dropping the arrivals of a writer whose tiles have
        valid rows (a zero-contribution writer) blocks its successor."""
        ranges = [(1, self._M + 2), (5, 5), (2 * self._M, 3 * self._M), (0, self._M)]
        writers = range_writers(ranges, self._M)
        self.assertEqual(blocks_of((self._M + 1, self._M + 2), self._M), (1,))
        order = head_major_order(writers)
        for workers, resident in ((1, 1), (2, 2), (3, 2)):
            self.assertCompletes(
                self._run(writers, order, workers, resident, ClaimMode.ATOMIC)
            )
        self.assertTrue(
            self._run(
                writers,
                order,
                2,
                2,
                ClaimMode.ATOMIC,
                drop_empty_arrivals_of=frozenset({1}),
            ).stuck
        )

    def test_range_merge_overlapping_pairs_publish_once_per_block(self):
        """A merged group's work tile writes its pairs back to back. Pairs
        overlapping in Q put up to four tiles of one writer in a block, so the
        per-tile weights do not sum to 2; arriving once with weight 2 after
        the last participant does, and the chain completes."""
        pairs = [(1, 2 * self._M + 1), (3, self._M + 3), (self._M, 2 * self._M)]
        tiles = tuple(t for a, b in pairs for t in logical_tiles(a, b, self._M))
        multi = Writer(1, WriterKind.MULTI_TILE, tuple((t,) for t in tiles))
        sums: dict[int, int] = {}
        for tile_arrivals in arrival_weights(multi, self._M):
            for arrivals in tile_arrivals:
                for b, weight in arrivals:
                    sums[b] = sums.get(b, 0) + weight
        self.assertNotEqual(set(sums.values()), {2})
        group = Writer(1, WriterKind.SINGLE_WORK_TILE, (tiles,))
        follower = Writer(
            2, WriterKind.SINGLE_WORK_TILE, (logical_tiles(0, 3 * self._M, self._M),)
        )
        writers = [group, follower]
        for workers in (1, 2):
            self.assertCompletes(
                self._run(
                    writers,
                    head_major_order(writers),
                    workers,
                    workers,
                    ClaimMode.ATOMIC,
                )
            )

    def test_conflict_scan_matches_global_predecessors(self):
        """The per-slot incremental scan (A1) yields the same predecessor of
        every tile as the global definition, for each worker's monotonic
        ticket subsequence."""
        rng = random.Random(0)
        for _ in range(200):
            ranges = []
            for _ in range(rng.randint(1, 6)):
                start = rng.randint(0, 5 * self._M)
                ranges.append((start, start + rng.randint(0, 3 * self._M)))
            writers = range_writers(ranges, self._M, heads=rng.randint(1, 3))
            pred = predecessors(writers, self._M)
            order = head_major_order(writers)
            by_number = {w.number: w for w in writers}
            num_slots = rng.randint(1, 3)
            scanners = [ConflictScanner(writers, self._M) for _ in range(num_slots)]
            for ticket, (number, unit) in enumerate(order):
                scanner = scanners[rng.randrange(num_slots)] if ticket else scanners[0]
                for rows in by_number[number].work_tiles[unit]:
                    blocks = blocks_of(rows, self._M)
                    self.assertEqual(
                        scanner.snapshot(number, blocks),
                        tuple(pred[(number, b)] for b in blocks),
                    )

    def test_random_overlapping_ranges_complete_in_order(self):
        """Random unaligned overlapping relations, 1-3 workers: every
        interleaving completes and every block merges in writer order."""
        rng = random.Random(1)
        for _ in range(40):
            ranges = []
            for _ in range(rng.randint(2, 4)):
                start = rng.randint(0, 2 * self._M)
                ranges.append((start, start + rng.randint(0, 2 * self._M)))
            writers = range_writers(ranges, self._M, heads=rng.randint(1, 2))
            order = head_major_order(writers)
            workers = rng.randint(1, 3)
            resident = rng.randint(1, workers)
            self.assertCompletes(
                self._run(writers, order, workers, resident, ClaimMode.ATOMIC)
            )

    def test_progress_iff_predecessor_tiles_are_claimed_first(self):
        """With one worker, a claim order progresses exactly when every ticket
        comes after all tiles that produce the predecessor numbers it waits
        for (design R1); head-major order is one such order."""
        writers = range_writers([(1, 2 * self._M + 1), (0, 2 * self._M)], self._M)
        by_number = {w.number: w for w in writers}
        pred = predecessors(writers, self._M)
        for order in itertools.permutations(head_major_order(writers)):
            position = {ticket: i for i, ticket in enumerate(order)}
            r1_holds = True
            for i, (number, unit) in enumerate(order):
                for rows in by_number[number].work_tiles[unit]:
                    for b in blocks_of(rows, self._M):
                        p = pred[(number, b)]
                        if p == 0:
                            continue
                        for j, tiles in enumerate(by_number[p].work_tiles):
                            touches = any(b in blocks_of(t, self._M) for t in tiles)
                            if touches and position[(p, j)] > i:
                                r1_holds = False
            outcome = self._run(writers, list(order), 1, 1, ClaimMode.ATOMIC)
            self.assertFalse(outcome.out_of_order, f"{order=}")
            self.assertEqual(outcome.stuck, not r1_holds, f"{order=}")


if __name__ == "__main__":
    unittest.main()
