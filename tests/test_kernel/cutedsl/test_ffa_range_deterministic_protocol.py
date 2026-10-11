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

A second simulator models the two CTAs of a 2-CTA cluster inside one
RangeMerge dQ writer: their asynchronous row-half additions, the per-tile
drains, and the pair handoff on two mbarriers per CTA with exact parity
semantics, and reports whether some interleaving deadlocks or adds a shared
row out of pair order.

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


def merged_dq_writers(
    groups: list[list[tuple[int, int]]], k_tiles: list[int], stride: int, tile: int
) -> list[Writer]:
    """bwd dQ writers of RangeMerge: merged K group g holds the Q ranges of
    its pairs; cluster K tile n < k_tiles[g] is writer g * stride + n + 1, one
    SINGLE_WORK_TILE writer whose work tile walks every Q tile of every pair
    in pair order (mask-skipped tiles included)."""
    writers = []
    for g, pairs in enumerate(groups):
        tiles = tuple(t for a, b in pairs for t in logical_tiles(a, b, tile))
        for n in range(k_tiles[g]):
            writers.append(
                Writer(g * stride + n + 1, WriterKind.SINGLE_WORK_TILE, (tiles,))
            )
    return writers


def merged_tile_arrivals(
    pairs: list[tuple[int, int]], tile: int
) -> list[tuple[tuple[int, int], ...]]:
    """The kernel's local arrival rule, per logical tile of the pair walk:
    tile m of pair j arrives with weight 2 on its first block, and on its
    last block when m is the pair's last tile, unless a later pair's Q range
    has a row in that block."""

    def covered_later(j: int, b: int) -> bool:
        return any(
            a < e and a // tile <= b <= (e - 1) // tile for a, e in pairs[j + 1 :]
        )

    out = []
    for j, (a, e) in enumerate(pairs):
        tiles = logical_tiles(a, e, tile)
        for m, rows in enumerate(tiles):
            left, right = blocks_of(rows, tile)[0], blocks_of(rows, tile)[-1]
            arrivals = []
            if not covered_later(j, left):
                arrivals.append((left, 2))
            if right != left and m == len(tiles) - 1 and not covered_later(j, right):
                arrivals.append((right, 2))
            out.append(tuple(arrivals))
    return out


def merged_predecessor(
    groups: list[list[tuple[int, int]]],
    k_tiles: list[int],
    stride: int,
    tile: int,
    g: int,
    n: int,
    block: int,
) -> int:
    """The kernel's predecessor of writer (g, n) on ``block``: writer - 1 for
    n > 0; for n == 0 the conflict column after scanning groups [0, g), each
    group recording its last writer on the blocks of every pair."""
    if n > 0:
        return g * stride + n
    column: dict[int, int] = {}
    for h in range(g):
        if not k_tiles[h]:
            continue
        for a, e in groups[h]:
            for b in blocks_of((a, e), tile):
                column[b] = h * stride + k_tiles[h]
    return column.get(block, 0)


def random_groups(rng: random.Random, tile: int, max_groups: int, max_pairs: int):
    groups = []
    for _ in range(rng.randint(1, max_groups)):
        pairs = []
        for _ in range(rng.randint(1, max_pairs)):
            start = rng.randint(0, 3 * tile)
            pairs.append((start, start + rng.randint(0, 2 * tile + 3)))
        groups.append(pairs)
    return groups


def compile_merged_tickets(
    groups: list[list[tuple[int, int]]],
    k_tiles: list[int],
    stride: int,
    tile: int,
) -> list[tuple[Step, ...]]:
    """Tickets of the merged dQ writers in (group, K tile) order, built from
    the kernel's local rules alone: merged_predecessor for the waits and
    merged_tile_arrivals for the arrivals."""
    tickets = []
    for g, pairs in enumerate(groups):
        tiles = [t for a, b in pairs for t in logical_tiles(a, b, tile)]
        arrivals = merged_tile_arrivals(pairs, tile)
        for n in range(k_tiles[g]):
            number = g * stride + n + 1
            steps = []
            for rows, tile_arrivals in zip(tiles, arrivals):
                blocks = blocks_of(rows, tile)
                waits = tuple(
                    (b, merged_predecessor(groups, k_tiles, stride, tile, g, n, b))
                    for b in blocks
                )
                steps.append(Step(number, waits, tile_arrivals, blocks))
            tickets.append(tuple(steps))
    return tickets


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

    def test_packed_rows_scan_span_and_chain_complete(self):
        """PackGQA (G dividing the tile height): token range [s, e) is the
        packed row range [s * G, e * G) of one kv head, cut into tiles from
        its start. The blocks of its tiles are exactly the span the conflict
        scan records from the token range, s * G // M to (e * G - 1) // M,
        and the chain over packed rows completes in writer order."""
        rng = random.Random(2)
        for group in (2, 4):
            for _ in range(20):
                ranges = []
                for _ in range(rng.randint(2, 3)):
                    start = rng.randint(0, 2 * self._M // group)
                    ranges.append((start, start + rng.randint(0, 2 * self._M // group)))
                packed = [(s * group, e * group) for s, e in ranges]
                for (s, e), (ps, pe) in zip(ranges, packed):
                    tile_blocks = {
                        b
                        for rows in logical_tiles(ps, pe, self._M)
                        for b in blocks_of(rows, self._M)
                    }
                    scan_span = (
                        set(range(s * group // self._M, (e * group - 1) // self._M + 1))
                        if e > s
                        else set()
                    )
                    self.assertEqual(tile_blocks, scan_span, f"{group=} {(s, e)=}")
                writers = range_writers(packed, self._M)
                workers = rng.randint(1, 2)
                self.assertCompletes(
                    self._run(
                        writers,
                        head_major_order(writers),
                        workers,
                        rng.randint(1, workers),
                        ClaimMode.ATOMIC,
                    )
                )

    def test_range_merge_local_arrivals_match_the_global_rule(self):
        """bwd RangeMerge: the kernel decides an arrival from the pair walk
        alone (the tile's own blocks, its pair's last tile, the Q ranges of
        the later pairs). On random groups of overlapping, unaligned and
        empty pairs this is the SINGLE_WORK_TILE rule: one weight-2 arrival
        per block, from the last tile with rows there."""
        rng = random.Random(2)
        for _ in range(300):
            (pairs,) = random_groups(rng, self._M, max_groups=1, max_pairs=5)
            (writer,) = merged_dq_writers([pairs], [1], 1, self._M)
            (global_rule,) = arrival_weights(writer, self._M)
            self.assertEqual(merged_tile_arrivals(pairs, self._M), global_rule)

    def test_range_merge_predecessors_match_the_global_definition(self):
        """The merged dQ writer (g, n) waits for writer - 1 when n > 0, and
        for n == 0 on the conflict column of the groups below g, each
        recording its last writer on the blocks of all its pairs. This equals
        the global predecessor, which is always a lower writer: a unit only
        waits for units claimed before it."""
        rng = random.Random(3)
        for _ in range(200):
            groups = random_groups(rng, self._M, max_groups=4, max_pairs=4)
            k_tiles = [rng.randint(0, 3) for _ in groups]
            stride = max(max(k_tiles), 1)
            writers = merged_dq_writers(groups, k_tiles, stride, self._M)
            pred = predecessors(writers, self._M)
            for (number, block), p in pred.items():
                g, n = divmod(number - 1, stride)
                self.assertLess(p, number)
                self.assertEqual(
                    merged_predecessor(groups, k_tiles, stride, self._M, g, n, block),
                    p,
                )

    def test_range_merge_random_overlapping_pairs_complete_in_order(self):
        """Tickets built from the kernel's local wait and arrival rules, on
        random merged groups with overlapping pairs and several K tiles,
        complete in every interleaving of 1-3 workers, merging every block
        in writer order."""
        rng = random.Random(4)
        for _ in range(40):
            groups = random_groups(rng, self._M, max_groups=3, max_pairs=3)
            k_tiles = [rng.randint(0, 2) for _ in groups]
            stride = max(max(k_tiles), 1)
            tickets = compile_merged_tickets(groups, k_tiles, stride, self._M)
            workers = rng.randint(1, 3)
            resident = rng.randint(1, workers)
            self.assertCompletes(explore(tickets, workers, resident, ClaimMode.ATOMIC))

    def test_range_merge_without_the_later_pair_check_merges_out_of_order(self):
        """Control: the unmerged arrival rule (first block always, last block
        on the pair's last tile) ignores the later pairs, so a block shared by
        two pairs publishes after the first and the next writer merges before
        the second; the later-pair check is what keeps the order."""
        pairs = [(1, 2 * self._M + 1), (3, self._M + 3)]
        tickets = compile_merged_tickets(
            [pairs, [(0, 3 * self._M)]], [1, 1], 1, self._M
        )
        self.assertCompletes(explore(tickets, 2, 2, ClaimMode.ATOMIC))
        unmerged_rule = []
        for a, b in pairs:
            tiles = logical_tiles(a, b, self._M)
            for m, rows in enumerate(tiles):
                blocks = blocks_of(rows, self._M)
                last = blocks[-1] != blocks[0] and m == len(tiles) - 1
                unmerged_rule.append(
                    ((blocks[0], 2),) + (((blocks[-1], 2),) if last else ())
                )
        group_steps = tuple(
            Step(s.writer, s.waits, arrivals, s.blocks)
            for s, arrivals in zip(tickets[0], unmerged_rule)
        )
        outcome = explore([group_steps, tickets[1]], 2, 2, ClaimMode.ATOMIC)
        self.assertTrue(outcome.out_of_order or outcome.stuck)

    def test_range_merge_two_cta_halves_of_one_writer_share_rows(self):
        """Under 2-CTA each CTA reduces one half of a logical Q tile. Two
        pairs of one merged writer whose Q ranges start in different rows of
        a tile put one physical row in the half of CTA 0 for one pair and of
        CTA 1 for the other, so the chain, which orders writers and not the
        two CTAs inside one, leaves that row's two additions unordered."""
        half = self._M // 2
        pairs = [(0, 2 * self._M), (half, 2 * self._M + half)]
        owners: dict[int, set[int]] = {}
        for a, b in pairs:
            for t in logical_tiles(a, b, self._M):
                for rank, (ha, hb) in enumerate(logical_tiles(*t, half)):
                    for row in range(ha, hb):
                        owners.setdefault(row, set()).add(rank)
        self.assertTrue(any(len(ranks) == 2 for ranks in owners.values()))


# ------------------------------------------------- 2-CTA pair handoff model


class HandoffMode(enum.Enum):
    # No cross-CTA order inside one writer.
    NONE = enum.auto()
    # Handshake k on bar[k & 1], waiting for parity (k >> 1) & 1.
    TWO_BARRIERS = enum.auto()
    # Handshake k on bar[0], waiting for parity k & 1.
    ONE_BARRIER = enum.auto()


@dataclass(frozen=True)
class MergedPair:
    """One pair of a merged K group seen from one cluster K tile: Q rows
    ``[q_start, q_end)`` cut into ``tile``-row Q tiles from ``q_start``; the
    mask leaves work on Q tiles ``[lo, hi)`` (none when ``hi <= lo``)."""

    q_start: int
    q_end: int
    lo: int
    hi: int


@dataclass(frozen=True)
class PairWalk:
    """The pair walk of one work tile (one dQ writer): every q head of the
    kv head walks every pair in order."""

    pairs: tuple[MergedPair, ...]
    q_heads: int = 1


def two_cta_half_rows(
    pair: MergedPair, m_block: int, rank: int, tile: int
) -> tuple[int, ...]:
    """Valid rows of Q tile ``m_block`` of ``pair`` that the CTA of ``rank``
    adds: the rank's half of the tile, clipped to the pair's end."""
    half = tile // 2
    start = pair.q_start + m_block * tile + rank * half
    return tuple(range(start, min(start + half, pair.q_end)))


def compile_pair_handoff(
    walks: list[PairWalk],
    rank: int,
    tile: int,
    mode: HandoffMode,
    skip_handoff_without_work: bool = False,
) -> tuple[tuple, ...]:
    """The reduce-role program of the CTA of ``rank`` over ``walks``.

    Ops: ``("arrive", bar)`` on the peer's barrier ``bar``; ``("wait", bar,
    parity)`` on the own one; ``("issue", adds)`` starts the asynchronous
    additions ``adds``, each ``((walk, q head, row), pair index)``;
    ``("drain",)`` waits for every addition of this CTA to land. Every pair
    from the second on arrives at its start and waits before issuing its
    first addition, or at once when the mask leaves it no work; the
    handshake counter ``k`` runs across q heads and walks.
    ``skip_handoff_without_work`` drops the handshake of pairs without work
    (the miscounted control)."""
    ops: list[tuple] = []
    k = 0
    for w, walk in enumerate(walks):
        for qh in range(walk.q_heads):
            for pj, pair in enumerate(walk.pairs):
                has_work = pair.hi > pair.lo
                wait = None
                if (
                    mode is not HandoffMode.NONE
                    and pj >= 1
                    and not (skip_handoff_without_work and not has_work)
                ):
                    if mode is HandoffMode.TWO_BARRIERS:
                        bar, parity = k & 1, (k >> 1) & 1
                    else:
                        bar, parity = 0, k & 1
                    ops.append(("arrive", bar))
                    wait = ("wait", bar, parity)
                    k += 1
                    if not has_work:
                        ops.append(wait)
                for it in range(max(pair.hi - pair.lo, 0)):
                    if it == 0 and wait is not None:
                        ops.append(wait)
                    rows = two_cta_half_rows(pair, pair.lo + it, rank, tile)
                    ops.append(("issue", tuple(((w, qh, r), pj) for r in rows)))
                    # Every Q tile waits for its additions to land.
                    ops.append(("drain",))
        # The work tile's closing drain.
        ops.append(("drain",))
    return tuple(ops)


def explore_pair_handoff(
    programs: tuple[tuple[tuple, ...], tuple[tuple, ...]],
    max_states: int = 2_000_000,
) -> Outcome:
    """Every interleaving of the two CTAs of a cluster running ``programs``.

    Each CTA owns two mbarriers of arrival count 1, as the kernel's: an
    arrival completes the current phase and flips its parity, and
    ``wait(parity)`` passes once the phase of that parity has completed,
    i.e. while the current parity differs. Issued additions complete one
    at a time in any order and at any later point. ``out_of_order``: some
    row's additions completed out of pair order; ``stuck``: some
    interleaving reaches a state where neither CTA nor any addition can
    move before both programs end."""
    # state: (pcs, barriers[cta][bar] = (parity, pending), in flight
    # additions (cta, row, pair), last completed pair per row)
    State = tuple[
        tuple[int, ...],
        tuple[tuple[tuple[int, int], ...], ...],
        frozenset[tuple[int, tuple[int, int, int], int]],
        frozenset[tuple[tuple[int, int, int], int]],
    ]
    init: State = (
        (0, 0),
        (((0, 1), (0, 1)), ((0, 1), (0, 1))),
        frozenset(),
        frozenset(),
    )
    seen: set[State] = set()
    stack: list[State] = [init]
    stuck = out_of_order = False
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        assert len(seen) <= max_states, "state budget exhausted; shrink the case"
        pcs, bars, flight, done = state
        successors: list[State] = []
        for c in (0, 1):
            if pcs[c] == len(programs[c]):
                continue
            op = programs[c][pcs[c]]
            new_pcs = tuple(p + 1 if i == c else p for i, p in enumerate(pcs))
            if op[0] == "arrive":
                peer, bar = 1 - c, op[1]
                parity, pending = bars[peer][bar]
                pending -= 1
                if pending == 0:
                    parity, pending = parity ^ 1, 1
                peer_bars = tuple(
                    (parity, pending) if i == bar else b
                    for i, b in enumerate(bars[peer])
                )
                new_bars = tuple(
                    peer_bars if i == peer else b for i, b in enumerate(bars)
                )
                successors.append((new_pcs, new_bars, flight, done))
            elif op[0] == "wait":
                _, bar, parity = op
                if bars[c][bar][0] != parity:
                    successors.append((new_pcs, bars, flight, done))
            elif op[0] == "issue":
                adds = frozenset((c, row, pj) for row, pj in op[1])
                successors.append((new_pcs, bars, flight | adds, done))
            else:
                assert op[0] == "drain"
                if not any(a[0] == c for a in flight):
                    successors.append((new_pcs, bars, flight, done))
        last = dict(done)
        for add in flight:
            _, row, pj = add
            if last.get(row, -1) > pj:
                out_of_order = True
            new_last = dict(last)
            new_last[row] = max(last.get(row, -1), pj)
            successors.append((pcs, bars, flight - {add}, frozenset(new_last.items())))
        finished = all(pcs[c] == len(programs[c]) for c in (0, 1)) and not flight
        if not successors and not finished:
            stuck = True
        stack.extend(successors)
    return Outcome(stuck=stuck, out_of_order=out_of_order)


def run_pair_handoff(
    walks: list[PairWalk],
    tile: int,
    mode: HandoffMode,
    rank1_skips_handoff_without_work: bool = False,
) -> Outcome:
    programs = (
        compile_pair_handoff(walks, 0, tile, mode),
        compile_pair_handoff(walks, 1, tile, mode, rank1_skips_handoff_without_work),
    )
    return explore_pair_handoff(programs)


def rows_shared_across_ranks(walks: list[PairWalk], tile: int) -> bool:
    """Whether some row of one walk and q head gets additions from two pairs
    issued by different CTAs."""
    for walk in walks:
        owners: dict[int, set[int]] = {}
        for pair in walk.pairs:
            for m in range(pair.lo, pair.hi):
                for rank in (0, 1):
                    for row in two_cta_half_rows(pair, m, rank, tile):
                        owners.setdefault(row, set()).add(rank)
        if any(len(ranks) == 2 for ranks in owners.values()):
            return True
    return False


def random_pair_walks(rng: random.Random, tile: int) -> list[PairWalk]:
    walks = []
    for _ in range(rng.randint(1, 3)):
        pairs = []
        for _ in range(rng.randint(1, 4)):
            start = rng.randint(0, tile)
            end = start + rng.randint(1, 2 * tile)
            num_tiles = -(-(end - start) // tile)
            lo = rng.randint(0, num_tiles - 1)
            hi = rng.randint(lo + 1, num_tiles) if rng.random() < 0.8 else lo
            pairs.append(MergedPair(start, end, lo, hi))
        walks.append(PairWalk(tuple(pairs), q_heads=rng.choice((1, 1, 2))))
    return walks


class TestRangeMergeTwoCtaPairHandoff(unittest.TestCase):
    """bwd RangeMerge under 2-CTA: the CTAs of a cluster add the two row
    halves of every Q tile, so two pairs of one writer can add to one row
    from different CTAs. Every pair from the second on hands off between
    the CTAs: each arrives on the peer's barrier at the pair's start, after
    its earlier additions landed, and waits on its own before its first
    addition of the pair."""

    _M = 4  # Q tile rows; each CTA adds two rows of a tile

    def _walks(self) -> list[PairWalk]:
        # Pair counts 3, 1, 2 (the last walk over two q heads): handshakes
        # k = 0..3 cover both barriers in both parities, and k carries
        # across walks and q heads. The second pair of the first walk has no
        # work and waits at its start. Pairs starting 2 rows apart give a
        # row to CTA 1 in one pair and to CTA 0 in the next.
        return [
            PairWalk(
                (
                    MergedPair(0, 8, 0, 2),
                    MergedPair(1, 5, 1, 1),
                    MergedPair(2, 10, 0, 2),
                )
            ),
            PairWalk((MergedPair(0, 4, 0, 1),)),
            PairWalk((MergedPair(1, 9, 0, 2), MergedPair(3, 7, 0, 1)), q_heads=2),
        ]

    def test_range_merge_two_cta_pair_handoff_orders_shared_rows(self):
        """With the two-barrier handshake every interleaving and every
        completion order of the additions ends, and every row's additions
        complete in pair order, on walks whose shared rows change CTA
        between pairs."""
        walks = self._walks()
        self.assertTrue(rows_shared_across_ranks(walks, self._M))
        self.assertEqual(
            run_pair_handoff(walks, self._M, HandoffMode.TWO_BARRIERS),
            Outcome(stuck=False, out_of_order=False),
        )

    def test_range_merge_two_cta_pair_handoff_random_walks(self):
        """Random walks of unaligned, overlapping and work-free pairs: the
        handshake orders every row in every interleaving; without it, the
        walks whose rows change CTA between pairs merge some row out of
        order."""
        rng = random.Random(5)
        shared = 0
        for _ in range(100):
            walks = random_pair_walks(rng, self._M)
            self.assertEqual(
                run_pair_handoff(walks, self._M, HandoffMode.TWO_BARRIERS),
                Outcome(stuck=False, out_of_order=False),
                f"{walks=}",
            )
            if rows_shared_across_ranks(walks, self._M):
                shared += 1
                self.assertTrue(
                    run_pair_handoff(walks, self._M, HandoffMode.NONE).out_of_order,
                    f"{walks=}",
                )
        self.assertGreater(shared, 30)

    def test_range_merge_two_cta_without_handoff_merges_out_of_order(self):
        """Control: without the handshake the two CTAs add a shared row in
        either order."""
        outcome = run_pair_handoff(self._walks(), self._M, HandoffMode.NONE)
        self.assertTrue(outcome.out_of_order)
        self.assertFalse(outcome.stuck)

    def test_range_merge_two_cta_one_handoff_barrier_deadlocks(self):
        """Control: with one barrier the peer can complete handshake k + 1
        on it before this CTA waits for k; two completions read as none and
        the wait never ends."""
        outcome = run_pair_handoff(self._walks(), self._M, HandoffMode.ONE_BARRIER)
        self.assertTrue(outcome.stuck)

    def test_range_merge_two_cta_unequal_handoff_counts_deadlock(self):
        """Control: a CTA that skips the handshake of pairs without work
        leaves its peer waiting for an arrival that never comes; both CTAs
        must hand off at the same pairs."""
        outcome = run_pair_handoff(
            self._walks(),
            self._M,
            HandoffMode.TWO_BARRIERS,
            rank1_skips_handoff_without_work=True,
        )
        self.assertTrue(outcome.stuck)


if __name__ == "__main__":
    unittest.main()
