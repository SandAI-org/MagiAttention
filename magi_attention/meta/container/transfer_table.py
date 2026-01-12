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

from dataclasses import dataclass
from typing import Any, Iterator

from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.utils import nvtx


class AttnRangeWithRank(AttnRange):
    def __init__(self, rank_set: set[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank_set = rank_set


class GroupCastRanges(AttnRanges):
    def __init__(
        self,
        cp_size: int,
        ranges_per_rank: list[AttnRanges],
        split: bool = True,
    ):
        super().__init__()

        assert len(ranges_per_rank) == cp_size
        self._ranges: list[AttnRangeWithRank] = []  # type: ignore

        for cp_rank, ranges in enumerate(ranges_per_rank):
            for r in ranges:
                self._ranges.append(
                    AttnRangeWithRank(rank_set={cp_rank}, start=r.start, end=r.end)
                )

        # sort by attn_range.start
        self._ranges.sort(key=lambda attn_range: attn_range.start)

        if split:
            self._split()

    @nvtx.instrument_nvtx
    def _split(self) -> None:
        """Split the ranges group as fragmented as possible"""

        if len(self._ranges) <= 1:
            return

        new_ranges: list[AttnRangeWithRank] = []

        # tell which original range cover this interval
        all_points = self.points
        for i in range(len(all_points) - 1):
            p1, p2 = all_points[i], all_points[i + 1]

            # find all ranges that cover this interval
            cover_rank_set = set()
            for r in self._ranges:
                if r.start <= p1 and r.end >= p2:
                    cover_rank_set.update(r.rank_set)

            if cover_rank_set:  # if there is a range that covers this interval
                new_ranges.append(
                    AttnRangeWithRank(rank_set=cover_rank_set, start=p1, end=p2)
                )

        self._ranges = new_ranges

    def __iter__(self) -> Iterator[AttnRangeWithRank]:
        return iter(self._ranges)


from magi_attention import is_cpp_backend_enable  # noqa: E402

if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRangeWithRank as _AttnRangeWithRank
        from magi_attention.magi_attn_ext import GroupCastRanges as _GroupCastRanges

        AttnRangeWithRank = _AttnRangeWithRank  # type: ignore[misc, assignment] # noqa: F811
        GroupCastRanges = _GroupCastRanges  # type: ignore[misc, assignment] # noqa: F811
    except ImportError:
        pass


@dataclass
class TransferInfo:
    k_ranges_global_recv_from_per_rank: list[AttnRanges]
    k_ranges_local_recv_from_per_rank: list[AttnRanges]
    k_ranges_global_send_to_per_rank: list[AttnRanges]
    k_ranges_local_send_to_per_rank: list[AttnRanges]

    group_cast_ranges_global_transfer: GroupCastRanges
    group_cast_ranges_local_send_to: GroupCastRanges


@dataclass
class TableEntry:
    """The entry dataclass for transfer table,
    where:
        1. k_ranges_global: global k ranges to send w.r.t. send rank's dispatch meta
        2. k_ranges_local_in_send_buf: local k ranges to send w.r.t. send rank's send buf
        3. k_ranges_local_in_recv_buf: local k ranges to send w.r.t. recv rank's recv buf
    """

    k_ranges_global: AttnRanges
    k_ranges_local_in_send_buf: AttnRanges
    k_ranges_local_in_recv_buf: AttnRanges


class TransferTable:
    """The transfer table class, maintaining [cp_size, cp_size] entries,
    where table[send_rank][recv_rank] is the send entry from send_rank to recv_rank

    Therefore:
        1. we can get the send args for group collective
            using 'k_ranges_local_in_send_buf' in the row of table[this_rank][...]
        2. we can get the recv args for group collective
            using 'k_ranges_local_in_recv_buf' in the column of table[...][this_rank]
    """

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        self._transfer_table: list[list[TableEntry]] = []

        # init each entry in the transfer table
        for send_rank in range(cp_size):
            self._transfer_table.append([])
            for recv_rank in range(cp_size):
                self._transfer_table[send_rank].append(
                    TableEntry(
                        k_ranges_global=AttnRanges(),
                        k_ranges_local_in_send_buf=AttnRanges(),
                        k_ranges_local_in_recv_buf=AttnRanges(),
                    )
                )

    # get
    def get_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_global

    def get_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf

    def get_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> AttnRanges:
        return self._transfer_table[send_rank][recv_rank].k_ranges_local_in_recv_buf

    # append
    def append_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_global.append(k_range)

    def append_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
        k_range: AttnRange,
    ) -> None:
        self._transfer_table[send_rank][recv_rank].k_ranges_local_in_send_buf.append(
            k_range
        )

    # sort
    def sort_k_ranges_global(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_global.sort()

    def sort_k_ranges_local_in_send_buf(
        self,
        send_rank: int,
        recv_rank: int,
    ) -> None:
        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf = self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_send_buf.sort()

    # make
    def make_k_ranges_local_in_recv_buf(
        self,
        send_rank: int,
        recv_rank: int,
        remote_k_ranges_global_for_recv_rank: AttnRanges,
    ) -> None:
        """Construct local k_ranges w.r.t. recv rank's recv buffer
        from host global k_ranges to send from send_rank to recv_rank
        and remote global k_ranges to recv from send_rank to recv_rank

        NOTE: this is the special attribute that should NOT be passed in from outside,
        but ONLY constructed internally
        """

        self._transfer_table[send_rank][
            recv_rank
        ].k_ranges_local_in_recv_buf = remote_k_ranges_global_for_recv_rank.make_ranges_local(
            self._transfer_table[send_rank][recv_rank].k_ranges_global,
            is_self_merged=True,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TransferTable):
            return False

        return (self.cp_size, self._transfer_table) == (
            other.cp_size,
            other._transfer_table,
        )
