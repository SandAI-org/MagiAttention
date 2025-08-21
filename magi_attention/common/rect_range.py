# Copyright (c) 2025 SandAI. All Rights Reserved.
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

from typing import Any, Union

from .range import AttnRange, NaiveRange, RangeError


class AttnRectRange(AttnRange):
    """A dataclass to manage any indices range like [start, end) for rectangle attention computation"""

    def __init__(self, start: int, end: int) -> None:
        super().__init__(start, end)

    @classmethod
    def from_parent(cls, attn_range: AttnRange):
        return cls(attn_range.start, attn_range.end)

    def to_parent(self) -> AttnRange:
        return AttnRange(self.start, self.end)

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, value) -> None:
        self._start = value

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, value) -> None:
        self._end = value

    @staticmethod
    def from_range(
        attn_range: Union[NaiveRange, list[int], "AttnRectRange", AttnRange],
        check: bool = False,
    ) -> "AttnRectRange":
        if isinstance(attn_range, AttnRectRange):  # just copy
            res = attn_range
        elif isinstance(attn_range, AttnRange):
            res = AttnRectRange(start=attn_range.start, end=attn_range.end)
        else:
            res = AttnRectRange(start=attn_range[0], end=attn_range[1])

        if check:
            res.check_valid()

        return res

    def offset(self, offset: int) -> "AttnRectRange":
        return AttnRectRange(start=self._start + offset, end=self._end + offset)

    def truncate(
        self, start: int | None = None, end: int | None = None
    ) -> "AttnRectRange":
        start = self._start if start is None else max(self._start, start)
        end = self._end if end is None else min(self._end, end)

        # NOTE: if start > end, then return empty range: [start, start)
        return AttnRectRange(start=start, end=max(start, end))

    # range is [start, end] closed interval
    def is_valid_close(self, start: int | None = None, end: int | None = None) -> bool:
        start = self._start if start is None else start
        end = self._end if end is None else end

        return start <= end

    # range is [start, end) Left-closed and right-open intervals
    def is_valid_open(self, start: int | None = None, end: int | None = None) -> bool:
        start = self._start if start is None else start
        end = self._end if end is None else end

        return start < end

    def check_valid(self, start: int | None = None, end: int | None = None) -> None:
        if not self.is_valid_close(start, end):
            raise RangeError(
                f"The attn_rect_range {(start, end)} is invalid against the rule: 'start <= end'"
            )

    def __len__(self) -> int:
        return self.seqlen

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRectRange):
            return self._start == other._start and self._end == other._end
        return False

    def __hash__(self) -> int:
        return hash((self._start, self._end))

    def __repr__(self) -> str:
        return f"[{self._start}, {self._end})"
