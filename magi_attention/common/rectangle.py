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

from typing import Any, Union

from . import is_cpp_backend_enable
from .enum import AttnMaskType
from .range import AttnRange

__all__ = [
    "AttnRectangle",
]

INT_MAX = 10**9
INT_MIN = -(10**9)


class AttnRectangle:
    """
    A dataclass to manage any indices rectangle like
    [start_q, end_q) [start_k, end_k) [start_d, end_d) mask_type
    for attention computation.
    d_range is d_index = k_index - q_index diagonal line range
    """

    def __init__(
        self,
        q_range: AttnRange,
        k_range: AttnRange,
        d_range: AttnRange | None = None,
        mask_type: AttnMaskType | int = AttnMaskType.FULL,
    ) -> None:
        self._q_range = q_range
        self._k_range = k_range
        # If there is no user-defined d_range, set it to -inf ~ inf
        d_range = AttnRange(INT_MIN, INT_MAX) if d_range is None else d_range
        self._d_range = d_range

        # Get enum type mask_type for subsequent logic
        if isinstance(mask_type, AttnMaskType):
            enum_mask_type = mask_type
        elif isinstance(mask_type, int):
            # Integer to enum mapping
            enum_mask_type = AttnMaskType.from_int_type(mask_type)
        else:
            raise TypeError(
                f"mask_type must be AttnMaskType or int type, but got {type(mask_type)}"
            )

        if (
            enum_mask_type == AttnMaskType.CAUSAL
            or enum_mask_type == AttnMaskType.BICAUSAL
        ):
            # d_index end is limit by the lower right corner
            self._d_range.end = min(self._d_range.end, k_range.end - q_range.end)
        else:
            # d_index end is limit by the top right corner
            self._d_range.end = min(self._d_range.end, k_range.end - 1 - q_range.start)

        if (
            enum_mask_type == AttnMaskType.INVCAUSAL
            or enum_mask_type == AttnMaskType.BICAUSAL
        ):
            # d_index start is limit by the top left corner
            self._d_range.start = max(
                self._d_range.start, k_range.start - q_range.start
            )
        else:
            # d_index start is limit by the lower left corner
            self._d_range.start = max(
                self._d_range.start, k_range.start - (q_range.end - 1)
            )

        self.shrink_d_range()
        self.shrink_q_range()
        self.shrink_k_range()
        self.check_valid()

    @property
    def q_range(self):
        return self._q_range

    @q_range.setter
    def q_range(self, value) -> None:
        self.check_valid(q_range=value)
        self._q_range = value

    @property
    def k_range(self):
        return self._k_range

    @k_range.setter
    def k_range(self, value) -> None:
        self.check_valid(k_range=value)
        self._k_range = value

    @property
    def d_range(self):
        return self._d_range

    @d_range.setter
    def d_range(self, value) -> None:
        self.check_valid(d_range=value)
        self._d_range = value

    def is_valid(
        self,
        q_range: AttnRange | None = None,
        k_range: AttnRange | None = None,
        d_range: AttnRange | None = None,
    ) -> bool:
        q_range = self.q_range if q_range is None else q_range
        k_range = self.k_range if k_range is None else k_range
        d_range = self.d_range if d_range is None else d_range
        if (
            q_range.is_valid_open()
            and k_range.is_valid_open()
            and d_range.is_valid_close()
        ):
            return True
        return False

    def check_valid(
        self,
        q_range: AttnRange | None = None,
        k_range: AttnRange | None = None,
        d_range: AttnRange | None = None,
    ) -> None:
        q_range = self.q_range if q_range is None else q_range
        k_range = self.k_range if k_range is None else k_range
        d_range = self.d_range if d_range is None else d_range
        if not self.is_valid(q_range, k_range, d_range):
            raise ValueError(
                f"Some of the {q_range=} {k_range=} {d_range=} is invalid, no area include"
            )

    def get_valid_or_none(self) -> Union["AttnRectangle", None]:
        return self if self.is_valid() else None

    def shrink_d_range(self) -> bool:
        d_range_min = self.k_range.start - (self.q_range.end - 1)
        d_range_max = (self.k_range.end - 1) - self.q_range.start
        self.d_range.start = max(self.d_range.start, d_range_min)
        self.d_range.end = min(self.d_range.end, d_range_max)
        return self.d_range.is_valid_close()

    def shrink_q_range(self) -> bool:
        # calc intersection of d_range end diagonal line & k_range start line
        intersection_q_start = self.k_range.start - self.d_range.end
        # calc instersection of d_range start diagonal line & k_range end line
        intersection_q_end = self.k_range.end - self.d_range.start
        self.q_range.start = max(self.q_range.start, intersection_q_start)
        self.q_range.end = min(self.q_range.end, intersection_q_end)
        return self.q_range.is_valid_open()

    def shrink_k_range(self) -> bool:
        # calc intersection of d_range start diagonal line & q_range start line
        intersection_k_start = self.d_range.start + self.q_range.start
        # calc intersection of d_range end diagonal line & q_range end line
        intersection_k_end = self.d_range.end + self.q_range.end
        self.k_range.start = max(self.k_range.start, intersection_k_start)
        self.k_range.end = min(self.k_range.end, intersection_k_end)
        return self.k_range.is_valid_open()

    def clone(self) -> "AttnRectangle":
        """Clone the current rectangle efficiently"""
        new_rect = self.__class__.__new__(self.__class__)
        new_rect._q_range = self._q_range.clone()
        new_rect._k_range = self._k_range.clone()
        new_rect._d_range = self._d_range.clone()
        return new_rect

    def cut_q(
        self, cut_pos: int
    ) -> tuple[Union["AttnRectangle", None], Union["AttnRectangle", None]]:
        if cut_pos <= self.q_range.start:
            return None, self
        if cut_pos >= self.q_range.end:
            return self, None
        cut_rect_left = self.clone()
        cut_rect_right = self.clone()
        cut_rect_left.q_range.end = cut_pos
        cut_rect_right.q_range.start = cut_pos
        cut_rect_left.shrink_d_range()
        cut_rect_left.shrink_k_range()
        cut_rect_right.shrink_d_range()
        cut_rect_right.shrink_k_range()
        return cut_rect_left, cut_rect_right

    def cut_k(
        self, cut_pos: int
    ) -> tuple[Union["AttnRectangle", None], Union["AttnRectangle", None]]:
        if cut_pos <= self.k_range.start:
            return None, self
        if cut_pos >= self.k_range.end:
            return self, None
        cut_rect_left = self.clone()
        cut_rect_right = self.clone()
        cut_rect_left.k_range.end = cut_pos
        cut_rect_right.k_range.start = cut_pos
        cut_rect_left.shrink_d_range()
        cut_rect_left.shrink_q_range()
        cut_rect_right.shrink_d_range()
        cut_rect_right.shrink_q_range()
        return cut_rect_left, cut_rect_right

    def get_rect_within_q_segment(
        self,
        q_start: int,
        q_end: int,
    ) -> Union["AttnRectangle", None]:
        """
        Obtain the part of the current rectangle within the interval q range
        """
        if q_end <= self.q_range.start or q_start >= self.q_range.end:
            return None
        rect_in_seg = self.clone()
        rect_in_seg.q_range.start = max(rect_in_seg.q_range.start, q_start)
        rect_in_seg.q_range.end = min(rect_in_seg.q_range.end, q_end)
        rect_in_seg.shrink_d_range()
        rect_in_seg.shrink_k_range()
        return rect_in_seg

    def get_rect_within_k_segment(
        self,
        k_start: int,
        k_end: int,
    ) -> Union["AttnRectangle", None]:
        """
        Obtain the part of the current rectangle within the interval k range
        """
        if k_end <= self.k_range.start or k_start >= self.k_range.end:
            return None
        rect_in_seg = self.clone()
        rect_in_seg.k_range.start = max(rect_in_seg.k_range.start, k_start)
        rect_in_seg.k_range.end = min(rect_in_seg.k_range.end, k_end)
        rect_in_seg.shrink_d_range()
        rect_in_seg.shrink_q_range()
        return rect_in_seg

    def intersection_q_id_on_left_boundary(self) -> int:
        """get k_start d_start intersection, which is q id on left boundary"""
        return self.k_range.start - self.d_range.start

    def intersection_q_id_on_right_boundary(self) -> int:
        """get k_end d_end intersection, which is q id on right boundary"""
        return self.k_range.end - 1 - self.d_range.end

    def is_full(self) -> bool:
        return (
            self.d_range.start <= self.k_range.start - (self.q_range.end - 1)
            and self.d_range.end >= self.k_range.end - 1 - self.q_range.start
        )

    def is_causal(self) -> bool:
        return (
            self.d_range.start <= self.k_range.start - (self.q_range.end - 1)
            and self.d_range.end == self.k_range.end - self.q_range.end
        )

    def is_inv_causal(self) -> bool:
        return (
            self.d_range.start == self.k_range.start - self.q_range.start
            and self.d_range.end >= self.k_range.end - 1 - self.q_range.start
        )

    def is_bi_causal(self) -> bool:
        return (
            self.d_range.start == self.k_range.start - self.q_range.start
            and self.d_range.end == self.k_range.end - self.q_range.end
        )

    def to_qk_range_mask_type(
        self,
    ) -> list[tuple[AttnRange, AttnRange, int]]:
        """
        Change rectangle to q k range and mask type style.
        Use recursive logic processing.
        """
        attn_arg: list[tuple[AttnRange, AttnRange, int]] = []

        # Direct_return
        # case 1: Full Mask
        if self.is_full():
            attn_arg.append((self.q_range, self.k_range, 0))
            return attn_arg
        # case 2: Causal Mask
        if self.is_causal():
            attn_arg.append((self.q_range, self.k_range, 1))
            return attn_arg
        # case 3: Inv Causal Mask
        if self.is_inv_causal():
            attn_arg.append((self.q_range, self.k_range, 2))
            return attn_arg
        # case 4: Bi Causal Mask
        if self.is_bi_causal():
            attn_arg.append((self.q_range, self.k_range, 3))
            return attn_arg

        # left boundary (k_start): q_start ~ q_id_l is Full, q_id_l ~ q_end is Inv Causal
        q_id_l = self.intersection_q_id_on_left_boundary()
        # right boundary (k_end) : q_start ~ q_id_r is Causal, q_id_r ~ q_end is Full
        q_id_r = self.intersection_q_id_on_right_boundary()

        if (
            q_id_l < self.q_range.start
            or q_id_l >= self.q_range.end
            or q_id_r < self.q_range.start
            or q_id_r >= self.q_range.end
        ):
            raise ValueError(f"rect{self} without shrinkage call to_qk_range_mask_type")

        # no d_range start cut
        if q_id_l == self.q_range.end - 1:
            # q_start ~ q_id_r  : left full, right causal
            # q_id_r + 1 ~ q_end    : left full, right full
            up_rect, down_rect = self.cut_q(q_id_r + 1)
            if up_rect is not None:
                attn_arg.extend(up_rect.to_qk_range_mask_type())
            if down_rect is not None:
                attn_arg.extend(down_rect.to_qk_range_mask_type())
            return attn_arg

        # no d_range end cut
        if q_id_r == self.q_range.start:
            # q_start ~ q_id_l - 1  : left full, right full
            # q_id_l ~ q_end    : left inv causal, right full
            up_rect, down_rect = self.cut_q(q_id_l)
            if up_rect is not None:
                attn_arg.extend(up_rect.to_qk_range_mask_type())
            if down_rect is not None:
                attn_arg.extend(down_rect.to_qk_range_mask_type())
            return attn_arg

        if q_id_r <= q_id_l:
            # q_start ~ q_id_r - 1  : left full, right causal
            # q_id_r ~ q_id_l - 1   : left full, right full
            # q_id_l ~ q_end    : left inv causal, right full
            up_rect, down_rect = self.cut_q(q_id_l)
            if up_rect is not None:
                attn_arg.extend(up_rect.to_qk_range_mask_type())
            if down_rect is not None:
                attn_arg.extend(down_rect.to_qk_range_mask_type())
            return attn_arg
        elif q_id_r == q_id_l + 1:
            # q_start ~ q_id_l  : left full, right causal
            # q_id_r ~ q_end    : left inv causal, right full
            up_rect, down_rect = self.cut_q(q_id_r)
            if up_rect is not None:
                attn_arg.extend(up_rect.to_qk_range_mask_type())
            if down_rect is not None:
                attn_arg.extend(down_rect.to_qk_range_mask_type())
            return attn_arg
        else:  # q_id_r > q_id_l + 1
            # q_start ~ q_id_l - 1  : left full, right causal
            # q_id_l ~ q_id_r   : left inv causal, right causal
            # q_id_r + 1 ~ q_end    : left inv causal, right full
            up_rect, mid_rect = self.cut_q(q_id_l)
            if up_rect is not None:
                attn_arg.extend(up_rect.to_qk_range_mask_type())
            if mid_rect is not None:
                mid_rect, down_rect = mid_rect.cut_q(q_id_r + 1)
                if mid_rect is not None:
                    attn_arg.extend(mid_rect.to_qk_range_mask_type())
                if down_rect is not None:
                    attn_arg.extend(down_rect.to_qk_range_mask_type())
            return attn_arg

    def area(self) -> int:
        return self.count_areas(
            self._q_range.start,
            self._q_range.end,
            self._k_range.start,
            self._k_range.end,
            self._d_range.start,
            self._d_range.end,
        )

    def count_areas(self, lq, rq, lk, rk, ld, rd) -> int:
        """
        Calculate the number of integer points (q, k) that satisfy the conditions
        Time complexity O(1)

        Parameters:
        lq, rq: Range of q [lq, rq)
        lk, rk: Range of k [lk, rk)
        ld, rd: Range of k-q [ld, rd]

        Returns:
        The number of points satisfying all conditions
        """
        # Handle boundary cases: invalid ranges
        if rq <= lq or rk <= lk or rd < ld:
            return 0

        # Convert to integer closed intervals [Q1, Q2] and [K1, K2]
        Q1, Q2 = lq, rq - 1  # Valid integer range for q
        K1, K2 = lk, rk - 1  # Valid integer range for k

        # Condition transformation: ld <= k-q <= rd -> q+ld <= k <= q+rd
        # For each q, the valid range of k is [max(K1, q+ld), min(K2, q+rd)]
        # The range is valid if and only if max(...) <= min(...)

        # Calculate critical points (dividing intervals for q)
        a = K1 - ld  # When q < a, q+ld < K1 -> lower bound of k is K1
        b = K2 - rd  # When q > b, q+rd > K2 -> upper bound of k is K2

        total = 0

        # Divide q into intervals, ensuring no overlap
        if a <= b:
            # Interval 1: q ∈ [Q1, min(a-1, Q2)] -> lower bound of k is K1, upper bound is q+rd
            q_start = Q1
            q_end = min(a - 1, Q2)
            if q_start <= q_end:
                # Valid q must satisfy: q+rd ≥ K1 (otherwise k range is invalid)
                m = max(q_start, K1 - rd)
                n = q_end
                if m <= n:
                    # Number of k for each q: (q+rd) - K1 + 1 = q + (rd - K1 + 1)
                    c = rd - K1 + 1
                    sum_q = (n * (n + 1) // 2) - ((m - 1) * m // 2)
                    sum_c = c * (n - m + 1)
                    total += sum_q + sum_c

            # Interval 2: q ∈ [max(a, Q1), min(b, Q2)] -> k range is [q+ld, q+rd] (completely within [K1,K2])
            q_start = max(a, Q1)
            q_end = min(b, Q2)
            if q_start <= q_end:
                # Number of k for each q: rd - ld + 1 (constant)
                k_count = rd - ld + 1
                total += k_count * (q_end - q_start + 1)

            # Interval 3: q ∈ [max(b+1, Q1), Q2] -> lower bound of k is q+ld, upper bound is K2
            q_start = max(b + 1, Q1)
            q_end = Q2
            if q_start <= q_end:
                # Valid q must satisfy: q+ld <= K2 (otherwise k range is invalid)
                m = q_start
                n = min(q_end, K2 - ld)
                if m <= n:
                    # Number of k for each q: K2 - (q+ld) + 1 = (K2 - ld + 1) - q
                    c = K2 - ld + 1
                    sum_c = c * (n - m + 1)
                    sum_q = (n * (n + 1) // 2) - ((m - 1) * m // 2)
                    total += sum_c - sum_q
        else:
            # When a > b, three intervals
            # Interval 1: q ∈ [Q1, min(b, Q2)] -> lower bound of k is K1, upper bound is q+rd
            q_start = Q1
            q_end = min(b, Q2)
            if q_start <= q_end:
                m = max(q_start, K1 - rd)
                n = q_end
                if m <= n:
                    c = rd - K1 + 1
                    sum_q = (n * (n + 1) // 2) - ((m - 1) * m // 2)
                    sum_c = c * (n - m + 1)
                    total += sum_q + sum_c

            # Interval 2: q ∈ [max(b+1, Q1), min(a-1, Q2)] -> k range is [K1, K2] (completely contained)
            q_start = max(b + 1, Q1)
            q_end = min(a - 1, Q2)
            if q_start <= q_end:
                # Number of k for each q: K2 - K1 + 1 (constant)
                if K1 <= K2:
                    k_count = K2 - K1 + 1
                    total += k_count * (q_end - q_start + 1)

            # Interval 3: q ∈ [max(a, Q1), Q2] -> lower bound of k is q+ld, upper bound is K2
            q_start = max(a, Q1)
            q_end = Q2
            if q_start <= q_end:
                m = q_start
                n = min(q_end, K2 - ld)
                if m <= n:
                    c = K2 - ld + 1
                    sum_c = c * (n - m + 1)
                    sum_q = (n * (n + 1) // 2) - ((m - 1) * m // 2)
                    total += sum_c - sum_q

        return total

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnRectangle):
            return (
                self._q_range == other._q_range
                and self._k_range == other._k_range
                and self._d_range == other._d_range
            )
        return False

    def __hash__(self) -> int:
        return hash((self._q_range, self._k_range, self._d_range))

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self._q_range} x {self._k_range} x {self._d_range}"


if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRectangle as _AttnRectangle

        AttnRectangle = _AttnRectangle  # type: ignore[misc, assignment] # noqa: F811
    except ImportError:
        pass
