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

from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

from magi_attention.utils import is_list_value_all, repr_matrix, vis_matrix

from .enum import AttnMaskType
from .range import AttnRange
from .ranges import AttnRanges


class AttnMask(nn.Module):
    """A dataclass to represent a attn mask as a 2d matrix with meta info in each cell
    and provide some helpful attributes and functions to fetch
    """

    _can_instantiate = False
    meta_info_dim_size = 1
    mask_flag_dim_idx = 0
    masked_flag = 0
    unmasked_flag = 1
    device = "cpu"

    def __new__(cls, *args, **kwargs):
        if not cls._can_instantiate:
            raise RuntimeError("Please use the factory methods to create an instance.")
        return super().__new__(cls)

    def __init__(
        self,
        mask_tensor: torch.Tensor,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
        total_seqlen_q: int,
        total_seqlen_k: int,
    ) -> None:
        """NOTE: user should not call this constructor directly,
        but use provided factory methods instead, including `from_ranges`, `from_mask`
        """
        super().__init__()

        self.mask_tensor = mask_tensor
        self.q_ranges = q_ranges
        self.k_ranges = k_ranges
        self.attn_mask_type = attn_mask_type
        self.total_seqlen_q = total_seqlen_q
        self.total_seqlen_k = total_seqlen_k

        self.mask_flag_array = self.mask_tensor[
            ..., self.__class__.mask_flag_dim_idx
        ].numpy()
        self.mask_flag_array = np.where(
            self.mask_flag_array == self.__class__.masked_flag,
            self.__class__.masked_flag,
            self.__class__.unmasked_flag,
        ).astype(np.int32)

        self._is_pure_full = None
        self._is_pure_causal = None
        self._is_empty = None

    def tuples(self) -> Iterable[tuple[AttnRange, AttnRange, AttnMaskType]]:
        for q_range, k_range, mask_type in zip(
            self.q_ranges,
            self.k_ranges,
            self.attn_mask_type,
        ):
            yield q_range, k_range, mask_type

    @classmethod
    def from_ranges(
        cls,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
        total_seqlen_q: int | None = None,
        total_seqlen_k: int | None = None,
    ) -> "AttnMask":
        """The (most common) factory method to construct a AttnMask instance,
        with q_ranges, k_ranges and corr. attn_mask_type

        Args:
            q_ranges (AttnRanges): the query ranges
            k_ranges (AttnRanges): the key ranges
            attn_mask_type (list[AttnMaskType]): the attn mask type list
            NOTE: the length of q_ranges, k_ranges and attn_mask_type should be equal
            total_seqlen_q (int | None): the total seqlen of query (i.e. number of rows)
            total_seqlen_k (int | None): the total seqlen of key (i.e. number of columns)
        Returns:
            AttnMask: the attn mask instance
        """
        assert (
            len(q_ranges) == len(k_ranges) == len(attn_mask_type)
        ), f"The length should be equal, but got: {len(q_ranges)=}, {len(k_ranges)=}, {len(attn_mask_type)=}"  # noqa

        with cls.can_instantiate_ctx():
            total_seqlen_q = (
                total_seqlen_q if total_seqlen_q is not None else q_ranges.end
            )
            total_seqlen_k = (
                total_seqlen_k if total_seqlen_k is not None else k_ranges.end
            )

            mask_tensor = torch.zeros(
                (total_seqlen_q, total_seqlen_k, cls.meta_info_dim_size),
                dtype=torch.int32,
            ).fill_(cls.masked_flag)

            for sample_idx, (q_range, k_range, mask_type) in enumerate(
                zip(q_ranges, k_ranges, attn_mask_type)
            ):
                if mask_type == AttnMaskType.FULL:
                    mask_tensor[
                        q_range.start : q_range.end,
                        k_range.start : k_range.end,
                        cls.mask_flag_dim_idx,
                    ] = cls.unmasked_flag
                elif mask_type == AttnMaskType.CAUSAL:
                    causal_mask = cls.make_causal_mask(
                        q_range.seqlen,
                        k_range.seqlen,
                        dtype=torch.int32,
                        device=cls.device,
                    )
                    causal_mask[causal_mask == 0] = cls.masked_flag
                    causal_mask[causal_mask == 1] = cls.unmasked_flag

                    mask_tensor[
                        q_range.start : q_range.end,
                        k_range.start : k_range.end,
                        cls.mask_flag_dim_idx,
                    ] = causal_mask
                else:
                    raise ValueError(f"Invalid mask type: {mask_type}")

            attn_mask_instance = AttnMask(
                mask_tensor=mask_tensor,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=total_seqlen_q,
                total_seqlen_k=total_seqlen_k,
            )

        return attn_mask_instance

    @classmethod
    def from_mask(
        cls,
        mask: list[list[int]] | torch.Tensor,
    ) -> "AttnMask":
        """The (less common) factory method to construct a AttnMask instance,
        with a 2d int32 mask tensor, where the nonzero cell indicates unmasked position,
        while the zero cell indicates masked position

        Args:
            mask (list[list[int]] | torch.Tensor): the 2d int32 mask tensor

        Returns:
            AttnMask: the attn mask instance
        """
        mask = torch.as_tensor(mask, dtype=torch.int32, device=cls.device)
        cls._check_mask_valid(mask)

        with cls.can_instantiate_ctx():
            # collect k_range for each row
            k_range_for_each_row = AttnRanges()
            for row in range(mask.shape[0]):
                col_idxs_in_this_row = mask[row, :].nonzero(as_tuple=True)[0]
                if col_idxs_in_this_row.numel() == 0:
                    k_range_for_each_row.append(AttnRange(row, row))
                    continue
                k_range_for_each_row.append(
                    AttnRange(
                        col_idxs_in_this_row.min().item(),
                        col_idxs_in_this_row.max().item() + 1,
                    )
                )

            # merge rows together to build tuples of (q_range, k_range, attn_mask_type)
            q_ranges, k_ranges, attn_mask_type = AttnRanges(), AttnRanges(), []

            def _add_new_row(row: int, k_range_for_this_row: AttnRange):
                q_ranges.append(AttnRange(row, row + 1))
                k_ranges.append(k_range_for_this_row)
                attn_mask_type.append(
                    # if empty, then it can only belong to a causal mask
                    AttnMaskType.CAUSAL
                    if k_range_for_this_row.is_empty()
                    else AttnMaskType.FULL
                )

            def _is_two_rows_causal(
                top_row_k_range: AttnRange,
                bot_row_k_range: AttnRange,
            ) -> bool:
                return (
                    top_row_k_range.start == bot_row_k_range.start
                    and top_row_k_range.end + 1 == bot_row_k_range.end
                )

            for row, k_range_for_this_row in enumerate(k_range_for_each_row):
                if row == 0:  # the first row
                    _add_new_row(row, k_range_for_this_row)
                    continue

                last_q_range, last_k_range, last_attn_mask_type = (
                    q_ranges[-1],
                    k_ranges[-1],
                    attn_mask_type[-1],
                )
                if last_attn_mask_type == AttnMaskType.FULL:
                    if k_range_for_this_row == last_k_range:
                        # the current full mask can be extended to this row
                        q_ranges[-1] = AttnRange(
                            q_ranges[-1].start, q_ranges[-1].end + 1
                        )
                    elif _is_two_rows_causal(last_k_range, k_range_for_this_row):
                        # the last row is the top row of a causal mask
                        if (
                            last_q_range.seqlen == 1
                        ):  # the last row is not part of a full mask
                            q_ranges[-1] = AttnRange(
                                q_ranges[-1].start, q_ranges[-1].end + 1
                            )
                            k_ranges[-1] = AttnRange(
                                k_ranges[-1].start, k_ranges[-1].end + 1
                            )
                            attn_mask_type[-1] = AttnMaskType.CAUSAL
                        else:  # the last row is part of a full mask
                            q_ranges[-1] = AttnRange(
                                q_ranges[-1].start, q_ranges[-1].end - 1
                            )
                            q_ranges.append(AttnRange(row - 1, row + 1))
                            k_ranges.append(k_range_for_this_row)
                            attn_mask_type.append(AttnMaskType.CAUSAL)
                    else:  # this row is the top row for a new mask
                        _add_new_row(row, k_range_for_this_row)
                elif last_attn_mask_type == AttnMaskType.CAUSAL:
                    if last_k_range.is_empty():
                        if k_range_for_this_row.is_empty():
                            # this row is still of the top empty part of a causal mask
                            q_ranges[-1] = AttnRange(
                                q_ranges[-1].start, q_ranges[-1].end + 1
                            )
                        elif k_range_for_this_row.seqlen == 1:
                            # this row is the top row of the non-empty part of a causal mask
                            q_ranges[-1] = AttnRange(
                                q_ranges[-1].start, q_ranges[-1].end + 1
                            )
                            k_ranges[-1] = k_range_for_this_row
                        else:
                            _add_new_row(row, k_range_for_this_row)
                    elif _is_two_rows_causal(last_k_range, k_range_for_this_row):
                        # the current causal mask can be extended to this row
                        q_ranges[-1] = AttnRange(
                            q_ranges[-1].start, q_ranges[-1].end + 1
                        )
                        k_ranges[-1] = AttnRange(
                            k_ranges[-1].start, k_ranges[-1].end + 1
                        )
                    else:
                        # this row is the top row for a new mask
                        _add_new_row(row, k_range_for_this_row)
                else:
                    raise ValueError(f"Invalid mask type: {last_attn_mask_type}")

            attn_mask_instance = AttnMask(
                mask_tensor=mask.unsqueeze(-1),
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=mask.shape[0],
                total_seqlen_k=mask.shape[1],
            )

        return attn_mask_instance

    @classmethod
    def _check_mask_valid(
        cls,
        mask: torch.Tensor,
    ) -> None:
        assert len(mask.shape) == 2, f"The mask should be 2d, but got: {mask.shape=}"
        unmasked_row_idxs, unmasked_col_idxs = mask.nonzero(as_tuple=True)
        for row in range(mask.shape[0]):
            col_idxs_in_this_row = unmasked_col_idxs[unmasked_row_idxs == row]
            assert col_idxs_in_this_row.numel() == 0 or (
                len(col_idxs_in_this_row)
                == col_idxs_in_this_row.max() - col_idxs_in_this_row.min() + 1
            ), f"The unmasked col idxs for {row}-th row should be continuous, but got: {col_idxs_in_this_row=}"  # noqa

    def _check_sub_range_valid(
        self,
        q_range: AttnRange,
        k_range: AttnRange,
    ) -> None:
        assert (
            q_range.end <= self.total_seqlen_q
        ), f"The {q_range.end=} should be no greater than {self.total_seqlen_q=}"  # noqa
        assert (
            k_range.end <= self.total_seqlen_k
        ), f"The {k_range.end=} should be no greater than {self.total_seqlen_k=}"  # noqa

    def calc_sub_area(
        self,
        q_range: AttnRange,
        k_range: AttnRange,
    ) -> int:
        self._check_sub_range_valid(q_range, k_range)

        return self.mask_flag_array[
            q_range.start : q_range.end,
            k_range.start : k_range.end,
        ].sum()

    def make_sub_mask(
        self,
        q_range: AttnRange,
        k_range: AttnRange,
    ) -> "AttnMask":
        """The method to make a sub mask from the self mask,
        with a q_range and a k_range indicating a rectangle area to shard the self mask

        Args:
            q_range (AttnRange): The q_range indicating the row idxs of the sub mask
            k_range (AttnRange): The k_range indicating the col idxs of the sub mask

        Returns:
            AttnMask: The sub mask
        """
        self._check_sub_range_valid(q_range, k_range)

        sub_mask_tensor = self.mask_tensor[
            q_range.start : q_range.end,
            k_range.start : k_range.end,
            self.__class__.mask_flag_dim_idx,
        ]

        return AttnMask.from_mask(sub_mask_tensor)

    @property
    def area(self) -> int:
        return self.mask_flag_array.sum()

    def is_square(self) -> bool:
        return self.total_seqlen_q == self.total_seqlen_k

    def is_pure_full(self) -> bool:
        if self._is_pure_full is None:
            self._is_pure_full = (
                self.area == self.total_seqlen_q * self.total_seqlen_k  # type: ignore
            )

        return self._is_pure_full  # type: ignore

    def is_pure_causal(self) -> bool:
        if self._is_pure_causal is None:
            pure_causal_mask = self.make_causal_mask(
                seqlen_q=self.total_seqlen_q,
                seqlen_k=self.total_seqlen_k,
                dtype=torch.int32,
                device=self.__class__.device,
            ).numpy()

            self._is_pure_causal = (self.mask_flag_array == pure_causal_mask).all()

        return self._is_pure_causal  # type: ignore[return-value]

    def is_varlen_full(self) -> bool:
        return (
            is_list_value_all(self.attn_mask_type, AttnMaskType.FULL)
            and self.q_ranges.is_cu_seqlens(self.total_seqlen_q)
            and self.k_ranges.is_cu_seqlens(self.total_seqlen_k)
        )

    def is_varlen_causal(self) -> bool:
        return (
            is_list_value_all(self.attn_mask_type, AttnMaskType.CAUSAL)
            and self.q_ranges.is_cu_seqlens(self.total_seqlen_q)
            and self.k_ranges.is_cu_seqlens(self.total_seqlen_k)
        )

    def is_empty(self) -> bool:
        if self._is_empty is None:
            self._is_empty = self.area == 0  # type: ignore

        return self._is_empty  # type: ignore

    @staticmethod
    def make_causal_mask(
        seqlen_q: int,
        seqlen_k: int,
        align: str = "bottom-right",
        dtype=torch.int32,
        device: str = "cpu",
    ) -> torch.Tensor:
        max_seqlen = max(seqlen_q, seqlen_k)
        causal_mask = torch.tril(torch.ones((max_seqlen, max_seqlen))).to(
            dtype=dtype, device=device
        )

        if align == "bottom-right":
            causal_mask = causal_mask[-seqlen_q:, -seqlen_k:]
        elif align == "top-left":
            causal_mask = causal_mask[:seqlen_q, :seqlen_k]
        else:
            raise ValueError(f"Invalid alignment mode: {align}")

        return causal_mask

    def visualize(self, save_path: str | None = None) -> None:
        """Visualize the attention mask as a boolean matrix."""
        vis_matrix(
            matrix=self.mask_flag_array,
            title="Attention Mask",
            xlabel="Key/Value",
            ylabel="Query",
            val_ticks=[self.__class__.masked_flag, self.__class__.unmasked_flag],
            format_ticks=lambda x, pos: "unmasked"
            if x == self.__class__.unmasked_flag
            else ("masked" if x == self.__class__.masked_flag else f"{x}"),
            save_path=save_path,
        )

    def __repr__(self) -> str:
        repr_str = [""]

        repr_str.append(
            f"{self.total_seqlen_q=} | {self.total_seqlen_k=} | {self.area=}"
        )
        repr_str.append(f"{self.q_ranges=}")
        repr_str.append(f"{self.k_ranges=}")
        repr_str.append(f"{self.attn_mask_type=}")

        mask_flag_repr_str = repr_matrix(self.mask_flag_array)
        repr_str.append("attn_mask=")
        repr_str.append(mask_flag_repr_str)

        return "\n".join(repr_str)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, AttnMask):
            return (
                torch.equal(self.mask_tensor, other.mask_tensor)
                and self.q_ranges == other.q_ranges
                and self.k_ranges == other.k_ranges
                and self.attn_mask_type == other.attn_mask_type
            )
        return False

    @classmethod
    @contextmanager
    def can_instantiate_ctx(cls):
        cls._can_instantiate = True
        yield
        cls._can_instantiate = False
