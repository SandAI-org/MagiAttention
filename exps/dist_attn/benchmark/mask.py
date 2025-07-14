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

import random

from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.utils import (
    generate_seqlen_for_one_time,
    generate_seqlens,
    seqlens2cu_seqlens,
    varlen_long_seqlen_distribution,
    varlen_short_seqlen_distribution,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges


class MaskGenerator:
    """This is a generator for multiple flash masks, it can be used with
    generate function with flash mask type and data distribution.
    """

    def __init__(self):
        self.generate_function = {
            FlashMaskType.FULL: self.generate_full_mask,
            FlashMaskType.CAUSAL: self.generate_causal_mask,
            FlashMaskType.CAUSAL_DOCUMENT: self.generate_causal_document_mask,
            FlashMaskType.FULL_DOCUMENT: self.generate_full_document_mask,
            FlashMaskType.SHARE_QUESTION: self.generate_share_question_mask,
            FlashMaskType.CAUSAL_BLOCKWISE: self.generate_causal_blockwise_mask,
            FlashMaskType.PREFIX_LM_CAUSAL: self.generate_prefix_lm_causal_mask,
            FlashMaskType.PREFIX_LM_DOCUMENT: self.generate_prefix_lm_document_mask,
            FlashMaskType.QK_SPARSE: self.generate_qk_sparse_mask,
        }

    def generate(
        self,
        flash_mask_type: FlashMaskType,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        to_attn_ranges: bool = True,
        rng: random.Random | None = None,
    ) -> (
        tuple[list[list[int]], list[list[int]], list[bool]]
        | tuple[AttnRanges, AttnRanges, list[AttnMaskType]]
    ):
        """generate a mask with mask type and data distribution

        Returns:
            The returned triple respectively describes the range of the query and key,
            as well as the type of Mask composed of them.
            NOTE: according to to_attn_ranges, it has two return types:
            1. AttnRanges, AttnRanges, list[AttnMaskType]
            2. list[list[int]], list[list[int]], list[bool]
        """
        q_ranges, k_ranges, attn_mask_type = self.generate_function[flash_mask_type](
            seqlen_distribute=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )

        if to_attn_ranges:
            q_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=q_ranges)
            k_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=k_ranges)
            attn_mask_type_: list[AttnMaskType] = [
                AttnMaskType.CAUSAL if mask_type else AttnMaskType.FULL
                for mask_type in attn_mask_type
            ]
            return (q_ranges_, k_ranges_, attn_mask_type_)

        return (q_ranges, k_ranges, attn_mask_type)

    def generate_causal_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate document causal mask (varlen causal mask)"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [True] * len(seqlens)

        return (ranges, ranges, is_causal_mapping)

    def generate_causal_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate causal mask"""
        ranges = [[0, total_seqlen]]
        is_causal_mapping = [True]

        return (ranges, ranges, is_causal_mapping)

    def generate_full_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate document full maks (varlen full mask)"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])

        is_causal_mapping = [False] * len(seqlens)

        return (ranges, ranges, is_causal_mapping)

    def generate_full_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate causal mask"""
        ranges = [[0, total_seqlen]]
        is_causal_mapping = [False]

        return (ranges, ranges, is_causal_mapping)

    def generate_share_question_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate share question mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[bool] = []
        for i in range(len(seqlens)):
            if i == 1:
                q_ranges[0] = [0, cu_seqlens[i + 1]]
                k_ranges[0] = [0, cu_seqlens[i + 1]]

                q_ranges.append([cu_seqlens[i + 1], total_seqlen])
                k_ranges.append([0, cu_seqlens[i]])
                is_causal_mapping.append(False)
            else:
                q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                is_causal_mapping.append(True)

        return (q_ranges, k_ranges, is_causal_mapping)

    def generate_causal_blockwise_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate causal blockwise mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        for i in range(len(seqlens)):
            q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
            k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        k_ranges[-1] = [0, total_seqlen]

        is_causal_mapping = [True] * len(seqlens)

        return (q_ranges, k_ranges, is_causal_mapping)

    def generate_prefix_lm_causal_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate prefix lm causal mask"""
        seqlen = generate_seqlen_for_one_time(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )

        if seqlen < total_seqlen:
            q_ranges = [[0, total_seqlen], [seqlen, total_seqlen]]
            k_ranges = [[0, seqlen], [seqlen, total_seqlen]]
            is_causal_mapping = [False, True]
        else:
            q_ranges = [[0, total_seqlen]]
            k_ranges = [[0, total_seqlen]]
            is_causal_mapping = [False]

        return (q_ranges, k_ranges, is_causal_mapping)

    def generate_prefix_lm_document_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate prefix lm document mask"""
        seqlens = generate_seqlens(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen,
            rng=rng,
        )
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        is_causal_mapping: list[bool] = []
        for i in range(len(seqlens)):
            full_seqlen = generate_seqlen_for_one_time(
                distribution=seqlen_distribute,
                total_seqlen=seqlens[i],
                rng=rng,
            )
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            if full_seqlen < seqlens[i]:
                q_ranges.append([start, end])
                k_ranges.append([start, start + full_seqlen])
                is_causal_mapping.append(False)

                q_ranges.append([start + full_seqlen, end])
                k_ranges.append([start + full_seqlen, end])
                is_causal_mapping.append(True)
            else:
                q_ranges.append([start, end])
                k_ranges.append([start, end])
                is_causal_mapping.append(False)

        return (q_ranges, k_ranges, is_causal_mapping)

    def generate_qk_sparse_mask(
        self,
        seqlen_distribute: dict[tuple[int, int], int],
        total_seqlen: int,
        rng: random.Random | None = None,
    ) -> tuple[list[list[int]], list[list[int]], list[bool]]:
        """generate qk sparse mask"""
        seqlen = generate_seqlen_for_one_time(
            distribution=seqlen_distribute,
            total_seqlen=total_seqlen // 2,
            rng=rng,
        )

        if seqlen == total_seqlen // 2:
            q_ranges = [[0, total_seqlen]]
            k_ranges = [[0, total_seqlen]]
            is_causal_mapping = [True]
            return (q_ranges, k_ranges, is_causal_mapping)

        full_mask_seqlen = total_seqlen - 2 * seqlen
        full_seqlen = generate_seqlen_for_one_time(
            distribution=seqlen_distribute,
            total_seqlen=full_mask_seqlen,
            rng=rng,
        )

        if full_seqlen == full_mask_seqlen:
            q_ranges = [
                [0, seqlen],
                [seqlen, seqlen + full_seqlen],
                [seqlen + full_seqlen, total_seqlen],
            ]
            k_ranges = [
                [0, seqlen],
                [0, seqlen + full_seqlen],
                [0, total_seqlen],
            ]
            is_causal_mapping = [True, False, True]
        else:
            q_ranges = [
                [0, seqlen],
                [seqlen, total_seqlen],
                [total_seqlen - seqlen - full_seqlen, total_seqlen],
                [total_seqlen - seqlen, total_seqlen],
            ]
            k_ranges = [
                [0, seqlen],
                [0, seqlen],
                [seqlen, seqlen + full_seqlen],
                [seqlen + full_seqlen, total_seqlen],
            ]
            is_causal_mapping = [True, False, False, True]

        return (q_ranges, k_ranges, is_causal_mapping)


class MaskIterator:
    """This is a iterator for multiple flash masks, it can be used with
    several params in init and get an iterator,
    """

    def __init__(
        self,
        generate_times: int,
        generate_mask: FlashMaskType,
        total_seqlen: int,
        distribution: dict[tuple[int, int], float] | None = None,
        to_attn_ranges: bool = True,
        seed: int | None = None,
    ):
        # set params in interator
        self.generate_times = generate_times
        self.current_times = 0
        self.generate_mask = generate_mask
        self.total_seqlen = total_seqlen
        self.to_attn_ranges = to_attn_ranges

        if distribution is not None:
            self.seqlen_distribution = distribution
        elif self.total_seqlen > 128 * 1024:  # 128k
            self.seqlen_distribution = varlen_long_seqlen_distribution()
        else:
            self.seqlen_distribution = varlen_short_seqlen_distribution()

        if seed is not None:
            self.random_number_generator = random.Random(seed)  # type: ignore
        else:
            self.random_number_generator = None  # type: ignore

        self.mask_generator = MaskGenerator()

    def __iter__(self):
        assert (
            self.generate_times > 0
        ), f"generate times must greater than 0, but got {self.generate_times}"
        return self

    def __next__(self):
        if self.current_times >= self.generate_times:
            raise StopIteration
        value = self.mask_generator.generate(
            flash_mask_type=self.generate_mask,
            seqlen_distribute=self.seqlen_distribution,
            total_seqlen=self.total_seqlen,
            to_attn_ranges=self.to_attn_ranges,
            rng=self.random_number_generator,
        )
        self.current_times += 1
        return value
