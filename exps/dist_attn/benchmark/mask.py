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
from dataclasses import dataclass

from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.utils import DatasetSampler, seqlens2cu_seqlens
from magi_attention.api.functools import infer_attn_mask_from_sliding_window
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges


@dataclass
class MaskFactors:
    cu_seqlens: list[int] | None = None
    cu_ranges: list[list[int]] | None = None
    prefix_length: int | None = None
    window_size: int | None = None
    block_size: int | None = None


DOCUMENT_LIKE_MASKS = [
    FlashMaskType.FULL_DOCUMENT,
    FlashMaskType.CAUSAL_DOCUMENT,
    FlashMaskType.SHARE_QUESTION,
    FlashMaskType.CAUSAL_BLOCKWISE,
    FlashMaskType.PREFIX_LM_DOCUMENT,
    FlashMaskType.BLOCK_CAUSAL_DOCUMENT,
]


class MaskIterator:
    def __init__(
        self,
        num_iterations: int,
        mask_type: FlashMaskType,
        total_seqlen: int,
        data_path: str | None = None,
        chunk_ratio: float = 0.25,
        is_binned: bool = True,
        window_size: tuple[int, int] | None = None,
        to_attn_ranges: bool = True,
        seed: int = 42,
        drop_thres: int = -1,
        **kwargs,
    ):
        """
        This is a iterator to generate
        """
        assert num_iterations > 0, f"Invalid iteration number, got {num_iterations}."
        assert total_seqlen > 0, f"Invalid total sequence length, got {total_seqlen}."
        self.num_iterations = num_iterations
        self.mask_type = mask_type
        self.total_seqlen = total_seqlen
        self.to_attn_ranges = to_attn_ranges
        self.window_size = window_size
        self.data_path = data_path
        self.cur_iter = 0
        self.drop_thres = drop_thres
        self.prefix_length = kwargs.get("prefix_length", None)
        self.block_size = kwargs.get("block_size", None)

        self.random_number_generator = random.Random(seed)  # type: ignore
        if mask_type in DOCUMENT_LIKE_MASKS and data_path is not None:
            self.sampler = DatasetSampler(
                data_path=data_path,
                pack_len=total_seqlen,
                chunk_ratio=chunk_ratio,
                is_binned=is_binned,
                seed=seed,
                drop_thres=drop_thres,
            )
        else:
            self.sampler = None  # type: ignore[assignment]
        self.gen_func = {
            FlashMaskType.FULL: self.generate_full_mask,
            FlashMaskType.CAUSAL: self.generate_causal_mask,
            FlashMaskType.CAUSAL_DOCUMENT: self.generate_causal_document_mask,
            FlashMaskType.FULL_DOCUMENT: self.generate_full_document_mask,
            FlashMaskType.SHARE_QUESTION: self.generate_share_question_mask,
            FlashMaskType.CAUSAL_BLOCKWISE: self.generate_causal_blockwise_mask,
            FlashMaskType.PREFIX_LM_CAUSAL: self.generate_prefix_lm_causal_mask,
            FlashMaskType.PREFIX_LM_DOCUMENT: self.generate_prefix_lm_document_mask,
            FlashMaskType.SLIDING_WINDOW: self.generate_sliding_window_mask,
            FlashMaskType.SLIDING_WINDOW_CAUSAL: self.generate_sliding_window_causal_mask,
            FlashMaskType.GLOBAL_SLIDING_WINDOW: self.generate_global_sliding_window_mask,
            FlashMaskType.BLOCK_CAUSAL_DOCUMENT: self.generate_block_causal_document_mask,
        }

    def generate(self):
        q_ranges, k_ranges, attn_mask_type, mask_factors = self.gen_func[
            self.mask_type
        ]()
        if self.to_attn_ranges:
            q_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=q_ranges)
            k_ranges_: AttnRanges = AttnRanges.from_ranges(ranges=k_ranges)
            attn_mask_type_: list[AttnMaskType] = [
                [
                    AttnMaskType.FULL,
                    AttnMaskType.CAUSAL,
                    AttnMaskType.INVCAUSAL,
                    AttnMaskType.BICAUSAL,
                ][mask_idx]
                for mask_idx in attn_mask_type
            ]
            return (q_ranges_, k_ranges_, attn_mask_type_, mask_factors)

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def reset(self, re_shuffle=False):
        self.cur_iter = 0
        if self.sampler:
            self.sampler.reset(re_shuffle=re_shuffle)

    def __iter__(self):
        assert (
            self.num_iterations > 0
        ), f"iteration times must greater than 0, but got {self.num_iterations}"
        return self

    def __next__(self):
        if self.cur_iter >= self.num_iterations:
            raise StopIteration
        mask_meta = self.generate()
        self.cur_iter += 1
        return mask_meta

    def generate_full_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate FULL mask"""
        ranges = [[0, self.total_seqlen]]
        attn_mask_type = [0]
        return (ranges, ranges, attn_mask_type, MaskFactors())

    def generate_causal_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate CAUSAL mask"""
        ranges = [[0, self.total_seqlen]]
        attn_mask_type = [1]
        return (ranges, ranges, attn_mask_type, MaskFactors())

    def generate_full_document_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate FULL DOCUMENT maks (varlen full)"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        attn_mask_type = [0] * len(seqlens)

        return (ranges, ranges, attn_mask_type, MaskFactors(cu_ranges=ranges))

    def generate_causal_document_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate CAUSAL DOCUMENT mask (varlen causal)"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        ranges = []
        for i in range(len(seqlens)):
            ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        attn_mask_type = [1] * len(seqlens)

        return (ranges, ranges, attn_mask_type, MaskFactors(cu_ranges=ranges))

    def generate_share_question_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate SHARE QUESTION mask"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_mask_type: list[int] = []
        for i in range(len(seqlens)):
            if i == 1:
                q_ranges[0] = [0, cu_seqlens[i + 1]]
                k_ranges[0] = [0, cu_seqlens[i + 1]]

                if len(seqlens) > 2:
                    q_ranges.append([cu_seqlens[i + 1], self.total_seqlen])
                    k_ranges.append([0, cu_seqlens[i]])
                    attn_mask_type.append(0)
            else:
                q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
                attn_mask_type.append(1)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
        )

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def generate_causal_blockwise_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate CAUSAL BLOCKWISE mask"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        for i in range(len(seqlens)):
            q_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
            k_ranges.append([cu_seqlens[i], cu_seqlens[i + 1]])
        k_ranges[-1] = [0, self.total_seqlen]
        attn_mask_type = [1] * len(seqlens)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
        )

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def generate_prefix_lm_causal_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate PREFIX LM CAUSAL mask"""
        seqlen = self.random_number_generator.randint(1, self.total_seqlen)
        if seqlen < self.total_seqlen:
            q_ranges = [[0, seqlen], [seqlen, self.total_seqlen]]
            k_ranges = [[0, seqlen], [0, self.total_seqlen]]
            attn_mask_type = [0, 1]
        else:
            q_ranges = [[0, self.total_seqlen]]
            k_ranges = [[0, self.total_seqlen]]
            attn_mask_type = [0]

        mask_factors = MaskFactors()
        mask_factors.prefix_length = (
            self.prefix_length if self.prefix_length else seqlen
        )

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def generate_prefix_lm_document_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate PREFIX LM DOCUMENT mask (prefix lm varlen)"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        min_seqlen = min(seqlens)
        assert min_seqlen > 0
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]
        prefix = (
            self.prefix_length
            if self.prefix_length
            else self.random_number_generator.randint(1, min_seqlen)
        )

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_mask_type: list[int] = []
        for i in range(len(seqlens)):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            if prefix < seqlens[i]:
                q_ranges.append([start, start + prefix])
                k_ranges.append([start, start + prefix])
                attn_mask_type.append(0)

                q_ranges.append([start + prefix, end])
                k_ranges.append([start, end])
                attn_mask_type.append(1)
            else:
                q_ranges.append([start, end])
                k_ranges.append([start, end])
                attn_mask_type.append(0)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
            prefix_length=prefix,
        )

        return (q_ranges, k_ranges, attn_mask_type, mask_factors)

    def generate_sliding_window_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate SLIDING WINDOW FULL mask"""
        assert self.window_size is not None and len(self.window_size) == 2
        mask_factors = MaskFactors(window_size=self.window_size[0])
        if self.window_size[0] + 1 >= self.total_seqlen:
            return self.generate_full_mask()

        q_ranges, k_ranges, attn_mask_type = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(start=0, end=self.total_seqlen),
            k_range=AttnRange(start=0, end=self.total_seqlen),
            window_size=self.window_size,  # type: ignore
        )
        attn_type_map = [
            {
                AttnMaskType.FULL: 0,
                AttnMaskType.CAUSAL: 1,
                AttnMaskType.INVCAUSAL: 2,
                AttnMaskType.BICAUSAL: 3,
            }[mask_type]
            for mask_type in attn_mask_type
        ]

        return (
            q_ranges.to_naive_ranges(),  # type: ignore
            k_ranges.to_naive_ranges(),
            attn_type_map,
            mask_factors,
        )

    def generate_sliding_window_causal_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate SLIDING WINDOW CAUSAL mask"""
        assert self.window_size is not None and len(self.window_size) == 2
        self.window_size = (self.window_size[0], 0)
        if self.window_size[0] + 1 >= self.total_seqlen:
            return self.generate_causal_mask()

        return self.generate_sliding_window_mask()

    def generate_global_sliding_window_mask(self):
        """generate GLOBAL SLIDING WINDOW mask"""
        assert self.window_size is not None and len(self.window_size) == 2
        if self.window_size[0] + 1 >= self.total_seqlen:
            return self.generate_full_mask()
        window_size_single = self.window_size[0]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_type_map: list[int] = []

        q_ranges.append([0, self.total_seqlen])
        k_ranges.append([0, window_size_single])
        attn_type_map.append(1)

        q_ranges.append([0, window_size_single])
        k_ranges.append([window_size_single, self.total_seqlen])
        attn_type_map.append(1)

        (
            sw_q_ranges,
            sw_k_ranges,
            sw_attn_mask_type,
        ) = infer_attn_mask_from_sliding_window(
            q_range=AttnRange(start=window_size_single, end=self.total_seqlen),
            k_range=AttnRange(start=window_size_single, end=self.total_seqlen),
            window_size=[window_size_single, window_size_single],
        )

        sw_attn_type_map = [
            {
                AttnMaskType.FULL: 0,
                AttnMaskType.CAUSAL: 1,
                AttnMaskType.INVCAUSAL: 2,
                AttnMaskType.BICAUSAL: 3,
            }[mask_type]
            for mask_type in sw_attn_mask_type
        ]

        q_ranges.extend(sw_q_ranges.to_naive_ranges())  # type: ignore
        k_ranges.extend(sw_k_ranges.to_naive_ranges())  # type: ignore
        attn_type_map.extend(sw_attn_type_map)

        mask_factors = MaskFactors(window_size=window_size_single)

        return (q_ranges, k_ranges, attn_type_map, mask_factors)

    def generate_block_causal_document_mask(
        self,
    ) -> tuple[list[list[int]], list[list[int]], list[int], MaskFactors]:
        """generate BLOCK CAUSAL DOCUMENT mask (block causal varlen)"""
        assert (
            self.data_path is not None
        ), "iterator needs dataset path to init DatasetSampler."
        seqlens = self.sampler.generate_pack_samples()
        cu_seqlens = seqlens2cu_seqlens(seqlens)
        cu_ranges = [
            [cu_seqlens[i], cu_seqlens[i + 1]] for i in range(len(cu_seqlens) - 1)
        ]

        q_ranges: list[list[int]] = []
        k_ranges: list[list[int]] = []
        attn_type_map: list[int] = []
        for seqlen, start_offset in zip(seqlens, cu_seqlens[:-1]):
            num_blocks = (seqlen + self.block_size - 1) // self.block_size
            for i in range(num_blocks):
                start = i * self.block_size
                end = min((i + 1) * self.block_size, seqlen)
                q_ranges.append([start + start_offset, end + start_offset])
                k_ranges.append([start_offset, end + start_offset])
                attn_type_map.append(0)

        mask_factors = MaskFactors(
            cu_seqlens=cu_seqlens,
            cu_ranges=cu_ranges,
            block_size=self.block_size,
        )

        return q_ranges, k_ranges, attn_type_map, mask_factors
