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

from typing import List

import torch.nn as nn

from magi_attention.common.enum import AttnMaskType
from magi_attention.common.mask import AttnMask
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.container import AttnBucket, AttnChunk, AttnSlice
from magi_attention.meta.solver.dispatch_solver import DispatchAlg


class GroundTruthDispatcher(nn.Module):
    """Balance dispatching tokens towards each cp rank along the sequence dimension for distributed attention
    where the "balance" has several folds of meaning:
        1. The number of tokens in each cp rank should be exactly balanced, i.e. equal to each other
        2. The computation cost, i.e. the area of the attn_mask matrix, in each cp rank should be roughly balanced
        3. The locality of the dispatched tokens in each cp rank should be maximized, i.e.

    NOTE: this is the ground-truth implementation of the dispatcher,
        which overwrites all of the intrinsic dispatching logics in a naive and inefficient way,
            so as to ONLY be used in testing, instead of production
    """

    def __init__(
        self,
        alg: DispatchAlg,
    ) -> None:
        super().__init__()

        self.alg = alg

        self._self_attn_mask: AttnMask = None  # type: ignore
        self._cross_attn_mask: AttnMask = None  # type: ignore

        self._chunk_masks: list[AttnMask] = []

    def _compute_self_attn_areas(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: List[AttnMaskType],
        chunk_size: int | None = None,
    ) -> AttnBucket:
        """Compute the self-attn areas, with constructing the global bucket,
        which is mainly consists of a list of all the chunks in ascending order, with a length of `cp_size`

        Args:
            q_ranges (AttnRanges): the query ranges
            k_ranges (AttnRanges): the key ranges
            attn_mask_type (List[AttnMaskType]): the attn mask type list
            chunk_size (int | None): the chunk size, which should be divisible by `cp_size`

        Returns:
            AttnBucket: the global bucket
        """

        ts = q_ranges.end
        if chunk_size is None:
            chunk_size = ts
        num_chunks = ts // chunk_size
        one_row_range = AttnRange(start=0, end=ts)

        # build the global mask
        self._self_attn_mask = AttnMask.from_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=q_ranges.end,
            total_seqlen_k=q_ranges.end,  # NOTE: self-attn uses the end of sq
        )

        # build the global bucket
        global_bucket = AttnBucket()
        for chunk_idx in range(num_chunks):  # for each chunk
            chunk_start, chunk_end = (
                chunk_idx * chunk_size,
                (chunk_idx + 1) * chunk_size,
            )
            chunk = AttnChunk(chunk_id=chunk_idx)
            chunk_mask = self._self_attn_mask.make_sub_mask(
                q_range=AttnRange(
                    start=chunk_start,
                    end=chunk_end,
                ),
                k_range=one_row_range,
            )
            self._chunk_masks.append(chunk_mask)
            for slice_idx, (q_range, k_range, mask_type) in enumerate(
                chunk_mask.tuples()
            ):
                slice = AttnSlice(
                    slice_id=slice_idx,
                    q_range=q_range.offset(chunk_start),
                    k_range=k_range,
                    mask_type=mask_type,
                )

                # HACK: Later the area calculation logic will be encapsulated in AttnSlice and area will be read-only,
                #       here we keep the functionality to set area directly
                slice.area = chunk_mask.calc_sub_area(
                    q_range=q_range,
                    k_range=k_range,
                )

                chunk.q_slices.append(slice)

            global_bucket.q_chunks.append(chunk)

        return global_bucket
