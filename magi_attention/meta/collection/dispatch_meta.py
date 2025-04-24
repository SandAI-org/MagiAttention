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

from dataclasses import dataclass

import torch

from magi_attention.common.enum import AttnMaskType, AttnRole, AttnType
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.container.bucket import AttnBucket


@dataclass
class DispatchMeta:
    """The meta info of sequence dispatch for distributed attention

    Args:
        TODO: finish docstring

        max_valid_ids(int): the maximum valid token ids in the seqlen dim.

    NOTE: global_bucket
    """

    attn_role: AttnRole
    attn_type: AttnType
    attn_mask_type: list[AttnMaskType]

    ranges: AttnRanges

    batch_size: int
    total_seqlen: int
    shard_seqlen: int
    max_valid_ids: int

    chunk_size: int
    num_chunks: int

    cp_rank: int
    cp_size: int

    seqlens: list[int]
    seqlens_permed: list[int]  # used but not enabled property
    seqlens_perm_idxs: list[int]  # used but not enabled property
    seqlens_unperm_idxs: list[int]  # used but not enabled property

    cu_seqlens: list[int]  # unused property
    cu_seqlens_permed: list[int]  # unused property

    # dispatch solver results
    partitions: list[list[int]]
    partitions_perm_idxs: list[int]
    partitions_unperm_idxs: list[int]

    global_bucket: AttnBucket
    buckets_per_rank: list[AttnBucket]

    high_bandwith_domain_size: int

    @property
    def host_ranges_per_rank(self) -> list[AttnRanges]:
        # NOTE: since we discard the q_ranges which are belonging to
        # certain empty slices due to causal mask, we can NOT recover
        # the host ranges from the buckets, instead, we can just
        # construct the host ranges using chunk ranges,
        # since they will be used ONLY in merged form anyway

        # return [bucket.q_ranges for bucket in self.buckets_per_rank]
        return [
            AttnRanges.from_ranges(
                [
                    [chunk_id * self.chunk_size, (chunk_id + 1) * self.chunk_size]
                    for chunk_id in partition
                ]
            )
            for partition in self.partitions
        ]

    @property
    def host_ranges_this_domain(self) -> AttnRanges:
        attn_ranges = AttnRanges()
        domain_rank = self.cp_rank // self.high_bandwith_domain_size
        for host_ranges_ith_rank in self.host_ranges_per_rank[
            self.high_bandwith_domain_size
            * domain_rank : self.high_bandwith_domain_size
            * (domain_rank + 1)
        ]:
            attn_ranges.extend(host_ranges_ith_rank)

        return attn_ranges

    @property
    def position_ids(self) -> torch.Tensor:
        chunk_size = self.chunk_size
        local_partition = self.partitions[self.cp_rank]

        position_ids = torch.tensor(
            [
                i
                for n in local_partition
                for i in range(n * chunk_size, (n + 1) * chunk_size)
            ],
            device=torch.cuda.current_device(),
        )

        position_ids = position_ids.clamp(max=self.max_valid_ids - 1)

        return position_ids

    def __post_init__(self) -> None:
        assert len(self.seqlens) == len(self.seqlens_permed) == self.batch_size
        assert (
            len(self.seqlens_perm_idxs)
            == len(self.seqlens_unperm_idxs)
            == self.batch_size
        )
        assert (
            len(self.cu_seqlens) == len(self.cu_seqlens_permed) == self.batch_size + 1
        )

        assert len(self.partitions) == self.cp_size
        assert (
            len(self.partitions_perm_idxs)
            == len(self.partitions_unperm_idxs)
            == self.num_chunks
        )
        assert len(self.buckets_per_rank) == self.cp_size
        assert len(self.host_ranges_per_rank) == self.cp_size

    def __repr__(self, width: int = 30) -> str:
        """Customized __repr__ method for BaseConfig,
        displaying all fields with their values in alphabetical order.
        """
        class_name = self.__class__.__name__
        repr_str = f"{'*' * width}   {class_name}   {'*' * width}\n"
        title_len = len(repr_str) - 1

        field_names = sorted(self.__dataclass_fields__.keys())
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, str):
                field_value = repr(field_value)
            repr_str += f"{field_name}: {field_value}\n"

        repr_str += f"{'*' * title_len}\n"

        return repr_str
