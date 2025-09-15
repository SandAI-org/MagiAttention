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

from magi_attention.common.enum import AttnRole, AttnType
from magi_attention.common.ranges import AttnRanges


@dataclass
class DispatchMeta:
    """The meta info of sequence dispatch for distributed attention

    Args:
        attn_role (AttnRole): The role of the attention inputs.
        attn_type (AttnType): The type of the attention pattern.
        total_seqlen (int): The total sequence length.
        shard_seqlen (int): The sharded sequence length for each rank.

            NOTE: for now, we handle padding logics outside of dispatch,
            so the sharded sequence length is the same for all cp ranks.
        max_valid_ids (int): The maximum valid token id.

            NOTE: this can be used to indicate the valid position ids
            to get rid of unused tokens (such as padding tokens)
        chunk_size (int): The chunk size to control the granularity of computation load-balance.
        num_chunks (int): The number of chunks.
        cp_rank (int): The cp rank of current process in the context-parallel group.
        cp_size (int): The world size of the context-parallel group.
        partitions[i][j]: the jth chunk idx for ranki
        partitions_perm_idxs[i]: the idx for chunki to permute, used in dispatch func
        partitions_unperm_idxs[j]: the idx for chunkj to unpermute, used in undispatch func
    """

    attn_role: AttnRole
    attn_type: AttnType

    total_seqlen: int
    shard_seqlen: int
    max_valid_ids: int

    chunk_size: int
    num_chunks: int

    cp_rank: int
    cp_size: int

    partitions: list[list[int]]
    partitions_perm_idxs: list[int]
    partitions_unperm_idxs: list[int]

    @property
    def host_ranges_per_rank(self) -> list[AttnRanges]:
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
        # sanity check
        assert len(self.partitions) == self.cp_size
        assert (
            len(self.partitions_perm_idxs)
            == len(self.partitions_unperm_idxs)
            == self.num_chunks
        )

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
