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
from itertools import chain

from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice


@dataclass
class HostRankEntry:
    """
    HostRankEntry is a dataclass that contains the host q/k ranges and the remote k ranges,
    it is a key data structure for calculating the remote rank entry.

    Args:
        host_q_ranges_global: global q ranges for this rank, merged
        host_k_ranges_global: global k ranges for this rank, merged

        attn_calc_slice_global_list: contains all slices to be calculated on this rank,
            including all slices from both host_stage and remote_stage
        attn_calc_host_slice_local_list: slices that need to be calculated in the host_stage

        remote_k_ranges_global: global remote k ranges for this rank, merged
        remote_k_ranges_global_per_chunk: global remote k ranges for each chunk,
            these are the k ranges needed by the attn slices in this chunk.
            the remote_k_ranges_global for each chunk is merged
        attn_calc_remote_slice_list_per_chunk: contains slices that need to be calculated for each chunk
    """

    host_q_ranges_global: AttnRanges
    host_k_ranges_global: AttnRanges
    attn_calc_slice_global_list: list[AttnSlice]
    attn_calc_host_slice_local_list: list[AttnSlice]

    remote_k_ranges_global: AttnRanges
    # NOTE: We only chunknize remote k_ranges,
    # so attn_calc_remote_slice_list_per_chunk only contains remote k_ranges
    remote_k_ranges_global_per_chunk: list[AttnRanges]
    # NOTE: this is a special attr to support multi-stage overlap
    # each multik_slice of which contains a q_range_local and a k_ranges_global
    # where the k_ranges_global won't be made local until the multi-stage overlap problem solved
    attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]]

    def __post_init__(self):
        assert len(self.remote_k_ranges_global_per_chunk) == len(
            self.attn_calc_remote_slice_list_per_chunk
        ), (
            f"The number of chunks is inconsistent: "
            f"{len(self.remote_k_ranges_global_per_chunk)=}, {len(self.attn_calc_remote_slice_list_per_chunk)=}"
        )

    def get_host_calc_area(self) -> int:
        """Get the host calc area"""
        return sum(
            attn_slice.area for attn_slice in self.attn_calc_host_slice_local_list
        )

    def get_remote_calc_area(self, chunk_idx: int | None = None) -> int:
        """Get the remote calc area (w.r.t. a specific chunk)"""
        if chunk_idx is None:  # return the remote calc area for all chunks
            return sum(
                attn_slice.area
                for attn_slice in chain(*self.attn_calc_remote_slice_list_per_chunk)
            )
        return sum(
            attn_slice.area
            for attn_slice in self.attn_calc_remote_slice_list_per_chunk[chunk_idx]
        )

    def get_remote_comm_size(self, chunk_idx: int | None = None) -> int:
        """Get the remote comm size (w.r.t. a specific chunk)"""
        if chunk_idx is None:  # return the remote comm size for all chunks
            return sum(
                remote_k_ranges.total_seqlen
                for remote_k_ranges in self.remote_k_ranges_global_per_chunk
            )

        return self.remote_k_ranges_global_per_chunk[chunk_idx].total_seqlen


@dataclass
class RemoteRankEntry:
    """
    RemoteRankEntry is a dataclass that contains the remote k ranges and the local k ranges,
    it is a key data structure for calculating the transfer table.

    Args:
        host_k_ranges_global: k_ranges_global owned by the host rank, merged.
        remote_k_ranges_global: k_ranges_global owned by the remote rank, merged.
            Represents the remote kv needed by the host rank in the current overlap stage.

        attn_calc_remote_slice_local_list: Represents the attention calculations that the
            host rank needs to perform in the current overlap stage.
    """

    host_k_ranges_global: AttnRanges
    remote_k_ranges_global: AttnRanges

    attn_calc_remote_slice_local_list: list[AttnSlice]
