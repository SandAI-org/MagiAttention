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

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from itertools import chain
from typing import Any, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.comm.primitive.grpcoll.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
)
from magi_attention.common import is_cpp_backend_enable
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta, GroupCollectiveArg
from magi_attention.meta.collection.dispatch_meta import DispatchMeta
from magi_attention.meta.container.bucket import AttnBucket
from magi_attention.meta.container.chunk import AttnChunk
from magi_attention.meta.container.rank_entry import HostRankEntry, RemoteRankEntry
from magi_attention.meta.container.slice import AttnSlice, MultiKAttnSlice
from magi_attention.meta.container.transfer_table import (
    GroupCastRanges,
    TransferInfo,
    TransferTable,
)
from magi_attention.utils import (
    is_same_device_mesh,
    is_same_process_group,
    nvtx,
    transpose_matrix,
)
from magi_attention.utils._utils import (
    argsort,
    flatten_nested_list,
    perm_idxs2unperm_idxs,
)

from .._make_dispatch_meta import make_bucket_per_rank_from_qk_ranges
from .overlap_solver import OverlapConfig, OverlapSolver, OverlapStageCost
from .slice_maker import HostAttnSliceMaker, RemoteAttnSliceMaker

USE_CPP_EXT = False
if is_cpp_backend_enable():
    try:
        from magi_attention import magi_attn_ext

        USE_CPP_EXT = True
    except ImportError:
        pass


class BaseDistAttnSolver(ABC):
    """The base abstract dist-attn solver class to
    provide necessary abstract methods as common interfaces for sub-classes to implement
    """

    @abstractmethod
    def solve(self, *args, **kwargs) -> None:
        ...

    @abstractmethod
    def make_comm_meta(self) -> CommMeta:
        ...

    @abstractmethod
    def make_calc_meta(self) -> CalcMeta:
        ...

    @property
    @abstractmethod
    def is_solved(self) -> bool:
        ...

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseDistAttnSolver):
            return False

        for key in self.__dict__:
            self_val = getattr(self, key)
            other_val = getattr(other, key, object())

            if isinstance(self_val, dist.ProcessGroup) or isinstance(
                other_val, dist.ProcessGroup
            ):
                if not is_same_process_group(self_val, other_val):
                    return False

            elif isinstance(self_val, DeviceMesh) or isinstance(other_val, DeviceMesh):
                if not is_same_device_mesh(self_val, other_val):
                    return False

            else:
                if self_val != other_val:
                    return False

        return True


class DistAttnSolver(BaseDistAttnSolver):
    """The dist-attn solver class to process dispatch meta for calc/comm meta"""

    @nvtx.instrument_nvtx
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        overlap_config: OverlapConfig,
        cp_mesh: DeviceMesh | None = None,
        num_heads_q: int = 1,
        num_heads_kv: int = 1,
    ):
        assert (
            not magi_attention.comm.is_qo_comm_enable()
        ), "QO comm is not supported for this dist-attn solver"

        self.cp_rank = dist.get_rank(cp_group)
        self.cp_size = dist.get_world_size(cp_group)
        self.cp_group = cp_group
        self.cp_mesh = cp_mesh
        self.deterministic = magi_attention.is_deterministic_mode_enable()
        self.overlap_config = overlap_config
        self.overlap_solver = OverlapSolver(alg=self.overlap_config.alg)

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.num_heads_group = 1

        # NOTE: the real overlap degree should be determined in the later code:
        # 1. if overlap mode is static, then its real value equals to the one in the overlap config
        # 2. if overlap mode is dynamic, then its real value is determined by overlap solver
        self.overlap_degree: int = -1

        # NOTE: the real overlap chunk size and the number of chunks should be determined in the later code:
        # 1. if the remote length is not too long (num_chunks <= max_num_chunks),
        #   then use the 'min_chunk_size' to chunk it, i.e. overlap_chunk_size = min_chunk_size
        # 2. otherwise, use the 'max_num_chunks' to calc a larger chunk size
        #   and assign it as overlap_chunk_size, i.e. overlap_num_chunks = max_num_chunks
        self.overlap_chunk_size: int = -1
        self.overlap_num_chunks: int = -1

        self._is_solved = False

    def _expand_attn_ranges(self, ranges: AttnRanges, stride: int) -> AttnRanges:
        if USE_CPP_EXT:
            return magi_attn_ext.expand_attn_ranges(
                ranges, stride, self.num_heads_group  # type: ignore[arg-type, return-value]
            )

        new_ranges = AttnRanges()
        for i in range(self.num_heads_group):
            offset = i * stride
            for r in ranges:
                new_ranges.append(AttnRange(r.start + offset, r.end + offset))
        return new_ranges

    def _expand_dispatch_meta(self, dispatch_meta: DispatchMeta) -> DispatchMeta:
        num_heads_group = self.num_heads_group
        orig_num_chunks = dispatch_meta.num_chunks

        new_total_seqlen = dispatch_meta.total_seqlen * num_heads_group
        new_shard_seqlen = dispatch_meta.shard_seqlen * num_heads_group
        new_num_chunks = dispatch_meta.num_chunks * num_heads_group

        new_partitions = []
        for partition in dispatch_meta.partitions:
            new_partition = []
            for i in range(num_heads_group):
                offset = i * orig_num_chunks
                new_partition.extend([c + offset for c in partition])
            new_partitions.append(new_partition)

        new_partitions_perm_idxs = flatten_nested_list(new_partitions)
        new_partitions_unperm_idxs = perm_idxs2unperm_idxs(new_partitions_perm_idxs)

        return replace(
            dispatch_meta,
            total_seqlen=new_total_seqlen,
            shard_seqlen=new_shard_seqlen,
            num_chunks=new_num_chunks,
            partitions=new_partitions,
            partitions_perm_idxs=new_partitions_perm_idxs,
            partitions_unperm_idxs=new_partitions_unperm_idxs,
        )

    @nvtx.instrument_nvtx
    def solve(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: Union[list[int], list[AttnMaskType], AttnMaskType, int],
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
    ) -> None:
        flatten_head_groups = magi_attention.is_flatten_head_groups_enable()
        if flatten_head_groups:
            self.num_heads_group = self.num_heads_kv
            self.num_heads_q = self.num_heads_q // self.num_heads_group
            self.num_heads_kv = 1

            # Expand seqlen and ranges for local q/k
            q_ranges = self._expand_attn_ranges(q_ranges, dispatch_meta_q.total_seqlen)
            k_ranges = self._expand_attn_ranges(k_ranges, dispatch_meta_k.total_seqlen)

            # Expand dispatch meta
            dispatch_meta_q = self._expand_dispatch_meta(dispatch_meta_q)
            dispatch_meta_k = self._expand_dispatch_meta(dispatch_meta_k)

        # normalize attn_mask_type to list[AttnMaskType]
        if isinstance(attn_mask_type, (AttnMaskType, int)):
            # HACK: for one mask type, wrap to list
            if isinstance(attn_mask_type, int):
                attn_mask_type = AttnMaskType.from_int_type(attn_mask_type)
            attn_mask_type_list = [attn_mask_type] * len(q_ranges)
        elif isinstance(attn_mask_type, list):
            if len(attn_mask_type) > 0 and isinstance(attn_mask_type[0], int):
                attn_mask_type = [
                    AttnMaskType.from_int_type(m) if isinstance(m, int) else m
                    for m in attn_mask_type
                ]
            if flatten_head_groups:
                attn_mask_type_list = attn_mask_type * self.num_heads_group  # type: ignore[assignment]
            else:
                attn_mask_type_list = attn_mask_type  # type: ignore[assignment]
        else:
            raise TypeError(f"Unsupported attn_mask_type type: {type(attn_mask_type)}")

        # init bucket this rank from dispatch_meta_q
        # assuming it is self-attn scenarios and the partitions of q,k are the same
        if magi_attention.is_sanity_check_enable():
            assert dispatch_meta_q.partitions == dispatch_meta_k.partitions
        bucket_this_rank = self._make_bucket_this_rank(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type_list,
            dispatch_meta=dispatch_meta_q,
        )

        # init host / remote q/k ranges global for this rank
        (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
        ) = self._init_host_remote_ranges_global_this_rank(
            dispatch_meta_q=dispatch_meta_q,
            dispatch_meta_k=dispatch_meta_k,
            bucket_this_rank=bucket_this_rank,
        )

        # set some attributes that might be fetched from outside
        self.bucket = bucket_this_rank
        self.host_q_ranges_global = host_q_ranges_global_this_rank
        self.host_k_ranges_global = host_k_ranges_global_this_rank
        self.remote_k_ranges_global = remote_k_ranges_global_this_rank
        self.total_seqlen_q = dispatch_meta_q.total_seqlen
        self.shard_seqlen_q = dispatch_meta_q.shard_seqlen
        self.total_seqlen_k = dispatch_meta_k.total_seqlen

        # init host rank entry for this rank
        self.host_rank_entry_this_rank = self._init_host_rank_entry_this_rank(
            host_q_ranges_global=host_q_ranges_global_this_rank,
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global=remote_k_ranges_global_this_rank,
            attn_calc_slice_global_list=bucket_this_rank.attn_slices,
        )

        # init remote rank entry for each stage for this rank
        self.remote_rank_entry_per_stage_this_rank = (
            self._init_remote_rank_entry_per_stage_this_rank(
                self.host_rank_entry_this_rank
            )
        )

        # init remote rank entry for each rank for each stage
        self.remote_rank_entry_per_rank_per_stage = (
            self._init_remote_rank_entry_per_rank_per_stage(
                self.remote_rank_entry_per_stage_this_rank
            )
        )

        # init transfer table per stage
        self.transfer_table_per_stage: list[
            TransferTable
        ] = self._init_transfer_table_per_stage(
            self.remote_rank_entry_per_rank_per_stage,
        )

        self._is_solved = True

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    def _make_bucket_this_rank(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
        dispatch_meta: DispatchMeta,
    ) -> AttnBucket:
        bucket_per_rank = make_bucket_per_rank_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            dispatch_meta=dispatch_meta,
        )
        bucket_this_rank = bucket_per_rank[self.cp_rank]

        return bucket_this_rank

    @nvtx.instrument_nvtx
    def _init_host_remote_ranges_global_this_rank(
        self,
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        bucket_this_rank: AttnBucket,
    ) -> tuple[AttnRanges, AttnRanges, AttnRanges]:
        
        if self.cp_size == 1: # cp1 shortcut
            host_q_ranges_global_this_rank = AttnRanges.from_ranges([[0, dispatch_meta_q.total_seqlen]])
            host_k_ranges_global_this_rank = AttnRanges.from_ranges([[0, dispatch_meta_k.total_seqlen]])
            remote_k_ranges_global_this_rank = AttnRanges.from_ranges([[0, 0]])
        else:
            # init host q_ranges global for this rank
            host_q_ranges_global_this_rank = dispatch_meta_q.host_ranges_per_rank[
                self.cp_rank
            ].merge()

            # init host k_ranges global for this rank
            host_k_ranges_global_this_rank = dispatch_meta_k.host_ranges_per_rank[
                self.cp_rank
            ].merge()

            # init remote k_ranges global for this rank
            # NOTE: this only contains the remote k ranges that we need to calculate from
            remote_k_ranges_global_this_rank = bucket_this_rank.k_ranges.find_hole_ranges(
                host_k_ranges_global_this_rank,
                is_other_merged=True,
            )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            # check if merged successfully
            assert host_q_ranges_global_this_rank.is_merged()
            assert host_k_ranges_global_this_rank.is_merged()
            assert remote_k_ranges_global_this_rank.is_merged()

            # whether q_ranges and k_ranges are one-one mapping in attn calc
            attn_calc_q_ranges_global = bucket_this_rank.q_ranges
            attn_calc_k_ranges_global = bucket_this_rank.k_ranges
            assert len(attn_calc_q_ranges_global) == len(attn_calc_k_ranges_global), (
                f"The {len(attn_calc_q_ranges_global)=} should be equal to "
                f"{len(attn_calc_k_ranges_global)=}."
            )

        return (
            host_q_ranges_global_this_rank,
            host_k_ranges_global_this_rank,
            remote_k_ranges_global_this_rank,
        )

    @nvtx.instrument_nvtx
    def _init_host_rank_entry_this_rank(
        self,
        host_q_ranges_global: AttnRanges,
        host_k_ranges_global: AttnRanges,
        remote_k_ranges_global: AttnRanges,
        attn_calc_slice_global_list: list[AttnSlice],
    ) -> HostRankEntry:
        """Initialize host rank entry for this rank"""

        # -------   chunk remote k ranges global  ------ #

        remote_k_ranges_global_per_chunk = self._chunk_remote_k_ranges_global(
            # NOTE: we only chunk the remote k ranges
            remote_k_ranges_global=remote_k_ranges_global,
        )

        # -------   calc attn calc host q ranges local  ------ #

        attn_calc_global_chunk = AttnChunk(q_slices=attn_calc_slice_global_list)
        attn_calc_host_q_ranges_local = host_q_ranges_global.make_ranges_local(
            attn_calc_global_chunk.q_ranges,
            is_self_merged=True,
        )

        # -------   make attn calc host/remote slices  ------ #

        attn_calc_host_slice_local_list: list[AttnSlice] = []
        attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]] = [
            [] for _ in range(len(remote_k_ranges_global_per_chunk))
        ]

        # sort attn_calc_slice_global_list with k_range of slice
        global_slice_indices = argsort(
            attn_calc_slice_global_list, key=lambda x: (x.k_range.start, x.k_range.end)
        )
        attn_calc_slice_global_list = [
            attn_calc_slice_global_list[i] for i in global_slice_indices
        ]
        attn_calc_host_q_ranges_local = [
            attn_calc_host_q_ranges_local[i] for i in global_slice_indices
        ]

        # get host_k_ranges start and end, init chunked_remote_k_ranges idx for iteration
        self.chunk_remote_k_ranges_global_start_idx = 0
        host_k_ranges_global_start: int = host_k_ranges_global.start
        host_k_ranges_global_end: int = host_k_ranges_global.end

        # get start and end for each chunked remote_k_ranges_global
        chunk_remote_k_ranges_global_boundary = [
            (attn_range.start, attn_range.end)
            for attn_range in remote_k_ranges_global_per_chunk
        ]

        for ith_attn_slice_global, ith_attn_calc_host_q_range_local in zip(
            attn_calc_slice_global_list,
            attn_calc_host_q_ranges_local,
        ):
            ith_attn_slice_global_mask_type: AttnMaskType = ith_attn_slice_global.mask_type  # type: ignore
            # HACK: wrap k range to k ranges,
            # to use the API of AttnRanges like find_overlap_ranges
            ith_attn_calc_k_ranges_global = AttnRanges()
            ith_attn_calc_k_ranges_global.append(ith_attn_slice_global.k_range)  # type: ignore

            # -------   make ith attn calc host slice local  ------ #

            # skip directly when there is no overlap
            if (
                host_k_ranges_global_start < ith_attn_slice_global.k_range.end  # type: ignore
                and host_k_ranges_global_end > ith_attn_slice_global.k_range.start  # type: ignore
            ):
                self._make_ith_attn_calc_host_slice(
                    host_k_ranges_global=host_k_ranges_global,
                    attn_calc_host_slice_local_list=attn_calc_host_slice_local_list,
                    ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                    ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                    ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
                )

            # -------   make ith attn calc remote slice per chunk  ------ #

            self._make_ith_attn_calc_remote_slice_per_chunk(
                remote_k_ranges_global_per_chunk=remote_k_ranges_global_per_chunk,
                attn_calc_remote_slice_list_per_chunk=attn_calc_remote_slice_list_per_chunk,
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
                chunk_remote_k_ranges_global_boundary=chunk_remote_k_ranges_global_boundary,
            )

        # delete chunked_remote_k_ranges idx
        del self.chunk_remote_k_ranges_global_start_idx

        host_rank_entry_this_rank = HostRankEntry(
            host_q_ranges_global=host_q_ranges_global,
            host_k_ranges_global=host_k_ranges_global,
            attn_calc_slice_global_list=attn_calc_slice_global_list,
            attn_calc_host_slice_local_list=attn_calc_host_slice_local_list,
            remote_k_ranges_global=remote_k_ranges_global,
            remote_k_ranges_global_per_chunk=remote_k_ranges_global_per_chunk,
            attn_calc_remote_slice_list_per_chunk=attn_calc_remote_slice_list_per_chunk,
        )

        return host_rank_entry_this_rank

    @nvtx.instrument_nvtx
    def _chunk_remote_k_ranges_global(
        self,
        remote_k_ranges_global: AttnRanges,
    ) -> list[AttnRanges]:
        """Chunk remote k ranges global for multi-stage overlap
        called in 'self._init_host_rank_entry_this_rank'
        """

        # determine the chunk size constrainted by min_chunk_size and max_num_chunks
        total_remote_k_seqlen = remote_k_ranges_global.total_seqlen
        num_chunks = (
            total_remote_k_seqlen + self.overlap_config.min_chunk_size - 1
        ) // self.overlap_config.min_chunk_size
        if num_chunks <= self.overlap_config.max_num_chunks:
            self.overlap_chunk_size = self.overlap_config.min_chunk_size
            self.overlap_num_chunks = num_chunks
        else:
            self.overlap_num_chunks = self.overlap_config.max_num_chunks
            self.overlap_chunk_size = (
                total_remote_k_seqlen + self.overlap_num_chunks - 1
            ) // self.overlap_num_chunks
            self.overlap_num_chunks = (
                total_remote_k_seqlen + self.overlap_chunk_size - 1
            ) // self.overlap_chunk_size

        # chunk the remote k ranges global for multi-stage overlapping
        remote_k_ranges_global_per_chunk: list[
            AttnRanges
        ] = remote_k_ranges_global.chunk(
            self.overlap_chunk_size, check=magi_attention.is_sanity_check_enable()
        )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert all(
                remote_k_ranges_global_ith_chunk.is_merged()
                for remote_k_ranges_global_ith_chunk in remote_k_ranges_global_per_chunk
            ), (
                f"Every remote_k_ranges_global for each chunk should be merged, "
                f"but {remote_k_ranges_global_per_chunk=}."
            )
            assert len(remote_k_ranges_global_per_chunk) == self.overlap_num_chunks, (
                f"{len(remote_k_ranges_global_per_chunk)=} should be equal "
                f"to {self.overlap_num_chunks=} with {self.overlap_chunk_size=}"
            )

        return remote_k_ranges_global_per_chunk

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_host_slice(
        self,
        host_k_ranges_global: AttnRanges,
        attn_calc_host_slice_local_list: list[AttnSlice],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
    ) -> None:
        """Make attn calc host slice local, appended to 'attn_calc_host_slice_local_list'
        called in 'self._init_host_rank_entry_this_rank'
        """

        # fine the overlap part with the global host k ranges
        # i.e. the global attn calc host k ranges
        ith_attn_calc_host_k_ranges_global = (
            ith_attn_calc_k_ranges_global.find_overlap_ranges(
                host_k_ranges_global,
                is_self_merged=True,
                is_other_merged=True,
            )
        )
        # no overlap ranges on host, nothing to do
        if len(ith_attn_calc_host_k_ranges_global) == 0:
            return
        # otherwise, make it local on global host k ranges but do NOT merge for now
        ith_attn_calc_host_k_ranges_local = host_k_ranges_global.make_ranges_local(
            ith_attn_calc_host_k_ranges_global,
            is_self_merged=True,
        )

        # make ith attn calc host slice local
        slice_maker = HostAttnSliceMaker(
            q_range_local=ith_attn_calc_host_q_range_local,
            k_ranges_local=ith_attn_calc_host_k_ranges_local,
            k_ranges_global=ith_attn_calc_host_k_ranges_global,
            calc_k_range_global=ith_attn_calc_k_ranges_global[0],
            mask_type_global=ith_attn_slice_global_mask_type,
        )
        attn_calc_host_slice_local_list.extend(slice_maker.make())

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_remote_slice_per_chunk(
        self,
        remote_k_ranges_global_per_chunk: list[AttnRanges],
        attn_calc_remote_slice_list_per_chunk: list[list[MultiKAttnSlice]],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
        chunk_remote_k_ranges_global_boundary: list[tuple[int, int]],
    ) -> None:
        """Make attn calc remote slice for the given remote k ranges global in each chunk,
            and append to 'attn_calc_remote_slice_list_per_chunk'
            called in'self._init_host_rank_entry_this_rank'
        HACK: inplace operation for 'attn_calc_remote_slice_list_per_chunk' for the purpose of performance,
              need further refactor.
        """
        for j in range(
            self.chunk_remote_k_ranges_global_start_idx,
            len(remote_k_ranges_global_per_chunk),
        ):
            jth_chunk_remote_k_ranges_global = remote_k_ranges_global_per_chunk[j]
            if (
                chunk_remote_k_ranges_global_boundary[j][0]
                >= ith_attn_calc_k_ranges_global[0].end
            ):
                break
            elif (
                chunk_remote_k_ranges_global_boundary[j][1]
                <= ith_attn_calc_k_ranges_global[0].start
            ):
                self.chunk_remote_k_ranges_global_start_idx += 1
                continue

            self._make_ith_attn_calc_remote_slice(
                remote_k_ranges_global=jth_chunk_remote_k_ranges_global,
                attn_calc_remote_slice_list=attn_calc_remote_slice_list_per_chunk[j],
                ith_attn_calc_host_q_range_local=ith_attn_calc_host_q_range_local,
                ith_attn_calc_k_ranges_global=ith_attn_calc_k_ranges_global,
                ith_attn_slice_global_mask_type=ith_attn_slice_global_mask_type,
            )

    @nvtx.instrument_nvtx
    def _make_ith_attn_calc_remote_slice(
        self,
        remote_k_ranges_global: AttnRanges,
        attn_calc_remote_slice_list: list[MultiKAttnSlice],
        ith_attn_calc_host_q_range_local: AttnRange,
        ith_attn_calc_k_ranges_global: AttnRanges,
        ith_attn_slice_global_mask_type: AttnMaskType,
    ) -> None:
        """Make ith attn calc remote slice for the given remote k ranges global,
            and append to given 'attn_calc_remote_slice_list',
            called in 'self._make_ith_attn_calc_remote_slice_per_chunk' for jth chunk of remote ranges
        HACK: inplace operation for 'attn_calc_remote_slice_list' for the purpose of performance,
              need further refactor.
        """

        # find the overlap part in the global remote k ranges
        ith_attn_calc_remote_k_ranges_global = (
            ith_attn_calc_k_ranges_global.find_overlap_ranges(
                remote_k_ranges_global,
                is_self_merged=True,
                is_other_merged=True,
            )
        )

        # no overlap with and ith slice
        if len(ith_attn_calc_remote_k_ranges_global) == 0:
            return

        # make ith attn calc remote multik slice
        slice_maker = RemoteAttnSliceMaker(
            q_range_local=ith_attn_calc_host_q_range_local,
            k_ranges_global=ith_attn_calc_remote_k_ranges_global,
            calc_k_range_global=ith_attn_calc_k_ranges_global[0],
            mask_type_global=ith_attn_slice_global_mask_type,
        )
        attn_calc_remote_slice_list.extend(slice_maker.make())  # type: ignore[arg-type]

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_per_stage_this_rank(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[RemoteRankEntry]:
        """Initialize remote rank entry per overlap stage for this rank"""

        # -------   calculate calc/comm cost pairs  ------ #

        chunk_costs = self._calc_cost_pairs_per_chunk(
            host_rank_entry_this_rank=host_rank_entry_this_rank,
        )

        # ------    solve the multi-stage overlap problem   ------ #
        # ------    and get chunk partitions for each stage   ------ #

        cost_partitions = self._solve_multi_stage_overlap(
            chunk_costs=chunk_costs,
        )

        # ------    calculate remote rank entry for each stage   ------ #

        remote_rank_entry_per_stage_this_rank = self._calc_remote_rank_entry_per_stage(
            host_rank_entry_this_rank=host_rank_entry_this_rank,
            cost_partitions=cost_partitions,
        )

        return remote_rank_entry_per_stage_this_rank

    @nvtx.instrument_nvtx
    def _init_remote_rank_entry_per_rank_per_stage(
        self, remote_rank_entry_per_stage_this_rank: list[RemoteRankEntry]
    ) -> list[list[RemoteRankEntry]]:
        """Initialize remote rank entry per rank for each overlap stage"""

        # all gather remote rank entry per stage from each rank
        remote_rank_entry_per_stage_per_rank = [None] * self.cp_size

        dist.all_gather_object(
            remote_rank_entry_per_stage_per_rank,
            remote_rank_entry_per_stage_this_rank,
            group=self.cp_group,
        )

        # check shape to be [cp_size, overlap_degree]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(remote_rank_entry_per_stage_per_rank) == self.cp_size
                and len(remote_rank_entry_per_stage_per_rank[0]) == self.overlap_degree  # type: ignore
            ), f"{len(remote_rank_entry_per_stage_per_rank)=}, {self.cp_size=}, {self.overlap_degree=}"

        # transpose to be remote rank entry per rank for each stage
        remote_rank_entry_per_rank_per_stage = transpose_matrix(
            remote_rank_entry_per_stage_per_rank  # type: ignore
        )

        # check shape to be [overlap_degree, cp_size]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(remote_rank_entry_per_rank_per_stage) == self.overlap_degree  # type: ignore
                and len(remote_rank_entry_per_rank_per_stage[0]) == self.cp_size
            ), f"{len(remote_rank_entry_per_rank_per_stage)=}, {self.overlap_degree=}, {self.cp_size=}"

        return remote_rank_entry_per_rank_per_stage

    @nvtx.instrument_nvtx
    def _calc_cost_pairs_per_chunk(
        self,
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[OverlapStageCost]:
        """Calculate the calc/comm cost pairs for each chunk
        called in 'self._init_remote_rank_entry_per_stage_this_rank'
        """
        # 1-1. host comm cost (must be 0. since no comm needs to be waited)
        host_comm_cost = 0.0

        # 1-2. host calc cost
        host_calc_area = host_rank_entry_this_rank.get_host_calc_area()
        host_calc_cost = self.overlap_config.calc_cost_factor * host_calc_area

        # 2-1. remote comm cost for each chunk
        remote_comm_size_per_chunk = [
            host_rank_entry_this_rank.get_remote_comm_size(chunk_idx)
            for chunk_idx in range(self.overlap_num_chunks)
        ]
        remote_comm_cost_per_chunk = [
            self.overlap_config.comm_cost_factor * comm_size
            for comm_size in remote_comm_size_per_chunk
        ]

        # 2-2. remote calc cost for each chunk
        remote_calc_area_per_chunk = [
            host_rank_entry_this_rank.get_remote_calc_area(chunk_idx)
            for chunk_idx in range(self.overlap_num_chunks)
        ]
        remote_calc_cost_per_chunk = [
            self.overlap_config.calc_cost_factor * remote_calc_area
            for remote_calc_area in remote_calc_area_per_chunk
        ]

        # 3-1. construct the stage cost pairs for each chunk
        chunk_costs = [
            OverlapStageCost(
                comm_cost=comm_cost,
                calc_cost=calc_cost,
            )
            for comm_cost, calc_cost in zip(
                [host_comm_cost] + remote_comm_cost_per_chunk,
                [host_calc_cost] + remote_calc_cost_per_chunk,
            )
        ]

        # 3-2. sanity check
        assert (
            len(chunk_costs) == self.overlap_num_chunks + 1
        ), f"{len(chunk_costs)=}, {self.overlap_num_chunks=}"

        return chunk_costs

    @nvtx.instrument_nvtx
    def _solve_multi_stage_overlap(
        self,
        chunk_costs: list[OverlapStageCost],
    ) -> list[list[int]]:
        """Solve the multi-stage overlap problem with the overlap solver
        called in 'self._init_remote_rank_entry_per_stage_this_rank'

        Args:
            chunk_costs: list of OverlapStageCost, each element is the cost pair for one chunk

        Returns:
            cost_partitions: list of list of int, each element is the chunk partition for one stage
        """

        # overlap solver will return the solution with the partitions of the chunk costs,
        # which is a list with length 'overlap_degree',
        # where the ith elem is a chunk idx list that
        # contains the chunk idxs that need be processed together in the ith stage
        # e.g. [[0, 2], [1, 4], [3, 5]] for 6 chunks and 3 overlap degree
        best_solution, solution_dict = self.overlap_solver.solve(
            stage_costs=chunk_costs,
            overlap_degree=self.overlap_config.degree,
            dynamic_max_degree=self.overlap_config.dynamic_max_degree,
        )

        # get the cost partitions of 1 host cost pair and n remote cost pairs
        cost_partitions = best_solution.partitions
        # sanity check
        assert (
            0 in cost_partitions[0]
        ), f"The host cost with index 0 must be in the first partition, but got {cost_partitions=}"

        # get the overlap degree w.r.t the best solution
        best_overlap_degree_this_rank = best_solution.overlap_degree
        if self.overlap_config.mode is AttnOverlapMode.STATIC:
            if magi_attention.is_sanity_check_enable():
                assert best_overlap_degree_this_rank == self.overlap_config.degree, (
                    f"in static mode, {best_overlap_degree_this_rank=} "
                    f"should be equal to {self.overlap_config.degree=}"
                )

            # if static mode, then each rank already has the same overlap degree
            # so there's no need to reduce and the overlap degree is the same as the config
            self.overlap_degree = best_overlap_degree_this_rank
        elif self.overlap_config.mode is AttnOverlapMode.DYNAMIC:
            # if dynamic mode, the final overlap degree is the maximum one among ranks
            # so we need to reduce the best overlap degree among ranks
            overlap_degree_reduce_tensor = torch.tensor(
                best_overlap_degree_this_rank,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(
                overlap_degree_reduce_tensor,
                op=dist.ReduceOp.MAX,
                group=self.cp_group,
            )
            final_overlap_degree = overlap_degree_reduce_tensor.item()

            for _ in range(best_overlap_degree_this_rank, final_overlap_degree):
                # HACK: for the rank with the best overlap degree < final overlap degree
                # we just append the idle stages to the last of the cost partitions
                cost_partitions.append([])

            self.overlap_degree = final_overlap_degree
        else:
            raise ValueError(f"Unknown overlap mode: {self.overlap_config.mode}")

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert (
                len(cost_partitions) == self.overlap_degree
            ), f"{len(cost_partitions)=}, {self.overlap_degree=}"

        return cost_partitions

    @nvtx.instrument_nvtx
    def _calc_remote_rank_entry_per_stage(
        self,
        cost_partitions: list[list[int]],
        host_rank_entry_this_rank: HostRankEntry,
    ) -> list[RemoteRankEntry]:
        """Calculate the remote rank entry per stage for this rank
        called in 'self._init_remote_rank_entry_per_stage_this_rank'
        """

        remote_rank_entry_per_stage_this_rank: list[RemoteRankEntry] = []

        for cost_partition in cost_partitions:
            remote_rank_entry_per_stage_this_rank.append(
                self._calc_remote_rank_entry_for_one_stage(
                    cost_partiton=cost_partition,
                    host_rank_entry_this_rank=host_rank_entry_this_rank,
                )
            )

        return remote_rank_entry_per_stage_this_rank

    # TODO delete some logic
    @nvtx.instrument_nvtx
    def _calc_remote_rank_entry_for_one_stage(
        self, cost_partiton: list[int], host_rank_entry_this_rank: HostRankEntry
    ) -> RemoteRankEntry:
        """Calculate the remote rank entry for one stage for this rank
        called in'self._calc_remote_rank_entry_per_stage'
        """

        # ------    merge the chunks into one overlap stage within each partition  ------ #
        # ------    and construct the remote rank entry for each stage   ------ #

        # init the args and some temp vars for remote rank entry
        remote_k_ranges_global_this_stage = AttnRanges()
        attn_calc_remote_slice_list_per_chunk_this_stage: list[
            list[MultiKAttnSlice]
        ] = []
        total_q_ranges_local_this_stage = AttnRanges()

        # find the remote_ k_ranges_global and remote_slice_local_list
        # within the chunks for this stage
        for cost_idx in cost_partiton:
            if cost_idx == 0:  # ignore the host cost
                continue
            chunk_idx = cost_idx - 1
            remote_k_ranges_global_this_stage.extend(
                host_rank_entry_this_rank.remote_k_ranges_global_per_chunk[chunk_idx]
            )
            attn_calc_remote_slice_list_per_chunk_this_stage.append(
                host_rank_entry_this_rank.attn_calc_remote_slice_list_per_chunk[
                    chunk_idx
                ]
            )
        remote_k_ranges_global_this_stage = remote_k_ranges_global_this_stage.merge()

        # add all q_ranges to total_q_ranges_this_bucket_local
        for attn_slice in chain(*attn_calc_remote_slice_list_per_chunk_this_stage):
            total_q_ranges_local_this_stage.append(attn_slice.q_range)
        total_q_ranges_boundary_local_this_stage = (
            total_q_ranges_local_this_stage.points
        )

        # get two dict as (key: q_range, value: k_ranges) and (key: q_range, value: masktype)
        (
            map_slice_q_range_to_k_ranges,
            map_slice_q_range_to_masktype,
        ) = self._calc_remote_rank_q_range_to_k_ranges_map(
            attn_calc_remote_slice_list_per_chunk_this_stage=attn_calc_remote_slice_list_per_chunk_this_stage,
            total_q_ranges_boundary_local_this_stage=total_q_ranges_boundary_local_this_stage,
        )

        # construct the attn_calc_remote_slice_local_list_this_stage
        q_range_k_ranges_tuples: list[tuple[AttnRange, AttnRanges]] = sorted(
            map_slice_q_range_to_k_ranges.items(),
            key=lambda t: t[0].start,  # sort by q_range.start
        )
        slice_tuples: list[tuple[AttnRange, AttnRanges, list[AttnMaskType]]] = [
            (q_range, k_ranges, map_slice_q_range_to_masktype[q_range])
            for q_range, k_ranges in q_range_k_ranges_tuples
        ]

        return self._make_remote_entry_for_one_stage(
            slice_tuples=slice_tuples,
            remote_k_ranges_global_this_stage=remote_k_ranges_global_this_stage,
            host_k_ranges_global_this_rank=host_rank_entry_this_rank.host_k_ranges_global,
        )

    @nvtx.instrument_nvtx
    def _make_remote_entry_for_one_stage(
        self,
        slice_tuples: list[tuple[AttnRange, AttnRanges, list[AttnMaskType]]],
        remote_k_ranges_global_this_stage: AttnRanges,
        host_k_ranges_global_this_rank: AttnRanges,
    ) -> RemoteRankEntry:
        """Make the remote entry for one stage
        called in 'self._calc_remote_rank_entry_for_one_stage' for each stage

        Args:
            slice_tuples: each tuple contains a q_range and it's corresponding k_ranges and mask types
            remote_k_ranges_global_this_stage: the remote k ranges for this stage
            host_k_ranges_global_this_rank: the host k ranges for this rank

        Returns:
            attn_calc_remote_slice_local_list_this_stage: the remote entry for this stage
        """
        attn_calc_remote_slice_local_list_this_stage: list[AttnSlice] = []

        for q_range, k_ranges, mask_types in slice_tuples:
            k_ranges = remote_k_ranges_global_this_stage.make_ranges_local(
                k_ranges,
                is_self_merged=True,
            )

            self._merge_ranges_and_masktypes(
                q_range=q_range,
                k_ranges=k_ranges,
                mask_types=mask_types,
                attn_calc_remote_slice_local_list_this_stage=attn_calc_remote_slice_local_list_this_stage,
            )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            assert remote_k_ranges_global_this_stage.is_merged()

        return RemoteRankEntry(
            host_k_ranges_global=host_k_ranges_global_this_rank,
            remote_k_ranges_global=remote_k_ranges_global_this_stage,
            attn_calc_remote_slice_local_list=attn_calc_remote_slice_local_list_this_stage,
        )

    @nvtx.instrument_nvtx
    def _merge_ranges_and_masktypes(
        self,
        q_range: AttnRange,
        k_ranges: AttnRanges,
        mask_types: list[AttnMaskType],
        attn_calc_remote_slice_local_list_this_stage: list[AttnSlice],
    ):
        """sort k_ranges and merge with masktype in cases,
        which q_range is overlaped with multi mask types
        """
        # sort k_ranges and masktypes with range start
        indices = argsort(k_ranges._ranges, key=lambda range: range.start)
        k_ranges_list = [k_ranges[i] for i in indices]
        mask_types = [mask_types[i] for i in indices]

        def _calc_new_masktype(
            old_masktype: AttnMaskType,
            new_masktype: AttnMaskType,
        ) -> AttnMaskType | None:
            """function to merge masktypes when k_range can be merged"""
            match (old_masktype, new_masktype):
                case (AttnMaskType.FULL, AttnMaskType.FULL):
                    return AttnMaskType.FULL
                case (AttnMaskType.FULL, AttnMaskType.CAUSAL):
                    return AttnMaskType.CAUSAL
                case (AttnMaskType.INVCAUSAL, AttnMaskType.FULL):
                    return AttnMaskType.INVCAUSAL
                case (AttnMaskType.INVCAUSAL, AttnMaskType.CAUSAL):
                    return AttnMaskType.BICAUSAL
                case _:
                    return None

        # init k_range and masktype
        k_range_start, k_range_end = k_ranges_list[0].start, k_ranges_list[0].end
        cur_mask_type = mask_types[0]

        for i in range(1, len(k_ranges)):
            can_be_merged = False
            if k_ranges_list[i].start == k_range_end:
                # calc new masktype when k_range can be merged
                new_masktype = _calc_new_masktype(cur_mask_type, mask_types[i])
                # invalid masktype means there is no suitable mask to merge the k_range.
                if new_masktype is not None:
                    k_range_end = k_ranges_list[i].end
                    cur_mask_type = new_masktype
                    can_be_merged = True

            if not can_be_merged:
                # add the current k_range to the array and set it as the new value
                k_range = AttnRange(start=k_range_start, end=k_range_end)
                attn_calc_remote_slice_local_list_this_stage.append(
                    AttnSlice(
                        q_range=q_range,
                        k_range=k_range,
                        mask_type=cur_mask_type,
                    )
                )

                # set to new value
                k_range_start, k_range_end = (
                    k_ranges_list[i].start,
                    k_ranges_list[i].end,
                )
                cur_mask_type = mask_types[i]

        # add the last k_range to the array
        k_range = AttnRange(start=k_range_start, end=k_range_end)
        attn_calc_remote_slice_local_list_this_stage.append(
            AttnSlice(
                q_range=q_range,
                k_range=k_range,
                mask_type=cur_mask_type,
            )
        )

    @nvtx.instrument_nvtx
    def _calc_remote_rank_q_range_to_k_ranges_map(
        self,
        attn_calc_remote_slice_list_per_chunk_this_stage: list[list[MultiKAttnSlice]],
        total_q_ranges_boundary_local_this_stage: list[int],
    ) -> tuple[
        defaultdict[AttnRange, AttnRanges], defaultdict[AttnRange, list[AttnMaskType]]
    ]:
        """Split the slice according to the boundary, and form maps from q_range to k_ranges and masktype.
        called in 'self._calc_remote_rank_entry_for_one_stage'
        """

        # init q_range->k_ranges map and q_range->masktype map
        map_slice_q_range_to_k_ranges: defaultdict[AttnRange, AttnRanges] = defaultdict(
            AttnRanges
        )
        map_slice_q_range_to_masktype: defaultdict[
            AttnRange, list[AttnMaskType]
        ] = defaultdict(list)

        for slice in chain(*attn_calc_remote_slice_list_per_chunk_this_stage):
            # find the start and end index in the boundary list
            slice_q_range_start, slice_q_range_end = (
                slice.q_range.start,
                slice.q_range.end,
            )
            boundary_left_index = bisect_left(
                total_q_ranges_boundary_local_this_stage, slice_q_range_start
            )
            boundary_right_index = bisect_left(
                total_q_ranges_boundary_local_this_stage, slice_q_range_end
            )

            # traverse from computed boundary start to end
            for boundary_idx in range(boundary_left_index, boundary_right_index):
                boundary_start, boundary_end = (
                    total_q_ranges_boundary_local_this_stage[boundary_idx],
                    total_q_ranges_boundary_local_this_stage[boundary_idx + 1],
                )

                # create the segmented q_range.
                q_range_this_slice = AttnRange(start=boundary_start, end=boundary_end)

                if (
                    slice.mask_types[-1] == AttnMaskType.FULL
                    and slice.mask_types[0] == AttnMaskType.FULL
                ):
                    # in the case of full, no need to handle k_ranges.
                    map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                        slice.k_ranges
                    )
                    map_slice_q_range_to_masktype[q_range_this_slice].extend(
                        [AttnMaskType.FULL] * len(slice.k_ranges)
                    )
                elif slice.mask_types[-1] == AttnMaskType.CAUSAL:
                    # in the case of causal, the end of the last range in k_ranges may need to be shortened
                    distance_to_slice_end = slice_q_range_end - boundary_end

                    if distance_to_slice_end == 0:
                        # don't need to shorten k_range
                        map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                            slice.k_ranges
                        )
                    else:
                        map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                            slice.k_ranges[:-1]
                        )
                        # shorten causal k_range
                        last_k_range = AttnRange(
                            start=slice.k_ranges[-1].start,
                            end=slice.k_ranges[-1].end - distance_to_slice_end,
                        )
                        map_slice_q_range_to_k_ranges[q_range_this_slice].append(
                            last_k_range
                        )

                    map_slice_q_range_to_masktype[q_range_this_slice].extend(
                        [AttnMaskType.FULL] * (len(slice.k_ranges) - 1)
                        + [AttnMaskType.CAUSAL]
                    )
                elif slice.mask_types[0] == AttnMaskType.INVCAUSAL:
                    # in the case of inv_causal, the start of the first range in k_ranges may need to lengthen
                    distance_to_slice_start = boundary_start - slice_q_range_start

                    if distance_to_slice_start == 0:
                        # don't need to lengthen k_range
                        map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                            slice.k_ranges
                        )
                    else:
                        # lengthen inv_causal k_range
                        first_k_range = AttnRange(
                            start=slice.k_ranges[0].start + distance_to_slice_start,
                            end=slice.k_ranges[0].end,
                        )
                        map_slice_q_range_to_k_ranges[q_range_this_slice].append(
                            first_k_range
                        )
                        map_slice_q_range_to_k_ranges[q_range_this_slice].extend(
                            slice.k_ranges[1:]
                        )

                    map_slice_q_range_to_masktype[q_range_this_slice].extend(
                        [AttnMaskType.INVCAUSAL]
                        + [AttnMaskType.FULL] * (len(slice.k_ranges) - 1)
                    )
                elif slice.mask_types[0] == AttnMaskType.BICAUSAL:
                    # in case of bicausal, the start and end may both need to change
                    if magi_attention.is_sanity_check_enable():
                        assert len(slice.k_ranges) == 1, (
                            f"when masktype is bi_causal, the length of k_ranges must be 1, "
                            f"but got {len(slice.k_ranges)=}"
                        )

                    distance_to_slice_start = boundary_start - slice_q_range_start
                    distance_to_slice_end = slice_q_range_end - boundary_end

                    bicausal_k_range = AttnRange(
                        start=slice.k_ranges[0].start + distance_to_slice_start,
                        end=slice.k_ranges[0].end - distance_to_slice_end,
                    )

                    map_slice_q_range_to_k_ranges[q_range_this_slice].append(
                        bicausal_k_range
                    )
                    map_slice_q_range_to_masktype[q_range_this_slice].append(
                        AttnMaskType.BICAUSAL
                    )
                else:
                    raise ValueError(
                        f"Only support 'full', 'causal', 'inv_causal' and 'bi_causal' mask, "
                        f"but got {slice.mask_types[-1]} and {slice.mask_types[0]}"
                    )

        return (
            map_slice_q_range_to_k_ranges,
            map_slice_q_range_to_masktype,
        )

    @nvtx.instrument_nvtx
    def _init_transfer_table_per_stage(
        self,
        remote_rank_entry_per_rank_per_stage: list[list[RemoteRankEntry]],
    ) -> list[TransferTable]:
        """Initialize transfer table per stage for this rank"""

        transfer_table_per_stage: list[TransferTable] = []

        transfer_info_per_stage_this_rank: list[TransferInfo] = [
            self._init_transfer_info_this_rank_for_one_stage(
                remote_rank_entry_per_rank_this_stage
            )
            for remote_rank_entry_per_rank_this_stage in (
                remote_rank_entry_per_rank_per_stage
            )
        ]

        transfer_info_per_rank_per_stage = self._init_transfer_info_per_rank_per_stage(
            transfer_info_per_stage_this_rank
        )

        transfer_table_per_stage = [
            self._init_transfer_table_for_one_stage(
                remote_rank_entry_per_rank_this_stage,
                transfer_info_per_rank_this_stage,
            )
            for (
                remote_rank_entry_per_rank_this_stage,
                transfer_info_per_rank_this_stage,
            ) in zip(
                remote_rank_entry_per_rank_per_stage,
                transfer_info_per_rank_per_stage,
            )
        ]

        return transfer_table_per_stage

    @nvtx.instrument_nvtx
    def _init_transfer_table_for_one_stage(
        self,
        remote_rank_entry_per_rank: list[RemoteRankEntry],
        transfer_info_per_rank: list[TransferInfo],
    ) -> TransferTable:
        """Initialize transfer table for each overlap stage
        called in 'self._init_transfer_table_per_stage'
        """

        # init transfer table entry for each rank pair: (send_ranki, recv_rankj)
        transfer_table = TransferTable(cp_size=self.cp_size)

        # fill up transfer table
        for send_rank in range(self.cp_size):  # for each send_ranki
            transfer_info = transfer_info_per_rank[send_rank]
            group_cast_ranges_global_transfer = (
                transfer_info.group_cast_ranges_global_transfer
            )
            group_cast_ranges_local_send_to = (
                transfer_info.group_cast_ranges_local_send_to
            )

            # for each non-overlapped global/local k range that send_ranki needs to send to
            # we tranverse each dest recv_rankj to recv it in the set,
            # and append it to k_ranges_local_in_send_buf at the (send_ranki, recv_rankj) table entry
            for r in group_cast_ranges_global_transfer:
                k_range = AttnRange(start=r.start, end=r.end)
                if send_rank == self.cp_rank:  # the send row for this rank
                    for recv_rank in r.rank_set:
                        transfer_table.append_k_ranges_global(
                            send_rank=self.cp_rank,
                            recv_rank=recv_rank,
                            k_range=k_range,
                        )
                elif self.cp_rank in r.rank_set:  # the recv col for this rank
                    transfer_table.append_k_ranges_global(
                        send_rank=send_rank,
                        recv_rank=self.cp_rank,
                        k_range=k_range,
                    )

            for r in group_cast_ranges_local_send_to:
                k_range = AttnRange(start=r.start, end=r.end)
                if send_rank == self.cp_rank:  # the send row for this rank
                    for recv_rank in r.rank_set:
                        transfer_table.append_k_ranges_local_in_send_buf(
                            send_rank=self.cp_rank,
                            recv_rank=recv_rank,
                            k_range=k_range,
                        )
                elif self.cp_rank in r.rank_set:  # the recv col for this rank
                    transfer_table.append_k_ranges_local_in_send_buf(
                        send_rank=send_rank,
                        recv_rank=self.cp_rank,
                        k_range=k_range,
                    )

            # sort the k ranges in each table entry
            if send_rank == self.cp_rank:  # the send row for this rank
                for recv_rank in range(self.cp_size):
                    transfer_table.sort_k_ranges_global(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                    )
                    transfer_table.sort_k_ranges_local_in_send_buf(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                    )
                    # fill k_ranges_local_in_recv_buf
                    transfer_table.make_k_ranges_local_in_recv_buf(
                        send_rank=self.cp_rank,
                        recv_rank=recv_rank,
                        remote_k_ranges_global_for_recv_rank=remote_rank_entry_per_rank[
                            recv_rank
                        ].remote_k_ranges_global,
                    )
            else:  # the recv col for this rank
                transfer_table.sort_k_ranges_global(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                transfer_table.sort_k_ranges_local_in_send_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                # fill k_ranges_local_in_recv_buf
                transfer_table.make_k_ranges_local_in_recv_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                    remote_k_ranges_global_for_recv_rank=remote_rank_entry_per_rank[
                        self.cp_rank
                    ].remote_k_ranges_global,
                )

        return transfer_table

    @nvtx.instrument_nvtx
    def _init_transfer_info_this_rank_for_one_stage(
        self,
        remote_rank_entry_per_rank: list[RemoteRankEntry],
    ) -> TransferInfo:
        """Initialize transfer info for this rank for certain stage
        called in 'self._init_transfer_table_per_stage'
        """

        host_k_ranges_global_this_rank = remote_rank_entry_per_rank[
            self.cp_rank
        ].host_k_ranges_global
        remote_k_ranges_global_this_rank = remote_rank_entry_per_rank[
            self.cp_rank
        ].remote_k_ranges_global

        # init k ranges global/local for send_to/recv_from per rank
        k_ranges_global_recv_from_per_rank: list[AttnRanges] = []
        k_ranges_local_recv_from_per_rank: list[AttnRanges] = []
        k_ranges_global_send_to_per_rank: list[AttnRanges] = []
        k_ranges_local_send_to_per_rank: list[AttnRanges] = []
        for rank in range(self.cp_size):
            if rank == self.cp_rank:  # no need to recv from / send to this rank
                k_ranges_global_recv_from_per_rank.append(AttnRanges())
                k_ranges_local_recv_from_per_rank.append(AttnRanges())
                k_ranges_global_send_to_per_rank.append(AttnRanges())
                k_ranges_local_send_to_per_rank.append(AttnRanges())
                continue

            # ----------    for k_ranges recv from     ---------- #

            # get the global k ranges that this rank needs to recv from current rank
            rank_host_k_ranges_global = remote_rank_entry_per_rank[
                rank
            ].host_k_ranges_global
            k_ranges_global_recv_from_rank = (
                remote_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_host_k_ranges_global,
                    is_self_merged=True,
                    is_other_merged=True,
                )
            )
            # make the global k ranges local w.r.t. self's recv buffer
            k_ranges_local_recv_from_rank = (
                remote_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_recv_from_rank,
                    is_self_merged=True,
                )
            )
            # add to recv transfer info for both global and local ones
            k_ranges_global_recv_from_per_rank.append(k_ranges_global_recv_from_rank)
            k_ranges_local_recv_from_per_rank.append(k_ranges_local_recv_from_rank)

            # ----------    for k_ranges send to     ---------- #

            # get the global k ranges that this rank needs to send to current rank
            rank_remote_k_ranges_global = remote_rank_entry_per_rank[
                rank
            ].remote_k_ranges_global
            k_ranges_global_send_to_rank = (
                host_k_ranges_global_this_rank.find_overlap_ranges(
                    rank_remote_k_ranges_global,
                    is_self_merged=True,
                    is_other_merged=True,
                )
            )
            # make the global k ranges local w.r.t. self's send buffer
            k_ranges_local_send_to_rank = (
                host_k_ranges_global_this_rank.make_ranges_local(
                    k_ranges_global_send_to_rank,
                    is_self_merged=True,
                )
            )
            # add to send transfer info for both global and local ones
            k_ranges_global_send_to_per_rank.append(k_ranges_global_send_to_rank)
            k_ranges_local_send_to_per_rank.append(k_ranges_local_send_to_rank)

        # init group_cast_ranges for global/local k ranges that send_ranki needs to send to
        # which splits the local ranges into non-overlapped local ranges
        group_cast_ranges_global_transfer = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=k_ranges_global_send_to_per_rank,
        )
        group_cast_ranges_local_send_to = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=k_ranges_local_send_to_per_rank,
        )

        transfer_info_this_rank = TransferInfo(
            k_ranges_global_recv_from_per_rank=k_ranges_global_recv_from_per_rank,
            k_ranges_local_recv_from_per_rank=k_ranges_local_recv_from_per_rank,
            k_ranges_global_send_to_per_rank=k_ranges_global_send_to_per_rank,
            k_ranges_local_send_to_per_rank=k_ranges_local_send_to_per_rank,
            group_cast_ranges_global_transfer=group_cast_ranges_global_transfer,
            group_cast_ranges_local_send_to=group_cast_ranges_local_send_to,
        )

        return transfer_info_this_rank

    @nvtx.instrument_nvtx
    def _init_transfer_info_per_rank_per_stage(
        self,
        transfer_info_per_stage_this_rank: list[TransferInfo],
    ) -> list[list[TransferInfo]]:
        """Initialize transfer info per rank for each stage
        called in 'self._init_transfer_table_per_stage'
        """

        # all gather transfer info per stage from each rank
        transfer_info_per_stage_per_rank = [None] * self.cp_size
        dist.all_gather_object(
            transfer_info_per_stage_per_rank,
            transfer_info_per_stage_this_rank,
            group=self.cp_group,
        )

        # check shape to be [cp_size, overlap_degree]
        if magi_attention.is_sanity_check_enable():
            assert (
                len(transfer_info_per_stage_per_rank) == self.cp_size
                and len(transfer_info_per_stage_per_rank[0]) == self.overlap_degree  # type: ignore
            )

        # transpose to be transfer info per rank for each stage
        transfer_info_per_rank_per_stage = transpose_matrix(
            transfer_info_per_stage_per_rank  # type: ignore
        )

        # sanity check
        if magi_attention.is_sanity_check_enable():
            # for each stage:
            #   for each rank pair (i≠j): (send_ranki, recv_rankj)
            #       whether the global k ranges that send_ranki needs to send to recv_rankj
            #       are equal to the ones that recv_rankj needs to recv from send_ranki
            for stage, transfer_info_per_rank in enumerate(
                transfer_info_per_rank_per_stage
            ):
                for send_rank in range(self.cp_size):
                    for recv_rank in range(self.cp_size):
                        if send_rank == recv_rank:
                            continue

                        send_info: TransferInfo = transfer_info_per_rank[send_rank]
                        recv_info: TransferInfo = transfer_info_per_rank[recv_rank]
                        k_ranges_global_recv_from_send_rank = (
                            recv_info.k_ranges_global_recv_from_per_rank[send_rank]
                        )
                        k_ranges_global_send_to_recv_rank = (
                            send_info.k_ranges_global_send_to_per_rank[recv_rank]
                        )

                        assert (
                            k_ranges_global_recv_from_send_rank
                            == k_ranges_global_send_to_recv_rank
                        ), (
                            f"The sanity check for transfer table at {stage=} failed:\n"
                            f"For rank pair ({send_rank=} {recv_rank=}), we got:\n"
                            f"{k_ranges_global_recv_from_send_rank=}\n"
                            f"{k_ranges_global_send_to_recv_rank=}"
                        )

        return transfer_info_per_rank_per_stage

    @nvtx.instrument_nvtx
    def make_comm_meta(self) -> CommMeta:
        """Calculate communication meta for kv and qo group collective"""

        num_remote_kv_tokens_per_stage: list[int] = []
        kv_group_collective_args_list: list[GroupCollectiveArg] = []

        # NOTE: this solver does not support qo comm for now
        # thus we assign empty args for qo comm
        num_remote_qo_tokens_per_stage: list[int] = [0] * self.overlap_degree
        qo_group_collective_args_list: list[GroupCollectiveArg] = [None] * self.overlap_degree  # type: ignore[list-item]

        for transfer_table_this_stage, remote_rank_entry_per_rank_this_stage in zip(
            self.transfer_table_per_stage,
            self.remote_rank_entry_per_rank_per_stage,
        ):
            total_seqlen_host_k = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].host_k_ranges_global.total_seqlen

            num_remote_kv_tokens = remote_rank_entry_per_rank_this_stage[
                self.cp_rank
            ].remote_k_ranges_global.total_seqlen

            kv_group_collective_arg = self._calc_kv_group_collective_arg(
                transfer_table_this_stage,
                total_seqlen_host_k,
            )

            num_remote_kv_tokens_per_stage.append(num_remote_kv_tokens)
            kv_group_collective_args_list.append(kv_group_collective_arg)

        # build comm meta
        comm_meta = CommMeta(
            num_remote_kv_tokens_per_stage=num_remote_kv_tokens_per_stage,
            kv_group_collective_args_list=kv_group_collective_args_list,
            num_remote_qo_tokens_per_stage=num_remote_qo_tokens_per_stage,
            qo_group_collective_args_list=qo_group_collective_args_list,
        )

        return comm_meta

    @nvtx.instrument_nvtx
    def _calc_kv_group_collective_arg(
        self,
        transfer_table: TransferTable,
        total_seqlen_host_k: int,
    ) -> GroupCollectiveArg:
        """Calculate group collective args from one transfer table
        called in 'self.make_comm_meta'
        """
        # retrieve group cast ranges for local k ranges that this rank needs to send to
        # which splits the local ranges into non-overlapped local ranges
        group_cast_ranges_local_send_to = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=[
                transfer_table.get_k_ranges_local_in_send_buf(
                    send_rank=self.cp_rank,
                    recv_rank=recv_rank,
                )
                for recv_rank in range(self.cp_size)
            ],
        )

        # calc input split size list with dst indices list
        input_split_size_list: list[int] = []
        dst_indices_list: list[list[int]] = []

        last_end = 0
        for r in group_cast_ranges_local_send_to:
            if r.start != last_end:  # [last_end, r.start) has no dest rank
                # FIXME: this branch is unreachable in the current test cases
                input_split_size_list.append(r.start - last_end)
                dst_indices_list.append([])

            input_split_size_list.append(r.seqlen)
            dst_indices_list.append(list(r.rank_set))
            last_end = r.end

        if last_end != total_seqlen_host_k:  # [last_end, seqlen) has no dest rank
            input_split_size_list.append(total_seqlen_host_k - last_end)
            dst_indices_list.append([])

        # retrieve group cast ranges for local k ranges that this rank needs to recv from
        group_cast_ranges_local_recv_from = GroupCastRanges(
            cp_size=self.cp_size,
            ranges_per_rank=[
                transfer_table.get_k_ranges_local_in_recv_buf(
                    send_rank=send_rank,
                    recv_rank=self.cp_rank,
                )
                for send_rank in range(self.cp_size)
            ],
            # NOTE: no need to split group cast ranges for recv
            split=False,
        )

        # calc output split size list with src index list
        output_split_size_list = []
        src_index_list = []

        if magi_attention.is_sanity_check_enable():
            # NOTE: as for group cast semantics,
            # there's only one src rank that sends the corr. data into
            # each non-overlapped range in recv buffer
            for r in group_cast_ranges_local_recv_from:
                assert len(r.rank_set) == 1

        for r in group_cast_ranges_local_recv_from:
            output_split_size_list.append(r.seqlen)
            src_index_list.append(r.rank_set.pop())

        # build group collective arg
        group_collective_arg = GroupCollectiveArg(
            input_split_size_list=input_split_size_list,
            output_split_size_list=output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            rank=self.cp_rank,
            world_size=self.cp_size,
            group=self.cp_group,
            device_mesh=self.cp_mesh,
            deterministic=self.deterministic,
        )

        # sanity check for group-cast arg per rank
        # NOTE: we don't need to do sanity check for group-reduce arg per rank
        # since they are symmetric in dist-attn
        if magi_attention.is_sanity_check_enable():
            group_collective_arg_per_rank: list[dict] = [None] * self.cp_size  # type: ignore
            dist.all_gather_object(
                group_collective_arg_per_rank,
                # since some attrs of GroupCollectiveArg like process group
                # can not be serialized, here we only gather the meta args
                dict(
                    input_split_size_list=input_split_size_list,
                    output_split_size_list=output_split_size_list,
                    dst_indices_list=dst_indices_list,
                    src_index_list=src_index_list,
                ),
                group=self.cp_group,
            )

            sanity_check_for_group_cast_meta_args_per_rank(
                input_split_size_list_per_rank=[
                    arg["input_split_size_list"]
                    for arg in group_collective_arg_per_rank
                ],
                output_split_size_list_per_rank=[
                    arg["output_split_size_list"]
                    for arg in group_collective_arg_per_rank
                ],
                dst_indices_list_per_rank=[
                    arg["dst_indices_list"] for arg in group_collective_arg_per_rank
                ],
                src_index_list_per_rank=[
                    arg["src_index_list"] for arg in group_collective_arg_per_rank
                ],
                world_size=self.cp_size,
                check_nccl_send_recv=True,
            )

        return group_collective_arg

    @nvtx.instrument_nvtx
    def make_calc_meta(self) -> CalcMeta:
        """Calculate flex-flash-attention calculation meta"""

        if magi_attention.is_sanity_check_enable():
            # check local attn calc
            assert all(
                attn_slice is not None
                for attn_slice in self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
            )

            # check remote attn calc for each overlap stage
            for (
                remote_rank_entry_this_stage_this_rank
            ) in self.remote_rank_entry_per_stage_this_rank:
                assert all(
                    attn_slice is not None
                    for attn_slice in remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
                )

        # ---   build local attn args   --- #

        host_slice_local_list = (
            self.host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )
        local_attn_arg = AttnArg(
            q_ranges=AttnRanges.from_ranges(
                [attn_slice.q_range for attn_slice in host_slice_local_list]  # type: ignore[arg-type]
            ),
            k_ranges=AttnRanges.from_ranges(
                [attn_slice.k_range for attn_slice in host_slice_local_list]  # type: ignore[arg-type]
            ),
            attn_type_map=[
                attn_slice.mask_type.to_int_type()  # type: ignore
                for attn_slice in host_slice_local_list
            ],
            total_area=sum(attn_slice.area for attn_slice in host_slice_local_list),
        )

        # ---   build remote attn args for each overlap stage   --- #

        remote_attn_args_list = []
        seqlen_k_per_remote_stage = []
        for (
            remote_rank_entry_this_stage_this_rank
        ) in self.remote_rank_entry_per_stage_this_rank:
            remote_slice_local_list = (
                remote_rank_entry_this_stage_this_rank.attn_calc_remote_slice_local_list
            )
            remote_attn_args_list.append(
                AttnArg(
                    q_ranges=AttnRanges.from_ranges(
                        [attn_slice.q_range for attn_slice in remote_slice_local_list]  # type: ignore[arg-type]
                    ),
                    k_ranges=AttnRanges.from_ranges(
                        [attn_slice.k_range for attn_slice in remote_slice_local_list]  # type: ignore[arg-type]
                    ),
                    attn_type_map=[
                        attn_slice.mask_type.to_int_type()  # type: ignore
                        for attn_slice in remote_slice_local_list
                    ],
                    total_area=sum(
                        attn_slice.area for attn_slice in remote_slice_local_list
                    ),
                )
            )
            # Get num_remote_kv_tokens for this stage
            num_remote_kv_tokens = (
                remote_rank_entry_this_stage_this_rank.remote_k_ranges_global.total_seqlen
            )
            seqlen_k_per_remote_stage.append(num_remote_kv_tokens)

        # ---   build attn calc meta   --- #
        calc_meta = CalcMeta(
            local_attn_arg=local_attn_arg,
            remote_attn_args_list=remote_attn_args_list,
            seqlen_q_shard=self.shard_seqlen_q,
            seqlen_k_local=self.total_seqlen_k - sum(seqlen_k_per_remote_stage),
            seqlen_k_per_remote_stage=seqlen_k_per_remote_stage,
        )

        return calc_meta

    def __repr__(self, title_len: int = 50) -> str:  # pragma: no cover
        repr_contents = []

        repr_summary = self._repr_host_info(
            self.host_rank_entry_this_rank, title_len=title_len
        )
        repr_contents.append(repr_summary)

        for stage, (
            transfer_table_this_stage,
            remote_rank_entry_per_rank_this_stage,
        ) in enumerate(
            zip(
                self.transfer_table_per_stage,
                self.remote_rank_entry_per_rank_per_stage,
            )
        ):
            repr_this_stage = self._repr_remote_info_for_one_stage(
                stage,
                transfer_table_this_stage,
                remote_rank_entry_per_rank_this_stage,
                title_len=title_len,
            )

            repr_contents.append(repr_this_stage)

        # add separator
        repr_contents.append("\n\n")

        return "\n\n".join(repr_contents)

    def _repr_host_info(
        self, host_rank_entry_this_rank: HostRankEntry, title_len: int = 50
    ) -> str:  # pragma: no cover
        repr_info = []

        # add summary info title
        stage_title = "  Host Info  "
        repr_info.append("\n" + "=" * title_len + stage_title + "=" * title_len + "\n")

        host_q_ranges_global = host_rank_entry_this_rank.host_q_ranges_global
        host_k_ranges_global = host_rank_entry_this_rank.host_k_ranges_global
        remote_k_ranges_global = host_rank_entry_this_rank.remote_k_ranges_global
        attn_calc_slice_global_list = (
            host_rank_entry_this_rank.attn_calc_slice_global_list
        )
        attn_calc_host_slice_local_list = (
            host_rank_entry_this_rank.attn_calc_host_slice_local_list
        )

        repr_info.append(f"host_q_ranges_global: {host_q_ranges_global}")
        repr_info.append(f"host_k_ranges_global: {host_k_ranges_global}")
        repr_info.append(f"remote_k_ranges_global: {remote_k_ranges_global}")
        repr_info.append(f"attn_calc_slice_global_list: {attn_calc_slice_global_list}")
        repr_info.append(
            f"attn_calc_host_slice_local_list: {attn_calc_host_slice_local_list}"
        )

        return "\n".join(repr_info)

    def _repr_remote_info_for_one_stage(
        self,
        stage: int,
        transfer_table_this_stage: TransferTable,
        remote_rank_entry_per_rank_this_stage: list[RemoteRankEntry],
        title_len: int = 50,
    ) -> str:  # pragma: no cover
        # calculate the max width for each cell
        cell_widths = [[0] * self.cp_size for _ in range(self.cp_size)]
        for send_rank in range(self.cp_size):
            for recv_rank in range(self.cp_size):
                send_str = f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}"
                recv_str = f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}"
                global_str = f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}"

                width = max(len(send_str), len(recv_str), len(global_str))
                cell_widths[send_rank][recv_rank] = width

        # calculate the max width for each column
        col_widths = [
            max(
                max(cell_widths[row][col] for row in range(self.cp_size)),
                len(
                    "host_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].host_k_ranges_global}"
                ),
                len(
                    "remote_k_ranges_global: "
                    f"{remote_rank_entry_per_rank_this_stage[col].remote_k_ranges_global}"
                ),
                len(
                    "attn_calc_remote_slice_local_list: "
                    f"{remote_rank_entry_per_rank_this_stage[col].attn_calc_remote_slice_local_list}"
                ),
            )
            for col in range(self.cp_size)
        ]

        # calculate the total width of the table
        # considering the separators " | " and the "row xx |" prefix
        table_width = (
            sum(col_widths) + 4 * (self.cp_size - 1) + 7
        )  # each column separator width is 4 and the prefix width is 7

        # construct table
        repr_info_this_stage = []

        # add a separator line for each overlap stage
        stage_title = f"  Remote Info for Stage {stage}  "
        repr_info_this_stage.append(
            "\n" + "=" * title_len + stage_title + "=" * title_len + "\n"
        )

        # add a title line for each col (expanded to 5 rows height)
        repr_info_this_stage.append("\n" + "-" * table_width)

        # first row: col number
        header_cells = [f"col{j:2d}".center(col_widths[j]) for j in range(self.cp_size)]
        repr_info_this_stage.append("r/c   | " + " | ".join(header_cells) + " |")

        # second row: host_k_ranges_global
        host_cells = [
            f"host_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].host_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(host_cells) + " |")

        # third row: remote_k_ranges_global
        remote_cells = [
            f"remote_k_ranges_global: {remote_rank_entry_per_rank_this_stage[j].remote_k_ranges_global}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_cells) + " |")

        # fourth row: attn_calc_remote_slice_local_list
        remote_slice_cells = [
            "attn_calc_remote_slice_local_list: "
            f"{remote_rank_entry_per_rank_this_stage[j].attn_calc_remote_slice_local_list}".ljust(
                col_widths[j]
            )
            for j in range(self.cp_size)
        ]
        repr_info_this_stage.append("      | " + " | ".join(remote_slice_cells) + " |")

        # add a separator line
        repr_info_this_stage.append("-" * table_width)

        # add each row
        for send_rank in range(self.cp_size):
            # add the cell content
            cell_lines = []
            for recv_rank in range(self.cp_size):
                col_width = col_widths[recv_rank]
                cell_content = [
                    f"send: {transfer_table_this_stage.get_k_ranges_local_in_send_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"recv: {transfer_table_this_stage.get_k_ranges_local_in_recv_buf(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                    f"global: {transfer_table_this_stage.get_k_ranges_global(send_rank, recv_rank)}".ljust(
                        col_width
                    ),
                ]
                cell_lines.append(cell_content)

            # concatenate the lines to form the cell
            for line_idx in range(3):
                prefix = f"row{send_rank:2d} |" if line_idx == 0 else "      |"
                line = [cell_lines[j][line_idx] for j in range(self.cp_size)]
                repr_info_this_stage.append(f"{prefix} " + " | ".join(line) + " |")

            repr_info_this_stage.append("-" * table_width)  # add a separator line

        return "\n".join(repr_info_this_stage)
