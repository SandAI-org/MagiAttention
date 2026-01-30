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

import itertools
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.comm.primitive.grpcoll._mgr import grpcoll_buffer_mgr
from magi_attention.common import AttnForwardMeta, AttnRanges
from magi_attention.common.enum import AttnMaskType, AttnRole
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    GrpCollConfig,
    OverlapConfig,
)
from magi_attention.functional.dispatch import dispatch_func, undispatch_func
from magi_attention.functional.dist_attn import DistAttnRuntime, dist_attn_func
from magi_attention.meta import (
    make_attn_meta_from_dispatch_meta,
    make_dispatch_meta_from_qk_ranges,
)
from magi_attention.meta.collection import DispatchMeta
from magi_attention.meta.collection.calc_meta import AttnArg, CalcMeta
from magi_attention.meta.collection.comm_meta import CommMeta
from magi_attention.meta.solver.dist_attn_solver import (
    BaseDistAttnSolver,
    DistAttnSolver,
)
from magi_attention.meta.solver.dynamic_attn_solver import DynamicAttnSolver
from magi_attention.utils import is_list_value_all, is_same_process_group, wrap_to_list


@dataclass(frozen=True)
class DistAttnRuntimeKey:
    q_ranges: AttnRanges
    k_ranges: AttnRanges
    attn_mask_type: tuple[AttnMaskType, ...]
    total_seqlen_q: int
    total_seqlen_k: int
    pad_size: int
    chunk_size: int
    cp_group: dist.ProcessGroup
    cp_mesh: DeviceMesh | None
    dist_attn_config: DistAttnConfig
    num_heads_q: int
    num_heads_kv: int
    # flags that might influence the runtime behavior
    is_deterministic_mode_enable: bool
    is_hierarchical_comm_enable: bool
    is_qo_comm_enable: bool
    is_native_grpcoll_enable: bool
    is_flatten_head_groups_enable: bool
    is_sdpa_backend_enable: bool
    is_fa4_backend_enable: bool


class DistAttnRuntimeMgr:
    def __init__(
        self,
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        dist_attn_config: DistAttnConfig,
        attn_solver: BaseDistAttnSolver,
        dist_attn_runtime: DistAttnRuntime,
        cp_group: dist.ProcessGroup,
        *,
        ref_q_ranges: AttnRanges,
        ref_k_ranges: AttnRanges,
        is_same_source: bool,
        is_q_permutable: bool,
        is_k_permutable: bool,
        num_heads_q: int,
        num_heads_kv: int,
    ):
        self.cp_group = cp_group
        self.dispatch_meta_q = dispatch_meta_q
        self.dispatch_meta_k = dispatch_meta_k
        self.dist_attn_config = dist_attn_config
        self.attn_solver = attn_solver

        self.dist_attn_runtime = dist_attn_runtime

        self.ref_q_ranges = ref_q_ranges
        self.ref_k_ranges = ref_k_ranges
        self.is_same_source = is_same_source
        self.is_q_permutable = is_q_permutable
        self.is_k_permutable = is_k_permutable

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self._q_position_ids: None | torch.Tensor = None
        self._k_position_ids: None | torch.Tensor = None

    def dispatch_qo(self, q_or_o: torch.Tensor) -> torch.Tensor:
        q_or_o = dispatch_func(
            x_global=q_or_o,
            group=self.cp_group,
            meta=self.dispatch_meta_q,
        )
        return q_or_o

    def dispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = dispatch_func(
            x_global=k_or_v,
            group=self.cp_group,
            meta=self.dispatch_meta_k,
        )
        return k_or_v

    def undispatch_qo(self, q_or_o: torch.Tensor) -> torch.Tensor:
        q_or_o = undispatch_func(
            x_local=q_or_o,
            group=self.cp_group,
            meta=self.dispatch_meta_q,
        )
        return q_or_o

    def undispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = undispatch_func(
            x_local=k_or_v,
            group=self.cp_group,
            meta=self.dispatch_meta_k,
        )
        return k_or_v

    def calc_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
    ) -> tuple[torch.Tensor, AttnForwardMeta]:
        return dist_attn_func(
            q=q,
            k=k,
            v=v,
            dist_attn_runtime=self.dist_attn_runtime,
            sink=sink,
            softmax_scale=softmax_scale,
            softcap=softcap,
        )

    def get_xattn_args(
        self,
        ref_xattn_q_ranges: AttnRanges,
        ref_xattn_k_ranges: AttnRanges,
        attn_mask_type: AttnMaskType | list[AttnMaskType],
        return_host_only: bool = False,
    ) -> AttnArg:
        """
        Get the attn arg for cross attention.

        Since dist_attn_runtime_mgr may modify q_ranges and k_ranges,
        if this query tensor needs to perform cross attention with other key tensors later,
        we may need to update the q_ranges and k_ranges for cross attention.

        Args:
            xattn_k_ranges(AttnRanges): The key ranges to be updated for cross attention
            attn_mask_type(AttnMaskType | list[AttnMaskType]): The attn mask type for cross attention
            return_host_only(bool): Whether to return the attn arg for cross attention on this rank only

        Returns:
            attn_arg(AttnArg): The attn arg for cross attention
        """
        attn_mask_type = wrap_to_list(attn_mask_type)
        assert is_list_value_all(
            attn_mask_type, AttnMaskType.FULL
        ), "Only supports all full attn mask for now."

        assert isinstance(
            self.attn_solver, (DistAttnSolver, DynamicAttnSolver)
        ), "Only supports either `DistAttnSolver` or `DynamicAttnSolver` for cross-attn by now."

        host_global_perm_merged_q_ranges = self.attn_solver.host_q_ranges_global
        # HACK: ref_xattn_q_ranges cannot be merged, so we hack it by setting is_self_merged=True,
        #       this way find_overlap_ranges won't merge ref_xattn_q_ranges
        host_global_perm_sorted_q_ranges = ref_xattn_q_ranges.find_overlap_ranges(
            host_global_perm_merged_q_ranges, is_self_merged=True
        )
        host_global_unperm_xattn_k_ranges = AttnRanges()
        for q_range in host_global_perm_sorted_q_ranges:
            is_found = False
            for i, ref_q_range in enumerate(ref_xattn_q_ranges):
                if q_range.is_subrange_of(ref_q_range):
                    host_global_unperm_xattn_k_ranges.append(ref_xattn_k_ranges[i])
                    is_found = True
            if not is_found:
                raise ValueError(
                    f"q_range: {q_range} is not in ref_q_ranges: {ref_xattn_q_ranges}"
                )

        if return_host_only:
            attn_arg = AttnArg(
                q_ranges=host_global_perm_sorted_q_ranges.make_ranges_local(
                    host_global_perm_sorted_q_ranges
                ),
                k_ranges=host_global_unperm_xattn_k_ranges,
                attn_type_map=[0] * len(host_global_perm_sorted_q_ranges),
            )
            return attn_arg

        cp_size = dist.get_world_size(self.cp_group)
        host_global_perm_sorted_q_ranges_per_rank: list[AttnRanges] = [None] * cp_size  # type: ignore[list-item]
        host_global_unperm_xattn_k_ranges_per_rank: list[AttnRanges] = [None] * cp_size  # type: ignore[list-item]

        dist.all_gather_object(
            host_global_perm_sorted_q_ranges_per_rank,
            host_global_perm_sorted_q_ranges,
            group=self.cp_group,
        )

        total_global_perm_sorted_q_ranges = AttnRanges.from_ranges(
            itertools.chain(*host_global_perm_sorted_q_ranges_per_rank)  # type: ignore[arg-type]
        )

        dist.all_gather_object(
            host_global_unperm_xattn_k_ranges_per_rank,
            host_global_unperm_xattn_k_ranges,
            group=self.cp_group,
        )

        total_global_unperm_xattn_k_ranges = AttnRanges.from_ranges(
            itertools.chain(*host_global_unperm_xattn_k_ranges_per_rank)  # type: ignore[arg-type]
        )

        attn_arg = AttnArg(
            q_ranges=total_global_perm_sorted_q_ranges,
            k_ranges=total_global_unperm_xattn_k_ranges,
            attn_type_map=[0] * len(total_global_perm_sorted_q_ranges),
        )
        return attn_arg

    def get_position_ids(self, attn_role: AttnRole = AttnRole.QUERY) -> torch.Tensor:
        """
        Get the position ids of local tensor to global tensor after dispatching.

        Args:
            attn_role (AttnRole): the role of the tensor to get position ids

        Returns:
            position_ids (torch.Tensor): postion_ids of local tensor to global tensor w.r.t. the attn_role.
        """

        if attn_role == AttnRole.QUERY:
            if self._q_position_ids is None:
                self._q_position_ids = self.dispatch_meta_q.position_ids
            return self._q_position_ids
        elif attn_role == AttnRole.KEY or attn_role == AttnRole.VALUE:
            if self._k_position_ids is None:
                self._k_position_ids = self.dispatch_meta_k.position_ids
            return self._k_position_ids
        else:
            raise ValueError(f"Invalid attn role: {attn_role}")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DistAttnRuntimeMgr):
            return False

        return (
            self.dispatch_meta_q,
            self.dispatch_meta_k,
            self.dist_attn_config,
            self.attn_solver,
            self.dist_attn_runtime,
            self.ref_q_ranges,
            self.ref_k_ranges,
            self.is_same_source,
            self.is_q_permutable,
            self.is_k_permutable,
        ) == (
            other.dispatch_meta_q,
            other.dispatch_meta_k,
            other.dist_attn_config,
            other.attn_solver,
            other.dist_attn_runtime,
            other.ref_q_ranges,
            other.ref_k_ranges,
            other.is_same_source,
            other.is_q_permutable,
            other.is_k_permutable,
        ) and is_same_process_group(
            self.cp_group, other.cp_group
        )


class DistAttnRuntimeDict(OrderedDict):
    """
    A fixed-length ordered dictionary to map DistAttnRuntimeKey to DistAttnRuntimeMgr
    which evicts the least recently used item (LRU policy) when capacity is exceeded
    """

    def __init__(self, max_size: int, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: DistAttnRuntimeKey, value: DistAttnRuntimeMgr):
        # If key exists, delete it first (to ensure it moves to end)
        if key in self:
            del self[key]
        # If at max capacity, remove the oldest item
        elif len(self) >= self.max_size:
            self.popitem(last=False)
        # Insert new key-value pair (automatically added to end)
        super().__setitem__(key, value)

    def get(self, key: DistAttnRuntimeKey, default=None) -> DistAttnRuntimeMgr | None:
        # Override get method to move accessed items to end (marking as recently used)
        if key in self:
            value = super().__getitem__(key)
            del self[key]
            super().__setitem__(key, value)
            return value
        return default

    def get_most_recent_key(self) -> DistAttnRuntimeKey | None:
        """
        Gets and returns the most recently added or accessed key.
        If the dictionary is empty, returns None.
        """
        if not self:
            return None

        return next(reversed(self.keys()))


def check_flag_comb() -> None:
    """Check some invalid flag combinations"""

    if magi_attention.comm.is_hierarchical_comm_enable():
        assert (  # TODO
            not magi_attention.comm.is_qo_comm_enable()
        ), "Hierarchical comm is not compatible with qo comm for now"

        assert (  # TODO
            not magi_attention.comm.is_native_grpcoll_enable()
        ), "Hierarchical comm is not compatible with native grpcoll for now"

    if magi_attention.comm.is_native_grpcoll_enable():
        assert (  # FIXME
            not magi_attention.is_deterministic_mode_enable()
        ), "Native grpcoll is not compatible with deterministic mode for now"

    if (
        magi_attention.is_fa4_backend_enable()
        and not magi_attention.is_sdpa_backend_enable()
    ):
        assert (  # TODO
            not magi_attention.is_deterministic_mode_enable()
        ), "FA4 backend is not compatible with deterministic mode for now"

        assert (  # TODO
            not magi_attention.comm.is_fwd_high_precision_reduce_enable()
            and not magi_attention.comm.is_bwd_high_precision_reduce_enable()
        ), "FA4 backend is not compatible with high-precision reduce for now"

        assert (  # TODO
            not magi_attention.comm.is_qo_comm_enable()
        ), "FA4 backend is not compatible with qo comm for now"


def init_dist_attn_runtime_key(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    total_seqlen_k: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    cp_mesh: DeviceMesh | None,
    dist_attn_config: DistAttnConfig,
    num_heads_q: int,
    num_heads_kv: int,
) -> DistAttnRuntimeKey:
    """Initialize DistAttnRuntimeKey"""

    # Check if flag combinations are valid
    check_flag_comb()

    return DistAttnRuntimeKey(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=tuple(attn_mask_type),
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        dist_attn_config=dist_attn_config,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        # auto set other flags that might influence the runtime behavior
        is_deterministic_mode_enable=magi_attention.is_deterministic_mode_enable(),
        is_hierarchical_comm_enable=magi_attention.comm.is_hierarchical_comm_enable(),
        is_qo_comm_enable=magi_attention.comm.is_qo_comm_enable(),
        is_native_grpcoll_enable=magi_attention.comm.is_native_grpcoll_enable(),
        is_flatten_head_groups_enable=magi_attention.is_flatten_head_groups_enable(),
        is_sdpa_backend_enable=magi_attention.is_sdpa_backend_enable(),
        is_fa4_backend_enable=magi_attention.is_fa4_backend_enable(),
    )


def init_grpcoll_buffer_mgr(
    comm_meta: CommMeta,
    calc_meta: CalcMeta,
    attn_solver: BaseDistAttnSolver,
    grpcoll_config: GrpCollConfig,
    cp_group: dist.ProcessGroup,
) -> None:
    if magi_attention.comm.is_native_grpcoll_enable():
        grpcoll_buffer_mgr.initialize(
            group=cp_group,
            config=grpcoll_config,
        )


def init_dist_attn_runtime_mgr(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: list[AttnMaskType],
    total_seqlen_q: int,
    total_seqlen_k: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    cp_mesh: DeviceMesh | None = None,
    num_heads_q: int = 1,
    num_heads_kv: int = 1,
    ref_dispatch_meta_q: DispatchMeta | None = None,
    ref_dispatch_meta_k: DispatchMeta | None = None,
) -> DistAttnRuntimeMgr:
    """

    Args:
        q_ranges (AttnRanges): the global query ranges
        k_ranges (AttnRanges): the global key ranges
        attn_mask_type (list[AttnMaskType]): the global attn mask type list

        total_seqlen_q (int): the total seqlen of query
        total_seqlen_k (int): the total seqlen of key

        chunk_size (int): chunk size to chunk the permutable tensor

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable
        NOTE: e.g.
                1. for decoder-only transformer like gpt, it applies 'self-attn' as follows:
                    a) is_same_source is True
                    b) both q and k are permutable, as long as they are permuted in the same way.
                2. for encoder-decoder transformer like t5, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is permutable but k is not
                3. for multi-modal transformer with external encoders, it applies 'cross-attn' as follows:
                    a) is_same_source is False
                    b) q is unpermutable cuz of self-attn, but k is permutable even in a different way

        dist_attn_config (DistAttnConfig): dist attn config

        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.

        num_heads_q (int): number of heads of query. Default: 1
        num_heads_kv (int): number of heads of key/value. Default: 1

    Returns:
        DistAttnRuntimeMgr: dist attn runtime mgr

    Example::
        >>> dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type=[AttnMaskType.FULL, AttnMaskType.CAUSAL],
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     chunk_size=512,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     is_same_source=True,
        ...     is_q_permutable=True,
        ...     is_k_permutable=True,
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=OverlapAlgType.UNIFORM,
        ...         ),
        ...     ),
        ... )
        >>> # Dispatch global query tensor to local query tensor
        >>> local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        >>> # Dispatch global key tensor to local key tensor
        >>> local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        >>> # Dispatch global value tensor to local value tensor
        >>> local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)
        >>> # Calculate local attention result
        >>> local_out, meta = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)
        >>> # Gather local attention results to global result
        >>> total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
    """

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    # make dispatch meta
    # to determine which rank should hold which chunks of seqlen
    dispatch_config: DispatchConfig = dist_attn_config.dispatch_config
    if ref_dispatch_meta_q is None or ref_dispatch_meta_k is None:
        # NOTE: in final, the dispatch meta is NOT supposed to contain any meta info about the mask
        # however, since in most of the distributed attention scenarios, the mask is static through the whole training pass
        # we can take advantage of this information to offer a better dispatch solution if permutable
        # so as to help reducing communication overhead while keep computation load-balance
        # therefore here, we will also pass the actual or initial mask info to the dispatch solver
        # as the arguments for some dispatch algorithms
        (
            dispatch_meta_q,
            dispatch_meta_k,
        ) = make_dispatch_meta_from_qk_ranges(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            dispatch_config=dispatch_config,
            is_same_source=is_same_source,
            is_q_permutable=is_q_permutable,
            is_k_permutable=is_k_permutable,
        )
    else:
        dispatch_meta_q = ref_dispatch_meta_q
        dispatch_meta_k = ref_dispatch_meta_k

    # make comm meta and calc meta
    # to organize the dist-attn calculation and communication
    overlap_config: OverlapConfig = dist_attn_config.overlap_config
    comm_meta, calc_meta, attn_solver = make_attn_meta_from_dispatch_meta(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        cp_group=cp_group,
        overlap_config=overlap_config,
        cp_mesh=cp_mesh,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
    )

    # init grpcoll buffer manager for native grpcoll kernels
    grpcoll_config: GrpCollConfig = dist_attn_config.grpcoll_config
    init_grpcoll_buffer_mgr(
        comm_meta=comm_meta,
        calc_meta=calc_meta,
        attn_solver=attn_solver,
        grpcoll_config=grpcoll_config,
        cp_group=cp_group,
    )

    # init dist attn runtime
    dist_attn_runtime = DistAttnRuntime(
        comm_meta=comm_meta,
        calc_meta=calc_meta,
        cp_group_gc=cp_group,
        cp_group_gr=cp_group,  # TODO: support interface to set distinct cp group for group-reduce
    )

    # init dist attn runtime mgr
    dist_attn_runtime_mgr = DistAttnRuntimeMgr(
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        dist_attn_config=dist_attn_config,
        attn_solver=attn_solver,
        dist_attn_runtime=dist_attn_runtime,
        cp_group=cp_group,
        ref_q_ranges=q_ranges,
        ref_k_ranges=k_ranges,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
    )

    return dist_attn_runtime_mgr
