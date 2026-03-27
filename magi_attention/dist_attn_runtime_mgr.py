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
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention

logger = logging.getLogger(__name__)
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
from magi_attention.functional.roll import roll_p2p as roll_func
from magi_attention.functional.roll import roll_simple_p2p as roll_simple_func
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
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    pad_size: int
    chunk_size: int
    cp_group: dist.ProcessGroup
    cp_mesh: DeviceMesh | None
    dist_attn_config: DistAttnConfig
    uneven_shard: bool

    # flags that might influence the runtime behavior
    is_deterministic_mode_enable: bool
    is_hierarchical_comm_enable: bool
    is_qo_comm_enable: bool
    is_native_grpcoll_enable: bool
    is_flatten_head_groups_enable: bool
    is_sdpa_backend_enable: bool
    is_fa4_backend_enable: bool
    is_auto_range_merge_enable: bool

    def __hash__(self) -> int:
        try:
            return self.__dict__["_cached_hash"]
        except KeyError:
            h = hash(
                (
                    self.q_ranges,
                    self.k_ranges,
                    self.attn_mask_type,
                    self.total_seqlen_q,
                    self.total_seqlen_k,
                    self.num_heads_q,
                    self.num_heads_kv,
                    self.head_dim,
                    self.pad_size,
                    self.chunk_size,
                    self.cp_group,
                    self.cp_mesh,
                    self.dist_attn_config,
                    self.uneven_shard,
                    self.is_deterministic_mode_enable,
                    self.is_hierarchical_comm_enable,
                    self.is_qo_comm_enable,
                    self.is_native_grpcoll_enable,
                    self.is_flatten_head_groups_enable,
                    self.is_sdpa_backend_enable,
                    self.is_fa4_backend_enable,
                    self.is_auto_range_merge_enable,
                )
            )
            object.__setattr__(self, "_cached_hash", h)
            return h


class DistAttnRuntimeMgr:
    def __init__(
        self,
        dispatch_meta_q: DispatchMeta,
        dispatch_meta_k: DispatchMeta,
        dist_attn_config: DistAttnConfig,
        attn_solver: BaseDistAttnSolver,
        dist_attn_runtime: DistAttnRuntime,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        cp_group: dist.ProcessGroup,
        *,
        ref_q_ranges: AttnRanges,
        ref_k_ranges: AttnRanges,
        is_same_source: bool,
        is_q_permutable: bool,
        is_k_permutable: bool,
    ):
        self.dispatch_meta_q = dispatch_meta_q
        self.dispatch_meta_k = dispatch_meta_k
        self.dist_attn_config = dist_attn_config
        self.attn_solver = attn_solver
        self.dist_attn_runtime = dist_attn_runtime

        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim

        self.cp_group = cp_group

        self.ref_q_ranges = ref_q_ranges
        self.ref_k_ranges = ref_k_ranges
        self.is_same_source = is_same_source
        self.is_q_permutable = is_q_permutable
        self.is_k_permutable = is_k_permutable

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

    def roll(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """Cyclically roll a dispatched local tensor via P2P communication.

        This avoids the full ``undispatch`` -> ``torch.roll`` -> ``dispatch``
        round-trip, keeping memory usage at O(N/P) instead of O(N).

        Args:
            x (torch.Tensor): the dispatched local tensor on this rank.
            shift (int): number of positions to roll (positive = shift right,
                wraps cyclically).
            dim (int): the dimension to roll along.

        Returns:
            torch.Tensor: rolled local tensor, same shape as *x*.
        """
        return roll_func(
            x_local=x,
            shift=shift,
            meta=self.dispatch_meta_q,
            group=self.cp_group,
            seq_dim=dim,
        )

    def roll_simple(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        """Cyclically roll a dispatched local tensor via simple (non-batched) P2P.

        Functionally identical to :meth:`roll` but uses plain ``dist.isend``
        / ``dist.irecv`` instead of ``dist.batch_isend_irecv``.

        Args:
            x (torch.Tensor): the dispatched local tensor on this rank.
            shift (int): number of positions to roll (positive = shift right,
                wraps cyclically).
            dim (int): the dimension to roll along.

        Returns:
            torch.Tensor: rolled local tensor, same shape as *x*.
        """
        return roll_simple_func(
            x_local=x,
            shift=shift,
            meta=self.dispatch_meta_q,
            group=self.cp_group,
            seq_dim=dim,
        )

    def calc_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        softcap: float = 0.0,
        return_max_logits: bool = False,
    ) -> tuple[torch.Tensor, AttnForwardMeta]:
        return dist_attn_func(
            q=q,
            k=k,
            v=v,
            dist_attn_runtime=self.dist_attn_runtime,
            sink=sink,
            softmax_scale=softmax_scale,
            softcap=softcap,
            return_max_logits=return_max_logits,
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
        if self.max_size <= 0:
            return
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
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    cp_mesh: DeviceMesh | None,
    dist_attn_config: DistAttnConfig,
    uneven_shard: bool = False,
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
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        dist_attn_config=dist_attn_config,
        uneven_shard=uneven_shard,
        # auto set other flags that might influence the runtime behavior
        is_deterministic_mode_enable=magi_attention.is_deterministic_mode_enable(),
        is_hierarchical_comm_enable=magi_attention.comm.is_hierarchical_comm_enable(),
        is_qo_comm_enable=magi_attention.comm.is_qo_comm_enable(),
        is_native_grpcoll_enable=magi_attention.comm.is_native_grpcoll_enable(),
        is_flatten_head_groups_enable=magi_attention.is_flatten_head_groups_enable(),
        is_sdpa_backend_enable=magi_attention.is_sdpa_backend_enable(),
        is_fa4_backend_enable=magi_attention.is_fa4_backend_enable(),
        is_auto_range_merge_enable=magi_attention.is_auto_range_merge_enable(),
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
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    cp_mesh: DeviceMesh | None = None,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
    ref_dispatch_meta_q: DispatchMeta | None = None,
    ref_dispatch_meta_k: DispatchMeta | None = None,
    uneven_shard: bool = False,
) -> DistAttnRuntimeMgr:
    """

    Args:
        q_ranges (AttnRanges): the global query ranges.
        k_ranges (AttnRanges): the global key ranges.
        attn_mask_type (list[AttnMaskType]): the global attn mask type list.

        total_seqlen_q (int): the total seqlen of query.
        total_seqlen_k (int): the total seqlen of key.

        num_heads_q (int): number of heads of query.
        num_heads_kv (int): number of heads of key/value.
        head_dim (int): dimension of each head.

        chunk_size (int): chunk size to chunk the permutable tensor.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.

        is_same_source (bool): is query tensor and key tensor share the same source.
            Default to ``True``.
        is_q_permutable (bool): is query tensor permutable.
            Default to ``True``.
        is_k_permutable (bool): is key tensor permutable.
            Default to ``True``.

        NOTE:
            1. for decoder-only transformer like gpt, it applies 'self-attn' as follows:
                a) is_same_source is True
                b) both q and k are permutable, as long as they are permuted in the same way.
            2. for encoder-decoder transformer like t5, it applies 'cross-attn' as follows:
                a) is_same_source is False
                b) q is permutable but k is not
            3. for multi-modal transformer with external encoders, it applies 'cross-attn' as follows:
                a) is_same_source is False
                b) q is unpermutable cuz of self-attn, but k is permutable even in a different way

        dist_attn_config (DistAttnConfig): dist attn config.

        uneven_shard (bool): when True, ``total_seqlen`` need not be divisible
            by ``chunk_size * cp_size``.  Passed through to
            ``make_dispatch_meta_from_qk_ranges`` to compute
            ``chunk_actual_sizes`` and ``split_sizes``.  Default to ``False``.

    Returns:
        DistAttnRuntimeMgr: dist attn runtime manager.

    Example::
        >>> # Step1. initialize the dist attn runtime manager
        >>> dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type=[AttnMaskType.FULL, AttnMaskType.CAUSAL],
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     chunk_size=512,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
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
        ...     is_same_source=True,
        ...     is_q_permutable=True,
        ...     is_k_permutable=True,
        ... )
        >>>
        >>> # Step2. dispatch global query tensor to local query tensor
        >>> local_q = dist_attn_runtime_mgr.dispatch_qo(total_q)
        >>>
        >>> # Step3. dispatch global key/value tensor to local key/value tensor
        >>> local_k = dist_attn_runtime_mgr.dispatch_kv(total_k)
        >>> local_v = dist_attn_runtime_mgr.dispatch_kv(total_v)
        >>>
        >>> # Step4. calculate distributed attention
        >>> local_out, meta = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)
        >>>
        >>> # Step5. undispatch local attention output to the global one if needed
        >>> total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
    """

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    logger.info(
        "============================================================\n"
        "  init_dist_attn_runtime_mgr START\n"
        "============================================================\n"
        "[Input Arguments]\n"
        "  q_ranges              : %s\n"
        "  k_ranges              : %s\n"
        "  attn_mask_type        : %s\n"
        "  total_seqlen_q        : %d\n"
        "  total_seqlen_k        : %d\n"
        "  num_heads_q           : %d\n"
        "  num_heads_kv          : %d\n"
        "  head_dim              : %d\n"
        "  chunk_size            : %d\n"
        "  cp_size               : %d\n"
        "  cp_rank               : %d\n"
        "  cp_mesh               : %s\n"
        "  is_same_source        : %s\n"
        "  is_q_permutable       : %s\n"
        "  is_k_permutable       : %s\n"
        "  uneven_shard          : %s\n"
        "  ref_dispatch_meta_q   : %s\n"
        "  ref_dispatch_meta_k   : %s\n"
        "[DistAttnConfig]\n"
        "  dispatch_config       : %r\n"
        "  overlap_config        : %r\n"
        "  grpcoll_config        : %r\n"
        "[Global Flags]\n"
        "  deterministic_mode    : %s\n"
        "  hierarchical_comm     : %s\n"
        "  qo_comm               : %s\n"
        "  native_grpcoll        : %s\n"
        "  flatten_head_groups   : %s\n"
        "  sdpa_backend          : %s\n"
        "  fa4_backend           : %s\n"
        "  auto_range_merge      : %s",
        q_ranges,
        k_ranges,
        attn_mask_type,
        total_seqlen_q,
        total_seqlen_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        chunk_size,
        cp_size,
        cp_rank,
        cp_mesh,
        is_same_source,
        is_q_permutable,
        is_k_permutable,
        uneven_shard,
        "provided" if ref_dispatch_meta_q is not None else "None (will compute)",
        "provided" if ref_dispatch_meta_k is not None else "None (will compute)",
        dist_attn_config.dispatch_config,
        dist_attn_config.overlap_config,
        dist_attn_config.grpcoll_config,
        magi_attention.is_deterministic_mode_enable(),
        magi_attention.comm.is_hierarchical_comm_enable(),
        magi_attention.comm.is_qo_comm_enable(),
        magi_attention.comm.is_native_grpcoll_enable(),
        magi_attention.is_flatten_head_groups_enable(),
        magi_attention.is_sdpa_backend_enable(),
        magi_attention.is_fa4_backend_enable(),
        magi_attention.is_auto_range_merge_enable(),
    )

    # Make dispatch meta
    # to determine which rank should hold which chunks of seqlen
    dispatch_config: DispatchConfig = dist_attn_config.dispatch_config
    if ref_dispatch_meta_q is None or ref_dispatch_meta_k is None:
        # NOTE: in final, the dispatch meta is NOT supposed to contain any meta info about the mask
        # however, since in most of the distributed attention scenarios, the mask is static through the whole training pass
        # we can take advantage of this information to offer a better dispatch solution if permutable
        # so as to help reducing communication overhead while keep computation load-balance
        # therefore here, we will also pass the actual or initial mask info to the dispatch solver
        # as the arguments for some dispatch algorithms
        logger.info(
            "[Dispatch] Computing dispatch meta from qk_ranges (ref metas not provided)..."
        )
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
            uneven_shard=uneven_shard,
        )
    else:
        logger.info("[Dispatch] Using provided ref_dispatch_meta_q and ref_dispatch_meta_k.")
        dispatch_meta_q = ref_dispatch_meta_q
        dispatch_meta_k = ref_dispatch_meta_k

    logger.info(
        "[Dispatch Meta Q]\n%r\n"
        "[Dispatch Meta K]\n%r\n"
        "[Dispatch Meta Q Details]\n"
        "  attn_role             : %s\n"
        "  attn_type             : %s\n"
        "  total_seqlen          : %d\n"
        "  shard_seqlen          : %d\n"
        "  max_valid_ids         : %d\n"
        "  chunk_size            : %d\n"
        "  num_chunks            : %d\n"
        "  cp_rank               : %d\n"
        "  cp_size               : %d\n"
        "  partitions            : %s\n"
        "  partitions_perm_idxs  : %s\n"
        "  partitions_unperm_idxs: %s\n"
        "  chunk_actual_sizes    : %s\n"
        "  split_sizes           : %s\n"
        "[Dispatch Meta K Details]\n"
        "  attn_role             : %s\n"
        "  attn_type             : %s\n"
        "  total_seqlen          : %d\n"
        "  shard_seqlen          : %d\n"
        "  max_valid_ids         : %d\n"
        "  chunk_size            : %d\n"
        "  num_chunks            : %d\n"
        "  cp_rank               : %d\n"
        "  cp_size               : %d\n"
        "  partitions            : %s\n"
        "  partitions_perm_idxs  : %s\n"
        "  partitions_unperm_idxs: %s\n"
        "  chunk_actual_sizes    : %s\n"
        "  split_sizes           : %s",
        dispatch_meta_q,
        dispatch_meta_k,
        dispatch_meta_q.attn_role,
        dispatch_meta_q.attn_type,
        dispatch_meta_q.total_seqlen,
        dispatch_meta_q.shard_seqlen,
        dispatch_meta_q.max_valid_ids,
        dispatch_meta_q.chunk_size,
        dispatch_meta_q.num_chunks,
        dispatch_meta_q.cp_rank,
        dispatch_meta_q.cp_size,
        dispatch_meta_q.partitions,
        dispatch_meta_q.partitions_perm_idxs,
        dispatch_meta_q.partitions_unperm_idxs,
        dispatch_meta_q.chunk_actual_sizes,
        dispatch_meta_q.split_sizes,
        dispatch_meta_k.attn_role,
        dispatch_meta_k.attn_type,
        dispatch_meta_k.total_seqlen,
        dispatch_meta_k.shard_seqlen,
        dispatch_meta_k.max_valid_ids,
        dispatch_meta_k.chunk_size,
        dispatch_meta_k.num_chunks,
        dispatch_meta_k.cp_rank,
        dispatch_meta_k.cp_size,
        dispatch_meta_k.partitions,
        dispatch_meta_k.partitions_perm_idxs,
        dispatch_meta_k.partitions_unperm_idxs,
        dispatch_meta_k.chunk_actual_sizes,
        dispatch_meta_k.split_sizes,
    )

    # Make comm meta and calc meta
    # to organize the dist-attn calculation and communication
    overlap_config: OverlapConfig = dist_attn_config.overlap_config
    logger.info("[Attn Meta] Building comm_meta, calc_meta, and attn_solver from dispatch meta...")
    comm_meta, calc_meta, attn_solver = make_attn_meta_from_dispatch_meta(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        overlap_config=overlap_config,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
    )

    logger.info(
        "[CommMeta]\n%r\n"
        "[CommMeta Details]\n"
        "  overlap_degree                  : %d\n"
        "  num_heads_q                     : %d\n"
        "  num_heads_kv                    : %d\n"
        "  num_heads_per_group             : %d\n"
        "  head_dim                        : %d\n"
        "  num_remote_kv_tokens_per_stage  : %s\n"
        "  num_remote_qo_tokens_per_stage  : %s\n"
        "  kv_group_collective_args_list   : [%d entries]\n"
        "  qo_group_collective_args_list   : [%d entries]",
        comm_meta,
        comm_meta.overlap_degree,
        comm_meta.num_heads_q,
        comm_meta.num_heads_kv,
        comm_meta.num_heads_per_group,
        comm_meta.head_dim,
        comm_meta.num_remote_kv_tokens_per_stage,
        comm_meta.num_remote_qo_tokens_per_stage,
        len(comm_meta.kv_group_collective_args_list),
        len(comm_meta.qo_group_collective_args_list),
    )

    logger.info(
        "[CalcMeta]\n%r\n"
        "[CalcMeta Details]\n"
        "  overlap_degree                  : %d\n"
        "  seqlen_q_shard                  : %d\n"
        "  seqlen_k_local                  : %d\n"
        "  seqlen_k_per_remote_stage       : %s\n"
        "  local_attn_arg                  : %r\n"
        "  remote_attn_args_list           : [%d entries]",
        calc_meta,
        calc_meta.overlap_degree,
        calc_meta.seqlen_q_shard,
        calc_meta.seqlen_k_local,
        calc_meta.seqlen_k_per_remote_stage,
        calc_meta.local_attn_arg,
        len(calc_meta.remote_attn_args_list),
    )

    for i, remote_arg in enumerate(calc_meta.remote_attn_args_list):
        logger.info(
            "[CalcMeta] remote_attn_args_list[%d]:\n%r",
            i,
            remote_arg,
        )

    logger.info(
        "[AttnSolver] type=%s\n%r",
        type(attn_solver).__name__,
        attn_solver,
    )

    # Init grpcoll buffer manager for native grpcoll
    grpcoll_config: GrpCollConfig = dist_attn_config.grpcoll_config
    logger.info(
        "[GrpColl] native_grpcoll_enabled=%s, grpcoll_config=%r",
        magi_attention.comm.is_native_grpcoll_enable(),
        grpcoll_config,
    )
    init_grpcoll_buffer_mgr(
        comm_meta=comm_meta,
        calc_meta=calc_meta,
        attn_solver=attn_solver,
        grpcoll_config=grpcoll_config,
        cp_group=cp_group,
    )

    # Init dist attn runtime
    logger.info("[DistAttnRuntime] Initializing DistAttnRuntime...")
    dist_attn_runtime = DistAttnRuntime(
        comm_meta=comm_meta,
        calc_meta=calc_meta,
        cp_group_gc=cp_group,
        cp_group_gr=cp_group,  # TODO: support interface to set distinct cp group for group-reduce
    )

    # Init dist attn runtime mgr
    dist_attn_runtime_mgr = DistAttnRuntimeMgr(
        dispatch_meta_q=dispatch_meta_q,
        dispatch_meta_k=dispatch_meta_k,
        dist_attn_config=dist_attn_config,
        attn_solver=attn_solver,
        dist_attn_runtime=dist_attn_runtime,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        cp_group=cp_group,
        ref_q_ranges=q_ranges,
        ref_k_ranges=k_ranges,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
    )

    logger.info(
        "[DistAttnRuntimeMgr] Initialized successfully.\n"
        "  dispatch_meta_q       : partitions=%s, shard_seqlen=%d\n"
        "  dispatch_meta_k       : partitions=%s, shard_seqlen=%d\n"
        "  num_heads_q           : %d\n"
        "  num_heads_kv          : %d\n"
        "  head_dim              : %d\n"
        "  ref_q_ranges          : %s\n"
        "  ref_k_ranges          : %s\n"
        "  is_same_source        : %s\n"
        "  is_q_permutable       : %s\n"
        "  is_k_permutable       : %s\n"
        "============================================================\n"
        "  init_dist_attn_runtime_mgr END (rank=%d/%d)\n"
        "============================================================",
        dispatch_meta_q.partitions,
        dispatch_meta_q.shard_seqlen,
        dispatch_meta_k.partitions,
        dispatch_meta_k.shard_seqlen,
        num_heads_q,
        num_heads_kv,
        head_dim,
        q_ranges,
        k_ranges,
        is_same_source,
        is_q_permutable,
        is_k_permutable,
        cp_rank,
        cp_size,
    )

    return dist_attn_runtime_mgr
