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

import itertools

import torch
import torch.distributed as dist

from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType, AttnRole
from magi_attention.config import DistAttnConfig
from magi_attention.functional.dispatch import dispatch_func, undispatch_func
from magi_attention.functional.dist_attn import DistFlashAttnRuntime, dist_attn_func
from magi_attention.meta import (
    calc_attn_meta_from_dispatch_meta,
    calc_dispatch_meta_from_qk_ranges,
)
from magi_attention.meta.collection import DispatchMeta
from magi_attention.meta.collection.calc_meta import AttnArg
from magi_attention.meta.solver.dist_attn_solver import DistAttnSolver
from magi_attention.utils import is_list_value_all, wrap_to_list


# @dataclass(frozen=True)
class DistAttnRuntimeKey:
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        pad_size: int,
        head_dim: int,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        attn_mask_type: list[AttnMaskType],
        total_seqlen_q: int,
        total_seqlen_k: int,
        dist_attn_config: DistAttnConfig,
    ):
        self.cp_group = cp_group
        self.pad_size = pad_size
        self.head_dim = head_dim
        self.q_ranges = q_ranges
        self.k_ranges = k_ranges
        self.attn_mask_type = attn_mask_type
        self.total_seqlen_q = total_seqlen_q
        self.total_seqlen_k = total_seqlen_k
        self.dist_attn_config = dist_attn_config

    def __hash__(self):
        mask_tuple = tuple(self.attn_mask_type)

        return hash(
            (
                self.cp_group,
                self.pad_size,
                self.head_dim,
                self.q_ranges,
                self.k_ranges,
                mask_tuple,
                self.total_seqlen_q,
                self.total_seqlen_k,
                self.dist_attn_config,
            )
        )


class DistAttnRuntimeMgr:
    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        q_dispatch_meta: DispatchMeta,
        k_dispatch_meta: DispatchMeta,
        chunk_size: int,
        dist_attn_config: DistAttnConfig,
        attn_solver: DistAttnSolver,
        dist_attn_runtime: DistFlashAttnRuntime,
        *,
        ref_q_ranges: AttnRanges,
        ref_k_ranges: AttnRanges,
        is_same_source: bool,
        is_q_permutable: bool,
        is_k_permutable: bool,
    ):
        self.cp_group = cp_group
        self.q_dispatch_meta = q_dispatch_meta
        self.k_dispatch_meta = k_dispatch_meta
        self.chunk_size = chunk_size
        self.dist_attn_config = dist_attn_config
        self.attn_solver = attn_solver
        self.dist_attn_runtime = dist_attn_runtime

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
            meta=self.q_dispatch_meta,
        )
        return q_or_o

    def dispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = dispatch_func(
            x_global=k_or_v,
            group=self.cp_group,
            meta=self.k_dispatch_meta,
        )
        return k_or_v

    def undispatch_qo(self, q_or_o: torch.Tensor) -> torch.Tensor:
        q_or_o = undispatch_func(
            x_local=q_or_o,
            group=self.cp_group,
            meta=self.q_dispatch_meta,
        )
        return q_or_o

    def undispatch_kv(self, k_or_v: torch.Tensor) -> torch.Tensor:
        k_or_v = undispatch_func(
            x_local=k_or_v,
            group=self.cp_group,
            meta=self.k_dispatch_meta,
        )
        return k_or_v

    def calc_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return dist_attn_func(q, k, v, self.dist_attn_runtime)

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
                is_causal_mapping=[False] * len(host_global_perm_sorted_q_ranges),
                shard_seqlen_q=host_global_perm_sorted_q_ranges.total_seqlen,
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
            is_causal_mapping=[False] * len(total_global_perm_sorted_q_ranges),
            shard_seqlen_q=total_global_perm_sorted_q_ranges.total_seqlen,
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
                self._q_position_ids = self.q_dispatch_meta.position_ids
            return self._q_position_ids
        elif attn_role == AttnRole.KEY or attn_role == AttnRole.VALUE:
            if self._k_position_ids is None:
                self._k_position_ids = self.k_dispatch_meta.position_ids
            return self._k_position_ids
        else:
            raise ValueError(f"Invalid attn role: {attn_role}")


def init_dist_attn_runtime_mgr(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: AttnMaskType | list[AttnMaskType],
    total_seqlen_q: int,
    total_seqlen_k: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup,
    is_same_source: bool,
    is_q_permutable: bool,
    is_k_permutable: bool,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    cp_intra_group: dist.ProcessGroup | None = None,
    cp_inter_group: dist.ProcessGroup | None = None,
) -> DistAttnRuntimeMgr:
    """

    Args:
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (AttnMaskType | list[AttnMaskType]): attn mask type (list)

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

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

    Returns:
        DistAttnRuntimeMgr: dist attn runtime mgr

    Example::
        >>> dist_attn_runtime_mgr = init_dist_attn_runtime_mgr(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type=AttnMaskType.FULL,
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
        >>> local_out = dist_attn_runtime_mgr.calc_attn(local_q, local_k, local_v)
        >>> # Gather local attention results to global result
        >>> total_out = dist_attn_runtime_mgr.undispatch_qo(local_out)
    """

    cp_size = dist.get_world_size(cp_group)
    cp_rank = dist.get_rank(cp_group)

    q_dispatch_meta, k_dispatch_meta, attn_buckets = calc_dispatch_meta_from_qk_ranges(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        chunk_size=chunk_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        dispatch_config=dist_attn_config.dispatch_config,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
        high_bandwith_domain_size=dist_attn_config.high_bandwith_domain_size,
    )

    comm_meta, attn_calc_meta, attn_solver = calc_attn_meta_from_dispatch_meta(
        dispatch_meta_q=q_dispatch_meta,
        dispatch_meta_k=k_dispatch_meta,
        bucket_per_rank=attn_buckets,
        cp_group=cp_group,
        high_bandwith_domain_size=dist_attn_config.high_bandwith_domain_size,
        overlap_config=dist_attn_config.overlap_config,
        cp_intra_group=cp_intra_group,
        cp_inter_group=cp_inter_group,
    )

    dist_attn_runtime = DistFlashAttnRuntime(
        comm_meta=comm_meta,
        calc_meta=attn_calc_meta,
        cp_group_kv=cp_group,
        cp_group_dkv=cp_group,  # TODO: support interface to set distinct cp group for dkv
        deterministic=dist_attn_config.deterministic,
    )

    return DistAttnRuntimeMgr(
        cp_group,
        q_dispatch_meta,
        k_dispatch_meta,
        chunk_size,
        dist_attn_config,
        attn_solver,
        dist_attn_runtime,
        ref_q_ranges=q_ranges,
        ref_k_ranges=k_ranges,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
    )
