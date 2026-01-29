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

from typing import Sequence, TypeAlias

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from typing_extensions import deprecated

import magi_attention
from magi_attention.common import AttnForwardMeta, AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DistAttnConfig
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeDict,
    DistAttnRuntimeKey,
    init_dist_attn_runtime_key,
    init_dist_attn_runtime_mgr,
)
from magi_attention.utils import wrap_to_list
from magi_attention.utils._utils import is_list_type_all

from .functools import (
    apply_padding,
    infer_attn_mask_from_cu_seqlens,
    pad_at_dim,
    unpad_at_dim,
)


def _get_cp_group_key(cp_group: dist.ProcessGroup) -> tuple:
    """Get a hashable key for a cp_group based on its ranks.

    This is used to create per-cp_group cache to avoid LRU eviction
    inconsistency across different ranks in the same cp_group.
    """
    if cp_group is None:
        return (0,)  # fallback for non-distributed case

    # Get global ranks for this group
    try:
        global_ranks = dist.get_process_group_ranks(cp_group)
        return tuple(sorted(global_ranks))
    except Exception:
        # Fallback: use group size and current rank
        return (cp_group.size(), dist.get_rank(cp_group))


class DistAttnRuntimeDictManager:
    """Manager for per-cp_group DistAttnRuntimeDict caches.

    Each cp_group has its own cache to avoid LRU eviction inconsistency
    across different ranks in the same cp_group. This prevents deadlocks
    where one rank has a cache hit while another has a cache miss,
    leading to asymmetric all_gather_object calls.
    """

    def __init__(self, max_size_per_group: int):
        self.max_size_per_group = max_size_per_group
        self._caches: dict[tuple, DistAttnRuntimeDict] = {}

    def _get_or_create_cp_group_cache(
        self, cp_group: dist.ProcessGroup
    ) -> DistAttnRuntimeDict:
        """Get or create the cache for a specific cp_group."""
        group_key = _get_cp_group_key(cp_group)
        if group_key not in self._caches:
            self._caches[group_key] = DistAttnRuntimeDict(
                max_size=self.max_size_per_group
            )
        return self._caches[group_key]

    def get(self, key: DistAttnRuntimeKey, default=None):
        """Get a value from the cache for the key's cp_group."""
        cache = self._get_or_create_cp_group_cache(key.cp_group)
        return cache.get(key, default)

    def __contains__(self, key: DistAttnRuntimeKey) -> bool:
        """Check if key exists in the cache for the key's cp_group."""
        cache = self._get_or_create_cp_group_cache(key.cp_group)
        return key in cache

    def __setitem__(self, key: DistAttnRuntimeKey, value):
        """Set a value in the cache for the key's cp_group."""
        cache = self._get_or_create_cp_group_cache(key.cp_group)
        cache[key] = value

    def __getitem__(self, key: DistAttnRuntimeKey):
        """Get a value from the cache for the key's cp_group."""
        cache = self._get_or_create_cp_group_cache(key.cp_group)
        return cache[key]

    def keys(self, cp_group: dist.ProcessGroup = None):
        """Get keys from a specific cp_group's cache or all caches."""
        if cp_group is not None:
            cache = self._get_or_create_cp_group_cache(cp_group)
            return cache.keys()
        # Return all keys from all caches
        all_keys: list = []
        for cache in self._caches.values():
            all_keys.extend(cache.keys())
        return all_keys

    def get_most_recent_key(
        self, cp_group: dist.ProcessGroup
    ) -> DistAttnRuntimeKey | None:
        """Get the most recently inserted key from a specific cp_group's cache."""
        if cp_group is None:
            raise ValueError("cp_group must be specified for get_most_recent_key")
        cache = self._get_or_create_cp_group_cache(cp_group)
        return cache.get_most_recent_key()


# per-cp_group magi-key cache manager
dist_attn_runtime_dict_mgr = DistAttnRuntimeDictManager(
    max_size_per_group=magi_attention.dist_attn_runtime_dict_size()
)


GeneralAttnMaskType: TypeAlias = str | AttnMaskType | Sequence[str | AttnMaskType]


def magi_attn_varlen_key(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
) -> DistAttnRuntimeKey:
    """This is a flash-attn-varlen like interface,
    to generate ``q_ranges``, ``k_ranges`` and ``attn_mask_type``
    from ``cu_seqlens_q``, ``cu_seqlens_k``, ``causal`` and ``window_size``,
    calculate ``dist_attn_runtime_key`` and generate the corr. inner ``dist_attn_runtime_mgr``.

    Args:
        cu_seqlens_q (torch.Tensor): the cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): the cumulative sequence lengths for keys.

        num_heads_q (int): the number of heads for query.
        num_heads_kv (int): the number of heads for key/value.
        head_dim (int): the dimension of each attention head.

        pad_size (int): the size to pad the global input tensor along sequence dim,
            due to the constraint that the sequence length need to be divisable by ``chunk_size * cp_size``.
        chunk_size (int): the size to chunk the global input tensor along the seqlen dim
            for later sharding and dispatching among the cp ranks
            as a granularity factor of computational load-balance.

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        causal (bool, optional): if ``True``, all mask types are set to ``CAUSAL``,
            otherwise, determine the mask types by ``window_size``. Defaults to ``False``.
        window_size (tuple[int, int], optional): window_size of sliding window mask
            which represents ``[window_size_left, window_size_right]``. The parameter is effective only
            when ``causal`` is ``False``; when ``causal`` is ``True``, it is required to be ``(-1, -1)``.
            Defaults to be ``(-1, -1)``.

        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        DistAttnRuntimeKey: the key points to the inner DistAttnRuntimeMgr.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_varlen_key, dispatch, undispatch, calc_attn
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>>
        >>> # Step1. generate a dist_attn_runtime_key to store and indicate the inner meta info
        >>> dist_attn_runtime_key = magi_attn_varlen_key(
        ...     cu_seqlen_q=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     cu_seqlen_k=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512), # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     causal=False,
        ...     window_size=(-1, -1),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. dispatch the global tensors to local tensors
        >>> local_x, local_label, local_rope = [
        ...     dispatch(tensor, dist_attn_runtime_key)
        ...     for tensor in [total_x, total_label, total_rope]
        ... ]
        >>>
        >>> # Step3. apply QKV projection on local tensors
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step4. calculate distributed attention to get the local attention output tensor
        >>> local_out, meta = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>>
        >>> # Step5. undispatch local attention output to the global one if needed
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """

    # Infer q_ranges, k_ranges and others
    # from cu_seqlens_q, cu_seqlens_k and causal
    (
        q_ranges,
        k_ranges,
        attn_mask_type,
        total_seqlen_q,
        total_seqlen_k,
    ) = infer_attn_mask_from_cu_seqlens(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=window_size,
    )

    # Call the API for flex key
    return magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
        dist_attn_config=dist_attn_config,
    )


@deprecated(
    "This API is deprecated and will be removed in future versions. "
    "Please use two steps calling of `magi_attn_varlen_key` + `dispatch` instead."
)
def magi_attn_varlen_dispatch(
    x: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
):
    """This is a flash-attn-varlen like interface,
    to generate ``q_ranges``, ``k_ranges`` and ``attn_mask_type``
    from ``cu_seqlens_q``, ``cu_seqlens_k``, ``causal`` and ``window_size``,
    calculate ``dist_attn_runtime_key`` and generate the corr. inner ``dist_attn_runtime_mgr``.

    Args:
        x (torch.Tensor): the global input tensor.

        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        num_heads_q (int): the number of heads for query.
        num_heads_kv (int): the number of heads for key/value.
        head_dim (int): the dimension of each attention head.

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by ``chunk_size * cp_size``.
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        causal (bool, optional): if ``True``, all mask types are set to ``CAUSAL``,
            otherwise, determine the mask types by ``window_size``. Defaults to ``False``.
        window_size (tuple[int, int], optional): window_size of sliding window mask
            which represents ``[window_size_left, window_size_right]``. The parameter is effective only
            when ``causal`` is ``False``; when ``causal`` is ``True``, it is required to be ``(-1, -1)``.
            Defaults to be ``(-1, -1)``.

        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        tuple[torch.Tensor, DistAttnRuntimeKey]:
            - x (torch.Tensor): the input tensor after padding.
            - key (DistAttnRuntimeKey): the key points to the inner DistAttnRuntimeMgr.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_varlen_dispatch, undispatch, calc_attn
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>>
        >>> # Step1. dispatch the global input tensor to local tensor
        >>> # with a dist_attn_runtime_key generated to store and indicate the inner meta info
        >>> local_x, dist_attn_runtime_key = magi_attn_varlen_dispatch(
        ...     x=torch.randn(
        ...         4096,  # seqlen
        ...         2048,  # hidden_size
        ...         device="cuda",
        ...         dtype=torch.bfloat16,
        ...         requires_grad=True
        ...     ),
        ...     cu_seqlen_q=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     cu_seqlen_k=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     causal=False,
        ...     window_size=(-1, -1),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. apply QKV projection on local tensors
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step3. calculate distributed attention to get the local attention output tensor
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>>
        >>> # Step4. undispatch local attention output to the global one if needed
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """

    key = magi_attn_varlen_key(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
        causal=causal,
        window_size=window_size,
        dist_attn_config=dist_attn_config,
    )

    local_x = dispatch(x, key)

    return (local_x, key)


def magi_attn_flex_key(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    total_seqlen_q: int,
    total_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
) -> DistAttnRuntimeKey:
    """This is the most flexible interface,
    directly passing in ``q_ranges``, ``k_ranges`` and ``attn_mask_type`` to
    generate ``dist_attn_runtime_key`` which stores and indicates the inner meta data
    as a required argument for following APIs including ``dispatch``, ``undispatch``, ``calc_attn``, etc.

    Args:
        q_ranges (AttnRanges): the global query ranges.
        k_ranges (AttnRanges): the global key ranges.
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]):
            the global attn mask type (list), represented by
            str or enum ``AttnMaskType`` or their mixed combination.

        total_seqlen_q (int): the total seqlen of query.
        total_seqlen_k (int): the total seqlen of key.

        num_heads_q (int): the number of heads for query.
        num_heads_kv (int): the number of heads for key/value.
        head_dim (int): the dimension of each attention head.

        pad_size (int): the size to pad the global input tensor along sequence dim,
            due to the constraint that the sequence length need to be divisable by ``chunk_size * cp_size``.
        chunk_size (int): the size to chunk the global input tensor along the seqlen dim
            for later sharding and dispatching among the cp ranks
            as a granularity factor of computational load-balance.

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        dist_attn_config (DistAttnConfig): dist attn config.

        is_same_source (bool): is query tensor and key tensor share the same source.
            Default to ``True``.
        is_q_permutable (bool): is query tensor permutable.
            Default to ``True``.
        is_k_permutable (bool): is key tensor permutable.
            Default to ``True``.

    Returns:
        DistAttnRuntimeKey: the key stores and indicates the inner meta data.

    NOTE:
        1. For decoder-only transformers (e.g., GPT), it applies 'self-attn' as follows:

            a. ``is_same_source`` is True.
            b. Both ``q`` and ``k`` are permutable, as long as they are permuted in the same way.

        2. For encoder-decoder transformers (e.g., T5), it applies 'cross-attn' as follows:

            a. ``is_same_source`` is False.
            b. ``q`` is permutable but ``k`` is not.

        3. For multi-modal transformers with external encoders, it applies 'cross-attn' as follows:

            a. ``is_same_source`` is False.
            b. ``q`` is unpermutable due to self-attn, but ``k`` is permutable even in a different way.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_flex_key, dispatch, undispatch, calc_attn
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>> from magi_attention.common import AttnRanges
        >>>
        >>> # Step1. generate a dist_attn_runtime_key to store and indicate the inner meta info
        >>> dist_attn_runtime_key = magi_attn_flex_key(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type="full",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. dispatch the global tensors to local tensors
        >>> local_x, local_label, local_rope = [
        ...     dispatch(tensor, dist_attn_runtime_key)
        ...     for tensor in [total_x, total_label, total_rope]
        ... ]
        >>>
        >>> # Step3. apply QKV projection on local tensors
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step4. calculate distributed attention to get the local attention output tensor
        >>> local_out, meta = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>>
        >>> # Step5. undispatch local attention output to the global one if needed
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """

    # Validate total_seqlen
    assert q_ranges.end <= total_seqlen_q and k_ranges.end <= total_seqlen_k, (
        f"The maximum endpoint in ranges must be less than total_seqlen, "
        f"but got {q_ranges.end=} when {total_seqlen_q=}, "
        f"and got {k_ranges.end=} when {total_seqlen_k=}"
    )

    # Validate and transform attn_mask_type
    attn_mask_type = wrap_to_list(attn_mask_type, broadcast_to_length=q_ranges.size)
    assert is_list_type_all(attn_mask_type, (str, AttnMaskType)), (
        f"attn_mask_type must be a list of str or AttnMaskType or their mixed combination, "
        f"but got {attn_mask_type=}"
    )
    attn_mask_type = [  # transform str to AttnMaskType, might raise ValueError
        AttnMaskType(type_name) for type_name in attn_mask_type
    ]
    assert len(attn_mask_type) == len(q_ranges), (
        f"the length of attn_mask_type must be same as q_ranges, "
        f"but got {len(attn_mask_type)=} and {len(q_ranges)=}"
    )

    # Validate process group (or device mesh)
    if isinstance(cp_group_or_mesh, dist.ProcessGroup):
        assert not magi_attention.comm.is_hierarchical_comm_enable(), (
            "A 2D cp_mesh must be provided when hierarchical comm is enabled, "
            "instead of a single cp_group"
        )
        cp_group = cp_group_or_mesh
        cp_mesh = None
    elif isinstance(cp_group_or_mesh, DeviceMesh):
        cp_mesh = cp_group_or_mesh
        assert cp_mesh.ndim <= 2, "cp_mesh must be 1D or 2D"
        if magi_attention.comm.is_hierarchical_comm_enable():
            assert (
                cp_mesh.ndim == 2
            ), "cp_mesh must be 2D when hierarchical comm is enabled"
        cp_group = cp_mesh._flatten().get_group()
    else:
        raise ValueError(
            f"cp_group_or_mesh must be a dist.ProcessGroup or dist.DistMesh, "
            f"but got {type(cp_group_or_mesh)=}"
        )

    # Apply padding
    if pad_size > 0:
        # Apply padding to the mask with the empty slice
        q_ranges, k_ranges, attn_mask_type = apply_padding(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen=total_seqlen_q,
            pad_size=pad_size,
        )
        # Apply padding to total_seqlen
        total_seqlen_q += pad_size
        total_seqlen_k += pad_size

    # Init dist attn runtime key
    key = init_dist_attn_runtime_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
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
    )

    # Init dist attn runtime mgr and map it to the key
    if key not in dist_attn_runtime_dict_mgr:
        dist_attn_runtime_dict_mgr[key] = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=cp_group,
            cp_mesh=cp_mesh,
            dist_attn_config=dist_attn_config,
            # TODO: think through other scnearios besides self-attn and cross-attn
            # and find a better way to represent these flags
            # now keep it here temporarily for consistency
            is_same_source=is_same_source,
            is_q_permutable=is_q_permutable,
            is_k_permutable=is_k_permutable,
        )

    return key


@deprecated(
    "This API is deprecated and will be removed in future versions. "
    "Please use two steps calling of `magi_attn_flex_key` + `dispatch` instead."
)
def magi_attn_flex_dispatch(
    x: torch.Tensor,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    total_seqlen_q: int,
    total_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
) -> tuple[torch.Tensor, DistAttnRuntimeKey]:
    """This is the most flexible interface,
    directly passing in ``q_ranges``, ``k_ranges`` and ``attn_mask_type`` to
    generate ``dist_attn_runtime_key`` which stores and indicates the inner meta data
    and then dispatch the global input tensor to local tensor.

    Args:
        x (torch.Tensor): the global input tensor.

        q_ranges (AttnRanges): the global query ranges.
        k_ranges (AttnRanges): the global key ranges.
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]):
            the global attn mask type (list), represented by
            str or enum ``AttnMaskType`` or their mixed combination.

        total_seqlen_q (int): the total seqlen of query.
        total_seqlen_k (int): the total seqlen of key.

        num_heads_q (int): the number of heads for query.
        num_heads_kv (int): the number of heads for key/value.
        head_dim (int): the dimension of each attention head.

        pad_size (int): the size to pad the global input tensor along sequence dim,
            due to the constraint that the sequence length need to be divisable by ``chunk_size * cp_size``.
        chunk_size (int): the size to chunk the global input tensor along the seqlen dim
            for later sharding and dispatching among the cp ranks
            as a granularity factor of computational load-balance.

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        dist_attn_config (DistAttnConfig): dist attn config.

        is_same_source (bool): is query tensor and key tensor share the same source.
            Default to ``True``.
        is_q_permutable (bool): is query tensor permutable.
            Default to ``True``.
        is_k_permutable (bool): is key tensor permutable.
            Default to ``True``.

    Returns:
        tuple[torch.Tensor, DistAttnRuntimeKey]:
            - local_x (torch.Tensor): the local input x after padding.
            - key (DistAttnRuntimeKey): the key points to the inner DistAttnRuntimeMgr.

    NOTE:
        1. For decoder-only transformers (e.g., GPT), it applies 'self-attn' as follows:

            a. ``is_same_source`` is True.
            b. Both ``q`` and ``k`` are permutable, as long as they are permuted in the same way.

        2. For encoder-decoder transformers (e.g., T5), it applies 'cross-attn' as follows:

            a. ``is_same_source`` is False.
            b. ``q`` is permutable but ``k`` is not.

        3. For multi-modal transformers with external encoders, it applies 'cross-attn' as follows:

            a. ``is_same_source`` is False.
            b. ``q`` is unpermutable due to self-attn, but ``k`` is permutable even in a different way.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_flex_dispatch, undispatch, calc_attn
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>> from magi_attention.common import AttnRanges
        >>>
        >>> # Step1. dispatch the global input tensor to local tensor
        >>> # with a dist_attn_runtime_key generated to store and indicate the inner meta info
        >>> local_x, dist_attn_runtime_key = magi_attn_flex_dispatch(
        ...     x = torch.randn(
        ...         4096,   # seqlen
        ...         2048,   # hidden_size
        ...         device="cuda",
        ...         dtype=torch.bfloat16,
        ...         requires_grad=True
        ...     ),
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type="full",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. apply QKV projection
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step3. calculate distributed attention to get the local attention output tensor
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>>
        >>> # Step4. undispatch local attention output to the global one if needed
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """

    key = magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
        dist_attn_config=dist_attn_config,
        is_same_source=is_same_source,
        is_q_permutable=is_q_permutable,
        is_k_permutable=is_k_permutable,
    )

    local_x = dispatch(x, key)
    return (local_x, key)


def dispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad and dispatch the global input tensor to local input tensor
    for each cp rank along the seqlen dim.

    Args:
        x (torch.Tensor): the global input tensor.
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as a required argument for many APIs of ``magi_attention``,
            which users don't have to bother with.
        pad_value (float): the specific value to pad to input tensor.
            Defaults to ``0``.

    Returns:
        torch.Tensor: the padded local input tensor.

    Raises:
        ValueError: If the provided ``key`` does not exist in cached ``dist_attn_runtime_dict``.
    """

    mgr = dist_attn_runtime_dict_mgr.get(key)
    if mgr is None:
        raise ValueError("The dist attn runtime key does not exist!")

    padded_x = pad_at_dim(x=x, dim=0, pad_size=key.pad_size, value=pad_value)
    padded_local_x = mgr.dispatch_qo(padded_x)

    return padded_local_x


def undispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> torch.Tensor:
    """
    Undispatch and unpad the local output tensor to global output tensor
    for each cp rank along the seqlen dim.

    Args:
        x (torch.Tensor): the local output tensor.
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as a required argument for many APIs of ``magi_attention``,
            which users don't have to bother with.

    Returns:
        torch.Tensor: the unpadded global output tensor.

    Raises:
        ValueError: If the provided ``key`` does not exist in cached ``dist_attn_runtime_dict``.
    """

    mgr = dist_attn_runtime_dict_mgr.get(key)
    if mgr is None:
        raise ValueError("The dist attn runtime key does not exist!")

    global_x = mgr.undispatch_qo(x)
    unpadded_global_x = unpad_at_dim(x=global_x, dim=0, pad_size=key.pad_size)

    return unpadded_global_x


def calc_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: DistAttnRuntimeKey,
    sink: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    softcap: float = 0.0,
) -> tuple[torch.Tensor, AttnForwardMeta]:
    """
    Calculate distributed attention with local q, k, v tensors.

    Args:
        q (torch.Tensor): the local query tensor.
        k (torch.Tensor): the local key tensor.
        v (torch.Tensor): the local value tensor.
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as a required argument for many APIs of ``magi_attention``,
            which users don't have to bother with.

        sink (torch.Tensor, optional): the global sink tensor (replicated among cp ranks).
            Defaults to ``None`` to not apply attention sink.

        softmax_scale (float, optional): softmax scale.
            Defaults to ``None`` to use the value: ``1/sqrt(head_dim)``.
        softcap (float, optional): softcap.
            Defaults to ``0.0``.

    Returns:
        tuple[torch.Tensor, AttnForwardMeta]:
            - out (torch.Tensor): local output tensor.
            - meta (AttnForwardMeta): attention forward meta.

    Shapes:
        - q: [num_tokens_q_local, num_heads_q, head_dim]
        - k: [num_tokens_kv_local, num_heads_kv, head_dim]
        - v: [num_tokens_kv_local, num_heads_kv, head_dim]
        - sink: [num_tokens_sink_global, num_heads_q]
        - out: [num_tokens_q_local, num_heads_q, head_dim]
        - lse: [num_tokens_q_local, num_heads_q]

    Raises:
        ValueError: If the provided ``key`` does not exist in cached ``dist_attn_runtime_dict``.
    """

    mgr = dist_attn_runtime_dict_mgr.get(key)
    if mgr is None:
        raise ValueError("The dist attn runtime key does not exist!")

    return mgr.calc_attn(
        q=q,
        k=k,
        v=v,
        sink=sink,
        softmax_scale=softmax_scale,
        softcap=softcap,
    )


def get_position_ids(key: DistAttnRuntimeKey) -> torch.Tensor:
    """
    Get the global positional ids of the local tensor,
    as it is sliced from the global tensor after dispatching.

    Args:
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as a required argument for many APIs of ``magi_attention``,
            which users don't have to bother with.

    Returns:
        torch.Tensor: the global positional ids.

    Raises:
        ValueError: If the provided ``key`` does not exist in cached ``dist_attn_runtime_dict``.
    """

    mgr = dist_attn_runtime_dict_mgr.get(key)
    if mgr is None:
        raise ValueError("The dist attn runtime key does not exist!")

    return mgr.get_position_ids()


def get_most_recent_key(
    cp_group: dist.ProcessGroup,
) -> DistAttnRuntimeKey:
    """Get the most recent inserted key.

    NOTE: this is useful when you can not access the key through the arguments,
    and meanwhile you only need the most recent inserted key.
    However, we strongly recommend you to access the key 
    passed through the arguments, in case of unexpected inconsistency.

    Returns:
        DistAttnRuntimeKey: the most recent inserted dist_attn_runtime_key.
    """

    key = dist_attn_runtime_dict_mgr.get_most_recent_key(cp_group)
    if key is None:
        raise ValueError(f"The dist attn runtime dict is empty for {cp_group}!")

    return key


def make_varlen_key_for_new_mask_after_dispatch(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    key_for_dispatch: DistAttnRuntimeKey,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dist_attn_config: DistAttnConfig | None = None,
) -> DistAttnRuntimeKey:
    """Make a new dist attn runtime key for a new mask after dispatch
    with the given arguments for the new mask in flash-attn-varlen style and the key used for dispatch

    NOTE: this API is useful when you want to apply more than one masks
    within the same training pass, if your model adopts hybrid-attn structure,
    in which case, we can only choose one of the masks to dispatch,
    while the others're supposed to reuse the same dispatch solution
    with different meta arguments for computation and communication

    WARNING: in such case, we can not guarantee all the masks are load-balanced in computation
    and optimized in communication.

    Args:
        cu_seqlens_q (torch.Tensor): the cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): the cumulative sequence lengths for keys.

        key_for_dispatch (DistAttnRuntimeKey): the key used for dispatch.

        causal (bool, optional): whether the varlen attention mask is causal.
            Defaults to ``False``.
        window_size (tuple[int, int], optional): window_size of sliding window mask
            which represents ``[window_size_left, window_size_right]``. The parameter is effective only
            when ``causal`` is ``False``; when ``causal`` is ``True``, it is required to be ``(-1, -1)``.
            Defaults to be ``(-1, -1)``.

        dist_attn_config (DistAttnConfig, optional): the optional new dist attn config.
            NOTE: if not provided, we will use the same config as the ``key_for_dispatch``,
            and if provided, the dispatch config of the new dist attn config won't be applied to the new mask

    Returns:
        DistAttnRuntimeKey: the new dist attn runtime key
            for new mask with the same dispatch solution as the ``key_for_dispatch``.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_varlen_key, dispatch, undispatch, calc_attn
        >>> from magi_attention.api import make_varlen_key_for_new_mask_after_dispatch
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>>
        >>> # Step1. generate a dist_attn_runtime_key to dispatch for flash-attn-varlen style mask
        >>> # in the following case, we use a causal mask as the key for dispatch, thus it will consider
        >>> # computation load-balance, communication optimization and computation-communication overlap
        >>> # according to the causal mask pattern
        >>> key_for_dispatch = magi_attn_varlen_key(
        ...     cu_seqlen_q=torch.tensor(
        ...         [0, 4096], dtype=torch.int32
        ...     ),
        ...     cu_seqlen_k=torch.tensor(
        ...         [0, 4096], dtype=torch.int32
        ...     ),
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512), # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     causal=True,
        ...     window_size=(-1, -1),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. dispatch the global tensors to local tensors with the same key_for_dispatch
        >>> local_x, local_label, local_rope = [
        ...     dispatch(tensor, key_for_dispatch)
        ...     for tensor in [total_x, total_label, total_rope]
        ... ]
        >>>
        >>> # Step3. make a new dist_attn_runtime_key from key_for_dispatch
        >>> # for a new mask, such as a sliding window causal mask below,
        >>> # with the same dispatch solution as the causal mask used for dispatch,
        >>> # i.e. this new key share the same dispatch meta as key_for_dispatch
        >>> # but it can handle the computation and communication of the new mask
        >>> # and calculate attn correctly as well, though no optimization is applied for now
        >>> new_key_for_swa_mask = make_varlen_key_for_new_mask_after_dispatch(
        ...     cu_seqlens_q=torch.tensor([0, 4096], dtype=torch.int32),
        ...     cu_seqlens_k=torch.tensor([0, 4096], dtype=torch.int32),
        ...     causal=False,
        ...     window_size=(512, 0), # sliding window causal mask
        ...     key_for_dispatch=key_for_dispatch,
        ... )
        >>>
        >>> # Step4. apply QKV projection on local tensors
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step5. calculate distributed attention
        >>> # for the causal mask used to dispatch with key_for_dispatch
        >>> local_out1, _ = calc_attn(local_q, local_k, local_v, key_for_dispatch)
        >>>
        >>> # Step6. calculate distributed attention
        >>> # for the new swa mask with the new key
        >>> # w/o undispatching back and re-dispatching again to avoid OOM
        >>> local_out2, _ = calc_attn(local_q, local_k, local_v, new_key_for_swa_mask)
        >>>
        >>> # Step7. undispatch local attention output to the global one if needed
        >>> total_out1 = undispatch(local_out1, key_for_dispatch)
        >>> total_out2 = undispatch(local_out2, new_key_for_swa_mask)
    """

    # Infer q_ranges, k_ranges and others
    # from cu_seqlens_q, cu_seqlens_k and causal
    (
        q_ranges,
        k_ranges,
        attn_mask_type,
        _,  # total_seqlen_q
        _,  # total_seqlen_k
    ) = infer_attn_mask_from_cu_seqlens(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        window_size=window_size,
    )

    # Call the API for flex key
    return make_flex_key_for_new_mask_after_dispatch(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        key_for_dispatch=key_for_dispatch,
        dist_attn_config=dist_attn_config,
    )


def make_flex_key_for_new_mask_after_dispatch(
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    key_for_dispatch: DistAttnRuntimeKey,
    dist_attn_config: DistAttnConfig | None = None,
) -> DistAttnRuntimeKey:
    """Make a new dist attn runtime key for a new mask after dispatch
    with the given arguments for the new mask and the key used for dispatch

    NOTE: this API is useful when you want to apply more than one masks
    within the same training pass, if your model adopts hybrid-attn structure,
    in which case, we can only choose one of the masks to dispatch,
    while the others're supposed to reuse the same dispatch solution
    with different meta arguments for computation and communication

    WARNING: in such case, we can not guarantee all the masks are load-balanced in computation
    and optimized in communication for now. However, we are working on it with the dynamic dist-attn solver
    to optimize the computation and communication for each distinct mask with the same dispatch solution

    Args:
        q_ranges (AttnRanges): the global query ranges.
        k_ranges (AttnRanges): the global key ranges.
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]):
            the global attn mask type (list), represented by
            str or enum ``AttnMaskType`` or their mixed combination.

        key_for_dispatch (DistAttnRuntimeKey): the key used for dispatch.

        dist_attn_config (DistAttnConfig, optional): the optional new dist attn config.
            NOTE: if not provided, we will use the same config as the ``key_for_dispatch``,
            and if provided, the dispatch config of the new dist attn config won't be applied to the new mask

    Returns:
        DistAttnRuntimeKey: the new dist attn runtime key
            for new mask with the same dispatch solution as the ``key_for_dispatch``

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from magi_attention.api import magi_attn_flex_key, dispatch, undispatch, calc_attn
        >>> from magi_attention.api import make_flex_key_for_new_mask_after_dispatch
        >>> from magi_attention.api.functools import compute_pad_size
        >>> from magi_attention.config import (
        ...     DistAttnConfig,
        ...     DispatchConfig,
        ...     OverlapConfig,
        ...     MinHeapDispatchAlg,
        ...     UniformOverlapAlg
        ... )
        >>> from magi_attention.common.enum import AttnOverlapMode
        >>> from magi_attention.common import AttnRanges
        >>>
        >>> # Step1. generate a dist_attn_runtime_key to dispatch for arbitrary mask represented by attn slices
        >>> # in the following case, we use a causal mask as the key for dispatch, thus it will consider
        >>> # computation load-balance, communication optimization and computation-communication overlap
        >>> # according to the causal mask pattern
        >>> key_for_dispatch = magi_attn_flex_key(
        ...     q_ranges=AttnRanges.from_ranges([[0, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 4096]]),
        ...     attn_mask_type="causal",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     num_heads_q=16,
        ...     num_heads_kv=4,
        ...     head_dim=128,
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
        ...     dist_attn_config=DistAttnConfig(
        ...         dispatch_config=DispatchConfig(alg=MinHeapDispatchAlg()),
        ...         overlap_config=OverlapConfig(
        ...             enable=True,
        ...             mode=AttnOverlapMode.STATIC,
        ...             degree=2,
        ...             min_chunk_size=512,
        ...             max_num_chunks=64,
        ...             alg=UniformOverlapAlg(),
        ...         ),
        ...     ),
        ... )
        >>>
        >>> # Step2. dispatch the global tensors to local tensors with the same key_for_dispatch
        >>> local_x, local_label, local_rope = [
        ...     dispatch(tensor, key_for_dispatch)
        ...     for tensor in [total_x, total_label, total_rope]
        ... ]
        >>>
        >>> # Step3. make a new dist_attn_runtime_key from key_for_dispatch
        >>> # for a new mask, such as a sliding window causal mask below,
        >>> # with the same dispatch solution as the causal mask used for dispatch,
        >>> # i.e. this new key share the same dispatch meta as key_for_dispatch
        >>> # but it can handle the computation and communication of the new mask
        >>> # and calculate attn correctly as well, though no optimization is applied for now
        >>> new_key_for_swa_mask = make_flex_key_for_new_mask_after_dispatch(
        ...     q_ranges=AttnRanges.from_ranges([[0, 512], [512, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 512], [0, 4096]]),
        ...     attn_mask_type=["causal", "bi_causal"], # sliding window causal mask
        ...     key_for_dispatch=key_for_dispatch,
        ... )
        >>>
        >>> # Step4. apply QKV projection on local tensors
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>>
        >>> # Step5. calculate distributed attention
        >>> # for the causal mask used to dispatch with key_for_dispatch
        >>> local_out1, _ = calc_attn(local_q, local_k, local_v, key_for_dispatch)
        >>>
        >>> # Step6. calculate distributed attention
        >>> # for the new swa mask with the new key
        >>> # w/o undispatching back and re-dispatching again to avoid OOM
        >>> local_out2, _ = calc_attn(local_q, local_k, local_v, new_key_for_swa_mask)
        >>>
        >>> # Step7. undispatch local attention output to the global one if needed
        >>> total_out1 = undispatch(local_out1, key_for_dispatch)
        >>> total_out2 = undispatch(local_out2, new_key_for_swa_mask)
    """

    # Validate and transform attn_mask_type
    attn_mask_type = wrap_to_list(attn_mask_type, broadcast_to_length=q_ranges.size)
    assert is_list_type_all(attn_mask_type, (str, AttnMaskType)), (
        f"attn_mask_type must be a list of str or AttnMaskType or their mixed combination, "
        f"but got {attn_mask_type=}"
    )
    attn_mask_type = [  # transform str to AttnMaskType, might raise ValueError
        AttnMaskType(type_name) for type_name in attn_mask_type
    ]
    assert len(attn_mask_type) == len(q_ranges), (
        f"the length of attn_mask_type must be same as q_ranges, "
        f"but got {len(attn_mask_type)=} and {len(q_ranges)=}"
    )

    # Extract the common attributes from the key for dispatch
    total_seqlen_q = key_for_dispatch.total_seqlen_q  # already padded
    total_seqlen_k = key_for_dispatch.total_seqlen_k  # already padded
    pad_size = key_for_dispatch.pad_size
    chunk_size = key_for_dispatch.chunk_size
    cp_group = key_for_dispatch.cp_group
    cp_mesh = key_for_dispatch.cp_mesh
    new_dist_attn_config = DistAttnConfig(
        dispatch_config=key_for_dispatch.dist_attn_config.dispatch_config,  # reuse the dispatch config
        overlap_config=dist_attn_config.overlap_config
        if dist_attn_config is not None
        else key_for_dispatch.dist_attn_config.overlap_config,
    )

    # Extract the common attributes from the mgr for dispatch
    mgr = dist_attn_runtime_dict_mgr.get(key_for_dispatch)
    if mgr is None:
        raise ValueError("The dist attn runtime key for dispatch does not exist!")

    ref_dispatch_meta_q = mgr.dispatch_meta_q
    ref_dispatch_meta_k = mgr.dispatch_meta_k

    is_same_source = mgr.is_same_source
    is_q_permutable = mgr.is_q_permutable
    is_k_permutable = mgr.is_k_permutable

    num_heads_q = mgr.num_heads_q
    num_heads_kv = mgr.num_heads_kv
    head_dim = mgr.head_dim

    # Apply padding
    if pad_size > 0:
        # Apply padding to the new mask with the empty slice
        q_ranges, k_ranges, attn_mask_type = apply_padding(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen=total_seqlen_q - pad_size,
            pad_size=pad_size,
        )

    # Init new dist attn runtime key
    new_key = init_dist_attn_runtime_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        dist_attn_config=new_dist_attn_config,
    )

    # init new dist attn runtime mgr and map it to the new key
    # Use per-cp_group cache to avoid LRU eviction inconsistency
    if new_key not in dist_attn_runtime_dict_mgr:
        dist_attn_runtime_dict_mgr[new_key] = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            chunk_size=chunk_size,
            cp_group=cp_group,
            cp_mesh=cp_mesh,
            dist_attn_config=new_dist_attn_config,
            is_same_source=is_same_source,
            is_q_permutable=is_q_permutable,
            is_k_permutable=is_k_permutable,
            ref_dispatch_meta_q=ref_dispatch_meta_q,
            ref_dispatch_meta_k=ref_dispatch_meta_k,
        )

    return new_key
