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

import copy
from typing import Sequence, TypeAlias

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

import magi_attention
from magi_attention.common import AttnRange, AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DistAttnConfig
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey, DistAttnRuntimeMgr
from magi_attention.functional.dist_attn import DistFlashAttnRuntime
from magi_attention.meta import (
    calc_attn_meta_from_dispatch_meta,
    calc_dispatch_meta_from_qk_ranges,
)
from magi_attention.utils import wrap_to_list
from magi_attention.utils._utils import is_list_type_all

from .functools import FixedLenDict, pad_at_dim, unpad_at_dim

DistAttnRuntimeDict = FixedLenDict(
    max_size=100
)  # [DistAttnRuntimeKey, DistAttnRuntimeMgr]


GeneralAttnMaskType: TypeAlias = str | AttnMaskType | Sequence[str | AttnMaskType]


def magi_attn_varlen_key(
    x: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    causal: bool = False,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
) -> tuple[torch.Tensor, DistAttnRuntimeKey]:
    """This is a flash_attn_varlen like interface, to
    generate q_ranges, k_ranges and attn_mask_type from cu_seqlens_q, cu_seqlens_k and causal,
    further to pad the input tensor, caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr.

    Args:
        x (torch.Tensor): input tensor

        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        head_dim (int): head dim for q k v. The head_dim must be divisible by 8 and <= 192.
        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
        to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now
        NOTE: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        causal (bool): if True, all attn_mask_type is CAUSAL. else, all attn_mask_type is FULL.
        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        x (torch.Tensor): the input tensor after padding.
        DistAttnRuntimeKey (DistAttnRuntimeKey): DistAttbRuntimeKey.

    Example:
        >>> local_x, dist_attn_runtime_key = magi_attn_varlen_key(
        ...     x=torch.randn(
        ...         4096, # seqlen
        ...         2048,  # hidden_size
        ...         device=device,
        ...         dtype=dtype,
        ...         requires_grad = True
        ...     ),
        ...     cu_seqlen_q=torch.tensor(
                    [0, 2048, 4096], dtype=torch.int32
                ),
        ...     cu_seqlen_k=torch.tensor(
                    [0, 2048, 4096], dtype=torch.int32
                ),
        ...     pad_size=compute_pad_size(4096, 4, 64, 512), # seqlne, cp_size, head_dim, chunk_size
        ...     chunk_size=512,
        ...     head_dim=64,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     cp_mesh=None,
        ...     causal=False,
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
        >>> local_q = dispatch(total_q, dist_attn_runtime_key)
        >>> # Dispatch global key tensor to local key tensor
        >>> local_k = dispatch(total_k, dist_attn_runtime_key)
        >>> # Dispatch global value tensor to local value tensor
        >>> local_v = dispatch(total_v, dist_attn_runtime_key)
        >>> # Calculate local attention result
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>> # Gather local attention results to global result
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """
    # generate q_ranges, k_ranges and attn_mask_type
    # Note: the q_ranges and k_ranges must come from list.
    q_ranges: AttnRanges = AttnRanges.from_ranges(
        torch.stack([cu_seqlens_q[:-1], cu_seqlens_q[1:]], dim=1).tolist()
    )
    k_ranges: AttnRanges = AttnRanges.from_ranges(
        torch.stack([cu_seqlens_k[:-1], cu_seqlens_k[1:]], dim=1).tolist()
    )

    total_seqlen_q: int = int(cu_seqlens_q[-1])
    total_seqlen_k: int = int(cu_seqlens_k[-1])

    attn_mask_type = [AttnMaskType.CAUSAL if causal else AttnMaskType.FULL] * len(
        q_ranges
    )

    # call magi_attn_flex_key
    # for flash_attn_varlen: is_same_source, is_q_permute and is_k_permute are all set to true.
    return magi_attn_flex_key(
        x=x,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
        dist_attn_config=dist_attn_config,
    )


def magi_attn_varlen_dispatch(
    x: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    causal: bool = False,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
):
    """This is a flash_attn_varlen like interface, to
    generate q_ranges, k_ranges and attn_mask_type from cu_seqlens_q, cu_seqlens_k and causal,
    further to pad the input tensor, caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr,
    and finally dispatch the input tensor to local tensor.

    Args:
        x (torch.Tensor): input tensor

        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        head_dim (int): head dim for q k v. The head_dim must be divisible by 8 and <= 192.
        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
        to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now
        NOTE: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        causal (bool): if True, all attn_mask_type is CAUSAL. else, all attn_mask_type is FULL.
        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        x (torch.Tensor): the input tensor after padding.
        DistAttnRuntimeKey (DistAttnRuntimeKey): DistAttbRuntimeKey.

    Example:
        >>> padded_x, dist_attn_runtime_key = magi_attn_varlen_dispatch(
        ...     x=torch.randn(
        ...         4096,  # seqlen
        ...         2048,  # hidden_size
        ...         device=device,
        ...         dtype=dtype,
        ...         requires_grad = True
        ...     ),
        ...     cu_seqlen_q=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     cu_seqlen_k=torch.tensor(
        ...         [0, 2048, 4096], dtype=torch.int32
        ...     ),
        ...     pad_size=compute_pad_size(4096, 4, 64, 512),  # seqlen, cp_size, head_dim, chunk_size
        ...     chunk_size=512,
        ...     head_dim=64,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     cp_mesh=None,
        ...     causal=False,
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
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>> # Do local attention computation
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>> # Gather local attention results to global result
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """
    padded_x, key = magi_attn_varlen_key(
        x=x,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        head_dim=head_dim,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        causal=causal,
        dist_attn_config=dist_attn_config,
    )

    local_x = dispatch(padded_x, key)

    return (local_x, key)


def magi_attn_flex_key(
    x: torch.Tensor,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    total_seqlen_q: int,
    total_seqlen_k: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
) -> tuple[torch.Tensor, DistAttnRuntimeKey]:
    """This is the most flexible interface,
    directly passing in q_ranges, k_ranges and attn_mask_type to
    pad the input tensor x, caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr.

    Args:
        x (torch.Tensor): input tensor
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]): attn mask type (list)
            represented by str or enum `AttnMaskType` or their mixed combination

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        head_dim (int): head dim for q k v. The head_dim must be divisible by 8 and <= 192.
        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
        to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now
        NOTE: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        dist_attn_config (DistAttnConfig): dist attn config

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

    Returns:
        x (torch.Tensor): the input tensor after padding.
        DistAttnRuntimeKey (DistAttnRuntimeKey): DistAttbRuntimeKey.

    Example:
        >>> padded_x, dist_attn_runtime_key = magi_attn_flex_key(
        ...     x = torch.randn(
        ...         4096,
        ...         2048,
        ...         device=device,
        ...         dtype=dtype,
        ...         requires_grad = True
        ...     ),
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type="full",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     pad_size=compute_pad_size(4096, 4, 64, 512),  # seqlen, cp_size, head_dim, chunk_size
        ...     chunk_size=512,
        ...     head_dim=64,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     cp_mesh=None,
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
        >>> local_q = dispatch(total_q, dist_attn_runtime_key)
        >>> # Dispatch global key tensor to local key tensor
        >>> local_k = dispatch(total_k, dist_attn_runtime_key)
        >>> # Dispatch global value tensor to local value tensor
        >>> local_v = dispatch(total_v, dist_attn_runtime_key)
        >>> # Calculate local attention result
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>> # Gather local attention results to global result
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
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

    # Validate process group (or device mesh)
    if cp_group is None and cp_mesh is None:
        raise ValueError("Either cp_group or cp_mesh must be provided")
    if cp_group is not None and cp_mesh is not None:
        raise ValueError("Only one of cp_group or cp_mesh can be provided")
    if cp_mesh is not None:
        assert cp_mesh.ndim <= 2, "cp_mesh must be 1D or 2D"
        if magi_attention.comm.is_hierarchical_comm_enable():
            assert (
                cp_mesh.ndim == 2
            ), "cp_mesh must be 2D when hierarchical comm is enabled"
        cp_group = cp_mesh._flatten().get_group()
    else:
        assert not magi_attention.comm.is_hierarchical_comm_enable(), (
            "A 2D cp_mesh must be provided when hierarchical comm is enabled, "
            "instead of a single cp_group"
        )

    # Validate head_dim
    if head_dim % 8 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by 8")
    if head_dim > 192:
        raise ValueError(f"head_dim ({head_dim}) must be ≤ 192")

    key = DistAttnRuntimeKey(
        cp_group,  # FIXME: ignore cp_mesh to be part of key for now
        pad_size,
        head_dim,
        q_ranges,
        k_ranges,
        attn_mask_type,
        total_seqlen_q,
        total_seqlen_k,
        dist_attn_config,
    )

    # deepcopy qk range and attn_mask and do padding to avoid the modification of key
    def apply_padding(q_ranges, k_ranges, attn_mask_type):
        q_range = AttnRanges.from_ranges(q_ranges.to_naive_ranges(), check=True)
        k_range = AttnRanges.from_ranges(k_ranges.to_naive_ranges(), check=True)
        attn_mask_types = copy.deepcopy(attn_mask_type)

        q_range.append(AttnRange(start=total_seqlen_q, end=total_seqlen_q + pad_size))
        k_range.append(AttnRange(start=0, end=0))
        attn_mask_types.append(AttnMaskType.FULL)

        return q_range, k_range, attn_mask_types

    # Apply padding at seq_dim(dim 0）
    if pad_size > 0:
        x = pad_at_dim(x, 0, pad_size)
        q_ranges, k_ranges, attn_mask_type = apply_padding(
            q_ranges, k_ranges, attn_mask_type
        )

        total_seqlen_q += pad_size
        total_seqlen_k += pad_size

    # Validate sequence length
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

    if key not in DistAttnRuntimeDict.keys():
        # calculate dist attn runtime key
        comm_meta, attn_calc_meta, attn_solver = calc_attn_meta_from_dispatch_meta(
            dispatch_meta_q=q_dispatch_meta,
            dispatch_meta_k=k_dispatch_meta,
            bucket_per_rank=attn_buckets,
            cp_group=cp_group,
            cp_mesh=cp_mesh,
            high_bandwith_domain_size=dist_attn_config.high_bandwith_domain_size,
            overlap_config=dist_attn_config.overlap_config,
        )

        dist_attn_runtime = DistFlashAttnRuntime(
            comm_meta=comm_meta,
            calc_meta=attn_calc_meta,
            cp_group_kv=cp_group,
            cp_group_dkv=cp_group,  # TODO: support interface to set distinct cp group for dkv
            deterministic=dist_attn_config.deterministic,
        )

        # generate DistAttnRuntimeMgr
        value = DistAttnRuntimeMgr(
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

        DistAttnRuntimeDict[key] = value

    return (x, key)


def magi_attn_flex_dispatch(
    x: torch.Tensor,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    total_seqlen_q: int,
    total_seqlen_k: int,
    head_dim: int,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
) -> tuple[torch.Tensor, DistAttnRuntimeKey]:
    """This is the most flexible interface,
    directly passing in q_ranges, k_ranges and attn_mask_type to
    pad the input tensor x, caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr,
    and dispatch the input tensor to local tensor.

    Args:
        x (torch.Tensor): input tensor
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]): attn mask type (list)
            represented by str or enum `AttnMaskType` or their mixed combination

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        head_dim (int): head dim for q k v. The head_dim must be divisible by 8 and <= 192.
        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
        to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now
        NOTE: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        dist_attn_config (DistAttnConfig): dist attn config

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

    Returns:
        local_x (torch.Tensor): the local input x after padding.
        key (DistAttnRuntimeKey): DistAttnRuntimeKey.

    Example:
        >>> local_x, dist_attn_runtime_key = magi_attn_flex_dispatch(
        ...     x = torch.randn(
        ...         4096,   # seqlen
        ...         2048,   # hidden_size
        ...         device=device,
        ...         dtype=dtype,
        ...         requires_grad = True
        ...     ),
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type="full",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     pad_size=compute_pad_size(4096, 4, 64, 512),  # seqlen, cp_size, head_dim, chun_size
        ...     chunk_size=512,
        ...     head_dim=64,
        ...     cp_group=dist.new_group(list(range(4)), backend="nccl"),
        ...     cp_mesh=None,
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
        >>> local_q, local_k, local_v = q_project(local_x), k_project(local_x), v_project(local_x)
        >>> # Do local attention computation
        >>> local_out, _ = calc_attn(local_q, local_k, local_v, dist_attn_runtime_key)
        >>> # Gather local attention results to global result
        >>> total_out = undispatch(local_out, dist_attn_runtime_key)
    """

    padded_x, key = magi_attn_flex_key(
        x=x,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        head_dim=head_dim,
        pad_size=pad_size,
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

    local_x = dispatch(padded_x, key)
    return (local_x, key)


def dispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> torch.Tensor:
    """
    Dispatch the input tensor to local tensor on each rank along dim0 (seqlen dim).
    args:
        x (torch.Tensor): input total tensor.
        key (DistAttnRuntimeKey): the object that holds some inner meta data
        as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    returns:
        local_x (torch.Tensor): the dispatched local tensor.

    Raises:
        ValueError: If the provided `key` does not exist in `DistAttnRuntimeDict`.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("DistRunTimeKey not exists!")

    return mgr.dispatch_qo(x)


def undispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> torch.Tensor:
    """
    Undispatch local tensor to total tensor and unpad the total tensor at dim0 (seqlen dim).
    args:
        x (torch.Tensor): local tensor
        key (DistAttnRuntimeKey): the object that holds some inner meta data
        as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    returns:
        unpad_total_x (torch.Tensor): the undispatched and unpadded tensor.

    Raises:
        ValueError: If the provided `key` does not exist in `DistAttnRuntimeDict`.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("DistRunTimeKey not exists!")

    total_x = mgr.undispatch_qo(x)
    pad_size = key.pad_size
    unpad_total_x = unpad_at_dim(total_x, 0, pad_size)

    return unpad_total_x


def calc_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Do attention computation.

    Args:
        q (torch.Tensor): Query tensor of shape `(num_tokens_q, num_heads, head_dim)`.
        k (torch.Tensor): Key tensor of shape `(num_tokens_k, num_heads, head_dim)`.
        v (torch.Tensor): Value tensor of shape `(num_tokens_v, num_heads, head_dim)`.
        key (DistAttnRuntimeKey): the object that holds some inner meta data
        as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    Returns:
        out (torch.Tensor): Attention output tensor of shape.
        lse (torch.Tensor): Log-sum-exp values for numerical stability.

    Raises:
        ValueError: If the provided `key` does not exist in `DistAttnRuntimeDict`.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("DistRunTimeKey not exists!")

    return mgr.calc_attn(q, k, v)


def get_position_ids(key: DistAttnRuntimeKey) -> torch.Tensor:
    """
    Get the position ids of local tensor to global tensor after dispatching.

    Args:
        key (DistAttnRuntimeKey): the object that holds some inner meta data
        as one argument for many other magi_attention APIs, about which the users may have no bother to care.
    Returns:
        position_ids (torch.Tensor): postion_ids of local tensor to global tensor.
    """
    mgr: DistAttnRuntimeMgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("DistRunTimeKey not exists!")

    return mgr.get_position_ids()


def get_most_recent_key() -> DistAttnRuntimeKey:
    """Get the most recent inserted key.

    Returns:
        key (DistAttnRuntimeKey): the most recent inserted key.
    """
    return DistAttnRuntimeDict.get_most_recent_key()
