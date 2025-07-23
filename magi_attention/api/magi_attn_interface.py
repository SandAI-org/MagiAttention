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

from typing import Sequence, TypeAlias

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from magi_attention.common import AttnRanges
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

from .functools import FixedLenDict, apply_padding, pad_at_dim, unpad_at_dim

DistAttnRuntimeDict = FixedLenDict(
    max_size=100
)  # [DistAttnRuntimeKey, DistAttnRuntimeMgr]


GeneralAttnMaskType: TypeAlias = str | AttnMaskType | Sequence[str | AttnMaskType]


def magi_attn_varlen_key(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    causal: bool = False,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
) -> DistAttnRuntimeKey:
    """This is a flash_attn_varlen like interface, to
    generate q_ranges, k_ranges and attn_mask_type from cu_seqlens_q, cu_seqlens_k and causal,
    caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr.

    Args:
        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.
            **NOTE**: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        causal (bool): if True, all attn_mask_type is CAUSAL. else, all attn_mask_type is FULL.
        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        DistAttnRuntimeKey: the key points to the inner DistAttnRuntimeMgr.

    Example:
        >>> dist_attn_runtime_key = magi_attn_varlen_key(
        ...     cu_seqlen_q=torch.tensor(
                    [0, 2048, 4096], dtype=torch.int32
                ),
        ...     cu_seqlen_k=torch.tensor(
                    [0, 2048, 4096], dtype=torch.int32
                ),
        ...     pad_size=compute_pad_size(4096, 4, 512), # seqlne, cp_size, chunk_size
        ...     chunk_size=512,
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
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
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
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    causal: bool = False,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
):
    """This is a flash_attn_varlen like interface, to
    generate q_ranges, k_ranges and attn_mask_type from cu_seqlens_q, cu_seqlens_k and causal flag,
    further caculate DistAttnRuntimeKey, generate the corr. inner DistAttnRuntimeMgr,
    finally pad and dispatch the input tensor to local tensor.

    Args:
        x (torch.Tensor): input tensor

        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.
            **NOTE**: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        causal (bool): if True, all attn_mask_type is CAUSAL. else, all attn_mask_type is FULL.
        dist_attn_config (DistAttnConfig): dist attn config.

    Returns:
        tuple[torch.Tensor, DistAttnRuntimeKey]:
            - x (torch.Tensor): the input tensor after padding.
            - key (DistAttnRuntimeKey): the key points to the inner DistAttnRuntimeMgr.

    Example:
        >>> local_x, dist_attn_runtime_key = magi_attn_varlen_dispatch(
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
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
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
    key = magi_attn_varlen_key(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group=cp_group,
        cp_mesh=cp_mesh,
        causal=causal,
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
    pad_size: int,
    chunk_size: int,
    cp_group: dist.ProcessGroup | None = None,
    cp_mesh: DeviceMesh | None = None,
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
    is_same_source: bool = True,
    is_q_permutable: bool = True,
    is_k_permutable: bool = True,
) -> DistAttnRuntimeKey:
    """This is the most flexible interface,
    directly passing in q_ranges, k_ranges and attn_mask_type to
    caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr.

    Args:
        x (torch.Tensor): input tensor
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]): attn mask type (list)
            represented by str or enum ``AttnMaskType`` or their mixed combination

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now.
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.
            **NOTE**: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        dist_attn_config (DistAttnConfig): dist attn config

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable

    Returns:
        DistAttnRuntimeKey: the key points to the inner DistAttnRuntimeMgr.

    Note:
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
        >>> dist_attn_runtime_key = magi_attn_flex_key(
        ...     q_ranges=AttnRanges.from_ranges([[0, 2048], [2048, 4096]]),
        ...     k_ranges=AttnRanges.from_ranges([[0, 2048], [0, 4096]]),
        ...     attn_mask_type="full",
        ...     total_seqlen_q=4096,
        ...     total_seqlen_k=4096,
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chunk_size
        ...     chunk_size=512,
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
        cp_group = cp_mesh._flatten().get_group()

    # Apply padding at seq_dim(dim 0）
    if pad_size > 0:
        q_ranges, k_ranges, attn_mask_type = apply_padding(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen=total_seqlen_q,
            pad_size=pad_size,
        )

        total_seqlen_q += pad_size
        total_seqlen_k += pad_size

    key = DistAttnRuntimeKey(
        cp_group,  # FIXME: ignore cp_mesh to be part of key for now
        chunk_size,
        pad_size,
        q_ranges,
        k_ranges,
        attn_mask_type,
        total_seqlen_q,
        total_seqlen_k,
        dist_attn_config,
    )

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

    return key


def magi_attn_flex_dispatch(
    x: torch.Tensor,
    q_ranges: AttnRanges,
    k_ranges: AttnRanges,
    attn_mask_type: GeneralAttnMaskType,
    total_seqlen_q: int,
    total_seqlen_k: int,
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
    caculate DistAttnRuntimeKey, generate the corr. inner DistAttnRuntimeMgr,
    finally pad and dispatch the input tensor to local tensor.

    Args:
        x (torch.Tensor): input tensor
        q_ranges (AttnRanges): global query ranges in the ref attn mask
        k_ranges (AttnRanges): global key ranges in the ref attn mask
        attn_mask_type (str | AttnMaskType | list[str | AttnMaskType]): attn mask type (list)
            represented by str or enum ``AttnMaskType`` or their mixed combination

        total_seqlen_q (int): the total seqlen of query (i.e. number of rows in the ref attn mask)
        total_seqlen_k (int): the total seqlen of key (i.e. number of columns in the ref attn mask)

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group (dist.ProcessGroup): process group, only support nccl backend for now
        cp_mesh (DeviceMesh): process mesh, only support 1D or 2D mesh for now.
            **NOTE**: cp_group and cp_mesh are mutually exclusive, one and only one of them needs be provided.

        dist_attn_config (DistAttnConfig): dist attn config

        is_same_source (bool): is query tensor and key tensor share the same source
        is_q_permutable (bool): is query tensor permutable
        is_k_permutable (bool): is key tensor permutable

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
        ...     pad_size=compute_pad_size(4096, 4, 512),  # seqlen, cp_size, chun_size
        ...     chunk_size=512,
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
    key = magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
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

    local_x = dispatch(x, key)
    return (local_x, key)


def dispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> torch.Tensor:
    """
    Pad and dispatch the input tensor to local tensor on each rank along the seqlen dim.

    Args:
        x (torch.Tensor): input total tensor.
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    Returns:
        torch.Tensor: the padded and dispatched local tensor.

    Raises:
        ValueError: If the provided ``key`` does not exist in ``DistAttnRuntimeDict``.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("The DistAttnRuntimeKey does not exist!")

    pad_size = key.pad_size
    padded_x = pad_at_dim(x, 0, pad_size)

    return mgr.dispatch_qo(padded_x)


def undispatch(
    x: torch.Tensor,
    key: DistAttnRuntimeKey,
) -> torch.Tensor:
    """
    Undispatch and unpad the local tensor to global tensor along the seqlen dim.

    Args:
        x (torch.Tensor): local tensor
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    Returns:
        torch.Tensor: the undispatched and unpadded tensor.

    Raises:
        ValueError: If the provided ``key`` does not exist in ``DistAttnRuntimeDict``.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("The DistAttnRuntimeKey does not exist!")

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
        q (torch.Tensor): Query tensor of shape ``(num_tokens_q, num_heads, head_dim)``.
        k (torch.Tensor): Key tensor of shape ``(num_tokens_k, num_heads, head_dim)``.
        v (torch.Tensor): Value tensor of shape ``(num_tokens_v, num_heads, head_dim)``.
        key (DistAttnRuntimeKey): the object that holds some inner meta data
            as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - out (torch.Tensor): Attention output tensor of shape.
            - lse (torch.Tensor): Log-sum-exp values for numerical stability.

    Raises:
        ValueError: If the provided ``key`` does not exist in ``DistAttnRuntimeDict``.
    """
    mgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("The DistAttnRuntimeKey does not exist!")

    return mgr.calc_attn(q, k, v)


def get_position_ids(key: DistAttnRuntimeKey) -> torch.Tensor:
    """
    Get the position ids of local tensor to global tensor after dispatching.

    Args:
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as one argument for many other magi_attention APIs, about which the users may have no bother to care.

    Returns:
        torch.Tensor: postion ids of local tensor w.r.t. global tensor.
    """
    mgr: DistAttnRuntimeMgr = DistAttnRuntimeDict.get(key)
    if mgr is None:
        raise ValueError("The DistAttnRuntimeKey does not exist!")

    return mgr.get_position_ids()


def get_most_recent_key() -> DistAttnRuntimeKey:
    """Get the most recent inserted key.

    This is useful when you can not access the key through the arguments,
    and meanwhile you only need the most recent inserted key.
    However, we strongly recommend you to access the key passed through the arguments,
    in case of unexpected inconsistency.

    Returns:
        DistAttnRuntimeKey: the most recent inserted key.
    """
    return DistAttnRuntimeDict.get_most_recent_key()
