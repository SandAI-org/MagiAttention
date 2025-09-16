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

import magi_attention
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.config import DistAttnConfig
from magi_attention.dist_attn_runtime_mgr import (
    DistAttnRuntimeDict,
    DistAttnRuntimeKey,
    DistAttnRuntimeMgr,
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

dist_attn_runtime_dict = DistAttnRuntimeDict(
    max_size=magi_attention.dist_attn_runtime_dict_size()
)  # dict[DistAttnRuntimeKey, DistAttnRuntimeMgr]


GeneralAttnMaskType: TypeAlias = str | AttnMaskType | Sequence[str | AttnMaskType]


def magi_attn_varlen_key(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dist_attn_config: DistAttnConfig = DistAttnConfig(),
) -> DistAttnRuntimeKey:
    """This is a flash-attn-varlen like interface,
    to generate q_ranges, k_ranges and attn_mask_type
    from cu_seqlens_q, cu_seqlens_k and causal,
    caculate DistAttnRuntimeKey and generate the corr. inner DistAttnRuntimeMgr.

    Args:
        cu_seqlens_q (torch.Tensor): Cumulative sequence lengths for queries.
        cu_seqlens_k (torch.Tensor): Cumulative sequence lengths for keys.

        pad_size (int): the size to pad along seq_dim. The seq_len need to be divisable by chunk_size * cp_size,
        chunk_size (int): chunk size to chunk the input tensor x along the seqlen dim for dispatch
            to control the granularity of computation load-balance.

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        causal (bool, optional): whether the varlen attention mask is causal. Defaults to False.
        window_size (tuple[int, int], optional): window_size of sliding window mask
            which represents ``[window_size_left, window_size_right]``. The parameter is effective only
            when ``causal`` is ``False``; when ``causal`` is ``True``, it is required to be ``(-1, -1)``.
            Defaults to be ``(-1, -1)``.
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
    # infer q_ranges, k_ranges and others from cu_seqlens_q, cu_seqlens_k and causal
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

    # call magi_attn_flex_key
    # NOTE: for flash-attn-varlen, we assume
    # is_same_source, is_q_permutable and is_k_permutable are all True.
    return magi_attn_flex_key(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_mask_type=attn_mask_type,
        total_seqlen_q=total_seqlen_q,
        total_seqlen_k=total_seqlen_k,
        pad_size=pad_size,
        chunk_size=chunk_size,
        cp_group_or_mesh=cp_group_or_mesh,
        dist_attn_config=dist_attn_config,
        is_same_source=True,
        is_q_permutable=True,
        is_k_permutable=True,
    )


def magi_attn_varlen_dispatch(
    x: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
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

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

        causal (bool): if True, all attn_mask_type is CAUSAL. else, determine masktype with ``window_size``.
        window_size (tuple[int, int], optional): window_size of sliding window mask
            which represents ``[window_size_left, window_size_right]``. The parameter is effective only
            when ``causal`` is ``False``; when ``causal`` is ``True``, it is required to be ``(-1, -1)``.
            Defaults to be (-1, -1).
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
    pad_size: int,
    chunk_size: int,
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
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

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

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
        ...     cp_group_or_mesh=dist.new_group(list(range(4)), backend="nccl"),
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
        ...             alg=UniformOverlapAlg(),
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

    # init dist attn runtime key
    key = init_dist_attn_runtime_key(
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
    )

    # init dist attn runtime mgr and map it to the key
    if key not in dist_attn_runtime_dict.keys():
        dist_attn_runtime_dict[key] = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_mask_type=attn_mask_type,
            total_seqlen_q=total_seqlen_q,
            total_seqlen_k=total_seqlen_k,
            chunk_size=chunk_size,
            cp_group=cp_group,
            is_same_source=is_same_source,
            is_q_permutable=is_q_permutable,
            is_k_permutable=is_k_permutable,
            dist_attn_config=dist_attn_config,
            cp_mesh=cp_mesh,
        )

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
    cp_group_or_mesh: dist.ProcessGroup | DeviceMesh,
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

        cp_group_or_mesh (dist.ProcessGroup | DeviceMesh): process group or device mesh.
            **NOTE**: for process group, we only support nccl backend for now,
            and for device mesh, we only support 1D or 2D mesh for now.

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
        cp_group_or_mesh=cp_group_or_mesh,
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
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Pad and dispatch the global input tensor to local tensor on each rank along the seqlen dim.

    Args:
        x (torch.Tensor): global input tensor.
        key (DistAttnRuntimeKey): the key that holds some inner meta data,
            as one argument for many other magi_attention APIs, about which the users may have no bother to care.
        pad_value (float): the specific value to pad to input tensor. Defaults to 0.

    Returns:
        torch.Tensor: the padded and dispatched local tensor.

    Raises:
        ValueError: If the provided ``key`` does not exist in ``dist_attn_runtime_dict``.
    """
    mgr = dist_attn_runtime_dict.get(key)
    if mgr is None:
        raise ValueError("The DistAttnRuntimeKey does not exist!")

    pad_size = key.pad_size
    padded_x = pad_at_dim(x, 0, pad_size, value=pad_value)

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
        ValueError: If the provided ``key`` does not exist in ``dist_attn_runtime_dict``.
    """
    mgr = dist_attn_runtime_dict.get(key)
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
        ValueError: If the provided ``key`` does not exist in ``dist_attn_runtime_dict``.
    """
    mgr = dist_attn_runtime_dict.get(key)
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
    mgr: DistAttnRuntimeMgr = dist_attn_runtime_dict.get(key)
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
    return dist_attn_runtime_dict.get_most_recent_key()
