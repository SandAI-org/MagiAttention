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

import torch
import torch.distributed as dist

from magi_attention.comm.functional import (
    all_gather_fwd_scatter_bwd,
    scatter_fwd_all_gather_bwd,
)
from magi_attention.common.enum import AttnType
from magi_attention.meta.collection import DispatchMeta
from magi_attention.utils import nvtx


@nvtx.instrument_nvtx
def dispatch_func(
    x_global: torch.Tensor,
    group: dist.ProcessGroup,
    meta: DispatchMeta,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Dispatch the global tensor 'x_global' along its sequence dim following the meta info,
    and return the dispatched local tensor 'x_local'

    Args:
        x_global (torch.Tensor): the global tensor to be dispatched
        group (dist.ProcessGroup): the process group to be used for communication
        meta (DispatchMeta): the meta info of the dispatch
        seq_dim (int): the sequence dimension of the tensor

    Returns:
        torch.Tensor: the dispatched local tensor 'x_local'
    """

    # --------------      pre-check args       -------------- #

    assert (
        meta.attn_type is AttnType.SELF_ATTN
    ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

    # --------------      dispatch       -------------- #

    x_chunked = torch.chunk(
        x_global,
        chunks=meta.num_chunks,
        dim=seq_dim,
    )
    x_perm = torch.concat(
        [x_chunked[i] for i in meta.partitions_perm_idxs],
        dim=seq_dim,
    )
    x_local = scatter_fwd_all_gather_bwd(x_perm, group=group, dim=0)

    return x_local


@nvtx.instrument_nvtx
def undispatch_func(
    x_local: torch.Tensor,
    meta: DispatchMeta,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Undispatch the local tensor 'x_local' along its sequence dim following the meta info,
    and return the undispatched global tensor 'x_global'

    Args:
        x_local (torch.Tensor): the local tensor to be undispatched
        group (dist.ProcessGroup): the process group to be used for communication
        meta (DispatchMeta): the meta info of the undispatch
        seq_dim (int): the sequence dimension of the tensor

    Returns:
        torch.Tensor: the undispatched global tensor 'x_global'
    """

    # --------------      pre-check args       -------------- #

    assert (
        meta.attn_type is AttnType.SELF_ATTN
    ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

    # --------------      all-gather-v       -------------- #

    x_gather = all_gather_fwd_scatter_bwd(x_local, group=group, dim=0)

    # --------------      undispatch       -------------- #

    x_chunked = torch.chunk(
        x_gather,
        chunks=meta.num_chunks,
        dim=seq_dim,
    )
    x_global = torch.concat(
        [x_chunked[i] for i in meta.partitions_unperm_idxs],
        dim=seq_dim,
    )

    return x_global
