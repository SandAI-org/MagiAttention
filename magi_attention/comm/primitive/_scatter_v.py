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

import torch
import torch.distributed as dist

from magi_attention.utils import nvtx

__all__ = ["scatter_v"]


@torch.no_grad()
@nvtx.instrument_nvtx
def scatter_v(
    x_global: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = 0,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """Scatter the global tensor 'x_global' along its dim,
    and return the scattered tensor 'x_scatter',
    if not equally split along the dim, then scatter indicated by the split sizes

    NOTE: this primitive only supports sync op mode for now

    Args:
        x_global (torch.Tensor): the global tensor to be scattered
        group (dist.ProcessGroup): the process group to be used
        dim (int, optional): the dim to be scattered along. Defaults to 0.
        split_sizes (list[int] | None): the split sizes along the dim,
            where len(split_sizes) should equal to the world size of the group,
                and split_sizes[rank] is the dim size of this local tensor,
                and sum(split_sizes) should equal to the dim size of the global tensor,
            NOTE: if None, then all local tensors should share the same shape

    Returns:
        torch.Tensor: the scattered tensor 'x_scatter'
    """

    rank, world_size = dist.get_rank(group), dist.get_world_size(group)

    if split_sizes is None:  # all local tensors share the same shape
        x_scatter = torch.chunk(x_global, chunks=world_size, dim=dim)[rank]
    else:  # each local tensor may have a different shape along the dim
        x_scatter = torch.split(x_global, split_sizes, dim=dim)[rank]

    return x_scatter
