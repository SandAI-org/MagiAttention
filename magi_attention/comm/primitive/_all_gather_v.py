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

from magi_attention.utils import nvtx

__all__ = ["all_gather_v"]


def _trans_with_dim0(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    is_first_dim = dim == 0 or (dim == -1 and len(x.shape) == 1)

    if not is_first_dim:
        x = x.transpose(0, dim)
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def _split_dims(
    x_shape: list[int],
    dim: int = 0,
) -> tuple[int, list[int]]:
    shape_len = len(x_shape)
    assert dim == -1 or 0 <= dim < len(
        x_shape
    ), f"dim should be in [0, {shape_len - 1}) or -1"

    this_dim = x_shape[dim]

    other_dims = x_shape.copy()
    other_dims[0] = this_dim
    other_dims[dim] = x_shape[0]
    other_dims = other_dims[1:]

    return this_dim, other_dims


@torch.no_grad()
@nvtx.instrument_nvtx
def all_gather_v(
    x_local: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = 0,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    """All-gather the local tensor 'x_local' along its dim,
    and return the gathered tensor 'x_gather',
    if not equally split along the dim, then gather indicated by the split sizes

    NOTE: this primitive only supports sync op mode for now

    Args:
        x_local (torch.Tensor): the local tensor to be gathered
        group (dist.ProcessGroup): the process group to be used
        dim (int): the dim to be gathered along. Defaults to 0.
        split_sizes (list[int] | None): the split sizes along the dim,
            where len(split_sizes) should equal to the world size of the group,
                and split_sizes[rank] is the dim size of this local tensor,
                and sum(split_sizes) should equal to the dim size of the global tensor

            NOTE: if None, then all local tensors should share the same shape

    Returns:
        torch.Tensor: the gathered tensor 'x_gather'
    """

    world_size = dist.get_world_size(group)
    this_dim, other_dims = _split_dims(list(x_local.shape), dim)

    x_local = _trans_with_dim0(x_local, dim)

    if split_sizes is None:  # all local tensors share the same shape
        x_gather = torch.empty(
            [this_dim * world_size] + other_dims,
            dtype=x_local.dtype,
            device=x_local.device,
        )
        dist.all_gather_into_tensor(x_gather, x_local, group=group)  # all-gather
    else:  # each local tensor may have a different shape along the dim
        x_gather_list = [
            torch.empty(
                [split_sizes[r]] + other_dims,
                dtype=x_local.dtype,
                device=x_local.device,
            )
            for r in range(world_size)
        ]
        dist.all_gather(x_gather_list, x_local, group=group)  # all-gather-v
        x_gather = torch.cat(x_gather_list, dim=0)

    x_gather = _trans_with_dim0(x_gather, dim)

    return x_gather
