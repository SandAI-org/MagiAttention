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

from ..primitive import all_gather_v, scatter_v


class AllGatherFwdScatterBwd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        dim: int,
        split_sizes: list[int] | None = None,
    ):
        ctx.group = group
        ctx.dim = dim
        ctx.split_sizes = split_sizes
        return all_gather_v(input, group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pragma: no cover
        group: dist.ProcessGroup = ctx.group
        dim: int = ctx.dim
        split_sizes: list[int] | None = ctx.split_sizes
        return scatter_v(grad_output, group, dim, split_sizes), None, None, None


def all_gather_fwd_scatter_bwd(
    input: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    return AllGatherFwdScatterBwd.apply(input, group, dim, split_sizes)


class ScatterFwdAllGatherBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group, dim, split_sizes):
        ctx.group = group
        ctx.dim = dim
        ctx.split_sizes = split_sizes
        return scatter_v(input, group, dim, split_sizes)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pragma: no cover
        return (
            all_gather_v(grad_output, ctx.group, ctx.dim, ctx.split_sizes),
            None,
            None,
            None,
        )


def scatter_fwd_all_gather_bwd(
    input: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int,
    split_sizes: list[int] | None = None,
) -> torch.Tensor:
    return ScatterFwdAllGatherBwd.apply(input, group, dim, split_sizes)
