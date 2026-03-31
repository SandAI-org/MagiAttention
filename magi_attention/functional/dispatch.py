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

from magi_attention.comm.primitive import all_gather_v
from magi_attention.common.enum import AttnType
from magi_attention.meta.collection import DispatchMeta
from magi_attention.utils import nvtx

__all__ = ["dispatch_func", "undispatch_func"]


def _select_local_chunks(
    x: torch.Tensor,
    meta: DispatchMeta,
    rank: int,
    seq_dim: int,
) -> torch.Tensor:
    """Split *x* into chunks and concat only the ones assigned to *rank*.

    ``torch.split`` / ``torch.chunk`` are view-only (zero-copy); the single
    ``torch.cat`` copies only O(shard_seqlen) data instead of the full sequence.
    """
    local_chunk_idxs = meta.partitions[rank]
    if meta.chunk_actual_sizes is not None:
        all_chunks = torch.split(x, meta.chunk_actual_sizes, dim=seq_dim)
    else:
        all_chunks = torch.chunk(x, chunks=meta.num_chunks, dim=seq_dim)
    return torch.cat([all_chunks[i] for i in local_chunk_idxs], dim=seq_dim)


def _gather_and_unpermute(
    x_local: torch.Tensor,
    group: dist.ProcessGroup,
    meta: DispatchMeta,
    seq_dim: int,
) -> torch.Tensor:
    """All-gather *x_local* from every rank, then reorder chunks back to the
    original (unpermuted) sequence order."""
    if meta.split_sizes is not None:
        x_perm = all_gather_v(x_local, group, dim=0, split_sizes=meta.split_sizes)
    else:
        x_perm = all_gather_v(x_local, group, dim=0)

    if meta.chunk_actual_sizes is not None:
        perm_sizes = [meta.chunk_actual_sizes[i] for i in meta.partitions_perm_idxs]
        perm_chunks = torch.split(x_perm, perm_sizes, dim=seq_dim)
    else:
        perm_chunks = torch.chunk(x_perm, meta.num_chunks, dim=seq_dim)

    return torch.cat(
        [perm_chunks[i] for i in meta.partitions_unperm_idxs],
        dim=seq_dim,
    )


def _permute_and_reduce_scatter(
    grad_global: torch.Tensor,
    group: dist.ProcessGroup,
    meta: DispatchMeta,
    seq_dim: int,
) -> torch.Tensor:
    """Permute chunks of *grad_global* into the dispatched order, then
    reduce-scatter along *seq_dim* so each rank receives the **sum** of
    its assigned chunks' gradients from all ranks.

    Each rank holds a *partial* grad_global (same shape, but only a partial
    contribution to the true gradient).  We need to:
      1. permute chunks to match the dispatched layout,
      2. split into per-rank pieces,
      3. reduce (sum) across ranks and scatter so each rank gets its piece.
    """
    if meta.chunk_actual_sizes is not None:
        orig_chunks = torch.split(grad_global, meta.chunk_actual_sizes, dim=seq_dim)
    else:
        orig_chunks = torch.chunk(grad_global, meta.num_chunks, dim=seq_dim)

    perm_grad = torch.cat(
        [orig_chunks[i] for i in meta.partitions_perm_idxs],
        dim=seq_dim,
    )

    world_size = dist.get_world_size(group)
    if meta.split_sizes is not None:
        input_splits = list(meta.split_sizes)
    else:
        per_rank = perm_grad.shape[seq_dim] // world_size
        input_splits = [per_rank] * world_size

    per_rank_chunks = list(torch.split(perm_grad, input_splits, dim=seq_dim))
    per_rank_chunks = [c.contiguous() for c in per_rank_chunks]

    rank = dist.get_rank(group)
    grad_local = torch.zeros_like(per_rank_chunks[rank])
    dist.reduce_scatter(grad_local, per_rank_chunks, op=dist.ReduceOp.SUM, group=group)

    return grad_local


class _DispatchFunc(torch.autograd.Function):
    """Fused dispatch: select-local-chunks in forward, gather-and-unpermute in
    backward.

    Compared with the previous concat-all-then-scatter approach, the forward
    only allocates O(shard_seqlen) rather than an O(total_seqlen) intermediate
    that stays alive as long as the local tensor is referenced.
    """

    @staticmethod
    def forward(ctx, x_global, group, meta, seq_dim):
        ctx.group = group
        ctx.meta = meta
        ctx.seq_dim = seq_dim
        rank = dist.get_rank(group)
        return _select_local_chunks(x_global, meta, rank, seq_dim)

    @staticmethod
    def backward(ctx, grad_local):  # pragma: no cover
        return (
            _gather_and_unpermute(grad_local, ctx.group, ctx.meta, ctx.seq_dim),
            None,
            None,
            None,
        )


class _UndispatchFunc(torch.autograd.Function):
    """Fused undispatch: gather-and-unpermute in forward, select-local-chunks in
    backward.

    The backward only allocates O(shard_seqlen) instead of reconstructing the
    full permuted tensor, mirroring the dispatch-forward optimisation.
    """

    @staticmethod
    def forward(ctx, x_local, group, meta, seq_dim):
        ctx.group = group
        ctx.meta = meta
        ctx.seq_dim = seq_dim
        return _gather_and_unpermute(x_local, group, meta, seq_dim)

    @staticmethod
    def backward(ctx, grad_global):  # pragma: no cover
        rank = dist.get_rank(ctx.group)
        return (
            _select_local_chunks(grad_global, ctx.meta, rank, ctx.seq_dim),
            None,
            None,
            None,
        )


class _UndispatchPartialGradFunc(torch.autograd.Function):
    """Same forward as _UndispatchFunc, but backward uses reduce_scatter
    to aggregate partial gradients across ranks.

    Use this when grad_global on each rank is a partial contribution
    (e.g., partial attention output gradient) that needs to be summed
    across ranks before scattering back.
    """

    @staticmethod
    def forward(ctx, x_local, group, meta, seq_dim):
        ctx.group = group
        ctx.meta = meta
        ctx.seq_dim = seq_dim
        return _gather_and_unpermute(x_local, group, meta, seq_dim)

    @staticmethod
    def backward(ctx, grad_global):  # pragma: no cover
        return (
            _permute_and_reduce_scatter(grad_global, ctx.group, ctx.meta, ctx.seq_dim),
            None,
            None,
            None,
        )


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

    return _DispatchFunc.apply(x_global, group, meta, seq_dim)


@nvtx.instrument_nvtx
def undispatch_func(
    x_local: torch.Tensor,
    meta: DispatchMeta,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
    is_partial_grad: bool = False,
) -> torch.Tensor:
    """Undispatch the local tensor 'x_local' along its sequence dim following the meta info,
    and return the undispatched global tensor 'x_global'

    Args:
        x_local (torch.Tensor): the local tensor to be undispatched
        group (dist.ProcessGroup): the process group to be used for communication
        meta (DispatchMeta): the meta info of the undispatch
        seq_dim (int): the sequence dimension of the tensor
        is_partial_grad (bool): when True, backward uses reduce_scatter to
            aggregate partial gradients across ranks instead of simply selecting
            local chunks. Defaults to False.

    Returns:
        torch.Tensor: the undispatched global tensor 'x_global'
    """

    # --------------      pre-check args       -------------- #

    assert (
        meta.attn_type is AttnType.SELF_ATTN
    ), f"We only support self-attention now, but got attn_type={meta.attn_type}"

    # --------------      undispatch       -------------- #

    if is_partial_grad:
        return _UndispatchPartialGradFunc.apply(x_local, group, meta, seq_dim)
    return _UndispatchFunc.apply(x_local, group, meta, seq_dim)
