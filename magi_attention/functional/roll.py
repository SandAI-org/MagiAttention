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

from __future__ import annotations

import torch
import torch.distributed as dist

from magi_attention.meta.collection import DispatchMeta
from magi_attention.utils import nvtx


def _build_chunk_mappings(
    partitions: list[list[int]],
    num_chunks: int,
) -> tuple[list[int], list[int]]:
    """Return (chunk_to_rank, chunk_to_local_idx) derived from partitions."""
    chunk_to_rank = [0] * num_chunks
    chunk_to_local_idx = [0] * num_chunks
    for rank, chunks in enumerate(partitions):
        for local_idx, chunk_id in enumerate(chunks):
            chunk_to_rank[chunk_id] = rank
            chunk_to_local_idx[chunk_id] = local_idx
    return chunk_to_rank, chunk_to_local_idx


def _compute_segments(
    c_out: int,
    shift: int,
    chunk_sizes: list[int] | None,
    chunk_size: int,
    num_chunks: int,
    total_seqlen: int,
) -> list[tuple[int, int, int, int]]:
    """Return ``(src_chunk, src_offset, length, dst_offset)`` segments for *c_out*.

    For uniform chunks returns 1 (whole-chunk) or 2 (split) segments.
    For variable chunks (last chunk smaller) may return up to 3 when the
    source range wraps around the small last chunk.
    """
    if chunk_sizes is None:
        k, r = divmod(shift, chunk_size)
        if r == 0:
            src = (c_out - k) % num_chunks
            return [(src, 0, chunk_size, 0)]
        src_prev = (c_out - k - 1) % num_chunks
        src_curr = (c_out - k) % num_chunks
        return [
            (src_prev, chunk_size - r, r, 0),
            (src_curr, 0, chunk_size - r, r),
        ]

    # Variable: every chunk c starts at c * chunk_size in global coords
    # because only the last chunk is shorter than chunk_size.
    out_start = c_out * chunk_size
    out_size = chunk_sizes[c_out]
    src_global = (out_start - shift) % total_seqlen

    segments: list[tuple[int, int, int, int]] = []
    remaining = out_size
    dst_off = 0
    pos = src_global
    while remaining > 0:
        cid = min(pos // chunk_size, num_chunks - 1)
        off = pos - cid * chunk_size
        take = min(chunk_sizes[cid] - off, remaining)
        segments.append((cid, off, take, dst_off))
        remaining -= take
        dst_off += take
        pos = (pos + take) % total_seqlen
    return segments


def _roll_p2p_impl(
    x_local: torch.Tensor,
    shift: int,
    meta: DispatchMeta,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """P2P implementation of distributed roll.

    Instead of all-gather -> roll -> scatter, this directly exchanges only the
    needed chunk slices between ranks via point-to-point communication.

    For uniform chunks (shift = k * chunk_size + r, 0 <= r < chunk_size):
      - r == 0: output chunk c = input chunk (c-k) mod N  (whole-chunk transfer)
      - r >  0: output chunk c = tail-r of input chunk (c-k-1) mod N
                                  ++ head-(chunk_size-r) of input chunk (c-k) mod N

    For variable chunks (meta.chunk_actual_sizes is not None), each output
    chunk's source segments are computed from global coordinates, allowing
    the last chunk to be smaller than chunk_size.

    Send/recv buffers for each rank pair are ordered by iterating the
    *destination* rank's partition so that sender and receiver agree on the
    concatenation order without extra coordination.
    """
    my_rank = meta.cp_rank
    cp_size = meta.cp_size
    num_chunks = meta.num_chunks
    chunk_size = meta.chunk_size
    total_seqlen = meta.total_seqlen
    chunk_sizes = meta.chunk_actual_sizes

    shift = shift % total_seqlen
    if shift == 0:
        return x_local.clone()

    chunk_to_rank, chunk_to_local_idx = _build_chunk_mappings(
        meta.partitions, num_chunks
    )
    my_partition = meta.partitions[my_rank]

    if chunk_sizes is not None:
        local_split: int | list[int] = [chunk_sizes[c] for c in my_partition]
    else:
        local_split = chunk_size
    local_chunks = x_local.split(local_split, dim=seq_dim)
    output = torch.empty_like(x_local)
    output_chunks = output.split(local_split, dim=seq_dim)

    def segs(c_out: int) -> list[tuple[int, int, int, int]]:
        return _compute_segments(
            c_out, shift, chunk_sizes, chunk_size, num_chunks, total_seqlen,
        )

    # ---- Phase 1: local copies (no communication) ---- #

    for out_idx, c_out in enumerate(my_partition):
        for src_cid, src_off, seg_len, dst_off in segs(c_out):
            if chunk_to_rank[src_cid] == my_rank:
                output_chunks[out_idx].narrow(seq_dim, dst_off, seg_len).copy_(
                    local_chunks[chunk_to_local_idx[src_cid]].narrow(
                        seq_dim, src_off, seg_len
                    )
                )

    # ---- Phase 2: build matched send / recv buffers per remote rank ---- #

    p2p_ops: list[dist.P2POp] = []
    recv_scatter_info: list[tuple[torch.Tensor, list[torch.Tensor]]] = []

    for remote_rank in range(cp_size):
        if remote_rank == my_rank:
            continue

        # -- send: iterate *remote_rank*'s partition (matches remote's recv order) --
        send_pieces: list[torch.Tensor] = []
        for c_out in meta.partitions[remote_rank]:
            for src_cid, src_off, seg_len, _dst_off in segs(c_out):
                if chunk_to_rank[src_cid] == my_rank:
                    send_pieces.append(
                        local_chunks[chunk_to_local_idx[src_cid]].narrow(
                            seq_dim, src_off, seg_len
                        )
                    )

        # -- recv: iterate *my* partition (matches remote's send order to me) --
        recv_dest_slices: list[torch.Tensor] = []
        for out_idx, c_out in enumerate(my_partition):
            for src_cid, _src_off, seg_len, dst_off in segs(c_out):
                if chunk_to_rank[src_cid] == remote_rank:
                    recv_dest_slices.append(
                        output_chunks[out_idx].narrow(seq_dim, dst_off, seg_len)
                    )

        if send_pieces:
            send_buf = torch.cat(send_pieces, dim=seq_dim).contiguous()
            p2p_ops.append(
                dist.P2POp(dist.isend, send_buf, remote_rank, group=group)
            )

        if recv_dest_slices:
            total_len = sum(s.size(seq_dim) for s in recv_dest_slices)
            shape = list(x_local.shape)
            shape[seq_dim] = total_len
            recv_buf = torch.empty(
                shape, dtype=x_local.dtype, device=x_local.device
            )
            p2p_ops.append(
                dist.P2POp(dist.irecv, recv_buf, remote_rank, group=group)
            )
            recv_scatter_info.append((recv_buf, recv_dest_slices))

    # ---- Phase 3: execute P2P ---- #

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    # ---- Phase 4: scatter received concat buffers into output slices ---- #

    for recv_buf, dest_slices in recv_scatter_info:
        offset = 0
        for dest in dest_slices:
            length = dest.size(seq_dim)
            dest.copy_(recv_buf.narrow(seq_dim, offset, length))
            offset += length

    return output


class _RollP2P(torch.autograd.Function):
    """Autograd wrapper: forward rolls by +shift, backward rolls by -shift."""

    @staticmethod
    def forward(
        ctx,
        x_local: torch.Tensor,
        shift: int,
        meta: DispatchMeta,
        group: dist.ProcessGroup,
        seq_dim: int,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.meta = meta
        ctx.group = group
        ctx.seq_dim = seq_dim
        return _roll_p2p_impl(x_local, shift, meta, group, seq_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pragma: no cover
        return (
            _roll_p2p_impl(
                grad_output, -ctx.shift, ctx.meta, ctx.group, ctx.seq_dim
            ),
            None,
            None,
            None,
            None,
        )


@nvtx.instrument_nvtx
def roll_p2p(
    x_local: torch.Tensor,
    shift: int,
    meta: DispatchMeta,
    group: dist.ProcessGroup,
    seq_dim: int = 0,
) -> torch.Tensor:
    """Roll a dispatched local tensor along the sequence dimension using P2P.

    Compared to the naive undispatch-roll-dispatch path this avoids
    materialising the full global tensor, cutting peak memory from O(N) to
    O(N/P) and communication volume by ~P times.

    Args:
        x_local: Local tensor on this rank after dispatch.
        shift:   Positions to roll (positive = shift right, wraps cyclically).
        meta:    DispatchMeta describing the chunk partitioning.
        group:   Process group for communication.
        seq_dim: Sequence dimension to roll along.

    Returns:
        Rolled local tensor, same shape as *x_local*.
    """
    return _RollP2P.apply(x_local, shift, meta, group, seq_dim)
