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

"""Index-sparse (DSA) attention helpers shared by the kernels, tests and benchmarks.

``partition_index_sparse_rows_by_k_chunk`` groups the ids of every index row by
K chunk for the K-chunked passes of ``ffa_bwd_sm100_index``.

The rest serves the sm90 IndexSparse contract with the SM100 forward / backward
and the SM120 forward, so the tests under ``tests/test_kernel/cutedsl/`` and the
benchmarks under ``exps/attn/cutedsl/`` agree on which sm90 inputs each kernel
can serve today.

sm90 contract (``flex_flash_attn_func(index_sparse_indices=..., q_block_size=qbs)``):

- q / out / do ``(total_q, num_heads_q, d)``, k / v ``(total_k, num_heads_kv, d)``,
  lse ``(total_q, num_heads_q)`` fp32 natural log.
- ``index_sparse_indices`` int32 ``(ceil(total_q / qbs), num_heads_kv, topk)``:
  per block of ``qbs`` q tokens and kv head, the K token ids
  (``sparse_k_block_size == 1``) or K block ids (block ``i`` covers K tokens
  ``[i * kbs, (i + 1) * kbs)``), -1 padded at the tail.

SM100 forward contract (``ffa_fwd_sm100_index``): the sm90 layout itself, with
``q_block_size`` counted in q tokens, plus two limits: ``qbs * G == 128`` for the
q-head group ``G = num_heads_q / num_heads_kv`` (one index row fills one 128-row
Q tile) and ``topk`` a multiple of 4.

SM100 backward contract (``ffa_bwd_sm100_index``): one index row per
``q_block_size`` consecutive q rows of a q head. The two meet for any G: the G
q heads of one (token, kv head) are refolded into G consecutive q rows of one
kv head, so a block of ``qbs`` tokens becomes ``qbs * G`` consecutive rows and,
with ``q_block_size = qbs * G``, its sm90 index row is exactly the SM100 index
row of those rows. ``qbs * G == 128`` puts one q block in each 128-row Q tile,
the case with no wasted MMA rows.

SM120 forward contract (``ffa_fwd_sm120_index``): the layout and limits of the
SM100 forward. There is no SM120 index-sparse backward.
"""

import math
from collections.abc import Sequence
from enum import Enum

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ffa_bwd_sm100_index imports partition_index_sparse_rows_by_k_chunk from this
# module when it loads, so the gap checks and adapters below import the kernel
# modules in their bodies.


@triton.jit
def _partition_index_sparse_rows_by_k_chunk_kernel(
    indices_ptr,  # input: (num_rows, topk) int32, valid ids first, -1 padded
    out_ptr,  # output: (num_rows, topk) int32
    starts_ptr,  # output: (num_chunks, num_rows) int32
    lens_ptr,  # output: (num_chunks, num_rows) int32
    num_rows,
    topk,
    chunk_size,
    NUM_CHUNKS: tl.constexpr,
    NUM_CHUNKS_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    in_row = offs < topk
    chunk_ids = tl.arange(0, NUM_CHUNKS_POW2)
    # persistent over rows: one row is too little work for a CTA of its own
    for row in tl.range(tl.program_id(0), num_rows, tl.num_programs(0)):
        row64 = row.to(tl.int64)
        ids = tl.load(indices_ptr + row64 * topk + offs, mask=in_row, other=-1)
        # -1 padding goes past the last chunk, i.e. stays at the tail
        chunk = tl.where(ids >= 0, ids // chunk_size, NUM_CHUNKS)

        # Counting sort by chunk: a chunk's ids keep their order within the row.
        # One int64 scan ranks 4 chunks at once, one 16-bit count per chunk
        # (the caller bounds topk below 2**16).
        base = 0
        # per-chunk start / length, stored once per row (a scalar store would
        # be issued by every thread)
        starts = tl.zeros([NUM_CHUNKS_POW2], dtype=tl.int32)
        lens = tl.zeros([NUM_CHUNKS_POW2], dtype=tl.int32)
        # the -1 tail keeps its slot: everything before it is a valid id
        pos = offs
        for g in tl.static_range((NUM_CHUNKS + 3) // 4):
            in_group = (chunk >= 4 * g) & (chunk < min(4 * g + 4, NUM_CHUNKS)) & in_row
            shift = ((chunk - 4 * g) * 16).to(tl.int64)
            one_hot = tl.where(in_group, tl.full([BLOCK], 1, tl.int64) << shift, 0)
            rank = (tl.cumsum(one_hot, axis=0) >> shift) & 0xFFFF
            totals = tl.sum(one_hot, axis=0)
            start = tl.zeros([BLOCK], dtype=tl.int32)
            for j in tl.static_range(4):
                if 4 * g + j < NUM_CHUNKS:
                    count = ((totals >> (16 * j)) & 0xFFFF).to(tl.int32)
                    starts = tl.where(chunk_ids == 4 * g + j, base, starts)
                    lens = tl.where(chunk_ids == 4 * g + j, count, lens)
                    start = tl.where(chunk == 4 * g + j, base, start)
                    base += count
            pos = tl.where(in_group, start + rank.to(tl.int32) - 1, pos)
        tl.store(out_ptr + row64 * topk + pos, ids, mask=in_row)
        in_chunks = chunk_ids < NUM_CHUNKS
        tl.store(starts_ptr + chunk_ids * num_rows + row64, starts, mask=in_chunks)
        tl.store(lens_ptr + chunk_ids * num_rows + row64, lens, mask=in_chunks)


def partition_index_sparse_rows_by_k_chunk(
    index_sparse_indices: torch.Tensor,
    chunk_size: int,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the ids of every index row by K chunk ``id // chunk_size``.

    Args:
        index_sparse_indices: (num_q_blocks, nhk, topk) int32. Valid ids come
            first and lie in ``[0, num_chunks * chunk_size)``, the tail is -1.
        chunk_size: K tokens per chunk.
        num_chunks: number of chunks.

    Returns:
        partitioned: same shape as the input. Row ``r`` holds the ids of chunk 0,
            then chunk 1, ..., each chunk in its original order, then the -1 tail.
        starts / lens: (num_chunks, num_q_blocks, nhk) int32. Chunk ``c`` of row
            ``r`` is ``partitioned[r][starts[c][r] : starts[c][r] + lens[c][r]]``.
    """
    assert index_sparse_indices.dim() == 3
    assert index_sparse_indices.dtype == torch.int32
    assert chunk_size >= 1 and num_chunks >= 1
    indices = index_sparse_indices.contiguous()
    num_q_blocks, nhk, topk = indices.shape
    # the kernel counts each chunk's ids in a 16-bit field
    assert topk < 2**16
    num_rows = num_q_blocks * nhk
    partitioned = torch.empty_like(indices)
    starts = torch.empty(
        num_chunks, num_q_blocks, nhk, dtype=torch.int32, device=indices.device
    )
    lens = torch.empty_like(starts)
    if num_rows == 0 or topk == 0:
        starts.zero_()
        lens.zero_()
        return partitioned, starts, lens
    block = triton.next_power_of_2(topk)
    num_sms = torch.cuda.get_device_properties(indices.device).multi_processor_count
    _partition_index_sparse_rows_by_k_chunk_kernel[(min(num_rows, 8 * num_sms),)](
        indices,
        partitioned,
        starts,
        lens,
        num_rows,
        topk,
        chunk_size,
        NUM_CHUNKS=num_chunks,
        NUM_CHUNKS_POW2=triton.next_power_of_2(num_chunks),
        BLOCK=block,
        num_warps=4,
    )
    return partitioned, starts, lens


class Sm100IndexGap(Enum):
    """sm90 IndexSparse features the SM100 forward or backward cannot serve yet.

    Remove a member once the kernels (and the adapters
    ``ffa_fwd_sm100_index_sm90_layout`` / ``ffa_bwd_sm100_index_sm90_layout``)
    support it; the cases that needed it then run the numerical comparison
    instead of reporting a gap.
    """

    HEAD_DIM = (
        "head_dim != 128: FFAFwdSm100Index and FFABwdSm100Index fix head_dim at 128"
    )
    SPARSE_K_BLOCK = (
        "sparse_k_block_size > 1: the kernels gather single K tokens, not K blocks"
    )
    Q_TILE_FILL = (
        "q_block_size * group != 128: FFAFwdSm100Index maps one index row onto "
        "one 128-row Q tile"
    )


class Sm100IndexUnsupportedError(NotImplementedError):
    """Raised before any launch when the sm90 inputs need a kernel gap member.

    ``gaps`` holds ``Sm100IndexGap`` members, or the gap members of an architecture
    whose error derives from this one so the shared case runner reports it as GAP.
    """

    def __init__(self, gaps: Sequence[Enum]):
        self.gaps = tuple(gaps)
        super().__init__("; ".join(f"{g.name}: {g.value}" for g in self.gaps))


def sm100_index_fwd_gaps(
    *,
    head_dim: int,
    sparse_k_block_size: int,
    q_block_size: int,
    group: int,
) -> list[Sm100IndexGap]:
    """The ``Sm100IndexGap`` members an sm90 IndexSparse forward call needs.

    ``q_block_size`` is the sm90 one (q tokens per index row) and ``group`` is
    ``num_heads_q / num_heads_kv``.
    """
    from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import FFAFwdSm100Index

    gaps = []
    if head_dim != FFAFwdSm100Index.head_dim:
        gaps.append(Sm100IndexGap.HEAD_DIM)
    if sparse_k_block_size != 1:
        gaps.append(Sm100IndexGap.SPARSE_K_BLOCK)
    if q_block_size * group != FFAFwdSm100Index.tile_m:
        gaps.append(Sm100IndexGap.Q_TILE_FILL)
    return gaps


def ffa_fwd_sm100_index_sm90_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    assume_valid_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IndexSparse forward with sm90 tensors, run by ``ffa_fwd_sm100_index``.

    ``q_block_size`` is the sm90 one: q tokens per index row. The tensors reach
    the kernel unchanged, except that an index whose ``topk`` is not a multiple
    of ``FFAFwdSm100Index.topk_align`` is copied with -1 columns appended.
    ``assume_valid_ids`` is forwarded to ``ffa_fwd_sm100_index`` and requires an
    aligned ``topk``, since the appended -1 columns would break its promise.
    Returns (out, lse) in the sm90 layout.

    Raises:
        Sm100IndexUnsupportedError: the call needs an ``Sm100IndexGap``.
    """
    from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import (
        FFAFwdSm100Index,
        ffa_fwd_sm100_index,
    )

    _, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape
    assert num_heads_q % num_heads_kv == 0
    gaps = sm100_index_fwd_gaps(
        head_dim=head_dim,
        sparse_k_block_size=sparse_k_block_size,
        q_block_size=q_block_size,
        group=num_heads_q // num_heads_kv,
    )
    if gaps:
        raise Sm100IndexUnsupportedError(gaps)

    topk_pad = -index_sparse_indices.shape[-1] % FFAFwdSm100Index.topk_align
    if topk_pad:
        assert not assume_valid_ids, (
            f"assume_valid_ids needs topk to be a multiple of "
            f"{FFAFwdSm100Index.topk_align}: aligning it appends -1 ids"
        )
        index_sparse_indices = F.pad(index_sparse_indices, (0, topk_pad), value=-1)
    return ffa_fwd_sm100_index(
        q,
        k,
        v,
        index_sparse_indices,
        softmax_scale=softmax_scale,
        q_block_size=q_block_size,
        assume_valid_ids=assume_valid_ids,
    )


def sm100_index_bwd_gaps(
    *,
    head_dim: int,
    sparse_k_block_size: int,
) -> list[Sm100IndexGap]:
    """The ``Sm100IndexGap`` members an sm90 IndexSparse backward call needs."""
    from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import FFABwdSm100Index

    gaps = []
    if head_dim != FFABwdSm100Index.head_dim:
        gaps.append(Sm100IndexGap.HEAD_DIM)
    if sparse_k_block_size != 1:
        gaps.append(Sm100IndexGap.SPARSE_K_BLOCK)
    return gaps


def ffa_bwd_sm100_index_sm90_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    gather_v_async: bool | None = None,
    num_k_passes: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IndexSparse backward with sm90 tensors, run by ``ffa_bwd_sm100_index``.

    ``q_block_size`` is the sm90 one: q tokens per index row. Assumes valid ids
    precede the -1 padding in every index row (both sm90 index builders produce
    that). ``gather_v_async`` and ``num_k_passes`` are forwarded as is.
    Returns (dq, dk, dv) in the sm90 layout.

    Raises:
        Sm100IndexUnsupportedError: the call needs an ``Sm100IndexGap``.
    """
    from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import ffa_bwd_sm100_index

    total_q, num_heads_q, head_dim = q.shape
    total_k, num_heads_kv, _ = k.shape
    gaps = sm100_index_bwd_gaps(
        head_dim=head_dim, sparse_k_block_size=sparse_k_block_size
    )
    if gaps:
        raise Sm100IndexUnsupportedError(gaps)
    assert num_heads_q % num_heads_kv == 0
    num_q_blocks = (total_q + q_block_size - 1) // q_block_size
    assert index_sparse_indices.shape[:2] == (num_q_blocks, num_heads_kv)
    assert lse.shape == (total_q, num_heads_q)

    group = num_heads_q // num_heads_kv

    # (total_q, nhk * group, ...) -> (total_q * group, nhk, ...): each token's
    # group q heads of one kv head become group consecutive q rows, so a q
    # block of q_block_size tokens becomes q_block_size * group rows.
    # A view when num_heads_kv == 1, a copy otherwise. The view must not go
    # through transpose: that leaves a non-unit stride on the size-1 head dim,
    # which the kernel rejects for lse.
    def fold(x: torch.Tensor) -> torch.Tensor:
        rest = x.shape[2:]
        if num_heads_kv == 1:
            return x.reshape(total_q * group, 1, *rest)
        x = x.view(total_q, num_heads_kv, group, *rest).transpose(1, 2)
        return x.reshape(total_q * group, num_heads_kv, *rest)

    def unfold(x: torch.Tensor) -> torch.Tensor:
        rest = x.shape[2:]
        if num_heads_kv == 1:
            return x.reshape(total_q, num_heads_q, *rest)
        x = x.view(total_q, group, num_heads_kv, *rest).transpose(1, 2)
        return x.reshape(total_q, num_heads_q, *rest)

    dq, dk, dv = ffa_bwd_sm100_index(
        fold(q),
        k,
        v,
        fold(out),
        fold(do),
        fold(lse),
        index_sparse_indices,
        softmax_scale=softmax_scale,
        q_block_size=q_block_size * group,
        gather_v_async=gather_v_async,
        num_k_passes=num_k_passes,
    )
    return unfold(dq), dk, dv


class Sm120IndexGap(Enum):
    """sm90 IndexSparse forward features the SM120 forward cannot serve yet.

    Remove a member once the kernel (and ``ffa_fwd_sm120_index_sm90_layout``)
    supports it; the cases that needed it then run the numerical comparison
    instead of reporting a gap.
    """

    HEAD_DIM = "head_dim != 128: FFAFwdSm120Index fixes head_dim at 128"
    SPARSE_K_BLOCK = (
        "sparse_k_block_size > 1: the kernel gathers single K tokens, not K blocks"
    )
    Q_TILE_FILL = (
        "q_block_size * group != 128: FFAFwdSm120Index maps one index row onto "
        "one 128-row Q tile"
    )


class Sm120IndexUnsupportedError(Sm100IndexUnsupportedError):
    """Raised before any launch when the sm90 inputs need an ``Sm120IndexGap``.

    Derives from the SM100 error so the shared case runner of
    ``test_index_sparse_bwd_sm100.py`` reports such a case as GAP.
    """


def sm120_index_fwd_gaps(
    *,
    head_dim: int,
    sparse_k_block_size: int,
    q_block_size: int,
    group: int,
) -> list[Sm120IndexGap]:
    """The ``Sm120IndexGap`` members an sm90 IndexSparse forward call needs.

    ``q_block_size`` is the sm90 one (q tokens per index row) and ``group`` is
    ``num_heads_q / num_heads_kv``.
    """
    from magi_attention.kernel.cutedsl.ffa_fwd_sm120_index import FFAFwdSm120Index

    gaps = []
    if head_dim != FFAFwdSm120Index.head_dim:
        gaps.append(Sm120IndexGap.HEAD_DIM)
    if sparse_k_block_size != 1:
        gaps.append(Sm120IndexGap.SPARSE_K_BLOCK)
    if q_block_size * group != FFAFwdSm120Index.tile_m:
        gaps.append(Sm120IndexGap.Q_TILE_FILL)
    return gaps


def ffa_fwd_sm120_index_sm90_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    assume_valid_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """IndexSparse forward with sm90 tensors, run by ``ffa_fwd_sm120_index``.

    ``q_block_size`` is the sm90 one: q tokens per index row. The tensors reach
    the kernel unchanged, except that an index whose ``topk`` is not a multiple
    of ``FFAFwdSm120Index.topk_align`` is copied with -1 columns appended.
    ``assume_valid_ids`` is forwarded to ``ffa_fwd_sm120_index`` and requires an
    aligned ``topk``, since the appended -1 columns would break its promise.
    Returns (out, lse) in the sm90 layout.

    Raises:
        Sm120IndexUnsupportedError: the call needs an ``Sm120IndexGap``.
    """
    from magi_attention.kernel.cutedsl.ffa_fwd_sm120_index import (
        FFAFwdSm120Index,
        ffa_fwd_sm120_index,
    )

    _, num_heads_q, head_dim = q.shape
    _, num_heads_kv, _ = k.shape
    assert num_heads_q % num_heads_kv == 0
    gaps = sm120_index_fwd_gaps(
        head_dim=head_dim,
        sparse_k_block_size=sparse_k_block_size,
        q_block_size=q_block_size,
        group=num_heads_q // num_heads_kv,
    )
    if gaps:
        raise Sm120IndexUnsupportedError(gaps)

    topk_pad = -index_sparse_indices.shape[-1] % FFAFwdSm120Index.topk_align
    if topk_pad:
        assert not assume_valid_ids, (
            f"assume_valid_ids needs topk to be a multiple of "
            f"{FFAFwdSm120Index.topk_align}: aligning it appends -1 ids"
        )
        index_sparse_indices = F.pad(index_sparse_indices, (0, topk_pad), value=-1)
    return ffa_fwd_sm120_index(
        q,
        k,
        v,
        index_sparse_indices,
        softmax_scale=softmax_scale,
        q_block_size=q_block_size,
        assume_valid_ids=assume_valid_ids,
    )


def index_sparse_ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    q_block_size: int = 1,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 gather forward for the sm90 IndexSparse contract; returns (out, lse).

    Token ``t`` uses index row ``t // q_block_size``. Gather semantics: a K id
    listed twice is attended twice. Rows without a valid id get out = 0 and
    lse = -inf.
    """
    total_q, num_heads_q, head_dim = q.shape
    num_heads_kv = k.shape[1]
    group = num_heads_q // num_heads_kv
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    lse = torch.empty(total_q, num_heads_q, dtype=torch.float32, device=q.device)
    head_kv = torch.arange(num_heads_kv, device=q.device)[None, :, None]
    block_offsets = torch.arange(sparse_k_block_size, device=q.device)
    for t0 in range(0, total_q, chunk_tokens):
        t1 = min(t0 + chunk_tokens, total_q)
        token_blocks = torch.arange(t0, t1, device=q.device) // q_block_size
        ids = index_sparse_indices[token_blocks].long()
        valid = ids >= 0
        if sparse_k_block_size > 1:
            ids = (ids[..., None] * sparse_k_block_size + block_offsets).flatten(-2)
            valid = valid.repeat_interleave(sparse_k_block_size, dim=-1)
        rows = ids.clamp(min=0)
        # k_sel / v_sel: (c, nhk, topk, d)
        k_sel = k[rows, head_kv].float()
        v_sel = v[rows, head_kv].float()
        q_c = q[t0:t1].float().view(t1 - t0, num_heads_kv, group, head_dim)
        s = torch.einsum("chgd,chtd->chgt", q_c, k_sel) * softmax_scale
        s = s.masked_fill(~valid[:, :, None, :], float("-inf"))
        lse_c = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - lse_c.masked_fill(lse_c.isneginf(), 0.0)[..., None])
        o = torch.einsum("chgt,chtd->chgd", p, v_sel)
        out[t0:t1] = o.reshape(t1 - t0, num_heads_q, head_dim).to(q.dtype)
        lse[t0:t1] = lse_c.reshape(t1 - t0, num_heads_q)
    return out, lse
