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

"""Serve the sm90 IndexSparse backward contract with ``ffa_bwd_sm100_index``.

Shared by ``test_index_sparse_bwd_sm100.py`` and
``exps/attn/cutedsl/run_index_sparse_sm100_benchmark.py`` so the test and the
bench agree on which sm90 inputs SM100 can serve today.

sm90 contract (``flex_flash_attn_func(index_sparse_indices=..., q_block_size=1)``):

- q / out / do ``(total_q, num_heads_q, d)``, k / v ``(total_k, num_heads_kv, d)``,
  lse ``(total_q, num_heads_q)`` fp32 natural log.
- ``index_sparse_indices`` int32 ``(total_q, num_heads_kv, topk)``: per q
  *token* and kv head, the K token ids (``sparse_k_block_size == 1``) or K block
  ids (block ``i`` covers K tokens ``[i * kbs, (i + 1) * kbs)``), -1 padded at
  the tail.

SM100 contract (``ffa_bwd_sm100_index``): one index row per 128-token Q tile
of a q head. The two meet when ``num_heads_q / num_heads_kv == 128``: the 128 q
heads of one (token, kv head) are refolded into one 128-row Q tile, so the sm90
index row of that token is exactly the SM100 index row of that tile.
"""

import math
from enum import Enum

import torch

from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import (
    FFABwdSm100Index,
    ffa_bwd_sm100_index,
)

SM100_INDEX_TILE_M = 128


class Sm100IndexGap(Enum):
    """sm90 IndexSparse features the SM100 backward cannot serve yet.

    Remove a member once the kernel (and the refold in
    ``ffa_bwd_sm100_index_sm90_layout``) supports it; the cases that needed it
    then run the numerical comparison instead of reporting a gap.
    """

    HEAD_DIM = "head_dim != 128: FFABwdSm100Index fixes head_dim at 128"
    Q_HEAD_GROUP = (
        "num_heads_q / num_heads_kv != 128: per-token indices only map onto the "
        "128-row Q tile when one token's q heads fill it exactly"
    )
    SPARSE_K_BLOCK = (
        "sparse_k_block_size > 1: the kernel gathers single K tokens, not K blocks"
    )


class Sm100IndexUnsupportedError(NotImplementedError):
    """Raised before any launch when the sm90 inputs need an ``Sm100IndexGap``."""

    def __init__(self, gaps: list[Sm100IndexGap]):
        self.gaps = tuple(gaps)
        super().__init__("; ".join(f"{g.name}: {g.value}" for g in self.gaps))


def sm100_index_bwd_gaps(
    *,
    head_dim: int,
    num_heads_q: int,
    num_heads_kv: int,
    sparse_k_block_size: int,
) -> list[Sm100IndexGap]:
    """The ``Sm100IndexGap`` members an sm90 IndexSparse backward call needs."""
    gaps = []
    if head_dim != FFABwdSm100Index.head_dim:
        gaps.append(Sm100IndexGap.HEAD_DIM)
    if num_heads_q != num_heads_kv * SM100_INDEX_TILE_M:
        gaps.append(Sm100IndexGap.Q_HEAD_GROUP)
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
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IndexSparse backward with sm90 tensors, run by ``ffa_bwd_sm100_index``.

    Assumes valid ids precede the -1 padding in every index row (both sm90
    index builders produce that). Returns (dq, dk, dv) in the sm90 layout.

    Raises:
        Sm100IndexUnsupportedError: the call needs an ``Sm100IndexGap``.
    """
    total_q, num_heads_q, head_dim = q.shape
    total_k, num_heads_kv, _ = k.shape
    gaps = sm100_index_bwd_gaps(
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        sparse_k_block_size=sparse_k_block_size,
    )
    if gaps:
        raise Sm100IndexUnsupportedError(gaps)
    assert index_sparse_indices.shape[:2] == (total_q, num_heads_kv)
    assert lse.shape == (total_q, num_heads_q)

    group = num_heads_q // num_heads_kv

    # (total_q, nhk * group, ...) -> (total_q * group, nhk, ...): each token's
    # group q heads of one kv head become the rows of one Q tile.
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
        tile_m=SM100_INDEX_TILE_M,
    )
    return unfold(dq), dk, dv


def index_sparse_ref_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    *,
    sparse_k_block_size: int = 1,
    softmax_scale: float | None = None,
    chunk_tokens: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 gather forward for the sm90 IndexSparse contract; returns (out, lse).

    Gather semantics: a K id listed twice is attended twice. Rows without a
    valid id get out = 0 and lse = -inf.
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
        ids = index_sparse_indices[t0:t1].long()
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
