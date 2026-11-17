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

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from magi_attention.kernel.cutedsl.ffa_bwd_sm100_index import (
    FFABwdSm100Index,
    ffa_bwd_sm100_index,
)
from magi_attention.kernel.cutedsl.ffa_fwd_sm100_index import (
    FFAFwdSm100Index,
    ffa_fwd_sm100_index,
)
from magi_attention.kernel.cutedsl.ffa_fwd_sm120_index import (
    FFAFwdSm120Index,
    ffa_fwd_sm120_index,
)
from magi_attention.testing.precision import assert_close
from magi_attention.testing.sparse import (
    DEFAULT_BWD_DQ_ATOL,
    DEFAULT_FWD_ATOL,
    SEED,
    compare_sdpa_bwd_all,
    compare_sdpa_fwd,
    sdpa_ref_bwd_grads,
    sdpa_ref_output,
)
from magi_attention.utils import set_random_seed
from magi_attention.utils.sparse_utils import (
    build_index_sparse_indices,
    get_sdpa_mask_from_index_sparse_indices,
)

# Serve the sm90 IndexSparse contract with the SM100 forward / backward and the
# SM120 forward. Shared by the tests under tests/test_kernel/cutedsl/ and the
# benchmarks under exps/attn/cutedsl/ so they agree on which sm90 inputs SM100
# and SM120 can serve today.
#
# sm90 contract (flex_flash_attn_func(index_sparse_indices=..., q_block_size=qbs)):
# - q / out / do (total_q, num_heads_q, d), k / v (total_k, num_heads_kv, d),
#   lse (total_q, num_heads_q) fp32 natural log.
# - index_sparse_indices int32 (ceil(total_q / qbs), num_heads_kv, topk): per
#   block of qbs q tokens and kv head, the K token ids (sparse_k_block_size == 1)
#   or K block ids (block i covers K tokens [i * kbs, (i + 1) * kbs)), -1 padded
#   at the tail.
#
# SM100 forward contract (ffa_fwd_sm100_index): the sm90 layout itself, with
# q_block_size counted in q tokens, plus two limits: qbs * G == 128 for the
# q-head group G = num_heads_q / num_heads_kv (one index row fills one 128-row
# Q tile) and topk a multiple of 4.
#
# SM100 backward contract (ffa_bwd_sm100_index): one index row per q_block_size
# consecutive q rows of a q head. The two meet for any G: the G q heads of one
# (token, kv head) are refolded into G consecutive q rows of one kv head, so a
# block of qbs tokens becomes qbs * G consecutive rows and, with
# q_block_size = qbs * G, its sm90 index row is exactly the SM100 index row of
# those rows. qbs * G == 128 puts one q block in each 128-row Q tile, the case
# with no wasted MMA rows.
#
# SM120 forward contract (ffa_fwd_sm120_index): the sm90 layout itself, with
# the limits of the SM100 forward: head_dim == 128, qbs * G == 128 and topk a
# multiple of 4. There is no SM120 index-sparse backward.


# ═══════════════════════════════════════════════════════════
# SM100
# ═══════════════════════════════════════════════════════════


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
    index_sparse_rows_sorted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """IndexSparse backward with sm90 tensors, run by ``ffa_bwd_sm100_index``.

    ``q_block_size`` is the sm90 one: q tokens per index row. Assumes valid ids
    precede the -1 padding in every index row (both sm90 index builders produce
    that). ``gather_v_async``, ``num_k_passes`` and ``index_sparse_rows_sorted``
    are forwarded as is.
    Returns (dq, dk, dv) in the sm90 layout.

    Raises:
        Sm100IndexUnsupportedError: the call needs an ``Sm100IndexGap``.
    """
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
        index_sparse_rows_sorted=index_sparse_rows_sorted,
    )
    return unfold(dq), dk, dv


# ═══════════════════════════════════════════════════════════
# SM120
# ═══════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════
# Reference
# ═══════════════════════════════════════════════════════════


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


# ---------------------------------------------------------------------------
# Shared numerical case runner and sweep reports
# ---------------------------------------------------------------------------

BwdGrads = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


FwdOutLse = tuple[torch.Tensor, torch.Tensor]


FWD_LSE_ATOL = 1e-3


def run_index_sparse_config(
    device,
    cfg: dict[str, Any],
    bwd_fn: Callable[..., BwdGrads] | None = ffa_bwd_sm100_index_sm90_layout,
    fwd_fn: Callable[..., FwdOutLse] | None = None,
) -> None:
    """Run one sm90 index_sparse_indices config on SM100 and assert against SDPA.

    Mirrors ``run_index_sparse_config`` of the sm90 test. ``fwd_fn`` computes
    out / lse; ``None`` uses the fp32 gather ``index_sparse_ref_fwd``. Either
    way out is held to SDPA with the sm90 forward tolerance; the lse of a
    ``fwd_fn`` is also held to the gather's lse. ``bwd_fn`` computes dq / dk / dv
    from that out / lse and is held to the SDPA gradients; ``None`` checks the
    forward only.
    """
    set_random_seed(SEED)
    B = cfg["B"]
    S = cfg.get("S", None)
    S_kv = cfg.get("S_kv", S)
    S_q = cfg.get("S_q", min(S_kv, 256))
    NHQ = cfg["NHQ"]
    NHK = cfg["NHK"]
    D = cfg.get("D", 128)
    topk = cfg["topk"]
    default_max = max(topk) if isinstance(topk, list) else topk
    max_topk = cfg.get("max_topk", default_max)
    sparse_k_block_size = cfg.get("sparse_k_block_size", 1)
    # q tokens per index row (the sm90 q_block_size); the sm90 test always uses 1
    q_block_size = cfg.get("q_block_size", 1)
    dtype = cfg.get("dtype", torch.bfloat16)
    atol = cfg.get("atol", DEFAULT_FWD_ATOL)
    bwd_atol = cfg.get("bwd_atol", DEFAULT_BWD_DQ_ATOL)
    # The sm90 test flattens kv heads into the sequence; keep_kv_heads feeds
    # the kernel num_heads_kv > 1 instead (sm90 accepts both).
    flatten_kv_heads = not cfg.get("keep_kv_heads", False)
    # Flattening interleaves kv heads into the token axis, which would split a
    # q block of real tokens across index rows.
    assert q_block_size == 1 or NHK == 1 or not flatten_kv_heads
    assert S_q % q_block_size == 0

    q = torch.randn(B, S_q, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, NHK, D, dtype=dtype, device=device)

    if NHK > 1 and flatten_kv_heads:
        q = rearrange(q, "b s (h1 h2) d -> b (s h1) h2 d", h1=NHK)
        k = rearrange(k, "b s h d -> b (s h) 1 d")
        v = rearrange(v, "b s h d -> b (s h) 1 d")
        S_q = S_q * NHK
        S_kv = S_kv * NHK
        NHQ = NHQ // NHK
        NHK = 1

    # (B * S_q / q_block_size, NHK, max_topk): one row per q block
    index_sparse_indices = build_index_sparse_indices(
        B,
        NHK,
        S_q // q_block_size,
        S_kv,
        topk,
        max_topk,
        device,
        sparse_k_block_size=sparse_k_block_size,
    )

    # Token-major with native heads. With NHK == 1 (every flattened case) this
    # is exactly the sm90 SEQ_MAJOR packing (pack_q_for_ffa / pack_kv_for_ffa),
    # which cannot express NHK > 1.
    q_ffa = rearrange(q, "b s h d -> (b s) h d").contiguous()
    k_ffa = rearrange(k, "b s h d -> (b s) h d").contiguous()
    v_ffa = rearrange(v, "b s h d -> (b s) h d").contiguous()
    ref_out, ref_lse = index_sparse_ref_fwd(
        q_ffa,
        k_ffa,
        v_ffa,
        index_sparse_indices,
        q_block_size=q_block_size,
        sparse_k_block_size=sparse_k_block_size,
    )

    test_case = (
        f"[NHQ={cfg['NHQ']},NHK={cfg['NHK']},S_q={cfg.get('S_q', cfg.get('S'))},"
        f"S_kv={cfg.get('S_kv', cfg.get('S'))},"
        f"B={B},D={D},topk={topk},max_topk={max_topk},q_block_size={q_block_size},"
        f"sparse_k_block_size={sparse_k_block_size},dtype={dtype},"
        f"flat:NHQ_eff={NHQ},NHK_eff={NHK},S_q_eff={S_q},S_kv_eff={S_kv}]"
    )

    # fwd_fn / bwd_fn raise Sm100IndexUnsupportedError before any launch for a
    # gap case.
    if fwd_fn is None:
        out, lse = ref_out, ref_lse
    else:
        out, lse = fwd_fn(
            q_ffa,
            k_ffa,
            v_ffa,
            index_sparse_indices,
            q_block_size=q_block_size,
            sparse_k_block_size=sparse_k_block_size,
        )
    if bwd_fn is not None:
        do = torch.randn_like(out)
        dq_ffa, dk_ffa, dv_ffa = bwd_fn(
            q_ffa,
            k_ffa,
            v_ffa,
            out,
            do,
            lse,
            index_sparse_indices,
            q_block_size=q_block_size,
            sparse_k_block_size=sparse_k_block_size,
        )

    # The mask builder takes one index row per q token
    sdpa_mask = get_sdpa_mask_from_index_sparse_indices(
        index_sparse_indices.repeat_interleave(q_block_size, dim=0),
        B,
        NHQ,
        NHK,
        S_q,
        S_kv,
        device,
        sparse_k_block_size=sparse_k_block_size,
    )

    # The forward feeds the backward: hold it to the sm90 forward bar.
    o_ffa = rearrange(out, "(b s) h d -> b s h d", b=B)
    o_ref = sdpa_ref_output(q, k, v, sdpa_mask, B=B, NHQ=NHQ, NHK=NHK)
    if fwd_fn is None:
        compare_sdpa_fwd(o_ffa, o_ref, atol=atol, test_case=f"{test_case} ref_fwd")
    else:
        compare_sdpa_fwd(o_ffa, o_ref, atol=atol, test_case=test_case)
        assert_close(
            lse,
            ref_lse,
            atol=cfg.get("lse_atol", FWD_LSE_ATOL),
            rtol=0.0,
            test_case=f"{test_case} => lse",
        )
    if bwd_fn is None:
        return

    do_unpacked = rearrange(do, "(b s) h d -> b s h d", b=B)
    sdpa_dq, sdpa_dk, sdpa_dv = sdpa_ref_bwd_grads(
        do_unpacked, q, k, v, sdpa_mask, NHQ=NHQ, NHK=NHK
    )
    dq_ffa = rearrange(dq_ffa, "(b s) h d -> b h s d", b=B)
    dk_ffa = rearrange(dk_ffa, "(b s) h d -> b h s d", b=B)
    dv_ffa = rearrange(dv_ffa, "(b s) h d -> b h s d", b=B)
    compare_sdpa_bwd_all(
        ffa_dq=dq_ffa,
        ffa_dk=dk_ffa,
        ffa_dv=dv_ffa,
        sdpa_dq=sdpa_dq,
        sdpa_dk=sdpa_dk,
        sdpa_dv=sdpa_dv,
        test_case=test_case,
        dq_atol=bwd_atol,
    )


class CaseStatus(Enum):
    PASS = "PASS"
    GAP = "GAP"
    FAIL = "FAIL"


@dataclass
class CaseResult:
    name: str
    status: CaseStatus
    detail: str


def _run_case(
    device,
    name: str,
    cfg: dict[str, Any],
    *,
    bwd_fn: Callable[..., BwdGrads] | None,
    fwd_fn: Callable[..., FwdOutLse] | None,
) -> CaseResult:
    try:
        run_index_sparse_config(device, cfg, bwd_fn=bwd_fn, fwd_fn=fwd_fn)
    except Sm100IndexUnsupportedError as e:
        return CaseResult(name, CaseStatus.GAP, ",".join(g.name for g in e.gaps))
    except Exception as e:
        msg = str(e).strip().splitlines()
        return CaseResult(
            name, CaseStatus.FAIL, f"{type(e).__name__}: {msg[0] if msg else ''}"
        )
    finally:
        torch.cuda.empty_cache()
    return CaseResult(name, CaseStatus.PASS, "")


def run_index_sparse_sweep(
    device,
    title: str,
    cases: list[tuple[str, dict[str, Any]]],
    *,
    bwd_fn: Callable[..., BwdGrads] | None = ffa_bwd_sm100_index_sm90_layout,
    fwd_fn: Callable[..., FwdOutLse] | None = None,
) -> None:
    """Run every case (no fail-fast), print the table, then assert.

    ``bwd_fn`` / ``fwd_fn`` are passed to ``run_index_sparse_config``. Not
    ``@parameterize``: it stops at the first failure, which would hide the
    status of the remaining cases.
    """
    results = [
        _run_case(device, name, cfg, bwd_fn=bwd_fn, fwd_fn=fwd_fn)
        for name, cfg in cases
    ]

    width = max(len(r.name) for r in results)
    print(f"\n===== {title}: {len(results)} cases =====")
    for r in results:
        print(f"{r.status.value:<5} {r.name:<{width}}  {r.detail}")
    counts = {s: sum(r.status is s for r in results) for s in CaseStatus}
    print(" | ".join(f"{s.value} {n}" for s, n in counts.items()))

    fails = [r for r in results if r.status is CaseStatus.FAIL]
    if fails:
        raise AssertionError(
            f"{title}: {len(fails)} supported case(s) failed:\n"
            + "\n".join(f"  {r.name}: {r.detail}" for r in fails)
        )
