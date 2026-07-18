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

"""Shared helpers for block_sparse and index_sparse attention tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func
from magi_attention.testing.precision import assert_close
from magi_attention.utils.sparse_utils import (
    build_index_sparse_indices,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
)

SEED = 42
DEFAULT_FWD_ATOL = 0.01
DEFAULT_FWD_RTOL = 0.05
DEFAULT_BWD_DQ_ATOL = 0.02
DEFAULT_BWD_DQ_RTOL = 0.3
DEFAULT_BWD_DKV_ATOL = 0.02
DEFAULT_BWD_DK_RTOL = 0.15
DEFAULT_BWD_DV_RTOL = 0.05
DEFAULT_MISMATCH_THRES = 0.01


class SparsePackLayout(Enum):
    """Q/K/V pack order for flex_flash_attn_func flat tensors."""

    HEAD_MAJOR = "head_major"  # block_sparse: (b h1 s) h2 d
    SEQ_MAJOR = "seq_major"  # index_sparse: (b s h1) h2 d


def pack_q_for_ffa(
    q: torch.Tensor,
    nhk: int,
    layout: SparsePackLayout = SparsePackLayout.SEQ_MAJOR,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    if layout == SparsePackLayout.HEAD_MAJOR:
        q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=nhk)
    else:
        q_ffa = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
    if requires_grad:
        return q_ffa.detach().clone().requires_grad_(True)
    if not q.requires_grad:
        return q_ffa.clone()
    return q_ffa


def pack_kv_for_ffa(
    k: torch.Tensor,
    v: torch.Tensor,
    layout: SparsePackLayout = SparsePackLayout.SEQ_MAJOR,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layout == SparsePackLayout.HEAD_MAJOR:
        k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
        v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")
    else:
        k_ffa = rearrange(k, "b s h d -> (b s h) 1 d")
        v_ffa = rearrange(v, "b s h d -> (b s h) 1 d")
    if requires_grad:
        k_ffa = k_ffa.detach().clone().requires_grad_(True)
        v_ffa = v_ffa.detach().clone().requires_grad_(True)
    return k_ffa, v_ffa


def unpack_ffa_output(
    o_sparse: torch.Tensor,
    *,
    B: int,
    S: int,
    NHK: int,
    layout: SparsePackLayout = SparsePackLayout.SEQ_MAJOR,
) -> torch.Tensor:
    if layout == SparsePackLayout.HEAD_MAJOR:
        return rearrange(o_sparse, "(b h1 s) h2 d -> b s (h1 h2) d", b=B, s=S, h1=NHK)
    return rearrange(o_sparse, "(b s h1) h2 d -> b s (h1 h2) d", b=B, h1=NHK, s=S)


@contextmanager
def inner_loop_env(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set inner-loop env vars for a test body."""
    for key, val in env.items():
        os.environ[key] = val
    try:
        yield
    finally:
        for key in env:
            os.environ.pop(key, None)


def sdpa_ref_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sdpa_mask: torch.Tensor,
    *,
    B: int,
    NHQ: int,
    NHK: int,
) -> torch.Tensor:
    """Compute SDPA reference output (B, S_q, NHQ, D).

    Runs per Q-head to avoid materializing the full (NHQ, S_q, S_kv) matrix
    for large MQA configs.
    sdpa_mask: (B, NHK, S_q, S_kv).
    q: (B, S_q, NHQ, D), k: (B, S_kv, NHK, D), v: (B, S_kv, NHK, D).
    """
    gqa = NHQ // NHK
    o_ref_all = []
    with torch.no_grad():
        for b_idx in range(B):
            o_ref_heads = []
            for h_q in range(NHQ):
                h_kv = h_q // gqa
                q_h = q[b_idx, :, h_q : h_q + 1, :].unsqueeze(0).transpose(1, 2)
                k_h = k[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                v_h = v[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                mask_h = sdpa_mask[b_idx, h_kv : h_kv + 1].unsqueeze(0)
                o_h = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=mask_h)
                o_ref_heads.append(o_h)
            o_batch = torch.cat(o_ref_heads, dim=1)
            o_ref_all.append(rearrange(o_batch, "1 h s d -> s h d"))
    return torch.stack(o_ref_all, dim=0)


def compare_sdpa_fwd(
    o_ffa: torch.Tensor,
    o_ref: torch.Tensor,
    *,
    test_case: str,
    atol: float = DEFAULT_FWD_ATOL,
    rtol: float = DEFAULT_FWD_RTOL,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRES,
) -> None:
    """Compare FFA forward output against pre-computed SDPA reference."""
    assert_close(
        o_ffa,
        o_ref,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => fwd_out",
    )


def sdpa_ref_bwd_grads(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sdpa_mask: torch.Tensor,
    *,
    NHQ: int,
    NHK: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SDPA backward gradients.

    Inputs in (B, S, H, D) layout. sdpa_mask: (B, NHK, S_q, S_kv).
    Returns (dq, dk, dv) each in (B, H, S, D) layout (transposed).
    """
    gqa = NHQ // NHK
    q_sdpa = q.transpose(1, 2).detach().clone().requires_grad_(True)
    k_sdpa = k.transpose(1, 2).detach().clone().requires_grad_(True)
    v_sdpa = v.transpose(1, 2).detach().clone().requires_grad_(True)

    k_expanded = k_sdpa.repeat_interleave(gqa, dim=1)
    v_expanded = v_sdpa.repeat_interleave(gqa, dim=1)
    mask_expanded = sdpa_mask.repeat_interleave(gqa, dim=1)

    o_sdpa = F.scaled_dot_product_attention(
        q_sdpa, k_expanded, v_expanded, attn_mask=mask_expanded
    )
    do_sdpa = do.transpose(1, 2)
    o_sdpa.backward(do_sdpa)
    return q_sdpa.grad, k_sdpa.grad, v_sdpa.grad


def compare_sdpa_bwd_all(
    ffa_dq: torch.Tensor | None = None,
    ffa_dk: torch.Tensor | None = None,
    ffa_dv: torch.Tensor | None = None,
    sdpa_dq: torch.Tensor | None = None,
    sdpa_dk: torch.Tensor | None = None,
    sdpa_dv: torch.Tensor | None = None,
    *,
    test_case: str,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRES,
    dq_atol: float = DEFAULT_BWD_DQ_ATOL,
    dq_rtol: float = DEFAULT_BWD_DQ_RTOL,
    dk_atol: float = DEFAULT_BWD_DKV_ATOL,
    dk_rtol: float = DEFAULT_BWD_DK_RTOL,
    dv_atol: float = DEFAULT_BWD_DKV_ATOL,
    dv_rtol: float = DEFAULT_BWD_DV_RTOL,
) -> None:
    """Compare FFA dQ/dK/dV against SDPA reference using assert_close.

    Any pair of (ffa_dx, sdpa_dx) that is None is silently skipped.
    """
    for ffa, ref, atol, rtol, label in [
        (ffa_dq, sdpa_dq, dq_atol, dq_rtol, "dq"),
        (ffa_dk, sdpa_dk, dk_atol, dk_rtol, "dk"),
        (ffa_dv, sdpa_dv, dv_atol, dv_rtol, "dv"),
    ]:
        if ffa is not None and ref is not None:
            assert_close(
                ffa,
                ref,
                atol=atol,
                rtol=rtol,
                mismatch_threshold=mismatch_threshold,
                test_case=f"{test_case} => {label}",
            )


def check_ffa_deterministic_twice(
    run_once: Callable[[], tuple[torch.Tensor, ...]],
    *,
    test_case: str,
    dq_atol: float | None = 2e-3,
) -> list[str]:
    """Run FFA twice in deterministic mode and compare all returned tensors.

    *dq_atol*: tolerance for dQ comparison (sparse dQ may have tiny FP
    differences from non-associative TMA reduce-add across CTAs).
    """
    results1 = run_once()
    results2 = run_once()

    labels = ["fwd_out", "dq", "dk", "dv"]
    err_msgs: list[str] = []
    for i, (t1, t2) in enumerate(zip(results1, results2)):
        if t1 is None or t2 is None:
            continue
        label = labels[i] if i < len(labels) else f"tensor_{i}"
        tol = dq_atol if label == "dq" and dq_atol is not None else None
        if tol is not None:
            if not torch.allclose(t1, t2, atol=tol, rtol=0):
                max_diff = (t1 - t2).abs().max().item()
                err_msgs.append(
                    f"For {test_case=}: {label} not deterministic "
                    f"(max_diff={max_diff:.2e} > atol={tol})"
                )
        elif not torch.equal(t1, t2):
            err_msgs.append(f"For {test_case=}: {label} not deterministic")
    return err_msgs


# ═══════════════════════════════════════════════════════════
# Closure builders for assert_deterministic / assert_overlap_safe
# ═══════════════════════════════════════════════════════════


def build_block_sparse_ffa_fn(
    device: Any,
    *,
    nhq: int = 128,
    nhk: int = 1,
    hd: int = 128,
    kbs: int = 128,
    sparsity: float = 0.2,
    seqlen: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = False,
    include_bwd: bool = False,
    swap_bwd_qk_loop: bool | None = None,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Build a zero-arg callable that runs block_sparse FFA once.

    Returns a closure suitable for :func:`assert_deterministic` or
    :func:`assert_overlap_safe`.
    """
    torch.manual_seed(SEED)
    num_q_blocks = seqlen
    num_kv_blocks = seqlen // kbs
    block_mask, _ = generate_block_sparse_pattern(
        num_q_heads=nhq,
        num_kv_heads=nhk,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        sparsity=sparsity,
        mode="per_kv_head",
        sparse_format="block_mask",
        device=device,
    )
    q_ranges, k_ranges = generate_ranges_from_block_mask_triton(block_mask, 1, kbs)
    attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

    q0 = torch.randn(1, seqlen, nhq, hd, dtype=dtype, device=device)
    k0 = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=device)
    v0 = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=device)
    do0 = (
        torch.randn(1, seqlen, nhq, hd, dtype=dtype, device=device)
        if include_bwd
        else None
    )

    q_packed = pack_q_for_ffa(q0, nhk, SparsePackLayout.HEAD_MAJOR)
    k_packed, v_packed = pack_kv_for_ffa(k0, v0, SparsePackLayout.HEAD_MAJOR)
    do_packed = (
        rearrange(do0, "b s (h1 h2) d -> (b h1 s) h2 d", h1=nhk)
        if do0 is not None
        else None
    )

    def fn() -> tuple[torch.Tensor, ...]:
        q = q_packed.clone().requires_grad_(include_bwd)
        k = k_packed.clone().requires_grad_(include_bwd)
        v = v_packed.clone().requires_grad_(include_bwd)
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            max_seqlen_q=1,
            attn_type_map=attn_type_map,
            range_merge=True,
            pack_gqa=True,
            block_sparse=True,
            deterministic=deterministic,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )
        if include_bwd:
            o.backward(do_packed)
            return o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()
        return (o.detach(),)

    return fn


def build_index_sparse_ffa_fn(
    device: Any,
    *,
    nhq: int = 128,
    nhk: int = 1,
    hd: int = 128,
    kbs: int = 1,
    topk: int = 128,
    seqlen: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = False,
    include_bwd: bool = False,
    swap_bwd_qk_loop: bool | None = None,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    """Build a zero-arg callable that runs index_sparse FFA once.

    Returns a closure suitable for :func:`assert_deterministic` or
    :func:`assert_overlap_safe`.
    """
    torch.manual_seed(SEED)
    q0 = torch.randn(1, seqlen, nhq, hd, dtype=dtype, device=device)
    k0 = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=device)
    v0 = torch.randn(1, seqlen, nhk, hd, dtype=dtype, device=device)

    index_sparse_indices = build_index_sparse_indices(
        1,
        nhk,
        seqlen,
        seqlen,
        topk,
        topk,
        device,
        sparse_k_block_size=kbs,
    )

    q_packed = pack_q_for_ffa(q0, nhk, SparsePackLayout.SEQ_MAJOR)
    k_packed, v_packed = pack_kv_for_ffa(k0, v0, SparsePackLayout.SEQ_MAJOR)
    do_packed = (
        rearrange(q0, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk) if include_bwd else None
    )

    def fn() -> tuple[torch.Tensor, ...]:
        q = q_packed.clone().requires_grad_(include_bwd)
        k = k_packed.clone().requires_grad_(include_bwd)
        v = v_packed.clone().requires_grad_(include_bwd)
        o, _ = flex_flash_attn_func(
            q,
            k,
            v,
            index_sparse_indices=index_sparse_indices,
            q_block_size=1,
            sparse_k_block_size=kbs,
            pack_gqa=True,
            deterministic=deterministic,
            swap_bwd_qk_loop=swap_bwd_qk_loop,
        )
        if include_bwd:
            o.backward(do_packed)
            return o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()
        return (o.detach(),)

    return fn
