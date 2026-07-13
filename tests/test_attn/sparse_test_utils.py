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
from typing import Iterator

import torch
import torch.nn.functional as F
from einops import rearrange

SEED = 42
DEFAULT_FWD_ATOL = 0.01
DEFAULT_BWD_DQ_ATOL = 0.05


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


def compare_sdpa_fwd(
    o_ffa: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sdpa_mask: torch.Tensor,
    *,
    B: int,
    NHQ: int,
    NHK: int,
    atol: float,
    test_case: str,
) -> None:
    """Compare FFA output against SDPA reference, batch by batch, head by head."""
    gqa = NHQ // NHK
    err_msgs: list[str] = []
    for b_idx in range(B):
        o_ref_heads = []
        with torch.no_grad():
            for h_q in range(NHQ):
                h_kv = h_q // gqa
                q_h = q[b_idx, :, h_q : h_q + 1, :].unsqueeze(0).transpose(1, 2)
                k_h = k[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                v_h = v[b_idx, :, h_kv : h_kv + 1, :].unsqueeze(0).transpose(1, 2)
                mask_h = sdpa_mask[b_idx, h_kv : h_kv + 1].unsqueeze(0)
                o_h = F.scaled_dot_product_attention(q_h, k_h, v_h, attn_mask=mask_h)
                o_ref_heads.append(o_h)
        o_ref = torch.cat(o_ref_heads, dim=1)
        o_ref = rearrange(o_ref, "1 h s d -> s h d")

        max_diff = (o_ffa[b_idx].float() - o_ref.float()).abs().max().item()
        if max_diff >= atol:
            err_msgs.append(
                f"batch {b_idx}: max_diff={max_diff:.6f} >= {atol} in {test_case}"
            )

    if err_msgs:
        raise AssertionError("\n".join(err_msgs))


def compare_sdpa_bwd_dq(
    dq_ffa_packed: torch.Tensor,
    do_packed: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sdpa_mask: torch.Tensor,
    *,
    B: int,
    S_q: int,
    NHQ: int,
    NHK: int,
    atol: float,
    test_case: str,
) -> None:
    """Compare packed FFA dQ against SDPA backward reference."""
    gqa = NHQ // NHK
    total_q = B * S_q

    q_sdpa = q.transpose(1, 2).detach().clone().requires_grad_(True)
    k_expanded = k.repeat_interleave(gqa, dim=2).transpose(1, 2).detach()
    v_expanded = v.repeat_interleave(gqa, dim=2).transpose(1, 2).detach()
    mask_expanded = sdpa_mask.repeat_interleave(gqa, dim=1)
    o_sdpa = F.scaled_dot_product_attention(
        q_sdpa, k_expanded, v_expanded, attn_mask=mask_expanded
    )
    do_sdpa = rearrange(do_packed, "(b s h1) h2 d -> b (h1 h2) s d", b=B, h1=NHK, s=S_q)
    o_sdpa.backward(do_sdpa)
    dq_ref = rearrange(q_sdpa.grad, "b h s d -> (b s) h d")[:total_q]

    dq_ffa_reshaped = rearrange(
        dq_ffa_packed, "(b s h1) h2 d -> b (h1 h2) s d", b=B, h1=NHK, s=S_q
    )
    dq_ref_reshaped = rearrange(dq_ref, "(b s) h d -> b h s d", b=B, s=S_q)

    err_msgs: list[str] = []
    for b_idx in range(B):
        max_dq_diff = (
            (dq_ffa_reshaped[b_idx].float() - dq_ref_reshaped[b_idx].float())
            .abs()
            .max()
            .item()
        )
        if max_dq_diff >= atol:
            err_msgs.append(
                f"BWD batch {b_idx}: dQ max_diff={max_dq_diff:.6f} >= {atol} in {test_case}"
            )
    if err_msgs:
        raise AssertionError("\n".join(err_msgs))


def check_ffa_deterministic_twice(
    run_once: Callable[[], tuple[torch.Tensor, torch.Tensor | None]],
    *,
    test_case: str,
    check_dq: bool = True,
) -> list[str]:
    """Run FFA twice in deterministic mode and compare (index_sparse style)."""
    o_det1, dq_det1 = run_once()
    o_det2, dq_det2 = run_once()

    err_msgs: list[str] = []
    if not torch.equal(o_det1, o_det2):
        err_msgs.append(f"For {test_case=}: forward output not deterministic")
    if check_dq and dq_det1 is not None and dq_det2 is not None:
        if not torch.equal(dq_det1, dq_det2):
            err_msgs.append(f"For {test_case=}: backward dQ not deterministic")
    return err_msgs


def check_ffa_deterministic_rerun(
    *,
    o_ref: torch.Tensor,
    dq_ref: torch.Tensor | None,
    dk_ref: torch.Tensor | None,
    dv_ref: torch.Tensor | None,
    run_deterministic: Callable[
        [], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    test_case: str,
) -> list[str]:
    """Rerun FFA in deterministic mode and compare to stored refs (block_sparse style)."""
    err_msgs: list[str] = []
    o_det, dq_det, dk_det, dv_det = run_deterministic()
    try:
        if not torch.equal(o_det, o_ref):
            err_msgs.append(f"For {test_case=}: forward output not deterministic")
        if dq_ref is not None and not torch.equal(dq_det, dq_ref):
            err_msgs.append(f"For {test_case=}: backward dq not deterministic")
        if dk_ref is not None and not torch.equal(dk_det, dk_ref):
            err_msgs.append(f"For {test_case=}: backward dk not deterministic")
        if dv_ref is not None and not torch.equal(dv_det, dv_ref):
            err_msgs.append(f"For {test_case=}: backward dv not deterministic")
    except Exception as e:
        err_msgs.append(str(e))
    return err_msgs
