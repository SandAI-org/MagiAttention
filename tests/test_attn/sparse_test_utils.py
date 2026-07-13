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

from magi_attention.testing.precision import assert_close

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


def sdpa_ref_bwd_dq(
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
) -> torch.Tensor:
    """Compute SDPA backward dQ reference, returned in (B*S_q, NHQ, D) layout."""
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
    return rearrange(q_sdpa.grad, "b h s d -> (b s) h d")[:total_q]


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
    test_case: str,
    atol: float = DEFAULT_FWD_ATOL,
    rtol: float = DEFAULT_FWD_RTOL,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRES,
) -> None:
    """Compare FFA forward output against SDPA reference using assert_close."""
    o_ref = sdpa_ref_output(q, k, v, sdpa_mask, B=B, NHQ=NHQ, NHK=NHK)
    assert_close(
        o_ffa,
        o_ref,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => fwd_out",
    )


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
    test_case: str,
    atol: float = DEFAULT_BWD_DQ_ATOL,
    rtol: float = DEFAULT_BWD_DQ_RTOL,
    mismatch_threshold: float = DEFAULT_MISMATCH_THRES,
) -> None:
    """Compare packed FFA dQ against SDPA backward reference using assert_close."""
    dq_ref = sdpa_ref_bwd_dq(
        do_packed, q, k, v, sdpa_mask, B=B, S_q=S_q, NHQ=NHQ, NHK=NHK
    )
    dq_ffa_reshaped = rearrange(
        dq_ffa_packed, "(b s h1) h2 d -> b (h1 h2) s d", b=B, h1=NHK, s=S_q
    )
    dq_ref_reshaped = rearrange(dq_ref, "(b s) h d -> b h s d", b=B, s=S_q)

    assert_close(
        dq_ffa_reshaped,
        dq_ref_reshaped,
        atol=atol,
        rtol=rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => dq",
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
    ffa_dq: torch.Tensor,
    ffa_dk: torch.Tensor,
    ffa_dv: torch.Tensor,
    sdpa_dq: torch.Tensor,
    sdpa_dk: torch.Tensor,
    sdpa_dv: torch.Tensor,
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
    """Compare FFA dQ/dK/dV against SDPA reference using assert_close."""
    assert_close(
        ffa_dq,
        sdpa_dq,
        atol=dq_atol,
        rtol=dq_rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => dq",
    )
    assert_close(
        ffa_dk,
        sdpa_dk,
        atol=dk_atol,
        rtol=dk_rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => dk",
    )
    assert_close(
        ffa_dv,
        sdpa_dv,
        atol=dv_atol,
        rtol=dv_rtol,
        mismatch_threshold=mismatch_threshold,
        test_case=f"{test_case} => dv",
    )


def check_ffa_deterministic_twice(
    run_once: Callable[[], tuple[torch.Tensor, ...]],
    *,
    test_case: str,
) -> list[str]:
    """Run FFA twice in deterministic mode and compare all returned tensors."""
    results1 = run_once()
    results2 = run_once()

    labels = ["fwd_out", "dq", "dk", "dv"]
    err_msgs: list[str] = []
    for i, (t1, t2) in enumerate(zip(results1, results2)):
        if t1 is not None and t2 is not None and not torch.equal(t1, t2):
            label = labels[i] if i < len(labels) else f"tensor_{i}"
            err_msgs.append(f"For {test_case=}: {label} not deterministic")
    return err_msgs
