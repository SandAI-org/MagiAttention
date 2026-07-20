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

"""DSA (Dense-Sparse Attention) interface.

Provides a unified entry point for per-token per-KV-head Top-K sparse attention,
supporting both forward and backward passes for training.

Usage::

    from magi_attn_extensions import dsa_attn_func

    # q: (sq, nhq, hd), k/v: (skv, nhkv, hd)
    # index_map: (nhkv, sq, topk) — top-K KV indices per Q-token per KV-head
    out, lse = dsa_attn_func(q, k, v, index_map, backend="ffa_index_sparse")
    out.sum().backward()  # gradients flow through
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.testing.ref_attn import _calc_attn_lse
from magi_attention.utils import nvtx

# ---------------------------------------------------------------------------
# Backend: ffa_index_sparse (recommended for training)
# ---------------------------------------------------------------------------


def ffa_index_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index-sparse attention via FFA kernel. Supports forward + backward.

    Args:
        q: (sq, nhq, hd)
        k: (skv, nhkv, hd)
        v: (skv, nhkv, hd)
        index_map: (nhkv, sq, topk) with K-token indices in [0, skv).
        softmax_scale: optional scaling factor (default: hd**-0.5).
        deterministic: if True, use bitwise-deterministic backward.

    Returns:
        out: (sq, nhq, hd)
        lse: (sq, nhq)
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    topk = index_map.shape[-1]
    group_size = nhq // nhkv

    # index_map: (nhkv, sq, topk) -> index_sparse_indices: (sq, nhkv, topk)
    index_sparse_indices = index_map.permute(1, 0, 2).to(torch.int32).contiguous()

    # pad topk to tile_size boundary (kernel requirement)
    tile_size = 128
    if topk % tile_size != 0:
        pad_size = tile_size - (topk % tile_size)
        index_sparse_indices = F.pad(index_sparse_indices, (0, pad_size), value=-1)

    out, meta = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=index_sparse_indices,
        q_block_size=1,
        sparse_k_block_size=1,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        pack_gqa=group_size > 1,
    )

    assert meta.lse is not None
    return out, meta.lse


# ---------------------------------------------------------------------------
# Backend: ffa_block_sparse
# ---------------------------------------------------------------------------


def ffa_block_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-sparse attention via FFA kernel. Supports forward + backward.

    Flattens Q/K/V by KV-head and constructs per-token ranges for block-sparse path.

    Args:
        q: (sq, nhq, hd)
        k: (skv, nhkv, hd)
        v: (skv, nhkv, hd)
        index_map: (nhkv, sq, topk) with K-token indices in [0, skv).
        softmax_scale: optional scaling factor (default: hd**-0.5).
        deterministic: if True, use bitwise-deterministic backward.

    Returns:
        out: (sq, nhq, hd)
        lse: (sq, nhq)
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    topk = index_map.shape[-1]
    group_size = nhq // nhkv

    # flatten along KV-head dimension
    q_flat = rearrange(
        q,
        "sq (nhkv group_size) hd -> (nhkv sq) group_size hd",
        nhkv=nhkv,
        group_size=group_size,
    )
    k_flat = rearrange(k, "skv nhkv hd -> (nhkv skv) 1 hd")
    v_flat = rearrange(v, "skv nhkv hd -> (nhkv skv) 1 hd")

    # build per-token q_ranges and k_ranges from index_map
    q_idx_flat = (
        rearrange(
            torch.arange(nhkv * sq, device=q.device, dtype=torch.int32), "n -> n 1"
        )
        .repeat(1, topk)
        .flatten()
    )
    h_kv_offset = rearrange(
        torch.arange(nhkv, device=q.device, dtype=torch.int32) * skv, "nhkv -> nhkv 1 1"
    )
    k_idx_flat = rearrange(index_map + h_kv_offset, "nhkv sq topk -> (nhkv sq topk)")

    q_ranges = torch.stack([q_idx_flat, q_idx_flat + 1], dim=-1)
    k_ranges = torch.stack([k_idx_flat, k_idx_flat + 1], dim=-1)

    ref_block_size = (64, 128) if group_size <= 64 else (128, 128)

    out_flat, meta = flex_flash_attn_func(
        q_flat,
        k_flat,
        v_flat,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        softmax_scale=softmax_scale,
        deterministic=deterministic,
        range_merge=True,
        block_sparse=True,
        pack_gqa=group_size > 1,
        ref_block_size=ref_block_size,
    )

    o = out_flat.view(nhkv, sq, group_size, hd).transpose(0, 1).reshape(sq, nhq, hd)
    assert meta.lse is not None
    lse = meta.lse.view(nhkv, sq, group_size).transpose(0, 1).reshape(sq, nhq)

    return o, lse


# ---------------------------------------------------------------------------
# Baseline: PyTorch flex_attention (forward-only benchmark reference)
# ---------------------------------------------------------------------------

_flex_attn_compiled = torch.compile(flex_attention)


@torch.compile
def _flex_attn_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    group_size = nhq // nhkv

    q_flex = rearrange(q, "sq nhq hd -> 1 nhq sq hd")
    k_flex = rearrange(k, "skv nhkv hd -> 1 nhkv skv hd")
    v_flex = rearrange(v, "skv nhkv hd -> 1 nhkv skv hd")

    mask_full = torch.zeros(nhkv, sq, skv, device=q.device, dtype=torch.bool)
    mask_full.scatter_(2, index_map.to(torch.int64), True)

    def topk_mask_mod(b, h, q_idx, kv_idx):
        h_kv = h // group_size
        return mask_full[h_kv, q_idx, kv_idx]

    block_mask = create_block_mask(topk_mask_mod, 1, nhq, sq, skv, device=q.device)
    with nvtx.add_nvtx_event("flex_attn_func"):
        o_flex, lse_flex = _flex_attn_compiled(
            q_flex,
            k_flex,
            v_flex,
            score_mod=None,
            block_mask=block_mask,
            scale=softmax_scale,
            enable_gqa=True,
            return_lse=True,
        )

    o = o_flex.squeeze(0).transpose(0, 1)
    lse = lse_flex.squeeze(0).transpose(0, 1)
    return o, lse


# ---------------------------------------------------------------------------
# Baseline: SDPA (forward-only benchmark reference)
# ---------------------------------------------------------------------------


@torch.compile
def _sdpa_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    group_size = nhq // nhkv

    q_sdpa = rearrange(q, "sq nhq hd -> 1 nhq sq hd")
    k_sdpa = rearrange(k, "skv nhkv hd -> 1 nhkv skv hd")
    v_sdpa = rearrange(v, "skv nhkv hd -> 1 nhkv skv hd")

    mask_kv = torch.zeros(nhkv, sq, skv, device=q.device, dtype=torch.bool)
    mask_kv.scatter_(2, index_map.to(torch.int64), True)
    mask_sdpa = (
        mask_kv.unsqueeze(1).expand(nhkv, group_size, sq, skv).reshape(1, nhq, sq, skv)
    )

    o_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=mask_sdpa,
        scale=softmax_scale,
        is_causal=False,
        enable_gqa=True,
    )

    o = o_sdpa.squeeze(0).transpose(0, 1)
    lse = _calc_attn_lse(
        q=q,
        k=k,
        mask=mask_sdpa.squeeze(0),
        softmax_scale=softmax_scale,
    ).squeeze(0)

    return o, lse


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def dsa_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
    deterministic: bool = False,
    backend: str = "ffa_index_sparse",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token per-KV-head Top-K sparse attention.

    Args:
        q: (sq, nhq, hd) — query tensor.
        k: (skv, nhkv, hd) — key tensor.
        v: (skv, nhkv, hd) — value tensor.
        index_map: (nhkv, sq, topk) — top-K KV indices for each Q-token per KV-head.
        softmax_scale: scaling factor (default: hd**-0.5).
        deterministic: if True, use bitwise-deterministic backward (FFA backends only).
        backend: one of "ffa_index_sparse" (default), "ffa_block_sparse", "flex", "sdpa".
            - "ffa_index_sparse": recommended for training (forward + backward).
            - "ffa_block_sparse": alternative FFA path (forward + backward).
            - "flex": PyTorch FlexAttention baseline (forward-only, torch.compile).
            - "sdpa": SDPA baseline (forward-only, torch.compile).

    Returns:
        out: (sq, nhq, hd) — attention output.
        lse: (sq, nhq) — log-sum-exp values.
    """
    if backend == "ffa_index_sparse":
        return ffa_index_sparse(q, k, v, index_map, softmax_scale, deterministic)
    elif backend == "ffa_block_sparse":
        return ffa_block_sparse(q, k, v, index_map, softmax_scale, deterministic)
    elif backend == "flex":
        return _flex_attn_sparse_fwd(q, k, v, index_map, softmax_scale)
    elif backend == "sdpa":
        return _sdpa_sparse_fwd(q, k, v, index_map, softmax_scale)
    else:
        raise ValueError(
            f"Invalid backend: {backend!r}. "
            f"Choose from: 'ffa_index_sparse', 'ffa_block_sparse', 'flex', 'sdpa'."
        )
