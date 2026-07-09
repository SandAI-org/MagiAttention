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
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.testing.ref_attn import _calc_attn_lse
from magi_attention.utils import nvtx

flex_attn_func = torch.compile(flex_attention)


@torch.compile
def ffa_block_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass using flex_flash_attn BlockSparse path (q_ranges/k_ranges + block_sparse=True).
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    topk = index_map.shape[-1]

    group_size = nhq // nhkv

    # q_flat: (nhkv * sq, group_size, hd)
    q_flat = rearrange(
        q,
        "sq (nhkv group_size) hd -> (nhkv sq) group_size hd",
        nhkv=nhkv,
        group_size=group_size,
    )
    # k_flat/v_flat: (nhkv * skv, 1, hd)
    k_flat = rearrange(k, "skv nhkv hd -> (nhkv skv) 1 hd")
    v_flat = rearrange(v, "skv nhkv hd -> (nhkv skv) 1 hd")

    # build q_ranges and k_ranges from index_map: (nhkv, sq, topk)
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
        range_merge=True,
        block_sparse=True,
        pack_gqa=True if group_size > 1 else False,
        ref_block_size=ref_block_size,
    )

    # out_flat: (nhkv * sq, group_size, hd) -> (sq, nhq, hd)
    o = out_flat.view(nhkv, sq, group_size, hd).transpose(0, 1).reshape(sq, nhq, hd)
    assert meta.lse is not None
    lse = meta.lse.view(nhkv, sq, group_size).transpose(0, 1).reshape(sq, nhq)

    return o, lse


@torch.compile
def ffa_index_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass using flex_flash_attn IndexSparse path (index_sparse_indices direct to kernel).

    Q/K/V are passed in natural Dense-style shapes without KV-head flattening.
    index_map: (nhkv, sq, topk) with logical K positions (0..skv-1).
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    topk = index_map.shape[-1]
    group_size = nhq // nhkv

    # index_map: (nhkv, sq, topk) → index_sparse_indices: (sq, nhkv, topk)
    index_sparse_indices = index_map.permute(1, 0, 2).to(torch.int32).contiguous()

    tile_size = 128
    if topk % tile_size != 0:
        pad_size = tile_size - (topk % tile_size)
        index_sparse_indices = torch.nn.functional.pad(
            index_sparse_indices, (0, pad_size), value=-1
        )

    out, meta = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=index_sparse_indices,
        q_block_size=1,
        sparse_k_block_size=1,
        softmax_scale=softmax_scale,
        pack_gqa=True if group_size > 1 else False,
    )

    assert meta.lse is not None
    return out, meta.lse


@torch.compile
def flex_attn_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for per-token per-K-head Top-K sparse attention using flex_attention.
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape

    # convert dimensions to match flex_attention's (B, H, S, D) format
    # here we assume B=1
    q_flex = rearrange(q, "sq nhq hd -> 1 nhq sq hd")
    k_flex = rearrange(k, "skv nhkv hd -> 1 nhkv skv hd")
    v_flex = rearrange(v, "skv nhkv hd -> 1 nhkv skv hd")

    # prepare mask auxiliary tensor
    # index_map: (nhkv, sq, topk)
    # we need an efficient method to check if (h_kv, i_q, i_k) is in index_map

    # define mask_mod
    # nhq and nhkv may be different (GQA)
    group_size = nhq // nhkv

    # precompute mask to improve efficiency
    # mask: (nhkv, sq, skv)
    mask_full = torch.zeros(nhkv, sq, skv, device=q.device, dtype=torch.bool)
    # use scatter to set the positions in index_map to True
    # index_map: (nhkv, sq, topk)
    mask_full.scatter_(2, index_map.to(torch.int64), True)

    def topk_mask_mod(b, h, q_idx, kv_idx):
        h_kv = h // group_size
        return mask_full[h_kv, q_idx, kv_idx]

    # create block_mask
    # for sparse attention, creating block_mask can significantly accelerate
    block_mask = create_block_mask(topk_mask_mod, 1, nhq, sq, skv, device=q.device)
    with nvtx.add_nvtx_event("flex_attn_func"):
        # use block_mask to handle sparsity, score_mod is set to None
        o_flex, lse_flex = flex_attn_func(
            q_flex,
            k_flex,
            v_flex,
            score_mod=None,
            block_mask=block_mask,
            scale=softmax_scale,
            enable_gqa=True,
            return_lse=True,
        )

    # o_flex: (1, nhq, sq, hd) -> (sq, nhq, hd)
    o = o_flex.squeeze(0).transpose(0, 1)
    # lse_flex: (1, nhq, sq) -> (sq, nhq)
    lse = lse_flex.squeeze(0).transpose(0, 1)

    return o, lse


@torch.compile
def sdpa_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass for per-token per-K-head Top-K sparse attention using torch.nn.functional.scaled_dot_product_attention.
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    group_size = nhq // nhkv

    # SDPA expects (B, H, S, D)
    q_sdpa = rearrange(q, "sq nhq hd -> 1 nhq sq hd")
    k_sdpa = rearrange(k, "skv nhkv hd -> 1 nhkv skv hd")
    v_sdpa = rearrange(v, "skv nhkv hd -> 1 nhkv skv hd")

    # Build mask: (1, nhq, sq, skv)
    # 1. Create base mask for KV heads: (nhkv, sq, skv)
    mask_kv = torch.zeros(nhkv, sq, skv, device=q.device, dtype=torch.bool)
    mask_kv.scatter_(2, index_map.to(torch.int64), True)

    # 2. Expand to Q heads for GQA: (nhkv, 1, sq, skv) -> (nhkv, group_size, sq, skv) -> (nhq, sq, skv)
    mask_sdpa = (
        mask_kv.unsqueeze(1).expand(nhkv, group_size, sq, skv).reshape(1, nhq, sq, skv)
    )

    # SDPA call
    # Note: SDPA does not return LSE directly in the functional interface
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
    # SDPA functional doesn't return LSE, calculate it using reference implementation
    lse = _calc_attn_lse(
        q=q,
        k=k,
        mask=mask_sdpa.squeeze(0),
        softmax_scale=softmax_scale,
    )

    lse = lse.squeeze(0)

    return o, lse


def dsa_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
    backend: str = "flex",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch reference implementation for per-token per-K-head Top-K sparse attention.

    Args:
        q: (sq, nhq, hd)
        k: (skv, nhkv, hd)
        v: (skv, nhkv, hd)
        index_map: (nhkv, sq, topk) - Stores the topk K-token indices for each KV-head.
        softmax_scale: float, scaling factor.
        backend: str, one of "flex", "ffa_block_sparse", "ffa_index_sparse", "sdpa".
    """
    if backend == "flex":
        return flex_attn_sparse_fwd(q, k, v, index_map, softmax_scale)
    elif backend == "ffa_block_sparse":
        return ffa_block_sparse_fwd(q, k, v, index_map, softmax_scale)
    elif backend == "ffa_index_sparse":
        return ffa_index_sparse_fwd(q, k, v, index_map, softmax_scale)
    elif backend == "sdpa":
        return sdpa_sparse_fwd(q, k, v, index_map, softmax_scale)
    else:
        raise ValueError(f"Invalid backend: {backend}")
