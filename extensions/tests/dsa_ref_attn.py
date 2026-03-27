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

from magi_attention.utils import max_fp_dtype


def dsa_ref_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_map: torch.Tensor,
    softmax_scale: float | None = None,
    high_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch reference implementation for per-token per-K-head Top-K sparse attention.

    Args:
        q: (sq, nhq, hd)
        k: (skv, nhkv, hd)
        v: (skv, nhkv, hd)
        index_map: (nhkv, sq, topk) - Stores the topk K-token indices for each KV-head.
        softmax_scale: float, scaling factor.

    Returns:
        o: (sq, nhq, hd)
        lse: (sq, nhq)
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    softmax_scale = hd**-0.5 if softmax_scale is None else softmax_scale

    group_size = nhq // nhkv

    # maybe cast input to high precision
    org_dtype = q.dtype
    lse_dtype = max_fp_dtype(org_dtype, torch.float32)

    if high_precision:
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        v = v.to(torch.float64)
    else:
        q = q
        k = k
        v = v

    out = torch.zeros_like(q)
    lse = torch.zeros((sq, nhq), dtype=lse_dtype, device=q.device)

    # loop over each KV head
    for h_kv in range(nhkv):
        curr_k = k[:, h_kv, :]
        curr_v = v[:, h_kv, :]

        # extract Q heads for the current KV head
        h_q_start = h_kv * group_size
        h_q_end = (h_kv + 1) * group_size
        curr_q = q[:, h_q_start:h_q_end, :]

        # extract topk K/V tokens for each Q token in the current KV head
        # index_map[h_kv]: (sq, topk)
        curr_indices = index_map[h_kv]  # (sq, topk)

        # extract K/V tokens using advanced indexing
        k_selected = curr_k[curr_indices]  # (sq, topk, hd)
        v_selected = curr_v[curr_indices]  # (sq, topk, hd)

        # compute attention scores
        # curr_q: (sq, group_size, hd)
        # k_selected: (sq, topk, hd)
        # scores: (sq, group_size, topk)
        scores = (
            torch.einsum("sq d, s t d -> s q t", curr_q, k_selected) * softmax_scale
        )

        # compute LSE and Softmax
        # curr_lse: (sq, group_size)
        curr_lse = torch.logsumexp(scores, dim=-1)
        # curr_probs: (sq, group_size, topk)
        curr_probs = torch.softmax(scores, dim=-1)

        # curr_probs: (sq, group_size, topk)
        # v_selected: (sq, topk, hd)
        # curr_out: (sq, group_size, hd)
        curr_out = torch.einsum("s q t, s t d -> s q d", curr_probs, v_selected)

        out[:, h_q_start:h_q_end, :] = curr_out
        lse[:, h_q_start:h_q_end] = curr_lse

    return out.to(org_dtype), lse.to(lse_dtype)
