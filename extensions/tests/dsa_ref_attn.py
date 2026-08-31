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

from magi_attention.utils.dtype import max_fp_dtype


def dsa_ref_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_sparse_indices: torch.Tensor,
    softmax_scale: float | None = None,
    high_precision: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Torch reference implementation for per-token per-K-head Top-K sparse attention.

    Args:
        q: (sq, nhq, hd)
        k: (skv, nhkv, hd)
        v: (skv, nhkv, hd)
        index_sparse_indices: (sq, nhkv, topk) - top-K KV indices per Q-token per KV-head.
        softmax_scale: float, scaling factor.

    Returns:
        o: (sq, nhq, hd)
        lse: (sq, nhq)
    """
    sq, nhq, hd = q.shape
    skv, nhkv, _ = k.shape
    softmax_scale = hd**-0.5 if softmax_scale is None else softmax_scale

    group_size = nhq // nhkv

    org_dtype = q.dtype
    lse_dtype = max_fp_dtype(org_dtype, torch.float32)

    if high_precision:
        q = q.to(torch.float64)
        k = k.to(torch.float64)
        v = v.to(torch.float64)

    out = torch.zeros_like(q)
    lse = torch.zeros((sq, nhq), dtype=lse_dtype, device=q.device)

    for h_kv in range(nhkv):
        curr_k = k[:, h_kv, :]
        curr_v = v[:, h_kv, :]

        h_q_start = h_kv * group_size
        h_q_end = (h_kv + 1) * group_size
        curr_q = q[:, h_q_start:h_q_end, :]

        # index_sparse_indices[:, h_kv, :] -> (sq, topk)
        curr_indices = index_sparse_indices[:, h_kv, :]

        k_selected = curr_k[curr_indices]  # (sq, topk, hd)
        v_selected = curr_v[curr_indices]  # (sq, topk, hd)

        scores = (
            torch.einsum("sq d, s t d -> s q t", curr_q, k_selected) * softmax_scale
        )

        curr_lse = torch.logsumexp(scores, dim=-1)
        curr_probs = torch.softmax(scores, dim=-1)

        curr_out = torch.einsum("s q t, s t d -> s q d", curr_probs, v_selected)

        out[:, h_q_start:h_q_end, :] = curr_out
        lse[:, h_q_start:h_q_end] = curr_lse

    return out.to(org_dtype), lse.to(lse_dtype)
