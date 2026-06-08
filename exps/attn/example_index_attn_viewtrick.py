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

"""
IndexAttn + PackGQA + View Trick correctness verification
==========================================================

Background:
  GQA4 model: NHQ=32, NHK=4, HD=128
  Want q_block_size=4 speedup, but IndexAttn requires q_block_size=1.
  Solution: View Trick -- fold consecutive 4 Q tokens into the head dim.

View Trick transform:
  +---------------------------------------------------------------------+
  |  Original: q = [S, 32, D]     k = [S, 4, D]       v = [S, 4, D]    |
  |                                                                      |
  |  Step 1: Flatten KV heads -- each head becomes an independent row    |
  |    k_flat = k.view(S*4, 1, D)    v_flat = v.view(S*4, 1, D)         |
  |                                                                      |
  |  Step 2: Q fold (QBS=4 tokens -> 128 heads)                         |
  |    q_viewed = q.view(S//4, 32*4, D) = [S//4, 128, D]                |
  |                                                                      |
  |  Result: NHQ_eff=128, NHK_eff=1 -> MQA, use pack_gqa=True           |
  +---------------------------------------------------------------------+

Indices encoding:
  indices shape: (S_new, NHK_eff=1, max_topk), dtype=int32
  Each value is a logical token position: batch_idx * S_kv + token_idx
  Invalid slots filled with -1, max_topk must be a multiple of tile_size(128)

Run:
  CUDA_HOME=/usr/local/cuda-13.0 python example_index_attn_viewtrick.py
"""

import torch

from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils.sparse_utils import (
    build_index_attn_indices,
    get_sdpa_mask_from_index_attn_indices,
)

# ========================================================
# Parameters
# ========================================================

NHQ_ORIG = 32  # original Q heads
NHK_ORIG = 4  # original KV heads (GQA ratio = 32/4 = 8)
HD = 128  # head dim
QBS = 4  # q_block_size: how many Q tokens to fold

# After view trick
NHQ_EFF = NHQ_ORIG * QBS  # 128
NHK_EFF = 1  # each original KV head becomes an independent row -> MQA

TILE_SIZE = 128  # IndexAttn kernel tile, max_topk must be a multiple of this
B = 1
DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 42


# ========================================================
# FFA forward + backward
# ========================================================


def run_ffa(q_orig, k_orig, v_orig, indices, S_new, grad_output):
    """Run FFA IndexAttn FWD+BWD, return (out, dq, dk, dv)."""
    S_kv_flat = k_orig.numel() // HD
    q_viewed = q_orig.reshape(B * S_new, NHQ_EFF, HD)
    k_flat = k_orig.reshape(B * S_kv_flat, NHK_EFF, HD)
    v_flat = v_orig.reshape(B * S_kv_flat, NHK_EFF, HD)

    q = q_viewed.detach().clone().requires_grad_(True)
    k = k_flat.detach().clone().requires_grad_(True)
    v = v_flat.detach().clone().requires_grad_(True)

    out, _ = flex_flash_attn_func(
        q,
        k,
        v,
        index_attn_indices=indices,
        q_block_size=1,
        k_block_size=1,
        pack_gqa=True,
    )
    out.backward(grad_output)

    return out, q.grad, k.grad, v.grad


# ========================================================
# SDPA reference forward + backward
# ========================================================


def run_sdpa_ref(q_orig, k_orig, v_orig, indices, S_new, grad_output):
    """Run PyTorch SDPA FWD+BWD as reference, return (out, dq, dk, dv)."""
    S_kv_flat = k_orig.numel() // HD
    q_viewed = q_orig.reshape(B * S_new, NHQ_EFF, HD)
    k_flat = k_orig.reshape(B * S_kv_flat, NHK_EFF, HD)
    v_flat = v_orig.reshape(B * S_kv_flat, NHK_EFF, HD)

    # indices -> dense bool mask [B, NHQ_EFF, S_new, S_kv_flat]
    mask = get_sdpa_mask_from_index_attn_indices(
        indices,
        B=B,
        NHQ=NHQ_EFF,
        NHK=NHK_EFF,
        S_q=S_new,
        S_kv=S_kv_flat,
        device=DEVICE,
    )

    # SDPA expects (B, H, S, D) layout
    q = (
        q_viewed.reshape(B, S_new, NHQ_EFF, HD)
        .transpose(1, 2)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    k = (
        k_flat.reshape(B, S_kv_flat, NHK_EFF, HD)
        .transpose(1, 2)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    v = (
        v_flat.reshape(B, S_kv_flat, NHK_EFF, HD)
        .transpose(1, 2)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.expand(B, NHQ_EFF, S_kv_flat, HD),
        v.expand(B, NHQ_EFF, S_kv_flat, HD),
        attn_mask=mask,
    )
    out.backward(grad_output.reshape(B, S_new, NHQ_EFF, HD).transpose(1, 2))

    return out, q.grad, k.grad, v.grad


# ========================================================
# Compare
# ========================================================


def compare(ffa_results, ref_results, S_new, S_kv_flat):
    """Compare FFA vs SDPA results, assert cosine similarity."""
    out_ffa, dq_ffa, dk_ffa, dv_ffa = ffa_results
    out_ref, dq_ref, dk_ref, dv_ref = ref_results

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten().float(), b.flatten().float(), dim=0
        ).item()

    # FFA output: (S_new, NHQ_EFF, HD) -> (B, NHQ_EFF, S_new, HD)
    cos_fwd = cos(out_ffa.reshape(B, S_new, NHQ_EFF, HD).transpose(1, 2), out_ref)
    cos_dq = cos(dq_ffa.reshape(B, S_new, NHQ_EFF, HD).transpose(1, 2), dq_ref)
    cos_dk = cos(dk_ffa.reshape(B, S_kv_flat, NHK_EFF, HD).transpose(1, 2), dk_ref)
    cos_dv = cos(dv_ffa.reshape(B, S_kv_flat, NHK_EFF, HD).transpose(1, 2), dv_ref)

    print(
        f"  FWD cos={cos_fwd:.6f}  BWD dQ={cos_dq:.6f} dK={cos_dk:.6f} dV={cos_dv:.6f}"
    )
    assert cos_fwd > 0.999 and cos_dq > 0.99 and cos_dk > 0.99 and cos_dv > 0.99


# ========================================================
# Test entry
# ========================================================


def run_test(S_orig: int, real_topk: int):
    S_new = S_orig // QBS
    S_kv_flat = S_orig * NHK_ORIG
    max_topk = ((real_topk + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE

    print(
        f"S_orig={S_orig} -> S_new={S_new}, NHQ={NHQ_ORIG}->{NHQ_EFF}, "
        f"real_topk={real_topk}, max_topk={max_topk} (pad {max_topk - real_topk})"
    )

    torch.manual_seed(SEED)
    q_orig = torch.randn(B, S_orig, NHQ_ORIG, HD, device=DEVICE, dtype=DTYPE)
    k_orig = torch.randn(B, S_orig, NHK_ORIG, HD, device=DEVICE, dtype=DTYPE)
    v_orig = torch.randn(B, S_orig, NHK_ORIG, HD, device=DEVICE, dtype=DTYPE)

    indices = build_index_attn_indices(
        B=B,
        NHK=NHK_EFF,
        S_q=S_new,
        S_kv=S_kv_flat,
        topk=real_topk,
        max_topk=max_topk,
        device=DEVICE,
    )
    if max_topk > real_topk:
        assert (indices[:, :, real_topk:] == -1).all(), "Padding should be all -1"

    # Same grad_output for both paths to ensure fair comparison
    grad_output = torch.randn(B * S_new, NHQ_EFF, HD, device=DEVICE, dtype=DTYPE)

    out_ffa, dq_ffa, dk_ffa, dv_ffa = run_ffa(
        q_orig, k_orig, v_orig, indices, S_new, grad_output
    )
    out_ref, dq_ref, dk_ref, dv_ref = run_sdpa_ref(
        q_orig, k_orig, v_orig, indices, S_new, grad_output
    )

    compare(
        (out_ffa, dq_ffa, dk_ffa, dv_ffa),
        (out_ref, dq_ref, dk_ref, dv_ref),
        S_new,
        S_kv_flat,
    )


if __name__ == "__main__":
    # Case 1: topk aligned (no padding)
    run_test(S_orig=1024, real_topk=2048)
    # Case 2: topk unaligned, pad with -1
    run_test(S_orig=1024, real_topk=1500)
    # Case 3: both S_orig and topk unaligned
    run_test(S_orig=2048, real_topk=1337)
    print("All tests passed!")
