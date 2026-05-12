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

"""Repro: run S=32768 then S=65536 sequentially (mimic benchmark flow)."""

import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func


def run_one(S, NHQ, NHK, D, sparsity, B=1, dtype=torch.bfloat16):
    device = "cuda"
    topk = max(1, int(S * sparsity))
    print(f"\n=== S={S}, topk={topk}, NHQ={NHQ}, NHK={NHK} ===")

    indices_per_q = torch.stack(
        [torch.randperm(S, device=device)[:topk].sort().values for _ in range(S)]
    ).int()
    sparse_kv_indices = (
        indices_per_q.unsqueeze(0).unsqueeze(0).expand(B, NHK, S, topk).contiguous()
    )

    q = torch.randn(B, S, NHQ, D, dtype=dtype, device=device)
    k = torch.randn(B, S, NHK, D, dtype=dtype, device=device)
    v = torch.randn(B, S, NHK, D, dtype=dtype, device=device)

    q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
    k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
    v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")

    torch.cuda.synchronize()
    try:
        with torch.no_grad():
            o, _ = flex_flash_attn_func(
                q_ffa,
                k_ffa,
                v_ffa,
                sparse_kv_indices=sparse_kv_indices,
                actual_topk=[topk] * B,
                q_block_size=1,
                k_block_size=1,
                pack_gqa=True,
            )
        torch.cuda.synchronize()
        print(f"  SUCCESS, output: {o.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")


NHQ, NHK, D = 16, 4, 128

# Mimic benchmark: dense first, then sparse for each seqlen
for S in [32768, 65536]:
    for sparsity in [0.05, 0.1, 0.2, 0.5]:
        run_one(S, NHQ, NHK, D, sparsity)
