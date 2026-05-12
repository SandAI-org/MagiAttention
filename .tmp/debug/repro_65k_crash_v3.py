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

"""Bisect the topk threshold where S=65536 crashes."""

import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func


def run_one(S, topk, NHQ=16, NHK=4, D=128, B=1, dtype=torch.bfloat16):
    device = "cuda"
    print(f"S={S}, topk={topk}, num_unique_q={B*NHK*S} ... ", end="", flush=True)

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
        print(f"OK (shape={o.shape})")
        del q, k, v, q_ffa, k_ffa, v_ffa, o, sparse_kv_indices, indices_per_q
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"CRASH: {e}")
        return False


S = 65536
# Bisect between 6553 (ok) and 13107 (crash)
# Try powers-of-128 aligned values
for topk in [
    7168,
    7680,
    8192,
    8704,
    9216,
    9728,
    10240,
    10752,
    11264,
    11776,
    12288,
    12800,
    13107,
]:
    ok = run_one(S, topk)
    if not ok:
        print(f"\n>>> Crash boundary: topk={topk}")
        break
