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

"""Minimal repro for S=65536 sparse_kv_indices CUDA illegal memory access."""

import torch
from einops import rearrange

from magi_attention.functional import flex_flash_attn_func

B = 1
S = 65536
NHQ = 16
NHK = 4
D = 128
topk = max(1, int(S * 0.05))  # 3276
dtype = torch.bfloat16
device = "cuda"

print(f"B={B}, S={S}, NHQ={NHQ}, NHK={NHK}, D={D}, topk={topk}")
print(f"num_unique_q = B*NHK*S = {B * NHK * S}")

# Build sparse_kv_indices [B, NHK, S, topk]
print("Building sparse_kv_indices...")
indices_per_q = torch.stack(
    [torch.randperm(S, device=device)[:topk].sort().values for _ in range(S)]
).int()
sparse_kv_indices = (
    indices_per_q.unsqueeze(0).unsqueeze(0).expand(B, NHK, S, topk).contiguous()
)
print(f"sparse_kv_indices shape: {sparse_kv_indices.shape}")

# Build QKV
q = torch.randn(B, S, NHQ, D, dtype=dtype, device=device)
k = torch.randn(B, S, NHK, D, dtype=dtype, device=device)
v = torch.randn(B, S, NHK, D, dtype=dtype, device=device)

q_ffa = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK)
k_ffa = rearrange(k, "b s h d -> (b h s) 1 d")
v_ffa = rearrange(v, "b s h d -> (b h s) 1 d")
print(f"q_ffa: {q_ffa.shape}, k_ffa: {k_ffa.shape}")

# Run
print("Running flex_flash_attn_func...")
torch.cuda.synchronize()
try:
    with torch.no_grad():
        o, meta = flex_flash_attn_func(
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
    print(f"SUCCESS! output shape: {o.shape}")
except Exception as e:
    print(f"FAILED: {e}")
