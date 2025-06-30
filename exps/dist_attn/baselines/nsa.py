# Copyright (c) 2025 SandAI. All Rights Reserved.
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
import torch.nn as nn
import torch.nn.functional as F
from flashattn_hopper.flash_attn_interface import flash_attn_func


class NSA(nn.Module):
    def __init__(
        self, hidden_dim, d, l_cmp, l_slc, window_size, block_size_q, slc_top_k, dtype
    ):
        super(NSA, self).__init__()
        self.d = d
        self.l_cmp = l_cmp
        self.l_slc = l_slc
        self.slc_top_k = slc_top_k
        self.window_size = window_size
        self.block_size_q = block_size_q
        self.hidden_dim = hidden_dim

        # cmp mlp layer
        self.cmp_linear_k = nn.Linear(self.l_cmp, 1, dtype=dtype)
        self.cmp_linear_v = nn.Linear(self.l_cmp, 1, dtype=dtype)
        # cmp/slc/win
        self.gate_proj = nn.Linear(hidden_dim, 3, dtype=dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        is_causal,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        bsz, seqlen_q, q_heads = q.shape[:3]
        kv_heads = k.shape[2]
        H = q_heads // kv_heads

        # b,h,n_cmp,l,d
        K_cmp_blocks = extract_blocks(k, self.l_cmp, self.d)
        V_cmp_blocks = extract_blocks(v, self.l_cmp, self.d)
        # b,h,n_cmp,d
        K_cmp = self.cmp_linear_k(K_cmp_blocks.transpose(-1, -2)).squeeze(-1)
        V_cmp = self.cmp_linear_v(V_cmp_blocks.transpose(-1, -2)).squeeze(-1)
        # GQA repeat
        K_cmp_mha = K_cmp.repeat_interleave(H, dim=1)
        V_cmp_mha = V_cmp.repeat_interleave(H, dim=1)

        # b,s,h,d @ b,h,n_cmp,d -> b,h,s,n_cmp
        attn_cmp = torch.einsum("bshd,bhnd->bhsn", q, K_cmp_mha)
        attn_cmp = attn_cmp / softmax_scale
        P_cmp = F.softmax(attn_cmp, dim=-1)
        # b,h,s,n_cmp @ b,h,n_cmp,d -> b,s,h,d
        out_cmp = torch.einsum("bhsn,bhnd->bshd", P_cmp, V_cmp_mha)

        # compute P_slc, b,h,s,n_slc
        if self.l_slc == self.l_cmp == self.d:
            P_slc = P_cmp
            num_blocks_slc = P_cmp.shape[-1]
            # b,h,n,l_slc,d
            K_slc_blocks = K_cmp_blocks
            V_slc_blocks = V_cmp_blocks
        else:
            assert self.l_slc > self.l_cmp, "l_slc must be greater than l_cmp"
            assert self.l_slc % self.d == 0, "l_slc must be divisible by d"
            assert self.l_cmp % self.d == 0, "l_cmp must be divisible by d"
            # b,h,n,l_slc,d
            K_slc_blocks = extract_blocks(k, self.l_slc, self.d)
            V_slc_blocks = extract_blocks(v, self.l_slc, self.d)
            num_blocks_slc = K_slc_blocks.shape[2]
            P_slc = compute_p_slc(P_cmp, self.l_slc, self.l_cmp, self.d, num_blocks_slc)

        # deal q_block_size
        P_slc = compute_blockq_p_slc(q, P_slc, self.block_size_q)
        # deal GQA
        P_slc = compute_gqa_p_slc(P_slc, kv_heads)

        assert (
            self.slc_top_k <= num_blocks_slc
        ), "slc_top_k must be less than or equal to num_blocks_slc"
        # b,h,s,k
        _, idx_slc = torch.topk(P_slc, dim=-1, k=self.slc_top_k)

        # b,h,s,n_slc,l_slc,d
        idx_exp = (
            idx_slc.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, -1, self.l_slc, self.hidden_dim)
        )
        # GQA b,h_kv,H,s,n_slc,l_slc,d
        idx_exp = idx_exp.view(bsz, kv_heads, -1, *idx_exp.shape[2:])
        # GQA b,h_kv,H,s,n_slc,l_slc,d
        K_slc_exp = (
            K_slc_blocks.unsqueeze(2)
            .unsqueeze(2)
            .expand(-1, -1, H, seqlen_q, -1, -1, -1)
        )
        V_slc_exp = (
            V_slc_blocks.unsqueeze(2)
            .unsqueeze(2)
            .expand(-1, -1, H, seqlen_q, -1, -1, -1)
        )
        # b,h,s,k*l_slc,d
        K_slc = torch.gather(K_slc_exp, dim=3, index=idx_exp).view(
            bsz, kv_heads, seqlen_q, -1, self.hidden_dim
        )
        V_slc = torch.gather(V_slc_exp, dim=3, index=idx_exp).view(
            bsz, kv_heads, seqlen_q, -1, self.hidden_dim
        )

        # b*s,1,h,d
        q_slc_fa = q.view(bsz * seqlen_q, 1, q_heads, self.hidden_dim).contiguous()
        # b*s,sk,h,d
        k_slc_fa = (
            K_slc.permute(0, 2, 3, 1, 4)
            .reshape(bsz * seqlen_q, -1, kv_heads, self.hidden_dim)
            .contiguous()
        )
        v_slc_fa = (
            V_slc.permute(0, 2, 3, 1, 4)
            .reshape(bsz * seqlen_q, -1, kv_heads, self.hidden_dim)
            .contiguous()
        )
        # b*s,1,h,d
        out_slc, _ = flash_attn_func(
            q_slc_fa,
            k_slc_fa,
            v_slc_fa,
            softmax_scale=softmax_scale,
            causal=is_causal,
            deterministic=False,
        )
        out_slc = out_slc.view(bsz, seqlen_q, q_heads, self.hidden_dim)

        out_win, _ = flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=(self.window_size, self.window_size),
            deterministic=False,
        )
        # b,s,h,3
        gate = self.gate_proj(q)
        gate_score = F.sigmoid(gate)

        # b,s,3,h,d
        out_stack = torch.stack([out_cmp, out_slc, out_win], dim=2)
        output = torch.einsum("bshc,bschd->bshd", gate_score, out_stack)

        return output


# extract input to blocks
def extract_blocks(input: torch.Tensor, l: int, d: int):  # noqa: E741
    bsz, seqlen, heads, dim = input.shape
    num_blocks = (seqlen - l) // d + 1
    device = input.device
    start_indices = torch.arange(0, num_blocks * d, d, device=device)
    offsets = torch.arange(l, device=device)
    # [b,num_blocks,l]
    gather_indices = start_indices[:, None] + offsets[None, :]
    gather_indices = gather_indices.unsqueeze(0).repeat(bsz, 1, 1)

    # b,s,l.h,d
    input_expand = input.unsqueeze(2).expand(-1, -1, l, -1, -1)
    # b,n,l,h,d
    gather_indices = (
        gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, heads, dim)
    )
    blocks = torch.gather(input_expand, dim=1, index=gather_indices)
    # b,h,n,l,d
    return blocks.permute(0, 3, 1, 2, 4).contiguous()


# compute P_slc from P_cmp
def compute_p_slc(P_cmp, l_slc, l_cmp, d, num_blocks_slc):
    bsz, heads, seqlen, num_blocks_cmp = P_cmp.shape
    alpha = l_slc // d
    beta = l_cmp // d

    dtype = P_cmp.dtype
    device = P_cmp.device
    P_slc = torch.zeros(bsz, heads, seqlen, num_blocks_slc, dtype=dtype, device=device)
    # b,h,s,n_slc
    for j in range(num_blocks_slc):
        for m in range(alpha):
            for n in range(beta):
                idx = alpha * j - m - n
                if 0 <= idx < num_blocks_cmp:
                    P_slc[:, :, :, j] += P_cmp[:, :, :, idx]

    return P_slc


def compute_gqa_p_slc(P_slc: torch.Tensor, kv_heads):
    b, q_heads, s, n_slc = P_slc.shape
    if q_heads == kv_heads:
        return P_slc
    group_size = q_heads // kv_heads
    P_slc_group = P_slc.view(b, -1, group_size, s, n_slc).sum(dim=2, keepdim=True)
    P_slc = P_slc_group.expand(-1, -1, group_size, -1, -1).reshape(b, q_heads, s, n_slc)
    return P_slc


def compute_blockq_p_slc(q: torch.Tensor, P_slc: torch.Tensor, block_size_q):
    bsz, seqlen_q, heads = q.shape[:3]
    num_blocks_slc = P_slc.shape[-1]
    # b,h,-1,block_size_q,n_slc
    P_slc_group = P_slc.view(bsz, heads, -1, block_size_q, num_blocks_slc)
    group_sum = P_slc_group.sum(dim=-2, keepdim=True)
    # b,h,s,n_slc
    P_slc = group_sum.expand(-1, -1, -1, block_size_q, -1).reshape(
        bsz, heads, seqlen_q, num_blocks_slc
    )
    return P_slc


dtype = torch.float16
nsa = NSA(
    hidden_dim=128,
    d=10,
    l_cmp=10,
    l_slc=20,
    window_size=10,
    block_size_q=5,
    slc_top_k=2,
    dtype=dtype,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nsa.to(device)

batch_size = 2
seqlen = 100
heads = 8
hidden_dim = 128
q = torch.randn(
    batch_size,
    seqlen,
    heads,
    hidden_dim,
    dtype=dtype,
    device="cuda",
    requires_grad=True,
)
k = torch.randn(
    batch_size,
    seqlen,
    heads,
    hidden_dim,
    dtype=dtype,
    device="cuda",
    requires_grad=True,
)
v = torch.randn(
    batch_size,
    seqlen,
    heads,
    hidden_dim,
    dtype=dtype,
    device="cuda",
    requires_grad=True,
)

output = nsa(q, k, v, None, False)

loss = output.sum()
loss.backward()

grad_q = q.grad
grad_k = k.grad
grad_v = v.grad

print(grad_q.shape, grad_k.shape, grad_v.shape)
