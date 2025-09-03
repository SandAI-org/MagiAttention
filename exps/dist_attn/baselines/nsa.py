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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from flashattn_hopper.flash_attn_interface import (
#     flash_attn_func,
#     flash_attn_varlen_func,
# )
from flash_attn_interface import flash_attn_func, flash_attn_varlen_func

from magi_attention.common.ranges import AttnRanges


# varlen thd format
class VarlenNSA(nn.Module):
    def __init__(
        self,
        hidden_dim,
        d,
        l_cmp,
        l_slc,
        window_size_left,
        window_size_right,
        block_size_q,
        slc_top_k,
        dtype,
        device,
    ):
        super(VarlenNSA, self).__init__()
        self.d = d
        self.l_cmp = l_cmp
        self.l_slc = l_slc
        self.slc_top_k = slc_top_k
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.block_size_q = block_size_q
        self.hidden_dim = hidden_dim

        # cmp mlp layer
        self.cmp_linear_k = nn.Linear(self.l_cmp, 1, dtype=dtype, device=device)
        self.cmp_linear_v = nn.Linear(self.l_cmp, 1, dtype=dtype, device=device)
        # cmp/slc/win
        self.gate_proj = nn.Linear(hidden_dim, 3, dtype=dtype, device=device)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges,
        k_ranges,
        softmax_scale,
        is_causal,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        total_seqlen_q, q_heads = q.shape[0], q.shape[1]
        total_seqlen_kv, kv_heads = k.shape[0], k.shape[1]
        H = q_heads // kv_heads

        host_cu_seqlens_q = q_ranges.to_cu_seqlens(total_seqlen_q)
        host_cu_seqlens_kv = k_ranges.to_cu_seqlens(total_seqlen_kv)
        cu_seqlens_kv_np = np.array(host_cu_seqlens_kv, dtype=np.int32)
        seqlens_kv_np = cu_seqlens_kv_np[1:] - cu_seqlens_kv_np[:-1]
        cmp_seqlens_kv_np = (seqlens_kv_np - self.l_cmp) // self.d + 1
        slc_blk_num_kv_np = (seqlens_kv_np - self.l_slc) // self.d + 1

        # h,n_cmp,l,d
        K_cmp_blocks = extract_blocks(k.unsqueeze(0), self.l_cmp, self.d)
        V_cmp_blocks = extract_blocks(v.unsqueeze(0), self.l_cmp, self.d)
        K_cmp_blocks = K_cmp_blocks.squeeze(0)
        V_cmp_blocks = V_cmp_blocks.squeeze(0)

        # h,n_cmp,d
        K_cmp = self.cmp_linear_k(K_cmp_blocks.transpose(-1, -2)).squeeze(-1)
        V_cmp = self.cmp_linear_v(V_cmp_blocks.transpose(-1, -2)).squeeze(-1)
        # GQA repeat
        K_cmp_mha = K_cmp.repeat_interleave(H, dim=0)
        V_cmp_mha = V_cmp.repeat_interleave(H, dim=0)

        if not (self.l_slc == self.l_cmp == self.d):
            assert self.l_slc > self.l_cmp, "l_slc must be greater than l_cmp"
            assert self.l_slc % self.d == 0, "l_slc must be divisible by d"
            assert self.l_cmp % self.d == 0, "l_cmp must be divisible by d"

        # t,h,d
        out_cmp = torch.empty_like(q)
        # h,t,k
        idx_slc = torch.empty(
            q_heads, total_seqlen_q, self.slc_top_k, dtype=torch.int64, device=q.device
        )
        total_seq_num = len(seqlens_kv_np)
        cmp_kv_off = 0
        total_slc_blocks = 0
        for i in range(total_seq_num):
            start_q, end_q = q_ranges[i].start, q_ranges[i].end
            q_part = q[start_q:end_q, :, :]
            k_part = K_cmp_mha[:, cmp_kv_off : cmp_kv_off + cmp_seqlens_kv_np[i], :]
            v_part = V_cmp_mha[:, cmp_kv_off : cmp_kv_off + cmp_seqlens_kv_np[i], :]
            cmp_kv_off += cmp_seqlens_kv_np[i]

            # s,h,d @ h,sk,d, -> h,s,sk
            attn_cmp_part = torch.einsum("shd,hnd->hsn", q_part, k_part)
            attn_cmp_part = attn_cmp_part.to(torch.float32) * softmax_scale
            P_cmp_part = F.softmax(attn_cmp_part, dim=-1).to(q.dtype)
            # h,s,sk @ h,sk,d -> s,h,d
            out_cmp_part = torch.einsum("hsn,hnd->shd", P_cmp_part, v_part)

            out_cmp[start_q:end_q, :, :] = out_cmp_part

            # compute P_slc, h,s,sk
            if self.l_slc == self.l_cmp == self.d:
                num_blocks_slc = P_cmp_part.shape[-1]
                P_slc_part = P_cmp_part
            else:
                num_blocks_slc = slc_blk_num_kv_np[i]
                P_slc_part = compute_p_slc(
                    P_cmp_part, self.l_slc, self.l_cmp, self.d, num_blocks_slc
                )

            # deal q_block_size
            P_slc_part = compute_blockq_p_slc(q_part, P_slc_part, self.block_size_q)
            # deal GQA
            P_slc_part = compute_gqa_p_slc(P_slc_part, kv_heads)

            assert (
                self.slc_top_k <= num_blocks_slc
            ), "slc_top_k must be less than or equal to num_blocks_slc"
            # h,sk,k
            _, idx_slc_part = torch.topk(P_slc_part, dim=-1, k=self.slc_top_k)
            idx_slc_part += total_slc_blocks
            total_slc_blocks += num_blocks_slc
            idx_slc[:, start_q:end_q, :] = idx_slc_part

        # compute out_slc
        if self.l_slc == self.l_cmp == self.d:
            # h,n,l_slc,d
            K_slc_blocks = K_cmp_blocks
            V_slc_blocks = V_cmp_blocks
        else:
            # h,n,l_slc,d
            K_slc_blocks = extract_blocks(k.unsqueeze(0), self.l_slc, self.d)
            V_slc_blocks = extract_blocks(v.unsqueeze(0), self.l_slc, self.d)
            K_slc_blocks = K_slc_blocks.squeeze(0)
            V_slc_blocks = V_slc_blocks.squeeze(0)

        # h,s,k,l_slc,d
        idx_exp = (
            idx_slc.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, self.l_slc, self.hidden_dim)
        )
        # GQA h_kv,H,s,k,l_slc,d
        idx_exp = idx_exp.view(kv_heads, -1, *idx_exp.shape[1:])
        # GQA h_kv,H,s,n_slc,l_slc,d
        K_slc_exp = (
            K_slc_blocks.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, H, total_seqlen_q, -1, -1, -1)
        )
        V_slc_exp = (
            V_slc_blocks.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, H, total_seqlen_q, -1, -1, -1)
        )
        # h,s,k*l_slc,d
        K_slc = torch.gather(K_slc_exp, dim=3, index=idx_exp)[:, 0, ...]
        K_slc = K_slc.view(kv_heads, total_seqlen_q, -1, self.hidden_dim)
        V_slc = torch.gather(V_slc_exp, dim=3, index=idx_exp)[:, 0, ...]
        V_slc = V_slc.view(kv_heads, total_seqlen_q, -1, self.hidden_dim)

        # q， t,h,d
        # kv， h,t,sk,d
        # b*s,1,h,d
        # t,sk,h,d
        q_slc_fa = q.view(total_seqlen_q, 1, q_heads, self.hidden_dim)
        k_slc_fa = K_slc.permute(1, 2, 0, 3).contiguous()
        v_slc_fa = V_slc.permute(1, 2, 0, 3).contiguous()
        out_slc = flash_attn_func(
            q_slc_fa,
            k_slc_fa,
            v_slc_fa,
            softmax_scale=softmax_scale,
            causal=is_causal,
            deterministic=True,
        )
        out_slc = out_slc.squeeze(1)  # t,h,d

        cu_seqlens_q = torch.tensor(
            host_cu_seqlens_q, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = torch.tensor(
            host_cu_seqlens_kv, dtype=torch.int32, device=k.device
        )
        max_seqlen_q = q_ranges.max_seqlen
        max_seqlen_k = k_ranges.max_seqlen
        out_win = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=(self.window_size_left, self.window_size_right),
            deterministic=True,
        )
        # t,h,3
        gate = self.gate_proj(q)
        gate_score = F.sigmoid(gate)

        # out_slc = torch.zeros_like(out_cmp)
        # out_win = torch.zeros_like(out_cmp)

        # t,3,h,d
        out_stack = torch.stack([out_cmp, out_slc, out_win], dim=1)
        output = torch.einsum("thc,tchd->thd", gate_score, out_stack)

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
    # bsz, heads, seqlen, num_blocks_cmp = P_cmp.shape
    is_varlen = P_cmp.dim() == 3
    p_slc_shape = list(P_cmp.shape)
    num_blocks_cmp = P_cmp.shape[-1]
    p_slc_shape[-1] = num_blocks_slc
    alpha = l_slc // d
    beta = l_cmp // d

    dtype = P_cmp.dtype
    device = P_cmp.device
    P_slc = torch.zeros(*p_slc_shape, dtype=dtype, device=device)
    # b,h,s,n_slc
    for j in range(num_blocks_slc):
        for m in range(alpha):
            for n in range(beta):
                idx = alpha * j - m - n
                if 0 <= idx < num_blocks_cmp:
                    if is_varlen:
                        P_slc[:, :, j] += P_cmp[:, :, idx]
                    else:
                        P_slc[:, :, :, j] += P_cmp[:, :, :, idx]

    return P_slc


def compute_gqa_p_slc(P_slc: torch.Tensor, kv_heads: int):
    is_varlen = P_slc.dim() == 3
    if is_varlen:
        P_slc = P_slc.unsqueeze(0)

    b, q_heads, s, n_slc = P_slc.shape
    if q_heads == kv_heads:
        return P_slc.squeeze(0) if is_varlen else P_slc

    group_size = q_heads // kv_heads
    P_slc_group = P_slc.view(b, kv_heads, group_size, s, n_slc).sum(dim=2, keepdim=True)
    P_slc = P_slc_group.expand(-1, -1, group_size, -1, -1).reshape(b, q_heads, s, n_slc)

    return P_slc.squeeze(0) if is_varlen else P_slc


def compute_blockq_p_slc(q: torch.Tensor, P_slc: torch.Tensor, block_size_q):
    is_varlen = q.dim() == 3  # (s, h, d)
    if is_varlen:
        bsz = 1
        seqlen_q, heads = q.shape[:2]
        P_slc = P_slc.unsqueeze(0)
    else:
        bsz, seqlen_q, heads = q.shape[:3]
    num_blocks_slc = P_slc.shape[-1]
    # b,h,-1,block_size_q,n_slc
    P_slc_group = P_slc.view(bsz, heads, -1, block_size_q, num_blocks_slc)
    group_sum = P_slc_group.sum(dim=-2, keepdim=True)
    # b,h,s,n_slc
    P_slc = group_sum.expand(-1, -1, -1, block_size_q, -1).reshape(
        bsz, heads, seqlen_q, num_blocks_slc
    )
    return P_slc.squeeze(0) if is_varlen else P_slc


if __name__ == "__main__":
    # -----   test non-varlen nsa  ---- #

    # dtype = torch.float16
    # nsa = NSA(
    #     hidden_dim=128,
    #     d=10,
    #     l_cmp=10,
    #     l_slc=20,
    #     window_size=10,
    #     block_size_q=5,
    #     slc_top_k=2,
    #     dtype=dtype,
    # )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nsa.to(device)

    # batch_size = 2
    # seqlen = 100
    # heads = 8
    # hidden_dim = 128
    # q = torch.randn(
    #     batch_size,
    #     seqlen,
    #     heads,
    #     hidden_dim,
    #     dtype=dtype,
    #     device="cuda",
    #     requires_grad=True,
    # )
    # k = torch.randn(
    #     batch_size,
    #     seqlen,
    #     heads,
    #     hidden_dim,
    #     dtype=dtype,
    #     device="cuda",
    #     requires_grad=True,
    # )
    # v = torch.randn(
    #     batch_size,
    #     seqlen,
    #     heads,
    #     hidden_dim,
    #     dtype=dtype,
    #     device="cuda",
    #     requires_grad=True,
    # )

    # output = nsa(q, k, v, None, False)

    # loss = output.sum()
    # loss.backward()

    # grad_q = q.grad
    # grad_k = k.grad
    # grad_v = v.grad

    # print(grad_q.shape, grad_k.shape, grad_v.shape)

    # -----   test varlen nsa  ---- #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16
    nsa_varlen = VarlenNSA(
        hidden_dim=128,
        d=16,
        l_cmp=16,
        l_slc=32,
        window_size_left=24,
        window_size_right=24,
        block_size_q=8,
        slc_top_k=2,
        dtype=dtype,
        device=device,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nsa_varlen.to(device)

    q_ranges = AttnRanges.from_ranges(
        [
            [0, 256],
            # [256, 512],
            # [512, 1024],
            # [1024, 1280],
            # [1280, 1536],
            # [1536, 1792],
            # [1792, 2048],
        ]
    )

    k_ranges = AttnRanges.from_ranges(
        [
            [0, 256],
            # [256, 512],
            # [512, 1024],
            # [1024, 1280],
            # [1280, 1536],
            # [1536, 1792],
            # [1792, 2048],
        ]
    )

    seqlen = 256
    heads = 8
    hidden_dim = 128
    q_thd = torch.randn(
        seqlen,
        heads,
        hidden_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    k_thd = torch.randn(
        seqlen,
        heads // 2,
        hidden_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    v_thd = torch.randn(
        seqlen,
        heads // 2,
        hidden_dim,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    output_thd = nsa_varlen(q_thd, k_thd, v_thd, q_ranges, k_ranges, None, False)

    loss_thd = output_thd.sum()
    loss_thd.backward()

    grad_q_thd = q_thd.grad
    grad_k_thd = k_thd.grad
    grad_v_thd = v_thd.grad

    # print(grad_q_thd.shape, grad_k_thd.shape, grad_v_thd.shape)
