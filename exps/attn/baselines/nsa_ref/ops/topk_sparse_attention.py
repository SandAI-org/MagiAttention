# Copyright 2025 Xunhao Lai & Jianqiao Lu.
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
import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from .utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel_orig(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    block_size,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # META parameters
    # q loop num
    num_q_loop: tl.constexpr,
    num_k_loop: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid = tl.program_id(0)

    Q = MAX_SEQ_LEN // num_q_loop
    HK = NUM_KV_HEADS // num_k_loop

    # 第几个 (b, kh_chunk, q_chunk)
    pid_b = pid // (HK * Q)
    pid_kh_chunk = (pid % (HK * Q)) // Q  # 每个block处理num_k_loop个KV head
    pid_q = pid % Q

    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    if pid_q * num_q_loop >= q_len:
        return
    real_q_loop = min(num_q_loop, q_len - pid_q * num_q_loop)

    for kh_offset in range(num_k_loop):
        pid_kh = pid_kh_chunk * num_k_loop + kh_offset
        pid_h = pid_kh * NUM_SHARE_Q_HEADS

        for j in range(real_q_loop):
            pid_q_j = pid_q * num_q_loop + j
            # init topk idx pointer
            off_t = tl.arange(0, BLOCK_SIZE_T)
            t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
            topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)

            """Removed causal attention, which should be:
            real_topk = tl.sum(
                tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // block_size), 1, 0),
                axis=0,
            )
            """
            real_topk = tl.sum(
                tl.where((topk_idx >= 0), 1, 0),
                axis=0,
            )
            # real_topk = tl.sum(
            #     tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // block_size), 1, 0),
            #     axis=0,
            # )
            # init qkv pointer
            q_ptrs = tl.make_block_ptr(
                base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
                shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
                strides=(stride_qh, stride_qd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(1, 0),
            )
            k_ptrs = tl.make_block_ptr(
                base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
                shape=(HEAD_DIM, k_len),
                strides=(stride_kd, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
                order=(0, 1),
            )
            v_ptrs = tl.make_block_ptr(
                base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
                shape=(k_len, HEAD_DIM),
                strides=(stride_vn, stride_vd),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                order=(1, 0),
            )
            # load q
            q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
            # init statistics
            off_h = tl.arange(0, BLOCK_SIZE_H)
            off_k = tl.arange(0, BLOCK_SIZE_K)
            m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
            acc_o = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_D), 0, dtype=tl.float32)
            # sparse attention
            for i in range(real_topk):
                # get current block start index
                c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
                t_ptr_j = t_ptr_j + stride_tk
                # load k
                k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")
                # compute qk
                qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
                # qk += tl.where((pid_q_j >= c + off_k)[None, :], 0, float("-inf"))
                # [BLOCK_SIZE_H, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
                qk += tl.dot(q, k) * qk_scale
                # compute m_ij and l_ij
                m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                # scale acc_o
                acc_o_scale = tl.exp2(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                # load v and update acc_o
                v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)
                # update statistics
                m_i = m_ij
                lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)

            # final scale
            acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
            # save output
            o_ptrs = tl.make_block_ptr(
                base=o_ptr + (q_start + pid_q_j) * stride_on + pid_h * stride_oh,
                shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
                strides=(stride_oh, stride_od),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
                order=(1, 0),
            )
            tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
            # save lse
            lse_ptrs = lse_ptr + (q_start + pid_q_j) * stride_ln + (pid_h + off_h) * stride_lh
            tl.store(lse_ptrs, lse_i, mask=off_h < NUM_SHARE_Q_HEADS)


@triton.jit
def backward_sum_o_do(
    o_ptr,  # O: n x h x d
    do_ptr,  # dO: n x h x d
    delta_ptr,  # D: h x n
    o_len,
    HEAD_DIM,
    stride_on,
    stride_oh,
    stride_od,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dh,
    stride_dn,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    off_o = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr + off_o[:, None] * stride_on + pid_h * stride_oh + off_d[None, :] * stride_od,
        mask=(off_o[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr + off_o[:, None] * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod,
        mask=(off_o[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(delta_ptr + pid_h * stride_dh + off_o * stride_dn, delta, mask=off_o < o_len)


@triton.jit
def count_kernel(
    x_ptr,  # [num_kv_heads, total_len, topk]
    y_ptr,  # [num_kv_heads, total_blocks]
    cu_seqlens,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    topk,
    stride_xh,
    stride_xn,
    stride_xk,
    stride_yh,
    stride_yn,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_b = tl.program_id(1)
    # get start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    seq_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    blocks_start = tl.load(cu_seqblocks + pid_b)
    num_blocks = tl.load(cu_seqblocks + pid_b + 1) - blocks_start
    # load x
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptr = x_ptr + pid_h * stride_xh + seq_start * stride_xn
    x_ptrs = x_ptr + off_n[:, None] * stride_xn + off_k[None, :] * stride_xk
    # init y
    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
    # loop
    for i in range(0, seq_len, BLOCK_SIZE_N):
        x = tl.load(
            x_ptrs,
            mask=(off_n < seq_len - i)[:, None] & (off_k < topk)[None, :],
            other=-1,
        )
        x = tl.ravel(x)
        y += tl.histogram(x, BLOCK_SIZE_R)
        x_ptrs += BLOCK_SIZE_N * stride_xn
    # store result
    off_r = tl.arange(0, BLOCK_SIZE_R)
    y_ptr = y_ptr + pid_h * stride_yh + blocks_start * stride_yn
    y_ptrs = y_ptr + off_r * stride_yn
    tl.store(y_ptrs, y.to(y_ptr.dtype.element_ty), mask=off_r < num_blocks)


def count_query(
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqblocks: torch.Tensor,
    block_size: int,
):
    num_kv_heads, total_len, topk = topk_idx.shape
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    seqblocks = cu_seqblocks[1:] - cu_seqblocks[:-1]
    batch_size = seqlens.shape[0]
    BLOCK_SIZE_K = triton.next_power_of_2(topk)
    BLOCK_SIZE_N = triton.next_power_of_2(4096 // BLOCK_SIZE_K)
    BLOCK_SIZE_R = triton.next_power_of_2(seqblocks.max().item() + 2)
    active_query_count = torch.zeros(num_kv_heads, cu_seqblocks[-1], dtype=torch.int32, device=topk_idx.device)
    grid = (num_kv_heads, batch_size)
    count_kernel[grid](
        topk_idx,
        active_query_count,
        cu_seqlens,
        cu_seqblocks,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        active_query_count.stride(0),
        active_query_count.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_R=BLOCK_SIZE_R,
        num_warps=4,
        num_stages=3,
    )
    return active_query_count


@triton.jit
def pad_topk_idx_kernel(
    t_ptr,
    p_ptr,
    cu_seqlens,
    topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_pk,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    # get q start and len after rmpad
    q_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - q_start
    if BLOCK_SIZE_N * pid_n >= q_len:
        return
    # init prts
    t_ptrs = tl.make_block_ptr(
        base=t_ptr + pid_h * stride_th + q_start * stride_tn,
        shape=(q_len, topk),
        strides=(stride_tn, stride_tk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_T),
        order=(1, 0),
    )
    p_ptrs = tl.make_block_ptr(
        base=p_ptr + pid_b * stride_pb + pid_h * stride_ph,
        shape=(q_len, topk),
        strides=(stride_pn, stride_pk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_T),
        order=(1, 0),
    )
    # load and save
    idxs = tl.load(t_ptrs, boundary_check=(0, 1))
    tl.store(p_ptrs, idxs, boundary_check=(0, 1))


@triton.jit
def save_topk_idx_kernel(
    p_ptr,
    t_ptr,
    cu_seqblocks,
    cu_topk_q_count,
    n_len,
    stride_pb,
    stride_ph,
    stride_pn,
    stride_th,
    stride_tn,
    stride_ch,
    stride_cn,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    # get q start and len after rmpad
    q_block_start = tl.load(cu_seqblocks + pid_b)
    q_block_end = tl.load(cu_seqblocks + pid_b + 1)
    c_start = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_start * stride_cn)
    c_end = tl.load(cu_topk_q_count + pid_h * stride_ch + q_block_end * stride_cn)
    c_len = c_end - c_start
    if c_len <= 0:
        return
    if pid_n * BLOCK_SIZE_N >= c_len:
        return
    # init ptrs
    p_ptrs = tl.make_block_ptr(
        base=p_ptr + pid_b * stride_pb + pid_h * stride_ph + (n_len - c_len) * stride_pn,
        shape=(c_len,),
        strides=(stride_pn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    t_ptrs = tl.make_block_ptr(
        base=t_ptr + pid_h * stride_th + c_start * stride_tn,
        shape=(c_len,),
        strides=(stride_tn,),
        offsets=(pid_n * BLOCK_SIZE_N,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )
    # load and save
    idxs = tl.load(p_ptrs, boundary_check=(0,))
    tl.store(t_ptrs, idxs, boundary_check=(0,))


def reorder_topk_idx(
    topk_idx: torch.Tensor,
    cu_topk_q_count: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqblocks: torch.Tensor,
    block_size: int,
):
    num_kv_heads, total_len, topk = topk_idx.shape
    batch_size = cu_seqlens.shape[0] - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seq_lens.max().item()
    # pad shape [num_kv_heads, total_seqlen, topk] to [batch_size, num_kv_heads, max_seqlen, topk]
    pad_topk_idx = torch.full(
        (batch_size, num_kv_heads, max_seqlen, topk),
        fill_value=-1,
        device=topk_idx.device,
        dtype=torch.int32,
    )
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    BLOCK_SIZE_N = min(triton.next_power_of_2(max_seqlen), triton.next_power_of_2(8192 // BLOCK_SIZE_T))
    grid = (batch_size, num_kv_heads, triton.cdiv(max_seqlen, BLOCK_SIZE_N))
    pad_topk_idx_kernel[grid](
        topk_idx,
        pad_topk_idx,
        cu_seqlens,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        pad_topk_idx.stride(0),
        pad_topk_idx.stride(1),
        pad_topk_idx.stride(2),
        pad_topk_idx.stride(3),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
    )
    # argsort
    pad_topk_q_idx = pad_topk_idx.view(batch_size, num_kv_heads, -1).argsort(-1) // topk
    pad_topk_q_idx = pad_topk_q_idx.to(torch.int32)
    # save as remove pad version
    topk_q_idx = torch.full(
        (num_kv_heads, cu_topk_q_count[:, -1].max().item()),
        fill_value=-1,
        device=topk_idx.device,
        dtype=torch.int32,
    )
    max_len = (cu_topk_q_count[:, cu_seqblocks][:, 1:] - cu_topk_q_count[:, cu_seqblocks][:, :-1]).max().item()
    BLOCK_SIZE_N = min(triton.next_power_of_2(max_len), 8192)
    grid = (batch_size, num_kv_heads, triton.cdiv(max_len, BLOCK_SIZE_N))
    save_topk_idx_kernel[grid](
        pad_topk_q_idx,
        topk_q_idx,
        cu_seqblocks,
        cu_topk_q_count,
        pad_topk_q_idx.shape[-1],
        pad_topk_q_idx.stride(0),
        pad_topk_q_idx.stride(1),
        pad_topk_q_idx.stride(2),
        topk_q_idx.stride(0),
        topk_q_idx.stride(1),
        cu_topk_q_count.stride(0),
        cu_topk_q_count.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return topk_q_idx


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    tq_ptr,  # topk_q_idx: kh x N
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DK: sh x n x kh x d
    # seqlens
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    cu_seqblocks,  # [batch_size + 1]
    cu_topk_q_count,  # [kh, total_blocks]
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tqh,
    stride_tqn,
    stride_ctqh,
    stride_ctqn,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    # get topk_q_idx
    b_start = tl.load(cu_seqblocks + pid_b)  # how many blocks before current sequence
    act_q_start = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + pid_k) * stride_ctqn)
    act_q_end = tl.load(cu_topk_q_count + pid_kh * stride_ctqh + (b_start + pid_k + 1) * stride_ctqn)
    act_q_len = act_q_end - act_q_start
    tq_ptr = tq_ptr + pid_kh * stride_tqh + act_q_start * stride_tqn
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dk_ptrs = tl.make_block_ptr(
        base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    dv_ptrs = tl.make_block_ptr(
        base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
        shape=(k_len, HEAD_DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    off_d = tl.arange(0, BLOCK_SIZE_D)
    # load k v and keep in SRAM
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dk dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    # init ptrs
    q_ptrs = q_ptr + q_start * stride_qn + pid_h * stride_qh + off_d[None, :] * stride_qd
    do_ptrs = do_ptr + q_start * stride_don + pid_h * stride_doh + off_d[None, :] * stride_dod
    d_ptrs = d_ptr + q_start * stride_dn + pid_h * stride_dh
    lse_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh
    # loop for q blocks
    for i in range(0, act_q_len, BLOCK_SIZE_Q):
        # load
        idx_q = tl.load(tq_ptr + i + off_q, mask=off_q < act_q_len - i, other=0).to(tl.int32)
        q = tl.load(
            q_ptrs + idx_q[:, None] * stride_qn,
            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
            other=0,
        )
        do = tl.load(
            do_ptrs + idx_q[:, None] * stride_don,
            mask=(off_q < act_q_len - i)[:, None] & (off_d < HEAD_DIM)[None, :],
            other=0,
        )
        lse = tl.load(
            lse_ptrs + idx_q[:, None] * stride_ln,
            mask=(off_q < act_q_len - i)[:, None],
            other=0,
        )
        d = tl.load(
            d_ptrs + idx_q[:, None] * stride_dn,
            mask=(off_q < act_q_len - i)[:, None],
            other=0,
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(idx_q[:, None] >= off_k[None, :], float(0.0), float("-inf"))
        qk += tl.dot(q, k.T) * qk_scale
        # compute p, ds
        p = tl.exp2(qk - lse)
        dp = tl.dot(do, v.T)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        # update dk and dv
        dk += tl.dot(ds.T, q)
        dv += tl.dot(p.T, do)
    # save dk dv
    tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def backward_dq(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    t_ptr,  # topk_idx: kh x n x k
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dq_ptr,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    TOPK,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_q = tl.program_id(2)
    pid_h = pid_kh * NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if pid_q * num_q_loop >= q_len:
        return
    real_q_loop = min(num_q_loop, q_len - pid_q * num_q_loop)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        off_t = tl.arange(0, BLOCK_SIZE_T)
        t_ptr_j = t_ptr + (q_start + pid_q_j) * stride_tn + pid_kh * stride_th
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < TOPK, other=-1)

        real_topk = tl.sum(
            tl.where((topk_idx >= 0) & (topk_idx <= pid_q_j // BLOCK_SIZE_K), 1, 0),
            axis=0,
        )
        # init pointers
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + (q_start + pid_q_j) * stride_qn + pid_h * stride_qh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_qh, stride_qd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        dq_ptrs = tl.make_block_ptr(
            base=dq_ptr + (q_start + pid_q_j) * stride_dqn + pid_h * stride_dqh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_dqh, stride_dqd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k_ptrs = tl.make_block_ptr(
            base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
            shape=(k_len, HEAD_DIM),
            strides=(stride_kn, stride_kd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
            order=(1, 0),
        )
        v_ptrs = tl.make_block_ptr(
            base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
            shape=(HEAD_DIM, k_len),
            strides=(stride_vd, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
            order=(0, 1),
        )
        do_ptrs = tl.make_block_ptr(
            base=do_ptr + (q_start + pid_q_j) * stride_don + pid_h * stride_doh,
            shape=(NUM_SHARE_Q_HEADS, HEAD_DIM),
            strides=(stride_doh, stride_dod),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(1, 0),
        )
        d_ptrs = tl.make_block_ptr(
            base=d_ptr + (q_start + pid_q_j) * stride_dn + pid_h * stride_dh,
            shape=(NUM_SHARE_Q_HEADS, 1),
            strides=(stride_dh, stride_dn),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, 1),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + (q_start + pid_q_j) * stride_ln + pid_h * stride_lh,
            shape=(NUM_SHARE_Q_HEADS, 1),
            strides=(stride_lh, stride_ln),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, 1),
            order=(1, 0),
        )
        # offsets
        off_k = tl.arange(0, BLOCK_SIZE_K)
        # load q, do, lse, delta, and keep in SRAM
        q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option="zero")
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
        # init dq
        dq = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
        # sparse
        for i in range(real_topk):
            # get current block start index
            c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            # load
            k = tl.load(tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero")
            v = tl.load(tl.advance(v_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero")
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((pid_q_j >= c + off_k)[None, :], 0, float("-inf"))
            # [BLOCK_SIZE_H, HEAD_DIM] @ [BLOCK_SIZE_K, HEAD_DIM].T -> [BLOCK_SIZE_H, BLOCK_SIZE_K]
            qk += tl.dot(q, tl.trans(k)) * qk_scale
            # compute p, ds
            p = tl.exp2(qk - lse)
            dp = tl.dot(do, v)
            ds = sm_scale * p * (dp - d)
            # cast dtype
            ds = ds.to(q.dtype)
            # update dq
            dq += tl.dot(ds, k)
        # save dq
        tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))


def _topk_sparse_attention_fwd(
    q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    # assert q_len == k_len and k_len == v_len
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    assert topk_idx.shape[1] == q_len
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # output tensor
    o = torch.zeros_like(q)

    lse = torch.zeros(num_q_heads, q_len, dtype=torch.float32, device=q.device)

    # launch kernel
    num_q_loop = num_k_loop = 1
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)

    def grid(meta):
        grid = (
            batch_size * triton.cdiv(num_k_heads, num_k_loop) * triton.cdiv(max_seqlen_q, num_q_loop),
        )
        return grid

    num_warps, num_stages = get_num_warps_stages(head_dim, block_size, IS_HOPPER_GPU)
    forward_kernel_orig[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        block_size,
        # num_q_loop,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_q_loop=num_q_loop,
        num_k_loop=num_k_loop,
        MAX_SEQ_LEN=max_seqlen_q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse


def _topk_sparse_attention_bwd(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    permute_results,
):

    assert block_size in {32, 64, 128, 256}
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    topk = topk_idx.shape[-1]
    # compute D
    delta = torch.zeros([num_o_heads, o_len], device=o.device, dtype=torch.float32)
    BLOCK_SIZE_O = 256
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O, IS_HOPPER_GPU)
    grid = (triton.cdiv(o_len, BLOCK_SIZE_O), num_o_heads)

    backward_sum_o_do[grid](
        o,
        do,
        delta,
        o_len,
        head_dim,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        delta.stride(0),
        delta.stride(1),
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # count active querys for each key block, shape: (num_k_heads, total_k_blocks)
    seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqblocks = torch.ceil(seqlens / block_size).to(torch.int32)
    cu_seqblocks = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(seqblocks, dim=0),
        ]
    ).to(torch.int32)

    topk_q_count = count_query(topk_idx, cu_seqlens_q, cu_seqblocks, block_size)

    cu_topk_q_count = torch.cat(
        [
            torch.zeros(topk_q_count.shape[0], 1, dtype=torch.int32, device=topk_idx.device),
            torch.cumsum(topk_q_count, dim=-1),
        ],
        dim=-1,
    ).to(torch.int32)
    # active query idx for each key block
    # how to get active query idx for sequence b, head h, kv block i?
    topk_q_idx = reorder_topk_idx(topk_idx, cu_topk_q_count, cu_seqlens_q, cu_seqblocks, block_size)
    # compute dk dv
    dk = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
    dv = torch.zeros(num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype)
    batch_size = cu_seqlens_q.shape[0] - 1
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
    grid = (batch_size, num_q_heads, triton.cdiv(max_seqlen_k, BLOCK_SIZE_K))
    backward_dkdv[grid](
        q,
        k,
        v,
        topk_q_idx,
        lse,
        delta,
        do,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqblocks,
        cu_topk_q_count,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        topk_q_idx.stride(0),
        topk_q_idx.stride(1),
        cu_topk_q_count.stride(0),
        cu_topk_q_count.stride(1),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dk = dk.sum(0)
    dv = dv.sum(0)
    # compute dq
    dq = torch.zeros_like(q)
    num_q_loop = max_seqlen_q // 32768 + 1  # calculate multiple querys in one kernel if seqlence length is too long
    grid = (batch_size, num_k_heads, triton.cdiv(max_seqlen_q, num_q_loop))
    BLOCK_SIZE_K = block_size
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    BLOCK_SIZE_H = max(16, triton.next_power_of_2(num_share_q_heads))
    BLOCK_SIZE_T = triton.next_power_of_2(topk)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)

    backward_dq[grid](
        q,
        k,
        v,
        topk_idx,
        lse,
        delta,
        do,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        topk,
        num_q_loop,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_T=BLOCK_SIZE_T,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return dq, dk, dv


class TopkSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # [total_len, num_q_heads, head_dim]
        k: torch.Tensor,  # [total_len, num_k_heads, head_dim]
        v: torch.Tensor,  # [total_len, num_k_heads, head_dim]
        topk_idx: torch.Tensor,  # [num_kv_heads, total_len, topk]
        block_size: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: torch.Tensor,
        max_seqlen_k: torch.Tensor,
        sm_scale=None,
    ):
        # dtype check
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert topk_idx.dtype == torch.int32
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])

        permute_results = None

        o, lse = _topk_sparse_attention_fwd(
            q,
            k,
            v,
            topk_idx,
            block_size,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx)
        ctx.permute_results = permute_results
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k, topk_idx = ctx.saved_tensors
        permute_results = ctx.permute_results

        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        block_size = ctx.block_size
        assert block_size in {32, 64, 128, 256}

        dq, dk, dv = _topk_sparse_attention_bwd(
            o,
            do,
            lse,
            q,
            k,
            v,
            topk_idx,
            block_size,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
            permute_results,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def topk_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size: int,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Topk sparse attention varlen version implemented in triton.

    Args:
        q (torch.Tensor): shape [total_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_len, num_kv_heads, head_dim]
        topk_idx (torch.Tensor): topk block idx for each query, shape [num_kv_heads, total_len, topk]. -1 means padding.
        block_size (int): key value block size.
        cu_seqlens (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens in flash_attn_func_varlen.
        softmax_scale (Optional[float], optional): Defaults to None, means 1/sqrt(head_dim).

    Returns:
        torch.Tensor: attention output, shape [total_len, num_q_heads, head_dim]
    """

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    return TopkSparseAttention.apply(
        q,
        k,
        v,
        topk_idx,
        block_size,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale,
    )
