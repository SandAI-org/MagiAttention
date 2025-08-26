# Copyright 2025 Ran Yan.
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
# See the License for the specific
import math
from typing import Any, Optional

import torch
import triton
import triton.language as tl

from ...nsa_ref.ops.topk_sparse_attention import (backward_sum_o_do,
                                               reorder_topk_idx)
from ...nsa_ref.ops.utils import get_num_warps_stages, is_hopper_gpu

IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def fused_fill_kernel(ptr_tile, ptr_m_i_cur_tiles, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    tl.store(ptr_tile + offsets, -1, mask=mask)  # fill int32 with -1
    tl.store(ptr_m_i_cur_tiles + offsets, float("-inf"), mask=mask)


def fused_fill(topk_idx_permuted_tile: torch.Tensor, m_i_cur_tiles):

    numel = topk_idx_permuted_tile.numel()
    BLOCK_SIZE = 1024

    # Flatten for pointer access
    tile_flat = topk_idx_permuted_tile.view(-1)

    m_i_cur_tiles_flat = m_i_cur_tiles.view(-1)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    fused_fill_kernel[grid](
        tile_flat,
        m_i_cur_tiles_flat,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=3,
    )


@triton.jit
def block_to_token_kernel(
    topk_idx_ptr,
    result_ptr,
    N_token,
    K,
    min_block_id,
    max_block_id,
    padding_value,
    ts_h,
    ts_b,
    ts_n,
    rs_h,
    rs_b,
    rs_n,
    num_q_loops: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # token index i
    pid_h = 0
    offs = tl.arange(0, BLOCK_K)  # [0, 1, ..., K-1]

    offs_q = tl.arange(0, num_q_loops)

    pid_j = pid * num_q_loops + offs_q

    topk_idx_offset = pid_h * ts_h + pid_j[None, :] * K + offs[:, None]
    block_ids = tl.load(
        topk_idx_ptr + topk_idx_offset, mask=(pid_j < N_token)[None, :] & (offs < K)[:, None], other=padding_value
    )

    result_ptrs = result_ptr + pid_h * rs_h + block_ids * N_token + pid_j[None, :]

    mask = (block_ids >= 0) & (block_ids != padding_value) & (pid_j < N_token)[None, :]
    tl.store(result_ptrs, pid_j[None, :], mask=mask)


def build_block_to_token_triton(
    result: torch.Tensor, topk_idx: torch.Tensor, min_block_id: int, max_block_id: int, padding_value: int = -1
):
    """
    Args:
        topk_idx: [num_heads, N_token, TopK], block indices per token, padded with padding_value for invalid blocks
        num_blocks: int
        padding_value: int

    Returns:
        result: [num_blocks, N_token], token indices per block, padded by padding_value
    """
    assert topk_idx.ndim == 3
    assert padding_value == -1
    num_heads, N_token, TopK = topk_idx.shape

    # 每个 token，每个head 一个 program
    num_q_loops = 4
    grid = (triton.cdiv(N_token, num_q_loops),)
    BLOCK_K = triton.next_power_of_2(TopK)
    block_to_token_kernel[grid](
        topk_idx,
        result,
        N_token,
        TopK,
        min_block_id,
        max_block_id,
        padding_value,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        result.stride(0),
        result.stride(1),
        result.stride(2),
        num_q_loops,
        BLOCK_K=BLOCK_K,
        num_warps=2,
        num_stages=3,
    )
    return result


@triton.jit
def reduce_kernel(
    lse_ptr,  # float32 [H, N]
    m_ij_ptr,  # float32 [H, B, N]
    l_ij_first_ptr,  # float32 [H, 1, N]
    l_ij_rest_ptr,  # float32 [H, B, N]
    m_ij_last_ptr,  # float32 [H, N]
    o_ptr,  # o: n x h x d
    o_tiles_first_ptr,  # o_tiles: n x h x 1 x d
    o_tiles_rest_ptr,  # o_tiles: n x h x b x d
    acc_o_scales_first_ptr,  # acc_o_scales: n x h x 1
    acc_o_scales_rest_ptr,  # acc_o_scales: n x h x b
    t_ptr,  # topk_idx: h x n x k
    token_index_mapping_ptr,
    start_head_id,
    num_qz_loop,
    TOPK,
    total_len,
    # stride
    stride_lse_h,
    stride_lse_n,
    stride_m_ij_h,
    stride_m_ij_b,
    stride_m_ij_n,
    stride_l_ij_fh,
    stride_l_ij_fb,
    stride_l_ij_fn,
    stride_l_ij_rh,
    stride_l_ij_rb,
    stride_l_ij_rn,
    stride_on,
    stride_oh,
    stride_od,
    stride_otfh,
    stride_otfb,
    stride_otfn,
    stride_otfd,
    stride_otrh,
    stride_otrb,
    stride_otrn,
    stride_otrd,
    stride_acc_fh,
    stride_acc_fb,
    stride_acc_fn,
    stride_acc_rh,
    stride_acc_rb,
    stride_acc_rn,
    stride_th,
    stride_tn,
    stride_tk,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_qy = tl.program_id(0)
    pid_q = tl.program_id(1)  # token

    pid_q_j = pid_q + pid_qy * num_qz_loop
    if pid_q_j >= total_len:
        return
    t_ptr_j = t_ptr + pid_q_j * stride_tn

    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = o_ptr + pid_q_j * stride_on + off_d
    last_acc_o = tl.load(o_ptrs, mask=off_d < BLOCK_SIZE_D, other=0.0)
    acc_o = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    acc_o += last_acc_o

    lse_ptrs = (lse_ptr + pid_q_j * stride_lse_n,)
    # Load lse
    lse = tl.load(lse_ptrs, mask=pid_q_j < total_len, other=float("-inf"))

    # the stride is 1 for m_ij_last
    m_ij_last = tl.load(m_ij_last_ptr + pid_q_j)

    for block_id in range(TOPK):
        t = tl.load(t_ptr_j + block_id * stride_tk, mask=block_id < TOPK, other=-1)
        if t != -1:
            if t == 0:
                real_block_pos = 0
                l_ij_ptr = l_ij_first_ptr
                o_tiles_ptr = o_tiles_first_ptr
                acc_o_scales_ptr = acc_o_scales_first_ptr
                stride_l_ij_b = stride_l_ij_fb
                stride_l_ij_n = stride_l_ij_fn
                stride_acc_b = stride_acc_fb
                stride_acc_n = stride_acc_fn
                stride_otb = stride_otfb
                stride_otn = stride_otfn
            else:
                real_block_pos = t - 1
                l_ij_ptr = l_ij_rest_ptr
                o_tiles_ptr = o_tiles_rest_ptr
                acc_o_scales_ptr = acc_o_scales_rest_ptr
                stride_l_ij_b = stride_l_ij_rb
                stride_l_ij_n = stride_l_ij_rn
                stride_acc_b = stride_acc_rb
                stride_acc_n = stride_acc_rn
                stride_otb = stride_otrb
                stride_otn = stride_otrn

            # init pointers
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + t.to(tl.int64) * stride_tim_b + (pid_q_j) * stride_tim_n
            )
            real_token_index = tl.load(token_index_mapping_ptrs)

            m_ij = tl.load(
                m_ij_ptr + t * stride_m_ij_b + pid_q_j * stride_m_ij_n, mask=pid_q_j < total_len, other=float("-inf")
            )
            l_ij = tl.load(
                l_ij_ptr + real_block_pos * stride_l_ij_b + real_token_index * stride_l_ij_n,
                mask=real_token_index < total_len,
                other=0.0,
            )
            delta = lse - m_ij

            log_delta = tl.exp2(delta) + l_ij

            # Update lse
            lse = m_ij + tl.log2(log_delta)

            o_tiles_ptrs = (
                o_tiles_ptr + real_block_pos.to(tl.int64) * stride_otb + (real_token_index) * stride_otn + off_d
            )
            acc_o_scales_ptrs = acc_o_scales_ptr + real_block_pos * stride_acc_b + (real_token_index) * stride_acc_n

            o_tiles = tl.load(o_tiles_ptrs)
            acc_o_scales_tiles = tl.load(acc_o_scales_ptrs)
            acc_o = o_tiles + acc_o * acc_o_scales_tiles

    # final scale
    acc_o = acc_o * tl.exp2(m_ij_last - lse)
    tl.store(o_ptrs, acc_o, mask=off_d < BLOCK_SIZE_D)

    # Store back
    tl.store(
        lse_ptrs,
        lse,
        mask=pid_q_j < total_len,
    )


@triton.jit
def qk_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    m_i_tiles_ptr,  # m_i: h x b x n
    selected_tokens_ptr,  # selected_tokens: sum(valid_lens),
    valid_lens_ptr,  # valid_lens: (h x b),
    valid_start_indices_ptr,  # valid_start_indices: (h x b),
    num_heads,
    num_blocks,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    HEAD_DIM,
    # sm_scale
    sm_scale,
    num_q_blocks,
    num_b_blocks,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_m_i_tiles_h,
    stride_m_i_tiles_b,
    stride_m_i_tiles_n,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_block_grid = tl.program_id(0) // num_heads  # block id
    head_id = tl.program_id(0) % num_heads
    pid_q = tl.program_id(1)  # token

    # get q k start and len after rmpad
    k_len = tl.load(cu_seqlens_k + 1)
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + head_id * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )

    for bb in range(num_b_blocks):
        pid_block = bb + pid_block_grid * num_b_blocks

        start_id = tl.load(valid_start_indices_ptr + head_id * num_blocks + pid_block)
        valid_tokens = tl.load(valid_lens_ptr + head_id * num_blocks + pid_block)
        if pid_q * BLOCK_SIZE_Q < valid_tokens:

            c = pid_block * BLOCK_SIZE_K

            # load k
            k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")

            off_k = tl.arange(0, BLOCK_SIZE_K)
            off_d = tl.arange(0, BLOCK_SIZE_D)
            for j in range(num_q_blocks):
                pid_q_j = pid_q * num_q_blocks + j
                # Enable early return
                if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
                    # one thread block for one KV block, a subset of selected tokens
                    st_offs = start_id + (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
                    # st should be in shape [BLOCK_SIZE_Q]
                    st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

                    st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
                    # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
                    q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd
                    q_ptrs = q_ptr + head_id * stride_qh + q_ptrs_off
                    # load q
                    q_mask = (st != -1)[:, None] & (off_d < HEAD_DIM)[None, :]
                    q = tl.load(q_ptrs, mask=q_mask, other=0)
                    # compute qk
                    qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
                    qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))
                    # [BLOCK_SIZE_Q, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_Q, BLOCK_SIZE_K]
                    qk += tl.dot(q, k) * qk_scale

                    m_i = tl.max(qk, axis=1)

                    m_i_tiles_ptrs = (
                        m_i_tiles_ptr
                        + head_id * stride_m_i_tiles_h
                        + pid_block * stride_m_i_tiles_b
                        + st * stride_m_i_tiles_n
                    )
                    tl.store(m_i_tiles_ptrs, m_i, mask=(st != -1))


@triton.jit
def forward_kernel_opt(
    q_ptr,
    k_ptr,
    v_ptr,  # V: n x h x d
    o_tiles_ptr,  # O: n x h x b x d
    acc_o_scales_ptr,  # acc_o_scales: h x b x n
    m_ij_tiles_ptr,
    l_ij_ptr,  # h x b x n
    token_index_mapping_ptr,
    selected_tokens_ptr,  # selected_tokens: sum(valid_lens),
    valid_lens_ptr,  # valid_lens: (h x b),
    valid_start_indices_ptr,  # valid_start_indices: (h x b),
    min_block_id,
    cur_max_valid_tokens,
    num_heads,
    num_blocks,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    HEAD_DIM,
    # sm_scale
    sm_scale,
    num_q_blocks,
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
    stride_oth,
    stride_otb,
    stride_otn,
    stride_otd,
    stride_acc_oh,
    stride_acc_ob,
    stride_acc_on,
    stride_m_ij_tiles_h,
    stride_m_ij_tiles_b,
    stride_m_ij_tiles_n,
    stride_l_ij_h,
    stride_l_ij_b,
    stride_l_ij_n,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    # get batch id and head id
    pid_block = tl.program_id(0) // num_heads  # block id
    head_id = tl.program_id(0) % num_heads
    pid_q = tl.program_id(1)  # token
    # seq packing is not supported yet
    q_start = 0
    k_start = 0

    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + head_id * num_blocks + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + head_id * num_blocks + pid_block)
    if num_q_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (min_block_id + pid_block) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + head_id * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    # load k
    k = tl.load(tl.advance(k_ptrs, (0, c)), boundary_check=(1, 0), padding_option="zero")

    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + head_id * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # load v
    v = tl.load(tl.advance(v_ptrs, (c, 0)), boundary_check=(0, 1), padding_option="zero")

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    for j in range(num_q_blocks):
        pid_q_j = pid_q * num_q_blocks + j
        if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
            # one thread block for one KV block, a subset of selected tokens
            st_offs = start_id + (q_start + pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            # st should be in shape [BLOCK_SIZE_Q]
            st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

            st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

            # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
            q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd

            # load m_i
            mask = st != -1

            m_ij_tiles_ptrs = (
                m_ij_tiles_ptr
                + head_id * stride_m_ij_tiles_h
                + (q_start + st) * stride_m_ij_tiles_n
                + (pid_block + min_block_id) * stride_m_ij_tiles_b
            )
            m_ij = tl.load(m_ij_tiles_ptrs, mask=mask, other=float("-inf"))

            m_ij_tiles_prev_ptrs = (
                m_ij_tiles_ptr
                + head_id * stride_m_ij_tiles_h
                + (q_start + st) * stride_m_ij_tiles_n
                + (pid_block + min_block_id - 1) * stride_m_ij_tiles_b
            )
            m_ij_prev = tl.load(m_ij_tiles_prev_ptrs, mask=mask & (pid_block + min_block_id > 0), other=float("-inf"))

            m_i_minus_m_ij = m_ij_prev - m_ij

            q_ptrs = q_ptr + q_start * stride_qn + head_id * stride_qh + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)

            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))

            # [BLOCK_SIZE_Q, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIZE_K] -> [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            qk_scale = sm_scale * 1.44269504
            qk += tl.dot(q, k) * qk_scale

            # init statistics
            acc_o_buffer = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0, dtype=tl.float32)

            # load m_ij and compute l_ij
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # load token index mapping
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + (st) * stride_tim_n + (pid_block + min_block_id) * stride_tim_b
            )
            token_index_mapping = tl.load(token_index_mapping_ptrs, mask=mask, other=-1)

            l_ij_ptrs = (
                l_ij_ptr
                + head_id * stride_l_ij_h
                + (q_start + token_index_mapping) * stride_l_ij_n
                + (pid_block) * stride_l_ij_b
            )
            tl.store(l_ij_ptrs, l_ij, mask=mask)
            # scale acc_o
            if pid_block + min_block_id == 0:
                acc_o_scale = tl.full((BLOCK_SIZE_Q,), 1.0, dtype=tl.float32)
            else:
                acc_o_scale = tl.exp2(m_i_minus_m_ij)

            tl.store(
                acc_o_scales_ptr
                + head_id * stride_acc_oh
                + (pid_block) * stride_acc_ob
                + (q_start + token_index_mapping) * stride_acc_on,
                acc_o_scale,
                mask=(st != -1),
            )

            p = p.to(v.dtype)
            acc_o_buffer = tl.dot(p, v)

            o_ptrs_off = token_index_mapping[:, None] * stride_otn + off_d[None, :] * stride_otd
            o_ptrs = o_tiles_ptr + head_id * stride_oth + o_ptrs_off + (pid_block).to(tl.int64) * stride_otb
            tl.store(o_ptrs, acc_o_buffer.to(o_tiles_ptr.dtype.element_ty), mask=q_mask)


def _topk_sparse_attention_fwd_opt(
    q: torch.Tensor,  # [total_len, num_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal=True,
):
    """
        TODO: Currently sequence packing is explicitly done in for loop, will merge in kernels.
    """
    o = torch.empty_like(q)
    total_len, num_heads, _ = q.shape
    lse = torch.empty((num_heads, total_len), dtype=torch.float32, device=q.device)

    permute_results = []
    for i in range(len(cu_seqlens_q) - 1):
        cu_seqlens_q_ = cu_seqlens_q[i: i + 2] - cu_seqlens_q[i]
        cu_seqlens_k_ = cu_seqlens_k[i: i + 2] - cu_seqlens_k[i]
        max_seqlen_q_ = cu_seqlens_q_[1] - cu_seqlens_q_[0]
        max_seqlen_k_ = cu_seqlens_k_[1] - cu_seqlens_k_[0]

        q_ = q[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        k_ = k[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        v_ = v[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        topk_idx_ = topk_idx[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        o_seq, lse_seq, permute_results_seq = _topk_sparse_attention_fwd_opt_per_seq(
            q_,
            k_,
            v_,
            topk_idx_,
            block_size,
            cu_seqlens_q_,
            cu_seqlens_k_,
            max_seqlen_q_,
            max_seqlen_k_,
            sm_scale,
            causal,
        )
        o[cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = o_seq

        lse[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = lse_seq
        permute_results.append(permute_results_seq)

    return o, lse, permute_results


@triton.jit
def index_mapping_kernel(
    token_index_mapping_ptr,
    selected_tokens_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    stride_im_h,
    stride_im_b,
    stride_im_n,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_q = tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_K + offs_q

    start_id = tl.load(valid_start_indices_ptr + pid_b)
    valid_tokens = tl.load(valid_lens_ptr + pid_b)

    st_offs = start_id + offs_n
    # st should be in shape [BLOCK_SIZE_K]
    st_mask = offs_n < valid_tokens

    st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)

    token_im_ptrs = token_index_mapping_ptr + pid_b * stride_im_b + st * stride_im_n

    tl.store(token_im_ptrs, offs_n, mask=st_mask)


def index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks):
    max_tokens = valid_lens.max()
    BLOCK_SIZE_K = 1024
    grid = (num_blocks, triton.cdiv(max_tokens, BLOCK_SIZE_K))

    index_mapping_kernel[grid](
        token_index_mapping,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        BLOCK_SIZE_K,
        num_warps=2,
        num_stages=3,
    )


def online_softmax(
    q_tile,
    k_tile,
    m_i_cur_tiles,
    valid_topk_idx_permuted_tile,
    valid_lens,
    valid_start_indices,
    compute_min_block_id,
    cur_max_valid_tokens,
    block_size,
    num_blocks,
    head_tile,
    head_dim,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
):

    # launch kernel
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_q_blocks = 8
    num_b_blocks = 1
    grid_qk = lambda META: (
        triton.cdiv(num_blocks, num_b_blocks),
        triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_q_blocks),
    )
    qk_kernel[grid_qk](
        q_tile,
        k_tile,
        m_i_cur_tiles,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        head_tile,
        num_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        head_dim,
        sm_scale,
        num_q_blocks,
        num_b_blocks,
        q_tile.stride(0),
        q_tile.stride(1),
        q_tile.stride(2),
        k_tile.stride(0),
        k_tile.stride(1),
        k_tile.stride(2),
        m_i_cur_tiles.stride(0),
        m_i_cur_tiles.stride(1),
        m_i_cur_tiles.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=8,
        num_stages=3,
    )

    m_ij_tiles = m_i_cur_tiles.cummax(dim=1).values
    m_ij_last = m_ij_tiles[:, -1]

    return m_ij_tiles, m_ij_last


def qkv_kernel(
    q_tile,
    k_tile,
    v_tile,
    o_tiles,
    acc_o_scales,
    m_ij_tiles,
    l_ij,
    token_index_mapping,
    valid_topk_idx_permuted_tile,
    valid_lens,
    valid_start_indices,
    compute_min_block_id,
    cur_max_valid_tokens,
    head_tile,
    compute_tile_size,
    cu_seqlens_q,
    cu_seqlens_k,
    head_dim,
    sm_scale,
    block_size,
):
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = triton.next_power_of_2(block_size)
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)

    # a heuristic that avoids large grid size, and redudant KV loading
    num_q_blocks = 8

    grid_fwd = lambda META: (
        compute_tile_size * head_tile,
        triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_q_blocks),
    )

    forward_kernel_opt[grid_fwd](
        q_tile,
        k_tile,
        v_tile,
        o_tiles,
        acc_o_scales,
        m_ij_tiles,
        l_ij,
        token_index_mapping,
        valid_topk_idx_permuted_tile,
        valid_lens,
        valid_start_indices,
        compute_min_block_id,
        cur_max_valid_tokens,
        head_tile,
        compute_tile_size,
        cu_seqlens_q,
        cu_seqlens_k,
        head_dim,
        sm_scale,
        num_q_blocks,
        q_tile.stride(0),
        q_tile.stride(1),
        q_tile.stride(2),
        k_tile.stride(0),
        k_tile.stride(1),
        k_tile.stride(2),
        v_tile.stride(0),
        v_tile.stride(1),
        v_tile.stride(2),
        o_tiles.stride(0),
        o_tiles.stride(1),
        o_tiles.stride(2),
        o_tiles.stride(3),
        acc_o_scales.stride(0),
        acc_o_scales.stride(1),
        acc_o_scales.stride(2),
        m_ij_tiles.stride(0),
        m_ij_tiles.stride(1),
        m_ij_tiles.stride(2),
        l_ij.stride(0),
        l_ij.stride(1),
        l_ij.stride(2),
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_stages=3,
        num_warps=4,
    )


def reduce_output(
    lse,
    o,
    o_tiles_first,
    o_tiles_rest,
    m_ij_tiles,
    l_ij_first,
    l_ij_rest,
    m_ij_last,
    acc_o_scales_first,
    acc_o_scales_rest,
    topk_idx_tile,
    token_index_mapping,
    h,
    head_tile,
    total_len,
    TOPK,
    head_dim,
):

    num_qy_loop = 4
    num_qz_loop = total_len // num_qy_loop

    grid_reduce = lambda META: (
        num_qy_loop + (total_len % num_qy_loop != 0),
        num_qz_loop,
    )

    reduce_kernel[grid_reduce](
        lse,
        m_ij_tiles,
        l_ij_first,
        l_ij_rest,
        m_ij_last,
        o,
        o_tiles_first,
        o_tiles_rest,
        acc_o_scales_first,
        acc_o_scales_rest,
        topk_idx_tile,
        token_index_mapping,
        h * head_tile,
        num_qz_loop,
        TOPK,
        total_len,
        lse.stride(0),
        lse.stride(1),
        m_ij_tiles.stride(0),
        m_ij_tiles.stride(1),
        m_ij_tiles.stride(2),
        l_ij_first.stride(0),
        l_ij_first.stride(1),
        l_ij_first.stride(2),
        l_ij_rest.stride(0),
        l_ij_rest.stride(1),
        l_ij_rest.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o_tiles_first.stride(0),
        o_tiles_first.stride(1),
        o_tiles_first.stride(2),
        o_tiles_first.stride(3),
        o_tiles_rest.stride(0),
        o_tiles_rest.stride(1),
        o_tiles_rest.stride(2),
        o_tiles_rest.stride(3),
        acc_o_scales_first.stride(0),
        acc_o_scales_first.stride(1),
        acc_o_scales_first.stride(2),
        acc_o_scales_rest.stride(0),
        acc_o_scales_rest.stride(1),
        acc_o_scales_rest.stride(2),
        topk_idx_tile.stride(0),
        topk_idx_tile.stride(1),
        topk_idx_tile.stride(2),
        token_index_mapping.stride(0),
        token_index_mapping.stride(1),
        token_index_mapping.stride(2),
        BLOCK_SIZE_T=triton.next_power_of_2(TOPK),
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        num_warps=1,
        num_stages=2,
    )


def _topk_sparse_attention_fwd_opt_per_seq(
    q: torch.Tensor,  # [total_len, num_heads, head_dim]
    k: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    v: torch.Tensor,  # [total_len, num_kv_heads, head_dim]
    topk_idx: torch.Tensor,  # [num_heads, total_len, topk]
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    causal=True,
):
    # dtype check
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert block_size in {32, 64, 128, 256}
    # shape

    total_len, num_heads, head_dim = q.shape
    total_len, num_kv_heads, head_dim = k.shape

    assert num_heads % num_kv_heads == 0
    gqa_deg = num_heads // num_kv_heads

    TOPK = topk_idx.shape[-1]

    real_num_blocks = math.ceil(total_len / block_size)
    num_blocks = max(real_num_blocks, TOPK)

    head_tile = 1
    reduce_tile_size = num_blocks - 1

    valid_lens_all = torch.zeros(
        (
            num_kv_heads,
            num_blocks,
        ),
        dtype=torch.int32,
        device=q.device,
    )
    for h in range(num_kv_heads):
        topk_idx_tile = topk_idx[h * head_tile: (h + 1) * head_tile]
        topk_idx_nonneg = topk_idx_tile[topk_idx_tile >= 0]
        valid_lens = torch.bincount(topk_idx_nonneg.view(-1), minlength=num_blocks)
        valid_lens_all[h * head_tile: (h + 1) * head_tile] = valid_lens

    global_max_valid_tokens = valid_lens_all[:, 1:].max() if num_blocks > 1 else valid_lens_all.max()

    o_full = torch.zeros_like(q)
    lse_full = torch.full((num_heads, total_len), float("-inf"), dtype=torch.float32, device=q.device)

    # New introduced buffers
    topk_idx_permuted_tile = torch.full((head_tile, num_blocks, total_len), -1, dtype=torch.int32, device=q.device)

    token_index_mapping = torch.full((head_tile, num_blocks, total_len), 0, dtype=torch.int32, device=q.device)
    # first KV block is computed seaprately
    o_tiles_first = torch.zeros((head_tile, 1, total_len, head_dim), dtype=torch.bfloat16, device=q.device)
    o_tiles_rest = torch.zeros(
        (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=q.device
    )

    # Statistics buffers
    # m_i_tiles: 历史最大, m_diff_tiles: 历史最大和当前最大的差值
    # m_i_cur_tiles: 当前最大, # m_ij_tiles: 考虑当前和历史后的最大
    m_i_cur_tiles: torch.Tensor = torch.full(
        (head_tile, num_blocks, total_len), float("-inf"), dtype=torch.float32, device=q.device
    )

    # first KV block is reduced separately
    l_ij_first = torch.full((head_tile, 1, total_len), 0, dtype=torch.float32, device=q.device)
    acc_o_scales_first = torch.full((head_tile, 1, total_len), 1, dtype=torch.float32, device=q.device)

    l_ij_rest = torch.full(
        (head_tile, reduce_tile_size, global_max_valid_tokens), 0, dtype=torch.float32, device=q.device
    )
    acc_o_scales_rest = torch.full(
        (head_tile, reduce_tile_size, global_max_valid_tokens), 1, dtype=torch.float32, device=q.device
    )

    permute_results = {}
    permute_results['global_max_valid_tokens'] = global_max_valid_tokens
    permute_results['num_blocks'] = num_blocks
    permute_results['real_num_blocks'] = real_num_blocks
    permute_results['valid_topk_idx_permuted_tile'] = []
    permute_results['valid_lens_all'] = valid_lens_all
    permute_results['valid_lens'] = []
    permute_results['valid_start_indices'] = []

    for h in range(num_heads // head_tile):
        q_tile = q[:, h * head_tile: (h + 1) * head_tile]
        k_tile = k[:, (h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]
        v_tile = v[:, (h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]
        o = o_full[:, h * head_tile: (h + 1) * head_tile]
        lse = lse_full[h * head_tile: (h + 1) * head_tile]

        permute_min_block_id = 0
        permute_max_block_id = min(permute_min_block_id + num_blocks, num_blocks)

        topk_idx_tile = topk_idx[(h // gqa_deg) * head_tile: ((h // gqa_deg + 1)) * head_tile]

        if h % gqa_deg == 0:
            topk_idx_permuted_tile = build_block_to_token_triton(
                topk_idx_permuted_tile, topk_idx_tile, permute_min_block_id, permute_max_block_id, padding_value=-1
            )

            valid_topk_idx_permuted_tile = topk_idx_permuted_tile[topk_idx_permuted_tile != -1]
            valid_lens = valid_lens_all[(h // gqa_deg) * head_tile, :]
            valid_start_indices = torch.nn.functional.pad(valid_lens.cumsum(0)[:-1], (1, 0), value=0)

            index_mapping(
                token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks
            )

            permute_results['valid_topk_idx_permuted_tile'].append(valid_topk_idx_permuted_tile)
            permute_results['valid_lens'].append(valid_lens)
            permute_results['valid_start_indices'].append(valid_start_indices)

            m_ij_tiles, m_ij_last = online_softmax(
                q_tile,
                k_tile,
                m_i_cur_tiles,
                valid_topk_idx_permuted_tile,
                valid_lens,
                valid_start_indices,
                0,
                total_len,
                block_size,
                num_blocks,
                head_tile,
                head_dim,
                sm_scale,
                cu_seqlens_q,
                cu_seqlens_k,
            )

            m_ij_tiles[:, :, :] = m_ij_tiles[:, :, 0][:, :, None]
            m_ij_last[:, :] = m_ij_last[:, 0]
        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                cur_max_valid_tokens = total_len
                cur_valid_lens = valid_lens[0]
                cur_valid_start_indices = valid_start_indices[0]
                o_tiles = o_tiles_first
                l_ij = l_ij_first
                acc_o_scales = acc_o_scales_first
                compute_tile_size = 1
            else:
                cur_max_valid_tokens = valid_lens[compute_min_block_id:].max()
                cur_valid_lens = valid_lens[compute_min_block_id:]
                cur_valid_start_indices = valid_start_indices[compute_min_block_id:]
                o_tiles = o_tiles_rest
                l_ij = l_ij_rest
                acc_o_scales = acc_o_scales_rest
                compute_tile_size = num_blocks - 1

            # launch kernel
            qkv_kernel(
                q_tile,
                k_tile,
                v_tile,
                o_tiles,
                acc_o_scales,
                m_ij_tiles,
                l_ij,
                token_index_mapping,
                valid_topk_idx_permuted_tile,
                cur_valid_lens,
                cur_valid_start_indices,
                compute_min_block_id,
                cur_max_valid_tokens,
                head_tile,
                compute_tile_size,
                cu_seqlens_q,
                cu_seqlens_k,
                head_dim,
                sm_scale,
                block_size,
            )

        reduce_output(
            lse,
            o,
            o_tiles_first,
            o_tiles_rest,
            m_ij_tiles,
            l_ij_first,
            l_ij_rest,
            m_ij_last,
            acc_o_scales_first,
            acc_o_scales_rest,
            topk_idx_tile,
            token_index_mapping,
            h,
            head_tile,
            total_len,
            TOPK,
            head_dim,
        )

        o_full[:, h * head_tile: (h + 1) * head_tile] = o
        lse_full[h * head_tile: (h + 1) * head_tile] = lse

        if h % gqa_deg == 0:
            fused_fill(topk_idx_permuted_tile, m_i_cur_tiles)

    return o_full, lse_full, permute_results


@triton.jit
def dq_compute_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    lse_ptr,
    delta_ptr,
    do_ptr,
    dq_tiles_ptr,
    token_index_mapping_ptr,
    selected_tokens_ptr,
    valid_lens_ptr,
    valid_start_indices_ptr,
    cur_max_valid_tokens,
    compute_min_block_id,
    head_tile,
    num_blocks,
    HEAD_DIM,
    cu_seqlens_k,
    num_dq_blocks,
    sm_scale,
    debug_ptr,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    stride_dqth,
    stride_dqtb,
    stride_dqtn,
    stride_dqtd,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):

    pid_block = tl.program_id(0)
    pid_q = tl.program_id(1)  # token
    # seq packing is not supported yet
    q_start = 0
    k_start = 0

    k_len = tl.load(cu_seqlens_k + 1) - k_start

    start_id = tl.load(valid_start_indices_ptr + pid_block)
    valid_tokens = tl.load(valid_lens_ptr + pid_block)
    if num_dq_blocks * pid_q * BLOCK_SIZE_Q >= valid_tokens:
        return

    c = (pid_block + compute_min_block_id) * BLOCK_SIZE_K
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # load k
    k = tl.load(tl.advance(k_ptrs, (c, 0)), boundary_check=(1, 0), padding_option="zero")
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn,
        shape=(HEAD_DIM, k_len),
        strides=(stride_vd, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )

    # load v
    v = tl.load(tl.advance(v_ptrs, (0, c)), boundary_check=(0, 1), padding_option="zero")

    qk_scale = sm_scale * 1.44269504

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    for j in range(num_dq_blocks):
        pid_q_j = pid_q * num_dq_blocks + j
        if pid_q_j * BLOCK_SIZE_Q < valid_tokens:
            # one thread block for one KV block, a subset of selected tokens
            st_offs = start_id + (q_start + pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            # st should be in shape [BLOCK_SIZE_Q]
            st_mask = (pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < valid_tokens

            st = tl.load(selected_tokens_ptr + st_offs, mask=st_mask, other=-1)
            tl.store(debug_ptr + tl.arange(0, BLOCK_SIZE_Q), st_offs)
            # otherwise, st selects a set of q tokens, selected_tokens_ptr should be sorted
            q_ptrs_off = st[:, None] * stride_qn + off_d[None, :] * stride_qd

            mask = st != -1

            q_ptrs = q_ptr + q_start * stride_qn + q_ptrs_off
            # load q
            q_mask = mask[:, None] & (off_d < HEAD_DIM)[None, :]
            q = tl.load(q_ptrs, mask=q_mask, other=0)
            do_ptrs = do_ptr + q_start * stride_qn + q_ptrs_off
            do = tl.load(do_ptrs, mask=q_mask, other=0)
            delta_ptrs = delta_ptr + st[:, None]
            d = tl.load(delta_ptrs, mask=mask[:, None], other=0)
            lse_ptrs = lse_ptr + st[:, None]
            lse = tl.load(lse_ptrs, mask=mask[:, None], other=0)

            dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((st[:, None] >= c + off_k[None, :]), 0, float("-inf"))
            qk += tl.dot(q, tl.trans(k)) * qk_scale  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            p = tl.exp2(qk - lse)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            dp = tl.dot(do, v)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            ds = sm_scale * p * (dp - d)  # [BLOCK_SIZE_Q, BLOCK_SIZE_K]
            ds = ds.to(q.dtype)
            dq = tl.dot(ds, k)  # [BLOCK_SIZE_Q, BLOCK_SIZE_D]

            # load token index mapping
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + (st) * stride_tim_n + (pid_block + compute_min_block_id) * stride_tim_b
            )
            token_index_mapping = tl.load(token_index_mapping_ptrs, mask=mask, other=-1)

            dq_ptrs_off = token_index_mapping[:, None] * stride_dqtn + off_d[None, :] * stride_dqtd
            dq_tiles_ptrs = dq_tiles_ptr + dq_ptrs_off + (pid_block).to(tl.int64) * stride_dqtb
            tl.store(dq_tiles_ptrs, dq.to(dq_tiles_ptr.dtype.element_ty), mask=q_mask)


@triton.jit
def dq_reduce_kernel(
    dq_buffer_first_ptr,  # [H, 1, N, D]
    dq_buffer_rest_ptr,  # [H, B, N, D]
    dq_ptr,  # o: n x h x d
    t_ptr,  # topk_idx: h x n x k
    token_index_mapping_ptr,
    num_qz_loop,
    TOPK,
    total_len,
    # stride
    stride_dqtfh,
    stride_dqtfb,
    stride_dqtfn,
    stride_dqtfd,
    stride_dqtrh,
    stride_dqtrb,
    stride_dqtrn,
    stride_dqtrd,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_tim_h,
    stride_tim_b,
    stride_tim_n,
    # META parameters
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_qy = tl.program_id(0)
    pid_q = tl.program_id(1)  # token

    pid_q_j = pid_q + pid_qy * num_qz_loop
    if pid_q_j >= total_len:
        return
    t_ptr_j = t_ptr + pid_q_j * stride_tn

    off_d = tl.arange(0, BLOCK_SIZE_D)
    dq_ptrs = dq_ptr + pid_q_j * stride_dqn + off_d
    acc_dq = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

    for block_id in range(TOPK):
        t = tl.load(t_ptr_j + block_id * stride_tk, mask=block_id < TOPK, other=-1)
        if t != -1:
            if t == 0:
                dq_buffer_ptr = dq_buffer_first_ptr
                stride_dqtb = stride_dqtfb
                stride_dqtn = stride_dqtfn
                real_block_pos = 0
            else:
                dq_buffer_ptr = dq_buffer_rest_ptr
                stride_dqtb = stride_dqtrb
                stride_dqtn = stride_dqtrn
                real_block_pos = t - 1

            # init pointers
            token_index_mapping_ptrs = (
                token_index_mapping_ptr + t.to(tl.int64) * stride_tim_b + (pid_q_j) * stride_tim_n
            )
            real_token_index = tl.load(token_index_mapping_ptrs)

            dq_buffer_ptrs = (
                dq_buffer_ptr + real_block_pos.to(tl.int64) * stride_dqtb + (real_token_index) * stride_dqtn + off_d
            )

            dq_buffers = tl.load(dq_buffer_ptrs)
            acc_dq = dq_buffers + acc_dq

    tl.store(dq_ptrs, acc_dq, mask=off_d < BLOCK_SIZE_D)


def backward_dq_opt(
    q,  # [total_len, num_heads, head_dim]
    k,  # [total_len, num_k_heads, head_dim]
    v,  # [total_len, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    lse,  # [num_heads, total_len]
    delta,  # [num_heads, total_len]
    do,  # [total_len, num_heads, head_dim]
    dq,  # [total_len, num_heads, head_dim]
    cu_seqlens_q,
    cu_seqlens_k,
    num_k_heads,
    num_share_q_heads,
    head_dim,
    topk,
    sm_scale,
    block_size,
    permute_results,
):
    """
        TODO: Currently sequence packing is explicitly done in for loop, will merge in kernels.
    """
    for i in range(len(cu_seqlens_q) - 1):
        cu_seqlens_q_ = cu_seqlens_q[i: i + 2] - cu_seqlens_q[i]
        cu_seqlens_k_ = cu_seqlens_k[i: i + 2] - cu_seqlens_k[i]

        permute_results_ = permute_results[i]

        q_ = q[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        k_ = k[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        v_ = v[cu_seqlens_k[i]: cu_seqlens_k[i + 1]]
        topk_idx_ = topk_idx[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        lse_ = lse[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        delta_ = delta[:, cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        do_ = do[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]
        dq_ = dq[cu_seqlens_q[i]: cu_seqlens_q[i + 1]]

        backward_dq_opt_per_seq(
            q_,
            k_,
            v_,
            topk_idx_,
            lse_,
            delta_,
            do_,
            dq_,
            cu_seqlens_q_,
            cu_seqlens_k_,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            topk,
            sm_scale,
            block_size,
            permute_results_,
        )

        dq[cu_seqlens_q[i]: cu_seqlens_q[i + 1]] = dq_

    return dq


def backward_dq_opt_per_seq(
    q,  # [total_len, num_k_heads, head_dim]
    k,  # [total_len, num_k_heads, head_dim]
    v,  # [total_len, num_k_heads, head_dim]
    topk_idx,  # [num_k_heads, total_len, topk]
    lse,  # [num_k_heads, total_len]
    delta,  # [num_k_heads, total_len]
    do,  # [total_len, num_k_heads, head_dim]
    dq,  # [total_len, num_k_heads, head_dim]
    cu_seqlens_q,
    cu_seqlens_k,
    num_k_heads,
    num_share_q_heads,
    head_dim,
    topk,
    sm_scale,
    block_size,
    permute_results,
):
    head_tile = 1
    total_len = topk_idx.shape[1]
    global_max_valid_tokens = permute_results['global_max_valid_tokens']
    num_blocks = permute_results['num_blocks']
    reduce_tile_size = num_blocks - 1
    dq_buffer_first = torch.zeros((head_tile, 1, total_len, head_dim), dtype=torch.bfloat16, device=dq.device)
    dq_buffer_rest = torch.zeros(
        (head_tile, reduce_tile_size, global_max_valid_tokens, head_dim), dtype=torch.bfloat16, device=dq.device
    )

    num_heads = num_share_q_heads * num_k_heads

    token_index_mapping = torch.full((head_tile, num_blocks, total_len), 0, dtype=torch.int32, device=q.device)
    for h in range(num_heads // head_tile):
        valid_topk_idx_permuted_tile = permute_results['valid_topk_idx_permuted_tile'][h // num_share_q_heads]

        valid_lens = permute_results['valid_lens'][h // num_share_q_heads]
        valid_start_indices = permute_results['valid_start_indices'][h // num_share_q_heads]

        index_mapping(token_index_mapping, valid_topk_idx_permuted_tile, valid_lens, valid_start_indices, num_blocks)
        q_tile = q[:, h * head_tile: (h + 1) * head_tile]
        k_tile = k[:, (h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        v_tile = v[:, (h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        do_tile = do[:, h * head_tile: (h + 1) * head_tile]
        lse_tile = lse[h * head_tile: (h + 1) * head_tile]
        topk_idx_tile = topk_idx[(h // num_share_q_heads) * head_tile: ((h // num_share_q_heads + 1)) * head_tile]
        delta_tile = delta[h * head_tile: (h + 1) * head_tile]
        dq_tile = dq[:, h * head_tile: (h + 1) * head_tile]

        for compute_min_block_id in range(min(2, num_blocks)):
            if compute_min_block_id == 0:
                compute_tile_size = 1
                cur_max_valid_tokens = total_len
                cur_valid_lens = valid_lens[0]
                cur_valid_start_indices = valid_start_indices[0]
                dq_buffer = dq_buffer_first
            else:
                compute_tile_size = num_blocks - 1
                cur_max_valid_tokens = valid_lens[compute_min_block_id:].max()
                cur_valid_lens = valid_lens[compute_min_block_id:]
                cur_valid_start_indices = valid_start_indices[compute_min_block_id:]
                dq_buffer = dq_buffer_rest

            BLOCK_SIZE_Q = 128
            num_dq_blocks = 8
            grid_dq = lambda META: (
                compute_tile_size,
                triton.cdiv(cur_max_valid_tokens, BLOCK_SIZE_Q * num_dq_blocks),
            )

            num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)
            BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
            BLOCK_SIZE_K = triton.next_power_of_2(block_size)
            debug = torch.zeros((BLOCK_SIZE_Q,), dtype=torch.int32, device=dq.device)
            dq_compute_kernel[grid_dq](
                q_tile,
                k_tile,
                v_tile,
                lse_tile,
                delta_tile,
                do_tile,
                dq_buffer,
                token_index_mapping,
                valid_topk_idx_permuted_tile,
                cur_valid_lens,
                cur_valid_start_indices,
                cur_max_valid_tokens,
                compute_min_block_id,
                head_tile,
                num_blocks,
                head_dim,
                cu_seqlens_k,
                num_dq_blocks,
                sm_scale,
                debug,
                q_tile.stride(0),
                q_tile.stride(1),
                q_tile.stride(2),
                k_tile.stride(0),
                k_tile.stride(1),
                k_tile.stride(2),
                v_tile.stride(0),
                v_tile.stride(1),
                v_tile.stride(2),
                token_index_mapping.stride(0),
                token_index_mapping.stride(1),
                token_index_mapping.stride(2),
                dq_buffer.stride(0),
                dq_buffer.stride(1),
                dq_buffer.stride(2),
                dq_buffer.stride(3),
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        num_qy_loop = 4
        num_qz_loop = total_len // num_qy_loop

        grid_reduce = lambda META: (
            num_qy_loop + (total_len % num_qy_loop != 0),
            num_qz_loop,
        )
        dq_reduce_kernel[grid_reduce](
            dq_buffer_first,
            dq_buffer_rest,
            dq_tile,
            topk_idx_tile,
            token_index_mapping,
            num_qz_loop,
            topk,
            total_len,
            dq_buffer_first.stride(0),
            dq_buffer_first.stride(1),
            dq_buffer_first.stride(2),
            dq_buffer_first.stride(3),
            dq_buffer_rest.stride(0),
            dq_buffer_rest.stride(1),
            dq_buffer_rest.stride(2),
            dq_buffer_rest.stride(3),
            dq_tile.stride(0),
            dq_tile.stride(1),
            dq_tile.stride(2),
            topk_idx_tile.stride(0),
            topk_idx_tile.stride(1),
            topk_idx_tile.stride(2),
            token_index_mapping.stride(0),
            token_index_mapping.stride(1),
            token_index_mapping.stride(2),
            BLOCK_SIZE_T=triton.next_power_of_2(topk),
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=1,
            num_stages=2,
        )

        dq[:, h * head_tile: (h + 1) * head_tile] = dq_tile

    return dq


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

    topk_q_count = torch.cat(
        [
            permute_results[i]['valid_lens_all'][:, : permute_results[i]['real_num_blocks']]
            for i in range(len(permute_results))
        ],
        dim=1,
    )

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
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)

    backward_dq_opt(
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
        sm_scale,
        block_size,
        permute_results,
    )

    return dq, dk, dv


class FSATopkSparseAttention(torch.autograd.Function):
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

        o, lse, permute_results = _topk_sparse_attention_fwd_opt(
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


def FSA_topk_sparse_attention(
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
    return FSATopkSparseAttention.apply(
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
