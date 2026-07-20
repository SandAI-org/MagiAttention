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
Token-sparse attention Triton kernel (index-based) — MQA-optimized.

Exploits MQA structure (nhk=1): all nhq query heads at the same position
share the same topk KV indices. This allows batching all heads as the M
dimension of a GEMM, using tl.dot for Tensor Core acceleration.

FWD: Grid (total_q_positions,) — 1 thread block processes all nhq heads.
     Q[BLOCK_M, D] @ gathered_K[BLOCK_N, D]^T → S[BLOCK_M, BLOCK_N]  (tl.dot)
     P[BLOCK_M, BLOCK_N] @ gathered_V[BLOCK_N, D] → O[BLOCK_M, D]    (tl.dot)

BWD: Five dKV strategies:
     (a) "split" (default): separate dQ kernel (fp32 accum, no atomics) +
         head-chunked dKV kernel (BLOCK_MH=32, BLOCK_N=64, bf16 atomics).
         Head chunking reduces register pressure from 576+ to ~240, eliminating
         spilling. Larger BLOCK_N=64 halves atomic write batches. ~98-105 TFLOPS.
     (b) "fused": single kernel computes dQ + dKV together.
         Computes S/P once, loads K/V once, BLOCK_N=32. ~75 TFLOPS.
     (c) "atomic": separate dQ + dKV kernels, fp32 atomic scatter. ~43 TFLOPS.
     (d) "loopq": LoopQ with block-level inverse index + bitmask (kbs=64).
     (e) "loopq_dense": LoopQ with kbs=128 dense S[NHQ,128], no atomics, no mask.
         dKV accumulated in fp32 registers via tl.dot. Requires kbs-aligned indices.

Input:
    q: (total_q, nhq, D)
    k: (total_kv, 1, D)
    v: (total_kv, 1, D)
    indices: (total_q, topk) — per-query-position topk kv token indices (int32)
"""

import math

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════════════
# FWD Kernel
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _token_sparse_fwd_kernel(
    Q,
    K,
    V,
    Indices,
    Out,
    Lse,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_it,
    stride_ot,
    stride_oh,
    NHQ: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_q = tl.program_id(0).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < NHQ

    # Load Q tile: (BLOCK_M, D) — all nhq heads for this position
    q_ptrs = Q + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    idx_base = Indices + pid_q * stride_it

    num_chunks = tl.cdiv(TOPK, BLOCK_N)
    for chunk_id in range(num_chunks):
        start = chunk_id * BLOCK_N
        chunk_offs = start + offs_n
        n_mask = chunk_offs < TOPK

        # Load indices: (BLOCK_N,)
        kv_idx = tl.load(idx_base + chunk_offs, mask=n_mask, other=0)

        # Gather K: (BLOCK_N, D)
        k_ptrs = K + kv_idx[:, None] * stride_kt + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # S = Q @ K^T: (BLOCK_M, BLOCK_N)
        s = tl.dot(q_tile, tl.trans(k_tile))
        s = s * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Gather V: (BLOCK_N, D)
        v_ptrs = V + kv_idx[:, None] * stride_kt + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # O += P @ V: (BLOCK_M, D)
        acc += tl.dot(p.to(v_tile.dtype), v_tile)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output: (BLOCK_M, D)
    out_ptrs = Out + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None])

    # Store LSE if requested
    if Lse is not None:
        lse_val = m_i + tl.log(l_i)
        lse_ptrs = Lse + pid_q * NHQ + offs_m
        tl.store(lse_ptrs, lse_val, mask=m_mask)


def token_sparse_fwd(q, k, v, indices, return_lse=False):
    """Token-sparse attention forward (MQA-optimized with tl.dot).

    Args:
        q: (total_q, nhq, D)
        k: (total_kv, 1, D)
        v: (total_kv, 1, D)
        indices: (total_q, topk) int32, per-position KV token indices

    Returns:
        o: (total_q, nhq, D)
        lse: (total_q, nhq) if return_lse else None
    """
    total_q, nhq, D = q.shape
    topk = indices.shape[-1]
    sm_scale = 1.0 / math.sqrt(D)

    assert D in (64, 128, 256), f"D={D} not supported, need power-of-2 in [64,256]"
    assert k.shape[1] == 1, "MQA: k must have shape (total_kv, 1, D)"

    o = torch.empty_like(q)
    lse = (
        torch.empty(total_q, nhq, device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )

    BLOCK_M = triton.next_power_of_2(nhq)
    BLOCK_N = 64

    grid = (total_q,)
    _token_sparse_fwd_kernel[grid](
        q,
        k.squeeze(1),
        v.squeeze(1),
        indices,
        o,
        lse,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        indices.stride(0),
        o.stride(0),
        o.stride(1),
        NHQ=nhq,
        TOPK=topk,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    if return_lse:
        return o, lse
    return o


# ═══════════════════════════════════════════════════════════════════════════════
# BWD Kernels
# ═══════════════════════════════════════════════════════════════════════════════


@triton.jit
def _token_sparse_bwd_dq_kernel(
    Q,
    K,
    V,
    Indices,
    dO,
    dQ,
    Lse,
    Delta,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_it,
    stride_ot,
    stride_oh,
    NHQ: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dQ for one query position (all heads)."""
    pid_q = tl.program_id(0).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < NHQ

    # Load Q, dO: (BLOCK_M, D)
    q_ptrs = Q + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    do_ptrs = dO + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

    # Load LSE and Delta: (BLOCK_M,)
    lse_ptrs = Lse + pid_q * NHQ + offs_m
    lse = tl.load(lse_ptrs, mask=m_mask, other=0.0)
    delta_ptrs = Delta + pid_q * NHQ + offs_m
    delta = tl.load(delta_ptrs, mask=m_mask, other=0.0)

    # Accumulate dQ
    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    idx_base = Indices + pid_q * stride_it

    num_chunks = tl.cdiv(TOPK, BLOCK_N)
    for chunk_id in range(num_chunks):
        start = chunk_id * BLOCK_N
        chunk_offs = start + offs_n
        n_mask = chunk_offs < TOPK

        kv_idx = tl.load(idx_base + chunk_offs, mask=n_mask, other=0)

        # Gather K: (BLOCK_N, D)
        k_ptrs = K + kv_idx[:, None] * stride_kt + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Recompute S = Q @ K^T
        s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))

        # P = exp(S - LSE)
        p = tl.exp(s - lse[:, None])

        # Gather V: (BLOCK_N, D)
        v_ptrs = V + kv_idx[:, None] * stride_kt + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # dP = dO @ V^T: (BLOCK_M, BLOCK_N)
        dp = tl.dot(do_tile, tl.trans(v_tile))

        # dS = P * (dP - Delta)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dQ += dS @ K: (BLOCK_M, D)
        dq_acc += tl.dot(ds.to(k_tile.dtype), k_tile)

    # Store dQ
    dq_ptrs = dQ + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _fused_dq_dkv_kernel(
    Q,
    K,
    V,
    Indices,
    dO,
    dQ,
    dK,
    dV,
    Lse,
    Delta,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_it,
    stride_ot,
    stride_oh,
    stride_dkt,
    NHQ: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused dQ + dKV kernel: computes S/P once (saves 2 redundant tl.dot vs separate
    dQ + dKV kernels), and loads K/V only once.
    dQ: accumulated locally (no atomics). dK/dV: bf16 atomic scatter.
    BLOCK_N=32 is optimal — smaller S/P tiles reduce register spilling, which
    dominates over loop-count overhead."""
    pid_q = tl.program_id(0).to(tl.int64)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < NHQ

    q_ptrs = Q + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
    do_ptrs = dO + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

    lse = tl.load(Lse + pid_q * NHQ + offs_m, mask=m_mask, other=0.0)
    delta = tl.load(Delta + pid_q * NHQ + offs_m, mask=m_mask, other=0.0)

    dq_acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    idx_base = Indices + pid_q * stride_it

    num_chunks = tl.cdiv(TOPK, BLOCK_N)
    for chunk_id in range(num_chunks):
        start = chunk_id * BLOCK_N
        chunk_offs = start + offs_n
        n_mask = chunk_offs < TOPK
        kv_idx = tl.load(idx_base + chunk_offs, mask=n_mask, other=0)

        k_ptrs = K + kv_idx[:, None] * stride_kt + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v_ptrs = V + kv_idx[:, None] * stride_kt + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))
        p = tl.exp(s - lse[:, None])

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

        dq_acc += tl.dot(ds, k_tile).to(tl.float32)

        dk_chunk = tl.dot(tl.trans(ds), q_tile)
        dv_chunk = tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)
        dk_ptrs = dK + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        dv_ptrs = dV + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk_chunk.to(tl.bfloat16), mask=n_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_chunk.to(tl.bfloat16), mask=n_mask[:, None])

    dq_ptrs = dQ + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def _dkv_headchunked_kernel(
    Q,
    K,
    V,
    Indices,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_it,
    stride_ot,
    stride_oh,
    stride_dkt,
    NHQ: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_MH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dKV with head chunking: processes heads in groups of BLOCK_MH to reduce
    register pressure from 576+ (full BLOCK_M=128) to ~240.
    dK/dV accumulated across head groups in fp32, then bf16 atomic scatter.
    Optimal: BLOCK_MH=32, BLOCK_N=64 — balances register budget vs loop count."""
    pid_q = tl.program_id(0).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    num_chunks: tl.constexpr = (TOPK + BLOCK_N - 1) // BLOCK_N
    num_hgroups: tl.constexpr = (NHQ + BLOCK_MH - 1) // BLOCK_MH

    idx_base = Indices + pid_q * stride_it

    for chunk_id in range(num_chunks):
        start = chunk_id * BLOCK_N
        chunk_offs = start + offs_n
        n_mask = chunk_offs < TOPK
        kv_idx = tl.load(idx_base + chunk_offs, mask=n_mask, other=0)

        k_tile = tl.load(
            K + kv_idx[:, None] * stride_kt + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        v_tile = tl.load(
            V + kv_idx[:, None] * stride_kt + offs_d[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )

        dk_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dv_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        for hg in range(num_hgroups):
            h_start = hg * BLOCK_MH
            offs_m = h_start + tl.arange(0, BLOCK_MH)
            m_mask = offs_m < NHQ

            q_hg = tl.load(
                Q + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            do_hg = tl.load(
                dO + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :],
                mask=m_mask[:, None],
                other=0.0,
            )
            lse_hg = tl.load(Lse + pid_q * NHQ + offs_m, mask=m_mask, other=0.0)
            delta_hg = tl.load(Delta + pid_q * NHQ + offs_m, mask=m_mask, other=0.0)

            s = tl.dot(q_hg, tl.trans(k_tile)) * sm_scale
            s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))
            p = tl.exp(s - lse_hg[:, None])

            dp = tl.dot(do_hg, tl.trans(v_tile))
            ds = (p * (dp - delta_hg[:, None]) * sm_scale).to(q_hg.dtype)

            dk_acc += tl.dot(tl.trans(ds), q_hg)
            dv_acc += tl.dot(tl.trans(p.to(do_hg.dtype)), do_hg)

        dk_ptrs = dK + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        dv_ptrs = dV + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        tl.atomic_add(dk_ptrs, dk_acc.to(tl.bfloat16), mask=n_mask[:, None])
        tl.atomic_add(dv_ptrs, dv_acc.to(tl.bfloat16), mask=n_mask[:, None])


@triton.jit
def _token_sparse_bwd_dkv_loopq_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    BlockInvQ,
    BlockInvMask,
    BlockOffsets,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_ot,
    stride_oh,
    stride_dkt,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    """dKV kernel — LoopQ with BLOCK_N KV tiling + PackGQA (tl.dot enabled).

    Grid: (num_kv_blocks, num_splits).
    Each TB owns BLOCK_N consecutive KV positions, iterates over Q refs from
    the block-level inverse index.

    Tile structure (enables ALL operations via tl.dot / Tensor Core):
      M direction: 128 q-heads of one Q position (PackGQA fills BLOCK_M=128)
      N direction: BLOCK_N consecutive KV positions

      S = Q @ K^T:   (BLOCK_M, D) @ (D, BLOCK_N) → (BLOCK_M, BLOCK_N)  tl.dot ✓
      dP = dO @ V^T: (BLOCK_M, D) @ (D, BLOCK_N) → (BLOCK_M, BLOCK_N)  tl.dot ✓
      dK = dS^T @ Q: (BLOCK_N, BLOCK_M) @ (BLOCK_M, D) → (BLOCK_N, D)  tl.dot ✓
      dV = P^T @ dO:  (BLOCK_N, BLOCK_M) @ (BLOCK_M, D) → (BLOCK_N, D)  tl.dot ✓

    Sparse mask: per Q ref, a BLOCK_N-bit mask encodes which KVs in the block
    are actually referenced. Non-referenced KVs get -inf in score → P=0 → zero gradient.
    """
    pid_block = tl.program_id(0).to(tl.int64)
    pid_split = tl.program_id(1)

    block_start = pid_block * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < NHQ

    # Load K_tile, V_tile: (BLOCK_N, D) — stays in SRAM for all Q iterations
    kv_base = (block_start + offs_n[:, None].to(tl.int64)) * stride_kt + offs_d[None, :]
    k_tile = tl.load(K + kv_base)
    v_tile = tl.load(V + kv_base)

    # dK, dV accumulators: (BLOCK_N, D) in float32
    dk_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    # Block-level inverse index range
    blk_start = tl.load(BlockOffsets + pid_block).to(tl.int64)
    blk_end = tl.load(BlockOffsets + pid_block + 1).to(tl.int64)
    num_q = blk_end - blk_start

    my_start = pid_split * SPLIT_SIZE

    for qi in tl.range(0, SPLIT_SIZE):
        entry_idx = my_start + qi
        if entry_idx < num_q:
            qp = tl.load(BlockInvQ + blk_start + entry_idx).to(tl.int64)
            mask_bits = tl.load(BlockInvMask + blk_start + entry_idx)

            # Decode BLOCK_N-bit mask → (BLOCK_N,) boolean
            n_mask = ((mask_bits >> offs_n.to(tl.int64)) & 1) == 1

            # Load Q[qp]: (BLOCK_M, D) — M = 128 heads packed (PackGQA)
            q_ptrs = Q + qp * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
            q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            # Load dO[qp]: (BLOCK_M, D)
            do_ptrs = (
                dO + qp * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
            )
            do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

            # LSE, Delta: (BLOCK_M,)
            lse = tl.load(Lse + qp * NHQ + offs_m, mask=m_mask, other=0.0)
            delta = tl.load(Delta + qp * NHQ + offs_m, mask=m_mask, other=0.0)

            # S = Q @ K^T: (BLOCK_M, BLOCK_N) — tl.dot!
            s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            # Apply sparse mask: only referenced KVs get valid scores
            s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))

            # P = exp(S - LSE): uses saved LSE from full topk FWD softmax
            p = tl.exp(s - lse[:, None])

            # dP = dO @ V^T: (BLOCK_M, BLOCK_N) — tl.dot!
            dp = tl.dot(do_tile, tl.trans(v_tile))

            # dS = P * (dP - Delta) * sm_scale
            ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

            # dK += dS^T @ Q: (BLOCK_N, BLOCK_M) @ (BLOCK_M, D) → (BLOCK_N, D) tl.dot!
            dk_acc += tl.dot(tl.trans(ds), q_tile).to(tl.float32)
            # dV += P^T @ dO: (BLOCK_N, BLOCK_M) @ (BLOCK_M, D) → (BLOCK_N, D) tl.dot!
            dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile).to(tl.float32)

    # Atomic write (only num_splits atomics per KV block, not per-Q-position)
    dk_out = (
        dK + (block_start + offs_n[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    dv_out = (
        dV + (block_start + offs_n[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    tl.atomic_add(dk_out, dk_acc.to(dK.dtype.element_ty))
    tl.atomic_add(dv_out, dv_acc.to(dV.dtype.element_ty))


@triton.jit
def _preprocess_bwd_kernel(
    Out,
    dO,
    Delta,
    stride_ot,
    stride_oh,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Compute Delta = rowsum(O * dO) for each (position, head)."""
    pid_q = tl.program_id(0).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < NHQ

    o_ptrs = Out + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
    do_ptrs = dO + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]

    o_tile = tl.load(o_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o_tile * do_tile, axis=1)

    delta_ptrs = Delta + pid_q * NHQ + offs_m
    tl.store(delta_ptrs, delta, mask=m_mask)


@triton.jit
def _token_sparse_bwd_dkv_atomic_kernel(
    Q,
    K,
    V,
    Indices,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_it,
    stride_ot,
    stride_oh,
    stride_dkt,
    TOTAL_Q: tl.constexpr,
    NHQ: tl.constexpr,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dKV kernel — LoopK direction with atomic scatter (FA-2 standard pattern).

    Grid: (total_q,). Each TB owns one Q position, iterates over its topk KV positions.
    Uses tl.dot for score/dK/dV computation (tiles BLOCK_N KV positions → matmul).
    Scatter dK/dV via atomic_add — topk atomics per KV position.

    This is FASTER for MQA (nhk=1) because the inner-loop loads the LIGHT tensor (K/V=256B)
    while keeping the HEAVY tensor (Q=32KB+dO=32KB) in registers.
    """
    pid_q = tl.program_id(0).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < NHQ

    q_ptrs = Q + pid_q * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    do_ptrs = dO + pid_q * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
    do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

    lse_ptrs = Lse + pid_q * NHQ + offs_m
    lse = tl.load(lse_ptrs, mask=m_mask, other=0.0)
    delta_ptrs = Delta + pid_q * NHQ + offs_m
    delta = tl.load(delta_ptrs, mask=m_mask, other=0.0)

    idx_base = Indices + pid_q * stride_it

    num_chunks = tl.cdiv(TOPK, BLOCK_N)
    for chunk_id in range(num_chunks):
        start = chunk_id * BLOCK_N
        chunk_offs = start + offs_n
        n_mask = chunk_offs < TOPK

        kv_idx = tl.load(idx_base + chunk_offs, mask=n_mask, other=0)

        k_ptrs = K + kv_idx[:, None] * stride_kt + offs_d[None, :]
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))
        p = tl.exp(s - lse[:, None])

        v_ptrs = V + kv_idx[:, None] * stride_kt + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

        dk_chunk = tl.dot(tl.trans(ds), q_tile)
        dv_chunk = tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile)

        dk_scatter_ptrs = dK + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        dv_scatter_ptrs = dV + kv_idx[:, None] * stride_dkt + offs_d[None, :]
        tl.atomic_add(dk_scatter_ptrs, dk_chunk, mask=n_mask[:, None])
        tl.atomic_add(dv_scatter_ptrs, dv_chunk, mask=n_mask[:, None])


@triton.jit
def _loopq_inverted_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    InnerIndices,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_ot,
    stride_oh,
    stride_dkt,
    stride_inv_slot,
    stride_inv_topk,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    KBS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    INNER_TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Unified LoopQ dKV kernel — token-level (kbs=1) and block-level (kbs=N).

    Grid: (num_kv_slots,)
    Each TB owns KBS consecutive KV positions, iterates over Q refs from
    the inverted index (gathered by token index). S[BLOCK_M, BLOCK_KV] is
    fully dense for valid positions — no mask waste.

    BLOCK_KV = max(KBS, 16) to satisfy tl.dot min dimension.
    """
    pid_slot = tl.program_id(0).to(tl.int64)
    kv_start = pid_slot * KBS

    offs_m = tl.arange(0, BLOCK_M)
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, D)

    m_mask = offs_m < NHQ
    kv_mask = offs_kv < KBS

    # Load K_tile, V_tile: [BLOCK_KV, D]
    kv_ptrs = (kv_start + offs_kv[:, None].to(tl.int64)) * stride_kt + offs_d[None, :]
    k_tile = tl.load(K + kv_ptrs, mask=kv_mask[:, None], other=0.0)
    v_tile = tl.load(V + kv_ptrs, mask=kv_mask[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    # InnerIndices: (num_kv_slots, nhk=1, inner_topk) — squeeze nhk dim via strides
    inv_base = InnerIndices + pid_slot * stride_inv_slot

    for qi in tl.range(0, INNER_TOPK):
        qp = tl.load(inv_base + qi * stride_inv_topk)
        if qp >= 0:
            qp_i64 = qp.to(tl.int64)

            q_ptrs = (
                Q + qp_i64 * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
            )
            q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            do_ptrs = (
                dO + qp_i64 * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
            )
            do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

            lse = tl.load(Lse + qp_i64 * NHQ + offs_m, mask=m_mask, other=0.0)
            delta = tl.load(Delta + qp_i64 * NHQ + offs_m, mask=m_mask, other=0.0)

            # S = Q @ K^T: [BLOCK_M, BLOCK_KV]
            s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            s = tl.where(m_mask[:, None] & kv_mask[None, :], s, -float("inf"))

            p = tl.exp(s - lse[:, None])

            # dP = dO @ V^T
            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

            # dK += dS^T @ Q, dV += P^T @ dO
            dk_acc += tl.dot(tl.trans(ds), q_tile).to(tl.float32)
            dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile).to(tl.float32)

    # Store — no atomics
    dk_out = (
        dK + (kv_start + offs_kv[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    dv_out = (
        dV + (kv_start + offs_kv[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    tl.store(dk_out, dk_acc.to(dK.dtype.element_ty), mask=kv_mask[:, None])
    tl.store(dv_out, dv_acc.to(dV.dtype.element_ty), mask=kv_mask[:, None])


def _build_dense_block_inverse(indices, total_kv, kbs):
    """Build dense block-level inverse index for LoopQ with kbs >= BLOCK_N.

    Converts token-level Q→KV indices into block-level KV_block→Q mapping.
    Each Q that references ANY token in KV block b is included in b's list.
    Unlike the bitmask approach, S[NHQ, kbs] is FULLY DENSE (all kbs KV
    positions are valid for every Q ref), so no mask waste.

    Args:
        indices: (total_q, topk) int32 — forward Q→KV token indices
        total_kv: int
        kbs: int — K block size (e.g. 128)

    Returns:
        inv_q: (num_kv_blocks, inner_topk) int32 — Q positions per KV block.
            Padded with -1.
        inner_topk: int — max Q refs across all blocks (padded to 64).
    """
    total_q, topk = indices.shape
    device = indices.device
    num_blocks = total_kv // kbs

    block_ids = (indices // kbs).long()
    flat_q = (
        torch.arange(total_q, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(total_q, topk)
        .reshape(-1)
    )
    flat_block = block_ids.reshape(-1)

    # Deduplicate (block, q) pairs
    combo = flat_block * total_q + flat_q.long()
    combo_unique = combo.unique()
    unique_block = (combo_unique // total_q).int()
    unique_q = (combo_unique % total_q).int()

    counts = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    counts.scatter_add_(
        0,
        unique_block.long(),
        torch.ones(len(unique_block), device=device, dtype=torch.int32),
    )
    max_refs = int(counts.max().item())
    inner_topk = ((max_refs + 63) // 64) * 64  # pad to 64

    sorted_order = unique_block.long().argsort(stable=True)
    sorted_q = unique_q[sorted_order]
    sorted_block = unique_block[sorted_order].long()

    group_starts = torch.zeros(num_blocks + 1, device=device, dtype=torch.int64)
    group_starts[1:] = counts.long().cumsum(0)

    offsets = torch.arange(len(sorted_q), device=device, dtype=torch.int64)
    offsets = offsets - group_starts[sorted_block]

    inv_q = torch.full((num_blocks, inner_topk), -1, device=device, dtype=torch.int32)
    inv_q[sorted_block, offsets.long()] = sorted_q
    return inv_q, inner_topk


@triton.jit
def _loopq_dense_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    InvQ,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_ot,
    stride_oh,
    stride_dkt,
    stride_inv,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    KBS: tl.constexpr,
    INNER_TOPK: tl.constexpr,
):
    """LoopQ dKV kernel with DENSE S[NHQ, KBS] — no mask waste.

    Grid: (num_kv_blocks,).
    Each TB owns one KV block of KBS consecutive positions.
    Iterates over Q refs from the block-level inverse index (gathered by index).
    S[NHQ, KBS] is fully dense because every Q ref references the entire block.

    All operations use tl.dot (Tensor Core):
      S  = Q @ K^T:   [NHQ, D] @ [D, KBS]  → [NHQ, KBS]
      dP = dO @ V^T:  [NHQ, D] @ [D, KBS]  → [NHQ, KBS]
      dK = dS^T @ Q:  [KBS, NHQ] @ [NHQ, D] → [KBS, D]
      dV = P^T @ dO:  [KBS, NHQ] @ [NHQ, D] → [KBS, D]
    """
    pid_block = tl.program_id(0).to(tl.int64)
    block_start = pid_block * KBS

    offs_m = tl.arange(0, NHQ)
    offs_n = tl.arange(0, KBS)
    offs_d = tl.arange(0, D)

    # Load K_tile, V_tile: [KBS, D] — stays in registers for all Q iterations
    kv_base = (block_start + offs_n[:, None].to(tl.int64)) * stride_kt + offs_d[None, :]
    k_tile = tl.load(K + kv_base)
    v_tile = tl.load(V + kv_base)

    dk_acc = tl.zeros([KBS, D], dtype=tl.float32)
    dv_acc = tl.zeros([KBS, D], dtype=tl.float32)

    inv_base = InvQ + pid_block * stride_inv

    for qi in tl.range(0, INNER_TOPK):
        qp = tl.load(inv_base + qi)
        if qp >= 0:
            qp_i64 = qp.to(tl.int64)

            q_ptrs = (
                Q + qp_i64 * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
            )
            q_tile = tl.load(q_ptrs)

            do_ptrs = (
                dO + qp_i64 * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
            )
            do_tile = tl.load(do_ptrs)

            lse = tl.load(Lse + qp_i64 * NHQ + offs_m)
            delta = tl.load(Delta + qp_i64 * NHQ + offs_m)

            s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            p = tl.exp(s - lse[:, None])

            dp = tl.dot(do_tile, tl.trans(v_tile))
            ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

            dk_acc += tl.dot(tl.trans(ds), q_tile).to(tl.float32)
            dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile).to(tl.float32)

    # Direct store (no atomics — each TB owns its KV block exclusively)
    dk_out = (
        dK + (block_start + offs_n[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    dv_out = (
        dV + (block_start + offs_n[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    tl.store(dk_out, dk_acc.to(dK.dtype.element_ty))
    tl.store(dv_out, dv_acc.to(dV.dtype.element_ty))


def _build_block_inverse_indices(indices, total_kv, BLOCK_N):
    """Build block-level inverse indices for LoopQ dKV with BLOCK_N KV tiling.

    For each KV block of BLOCK_N consecutive positions, produces:
    - List of Q positions referencing ANY KV in the block (deduplicated)
    - Per-entry BLOCK_N-bit mask indicating which specific KVs are referenced

    Args:
        indices: (total_q, topk) int32 — forward Q→KV mapping
        total_kv: int
        BLOCK_N: int — KV block size (must be <= 64 for int64 mask)

    Returns:
        block_inv_q: (num_entries,) int32 — Q positions sorted by block
        block_inv_mask: (num_entries,) int64 — BLOCK_N-bit sparse mask per entry
        block_offsets: (num_blocks + 1,) int32 — CSR offsets per KV block
    """
    assert BLOCK_N <= 64, "BLOCK_N must be <= 64 for int64 bitmask"
    total_q, topk = indices.shape
    device = indices.device
    num_blocks = (total_kv + BLOCK_N - 1) // BLOCK_N

    block_ids = (indices // BLOCK_N).int()
    local_ids = (indices % BLOCK_N).int()

    flat_q = (
        torch.arange(total_q, device=device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(total_q, topk)
        .reshape(-1)
    )
    flat_block = block_ids.reshape(-1).long()
    flat_local = local_ids.reshape(-1).long()

    # Sort by (block_id, q_pos) for grouping
    sort_key = flat_block * total_q + flat_q.long()
    sort_order = sort_key.argsort(stable=True)
    sorted_q = flat_q[sort_order]
    sorted_block = flat_block[sort_order]
    sorted_local = flat_local[sort_order]

    # Find unique (block, q) pairs and aggregate masks via scatter_add
    pair_key = sorted_block * total_q + sorted_q.long()
    unique_keys, inverse = torch.unique_consecutive(pair_key, return_inverse=True)

    num_entries = unique_keys.shape[0]
    block_inv_q = (unique_keys % total_q).int()
    block_inv_block = (unique_keys // total_q).int()

    # Build bitmask: for each unique (block, q), OR the local_id bits.
    # Dedup (block, q, local) triples — scatter_add_ does ADD not OR,
    # so duplicate local_ids within the same (block, q) pair corrupt the mask.
    triple_key = pair_key * BLOCK_N + sorted_local
    first_occ = torch.ones(len(triple_key), device=device, dtype=torch.bool)
    first_occ[1:] = triple_key[1:] != triple_key[:-1]

    block_inv_mask = torch.zeros(num_entries, device=device, dtype=torch.int64)
    bit_values = (
        torch.ones(sorted_local.shape[0], device=device, dtype=torch.int64)
        << sorted_local
    )
    block_inv_mask.scatter_add_(0, inverse[first_occ], bit_values[first_occ])

    # CSR offsets per block
    block_counts = torch.bincount(block_inv_block, minlength=num_blocks)
    block_offsets = torch.zeros(num_blocks + 1, device=device, dtype=torch.int32)
    torch.cumsum(block_counts, dim=0, out=block_offsets[1:])

    return block_inv_q, block_inv_mask, block_offsets


def token_sparse_bwd(
    q, k, v, indices, o, do, lse, dkv_mode="split", sparse_k_block_size=1
):
    """Token-sparse attention backward (MQA-optimized).

    Args:
        q: (total_q, nhq, D)
        k: (total_kv, 1, D)
        v: (total_kv, 1, D)
        indices: (total_q, topk) int32
        o: (total_q, nhq, D) — FWD output
        do: (total_q, nhq, D) — gradient of output
        lse: (total_q, nhq) — log-sum-exp from FWD
        dkv_mode: "split" | "fused" | "atomic" | "loopq" | "loopq_dense" | "loopq_token"
        sparse_k_block_size: int (default 1). For loopq_token: 1=token-level, >1=block-level.

    Returns:
        dq: (total_q, nhq, D)
        dk: (total_kv, 1, D)
        dv: (total_kv, 1, D)
    """
    total_q, nhq, D = q.shape
    total_kv = k.shape[0]
    topk = indices.shape[-1]
    sm_scale = 1.0 / math.sqrt(D)

    BLOCK_M = triton.next_power_of_2(nhq)

    # Preprocess: Delta = rowsum(O * dO)
    delta = torch.empty(total_q, nhq, device=q.device, dtype=torch.float32)
    _preprocess_bwd_kernel[(total_q,)](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        NHQ=nhq,
        D=D,
        BLOCK_M=BLOCK_M,
    )

    k_flat = k.squeeze(1)
    v_flat = v.squeeze(1)

    if dkv_mode == "loopq_token":
        # Unified LoopQ with block-grouped inverted index + bitmask.
        # Always groups by BLOCK_KV positions for efficient tl.dot tiles.
        # For kbs >= BLOCK_KV: S[NHQ, kbs] fully dense (no mask needed).
        # For kbs < BLOCK_KV: bitmask indicates valid KV positions per Q ref,
        #   so S[NHQ, BLOCK_KV] uses the full tile width (no padding waste).
        # dKV accumulated in fp32 registers — NO atomics.
        from magi_attention.utils.sparse_utils import invert_index_sparse_indices

        kbs = sparse_k_block_size
        BLOCK_KV = 64  # tile size for KV dim (matches tl.dot efficiency)

        # dQ via LoopK (same as split mode, always)
        BLOCK_N_DQ = 64
        dq = torch.empty_like(q)
        _token_sparse_bwd_dq_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dq,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N_DQ,
        )

        if kbs >= BLOCK_KV:
            # Block-aligned: S[NHQ, kbs] fully dense, use _loopq_inverted_dkv_kernel.
            indices_3d = indices.unsqueeze(1)  # (total_q, 1, topk)
            block_indices = indices_3d // kbs
            inner_indices, inner_topk = invert_index_sparse_indices(
                block_indices,
                seqlen_k=total_kv,
                sparse_k_block_size=kbs,
                pad_multiple=64,
            )
            num_kv_slots = total_kv // kbs

            dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)
            dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)
            inner_indices_contig = inner_indices.contiguous()
            _loopq_inverted_dkv_kernel[(num_kv_slots,)](
                q,
                k_flat,
                v_flat,
                do,
                dk,
                dv,
                lse,
                delta,
                inner_indices_contig,
                sm_scale,
                q.stride(0),
                q.stride(1),
                k_flat.stride(0),
                do.stride(0),
                do.stride(1),
                dk.stride(0),
                inner_indices_contig.stride(0),
                inner_indices_contig.stride(2),
                NHQ=nhq,
                D=D,
                KBS=kbs,
                BLOCK_KV=triton.next_power_of_2(kbs),
                INNER_TOPK=inner_topk,
                BLOCK_M=BLOCK_M,
            )
        else:
            # Token-level sparse (kbs < BLOCK_KV): group by BLOCK_KV positions,
            # use bitmask to mark valid KV positions per Q ref.
            # S[NHQ, BLOCK_KV] fully utilizes tl.dot; invalid positions get -inf.
            block_inv_q, block_inv_mask, block_offsets = _build_block_inverse_indices(
                indices, total_kv, BLOCK_KV
            )
            num_kv_blocks = (total_kv + BLOCK_KV - 1) // BLOCK_KV
            max_refs = int((block_offsets[1:] - block_offsets[:-1]).max().item())
            SPLIT_SIZE = ((max_refs + 63) // 64) * 64

            dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)
            dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)
            _token_sparse_bwd_dkv_loopq_kernel[(num_kv_blocks, 1)](
                q,
                k_flat,
                v_flat,
                do,
                dk,
                dv,
                lse,
                delta,
                block_inv_q,
                block_inv_mask,
                block_offsets,
                sm_scale,
                q.stride(0),
                q.stride(1),
                k_flat.stride(0),
                do.stride(0),
                do.stride(1),
                dk.stride(0),
                NHQ=nhq,
                D=D,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_KV,
                SPLIT_SIZE=SPLIT_SIZE,
            )

        dk = dk.unsqueeze(1).to(q.dtype)
        dv = dv.unsqueeze(1).to(q.dtype)
        return dq, dk, dv

    if dkv_mode == "loopq_dense":
        # LoopQ with dense S[NHQ, kbs]: outer KV blocks, inner Q refs via inverse index.
        # Requires kbs >= 16 so tl.dot can form S[NHQ, kbs] efficiently.
        # dKV accumulated in fp32 registers — NO atomics.
        # dQ uses LoopK (separate kernel, fp32, no atomics).
        kbs = 128  # block size for KV grouping
        assert (
            total_kv % kbs == 0
        ), f"total_kv ({total_kv}) must be divisible by kbs ({kbs})"

        inv_q, inner_topk = _build_dense_block_inverse(indices, total_kv, kbs)
        num_kv_blocks = total_kv // kbs

        # dQ via LoopK (same as split mode)
        BLOCK_N_DQ = 64
        dq = torch.empty_like(q)
        _token_sparse_bwd_dq_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dq,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N_DQ,
        )

        # dKV via LoopQ — dense, no atomics
        dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)
        dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)
        _loopq_dense_dkv_kernel[(num_kv_blocks,)](
            q,
            k_flat,
            v_flat,
            do,
            dk,
            dv,
            lse,
            delta,
            inv_q,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            inv_q.stride(0),
            NHQ=nhq,
            D=D,
            KBS=kbs,
            INNER_TOPK=inner_topk,
        )

        dk = dk.unsqueeze(1).to(q.dtype)
        dv = dv.unsqueeze(1).to(q.dtype)
        return dq, dk, dv

    if dkv_mode == "split":
        # Split: separate dQ (fp32, no atomics) + head-chunked dKV (bf16 atomics).
        # Head chunking (BLOCK_MH=32) reduces register pressure from 576+ to ~240,
        # eliminating spilling. BLOCK_N=64 halves atomic write batches vs BN=32.
        BLOCK_N_DQ = 64
        BLOCK_MH = 32
        BLOCK_N_DKV = 64

        dq = torch.empty_like(q)
        _token_sparse_bwd_dq_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dq,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N_DQ,
        )

        dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)
        dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)
        _dkv_headchunked_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dk,
            dv,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_MH=BLOCK_MH,
            BLOCK_N=BLOCK_N_DKV,
        )
        dk = dk.unsqueeze(1).to(q.dtype)
        dv = dv.unsqueeze(1).to(q.dtype)
        return dq, dk, dv

    if dkv_mode == "fused":
        # Fused dQ+dKV: single kernel computes S/P once (saves 2 redundant
        # tl.dot vs separate kernels) and loads K/V once. dK/dV use bf16
        # hardware atomics. BLOCK_N=32 minimizes register spilling.
        BLOCK_N = 32
        dq = torch.empty_like(q)
        dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)
        dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.bfloat16)

        _fused_dq_dkv_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dq,
            dk,
            dv,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        dk = dk.unsqueeze(1).to(q.dtype)
        dv = dv.unsqueeze(1).to(q.dtype)
        return dq, dk, dv

    BLOCK_N = 64

    # dQ kernel (LoopK direction: per Q position, iterate over topk K)
    dq = torch.empty_like(q)

    _token_sparse_bwd_dq_kernel[(total_q,)](
        q,
        k_flat,
        v_flat,
        indices,
        do,
        dq,
        lse,
        delta,
        sm_scale,
        q.stride(0),
        q.stride(1),
        k_flat.stride(0),
        indices.stride(0),
        do.stride(0),
        do.stride(1),
        NHQ=nhq,
        TOPK=topk,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    dk = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)
    dv = torch.zeros(total_kv, D, device=q.device, dtype=torch.float32)

    if dkv_mode == "loopq":
        LOOPQ_BLOCK_N = min(BLOCK_N, 64)
        block_inv_q, block_inv_mask, block_offsets = _build_block_inverse_indices(
            indices, total_kv, LOOPQ_BLOCK_N
        )
        num_kv_blocks = (total_kv + LOOPQ_BLOCK_N - 1) // LOOPQ_BLOCK_N
        max_entries = int((block_offsets[1:] - block_offsets[:-1]).max().item())
        SPLIT_SIZE = 64
        num_splits = (max_entries + SPLIT_SIZE - 1) // SPLIT_SIZE

        _token_sparse_bwd_dkv_loopq_kernel[(num_kv_blocks, num_splits)](
            q,
            k_flat,
            v_flat,
            do,
            dk,
            dv,
            lse,
            delta,
            block_inv_q,
            block_inv_mask,
            block_offsets,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            NHQ=nhq,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=LOOPQ_BLOCK_N,
            SPLIT_SIZE=SPLIT_SIZE,
        )
    else:
        _token_sparse_bwd_dkv_atomic_kernel[(total_q,)](
            q,
            k_flat,
            v_flat,
            indices,
            do,
            dk,
            dv,
            lse,
            delta,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            indices.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            TOTAL_Q=total_q,
            NHQ=nhq,
            TOPK=topk,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

    dk = dk.unsqueeze(1).to(q.dtype)
    dv = dv.unsqueeze(1).to(q.dtype)
    return dq, dk, dv


# ═══════════════════════════════════════════════════════════════════════════════
# Autograd wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TokenSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices):
        o, lse = token_sparse_fwd(q, k, v, indices, return_lse=True)
        ctx.save_for_backward(q, k, v, indices, o, lse)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, indices, o, lse = ctx.saved_tensors
        dq, dk, dv = token_sparse_bwd(q, k, v, indices, o, do.contiguous(), lse)
        return dq, dk, dv, None


def token_sparse_attn(q, k, v, indices):
    """Token-sparse attention with autograd support.

    Args:
        q: (total_q, nhq, D) — requires_grad
        k: (total_kv, 1, D) — requires_grad
        v: (total_kv, 1, D) — requires_grad
        indices: (total_q, topk) int32

    Returns:
        o: (total_q, nhq, D)
    """
    return TokenSparseAttnFunc.apply(q, k, v, indices)
