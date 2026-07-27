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

BWD: Two modes controlled by bwd_mode (= inner loop direction):
     (a) "loopk" (default): inner loop over K. Each CTA owns one Q position
         and iterates over its topk KV positions.
         dQ: local register accumulation, no write conflicts.
         dK/dV: bf16 atomic scatter (multiple Q CTAs write the same KV).
         Implemented as separate dQ kernel + head-chunked dKV kernel.
     (b) "loopq": inner loop over Q. Each CTA owns one KV tile and iterates
         over all Q positions that reference it (via inverted index).
         dK/dV: local register accumulation, no write conflicts, no atomics.
         dQ: computed by a separate LoopK-direction dQ kernel.
         kbs=1: CSR + 64-bit bitmask; kbs=128: block-level inverse index.

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
def _bwd_dq_kernel(
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
    """BWD dQ kernel (LoopK direction): inner loop over K positions.
    Each CTA owns one Q position and loops over its topk KV tiles.
    dQ accumulated locally in fp32, no write conflicts."""
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
def _bwd_loopk_dkv_kernel(
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
    """BWD dKV kernel (LoopK direction): inner loop over K positions.
    Each CTA loops over Q positions that reference a given KV tile.
    Heads are chunked in groups of BLOCK_MH to reduce register pressure.
    dK/dV accumulated in fp32 across head groups, then bf16 atomic scatter."""
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
def _bwd_loopq_dkv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    Lse,
    Delta,
    InnerIndices,
    InnerMasks,
    InnerOffsets,
    InnerCounts,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_ot,
    stride_oh,
    stride_dkt,
    stride_inv_slot,
    stride_inv_topk,
    TOTAL_KV,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_CSR: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    """BWD dKV kernel (LoopQ direction): inner loop over Q positions.
    Each CTA owns one KV tile exclusively and iterates over all Q positions
    that reference it (via inverted index). dK/dV accumulated locally in fp32,
    no atomics needed. kbs=1 uses CSR + 64-bit bitmask for token-level masking;
    kbs=128 uses block-level inverse index without masks.
    """
    pid_slot = tl.program_id(0).to(tl.int64)
    kv_start = pid_slot * BLOCK_KV

    offs_m = tl.arange(0, BLOCK_M)
    offs_kv = tl.arange(0, BLOCK_KV)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < NHQ
    kv_mask = kv_start + offs_kv < TOTAL_KV

    kv_ptrs = (kv_start + offs_kv[:, None].to(tl.int64)) * stride_kt + offs_d[None, :]
    k_tile = tl.load(K + kv_ptrs, mask=kv_mask[:, None], other=0.0)
    v_tile = tl.load(V + kv_ptrs, mask=kv_mask[:, None], other=0.0)

    dk_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_KV, D], dtype=tl.float32)

    if USE_CSR:
        ref_start = tl.load(InnerOffsets + pid_slot).to(tl.int64)
        ref_end = tl.load(InnerOffsets + pid_slot + 1).to(tl.int64)
        num_q_refs = ref_end - ref_start
    else:
        ref_start = pid_slot * stride_inv_slot
        num_q_refs = tl.load(InnerCounts + pid_slot).to(tl.int32)

    for qi in tl.range(0, num_q_refs):
        if USE_CSR:
            ref_idx = ref_start + qi
            qp = tl.load(InnerIndices + ref_idx).to(tl.int64)
        else:
            ref_idx = ref_start + qi * stride_inv_topk
            qp = tl.load(InnerIndices + ref_idx).to(tl.int64)

        if USE_MASK:
            mask_bits = tl.load(InnerMasks + ref_idx)
            n_mask = (((mask_bits >> offs_kv.to(tl.int64)) & 1) == 1) & kv_mask
        else:
            n_mask = kv_mask

        q_ptrs = Q + qp * stride_qt + offs_m[:, None] * stride_qh + offs_d[None, :]
        q_tile = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        do_ptrs = dO + qp * stride_ot + offs_m[:, None] * stride_oh + offs_d[None, :]
        do_tile = tl.load(do_ptrs, mask=m_mask[:, None], other=0.0)

        lse = tl.load(Lse + qp * NHQ + offs_m, mask=m_mask, other=0.0)
        delta = tl.load(Delta + qp * NHQ + offs_m, mask=m_mask, other=0.0)

        s = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
        s = tl.where(m_mask[:, None] & n_mask[None, :], s, -float("inf"))
        p = tl.exp(s - lse[:, None])

        dp = tl.dot(do_tile, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * sm_scale).to(q_tile.dtype)

        dk_acc += tl.dot(tl.trans(ds), q_tile).to(tl.float32)
        dv_acc += tl.dot(tl.trans(p.to(do_tile.dtype)), do_tile).to(tl.float32)

    dk_out = (
        dK + (kv_start + offs_kv[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    dv_out = (
        dV + (kv_start + offs_kv[:, None].to(tl.int64)) * stride_dkt + offs_d[None, :]
    )
    tl.store(dk_out, dk_acc.to(dK.dtype.element_ty), mask=kv_mask[:, None])
    tl.store(dv_out, dv_acc.to(dV.dtype.element_ty), mask=kv_mask[:, None])


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
    q, k, v, indices, o, do, lse, bwd_mode="loopk", sparse_k_block_size=1
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
        bwd_mode: "loopk" | "loopq" — controls BWD inner loop direction.
            "loopk": dQ local, dKV via bf16 atomic scatter.
            "loopq": dKV local (no atomics), needs inverted index.
        sparse_k_block_size: int (default 1). For loopq: 1=token-level, 128=block-level.

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

    if bwd_mode == "loopq":
        # Standard-grid LoopQ: one CTA exclusively owns each physical KV tile.
        # kbs=1 uses CSR inverse metadata plus a 64-bit token mask. The mask
        # preserves sparse semantics but masked lanes still consume tl.dot work.
        # kbs=128 uses block inverse metadata and runtime InnerCounts.
        kbs = sparse_k_block_size
        assert kbs in (1, 128), "loopq_token supports sparse_k_block_size 1 or 128"
        BLOCK_KV = 64 if kbs == 1 else 128
        if kbs == 128:
            assert (
                total_kv % BLOCK_KV == 0
            ), f"total_kv ({total_kv}) must be divisible by {BLOCK_KV} for kbs=128"

        # dQ remains standard LoopK for both LoopQ metadata layouts.
        BLOCK_N_DQ = 64
        dq = torch.empty_like(q)
        _bwd_dq_kernel[(total_q,)](
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

        num_kv_slots = (total_kv + BLOCK_KV - 1) // BLOCK_KV
        if kbs == 1:
            inner_indices, inner_masks, inner_offsets = _build_block_inverse_indices(
                indices, total_kv, BLOCK_KV
            )
            inner_counts = inner_offsets
            stride_inv_slot = 0
            stride_inv_topk = 1
            use_csr = True
            use_mask = True
        else:
            from magi_attention.utils.sparse_utils import invert_index_sparse_indices

            block_indices = indices.unsqueeze(1) // BLOCK_KV
            padded_indices, _ = invert_index_sparse_indices(
                block_indices,
                seqlen_k=total_kv,
                sparse_k_block_size=BLOCK_KV,
                pad_multiple=64,
            )
            inner_indices = padded_indices.contiguous()
            inner_counts = (
                (inner_indices[:, 0, :] >= 0).sum(dim=-1).to(torch.int32).contiguous()
            )
            # Compile-time-disabled pointer arguments share the same kernel ABI.
            inner_masks = inner_counts
            inner_offsets = inner_counts
            stride_inv_slot = inner_indices.stride(0)
            stride_inv_topk = inner_indices.stride(2)
            use_csr = False
            use_mask = False

        dk = torch.empty(total_kv, D, device=q.device, dtype=torch.float32)
        dv = torch.empty(total_kv, D, device=q.device, dtype=torch.float32)
        _bwd_loopq_dkv_kernel[(num_kv_slots,)](
            q,
            k_flat,
            v_flat,
            do,
            dk,
            dv,
            lse,
            delta,
            inner_indices,
            inner_masks,
            inner_offsets,
            inner_counts,
            sm_scale,
            q.stride(0),
            q.stride(1),
            k_flat.stride(0),
            do.stride(0),
            do.stride(1),
            dk.stride(0),
            stride_inv_slot,
            stride_inv_topk,
            total_kv,
            NHQ=nhq,
            D=D,
            BLOCK_KV=BLOCK_KV,
            BLOCK_M=BLOCK_M,
            USE_CSR=use_csr,
            USE_MASK=use_mask,
            num_warps=8,
        )

        dk = dk.unsqueeze(1).to(q.dtype)
        dv = dv.unsqueeze(1).to(q.dtype)
        return dq, dk, dv

    if bwd_mode == "loopk":
        # Split: separate dQ (fp32, no atomics) + head-chunked dKV (bf16 atomics).
        # Head chunking (BLOCK_MH=32) reduces register pressure from 576+ to ~240,
        # eliminating spilling. BLOCK_N=64 halves atomic write batches vs BN=32.
        BLOCK_N_DQ = 64
        BLOCK_MH = 32
        BLOCK_N_DKV = 64

        dq = torch.empty_like(q)
        _bwd_dq_kernel[(total_q,)](
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
        _bwd_loopk_dkv_kernel[(total_q,)](
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

    raise ValueError(f"Unknown bwd_mode: {bwd_mode!r}. Use 'split' or 'loopq_token'.")


# ═══════════════════════════════════════════════════════════════════════════════
# Autograd wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TokenSparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, bwd_mode, sparse_k_block_size):
        o, lse = token_sparse_fwd(q, k, v, indices, return_lse=True)
        ctx.save_for_backward(q, k, v, indices, o, lse)
        ctx.bwd_mode = bwd_mode
        ctx.sparse_k_block_size = sparse_k_block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, indices, o, lse = ctx.saved_tensors
        dq, dk, dv = token_sparse_bwd(
            q,
            k,
            v,
            indices,
            o,
            do.contiguous(),
            lse,
            bwd_mode=ctx.bwd_mode,
            sparse_k_block_size=ctx.sparse_k_block_size,
        )
        return dq, dk, dv, None, None, None


def token_sparse_attn(q, k, v, indices, bwd_mode="loopk", sparse_k_block_size=1):
    """Token-sparse attention with autograd support.

    Args:
        q: (total_q, nhq, D) — requires_grad
        k: (total_kv, 1, D) — requires_grad
        v: (total_kv, 1, D) — requires_grad
        indices: (total_q, topk) int32
        bwd_mode: "loopk" | "loopq" — BWD inner loop direction.
        sparse_k_block_size: sparsity granularity for LoopQ inverse index (1 or 128).

    Returns:
        o: (total_q, nhq, D)
    """
    return TokenSparseAttnFunc.apply(q, k, v, indices, bwd_mode, sparse_k_block_size)
