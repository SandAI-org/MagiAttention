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

BWD: Two-pass architecture with selectable dKV strategy:
     dQ kernel (LoopK): per query-position, inner-loop over topk KV positions.
         Grid=(total_q,). dQ accumulated locally, no atomics. Uses tl.dot.
     dKV kernel — two variants:
       (a) "atomic" (default): LoopK + atomic scatter.
           Grid=(total_q,). Uses tl.dot. Best for RANDOM indices.
           ~36 TFLOPS. Bottleneck: atomic contention.
       (b) "loopq": LoopQ with BLOCK_N KV tiling + PackGQA + block-level inner_indices.
           Grid=(num_kv_blocks, num_splits). ALL ops use tl.dot (Tensor Core).
           - Structured/local indices: ~150 TFLOPS (4x faster than atomic!)
           - Random indices: ~15 TFLOPS (sparse mask wastes N dimension)
           Tile: M=128 (PackGQA nhq heads), N=BLOCK_N (consecutive KV positions).
           Preprocess: block inverse index + per-(block,Q) bitmask.

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

    # Build bitmask: for each unique (block, q), OR the local_id bits
    block_inv_mask = torch.zeros(num_entries, device=device, dtype=torch.int64)
    bit_values = (
        torch.ones(sorted_local.shape[0], device=device, dtype=torch.int64)
        << sorted_local
    )
    block_inv_mask.scatter_add_(0, inverse, bit_values)

    # CSR offsets per block
    block_counts = torch.bincount(block_inv_block, minlength=num_blocks)
    block_offsets = torch.zeros(num_blocks + 1, device=device, dtype=torch.int32)
    torch.cumsum(block_counts, dim=0, out=block_offsets[1:])

    return block_inv_q, block_inv_mask, block_offsets


def token_sparse_bwd(q, k, v, indices, o, do, lse, dkv_mode="atomic"):
    """Token-sparse attention backward (MQA-optimized).

    dQ: LoopK direction (per Q, iterate topk K) — uses tl.dot, no atomics.
    dKV: selectable strategy via `dkv_mode`:
        - "atomic" (default): LoopK + atomic scatter. Uses tl.dot. Best for random indices.
        - "loopq": LoopQ with BLOCK_N KV tiling + PackGQA + bitmask.
          ALL operations use tl.dot. Best for structured/local indices (4x faster).
          With random indices: ~2.4x slower due to sparse mask inefficiency.

    Args:
        q: (total_q, nhq, D)
        k: (total_kv, 1, D)
        v: (total_kv, 1, D)
        indices: (total_q, topk) int32
        o: (total_q, nhq, D) — FWD output
        do: (total_q, nhq, D) — gradient of output
        lse: (total_q, nhq) — log-sum-exp from FWD
        dkv_mode: "atomic" | "loopq"

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
    BLOCK_N = 64

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

    # dQ kernel (LoopK direction: per Q position, iterate over topk K)
    dq = torch.empty_like(q)
    k_flat = k.squeeze(1)
    v_flat = v.squeeze(1)

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
        # LoopQ with BLOCK_N KV tiling: per KV-BLOCK, iterate Q refs with tl.dot.
        # Uses block-level inverse index + bitmask for sparse pattern.
        LOOPQ_BLOCK_N = min(BLOCK_N, 64)  # must fit in int64 bitmask
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
        # LoopK + atomic: per Q position, tiles BLOCK_N KVs for tl.dot.
        # Faster for MQA (inner-loop loads light K/V, heavy Q stays in regs).
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
