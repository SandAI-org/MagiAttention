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
Benchmark: Token-sparse attention (topk=2048) — FFA IndexAttn vs baselines.

MQA (nhq=128, nhk=1), head_dim=128, topk=2048 fixed, sweep seqlen.
Token-level sparsity: q_block_size=k_block_size=1 (NOT block-sparse).

Methods compared:
  - FFA IndexAttn (token-sparse, block_size=1)
  - FlexAttention (PyTorch flex_attention with sparse block mask, enable_gqa=True)
  - Triton Token-Sparse (per-token index-based, same interface as FFA)
  - Triton Block-Sparse (block_size=64)
  - EffectiveKernels (Kwai-Keye DSA topk_block_unique pipeline)
  - TileLang Sparse MLA (pipelined, examples/deepseek_v32)

FWD-only, reporting effective sparse TFLOPs/s (flops = 4*S*topk*nhq*hd).

NOTE on TFLOPS drop at large S:
  FFA IndexAttn peaks at S~16k then drops ~9% at S=102k. This is REAL
  (not a bench artifact): random gather from larger KV pool causes L2 cache
  thrashing when many thread blocks compete for cache lines simultaneously.
  See .tmp/038-index-attn-bench-analysis/analysis.md for details.
"""

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func
from baselines.block_sparse_attn_triton import block_sparse_fwd
from baselines.token_sparse_attn_triton import token_sparse_fwd
from baselines.utils import seed_everything
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

# ─── Optional dependencies ────────────────────────────────────────────────────

try:
    from effective_kernels.ops.topk_block_unique import topk_block_unique  # noqa: F401
    from effective_kernels.ops.sparse_attention import sparse_attention_forward

    HAS_EFFECTIVE_KERNELS = True
except ImportError:
    # Try loading from local build
    import sys as _sys

    _ek_path = "/tmp/EffectiveKernels"
    if _ek_path not in _sys.path:
        _sys.path.insert(0, _ek_path)
    try:
        from effective_kernels.ops.topk_block_unique import topk_block_unique
        from effective_kernels.ops.sparse_attention import sparse_attention_forward

        HAS_EFFECTIVE_KERNELS = True
    except ImportError:
        HAS_EFFECTIVE_KERNELS = False

HAS_TILELANG = False  # Lazy-imported on first use to avoid CUDA context conflicts

# ─── Helpers ──────────────────────────────────────────────────────────────────


def build_index_attn_indices(b, S, nhk, topk, device):
    """Build index_attn_indices: (b*S, nhk, topk) int32 for token-sparse FFA."""
    total_q = b * S
    local_pos = torch.randint(0, S, (total_q, topk), device=device).sort(dim=1).values
    batch_idx = torch.arange(total_q, device=device) // S
    global_pos = batch_idx.unsqueeze(1) * S + local_pos
    h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
    return (global_pos.unsqueeze(1) * nhk + h_offsets).int()


def build_triton_block_sparse_indices(S, topk, block_size=64, device="cuda"):
    """Build q2k_index and q2k_num for Triton block-sparse FWD kernel."""
    num_q_blocks = S // block_size
    num_kv_blocks = S // block_size
    kv_blocks_needed = min((topk + block_size - 1) // block_size, num_kv_blocks)
    perm = (
        torch.rand(num_q_blocks, num_kv_blocks, device=device)
        .argsort(dim=1)[:, :kv_blocks_needed]
        .sort(dim=1)
        .values
    )
    q2k_index = perm.unsqueeze(0).unsqueeze(0).int()
    q2k_num = torch.full((1, 1, num_q_blocks), kv_blocks_needed, device=device).int()
    return q2k_index, q2k_num


def build_flex_sparse_block_mask(b, S, topk, nhq, device="cuda"):
    """Build a flex_attention block_mask for token-sparse pattern.

    FlexAttention uses 128x128 blocks. For topk=2048, we select
    topk // 128 = 16 KV blocks per query block.
    """
    FLEX_BLOCK = 128
    num_q_blocks = S // FLEX_BLOCK
    num_kv_blocks = S // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected_kv_blocks = (
        torch.rand(num_q_blocks, num_kv_blocks, device=device)
        .argsort(dim=1)[:, :kv_blocks_needed]
    )
    mask_dense = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)
    mask_dense.scatter_(1, selected_kv_blocks, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_block = q_idx // FLEX_BLOCK
        kv_block = kv_idx // FLEX_BLOCK
        return mask_dense[q_block, kv_block]

    block_mask = create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )
    return block_mask


# ─── TileLang Sparse MLA kernel (adapted from examples/deepseek_v32) ─────────

_tilelang_sparse_mla_kernel = None


def _ensure_tilelang():
    global HAS_TILELANG
    if not HAS_TILELANG:
        try:
            import tilelang as _tl  # noqa: F401

            HAS_TILELANG = True
        except ImportError:
            pass
    return HAS_TILELANG


def get_tilelang_sparse_mla_kernel():
    """JIT-compile the tilelang sparse attention forward kernel (pipelined).

    Warp-specialized version with producer-consumer overlap and double buffering.
    Hardcoded for benchmark config: nhq=128, nhk=1, hd=128, topk=2048.

    Key optimizations (from deepseek_v32/sparse_mla_fwd_pipelined.py):
    - 3 warp groups: WG0 = consumer (QK + softmax + PV_left),
                     WG1 = consumer (PV_right), WG2 = producer (KV loads)
    - Double buffering for KV shared memory
    - WGMMA for QK attention score computation
    - cp.async for asynchronous KV data loading
    - Split-D: output dimension split into left/right halves
    - Barrier-based fine-grained producer-consumer synchronization
    """
    global _tilelang_sparse_mla_kernel
    if _tilelang_sparse_mla_kernel is not None:
        return _tilelang_sparse_mla_kernel

    import tilelang
    from tilelang import language as T  # noqa: F811

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def sparse_attn_fwd_pipelined(
        Q,
        KV,
        Indices,
        heads=128,
        dim=128,
        topk=2048,
        kv_group=1,
        sm_scale=0.12585414,
        block_I=64,
        num_stages=0,
        threads=384,
    ):
        batch, seq_len, seq_len_kv = T.dynamic("batch, seq_len, seq_len_kv")

        head_kv = heads // kv_group
        q_shape = [batch, seq_len, heads, dim]
        kv_shape = [batch, seq_len_kv, kv_group, dim]
        o_shape = [batch, seq_len, heads, dim]
        indices_shape = [batch, seq_len, kv_group, topk]
        lse_shape = [batch, seq_len, heads]
        indices_dtype = T.int32
        dtype = T.bfloat16
        accum_dtype = T.float32

        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        D = dim
        D_half = D // 2

        REPLICATE_H = head_kv // 64 if head_kv > 64 else 1
        H_per_block = padded_H if REPLICATE_H == 1 else 64

        Q: T.Tensor(q_shape, dtype)
        KV: T.Tensor(kv_shape, dtype)
        Indices: T.Tensor(indices_shape, indices_dtype)
        Output = T.empty(o_shape, dtype)
        Lse = T.empty(lse_shape, accum_dtype)

        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx, by, bz,
        ):
            Q_shared_l = T.alloc_shared([H_per_block, D_half], dtype)
            Q_shared_r = T.alloc_shared([H_per_block, D_half], dtype)
            KV_shared_0_l = T.alloc_shared([BI, D_half], dtype)
            KV_shared_0_r = T.alloc_shared([BI, D_half], dtype)
            KV_shared_1_l = T.alloc_shared([BI, D_half], dtype)
            KV_shared_1_r = T.alloc_shared([BI, D_half], dtype)
            O_shared_l = Q_shared_l
            O_shared_r = Q_shared_r

            acc_o_l = T.alloc_fragment([H_per_block, D_half], accum_dtype)
            acc_o_r = T.alloc_fragment([H_per_block, D_half], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
            alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_var(indices_dtype)

            bar_q = T.alloc_barrier(arrive_count=384)
            bar_k_0_ready = T.alloc_barrier(arrive_count=128)
            bar_k_1_ready = T.alloc_barrier(arrive_count=128)
            bar_k_0_free = T.alloc_barrier(arrive_count=256)
            bar_k_1_free = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
            bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)

            H0 = g_i * padded_H + (
                0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            )
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()

            T.tma_copy(Q[b_i, s_i, H0:H1, 0:D_half], Q_shared_l, barrier=bar_q)
            T.tma_copy(Q[b_i, s_i, H0:H1, D_half:D], Q_shared_r, barrier=bar_q)
            T.barrier_arrive(bar_q)

            if tx < 128:
                # WG0: consumer — QK GEMM + softmax + PV_left GEMM
                T.set_max_nreg(240, 1)
                T.fill(sumexp, 0)
                T.fill(m_i, -(2**30))
                T.fill(acc_o_l, 0)
                T.barrier_wait(bar_q, 0)

                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_ready[0], (i_i & 1))
                    T.fill(acc_s, 0)
                    T.wgmma_gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True)
                    T.wgmma_gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True)
                    T.wait_wgmma(0)

                    if i_i != 0:
                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(
                            bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1
                        )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                        )
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D_half):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_0_l, acc_o_l)
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_0_free[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_ready[0], (i_i & 1))
                    T.fill(acc_s, 0)
                    T.wgmma_gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True)
                    T.wgmma_gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True)
                    T.wait_wgmma(0)

                    T.barrier_arrive(bar_sScale_and_sS_free)
                    T.barrier_wait(
                        bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1
                    )

                    T.copy(m_i, m_i_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                    for h_i in T.Parallel(H_per_block):
                        alpha_local[h_i] = T.exp2(
                            (m_i_prev[h_i] - m_i[h_i]) * sm_scale
                        )
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D_half):
                        acc_o_l[h_i, d_i] *= alpha_local[h_i]
                    T.copy(alpha_local, alpha_shared)
                    T.copy(acc_s, S_shared)
                    T.gemm(S_shared, KV_shared_1_l, acc_o_l)
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_arrive(bar_k_1_free[0])

                for h_i in T.Parallel(H_per_block):
                    sum_exp_shared[h_i] = sumexp[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D_half):
                    acc_o_l[h_i, d_i] /= sumexp[h_i]
                T.copy(acc_o_l, O_shared_l)
                T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D_half])

            elif tx >= 128 and tx < 256:
                # WG1: consumer — PV_right GEMM
                T.set_max_nreg(168, 1)
                T.fill(acc_o_r, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D_half):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                    T.barrier_arrive(bar_k_0_free[0])
                    T.barrier_arrive(bar_sScale_and_sS_free)

                    # Buffer 1
                    T.barrier_arrive(bar_sScale_and_sS_ready)
                    T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                    for h_i, d_i in T.Parallel(H_per_block, D_half):
                        acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                    T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                    T.barrier_arrive(bar_k_1_free[0])
                    if i_i != T.ceildiv(NI, 2) - 1:
                        T.barrier_arrive(bar_sScale_and_sS_free)

                for h_i, d_i in T.Parallel(H_per_block, D_half):
                    acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]
                T.copy(acc_o_r, O_shared_r)
                T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D_half:D])

            elif tx >= 256:
                # WG2: producer — load KV via cp.async
                T.set_max_nreg(80, 0)
                for i_i in T.serial(T.ceildiv(NI, 2)):
                    # Buffer 0
                    T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local = Indices[
                            b_i, s_i, g_i,
                            (i_i * 2) * BI + r * 16 + (tx - 256) // 8,
                        ]
                        for u in T.serial(4):
                            T.ptx_cp_async(
                                T.access_ptr(
                                    KV_shared_0_l[
                                        r * 16 + (tx - 256) // 8,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "w", 8,
                                ),
                                T.access_ptr(
                                    KV[
                                        b_i, indices_local, g_i,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "r", 8,
                                ),
                                8,
                            )
                            T.ptx_cp_async(
                                T.access_ptr(
                                    KV_shared_0_r[
                                        r * 16 + (tx - 256) // 8,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "w", 8,
                                ),
                                T.access_ptr(
                                    KV[
                                        b_i, indices_local, g_i,
                                        D_half + 32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "r", 8,
                                ),
                                8,
                            )
                    T.cp_async_barrier_noinc(bar_k_0_ready[0])

                    # Buffer 1
                    T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                    for r in T.serial(4):
                        indices_local = Indices[
                            b_i, s_i, g_i,
                            (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8,
                        ]
                        for u in T.serial(4):
                            T.ptx_cp_async(
                                T.access_ptr(
                                    KV_shared_1_l[
                                        r * 16 + (tx - 256) // 8,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "w", 8,
                                ),
                                T.access_ptr(
                                    KV[
                                        b_i, indices_local, g_i,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "r", 8,
                                ),
                                8,
                            )
                            T.ptx_cp_async(
                                T.access_ptr(
                                    KV_shared_1_r[
                                        r * 16 + (tx - 256) // 8,
                                        32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "w", 8,
                                ),
                                T.access_ptr(
                                    KV[
                                        b_i, indices_local, g_i,
                                        D_half + 32 * u + (tx - 256) % 8 * 4,
                                    ],
                                    "r", 8,
                                ),
                                8,
                            )
                    T.cp_async_barrier_noinc(bar_k_1_ready[0])

        return Output, Lse

    _tilelang_sparse_mla_kernel = sparse_attn_fwd_pipelined
    return _tilelang_sparse_mla_kernel


# ─── Config ───────────────────────────────────────────────────────────────────

b = 1
nhq = 128
nhk = 1  # MQA
hd = 128
topk = 2048
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]

seqlen_vals = [4096, 8192, 16384, 32768, 65536, 102400]

METHODS = [
    "ffa_index_attn",
    "flexattention",
    "triton_token_sparse",
    "triton_block_sparse",
    "effective_kernels",
    "tilelang_sparse_mla",
]
METHOD_NAMES = [
    "FFA IndexAttn (token-sparse)",
    "FlexAttention (GQA, sparse mask)",
    "Triton Token-Sparse (indices)",
    "Triton Block-Sparse (bs=64)",
    "EffectiveKernels (DSA)",
    "TileLang Sparse MLA (pipelined)",
]
METHOD_STYLES = [
    ("red", "-"),
    ("green", "-."),
    ("blue", "--"),
    ("orange", ":"),
    ("purple", "-"),
    ("brown", "-."),
]

seed_everything()

attn_flops_configs = [
    Benchmark(
        x_names=["S"],
        x_vals=seqlen_vals,
        x_log=True,
        line_arg="method",
        line_vals=METHODS,
        line_names=METHOD_NAMES,
        styles=METHOD_STYLES,
        ylabel={"flops": "Effective Sparse TFLOPs/s"},
        plot_name=(
            f"Token-Sparse Attention (topk={topk}) — MQA (PackGQA)\n"
            f"nhq={nhq}, nhk={nhk}, D={hd}, FWD"
        ),
        args={},
    ),
]


@perf_report(attn_flops_configs)
def comparison_benchmark(S, method):
    device = torch.cuda.current_device()
    sparse_flops = 4 * S * topk * nhq * hd

    try:
        if method == "ffa_index_attn":
            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)

            index_attn_indices = build_index_attn_indices(b, S, nhk, topk, device)
            q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
            k_t = rearrange(k, "b s h d -> (b s h) 1 d")
            v_t = rearrange(v, "b s h d -> (b s h) 1 d")
            del q, k, v
            torch.cuda.empty_cache()

            def fn():
                return ffa_func(
                    q_t, k_t, v_t,
                    index_attn_indices=index_attn_indices,
                    q_block_size=1,
                    k_block_size=1,
                    pack_gqa=True,
                )

        elif method == "flexattention":
            assert S % 128 == 0, "FlexAttention requires S divisible by 128"
            q = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            k = torch.randn(b, nhk, S, hd, device=device, dtype=dtype)
            v = torch.randn(b, nhk, S, hd, device=device, dtype=dtype)

            block_mask = build_flex_sparse_block_mask(b, S, topk, nhq, device)
            _flex_fn = torch.compile(flex_attention)

            def fn():
                return _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

        elif method == "triton_token_sparse":
            q = torch.randn(b * S, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b * S, 1, hd, device=device, dtype=dtype)
            v = torch.randn(b * S, 1, hd, device=device, dtype=dtype)
            tri_indices = torch.randint(
                0, b * S, (b * S, topk), device=device, dtype=torch.int32
            ).sort(dim=1).values

            def fn():
                return token_sparse_fwd(q, k, v, tri_indices)

        elif method == "triton_block_sparse":
            triton_block = 64
            assert S % triton_block == 0
            q = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            k = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            v = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            q2k_index, q2k_num = build_triton_block_sparse_indices(
                S, topk, triton_block, device
            )
            q2k_index = q2k_index.expand(b, nhq, -1, -1).contiguous()
            q2k_num = q2k_num.expand(b, nhq, -1).contiguous()

            def fn():
                return block_sparse_fwd(q, k, v, q2k_index, q2k_num)

        elif method == "effective_kernels":
            if not HAS_EFFECTIVE_KERNELS:
                raise ImportError(
                    "effective_kernels not installed. "
                    "Install from: https://github.com/Kwai-Keye/EffectiveKernels"
                )

            # DSA pipeline: topk_block_unique (preprocessing) + sparse_attention_forward
            q_ek = torch.randn(b * S, nhq, hd, device=device, dtype=dtype)
            k_ek = torch.randn(b * S, nhk, hd, device=device, dtype=dtype)
            v_ek = torch.randn(b * S, nhk, hd, device=device, dtype=dtype)
            cu_seqlens_q = torch.tensor([0, b * S], dtype=torch.int32, device=device)

            # Preprocess: token-level topk indices → block-unique format
            topk_block = 16  # EffectiveKernels uses block_size=16 for topk=2048
            topk_vals = torch.randint(
                0, S, (b * S, topk), device=device, dtype=torch.int32
            ).sort(dim=1).values
            seqlens = torch.tensor([b * S], dtype=torch.int32, device=device)
            unique_vals, qmask, block_counts = topk_block_unique(
                topk_vals, seqlens, topk_block, S, S, is_sorted=True
            )

            def fn():
                return sparse_attention_forward(
                    q_ek, k_ek, v_ek, cu_seqlens_q,
                    unique_vals, qmask, block_counts, topk=topk,
                )

        elif method == "tilelang_sparse_mla":
            if not _ensure_tilelang():
                raise ImportError("tilelang not installed")

            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            kv = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            indices = (
                torch.randint(0, S, (b, S, nhk, topk), device=device)
                .sort(dim=-1)
                .values.int()
            )

            kernel = get_tilelang_sparse_mla_kernel()

            def fn():
                return kernel(q, kv, indices)

        else:
            raise ValueError(f"Unknown method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return sparse_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        print(f"[{method}] S={S}: {e}")
        perf_dict = {"flops": [-1, -1, -1]}

    return perf_dict


# ─── Sanity Check ─────────────────────────────────────────────────────────────


def _ref_token_sparse_attn(q, k, v, indices, sm_scale=None):
    """Reference implementation: naive token-sparse attention for correctness.

    Args:
        q: (total_q, Hq, D)
        k: (total_kv, 1, D)
        v: (total_kv, 1, D)
        indices: (total_q, topk) int — per-query KV token indices

    Returns:
        o: (total_q, Hq, D)
    """
    total_q, Hq, D = q.shape
    topk_n = indices.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / (D**0.5)

    q_f = q.float()
    k_f = k.squeeze(1).float()
    v_f = v.squeeze(1).float()

    o = torch.zeros_like(q_f)
    for i in range(total_q):
        idx = indices[i].long()
        ki = k_f[idx]  # (topk, D)
        vi = v_f[idx]  # (topk, D)
        qi = q_f[i]  # (Hq, D)
        scores = (qi @ ki.T) * sm_scale  # (Hq, topk)
        weights = torch.softmax(scores, dim=-1)  # (Hq, topk)
        o[i] = weights @ vi  # (Hq, D)
    return o.to(q.dtype)


def sanity_check(S_check=512, topk_check=128):
    """Run a small correctness check for each method before benchmarking."""
    device = torch.cuda.current_device()
    print(f"\n{'='*60}")
    print(f"Sanity Check: S={S_check}, topk={topk_check}, nhq={nhq}, nhk={nhk}, hd={hd}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Shared data for reference
    q_3d = torch.randn(b * S_check, nhq, hd, device=device, dtype=dtype)
    k_3d = torch.randn(b * S_check, 1, hd, device=device, dtype=dtype)
    v_3d = torch.randn(b * S_check, 1, hd, device=device, dtype=dtype)
    indices_2d = torch.randint(
        0, b * S_check, (b * S_check, topk_check), device=device, dtype=torch.int32
    ).sort(dim=1).values

    # Reference output
    ref_out = _ref_token_sparse_attn(q_3d, k_3d, v_3d, indices_2d)

    results = {}

    # 1. FFA IndexAttn
    try:
        total_q = b * S_check
        local_pos = indices_2d.long()
        h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
        ffa_indices = (local_pos.unsqueeze(1) * nhk + h_offsets).int()
        ffa_out, _ = ffa_func(
            q_3d, k_3d, v_3d,
            index_attn_indices=ffa_indices,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=True,
        )
        err = (ffa_out.float() - ref_out.float()).abs().max().item()
        results["ffa_index_attn"] = err
    except Exception as e:
        results["ffa_index_attn"] = f"ERROR: {e}"

    # 2. Triton Token-Sparse
    try:
        tri_out = token_sparse_fwd(q_3d, k_3d, v_3d, indices_2d)
        err = (tri_out.float() - ref_out.float()).abs().max().item()
        results["triton_token_sparse"] = err
    except Exception as e:
        results["triton_token_sparse"] = f"ERROR: {e}"

    # 3. TileLang
    try:
        if _ensure_tilelang():
            q_4d = q_3d.view(b, S_check, nhq, hd)
            kv_4d = k_3d.view(b, S_check, nhk, hd)
            tl_indices = indices_2d.view(b, S_check, nhk, topk_check).int()
            kernel = get_tilelang_sparse_mla_kernel()
            tl_out, _ = kernel(q_4d, kv_4d, tl_indices)
            tl_out_3d = tl_out.view(b * S_check, nhq, hd)
            err = (tl_out_3d.float() - ref_out.float()).abs().max().item()
            results["tilelang_sparse_mla"] = err
        else:
            results["tilelang_sparse_mla"] = "SKIP (not installed)"
    except Exception as e:
        results["tilelang_sparse_mla"] = f"ERROR: {e}"

    # Print results
    print(f"\n{'Method':<30} {'Max Abs Error':<20} {'Status'}")
    print("-" * 60)
    for method, val in results.items():
        if isinstance(val, float):
            status = "PASS" if val < 0.05 else "FAIL"
            print(f"{method:<30} {val:<20.6f} {status}")
        else:
            print(f"{method:<30} {val}")
    print()

    return results


# ─── BWD Benchmark ────────────────────────────────────────────────────────────

BWD_METHODS = ["ffa_index_attn"]
BWD_METHOD_NAMES = ["FFA IndexAttn (token-sparse)"]
BWD_METHOD_STYLES = [("red", "-")]

bwd_flops_configs = [
    Benchmark(
        x_names=["S"],
        x_vals=seqlen_vals,
        x_log=True,
        line_arg="method",
        line_vals=BWD_METHODS,
        line_names=BWD_METHOD_NAMES,
        styles=BWD_METHOD_STYLES,
        ylabel={"flops": "Effective Sparse TFLOPs/s"},
        plot_name=(
            f"Token-Sparse Attention BWD (topk={topk}) — MQA (PackGQA)\n"
            f"nhq={nhq}, nhk={nhk}, D={hd}, BWD"
        ),
        args={},
    ),
]


@perf_report(bwd_flops_configs)
def bwd_benchmark(S, method):
    device = torch.cuda.current_device()
    sparse_flops = 4 * S * topk * nhq * hd * 2.5  # BWD ~2.5x FWD flops

    try:
        if method == "ffa_index_attn":
            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)

            index_attn_indices = build_index_attn_indices(b, S, nhk, topk, device)
            q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
            k_t = rearrange(k, "b s h d -> (b s h) 1 d")
            v_t = rearrange(v, "b s h d -> (b s h) 1 d")
            del q, k, v
            torch.cuda.empty_cache()

            q_t.requires_grad_(True)
            k_t.requires_grad_(True)
            v_t.requires_grad_(True)

            out, _ = ffa_func(
                q_t, k_t, v_t,
                index_attn_indices=index_attn_indices,
                q_block_size=1,
                k_block_size=1,
                pack_gqa=True,
            )
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        else:
            raise ValueError(f"Unknown BWD method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return sparse_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        print(f"[BWD {method}] S={S}: {e}")
        perf_dict = {"flops": [-1, -1, -1]}

    return perf_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Token-Sparse Attention Benchmark")
    parser.add_argument("--skip-sanity", action="store_true", help="Skip correctness check")
    parser.add_argument("--bwd", action="store_true", help="Also run BWD benchmark")
    parser.add_argument("--fwd-only", action="store_true", help="Only run FWD benchmark")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Subset of methods to run (e.g., --methods ffa_index_attn triton_token_sparse)"
    )
    args = parser.parse_args()

    if args.methods:
        valid_methods = METHODS[:]
        for m in args.methods:
            assert m in valid_methods, f"Unknown method '{m}'. Valid: {valid_methods}"
        idx_map = {m: i for i, m in enumerate(valid_methods)}
        selected_idx = [idx_map[m] for m in args.methods]
        METHODS = [valid_methods[i] for i in selected_idx]
        METHOD_NAMES = [METHOD_NAMES[i] for i in selected_idx]
        METHOD_STYLES = [METHOD_STYLES[i] for i in selected_idx]
        attn_flops_configs[0] = Benchmark(
            x_names=["S"],
            x_vals=seqlen_vals,
            x_log=True,
            line_arg="method",
            line_vals=METHODS,
            line_names=METHOD_NAMES,
            styles=METHOD_STYLES,
            ylabel={"flops": "Effective Sparse TFLOPs/s"},
            plot_name=attn_flops_configs[0].plot_name,
            args={},
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir,
        os.path.join("outs", f"bench_index_attn_comparison_{current_time}"),
    )

    # Sanity check before benchmarking
    if not args.skip_sanity:
        sanity_check()

    # FWD benchmark
    print("\n" + "=" * 60)
    print("FWD Benchmark")
    print("=" * 60)
    comparison_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )

    # BWD benchmark
    if args.bwd and not args.fwd_only:
        print("\n" + "=" * 60)
        print("BWD Benchmark")
        print("=" * 60)
        bwd_benchmark.run(
            print_data=True, print_value_on_bar=False, save_path=out_root
        )
