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

"""TileLang sparse attention kernel for token-sparse attention benchmark.

Adapted from TileLang examples/deepseek_v32 sparse_mla_fwd.
Uses T.copy + T.gemm without manual ptx_cp_async to avoid alignment issues.

Config: nhq=128, nhk=1, hd=128, topk=2048
"""

import torch

_kernel_cache: dict = {}


def get_tilelang_sparse_attn_kernel(heads=128, dim=128, topk=2048, kv_group=1):
    cache_key = (heads, dim, topk, kv_group)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    sm_scale_val = dim**-0.5

    # TileLang JIT cannot capture closures — must use exec to inject literal values
    kernel_code = f"""
import tilelang
from tilelang import language as T

@tilelang.jit(
    pass_configs={{tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}},
)
def sparse_attn_fwd(
    Q,
    KV,
    Indices,
    heads={heads},
    dim={dim},
    topk={topk},
    kv_group={kv_group},
    sm_scale={sm_scale_val},
    block_I=64,
    num_stages=2,
    threads=128,
):
    batch, seq_len, seq_len_kv = T.dynamic("batch, seq_len, seq_len_kv")
    _head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    dtype = T.bfloat16
    accum_dtype = T.float32
    indices_dtype = T.int32

    BI = block_I
    NI = tilelang.cdiv(topk, BI)
    D = dim

    lse_shape = [batch, seq_len, heads]

    Q: T.Tensor(q_shape, dtype)
    KV: T.Tensor(kv_shape, dtype)
    Indices: T.Tensor(indices_shape, indices_dtype)
    Output = T.empty(o_shape, dtype)
    Lse_out = T.empty(lse_shape, accum_dtype)

    with T.Kernel(seq_len, batch, kv_group, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([_head_kv, D], dtype)
        KV_shared = T.alloc_shared([BI, D], dtype)
        S_shared = T.alloc_shared([_head_kv, BI], dtype)
        O_local = T.alloc_fragment([_head_kv, D], accum_dtype)
        S_local = T.alloc_fragment([_head_kv, BI], accum_dtype)
        m_i = T.alloc_fragment([_head_kv], accum_dtype)
        m_i_new = T.alloc_fragment([_head_kv], accum_dtype)
        l_i = T.alloc_fragment([_head_kv], accum_dtype)
        l_i_new = T.alloc_fragment([_head_kv], accum_dtype)

        s_i = bx
        b_i = by
        g_i = bz

        T.copy(Q[b_i, s_i, g_i * _head_kv:(g_i + 1) * _head_kv, 0:D], Q_shared)
        T.fill(O_local, 0)
        T.fill(m_i, -1e30)
        T.fill(l_i, 0)

        for i in T.serial(NI):
            for bi in T.serial(BI):
                idx = Indices[b_i, s_i, g_i, i * BI + bi]
                T.copy(KV[b_i, idx, g_i, 0:D], KV_shared[bi, 0:D])

            T.fill(S_local, 0)
            T.gemm(Q_shared, KV_shared, S_local, transpose_B=True)

            # Apply sm_scale before online softmax (keep m_i in scaled space)
            for h, j in T.Parallel(_head_kv, BI):
                S_local[h, j] *= sm_scale

            # Online softmax
            T.copy(m_i, m_i_new)
            T.reduce_max(S_local, m_i_new, dim=1, clear=False)

            for h in T.Parallel(_head_kv):
                alpha = T.exp2((m_i[h] - m_i_new[h]) * 1.4426950408889634)
                l_i[h] = l_i[h] * alpha
            for h, d in T.Parallel(_head_kv, D):
                alpha = T.exp2((m_i[h] - m_i_new[h]) * 1.4426950408889634)
                O_local[h, d] *= alpha

            for h, j in T.Parallel(_head_kv, BI):
                S_local[h, j] = T.exp2(
                    (S_local[h, j] - m_i_new[h]) * 1.4426950408889634
                )
            T.reduce_sum(S_local, l_i_new, dim=1)
            for h in T.Parallel(_head_kv):
                l_i[h] = l_i[h] + l_i_new[h]

            T.copy(S_local, S_shared)
            T.gemm(S_shared, KV_shared, O_local)

            T.copy(m_i_new, m_i)

        for h, d in T.Parallel(_head_kv, D):
            O_local[h, d] /= l_i[h]

        T.copy(O_local, Output[b_i, s_i, g_i * _head_kv:(g_i + 1) * _head_kv, 0:D])

        # Store LSE = m_i + log(l_i) (in log2 space: m_i * log2e + log2(l_i))
        for h in T.Parallel(_head_kv):
            Lse_out[b_i, s_i, g_i * _head_kv + h] = m_i[h] + T.log2(l_i[h]) / 1.4426950408889634

    return Output, Lse_out
"""
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(kernel_code)
        f.flush()
        spec = importlib.util.spec_from_file_location("_tl_kernel", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    _kernel_cache[cache_key] = mod.sparse_attn_fwd
    return mod.sparse_attn_fwd


def tilelang_sparse_attn_fwd(q, kv, indices, heads=128, dim=128, topk=2048, kv_group=1):
    """Run tilelang sparse attention forward.

    Args:
        q: (B, S, nhq, D) bf16
        kv: (B, S, nhk, D) bf16
        indices: (B, S, nhk, topk) int32
    Returns:
        output: (B, S, nhq, D) bf16
        lse: (B, S, nhq) float32
    """
    kernel = get_tilelang_sparse_attn_kernel(heads, dim, topk, kv_group)
    return kernel(q, kv, indices)


_bwd_kernel_cache: dict = {}


def get_tilelang_sparse_attn_bwd_kernels(heads=128, dim=128, topk=2048, kv_group=1):
    """Get TileLang BWD kernels (preprocess + bwd + postprocess).

    Adapted from tilelang/examples/deepseek_v32/sparse_mla_bwd.py.
    Simplified for our case: K=V (MLA), no D_tail split, is_causal=False.
    """
    cache_key = (heads, dim, topk, kv_group)
    if cache_key in _bwd_kernel_cache:
        return _bwd_kernel_cache[cache_key]

    sm_scale_val = dim**-0.5

    kernel_code = f"""
import tilelang
from tilelang import language as T

@tilelang.jit
def preprocess(
    O,
    dO,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    B, S, H, D_dim = T.const("B, S, H, D_dim")
    shape = [B, S, H, D_dim]
    O: T.Tensor(shape, dtype)
    dO: T.Tensor(shape, dtype)
    Delta = T.empty([B, S, H], accum_dtype)

    with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
        o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        delta = T.alloc_fragment([block_ND], accum_dtype)
        acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
        T.clear(acc)
        for k in T.Pipelined(T.ceildiv(D_dim, block_ND), num_stages=num_stages):
            T.copy(O[bz, by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND], o)
            T.copy(dO[bz, by * block_ND:(by + 1) * block_ND, bx, k * block_ND:(k + 1) * block_ND], do)
            for i, j in T.Parallel(block_ND, block_ND):
                acc[i, j] += o[i, j] * do[i, j]
        T.reduce_sum(acc, delta, 1)
        T.copy(delta, Delta[bz, by * block_ND:(by + 1) * block_ND, bx])
    return Delta


@tilelang.jit(
    pass_configs={{
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    }},
)
def bwd_kernel(
    Q,
    KV,
    dO,
    Indices,
    Lse,
    Delta,
    dKV,
    heads={heads},
    dim={dim},
    topk={topk},
    kv_group={kv_group},
    sm_scale={sm_scale_val},
    block_size=32,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    B, S, S_kv = T.const("B, S, S_kv")
    sm_scale_log2 = sm_scale * 1.44269504
    H_kv = heads // kv_group
    q_shape = [B, S, heads, dim]
    kv_shape = [B, S_kv, kv_group, dim]
    o_shape = [B, S, heads, dim]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, heads]
    lse_shape = [B, S, heads]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    Q: T.Tensor(q_shape, dtype)
    KV: T.Tensor(kv_shape, dtype)
    dO: T.Tensor(o_shape, dtype)
    Indices: T.Tensor(indices_shape, indices_dtype)
    Lse: T.Tensor(lse_shape, accum_dtype)
    Delta: T.Tensor(delta_shape, accum_dtype)
    dQ = T.empty(q_shape, dtype)
    dKV: T.Tensor(kv_shape, accum_dtype)

    with T.Kernel(S, B, kv_group * NH, threads=threads) as (s_i, by, bz):
        Q_shared = T.alloc_shared([block_H, dim], dtype)
        KV_shared = T.alloc_shared([BS, dim], dtype)
        dO_shared = T.alloc_shared([block_H, dim], dtype)

        P_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
        dQ_shared = T.alloc_shared([block_H, dim], dtype)

        acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
        acc_dq = T.alloc_fragment([block_H, dim], accum_dtype)
        acc_dkv = T.alloc_fragment([BS, dim], accum_dtype)
        acc_dkv_shared = T.alloc_shared([BS, dim], accum_dtype)

        T.copy(Q[by, s_i, bz * block_H:(bz + 1) * block_H, 0:dim], Q_shared)
        T.copy(dO[by, s_i, bz * block_H:(bz + 1) * block_H, 0:dim], dO_shared)
        T.clear(acc_dq)

        for i_i in T.Pipelined(NS, num_stages=num_stages):
            for bi_i in T.Parallel(BS):
                acc_p[0, bi_i] = 0

            for bi_i, d_i in T.Parallel(BS, dim):
                KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

            T.clear(acc_p)
            T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_log2 - Lse[by, s_i, bz * block_H + h_i])

            T.copy(acc_p, P_shared_cast)

            T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

            for h_i, bi_i in T.Parallel(block_H, BS):
                acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale

            T.copy(acc_dp, dP_shared_cast)
            T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

            T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
            T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

            T.copy(acc_dkv, acc_dkv_shared)
            for bi_i, d_i in T.Parallel(BS, dim // 4):
                T.atomic_addx4(
                    dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i * 4],
                    acc_dkv_shared[bi_i, d_i * 4],
                )

        T.copy(acc_dq, dQ_shared)
        T.copy(dQ_shared, dQ[by, s_i, bz * block_H:(bz + 1) * block_H, 0:dim])

    return dQ
"""
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(kernel_code)
        f.flush()
        spec = importlib.util.spec_from_file_location("_tl_bwd_kernel", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    _bwd_kernel_cache[cache_key] = (mod.preprocess, mod.bwd_kernel)
    return mod.preprocess, mod.bwd_kernel


def tilelang_sparse_attn_bwd(
    q, kv, o, do, indices, lse, heads=128, dim=128, topk=2048, kv_group=1
):
    """Run tilelang sparse attention backward.

    Args:
        q: (B, S, nhq, D) bf16
        kv: (B, S_kv, nhk, D) bf16
        o: (B, S, nhq, D) bf16 — FWD output
        do: (B, S, nhq, D) bf16 — grad of output
        indices: (B, S, nhk, topk) int32
        lse: (B, S, nhq) float32 — from FWD

    Returns:
        dq: (B, S, nhq, D) bf16
        dkv: (B, S_kv, nhk, D) bf16
    """
    preprocess_fn, bwd_fn = get_tilelang_sparse_attn_bwd_kernels(
        heads, dim, topk, kv_group
    )

    delta = preprocess_fn(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_fn(q, kv, do, indices, lse, delta, dkv)
    dkv = dkv.to(kv.dtype)
    return dq, dkv


if __name__ == "__main__":
    torch.manual_seed(0)
    B, S, nhq, nhk, D, topk = 1, 4096, 128, 1, 128, 2048
    kv_group = nhk

    q = torch.randn(B, S, nhq, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, S, nhk, D, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, S, (B, S, nhk, topk), device="cuda", dtype=torch.int32)
    indices = indices.sort(dim=-1).values

    print(f"Compiling kernel: B={B}, S={S}, nhq={nhq}, nhk={nhk}, D={D}, topk={topk}")
    print(f"  kv_group={kv_group}, head_kv={nhq // kv_group}")
    out = tilelang_sparse_attn_fwd(
        q, kv, indices, heads=nhq, dim=D, topk=topk, kv_group=kv_group
    )
    print(f"Output: {out.shape}, dtype={out.dtype}")
    print(f"Output sample: {out[0, 0, 0, :4]}")
    print("SUCCESS")
