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
  - FlexAttention (PyTorch flex_attention with sparse block mask)
  - Triton Block-Sparse (block_size=64)
  - EffectiveKernels (Kwai-Keye DSA topk_block_unique pipeline)
  - TileLang Sparse MLA (examples/deepseek_v32)

FWD-only, reporting effective sparse TFLOPs/s (flops = 4*S*topk*nhq*hd).
"""

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func, flex_attn_func
from baselines.block_sparse_attn_triton import block_sparse_fwd
from baselines.utils import seed_everything
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

# ─── Optional dependencies ────────────────────────────────────────────────────

try:
    from effective_kernels.ops.topk_block_unique import topk_block_unique

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
    """JIT-compile the tilelang sparse attention forward kernel.

    Hardcoded for benchmark config: nhq=128, nhk=1, hd=128, topk=2048.
    TileLang JIT requires all constants to be literals in the function source.
    """
    global _tilelang_sparse_mla_kernel
    if _tilelang_sparse_mla_kernel is not None:
        return _tilelang_sparse_mla_kernel

    import tilelang
    from tilelang import language as T  # noqa: F811

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def sparse_attn_fwd(
        Q,
        KV,
        Indices,
        heads=128,
        dim=128,
        topk=2048,
        kv_group=1,
        sm_scale=0.12585414,
        block_I=64,
        num_stages=2,
        threads=256,
    ):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

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
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)

            for i_i in T.Pipelined(NI, num_stages=2):
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]

                T.fill(acc_s, 0)
                T.gemm(
                    Q_shared, KV_shared, acc_s, transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

        return Output, Lse

    _tilelang_sparse_mla_kernel = sparse_attn_fwd
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
    "triton_block_sparse",
    "effective_kernels",
    "tilelang_sparse_mla",
]
METHOD_NAMES = [
    "FFA IndexAttn (token-sparse)",
    "FlexAttention (sparse mask)",
    "Triton Block-Sparse (bs=64)",
    "EffectiveKernels (DSA)",
    "TileLang Sparse MLA",
]
METHOD_STYLES = [
    ("red", "-"),
    ("green", "-."),
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
            k = torch.randn(b, nhk, S, hd, device=device, dtype=dtype).expand(
                b, nhq, S, hd
            )
            v = torch.randn(b, nhk, S, hd, device=device, dtype=dtype).expand(
                b, nhq, S, hd
            )

            block_mask = build_flex_sparse_block_mask(b, S, topk, nhq, device)

            def fn():
                return flex_attn_func(q, k, v, block_mask=block_mask)

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
            raise NotImplementedError(
                "EffectiveKernels DSA benchmark requires custom integration. "
                "topk_block_unique is a preprocessing kernel — "
                "full DSA pipeline integration pending."
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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir,
        os.path.join("outs", f"bench_index_attn_comparison_{current_time}"),
    )

    comparison_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
