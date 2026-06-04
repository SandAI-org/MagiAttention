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

Methods compared (FWD):
  - FFA IndexAttn (token-sparse, block_size=1, PackGQA)
  - FlexAttention (PyTorch flex_attention with sparse block mask, enable_gqa=True)
  - Triton Token-Sparse (MQA batching, all 128 heads as tl.dot GEMM)
  - TileLang Sparse Attn (T.copy + T.gemm, online softmax)
  - EffectiveKernels (Kwai-Keye DSA topk_block_unique pipeline, nhk=16 due to GQA limit)

Methods compared (BWD):
  - FFA IndexAttn
  - FlexAttention (autograd + torch.compile)
  - Triton Token-Sparse (dQ via tl.dot, dK/dV via atomic scatter)

Reporting effective sparse TFLOPs/s (FWD flops = 4*S*topk*nhq*hd, BWD ~2.5x FWD).

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
from baselines.token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd
from baselines.utils import seed_everything
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

# ─── Optional dependencies ────────────────────────────────────────────────────

try:
    from effective_kernels.ops.sparse_attention import sparse_attention_forward
    from effective_kernels.ops.topk_block_unique import topk_block_unique  # noqa: F401

    HAS_EFFECTIVE_KERNELS = True
except ImportError:
    # Try loading from local build
    import sys as _sys

    _ek_path = "/tmp/EffectiveKernels"
    if _ek_path not in _sys.path:
        _sys.path.insert(0, _ek_path)
    try:
        from effective_kernels.ops.sparse_attention import sparse_attention_forward
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


def build_flex_sparse_block_mask(b, S, topk, nhq, device="cuda"):
    """Build a flex_attention block_mask for token-sparse pattern.

    FlexAttention uses 128x128 blocks. For topk=2048, we select
    topk // 128 = 16 KV blocks per query block.
    """
    FLEX_BLOCK = 128
    num_q_blocks = S // FLEX_BLOCK
    num_kv_blocks = S // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected_kv_blocks = torch.rand(num_q_blocks, num_kv_blocks, device=device).argsort(
        dim=1
    )[:, :kv_blocks_needed]
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected_kv_blocks, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_block = q_idx // FLEX_BLOCK
        kv_block = kv_idx // FLEX_BLOCK
        return mask_dense[q_block, kv_block]

    block_mask = create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )
    return block_mask


# ─── TileLang Sparse Attention (T.copy + T.gemm, no manual cp.async) ──────────


def _ensure_tilelang():
    global HAS_TILELANG
    if not HAS_TILELANG:
        try:
            import tilelang as _tl  # noqa: F401

            HAS_TILELANG = True
        except ImportError:
            pass
    return HAS_TILELANG


def _get_tilelang_kernel():
    """Import and return the tilelang sparse attention kernel."""
    from baselines.tilelang_sparse_attn import get_tilelang_sparse_attn_kernel

    return get_tilelang_sparse_attn_kernel(heads=nhq, dim=hd, topk=topk, kv_group=nhk)


# ─── Config ───────────────────────────────────────────────────────────────────

b = 1
nhq = 128
nhk = 1  # MQA
hd = 128
topk = 2048
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]

seqlen_vals = [32768, 65536, 131072, 262144, 524288]

METHODS = [
    "ffa_index_attn",
    "flexattention",
    "triton_token_sparse",
    "tilelang_sparse_mla",
    "effective_kernels",  # last: may crash CUDA context at large S
]
METHOD_NAMES = [
    "FFA IndexAttn (token-sparse)",
    "FlexAttention (GQA, sparse mask)",
    "Triton Token-Sparse (MQA tl.dot)",
    "TileLang Sparse Attn (T.gemm)",
    "EffectiveKernels (DSA, nhk=16)",
]
METHOD_STYLES = [
    ("red", "-"),
    ("green", "-."),
    ("blue", "--"),
    ("brown", "-."),
    ("purple", "-"),
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


_cuda_corrupted = False
_disabled_methods = set()


@perf_report(attn_flops_configs)
def comparison_benchmark(S, method):
    global _cuda_corrupted
    if _cuda_corrupted or method in _disabled_methods:
        return {"flops": [-1, -1, -1]}

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
                    q_t,
                    k_t,
                    v_t,
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
            tri_indices = (
                torch.randint(0, b * S, (b * S, topk), device=device, dtype=torch.int32)
                .sort(dim=1)
                .values
            )

            def fn():
                return token_sparse_fwd(q, k, v, tri_indices)

        elif method == "effective_kernels":
            if not HAS_EFFECTIVE_KERNELS:
                raise ImportError(
                    "effective_kernels not installed. "
                    "Install from: https://github.com/Kwai-Keye/EffectiveKernels"
                )
            if S > 16384:
                raise RuntimeError(
                    f"EffectiveKernels crashes at S>{16384} (library bug)"
                )

            # EffectiveKernels requires qhead_per_kvhead <= 8.
            # Use nhk_ek=16 (qhead_per_kvhead=8) — same nhq=128, same effective FLOPs.
            nhk_ek = 16
            q_ek = torch.randn(b * S, nhq, hd, device=device, dtype=dtype)
            k_ek = torch.randn(b * S, nhk_ek, hd, device=device, dtype=dtype)
            v_ek = torch.randn(b * S, nhk_ek, hd, device=device, dtype=dtype)
            cu_seqlens_q = torch.tensor([0, b * S], dtype=torch.int32, device=device)

            topk_block = 16
            topk_vals = (
                torch.randint(0, S, (b * S, topk), device=device, dtype=torch.int32)
                .sort(dim=1)
                .values
            )
            seqlens = torch.tensor([b * S], dtype=torch.int32, device=device)
            unique_vals, qmask, block_counts = topk_block_unique(
                topk_vals, seqlens, topk_block, S, S, is_sorted=True
            )

            def fn():
                return sparse_attention_forward(
                    q_ek,
                    k_ek,
                    v_ek,
                    cu_seqlens_q,
                    unique_vals,
                    qmask,
                    block_counts,
                    topk=topk,
                )

        elif method == "tilelang_sparse_mla":
            if not _ensure_tilelang():
                raise ImportError("tilelang not installed")

            # kv_group = nhk (number of KV head groups)
            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            kv = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            indices = (
                torch.randint(0, S, (b, S, nhk, topk), device=device)
                .sort(dim=-1)
                .values.int()
            )

            kernel = _get_tilelang_kernel()

            def fn():
                return kernel(q, kv, indices)

        else:
            raise ValueError(f"Unknown method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return sparse_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except torch.cuda.OutOfMemoryError as e:
        print(f"[{method}] S={S}: OOM — {e}")
        perf_dict = {"flops": [-1, -1, -1]}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{method}] S={S}: {e}")
        perf_dict = {"flops": [-1, -1, -1]}
        if "CUDA error" in str(e) or "illegal memory" in str(e).lower():
            _disabled_methods.add(method)
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            try:
                torch.zeros(1, device="cuda")
            except RuntimeError:
                print(
                    f"  [WARNING] CUDA context corrupted by '{method}'. "
                    f"Disabling ALL remaining benchmarks."
                )
                _cuda_corrupted = True
        else:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
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
    _ = indices.shape[1]
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
    print(f"\n{'=' * 60}")
    print(
        f"Sanity Check: S={S_check}, topk={topk_check}, nhq={nhq}, nhk={nhk}, hd={hd}"
    )
    print(f"{'=' * 60}")

    torch.manual_seed(42)

    # Shared data for reference
    q_3d = torch.randn(b * S_check, nhq, hd, device=device, dtype=dtype)
    k_3d = torch.randn(b * S_check, 1, hd, device=device, dtype=dtype)
    v_3d = torch.randn(b * S_check, 1, hd, device=device, dtype=dtype)
    indices_2d = (
        torch.randint(
            0, b * S_check, (b * S_check, topk_check), device=device, dtype=torch.int32
        )
        .sort(dim=1)
        .values
    )

    # Reference output
    ref_out = _ref_token_sparse_attn(q_3d, k_3d, v_3d, indices_2d)

    results = {}

    # 1. FFA IndexAttn
    try:
        local_pos = indices_2d.long()
        h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
        ffa_indices = (local_pos.unsqueeze(1) * nhk + h_offsets).int()
        ffa_out, _ = ffa_func(
            q_3d,
            k_3d,
            v_3d,
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

    # 3. TileLang (MLA-style: uses single KV tensor for both K and V roles)
    try:
        if _ensure_tilelang():
            from baselines.tilelang_sparse_attn import get_tilelang_sparse_attn_kernel

            q_4d = q_3d.view(b, S_check, nhq, hd)
            kv_4d = k_3d.view(b, S_check, nhk, hd)
            tl_indices = indices_2d.view(b, S_check, nhk, topk_check).int()
            kernel = get_tilelang_sparse_attn_kernel(
                heads=nhq, dim=hd, topk=topk_check, kv_group=nhk
            )
            tl_result = kernel(q_4d, kv_4d, tl_indices)
            tl_out = tl_result[0] if isinstance(tl_result, (list, tuple)) else tl_result
            tl_out_3d = tl_out.view(b * S_check, nhq, hd)
            # Reference with K=V (TileLang uses same KV for both attention and output)
            ref_tl = _ref_token_sparse_attn(q_3d, k_3d, k_3d, indices_2d)
            err = (tl_out_3d.float() - ref_tl.float()).abs().max().item()
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

BWD_METHODS = [
    "ffa_index_attn",
    "flexattention",
    "triton_token_sparse",
    "tilelang_sparse_mla",
]
BWD_METHOD_NAMES = [
    "FFA IndexAttn (token-sparse)",
    "FlexAttention (GQA, sparse mask)",
    "Triton Token-Sparse (MQA tl.dot)",
    "TileLang Sparse Attn (T.gemm)",
]
BWD_METHOD_STYLES = [("red", "-"), ("green", "-."), ("blue", "--"), ("brown", "-.")]

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
    global _cuda_corrupted
    if _cuda_corrupted or method in _disabled_methods:
        return {"flops": [-1, -1, -1]}

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
                q_t,
                k_t,
                v_t,
                index_attn_indices=index_attn_indices,
                q_block_size=1,
                k_block_size=1,
                pack_gqa=True,
            )
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        elif method == "flexattention":
            assert S % 128 == 0, "FlexAttention requires S divisible by 128"
            q = torch.randn(
                b, nhq, S, hd, device=device, dtype=dtype, requires_grad=True
            )
            k = torch.randn(
                b, nhk, S, hd, device=device, dtype=dtype, requires_grad=True
            )
            v = torch.randn(
                b, nhk, S, hd, device=device, dtype=dtype, requires_grad=True
            )

            block_mask = build_flex_sparse_block_mask(b, S, topk, nhq, device)
            _flex_fn = torch.compile(flex_attention)
            out = _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        elif method == "triton_token_sparse":
            q = torch.randn(
                b * S, nhq, hd, device=device, dtype=dtype, requires_grad=True
            )
            k = torch.randn(
                b * S, 1, hd, device=device, dtype=dtype, requires_grad=True
            )
            v = torch.randn(
                b * S, 1, hd, device=device, dtype=dtype, requires_grad=True
            )
            tri_indices = (
                torch.randint(0, b * S, (b * S, topk), device=device, dtype=torch.int32)
                .sort(dim=1)
                .values
            )

            out = token_sparse_attn(q, k, v, tri_indices)
            do = torch.randn_like(out)

            def fn():
                out.backward(do, retain_graph=True)

        elif method == "tilelang_sparse_mla":
            if not _ensure_tilelang():
                raise ImportError("tilelang not installed")

            from baselines.tilelang_sparse_attn import (
                get_tilelang_sparse_attn_kernel,
                tilelang_sparse_attn_bwd,
            )

            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            kv = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            tl_indices = (
                torch.randint(0, S, (b, S, nhk, topk), device=device)
                .sort(dim=-1)
                .values.int()
            )

            kernel = get_tilelang_sparse_attn_kernel(
                heads=nhq, dim=hd, topk=topk, kv_group=nhk
            )
            fwd_result = kernel(q, kv, tl_indices)
            out, lse = fwd_result[0], fwd_result[1]
            do = torch.randn_like(out)

            def fn():
                return tilelang_sparse_attn_bwd(
                    q,
                    kv,
                    out,
                    do,
                    tl_indices,
                    lse,
                    heads=nhq,
                    dim=hd,
                    topk=topk,
                    kv_group=nhk,
                )

        else:
            raise ValueError(f"Unknown BWD method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return sparse_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except torch.cuda.OutOfMemoryError as e:
        print(f"[BWD {method}] S={S}: OOM — {e}")
        perf_dict = {"flops": [-1, -1, -1]}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[BWD {method}] S={S}: {e}")
        perf_dict = {"flops": [-1, -1, -1]}
        if "CUDA error" in str(e) or "illegal memory" in str(e).lower():
            _disabled_methods.add(method)
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            torch.cuda.empty_cache()
            try:
                torch.zeros(1, device="cuda")
            except RuntimeError:
                _cuda_corrupted = True
        else:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return perf_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Token-Sparse Attention Benchmark")
    parser.add_argument(
        "--skip-sanity", action="store_true", help="Skip correctness check"
    )
    parser.add_argument("--bwd", action="store_true", help="Also run BWD benchmark")
    parser.add_argument(
        "--fwd-only", action="store_true", help="Only run FWD benchmark (skip BWD)"
    )
    parser.add_argument(
        "--bwd-only", action="store_true", help="Only run BWD benchmark (skip FWD)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (e.g., --methods ffa_index_attn triton_token_sparse)",
    )
    args = parser.parse_args()

    if args.methods:
        valid_fwd = METHODS[:]
        valid_bwd = BWD_METHODS[:]
        all_valid = list(set(valid_fwd + valid_bwd))
        for m in args.methods:
            assert m in all_valid, f"Unknown method '{m}'. Valid: {all_valid}"

        # Filter FWD methods
        fwd_selected = [m for m in args.methods if m in valid_fwd]
        if fwd_selected:
            idx_map = {m: i for i, m in enumerate(valid_fwd)}
            selected_idx = [idx_map[m] for m in fwd_selected]
            METHODS = [valid_fwd[i] for i in selected_idx]
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

        # Filter BWD methods
        bwd_selected = [m for m in args.methods if m in valid_bwd]
        if bwd_selected:
            bwd_idx_map = {m: i for i, m in enumerate(valid_bwd)}
            bwd_selected_idx = [bwd_idx_map[m] for m in bwd_selected]
            BWD_METHODS = [valid_bwd[i] for i in bwd_selected_idx]
            BWD_METHOD_NAMES = [BWD_METHOD_NAMES[i] for i in bwd_selected_idx]
            BWD_METHOD_STYLES = [BWD_METHOD_STYLES[i] for i in bwd_selected_idx]
            bwd_flops_configs[0] = Benchmark(
                x_names=["S"],
                x_vals=seqlen_vals,
                x_log=True,
                line_arg="method",
                line_vals=BWD_METHODS,
                line_names=BWD_METHOD_NAMES,
                styles=BWD_METHOD_STYLES,
                ylabel={"flops": "Effective Sparse TFLOPs/s"},
                plot_name=bwd_flops_configs[0].plot_name,
                args={},
            )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir,
        os.path.join("outs", f"bench_index_attn_comparison_{current_time}"),
    )

    # Sanity check before benchmarking — auto-disable methods that crash
    if not args.skip_sanity:
        sanity_results = sanity_check()
        for m, val in sanity_results.items():
            if isinstance(val, str) and ("ERROR" in val or "SKIP" in val):
                _disabled_methods.add(m)
                print(f"  [AUTO-DISABLE] '{m}' disabled for benchmark (sanity failed)")
        if _disabled_methods:
            print()

    # FWD benchmark
    run_fwd = not args.bwd_only
    run_bwd = args.bwd or args.bwd_only

    if run_fwd:
        print("\n" + "=" * 60)
        print("FWD Benchmark")
        print("=" * 60)
        comparison_benchmark.run(
            print_data=True, print_value_on_bar=False, save_path=out_root
        )

    # BWD benchmark
    if run_bwd:
        print("\n" + "=" * 60)
        print("BWD Benchmark")
        print("=" * 60)
        bwd_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
