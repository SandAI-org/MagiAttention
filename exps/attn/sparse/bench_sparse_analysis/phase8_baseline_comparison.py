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

"""Phase 8: Baseline Comparison — FFA IndexSparse/BlockSparse vs external kernels.

Compare MagiAttention sparse kernels against publicly available baselines:
  - FlexAttention: PyTorch flex_attention + block_mask (Triton compiled)
  - Triton Token-Sparse: Hand-written Triton MQA-optimized (tl.dot)

Two sub-benchmarks:
  8a) IndexSparse (kbs=1): token-level sparsity
  8b) BlockSparse (kbs=128): block-level sparsity (FlexAttention only)

Config: nhq=128, nhk=1, hd=128, topk=2048 (fixed), sweep S=[32K..512K].
Passes: FWD, BWD.
"""

import gc
import os
import sys

from bench_sparse_analysis._common import (
    HD,
    NHK,
    NHQ,
    _bench_kernel,
    _load_results,
    _out_dir,
    _save_results,
    _set_gpu,
    _ts,
)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

PHASE = "8-baseline-comparison"
TOPK = 2048
KBS_BLOCK = 128
SEQLEN_VALS = [32768, 65536, 131072, 262144, 524288]
PASSES = ["fwd", "bwd"]

# Index Sparse methods (kbs=1)
IS_METHODS = ["ffa_index_sparse", "flexattention", "triton_token_sparse"]
IS_METHOD_LABELS = {
    "ffa_index_sparse": "FFA IndexSparse",
    "flexattention": "FlexAttention",
    "triton_token_sparse": "Triton Token-Sparse",
}

# Block Sparse methods (kbs=128)
BS_METHODS = ["ffa_block_sparse", "flexattention"]
BS_METHOD_LABELS = {
    "ffa_block_sparse": "FFA BlockSparse",
    "flexattention": "FlexAttention",
}


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _calc_sparse_flops(S, topk, is_bwd):
    fwd = 4 * S * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_index_sparse_indices(S, topk, device):
    """(S, NHK, topk) int32 — random per-Q-position token indices."""
    import torch

    idx = torch.randint(0, S, (S, topk), device=device, dtype=torch.int32)
    idx = idx.sort(dim=1).values
    return idx.unsqueeze(1).expand(-1, NHK, -1).contiguous()


def _build_flex_block_mask(S, topk, device):
    """FlexAttention block_mask for sparse pattern (128x128 blocks)."""
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_BLOCK = 128
    num_q_blocks = S // FLEX_BLOCK
    num_kv_blocks = S // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected = torch.rand(num_q_blocks, num_kv_blocks, device=device).argsort(dim=1)[
        :, :kv_blocks_needed
    ]
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_block = q_idx // FLEX_BLOCK
        kv_block = kv_idx // FLEX_BLOCK
        return mask_dense[q_block, kv_block]

    return create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=device
    )


def _build_block_sparse_indices(S, topk, device):
    """(S, NHK, n_topk_blocks) int32 — block-level indices for FFA BlockSparse."""
    import torch

    n_kv_blocks = S // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    if n_topk_blocks >= n_kv_blocks:
        idx = torch.arange(n_kv_blocks, dtype=torch.int32, device=device)
        return idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
    idx = (
        torch.rand(S, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values
    )
    return idx.unsqueeze(1).expand(-1, NHK, -1).to(torch.int32).contiguous()


def _has_entry(results, key, S):
    d = results.get(key, {})
    if not d or "seqlen" not in d:
        return False
    try:
        idx = d["seqlen"].index(S)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, S, tflops, ms):
    if key not in results:
        results[key] = {"seqlen": [], "tflops": [], "ms": []}
    d = results[key]
    if S in d["seqlen"]:
        idx = d["seqlen"].index(S)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["seqlen"].append(S)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


# ═══════════════════════════════════════════════════════════════
#  Index Sparse Runners
# ═══════════════════════════════════════════════════════════════


def _run_ffa_index_sparse(S, topk, is_bwd, device):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = _build_index_sparse_indices(S, topk, device)

    kw = dict(
        index_sparse_indices=indices,
        q_block_size=1,
        sparse_k_block_size=1,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    flops = _calc_sparse_flops(S, topk, is_bwd)
    return _bench_kernel(run_fn, flops, device)


def _run_flexattention(S, topk, is_bwd, device):
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    q = torch.randn(1, NHQ, S, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, NHK, S, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, NHK, S, HD, dtype=torch.bfloat16, device=device)
    block_mask = _build_flex_block_mask(S, topk, device)
    _flex_fn = torch.compile(flex_attention)

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

    flops = _calc_sparse_flops(S, topk, is_bwd)
    return _bench_kernel(run_fn, flops, device)


def _run_triton_token_sparse(S, topk, is_bwd, device):
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd

    q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(S, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(S, 1, HD, dtype=torch.bfloat16, device=device)
    tri_indices = (
        torch.randint(0, S, (S, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values
    )

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = token_sparse_attn(q, k, v, tri_indices)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    flops = _calc_sparse_flops(S, topk, is_bwd)
    return _bench_kernel(run_fn, flops, device)


# ═══════════════════════════════════════════════════════════════
#  Block Sparse Runners
# ═══════════════════════════════════════════════════════════════


def _run_ffa_block_sparse(S, topk, is_bwd, device):
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = _build_block_sparse_indices(S, topk, device)

    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_BLOCK, num_k_blocks=S // KBS_BLOCK
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)

    kw = dict(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=atm,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    flops = _calc_sparse_flops(S, topk, is_bwd)
    return _bench_kernel(run_fn, flops, device)


# ═══════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════

_RUNNERS = {
    "ffa_index_sparse": _run_ffa_index_sparse,
    "ffa_block_sparse": _run_ffa_block_sparse,
    "flexattention": _run_flexattention,
    "triton_token_sparse": _run_triton_token_sparse,
}


# ═══════════════════════════════════════════════════════════════
#  --exp
# ═══════════════════════════════════════════════════════════════


def _phase8_bench(force=False, rerun_filter=None):
    import torch

    results = _load_results(PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    print(f"[{_ts()}] Phase 8: Baseline Comparison (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, topk={TOPK}, "
        f"S=[{','.join(f'{s // 1024}k' for s in SEQLEN_VALS)}]",
        flush=True,
    )
    print(f"  Passes: {PASSES}\n", flush=True)

    # --- 8a: IndexSparse ---
    print("  " + "─" * 50, flush=True)
    print("  8a) IndexSparse (kbs=1) vs Baselines", flush=True)
    print("  " + "─" * 50, flush=True)
    for pass_type in PASSES:
        is_bwd = pass_type == "bwd"
        for method in IS_METHODS:
            for S in SEQLEN_VALS:
                key = f"is/{pass_type}/{method}"
                if rerun_filter and (pass_type, method, S) not in rerun_filter:
                    continue
                if not force and _has_entry(results, key, S):
                    d = results[key]
                    idx = d["seqlen"].index(S)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"{tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                try:
                    runner = _RUNNERS[method]
                    tf, ms = runner(S, TOPK, is_bwd, device)
                    _set_entry(results, key, S, round(tf, 1), round(ms, 3))
                    _save_results(PHASE, results)
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"{tf:>7.1f} T  ({ms:.3f} ms)",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: OOM",
                        flush=True,
                    )
                    _set_entry(results, key, S, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"ERROR — {e}",
                        flush=True,
                    )
                    _set_entry(results, key, S, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()

    # --- 8b: BlockSparse ---
    print("\n  " + "─" * 50, flush=True)
    print(f"  8b) BlockSparse (kbs={KBS_BLOCK}) vs Baselines", flush=True)
    print(f"  {'─' * 50}", flush=True)
    for pass_type in PASSES:
        is_bwd = pass_type == "bwd"
        for method in BS_METHODS:
            for S in SEQLEN_VALS:
                key = f"bs/{pass_type}/{method}"
                if rerun_filter and (pass_type, method, S) not in rerun_filter:
                    continue
                if not force and _has_entry(results, key, S):
                    d = results[key]
                    idx = d["seqlen"].index(S)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"{tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                try:
                    runner = _RUNNERS[method]
                    tf, ms = runner(S, TOPK, is_bwd, device)
                    _set_entry(results, key, S, round(tf, 1), round(ms, 3))
                    _save_results(PHASE, results)
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"{tf:>7.1f} T  ({ms:.3f} ms)",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: OOM",
                        flush=True,
                    )
                    _set_entry(results, key, S, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(
                        f"    {pass_type:4s} {method:22s} S={S // 1024:>4d}k: "
                        f"ERROR — {e}",
                        flush=True,
                    )
                    _set_entry(results, key, S, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()

    print(f"\n[{_ts()}] Phase 8 done.", flush=True)


# ═══════════════════════════════════════════════════════════════
#  --plot
# ═══════════════════════════════════════════════════════════════


def _phase8_plot():
    import matplotlib.pyplot as plt

    results = _load_results(PHASE)
    if not results:
        print("  [WARN] No results found. Run --exp first.")
        return

    out = _out_dir(PHASE)
    os.makedirs(out, exist_ok=True)

    for sub, methods, labels in [
        ("is", IS_METHODS, IS_METHOD_LABELS),
        ("bs", BS_METHODS, BS_METHOD_LABELS),
    ]:
        for pass_type in PASSES:
            fig, ax = plt.subplots(figsize=(10, 6))
            has_data = False
            for method in methods:
                key = f"{sub}/{pass_type}/{method}"
                d = results.get(key)
                if not d:
                    continue
                seqlens = d["seqlen"]
                tflops = d["tflops"]
                valid = [(s, t) for s, t in zip(seqlens, tflops) if t is not None]
                if not valid:
                    continue
                xs, ys = zip(*valid)
                ax.plot(
                    [x // 1024 for x in xs],
                    ys,
                    marker="o",
                    label=labels[method],
                    linewidth=2,
                )
                has_data = True

            if not has_data:
                plt.close(fig)
                continue

            kbs_label = "kbs=1" if sub == "is" else f"kbs={KBS_BLOCK}"
            title = (
                f"Sparse Attention {pass_type.upper()} — {kbs_label}\n"
                f"nhq={NHQ}, nhk={NHK}, hd={HD}, topk={TOPK}"
            )
            ax.set_title(title)
            ax.set_xlabel("Sequence Length (K)")
            ax.set_ylabel("Effective Sparse TFLOPS/s")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)
            ax.set_xticks([s // 1024 for s in SEQLEN_VALS])
            ax.get_xaxis().set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{int(x)}K")
            )

            fname = f"{sub}_{pass_type}.png"
            fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {os.path.join(out, fname)}", flush=True)

    # Summary table
    print("\n  " + "─" * 60)
    print("  Summary Table (TFLOPS)")
    print("  " + "─" * 60)
    for sub, methods, labels in [
        ("is", IS_METHODS, IS_METHOD_LABELS),
        ("bs", BS_METHODS, BS_METHOD_LABELS),
    ]:
        kbs_label = "kbs=1" if sub == "is" else f"kbs={KBS_BLOCK}"
        for pass_type in PASSES:
            print(f"\n  {pass_type.upper()} ({kbs_label}):")
            header = f"  {'S':>6s}"
            for m in methods:
                header += f"  {labels[m]:>20s}"
            print(header)
            for S in SEQLEN_VALS:
                row = f"  {S // 1024:>5d}k"
                for m in methods:
                    key = f"{sub}/{pass_type}/{m}"
                    d = results.get(key, {})
                    if d and S in d.get("seqlen", []):
                        idx = d["seqlen"].index(S)
                        tf = d["tflops"][idx]
                        row += f"  {tf:>20.1f}" if tf else f"  {'—':>20s}"
                    else:
                        row += f"  {'—':>20s}"
                print(row)
