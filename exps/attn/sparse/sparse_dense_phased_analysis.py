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

"""Sparse/Dense phased TFLOPS difference analysis.

Experiment 1 — S=topk parity:
  When S=topk (no empty blocks, no scatter), IndexSparse kbs=128 should match
  Dense-1B across FWD / BWD-LoopQ / BWD-LoopK.
  Plot: 1 row x 3 cols, each with Dense-1B vs IndexSparse, x-axis = S value.

Experiment 2 — LoopQ vs LoopK L2 divergence:
  With fixed S=32K and topk shrinking, BWD LoopQ drops dramatically while
  LoopK stays stable. Root cause: LoopQ inner-loop accesses Q/dO via inv_indices
  (scattered) -> L2 thrash; LoopK inner-loop accesses sorted K-block indices
  (contiguous) -> L2 hit.
  Plot: 2 subplots — left: TFLOPS curves, right: L2 working-set annotation.

Usage:
  python exps/attn/sparse/sparse_dense_phased_analysis.py --bench1
  python exps/attn/sparse/sparse_dense_phased_analysis.py --plot1
  python exps/attn/sparse/sparse_dense_phased_analysis.py --plot2
  python exps/attn/sparse/sparse_dense_phased_analysis.py --all
"""

import argparse
import gc
import json
import os
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_L2_RESULTS = os.path.join(_SCRIPT_DIR, "outs", "l2_cache_sparse", "results.json")
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "phased_analysis")

NHQ, NHK, HD, KBS = 128, 1, 128, 128
WARMUP, ITERS = 8, 20
EXP1_S_VALS = [32768, 16384, 8192, 4096, 2048, 1024]
TOPK_VALS_FULL = [32768, 16384, 8192, 4096, 2048]


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _find_free_gpu():
    import torch

    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    best_idx, best_used = 0, float("inf")
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        used = (total - free) / (1024**2)
        if used < best_used:
            best_idx, best_used = i, used
    return best_idx


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1: Bench — S=topk parity (supplement missing S values)
# ═══════════════════════════════════════════════════════════════════
def run_bench1():
    """Benchmark Dense-1B and IndexSparse kbs=128 at S=topk for each S value."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    gpu = _find_free_gpu()
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    DTYPE = torch.bfloat16

    os.makedirs(_OUT_DIR, exist_ok=True)
    results_path = os.path.join(_OUT_DIR, "exp1_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    def bench_kernel(run_fn, flops):
        for _ in range(WARMUP):
            run_fn()
        torch.cuda.synchronize()
        l2_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            l2_flush.zero_()
            starts[i].record()
            run_fn()
            ends[i].record()
        torch.cuda.synchronize()
        del l2_flush
        times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
        ms = times[len(times) // 2]
        return flops / ms * 1e-9, ms

    def calc_flops(s, topk, is_bwd):
        fwd = 4 * s * topk * NHQ * HD
        return fwd * 2.5 if is_bwd else fwd

    PASSES = [
        ("fwd", "FWD", False),
        ("bwd_loopq", "BWD LoopQ", True),
        ("bwd_loopk", "BWD LoopK", True),
    ]

    for S in EXP1_S_VALS:
        for pass_id, pass_name, is_bwd in PASSES:
            topk = S
            flops = calc_flops(S, topk, is_bwd)
            n_kblocks = S // KBS

            for method_id, method_name in [
                ("d1b", "Dense-1B"),
                ("ia128", "IndexSparse kbs=128"),
            ]:
                key = f"{pass_id}/{method_id}/{S}"
                if key in results:
                    cached = results[key]
                    print(
                        f"  [{_ts()}] {pass_name} {method_name} S={S}: "
                        f"{cached['tflops']:.1f} T (cached)",
                        flush=True,
                    )
                    continue

                q = torch.randn(S, NHQ, HD, dtype=DTYPE, device=device)
                k = torch.randn(S, NHK, HD, dtype=DTYPE, device=device)
                v = torch.randn(S, NHK, HD, dtype=DTYPE, device=device)
                if is_bwd:
                    q.requires_grad_(True)
                    k.requires_grad_(True)
                    v.requires_grad_(True)

                if method_id == "d1b":
                    q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
                    k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
                    atm = torch.zeros(1, dtype=torch.int32, device=device)
                    kw = dict(
                        q_ranges=q_ranges,
                        k_ranges=k_ranges,
                        attn_type_map=atm,
                        pack_gqa=False,
                    )
                else:
                    idx = torch.arange(n_kblocks, dtype=torch.int32, device=device)
                    idx = idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
                    kw = dict(
                        index_sparse_indices=idx,
                        k_block_size=KBS,
                        index_sparse=True,
                        pack_gqa=True,
                    )

                if is_bwd:
                    kw["swap_bwd_qk_loop"] = pass_id == "bwd_loopk"

                out, *_ = flex_flash_attn_func(q, k, v, **kw)

                if not is_bwd:

                    def run_fn():
                        flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

                else:
                    do = torch.randn_like(out)

                    def run_fn():
                        out.backward(do, retain_graph=True)  # noqa: F821

                tf, ms = bench_kernel(run_fn, flops)
                results[key] = {
                    "tflops": round(tf, 1),
                    "ms": round(ms, 3),
                    "S": S,
                    "pass": pass_id,
                    "method": method_id,
                }
                print(
                    f"  [{_ts()}] {pass_name} {method_name} S={S}: "
                    f"{tf:.1f} T ({ms:.3f}ms) [gpu{gpu}]",
                    flush=True,
                )

                del q, k, v, out
                if is_bwd:
                    del do
                gc.collect()
                torch.cuda.empty_cache()

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n[{_ts()}] Exp1 bench done → {results_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Experiment 1: Plot — S=topk parity (3-panel)
# ═══════════════════════════════════════════════════════════════════
def run_plot1():
    """Generate 3-panel plot: FWD / BWD-LoopQ / BWD-LoopK at S=topk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(_OUT_DIR, exist_ok=True)

    # Load exp1 results (our bench)
    exp1_path = os.path.join(_OUT_DIR, "exp1_results.json")
    if not os.path.exists(exp1_path):
        print(f"ERROR: {exp1_path} not found. Run --bench1 first.")
        return
    with open(exp1_path) as f:
        exp1 = json.load(f)

    # Also try to pull from l2_cache_sparse results (dashed = S=topk)
    l2_data = {}
    if os.path.exists(_L2_RESULTS):
        with open(_L2_RESULTS) as f:
            l2_data = json.load(f)

    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]
    METHODS = [
        ("d1b", "Dense-1B", "#555555", "o", "-"),
        ("ia128", "IndexSparse kbs=128", "#C45680", "^", "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    for col, (pid, pname) in enumerate(PASSES):
        ax = axes[col]

        for mid, mlabel, color, marker, ls in METHODS:
            s_vals = []
            tflops_vals = []

            for S in EXP1_S_VALS:
                key = f"{pid}/{mid}/{S}"
                if key in exp1:
                    s_vals.append(S)
                    tflops_vals.append(exp1[key]["tflops"])
                else:
                    # Fallback: l2_cache_sparse dashed data
                    l2_method = "d1b_dashed" if mid == "d1b" else "ia_dashed"
                    pd = l2_data.get(pid, {}).get(l2_method, {})
                    if pd and S in pd.get("topk", []):
                        idx = pd["topk"].index(S)
                        val = pd["tflops"][idx]
                        if val is not None:
                            s_vals.append(S)
                            tflops_vals.append(val)

            if not s_vals:
                continue

            x = np.arange(len(s_vals))
            ax.plot(
                x,
                tflops_vals,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=7,
                linewidth=2.2,
                label=mlabel,
                alpha=0.9,
            )
            for i, (sv, tv) in enumerate(zip(s_vals, tflops_vals)):
                ax.annotate(
                    f"{tv:.0f}",
                    (i, tv),
                    textcoords="offset points",
                    xytext=(0, 8),
                    fontsize=7.5,
                    ha="center",
                    fontweight="bold",
                    color=color,
                )

            ax.set_xticks(np.arange(len(s_vals)))
            ax.set_xticklabels([f"{s // 1024}K" for s in s_vals])

        ax.set_title(pname, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("S = topk (sequence length)")
        ax.set_ylabel("TFLOPS")
        ax.legend(fontsize=9, loc="lower left", framealpha=0.8)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Experiment 1: S=topk Parity — IndexSparse kbs=128 vs Dense-1B\n"
        "(nhq=128 nhk=1 hd=128 bf16 H100, no empty blocks / no scatter)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "exp1_parity.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[{_ts()}] Exp1 plot → {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Experiment 2: Plot — LoopQ vs LoopK L2 divergence
# ═══════════════════════════════════════════════════════════════════
def run_plot2():
    """Generate 2-panel plot: TFLOPS curves + L2 working-set annotation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(_OUT_DIR, exist_ok=True)

    if not os.path.exists(_L2_RESULTS):
        print(f"ERROR: {_L2_RESULTS} not found. Run bench_l2_cache_sparse.py first.")
        return

    with open(_L2_RESULTS) as f:
        results = json.load(f)

    def get_tflops(pass_id, method_id):
        pd = results.get(pass_id, {}).get(method_id, {})
        if not pd:
            return [None] * len(TOPK_VALS_FULL)
        vals = []
        for tk in TOPK_VALS_FULL:
            if tk in pd.get("topk", []):
                idx = pd["topk"].index(tk)
                vals.append(pd["tflops"][idx])
            else:
                vals.append(None)
        return vals

    x = np.arange(len(TOPK_VALS_FULL))
    x_labels = [f"{t // 1024}K" for t in TOPK_VALS_FULL]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    # ─── Left: TFLOPS curves ───
    lines = [
        ("bwd_loopq", "ia_solid", "LoopQ IndexSparse (S=32K)", "#E74C3C", "-", "D"),
        ("bwd_loopk", "ia_solid", "LoopK IndexSparse (S=32K)", "#2E86C1", "-", "s"),
        ("bwd_loopq", "d1b_solid", "LoopQ Dense-1B (S=32K)", "#E74C3C", "--", "^"),
        ("bwd_loopk", "d1b_solid", "LoopK Dense-1B (S=32K)", "#2E86C1", "--", "v"),
    ]

    for pid, mid, label, color, ls, marker in lines:
        vals = get_tflops(pid, mid)
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            continue
        xi, yi = zip(*valid)
        ax1.plot(
            xi,
            yi,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            label=label,
            alpha=0.9,
        )

    # Drop annotation
    loopq_ia = get_tflops("bwd_loopq", "ia_solid")
    if loopq_ia[0] and loopq_ia[1]:
        drop_pct = (1 - loopq_ia[1] / loopq_ia[0]) * 100
        ax1.annotate(
            f"L2 thrash\n-{drop_pct:.0f}%",
            xy=(1, loopq_ia[1]),
            xytext=(1.5, loopq_ia[1] + 80),
            fontsize=9,
            ha="center",
            color="#E74C3C",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5),
        )

    ax1.set_title(
        "BWD TFLOPS: LoopQ vs LoopK (S=32K fixed, topk varies)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("topk")
    ax1.set_ylabel("TFLOPS")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0, top=600)

    # ─── Right: L2 working-set analysis ───
    S_FULL = 32768
    L2_CAPACITY_MB = 50

    topk_arr = np.array(TOPK_VALS_FULL, dtype=float)

    # LoopK inner WS: topk_blocks * KBS * HD * 2B (K tile) per outer-Q-tile
    loopk_ws_mb = topk_arr * HD * 2 / 1e6

    # LoopQ inner WS: inv_topk (approx S when topk < S) * HD * 2B * 2 (Q + dO)
    # inv_topk per K-tile ~ S * topk / S = topk for uniform random mask
    # But worst case: each K-block is referenced by ALL Q-rows -> inv_topk ~ S
    loopq_ws_mb = np.full_like(topk_arr, S_FULL * HD * 2 * 2 / 1e6)

    ax2.semilogy(
        x,
        loopq_ws_mb,
        color="#E74C3C",
        marker="D",
        markersize=8,
        linewidth=2.5,
        label="LoopQ inner WS (Q+dO via inv_indices)",
    )
    ax2.semilogy(
        x,
        loopk_ws_mb,
        color="#2E86C1",
        marker="s",
        markersize=8,
        linewidth=2.5,
        label="LoopK inner WS (K tiles)",
    )
    ax2.axhline(
        y=L2_CAPACITY_MB,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"H100 L2 capacity ({L2_CAPACITY_MB}MB)",
    )

    ax2.set_title(
        "L2 Working Set per Outer Tile",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("topk")
    ax2.set_ylabel("Working Set (MB, log scale)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    ax2.grid(alpha=0.3)

    # Annotation for the key insight
    ax2.text(
        0.5,
        0.15,
        "LoopQ: inv_indices scatter → WS = S × 512B ≈ 16MB (always > L2)\n"
        "LoopK: sorted K-blocks → WS = topk × 256B (fits L2 when topk ≤ 16K)",
        transform=ax2.transAxes,
        fontsize=8,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        fontfamily="monospace",
    )

    fig.suptitle(
        "Experiment 2: Why LoopQ Drops but LoopK Stays Stable\n"
        "(nhq=128 nhk=1 hd=128 kbs=128 bf16 H100)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "exp2_l2_divergence.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[{_ts()}] Exp2 plot → {out_path}")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Sparse/Dense phased TFLOPS difference analysis"
    )
    parser.add_argument("--bench1", action="store_true", help="Run exp1 benchmark")
    parser.add_argument("--plot1", action="store_true", help="Generate exp1 plot")
    parser.add_argument("--plot2", action="store_true", help="Generate exp2 plot")
    parser.add_argument("--all", action="store_true", help="Run bench1 + plot1 + plot2")
    args = parser.parse_args()

    if args.all:
        args.bench1 = args.plot1 = args.plot2 = True

    if not any([args.bench1, args.plot1, args.plot2]):
        parser.error("Specify --bench1, --plot1, --plot2, or --all")

    if args.bench1:
        print(f"[{_ts()}] === Experiment 1: Bench (S=topk parity) ===", flush=True)
        run_bench1()
    if args.plot1:
        print(f"[{_ts()}] === Experiment 1: Plot ===", flush=True)
        run_plot1()
    if args.plot2:
        print(f"[{_ts()}] === Experiment 2: Plot (L2 divergence) ===", flush=True)
        run_plot2()

    print(f"[{_ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
