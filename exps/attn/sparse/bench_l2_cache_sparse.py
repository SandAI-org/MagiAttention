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

"""L2 cache / sparse-framework overhead benchmark.

Measures TFLOPS across 4 attention methods as topk varies (block-sparse MQA):
  Dense-1B  : Traditional dense, pack_gqa=False, single batch (baseline ceiling)
  Dense     : Same random IA mask via SL q_ranges/k_ranges (proves SL zero overhead)
  IndexSparse : Random K blocks via IA kernel path
  BlockSparse: Same random IA mask via SL q_ranges/k_ranges

Each method has two variants:
  solid  (S=S_FULL): fixed full seqlen, topk varies -> shows L2 cache + CTA effects
  dashed (S=topk) : seqlen = topk -> shows "ideal" small-problem performance

Passes: FWD, BWD LoopQ, BWD LoopK.

Config: nhq=128, nhk=1 (MQA), hd=128, k_block_size=128, bf16.

Usage (subcommands):
  python bench_l2_cache_sparse.py precompile
  python bench_l2_cache_sparse.py data [--force] [--rerun "bwd_loopq/d1b_solid"]
  python bench_l2_cache_sparse.py plot-bars
"""

import argparse
import datetime
import gc
import json
import os
import time

# ── Config ─────────────────────────────────────────────────────
NHQ, NHK, HD, KBS = 128, 1, 128, 128
WARMUP, ITERS = 8, 20
S_FULL = 32768
TOPK_VALS = [32768, 16384, 8192, 4096, 2048]
YLIM_MAX = 700

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_SCRIPT_DIR, "outs", "l2_cache_sparse")

OUT_DIR = os.environ.get("L2_BENCH_OUT", _DEFAULT_OUT)
RESULTS_FILE = os.path.join(OUT_DIR, "results.json")


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


# ── GPU selection ──────────────────────────────────────────────
def _find_free_gpu():
    """Return logical CUDA index with lowest memory usage."""
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


def _set_gpu():
    import torch

    gpu = _find_free_gpu()
    torch.cuda.set_device(gpu)
    return gpu


# ── JSON persistence (incremental, atomic) ────────────────────
def _load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def _save_results(results):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.getsize(tmp) > 0:
        os.replace(tmp, RESULTS_FILE)


def _has_entry(results, pass_id, method_id, topk):
    pd = results.get(pass_id, {}).get(method_id, {})
    if not pd or "topk" not in pd:
        return False
    try:
        idx = pd["topk"].index(topk)
        return pd["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, pass_id, method_id, topk, tflops, ms):
    if pass_id not in results:
        results[pass_id] = {}
    if method_id not in results[pass_id]:
        results[pass_id][method_id] = {"topk": [], "tflops": [], "ms": []}
    d = results[pass_id][method_id]
    if topk in d["topk"]:
        idx = d["topk"].index(topk)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["topk"].append(topk)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


def _parse_rerun(s):
    if not s:
        return None
    entries = set()
    for item in s.split(","):
        parts = item.strip().split("/")
        if len(parts) == 3:
            entries.add((parts[0], parts[1], int(parts[2])))
        elif len(parts) == 2:
            for tk in TOPK_VALS:
                entries.add((parts[0], parts[1], tk))
    return entries or None


# ═══════════════════════════════════════════════════════════════
#  Precompile
# ═══════════════════════════════════════════════════════════════
def run_precompile():
    """Run precompileall to warm JIT cache."""
    import subprocess
    import sys

    print(f"[{_ts()}] Running precompileall...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "magi_attention.precompileall"],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"[{_ts()}] Precompile done.", flush=True)
    else:
        print(f"[{_ts()}] Precompile failed (rc={result.returncode})", flush=True)


# ═══════════════════════════════════════════════════════════════
#  Benchmark (data)
# ═══════════════════════════════════════════════════════════════
def run_bench(force=False, rerun_filter=None):
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    DTYPE = torch.bfloat16
    results = _load_results()
    n_cached = sum(
        len(v.get("topk", [])) for pd in results.values() for v in pd.values()
    )
    print(f"[{_ts()}] Loaded {RESULTS_FILE}: {n_cached} entries", flush=True)

    # ── helpers ────────────────────────────────────────────────
    def bench_kernel(run_fn, flops, device):
        for _ in range(WARMUP):
            run_fn()
        torch.cuda.synchronize(device)
        l2 = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            l2.zero_()
            starts[i].record()
            run_fn()
            ends[i].record()
        torch.cuda.synchronize(device)
        del l2
        times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
        ms = times[len(times) // 2]
        return flops / ms * 1e-9, ms

    def make_tensors(S, device, grad=False):
        q = torch.randn(S, NHQ, HD, dtype=DTYPE, device=device)
        k = torch.randn(S, NHK, HD, dtype=DTYPE, device=device)
        v = torch.randn(S, NHK, HD, dtype=DTYPE, device=device)
        if grad:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
        return q, k, v

    def calc_flops(S, topk, is_bwd):
        fwd = 4 * S * topk * NHQ * HD
        return fwd * 2.5 if is_bwd else fwd

    # ── index builders ─────────────────────────────────────────
    def build_random_indices(S, topk, device):
        n_total, n_topk = S // KBS, topk // KBS
        if n_topk >= n_total:
            idx = torch.arange(n_total, dtype=torch.int32, device=device)
            return idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
        gen = torch.Generator().manual_seed(42)
        rand_vals = torch.rand(S, n_total, generator=gen)
        perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
        return (
            perms.unsqueeze(1)
            .expand(-1, NHK, -1)
            .to(dtype=torch.int32, device=device)
            .contiguous()
        )

    def indices_to_sl_kwargs(indices, S, pass_type):
        ia_3d = indices.permute(1, 0, 2).contiguous()
        q_ranges, k_ranges = generate_ranges_from_topk_indices(
            ia_3d, block_m=1, block_n=KBS, num_k_blocks=S // KBS
        )
        atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=indices.device)
        kw = dict(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=atm,
            block_sparse=True,
            auto_range_merge=True,
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return kw

    def _bench_ffa(S, topk, pass_type, kw, device):
        """Common bench: warmup + timed iterations for flex_flash_attn_func."""
        is_bwd = pass_type != "fwd"
        q, k, v = make_tensors(S, device, grad=is_bwd)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        flops = calc_flops(S, topk, is_bwd)

        if not is_bwd:

            def run_fn():
                flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

        else:
            do = torch.randn_like(o)

            def run_fn():  # noqa: F811
                o.backward(do, retain_graph=True)  # noqa: F821

        tf, ms = bench_kernel(run_fn, flops, device)
        del q, k, v, o
        gc.collect()
        torch.cuda.empty_cache()
        return tf, ms

    def _run_sl_path(S, topk, pass_type, indices):
        gpu = _set_gpu()
        device = f"cuda:{gpu}"
        indices = indices.to(device=device)
        kw = indices_to_sl_kwargs(indices, S, pass_type)
        tf, ms = _bench_ffa(S, topk, pass_type, kw, device)
        del indices
        return round(tf, 1), round(ms, 3), gpu

    # ── method runners ─────────────────────────────────────────
    def run_dense1b(S, topk, pass_type):
        """Dense-1B: pack_gqa=False, single batch [0:S]x[0:topk]."""
        gpu = _set_gpu()
        device = f"cuda:{gpu}"
        q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
        atm = torch.zeros(1, dtype=torch.int32, device=device)
        kw = dict(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=atm,
            pack_gqa=False,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        tf, ms = _bench_ffa(S, topk, pass_type, kw, device)
        return round(tf, 1), round(ms, 3), gpu

    def run_dense(S, topk, pass_type):
        """Dense via SL path with same random IA mask."""
        indices = build_random_indices(S, topk, "cpu")
        return _run_sl_path(S, topk, pass_type, indices)

    def run_ia(S, topk, pass_type):
        """IndexSparse with random K blocks."""
        gpu = _set_gpu()
        device = f"cuda:{gpu}"
        indices = build_random_indices(S, topk, device)
        kw = dict(
            index_sparse_indices=indices,
            k_block_size=KBS,
            index_sparse=True,
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        tf, ms = _bench_ffa(S, topk, pass_type, kw, device)
        del indices
        return round(tf, 1), round(ms, 3), gpu

    def run_sl(S, topk, pass_type):
        """BlockSparse with same random IA mask, converted to ranges."""
        indices = build_random_indices(S, topk, "cpu")
        return _run_sl_path(S, topk, pass_type, indices)

    METHODS = [
        ("d1b_solid", "D1B(S=32K)", lambda tk, pt: run_dense1b(S_FULL, tk, pt)),
        ("d1b_dashed", "D1B(S=topk)", lambda tk, pt: run_dense1b(tk, tk, pt)),
        ("dense_solid", "Dense(S=32K)", lambda tk, pt: run_dense(S_FULL, tk, pt)),
        ("dense_dashed", "Dense(S=topk)", lambda tk, pt: run_dense(tk, tk, pt)),
        ("ia_solid", "IA(S=32K)", lambda tk, pt: run_ia(S_FULL, tk, pt)),
        ("ia_dashed", "IA(S=topk)", lambda tk, pt: run_ia(tk, tk, pt)),
        ("sl_solid", "SL(S=32K)", lambda tk, pt: run_sl(S_FULL, tk, pt)),
        ("sl_dashed", "SL(S=topk)", lambda tk, pt: run_sl(tk, tk, pt)),
    ]
    PASSES = [
        ("fwd", "FWD"),
        ("bwd_loopq", "BWD LoopQ"),
        ("bwd_loopk", "BWD LoopK"),
    ]

    total = len(PASSES) * len(METHODS) * len(TOPK_VALS)
    done, skipped, ran = 0, 0, 0

    for pass_id, pass_name in PASSES:
        print(f"\n{'=' * 60}\n[{_ts()}] === {pass_name} ===", flush=True)
        for method_id, method_name, method_fn in METHODS:
            print(f"\n  [{_ts()}] {method_name}:", flush=True)
            for topk in TOPK_VALS:
                done += 1
                pct = done * 100 // total

                should_run = True
                if rerun_filter is not None:
                    should_run = (pass_id, method_id, topk) in rerun_filter
                elif not force and _has_entry(results, pass_id, method_id, topk):
                    should_run = False

                if not should_run:
                    existing = results.get(pass_id, {}).get(method_id, {})
                    if existing and topk in existing.get("topk", []):
                        idx = existing["topk"].index(topk)
                        etf = existing["tflops"][idx]
                        print(
                            f"    [{pct:>3d}%] topk={topk:>5d}: {etf:>7.1f} T  (cached)",
                            flush=True,
                        )
                    else:
                        print(f"    [{pct:>3d}%] topk={topk:>5d}: SKIP", flush=True)
                    skipped += 1
                    continue

                try:
                    t0 = time.time()
                    tf, ms, gpu = method_fn(topk, pass_id)
                    elapsed = time.time() - t0
                    _set_entry(results, pass_id, method_id, topk, tf, ms)
                    ran += 1
                    print(
                        f"    [{pct:>3d}%] topk={topk:>5d}: {tf:>7.1f} T  "
                        f"({ms:.3f}ms, {elapsed:.0f}s) [gpu{gpu}]",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, pass_id, method_id, topk, None, None)
                    ran += 1
                    print(f"    [{pct:>3d}%] topk={topk:>5d}: FAIL - {e}", flush=True)

            _save_results(results)

    print(
        f"\n[{_ts()}] Bench DONE: {ran} ran, {skipped} cached -> {RESULTS_FILE}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════
#  Plot: Bar chart (original --plot)
# ═══════════════════════════════════════════════════════════════
def run_plot_bars():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import seaborn as sns

        sns.set_theme(
            style="whitegrid",
            context="notebook",
            rc={
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.labelsize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            },
        )
    except ImportError:
        pass

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    COL_D1B = (0.580, 0.580, 0.580)
    COL_DENSE = (0.220, 0.373, 0.706)
    COL_IA = (0.769, 0.337, 0.494)
    COL_SL = (0.290, 0.569, 0.604)

    ROWS = [
        {
            "suffix": "S=32K",
            "methods": [
                ("d1b_solid", "Dense-1B", COL_D1B, ""),
                ("dense_solid", "Dense", COL_DENSE, ""),
                ("ia_solid", "IndexSparse", COL_IA, ""),
                ("sl_solid", "BlockSparse", COL_SL, ""),
            ],
        },
        {
            "suffix": "S=topk",
            "methods": [
                ("d1b_dashed", "Dense-1B", COL_D1B, "//"),
                ("dense_dashed", "Dense", COL_DENSE, "//"),
                ("ia_dashed", "IndexSparse", COL_IA, "//"),
                ("sl_dashed", "BlockSparse", COL_SL, "//"),
            ],
        },
    ]
    PASSES = [
        ("fwd", "FWD"),
        ("bwd_loopq", "BWD LoopQ"),
        ("bwd_loopk", "BWD LoopK"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(30, 14), dpi=150)

    for row_idx, row_cfg in enumerate(ROWS):
        for col_idx, (pid, pname) in enumerate(PASSES):
            ax = axes[row_idx, col_idx]
            pd_ = results.get(pid, {})
            methods = row_cfg["methods"]
            present = [
                (mid, lbl, col, h)
                for mid, lbl, col, h in methods
                if mid in pd_ and pd_[mid].get("tflops")
            ]
            nm = len(present)
            x = np.arange(len(TOPK_VALS))
            bw = min(0.80 / max(nm, 1), 0.20)

            for i, (mid, lbl, col, hatch) in enumerate(present):
                d = pd_[mid]
                vals = []
                for tk in TOPK_VALS:
                    if tk in d["topk"]:
                        idx = d["topk"].index(tk)
                        v = d["tflops"][idx] if d["tflops"][idx] else 0
                    else:
                        v = 0
                    vals.append(v)
                off = (i - nm / 2 + 0.5) * bw
                bars = ax.bar(
                    x + off,
                    vals,
                    width=bw,
                    label=lbl,
                    color=col,
                    hatch=hatch,
                    edgecolor="white" if not hatch else col,
                    linewidth=0.5,
                    alpha=0.80,
                    zorder=2,
                )
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 2,
                            f"{v:.0f}",
                            ha="center",
                            va="bottom",
                            fontsize=6.5,
                            fontweight="bold",
                            zorder=4,
                        )

            ax.set_title(
                f"{pname}  ({row_cfg['suffix']})",
                fontsize=13,
                fontweight="bold",
                pad=8,
            )
            ax.set_xlabel("topk")
            ax.set_ylabel("TFLOPS")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{t // 1024}K" for t in TOPK_VALS])
            ax.set_ylim(0, YLIM_MAX)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
            ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle(
        f"Attention TFLOPS vs topk   (nhq={NHQ}  nhk={NHK}  hd={HD}  bf16  H100)\n"
        f"D1B=dense pack_gqa=False (baseline)  |  Dense=IA random mask "
        f"(SL path, pack_gqa=True)  |  IA/SL=random K",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bars.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[{_ts()}] Plot bars -> {out_path}", flush=True)


# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="L2 cache / sparse-framework overhead benchmark"
    )
    sub = parser.add_subparsers(dest="cmd")

    # precompile
    sub.add_parser("precompile", help="Run precompileall to warm JIT cache")

    # data
    p_data = sub.add_parser("data", help="Run benchmarks -> results.json")
    p_data.add_argument(
        "--force", action="store_true", help="Re-run all (ignore cache)"
    )
    p_data.add_argument(
        "--rerun",
        type=str,
        default=None,
        help="Re-run subset: 'pass/method' or 'pass/method/topk', comma-separated",
    )
    p_data.add_argument("--out-dir", default=None, help="Override output directory")

    # plot-bars
    p_bars = sub.add_parser("plot-bars", help="Generate bar chart")
    p_bars.add_argument("--out-dir", default=None)

    # Legacy compat: --bench / --plot
    parser.add_argument("--bench", action="store_true", help="(legacy) same as 'data'")
    parser.add_argument(
        "--plot", action="store_true", help="(legacy) same as 'plot-bars'"
    )
    parser.add_argument("--force", action="store_true", help="(legacy) force re-run")
    parser.add_argument("--rerun", type=str, default=None, help="(legacy) rerun filter")
    parser.add_argument("--out-dir", default=None, help="(legacy) output dir override")

    args = parser.parse_args()

    global OUT_DIR, RESULTS_FILE
    out_dir_override = getattr(args, "out_dir", None)
    if out_dir_override:
        OUT_DIR = out_dir_override
        RESULTS_FILE = os.path.join(OUT_DIR, "results.json")

    if args.cmd == "precompile":
        run_precompile()
    elif args.cmd == "data":
        rerun_filter = _parse_rerun(args.rerun) if args.rerun else None
        if rerun_filter:
            print(f"[{_ts()}] Re-run filter: {len(rerun_filter)} entries", flush=True)
        print(
            f"[{_ts()}] L2 Cache Benchmark -- "
            f"D1B=pack_gqa=False  Dense/SL=IA random mask  IA=random K",
            flush=True,
        )
        run_bench(force=args.force, rerun_filter=rerun_filter)
    elif args.cmd == "plot-bars":
        run_plot_bars()
    elif args.bench or args.plot:
        if args.bench:
            rerun_filter = _parse_rerun(args.rerun) if args.rerun else None
            run_bench(force=args.force, rerun_filter=rerun_filter)
        if args.plot:
            run_plot_bars()
    else:
        parser.print_help()
        parser.error("Specify a subcommand: precompile, data, plot-bars")

    print(f"[{_ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
