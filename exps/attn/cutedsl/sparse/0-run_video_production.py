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

"""CuTeDSL SM100 sparse benchmark — Video Production scenario.

Mirrors ``exps/attn/sparse/bench_sparse_analysis/phase6_video_production.py``
but drives the CuTeDSL kernel entry points directly (``_flex_flash_attn_fwd`` /
``_flex_flash_attn_bwd`` + ``prepare_index_sparse_tiles``) on Blackwell (SM100).

Production config: 1080p, qhead=32, kvhead=8, hd=128.
32-GPU training: 8 KV-heads distributed -> per-rank nhk=1, nhq=128.
Per-rank: qseqlen = kvseqlen/64, topk = kvseqlen/8.
Post-distribution: NHQ=128, NHK=1, PackGQA, bf16.

Methods compared:
  - Dense (full attention, CuTeDSL SM100 FWD/BWD)
  - IndexSparse (token-level scatter, CuTeDSL SM100 FWD/BWD-LoopK)

Run:
    cd exps/attn/cutedsl/sparse
    PYTHONPATH=../../../.. python 0-run_video_production.py --exp
    PYTHONPATH=../../../.. python 0-run_video_production.py --plot
    PYTHONPATH=../../../.. python 0-run_video_production.py --exp --plot
"""

import argparse
import datetime
import gc
import json
import os
import sys

import torch

# ═══════════════════════════════════════════════════════════════
#  Global Config (same as phase6)
# ═══════════════════════════════════════════════════════════════
NHQ, NHK, HD = 128, 1, 128
N_BLOCK_SIZE = 128
WARMUP, ITERS = 8, 20

# kvseqlen -> (qseqlen, topk)
# qseqlen = kvseqlen / 64, topk = kvseqlen / 8
SCENARIOS = [
    # (kvseqlen, qseqlen, topk)
    (32768, 512, 4096),
    (65536, 1024, 8192),
    (131072, 2048, 16384),
    (262144, 4096, 32768),
    (524288, 8192, 65536),
]

PASSES = ["fwd", "bwd"]
METHODS = ["dense", "index_sparse"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "0-video-production")
_RESULTS_PATH = os.path.join(_OUT_DIR, "results.json")


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════
def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _load_results():
    if os.path.exists(_RESULTS_PATH):
        with open(_RESULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_results(results):
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    tmp = _RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.getsize(tmp) > 0:
        os.replace(tmp, _RESULTS_PATH)


def _has_entry(results, key, kvseqlen):
    d = results.get(key, {})
    if not d or "kvseqlen" not in d:
        return False
    try:
        idx = d["kvseqlen"].index(kvseqlen)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, kvseqlen, tflops, ms):
    if key not in results:
        results[key] = {"kvseqlen": [], "tflops": [], "ms": []}
    d = results[key]
    if kvseqlen in d["kvseqlen"]:
        idx = d["kvseqlen"].index(kvseqlen)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["kvseqlen"].append(kvseqlen)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


def _bench_kernel(run_fn, flops, device):
    """Median-of-N timing with L2 flush between iterations."""
    for _ in range(WARMUP):
        run_fn()
    torch.cuda.synchronize(device)
    l2_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for i in range(ITERS):
        l2_flush.zero_()
        starts[i].record()
        run_fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    del l2_flush
    times = sorted([s.elapsed_time(e) for s, e in zip(starts, ends)])
    ms = times[len(times) // 2]
    return flops / ms * 1e-9, ms


def _calc_flops(qseqlen, topk_or_kvseqlen, is_bwd):
    """Effective sparse TFLOPS formula: FWD = 4 * SQ * SK * NHQ * HD."""
    fwd = 4 * qseqlen * topk_or_kvseqlen * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _fwd_sparse_m_block(seqlen_q, qhpk, pack_gqa):
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


# ═══════════════════════════════════════════════════════════════
#  Experiment
# ═══════════════════════════════════════════════════════════════
def _run_experiment(force=False, max_kvseqlen=None):
    from magi_attention.kernel.cutedsl.ffa_utils import TorchFlexAttnArgs
    from magi_attention.kernel.cutedsl.flex_flash_attn import (
        _flex_flash_attn_bwd,
        _flex_flash_attn_fwd,
    )
    from magi_attention.kernel.cutedsl.sparse_utils import prepare_index_sparse_tiles

    device = "cuda"
    results = _load_results()

    scenarios = SCENARIOS
    if max_kvseqlen is not None:
        scenarios = [(kv, q, t) for kv, q, t in SCENARIOS if kv <= max_kvseqlen]

    print(
        f"[{_ts()}] CuTeDSL SM100 Sparse Bench: Video Production",
        flush=True,
    )
    print(
        f"  NHQ={NHQ}, NHK={NHK}, HD={HD}, PackGQA, bf16, B300\n",
        flush=True,
    )

    qhpk = NHQ // NHK

    for kvseqlen, qseqlen, topk in scenarios:
        print(
            f"  ── kvseqlen={kvseqlen // 1024}k, "
            f"qseqlen={qseqlen}, topk={topk // 1024}k ──",
            flush=True,
        )

        for pass_type in PASSES:
            is_bwd = pass_type == "bwd"

            for method in METHODS:
                key = f"{pass_type}/{method}"

                if not force and _has_entry(results, key, kvseqlen):
                    d = results[key]
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:4s} {method:14s}: " f"{tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    torch.manual_seed(42)

                    if method == "dense":
                        # Dense: Q attends to full topk-length KV (simulates
                        # the "gather topk tokens then dense attention" baseline)
                        B = 1
                        q = torch.randn(
                            B, qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            B, topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            B, topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )

                        flops = _calc_flops(qseqlen, topk, is_bwd)

                        # Warmup + compile
                        out, lse = _flex_flash_attn_fwd(
                            q, k, v, softmax_scale=HD**-0.5, pack_gqa=True
                        )

                        if not is_bwd:

                            def run_fn():
                                _flex_flash_attn_fwd(
                                    q, k, v, softmax_scale=HD**-0.5, pack_gqa=True
                                )

                        else:
                            dO = torch.randn_like(out)

                            def run_fn():
                                _flex_flash_attn_bwd(
                                    q,
                                    k,
                                    v,
                                    out,
                                    lse,
                                    dO,
                                    softmax_scale=HD**-0.5,
                                    pack_gqa=True,
                                )

                    elif method == "index_sparse":
                        B = 1
                        q = torch.randn(
                            B, qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            B, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            B, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )

                        # Build random sorted token indices (B, NHQ, SQ, topk)
                        gen = torch.Generator(device="cpu").manual_seed(42)
                        rand_vals = torch.rand(B, qseqlen, kvseqlen, generator=gen)
                        indices = (
                            rand_vals.argsort(dim=-1)[..., :topk]
                            .sort(dim=-1)
                            .values.int()
                            .to(device=device)
                        )
                        # Expand to (B, NHQ, SQ, topk) — all Q heads share same pattern
                        indices = (
                            indices.unsqueeze(2)
                            .expand(B, qseqlen, NHQ, topk)
                            .permute(0, 2, 1, 3)
                            .contiguous()
                        )

                        flops = _calc_flops(qseqlen, topk, is_bwd)

                        # Prepare FWD tiles
                        fwd_m_block = _fwd_sparse_m_block(qseqlen, qhpk, True)
                        fwd_tiles = prepare_index_sparse_tiles(
                            indices,
                            batch_size=B,
                            seqlen_q=qseqlen,
                            seqlen_k=kvseqlen,
                            num_kv_heads=NHK,
                            num_q_heads=NHQ,
                            m_block_size=fwd_m_block,
                            n_block_size=N_BLOCK_SIZE,
                            pack_gqa=True,
                        )

                        # Warmup FWD
                        out, lse = _flex_flash_attn_fwd(
                            q,
                            k,
                            v,
                            softmax_scale=HD**-0.5,
                            flex_attn_args=TorchFlexAttnArgs(
                                index_sparse_tiles=fwd_tiles
                            ),
                            pack_gqa=True,
                        )

                        if not is_bwd:

                            def run_fn():
                                _flex_flash_attn_fwd(
                                    q,
                                    k,
                                    v,
                                    softmax_scale=HD**-0.5,
                                    flex_attn_args=TorchFlexAttnArgs(
                                        index_sparse_tiles=fwd_tiles
                                    ),
                                    pack_gqa=True,
                                )

                        else:
                            # BWD tiles (LoopK uses 128-row M blocks)
                            bwd_tiles = prepare_index_sparse_tiles(
                                indices,
                                batch_size=B,
                                seqlen_q=qseqlen,
                                seqlen_k=kvseqlen,
                                num_kv_heads=NHK,
                                num_q_heads=NHQ,
                                m_block_size=128,
                                n_block_size=N_BLOCK_SIZE,
                                pack_gqa=True,
                            )
                            dO = torch.randn_like(out)

                            def run_fn():
                                _flex_flash_attn_bwd(
                                    q,
                                    k,
                                    v,
                                    out,
                                    lse,
                                    dO,
                                    softmax_scale=HD**-0.5,
                                    flex_attn_args=TorchFlexAttnArgs(
                                        index_sparse_tiles=bwd_tiles
                                    ),
                                    swap_bwd_qk_loop=True,
                                    pack_gqa=True,
                                )

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                    print(
                        f"    {pass_type:4s} {method:14s}: "
                        f"{tf:>7.1f} T ({ms:.3f}ms)",
                        flush=True,
                    )

                except torch.cuda.OutOfMemoryError:
                    _set_entry(results, key, kvseqlen, None, None)
                    print(
                        f"    {pass_type:4s} {method:14s}: OOM",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, kvseqlen, None, None)
                    print(
                        f"    {pass_type:4s} {method:14s}: FAIL - {e}",
                        flush=True,
                    )
                finally:
                    # Free tensors
                    for name in ["q", "k", "v", "out", "lse", "dO", "indices"]:
                        if name in dir():
                            exec(f"del {name}", {}, locals())
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(results)

    print(f"\n[{_ts()}] Experiment DONE -> {_RESULTS_PATH}", flush=True)
    _print_summary(results)


def _print_summary(results):
    """Print summary table."""
    print("\n  ╔═════════════════════════════════╦════════════════════════════╗")
    print("  ║ kvseqlen (qseq, topk)          ║  fwd_dense  fwd_is  bwd_d  bwd_is ║")
    print("  ╠═════════════════════════════════╬════════════════════════════╣")
    for kvseqlen, qseqlen, topk in SCENARIOS:
        label = f"{kvseqlen // 1024:>3d}k (q={qseqlen:>5d}, top={topk // 1024:>3d}k)"
        vals = []
        for pass_type in PASSES:
            for method in METHODS:
                key = f"{pass_type}/{method}"
                d = results.get(key, {})
                if kvseqlen in d.get("kvseqlen", []):
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    vals.append(f"{tf:>6.0f}" if tf else "  FAIL")
                else:
                    vals.append("     -")
        print(f"  ║ {label:<31s} ║ {'  '.join(vals)} ║")
    print("  ╚═════════════════════════════════╩════════════════════════════╝")


# ═══════════════════════════════════════════════════════════════
#  Plot
# ═══════════════════════════════════════════════════════════════
def _plot():
    """Generate grouped bar chart: Dense vs IndexSparse TFLOPS by kvseqlen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results()
    if not results:
        print(f"ERROR: {_RESULTS_PATH} not found. Run --exp first.")
        return

    os.makedirs(_OUT_DIR, exist_ok=True)

    PLOT_PASSES = [
        ("fwd", "FWD"),
        ("bwd", "BWD (LoopK)"),
    ]
    PLOT_METHODS = [
        ("dense", "Dense (K=topk, PackGQA)", (0.58, 0.58, 0.58)),
        ("index_sparse", "IndexSparse (scatter, PackGQA)", (0.77, 0.34, 0.49)),
    ]

    kvseqlens = [s[0] for s in SCENARIOS]
    x = np.arange(len(kvseqlens))
    bw = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

    for col_idx, (pid, pname) in enumerate(PLOT_PASSES):
        ax = axes[col_idx]
        n_m = len(PLOT_METHODS)
        for i, (mid, lbl, col) in enumerate(PLOT_METHODS):
            key = f"{pid}/{mid}"
            d = results.get(key, {})
            vals = []
            for kv in kvseqlens:
                if kv in d.get("kvseqlen", []):
                    idx = d["kvseqlen"].index(kv)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            off = (i - n_m / 2 + 0.5) * bw
            bars = ax.bar(
                x + off,
                vals,
                width=bw,
                label=lbl,
                color=col,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5,
                        f"{v:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_title(pname, fontsize=14, fontweight="bold")
        ax.set_xlabel("kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)", fontsize=10)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 8192}K)" for kv in kvseqlens],
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "CuTeDSL SM100 IndexSparse — Video Production Scenario\n"
        f"NHQ={NHQ}, NHK={NHK}, HD={HD}, PackGQA, bf16, B300",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(_OUT_DIR, "0_video_production.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot saved -> {path}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CuTeDSL SM100 sparse benchmark — Video Production scenario"
    )
    parser.add_argument("--exp", action="store_true", help="Run benchmark experiment")
    parser.add_argument(
        "--plot", action="store_true", help="Generate plot from results"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run all (ignore cache)"
    )
    parser.add_argument(
        "--max-kvseqlen",
        type=int,
        default=None,
        help="Cap kvseqlen (e.g. 65536 for quick test)",
    )
    args = parser.parse_args()

    if not args.exp and not args.plot:
        parser.print_help()
        sys.exit(1)

    if args.exp:
        _run_experiment(force=args.force, max_kvseqlen=args.max_kvseqlen)
    if args.plot:
        _plot()


if __name__ == "__main__":
    main()
