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

"""kbs=1 (IndexSparse) vs kbs=128 (IndexSparse/BlockSparse): BWD LoopK TFLOPS.

Purpose: Quantify the overhead of cp.async scatter (kbs=1) vs TMA (kbs=128)
in BWD LoopK, which is the only valid BWD direction for IndexSparse kbs=1.

Benchmark configs:
  - Dense BWD LoopK (q_ranges baseline, pack_gqa=False)
  - IndexSparse kbs=128 BWD LoopK (TMA 2D contiguous tiles)
  - IndexSparse kbs=1 BWD LoopK (cp.async per-row scatter)

All with: S=32K, nhq=128, nhk=1, hd=128, topk variable.

Usage:
  python exps/attn/sparse/bench_kbs1_vs_kbs128.py --bench
  python exps/attn/sparse/bench_kbs1_vs_kbs128.py --plot
"""

import argparse
import gc
import json
import os
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "kbs1_vs_kbs128")

NHQ, NHK, HD = 128, 1, 128
S = 32768
TOPK_VALS = [32768, 16384, 8192, 4096, 2048]
WARMUP, REPEAT = 5, 15


def find_free_gpu():
    """Find GPU with lowest memory usage."""
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, mem = line.split(",")
        gpus.append((int(mem.strip()), int(idx.strip())))
    gpus.sort()
    return gpus[0][1]


def run_benchmark():
    import torch

    gpu_id = find_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
    print(f"[{datetime.now()}] Using GPU {gpu_id}")

    torch.cuda.init()
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

    os.makedirs(_OUT_DIR, exist_ok=True)
    results = {}
    device = "cuda"

    def calc_bwd_tflops(s, topk, time_ms):
        flops_bwd = 2 * 2 * s * topk * NHQ * HD * 2
        return flops_bwd / (time_ms * 1e-3) / 1e12

    def bench_one(run_fn):
        for _ in range(WARMUP):
            run_fn()
            torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]

        for i in range(REPEAT):
            start_events[i].record()
            run_fn()
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return sorted(times)[REPEAT // 2]

    for topk in TOPK_VALS:
        print(f"\n[{datetime.now()}] topk={topk}")
        topk_blocks = topk // 128
        n_kblocks = S // 128

        # --- Dense BWD LoopK (baseline) ---
        torch.manual_seed(42)
        q = torch.randn(
            S, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        k = torch.randn(
            topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        v = torch.randn(
            topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
        atm = torch.zeros(1, dtype=torch.int32, device=device)

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=atm,
            pack_gqa=False,
            swap_bwd_qk_loop=True,
        )
        do = torch.randn_like(out)

        def run_dense():
            out.backward(do, retain_graph=True)

        ms = bench_one(run_dense)
        tf = calc_bwd_tflops(S, topk, ms)
        results.setdefault("dense_loopk", {"topk": [], "tflops": [], "ms": []})
        results["dense_loopk"]["topk"].append(topk)
        results["dense_loopk"]["tflops"].append(round(tf, 1))
        results["dense_loopk"]["ms"].append(round(ms, 3))
        print(f"  Dense LoopK (S={S}, K={topk}): {ms:.2f}ms, {tf:.1f} TFLOPS")
        del q, k, v, out, do
        gc.collect()
        torch.cuda.empty_cache()

        # --- IndexSparse kbs=128 BWD LoopK (TMA) ---
        torch.manual_seed(42)
        q = torch.randn(
            S, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        k = torch.randn(
            S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        v = torch.randn(
            S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )

        if topk_blocks >= n_kblocks:
            idx128 = torch.arange(n_kblocks, dtype=torch.int32, device=device)
        else:
            gen = torch.Generator().manual_seed(42)
            rand_vals = torch.rand(S, n_kblocks, generator=gen)
            idx128 = rand_vals.argsort(dim=1)[:, :topk_blocks].sort(dim=1).values
            idx128 = idx128.to(dtype=torch.int32, device=device)
        idx128 = (
            idx128.view(S, 1, topk_blocks)
            if idx128.dim() == 2
            else idx128.unsqueeze(0).unsqueeze(0)
        )
        idx128 = idx128.expand(S, NHK, -1).contiguous()

        out, _ = flex_flash_attn_func(
            q,
            k,
            v,
            index_sparse_indices=idx128,
            k_block_size=128,
            index_sparse=True,
            pack_gqa=True,
            swap_bwd_qk_loop=True,
        )
        do = torch.randn_like(out)

        def run_is128():
            out.backward(do, retain_graph=True)

        ms = bench_one(run_is128)
        tf = calc_bwd_tflops(S, topk, ms)
        results.setdefault("is128_loopk", {"topk": [], "tflops": [], "ms": []})
        results["is128_loopk"]["topk"].append(topk)
        results["is128_loopk"]["tflops"].append(round(tf, 1))
        results["is128_loopk"]["ms"].append(round(ms, 3))
        print(f"  IndexSparse kbs=128 LoopK: {ms:.2f}ms, {tf:.1f} TFLOPS")
        del q, k, v, out, do, idx128
        gc.collect()
        torch.cuda.empty_cache()

        # --- IndexSparse kbs=1 BWD LoopK (cp.async scatter) ---
        torch.manual_seed(42)
        q = torch.randn(
            S, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        k = torch.randn(
            S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )
        v = torch.randn(
            S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
        )

        if topk >= S:
            idx1 = torch.arange(S, dtype=torch.int32, device=device)
            idx1 = idx1.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
        else:
            gen = torch.Generator().manual_seed(42)
            rand_vals = torch.rand(S, S, generator=gen)
            idx1 = rand_vals.argsort(dim=1)[:, :topk].sort(dim=1).values
            idx1 = (
                idx1.unsqueeze(1)
                .expand(-1, NHK, -1)
                .to(dtype=torch.int32, device=device)
                .contiguous()
            )

        try:
            out, _ = flex_flash_attn_func(
                q,
                k,
                v,
                index_sparse_indices=idx1,
                k_block_size=1,
                index_sparse=True,
                pack_gqa=True,
                swap_bwd_qk_loop=True,
            )
            do = torch.randn_like(out)

            def run_is1():
                out.backward(do, retain_graph=True)

            ms = bench_one(run_is1)
            tf = calc_bwd_tflops(S, topk, ms)
            results.setdefault("is1_loopk", {"topk": [], "tflops": [], "ms": []})
            results["is1_loopk"]["topk"].append(topk)
            results["is1_loopk"]["tflops"].append(round(tf, 1))
            results["is1_loopk"]["ms"].append(round(ms, 3))
            print(f"  IndexSparse kbs=1 LoopK: {ms:.2f}ms, {tf:.1f} TFLOPS")
        except Exception as e:
            print(f"  IndexSparse kbs=1 LoopK: FAILED — {e}")
        finally:
            del q, k, v, idx1
            gc.collect()
            torch.cuda.empty_cache()

    out_path = os.path.join(_OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{datetime.now()}] Results saved to {out_path}")


def run_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    src = os.path.join(_OUT_DIR, "results.json")
    if not os.path.exists(src):
        print(f"ERROR: {src} not found. Run --bench first.")
        return

    with open(src) as f:
        results = json.load(f)

    x = np.arange(len(TOPK_VALS))
    x_labels = [f"{t // 1024}K" for t in TOPK_VALS]

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=150)

    configs = [
        ("dense_loopk", "Dense LoopK (baseline)", "#888888", "-", "o"),
        ("is128_loopk", "IndexSparse kbs=128 LoopK (TMA)", "#2E86C1", "-", "s"),
        ("is1_loopk", "IndexSparse kbs=1 LoopK (cp.async)", "#E74C3C", "-", "D"),
    ]

    for key, label, color, ls, marker in configs:
        data = results.get(key, {})
        if not data:
            continue
        topks = data["topk"]
        tflops = data["tflops"]
        xi = [TOPK_VALS.index(t) for t in topks if t in TOPK_VALS]
        yi = [tflops[topks.index(t)] for t in topks if t in TOPK_VALS]
        ax.plot(
            xi,
            yi,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            label=label,
        )

    ax.set_title(
        "BWD LoopK: kbs=1 (cp.async scatter) vs kbs=128 (TMA)\n"
        f"(S={S}, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16, H100)",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("topk (K tokens selected)", fontsize=10)
    ax.set_ylabel("TFLOPS", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(_OUT_DIR, "kbs1_vs_kbs128.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Plot → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument(
        "--plot", action="store_true", help="Generate plot from results"
    )
    args = parser.parse_args()

    if args.bench:
        run_benchmark()
    elif args.plot:
        run_plot()
    else:
        parser.error("Specify --bench or --plot")


if __name__ == "__main__":
    main()
