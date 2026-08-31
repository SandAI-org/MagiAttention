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

"""Phase 9: sorted vs shuffled — Impact of index ordering on index-sparse performance.

Measures FWD and BWD TFLOPS with ascending-sorted indices versus randomly-shuffled
indices under the video-production scenario (shared with Phase 6 & 8).

Production config: 1080p, qhead=32, kvhead=8, hd=128.
32-GPU training: per-rank NHQ=128, NHK=1, PackGQA.
Per-rank: qseqlen = kvseqlen/64, topk = kvseqlen/8.

Key finding: BWD is up to 53% slower with unsorted indices at large kvseqlen due to
L2 cache thrashing in the LoopK inner loop's scatter KV loads.
"""

import gc
import os

import numpy as np
from bench_sparse_analysis._common import (
    HD,
    NHK,
    NHQ,
    PLOT_DPI_SAVE,
    VIDEO_SCENARIOS,
    _bench_kernel,
    _calc_flops,
    _load_results,
    _out_dir,
    _save_results,
    _set_gpu,
    _ts,
)

PASSES = ["fwd", "bwd"]
ORDERS = ["sorted", "shuffled"]


def _build_indices(qseqlen, kvseqlen, topk, device, sorted_order: bool):
    """Build index_sparse_indices (qseqlen, NHK, topk) int32.

    Uses argsort to select topk random KV positions per Q-token.
    When sorted_order=True, the selected indices are sorted ascending (better L2 locality).
    When sorted_order=False, the argsort order is kept (effectively random permutation).
    """
    import torch

    gen = torch.Generator(device="cpu").manual_seed(42)
    rand_vals = torch.rand(qseqlen, kvseqlen, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :topk]
    if sorted_order:
        perms = perms.sort(dim=1).values
    return (
        perms.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(dtype=torch.int32, device=device)
        .contiguous()
    )


def _make_run_fn_fwd(q, k, v, indices, flex_flash_attn_func):
    def run_fn():
        flex_flash_attn_func(
            q,
            k,
            v,
            index_sparse_indices=indices,
            q_block_size=1,
            sparse_k_block_size=1,
            pack_gqa=True,
            index_sparse=True,
        )

    return run_fn


def _make_run_fn_bwd(q_g, k_g, v_g, indices, flex_flash_attn_func):
    def run_fn():
        out, _ = flex_flash_attn_func(
            q_g,
            k_g,
            v_g,
            index_sparse_indices=indices,
            q_block_size=1,
            sparse_k_block_size=1,
            pack_gqa=True,
            index_sparse=True,
        )
        out.sum().backward()
        q_g.grad = None
        k_g.grad = None
        v_g.grad = None

    return run_fn


def _phase9_bench(force=False, max_kvseqlen=None):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "9-sorted-vs-shuffled"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    scenarios = VIDEO_SCENARIOS
    if max_kvseqlen is not None:
        scenarios = [(kv, q, t) for kv, q, t in VIDEO_SCENARIOS if kv <= max_kvseqlen]

    print(f"[{_ts()}] Phase 9: Sorted vs Shuffled (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, kbs=1, PackGQA, bf16",
        flush=True,
    )
    print(
        f"  qseqlen=kvseqlen/64, topk=kvseqlen/8 (video-production)"
        f"{f', max_kvseqlen={max_kvseqlen // 1024}k' if max_kvseqlen else ''}\n",
        flush=True,
    )

    for kvseqlen, qseqlen, topk in scenarios:
        q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
        k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
        v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)

        for order in ORDERS:
            indices = _build_indices(
                qseqlen, kvseqlen, topk, device, sorted_order=(order == "sorted")
            )

            for direction in PASSES:
                key = f"{order}/{direction}/kv{kvseqlen // 1024}k"
                if not force and key in results and results[key].get("tflops"):
                    print(f"  [skip] {key}", flush=True)
                    continue

                is_bwd = direction == "bwd"
                flops = _calc_flops(qseqlen, topk, is_bwd)

                if is_bwd:
                    q_g = q.detach().requires_grad_(True)
                    k_g = k.detach().requires_grad_(True)
                    v_g = v.detach().requires_grad_(True)
                    run_fn = _make_run_fn_bwd(
                        q_g, k_g, v_g, indices, flex_flash_attn_func
                    )
                else:
                    run_fn = _make_run_fn_fwd(q, k, v, indices, flex_flash_attn_func)

                try:
                    tflops, ms = _bench_kernel(run_fn, flops, device)
                    results[key] = {
                        "tflops": round(tflops, 2),
                        "ms": round(ms, 4),
                        "kvseqlen": kvseqlen,
                        "qseqlen": qseqlen,
                        "topk": topk,
                        "order": order,
                        "direction": direction,
                    }
                    print(
                        f"  {key:40s} -> {tflops:7.1f} TFLOPS  ({ms:.3f} ms)",
                        flush=True,
                    )
                except Exception as e:
                    results[key] = {"tflops": None, "ms": None, "error": str(e)}
                    print(f"  {key:40s} -> ERROR: {e}", flush=True)

                if is_bwd:
                    del q_g, k_g, v_g

            del indices

        del q, k, v
        gc.collect()
        torch.cuda.empty_cache()
        _save_results(phase, results)

    return results


def _phase9_plot(results=None):
    """Generate comparison plot: sorted vs shuffled across kvseqlen."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = "9-sorted-vs-shuffled"
    if results is None:
        results = _load_results(phase)
    if not results:
        print("  [WARN] No results to plot.")
        return None

    kv_seqlens = sorted(
        set(
            v["kvseqlen"]
            for v in results.values()
            if isinstance(v, dict) and "kvseqlen" in v
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Index Sparse: Sorted vs Shuffled Indices\n"
        f"NHQ={NHQ}, NHK={NHK}, HD={HD}, bf16, PackGQA, kbs=1\n"
        "qseqlen=kvseqlen/64, topk=kvseqlen/8 (video-production)",
        fontsize=12,
    )

    for ax_idx, direction in enumerate(["fwd", "bwd"]):
        ax = axes[ax_idx]
        sorted_tflops = []
        shuffled_tflops = []
        sorted_ms_vals = []
        shuffled_ms_vals = []
        speedups = []
        valid_kv = []

        for kv in kv_seqlens:
            key_s = f"sorted/{direction}/kv{kv // 1024}k"
            key_u = f"shuffled/{direction}/kv{kv // 1024}k"
            s_data = results.get(key_s, {})
            u_data = results.get(key_u, {})
            s_val = s_data.get("tflops")
            u_val = u_data.get("tflops")
            if s_val and u_val:
                sorted_tflops.append(s_val)
                shuffled_tflops.append(u_val)
                sorted_ms_vals.append(s_data.get("ms", 0))
                shuffled_ms_vals.append(u_data.get("ms", 0))
                speedups.append(s_val / u_val)
                valid_kv.append(kv)

        if not valid_kv:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        x = np.arange(len(valid_kv))
        width = 0.35

        ax.bar(
            x - width / 2,
            sorted_tflops,
            width,
            label="Sorted (ascending)",
            color=(0.29, 0.57, 0.60),
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            shuffled_tflops,
            width,
            label="Shuffled (random)",
            color=(0.77, 0.34, 0.49),
            alpha=0.85,
        )

        for i, (sp, y_s, y_u) in enumerate(
            zip(speedups, sorted_tflops, shuffled_tflops)
        ):
            y_max = max(y_s, y_u)
            if sp > 1.05:
                ax.annotate(
                    f"+{(sp - 1) * 100:.0f}%",
                    xy=(x[i], y_max + 5),
                    ha="center",
                    fontsize=8,
                    color="green",
                    fontweight="bold",
                )

        ax.set_xlabel("kvseqlen")
        ax.set_ylabel("TFLOPS")
        ax.set_title(f"{'Forward' if direction == 'fwd' else 'Backward'}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{kv // 1024}k" for kv in valid_kv], fontsize=9)
        ax.legend(loc="upper left", fontsize=9)
        ax.set_ylim(0, max(sorted_tflops + shuffled_tflops) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out_dir = _out_dir(phase)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "sorted_vs_shuffled.png")
    plt.savefig(plot_path, dpi=PLOT_DPI_SAVE, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")
    return plot_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9: sorted vs shuffled bench")
    parser.add_argument("--max-kvseqlen", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        _phase9_plot()
    else:
        results = _phase9_bench(force=args.force, max_kvseqlen=args.max_kvseqlen)
        _phase9_plot(results)


if __name__ == "__main__":
    main()
