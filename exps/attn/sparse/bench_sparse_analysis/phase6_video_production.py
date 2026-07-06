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

"""Phase 6: video-production — Realistic video-gen scenario scaling.

Production config: 1080p, qhead=32, kvhead=8, qhpkh=4, qblocksize=16.
32-GPU training: Q distributed across 4 ranks after KV-head split.
Per-rank: qseqlen = kvseqlen/64, topk = kvseqlen/4.

Key question: does small qseqlen cause LoopQ outer-parallelism starvation?
"""

import gc
import os
import subprocess
import sys
import time

from bench_sparse_analysis._common import (
    _bench_kernel,
    _find_free_gpu,
    _has_entry,
    _load_results,
    _out_dir,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)

# ═══════════════════════════════════════════════════════════════
#  Phase 6: Video Production Scenario
# ═══════════════════════════════════════════════════════════════
# Per-rank params (after 32-GPU KV-head split + Q partition)
NHQ_V, NHK_V, HD_V = 32, 8, 128
QHPKH = NHQ_V // NHK_V  # 4
KBS_V = 128
Q_BLOCK_SIZE = 16

# kvseqlen → (qseqlen, topk)
# qseqlen = kvseqlen / Q_BLOCK_SIZE / (32 // NHK_V) = kvseqlen / 64
# topk = kvseqlen / 4
SCENARIOS = [
    # (kvseqlen, qseqlen, topk)
    (32768, 512, 8192),
    (65536, 1024, 16384),
    (131072, 2048, 32768),
    (262144, 4096, 65536),
    (524288, 8192, 131072),
]

PASSES = ["fwd", "bwd_loopk", "bwd_loopq"]
METHODS = ["dense", "block_sparse", "index_sparse"]


def _phase6_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "6-video-production"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(
        f"[{_ts()}] Phase 6: Video Production (gpu{gpu})",
        flush=True,
    )
    print(
        f"  nhq={NHQ_V}, nhk={NHK_V}, hd={HD_V}, kbs={KBS_V}, "
        f"qblksize={Q_BLOCK_SIZE}, PackGQA, bf16\n",
        flush=True,
    )

    for kvseqlen, qseqlen, topk in SCENARIOS:
        print(
            f"  ── kvseqlen={kvseqlen // 1024}k, "
            f"qseqlen={qseqlen}, topk={topk // 1024}k ──",
            flush=True,
        )

        for pass_type in PASSES:
            is_bwd = pass_type != "fwd"
            swap_qk = pass_type == "bwd_loopk"

            for method in METHODS:
                key = f"{pass_type}/{method}"
                if not force and _has_entry(results, key, kvseqlen):
                    d = results[key]
                    idx = d["topk"].index(kvseqlen)
                    print(
                        f"    {pass_type:10s} {method:14s}: "
                        f"{d['tflops'][idx]:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    torch.manual_seed(42)

                    if method == "dense":
                        # Dense: Q attends to all topk positions (no sparse overhead)
                        q = torch.randn(
                            qseqlen,
                            NHQ_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        k = torch.randn(
                            topk,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        v = torch.randn(
                            topk,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        q_ranges = torch.tensor(
                            [[0, qseqlen]], dtype=torch.int32, device=device
                        )
                        k_ranges = torch.tensor(
                            [[0, topk]], dtype=torch.int32, device=device
                        )
                        atm = torch.zeros(1, dtype=torch.int32, device=device)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=True,
                        )
                        flops = 4 * qseqlen * topk * NHQ_V * HD_V

                    elif method == "block_sparse":
                        # BlockSparse: Q has qseqlen tokens, K has kvseqlen,
                        # each Q token selects topk positions from kvseqlen
                        q = torch.randn(
                            qseqlen,
                            NHQ_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        k = torch.randn(
                            kvseqlen,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        v = torch.randn(
                            kvseqlen,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_idx_kbs128_video(
                            qseqlen, kvseqlen, topk, NHK_V, device
                        )
                        q_ranges, k_ranges, atm = _indices_to_ranges_video(
                            indices, qseqlen, kvseqlen, NHK_V, device
                        )
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=True,
                            block_sparse=True,
                            k_block_size=KBS_V,
                        )
                        flops = 4 * qseqlen * topk * NHQ_V * HD_V

                    else:  # index_sparse
                        q = torch.randn(
                            qseqlen,
                            NHQ_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        k = torch.randn(
                            kvseqlen,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        v = torch.randn(
                            kvseqlen,
                            NHK_V,
                            HD_V,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_idx_kbs128_video(
                            qseqlen, kvseqlen, topk, NHK_V, device
                        )
                        kw = dict(
                            index_sparse_indices=indices,
                            pack_gqa=True,
                            k_block_size=KBS_V,
                        )
                        flops = 4 * qseqlen * topk * NHQ_V * HD_V

                    if is_bwd:
                        kw["swap_bwd_qk_loop"] = swap_qk
                        flops = int(flops * 2.5)
                    else:
                        flops = int(flops)

                    t0 = time.time()
                    o, *_ = flex_flash_attn_func(q, k, v, **kw)

                    if is_bwd:
                        do = torch.randn_like(o)

                        def run_fn():
                            o.backward(do, retain_graph=True)

                    else:

                        def run_fn():
                            flex_flash_attn_func(q, k, v, **kw)

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    elapsed = time.time() - t0
                    _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                    print(
                        f"    {pass_type:10s} {method:14s}: "
                        f"{tf:>7.1f} T ({ms:.3f}ms, {elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, kvseqlen, None, None)
                    print(
                        f"    {pass_type:10s} {method:14s}: FAIL - {e}",
                        flush=True,
                    )
                finally:
                    q = k = v = None
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(phase, results)

    print(f"\n[{_ts()}] Phase 6 DONE -> {_results_path(phase)}", flush=True)
    _print_summary(results)


def _build_idx_kbs128_video(qseqlen, kvseqlen, topk, nhk, device):
    """Build block-sparse indices: (qseqlen, nhk, n_topk_blocks) int32."""
    import torch

    n_total_blocks = kvseqlen // KBS_V
    n_topk_blocks = topk // KBS_V

    if n_topk_blocks >= n_total_blocks:
        idx = torch.arange(n_total_blocks, dtype=torch.int32, device=device)
        return idx.unsqueeze(0).unsqueeze(0).expand(qseqlen, nhk, -1).contiguous()

    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(qseqlen, n_total_blocks, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :n_topk_blocks].sort(dim=1).values
    return (
        perms.unsqueeze(1)
        .expand(-1, nhk, -1)
        .to(dtype=torch.int32, device=device)
        .contiguous()
    )


def _indices_to_ranges_video(indices, qseqlen, kvseqlen, nhk, device):
    """Convert block indices to q_ranges/k_ranges for video scenario."""
    import torch

    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    # indices: (qseqlen, nhk, n_topk_blocks)
    # generate_ranges expects: (nhk, qseqlen, n_topk_blocks)
    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_V, num_k_blocks=kvseqlen // KBS_V
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)
    return q_ranges, k_ranges, atm


def _print_summary(results):
    """Print summary table."""
    print("\n  ╔══════════════════════════════╦══════════════════════════════════════╗")
    print("  ║ kvseqlen (qseq, topk)       ║  fwd      bwd_loopk   bwd_loopq     ║")
    print("  ╠══════════════════════════════╬══════════════════════════════════════╣")
    for kvseqlen, qseqlen, topk in SCENARIOS:
        label = f"{kvseqlen // 1024:>3d}k ({qseqlen:>5d}, {topk // 1024:>3d}k)"
        print(f"  ║ {label:<28s} ║", end="")
        for pass_type in PASSES:
            for method in METHODS:
                key = f"{pass_type}/{method}"
                d = results.get(key, {})
                if kvseqlen in d.get("topk", []):
                    idx = d["topk"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    if tf is not None:
                        print(f" {tf:>5.0f}", end="")
                    else:
                        print("  FAIL", end="")
                else:
                    print("     -", end="")
            print(" │", end="")
        print(" ║")
    print("  ╚══════════════════════════════╩══════════════════════════════════════╝")
    print("  (columns per pass: dense / block_sparse / index_sparse)")


def _phase6_plot():
    """Generate scaling plot: TFLOPS vs kvseqlen, LoopK vs LoopQ comparison."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "6-video-production"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    kvseqlens = [s[0] for s in SCENARIOS]
    x_labels = [f"{s // 1024}k" for s in kvseqlens]
    x = np.arange(len(kvseqlens))

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    colors_method = {
        "dense": "#1565C0",
        "block_sparse": "#C62828",
        "index_sparse": "#F57C00",
    }
    linestyles_pass = {"bwd_loopk": "-", "bwd_loopq": "--"}

    # Panel 1: FWD scaling (all methods)
    ax = axes[0]
    for method in METHODS:
        key = f"fwd/{method}"
        d = results.get(key, {})
        vals = []
        for kv in kvseqlens:
            if kv in d.get("topk", []):
                idx = d["topk"].index(kv)
                vals.append(d["tflops"][idx] or 0)
            else:
                vals.append(0)
        ax.plot(
            x,
            vals,
            "o-",
            color=colors_method[method],
            linewidth=2,
            markersize=8,
            label=method,
        )
    ax.set_title("FWD: TFLOPS vs kvseqlen", fontsize=13, fontweight="bold")
    ax.set_xlabel("kvseqlen", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: BWD LoopK vs LoopQ for block_sparse
    ax = axes[1]
    for pass_type in ["bwd_loopk", "bwd_loopq"]:
        for method in METHODS:
            key = f"{pass_type}/{method}"
            d = results.get(key, {})
            vals = []
            for kv in kvseqlens:
                if kv in d.get("topk", []):
                    idx = d["topk"].index(kv)
                    vals.append(d["tflops"][idx] or 0)
                else:
                    vals.append(0)
            label = f"{pass_type.replace('bwd_', '')} {method}"
            ax.plot(
                x,
                vals,
                marker="o" if pass_type == "bwd_loopk" else "s",
                linestyle=linestyles_pass[pass_type],
                color=colors_method[method],
                linewidth=2,
                markersize=7,
                label=label,
            )
    ax.set_title("BWD: LoopK vs LoopQ × method", fontsize=13, fontweight="bold")
    ax.set_xlabel("kvseqlen", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # Panel 3: LoopK/LoopQ ratio (block_sparse only)
    ax = axes[2]
    for method in METHODS:
        ratios = []
        for kv in kvseqlens:
            lk_key = f"bwd_loopk/{method}"
            lq_key = f"bwd_loopq/{method}"
            lk_d = results.get(lk_key, {})
            lq_d = results.get(lq_key, {})
            lk_v = lq_v = 0
            if kv in lk_d.get("topk", []):
                lk_v = lk_d["tflops"][lk_d["topk"].index(kv)] or 0
            if kv in lq_d.get("topk", []):
                lq_v = lq_d["tflops"][lq_d["topk"].index(kv)] or 0
            if lq_v > 0 and lk_v > 0:
                ratios.append(lk_v / lq_v)
            else:
                ratios.append(0)
        ax.plot(
            x,
            ratios,
            "o-",
            color=colors_method[method],
            linewidth=2,
            markersize=8,
            label=method,
        )
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title(
        "LoopK / LoopQ Ratio\n(>1 = LoopQ better)", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("kvseqlen", fontsize=11)
    ax.set_ylabel("Ratio", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Phase 6: Video Production Scenario\n"
        f"nhq={NHQ_V}, nhk={NHK_V}, hd={HD_V}, kbs={KBS_V}, "
        f"qblksize={Q_BLOCK_SIZE}, PackGQA, bf16, H100",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out, "phase6_video_production.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Phase 6 plot -> {path}")


def _phase6_ncu():
    """NCU at kvseqlen=128k for block_sparse loopk vs loopq."""
    phase = "6-video-production"
    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    ncu_bin = "/usr/local/cuda/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = os.path.join(
            os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "ncu"
        )

    metrics = (
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,"
        "launch__registers_per_thread,"
        "sm__cycles_elapsed.avg,"
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,"
        "dram__bytes.sum"
    )

    gpu = _find_free_gpu()
    # Use kvseqlen=128k scenario
    kvseqlen, qseqlen, topk = 131072, 2048, 32768

    ncu_configs = [
        ("bs_loopk_128k", True),
        ("bs_loopq_128k", False),
    ]

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    script_template = """\
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{GPU}"
sys.path.insert(0, "/home/niubility2/cenzhiyao/MagiAttention")
sys.path.insert(0, "/home/niubility2/cenzhiyao/MagiAttention/exps/attn/sparse")
from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices
torch.manual_seed(42)
NHQ, NHK, HD, KBS = {NHQ}, {NHK}, {HD}, {KBS}
kvseqlen, qseqlen, topk = {KVSEQLEN}, {QSEQLEN}, {TOPK}
device = "cuda"
q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
n_total = kvseqlen // KBS
n_topk = topk // KBS
gen = torch.Generator().manual_seed(42)
rand_vals = torch.rand(qseqlen, n_total, generator=gen)
perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
indices = perms.unsqueeze(1).expand(-1, NHK, -1).to(dtype=torch.int32, device=device).contiguous()
ia_3d = indices.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(ia_3d, block_m=1, block_n=KBS, num_k_blocks=n_total)
atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)
kw = dict(q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    pack_gqa=True, block_sparse=True, k_block_size=KBS, swap_bwd_qk_loop={SWAP_QK})
out, _ = flex_flash_attn_func(q, k, v, **kw)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE]")
"""

    for name, is_loopk in ncu_configs:
        script_text = script_template.format(
            GPU=str(gpu),
            NHQ=NHQ_V,
            NHK=NHK_V,
            HD=HD_V,
            KBS=KBS_V,
            KVSEQLEN=kvseqlen,
            QSEQLEN=qseqlen,
            TOPK=topk,
            SWAP_QK="True" if is_loopk else "False",
        )
        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(script_path, "w") as f:
            f.write(script_text)

        rep_path = os.path.join(out, f"ncu_{name}.ncu-rep")
        csv_path = os.path.join(out, f"ncu_{name}.csv")
        cmd = [
            ncu_bin,
            "-f",
            "--kernel-name",
            "regex:device_kernel",
            "--launch-skip",
            "3",
            "--launch-count",
            "1",
            "--metrics",
            metrics,
            "--csv",
            "-o",
            rep_path.replace(".ncu-rep", ""),
            sys.executable,
            script_path,
        ]
        print(f"  [{_ts()}] NCU {name}...", end=" ", flush=True)
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=1800)
        print("done", flush=True)

    print(f"\n[{_ts()}] Phase 6 NCU results in {out}/ncu_*.csv")
