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

Production config: 1080p, qhead=32, kvhead=8, hd=128.
32-GPU training: 8 KV-heads distributed → per-rank nhk=1, nhq=128/8*32=128 (4 Q-ranks share 1 KV-head).
Per-rank: qseqlen = kvseqlen/64, topk = kvseqlen/4.
Post-distribution: NHQ=128, NHK=1, PackGQA, q_block_size=1, kbs=128(block)/1(index).

Key question: does small qseqlen cause LoopQ outer-parallelism starvation?
"""

import gc
import os
import subprocess
import sys
import time

from bench_sparse_analysis._common import (
    HD,
    KBS,
    NHK,
    NHQ,
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
# Per-rank params after distribution (same head config as phase5)
# NHQ=128, NHK=1, HD=128, KBS=128 — imported from _common
# pack_gqa_factor = 128 → effective M per Q position = 128 (ideal for tensor cores)

# kvseqlen → (qseqlen, topk)
# qseqlen = kvseqlen / 64 (Q_BLOCK_SIZE=16 × 4 Q-ranks)
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
METHODS = ["d1b", "d1b_nopg", "dense_nb", "block_sparse", "index_sparse"]


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
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, "
        f"q_block_size=1, PackGQA, bf16\n",
        flush=True,
    )

    def _make_nb_ranges(seqlen):
        """Per-Qblock ranges: each Q block's k_range covers full [0, seqlen)."""
        n_qblocks = seqlen // 128
        q_starts = torch.arange(0, seqlen, 128, dtype=torch.int32, device=device)
        q_ends = q_starts + 128
        q_r = torch.stack([q_starts, q_ends], dim=-1)
        k_r = torch.zeros(n_qblocks, 2, dtype=torch.int32, device=device)
        k_r[:, 1] = seqlen
        atm = torch.zeros(n_qblocks, dtype=torch.int32, device=device)
        return q_r, k_r, atm

    def _build_sparse_indices(qseqlen_l, kvseqlen_l, topk_l):
        """Build per-Q-position block indices selecting topk_l/KBS blocks from kvseqlen_l/KBS."""
        n_kv_blocks = kvseqlen_l // KBS
        n_topk_blocks = topk_l // KBS
        if n_topk_blocks >= n_kv_blocks:
            idx = torch.arange(n_kv_blocks, dtype=torch.int32, device=device)
            return idx.unsqueeze(0).unsqueeze(0).expand(qseqlen_l, NHK, -1).contiguous()
        gen = torch.Generator().manual_seed(42)
        rand_vals = torch.rand(qseqlen_l, n_kv_blocks, generator=gen)
        perms = rand_vals.argsort(dim=1)[:, :n_topk_blocks].sort(dim=1).values
        return (
            perms.unsqueeze(1)
            .expand(-1, NHK, -1)
            .to(dtype=torch.int32, device=device)
            .contiguous()
        )

    def _to_ranges(indices, kvseqlen_l):
        """Convert block indices to q_ranges/k_ranges using kvseqlen for K block count."""
        from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

        ia_3d = indices.permute(1, 0, 2).contiguous()
        q_ranges, k_ranges = generate_ranges_from_topk_indices(
            ia_3d, block_m=1, block_n=KBS, num_k_blocks=kvseqlen_l // KBS
        )
        atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=indices.device)
        return q_ranges, k_ranges, atm

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
                    tf = d["tflops"][idx]
                    if tf is not None:
                        print(
                            f"    {pass_type:10s} {method:14s}: "
                            f"{tf:>7.1f} T (cached)",
                            flush=True,
                        )
                    else:
                        print(
                            f"    {pass_type:10s} {method:14s}:    SKIP (cached)",
                            flush=True,
                        )
                    continue

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    torch.manual_seed(42)

                    if method == "d1b":
                        # Dense-SingleBatch: Q=K=topk, single range, PackGQA
                        q = torch.randn(
                            topk, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        q_ranges = torch.tensor(
                            [[0, topk]], dtype=torch.int32, device=device
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
                        flops_S = topk

                    elif method == "d1b_nopg":
                        # Dense-SingleBatch-noPackGQA
                        q = torch.randn(
                            topk, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        q_ranges = torch.tensor(
                            [[0, topk]], dtype=torch.int32, device=device
                        )
                        k_ranges = torch.tensor(
                            [[0, topk]], dtype=torch.int32, device=device
                        )
                        atm = torch.zeros(1, dtype=torch.int32, device=device)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=False,
                        )
                        flops_S = topk

                    elif method == "dense_nb":
                        # Dense-MultiBatch: per-Qblock ranges, PackGQA
                        q = torch.randn(
                            topk, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        q_ranges, k_ranges, atm = _make_nb_ranges(topk)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=True,
                        )
                        flops_S = topk

                    elif method == "block_sparse":
                        # BlockSparse: Q=qseqlen, K=kvseqlen, kbs=128
                        q = torch.randn(
                            qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_sparse_indices(qseqlen, kvseqlen, topk)
                        q_ranges, k_ranges, atm = _to_ranges(indices, kvseqlen)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=True,
                            block_sparse=True,
                            k_block_size=KBS,
                        )
                        flops_S = qseqlen

                    else:  # index_sparse
                        # IndexSparse: Q=qseqlen, K=kvseqlen, kbs=128
                        q = torch.randn(
                            qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device
                        )
                        k = torch.randn(
                            kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_sparse_indices(qseqlen, kvseqlen, topk)
                        kw = dict(
                            index_sparse_indices=indices,
                            pack_gqa=True,
                            k_block_size=KBS,
                        )
                        flops_S = qseqlen

                    if is_bwd:
                        kw["swap_bwd_qk_loop"] = swap_qk

                    from bench_sparse_analysis._common import _calc_flops

                    flops = _calc_flops(flops_S, topk, is_bwd)

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
    print(
        "  (columns per pass: d1b / d1b_nopg / dense_nb / block_sparse / index_sparse)"
    )


def _phase6_plot():
    """Generate grouped bar chart: TFLOPS by kvseqlen for each method × pass.

    Style aligned with phase5/phase0: subplot order FWD/BWD-InnerLoopQ/BWD-InnerLoopK,
    color scheme matching existing phases (gray=Dense, teal=BlockSparse, red-pink=IndexSparse).
    """
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

    PLOT_PASSES = [
        ("fwd", "FWD"),
        ("bwd_loopq", "BWD InnerLoopQ"),
        ("bwd_loopk", "BWD InnerLoopK"),
    ]
    PLOT_METHODS = [
        ("d1b", "Dense-1Batch", (0.58, 0.58, 0.58)),
        ("d1b_nopg", "Dense-1Batch-noGQA", (0.78, 0.78, 0.78)),
        ("dense_nb", "Dense-MultiBatch", (0.22, 0.37, 0.71)),
        ("block_sparse", "BlockSparse", (0.29, 0.57, 0.60)),
        ("index_sparse", "IndexSparse", (0.77, 0.34, 0.49)),
    ]

    kvseqlens = [s[0] for s in SCENARIOS]
    x = np.arange(len(kvseqlens))
    n_methods = len(PLOT_METHODS)
    bw = 0.15

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)

    for col_idx, (pid, pname) in enumerate(PLOT_PASSES):
        ax = axes[col_idx]
        for i, (mid, lbl, col) in enumerate(PLOT_METHODS):
            key = f"{pid}/{mid}"
            d = results.get(key, {})
            vals = []
            for kv in kvseqlens:
                if kv in d.get("topk", []):
                    idx = d["topk"].index(kv)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            off = (i - n_methods / 2 + 0.5) * bw
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
                        fontsize=6,
                        fontweight="bold",
                    )

        ax.set_title(pname, fontsize=14, fontweight="bold")
        ax.set_xlabel("kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/4)", fontsize=10)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 4096}K)" for kv in kvseqlens],
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, 750)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Phase 6: Video Production Scenario \u2014 "
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, "
        f"PackGQA, bf16, H100",
        fontsize=14,
        fontweight="bold",
        y=1.02,
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
        ncu_bin = "/usr/local/cuda-12.8/bin/ncu"
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
NHQ, NHK, HD, KBS = 128, 1, 128, 128
kvseqlen, qseqlen, topk = {KVSEQLEN}, {QSEQLEN}, {TOPK}
device = "cuda"
q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
n_kv_blocks = kvseqlen // KBS
n_topk_blocks = topk // KBS
gen = torch.Generator().manual_seed(42)
rand_vals = torch.rand(qseqlen, n_kv_blocks, generator=gen)
perms = rand_vals.argsort(dim=1)[:, :n_topk_blocks].sort(dim=1).values
indices = perms.unsqueeze(1).expand(-1, NHK, -1).to(torch.int32).to(device).contiguous()
ia_3d = indices.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(
    ia_3d, block_m=1, block_n=KBS, num_k_blocks=n_kv_blocks
)
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
