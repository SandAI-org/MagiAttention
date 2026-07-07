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

"""Phase 5: scaling — TFLOPS across S=32k..256k with fixed topk=S/4 ratio.

Tests fwd/bwd_loopk/bwd_loopq × dense(d1b)/block_sparse/index_sparse.
Fixed ratio topk=S/4 reveals how each method scales with sequence length.
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
    _build_idx_kbs128,
    _calc_flops,
    _find_free_gpu,
    _has_entry,
    _indices_to_ranges,
    _load_results,
    _out_dir,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)

# ═══════════════════════════════════════════════════════════════
#  Phase 5: Sequence scaling (topk = S/4, nhq=128, nhk=1, hd=128)
# ═══════════════════════════════════════════════════════════════
S_VALUES = [32768, 65536, 131072, 262144]
TOPK_RATIO = 4  # topk = S / TOPK_RATIO
PASSES = ["fwd", "bwd_loopk", "bwd_loopq"]
# Methods: multiple dense variants for fair comparison (matching phase0 convention)
#   d1b:       Dense-SingleBatch — single [0,topk) range, pack_gqa=True. S=topk. Peak compute.
#   d1b_nopg:  Dense-SingleBatch-noPackGQA — same but pack_gqa=False. Shows PackGQA benefit.
#   dense_nb:  Dense-MultiBatch — per-Qblock ranges to [0,topk), pack_gqa=True. Shows metadata overhead.
#   bs:        BlockSparse — S=4×topk, each Q attends topk. Shows sparse indexing cost.
#   is:        IndexSparse — S=4×topk, each Q attends topk. Alternative sparse method.
METHODS = ["d1b", "d1b_nopg", "dense_nb", "block_sparse", "index_sparse"]


def _phase5_bench(force=False, max_kvseqlen=None):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "5-scaling"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    s_values = S_VALUES
    if max_kvseqlen is not None:
        s_values = [s for s in S_VALUES if s <= max_kvseqlen]

    print(
        f"[{_ts()}] Phase 5: Scaling (S=32k..{s_values[-1] // 1024}k, topk=S/4, gpu{gpu})",
        flush=True,
    )
    print(f"  nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, bf16, PackGQA\n", flush=True)

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

    for S in s_values:
        topk = S // TOPK_RATIO
        print(f"  ── S={S // 1024}k, topk={topk // 1024}k ──", flush=True)

        for pass_type in PASSES:
            is_bwd = pass_type != "fwd"
            swap_qk = pass_type == "bwd_loopk"

            for method in METHODS:
                key = f"{pass_type}/{method}"
                if not force and _has_entry(results, key, S):
                    d = results[key]
                    idx = d["topk"].index(S)
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

                    if method == "d1b":
                        # Dense-SingleBatch: S=topk, single range, PackGQA
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
                        # Dense-SingleBatch-noPackGQA: same but pack_gqa=False
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
                        # Dense-MultiBatch: per-Qblock ranges, S=topk, PackGQA
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
                        # BlockSparse: S=4×topk, per-Q selects topk KV
                        q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
                        k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                        v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_idx_kbs128(S, topk, device)
                        q_ranges, k_ranges, atm = _indices_to_ranges(indices, S)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=True,
                            block_sparse=True,
                            sparse_k_block_size=KBS,
                        )
                        flops_S = S

                    else:  # index_sparse
                        q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device)
                        k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                        v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device)
                        if is_bwd:
                            q.requires_grad_(True)
                            k.requires_grad_(True)
                            v.requires_grad_(True)
                        indices = _build_idx_kbs128(S, topk, device)
                        kw = dict(
                            index_sparse_indices=indices,
                            pack_gqa=True,
                            sparse_k_block_size=KBS,
                        )
                        flops_S = S

                    if is_bwd:
                        kw["swap_bwd_qk_loop"] = swap_qk

                    t0 = time.time()
                    o, *_ = flex_flash_attn_func(q, k, v, **kw)
                    flops = _calc_flops(flops_S, topk, is_bwd)

                    if is_bwd:
                        do = torch.randn_like(o)

                        def run_fn():
                            o.backward(do, retain_graph=True)

                    else:

                        def run_fn():
                            flex_flash_attn_func(q, k, v, **kw)

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    elapsed = time.time() - t0
                    _set_entry(results, key, S, round(tf, 1), round(ms, 3))
                    print(
                        f"    {pass_type:10s} {method:14s}: "
                        f"{tf:>7.1f} T ({ms:.3f}ms, {elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, S, None, None)
                    print(
                        f"    {pass_type:10s} {method:14s}: FAIL - {e}",
                        flush=True,
                    )
                finally:
                    q = k = v = None
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(phase, results)

    print(f"\n[{_ts()}] Phase 5 DONE -> {_results_path(phase)}", flush=True)
    _print_summary(results)


def _print_summary(results):
    """Print a summary table of all S × pass × method."""
    print(
        "\n  ╔════════════╦════════════════════════════════════════"
        "════════════════════════════════════════════════════════════╗"
    )
    print(
        "  ║   S (topk) ║  fwd                          "
        "bwd_loopk                    bwd_loopq                   ║"
    )
    print(
        "  ╠════════════╬════════════════════════════════════════"
        "════════════════════════════════════════════════════════════╣"
    )
    for S in S_VALUES:
        topk = S // TOPK_RATIO
        print(f"  ║ {S // 1024:>3d}k ({topk // 1024:>2d}k) ║", end="")
        for pass_type in PASSES:
            for method in METHODS:
                key = f"{pass_type}/{method}"
                d = results.get(key, {})
                if S in d.get("topk", []):
                    idx = d["topk"].index(S)
                    tf = d["tflops"][idx]
                    if tf is not None:
                        print(f" {tf:>5.0f}", end="")
                    else:
                        print("  FAIL", end="")
                else:
                    print("     -", end="")
            print(" │", end="")
        print(" ║")
    print(
        "  ╚════════════╩════════════════════════════════════════"
        "════════════════════════════════════════════════════════════╝"
    )
    print(
        "  (columns per pass: d1b / d1b_nopg / dense_nb / block_sparse / index_sparse)"
    )


def _phase5_plot():
    """Generate grouped bar chart: TFLOPS by topk for each method × pass.

    Style aligned with phase0: subplot order FWD/BWD-InnerLoopQ/BWD-InnerLoopK,
    color scheme matching existing phases (gray=Dense, teal=BlockSparse, red-pink=IndexSparse).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "5-scaling"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    # Subplot order: FWD, BWD InnerLoopQ, BWD InnerLoopK (matching phase0 convention)
    PLOT_PASSES = [
        ("fwd", "FWD"),
        ("bwd_loopq", "BWD InnerLoopQ"),
        ("bwd_loopk", "BWD InnerLoopK"),
    ]
    # Colors matching phase0 (5 methods with distinguishable colors)
    PLOT_METHODS = [
        ("d1b", "Dense-1Batch", (0.58, 0.58, 0.58)),
        ("d1b_nopg", "Dense-1Batch-noGQA", (0.78, 0.78, 0.78)),
        ("dense_nb", "Dense-MultiBatch", (0.22, 0.37, 0.71)),
        ("block_sparse", "BlockSparse", (0.29, 0.57, 0.60)),
        ("index_sparse", "IndexSparse", (0.77, 0.34, 0.49)),
    ]

    topk_values = [S // TOPK_RATIO for S in S_VALUES]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)
    x = np.arange(len(S_VALUES))
    n_methods = len(PLOT_METHODS)
    bw = 0.15

    for col_idx, (pid, pname) in enumerate(PLOT_PASSES):
        ax = axes[col_idx]
        for i, (mid, lbl, col) in enumerate(PLOT_METHODS):
            key = f"{pid}/{mid}"
            d = results.get(key, {})
            vals = []
            for S in S_VALUES:
                if S in d.get("topk", []):
                    idx = d["topk"].index(S)
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
        ax.set_xlabel("topk (Dense: S=topk; Sparse: S=4\u00d7topk)", fontsize=10)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{t // 1024}K\n(S={S // 1024}K)" for t, S in zip(topk_values, S_VALUES)],
            fontsize=10,
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, 750)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Phase 5: Scaling \u2014 topk = S/4, "
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, bf16, PackGQA, H100",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "phase5_scaling.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Phase 5 plot -> {path}")


def _phase5_ncu():
    """NCU profiling at S=128k for all methods × bwd_loopk."""
    phase = "5-scaling"
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
        "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,"
        "launch__registers_per_thread,"
        "sm__cycles_elapsed.avg,"
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,"
        "dram__bytes.sum"
    )

    gpu = _find_free_gpu()
    S = 131072  # 128k
    topk = S // TOPK_RATIO

    ncu_configs = [
        ("d1b_loopk", "d1b", "bwd_loopk"),
        ("dense_nb_loopk", "dense_nb", "bwd_loopk"),
        ("bs_loopk", "block_sparse", "bwd_loopk"),
        ("is_loopk", "index_sparse", "bwd_loopk"),
        ("d1b_loopq", "d1b", "bwd_loopq"),
        ("dense_nb_loopq", "dense_nb", "bwd_loopq"),
        ("bs_loopq", "block_sparse", "bwd_loopq"),
    ]

    script_template = """\
import os, sys, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{GPU}"
sys.path.insert(0, "/home/niubility2/cenzhiyao/MagiAttention")
sys.path.insert(0, "/home/niubility2/cenzhiyao/MagiAttention/exps/attn/sparse")
from bench_sparse_analysis._common import NHQ, NHK, HD, KBS, _build_idx_kbs128, _indices_to_ranges
from magi_attention.functional import flex_flash_attn_func
torch.manual_seed(42)
S, topk = {S}, {TOPK}
device = "cuda"
q = torch.randn(S, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(S, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
{setup_code}
out, _ = flex_flash_attn_func(q, k, v, **kw)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE]")
"""

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    setup_d1b = """\
q_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
k_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
atm = torch.zeros(1, dtype=torch.int32, device=device)
kw = dict(q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm, pack_gqa=True, swap_bwd_qk_loop={SWAP_QK})
q = torch.randn(topk, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)"""

    setup_dense_nb = """\
n_qblocks = topk // 128
q_starts = torch.arange(0, topk, 128, dtype=torch.int32, device=device)
q_ends = q_starts + 128
q_r = torch.stack([q_starts, q_ends], dim=-1)
k_r = torch.zeros(n_qblocks, 2, dtype=torch.int32, device=device)
k_r[:, 1] = topk
atm = torch.zeros(n_qblocks, dtype=torch.int32, device=device)
kw = dict(q_ranges=q_r, k_ranges=k_r, attn_type_map=atm, pack_gqa=True, swap_bwd_qk_loop={SWAP_QK})
q = torch.randn(topk, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
k = torch.randn(topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)
v = torch.randn(topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True)"""

    setup_bs = """\
indices = _build_idx_kbs128(S, topk, device)
q_ranges, k_ranges, atm = _indices_to_ranges(indices, S)
kw = dict(q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    pack_gqa=True, block_sparse=True, sparse_k_block_size=KBS, swap_bwd_qk_loop={SWAP_QK})"""

    setup_is = """\
indices = _build_idx_kbs128(S, topk, device)
kw = dict(index_sparse_indices=indices, pack_gqa=True, sparse_k_block_size=KBS, swap_bwd_qk_loop={SWAP_QK})"""

    for name, method, pass_type in ncu_configs:
        swap_qk = "True" if pass_type == "bwd_loopk" else "False"
        if method == "d1b":
            setup = setup_d1b.format(SWAP_QK=swap_qk)
        elif method == "dense_nb":
            setup = setup_dense_nb.format(SWAP_QK=swap_qk)
        elif method == "block_sparse":
            setup = setup_bs.format(SWAP_QK=swap_qk)
        else:
            setup = setup_is.format(SWAP_QK=swap_qk)

        script_text = script_template.format(
            GPU=str(gpu), S=S, TOPK=topk, setup_code=setup
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

    print(f"\n[{_ts()}] Phase 5 NCU results in {out}/ncu_*.csv")
