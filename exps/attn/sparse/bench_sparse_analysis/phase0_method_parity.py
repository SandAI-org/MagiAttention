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
    S_FULL,
    TOPK_VALS,
    _bench_ffa,
    _build_idx_kbs128,
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
#  Phase 0: method-parity (S=topk, 5 methods)
# ═══════════════════════════════════════════════════════════════
def _phase0_bench(force=False, rerun_filter=None):
    import torch

    phase = "0-method-parity"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 0: method-parity S=topk (gpu{gpu})", flush=True)

    def run_d1b(topk, pass_type):
        kw = dict(
            q_ranges=torch.tensor([[0, topk]], dtype=torch.int32, device=device),
            k_ranges=torch.tensor([[0, topk]], dtype=torch.int32, device=device),
            attn_type_map=torch.zeros(1, dtype=torch.int32, device=device),
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    def run_d1b_nopg(topk, pass_type):
        kw = dict(
            q_ranges=torch.tensor([[0, topk]], dtype=torch.int32, device=device),
            k_ranges=torch.tensor([[0, topk]], dtype=torch.int32, device=device),
            attn_type_map=torch.zeros(1, dtype=torch.int32, device=device),
            pack_gqa=False,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    def _make_nb_ranges(seqlen):
        n_qblocks = seqlen // 128
        q_starts = torch.arange(0, seqlen, 128, dtype=torch.int32, device=device)
        q_ends = q_starts + 128
        q_r = torch.stack([q_starts, q_ends], dim=-1)
        k_r = torch.zeros(n_qblocks, 2, dtype=torch.int32, device=device)
        k_r[:, 1] = seqlen
        atm = torch.zeros(n_qblocks, dtype=torch.int32, device=device)
        return q_r, k_r, atm

    def run_dense_nb(topk, pass_type):
        q_r, k_r, atm = _make_nb_ranges(topk)
        kw = dict(
            q_ranges=q_r,
            k_ranges=k_r,
            attn_type_map=atm,
            block_sparse=False,
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    def run_dense_nb_nopg(topk, pass_type):
        q_r, k_r, atm = _make_nb_ranges(topk)
        kw = dict(
            q_ranges=q_r,
            k_ranges=k_r,
            attn_type_map=atm,
            block_sparse=False,
            pack_gqa=False,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    def run_ia(topk, pass_type):
        indices = _build_idx_kbs128(topk, topk, device)
        kw = dict(
            index_sparse_indices=indices,
            sparse_k_block_size=KBS,
            index_sparse=True,
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    def run_sl(topk, pass_type):
        indices = _build_idx_kbs128(topk, topk, "cpu").to(device)
        q_ranges, k_ranges, atm = _indices_to_ranges(indices, topk)
        kw = dict(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=atm,
            block_sparse=True,
            range_merge=True,
            pack_gqa=True,
            sparse_k_block_size=KBS,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(topk, topk, pass_type, kw, device)

    METHODS = [
        ("d1b", "Dense-SingleBatch", run_d1b),
        ("d1b_nopg", "Dense-SingleBatch-noPackGQA", run_d1b_nopg),
        ("dense_nb", "Dense-MultiBatch", run_dense_nb),
        ("dense_nb_nopg", "Dense-MultiBatch-noPackGQA", run_dense_nb_nopg),
        ("ia", "IndexSparse", run_ia),
        ("sl", "BlockSparse", run_sl),
    ]
    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]

    for pass_id, pass_name in PASSES:
        print(f"\n{'=' * 60}\n[{_ts()}] {pass_name}", flush=True)
        for method_id, method_name, method_fn in METHODS:
            print(f"  {method_name}:", flush=True)
            for topk in TOPK_VALS:
                key = f"{pass_id}/{method_id}"

                should_run = True
                if rerun_filter is not None:
                    should_run = (pass_id, method_id, topk) in rerun_filter
                elif not force and _has_entry(results, key, topk):
                    should_run = False

                if not should_run:
                    d = results.get(key, {})
                    if d and topk in d.get("topk", []):
                        idx = d["topk"].index(topk)
                        print(
                            f"    topk={topk:>5d}: {d['tflops'][idx]:>7.1f} T (cached)",
                            flush=True,
                        )
                    else:
                        print(f"    topk={topk:>5d}: SKIP", flush=True)
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                try:
                    t0 = time.time()
                    tf, ms = method_fn(topk, pass_id)
                    elapsed = time.time() - t0
                    _set_entry(results, key, topk, round(tf, 1), round(ms, 3))
                    print(
                        f"    topk={topk:>5d}: {tf:>7.1f} T "
                        f"({ms:.3f}ms, {elapsed:.0f}s)",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, topk, None, None)
                    print(f"    topk={topk:>5d}: FAIL - {e}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()

            _save_results(phase, results)

    print(f"\n[{_ts()}] Phase 0 DONE -> {_results_path(phase)}", flush=True)


def _phase0_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "0-method-parity"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]
    METHODS = [
        ("d1b", "Dense-SingleBatch", (0.58, 0.58, 0.58)),
        ("d1b_nopg", "Dense-SingleBatch-noPackGQA", (0.78, 0.78, 0.78)),
        ("dense_nb", "Dense-MultiBatch", (0.22, 0.37, 0.71)),
        ("dense_nb_nopg", "Dense-MultiBatch-noPackGQA", (0.47, 0.62, 0.86)),
        ("ia", "IndexSparse", (0.77, 0.34, 0.49)),
        ("sl", "BlockSparse", (0.29, 0.57, 0.60)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), dpi=150)
    x = np.arange(len(TOPK_VALS))
    bw = 0.12

    for col_idx, (pid, pname) in enumerate(PASSES):
        ax = axes[col_idx]
        for i, (mid, lbl, col) in enumerate(METHODS):
            key = f"{pid}/{mid}"
            d = results.get(key, {})
            vals = []
            for tk in TOPK_VALS:
                if tk in d.get("topk", []):
                    idx = d["topk"].index(tk)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            off = (i - len(METHODS) / 2 + 0.5) * bw
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

        ax.set_title(f"{pname} (S=topk)", fontsize=14, fontweight="bold")
        ax.set_xlabel("topk", fontsize=12)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t // 1024}K" for t in TOPK_VALS], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, 800)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 0: Method Parity at S=topk "
        f"(nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, bf16)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "method_parity.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot -> {path}")


def _phase0_ncu():
    phase = "0-method-parity"
    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    ncu_bin = "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"

    metrics = ",".join(
        [
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "l1tex__t_sectors.sum",
        ]
    )

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    S = S_FULL
    configs = [
        ("fwd_d1b", "fwd", False, False),
        ("fwd_ia", "fwd", True, True),
        ("fwd_sl", "fwd", True, False),
        ("bwd_loopq_d1b", "bwd_loopq", False, False),
        ("bwd_loopq_ia", "bwd_loopq", True, True),
        ("bwd_loopq_sl", "bwd_loopq", True, False),
        ("bwd_loopk_d1b", "bwd_loopk", False, False),
        ("bwd_loopk_ia", "bwd_loopk", True, True),
        ("bwd_loopk_sl", "bwd_loopk", True, False),
    ]

    for name, pass_type, pack_gqa, index_sparse in configs:
        is_bwd = pass_type != "fwd"
        swap_loopk = "True" if pass_type == "bwd_loopk" else "False"
        launch_skip = 3 if is_bwd else 0

        grad_line = (
            "q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)"
            if is_bwd
            else ""
        )
        bwd_code = "do = torch.randn_like(out); out.backward(do)" if is_bwd else ""

        if index_sparse:
            call = (
                f"idx = torch.arange({S}//{KBS}, dtype=torch.int32, device='cuda')\n"
                f"idx = idx.unsqueeze(0).unsqueeze(0).expand({S}, {NHK}, -1).contiguous()\n"
                f"out, _ = flex_flash_attn_func(q, k, v,\n"
                f"    index_sparse_indices=idx, sparse_k_block_size={KBS},\n"
                f"    index_sparse=True, pack_gqa=True,\n"
                f"    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})"
            )
        elif pack_gqa:
            call = (
                f"from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices\n"
                f"idx = torch.arange({S}//{KBS}, dtype=torch.int32, device='cuda')\n"
                f"idx = idx.unsqueeze(0).unsqueeze(0).expand({S}, {NHK}, -1).contiguous()\n"
                f"ia_3d = idx.permute(1, 0, 2).contiguous()\n"
                f"q_ranges, k_ranges = generate_ranges_from_topk_indices(\n"
                f"    ia_3d, block_m=1, block_n={KBS}, num_k_blocks={S}//{KBS})\n"
                f"atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device='cuda')\n"
                f"out, _ = flex_flash_attn_func(q, k, v,\n"
                f"    q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,\n"
                f"    block_sparse=True, range_merge=True, pack_gqa=True,\n"
                f"    sparse_k_block_size={KBS},\n"
                f"    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})"
            )
        else:
            call = (
                f"q_ranges = torch.tensor([[0, {S}]], dtype=torch.int32, device='cuda')\n"
                f"k_ranges = torch.tensor([[0, {S}]], dtype=torch.int32, device='cuda')\n"
                f"atm = torch.zeros(1, dtype=torch.int32, device='cuda')\n"
                f"out, _ = flex_flash_attn_func(q, k, v,\n"
                f"    q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,\n"
                f"    pack_gqa=False,\n"
                f"    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})"
            )

        script = f"""\
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-13.0'
import torch
from magi_attention.functional import flex_flash_attn_func
torch.manual_seed(42)
q = torch.randn({S}, {NHQ}, {HD}, dtype=torch.bfloat16, device='cuda')
k = torch.randn({S}, {NHK}, {HD}, dtype=torch.bfloat16, device='cuda')
v = torch.randn({S}, {NHK}, {HD}, dtype=torch.bfloat16, device='cuda')
{grad_line}
{call}
{bwd_code}
torch.cuda.synchronize()
print('[DONE] {name}')
"""
        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(script_path, "w") as f:
            f.write(script)

        csv_path = os.path.join(out, f"ncu_{name}.csv")
        cmd = [
            ncu_bin,
            "--kernel-name",
            "regex:device_kernel",
            "--launch-skip",
            str(launch_skip),
            "--launch-count",
            "1",
            "--metrics",
            metrics,
            "--csv",
            sys.executable,
            script_path,
        ]
        print(f"  [{_ts()}] NCU {name}...", end=" ", flush=True)
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=600)
        print("done", flush=True)

    print(f"\n[{_ts()}] Phase 1 NCU results in {out}/ncu_*.csv")
