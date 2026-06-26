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

"""Sparse vs Dense comparison plot — phased experiment results.

Generates a 2-row figure:
  Row 1 (subplot 1): TFLOPS bar chart at S=topk=32K across FWD/BWD-LoopQ/BWD-LoopK
  Row 2 (subplot 2): NCU L2 hit ratio evidence at S=topk=32K for same passes

Experiment conclusions:
  1. At S=topk=32K, all sparse methods (Dense-nB, IndexSparse, BlockSparse) achieve
     90-96% of D1B TFLOPS — sparse framework overhead is minimal.
  2. FWD: IA slightly > D1B (+2.5% tensor util) due to fewer global metadata loads
     (65% fewer L1TEX sectors: 233K vs 672K).
  3. BWD LoopK: stable 93% across all sparse methods.
  4. BWD LoopQ: sparse achieves 90% at S=32K (K_blocks=256 > 132 SMs, no SM starvation).

Usage:
  python bench_cmp_sparse_dense.py          # Generate plot from existing results.json
  python bench_cmp_sparse_dense.py --ncu    # Also run NCU for L2 evidence
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "outs", "l2_cache_sparse", "results.json")
OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "cmp_sparse_dense")

NHQ, NHK, HD, KBS = 128, 1, 128, 128
S_TOPK = 32768


def run_ncu_l2_evidence():
    """Run NCU at S=topk=32K for D1B/IA/SL across FWD/BWD-LoopQ/BWD-LoopK."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ncu = "/usr/local/cuda-12.8/bin/ncu"
    metrics = (
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,"
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum"
    )

    scripts_dir = os.path.join(OUT_DIR, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    configs = [
        ("fwd", "d1b", False, False, 0),
        ("fwd", "ia", True, True, 0),
        ("fwd", "sl", True, False, 0),
        ("bwd_loopq", "d1b", False, False, 3),
        ("bwd_loopq", "ia", True, True, 3),
        ("bwd_loopq", "sl", True, False, 3),
        ("bwd_loopk", "d1b", False, False, 3),
        ("bwd_loopk", "ia", True, True, 3),
        ("bwd_loopk", "sl", True, False, 3),
    ]

    ncu_results = {}
    for pass_type, method, pack_gqa, index_sparse, launch_skip in configs:
        key = f"{pass_type}/{method}"
        script_path = os.path.join(scripts_dir, f"ncu_{pass_type}_{method}.py")

        swap_loopk = "True" if pass_type == "bwd_loopk" else "False"
        is_bwd = pass_type != "fwd"

        if index_sparse:
            call_code = f"""
idx = torch.arange({S_TOPK}//{KBS}, dtype=torch.int32, device='cuda')
idx = idx.unsqueeze(0).unsqueeze(0).expand({S_TOPK}, {NHK}, -1).contiguous()
out, _ = flex_flash_attn_func(q, k, v,
    index_sparse_indices=idx, k_block_size={KBS},
    index_sparse=True, pack_gqa=True,
    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})
"""
        elif pack_gqa:
            call_code = f"""
from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices
idx = torch.arange({S_TOPK}//{KBS}, dtype=torch.int32, device='cuda')
idx = idx.unsqueeze(0).unsqueeze(0).expand({S_TOPK}, {NHK}, -1).contiguous()
ia_3d = idx.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(ia_3d, block_m=1, block_n={KBS}, num_k_blocks={S_TOPK}//{KBS})
atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device='cuda')
out, _ = flex_flash_attn_func(q, k, v,
    q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    block_sparse=True, auto_range_merge=True, pack_gqa=True,
    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})
"""
        else:
            call_code = f"""
q_ranges = torch.tensor([[0, {S_TOPK}]], dtype=torch.int32, device='cuda')
k_ranges = torch.tensor([[0, {S_TOPK}]], dtype=torch.int32, device='cuda')
atm = torch.zeros(1, dtype=torch.int32, device='cuda')
out, _ = flex_flash_attn_func(q, k, v,
    q_ranges=q_ranges, k_ranges=k_ranges, attn_type_map=atm,
    pack_gqa=False,
    {'swap_bwd_qk_loop=' + swap_loopk + ',' if is_bwd else ''})
"""

        grad_code = ""
        if is_bwd:
            grad_code = (
                "q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)"
            )

        script = f"""import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-13.0'
import torch
from magi_attention.functional import flex_flash_attn_func
q = torch.randn({S_TOPK}, {NHQ}, {HD}, dtype=torch.bfloat16, device='cuda')
k = torch.randn({S_TOPK}, {NHK}, {HD}, dtype=torch.bfloat16, device='cuda')
v = torch.randn({S_TOPK}, {NHK}, {HD}, dtype=torch.bfloat16, device='cuda')
{grad_code}
{call_code}
{"do = torch.randn_like(out); out.backward(do)" if is_bwd else ""}
torch.cuda.synchronize()
print('[DONE] {key}')
"""
        with open(script_path, "w") as f:
            f.write(script)

        csv_path = os.path.join(OUT_DIR, f"ncu_{pass_type}_{method}.csv")
        cmd = [
            ncu,
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
        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] NCU {key}...",
            end=" ",
            flush=True,
        )
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=300)
        print("done")

        # Parse
        import csv
        import io

        with open(csv_path) as f:
            lines = [line for line in f if "device_kernel" in line]
        reader = csv.reader(io.StringIO("".join(lines)))
        hit, miss = 0, 0
        for row in reader:
            if len(row) > 14:
                if "hit" in row[12]:
                    hit = int(row[14])
                elif "miss" in row[12]:
                    miss = int(row[14])
        ratio = hit / (hit + miss) * 100 if (hit + miss) > 0 else 0
        ncu_results[key] = {"hit": hit, "miss": miss, "ratio": round(ratio, 1)}
        print(f"    L2 hit ratio: {ratio:.1f}%")

    ncu_path = os.path.join(OUT_DIR, "ncu_l2_results.json")
    with open(ncu_path, "w") as f:
        json.dump(ncu_results, f, indent=2)
    print(f"  NCU results saved to {ncu_path}")
    return ncu_results


def make_plot(ncu_results=None):
    """Generate the 2-row comparison plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Load NCU if exists
    ncu_path = os.path.join(OUT_DIR, "ncu_l2_results.json")
    if ncu_results is None and os.path.exists(ncu_path):
        with open(ncu_path) as f:
            ncu_results = json.load(f)

    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]
    METHODS = [
        ("d1b_dashed", "Dense-1B", (0.58, 0.58, 0.58)),
        ("dense_dashed", "Dense-nB", (0.22, 0.37, 0.71)),
        ("ia_dashed", "IndexSparse", (0.77, 0.34, 0.49)),
        ("sl_dashed", "BlockSparse", (0.29, 0.57, 0.60)),
    ]
    NCU_METHODS = [
        ("d1b", "Dense-1B", (0.58, 0.58, 0.58)),
        ("ia", "IndexSparse", (0.77, 0.34, 0.49)),
        ("sl", "BlockSparse", (0.29, 0.57, 0.60)),
    ]

    has_ncu = ncu_results is not None and len(ncu_results) > 0
    nrows = 2 if has_ncu else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows), dpi=150)
    if nrows == 1:
        axes = [axes]

    # Row 1: TFLOPS bars
    for col_idx, (pid, pname) in enumerate(PASSES):
        ax = axes[0][col_idx]
        pd_ = results.get(pid, {})
        x = np.arange(1)
        bw = 0.18
        for i, (mid, lbl, col) in enumerate(METHODS):
            d = pd_.get(mid, {})
            val = 0
            if 32768 in d.get("topk", []):
                idx = d["topk"].index(32768)
                val = d["tflops"][idx] if d["tflops"][idx] else 0
            offset = (i - len(METHODS) / 2 + 0.5) * bw
            bar = ax.bar(
                x + offset,
                [val],
                bw,
                color=col,
                label=lbl,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.bar_label(bar, fmt="%.0f", fontsize=8)

        ax.set_title(f"{pname} — S=topk=32K", fontsize=12, fontweight="bold")
        ax.set_ylabel("TFLOPS")
        ax.set_ylim(0, 700)
        ax.set_xticks([])
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Row 2: NCU L2 hit ratio
    if has_ncu:
        for col_idx, (pid, pname) in enumerate(PASSES):
            ax = axes[1][col_idx]
            x = np.arange(1)
            bw = 0.22
            for i, (mid, lbl, col) in enumerate(NCU_METHODS):
                key = f"{pid}/{mid}"
                val = ncu_results.get(key, {}).get("ratio", 0)
                offset = (i - len(NCU_METHODS) / 2 + 0.5) * bw
                bar = ax.bar(
                    x + offset,
                    [val],
                    bw,
                    color=col,
                    label=lbl,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.bar_label(bar, fmt="%.1f%%", fontsize=8)

            ax.set_title(f"{pname} — L2 Hit Ratio (S=topk=32K)", fontsize=11)
            ax.set_ylabel("L2 Hit Ratio (%)")
            ax.set_ylim(80, 100)
            ax.set_xticks([])
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 1: Sparse ≈ Dense at S=topk=32K (nhq=128, nhk=1, hd=128, kbs=128)\n"
        "FWD: IA=96% D1B | BWD-LoopQ: 90% D1B | BWD-LoopK: 93% D1B — framework overhead minimal",
        fontsize=11,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "exp1_sparse_vs_dense_32k.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ncu", action="store_true", help="Run NCU profiling for L2 evidence"
    )
    args = parser.parse_args()

    ncu_results = None
    if args.ncu:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Running NCU L2 evidence collection..."
        )
        ncu_results = run_ncu_l2_evidence()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating plot...")
    make_plot(ncu_results)


if __name__ == "__main__":
    main()
