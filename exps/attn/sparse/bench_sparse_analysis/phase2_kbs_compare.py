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

from bench_sparse_analysis._common import (
    HD,
    KBS,
    NHK,
    NHQ,
    S_FULL,
    TOPK_VALS,
    _bench_kernel,
    _build_idx_kbs1,
    _build_idx_kbs128,
    _build_idx_kbs128_cpasync,
    _calc_flops,
    _has_entry,
    _load_results,
    _make_tensors,
    _make_tensors_kv_short,
    _out_dir,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)


# ═══════════════════════════════════════════════════════════════
#  Phase 2: kbs-compare
# ═══════════════════════════════════════════════════════════════
def _phase2_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "2-kbs-compare"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 2: kbs=1 vs kbs=128 (gpu{gpu})", flush=True)

    CONFIGS = [
        ("fwd", "FWD"),
        ("bwd_loopk", "BWD LoopK"),
    ]
    METHODS = [
        ("d1b", "Dense-SingleBatch"),
        ("d1b_nopg", "Dense-SingleBatch-noPackGQA"),
        ("dense_nb", "Dense-MultiBatch"),
        ("dense_nb_nopg", "Dense-MultiBatch-noPackGQA"),
        ("is128", "kbs=128 TMA"),
        ("is128cp", "kbs=128 CpAsync"),
        ("is1", "kbs=1 CpAsync"),
    ]

    for pass_id, pass_name in CONFIGS:
        is_bwd = pass_id != "fwd"
        print(f"\n{'=' * 60}\n[{_ts()}] {pass_name}", flush=True)

        for method_id, method_name in METHODS:
            print(f"  {method_name}:", flush=True)
            for topk in TOPK_VALS:
                key = f"{pass_id}/{method_id}"
                if not force and _has_entry(results, key, topk):
                    d = results[key]
                    idx = d["topk"].index(topk)
                    print(
                        f"    topk={topk:>5d}: {d['tflops'][idx]:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(42)
                try:
                    if method_id in ("d1b", "d1b_nopg"):
                        pg = method_id == "d1b"
                        q_ranges = torch.tensor(
                            [[0, S_FULL]], dtype=torch.int32, device=device
                        )
                        k_ranges = torch.tensor(
                            [[0, topk]], dtype=torch.int32, device=device
                        )
                        atm = torch.zeros(1, dtype=torch.int32, device=device)
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            pack_gqa=pg,
                        )
                        if is_bwd:
                            kw["swap_bwd_qk_loop"] = True
                        q, k, v = _make_tensors_kv_short(
                            S_FULL, topk, device, torch.bfloat16, grad=is_bwd
                        )
                        flops = _calc_flops(S_FULL, topk, is_bwd)

                    elif method_id in ("dense_nb", "dense_nb_nopg"):
                        pg = method_id == "dense_nb"
                        n_qb = S_FULL // 128
                        qs = torch.arange(
                            0, S_FULL, 128, dtype=torch.int32, device=device
                        )
                        qe = qs + 128
                        q_r = torch.stack([qs, qe], dim=-1)
                        k_r = torch.zeros(n_qb, 2, dtype=torch.int32, device=device)
                        k_r[:, 1] = topk
                        atm = torch.zeros(n_qb, dtype=torch.int32, device=device)
                        kw = dict(
                            q_ranges=q_r,
                            k_ranges=k_r,
                            attn_type_map=atm,
                            block_sparse=False,
                            pack_gqa=pg,
                        )
                        if is_bwd:
                            kw["swap_bwd_qk_loop"] = True
                        q, k, v = _make_tensors_kv_short(
                            S_FULL, topk, device, torch.bfloat16, grad=is_bwd
                        )
                        flops = _calc_flops(S_FULL, topk, is_bwd)

                    elif method_id == "is128":
                        idx128 = _build_idx_kbs128(S_FULL, topk, device)
                        kw = dict(
                            index_sparse_indices=idx128,
                            sparse_k_block_size=128,
                            index_sparse=True,
                            pack_gqa=True,
                        )
                        if is_bwd:
                            kw["swap_bwd_qk_loop"] = True
                        q, k, v = _make_tensors(
                            S_FULL, device, torch.bfloat16, grad=is_bwd
                        )
                        flops = _calc_flops(S_FULL, topk, is_bwd)

                    elif method_id == "is128cp":
                        idx128cp = _build_idx_kbs128_cpasync(S_FULL, topk, device)
                        kw = dict(
                            index_sparse_indices=idx128cp,
                            sparse_k_block_size=1,
                            index_sparse=True,
                            pack_gqa=True,
                        )
                        if is_bwd:
                            kw["swap_bwd_qk_loop"] = True
                        q, k, v = _make_tensors(
                            S_FULL, device, torch.bfloat16, grad=is_bwd
                        )
                        flops = _calc_flops(S_FULL, topk, is_bwd)

                    else:  # is1
                        idx1 = _build_idx_kbs1(S_FULL, topk, device)
                        kw = dict(
                            index_sparse_indices=idx1,
                            sparse_k_block_size=1,
                            index_sparse=True,
                            pack_gqa=True,
                        )
                        if is_bwd:
                            kw["swap_bwd_qk_loop"] = True
                        q, k, v = _make_tensors(
                            S_FULL, device, torch.bfloat16, grad=is_bwd
                        )
                        flops = _calc_flops(S_FULL, topk, is_bwd)

                    o, *_ = flex_flash_attn_func(q, k, v, **kw)

                    if not is_bwd:

                        def run_fn():
                            flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

                    else:
                        do = torch.randn_like(o)

                        def run_fn():  # noqa: F811
                            o.backward(do, retain_graph=True)  # noqa: F821

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    _set_entry(results, key, topk, round(tf, 1), round(ms, 3))
                    print(
                        f"    topk={topk:>5d}: {tf:>7.1f} T ({ms:.3f}ms)",
                        flush=True,
                    )
                except Exception as e:
                    _set_entry(results, key, topk, None, None)
                    print(f"    topk={topk:>5d}: FAIL - {e}", flush=True)
                finally:
                    q = k = v = None
                    gc.collect()
                    torch.cuda.empty_cache()

            _save_results(phase, results)

    print(f"\n[{_ts()}] Phase 0 DONE -> {_results_path(phase)}", flush=True)


def _phase2_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "2-kbs-compare"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    x = np.arange(len(TOPK_VALS))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    bw = 0.10
    for ax_idx, (pass_id, title) in enumerate(
        [("fwd", "FWD"), ("bwd_loopk", "BWD LoopK")]
    ):
        ax = axes[ax_idx]
        configs = [
            (f"{pass_id}/d1b", "Dense-SingleBatch", (0.58, 0.58, 0.58)),
            (f"{pass_id}/d1b_nopg", "Dense-SingleBatch-noPackGQA", (0.78, 0.78, 0.78)),
            (f"{pass_id}/dense_nb", "Dense-MultiBatch", (0.22, 0.37, 0.71)),
            (
                f"{pass_id}/dense_nb_nopg",
                "Dense-MultiBatch-noPackGQA",
                (0.47, 0.62, 0.86),
            ),
            (f"{pass_id}/is128", "kbs=128 TMA", (0.18, 0.53, 0.76)),
            (f"{pass_id}/is128cp", "kbs=128 CpAsync", (0.95, 0.55, 0.20)),
            (f"{pass_id}/is1", "kbs=1 CpAsync", (0.91, 0.30, 0.24)),
        ]
        for i, (key, label, color) in enumerate(configs):
            d = results.get(key, {})
            vals = []
            for tk in TOPK_VALS:
                if tk in d.get("topk", []):
                    idx = d["topk"].index(tk)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            off = (i - len(configs) / 2 + 0.5) * bw
            bars = ax.bar(
                x + off,
                vals,
                width=bw,
                label=label,
                color=color,
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

        ax.set_title(
            f"{title}: kbs=1 vs kbs=128\n(S={S_FULL // 1024}K, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("topk", fontsize=12)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t // 1024}K" for t in TOPK_VALS], fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, 800)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, "kbs1_vs_kbs128.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot -> {path}")


def _phase2_ncu():
    phase = "2-kbs-compare"
    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    ncu_bin = "/usr/local/cuda-13.0/bin/ncu"
    if not os.path.exists(ncu_bin):
        ncu_bin = "ncu"

    metrics = ",".join(
        [
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
            "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
            "dram__bytes_read.sum",
            "launch__registers_per_thread",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
            "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        ]
    )

    configs = [
        ("fwd_kbs128", False, True, 128),
        ("fwd_kbs1", False, True, 1),
        ("bwd_kbs128", True, True, 128),
        ("bwd_kbs1", True, True, 1),
    ]

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for name, is_bwd, index_sparse, kbs in configs:
        S = S_FULL
        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")

        grad_line = (
            "q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)"
            if is_bwd
            else ""
        )
        if kbs == 128:
            idx_code = (
                f"idx = torch.arange({S}//{KBS}, dtype=torch.int32, device='cuda')\n"
                f"idx = idx.unsqueeze(0).unsqueeze(0).expand({S}, {NHK}, -1).contiguous()"
            )
        else:
            idx_code = (
                f"idx = torch.arange({S}, dtype=torch.int32, device='cuda')\n"
                f"idx = idx.unsqueeze(0).unsqueeze(0).expand({S}, {NHK}, -1).contiguous()"
            )

        swap_arg = "swap_bwd_qk_loop=True," if is_bwd else ""
        bwd_code = "do = torch.randn_like(out); out.backward(do)" if is_bwd else ""
        launch_skip = 3 if is_bwd else 0

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
{idx_code}
out, _ = flex_flash_attn_func(q, k, v,
    index_sparse_indices=idx, sparse_k_block_size={kbs},
    index_sparse=True, pack_gqa=True, {swap_arg})
{bwd_code}
torch.cuda.synchronize()
print('[DONE] {name}')
"""
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

    print(f"\n[{_ts()}] NCU results in {out}/ncu_*.csv", flush=True)
    print("  Parse with: grep -E 'tensor_cycles|pipe_lsu|local_op' ncu_*.csv")
