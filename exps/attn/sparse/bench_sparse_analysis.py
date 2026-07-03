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

"""Unified sparse attention benchmark & analysis.

Phases (TFLOPS high→low, ideal→realistic):
  0-method-parity  : 5 methods at S=topk (sparse framework overhead baseline)
  1-topk-sweep     : Fixed S=32K varying topk (L2 cache / CTA starvation)
  2-kbs-compare    : kbs=1 (CpAsync) vs kbs=128 (TMA2D) TFLOPS (FWD+BWD)
  3-l2-inflection  : NCU at specific TFLOPS inflection points

Usage:
  python bench_sparse_analysis.py --exp  0-method-parity      # run benchmark
  python bench_sparse_analysis.py --plot 0-method-parity      # generate plot
  python bench_sparse_analysis.py --ncu  0-method-parity      # NCU profiling

  python bench_sparse_analysis.py --exp  1-topk-sweep
  python bench_sparse_analysis.py --plot 1-topk-sweep

  python bench_sparse_analysis.py --exp  2-kbs-compare
  python bench_sparse_analysis.py --plot 2-kbs-compare
  python bench_sparse_analysis.py --ncu  2-kbs-compare

  python bench_sparse_analysis.py --ncu  3-l2-inflection      # generate + run + parse

Options:
  --force           Re-run all (ignore cached results)
  --rerun FILTER    Re-run subset: 'pass/method' or 'pass/method/topk'
"""

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import time

# ── Global config ──────────────────────────────────────────────
NHQ, NHK, HD, KBS = 128, 1, 128, 128
S_FULL = 32768
TOPK_VALS = [32768, 16384, 8192, 4096, 2048]
WARMUP, ITERS = 8, 20

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_OUT = os.path.join(_SCRIPT_DIR, "outs", "sparse_analysis")

PHASES = [
    "0-method-parity",
    "1-topk-sweep",
    "2-kbs-compare",
    "3-l2-inflection",
    "4-loopk-fantasy",
    "5-seqlen-sweep",
]


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _out_dir(phase):
    return os.path.join(_BASE_OUT, phase)


def _results_path(phase):
    return os.path.join(_out_dir(phase), "results.json")


# ── GPU selection ──────────────────────────────────────────────
def _find_free_gpu():
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


# ── JSON persistence ──────────────────────────────────────────
def _load_results(phase):
    path = _results_path(phase)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(phase, results):
    path = _results_path(phase)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.getsize(tmp) > 0:
        os.replace(tmp, path)


def _has_entry(results, key, topk):
    d = results.get(key, {})
    if not d or "topk" not in d:
        return False
    try:
        idx = d["topk"].index(topk)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, topk, tflops, ms):
    if key not in results:
        results[key] = {"topk": [], "tflops": [], "ms": []}
    d = results[key]
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
        if len(parts) == 2:
            for tk in TOPK_VALS:
                entries.add((parts[0], parts[1], tk))
        elif len(parts) == 3:
            entries.add((parts[0], parts[1], int(parts[2])))
    return entries or None


# ── Timing infrastructure ─────────────────────────────────────
def _bench_kernel(run_fn, flops, device):
    import torch

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


# ── Tensor & index helpers ────────────────────────────────────
def _make_tensors(S, device, dtype, grad=False):
    import torch

    q = torch.randn(S, NHQ, HD, dtype=dtype, device=device)
    k = torch.randn(S, NHK, HD, dtype=dtype, device=device)
    v = torch.randn(S, NHK, HD, dtype=dtype, device=device)
    if grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _make_tensors_kv_short(S, topk, device, dtype, grad=False):
    import torch

    q = torch.randn(S, NHQ, HD, dtype=dtype, device=device)
    k = torch.randn(topk, NHK, HD, dtype=dtype, device=device)
    v = torch.randn(topk, NHK, HD, dtype=dtype, device=device)
    if grad:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
    return q, k, v


def _calc_flops(S, topk, is_bwd):
    fwd = 4 * S * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_idx_kbs128(S, topk, device):
    import torch

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


def _build_idx_kbs128_cpasync(S, topk, device):
    """Build kbs=128 indices expanded to per-token (kbs=1) for CpAsync path.

    Same contiguous block pattern as kbs=128 TMA, but expressed as token-level
    indices so the kernel uses CpAsync scatter instead of TMA 2D.
    """
    import torch

    block_idx = _build_idx_kbs128(S, topk, device)
    B_q, nhk, n_blocks = block_idx.shape
    expanded = block_idx.unsqueeze(-1) * KBS + torch.arange(
        KBS, device=device, dtype=torch.int32
    )
    return expanded.reshape(B_q, nhk, n_blocks * KBS).contiguous()


def _build_idx_kbs1(S, topk, device):
    import torch

    if topk >= S:
        idx = torch.arange(S, dtype=torch.int32, device=device)
        return idx.unsqueeze(0).unsqueeze(0).expand(S, NHK, -1).contiguous()
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, S, generator=gen)
    idx = rand_vals.argsort(dim=1)[:, :topk].sort(dim=1).values
    return (
        idx.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(dtype=torch.int32, device=device)
        .contiguous()
    )


def _indices_to_ranges(indices, S):
    import torch

    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS, num_k_blocks=S // KBS
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=indices.device)
    return q_ranges, k_ranges, atm


# ── Common bench wrapper ──────────────────────────────────────
def _bench_ffa(S, topk, pass_type, kw, device):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    is_bwd = pass_type != "fwd"
    if "q_ranges" in kw or "index_sparse_indices" in kw:
        q, k, v = _make_tensors(S, device, torch.bfloat16, grad=is_bwd)
    else:
        q, k, v = _make_tensors_kv_short(S, topk, device, torch.bfloat16, grad=is_bwd)
    o, *_ = flex_flash_attn_func(q, k, v, **kw)
    flops = _calc_flops(S, topk, is_bwd)

    if not is_bwd:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

    else:
        do = torch.randn_like(o)

        def run_fn():  # noqa: F811
            o.backward(do, retain_graph=True)  # noqa: F821

    tf, ms = _bench_kernel(run_fn, flops, device)
    q = k = v = o = None
    gc.collect()
    torch.cuda.empty_cache()
    return tf, ms


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
                            k_block_size=128,
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
                            k_block_size=1,
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
                            k_block_size=1,
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
    index_sparse_indices=idx, k_block_size={kbs},
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
            k_block_size=KBS,
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
            auto_range_merge=True,
            pack_gqa=True,
            k_block_size=KBS,
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
                f"    index_sparse_indices=idx, k_block_size={KBS},\n"
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
                f"    block_sparse=True, auto_range_merge=True, pack_gqa=True,\n"
                f"    k_block_size={KBS},\n"
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


# ═══════════════════════════════════════════════════════════════
#  Phase 1: topk-sweep (S=32K fixed, topk varies)
# ═══════════════════════════════════════════════════════════════
def _phase1_bench(force=False, rerun_filter=None):
    import torch

    phase = "1-topk-sweep"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 1: topk-sweep S={S_FULL} (gpu{gpu})", flush=True)

    def _p1_d1b(topk, pass_type, pg):
        kw = dict(
            q_ranges=torch.tensor([[0, S_FULL]], dtype=torch.int32, device=device),
            k_ranges=torch.tensor([[0, topk]], dtype=torch.int32, device=device),
            attn_type_map=torch.zeros(1, dtype=torch.int32, device=device),
            pack_gqa=pg,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        q, k, v = _make_tensors_kv_short(
            S_FULL, topk, device, torch.bfloat16, grad=(pass_type != "fwd")
        )
        from magi_attention.functional import flex_flash_attn_func

        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        flops = _calc_flops(S_FULL, topk, pass_type != "fwd")
        if pass_type == "fwd":

            def fn():
                flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

        else:
            do = torch.randn_like(o)

            def fn():  # noqa: F811
                o.backward(do, retain_graph=True)  # noqa: F821

        tf, ms = _bench_kernel(fn, flops, device)
        q = k = v = o = None
        gc.collect()
        torch.cuda.empty_cache()
        return tf, ms

    def _p1_nb(topk, pass_type, pg):
        n_qblocks = S_FULL // 128
        q_starts = torch.arange(0, S_FULL, 128, dtype=torch.int32, device=device)
        q_ends = q_starts + 128
        q_r = torch.stack([q_starts, q_ends], dim=-1)
        k_r = torch.zeros(n_qblocks, 2, dtype=torch.int32, device=device)
        k_r[:, 1] = topk
        atm = torch.zeros(n_qblocks, dtype=torch.int32, device=device)
        kw = dict(
            q_ranges=q_r,
            k_ranges=k_r,
            attn_type_map=atm,
            block_sparse=False,
            pack_gqa=pg,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        q, k, v = _make_tensors_kv_short(
            S_FULL, topk, device, torch.bfloat16, grad=(pass_type != "fwd")
        )
        from magi_attention.functional import flex_flash_attn_func

        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        flops = _calc_flops(S_FULL, topk, pass_type != "fwd")
        if pass_type == "fwd":

            def fn():
                flex_flash_attn_func(q, k, v, **kw)  # noqa: F821

        else:
            do = torch.randn_like(o)

            def fn():  # noqa: F811
                o.backward(do, retain_graph=True)  # noqa: F821

        tf, ms = _bench_kernel(fn, flops, device)
        q = k = v = o = None
        gc.collect()
        torch.cuda.empty_cache()
        return tf, ms

    def run_ia(topk, pass_type):
        indices = _build_idx_kbs128(S_FULL, topk, device)
        kw = dict(
            index_sparse_indices=indices,
            k_block_size=KBS,
            index_sparse=True,
            pack_gqa=True,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(S_FULL, topk, pass_type, kw, device)

    def run_sl(topk, pass_type):
        indices = _build_idx_kbs128(S_FULL, topk, "cpu").to(device)
        q_ranges, k_ranges, atm = _indices_to_ranges(indices, S_FULL)
        kw = dict(
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=atm,
            block_sparse=True,
            auto_range_merge=True,
            pack_gqa=True,
            k_block_size=KBS,
        )
        if pass_type != "fwd":
            kw["swap_bwd_qk_loop"] = pass_type == "bwd_loopk"
        return _bench_ffa(S_FULL, topk, pass_type, kw, device)

    METHODS = [
        ("d1b", "Dense-SingleBatch", lambda t, p: _p1_d1b(t, p, True)),
        ("d1b_nopg", "Dense-SingleBatch-noPackGQA", lambda t, p: _p1_d1b(t, p, False)),
        ("dense_nb", "Dense-MultiBatch", lambda t, p: _p1_nb(t, p, True)),
        (
            "dense_nb_nopg",
            "Dense-MultiBatch-noPackGQA",
            lambda t, p: _p1_nb(t, p, False),
        ),
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

    print(f"\n[{_ts()}] Phase 1 DONE -> {_results_path(phase)}", flush=True)


def _phase1_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "1-topk-sweep"
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

        ax.set_title(
            f"{pname} (S={S_FULL // 1024}K, topk varies)",
            fontsize=14,
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

    fig.suptitle(
        "Phase 1: topk Sweep at S=32K "
        f"(nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, bf16)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "topk_sweep.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot -> {path}")


# ═══════════════════════════════════════════════════════════════
#  Phase 3: l2-inflection (NCU at specific inflection points)
# ═══════════════════════════════════════════════════════════════
def _phase3_ncu():
    phase = "3-l2-inflection"
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
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
        ]
    )

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    fmt = dict(NHQ=NHQ, NHK=NHK, HD=HD, KBS=KBS)

    TEMPLATE_D1B_BWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func
S, TOPK = {S}, {TOPK}
torch.manual_seed(42)
q = torch.randn(S, {NHQ}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(TOPK, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(TOPK, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
k_ranges = torch.tensor([[0, TOPK]], dtype=torch.int32, device="cuda")
atm = torch.zeros(1, dtype=torch.int32, device="cuda")
out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
    attn_type_map=atm, pack_gqa=False, swap_bwd_qk_loop=False)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] D1B BWD LoopQ S={S} TOPK={TOPK}")
"""

    TEMPLATE_DENSE_NB_BWD = """\
import os
os.environ["CUDA_HOME"] = "/usr/local/cuda-13.0"
import torch
from magi_attention.functional import flex_flash_attn_func
from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices
S, TOPK = {S}, {TOPK}
torch.manual_seed(42)
q = torch.randn(S, {NHQ}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
n_total, n_topk = S // {KBS}, TOPK // {KBS}
if n_topk >= n_total:
    idx = torch.arange(n_total, dtype=torch.int32, device="cuda")
    idx = idx.unsqueeze(0).unsqueeze(0).expand(S, {NHK}, -1).contiguous()
else:
    gen = torch.Generator().manual_seed(42)
    rand_vals = torch.rand(S, n_total, generator=gen)
    perms = rand_vals.argsort(dim=1)[:, :n_topk].sort(dim=1).values
    idx = perms.unsqueeze(1).expand(-1, {NHK}, -1).to(dtype=torch.int32, device="cuda").contiguous()
ia_3d = idx.permute(1, 0, 2).contiguous()
q_ranges, k_ranges = generate_ranges_from_topk_indices(ia_3d, block_m=1, block_n={KBS}, num_k_blocks=n_total)
atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device="cuda")
out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
    attn_type_map=atm, block_sparse=True, auto_range_merge=True, pack_gqa=True,
    k_block_size={KBS}, swap_bwd_qk_loop=False)
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE] Dense-MultiBatch BWD LoopQ S={S} TOPK={TOPK}")
"""

    scenarios = [
        (
            "A_d1b",
            TEMPLATE_D1B_BWD,
            dict(S=8192, TOPK=8192, **fmt),
            "BWD LoopQ S=topk=8K: Dense-SingleBatch",
        ),
        (
            "A_dense_nb",
            TEMPLATE_DENSE_NB_BWD,
            dict(S=8192, TOPK=8192, **fmt),
            "BWD LoopQ S=topk=8K: Dense-MultiBatch",
        ),
        (
            "B_d1b",
            TEMPLATE_D1B_BWD,
            dict(S=32768, TOPK=16384, **fmt),
            "BWD LoopQ S=32K topk=16K: Dense-SingleBatch",
        ),
        (
            "B_dense_nb",
            TEMPLATE_DENSE_NB_BWD,
            dict(S=32768, TOPK=16384, **fmt),
            "BWD LoopQ S=32K topk=16K: Dense-MultiBatch",
        ),
    ]

    for name, template, params, desc in scenarios:
        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(script_path, "w") as f:
            f.write(template.format(**params))

        csv_path = os.path.join(out, f"ncu_{name}.csv")
        cmd = [
            ncu_bin,
            "--kernel-name",
            "regex:device_kernel",
            "--launch-skip",
            "3",
            "--launch-count",
            "1",
            "--metrics",
            metrics,
            "--csv",
            sys.executable,
            script_path,
        ]
        print(f"  [{_ts()}] NCU {desc}...", end=" ", flush=True)
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=600)
        print("done", flush=True)

    print(f"\n[{_ts()}] Phase 3 NCU results in {out}/ncu_*.csv")

    # Parse L2 hit ratios
    print("\n  L2 hit ratio summary:")
    for name, _, _, desc in scenarios:
        csv_path = os.path.join(out, f"ncu_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"    {name}: NOT FOUND")
            continue
        hit, miss = 0, 0
        with open(csv_path) as f:
            for line in f:
                if "lookup_hit" in line:
                    for p in line.split(","):
                        try:
                            hit = float(p.strip().replace('"', ""))
                        except ValueError:
                            pass
                if "lookup_miss" in line:
                    for p in line.split(","):
                        try:
                            miss = float(p.strip().replace('"', ""))
                        except ValueError:
                            pass
        if hit + miss > 0:
            ratio = hit / (hit + miss) * 100
            print(f"    {name}: L2 hit = {ratio:.1f}%")
        else:
            print(f"    {name}: could not parse")


# ═══════════════════════════════════════════════════════════════
#  Phase 4: loopk-fantasy (LoopK vs LoopQ gap analysis)
# ═══════════════════════════════════════════════════════════════
# Fair ablation: PV MMA always preserved. Uses non-bypass (SMEM+TMA) path.
# SkipVLoad loads V from block 0 (L2 cached) — pipeline intact, minimal BW.
# SkipDvStore/SkipDkStore only skip the TMA reduce-add, barrier sync preserved.
#
# Per-switch env vars (correctness NOT guaranteed):
#   MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD=1   lightweight V load (block 0, L2 cached)
#   MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE=1 skip dV TMA store (barrier protocol intact)
#   MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE=1 skip dK TMA store (barrier protocol intact)
#   MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA=1   skip dV MMA (unfair but diagnostic)

_FANTASY_ENV_KEYS = [
    "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA",
    "MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
]

# Symmetric configs: same skip flags on BOTH LoopK and LoopQ.
# Gap_contribution(X) = cost_in_LoopK(X) - cost_in_LoopQ(X)
_SKIP_FACTORS = [
    # (factor_key, env_overrides, short_name)
    ("baseline", {}, "baseline"),
    ("light_v_load", {"MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1"}, "light V load"),
    ("skip_dv_store", {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1"}, "no dV store"),
    ("skip_dk_store", {"MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1"}, "no dK store"),
    ("skip_dv_mma", {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1"}, "no dV MMA"),
    (
        "skip_dkdv_store",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        "no dK+dV store",
    ),
    (
        "skip_all",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
        },
        "skip all",
    ),
]

_FANTASY_CONFIGS = []
for factor_key, env_ov, short in _SKIP_FACTORS:
    _FANTASY_CONFIGS.append((f"loopk_{factor_key}", env_ov, False, f"LoopK: {short}"))
    _FANTASY_CONFIGS.append((f"loopq_{factor_key}", env_ov, True, f"LoopQ: {short}"))


def _phase4_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "4-loopk-fantasy"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 4: LoopK vs LoopQ gap isolation (gpu{gpu})", flush=True)
    print(f"  S=topk={S_FULL}, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16\n", flush=True)

    for label, env_overrides, is_loopq, desc in _FANTASY_CONFIGS:
        key = f"bwd/{label}"
        topk = S_FULL

        if not force and _has_entry(results, key, topk):
            d = results[key]
            idx = d["topk"].index(topk)
            print(f"  {desc}: {d['tflops'][idx]:>7.1f} T (cached)", flush=True)
            continue

        gc.collect()
        torch.cuda.empty_cache()

        # Clear all relevant env vars
        for env_key in _FANTASY_ENV_KEYS:
            os.environ.pop(env_key, None)

        # Set this experiment's env vars
        for ek, ev in env_overrides.items():
            os.environ[ek] = ev

        try:
            torch.manual_seed(42)
            q = torch.randn(
                topk, NHQ, HD, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            k = torch.randn(
                topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            v = torch.randn(
                topk, NHK, HD, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            q_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, topk]], dtype=torch.int32, device=device)
            atm = torch.zeros(1, dtype=torch.int32, device=device)

            kw = dict(
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=atm,
                pack_gqa=True,
                swap_bwd_qk_loop=not is_loopq,
            )

            t0 = time.time()
            o, *_ = flex_flash_attn_func(q, k, v, **kw)
            do = torch.randn_like(o)
            flops = _calc_flops(topk, topk, True)

            def run_fn():
                o.backward(do, retain_graph=True)

            tf, ms = _bench_kernel(run_fn, flops, device)
            elapsed = time.time() - t0
            _set_entry(results, key, topk, round(tf, 1), round(ms, 3))
            print(f"  {desc}: {tf:>7.1f} T ({ms:.3f}ms, {elapsed:.0f}s)", flush=True)
        except Exception as e:
            _set_entry(results, key, topk, None, None)
            print(f"  {desc}: FAIL - {e}", flush=True)
        finally:
            q = k = v = None
            gc.collect()
            torch.cuda.empty_cache()

        _save_results(phase, results)

    # Clean up env
    for env_key in _FANTASY_ENV_KEYS:
        os.environ.pop(env_key, None)

    print(f"\n[{_ts()}] Phase 4 DONE -> {_results_path(phase)}", flush=True)

    # Print summary table with ms-based gap decomposition
    results = _load_results(phase)
    base_ms = None
    loopq_ms = None
    for label, _, is_loopq, _ in _FANTASY_CONFIGS:
        key = f"bwd/{label}"
        d = results.get(key, {})
        if S_FULL in d.get("topk", []):
            idx = d["topk"].index(S_FULL)
            ms_val = d.get("ms", [None])[idx] if "ms" in d else None
            if label == "loopk_baseline" and ms_val:
                base_ms = ms_val
            elif label == "loopq_baseline" and ms_val:
                loopq_ms = ms_val

    total_gap = (base_ms - loopq_ms) if base_ms and loopq_ms else None
    print("\n  ╔══════════════════════════════════════╦═════════╦═════════╦══════════╗")
    print("  ║ Experiment                           ║  TFLOPS ║   ms    ║ gap frac ║")
    print("  ╠══════════════════════════════════════╬═════════╬═════════╬══════════╣")
    for label, _, _, desc in _FANTASY_CONFIGS:
        key = f"bwd/{label}"
        d = results.get(key, {})
        if S_FULL in d.get("topk", []):
            idx = d["topk"].index(S_FULL)
            tf = d["tflops"][idx]
            ms_val = d.get("ms", [None])[idx] if "ms" in d else None
            if tf is not None and ms_val is not None:
                saved = base_ms - ms_val if base_ms else 0
                frac = (
                    f"{saved / total_gap * 100:+.1f}%"
                    if total_gap and total_gap > 0
                    else ""
                )
                print(
                    f"  ║ {desc:<36s} ║ {tf:>5.0f} T ║ {ms_val:>6.1f}  ║ {frac:>8s} ║"
                )
            elif tf is not None:
                print(f"  ║ {desc:<36s} ║ {tf:>5.0f} T ║    N/A  ║          ║")
            else:
                print(f"  ║ {desc:<36s} ║  FAIL  ║    N/A  ║          ║")
        else:
            print(f"  ║ {desc:<36s} ║   N/A  ║    N/A  ║          ║")
    print("  ╚══════════════════════════════════════╩═════════╩═════════╩══════════╝")
    if total_gap:
        print(f"\n  Total LoopK-LoopQ gap: {total_gap:.1f} ms")


def _phase4_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "4-loopk-fantasy"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    # ── Plot 1: Paired bar chart (LoopK cost vs LoopQ cost per factor) ──
    factor_names, loopk_costs, loopq_costs, gap_contribs = [], [], [], []
    loopk_base_ms = _get_ms(results, "loopk_baseline")
    loopq_base_ms = _get_ms(results, "loopq_baseline")
    total_gap = (
        (loopk_base_ms - loopq_base_ms) if loopk_base_ms and loopq_base_ms else None
    )

    for factor_key, _, short in _SKIP_FACTORS:
        if factor_key == "baseline":
            continue
        lk_ms = _get_ms(results, f"loopk_{factor_key}")
        lq_ms = _get_ms(results, f"loopq_{factor_key}")
        if lk_ms is None or lq_ms is None:
            continue
        cost_lk = loopk_base_ms - lk_ms
        cost_lq = loopq_base_ms - lq_ms
        factor_names.append(short)
        loopk_costs.append(cost_lk)
        loopq_costs.append(cost_lq)
        gap_contribs.append(cost_lk - cost_lq)

    if factor_names and total_gap:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 7), dpi=150, gridspec_kw={"width_ratios": [2, 1]}
        )

        x = np.arange(len(factor_names))
        w = 0.35
        bars_k = ax1.bar(
            x - w / 2, loopk_costs, w, label="LoopK cost", color="#E53935", alpha=0.8
        )
        bars_q = ax1.bar(
            x + w / 2, loopq_costs, w, label="LoopQ cost", color="#388E3C", alpha=0.8
        )

        for bar, v in zip(bars_k, loopk_costs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                max(v, 0) + 2,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#B71C1C",
            )
        for bar, v in zip(bars_q, loopq_costs):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                max(v, 0) + 2,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#1B5E20",
            )

        ax1.set_title(
            f"Symmetric Cost Comparison (ms saved by skip)\n"
            f"S=topk={S_FULL // 1024}K, nhq={NHQ}, hd={HD}, bf16 | gap={total_gap:.0f}ms",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_ylabel("Time saved by skip (ms)", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(factor_names, fontsize=9, rotation=25, ha="right")
        ax1.legend(fontsize=10)
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.grid(axis="y", alpha=0.2)

        # Gap contribution waterfall
        single_factors = [
            (n, g)
            for n, g in zip(factor_names, gap_contribs)
            if n not in ("no dK+dV store", "skip all")
        ]
        residual = total_gap - sum(g for _, g in single_factors)
        wf_names = [n for n, _ in single_factors] + ["residual\n(tile+reload)"]
        wf_vals = [g for _, g in single_factors] + [residual]

        sort_idx = sorted(range(len(wf_names)), key=lambda i: -wf_vals[i])
        wf_names = [wf_names[i] for i in sort_idx]
        wf_vals = [wf_vals[i] for i in sort_idx]
        wf_colors = ["#1565C0" if "residual" not in n else "#BDBDBD" for n in wf_names]

        ax2.barh(range(len(wf_names)), wf_vals, color=wf_colors, edgecolor="white")
        for i, v in enumerate(wf_vals):
            ax2.text(
                max(v, 0) + 1,
                i,
                f"{v:.0f}ms ({v / total_gap * 100:.0f}%)",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        ax2.set_yticks(range(len(wf_names)))
        ax2.set_yticklabels(wf_names, fontsize=10)
        ax2.set_xlabel("Gap contribution (ms)", fontsize=11)
        ax2.set_title(
            "Gap Attribution\n(LoopK cost − LoopQ cost)", fontsize=12, fontweight="bold"
        )
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.2)

        plt.tight_layout()
        path = os.path.join(out, "loopk_fantasy.png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"[{_ts()}] Plot -> {path}")
    else:
        print("Insufficient data for plot. Run --exp first.")


def _get_ms(results, label):
    key = f"bwd/{label}"
    d = results.get(key, {})
    if S_FULL in d.get("topk", []) and "ms" in d:
        idx = d["topk"].index(S_FULL)
        return d["ms"][idx]
    return None


def _phase4_ncu():
    phase = "4-loopk-fantasy"
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
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,"
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,"
        "smsp__cycles_active.avg.pct_of_peak_sustained_active,"
        "sm__inst_executed.avg.per_cycle_active"
    )

    gpu = _find_free_gpu()
    ncu_configs = [
        ("loopk_baseline", {}, False),
        ("loopk_skipDvStore", {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1"}, False),
        ("loopq_baseline", {}, True),
    ]

    fmt = dict(NHQ=NHQ, NHK=NHK, HD=HD, S=S_FULL, GPU=gpu)
    script_template = """\
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{GPU}"
{env_lines}
from magi_attention.functional import flex_flash_attn_func
torch.manual_seed(42)
S = {S}
q = torch.randn(S, {NHQ}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn(S, {NHK}, {HD}, dtype=torch.bfloat16, device="cuda", requires_grad=True)
q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device="cuda")
atm = torch.zeros(1, dtype=torch.int32, device="cuda")
out, _ = flex_flash_attn_func(q, k, v, q_ranges=q_ranges, k_ranges=k_ranges,
    attn_type_map=atm, pack_gqa=True, swap_bwd_qk_loop={swap_qk})
do = torch.randn_like(out)
out.backward(do)
torch.cuda.synchronize()
print("[DONE]")
"""

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for name, env_overrides, is_loopq in ncu_configs:
        env_lines = "\n".join(
            f'os.environ["{ek}"] = "{ev}"' for ek, ev in env_overrides.items()
        )
        swap_qk = "True" if not is_loopq else "False"
        script_text = script_template.format(
            **fmt, env_lines=env_lines, swap_qk=swap_qk
        )

        script_path = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(script_path, "w") as f:
            f.write(script_text)

        rep_path = os.path.join(out, f"ncu_{name}.ncu-rep")
        csv_path = os.path.join(out, f"ncu_{name}.csv")
        cmd = [
            ncu_bin,
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
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=1200)
        print("done", flush=True)

    print(f"\n[{_ts()}] Phase 4 NCU results in {out}/ncu_*.csv")

    for name, _, _ in ncu_configs:
        csv_path = os.path.join(out, f"ncu_{name}.csv")
        if not os.path.exists(csv_path):
            print(f"  {name}: NOT FOUND")
            continue
        print(f"\n  === {name} ===")
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if any(
                    k in line
                    for k in (
                        "local_op_ld",
                        "local_op_st",
                        "registers_per_thread",
                        "stalled_barrier",
                        "inst_executed",
                        "cycles_active",
                    )
                ):
                    print(f"    {line}")


# ═══════════════════════════════════════════════════════════════
#  Phase 5: seqlen-sweep (S varies, topk=2048 fixed)
# ═══════════════════════════════════════════════════════════════
# Measures how TFLOPS degrades with increasing S (fixed topk=2048).
# For BlockSparse (kbs=128): BWD LoopK atomicAdd RED traffic ∝ S/topk → degrades.
# For IndexSparse (kbs=1): CpAsync scatter + L2 thrashing from large KV pool.
#
# Layout: 2 rows (BlockSparse, IndexSparse) × 3 cols (FWD, BWD-LoopQ, BWD-LoopK)

_P5_TOPK = 2048
_P5_SEQLENS = [32768, 65536, 131072, 262144]
_P5_KBS_BLOCK = 128
_P5_KBS_INDEX = 1


def _phase5_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_block_mask_triton

    phase = "5-seqlen-sweep"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(
        f"[{_ts()}] Phase 5: seqlen-sweep topk={_P5_TOPK} (gpu{gpu})",
        flush=True,
    )

    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]
    METHODS = [
        ("block_sparse", "BlockSparse (kbs=128)"),
        ("index_sparse", "IndexSparse (kbs=1)"),
    ]

    for pass_id, pass_name in PASSES:
        is_bwd = pass_id != "fwd"
        print(f"\n{'=' * 60}\n[{_ts()}] {pass_name}", flush=True)

        for method_id, method_name in METHODS:
            print(f"  {method_name}:", flush=True)
            for S in _P5_SEQLENS:
                key = f"{pass_id}/{method_id}"
                if not force and _has_entry(results, key, S):
                    d = results[key]
                    idx = d["topk"].index(S)
                    print(
                        f"    S={S:>6d}: {d['tflops'][idx]:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(42)

                try:
                    topk = min(_P5_TOPK, S)
                    flops = _calc_flops(S, topk, is_bwd)

                    if method_id == "block_sparse":
                        kbs = _P5_KBS_BLOCK
                        n_q_blocks = S
                        n_k_blocks = S // kbs
                        actual_attend = min(topk // kbs, n_k_blocks)

                        block_mask = torch.zeros(
                            1,
                            NHK,
                            n_q_blocks,
                            n_k_blocks,
                            dtype=torch.bool,
                            device=device,
                        )
                        for qb in range(n_q_blocks):
                            perm = torch.randperm(n_k_blocks, device=device)[
                                :actual_attend
                            ]
                            block_mask[0, 0, qb, perm] = True

                        q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
                            block_mask, 1, kbs
                        )
                        atm = torch.zeros(
                            len(q_ranges), dtype=torch.int32, device=device
                        )
                        q, k, v = _make_tensors(S, device, torch.bfloat16, grad=is_bwd)

                        swap = pass_id == "bwd_loopk"
                        kw = dict(
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            attn_type_map=atm,
                            block_sparse=True,
                            auto_range_merge=True,
                            pack_gqa=True,
                            swap_bwd_qk_loop=swap,
                        )

                    else:  # index_sparse
                        kbs = _P5_KBS_INDEX
                        idx = _build_idx_kbs1(S, topk, device)
                        q, k, v = _make_tensors(S, device, torch.bfloat16, grad=is_bwd)

                        swap = pass_id == "bwd_loopk"
                        kw = dict(
                            index_sparse_indices=idx,
                            k_block_size=kbs,
                            index_sparse=True,
                            pack_gqa=True,
                            swap_bwd_qk_loop=swap,
                        )

                    o, *_ = flex_flash_attn_func(q, k, v, **kw)

                    if not is_bwd:

                        def run_fn():
                            flex_flash_attn_func(q, k, v, **kw)

                    else:
                        do = torch.randn_like(o)

                        def run_fn():
                            o.backward(do, retain_graph=True)

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    _set_entry(results, key, S, round(tf, 1), round(ms, 3))
                    print(f"    S={S:>6d}: {tf:>7.1f} T ({ms:.3f}ms)", flush=True)

                except Exception as e:
                    _set_entry(results, key, S, None, None)
                    print(f"    S={S:>6d}: FAIL - {e}", flush=True)
                finally:
                    q = k = v = None
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(phase, results)

    print(f"\n[{_ts()}] Phase 5 DONE -> {_results_path(phase)}", flush=True)


def _phase5_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = "5-seqlen-sweep"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    PASSES = [("fwd", "FWD"), ("bwd_loopq", "BWD LoopQ"), ("bwd_loopk", "BWD LoopK")]
    METHODS = [
        ("block_sparse", "BlockSparse (kbs=128)", "#1565C0", "-o"),
        ("index_sparse", "IndexSparse (kbs=1)", "#E53935", "-s"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    x_labels = [f"{s // 1024}K" for s in _P5_SEQLENS]

    for col_idx, (pid, pname) in enumerate(PASSES):
        ax = axes[col_idx]
        for method_id, method_name, color, marker in METHODS:
            key = f"{pid}/{method_id}"
            d = results.get(key, {})
            vals = []
            for s in _P5_SEQLENS:
                if s in d.get("topk", []):
                    idx = d["topk"].index(s)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            ax.plot(
                range(len(_P5_SEQLENS)),
                vals,
                marker,
                color=color,
                label=method_name,
                linewidth=2,
                markersize=8,
            )
            for i, v in enumerate(vals):
                if v > 0:
                    ax.annotate(
                        f"{v:.0f}",
                        (i, v),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=8,
                        fontweight="bold",
                        color=color,
                    )

        ax.set_title(f"{pname}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Sequence Length (S)", fontsize=11)
        ax.set_ylabel("TFLOPS", fontsize=11)
        ax.set_xticks(range(len(_P5_SEQLENS)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(600, ax.get_ylim()[1] * 1.1))

    fig.suptitle(
        f"Phase 5: Seqlen Sweep (topk={_P5_TOPK}, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16, PackGQA)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "seqlen_sweep.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot -> {path}")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp", choices=PHASES, help="Run benchmark experiment")
    group.add_argument("--plot", choices=PHASES, help="Generate plot from results")
    group.add_argument("--ncu", choices=PHASES, help="Run NCU profiling")
    parser.add_argument(
        "--force", action="store_true", help="Re-run all (ignore cache)"
    )
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        help="Re-run subset: 'pass/method' or 'pass/method/topk', comma-separated",
    )

    args = parser.parse_args()
    rerun_filter = _parse_rerun(args.rerun) if args.rerun else None

    if args.exp:
        phase = args.exp
        print(f"[{_ts()}] === --exp {phase} ===", flush=True)
        if phase == "0-method-parity":
            _phase0_bench(force=args.force, rerun_filter=rerun_filter)
        elif phase == "1-topk-sweep":
            _phase1_bench(force=args.force, rerun_filter=rerun_filter)
        elif phase == "2-kbs-compare":
            _phase2_bench(force=args.force)
        elif phase == "3-l2-inflection":
            parser.error("Phase 3 has no --exp. Use --ncu 3-l2-inflection")
        elif phase == "4-loopk-fantasy":
            _phase4_bench(force=args.force)
        elif phase == "5-seqlen-sweep":
            _phase5_bench(force=args.force)
    elif args.plot:
        phase = args.plot
        print(f"[{_ts()}] === --plot {phase} ===", flush=True)
        if phase == "0-method-parity":
            _phase0_plot()
        elif phase == "1-topk-sweep":
            _phase1_plot()
        elif phase == "2-kbs-compare":
            _phase2_plot()
        elif phase == "3-l2-inflection":
            parser.error("Phase 3 has no --plot. Use --ncu 3-l2-inflection")
        elif phase == "4-loopk-fantasy":
            _phase4_plot()
        elif phase == "5-seqlen-sweep":
            _phase5_plot()
    elif args.ncu:
        phase = args.ncu
        print(f"[{_ts()}] === --ncu {phase} ===", flush=True)
        if phase == "0-method-parity":
            _phase0_ncu()
        elif phase == "1-topk-sweep":
            parser.error("Phase 1 has no --ncu")
        elif phase == "2-kbs-compare":
            _phase2_ncu()
        elif phase == "3-l2-inflection":
            _phase3_ncu()
        elif phase == "4-loopk-fantasy":
            _phase4_ncu()
        elif phase == "5-seqlen-sweep":
            parser.error("Phase 5 has no --ncu. Use --exp or --plot")

    print(f"\n[{_ts()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
