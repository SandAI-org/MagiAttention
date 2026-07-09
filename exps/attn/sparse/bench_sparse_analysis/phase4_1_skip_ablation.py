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

"""Phase 4-1: skip-ablation — LoopK vs LoopQ gap analysis via perf-debug skip flags.

Fair ablation approach:
- PV MMA always preserved
- Uses non-bypass (SMEM+TMA) path
- SkipVLoad: loads V from block 0 (L2 cached), pipeline intact
- SkipDvStore/SkipDkStore: skip TMA reduce-add only, barrier protocol preserved
- SkipDvMMA: skip dV MMA (unfair but diagnostic)

Baseline = ununion+stgV1 (JIT default for InnerLoopK post-merge 2026-07).
Legacy (union+stgV2) can be restored via MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2=1.

Env vars:
  MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD=1   lightweight V load
  MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE=1 skip dV TMA store
  MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE=1 skip dK TMA store
  MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA=1   skip dV MMA
"""

import gc
import os
import subprocess
import sys
import time

from bench_sparse_analysis._common import (
    HD,
    NHK,
    NHQ,
    S_FULL,
    _bench_kernel,
    _calc_flops,
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

_PHASE = "4-loopk-debug"

_ENV_KEYS = [
    "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA",
    "MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS",
    "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES_DS",
    "MAGI_ATTENTION_FFA_BWD_STAGES_V",
    "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2",
    "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES",
]


def _e(**kw):
    """Shorthand env-override dict builder."""
    _PREFIX = "MAGI_ATTENTION_FFA_BWD_"
    return {_PREFIX + k: v for k, v in kw.items()}


def _skip(**kw):
    return _e(
        **{f"SKIP_{k}": "1" for k in kw},
        **{k: v for k, v in kw.items() if not isinstance(v, bool)},
    )


def _clear_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


# ── Skip factor configs (symmetric LoopK/LoopQ pairs) ────────
_SKIP_ALL = {
    "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
}

# (label_suffix, env, description) — each generates both loopk_ and loopq_ variants
_SKIP_FACTORS = [
    ("baseline", {}, "baseline (ununion+stgV1)"),
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
    ("skip_all", _SKIP_ALL, "skip all"),
]

# ── Extended configs (LoopK only) ─────────────────────────────
# (label, env, is_loopq, description)
_EXTENDED_CONFIGS = [
    # Structural: tile variants
    ("loopk_baseline", {}, False, "LoopK: baseline"),
    (
        "loopk_m64n64",
        {
            "MAGI_ATTENTION_FFA_BWD_TILE_M": "64",
            "MAGI_ATTENTION_FFA_BWD_TILE_N": "64",
        },
        False,
        "LoopK: M64N64",
    ),
    # Structural: skip_all + tile/staging variants
    ("loopk_skipall", _SKIP_ALL, False, "LoopK: skip all"),
    (
        "loopk_skipall_m64n128",
        {
            **_SKIP_ALL,
            "MAGI_ATTENTION_FFA_BWD_TILE_M": "64",
            "MAGI_ATTENTION_FFA_BWD_TILE_N": "128",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_DS": "1",
        },
        False,
        "LoopK: skip all + M64N128",
    ),
    (
        "loopk_skipall_ds2",
        {
            **_SKIP_ALL,
            "MAGI_ATTENTION_FFA_BWD_STAGES_DS": "2",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: skip all + dS=2 stgV1",
    ),
    # Fine-grained dV/dK decomposition
    (
        "loopk_skip_dv_both",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
        },
        False,
        "LoopK: no dV MMA+store",
    ),
    (
        "loopk_skip_dv_both_dk",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: no dV path + no dK store",
    ),
    # Symmetric ablation: progressive removal
    (
        "loopk_skip_dv_writeback",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "InnerLoopK: no dV writeback",
    ),
    (
        "loopk_symmetric",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "InnerLoopK: symmetric (no V, no dV wb)",
    ),
    (
        "loopk_symmetric_no_dk_store",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "InnerLoopK: symmetric + no dK store",
    ),
    # Optimization validation
    (
        "loopk_legacy_union_stgv2",
        {
            "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2": "1",
        },
        False,
        "LoopK: legacy (union+stgV2)",
    ),
    (
        "loopk_ununion_stgv1_svl",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
        },
        False,
        "LoopK: baseline+SVL",
    ),
    (
        "loopk_ununion_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+SVW",
    ),
    (
        "loopk_bypass",
        {
            "MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS": "1",
        },
        False,
        "LoopK: bypass atomicAdd (O2)",
    ),
    # dV/dK symmetry verification
    (
        "loopk_ununion_stgv1_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SKW",
    ),
    (
        "loopk_ununion_stgv1_svs",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+SVS",
    ),
    (
        "loopk_ununion_stgv1_sks",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SKS",
    ),
    (
        "loopk_ununion_stgv1_svw_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SVW+SKW",
    ),
    # Stage alternatives
    (
        "loopk_ununion_stgk1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
        },
        False,
        "LoopK: baseline+stgK1",
    ),
    (
        "loopk_ununion_stgk1_stgv1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: baseline+stgK1+stgV1",
    ),
    (
        "loopk_stgv1_only",
        {
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: stgV1 only (=baseline, redundant)",
    ),
    (
        "loopk_stgk1_only",
        {
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
        },
        False,
        "LoopK: stgK1 only (+JIT ununion+stgV1)",
    ),
    (
        "loopk_ununion_stgk1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+stgK1+SVW",
    ),
    (
        "loopk_ununion_stgk1_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+stgK1V1+SVW",
    ),
]


def _build_all_configs():
    """Build full config list: symmetric skip pairs + extended LoopK-only configs."""
    configs = []
    for suffix, env, desc in _SKIP_FACTORS:
        configs.append((f"loopk_{suffix}", env, False, f"LoopK: {desc}"))
        configs.append((f"loopq_{suffix}", env, True, f"LoopQ: {desc}"))
    configs.extend(_EXTENDED_CONFIGS)
    return configs


ALL_CONFIGS = _build_all_configs()


# ── Benchmark ─────────────────────────────────────────────────
def _phase4_1_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    results = _load_results(_PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 4-1: LoopK vs LoopQ skip ablation (gpu{gpu})")
    print(f"  S=topk={S_FULL}, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16\n", flush=True)

    for label, env_ov, is_loopq, desc in ALL_CONFIGS:
        key = f"bwd/{label}"
        topk = S_FULL
        if not force and _has_entry(results, key, topk):
            d = results[key]
            idx = d["topk"].index(topk)
            print(f"  {desc}: {d['tflops'][idx]:>7.1f} T (cached)", flush=True)
            continue

        gc.collect()
        torch.cuda.empty_cache()
        _clear_env()
        for ek, ev in env_ov.items():
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
            _set_entry(results, key, topk, round(tf, 1), round(ms, 3))
            print(
                f"  {desc}: {tf:>7.1f} T ({ms:.3f}ms, {time.time() - t0:.0f}s)",
                flush=True,
            )
        except Exception as e:
            _set_entry(results, key, topk, None, None)
            print(f"  {desc}: FAIL - {e}", flush=True)
        finally:
            q = k = v = None
            gc.collect()
            torch.cuda.empty_cache()
        _save_results(_PHASE, results)

    _clear_env()
    _print_summary(results)


def _print_summary(results=None):
    if results is None:
        results = _load_results(_PHASE)

    base_ms = _get_val(results, "loopk_baseline", "ms")
    loopq_ms = _get_val(results, "loopq_baseline", "ms")
    total_gap = (base_ms - loopq_ms) if base_ms and loopq_ms else None

    print(f"\n[{_ts()}] Phase 4-1 Summary")
    print(f"  {'Experiment':<38} {'TFLOPS':>7} {'ms':>7} {'gap%':>8}")  # noqa: F541
    print("  " + "─" * 62)
    for label, _, _, desc in ALL_CONFIGS:
        tf, ms = _get_val(results, label, "tflops"), _get_val(results, label, "ms")
        if tf is not None and ms is not None:
            saved = base_ms - ms if base_ms else 0
            frac = (
                f"{saved / total_gap * 100:+.1f}%"
                if total_gap and total_gap > 0
                else ""
            )
            print(f"  {desc:<38} {tf:>5.0f} T {ms:>6.1f}  {frac:>8}")
        elif tf is not None:
            print(f"  {desc:<38} {tf:>5.0f} T {'N/A':>6}  {'':>8}")
        else:
            print(f"  {desc:<38} {'FAIL':>7} {'N/A':>6}  {'':>8}")

    if total_gap:
        print(f"\n  Total LoopK-LoopQ gap: {total_gap:.1f} ms")
    print(f"  Results: {_results_path(_PHASE)}", flush=True)


def _get_val(results, label, field):
    d = results.get(f"bwd/{label}", {})
    if S_FULL in d.get("topk", []) and field in d:
        return d[field][d["topk"].index(S_FULL)]
    return None


# ── Plot ──────────────────────────────────────────────────────
def _phase4_1_plot():
    """Combined optimization landscape + dV/dK symmetry chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results(_PHASE)
    if not results:
        print(f"ERROR: {_results_path(_PHASE)} not found.")
        return

    out = _out_dir(_PHASE)
    os.makedirs(out, exist_ok=True)

    def tf(label):
        return _get_val(results, label, "tflops")

    lk, lq = tf("loopk_baseline"), tf("loopq_baseline")
    legacy = tf("loopk_legacy_union_stgv2")
    svw = tf("loopk_ununion_stgv1_svw")
    skw = tf("loopk_ununion_stgv1_skw")
    svw_skw = tf("loopk_ununion_stgv1_svw_skw")

    if not all([lk, lq]):
        print("Insufficient data.")
        return

    fig, axes = plt.subplots(
        1, 2, figsize=(20, 7), dpi=150, gridspec_kw={"width_ratios": [3, 2]}
    )

    # ── Left: Optimization landscape ──
    ax = axes[0]
    bar_specs = [
        ("LoopQ", lq, "#212121"),
    ]
    if legacy:
        bar_specs.append(("Legacy\n(union+stgV2)", legacy, "#FF8A65"))
    bar_specs.append(("LoopK\nbaseline", lk, "#1565C0"))
    for lbl, key, col in [
        ("baseline+SVL", "loopk_ununion_stgv1_svl", "#42A5F5"),
        ("baseline+SVS", "loopk_ununion_stgv1_svs", "#F57C00"),
        ("baseline+SKS", "loopk_ununion_stgv1_sks", "#FFB74D"),
        ("baseline+SKW", "loopk_ununion_stgv1_skw", "#7B1FA2"),
        ("baseline+SVW", "loopk_ununion_stgv1_svw", "#C62828"),
        ("baseline+SVW+SKW", "loopk_ununion_stgv1_svw_skw", "#880E4F"),
    ]:
        v = tf(key)
        if v:
            bar_specs.append((lbl, v, col))

    x = np.arange(len(bar_specs))
    vals = [s[1] for s in bar_specs]
    colors = [s[2] for s in bar_specs]
    bars = ax.bar(
        x, vals, width=0.65, color=colors, edgecolor="white", linewidth=0.8, alpha=0.88
    )
    for bar, v in zip(bars, vals):
        if v:
            delta = v - lk
            label_txt = f"{v:.0f}T" if abs(delta) < 1 else f"{v:.0f}T\n({delta:+.0f})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 5,
                label_txt,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333",
            )

    ax.axhline(
        y=lk,
        color="#1565C0",
        ls="--",
        lw=1.2,
        alpha=0.5,
        label=f"LoopK baseline ({lk:.0f}T)",
    )
    ax.axhline(
        y=lq,
        color="#212121",
        ls="--",
        lw=1.2,
        alpha=0.6,
        label=f"LoopQ baseline ({lq:.0f}T)",
    )
    ax.set_title(
        f"InnerLoopK Optimization Landscape\nS={S_FULL // 1024}K, nhq={NHQ}, hd={HD}, H100",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("TFLOPS (BWD)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in bar_specs], fontsize=9)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.set_ylim(0, max(v for v in vals if v) * 1.2)

    # ── Right: dV/dK symmetry ──
    ax2 = axes[1]
    sym_specs = [
        ("Baseline", lk, "#9E9E9E"),
        ("Skip dV wb", svw, "#C62828"),
        ("Skip dK wb", skw, "#1565C0"),
        ("Skip both", svw_skw, "#7B1FA2"),
    ]
    sym_specs = [(n, v, c) for n, v, c in sym_specs if v]
    if len(sym_specs) >= 3:
        x2 = np.arange(len(sym_specs))
        bars2 = ax2.bar(
            x2,
            [s[1] for s in sym_specs],
            width=0.55,
            color=[s[2] for s in sym_specs],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        for bar, (_, v, _) in zip(bars2, sym_specs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        if lq:
            ax2.axhline(
                y=lq, color="#2E7D32", ls="--", lw=1.2, label=f"LoopQ ({lq:.0f}T)"
            )
        ax2.set_title("dV/dK Writeback Symmetry", fontsize=13, fontweight="bold")
        ax2.set_ylabel("TFLOPS", fontsize=12)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([s[0] for s in sym_specs], fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_ylim(0, max(s[1] for s in sym_specs) * 1.18)

    plt.tight_layout()
    path = os.path.join(out, "loopk_skip_ablation_summary.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Skip ablation plot -> {path}")

    # Key conclusions
    if svw:
        gap = lq - lk
        print("\n  ── Key Conclusions ──")
        print(
            f"    LoopK baseline: {lk:.0f} T | LoopQ: {lq:.0f} T | Gap: {gap:.0f} T ({gap / lq * 100:.0f}%)"
        )
        print(
            f"    SVW ceiling: +{svw - lk:.0f}T ({(svw - lk) / gap * 100:.0f}% of gap)"
        )
        if svw and skw:
            print(f"    dV wb cost = {svw - lk:.0f}T | dK wb cost = {skw - lk:.0f}T")


# ── NCU ───────────────────────────────────────────────────────
def _phase4_1_ncu():
    out = _out_dir(_PHASE)
    os.makedirs(out, exist_ok=True)

    ncu_bin = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "ncu")

    metrics = (
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,"
        "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,"
        "launch__registers_per_thread,"
        "sm__cycles_elapsed.avg,"
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,"
        "smsp__cycles_active.avg.pct_of_peak_sustained_active,"
        "sm__inst_executed.avg.per_cycle_active,"
        "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,"
        "dram__bytes.sum"
    )

    gpu = _find_free_gpu()
    ncu_configs = [
        ("loopk_baseline", {}, False),
        ("loopk_skipDvStore", {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1"}, False),
        ("loopq_baseline", {}, True),
    ]

    scripts_dir = os.path.join(out, "ncu_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    script_tpl = """\
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
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
"""

    for name, env_ov, is_loopq in ncu_configs:
        env_lines = "\n".join(f'os.environ["{k}"] = "{v}"' for k, v in env_ov.items())
        script = script_tpl.format(
            gpu=gpu,
            env_lines=env_lines,
            swap_qk="True" if not is_loopq else "False",
            S=S_FULL,
            NHQ=NHQ,
            NHK=NHK,
            HD=HD,
        )
        sp = os.path.join(scripts_dir, f"ncu_{name}.py")
        with open(sp, "w") as f:
            f.write(script)

        rep = os.path.join(out, f"ncu_{name}")
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
            rep,
            sys.executable,
            sp,
        ]
        print(f"  [{_ts()}] NCU {name}...", end=" ", flush=True)
        with open(csv_path, "w") as out_f:
            subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, timeout=1200)
        print("done", flush=True)

    print(f"\n[{_ts()}] NCU results in {out}/ncu_*.csv")
