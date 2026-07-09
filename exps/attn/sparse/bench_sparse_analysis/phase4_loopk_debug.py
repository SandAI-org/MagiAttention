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

"""Phase 4: loopk-debug — LoopK vs LoopQ gap analysis with perf-debug skip flags."""

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

# ═══════════════════════════════════════════════════════════════
#  Phase 4: loopk-debug (LoopK vs LoopQ gap analysis — perf-debug skip flags for TFLOPS profiling)
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

_DEBUG_ENV_KEYS = [
    "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA",
    "MAGI_ATTENTION_FFA_BWD_DKV_USE_SMEM",
    "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES_DS",
    "MAGI_ATTENTION_FFA_BWD_STAGES_V",
    "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2",
    "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES",
]

# Symmetric configs: same skip flags on BOTH LoopK and LoopQ.
# Gap_contribution(X) = cost_in_LoopK(X) - cost_in_LoopQ(X)
#
# NOTE (post-merge 2026-07): JIT now auto-applies ununion+stgV1 for InnerLoopK (swap_bwd_qk_loop=True).
# The "baseline" below ({} env) IS ununion+stgV1 — this is the true InnerLoopK baseline.
# Legacy behavior (union+stgV2) can be restored via MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2=1.
_SKIP_FACTORS = [
    # (factor_key, env_overrides, short_name)
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

_DEBUG_CONFIGS = []
for factor_key, env_ov, short in _SKIP_FACTORS:
    _DEBUG_CONFIGS.append((f"loopk_{factor_key}", env_ov, False, f"LoopK: {short}"))
    _DEBUG_CONFIGS.append((f"loopq_{factor_key}", env_ov, True, f"LoopQ: {short}"))

# Phase 4b: structural experiments.
# Goal: when skip flags free SMEM, previously infeasible tile/staging configs may work.
# SMEM budget (H100 limit 228KB):
#   LoopK M128N64 baseline:  198KB (stg=2 stgV=2 stg_dS=1)
#   LoopK M64N128 baseline:  260KB (EXCEEDS by 32KB without skip)
#   LoopK M128N64 stg_dS=2:  230KB (EXCEEDS by 2KB without skip)
# skip_v_load frees ~32KB (smem_v stages), skip_dv may free dvacc buffer.
_STRUCTURAL_CONFIGS = [
    # ── Baseline structural params ──
    ("loopk_baseline", {}, False, "LoopK: baseline"),
    (
        "loopk_m64n64",
        {"MAGI_ATTENTION_FFA_BWD_TILE_M": "64", "MAGI_ATTENTION_FFA_BWD_TILE_N": "64"},
        False,
        "LoopK: M64N64",
    ),
    # ── skip_all + structural (freed SMEM enables new configs) ──
    (
        "loopk_skipall",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
        },
        False,
        "LoopK: skip all",
    ),
    # skip_all + M64N128 + stgV=1 + dS=1: force single-stage dS to fit SMEM
    # (M64N128 heuristic defaults dS=2 → 228KB, barely exceeds with pipeline barriers)
    (
        "loopk_skipall_m64n128",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
            "MAGI_ATTENTION_FFA_BWD_TILE_M": "64",
            "MAGI_ATTENTION_FFA_BWD_TILE_N": "128",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_DS": "1",
        },
        False,
        "LoopK: skip all + M64N128",
    ),
    # skip_all + dS_stage=2 + stgV=1: stgV1 saves 16KB → 230-16=214KB
    (
        "loopk_skipall_ds2",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_DS": "2",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: skip all + dS=2 stgV1",
    ),
    # skip_all + M64N128 + lseU=1 + stgV=1 + dS=1 (closest to LoopQ structural parity)
    # dS=2 not feasible with M64N128 even with all skips (228KB = at limit)
    (
        "loopk_skipall_loopq_struct",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
            "MAGI_ATTENTION_FFA_BWD_TILE_M": "64",
            "MAGI_ATTENTION_FFA_BWD_TILE_N": "128",
            "MAGI_ATTENTION_FFA_BWD_STAGES_DS": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: skip all + M64N128",
    ),
    # ── Fine-grained decomposition ──
    # skip_dv_mma + skip_dv_store together (vs individually) to see interaction
    (
        "loopk_skip_dv_both",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
        },
        False,
        "LoopK: no dV MMA+store",
    ),
    # skip_dv_both + skip_dk_store
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
    # ══════ P5-v2: Symmetric ablation experiments ══════
    # Goal: progressively remove InnerLoopK's extra overhead vs InnerLoopQ, keeping all MMA.
    # Extra overhead = V load (inner) + dV writeback pipeline (R2S+barrier+TMA)
    # Symmetric baseline: InnerLoopQ inner = load Q,dO + MMA(S,P,dS,dV,dK,dQ) + store dQ(atomicAdd)
    #
    # Config A: remove dV writeback only (R2S+barrier+TMA), keep V load
    (
        "loopk_skip_dv_writeback",
        {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1"},
        False,
        "InnerLoopK: no dV writeback",
    ),
    # Config B (core): remove V load + dV writeback -> InnerLoopK ~ symmetric InnerLoopQ
    (
        "loopk_symmetric",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "InnerLoopK: symmetric (no V, no dV wb)",
    ),
    # Config C: remove V load + dV writeback + dK store -> only MMA + K load (upper bound)
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
]

# ══════ P5-v3: Optimization validation ══════
# Post-merge: baseline IS ununion+stgV1 (JIT default for InnerLoopK).
# Legacy config: force union+stgV2 to show the pre-optimization regression.
# Remaining configs: additional optimizations ON TOP of the new baseline.
_OPTIMIZATION_CONFIGS = [
    # Legacy: force old union+stgV2 behavior (pre-merge baseline, ~-38T regression)
    (
        "loopk_legacy_union_stgv2",
        {"MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2": "1"},
        False,
        "LoopK: legacy (union+stgV2)",
    ),
    # baseline + lightweight V load (diagnostic)
    (
        "loopk_ununion_stgv1_svl",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
        },
        False,
        "LoopK: baseline+SVL",
    ),
    # baseline + skip dV writeback (theoretical ceiling — shows dV wb cost)
    (
        "loopk_ununion_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+SVW",
    ),
    # O2: bypass scalar atomicAdd (correct but slow for dense)
    (
        "loopk_bypass",
        {"MAGI_ATTENTION_FFA_BWD_DKV_USE_SMEM": "1"},
        False,
        "LoopK: bypass atomicAdd (O2)",
    ),
]

# ══════ P5-v4: dV/dK symmetry verification ══════
# Goal: verify whether dV and dK writeback pipeline costs are symmetric after ununion.
# NOTE: baseline IS ununion+stgV1 now. All configs below operate on top of baseline.
# If baseline+SVW ~ baseline+SKW, ununion fully deserialized dV/dK -> both paths independent and equivalent.
# If baseline+SVW != baseline+SKW, dV and dK have intrinsic asymmetry (softmax_scale, barrier ordering).
_SYMMETRY_CONFIGS = [
    # baseline + skip dK writeback (symmetric to baseline+SVW)
    (
        "loopk_ununion_stgv1_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SKW",
    ),
    # baseline + skip dV store only (isolate TMA vs full writeback)
    (
        "loopk_ununion_stgv1_svs",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+SVS",
    ),
    # baseline + skip dK store only (symmetric to baseline+SVS)
    (
        "loopk_ununion_stgv1_sks",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SKS",
    ),
    # baseline + skip both writebacks (ceiling: no writeback overhead)
    (
        "loopk_ununion_stgv1_svw_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: baseline+SVW+SKW",
    ),
    # baseline + defer dV R2S after MMA5 (test pipeline reorder)
]

# ══════ P5-v6: Stage alternatives — stgK=1 vs stgV=1 ══════
# NOTE: baseline already applies ununion+stgV1. Some configs below are NOW REDUNDANT:
#   - "stgV1 only" = baseline (JIT default already sets both ununion+stgV1)
#   - "stgK1 only" = ununion+stgK1+stgV1 (JIT adds ununion+stgV1 on top)
# Kept for backward compat with cached results, marked accordingly.
_STAGE_CONFIGS = [
    (
        "loopk_ununion_stgk1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
        },
        False,
        "LoopK: baseline+stgK1",
    ),
    (
        "loopk_ununion_stgk1_stgv1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: baseline+stgK1+stgV1",
    ),
    (
        "loopk_stgv1_only",
        {"MAGI_ATTENTION_FFA_BWD_STAGES_V": "1"},
        False,
        "LoopK: stgV1 only (=baseline, redundant)",
    ),
    (
        "loopk_stgk1_only",
        {"MAGI_ATTENTION_FFA_BWD_STAGES": "1"},
        False,
        "LoopK: stgK1 only (+JIT ununion+stgV1)",
    ),
    (
        "loopk_ununion_stgk1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+stgK1+SVW",
    ),
    (
        "loopk_ununion_stgk1_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "0",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: baseline+stgK1V1+SVW",
    ),
]

_DEBUG_CONFIGS.extend(_STRUCTURAL_CONFIGS)
_DEBUG_CONFIGS.extend(_OPTIMIZATION_CONFIGS)
_DEBUG_CONFIGS.extend(_SYMMETRY_CONFIGS)
_DEBUG_CONFIGS.extend(_STAGE_CONFIGS)


def _phase4_bench(force=False):
    import torch

    from magi_attention.functional import flex_flash_attn_func

    phase = "4-loopk-debug"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 4: LoopK vs LoopQ gap isolation (gpu{gpu})", flush=True)
    print(f"  S=topk={S_FULL}, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16\n", flush=True)

    for label, env_overrides, is_loopq, desc in _DEBUG_CONFIGS:
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
        for env_key in _DEBUG_ENV_KEYS:
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
    for env_key in _DEBUG_ENV_KEYS:
        os.environ.pop(env_key, None)

    print(f"\n[{_ts()}] Phase 4 DONE -> {_results_path(phase)}", flush=True)

    # Print summary table with ms-based gap decomposition
    results = _load_results(phase)
    base_ms = None
    loopq_ms = None
    for label, _, is_loopq, _ in _DEBUG_CONFIGS:
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
    for label, _, _, desc in _DEBUG_CONFIGS:
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
    """Deprecated: symmetric cost comparison was misleading. Use _phase4_summary_plot() instead."""
    print(
        "[SKIP] _phase4_plot() deprecated — use --plot to generate summary + symmetry charts only."
    )


def _get_ms(results, label):
    key = f"bwd/{label}"
    d = results.get(key, {})
    if S_FULL in d.get("topk", []) and "ms" in d:
        idx = d["topk"].index(S_FULL)
        return d["ms"][idx]
    return None


def _phase4_opt_plot():
    """Focused paired bar chart: dV vs dK writeback/store symmetry on baseline (ununion+stgV1).

    Left: Writeback symmetry (R2S + barrier + TMA) — dV vs dK
    Right: Store symmetry (TMA only) — dV vs dK
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "4-loopk-debug"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    def _get_tf(label):
        key = f"bwd/{label}"
        d = results.get(key, {})
        if S_FULL in d.get("topk", []) and d.get("tflops"):
            idx = d["topk"].index(S_FULL)
            return d["tflops"][idx]
        return None

    # baseline IS ununion+stgV1 post-merge; fall back to old key for cached data
    o1_tf = _get_tf("loopk_baseline")
    svw_tf = _get_tf("loopk_ununion_stgv1_svw")
    skw_tf = _get_tf("loopk_ununion_stgv1_skw")
    svs_tf = _get_tf("loopk_ununion_stgv1_svs")
    sks_tf = _get_tf("loopk_ununion_stgv1_sks")
    both_tf = _get_tf("loopk_ununion_stgv1_svw_skw")
    lk_tf = _get_tf("loopk_baseline")
    lq_tf = _get_tf("loopq_baseline")
    legacy_tf = _get_tf("loopk_legacy_union_stgv2")

    if not all([o1_tf, svw_tf, skw_tf, svs_tf, sks_tf]):
        print("Insufficient symmetry data for plot.")
        return

    # RGB color tuples matching phase 0/2 style
    COL_DV = (0.77, 0.34, 0.49)  # dV side (red-pink)
    COL_DK = (0.22, 0.37, 0.71)  # dK side (blue)
    COL_O1 = (0.58, 0.58, 0.58)  # O1 baseline (gray)
    COL_BOTH = (0.45, 0.20, 0.55)  # both (purple)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    bw = 0.30  # noqa: F841

    # ── Left: Writeback symmetry (R2S + barrier + TMA) ──
    groups_wb = [
        "Baseline\n(ununion+stgV1)",
        "Skip dV\nwriteback",
        "Skip dK\nwriteback",
        "Skip\nboth",
    ]
    vals_wb = [o1_tf, svw_tf, skw_tf, both_tf if both_tf else 0]
    cols_wb = [COL_O1, COL_DV, COL_DK, COL_BOTH]

    x_wb = np.arange(len(groups_wb))
    bars_wb = ax1.bar(
        x_wb,
        vals_wb,
        width=0.55,
        color=cols_wb,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    for bar, v in zip(bars_wb, vals_wb):
        if v > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                v + 8,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    if lk_tf:
        ax1.axhline(
            y=lk_tf,
            color=(0.78, 0.22, 0.22),
            linestyle="--",
            linewidth=1.2,
            label=f"LoopK baseline ({lk_tf:.0f}T)",
        )
    if legacy_tf:
        ax1.axhline(
            y=legacy_tf,
            color=(0.85, 0.55, 0.22),
            linestyle="-.",
            linewidth=1.0,
            label=f"Legacy union+stgV2 ({legacy_tf:.0f}T)",
        )
    if lq_tf:
        ax1.axhline(
            y=lq_tf,
            color=(0.20, 0.50, 0.20),
            linestyle="--",
            linewidth=1.2,
            label=f"LoopQ ({lq_tf:.0f}T)",
        )
    ax1.set_title(
        "Writeback Symmetry: dV vs dK\n(R2S + barrier + TMA store)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("TFLOPS", fontsize=12)
    ax1.set_xticks(x_wb)
    ax1.set_xticklabels(groups_wb, fontsize=11)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(vals_wb) * 1.18)

    # ── Right: Store symmetry (TMA only) ──
    groups_st = ["Baseline\n(ununion+stgV1)", "Skip dV\nstore", "Skip dK\nstore"]
    vals_st = [o1_tf, svs_tf, sks_tf]
    cols_st = [COL_O1, COL_DV, COL_DK]

    x_st = np.arange(len(groups_st))
    bars_st = ax2.bar(
        x_st,
        vals_st,
        width=0.55,
        color=cols_st,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    for bar, v in zip(bars_st, vals_st):
        if v > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                v + 8,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    if lk_tf:
        ax2.axhline(
            y=lk_tf,
            color=(0.78, 0.22, 0.22),
            linestyle="--",
            linewidth=1.2,
            label=f"LoopK baseline ({lk_tf:.0f}T)",
        )
    if legacy_tf:
        ax2.axhline(
            y=legacy_tf,
            color=(0.85, 0.55, 0.22),
            linestyle="-.",
            linewidth=1.0,
            label=f"Legacy union+stgV2 ({legacy_tf:.0f}T)",
        )
    if lq_tf:
        ax2.axhline(
            y=lq_tf,
            color=(0.20, 0.50, 0.20),
            linestyle="--",
            linewidth=1.2,
            label=f"LoopQ ({lq_tf:.0f}T)",
        )
    ax2.set_title(
        "Store Symmetry: dV vs dK\n(TMA store only, R2S still runs)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.set_ylabel("TFLOPS", fontsize=12)
    ax2.set_xticks(x_st)
    ax2.set_xticklabels(groups_st, fontsize=11)
    ax2.tick_params(axis="y", labelsize=11)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(vals_st) * 1.18)

    fig.suptitle(
        f"dV/dK Pipeline Symmetry (baseline = ununion+stgV1)\n"
        f"S=topk={S_FULL // 1024}K, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "loopk_optimization_symmetry.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Optimization plot -> {path}")

    # Symmetry comparison table
    svw_ms = _get_ms(results, "loopk_ununion_stgv1_svw")
    skw_ms = _get_ms(results, "loopk_ununion_stgv1_skw")
    svs_ms = _get_ms(results, "loopk_ununion_stgv1_svs")
    sks_ms = _get_ms(results, "loopk_ununion_stgv1_sks")
    if svw_ms and skw_ms:
        print("\n  ── dV/dK Writeback Symmetry ──")
        print(f"    O1+SVW (skip dV writeback): {svw_ms:.1f} ms")
        print(f"    O1+SKW (skip dK writeback): {skw_ms:.1f} ms")
        delta = abs(svw_ms - skw_ms)
        print(f"    Delta: {delta:.1f} ms ({delta / max(svw_ms, skw_ms) * 100:.1f}%)")
    if svs_ms and sks_ms:
        print("\n  ── dV/dK Store Symmetry ──")
        print(f"    O1+SVS (skip dV store): {svs_ms:.1f} ms")
        print(f"    O1+SKS (skip dK store): {sks_ms:.1f} ms")
        delta = abs(svs_ms - sks_ms)
        print(f"    Delta: {delta:.1f} ms ({delta / max(svs_ms, sks_ms) * 100:.1f}%)")


def _phase4_summary_plot():
    """Comprehensive summary: LoopK optimization landscape.

    Baseline = ununion+stgV1 (JIT default post-merge).
    Shows baseline → SVW ceiling, with key experiments and legacy regression.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "4-loopk-debug"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    def _tf(label):
        key = f"bwd/{label}"
        d = results.get(key, {})
        if S_FULL in d.get("topk", []) and d.get("tflops"):
            idx = d["topk"].index(S_FULL)
            return d["tflops"][idx]
        return None

    # Collect key data points
    lk = _tf("loopk_baseline")
    lq = _tf("loopq_baseline")
    legacy = _tf("loopk_legacy_union_stgv2")
    svw = _tf("loopk_ununion_stgv1_svw")
    skw = _tf("loopk_ununion_stgv1_skw")
    svs = _tf("loopk_ununion_stgv1_svs")
    sks = _tf("loopk_ununion_stgv1_sks")
    svw_skw = _tf("loopk_ununion_stgv1_svw_skw")
    ddv = _tf("loopk_ununion_stgv1_ddv")
    bypass = _tf("loopk_bypass")  # noqa: F841
    svl = _tf("loopk_ununion_stgv1_svl")
    # Stage alternatives
    o1k = _tf("loopk_ununion_stgk1")
    o1kv = _tf("loopk_ununion_stgk1_stgv1")

    if not all([lk, lq, svw]):
        print("Insufficient data for summary plot.")
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(22, 8), dpi=150, gridspec_kw={"width_ratios": [3, 2]}
    )

    # ── Left: Optimization landscape bar chart ──
    configs = []
    # Reference: LoopQ (leftmost)
    configs.append(("LoopQ\nbaseline", lq, "#212121", "ref"))
    # Legacy regression (union+stgV2)
    if legacy:
        configs.append(("Legacy\n(union+stgV2)", legacy, "#FF8A65", "legacy"))
    # Tier 0: Stage alternatives (worse than baseline)
    if o1k:
        configs.append(("baseline\n+stgK1", o1k, "#EF9A9A", "bad"))
    if o1kv:
        configs.append(("baseline\n+stgK1V1", o1kv, "#E57373", "bad"))
    # Tier 1: LoopK baseline (= ununion+stgV1)
    configs.append(("LoopK\nbaseline", lk, "#1565C0", "baseline"))
    # Tier 2: baseline variants
    if svl:
        configs.append(("baseline\n+SVL", svl, "#42A5F5", "variant"))
    if ddv:
        configs.append(("baseline\n+DDV", ddv, "#64B5F6", "variant"))
    # Tier 3: Writeback/Store skip (diagnostic ceiling)
    if svs:
        configs.append(("baseline\n+SVS", svs, "#F57C00", "store"))
    if sks:
        configs.append(("baseline\n+SKS", sks, "#FFB74D", "store"))
    if skw:
        configs.append(("baseline\n+SKW", skw, "#7B1FA2", "writeback"))
    configs.append(("baseline\n+SVW", svw, "#C62828", "ceiling"))
    if svw_skw:
        configs.append(("baseline\n+SVW+SKW", svw_skw, "#880E4F", "ceiling"))

    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    x = np.arange(len(configs))
    bars = ax1.bar(
        x,
        values,
        width=0.65,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
    )
    for bar, v, cfg in zip(bars, values, configs):
        if not v:
            continue
        delta = v - lk
        delta_str = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"
        tag = cfg[3]
        if tag in ("baseline", "ref"):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                v + 5,
                f"{v:.0f}T",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#333",
            )
        else:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                v + 5,
                f"{v:.0f}T\n({delta_str})",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333",
            )

    ax1.axhline(
        y=lk,
        color="#1565C0",
        linestyle="--",
        linewidth=1.2,
        alpha=0.5,
        label=f"LoopK baseline ({lk:.0f}T)",
    )
    ax1.axhline(
        y=lq,
        color="#212121",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
        label=f"LoopQ baseline ({lq:.0f}T)",
    )
    if legacy:
        ax1.axhline(
            y=legacy,
            color="#FF8A65",
            linestyle="-.",
            linewidth=1,
            alpha=0.4,
            label=f"Legacy union+stgV2 ({legacy:.0f}T)",
        )
    if svw:
        ax1.axhline(y=svw, color="#C62828", linestyle=":", linewidth=1, alpha=0.4)

    ax1.set_title(
        f"InnerLoopK Optimization Landscape (baseline = ununion+stgV1)\n"
        f"S=topk={S_FULL // 1024}K, nhq={NHQ}, nhk={NHK}, hd={HD}, bf16, H100",
        fontsize=13,
        fontweight="bold",
    )
    ax1.set_ylabel("TFLOPS (BWD)", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_ylim(0, max(v for v in values if v) * 1.2)

    # ── Right: Gap decomposition waterfall ──
    gap = lq - lk if lq and lk else 0
    items = []
    items.append(("LoopK baseline\n(ununion+stgV1)", lk, "#1565C0"))
    if svw:
        items.append(("+ skip dV wb\n(R2S+barr+TMA)", svw - lk, "#C62828"))
    if svw_skw and svw:
        items.append(("+ skip dK wb", svw_skw - svw, "#7B1FA2"))

    running = lk
    labels_wf = []
    vals_wf = []
    bottoms = []
    colors_wf = []

    for name, delta, col in items:
        if name.startswith("LoopK baseline"):
            labels_wf.append(name)
            vals_wf.append(delta)
            bottoms.append(0)
            colors_wf.append(col)
            running = delta
        else:
            labels_wf.append(name)
            vals_wf.append(delta)
            bottoms.append(running)
            colors_wf.append(col)
            running += delta

    labels_wf.append(f"= {running:.0f}T\nvs LoopQ {lq:.0f}T")
    vals_wf.append(0)
    bottoms.append(running)
    colors_wf.append("#2E7D32")

    y_pos = np.arange(len(labels_wf))
    bars_wf = ax2.barh(  # noqa: F841
        y_pos,
        vals_wf,
        left=bottoms,
        color=colors_wf,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
        height=0.6,
    )

    for i, (v, b) in enumerate(zip(vals_wf, bottoms)):
        if v > 0:
            ax2.text(
                b + v + 5,
                i,
                f"+{v:.0f}T",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=colors_wf[i],
            )
        elif i == 0:
            ax2.text(
                b + v / 2,
                i,
                f"{v:.0f}T",
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )

    ax2.axvline(
        x=lq,
        color="#2E7D32",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"LoopQ ({lq:.0f}T)",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_wf, fontsize=10)
    ax2.set_xlabel("TFLOPS", fontsize=11)
    ax2.set_title(
        "Gap Decomposition: LoopK → LoopQ\n(cumulative skip experiments)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.2)
    ax2.set_xlim(0, max(running, lq) * 1.15 if running and lq else 700)

    plt.tight_layout()
    path = os.path.join(out, "loopk_optimization_summary.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Summary plot -> {path}")

    # Print legend
    print("\n  ── Abbreviations ──")
    print("    Baseline = ununion+stgV1 (JIT default for InnerLoopK post-merge)")
    print("    Legacy   = union+stgV2 (pre-merge default, ~-38T regression)")
    print("    SVL      = Skip V Load (debug: lightweight TMA load V)")
    print("    DDV      = Defer dV R2S (move R2S after MMA5)")
    print("    SVS      = Skip dV Store (debug: no TMA reduce-add dV)")
    print("    SKS      = Skip dK Store (debug: no TMA reduce-add dK)")
    print("    SKW      = Skip dK Writeback (debug: no dK R2S+barrier+TMA)")
    print("    SVW      = Skip dV Writeback (debug: no dV R2S+barrier+TMA)")

    # Print key conclusions
    print("\n  ── Key Conclusions ──")
    print(f"    LoopK baseline (ununion+stgV1): {lk:.0f} TFLOPS")
    print(f"    LoopQ baseline:                 {lq:.0f} TFLOPS")
    gap = lq - lk
    print(
        f"    Gap:                            {gap:.0f} TFLOPS ({gap / lq * 100:.0f}% of LoopQ)"
    )
    if legacy:
        print(
            f"    Legacy (union+stgV2):           {legacy:.0f}T ({legacy - lk:+.0f}T vs baseline)"
        )
    if svw:
        print(
            f"    SVW ceiling (no dV wb):         +{svw - lk:.0f}T ({(svw - lk) / gap * 100:.0f}% of gap) — debug only"
        )
    print("    Root cause: 2 writebacks/iter (dV+dK) vs 1 (dQ) in LoopQ")
    print(
        f"    dV writeback (R2S+barrier+TMA) = {(svw - lk):.0f}T of gap" if svw else ""
    )


def _phase4_ncu():
    phase = "4-loopk-debug"
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
        "l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum,"
        "launch__registers_per_thread,"
        "sm__cycles_elapsed.avg,"
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,"
        "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio,"
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,"
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
#  Phase 4 ISS: InnerStoreStages sparse LoopK benchmark (multi-scale)
# ═══════════════════════════════════════════════════════════════

_ISS_CONFIGS = [
    # (label, env_overrides, short_name)
    (
        "iss_m128n64_iss1_ud0",
        {"MAGI_ATTENTION_FFA_BWD_STAGES_V": "1"},
        "M128N64 ISS=1 UD=0 (baseline)",
    ),
    (
        "iss_m128n64_iss1_ud1",
        {
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "1",
        },
        "M128N64 ISS=1 UD=1 (union)",
    ),
    (
        "iss_m128n64_iss2_ud1",
        {
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM": "1",
            "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES": "2",
        },
        "M128N64 ISS=2 UD=1 (union+DB)",
    ),
]

_ISS_KVSEQLENS = [32768, 65536, 131072, 262144, 524288]
_ISS_QSEQLEN = 8192


def _phase4_iss_bench(force=False):
    """Benchmark ISS=1 vs ISS=2 for sparse LoopK at multiple kvseqlens."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    phase = "4-loopk-debug"
    results = _load_results(phase)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 4 ISS: sparse LoopK double-buffer evaluation (gpu{gpu})")
    print(
        f"  qseqlen={_ISS_QSEQLEN}, nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, "
        f"block_sparse, pack_gqa, bf16"
    )
    print(f"  kvseqlens: {[s // 1024 for s in _ISS_KVSEQLENS]}K\n", flush=True)

    for label, env_overrides, desc in _ISS_CONFIGS:
        print(f"  ── {desc} ──", flush=True)

        for env_key in _DEBUG_ENV_KEYS:
            os.environ.pop(env_key, None)
        for ek, ev in env_overrides.items():
            os.environ[ek] = ev

        for kvseqlen in _ISS_KVSEQLENS:
            topk = kvseqlen // 4
            key = f"bwd_iss/{label}"

            if not force and _has_entry(results, key, kvseqlen):
                d = results[key]
                idx = d["topk"].index(kvseqlen)
                print(
                    f"    kvseqlen={kvseqlen // 1024}K: "
                    f"{d['tflops'][idx]:>7.1f} T (cached)",
                    flush=True,
                )
                continue

            gc.collect()
            torch.cuda.empty_cache()

            try:
                n_kv_blocks = kvseqlen // KBS
                n_topk_blocks = topk // KBS

                torch.manual_seed(42)
                q = torch.randn(
                    _ISS_QSEQLEN,
                    NHQ,
                    HD,
                    dtype=torch.bfloat16,
                    device=device,
                    requires_grad=True,
                )
                k = torch.randn(
                    kvseqlen,
                    NHK,
                    HD,
                    dtype=torch.bfloat16,
                    device=device,
                    requires_grad=True,
                )
                v = torch.randn(
                    kvseqlen,
                    NHK,
                    HD,
                    dtype=torch.bfloat16,
                    device=device,
                    requires_grad=True,
                )

                gen = torch.Generator().manual_seed(42)
                rand_vals = torch.rand(_ISS_QSEQLEN, n_kv_blocks, generator=gen)
                perms = rand_vals.argsort(dim=1)[:, :n_topk_blocks].sort(dim=1).values
                indices = (
                    perms.unsqueeze(1)
                    .expand(-1, NHK, -1)
                    .to(torch.int32)
                    .to(device)
                    .contiguous()
                )
                ia_3d = indices.permute(1, 0, 2).contiguous()
                q_ranges, k_ranges = generate_ranges_from_topk_indices(
                    ia_3d, block_m=1, block_n=KBS, num_k_blocks=n_kv_blocks
                )
                atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)

                kw = dict(
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=atm,
                    pack_gqa=True,
                    block_sparse=True,
                    sparse_k_block_size=KBS,
                    swap_bwd_qk_loop=True,
                )

                t0 = time.time()
                o, *_ = flex_flash_attn_func(q, k, v, **kw)
                do = torch.randn_like(o)
                flops = _calc_flops(_ISS_QSEQLEN, topk, True)

                def run_fn():
                    o.backward(do, retain_graph=True)

                tf, ms = _bench_kernel(run_fn, flops, device)
                elapsed = time.time() - t0
                _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                print(
                    f"    kvseqlen={kvseqlen // 1024}K: "
                    f"{tf:>7.1f} T ({ms:.3f}ms, {elapsed:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                _set_entry(results, key, kvseqlen, None, None)
                print(f"    kvseqlen={kvseqlen // 1024}K: FAIL - {e}", flush=True)
            finally:
                q = k = v = None
                gc.collect()
                torch.cuda.empty_cache()

            _save_results(phase, results)

    for env_key in _DEBUG_ENV_KEYS:
        os.environ.pop(env_key, None)

    # Summary table
    print(f"\n[{_ts()}] Phase 4 ISS Summary")
    print(
        "  ╔═══════════════════════════════════╦"
        + "═════════╦" * len(_ISS_KVSEQLENS)
        + "╗"
    )
    header = "  ║ Config                            ║"
    for s in _ISS_KVSEQLENS:
        header += f" {s // 1024:>5}K ║"
    print(header)
    print(
        "  ╠═══════════════════════════════════╬"
        + "═════════╬" * len(_ISS_KVSEQLENS)
        + "╣"
    )

    baseline_tfs = {}
    for label, _, desc in _ISS_CONFIGS:
        key = f"bwd_iss/{label}"
        d = results.get(key, {})
        row = f"  ║ {desc[:33]:<33s} ║"
        for kvseqlen in _ISS_KVSEQLENS:
            if kvseqlen in d.get("topk", []):
                idx = d["topk"].index(kvseqlen)
                tf = d["tflops"][idx]
                if tf is not None:
                    if label == "iss_m128n64_iss1_ud0":
                        baseline_tfs[kvseqlen] = tf
                    row += f" {tf:>5.0f} T ║"
                else:
                    row += "  FAIL  ║"
            else:
                row += "   N/A  ║"
        print(row)

    print(
        "  ╚═══════════════════════════════════╩"
        + "═════════╩" * len(_ISS_KVSEQLENS)
        + "╝"
    )

    if baseline_tfs:
        print("\n  ── Delta vs M128N64 baseline ──")
        for label, _, desc in _ISS_CONFIGS:
            if label == "iss_m128n64_iss1_ud0":
                continue
            key = f"bwd_iss/{label}"
            d = results.get(key, {})
            row = f"    {desc[:35]:<35s}"
            for kvseqlen in _ISS_KVSEQLENS:
                base = baseline_tfs.get(kvseqlen)
                if (
                    base
                    and kvseqlen in d.get("topk", [])
                    and d["tflops"][d["topk"].index(kvseqlen)] is not None
                ):
                    idx = d["topk"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    delta = (tf - base) / base * 100
                    row += f"  {delta:>+5.1f}%"
                else:
                    row += "    N/A"
            print(row)

    print(f"\n[{_ts()}] Phase 4 ISS DONE -> {_results_path(phase)}", flush=True)


def _phase4_iss_plot():
    """Grouped bar chart: ISS configs at multiple kvseqlens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    phase = "4-loopk-debug"
    results = _load_results(phase)
    if not results:
        print(f"ERROR: {_results_path(phase)} not found. Run --exp first.")
        return

    out = _out_dir(phase)
    os.makedirs(out, exist_ok=True)

    configs_with_data = []
    for label, _, desc in _ISS_CONFIGS:
        key = f"bwd_iss/{label}"
        d = results.get(key, {})
        if d and "topk" in d:
            tfs = []
            for kvseqlen in _ISS_KVSEQLENS:
                if kvseqlen in d["topk"]:
                    idx = d["topk"].index(kvseqlen)
                    tfs.append(d["tflops"][idx])
                else:
                    tfs.append(None)
            configs_with_data.append((label, desc, tfs))

    if not configs_with_data:
        print("No ISS data found. Run --exp 4-loopk-debug --iss first.")
        return

    colors = ["#1565C0", "#7B1FA2", "#C62828", "#2E7D32", "#FF8F00"]
    n_groups = len(_ISS_KVSEQLENS)
    n_bars = len(configs_with_data)
    bar_w = 0.75 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    for i, (label, desc, tfs) in enumerate(configs_with_data):
        vals = [tf if tf is not None else 0 for tf in tfs]
        offset = (i - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_w * 0.9,
            label=desc,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 3,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_title(
        "InnerStoreStages Double-Buffer: Sparse LoopK BWD\n"
        f"qseqlen={_ISS_QSEQLEN}, topk=kvseqlen/4, nhq={NHQ}, kbs={KBS}, "
        f"pack_gqa, bf16, H100",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("TFLOPS (BWD)", fontsize=12)
    ax.set_xlabel("kvseqlen", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s // 1024}K" for s in _ISS_KVSEQLENS], fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    valid_max = max(
        (tf for _, _, tfs in configs_with_data for tf in tfs if tf), default=100
    )
    ax.set_ylim(0, valid_max * 1.15)

    plt.tight_layout()
    path = os.path.join(out, "iss_double_buffer_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] ISS plot -> {path}")
