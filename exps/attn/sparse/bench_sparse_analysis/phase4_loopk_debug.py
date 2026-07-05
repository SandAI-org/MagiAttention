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
    "MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS",
    "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES_DS",
    "MAGI_ATTENTION_FFA_BWD_LSE_UNION",
    "MAGI_ATTENTION_FFA_BWD_STAGES_V",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK",
    "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK",
    "MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S",
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
    ("loopk_lseu1", {"MAGI_ATTENTION_FFA_BWD_LSE_UNION": "1"}, False, "LoopK: lseU=1"),
    (
        "loopk_m64n64",
        {"MAGI_ATTENTION_FFA_BWD_TILE_M": "64", "MAGI_ATTENTION_FFA_BWD_TILE_N": "64"},
        False,
        "LoopK: M64N64",
    ),
    # ── skip_all + structural (freed SMEM enables new configs) ──
    (
        "loopk_skipall_lseu1",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA": "1",
            "MAGI_ATTENTION_FFA_BWD_LSE_UNION": "1",
        },
        False,
        "LoopK: skip all + lseU1",
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
            "MAGI_ATTENTION_FFA_BWD_LSE_UNION": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: skip all + M64N128 lseU1",
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
    # ══════ P5-v2: 正确的对称实验 ══════
    # 目标：逐步消除 InnerLoopK 相对 InnerLoopQ 的额外开销，保留全部 MMA（公平比较）。
    # 额外开销 = V load (inner) + dV writeback pipeline (R2S+barrier+TMA)
    # 对称基准：InnerLoopQ inner = load Q,dO + MMA(S,P,dS,dV,dK,dQ) + store dQ(atomicAdd)
    #
    # Config A: 仅消除 dV writeback（R2S+barrier+TMA），保留 V load
    (
        "loopk_skip_dv_writeback",
        {"MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1"},
        False,
        "InnerLoopK: no dV writeback",
    ),
    # Config B (核心): 消除 V load + dV writeback → InnerLoopK ≈ symmetric InnerLoopQ
    (
        "loopk_symmetric",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
        },
        False,
        "InnerLoopK: symmetric (no V, no dV wb)",
    ),
    # Config C: 消除 V load + dV writeback + dK store → 只剩 MMA 和 K load（上界）
    (
        "loopk_symmetric_no_dk_store",
        {
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "InnerLoopK: symmetric + no dK store",
    ),
]

# ══════ P5-v3: 实际优化方案验证 ══════
# O1: ununion SMEM — dV/dK 使用独立 SMEM buffer → 解除串行化 → dV TMA 和 dK R2S 可并行
# O2: DkvaccBypassSmem — register atomicAdd to GMEM (dense = scalar, scatter = per-element)
_OPTIMIZATION_CONFIGS = [
    # O1: ununion + stgV=1 (214KB fits 228KB limit)
    (
        "loopk_ununion_stgv1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: ununion+stgV1 (O1)",
    ),
    # O1 + SVL: ununion with lightweight V load
    (
        "loopk_ununion_stgv1_svl",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD": "1",
        },
        False,
        "LoopK: ununion+stgV1+SVL",
    ),
    # O1 + SVW: ununion + skip dV writeback (theoretical ceiling of ununion)
    (
        "loopk_ununion_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
        },
        False,
        "LoopK: ununion+stgV1+SVW",
    ),
    # O2: bypass scalar atomicAdd (correct but slow for dense)
    (
        "loopk_bypass",
        {"MAGI_ATTENTION_FFA_BWD_DKVACC_BYPASS": "1"},
        False,
        "LoopK: bypass atomicAdd (O2)",
    ),
]

# ══════ P5-v4: dV/dK 对称性验证 ══════
# 目标：验证 ununion 后 dV 和 dK writeback pipeline 的开销是否对称。
# 如果 O1+SVW ≈ O1+SKW，说明 ununion 消除了 dV/dK 串行化 → 两条路径独立且等价。
# 如果 O1+SVW ≠ O1+SKW，说明 dV 和 dK 有内在的非对称性（如 softmax_scale、barrier 顺序）。
_SYMMETRY_CONFIGS = [
    # O1 + skip dK writeback (symmetric to O1 + SVW)
    (
        "loopk_ununion_stgv1_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK": "1",
        },
        False,
        "LoopK: ununion+stgV1+SKW",
    ),
    # O1 + skip dV store only (isolate TMA vs full writeback)
    (
        "loopk_ununion_stgv1_svs",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE": "1",
        },
        False,
        "LoopK: ununion+stgV1+SVS",
    ),
    # O1 + skip dK store only (symmetric to O1+SVS)
    (
        "loopk_ununion_stgv1_sks",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE": "1",
        },
        False,
        "LoopK: ununion+stgV1+SKS",
    ),
    # O1 + skip both writebacks (ceiling: no writeback overhead)
    (
        "loopk_ununion_stgv1_svw_skw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK": "1",
        },
        False,
        "LoopK: ununion+stgV1+SVW+SKW",
    ),
    # O1 + defer dV R2S after MMA5 (test pipeline reorder)
    (
        "loopk_ununion_stgv1_ddv",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S": "1",
        },
        False,
        "LoopK: ununion+stgV1+DeferDvR2S",
    ),
]

# ══════ P5-v6: Stage alternatives — stgK=1 vs stgV=1 ══════
# Ununion needs +32KB → 230KB > 228KB. We freed SMEM via stgV=1 (O1).
# But stgK=1 (Stages=1, K pipeline 2→1) saves the same 16KB.
# Question: is stgK=1 better/worse than stgV=1?
# Also: stgK=1+stgV=1 saves 32KB → ununion fits at baseline 198KB!
_STAGE_CONFIGS = [
    (
        "loopk_ununion_stgk1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
        },
        False,
        "LoopK: ununion+stgK1",
    ),
    (
        "loopk_ununion_stgk1_stgv1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
        },
        False,
        "LoopK: ununion+stgK1+stgV1",
    ),
    (
        "loopk_stgv1_only",
        {"MAGI_ATTENTION_FFA_BWD_STAGES_V": "1"},
        False,
        "LoopK: stgV1 only (no ununion)",
    ),
    (
        "loopk_stgk1_only",
        {"MAGI_ATTENTION_FFA_BWD_STAGES": "1"},
        False,
        "LoopK: stgK1 only (no ununion)",
    ),
    (
        "loopk_ununion_stgk1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
        },
        False,
        "LoopK: ununion+stgK1+SVW",
    ),
    (
        "loopk_ununion_stgk1_stgv1_svw",
        {
            "MAGI_ATTENTION_FFA_BWD_UNUNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
            "MAGI_ATTENTION_FFA_BWD_STAGES_V": "1",
            "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK": "1",
        },
        False,
        "LoopK: ununion+stgK1V1+SVW",
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
    """Focused paired bar chart: dV vs dK writeback/store symmetry on O1 (ununion+stgV1).

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

    # Collect data
    o1_tf = _get_tf("loopk_ununion_stgv1")
    svw_tf = _get_tf("loopk_ununion_stgv1_svw")
    skw_tf = _get_tf("loopk_ununion_stgv1_skw")
    svs_tf = _get_tf("loopk_ununion_stgv1_svs")
    sks_tf = _get_tf("loopk_ununion_stgv1_sks")
    both_tf = _get_tf("loopk_ununion_stgv1_svw_skw")
    lk_tf = _get_tf("loopk_baseline")
    lq_tf = _get_tf("loopq_baseline")

    if not all([o1_tf, svw_tf, skw_tf, svs_tf, sks_tf]):
        print("Insufficient symmetry data for plot.")
        return

    # RGB color tuples matching phase 0/2 style
    COL_DV = (0.77, 0.34, 0.49)  # dV side (red-pink)
    COL_DK = (0.22, 0.37, 0.71)  # dK side (blue)
    COL_O1 = (0.58, 0.58, 0.58)  # O1 baseline (gray)
    COL_BOTH = (0.45, 0.20, 0.55)  # both (purple)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)
    bw = 0.30

    # ── Left: Writeback symmetry (R2S + barrier + TMA) ──
    groups_wb = [
        "O1\nbaseline",
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
            label=f"LoopK no-opt ({lk_tf:.0f}T)",
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
    groups_st = ["O1\nbaseline", "Skip dV\nstore", "Skip dK\nstore"]
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
            label=f"LoopK no-opt ({lk_tf:.0f}T)",
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
        f"dV/dK Pipeline Symmetry on O1 (ununion+stgV1)\n"
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

    Shows 3 tiers: baseline → O1 (ununion) → SVW ceiling, with key experiments
    annotated to show what works, what doesn't, and why.
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
    o1 = _tf("loopk_ununion_stgv1")
    svw = _tf("loopk_ununion_stgv1_svw")
    skw = _tf("loopk_ununion_stgv1_skw")
    svs = _tf("loopk_ununion_stgv1_svs")
    sks = _tf("loopk_ununion_stgv1_sks")
    svw_skw = _tf("loopk_ununion_stgv1_svw_skw")
    ddv = _tf("loopk_ununion_stgv1_ddv")
    bypass = _tf("loopk_bypass")
    svl = _tf("loopk_ununion_stgv1_svl")
    # Stage alternatives
    o1k = _tf("loopk_ununion_stgk1")
    o1kv = _tf("loopk_ununion_stgk1_stgv1")
    stgv1_only = _tf("loopk_stgv1_only")
    stgk1_only = _tf("loopk_stgk1_only")

    if not all([lk, lq, o1, svw]):
        print("Insufficient data for summary plot.")
        return

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(22, 8), dpi=150, gridspec_kw={"width_ratios": [3, 2]}
    )

    # ── Left: Optimization landscape bar chart ──
    configs = []
    # Reference baseline (leftmost, dark)
    configs.append(("LoopQ\nbaseline", lq, "#212121", "ref"))
    # Tier 0: Stage alternatives (bad ones)
    if stgk1_only:
        configs.append(("stgK1\nonly", stgk1_only, "#FFCDD2", "bad"))
    if o1k:
        configs.append(("ununion\n+stgK1", o1k, "#EF9A9A", "bad"))
    if o1kv:
        configs.append(("ununion\n+stgK1V1", o1kv, "#E57373", "bad"))
    # Tier 1: LoopK baseline
    configs.append(("LoopK\nbaseline", lk, "#D32F2F", "baseline"))
    if stgv1_only:
        configs.append(("stgV1\nonly", stgv1_only, "#90CAF9", "neutral"))
    # Tier 2: O1 variants
    configs.append(("O1\n(ununion\n+stgV1)", o1, "#1565C0", "O1"))
    if svl:
        configs.append(("O1\n+SVL", svl, "#42A5F5", "O1"))
    if ddv:
        configs.append(("O1\n+DDV", ddv, "#64B5F6", "O1"))
    # Tier 3: Writeback/Store skip
    if svs:
        configs.append(("O1\n+SVS", svs, "#F57C00", "store"))
    if sks:
        configs.append(("O1\n+SKS", sks, "#FFB74D", "store"))
    if skw:
        configs.append(("O1\n+SKW", skw, "#7B1FA2", "writeback"))
    configs.append(("O1\n+SVW", svw, "#C62828", "ceiling"))
    if svw_skw:
        configs.append(("O1\n+SVW\n+SKW", svw_skw, "#880E4F", "ceiling"))

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
                color="white" if tag == "ref" else "#333",
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
        color="#D32F2F",
        linestyle="--",
        linewidth=1,
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
    if svw:
        ax1.axhline(y=svw, color="#C62828", linestyle=":", linewidth=1, alpha=0.4)

    # Annotate stgV1-only ablation story
    if stgv1_only:
        stgv1_idx = next(i for i, c in enumerate(configs) if c[0].startswith("stgV1"))
        ax1.annotate(
            "V stage 2→1: perf neutral\n(-16KB SMEM saved)\n→ V load NOT bottleneck",
            xy=(stgv1_idx, stgv1_only),
            xytext=(stgv1_idx + 1.5, stgv1_only + 60),
            fontsize=9,
            fontweight="bold",
            color="#1565C0",
            arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5),
            ha="center",
        )

    ax1.set_title(
        f"InnerLoopK Optimization Landscape\n"
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
    items.append(("LoopK baseline", lk, "#D32F2F"))
    if o1:
        items.append(("+ O1 ununion", o1 - lk, "#1565C0"))
    if svw and o1:
        items.append(("+ skip dV wb\n(R2S+barr+TMA)", svw - o1, "#C62828"))
    if svw_skw and svw:
        items.append(("+ skip dK wb", svw_skw - svw, "#7B1FA2"))

    running = lk
    labels_wf = []
    vals_wf = []
    bottoms = []
    colors_wf = []

    for name, delta, col in items:
        if name == "LoopK baseline":
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
    bars_wf = ax2.barh(
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
        "Gap Decomposition: LoopK → LoopQ\n(cumulative optimization)",
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
    print("    O1    = Ununion dKV accumulators + dS_stage=1 (landable optimization)")
    print("    SVL   = Skip V Load (debug: lightweight TMA load V)")
    print("    DDV   = Defer dV R2S (move R2S after MMA5)")
    print("    SVS   = Skip dV Store (debug: no TMA reduce-add dV)")
    print("    SKS   = Skip dK Store (debug: no TMA reduce-add dK)")
    print("    SKW   = Skip dK Writeback (debug: no dK R2S+barrier+TMA)")
    print("    SVW   = Skip dV Writeback (debug: no dV R2S+barrier+TMA)")

    # Print key conclusions
    print("\n  ── Key Conclusions ──")
    print(f"    LoopK baseline:         {lk:.0f} TFLOPS")
    print(f"    LoopQ baseline:         {lq:.0f} TFLOPS")
    print(
        f"    Gap:                    {gap:.0f} TFLOPS ({gap / lq * 100:.0f}% of LoopQ)"
    )
    if stgv1_only:
        print(
            f"    stgV1 only (no ununion): {stgv1_only - lk:+.0f}T (noise) — V load stages perf-neutral, saves 16KB SMEM"
        )
    if o1:
        ununion_delta = (o1 - stgv1_only) if stgv1_only else (o1 - lk)
        print(
            f"    O1 (ununion+stgV1):     +{o1 - lk:.0f}T ({(o1 - lk) / gap * 100:.0f}% of gap) — landable"
        )
        if stgv1_only:
            print(f"      ├─ stgV1 贡献: +{stgv1_only - lk:.0f}T (V pipeline 2→1)")
            print(f"      └─ ununion 贡献: +{ununion_delta:.0f}T (dKV accumulator 分离)")
    if svw:
        print(
            f"    SVW ceiling (no dV wb): +{svw - lk:.0f}T ({(svw - lk) / gap * 100:.0f}% of gap) — debug only"
        )
    print("    Root cause: 2 writebacks/iter (dV+dK) vs 1 (dQ) in LoopQ")
    print(
        f"    dV writeback (R2S+barrier+TMA) = {(svw - o1):.0f}T of gap"
        if svw and o1
        else ""
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
