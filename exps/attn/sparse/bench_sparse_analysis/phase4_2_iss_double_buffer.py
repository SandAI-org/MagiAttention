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

"""Phase 4-2: ISS double-buffer — InnerStoreStages / union / stg / tile ablation.

Evaluates dKVacc double-buffer (ISS=2) and related SMEM trade-offs for LoopK BWD.
Separated from phase4_1 (skip-flag gap analysis) to keep concerns independent.

Configs are defined as (label, env_overrides, short_description).
All configs use sparse LoopK (swap_bwd_qk_loop=True) with PackGQA.
"""

import gc
import os
import time

from bench_sparse_analysis._common import (
    HD,
    KBS,
    NHK,
    NHQ,
    _bench_kernel,
    _calc_flops,
    _has_entry,
    _load_results,
    _out_dir,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)

_PHASE = "4_2-iss-double-buffer"

_ENV_KEYS = [
    "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC",
    "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES",
    "MAGI_ATTENTION_FFA_BWD_STAGES_V",
    "MAGI_ATTENTION_FFA_BWD_STAGES_DS",
    "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2",
    "MAGI_ATTENTION_FFA_BWD_TILE_M",
    "MAGI_ATTENTION_FFA_BWD_TILE_N",
]

# ── Config definitions ────────────────────────────────────────
# (label, env_overrides, description, smem_kb)
ISS_CONFIGS = [
    ("iss1_ud0_stg2", {}, "ISS1 UD0 stg2 (baseline)", 214),
    (
        "iss1_ud1_stg2",
        {"MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "1"},
        "ISS1 UD1 stg2 (union)",
        182,
    ),
    (
        "iss2_ud1_stg2",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES": "2",
        },
        "ISS2 UD1 stg2 (union+DB)",
        214,
    ),
    (
        "iss1_ud0_stg1",
        {"MAGI_ATTENTION_FFA_BWD_STAGES": "1"},
        "ISS1 UD0 stg1 (no K pipe)",
        198,
    ),
    (
        "iss2_ud1_stg1",
        {
            "MAGI_ATTENTION_FFA_BWD_UNION_DKVACC": "1",
            "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES": "2",
            "MAGI_ATTENTION_FFA_BWD_STAGES": "1",
        },
        "ISS2 UD1 stg1",
        198,
    ),
    (
        "legacy_union_stgv2",
        {"MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV2": "1"},
        "legacy (union+stgV2)",
        198,
    ),
]

KVSEQLENS = [32768, 65536, 131072, 262144]
QSEQLEN = 8192


def _clear_env():
    for k in _ENV_KEYS:
        os.environ.pop(k, None)


def _phase4_2_bench(force=False):
    """Benchmark all ISS/union/stg configs at multiple kvseqlens."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    results = _load_results(_PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"
    print(f"[{_ts()}] Phase 4-2: ISS double-buffer eval (gpu{gpu})")
    print(
        f"  qseqlen={QSEQLEN}, nhq={NHQ}, nhk={NHK}, hd={HD}, kbs={KBS}, pack_gqa, bf16"
    )
    print(f"  kvseqlens: {[s // 1024 for s in KVSEQLENS]}K\n", flush=True)

    for label, env_ov, desc, smem in ISS_CONFIGS:
        print(f"  ── {desc} ({smem}KB) ──", flush=True)
        _clear_env()
        for ek, ev in env_ov.items():
            os.environ[ek] = ev

        for kvseqlen in KVSEQLENS:
            topk = kvseqlen // 4
            key = f"bwd/{label}"

            if not force and _has_entry(results, key, kvseqlen):
                d = results[key]
                idx = d["topk"].index(kvseqlen)
                print(
                    f"    {kvseqlen // 1024}K: {d['tflops'][idx]:>7.1f} TF (cached)",
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
                    QSEQLEN,
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
                perms = (
                    torch.rand(QSEQLEN, n_kv_blocks, generator=gen)
                    .argsort(1)[:, :n_topk_blocks]
                    .sort(1)
                    .values
                )
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

                t0 = time.time()
                o, *_ = flex_flash_attn_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=atm,
                    pack_gqa=True,
                    block_sparse=True,
                    sparse_k_block_size=KBS,
                    swap_bwd_qk_loop=True,
                )
                do = torch.randn_like(o)
                flops = _calc_flops(QSEQLEN, topk, True)

                def run_fn():
                    o.backward(do, retain_graph=True)

                tf, ms = _bench_kernel(run_fn, flops, device)
                _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                print(
                    f"    {kvseqlen // 1024}K: {tf:>7.1f} TF ({ms:.3f}ms, {time.time() - t0:.0f}s)",
                    flush=True,
                )
            except Exception as e:
                _set_entry(results, key, kvseqlen, None, None)
                print(f"    {kvseqlen // 1024}K: FAIL - {e}", flush=True)
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
    print(f"\n[{_ts()}] Phase 4-2 Summary (TFLOPS)")
    hdr = f"  {'Config':<30} {'SMEM':>5}"
    for s in KVSEQLENS:
        hdr += f" {s // 1024:>6}K"
    print(hdr)
    print("  " + "─" * (35 + 8 * len(KVSEQLENS)))

    base_tfs = {}
    for label, _, desc, smem in ISS_CONFIGS:
        d = results.get(f"bwd/{label}", {})
        row = f"  {desc:<30} {smem:>4}K"
        for kvseqlen in KVSEQLENS:
            if kvseqlen in d.get("topk", []):
                tf = d["tflops"][d["topk"].index(kvseqlen)]
                if tf is not None:
                    row += f" {tf:>6.0f}"
                    if label == "iss1_ud0_stg2":
                        base_tfs[kvseqlen] = tf
                else:
                    row += "   FAIL"
            else:
                row += "    N/A"
        print(row)

    if base_tfs:
        print(f"\n  {'Δ vs baseline':<30} {'':>5}", end="")
        for s in KVSEQLENS:
            print(f" {s // 1024:>6}K", end="")
        print()
        for label, _, desc, _ in ISS_CONFIGS:
            if label == "iss1_ud0_stg2":
                continue
            d = results.get(f"bwd/{label}", {})
            row = f"  {desc:<30}      "
            for kvseqlen in KVSEQLENS:
                base = base_tfs.get(kvseqlen)
                if base and kvseqlen in d.get("topk", []):
                    tf = d["tflops"][d["topk"].index(kvseqlen)]
                    if tf is not None:
                        row += f" {(tf - base) / base * 100:>+5.1f}%"
                        continue
                row += "    N/A"
            print(row)

    print(f"\n  Results: {_results_path(_PHASE)}", flush=True)


def _phase4_2_plot():
    """Grouped bar chart: ISS configs at multiple kvseqlens."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results(_PHASE)
    if not results:
        print(f"ERROR: {_results_path(_PHASE)} not found. Run --exp first.")
        return

    out = _out_dir(_PHASE)
    os.makedirs(out, exist_ok=True)

    configs_data = []
    for label, _, desc, smem in ISS_CONFIGS:
        d = results.get(f"bwd/{label}", {})
        if not d or "topk" not in d:
            continue
        tfs = [
            d["tflops"][d["topk"].index(s)] if s in d["topk"] else None
            for s in KVSEQLENS
        ]
        if any(t is not None for t in tfs):
            configs_data.append((desc, tfs))

    if not configs_data:
        print("No data found.")
        return

    colors = ["#1565C0", "#7B1FA2", "#C62828", "#FF8F00", "#2E7D32", "#795548"]
    n_groups = len(KVSEQLENS)
    n_bars = len(configs_data)
    bar_w = 0.75 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    for i, (desc, tfs) in enumerate(configs_data):
        vals = [tf if tf else 0 for tf in tfs]
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
        f"ISS Double-Buffer: Sparse LoopK BWD\n"
        f"qseqlen={QSEQLEN}, topk=kvseqlen/4, nhq={NHQ}, kbs={KBS}, pack_gqa, bf16, H100",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("TFLOPS (BWD)", fontsize=12)
    ax.set_xlabel("kvseqlen", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s // 1024}K" for s in KVSEQLENS], fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    valid_max = max((tf for _, tfs in configs_data for tf in tfs if tf), default=100)
    ax.set_ylim(0, valid_max * 1.15)

    plt.tight_layout()
    path = os.path.join(out, "iss_double_buffer_comparison.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] ISS plot -> {path}")
