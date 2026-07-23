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
import time

from bench_sparse_analysis._common import (
    HD,
    KBS,
    NHK,
    NHQ,
    S_FULL,
    TOPK_VALS,
    _bench_ffa,
    _bench_kernel,
    _build_idx_kbs128,
    _calc_flops,
    _has_entry,
    _indices_to_ranges,
    _load_results,
    _make_tensors_kv_short,
    _out_dir,
    _results_path,
    _save_results,
    _set_entry,
    _set_gpu,
    _ts,
)


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
            sparse_k_block_size=KBS,
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
            range_merge=True,
            pack_gqa=True,
            sparse_k_block_size=KBS,
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
