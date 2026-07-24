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

"""CuTeDSL SM100 sparse benchmark -- Video Production scenario.

Mirrors exps/attn/sparse/bench_sparse_analysis/phase6_video_production.py
but drives the CuTeDSL kernel entry points directly on Blackwell (SM100).

Production config: 1080p, qhead=32, kvhead=8, hd=128.
32-GPU training: 8 KV-heads distributed -> per-rank nhk=1, nhq=128.
Per-rank: qseqlen = kvseqlen/64, topk = kvseqlen/8.
Post-distribution: NHQ=128, NHK=1, PackGQA, bf16.

Methods compared (all kbs=128):
  - Dense: gathered KV baseline (K/V size = topk)
  - BlockSparse: block-level masks on full KV
  - IndexSparse-TMA: token-index TMA loads on full KV

Run:
    cd exps/attn/cutedsl/sparse
    PYTHONPATH=../../../.. python 0-run_video_production.py --exp
    PYTHONPATH=../../../.. python 0-run_video_production.py --plot
    PYTHONPATH=../../../.. python 0-run_video_production.py --exp --plot
"""

import argparse
import datetime
import gc
import json
import os
import sys
from math import ceil

import torch

from magi_attention.kernel.cutedsl.ffa_utils import TorchFlexAttnArgs
from magi_attention.kernel.cutedsl.flex_flash_attn import (
    _flex_flash_attn_bwd,
    _flex_flash_attn_fwd,
)
from magi_attention.kernel.cutedsl.sparse_utils import (
    BlockSparseTensorsTorch,
    prepare_index_sparse_tiles,
)

NHQ, NHK, HD = 128, 1, 128
N_BLOCK_SIZE = 128
WARMUP, ITERS = 8, 20

SCENARIOS = [
    # (kvseqlen, qseqlen, topk)
    (32768, 512, 4096),
    (65536, 1024, 8192),
    (131072, 2048, 16384),
    (262144, 4096, 32768),
    (524288, 8192, 65536),
]

METHODS = ["dense", "block_sparse", "index_sparse"]
PASSES = ["fwd", "bwd"]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR = os.path.join(_SCRIPT_DIR, "outs", "0-video-production")
_RESULTS_PATH = os.path.join(_OUT_DIR, "results.json")


def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _load_results():
    if os.path.exists(_RESULTS_PATH):
        with open(_RESULTS_PATH) as f:
            return json.load(f)
    return {}


def _save_results(results):
    os.makedirs(os.path.dirname(_RESULTS_PATH), exist_ok=True)
    tmp = _RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    if os.path.getsize(tmp) > 0:
        os.replace(tmp, _RESULTS_PATH)


def _has_entry(results, key, kvseqlen):
    d = results.get(key, {})
    if not d or "kvseqlen" not in d:
        return False
    try:
        idx = d["kvseqlen"].index(kvseqlen)
        return d["tflops"][idx] is not None
    except (ValueError, IndexError):
        return False


def _set_entry(results, key, kvseqlen, tflops, ms):
    if key not in results:
        results[key] = {"kvseqlen": [], "tflops": [], "ms": []}
    d = results[key]
    if kvseqlen in d["kvseqlen"]:
        idx = d["kvseqlen"].index(kvseqlen)
        d["tflops"][idx] = tflops
        d["ms"][idx] = ms
    else:
        d["kvseqlen"].append(kvseqlen)
        d["tflops"].append(tflops)
        d["ms"].append(ms)


def _bench_kernel(run_fn, flops, device):
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


def _calc_flops(qseqlen, topk, is_bwd):
    fwd = 4 * qseqlen * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _fwd_m_block(seqlen_q, qhpk, pack_gqa):
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


def _make_bst_fwd(sq, n_blocks, topk_blocks, sel):
    """BST for FWD or BWD-LoopK (M-indexed -> N-block list)."""
    qhpk = NHQ // NHK
    m_bs = _fwd_m_block(sq, qhpk, True)
    M = ceil(sq * qhpk / m_bs)
    mask_cnt = torch.zeros(1, NHK, M, dtype=torch.int32, device="cuda")
    mask_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_cnt = torch.full((1, NHK, M), topk_blocks, dtype=torch.int32, device="cuda")
    full_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_idx[:, :, :, :topk_blocks] = sel.view(1, 1, 1, -1).expand(
        1, NHK, M, topk_blocks
    )
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(m_bs, N_BLOCK_SIZE),
    )


def _make_bst_bwd(sq, n_blocks, topk_blocks, sel):
    """BST for BWD-LoopK (M-indexed, m_bs=128)."""
    qhpk = NHQ // NHK
    m_bs = 128
    M = ceil(sq * qhpk / m_bs)
    mask_cnt = torch.zeros(1, NHK, M, dtype=torch.int32, device="cuda")
    mask_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_cnt = torch.full((1, NHK, M), topk_blocks, dtype=torch.int32, device="cuda")
    full_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_idx[:, :, :, :topk_blocks] = sel.view(1, 1, 1, -1).expand(
        1, NHK, M, topk_blocks
    )
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(m_bs, N_BLOCK_SIZE),
    )


def _make_is_tiles(sq, sk, topk_blocks, sel):
    """IS-TMA tiles for FWD (m=256) and BWD-LoopK (m=128), kbs=128."""
    topk = topk_blocks * N_BLOCK_SIZE
    indices = (
        sel.unsqueeze(-1) * N_BLOCK_SIZE
        + torch.arange(N_BLOCK_SIZE, device="cuda").unsqueeze(0)
    ).reshape(-1)
    indices = (
        indices.unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, NHQ, sq, topk)
        .contiguous()
        .int()
    )
    qhpk = NHQ // NHK
    fwd_m = _fwd_m_block(sq, qhpk, True)
    fwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=1,
        seqlen_q=sq,
        seqlen_k=sk,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=fwd_m,
        n_block_size=N_BLOCK_SIZE,
        pack_gqa=True,
        sparse_k_block_size=128,
    )
    bwd_tiles = prepare_index_sparse_tiles(
        indices,
        batch_size=1,
        seqlen_q=sq,
        seqlen_k=sk,
        num_kv_heads=NHK,
        num_q_heads=NHQ,
        m_block_size=128,
        n_block_size=N_BLOCK_SIZE,
        pack_gqa=True,
        sparse_k_block_size=128,
    )
    return fwd_tiles, bwd_tiles


def _run_experiment(force=False, max_kvseqlen=None):
    device = "cuda"
    results = {} if force else _load_results()

    scenarios = SCENARIOS
    if max_kvseqlen is not None:
        scenarios = [(kv, q, t) for kv, q, t in SCENARIOS if kv <= max_kvseqlen]

    print(f"[{_ts()}] CuTeDSL SM100 Sparse Bench: Video Production", flush=True)
    print(f"  NHQ={NHQ}, NHK={NHK}, HD={HD}, PackGQA, bf16, B300", flush=True)
    print("  Methods: Dense (gathered KV) / BS kbs=128 / IS-TMA kbs=128\n", flush=True)

    qhpk = NHQ // NHK
    scale = HD**-0.5

    for kvseqlen, qseqlen, topk in scenarios:
        topk_blocks = topk // N_BLOCK_SIZE
        n_blocks = kvseqlen // N_BLOCK_SIZE

        print(
            f"  -- kvseqlen={kvseqlen // 1024}k, qseqlen={qseqlen}, "
            f"topk={topk // 1024}k ({topk_blocks} blocks) --",
            flush=True,
        )

        torch.manual_seed(42)
        sel = torch.randperm(n_blocks, device=device)[:topk_blocks].sort().values

        for pass_type in PASSES:
            is_bwd = pass_type == "bwd"
            flops = _calc_flops(qseqlen, topk, is_bwd)

            for method in METHODS:
                key = f"{pass_type}/{method}"

                if not force and _has_entry(results, key, kvseqlen):
                    d = results[key]
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:4s} {method:14s}: {tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()

                try:
                    # Clear JIT cache to avoid type mismatches between methods
                    _flex_flash_attn_fwd.compile_cache.clear()
                    _flex_flash_attn_bwd.compile_cache.clear()

                    torch.manual_seed(42)
                    q = torch.randn(
                        1, qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device
                    )

                    if method == "dense":
                        k = torch.randn(
                            1, topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, topk, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        fwd_args = TorchFlexAttnArgs()
                        out, lse = _flex_flash_attn_fwd(
                            q,
                            k,
                            v,
                            softmax_scale=scale,
                            flex_attn_args=fwd_args,
                            pack_gqa=True,
                        )
                        if not is_bwd:

                            def run_fn():
                                _flex_flash_attn_fwd(
                                    q,
                                    k,
                                    v,
                                    softmax_scale=scale,
                                    flex_attn_args=fwd_args,
                                    pack_gqa=True,
                                )

                        else:
                            dO = torch.randn_like(out)

                            def run_fn():
                                _flex_flash_attn_bwd(
                                    q,
                                    k,
                                    v,
                                    out,
                                    lse,
                                    dO,
                                    softmax_scale=scale,
                                    flex_attn_args=fwd_args,
                                    pack_gqa=True,
                                )

                    elif method == "block_sparse":
                        k = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        fwd_bst = _make_bst_fwd(qseqlen, n_blocks, topk_blocks, sel)
                        fwd_args = TorchFlexAttnArgs(block_sparse_tensors=fwd_bst)
                        out, lse = _flex_flash_attn_fwd(
                            q,
                            k,
                            v,
                            softmax_scale=scale,
                            flex_attn_args=fwd_args,
                            pack_gqa=True,
                        )
                        if not is_bwd:

                            def run_fn():
                                _flex_flash_attn_fwd(
                                    q,
                                    k,
                                    v,
                                    softmax_scale=scale,
                                    flex_attn_args=fwd_args,
                                    pack_gqa=True,
                                )

                        else:
                            bwd_bst = _make_bst_bwd(qseqlen, n_blocks, topk_blocks, sel)
                            bwd_args = TorchFlexAttnArgs(block_sparse_tensors=bwd_bst)
                            dO = torch.randn_like(out)

                            def run_fn():
                                _flex_flash_attn_bwd(
                                    q,
                                    k,
                                    v,
                                    out,
                                    lse,
                                    dO,
                                    softmax_scale=scale,
                                    flex_attn_args=bwd_args,
                                    pack_gqa=True,
                                    swap_bwd_qk_loop=True,
                                )

                    elif method == "index_sparse":
                        k = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        fwd_tiles, bwd_tiles = _make_is_tiles(
                            qseqlen, kvseqlen, topk_blocks, sel
                        )
                        fwd_args = TorchFlexAttnArgs(index_sparse_tiles=fwd_tiles)
                        out, lse = _flex_flash_attn_fwd(
                            q,
                            k,
                            v,
                            softmax_scale=scale,
                            flex_attn_args=fwd_args,
                            pack_gqa=True,
                        )
                        if not is_bwd:

                            def run_fn():
                                _flex_flash_attn_fwd(
                                    q,
                                    k,
                                    v,
                                    softmax_scale=scale,
                                    flex_attn_args=fwd_args,
                                    pack_gqa=True,
                                )

                        else:
                            bwd_args = TorchFlexAttnArgs(index_sparse_tiles=bwd_tiles)
                            dO = torch.randn_like(out)

                            def run_fn():
                                _flex_flash_attn_bwd(
                                    q,
                                    k,
                                    v,
                                    out,
                                    lse,
                                    dO,
                                    softmax_scale=scale,
                                    flex_attn_args=bwd_args,
                                    pack_gqa=True,
                                    swap_bwd_qk_loop=True,
                                )

                    tf, ms = _bench_kernel(run_fn, flops, device)
                    _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                    print(
                        f"    {pass_type:4s} {method:14s}: {tf:>7.1f} T ({ms:.3f}ms)",
                        flush=True,
                    )

                except torch.cuda.OutOfMemoryError:
                    _set_entry(results, key, kvseqlen, None, None)
                    print(f"    {pass_type:4s} {method:14s}: OOM", flush=True)
                except Exception as e:
                    _set_entry(results, key, kvseqlen, None, None)
                    print(f"    {pass_type:4s} {method:14s}: FAIL - {e}", flush=True)
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

                _save_results(results)

    print(f"\n[{_ts()}] Experiment DONE -> {_RESULTS_PATH}", flush=True)
    _print_summary(results)


def _print_summary(results):
    scenarios = [
        (kv, q, t)
        for kv, q, t in SCENARIOS
        if any(
            kv in results.get(f"{p}/{m}", {}).get("kvseqlen", [])
            for p in PASSES
            for m in METHODS
        )
    ]
    header = f"  {'kv':>6s} {'sq':>6s} {'topk':>6s}"
    for p in PASSES:
        for m in METHODS:
            header += f" | {p[:1].upper()+'_'+m[:5]:>9s}"
    print(f"\n{header}")
    print("  " + "-" * len(header))
    for kvseqlen, qseqlen, topk in scenarios:
        row = f"  {kvseqlen//1024:>5d}k {qseqlen:>6d} {topk//1024:>5d}k"
        for p in PASSES:
            for m in METHODS:
                key = f"{p}/{m}"
                d = results.get(key, {})
                if kvseqlen in d.get("kvseqlen", []):
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    row += f" | {tf:>7.0f} T" if tf else " |    FAIL "
                else:
                    row += " |       - "
        print(row)


def _plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results()
    if not results:
        print(f"ERROR: {_RESULTS_PATH} not found. Run --exp first.")
        return
    os.makedirs(_OUT_DIR, exist_ok=True)

    PLOT_PASSES = [("fwd", "FWD"), ("bwd", "BWD (LoopK)")]
    PLOT_METHODS = [
        ("dense", "Dense (gathered KV)", "#5A5A5A"),
        ("block_sparse", "BlockSparse (kbs=128)", "#2E86C1"),
        ("index_sparse", "IndexSparse-TMA (kbs=128)", "#C0392B"),
    ]

    kvseqlens = sorted(
        set(kv for d in results.values() for kv in d.get("kvseqlen", []))
    )
    if not kvseqlens:
        print("No data to plot.")
        return

    x = np.arange(len(kvseqlens))
    bw = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

    for col_idx, (pid, pname) in enumerate(PLOT_PASSES):
        ax = axes[col_idx]
        n_m = len(PLOT_METHODS)
        for i, (mid, lbl, col) in enumerate(PLOT_METHODS):
            key = f"{pid}/{mid}"
            d = results.get(key, {})
            vals = []
            for kv in kvseqlens:
                if kv in d.get("kvseqlen", []):
                    idx = d["kvseqlen"].index(kv)
                    v = d["tflops"][idx] if d["tflops"][idx] else 0
                else:
                    v = 0
                vals.append(v)
            off = (i - n_m / 2 + 0.5) * bw
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
                        fontsize=7,
                        fontweight="bold",
                    )

        ax.set_title(pname, fontsize=14, fontweight="bold")
        ax.set_xlabel("kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)", fontsize=10)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 8192}K)" for kv in kvseqlens],
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "CuTeDSL SM100 Sparse -- Video Production Scenario (kbs=128)\n"
        f"NHQ={NHQ}, NHK={NHK}, HD={HD}, PackGQA, bf16, B300",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(_OUT_DIR, "0_video_production.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[{_ts()}] Plot saved -> {path}")


def main():
    parser = argparse.ArgumentParser(
        description="CuTeDSL SM100 sparse benchmark -- Video Production scenario"
    )
    parser.add_argument("--exp", action="store_true", help="Run benchmark experiment")
    parser.add_argument(
        "--plot", action="store_true", help="Generate plot from results"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run all (ignore cache)"
    )
    parser.add_argument(
        "--max-kvseqlen",
        type=int,
        default=None,
        help="Cap kvseqlen (e.g. 131072 for quick test)",
    )
    args = parser.parse_args()
    if not args.exp and not args.plot:
        parser.print_help()
        sys.exit(1)
    if args.exp:
        _run_experiment(force=args.force, max_kvseqlen=args.max_kvseqlen)
    if args.plot:
        _plot()


if __name__ == "__main__":
    main()
