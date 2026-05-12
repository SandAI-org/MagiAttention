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

"""
FFA Sparse Multi-Scenario Efficiency Benchmark
================================================
Sweep dimensions: qhead config x seqlen x topk
Methods compared:
  Full:   SDPA-full, FFA-full, FA3-full
  Sparse: SDPA-mask, FFA-sparse, TL-sparse, FlexAttn-sparse

Focus: identify efficiency highs/lows across scenarios, and whether
FFA-sparse is universally slow or only in specific configs.
Incompatible configs are marked [UNSUPPORTED] without affecting others.

Usage:
    cd /path/to/MagiAttention
    python exps/attn/run_sparse_sweep/bench_sparse_sweep.py
    python exps/attn/run_sparse_sweep/bench_sparse_sweep.py --simple
"""
import argparse
import math
import sys
import traceback
from dataclasses import dataclass

import torch
from einops import rearrange

try:
    torch._functorch.config.donated_buffer = False
except AttributeError:
    pass

# ─── Global Constants ─── #
D = 128
DTYPE = torch.bfloat16
DEV = "cuda"
WARMUP, REPEAT = 3, 7
H100_PEAK = 989.4  # BF16 Tensor Core TFLOPS

# ─── Sweep Configs ─── #

QHEAD_CONFIGS = [
    # (NHQ, NHK)
    (128, 1),  # R=128 MQA — baseline
    (64, 1),  # R=64
    (32, 1),  # R=32  — 50% M-dim utilization
    (16, 1),  # R=16  — FFA UNSUPPORTED
    (8, 1),  # R=8   — FFA UNSUPPORTED
    (1, 1),  # R=1   MHA — FFA UNSUPPORTED
]

SEQLEN_TOPK_CONFIGS = [
    # (S, B, topk) — 25% and 5% sparsity, seqlen descending
    (32768, 1, 8192),  # 25%
    (32768, 1, 1638),  # ~5%
    (16384, 2, 4096),  # 25%
    (16384, 2, 819),  # ~5%
    (8192, 2, 2048),  # 25%
    (8192, 2, 410),  # ~5%
]


# ─── Utilities ─── #


def flops_attn(b, sq, sk, nh, d, sp=1.0):
    return 4 * b * sq * sk * nh * d * sp


def bench_fn(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        with torch.no_grad():
            fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(starts[i].elapsed_time(ends[i]) for i in range(repeat))
    return times[len(times) // 2]


@dataclass
class ScenarioData:
    B: int
    S: int
    NHQ: int
    NHK: int
    topk: int
    q: torch.Tensor  # (B, S, NHQ, D)
    k: torch.Tensor  # (B, S, NHK, D)
    v: torch.Tensor  # (B, S, NHK, D)
    block_mask: torch.Tensor  # (1, 1, S, S) bool
    sparsity: float
    topk_indices: torch.Tensor  # (B, S, topk) int32


def make_scenario(B, S, NHQ, NHK, topk) -> ScenarioData:
    torch.manual_seed(42)
    q = torch.randn(B, S, NHQ, D, dtype=DTYPE, device=DEV)
    k = torch.randn(B, S, NHK, D, dtype=DTYPE, device=DEV)
    v = torch.randn(B, S, NHK, D, dtype=DTYPE, device=DEV)

    # per-row random topk mask
    mask = torch.zeros(1, 1, S, S, dtype=torch.bool, device=DEV)
    for i in range(S):
        idx = torch.randperm(S, device=DEV)[:topk]
        mask[0, 0, i, idx] = True
    sparsity = topk / S

    topk_indices = torch.zeros(B, S, topk, dtype=torch.int32, device=DEV)
    for i in range(S):
        nz = mask[0, 0, i].nonzero(as_tuple=False).squeeze(-1)[:topk]
        topk_indices[:, i, : len(nz)] = nz.int()

    return ScenarioData(
        B=B,
        S=S,
        NHQ=NHQ,
        NHK=NHK,
        topk=topk,
        q=q,
        k=k,
        v=v,
        block_mask=mask,
        sparsity=sparsity,
        topk_indices=topk_indices,
    )


# ─── Method Implementations ─── #


def run_sdpa_full(sd: ScenarioData):
    qb = rearrange(sd.q, "b s h d -> b h s d").contiguous()
    kb = rearrange(sd.k, "b s h d -> b h s d").contiguous()
    vb = rearrange(sd.v, "b s h d -> b h s d").contiguous()

    def fn():
        return torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, enable_gqa=True
        )

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_ffa_full(sd: ScenarioData):
    from magi_attention.functional import flex_flash_attn_func

    NHK = sd.NHK
    q_t = rearrange(sd.q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK).contiguous()
    k_t = rearrange(sd.k, "b s h d -> (b h s) 1 d").contiguous()
    v_t = rearrange(sd.v, "b s h d -> (b h s) 1 d").contiguous()
    qr = torch.tensor(
        [[i * sd.S, (i + 1) * sd.S] for i in range(sd.B)], dtype=torch.int32, device=DEV
    )
    kr = qr.clone()
    atm = torch.zeros(sd.B, dtype=torch.int32, device=DEV)

    def fn():
        return flex_flash_attn_func(
            q_t,
            k_t,
            v_t,
            q_ranges=qr,
            k_ranges=kr,
            attn_type_map=atm,
            pack_gqa=True,
            ref_block_size=(128, 128),
        )

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_fa3_full(sd: ScenarioData):
    from flash_attn_interface import flash_attn_func

    q_fa = sd.q.contiguous()
    k_fa = sd.k.contiguous()
    v_fa = sd.v.contiguous()

    def fn():
        return flash_attn_func(q_fa, k_fa, v_fa)

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_sdpa_mask(sd: ScenarioData):
    qb = rearrange(sd.q, "b s h d -> b h s d").contiguous()
    kb = rearrange(sd.k, "b s h d -> b h s d").contiguous()
    vb = rearrange(sd.v, "b s h d -> b h s d").contiguous()
    am = torch.zeros(1, 1, sd.S, sd.S, dtype=DTYPE, device=DEV)
    am.masked_fill_(~sd.block_mask[0, 0], float("-inf"))

    def fn():
        return torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=am, enable_gqa=True
        )

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D, sd.sparsity)
    return ms, fl / ms * 1e-9


def run_ffa_sparse(sd: ScenarioData):
    """FFA sparse via sparse_kv_indices direct-to-kernel path (no q/k ranges)."""
    from magi_attention.functional import flex_flash_attn_func

    q_t = rearrange(sd.q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=sd.NHK).contiguous()
    k_t = rearrange(sd.k, "b s h d -> (b h s) 1 d").contiguous()
    v_t = rearrange(sd.v, "b s h d -> (b h s) 1 d").contiguous()

    # sparse_kv_indices: [B, NHK, S, topk] with local KV token ids
    indices = sd.topk_indices.unsqueeze(1).expand(-1, sd.NHK, -1, -1).contiguous()

    def fn():
        return flex_flash_attn_func(
            q_t,
            k_t,
            v_t,
            sparse_kv_indices=indices,
            actual_topk=[sd.topk] * sd.B,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=True,
        )

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.topk, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_tl_sparse(sd: ScenarioData):
    sys.path.insert(
        0,
        "/home/niubility2/cenzhiyao/SparseAttention/09_deepseek_sparse/00_deepseek_v4/inference",
    )
    from kernel import sparse_attn

    q_tl = sd.q.contiguous()
    kv_tl = sd.k[:, :, 0, :].contiguous()  # MLA: KV shared
    sink = torch.full((sd.NHQ,), -1e9, dtype=torch.float32, device=DEV)
    scale = 1.0 / math.sqrt(D)

    def fn():
        return sparse_attn(q_tl, kv_tl, sink, sd.topk_indices, scale)

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.topk, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_flex_attn_sparse(sd: ScenarioData):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    qb = rearrange(sd.q, "b s h d -> b h s d").contiguous()
    kb = rearrange(sd.k, "b s h d -> b h s d").contiguous()
    vb = rearrange(sd.v, "b s h d -> b h s d").contiguous()
    bm = sd.block_mask[0, 0]

    def mask_mod(b, h, q_idx, kv_idx):
        return bm[q_idx, kv_idx]

    flex_bm = create_block_mask(
        mask_mod, B=1, H=1, Q_LEN=sd.S, KV_LEN=sd.S, device=DEV, BLOCK_SIZE=128
    )
    flex_fn = torch.compile(flex_attention, mode="max-autotune")
    # warmup compile
    for _ in range(2):
        with torch.no_grad():
            flex_fn(qb, kb, vb, block_mask=flex_bm, enable_gqa=True)

    def fn():
        return flex_fn(qb, kb, vb, block_mask=flex_bm, enable_gqa=True)

    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D, sd.sparsity)
    return ms, fl / ms * 1e-9


# ─── Runner ─── #


METHODS_FULL = [
    ("SDPA-full", run_sdpa_full),
    ("FFA-full", run_ffa_full),
    ("FA3-full", run_fa3_full),
]

METHODS_SPARSE = [
    ("SDPA-mask", run_sdpa_mask),
    ("FFA-sparse", run_ffa_sparse),
    ("TL-sparse", run_tl_sparse),
    ("FlexAttn-sp", run_flex_attn_sparse),
]


def sanity_check_ffa_sparse(sd: ScenarioData):
    """One-shot correctness check: FFA sparse vs SDPA with explicit mask."""
    from magi_attention.functional import flex_flash_attn_func

    q_t = rearrange(sd.q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=sd.NHK).contiguous()
    k_t = rearrange(sd.k, "b s h d -> (b h s) 1 d").contiguous()
    v_t = rearrange(sd.v, "b s h d -> (b h s) 1 d").contiguous()
    indices = sd.topk_indices.unsqueeze(1).expand(-1, sd.NHK, -1, -1).contiguous()

    with torch.no_grad():
        o_ffa, _ = flex_flash_attn_func(
            q_t.clone(), k_t.clone(), v_t.clone(),
            sparse_kv_indices=indices,
            actual_topk=[sd.topk] * sd.B,
            q_block_size=1, k_block_size=1, pack_gqa=True,
        )
    o_ffa = rearrange(o_ffa, "(b h1 s) h2 d -> b s (h1 h2) d", b=sd.B, h1=sd.NHK, s=sd.S)

    # Build dense mask from topk_indices for SDPA reference
    gqa = sd.NHQ // sd.NHK
    max_diffs = []
    for b in range(sd.B):
        mask_b = torch.full((sd.NHQ, sd.S, sd.S), float("-inf"), dtype=DTYPE, device=DEV)
        for qi in range(sd.S):
            valid = sd.topk_indices[b, qi]
            valid = valid[valid >= 0]
            for hk in range(sd.NHK):
                for hq_off in range(gqa):
                    mask_b[hk * gqa + hq_off, qi, valid.long()] = 0.0

        q_sdpa = rearrange(sd.q[b], "s h d -> 1 h s d")
        k_sdpa = rearrange(sd.k[b], "s h d -> 1 h s d")
        v_sdpa = rearrange(sd.v[b], "s h d -> 1 h s d")
        if gqa > 1:
            k_sdpa = k_sdpa.repeat_interleave(gqa, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(gqa, dim=1)

        with torch.no_grad():
            o_ref = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=mask_b.unsqueeze(0)
            )
        o_ref = rearrange(o_ref, "1 h s d -> s h d")
        max_diffs.append((o_ffa[b].float() - o_ref.float()).abs().max().item())

    return max(max_diffs)


def run_scenario(nhq, nhk, S, B, topk, methods_full, methods_sparse):
    R = nhq // nhk
    sp = topk / S
    tag = f"R={R:>3} S={S:>5} B={B} topk={topk:>5} sp={sp:.3f}"
    print(f"\n{'─' * 78}")
    print(f"  {tag}")
    print(f"{'─' * 78}")

    sd = make_scenario(B, S, nhq, nhk, topk)
    row = {"R": R, "NHQ": nhq, "S": S, "B": B, "topk": topk, "sp": sp}

    # Correctness sanity check (only if FFA-sparse is in the method list)
    has_ffa_sparse = any(n == "FFA-sparse" for n, _ in methods_sparse)
    if has_ffa_sparse and S <= 1024:
        try:
            max_diff = sanity_check_ffa_sparse(sd)
            status = "PASS" if max_diff < 0.02 else "FAIL"
            print(f"  [sanity] FFA-sparse vs SDPA: max_diff={max_diff:.6f} [{status}]")
        except Exception as e:
            print(f"  [sanity] ERROR: {str(e)[:80]}")

    for name, fn in methods_full:
        try:
            ms, tflops = fn(sd)
            row[name] = (ms, tflops)
            print(f"  {name:<14} {ms:>8.3f} ms  {tflops:>7.1f} TFLOPS")
        except Exception as e:
            row[name] = None
            print(f"  {name:<14} ERROR: {str(e)[:60]}")

    for name, fn in methods_sparse:
        try:
            ms, tflops = fn(sd)
            row[name] = (ms, tflops)
            print(f"  {name:<14} {ms:>8.3f} ms  {tflops:>7.1f} TFLOPS")
        except Exception as e:
            err = str(e).split("\n")[0][:80]
            row[name] = None
            print(f"  {name:<14} [UNSUPPORTED] {err}")

    del sd
    torch.cuda.empty_cache()
    return row


def print_summary(all_results, methods_full, methods_sparse):
    all_methods = [n for n, _ in methods_full] + [n for n, _ in methods_sparse]
    print(f"\n{'=' * 120}")
    print(
        f"  Summary: FFA Sparse Multi-Scenario Benchmark  (H100 peak = {H100_PEAK:.0f} TFLOPS)"
    )
    print(f"{'=' * 120}")

    # per-method columns: TFLOPS + MFU
    hdr = f"  {'R':>3} {'S':>5} {'topk':>5} {'sp':>5}"
    for m in all_methods:
        hdr += f"  {m + ' T':>12} {'MFU':>5}"
    # speedup column: best_full_ms / best_sparse_ms
    hdr += f"  {'speedup':>8} {'ideal':>6}"
    print(hdr)
    sep = f"  {'─' * 3} {'─' * 5} {'─' * 5} {'─' * 5}"
    for m in all_methods:
        sep += f"  {'─' * 12} {'─' * 5}"
    sep += f"  {'─' * 8} {'─' * 6}"
    print(sep)

    for row in all_results:
        line = f"  {row['R']:>3} {row['S']:>5} {row['topk']:>5} {row['sp']:>5.2f}"
        best_full_ms = None
        best_sparse_ms = None
        for m in all_methods:
            val = row.get(m)
            if val is None:
                line += f"  {'--':>12} {'--':>5}"
            else:
                ms, tflops = val
                mfu = tflops / H100_PEAK * 100
                line += f"  {tflops:>10.0f}T {mfu:>4.1f}%"
                is_full = any(m == n for n, _ in methods_full)
                if is_full:
                    if best_full_ms is None or ms < best_full_ms:
                        best_full_ms = ms
                else:
                    if best_sparse_ms is None or ms < best_sparse_ms:
                        best_sparse_ms = ms

        if best_full_ms and best_sparse_ms:
            speedup = best_full_ms / best_sparse_ms
            ideal = 1.0 / row["sp"]
            line += f"  {speedup:>7.2f}x {ideal:>5.1f}x"
        else:
            line += f"  {'--':>8} {'--':>6}"
        print(line)

    print(
        f"""
  ─────────────────────────────────────────────────────────────
  Notes:
    - T = TFLOPS, MFU = TFLOPS / {H100_PEAK:.0f}T
    - sparse TFLOPS based on sparse FLOPs (4 * B * S * topk * NHQ * D)
    - full TFLOPS based on dense FLOPs (4 * B * S * S * NHQ * D)
    - speedup = best_full_ms / best_sparse_ms
    - ideal = 1/sparsity (theoretical upper bound, assuming constant MFU)
    - FFA-sparse [UNSUPPORTED] configs: see 1-gqa_sparse_compat/
  ─────────────────────────────────────────────────────────────"""
    )


# ─── Main ─── #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Only run FFA-full and FFA-sparse, skip incompatible configs",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"D={D}, dtype={DTYPE}, warmup={WARMUP}, repeat={REPEAT}")
    print(f"Mode: {'simple (FFA only)' if args.simple else 'full (all methods)'}")

    if args.simple:
        qhead_cfgs = [
            (128, 1),
            (64, 1),
            (32, 1),
        ]
        seqlen_cfgs = [
            (512, 1, 128),  # small S for sanity check
        ] + SEQLEN_TOPK_CONFIGS
        mf = [("FFA-full", run_ffa_full)]
        ms = [("FFA-sparse", run_ffa_sparse)]
    else:
        qhead_cfgs = QHEAD_CONFIGS
        seqlen_cfgs = SEQLEN_TOPK_CONFIGS
        mf = METHODS_FULL
        ms = METHODS_SPARSE

    all_results = []
    for nhq, nhk in qhead_cfgs:
        for S, B, topk in seqlen_cfgs:
            try:
                row = run_scenario(nhq, nhk, S, B, topk, mf, ms)
                all_results.append(row)
            except Exception as e:
                print(f"  SCENARIO ERROR: {e}")
                traceback.print_exc()

    print_summary(all_results, mf, ms)
