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
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real:\\
/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    cd /path/to/MagiAttention
    python exps/attn/run_sparse_sweep/bench_sparse_sweep.py
    python exps/attn/run_sparse_sweep/bench_sparse_sweep.py --simple
"""
import argparse
import math
import sys
import time
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
    (128, 1),   # R=128 MQA — baseline
    (64,  1),   # R=64
    (32,  1),   # R=32  — 50% M-dim utilization
    (16,  1),   # R=16  — FFA UNSUPPORTED
    (8,   1),   # R=8   — FFA UNSUPPORTED
    (1,   1),   # R=1   MHA — FFA UNSUPPORTED
]

SEQLEN_TOPK_CONFIGS = [
    # (S, B, topk) — 5% and 25% sparsity
    (8192,  2,  410),  # ~5%
    (8192,  2, 2048),  # 25%
    (16384, 2,  819),  # ~5%
    (16384, 2, 4096),  # 25%
    (32768, 1, 1638),  # ~5%
    (32768, 1, 8192),  # 25%
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
    q: torch.Tensor      # (B, S, NHQ, D)
    k: torch.Tensor      # (B, S, NHK, D)
    v: torch.Tensor      # (B, S, NHK, D)
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
        topk_indices[:, i, :len(nz)] = nz.int()

    return ScenarioData(B=B, S=S, NHQ=NHQ, NHK=NHK, topk=topk,
                        q=q, k=k, v=v, block_mask=mask,
                        sparsity=sparsity, topk_indices=topk_indices)


# ─── Method Implementations ─── #

def run_sdpa_full(sd: ScenarioData):
    qb = rearrange(sd.q, "b s h d -> b h s d").contiguous()
    kb = rearrange(sd.k, "b s h d -> b h s d").contiguous()
    vb = rearrange(sd.v, "b s h d -> b h s d").contiguous()
    def fn():
        return torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, enable_gqa=True)
    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_ffa_full(sd: ScenarioData):
    from magi_attention.functional import flex_flash_attn_func
    NHK = sd.NHK
    q_t = rearrange(sd.q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=NHK).contiguous()
    k_t = rearrange(sd.k, "b s h d -> (b h s) 1 d").contiguous()
    v_t = rearrange(sd.v, "b s h d -> (b h s) 1 d").contiguous()
    qr = torch.tensor([[i * sd.S, (i + 1) * sd.S] for i in range(sd.B)],
                       dtype=torch.int32, device=DEV)
    kr = qr.clone()
    atm = torch.zeros(sd.B, dtype=torch.int32, device=DEV)
    def fn():
        return flex_flash_attn_func(q_t, k_t, v_t, q_ranges=qr, k_ranges=kr,
                                    attn_type_map=atm, pack_gqa=True,
                                    ref_block_size=(128, 128))
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
            qb, kb, vb, attn_mask=am, enable_gqa=True)
    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.S, sd.NHQ, D, sd.sparsity)
    return ms, fl / ms * 1e-9


def run_ffa_sparse(sd: ScenarioData):
    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import (
        choose_ref_block, generate_ranges_from_block_mask_triton,
    )
    R = sd.NHQ // sd.NHK
    ref_params = choose_ref_block((1, 1), qhead_per_khead=R)

    q_t = rearrange(sd.q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=sd.NHK).contiguous()
    k_t = rearrange(sd.k, "b s h d -> (b h s) 1 d").contiguous()
    v_t = rearrange(sd.v, "b s h d -> (b h s) 1 d").contiguous()

    qr_1, kr_1 = generate_ranges_from_block_mask_triton(sd.block_mask, 1, 1)
    qr_list, kr_list = [], []
    for bi in range(sd.B):
        qr_list.append(qr_1 + bi * sd.S)
        kr_list.append(kr_1 + bi * sd.S)
    qr = torch.cat(qr_list, dim=0)
    kr = torch.cat(kr_list, dim=0)
    atm = torch.zeros(len(qr), dtype=torch.int32, device=DEV)

    def fn():
        return flex_flash_attn_func(q_t, k_t, v_t,
                                    q_ranges=qr, k_ranges=kr,
                                    attn_type_map=atm, auto_range_merge=True,
                                    max_seqlen_q=1, **ref_params)
    ms = bench_fn(fn)
    fl = flops_attn(sd.B, sd.S, sd.topk, sd.NHQ, D)
    return ms, fl / ms * 1e-9


def run_tl_sparse(sd: ScenarioData):
    sys.path.insert(0,
        "/home/niubility2/cenzhiyao/SparseAttention/09_deepseek_sparse/00_deepseek_v4/inference")
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
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    qb = rearrange(sd.q, "b s h d -> b h s d").contiguous()
    kb = rearrange(sd.k, "b s h d -> b h s d").contiguous()
    vb = rearrange(sd.v, "b s h d -> b h s d").contiguous()
    bm = sd.block_mask[0, 0]
    def mask_mod(b, h, q_idx, kv_idx):
        return bm[q_idx, kv_idx]
    flex_bm = create_block_mask(mask_mod, B=1, H=1, Q_LEN=sd.S, KV_LEN=sd.S,
                                device=DEV, BLOCK_SIZE=128)
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
    ("FFA-full",  run_ffa_full),
    ("FA3-full",  run_fa3_full),
]

METHODS_SPARSE = [
    ("SDPA-mask",    run_sdpa_mask),
    ("FFA-sparse",   run_ffa_sparse),
    ("TL-sparse",    run_tl_sparse),
    ("FlexAttn-sp",  run_flex_attn_sparse),
]


def run_scenario(nhq, nhk, S, B, topk, methods_full, methods_sparse):
    R = nhq // nhk
    sp = topk / S
    tag = f"R={R:>3} S={S:>5} B={B} topk={topk:>5} sp={sp:.3f}"
    print(f"\n{'─'*78}")
    print(f"  {tag}")
    print(f"{'─'*78}")

    sd = make_scenario(B, S, nhq, nhk, topk)
    row = {"R": R, "NHQ": nhq, "S": S, "B": B, "topk": topk, "sp": sp}

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
    print(f"\n{'='*120}")
    print(f"  Summary: FFA Sparse Multi-Scenario Benchmark  (H100 peak = {H100_PEAK:.0f} TFLOPS)")
    print(f"{'='*120}")

    # per-method columns: TFLOPS + MFU
    hdr = f"  {'R':>3} {'S':>5} {'topk':>5} {'sp':>5}"
    for m in all_methods:
        hdr += f"  {m + ' T':>12} {'MFU':>5}"
    # speedup column: best_full_ms / best_sparse_ms
    hdr += f"  {'speedup':>8} {'ideal':>6}"
    print(hdr)
    sep = f"  {'─'*3} {'─'*5} {'─'*5} {'─'*5}"
    for m in all_methods:
        sep += f"  {'─'*12} {'─'*5}"
    sep += f"  {'─'*8} {'─'*6}"
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
            ideal = 1.0 / row['sp']
            line += f"  {speedup:>7.2f}x {ideal:>5.1f}x"
        else:
            line += f"  {'--':>8} {'--':>6}"
        print(line)

    print(f"""
  ─────────────────────────────────────────────────────────────
  Notes:
    - T = TFLOPS, MFU = TFLOPS / {H100_PEAK:.0f}T
    - sparse TFLOPS based on sparse FLOPs (4*B*S*topk*NHQ*D)
    - full TFLOPS based on dense FLOPs (4*B*S*S*NHQ*D)
    - speedup = best_full_ms / best_sparse_ms
    - ideal = 1/sparsity (theoretical upper bound, assuming constant MFU)
    - FFA-sparse [UNSUPPORTED] configs: see 1-gqa_sparse_compat/
  ─────────────────────────────────────────────────────────────""")


# ─── Main ─── #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", action="store_true",
                        help="Only run FFA-full and FFA-sparse, skip incompatible configs")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"D={D}, dtype={DTYPE}, warmup={WARMUP}, repeat={REPEAT}")
    print(f"Mode: {'simple (FFA only)' if args.simple else 'full (all methods)'}")

    if args.simple:
        qhead_cfgs = [
            (128, 1), (64, 1), (32, 1),
        ]
        seqlen_cfgs = SEQLEN_TOPK_CONFIGS
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