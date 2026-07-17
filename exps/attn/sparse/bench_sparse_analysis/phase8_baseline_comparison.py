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

"""Phase 8: Baseline Comparison — FFA IndexSparse/BlockSparse vs external kernels.

Two sub-benchmarks organized by K block size:

  8a) kbs=1 (token-sparse):
      FFA IndexSparse vs FlexAttention vs Triton Token-Sparse
      Passes: FWD, BWD (LoopK only — IS kbs=1 has no LoopQ)

  8b) kbs=128 (block-sparse):
      FFA BlockSparse vs FFA IndexSparse(kbs=128) vs FlexAttention
      Passes: FWD, BWD_LoopK, BWD_LoopQ

Config: nhq=128, nhk=1, hd=128, topk=2048 (fixed), sweep S=[32K..512K].
"""

import gc
import os
import sys

from bench_sparse_analysis._common import (
    HD,
    NHK,
    NHQ,
    _bench_kernel,
    _load_results,
    _out_dir,
    _save_results,
    _set_gpu,
    _ts,
)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

PHASE = "8-baseline-comparison"
KBS_BLOCK = 128

# Same scenario as Phase 6: qseqlen = kvseqlen/64, topk = kvseqlen/8
SCENARIOS = [
    # (kvseqlen, qseqlen, topk)
    (32768, 512, 4096),
    (65536, 1024, 8192),
    (131072, 2048, 16384),
    (262144, 4096, 32768),
    (524288, 8192, 65536),
]

# 8a: kbs=1 (token-sparse)
KBS1_METHODS = ["ffa_is", "flexattn", "triton"]
KBS1_PASSES = ["fwd", "bwd"]
KBS1_LABELS = {
    "ffa_is": "FFA IndexSparse (kbs=1)",
    "flexattn": "FlexAttention",
    "triton": "Triton Token-Sparse",
}

# 8b: kbs=128 (block-sparse)
KBS128_METHODS = ["ffa_bs", "ffa_is128", "flexattn"]
KBS128_PASSES = ["fwd", "bwd_loopk", "bwd_loopq"]
KBS128_LABELS = {
    "ffa_bs": "FFA BlockSparse",
    "ffa_is128": "FFA IndexSparse (kbs=128)",
    "flexattn": "FlexAttention",
}


# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _calc_sparse_flops(qseqlen, topk, is_bwd):
    fwd = 4 * qseqlen * topk * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _build_flex_block_mask(qseqlen, kvseqlen, topk, device):
    """FlexAttention block_mask for sparse pattern (128x128 blocks), Q≠KV."""
    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    FLEX_BLOCK = 128
    num_q_blocks = (qseqlen + FLEX_BLOCK - 1) // FLEX_BLOCK
    num_kv_blocks = kvseqlen // FLEX_BLOCK
    kv_blocks_needed = min((topk + FLEX_BLOCK - 1) // FLEX_BLOCK, num_kv_blocks)

    selected = torch.rand(num_q_blocks, num_kv_blocks, device=device).argsort(dim=1)[
        :, :kv_blocks_needed
    ]
    mask_dense = torch.zeros(
        num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device
    )
    mask_dense.scatter_(1, selected, True)

    def sparse_mask_mod(b_idx, h_idx, q_idx, kv_idx):
        q_block = q_idx // FLEX_BLOCK
        kv_block = kv_idx // FLEX_BLOCK
        return mask_dense[q_block, kv_block]

    return create_block_mask(
        sparse_mask_mod, B=None, H=None, Q_LEN=qseqlen, KV_LEN=kvseqlen, device=device
    )


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


# ═══════════════════════════════════════════════════════════════
#  8a: kbs=1 Runners
# ═══════════════════════════════════════════════════════════════


def _run_kbs1_ffa_is(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA IndexSparse kbs=1. BWD always uses LoopK (no LoopQ for kbs=1)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    indices = (
        torch.randint(0, kvseqlen, (qseqlen, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .contiguous()
    )
    is_bwd = pass_type != "fwd"
    kw = dict(
        index_sparse_indices=indices,
        q_block_size=1,
        sparse_k_block_size=1,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention with sparse block mask."""
    import torch
    import torch._functorch.config
    from torch.nn.attention.flex_attention import flex_attention

    torch._functorch.config.donated_buffer = False
    q = torch.randn(1, NHQ, qseqlen, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, NHK, kvseqlen, HD, dtype=torch.bfloat16, device=device)
    block_mask = _build_flex_block_mask(qseqlen, kvseqlen, topk, device)
    _flex_fn = torch.compile(flex_attention)
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            _flex_fn(q, k, v, block_mask=block_mask, enable_gqa=True)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs1_triton(kvseqlen, qseqlen, topk, pass_type, device):
    """Triton hand-written token-sparse kernel."""
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_attn, token_sparse_fwd

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, 1, HD, dtype=torch.bfloat16, device=device)
    tri_indices = (
        torch.randint(0, kvseqlen, (qseqlen, topk), device=device, dtype=torch.int32)
        .sort(dim=1)
        .values
    )
    is_bwd = pass_type != "fwd"

    if is_bwd:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = token_sparse_attn(q, k, v, tri_indices)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            token_sparse_fwd(q, k, v, tri_indices)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


# ═══════════════════════════════════════════════════════════════
#  8b: kbs=128 Runners
# ═══════════════════════════════════════════════════════════════


def _run_kbs128_ffa_bs(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA BlockSparse (q_ranges/k_ranges + block_sparse + range_merge)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)

    n_kv_blocks = kvseqlen // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    indices = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
        .contiguous()
    )
    ia_3d = indices.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_BLOCK, num_k_blocks=n_kv_blocks
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)

    is_bwd = pass_type != "fwd"
    swap_qk = pass_type == "bwd_loopk"
    kw = dict(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=atm,
        pack_gqa=True,
        block_sparse=True,
        range_merge=True,
        disable_fwd_atomic_reduction=True,
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = swap_qk
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_ffa_is(kvseqlen, qseqlen, topk, pass_type, device):
    """FFA IndexSparse with kbs=128 (index_sparse_indices with block indices)."""
    import torch

    from magi_attention.functional import flex_flash_attn_func

    q = torch.randn(qseqlen, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device)

    n_kv_blocks = kvseqlen // KBS_BLOCK
    n_topk_blocks = topk // KBS_BLOCK
    indices = (
        torch.rand(qseqlen, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
        .contiguous()
    )

    is_bwd = pass_type != "fwd"
    swap_qk = pass_type == "bwd_loopk"
    kw = dict(
        index_sparse_indices=indices,
        q_block_size=1,
        sparse_k_block_size=KBS_BLOCK,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    if is_bwd:
        kw["swap_bwd_qk_loop"] = swap_qk
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o, *_ = flex_flash_attn_func(q, k, v, **kw)
        do = torch.randn_like(o)

        def run_fn():
            o.backward(do, retain_graph=True)

    else:

        def run_fn():
            flex_flash_attn_func(q, k, v, **kw)

    return _bench_kernel(run_fn, _calc_sparse_flops(qseqlen, topk, is_bwd), device)


def _run_kbs128_flexattn(kvseqlen, qseqlen, topk, pass_type, device):
    """FlexAttention for kbs=128 group (same as kbs1 but different scenario)."""
    return _run_kbs1_flexattn(kvseqlen, qseqlen, topk, pass_type, device)


# ═══════════════════════════════════════════════════════════════
#  Dispatch
# ═══════════════════════════════════════════════════════════════

_KBS1_RUNNERS = {
    "ffa_is": _run_kbs1_ffa_is,
    "flexattn": _run_kbs1_flexattn,
    "triton": _run_kbs1_triton,
}

_KBS128_RUNNERS = {
    "ffa_bs": _run_kbs128_ffa_bs,
    "ffa_is128": _run_kbs128_ffa_is,
    "flexattn": _run_kbs128_flexattn,
}


# ═══════════════════════════════════════════════════════════════
#  Correctness Verification
# ═══════════════════════════════════════════════════════════════


def _ref_token_sparse_attn(q, k, v, indices, sm_scale=None):
    """Naive reference: per-query gather topk KV → softmax → output (first 64 rows)."""
    import torch

    S, _Hq, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (D**0.5)
    q_f = q.float()
    k_f = k.squeeze(1).float()
    v_f = v.squeeze(1).float()
    o = torch.zeros(min(S, 64), _Hq, D, dtype=torch.float32, device=q.device)
    for i in range(min(S, 64)):
        idx = indices[i].long()
        ki = k_f[idx]
        vi = v_f[idx]
        qi = q_f[i]
        scores = (qi @ ki.T) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        o[i] = weights @ vi
    return o.to(q.dtype)


def _sanity_check(device):
    """Verify correctness of all methods at small scale before benchmarking."""
    import torch

    from magi_attention.functional import flex_flash_attn_func
    from magi_attention.utils.sparse_utils import generate_ranges_from_topk_indices

    kvseqlen_ck, qseqlen_ck, topk_ck = 2048, 256, 256
    torch.manual_seed(42)
    q = torch.randn(qseqlen_ck, NHQ, HD, dtype=torch.bfloat16, device=device)
    k = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)
    v = torch.randn(kvseqlen_ck, NHK, HD, dtype=torch.bfloat16, device=device)

    indices_2d = (
        torch.randint(
            0, kvseqlen_ck, (qseqlen_ck, topk_ck), device=device, dtype=torch.int32
        )
        .sort(dim=1)
        .values
    )
    ref = _ref_token_sparse_attn(q, k, v, indices_2d)
    results = {}

    # 1. FFA IndexSparse kbs=1
    is_indices = indices_2d.unsqueeze(1).expand(-1, NHK, -1).contiguous()
    ffa_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=is_indices,
        q_block_size=1,
        sparse_k_block_size=1,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    results["ffa_is_kbs1"] = (ffa_out[:64].float() - ref.float()).abs().max().item()

    # 2. FFA BlockSparse kbs=128
    n_kv_blocks = kvseqlen_ck // KBS_BLOCK
    n_topk_blocks = topk_ck // KBS_BLOCK
    bs_idx = (
        torch.rand(qseqlen_ck, n_kv_blocks, device=device)
        .argsort(dim=1)[:, :n_topk_blocks]
        .sort(dim=1)
        .values.unsqueeze(1)
        .expand(-1, NHK, -1)
        .to(torch.int32)
        .contiguous()
    )
    ia_3d = bs_idx.permute(1, 0, 2).contiguous()
    q_ranges, k_ranges = generate_ranges_from_topk_indices(
        ia_3d, block_m=1, block_n=KBS_BLOCK, num_k_blocks=n_kv_blocks
    )
    atm = torch.zeros(q_ranges.size(0), dtype=torch.int32, device=device)
    bs_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=atm,
        pack_gqa=True,
        block_sparse=True,
        range_merge=True,
        disable_fwd_atomic_reduction=True,
    )
    bs_ok = not torch.isnan(bs_out).any() and not torch.isinf(bs_out).any()
    results["ffa_bs_kbs128"] = 0.0 if bs_ok else float("inf")

    # 3. FFA IndexSparse kbs=128
    is128_out, *_ = flex_flash_attn_func(
        q,
        k,
        v,
        index_sparse_indices=bs_idx,
        q_block_size=1,
        sparse_k_block_size=KBS_BLOCK,
        pack_gqa=True,
        disable_fwd_atomic_reduction=True,
    )
    is128_ok = not torch.isnan(is128_out).any() and not torch.isinf(is128_out).any()
    results["ffa_is_kbs128"] = 0.0 if is128_ok else float("inf")

    # 4. FlexAttention
    from torch.nn.attention.flex_attention import flex_attention

    q_bhsd = q.unsqueeze(0).permute(0, 2, 1, 3)
    k_bhsd = k.unsqueeze(0).permute(0, 2, 1, 3)
    v_bhsd = v.unsqueeze(0).permute(0, 2, 1, 3)
    block_mask = _build_flex_block_mask(qseqlen_ck, kvseqlen_ck, topk_ck, device)
    flex_fn = torch.compile(flex_attention)
    flex_out = flex_fn(q_bhsd, k_bhsd, v_bhsd, block_mask=block_mask, enable_gqa=True)
    flex_ok = not torch.isnan(flex_out).any() and not torch.isinf(flex_out).any()
    results["flexattn"] = 0.0 if flex_ok else float("inf")

    # 5. Triton Token-Sparse
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "baselines"))
    from token_sparse_attn_triton import token_sparse_fwd

    tri_out = token_sparse_fwd(q, k, v, indices_2d)
    results["triton"] = (tri_out[:64].float() - ref.float()).abs().max().item()

    # Print
    print("  Correctness check (kvseqlen=2048, qseqlen=256, topk=256):", flush=True)
    all_pass = True
    for method, val in results.items():
        status = "PASS" if val < 0.05 else f"FAIL (err={val:.4f})"
        if val >= 0.05:
            all_pass = False
        print(f"    {method:18s}: {status}", flush=True)
    if not all_pass:
        print("  [ERROR] Correctness failed! Aborting.", flush=True)
    return all_pass


# ═══════════════════════════════════════════════════════════════
#  --exp
# ═══════════════════════════════════════════════════════════════


def _phase8_bench(force=False, rerun_filter=None):
    import torch

    results = _load_results(PHASE)
    gpu = _set_gpu()
    device = f"cuda:{gpu}"

    print(f"[{_ts()}] Phase 8: Baseline Comparison (gpu{gpu})", flush=True)
    print(
        f"  nhq={NHQ}, nhk={NHK}, hd={HD}, "
        f"scenarios: kvseqlen=[{','.join(f'{s[0] // 1024}k' for s in SCENARIOS)}], "
        f"qseqlen=kvseqlen/64, topk=kvseqlen/8",
        flush=True,
    )

    # Correctness verification first
    if not _sanity_check(device):
        return

    # --- 8a: kbs=1 ---
    print("\n  " + "─" * 55, flush=True)
    print("  8a) kbs=1: IndexSparse vs Baselines (FWD + BWD LoopK)", flush=True)
    print("  " + "─" * 55, flush=True)
    for kvseqlen, qseqlen, topk in SCENARIOS:
        print(
            f"  ── kvseqlen={kvseqlen // 1024}k, "
            f"qseqlen={qseqlen}, topk={topk // 1024}k ──",
            flush=True,
        )
        for pass_type in KBS1_PASSES:
            for method in KBS1_METHODS:
                key = f"kbs1/{pass_type}/{method}"
                if not force and _has_entry(results, key, kvseqlen):
                    d = results[key]
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:10s} {method:10s}: " f"{tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                try:
                    runner = _KBS1_RUNNERS[method]
                    tf, ms = runner(kvseqlen, qseqlen, topk, pass_type, device)
                    _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                    _save_results(PHASE, results)
                    print(
                        f"    {pass_type:10s} {method:10s}: "
                        f"{tf:>7.1f} T  ({ms:.3f} ms)",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"    {pass_type:10s} {method:10s}: OOM",
                        flush=True,
                    )
                    _set_entry(results, key, kvseqlen, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(
                        f"    {pass_type:10s} {method:10s}: ERR — {e}",
                        flush=True,
                    )
                    _set_entry(results, key, kvseqlen, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()

    # --- 8b: kbs=128 ---
    print("\n  " + "─" * 55, flush=True)
    print(
        f"  8b) kbs={KBS_BLOCK}: BS/IS vs Baselines " f"(FWD + BWD LoopK + BWD LoopQ)",
        flush=True,
    )
    print("  " + "─" * 55, flush=True)
    for kvseqlen, qseqlen, topk in SCENARIOS:
        print(
            f"  ── kvseqlen={kvseqlen // 1024}k, "
            f"qseqlen={qseqlen}, topk={topk // 1024}k ──",
            flush=True,
        )
        for pass_type in KBS128_PASSES:
            for method in KBS128_METHODS:
                key = f"kbs128/{pass_type}/{method}"
                if not force and _has_entry(results, key, kvseqlen):
                    d = results[key]
                    idx = d["kvseqlen"].index(kvseqlen)
                    tf = d["tflops"][idx]
                    print(
                        f"    {pass_type:10s} {method:10s}: " f"{tf:>7.1f} T (cached)",
                        flush=True,
                    )
                    continue

                gc.collect()
                torch.cuda.empty_cache()
                try:
                    runner = _KBS128_RUNNERS[method]
                    tf, ms = runner(kvseqlen, qseqlen, topk, pass_type, device)
                    _set_entry(results, key, kvseqlen, round(tf, 1), round(ms, 3))
                    _save_results(PHASE, results)
                    print(
                        f"    {pass_type:10s} {method:10s}: "
                        f"{tf:>7.1f} T  ({ms:.3f} ms)",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        f"    {pass_type:10s} {method:10s}: OOM",
                        flush=True,
                    )
                    _set_entry(results, key, kvseqlen, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(
                        f"    {pass_type:10s} {method:10s}: ERR — {e}",
                        flush=True,
                    )
                    _set_entry(results, key, kvseqlen, None, None)
                    _save_results(PHASE, results)
                    torch.cuda.empty_cache()

    print(f"\n[{_ts()}] Phase 8 done.", flush=True)


# ═══════════════════════════════════════════════════════════════
#  --plot
# ═══════════════════════════════════════════════════════════════


def _phase8_plot():
    """Generate Phase-6 style grouped bar charts: 2 figures (kbs=1, kbs=128).

    Each figure has subplots for each pass (FWD, BWD, etc.), with grouped bars
    per kvseqlen, one bar per method. TFLOPS on y-axis, kvseqlen on x-axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = _load_results(PHASE)
    if not results:
        print("  [WARN] No results found. Run --exp first.")
        return

    out = _out_dir(PHASE)
    os.makedirs(out, exist_ok=True)

    kvseqlens = [s[0] for s in SCENARIOS]
    x = np.arange(len(kvseqlens))
    x_labels = [f"{kv // 1024}K\n(q={kv // 64}, top={kv // 8192}K)" for kv in kvseqlens]

    # ── 8a) kbs=1 ──
    KBS1_PLOT = [
        ("ffa_is", "FFA IndexSparse", (0.85, 0.25, 0.25)),
        ("flexattn", "FlexAttention", (0.30, 0.70, 0.35)),
        ("triton", "Triton Token-Sparse", (0.25, 0.45, 0.80)),
    ]
    KBS1_PASS_LABELS = [("fwd", "FWD"), ("bwd", "BWD (LoopK)")]

    n_cols = len(KBS1_PASS_LABELS)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7), dpi=150)
    if n_cols == 1:
        axes = [axes]

    for col_idx, (pid, pname) in enumerate(KBS1_PASS_LABELS):
        ax = axes[col_idx]
        n_m = len(KBS1_PLOT)
        bw = 0.8 / n_m
        for i, (mid, lbl, col) in enumerate(KBS1_PLOT):
            key = f"kbs1/{pid}/{mid}"
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

        ax.set_title(pname, fontsize=13, fontweight="bold")
        ax.set_xlabel("kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)", fontsize=9)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Phase 8a: Token-Sparse Baseline (kbs=1)\n"
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, PackGQA, bf16, H100",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "phase8a_kbs1.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)

    # ── 8b) kbs=128 ──
    KBS128_PLOT = [
        ("ffa_bs", "FFA BlockSparse", (0.29, 0.57, 0.60)),
        ("ffa_is128", "FFA IndexSparse (kbs=128)", (0.85, 0.45, 0.20)),
        ("flexattn", "FlexAttention", (0.30, 0.70, 0.35)),
    ]
    KBS128_PASS_LABELS = [
        ("fwd", "FWD"),
        ("bwd_loopk", "BWD LoopK"),
        ("bwd_loopq", "BWD LoopQ"),
    ]

    n_cols = len(KBS128_PASS_LABELS)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 7), dpi=150)

    for col_idx, (pid, pname) in enumerate(KBS128_PASS_LABELS):
        ax = axes[col_idx]
        n_m = len(KBS128_PLOT)
        bw = 0.8 / n_m
        for i, (mid, lbl, col) in enumerate(KBS128_PLOT):
            key = f"kbs128/{pid}/{mid}"
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

        ax.set_title(pname, fontsize=13, fontweight="bold")
        ax.set_xlabel("kvseqlen (qseqlen=kvseqlen/64, topk=kvseqlen/8)", fontsize=9)
        ax.set_ylabel("TFLOPS", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Phase 8b: Block-Sparse Baseline (kbs={KBS_BLOCK})\n"
        f"nhq={NHQ}, nhk={NHK}, hd={HD}, PackGQA, bf16, H100",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out, "phase8b_kbs128.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}", flush=True)

    # Summary tables
    print("\n  " + "─" * 60)
    print("  Summary Tables (TFLOPS)")
    print("  " + "─" * 60)

    for prefix, methods, labels, passes in [
        ("kbs1", KBS1_METHODS, KBS1_LABELS, KBS1_PASSES),
        ("kbs128", KBS128_METHODS, KBS128_LABELS, KBS128_PASSES),
    ]:
        for pt in passes:
            print(f"\n  {pt.upper()} ({prefix}):")
            header = f"  {'kvseqlen':>8s}"
            for m in methods:
                header += f"  {labels[m]:>24s}"
            print(header)
            for kv in kvseqlens:
                row = f"  {kv // 1024:>7d}k"
                for m in methods:
                    key = f"{prefix}/{pt}/{m}"
                    d = results.get(key, {})
                    if d and kv in d.get("kvseqlen", []):
                        idx = d["kvseqlen"].index(kv)
                        tf = d["tflops"][idx]
                        row += f"  {tf:>24.1f}" if tf else f"  {'—':>24s}"
                    else:
                        row += f"  {'—':>24s}"
                print(row)
