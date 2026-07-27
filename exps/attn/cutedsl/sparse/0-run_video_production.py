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
    _build_is_tma_bwd_loopq_bst,
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

METHODS = ["dense_nb", "d1b", "block_sparse", "index_sparse"]
PASSES = ["fwd", "bwd_q", "bwd"]

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


def _calc_flops(qseqlen, effective_k, is_bwd):
    """FLOPS: effective_k = kvseqlen for dense, topk for sparse."""
    fwd = 4 * qseqlen * effective_k * NHQ * HD
    return fwd * 2.5 if is_bwd else fwd


def _fwd_m_block(seqlen_q, qhpk, pack_gqa):
    seqlen_q_packgqa = seqlen_q * qhpk if pack_gqa else seqlen_q
    return 256 if seqlen_q_packgqa > 128 else 128


def _make_bst_fwd(sq, n_blocks, topk_blocks, per_q_sel):
    """BST for FWD or BWD-LoopK with per-Q selection."""
    qhpk = NHQ // NHK
    m_bs = _fwd_m_block(sq, qhpk, True)
    M = ceil(sq * qhpk / m_bs)
    mask_cnt = torch.zeros(1, NHK, M, dtype=torch.int32, device="cuda")
    mask_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_cnt = torch.full((1, NHK, M), topk_blocks, dtype=torch.int32, device="cuda")
    full_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    for m in range(M):
        q_pos = min((m * m_bs) // qhpk, sq - 1)
        full_idx[0, 0, m, :topk_blocks] = per_q_sel[q_pos]
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(m_bs, N_BLOCK_SIZE),
    )


def _make_bst_bwd(sq, n_blocks, topk_blocks, per_q_sel):
    """BST for BWD-LoopK (M-indexed, m_bs=128) with per-Q selection."""
    qhpk = NHQ // NHK
    m_bs = 128
    M = ceil(sq * qhpk / m_bs)
    mask_cnt = torch.zeros(1, NHK, M, dtype=torch.int32, device="cuda")
    mask_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    full_cnt = torch.full((1, NHK, M), topk_blocks, dtype=torch.int32, device="cuda")
    full_idx = torch.zeros(1, NHK, M, n_blocks, dtype=torch.int32, device="cuda")
    for m in range(M):
        q_pos = min((m * m_bs) // qhpk, sq - 1)
        full_idx[0, 0, m, :topk_blocks] = per_q_sel[q_pos]
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(m_bs, N_BLOCK_SIZE),
    )


def _make_bst_loopq(sq, n_blocks, topk_blocks, per_q_sel):
    """BST for BWD-InnerLoopQ (N-indexed -> M-block list) with per-Q inversion.

    For each K-block, finds which coarse Q-blocks (packed) attend to it.
    With per-Q selection, almost all K-blocks have work -> no wave quantization.
    """
    qhpk = NHQ // NHK
    sq_packed = sq * qhpk
    sparse_q = 2 * 128  # subtile_factor * m_block_size
    M_coarse = max(ceil(sq_packed / sparse_q), 1)
    device = per_q_sel.device

    # Map Q positions to coarse Q-block starts in packed space
    q_arange = torch.arange(sq, device=device)
    coarse_starts = (q_arange * qhpk) // sparse_q  # (SQ,)

    # Flatten (q_pos, k_block) pairs
    flat_k = per_q_sel.reshape(-1).long()  # (SQ * topk_blocks,)
    flat_cq = coarse_starts.unsqueeze(1).expand(-1, topk_blocks).reshape(-1)

    # Group by K-block and deduplicate coarse Q indices
    sort_idx = flat_k.argsort(stable=True)
    sorted_k = flat_k[sort_idx]
    sorted_cq = flat_cq[sort_idx]

    changes = torch.cat(
        [torch.tensor([True], device=device), sorted_k[1:] != sorted_k[:-1]]
    )
    group_starts = changes.nonzero(as_tuple=True)[0]
    group_k_vals = sorted_k[group_starts]

    bwd_cnt = torch.zeros(1, NHK, n_blocks, dtype=torch.int32, device=device)
    max_cnt = 0
    k_to_cq = {}
    for gi in range(len(group_starts)):
        k_val = int(group_k_vals[gi].item())
        start = int(group_starts[gi].item())
        end = (
            int(group_starts[gi + 1].item())
            if gi + 1 < len(group_starts)
            else len(sorted_k)
        )
        cq_unique = sorted_cq[start:end].unique()
        cq_unique = cq_unique[cq_unique < M_coarse]
        k_to_cq[k_val] = cq_unique
        cnt = len(cq_unique)
        bwd_cnt[0, 0, k_val] = cnt
        max_cnt = max(max_cnt, cnt)

    max_cnt = max(max_cnt, 1)
    bwd_idx = torch.zeros(1, NHK, n_blocks, max_cnt, dtype=torch.int32, device=device)
    for k_val, cq_vals in k_to_cq.items():
        if len(cq_vals) > 0:
            bwd_idx[0, 0, k_val, : len(cq_vals)] = cq_vals.int()

    empty_cnt = torch.zeros(1, NHK, n_blocks, dtype=torch.int32, device=device)
    empty_idx = torch.zeros(1, NHK, n_blocks, 1, dtype=torch.int32, device=device)
    return BlockSparseTensorsTorch(
        mask_block_cnt=empty_cnt,
        mask_block_idx=empty_idx,
        full_block_cnt=bwd_cnt,
        full_block_idx=bwd_idx,
        block_size=(sparse_q, N_BLOCK_SIZE),
    )


def _make_perq_block_sel(sq, n_blocks, topk_blocks):
    """Per-coarse-Q-block random K-block selection.

    Adjacent pairs of Q positions share the same K-block selection, aligned to
    BWD InnerLoopQ coarse block boundary (subtile_factor=2, tile_m=128,
    qhpk=128 -> 2 Q per coarse block). This eliminates wasted computation from
    partially-valid coarse blocks while preserving the same per-Q FLOPS formula.
    """
    group_size = 2
    n_groups = (sq + group_size - 1) // group_size
    gen = torch.Generator(device="cuda").manual_seed(42)
    rand_vals = torch.rand(n_groups, n_blocks, generator=gen, device="cuda")
    group_sel = rand_vals.argsort(dim=1)[:, :topk_blocks].sort(dim=1).values
    per_q_sel = group_sel.repeat_interleave(group_size, dim=0)[:sq]
    return per_q_sel.int()


def _make_is_indices(sq, n_blocks, topk_blocks, per_q_sel):
    """Build block-aligned token indices from per-Q selection.

    per_q_sel: (SQ, topk_blocks) K-block indices per Q position.
    Expands to token addresses: block_id * 128 + offset.
    """
    topk = topk_blocks * N_BLOCK_SIZE
    offsets = torch.arange(N_BLOCK_SIZE, device="cuda", dtype=torch.int32)
    tokens = per_q_sel.unsqueeze(-1) * N_BLOCK_SIZE + offsets.view(1, 1, -1)
    tokens = tokens.reshape(sq, topk).int()
    return tokens.unsqueeze(0).unsqueeze(0).expand(1, NHQ, sq, topk)


def _make_is_tiles_fwd(sq, sk, n_blocks, topk_blocks, per_q_sel):
    """IS tiles for FWD (m=256), kbs=128."""
    indices = _make_is_indices(sq, n_blocks, topk_blocks, per_q_sel)
    qhpk = NHQ // NHK
    fwd_m = _fwd_m_block(sq, qhpk, True)
    return prepare_index_sparse_tiles(
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


def _make_is_tiles_bwd(sq, sk, n_blocks, topk_blocks, per_q_sel):
    """IS tiles for BWD-LoopK (m=128), kbs=128."""
    indices = _make_is_indices(sq, n_blocks, topk_blocks, per_q_sel)
    return prepare_index_sparse_tiles(
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


def _run_experiment(force=False, max_kvseqlen=None):
    device = "cuda"
    results = {} if force else _load_results()

    scenarios = SCENARIOS
    if max_kvseqlen is not None:
        scenarios = [(kv, q, t) for kv, q, t in SCENARIOS if kv <= max_kvseqlen]

    print(f"[{_ts()}] CuTeDSL SM100 Sparse Bench: Video Production", flush=True)
    print(f"  NHQ={NHQ}, NHK={NHK}, HD={HD}, PackGQA, bf16, B300", flush=True)
    print("  Methods: Dense (gathered KV) / BS kbs=128 / IS kbs=128\n", flush=True)

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
        per_q_sel = _make_perq_block_sel(qseqlen, n_blocks, topk_blocks)

        for pass_type in PASSES:
            is_bwd = pass_type in ("bwd", "bwd_q")
            swap_loop = pass_type == "bwd"  # bwd=LoopK, bwd_q=InnerLoopQ

            for method in METHODS:
                effective_k = kvseqlen if method == "dense_nb" else topk
                flops = _calc_flops(qseqlen, effective_k, is_bwd)
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

                    if method == "dense_nb":
                        # Dense-NB: full KV (seqlen_k=kvseqlen), no sparsity
                        # outer K-tiles = kvseqlen/128 -> no wave quantization
                        # TFLOPS for reference only (FLOPS != sparse FLOPS)
                        k = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
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
                                    swap_bwd_qk_loop=swap_loop,
                                )

                    elif method == "d1b":
                        # Dense-1B: gathered KV (seqlen_k=topk), same FLOPS as sparse
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
                                    swap_bwd_qk_loop=swap_loop,
                                )

                    elif method == "block_sparse":
                        k = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        fwd_bst = _make_bst_fwd(
                            qseqlen, n_blocks, topk_blocks, per_q_sel
                        )
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
                            if swap_loop:
                                bwd_bst = _make_bst_bwd(
                                    qseqlen, n_blocks, topk_blocks, per_q_sel
                                )
                                bwd_args = TorchFlexAttnArgs(
                                    block_sparse_tensors=bwd_bst
                                )
                            else:
                                loopq_bst = _make_bst_loopq(
                                    qseqlen, n_blocks, topk_blocks, per_q_sel
                                )
                                bwd_args = TorchFlexAttnArgs(
                                    block_sparse_tensors_bwd=loopq_bst
                                )
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
                                    swap_bwd_qk_loop=swap_loop,
                                )

                    elif method == "index_sparse":
                        k = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        v = torch.randn(
                            1, kvseqlen, NHK, HD, dtype=torch.bfloat16, device=device
                        )
                        if not is_bwd:
                            fwd_tiles = _make_is_tiles_fwd(
                                qseqlen, kvseqlen, n_blocks, topk_blocks, per_q_sel
                            )
                            fwd_args = TorchFlexAttnArgs(index_sparse_tiles=fwd_tiles)
                            _flex_flash_attn_fwd(
                                q,
                                k,
                                v,
                                softmax_scale=scale,
                                flex_attn_args=fwd_args,
                                pack_gqa=True,
                            )

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
                            fwd_tiles = _make_is_tiles_fwd(
                                qseqlen, kvseqlen, n_blocks, topk_blocks, per_q_sel
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
                            del fwd_tiles, fwd_args
                            gc.collect()
                            torch.cuda.empty_cache()

                            if swap_loop:
                                # BWD LoopK: pass IS tiles directly
                                bwd_tiles = _make_is_tiles_bwd(
                                    qseqlen, kvseqlen, n_blocks, topk_blocks, per_q_sel
                                )
                                bwd_args = TorchFlexAttnArgs(
                                    index_sparse_tiles=bwd_tiles
                                )
                            else:
                                # BWD InnerLoopQ: pre-compute inverted BST
                                bwd_tiles = _make_is_tiles_bwd(
                                    qseqlen, kvseqlen, n_blocks, topk_blocks, per_q_sel
                                )
                                is_loopq_bst = _build_is_tma_bwd_loopq_bst(
                                    bwd_tiles,
                                    seqlen_q=qseqlen,
                                    seqlen_k=kvseqlen,
                                    num_kv_heads=NHK,
                                    num_q_heads=NHQ,
                                    m_block_size=128,
                                    n_block_size=N_BLOCK_SIZE,
                                    pack_gqa=True,
                                )
                                del bwd_tiles
                                bwd_args = TorchFlexAttnArgs(
                                    block_sparse_tensors_bwd=is_loopq_bst
                                )
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
                                    swap_bwd_qk_loop=swap_loop,
                                )

                            # Warmup: first call triggers JIT
                            try:
                                run_fn()
                            except Exception:
                                pass

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
        row = f"  {kvseqlen // 1024:>5d}k {qseqlen:>6d} {topk // 1024:>5d}k"
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

    PLOT_PASSES = [
        ("fwd", "FWD"),
        ("bwd_q", "BWD InnerLoopQ"),
        ("bwd", "BWD InnerLoopK"),
    ]
    PLOT_METHODS = [
        ("dense_nb", "Dense-NB (full KV)", (0.2, 0.2, 0.2)),
        ("d1b", "Dense (K=topk)", (0.58, 0.58, 0.58)),
        ("block_sparse", "BlockSparse (kbs=128)", (0.29, 0.57, 0.60)),
        ("index_sparse", "IndexSparse (kbs=128)", (0.77, 0.34, 0.49)),
    ]

    kvseqlens = sorted(
        set(kv for d in results.values() for kv in d.get("kvseqlen", []))
    )
    if not kvseqlens:
        print("No data to plot.")
        return

    x = np.arange(len(kvseqlens))
    bw = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(26, 7), dpi=150)

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
        ax.set_ylim(0, 2500)
        ax.axhline(y=2500, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
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
