"""Precompile all FFA JIT kernels needed by TestBlockSparseSimple / TestIndexSparseSimple.

Usage:
    python tests/precompile_sparse_tests.py

Strategy:
  1. Build 128hd_common once (avoids race condition)
  2. Build all kernel specs in parallel (ThreadPoolExecutor)
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from magi_attention.functional._flex_flash_attn_jit import get_ffa_jit_spec

BF16 = torch.bfloat16
ARCH = (9, 0)


def _spec(*, direction, head_dim=128, ref_block_size=None,
           disable_atomic=False, disable_dq_atomic=False,
           auto_range_merge=False, swap_ab=False,
           pack_gqa=False, pack_gqa_factor=1,
           block_sparse=False, index_sparse=False,
           swap_bwd_qk_loop=False, k_block_size=1,
           deterministic=False,
           bwd_dq_bf16=False, bwd_dkv_bf16=False):
    out_dtype = BF16 if direction == "fwd" else None
    dq_dtype = (BF16 if bwd_dq_bf16 else torch.float32) if direction == "bwd" else None
    dkv_dtype = (BF16 if bwd_dkv_bf16 else torch.float32) if direction == "bwd" else None
    spec, uri = get_ffa_jit_spec(
        arch=ARCH, direction=direction, head_dim=head_dim,
        compute_dtype=BF16, output_dtype=out_dtype,
        softcap=False,
        disable_atomic_reduction=disable_atomic,
        disable_dq_atomic_reduction=disable_dq_atomic,
        deterministic=deterministic,
        ref_block_size=ref_block_size,
        auto_range_merge=auto_range_merge,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa, pack_gqa_factor=pack_gqa_factor,
        block_sparse=block_sparse, index_sparse=index_sparse,
        swap_bwd_qk_loop=swap_bwd_qk_loop,
        dq_dtype=dq_dtype, dkv_dtype=dkv_dtype,
        k_block_size=k_block_size,
    )
    return spec, uri


def collect_specs():
    """Enumerate every (direction, params) needed by Simple tests."""
    specs = []

    # ── TestBlockSparseSimple ──

    # test_very_simple_block_sparse: 4 configs x fwd+bwd
    for ref_bs, swap_bwd in [
        ((64, 128), True),   # loopk q64k64
        ((128, 128), True),  # loopk q128k1
        ((64, 128), False),  # loopq q64k64
        ((128, 128), False), # loopq q128k1
    ]:
        for d in ("fwd", "bwd"):
            pack_f = 4  # nhq=16, nhk=4
            sbql = swap_bwd if d == "bwd" else False
            kbs = ref_bs[1] // 128 if ref_bs[1] >= 128 else 1
            if d == "fwd" or swap_bwd:
                specs.append(_spec(direction=d, ref_block_size=ref_bs if d == "fwd" else None,
                                   pack_gqa=True, pack_gqa_factor=pack_f,
                                   block_sparse=True, auto_range_merge=True,
                                   swap_bwd_qk_loop=sbql, k_block_size=kbs))
            else:
                specs.append(_spec(direction=d,
                                   pack_gqa=False, pack_gqa_factor=1,
                                   block_sparse=True, auto_range_merge=True,
                                   swap_bwd_qk_loop=False, k_block_size=kbs))

    # test_block_sparse_loopq_packgqa: nhq=128,nhk=1 dense ref + sparse loopq
    for kbs in (128, 96):
        for d in ("fwd", "bwd"):
            # dense reference (block_sparse=False)
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                               auto_range_merge=True))
            # sparse loopq (bwd only uses swap_bwd_qk_loop=False for loopq)
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                               block_sparse=True, auto_range_merge=True,
                               k_block_size=kbs // 128 if kbs >= 128 else 1))

    # test_block_sparse_swapab: 6 configs nhq=16,nhk=4
    for swap_ab, ref_bs in [
        (False, (128, 128)), (False, (64, 128)), (False, (128, 128)), (False, (64, 128)),
        (True, (32, 64)), (True, (16, 64)),
    ]:
        # FWD: swap_ab and ref_block_size as specified
        specs.append(_spec(direction="fwd", ref_block_size=ref_bs, swap_ab=swap_ab,
                           block_sparse=True, auto_range_merge=True))
        # BWD: no swap_ab, ref_block_size=None (default)
        specs.append(_spec(direction="bwd", block_sparse=True, auto_range_merge=True,
                           swap_bwd_qk_loop=True))

    # test_disable_atomic_* (block_sparse): nhq=128,nhk=1 or nhq=1,nhk=1
    # InnerLoopQ: disable_dkv_atomic → dkv=bf16, dq=f32
    # InnerLoopK: disable_dq_atomic → dq=bf16, dkv=f32
    # MHA: all disabled → both bf16
    for swap_bwd, nhq, nhk, pgqa, dq_bf, dkv_bf in [
        (False, 128, 1, True, False, True),   # innerloopq
        (True, 128, 1, True, True, False),    # innerloopk
        (False, 1, 1, False, True, True),     # mha
    ]:
        pack_f = nhq // nhk if pgqa else 1
        for d in ("fwd", "bwd"):
            sbql = swap_bwd if d == "bwd" else False
            specs.append(_spec(direction=d, pack_gqa=pgqa, pack_gqa_factor=pack_f,
                               block_sparse=True, auto_range_merge=True,
                               swap_bwd_qk_loop=sbql,
                               disable_atomic=True,
                               disable_dq_atomic=(d == "bwd" and swap_bwd),
                               k_block_size=1,
                               bwd_dq_bf16=dq_bf if d == "bwd" else False,
                               bwd_dkv_bf16=dkv_bf if d == "bwd" else False))

    # ── TestIndexSparseSimple ──

    # test_index_sparse_simple: 10 configs with various nhq/nhk/hd/pack_gqa
    idx_configs = [
        (128, 1, 128, True), (32, 4, 128, True), (4, 4, 64, False),
        (4, 2, 64, False), (4, 4, 64, False), (8, 2, 64, False),
        (4, 2, 128, True), (8, 2, 128, True), (4, 1, 128, True), (8, 4, 128, True),
    ]
    for nhq, nhk, hd, pgqa in idx_configs:
        pack_f = nhq // nhk if pgqa else 1
        for d in ("fwd", "bwd"):
            sbql = True if d == "bwd" else False
            specs.append(_spec(direction=d, head_dim=hd,
                               pack_gqa=pgqa, pack_gqa_factor=pack_f,
                               index_sparse=True, swap_bwd_qk_loop=sbql,
                               k_block_size=1))

    # view-trick tests: mqa/mha with index_sparse
    for nhq, nhk, hd in [(2, 1, 128), (4, 4, 128), (32, 32, 128)]:
        pack_f = nhq // nhk if nhk == 1 else 1
        pgqa = nhk == 1
        for d in ("fwd", "bwd"):
            sbql = True if d == "bwd" else False
            specs.append(_spec(direction=d, head_dim=hd,
                               pack_gqa=pgqa, pack_gqa_factor=pack_f,
                               index_sparse=True, swap_bwd_qk_loop=sbql,
                               k_block_size=1))

    # disable_atomic tests (index_sparse)
    # InnerLoopK (swap_bwd=True): disable_dq → dq=bf16, dkv=f32
    # FWD only (swap_bwd=None): no BWD needed, but if tested: default f32
    for swap_bwd, dq_bf, dkv_bf in [
        (True, True, False),   # innerloopk
        (None, False, False),  # fwd only
    ]:
        for d in ("fwd", "bwd"):
            sbql = (swap_bwd is True) if d == "bwd" else False
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                               index_sparse=True,
                               swap_bwd_qk_loop=sbql,
                               disable_atomic=True,
                               disable_dq_atomic=(d == "bwd" and swap_bwd is True),
                               k_block_size=1,
                               bwd_dq_bf16=dq_bf if d == "bwd" else False,
                               bwd_dkv_bf16=dkv_bf if d == "bwd" else False))

    # ── TestBlockSparseComprehensiveSweep ──

    # MQA128 configs: pack_gqa_factor=128, kBlockM=64 (assertion-fixed)
    for d in ("fwd", "bwd"):
        sbql = True if d == "bwd" else False
        specs.append(_spec(direction=d, ref_block_size=(64, 128) if d == "fwd" else None,
                           pack_gqa=True, pack_gqa_factor=128,
                           block_sparse=True, auto_range_merge=True,
                           swap_bwd_qk_loop=sbql, k_block_size=1))

    # D=64 variants
    for d in ("fwd", "bwd"):
        sbql = True if d == "bwd" else False
        # MHA D=64
        specs.append(_spec(direction=d, head_dim=64,
                           ref_block_size=(64, 128) if d == "fwd" else None,
                           pack_gqa=False, pack_gqa_factor=1,
                           block_sparse=True, auto_range_merge=True,
                           swap_bwd_qk_loop=sbql))
        # GQA D=64
        specs.append(_spec(direction=d, head_dim=64,
                           ref_block_size=(64, 128) if d == "fwd" else None,
                           pack_gqa=True, pack_gqa_factor=2,
                           block_sparse=True, auto_range_merge=True,
                           swap_bwd_qk_loop=sbql))

    # ── TestIndexSparseComprehensiveSweep ──

    # kbs=256 (FWD only — BWD BwdTileN=256 exceeds SM90 smem limit)
    specs.append(_spec(direction="fwd", pack_gqa=True, pack_gqa_factor=128,
                       index_sparse=True, swap_bwd_qk_loop=False,
                       k_block_size=256))

    # kbs=128 (already in Simple, but ensure included)
    for d in ("fwd", "bwd"):
        specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                           index_sparse=True, swap_bwd_qk_loop=False,
                           k_block_size=128))

    # kbs=8, kbs=32
    for kbs in (8, 32):
        for d in ("fwd", "bwd"):
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                               index_sparse=True, swap_bwd_qk_loop=False,
                               k_block_size=kbs))

    # Various GQA factors for IndexSparse comprehensive
    for nhq, nhk in [(64, 1), (32, 1), (16, 1), (4, 1), (128, 2), (32, 4), (4, 2)]:
        pack_f = nhq // nhk
        for d in ("fwd", "bwd"):
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=pack_f,
                               index_sparse=True, swap_bwd_qk_loop=False,
                               k_block_size=1))

    # D=64 IndexSparse variants (D=64 uses kBlockM=192, PackGQA epilogue not supported)
    for nhq, nhk in [(64, 1), (8, 2), (4, 4)]:
        for d in ("fwd", "bwd"):
            specs.append(_spec(direction=d, head_dim=64,
                               pack_gqa=False, pack_gqa_factor=1,
                               index_sparse=True, swap_bwd_qk_loop=False,
                               k_block_size=1))

    # Inner-mode variants: require env vars set during spec generation
    inner_mode_envs = [
        {"MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": "true"},
        {"MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN": "false"},
        {"MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD": "tma1d"},
        {"MAGI_ATTENTION_FFA_SPARSE_INNER_LOAD": "cpasync"},
        {"MAGI_ATTENTION_FFA_SPARSE_INNER_STORE": "cpasync"},
        {"MAGI_ATTENTION_FFA_SPARSE_INNER_STORE": "tma1d"},
    ]
    for env_dict in inner_mode_envs:
        for key, val in env_dict.items():
            os.environ[key] = val
        for d in ("fwd", "bwd"):
            specs.append(_spec(direction=d, pack_gqa=True, pack_gqa_factor=128,
                               index_sparse=True, swap_bwd_qk_loop=False,
                               k_block_size=1))
        for key in env_dict:
            os.environ.pop(key, None)

    return specs


def main():
    all_specs = collect_specs()

    # Deduplicate by URI
    seen = {}
    for spec, uri in all_specs:
        if uri not in seen:
            seen[uri] = spec
    print(f"Total unique kernels to precompile: {len(seen)}")

    # Step 1: Build 128hd_common first (avoid race condition)
    print("Step 1: Building 128hd_common (single thread)...")
    first_spec = next(iter(seen.values()))
    t0 = time.time()
    first_spec.build()
    print(f"  128hd_common ready ({time.time() - t0:.1f}s)")

    # Step 2: Build all remaining in parallel
    remaining = {uri: spec for uri, spec in seen.items()}
    print(f"Step 2: Building {len(remaining)} kernels in parallel...")
    t0 = time.time()

    def build_one(uri, spec):
        t = time.time()
        spec.build()
        return uri, time.time() - t

    with ThreadPoolExecutor(max_workers=len(remaining)) as ex:
        futs = {ex.submit(build_one, uri, spec): uri for uri, spec in remaining.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            uri, dt = fut.result()
            short = uri.split("/")[-1] if "/" in uri else uri
            print(f"  [{i}/{len(remaining)}] {short} ({dt:.1f}s)")

    print(f"\nAll {len(remaining)} kernels precompiled in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
