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


def _spec(
    *,
    direction,
    head_dim=128,
    ref_block_size=None,
    disable_atomic=False,
    disable_dq_atomic=False,
    auto_range_merge=False,
    swap_ab=False,
    pack_gqa=False,
    pack_gqa_factor=1,
    block_sparse=False,
    index_sparse=False,
    bwd_inner_loop_k=False,
    sparse_k_block_size=1,
    deterministic=False,
    bwd_dq_bf16=False,
    bwd_dkv_bf16=False,
):
    out_dtype = BF16 if direction == "fwd" else None
    dq_dtype = (BF16 if bwd_dq_bf16 else torch.float32) if direction == "bwd" else None
    dkv_dtype = (
        (BF16 if bwd_dkv_bf16 else torch.float32) if direction == "bwd" else None
    )
    spec, uri = get_ffa_jit_spec(
        arch=ARCH,
        direction=direction,
        head_dim=head_dim,
        compute_dtype=BF16,
        output_dtype=out_dtype,
        softcap=False,
        disable_atomic_reduction=disable_atomic,
        disable_dq_atomic_reduction=disable_dq_atomic,
        deterministic=deterministic,
        ref_block_size=ref_block_size,
        auto_range_merge=auto_range_merge,
        swap_ab=swap_ab,
        pack_gqa=pack_gqa,
        pack_gqa_factor=pack_gqa_factor,
        block_sparse=block_sparse,
        index_sparse=index_sparse,
        bwd_inner_loop_k=bwd_inner_loop_k,
        dq_dtype=dq_dtype,
        dkv_dtype=dkv_dtype,
        sparse_k_block_size=sparse_k_block_size,
    )
    return spec, uri


def collect_specs():
    """Enumerate every (direction, params) needed by Simple tests."""
    specs = []

    # ── TestBlockSparseSimple ──

    # test_very_simple_block_sparse: 4 configs x fwd+bwd
    for ref_bs, swap_bwd in [
        ((64, 128), True),  # loopk q64k64
        ((128, 128), True),  # loopk q128k1
        ((64, 128), False),  # loopq q64k64
        ((128, 128), False),  # loopq q128k1
    ]:
        for d in ("fwd", "bwd"):
            pack_f = 4  # nhq=16, nhk=4
            sbql = swap_bwd if d == "bwd" else False
            kbs = ref_bs[1] // 128 if ref_bs[1] >= 128 else 1
            if d == "fwd" or swap_bwd:
                specs.append(
                    _spec(
                        direction=d,
                        ref_block_size=ref_bs if d == "fwd" else None,
                        pack_gqa=True,
                        pack_gqa_factor=pack_f,
                        block_sparse=True,
                        auto_range_merge=True,
                        bwd_inner_loop_k=sbql,
                        sparse_k_block_size=kbs,
                    )
                )
            else:
                specs.append(
                    _spec(
                        direction=d,
                        pack_gqa=False,
                        pack_gqa_factor=1,
                        block_sparse=True,
                        auto_range_merge=True,
                        bwd_inner_loop_k=False,
                        sparse_k_block_size=kbs,
                    )
                )

    # test_block_sparse_loopq_packgqa: nhq=128,nhk=1 dense ref + sparse loopq
    for kbs in (128, 96):
        for d in ("fwd", "bwd"):
            # dense reference (block_sparse=False)
            specs.append(
                _spec(
                    direction=d,
                    pack_gqa=True,
                    pack_gqa_factor=128,
                    auto_range_merge=True,
                )
            )
            # sparse loopq (bwd only uses bwd_inner_loop_k=False for loopq)
            specs.append(
                _spec(
                    direction=d,
                    pack_gqa=True,
                    pack_gqa_factor=128,
                    block_sparse=True,
                    auto_range_merge=True,
                    sparse_k_block_size=kbs // 128 if kbs >= 128 else 1,
                )
            )

    # test_block_sparse_swapab: 6 configs nhq=16,nhk=4
    for swap_ab, ref_bs in [
        (False, (128, 128)),
        (False, (64, 128)),
        (False, (128, 128)),
        (False, (64, 128)),
        (True, (32, 64)),
        (True, (16, 64)),
    ]:
        # FWD: swap_ab and ref_block_size as specified
        specs.append(
            _spec(
                direction="fwd",
                ref_block_size=ref_bs,
                swap_ab=swap_ab,
                block_sparse=True,
                auto_range_merge=True,
            )
        )
        # BWD: no swap_ab, ref_block_size=None (default)
        specs.append(
            _spec(
                direction="bwd",
                block_sparse=True,
                auto_range_merge=True,
                bwd_inner_loop_k=True,
            )
        )

    # test_disable_atomic_* (block_sparse): nhq=128,nhk=1 or nhq=1,nhk=1
    # InnerLoopQ: disable_dkv_atomic → dkv=bf16, dq=f32
    # InnerLoopK: disable_dq_atomic → dq=bf16, dkv=f32
    # MHA: all disabled → both bf16
    for swap_bwd, nhq, nhk, pgqa, dq_bf, dkv_bf in [
        (False, 128, 1, True, False, True),  # innerloopq
        (True, 128, 1, True, True, False),  # innerloopk
        (False, 1, 1, False, True, True),  # mha
    ]:
        pack_f = nhq // nhk if pgqa else 1
        for d in ("fwd", "bwd"):
            sbql = swap_bwd if d == "bwd" else False
            specs.append(
                _spec(
                    direction=d,
                    pack_gqa=pgqa,
                    pack_gqa_factor=pack_f,
                    block_sparse=True,
                    auto_range_merge=True,
                    bwd_inner_loop_k=sbql,
                    disable_atomic=True,
                    disable_dq_atomic=(d == "bwd" and swap_bwd),
                    sparse_k_block_size=1,
                    bwd_dq_bf16=dq_bf if d == "bwd" else False,
                    bwd_dkv_bf16=dkv_bf if d == "bwd" else False,
                )
            )

    # ── TestIndexSparseSimple ──

    # test_index_sparse_simple: 10 configs with various nhq/nhk/hd/pack_gqa
    idx_configs = [
        (128, 1, 128, True),
        (32, 4, 128, True),
        (4, 4, 64, False),
        (4, 2, 64, False),
        (4, 4, 64, False),
        (8, 2, 64, False),
        (4, 2, 128, True),
        (8, 2, 128, True),
        (4, 1, 128, True),
        (8, 4, 128, True),
    ]
    for nhq, nhk, hd, pgqa in idx_configs:
        pack_f = nhq // nhk if pgqa else 1
        for d in ("fwd", "bwd"):
            sbql = True if d == "bwd" else False
            specs.append(
                _spec(
                    direction=d,
                    head_dim=hd,
                    pack_gqa=pgqa,
                    pack_gqa_factor=pack_f,
                    index_sparse=True,
                    bwd_inner_loop_k=sbql,
                    sparse_k_block_size=1,
                )
            )

    # view-trick tests: mqa/mha with index_sparse
    for nhq, nhk, hd in [(2, 1, 128), (4, 4, 128), (32, 32, 128)]:
        pack_f = nhq // nhk if nhk == 1 else 1
        pgqa = nhk == 1
        for d in ("fwd", "bwd"):
            sbql = True if d == "bwd" else False
            specs.append(
                _spec(
                    direction=d,
                    head_dim=hd,
                    pack_gqa=pgqa,
                    pack_gqa_factor=pack_f,
                    index_sparse=True,
                    bwd_inner_loop_k=sbql,
                    sparse_k_block_size=1,
                )
            )

    # disable_atomic tests (index_sparse)
    # InnerLoopK (swap_bwd=True): disable_dq → dq=bf16, dkv=f32
    # FWD only (swap_bwd=None): no BWD needed, but if tested: default f32
    for swap_bwd, dq_bf, dkv_bf in [
        (True, True, False),  # innerloopk
        (None, False, False),  # fwd only
    ]:
        for d in ("fwd", "bwd"):
            sbql = (swap_bwd is True) if d == "bwd" else False
            specs.append(
                _spec(
                    direction=d,
                    pack_gqa=True,
                    pack_gqa_factor=128,
                    index_sparse=True,
                    bwd_inner_loop_k=sbql,
                    disable_atomic=True,
                    disable_dq_atomic=(d == "bwd" and swap_bwd is True),
                    sparse_k_block_size=1,
                    bwd_dq_bf16=dq_bf if d == "bwd" else False,
                    bwd_dkv_bf16=dkv_bf if d == "bwd" else False,
                )
            )

    # ── TestBlockSparseComprehensiveSweep ──
    # Runtime auto-flags for block_sparse (flex_flash_attn_func):
    #   FWD: disable_atomic=True, ref_block_size overridden to (128,128)
    #   BWD: disable_dq_atomic=True + dq_dtype=bf16 when bwd_inner_loop_k=True
    #   sparse_k_block_size: auto-derived from k_ranges (= k_size in block config)
    comprehensive_block_sparse = [
        # (pack_gqa_factor, sparse_k_block_size_list, head_dim)
        # Cross-product: nhq_nhk_hd × q_size_k_size → kbs from k_size
        (1, [1, 8, 64, 128], 128),  # MHA8 (nhq=8,nhk=8)
        (4, [1, 8, 64, 128], 128),  # GQA16x4 (nhq=16,nhk=4)
        (128, [1, 8, 64, 128], 128),  # MQA128 (nhq=128,nhk=1)
        (1, [1, 8, 64, 128], 64),  # MHA1 D=64 (nhq=1,nhk=1)
        (2, [1, 8, 64, 128], 64),  # GQA4x2 D=64 (nhq=4,nhk=2)
    ]
    for pgqa, kbs_list, hd in comprehensive_block_sparse:
        for kbs in kbs_list:
            specs.append(
                _spec(
                    direction="fwd",
                    head_dim=hd,
                    ref_block_size=(128, 128),
                    disable_atomic=True,
                    pack_gqa=True,
                    pack_gqa_factor=pgqa,
                    block_sparse=True,
                    auto_range_merge=True,
                    sparse_k_block_size=kbs,
                )
            )
            specs.append(
                _spec(
                    direction="bwd",
                    head_dim=hd,
                    disable_dq_atomic=True,
                    pack_gqa=True,
                    pack_gqa_factor=pgqa,
                    block_sparse=True,
                    auto_range_merge=True,
                    bwd_inner_loop_k=True,
                    sparse_k_block_size=kbs,
                    bwd_dq_bf16=True,
                )
            )

    # ── TestIndexSparseComprehensiveSweep ──
    # Cross-product: nhq_nhk_hd_packgqa × kbs
    # kbs>1 only for NHK=1, pack_gqa=True, D=128

    # kbs=1: all unique (pack_gqa, pack_gqa_factor, head_dim) after view-trick
    idx_comp_kbs1 = [
        (True, 128, 128),  # MQA128
        (True, 64, 128),  # MQA64 / GQA128x2 rearranged
        (True, 32, 128),  # MQA32
        (True, 16, 128),  # MQA16
        (True, 8, 128),  # GQA32x4 rearranged
        (True, 4, 128),  # MQA4
        (True, 2, 128),  # GQA4x2 rearranged
        (True, 1, 128),  # MHA8 rearranged
        (False, 1, 64),  # D=64 no packgqa
    ]
    for pgqa_on, pgqa_f, hd in idx_comp_kbs1:
        for d in ("fwd", "bwd"):
            specs.append(
                _spec(
                    direction=d,
                    head_dim=hd,
                    pack_gqa=pgqa_on,
                    pack_gqa_factor=pgqa_f,
                    index_sparse=True,
                    bwd_inner_loop_k=False,
                    sparse_k_block_size=1,
                )
            )

    # kbs>1: MQA configs (NHK=1, pack_gqa=True, D=128)
    for pgqa_f in [128, 64, 32, 16, 4]:
        for kbs in [8, 32, 128]:
            for d in ("fwd", "bwd"):
                specs.append(
                    _spec(
                        direction=d,
                        pack_gqa=True,
                        pack_gqa_factor=pgqa_f,
                        index_sparse=True,
                        bwd_inner_loop_k=False,
                        sparse_k_block_size=kbs,
                    )
                )
        # kbs=256: FWD only (BWD exceeds SM90 smem limit)
        specs.append(
            _spec(
                direction="fwd",
                pack_gqa=True,
                pack_gqa_factor=pgqa_f,
                index_sparse=True,
                bwd_inner_loop_k=False,
                sparse_k_block_size=256,
            )
        )

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
            specs.append(
                _spec(
                    direction=d,
                    pack_gqa=True,
                    pack_gqa_factor=128,
                    index_sparse=True,
                    bwd_inner_loop_k=False,
                    sparse_k_block_size=1,
                )
            )
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
