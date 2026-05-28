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

"""Repro script for CI #600 hang: InvCausal + swap_bwd_qk_loop.

All 6 hanging cases in CI had attn_type=2 (InvCausal) + swap_bwd_qk_loop=True.
This script tests various seqlen/head configs to reproduce the hang locally.
"""

import datetime
import sys

import torch

sys.path.insert(0, "/home/niubility2/cenzhiyao/MagiAttention")


def log(msg: str):
    print(f"[{datetime.datetime.now()}] {msg}", flush=True)


def run_case(
    name: str,
    seqlen_q: int,
    seqlen_k: int,
    nheads_q: int,
    nheads_kv: int,
    headdim: int,
    dtype: torch.dtype,
    swap_ab: bool = False,
    pack_gqa: bool = False,
    num_pairs: int = 10,
):
    from magi_attention.functional.flex_flash_attn import flex_flash_attn_func

    log(
        f"START [{name}] q={seqlen_q} k={seqlen_k} nhq={nheads_q} nhkv={nheads_kv}"
        f" hd={headdim} dtype={dtype} swap_ab={swap_ab} pack_gqa={pack_gqa}"
    )

    device = "cuda"

    # Q/K/V shape: (total_seqlen, nheads, headdim) — single-batch
    q = torch.randn(
        seqlen_q, nheads_q, headdim, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        seqlen_k, nheads_kv, headdim, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        seqlen_k, nheads_kv, headdim, device=device, dtype=dtype, requires_grad=True
    )

    q_ranges = torch.tensor([[0, seqlen_q]], device=device, dtype=torch.int32)
    k_ranges = torch.tensor([[0, seqlen_k]], device=device, dtype=torch.int32)
    attn_type_map = torch.tensor([2], device=device, dtype=torch.int32)  # InvCausal

    log("  tensors created, calling flex_flash_attn_func (FWD)...")

    try:
        out, meta = flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            swap_ab=swap_ab,
            pack_gqa=pack_gqa,
            swap_bwd_qk_loop=True,
        )
        log(f"  FWD done, out.shape={out.shape}, calling backward...")

        dout = torch.randn_like(out)
        out.backward(dout)

        log(f"  BWD done, dq.shape={q.grad.shape}")
    except Exception as e:
        log(f"  ERROR: {e}")

    log(f"FINISH [{name}]")
    print()


if __name__ == "__main__":
    log("=== repro_ci600.py: InvCausal + swap_bwd_qk_loop hang repro ===")

    # Case 1: mha hd128 q1024_k1024 (should PASS - passed in CI)
    run_case("mha_hd128_q1k_k1k_bf16", 1024, 1024, 8, 8, 128, torch.bfloat16)

    # Case 2: mha hd128 q2048_k2048 fp16 (HANG in CI - L1758)
    run_case("mha_hd128_q2k_k2k_fp16", 2048, 2048, 8, 8, 128, torch.float16)

    # Case 3: mha hd128 q2048_k4096 sparse-like (HANG in CI - L1897)
    run_case("mha_hd128_q2k_k4k_bf16", 2048, 4096, 8, 8, 128, torch.bfloat16)

    # Case 4: gqa hd128 q2048_k4096 pack_gqa (HANG in CI - L2711)
    run_case(
        "gqa_hd128_q2k_k4k_packgqa",
        2048,
        4096,
        32,
        4,
        128,
        torch.bfloat16,
        pack_gqa=True,
    )

    # Case 5: gqa hd128 q2048_k2048 swap_ab (HANG in CI - L3341)
    run_case(
        "gqa_hd128_q2k_k2k_swapab", 2048, 2048, 32, 4, 128, torch.bfloat16, swap_ab=True
    )

    # Case 6: gqa hd64 q1024_k1024 swap_ab+pack_gqa (HANG in CI - L3741)
    run_case(
        "gqa_hd64_q1k_k1k_swapab_packgqa",
        1024,
        1024,
        4,
        2,
        64,
        torch.float16,
        swap_ab=True,
        pack_gqa=True,
    )

    log("=== ALL CASES DONE ===")
