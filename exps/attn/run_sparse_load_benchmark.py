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
Benchmark: FFA SparseLoad FWD+BWD performance.

nhq=128, nhk=1 (MQA), head_dim=128, PackGQA, block-sparse.
q_block_size=1, k_block_size=128. Fixed effective_kv=2048 (16 k-blocks).

Usage:
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/run_sparse_load_benchmark.py
    CUDA_HOME=/usr/local/cuda-13.0 python exps/attn/run_sparse_load_benchmark.py --bwd
"""

import torch
from baselines.attn_impl import ffa_func
from baselines.utils import seed_everything
from triton.testing import do_bench

from magi_attention.benchmarking import BENCH_CASE_OOM
from magi_attention.utils.sparse_utils import generate_ranges_from_block_mask_triton

# ─── Config ───────────────────────────────────────────────────────────────────

nhq = 128
nhk = 1
hd = 128
q_block_size = 1
k_block_size = 128
effective_kv = 2048  # each q-token attends to 2048 KV tokens = 16 k-blocks
n_attend = effective_kv // k_block_size  # 16
dtype = torch.bfloat16

seqlen_vals = [32768, 65536, 131072, 262144, 524288]

seed_everything()


def build_block_sparse_inputs(S, device, requires_grad=False):
    """Build q/k/v + q_ranges/k_ranges for SparseLoad block-sparse path."""
    n_q_blocks = S // q_block_size  # = S (one block per token)
    n_k_blocks = S // k_block_size
    actual_attend = min(n_attend, n_k_blocks)

    # 4D block_mask: (1, nhk, n_q_blocks, n_k_blocks)
    block_mask = torch.zeros(
        1, nhk, n_q_blocks, n_k_blocks, dtype=torch.bool, device=device
    )
    for qb in range(n_q_blocks):
        perm = torch.randperm(n_k_blocks, device=device)[:actual_attend]
        block_mask[0, 0, qb, perm] = True

    q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
        block_mask, q_block_size, k_block_size
    )
    attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)

    q = torch.randn(S, nhq, hd, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(S, nhk, hd, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(S, nhk, hd, device=device, dtype=dtype, requires_grad=requires_grad)

    return q, k, v, q_ranges, k_ranges, attn_type_map


def bench_fwd(S):
    device = torch.cuda.current_device()
    n_k_blocks = S // k_block_size
    actual_attend = min(n_attend, n_k_blocks)
    actual_eff_kv = actual_attend * k_block_size
    sparse_flops = 4 * S * actual_eff_kv * nhq * hd

    try:
        q, k, v, q_ranges, k_ranges, attn_type_map = build_block_sparse_inputs(
            S, device
        )
        torch.cuda.empty_cache()

        def fn():
            return ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                sparse_load=True,
                auto_range_merge=True,
                pack_gqa=True,
            )

        # TODO: switch to magi_attention.benchmarking.do_bench instead of triton's do_bench
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = sparse_flops / ms * 1e-9
        return tflops
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [FWD] S={S}: OOM — {e}")
        torch.cuda.empty_cache()
        return BENCH_CASE_OOM
    except Exception as e:
        print(f"  [FWD] S={S}: {e}")
        torch.cuda.empty_cache()
        return BENCH_CASE_OOM


def bench_bwd(S):
    device = torch.cuda.current_device()
    n_k_blocks = S // k_block_size
    actual_attend = min(n_attend, n_k_blocks)
    actual_eff_kv = actual_attend * k_block_size
    sparse_flops = 4 * S * actual_eff_kv * nhq * hd * 2.5

    try:
        q, k, v, q_ranges, k_ranges, attn_type_map = build_block_sparse_inputs(
            S, device, requires_grad=True
        )
        torch.cuda.empty_cache()

        out, _ = ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            sparse_load=True,
            auto_range_merge=True,
            pack_gqa=True,
            swap_bwd_qk_loop=True,
        )
        do = torch.randn_like(out)

        def fn():
            out.backward(do, retain_graph=True)

        ms = do_bench(fn, warmup=25, rep=100)
        tflops = sparse_flops / ms * 1e-9
        return tflops
    except torch.cuda.OutOfMemoryError as e:
        print(f"  [BWD] S={S}: OOM — {e}")
        torch.cuda.empty_cache()
        return BENCH_CASE_OOM
    except Exception as e:
        print(f"  [BWD] S={S}: {e}")
        torch.cuda.empty_cache()
        return BENCH_CASE_OOM


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SparseLoad Block-Sparse Benchmark")
    parser.add_argument("--bwd", action="store_true", help="Also run BWD benchmark")
    parser.add_argument("--bwd-only", action="store_true", help="Only BWD")
    args = parser.parse_args()

    run_fwd = not args.bwd_only
    run_bwd = args.bwd or args.bwd_only

    print(
        f"Config: nhq={nhq}, nhk={nhk}, hd={hd}, qblk={q_block_size}, kblk={k_block_size}, "
        f"effective_kv={effective_kv} (n_attend={n_attend}), PackGQA=True, dtype={dtype}"
    )
    print(f"Seqlens: {seqlen_vals}")

    if run_fwd:
        print("\n" + "=" * 50)
        print("FWD Benchmark (SparseLoad block-sparse)")
        print("=" * 50)
        print(f"{'seqlen':>10} {'TFLOPS':>10}")
        print("-" * 22)
        for S in seqlen_vals:
            t = bench_fwd(S)
            print(f"{S:>10} {t:>10.2f}")
            torch.cuda.empty_cache()

    if run_bwd:
        print("\n" + "=" * 50)
        print("BWD Benchmark (SparseLoad block-sparse)")
        print("=" * 50)
        print(f"{'seqlen':>10} {'TFLOPS':>10}")
        print("-" * 22)
        for S in seqlen_vals:
            t = bench_bwd(S)
            print(f"{S:>10} {t:>10.2f}")
            torch.cuda.empty_cache()
