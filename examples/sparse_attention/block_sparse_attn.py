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
Simple example demonstrating block_sparse_attn usage with TFLOPS measurement.

This script shows the minimal usage of block_sparse_attn interface and measures
the performance in TFLOPS for forward pass only.
"""

import time

import torch

from magi_attention.functional import block_sparse_attn
from magi_attention.utils.sparse_utils import generate_block_sparse_pattern


def generate_data(
    seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim, dtype, device
):
    """Generate random data for block sparse attention."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    q = torch.randn(seqlen_q, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(seqlen_k, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(seqlen_k, num_kv_heads, head_dim, dtype=dtype, device=device)

    return q, k, v


def generate_topk_indices(
    seqlen_q,
    seqlen_k,
    num_q_heads,
    num_kv_heads,
    q_block_size,
    k_block_size,
    sparsity_ratio,
    device,
):
    """Generate topk indices for block sparse attention."""
    num_q_blocks = seqlen_q // q_block_size
    num_kv_blocks = seqlen_k // k_block_size

    topk_indices, _ = generate_block_sparse_pattern(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        sparsity=sparsity_ratio,
        mode="per_kv_head",
        sparse_format="topk",
        device=device,
    )
    # block_mask is topk_indices with shape [num_kv_heads, num_q_blocks, topk]
    return topk_indices


def test_block_sparse_attn_speed(
    seqlen_q,
    seqlen_k,
    num_q_heads,
    num_kv_heads,
    head_dim,
    q_block_size,
    k_block_size,
    sparsity_ratio,
    dtype=torch.bfloat16,
):
    """Speed test for block_sparse_attn."""
    device = torch.cuda.current_device()

    # Generate data
    q, k, v = generate_data(
        seqlen_q, seqlen_k, num_q_heads, num_kv_heads, head_dim, dtype, device
    )
    topk_indices = generate_topk_indices(
        seqlen_q,
        seqlen_k,
        num_q_heads,
        num_kv_heads,
        q_block_size,
        k_block_size,
        sparsity_ratio,
        device,
    )

    warmup_iters = 10
    perf_test_iters = 100

    # Warmup
    for _ in range(warmup_iters):
        _ = block_sparse_attn(
            q=q,
            k=k,
            v=v,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            topk_indices=topk_indices,
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(perf_test_iters):
        out, lse = block_sparse_attn(
            q=q,
            k=k,
            v=v,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            topk_indices=topk_indices,
        )
    torch.cuda.synchronize()
    elapsed_time_ms = (time.perf_counter() - start) / perf_test_iters * 1000

    # Calculate TFLOPS
    attn_flops = 4 * seqlen_q * seqlen_k * num_q_heads * head_dim * sparsity_ratio
    tflops = attn_flops / elapsed_time_ms * 1e-9

    print(
        f"\nseqlen_q:{seqlen_q} seqlen_k:{seqlen_k} "
        f"num_q_heads:{num_q_heads} num_kv_heads:{num_kv_heads} "
        f"head_dim:{head_dim} q_block_size:{q_block_size} k_block_size:{k_block_size} "
        f"sparsity_ratio:{sparsity_ratio}"
    )
    print(f"Time: {elapsed_time_ms:.2f}ms, TFLOPS: {tflops:.2f}")


if __name__ == "__main__":
    seqlen_q = 48 * 1024
    seqlen_k = 48 * 1024

    # TODO: more format print.
    print("Test block sparse attention with block (64, 64) and MHA setting.")
    test_block_sparse_attn_speed(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_q_heads=64,
        num_kv_heads=64,
        head_dim=128,
        q_block_size=64,
        k_block_size=64,
        sparsity_ratio=0.1,
    )

    print("Test block sparse attention with block (1, 128) and GQA setting.")
    test_block_sparse_attn_speed(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        q_block_size=1,
        k_block_size=128,
        sparsity_ratio=0.1,
    )

    print("Test block sparse attention with block (128, 1) and GQA setting.")
    test_block_sparse_attn_speed(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_q_heads=64,
        num_kv_heads=8,
        head_dim=128,
        q_block_size=128,
        k_block_size=1,
        sparsity_ratio=0.1,
    )
