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
Benchmark: FFA IndexAttn vs baselines on a canonical sparse-attention scenario.

Scenario: S=600k-class seqlens, head_dim=128, MQA (nhq=128, nhk=1),
           q_block_size=k_block_size=1, full (non-causal) attention.

X-axis:  topk (number of selected KV tokens per query)
Lines:   FFA IndexAttn, FFA Dense, Triton Block-Sparse, FlexAttention (SDPA)

The benchmark measures FWD throughput in TFLOPs/s.
"""

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func, sdpa_func
from baselines.block_sparse_attn_triton import attention as triton_block_sparse_attn
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report


def build_index_attn_indices(b, S, nhk, topk, device):
    """Vectorized construction of index_attn_indices: (b*S, nhk, topk) int32."""
    total_q = b * S
    perm = (
        torch.rand(total_q, S, device=device)
        .argsort(dim=1)[:, :topk]
        .sort(dim=1)
        .values
    )
    batch_idx = torch.arange(total_q, device=device) // S
    global_pos = batch_idx.unsqueeze(1) * S + perm
    h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
    return (global_pos.unsqueeze(1) * nhk + h_offsets).int()


def build_block_sparse_mask(S, topk, block_size, device):
    """Build a block-sparse boolean mask (S, S) selecting ~topk tokens per row."""
    n_blocks = S // block_size
    blocks_needed = (topk + block_size - 1) // block_size
    blocks_needed = min(blocks_needed, n_blocks)
    perm = torch.rand(S, n_blocks, device=device).argsort(dim=1)[:, :blocks_needed]
    mask = torch.zeros(S, n_blocks, dtype=torch.bool, device=device)
    mask.scatter_(1, perm, True)
    return mask.repeat_interleave(block_size, dim=1)[:, :S]


# ─── Config ───────────────────────────────────────────────────────────────────

S = 32768  # Use 32k for practical benchmark runtime; scale up as needed
b = 1
nhq = 128
nhk = 1  # MQA
hd = 128
dtype = torch.bfloat16

topk_vals = [128, 256, 512, 1024, 2048, 4096]

METHODS = ["ffa_index_attn", "ffa_dense", "triton_block_sparse", "sdpa_sparse"]

quantiles = [0.5, 0.2, 0.8]

seed_everything()

attn_flops_configs = [
    Benchmark(
        x_names=["topk"],
        x_vals=topk_vals,
        x_log=True,
        line_arg="method",
        line_vals=METHODS,
        line_names=[
            "FFA IndexAttn",
            "FFA Dense (full S×S)",
            "Triton Block-Sparse",
            "SDPA (masked)",
        ],
        styles=[
            ("red", "-"),
            ("blue", "--"),
            ("green", "-."),
            ("orange", ":"),
        ],
        ylabel={"flops": "Throughput (TFLOPs/s)"},
        plot_name=(
            f"IndexAttn Comparison — MQA (nhq={nhq}, nhk={nhk})\n"
            f"S={S}, D={hd}, FWD only"
        ),
        args={},
    )
]


@perf_report(attn_flops_configs)
def comparison_benchmark(topk, method):
    device = torch.cuda.current_device()
    assert topk % 128 == 0

    q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)

    if method == "ffa_dense":
        attn_flops = 4 * S * S * nhq * hd
    else:
        attn_flops = 4 * S * topk * nhq * hd

    try:
        if method == "ffa_index_attn":
            index_attn_indices = build_index_attn_indices(b, S, nhk, topk, device)
            q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
            k_t = rearrange(k, "b s h d -> (b s h) 1 d")
            v_t = rearrange(v, "b s h d -> (b s h) 1 d")

            def fn():
                return ffa_func(
                    q_t,
                    k_t,
                    v_t,
                    index_attn_indices=index_attn_indices,
                    q_block_size=1,
                    k_block_size=1,
                    pack_gqa=True,
                )

        elif method == "ffa_dense":
            q_dense = q.view(b * S, nhq, hd)
            k_dense = k.view(b * S, nhk, hd)
            v_dense = v.view(b * S, nhk, hd)
            q_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, S]], dtype=torch.int32, device=device)
            attn_type_map = torch.zeros(1, dtype=torch.int32, device=device)

            def fn():
                return ffa_func(
                    q_dense,
                    k_dense,
                    v_dense,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                )

        elif method == "triton_block_sparse":
            block_size = 128
            sm_mask = build_block_sparse_mask(S, topk, block_size, device)
            # triton kernel expects (b, h, S, d) layout
            q_bh = q.permute(0, 2, 1, 3).contiguous()
            k_bh = k.permute(0, 2, 1, 3).contiguous()
            if nhk < nhq:
                k_bh = k_bh.repeat_interleave(nhq // nhk, dim=1)
            v_bh = v.permute(0, 2, 1, 3).contiguous()
            if nhk < nhq:
                v_bh = v_bh.repeat_interleave(nhq // nhk, dim=1)
            sm_scale = 1.0 / (hd**0.5)

            def fn():
                return triton_block_sparse_attn(
                    q_bh, k_bh, v_bh, sm_mask, sm_scale, block_size
                )

        elif method == "sdpa_sparse":
            q_sdpa = q.permute(0, 2, 1, 3).contiguous()
            k_sdpa = k.permute(0, 2, 1, 3).contiguous()
            if nhk < nhq:
                k_sdpa = k_sdpa.repeat_interleave(nhq // nhk, dim=1)
            v_sdpa = v.permute(0, 2, 1, 3).contiguous()
            if nhk < nhq:
                v_sdpa = v_sdpa.repeat_interleave(nhq // nhk, dim=1)
            sparse_mask = build_block_sparse_mask(S, topk, 128, device)
            attn_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(b, nhq, -1, -1)

            def fn():
                return sdpa_func(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask)

        else:
            raise ValueError(f"Unknown method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        print(f"Error running {method} topk={topk}: {e}")
        perf_dict = {"flops": [-1, -1, -1]}

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir,
        os.path.join("outs", f"bench_index_attn_comparison_{current_time}"),
    )

    comparison_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
