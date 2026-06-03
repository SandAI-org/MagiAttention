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
Benchmark: FFA IndexAttn vs baselines under canonical large-scale sparse scenarios.

MQA (nhq=128, nhk=1), head_dim=128, q_block_size=k_block_size=1, full (non-causal).
nhq=128 is chosen so that PackGQA is exercised.

Config 1 — S=102400 (~100k), sweep topk:
  FFA IndexAttn vs Triton Block-Sparse head-to-head.

Config 2 — S=32768 (32k), sweep topk:
  Same comparison at smaller scale (validates both methods).

FWD-only throughput in TFLOPs/s.
"""

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func
from baselines.block_sparse_attn_triton import block_sparse_fwd
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

# ─── Helpers ──────────────────────────────────────────────────────────────────


def build_index_attn_indices(b, S, nhk, topk, device):
    """Build index_attn_indices: (b*S, nhk, topk) int32.

    Uses torch.randint for O(total_q × topk) memory (safe at large S).
    Duplicates are benign for throughput benchmarking.
    """
    total_q = b * S
    local_pos = torch.randint(0, S, (total_q, topk), device=device).sort(dim=1).values
    batch_idx = torch.arange(total_q, device=device) // S
    global_pos = batch_idx.unsqueeze(1) * S + local_pos
    h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
    return (global_pos.unsqueeze(1) * nhk + h_offsets).int()


def build_triton_block_sparse_indices(S, topk, block_size=64, device="cuda"):
    """Build q2k_index and q2k_num for Triton block-sparse FWD kernel.

    Returns:
        q2k_index: (1, 1, num_q_blocks, kv_blocks_needed) int32
        q2k_num:   (1, 1, num_q_blocks) int32
    """
    num_q_blocks = S // block_size
    num_kv_blocks = S // block_size
    kv_blocks_needed = min((topk + block_size - 1) // block_size, num_kv_blocks)
    perm = (
        torch.rand(num_q_blocks, num_kv_blocks, device=device)
        .argsort(dim=1)[:, :kv_blocks_needed]
        .sort(dim=1)
        .values
    )
    q2k_index = perm.unsqueeze(0).unsqueeze(0).int()
    q2k_num = torch.full((1, 1, num_q_blocks), kv_blocks_needed, device=device).int()
    return q2k_index, q2k_num


# ─── Config ───────────────────────────────────────────────────────────────────

b = 1
nhq = 128
nhk = 1  # MQA
hd = 128
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]

S_100k = 102400  # ~100k, divisible by 128 and 64
S_32k = 32768

topk_vals = [128, 256, 512, 1024, 2048, 4096]

METHODS = ["ffa_index_attn", "triton_block_sparse"]
METHOD_NAMES = ["FFA IndexAttn", "Triton Block-Sparse"]
METHOD_STYLES = [("red", "-"), ("green", "-.")]

seed_everything()

attn_flops_configs = [
    # Config 1: S~100k — large-scale comparison
    Benchmark(
        x_names=["topk"],
        x_vals=topk_vals,
        x_log=True,
        line_arg="method",
        line_vals=METHODS,
        line_names=METHOD_NAMES,
        styles=METHOD_STYLES,
        ylabel={"flops": "Throughput (TFLOPs/s)"},
        plot_name=(
            f"IndexAttn vs Triton Block-Sparse — MQA (PackGQA)\n"
            f"S={S_100k}, nhq={nhq}, nhk={nhk}, D={hd}, FWD"
        ),
        args={"S": S_100k},
    ),
    # Config 2: S=32k — smaller-scale validation
    Benchmark(
        x_names=["topk"],
        x_vals=topk_vals,
        x_log=True,
        line_arg="method",
        line_vals=METHODS,
        line_names=METHOD_NAMES,
        styles=METHOD_STYLES,
        ylabel={"flops": "Throughput (TFLOPs/s)"},
        plot_name=(
            f"IndexAttn vs Triton Block-Sparse — MQA (PackGQA)\n"
            f"S={S_32k}, nhq={nhq}, nhk={nhk}, D={hd}, FWD"
        ),
        args={"S": S_32k},
    ),
]


@perf_report(attn_flops_configs)
def comparison_benchmark(topk, method, S):
    device = torch.cuda.current_device()
    assert topk % 128 == 0

    attn_flops = 4 * S * topk * nhq * hd

    try:
        if method == "ffa_index_attn":
            q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype)

            index_attn_indices = build_index_attn_indices(b, S, nhk, topk, device)
            q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
            k_t = rearrange(k, "b s h d -> (b s h) 1 d")
            v_t = rearrange(v, "b s h d -> (b s h) 1 d")
            del q, k, v
            torch.cuda.empty_cache()

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

        elif method == "triton_block_sparse":
            triton_block = 64
            assert S % triton_block == 0
            q = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            k = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            v = torch.randn(b, nhq, S, hd, device=device, dtype=dtype)
            q2k_index, q2k_num = build_triton_block_sparse_indices(
                S, topk, triton_block, device
            )
            q2k_index = q2k_index.expand(b, nhq, -1, -1).contiguous()
            q2k_num = q2k_num.expand(b, nhq, -1).contiguous()

            def fn():
                return block_sparse_fwd(q, k, v, q2k_index, q2k_num)

        else:
            raise ValueError(f"Unknown method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        print(f"[{method}] topk={topk} S={S}: {e}")
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
