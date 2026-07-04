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
Benchmark: Head Group Size impact on IndexAttn performance.

Sweeps NHQ/NHK ratio with PackGQA=True. When ratio <= 16, SwapAB is
automatically enabled.

X-axis: ratio (NHQ / NHK)
Fixed:  S=32768, topk=1024, NHQ=128, D=128, PackGQA=True
"""

import torch
from baselines.attn_impl import ffa_func
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import (
    BENCH_CASE_OOM,
    Benchmark,
    do_bench_flops,
    gen_save_path,
    perf_report,
)


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


S = 32768
topk = 1024
nhq = 128
hd = 128
b = 1

ratios = [128, 64, 32, 16, 8]

dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]

attn_flops_configs = [
    Benchmark(
        x_names=["ratio"],
        x_vals=ratios,
        x_log=False,
        line_arg="line_label",
        line_vals=["IndexAttn"],
        line_names=["IndexAttn (PackGQA=True)"],
        styles=[("red", "-")],
        ylabel={"flops": "Throughout (TFLOPs/s)"},
        plot_name=(
            f"FFA-IndexAttn Head Group Size Sweep\n"
            f"S={S} topk={topk} NHQ={nhq} D={hd}"
        ),
        args={},
    )
]

seed_everything()


@perf_report(attn_flops_configs)
def head_group_benchmark(ratio, line_label):
    assert b == 1, "for now, we only supports b=1 for ffa"

    device = torch.cuda.current_device()
    nhk = nhq // ratio
    swap_ab = ratio <= 16
    attn_flops = 4 * S * topk * nhq * hd

    q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)

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
            swap_ab=swap_ab,
        )

    try:
        perf_dict = do_bench_flops(
            fn,
            quantiles=quantiles,
        )

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        if "CUDA out of memory" not in str(e):
            print(
                f"Error running ratio={ratio} swap_ab={swap_ab} "
                f"when S={S}, hd={hd}: {e=}"
            )
        perf_dict = {"flops": [BENCH_CASE_OOM, BENCH_CASE_OOM, BENCH_CASE_OOM]}
        print(f"Error: {e}")

    return perf_dict


if __name__ == "__main__":
    out_root = gen_save_path("bench_attn_ffa_index_attn_head_group")

    head_group_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        # only 1 benchmark here; bump to torch.cuda.device_count() if more are added
        num_workers=1,
    )
