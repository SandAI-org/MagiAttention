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
Benchmark: pack_gqa gain for sparse_kv_indices direct path.

Compares pack_gqa=True vs pack_gqa=False when using the sparse_kv_indices
direct-to-kernel path. GQA mode only.

X-axis: sparsity_ratio
Lines:  Pack GQA: False, Pack GQA: True
"""

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

# This benchmark is used to compare the performance of sparse KV with pack GQA
# enabled or not. Enable pack GQA together with sparse KV to obtain performance
# gain in both small Q and K block size.

# actual seqlen
seqlens = [32768 * (i + 1) for i in range(0, 2)]

# current block sparse attention always has low sparsity
sparsity_ratio = [0.05, 0.1, 0.2, 0.5]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [16]
num_groups = [4]

# Test pack gqa values
pack_gqa_vals = [False, True]

b = 1

dtype = torch.bfloat16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]


attn_flops_configs = [
    Benchmark(
        x_names=["sparsity_ratio"],
        x_vals=sparsity_ratio,
        x_log=False,
        line_arg="pack_gqa",
        line_vals=pack_gqa_vals,
        line_names=["Pack GQA: False", "Pack GQA: True"],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
        },
        plot_name=(
            f"FFA-SparseKV-PackGQA attn_mode-{attn_mode} "
            f"{'n_head-' + str(nhq) if attn_mode == 'MHA' else f'n_head-{nhq}:{nhq // num_group}'}\n"
            f"seq_len {seqlen}"
        ),
        args={
            "hd": hd,
            "wd": wd,
            "seqlen": seqlen,
            "num_group": num_group,
            "attn_mode": attn_mode,
            "nhq": nhq,
        },
    )
    for hd in ds
    for wd in wds
    for seqlen in seqlens
    for num_group in num_groups
    for attn_mode in attn_modes
    for nhq in nhqs
]

seed_everything()


@perf_report(attn_flops_configs)
def sparse_attn_benchmark(
    sparsity_ratio,
    hd,
    wd,
    seqlen,
    num_group,
    attn_mode,
    nhq,
    pack_gqa,
):
    assert b == 1, "for now, we only supports b=1 for ffa"
    assert attn_mode == "GQA", "only support GQA for pack gqa benchmark"

    device = torch.cuda.current_device()
    S = seqlen

    if attn_mode == "MHA":
        nhk = nhq
    elif attn_mode == "GQA":
        nhk = nhq // num_group
    else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    topk = max(1, int(S * sparsity_ratio))
    attn_flops = 4 * S * topk * nhq * hd

    # --------- prepare data --------- #
    q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)

    # sparse_kv_indices direct path: (total_q, nhk, topk) with global row ids
    total_q = b * S
    sparse_kv_indices = torch.empty(
        (total_q, nhk, topk), dtype=torch.int32, device=device
    )
    for bi in range(b):
        for qi in range(S):
            row = bi * S + qi
            perm = torch.randperm(S, device=device)[:topk].sort().values
            for h in range(nhk):
                sparse_kv_indices[row, h, :] = ((bi * S + perm) * nhk + h).int()

    q_t = rearrange(q, "b s (h1 h2) d -> (b s h1) h2 d", h1=nhk)
    k_t = rearrange(k, "b s h d -> (b s h) 1 d")
    v_t = rearrange(v, "b s h d -> (b s h) 1 d")

    def fn():
        return ffa_func(
            q_t,
            k_t,
            v_t,
            sparse_kv_indices=sparse_kv_indices,
            q_block_size=1,
            k_block_size=1,
            pack_gqa=pack_gqa,
        )

    # --------- try do the bench --------- #
    try:
        perf_dict = do_bench_flops(
            fn,
            quantiles=quantiles,
            mem_record_mode="peak",
        )

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        if "CUDA out of memory" not in str(e):
            print(
                f"Error running {attn_mode} pack_gqa={pack_gqa} "
                f"when {seqlen=}, {hd=} during {wd}: {e=}"
            )
        perf_dict = {"flops": [-1, -1, -1]}
        print(f"Error: {e}")

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir,
        os.path.join("outs", f"bench_attn_ffa_sparse_kv_pack_gqa_cmp_{current_time}"),
    )

    sparse_attn_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
