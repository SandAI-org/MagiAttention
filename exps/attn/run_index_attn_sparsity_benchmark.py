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
Benchmark: index_attn_indices direct path at various sparsity ratios.

Compares index_attn=True (index_attn_indices path, topk = S * sparsity_ratio)
against index_attn=False (dense attention via q/k ranges).

X-axis: sparsity_ratio
Lines:  IndexAttn=False, IndexAttn=True
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

# actual seqlen
seqlens = [32768]

# topk values (must be multiples of tile_size=128)
topk_vals = [128, 512, 1024, 2048, 4096]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [128]
num_groups = [128]  # nhq // num_group = nhk; 128//128 = 1 → MQA

index_attn_vals = [False, True]

b = 1

dtype = torch.bfloat16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]


attn_flops_configs = [
    Benchmark(
        x_names=["topk"],
        x_vals=topk_vals,
        x_log=False,
        line_arg="index_attn",
        line_vals=index_attn_vals,
        line_names=["IndexAttn: False (dense)", "IndexAttn: True"],
        styles=[
            ("blue", "--"),
            ("red", "-"),
        ],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
        },
        plot_name=(
            f"FFA-IndexAttn-Compare attn_mode-{attn_mode} "
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


def build_index_attn_indices(b, S, nhk, topk, device):
    """Vectorized construction of index_attn_indices: (b*S, nhk, topk) int32.

    Each query selects `topk` sorted random KV positions, encoded as global row IDs.
    """
    total_q = b * S
    # Batched randperm via argsort of random values: (total_q, topk) sorted local indices
    perm = (
        torch.rand(total_q, S, device=device)
        .argsort(dim=1)[:, :topk]
        .sort(dim=1)
        .values
    )
    # Global token position per query: batch_offset + local_position
    batch_idx = torch.arange(total_q, device=device) // S
    global_pos = batch_idx.unsqueeze(1) * S + perm  # (total_q, topk)
    # Expand across nhk heads: global_pos * nhk + h
    h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
    return (global_pos.unsqueeze(1) * nhk + h_offsets).int()  # (total_q, nhk, topk)


seed_everything()


@perf_report(attn_flops_configs)
def sparse_attn_benchmark(
    topk,
    hd,
    wd,
    seqlen,
    num_group,
    attn_mode,
    nhq,
    index_attn,
):
    assert b == 1, "for now, we only supports b=1 for ffa"

    device = torch.cuda.current_device()
    S = seqlen

    if attn_mode == "MHA":
        nhk = nhq
    elif attn_mode == "GQA":
        nhk = nhq // num_group
    else:
        raise ValueError(f"Unknown attn_mode: {attn_mode}")

    assert topk % 128 == 0, f"topk={topk} must be a multiple of 128"
    assert topk <= S, f"topk={topk} > S={S}"
    attn_flops = 4 * S * topk * nhq * hd if index_attn else 4 * S * S * nhq * hd

    # --------- prepare data --------- #
    q = torch.randn(b, S, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, S, nhk, hd, device=device, dtype=dtype, requires_grad=False)

    if index_attn:
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

    else:
        # dense path: full attention via q/k ranges
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
                auto_range_merge=True,
            )

    # --------- try do the bench --------- #
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
                f"Error running {attn_mode} index_attn={index_attn} "
                f"when {seqlen=}, {hd=} during {wd}: {e=}"
            )
        perf_dict = {"flops": [BENCH_CASE_OOM, BENCH_CASE_OOM, BENCH_CASE_OOM]}
        print(f"Error: {e}")

    return perf_dict


if __name__ == "__main__":
    out_root = gen_save_path("bench_attn_ffa_index_attn_cmp")

    sparse_attn_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        # only 1 benchmark here; bump to torch.cuda.device_count() if more are added
        num_workers=1,
    )
