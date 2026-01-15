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

import os
from datetime import datetime

import torch
from baselines.attn_impl import ffa_func
from baselines.utils import block_sparse_available, seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.utils.sparse_utils import (
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
)

# actual seqlen
seqlens = [32768 * (i + 1) for i in range(0, 2)]

# current block sparse attention always has low sparsity
sparsity_ratio = [0.05, 0.1, 0.2, 0.5]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [16]
num_groups = [4]
# small K block
q_block_sizes = [128, 128, 128, 128, 128, 128]
k_block_sizes = [128, 64, 32, 16, 8, 1]

sparse_load_vals = [False, True]

assert len(q_block_sizes) == len(k_block_sizes)

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
        line_arg="sparse_load",
        line_vals=sparse_load_vals,
        line_names=["Sparse Load: False", "Sparse Load: True"],
        styles=[
            ("blue", "--"),
            ("red", "-"),
        ],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        plot_name=(
            f"FFA-Load-Compare attn_mode-{attn_mode} "
            f"{'n_head-' + str(nhq) if attn_mode == 'MHA' else f'n_head-{nhq}:{nhq // num_group}'}\n"
            f"block_size-{q_block_size}:{k_block_size} seq_len {seqlen}"
        ),
        args={
            "hd": hd,
            "wd": wd,
            "q_block_size": q_block_size,
            "k_block_size": k_block_size,
            "seqlen": seqlen,
            "num_group": num_group,
            "attn_mode": attn_mode,
            "nhq": nhq,
            "attn_impl": "ffa",
        },
    )
    for hd in ds
    for wd in wds
    for q_block_size, k_block_size in zip(q_block_sizes, k_block_sizes)
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
    q_block_size,
    k_block_size,
    seqlen,
    num_group,
    attn_mode,
    nhq,
    attn_impl,
    sparse_load,
):
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    orig_seq_len_q = orig_seq_len_k = seqlen
    block_m = q_block_size
    block_n = k_block_size

    num_q_blocks_orig = orig_seq_len_q // block_m
    num_kv_blocks_orig = orig_seq_len_k // block_n

    if attn_mode == "MHA":
        nhk = nhq
    elif attn_mode == "GQA":
        nhk = nhq // num_group

    block_mask, scores = generate_block_sparse_pattern(
        num_q_heads=nhq,
        num_kv_heads=nhk,
        num_q_blocks=num_q_blocks_orig,
        num_kv_blocks=num_kv_blocks_orig,
        sparsity=sparsity_ratio,
        device="cuda",
    )

    attn_flops = 4 * orig_seq_len_q * orig_seq_len_k * nhq * hd * sparsity_ratio

    # --------- prepare data --------- #
    q = torch.randn(
        b, orig_seq_len_q, nhq, hd, device=device, dtype=dtype, requires_grad=False
    )
    k = torch.randn(
        b, orig_seq_len_k, nhk, hd, device=device, dtype=dtype, requires_grad=False
    )
    v = torch.randn(
        b, orig_seq_len_k, nhk, hd, device=device, dtype=dtype, requires_grad=False
    )

    # ffa style shape: (t,h,d)
    if attn_impl in ("ffa"):
        # q = rearrange(q, "b s h d -> (b h s) 1 d")
        # repeats = nhq // nhk
        # k = torch.repeat_interleave(
        #   k, repeats=repeats, dim=2
        # )  # we need to flatten k, v along head dimension for GQA setting.
        # v = torch.repeat_interleave(v, repeats=repeats, dim=2)
        # k = rearrange(k, "b s h d -> (b h s) 1 d")
        # v = rearrange(v, "b s h d -> (b h s) 1 d")
        # q = q.view(b * orig_seq_len_q * nhq, 1, hd)
        # k = k.view(b * orig_seq_len_k * nhk, 1, hd)
        # v = v.view(b * orig_seq_len_k * nhk, 1, hd)
        h1 = nhk
        # h2 = nhq // nhk
        q = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=h1)
        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")

    # --------- prepare grads --------- #

    if wd == "bwd":
        attn_flops = attn_flops * 2.5
        do = torch.randn_like(q)
        [x.requires_grad_(True) for x in [q, k, v, do]]

    # --------- prepare func --------- #
    is_attn_impl_support_this_mask = block_sparse_available(
        attn_impl, nhq, nhk, q_block_size, k_block_size, wd
    )

    if is_attn_impl_support_this_mask:
        if attn_impl == "ffa":
            q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
                block_mask, block_m, block_n
            )
            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            # qhead_per_khead = nhq // nhk
            # ref_block_params = choose_ref_block(
            #     (q_block_size, k_block_size), qhead_per_khead=qhead_per_khead
            # )
            # TODO: find a better way to choose ref block size from specific arguments
            ref_block_size = (128, 128)

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    auto_range_merge=True,
                    sparse_load=sparse_load,
                    ref_block_size=ref_block_size,
                    disable_fwd_atomic_reduction=True,
                )

            if wd == "bwd":
                try:
                    o, *rest = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(f"Error: {e}")
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

    # --------- try do the bench --------- #
    if is_attn_impl_support_this_mask:
        if already_known_oom_before_run:
            perf_dict = {"flops": [-1, -1, -1], "mem": [-1, -1, -1]}
        else:
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
                        f"Error occured before running {attn_impl} with "
                        f"{q_block_size=}, {k_block_size=} "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                perf_dict = {"flops": [-1, -1, -1], "mem": [-1, -1, -1]}
                print(f"Error: {e}")
    else:
        perf_dict = {"flops": [-2, -2, -2], "mem": [-2, -2, -2]}

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_ffa_load_cmp_{current_time}")
    )

    sparse_attn_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
