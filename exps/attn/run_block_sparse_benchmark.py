# Copyright (c) 2025 SandAI. All Rights Reserved.
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
from baselines.attn_impl import ffa_func, sdpa_func
from baselines.utils import (
    flatten_head_mask,
    generate_global_block_sparse_pattern,
    generate_gqa_ranges_from_3d_mask,
    generate_ranges_from_mask,
    get_sdpa_mask_from_block_sparse_mask,
)
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report

impls = ["ffa"]

# actual seqlen
seqlen = 49152

sparsity_ratio = [0.1, 0.2, 0.5, 0.8, 1.0]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd", "bwd"]
block_sizes = [64, 128]

b = 1
nhq = 48
nhk = 8
dtype = torch.bfloat16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]


attn_flops_configs = [
    Benchmark(
        x_names=["sparsity_ratio"],  # Argument names to use as an x-axis for the plot.
        x_vals=sparsity_ratio,  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="attn_impl",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=impls,  # Possible values for `line_arg`.
        line_names=impls,  # Label name for the lines.
        styles=[  # Line styles.
            ("green", "--"),
            ("orange", "--"),
            ("steelblue", "--"),
            ("red", "-"),
        ],
        ylabel={  # Label name for the y-axis.
            "flops": "Throughout (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        plot_name=f"block sparse attn-{wd} block_size-{block_size}",
        # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "hd": hd,
            "wd": wd,
            "block_size": block_size,
        },
    )
    for hd in ds
    for wd in wds
    for block_size in block_sizes
]


@perf_report(attn_flops_configs)
def sparse_attn_benchmark(sparsity_ratio, hd, wd, block_size, attn_impl):
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    orig_seq_len_q = orig_seq_len_k = seqlen  # fi square mask where sq == sk
    block_m = block_n = block_size

    num_q_blocks_orig = orig_seq_len_q // block_m
    num_kv_blocks_orig = orig_seq_len_k // block_n
    orig_head = nhq

    max_seqlen_q = block_m
    max_seqlen_k = block_n

    # prepare q, k ranges and caculate attn_flops
    # for now, we only do bench for block sparse mask.
    block_mask = generate_global_block_sparse_pattern(
        orig_head, num_q_blocks_orig, num_kv_blocks_orig, sparsity_ratio, device="cuda"
    )
    attn_flops = 4 * orig_seq_len_q * orig_seq_len_k * orig_head * hd * sparsity_ratio

    # --------- prepare data --------- #
    # flash style shape: (b,s,h,d)
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
        q = q.view(b * orig_seq_len_q * nhq, 1, hd)
        k = k.view(b * orig_seq_len_k * nhk, 1, hd)
        v = v.view(b * orig_seq_len_k * nhk, 1, hd)

    if attn_impl in ("sdpa"):
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

    # --------- prepare grads --------- #

    if wd == "bwd":
        attn_flops = attn_flops * 2.5
        do = torch.randn_like(q)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do]]

    # --------- prepare func --------- #
    if attn_impl == "ffa":
        # flatten headdim for ffa cause
        flat_global_sparse_mask = flatten_head_mask(block_mask)
        # 3. Generate ranges from the flattened 2D mask
        q_ranges, k_ranges = generate_ranges_from_mask(
            flat_global_sparse_mask, block_m, block_n
        )
        attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

        q_ranges, k_ranges = generate_gqa_ranges_from_3d_mask(
            mask_3d=block_mask,
            block_m=block_m,
            block_n=block_n,
            num_q_heads=orig_head,  # nheads
            num_k_heads=orig_head,  # nheads (MHA)
            seq_len=orig_seq_len_q,
        )

        def fn():
            return ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
            )

        if wd == "bwd":
            try:
                o, *rest = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    print(
                        f"Error occured before running {attn_impl} with {block_size} block_size "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                already_known_oom_before_run = True

            def fn():
                o.backward(do, retain_graph=True)

    elif attn_impl == "sdpa":
        sdpa_mask = get_sdpa_mask_from_block_sparse_mask(
            block_mask,
            seq_len_q=orig_seq_len_q,
            seq_len_k=orig_seq_len_q,
            block_size_q=block_m,
            block_size_k=block_n,
        )

        def fn():
            return sdpa_func(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                is_causal=False,
                scale=softmax_scale,
                dropout_p=dropout_p,
                enable_gqa=True,
            )

        if wd == "bwd":
            try:
                o = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    print(
                        f"Error occured before running {attn_impl} with {block_size} mask "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                already_known_oom_before_run = True

            def fn():
                o.backward(do, retain_graph=True)

    # --------- try do the bench --------- #
    if is_attn_impl_support_this_mask:
        if already_known_oom_before_run:
            # -1 indicates oom
            perf_dict = {
                "flops": [-1, -1, -1],
                # "mem": [-1, -1, -1],
            }
        else:
            try:
                # disable mem test to only test flops for now
                perf_dict = do_bench_flops(
                    fn,
                    quantiles=quantiles,
                    mem_record_mode="peak",
                )

                # --------- process report --------- #

                # post process the perf_dict
                def ms_to_tflops(ms: float) -> float:
                    return attn_flops / ms * 1e-9

                perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

                # disable mem test
                # def gb(m):
                #     return m / 1024**3

                # perf_dict["mem"] = list(map(gb, perf_dict["mem"]))
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    print(
                        f"Error occured when running {attn_impl} with {block_size} block_size "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                # -1 indicates oom
                perf_dict = {
                    "flops": [-1, -1, -1],
                    # "mem": [-1, -1, -1],
                }
                print(
                    f"OOM error occured when running for {attn_impl} with {block_size} block_size "
                    f"when {seqlen=}, {hd=} during {wd}: {e=}"
                )
    else:
        # -2 indicates not support
        perf_dict = {
            "flops": [-2, -2, -2],
            # "mem": [-2, -2, -2],
        }

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_{current_time}")
    )

    sparse_attn_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
