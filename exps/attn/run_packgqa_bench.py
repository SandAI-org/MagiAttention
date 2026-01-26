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
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.sparse_args import FFaSparseArgs
from magi_attention.utils.sparse_utils import (
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask_triton,
    generate_ranges_from_topk_indices_triton,
)

impls = ["ffa_packgqa_true", "ffa_packgqa_false"]

# actual seqlen
seqlens = [8192 * 4, 8192 * 8, 8192 * 16]

# current block sparse attention always has low sparsity
sparsity_ratio = [0.05, 0.1, 0.2, 0.5, 1.0]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [64]
num_groups = [4]
# small K block
# q_block_sizes = [64, 64, 64, 64, 64]
# k_block_sizes = [64, 32, 16, 8, 1]
# small Q block
# q_block_sizes = [64, 32, 16, 8]
# k_block_sizes = [64, 64, 64, 64]
q_block_sizes = [64, 16]
k_block_sizes = [64, 64]
# large Q block and K block
# q_block_sizes = [64, 128]
# k_block_sizes = [64, 128]

assert len(q_block_sizes) == len(k_block_sizes)

b = 1

dtype = torch.bfloat16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]

sparse_format = "topk"

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
        plot_name=(
            f"block sparse attn-{wd} attn_mode-{attn_mode} "
            f"{'n_head-' + str(nhq) if attn_mode == 'MHA' else f'n_head-{nhq}:{nhq // num_group}'}\n"
            f"block_size-{q_block_size}:{k_block_size} seq_len {seqlen}"
        ),
        # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "hd": hd,
            "wd": wd,
            "q_block_size": q_block_size,
            "k_block_size": k_block_size,
            "seqlen": seqlen,
            "num_group": num_group,
            "attn_mode": attn_mode,
            "nhq": nhq,
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
):
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    orig_seq_len_q = orig_seq_len_k = seqlen  # fi square mask where sq == sk
    block_m = q_block_size
    block_n = k_block_size

    num_q_blocks_orig = orig_seq_len_q // block_m
    num_kv_blocks_orig = orig_seq_len_k // block_n
    orig_head = nhq
    if attn_mode == "MHA":
        nhk = nhq
    elif attn_mode == "GQA":
        nhk = nhq // num_group

    # prepare q, k ranges and calculate attn_flops
    # for now, we only do bench for block sparse mask.
    # block_mask, scores = generate_global_block_sparse_pattern(
    #    orig_head, num_q_blocks_orig, num_kv_blocks_orig, sparsity_ratio, device="cuda"
    # )

    block_mask, scores = generate_block_sparse_pattern(
        num_q_heads=nhq,
        num_kv_heads=nhk,
        num_q_blocks=num_q_blocks_orig,
        num_kv_blocks=num_kv_blocks_orig,
        sparsity=sparsity_ratio,
        sparse_format=sparse_format,
        device="cuda",
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
    if attn_impl in ("ffa_packgqa_false", "ffa_packgqa_true"):
        h1 = nhk
        q = rearrange(q, "b s (h1 h2) d -> (b h1 s) h2 d", h1=h1)
        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")

    if attn_impl in ("sdpa", "vsa", "vsa_triton", "flashinfer", "flex"):
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
    # is_attn_impl_support_this_mask = block_sparse_available(
    #    attn_impl, nhq, nhk, q_block_size, k_block_size, wd
    # )
    is_attn_impl_support_this_mask = True
    if is_attn_impl_support_this_mask:
        if attn_impl == "ffa_packgqa_false":
            if sparse_format == "block_mask":
                q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
                    block_mask, block_m, block_n
                )
            elif sparse_format == "topk":
                q_ranges, k_ranges = generate_ranges_from_topk_indices_triton(
                    block_mask, block_m, block_n, num_kv_blocks_orig
                )

            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            # TODO: we need to optimize choose_ref_block.
            # You'd better set ref_blocks manually now
            # ref_block_size = choose_ref_block((q_block_size, k_block_size))
            ref_block_size = (64, 64)

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    disable_fwd_atomic_reduction=True,
                    sparse_args=FFaSparseArgs(
                        auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
                        ffa_tile_size=ref_block_size,
                        pack_gqa=False,
                    ),
                )

            if wd == "bwd":
                try:
                    o, *rest = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with "
                            f"{q_block_size=}, {k_block_size=} "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "ffa_packgqa_true":
            if sparse_format == "block_mask":
                q_ranges, k_ranges = generate_ranges_from_block_mask_triton(
                    block_mask, block_m, block_n
                )
            elif sparse_format == "topk":
                q_ranges, k_ranges = generate_ranges_from_topk_indices_triton(
                    block_mask, block_m, block_n, num_kv_blocks_orig
                )

            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            # ref_block_size = choose_ref_block((q_block_size, k_block_size))
            ref_block_size = (64, 64)

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    disable_fwd_atomic_reduction=True,
                    sparse_args=FFaSparseArgs(
                        auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
                        ffa_tile_size=ref_block_size,
                        pack_gqa=True,
                    ),
                )

            if wd == "bwd":
                try:
                    o, *rest = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with "
                            f"{q_block_size=}, {k_block_size=} "
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
                "mem": [-1, -1, -1],
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
                def gb(m):
                    return m / 1024**3

                # perf_dict["mem"] = list(map(gb, perf_dict["mem"]))
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    print(
                        f"Error occured before running {attn_impl} with "
                        f"{q_block_size=}, {k_block_size=} "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    perf_dict = {
                        "flops": [-2, -2, -2],
                        "mem": [-2, -2, -2],
                    }
                    # raise e
                # -1 indicates oom
                perf_dict = {
                    "flops": [-1, -1, -1],
                    "mem": [-1, -1, -1],
                }
                print(
                    f"Error occured before running {attn_impl} with {q_block_size=}, {k_block_size=} "
                    f"when {seqlen=}, {hd=} during {wd}: {e=}"
                )
    else:
        # -2 indicates not support
        perf_dict = {
            "flops": [-2, -2, -2],
            "mem": [-2, -2, -2],
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
