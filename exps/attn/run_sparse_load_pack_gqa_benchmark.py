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
    choose_ref_block,
    flatten_block_mask_to_kv_shape,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask,
    generate_ranges_from_block_mask_triton,
)

# This benchmark is used to compare the performance of sparse load with pack GQA enabled or not.
# Enable pack GQA together with sparse load with obtain performance gain in both small Q and K block size.

# actual seqlen
seqlens = [32768 * (i + 1) for i in range(0, 2)]

# current block sparse attention always has low sparsity
sparsity_ratio = [0.05, 0.1, 0.2, 0.5]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [16]
num_groups = [4]

# Block sizes for Pack GQA test
q_block_sizes = [32, 32]
k_block_sizes = [8, 1]

# Test pack gqa values
pack_gqa_vals = [False, True]
# Sparse load is always True for this benchmark
sparse_load_vals = [True]

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
        line_arg="pack_gqa",
        line_vals=pack_gqa_vals,
        line_names=["Pack GQA: False", "Pack GQA: True"],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
            "mem": "Peak Memory (GB)",
        },
        plot_name=(
            f"FFA-SparseLoad-PackGQA attn_mode-{attn_mode} "
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
            "sparse_load": True,
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
    pack_gqa,
):
    assert b == 1, "for now, we only supports b=1 for ffa"
    assert attn_mode == "GQA", "only support GQA for pack gqa benchmark"
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
        h1 = nhk
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
            # flatten headdim for ffa cause
            mask_creation_start = torch.cuda.Event(enable_timing=True)
            mask_creation_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            mask_creation_start.record()
            flat_block_sparse_mask = flatten_block_mask_to_kv_shape(block_mask)

            q_ranges, k_ranges = generate_ranges_from_block_mask(
                flat_block_sparse_mask, block_m, block_n
            )
            mask_creation_end.record()
            torch.cuda.synchronize()
            mask_creation_time = mask_creation_start.elapsed_time(mask_creation_end)
            print(
                f"Original Impl: FFA block sparse mask creation time: {mask_creation_time} ms"
            )

            torch.cuda.synchronize()
            mask_creation_start.record()
            q_ranges2, k_ranges2 = generate_ranges_from_block_mask_triton(
                block_mask, block_m, block_n
            )
            mask_creation_end.record()
            torch.cuda.synchronize()
            print(
                f"Triton Impl: FFA block sparse mask creation time: {mask_creation_start.elapsed_time(mask_creation_end)} ms"
            )

            torch.testing.assert_close(q_ranges, q_ranges2)
            torch.testing.assert_close(k_ranges, k_ranges2)
            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            ref_block_size = choose_ref_block(
                (q_block_size, k_block_size),
                sparse_load=sparse_load,
                pack_gqa=pack_gqa,
                qhead_per_khead=num_group,
            )

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    auto_range_merge=True,
                    ref_block_size=ref_block_size,
                    sparse_load=sparse_load,
                    pack_gqa=pack_gqa,
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
        script_dir,
        os.path.join("outs", f"bench_attn_ffa_sparse_load_pack_gqa_cmp_{current_time}"),
    )

    sparse_attn_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
