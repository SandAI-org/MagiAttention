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
from baselines.utils import (
    calculate_attn_flops,
    generate_ranges_from_seqlens,
    generate_seqlens,
    seqlens2curanges,
)

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges

impls = ["ffa", "ffa_swap-qk-loop"]

mask_types = ["full"]
# mask_types = ["causal"]
# mask_types = ["varlen_full"]
# mask_types = ["varlen_causal"]
# mask_types = ["sliding_window_causal"]
# mask_types = ["varlen_block_causal"]


varlen_seqlen_distribution = {
    (0, 2 * 1024): 0.16,
    (2 * 1024, 4 * 1024): 0.05,
    (4 * 1024, 8 * 1024): 0.04,
    (8 * 1024, 16 * 1024): 0.06,
    (16 * 1024, 32 * 1024): 0.08,
    (32 * 1024, 64 * 1024): 0.21,
    (64 * 1024, 128 * 1024): 0.4,
    (128 * 1024, 256 * 1024): 0.2,
    (256 * 1024, 512 * 1024): 0.05,
    (512 * 1024, 1024 * 1024): 0.04,
    (1024 * 1024, 2048 * 1024): 0.01,
    (2048 * 1024, 4096 * 1024): 0.01,
}


ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32]]
ds = [64, 128]
wds = ["bwd"]


b = 1
nhq = 8
nhks = [8, 2]
dtype = torch.bfloat16

window_size = 1024
block_size = 2048
num_varlen_samples = 16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]


attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],  # Argument names to use as an x-axis for the plot.
        x_vals=ss,  # Different possible values for `x_name`.
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
        # Name for the plot. Used also as a file name for saving the plot.
        plot_name=f"attn-{wd}-{nhq=}{nhk=}-{hd=} with {mask_type} mask",
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "hd": hd,
            "wd": wd,
            "mask_type": mask_type,
            "nhk": nhk,
        },
    )
    for hd in ds
    for wd in wds
    for mask_type in mask_types
    for nhk in nhks
]


@perf_report(attn_flops_configs)
def attn_benchmark(seqlen, hd, wd, mask_type, nhk, attn_impl):
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    sq = sk = seqlen  # fi square mask where sq == sk
    causal = "causal" in mask_type and "block_causal" not in mask_type

    # calculate attn flops
    if mask_type == "sliding_window_causal":
        q_ranges_ = AttnRanges.from_ranges([[0, window_size]])
        k_ranges_ = AttnRanges.from_ranges([[0, window_size]])
        is_causal_mapping_ = [True]

        for start in range(window_size, seqlen):
            q_ranges_.append(AttnRange(start, start + 1))
            k_ranges_.append(AttnRange(start - window_size + 1, start + 1))
            is_causal_mapping_.append(False)

        attn_flops_dict = calculate_attn_flops(
            q_ranges=q_ranges_,
            k_ranges=k_ranges_,
            attn_mask_type=[
                AttnMaskType.CAUSAL if c else AttnMaskType.FULL
                for c in is_causal_mapping_
            ],
            total_seqlen_q=sq,
            num_heads_q=nhq,
            head_dim=hd,
        )

        q_ranges = torch.tensor(
            [[0, window_size], [window_size, seqlen]], dtype=torch.int32, device=device
        )
        k_ranges = torch.tensor(
            [[0, window_size], [0, seqlen]], dtype=torch.int32, device=device
        )
        attn_type_map = torch.tensor([1, 3], dtype=torch.int32, device=device)
    elif "varlen" in mask_type:
        if "block_causal" in mask_type:
            assert not causal

            seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
            q_ranges_, k_ranges_ = generate_ranges_from_seqlens(seqlens, block_size)
            is_causal_mapping_ = [False] * len(q_ranges_)
            cu_ranges = seqlens2curanges(seqlens)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=[AttnMaskType.FULL] * len(q_ranges_),
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            q_ranges = torch.tensor(
                q_ranges_.to_naive_ranges(), dtype=torch.int32, device=device
            )
            k_ranges = torch.tensor(
                k_ranges_.to_naive_ranges(), dtype=torch.int32, device=device
            )
            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device=device)
        else:
            seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
            cu_ranges = seqlens2curanges(seqlens)

            q_ranges_ = AttnRanges.from_ranges(cu_ranges)
            k_ranges_ = AttnRanges.from_ranges(cu_ranges)
            is_causal_mapping_ = [causal] * len(cu_ranges)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL]
                * len(cu_ranges),
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            q_ranges = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
            k_ranges = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
            attn_type_map = (
                torch.ones(len(cu_ranges), dtype=torch.int32, device=device)
                if causal
                else torch.zeros(len(cu_ranges), dtype=torch.int32, device=device)
            )
    else:
        attn_flops_dict = calculate_attn_flops(
            q_ranges=AttnRanges.from_ranges([[0, sq]]),
            k_ranges=AttnRanges.from_ranges([[0, sk]]),
            attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL],
            total_seqlen_q=sq,
            num_heads_q=nhq,
            head_dim=hd,
        )

        q_ranges = torch.tensor([[0, sq]], dtype=torch.int32, device=device)
        k_ranges = torch.tensor([[0, sk]], dtype=torch.int32, device=device)
        attn_type_map = torch.tensor(
            [1 if causal else 0], dtype=torch.int32, device=device
        )

    attn_flops = attn_flops_dict[wd]

    # --------- prepare data --------- #

    # ffa style shape: (t,h,d)
    q = torch.randn(b * sq, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)

    # --------- prepare grads --------- #

    if wd == "bwd":
        do = torch.randn_like(q)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do]]

    # --------- prepare func --------- #

    def fn():
        return ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            swap_bwd_qk_loop="swap-qk-loop" in attn_impl,
        )

    if wd == "bwd":
        try:
            o, *rest = fn()
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                print(
                    f"Error occured before running {attn_impl} with {mask_type} mask "
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
                        f"Error occured when running {attn_impl} with {mask_type} mask "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                # -1 indicates oom
                perf_dict = {
                    "flops": [-1, -1, -1],
                    # "mem": [-1, -1, -1],
                }
                print(
                    f"OOM error occured when running for {attn_impl} with {mask_type} mask "
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

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
