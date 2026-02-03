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
from baselines.utils import calculate_attn_flops, generate_seqlens, seqlens2curanges

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

impls = ["ffa", "ffa_max_logits"]

mask_types = ["full", "causal", "varlen_full", "varlen_causal"]

# real-world varlen seqlen distribution
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

ss = [k * 1024 for k in [1, 2, 4, 8, 16]]
ds = [128]
wds = ["fwd"]

b = 1
nhq = 64
nhk = 8
dtype = torch.bfloat16

softmax_scale = None
quantiles = [0.5, 0.2, 0.8]


attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],
        x_vals=ss,
        x_log=False,
        line_arg="attn_impl",
        line_vals=impls,
        line_names=impls,
        styles=[
            ("green", "-"),
            ("red", "--"),
        ],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
        },
        plot_name=f"attn-{wd} with {mask_type} mask (max_logits comparison)",
        args={
            "hd": hd,
            "wd": wd,
            "mask_type": mask_type,
        },
    )
    for hd in ds
    for wd in wds
    for mask_type in mask_types
]


@perf_report(attn_flops_configs)
def attn_benchmark(seqlen, hd, wd, mask_type, attn_impl):
    assert b == 1
    device = torch.cuda.current_device()
    sq = sk = seqlen
    causal = "causal" in mask_type

    # --------- prepare arguments --------- #
    if "varlen" in mask_type:
        # same varlen distribution for different impl
        import numpy as np

        state = np.random.get_state()
        np.random.seed(seqlen)

        seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)

        # restore state to avoid affecting other parts
        np.random.set_state(state)

        cu_ranges = seqlens2curanges(seqlens)

        q_ranges_ = AttnRanges.from_ranges(cu_ranges)
        k_ranges_ = AttnRanges.from_ranges(cu_ranges)

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
    q = torch.randn(sq, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(sk, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(sk, nhk, hd, device=device, dtype=dtype)

    # --------- prepare func --------- #
    return_max_logits = attn_impl == "ffa_max_logits"

    def fn():
        return ffa_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            return_max_logits=return_max_logits,
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

        flops = perf_dict["flops"]
        if not isinstance(flops, list):
            flops = [flops]
        perf_dict["flops"] = list(map(ms_to_tflops, flops))
    except Exception as e:
        if "CUDA out of memory" not in str(e):
            print(f"Error: {e}")
            raise e
        perf_dict = {"flops": [-1, -1, -1]}

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_max_logits_{current_time}")
    )

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
