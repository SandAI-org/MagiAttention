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
Benchmark: index_attn_indices direct path vs dense attention on dense masks.

Measures the overhead of the index_attn_indices path (topk = full seqlen)
compared to standard dense attention across different mask types and seqlens.

X-axis: seqlen
Lines:  index_attn=False (dense), index_attn=True (index_attn_indices with topk=S)
"""

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
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges


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


index_attn_options = [False, True]

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


ss = [512, 1024, 4096, 16384, 65536]
ds = [128]
wds = ["fwd"]


b = 1
nhq = 128
nhk = 1  # MQA
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
        x_names=["seqlen"],
        x_vals=ss,
        x_log=False,
        line_arg="index_attn",
        line_vals=index_attn_options,
        line_names=[f"IndexAttn={str(opt)}" for opt in index_attn_options],
        styles=[
            ("green", "--"),
            ("red", "-"),
        ],
        ylabel={
            "flops": "Throughout (TFLOPs/s)",
        },
        plot_name=f"ffa-sparse-kv-{wd}-{mask_type}",
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
def attn_benchmark(seqlen, hd, wd, mask_type, index_attn):
    assert b == 1, "for now, we only supports b=1 for ffa"

    device = torch.cuda.current_device()
    sq = sk = seqlen
    causal = "causal" in mask_type and "block_causal" not in mask_type

    # --------- calculate attn flops --------- #
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

    q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)

    # --------- prepare func --------- #

    if index_attn:
        # index_attn_indices path: topk = full seqlen (measures overhead only)
        topk = sq  # all seqlens are multiples of 1024, already aligned to 128
        total_q = b * sq
        # Dense index: every query attends to all KV tokens (sequential, not random)
        batch_idx = torch.arange(total_q, device=device) // sq
        local_pos = torch.arange(topk, device=device).unsqueeze(0).expand(total_q, -1)
        global_pos = batch_idx.unsqueeze(1) * sq + local_pos
        h_offsets = torch.arange(nhk, device=device).view(1, -1, 1)
        index_attn_indices = (global_pos.unsqueeze(1) * nhk + h_offsets).int()

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
        # dense path: standard q_ranges / k_ranges
        q_dense = q.view(b * sq, nhq, hd)
        k_dense = k.view(b * sk, nhk, hd)
        v_dense = v.view(b * sk, nhk, hd)

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

    if wd == "bwd":
        try:
            o, *rest = fn()
        except Exception as e:
            if "CUDA out of memory" not in str(e):
                print(
                    f"Error before running ffa (index_attn={index_attn}) with {mask_type} mask "
                    f"when {seqlen=}, {hd=} during {wd}: {e=}"
                )
                raise e
            return {"flops": [-1, -1, -1]}

        do = torch.randn_like(o)

        def fn_bwd():
            o.backward(do, retain_graph=True)

        bench_fn = fn_bwd
    else:
        bench_fn = fn

    # --------- try do the bench --------- #

    try:
        perf_dict = do_bench_flops(
            bench_fn,
            quantiles=quantiles,
            mem_record_mode="peak",
        )

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except Exception as e:
        if "CUDA out of memory" not in str(e):
            print(
                f"Error when running ffa (index_attn={index_attn}) with {mask_type} mask "
                f"when {seqlen=}, {hd=} during {wd}: {e=}"
            )
            raise e
        perf_dict = {"flops": [-1, -1, -1]}
        print(
            f"OOM when running ffa (index_attn={index_attn}) with {mask_type} mask "
            f"when {seqlen=}, {hd=} during {wd}: {e=}"
        )

    return perf_dict


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_ffa_index_attn_{current_time}")
    )

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
