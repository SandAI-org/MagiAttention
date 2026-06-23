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

"""BWD LoopQ vs LoopK benchmark for FFA dense attention.

Compares the two BWD loop directions in a dense (full/causal) scenario:
- LoopQ (default, swap_bwd_qk_loop=False): K outer-loop, Q inner-loop
    → dK/dV accumulated locally, dQ via atomic
- LoopK (swapped, swap_bwd_qk_loop=True): Q outer-loop, K inner-loop
    → dQ accumulated locally, dK/dV via atomic

Also includes FA3 and cuDNN as reference baselines.
"""

import torch
from baselines.attn_impl import cudnn_fused_attn_func, fa3_func
from baselines.utils import calculate_attn_flops

from magi_attention.benchmarking import (
    BENCH_CASE_OOM,
    Benchmark,
    do_bench_flops,
    gen_save_path,
    perf_report,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange  # noqa: F401
from magi_attention.common.ranges import AttnRanges
from magi_attention.functional import flex_flash_attn_func as ffa_func

b = 1
nhq = 128
nhk = 1  # MQA — same as index_sparse benchmark
hd = 128
dtype = torch.bfloat16
quantiles = [0.5, 0.2, 0.8]

seqlen_vals = [4096, 8192, 16384, 32768, 65536]

METHODS = [
    "ffa_loopq_bwd",
    "ffa_loopk_bwd",
    "fa3_bwd",
    "cudnn_bwd",
]
METHOD_NAMES = [
    "FFA LoopQ BWD (dKV local, dQ atomic)",
    "FFA LoopK BWD (dQ local, dKV atomic)",
    "FA3 BWD",
    "cuDNN BWD",
]
METHOD_STYLES = [
    ("blue", "-"),
    ("red", "-"),
    ("green", "--"),
    ("orange", "--"),
]

MASK_TYPE = "causal"

attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],
        x_vals=seqlen_vals,
        x_log=True,
        line_arg="method",
        line_vals=METHODS,
        line_names=METHOD_NAMES,
        styles=METHOD_STYLES,
        ylabel={"flops": "TFLOPs/s (BWD)"},
        plot_name=(
            f"Dense Attention BWD: LoopQ vs LoopK\n"
            f"nhq={nhq}, nhk={nhk}, D={hd}, mask={MASK_TYPE}"
        ),
        args={},
    ),
]


@perf_report(attn_flops_configs)
def bwd_loop_benchmark(seqlen, method):
    device = torch.cuda.current_device()
    sq = sk = seqlen
    causal = MASK_TYPE == "causal"

    attn_flops_dict = calculate_attn_flops(
        q_ranges=AttnRanges.from_ranges([[0, sq]]),
        k_ranges=AttnRanges.from_ranges([[0, sk]]),
        attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL],
        total_seqlen_q=sq,
        num_heads_q=nhq,
        head_dim=hd,
    )
    attn_flops = attn_flops_dict["bwd"]

    try:
        if method in ("ffa_loopq_bwd", "ffa_loopk_bwd"):
            swap = method == "ffa_loopk_bwd"
            q = torch.randn(b * sq, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            q_ranges = torch.tensor([[0, sq]], dtype=torch.int32, device=device)
            k_ranges = torch.tensor([[0, sk]], dtype=torch.int32, device=device)
            attn_type_map = torch.tensor(
                [1 if causal else 0], dtype=torch.int32, device=device
            )

            o, *_ = ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                swap_bwd_qk_loop=swap,
            )
            do = torch.randn_like(o)

            def fn():
                o.backward(do, retain_graph=True)

        elif method == "fa3_bwd":
            q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            o = fa3_func(q, k, v, causal=causal)
            do = torch.randn_like(o)

            def fn():
                o.backward(do, retain_graph=True)

        elif method == "cudnn_bwd":
            q = torch.randn(b * sq, nhq, hd, device=device, dtype=dtype)
            k = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype)
            v = torch.randn(b * sk, nhk, hd, device=device, dtype=dtype)
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            o = cudnn_fused_attn_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=sq,
                max_seqlen_kv=sk,
                is_causal=causal,
                is_training=True,
            )
            do = torch.randn_like(o)

            def fn():
                o.backward(do, retain_graph=True)

        else:
            raise ValueError(f"Unknown method: {method}")

        perf_dict = do_bench_flops(fn, quantiles=quantiles)

        def ms_to_tflops(ms: float) -> float:
            return attn_flops / ms * 1e-9

        perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

    except torch.cuda.OutOfMemoryError as e:
        print(f"[{method}] seqlen={seqlen}: OOM — {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM, BENCH_CASE_OOM, BENCH_CASE_OOM]}
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[{method}] seqlen={seqlen}: {e}")
        perf_dict = {"flops": [BENCH_CASE_OOM, BENCH_CASE_OOM, BENCH_CASE_OOM]}
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return perf_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BWD LoopQ vs LoopK Benchmark")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run",
    )
    parser.add_argument(
        "--mask",
        choices=["full", "causal"],
        default="causal",
        help="Mask type (default: causal)",
    )
    args = parser.parse_args()

    if args.mask:
        MASK_TYPE = args.mask

    if args.methods:
        idx_map = {m: i for i, m in enumerate(METHODS)}
        selected = [m for m in args.methods if m in idx_map]
        selected_idx = [idx_map[m] for m in selected]
        METHODS = [METHODS[i] for i in selected_idx]
        METHOD_NAMES = [METHOD_NAMES[i] for i in selected_idx]
        METHOD_STYLES = [METHOD_STYLES[i] for i in selected_idx]
        attn_flops_configs[0] = Benchmark(
            x_names=["seqlen"],
            x_vals=seqlen_vals,
            x_log=True,
            line_arg="method",
            line_vals=METHODS,
            line_names=METHOD_NAMES,
            styles=METHOD_STYLES,
            ylabel={"flops": "TFLOPs/s (BWD)"},
            plot_name=attn_flops_configs[0].plot_name,
            args={},
        )

    out_root = gen_save_path("bench_bwd_loopq_loopk")

    print(f"Configuration: nhq={nhq}, nhk={nhk}, D={hd}, mask={MASK_TYPE}")
    print(f"seqlen_vals={seqlen_vals}")
    print(f"Methods: {METHOD_NAMES}")
    print()

    bwd_loop_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        # only 1 benchmark here; bump to torch.cuda.device_count() if more are added
        num_workers=1,
    )
