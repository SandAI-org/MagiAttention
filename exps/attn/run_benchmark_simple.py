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

"""Simple regression benchmark for the forked cutedsl kernel.

Covers only the most fundamental training scenarios so that after each round
of changes we can quickly verify no performance regression has occurred:

  * mask_type : full | causal
  * direction : fwd | bwd
  * seqlen    : 1k .. 32k
  * head_dim  : 128

Compared baselines: ffa (current) vs fa3 (reference).
If the machine is Blackwell (SM100) fa4 / ffa_fa4 are added automatically.

Run:
    cd exps/attn
    python run_benchmark_simple.py
"""

import os
from datetime import datetime

import torch
from baselines.utils import calculate_attn_flops

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.kernel.cutedsl.legacy.interface import flash_attn_func as ffa_func

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

compute_capability = torch.cuda.get_device_capability()[0]
IS_SM90 = compute_capability == 9
IS_SM100 = compute_capability >= 10
arch = f"sm{compute_capability}0"

# fa3 only runs on SM90; fa4/ffa_fa4 only on SM100+
if IS_SM90:
    from baselines.attn_impl import fa3_func  # noqa: F401

    impls = ["ffa", "fa3"]
elif IS_SM100:
    from baselines.attn_impl import fa4_func, ffa_fa4_func  # noqa: F401

    impls = ["ffa", "fa4", "ffa_fa4"]
else:
    impls = ["ffa"]

# Scenarios: simple non-varlen full / causal
mask_types = ["full", "causal"]

ss = [k * 1024 for k in [1, 2, 4, 8, 16, 32]]
ds = [128]
wds = ["fwd", "bwd"]

b = 1
nhq = 48
nhks = [48, 8]  # 48: MHA, 8: GQA
dtype = torch.bfloat16
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False
quantiles = [0.5, 0.2, 0.8]

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configs
# ─────────────────────────────────────────────────────────────────────────────

# Build one style-entry per impl so the list always matches impls length
_style_cycle = [
    ("green", "--"),
    ("steelblue", "--"),
    ("orange", "-"),
    ("red", "-"),
    ("purple", "-"),
    ("brown", "-"),
]

attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],
        x_vals=ss,
        x_log=False,
        line_arg="attn_impl",
        line_vals=impls,
        line_names=impls,
        styles=_style_cycle[: len(impls)],
        ylabel={
            "flops": "Throughput (TFLOPs/s)",
        },
        plot_name=f"simple-attn-{wd}-{mask_type}-hd{hd}-{'mha' if nhk == nhq else 'gqa'}",
        args={"hd": hd, "wd": wd, "mask_type": mask_type, "nhk": nhk},
    )
    for hd in ds
    for wd in wds
    for mask_type in mask_types
    for nhk in nhks
]


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark function
# ─────────────────────────────────────────────────────────────────────────────


@perf_report(attn_flops_configs)
def attn_benchmark(seqlen, hd, wd, mask_type, nhk, attn_impl):
    is_supported = True
    already_oom = False

    device = torch.cuda.current_device()
    sq = sk = seqlen
    causal = mask_type == "causal"

    # ── attn flops ──
    attn_flops_dict = calculate_attn_flops(
        q_ranges=AttnRanges.from_ranges([[0, sq]]),
        k_ranges=AttnRanges.from_ranges([[0, sk]]),
        attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL],
        total_seqlen_q=sq,
        num_heads_q=nhq,
        head_dim=hd,
    )
    attn_flops = attn_flops_dict[wd]

    # ── common tensor args ──
    q_ranges_t = torch.tensor([[0, sq]], dtype=torch.int32, device=device)
    k_ranges_t = torch.tensor([[0, sk]], dtype=torch.int32, device=device)
    attn_type_map = torch.tensor([1 if causal else 0], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, sk], dtype=torch.int32, device=device)
    window_size_tuple = (-1, -1)

    # ── tensors (flash style: b,s,h,d) ──
    q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)

    # ffa_fa4 uses (t,h,d); ffa (legacy flash_attn_func) keeps (b,s,h,d)
    if attn_impl == "ffa_fa4":
        q = q.view(b * sq, nhq, hd)
        k = k.view(b * sk, nhk, hd)
        v = v.view(b * sk, nhk, hd)

    # ── define fn ──
    if attn_impl == "ffa":

        def fn():
            return ffa_func(q, k, v, causal=causal)

        if wd == "bwd":
            try:
                o, *_ = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    raise
                already_oom = True
            if not already_oom:
                do = torch.randn_like(o)
                [x.requires_grad_(True) for x in [q, k, v]]
                o, *_ = fn()

                def fn():
                    o.backward(do, retain_graph=True)

    elif attn_impl == "ffa_fa4":
        # warmup to build cached FA4AttnArg
        _ = ffa_fa4_func(
            q,
            k,
            v,
            q_ranges=q_ranges_t,
            k_ranges=k_ranges_t,
            attn_type_map=attn_type_map,
            reuse_attn_arg=False,
        )
        torch.cuda.synchronize()

        def fn():
            return ffa_fa4_func(
                q,
                k,
                v,
                q_ranges=q_ranges_t,
                k_ranges=k_ranges_t,
                attn_type_map=attn_type_map,
                reuse_attn_arg=True,
            )

        if wd == "bwd":
            try:
                o, *_ = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    raise
                already_oom = True
            if not already_oom:
                do = torch.randn_like(o)
                [x.requires_grad_(True) for x in [q, k, v]]
                o, *_ = fn()

                def fn():
                    o.backward(do, retain_graph=True)

    elif attn_impl == "fa3":

        def fn():
            return fa3_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size_tuple,
            )

        if wd == "bwd":
            try:
                o = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    raise
                already_oom = True
            if not already_oom:
                do = torch.randn_like(o)
                [x.requires_grad_(True) for x in [q, k, v]]
                o = fn()

                def fn():
                    o.backward(do, retain_graph=True)

    elif attn_impl == "fa4":

        def fn():
            return fa4_func(
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=tuple(None if x == -1 else x for x in window_size_tuple),
            )[0]

        if wd == "bwd":
            try:
                o = fn()
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    raise
                already_oom = True
            if not already_oom:
                do = torch.randn_like(o)
                [x.requires_grad_(True) for x in [q, k, v]]
                o = fn()

                def fn():
                    o.backward(do, retain_graph=True)

    else:
        is_supported = False

    # ── bench ──
    if not is_supported:
        return {"flops": [-2, -2, -2]}

    if already_oom:
        return {"flops": [-1, -1, -1]}

    try:
        perf_dict = do_bench_flops(fn, quantiles=quantiles, mem_record_mode="peak")

        def ms_to_tflops(ms):
            return attn_flops / ms * 1e-9

        flops = perf_dict["flops"]
        if not isinstance(flops, list):
            flops = [flops]
        perf_dict["flops"] = list(map(ms_to_tflops, flops))
    except Exception as e:
        if "CUDA out of memory" not in str(e):
            print(f"Error: {attn_impl} {mask_type} {seqlen=} {hd=} {wd}: {e}")
            raise
        return {"flops": [-1, -1, -1]}

    return perf_dict


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(script_dir, "outs", f"bench_simple_{arch}")

    print(f"Running simple benchmark and saving results to {out_root} ...")

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
