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

  * mask_type : full | causal | varlen_full | varlen_causal
  * direction : fwd | bwd
  * seqlen    : 1k .. 32k
  * head_dim  : 128

Compared baselines: ffa (current) vs fa3 (reference).
If the machine is Blackwell (SM100) fa4 / ffa_fa4 are added automatically.

Run:
    cd exps/attn/cutedsl
    PYTHONPATH=../../.. python run_benchmark_simple.py

    # force the forked SM80 ffa kernel path (compared against fa2) even on sm90/sm100:
    BENCH_FORCE_SM80=1 PYTHONPATH=../../.. python run_benchmark_simple.py
"""

import os

import torch

from magi_attention.benchmarking import (
    BENCH_CASE_NOT_SUPPORTED,
    BENCH_CASE_OOM,
    Benchmark,
    do_bench_flops,
    gen_save_path,
    perf_report,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges
from magi_attention.kernel.cutedsl import MT_MAP
from magi_attention.kernel.cutedsl import flex_flash_attn_func as ffa_func
from magi_attention.utils.arch import (
    get_dev_cap_str,
    is_ampere,
    is_blackwell,
    is_hopper,
)

# isort: split
from exps.attn.baselines.utils import (
    calculate_attn_flops,
    generate_seqlens,
    seqlens2cu_seqlens,
    seqlens2curanges,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Force the forked SM80 ffa kernel path even on newer GPUs (sm90/sm100).
# This only changes kernel *selection* via MAGI_ATTENTION_FFA_CUTEDSL_ARCH; compilation still
# targets the real device so the cubin actually runs. The compared baseline is fa2.
FORCE_SM80 = os.environ.get("BENCH_FORCE_SM80", "0") == "1"
if FORCE_SM80:
    # must be set before the first ffa kernel call so get_device_arch() picks it up
    os.environ["MAGI_ATTENTION_FFA_CUTEDSL_ARCH"] = "sm_80"

IS_SM80 = is_ampere() or FORCE_SM80
IS_SM90 = is_hopper()
IS_SM100 = is_blackwell()
arch = "sm80" if FORCE_SM80 else get_dev_cap_str()

# fa3 only runs on SM90; fa4/ffa_fa4 only on SM100+; fa2 is the SM80 baseline
if IS_SM80:
    from exps.attn.baselines.attn_impl import fa2_func, fa2_varlen_func  # noqa: F401

    impls = ["ffa", "fa2"]
elif IS_SM90:
    from exps.attn.baselines.attn_impl import fa3_func, fa3_varlen_func  # noqa: F401

    impls = ["ffa", "fa3"]
elif IS_SM100:
    from exps.attn.baselines.attn_impl import (  # noqa: F401
        fa4_func,
        fa4_varlen_func,
        ffa_fa4_func,
    )

    impls = ["ffa", "fa4", "ffa_fa4"]
else:
    impls = ["ffa"]

# Scenarios: full / causal (non-varlen) and varlen_full / varlen_causal (packed)
mask_types = ["full", "causal", "varlen_full", "varlen_causal"]

# Simple varlen seqlen distribution (doc length intervals -> weight). Each
# benchmarked seqlen is split into a list of doc lengths following this.
# The first interval must start at 0 so any remaining length can be filled.
varlen_seqlen_distribution = {
    (0, 1024): 0.4,
    (1024, 2048): 0.3,
    (2048, 4096): 0.2,
    (4096, 8192): 0.1,
}

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
    causal = "causal" in mask_type
    # ffa takes a mask_types int (single mask type shared by all ranges) while
    # the other baselines keep using the legacy `causal` bool.
    ffa_mask_types = MT_MAP.causal if causal else MT_MAP.full
    is_varlen = "varlen" in mask_type
    window_size_tuple = (-1, -1)

    # ── ranges / cu_seqlens / attn flops ──
    if is_varlen:
        # split the total seqlen into a list of variable-length docs
        seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
        cu_ranges = seqlens2curanges(seqlens)
        cu_seqlens = seqlens2cu_seqlens(seqlens)

        q_ranges_ = AttnRanges.from_ranges(cu_ranges)
        k_ranges_ = AttnRanges.from_ranges(cu_ranges)
        max_seqlen_q = q_ranges_.max_seqlen
        max_seqlen_k = k_ranges_.max_seqlen

        attn_flops_dict = calculate_attn_flops(
            q_ranges=q_ranges_,
            k_ranges=k_ranges_,
            attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL]
            * len(cu_ranges),
            total_seqlen_q=sq,
            num_heads_q=nhq,
            head_dim=hd,
        )

        q_ranges_t = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
        k_ranges_t = torch.tensor(cu_ranges, dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        attn_type_map = torch.full(
            (len(cu_ranges),), 1 if causal else 0, dtype=torch.int32, device=device
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

        q_ranges_t = torch.tensor([[0, sq]], dtype=torch.int32, device=device)
        k_ranges_t = torch.tensor([[0, sk]], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, sk], dtype=torch.int32, device=device)
        max_seqlen_q = sq
        max_seqlen_k = sk
        attn_type_map = torch.tensor(
            [1 if causal else 0], dtype=torch.int32, device=device
        )

    attn_flops = attn_flops_dict[wd]

    # ── tensors (flash style: b,s,h,d) ──
    q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype)
    k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)
    v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype)

    # ffa_fa4 always uses (t,h,d); ffa / fa2 / fa3 / fa4 use packed (t,h,d) only for varlen
    if attn_impl == "ffa_fa4" or (
        is_varlen and attn_impl in ("ffa", "fa2", "fa3", "fa4")
    ):
        q = q.view(b * sq, nhq, hd)
        k = k.view(b * sk, nhk, hd)
        v = v.view(b * sk, nhk, hd)

    # ── define fn ──
    if attn_impl == "ffa":
        if is_varlen:

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges_t,
                    k_ranges=k_ranges_t,
                    mask_types=ffa_mask_types,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )

        else:

            def fn():
                return ffa_func(q, k, v, mask_types=ffa_mask_types)

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

    elif attn_impl == "fa2":
        if is_varlen:

            def fn():
                return fa2_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size_tuple,
                )

        else:

            def fn():
                return fa2_func(
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

    elif attn_impl == "fa3":
        if is_varlen:

            def fn():
                return fa3_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size_tuple,
                )

        else:

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
        fa4_window = tuple(None if x == -1 else x for x in window_size_tuple)
        if is_varlen:

            def fn():
                return fa4_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=fa4_window,
                )[0]

        else:

            def fn():
                return fa4_func(
                    q,
                    k,
                    v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=fa4_window,
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
        return {"flops": [BENCH_CASE_NOT_SUPPORTED] * 3}

    if already_oom:
        return {"flops": [BENCH_CASE_OOM] * 3}

    try:
        perf_dict = do_bench_flops(fn, quantiles=quantiles)

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
        return {"flops": [BENCH_CASE_OOM] * 3}

    return perf_dict


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_root = gen_save_path(f"bench_simple_{arch}", add_timestamp_suffix=False)

    attn_benchmark.run(
        print_data=True,
        print_value_on_bar=False,
        save_path=out_root,
        num_workers=torch.cuda.device_count(),
    )
