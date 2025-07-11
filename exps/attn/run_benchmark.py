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
from baselines.attn_impl import (
    cudnn_fused_attn_func,
    fa2_func,
    fa2_varlen_func,
    fa3_func,
    fa3_varlen_func,
    ffa_func,
    flex_attn_func,
    sdpa_func,
    torch_attn_func,
)
from baselines.utils import (
    calculate_attn_flops,
    curanges2document_id,
    generate_ranges_from_seqlens,
    generate_seqlens,
    make_causal_block_mask,
    make_causal_mask_score_mod,
    make_sliding_window_causal_block_mask,
    make_sliding_window_causal_mask_score_mod,
    make_varlen_block_causal_block_mask,
    make_varlen_block_causal_mask_score_mod,
    make_varlen_causal_block_mask,
    make_varlen_causal_mask_score_mod,
    make_varlen_full_block_mask,
    make_varlen_full_mask_score_mod,
    seqlens2cu_seqlens,
    seqlens2curanges,
)
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.utils._utils import get_attn_mask_from_ffa_args

# impls = ["sdpa", "fa2", "fa3", "ffa", "torch"]
# impls = ["sdpa", "fa2", "fa3", "ffa"]  # ignore torch native to avoid OOM
# impls = ["fa2", "fa3", "ffa"]  # compare to fa family
# impls = ["cudnn", "fa3", "ffa"]  # compare to performance top-3 sota
# impls = ["ffa", "fa3", "cudnn", "fa2", "flex", "sdpa"]  # all except torch native
impls = ["ffa"]


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


ss = [k * 1024 for k in [1, 2, 4, 8, 16]]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd", "bwd"]


b = 1
nhq = 48
nhk = 8
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
        plot_name=f"attn-{wd} with {mask_type} mask",  # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
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
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    sq = sk = seqlen  # fi square mask where sq == sk
    causal = "causal" in mask_type and "block_causal" not in mask_type
    sdpa_mask = None

    # calculate attn flops
    if mask_type == "sliding_window_causal":
        q_ranges_ = AttnRanges.from_ranges([[0, window_size]])
        k_ranges_ = AttnRanges.from_ranges([[0, window_size]])
        is_causal_mapping_ = [True]

        for start in range(window_size, seqlen):
            q_ranges_.append(AttnRange(start, start + 1))
            k_ranges_.append(AttnRange(start - window_size + 1, start + 1))
            is_causal_mapping_.append(False)

        window_size_tuple = (window_size, 0)
        max_seqlen_q = sq
        max_seqlen_k = sk
        max_seqlen_q = sq
        max_seqlen_kv = sk
        cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

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

            max_seqlen_q = q_ranges_.max_seqlen
            max_seqlen_k = k_ranges_.max_seqlen
            max_seqlen_kv = max_seqlen_k

            cu_seqlens = seqlens2cu_seqlens(seqlens)
            cu_ranges = seqlens2curanges(seqlens)
            document_id = curanges2document_id(cu_ranges)

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

            cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

            window_size_tuple = (-1, -1)
        else:
            seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
            cu_seqlens = seqlens2cu_seqlens(seqlens)
            cu_ranges = seqlens2curanges(seqlens)
            document_id = curanges2document_id(cu_ranges)

            q_ranges_ = AttnRanges.from_ranges(cu_ranges)
            k_ranges_ = AttnRanges.from_ranges(cu_ranges)
            max_seqlen_q = q_ranges_.max_seqlen
            max_seqlen_k = k_ranges_.max_seqlen
            max_seqlen_kv = max_seqlen_k
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

            cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            attn_type_map = (
                torch.ones(len(cu_ranges), dtype=torch.int32, device=device)
                if causal
                else torch.zeros(len(cu_ranges), dtype=torch.int32, device=device)
            )

            window_size_tuple = (-1, -1)
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
        max_seqlen_q = sq
        max_seqlen_k = sk
        max_seqlen_q = sq
        max_seqlen_kv = sk
        cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
        cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)
        attn_type_map = torch.tensor(
            [1 if causal else 0], dtype=torch.int32, device=device
        )

        window_size_tuple = (-1, -1)

    attn_flops = attn_flops_dict[wd]

    # --------- prepare data --------- #

    # flash style shape: (b,s,h,d)
    q = torch.randn(b, sq, nhq, hd, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(b, sk, nhk, hd, device=device, dtype=dtype, requires_grad=False)

    # sdpa style shape: (b,h,s,d)
    if attn_impl in ("sdpa", "torch", "flex"):
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        # make block mask
        if attn_impl == "flex":
            if mask_type == "full":
                score_mod = None
                block_mask = None
            elif mask_type == "causal":
                try:
                    block_mask = make_causal_block_mask(sq, sk)
                    score_mod = None
                except RuntimeError:
                    score_mod = make_causal_mask_score_mod()
                    block_mask = None
            elif mask_type == "sliding_window_causal":
                try:
                    block_mask = make_sliding_window_causal_block_mask(
                        sq, sk, window_size=window_size
                    )
                    score_mod = None
                except RuntimeError:
                    score_mod = make_sliding_window_causal_mask_score_mod(
                        window_size=window_size
                    )
                    block_mask = None
            elif "varlen" in mask_type:
                if causal:
                    try:
                        block_mask = make_varlen_causal_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_varlen_causal_mask_score_mod(document_id)
                        block_mask = None
                else:
                    if "block_causal" in mask_type:
                        try:
                            block_mask = make_varlen_block_causal_block_mask(
                                sq, sk, block_size, document_id
                            )
                            score_mod = None
                        except RuntimeError:
                            score_mod = make_varlen_block_causal_mask_score_mod(
                                block_size, document_id
                            )
                            block_mask = None
                    else:
                        try:
                            block_mask = make_varlen_full_block_mask(
                                sq, sk, document_id
                            )
                            score_mod = None
                        except RuntimeError:
                            score_mod = make_varlen_full_mask_score_mod(document_id)
                            block_mask = None
            else:
                raise NotImplementedError(
                    f"mask type {mask_type} not supported for flex attn"
                )
        elif "varlen" in mask_type or mask_type == "sliding_window_causal":
            try:
                # sdpa_mask = make_varlen_causal_sdpa_mask(sq, sk, cu_ranges)
                attn_type_mapping = [
                    1 if mapping else 0 for mapping in is_causal_mapping_
                ]
                sdpa_mask = get_attn_mask_from_ffa_args(
                    q_ranges=q_ranges_,
                    k_ranges=k_ranges_,
                    attn_type_map=attn_type_mapping,
                    total_seqlen_q=sq,
                    total_seqlen_k=sk,
                    device=torch.cuda.current_device(),
                )
            except RuntimeError as e:
                print(f"make varlen causal sdpa mask failed: {e}")

    # ffa style shape: (t,h,d)
    if attn_impl in ("ffa", "cudnn"):
        q = q.view(b * sq, nhq, hd)
        k = k.view(b * sk, nhk, hd)
        v = v.view(b * sk, nhk, hd)

        if attn_impl == "cudnn":
            if "varlen_block_causal" in mask_type:
                is_attn_impl_support_this_mask = False

    # fa style shape:
    #   non-varlen: (b,s,h,d)
    #   varlen: (t,h,d)
    if attn_impl in ("fa2", "fa3"):
        if "varlen" in mask_type:
            q = q.view(b * sq, nhq, hd)
            k = k.view(b * sk, nhk, hd)
            v = v.view(b * sk, nhk, hd)

        if "block_causal" in mask_type:
            is_attn_impl_support_this_mask = False

    # --------- prepare grads --------- #

    if wd == "bwd":
        do = torch.randn_like(q)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do]]

    # --------- prepare func --------- #

    if attn_impl == "torch":

        def fn():
            return torch_attn_func(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=causal if sdpa_mask is None else False,
                scale=softmax_scale,
                return_attn_probs=return_attn_probs,
            )

        if wd == "bwd":
            try:
                o = fn()
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

    elif attn_impl == "sdpa":

        def fn():
            return sdpa_func(
                q,
                k,
                v,
                attn_mask=sdpa_mask,
                is_causal=causal if sdpa_mask is None else False,
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
                        f"Error occured before running {attn_impl} with {mask_type} mask "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                already_known_oom_before_run = True

            def fn():
                o.backward(do, retain_graph=True)

    elif attn_impl == "fa2":
        if "varlen" in mask_type:

            def fn():
                return fa2_varlen_func(
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_attn_probs=return_attn_probs,
                )

        else:

            def fn():
                return fa2_func(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size_tuple,
                    return_attn_probs=return_attn_probs,
                )

        if wd == "bwd":
            try:
                o = fn()
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

    elif attn_impl == "fa3":
        if "varlen" in mask_type:

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

    elif attn_impl == "cudnn":

        def fn():
            return cudnn_fused_attn_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                softmax_scale=softmax_scale,
                is_causal=causal,
                dropout_p=dropout_p,
                window_size=window_size_tuple,
                is_training=wd == "bwd",
            )

        if wd == "bwd":
            try:
                o = fn()
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

    elif attn_impl == "ffa":

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

    elif attn_impl == "flex":

        def fn():
            return flex_attn_func(
                q,
                k,
                v,
                scale=softmax_scale,
                enable_gqa=True,
                score_mod=score_mod,
                block_mask=block_mask,
            )

        if wd == "bwd":
            try:
                o = fn()
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
                # perf_dict = do_bench(
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
