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
from baselines.attn_impl import ffa_func, flex_attn_func, sdpa_func
from baselines.utils import (
    block_sparse_available,
    get_flashinfer_uniform_block_index,
    get_flex_mask_from_block_mask,
    get_vsa_mask_from_block_sparse_score,
    seed_everything,
)
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.utils.sparse_utils import (
    choose_ref_block,
    flatten_block_mask_to_kv_shape,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask,
    generate_ranges_from_block_mask_triton,
    generate_ranges_from_topk_indices,
    generate_ranges_from_topk_indices_triton,
    get_sdpa_mask_from_block_sparse_mask,
)

impls = ["ffa"]

# actual seqlen
seqlens = [32768 * (i + 1) for i in range(0, 2)]
seqlens = [8192, 16384, 32768, 65536]

# current block sparse attention always has low sparsity
sparsity_ratio = [0.05, 0.1, 0.2, 0.5]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [64]
num_groups = [4]
# small K block
q_block_sizes = [128, 128, 128, 128, 128, 128]
k_block_sizes = [128, 64, 32, 16, 8, 1]
# small Q block
# q_block_sizes = [64, 32, 16, 8]
# k_block_sizes = [64, 64, 64, 64]
# large Q block and K block
# q_block_sizes = [64, 128]
# k_block_sizes = [64, 128]

sparse_load = True
sparse_format = "topk"

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
    print(f"=====Running with {q_block_size=}, {k_block_size=} {seqlen=} ")
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
    # generate block mask totally random.
    """
    block_mask  = (
            torch.rand(1, nhk, num_q_blocks_orig, num_kv_blocks_orig, device='cuda') < sparsity_ratio
        )

    repeats = nhq // nhk
    block_mask = torch.repeat_interleave(block_mask, repeats=repeats, dim=1)
    """

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
        # q = rearrange(q, "b s h d -> (b h s) 1 d")
        # repeats = nhq // nhk
        # k = torch.repeat_interleave(
        #   k, repeats=repeats, dim=2
        # )  # we need to flatten k, v along head dimension for GQA setting.
        # v = torch.repeat_interleave(v, repeats=repeats, dim=2)
        # k = rearrange(k, "b s h d -> (b h s) 1 d")
        # v = rearrange(v, "b s h d -> (b h s) 1 d")
        # q = q.view(b * orig_seq_len_q * nhq, 1, hd)
        # k = k.view(b * orig_seq_len_k * nhk, 1, hd)
        # v = v.view(b * orig_seq_len_k * nhk, 1, hd)
        h1 = nhk
        # h2 = nhq // nhk
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
            # flat_block_sparse_mask = flatten_block_mask(block_mask, nhq, nhk)

            if sparse_format == "block_mask":
                flat_block_sparse_mask = flatten_block_mask_to_kv_shape(block_mask)
                q_ranges, k_ranges = generate_ranges_from_block_mask(
                    flat_block_sparse_mask, block_m, block_n
                )
            elif sparse_format == "topk":
                q_ranges, k_ranges = generate_ranges_from_topk_indices(
                    block_mask, block_m, block_n, orig_seq_len_k // block_n
                )
            mask_creation_end.record()
            torch.cuda.synchronize()
            mask_creation_time = mask_creation_start.elapsed_time(mask_creation_end)
            print(
                f"Original Impl for =={sparse_format}== sparse format: FFA mask creation time: {mask_creation_time} ms"
            )

            torch.cuda.synchronize()
            mask_creation_start.record()

            if sparse_format == "block_mask":
                q_ranges2, k_ranges2 = generate_ranges_from_block_mask_triton(
                    block_mask, block_m, block_n
                )
            elif sparse_format == "topk":
                q_ranges2, k_ranges2 = generate_ranges_from_topk_indices_triton(
                    block_mask, block_m, block_n, orig_seq_len_k // block_n
                )
            mask_creation_end.record()
            torch.cuda.synchronize()
            print(
                f"Triton Impl for =={sparse_format}== sparse format: \
                FFA mask creation time: {mask_creation_start.elapsed_time(mask_creation_end)} ms"
            )

            torch.testing.assert_close(q_ranges, q_ranges2)
            torch.testing.assert_close(k_ranges, k_ranges2)

            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            # TODO: SwapAB will change this constraint
            ref_block_size = choose_ref_block(
                (q_block_size, k_block_size), sparse_load=sparse_load
            )

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
                    ref_block_size=ref_block_size,
                    sparse_load=sparse_load,
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

        elif attn_impl == "vsa":
            try:
                from vsa import block_sparse_bwd, block_sparse_fwd
            except ImportError:
                raise ImportError(
                    "Please install FastVideo VSA following https://github.com/hao-ai-lab/FastVideo/tree/main/csrc/attn."
                )

            topk = int(sparsity_ratio * num_kv_blocks_orig)
            (
                q2k_block_sparse_index,
                q2k_block_sparse_num,
                k2q_block_sparse_index,
                k2q_block_sparse_num,
            ) = get_vsa_mask_from_block_sparse_score(
                scores,
                k=topk,
            )
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            def fn():
                return block_sparse_fwd(
                    q, k, v, q2k_block_sparse_index, q2k_block_sparse_num
                )

            if wd == "bwd":
                do = do.contiguous()
                try:
                    o, l_vec = fn()
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
                    block_sparse_bwd(
                        q,
                        k,
                        v,
                        o,
                        l_vec,
                        do,
                        k2q_block_sparse_index,
                        k2q_block_sparse_num,
                    )

        elif attn_impl == "vsa_triton":
            from baselines.block_sparse_attn_triton import (
                block_sparse_bwd,
                block_sparse_fwd,
            )

            topk = int(sparsity_ratio * num_kv_blocks_orig)
            (
                q2k_block_sparse_index,
                q2k_block_sparse_num,
                k2q_block_sparse_index,
                k2q_block_sparse_num,
            ) = get_vsa_mask_from_block_sparse_score(
                scores,
                k=topk,
            )
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

            def fn():
                return block_sparse_fwd(
                    q, k, v, q2k_block_sparse_index, q2k_block_sparse_num
                )

            if wd == "bwd":
                try:
                    o, l_vec = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with "
                            f"{q_block_size=}, {k_block_size=} "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True
                do = do.contiguous()

                def fn():
                    block_sparse_bwd(
                        q,
                        k,
                        v,
                        o,
                        l_vec,
                        do,
                        q2k_block_sparse_index,
                        q2k_block_sparse_num,
                        k2q_block_sparse_index,
                        k2q_block_sparse_num,
                    )

        elif attn_impl == "flashinfer":
            try:
                import flashinfer
            except ImportError:
                raise ImportError(
                    "Please install FlashInfer v0.2.9 with this bug-fixed PR: \n"
                    "https://github.com/flashinfer-ai/flashinfer/pull/1383"
                )

            q = q.view(b * nhq, orig_seq_len_q, hd).contiguous()
            k = k.view(b * nhk, orig_seq_len_k, hd).contiguous()
            v = v.view(b * nhk, orig_seq_len_k, hd).contiguous()
            # BUG: using original block mask will cause illegal access sometimes
            # block_mask_cpu = block_mask.squeeze(0).detach().cpu()
            block_mask_cpu = (
                torch.rand(nhk, num_q_blocks_orig, num_kv_blocks_orig) < sparsity_ratio
            )

            block_row_sz, block_col_sz = get_flashinfer_uniform_block_index(
                num_q_blocks_orig,
                num_kv_blocks_orig,
                orig_seq_len_q,
                orig_seq_len_k,
                nhk,
            )

            # allocate 128MB workspace buffer
            kv_lens_buffer_size = nhk * block_row_sz.shape[-1] + 1024
            workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=block_mask.device
            )
            wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
                workspace_buffer, backend="fa3", kv_lens_buffer_size=kv_lens_buffer_size
            )
            try:
                wrapper.plan(
                    block_mask_map=block_mask_cpu,
                    block_row_sz=block_row_sz,
                    block_col_sz=block_col_sz,
                    num_qo_heads=nhq,
                    num_kv_heads=nhk,
                    head_dim=hd,
                    q_data_type=q.dtype,
                )
            except Exception as e:
                print(
                    f"Error occured before running {attn_impl} with {q_block_size=}, {k_block_size=} "
                    f"when {seqlen=}, {hd=} during {wd}: {e=}"
                )
                is_attn_impl_support_this_mask = False

            def fn():
                return wrapper.run(q, k, v)

        elif attn_impl == "fa2_sparse":
            try:
                from block_sparse_attn import block_sparse_attn_func
            except ImportError:
                raise ImportError(
                    "Please install FA2 sparse attention following \
                    https://github.com/mit-han-lab/Block-Sparse-Attention/blob/main/README.md."
                )

            cu_seqlens = torch.arange(
                0, (b + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device
            )
            q = q.reshape(b * orig_seq_len_q, nhq, hd).contiguous()
            k = k.reshape(b * orig_seq_len_k, nhk, hd).contiguous()
            v = v.reshape(b * orig_seq_len_k, nhk, hd).contiguous()
            head_mask_type = torch.tensor([1] * nhq, device=q.device, dtype=torch.int32)
            streaming_info = None

            def fn():
                return block_sparse_attn_func(
                    q,
                    k,
                    v,
                    cu_seqlens,
                    cu_seqlens,
                    head_mask_type,
                    streaming_info,
                    block_mask,
                    orig_seq_len_q,
                    orig_seq_len_k,
                    p_dropout=dropout_p,
                    softmax_scale=softmax_scale,
                    sparse_block_size=q_block_size,
                )

        elif attn_impl == "flex":
            try:
                flex_mask = get_flex_mask_from_block_mask(
                    block_mask, orig_seq_len_q, orig_seq_len_k, nhq, nhk, bsz=b
                )
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
                return flex_attn_func(
                    q,
                    k,
                    v,
                    block_mask=flex_mask,
                    scale=softmax_scale,
                    enable_gqa=True,
                )

            if wd == "bwd":
                if not already_known_oom_before_run:
                    try:
                        o = fn()
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

                # Store original latency values before converting to FLOPS
                original_latency_ms = perf_dict["flops"].copy()
                perf_dict["flops"] = list(map(ms_to_tflops, perf_dict["flops"]))

                # Print latency information
                print(f"Latency (ms) - {original_latency_ms[0]:.2f}")

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
                else:
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
