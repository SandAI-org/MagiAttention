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
from baselines.attn_impl import ffa_func, flex_attn_func, sdpa_func
from baselines.utils import (
    get_flex_mask_from_block_mask,
    seed_everything,
    var_block_sparse_available,
)
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.utils.sparse_utils import (
    flatten_block_mask,
    generate_ranges_from_var_block_mask,
    generate_variable_block_sparse_pattern,
    get_sdpa_mask_from_var_block_mask,
)

impls = ["ffa", "flashinfer", "flex"]

# actual seqlen
seqlens = [49152, 16384]
# sparsity_ratio = 1.0 will cause illegal access sometimes
sparsity_ratio = [0.1, 0.2, 0.5, 0.8, 0.9]
# sparsity_ratio = [1.0]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd", "bwd"]
block_sizes = [
    # 64,
    128,
    256,
    512,
    1024,
]  # average block size for variable block sparse attention
min_q_block_size = 128
min_kv_block_size = 128

b = 1
attn_modes = ["MHA"]  # MHA, GQA
nhqs = [4]
num_group = 4
# nhk = 16
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
            f"block sparse attn-{wd} attn_mode-{attn_mode}\n"
            f"{'n_head-' + str(nhq) if attn_mode == 'MHA' else f'n_head-{nhq}:{nhq // num_group}'} "
            f"avg_block_size-{block_size} seq_len {seqlen}"
        ),
        # Name for the plot. Used also as a file name for saving the plot.
        args={  # Values for function arguments not in `x_names` and `y_name`.
            "hd": hd,
            "wd": wd,
            "block_size": block_size,
            "seqlen": seqlen,
            "attn_mode": attn_mode,
            "nhq": nhq,
        },
    )
    for hd in ds
    for wd in wds
    for block_size in block_sizes
    for seqlen in seqlens
    for attn_mode in attn_modes
    for nhq in nhqs
]

seed_everything()


@perf_report(attn_flops_configs)
def sparse_attn_benchmark(
    sparsity_ratio, hd, wd, block_size, seqlen, attn_mode, nhq, attn_impl
):
    assert b == 1, "for now, we only supports b=1 for ffa"
    is_attn_impl_support_this_mask = True
    already_known_oom_before_run = False

    # --------- prepare arguments --------- #

    device = torch.cuda.current_device()
    orig_seq_len_q = orig_seq_len_k = seqlen  # fi square mask where sq == sk
    block_m = block_n = block_size

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
    # TODO: remove with a unified variable block mask generation function
    block_mask, block_row_sz, block_col_sz = generate_variable_block_sparse_pattern(
        nhq,
        nhk,
        seqlen,
        seqlen,
        num_q_blocks_orig,
        num_kv_blocks_orig,
        min_q_block_size=min_q_block_size,
        min_kv_block_size=min_kv_block_size,
        sparsity=sparsity_ratio,
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
    max_seqlen_q = block_row_sz.max().item()
    max_seqlen_k = block_col_sz.max().item()

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
        q = rearrange(q, "b s h d -> (b h s) 1 d")
        repeats = nhq // nhk
        k = torch.repeat_interleave(
            k, repeats=repeats, dim=2
        )  # we need to flatten k, v along head dimension for GQA setting.
        v = torch.repeat_interleave(v, repeats=repeats, dim=2)
        k = rearrange(k, "b s h d -> (b h s) 1 d")
        v = rearrange(v, "b s h d -> (b h s) 1 d")
        # q = q.view(b * orig_seq_len_q * nhq, 1, hd)
        # k = k.view(b * orig_seq_len_k * nhk, 1, hd)
        # v = v.view(b * orig_seq_len_k * nhk, 1, hd)

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
    is_attn_impl_support_this_mask = var_block_sparse_available(attn_impl, wd)
    if is_attn_impl_support_this_mask:
        if attn_impl == "ffa":
            # flatten headdim for ffa cause
            # flat_block_sparse_mask = flatten_head_mask(block_mask)
            flat_block_sparse_mask = flatten_block_mask(block_mask, nhq, nhk)

            # 3. Generate ranges from the flattened 2D mask
            q_ranges, k_ranges = generate_ranges_from_var_block_mask(
                flat_block_sparse_mask, block_row_sz, block_col_sz, nhq, nhk
            )
            # print(f"Number of non-empty blocks: {block_mask.sum().item()}")
            # print(q_ranges.shape, k_ranges.shape)
            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

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
                    auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
                )

            if wd == "bwd":
                try:
                    o, *rest = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {block_size} block_size "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "flashinfer":
            try:
                import flashinfer
            except ImportError:
                raise ImportError("Please install FlashInfer first.")

            q = q.view(b * nhq, orig_seq_len_q, hd).contiguous()
            k = k.view(b * nhk, orig_seq_len_k, hd).contiguous()
            v = v.view(b * nhk, orig_seq_len_k, hd).contiguous()
            # BUG: using original block mask will cause illegal access sometimes
            # block_mask_cpu = block_mask.detach().squeeze(0).cpu()
            kv_head_indices = torch.arange(0, nhq, nhq // nhk, device=device)
            block_mask_cpu = (
                torch.rand(nhk, num_q_blocks_orig, num_kv_blocks_orig) < sparsity_ratio
            )
            # flashinfer requires the row_sz to be of shape [num_q_heads, num_kv_block].
            block_row_sz_cpu = block_row_sz[..., kv_head_indices, :].detach().cpu()
            block_col_sz_cpu = block_col_sz.detach().cpu()

            # allocate 128MB workspace buffer
            workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=block_mask.device
            )
            wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
                workspace_buffer, backend="fa3"
            )

            wrapper.plan(
                block_mask_map=block_mask_cpu,
                block_row_sz=block_row_sz_cpu,
                block_col_sz=block_col_sz_cpu,
                num_qo_heads=nhq,
                num_kv_heads=nhk,
                head_dim=hd,
                q_data_type=q.dtype,
            )

            def fn():
                return wrapper.run(q, k, v)

        elif attn_impl == "flex":
            try:
                flex_mask = get_flex_mask_from_block_mask(
                    block_mask,
                    orig_seq_len_q,
                    orig_seq_len_k,
                    nhq,
                    nhk,
                    bsz=b,
                    block_row_sz=block_row_sz,
                    block_col_sz=block_col_sz,
                )
            except Exception as e:
                if "CUDA out of memory" not in str(e):
                    print(
                        f"Error occured before running {attn_impl} with {block_size} mask "
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
                                f"Error occured before running {attn_impl} with {block_size} mask "
                                f"when {seqlen=}, {hd=} during {wd}: {e=}"
                            )
                            raise e
                        already_known_oom_before_run = True

                    def fn():
                        o.backward(do, retain_graph=True)

        elif attn_impl == "sdpa":
            sdpa_mask = get_sdpa_mask_from_var_block_mask(
                block_mask,
                orig_seq_len_q,
                orig_seq_len_k,
                block_row_sz,
                block_col_sz,
                b,
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
                            f"Error occured before running {attn_impl} with {block_size} mask "
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
                        f"Error occured when running {attn_impl} with {block_size} block_size "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    raise e
                # -1 indicates oom
                perf_dict = {
                    "flops": [-1, -1, -1],
                    # "mem": [-1, -1, -1],
                }
                print(
                    f"OOM error occured when running for {attn_impl} with {block_size} block_size "
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

    sparse_attn_benchmark.run(
        print_data=True, print_value_on_bar=False, save_path=out_root
    )
