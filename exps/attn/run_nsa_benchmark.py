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

import math
import os
from datetime import datetime
from functools import partial

import torch
from baselines.attn_impl import ffa_func
from baselines.fsa.ops.FSA_topk_sparse_attention import (
    _topk_sparse_attention_fwd_opt_per_seq,
)
from baselines.nsa_ref.ops import compressed_attention, linear_compress
from baselines.nsa_ref.ops.topk_sparse_attention import _topk_sparse_attention_fwd
from baselines.nsa_ref.ops.utils import is_hopper_gpu
from baselines.utils import seed_everything
from einops import rearrange

from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.utils.sparse_utils import (
    generate_ranges_from_topk_index_token_major,
)

IS_HOPPER_GPU = is_hopper_gpu()

os.environ["PYTHONPATH"] = "exps/attn/baselines"


def create_cu_seqlens(seqlen: int) -> torch.Tensor:
    """Create cumulative sequence lengths tensor for batch processing."""
    return torch.arange(0, 2 * seqlen, seqlen, dtype=torch.int32)


impls = ["ffa", "fsa", "nsa_ref"]

# actual seqlen
seqlens = [65536]

sparsity_ratio = [0.1, 0.2]
# ss = [k * 1024 for k in [4, 96, 128]]
ds = [128]
wds = ["fwd"]
attn_modes = ["GQA"]  # MHA, GQA
nhqs = [32]
num_group = 8
block_sizes = [64, 128]

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
            f"{'n_head-' + str(nhq) if attn_mode == 'MHA' else f'n_head-{nhq}:{nhq // num_group}'} "
            f"block_size-{block_size} seq_len {seqlen}"
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
    block_m = num_group
    block_n = block_size
    num_q_blocks = orig_seq_len_q // block_m
    num_k_blocks = orig_seq_len_k // block_n
    topk = int(sparsity_ratio * num_k_blocks)

    num_q_heads = nhq
    if attn_mode == "MHA":
        num_k_heads = num_q_heads
    elif attn_mode == "GQA":
        num_k_heads = nhq // num_group
    head_dim = hd
    kernel_size = 32
    kernel_stride = 16

    # Create test data
    device = "cuda"

    assert (
        num_q_heads % num_k_heads == 0
    ), "num_k_heads must be divisible by num_q_heads"
    num_share_q_heads = num_q_heads // num_k_heads

    # Generate random q, k, v tensors
    q = torch.randn(seqlen, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(seqlen, num_k_heads, head_dim, device=device, dtype=dtype)

    # generate nsa parameters
    compress_key = torch.randn(
        num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype
    )
    compress_value = torch.randn(
        num_k_heads, head_dim * kernel_size, head_dim, device=device, dtype=dtype
    )
    intra_block_pe = torch.randn(
        num_k_heads, kernel_size, head_dim, device=device, dtype=dtype
    )

    # Create cumulative sequence lengths
    cu_seqlens = create_cu_seqlens(seqlen).to(device)

    print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"block_size: {block_size}, topk: {topk}")

    # Compute topk_idx using compressed_attention
    print("Computing topk_idx using compressed_attention...")
    compressed_k, compressed_cu_seqlens = linear_compress(
        k,
        compress_key,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        intra_block_pe,
    )
    compressed_v, _ = linear_compress(
        v,
        compress_value,
        cu_seqlens,
        kernel_size,
        kernel_stride,
        None,
    )

    compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    sm_scale = 1 / math.sqrt(head_dim)

    _, topk_idx = compressed_attention(
        q=q,
        k=compressed_k,
        v=compressed_v,
        kernel_size=kernel_size,
        kernel_stride=kernel_stride,
        block_size=block_size,
        topk=topk,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=compressed_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=compressed_seqlens.max().item(),
        sm_scale=None,
        init_blocks=1,
        local_blocks=2,
        parallel_topk_compute=False,
    )

    H, N, TopK = topk_idx.shape
    num_blocks = topk_idx.max().item() + 1
    causal = (topk_idx == -1).sum().item() != 0
    real_sparsity = (topk_idx != -1).sum().item() * block_n / (orig_seq_len_k * N * H)
    print(f"Real Sparsity: {real_sparsity * 100:.4f}% need to compute")
    attn_flops = 4 * orig_seq_len_q * orig_seq_len_k * nhq * hd * real_sparsity

    if attn_impl in ("ffa"):
        q = rearrange(
            q, "s h d -> (s h) 1 d"
        )  # NOTE: permuted for contiguous access of same group!
        k = rearrange(k, "s h d -> (h s) 1 d")
        v = rearrange(v, "s h d -> (h s) 1 d")

    print(f"topk_idx shape: {topk_idx.shape}, dtype: {topk_idx.dtype}")
    print(f"topk_idx range: [{topk_idx.min().item()}, {topk_idx.max().item()}]")
    print(f"H, N, TopK: {H}, {N}, {TopK}")
    print(f"num_blocks: {num_blocks}")
    print(f"causal: {causal}")
    print(f"cu_seqlen: {cu_seqlens.shape}")

    # --------- prepare grads --------- #

    if wd == "bwd":
        attn_flops = attn_flops * 2.5
        do = torch.randn_like(q)
        # require grads
        [x.requires_grad_(True) for x in [q, k, v, do]]

    if is_attn_impl_support_this_mask:
        if attn_impl == "ffa":
            q_ranges, k_ranges = generate_ranges_from_topk_index_token_major(
                topk_idx, num_group, block_size, orig_seq_len_k
            )
            attn_type_map = torch.zeros(len(q_ranges), dtype=torch.int32, device="cuda")

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    max_seqlen_q=num_group,
                    max_seqlen_k=block_size,
                    softmax_scale=sm_scale,
                    auto_range_merge=True,  # we should enable auto_range_merge for block sparse mask.
                    # ref_block_size=[num_group, block_size], # TODO: support SwapAB
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

        elif attn_impl == "fsa":
            func_opt = partial(
                _topk_sparse_attention_fwd_opt_per_seq,
                q,
                k,
                v,
                topk_idx,
                block_size,
                cu_seqlens,
                cu_seqlens,
                seqlen,
                seqlen,
                sm_scale,
                causal=causal,
            )

            def fn():
                return func_opt()

            if wd == "bwd":
                try:
                    o_opt, lse_opt, permute_results = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {block_size} block_size "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    pass

        elif attn_impl == "nsa_ref":
            func_ref = partial(
                _topk_sparse_attention_fwd,
                q,
                k,
                v,
                topk_idx,
                block_size,
                cu_seqlens,
                cu_seqlens,
                seqlen,
                seqlen,
                sm_scale,
            )

            def fn():
                return func_ref()

            if wd == "bwd":
                do = do.contiguous()
                try:
                    o_ref, lse_ref = fn()
                except Exception as e:
                    if "CUDA out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {block_size} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    pass

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
