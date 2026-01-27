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
from typing import Any

import torch
from baselines.attn_impl import (
    cudnn_fused_attn_func,
    fa2_func,
    fa2_varlen_func,
    fa3_func,
    fa3_varlen_func,
    fa4_func,
    fa4_varlen_func,
    ffa_fa4_func,
    ffa_func,
    flashmask_func,
    flex_attn_func,
    sdpa_func,
    torch_attn_func,
)
from baselines.utils import (
    calculate_attn_flops,
    curanges2document_id,
    generate_flashmask_indices,
    make_block_causal_varlen_block_mask,
    make_block_causal_varlen_score_mod,
    make_causal_block_mask,
    make_causal_blockwise_block_mask,
    make_causal_blockwise_mask_score_mod,
    make_causal_mask_score_mod,
    make_global_sliding_window_block_mask,
    make_global_sliding_window_mask_score_mod,
    make_prefix_lm_causal_block_mask,
    make_prefix_lm_causal_mask_score_mod,
    make_prefix_lm_varlen_block_mask,
    make_prefix_lm_varlen_mask_score_mod,
    make_share_question_block_mask,
    make_share_question_mask_score_mod,
    make_sliding_window_causal_block_mask,
    make_sliding_window_causal_mask_score_mod,
    make_sliding_window_full_block_mask,
    make_sliding_window_full_mask_score_mod,
    make_varlen_causal_block_mask,
    make_varlen_causal_mask_score_mod,
    make_varlen_full_block_mask,
    make_varlen_full_mask_score_mod,
    seed_everything,
)
from einops import rearrange

from exps.dist_attn.benchmark.enums import FlashMaskType
from exps.dist_attn.benchmark.mask import MaskIterator
from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.utils._utils import make_attn_mask_from_ffa_args

_ENABLE_GC = False  # whether to gc.collect when running bench

# all attn baselines
impls = [
    # "fa2",
    # "fa3",
    # "fa4",
    # "ffa",
    # "ffa_fa4",
    # "cudnn",
    # "flex",
    # "flash_mask",
    # "torch",
    "sdpa",
]  # ignore torch native to avoid OOMzsZasZ·

mask_types = [
    "full",
    # "causal",
    # "full_document",
    # "causal_document",
    # "sliding_window",
    # "sliding_window_causal",
    # "share_question",
    # "causal_blockwise",
    # "prefix_lm_causal",
    # "prefix_lm_document",
    # "global_sliding_window",
    # "block_causal_document",
]

# total seqlen
ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32, 48, 56, 64]]
ds = [128]  # head dim
# workload to bench
wds = [
    "fwd",
    # "bwd",
]

b = 1  # batch size
nhq = 64  # num heads query
nhk = 8  # num heads key/value
dtype = torch.bfloat16  # data dtype

WINDOW_SIZE = 1024  # window size for sliding window mask
BLOCK_SIZE = 512  # block size for varlen block causal
PREFIX_LENGTH = 512  # prefix length for prefix lm mask
SEED = 42  # random seed

quantiles = [0.5, 0.2, 0.8]
mask_nums = 3  # number of masks to sample

# cuDNN, FA2, FA3 and FA4 only support non-heterogeneous mask types.
regular_mask = [
    FlashMaskType.FULL,
    FlashMaskType.CAUSAL,
    FlashMaskType.FULL_DOCUMENT,
    FlashMaskType.CAUSAL_DOCUMENT,
    FlashMaskType.SLIDING_WINDOW_CAUSAL,
    FlashMaskType.SLIDING_WINDOW,
]

softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

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
            "mask_nums": mask_nums,
        },
    )
    for hd in ds
    for wd in wds
    for mask_type in mask_types
]


attn_mask_mapping = {
    AttnMaskType.FULL: 0,
    AttnMaskType.CAUSAL: 1,
    AttnMaskType.INVCAUSAL: 2,
    AttnMaskType.BICAUSAL: 3,
}


def to_attn_mask_type_tensor(attn_mask_type: list[AttnMaskType], device: torch.device):
    attn_mask_type_idx = [attn_mask_mapping[mask] for mask in attn_mask_type]
    return torch.tensor(attn_mask_type_idx, dtype=torch.int32, device=device)


# Cache the mask_iterator to avoid redundant reconstruction when benchmarking multiple seqlens.
_MASK_ITERATOR_CACHE: dict[Any, MaskIterator] = {}


@perf_report(attn_flops_configs)
def attn_benchmark(seqlen, hd, wd, mask_type, attn_impl, mask_nums):
    seed_everything(SEED)
    perf_dict_total = {
        "flops": [0, 0, 0],
        "mem": [0, 0, 0],
    }

    flash_mask_type = FlashMaskType(mask_type)
    # -----    init mask iterator with dataset sampler   ---- #

    cache_key = (flash_mask_type.value, seqlen)
    if cache_key in _MASK_ITERATOR_CACHE:
        mask_iterator = _MASK_ITERATOR_CACHE[cache_key]
        mask_iterator.reset(re_shuffle=False)
    else:
        mask_iterator = MaskIterator(
            num_iterations=mask_nums,
            mask_type=flash_mask_type,
            total_seqlen=seqlen,
            data_path="../dist_attn/benchmark/datasets/default/doc_length_distribution.csv",
            chunk_ratio=0.25,
            is_binned=True,
            window_size=(WINDOW_SIZE, WINDOW_SIZE),
            to_attn_ranges=True,
            seed=SEED,
            drop_thres=-1,
            **{
                "prefix_length": PREFIX_LENGTH,
                "block_size": BLOCK_SIZE,
            },
        )
        _MASK_ITERATOR_CACHE[cache_key] = mask_iterator

    for mask_idx, (q_ranges_, k_ranges_, attn_mask_type, mask_factors) in enumerate(
        mask_iterator
    ):
        assert b == 1, "for now, we only supports b=1 for ffa"
        is_attn_impl_support_this_mask = True
        already_known_oom_before_run = False

        # --------- prepare arguments --------- #

        device = torch.cuda.current_device()
        sq = sk = seqlen  # fi square mask where sq == sk
        sdpa_mask = None

        # calculate attn flops
        if flash_mask_type == FlashMaskType.SLIDING_WINDOW_CAUSAL:
            causal = True
            window_size_tuple = (WINDOW_SIZE, 0)
            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

        elif flash_mask_type == FlashMaskType.SLIDING_WINDOW:
            causal = False
            window_size_tuple = (WINDOW_SIZE, WINDOW_SIZE)
            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

        elif (
            flash_mask_type == FlashMaskType.FULL_DOCUMENT
            or flash_mask_type == FlashMaskType.CAUSAL_DOCUMENT
        ):
            causal = attn_mask_type[0] == AttnMaskType.CAUSAL
            cu_seqlens = q_ranges_.to_cu_seqlens(seqlen)
            cu_ranges = q_ranges_.to_naive_ranges()
            document_id = curanges2document_id(cu_ranges)
            max_seqlen_q = q_ranges_.max_seqlen
            max_seqlen_kv = k_ranges_.max_seqlen

            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            cu_seqlens_q = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
            window_size_tuple = (-1, -1)
        elif (
            flash_mask_type == FlashMaskType.FULL
            or flash_mask_type == FlashMaskType.CAUSAL
        ):
            causal = attn_mask_type[0] == AttnMaskType.CAUSAL
            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            max_seqlen_q = sq
            max_seqlen_kv = sk
            cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_k = torch.tensor([0, sq], dtype=torch.int32, device=device)
            cu_seqlens_kv = torch.tensor([0, sk], dtype=torch.int32, device=device)

            window_size_tuple = (-1, -1)
        else:
            causal = None
            # other mask logic
            attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=attn_mask_type,
                total_seqlen_q=sq,
                num_heads_q=nhq,
                head_dim=hd,
            )

            if (
                flash_mask_type == FlashMaskType.PREFIX_LM_CAUSAL
                or flash_mask_type == FlashMaskType.PREFIX_LM_DOCUMENT
            ):
                cu_ranges = mask_factors.cu_ranges
                prefix_length = mask_factors.prefix_length
                if mask_factors.cu_seqlens is not None:
                    cu_seqlens_kv = torch.tensor(
                        mask_factors.cu_seqlens, dtype=torch.int32, device=device
                    )
                    document_id = curanges2document_id(cu_ranges)

            if (
                flash_mask_type == FlashMaskType.SHARE_QUESTION
                or flash_mask_type == FlashMaskType.CAUSAL_BLOCKWISE
            ):
                cu_ranges = mask_factors.cu_ranges
                document_id = curanges2document_id(cu_ranges)

            if flash_mask_type == FlashMaskType.GLOBAL_SLIDING_WINDOW:
                global_window_size = WINDOW_SIZE

            if flash_mask_type == FlashMaskType.BLOCK_CAUSAL_DOCUMENT:
                cu_seqlens = mask_factors.cu_seqlens
                cu_ranges = mask_factors.cu_ranges
                block_size = mask_factors.block_size

            max_seqlen_q = sq
            max_seqlen_kv = sk
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
                prefix_length = mask_factors.prefix_length
                if flash_mask_type == FlashMaskType.FULL:
                    score_mod = None
                    block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL:
                    try:
                        block_mask = make_causal_block_mask(sq, sk)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_causal_mask_score_mod()
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SLIDING_WINDOW_CAUSAL:
                    try:
                        block_mask = make_sliding_window_causal_block_mask(
                            sq, sk, window_size=WINDOW_SIZE
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_sliding_window_causal_mask_score_mod(
                            window_size=WINDOW_SIZE
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL_DOCUMENT:
                    try:
                        block_mask = make_varlen_causal_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_varlen_causal_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.FULL_DOCUMENT:
                    try:
                        block_mask = make_varlen_full_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_varlen_full_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SHARE_QUESTION:
                    try:
                        block_mask = make_share_question_block_mask(sq, sk, document_id)
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_share_question_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.CAUSAL_BLOCKWISE:
                    try:
                        block_mask = make_causal_blockwise_block_mask(
                            sq, sk, document_id
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_causal_blockwise_mask_score_mod(document_id)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.PREFIX_LM_CAUSAL:
                    try:
                        block_mask = make_prefix_lm_causal_block_mask(
                            sq, sk, prefix_length
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_prefix_lm_causal_mask_score_mod(prefix_length)
                        block_mask = None
                elif flash_mask_type == FlashMaskType.PREFIX_LM_DOCUMENT:
                    try:
                        block_mask = make_prefix_lm_varlen_block_mask(
                            sq, sk, prefix_length, document_id, cu_seqlens_kv
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_prefix_lm_varlen_mask_score_mod(
                            prefix_length, document_id, cu_seqlens_kv
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.GLOBAL_SLIDING_WINDOW:
                    try:
                        block_mask = make_global_sliding_window_block_mask(
                            sq, sk, global_window_size
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_global_sliding_window_mask_score_mod(
                            window_size=global_window_size
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.SLIDING_WINDOW:
                    try:
                        block_mask = make_sliding_window_full_block_mask(
                            sq, sk, WINDOW_SIZE
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_sliding_window_full_mask_score_mod(
                            window_size=WINDOW_SIZE
                        )
                        block_mask = None
                elif flash_mask_type == FlashMaskType.BLOCK_CAUSAL_DOCUMENT:
                    document_id = curanges2document_id(cu_ranges)
                    try:
                        block_mask = make_block_causal_varlen_block_mask(
                            sq, sk, block_size, document_id
                        )
                        score_mod = None
                    except RuntimeError:
                        score_mod = make_block_causal_varlen_score_mod(
                            block_size, document_id
                        )
                        block_mask = None
                else:
                    raise NotImplementedError(
                        f"mask type {mask_type} not supported for flex attn"
                    )
            elif (
                flash_mask_type != FlashMaskType.FULL
                and flash_mask_type != FlashMaskType.CAUSAL
            ):
                if "sliding_window" in mask_type and WINDOW_SIZE + 1 >= seqlen:
                    causal = "causal" in mask_type
                    sdpa_mask = None
                else:
                    try:
                        attn_mask_type_num = [
                            attn_mask_mapping[mask] for mask in attn_mask_type
                        ]
                        sdpa_mask = make_attn_mask_from_ffa_args(
                            q_ranges=q_ranges_,
                            k_ranges=k_ranges_,
                            attn_type_map=attn_mask_type_num,
                            total_seqlen_q=sq,
                            total_seqlen_k=sk,
                            device=torch.cuda.current_device(),
                        )
                    except RuntimeError as e:
                        print(f"make varlen causal sdpa mask failed: {e}")

        # ffa style shape: (t,h,d)
        if attn_impl in ("ffa", "ffa_fa4", "cudnn"):
            q = q.view(b * sq, nhq, hd)
            k = k.view(b * sk, nhk, hd)
            v = v.view(b * sk, nhk, hd)

            if attn_impl == "cudnn":
                if flash_mask_type not in regular_mask:
                    is_attn_impl_support_this_mask = False

        # fa style shape:
        #   non-varlen: (b,s,h,d)
        #   varlen: (t,h,d)
        if attn_impl in ("fa2", "fa3", "fa4"):
            if "document" in mask_type:
                q = q.view(b * sq, nhq, hd)
                k = k.view(b * sk, nhk, hd)
                v = v.view(b * sk, nhk, hd)

            if flash_mask_type not in regular_mask:
                is_attn_impl_support_this_mask = False

            if attn_impl == "fa4":
                window_size_tuple = tuple(
                    [None if x == -1 else x for x in window_size_tuple]
                )

        if attn_impl in ("torch") and nhq != nhk:
            assert nhq % nhk == 0
            repeat_times = nhq // nhk
            k = torch.repeat_interleave(k, repeat_times, dim=1)
            v = torch.repeat_interleave(v, repeat_times, dim=1)

        if attn_impl in ("flash_mask"):
            import paddle

            q = paddle.to_tensor(
                q.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )
            k = paddle.to_tensor(
                k.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )
            v = paddle.to_tensor(
                v.detach().cpu().to(torch.float32).numpy(),
                dtype="bfloat16",
                place=paddle.CUDAPlace(0),
            )

        # --------- prepare grads --------- #

        if wd in ["bwd", "1f1b"]:
            if attn_impl not in ("flash_mask"):
                do = torch.randn_like(q)
                # require grads
                [x.requires_grad_(True) for x in [q, k, v, do]]
            else:
                import paddle

                do = paddle.randn_like(q)
                for x in [q, k, v, do]:
                    x.stop_gradient = False

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

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
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

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "fa2":
            if "document" in mask_type:

                def fn():
                    return fa2_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_kv,
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

            if wd == "bwd" and is_attn_impl_support_this_mask:
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
            if "document" in mask_type:

                def fn():
                    return fa3_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_kv,
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

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "fa4":
            if "document" in mask_type:

                def fn():
                    return fa4_varlen_func(
                        q,
                        k,
                        v,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size_tuple,
                    )[0]

            else:

                def fn():
                    return fa4_func(
                        q,
                        k,
                        v,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size_tuple,
                    )[0]

            if wd == "bwd" and is_attn_impl_support_this_mask:
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

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
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

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured before running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    already_known_oom_before_run = True

                def fn():
                    o.backward(do, retain_graph=True)

        elif attn_impl == "ffa":
            q_ranges_tensor = q_ranges_.to_tensor(device)
            k_ranges_tensor = k_ranges_.to_tensor(device)
            attn_type_map_tensor = to_attn_mask_type_tensor(attn_mask_type, device)

            def fn():
                return ffa_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges_tensor,
                    k_ranges=k_ranges_tensor,
                    attn_type_map=attn_type_map_tensor,
                )

            if wd == "bwd" and is_attn_impl_support_this_mask:
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

        elif attn_impl == "ffa_fa4":
            q_ranges_tensor = q_ranges_.to_tensor(device)
            k_ranges_tensor = k_ranges_.to_tensor(device)
            attn_type_map_tensor = to_attn_mask_type_tensor(attn_mask_type, device)
            # Warmup call to create cached FA4AttnArg (reuse_attn_arg=False)
            _ = ffa_fa4_func(
                q,
                k,
                v,
                q_ranges=q_ranges_tensor,
                k_ranges=k_ranges_tensor,
                attn_type_map=attn_type_map_tensor,
                reuse_attn_arg=False,
            )
            torch.cuda.synchronize()  # Wait for warmup kernel to complete

            def fn():
                # Use cached FA4AttnArg for accurate kernel timing
                return ffa_fa4_func(
                    q,
                    k,
                    v,
                    q_ranges=q_ranges_tensor,
                    k_ranges=k_ranges_tensor,
                    attn_type_map=attn_type_map_tensor,
                    reuse_attn_arg=True,
                )

            if wd == "bwd" and is_attn_impl_support_this_mask:
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

        elif attn_impl == "flash_mask":
            (
                attn_mask_startend_row_indices,
                flashmask_is_causal,
            ) = generate_flashmask_indices(
                sq,
                sk,
                flash_mask_type,
                window_size=WINDOW_SIZE,
                cu_ranges=mask_factors.cu_ranges,
                prefix_length=mask_factors.prefix_length,
            )

            def fn():
                return flashmask_func(
                    q,
                    k,
                    v,
                    startend_row_indices=attn_mask_startend_row_indices,
                    causal=flashmask_is_causal,
                )

            if wd == "bwd" and is_attn_impl_support_this_mask:
                try:
                    o = fn()
                except Exception as e:
                    if "out of memory" not in str(e):
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
                    "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                    "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                }
                perf_dict_total = {
                    "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                    "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                }
                break
            else:
                try:
                    # disable mem test to only test flops for now
                    if _ENABLE_GC:
                        do_bench_kwargs = {
                            "to_gc_collect": (mask_idx >= mask_nums - 1),
                            "to_empty_cache": (mask_idx >= mask_nums - 1),
                        }
                    else:
                        do_bench_kwargs = {
                            "to_gc_collect": False,
                            "to_empty_cache": (mask_idx >= mask_nums - 1),
                        }
                    perf_dict = do_bench_flops(
                        fn,
                        quantiles=quantiles,
                        mem_record_mode="peak",
                        warmup=5,
                        rep=20,
                        **do_bench_kwargs,
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
                    if "out of memory" not in str(e):
                        print(
                            f"Error occured when running {attn_impl} with {mask_type} mask "
                            f"when {seqlen=}, {hd=} during {wd}: {e=}"
                        )
                        raise e
                    # -1 indicates oom
                    perf_dict = {
                        "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                        "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                    }
                    perf_dict_total = {
                        "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                        "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
                    }
                    print(
                        f"OOM error occured when running for {attn_impl} with {mask_type} mask "
                        f"when {seqlen=}, {hd=} during {wd}: {e=}"
                    )
                    break
        else:
            # -2 indicates not support
            perf_dict = {
                "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
            }
            perf_dict_total = {
                "flops": [-1 * mask_nums, -1 * mask_nums, -1 * mask_nums],
                "mem": [-2 * mask_nums, -2 * mask_nums, -2 * mask_nums],
            }
            break

        perf_dict_total["flops"] = [
            perf_dict_total["flops"][i] + perf_dict["flops"][i]
            for i in range(len(perf_dict_total["flops"]))
        ]
        # perf_dict_total["mem"] = [
        #     perf_dict_total["mem"][i] + perf_dict["mem"][i]
        #     for i in range(len(perf_dict_total["mem"]))
        # ]

    perf_dict_total["flops"] = [
        metric / mask_nums for metric in perf_dict_total["flops"]
    ]
    # perf_dict_total["mem"] = [metric / mask_nums for metric in perf_dict_total["mem"]]

    return perf_dict_total


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_{current_time}")
    )

    attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)

    del _MASK_ITERATOR_CACHE
    _MASK_ITERATOR_CACHE = None  # type: ignore[assignment]
    import gc

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
