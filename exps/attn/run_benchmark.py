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
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
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

from magi_attention import init_dist_attn_runtime_mgr
from magi_attention.benchmarking import Benchmark, do_bench_flops, perf_report
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.range import AttnRange
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    DispatchConfig,
    DistAttnConfig,
    GrpCollConfig,
    MinHeapDispatchAlg,
    OverlapConfig,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeMgr
from magi_attention.utils import (
    get_a2a_corr_factor,
    get_calc_cost_factor,
    get_comm_cost_factor,
)
from magi_attention.utils._utils import make_attn_mask_from_ffa_args
from magi_attention.testing.precision import (
    H100_MATMUL_MFU,
    H100_NVLINK_A2A_BWU,
    H100_NVLINK_BANDWIDTH,
    H100_TFLOPS_16,
)

# impls = ["ffa", "fa3", "fa4", "cudnn", "fa2", "flex", "sdpa"]  # all except torch native
# impls = ["cudnn", "fa4", "ffa_fa4"] # for blackwell
# impls = ["ffa", "cudnn", "fa3", "fa4"]  # for hopper
impls = ["magi", "cudnn"]  # for CP benchmark: magi vs cudnn

# --------- global variables for magi distributed attention --------- #
_magi_initialized = False
_cp_group = None
_cp_mesh = None
_world_size = 1
_rank = 0

mask_types = ["full"]
# mask_types = ["causal"]
# mask_types = ["varlen_full"]
# mask_types = ["varlen_causal"]
# mask_types = ["sliding_window_causal"]
# mask_types = ["varlen_block_causal"]

# uniform varlen, each doc with fixed seqlen
# varlen_seqlen_distribution = {
#     (2048, 2049): 1.0, # 2k seqlen per doc
# }

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

# ss = [k * 1024 for k in [1, 2, 4, 8, 16, 24, 32, 64]]  # original
ss = [k * 1024 for k in [16, 24, 32, 64, 128]]  # larger seqlens for magi CP benchmark
ds = [128]
wds = ["fwd"]  # Only fwd for quick test, add "bwd" for full benchmark


b = 1
nhq = 48  # query heads per GPU (after TP split)
nhk = 8   # kv heads per GPU (after TP split)
dtype = torch.bfloat16

# For magi CP benchmark: simulate TP vs CP comparison
# - cuDNN: runs with nhq, nhk (simulating TP split heads per GPU)
# - MAGI: runs with nhq * world_size, nhk * world_size (full heads, CP splits sequence)

window_size = 1024
block_size = 2048
num_varlen_samples = 16

bias = None
softmax_scale = None
dropout_p = 0.0
return_attn_probs = False

quantiles = [0.5, 0.2, 0.8]


def init_magi_distributed():
    """Check that distributed environment is initialized for magi CP benchmark.
    
    Actual initialization happens in run_benchmark_worker() via mp.spawn().
    """
    global _magi_initialized
    
    if not _magi_initialized:
        raise RuntimeError(
            "Magi distributed environment not initialized. "
            "This should be called from run_benchmark_worker()."
        )


attn_flops_configs = [
    Benchmark(
        x_names=["seqlen"],  # Argument names to use as an x-axis for the plot.
        x_vals=ss,  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="attn_impl",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=impls,  # Possible values for `line_arg`.
        line_names=impls,  # Label name for the lines.
        styles=[  # Line styles.
            ("green", "-"),   # magi
            ("orange", "--"),  # cudnn
            # ("steelblue", "--"),
            # ("red", "-"),
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
                sdpa_mask = make_attn_mask_from_ffa_args(
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
    if attn_impl in ("ffa", "ffa_fa4", "cudnn"):
        q = q.view(b * sq, nhq, hd)
        k = k.view(b * sk, nhk, hd)
        v = v.view(b * sk, nhk, hd)

        if attn_impl == "cudnn":
            if "varlen_block_causal" in mask_type:
                is_attn_impl_support_this_mask = False

    # fa style shape:
    #   non-varlen: (b,s,h,d)
    #   varlen: (t,h,d)
    if attn_impl in ("fa2", "fa3", "fa4"):
        if "varlen" in mask_type:
            q = q.view(b * sq, nhq, hd)
            k = k.view(b * sk, nhk, hd)
            v = v.view(b * sk, nhk, hd)

        if "block_causal" in mask_type:
            is_attn_impl_support_this_mask = False

        if attn_impl == "fa4":
            window_size_tuple = tuple(
                [None if x == -1 else x for x in window_size_tuple]
            )

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

    elif attn_impl == "fa4":
        if "varlen" in mask_type:

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

    elif attn_impl == "magi":
        # Initialize distributed environment for magi
        init_magi_distributed()
        
        # For TP vs CP comparison:
        # - cuDNN runs with nhq, nhk (TP splits heads, each GPU has nhq/nhk heads)
        # - MAGI runs with nhq * world_size, nhk * world_size (full heads, CP splits sequence)
        magi_nhq = nhq * _world_size
        magi_nhk = nhk * _world_size
        
        # Recalculate attn_flops with magi head counts for fair comparison
        if "varlen" in mask_type or mask_type == "sliding_window_causal":
            magi_attn_flops_dict = calculate_attn_flops(
                q_ranges=q_ranges_,
                k_ranges=k_ranges_,
                attn_mask_type=[
                    AttnMaskType.CAUSAL if c else AttnMaskType.FULL
                    for c in is_causal_mapping_
                ],
                total_seqlen_q=sq,
                num_heads_q=magi_nhq,
                head_dim=hd,
            )
        else:
            magi_attn_flops_dict = calculate_attn_flops(
                q_ranges=AttnRanges.from_ranges([[0, sq]]),
                k_ranges=AttnRanges.from_ranges([[0, sk]]),
                attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL],
                total_seqlen_q=sq,
                num_heads_q=magi_nhq,
                head_dim=hd,
            )
        # Use per-GPU FLOPS for fair comparison with cuDNN (single GPU)
        # Total FLOPS / world_size = per-GPU FLOPS
        attn_flops = magi_attn_flops_dict[wd] / _world_size
        
        # Build AttnRanges objects (reuse already calculated ranges if available)
        if "varlen" in mask_type or mask_type == "sliding_window_causal":
            # q_ranges_ and k_ranges_ are already calculated above for these cases
            q_ranges_obj = q_ranges_
            k_ranges_obj = k_ranges_
            # Use is_causal_mapping_ calculated above
            attn_mask_type_list = [
                AttnMaskType.CAUSAL if c else AttnMaskType.FULL
                for c in is_causal_mapping_
            ]
        else:
            # Simple full or causal case
            q_ranges_obj = AttnRanges.from_ranges([[0, sq]])
            k_ranges_obj = AttnRanges.from_ranges([[0, sk]])
            # Build attn_mask_type list
            attn_mask_type_list = [
                AttnMaskType.CAUSAL if causal else AttnMaskType.FULL
            ]
        
        # Create DistAttnConfig with magi head counts
        dist_attn_config = DistAttnConfig(
            dispatch_config=DispatchConfig(
                alg=MinHeapDispatchAlg()
            ),
            overlap_config=OverlapConfig(
                enable=True,
                degree=4,
                min_chunk_size=256,
                max_num_chunks=64,
                calc_cost_factor=get_calc_cost_factor(
                    num_heads_q=magi_nhq,
                    head_dim=hd,
                    tflops=H100_TFLOPS_16,
                    mfu=H100_MATMUL_MFU,
                ),
                comm_cost_factor=get_comm_cost_factor(
                    num_heads_kv=magi_nhk,
                    head_dim=hd,
                    bandwidth=H100_NVLINK_BANDWIDTH,
                    bwu=H100_NVLINK_A2A_BWU,
                    corr_factor=get_a2a_corr_factor(_world_size),
                ),
            ),
            grpcoll_config=GrpCollConfig(),
        )
        
        # Initialize DistAttnRuntimeMgr with magi head counts
        dist_attn_runtime_mgr: DistAttnRuntimeMgr = init_dist_attn_runtime_mgr(
            q_ranges=q_ranges_obj,
            k_ranges=k_ranges_obj,
            attn_mask_type=attn_mask_type_list,
            total_seqlen_q=sq,
            total_seqlen_k=sk,
            chunk_size=512,
            cp_group=_cp_group,
            is_same_source=True,
            is_q_permutable=True,
            is_k_permutable=True,
            dist_attn_config=dist_attn_config,
            cp_mesh=_cp_mesh,
            num_heads_q=magi_nhq,
            num_heads_kv=magi_nhk,
        )
        
        # Create new qkv tensors with magi head counts (full heads for CP)
        q_magi = torch.randn(b * sq, magi_nhq, hd, device=device, dtype=dtype, requires_grad=False)
        k_magi = torch.randn(b * sk, magi_nhk, hd, device=device, dtype=dtype, requires_grad=False)
        v_magi = torch.randn(b * sk, magi_nhk, hd, device=device, dtype=dtype, requires_grad=False)
        
        # Sync data across ranks
        dist.all_reduce(q_magi.data, group=_cp_group)
        dist.all_reduce(k_magi.data, group=_cp_group)
        dist.all_reduce(v_magi.data, group=_cp_group)
        
        # Dispatch global qkv to local qkv
        local_q = dist_attn_runtime_mgr.dispatch_qo(q_magi)
        local_k = dist_attn_runtime_mgr.dispatch_kv(k_magi)
        local_v = dist_attn_runtime_mgr.dispatch_kv(v_magi)
        
        def fn():
            # Only measure calc_attn time (no dispatch/undispatch overhead)
            local_out, local_lse = dist_attn_runtime_mgr.calc_attn(
                q=local_q,
                k=local_k,
                v=local_v,
                sink=None,
                softmax_scale=softmax_scale,
                softcap=0.0,
            )
            return local_out
        
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

            if wd == "bwd":
                # Create do with magi head counts
                do_magi = torch.randn(b * sq, magi_nhq, hd, device=device, dtype=dtype)
                dist.all_reduce(do_magi.data, group=_cp_group)

            def fn():
                o.backward(do_magi, retain_graph=True)

    elif attn_impl == "ffa":

        def fn():
            return ffa_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
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

    elif attn_impl == "ffa_fa4":
        # Warmup call to create cached FA4AttnArg (reuse_attn_arg=False)
        _ = ffa_fa4_func(
            q,
            k,
            v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            reuse_attn_arg=False,
        )
        torch.cuda.synchronize()  # Wait for warmup kernel to complete

        def fn():
            # Use cached FA4AttnArg for accurate kernel timing
            return ffa_fa4_func(
                q,
                k,
                v,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                reuse_attn_arg=True,
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

                flops = perf_dict["flops"]
                if not isinstance(flops, list):
                    flops = [flops]  # type: ignore[unreachable]
                perf_dict["flops"] = list(map(ms_to_tflops, flops))

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


def run_benchmark_worker(rank: int, world_size: int, out_root: str):
    """Worker function for each process in distributed benchmark."""
    global _magi_initialized, _cp_group, _cp_mesh, _world_size, _rank
    
    # Set device for this rank
    torch.cuda.set_device(rank)
    
    # Initialize distributed process group
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method="env://",
    )
    
    _world_size = world_size
    _rank = rank
    
    # Create CP group with all ranks
    _cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    
    # Create device mesh for hierarchical comm
    world_size_inter_node, world_size_intra_node = {
        1: (1, 1),
        2: (1, 2),
        3: (3, 1),
        4: (2, 2),
        5: (1, 5),
        6: (3, 2),
        7: (1, 7),
        8: (2, 4),
    }.get(world_size, (1, world_size))
    
    _cp_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size_inter_node, world_size_intra_node),
        mesh_dim_names=("inter", "intra"),
    )
    
    _magi_initialized = True
    
    if rank == 0:
        print(f"[MAGI] Initialized distributed environment with world_size={world_size}")
    
    # Run benchmark (only rank 0 prints results)
    attn_benchmark.run(
        print_data=(rank == 0),
        print_value_on_bar=False,
        save_path=out_root if rank == 0 else None,
    )
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    out_root = os.path.join(
        script_dir, os.path.join("outs", f"bench_attn_{current_time}")
    )

    if "magi" in impls:
        # Use torch.multiprocessing to spawn multiple processes
        import torch.multiprocessing as mp
        
        world_size = torch.cuda.device_count()
        print(f"[MAGI] Spawning {world_size} processes for distributed benchmark...")
        
        # Set environment variables for init_method="env://"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        
        mp.spawn(
            run_benchmark_worker,
            args=(world_size, out_root),
            nprocs=world_size,
            join=True,
        )
    else:
        # Non-distributed benchmark
        attn_benchmark.run(print_data=True, print_value_on_bar=False, save_path=out_root)
