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

from dataclasses import dataclass

import torch

from exps.dist_attn.baselines.interface import AttnImpl
from exps.dist_attn.baselines.utils_cp import AttnBackend
from exps.dist_attn.benchmark.enums import FlashMaskType
from magi_attention.common.enum import AttnOverlapMode
from magi_attention.meta.solver.dispatch_solver import MinHeapDispatchAlg

"""
This file defines all cp benchmark configurations.
Each config name should start with a capital letter to be cnocluded,
all other keys will be discarded. The config will be loaded by `run_benchmark.py`
and specified via `--config` in `run_benchmark.sh`.
"""


SEED = 42


@dataclass
class ENVVAR_CONFIG:
    """
    Define env vars for MagiAttention here and dynamic switch in `run_benchmark.py`, avoid modify bash script
    and simplify benchmarking process.
        - EXTEND_ENVVAR_CONFIG: Specifies all environment variable product combinations used in benchmarking,
            use extend_labels to assign a custom name to each extension.
            If provided, the number of extend_labels must exactly match the number of generated extensions.
            If not provided, the benchmark will assign default suffixes such as -0, -1, etc.
        - use_extend_labels: specifies whether the values in extend_labels are appended to the result labels.
            This option is only valid when each baseline has exactly one environment-variable extension;
            otherwise, an error will be raised.
    By default, use_extend_labels is set to False, which means no label extensions are applied to any baseline.

    Example of using extensions:
        1.  Define multiple values for certain environment variables, e.g., NCCL_CGA_CLUSTER_SIZE: [1, 4].
        2.  Define multiple extend_labels, e.g., ["exp0", "exp1"], or leave it empty.
        3.  Enable use_extend_labels to automatically extend labels for each configuration combination.
    """

    EXTEND_ENVVAR_CONFIG = {
        AttnImpl.MAGI_ATTENTION: {
            "envvars": {
                "MAGI_ATTENTION_HIERARCHICAL_COMM": [False],
                "NCCL_CGA_CLUSTER_SIZE": [1],
            },
            "extend_labels": ["exp0"],
        }
    }
    use_extend_labels = False


@dataclass
class BENCH_MODE:
    """
    Benchmark runtime mode configuration.
        - enable_profile: whether to enable nsys profiling.
        - profile_only: if True, only profile the benchmark; skip flops/memory recording and skip plotting results.
        - stat_warmup_iters: number of warmup iterations for statistical benchmark recording.
        - stat_iters: number of iterations for statistical benchmark recording (flops/memory).
        - profile_iters: number of iterations for profiling.
        - profile_warmup_iters: number of warmup iterations for profiling.
    """

    enable_profile = False
    profile_only = False
    stat_warmup_iters = 5
    stat_iters = 20
    profile_warmup_iters = 1
    profile_iters = 3


@dataclass
class BENCH_CONFIG:
    """
    Benchmark combination configuration.
        - quantiles: quantile points used for summarizing latency/throughput results.
        - bench_flops: whether to benchmark flops.
        - bench_mem: whether to benchmark memory.
        - bench_mode: mode to summarize latency/throughput results (mean, median, min, max).
        - output_path: output folder.
        - mask_pattern:
            list of attention masks to run evaluate (FULL, CAUSAL, Varlen-FULL, Varlen-CAUSAL).
        - dist_attn_impl:
            distributed attention implementations to evaluate (Ulysess, Ring-P2P, Ring-AllGather,
            USP, LoongTrain, MagiAttention).
        - workload:
            pipeline schedule modes to evaluate ("fwd"=forward only, "bwd"=backward only, "1f1b"=forward+backward).

        e.g.
        for mask in mask_patterm:
            for dist_attn in dist_attn_impl:
                for wd in workload:
                    do_bench
    """

    quantiles = [0.5, 0.2, 0.8]
    bench_flops = True
    bench_mem = False
    bench_mode = "mean"
    output_path = "./outs"
    mask_pattern = [
        FlashMaskType.FULL,
        FlashMaskType.CAUSAL,
        FlashMaskType.FULL_DOCUMENT,
        FlashMaskType.CAUSAL_DOCUMENT,
    ]
    dist_attn_impl = [
        AttnImpl.ULYSSES,
        AttnImpl.RING_P2P,
        AttnImpl.RING_ALLGATHER,
        AttnImpl.USP,
        AttnImpl.LOONGTRAIN,
        AttnImpl.MAGI_ATTENTION,
        AttnImpl.HYBRID_DCP,
    ]
    workload = [
        "fwd",
        "bwd",
        "1f1b",
    ]


@dataclass
class SAMPLE_CONFIG:
    """
    Mask sampler configuration.
        - dataset_path: path to the csv or json file of dataset length distribution.
        - pack_num: number of data packs to evaluate.
        - chunk_ratio: ratio used to determine chunk size; sequences longer than `pack_len * chunk_ratio`
            are split into chunks of this length.
        - is_binned: whether the dataset statistics are provided as intervals with counts (binned)
            or as individual lengths with counts.
        - to_attn_ranges: convert to attn_ranges.
        - drop_thres: whether to drop large sample to generate short-varlen.
    """

    dataset_path = "./benchmark/datasets/default/doc_length_distribution.csv"
    pack_num = 20
    chunk_ratio = 0.25
    is_binned = True
    to_attn_ranges = True
    drop_thres = -1


@dataclass
class DATA_CONFIG:
    """
    Data configuration.
        - seqlen_per_rank: sequence length per rank, total seqlen = seqlen_per_rank * world_size.
        - embed_dim: embedding dimension.
        - hidden_size: hidden size.
        - heads_q: number of query heads.
        - heads_kv: number of key/value heads.
        - dtype: data dtype.
    """

    seqlen_per_rank = 8 * 1024
    embed_dim = 1024
    hidden_size = 128
    heads_q = 64
    heads_kv = 8
    dtype = torch.bfloat16


@dataclass
class ATTN_CONFIG:
    """
    Baseline impl configuration.
        - attn_backend: baseline attention backend to use (FA3, TE)
        - dropout: dropout rate.
        - softmax_scale: softmax scale.
        - deterministic: whether to use deterministic mode.
    MagiAttention impl configuration.
        - chunk_size
        - dispatch_alg
        - OverlapConfig
    """

    # -----    cp baselie dist-attn conf   ---- #
    attn_backend = AttnBackend.FA3
    dropout = 0.0
    softmax_scale = None
    deterministic = False

    # -----    magi-attention conf   ---- #
    chunk_size = 2048
    dispatch_alg = MinHeapDispatchAlg

    enable_overlap = True
    overlap_mode = AttnOverlapMode.STATIC
    degree = 2
    min_chunk_size = 512
    max_num_chunks = 64

    # -----    magi-attention native grpcoll conf   ---- #
    num_sms = 88
    nvl_chunk_size = 8
    nvl_buffer_size = 256
    rdma_chunk_size = 4
    rdma_buffer_size = 128
    num_nvl_bytes = int(3e9)  # ~3GB

    # only valid for internode
    num_rdma_bytes = int(1e9)  # ~1GB
