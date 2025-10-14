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

import argparse
import csv
import random
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import nvtx
import torch

from exps.attn.baselines.utils import (
    calculate_attn_flops,
    generate_seqlens,
    seqlens2curanges,
)
from magi_attention.common.enum import AttnMaskType
from magi_attention.common.ranges import AttnRanges

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
from magi_attention.functional import flex_flash_attn_func as ffa_func
from magi_attention.utils.sparse_utils import (
    flatten_block_mask,
    generate_block_sparse_pattern,
    generate_ranges_from_block_mask,
)


# -----------------------------------------------------------------------------
# Common Helper Functions
# -----------------------------------------------------------------------------
def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seeds set to {seed}.")


def prepare_cuda_events(
    num_events: int = 5,
) -> Tuple[List[torch.cuda.Event], List[torch.cuda.Event]]:
    start_events, end_events = [], []
    for i in range(num_events):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        if i != 0:
            start.record()
            end.record()
        start_events.append(start)
        end_events.append(end)
    return start_events, end_events


def collect_event_timings(
    start_events: List[torch.cuda.Event],
    end_events: List[torch.cuda.Event],
    timings_list: List[List[float]],
):
    for i in range(len(start_events)):
        if start_events[i].cuda_event and end_events[i].cuda_event:
            elapsed_time_ms = start_events[i].elapsed_time(end_events[i])
            timings_list[i].append(elapsed_time_ms)
        else:
            timings_list[i].append(0.0)


def print_performance_results(
    title: str,
    total_run_times: List[float],
    internal_timings: List[List[float]],
    internal_labels: List[str],
    flops: float = 0.0,
):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    total_avg_time_ms = np.mean(total_run_times) if total_run_times else 0.0
    print(f"{'Total Runtime (ms)':<20} | {total_avg_time_ms:.4f}")
    if flops > 0 and total_avg_time_ms > 0:
        tflops_per_sec = (flops / (total_avg_time_ms / 1000)) / 1e12
        print(f"{'Achieved TFLOP/s':<20} | {tflops_per_sec:.4f}")
    print("-" * 60)
    print("Internal Timing Breakdown:")
    for i, label in enumerate(internal_labels):
        if i < len(internal_timings) and internal_timings[i]:
            avg_time = np.mean(internal_timings[i])
            print(f"  - {label:<17} | {avg_time:.4f} ms")
        else:
            print(f"  - {label:<17} | N/A (no data)")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Test-Specific Functions
# -----------------------------------------------------------------------------
# --- Dense / Varlen Specific ---
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


def prepare_dense_ffa_args(
    mask_type: str, seqlen: int, nhq: int, hd: int, device: torch.device, **kwargs
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    sq = sk = seqlen
    causal = "causal" in mask_type
    if "varlen" in mask_type:
        seqlens = generate_seqlens(varlen_seqlen_distribution, seqlen)
        cu_ranges = seqlens2curanges(seqlens)
        q_ranges_ = AttnRanges.from_ranges(cu_ranges)
        k_ranges_ = AttnRanges.from_ranges(cu_ranges)
        attn_flops_dict = calculate_attn_flops(
            q_ranges=q_ranges_,
            k_ranges=k_ranges_,
            attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL]
            * len(cu_ranges),
            total_seqlen_q=sq,
            num_heads_q=nhq,
            head_dim=hd,
        )
        args = {
            "q_ranges": torch.tensor(cu_ranges, dtype=torch.int32, device=device),
            "k_ranges": torch.tensor(cu_ranges, dtype=torch.int32, device=device),
            "max_seqlen_q": max(seqlens),
            "max_seqlen_k": max(seqlens),
            "attn_type_map": torch.ones(
                len(cu_ranges), dtype=torch.int32, device=device
            )
            if causal
            else torch.zeros(len(cu_ranges), dtype=torch.int32, device=device),
        }
    else:
        attn_flops_dict = calculate_attn_flops(
            q_ranges=AttnRanges.from_ranges([[0, sq]]),
            k_ranges=AttnRanges.from_ranges([[0, sk]]),
            attn_mask_type=[AttnMaskType.CAUSAL if causal else AttnMaskType.FULL],
            total_seqlen_q=sq,
            num_heads_q=nhq,
            head_dim=hd,
        )
        args = {
            "q_ranges": torch.tensor([[0, sq]], dtype=torch.int32, device=device),
            "k_ranges": torch.tensor([[0, sk]], dtype=torch.int32, device=device),
            "max_seqlen_q": sq,
            "max_seqlen_k": sk,
            "attn_type_map": torch.tensor(
                [1 if causal else 0], dtype=torch.int32, device=device
            ),
        }
    return args, attn_flops_dict


def generate_dense_qkv(
    seqlen: int,
    nhq: int,
    nhk: int,
    hd: int,
    dtype: torch.dtype,
    device: torch.device,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(seqlen, nhq, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(seqlen, nhk, hd, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(seqlen, nhk, hd, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


# --- Block-Sparse Specific ---
def prepare_block_sparse_ffa_args(
    seqlen: int,
    sparsity_ratio: float,
    block_size: int,
    nhq: int,
    nhk: int,
    device: str,
    **kwargs,
) -> Tuple[Dict[str, Any], None]:
    if seqlen % block_size != 0:
        raise ValueError("Sequence length must be divisible by block_size.")
    num_blocks = seqlen // block_size
    block_mask, _ = generate_block_sparse_pattern(
        num_q_heads=nhq,
        num_kv_heads=nhk,
        num_q_blocks=num_blocks,
        num_kv_blocks=num_blocks,
        sparsity=sparsity_ratio,
        device=device,
    )
    flat_mask = flatten_block_mask(block_mask, nhq, nhk)
    q_ranges, k_ranges = generate_ranges_from_block_mask(
        flat_mask, block_size, block_size
    )
    attn_type_map = torch.zeros(q_ranges.shape[0], dtype=torch.int32, device=device)
    args = {
        "q_ranges": q_ranges,
        "k_ranges": k_ranges,
        "attn_type_map": attn_type_map,
        "max_seqlen_q": block_size,
        "max_seqlen_k": block_size,
        "auto_range_merge": True,
    }
    return args, None


def generate_block_sparse_qkv(
    seqlen: int, nhq: int, nhk: int, hd: int, dtype: torch.dtype, device: str, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q_tokens = seqlen * nhq
    total_kv_tokens = (
        seqlen * nhk
    )  # <<< MODIFIED: Calculate token count for K/V based on nhk

    q = torch.randn(
        total_q_tokens, 1, hd, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        total_kv_tokens, 1, hd, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total_kv_tokens, 1, hd, device=device, dtype=dtype, requires_grad=True
    )

    return q, k, v


fwd_labels = ["range_merge", "Prepare", "Run", "Fill", "to"]
bwd_labels = ["range_merge", "Prepare", "Preprocess", "Run", "to"]


# -----------------------------------------------------------------------------
# Generic Benchmark Framework
# -----------------------------------------------------------------------------
def run_benchmark_framework(
    output_csv_path: str,
    test_name: str,
    configs_to_test: List[Dict[str, Any]],
    csv_header: List[str],
    prepare_args_func: Callable,
    generate_qkv_func: Callable,
    calculate_flops_func: Callable,
    common_params: Dict[str, Any],
):
    """
    A generic benchmark runner that handles all the repetitive testing logic.
    """
    print(f"\nStarting {test_name.upper()} benchmark...")

    # Unpack common parameters
    # nhq, hd = common_params["nhq"], common_params["hd"]
    warmup_iters, run_iters = common_params["warmup_iters"], common_params["run_iters"]
    # device = common_params["device"]

    fwd_labels = ["range_merge", "Prepare", "Run", "Fill", "to"]
    bwd_labels = ["range_merge", "Prepare", "Preprocess", "Run", "to"]

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        for config in configs_to_test:
            print("\n" + "=" * 80)
            print(f"Testing Config: {config}")
            print("=" * 80)

            try:
                # A. Prepare
                current_params = {**common_params, **config}
                ffa_args, flops_meta = prepare_args_func(**current_params)

                q, k, v = generate_qkv_func(**current_params)
                fwd_flops, bwd_flops = calculate_flops_func(
                    config, flops_meta, common_params
                )

                fwd_start_events, fwd_end_events = prepare_cuda_events(len(fwd_labels))
                bwd_start_events, bwd_end_events = prepare_cuda_events(len(bwd_labels))
                bench_fwd_start, bench_fwd_end = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)
                bench_bwd_start, bench_bwd_end = torch.cuda.Event(
                    enable_timing=True
                ), torch.cuda.Event(enable_timing=True)

                ffa_args.update(
                    {
                        "fwd_start_events": fwd_start_events,
                        "fwd_end_events": fwd_end_events,
                        "bwd_start_events": bwd_start_events,
                        "bwd_end_events": bwd_end_events,
                        "disable_fwd_atomic_reduction": True,
                    }
                )
                # Add block_size for sparse tests if available in config
                if "block_size" in config:
                    ffa_args["ref_block_size"] = (
                        config["block_size"],
                        config["block_size"],
                    )

                # B. Forward pass
                fwd_timings: List[List[float]] = [[] for _ in fwd_labels]
                total_fwd_times: List[float] = []
                for _ in range(warmup_iters):
                    out, _ = ffa_func(q, k, v, **ffa_args)
                torch.cuda.synchronize()

                for _ in range(run_iters):
                    bench_fwd_start.record()
                    rng = nvtx.start_range(message="forward_pass")
                    out, _ = ffa_func(q, k, v, **ffa_args)
                    nvtx.end_range(rng)
                    bench_fwd_end.record()
                    torch.cuda.synchronize()
                    total_fwd_times.append(bench_fwd_start.elapsed_time(bench_fwd_end))
                    collect_event_timings(fwd_start_events, fwd_end_events, fwd_timings)

                print_performance_results(
                    "FORWARD PERFORMANCE",
                    total_fwd_times,
                    fwd_timings,
                    fwd_labels,
                    flops=fwd_flops,
                )

                # Write FWD CSV
                avg_total_fwd_ms = np.mean(total_fwd_times)
                tflops = (
                    (fwd_flops / (avg_total_fwd_ms / 1000) / 1e12)
                    if avg_total_fwd_ms > 0
                    else 0
                )
                fwd_row = {
                    **config,
                    "direction": "fwd",
                    "latency_ms": f"{avg_total_fwd_ms:.4f}",
                    "tflops_per_sec": f"{tflops:.4f}",
                }
                for i, label in enumerate(fwd_labels):
                    fwd_row[f"fwd_{label.lower()}_ms"] = (
                        f"{np.mean(fwd_timings[i]):.4f}" if fwd_timings[i] else "0.0000"
                    )
                writer.writerow(fwd_row)

                # C. Backward pass
                do = torch.rand_like(out)

                bwd_timings: List[List[float]] = [[] for _ in fwd_labels]
                total_bwd_times: List[float] = []

                for _ in range(warmup_iters):
                    if q.grad is not None:
                        q.grad.zero_()
                    if k.grad is not None:
                        k.grad.zero_()
                    if v.grad is not None:
                        v.grad.zero_()
                    out.backward(do, retain_graph=True)
                torch.cuda.synchronize()

                for _ in range(run_iters):
                    if q.grad is not None:
                        q.grad.zero_()
                    if k.grad is not None:
                        k.grad.zero_()
                    if v.grad is not None:
                        v.grad.zero_()
                    bench_bwd_start.record()
                    rng = nvtx.start_range(message="backward_pass")
                    out.backward(do, retain_graph=True)
                    nvtx.end_range(rng)
                    torch.cuda.nvtx.range_pop()
                    bench_bwd_end.record()
                    torch.cuda.synchronize()
                    total_bwd_times.append(bench_bwd_start.elapsed_time(bench_bwd_end))
                    collect_event_timings(bwd_start_events, bwd_end_events, bwd_timings)

                print_performance_results(
                    "BACKWARD PERFORMANCE",
                    total_bwd_times,
                    bwd_timings,
                    bwd_labels,
                    flops=bwd_flops,
                )

                # Write BWD CSV
                avg_total_bwd_ms = np.mean(total_bwd_times)
                tflops = (
                    (bwd_flops / (avg_total_bwd_ms / 1000) / 1e12)
                    if avg_total_bwd_ms > 0
                    else 0
                )
                bwd_row = {
                    **config,
                    "direction": "bwd",
                    "latency_ms": f"{avg_total_bwd_ms:.4f}",
                    "tflops_per_sec": f"{tflops:.4f}",
                }
                for i, label in enumerate(bwd_labels):
                    bwd_row[f"bwd_{label.lower()}_ms"] = (
                        f"{np.mean(bwd_timings[i]):.4f}" if bwd_timings[i] else "0.0000"
                    )
                writer.writerow(bwd_row)

            except Exception as e:
                print(f"    ❌ FAILED for config {config}: {e}")
                import traceback

                traceback.print_exc()


# -----------------------------------------------------------------------------
# Test Setup Functions
# -----------------------------------------------------------------------------
def run_dense_tests(args, common_params):
    seqlens_to_test = [8192]
    # mask_types_to_test = ["full", "causal", "varlen_full", "varlen_causal"]
    mask_types_to_test = ["full"]
    configs_to_test = [
        {"seqlen": sl, "mask_type": mt}
        for sl in seqlens_to_test
        for mt in mask_types_to_test
    ]

    def calculate_dense_flops(config, flops_meta, params):
        return flops_meta.get("fwd", 0), flops_meta.get("bwd", 0)

    config_keys = list(configs_to_test[0].keys())
    standard_keys = ["direction", "latency_ms", "tflops_per_sec"]
    fwd_timing_keys = [f"fwd_{label.lower()}_ms" for label in fwd_labels]
    bwd_timing_keys = [f"bwd_{label.lower()}_ms" for label in bwd_labels]
    csv_header = config_keys + standard_keys + fwd_timing_keys + bwd_timing_keys

    run_benchmark_framework(
        output_csv_path=args.output_csv_path,
        test_name="Dense",
        configs_to_test=configs_to_test,
        csv_header=csv_header,
        prepare_args_func=prepare_dense_ffa_args,
        generate_qkv_func=generate_dense_qkv,
        calculate_flops_func=calculate_dense_flops,
        common_params=common_params,
    )


def run_block_sparse_tests(args, common_params):
    seqlens_to_test = [16384, 49152]
    sparsity_ratios_to_test = [0.2, 0.5, 0.8]
    block_sizes_to_test = [128]
    configs_to_test = [
        {"seqlen": sl, "sparsity_ratio": sr, "block_size": bs}
        for sl in seqlens_to_test
        for sr in sparsity_ratios_to_test
        for bs in block_sizes_to_test
        if sl % bs == 0
    ]

    config_keys = list(configs_to_test[0].keys())
    standard_keys = ["direction", "latency_ms", "tflops_per_sec"]
    fwd_timing_keys = [f"fwd_{label.lower()}_ms" for label in fwd_labels]
    bwd_timing_keys = [f"bwd_{label.lower()}_ms" for label in bwd_labels]
    csv_header = config_keys + standard_keys + fwd_timing_keys + bwd_timing_keys

    def calculate_sparse_flops(config, flops_meta, params):
        fwd_flops = (
            4
            * params["nhq"]
            * config["seqlen"] ** 2
            * params["hd"]
            * config["sparsity_ratio"]
        )
        bwd_flops = fwd_flops * 2.5
        return fwd_flops, bwd_flops

    run_benchmark_framework(
        output_csv_path=args.output_csv_path,
        test_name="Block-Sparse",
        configs_to_test=configs_to_test,
        csv_header=csv_header,
        prepare_args_func=prepare_block_sparse_ffa_args,
        generate_qkv_func=generate_block_sparse_qkv,
        calculate_flops_func=calculate_sparse_flops,
        common_params=common_params,
    )


# -----------------------------------------------------------------------------
# Main Execution Logic
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run unified performance benchmark for FlexFlash Attention.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test_type",
        type=str,
        required=True,
        choices=["dense", "block_sparse"],
        help="Type of benchmark to run.\n  -"
        " 'dense': Tests full, causal, and varlen attention masks.\n  - 'block_sparse': Tests block-sparse attention masks.",
    )
    parser.add_argument(
        "-o",
        "--output_csv_path",
        type=str,
        default="performance_results.csv",
        help="Path to the output CSV file for storing results.",
    )
    args = parser.parse_args()

    set_seeds(42)

    if not torch.cuda.is_available():
        print(
            "⚠️ WARNING: CUDA is not available. Running on CPU, results will not be meaningful."
        )

    # Define common parameters for all tests
    common_params = {
        "nhq": 64,
        "nhk": 8,  # Used by dense QKV
        "hd": 128,
        "dtype": torch.bfloat16,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "warmup_iters": 100,
        "run_iters": 100,
    }

    if args.test_type == "dense":
        run_dense_tests(args, common_params)
    elif args.test_type == "block_sparse":
        run_block_sparse_tests(args, common_params)

    print(
        f"\n✅ All tests complete. Results have been saved to '{args.output_csv_path}'"
    )
