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

"""Benchmark the CuTe DSL radix-select and streaming top-k against ``torch.topk``.

All backends return indices only; timing is measured via CUDA events.

Usage examples::

    # Single config (default: rows=4096, n=600000, k=10000)
    python exps/topk/bench_topk.py --dtype bf16 --warmup 30 --iters 500

    # Preset configs
    python exps/topk/bench_topk.py --preset both --dtype bf16 --warmup 30 --iters 500

    # Batch test: sweep n x k grid, print per-backend tables
    python exps/topk/bench_topk.py --batch-test --warmup 30 --iters 500 --dtype bf16

    # Batch test with custom params
    python exps/topk/bench_topk.py --batch-test --batch-rows 2048 \
        --batch-n 160000 320000 640000 --batch-k 10000 20000 40000 80000 \
        --warmup 30 --iters 500 --dtype bf16 --backends torch cutedsl
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import torch

from magi_attention.kernel.cutedsl.topk import cutedsl_topk
from magi_attention.kernel.cutedsl.topk_stream import cutedsl_topk_stream

# ---------------------------------------------------------------------------
# Backend dispatch helpers (all return indices only)
# ---------------------------------------------------------------------------


def _torch_topk_fn(x: torch.Tensor, k: int, *, sorted: bool) -> torch.Tensor:
    return torch.topk(x, k, dim=-1, sorted=sorted).indices


def _cutedsl_topk_fn(x: torch.Tensor, k: int, *, sorted: bool) -> torch.Tensor:
    return cutedsl_topk(x, k, sorted=sorted)


def _cutedsl_stream_topk_fn(x: torch.Tensor, k: int, *, sorted: bool) -> torch.Tensor:
    return cutedsl_topk_stream(x, k, sorted=sorted)


BACKENDS = {
    "torch": _torch_topk_fn,
    "cutedsl": _cutedsl_topk_fn,
    "cutedsl_stream": _cutedsl_stream_topk_fn,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def _bytes_to_mib(nbytes: int | float) -> float:
    return float(nbytes) / 1024**2


def _bytes_to_gib(nbytes: int | float) -> float:
    return float(nbytes) / 1024**3


# ---------------------------------------------------------------------------
# Per-backend benchmark
# ---------------------------------------------------------------------------


@torch.no_grad()
def _bench_backend(
    backend_name: str,
    fn,
    x: torch.Tensor,
    k: int,
    *,
    sorted: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    print(f"\n--- backend: {backend_name} ---")

    try:
        for _ in range(warmup):
            out_idx = fn(x, k, sorted=sorted)
        torch.cuda.synchronize(device)
    except Exception as exc:
        print(f"  SKIP: {type(exc).__name__}: {exc}")
        return None
    del out_idx
    torch.cuda.empty_cache()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out_idx = fn(x, k, sorted=sorted)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms_total = start.elapsed_time(end)
    elapsed_ms = elapsed_ms_total / iters

    input_bytes = x.numel() * x.element_size()
    output_bytes = out_idx.numel() * out_idx.element_size()
    io_bytes = input_bytes + output_bytes
    throughput_gib_s = _bytes_to_gib(io_bytes) / (elapsed_ms / 1_000.0)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out_idx = fn(x, k, sorted=sorted)
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    out_bytes = out_idx.numel() * out_idx.element_size()
    peak_extra_bytes = peak_allocated - baseline_allocated
    estimated_temp_bytes = max(0, peak_extra_bytes - out_bytes)

    stats = {
        "backend": backend_name,
        "elapsed_ms": elapsed_ms,
        "elapsed_ms_total": elapsed_ms_total,
        "throughput_gib_s": throughput_gib_s,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "baseline_allocated": baseline_allocated,
        "peak_extra_bytes": peak_extra_bytes,
        "estimated_temp_bytes": estimated_temp_bytes,
        "peak_allocated": peak_allocated,
        "output_shape": tuple(out_idx.shape),
        "output_dtype": str(out_idx.dtype),
    }

    print(
        f"time: iters={iters}, mean={elapsed_ms:.3f} ms, "
        f"total={elapsed_ms_total:.3f} ms"
    )
    print(
        f"io: input={_bytes_to_mib(input_bytes):.3f} MiB, "
        f"output={_bytes_to_mib(output_bytes):.3f} MiB, "
        f"throughput={throughput_gib_s:.3f} GiB/s"
    )
    print(
        "memory: "
        f"baseline_allocated={_bytes_to_mib(baseline_allocated):.3f} MiB, "
        f"peak_extra={_bytes_to_mib(peak_extra_bytes):.3f} MiB, "
        f"estimated_temp={_bytes_to_mib(estimated_temp_bytes):.3f} MiB, "
        f"peak_allocated={_bytes_to_mib(peak_allocated):.3f} MiB"
    )
    print(f"output: shape={tuple(out_idx.shape)}, dtype={out_idx.dtype}")

    return stats


# ---------------------------------------------------------------------------
# Batch test: sweep n x k grid per backend, print time table
# ---------------------------------------------------------------------------


@torch.no_grad()
def _bench_time_only(
    fn,
    x: torch.Tensor,
    k: int,
    *,
    sorted: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> Optional[float]:
    """Return mean elapsed ms, or None if the backend fails."""
    try:
        for _ in range(warmup):
            fn(x, k, sorted=sorted)
        torch.cuda.synchronize(device)
    except Exception:
        return None

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x, k, sorted=sorted)
    end.record()
    torch.cuda.synchronize(device)
    return start.elapsed_time(end) / iters


def _run_batch_test(
    backends: list[str],
    rows: int,
    n_list: list[int],
    k_list: list[int],
    *,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    device: torch.device,
    seed: int,
) -> None:
    """Sweep every (n, k) pair for each backend and print a table."""

    print("=" * 70)
    print(f"BATCH TEST  rows={rows}  dtype={dtype}  warmup={warmup}  iters={iters}")
    print(f"n  = {n_list}")
    print(f"k  = {k_list}")
    print(f"device: {torch.cuda.get_device_name(device)}")
    print("=" * 70)

    # results[backend][(n, k)] = elapsed_ms or None
    results: dict[str, dict[tuple[int, int], Optional[float]]] = {
        b: {} for b in backends
    }

    for n in n_list:
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        x = torch.randn((rows, n), dtype=dtype, device=device)
        torch.cuda.synchronize(device)

        for k in k_list:
            if k > n:
                for b in backends:
                    results[b][(n, k)] = None
                continue

            tag = f"[rows={rows}, n={n}, k={k}]"
            for b in backends:
                fn = BACKENDS[b]
                ms = _bench_time_only(
                    fn,
                    x,
                    k,
                    sorted=False,
                    warmup=warmup,
                    iters=iters,
                    device=device,
                )
                results[b][(n, k)] = ms
                status = f"{ms:.3f} ms" if ms is not None else "SKIP"
                print(f"  {tag} {b}: {status}")

        del x
        torch.cuda.empty_cache()

    elem_size = torch.tensor([], dtype=dtype).element_size()

    def _compute_bw(n: int, k: int, ms: float) -> float:
        """IO bandwidth in GiB/s: (input + output) bytes / time."""
        input_bytes = rows * n * elem_size
        output_bytes = rows * k * 4  # indices are int32/int64 → 4 bytes typical
        return _bytes_to_gib(input_bytes + output_bytes) / (ms / 1_000.0)

    def _cell(n: int, k: int, ms: Optional[float]) -> str:
        if ms is None:
            return "N/A"
        bw = _compute_bw(n, k, ms)
        return f"{ms:.3f}ms/{bw:.1f}"

    # pretty-print one table per backend
    k_labels = [f"k={k}" for k in k_list]
    col_w = max(18, max(len(lab) for lab in k_labels) + 2)
    header = "n \\\\ k".ljust(16) + "".join(lab.rjust(col_w) for lab in k_labels)

    for b in backends:
        print(f"\n{'=' * 70}")
        print(f"  {b}  (ms / GiB/s, rows={rows}, warmup={warmup}, iters={iters})")
        print("=" * 70)
        print(header)
        print("-" * len(header))
        for n in n_list:
            row_label = f"n={n}"
            cells = []
            for k in k_list:
                ms = results[b].get((n, k))
                cells.append(_cell(n, k, ms))
            print(f"{row_label:<16}" + "".join(c.rjust(col_w) for c in cells))
        print("-" * len(header))

    # cross-backend comparison: speedup of each later backend over each earlier one
    pairs = [
        (base, new)
        for base, new in (
            ("torch", "cutedsl"),
            ("torch", "cutedsl_stream"),
            ("cutedsl", "cutedsl_stream"),
        )
        if base in backends and new in backends
    ]
    for base, new in pairs:
        print(f"\n{'=' * 70}")
        print(f"  speedup: {base} / {new}  (>1 means {new} is faster)")
        print("=" * 70)
        print(header)
        print("-" * len(header))
        for n in n_list:
            row_label = f"n={n}"
            cells = []
            for k in k_list:
                base_ms = results[base].get((n, k))
                new_ms = results[new].get((n, k))
                if base_ms is not None and new_ms is not None and new_ms > 0:
                    cells.append(f"{base_ms / new_ms:.2f}x")
                else:
                    cells.append("N/A")
            print(f"{row_label:<16}" + "".join(c.rjust(col_w) for c in cells))
        print("-" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark top-k: torch.topk (baseline) vs CuTe DSL radix select."
    )
    parser.add_argument("--n", type=int, default=None, help="last-dim input size")
    parser.add_argument("--k", type=int, default=None, help="top-k size")
    parser.add_argument(
        "--rows", type=int, default=None, help="leading rows; shape is [rows, n]"
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=("dsa1", "dsa2", "both"),
        help="preset configs: dsa1=4096x600000/10000, dsa2=1080x128520/10240, both=run both",
    )
    parser.add_argument(
        "--one-dim", action="store_true", help="use shape [n] instead of [rows, n]"
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["torch", "cutedsl", "cutedsl_stream"],
        choices=list(BACKENDS.keys()),
        help="backends to benchmark (default: torch cutedsl cutedsl_stream)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--batch-test",
        action="store_true",
        help="run batch sweep over n_list x k_list grid and print per-backend tables",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=2048,
        help="rows for batch test (default: 2048)",
    )
    parser.add_argument(
        "--batch-n",
        type=int,
        nargs="+",
        default=[160_000, 320_000, 640_000],
        help="n values for batch test (default: 160000 320000 640000)",
    )
    parser.add_argument(
        "--batch-k",
        type=int,
        nargs="+",
        default=[10_000, 20_000, 40_000, 80_000],
        help="k values for batch test (default: 10000 20000 40000 80000)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype_from_name(args.dtype)

    if args.batch_test:
        _run_batch_test(
            backends=args.backends,
            rows=args.batch_rows,
            n_list=args.batch_n,
            k_list=args.batch_k,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            device=device,
            seed=args.seed,
        )
        return 0

    PRESETS = {
        "dsa1": (4096, 600_000, 10_000),
        "dsa2": (1080, 128_520, 10_240),
    }

    configs: list[tuple[int, int, int]] = []
    if args.preset == "both":
        configs = [PRESETS["dsa1"], PRESETS["dsa2"]]
    elif args.preset is not None:
        configs = [PRESETS[args.preset]]
    else:
        rows = args.rows or 4096
        n = args.n or 600_000
        k = args.k or 10_000
        configs = [(rows, n, k)]

    for cfg_idx, (rows, n, k) in enumerate(configs):
        if k > n:
            raise SystemExit(f"k must be <= n, got k={k}, n={n}")

        torch.manual_seed(args.seed)
        torch.cuda.empty_cache()
        shape = (n,) if args.one_dim else (rows, n)
        x = torch.randn(shape, dtype=dtype, device=device)
        torch.cuda.synchronize(device)

        if cfg_idx > 0:
            print("\n\n" + "#" * 70)
        print("=== top-k benchmark ===")
        print(f"case: input_shape={tuple(x.shape)}, k={k}, dtype={x.dtype}")
        print(f"device: {torch.cuda.get_device_name(device)}")

        all_stats: list[dict[str, Any]] = []
        for backend_name in args.backends:
            fn = BACKENDS[backend_name]
            stats = _bench_backend(
                backend_name,
                fn,
                x,
                k,
                sorted=False,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
            )
            if stats is not None:
                all_stats.append(stats)

        if len(all_stats) > 1:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"{'backend':<15} {'mean (ms)':>10} {'GiB/s':>10} {'temp MiB':>10}")
            print("-" * 60)
            best_ms = min(s["elapsed_ms"] for s in all_stats)
            for s in all_stats:
                ms = s["elapsed_ms"]
                marker = " <-- best" if ms == best_ms else ""
                print(
                    f"{s['backend']:<15} {ms:>10.3f} {s['throughput_gib_s']:>10.3f} "
                    f"{_bytes_to_mib(s['estimated_temp_bytes']):>10.3f}{marker}"
                )
            print("-" * 60)

            if best_ms > 0:
                for s in all_stats:
                    if s["elapsed_ms"] != best_ms:
                        slowdown = s["elapsed_ms"] / best_ms
                        print(
                            f"  {s['backend']} is {slowdown:.2f}x slower than "
                            f"{[ss['backend'] for ss in all_stats if ss['elapsed_ms'] == best_ms][0]}"
                        )

        del x
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
