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

import inspect
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist

# -------------------       sentinel values     ------------------- #

# These sentinels are used in perf_dict values to signal special conditions
# that the plotting code in Mark.draw_plot() recognises and annotates.
BENCH_CASE_OOM: int = -1  # kernel ran out of GPU memory
BENCH_CASE_NOT_SUPPORTED: int = (
    -2
)  # combination is not supported on this hardware/backend

# -------------------       visual style constants     ------------------- #

# Warm-toned custom palette (RGB 0-255).  Up to 8 distinct colours; seaborn
# "tab10" fills in for any additional series beyond 8.
_PALETTE_RGB_255 = [
    [196, 86, 126],
    [216, 130, 122],
    [255, 188, 167],
    [237, 216, 163],
    [145, 196, 164],
    [74, 145, 154],
    [61, 119, 180],
    [56, 95, 122],
]


def build_color_palette(n: int) -> list:
    """Return *n* colours: warm custom palette first, seaborn tab10 overflow."""
    base = [tuple(c / 255 for c in rgb) for rgb in _PALETTE_RGB_255]
    if n <= len(base):
        return base[:n]
    extra = sns.color_palette("tab10", n_colors=n - len(base))
    return base + list(extra)


# -------------------       benchmark utils     ------------------- #


@dataclass
class StatsConfig:
    """Configuration for the statistics summary written by ``report_all_from_perf``.

    Add new fields here whenever you want to extend the stats output — no need
    to touch the call chain anywhere else.

    Attributes:
        best_select: ``"max"`` to report the legend with the highest value per
            x-row, ``"min"`` for the lowest.  Sentinel values
            ``BENCH_CASE_OOM`` and ``BENCH_CASE_NOT_SUPPORTED`` are always
            excluded from selection.
        n_x_cols:   Number of leading CSV columns that together form the x-axis
            key (default 1).  Use a larger value when a ``Benchmark`` has
            multiple ``x_names``.
    """

    best_select: str = "max"
    n_x_cols: int = 1

    def __post_init__(self):
        if self.best_select not in ("max", "min"):
            raise ValueError(
                f"StatsConfig.best_select must be 'max' or 'min', got {self.best_select!r}"
            )


class TimeManager:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        print(f"elapsed time: {self.elapsed_time:.3e}")


MemRecordMode: TypeAlias = Literal["allocated", "peak"]


class MemRecorder:
    """Records GPU memory around a code block
    using the PyTorch CUDA caching allocator.

    Modes:
        - ``"peak"`` *(default)*: reset the allocator's peak-memory counter on
          entry and report ``torch.cuda.max_memory_allocated()`` on exit — the
          peak allocated bytes reached inside the block.
        - ``"allocated"``: report the currently allocated bytes
          (``torch.cuda.memory_allocated()``) on exit.

    Both paths are pure Python/torch calls with negligible host overhead, so
    the recorder is safe to place inside a timing window.
    """

    def __init__(self, mode: MemRecordMode = "peak", device_idx: int = 0) -> None:
        self.memory = None
        self.mode = mode
        self.device_idx = device_idx

    def __enter__(self):
        if self.mode == "peak":
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "peak":
            self.memory = torch.cuda.max_memory_allocated()
        elif self.mode == "allocated":
            self.memory = torch.cuda.memory_allocated()


def maybe_dist_sync():
    """Sync before each sweep for distributed bench"""
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()


def get_timestamp():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


def get_device_name():
    device_name = torch.cuda.get_device_name()
    # search for the registered name patterns
    register_names = {
        ("3090",): "3090",
        ("3090ti", "3090Ti"): "3090ti",
        ("4090ti", "4090Ti"): "4090ti",
        ("4090",): "4090",
        ("A6000",): "A6000",
        ("A100",): "A100",
        ("A800",): "A800",
        ("H100",): "H100",
        ("H800",): "H800",
        ("V100",): "V100",
        ("T4",): "T4",
        ("P4",): "P4",
        ("P40",): "P40",
    }
    for ks, v in register_names.items():
        for k in ks:
            if k in device_name:
                return v

    # not regitered, default use the last word in the device name
    device_name_match = re.search(r"(\w+)$", device_name)
    if device_name_match:
        device_name = device_name_match.group(1)

    return device_name


def gen_save_path(
    name: str,
    out_dir: str = "outs",
    add_timestamp_suffix: bool = True,
    print_save_path: bool = True,
) -> str:
    """Return an output directory path rooted at the caller's script location.

    The path has the form ``<script_dir>/<out_dir>/<name>[_<timestamp>]``,
    mirroring the pattern used in every benchmark / pretune entry-point.

    Args:
        name:                 Base name for the output directory (e.g. ``"bench_flash_mh_moe"``).
        out_dir:              Subdirectory relative to the caller's script to place outputs in.
                                Defaults to ``"outs"``.
        add_timestamp_suffix: Whether to append a ``_<timestamp>`` suffix to the directory name.
                                Defaults to ``True``.
        print_save_path:      Whether to print the generated save path to stdout.
                                Defaults to ``True``.

    Returns:
        Absolute path string suitable for passing to ``Mark.run(save_path=...)``.
    """
    caller_file = inspect.stack()[1].filename
    script_dir = os.path.dirname(os.path.abspath(caller_file))
    if add_timestamp_suffix:
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
        dir_name = f"{name}_{timestamp}"
    else:
        dir_name = name
    save_path = os.path.join(script_dir, out_dir, dir_name)
    if print_save_path:
        print(f"[{name}] Saving results to: {save_path}")
    return save_path


def collect_best_legend(
    save_root: str,
    cfg: StatsConfig,
) -> dict:
    """Walk save_root sub-folders, read every CSV and return the legend with the
    best (highest or lowest) value per x-row, skipping sentinel values.

    Returns a nested dict:
        {folder_name: {perf_key: {x_key: {"legend": str, "value": float} | None}}}
    """
    _SPECIAL = frozenset({float(BENCH_CASE_OOM), float(BENCH_CASE_NOT_SUPPORTED)})
    result: dict = {}

    for folder_name in sorted(os.listdir(save_root)):
        folder_path = os.path.join(save_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        csv_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".csv"))
        if not csv_files:
            continue

        folder_result: dict = {}
        for csv_file in csv_files:
            perf_key = os.path.splitext(csv_file)[0]
            df = pd.read_csv(os.path.join(folder_path, csv_file))

            if df.shape[1] <= cfg.n_x_cols:
                continue  # no legend columns present

            x_cols = list(df.columns[: cfg.n_x_cols])
            legend_cols = list(df.columns[cfg.n_x_cols :])

            perf_result: dict = {}
            for _, row in df.iterrows():
                # Build a JSON-safe string key from the x column(s)
                x_key = (
                    str(row[x_cols[0]])
                    if cfg.n_x_cols == 1
                    else "_".join(str(row[c]) for c in x_cols)
                )

                best_legend: str | None = None
                best_val: float | None = None
                for col in legend_cols:
                    try:
                        val = float(row[col])
                    except (ValueError, TypeError):
                        continue
                    if np.isnan(val) or val in _SPECIAL:
                        continue
                    if (
                        best_val is None
                        or (cfg.best_select == "max" and val > best_val)
                        or (cfg.best_select == "min" and val < best_val)
                    ):
                        best_val = val
                        best_legend = col

                perf_result[x_key] = (
                    {"legend": best_legend, "value": best_val}
                    if best_legend is not None
                    else None
                )

            folder_result[perf_key] = perf_result

        if folder_result:
            result[folder_name] = folder_result

    return result
