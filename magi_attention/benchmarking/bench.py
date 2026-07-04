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


import gc
import json
import os
import sys
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
from matplotlib import patheffects as pe
from tqdm import tqdm

from .image_grid import make_img_grid
from .utils import (
    BENCH_CASE_NOT_SUPPORTED,
    BENCH_CASE_OOM,
    MemRecorder,
    StatsConfig,
    build_color_palette,
    collect_best_legend,
    maybe_dist_sync,
)

# -------------------       benchmark wrapper     ------------------- #


# copied and modified from triton.testing.do_bench to add flops report with peak memory report
# see https://github.com/openai/triton/blob/ccc25eb0d6261587a61b8ce8cff6ff1ad1d579fd/python/triton/testing.py#L79
def do_bench(
    fn: Callable,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    grad_to_none: Optional[list] = None,
    quantiles: Optional[list[float]] = None,
    fast_flush: bool = True,
    return_mode: str = "mean",
    return_flops: bool = True,
    return_mem: bool = True,
    mem_record_mode: str = "allocated",
    device_idx: int = 0,
    to_gc_collect: bool = True,
    to_empty_cache: bool = True,
) -> dict:
    """Benchmark the FLOPS and/or peak memory of the provided function.

    Args:
        fn: The zero-argument callable to benchmark.
        warmup_iters: Number of warm-up iterations run before timing starts.
            Defaults to ``25``.
        rep_iters: Number of timed measurement iterations.
            Defaults to ``100``.
        grad_to_none: List of tensors whose ``.grad`` should be set to
            ``None`` before each iteration, preventing gradient accumulation
            across benchmark iterations.
        quantiles: Percentiles (in ``[0, 1]``) at which to report results.
            When set, the return values are lists of per-quantile scalars
            instead of a single aggregated scalar.
        fast_flush: When ``True``, use a compact int32 buffer to flush the
            L2 cache between iterations (faster). When ``False``, use a
            larger int8 buffer. Defaults to ``True``.
        return_mode: Aggregation mode used when ``quantiles`` is ``None``.
            One of ``"min"``, ``"max"``, ``"mean"``, ``"median"``.
            Defaults to ``"mean"``.
        return_flops: Include a ``"flops"`` key (timing in ms) in the
            returned dict. Defaults to ``True``.
        return_mem: Include a ``"mem"`` key (peak allocated memory in bytes)
            in the returned dict. Defaults to ``True``.
        mem_record_mode: Memory recording mode forwarded to
            :class:`MemRecorder`. Defaults to ``"allocated"``.
        device_idx: CUDA device index used for memory recording.
            Defaults to ``0``.
        to_gc_collect: Call ``gc.collect()`` after the run to free Python
            objects promptly. Defaults to ``True``.
        to_empty_cache: Call ``torch.cuda.empty_cache()`` after the run.
            Defaults to ``True``.

    Returns:
        A dict containing one or both of:

        - ``"flops"``: timing in milliseconds (scalar or per-quantile list).
        - ``"mem"``: peak allocated memory in bytes (scalar or per-quantile
          list).

        Exactly which keys are present depends on ``return_flops`` and
        ``return_mem``.
    """
    assert return_mode in ["min", "max", "mean", "median"]
    assert return_flops or return_mem

    def _get_ret(flops, mem):
        return (
            dict(flops=flops, mem=mem)
            if return_flops and return_mem
            else (dict(flops=flops) if return_flops else dict(mem=mem))
        )

    def _get_item(ret):
        return ret[0] if len(ret) == 1 else ret

    fn()
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")

    # Init events and memory buffer for recording per-iteration stats
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    mems = [0.0] * rep_iters

    # Warm-up
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(warmup_iters):
        fn()
    torch.cuda.nvtx.range_pop()

    # Synchronize before starting benchmark
    if dist.is_initialized():
        dist.all_reduce(cache, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    # Benchmark
    for i in range(rep_iters):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()

        # HACK: get attn-impl and workload names for fn with iters as profile range
        profile_range = getattr(fn, "profile_range", "") + f"_iter{i}"

        # barrier before starting timing
        if dist.is_initialized():
            dist.all_reduce(cache, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        torch.cuda.nvtx.range_push(profile_range)

        start_event[i].record()
        # record mem of `fn`
        with MemRecorder(mode=mem_record_mode, device_idx=device_idx) as recoder:
            fn()

        mems[i] = recoder.memory
        end_event[i].record()
        torch.cuda.nvtx.range_pop()

    # Synchronize before recording clocks
    if dist.is_initialized():
        dist.all_reduce(cache, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    # Record clocks across different runs
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
        device=torch.device("cuda"),
    )

    # Reduce clocks across ranks (worst-case) when in distributed mode
    if dist.is_initialized():
        dist.all_reduce(times, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    times = times.to(device=torch.device("cpu"))
    mems = torch.tensor(mems, dtype=torch.float)

    # Clean up
    if to_empty_cache:
        torch.cuda.empty_cache()
    if to_gc_collect:
        gc.collect()

    # Get quantiles
    if quantiles is not None:
        ret_flops = _get_item(
            torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        )
        ret_mem = _get_item(
            torch.quantile(mems, torch.tensor(quantiles, dtype=torch.float)).tolist()
        )
        return _get_ret(ret_flops, ret_mem)

    # Return aggregated scalar
    return _get_ret(
        getattr(torch, return_mode)(times).item(),
        getattr(torch, return_mode)(mems).item(),
    )


def do_bench_flops(
    fn: Callable,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    grad_to_none: Optional[list] = None,
    quantiles: Optional[list[float]] = None,
    fast_flush: bool = True,
    return_mode: str = "mean",
    mem_record_mode: str = "allocated",
    device_idx: int = 0,
    to_gc_collect: bool = True,
    to_empty_cache: bool = True,
) -> dict:
    """Benchmark the FLOPS of the provided function (memory recording disabled).

    Thin wrapper around :func:`do_bench` with ``return_flops=True`` and
    ``return_mem=False`` fixed.  All other parameters are forwarded verbatim.

    Args:
        fn (Callable): Function to benchmark.
        warmup_iters (int): Number of warm-up iterations.
            Defaults to ``25``.
        rep_iters (int): Number of measurement iterations.
            Defaults to ``100``.
        grad_to_none (list[torch.Tensor], optional): Tensors whose ``.grad``
            should be set to ``None`` before each iteration.
        quantiles (list[float], optional): Percentiles to return. When set,
            the return value is a list of per-quantile timings instead of a
            single scalar.
        fast_flush (bool): Use a faster (int32) L2-flush buffer. Defaults to
            ``True``.
        return_mode (str): Aggregation mode when ``quantiles`` is ``None``.
            One of ``"min"``, ``"max"``, ``"mean"``, ``"median"``.
            Defaults to ``"mean"``.
        mem_record_mode (str): Memory recording mode passed to
            :class:`MemRecorder`. Defaults to ``"allocated"``.
        device_idx (int): CUDA device index for memory recording.
            Defaults to ``0``.
        to_gc_collect (bool): Call ``gc.collect()`` after the run.
            Defaults to ``True``.
        to_empty_cache (bool): Call ``torch.cuda.empty_cache()`` after the
            run. Defaults to ``True``.

    Returns:
        dict: ``{"flops": <timing_ms>}`` — a scalar when ``quantiles`` is
        ``None``, or a list of per-quantile values otherwise.
    """
    return do_bench(
        fn,
        warmup_iters=warmup_iters,
        rep_iters=rep_iters,
        grad_to_none=grad_to_none,
        quantiles=quantiles,
        fast_flush=fast_flush,
        return_mode=return_mode,
        return_flops=True,
        return_mem=False,
        mem_record_mode=mem_record_mode,
        device_idx=device_idx,
        to_gc_collect=to_gc_collect,
        to_empty_cache=to_empty_cache,
    )


def do_bench_mem(
    fn: Callable,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    grad_to_none: Optional[list] = None,
    quantiles: Optional[list[float]] = None,
    fast_flush: bool = True,
    return_mode: str = "mean",
    mem_record_mode: str = "allocated",
    device_idx: int = 0,
    to_gc_collect: bool = True,
    to_empty_cache: bool = True,
) -> dict:
    """Benchmark the peak memory of the provided function (FLOPS recording disabled).

    Thin wrapper around :func:`do_bench` with ``return_flops=False`` and
    ``return_mem=True`` fixed.  All other parameters are forwarded verbatim.

    Args:
        fn (Callable): Function to benchmark.
        warmup_iters (int): Number of warm-up iterations.
            Defaults to ``25``.
        rep_iters (int): Number of measurement iterations.
            Defaults to ``100``.
        grad_to_none (list[torch.Tensor], optional): Tensors whose ``.grad``
            should be set to ``None`` before each iteration.
        quantiles (list[float], optional): Percentiles to return. When set,
            the return value is a list of per-quantile memory values instead
            of a single scalar.
        fast_flush (bool): Use a faster (int32) L2-flush buffer. Defaults to
            ``True``.
        return_mode (str): Aggregation mode when ``quantiles`` is ``None``.
            One of ``"min"``, ``"max"``, ``"mean"``, ``"median"``.
            Defaults to ``"mean"``.
        mem_record_mode (str): Memory recording mode passed to
            :class:`MemRecorder`. Defaults to ``"allocated"``.
        device_idx (int): CUDA device index for memory recording.
            Defaults to ``0``.
        to_gc_collect (bool): Call ``gc.collect()`` after the run.
            Defaults to ``True``.
        to_empty_cache (bool): Call ``torch.cuda.empty_cache()`` after the
            run. Defaults to ``True``.

    Returns:
        dict: ``{"mem": <peak_memory_bytes>}`` — a scalar when ``quantiles``
        is ``None``, or a list of per-quantile values otherwise.
    """
    return do_bench(
        fn,
        warmup_iters=warmup_iters,
        rep_iters=rep_iters,
        grad_to_none=grad_to_none,
        quantiles=quantiles,
        fast_flush=fast_flush,
        return_mode=return_mode,
        return_flops=False,
        return_mem=True,
        mem_record_mode=mem_record_mode,
        device_idx=device_idx,
        to_gc_collect=to_gc_collect,
        to_empty_cache=to_empty_cache,
    )


# copied from triton.testing.Benchmark
# see https://github.com/openai/triton/blob/ccc25eb0d6261587a61b8ce8cff6ff1ad1d579fd/python/triton/testing.py#L192
class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = "",
        ylabel: str | dict[str, str] = "",
        x_log: bool = False,
        y_log: bool = False,
        styles=None,
        dir_name: str | None = None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles

        # plot info
        self.xlabel = xlabel or x_names[0]
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args
        self.dir_name = dir_name

    @property
    def folder_name(self) -> str:
        """Directory name to use for saving outputs.  Falls back to ``plot_name``."""
        return self.dir_name if self.dir_name is not None else self.plot_name

    @staticmethod
    def from_csv(
        csv_path: str,
        plot_name: str,
        line_arg: str,
        ylabel: str | dict[str, str] = "",
        x_int: bool = False,
        x_log: bool = False,
    ) -> "Benchmark":
        df = pd.read_csv(csv_path)

        x_names = [df.columns[0]]
        x_vals = df[x_names[0]].unique().tolist()
        if x_int:
            x_vals = [int(x) for x in x_vals]
        line_vals = line_names = df.columns[1:].tolist()

        return Benchmark(
            x_names=x_names,
            x_vals=x_vals,
            line_arg=line_arg,
            line_vals=line_vals,
            line_names=line_names,
            plot_name=plot_name,
            args={},
            x_log=x_log,
            ylabel=ylabel,
        )


# copied and modified from triton.testing.Mark to add flops report with peak memory report
# see https://github.com/openai/triton/blob/ccc25eb0d6261587a61b8ce8cff6ff1ad1d579fd/python/triton/testing.py#L258
class Mark:
    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    def _call(self, bench: Benchmark, **kwargs):
        x_names = list(bench.x_names)
        y_mean = bench.line_names

        # y_min = [f"{x}-min" for x in bench.line_names]
        # y_max = [f"{x}-max" for x in bench.line_names]

        df_init = pd.DataFrame(columns=x_names + y_mean)
        # df_init = pd.DataFrame(columns=x_names + y_mean + y_min + y_max)

        dfs = {}
        for x in bench.x_vals:
            maybe_dist_sync()
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_mean: dict[str, list] = {}
            # row_min: dict[str, list] = {}
            # row_max: dict[str, list] = {}
            for y in bench.line_vals:
                ret_dict = self.fn(
                    **x_args, **{bench.line_arg: y}, **bench.args, **kwargs
                )
                for k, v in ret_dict.items():
                    try:
                        y_mean, _, _ = v
                        # y_mean, y_min, y_max = v
                    except ValueError:
                        try:
                            y_mean = v[0]
                        except TypeError:
                            y_mean = v
                        # y_mean, y_min, y_max = v, None, None  # type: ignore
                    row_mean.setdefault(k, []).append(y_mean)
                    # row_min.setdefault(k, []).append(y_min)
                    # row_max.setdefault(k, []).append(y_max)
            for k in row_mean:
                if k not in dfs:
                    dfs[k] = deepcopy(df_init)
                dfs[k].loc[len(dfs[k])] = list(x) + row_mean[k]
                # dfs[k].loc[len(dfs[k])] = (
                #     list(x) + row_mean[k] + row_min[k] + row_max[k]
                # )

        return dfs, x_names

    @classmethod
    def draw_plot(
        cls,
        dfs: dict[str, pd.DataFrame],
        bench: Benchmark,
        save_path: str,
        print_value_on_bar: bool,
        show_plots: bool,
        save_csv: bool = True,
        save_pdf: bool = False,
        **kwargs,
    ):
        plt.style.use("seaborn-v0_8")
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            rc={
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "grid.linewidth": 1.2,
            },
        )
        COLOR_PALETTE = build_color_palette(len(bench.line_names))

        for perf_key in dfs:
            plt.figure(figsize=(16, 8), dpi=150)
            ax = plt.gca()

            all_data = []
            labels = bench.line_names
            xvars = list(bench.x_vals)
            x_indices = np.arange(len(xvars))
            # Each group of bars occupies GROUP_FILL of the unit x-spacing,
            # guaranteeing a visible gap between consecutive x-groups regardless
            # of the number of labels.  Individual bars are capped at 0.6 so a
            # single-label plot doesn't produce an overly fat bar.
            _GROUP_FILL = 0.75
            bar_width = min(_GROUP_FILL / max(len(labels), 1), 0.6)

            for provider in bench.line_names:
                data = dfs[perf_key][provider].dropna().values
                all_data.append(data)

            # draw bar plots
            for i, (data, label) in enumerate(zip(all_data, labels)):
                color = COLOR_PALETTE[i]
                ax.bar(
                    x_indices + i * bar_width,
                    data,
                    width=bar_width,
                    label=label,
                    color=color,
                    edgecolor=None,
                    linewidth=0,
                    alpha=0.65,
                    zorder=2,
                )

                # Annotate bars
                for idx, value in enumerate(data):
                    if value == BENCH_CASE_OOM:  # OOM — ran but crashed
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 0.2,  # Position text slightly above the bar
                            "E",  # "OOM",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            fontweight="bold",
                            color="#d63031",  # warning red: tried but failed
                            zorder=4,
                        )
                    elif value == BENCH_CASE_NOT_SUPPORTED:  # not supported — N/A
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 0.2,
                            "X",
                            ha="center",
                            va="bottom",
                            fontsize=14,
                            fontweight="bold",
                            color="#636e72",  # neutral grey: not applicable
                            zorder=4,
                        )
                    elif print_value_on_bar:  # normal value
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 1.0,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="black",
                            zorder=4,
                        )

            # draw line plots — diamond markers with white-stroke path effect
            for i, (data, label) in enumerate(zip(all_data, labels)):
                plot_data = data.copy().astype(float)
                # Break the line at OOM / not-supported sentinels
                plot_data[
                    (plot_data == BENCH_CASE_OOM)
                    | (plot_data == BENCH_CASE_NOT_SUPPORTED)
                ] = np.nan

                ax.plot(
                    x_indices + i * bar_width,
                    plot_data,
                    color=COLOR_PALETTE[i],
                    marker="D",
                    markersize=4,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    linestyle="-",
                    linewidth=1.5,
                    path_effects=[
                        pe.Stroke(linewidth=1, foreground="white"),
                        pe.Normal(),
                    ],
                    zorder=3,
                )

            # always start y from zero
            y_min, y_max = 0.0, np.max(all_data) * 1.15
            ax.set_ylim(y_min, y_max)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.grid(axis="y", alpha=0.4, linestyle="-", linewidth=1.2)
            ax.grid(axis="x", visible=False)

            # set the xticks at the center of each group
            ax.set_xticks(x_indices + bar_width * (len(all_data) - 1) / 2)

            short_for_xlables = kwargs.get("short_for_xlables", None)
            if isinstance(short_for_xlables, dict):
                xvars = [
                    next(
                        (
                            x.replace(k, v)
                            for k, v in short_for_xlables.items()
                            if k in x
                        ),
                        x,
                    )
                    for x in xvars
                ]
            use_extend_labels = kwargs.get("use_extend_labels", True)
            if not use_extend_labels:
                xvars = [x.split("-")[0] for x in xvars]
            ax.set_xticklabels(xvars, rotation=0, fontsize=16)

            ax.tick_params(axis="y", labelsize=16)

            # set xlabel and ylabel
            ax.set_xlabel(
                bench.xlabel,
                fontsize=15,
                labelpad=12,
                fontweight="semibold",
            )
            ax.set_ylabel(
                bench.ylabel[perf_key]
                if isinstance(bench.ylabel, dict)
                else bench.ylabel,
                fontsize=15,
                labelpad=12,
                fontweight="semibold",
            )

            ax.set_title(
                f"Benchmark of {bench.plot_name} — {perf_key}",
                fontsize=19,
                pad=18,
                fontweight="bold",
                color="#2d3436",
            )

            # Legend placement: use a top-centred single row when labels are
            # short enough to fit comfortably (≤ 60 total chars); fall back to
            # a right-side vertical legend for longer / more numerous labels.
            _label_chars = sum(len(label) for label in labels)
            if _label_chars <= 60:
                legend = ax.legend(
                    frameon=True,
                    fontsize=11,
                    title=bench.line_arg,
                    title_fontsize="11",
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.0),
                    ncol=len(labels),
                    borderpad=0.5,
                )
                legend.get_frame().set_facecolor("#FFFFFFDD")
                legend.get_frame().set_edgecolor("#dfe6e9")
                legend.get_frame().set_linewidth(1.5)
                plt.tight_layout()
            else:
                legend = ax.legend(
                    frameon=True,
                    fontsize=11,
                    title=bench.line_arg,
                    title_fontsize="11",
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    borderaxespad=0,
                    borderpad=0.6,
                )
                legend.get_frame().set_facecolor("#FFFFFFDD")
                legend.get_frame().set_edgecolor("#dfe6e9")
                legend.get_frame().set_linewidth(1.5)
                # Reserve room for the right-side legend without clipping it.
                plt.tight_layout()
                plt.subplots_adjust(right=0.78)

            if save_path:
                if save_pdf:
                    plt.savefig(
                        os.path.join(save_path, f"{perf_key}_report.pdf"),
                        dpi=100,
                        bbox_inches="tight",
                        transparent=False,
                        facecolor="white",
                    )
                plt.savefig(
                    os.path.join(save_path, f"{perf_key}_report.png"),
                    dpi=100,
                    bbox_inches="tight",
                    transparent=False,
                    facecolor="white",
                )
            if show_plots:
                plt.show()
            plt.close()

        if save_path and save_csv:
            for name, df in dfs.items():
                df.to_csv(os.path.join(save_path, f"{name}.csv"), index=False)

    def _run(
        self,
        bench: Benchmark,
        save_path: str,
        show_plots: bool,
        print_data: bool,
        print_value_on_bar: bool,
        **kwargs,
    ):
        # Pop internal control flags before forwarding kwargs to the benchmark fn.
        save_pdf = kwargs.pop("save_pdf", False)

        # run the benchmark functions
        dfs, _ = self._call(bench, **kwargs)

        if print_data:
            for name, df in dfs.items():
                print(f"{name}: \n{df}\n")

        if not bench.plot_name:
            return

        should_plot = (not dist.is_initialized()) or dist.get_rank() == 0
        if should_plot:
            self.draw_plot(
                dfs=dfs,
                bench=bench,
                save_path=save_path,
                show_plots=show_plots,
                print_value_on_bar=print_value_on_bar,
                save_pdf=save_pdf,
                **kwargs,
            )

        return dfs

    def run(
        self,
        show_plots: bool = False,
        print_data: bool = False,
        print_value_on_bar: bool = False,
        save_path: str = "",
        return_df: bool = False,
        report_all_name: str = "perf_report_all",
        num_workers: int = 1,
        parallel_mode: Literal["inter", "intra"] = "inter",
        save_pdf: bool = False,
        save_html: bool = False,
        save_stats: bool = True,
        stats_config: StatsConfig | None = None,
        **kwargs,
    ):
        """Run all benchmarks and optionally save results to disk.

        Args:
            show_plots: Display the generated bar/line charts interactively via
                ``plt.show()``. Defaults to ``False``.
            print_data: Print each benchmark's result DataFrame to stdout after it
                finishes. Defaults to ``False``.
            print_value_on_bar: Annotate each bar in the chart with its numeric
                value. Defaults to ``False``.
            save_path: Root directory for all output files (CSV, PNG, PDF, HTML,
                stats JSON). Each benchmark's outputs are written to
                ``{save_path}/{bench.folder_name}/``. If empty, nothing is written
                to disk. Defaults to ``""``.
            return_df: Return the collected result DataFrames. When ``True``,
                returns a single ``dict[str, DataFrame]`` for a single
                ``Benchmark``, or a list of such dicts for multiple benchmarks.
                Defaults to ``False``.
            report_all_name: Base filename (without extension) for the combined
                report artefacts (e.g. ``perf_report_all.png`` / ``.pdf`` /
                ``.html``). Defaults to ``"perf_report_all"``.
            num_workers: Number of GPU workers to use for parallel execution. Must
                be between 1 and ``torch.cuda.device_count()``. When set to 1
                (default), benchmarks run sequentially on the current process
                without spawning sub-processes.
            parallel_mode: Parallelism strategy when ``num_workers > 1``:

                - ``"inter"`` *(default)* — **across benchmarks**. Each benchmark
                  object is assigned to a single GPU via round-robin; all benchmarks
                  run concurrently, each on its own GPU. Best when the number of
                  distinct benchmarks is large relative to ``num_workers``.
                - ``"intra"`` — **within each benchmark**. Benchmarks run one at a
                  time; for each benchmark the ``line_vals`` (tile configs / variants)
                  are split across GPUs via round-robin and evaluated in parallel.
                  Workers write per-rank partial CSVs that the main process merges
                  before plotting. Best for pretune workloads where each benchmark
                  has many ``line_vals`` and comparatively few benchmark objects.

            save_pdf: Additionally save each plot as a PDF alongside the PNG.
                Defaults to ``False``.
            save_html: Write a combined ``{report_all_name}.html`` file that embeds
                all generated PNG images. Defaults to ``False``.
            save_stats: Compute and save per-benchmark statistics (best config, peak
                performance, etc.) as part of the combined report.
                Defaults to ``True``.
            stats_config: Configuration object controlling which statistics are
                computed and how they are formatted. Falls back to
                ``StatsConfig()`` (defaults) when ``None``.
            **kwargs: Extra keyword arguments forwarded verbatim to the benchmark
                function for every call.
        """
        n_available = torch.cuda.device_count()
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if num_workers > n_available:
            raise ValueError(
                f"num_workers ({num_workers}) exceeds the number of available GPUs ({n_available})"
            )

        kwargs["save_pdf"] = save_pdf

        if stats_config is None:
            stats_config = StatsConfig()

        if num_workers > 1:
            return self._run_parallel(
                num_workers=num_workers,
                parallel_mode=parallel_mode,
                show_plots=show_plots,
                print_data=print_data,
                print_value_on_bar=print_value_on_bar,
                save_path=save_path,
                report_all_name=report_all_name,
                save_html=save_html,
                save_stats=save_stats,
                stats_config=stats_config,
                **kwargs,
            )

        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []

        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            if save_html:
                html = open(os.path.join(save_path, f"{report_all_name}.html"), "w")
                html.write("<html><body>\n")

        pbar = tqdm(benchmarks, total=len(benchmarks))
        for bench in pbar:
            maybe_dist_sync()
            bench_save_path = (
                os.path.join(save_path, bench.folder_name) if save_path else save_path
            )
            if bench_save_path:
                os.makedirs(bench_save_path, exist_ok=True)

            dfs = self._run(
                bench,
                bench_save_path,
                show_plots,
                print_data,
                print_value_on_bar,
                **kwargs,
            )
            result_dfs.append(dfs)

            should_plot = (not dist.is_initialized()) or dist.get_rank() == 0
            if save_html and bench_save_path and should_plot:
                for k in dfs:
                    html.write(f'<image src="{bench.folder_name}/{k}_report.png"/>\n')

        should_plot = (not dist.is_initialized()) or dist.get_rank() == 0
        if save_path and should_plot:
            if save_html:
                html.write("</body></html>\n")
                html.close()

            report_all_from_perf(
                save_root=save_path,
                report_all_name=report_all_name,
                save_pdf=kwargs.get("save_pdf", False),
                save_stats=save_stats,
                stats_config=stats_config,
            )

        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs

        return None

    # ------------------------------------------------------------------
    # Multi-GPU parallel execution (internal)
    # ------------------------------------------------------------------

    @staticmethod
    def _worker_run(
        rank: int,
        mark_name: str,
        benchmarks: "list[Benchmark]",
        bench_indices: "list[int]",
        save_path: str,
        show_plots: bool,
        print_data: bool,
        print_value_on_bar: bool,
        run_kwargs: dict,
    ) -> None:
        """Worker function executed in a spawned subprocess on a single GPU.

        ``mark_name`` is the module-level attribute name of the Mark instance in
        the user's script.  The worker resolves it after the script has been
        re-imported by spawn, then retrieves the original benchmark function.
        """
        torch.cuda.set_device(rank)

        # After spawn the user's script is imported as '__mp_main__' (or '__main__'
        # in fork mode).  Resolve the Mark from there and extract the real fn.
        main_mod = sys.modules.get("__mp_main__") or sys.modules.get("__main__")
        mark_obj = getattr(main_mod, mark_name)
        fn = mark_obj.fn if isinstance(mark_obj, Mark) else mark_obj

        mark = Mark(fn, benchmarks)
        pbar = tqdm(
            bench_indices,
            total=len(bench_indices),
            desc=f"GPU{rank}",
            position=rank,
            leave=True,
        )
        for idx in pbar:
            bench = benchmarks[idx]
            bench_save_path = (
                os.path.join(save_path, bench.folder_name) if save_path else save_path
            )
            if bench_save_path:
                os.makedirs(bench_save_path, exist_ok=True)
            mark._run(
                bench,
                bench_save_path,
                show_plots,
                print_data,
                print_value_on_bar,
                **run_kwargs,
            )

    def _run_parallel(
        self,
        num_workers: int,
        parallel_mode: str,
        show_plots: bool,
        print_data: bool,
        print_value_on_bar: bool,
        save_path: str,
        report_all_name: str,
        save_html: bool = False,
        save_stats: bool = True,
        stats_config: StatsConfig | None = None,
        **kwargs,
    ):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Find the module-level name of this Mark instance in __main__ so that
        # workers can look it up after re-importing the script (spawn re-imports
        # the script as '__mp_main__', making fn itself unpicklable via its
        # original qualname).
        main_mod = sys.modules.get("__main__")
        mark_name: str | None = None
        if main_mod is not None:
            for attr, val in vars(main_mod).items():
                if val is self:
                    mark_name = attr
                    break
        if mark_name is None:
            raise RuntimeError(
                "Could not locate the Mark instance as a module-level name in __main__. "
                "Make sure the @perf_report-decorated function is defined at module level."
            )

        if parallel_mode == "intra":
            return self._run_intra_parallel(
                num_workers=num_workers,
                benchmarks=benchmarks,
                mark_name=mark_name,
                show_plots=show_plots,
                print_data=print_data,
                print_value_on_bar=print_value_on_bar,
                save_path=save_path,
                report_all_name=report_all_name,
                save_html=save_html,
                save_stats=save_stats,
                stats_config=stats_config,
                **kwargs,
            )

        # Round-robin assignment: worker k gets benchmarks at indices k, k+N, k+2N, …
        bench_indices_per_worker: list[list[int]] = [[] for _ in range(num_workers)]
        for i in range(len(benchmarks)):
            bench_indices_per_worker[i % num_workers].append(i)
        ctx = torch.multiprocessing.get_context("spawn")
        processes: list[torch.multiprocessing.Process] = []
        for rank in range(num_workers):
            p = ctx.Process(
                target=Mark._worker_run,
                args=(
                    rank,
                    mark_name,
                    benchmarks,
                    bench_indices_per_worker[rank],
                    save_path,
                    show_plots,
                    print_data,
                    print_value_on_bar,
                    kwargs,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        failed = [rank for rank, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            raise RuntimeError(
                f"Worker processes on GPU(s) {failed} exited with non-zero exit codes: "
                f"{[processes[r].exitcode for r in failed]}"
            )

        # ── Main process: generate combined report ─────────────────────
        if save_path:
            if save_html:
                html_path = os.path.join(save_path, f"{report_all_name}.html")
                with open(html_path, "w") as html:
                    html.write("<html><body>\n")
                    for bench in benchmarks:
                        bench_dir = os.path.join(save_path, bench.folder_name)
                        if os.path.isdir(bench_dir):
                            for fname in sorted(os.listdir(bench_dir)):
                                if fname.endswith("_report.png"):
                                    html.write(
                                        f'<image src="{bench.folder_name}/{fname}"/>\n'
                                    )
                    html.write("</body></html>\n")

            report_all_from_perf(
                save_root=save_path,
                report_all_name=report_all_name,
                save_pdf=kwargs.get("save_pdf", False),
                save_stats=save_stats,
                stats_config=stats_config,
            )

    @staticmethod
    def _worker_run_intra(
        rank: int,
        mark_name: str,
        bench: "Benchmark",
        my_line_indices: "list[int]",
        bench_save_path: str,
        run_kwargs: dict,
    ) -> None:
        """Worker for intra-parallel mode: runs a subset of line_vals on one GPU.

        Each worker handles ``my_line_indices`` of ``bench.line_vals``, writes
        per-perf-key partial CSVs to ``{bench_save_path}/_partial/rank{rank}_{perf_key}.csv``.
        """
        torch.cuda.set_device(rank)

        main_mod = sys.modules.get("__mp_main__") or sys.modules.get("__main__")
        mark_obj = getattr(main_mod, mark_name)
        fn = mark_obj.fn if isinstance(mark_obj, Mark) else mark_obj

        mark = Mark(fn, bench)

        # Build a partial bench with only this worker's line_vals / line_names
        partial_bench = deepcopy(bench)
        partial_bench.line_vals = [bench.line_vals[i] for i in my_line_indices]
        partial_bench.line_names = [bench.line_names[i] for i in my_line_indices]

        # Strip plot-control flags that must not reach the benchmark fn
        call_kwargs = dict(run_kwargs)
        call_kwargs.pop("save_pdf", None)

        dfs, _ = mark._call(partial_bench, **call_kwargs)

        if bench_save_path:
            partial_dir = os.path.join(bench_save_path, "_partial")
            os.makedirs(partial_dir, exist_ok=True)
            for perf_key, df in dfs.items():
                df.to_csv(
                    os.path.join(partial_dir, f"rank{rank}_{perf_key}.csv"),
                    index=False,
                )

    def _run_intra_parallel(
        self,
        num_workers: int,
        benchmarks: "list[Benchmark]",
        mark_name: str,
        show_plots: bool,
        print_data: bool,
        print_value_on_bar: bool,
        save_path: str,
        report_all_name: str,
        save_html: bool = False,
        save_stats: bool = True,
        stats_config: StatsConfig | None = None,
        **kwargs,
    ) -> None:
        """Intra-parallel execution: benchmarks run serially; line_vals are split across GPUs."""
        save_pdf = kwargs.get("save_pdf", False)
        ctx = torch.multiprocessing.get_context("spawn")

        pbar = tqdm(benchmarks, total=len(benchmarks), desc="Benchmarks (intra)")
        for bench in pbar:
            bench_save_path = (
                os.path.join(save_path, bench.folder_name) if save_path else ""
            )
            if bench_save_path:
                os.makedirs(bench_save_path, exist_ok=True)

            # Assign line_vals to workers in round-robin order
            line_val_indices_per_worker: list[list[int]] = [
                [] for _ in range(num_workers)
            ]
            for i in range(len(bench.line_vals)):
                line_val_indices_per_worker[i % num_workers].append(i)

            # Spawn one worker per GPU
            processes: list[tuple[int, torch.multiprocessing.Process]] = []
            for rank in range(num_workers):
                if not line_val_indices_per_worker[rank]:
                    continue
                p = ctx.Process(
                    target=Mark._worker_run_intra,
                    args=(
                        rank,
                        mark_name,
                        bench,
                        line_val_indices_per_worker[rank],
                        bench_save_path,
                        kwargs,
                    ),
                )
                p.start()
                processes.append((rank, p))

            for _, p in processes:
                p.join()

            failed = [rank for rank, p in processes if p.exitcode != 0]
            if failed:
                raise RuntimeError(
                    f"Worker processes on GPU(s) {failed} exited with non-zero exit codes: "
                    f"{[p.exitcode for rank, p in processes if rank in failed]}"
                )

            # ── Merge partial CSVs and draw plot ───────────────────────
            if bench_save_path:
                partial_dir = os.path.join(bench_save_path, "_partial")
                merged_dfs: dict[str, pd.DataFrame] = {}

                if os.path.isdir(partial_dir):
                    # Collect perf_keys from filenames: rank{rank}_{perf_key}.csv
                    perf_keys: set[str] = set()
                    for fname in os.listdir(partial_dir):
                        if fname.endswith(".csv"):
                            # strip "rank{N}_" prefix
                            tail = fname[:-4]  # remove .csv
                            underscore_pos = tail.index("_")
                            perf_keys.add(tail[underscore_pos + 1 :])

                    x_cols = list(bench.x_names)

                    for perf_key in sorted(perf_keys):
                        frames = []
                        for rank in range(num_workers):
                            fpath = os.path.join(
                                partial_dir, f"rank{rank}_{perf_key}.csv"
                            )
                            if os.path.exists(fpath):
                                frames.append(pd.read_csv(fpath))

                        if not frames:
                            continue

                        # All frames share the same x_cols rows; join on them
                        merged = frames[0]
                        for df in frames[1:]:
                            value_cols = [c for c in df.columns if c not in x_cols]
                            merged = merged.merge(
                                df[x_cols + value_cols], on=x_cols, how="outer"
                            )

                        # Reorder line columns to match original bench.line_names order
                        existing_line_cols = [
                            c for c in bench.line_names if c in merged.columns
                        ]
                        merged = merged[x_cols + existing_line_cols]
                        merged_dfs[perf_key] = merged

                if merged_dfs and bench.plot_name:
                    self.draw_plot(
                        dfs=merged_dfs,
                        bench=bench,
                        save_path=bench_save_path,
                        show_plots=show_plots,
                        print_value_on_bar=print_value_on_bar,
                        save_pdf=save_pdf,
                        save_csv=True,
                    )

                if print_data:
                    for perf_key, df in merged_dfs.items():
                        print(f"{perf_key}: \n{df}\n")

        if save_path:
            if save_html:
                html_path = os.path.join(save_path, f"{report_all_name}.html")
                with open(html_path, "w") as html:
                    html.write("<html><body>\n")
                    for bench in benchmarks:
                        bench_dir = os.path.join(save_path, bench.folder_name)
                        if os.path.isdir(bench_dir):
                            for fname in sorted(os.listdir(bench_dir)):
                                if fname.endswith("_report.png"):
                                    html.write(
                                        f'<image src="{bench.folder_name}/{fname}"/>\n'
                                    )
                    html.write("</body></html>\n")

            report_all_from_perf(
                save_root=save_path,
                report_all_name=report_all_name,
                save_pdf=save_pdf,
                save_stats=save_stats,
                stats_config=stats_config,
            )

    @classmethod
    def draw_from_csv(
        cls,
        csv_path: str,
        perf_key: str,
        plot_name: str,
        line_arg: str,
        save_path: str,
        print_value_on_bar: bool = False,
        show_plots: bool = True,
        ylabel: str | dict[str, str] = "",
        x_int: bool = False,
        x_log: bool = False,
        **kwargs,
    ):
        benchmark = Benchmark.from_csv(
            csv_path=csv_path,
            plot_name=plot_name,
            line_arg=line_arg,
            ylabel=ylabel,
            x_int=x_int,
            x_log=x_log,
        )

        dfs: dict[str, pd.DataFrame] = {}
        dfs[perf_key] = pd.read_csv(csv_path)

        cls.draw_plot(
            dfs=dfs,
            bench=benchmark,
            save_path=save_path,
            show_plots=show_plots,
            print_value_on_bar=print_value_on_bar,
            save_csv=False,
            **kwargs,
        )


# copied from triton.testing.perf_report
# see https://github.com/openai/triton/blob/ccc25eb0d6261587a61b8ce8cff6ff1ad1d579fd/python/triton/testing.py#L357
def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """

    def wrapper(fn):
        return Mark(fn, benchmarks)

    return wrapper


def report_all_from_perf(
    save_root: str,
    report_all_name: str = "perf_report_all",
    save_pdf: bool = False,
    save_stats: bool = True,
    stats_config: StatsConfig | None = None,
):
    """Generate a combined image grid and a JSON stats file.

    Args:
        save_root:       Root directory that contains per-benchmark sub-folders.
        report_all_name: Base name (no extension) for the combined PNG/PDF.
        save_pdf:        Also save the combined grid as a PDF.
        save_stats:      If True (default), write
                         ``{report_all_name}_stats.json`` to save_root.
        stats_config:    A :class:`StatsConfig` instance controlling how stats
                         are computed.  Uses ``StatsConfig()`` defaults when
                         *None*.
    """
    make_img_grid(
        img_dir=save_root,
        save_path=os.path.join(save_root, f"{report_all_name}.png"),
        ignore_patterns=[report_all_name],
        save_pdf=save_pdf,
    )

    if save_stats:
        if stats_config is None:
            stats_config = StatsConfig()
        stats = collect_best_legend(save_root, stats_config)
        out_path = os.path.join(save_root, f"{report_all_name}_stats.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
