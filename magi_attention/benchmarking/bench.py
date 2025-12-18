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

import gc
import os
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
from matplotlib import patheffects as pe
from py3nvml import py3nvml
from tqdm import tqdm

from .image_grid import make_img_grid

# -------------------       bench utils     ------------------- #


@contextmanager
def nvml_context():
    py3nvml.nvmlInit()
    yield
    py3nvml.nvmlShutdown()


class MemRecorder:
    def __init__(self, mode="allocated", device_idx=0) -> None:
        self.memory = None
        self.mode = mode
        self.device_idx = device_idx

    def get_alloc_memory_from_torch(self):
        return torch.cuda.memory_allocated()

    @nvml_context()
    def get_alloc_memory_from_nvml(self):
        handle = py3nvml.nvmlDeviceGetHandleByIndex(self.device_idx)
        meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        return meminfo.used

    def __enter__(self):
        if self.mode == "peak":
            torch.cuda.reset_peak_memory_stats()
        elif self.mode == "allocated":
            # self.memory = self.get_alloc_memory_from_torch()
            self.memory = self.get_alloc_memory_from_nvml()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "peak":
            self.memory = torch.cuda.max_memory_allocated()
        elif self.mode == "allocated":
            # self.memory = self.get_alloc_memory_from_torch() - self.memory
            # self.memory = self.get_alloc_memory_from_nvml() - self.memory
            self.memory = self.get_alloc_memory_from_nvml()


# copied and modified from triton.testing.do_bench to add flops report with peak memory report
# see https://github.com/openai/triton/blob/ccc25eb0d6261587a61b8ce8cff6ff1ad1d579fd/python/triton/testing.py#L79
def do_bench(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    fast_flush=True,
    return_mode="mean",
    return_flops=True,
    return_mem=True,
    mem_record_mode="allocated",
    device_idx=0,
):
    """
    Benchmark the flops / peak memory of the provided function.
    By default, return the median flops and peak memory

    Args:
        fn (Callable): Function to benchmark
        warmup (int): Warmup time (in ms)
        rep (int): Repetition time (in ms)
        grad_to_none (torch.tensor, optional): Reset the gradient of the provided tensor to None
        quantiles (list[float], optional): Performance percentile to return in addition to the median
        fast_flush (bool): Whether to use faster kernel to flush L2 between measurements
        return_mode (str): the statistics mode to return if `quantiles` is None, choosed from ["min", "max", "mean", "median"]
        return_flops (bool): whether to return flops report
        return_mem (bool): whether to return mem report

    Returns:
        ret (dict): the statistics of flops and / or peak memory
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

    # compute number of warmup and repeat
    n_warmup = warmup
    n_repeat = rep
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    mems = [0.0] * n_repeat

    # Warm-up
    torch.cuda.nvtx.range_push("warmup")
    for _ in range(n_warmup):
        fn()
    torch.cuda.nvtx.range_pop()

    # Benchmark
    if dist.is_initialized():
        dist.all_reduce(cache, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    for i in range(n_repeat):
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

    if dist.is_initialized():
        dist.all_reduce(cache, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
        device=torch.device("cuda"),
    )
    dist.all_reduce(times, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
    times = times.to(device=torch.device("cpu"))
    mems = torch.tensor(mems, dtype=torch.float)

    torch.cuda.empty_cache()
    gc.collect()

    # get quantiles
    if quantiles is not None:
        ret_flops = _get_item(
            torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        )
        ret_mem = _get_item(
            torch.quantile(mems, torch.tensor(quantiles, dtype=torch.float)).tolist()
        )
        return _get_ret(ret_flops, ret_mem)

    return _get_ret(
        getattr(torch, return_mode)(times).item(),
        getattr(torch, return_mode)(mems).item(),
    )


def do_bench_flops(*args, **kwargs):
    # just use the same bench func from triton.testing
    # return triton.testing.do_bench(*args, **kwargs)

    return partial(do_bench, return_flops=True, return_mem=False)(*args, **kwargs)


def do_bench_mem(*args, **kwargs):
    # just use the same bench func from triton.testing
    # return triton.testing.do_bench(*args, **kwargs)

    return partial(do_bench, return_flops=False, return_mem=True)(*args, **kwargs)


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


# to sync before each sweep for distributed bench
def maybe_dist_sync():
    if dist.is_initialized():
        torch.cuda.synchronize()
        dist.barrier()


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
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
                "grid.linewidth": 1.2,
            },
        )
        COLOR_PALETTE = sns.color_palette("viridis", n_colors=len(bench.line_names))

        for perf_key in dfs:
            plt.figure(figsize=(14, 8), dpi=100)
            ax = plt.gca()

            all_data = []
            labels = bench.line_names
            xvars = bench.x_vals
            x_indices = np.arange(len(xvars))
            bar_width = 0.25 if len(labels) < 4 else (0.15 if len(labels) < 6 else 0.12)

            for provider in bench.line_names:
                data = dfs[perf_key][provider].dropna().values
                all_data.append(data)

            # draw bar plots
            for i, (data, label) in enumerate(zip(all_data, labels)):
                edge_color = COLOR_PALETTE[i] + (0.7,)
                ax.bar(
                    x_indices + i * bar_width,
                    data,
                    width=bar_width,
                    label=label,
                    color=COLOR_PALETTE[i],
                    edgecolor=edge_color,
                    linewidth=1.5,
                    alpha=0.65,
                    zorder=2,
                )

                # Annotate bars
                for idx, value in enumerate(data):
                    if value == -1:  # OOM
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 0.2,  # Position text slightly above the bar
                            # "OOM",
                            "E",
                            ha="center",
                            va="bottom",
                            fontsize=15,
                            fontweight="bold",  # Add this line to make the text bold
                            # color=COLOR_PALETTE[i],
                            color="red",
                            zorder=4,
                        )
                    elif value == -2:  # not supported
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 0.2,
                            "X",
                            ha="center",
                            va="bottom",
                            fontsize=15,
                            fontweight="bold",  # Add this line to make the text bold
                            # color=COLOR_PALETTE[i],
                            color="red",
                            zorder=4,
                        )
                    elif print_value_on_bar:  # normal value
                        ax.text(
                            x_indices[idx] + i * bar_width,
                            value + 1.0,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            # color=COLOR_PALETTE[i],
                            color="black",
                            zorder=4,
                        )

            # draw line plots
            for i, (data, label) in enumerate(zip(all_data, labels)):
                # Create a copy of the data to modify
                plot_data = data.copy().astype(float)

                # Insert np.nan where value is -1 or -2 to break the line
                plot_data[(plot_data == -1) | (plot_data == -2)] = np.nan

                ax.plot(
                    x_indices + i * bar_width,
                    plot_data,
                    color=COLOR_PALETTE[i],
                    # label=label, # ignore the plot label
                    marker="D",
                    markersize=8,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    linestyle="-",
                    linewidth=2.5,
                    path_effects=[
                        pe.Stroke(linewidth=4, foreground="white"),
                        pe.Normal(),
                    ],
                    zorder=3,
                )

            # y_min, y_max = np.min(all_data) * 0.9, np.max(all_data) * 1.15
            # always start from zero
            y_min, y_max = 0.0, np.max(all_data) * 1.15
            ax.set_ylim(y_min, y_max)

            ax.legend()

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=1.2)

            # set the xticks t the center of each group with right xticklabels
            ax.set_xticks(x_indices + bar_width * (len(all_data) - 1) / 2)

            short_for_xlables = kwargs.get("short_for_xlables", None)
            if short_for_xlables is not None and isinstance(short_for_xlables, dict):
                xvars = [
                    short_for_xlables[x] if x in short_for_xlables else x for x in xvars
                ]
            ax.set_xticklabels(xvars, rotation=0)

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
                f"The benchmark of {perf_key}\n{bench.plot_name}",
                fontsize=19,
                pad=18,
                fontweight="bold",
                color="#2d3436",
            )

            legend = ax.legend(
                frameon=True,
                shadow=True,
                fontsize=15,
                borderpad=1,
                title=bench.line_arg,
                title_fontsize="18",
                loc="upper left",
                bbox_to_anchor=(1, 1),
            )
            legend.get_frame().set_facecolor("#FFFFFFDD")
            legend.get_frame().set_edgecolor("#dfe6e9")
            legend.get_frame().set_linewidth(1.5)

            plt.tight_layout()
            if save_path:
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
        **kwargs,
    ):
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []

        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            html = open(os.path.join(save_path, f"{report_all_name}.html"), "w")
            html.write("<html><body>\n")

        pbar = tqdm(benchmarks, total=len(benchmarks))
        for bench in pbar:
            maybe_dist_sync()
            bench_save_path = (
                os.path.join(save_path, bench.plot_name) if save_path else save_path
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
            if bench_save_path and should_plot:
                for k in dfs:
                    html.write(f'<image src="{bench.plot_name}/{k}_report.png"/>\n')

        should_plot = (not dist.is_initialized()) or dist.get_rank() == 0
        if save_path and should_plot:
            html.write("</body></html>\n")
            html.close()

            report_all_from_perf(
                save_root=save_path,
                report_all_name=report_all_name,
            )

        if return_df:
            if has_single_bench:
                return result_dfs[0]
            else:
                return result_dfs

        return None

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
):
    make_img_grid(
        img_dir=save_root,
        save_path=os.path.join(save_root, f"{report_all_name}.png"),
        ignore_patterns=[report_all_name],
    )
