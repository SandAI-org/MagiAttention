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

import fnmatch
import logging
import os
import re
import threading
import time
import warnings
from contextlib import contextmanager
from functools import wraps
from glob import escape as glob_escape
from glob import glob
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    Optional,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import torch
import torch.distributed as dist

from .general import wrap_to_list

logger = logging.getLogger(__name__)

__version__ = "1.1.0"

# Global flag to enable/disable the profiler
_PROFILER_ENABLED = False
_EMIT_NVTX_CTX: None | torch.autograd.profiler.emit_nvtx = None
_NSYS_REPORT_FILES_BEFORE: dict[str, tuple[int, int]] = {}
_NSYS_REPORT_DIR: Optional[str] = None
_NSYS_REPORT_BASE: Optional[str] = None
_NSYS_COMMAND: Optional[tuple[list[str], str]] = None
_NSYS_COMMAND_DISCOVERED = False
_NSYS_CAPTURE_LIMIT_WARNED = False
NSYS_CAPTURE_PHASE_ENV = "MAGI_ATTENTION_NSYS_CAPTURE_PHASE"

# Nsight capture is process-global. Nested matching phases are represented by
# nested NVTX ranges inside the outer capture rather than starting another one.
_ACTIVE_NSYS_PHASE: Optional[str] = None

# fixed the mypy type check missing bug
# when a func is wrapped
# issue: https://stackoverflow.com/questions/65621789/mypy-untyped-decorator-makes-function-my-method-untyped
F = TypeVar("F", bound=Callable[..., Any])
ProfileType: TypeAlias = Literal["memory", "nsys"]

# Prefix automatically prepended to every NVTX range recorded via
# `add_nvtx_event` / `instrument_nvtx`, giving a consistent naming convention
# when profiling MagiAttention operations.
NVTX_EVENT_PREFIX = ""


def _forward_pre_hook(layer, inputs):
    """
    Pre-hook for the forward pass of a layer. If profiling is enabled,
    pushes a range onto the NVTX stack.

    Args:
    - layer: The layer for which this hook is called.
    - inputs: The inputs to the layer.
    """
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_push(layer.__class__.__name__ + "_fwd")


def _forward_post_hook(module, inputs, outputs):
    """
    Post-hook for the forward pass of a module. If profiling is enabled,
    pops a range from the NVTX stack.

    Args:
    - module: The module for which this hook is called.
    - inputs: The inputs to the module.
    - outputs: The outputs from the module.
    """
    if _PROFILER_ENABLED:
        torch.cuda.nvtx.range_pop()


def _register_hook_recursively(module, pre_hook, post_hook):
    """
    Recursively registers pre and post hooks to all submodules of the given module.

    Args:
    - module: The root module to register hooks for.
    - pre_hook: The pre-hook function to be registered.
    - post_hook: The post-hook function to be registered.
    """
    if not isinstance(module, torch.nn.Module):
        return

    # Recursively apply hooks to all submodules
    for submodule in module.children():
        _register_hook_recursively(submodule, pre_hook, post_hook)

    # Register hooks
    if pre_hook is not None:
        module.register_forward_pre_hook(hook=pre_hook)
    if post_hook is not None:
        module.register_forward_hook(hook=post_hook)


def register_profile_hook(model):
    """
    Registers profiling hooks for a PyTorch model or list of models.

    Args:
    - model: A PyTorch model or a list of PyTorch models.
    """
    if isinstance(model, torch.nn.Module):
        _register_hook_recursively(model, _forward_pre_hook, _forward_post_hook)
    elif isinstance(model, list):
        raise RuntimeError("register_profile_hook given a list of models")


def _get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_local_world_size() -> int:
    return int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("NPROC_PER_NODE", "1"))
    )


def _is_nsys_controller(rank: int, profile_ranks: list[int]) -> bool:
    """Return whether this process controls the node-level Nsight session."""
    local_rank = _get_local_rank()
    local_world_size = _get_local_world_size()

    if local_world_size <= 0 or not 0 <= local_rank < local_world_size:
        raise RuntimeError(
            "LOCAL_RANK must be in [0, LOCAL_WORLD_SIZE) to select one "
            "Nsight controller per node"
        )

    node_rank_start = rank - local_rank
    node_rank_end = node_rank_start + local_world_size
    node_is_profiled = any(
        node_rank_start <= profile_rank < node_rank_end
        for profile_rank in profile_ranks
    )
    return local_rank == 0 and node_is_profiled


def _get_profile_window(
    iter_id: int, start: int, end: int, interval: int
) -> Optional[tuple[int, int]]:
    """Return the active profiling window for ``iter_id``."""
    if start < 0:
        return None
    if end <= start:
        raise ValueError("profiling end must be greater than start")
    if interval < 0:
        raise ValueError("profiling interval must be non-negative")

    if interval == 0:
        return (start, end) if start <= iter_id <= end else None

    window_span = end - start
    if interval <= window_span:
        raise ValueError(
            "profiling interval must be greater than end - start "
            "so profiling windows do not overlap"
        )
    if iter_id < start:
        return None

    window_index = (iter_id - start) // interval
    window_start = start + window_index * interval
    window_end = end + window_index * interval
    return (window_start, window_end) if iter_id <= window_end else None


def _get_parent_pid(pid: int) -> int:
    """Return ``pid``'s parent from Linux procfs."""
    with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
        fields_after_comm = f.read().rsplit(")", 1)[1].split()
    return int(fields_after_comm[1])


def _get_process_args(pid: int) -> list[str]:
    """Return a process command line from Linux procfs."""
    with open(f"/proc/{pid}/cmdline", "rb") as f:
        return [
            arg.decode(errors="surrogateescape") for arg in f.read().split(b"\0") if arg
        ]


def _get_nsys_output_base(args: list[str], cwd: str) -> Optional[str]:
    """Extract the absolute output base from an ``nsys profile`` command."""
    if not args or os.path.basename(args[0]) != "nsys" or "profile" not in args[1:]:
        return None

    output: Optional[str] = None
    for index, arg in enumerate(args[1:], start=1):
        if arg in ("-o", "--output") and index + 1 < len(args):
            output = args[index + 1]
            break
        if arg.startswith("--output="):
            output = arg.split("=", 1)[1]
            break

    if output is None:
        output = "report"
    output = os.path.expanduser(output)
    if not os.path.isabs(output):
        output = os.path.join(cwd, output)
    output = os.path.abspath(output)
    return output.removesuffix(".nsys-rep")


def _get_nsys_output_dir(args: list[str], cwd: str) -> Optional[str]:
    """Extract the report directory from an ``nsys profile`` command line."""
    output_base = _get_nsys_output_base(args, cwd)
    return os.path.dirname(output_base) if output_base is not None else None


def _find_nsys_command() -> Optional[tuple[list[str], str]]:
    """Return the wrapping ``nsys profile`` arguments and working directory."""
    pid = os.getpid()
    while pid > 1:
        try:
            args = _get_process_args(pid)
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if _get_nsys_output_base(args, cwd) is not None:
                return args, cwd
            pid = _get_parent_pid(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            return None
    return None


def _get_nsys_option(args: list[str], option: str) -> Optional[str]:
    """Read ``--option value`` or ``--option=value`` from an Nsight command."""
    for index, arg in enumerate(args):
        if arg == option:
            return args[index + 1] if index + 1 < len(args) else None
        if arg.startswith(f"{option}="):
            return arg.split("=", 1)[1]
    return None


def _validate_nsys_command(
    args: list[str],
    *,
    exit_after_end_iter: bool,
    interval: int,
    max_capture_ranges: int,
) -> Optional[int]:
    """Validate capture-end behavior and return the repeat capture limit."""
    if max_capture_ranges <= 0:
        raise ValueError("max_capture_ranges must be positive")

    capture_range_end = _get_nsys_option(args, "--capture-range-end")
    if capture_range_end is None:
        raise ValueError("nsys profile must specify --capture-range-end")

    if exit_after_end_iter:
        if interval > 0:
            raise ValueError("interval must be 0 when exit_after_end_iter=True")
        if capture_range_end not in ("stop", "stop-shutdown"):
            raise ValueError(
                "exit_after_end_iter=True requires "
                "--capture-range-end=stop or stop-shutdown"
            )
        return 1

    match = re.fullmatch(r"repeat(?::([1-9][0-9]*))?:sync", capture_range_end)
    if match is None:
        raise ValueError(
            "exit_after_end_iter=False requires "
            "--capture-range-end=repeat:N:sync or repeat:sync"
        )
    capture_limit = int(match.group(1)) if match.group(1) is not None else None
    if capture_limit is not None and capture_limit < max_capture_ranges:
        raise ValueError(
            f"Nsight --capture-range-end allows {capture_limit} capture range(s), "
            f"but max_capture_ranges={max_capture_ranges}; increase the Nsight "
            "repeat count or lower max_capture_ranges"
        )
    return capture_limit


def _get_nsys_command() -> Optional[tuple[list[str], str]]:
    """Discover the wrapping Nsight command at most once per process."""
    global _NSYS_COMMAND
    global _NSYS_COMMAND_DISCOVERED
    if not _NSYS_COMMAND_DISCOVERED:
        _NSYS_COMMAND = _find_nsys_command()
        _NSYS_COMMAND_DISCOVERED = True
    return _NSYS_COMMAND


def _get_nsys_reports() -> list[str]:
    """Return reports belonging to this process's exact Nsight output base."""
    if _NSYS_REPORT_BASE is None:
        return []

    report_paths = []
    output_name = os.path.basename(_NSYS_REPORT_BASE)
    numbered_report = re.compile(rf"^{re.escape(output_name)}\.[0-9]+\.nsys-rep$")
    for path in glob(f"{glob_escape(_NSYS_REPORT_BASE)}*.nsys-rep"):
        basename = os.path.basename(path)
        if basename == f"{output_name}.nsys-rep" or numbered_report.fullmatch(basename):
            report_paths.append(path)
    return report_paths


def _snapshot_nsys_reports() -> None:
    """Remember existing reports so the next synchronous report can be found."""
    global _NSYS_REPORT_DIR
    global _NSYS_REPORT_BASE
    if _NSYS_COMMAND is None:
        return
    args, cwd = _NSYS_COMMAND
    _NSYS_REPORT_BASE = _get_nsys_output_base(args, cwd)
    if _NSYS_REPORT_BASE is None:
        return
    _NSYS_REPORT_DIR = os.path.dirname(_NSYS_REPORT_BASE)

    global _NSYS_REPORT_FILES_BEFORE
    _NSYS_REPORT_FILES_BEFORE = {}
    for path in _get_nsys_reports():
        stat = os.stat(path)
        _NSYS_REPORT_FILES_BEFORE[path] = (stat.st_mtime_ns, stat.st_size)


def _rename_latest_nsys_report(report_name: str) -> None:
    """Rename the report generated by the latest synchronous Nsight capture."""
    if _NSYS_REPORT_DIR is None:
        return

    changed_reports = []
    for path in _get_nsys_reports():
        stat = os.stat(path)
        if _NSYS_REPORT_FILES_BEFORE.get(path) != (stat.st_mtime_ns, stat.st_size):
            changed_reports.append((stat.st_mtime_ns, path))

    if not changed_reports:
        raise RuntimeError("Nsight did not generate a synchronous report")

    _, report_path = max(changed_reports)
    target_path = os.path.join(os.path.dirname(report_path), report_name)
    os.replace(report_path, target_path)


def _rename_nsys_report(window_start: int, window_end: int, node_rank: int) -> None:
    """Rename an iteration-window report generated by Nsight."""
    _rename_latest_nsys_report(
        f"profile_iter({window_start}-{window_end})_n{node_rank}.nsys-rep"
    )


def _get_node_rank(rank: int) -> int:
    """Derive the node index from local-rank/local-world-size launcher variables."""
    local_rank = _get_local_rank()
    local_world_size = _get_local_world_size()
    if local_world_size <= 0:
        return 0
    return (rank - local_rank) // local_world_size


def _matches_phase_selector(name: str, selector: Optional[str] = None) -> bool:
    """Match a phase against comma-separated exact, glob, or ``re:`` selectors."""
    selector = os.getenv(NSYS_CAPTURE_PHASE_ENV, "") if selector is None else selector
    for pattern in (item.strip() for item in selector.split(",")):
        if not pattern:
            continue
        if pattern.startswith("re:"):
            if re.fullmatch(pattern.removeprefix("re:"), name) is not None:
                return True
        elif fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def _phase_report_name(name: str, node_rank: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "unnamed"
    return f"profile_phase({safe_name})_n{node_rank}.nsys-rep"


@contextmanager
def profile_phase(
    name: str,
    *,
    category: Literal["cold_start", "runtime"] = "runtime",
    capture_nsys: bool = True,
) -> Generator[None, None, None]:
    """Measure a phase's CPU wall time and optionally capture it with Nsight.

    CPU timing is always logged and deliberately does not synchronize CUDA.
    Nsight capture is enabled when ``capture_nsys`` is true, ``name`` matches a
    comma-separated selector in ``MAGI_ATTENTION_NSYS_CAPTURE_PHASE``, and the process is
    wrapped by ``nsys profile --capture-range=cudaProfilerApi``. Selectors are
    exact or shell-style globs; prefix one with ``re:`` for a regular expression.

    Nsight capture is process-global and therefore only starts on the main
    thread. A matching phase nested inside an active capture adds an NVTX range
    but does not start another capture range.
    """
    global _ACTIVE_NSYS_PHASE

    selected = capture_nsys and _matches_phase_selector(name)
    on_main_thread = threading.current_thread() is threading.main_thread()
    nsys_command = _get_nsys_command() if selected and on_main_thread else None
    capture_available = nsys_command is not None
    nested_capture = capture_available and _ACTIVE_NSYS_PHASE is not None
    starts_capture = capture_available and not nested_capture
    distributed = starts_capture and dist.is_initialized()
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_controller = starts_capture and _is_nsys_controller(rank, [rank])
    pushed_nvtx = False

    if selected and not on_main_thread:
        logger.warning(
            f"Skipping Nsight capture for phase {name!r}: profiler control is "
            "only supported on the main thread"
        )

    if starts_capture:
        assert nsys_command is not None  # mypy
        args, _ = nsys_command
        _validate_nsys_command(
            args,
            exit_after_end_iter=False,
            interval=0,
            max_capture_ranges=1,
        )
        if distributed:
            dist.barrier()
        if is_controller:
            _snapshot_nsys_reports()
            torch.cuda.cudart().cudaProfilerStart()
        if distributed:
            dist.barrier()
        _ACTIVE_NSYS_PHASE = name

    if capture_available:
        torch.cuda.nvtx.range_push(name)
        pushed_nvtx = True

    start = time.perf_counter()
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        elapsed = time.perf_counter() - start
        cleanup_error: Optional[Exception] = None
        try:
            if pushed_nvtx:
                torch.cuda.nvtx.range_pop()
        except Exception as exc:
            cleanup_error = exc
        if starts_capture:
            try:
                if distributed:
                    dist.barrier()
                if is_controller:
                    torch.cuda.cudart().cudaProfilerStop()
                    _rename_latest_nsys_report(
                        _phase_report_name(name, _get_node_rank(rank))
                    )
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                _ACTIVE_NSYS_PHASE = None
                try:
                    if distributed:
                        dist.barrier()
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        logger.info(f"[PHASE][{category}][cpu-wall] {name}: {elapsed:.2f}s")
        if cleanup_error is not None and not body_failed:
            raise cleanup_error
        if cleanup_error is not None:
            logger.error(
                f"Failed to clean up profiling phase {name!r}: {cleanup_error!r}"
            )


# NOTE: Since normally "switch_profile" is used in the training loop instead of inside the model,
# we don't have to make it compatible with torch.compile
def switch_profile(
    iter_id: int,
    start: int,
    end: int,
    profile_ranks: list[int] = [0],
    profile_type: ProfileType | list[ProfileType] = "nsys",
    event_name: Optional[str] = None,
    mem_snapshot_root: Optional[str] = None,
    mem_snapshot_name: Optional[str] = None,
    record_shape: bool = True,
    enable: bool = True,
    interval: int = 0,
    max_capture_ranges: int = 1000,
    exit_after_end_iter: bool = False,
):
    """
    Controls the profiler state based on the iteration number. Turns on profiling
    at the start iteration and turns it off at the end iteration.

    Args:
        iter_id (int): The current iteration number.
        start (int): The iteration number to start profiling.
        end (int): The iteration number to end profiling.
        profile_ranks (list[int]): List of ranks to be profiled.
            Defaults to [0] to profile only rank0.
        profile_type (ProfileType | list[ProfileType], optional):
            The profiler type or list of profiler types to be used.
            Supports "nsys" or "memory".
        event_name (str, optional): Custom name for the profiling event.
            If None, defaults to 'iter{iter_id}'.
        mem_snapshot_root (str, optional): Root directory for the memory snapshot file.
            If None, defaults to './mem_snapshot'.
        mem_snapshot_name (str, optional): Name of the memory snapshot file.
            If None, defaults to 'memshot_iter({start}-{end})_r{rank}',
            otherwise, appends '_r{rank}' to the provided name.
        record_shape (bool, optional): Whether to record the operand shape of each operation
            with `torch.autograd.profiler.emit_nvtx`,
            NOTE: this might increase the CPU overhead for extra recording,
            as well as much more recompilation when using torch.compile.
        enable (bool): Whether to enable profiling. Useful to unify the code with a flag to control.
        interval (int): Distance between profiling-window starts. A positive
            value repeats the original ``[start, end]`` window every
            ``interval`` iterations. Defaults to 0 for one-shot profiling.
        max_capture_ranges (int): Maximum number of profiling windows triggered
            by this function. A numeric Nsight ``repeat:N:sync`` capacity must
            be at least this value. Defaults to 1000.
        exit_after_end_iter (bool): Expect Nsight to end the profiling session
            after the profile window. This requires one-shot profiling and
            Nsight ``--capture-range-end=stop`` or ``stop-shutdown``. Process
            termination, if configured, is managed by Nsight rather than an
            explicit Python exit.

    Note:
        ``profile_ranks`` controls per-rank instrumentation. Because ``nsys``
        wraps the whole local ``torchrun`` process tree, only local rank 0 on
        each profiled node controls its node-level Nsight session. All ranks
        must call this function so its synchronization points can ensure that
        capture has started before training resumes and that all reports have
        been generated before training continues after ``end``.
    """
    # --- Checks and early returns / raises ---

    if not enable:
        return

    # The launcher-level Nsight wrapper is the high-level profiling switch.
    # When it is absent, disable both Nsight and memory profiling so users can
    # opt out without modifying every training configuration.
    nsys_command = _get_nsys_command()
    if nsys_command is None:
        return

    profile_window = _get_profile_window(iter_id, start, end, interval)
    if profile_window is None:
        return

    window_start, window_end = profile_window
    window_index = 0 if interval == 0 else (window_start - start) // interval
    if max_capture_ranges <= 0:
        raise ValueError("max_capture_ranges must be positive")
    if window_index >= max_capture_ranges:
        return

    distributed = dist.is_initialized()
    if not distributed:
        assert profile_ranks == [0], (
            "profile_ranks can only contain rank0 "
            "if ``torch.distributed`` is not initialized"
        )
        rank = 0
    else:
        rank = dist.get_rank()

    if not profile_ranks:
        return

    # --- Setup profiling environment ---

    global _PROFILER_ENABLED
    global _EMIT_NVTX_CTX

    event_name = f"iter_{iter_id}" if event_name is None else event_name
    mem_snapshot_root = (
        "./mem_snapshot"
        if mem_snapshot_root is None or mem_snapshot_root == ""
        else mem_snapshot_root
    )
    mem_snapshot_name = (
        f"memshot_iter({window_start}-{window_end})_r{rank}"
        if mem_snapshot_name is None or mem_snapshot_name == ""
        else (
            f"{mem_snapshot_name}_iter({window_start}-{window_end})_r{rank}"
            if interval > 0
            else f"{mem_snapshot_name}_r{rank}"
        )
    )
    profile_type = wrap_to_list(profile_type)  # type: ignore[assignment]
    need_profile_memory = "memory" in profile_type
    is_profile_rank = rank in profile_ranks
    need_profile_nsys = "nsys" in profile_type
    if need_profile_nsys:
        nsys_args, _ = nsys_command
        _validate_nsys_command(
            nsys_args,
            exit_after_end_iter=exit_after_end_iter,
            interval=interval,
            max_capture_ranges=max_capture_ranges,
        )
    nsys_enabled_for_window = need_profile_nsys
    is_nsys_controller = nsys_enabled_for_window and _is_nsys_controller(
        rank, profile_ranks
    )

    global _NSYS_CAPTURE_LIMIT_WARNED
    if (
        is_nsys_controller
        and not exit_after_end_iter
        and interval > 0
        and not _NSYS_CAPTURE_LIMIT_WARNED
    ):
        warnings.warn(
            f"max_capture_ranges={max_capture_ranges} limits this run to that "
            "many profiling window(s); later periodic windows will be skipped.",
            stacklevel=2,
        )
        _NSYS_CAPTURE_LIMIT_WARNED = True

    if is_profile_rank and need_profile_memory:
        os.makedirs(mem_snapshot_root, exist_ok=True)

    # --- Enter profiling (Prologue) ---

    if iter_id == window_start:
        if nsys_enabled_for_window:
            try:
                if is_nsys_controller:
                    _snapshot_nsys_reports()
                    torch.cuda.cudart().cudaProfilerStart()
            finally:
                if distributed:
                    # No rank may enter the profiled iteration before every
                    # node-level Nsight session has started. Keep this in a
                    # finally block so a local CUDA API failure cannot strand
                    # the other ranks at this synchronization point.
                    dist.barrier()
            if is_profile_rank and record_shape:
                emit_nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                _EMIT_NVTX_CTX = emit_nvtx_ctx.__enter__()
            if is_profile_rank:
                torch.cuda.nvtx.range_push(event_name)

        if is_profile_rank and need_profile_memory:
            # Use a large-but-FINITE cap so we don't lose events in the short
            # [start, end] window while still bounding host memory. Recording is
            # explicitly stopped in the ``end`` branch below, so this cap is a
            # safety net rather than the primary guard against blowing up Resident Set Size (RSS).
            torch.cuda.memory._record_memory_history(max_entries=10_000_000)

        _PROFILER_ENABLED = is_profile_rank and (
            nsys_enabled_for_window or need_profile_memory
        )

    # --- Exit profiling (Epilogue) ---

    elif iter_id == window_end:
        profile_error: Optional[Exception] = None
        if nsys_enabled_for_window:
            cleanup_error: Optional[Exception] = None
            try:
                if is_profile_rank:
                    torch.cuda.nvtx.range_pop()
                if is_profile_rank and record_shape:
                    try:
                        _EMIT_NVTX_CTX.__exit__(  # type: ignore[union-attr]
                            None, None, None
                        )
                    finally:
                        _EMIT_NVTX_CTX = None
            except Exception as exc:
                cleanup_error = exc
            finally:
                if distributed:
                    # Wait until every instrumented rank has left the capture
                    # range before a controller stops its node-level session.
                    # A local NVTX cleanup error must not strand other ranks.
                    dist.barrier()

            stop_error: Optional[Exception] = None
            try:
                if is_nsys_controller:
                    # With capture-range-end=repeat:N:sync this blocks until
                    # the node's report has been generated, without shutting
                    # down the target application.
                    torch.cuda.cudart().cudaProfilerStop()
                    if not exit_after_end_iter:
                        _rename_nsys_report(
                            window_start, window_end, _get_node_rank(rank)
                        )
            except Exception as exc:
                stop_error = exc
            finally:
                if distributed:
                    # A faster node must not resume training collectives while
                    # another node is still generating its report.
                    dist.barrier()

            profile_error = cleanup_error or stop_error

        if is_profile_rank and need_profile_memory:
            # Always turn recording OFF once the window ends -- even if the dump
            # fails -- otherwise memory history keeps accumulating on the host
            # for the rest of training (unbounded RSS growth / slowdown).
            memory_error: Optional[Exception] = None
            try:
                torch.cuda.memory._dump_snapshot(
                    os.path.join(mem_snapshot_root, f"{mem_snapshot_name}.pickle")
                )
            except Exception as exc:
                memory_error = exc
            finally:
                try:
                    torch.cuda.memory._record_memory_history(enabled=None)
                except Exception as exc:
                    if memory_error is None:
                        memory_error = exc
                finally:
                    if distributed:
                        # Snapshot failure must not strand non-profile ranks.
                        dist.barrier()
            if profile_error is None:
                profile_error = memory_error
        elif distributed and need_profile_memory:
            dist.barrier()

        _PROFILER_ENABLED = False
        if profile_error is not None:
            raise profile_error

    # --- Engage profiling (Mainloop) ---

    elif is_profile_rank and iter_id > window_start and iter_id < window_end:
        if nsys_enabled_for_window:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(event_name)


@torch.library.custom_op("magi_attn::nvtx_range_push", mutates_args=())
def nvtx_range_push(event_name: str) -> None:
    """torch.ops.magi_attn.nvtx_range_push"""
    torch.cuda.nvtx.range_push(event_name)


@nvtx_range_push.register_fake
def _(event_name: str) -> None:
    pass


@torch.library.custom_op("magi_attn::nvtx_range_pop", mutates_args=())
def nvtx_range_pop() -> None:
    """torch.ops.magi_attn.nvtx_range_pop"""
    torch.cuda.nvtx.range_pop()


@nvtx_range_pop.register_fake
def _() -> None:
    pass


# NOTE: since torch.compile does not support @contextlib.contextmanager,
# we use the class-based context manager
class add_nvtx_event:
    """
    Context manager to add an NVTX event around a code block.

    The recorded range name is automatically prefixed with ``NVTX_EVENT_PREFIX``
    for a consistent naming convention.

    Args:
        event_name (str): The name of the event to be recorded.
    """

    def __init__(self, event_name: str):
        self.enter_name = f"{NVTX_EVENT_PREFIX}{event_name}"

    def __enter__(self):
        if torch.compiler.is_compiling():
            # NOTE: torch.compile supports neither retrieving the attributes from "self"
            # nor modifying a variable not in the current scope
            # so we have no choice but assign a constant event name when compiling
            nvtx_range_push("MagiAttention::torch compile region")
        else:
            torch.cuda.nvtx.range_push(self.enter_name)
        return self

    def __exit__(self, *excinfo):
        if torch.compiler.is_compiling():
            nvtx_range_pop()
        else:
            torch.cuda.nvtx.range_pop()


@overload
def instrument_nvtx(func: F) -> F:
    ...


@overload
def instrument_nvtx(*, label_fn: Callable[..., str]) -> Callable[[F], F]:
    ...


def instrument_nvtx(
    func: Optional[F] = None,
    *,
    label_fn: Optional[Callable[..., str]] = None,
) -> Any:
    """
        Decorator that records an NVTX range for the duration of the function call.

        The range name defaults to the function's qualified name. It can be
        customized in two ways (both ignored while compiling, where a constant name
        is used instead):

        - Protocol: if the wrapped callable's first positional argument (e.g. ``self``)
            exposes a ``_nvtx_label(name)`` method, it is used to derive the label.
            This lets instance methods append context such as their module FQN.
        - ``label_fn``: an explicit callable ``label_fn(name, *args, **kwargs) -> str``
            that builds the label from the call arguments. Useful for staticmethods
            where the ``_nvtx_label`` protocol is not available.

    Args:
        func (Callable): The function to be decorated (bare usage).
        label_fn (Callable, optional): Explicit label builder (parameterized usage).

    Returns:
        Callable: The wrapped function that is now being profiled.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            if torch.compiler.is_compiling():
                # NOTE: we can access neither func.__qualname__ nor instance
                # hooks when compiling, so fall back to func.__name__
                label = fn.__name__
            elif label_fn is not None:
                label = label_fn(fn.__qualname__, *args, **kwargs)
            elif args and hasattr(args[0], "_nvtx_label"):
                label = args[0]._nvtx_label(fn.__qualname__)
            else:
                label = fn.__qualname__

            with add_nvtx_event(label):
                ret_val = fn(*args, **kwargs)
            return ret_val

        return cast(F, wrapped_fn)

    if func is not None:
        # Bare usage: @instrument_nvtx
        return decorator(func)

    # Parameterized usage: @instrument_nvtx(label_fn=...)
    return decorator


class NvtxModuleMixin:
    """
    Mixin for ``nn.Module`` that supplies an NVTX label carrying the module's
    fully-qualified name (FQN) within the model tree.

    Combine it with ``@instrument_nvtx`` on ``forward`` (or any method) so the
    recorded range reads e.g. ``LlamaLikeBlock.forward (layers.0)``. Call
    ``assign_module_nvtx_fqns(model)`` once after building the model to populate
    each module's FQN (defaults to an empty FQN, i.e. no suffix, until then).
    """

    _nvtx_fqn: str = ""

    def _nvtx_label(self, name: str) -> str:
        return f"{name} ({self._nvtx_fqn})" if self._nvtx_fqn else name


def assign_module_nvtx_fqns(model: "torch.nn.Module") -> None:
    """
    Populate each ``NvtxModuleMixin`` submodule's ``_nvtx_fqn`` with its
    fully-qualified name from ``model.named_modules()`` so that NVTX ranges can
    distinguish otherwise identically-named modules (e.g. per-layer blocks).
    """
    for fqn, module in model.named_modules():
        if isinstance(module, NvtxModuleMixin):
            module._nvtx_fqn = fqn
