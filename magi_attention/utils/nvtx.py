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

from functools import wraps
from typing import Any, Callable, TypeVar, cast

import torch

# fixed the mypy type check missing bug
# when a func is wrapped
# issue: https://stackoverflow.com/questions/65621789/mypy-untyped-decorator-makes-function-my-method-untyped
F = TypeVar("F", bound=Callable[..., Any])

# global var for torch.autograd.profiler.emit_nvtx
_EMIT_NVTX_CTX: None | torch.autograd.profiler.emit_nvtx = None


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

    Args:
        event_name (str): The name of the event to be recorded.
    """

    def __init__(self, event_name: str):
        self.enter_name = event_name

    def __enter__(self):
        if torch.compiler.is_compiling():
            # NOTE: torch.compile supports neither retrieving the attributes from "self"
            # nor modifying a variable not in the current scope
            # so we have no choice but assign a constant event name when compiling
            nvtx_range_push("torch compile region")
        else:
            torch.cuda.nvtx.range_push(self.enter_name)
        return self

    def __exit__(self, *excinfo):
        if torch.compiler.is_compiling():
            nvtx_range_pop()
        else:
            torch.cuda.nvtx.range_pop()


def instrument_nvtx(func: F) -> F:
    """
    Decorator that records an NVTX range for the duration of the function call.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function that is now being profiled.
    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs):
        if torch.compiler.is_compiling():
            # NOTE: we can not access func.__qualname__ when compiling
            # thus use func.__name__ instead
            func_name = func.__name__
        else:
            func_name = func.__qualname__

        with add_nvtx_event(func_name):
            ret_val = func(*args, **kwargs)
        return ret_val

    return cast(F, wrapped_fn)


# NOTE: since normally "switch_profile" is used in the training loop instead of inside the model,
# we don't have to make it compatible with torch.compile
def switch_profile(
    iter_id: int,
    start: int,
    end: int,
    profile_ranks: list[int],
    event_name: str | None = None,
    record_shape: bool = True,
) -> None:
    """
    Controls the profiler state based on the iteration number. Turns on profiling
    at the start iteration and turns it off at the end iteration.

    Args:
        iter_id (int): The current iteration number.
        start (int): The iteration number to start profiling.
        end (int): The iteration number to end profiling.
        profile_ranks (list[int]): List of ranks to be profiled.
        event_name (str, optional): Custom name for the profiling event. If None, defaults to 'iter{iter_id}'.
        record_shape (bool, optional): Whether to record the operand shape of each operation
            with `torch.autograd.profiler.emit_nvtx`,
            NOTE: this might increase the CPU overhead for extra recording,
            as well as much more recompilation when using torch.compile.
    """

    if not torch.distributed.is_initialized():
        assert profile_ranks == [
            0
        ], "profile_ranks can only contain rank0 if torch.distributed is not initialized"
    else:
        if torch.distributed.get_rank() not in profile_ranks:
            return

    global _EMIT_NVTX_CTX

    if event_name is None:
        event_name = f"iter{iter_id}"

    # Start profiling
    if iter_id == start:
        if record_shape:
            emit_nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
            _EMIT_NVTX_CTX = emit_nvtx_ctx.__enter__()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(event_name)

    # Stop profiling
    elif iter_id == end:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        if record_shape:
            _EMIT_NVTX_CTX.__exit__(None, None, None)  # type: ignore[union-attr]
            _EMIT_NVTX_CTX = None

    # Continue profiling
    elif iter_id > start and iter_id < end:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(event_name)
