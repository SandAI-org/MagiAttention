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

import contextlib
import functools
import itertools
import os
import re
from functools import partial, wraps
from typing import Any, Callable

import torch
import torch.distributed as dist

from . import common_fsdp
from .common_fsdp import str2seed

__all__ = [
    "common_fsdp",
    "parameterize",
    "calc_inf_norm",
    "switch_envvar_decorator",
]


def parameterize(argument: str, values: list[Any]) -> Callable:
    """
    This function simulates pytest.mark.parameterize with multi-process support.

    Default Behavior (Replication Mode):
    In a distributed environment, every rank will execute every single test case.
    This is necessary for tests that require collective communication.

    Optional Behavior (Distribution Mode):
    If the test function is decorated with `@distribute_parameterized_test_cases`,
    the test cases will be split among the available ranks. This is ideal for
    speeding up tests where each case is independent.

    This version implements "fail-fast": the run stops on the first failure.

    Args:
        argument (str): The name of the argument to parameterize.
        values (list[Any]): A list of values for this argument.
    """

    def _wrapper(func: Callable):
        # Decorators are applied from the inside out (bottom-up). We check if the
        # wrapped function (func) already has an _param_info attribute. If so, it
        # means it has been processed by an inner parameterize decorator.
        inner_params = getattr(func, "_param_info", [])
        all_params = [(argument, values)] + inner_params

        # Trace back to find the original, unwrapped test function.
        # If func doesn't have _original_func, it means func itself is the original one.
        original_func = getattr(func, "_original_func", func)

        @functools.wraps(func)
        def _parameterized_func(*args, **kwargs):
            # Only the outermost decorator will execute this logic. The _parameterized_func
            # from inner decorators is never called directly; it only serves as a carrier
            # for parameter info.

            # --- Distributed Setup --- #

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                is_dist_setup = True
            else:
                rank = 0
                world_size = 1
                is_dist_setup = False

            # --- BEHAVIOR CONTROL --- #

            # Check the environment variable to decide the execution mode.
            # Defaults to '0' (replication mode) if the var is not set.
            is_run_in_mp = is_dist_setup and os.environ.get("RUN_IN_MP", "0") == "1"

            # --- Test Case Generation and Execution --- #
            arg_names = [name for name, _ in all_params]
            value_lists = [vals for _, vals in all_params]
            all_combinations = itertools.product(*value_lists)

            for test_case_id, combination in enumerate(all_combinations):
                # --- Work Distribution/Replication Logic --- #

                # Only apply the distribution logic if the mode is enabled AND we are in a multi-rank setting.
                if is_run_in_mp:
                    # since we might jump some combinations inside the test function,
                    # thus using test_case_id as the assign_id might encounter work imbalance among ranks
                    # assign_id = test_case_id
                    assign_id = str2seed(str(combination))
                    if assign_id % world_size != rank:
                        continue

                # In replication mode (default), this block is skipped, and every rank runs the code below.
                current_params_kwargs = dict(zip(arg_names, combination))
                final_kwargs = {**kwargs, **current_params_kwargs}

                try:
                    # Directly call the original function with the current set of parameters.
                    original_func(*args, **final_kwargs)
                except Exception as e:
                    # If an exception occurs, we format a comprehensive error message
                    # and re-raise immediately, which stops the execution.
                    param_str_list = []
                    for name, value_list in all_params:
                        current_val = current_params_kwargs[name]
                        try:
                            val_idx = value_list.index(current_val)
                            param_str_list.append(f"{name}[{val_idx}]")
                        except ValueError:
                            # If the value is not in the list (e.g., for complex objects),
                            # display the value directly.
                            param_str_list.append(f"{name}={current_val}")

                    error_header = " x ".join(param_str_list)
                    error_msg = "".join(
                        [
                            "\n-->",
                            f" [Rank {rank}] " if is_dist_setup else " ",
                            f"Test case failed with parameters: {error_header}\n",
                            f"    {type(e).__name__}: {e}",
                        ]
                    )

                    # Re-raise the original exception type with the new, clean message.
                    # 'from e' preserves the original traceback for better debugging.
                    raise type(e)(error_msg) from e

        # Attach metadata to the newly created wrapper function for outer decorators to use.
        _parameterized_func._param_info = all_params  # type: ignore[attr-defined]
        _parameterized_func._original_func = original_func  # type: ignore[attr-defined]

        return _parameterized_func

    return _wrapper


@torch.no_grad
def calc_inf_norm(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    return (a.float() - b.float()).norm(p=float("inf")).item()


def extract_mismatch_info(error_msg: str) -> tuple[int, int, float]:
    match = re.search(r"Mismatched elements: (\d+) / (\d+)", error_msg)

    if match:
        mismatched_elements = int(match.group(1))
        total_elements = int(match.group(2))
        mismatch_ratio = mismatched_elements / total_elements
        return mismatched_elements, total_elements, mismatch_ratio
    else:
        raise ValueError(f"Could not find mismatch elements in {error_msg=}")


@torch.no_grad
def extract_mismatch_threshold(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    mismatch_thres_ratio: float = 1.0,
) -> float:
    mismatch_threshold = 0.0
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as e:
        error_msg = str(e)
        _, _, mismatch_threshold = extract_mismatch_info(error_msg)

    # scale it by `mismatch_thres_ratio`, and clamp it in [0, 1]
    return min(max(mismatch_threshold * mismatch_thres_ratio, 0.0), 1.0)


@torch.no_grad
def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mismatch_threshold: float = 0,
    test_case: str = "",
) -> None:
    assert (
        0 <= mismatch_threshold <= 1
    ), f"{mismatch_threshold=} must be between 0 and 1"
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        no_mismatch_info = f"[{test_case}]: has no mismatch"
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(no_mismatch_info)
        else:
            print(no_mismatch_info)
    except AssertionError as e:
        error_msg = str(e)
        mismatched_elements, total_elements, mismatch_ratio = extract_mismatch_info(
            error_msg
        )

        mismatch_info = (
            f"[{test_case}]: mismatch_ratio = {mismatched_elements} / {total_elements} "
            f"= {mismatch_ratio * 100:.4f} % | mismatch_threshold={mismatch_threshold * 100:.2f} %"
        )

        if mismatch_ratio <= mismatch_threshold:
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    print(mismatch_info)
            else:
                print(mismatch_info)
            return
        else:
            raise type(e)(
                f"\n>>>>>>>  Torch Error Message: \n\n{error_msg}\n\n"
                f">>>>>>>  Mismatch Detailed Info: \n\n{mismatch_info}\n\n"
            ) from e


def wrap_to_list(x: Any, broadcast_to_length: int = 1) -> list[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x] * broadcast_to_length


@contextlib.contextmanager
def switch_envvar_context(envvar_name: str | list[str], enable: bool = True):
    envvar_name_list = wrap_to_list(envvar_name)
    old_value_list = []

    for envvar_name in envvar_name_list:
        old_value = os.environ.get(envvar_name, None)
        os.environ[envvar_name] = "1" if enable else "0"
        old_value_list.append(old_value)

    yield

    for envvar_name, old_value in zip(envvar_name_list, old_value_list):
        if old_value is not None:
            os.environ[envvar_name] = old_value
        else:
            del os.environ[envvar_name]


def switch_envvar_decorator(
    envvar_name: str | list[str] | None = None, enable: bool = True
):
    def decorator(func=None, *, envvar_name=envvar_name, enable=enable):
        if func is None:
            return partial(decorator, envvar_name=envvar_name, enable=enable)
        assert envvar_name is not None

        @wraps(func)
        def wrapper(*args, **kwargs):
            with switch_envvar_context(envvar_name=envvar_name, enable=enable):
                return func(*args, **kwargs)

        return wrapper

    return decorator
