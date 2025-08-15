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

import functools
import itertools
from typing import Any, Callable

from . import dist_common
from .gt_dispatcher import GroundTruthDispatcher
from .precision import EPSILON, assert_close, torch_attn_ref

__all__ = [
    "dist_common",
    "GroundTruthDispatcher",
    "assert_close",
    "torch_attn_ref",
    "parameterize",
    "EPSILON",
]


def parameterize(argument: str, values: list[Any]) -> Callable:
    """
    This function simulates the behavior of pytest.mark.parameterize.

    When multiple decorators are stacked, parameter information is collected and
    passed upwards. The outermost decorator becomes the sole executor, responsible
    for creating a Cartesian product of all parameters and running the tests.

    This version implements a "fail-fast" behavior: the test run stops
    immediately on the first failure, raising an exception with the full
    parameter context of the failing test case. The error message is clean
    and not nested.

    If a function is wrapped with this wrapper, non-parametrized arguments must be
    keyword arguments; positional arguments are not allowed.

    Args:
        argument (str): The name of the argument to parameterize.
        values (list[Any]): A list of values to iterate for this argument.
    """

    def _wrapper(func: Callable):
        # Decorators are applied from the inside out (bottom-up). We check if the
        # wrapped function (func) already has an _param_info attribute. If so, it
        # means it has been processed by an inner parameterize decorator.
        inner_params = getattr(func, "_param_info", [])

        # Prepend the current decorator's parameter info to the list.
        all_params = [(argument, values)] + inner_params

        # Trace back to find the original, unwrapped test function.
        # If func doesn't have _original_func, it means func itself is the original one.
        original_func = getattr(func, "_original_func", func)

        @functools.wraps(func)
        def _parameterized_func(*args, **kwargs):
            # Only the outermost decorator will execute this logic. The _parameterized_func
            # from inner decorators is never called directly; it only serves as a carrier
            # for parameter info.

            arg_names = [name for name, _ in all_params]
            value_lists = [vals for _, vals in all_params]

            # Create the Cartesian product of all parameter values.
            all_combinations = itertools.product(*value_lists)

            for combination in all_combinations:
                current_params_kwargs = dict(zip(arg_names, combination))
                final_kwargs = {**kwargs, **current_params_kwargs}

                # The try...except block is now INSIDE the loop.
                try:
                    # Directly call the original function with the current set of parameters.
                    original_func(*args, **final_kwargs)
                except Exception as e:
                    # If an exception occurs, we format a comprehensive error message
                    # and re-raise immediately, which stops the execution.
                    param_str_list = []
                    # Reverse the parameter order to match the visual decorator stacking
                    # order in the code (top-down).
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
                    error_msg = (
                        f"\n--> Test case failed with parameters: {error_header}\n"
                        f"    {type(e).__name__}: {e}"
                    )

                    # Re-raise the original exception type with the new, clean message.
                    # 'from e' preserves the original traceback for better debugging.
                    raise type(e)(error_msg) from e

        # Attach metadata to the newly created wrapper function for outer decorators to use.
        _parameterized_func._param_info = all_params  # type: ignore[attr-defined]
        _parameterized_func._original_func = original_func  # type: ignore[attr-defined]

        return _parameterized_func

    return _wrapper
