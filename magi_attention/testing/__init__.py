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
import os
from typing import Any, Callable

import torch.distributed as dist

from magi_attention.utils import str2seed

from . import dist_common, utils
from .dist_common import RUN_IN_MP
from .gt_dispatcher import GroundTruthDispatcher
from .precision import assert_close, ref_attn_func

__all__ = [
    "dist_common",
    "utils",
    "GroundTruthDispatcher",
    "assert_close",
    "ref_attn_func",
    "parameterize",
]


# HACK: enable sanity check in every unitest by default
# TODO: inherit the unitest.TestCaseBase and enable/disable it inside the base class's setUp/tearDown
os.environ["MAGI_ATTENTION_SANITY_CHECK"] = "1"


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
            is_run_in_mp = is_dist_setup and os.environ.get(RUN_IN_MP, "0") == "1"

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
