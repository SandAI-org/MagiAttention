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

import os
import time
from contextlib import ContextDecorator, contextmanager
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Union

import torch

from magi_attention.utils import wrap_to_list


@contextmanager
def switch_envvar_context(
    envvar_name: str | list[str],
    enable: bool = True,
    enable_value: str = "1",
    disable_value: str = "0",
):
    envvar_name_list = wrap_to_list(envvar_name)
    old_value_list = []

    for envvar_name in envvar_name_list:
        old_value = os.environ.get(envvar_name, None)
        os.environ[envvar_name] = enable_value if enable else disable_value
        old_value_list.append(old_value)

    yield

    for envvar_name, old_value in zip(envvar_name_list, old_value_list):
        if old_value is not None:
            os.environ[envvar_name] = old_value
        else:
            del os.environ[envvar_name]


def switch_envvar_decorator(
    envvar_name: str | list[str] | None = None,
    enable: bool = True,
    enable_value: str = "1",
    disable_value: str = "0",
):
    def decorator(
        func=None,
        *,
        envvar_name=envvar_name,
        enable=enable,
        enable_value=enable_value,
        disable_value=disable_value,
    ):
        if func is None:
            return partial(
                decorator,
                envvar_name=envvar_name,
                enable=enable,
                enable_value=enable_value,
                disable_value=disable_value,
            )
        assert envvar_name is not None

        @wraps(func)
        def wrapper(*args, **kwargs):
            with switch_envvar_context(
                envvar_name=envvar_name,
                enable=enable,
                enable_value=enable_value,
                disable_value=disable_value,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


switch_deterministic_mode_context = partial(
    switch_envvar_context, envvar_name="MAGI_ATTENTION_DETERMINISTIC_MODE"
)

switch_deterministic_mode_decorator = switch_envvar_decorator(
    envvar_name="MAGI_ATTENTION_DETERMINISTIC_MODE"
)

switch_sdpa_backend_context = partial(
    switch_envvar_context, envvar_name="MAGI_ATTENTION_SDPA_BACKEND"
)

switch_sdpa_backend_decorator = switch_envvar_decorator(
    envvar_name="MAGI_ATTENTION_SDPA_BACKEND"
)

switch_kernel_backend_context = partial(
    switch_envvar_context, envvar_name="MAGI_ATTENTION_KERNEL_BACKEND"
)

switch_kernel_backend_decorator = switch_envvar_decorator(
    envvar_name="MAGI_ATTENTION_KERNEL_BACKEND"
)

switch_ffa_verbose_jit_build_context = partial(
    switch_envvar_context, envvar_name="MAGI_ATTENTION_BUILD_VERBOSE"
)

switch_ffa_verbose_jit_build_decorator = switch_envvar_decorator(
    envvar_name="MAGI_ATTENTION_BUILD_VERBOSE"
)


def switch_envvars(
    envvar_name_list: list[str],
    enable_dict: dict[str, bool] = {},
    enable_value_dict: dict[str, str] = {},
    disable_value_dict: dict[str, str] = {},
) -> Callable[[], None]:
    """Switch the given list of environment variables
    to the corr. given value indicated by the list of `enable`,
    and return a call back function to switch them back.

    NOTE: this is helpful when switching multiple environment variables at once,
    as chaining several ``switch_envvar_context`` calls in a single ``with`` statement
    can become unwieldy and lead to excessive indentation.

    Args:
        envvar_name_list (list[str]): the list of environment variables to switch.
        enable_dict (dict[str, bool], optional): the value to set the
            environment variable to when it is enabled.
            If not found, we use the default value ``True``. Defaults to ``{}``.
        enable_value_dict (dict[str, str], optional): the value to set the
            environment variable to when it is enabled.
            If not found, we use the default value "1". Defaults to ``{}``.
        disable_value_dict (dict[str, str], optional): the value to set the
            environment variable to when it is disabled. Defaults to ``{}``.

    Returns:
        Callable[[], None]: the call back function to switch the environment variables back
    """

    enable_list = [
        enable_dict.get(envvar_name, True) for envvar_name in envvar_name_list
    ]
    enable_value_list = [
        enable_value_dict.get(envvar_name, "1") for envvar_name in envvar_name_list
    ]
    disable_value_list = [
        disable_value_dict.get(envvar_name, "0") for envvar_name in envvar_name_list
    ]

    ctx_mgr_lis = []
    for envvar_name, enable, enable_value, disable_value in zip(
        envvar_name_list, enable_list, enable_value_list, disable_value_list
    ):
        ctx_mgr = switch_envvar_context(
            envvar_name=envvar_name,
            enable=enable,
            enable_value=enable_value,
            disable_value=disable_value,
        )
        ctx_mgr.__enter__()
        ctx_mgr_lis.append(ctx_mgr)

    def switch_envvars_back():
        for ctx_mgr in reversed(ctx_mgr_lis):
            ctx_mgr.__exit__(None, None, None)

    return switch_envvars_back


class switch_env(ContextDecorator):
    """
    A unified utility that acts as both a Context Manager and a Decorator to temporarily
    modify environment variables.

    It supports:
    1. Boolean toggles (converting True/False to "1"/"0").
    2. Enum values (extracting the value property).
    3. dictionary inputs for setting multiple variables at once.
    4. Automatic restoration of previous environment states.

    Args:
        vars_or_name (str | dict[str, Any]): The name of the environment variable,
            or a dictionary of {variable_name: value}.
        value (Any, optional): The value to set if `vars_or_name` is a string.
            - If bool: converts to "1" or "0".
            - If Enum: uses enum.value.
            - Otherwise: converts to string.
            Defaults to None (implies `vars_or_name` is a dict).

    Usage:
        >>> # 1. As a Context Manager (Simple Boolean)
        >>> with switch_env("DEBUG", True):
        ...     pass
        ...
        >>> # 2. As a Decorator (Enum Backend)
        >>> class MyBackend(Enum):
        ...     FLASH = "flash_attn"
        ...
        >>> @switch_env("ATTN_BACKEND", MyBackend.FLASH)
        ... def run_model():
        ...     pass
        ...
        >>> # 3. Multiple Variables (dict)
        >>> with switch_env({"FEATURE_A": "1", "BACKEND": "native"}):
        ...     pass
    """

    def __init__(self, vars_or_name: Union[str, dict[str, Any]], value: Any = None):
        self.original_values: dict[str, Union[str, None]] = {}
        self.target_values: dict[str, str] = {}

        # Normalize input into a dictionary of {name: str_value}
        if isinstance(vars_or_name, dict):
            raw_targets = vars_or_name
        else:
            if value is None:
                raise ValueError(
                    "If 'vars_or_name' is a string, 'value' must be provided."
                )
            raw_targets = {vars_or_name: value}

        for k, v in raw_targets.items():
            self.target_values[k] = self._normalize_value(v)

    def _normalize_value(self, value: Any) -> str:
        """Helper to convert various types to environment variable string format."""
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    def __enter__(self):
        # Save original state and apply new values
        for name, new_val in self.target_values.items():
            self.original_values[name] = os.environ.get(name, None)
            os.environ[name] = new_val
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original state
        for name, old_val in self.original_values.items():
            if old_val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_val


def poll_cuda_event(
    *,
    stream: torch.cuda.Stream | None = None,
    timeout: float = 30.0,
    poll_interval: float = 0.05,
    error_msg: str = "CUDA hang detected",
) -> None:
    """Record a CUDA event on *stream* and CPU-poll until it completes.

    Raises :class:`RuntimeError` if the event does not complete within
    *timeout* seconds, allowing hangs to be detected quickly without
    waiting for the (much longer) distributed process-group timeout.

    Args:
        stream (torch.cuda.Stream | None, optional): Stream on which to record
            the event.  ``None`` records on the current default CUDA stream.
            Defaults to ``None``.
        timeout (float, optional): Maximum number of seconds to wait.
            Defaults to ``30.0``.
        poll_interval (float, optional): Seconds to sleep between
            :meth:`~torch.cuda.Event.query` polls.  Defaults to ``0.05``.
        error_msg (str, optional): Message for the :class:`RuntimeError`
            raised on timeout.  Defaults to ``"CUDA hang detected"``.
    """
    evt = torch.cuda.Event()
    if stream is None:
        evt.record()
    else:
        evt.record(stream)
    t0 = time.monotonic()
    while not evt.query():
        if time.monotonic() - t0 > timeout:
            raise RuntimeError(error_msg)
        time.sleep(poll_interval)
