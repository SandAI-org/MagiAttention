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

import os
from contextlib import contextmanager
from functools import partial, wraps
from typing import Callable

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
