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

from magi_attention.utils import wrap_to_list


@contextmanager
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
