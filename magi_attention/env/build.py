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

"""JIT / build / cache environment variables for magi_attention."""

import os
import pathlib

__all__ = [
    "is_no_build_cache",
    "workspace_base_dir",
    "is_force_jit_build",
    "is_build_verbose",
    "is_build_debug",
    "nvcc_threads",
]


def is_no_build_cache() -> bool:
    """
    Toggle this env variable to ``1`` to disable JIT build caching
    for flex-flash-attention kernels.

    Default value is ``0`` (caching enabled).
    """
    return os.getenv("MAGI_ATTENTION_NO_BUILD_CACHE", "0") == "1"


def workspace_base_dir() -> str:
    """
    Set ``MAGI_ATTENTION_WORKSPACE_BASE`` to override the base directory
    for JIT compilation caches.

    Default value is ``$HOME``.
    """
    return os.getenv("MAGI_ATTENTION_WORKSPACE_BASE", pathlib.Path.home().as_posix())


def is_force_jit_build() -> bool:
    """
    Toggle this env variable to ``1`` to force JIT build even when AOT artifacts exist.

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_FORCE_JIT_BUILD", "0") == "1"


def is_build_verbose() -> bool:
    """
    Toggle this env variable to ``1`` to enable verbose output during JIT compilation.

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_BUILD_VERBOSE", "0") == "1"


def is_build_debug() -> bool:
    """
    Toggle this env variable to ``1`` to enable debug mode during JIT compilation.

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_BUILD_DEBUG", "0") == "1"


def nvcc_threads() -> str:
    """
    Set ``NVCC_THREADS`` to control the number of parallel NVCC compilation threads.

    Default value is ``4``
    """
    return os.getenv("NVCC_THREADS", "4")
