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

import importlib.util
import os
import warnings

from . import comm, config, functional
from .dist_attn_runtime_mgr import init_dist_attn_runtime_mgr

if importlib.util.find_spec("magi_attention._version") is None:
    warnings.warn(
        "You are using magi_attention without installing it. This may cause some unexpected errors."
    )
    version = None
else:
    from ._version import __version__ as git_version

    version = git_version

__version__: str | None = version

__all__ = [
    "init_dist_attn_runtime_mgr",
    "is_sanity_check_enable",
    "is_cuda_device_max_connections_one",
    "config",
    "comm",
    "functional",
]


def is_sanity_check_enable() -> bool:
    """
    Toggle this env variable to ``1`` can enable many sanity check codes inside magi_attention

    Default value is ``0``

    NOTE: this is only supposed to be used for testing or debugging,
    since the extra sanity-check overhead might be non-negligible
    """
    return os.environ.get("MAGI_ATTENTION_SANITY_CHECK", "0") == "1"


def is_sdpa_backend_enable() -> bool:
    """
    Toggle this env variable to ``1`` can switch the attn kernel backend
    from ffa to sdpa-math, to support higher precision like fp32 or fp64

    Default value is ``0``

    NOTE: this is only supposed to be used for testing or debugging,
    since the performance is not acceptable
    """
    return os.environ.get("MAGI_ATTENTION_SDPA_BACKEND", "0") == "1"


def is_cuda_device_max_connections_one() -> bool:
    """
    Check if "CUDA_DEVICE_MAX_CONNECTIONS" is set to ``1``,
    which will prevent the concurrency among multiple cuda streams
    """
    return os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8") == "1"


def is_deterministic_mode_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable deterministic mode
    to use deterministic algorithms for all magi_attention kernels

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_DETERMINISTIC_MODE", "0") == "1"


def dist_attn_runtime_dict_size() -> int:
    """
    Set the value of this env variable to control
    the size of ``dist_attn_runtime_dict``

    Default value is ``100``
    """
    return int(os.environ.get("MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "100"))
