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

import importlib.util
import logging
import os
import warnings

from . import comm, config, functional
from .dist_attn_runtime_mgr import (
    init_dist_attn_runtime_key,
    init_dist_attn_runtime_mgr,
)

try:
    from . import magi_attn_ext  # type: ignore[attr-defined]  # noqa: F401
except ImportError as e:
    warnings.warn(
        f"Failed to import magi_attn_ext extension module. "
        f"Please make sure MagiAttention is properly installed. "
        f"Original error message: {e}"
    )

try:
    from . import magi_attn_comm  # type: ignore[attr-defined]  # noqa: F401
except ImportError as e:
    warnings.warn(
        f"Failed to import magi_attn_comm extension module. "
        f"Please make sure MagiAttention is properly installed. "
        f"Original error message: {e}"
    )

try:
    from . import (  # type: ignore[attr-defined] # noqa: F401
        flexible_flash_attention_utils_cuda,
    )
except ImportError as e:
    warnings.warn(
        f"Failed to import flexible_flash_attention_utils_cuda extension module. "
        f"Please make sure MagiAttention is properly installed. "
        f"Original error message: {e}"
    )


if importlib.util.find_spec("magi_attention._version") is None:
    warnings.warn(
        "You are using magi_attention without installing it. This may cause some unexpected errors."
    )
    version = None
else:
    from ._version import __version__ as git_version

    version = git_version

__version__: str | None = version

# Initialize a logger specific to this module/namespace
logger = logging.getLogger(__name__)

# Add a NullHandler to prevent logging warnings ("No handlers could be found...")
# if the application using this library hasn't configured logging.
logger.addHandler(logging.NullHandler())


def is_sanity_check_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable many sanity check codes inside magi_attention

    Default value is ``0``

    NOTE: this is only supposed to be used for testing or debugging,
    since the extra sanity-check overhead might be non-negligible
    """
    return os.environ.get("MAGI_ATTENTION_SANITY_CHECK", "0") == "1"


def is_flatten_head_groups_enable() -> bool:
    """
    Toggle this env variable to ``1`` to flatten head groups
    within GQA/MQA attention to optimize dynamic solver performance

    Default value is ``0``

    NOTE: this feature is experimental and under active development for now
    and not compatible with many other features,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_FLATTEN_HEAD_GROUPS", "0") == "1"


def is_sdpa_backend_enable() -> bool:
    """
    Toggle this env variable to ``1`` to switch the attn kernel backend
    from ffa to customized sdpa implementation,
    to support higher precision like fp32 or fp64

    Default value is ``0``

    NOTE: this is only supposed to be used for testing or debugging,
    since the performance is not acceptable
    """
    return os.environ.get("MAGI_ATTENTION_SDPA_BACKEND", "0") == "1"


def is_fa4_backend_enable() -> bool:
    """
    Toggle this env variable to ``1`` to switch the attn kernel backend
    from ffa to a monkey patch version of flash-attention 4 implementation,
    to temporarily support arbitrary mask on Blackwell GPUs

    Default value is ``0``

    NOTE: this is a beta feature, under development
    """
    return os.environ.get("MAGI_ATTENTION_FA4_BACKEND", "0") == "1"


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


def is_profile_mode_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable profiling mode
    to profile all magi_attention kernels, by now mainly for ffa kernels

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_PROFILE_MODE", "0") == "1"


def is_auto_range_merge_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable automatic range merging for flex-flash-attention,
    to improve performance by reducing the number of attention ranges

    Default value is ``0``

    NOTE: this feature is experimental and under active development for now,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_AUTO_RANGE_MERGE", "0") == "1"


def is_cat_gqa_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable CatGQA mode for flex-flash-attention,
    to further optimize the performance of GQA attention by concatenating multiple Q heads

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_CATGQA", "0") == "1"


def dist_attn_runtime_dict_size() -> int:
    """
    Set the value of this env variable to control
    the maximum LRU cache size of ``dist_attn_runtime_dict_mgr``

    Default value is ``1000``
    """
    return int(os.environ.get("MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "1000"))


def dist_attn_backward_hide_tail_reduce() -> bool:
    """
    Set the value of this env variable to control
    whether save the last stage for backward to get better overlaping

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE", "0") == "1"


__all__ = [
    "init_dist_attn_runtime_key",
    "init_dist_attn_runtime_mgr",
    "is_sanity_check_enable",
    "is_flatten_head_groups_enable",
    "is_cuda_device_max_connections_one",
    "dist_attn_runtime_dict_size",
    "config",
    "comm",
    "functional",
    "magi_attn_ext",
    "magi_attn_comm",
    "flexible_flash_attention_utils_cuda",
]
