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

from __future__ import annotations

import importlib.util
import logging
import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common.enum import MagiAttentionKernelBackend, MagiAttentionPrecision

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


def kernel_backend() -> "MagiAttentionKernelBackend":
    """
    Set env variable ``MAGI_ATTENTION_KERNEL_BACKEND`` to choose the attn kernel backend.

    Valid values: ``"ffa"`` (default), ``"sdpa"``, ``"sdpa_ol"``, ``"fa4"``

    - ``ffa``: flex-flash-attention (default, high-performance persistent kernel)
    - ``sdpa``: offline SDPA implementation (for testing / high precision like fp32/fp64)
    - ``sdpa_ol``: online (block-wise) SDPA implementation (for testing, lower memory than sdpa)
    - ``fa4``: Flash-Attention 4 monkey-patch (workaround for Blackwell GPUs)

    Backward compatibility: the legacy env vars ``MAGI_ATTENTION_SDPA_BACKEND=1``
    and ``MAGI_ATTENTION_FA4_BACKEND=1`` are still supported, but must NOT be set
    at the same time as ``MAGI_ATTENTION_KERNEL_BACKEND``.
    """
    from magi_attention.common.enum import MagiAttentionKernelBackend

    has_unified = os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND") is not None
    has_legacy_sdpa = os.environ.get("MAGI_ATTENTION_SDPA_BACKEND", "0") == "1"
    has_legacy_fa4 = os.environ.get("MAGI_ATTENTION_FA4_BACKEND", "0") == "1"

    assert not (has_unified and (has_legacy_sdpa or has_legacy_fa4)), (
        "MAGI_ATTENTION_KERNEL_BACKEND cannot be set together with the legacy "
        "MAGI_ATTENTION_SDPA_BACKEND / MAGI_ATTENTION_FA4_BACKEND env vars. "
        "Please use only MAGI_ATTENTION_KERNEL_BACKEND."
    )
    assert not (has_legacy_sdpa and has_legacy_fa4), (
        "MAGI_ATTENTION_SDPA_BACKEND and MAGI_ATTENTION_FA4_BACKEND "
        "cannot both be set to 1 at the same time."
    )

    if has_legacy_sdpa:
        return MagiAttentionKernelBackend.SDPA
    if has_legacy_fa4:
        return MagiAttentionKernelBackend.FA4

    raw = os.environ.get("MAGI_ATTENTION_KERNEL_BACKEND", "ffa").lower()
    return MagiAttentionKernelBackend(raw)


def precision() -> "MagiAttentionPrecision | None":
    """
    Set env variable ``MAGI_ATTENTION_PRECISION`` to override the compute dtype
    for attention kernels.

    Valid values: ``"bf16"``, ``"fp16"``, ``"fp32"``, ``"fp64"``

    When set, input Q/K/V are cast to the specified dtype before attention
    computation, and the output is cast back to the original input dtype.

    When unset (default), the input dtype is used as-is (no casting).
    """
    from magi_attention.common.enum import MagiAttentionPrecision

    raw = os.environ.get("MAGI_ATTENTION_PRECISION")
    if raw is None:
        return None
    return MagiAttentionPrecision(raw.lower())


def is_cuda_device_max_connections_one() -> bool:
    """
    Check if "CUDA_DEVICE_MAX_CONNECTIONS" is set to ``1``,
    which will prevent the concurrency among multiple cuda streams

    Default value is ``8``
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
    Toggle this env variable to ``1`` to enable CatGQA mode for flex-flash-attention backward,
    to further optimize the performance under GQA settings
    by concatenating multiple Q heads sharing the same KV head.

    Default value is ``0``

    NOTE: this feature is experimental and under active development for now,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_CATGQA", "0") == "1"


def dist_attn_backward_hide_tail_reduce() -> bool:
    """
    Toggle this env variable to ``1`` to trade saving the last remote `kv`
    activation for reordering overlap stages during backward,
    hiding the final remote `group_reduce` with the host FFA stage

    Default value is ``0``

    NOTE: this feature is experimental and under active development for now,
    and not compatible with many other features like qo comm,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_BWD_HIDE_TAIL_REDUCE", "0") == "1"


def dist_attn_runtime_dict_size() -> int:
    """
    Set the value of this env variable to control
    the maximum LRU cache size of ``dist_attn_runtime_dict_mgr``

    Default value is ``1000``
    """
    return int(os.environ.get("MAGI_ATTENTION_DIST_ATTN_RUNTIME_DICT_SIZE", "1000"))


def min_chunks_per_rank() -> int:
    """
    Set the value of this env variable to control
    the minimum number of chunks per context parallel rank, 
    to control the granularity of computational load-balance.

    Default value is ``8``
    """
    return int(os.environ.get("MAGI_ATTENTION_MIN_CHUNKS_PER_RANK", "8"))


__all__ = [
    "init_dist_attn_runtime_key",
    "init_dist_attn_runtime_mgr",
    "kernel_backend",
    "precision",
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
