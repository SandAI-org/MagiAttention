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
import warnings

from . import comm, config, env, functional
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


__all__ = [
    # Sub-packages
    "config",
    "comm",
    "env",
    "functional",
    "magi_attn_ext",
    "magi_attn_comm",
    # Runtime initialisation
    "init_dist_attn_runtime_key",
    "init_dist_attn_runtime_mgr",
]
