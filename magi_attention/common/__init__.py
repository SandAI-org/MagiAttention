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

from magi_attention.env.general import is_cpp_backend_enable

from . import enum, jit, range_op  # noqa: E402
from .forward_meta import AttnForwardMeta  # noqa: E402
from .mask import AttnMask  # noqa: E402
from .range import RangeError  # noqa: E402

# ---------------------------------------------------------------------------
# Unified routing layer: C++ backend vs Python backend
#
# All switching logic is centralized here. Individual implementation files
# (range.py, ranges.py, etc.) contain only their pure Python implementations
# and do NOT perform any backend switching themselves.
# ---------------------------------------------------------------------------


USE_CPP_BACKEND = False

if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import (  # type: ignore[assignment]
            AttnMaskType,
            AttnRange,
            AttnRanges,
            AttnRectangle,
            AttnRectangles,
        )

        USE_CPP_BACKEND = True
    except ImportError:
        pass

if not USE_CPP_BACKEND:
    from .enum import AttnMaskType  # type: ignore[no-redef]  # noqa: F811
    from .range import AttnRange  # type: ignore[no-redef]  # noqa: F811
    from .ranges import AttnRanges  # type: ignore[no-redef]  # noqa: F811
    from .rectangle import AttnRectangle  # type: ignore[no-redef]  # noqa: F811
    from .rectangles import AttnRectangles  # type: ignore[no-redef]  # noqa: F811


__all__ = [
    "enum",
    "jit",
    "AttnMask",
    "AttnMaskType",
    "AttnRange",
    "RangeError",
    "AttnRanges",
    "AttnForwardMeta",
    "AttnRectangle",
    "AttnRectangles",
    "range_op",
    "USE_CPP_BACKEND",
    "is_cpp_backend_enable",
]
