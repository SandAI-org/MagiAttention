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


def is_cpp_backend_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable C++ backend
    for core data structures (AttnRange, AttnMaskType, etc.)
    and fall back to Python implementation.

    Default value is ``0``
    """
    return os.environ.get("MAGI_ATTENTION_CPP_BACKEND", "0") == "1"


from . import enum, jit, range_op  # noqa: E402
from .mask import AttnMask  # noqa: E402
from .range import AttnRange, RangeError  # noqa: E402
from .ranges import AttnRanges  # noqa: E402
from .rectangle import AttnRectangle  # noqa: E402
from .rectangles import AttnRectangles  # noqa: E402

# Try to use C++ extensions for core data structures to avoid Python overhead
# The submodules (range, ranges, rectangle, rectangles, enum) already handle
# the C++ backend replacement internally. We just need to set USE_CPP_BACKEND
# for informational purposes and external visibility.

USE_CPP_BACKEND = False
if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRange as _CppAttnRange

        if AttnRange is _CppAttnRange:  # type: ignore[comparison-overlap]
            USE_CPP_BACKEND = True
    except ImportError:
        pass

__all__ = [
    "enum",
    "jit",
    "AttnMask",
    "AttnRange",
    "RangeError",
    "AttnRanges",
    "AttnRectangle",
    "AttnRectangles",
    "range_op",
    "USE_CPP_BACKEND",
    "is_cpp_backend_enable",
]
