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

from magi_attention import is_cpp_backend_enable

from . import enum, jit, range_op
from .mask import AttnMask
from .range import AttnRange, RangeError
from .ranges import AttnRanges
from .rectangle import AttnRectangle
from .rectangles import AttnRectangles

# Try to use C++ extensions for core data structures to avoid Python overhead
# The submodules (range, ranges, rectangle, rectangles, enum) already handle
# the C++ backend replacement internally. We just need to set HAS_CPP_CORE
# for informational purposes and external visibility.

HAS_CPP_CORE = False
if is_cpp_backend_enable():
    try:
        from magi_attention.magi_attn_ext import AttnRange as _CppAttnRange

        if AttnRange is _CppAttnRange:
            HAS_CPP_CORE = True
    except ImportError:
        pass

if HAS_CPP_CORE:
    print(
        "[MagiAttention] INFO: Using C++ backend for core data structures (AttnRange, AttnMaskType, etc.)"
    )
elif is_cpp_backend_enable():
    print(
        "[MagiAttention] WARNING: C++ backend requested but not found, falling back to Python implementation"
    )
else:
    # NOTE: We don't print anything if it's disabled by default or explicitly to avoid noise
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
    "HAS_CPP_CORE",
]
