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

import importlib
import os
import sys

import pytest


def _reload_common_modules():
    """Reload magi_attention.common so the backend switch takes effect."""
    if "magi_attention.common" in sys.modules:
        importlib.reload(sys.modules["magi_attention.common"])


def _has_cpp_backend():
    """Check if the C++ backend is available."""
    try:
        import magi_attention.magi_attn_ext  # noqa: F401

        return True
    except ImportError:
        return False


_cpp_available = _has_cpp_backend()


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    """Parametrize tests over both Python and C++ backends.

    Sets MAGI_ATTENTION_CPP_BACKEND env var, reloads common modules,
    then restores the original state after the test.
    """
    if request.param == "cpp" and not _cpp_available:
        pytest.skip("magi_attn_ext not available")

    use_cpp = request.param == "cpp"
    old_val = os.environ.get("MAGI_ATTENTION_CPP_BACKEND")

    os.environ["MAGI_ATTENTION_CPP_BACKEND"] = "1" if use_cpp else "0"
    _reload_common_modules()

    yield request.param

    if old_val is not None:
        os.environ["MAGI_ATTENTION_CPP_BACKEND"] = old_val
    else:
        os.environ.pop("MAGI_ATTENTION_CPP_BACKEND", None)
    _reload_common_modules()
