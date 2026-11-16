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

import pytest


@pytest.fixture(scope="session", autouse=True)
def backend():
    """Verify the backend fixed before collection, without changing module state."""
    from magi_attention import common
    from magi_attention.common.range import AttnRange as PythonRange

    selected = os.environ["MAGI_ATTENTION_CPP_BACKEND"]
    assert selected in ("0", "1"), selected
    use_cpp = selected == "1"
    assert common.USE_CPP_BACKEND == use_cpp, "requested backend was not loaded"
    if use_cpp:
        from magi_attention.magi_attn_ext import AttnRange as CppRange

        assert common.AttnRange is CppRange
    else:
        assert common.AttnRange is PythonRange
    return "cpp" if use_cpp else "python"
