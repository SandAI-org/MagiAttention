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

"""Centralised access to every ``MAGI_ATTENTION_*`` environment variable.

Usage::

    from magi_attention.env.general import is_sanity_check_enable
    from magi_attention.env.comm import is_qo_comm_enable
    from magi_attention.env.build import is_force_jit_build

Environment variables owned by the *testing* helpers
(``MAGI_ATTENTION_TEST_*``, ``MAGI_ATTENTION_PARAMETERIZE_RUN_IN_MP``)
and by the third-party ``flash-attention`` sub-package are intentionally
**not** included — they live closer to the code that consumes them.
"""

from . import build, comm, general

__all__ = ["general", "comm", "build"]
