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

import os

from . import dist_attn
from .dispatch import dispatch_func, undispatch_func
from .dist_attn import dist_attn_func, result_correction
from .flex_flash_attn import flex_flash_attn_func

__all__ = [
    "flex_flash_attn_func",
    "dist_attn",
    "dist_attn_func",
    "result_correction",
    "dispatch_func",
    "undispatch_func",
]


def is_ffa_fwd_inplace_correct_enable() -> bool:
    """
    Toggling this env variable to `1` can enable inplace-correct for out and lse in ffa forward
    to avoid the storage of partial results and the memory-bound `result_correction` as a forward post process

    NOTE: this feature will be enabled by default as long as it's stable (i.e. no effect on accuracy or performance)
    """
    return os.environ.get("MAGI_ATTENTION_FFA_FORWARD_INPLACE_CORRECT", "0") == "1"
