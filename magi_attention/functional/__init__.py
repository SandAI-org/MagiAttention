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

from .dispatch import dispatch_func, undispatch_func
from .dist_attn import dist_attn_func
from .flex_flash_attn import flex_flash_attn_func
from .utils import correct_attn_fwd_result, correct_attn_lse, correct_attn_out

__all__ = [
    "flex_flash_attn_func",
    "dist_attn_func",
    "dispatch_func",
    "undispatch_func",
    "correct_attn_fwd_result",
    "correct_attn_out",
    "correct_attn_lse",
]


def fa4_hsfu_max_num_funcs() -> int:
    """
    HACK: since the FA4 with the attention mask representation of HSFU Functions
    requires the maximum number of functions to be set, 
    and it is hard to pass through the user API to the arguments,
    thus we let user set it through the environment variable, if using FA4 backend.

    NOTE: this is a beta feature, under development
    """
    
    max_num_funcs = int(os.environ.get("MAGI_ATTENTION_FA4_HSFU_MAX_NUM_FUNCS", "0"))
    
    if max_num_funcs == 0:
        raise ValueError("MAGI_ATTENTION_FA4_HSFU_MAX_NUM_FUNCS is not set")
    
    return max_num_funcs
