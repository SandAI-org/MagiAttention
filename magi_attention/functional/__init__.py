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


def is_ffa_fwd_inplace_correct_enable() -> bool:
    """
    Toggle this env variable to ``1`` can enable inplace-correct for out and lse in ffa forward
    to avoid the storage of partial results and the memory-bound ``result_correction`` as a forward post process

    Default value is ``0``

    NOTE: this feature will be enabled by default as long as it's stable (i.e. no effect on accuracy or performance)
    """
    return os.environ.get("MAGI_ATTENTION_FFA_FORWARD_INPLACE_CORRECT", "0") == "1"


def is_ffa_bwd_high_precision_reduce_enable() -> bool:
    """
    Toggle this env variable to ``1`` can enable high-precision (fp32) reduce for dkv among ranks in ffa backward
    to increase the precision at the cost of double comm overhead

    Default value is ``0``

    NOTE: inside the ffa backward kernel, we always use high-precision (fp32) accumulation for partial dkv,
    however, by default we will downcast it to kv dtype before reducing among ranks to decrease comm overhead
    """
    return (
        os.environ.get("MAGI_ATTENTION_FFA_BACKWARD_HIGH_PRECISION_REDUCE", "0") == "1"
    )
