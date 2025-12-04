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

# Copyright (c) Meta Platforms, Inc. and affiliates

import os

from ._fsdp_api import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from ._fsdp_ckpt_state import EMAParamStatefulWrapper, MainParamStatefulWrapper
from ._fsdp_mem_tracker import MagiFSDPMemTracker
from ._fsdp_module import (
    MagiFSDPModule,
    UnshardHandle,
    magi_fsdp_switch_params,
    magi_fsdp_use_params,
)
from ._fully_shard import fully_shard, register_fsdp_forward_method

__all__ = [
    "CPUOffloadPolicy",
    "MagiFSDPModule",
    "fully_shard",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "register_fsdp_forward_method",
    "UnshardHandle",
    "MagiFSDPMemTracker",
    "magi_fsdp_use_params",
    "magi_fsdp_switch_params",
    "MainParamStatefulWrapper",
    "EMAParamStatefulWrapper",
]


def is_multi_dtype_reduce_enable() -> bool:
    """
    Check whether multi-dtype reduce is enabled.

    Controlled by the environment variable ``MAGI_FSDP_MULTI_DTYPE_REDUCE``:
      - ``"1"``: use the foreach_dtype_reduce (allows mixed precision reduce across dtypes such as fp32/bf16).
      - ``"0"`` (default): use the foreach_reduce.

    NOTE: This option is intended only for testing or debugging, as the correctness
    of custom implementation is not fully guaranteed.
    """
    return os.environ.get("MAGI_FSDP_MULTI_DTYPE_REDUCE", "0") == "1"
