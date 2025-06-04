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

# TODO: cross-attn
# TODO: comm-kernel
# TODO: abitrary-attn-mask
#           * bi-causal and inv-causal for magi_attention
#           * q_ranges overlap for magi_attention
# TODO: more efficient solver
# TODO: more usable affinity dispatch
# TODO: backward kernel support for sm margin

"""
1. is_same_source
    a. (Done) q is permutable, k is permutable -> self-attn
    b. q is permutable, k is not permutable -> invalid
    c. q is not permutable, k is permutable -> invalid
    d. q is not permutable, k is not permutable -> output meta

2. is_not_same_source
    a. q is permutable, k is permutable -> pure cross attn
    b. q is permutable, k is not permutable -> t5
    c. q is not permutable, k is permutable -> multi-modal
    d. (TODO) q is not permutable, k is not permutable -> output meta
"""

import os

from . import config
from .dist_attn_runtime_mgr import init_dist_attn_runtime_mgr

__all__ = [
    "init_dist_attn_runtime_mgr",
    "is_sanity_check_enable",
    "is_cuda_device_max_connections_one",
    "config",
]


def is_sanity_check_enable() -> bool:
    """
    Toggling this env variable to 1 can enable many sanity check codes inside magi_attention
    which is only supposed to be used for testing or debugging,
    since these codes involve performance overhead
    """
    return os.environ.get("MAGI_ATTENTION_SANITY_CHECK", "0") == "1"


def is_sdpa_backend_enable() -> bool:
    """
    Toggling this env variable to 1 can switch the attn kernel backend
    from ffa to sdpa-math, to support higher precision like fp32, fp64,
    which is only supposed to be used for testing or debugging,
    since the performance is not acceptable
    """
    return os.environ.get("MAGI_ATTENTION_SDPA_BACKEND", "0") == "1"


def is_refactor_bwd_args_enable() -> bool:
    """
    Toggling this env variable to 1 to enable
    using the refactored ffa args for backward dkv load-store efficiency

    NOTE: this flag is only for the experimental stage,
    if it works, we should always enable it and remote this env variable
    """
    return os.environ.get("MAGI_ATTENTION_REFACTOR_BWD_ARGS", "0") == "1"


def is_cuda_device_max_connections_one() -> bool:
    """
    Toggle this env variable to 1 to allow cuda device to have only one connection
    """
    return os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8") == "1"


def is_hierarchical_comm_enable() -> bool:
    """
    Toggling this env variable to 1 to enable hierarchical group-collective comm
    within 2-dim cp group (inter_node group + intra_node group)

    NOTE: this is for now a temporary solution to reduce the redundant inter-node comm
    and should be removed or updated in the future
    """
    return os.environ.get("MAGI_ATTENTION_HIERARCHICAL_COMM", "0") == "1"
