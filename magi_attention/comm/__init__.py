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

from . import functional, primitive
from .work import WorkWithPostProcessFn

__all__ = [
    "primitive",
    "functional",
    "WorkWithPostProcessFn",
]


def is_hierarchical_comm_enable() -> bool:
    """
    Toggling this env variable to 1 to enable hierarchical group-collective comm
    within 2-dim cp group (inter_node group + intra_node group)

    NOTE: this is for now a temporary solution to reduce the redundant inter-node comm
    and might be removed or updated in the future
    """
    return os.environ.get("MAGI_ATTENTION_HIERARCHICAL_COMM", "0") == "1"


def ffa_fwd_sm_margin_save_for_comm() -> int:
    """
    The sm margin number of ffa forward kernel saved for comm kernels
    """

    sm_margin = os.environ.get("MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN", None)
    if sm_margin is None:  # set by default
        max_connections = int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        sm_margin = "8" if max_connections > 1 else "0"

    return int(sm_margin)


def ffa_bwd_sm_margin_save_for_comm() -> int:
    """
    The sm margin number of ffa backward kernel saved for comm kernels
    """

    sm_margin = os.environ.get("MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN", None)
    if sm_margin is None:  # set by default
        max_connections = int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        sm_margin = "8" if max_connections > 1 else "0"

    return int(sm_margin)
