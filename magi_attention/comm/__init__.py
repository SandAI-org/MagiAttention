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
    and should be removed or updated in the future
    """
    return os.environ.get("MAGI_ATTENTION_HIERARCHICAL_COMM", "0") == "1"


def is_magi_nccl_backend_enable() -> bool:
    """
    Toggling this env variable to 1 to enable magi_nccl_backend
    for native group collective communication

    NOTE: This flag only influences the dispatch of our group collective communication operation,
    and you need to activate `magi_nccl` as the progress group backend for cuda device by setting:
        `torch.distributed.init_process_group(backend='cpu:gloo,cuda:magi_nccl', ...)`

    Then, everything is exactly the same as using `nccl` backend,
    except that you can use more APIs that we added through `magi_attention.comm.primitive.magi_nccl_interface`,
    including `nccl_stream`, `group_cast`, `group_reduce`, etc.
    """
    return os.environ.get("MAGI_NCCL_BACKEND", "0") == "1"


def use_batch_p2p_for_group_collective() -> bool:
    """
    Toggling this env variable to 1 to use batch p2p for group collective using `nccl` backend
    instead of simulating group collective through all2all-v plus pre-/post-process
    """
    return (
        os.environ.get("MAGI_ATTENTION_USE_BATCH_P2P_FOR_GROUP_COLLECTIVE", "0") == "1"
    )
