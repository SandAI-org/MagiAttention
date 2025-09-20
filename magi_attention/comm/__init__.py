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
    Toggle this env variable to ``1`` to enable hierarchical group-collective comm
    within 2-dim cp group (inter_node group + intra_node group)

    Default value is ``0``

    NOTE: this is for now a temporary solution to reduce the redundant inter-node comm
    and might be removed or updated in the future
    """
    return os.environ.get("MAGI_ATTENTION_HIERARCHICAL_COMM", "0") == "1"


def ffa_fwd_sm_margin_save_for_comm() -> int:
    """
    Set the value of this env variable to control
    the number of SMs of the ffa forward kernel saved for comm kernels

    Default value is ``4`` if "CUDA_DEVICE_MAX_CONNECTIONS" > ``1``, otherwise ``0``
    """

    sm_margin = os.environ.get("MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN", None)
    if sm_margin is None:  # set by default
        max_connections = int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        sm_margin = "4" if max_connections > 1 else "0"

    return int(sm_margin)


def ffa_bwd_sm_margin_save_for_comm() -> int:
    """
    Set the value of this env variable to control
    the number of SMs of the ffa backward kernel saved for comm kernels

    Default value is ``4`` if "CUDA_DEVICE_MAX_CONNECTIONS" > ``1``, otherwise ``0``
    """

    sm_margin = os.environ.get("MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN", None)
    if sm_margin is None:  # set by default
        max_connections = int(os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "8"))
        sm_margin = "4" if max_connections > 1 else "0"

    return int(sm_margin)


def is_qo_comm_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable query/output communication,
    including fetching remote q (fwd), reducing partial out and lse (fwd),
    fetching remote q,o,lse,do (bwd), reducing partial dq (bwd),
    to eliminate the restriction that communication is limited solely to key/value

    Default value is ``0``

    NOTE: this feature is experimental and under development for now,
    which dose NOT support neither multi-stage overlap nor hierarchical comm
    """
    return os.environ.get("MAGI_ATTENTION_QO_COMM", "0") == "1"


def is_ffa_fwd_high_precision_reduce_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable high-precision (fp32) reduce
    for partial out during dist-attn forward
    to trade-off double comm overhead for increased precision and less dtype-cast overhead

    Default value is ``0``

    NOTE:
    1. inside the ffa forward kernel, we always use high-precision (fp32) accumulation
            for partial out

    2. we always use high-precision (fp32) lse everywhere

    3. this feature works for out only when enabling qo comm
    """
    return (
        os.environ.get("MAGI_ATTENTION_FFA_FORWARD_HIGH_PRECISION_REDUCE", "0") == "1"
    )


def is_ffa_bwd_high_precision_reduce_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable high-precision (fp32) reduce
    for partial dq,dk,dv during dist-attn backward
    to trade-off double comm overhead for increased precision and less dtype-cast overhead

    Default value is ``0``

    NOTE:
    1. inside the ffa backward kernel, we always use high-precision (fp32) accumulation
        for partial dq,dk,dv

    2. this feature works for dq only when enabling qo comm
    """
    return (
        os.environ.get("MAGI_ATTENTION_FFA_BACKWARD_HIGH_PRECISION_REDUCE", "0") == "1"
    )


def is_native_grpcoll_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable native grpcoll comm kernel implementation

    Default value is ``0``

    NOTE: this feature for now is experimental and under early development,
    and please do NOT enable it unless you know what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_NATIVE_GRPCOLL", "0") == "1"
