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

    NOTE: this feature is experimental and under early development for now
    and not compatible with many other features,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_QO_COMM", "0") == "1"


def is_fwd_high_precision_reduce_enable() -> bool:
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
    return os.environ.get("MAGI_ATTENTION_FORWARD_HIGH_PRECISION_REDUCE", "0") == "1"


def is_bwd_high_precision_reduce_enable() -> bool:
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
    return os.environ.get("MAGI_ATTENTION_BACKWARD_HIGH_PRECISION_REDUCE", "0") == "1"


def is_native_grpcoll_enable() -> bool:
    """
    Toggle this env variable to ``1`` to enable native kernel implementation for group collective comm

    Default value is ``0``

    NOTE: this feature is experimental and under early development for now
    and not compatible with many other features,
    thus please do NOT enable it unless you know exactly what you are doing
    """
    return os.environ.get("MAGI_ATTENTION_NATIVE_GRPCOLL", "0") == "1"


def dsink_all_reduce_op() -> str:
    """
    Set the value of this env variable to control the all-reduce op
    for sink gradients within ``dist_attn_func`` when involving attention sink.

    Default value is ``none``. And options are within {``none``, ``sum``, ``avg``}.

    NOTE: When involving attention sink,
    for now we only accept global replicated sink tensor as input to feed into ``dist_attn_func``,
    and the gradients of sink in each cp rank are partial and requires to be sum-reduced across cp ranks.

    However, since sink tensor is learnable, it will be considered as a regular parameter in the model
    similar to ``bias`` in ``nn.Linear`` layer.

    So under some popular training frameworks, such as Megatron-LM, FSDP, the sum-reduction across cp ranks
    of the partial gradients of sink might be automatically applied within the whole ``dp x cp`` mesh.

    To avoid repeated reduction, we provide this environment variable
    to specify the all-reduce op for sink gradients within ``dist_attn_func``.

    Defaults to ``none`` to NOT apply any reduction to sink gradients by ``dist_attn_func`` and let the framework handle it.

    However, under the scenarios w/o any framework mechanism to reduce parameters across cp ranks,
    you have to specify this environment variable to ``sum``.

    And sometimes, ``avg`` might also be an option when you need to scale the sink gradients by ``1/cp``.
    """

    op = os.environ.get("MAGI_ATTENTION_DSINK_ALL_REDUCE_OP", "none")
    assert op in (
        "none",
        "sum",
        "avg",
    ), f"Invalid value of MAGI_ATTENTION_DSINK_ALL_REDUCE_OP: {op}"

    return op
