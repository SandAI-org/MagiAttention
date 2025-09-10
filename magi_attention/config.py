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

from dataclasses import dataclass

import magi_attention
from magi_attention.meta.solver.dispatch_solver import (
    BSDispatchAlg,
    DispatchAlg,
    DispatchConfig,
    DPDispatchAlg,
    LBDispatchAlg,
    MinHeapDispatchAlg,
    SequentialDispatchAlg,
)
from magi_attention.meta.solver.overlap_solver import (
    GreedyOverlapAlg,
    OverlapAlg,
    OverlapConfig,
    UniformOverlapAlg,
)

__all__ = [
    "DistAttnConfig",
    "DispatchConfig",
    "DispatchAlg",
    "LBDispatchAlg",
    "DPDispatchAlg",
    "BSDispatchAlg",
    "MinHeapDispatchAlg",
    "SequentialDispatchAlg",
    "OverlapConfig",
    "OverlapAlg",
    "UniformOverlapAlg",
    "GreedyOverlapAlg",
]


@dataclass(frozen=True)
class DistAttnConfig:
    """The overall config dataclass for dist-attn
    containing sub-configs for sub-modules to be assigned
    """

    dispatch_config: DispatchConfig = DispatchConfig()
    # TODO: add distinct overlap config for fwd/bwd in the future
    overlap_config: OverlapConfig = OverlapConfig()

    def __post_init__(self):
        if magi_attention.comm.is_qo_comm_enable():
            # HACK: for now, if enabling qo comm,
            # we only support sequential dispatch
            object.__setattr__(
                self, "dispatch_config", DispatchConfig(alg=SequentialDispatchAlg())
            )
            # HACK: for now, if enabling qo comm,
            # we does NOT support multi-stage overlap
            object.__setattr__(self, "overlap_config", OverlapConfig(enable=False))
