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

from magi_attention.meta.solver.dispatch_solver import (
    BSDispatchAlg,
    DispatchAlg,
    DispatchConfig,
    DPDispatchAlg,
    LBDispatchAlg,
    MinHeapDispatchAlg,
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
    overlap_config: OverlapConfig = (
        OverlapConfig()
    )  # TODO: add distinct overlap config for fwd/bwd in the future
    deterministic: bool = False
