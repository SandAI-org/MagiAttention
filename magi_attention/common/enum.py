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

from enum import Enum
from typing import Literal, TypeAlias

import torch

GroupReduceOp: TypeAlias = Literal["sum", "avg", "lse"]

OutMaybeWithLSE: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]

AttnSinkLayout: TypeAlias = Literal["sh", "shd", "ssh"]


class AttnType(Enum):
    """The enum used to specify the type of attention calculation we support"""

    SELF_ATTN = "self_attn"
    CROSS_ATTN = "cross_attn"


class AttnRole(Enum):
    """The enum used to specify the tensor role in attention"""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"


class AttnMaskType(Enum):
    """The enum used to specify the unit type of attention mask we support"""

    FULL = "full"
    CAUSAL = "causal"
    BICAUSAL = "bi_causal"
    INVCAUSAL = "inv_causal"

    _FROM_INT_MAP: dict[int, "AttnMaskType"]
    _TO_INT_MAP: dict["AttnMaskType", int]

    @classmethod
    def _lazy_init_from_int_map(cls) -> None:
        if "_FROM_INT_MAP" in cls.__dict__:
            return

        cls._FROM_INT_MAP = {
            0: cls.FULL,
            1: cls.CAUSAL,
            2: cls.INVCAUSAL,
            3: cls.BICAUSAL,
        }

    @classmethod
    def _lazy_init_to_int_map(cls) -> None:
        if "_TO_INT_MAP" in cls.__dict__:
            return

        cls._TO_INT_MAP = {
            cls.FULL: 0,
            cls.CAUSAL: 1,
            cls.INVCAUSAL: 2,
            cls.BICAUSAL: 3,
        }

    @classmethod
    def from_int_type(cls, int_type: int) -> "AttnMaskType":
        cls._lazy_init_from_int_map()
        return cls._FROM_INT_MAP[int_type]  # type: ignore[index]

    def to_int_type(self) -> int:
        self.__class__._lazy_init_to_int_map()
        return self._TO_INT_MAP[self]


class AttnOverlapMode(Enum):
    """The enum used to specify the overlap mode for multi-stage overlapping"""

    STATIC = "static"
    DYNAMIC = "dynamic"


class DispatchAlgType(Enum):
    """The enum used to specify the algorithm type for load-balanced dispatching"""

    LOWER_BOUND = "lower_bound"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    BINARY_SEARCH = "binary_search"
    MIN_HEAP = "min_heap"
    TOPP_HEAP = "topp_heap"
    BACKTRACKING_PRUNING = "backtracing_pruning"
    RANDOM_SELECT = "random_select"
    SEQUENTIAL_SELECT = "sequential_select"
    BATCH_TOPP_HEAP = "batch_topp_heap"
    SORTED_SEQUENTIAL_SELECT = "sorted_sequential_select"


class OverlapAlgType(Enum):
    """The enum used to specify the algorithm type for multi-stage overlapping"""

    UNIFORM = "uniform"
    GREEDY = "greedy"


class DynamicAttnAlgType(Enum):
    """The enum used to specify the algorithm type for dynamic attn mask dispatching"""

    NON_COMMUNICATION_QO = "non_communication_qo"
    GREEDY_RANDOM_GRID = "greedy_random_grid"
    SIMPLEX_NETWORK_FLOW = "simplex_network_flow"
    FAST_SIMPLEX_NETWORK_FLOW = "fast_simplex_network_flow"
