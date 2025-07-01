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
    CAUSAL = "causal"  # NOTE: The causal mask aligns to the bottom-right corner if it's not square
    BICAUSAL = "bicausal"
    INVCAUSAL = "invcausal"


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
