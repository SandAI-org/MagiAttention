# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
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

from magi_attention.common import AttnForwardMeta
from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import (
    BSDispatchAlg,
    DispatchAlg,
    DispatchConfig,
    DistAttnConfig,
    DPDispatchAlg,
    GreedyOverlapAlg,
    GrpCollConfig,
    LBDispatchAlg,
    MinHeapDispatchAlg,
    OverlapAlg,
    OverlapConfig,
    SequentialDispatchAlg,
    SortedSequentialSelectAlg,
    UniformOverlapAlg,
)
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey
from magi_attention.functional import flex_flash_attn_func
from magi_attention.meta.solver.dispatch_solver import ToppHeapDispatchAlg

from .functools import (
    compute_pad_size,
    infer_attn_mask_from_cu_seqlens,
    infer_attn_mask_from_sliding_window,
    infer_varlen_mask_from_batch,
    squash_batch_dim,
)
from .magi_attn_interface import (
    DistAttnRuntimeDictManager,
    GeneralAttnMaskType,
    calc_attn,
    clear_cache,
    dispatch,
    dist_attn_runtime_dict_mgr,
    get_most_recent_key,
    get_position_ids,
    magi_attn_flex_dispatch,
    magi_attn_flex_key,
    magi_attn_varlen_dispatch,
    magi_attn_varlen_key,
    make_flex_key_for_new_mask_after_dispatch,
    make_varlen_key_for_new_mask_after_dispatch,
    roll,
    roll_simple,
    undispatch,
)

__all__ = [
    # ---- public API functions ----
    "magi_attn_varlen_key",
    "magi_attn_varlen_dispatch",
    "magi_attn_flex_key",
    "magi_attn_flex_dispatch",
    "dispatch",
    "undispatch",
    "roll",
    "roll_simple",
    "calc_attn",
    "clear_cache",
    "get_most_recent_key",
    "get_position_ids",
    "make_varlen_key_for_new_mask_after_dispatch",
    "make_flex_key_for_new_mask_after_dispatch",
    "flex_flash_attn_func",
    # ---- helper / functools ----
    "compute_pad_size",
    "squash_batch_dim",
    "infer_varlen_mask_from_batch",
    "infer_attn_mask_from_sliding_window",
    "infer_attn_mask_from_cu_seqlens",
    # ---- data structures & types used in API signatures ----
    "AttnForwardMeta",
    "AttnMaskType",
    "AttnOverlapMode",
    "AttnRanges",
    "DistAttnRuntimeKey",
    "GeneralAttnMaskType",
    # ---- config classes ----
    "DistAttnConfig",
    "DispatchConfig",
    "OverlapConfig",
    "GrpCollConfig",
    # ---- dispatch algorithms ----
    "DispatchAlg",
    "MinHeapDispatchAlg",
    "ToppHeapDispatchAlg",
    "SequentialDispatchAlg",
    "SortedSequentialSelectAlg",
    "LBDispatchAlg",
    "DPDispatchAlg",
    "BSDispatchAlg",
    # ---- overlap algorithms ----
    "OverlapAlg",
    "UniformOverlapAlg",
    "GreedyOverlapAlg",
    # ---- runtime manager ----
    "DistAttnRuntimeDictManager",
    "dist_attn_runtime_dict_mgr",
]
