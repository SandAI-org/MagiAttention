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

from magi_attention.common.enum import AttnMaskType, AttnOverlapMode
from magi_attention.common.ranges import AttnRanges
from magi_attention.config import DistAttnConfig
from magi_attention.functional import flex_flash_attn_func
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    MinHeapDispatchAlg,
)
from magi_attention.meta.solver.overlap_solver import OverlapConfig, UniformOverlapAlg

from .functools import (
    compute_pad_size,
    from_mask,
    full_attention_to_varlen_attention,
    infer_attn_mask_from_sliding_window,
    infer_attn_mask_from_window_size,
    init_hierarchical_mesh,
    squash_batch_dim,
)
from .magi_attn_interface import (
    calc_attn,
    dispatch,
    get_position_ids,
    magi_attn_flex_dispatch,
    magi_attn_flex_key,
    magi_attn_varlen_dispatch,
    magi_attn_varlen_key,
    undispatch,
)

__all__ = [
    "calc_attn",
    "dispatch",
    "magi_attn_flex_dispatch",
    "magi_attn_flex_key",
    "magi_attn_varlen_dispatch",
    "magi_attn_varlen_key",
    "undispatch",
    "flex_flash_attn_func",
    "get_position_ids",
    "compute_pad_size",
    "squash_batch_dim",
    "full_attention_to_varlen_attention",
    "from_mask",
    "AttnMaskType",
    "AttnOverlapMode",
    "AttnRanges",
    "DistAttnConfig",
    "DispatchConfig",
    "MinHeapDispatchAlg",
    "OverlapConfig",
    "UniformOverlapAlg",
    "init_hierarchical_mesh",
    "infer_attn_mask_from_window_size",
    "infer_attn_mask_from_sliding_window",
]
