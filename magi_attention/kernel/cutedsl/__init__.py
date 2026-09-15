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

from .ffa_utils import (
    MT_MAP,
    TorchFlexAttnArgs,
    normalize_mask_types,
    validate_range_feature_support,
    validate_true_ranges,
)
from .flex_flash_attn import flex_flash_attn_func
from .range_merge import RangeMergePlan, plan_range_merge, plan_range_merge_bwd

__all__ = [
    "flex_flash_attn_func",
    "RangeMergePlan",
    "plan_range_merge",
    "plan_range_merge_bwd",
    "TorchFlexAttnArgs",
    "MT_MAP",
    "normalize_mask_types",
    "validate_range_feature_support",
    "validate_true_ranges",
]


def is_ffa_debug_mode_enabled() -> bool:
    """Check if the debug mode of CuteDSL-based FFA
    is enabled via environment variable.
    """
    import os

    return os.getenv("MAGI_ATTENTION_FFA_CUTEDSL_DEBUG_MODE", "0") == "1"
