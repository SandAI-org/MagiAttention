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

"""Centralised env-var names for flex-flash-attention (FFA) kernel tuning.

Every ``MAGI_ATTENTION_FFA_*`` environment variable used anywhere in the
package is defined here as a module-level string constant.  Call-sites
import the constant and pass it to ``os.environ.get(...)`` — the read /
validation logic stays local to the consumer because each var has unique
semantics, but the *name* is defined in exactly one place.

Renaming an env var is a one-line change here; any stale reference
elsewhere will cause an ``ImportError`` or ``AttributeError`` instead of
a silent mismatch.
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------ #
#  FWD / BWD kernel tuning knobs
# ------------------------------------------------------------------ #

BWD_PRODUCER_REGS = "MAGI_ATTENTION_FFA_BWD_PRODUCER_REGS"
INTRA_WG_OVERLAP = "MAGI_ATTENTION_FFA_INTRA_WG_OVERLAP"
INNER_DIR_MAX_TO_MIN = "MAGI_ATTENTION_FFA_INNER_DIR_MAX_TO_MIN"
MASK_MODE = "MAGI_ATTENTION_FFA_MASK_MODE"
INNER_STORE_IN_PRODUCER = "MAGI_ATTENTION_FFA_INNER_STORE_IN_PRODUCER"
OUTER_STORE_MODE = "MAGI_ATTENTION_FFA_OUTER_STORE_MODE"
USE_MASK_DISPATCH = "MAGI_ATTENTION_FFA_USE_MASK_DISPATCH"

# ------------------------------------------------------------------ #
#  Inner load / store modes
# ------------------------------------------------------------------ #

INNER_LOAD_MODE = "MAGI_ATTENTION_FFA_INNER_LOAD_MODE"
INNER_STORE_MODE = "MAGI_ATTENTION_FFA_INNER_STORE_MODE"

# ------------------------------------------------------------------ #
#  BWD tile / stage / SMEM overrides
# ------------------------------------------------------------------ #

BWD_TILE_M = "MAGI_ATTENTION_FFA_BWD_TILE_M"
BWD_TILE_N = "MAGI_ATTENTION_FFA_BWD_TILE_N"
BWD_STAGES = "MAGI_ATTENTION_FFA_BWD_STAGES"
BWD_STAGES_DS = "MAGI_ATTENTION_FFA_BWD_STAGES_DS"
BWD_STAGES_V = "MAGI_ATTENTION_FFA_BWD_STAGES_V"
BWD_UNION_DKV_SMEM = "MAGI_ATTENTION_FFA_BWD_UNION_DKV_SMEM"
BWD_PERF_UNION_STGV = "MAGI_ATTENTION_FFA_BWD_PERF_UNION_STGV"
BWD_INNER_STORE_STAGES = "MAGI_ATTENTION_FFA_BWD_INNER_STORE_STAGES"
BWD_DKV_USE_SMEM = "MAGI_ATTENTION_FFA_BWD_DKV_USE_SMEM"

# ------------------------------------------------------------------ #
#  BWD debug skip switches (correctness NOT guaranteed)
# ------------------------------------------------------------------ #

BWD_SKIP_V_LOAD = "MAGI_ATTENTION_FFA_BWD_SKIP_V_LOAD"
BWD_SKIP_DV_STORE = "MAGI_ATTENTION_FFA_BWD_SKIP_DV_STORE"
BWD_SKIP_DK_STORE = "MAGI_ATTENTION_FFA_BWD_SKIP_DK_STORE"
BWD_SKIP_DV_MMA = "MAGI_ATTENTION_FFA_BWD_SKIP_DV_MMA"
BWD_SKIP_DV_WRITEBACK = "MAGI_ATTENTION_FFA_BWD_SKIP_DV_WRITEBACK"
BWD_SKIP_DK_WRITEBACK = "MAGI_ATTENTION_FFA_BWD_SKIP_DK_WRITEBACK"
BWD_DEFER_DV_R2S = "MAGI_ATTENTION_FFA_BWD_DEFER_DV_R2S"
BWD_FORCE_MMA_DKV_SS = "MAGI_ATTENTION_FFA_BWD_FORCE_MMA_DKV_SS"

# ------------------------------------------------------------------ #
#  Comm SM margin (also read by magi_attention/env/comm.py)
# ------------------------------------------------------------------ #

FORWARD_SM_MARGIN = "MAGI_ATTENTION_FFA_FORWARD_SM_MARGIN"
BACKWARD_SM_MARGIN = "MAGI_ATTENTION_FFA_BACKWARD_SM_MARGIN"

# ------------------------------------------------------------------ #
#  FA4 cache
# ------------------------------------------------------------------ #

FA4_CACHE_DIR = "MAGI_ATTENTION_FFA_FA4_CACHE_DIR"

# ------------------------------------------------------------------ #
#  LRU-cache invalidation: every env key that can change JIT output
# ------------------------------------------------------------------ #

ENV_KEYS_AFFECTING_COMPILATION: tuple[str, ...] = (
    INTRA_WG_OVERLAP,
    USE_MASK_DISPATCH,
    INNER_DIR_MAX_TO_MIN,
    MASK_MODE,
    INNER_STORE_IN_PRODUCER,
    INNER_LOAD_MODE,
    INNER_STORE_MODE,
    OUTER_STORE_MODE,
    BWD_PRODUCER_REGS,
    BWD_FORCE_MMA_DKV_SS,
    BWD_TILE_M,
    BWD_TILE_N,
    BWD_STAGES,
    BWD_STAGES_DS,
    BWD_STAGES_V,
    BWD_DKV_USE_SMEM,
    BWD_UNION_DKV_SMEM,
    BWD_INNER_STORE_STAGES,
    BWD_SKIP_V_LOAD,
    BWD_SKIP_DV_STORE,
    BWD_SKIP_DK_STORE,
    BWD_SKIP_DV_MMA,
    BWD_SKIP_DV_WRITEBACK,
    BWD_SKIP_DK_WRITEBACK,
    BWD_DEFER_DV_R2S,
    BWD_PERF_UNION_STGV,
)


def snapshot_env() -> tuple[tuple[str, str | None], ...]:
    """Capture all env vars that affect kernel compilation into a hashable key.

    Returned as ``_env_snapshot`` to ``get_ffa_jit_mod`` so that
    ``lru_cache`` sees different keys when env vars change between calls.
    """
    return tuple((k, os.environ.get(k)) for k in ENV_KEYS_AFFECTING_COMPILATION)
