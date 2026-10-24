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

# Use a fresh process per configuration to capture import and tracing warnings.
# Keep DSL imports inside the warning recorder.
# Usage: python dsl_deprecation_compile.py <configuration name>

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass

RESULT_PREFIX = "DSL_DEPRECATION_RESULT "


@dataclass(frozen=True)
class CompileConfig:
    """One SM100 attention configuration for deprecation checks."""

    name: str
    head_dim: int
    has_ranges: bool = False
    mask_type: int = 0


# Cover persistent fwd / 2-CTA bwd and non-persistent fwd / 1-CTA bwd.
CONFIGS: tuple[CompileConfig, ...] = (
    CompileConfig("d128_full", 128),
    CompileConfig("r64_causal", 64, has_ranges=True, mask_type=1),
)


def _run(config: CompileConfig) -> dict:
    import torch

    from magi_attention.kernel.cutedsl import cache_utils, flex_flash_attn

    torch.manual_seed(0)
    seqlen, num_heads = 1024, 2
    if config.has_ranges:
        q, k, v = (
            torch.randn(
                (2 * seqlen, num_heads, config.head_dim),
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        )
        out, _ = flex_flash_attn.flex_flash_attn_func(
            q,
            k,
            v,
            q_ranges=torch.tensor(
                [[0, seqlen], [seqlen, 2 * seqlen]],
                device="cuda",
                dtype=torch.int32,
            ),
            k_ranges=torch.tensor(
                [[0, seqlen], [0, 2 * seqlen]], device="cuda", dtype=torch.int32
            ),
            mask_types=config.mask_type,
            max_seqlen_q=seqlen,
            max_seqlen_k=2 * seqlen,
        )
    else:
        q, k, v = (
            torch.randn(
                (1, seqlen, num_heads, config.head_dim),
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        )
        out, _ = flex_flash_attn.flex_flash_attn_func(
            q, k, v, mask_types=config.mask_type
        )
    torch.autograd.grad(out, (q, k, v), torch.randn_like(out))
    torch.cuda.synchronize()

    return {"cache_enabled": cache_utils.CUTE_DSL_CACHE_ENABLED}


def main(config_name: str) -> None:
    (config,) = (config for config in CONFIGS if config.name == config_name)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always", DeprecationWarning)
        import magi_attention

        result = _run(config)

    import cutlass

    result["magi_attention_file"] = magi_attention.__file__
    result["cutlass_dir"] = os.path.dirname(os.path.realpath(cutlass.__file__))
    result["records"] = [
        {
            "category": f"{record.category.__module__}.{record.category.__qualname__}",
            "is_deprecation": issubclass(record.category, DeprecationWarning),
            "message": str(record.message),
            "filename": record.filename,
            "lineno": record.lineno,
        }
        for record in records
    ]
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
