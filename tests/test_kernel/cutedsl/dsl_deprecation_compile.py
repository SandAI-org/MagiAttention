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

# Child process of test_dsl_deprecations.py: compiles and runs the forward and
# backward of one CuTeDSL FFA configuration on the device and prints, as the
# last stdout line, every warning recorded from the first import of
# magi_attention to the end of the run.
#
# The process must be fresh for each configuration: its in-process compile
# caches start empty, so every kernel of the run is traced, and part of the
# deprecated-name accesses happen at import time. The module imports only the
# standard library at top level, so the warning recorder is installed before
# magi_attention and the DSL are imported.
#
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
    """One SM100 kernel configuration of the deprecation gate.

    has_ranges: the run passes ``q_ranges`` / ``k_ranges`` (packed layout);
        otherwise it uses the dense batched layout.
    mask_type: the ``mask_types`` argument of ``flex_flash_attn_func``.
    """

    name: str
    head_dim: int
    has_ranges: bool = False
    mask_type: int = 0


# Dense full attention launches the persistent forward and, at hd128, the
# 2-CTA backward; the hd64 causal ranges run launches the non-persistent
# forward and the 1-CTA backward.
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
