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

# Check import and tracing deprecations in fresh processes with caching disabled.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import magi_attention
from tests.test_kernel.cutedsl.dsl_deprecation_compile import (
    CONFIGS,
    RESULT_PREFIX,
    CompileConfig,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the gate compiles and launches the SM100 kernels on a GPU of "
    "compute capability 10",
)

_CHILD = Path(__file__).with_name("dsl_deprecation_compile.py")
_PACKAGE_DIR = Path(magi_attention.__file__).resolve().parent
# Keep child imports tied to the package under test.
_PACKAGE_PARENT = _PACKAGE_DIR.parent
_CHILD_TIMEOUT_S = 120

_CACHE_ENABLED_ENV = "MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED"
_CACHE_DIR_ENV = "MAGI_ATTENTION_FFA_CUTEDSL_CACHE_DIR"
_DISABLE_2CTA_ENV = "MAGI_ATTENTION_FFA_CUTEDSL_DISABLE_2CTA"
# Preserve SM100 dispatch and the default 2-CTA selection.
_KERNEL_SELECTION_ENVS = (_DISABLE_2CTA_ENV, "MAGI_ATTENTION_FFA_CUTEDSL_ARCH")
# Remove DSL filters that could suppress warnings or turn them into errors.
_DSL_WARNING_FILTER_ENVS = ("CUTE_DSL_WARNINGS_AS_ERRORS", "CUTE_DSL_WARNINGS_IGNORE")


def _pinned_conflicts() -> list[str]:
    """Kernel-selection variables the caller pinned to a non-default value."""
    conflicts = []
    for var in _KERNEL_SELECTION_ENVS:
        pinned = os.environ.get(var)
        if var == _DISABLE_2CTA_ENV and pinned == "0":
            pinned = None
        if pinned is not None:
            conflicts.append(f"{var}={pinned}")
    return conflicts


def _gated_deprecations(records: list[dict], cutlass_dir: Path) -> list[dict]:
    """Select deprecations attributed to files under magi_attention or cutlass.

    Both locations are needed because DSL warnings can name the caller via stacklevel.
    """
    source_dirs = (_PACKAGE_DIR, cutlass_dir.resolve())
    return [
        record
        for record in records
        if record["is_deprecation"]
        and any(
            Path(record["filename"]).resolve().is_relative_to(source_dir)
            for source_dir in source_dirs
        )
    ]


def _compile_in_fresh_interpreter(config: CompileConfig, cache_dir: Path) -> dict:
    env = dict(os.environ)
    env.pop(_CACHE_ENABLED_ENV, None)
    env[_CACHE_DIR_ENV] = str(cache_dir)
    for var in (*_DSL_WARNING_FILTER_ENVS, *_KERNEL_SELECTION_ENVS):
        env.pop(var, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PACKAGE_PARENT), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, "-u", str(_CHILD), config.name],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    (line,) = (
        line for line in proc.stdout.splitlines() if line.startswith(RESULT_PREFIX)
    )
    return json.loads(line[len(RESULT_PREFIX) :])


@pytest.mark.parametrize("config", CONFIGS, ids=[config.name for config in CONFIGS])
def test_compile_emits_no_dsl_deprecation_warning(
    config: CompileConfig, tmp_path: Path
) -> None:
    conflicts = _pinned_conflicts()
    if conflicts:
        pytest.skip(f"{config.name} needs {', '.join(conflicts)} unset")

    result = _compile_in_fresh_interpreter(config, tmp_path / "ffa_cutedsl_cache")

    assert (
        Path(result["magi_attention_file"]).resolve()
        == Path(magi_attention.__file__).resolve()
    )
    # A kernel loaded from the persistent cache is not traced.
    assert result["cache_enabled"] is False

    deprecations = _gated_deprecations(result["records"], Path(result["cutlass_dir"]))
    assert not deprecations, (
        f"{len(deprecations)} DeprecationWarning(s) while compiling "
        f"{config.name}:\n"
        + "\n".join(
            f"  {record['filename']}:{record['lineno']}: {record['message']}"
            for record in deprecations
        )
    )
