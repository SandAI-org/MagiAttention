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

import os
import subprocess
import sys

import magi_attention

# A flash_attn_cute built against another CUTLASS DSL release fails while its
# package body executes, e.g. on a removed `cute.core.ThrMma` annotation.
_UNLOADABLE_FA4_INIT = (
    "raise AttributeError(\"module 'cutlass.cute.core' has no attribute 'ThrMma'\")\n"
)

_PROBE = (
    "import magi_attention\n"
    "from magi_attention.functional import fa4\n"
    "from magi_attention.meta.collection import calc_meta\n"
    "print(magi_attention.__file__)\n"
    "print(calc_meta.is_fa4_installed, fa4.is_fa4_installed)\n"
    "from magi_attention.common import AttnRanges\n"
    "try:\n"
    "    calc_meta.FA4AttnArg(\n"
    "        q_ranges=AttnRanges(), k_ranges=AttnRanges(), attn_type_map=[]\n"
    "    )\n"
    "except AssertionError as exc:\n"
    "    print(exc)\n"
    "else:\n"
    "    print('FA4AttnArg raised no AssertionError')\n"
)


def test_unloadable_flash_attn_cute_counts_as_not_installed(tmp_path):
    stub_pkg = tmp_path / "flash_attn_cute"
    stub_pkg.mkdir()
    (stub_pkg / "__init__.py").write_text(_UNLOADABLE_FA4_INIT)

    # The package import is cached per process, so the stub must shadow the
    # real flash_attn_cute in a fresh interpreter.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(tmp_path), env.get("PYTHONPATH")) if p
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    magi_file, flags, not_installed_msg = result.stdout.strip().splitlines()[-3:]
    assert magi_file == magi_attention.__file__
    assert flags == "False False"
    # The not-installed error must include the import exception text.
    assert "has no attribute 'ThrMma'" in not_installed_msg, not_installed_msg
