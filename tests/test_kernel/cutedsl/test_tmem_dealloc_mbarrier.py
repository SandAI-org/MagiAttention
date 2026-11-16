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

# A minimal two-CTA allocator probe checks both generated initialization
# and completion with no work tile; it does not assert attention numerics.

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

import magi_attention
from magi_attention.utils.arch import is_sm10x

_CHILD = Path(__file__).with_name("tmem_dealloc_mbarrier_test_utils.py")
# Keep child imports tied to the package under test.
_PACKAGE_PARENT = Path(magi_attention.__file__).resolve().parent.parent
_CHILD_TIMEOUT_S = 120

RESULT_PREFIX = "TMEM_DEALLOC_MBARRIER_RESULT "


def _trace_in_fresh_interpreter(dump_dir: Path) -> dict:
    env = dict(os.environ)
    env["CUTE_DSL_KEEP_IR"] = "1"
    env["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_PACKAGE_PARENT), *filter(None, [env.get("PYTHONPATH")])]
    )
    proc = subprocess.run(
        [sys.executable, "-u", str(_CHILD)],
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


class TestTmemDeallocMbarrier(TestCase):
    @unittest.skipUnless(is_sm10x(), "requires SM10.x")
    def test_allocator_reconstruction_initializes_dealloc_mbarrier_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _trace_in_fresh_interpreter(Path(tmp_dir))

        self.assertEqual(
            Path(result["magi_attention_file"]).resolve(),
            Path(magi_attention.__file__).resolve(),
        )
        self.assertEqual(result["arrival_counts"], [32], result)
        self.assertEqual(result["completed_ctas"], [1, 1], result)


if __name__ == "__main__":
    unittest.main()
