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

# A 2-CTA kernel must initialize the TmemAllocator dealloc mbarrier exactly
# once. A second init can erase the dealloc arrival of the peer CTA, after
# which TmemAllocator.free never returns.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from torch.testing._internal.common_utils import run_tests

import magi_attention
from magi_attention.kernel.cutedsl.ffa_utils import (
    get_device_arch,
    is_ffa_2cta_disabled,
)
from tests.test_kernel.cutedsl.tmem_dealloc_mbarrier_test_utils import RESULT_PREFIX

_CHILD = Path(__file__).with_name("tmem_dealloc_mbarrier_test_utils.py")
# Keep child imports tied to the package under test.
_PACKAGE_PARENT = Path(magi_attention.__file__).resolve().parent.parent
_CHILD_TIMEOUT_S = 120

_CACHE_ENABLED_ENV = "MAGI_ATTENTION_FFA_CUTEDSL_CACHE_ENABLED"

# TmemAllocator initializes the dealloc mbarrier for the 32 threads of the
# allocator warp. No other mbarrier of the SM100 backward kernel takes 32
# arrivals.
_DEALLOC_ARRIVAL_COUNT = 32


def _trace_in_fresh_interpreter(dump_dir: Path) -> dict:
    env = dict(os.environ)
    # A kernel loaded from the persistent cache is not traced and has no IR.
    env.pop(_CACHE_ENABLED_ENV, None)
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
    def setUp(self) -> None:
        if not torch.cuda.is_available() or get_device_arch()[1] != 10:
            self.skipTest("the check traces and launches the SM100 kernels")
        if is_ffa_2cta_disabled():
            self.skipTest("the backward kernel is 1-CTA when 2-CTA is disabled")

    def test_2cta_bwd_initializes_dealloc_mbarrier_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = _trace_in_fresh_interpreter(Path(tmp_dir))

        assert (
            Path(result["magi_attention_file"]).resolve()
            == Path(magi_attention.__file__).resolve()
        )
        arrival_counts = result["arrival_counts"]
        # A 1-CTA kernel has no dealloc mbarrier, so one init also shows that
        # the traced kernel is the 2-CTA one.
        dealloc_inits = arrival_counts.count(_DEALLOC_ARRIVAL_COUNT)
        assert dealloc_inits == 1, (
            f"the backward kernel initializes the dealloc mbarrier "
            f"{dealloc_inits} times on nvidia-cutlass-dsl "
            f"{result['cutlass_version']}; arrival counts of all "
            f"{len(arrival_counts)} mbarrier inits: {sorted(arrival_counts)}"
        )


if __name__ == "__main__":
    run_tests()
