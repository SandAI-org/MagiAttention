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

"""Preflight contracts with mocked CUDA executables, not runner qualification."""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / ".github/scripts/verify_task_runner.sh"
_DECLARED_TAG = (_ROOT / ".github/configs/base_image_tag.txt").read_text().strip()


class TestTaskRunnerPreflight(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        commands = self.root / "bin"
        commands.mkdir()
        # Run the real platform/profile validation; mock only the CUDA probes.
        python = commands / "python"
        python.write_text(
            "#!/bin/sh\n"
            'if [ "${1:-}" = "-" ]; then cat >/dev/null; echo mocked-CUDA-probe; exit 0; fi\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        smi = commands / "nvidia-smi"
        smi.write_text("#!/bin/sh\necho mocked-nvidia-smi\n")
        python.chmod(0o755)
        smi.chmod(0o755)
        self.env = dict(os.environ)
        # Host shell initialization can replace PATH and bypass the probe mocks.
        self.env.pop("BASH_ENV", None)
        self.env.pop("ENV", None)
        self.env.pop("MAGI_BASE_IMAGE_TAG", None)
        self.env.pop("TASK_CI_BASE_IMAGE", None)
        self.env.update(
            PATH=os.pathsep.join([str(commands), self.env.get("PATH", "")]),
            VALIDATION_CACHE_TRUSTED="false",
            RUNNER_TEMP=str(self.root),
            CI_WORKSPACE_ROOT=str(self.root / "magi-attention-ci"),
            TASK_CI_PLATFORM="h100",
        )

    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(_SCRIPT)],
            cwd=_ROOT,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_optional_runtime_report(self) -> None:
        for reported in (None, _DECLARED_TAG):
            with self.subTest(reported=reported):
                if reported is not None:
                    self.env["MAGI_BASE_IMAGE_TAG"] = reported
                result = self._run()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"declared_base_tag={_DECLARED_TAG}", result.stdout)
                self.assertIn(
                    f"runtime_reported_base_tag={reported or '<unset>'}", result.stdout
                )
                self.assertIn("mocked-CUDA-probe", result.stdout)
                self.assertFalse(
                    list((self.root / "magi-attention-ci").glob(".*write-probe.*"))
                )

    def test_runtime_tag_mismatch_is_rejected(self) -> None:
        self.env["MAGI_BASE_IMAGE_TAG"] = "0.0.0"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match declared", result.stderr)
        self.assertNotIn("mocked-CUDA-probe", result.stdout)

    def test_untrusted_storage_boundary_is_preserved(self) -> None:
        self.env["CI_WORKSPACE_ROOT"] = str(self.root / "wrong")
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Untrusted CI must use runner-local storage", result.stderr)
        self.assertFalse((self.root / "wrong").exists())

    def test_invalid_platform_is_rejected(self) -> None:
        self.env["TASK_CI_PLATFORM"] = "not-a-platform"
        result = self._run()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("mocked-CUDA-probe", result.stdout)


if __name__ == "__main__":
    unittest.main()
