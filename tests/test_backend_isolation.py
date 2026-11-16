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

"""Run import-selected Python/C++ contracts in separate fresh interpreters."""

import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import magi_attention

_CASES = Path(__file__).with_name("backend_cases")
_PACKAGE_PARENT = Path(magi_attention.__file__).resolve().parent.parent


@pytest.mark.arch_independent
class TestBackendIsolation(unittest.TestCase):
    def _run_backend(self, use_cpp: bool) -> None:
        env = dict(os.environ, MAGI_ATTENTION_CPP_BACKEND=str(int(use_cpp)))
        # Preserve wheel-vs-source selection made by the invoking test runner.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_PACKAGE_PARENT), *filter(None, [env.get("PYTHONPATH")])]
        )
        runner = [sys.executable, "-m", "pytest"]
        if env.get("COVERAGE_RUN", "").lower() == "true":
            runner = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "--source=magi_attention",
                "-m",
                "pytest",
            ]
        case_files = sorted(_CASES.glob("*_cases.py"))
        self.assertTrue(case_files, "no backend case modules found")
        with tempfile.TemporaryDirectory() as tmp_dir:
            report = Path(tmp_dir) / "results.xml"
            proc = subprocess.run(
                [
                    *runner,
                    "-q",
                    "--import-mode=append",
                    f"--junitxml={report}",
                    *(str(path) for path in case_files),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            suites = list(ET.parse(report).getroot().iter("testsuite"))
            total = sum(int(suite.attrib["tests"]) for suite in suites)
            skipped = sum(int(suite.attrib["skipped"]) for suite in suites)
            self.assertGreater(total - skipped, 0, "backend child executed no tests")
        print(
            f"backend={'cpp' if use_cpp else 'python'}: {total - skipped} ran, {skipped} skipped"
        )

    def test_python_backend(self) -> None:
        self._run_backend(False)

    def test_cpp_backend(self) -> None:
        self._run_backend(True)


if __name__ == "__main__":
    unittest.main()
