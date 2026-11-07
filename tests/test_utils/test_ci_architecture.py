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

"""Exercise the root collection policy through isolated real pytest processes."""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestCIArchitecture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copyfile(REPO_ROOT / "conftest.py", self.root / "conftest.py")
        (self.root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        (self.root / "tests").mkdir()
        (self.root / "tests/test_probe.py").write_text(
            textwrap.dedent(
                """\
            import os
            import unittest
            import pytest

            @pytest.mark.arch_independent
            class TestMarked(unittest.TestCase):
                def test_class_marker(self):
                    pass

                @pytest.mark.slow
                def test_slow(self):
                    pass

            class TestMixed(unittest.TestCase):
                @pytest.mark.arch_independent
                def test_method_marker(self):
                    pass

                def test_unmarked(self):
                    assert os.environ.get('MAGI_ATTENTION_TEST_ATTN_CONFIG') == os.environ.get('EXPECTED_FILTER')

                @pytest.mark.skip(reason='existing hardware restriction')
                def test_native_skip(self):
                    pass
            """
            )
        )
        # Stubs control hardware identity, not pytest's policy or results.
        (self.root / "torch.py").write_text(
            "import os\n"
            "if os.environ.get('DENY_HARDWARE_IMPORT'):\n"
            "    raise RuntimeError('unexpected hardware import')\n"
            "class cuda:\n"
            "    @staticmethod\n"
            "    def is_available(): return os.environ.get('HAS_CUDA') == '1'\n"
        )
        package = self.root / "magi_attention/utils"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").touch()
        (package / "__init__.py").touch()
        (package / "arch.py").write_text(
            "import os\n"
            "def is_blackwell():\n"
            "    return os.environ.get('IS_BLACKWELL') == '1'\n"
        )

    def _run(
        self, platform=None, *, cuda=True, blackwell=True, deny_import=False, options=()
    ):
        env = os.environ.copy()
        for key in (
            "TASK_CI_PLATFORM",
            "PYTEST_ADDOPTS",
            "MAGI_ATTENTION_TEST_ATTN_CONFIG",
            "EXPECTED_FILTER",
            "DENY_HARDWARE_IMPORT",
        ):
            env.pop(key, None)
        if platform is not None:
            env["TASK_CI_PLATFORM"] = platform
        env.update(
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONPATH=str(self.root),
            HAS_CUDA=str(int(cuda)),
            IS_BLACKWELL=str(int(blackwell)),
        )
        if deny_import:
            env["DENY_HARDWARE_IMPORT"] = "1"
        if "--test-attn-config" in options:
            env["EXPECTED_FILTER"] = options[options.index("--test-attn-config") + 1]
        # Absolute test path from an unrelated cwd matches the portable runner.
        cwd = self.root / "runner"
        cwd.mkdir(exist_ok=True)
        report = self.root / "result.xml"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--strict-markers",
                "--strict-config",
                "--import-mode=append",
                f"--junitxml={report}",
                *options,
                str(self.root / "tests"),
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if "--collect-only" in options:
            return result.stdout, 0, 0
        suite = ET.parse(report).getroot().find("testsuite")
        return result.stdout, int(suite.attrib["tests"]), int(suite.attrib["skipped"])

    def test_local_and_hopper_run_shared_coverage_without_hardware_import(self):
        for platform in (None, "h100"):
            with self.subTest(platform=platform):
                _, count, skipped = self._run(platform, deny_import=True)
                self.assertEqual((count, skipped), (5, 1))

    def test_blackwell_profiles_skip_only_marked_items(self):
        for platform in ("b300", "b300_fa4"):
            with self.subTest(platform=platform):
                _, count, skipped = self._run(platform)
                self.assertEqual((count, skipped), (5, 4))

    def test_platform_label_without_blackwell_does_not_skip(self):
        _, count, skipped = self._run("b300", blackwell=False)
        self.assertEqual((count, skipped), (5, 1))

    def test_cpu_collection_is_not_blackwell_evidence(self):
        _, count, skipped = self._run("b300", cuda=False)
        self.assertEqual((count, skipped), (5, 1))

    def test_unmarked_collection_does_not_import_hardware(self):
        _, count, skipped = self._run(
            "b300", deny_import=True, options=("-k", "unmarked")
        )
        self.assertEqual((count, skipped), (1, 0))

    def test_slow_skip_and_attention_filter_remain_effective(self):
        _, count, skipped = self._run(
            options=("--skip-slow", "--test-attn-config", "full_attn_*")
        )
        self.assertEqual((count, skipped), (5, 2))

    def test_collection_nodeids_do_not_depend_on_platform(self):
        collections = []
        for platform in (None, "h100", "b300", "b300_fa4"):
            output, _, _ = self._run(platform, options=("--collect-only",))
            collections.append(
                [
                    line
                    for line in output.splitlines()
                    if line.startswith("tests/test_probe.py::")
                ]
            )
        self.assertEqual(len(collections[0]), 5)
        for collection in collections[1:]:
            self.assertEqual(collection, collections[0])


if __name__ == "__main__":
    unittest.main()
