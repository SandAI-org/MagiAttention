#!/usr/bin/env python3

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

from __future__ import annotations

import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/install_source_dependencies.py"
spec = importlib.util.spec_from_file_location("source_dependencies", SCRIPT)
assert spec and spec.loader
source_dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source_dependencies)


def test_git_auth_policy() -> None:
    commands = []
    original_run = source_dependencies.subprocess.run
    original_token = os.environ.get("DEPENDENCY_REPO_TOKEN")
    original_opt_in = os.environ.get("CI_DEPENDENCY_USE_TOKEN")

    def capture_run(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="resolved\n")

    try:
        source_dependencies.subprocess.run = capture_run
        os.environ["DEPENDENCY_REPO_TOKEN"] = "invalid-inherited-token"
        os.environ.pop("CI_DEPENDENCY_USE_TOKEN", None)
        source_dependencies.git(["ls-remote", "https://github.com/example/repo"])
        assert "credential.helper=" in commands[-1]
        assert "http.https://github.com/.extraheader=" in commands[-1]
        assert not any("AUTHORIZATION" in argument for argument in commands[-1])

        os.environ["CI_DEPENDENCY_USE_TOKEN"] = "true"
        source_dependencies.git(["ls-remote", "https://github.com/example/repo"])
        reset_index = commands[-1].index("http.https://github.com/.extraheader=")
        auth_index = next(
            index
            for index, argument in enumerate(commands[-1])
            if "AUTHORIZATION" in argument
        )
        assert reset_index < auth_index
        assert any("AUTHORIZATION" in argument for argument in commands[-1])
    finally:
        source_dependencies.subprocess.run = original_run
        if original_token is None:
            os.environ.pop("DEPENDENCY_REPO_TOKEN", None)
        else:
            os.environ["DEPENDENCY_REPO_TOKEN"] = original_token
        if original_opt_in is None:
            os.environ.pop("CI_DEPENDENCY_USE_TOKEN", None)
        else:
            os.environ["CI_DEPENDENCY_USE_TOKEN"] = original_opt_in


def run(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def main() -> None:
    test_git_auth_policy()
    with tempfile.TemporaryDirectory(prefix="dependency-cache-test.") as temporary:
        root = Path(temporary)
        checkout = root / "dependency"
        checkout.mkdir()
        run("git", "init", "-q", cwd=checkout)
        run("git", "config", "user.name", "CI", cwd=checkout)
        run("git", "config", "user.email", "ci@example.invalid", cwd=checkout)
        (checkout / "package.py").write_text("VALUE = 1\n")
        run("git", "add", ".", cwd=checkout)
        run("git", "commit", "-qm", "initial", cwd=checkout)

        builds = 0
        original_run = source_dependencies.subprocess.run

        def fake_run(args, *positional, **kwargs):
            nonlocal builds
            if len(args) >= 3 and args[1:3] == ["-m", "build"]:
                builds += 1
                output = Path(args[args.index("--outdir") + 1])
                (output / "example-1-py3-none-any.whl").write_bytes(
                    f"wheel-{builds}".encode()
                )
                return subprocess.CompletedProcess(args, 0)
            return original_run(args, *positional, **kwargs)

        source_dependencies.subprocess.run = fake_run
        source_dependencies.install_wheels = lambda wheels: None  # type: ignore[attr-defined]
        common = {
            "checkout": checkout,
            "relative": ".",
            "dependency_id": "example",
            "repository": "example/dependency",
            "consumer": "consumer",
            "cache_root": root / "cache",
            "base_tag": "26.05.2",
            "platform": "h100",
            "recipe": "recipe",
        }
        first_source, first_artifact = source_dependencies.build_or_reuse(**common)
        second_source, second_artifact = source_dependencies.build_or_reuse(**common)
        assert builds == 1
        assert first_source == second_source
        assert first_artifact == second_artifact

        wheel = next(first_artifact.glob("*.whl"))
        wheel.write_bytes(b"corrupt")
        try:
            source_dependencies.build_or_reuse(**common)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt dependency wheel was accepted")

        wheel.write_bytes(b"wheel-1")
        (checkout / "package.py").write_text("VALUE = 2\n")
        run("git", "add", ".", cwd=checkout)
        run("git", "commit", "-qm", "change", cwd=checkout)
        changed_source, changed_artifact = source_dependencies.build_or_reuse(**common)
        assert builds == 2
        assert changed_source != first_source
        assert changed_artifact != first_artifact

    print("source dependency cache tests passed")


if __name__ == "__main__":
    main()
