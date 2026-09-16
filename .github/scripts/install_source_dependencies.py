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

"""Resolve, checkout, and install CI dependencies from immutable Git commits."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def git(args: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    command = ["git"]
    token = os.environ.get("DEPENDENCY_REPO_TOKEN", "")
    if token:
        credential = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        command += [
            "-c",
            f"http.https://github.com/.extraheader=AUTHORIZATION: basic {credential}",
        ]
    result = subprocess.run(
        command + args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def resolve(url: str, ref_type: str, ref: str) -> str:
    if ref_type == "commit":
        if len(ref) != 40 or any(char not in "0123456789abcdefABCDEF" for char in ref):
            raise SystemExit(f"Commit refs must be full 40-character SHAs: {ref!r}")
        return ref.lower()
    query = f"refs/heads/{ref}" if ref_type == "branch" else f"refs/tags/{ref}^{{}}"
    output = git(["ls-remote", url, query], capture=True)
    if ref_type == "tag" and not output:
        output = git(
            ["ls-remote", "--exit-code", url, f"refs/tags/{ref}"], capture=True
        )
    lines = [line for line in output.splitlines() if line]
    if ref_type not in {"branch", "tag"} or len(lines) != 1:
        raise SystemExit(f"Cannot resolve {ref_type}:{ref} in {url}")
    return lines[0].split()[0]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config = json.loads((repo_root / ".github/ci_dependencies.json").read_text())
    if config.get("schema_version") != 1:
        raise SystemExit("Unsupported CI dependency schema")
    checkout_root = Path(
        os.environ.get(
            "CI_DEPENDENCY_ROOT",
            Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "ci-source-dependencies",
        )
    )
    lock_path = Path(
        os.environ.get("CI_DEPENDENCY_RUNTIME_LOCK", checkout_root / "resolved.json")
    )
    checkout_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, object] = {"schema_version": 1, "repositories": {}}
    for dependency in config.get("repositories", []):
        dep_id, repository = dependency["id"], dependency["repository"]
        url = f"https://github.com/{repository}.git"
        commit = resolve(url, dependency["ref_type"], dependency["ref"])
        checkout = checkout_root / dep_id
        shutil.rmtree(checkout, ignore_errors=True)
        git(["init", str(checkout)])
        git(["remote", "add", "origin", url], cwd=checkout)
        git(["fetch", "--depth=1", "origin", commit], cwd=checkout)
        git(["checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        if git(["rev-parse", "HEAD"], cwd=checkout, capture=True) != commit:
            raise SystemExit(f"Checkout identity mismatch for {repository}")
        git(["submodule", "update", "--init", "--recursive", "--depth=1"], cwd=checkout)
        for relative in dependency.get("install_paths", ["."]):
            package_path = checkout / relative
            requirements = package_path / "requirements.txt"
            if requirements.is_file():
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                    check=True,
                )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-build-isolation",
                    "--no-deps",
                    "--force-reinstall",
                    str(package_path),
                ],
                check=True,
            )
        resolved["repositories"][dep_id] = {  # type: ignore[index]
            "path": str(checkout),
            "repository": repository,
            "requested_ref": dependency["ref"],
            "requested_ref_type": dependency["ref_type"],
            "resolved_commit": commit,
        }
        print(
            f"Resolved {dep_id}: {repository} {dependency['ref_type']}:{dependency['ref']} -> {commit}"
        )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"Resolved dependency lock: {lock_path}")


if __name__ == "__main__":
    main()
