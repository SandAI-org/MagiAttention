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

"""Select and verify the nvidia-cutlass-dsl / quack-kernels pair for CI installs.

If a file in PIP_CONSTRAINT pins either package with ``==``, the pins are the
expected pair; otherwise the expected pair is the upper bound of each package in
requirements.txt, read from the current directory. Commands:

- ``source``: print ``constraint`` or ``upgrade``.
- ``upgrade-requirements``: print the requirements.txt line of each package.
- ``expected``: print the expected pair.
- ``verify``: print the installed pair; exit non-zero if it differs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels")
REQUIREMENT_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def requirements(path: str) -> dict[str, tuple[str, Requirement]]:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}")
    found = {}
    # Leave unrelated requirements and pip options to pip.
    for number, line in enumerate(lines, start=1):
        text = line.split("#", 1)[0].strip()
        name = REQUIREMENT_NAME_RE.match(text)
        if name is None or canonicalize_name(name.group()) not in PACKAGES:
            continue
        try:
            requirement = Requirement(text)
        except InvalidRequirement as exc:
            raise SystemExit(f"{path}:{number}: {exc}")
        found[canonicalize_name(requirement.name)] = (text, requirement)
    return found


def declared() -> dict[str, tuple[str, Requirement]]:
    found = requirements("requirements.txt")
    missing = [name for name in PACKAGES if name not in found]
    if missing:
        raise SystemExit(f"requirements.txt does not declare {', '.join(missing)}")
    return found


def upper_bound(path: str, requirement: Requirement) -> str:
    bounds = [s.version for s in requirement.specifier if s.operator in ("<=", "==")]
    if len(bounds) != 1:
        raise SystemExit(f"{path}: {requirement} needs exactly one '<=' or '==' bound")
    return bounds[0]


def constraint_pins() -> dict[str, tuple[str, str]]:
    pins = {}
    for path in os.environ.get("PIP_CONSTRAINT", "").split():
        for name, (_, requirement) in requirements(path).items():
            pinned = [s.version for s in requirement.specifier if s.operator == "=="]
            if pinned:
                pins[name] = (path, pinned[0])
    return pins


def expected() -> dict[str, str]:
    pins = constraint_pins()
    if pins:
        return {name: version for name, (_, version) in pins.items()}
    return {
        name: upper_bound("requirements.txt", req)
        for name, (_, req) in declared().items()
    }


def installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("source", "upgrade-requirements", "expected", "verify")
    )
    command = parser.parse_args().command
    if command == "source":
        print("constraint" if constraint_pins() else "upgrade")
    elif command == "upgrade-requirements":
        for name in PACKAGES:
            print(declared()[name][0])
    elif command == "expected":
        pair = expected()
        print(
            "Expected CuTeDSL pair: " + ", ".join(f"{n}=={v}" for n, v in pair.items())
        )
    elif command == "verify":
        pair = expected()
        versions = {name: installed(name) for name in PACKAGES}
        print(
            "Installed CuTeDSL pair: "
            + ", ".join(f"{n}=={v}" for n, v in versions.items())
        )
        errors = [
            f"{name}: expected {version}, installed {versions[name]}"
            for name, version in pair.items()
            if versions[name] != version
        ]
        if errors:
            raise SystemExit(
                "Installed CuTeDSL pair does not match:\n  " + "\n  ".join(errors)
            )


if __name__ == "__main__":
    main()
