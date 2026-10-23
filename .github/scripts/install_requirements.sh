#!/bin/bash

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

# If PIP_CONSTRAINT pins either CuTeDSL package with ==, verify those pins;
# otherwise upgrade only the pair and require its requirements.txt upper bounds.

set -euo pipefail

echo "=== Requirements Installation ==="

unset http_proxy
unset https_proxy

pip_index_url=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}
echo "Using Python package index: $pip_index_url"

cutedsl_pair() {
    python - "${1:?command is required}" <<'PY'
import importlib.metadata
import os
import re
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

PACKAGES = ("nvidia-cutlass-dsl", "quack-kernels")
command = sys.argv[1]


def requirements(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}")
    found = {}
    # Leave unrelated requirements and pip options to pip.
    for number, line in enumerate(lines, start=1):
        text = line.split("#", 1)[0].strip()
        name = re.match(r"[A-Za-z0-9._-]*", text).group()
        if not name or canonicalize_name(name) not in PACKAGES:
            continue
        try:
            requirement = Requirement(text)
        except InvalidRequirement as exc:
            raise SystemExit(f"{path}:{number}: {exc}")
        found[canonicalize_name(requirement.name)] = (text, requirement)
    return found


def declared():
    found = requirements("requirements.txt")
    missing = [name for name in PACKAGES if name not in found]
    if missing:
        raise SystemExit(f"requirements.txt does not declare {', '.join(missing)}")
    return found


def upper_bound(path, requirement):
    bounds = [s.version for s in requirement.specifier if s.operator in ("<=", "==")]
    if len(bounds) != 1:
        raise SystemExit(f"{path}: {requirement} needs exactly one '<=' or '==' bound")
    return bounds[0]


def constraint_pins():
    pins = {}
    for path in os.environ.get("PIP_CONSTRAINT", "").split():
        for name, (_, requirement) in requirements(path).items():
            pinned = [s.version for s in requirement.specifier if s.operator == "=="]
            if pinned:
                pins[name] = (path, pinned[0])
    return pins


def expected():
    pins = constraint_pins()
    if pins:
        return {name: version for name, (_, version) in pins.items()}
    return {name: upper_bound("requirements.txt", req) for name, (_, req) in declared().items()}


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


if command == "source":
    print("constraint" if constraint_pins() else "upgrade")
elif command == "upgrade-requirements":
    for name in PACKAGES:
        print(declared()[name][0])
elif command == "expected":
    pair = expected()
    print("Expected CuTeDSL pair: " + ", ".join(f"{n}=={v}" for n, v in pair.items()))
elif command == "verify":
    pair = expected()
    versions = {name: installed(name) for name in PACKAGES}
    print("Installed CuTeDSL pair: " + ", ".join(f"{n}=={v}" for n, v in versions.items()))
    errors = [
        f"{name}: expected {version}, installed {versions[name]}"
        for name, version in pair.items()
        if versions[name] != version
    ]
    if errors:
        raise SystemExit("Installed CuTeDSL pair does not match:\n  " + "\n  ".join(errors))
else:
    raise SystemExit(f"unknown command: {command}")
PY
}

if [ -f requirements.txt ]; then
    echo "📦 Installing requirements.txt..."
    pip install -r requirements.txt --index-url "$pip_index_url"
    echo "✅ Base requirements installed successfully!"
else
    echo "⚠️  No requirements.txt found. Skipping."
fi

cutedsl_source=$(cutedsl_pair source)
echo "CuTeDSL pair source: $cutedsl_source"
cutedsl_pair expected
if [[ "$cutedsl_source" == upgrade ]]; then
    cutedsl_requirements=$(cutedsl_pair upgrade-requirements)
    mapfile -t cutedsl_requirements <<< "$cutedsl_requirements"
    echo "Upgrading the CuTeDSL pair: ${cutedsl_requirements[*]}"
    pip install --upgrade "${cutedsl_requirements[@]}" --index-url "$pip_index_url"
fi

if [ -f requirements_dev.txt ]; then
    echo "🛠️  Installing development requirements..."
    pip install -r requirements_dev.txt --index-url "$pip_index_url"
    echo "✅ Development requirements installed successfully!"
else
    echo "⚠️  No requirements_dev.txt found. Skipping."
fi

cutedsl_pair verify
