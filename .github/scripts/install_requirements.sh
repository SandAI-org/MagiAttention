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
    python "$(dirname "${BASH_SOURCE[0]}")/cutedsl_pair.py" "${1:?command is required}"
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
