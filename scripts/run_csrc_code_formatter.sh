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

set -euo pipefail

CLANG_FORMAT_VERSION=20.1.8
CLANG_FORMAT_BIN="clang-format-${CLANG_FORMAT_VERSION%%.*}"

if ! command -v "$CLANG_FORMAT_BIN" >/dev/null 2>&1; then
    echo "Error: ${CLANG_FORMAT_BIN} not found. Run sudo bash scripts/install_clang_format.sh."
    exit 1
fi
INSTALLED_VERSION=$("$CLANG_FORMAT_BIN" --version | sed -nE 's/.*version ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p')
if [ "$INSTALLED_VERSION" != "$CLANG_FORMAT_VERSION" ]; then
    echo "Error: expected clang-format ${CLANG_FORMAT_VERSION}, found ${INSTALLED_VERSION}."
    echo "Run sudo bash scripts/install_clang_format.sh."
    exit 1
fi
exec "$CLANG_FORMAT_BIN" "$@"
