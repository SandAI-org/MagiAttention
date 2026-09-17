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

repo_root=$(git rev-parse --show-toplevel)
protocol="$repo_root/.github/scripts/portable_validation.sh"
policy="$repo_root/.github/configs/ci_input_policy.json"
policy_helper="$repo_root/.github/scripts/ci_input_policy.py"
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT

standalone="$workspace/standalone"
consumer="$workspace/consumer"
shared="$workspace/shared"
mkdir -p "$standalone/.github/scripts" "$standalone/.github/workflows"
mkdir -p "$standalone/magi_attention" "$standalone/extensions/tests" "$standalone/tests"
mkdir -p "$consumer/vendor/MagiAttention/.github/scripts"
mkdir -p "$consumer/vendor/MagiAttention/magi_attention"
mkdir -p "$consumer/vendor/MagiAttention/extensions/tests" "$consumer/vendor/MagiAttention/tests"

cp "$protocol" "$standalone/.github/scripts/portable_validation.sh"
cp "$protocol" "$consumer/vendor/MagiAttention/.github/scripts/portable_validation.sh"
mkdir -p "$standalone/.github/configs" "$consumer/vendor/MagiAttention/.github/configs"
cp "$policy" "$standalone/.github/configs/ci_input_policy.json"
cp "$policy" "$consumer/vendor/MagiAttention/.github/configs/ci_input_policy.json"
cp "$policy_helper" "$standalone/.github/scripts/ci_input_policy.py"
cp "$policy_helper" "$consumer/vendor/MagiAttention/.github/scripts/ci_input_policy.py"
printf '26.05.2\n' > "$standalone/.github/configs/base_image_tag.txt"
printf '26.05.2\n' > "$consumer/runtime_tag.txt"
printf 'main source\n' > "$standalone/magi_attention/package.py"
printf 'main test\n' > "$standalone/tests/test.txt"
printf 'extension source\n' > "$standalone/extensions/package.txt"
printf 'extension test\n' > "$standalone/extensions/tests/test.txt"
cp "$standalone/magi_attention/package.py" "$consumer/vendor/MagiAttention/magi_attention/package.py"
cp "$standalone/tests/test.txt" "$consumer/vendor/MagiAttention/tests/test.txt"
cp "$standalone/extensions/package.txt" "$consumer/vendor/MagiAttention/extensions/package.txt"
cp "$standalone/extensions/tests/test.txt" "$consumer/vendor/MagiAttention/extensions/tests/test.txt"

git -C "$standalone" init -q
git -C "$standalone" config user.name test
git -C "$standalone" config user.email test@example.com
git -C "$standalone" add .
git -C "$standalone" commit -qm fixture
git -C "$consumer" init -q
git -C "$consumer" config user.name test
git -C "$consumer" config user.email test@example.com
git -C "$consumer" add .
git -C "$consumer" commit -qm fixture

standalone_main=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 bash .github/scripts/portable_validation.sh fingerprint magi_attention)
consumer_main=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh fingerprint magi_attention)
[[ "$standalone_main" == "$consumer_main" ]]

standalone_ext=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 bash .github/scripts/portable_validation.sh fingerprint magi_attn_extensions)
consumer_ext=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh fingerprint magi_attn_extensions)
[[ "$standalone_ext" == "$consumer_ext" ]]

(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 VALIDATION_CACHE_TRUSTED=true GITHUB_REPOSITORY=SandAI-org/MagiAttention GITHUB_SHA=test \
    bash .github/scripts/portable_validation.sh write-success magi_attention >/dev/null)
(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt \
    bash vendor/MagiAttention/.github/scripts/portable_validation.sh verify magi_attention)

marker_dir=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh marker-dir magi_attention)
printf '{bad json\n' > "$marker_dir/success.json"
set +e
(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt \
    bash vendor/MagiAttention/.github/scripts/portable_validation.sh verify magi_attention >/dev/null 2>&1)
verify_rc=$?
set -e
[[ $verify_rc -eq 3 ]] || {
    echo "Expected malformed portable marker to return 3, got $verify_rc" >&2
    exit 1
}

echo "portable validation tests passed"
