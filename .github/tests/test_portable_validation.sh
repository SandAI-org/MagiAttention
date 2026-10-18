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
cp "$repo_root/.github/scripts/ci_platforms.py" "$standalone/.github/scripts/ci_platforms.py"
cp "$repo_root/.github/scripts/ci_platforms.py" "$consumer/vendor/MagiAttention/.github/scripts/ci_platforms.py"
cp "$repo_root/.github/configs/ci_platforms.json" "$standalone/.github/configs/ci_platforms.json"
cp "$repo_root/.github/configs/ci_platforms.json" "$consumer/vendor/MagiAttention/.github/configs/ci_platforms.json"
cp "$repo_root/.github/configs/ci_constraints.txt" "$standalone/.github/configs/ci_constraints.txt"
cp "$repo_root/.github/configs/ci_constraints.txt" "$consumer/vendor/MagiAttention/.github/configs/ci_constraints.txt"
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
# The flash_attn_cute install records the PIP_CONSTRAINT it receives and
# fails, so run-test stops before pytest.
mkdir -p "$standalone/scripts" "$consumer/vendor/MagiAttention/scripts"
printf '%s\n' \
    '#!/bin/bash' \
    'echo "$PIP_CONSTRAINT" > "$FAKE_PIP_CONSTRAINT_RECORD"' \
    'exit 1' \
    > "$standalone/scripts/install_flash_attn_cute.sh"
cp "$standalone/scripts/install_flash_attn_cute.sh" "$consumer/vendor/MagiAttention/scripts/install_flash_attn_cute.sh"

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

standalone_main_b300=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300 bash .github/scripts/portable_validation.sh fingerprint magi_attention)
[[ "$standalone_main" != "$standalone_main_b300" ]]

standalone_ext=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 bash .github/scripts/portable_validation.sh fingerprint magi_attn_extensions)
consumer_ext=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh fingerprint magi_attn_extensions)
[[ "$standalone_ext" == "$consumer_ext" ]]

standalone_main_fa4=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300_fa4 bash .github/scripts/portable_validation.sh fingerprint magi_attention)
consumer_main_fa4=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300_fa4 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh fingerprint magi_attention)
[[ "$standalone_main_fa4" == "$consumer_main_fa4" ]]
[[ "$standalone_main_fa4" != "$standalone_main_b300" ]]

(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 VALIDATION_CACHE_TRUSTED=true GITHUB_REPOSITORY=SandAI-org/MagiAttention GITHUB_SHA=test \
    bash .github/scripts/portable_validation.sh write-success magi_attention >/dev/null)
(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt \
    bash vendor/MagiAttention/.github/scripts/portable_validation.sh verify magi_attention)

marker_dir=$(cd "$consumer" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 PORTABLE_SOURCE_ROOT=vendor/MagiAttention PORTABLE_BASE_TAG_FILE=runtime_tag.txt bash vendor/MagiAttention/.github/scripts/portable_validation.sh marker-dir magi_attention)
[[ "$marker_dir" == *"/magi-attention/h100/"* ]]
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

# run-test passes every PIP_CONSTRAINT file to the flash_attn_cute install:
# an inherited file of the base image stays, and the relative profile file
# becomes absolute because the install runs pip in another directory.
constraint_record="$workspace/install_constraint.txt"
standalone_root=$(git -C "$standalone" rev-parse --show-toplevel)
mkdir -p "$workspace/runner_temp"
for inherited in "" /etc/pip/constraint.txt; do
    rm -f "$constraint_record"
    if (cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300_fa4 \
        RUNNER_TEMP="$workspace/runner_temp" PIP_CONSTRAINT="$inherited" \
        FAKE_PIP_CONSTRAINT_RECORD="$constraint_record" \
        bash .github/scripts/portable_validation.sh run-test magi_attention >/dev/null 2>&1); then
        echo "Expected the failing flash_attn_cute install to fail run-test" >&2
        exit 1
    fi
    expected_constraint="${inherited:+$inherited }$standalone_root/.github/configs/ci_constraints.txt"
    [[ "$(cat "$constraint_record")" == "$expected_constraint" ]] || {
        echo "Expected install PIP_CONSTRAINT '$expected_constraint', got '$(cat "$constraint_record")'" >&2
        exit 1
    }
done

# The b300_fa4 profile names its pip constraint file by path, and .github is
# not source input, so only the recipe can carry the constraint content: a
# content change must move the b300_fa4 fingerprint and leave h100 unchanged.
printf '# changed\n' >> "$standalone/.github/configs/ci_constraints.txt"
git -C "$standalone" commit -qam "change constraints"
changed_main_fa4=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300_fa4 bash .github/scripts/portable_validation.sh fingerprint magi_attention)
changed_main=$(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=h100 bash .github/scripts/portable_validation.sh fingerprint magi_attention)
[[ "$changed_main_fa4" != "$standalone_main_fa4" ]]
[[ "$changed_main" == "$standalone_main" ]]

git -C "$standalone" rm -q .github/configs/ci_constraints.txt
git -C "$standalone" commit -qm "remove constraints"
set +e
(cd "$standalone" && CI_WORKSPACE_ROOT="$shared" TASK_CI_PLATFORM=b300_fa4 \
    bash .github/scripts/portable_validation.sh fingerprint magi_attention >/dev/null 2>&1)
missing_rc=$?
set -e
[[ $missing_rc -ne 0 ]] || {
    echo "Expected a missing PIP_CONSTRAINT file to fail the fingerprint" >&2
    exit 1
}

echo "portable validation tests passed"
