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

# Check CuTeDSL pair selection without installing packages.
# The pip stub records arguments; optional upgrades replace test metadata.

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
script="$repo_root/.github/scripts/install_requirements.sh"
workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT

mkdir -p "$workspace/bin" "$workspace/project" "$workspace/project_strict_upper"
printf '%s\n' \
    '#!/bin/bash' \
    'echo "pip $*" >> "$FAKE_PIP_LOG"' \
    'if [[ -n "${FAKE_PIP_UPGRADED_SITE:-}" && " $* " == *" --upgrade "* ]]; then' \
    '    rm -rf "$PYTHONPATH"' \
    '    cp -R "$FAKE_PIP_UPGRADED_SITE" "$PYTHONPATH"' \
    'fi' \
    'exit "${FAKE_PIP_STATUS:-0}"' \
    > "$workspace/bin/pip"
chmod +x "$workspace/bin/pip"
pip_log="$workspace/pip.log"
output="$workspace/output.log"

printf '%s\n' \
    'torch' \
    'nvidia-cutlass-dsl[cu13]>=4.4.2,<=4.8.0,!=4.6.0,!=4.6.1  # the DSL' \
    'quack-kernels>=0.4.1,<=0.6.5' \
    > "$workspace/project/requirements.txt"
printf '%s\n' \
    'nvidia-cutlass-dsl[cu13]>=4.4.2,<4.9' \
    'quack-kernels>=0.4.1,<=0.6.5' \
    > "$workspace/project_strict_upper/requirements.txt"

# Pair validation must ignore unrelated packages and pip options.
pair_constraint="$workspace/pair_constraint.txt"
printf '%s\n' \
    '# pinned pair' \
    '--extra-index-url https://example.invalid/simple' \
    'torch==2.9.0' \
    'nvidia-cutlass-dsl==4.4.2' \
    'quack-kernels==0.4.1' \
    > "$pair_constraint"
dsl_only_constraint="$workspace/dsl_only_constraint.txt"
printf 'nvidia-cutlass-dsl==4.4.2\n' > "$dsl_only_constraint"
# Image constraints can leave both packages unpinned.
image_constraint="$workspace/image_constraint.txt"
printf '%s\n' \
    '--extra-index-url https://example.invalid/simple' \
    'torch==2.9.0' \
    > "$image_constraint"

# installed <nvidia-cutlass-dsl version> <quack-kernels version>
installed() {
    pair_metadata "$workspace/site" "$1" "$2"
}

# upgraded <nvidia-cutlass-dsl version> <quack-kernels version>
#   Metadata that the stub pip installs on an upgrade call.
upgraded_site="$workspace/upgraded_site"
upgraded() {
    pair_metadata "$upgraded_site" "$1" "$2"
}

# pair_metadata <site> <nvidia-cutlass-dsl version> <quack-kernels version>
pair_metadata() {
    rm -rf "$1"
    write_distribution "$1" nvidia_cutlass_dsl nvidia-cutlass-dsl "$2"
    write_distribution "$1" quack_kernels quack-kernels "$3"
}

write_distribution() {
    local info="$1/$2-$4.dist-info"
    mkdir -p "$info"
    printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' "$3" "$4" > "$info/METADATA"
}

# run_install <expected exit status> <project dir> [NAME=VALUE...]
run_install() {
    local expected=$1 project=$2 status=0
    shift 2
    : > "$pip_log"
    (
        cd "$project"
        env -u PIP_CONSTRAINT -u PIP_INDEX_URL \
            PATH="$workspace/bin:$PATH" PYTHONPATH="$workspace/site" \
            FAKE_PIP_LOG="$pip_log" "$@" bash "$script"
    ) > "$output" 2>&1 || status=$?
    if [[ "$status" -ne "$expected" ]]; then
        echo "install_requirements.sh $*: expected exit status $expected, got $status" >&2
        cat "$output" >&2
        exit 1
    fi
}

# expect_line <file> <line>
expect_line() {
    if ! grep -qxF -- "$2" "$1"; then
        echo "missing line in $1: $2" >&2
        cat "$1" >&2
        exit 1
    fi
}

# expect_text <file> <text>
expect_text() {
    if ! grep -qF -- "$2" "$1"; then
        echo "missing text in $1: $2" >&2
        cat "$1" >&2
        exit 1
    fi
}

expect_output() {
    expect_line "$output" "$1"
}

expect_no_upgrade() {
    if grep -q -- '--upgrade' "$pip_log"; then
        echo "unexpected upgrade:" >&2
        cat "$pip_log" >&2
        exit 1
    fi
}

project="$workspace/project"
index=https://pypi.tuna.tsinghua.edu.cn/simple/

# Without PIP_CONSTRAINT, only the pair is upgraded, and both upper bounds are required.
installed 4.8.0 0.6.5
run_install 0 "$project"
expect_output "Expected CuTeDSL pair: nvidia-cutlass-dsl==4.8.0, quack-kernels==0.6.5"
expect_output "Installed CuTeDSL pair: nvidia-cutlass-dsl==4.8.0, quack-kernels==0.6.5"
expect_line "$pip_log" "pip install --upgrade nvidia-cutlass-dsl[cu13]>=4.4.2,<=4.8.0,!=4.6.0,!=4.6.1 quack-kernels>=0.4.1,<=0.6.5 --index-url $index"

installed 4.7.0 0.6.5
run_install 1 "$project"
expect_output "  nvidia-cutlass-dsl: expected 4.8.0, installed 4.7.0"

# Constraints on unrelated packages must still allow the pair upgrade.
installed 4.7.0 0.6.5
upgraded 4.8.0 0.6.5
run_install 0 "$project" PIP_CONSTRAINT="$image_constraint" FAKE_PIP_UPGRADED_SITE="$upgraded_site"
expect_output "CuTeDSL pair source: upgrade"
expect_output "Expected CuTeDSL pair: nvidia-cutlass-dsl==4.8.0, quack-kernels==0.6.5"
expect_output "Installed CuTeDSL pair: nvidia-cutlass-dsl==4.8.0, quack-kernels==0.6.5"
expect_line "$pip_log" "pip install --upgrade nvidia-cutlass-dsl[cu13]>=4.4.2,<=4.8.0,!=4.6.0,!=4.6.1 quack-kernels>=0.4.1,<=0.6.5 --index-url $index"

# A pair pin selects constraint mode.
installed 4.4.2 0.4.1
run_install 0 "$project" PIP_CONSTRAINT="$pair_constraint"
expect_output "CuTeDSL pair source: constraint"
expect_output "Expected CuTeDSL pair: nvidia-cutlass-dsl==4.4.2, quack-kernels==0.4.1"
expect_no_upgrade

# A pair pin in any PIP_CONSTRAINT file selects constraint mode.
installed 4.4.2 0.4.1
upgraded 4.8.0 0.6.5
run_install 0 "$project" PIP_CONSTRAINT="$image_constraint $pair_constraint" \
    FAKE_PIP_UPGRADED_SITE="$upgraded_site"
expect_output "CuTeDSL pair source: constraint"
expect_output "Expected CuTeDSL pair: nvidia-cutlass-dsl==4.4.2, quack-kernels==0.4.1"
expect_output "Installed CuTeDSL pair: nvidia-cutlass-dsl==4.4.2, quack-kernels==0.4.1"
expect_no_upgrade

installed 4.8.0 0.6.5
run_install 1 "$project" PIP_CONSTRAINT="$pair_constraint"
expect_output "  nvidia-cutlass-dsl: expected 4.4.2, installed 4.8.0"
expect_output "  quack-kernels: expected 0.4.1, installed 0.6.5"

# Only the pinned package is checked.
installed 4.4.2 0.4.1
run_install 0 "$project" PIP_CONSTRAINT="$dsl_only_constraint"
expect_output "Expected CuTeDSL pair: nvidia-cutlass-dsl==4.4.2"

# Invalid pair specifications must fail without upgrading.
installed 4.8.0 0.6.5
run_install 1 "$workspace/project_strict_upper"
expect_text "$output" "needs exactly one '<=' or '==' bound"
expect_no_upgrade

run_install 1 "$project" PIP_CONSTRAINT="$workspace/missing_constraint.txt"
expect_text "$output" "Cannot read $workspace/missing_constraint.txt"

run_install 1 "$project" FAKE_PIP_STATUS=1
expect_no_upgrade

echo "install_requirements.sh tests passed"
