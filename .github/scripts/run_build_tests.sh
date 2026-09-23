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

prepare_runtime_lock() {
    export CI_DEPENDENCY_RUNTIME_LOCK="${RUNNER_TEMP:-/tmp}/ci-source-dependencies/resolved.json"
}

install_dependencies() {
    bash -x .github/scripts/install_requirements.sh
    prepare_runtime_lock
    python .github/scripts/install_source_dependencies.py
    pip install -r extensions/requirements.txt \
        --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
}

build() {
    export MAGI_ATTENTION_PREBUILD_LEVEL=ci
    rm -rf "${HOME:?HOME is required}/.cache/magi_attention/"
    bash .github/scripts/build_v2_wheel.sh . MagiAttention magi_attention build
    bash .github/scripts/build_v2_wheel.sh \
        extensions MagiAttnExtensions magi_attn_extensions build

    probe_wheel_imports
}

install_wheels() {
    export MAGI_ATTENTION_PREBUILD_LEVEL=ci
    bash .github/scripts/build_v2_wheel.sh . MagiAttention magi_attention install
    bash .github/scripts/build_v2_wheel.sh \
        extensions MagiAttnExtensions magi_attn_extensions install

    probe_wheel_imports
}

probe_wheel_imports() {
    local import_probe_dir
    local status=0
    import_probe_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/magi-attention-wheel-import.XXXXXX")
    (
        cd "$import_probe_dir"
        python -c "import magi_attention; print('MagiAttention wheel import succeeded')"
        python -c "import magi_attn_extensions; print('MagiAttnExtensions wheel import succeeded')"
    ) || status=$?
    rm -rf -- "$import_probe_dir"
    return "$status"
}

test_packages() {
    local main_changed=${1:?main change flag is required}
    local ci_changed=${2:?CI protocol change flag is required}
    local trusted=${3:?trust flag is required}
    local coverage_generated=false

    prepare_runtime_lock
    bash .github/tests/test_portable_validation.sh
    python .github/tests/test_source_dependency_cache.py
    python .github/tests/test_compiled_artifact_cache.py
    if [[ "$main_changed" == true || "$ci_changed" == true ]]; then
        if [[ "$trusted" == true ]] && \
            bash .github/scripts/portable_validation.sh verify magi_attention; then
            echo "Reused portable MagiAttention validation"
        else
            if [[ "${TASK_CI_PLATFORM:?TASK_CI_PLATFORM is required}" == h100 ]]; then
                COVERAGE_RUN=True \
                    PORTABLE_VALIDATION_COVERAGE=true \
                    bash .github/scripts/portable_validation.sh run-test magi_attention
                coverage_generated=true
            else
                bash .github/scripts/portable_validation.sh run-test magi_attention
            fi
            if [[ "$trusted" == true ]]; then
                bash .github/scripts/portable_validation.sh write-success magi_attention
            fi
        fi
    fi

    if [[ "$trusted" == true ]] && \
        bash .github/scripts/portable_validation.sh verify magi_attn_extensions; then
        echo "Reused portable MagiAttnExtensions validation"
    else
        bash .github/scripts/portable_validation.sh run-test magi_attn_extensions
        if [[ "$trusted" == true ]]; then
            bash .github/scripts/portable_validation.sh write-success magi_attn_extensions
        fi
    fi

    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        echo "coverage_generated=$coverage_generated" >> "$GITHUB_OUTPUT"
    fi
}

command=${1:-}
shift || true
case "$command" in
    install) install_dependencies "$@" ;;
    build) build "$@" ;;
    install-wheel) install_wheels "$@" ;;
    test) test_packages "$@" ;;
    *) echo "Usage: $0 {install|build|install-wheel|test [main_changed ci_changed trusted]}" >&2; exit 2 ;;
esac
