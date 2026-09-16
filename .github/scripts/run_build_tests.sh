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

main_changed=${1:?main change flag is required}
ci_changed=${2:?CI protocol change flag is required}
trusted=${3:?trust flag is required}

bash .github/scripts/verify_task_runner.sh
bash .github/scripts/test_portable_validation.sh
python .github/scripts/test_source_dependency_cache.py
bash -x .github/scripts/install_requirements.sh
export CI_DEPENDENCY_RUNTIME_LOCK="${RUNNER_TEMP:-/tmp}/ci-source-dependencies/resolved.json"
python .github/scripts/install_source_dependencies.py
rm -rf "${HOME:?HOME is required}/.cache/magi_attention/"
bash .github/scripts/build_v2_wheel.sh . MagiAttention magi_attention
import_probe_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/magi-attention-wheel-import.XXXXXX")
trap 'rm -rf "$import_probe_dir"' EXIT
(
    cd "$import_probe_dir"
    python -c "import magi_attention; print('MagiAttention wheel import succeeded')"
)

coverage_generated=false
if [[ "$main_changed" == true || "$ci_changed" == true ]]; then
    if [[ "$trusted" == true ]] && \
        bash .github/scripts/portable_validation.sh verify magi_attention; then
        echo "Reused portable MagiAttention validation"
    else
        COVERAGE_RUN=True \
            PORTABLE_VALIDATION_COVERAGE=true \
            MAGI_ATTENTION_JIT_COMPILE_DISABLED=1 \
            bash .github/scripts/portable_validation.sh run-test magi_attention
        coverage_generated=true
        if [[ "$trusted" == true ]]; then
            bash .github/scripts/portable_validation.sh write-success magi_attention
        fi
    fi
fi

pip install -r extensions/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
bash .github/scripts/build_v2_wheel.sh \
    extensions MagiAttnExtensions magi_attn_extensions
(
    cd "$import_probe_dir"
    python -c "import magi_attn_extensions; print('MagiAttnExtensions wheel import succeeded')"
)
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
