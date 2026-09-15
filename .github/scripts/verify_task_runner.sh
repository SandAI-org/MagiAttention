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

expected_root=/home/niubility2/ci_workspace
mounted_target=$(findmnt -T "$expected_root" -n -o TARGET 2>/dev/null || true)
if [[ "${VALIDATION_CACHE_TRUSTED:-false}" == true ]]; then
    if [[ "${CI_WORKSPACE_ROOT:-}" != "$expected_root" ]]; then
        echo "::error::Unexpected trusted CI_WORKSPACE_ROOT: ${CI_WORKSPACE_ROOT:-<unset>}" >&2
        exit 1
    fi
    if [[ "$mounted_target" != "$expected_root" ]]; then
        echo "::error::Shared CI storage is not mounted at $expected_root" >&2
        exit 1
    fi
else
    if [[ "$mounted_target" == "$expected_root" ]]; then
        echo "::error::Shared CI storage must not be visible to an untrusted run" >&2
        exit 1
    fi
    expected_root=${RUNNER_TEMP:?RUNNER_TEMP is required}/magi-attention-ci
    if [[ "${CI_WORKSPACE_ROOT:-}" != "$expected_root" ]]; then
        echo "::error::Untrusted CI must use runner-local storage: ${CI_WORKSPACE_ROOT:-<unset>}" >&2
        exit 1
    fi
fi
mkdir -p "$CI_WORKSPACE_ROOT"
probe=$(mktemp "$CI_WORKSPACE_ROOT/.magi-attention-write-probe.XXXXXX")
trap 'rm -f "$probe"' EXIT

printf 'runner=%s platform=%s job=%s sha=%s utc=%s\n' \
    "${RUNNER_NAME:-unknown}" "${TASK_CI_PLATFORM:-unknown}" \
    "${GITHUB_JOB:-unknown}" "${GITHUB_SHA:-unknown}" "$(date -u +%FT%TZ)"
printf 'shared_cache=%s\n' "$CI_WORKSPACE_ROOT"
declared_base_tag=$(tr -d '[:space:]' < .github/workflows/base_image_tag.txt)
printf 'workflow_declared_base_tag=%s\n' "$declared_base_tag"
expected_image="registry.cn-sh-01.sensecore.cn/sandai-ccr/magi-base:$declared_base_tag"
if [[ "${TASK_CI_BASE_IMAGE:-}" != "$expected_image" ]]; then
    echo "::error::Expected task runner image $expected_image, got ${TASK_CI_BASE_IMAGE:-<unset>}" >&2
    exit 1
fi
printf 'workflow_declared_base_image=%s\n' "$TASK_CI_BASE_IMAGE"
if [[ -n "${MAGI_BASE_IMAGE_TAG:-}" && "$MAGI_BASE_IMAGE_TAG" != "$declared_base_tag" ]]; then
    echo "::error::Task runner base tag $MAGI_BASE_IMAGE_TAG does not match declared $declared_base_tag" >&2
    exit 1
fi
if [[ "${VALIDATION_CACHE_TRUSTED:-false}" == true ]]; then
    findmnt -T "$expected_root" -n -o TARGET,SOURCE,FSTYPE
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import sys
import torch

print(
    f"Python={sys.version.split()[0]} Torch={torch.__version__} "
    f"CUDA={torch.version.cuda} GPUs={torch.cuda.device_count()}"
)
assert torch.cuda.is_available(), "Task runner has no usable CUDA device"
PY
