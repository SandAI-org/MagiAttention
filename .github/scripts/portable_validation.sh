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

PORTABLE_SCHEMA=1
PORTABLE_MAIN_RECIPE_VERSION=3
PORTABLE_EXTENSIONS_RECIPE_VERSION=4
PORTABLE_ROOT="${CI_WORKSPACE_ROOT:-/workspace}/v2/portable-validations/magi-attention"
PORTABLE_PRODUCER=SandAI-org/MagiAttention

repo_root=$(git rev-parse --show-toplevel)
source_root=${PORTABLE_SOURCE_ROOT:-.}
base_tag_file=${PORTABLE_BASE_TAG_FILE:-.github/configs/base_image_tag.txt}
[[ "$base_tag_file" == /* ]] || base_tag_file=$repo_root/$base_tag_file

check_node() {
    case "${1:-}" in
        magi_attention|magi_attn_extensions) ;;
        *) echo "Unknown portable validation node: ${1:-<empty>}" >&2; return 2 ;;
    esac
}

base_tag() {
    local tag
    tag=$(tr -d '[:space:]' < "$base_tag_file")
    [[ "$tag" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
        echo "Invalid base image tag for portable validation: $tag" >&2
        return 2
    }
    echo "$tag"
}

source_digest() {
    local node=${1:?node is required}
    local policy helper
    check_node "$node"
    git -C "$repo_root" diff --quiet -- "$source_root" &&
        git -C "$repo_root" diff --cached --quiet -- "$source_root" || {
        echo "Refusing to certify a dirty MagiAttention worktree" >&2
        return 3
    }
    policy="$repo_root/$source_root/.github/configs/ci_input_policy.json"
    helper="$repo_root/$source_root/.github/scripts/ci_input_policy.py"
    python "$helper" --repo-root "$repo_root" --policy "$policy" \
        digest --layer portable --node "$node"
}

recipe_digest() {
    local node=${1:?node is required}
    local recipe
    case "$node" in
        magi_attention) recipe=$PORTABLE_MAIN_RECIPE_VERSION ;;
        magi_attn_extensions) recipe=$PORTABLE_EXTENSIONS_RECIPE_VERSION ;;
        *) check_node "$node"; return ;;
    esac
    python - "$repo_root/$source_root/.github/configs/ci_input_policy.json" "$node" "$recipe" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path, node, recipe = sys.argv[1:]
policy = json.loads(Path(path).read_text())
projection = {
    "exclusions": sorted(policy["nodes"][node]["exclusions"]["portable"]),
    "layer": "portable",
    "node": node,
    "protocol": "magi-attention-portable-v1",
    "recipe_version": int(recipe),
    "root": policy["nodes"][node]["root"],
    "version": policy["version"],
}
payload = json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

inputs_json() {
    local node=${1:?node is required}
    local tag source dependency=none recipe
    check_node "$node"
    if [[ "$node" == magi_attn_extensions ]]; then
        dependency=$(fingerprint magi_attention) || return
    fi
    tag=$(base_tag) || return
    source=$(source_digest "$node") || return
    recipe=$(recipe_digest "$node") || return
    python - "$PORTABLE_SCHEMA" "$node" "$tag" \
        "${TASK_CI_PLATFORM:-h100}" "$source" "$dependency" "$recipe" <<'PY'
import json
import sys

schema, node, base_tag, platform, source, dependency, recipe = sys.argv[1:]
print(json.dumps({
    "base_image_tag": base_tag,
    "dependency_fingerprint": dependency,
    "node": node,
    "platform": platform,
    "recipe_digest": recipe,
    "schema_version": int(schema),
    "source_digest": source,
}, sort_keys=True, separators=(",", ":")))
PY
}

fingerprint() {
    local inputs
    inputs=$(inputs_json "${1:?node is required}") || return
    printf 'v%s-%s\n' "$PORTABLE_SCHEMA" \
        "$(printf '%s' "$inputs" | sha256sum | awk '{print $1}')"
}

marker_dir() {
    local node=${1:?node is required}
    local fingerprint
    fingerprint=$(fingerprint "$node") || return
    printf '%s/v%s/%s/%s\n' "$PORTABLE_ROOT" "$PORTABLE_SCHEMA" \
        "$node" "$fingerprint"
}

verify() {
    local node=${1:?node is required}
    local inputs expected_fingerprint directory marker
    inputs=$(inputs_json "$node") || return
    expected_fingerprint=$(fingerprint "$node") || return
    directory=$(marker_dir "$node") || return
    marker="$directory/success.json"
    [[ -e "$directory" ]] || return 1
    if [[ -L "$directory" || ! -d "$directory" || -L "$marker" || ! -f "$marker" ]]; then
        echo "Invalid portable validation marker path: $marker" >&2
        return 3
    fi
    if ! python - "$marker" "$expected_fingerprint" "$inputs" "$PORTABLE_PRODUCER" <<'PY'
import json
import sys
from pathlib import Path

path, fingerprint, inputs, producer_repository = sys.argv[1:]
try:
    data = json.loads(Path(path).read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Invalid portable validation marker {path}: {exc}")
expected = {
    "fingerprint": fingerprint,
    "inputs": json.loads(inputs),
    "producer_repository": producer_repository,
    "status": "success",
}
errors = [
    f"{key}: expected {value!r}, got {data.get(key)!r}"
    for key, value in expected.items()
    if data.get(key) != value
]
if errors:
    raise SystemExit("Incompatible portable validation marker:\n  " + "\n  ".join(errors))
PY
    then
        return 3
    fi
}

write_success() {
    local node=${1:?node is required}
    local inputs expected_fingerprint target parent staging
    [[ "${VALIDATION_CACHE_TRUSTED:-false}" == true ]] || {
        echo "Refusing to write a portable marker from an untrusted run" >&2
        return 3
    }
    [[ "${GITHUB_REPOSITORY:-}" == "$PORTABLE_PRODUCER" ]] || {
        echo "Only $PORTABLE_PRODUCER may publish portable markers" >&2
        return 3
    }
    inputs=$(inputs_json "$node") || return
    expected_fingerprint=$(fingerprint "$node") || return
    target=$(marker_dir "$node") || return
    parent=$(dirname "$target")
    mkdir -p "$parent"
    if [[ -e "$target" ]]; then
        verify "$node"
        echo "Reusing portable validation marker: $target"
        return
    fi
    staging=$(mktemp -d "$parent/.publish.XXXXXX")
    trap 'rm -rf "${staging:-}"' RETURN
    python - "$staging/success.json" "$expected_fingerprint" "$inputs" "$PORTABLE_PRODUCER" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, fingerprint, inputs, repository = sys.argv[1:]
data = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "fingerprint": fingerprint,
    "inputs": json.loads(inputs),
    "producer": {
        "event_name": os.environ.get("GITHUB_EVENT_NAME", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "sha": os.environ.get("GITHUB_SHA", "local"),
    },
    "producer_repository": repository,
    "status": "success",
}
with Path(path).open("w") as stream:
    json.dump(data, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
    if ! mv -T "$staging" "$target" 2>/dev/null; then
        verify "$node"
    fi
    staging=
    trap - RETURN
    verify "$node"
    echo "Recorded portable validation success: $node $expected_fingerprint"
}

run_test() {
    local test_cwd package_root clean_pythonpath
    package_root=$(cd "$repo_root/$source_root" && pwd)
    test_cwd=$(mktemp -d "${RUNNER_TEMP:-/tmp}/magi-attention-wheel-tests.XXXXXX")
    trap 'rm -rf "$test_cwd"' RETURN
    clean_pythonpath=$(python - "$repo_root" "$package_root" <<'PY'
import os
import sys
from pathlib import Path

excluded = {Path(path).resolve() for path in sys.argv[1:]}
print(os.pathsep.join(
    entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if entry and Path(entry).resolve() not in excluded
))
PY
    )
    case "${1:?node is required}" in
        magi_attention)
            if [[ "${PORTABLE_VALIDATION_COVERAGE:-false}" == true ]]; then
                (cd "$test_cwd" && \
                    PYTHONPATH="$clean_pythonpath" \
                    COVERAGE_FILE="$repo_root/.coverage" \
                    MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH=0 \
                    MAGI_ATTENTION_TEST_BACKEND="sdpa,ffa" \
                    coverage run --source magi_attention -m pytest \
                        -q -s --skip-slow --import-mode=append "$package_root/tests")
                (cd "$repo_root" && coverage xml -i)
            else
                (cd "$test_cwd" && \
                    PYTHONPATH="$clean_pythonpath" \
                    MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH=0 \
                    MAGI_ATTENTION_TEST_BACKEND="sdpa,ffa" \
                    python -m pytest -q -s --skip-slow --import-mode=append \
                        "$package_root/tests")
            fi
            ;;
        magi_attn_extensions)
            (cd "$test_cwd" && \
                PYTHONPATH="$clean_pythonpath" \
                MAGI_ATTENTION_TEST_PRINT_NO_MISMATCH=0 \
                python -m pytest -q -s --skip-slow --import-mode=append \
                    "$package_root/extensions/tests")
            ;;
        *) check_node "$1" ;;
    esac
    rm -rf "$test_cwd"
    trap - RETURN
}

case "${1:-}" in
    source-digest) source_digest "${2:?node is required}" ;;
    fingerprint) fingerprint "${2:?node is required}" ;;
    marker-dir) marker_dir "${2:?node is required}" ;;
    verify) verify "${2:?node is required}" ;;
    write-success) write_success "${2:?node is required}" ;;
    run-test) run_test "${2:?node is required}" ;;
    *) echo "Usage: $0 {source-digest|fingerprint|marker-dir|verify|write-success|run-test} <node>" >&2; exit 2 ;;
esac
