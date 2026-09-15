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

package_dir=${1:?usage: build_v2_wheel.sh <package-dir> <cache-name> <validation-node>}
cache_name=${2:?usage: build_v2_wheel.sh <package-dir> <cache-name> <validation-node>}
node=${3:?usage: build_v2_wheel.sh <package-dir> <cache-name> <validation-node>}
repo_root=$(git rev-parse --show-toplevel)
cache_root="${CI_WORKSPACE_ROOT:-/workspace}/v2/standalone-artifacts/magi-attention"
base_tag=$(tr -d '[:space:]' < "$repo_root/.github/workflows/base_image_tag.txt")
[[ "$base_tag" =~ ^([0-9]+\.[0-9]+)\.[0-9]+$ ]] || {
    echo "Invalid base image tag: $base_tag" >&2
    exit 2
}
base_family=${BASH_REMATCH[1]}
source_digest=$(bash "$repo_root/.github/scripts/portable_validation.sh" source-digest "$node") || exit
recipe_digest=$(sha256sum "$repo_root/.github/scripts/build_v2_wheel.sh" | awk '{print $1}')
package_version=$(cd "$repo_root/$package_dir" && python -m versioningit) || exit
dependency=none
if [[ "$node" == magi_attn_extensions ]]; then
    dependency=$(bash "$repo_root/.github/scripts/portable_validation.sh" fingerprint magi_attention) || exit
fi
input_digest=$(printf 'schema=1\ncache=%s\nbase_family=%s\nsource=%s\ndependency=%s\nrecipe=%s\nversion=%s\n' \
    "$cache_name" "$base_family" "$source_digest" "$dependency" "$recipe_digest" "$package_version" |
    sha256sum | awk '{print $1}')
fingerprint="v1-$base_family-$input_digest"
target="$cache_root/$cache_name/$fingerprint"

verify() {
    local directory=${1:?directory is required}
    [[ -d "$directory" && ! -L "$directory" ]] || return 1
    [[ -f "$directory/manifest.json" && ! -L "$directory/manifest.json" ]] || return 1
    shopt -s nullglob
    local wheels=("$directory"/*.whl)
    [[ ${#wheels[@]} -gt 0 ]] || return 1
    local wheel
    for wheel in "${wheels[@]}"; do
        [[ -f "$wheel" && ! -L "$wheel" ]] || return 1
    done
    python - "$directory/manifest.json" "$cache_name" "$base_family" \
        "$source_digest" "$dependency" "$recipe_digest" "$package_version" "$fingerprint" <<'PY'
import json
import sys
from pathlib import Path

path, cache, family, source, dependency, recipe, version, fingerprint = sys.argv[1:]
data = json.loads(Path(path).read_text())
expected = {
    "schema_version": 1,
    "cache_name": cache,
    "base_image_family": family,
    "source_digest": source,
    "dependency_fingerprint": dependency,
    "recipe_digest": recipe,
    "package_version": version,
    "fingerprint": fingerprint,
    "status": "built",
}
errors = [f"{key}: expected {value!r}, got {data.get(key)!r}" for key, value in expected.items() if data.get(key) != value]
if errors:
    raise SystemExit("Incompatible standalone wheel manifest:\n  " + "\n  ".join(errors))
PY
}

if verify "$target" 2>/dev/null; then
    pip install --no-deps --force-reinstall "$target"/*.whl
    echo "Reused standalone wheel artifact: $target"
    exit
fi

parent=$(dirname "$target")
mkdir -p "$parent"
staging=$(mktemp -d "$parent/.build.XXXXXX")
trap 'rm -rf "${staging:-}"' EXIT
python -m build --wheel --no-isolation --outdir "$staging" "$repo_root/$package_dir"
python - "$staging/manifest.json" "$cache_name" "$base_tag" "$base_family" \
    "$source_digest" "$dependency" "$recipe_digest" "$package_version" "$fingerprint" <<'PY'
import json
import os
import sys
from pathlib import Path

path, cache, tag, family, source, dependency, recipe, version, fingerprint = sys.argv[1:]
data = {
    "base_image_family": family,
    "base_image_tag": tag,
    "cache_name": cache,
    "dependency_fingerprint": dependency,
    "fingerprint": fingerprint,
    "package_version": version,
    "producer_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    "producer_sha": os.environ.get("GITHUB_SHA", "local"),
    "recipe_digest": recipe,
    "schema_version": 1,
    "source_digest": source,
    "status": "built",
}
Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
verify "$staging"
if ! mv -T "$staging" "$target" 2>/dev/null; then
    verify "$target"
fi
staging=
pip install --no-deps --force-reinstall "$target"/*.whl
echo "Published standalone wheel artifact: $target"
