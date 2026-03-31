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

# =============================================================================
# MagiAttention SCM Build Script
# =============================================================================
# This script builds and packages MagiAttention as a wheel for SCM deployment.
# It handles dependency installation, submodule setup, CUDA configuration,
# and final wheel generation.
#
# Customizable environment variables:
#   CUSTOM_MAX_JOBS                                  - Max parallel build jobs (default: 8)
#   CUSTOM_NVCC_THREADS                         - Max NVCC threads (default: 8)
#   CUSTOM_MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY   - Target GPU arch (default: "90,100")
#
# Usage:
#   git clone https://github.com/SandAI-org/MagiAttention && cd MagiAttention
#   bash scripts/install_on_scm.sh
# =============================================================================

set -ex

LOG_PREFIX="[MagiAttention-SCM]"

log_step() {
    echo ""
    echo "================================================================="
    echo "${LOG_PREFIX} $1"
    echo "================================================================="
}

# --- Step 1. Install essential build-time Python dependencies
log_step "Step 1/9: Installing build-time Python dependencies..."

pip3 install packaging ninja versioningit debugpy einops tqdm \
    -i http://bytedpypi.byted.org/simple --trusted-host bytedpypi.byted.org

# --- Step 2. Configure CUDA and build environment
log_step "Step 2/9: Configuring CUDA and build environment..."

export LDFLAGS=-L/usr/local/cuda/lib64/stubs
export MAX_JOBS=${CUSTOM_MAX_JOBS:-8}
export NVCC_THREADS=${CUSTOM_NVCC_THREADS:-8}
export PATH=$PATH:/usr/local/cuda/bin

echo "${LOG_PREFIX} PATH=$PATH"
echo "${LOG_PREFIX} MAX_JOBS=$MAX_JOBS"
echo "${LOG_PREFIX} NVCC_THREADS=$NVCC_THREADS"
nvcc -V

echo "${LOG_PREFIX} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"
ls "${CUDA_HOME:-/usr/local/cuda}/include/cuda/std/utility" 2>/dev/null \
    && echo "${LOG_PREFIX} cuda/std/utility found (legacy path)" \
    || echo "${LOG_PREFIX} cuda/std/utility NOT found in legacy path"
ls -la "${CUDA_HOME:-/usr/local/cuda}/include/cccl/" 2>/dev/null \
    && echo "${LOG_PREFIX} cccl directory exists" \
    || echo "${LOG_PREFIX} cccl directory NOT found"
find "${CUDA_HOME:-/usr/local/cuda}" -name "utility" -path "*/cuda/std/*" 2>/dev/null \
    | head -5 || echo "${LOG_PREFIX} cuda/std/utility not found anywhere under CUDA_HOME"

# --- Step 3. Determine target GPU architectures
log_step "Step 3/9: Determining target GPU architectures..."

export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=${CUSTOM_MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY:-"90,100"}
ARCH_ARG=$(echo "$MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY" | sed 's/\([0-9]\+\)/sm\1/g')

echo "${LOG_PREFIX} MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=$MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY"
echo "${LOG_PREFIX} ARCH_ARG=$ARCH_ARG (for flash_attn_cute)"

# --- Step 4. Initialize and update git submodules
log_step "Step 4/9: Initializing git submodules..."

git submodule update --init --recursive

# --- Step 5. Install flash_attn_cute (FA4 backend kernels)
log_step "Step 5/9: Installing flash_attn_cute for architectures: ${ARCH_ARG}..."

export MAGI_WHEEL_DIR="$(pwd)/output"
mkdir -p "$MAGI_WHEEL_DIR"

bash scripts/install_flash_attn_cute.sh "$ARCH_ARG"

echo "${LOG_PREFIX} Collected sub-package wheels:"
ls -lh "$MAGI_WHEEL_DIR"/*.whl 2>/dev/null || echo "${LOG_PREFIX} (none yet)"

# --- Step 6. Install remaining Python dependencies
log_step "Step 6/9: Installing Python dependencies from requirements.txt..."

pip install -r requirements.txt

# --- Step 7. Build and install MagiAttention in editable mode
log_step "Step 7/9: Building MagiAttention (editable install)..."

export MAGI_ATTENTION_PREBUILD_FFA=1
export MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1
export MAGI_ATTENTION_PREBUILD_FFA_JOBS=${CUSTOM_MAX_JOBS:-256}
export MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=1

echo "${LOG_PREFIX} MAGI_ATTENTION_PREBUILD_FFA=$MAGI_ATTENTION_PREBUILD_FFA"
echo "${LOG_PREFIX} MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=$MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY"
echo "${LOG_PREFIX} MAGI_ATTENTION_PREBUILD_FFA_JOBS=$MAGI_ATTENTION_PREBUILD_FFA_JOBS"
echo "${LOG_PREFIX} MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12=$MAGI_ATTENTION_ALLOW_BUILD_WITH_CUDA12"

pip install -e . -v --no-build-isolation

# --- Step 8. Enable FA4 backend for Blackwell FFA_FA4 kernels
log_step "Step 8/9: Enabling FA4 backend..."

export MAGI_ATTENTION_FA4_BACKEND=1
echo "${LOG_PREFIX} MAGI_ATTENTION_FA4_BACKEND=$MAGI_ATTENTION_FA4_BACKEND"

# --- Step 9. Package the wheel for SCM distribution
log_step "Step 9/9: Building wheel and copying to output/..."

mkdir -p output
cp -f scm_setup.py output/setup.py

python3 setup.py bdist_wheel
cp -f dist/*.whl output/

echo ""
echo "${LOG_PREFIX} Build complete. Wheel(s) available in output/:"
ls -lh output/*.whl
