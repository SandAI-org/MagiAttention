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
# MagiAttention Local Install Script (Skip All Builds)
# =============================================================================
# This script installs MagiAttention locally without building any CUDA
# extensions or pre-building FFA kernels. Useful for development, testing,
# or environments without GPU/CUDA toolchain.
#
# Usage:
#   bash scripts/install_skip_all.sh
# =============================================================================

set -ex

LOG_PREFIX="[MagiAttention-SkipBuild]"

log_step() {
    echo ""
    echo "================================================================="
    echo "${LOG_PREFIX} $1"
    echo "================================================================="
}

# --- Step 1. Install essential build-time Python dependencies
log_step "Step 1/4: Installing build-time Python dependencies..."

pip3 install packaging ninja versioningit debugpy einops tqdm

# --- Step 2. Initialize and update git submodules
log_step "Step 2/4: Initializing git submodules..."

# git submodule update --init --recursive

# --- Step 3. Install Python dependencies
log_step "Step 3/4: Installing Python dependencies from requirements.txt..."

pip install -r requirements.txt

# --- Step 4. Install MagiAttention in editable mode (skip all CUDA builds)
log_step "Step 4/4: Installing MagiAttention (editable, no CUDA build)..."

export MAGI_ATTENTION_SKIP_CUDA_BUILD=1
export MAGI_ATTENTION_PREBUILD_FFA=0
export MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD=1
export MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=1

echo "${LOG_PREFIX} MAGI_ATTENTION_SKIP_CUDA_BUILD=$MAGI_ATTENTION_SKIP_CUDA_BUILD"
echo "${LOG_PREFIX} MAGI_ATTENTION_PREBUILD_FFA=$MAGI_ATTENTION_PREBUILD_FFA"
echo "${LOG_PREFIX} MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD=$MAGI_ATTENTION_SKIP_MAGI_ATTN_EXT_BUILD"
echo "${LOG_PREFIX} MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD=$MAGI_ATTENTION_SKIP_MAGI_ATTN_COMM_BUILD"

pip install -e . -v --no-build-isolation

echo ""
echo "${LOG_PREFIX} Install complete (all CUDA builds skipped)."
