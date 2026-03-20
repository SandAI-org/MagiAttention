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

# --- Step 3. Initialize and update git submodules
log_step "Step 3/9: Initializing git submodules..."

git submodule update --init --recursive

# --- Step 4. Install flash_attn_cute (core dependency for FA4 backend on Blackwell)
log_step "Step 4/9: Installing flash_attn_cute..."

bash scripts/install_flash_attn_cute.sh

# --- Step 5. Install remaining Python dependencies
log_step "Step 5/9: Installing Python dependencies from requirements.txt..."

pip install -r requirements.txt

# --- Step 6. Build and install MagiAttention in editable mode
# Pre-build FFA kernels targeting Hopper (sm90) and Blackwell (sm100).
log_step "Step 6/9: Building MagiAttention (editable install)..."

export MAGI_ATTENTION_PREBUILD_FFA=1
export MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=${CUSTOM_MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY:-"90,100"}
export MAGI_ATTENTION_PREBUILD_FFA_JOBS=${CUSTOM_MAX_JOBS:-256}

echo "${LOG_PREFIX} MAGI_ATTENTION_PREBUILD_FFA=$MAGI_ATTENTION_PREBUILD_FFA"
echo "${LOG_PREFIX} MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY=$MAGI_ATTENTION_BUILD_COMPUTE_CAPABILITY"
echo "${LOG_PREFIX} MAGI_ATTENTION_PREBUILD_FFA_JOBS=$MAGI_ATTENTION_PREBUILD_FFA_JOBS"

pip install -e . -v --no-build-isolation

# --- Step 7. Enable FA4 backend for Blackwell FFA_FA4 kernels
log_step "Step 7/9: Enabling FA4 backend..."

export MAGI_ATTENTION_FA4_BACKEND=1
echo "${LOG_PREFIX} MAGI_ATTENTION_FA4_BACKEND=$MAGI_ATTENTION_FA4_BACKEND"

# --- Step 8. Package the wheel for SCM distribution
log_step "Step 9/9: Building wheel and copying to output/..."

mkdir -p output
cp -f scm_setup.py output/setup.py

python3 setup.py bdist_wheel
cp -f dist/*.whl output/

echo ""
echo "${LOG_PREFIX} Build complete. Wheel(s) available in output/:"
ls -lh output/*.whl
