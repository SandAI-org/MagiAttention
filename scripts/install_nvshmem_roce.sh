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

# Stop the script immediately if any command fails
set -e

# --- Configuration Variables ---

# We will install everything directly into /opt/nvshmem to keep paths clean
NVSHMEM_DIR="/opt/nvshmem"
NVSHMEM_VERSION="3.4.5-0"
# Temporary directory for source code
SOURCE_BASE="/tmp/nvshmem_build"
NVSHMEM_SRC_DIR="${SOURCE_BASE}/nvshmem-${NVSHMEM_VERSION}"
INSTALL_DIR="${NVSHMEM_DIR}"
CUDA_ARCHITECTURES="80-real;89-real;90-real;100-real;120"

# --- Handle Local File Argument ---

LOCAL_TARBALL=$1

echo "=== Starting NVSHMEM v${NVSHMEM_VERSION} Custom Build ==="

# 1. Setup Directories
mkdir -p "${NVSHMEM_DIR}"
mkdir -p "${SOURCE_BASE}"

# 2. Source Code Acquisition
if [ -n "$LOCAL_TARBALL" ]; then
    if [ -f "$LOCAL_TARBALL" ]; then
        echo "--- Using local tarball: $LOCAL_TARBALL ---"
        ABS_TARBALL=$(realpath "$LOCAL_TARBALL")
        cp "$ABS_TARBALL" "${SOURCE_BASE}/v${NVSHMEM_VERSION}.tar.gz"
    else
        echo "Error: Local file $LOCAL_TARBALL not found!"
        exit 1
    fi
else
    echo "--- No local file provided. Attempting to download... ---"
    cd "${SOURCE_BASE}"
    wget -O "v${NVSHMEM_VERSION}.tar.gz" "https://github.com/NVIDIA/nvshmem/archive/refs/tags/v${NVSHMEM_VERSION}.tar.gz"
fi

# 3. Extraction
echo "--- Extracting Source ---"

cd "${SOURCE_BASE}"
tar -zxf "v${NVSHMEM_VERSION}.tar.gz"
cd "${NVSHMEM_SRC_DIR}"

# 4. Environment Fixes (Symlink for libmlx5)
echo "--- Applying System Library Fixes ---"

MLX5_SO_PATH="/usr/lib/x86_64-linux-gnu/libmlx5.so"
MLX5_SO_REAL="/usr/lib/x86_64-linux-gnu/libmlx5.so.1"

if [ ! -f "$MLX5_SO_PATH" ]; then
    echo "Creating symlink for libmlx5.so..."
    if command -v sudo >/dev/null 2>&1; then
        sudo ln -s "$MLX5_SO_REAL" "$MLX5_SO_PATH" || true
    else
        ln -s "$MLX5_SO_REAL" "$MLX5_SO_PATH" || true
    fi
fi

# 5. Patching CMake and Source Code
echo "--- Applying Source Code Patches ---"

find . -name "CMakeLists.txt" -print0 | xargs -0 sed -i 's/\$<BUILD_INTERFACE:\${MPI_CXX_INCLUDE_DIRS}>/\${MPI_CXX_INCLUDE_DIRS}/g'
sed -i "s/nvshmemi_call_rdxn_on_stream_kernel/nvshmemi_reduce_on_stream/g" src/host/team/team_internal.cpp

# 6. Configuration via CMake
echo "--- Configuring NVSHMEM with IBGDA Support ---"

NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=1 \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_PMIX_SUPPORT=0 \
NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
NVSHMEM_USE_GDRCOPY=1 \
MPI_HOME=/opt/hpcx/ompi \
cmake -S . -B build/ \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    -DMLX5_lib="$MLX5_SO_PATH"

# 7. Build and Install
echo "--- Compiling and Installing ---"

cd build
make -j 16
make install

# 8. Inject Environment Variables into .bashrc
echo "--- Updating ~/.bashrc ---"

# Define the lines to be injected
# We use a marker to easily identify and avoid duplicate injections
BASHRC_FILE="$HOME/.bashrc"
MARKER="# >>> NVSHMEM CUSTOM BUILD SETTINGS <<<"

if ! grep -q "$MARKER" "$BASHRC_FILE"; then
    echo "Injecting NVSHMEM variables into $BASHRC_FILE..."
    cat >> "$BASHRC_FILE" << EOF

$MARKER
export NVSHMEM_DIR=${INSTALL_DIR}
export NVSHMEM_HOME=${INSTALL_DIR}
export LD_LIBRARY_PATH="\${NVSHMEM_DIR}/lib:\$LD_LIBRARY_PATH"
export PATH="\${NVSHMEM_DIR}/bin:\$PATH"
# >>> END OF NVSHMEM SETTINGS <<<
EOF
else
    echo "NVSHMEM settings already exist in $BASHRC_FILE. Skipping injection."
fi

echo "=== NVSHMEM Successfully Installed to: ${INSTALL_DIR} ==="
echo "IMPORTANT: To apply changes to your current shell, run:"
echo "    source ~/.bashrc"

# Optional: Clean up build files
# rm -rf "${SOURCE_BASE}"
