#!/bin/bash

# Copyright (c) 2025 SandAI. All Rights Reserved.
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

# ==============================================================================
# 0. Environment & SCO Setup
# ==============================================================================

# Check for required authentication environment variables
if [ -z "$ACCESS_KEY_ID" ] || [ -z "$ACCESS_KEY_SECRET" ]; then
    echo "❌ Error: Please set ACCESS_KEY_ID and ACCESS_KEY_SECRET environment variables."
    exit 1
fi

# Handle default username (use current system user if SCO_USERNAME is not set)
CURRENT_USER=${SCO_USERNAME:-$(whoami)}

# 1. Check and install sco if not present
if ! command -v sco &> /dev/null; then
    echo "⬇️  'sco' command not found. Starting installation..."
    curl -sSfL https://sco.sensecore.cn/registry/sco/install.sh | sh
    
    # Temporarily add sco to PATH for this script session
    export PATH=~/.sco/bin:$PATH
    
    if ! command -v sco &> /dev/null; then
        echo "❌ Error: sco installation failed or not found in PATH."
        exit 1
    fi
    echo "✅ sco installed successfully."
else
    echo "✅ sco is already installed."
fi

# 2. Directly generate configuration file (bypass interactive sco init)
SCO_CONFIG_DIR="$HOME/.config/sco/profiles"
SCO_CONFIG_FILE="$SCO_CONFIG_DIR/default.toml"

echo "⚙️  Generating SCO configuration at ${SCO_CONFIG_FILE}..."

# Ensure directory exists
mkdir -p "$SCO_CONFIG_DIR"

# Write TOML configuration file
cat > "$SCO_CONFIG_FILE" <<EOF
access_key_id = '${ACCESS_KEY_ID}'
access_key_secret = '${ACCESS_KEY_SECRET}'
username = '${CURRENT_USER}'
zone = 'cn-sh-01e'
EOF

cat "$SCO_CONFIG_FILE"

echo "✅ Configuration created. Zone set to 'cn-sh-01e'."

echo "Starting sco initialization of components..."

# Install/Initialize necessary components (acp)
sco components install acp

echo "✅ sco components initialized."
echo "----------------------------------------------------------------"

# ==============================================================================
# 1. Argument Parsing
# ==============================================================================
# Check if we have at least 2 arguments (Nodes + at least 1 test file)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <NODES> <TEST_FILE_1> [TEST_FILE_2 ...]"
    echo "Example: $0 2 tests/test_attn.py tests/test_gemm.py"
    exit 1
fi

# 1. Get Nodes count from first argument
NODES_ARG="$1"
shift # Remove first argument, leaving only the file paths in $@

# 2. Verify NODES is a number
if ! [[ "$NODES_ARG" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: NODES must be an integer."
    exit 1
fi

echo "=== Job Configuration ==="
echo "🖥️  Nodes: ${NODES_ARG}"
echo "📄 Test Files: $*"
echo "🌐 Proxy Settings (Local):"
echo "   http_proxy: ${http_proxy}"
echo "   https_proxy: ${https_proxy}"
echo "-------------------------"

# ==============================================================================
# 2. Configuration
# ==============================================================================
WORKSPACE_NAME="workspace-01e"
AEC2_NAME="infra-test"
IMAGE_URL="registry.cn-sh-01.sensecore.cn/sandai-ccr/magi-base:25.10.1"
FILESYSTEM_MOUNT="80433778-429e-11ef-bc97-4eca24dcdba9"
JOB_NAME="test-magi-attention-multi-nodes"

# ==============================================================================
# 3. Build Test Commands Loop
# ==============================================================================
# We construct the execution string for the test files here.
TEST_EXECUTION_BLOCK=""
for test_file in "$@"; do
    TEST_EXECUTION_BLOCK="${TEST_EXECUTION_BLOCK}
echo \"👉 Running Test: ${test_file}\"
python -m torch.distributed.run \"\${DISTRIBUTED_ARGS[@]}\" -m pytest -qs ${test_file}
"
done

# ==============================================================================
# 4. Define Training Command
# ==============================================================================
# Note:
# 1. \${VAR} indicates variables evaluated inside the container at runtime.
# 2. ${VAR} indicates variables evaluated on the host machine (injected into script).

TRAINING_COMMAND="""
export MASTER_ADDR=\${MASTER_ADDR:-localhost}
export MASTER_PORT=\${MASTER_PORT:-6009}
export NNODES=\${WORLD_SIZE}
export GPUS_PER_NODE=8
export NODE_RANK=\${RANK}

echo \"DEBUG: MASTER_ADDR=\${MASTER_ADDR} | MASTER_PORT=\${MASTER_PORT} | GPUS=\${GPUS_PER_NODE} | NODES=\${NNODES} | RANK=\${NODE_RANK}\"

# 1. Setup SSH Keys
mkdir -p /root/.ssh
cp /home/niubility2/protected_folder/key/id_rsa /root/.ssh/id_rsa
source /home/niubility2/protected_folder/key/env.sh;

# 2. Setup Environment Variables
NCCL_DEBUG=INFO
# Inject local proxy settings
export http_proxy=${http_proxy}
export https_proxy=${https_proxy}

# 3. Install/Setup Dependencies
MEGATRON_HOME=/home/niubility2/littsk/butao/backup/Megatron-LM
cd \${MEGATRON_HOME}
pip install --no-build-isolation -e .

# Install Python packages using Tsinghua mirror
pip install seaborn==0.13.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install py3nvml==0.2.7 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==2.3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install Megatron==0.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Debug info
pip show flash_attn
pip show transformer_engine
pip show magi_attention

# 4. Run Tests
echo \"----------------------------------------------------------------\"
echo \"🧪 Starting Distributed Tests...\"
echo \"----------------------------------------------------------------\"

# Define the Distributed Arguments Array (Bash Array)
DISTRIBUTED_ARGS=(
    --nproc_per_node \${GPUS_PER_NODE}
    --nnodes \${NNODES}
    --node_rank \${NODE_RANK}
    --master_addr \${MASTER_ADDR}
    --master_port \${MASTER_PORT}
)

# Execute the dynamic list of tests
${TEST_EXECUTION_BLOCK}

echo \"✅ All tests execution flow finished.\"
"""

# ==============================================================================
# 5. Job Submission
# ==============================================================================
echo "----------------------------------------------------------------"
echo "Submitting Job to Workspace: ${WORKSPACE_NAME}"
echo "----------------------------------------------------------------"

SUBMISSION_OUTPUT=$(sco acp jobs create \
    --workspace-name="${WORKSPACE_NAME}" \
    --aec2-name="${AEC2_NAME}" \
    --job-name="${JOB_NAME}" \
    --priority=HIGHEST \
    --container-image-url="${IMAGE_URL}" \
    --training-framework=pytorch \
    --storage-mount "${FILESYSTEM_MOUNT}:/home/niubility2/" \
    --worker-nodes=${NODES_ARG} \
    --worker-spec='N6lS.Iu.I10.8' \
    --env="NCCL_IB_TIMEOUT:22,NCCL_IB_RETRY_CNT:13,NCCL_IB_AR_THRESHOLD:0" \
    --command="${TRAINING_COMMAND}")

echo "$SUBMISSION_OUTPUT"

# ==============================================================================
# 6. ID Extraction & Log Streaming
# ==============================================================================

JOB_ID=$(echo "$SUBMISSION_OUTPUT" | grep "job id" | awk -F':' '{print $2}' | tr -d '[:space:]')

if [ -z "$JOB_ID" ]; then
    echo "----------------------------------------------------------------"
    echo "ERROR: Failed to extract Job ID. Please check the submission output above."
    echo "----------------------------------------------------------------"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Job submitted successfully. Job ID: ${JOB_ID}"
echo "Waiting 5 seconds for the scheduler..."
echo "----------------------------------------------------------------"
sleep 5

echo "Starting Log Stream..."
echo "Command: sco acp jobs stream-logs --workspace-name=${WORKSPACE_NAME} ${JOB_ID} --follow"
echo "----------------------------------------------------------------"

sco acp jobs stream-logs \
    --workspace-name="${WORKSPACE_NAME}" \
    "${JOB_ID}" \
    --follow
