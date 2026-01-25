#! /bin/bash

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

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} # replace with your own master node IP
export MASTER_PORT=${MASTER_PORT:-16988}
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$((NPROC_PER_NODE * NNODES))

echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT, NNODES=$NNODES, NPROC_PER_NODE=$NPROC_PER_NODE, NODE_RANK=$NODE_RANK"

# to provide custom profile output name by `--profile` argument
export PROFILE_NAME=${PROFILE_NAME:-"cp_benchmark"}
# specify the config file
CONFIG_PATH=${CONFIG_PATH:-"benchmark_conf.py"}
while [[ $# -gt 0 ]]; do
    case "$1" in
    # --profile=xxx
        --profile=*)
            PROFILE_NAME="${1#*=}"
            shift 1
            ;;
        --profile)
            # --profile xxx
            if [[ -n "$2" && "$2" != --* ]]; then
                PROFILE_NAME="$2"
                shift 2
            else
            # --profile
                shift 1
                fi
            ;;
        --config=*)
            CONFIG_PATH="${1#*=}"
            shift 1
            ;;
        --config)
            # --profile xxx
            if [[ -n "$2" && "$2" != --* ]]; then
                CONFIG_PATH="$2"
                shift 2
            else
            # --profile
                shift 1
                fi
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export PYTHONPATH=../../

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"

# generate a timestamp for the nsys output file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_NAME="${PROFILE_NAME}_node${NODE_RANK}_${TIMESTAMP}"
echo "Config: ${CONFIG_PATH}, Profile output(if enabled): ${OUT_NAME}.nsys-rep"
CMD="
    nsys profile \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    -o ${OUT_NAME} \
    ${TORCHRUN_CMD} --config ${CONFIG_PATH}
"

$CMD
