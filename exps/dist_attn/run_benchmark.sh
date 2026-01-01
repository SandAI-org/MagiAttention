#! /bin/bash

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

export NNODES=${NNODES:-1}
export GPUS_PER_NODE=8
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))
export NODE_RANK=${RANK:-0}

# dynamic generate nsys profile using `--profile` argument
PROFILE="0"
PROFILE_NAME="cp_benchmark"
# specify the config file
CONFIG_PATH=${CONFIG_PATH:-"benchmark_conf.py"}
while [[ $# -gt 0 ]]; do
    case "$1" in
    # --profile=xxx
        --profile=*)
            PROFILE=1
            PROFILE_NAME="${1#*=}"
            shift 1
            ;;
        --profile)
            PROFILE=1
            # --profile xxx
            if [[ -n "$2" && "$2" != --* ]]; then
                PROFILE_NAME="$2"
                shift 2
            else
            # --profile
                PROFILE=1
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
export PROFILE=$PROFILE

if [[ $NNODES -eq 1 ]]; then # single-node
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-16988}
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

export PYTHONPATH=../../

export CUDA_DEVICE_MAX_CONNECTIONS=8
echo "set CUDA_DEVICE_MAX_CONNECTIONS=8"
export NCCL_CGA_CLUSTER_SIZE=1
export MAGI_ATTENTION_HIERARCHICAL_COMM=0

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo $DISTRIBUTED_ARGS

TORCHRUN_CMD="torchrun $DISTRIBUTED_ARGS run_benchmark.py"

# generate a timestamp for the nsys output file
# --capture-range=cudaProfilerApi \
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ $PROFILE -eq 1 ]]; then
    OUT_NAME="${PROFILE_NAME}_node${NODE_RANK}_${TIMESTAMP}"
    echo "Config: ${CONFIG_PATH}, Profiling enabled, output: ${OUT_NAME}.nsys-rep"
    export PROFILE_ITER=3
    export PROFILE_WARMUP=0
    CMD="
        nsys profile \
        --force-overwrite true \
        --capture-range=cudaProfilerApi \
        -o ${OUT_NAME}.nsys-rep \
        ${TORCHRUN_CMD} --config ${CONFIG_PATH}
    "
else
    echo "Config: ${CONFIG_PATH}, Profiling disabled."
    CMD="$TORCHRUN_CMD --config $CONFIG_PATH"
fi

$CMD
